# MOSS-Image support for Forge Neo
# The MOSS-Image diffusion engine: text-to-image, image-to-image and, with MOSS Image Edit's LoRA, editing from
# reference images (edit.py). Follows the FLUX 3 extension's engine: the FLUX.2 autoencoder, the Qwen3.5 text encoder
# through MossTextEngine.

import logging

import torch
from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from . import edit, lora
from .model import TEXT_ENCODER, MossImage
from .text_engine import MAX_LENGTH, MossTextEngine

logger = logging.getLogger("forge_moss")


class MossImageEngine(ForgeDiffusionEngine):
    matched_guesses = [MossImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={TEXT_ENCODER: huggingface_components["text_encoder"]},
                    tokenizer_dict={TEXT_ENCODER: huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.text_engine_moss = MossTextEngine(text_encoder=getattr(clip.cond_stage_model, TEXT_ENCODER),
                                               tokenizer=getattr(clip.tokenizer, TEXT_ENCODER))

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # not FLUX.2: its Klein flag would swap the Automatic schedule for the resolution-dependent Flux2 one
        dynamic_args.klein = False
        # Forge's own MOSS-Image flag (2.31+) turns on its Skip Layer Guidance by checkpoint path; this DiT skips the
        # negative-pass blocks itself, so the extension's model never gets it, wherever the extension is installed
        dynamic_args.moss = False
        self.is_moss_image = True
        # Forge's LoRA extension is imported by now, whatever the extensions' load order was
        lora.ensure_installed()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        if not getattr(prompt, "is_negative_prompt", False):
            # the positive conditioning sets the references, as for Forge's edit models: the img2img input first,
            # then ImageStitch's images; the DiT reads them for both CFG passes (edit.references)
            references = [*self.ref_latents]
            if self.ini_latent is not None:
                references.insert(0, self.ini_latent)
                self.ini_latent = None
            if not edit.enabled():
                references = []
            if len(references) > edit.MAX_REFERENCES:
                logger.warning(f"[MOSS-Image] {len(references)} reference images, the edit uses the first "
                               f"{edit.MAX_REFERENCES}")
                references = references[:edit.MAX_REFERENCES]
            dynamic_args.ref_latents = references
            if references:
                dynamic_args.last_extra_generation_params["MOSS references"] = len(references)
        return self.text_engine_moss(prompt)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        samples = super().encode_first_stage(x)
        if edit.enabled():
            # a reference is encoded at its own, smaller size (edit.SIZE_OPTION); the target keeps the job's
            image = edit.fit(x[:1], edit.max_side())
            latent = samples[:1] if image is None else super().encode_first_stage(image)
            latent = latent.detach().clone().cpu()
            if dynamic_args.is_referencing:  # ImageStitch
                self.ref_latents.append(latent)
            else:  # the img2img input
                self.ini_latent = latent
        return samples

    @torch.inference_mode()
    def decode_first_stage(self, x):
        # sampling is over: the references' keys and values leave the GPU
        diffusion_model = getattr(self.forge_objects.unet.model, "diffusion_model", None)
        if hasattr(diffusion_model, "clear_reference_cache"):
            diffusion_model.clear_reference_cache()
        return super().decode_first_stage(x)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        return self.text_engine_moss.tokenize(prompt), MAX_LENGTH
