# MOSS-Image support for Forge Neo
# The MOSS-Image diffusion engine: text-to-image and image-to-image (upstream has no reference-image editing).
# Follows the FLUX 3 extension's engine: the FLUX.2 autoencoder, the Qwen3.5 text encoder through MossTextEngine.

import torch
from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from . import lora
from .model import TEXT_ENCODER, MossImage
from .text_engine import MAX_LENGTH, MossTextEngine


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
        return self.text_engine_moss(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        return self.text_engine_moss.tokenize(prompt), MAX_LENGTH
