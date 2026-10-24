# MOSS-Image support for Forge Neo
# huggingface_guess registration, following the FLUX 3 extension's model.py.

import os

import torch
from huggingface_guess import latent
from huggingface_guess.model_list import BASE, ModelType

from . import detect as detection
from .scheduler import SHIFT

# absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part.
# The folder name must not contain "klein" (Forge Neo's FLUX.2 Klein handling keys on that word), nor "MOSS-Image"
# (Forge's built-in Skip Layer Guidance keys on that one; the DiT does its own negative-pass skipping)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "moss-image")
TEXT_ENCODER = "qwen35_9b"


class MossImage(BASE):
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": detection.IMAGE_MODEL}

    # rectified flow; the DiT multiplies the time by 1000 itself, so the timestep is the sigma
    sampling_settings = {"shift": SHIFT, "multiplier": 1.0}

    # FLUX.2 Klein 4B's, as for FLUX 3: the same width (3072) on the same latent grid (16x, 128 channels)
    memory_usage_factor = 14.6
    # fp16 is allowed for GPUs without bfloat16 (RTX 20 and older): the DiT keeps its residual stream in float32
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    unet_extra_config = {}
    latent_format = latent.Flux2

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        # filled by patches._hook_text_encoder_file from the file selected under VAE / Text Encoder
        return {f"{TEXT_ENCODER}.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLOW


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    return detection.detect(state_dict, key_prefix)
