# MOSS-Image support for Forge Neo
# Recognize a MOSS-Image DiT and read its configuration from tensor shapes.
#
# Works on a loaded state dict or on a safetensors header: shapes are read through shape_of, so no tensor has to be
# materialized. Same identity keys as the official ComfyUI node's model_config (comfyui/moss-image/loaders.py).

import re

from .checkpoints import shape_of

# not Forge Neo's own "moss_image": since 2.31 Forge has a built-in MOSS-Image guess that matches that value, sits
# earlier in model_list.models and would take this config (freq_dim and all) to its own DiT class (issue #4)
IMAGE_MODEL = "moss_image_extension"

IDENTITY_KEYS = ("text_embedder.proj.weight", "shared_block_modulation.weight", "video_head_modulation.weight",
                 "final_linear_video.weight", "layers.0.proj.ffn_up_gate.weight")


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_moss_image(state_dict, key_prefix=""):
    return all(f"{key_prefix}{k}" in state_dict for k in IDENTITY_KEYS)


def detect(state_dict, key_prefix=""):
    """The MossImageDiT keyword arguments for this checkpoint, or None when it is not a MOSS-Image DiT."""
    if not is_moss_image(state_dict, key_prefix):
        return None

    def shape(key):
        return shape_of(state_dict[f"{key_prefix}{key}"])

    # widths come from weights kept in bf16 in every build (embedders, norms, modulations)
    dim, latent_channels = shape("video_embedder.weight")
    return {
        "image_model": IMAGE_MODEL,
        "dim": dim,
        "num_heads": dim // shape("layers.0.q_norm.weight")[0],
        "num_layers": count_blocks(list(state_dict.keys()), f"{key_prefix}layers."),
        # ffn_up_gate carries the MLP's value and gate halves
        "ffn_dim": shape("layers.0.proj.ffn_up_gate.weight")[0] // 2,
        "text_dim": shape("text_embedder.proj.weight")[1],
        "freq_dim": shape("time_embedding.0.weight")[1],
        "latent_channels": latent_channels,
        # Comfy-Org's builds fuse q, k, v into one projection; upstream's diffusers export keeps them apart
        "fused_qkv": f"{key_prefix}layers.0.proj.qkv.weight" in state_dict,
    }
