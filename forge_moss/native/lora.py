# MOSS-Image support for Forge Neo
# MOSS-Image LoRAs on Forge's own LoRA path. Forge's generic key map already matches "diffusion_model.<layer>" and
# "lora_unet_<layer>"; this adds the "transformer." prefix (diffusers / PEFT) and bare layer names, and, for
# Comfy-Org's builds, which fuse each block's q, k and v into one proj.qkv, sends proj.q / proj.k / proj.v LoRAs (the
# names of upstream's model, which LoRAs are trained on) to their third of it.
#
# The hook wraps model_lora_keys_unet where sd_forge_lora (Forge's built-in LoRA extension) and backend.patcher.lora
# hold it, and only adds keys for a MOSS-Image DiT; everything else gets Forge's map untouched. Same mechanism as the
# FLUX 3 extension's lora.py.

import re
import sys

PROJECTIONS = ("q", "k", "v")
FUSED = re.compile(r"(layers\.\d+\.proj)\.qkv$")
_hooked = False


def moss_lora_keys(state_dict) -> dict:
    """LoRA key -> model key, or (model key, (dim, offset, size)) for a third of a fused qkv weight."""
    key_map = {}
    for key in state_dict:
        if not (key.startswith("diffusion_model.") and key.endswith(".weight")):
            continue
        layer = key[len("diffusion_model."):-len(".weight")]
        targets = [(layer, key)]
        if fused := FUSED.match(layer):
            third = state_dict[key].shape[0] // 3
            targets += [(f"{fused.group(1)}.{p}", (key, (0, i * third, third))) for i, p in enumerate(PROJECTIONS)]
        for name, to in targets:
            for alias in (name, f"diffusion_model.{name}", f"transformer.{name}", f"lora_unet_{name.replace('.', '_')}",
                          f"lycoris_{name.replace('.', '_')}"):
                key_map[alias] = to
    return key_map


def _is_moss_image(model) -> bool:
    from .transformer import MossImageDiT

    return isinstance(getattr(model, "diffusion_model", None), MossImageDiT)


def _wrap(original):
    if getattr(original, "_moss_image", False):
        return original

    def model_lora_keys_unet(model, key_map=None):
        key_map = original(model, {} if key_map is None else key_map)
        if _is_moss_image(model):
            key_map.update(moss_lora_keys(model.state_dict()))
        return key_map

    model_lora_keys_unet._moss_image = True
    return model_lora_keys_unet


def install() -> bool:
    """Wrap the key map in backend.patcher.lora and, once it is imported, in sd_forge_lora's networks."""
    global _hooked
    from backend.patcher import lora as forge_lora

    forge_lora.model_lora_keys_unet = _wrap(forge_lora.model_lora_keys_unet)
    networks = sys.modules.get("networks")
    if networks is None or not hasattr(networks, "model_lora_keys_unet"):
        return False
    networks.model_lora_keys_unet = _wrap(networks.model_lora_keys_unet)
    _hooked = True
    return True


def ensure_installed() -> None:
    if not _hooked:
        install()
