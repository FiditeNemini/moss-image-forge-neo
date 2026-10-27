# MOSS-Image support for Forge Neo
# INT4 ConvRot checkpoints (quantization format "convrot_w4a4", the ~3 GB builds): their metadata asks comfy-kitchen
# to quantize the activations to INT4 as well (W4A4, linear_dtype "int4"). comfy-kitchen's other path for the same
# format keeps the INT4 weights and quantizes the activations to INT8 (W4A8, linear_dtype "int8"), which removes the
# activation part of the error; ComfyUI documents both (QUANTIZATION.md). A setting picks the path.
#
# The choice is written into the loaded DiT's weights at the first step of every sampling run, so a change applies
# at the next generation without reloading the checkpoint, for the extension's DiT and Forge's built-in one alike.
# Forge imports stay inside the functions: the CPU tests import this module outside Forge.

import dataclasses
import logging

logger = logging.getLogger("forge_moss")

OPTION = "moss_int4_activations"
INT8 = "INT8 (W4A8)"
STORED = "As stored (W4A4)"
DEFAULT = INT8
LABEL = "[MOSS-Image] Activations of INT4 checkpoints"
INFO = ("INT8 = better image, same INT4 file; As stored = the checkpoint's own INT4 activations; "
        "only INT4 ConvRot checkpoints, takes effect at the next generation")

LAYOUT = "TensorCoreConvRotW4A4Layout"
# the checkpoint's own value, kept on the module the first time it is changed
STORED_ATTRIBUTE = "_moss_stored_linear_dtype"


def _opts():
    try:
        from modules.shared import opts

        return opts
    except Exception:  # outside Forge Neo (CPU tests)
        return None


def choice() -> str:
    return getattr(_opts(), OPTION, DEFAULT)


def wanted(stored: str) -> str:
    return "int8" if choice() == INT8 else stored


def apply(model) -> int:
    """Set every ConvRot W4A4 weight of model to the chosen activation path; returns how many changed."""
    changed = 0
    for module in model.modules():
        weight = module._parameters.get("weight")
        if getattr(weight, "_layout_cls", None) != LAYOUT:
            continue
        params = weight._params
        stored = module.__dict__.setdefault(STORED_ATTRIBUTE, params.linear_dtype)
        linear_dtype = wanted(stored)
        if params.linear_dtype != linear_dtype:
            weight._params = dataclasses.replace(params, linear_dtype=linear_dtype)
            changed += 1
    return changed


def _diffusion_model():
    from modules import shared

    sd_model = shared.sd_model
    if not (getattr(sd_model, "is_moss_image", False) or type(sd_model).__name__ == "MOSSImage"):
        return None
    return getattr(getattr(sd_model.forge_objects.unet, "model", None), "diffusion_model", None)


def on_cfg_denoiser(params) -> None:
    # script_callbacks.on_cfg_denoiser; the denoiser counts its calls from 0 in each sampling run (hires pass included)
    if getattr(getattr(params, "denoiser", None), "step", None) != 0:
        return
    try:
        model = _diffusion_model()
        if model is not None and apply(model):
            print(f"[MOSS-Image] INT4 checkpoint activations: {choice()}")
    except Exception as e:
        # the generation goes on with the checkpoint's own path
        logger.warning(f"[MOSS-Image] could not set the INT4 activations: {e}")


def register_settings() -> None:
    import gradio as gr
    from modules import shared
    from modules.options import OptionInfo

    from . import settings

    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(DEFAULT, LABEL, gr.Radio, {"choices": (INT8, STORED)},
                                                  section=settings.SECTION, category_id=settings.CATEGORY)
                               .info(INFO))
