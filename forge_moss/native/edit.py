# MOSS-Image support for Forge Neo
# MOSS Image Edit (chinoll/Moss-Image-Edit): an editing LoRA for MOSS-Image that conditions the DiT on reference images.
# The references are VAE latents packed like the target; the DiT runs them as a prefix branch at t = 0 whose keys and
# values every target and text token also attends to (transformer.MossImageDiT.reference_kv).
#
# Forge's own convention for edit models (Klein, Krea 2 Edit...): with the "Enable Reference" setting on, the img2img
# input and the images of ImageStitch become the references (engine.MossImageEngine); off, txt2img and img2img stay
# as they are. The LoRA goes through Forge's LoRA path, its alpha read from the adapter_config embedded in the file.
#
# Forge imports stay inside the functions: the DiT and the CPU tests import this module outside Forge.

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger("forge_moss")

OPTION = "moss_edit_reference"
LABEL = "[MOSS-Image] Enable Reference"
INFO = "enable Edit ; disable img2img"
INFO_LORA = "requires the MOSS Image Edit LoRA (chinoll/Moss-Image-Edit)"
SIZE_OPTION = "moss_edit_reference_size"
SIZE_DEFAULT = 512  # the reference_max_edge default of the LoRA authors' ComfyUI node
SIZE_LABEL = "[MOSS-Image] Reference Maximum Side Length"
SIZE_INFO = "every reference is scaled down to this long side (multiples of 16) before encoding; 0 = no limit"
MAX_REFERENCES = 4  # what the LoRA was trained with
FORMAT = "moss-edit-lora"

_generation = 0


def generation() -> int:
    """Bumped at the first step of every sampling run: the DiT's reference cache lives for one run."""
    return _generation


def new_generation() -> None:
    global _generation
    _generation += 1


def on_cfg_denoiser(params) -> None:
    # script_callbacks.on_cfg_denoiser; the denoiser counts its calls from 0 in each sampling run (hires pass included)
    if getattr(getattr(params, "denoiser", None), "step", None) == 0:
        new_generation()


def _opts():
    try:
        from modules.shared import opts

        return opts
    except Exception:  # outside Forge Neo (CPU tests)
        return None


def enabled() -> bool:
    """The setting is on and the loaded model is this extension's MOSS-Image."""
    opts = _opts()
    if opts is None or not getattr(opts, OPTION, False):
        return False
    from modules import shared

    return bool(getattr(shared.sd_model, "is_moss_image", False))


def references() -> list:
    """The reference latents Forge holds for this generation (dynamic_args.ref_latents), or none."""
    if not enabled():
        return []
    from backend.args import dynamic_args

    return list(dynamic_args.ref_latents or [])


def max_side() -> int:
    opts = _opts()
    return int(getattr(opts, SIZE_OPTION, SIZE_DEFAULT) or 0) if opts is not None else SIZE_DEFAULT


def fit(image: torch.Tensor, limit: int) -> torch.Tensor | None:
    """image (B, 3, H, W) in [-1, 1] scaled so its long side is at most limit, both sides on multiples of 16; None
    when it already is."""
    height, width = image.shape[-2:]
    scale = min(1.0, limit / max(height, width)) if limit > 0 else 1.0
    size = tuple(max(16, round(side * scale / 16) * 16) for side in (height, width))
    if size == (height, width):
        return None
    out = torch.nn.functional.interpolate(image.float(), size=size, mode="bicubic", antialias=True)
    return out.clamp(-1, 1).to(image.dtype)


def adapter_alphas(filename) -> dict:
    """module name -> LoRA alpha from a moss-edit-lora adapter_config: the one embedded in the safetensors metadata,
    else an adapter_config.json beside the file; empty for any other LoRA."""
    config = None
    path = Path(str(filename))
    if path.suffix == ".safetensors" and path.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as handle:
                config = json.loads((handle.metadata() or {}).get("adapter_config", "null"))
        except Exception as e:
            logger.warning(f"[MOSS-Image] could not read the metadata of {path.name}: {e}")
    if config is None and (path.parent / "adapter_config.json").is_file():
        try:
            config = json.loads((path.parent / "adapter_config.json").read_text(encoding="utf-8"))
        except Exception:
            config = None
    if not isinstance(config, dict) or config.get("format") != FORMAT:
        return {}
    return {name: float(spec["alpha"]) for name, spec in (config.get("modules") or {}).items()
            if isinstance(spec, dict) and "alpha" in spec}


def with_alphas(lora: dict, alphas: dict) -> dict:
    """The LoRA state dict with an alpha tensor for every adapter module that has none (Forge scales by alpha / rank,
    and by 1 without alpha). The modules' lora_A / lora_B keys are a format Forge already reads."""
    missing = {f"{name}.alpha": torch.tensor(alpha) for name, alpha in alphas.items()
               if f"{name}.lora_A" in lora and f"{name}.alpha" not in lora}
    return {**lora, **missing} if missing else lora


def register_settings() -> None:
    import gradio as gr
    from modules import shared
    from modules.options import OptionInfo

    from . import settings

    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(False, LABEL, section=settings.SECTION, category_id=settings.CATEGORY)
                               .info(INFO).info(INFO_LORA)
                               .info("pin to <b>Quicksettings</b> is recommended if changed often"))
        shared.opts.onchange(OPTION, clear_references, call=False)
    if SIZE_OPTION not in shared.opts.data_labels:
        shared.opts.add_option(SIZE_OPTION, OptionInfo(SIZE_DEFAULT, SIZE_LABEL, gr.Slider,
                                                       {"minimum": 0, "maximum": 2048, "step": 16},
                                                       section=settings.SECTION, category_id=settings.CATEGORY)
                               .info(SIZE_INFO))


def clear_references() -> None:
    # like Forge's own clear_references for [Klein] Enable Reference, and the prompt cache too: the cached positive
    # conditioning is what sets the references, so the next generation encodes again
    from backend.args import dynamic_args
    from modules.processing import StableDiffusionProcessing

    dynamic_args.ref_latents.clear()
    StableDiffusionProcessing.cached_c = [None, None, None]
    StableDiffusionProcessing.cached_uc = [None, None, None]
