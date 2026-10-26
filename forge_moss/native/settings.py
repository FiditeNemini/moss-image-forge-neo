# MOSS-Image support for Forge Neo
# The negative-prompt block skipping of upstream's guidance, as a Forge setting: the blocks (zero-based) the DiT
# leaves out in the CFG negative pass. Upstream's default is block 22; an empty value is plain CFG.

OPTION = "moss_uncond_skip_layers"
DEFAULT = "22"
LABEL = "[MOSS-Image] Blocks skipped in the negative-prompt pass"
INFO = "zero-based, comma-separated; 22 = upstream's guidance (24 and 26 also help); empty = plain CFG"
# A settings page of its own under the "Stable Diffusion" category, Settings > Stable Diffusion > MOSS-Image, as the
# Qwen-Image 2.1 extension does (options_section splits ("id", "Label", "category") the same way)
SECTION = ("moss_image", "MOSS-Image")
CATEGORY = "sd"


# Forge Neo 2.31+ has MOSS-Image support of its own: who runs the MOSS-Image checkpoints. With "Forge built-in" the
# extension's detection steps aside and Forge's own guess, engine and Skip Layer Guidance take the checkpoint
BACKEND_OPTION = "moss_backend"
EXTENSION = "Extension"
BUILT_IN = "Forge built-in"
BACKEND_LABEL = "[MOSS-Image] Backend"
BACKEND_INFO = ("who runs MOSS-Image checkpoints: this extension, or Forge Neo's built-in support (its own "
                "[Moss-Image] Skip Layer Guidance options apply then); takes effect at the next generation")


def forge_has_built_in() -> bool:
    try:
        from huggingface_guess import model_list

        return hasattr(model_list, "MOSSImage")
    except Exception:  # outside Forge Neo (CPU tests)
        return False


def use_extension() -> bool:
    # an older Forge Neo has nothing to fall back to: the extension runs the checkpoints whatever the option says
    if not forge_has_built_in():
        return True
    try:
        from modules.shared import opts

        return getattr(opts, BACKEND_OPTION, EXTENSION) != BUILT_IN
    except Exception:  # before the options exist
        return True


def _reload_moss_model() -> None:
    # Forge reloads a checkpoint only when its loading parameters change; a MOSS-Image model already loaded is
    # dropped so that the next generation loads it again with the backend just chosen
    from modules import sd_models

    current = sd_models.model_data.sd_model
    if getattr(current, "is_moss_image", False) or type(current).__name__ == "MOSSImage":
        sd_models.model_data.forge_hash = ""


def parse(text: str, depth: int) -> frozenset[int]:
    """The valid block indices in text; anything else is ignored."""
    layers = set()
    for part in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        if part.strip().lstrip("-").isdigit() and 0 <= int(part) < depth:
            layers.add(int(part))
    return frozenset(layers)


def uncond_skip_layers(depth: int) -> frozenset[int]:
    try:
        from modules.shared import opts

        text = getattr(opts, OPTION, DEFAULT)
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        text = DEFAULT
    return parse(text, depth)


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if forge_has_built_in() and BACKEND_OPTION not in shared.opts.data_labels:
        import gradio as gr

        shared.opts.add_option(BACKEND_OPTION, OptionInfo(EXTENSION, BACKEND_LABEL, gr.Radio,
                                                          {"choices": (EXTENSION, BUILT_IN)},
                                                          onchange=_reload_moss_model, section=SECTION,
                                                          category_id=CATEGORY).info(BACKEND_INFO))
    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(DEFAULT, LABEL, section=SECTION, category_id=CATEGORY,
                                                  infotext="MOSS skip").info(INFO))
