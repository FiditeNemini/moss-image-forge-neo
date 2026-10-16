# MOSS-Image support for Forge Neo
# Checks that the installed Forge Neo has every piece the MOSS-Image backend builds on, before anything is patched.
# Same mechanism as the FLUX 3 extension's compat.py.

import importlib
import inspect

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.args": ("dynamic_args",),
    "backend.attention": ("attention_function",),
    "backend.memory_management": ("load_model_gpu", "cast_to"),
    "backend.loader": ("load_huggingface_component", "possible_models", "replace_state_dict"),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.nn.llm.llama": ("Qwen3_8B",),
    "backend.nn.vae": ("AutoencoderKLFlux2",),
    "backend.operations": ("ForgeOperations",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "backend.patcher.clip": ("CLIP",),
    "backend.patcher.lora": ("model_lora_keys_unet",),
    "backend.patcher.unet": ("UnetPatcher",),
    "backend.patcher.vae": ("VAE",),
    "backend.sampling.condition": ("ConditionCrossAttn",),
    "backend.text_processing._comfy": ("SDClipModel", "SDTokenizer", "INF"),
    "backend.text_processing.emphasis": ("EmphasisNone", "uses_emphasis"),
    "modules.sd_schedulers": ("Scheduler", "all_schedulers", "schedulers", "schedulers_map"),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "register"),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
    "huggingface_guess.latent": ("Flux2",),
}

# the transformer goes through the Krea 2 branch of the loader, the autoencoder through its own, the text encoder
# through the Qwen3 one (which must still pick its Qwen3-8B class by name, inside the call)
LOADER_BRANCHES = ("Krea2Transformer2DModel", "AutoencoderKLFlux2", "Qwen3Model")
LOADER_SOURCE = ("from backend.nn.llm.llama import Qwen3_8B", "from backend.nn.krea import SingleStreamDiT")


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend import loader, operations

        source = inspect.getsource(loader.load_huggingface_component)
        missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
        if missing:
            problems.append(f"the model loader has no {', '.join(missing)} branch")
        missing = [s for s in LOADER_SOURCE if s not in source]
        if missing:
            problems.append(f"the model loader no longer does {'; '.join(missing)}")
        # the Qwen3.5 norms are Forge's RMSNorm(add=True), the 1 + weight scale its Qwen3-VL uses
        if "add" not in inspect.signature(operations.ForgeOperations.RMSNorm.__init__).parameters:
            problems.append("backend.operations.RMSNorm has no add argument")

    return problems


def report(problems: list[str]) -> None:
    print("[MOSS-Image] this Forge Neo is too old or too new for the extension, so it stays disabled "
          "(Forge Neo itself keeps working as usual):")
    for problem in problems:
        print(f"          {problem}")
