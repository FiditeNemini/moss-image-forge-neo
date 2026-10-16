# MOSS-Image support for Forge Neo
# Builds the Qwen3.5 text encoder through Forge Neo's own Qwen3 loader branch, which does the dtype choice,
# quantized weights (INT8 ConvRot, FP8, GGUF), weightless construction and state-dict loading. The embedded
# configuration names the encoder "Qwen3Model" with hidden size 4096, so the branch imports
# backend.nn.llm.llama.Qwen3_8B inside the call; it is swapped for Qwen35TextModel for that call only (the FLUX 3
# extension routes its DiT through the Krea 2 branch the same way).

from .checkpoints import prepare_text_encoder
from .qwen35 import Qwen35TextModel

LOADER_CLASS = "Qwen3Model"


def load(original_loader, guess, component_name, lib_name, repo_path, state_dict):
    from backend.nn.llm import llama

    state_dict = prepare_text_encoder(state_dict)
    original_class = llama.Qwen3_8B
    llama.Qwen3_8B = Qwen35TextModel
    try:
        return original_loader(guess, component_name, lib_name, LOADER_CLASS, repo_path, state_dict)
    finally:
        llama.Qwen3_8B = original_class
