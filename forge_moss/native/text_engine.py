# MOSS-Image support for Forge Neo
# The MOSS-Image text encoder: Qwen3.5-9B's last hidden state (after the final norm) for the prompt in Qwen's chat
# template with thinking on, as upstream's encode_prompt renders it through apply_chat_template(add_generation_prompt,
# enable_thinking) and ComfyUI's MOSS-Image node spells it out. 4096 features per token.
#
# Each prompt is encoded on its own and unpadded, at most 4096 tokens (upstream's truncation); the DiT reads every
# token. An empty prompt is upstream's default negative prompt.

import torch
from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import INF, SDClipModel, SDTokenizer

# <|endoftext|>: never part of the template, so Forge's end-of-text masking never cuts the prompt
PAD = 248044
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
MAX_LENGTH = 4096
TEXT_DIM = 4096


class MossTextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="last", layer_idx=None, special_tokens={"pad": PAD},
                                        layer_norm_hidden_state=False)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def token_ids(self, text: str) -> list[int]:
        # surrounding whitespace is dropped (as Forge's Krea 2 engine does), so the spaces a removed LoRA tag leaves
        # behind do not change the prompt, and "", " " or only LoRA tags are upstream's empty negative prompt
        return self.tokenizer.tokenizer.encode(TEMPLATE.format(text.strip()), add_special_tokens=False)[:MAX_LENGTH]

    def tokenize(self, texts: str | list[str]) -> int | list[int]:
        if isinstance(texts, str):
            return len(self.token_ids(texts))
        return [len(self.token_ids(t)) for t in texts]

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                cache[line] = self.encode(line)
            zs.append(cache[line])
        return zs

    def encode(self, line: str) -> torch.Tensor:
        states = self.text_encoder.encode_token_weights([[(token, 1.0) for token in self.token_ids(line)]])[0]
        # (1, T, 4096) -> (T, 4096): no batch dimension, as Forge's native engines return their conditioning
        return states[0]
