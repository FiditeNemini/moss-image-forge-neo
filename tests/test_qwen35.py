# MOSS-Image support for Forge Neo
# The Qwen3.5 text encoder port against transformers' text decoder (tests/reference/qwen35_hf.py, token-by-token
# delta rule), its configuration against the released ones, and its parameter names and shapes against the real
# Comfy-Org text encoder file (header in tests/fixtures).

import json
import unittest
from pathlib import Path

import torch

from forge_moss.native import checkpoints, qwen35
from tests import forge_stubs
from tests.reference import qwen35_hf

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
EMBEDDED_CONFIG = ROOT / "forge_moss" / "native" / "huggingface" / "moss-image" / "text_encoder" / "config.json"
# OpenMOSS-Team/MOSS-Image-SFT text_encoder/config.json, the parts that shape the language tower
RELEASED_CONFIG = {
    "architectures": ["Qwen3_5Model"], "model_type": "qwen3_5",
    "text_config": {
        "attn_output_gate": True, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
        "hidden_size": 4096, "intermediate_size": 12288, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4, "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06, "vocab_size": 248320,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000, "rope_type": "default"},
    },
}
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")

# small enough for the CPU, large enough for every path: grouped heads in both attentions, a full-attention layer,
# partial rotary, several delta-rule chunks
TINY = {
    "vocab_size": 300, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 8,
    "full_attention_interval": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "partial_rotary_factor": 0.25, "rope_theta": 10000.0, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
    "linear_key_head_dim": 16, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6,
}


def reference_config(config):
    layer_types = ["full_attention" if (i + 1) % config["full_attention_interval"] == 0 else "linear_attention"
                   for i in range(config["num_hidden_layers"])]
    return {**config, "layer_types": layer_types, "mrope_section": [2, 1, 1]}


def randomized(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("A_log"):
                param.copy_(torch.empty(param.shape).uniform_(0.01, 16, generator=generator).log())
            elif "norm" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.2)
            else:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.15)
    return model


def built(config):
    with forge_stubs.forge_operations():
        return qwen35.Qwen35TextModel(config).eval()


class TestQwen35(unittest.TestCase):
    def test_matches_transformers(self):
        reference = randomized(qwen35_hf.Qwen3_5TextModel(reference_config(TINY)), seed=0).eval()
        model = built(TINY)
        missing, unexpected = model.load_state_dict(
            {f"model.{k}": v for k, v in reference.state_dict().items()}, strict=False)
        self.assertEqual((missing, unexpected), ([], []))

        ids = torch.randint(0, TINY["vocab_size"], (1, 150), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            expected = reference(ids)
            got, intermediate = model(None, embeds=model.get_input_embeddings()(ids).float(), dtype=torch.float32)
        self.assertIsNone(intermediate)
        self.assertEqual(got.shape, (1, 150, TINY["hidden_size"]))
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)

    def test_prefix_is_causal(self):
        # a token's state depends on the tokens before it only: the first 70 of 150 encode as 70 alone
        model = randomized(built(TINY), seed=2)
        ids = torch.randint(0, TINY["vocab_size"], (1, 150), generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            whole = model(ids)[0]
            prefix = model(ids[:, :70])[0]
        torch.testing.assert_close(whole[:, :70], prefix, rtol=1e-4, atol=1e-4)

    def test_chunked_delta_rule_matches_recurrent(self):
        generator = torch.Generator().manual_seed(4)
        for length in (1, 63, 64, 65, 200):
            q, k = (torch.randn(2, length, 3, 16, generator=generator) for _ in range(2))
            v = torch.randn(2, length, 3, 8, generator=generator)
            g = -torch.rand(2, length, 3, generator=generator) * 2
            beta = torch.rand(2, length, 3, generator=generator)
            expected = qwen35_hf.torch_recurrent_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
            got = qwen35.chunk_gated_delta_rule(q, k, v, g, beta)
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5, msg=f"length {length}")

    def test_configuration(self):
        released = qwen35.Qwen35Config.from_dict(RELEASED_CONFIG)
        embedded = qwen35.Qwen35Config.from_dict(json.loads(EMBEDDED_CONFIG.read_text()))
        self.assertEqual(released, qwen35.Qwen35Config())
        self.assertEqual(embedded, qwen35.Qwen35Config())
        # Forge's Qwen3 loader branch picks its 8B class (the one swapped for the port) on this hidden size
        self.assertEqual(json.loads(EMBEDDED_CONFIG.read_text())["hidden_size"], 4096)
        layer_types = [released.is_full_attention(i) for i in range(released.num_hidden_layers)]
        self.assertEqual(layer_types, [i % 4 == 3 for i in range(32)])


class TestRealCheckpoint(unittest.TestCase):
    def setUp(self):
        self.header = json.loads((FIXTURES / "qwen3.5_9b_int8_convrot.shapes.json").read_text())

    def test_weights_match_the_model(self):
        prepared = checkpoints.prepare_text_encoder(dict(self.header))
        with torch.device("meta"):
            model = built({})
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        weights = {k: tuple(v["shape"]) for k, v in prepared.items() if not k.endswith(QUANT_SUFFIXES)}
        self.assertEqual(weights, expected)
        # every INT8 layer is a Linear of the port, or the token embedding (Forge's mixed-precision Embedding)
        quantized = {k.rsplit(".", 1)[0] for k in prepared if k.endswith(".comfy_quant")}
        layers = {n for n, m in model.named_modules() if isinstance(m, (torch.nn.Linear, torch.nn.Embedding))}
        self.assertIn("model.embed_tokens", quantized)
        self.assertTrue(quantized <= layers)

    def test_other_parts_are_left_out(self):
        state_dict = dict(self.header)
        prepared = checkpoints.prepare_text_encoder(state_dict)
        self.assertTrue(all(k.startswith("model.") and ".visual." not in k for k in prepared))
        self.assertFalse(any(k.startswith("model.language_model.") for k in state_dict))
        self.assertTrue(any(k.startswith("model.visual.") for k in state_dict))
        self.assertTrue(any(k.startswith(("mtp.", "lm_head.")) for k in state_dict))

    def test_other_layouts(self):
        tower = {k[len("model.language_model."):]: v for k, v in self.header.items()
                 if k.startswith("model.language_model.")}
        for prefix in ("language_model.", "model."):
            prepared = checkpoints.prepare_text_encoder({f"{prefix}{k}": v for k, v in tower.items()})
            self.assertEqual(set(prepared), {f"model.{k}" for k in tower})

    def test_wrong_files_say_so(self):
        with self.assertRaisesRegex(ValueError, "needs its Qwen3.5 9B"):
            checkpoints.prepare_text_encoder({"x": {"shape": [1]}})
        qwen3vl = json.loads((FIXTURES / "qwen3.5_9b_int8_convrot.shapes.json").read_text())
        qwen3vl = {k: v for k, v in qwen3vl.items() if "linear_attn" not in k}
        with self.assertRaisesRegex(ValueError, "needs its Qwen3.5 9B"):
            checkpoints.prepare_text_encoder(qwen3vl)
        smaller = dict(self.header)
        smaller["model.language_model.norm.weight"] = {"dtype": "BF16", "shape": [2560]}
        with self.assertRaisesRegex(ValueError, "another size"):
            checkpoints.prepare_text_encoder(smaller)


if __name__ == "__main__":
    unittest.main()
