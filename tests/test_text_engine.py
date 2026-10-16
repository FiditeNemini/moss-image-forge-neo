# MOSS-Image support for Forge Neo
# The text engine: upstream's chat template with thinking on, no padding, the 4096-token cap and the (T, D) layout.

import importlib
import unittest

import torch

from tests import forge_stubs


class TestTextEngine(unittest.TestCase):
    def setUp(self):
        self.ctx = forge_stubs.installed(forge_stubs.text_processing())
        self.ctx.__enter__()
        self.te = importlib.import_module("forge_moss.native.text_engine")
        model = torch.nn.Module()
        model.num_layers = 32
        self.engine = self.te.MossTextEngine(model, forge_stubs.FakeHFTokenizer())

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_encoder_settings(self):
        kwargs = self.engine.text_encoder.kwargs
        self.assertEqual(kwargs["layer"], "last")
        self.assertEqual(kwargs["special_tokens"], {"pad": 248044})

    def test_template_without_padding(self):
        self.engine(["a cat"])
        text = "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.assertEqual(self.engine.text_encoder.calls[-1], [1000 + ord(c) for c in text])

    def test_whitespace_is_the_empty_prompt(self):
        self.assertEqual(self.engine.token_ids("  "), self.engine.token_ids(""))
        self.assertEqual(self.engine.token_ids(" a cat  "), self.engine.token_ids("a cat"))

    def test_cap_at_4096_tokens(self):
        self.assertEqual(len(self.engine.token_ids("x" * 5000)), 4096)
        lengths = [len(self.te.TEMPLATE.format(text)) for text in ("", "ab")]
        self.assertEqual(self.engine.tokenize(["", "ab"]), lengths)

    def test_layout_and_cache(self):
        cond = self.engine(["x", "x", "yz"])
        self.assertEqual(cond[0].shape, (len(self.engine.token_ids("x")), 3))
        self.assertIs(cond[0], cond[1])
        self.assertEqual(len(self.engine.text_encoder.calls), 2)

    def test_emphasis_is_reported_off(self):
        self.engine(["(a cat)"])
        dynamic_args = importlib.import_module("backend.args").dynamic_args
        self.assertEqual(dynamic_args.last_extra_generation_params["Emphasis"], "None")


if __name__ == "__main__":
    unittest.main()
