# MOSS-Image support for Forge Neo
# The text encoder goes through Forge's Qwen3 loader branch with the port in place of its Qwen3-8B, for that call
# only, and with the language tower's keys only.

import importlib
import json
import unittest
from pathlib import Path

from tests import forge_stubs

FIXTURES = Path(__file__).parent / "fixtures"


class TestTextEncoderLoad(unittest.TestCase):
    def setUp(self):
        self.qwen3_8b = object()
        llama = forge_stubs.module("backend.nn.llm.llama", Qwen3_8B=self.qwen3_8b)
        llm = forge_stubs.module("backend.nn.llm", llama=llama)
        nn = forge_stubs.module("backend.nn", llm=llm)
        self.llama = llama
        self.ctx = forge_stubs.installed({"backend": forge_stubs.module("backend", nn=nn), "backend.nn": nn,
                                          "backend.nn.llm": llm, "backend.nn.llm.llama": llama})
        self.ctx.__enter__()
        self.text_encoder = importlib.import_module("forge_moss.native.text_encoder")

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_routes_through_the_qwen3_branch(self):
        seen = {}

        def loader(guess, component_name, lib_name, cls_name, repo_path, state_dict):
            seen.update(cls_name=cls_name, cls=self.llama.Qwen3_8B, keys=set(state_dict))
            return "model"

        header = json.loads((FIXTURES / "qwen3.5_9b_int8_convrot.shapes.json").read_text())
        result = self.text_encoder.load(loader, None, "text_encoder", "transformers", "repo", header)
        self.assertEqual(result, "model")
        self.assertEqual(seen["cls_name"], "Qwen3Model")
        self.assertIs(seen["cls"], importlib.import_module("forge_moss.native.qwen35").Qwen35TextModel)
        self.assertTrue(all(k.startswith("model.") and not k.startswith("model.visual.") for k in seen["keys"]))
        self.assertIs(self.llama.Qwen3_8B, self.qwen3_8b)

    def test_restores_on_failure(self):
        def loader(*args):
            raise RuntimeError("load failed")

        header = json.loads((FIXTURES / "qwen3.5_9b_int8_convrot.shapes.json").read_text())
        with self.assertRaises(RuntimeError):
            self.text_encoder.load(loader, None, "text_encoder", "transformers", "repo", header)
        self.assertIs(self.llama.Qwen3_8B, self.qwen3_8b)


if __name__ == "__main__":
    unittest.main()
