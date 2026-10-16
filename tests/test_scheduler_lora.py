# MOSS-Image support for Forge Neo
# The MOSS scheduler against upstream's MOSSImageScheduler grid, and the LoRA key map for fused and unfused builds.

import unittest

import torch

from forge_moss.native import lora, scheduler
from tests.reference import moss_image_dit


class TestScheduler(unittest.TestCase):
    def test_matches_upstream(self):
        for steps in (1, 8, 20, 28, 50, 100, 250):
            expected = torch.cat((moss_image_dit.scheduler_sigmas(steps), torch.zeros(1)))
            torch.testing.assert_close(scheduler.sigmas(steps), expected, rtol=0, atol=0, msg=f"{steps} steps")

    def test_fifty_steps(self):
        sigmas = scheduler.sigmas(50)
        self.assertEqual(len(sigmas), 51)
        self.assertEqual(float(sigmas[0]), 1.0)
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))


class TestLoraKeys(unittest.TestCase):
    def test_fused_qkv_thirds(self):
        state = {"diffusion_model.layers.3.proj.qkv.weight": torch.zeros(9, 2),
                 "diffusion_model.text_embedder.blocks.0.qkv.weight": torch.zeros(9, 2),
                 "diffusion_model.layers.3.proj.o.weight": torch.zeros(3, 3)}
        keys = lora.moss_lora_keys(state)
        fused = "diffusion_model.layers.3.proj.qkv.weight"
        self.assertEqual(keys["transformer.layers.3.proj.q"], (fused, (0, 0, 3)))
        self.assertEqual(keys["lora_unet_layers_3_proj_k"], (fused, (0, 3, 3)))
        self.assertEqual(keys["layers.3.proj.v"], (fused, (0, 6, 3)))
        self.assertEqual(keys["transformer.layers.3.proj.qkv"], fused)
        self.assertEqual(keys["transformer.layers.3.proj.o"], "diffusion_model.layers.3.proj.o.weight")
        # the text refiner's qkv is upstream's own fused projection: no thirds
        self.assertNotIn("transformer.text_embedder.blocks.0.q", keys)
        self.assertEqual(keys["transformer.text_embedder.blocks.0.qkv"],
                         "diffusion_model.text_embedder.blocks.0.qkv.weight")


if __name__ == "__main__":
    unittest.main()
