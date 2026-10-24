# MOSS-Image support for Forge Neo
# The fused comfy-kitchen paths of kernels.py against the eager code they replace: the norm weight folded into the
# AdaLN scale, and the sin | cos RoPE table turned into comfy-kitchen's rotation matrices. Runs on CPU through
# comfy-kitchen's eager backend; skipped where comfy-kitchen is not installed.

import unittest
from unittest import mock

import torch

from forge_moss.native import kernels, transformer

try:
    import comfy_kitchen
except ImportError:
    comfy_kitchen = None


@unittest.skipIf(comfy_kitchen is None, "comfy-kitchen is not installed")
class FusedKernels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    @staticmethod
    def fused():
        return mock.patch.object(kernels, "_fused", lambda x: True)

    @staticmethod
    def norm(dim):
        norm = transformer.RMSNorm(dim)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(dim) * 0.3 + 1.0)
        return norm

    def test_modulated_rms_norm(self):
        norm = self.norm(64)
        x = torch.randn(2, 10, 64)
        scale, shift = torch.randn(2, 1, 64) * 0.5, torch.randn(2, 1, 64) * 0.5
        expected = kernels.modulated_rms_norm(norm, x, scale, shift, transformer.EPS)
        with self.fused():
            actual = kernels.modulated_rms_norm(norm, x, scale, shift, transformer.EPS)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_norm_rope(self):
        heads, head_dim, height, width, text = 4, 32, 3, 4, 5
        rope = transformer.rope_table(height, width, text, head_dim, 10000.0, "cpu")
        q = torch.randn(2, height * width + text, heads, head_dim)
        k = torch.randn_like(q)
        q_norm, k_norm = self.norm(head_dim), self.norm(head_dim)
        expected = kernels.norm_rope(q, k, q_norm, k_norm, rope, transformer.EPS, transformer.rotate)
        with self.fused():
            actual = kernels.norm_rope(q, k, q_norm, k_norm, rope, transformer.EPS, transformer.rotate)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-5)

    def test_block(self):
        dim, heads, height, width, text = 64, 4, 3, 4, 5
        block = transformer.Block(dim, 96, heads, fused_qkv=True)
        with torch.no_grad():
            for name, param in block.named_parameters():
                param.copy_(torch.randn(param.shape) * (0.3 if "norm" in name else 0.1) + (1.0 if "norm" in name else 0.0))
        x = torch.randn(2, height * width + text, dim)
        mod = torch.randn(2, 6, dim) * 0.5
        rope = transformer.rope_table(height, width, text, dim // heads, 10000.0, "cpu")
        with torch.no_grad():
            expected = block(x, rope, mod, torch.float32)
            with self.fused():
                actual = block(x, rope, mod, torch.float32)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertIsNot(actual, x)  # out of place: the skipped blocks keep their input


if __name__ == "__main__":
    unittest.main()
