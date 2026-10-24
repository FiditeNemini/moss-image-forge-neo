# MOSS-Image support for Forge Neo
# MOSS Image Edit: the DiT's reference branch against the contract written as one masked joint sequence
# (tests/reference/moss_image_edit.py) - masking, t = 0 modulation of the references, their RoPE, the per-run cache,
# the negative pass's skipped blocks - and the edit LoRA's alpha and key names.

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from forge_moss.native import edit, lora, settings
from forge_moss.native.transformer import reference_rope
from tests.reference import moss_image_dit, moss_image_edit
from tests.test_transformer import FIXTURES, TINY, inputs, pair


def references(sizes=((2, 3), (3, 2)), seed=5):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, TINY["latent_channels"], h, w, generator=generator) for h, w in sizes]


def tolerance(got, expected):
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


class TestReferenceBranch(unittest.TestCase):
    def setUp(self):
        edit.new_generation()

    def test_matches_masked_joint_sequence(self):
        reference, model = pair(seed=4)
        x, sigma, context = inputs(batch=2)
        refs = references()
        with torch.no_grad():
            got = model(x, sigma, context, ref_latents=refs)
            expected = moss_image_edit.edit_forward(reference, x, sigma * 1000, context, refs)
        self.assertEqual(got.shape, x.shape)
        tolerance(got, expected)
        # the references change the prediction
        self.assertFalse(torch.allclose(got, reference(x, sigma * 1000, context), atol=1e-3))

    def test_fused_qkv(self):
        reference, model = pair(seed=6, fused=True)
        x, sigma, context = inputs(batch=1)
        refs = references(((3, 3),))
        with torch.no_grad():
            tolerance(model(x, sigma, context, ref_latents=refs),
                      moss_image_edit.edit_forward(reference, x, sigma * 1000, context, refs))

    def test_the_checks_are_sensitive(self):
        # the masked reference itself tells a wrong mask and a wrong reference time apart
        reference, _ = pair(seed=4)
        x, sigma, context = inputs(batch=1)
        refs = references()
        with torch.no_grad():
            right = moss_image_edit.edit_forward(reference, x, sigma * 1000, context, refs)
            with mock.patch.object(torch.nn.functional, "scaled_dot_product_attention",
                                   new=lambda q, k, v, attn_mask=None: torch.nn.functional.softmax(
                                       q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, -1) @ v):
                unmasked = moss_image_edit.edit_forward(reference, x, sigma * 1000, context, refs)
            original = moss_image_edit._sinusoid
            with mock.patch.object(moss_image_edit, "_sinusoid",
                                   side_effect=lambda t, d: original(t + (t == 0) * 730.0, d)):
                timed = moss_image_edit.edit_forward(reference, x, sigma * 1000, context, refs)
        self.assertFalse(torch.allclose(right, unmasked, atol=1e-3))
        self.assertFalse(torch.allclose(right, timed, atol=1e-3))

    def test_references_attend_only_to_references(self):
        _, model = pair(seed=7)
        refs = references()
        with torch.no_grad():
            x, sigma, context = inputs(batch=1, seed=1)
            model(x, sigma, context, ref_latents=refs)
            first = model._reference_cache[2]
            model.clear_reference_cache()
            x, sigma, context = inputs(batch=1, height=4, width=2, text=3, seed=9)
            model(x, sigma * 0.1, context, ref_latents=refs)
            second = model._reference_cache[2]
        self.assertIsNot(first, second)
        for (k1, v1), (k2, v2) in zip(first, second):
            torch.testing.assert_close(k1, k2, rtol=0, atol=0)
            torch.testing.assert_close(v1, v2, rtol=0, atol=0)

    def test_reference_rope(self):
        sizes = [(2, 3), (3, 2)]
        head_dim = TINY["dim"] // TINY["num_heads"]
        bands = moss_image_dit._RoPE(head_dim, TINY["rope_base"]).bands
        coordinates = [moss_image_edit._grid_coordinates(h, w, i + 1, 0) for i, (h, w) in enumerate(sizes)]
        expected = moss_image_edit._rope(torch.cat(coordinates), bands)[0]
        torch.testing.assert_close(reference_rope(sizes, head_dim, TINY["rope_base"], "cpu"), expected,
                                   rtol=0, atol=0)
        # reference 2 (3 x 2): first axis 2, f 0, its own grid centred
        self.assertEqual(coordinates[1][0].tolist(), [2.0, 0.0, -1.0, -0.5])

    def test_kept_for_one_sampling_run(self):
        _, model = pair(seed=8)
        x, sigma, context = inputs(batch=1)
        refs = references()
        with torch.no_grad():
            model(x, sigma, context, ref_latents=refs)
            cached = model._reference_cache[2]
            model(x, sigma * 0.5, context[:, :4], ref_latents=refs)  # next step / the other CFG pass
            self.assertIs(model._reference_cache[2], cached)
            model(x, sigma, context, ref_latents=[r.clone() for r in refs])  # other references
            self.assertIsNot(model._reference_cache[2], cached)
            cached = model._reference_cache[2]
            edit.new_generation()  # next sampling run (a LoRA may have changed)
            model(x, sigma, context, ref_latents=model._reference_cache[1])
            self.assertIsNot(model._reference_cache[2], cached)

    def test_no_references_is_plain_model(self):
        reference, model = pair(seed=9)
        x, sigma, context = inputs(batch=2)
        with torch.no_grad():
            # outside Forge edit.references() is empty: txt2img and img2img are unchanged
            tolerance(model(x, sigma, context), reference(x, sigma * 1000, context))
            tolerance(model(x, sigma, context, ref_latents=[]), reference(x, sigma * 1000, context))
        self.assertIsNone(model._reference_cache)

    def test_negative_pass_skips_target_and_text_blocks(self):
        reference, model = pair(seed=10)
        x, sigma, context = inputs(batch=2)
        refs = references()
        with torch.no_grad(), mock.patch.object(settings, "uncond_skip_layers", return_value=frozenset({1})):
            got = model(x, sigma, context, transformer_options={"cond_or_uncond": [0, 1]}, ref_latents=refs)
            only_uncond = model(x[1:], sigma[1:], context[1:], transformer_options={"cond_or_uncond": [1]},
                                ref_latents=refs)
            plain = moss_image_edit.edit_forward(reference, x, sigma * 1000, context, refs)
            skipped = moss_image_edit.edit_forward(reference, x, sigma * 1000, context, refs, skip_layers={1})
        tolerance(got[0], plain[0])
        tolerance(got[1], skipped[1])
        tolerance(only_uncond[0], skipped[1])
        self.assertFalse(torch.allclose(plain[1], skipped[1], atol=1e-3))


class TestReferenceImages(unittest.TestCase):
    def test_fit(self):
        image = torch.rand(1, 3, 600, 1000) * 2 - 1
        fitted = edit.fit(image, 512)
        self.assertEqual(tuple(fitted.shape), (1, 3, 304, 512))
        self.assertLessEqual(float(fitted.abs().max()), 1.0)
        self.assertIsNone(edit.fit(torch.zeros(1, 3, 320, 512), 512))
        self.assertIsNone(edit.fit(torch.zeros(1, 3, 1024, 768), 0))
        # never upscaled, only put on the 16 grid
        self.assertEqual(tuple(edit.fit(torch.zeros(1, 3, 100, 200), 512).shape), (1, 3, 96, 192))


class TestEditLora(unittest.TestCase):
    def adapter(self, folder, alpha=32.0, rank=64, embed=True):
        modules = {f"layers.{i}.proj.{p}": {"alpha": alpha, "rank": rank, "dropout": 0.0, "param_dtype": "bfloat16",
                                            "in_features": 8, "out_features": 8}
                   for i in range(2) for p in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")}
        config = {"format": edit.FORMAT, "format_version": 1, "modules": modules, "targets": sorted(modules)}
        state = {}
        for name in modules:
            state[f"{name}.lora_A"] = torch.zeros(rank, 8)
            state[f"{name}.lora_B"] = torch.zeros(8, rank)
        path = Path(folder) / "moss_image_edit.safetensors"
        save_file(state, str(path), metadata={"adapter_config": json.dumps(config)} if embed else None)
        if not embed:
            (Path(folder) / "adapter_config.json").write_text(json.dumps(config))
        return path, state

    def test_alpha_from_embedded_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path, state = self.adapter(folder)
            alphas = edit.adapter_alphas(path)
        self.assertEqual(len(alphas), 12)
        self.assertEqual(alphas["layers.1.proj.ffn_down"], 32.0)
        state["layers.0.proj.q.alpha"] = torch.tensor(8.0)  # an alpha the file has wins
        patched = edit.with_alphas(state, alphas)
        self.assertEqual(float(patched["layers.0.proj.k.alpha"]), 32.0)
        self.assertEqual(float(patched["layers.0.proj.q.alpha"]), 8.0)
        self.assertEqual(len(patched), len(state) + 11)

    def test_alpha_from_adapter_config_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.adapter(folder, alpha=64.0, embed=False)
            self.assertEqual(set(edit.adapter_alphas(path).values()), {64.0})

    def test_other_loras_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "other.safetensors"
            save_file({"x.lora_A": torch.zeros(1, 1)}, str(path), metadata={"ss_network_dim": "4"})
            self.assertEqual(edit.adapter_alphas(path), {})
        self.assertEqual(edit.adapter_alphas(Path(folder) / "missing.safetensors"), {})

    def test_every_target_maps_on_comfy_org_builds(self):
        # the adapter's 252 targets (layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}) against the fused bf16 build
        header = json.loads((FIXTURES / "moss_image_sft_bf16.shapes.json").read_text())
        state = {f"diffusion_model.{k}": torch.empty(v["shape"], device="meta") for k, v in header.items()
                 if k.endswith(".weight")}
        keys = lora.moss_lora_keys(state)
        targets = [f"layers.{i}.proj.{p}" for i in range(42) for p in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")]
        self.assertTrue(all(t in keys for t in targets))
        self.assertEqual(keys["layers.41.proj.v"], ("diffusion_model.layers.41.proj.qkv.weight", (0, 6144, 3072)))
        self.assertEqual(keys["layers.0.proj.ffn_up_gate"], "diffusion_model.layers.0.proj.ffn_up_gate.weight")


if __name__ == "__main__":
    unittest.main()
