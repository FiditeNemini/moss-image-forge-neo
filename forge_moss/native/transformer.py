# MOSS-Image support for Forge Neo
# The MOSS-Image DiT, ported from the official pipeline (OpenMOSS-Team/MOSS-Image-SFT, pipeline_moss_image.py,
# MOSSImageTransformer; see MOSS_IMAGE_REVISION in native/__init__.py).
#
# The prompt goes through a projection and two text-only refinement blocks; then 42 single-stream blocks run the
# joint sequence [image, text] under one adaLN modulation shared by every block (computed once from the time). Each
# block is attention + a SiTU-gated MLP, with per-head RMS norms on q and k and a 4-axis RoPE: (r, f, h, w), the image
# centred on its grid at f = text length, the text along f. The final layer reads the image tokens only.
#
# Upstream's guidance skips block 22 in the negative-prompt pass of CFG ("uncond_skip_layers"): Forge runs the
# positive and negative passes as one batch or two and says which is which in transformer_options["cond_or_uncond"],
# so the skipped blocks are skipped per sample (settings.uncond_skip_layers).
#
# MOSS Image Edit (edit.py) adds reference images: their latents, embedded like the target's, run as a prefix branch
# under the modulation of t = 0, attending only to themselves, at RoPE (r = reference index + 1, f = 0, h and w centred
# on their own grid). Every target and text token also attends to the references' keys and values, block by block.
# Those depend on nothing but the references, so they are computed once per sampling run and kept for every step and
# both CFG passes (upstream's sequence is [text, references, target]; attention is the same under any token order).
#
# Layers are plain torch modules: inside Forge's using_forge_operations they become Forge operations (dtype,
# INT8 ConvRot quantization, offload, LoRA), outside it they stay torch modules for CPU tests. Differences from
# upstream: a batch runs together instead of sample by sample (Forge never batches prompts of different lengths for
# this model, see patches._hook_cond_concat, so no sample needs a text mask); Forge's attention runs through
# kernels.attention; under float16 the residual stream stays in float32.

import torch
import torch.nn as nn

from . import edit, kernels, settings

TIME_DIM = 256
EPS = 1e-6
UNCOND = 1  # backend.sampling.sampling_function.UNCOND


def time_embedding(sigma: torch.Tensor, dim: int = TIME_DIM) -> torch.Tensor:
    # upstream's timesteps are the scheduler's float32 sigmas times 1000; the sinusoid is computed in float64 and fed
    # in float32
    times = (sigma.float() * 1000.0).double()
    half = dim // 2
    freq = 10000 ** (-torch.arange(half, device=sigma.device, dtype=torch.float64) / half)
    phase = times[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), dim=-1).float()


def _bands(head_dim: int, base: float, device) -> torch.Tensor:
    num_bands = head_dim // 8
    return (1 / (base ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))).to(device)


def _grid(height: int, width: int, r: float, f: float, device) -> torch.Tensor:
    """(H*W, 4) coordinates (r, f, h, w) of a latent grid, h and w centred on it."""
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    grid = torch.stack((torch.full_like(h, r), torch.full_like(h, f), h, w), -1).flatten(0, 1).float()
    grid[:, 2] -= (height - 1) / 2
    grid[:, 3] -= (width - 1) / 2
    return grid


def _table(coordinates: torch.Tensor, head_dim: int, base: float) -> torch.Tensor:
    phase = coordinates.unsqueeze(-1) * _bands(head_dim, base, coordinates.device)
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)


def rope_table(height: int, width: int, text_length: int, head_dim: int, base: float, device) -> torch.Tensor:
    """(H*W + T, head_dim): sin then cos of the four axes' phases, image tokens first, in float32."""
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    return _table(torch.cat((_grid(height, width, 0, text_length, device), text), 0), head_dim, base)


def reference_rope(sizes, head_dim: int, base: float, device) -> torch.Tensor:
    """(sum of h*w, head_dim) for reference latents of the given (h, w): reference i at r = i + 1, f = 0."""
    grids = [_grid(h, w, index + 1, 0, device) for index, (h, w) in enumerate(sizes)]
    return _table(torch.cat(grids, 0), head_dim, base)


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x (B, L, H, D), rope (L, D): rotation by halves, in x's dtype (upstream casts the table to the activations')
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1)[:, None]
    cos = torch.cat((cos, cos), -1)[:, None]
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def heads_qkv(q, k, v, q_norm, k_norm, rope, heads):
    """q, k, v (B, L, H*D) -> (B, H, L, D) each, per-head norms and RoPE on q and k."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q, k = kernels.norm_rope(q.reshape(shape), k.reshape(shape), q_norm, k_norm, rope, EPS, rotate)
    v = v.reshape(shape).transpose(1, 2)
    return q.transpose(1, 2).to(v.dtype), k.transpose(1, 2).to(v.dtype), v


def with_reference(k, v, reference):
    """k, v (B, H, L, D) followed by the references' (1, H, R, D), which every sample of the batch shares."""
    ref_k, ref_v = (r.to(k.dtype).expand(k.shape[0], -1, -1, -1) for r in reference)
    return torch.cat((k, ref_k), 2), torch.cat((v, ref_v), 2)


def attention(q, k, v, q_norm, k_norm, rope, heads, transformer_options=None):
    """q, k, v (B, L, H*D) -> (B, L, H*D), per-head norms and RoPE on q and k, every token attending to every other."""
    q, k, v = heads_qkv(q, k, v, q_norm, k_norm, rope, heads)
    return kernels.attention(q, k, v, transformer_options=transformer_options)


def gated_residual(x, y, gate):
    # one pass instead of a product, a cast and a sum; out of place, the skipped blocks keep their input
    return torch.addcmul(x, y.to(x.dtype), gate.to(x.dtype))


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # torch.ones, not empty: the checkpoint overwrites it
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)
        return y.to(x.dtype) * kernels.cast_to(self.weight, x.dtype, x.device)


def head_norm(norm, x, heads):
    # unused by the fused path (kernels.norm_rope); kept for NegPiP, whose block forward checks for and calls it
    return norm(x.unflatten(-1, (heads, -1))).flatten(2)


class TextBlock(nn.Module):
    """Text refinement: pre-norm attention and SiTU MLP, no modulation, biases on every projection."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, dtype, transformer_options=None):
        q, k, v = self.qkv(self.norm1(x).to(dtype)).chunk(3, -1)
        out = attention(q, k, v, self.q_norm, self.k_norm, rope, self.heads, transformer_options)
        x = x + self.proj(out.to(dtype)).to(x.dtype)
        y = self.norm2(x).to(dtype)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y))).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, context, rope, dtype, res_dtype, transformer_options=None):
        x = self.proj(context.to(dtype)).to(res_dtype)
        for block in self.blocks:
            x = block(x, rope, dtype, transformer_options)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads, fused_qkv=False):
        super().__init__()
        self.heads = heads
        self.fused_qkv = fused_qkv
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        # Comfy-Org's files fuse q, k, v into one projection; upstream's diffusers export keeps them apart
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({n: nn.Linear(dim, 3 * dim if n == "qkv" else dim, bias=False) for n in names})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, mod, dtype, transformer_options=None, reference=None, keep_kv=False):
        """mod (B, 6, D): attention shift, scale, gate, then the MLP's. reference: the references' (k, v) of this
        block, attended to as well; keep_kv also returns this call's own (k, v) (the reference branch)."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in mod.unbind(1))
        y = kernels.modulated_rms_norm(self.norm1, x, scale_a, shift_a, EPS).to(dtype)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, -1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        q, k, v = heads_qkv(q, k, v, self.q_norm, self.k_norm, rope, self.heads)
        own = k, v
        if reference is not None:
            k, v = with_reference(k, v, reference)
        out = kernels.attention(q, k, v, transformer_options=transformer_options)
        x = gated_residual(x, self.proj["o"](out.to(dtype)), gate_a)
        y = kernels.modulated_rms_norm(self.norm2, x, scale_f, shift_f, EPS).to(dtype)
        out = kernels.linear_situ(self.proj["ffn_down"], self.proj["ffn_up_gate"](y), situ)
        x = gated_residual(x, out, gate_f)
        return (x, own) if keep_kv else x


class MossImageDiT(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=TIME_DIM,
                 latent_channels=128, rope_base=10000.0, fused_qkv=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.rope_base = rope_base
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads, fused_qkv) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        # (sampling run, reference latents, per-block (k, v)): see reference_kv
        self._reference_cache = None

    def clear_reference_cache(self):
        self._reference_cache = None

    def reference_kv(self, latents, dtype, res_dtype, device, transformer_options=None):
        """Per block, the (k, v) (1, H, R, D) of the reference latents (each (1, C, h, w)), or None without any.

        The references run through every block as their own sequence, modulated as at t = 0. Kept while the same
        latent tensors come back within one sampling run (edit.generation): every step, both CFG passes.
        """
        if not latents:
            return None
        key = (edit.generation(), dtype, device)
        cache = self._reference_cache
        if (cache is not None and cache[0] == key and len(cache[1]) == len(latents)
                and all(a is b for a, b in zip(cache[1], latents))):
            return cache[2]
        self._reference_cache = None
        time = self.time_embedding(time_embedding(torch.zeros(1, device=device), self.freq_dim).to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(1, 6, -1)
        rope = reference_rope([r.shape[-2:] for r in latents], self.dim // self.num_heads, self.rope_base, device)
        rope = rope.to(dtype)
        tokens = torch.cat([self.video_embedder(r.to(device=device, dtype=dtype).flatten(2).transpose(1, 2))
                            for r in latents], 1).to(res_dtype)
        kv = []
        for layer in self.layers:
            tokens, own = layer(tokens, rope, mod, dtype, transformer_options, keep_kv=True)
            kv.append(own)
        self._reference_cache = (key, tuple(latents), kv)
        return kv

    def skipped(self, transformer_options, batch, device):
        """(B,) True where block skipping applies (the negative-prompt pass), or None when nothing is skipped."""
        layers = settings.uncond_skip_layers(len(self.layers))
        chunks = (transformer_options or {}).get("cond_or_uncond")
        if not layers or not chunks or UNCOND not in chunks or batch % len(chunks):
            return None
        per_chunk = batch // len(chunks)
        flags = torch.tensor([c == UNCOND for c in chunks], device=device).repeat_interleave(per_chunk)
        return layers, flags

    def forward(self, x, timestep, context, transformer_options=None, ref_latents=None, **kwargs):
        """x (B, 128, H, W) latents, timestep (B,) = sigma, context (B, T, 4096) -> velocity (B, 128, H, W).

        ref_latents: MOSS Image Edit's reference latents, each (1, 128, h, w); by default the ones Forge holds when
        the edit setting is on (edit.references).
        """
        b, _, h, w = x.shape
        device = x.device
        # Forge hands the inputs over in the computation dtype (k_model.apply_model); the weights may be stored in
        # another one (INT8, manual cast), so the context is what says which
        dtype = context.dtype
        # fp16 overflows the residual stream of large DiTs (the Iris and FLUX 3 ports saw it); float32 there
        res_dtype = torch.float32 if dtype == torch.float16 else dtype
        context = context.to(device=device, dtype=dtype)
        if context.shape[0] != b:
            context = context.expand(b, -1, -1)
        references = self.reference_kv(edit.references() if ref_latents is None else ref_latents, dtype, res_dtype,
                                       device, transformer_options)

        time = self.time_embedding(time_embedding(timestep.reshape(-1).expand(b), self.freq_dim).to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(b, 6, -1)
        head = self.video_head_modulation(time.to(dtype)).view(b, 2, -1)

        image_tokens = h * w
        rope = rope_table(h, w, context.shape[1], self.dim // self.num_heads, self.rope_base, device).to(dtype)
        video = self.video_embedder(x.flatten(2).transpose(1, 2).to(dtype)).to(res_dtype)
        text = self.text_embedder(context, rope[image_tokens:], dtype, res_dtype, transformer_options)
        tokens = torch.cat((video, text), 1)
        del video, text

        skip = self.skipped(transformer_options, b, device)
        for index, layer in enumerate(self.layers):
            # no extra argument without references: other extensions' block forwards (NegPiP) keep working
            extra = {} if references is None else {"reference": references[index]}
            if skip is not None and index in skip[0]:
                if bool(skip[1].all()):
                    continue
                tokens = torch.where(skip[1][:, None, None], tokens,
                                     layer(tokens, rope, mod, dtype, transformer_options, **extra))
            else:
                tokens = layer(tokens, rope, mod, dtype, transformer_options, **extra)

        out = self.final_norm_video(tokens[:, :image_tokens]) * (1 + head[:, 1, None]) + head[:, 0, None]
        out = self.final_linear_video(out.to(dtype))
        return out.transpose(1, 2).reshape(b, self.latent_channels, h, w)
