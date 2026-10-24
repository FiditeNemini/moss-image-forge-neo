# MOSS-Image support for Forge Neo
# The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests).
# Same module as the FLUX 3 and Iris 3B extensions' kernels.py.

import torch

try:
    from backend import attention as _forge_attention
    from backend.memory_management import cast_to as _forge_cast_to
except ImportError:  # outside Forge Neo: eager backend on CPU
    _forge_attention = None
    _forge_cast_to = None

try:
    from backend.operations_mixed_precision import INPUT_ACT_EAGER as _forge_input_acts
    from backend.operations_mixed_precision import linear_input_act as _forge_linear_input_act
except ImportError:  # outside Forge Neo, or a Forge Neo without fused input activations
    _forge_input_acts = {}
    _forge_linear_input_act = None

try:
    import comfy_kitchen as _ck
except ImportError:
    _ck = None


def _ck_version():
    try:
        from importlib.metadata import version

        return tuple(int(p) for p in version("comfy-kitchen").split(".")[:3])
    except Exception:
        return (0,)


def _situ_eager(x):
    hidden, gate = x.chunk(2, -1)
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


# Forge Neo registers no SiTU input activation of its own; register it here so the INT8 ConvRot path folds SiTU into
# comfy-kitchen's activation quantization (0.2.38+) without a patched Forge. Forge reads this dict by name per call
if _forge_linear_input_act is not None and "situ" not in _forge_input_acts and _ck_version() >= (0, 2, 38):
    _forge_input_acts["situ"] = _situ_eager

__all__ = ["attention", "cast_to", "linear_situ", "modulated_rms_norm", "norm_rope"]


def cast_to(weight, dtype=None, device=None):
    if _forge_cast_to is not None:
        return _forge_cast_to(weight, dtype=dtype, device=device)
    return weight.to(dtype=dtype, device=device)


def _fused(x):
    # comfy-kitchen's fused kernels have no autograd: GPU inference inside Forge only, the CPU tests stay eager
    return _ck is not None and _forge_attention is not None and x.is_cuda and not torch.is_grad_enabled()


def modulated_rms_norm(norm, x, scale, shift, eps):
    """norm(x) * (1 + scale) + shift.

    Fused into one comfy-kitchen pass the way ComfyUI does it: the norm weight is folded into the scale,
    rmsnorm(x) * w * (1 + scale) = rmsnorm(x) * (1 + (w * (1 + scale) - 1)).
    """
    if _fused(x):
        weight = cast_to(norm.weight, torch.float32, x.device)
        scale = (weight * (1 + scale.float()) - 1).to(x.dtype)
        return _ck.rms_adaln(x, scale, shift.to(x.dtype), eps)
    return norm(x) * (1 + scale) + shift


_rope_cache = [None, None]


def _rope_rotations(rope):
    # (L, D) sin | cos table -> (1, L, 1, D / 2, 2, 2) rotation matrices, the layout of comfy-kitchen's RoPE kernels.
    # Every block of a step gets the same table, so the last one is kept
    if _rope_cache[0] is not rope:
        sin, cos = rope.float().chunk(2, -1)
        _rope_cache[:] = rope, torch.stack((cos, -sin, sin, cos), -1).unflatten(-1, (2, 2))[None, :, None]
    return _rope_cache[1]


def norm_rope(q, k, q_norm, k_norm, rope, eps, rotate):
    """q, k (B, L, H, D): per-head RMSNorm, then RoPE by halves; one comfy-kitchen pass for both inside Forge."""
    if _fused(q):
        q_scale = cast_to(q_norm.weight, q.dtype, q.device)
        k_scale = cast_to(k_norm.weight, k.dtype, k.device)
        return _ck.rms_rope_split_half(q, k, _rope_rotations(rope), q_scale, k_scale, eps)
    return rotate(q_norm(q), rope), rotate(k_norm(k), rope)


def linear_situ(linear, x, situ):
    """linear(situ(hidden, gate)) for x = [hidden | gate].

    With an INT8 ConvRot weight, Forge folds SiTU into comfy-kitchen's activation quantization (comfy-kitchen
    0.2.38+), so the activation never goes through memory on its own; any other weight runs it eagerly. Older Forge
    builds and the CPU tests use the plain chunk + situ + linear.
    """
    if _forge_linear_input_act is not None and "situ" in _forge_input_acts:
        return _forge_linear_input_act(linear, x, "situ")
    hidden, gate = x.chunk(2, -1)
    return linear(situ(hidden, gate))


def attention(q, k, v, transformer_options=None):
    """q, k, v: [B, H, S, D] -> [B, S, H * D], no mask: every token of a sample attends to every other.

    transformer_options reach Forge's attention wrapper, which applies attention overrides such as the Sparse
    Attention Integrated extension. The torch fallback (CPU tests) pins the MATH backend, the plain reference
    implementation.
    """
    if _forge_attention is not None:
        # looked up per call: Forge picks its attention function at import time and other extensions may wrap it
        return _forge_attention.attention_function(q, k, v, q.shape[1], skip_reshape=True,
                                                   transformer_options=transformer_options or {})
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
