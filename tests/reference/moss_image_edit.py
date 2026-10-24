# Reference for the tests only: MOSS Image Edit's reference conditioning (chinoll/Moss-Image-Edit, edit contract
# "ref-prefix-only-t0__target-text-original-moss-v1") written as ONE masked joint sequence over the official DiT's
# modules (tests/reference/moss_image_dit.py), the plain formulation of the contract rather than the extension's
# cached prefix:
#
#   tokens [references, target, text]; a reference token is modulated as at t = 0, every other token at t;
#   reference queries see reference keys only, target and text queries see every key;
#   RoPE (r, f, h, w): reference i at (i + 1, 0, centred h, w), target at (0, text length, centred h, w), text at
#   (0, index, 0, 0).
#
# Sample by sample, no batching, everything recomputed at every call.

import torch
from torch.nn import functional as F

from .moss_image_dit import _rotate, _situ


def _sinusoid(times, dim):
    half = dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()


def _grid_coordinates(height, width, r, f):
    rows = []
    for i in range(height):
        for j in range(width):
            rows.append((r, f, i - (height - 1) / 2, j - (width - 1) / 2))
    return torch.tensor(rows, dtype=torch.float32)


def _rope(coordinates, bands):
    phase = coordinates.unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def _block(layer, x, rope, modulation, mask):
    # modulation (1, L, 6, D): per-token adaLN parameters
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(2)
    y = layer.norm1(x) * (1 + scale_a) + shift_a
    heads = layer.heads
    q = layer.q_norm(layer.proj["q"](y).unflatten(-1, (heads, -1)))
    k = layer.k_norm(layer.proj["k"](y).unflatten(-1, (heads, -1)))
    v = layer.proj["v"](y).unflatten(-1, (heads, -1)).transpose(1, 2)
    q, k = _rotate(q, rope).transpose(1, 2), _rotate(k, rope).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).flatten(2)
    x = x + layer.proj["o"](out) * gate_a
    y = layer.norm2(x) * (1 + scale_f) + shift_f
    hidden, gate = layer.proj["ffn_up_gate"](y).chunk(2, -1)
    return x + layer.proj["ffn_down"](_situ(hidden, gate)) * gate_f


def edit_forward(model, hidden_states, timestep, encoder_hidden_states, references, skip_layers=()):
    """model: moss_image_dit.MOSSImageTransformer; hidden_states (B, C, H, W); timestep (B,) in [0, 1000];
    encoder_hidden_states (B, T, text_dim); references: list of (1, C, h, w). skip_layers: blocks the target and text
    leave out (the references always run every block)."""
    batch, _, height, width = hidden_states.shape
    bands = model.rope.bands
    zero = model.shared_block_modulation(model.time_embedding(_sinusoid(torch.zeros(1), 256))).view(1, 6, -1)
    ref_tokens = [model.video_embedder(r.flatten(2).transpose(1, 2)) for r in references]
    ref_coordinates = [_grid_coordinates(r.shape[-2], r.shape[-1], index + 1, 0) for index, r in enumerate(references)]
    prefix = sum(t.shape[1] for t in ref_tokens)
    outputs = []
    for index in range(batch):
        context = encoder_hidden_states[index:index + 1]
        text_length = context.shape[1]
        time = model.time_embedding(_sinusoid(timestep[index:index + 1], 256))
        mod = model.shared_block_modulation(time).view(1, 6, -1)
        head = model.video_head_modulation(time).view(1, 2, -1)
        text_coordinates = torch.zeros(text_length, 4)
        text_coordinates[:, 1] = torch.arange(text_length)
        target_coordinates = _grid_coordinates(height, width, 0, text_length)
        rope = _rope(torch.cat((*ref_coordinates, target_coordinates, text_coordinates)), bands)
        text = model.text_embedder(context, _rope(text_coordinates, bands))
        video = model.video_embedder(hidden_states[index:index + 1].flatten(2).transpose(1, 2))
        x = torch.cat((*ref_tokens, video, text), 1)
        length = x.shape[1]
        modulation = torch.cat((zero.unsqueeze(1).expand(1, prefix, 6, -1),
                                mod.unsqueeze(1).expand(1, length - prefix, 6, -1)), 1)
        mask = torch.ones(length, length, dtype=torch.bool)
        mask[:prefix, prefix:] = False
        for layer_index, layer in enumerate(model.layers):
            out = _block(layer, x, rope, modulation, mask)
            if layer_index in skip_layers:
                out = torch.cat((out[:, :prefix], x[:, prefix:]), 1)
            x = out
        x = model.final_norm_video(x[:, prefix:prefix + height * width])
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = model.final_linear_video(x)
        outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
    return torch.cat(outputs)
