# MOSS-Image support for Forge Neo
# The "MOSS" scheduler: upstream's MOSSImageScheduler (pipeline_moss_image.py). The time-shifted grid
# shift v / (1 + (shift - 1) v) over v = linspace(1, shift / (1000 + shift - 1), steps), each sigma snapped to the
# nearest of the 1000 training sigmas (linspace(1, 1/1000)), repeats dropped, then 0. Float32 throughout, as
# upstream computes it. Forge's Simple scheduler gives the same shifted curve without the snapping.

import torch

SHIFT = 5.0
TRAIN_STEPS = 1000

NAME = "moss"
LABEL = "MOSS"
ALIASES = ("MOSS-Image", "moss_image")


def _grid(count: int, shift: float) -> torch.Tensor:
    lower = shift / (TRAIN_STEPS + shift - 1)
    values = torch.linspace(1, lower, count)
    return shift * values / (1 + (shift - 1) * values)


def schedule(n: int, shift: float = SHIFT) -> torch.Tensor:
    if n < 1:
        raise ValueError("MOSS needs at least one step")
    sigmas = _grid(n, shift)
    training = _grid(TRAIN_STEPS, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=TRAIN_STEPS - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.cat((torch.unique_consecutive(training[nearest]), torch.zeros(1)))


def sigmas(n, sigma_min=None, sigma_max=None, inner_model=None, device="cpu", **kwargs):
    return schedule(n).to(device)
