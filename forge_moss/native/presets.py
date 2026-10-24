# MOSS-Image support for Forge Neo
# A "moss" UI preset, added at runtime: PresetArch gains a member and the preset's options are registered through
# Forge Neo's own presets.register, so they match every other preset. Same mechanism as the FLUX 3 extension.
# Forge Neo 2.31+ already has the member (its built-in MOSS-Image support): the extension then reuses it.

from enum import Enum

from modules_forge import presets

PRESET = "moss"

# upstream's sampling settings, for the SFT and Base checkpoints alike: flow-matching Euler, 50 steps, CFG 3
SAMPLER = "Euler"
SCHEDULER = "MOSS"
FALLBACK_SCHEDULER = "Simple"
STEPS = 50
CFG = 3.0


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return key.startswith(f"{PRESET}_") or key.endswith(f"_{PRESET}")


def register() -> None:
    arch = _add_enum_member(presets.PresetArch, PRESET)

    from modules import sd_schedulers

    presets.SAMPLERS[arch] = SAMPLER
    # without the MOSS scheduler (it failed to register), Simple: the same shifted curve, not snapped
    presets.SCHEDULERS[arch] = SCHEDULER if SCHEDULER in sd_schedulers.schedulers_map else FALLBACK_SCHEDULER
    presets.STEPS[arch] = STEPS
    presets.CFG[arch] = CFG

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if not _is_preset_option(key):
            continue
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
        elif key.endswith("_scheduler"):
            # Forge Neo 2.31+ has a "moss" preset of its own, made for its built-in backend (Simple); the extension's
            # model takes the MOSS scheduler. Only the default changes: a value saved in Settings still wins
            shared.opts.data_labels[key].default = info.default
