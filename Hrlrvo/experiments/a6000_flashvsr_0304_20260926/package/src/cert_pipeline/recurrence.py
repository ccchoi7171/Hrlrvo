"""CPU-only recurrence contract. Shared JSON validation; no model imports."""

import copy
import math


DEFAULT_RECURRENCE = {
    "rebase_temporal_position": True,
    "use_latent_carry": True,
    "generated_prefix_strength": 1.0,
    "lineart_global_prior": True,
    "lineart_bias_schedule": [
        {"start_step": 0, "end_step": 9, "strength": 0.5},
        {"start_step": 10, "end_step": 19, "strength": 0.3},
        {"start_step": 20, "end_step": 29, "strength": 0.1},
    ],
    "lineart_source": "chunk00_first_final_rgb_frame",
    "lineart_start_chunk": 1,
    "lineart_reference_resolution": [3840, 2160],
    "lineart_source_resize": "none",
    "lineart_global_map": [30, 17],
    "lineart_density_method": "direct_4k_global_footprint_density",
    "lineart_density_padding": {"top": 8, "bottom": 8, "left": 0, "right": 0, "value": 0},
    "lineart_density_normalizer": 0.05,
    "lineart_extraction": {
        "grayscale": "RGB2GRAY",
        "gaussian_kernel": [3, 3],
        "gaussian_sigma": 0.8,
        "border": "BORDER_REFLECT_101",
        "canny_low": 35,
        "canny_high": 100,
        "aperture": 3,
        "L2gradient": True,
        "component_connectivity": 8,
        "component_min_pixels": 12,
        "component_min_long_span": 12,
        "clahe": False,
        "morphology": False,
        "dilation": False,
        "thinning": False,
    },
}


def validate_recurrence(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULT_RECURRENCE):
        raise ValueError("outpainting.recurrence requires the complete recorded contract")
    for key in ("rebase_temporal_position", "use_latent_carry", "lineart_global_prior"):
        if type(value[key]) is not bool:
            raise ValueError(f"recurrence.{key} must be boolean")
    strength = value["generated_prefix_strength"]
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("generated_prefix_strength must be finite in [0,1]")
    adjustable = {"rebase_temporal_position", "use_latent_carry",
                  "generated_prefix_strength", "lineart_global_prior"}
    for key, expected in DEFAULT_RECURRENCE.items():
        if key not in adjustable and value[key] != expected:
            raise ValueError(f"Unsupported recurrence.{key}: expected {expected!r}")
    return copy.deepcopy(value)
