"""Constants and WebUI <-> ComfyUI name mappings for the Hires Fix port.

All values are copied from the WebUI source tree installed on this machine;
comments cite the original definition.
"""

from __future__ import annotations

import logging

# Ported from stable-diffusion-webui modules/shared.py:55-62
LATENT_UPSCALE_MODES: dict[str, dict[str, object]] = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

LATENT_UPSCALE_MODE_NAMES: tuple[str, ...] = tuple(LATENT_UPSCALE_MODES)

# Ported from stable-diffusion-webui modules/upscaler.py:108-150
# ("None", "Lanczos" and "Nearest" are regular entries of shared.sd_upscalers).
NON_MODEL_UPSCALERS: tuple[str, ...] = ("None", "Lanczos", "Nearest")

# Ported from stable-diffusion-webui modules/sd_samplers_kdiffusion.py:11-27
# WebUI sampler label -> ComfyUI sampler name (see SPEC.md section 2.7).
# Entries marked "approximate" have no exact ComfyUI equivalent.
WEBUI_SAMPLER_MAP: dict[str, str] = {
    "Euler": "euler",
    "Euler a": "euler_ancestral",
    "LMS": "lms",
    "Heun": "heun",
    "DPM2": "dpm_2",
    "DPM2 a": "dpm_2_ancestral",
    "DPM++ 2S a": "dpmpp_2s_ancestral",
    "DPM++ 2M": "dpmpp_2m",
    "DPM++ SDE": "dpmpp_sde",
    "DPM++ 2M SDE": "dpmpp_2m_sde",
    "DPM++ 2M SDE Heun": "dpmpp_2m_sde_heun",
    "DPM++ 3M SDE": "dpmpp_3m_sde",
    "DPM fast": "dpm_fast",
    "DPM adaptive": "dpm_adaptive",
    "DDIM": "ddim",
    "UniPC": "uni_pc",
    "LCM": "lcm",
    "PLMS": "euler",  # approximate, no ComfyUI equivalent
    "Restart": "euler",  # approximate, no ComfyUI equivalent
}

# Ported from stable-diffusion-webui modules/sd_schedulers.py:130-143
# WebUI scheduler label -> ComfyUI scheduler name (see SPEC.md section 2.7).
WEBUI_SCHEDULER_MAP: dict[str, str | None] = {
    "Automatic": None,  # falls back to the sampler's default scheduler
    "Uniform": "normal",
    "Karras": "karras",
    "Exponential": "exponential",
    "Polyexponential": "exponential",  # approximate, no ComfyUI equivalent
    "SGM Uniform": "sgm_uniform",
    "SGMUniform": "sgm_uniform",
    "KL Optimal": "kl_optimal",
    "Align Your Steps": None,  # no core equivalent (comfy_extras/nodes_align_your_steps)
    "Simple": "simple",
    "Normal": "normal",
    "DDIM": "ddim_uniform",
    "Beta": "beta",
}

# Ported from stable-diffusion-webui modules/sd_samplers_kdiffusion.py:17,22,23
# (DPM2, DPM2 a and DPM++ 3M SDE use "discard_next_to_last_sigma").
DISCARD_SIGMA_SAMPLERS: frozenset[str] = frozenset(("dpm_2", "dpm_2_ancestral", "dpmpp_3m_sde"))

# Ported from stable-diffusion-webui ui-config.json / config.json
# (ESRGAN_tile = 192, ESRGAN_tile_overlap = 8).
WEBUI_ESRGAN_TILE: int = 192
WEBUI_ESRGAN_TILE_OVERLAP: int = 8

USE_SAME_SAMPLER = "Use same sampler"
USE_SAME_SCHEDULER = "Use same scheduler"


def _comfy_sampler_names() -> tuple[str, ...]:
    try:
        import comfy.samplers
        return tuple(comfy.samplers.KSampler.SAMPLERS)
    except Exception:  # pragma: no cover - only when imported outside ComfyUI
        return ()


def _comfy_scheduler_names() -> tuple[str, ...]:
    try:
        import comfy.samplers
        return tuple(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:  # pragma: no cover - only when imported outside ComfyUI
        return ()


def _upscale_model_names() -> list[str]:
    try:
        import folder_paths
        return list(folder_paths.get_filename_list("upscale_models"))
    except Exception as exc:  # pragma: no cover - only when folder_paths is unavailable
        logging.warning("WebUI Hires Fix: could not enumerate upscale_models: %s", exc)
        return []


def is_latent_upscaler(name: str | None) -> bool:
    """True if name selects one of the six WebUI latent upscale modes."""
    return name in LATENT_UPSCALE_MODES


def is_model_upscaler(name: str | None) -> bool:
    """True if name selects a spandrel upscale model file."""
    if name is None or name in LATENT_UPSCALE_MODES or name in NON_MODEL_UPSCALERS:
        return False
    return name in _upscale_model_names()


def is_image_upscaler(name: str | None) -> bool:
    """True if name selects an image-space (pixel) upscaler."""
    return is_model_upscaler(name) or name in NON_MODEL_UPSCALERS


def list_upscalers() -> list[str]:
    """The WebUI Upscaler dropdown: latent modes + built-ins + model files."""
    names: list[str] = []
    for name in list(LATENT_UPSCALE_MODE_NAMES) + list(NON_MODEL_UPSCALERS) + _upscale_model_names():
        if name not in names:
            names.append(name)
    return names


def to_comfy_sampler_name(name: str | None) -> str | None:
    """Map a WebUI/ComfyUI sampler name to a ComfyUI sampler name (None = unset)."""
    if not name or not isinstance(name, str):
        return None
    if name in (USE_SAME_SAMPLER, USE_SAME_SCHEDULER):
        return None
    mapped = WEBUI_SAMPLER_MAP.get(name)
    if mapped is not None:
        return mapped
    comfy_names = _comfy_sampler_names()
    if comfy_names:
        return name if name in comfy_names else None
    return name  # ComfyUI not importable: assume the caller passed a ComfyUI name


def to_comfy_scheduler_name(name: str | None) -> str | None:
    """Map a WebUI/ComfyUI scheduler name to a ComfyUI scheduler name."""
    if not name or not isinstance(name, str):
        return None
    if name in (USE_SAME_SAMPLER, USE_SAME_SCHEDULER):
        return None
    if name in WEBUI_SCHEDULER_MAP:
        return WEBUI_SCHEDULER_MAP[name]
    comfy_names = _comfy_scheduler_names()
    if comfy_names:
        return name if name in comfy_names else None
    return name
