"""Latent-space upscaling, ported 1:1 from WebUI.

Ported from stable-diffusion-webui modules/processing.py:1388-1392.

WebUI calls torch.nn.functional.interpolate directly on the latent batch; it does
NOT go through ComfyUI's common_upscale (no 5D reshape, no crop handling), so the
ported code does the same thing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import LATENT_UPSCALE_MODES, LATENT_UPSCALE_MODE_NAMES
from .target import OPT_F, TargetResolution


def upscale_latent_samples(
    samples: torch.Tensor,
    tr: TargetResolution,
    upscaler: str,
    compression: int = OPT_F,
) -> torch.Tensor:
    """Upscale a latent batch (B, C, H, W) with a WebUI latent upscale mode.

    The output size is (upscale_to_y // 8, upscale_to_x // 8); WebUI applies the
    truncation crop later (see webui_hires.target.truncate_latent). Computation
    is done in float32 and the result is returned as float32 as well, matching
    WebUI's k-diffusion pipeline.
    """
    # Ported from stable-diffusion-webui modules/processing.py:1388-1392
    mode = LATENT_UPSCALE_MODES.get(upscaler)
    if mode is None:
        raise ValueError(
            f"WebUI Hires Fix: {upscaler!r} is not a latent upscale mode. "
            f"Available latent modes: {list(LATENT_UPSCALE_MODE_NAMES)}"
        )
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"WebUI Hires Fix: expected a torch.Tensor latent, got {type(samples)!r}.")
    if samples.ndim == 5 and samples.shape[2] != 1:
        raise ValueError(
            "WebUI Hires Fix: only single-frame 5D latents (B, C, 1, H, W) are supported; "
            f"got shape {tuple(samples.shape)}."
        )
    if samples.ndim not in (4, 5):
        raise ValueError(
            "WebUI Hires Fix: latent upscaling requires a 4D image latent "
            f"(B, C, H, W) or single-frame 5D latent (B, C, 1, H, W), got shape {tuple(samples.shape)}."
        )

    is_video = samples.ndim == 5
    work = samples[:, :, 0] if is_video else samples
    target_h = max(1, tr.upscale_to_y // int(compression))
    target_w = max(1, tr.upscale_to_x // int(compression))
    out = F.interpolate(
        work.float(),
        size=(target_h, target_w),
        mode=mode["mode"],
        antialias=bool(mode["antialias"]),
    )
    return out.unsqueeze(2) if is_video else out
