"""Target resolution and truncation, ported 1:1 from WebUI.

Ported from stable-diffusion-webui modules/processing.py:1213-1250 (target
resolution) and modules/processing.py:1427 (latent truncation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Ported from stable-diffusion-webui modules/processing.py:40
OPT_F = 8
"""WebUI's fixed ratio between image pixels and latent pixels."""


@dataclass(frozen=True)
class TargetResolution:
    """Result of compute_target_resolution.

    Attributes
    ----------
    target_w, target_h:
        The requested resize target (WebUI hr_resize_x / hr_resize_y); 0 when the
        size is derived from upscale_by. Units: image pixels.
    upscale_to_x, upscale_to_y:
        The size the first pass image is actually upscaled to before sampling.
    truncate_x, truncate_y:
        Latent-space cropping applied after upscaling when an explicit
        width/height with a mismatching aspect ratio was requested.
        Units: latent pixels (WebUI already divides by OPT_F = 8).
    """

    target_w: int
    target_h: int
    upscale_to_x: int
    upscale_to_y: int
    truncate_x: int
    truncate_y: int
    compression: int = OPT_F
    """Latent spatial compression ratio (8 for SD/FLUX/Krea2/Qwen-Image,
    16 for Flux2/MOSS/Qwen-Image 2.1)."""

    @property
    def latent_w(self) -> int:
        return self.upscale_to_x // self.compression

    @property
    def latent_h(self) -> int:
        return self.upscale_to_y // self.compression


def compute_target_resolution(
    first_w: int,
    first_h: int,
    upscale_by: float,
    resize_width: int,
    resize_height: int,
    compression: int = OPT_F,
) -> TargetResolution:
    """Compute WebUI's hires target resolution.

    first_w/first_h are the first pass image dimensions
    (latent W*compression / latent H*compression). WebUI hardcodes compression=8;
    newer ComfyUI models use 16 (Flux2/MOSS/Qwen-Image 2.1), which the plugin
    detects from the VAE / model latent format.
    Mirrors StableDiffusionProcessingTxt2Img.calculate_target_resolution.
    """
    # Ported from stable-diffusion-webui modules/processing.py:1213-1250
    first_w = int(first_w)
    first_h = int(first_h)
    if first_w <= 0 or first_h <= 0:
        raise ValueError(
            "WebUI Hires Fix: invalid first pass size "
            f"{first_w}x{first_h}; latent width/height must be positive."
        )
    resize_width = int(resize_width or 0)
    resize_height = int(resize_height or 0)
    upscale_by = float(upscale_by)
    if resize_width < 0 or resize_height < 0:
        raise ValueError(
            "WebUI Hires Fix: resize_width/resize_height must be >= 0, "
            f"got {resize_width}/{resize_height}."
        )
    if resize_width == 0 and resize_height == 0 and upscale_by <= 0:
        raise ValueError(f"WebUI Hires Fix: upscale_by must be > 0, got {upscale_by}.")

    upscale_to_x = 0
    upscale_to_y = 0
    truncate_x = 0
    truncate_y = 0

    if resize_width == 0 and resize_height == 0:
        upscale_to_x = int(first_w * upscale_by)
        upscale_to_y = int(first_h * upscale_by)
    elif resize_height == 0:
        upscale_to_x = resize_width
        upscale_to_y = resize_width * first_h // first_w
    elif resize_width == 0:
        upscale_to_x = resize_height * first_w // first_h
        upscale_to_y = resize_height
    else:
        target_w = resize_width
        target_h = resize_height
        src_ratio = first_w / first_h
        dst_ratio = resize_width / resize_height
        if src_ratio < dst_ratio:
            upscale_to_x = resize_width
            upscale_to_y = resize_width * first_h // first_w
        else:
            upscale_to_x = resize_height * first_w // first_h
            upscale_to_y = resize_height
        truncate_x = (upscale_to_x - target_w) // compression
        truncate_y = (upscale_to_y - target_h) // compression

    return TargetResolution(
        target_w=resize_width,
        target_h=resize_height,
        upscale_to_x=upscale_to_x,
        upscale_to_y=upscale_to_y,
        truncate_x=truncate_x,
        truncate_y=truncate_y,
        compression=compression,
    )


def truncate_latent(samples: torch.Tensor, tr: TargetResolution) -> torch.Tensor:
    """Apply WebUI's asymmetric truncation to a latent batch.

    Works for both 4D (B, C, H, W) and 5D (B, C, T, H, W, T=1) image latents:
    it always crops the last two (spatial) dimensions.
    """
    # Ported from stable-diffusion-webui modules/processing.py:1427
    if tr.truncate_x == 0 and tr.truncate_y == 0:
        return samples
    return samples[
        ...,
        tr.truncate_y // 2 : samples.shape[-2] - (tr.truncate_y + 1) // 2,
        tr.truncate_x // 2 : samples.shape[-1] - (tr.truncate_x + 1) // 2,
    ]
