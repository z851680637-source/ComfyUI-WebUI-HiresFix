"""VAE <-> PIL <-> latent roundtrip with WebUI's 8-bit quantisation semantics.

Ported from stable-diffusion-webui modules/processing.py:1400-1423 and
modules/sd_samplers_common.py:54-59.

Notes on scaling (important for 1:1 behaviour):

* WebUI stores "scaled" latents (multiplied by 0.18215 / 0.13025 inside its VAE
  wrapper). ComfyUI's VAE.encode returns unscaled latents and KSampler applies
  model_sampling.process_latent_in itself, so this module never multiplies by a
  latent scale factor - it just calls VAE.encode / VAE.decode.
* ComfyUI's VAE.decode already returns pixels in [0, 1] (it applies
  (x + 1) / 2 and clamps), which is exactly WebUI's
  clamp((decoded + 1.0) / 2.0, 0, 1) before the 255.0 * uint8 conversion.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)


def decode_latent_batch(vae, samples: torch.Tensor, check_for_nans: bool = False) -> torch.Tensor:
    """Decode a latent batch to a BHWC float32 pixel batch in [0, 1] on CPU.

    Supports 4D image latents (B, C, H, W) and single-frame 5D latents
    (B, C, 1, H, W) such as Krea2 / Qwen-Image (Wan-style VAE). Multi-frame
    video latents (T > 1) are rejected: this node only does image hires fix.

    Ported from stable-diffusion-webui modules/processing.py:1354-1357 /
    modules/sd_samplers_common.py:54-59 (decode_latent_batch). The returned
    tensor equals WebUI's "lowres_samples" = clamp((decoded + 1) / 2).
    """
    if vae is None:
        raise ValueError(
            "WebUI Hires Fix: a VAE input is required to decode/encode images "
            "(the image-space upscale path and the final IMAGE output need it)."
        )
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
        raise ValueError(
            "WebUI Hires Fix: VAE decode expects a 4D image latent (B, C, H, W) or "
            f"single-frame 5D latent (B, C, 1, H, W), got {type(samples)!r}."
        )
    if samples.ndim == 5 and samples.shape[2] != 1:
        raise ValueError(
            "WebUI Hires Fix: only single-frame video latents (T=1) are supported; "
            f"got shape {tuple(samples.shape)}."
        )
    # WebUI decodes inside torch.no_grad() (modules/sd_samplers_common.py:57).
    # ComfyUI's execution loop also runs nodes under torch.inference_mode(), but
    # doing it explicitly keeps this helper safe when called directly (tests,
    # scripts, custom nodes that call the module).
    with torch.no_grad():
        decoded = vae.decode(samples.float())
    decoded = decoded.float()
    if decoded.ndim == 5:
        # Video VAE decoders return (B, T, H, W, C); keep the single frame.
        decoded = decoded[:, 0]
    if check_for_nans and not bool(torch.isfinite(decoded).all()):
        raise ValueError(
            "WebUI Hires Fix: the decoded image contains NaN/Inf values; "
            "the model or VAE produced invalid output."
        )
    return decoded.cpu()


def samples_to_pil_images(lowres_samples: torch.Tensor) -> list[Image.Image]:
    """Convert a BHWC [0, 1] pixel batch to RGB PIL images (WebUI 8-bit semantics).

    Ported from stable-diffusion-webui modules/processing.py:1401-1407:
    the 255.0 * value result is truncated (not rounded) to uint8.
    """
    if not isinstance(lowres_samples, torch.Tensor) or lowres_samples.ndim != 4:
        raise ValueError(
            "WebUI Hires Fix: samples_to_pil_images expects a 4D BHWC pixel batch, "
            f"got {type(lowres_samples)!r}."
        )
    images: list[Image.Image] = []
    for i, x_sample in enumerate(lowres_samples):
        # Ported from stable-diffusion-webui modules/processing.py:1405-1407
        x_sample = 255.0 * x_sample.cpu().float().numpy()
        x_sample = x_sample.astype(np.uint8)
        images.append(Image.fromarray(x_sample))
    return images


def pil_images_to_pixel_tensor(images: list[Image.Image]) -> torch.Tensor:
    """Convert RGB PIL images to a BHWC float32 pixel batch in [0, 1].

    Ported from stable-diffusion-webui modules/processing.py:1412-1417.
    """
    if not images:
        raise ValueError("WebUI Hires Fix: no images were produced by the image upscaler.")
    batch = [np.array(image).astype(np.float32) / 255.0 for image in images]
    return torch.from_numpy(np.stack(batch, axis=0))


def match_latent_layout(latent: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Make an encoded latent have the same 4D/5D layout as the reference latent.

    Image VAEs return 4D latents; Wan-style video VAEs return (B, C, 1, H, W).
    The hires fix pipeline keeps whatever layout the incoming latent had.
    """
    if reference.ndim == 5:
        if latent.ndim == 4:
            return latent.unsqueeze(2)
        if latent.ndim == 5 and latent.shape[2] == 1:
            return latent
        raise ValueError(
            "WebUI Hires Fix: encoded latent does not match the single-frame video "
            f"layout, got shape {tuple(latent.shape)}."
        )
    if latent.ndim == 5:
        if latent.shape[2] == 1:
            return latent[:, :, 0]
        raise ValueError(
            "WebUI Hires Fix: only single-frame video latents (T=1) are supported, "
            f"got encoded shape {tuple(latent.shape)}."
        )
    return latent


def encode_image_batch_webui(vae, pixels_bhwc: torch.Tensor) -> torch.Tensor:
    """Encode a BHWC [0, 1] pixel batch to an unscaled latent batch.

    Ported from stable-diffusion-webui modules/processing.py:1419-1421
    (images_tensor_to_samples with approximation="Full"). WebUI does not crop the
    pixels before encoding, while ComfyUI's VAE.encode centre-crops to a multiple
    of the spatial compression ratio by default; crop_input is therefore disabled
    for this call so non-multiple-of-8 targets keep WebUI's latent size. If the
    VAE refuses to encode without cropping, fall back to ComfyUI's default.
    """
    if vae is None:
        raise ValueError(
            "WebUI Hires Fix: a VAE input is required for the image-space upscale path."
        )
    if not isinstance(pixels_bhwc, torch.Tensor) or pixels_bhwc.ndim != 4:
        raise ValueError(
            "WebUI Hires Fix: encode expects a 4D BHWC pixel batch, "
            f"got {type(pixels_bhwc)!r}."
        )
    def _encode(pixels):
        # WebUI's images_tensor_to_samples runs the VAE under torch.no_grad().
        with torch.no_grad():
            return vae.encode(pixels).float()

    previous_crop_input = getattr(vae, "crop_input", None)
    if previous_crop_input is None:
        return _encode(pixels_bhwc)
    try:
        vae.crop_input = False
        try:
            return _encode(pixels_bhwc)
        except Exception as exc:
            LOGGER.warning(
                "WebUI Hires Fix: VAE encode failed without ComfyUI's 8px crop (%s); "
                "retrying with the default crop. Non-multiple-of-8 hires targets may "
                "differ from WebUI.",
                exc,
            )
            vae.crop_input = previous_crop_input
            return _encode(pixels_bhwc)
    finally:
        vae.crop_input = previous_crop_input
