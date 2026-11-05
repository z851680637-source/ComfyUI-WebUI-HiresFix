"""WebUI hires second pass: sigma slicing and the sampler call.

Ported from stable-diffusion-webui modules/sd_samplers_common.py:22-31,
modules/sd_samplers_kdiffusion.py:79-188 and modules/processing.py:1427-1454.
"""

from __future__ import annotations

import logging
import math

import torch

from .constants import (
    DISCARD_SIGMA_SAMPLERS,
    to_comfy_sampler_name,
    to_comfy_scheduler_name,
)

LOGGER = logging.getLogger(__name__)

FALLBACK_SCHEDULER = "normal"


def webui_hires_sigmas(
    model_sampling,
    sampler_name: str,
    scheduler: str,
    requested_steps: int,
    denoising_strength: float,
) -> torch.Tensor:
    """Compute WebUI's hires sigma schedule (the exact slice passed to the sampler).

    Mirrors setup_img2img_steps() + KDiffusionSampler.get_sigmas() +
    sample_img2img()'s sigma_sched slicing:

        requested = steps
        derived   = int(requested / min(denoising_strength, 0.999))
        t_enc     = requested - 1
        sigmas    = get_sigmas(derived)   # +1 step and discard for DPM2/DPM2a/3M SDE
        sigma_sched = sigmas[derived - t_enc - 1:]

    Returns a CPU float32 tensor. For denoising_strength <= 0 an empty tensor is
    returned because WebUI has no steps to run in that case.
    """
    # Ported from stable-diffusion-webui modules/sd_samplers_common.py:22-31
    # Ported from stable-diffusion-webui modules/sd_samplers_kdiffusion.py:79-138
    import comfy.samplers

    requested = max(1, int(requested_steps))
    denoise = float(denoising_strength)
    if denoise <= 0.0:
        return torch.zeros(0, dtype=torch.float32)

    derived = int(requested / min(denoise, 0.999))
    t_enc = requested - 1

    comfy_sampler = to_comfy_sampler_name(sampler_name)
    if comfy_sampler is None:
        comfy_sampler = sampler_name if isinstance(sampler_name, str) else ""
    comfy_scheduler = to_comfy_scheduler_name(scheduler) or FALLBACK_SCHEDULER

    steps = derived
    discard_next_to_last_sigma = comfy_sampler in DISCARD_SIGMA_SAMPLERS
    if discard_next_to_last_sigma:
        steps += 1

    # Ported from stable-diffusion-webui modules/sd_samplers_kdiffusion.py:92-132
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, comfy_scheduler, steps)
    if discard_next_to_last_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

    # Ported from stable-diffusion-webui modules/sd_samplers_kdiffusion.py:137-138
    sigma_sched = sigmas[derived - t_enc - 1:]
    return sigma_sched.float().cpu()


def webui_max_denoise(model_sampling, sigma: float) -> bool:
    """True when ComfyUI will apply its max-denoise noise scaling for this sigma."""
    sigma_max = float(model_sampling.sigma_max)
    return math.isclose(sigma_max, float(sigma), rel_tol=1e-05) or float(sigma) > sigma_max


def run_hires_pass(
    model,
    positive,
    negative,
    latent_samples: torch.Tensor,
    *,
    cfg: float,
    seed: int,
    requested_steps: int,
    denoising_strength: float,
    sampler_name: str,
    scheduler: str,
    device=None,
) -> torch.Tensor:
    """Run WebUI's hires img2img pass on an already-upscaled latent batch.

    WebUI does:

        sigma_sched = <webui_hires_sigmas>
        xi = x + noise * sigma_sched[0]          # WebUI/k-diffusion
        samples = sampler(xi, sigma_sched, cond, uncond, cfg)

    ComfyUI's KSampler adds noise as noise * sqrt(1 + sigma^2) when
    sigma_sched[0] >= sigma_max (denoise ~ 1), so in that case the noise is
    pre-scaled by sigma / sqrt(1 + sigma^2): the initial latent then matches
    WebUI exactly for deterministic samplers.
    """
    # Ported from stable-diffusion-webui modules/sd_samplers_kdiffusion.py:134-188
    if not isinstance(latent_samples, torch.Tensor) or latent_samples.ndim not in (4, 5):
        raise ValueError(
            "WebUI Hires Fix: the hires pass requires a 4D image latent (B, C, H, W) "
            "or single-frame 5D latent (B, C, 1, H, W), "
            f"got shape {getattr(latent_samples, 'shape', None)}."
        )
    if latent_samples.ndim == 5 and latent_samples.shape[2] != 1:
        raise ValueError(
            "WebUI Hires Fix: only single-frame video latents (T=1) are supported; "
            f"got shape {tuple(latent_samples.shape)}."
        )
    denoise = float(denoising_strength)
    if denoise <= 0.0:
        return latent_samples

    import comfy.samplers
    import comfy.sample

    model_sampling = model.get_model_object("model_sampling")
    sigmas = webui_hires_sigmas(
        model_sampling,
        sampler_name,
        scheduler,
        requested_steps,
        denoise,
    )
    if sigmas.numel() == 0:
        return latent_samples

    latent_image = latent_samples.float()
    noise = comfy.sample.prepare_noise(latent_image, seed)

    # Ported from stable-diffusion-webui modules/sd_samplers_kdiffusion.py:140-143
    # (xi = x + noise * sigma_sched[0]) with ComfyUI's noise_scaling compensation.
    sigma0 = float(sigmas[0])
    if webui_max_denoise(model_sampling, sigma0):
        noise = noise * (sigma0 / math.sqrt(1.0 + sigma0 * sigma0))

    comfy_sampler = to_comfy_sampler_name(sampler_name)
    if comfy_sampler is None:
        comfy_sampler = sampler_name if isinstance(sampler_name, str) else "euler"
    comfy_scheduler = to_comfy_scheduler_name(scheduler) or FALLBACK_SCHEDULER

    sampler = comfy.samplers.KSampler(
        model,
        steps=max(1, int(requested_steps)),
        device=device if device is not None else model.load_device,
        sampler=comfy_sampler,
        scheduler=comfy_scheduler,
        denoise=None,
        model_options=model.model_options,
    )
    # ComfyUI progress bar + live previews: the core KSampler registers a
    # latent_preview callback (which owns the ProgressBar) and forwards it to the
    # sampler. Without it the frontend shows no progress while this node runs.
    callback = None
    try:
        import latent_preview

        callback = latent_preview.prepare_callback(model, max(1, int(requested_steps)))
    except Exception as exc:  # pragma: no cover - defensive (older ComfyUI builds)
        LOGGER.warning("WebUI Hires Fix: could not create the progress callback: %s", exc)

    dis = not _progress_bar_enabled()
    samples = sampler.sample(
        noise,
        positive,
        negative,
        cfg=float(cfg),
        latent_image=latent_image,
        sigmas=sigmas,
        seed=int(seed),
        callback=callback,
        disable_pbar=dis,
    )
    return samples.float()


def _progress_bar_enabled() -> bool:
    try:
        import comfy.utils
        return bool(comfy.utils.PROGRESS_BAR_ENABLED)
    except Exception:  # pragma: no cover
        return False
