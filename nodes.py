"""ComfyUI V3 nodes reproducing WebUI's "Hires. fix".

Node id
-------
WebUIHiresFix
    Full hires fix: upscale the first pass latent + WebUI hires second pass.
    Outputs LATENT (final), IMAGE (final decoded) and UPSCALED_LATENT (the state
    right before the second pass).

The implementation lives in the webui_hires subpackage; this module only defines
the schema and wires the pieces together.
"""

from __future__ import annotations

import logging

import torch
from typing_extensions import override

import comfy.samplers
from comfy_api.latest import ComfyExtension, io

from .webui_hires import constants
from .webui_hires.image_upscale import upscale_latent_image_space, upscale_latent_image_space_native
from .webui_hires.latent_upscale import upscale_latent_samples
from .webui_hires.sampling import run_hires_pass
from .webui_hires.target import OPT_F, compute_target_resolution, truncate_latent
from .webui_hires.vae_roundtrip import decode_latent_batch

LOGGER = logging.getLogger(__name__)

NODE_ID_HIRES = "WebUIHiresFix"

SAMPLER_OPTIONS = [constants.USE_SAME_SAMPLER, *comfy.samplers.KSampler.SAMPLERS]
SCHEDULER_OPTIONS = [constants.USE_SAME_SCHEDULER, *comfy.samplers.KSampler.SCHEDULERS]


def _latent_tensor(latent, node_id: str):
    """Validate the LATENT input and return its latent tensor.

    Accepts 4D image latents (B, C, H, W) and single-frame 5D latents
    (B, C, 1, H, W) used by Krea2 / Qwen-Image / Wan-style VAEs. Multi-frame
    video latents and nested latents are rejected.
    """
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(f"{node_id}: latent input must be a LATENT dict containing 'samples'.")
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        raise ValueError(
            f"{node_id}: nested latents are not supported by WebUI Hires Fix; "
            "connect a 4D image latent (B, C, H, W) or single-frame 5D latent (B, C, 1, H, W)."
        )
    if not hasattr(samples, "ndim") or samples.ndim not in (4, 5):
        raise ValueError(
            f"{node_id}: WebUI Hires Fix only supports 4D image latents (B, C, H, W) "
            "or single-frame 5D latents (B, C, 1, H, W), "
            f"got shape {getattr(samples, 'shape', None)}."
        )
    if samples.ndim == 5 and samples.shape[2] != 1:
        raise ValueError(
            f"{node_id}: WebUI Hires Fix only supports single-frame video latents (T=1); "
            f"got shape {tuple(samples.shape)}. Multi-frame video is not supported."
        )
    return samples


def _detect_latent_compression(vae, model) -> int:
    """Spatial compression ratio between pixels and latents (8 or 16).

    WebUI hardcodes 8; newer ComfyUI models use 16 (Flux2 / MOSS /
    Qwen-Image 2.1). The VAE is authoritative; the model's latent format is the
    fallback for latent-only paths.
    """
    if vae is not None:
        for method_name in ("spacial_compression_decode", "spacial_compression_encode"):
            method = getattr(vae, method_name, None)
            if callable(method):
                try:
                    ratio = int(method())
                except Exception:
                    continue
                if ratio > 0:
                    return ratio
        downscale = getattr(vae, "downscale_ratio", None)
        if isinstance(downscale, int) and downscale > 0:
            return downscale
    try:
        latent_format = model.get_model_object("latent_format")
        ratio = int(getattr(latent_format, "spacial_downscale_ratio", 0))
        if ratio > 0:
            return ratio
    except Exception:
        pass
    return OPT_F


def _latent_out(latent: dict, samples, node_id: str) -> dict:
    """Rebuild a LATENT dict around the new samples, like ComfyUI's KSampler."""
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    noise_mask = out.get("noise_mask")
    if noise_mask is not None:
        try:
            matches = tuple(noise_mask.shape[-2:]) == tuple(samples.shape[-2:])
        except AttributeError:
            matches = False
        if not matches:
            out.pop("noise_mask", None)
    out["samples"] = samples
    return out


def _decode_image(vae, samples, node_id: str):
    """Decode a latent batch into a ComfyUI IMAGE tensor (BHWC, [0, 1])."""
    if vae is None:
        raise ValueError(f"{node_id}: a VAE input is required to decode the IMAGE output.")
    return decode_latent_batch(vae, samples, check_for_nans=True)


def _upscale_samples(samples, vae, tr, upscaler: str, node_id: str, compression: int = OPT_F):
    """Run the WebUI upscale-stage branch selected by the 'upscaler' combo."""
    if constants.is_latent_upscaler(upscaler):
        # Ported from stable-diffusion-webui modules/processing.py:1388-1399
        return upscale_latent_samples(samples, tr, upscaler, compression=compression)
    if constants.is_image_upscaler(upscaler):
        # Ported from stable-diffusion-webui modules/processing.py:1400-1423
        # ComfyUI-native float image path (single Lanczos, no 8-bit PIL roundtrip):
        # WebUI's PIL/8-bit two-pass resize is visibly softer, especially at 1.5x.
        return upscale_latent_image_space_native(samples, vae, tr, upscaler)
    raise ValueError(
        f"{node_id}: could not find upscaler named {upscaler!r}. "
        f"Available upscalers: {constants.list_upscalers()}"
    )


def _encode_hr_prompt(clip, text: str, target_w: int, target_h: int):
    """Re-encode a hires prompt through ComfyUI's CLIP, WebUI style.

    Same call as the core CLIPTextEncode node (nodes.py:75-79).  The target size
    is passed as SDXL size conditioning, matching WebUI's
    prompt_parser.SdConditioning(width=hr_upscale_to_x, height=hr_upscale_to_y)
    (modules/processing.py:1499-1511 and modules/sd_models_xl.py:16-29).
    """
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(
        tokens,
        add_dict={
            "width": int(target_w),
            "height": int(target_h),
            "target_width": int(target_w),
            "target_height": int(target_h),
        },
    )


class WebUIHiresFix(io.ComfyNode):
    """WebUI Hires. fix: upscale + hires second pass."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID_HIRES,
            display_name="WebUI Hires Fix (高清修复)",
            category="latent/upscale",
            description=(
                "WebUI Hires. fix ported to ComfyUI. Upscales the first pass latent "
                "with WebUI's latent modes or pixel-space upscalers (192/8 tiling, 8-bit VAE "
                "roundtrip), then runs WebUI's hires second pass (WebUI sigma schedule, "
                "denoising strength, optional hires sampler/scheduler/prompt/model). "
                "Supports 4D image latents and single-frame 5D latents (Krea2 / Qwen-Image); "
                "the pixel<->latent compression ratio (8 or 16) is auto-detected from the VAE."
            ),
            inputs=[
                io.Boolean.Input(
                    "enable_hr",
                    default=True,
                    label_on="enabled",
                    label_off="disabled",
                    tooltip="WebUI 'Enable Hires. fix'. When disabled the node is a pass-through.",
                ),
                io.Model.Input("model", tooltip="First pass diffusion model (use the one that produced the latent)."),
                io.Conditioning.Input("positive", tooltip="First pass positive conditioning (used unless hr_prompt is set)."),
                io.Conditioning.Input("negative", tooltip="First pass negative conditioning (used unless hr_negative_prompt is set)."),
                io.Latent.Input("latent", tooltip="First pass LATENT (output of a KSampler / VAEEncode)."),
                io.Vae.Input("vae", tooltip="VAE matching the first pass model; required for pixel-space upscalers and the IMAGE output."),
                io.Combo.Input(
                    "upscaler",
                    options=constants.list_upscalers(),
                    default="Latent",
                    tooltip="WebUI 'Upscaler': 6 latent modes, None/Lanczos/Nearest or an upscale model file.",
                ),
                io.Float.Input("upscale_by", default=2.0, min=1.0, max=4.0, step=0.05, display_mode=io.NumberDisplay.number,
                               tooltip="WebUI 'Upscale by'."),
                io.Int.Input("resize_width", default=0, min=0, max=2048, step=8, display_mode=io.NumberDisplay.number,
                             tooltip="WebUI 'Resize width to'; 0 = derive from upscale_by."),
                io.Int.Input("resize_height", default=0, min=0, max=2048, step=8, display_mode=io.NumberDisplay.number,
                             tooltip="WebUI 'Resize height to'; 0 = derive from upscale_by."),
                io.Float.Input("denoising_strength", default=0.7, min=0.0, max=1.0, step=0.01,
                               display_mode=io.NumberDisplay.number, tooltip="WebUI 'Denoising strength' (hires pass)."),
                io.Int.Input("hr_second_pass_steps", default=0, min=0, max=150, step=1, display_mode=io.NumberDisplay.number,
                             tooltip="WebUI 'Hires steps'; 0 = use 'steps'."),
                io.Int.Input("steps", default=20, min=1, max=10000, step=1, display_mode=io.NumberDisplay.number,
                             tooltip="First pass steps (used when hr_second_pass_steps is 0 and for 'Use same' fallbacks)."),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1, display_mode=io.NumberDisplay.number,
                               tooltip="CFG scale for the hires pass. Must match the first pass / the model (Krea2-turbo uses 1.0; high cfg on turbo/LoRA models can produce NaN)."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler",
                               tooltip="First pass sampler (used when hr_sampler_name is 'Use same sampler')."),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal",
                               tooltip="First pass scheduler (used when hr_scheduler is 'Use same scheduler')."),
                io.Combo.Input("hr_sampler_name", options=SAMPLER_OPTIONS, default=constants.USE_SAME_SAMPLER,
                               tooltip="WebUI 'Hires sampling method'."),
                io.Combo.Input("hr_scheduler", options=SCHEDULER_OPTIONS, default=constants.USE_SAME_SCHEDULER,
                               tooltip="WebUI 'Hires schedule type'."),
                io.String.Input("hr_prompt", multiline=True, default="",
                                tooltip="WebUI 'Hires prompt'; empty = use the first pass conditioning."),
                io.String.Input("hr_negative_prompt", multiline=True, default="",
                                tooltip="WebUI 'Hires negative prompt'; empty = use the first pass negative conditioning."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="Seed for the hires pass noise (WebUI uses the same seed for the hires pass)."),
                io.Clip.Input("clip", optional=True,
                              tooltip="Optional CLIP used to re-encode hr_prompt/hr_negative_prompt."),
                io.Model.Input("hr_model", optional=True,
                               tooltip="Optional second pass model (WebUI 'Hires checkpoint'); unconnected = use 'model'."),
            ],
            outputs=[
                io.Latent.Output(display_name="LATENT", tooltip="Final hires latent."),
                io.Image.Output(display_name="IMAGE", tooltip="Final decoded image."),
                io.Latent.Output(display_name="UPSCALED_LATENT", tooltip="Upscaled latent before the hires second pass."),
            ],
        )

    @classmethod
    def execute(
        cls,
        enable_hr,
        model,
        positive,
        negative,
        latent,
        vae,
        upscaler,
        upscale_by,
        resize_width,
        resize_height,
        denoising_strength,
        hr_second_pass_steps,
        steps,
        cfg,
        sampler_name,
        scheduler,
        hr_sampler_name,
        hr_scheduler,
        hr_prompt,
        hr_negative_prompt,
        seed,
        clip=None,
        hr_model=None,
    ) -> io.NodeOutput:
        samples = _latent_tensor(latent, NODE_ID_HIRES)

        expected_channels = getattr(vae, "latent_channels", None) if vae is not None else None
        if expected_channels is None and vae is not None:
            expected_channels = getattr(getattr(vae, "latent_format", None), "latent_channels", None)
        if expected_channels is not None and int(samples.shape[1]) != int(expected_channels):
            raise ValueError(
                f"{NODE_ID_HIRES}: the latent has {int(samples.shape[1])} channels but the connected VAE "
                f"expects {int(expected_channels)} channels. Connect the LATENT output of the first-pass "
                "KSampler / VAEEncode (not the EmptyLatentImage) and use the VAE that matches the model. "
                "For Qwen-Image 2.1 the first-pass latent is 4D 64-channel; EmptyLatentImage is only "
                "4-channel and cannot be decoded by the Qwen-Image 2.1 VAE."
            )

        if not enable_hr:
            # WebUI bypass: the hires pass is skipped entirely, the first pass
            # image is returned as-is.
            image = _decode_image(vae, samples, NODE_ID_HIRES)
            return io.NodeOutput(
                _latent_out(latent, samples, NODE_ID_HIRES),
                image,
                _latent_out(latent, samples, NODE_ID_HIRES),
            )

        compression = _detect_latent_compression(vae, model)
        first_w = samples.shape[-1] * compression
        first_h = samples.shape[-2] * compression
        tr = compute_target_resolution(
            first_w, first_h, upscale_by, resize_width, resize_height, compression=compression
        )

        upscaled = _upscale_samples(samples, vae, tr, upscaler, NODE_ID_HIRES, compression=compression)
        upscaled = truncate_latent(upscaled, tr)
        if not bool(torch.isfinite(upscaled).all()):
            raise ValueError(
                f"{NODE_ID_HIRES}: the upscale stage produced NaN/Inf latents "
                f"(shape {tuple(upscaled.shape)}, upscaler={upscaler!r}, compression={compression}). "
                "This usually means the VAE does not match the model, or the pixel-space upscaler "
                "produced invalid pixels. Try upscaler='Latent' or connect the same VAE used for "
                "the first pass."
            )

        # WebUI 'Use same sampler/scheduler' and hr_second_pass_steps == 0 fallbacks.
        effective_sampler = sampler_name
        if hr_sampler_name and hr_sampler_name != constants.USE_SAME_SAMPLER:
            effective_sampler = hr_sampler_name
        effective_scheduler = scheduler
        if hr_scheduler and hr_scheduler != constants.USE_SAME_SCHEDULER:
            effective_scheduler = hr_scheduler
        effective_steps = int(hr_second_pass_steps) if hr_second_pass_steps else int(steps)

        # WebUI 'Hires prompt' / 'Hires negative prompt' overrides.
        hr_positive = positive
        hr_negative = negative
        has_hr_prompt = bool(hr_prompt and hr_prompt.strip())
        has_hr_negative_prompt = bool(hr_negative_prompt and hr_negative_prompt.strip())
        if has_hr_prompt or has_hr_negative_prompt:
            if clip is None:
                LOGGER.warning(
                    "WebUI Hires Fix: hr_prompt/hr_negative_prompt are set but no CLIP input is "
                    "connected; using the first pass conditioning."
                )
            else:
                if has_hr_prompt:
                    hr_positive = _encode_hr_prompt(
                        clip, hr_prompt, tr.upscale_to_x, tr.upscale_to_y
                    )
                if has_hr_negative_prompt:
                    hr_negative = _encode_hr_prompt(
                        clip, hr_negative_prompt, tr.upscale_to_x, tr.upscale_to_y
                    )

        # WebUI 'Hires checkpoint': only the second pass uses the other model.
        second_pass_model = hr_model if hr_model is not None else model

        if float(denoising_strength) <= 0.0:
            # WebUI with denoising strength 0 has nothing to denoise.
            final_samples = upscaled
        else:
            final_samples = run_hires_pass(
                second_pass_model,
                hr_positive,
                hr_negative,
                upscaled,
                cfg=cfg,
                seed=seed,
                requested_steps=effective_steps,
                denoising_strength=denoising_strength,
                sampler_name=effective_sampler,
                scheduler=effective_scheduler,
            )

        if not bool(torch.isfinite(final_samples).all()):
            raise ValueError(
                f"{NODE_ID_HIRES}: the hires second pass produced NaN/Inf latents "
                f"(shape {tuple(final_samples.shape)}). This is a model/settings issue, not a "
                "size issue. Check: (1) the node VAE matches the model (Krea2/Qwen-Image use "
                "their Wan-style VAE); (2) no incompatible LoRA/model patch; (3) cfg / sampler / "
                "scheduler appropriate for the model (e.g. Krea2-turbo: cfg=1.0, euler, simple); "
                "(4) try upscaler='Latent' and/or a lower denoising_strength. "
                f"effective cfg={float(cfg):g}, sampler={effective_sampler!r}, "
                f"scheduler={effective_scheduler!r}, denoise={float(denoising_strength):g}, "
                f"steps={int(effective_steps)}."
            )

        final_image = _decode_image(vae, final_samples, NODE_ID_HIRES)
        return io.NodeOutput(
            _latent_out(latent, final_samples, NODE_ID_HIRES),
            final_image,
            _latent_out(latent, upscaled, NODE_ID_HIRES),
        )


class WebUIHiresFixExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WebUIHiresFix,
        ]


async def comfy_entrypoint() -> WebUIHiresFixExtension:
    return WebUIHiresFixExtension()