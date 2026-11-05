"""Image-space (pixel) hires upscaling, ported 1:1 from WebUI.

The whole path mirrors WebUI's implementation:

* resize / upscaler selection: modules/images.py:252-289 + modules/upscaler.py:54-150
* tiled model upscaling:       modules/upscaler_utils.py:51-88
* split / combine grid:        modules/images.py:78-141
* model inference:             modules/upscaler_utils.py:14-48
* model loading:               modules/modelloader.py:164-197 (spandrel)

The model itself is loaded through ComfyUI's bundled spandrel and scheduled with
ComfyUI's model_management; tiling and stitching use WebUI's PIL logic.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import namedtuple

import numpy as np
import torch
from PIL import Image

from .constants import (
    WEBUI_ESRGAN_TILE,
    WEBUI_ESRGAN_TILE_OVERLAP,
)
from .target import TargetResolution
from .vae_roundtrip import (
    decode_latent_batch,
    encode_image_batch_webui,
    match_latent_layout,
    pil_images_to_pixel_tensor,
    samples_to_pil_images,
)

LOGGER = logging.getLogger(__name__)

# Ported from stable-diffusion-webui modules/upscaler.py:10-11
LANCZOS = Image.Resampling.LANCZOS
NEAREST = Image.Resampling.NEAREST

WEBUI_TILE_SIZE = WEBUI_ESRGAN_TILE
WEBUI_TILE_OVERLAP = WEBUI_ESRGAN_TILE_OVERLAP


class Grid(namedtuple("_Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])):
    # Ported from stable-diffusion-webui modules/images.py:69-75

    @property
    def tile_count(self) -> int:
        return sum(len(row[2]) for row in self.tiles)


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    """Split a PIL image into an overlapping tile grid (verbatim WebUI port)."""
    # Ported from stable-diffusion-webui modules/images.py:78-111
    w, h = image.size

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid: Grid) -> Image.Image:
    """Combine a tile grid back into one PIL image with WebUI's overlap masks."""
    # Ported from stable-diffusion-webui modules/images.py:114-141
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, "L")

    mask_w = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0)
    )
    mask_h = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1)
    )

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    # Ported from stable-diffusion-webui modules/upscaler_utils.py:14-19
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # flip RGB to BGR
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
    return torch.from_numpy(img)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    # Ported from stable-diffusion-webui modules/upscaler_utils.py:22-35
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{tuple(tensor.shape)} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tuple(tensor.shape)} does not describe a CHW tensor"
    arr = tensor.float().cpu().clamp_(0, 1).numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")


def _model_param(model):
    try:
        return next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return None


def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """Upscale one PIL patch with a callable upscaler model (WebUI port)."""
    # Ported from stable-diffusion-webui modules/upscaler_utils.py:38-48
    param = _model_param(model)
    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)
        if param is not None:
            tensor = tensor.to(device=param.device, dtype=param.dtype)
        return torch_bgr_to_pil_image(model(tensor))


def upscale_with_model(
    model,
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc: str = "tiled upscale",
) -> Image.Image:
    """Tile an image, run the model patch by patch and stitch it back together."""
    # Ported from stable-diffusion-webui modules/upscaler_utils.py:51-88
    if tile_size <= 0:
        LOGGER.debug("WebUI Hires Fix: upscaling %s without tiling", img.size)
        return upscale_pil_patch(model, img)

    grid = split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []
    scale_factor = 1

    # NOTE: WebUI's tiled model upscaler only prints its tile progress to the
    # console (enable_upscale_progressbar); it does NOT drive the main UI
    # progress bar. We deliberately keep it out of ComfyUI's global progress
    # bar as well, so the only progress shown is the hires sampling pass.
    for y, h, row in grid.tiles:
        newrow = []
        for x, w, tile in row:
            output = upscale_pil_patch(model, tile)
            scale_factor = output.width // tile.width
            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = Grid(
        newtiles,
        tile_w=grid.tile_w * scale_factor,
        tile_h=grid.tile_h * scale_factor,
        image_w=grid.image_w * scale_factor,
        image_h=grid.image_h * scale_factor,
        overlap=grid.overlap * scale_factor,
    )
    return combine_grid(newgrid)


class SpandrelUpscaler:
    """A spandrel image upscaler, callable with a BCHW float tensor."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.tile = True
        descriptor = _load_spandrel_descriptor(path)
        self._descriptor = descriptor
        self.scale = float(getattr(descriptor, "scale", 1.0) or 1.0)

        import comfy.model_management as model_management
        import comfy.model_patcher

        self.patcher = comfy.model_patcher.CoreModelPatcher(
            descriptor.model,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        import comfy.model_management as model_management

        model = self._descriptor.model
        memory_required = (
            tensor.shape[-2] * tensor.shape[-1] * tensor.shape[-3] * tensor.element_size()
            * max(self.scale, 1.0) * 384.0
        )
        model_management.load_models_gpu(
            [self.patcher],
            memory_required=max(1, int(memory_required)),
            force_full_load=True,
        )
        param = _model_param(model)
        if param is not None:
            tensor = tensor.to(device=param.device, dtype=param.dtype)
        with torch.inference_mode():
            return model(tensor)


def _load_spandrel_descriptor(path: str):
    try:
        import comfy.utils
        from spandrel import ImageModelDescriptor, ModelLoader
    except Exception as exc:  # pragma: no cover - environment without ComfyUI/spandrel
        raise RuntimeError(
            f"WebUI Hires Fix: spandrel is required for model upscalers but could not be "
            f"imported ({exc})."
        ) from exc

    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
        state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"module.": ""})
    try:
        descriptor = ModelLoader().load_from_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"WebUI Hires Fix: failed to load upscale model {path!r} with spandrel: {exc}"
        ) from exc
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(
            f"WebUI Hires Fix: upscale model {path!r} is not a single-image upscale model."
        )
    descriptor.model.eval()
    LOGGER.info(
        "WebUI Hires Fix: loaded upscale model %s (%s, scale=%s)",
        path, getattr(descriptor, "architecture", None), getattr(descriptor, "scale", None),
    )
    return descriptor


def _resolve_upscale_model_path(name: str) -> str:
    try:
        import folder_paths
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"WebUI Hires Fix: folder_paths is required to resolve upscale models ({exc})."
        ) from exc
    available = list(folder_paths.get_filename_list("upscale_models"))
    if name not in available:
        raise ValueError(
            f"WebUI Hires Fix: could not find upscale model {name!r}. "
            f"Available upscale models: {available}"
        )
    return folder_paths.get_full_path_or_raise("upscale_models", name)


class ModelUpscalerRegistry:
    """Lazily loads and caches spandrel upscale models by file name."""

    def __init__(self):
        self._models: dict[str, SpandrelUpscaler] = {}
        self._lock = threading.Lock()

    def load(self, name: str) -> SpandrelUpscaler:
        with self._lock:
            cached = self._models.get(name)
            if cached is not None:
                return cached
            model = SpandrelUpscaler(name, _resolve_upscale_model_path(name))
            self._models[name] = model
            return model

    def clear(self) -> None:
        with self._lock:
            self._models.clear()


_DEFAULT_REGISTRY: ModelUpscalerRegistry | None = None
_DEFAULT_REGISTRY_LOCK = threading.Lock()


def get_default_model_registry() -> ModelUpscalerRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        with _DEFAULT_REGISTRY_LOCK:
            if _DEFAULT_REGISTRY is None:
                _DEFAULT_REGISTRY = ModelUpscalerRegistry()
    return _DEFAULT_REGISTRY


def _upscaler_do_upscale(img: Image.Image, scale: float, upscaler_name: str, registry: ModelUpscalerRegistry) -> Image.Image:
    # Ported from stable-diffusion-webui modules/upscaler.py:108-150 (do_upscale)
    if upscaler_name is None or upscaler_name == "None":
        return img
    if upscaler_name == "Lanczos":
        return img.resize((int(img.width * scale), int(img.height * scale)), resample=LANCZOS)
    if upscaler_name == "Nearest":
        return img.resize((int(img.width * scale), int(img.height * scale)), resample=NEAREST)
    model = registry.load(upscaler_name)
    tile_size = WEBUI_ESRGAN_TILE if getattr(model, "tile", True) else 0
    return upscale_with_model(model, img, tile_size=tile_size, tile_overlap=WEBUI_ESRGAN_TILE_OVERLAP)


def upscale_pil_with_webui_upscaler(
    img: Image.Image,
    scale: float,
    upscaler_name: str,
    registry: ModelUpscalerRegistry,
) -> Image.Image:
    """WebUI Upscaler.upscale: pick the scheduler loop and exact dest size."""
    # Ported from stable-diffusion-webui modules/upscaler.py:54-76
    dest_w = int((img.width * scale) // 8 * 8)
    dest_h = int((img.height * scale) // 8 * 8)

    for i in range(3):
        if img.width >= dest_w and img.height >= dest_h and (i > 0 or scale != 1):
            break

        shape = (img.width, img.height)

        img = _upscaler_do_upscale(img, scale, upscaler_name, registry)

        if shape == (img.width, img.height):
            break

    if img.width != dest_w or img.height != dest_h:
        img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

    return img


def resize_pil_with_webui_upscaler(
    image: Image.Image,
    target_w: int,
    target_h: int,
    upscaler: str,
    model_registry: ModelUpscalerRegistry | None = None,
) -> Image.Image:
    """WebUI modules/images.py resize_image(resize_mode=0, ...) ported 1:1."""
    # Ported from stable-diffusion-webui modules/images.py:252-289
    registry = model_registry if model_registry is not None else get_default_model_registry()
    upscaler_name = upscaler

    def resize(im: Image.Image, w: int, h: int) -> Image.Image:
        # Ported from stable-diffusion-webui modules/images.py:269-288
        if upscaler_name is None or upscaler_name == "None" or im.mode == "L":
            return im.resize((w, h), resample=LANCZOS)

        scale = max(w / im.width, h / im.height)

        if scale > 1.0:
            im = upscale_pil_with_webui_upscaler(im, scale, upscaler_name, registry)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    return resize(image, int(target_w), int(target_h))


def resize_pil_precise(
    image: Image.Image,
    target_w: int,
    target_h: int,
    upscaler: str,
    model_registry: ModelUpscalerRegistry | None = None,
) -> Image.Image:
    """Resize with the upscaler model but only ONE final Lanczos resample.

    WebUI's images.resize_image first resizes the model output to an 8-aligned
    dest size and then resizes again to the exact requested size (two Lanczos
    passes for non-multiple-of-8 targets), which visibly softens the result.
    This variant skips the intermediate 8-aligned resize and goes straight from
    the model output to the exact target, matching ComfyUI's native image path.
    """
    registry = model_registry if model_registry is not None else get_default_model_registry()
    target_w, target_h = int(target_w), int(target_h)
    if upscaler is None or upscaler == "None":
        return image.resize((target_w, target_h), resample=LANCZOS)

    scale = max(target_w / image.width, target_h / image.height)
    if scale > 1.0:
        for _ in range(3):
            if image.width >= target_w and image.height >= target_h:
                break
            shape = (image.width, image.height)
            image = _upscaler_do_upscale(image, scale, upscaler, registry)
            if shape == (image.width, image.height):
                break

    if image.width != target_w or image.height != target_h:
        image = image.resize((target_w, target_h), resample=LANCZOS)
    return image


def upscale_latent_image_space_native(
    samples: torch.Tensor,
    vae,
    tr: TargetResolution,
    upscaler: str,
    model_registry: ModelUpscalerRegistry | None = None,
) -> torch.Tensor:
    """Image-space upscale through ComfyUI's native float pipeline.

    This is the sharp path: VAE decode (float, no 8-bit quantisation) ->
    model upscaler with comfy tiled_scale (or comfy common_upscale for
    None/Lanczos/Nearest) -> single Lanczos to the exact target size -> VAE
    encode. It matches ComfyUI's ImageUpscaleWithModel + ImageScale + VAEEncode
    chain and avoids WebUI's PIL 8-bit double-resample softness.
    """
    import comfy.model_management
    import comfy.utils

    if samples.ndim == 5 and samples.shape[2] != 1:
        raise ValueError(
            "WebUI Hires Fix: only single-frame video latents (T=1) are supported; "
            f"got shape {tuple(samples.shape)}."
        )
    if vae is None:
        raise ValueError(
            f"WebUI Hires Fix: upscaler {upscaler!r} works in image space and therefore "
            "requires a VAE input (connect the VAE that matches the first pass model)."
        )

    pixels = decode_latent_batch(vae, samples, check_for_nans=True)[..., :3].float()
    target_w = max(1, int(tr.upscale_to_x))
    target_h = max(1, int(tr.upscale_to_y))

    if upscaler is None or upscaler == "None":
        method = "lanczos"
        pixels = comfy.utils.common_upscale(
            pixels.movedim(-1, -3), target_w, target_h, method, "disabled"
        ).movedim(-3, -1)
    elif upscaler in ("Lanczos", "Nearest"):
        method = "lanczos" if upscaler == "Lanczos" else "nearest-exact"
        pixels = comfy.utils.common_upscale(
            pixels.movedim(-1, -3), target_w, target_h, method, "disabled"
        ).movedim(-3, -1)
    else:
        registry = model_registry if model_registry is not None else get_default_model_registry()
        model = registry.load(upscaler)
        memory_required = (512 * 512 * 3) * 4 * max(model.scale, 1.0) * 384.0 + pixels.numel() * 4
        comfy.model_management.load_models_gpu(
            [model.patcher], memory_required=int(memory_required), force_full_load=True
        )
        in_img = pixels.movedim(-1, -3).to(model.patcher.load_device)
        output = comfy.utils.tiled_scale(
            in_img,
            lambda tile: model(tile.float()),
            tile_x=512,
            tile_y=512,
            overlap=32,
            upscale_amount=model.scale,
            output_device=comfy.model_management.intermediate_device(),
        )
        output = torch.clamp(output.movedim(-3, -1), 0.0, 1.0).float()
        if output.shape[1] != target_h or output.shape[2] != target_w:
            output = comfy.utils.common_upscale(
                output.movedim(-1, -3), target_w, target_h, "lanczos", "disabled"
            ).movedim(-3, -1)
        pixels = output

    with torch.no_grad():
        latent = vae.encode(pixels.cpu()).float()
    return match_latent_layout(latent, samples)


def upscale_latent_image_space(
    samples: torch.Tensor,
    vae,
    tr: TargetResolution,
    upscaler: str,
    device=None,
    precise_resize: bool = False,
) -> torch.Tensor:
    """Upscale a latent batch through the pixel space, WebUI style.

    Ported from stable-diffusion-webui modules/processing.py:1400-1423:
    decode -> clamp/8-bit PIL -> resize/upscale -> [0, 1] tensor -> VAE encode.

    device is accepted for API compatibility; the VAE and the upscaler model own
    their execution device.
    """
    # Ported from stable-diffusion-webui modules/processing.py:1400-1423
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
        raise ValueError(
            "WebUI Hires Fix: the image-space upscale path requires a 4D image latent "
            "(B, C, H, W) or single-frame 5D latent (B, C, 1, H, W), "
            f"got shape {getattr(samples, 'shape', None)}."
        )
    if samples.ndim == 5 and samples.shape[2] != 1:
        raise ValueError(
            "WebUI Hires Fix: only single-frame video latents (T=1) are supported; "
            f"got shape {tuple(samples.shape)}."
        )
    if vae is None:
        raise ValueError(
            f"WebUI Hires Fix: upscaler {upscaler!r} works in image space and therefore "
            "requires a VAE input (connect the VAE that matches the first pass model)."
        )

    decoded_samples = decode_latent_batch(vae, samples, check_for_nans=True)
    # Ported from stable-diffusion-webui modules/processing.py:1401-1414
    lowres_images = samples_to_pil_images(decoded_samples)
    if precise_resize:
        batch_images = [
            resize_pil_precise(image, tr.upscale_to_x, tr.upscale_to_y, upscaler)
            for image in lowres_images
        ]
    else:
        batch_images = [
            resize_pil_with_webui_upscaler(image, tr.upscale_to_x, tr.upscale_to_y, upscaler)
            for image in lowres_images
        ]
    pixels = pil_images_to_pixel_tensor(batch_images)
    latent = encode_image_batch_webui(vae, pixels)
    return match_latent_layout(latent, samples)
