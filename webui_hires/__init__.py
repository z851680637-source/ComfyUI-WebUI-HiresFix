"""Core of the WebUI Hires Fix port.

This package is a 1:1 port of WebUI's "Hires. fix" pipeline into ComfyUI
terms. Every function that mirrors WebUI code carries a
"# Ported from stable-diffusion-webui modules/xxx.py:<line>" comment pointing at
the original implementation.
"""

__all__ = [
    "constants",
    "target",
    "latent_upscale",
    "image_upscale",
    "vae_roundtrip",
    "sampling",
]
