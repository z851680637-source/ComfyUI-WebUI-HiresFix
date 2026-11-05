# NOTICE / 第三方来源说明

本插件是 AUTOMATIC1111/stable-diffusion-webui 的「高清修复（Hires. fix）」功能的 ComfyUI 迁移实现。

## 直接移植的代码

以下模块包含逐行移植自 stable-diffusion-webui（AGPL-3.0）的代码，源位置在注释中以
Ported from stable-diffusion-webui modules/xxx.py:<line> 标注：

- webui_hires/target.py        <- modules/processing.py（目标分辨率、truncate）
- webui_hires/latent_upscale.py <- modules/processing.py（latent F.interpolate）
- webui_hires/image_upscale.py <- modules/images.py、modules/upscaler.py、modules/upscaler_utils.py
  （resize_image、Upscaler.upscale、split_grid/combine_grid、8bit 往返）
- webui_hires/vae_roundtrip.py <- modules/processing.py、modules/sd_samplers_common.py
- webui_hires/sampling.py      <- modules/sd_samplers_common.py、modules/sd_samplers_kdiffusion.py
  （setup_img2img_steps、get_sigmas、sigma 切片、discard 规则）

## 运行时依赖

- ComfyUI（GPL-3.0）：节点 API、VAE、sampler、ProgressBar/latent_preview、spandrel 加载器与 model_management。
- PyTorch / NumPy / Pillow / spandrel（及其各自的许可证）。

## 许可

由于包含 stable-diffusion-webui 的移植代码，本插件整体按 AGPL-3.0 发布（见 LICENSE）。
再分发时请保留本 NOTICE、源码中的来源注释，并遵守 AGPL-3.0 与 ComfyUI（GPL-3.0）的条款。