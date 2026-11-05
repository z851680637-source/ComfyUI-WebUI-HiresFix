# Changelog

## 1.0

- 将 WebUI 的 Hires. fix（高清修复）以 ComfyUI 自定义节点形式迁移：新增节点 WebUIHiresFix，接在第一次采样的 latent 后。
- 放大阶段复刻 WebUI：目标分辨率 4 分支 + truncate、6 种 latent 模式、None / Lanczos / Nearest、
  ESRGAN / RealESRGAN / SwinIR / 4x-UltraSharp 等模型放大器；图像空间使用 ComfyUI 原生浮点管线
  （VAE float decode → comfy tiled_scale → 单次 Lanczos → VAE encode）。
- 二次采样复刻 WebUI 的 denoise / sigma 语义、discard_next_to_last_sigma 规则与 denoise≈1 噪声补偿。
- 支持 4D image latent 与单帧 5D latent；自动检测 8x / 16x 像素↔latent 压缩比，兼容 Krea2、MOSS-Image、
  Qwen-Image 2.1、Anima 等模型。
- 支持 Hires sampler / scheduler、Hires prompt / negative prompt、Hires checkpoint 覆盖。
- 内置 NaN/Inf 与 latent 通道数校验，错误信息指出具体阶段/接线问题。
- 55 项单元测试 + 14 项集成测试。
