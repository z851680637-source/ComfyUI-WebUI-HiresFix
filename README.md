# ComfyUI-WebUI-HiresFix

把 WebUI 的 **Hires. fix（高清修复）** 1:1 迁移为 ComfyUI 自定义节点。
接在产生第一次采样的 latent 的节点（KSampler / VAEEncode）之后，用 WebUI 的放大逻辑放大，
再用 WebUI 的 denoise / sigma 语义跑第二次采样。

- 测试环境：ComfyUI v0.43.0（V3 API）、Python 3.13、torch 2.13 + cu130、RTX 5070 Ti 16GB
- 不新增 pip 依赖：只用 ComfyUI 自带的 torch / numpy / Pillow / spandrel / comfy.*
- 新增节点：**WebUIHiresFix**

> 纯放大可将本文节点的 `denoising_strength` 设为 0（放大阶段结果会原样输出）。

## 功能特性

- 放大阶段完全复刻 WebUI：目标分辨率 4 分支 + truncate、6 种 latent 模式、
  None / Lanczos / Nearest、模型放大器（ESRGAN / RealESRGAN / SwinIR / 4x-UltraSharp 等）。
- 图像空间路径：模型放大器走 ComfyUI 原生浮点管线（VAE float decode → comfy tiled_scale → 单次
  Lanczos 精确尺寸 → VAE encode），避免 WebUI PIL 8bit 双重重采样带来的额外软化；
  旧 WebUI PIL 路径保留在 reference/legacy 函数中（节点默认使用更锐的浮点路径）。
- 二次采样复刻 WebUI：`derived=int(steps/min(denoise,0.999))`、`sigma_sched=sigmas[derived-steps:]`、
  WebUI 的 `discard_next_to_last_sigma` 规则、denoise≈1 时的噪声补偿。
- 支持 Hires sampler / scheduler、Hires prompt / negative prompt、Hires checkpoint 覆盖。
- 正常显示 ComfyUI 进度条，只反映采样阶段；模型放大器分块阶段不占用进度条（与 WebUI 的 UI 行为一致，WebUI 的 tile 进度只在控制台）。
- 兼容 4D image latent 与单帧 5D latent（Krea2 / Qwen-Image），自动检测 8x / 16x 的像素↔latent 压缩比（SD/FLUX/Krea2/Qwen-Image=8，MOSS-Image/Qwen-Image 2.1=16）。
- 已实测 Krea2、MOSS-Image、Qwen-Image 2.1、Anima 的 512→1024 高清修复（含图像空间放大器路径）。
- 自带 UI / API 两种格式的示例 workflow，以及 55 项单元测试 + 14 项集成测试。

## 模型兼容性

节点的硬性要求是：**4D 图像 latent，或单帧 5D latent（B,C,1,H,W）**，且能从 VAE / 模型读取像素↔latent 压缩比。

| 模型 | latent 形态 | 压缩比 | 测试结果 |
| --- | --- | --- | --- |
| SD1.5 / SDXL | 4D, 4ch | 8x | ✅ 主验证（49+6 单元、14 集成） |
| Krea2（krea2-turbo） | 5D T=1, 16ch | 8x | ✅ 512→1024（23.1s）；图像空间 None 256→512 也通过 |
| MOSS-Image | 4D, 128ch | 16x | ✅ 512→1024（14.3s） |
| Qwen-Image 2.1 | 4D, 64ch | 16x | ✅ 512→1024（32.1s latent / 图像空间；1000→992 resize 14.0s） |
| Anima（WAI-Anima） | 5D T=1, 16ch | 8x | ✅ 基础兼容；但**受训练集限制，放大效果并不好**：实测 **`upscale_by=1.2` 效果最佳**，1.5 提升有限，2.0 及以上会出现明显噪点/劣化（ComfyUI 原生 KSampler 同样） |
| Qwen-Image（原版） | 5D T=1, 16ch | 8x | 结构同 Wan21（理论支持，本机无该权重未实测） |
| 视频模型（T>1） | 5D, T>1 | - | ❌ 明确拒绝（只支持单帧图像） |

注意：

- **Krea2 必须用 `CLIPLoader` 的 `type = krea2`** 加载它的 Qwen3-VL 文本编码器；用错类型会报 `Krea2 expects conditioning with 12x2560=30720 features ...`，这不是本节点的问题。
- MOSS-Image 用 `CLIPLoader type = moss_image` + `flux2-vae`；Qwen-Image 2.1 用 `CLIPLoader type = qwen_image` + `qwen3vl_8b_bf16`（或 int8）+ `qwen_image_2.1_vae_bf16`（16x latent）。
- Anima 用 `CLIPLoader type = wan` + `qwen_3_06b_base`、`UNETLoader` 加载 Anima 模型、`VAE = wan_2.1_vae`（5D T=1 / 8x，与 Wan21 latent 相同）。
- **Anima 的放大效果受训练集限制**：不要把 Anima 当高倍率超分模型用；高清修复建议 `upscale_by = 1.2`（实测最佳），不要用 2.0 硬拉到 2048×3072。
- 特殊 conditioning（MOSS layout/color-depth control、Krea2 edit LoRA、Qwen edit 参考图）需要先按官方节点接好，再进本节点；本节点只负责放大 + 二次采样。
- **cfg 必须与第一次采样的 KSampler 一致**（Krea2-turbo 用 1.0）。实测 Krea2-turbo + LoRA，
  cfg=7、840x1256→1680x2512 时二次采样会产生 NaN；同一工作流 cfg=1 正常出图（1680x2512，139.6s）。
- cfg / sampler / scheduler 建议直接沿用官方工作流和第一次采样的设置；节点不替模型选择采样配置。

### 精度兼容性

| 组件 | 已验证精度 | 说明 |
| --- | --- | --- |
| MODEL | bf16/default、fp8_e4m3fn（Krea2 / Qwen-Image 2.1 / Anima 实测） | 插件把 MODEL 当作黑盒；fp16/fp32 由 ComfyUI 启动参数或 loader 决定。GGUF / 量化 loader 只要返回标准 MODEL 即可，插件不依赖具体 loader |
| VAE | bf16（默认）、fp32（`--fp32-vae`）、fp16（`--fp16-vae`） | Krea2/Qwen2.1/Anima 均有实测；fp16 VAE ComfyUI 官方标注可能黑图，这是 VAE 数值问题，不是插件 |
| CLIP | bf16、fp8_scaled、int8 | 只有 `hr_prompt` / `hr_negative_prompt` 非空时插件才调用 CLIP；否则直接透传第一次采样的 conditioning，精度与插件无关 |

说明：

- Qwen-Image 2.1 官方支持的推理 dtype 是 bf16/fp32；强行 fp16 可能在模型侧失败，与本节点无关。
- fp8 的 `e4m3fn_fast` / `e5m2` 与 `e4m3fn` 走同一 ComfyUI 量化路径；本节点不做额外 cast。
- `--fp16-intermediates` 只会改变节点间张量精度；插件内部对 latent 的插值/检查会沿用输入 dtype，并在 VAE 边界使用 float。

## 安装

把本目录放到：

    <ComfyUI>/custom_nodes/ComfyUI-WebUI-HiresFix

重启 ComfyUI。节点位于 **latent/upscale** 分类，搜索 `WebUI Hires Fix`、`Hires`、`高清修复`。

要求 ComfyUI 支持 V3 自定义节点 API（本机验证 v0.43.0）。无 requirements.txt，不需要 pip 安装。

命令行手动启动 ComfyUI 若遇到 `huggingface-hub` / `transformers` 版本冲突
（`ImportError: huggingface-hub>=0.34.0,<1.0`），先设置：

    $env:PYTHONNOUSERSITE='1'

再启动（启动器一般已处理好环境）。

## 节点参数

### 必需输入（参考 WebUI 的使用方式）

> 本节点接在官方工作流之后，以下必填参数的用途、取值和交互方式均参考 WebUI 的 Hires. fix。

| 参数 | WebUI 对应 | 默认 | 说明 |
| --- | --- | --- | --- |
| enable_hr | Enable Hires. fix | True | 关掉时直通第一次采样的 latent |
| model | - | 必填 | 第一次采样的模型；二次采样默认也用它 |
| positive / negative | - | 必填 | 第一次采样的条件；hr_prompt 为空时复用之 |
| latent | - | 必填 | 第一次采样 LATENT |
| vae | - | 必填 | 解码/编码；图像空间放大器与 IMAGE 输出需要 |
| upscaler | Upscaler | Latent | 6 latent 模式 + None/Lanczos/Nearest + upscale_models 文件 |
| upscale_by | Upscale by | 2.0 | 1.0-4.0，step 0.05 |
| resize_width | Resize width to | 0 | 0-2048 step 8，0 = 由 upscale_by 推导 |
| resize_height | Resize height to | 0 | 同上 |
| denoising_strength | Denoising strength | 0.7 | 0-1 step 0.01；设 0 = 只放大不采样 |
| hr_second_pass_steps | Hires steps | 0 | 0-150，0 = 使用 steps |
| steps | 第一次采样的 Steps | 20 | 二次采样步数为 0 时使用 |
| cfg | CFG Scale | 7.0 | 二次采样 CFG |
| sampler_name | Sampler | euler | 第一次采样的采样器（ComfyUI 名） |
| scheduler | Schedule type | normal | 第一次采样的调度器（ComfyUI 名） |
| hr_sampler_name | Hires sampling method | Use same sampler | 二次采样采样器 |
| hr_scheduler | Hires schedule type | Use same scheduler | 二次采样调度器 |
| hr_prompt | Hires prompt | 空 | 非空且接 clip 时重新编码目标尺寸条件 |
| hr_negative_prompt | Hires negative prompt | 空 | 同上 |
| seed | Seed | 0 | 二次采样噪声种子 |

### 可选输入

| 参数 | 对应 | 说明 |
| --- | --- | --- |
| clip | - | 用于把 hr_prompt / hr_negative_prompt 重新编码为 CONDITIONING |
| hr_model | Hires checkpoint | 接另一个 CheckpointLoader 的 MODEL，仅二次采样换模型 |

### 输出

| 输出 | 内容 |
| --- | --- |
| LATENT | 最终二次采样后的 latent |
| IMAGE | 最终解码图（BHWC，[0,1]） |
| UPSCALED_LATENT | 二次采样前的放大 latent（调试/对比用） |

## 工作原理（与 WebUI 对应）

详细逐行分解见 `docs/01-webui-hires-decomposition.md`（1192 行），迁移规格见 `SPEC.md`。

### 放大阶段

1. 目标分辨率（processing.py:1213-1250）：
   - resize 都为 0：upscale_to = int(first * upscale_by)
   - 只给一个：按第一次采样的宽高比推另一个（Python //）
   - 都给：取覆盖尺寸，并计算 truncate_x/y（latent px）= (upscale_to - resize) // 8
2. latent 模式：直接 `torch.nn.functional.interpolate`，size = floor(upscale_to/8)。
3. 图像空间：VAE float decode -> 模型放大器（comfy tiled_scale，512/32）或 comfy common_upscale
   -> 单次 Lanczos 到精确目标尺寸 -> VAE encode（为更锐的 ComfyUI 原生浮点路径；
   WebUI 的 PIL 8bit 双重重采样旧路径保留为参考实现）。
4. 模型放大器：WebUI 的 192/8 PIL 分块 + 渐变 mask 拼接 + BGR + round。
5. 统一 truncate 裁剪（processing.py:1427）。

### 二次采样阶段

    requested = hr_second_pass_steps or steps
    derived   = int(requested / min(denoising_strength, 0.999))
    t_enc     = requested - 1
    sigmas    = calculate_sigmas(derived [, +1 若 sampler 需要 discard])
    sigma_sched = sigmas[derived - requested:]
    xi = latent + noise * sigma_sched[0]
    用 sigma_sched 跑 sampler（CFG 不变）

插件按 WebUI 的 discard 集合（DPM2 / DPM2 a / DPM++ 3M SDE）自行加 1 步并丢弃倒数第二个 sigma；
denoise≈1 时做 `noise *= sigma / sqrt(1 + sigma^2)` 补偿，使初始 xi 与 WebUI 一致。

## 与 WebUI 的已知差异

| 项 | 说明 |
| --- | --- |
| 随机数源 | WebUI 本机 randn_source=GPU，ComfyUI 在 CPU 生成噪声后搬运；同 seed 不保证与 WebUI 逐像素一致，但插件内部同 seed/同 workflow 可复现 |
| SDXL aesthetic_score | WebUI 会给 SDXL 条件附加 positive 6.0 / negative 2.5；ComfyUI CLIP 编码不含该参数，插件使用 ComfyUI 默认 |
| prompt 高级特性 | WebUI 的 emphasis 语法、styles、extra networks、hires additional modules 未迁移；用 ComfyUI Conditioning/LoRA 节点表达 |
| Hires refiner 等选项 | hires_fix_use_firstpass_conds、use_old_hires_fix_width_height、save_images_before_highres_fix、hires refiner 未迁移 |
| sampler/scheduler | 使用 ComfyUI 名称；WebUI 的 DDIM/UniPC/LCM/PLMS/Restart 与 ComfyUI 实现不完全等价 |
| dpmpp_3m_sde | 插件按 WebUI 做 discard_next_to_last_sigma，与 ComfyUI 原生 KSampler 不同（有意复刻） |
| 压缩比 | WebUI 固定按 8x 换算；插件自动按 VAE 检测 8x（SD/FLUX/Krea2/Qwen-Image）或 16x（MOSS/Qwen-Image 2.1），因此非 8 倍数目标与 16x 模型的尺寸规则更准确 |
| antialias | PyTorch 的 antialias 只对下采样生效，上采样时 Latent (antialiased) 与 Latent 相同（WebUI 同行为） |
| VAE/精度 | ComfyUI 的 VAE 实现/精度路径与 WebUI 不同，可能有 1e-3 级像素差异 |
| 图像空间放大 | 旧 WebUI PIL 8bit 双重重采样路径保留为 legacy/reference；节点默认改用 ComfyUI 原生浮点单次 Lanczos 路径，细节更锐（有意优化） |

## 测试

单元/数值测试（独立 oracle，55 项）：

    $env:PYTHONNOUSERSITE='1'
    & 'E:\ComfyUI-aki-v3.2\python\python.exe' tests\run_tests.py
    # 或真实 pytest（借用本机 Anaconda 的纯 Python pytest，不安装依赖）
    & tests\run_pytest.ps1

集成测试（自动启停 8191 白名单服务，14 项）：

    $env:PYTHONNOUSERSITE='1'
    & 'E:\ComfyUI-aki-v3.2\python\python.exe' tests\run_integration.py

其他脚本（需要先启动服务）：

    python tests\check_progress.py --server http://127.0.0.1:8191   # 验证进度条事件

最近一次验证：单元 55/55、参考实现 35/35、交叉验证 8/8、集成 14/14、进度事件 8/8（8 步）。
报告见 `verify/report.md`。

## 故障排查

- **节点不出现**：ComfyUI 需重启；搜索 `WebUI` / `Hires` / `高清修复`；分类 `latent/upscale`。
- **导入失败**：查看启动日志 `Import times for custom nodes`；手动命令行启动时设置 `PYTHONNOUSERSITE=1`。
- **upscaler 下拉没有新模型**：模型列表在启动时生成，添加文件到 `ComfyUI/models/upscale_models` 或
  `extra_model_paths.yaml` 的 `upscale_models` 后重启 ComfyUI。
- **CUDA OOM**：优先用 Latent 模式，或降低第一次采样的分辨率 / upscale_by。
- **输出尺寸不是请求值**：WebUI 的压缩倍数/truncate 规则；插件会按 VAE 自动用 8 或 16 对齐。想精确控制比例请用 resize_width/height。
- **hr_prompt 不生效**：需要接 `clip` 输入。
- **换 hr_model 后报条件不匹配**：二次采样 CONDITIONING 必须和新模型/CLIP 兼容。
- **报 `expected input[1, 4, ...] to have 64 channels` / latent 通道数不匹配**：latent 接错了。
  要把第一次采样 **KSampler / VAEEncode 的 LATENT 输出**接到 WebUIHiresFix，不要接 EmptyLatentImage；
  Qwen-Image 2.1 的第一次采样的 latent 是 4D 64ch，EmptyLatentImage 只有 4ch。节点会直接报这个接线错误。
- **Anima 放大效果不好 / 高分辨率出噪点**：Anima 受训练集分辨率与分布限制，高清修复的放大效果本身有限。
  实测 **`upscale_by = 1.2` 效果最佳**；1.5 提升有限，2.0 及以上（尤其 1024×1536→2048×3072）会出现明显噪点或细节劣化。
  插件与 ComfyUI 原生 KSampler 的输出逐像素一致，这不是插件 bug。
  如果必须要 2048×3072，建议先按 1.2x 做高清修复，再用图像放大器无采样放大到目标尺寸，不要直接 2x 二次采样。
- **二次采样/解码报 NaN/Inf**：会指出是放大阶段还是采样阶段，并打印形状与 cfg/sampler/scheduler/denoise/steps。
  常见原因：VAE 与模型不匹配（Krea2/Qwen-Image 必须用其 Wan 风格 VAE）、不兼容的 LoRA/模型补丁、
  cfg 过高（Krea2-turbo 必须 cfg=1、euler、simple；实测 cfg=7 + LoRA 的高分辨率二次采样会出 NaN）、
  denoise 过高。先断开 hr_model 和 LoRA，改用 upscaler=Latent 逐项排查。

## 文件结构

    ComfyUI-WebUI-HiresFix/
      __init__.py / nodes.py          # V3 节点
      webui_hires/
        constants.py                  # latent 模式表、sampler/scheduler 映射、discard 集合
        target.py                     # 目标分辨率 + truncate
        latent_upscale.py             # latent F.interpolate
        image_upscale.py              # resize_image/Upscaler/split_grid/combine_grid/模型放大器
        vae_roundtrip.py              # VAE decode/8bit/encode
        sampling.py                   # WebUI sigma 切片 + 二次采样 + 进度回调
      workflows/                      # UI / API 示例 + 运行脚本
      docs/01-webui-hires-decomposition.md
      reference/                      # 独立参考实现（验证用）
      tests/                          # 55 单元 + 14 集成 + 进度脚本
      verify/                         # 验证报告
      SPEC.md / README.md / CHANGELOG.md / LICENSE / NOTICE.md

## 致谢与许可

- 放大与二次采样逻辑移植自 [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)（AGPL-3.0），
  源码位置在代码注释中以 `Ported from stable-diffusion-webui modules/xxx.py:<line>` 标注。
- 运行于 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（GPL-3.0）并使用其节点 API、VAE、sampler、
  ProgressBar/latent_preview、spandrel 加载器。
- 由于包含 WebUI 移植代码，本插件整体按 **AGPL-3.0** 发布，见 `LICENSE`。