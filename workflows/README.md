# workflows

| 文件 | 说明 |
| --- | --- |
| webui_hires_fix_ui.json | ComfyUI 界面工作流：Checkpoint -> txt2img(512x512) -> WebUIHiresFix(Latent 2x, hires 8 steps, denoise 0.5) -> 保存 before/after |
| webui_hires_fix_api.json | 同一工作流的 API 格式（POST /prompt 用；本目录脚本可直接运行） |
| run_api_workflow.py | 运行 API 格式 workflow，自动等待并打印输出图片 |

纯放大用法：把工作流里 WebUIHiresFix 的 denoising_strength 设为 0（第二遍采样跳过，直接输出放大结果）。

## UI 工作流

ComfyUI 界面直接拖入 webui_hires_fix_ui.json 即可；默认 checkpoint 使用
waiIllustriousSDXL_v170.safetensors（随本机 WebUI 模型目录映射）。换模型后请按需调整。

## API 工作流

    $env:PYTHONNOUSERSITE='1'
    & 'E:\ComfyUI-aki-v3.2\python\python.exe' run_api_workflow.py ^
        --server http://127.0.0.1:8188 ^
        --workflow webui_hires_fix_api.json

注意：API 格式 JSON 顶层是节点映射；run_api_workflow.py 会自动包成 {"prompt": ...}。