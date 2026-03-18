# 使用说明

## 环境要求

- Python 3.10+
- PyTorch 2.0+（CPU 可跑 smoke；GPU 需 CUDA 版 Torch）

## 安装

```bash
cd nano-Infer
pip install -e ".[dev,tokenizer]"
```

## 下载模型（ModelScope，≤7B）

```bash
pip install -e ".[download]"
python scripts/download_model.py --local-dir ~/autodl-tmp/models/Qwen2-1.5B
```

支持 `--model` 指定模型 ID（默认 `Qwen/Qwen2-1.5B-Instruct`），`--local-dir` 指定本地目录。

| 模型 ID | 参数量 | 说明 |
|---------|--------|------|
| `Qwen/Qwen2-1.5B-Instruct` | 1.5B | 默认，下载快 |
| `Qwen/Qwen2-7B-Instruct` | 7B | 效果更好，显存需求高 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | 轻量 |

## 快速启动 API（占位模型）

```bash
python -m nano_infer.entrypoints.openai_api
```

默认 `http://127.0.0.1:8000`，文档 `http://127.0.0.1:8000/docs`。

## 配置（环境变量示例）

| 变量 | 说明 | 默认 |
|------|------|------|
| `NANO_INFER_HOST` | 绑定地址 | `127.0.0.1` |
| `NANO_INFER_PORT` | 端口 | `8000` |
| `NANO_INFER_SCHEDULER` | `fcfs` / `radix` | `fcfs` |
| `NANO_INFER_ATTENTION_BACKEND` | `torch` / `triton` / `flashinfer` | `torch` |

## 运行测试

```bash
pytest tests/ -q
```

## 以库方式使用

```python
from nano_infer.config import EngineConfig
from nano_infer.engine import AsyncLLMEngine
# 详见 nano_infer.engine.async_llm_engine
```
