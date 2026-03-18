# 使用说明

## 环境要求

- Python 3.10+
- PyTorch 2.0+（CPU 可跑 smoke；GPU 需 CUDA 版 Torch）

## 安装

```bash
cd nano-Infer
pip install -e ".[dev]"
```

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
