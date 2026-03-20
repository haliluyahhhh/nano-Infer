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
| `Qwen/Qwen3-0.6B` 等 | 0.6B+ | `model_type=qwen3`，独立模块 `models/qwen3.py`（含 q_norm/k_norm） |
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

## Qwen2 与 `attention_bias`

实现位于 `nano_infer.models.qwen2`（独立模块，不继承 Llama）。

部分 Qwen2 的 `config.json` **不写** `attention_bias`，但 `model.safetensors` 里仍有 `q_proj/k_proj/v_proj` 的 **bias**。  
`build_engine` 会在**创建模型前**扫描已映射权重里是否存在 `layers.*.self_attn.q_proj.bias`，若有则自动 `attention_bias=True`，否则 QKV bias 无法加载，输出会像乱码。

离线验证：

```bash
nano-infer-run -m /path/to/Qwen2-1.5B -p "你好" -n 32 -v
export NANO_INFER_DEBUG=config,weights
nano-infer-run -m /path/to/Qwen2-1.5B -p "你好" -n 8
# 日志中应出现 attention_bias=True (from_weights=True)，且 missing=0
```

### 权重加载详细对照

```bash
export NANO_INFER_DEBUG=weights
export NANO_INFER_DEBUG_WEIGHTS_VERBOSE=1   # 列出全部 missing/unexpected、按层统计交集键数量
nano-infer-run -m /path/to/Qwen2-1.5B -p "你好" -n 1
```

日志会打印：`alignment`（模型参数数 vs checkpoint 张量数）、`SHAPE_MISMATCH`、`MISSING` / `UNEXPECTED`，以及 `OK: 全部模型参数...` 表示与 checkpoint 完全对齐。

## 采样参数（抑制重复）

`nano-infer-run` 默认已开启重复惩罚与 top-k / top-p 采样，无需额外设置。

| 参数 | CLI 选项 | 默认值 | 说明 |
|------|----------|--------|------|
| temperature | `-t` / `--temperature` | 0.7 | 采样温度（0 为贪心） |
| top_p | `--top-p` | 0.9 | Nucleus sampling 截断概率 |
| top_k | `--top-k` | 50 | 仅保留概率最高的 k 个 token（0 = 不截断） |
| repetition_penalty | `--repetition-penalty` / `--rep` | 1.2 | >1 抑制重复（HF 风格） |

示例：

```bash
nano-infer-run -m /path/to/Qwen3-4B -p "北京是哪个国家的首都？" -n 64
nano-infer-run -m /path/to/Qwen3-4B -p "北京是哪个国家的首都？" -n 64 -t 0.8 --rep 1.3 --top-k 40
nano-infer-run -m /path/to/Qwen3-4B -p "北京是哪个国家的首都？" -n 64 -t 0 --rep 1.0  # 纯贪心
```

OpenAI API 同样支持 `temperature`、`top_p`、`top_k`、`repetition_penalty` 参数。

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
