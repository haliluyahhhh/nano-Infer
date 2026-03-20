# 开发日志

## 2025-03-18 初始骨架与架构定稿

**类型**：feat + docs  
**范围**：全仓库

### 做了什么
- 补充 `docs/architecture.md`：Sequence 生命周期、`SchedulerOutput`/`AttentionMetadata` 字段约定、配置与演进路线。
- 实现 `nano_infer` 包：Config、Sequence、FCFS 调度、Paged BlockManager、ModelRunner、BaseCausalLM、Kernel 路由、Dummy 模型、OpenAI 兼容 FastAPI 入口。
- Radix 调度/内存为可替换占位，接口与文档对齐。

### 原因 / 背景
用户需要可扩展推理引擎设计与首版可运行代码。

### 验证
- 本地执行：`pip install -e ".[dev]"` 后 `pytest tests/ -q`

---

## 2025-03-18 修复 Paged Attention 维度错误 + 离线推理 CLI

**类型**：fix + feat  
**范围**：`kernels/interfaces.py`、`entrypoints/offline_run.py`

### 问题
- API 报错：`tensor a (12) must match tensor b (4) at non-singleton dimension 2`
- 根因：`Q_seq @ K.transpose(-2,-1)` 在 Q[n_q,n_heads,d]、K[seq_len,n_heads,d] 下，当 n_q==seq_len 时得到 [n_q,n_heads,n_heads] 而非 [n_q,n_heads,seq_len]，导致 attn 最后一维是 n_heads(12)，mask 是 seq_len(4)，广播失败。

### 修改
1. **interfaces.py**：用 `torch.einsum("qhd,khd->qhk", Q_seq, K)` 替代错误 matmul，正确得到 [n_q, n_heads, seq_len]。
2. **offline_run.py**：新增离线推理 CLI `nano-infer-run`，不启动 API，便于 `pdb` 调试。用法：
   ```bash
   nano-infer-run -m /path/to/model -p "你好" -n 32 -v
   python -m pdb -m nano_infer.entrypoints.offline_run -m /path/to/model -p "你好"
   ```

---

## 2025-03-18 统一分模块调试系统

**类型**：feat  
**范围**：全仓库（新增 `debug.py`，插桩 8 个模块）

### 做了什么
新增 `nano_infer/debug.py` 统一调试系统，通过环境变量 `NANO_INFER_DEBUG` 控制输出，支持 9 个模块标签：

| 标签 | 模块 | 输出内容 |
|------|------|----------|
| `tokenizer` | tokenizer.py | 编码/解码的输入输出、tokenizer 类型 |
| `config` | config.py / build_engine | 模型配置合并、类型识别、device/dtype |
| `weights` | weight_loader.py / llama3.py | 键映射统计、missing/unexpected/skipped |
| `model` | llama3.py | embedding/layer[0]/lm_head 的 shape+数值摘要 |
| `attention` | interfaces.py | Q/K/V shape、causal mask、attn_weights 数值 |
| `runner` | model_runner.py | input_ids/positions、metadata、kv_cache 分配 |
| `scheduler` | vllm_scheduler.py | PREFILL/DECODE 批次决策 |
| `memory` | block_manager.py | 块分配/释放、剩余空闲数 |
| `engine` | async_llm_engine.py | step 阶段、采样 token、top-5 logits |

用法：
```bash
export NANO_INFER_DEBUG=all                    # 全部
export NANO_INFER_DEBUG=weights,attention      # 仅特定模块
nano-infer-run -m /path/to/model -p "你好" -n 32 -v
```

同时移除了旧的 `NANO_INFER_DEBUG_SHAPES` / `NANO_INFER_DEBUG_WEIGHTS` ad-hoc 检查，统一到新系统。

---

## 2025-03-18 Qwen2：从权重推断 attention_bias

**类型**：fix  
**范围**：`engine/async_llm_engine.py`、`docs/usage.md`

### 问题
- Qwen2 官方 `config.json` 常省略 `attention_bias`，解析结果为 `False`，但 checkpoint 含 QKV bias（338 张量可证）。
- 模型以 `Linear(..., bias=False)` 构建，`load_state_dict` 丢弃全部 bias，输出仍像乱码。

### 修改
- `build_engine`：先 `load_hf_weights`，若存在 `layers.*.self_attn.q_proj.bias` 则设 `cfg.attention_bias=True`，再实例化模型并 `load_weights`。

---
