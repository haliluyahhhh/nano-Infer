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
