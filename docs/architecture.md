# nano-Infer 系统架构（定稿 v0.1）

## 1. 目标与边界

| 解决 | 不解决（当前阶段） |
|------|-------------------|
| 统一 OpenAI 风格 API，内部可切换调度/内存范式 | 多机分布式、流水线并行 |
| 白盒核心类型（`Sequence`、`SchedulerOutput`、`AttentionMetadata`） | 与 HuggingFace 推理 API 100% 行为对齐 |
| 模型与 Kernel 解耦，便于换算子 | 生产级量化/投机解码（可后续插件化） |

## 2. 上下文与依赖

```
Client → HTTP(SSE) → Frontend → AsyncLLMEngine
                              → Scheduler + Memory
                              → ModelRunner → BaseCausalLM → kernels.*
```

- **运行时**：Python 3.10+，PyTorch（CUDA 可选）。
- **可选**：`transformers`（仅 Tokenizer/权重名映射）、FlashInfer/Triton（高性能 Attention）。

## 3. 逻辑视图（五层）

| 层 | 职责 | 扩展点 |
|----|------|--------|
| Frontend | OpenAI 协议、SSE、请求→`Sequence` | 新增路由/gRPC |
| Engine | 事件循环、请求队列、选 Scheduler/Memory | `SchedulerBase`、`MemoryManagerBase` |
| Runner | `SchedulerOutput`→`AttentionMetadata`、KV Pool、CUDA Graph（预留） | `prepare_*` 策略 |
| Model Registry | `nn.Module`、权重加载 | 新模型类 + `registry.register` |
| Kernels | `paged_attention` 等统一入口 | `backend` / 环境变量路由 |

## 4. 请求生命周期（Sequence）

```
WAITING → PREFILL(可分段) → DECODE → FINISHED
```

- **WAITING**：在引擎队列中，尚未分配 KV 块。
- **PREFILL**：消费 prompt token，写入 KV；可连续批处理多条 prefill（MVP 可先整段 prefill）。
- **DECODE**：每步 1 token，直到 EOS 或 `max_tokens`。
- **FINISHED**：释放块（由 Memory 管理），可从活跃批次移除。

## 5. 核心数据结构（白盒约定）

### 5.1 `Sequence`

- `seq_id`, `prompt_ids`, `output_ids`, `status`
- `num_prompt_tokens`, `num_computed_tokens`（已写入 KV 的 token 数）
- `block_table: List[int]`：逻辑→物理块映射（Paged 范式）
- `max_tokens`, `temperature` 等采样元数据（Runner 可不读，由 Engine 后处理）

### 5.2 `SchedulerOutput`（调度 → Runner）

| 字段 | 含义 |
|------|------|
| `sequences` | 本步参与的 `Sequence` 列表（有序） |
| `num_tokens_per_seq` | 每序列本步提交的 token 数（prefill 多 token / decode 为 1） |
| `is_prefill` | 是否以 prefill 为主（便于 Graph 分桶，可逐 seq 细化） |

Runner 据此展平 `input_ids`、`positions`，并计算 `slot_mapping`、`block_tables`。

### 5.3 `AttentionMetadata`（Runner → Model）

与 PagedAttention 常见实现对齐，便于对接 FlashInfer/Triton：

- `seq_lens` / `context_lens`：本步前各 seq 的 KV 长度
- `max_seqlen_q`, `max_seqlen_k`：本步 Q/K 侧最大长度
- `block_tables`：`[num_seqs, max_blocks]`，物理块号
- `slot_mapping`：展平后每个新写入 token 在 KV pool 中的 slot
- `num_seqs`, `total_tokens`：批形状辅助

*具体张量 dtype/device 由 `ModelRunner` 与 `kv_cache` layout 约定一致。*

## 6. 范式 A：FCFS + Paged Memory（vLLM 风格）

- 调度：先到先得，容量内组 batch；decode 与 prefill 可同批（MVP 可分步简化）。
- 内存：`BlockManager` 维护空闲块链表；每 seq 按 token 数申请/释放块。

## 7. 范式 B：Radix + 前缀缓存（SGLang 风格）

- **调度**：`RadixScheduler` 在树上前缀匹配，命中则共享块、仅 prefill 未命中后缀。
- **内存**：`RadixTree` 节点持有块引用与引用计数；淘汰策略（LRU/深度优先）可插拔。
- **对用户透明**：由配置 `use_radix_cache=True` 切换，API 不变。

*当前仓库：接口与占位实现齐全，Radix 完整策略可迭代填充。*

## 8. 配置模型（建议）

统一 `EngineConfig`（或 YAML/env）：

- `model_name` / `model_path`
- `scheduler: "fcfs" | "radix"`
- `memory: "paged" | "radix"`
- `max_num_seqs`, `max_model_len`, `block_size`, `num_gpu_blocks`
- `attention_backend: "torch" | "triton" | "flashinfer"`

## 9. ModelRunner 执行步骤（不变式）

1. `scheduler.schedule()` → `SchedulerOutput`
2. `prepare_inputs` → `input_ids`, `positions`
3. `prepare_metadata` → `AttentionMetadata`
4. `model(..., kv_cache, attn_metadata)` → `logits`
5. Engine 采样、追加 token、更新 `num_computed_tokens`、判断是否结束

## 10. 模块文档索引

- 用法与命令见 [usage.md](./usage.md)。
- 代码包内各子包 docstring 与类型注解为**第二真相来源**（与本文冲突时以版本化文档为准）。

## 11. 演进路线

1. **MVP**：FCFS + Paged + Torch 参考 Attention + Dummy/小模型打通 API。
2. **性能**：接入 Triton/FlashInfer，`block_tables` 零拷贝。
3. **Radix**：完整树更新与驱逐，与 Scheduler 深度集成。
