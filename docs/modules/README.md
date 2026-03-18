# 模块索引

| 包路径 | 职责 |
|--------|------|
| `nano_infer.entrypoints` | OpenAI HTTP/SSE |
| `nano_infer.engine` | AsyncLLMEngine、Sequence、调度与内存 |
| `nano_infer.runner` | ModelRunner、AttentionMetadata |
| `nano_infer.models` | BaseCausalLM、Registry、Llama3/Qwen3/Dummy |
| `nano_infer.kernels` | `paged_attention` 路由 |

依赖关系：`entrypoints` → `engine` → `runner` → `models` → `kernels`。
