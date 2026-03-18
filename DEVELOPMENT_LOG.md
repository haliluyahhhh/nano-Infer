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
