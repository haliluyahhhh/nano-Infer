"""引擎与 Runner 统一配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SchedulerKind = Literal["fcfs", "radix"]
MemoryKind = Literal["paged", "radix"]
AttentionBackend = Literal["torch", "triton", "flashinfer"]


@dataclass
class EngineConfig:
    """全局推理配置（可由 env / YAML 扩展）。"""

    model_name: str = "dummy"
    model_path: str | None = None
    scheduler: SchedulerKind = "fcfs"
    memory: MemoryKind = "paged"
    use_radix_cache: bool = False  # True 时建议 scheduler=radix, memory=radix

    max_num_seqs: int = 8
    max_model_len: int = 4096
    max_num_batched_tokens: int = 4096
    block_size: int = 16
    num_gpu_blocks: int = 256

    attention_backend: AttentionBackend = "torch"
    device: str = "cpu"
    dtype: str = "float16"

    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12

    # 采样默认
    temperature: float = 0.0
    max_tokens: int = 64

    def effective_scheduler(self) -> SchedulerKind:
        if self.use_radix_cache:
            return "radix"
        return self.scheduler

    def effective_memory(self) -> MemoryKind:
        if self.use_radix_cache:
            return "radix"
        return self.memory


def config_from_env() -> EngineConfig:
    import os

    def _i(key: str, default: int) -> int:
        v = os.environ.get(key)
        return int(v) if v is not None else default

    sch = os.environ.get("NANO_INFER_SCHEDULER", "fcfs")
    mem = os.environ.get("NANO_INFER_MEMORY", "paged")
    backend = os.environ.get("NANO_INFER_ATTENTION_BACKEND", "torch")
    return EngineConfig(
        scheduler=sch if sch in ("fcfs", "radix") else "fcfs",  # type: ignore
        memory=mem if mem in ("paged", "radix") else "paged",  # type: ignore
        attention_backend=backend if backend in ("torch", "triton", "flashinfer") else "torch",  # type: ignore
        max_num_seqs=_i("NANO_INFER_MAX_SEQS", 8),
        device=os.environ.get("NANO_INFER_DEVICE", "cpu"),
    )
