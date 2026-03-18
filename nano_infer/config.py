"""引擎与 Runner 统一配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None  # GQA: 若为 None 则 = num_attention_heads
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # 采样默认
    temperature: float = 0.0
    max_tokens: int = 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    def effective_scheduler(self) -> SchedulerKind:
        if self.use_radix_cache:
            return "radix"
        return self.scheduler

    def effective_memory(self) -> MemoryKind:
        if self.use_radix_cache:
            return "radix"
        return self.memory

    def merge_from_model_config(self, mc: "ModelConfig") -> None:
        """从 ModelConfig 合并模型结构参数。"""
        from nano_infer.models.model_config import ModelConfig

        self.vocab_size = mc.vocab_size
        self.hidden_size = mc.hidden_size
        self.num_hidden_layers = mc.num_hidden_layers
        self.num_attention_heads = mc.num_attention_heads
        self.num_key_value_heads = mc.num_key_value_heads
        self.intermediate_size = mc.intermediate_size
        self.rms_norm_eps = mc.rms_norm_eps
        self.rope_theta = mc.rope_theta
        if mc.max_position_embeddings > 0:
            self.max_model_len = min(self.max_model_len, mc.max_position_embeddings)


def load_model_config_from_path(model_path: str | Path) -> "ModelConfig":
    from nano_infer.models.model_config import ModelConfig

    return ModelConfig.from_json(Path(model_path))


def config_from_env() -> EngineConfig:
    import os

    def _i(key: str, default: int) -> int:
        v = os.environ.get(key)
        return int(v) if v is not None else default

    sch = os.environ.get("NANO_INFER_SCHEDULER", "fcfs")
    mem = os.environ.get("NANO_INFER_MEMORY", "paged")
    backend = os.environ.get("NANO_INFER_ATTENTION_BACKEND", "torch")
    dev = os.environ.get("NANO_INFER_DEVICE", "cpu")
    # CPU 上使用 float32，避免 float16 导致的数值/算子问题
    dtype = "float32" if dev == "cpu" else os.environ.get("NANO_INFER_DTYPE", "float16")
    cfg = EngineConfig(
        scheduler=sch if sch in ("fcfs", "radix") else "fcfs",  # type: ignore
        memory=mem if mem in ("paged", "radix") else "paged",  # type: ignore
        attention_backend=backend if backend in ("torch", "triton", "flashinfer") else "torch",  # type: ignore
        max_num_seqs=_i("NANO_INFER_MAX_SEQS", 8),
        device=dev,
        dtype=dtype,
    )
    if os.environ.get("NANO_INFER_MODEL_PATH"):
        cfg.model_path = os.environ["NANO_INFER_MODEL_PATH"]
        cfg.model_name = os.environ.get("NANO_INFER_MODEL", "llama3")
    if os.environ.get("NANO_INFER_MODEL") and not cfg.model_path:
        cfg.model_name = os.environ["NANO_INFER_MODEL"]
    return cfg
