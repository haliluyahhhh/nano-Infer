"""
HuggingFace config.json 解析 → ModelConfig。
供 EngineConfig 与模型初始化使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """从 HuggingFace config.json 解析的模型结构参数。"""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # HF Qwen3 等可能在 config 中显式给出 head_dim（与 hidden/num_heads 推导一致或用于校验）
    head_dim_explicit: int | None = None

    @property
    def head_dim(self) -> int:
        if self.head_dim_explicit is not None:
            return self.head_dim_explicit
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_head_dim(self) -> int:
        """K/V 每个头的维度（GQA 时可能与 head_dim 不同）。"""
        return self.hidden_size // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """从 config.json 或目录路径加载。"""
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        if not p.exists():
            raise FileNotFoundError(f"config.json not found: {p}")

        with open(p, encoding="utf-8") as f:
            raw = json.load(f)

        hd = raw.get("head_dim")
        head_dim_explicit = int(hd) if hd is not None else None

        # 兼容不同 HF 变体字段名
        return cls(
            vocab_size=raw.get("vocab_size", 32000),
            hidden_size=raw.get("hidden_size", 4096),
            intermediate_size=raw.get("intermediate_size", 11008),
            num_hidden_layers=raw.get("num_hidden_layers", 32),
            num_attention_heads=raw.get("num_attention_heads", 32),
            num_key_value_heads=raw.get("num_key_value_heads", raw.get("num_attention_heads", 32)),
            max_position_embeddings=raw.get("max_position_embeddings", 4096),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            attention_bias=bool(raw.get("attention_bias", raw.get("use_bias", False))),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            head_dim_explicit=head_dim_explicit,
        )


def detect_model_type(model_path: str | Path) -> str:
    """
    从 config.json 的 model_type 推断 nano-Infer 模型名。
    返回 "qwen3" | "qwen2" | "llama3" | "dummy"。
    """
    p = Path(model_path)
    if p.is_dir():
        p = p / "config.json"
    if not p.exists():
        return "llama3"
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    mt = raw.get("model_type", "").lower()
    arch = raw.get("architectures", [])
    arch_str = " ".join(str(a) for a in arch).lower()
    if "qwen3" in mt or "qwen3" in arch_str:
        return "qwen3"
    if "qwen2" in mt or "qwen2" in arch_str:
        return "qwen2"
    return "llama3"
