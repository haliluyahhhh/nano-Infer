"""Qwen3 占位 — 与 Llama3 同构时可复用层堆叠逻辑。"""

from __future__ import annotations

from nano_infer.config import EngineConfig
from nano_infer.models.llama3 import Llama3ForCausalLM
from nano_infer.models.registry import register_model


@register_model("qwen3")
class Qwen3ForCausalLM(Llama3ForCausalLM):
    """后续替换 config 与权重映射。"""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
