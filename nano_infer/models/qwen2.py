"""
Qwen2 模型 — 与 Llama 同构，支持 Qwen2-1.5B / 7B 等权重加载。
"""

from __future__ import annotations

from typing import Any, Dict

from nano_infer.config import EngineConfig
from nano_infer.models.llama3 import Llama3ForCausalLM
from nano_infer.models.registry import register_model


@register_model("qwen2")
class Qwen2ForCausalLM(Llama3ForCausalLM):
    """
    Qwen2 与 Llama 结构一致，复用 Llama3 网络定义。
    权重键与 HF Llama 相同，共用 map_hf_llama_to_nano 映射。
    """

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        from nano_infer.models.weight_loader import map_hf_llama_to_nano

        # Qwen2 权重键在 nano-Infer 内部与 Llama 同构。
        # 若 state_dict 已是 nano 键，直接加载；否则做一次 HF->nano 映射。
        if any(k.startswith("layers.") or k.startswith("embed_tokens.") for k in state_dict.keys()):
            mapped = state_dict
        else:
            mapped = map_hf_llama_to_nano(state_dict)
        self.load_state_dict(mapped, strict=False)
