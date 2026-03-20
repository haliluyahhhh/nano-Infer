"""
Qwen2 模型 — 与 Llama 同构，支持 Qwen2-1.5B / 7B 等权重加载。
"""

from __future__ import annotations

from nano_infer.config import EngineConfig
from nano_infer.models.llama3 import Llama3ForCausalLM
from nano_infer.models.registry import register_model


@register_model("qwen2")
class Qwen2ForCausalLM(Llama3ForCausalLM):
    """
    Qwen2 与 Llama 结构一致，复用 Llama3 网络定义。
    权重加载、tie_word_embeddings、详细调试报告一律使用父类 Llama3ForCausalLM.load_weights。
    切勿在此重复实现 load_weights —— 否则会跳过 tie lm_head 与 log_weight_load_report。
    """

    pass
