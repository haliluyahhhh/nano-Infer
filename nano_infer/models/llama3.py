"""Llama3 风格结构占位 — 接 Kernel 时可替换为真实层。"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from nano_infer.config import EngineConfig
from nano_infer.kernels.interfaces import paged_attention
from nano_infer.models.base import BaseCausalLM
from nano_infer.models.registry import register_model


@register_model("llama3")
class Llama3ForCausalLM(BaseCausalLM):
    """
    精简骨架：Embedding + 单层示意 + LM head。
    真实推理应堆叠 DecoderLayer 并在层内调用 paged_attention。
    """

    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_cache, attn_metadata):
        del positions, kv_cache
        x = self.embed(input_ids)
        q = self.q_proj(x).unsqueeze(1)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)
        # 占位：无真实 KV 缓存时 paged_attention 走 torch 参考路径
        h = paged_attention(
            q, k, v, kv_cache, attn_metadata, backend=self.config.attention_backend
        )
        h = h.squeeze(1)
        h = self.o_proj(h)
        return self.lm_head(h)

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict(state_dict, strict=False)


@register_model("dummy")
class DummyCausalLM(BaseCausalLM):
    """快速打通 API；不依赖 GPU Kernel。"""

    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_cache, attn_metadata):
        del positions, kv_cache, attn_metadata
        return self.lm_head(self.embed(input_ids))

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict(state_dict, strict=False)
