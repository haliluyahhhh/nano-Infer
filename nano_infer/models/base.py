"""因果 LM 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from nano_infer.runner.attention_meta import AttentionMetadata


class BaseCausalLM(nn.Module, ABC):
    """所有开源模型继承此类；Runner 只依赖此接口。"""

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        """返回 logits，形状 [total_tokens, vocab_size]。"""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError
