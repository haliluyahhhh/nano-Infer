"""请求序列与状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Sequence:
    """
    单条生成序列。
    block_table: 各层共享的物理块号列表（MVP 单层逻辑可映射到同一表）。
    """

    seq_id: int
    prompt_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    block_table: List[int] = field(default_factory=list)
    max_tokens: int = 64
    temperature: float = 0.0
    eos_token_id: int | None = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    def total_len(self) -> int:
        return self.num_prompt_tokens + len(self.output_ids)

    def prefill_remaining(self) -> int:
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    def is_prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens
