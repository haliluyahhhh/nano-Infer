"""调度器抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from nano_infer.config import EngineConfig
    from nano_infer.engine.memory.block_manager import BlockManager
    from nano_infer.engine.sequence import Sequence


@dataclass
class SchedulerOutput:
    """本步参与前向的序列及每序列 token 数。"""

    sequences: List["Sequence"]
    num_tokens_per_seq: List[int]
    is_prefill: bool

    def total_tokens(self) -> int:
        return sum(self.num_tokens_per_seq)


class SchedulerBase(ABC):
    def __init__(self, config: "EngineConfig"):
        self.config = config

    @abstractmethod
    def schedule(
        self,
        waiting: List["Sequence"],
        running: List["Sequence"],
        block_manager: "BlockManager",
    ) -> SchedulerOutput | None:
        """若无就绪序列返回 None。"""


def _ensure_blocks(
    seq: "Sequence",
    block_manager: "BlockManager",
    total_tokens_after_step: int,
) -> bool:
    """为序列预留到 total_tokens_after_step 所需的块；失败返回 False。"""
    need = block_manager.num_blocks_for_tokens(total_tokens_after_step)
    have = len(block_manager.get_block_table(seq.seq_id))
    if need <= have:
        seq.block_table = block_manager.get_block_table(seq.seq_id)
        return True
    extra = need - have
    got = block_manager.allocate_blocks(seq.seq_id, extra)
    if len(got) < extra:
        return False
    seq.block_table = block_manager.get_block_table(seq.seq_id)
    return True
