"""Paged KV 物理块管理（Free-list）。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from nano_infer.debug import log as dlog


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: Set[int] = set(range(num_blocks))
        self._seq_blocks: Dict[int, List[int]] = defaultdict(list)

    def num_blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def allocate_blocks(self, seq_id: int, count: int) -> List[int]:
        if count <= 0:
            return []
        if len(self._free) < count:
            dlog("memory", f"alloc FAILED: seq={seq_id} need={count} free={len(self._free)}")
            return []
        out: List[int] = []
        for _ in range(count):
            b = self._free.pop()
            out.append(b)
        self._seq_blocks[seq_id].extend(out)
        dlog("memory", f"alloc: seq={seq_id} +{count} blocks={out} free_remaining={len(self._free)}")
        return out

    def get_block_table(self, seq_id: int) -> List[int]:
        return list(self._seq_blocks.get(seq_id, []))

    def free_sequence(self, seq_id: int) -> None:
        blocks = self._seq_blocks.pop(seq_id, [])
        for b in blocks:
            self._free.add(b)
        if blocks:
            dlog("memory", f"free: seq={seq_id} returned={len(blocks)} blocks, free_now={len(self._free)}")

    def num_free_blocks(self) -> int:
        return len(self._free)
