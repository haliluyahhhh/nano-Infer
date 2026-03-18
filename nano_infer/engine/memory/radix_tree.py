"""
Radix 树前缀缓存（SGLang 风格）— 占位与扩展点。

完整实现需：节点持有 block 引用、引用计数、与 Scheduler 协同驱逐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RadixNode:
    token_prefix: Tuple[int, ...] = ()
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    block_ids: List[int] = field(default_factory=list)


class RadixTree:
    """MVP：仅提供 insert/match 骨架，块复用逻辑待与 BlockManager 打通。"""

    def __init__(self) -> None:
        self.root = RadixNode()

    def longest_prefix_match(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """
        返回 (匹配长度, 可复用的 block_ids 占位)。
        当前返回 (0, [])，后续按路径合并块。
        """
        return 0, []

    def insert(self, token_ids: List[int], block_ids: List[int]) -> None:
        """将路径与块关联写入树（简化占位）。"""
        node = self.root
        for tid in token_ids[:1]:  # 占位：只演示单层
            if tid not in node.children:
                node.children[tid] = RadixNode()
            node = node.children[tid]
        node.block_ids = list(block_ids)
