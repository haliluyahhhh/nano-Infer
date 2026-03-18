"""Radix 前缀感知调度 — 当前委托 FCFS 逻辑，树匹配在 Engine 层可叠加。"""

from __future__ import annotations

from nano_infer.engine.memory.radix_tree import RadixTree
from nano_infer.engine.scheduler.vllm_scheduler import VLLMScheduler


class RadixScheduler(VLLMScheduler):
    """
    与 VLLMScheduler 相同调度顺序；保留 RadixTree 供后续：
    - 命中前缀时减少 prefill token 与块复制。
    """

    def __init__(self, config):
        super().__init__(config)
        self.tree = RadixTree()

    def schedule(self, waiting, running, block_manager):
        # TODO: 对 waiting 中 seq 做 longest_prefix_match，调整 num_computed_tokens
        return super().schedule(waiting, running, block_manager)
