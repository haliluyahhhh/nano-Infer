"""Attention 元数据 — Runner 填充，Model / Kernel 消费。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionMetadata:
    """
    PagedAttention 类接口约定（与 FlashInfer / Triton kernel 对齐时可扩展）。
    """

    num_seqs: int
    total_tokens: int
    seq_lens: torch.Tensor  # 各序列本步结束后总 KV 长度
    context_lens: torch.Tensor  # 本步开始前各序列已有 KV 长度
    max_seqlen_q: int
    max_seqlen_k: int
    block_tables: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    slot_mapping: torch.Tensor  # [total_tokens] 新 token 写入 KV pool 的 slot
    block_size: int = 16
    device: torch.device
    dtype: torch.dtype
