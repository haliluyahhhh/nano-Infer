"""算子统一入口 — 按 backend 路由。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nano_infer.runner.attention_meta import AttentionMetadata


def _torch_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """占位：逐位置自注意力（只看当前 K/V），无真实 KV 缓存时使用。"""
    del key
    return value


def _torch_paged(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: "AttentionMetadata",
) -> torch.Tensor:
    """
    Paged KV Cache 的 torch 参考实现：scatter → gather → causal attention。
    kv_cache: [2, num_blocks, block_size, num_heads, head_dim]
    """
    import torch.nn.functional as F

    n_heads = query.shape[1]
    head_dim = query.shape[2]
    block_size = attn_metadata.block_size
    device = query.device
    dtype = attn_metadata.dtype  # 与 kv_cache 一致，避免 Half/Float 混用

    # 1. Scatter: 将新 K/V 写入 cache（必要时转 dtype）
    slots = attn_metadata.slot_mapping
    for i in range(attn_metadata.total_tokens):
        slot = int(slots[i].item())
        blk = slot // block_size
        off = slot % block_size
        kv_cache[0, blk, off, :, :] = key[i].to(dtype)
        kv_cache[1, blk, off, :, :] = value[i].to(dtype)

    # 2. Gather: 为每个 seq 构建完整 K/V 并做 causal attention
    outputs: list[torch.Tensor] = []
    seq_lens = attn_metadata.seq_lens.cpu().tolist()
    context_lens = attn_metadata.context_lens.cpu().tolist()
    block_tables = attn_metadata.block_tables

    offset = 0
    for seq_idx in range(attn_metadata.num_seqs):
        seq_len = seq_lens[seq_idx]
        n_blocks = (seq_len + block_size - 1) // block_size

        # 从 cache 收集 K/V
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []
        for p in range(seq_len):
            blk_idx = p // block_size
            off = p % block_size
            phys_blk = block_tables[seq_idx, blk_idx].item()
            k_list.append(kv_cache[0, phys_blk, off, :, :])
            v_list.append(kv_cache[1, phys_blk, off, :, :])
        K = torch.stack(k_list, dim=0)
        V = torch.stack(v_list, dim=0)

        # 本步的 Q（prefill 多 token / decode 单 token）
        n_q = seq_len - context_lens[seq_idx]
        if n_q == 0:
            continue
        Q_seq = query[offset : offset + n_q].to(dtype)
        offset += n_q

        K = K.to(dtype)
        V = V.to(dtype)
        scale = head_dim ** -0.5
        # 正确 matmul: Q[n_q,n_heads,d] @ K[seq_len,n_heads,d]^T -> [n_q,n_heads,seq_len]
        # 原 Q@K.T 在 n_q==seq_len 时得到 [n_q,n_heads,n_heads]，维度错误
        attn = torch.einsum("qhd,khd->qhk", Q_seq, K) * scale  # [n_q, n_heads, seq_len]
        # Causal mask: query 局部位置 i(对应全局 ctx+i) 只能 attend 到 key 0..ctx+i
        # mask 形状 [n_q, 1, seq_len] 以与 attn [n_q, n_heads, seq_len] 广播
        ctx = context_lens[seq_idx]
        row = torch.arange(n_q, device=device, dtype=torch.long).unsqueeze(1)
        col = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        causal = col <= ctx + row
        mask = torch.where(
            causal,
            torch.zeros(1, device=device, dtype=dtype),
            torch.full((1,), float("-inf"), device=device, dtype=dtype),
        ).reshape(n_q, seq_len).unsqueeze(1)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ V
        outputs.append(out)

    return torch.cat(outputs, dim=0)


def paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: "AttentionMetadata | None",
    backend: str = "torch",
) -> torch.Tensor:
    """
    query/key/value: [total_tokens, num_heads, head_dim]。
    """
    if backend == "torch":
        if attn_metadata is not None and kv_cache is not None and kv_cache.numel() > 1:
            return _torch_paged(query, key, value, kv_cache, attn_metadata)
        return _torch_reference(query, key, value)
    if backend == "triton":
        return _triton_paged(query, key, value, attn_metadata)
    if backend == "flashinfer":
        return _flashinfer_paged(query, key, value, attn_metadata)
    return _torch_reference(query, key, value)


def _triton_paged(q, k, v, meta):
    del meta
    # TODO: Triton paged kernel
    return _torch_reference(q, k, v)


def _flashinfer_paged(q, k, v, meta):
    del meta
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        pass
    return _torch_reference(q, k, v)
