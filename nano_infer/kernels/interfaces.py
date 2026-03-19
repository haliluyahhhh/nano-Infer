"""算子统一入口 — 按 backend 路由。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nano_infer.debug import enabled as dbg_enabled
from nano_infer.debug import log as dlog
from nano_infer.debug import tensor_summary

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

    同时支持 MHA 和 GQA：
      - MHA: n_q_heads == n_kv_heads，所有头一一对应
      - GQA: n_q_heads > n_kv_heads，多个 Q 头共享一组 KV 头
    KV Cache 按紧凑的 n_kv_heads 存储，计算时再扩展，节省内存。

    kv_cache: [2, num_blocks, block_size, n_kv_heads, head_dim]
    query:    [total_tokens, n_q_heads, head_dim]
    key:      [total_tokens, n_kv_heads, head_dim]
    value:    [total_tokens, n_kv_heads, head_dim]
    """
    import torch.nn.functional as F

    n_q_heads = query.shape[1]
    n_kv_heads = key.shape[1]
    head_dim = query.shape[2]
    block_size = attn_metadata.block_size
    device = query.device
    dtype = attn_metadata.dtype

    # GQA: 每组有多少个 Q 头共享一个 KV 头
    assert n_q_heads % n_kv_heads == 0, (
        f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
    )
    gqa_group_size = n_q_heads // n_kv_heads
    is_gqa = gqa_group_size > 1

    dlog("attention", f"_torch_paged entry: {tensor_summary(query, 'query')} "
         f"n_q_heads={n_q_heads} n_kv_heads={n_kv_heads} "
         f"gqa={'yes group=' + str(gqa_group_size) if is_gqa else 'no (MHA)'} "
         f"kv_cache={list(kv_cache.shape)} n_seqs={attn_metadata.num_seqs}")

    # ── 1. Scatter: 将新 K/V 写入 cache（紧凑的 n_kv_heads）──
    slots = attn_metadata.slot_mapping
    for i in range(attn_metadata.total_tokens):
        slot = int(slots[i].item())
        blk = slot // block_size
        off = slot % block_size
        kv_cache[0, blk, off, :, :] = key[i].to(dtype)
        kv_cache[1, blk, off, :, :] = value[i].to(dtype)

    # ── 2. Gather + Attention: 逐 seq 构建 K/V 并计算 ──
    outputs: list[torch.Tensor] = []
    seq_lens = attn_metadata.seq_lens.cpu().tolist()
    context_lens = attn_metadata.context_lens.cpu().tolist()
    block_tables = attn_metadata.block_tables

    offset = 0
    for seq_idx in range(attn_metadata.num_seqs):
        seq_len = seq_lens[seq_idx]

        # 从 cache 收集 K/V — 紧凑形式 [seq_len, n_kv_heads, d]
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []
        for p in range(seq_len):
            blk_idx = p // block_size
            off = p % block_size
            phys_blk = block_tables[seq_idx, blk_idx].item()
            k_list.append(kv_cache[0, phys_blk, off, :, :])
            v_list.append(kv_cache[1, phys_blk, off, :, :])
        K = torch.stack(k_list, dim=0).to(dtype)   # [seq_len, n_kv_heads, d]
        V = torch.stack(v_list, dim=0).to(dtype)   # [seq_len, n_kv_heads, d]

        # 本步的 Q（prefill 多 token / decode 单 token）
        n_q = seq_len - context_lens[seq_idx]
        if n_q == 0:
            continue
        Q_seq = query[offset : offset + n_q].to(dtype)  # [n_q, n_q_heads, d]
        offset += n_q

        # GQA 扩展: 将 K/V 从 n_kv_heads 重复到 n_q_heads
        if is_gqa:
            K = K.repeat_interleave(gqa_group_size, dim=1)  # [seq_len, n_q_heads, d]
            V = V.repeat_interleave(gqa_group_size, dim=1)  # [seq_len, n_q_heads, d]

        dlog("attention", f"  seq[{seq_idx}]: n_q={n_q} seq_len={seq_len} ctx={context_lens[seq_idx]} "
             f"Q={tuple(Q_seq.shape)} K={tuple(K.shape)} V={tuple(V.shape)}")

        scale = head_dim ** -0.5

        # ── Scaled Dot-Product Attention (标准 bmm) ──
        # 将 head 维度提到 batch 位:
        #   Q: [n_q, n_q_heads, d] → [n_q_heads, n_q, d]
        #   K: [seq_len, n_q_heads, d] → [n_q_heads, d, seq_len]
        #   V: [seq_len, n_q_heads, d] → [n_q_heads, seq_len, d]
        Q_b = Q_seq.permute(1, 0, 2)               # [n_q_heads, n_q, d]
        K_b = K.permute(1, 2, 0)                   # [n_q_heads, d, seq_len]
        V_b = V.permute(1, 0, 2)                   # [n_q_heads, seq_len, d]

        # Q @ K^T → attention scores
        attn = torch.bmm(Q_b, K_b) * scale         # [n_q_heads, n_q, seq_len]

        # Causal mask: 位置 i（全局 ctx+i）只能看到 key 0..ctx+i
        ctx = context_lens[seq_idx]
        row = torch.arange(n_q, device=device, dtype=torch.long).unsqueeze(1)
        col = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        causal = col <= ctx + row                   # [n_q, seq_len]
        mask = torch.where(
            causal,
            torch.zeros(1, device=device, dtype=dtype),
            torch.full((1,), float("-inf"), device=device, dtype=dtype),
        )                                           # [n_q, seq_len]
        attn = attn + mask.unsqueeze(0)             # [1, n_q, seq_len] 广播到全部 head
        attn = F.softmax(attn, dim=-1)              # [n_q_heads, n_q, seq_len]

        if dbg_enabled("attention"):
            dlog("attention", f"  attn_weights: {tensor_summary(attn, 'attn')}")

        # attn @ V → output
        out = torch.bmm(attn, V_b)                 # [n_q_heads, n_q, d]
        out = out.permute(1, 0, 2)                  # [n_q, n_q_heads, d]
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
    支持 MHA 和 GQA:
      query: [total_tokens, n_q_heads, head_dim]
      key:   [total_tokens, n_kv_heads, head_dim]
      value: [total_tokens, n_kv_heads, head_dim]
    GQA 时 n_q_heads > n_kv_heads，内部自动扩展 K/V。
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
