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
    """MVP：逐位置自注意力占位（等价于只看当前 K/V），保证形状正确。"""
    del key
    return value


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
    del kv_cache
    if backend == "torch":
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
