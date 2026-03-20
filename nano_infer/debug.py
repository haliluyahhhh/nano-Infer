"""
统一调试系统 — 通过环境变量 NANO_INFER_DEBUG 控制各模块的调试输出。

用法:
    export NANO_INFER_DEBUG=all                     # 全部模块
    export NANO_INFER_DEBUG=attention,scheduler      # 仅 attention + scheduler
    export NANO_INFER_DEBUG=weights,engine,runner     # 仅 weights + engine + runner

支持的模块名（tag）:
    tokenizer   — tokenize 编码/解码阶段
    config      — 配置加载与合并
    weights     — 权重加载、键映射、missing/unexpected（可加 NANO_INFER_DEBUG_WEIGHTS_VERBOSE=1 输出完整键列表）
    model       — 模型 forward：embedding / norm / lm_head 形状与数值摘要
    attention   — QKV 投影、RoPE、paged attention 内部张量形状
    runner      — ModelRunner 输入准备、metadata、kv_cache 分配
    scheduler   — 调度决策：prefill/decode 批次选取
    memory      — 物理块分配/释放
    engine      — 引擎主循环：step / sample / 序列状态
    all         — 以上全部
"""

from __future__ import annotations

import os
import sys
from typing import Any, Set

import torch

_VALID_TAGS = frozenset({
    "tokenizer", "config", "weights", "model", "attention",
    "runner", "scheduler", "memory", "engine",
})

_enabled_tags: Set[str] | None = None
_all_enabled: bool = False


def _parse() -> None:
    global _enabled_tags, _all_enabled
    raw = os.environ.get("NANO_INFER_DEBUG", "").strip()
    if not raw:
        _enabled_tags = set()
        _all_enabled = False
        return
    tags = {t.strip().lower() for t in raw.split(",") if t.strip()}
    if "all" in tags or "1" in tags:
        _all_enabled = True
        _enabled_tags = set(_VALID_TAGS)
    else:
        _all_enabled = False
        _enabled_tags = tags & _VALID_TAGS
        unknown = tags - _VALID_TAGS - {"all", "1"}
        if unknown:
            print(
                f"[nano-debug] WARNING: unknown debug tags: {unknown}. "
                f"valid: {sorted(_VALID_TAGS)}",
                file=sys.stderr,
            )


def enabled(tag: str) -> bool:
    """判断某 tag 是否启用。"""
    if _enabled_tags is None:
        _parse()
    return _all_enabled or tag in _enabled_tags  # type: ignore[operator]


def log(tag: str, *args: Any, **kwargs: Any) -> None:
    """若 tag 启用，输出一行调试信息。"""
    if not enabled(tag):
        return
    parts = " ".join(str(a) for a in args)
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    line = f"[nano-debug:{tag}] {parts}"
    if extra:
        line += f"  {extra}"
    print(line, file=sys.stderr, flush=True)


def tensor_summary(t: torch.Tensor, name: str = "") -> str:
    """返回 tensor 的形状 + dtype + 数值范围摘要字符串。"""
    prefix = f"{name} " if name else ""
    with torch.no_grad():
        flat = t.float()
        mn = flat.min().item()
        mx = flat.max().item()
        mean = flat.mean().item()
        has_nan = bool(torch.isnan(flat).any().item())
        has_inf = bool(torch.isinf(flat).any().item())
    flags = ""
    if has_nan:
        flags += " !!NaN"
    if has_inf:
        flags += " !!Inf"
    return f"{prefix}{tuple(t.shape)} {t.dtype} min={mn:.4g} max={mx:.4g} mean={mean:.4g}{flags}"
