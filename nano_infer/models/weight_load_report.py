"""
权重加载对照报告：模型参数 vs checkpoint（映射后）键、形状、missing/unexpected。

环境变量:
  NANO_INFER_DEBUG 含 weights 时输出摘要。
  NANO_INFER_DEBUG_WEIGHTS_VERBOSE=1 时输出完整键列表、按层统计、形状不匹配详情。
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch.nn as nn

from nano_infer.debug import enabled as dbg_enabled
from nano_infer.debug import log as dlog


def _weights_verbose() -> bool:
    v = os.environ.get("NANO_INFER_DEBUG_WEIGHTS_VERBOSE", "").strip().lower()
    return v in ("1", "true", "yes", "full", "all")


def report_before_load_state_dict(
    model: nn.Module,
    mapped: Dict[str, Any],
) -> Tuple[
    List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]],
    List[str],
    List[str],
    int,
]:
    """
    在 load_state_dict 之前对照 model 与 mapped。
    返回 (shape_mismatches, only_in_model_sorted, only_in_mapped_sorted, intersection_count).
    """
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    mapped_keys = set(mapped.keys())

    only_model = sorted(model_keys - mapped_keys)
    only_mapped = sorted(mapped_keys - model_keys)
    both = model_keys & mapped_keys

    shape_mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for k in sorted(both):
        pt = model_sd[k]
        ckpt = mapped[k]
        if not hasattr(ckpt, "shape"):
            continue
        ms = tuple(pt.shape)
        cs = tuple(ckpt.shape)
        if ms != cs:
            shape_mismatches.append((k, ms, cs))

    return shape_mismatches, only_model, only_mapped, len(both)


def log_weight_load_report(
    model: nn.Module,
    mapped: Dict[str, Any],
    missing_keys: List[str],
    unexpected_keys: List[str],
) -> None:
    """在 load_state_dict 之后调用；内部仍用当前 model 与 mapped 做键/形状对照（形状与 load 前一致）。"""
    if not dbg_enabled("weights"):
        return

    verbose = _weights_verbose()
    shape_mm, only_model, only_mapped, n_both = report_before_load_state_dict(model, mapped)
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(mapped.keys())

    dlog(
        "weights",
        f"alignment: model_params={len(model_keys)} ckpt_tensors={len(mapped_keys)} "
        f"intersection={n_both} only_in_model={len(only_model)} only_in_ckpt_unused={len(only_mapped)}",
    )

    dlog(
        "weights",
        f"load_state_dict: missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
        f"shape_mismatch(in_intersection)={len(shape_mm)}",
    )

    if shape_mm:
        limit = 100 if verbose else 20
        for k, ms, cs in shape_mm[:limit]:
            dlog("weights", f"  SHAPE_MISMATCH {k}: model{ms} vs ckpt{cs}")
        if len(shape_mm) > limit:
            dlog("weights", f"  ... {len(shape_mm) - limit} more (NANO_INFER_DEBUG_WEIGHTS_VERBOSE=1 for up to 100)")

    if missing_keys:
        dlog("weights", f"  MISSING (checkpoint 无此参数，模型保持初始化) count={len(missing_keys)}")
        lim = len(missing_keys) if verbose else min(20, len(missing_keys))
        for k in missing_keys[:lim]:
            dlog("weights", f"    - {k}")
        if len(missing_keys) > lim:
            dlog("weights", f"    ... +{len(missing_keys) - lim} (NANO_INFER_DEBUG_WEIGHTS_VERBOSE=1 列出全部)")

    if unexpected_keys:
        dlog("weights", f"  UNEXPECTED (checkpoint 有但模型无对应参数，未加载) count={len(unexpected_keys)}")
        lim = len(unexpected_keys) if verbose else min(20, len(unexpected_keys))
        for k in unexpected_keys[:lim]:
            dlog("weights", f"    - {k}")
        if len(unexpected_keys) > lim:
            dlog("weights", f"    ... +{len(unexpected_keys) - lim}")

    # 一致性：only_in_model 应与 missing 一致（strict=False 时）
    miss_set = set(missing_keys)
    only_m_set = set(only_model)
    if only_m_set != miss_set:
        extra_miss = sorted(only_m_set - miss_set)
        extra_ret = sorted(miss_set - only_m_set)
        if extra_miss or extra_ret:
            dlog(
                "weights",
                f"  WARN only_in_model 与 missing_keys 不一致: "
                f"only_model_minus_missing={len(extra_miss)} missing_minus_only_model={len(extra_ret)}",
            )
            if verbose:
                for k in extra_miss[:25]:
                    dlog("weights", f"    only_model 有但不在 missing: {k}")
                for k in extra_ret[:25]:
                    dlog("weights", f"    missing 有但不在 only_model: {k}")

    if verbose:
        layer_counts: defaultdict[str, int] = defaultdict(int)
        for k in model_keys & mapped_keys:
            m = re.match(r"^layers\.(\d+)\.", k)
            if m:
                layer_counts[f"layer_{m.group(1)}"] += 1
            elif k.startswith("embed_tokens"):
                layer_counts["embed_tokens"] += 1
            elif k.startswith("norm."):
                layer_counts["final_norm"] += 1
            elif k.startswith("lm_head"):
                layer_counts["lm_head"] += 1

        def _sort_key(item: tuple[str, int]) -> tuple:
            name, _c = item
            if name == "embed_tokens":
                return (0, 0)
            if name == "final_norm":
                return (9000, 0)
            if name == "lm_head":
                return (9001, 0)
            if name.startswith("layer_"):
                return (1, int(name.split("_", 1)[1]))
            return (2, name)

        ordered = dict(sorted(layer_counts.items(), key=_sort_key))
        dlog("weights", f"  per-group key counts (intersection): {ordered}")

    if not missing_keys and not unexpected_keys and not shape_mm:
        dlog("weights", "  OK: 全部模型参数在 checkpoint 中有对应张量且形状一致，无多余未使用键。")
