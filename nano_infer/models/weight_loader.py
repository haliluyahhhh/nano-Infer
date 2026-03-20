"""
HuggingFace 权重加载：safetensors / pytorch_model.bin + 名称映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from nano_infer.debug import enabled as dbg_enabled
from nano_infer.debug import log as dlog


def _map_llama_hf_to_nano(hf_key: str) -> str | None:
    """
    HF 键 → nano-Infer 键。支持 Llama / Qwen2 / Qwen3（含 q_norm、k_norm）等。

    HF 层内命名变体：
      - input_layernorm        (Qwen2/Llama3 标准)
      - input_layer_norm       (部分 Llama2 变体)
      - post_attention_layernorm (Qwen2/Llama3 标准)
      - post_attention_layer_norm (部分变体)
    全部映射到 nano 的 input_layers_norm / post_attention_layers_norm。
    """
    if hf_key == "model.embed_tokens.weight":
        return "embed_tokens.weight"
    if hf_key == "model.norm.weight":
        return "norm.weight"
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    if not hf_key.startswith("model.layers."):
        return None

    rest = hf_key[len("model.layers."):]
    i = rest.split(".")[0]
    suffix = rest[len(i) + 1:]  # e.g. "self_attn.q_proj.weight"

    # ── Qwen3：RoPE 前的 Q/K RMSNorm ──
    if suffix == "self_attn.q_norm.weight":
        return f"layers.{i}.self_attn.q_norm.weight"
    if suffix == "self_attn.k_norm.weight":
        return f"layers.{i}.self_attn.k_norm.weight"

    # ── self_attn 投影 (weight + bias) ──
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        for param in ("weight", "bias"):
            if suffix == f"self_attn.{proj}.{param}":
                return f"layers.{i}.self_attn.{proj}.{param}"

    # ── MLP 投影 ──
    for proj in ("gate_proj", "up_proj", "down_proj"):
        if suffix == f"mlp.{proj}.weight":
            return f"layers.{i}.mlp.{proj}.weight"

    # ── LayerNorm（兼容所有 HF 命名变体）──
    # input_layernorm / input_layer_norm / input_layers_norm → input_layers_norm
    if suffix in ("input_layernorm.weight", "input_layer_norm.weight", "input_layers_norm.weight"):
        return f"layers.{i}.input_layers_norm.weight"
    # post_attention_layernorm / post_attention_layer_norm → post_attention_layers_norm
    if suffix in ("post_attention_layernorm.weight", "post_attention_layer_norm.weight",
                   "post_attention_layers_norm.weight"):
        return f"layers.{i}.post_attention_layers_norm.weight"

    return None


def map_hf_llama_to_nano(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """将 HF Llama state_dict 映射为 nano-Infer 键；供 load_weights 使用。"""
    out: Dict[str, Any] = {}
    skipped = []
    for k, v in state_dict.items():
        our_key = _map_llama_hf_to_nano(k)
        if our_key:
            out[our_key] = v
        else:
            skipped.append(k)
    dlog("weights", f"map_hf_to_nano: {len(state_dict)} -> {len(out)} mapped, {len(skipped)} skipped")
    if dbg_enabled("weights") and skipped:
        dlog("weights", f"  skipped keys (first 10): {skipped[:10]}")
    return out


def load_hf_weights(
    model_path: str | Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    从 model_path 加载 HuggingFace 格式权重。
    支持 safetensors（优先）和 pytorch_model.bin。
    返回已映射到 nano-Infer 键的 state_dict。
    """
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"model_path must be a directory: {path}")

    dlog("weights", f"loading weights from {path}, device={device}")
    state_dict: Dict[str, Any] = {}

    # 1. 尝试 safetensors（单文件或多文件）
    st_files = list(path.glob("*.safetensors"))
    if st_files:
        dlog("weights", f"found {len(st_files)} safetensors files")
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ImportError:
            st_files = []

        if st_files:
            # 单文件 model.safetensors 或 model-00001-of-00003.safetensors
            for f in sorted(st_files):
                with safe_open(f, framework="pt", device=device) as sf:
                    for k in sf.keys():
                        t = sf.get_tensor(k)
                        our_key = _map_llama_hf_to_nano(k)
                        if our_key:
                            state_dict[our_key] = t
            if state_dict:
                dlog("weights", f"safetensors loaded: {len(state_dict)} params, keys[:5]={list(state_dict.keys())[:5]}")
                return state_dict

    # 2. pytorch_model.bin
    pt_file = path / "pytorch_model.bin"
    if not pt_file.exists():
        pt_file = path / "model.safetensors"  # 有的仓库只用 safetensors 但无 .safetensors 后缀
    if not pt_file.exists():
        # 尝试 model-00001-of-00001.bin 等
        bins = list(path.glob("pytorch_model*.bin"))
        if bins:
            pt_file = sorted(bins)[0]

    if isinstance(pt_file, Path) and pt_file.exists():
        import torch

        raw = torch.load(pt_file, map_location=device, weights_only=True)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected checkpoint format: {type(raw)}")
        for hf_key, t in raw.items():
            our_key = _map_llama_hf_to_nano(hf_key)
            if our_key:
                state_dict[our_key] = t
        dlog("weights", f"pytorch_model loaded: {len(state_dict)} params")
        return state_dict

    raise FileNotFoundError(
        f"No weights found in {path}. Need *.safetensors or pytorch_model*.bin"
    )
