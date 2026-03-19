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
    HF Llama 键 → nano-Infer Llama3 结构键。
    映射不到则返回 None（跳过）。
    """
    # embed_tokens
    if hf_key == "model.embed_tokens.weight":
        return "embed_tokens.weight"

    # layers
    if hf_key.startswith("model.layers."):
        rest = hf_key[len("model.layers.") :]
        i = rest.split(".")[0]
        # self_attn
        if f"model.layers.{i}.self_attn.q_proj.weight" == hf_key:
            return f"layers.{i}.self_attn.q_proj.weight"
        if f"model.layers.{i}.self_attn.k_proj.weight" == hf_key:
            return f"layers.{i}.self_attn.k_proj.weight"
        if f"model.layers.{i}.self_attn.v_proj.weight" == hf_key:
            return f"layers.{i}.self_attn.v_proj.weight"
        if f"model.layers.{i}.self_attn.o_proj.weight" == hf_key:
            return f"layers.{i}.self_attn.o_proj.weight"
        # mlp
        if f"model.layers.{i}.mlp.gate_proj.weight" == hf_key:
            return f"layers.{i}.mlp.gate_proj.weight"
        if f"model.layers.{i}.mlp.up_proj.weight" == hf_key:
            return f"layers.{i}.mlp.up_proj.weight"
        if f"model.layers.{i}.mlp.down_proj.weight" == hf_key:
            return f"layers.{i}.mlp.down_proj.weight"
        # norm
        if f"model.layers.{i}.input_layers_norm.weight" == hf_key:
            return f"layers.{i}.input_layers_norm.weight"
        if f"model.layers.{i}.post_attention_layers_norm.weight" == hf_key:
            return f"layers.{i}.post_attention_layers_norm.weight"
        # 兼容 input_layer_norm（无 s）
        if hf_key == f"model.layers.{i}.input_layers_norm.weight" or "input_layer_norm" in rest:
            return f"layers.{i}.input_layers_norm.weight"
        if hf_key == f"model.layers.{i}.post_attention_layers_norm.weight" or "post_attention_layers_norm" in rest:
            return f"layers.{i}.post_attention_layers_norm.weight"

    # model.norm（Llama2/3 皆用 model.norm）
    if hf_key == "model.norm.weight":
        return "norm.weight"

    # lm_head
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # HF 变体：input_layer_norm（无 s）
    if hf_key.startswith("model.layers.") and "input_layer_norm" in hf_key and "input_layers_norm" not in hf_key:
        rest = hf_key[len("model.layers.") :]
        i = rest.split(".")[0]
        return f"layers.{i}.input_layers_norm.weight"
    if hf_key.startswith("model.layers.") and "post_attention_layer_norm" in hf_key:
        rest = hf_key[len("model.layers.") :]
        i = rest.split(".")[0]
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
