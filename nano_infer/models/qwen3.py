"""
Qwen3 因果语言模型（对齐 HuggingFace Qwen3ForCausalLM）。

本模块**独立实现**，不继承、不导入 `llama3`，避免与 Llama 代码耦合。
与 Llama/Qwen2 的主要差异：注意力在 RoPE 之前对 Q、K 使用 RMSNorm（q_norm / k_norm）。

权重键仍经 `weight_loader.map_hf_llama_to_nano` 映射为 nano 内部命名（与 HF `model.*` 前缀对应）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_infer.config import EngineConfig
from nano_infer.debug import enabled as dbg_enabled
from nano_infer.debug import log as dlog
from nano_infer.debug import tensor_summary
from nano_infer.kernels.interfaces import paged_attention
from nano_infer.models.base import BaseCausalLM
from nano_infer.models.registry import register_model

if TYPE_CHECKING:
    from nano_infer.runner.attention_meta import AttentionMetadata


class Qwen3RMSNorm(nn.Module):
    """RMSNorm，末维归一化。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.rsqrt(variance) * self.weight


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """RoPE。x: [T, H, D]，cos/sin: [T, D//2]。"""
    half = x.shape[-1] // 2
    x1 = x[..., half:]
    x2 = x[..., :half]
    x_rot = torch.cat([-x1, x2], dim=-1)
    cos_b = cos.unsqueeze(1).repeat_interleave(2, dim=-1).expand(*x.shape[:-1], x.shape[-1])
    sin_b = sin.unsqueeze(1).repeat_interleave(2, dim=-1).expand(*x.shape[:-1], x.shape[-1])
    return x * cos_b + x_rot * sin_b


def _precompute_rope_freqs(
    dim: int,
    max_pos: int,
    base: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    """Qwen3 注意力：QKV → q_norm / k_norm → RoPE → paged_attention → o_proj。"""

    def __init__(self, config: EngineConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        h = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv = config.num_key_value_heads or n_heads
        head_dim = config.head_dim
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        has_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(h, n_heads * head_dim, bias=has_bias)
        self.k_proj = nn.Linear(h, n_kv * head_dim, bias=has_bias)
        self.v_proj = nn.Linear(h, n_kv * head_dim, bias=has_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, h, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(T, self.n_kv, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if getattr(self.config, "rope_theta", 0):
            max_pos = int(positions.max().item()) + 1
            cos, sin = _precompute_rope_freqs(
                self.head_dim,
                max_pos,
                base=float(getattr(self.config, "rope_theta", 1e6)),
                device=x.device,
            )
            pos_flat = positions.view(-1).long().clamp(0, cos.shape[0] - 1)
            q = _apply_rotary_emb(q, cos[pos_flat], sin[pos_flat])
            k = _apply_rotary_emb(k, cos[pos_flat], sin[pos_flat])

        if dbg_enabled("model") and self.layer_idx == 0:
            dlog("model", f"qwen3 attn[0]: {tensor_summary(q, 'q')} {tensor_summary(k, 'k')}")

        out = paged_attention(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
            backend=self.config.attention_backend,
        )
        if out.dim() == 3:
            out = out.squeeze(1)
        out = out.reshape(T, -1)
        return self.o_proj(out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: EngineConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layers_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layers_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        residual = x
        x = self.input_layers_norm(x)
        x = self.self_attn(x, positions, kv_cache, attn_metadata)
        x = residual + x
        residual = x
        x = self.post_attention_layers_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x


@register_model("qwen3")
class Qwen3ForCausalLM(BaseCausalLM):
    """Qwen3：Embedding + DecoderLayer × N + RMSNorm + lm_head。"""

    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        dlog("model", f"qwen3 embed: {tensor_summary(x, 'x')}")
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache.dim() >= 2 else kv_cache
            x = layer(x, positions, layer_kv, attn_metadata)
            if dbg_enabled("model") and i == 0:
                dlog("model", f"qwen3 after layer[0]: {tensor_summary(x, 'x')}")
        x = self.norm(x)
        logits = self.lm_head(x)
        dlog("model", f"qwen3 lm_head: {tensor_summary(logits, 'logits')}")
        return logits

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        from nano_infer.models.weight_loader import map_hf_llama_to_nano
        from nano_infer.models.weight_load_report import log_weight_load_report

        if any(k.startswith("layers.") or k.startswith("embed_tokens.") for k in state_dict.keys()):
            mapped = state_dict
            dlog("weights", "qwen3: state_dict already nano keys, skip map")
        else:
            mapped = map_hf_llama_to_nano(state_dict)

        if "lm_head.weight" not in mapped and "embed_tokens.weight" in mapped:
            mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
            dlog("weights", "qwen3 tie_word_embeddings: embed_tokens -> lm_head.weight")

        dlog("weights", f"qwen3 load_state_dict: mapped_keys={len(mapped)}")
        ret = self.load_state_dict(mapped, strict=False)
        log_weight_load_report(self, mapped, list(ret.missing_keys), list(ret.unexpected_keys))
