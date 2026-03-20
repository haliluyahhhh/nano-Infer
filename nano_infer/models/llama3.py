"""Llama3 完整结构：多层 DecoderLayer、RMSNorm、RoPE、MLP。"""

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


class RMSNorm(nn.Module):
    """Llama 使用的 RMSNorm。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.rsqrt(variance) * self.weight


def _rotary_embeddings(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对 x 应用 RoPE。x: [T, H, D]，cos/sin: [T, D//2]，按坐标对 (2i,2i+1) 旋转。"""
    half = x.shape[-1] // 2
    x1 = x[..., half:]
    x2 = x[..., :half]
    x_rot = torch.cat([-x1, x2], dim=-1)
    # cos/sin [T, D/2] 每元素对应一对坐标，需扩展到 [T, H, D] 以便与 x 广播
    cos_b = cos.unsqueeze(1).repeat_interleave(2, dim=-1).expand(*x.shape[:-1], x.shape[-1])
    sin_b = sin.unsqueeze(1).repeat_interleave(2, dim=-1).expand(*x.shape[:-1], x.shape[-1])
    return x * cos_b + x_rot * sin_b


def _precompute_freqs(
    dim: int,
    max_pos: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE cos/sin。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class LlamaMLP(nn.Module):
    """Llama 的 SwiGLU MLP：gate_proj, up_proj, down_proj。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """单层 Attention：QKV 投影 + paged_attention + o_proj。"""

    def __init__(self, config: EngineConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        h = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv = config.num_key_value_heads or n_heads
        # Llama 仅按几何维度推导 head_dim，不使用 head_dim_override（Qwen3 见 models/qwen3.py）
        head_dim = h // n_heads
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        has_bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(h, n_heads * head_dim, bias=has_bias)
        self.k_proj = nn.Linear(h, n_kv * head_dim, bias=has_bias)
        self.v_proj = nn.Linear(h, n_kv * head_dim, bias=has_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, h, bias=False)

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

        if hasattr(self.config, "rope_theta") and self.config.rope_theta:
            max_pos = int(positions.max().item()) + 1
            cos, sin = _precompute_freqs(
                self.head_dim, max_pos, base=getattr(self.config, "rope_theta", 10000.0), device=x.device
            )
            pos_flat = positions.view(-1).long().clamp(0, cos.shape[0] - 1)
            cos_sel = cos[pos_flat]
            sin_sel = sin[pos_flat]
            q = _rotary_embeddings(q, cos_sel, sin_sel)
            k = _rotary_embeddings(k, cos_sel, sin_sel)

        if dbg_enabled("model") and self.layer_idx == 0:
            dlog("model", f"attn layer[0]: {tensor_summary(q, 'q')} {tensor_summary(k, 'k')}")

        h = paged_attention(
            q.unsqueeze(1) if q.dim() == 2 else q,
            k.unsqueeze(1) if k.dim() == 2 else k,
            v.unsqueeze(1) if v.dim() == 2 else v,
            kv_cache,
            attn_metadata,
            backend=self.config.attention_backend,
        )
        if h.dim() == 3:
            h = h.squeeze(1)
        h = h.reshape(T, -1)
        return self.o_proj(h)


class LlamaDecoderLayer(nn.Module):
    """Llama Decoder 层：input_norm → attention → residual → post_norm → mlp → residual。"""

    def __init__(self, config: EngineConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layers_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layers_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


@register_model("llama3")
class Llama3ForCausalLM(BaseCausalLM):
    """
    完整 Llama 结构：Embedding + N 层 DecoderLayer + norm + LM head。
    支持从 HuggingFace 格式加载权重。
    """

    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        dlog("model", f"embed: {tensor_summary(x, 'x')}")
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache.dim() >= 2 else kv_cache
            x = layer(x, positions, layer_kv, attn_metadata)
            if dbg_enabled("model") and i == 0:
                dlog("model", f"after layer[0]: {tensor_summary(x, 'x')}")
        x = self.norm(x)
        logits = self.lm_head(x)
        dlog("model", f"lm_head: {tensor_summary(logits, 'logits')}")
        return logits

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        from nano_infer.models.weight_loader import map_hf_llama_to_nano
        from nano_infer.models.weight_load_report import log_weight_load_report

        if any(k.startswith("layers.") or k.startswith("embed_tokens.") for k in state_dict.keys()):
            mapped = state_dict
            dlog("weights", "state_dict already in nano format, skip re-mapping")
        else:
            mapped = map_hf_llama_to_nano(state_dict)

        # tied embeddings: checkpoint 没有 lm_head.weight 时，复用 embed_tokens.weight
        if "lm_head.weight" not in mapped and "embed_tokens.weight" in mapped:
            mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
            dlog("weights", "tie_word_embeddings: copied embed_tokens.weight -> lm_head.weight")

        dlog("weights", f"about to load_state_dict: raw_keys={len(state_dict)} mapped_keys={len(mapped)}")
        ret = self.load_state_dict(mapped, strict=False)
        log_weight_load_report(self, mapped, list(ret.missing_keys), list(ret.unexpected_keys))


@register_model("dummy")
class DummyCausalLM(BaseCausalLM):
    """快速打通 API；不依赖 GPU Kernel。"""

    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_cache, attn_metadata):
        del positions, kv_cache, attn_metadata
        return self.lm_head(self.embed(input_ids))

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        self.load_state_dict(state_dict, strict=False)
