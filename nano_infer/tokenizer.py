"""
Tokenizer 抽象：优先 HuggingFace AutoTokenizer，无则回退字符级占位。
"""

from __future__ import annotations

from typing import Callable, List, Optional


def _fallback_encode(text: str, vocab_size: int = 50257) -> List[int]:
    """占位：字符 → id，无 tokenizer 时使用。"""
    return [min(vocab_size - 1, ord(c) % vocab_size) for c in text[:2048]]


def _fallback_decode(ids: List[int]) -> str:
    """占位：id → 字符。"""
    return "".join(chr(i) if 0 <= i < 65536 else "?" for i in ids)


def get_tokenizer(
    model_path: Optional[str] = None,
    vocab_size: int = 50257,
) -> tuple[Callable[[str], List[int]], Callable[[List[int]], str], Optional[int]]:
    """
    返回 (encode, decode, eos_token_id)。
    若 model_path 存在且可加载 transformers，使用 AutoTokenizer；
    否则使用字符级占位，eos_token_id 为 None。
    """
    if model_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            def encode(text: str) -> List[int]:
                out = tok.encode(text, add_special_tokens=True)
                return out[:2048]

            def decode(ids: List[int]) -> str:
                return tok.decode(ids, skip_special_tokens=True)

            eos = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None)
            return encode, decode, eos
        except Exception:
            pass
    return (
        lambda t: _fallback_encode(t, vocab_size),
        _fallback_decode,
        None,
    )
