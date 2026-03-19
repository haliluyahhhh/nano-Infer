"""
Tokenizer 抽象：优先 HuggingFace AutoTokenizer，无则回退字符级占位。
"""

from __future__ import annotations

from typing import Callable, List, Optional

from nano_infer.debug import log as dlog


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
            dlog("tokenizer", f"loaded HF tokenizer from {model_path}",
                 vocab_size=tok.vocab_size,
                 eos_token_id=getattr(tok, "eos_token_id", None))

            def encode(text: str) -> List[int]:
                out = tok.encode(text, add_special_tokens=True)
                out = out[:2048]
                dlog("tokenizer", f"encode: {text!r:.80s} -> {len(out)} ids, first5={out[:5]}")
                return out

            def decode(ids: List[int]) -> str:
                result = tok.decode(ids, skip_special_tokens=True)
                dlog("tokenizer", f"decode: {len(ids)} ids -> {result!r:.80s}")
                return result

            eos = getattr(tok, "eos_token_id", None) or getattr(tok, "pad_token_id", None)
            return encode, decode, eos
        except Exception as e:
            dlog("tokenizer", f"HF tokenizer failed: {e}, falling back to char-level")
    dlog("tokenizer", "using fallback char-level tokenizer")
    return (
        lambda t: _fallback_encode(t, vocab_size),
        _fallback_decode,
        None,
    )
