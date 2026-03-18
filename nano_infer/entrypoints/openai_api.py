"""
OpenAI 兼容 HTTP 入口（Chat Completions + 流式 SSE）。
当 model_path 设置时使用 HuggingFace Tokenizer；否则使用字符级占位。
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nano_infer.config import config_from_env
from nano_infer.engine.async_llm_engine import AsyncLLMEngine, build_engine
from nano_infer.tokenizer import get_tokenizer

_engine: AsyncLLMEngine | None = None
_encode_fn = None
_decode_fn = None
_eos_token_id: Optional[int] = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _engine, _encode_fn, _decode_fn, _eos_token_id
    cfg = config_from_env()
    if os.environ.get("NANO_INFER_MODEL"):
        cfg.model_name = os.environ["NANO_INFER_MODEL"]
    _encode_fn, _decode_fn, _eos_token_id = get_tokenizer(
        cfg.model_path,
        vocab_size=cfg.vocab_size,
    )
    _engine, _ = build_engine(cfg)
    yield
    # shutdown（如有资源释放可在此执行）


app = FastAPI(title="nano-Infer", version="0.1.0", lifespan=_lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "dummy"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=64, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    parts = []
    for m in messages:
        parts.append(f"{m.role}: {m.content}")
    return "\n".join(parts)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "nano-infer"}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    import traceback

    if _engine is None or _encode_fn is None or _decode_fn is None:
        return {"error": "Engine not initialized", "detail": "Startup may have failed."}
    try:
        prompt = _messages_to_prompt(body.messages)
        prompt_ids = _encode_fn(prompt)
    except Exception as e:
        return {"error": str(e), "detail": traceback.format_exc()}

    try:
        if body.stream:

        async def gen() -> AsyncIterator[str]:
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            buffer: List[int] = []
            async for tid in _engine.generate_tokens(
                prompt_ids,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                eos_token_id=_eos_token_id,
            ):
                buffer.append(tid)
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": _decode_fn([tid])},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            end = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    out: List[int] = []
    async for tid in _engine.generate_tokens(
        prompt_ids,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        eos_token_id=_eos_token_id,
    ):
        out.append(tid)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    content = _decode_fn(out) if out else ""
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out), "total_tokens": len(prompt_ids) + len(out)},
    }


def main() -> None:
    import uvicorn

    host = os.environ.get("NANO_INFER_HOST", "127.0.0.1")
    port = int(os.environ.get("NANO_INFER_PORT", "8000"))
    uvicorn.run("nano_infer.entrypoints.openai_api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
