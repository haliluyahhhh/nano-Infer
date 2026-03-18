"""
OpenAI 兼容 HTTP 入口（Chat Completions + 流式 SSE）。
不含框架特定逻辑；Tokenizer 为极简占位，生产请接 HuggingFaceTokenizer。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nano_infer.config import EngineConfig, config_from_env
from nano_infer.engine.async_llm_engine import AsyncLLMEngine, build_engine

app = FastAPI(title="nano-Infer", version="0.1.0")
_engine: AsyncLLMEngine | None = None


def _encode_prompt(text: str, vocab_size: int = 50257) -> List[int]:
    """占位：字符 → id，仅用于无 tokenizer 时打通链路。"""
    return [min(vocab_size - 1, ord(c) % vocab_size) for c in text[:2048]]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "dummy"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=64, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False


@app.on_event("startup")
def _startup() -> None:
    global _engine
    cfg = config_from_env()
    if os.environ.get("NANO_INFER_MODEL"):
        cfg.model_name = os.environ["NANO_INFER_MODEL"]
    _engine, _ = build_engine(cfg)


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
    assert _engine is not None
    prompt = _messages_to_prompt(body.messages)
    prompt_ids = _encode_prompt(prompt)

    if body.stream:

        async def gen() -> AsyncIterator[str]:
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            async for tid in _engine.generate_tokens(
                prompt_ids, max_tokens=body.max_tokens, temperature=body.temperature
            ):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"[{tid}]"},
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
        prompt_ids, max_tokens=body.max_tokens, temperature=body.temperature
    ):
        out.append(tid)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    content = "".join(f"[{t}]" for t in out)
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
