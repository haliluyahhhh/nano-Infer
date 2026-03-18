"""
离线推理 CLI — 不启动 API，直接对 prompt 做推理。
便于本地调试：python -m nano_infer.entrypoints.offline_run --model-path /path/to/model --prompt "你好"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from nano_infer.config import config_from_env
from nano_infer.engine.async_llm_engine import AsyncLLMEngine, build_engine
from nano_infer.engine.sequence import SequenceStatus
from nano_infer.tokenizer import get_tokenizer


def _run_sync(
    engine: AsyncLLMEngine,
    prompt_ids: List[int],
    max_tokens: int,
    temperature: float,
    eos_token_id: int | None,
    decode_fn,
    verbose: bool,
) -> str:
    """同步执行推理，返回生成的文本。"""
    seq = engine.create_sequence(
        prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )
    engine.add_sequence(seq)

    out: List[int] = []
    step_count = 0
    while seq.status.value != "finished":
        prev_len = len(seq.output_ids)
        engine.step()
        step_count += 1
        for j in range(prev_len, len(seq.output_ids)):
            tid = seq.output_ids[j]
            out.append(tid)
            if verbose:
                print(f"[step {step_count}] token={tid} -> {decode_fn([tid])!r}", flush=True)
        if not engine._running and not engine._waiting and seq.status != SequenceStatus.FINISHED:
            break

    return decode_fn(out) if out else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="nano-Infer 离线推理（不启动 API，便于调试）")
    parser.add_argument("--model-path", "-m", help="模型目录（含 config.json）")
    parser.add_argument("--prompt", "-p", default="你好", help="输入 prompt")
    parser.add_argument("--max-tokens", "-n", type=int, default=32, help="最大生成 token 数")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="采样温度")
    parser.add_argument("--device", default=None, help="设备 (cpu/cuda/cuda:0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="逐 token 打印")
    parser.add_argument("--debug", action="store_true", help="等同 --verbose，便于 pdb 等调试")
    args = parser.parse_args()

    if args.model_path:
        os.environ["NANO_INFER_MODEL_PATH"] = args.model_path
        os.environ["NANO_INFER_MODEL"] = "llama3"  # 或由 detect 自动识别
    if args.device:
        os.environ["NANO_INFER_DEVICE"] = args.device

    cfg = config_from_env()
    if not cfg.model_path:
        print("错误: 未指定 model_path。请使用 --model-path 或设置 NANO_INFER_MODEL_PATH", file=sys.stderr)
        return 1

    encode_fn, decode_fn, eos_token_id = get_tokenizer(cfg.model_path, vocab_size=cfg.vocab_size)
    prompt_ids = encode_fn(args.prompt)
    if not prompt_ids:
        print("错误: tokenizer 返回空序列", file=sys.stderr)
        return 1

    verbose = args.verbose or args.debug
    if verbose:
        print(f"model_path={cfg.model_path} device={cfg.device} dtype={cfg.dtype}")
        print(f"prompt={args.prompt!r} -> {len(prompt_ids)} tokens")
        print("---")

    try:
        engine, _ = build_engine(cfg)
    except Exception as e:
        print(f"引擎初始化失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    try:
        result = _run_sync(
            engine,
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            eos_token_id=eos_token_id,
            decode_fn=decode_fn,
            verbose=verbose,
        )
    except Exception as e:
        print(f"推理失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    if verbose:
        print("---")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
