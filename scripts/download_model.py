#!/usr/bin/env python3
"""
从 ModelScope 下载模型权重到本地。
用法: python scripts/download_model.py [--model 模型ID] [--output 输出目录]

示例:
  python scripts/download_model.py
  python scripts/download_model.py --model Qwen/Qwen2-1.5B-Instruct --output ./models/qwen2-1.5b
  python scripts/download_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output ./models/tinyllama
"""

from __future__ import annotations

import argparse
import sys


# 推荐模型（≤7B，Llama 架构兼容）
DEFAULT_MODELS = {
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",      # ~1.5B, ~3GB
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",            # 7B, ~14GB
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B, ~2GB
}


def main() -> int:
    parser = argparse.ArgumentParser(description="从 ModelScope 下载模型")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODELS["qwen2-1.5b"],
        help=f"模型 ID，默认 Qwen2-1.5B。可选: {', '.join(DEFAULT_MODELS.keys())}",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./models/qwen2-1.5b",
        help="本地保存目录",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出预置模型 ID",
    )
    args = parser.parse_args()

    if args.list:
        for name, mid in DEFAULT_MODELS.items():
            print(f"  {name}: {mid}")
        return 0

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("请先安装 modelscope: pip install modelscope")
        return 1

    model_id = DEFAULT_MODELS.get(args.model, args.model)
    print(f"正在从 ModelScope 下载: {model_id}")
    print(f"保存路径: {args.output}")

    try:
        path = snapshot_download(
            model_id,
            local_dir=args.output,
            local_files_only=False,
        )
        print(f"下载完成: {path}")
        return 0
    except Exception as e:
        print(f"下载失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
