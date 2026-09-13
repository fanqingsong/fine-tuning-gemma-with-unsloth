#!/usr/bin/env python3
"""Run inference with a Gemma 3 LoRA adapter trained by src/train.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with a fine-tuned Gemma 3 adapter")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "unsloth/gemma-3-4b-it"),
    )
    parser.add_argument("--adapter-dir", default="outputs/lora")
    parser.add_argument("--prompt", default="Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    adapter_dir = Path(args.adapter_dir)

    model_name = str(adapter_dir) if adapter_dir.exists() else args.model_name
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        token=token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    FastModel.for_inference(model)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": args.prompt}],
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    print(tokenizer.batch_decode(outputs)[0])


if __name__ == "__main__":
    main()
