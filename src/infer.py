#!/usr/bin/env python3
"""Run inference with a Gemma 3 270M LoRA adapter trained by src/train.py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_setup  # noqa: E402  — sanitize HF env before unsloth
import unsloth  # noqa: F401,E402  — must precede transformers / peft

from transformers import TextStreamer  # noqa: E402
from unsloth import FastModel  # noqa: E402
from unsloth.chat_templates import get_chat_template  # noqa: E402

SYSTEM_PROMPT = (
    "You are a master storyteller. Write a short, imaginative story based on "
    "the user's request. The story should be concise and suitable for a general audience."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with a fine-tuned Gemma 3 270M adapter")
    parser.add_argument(
        "--model-name",
        default=hf_setup.model_name_from_env(),
    )
    parser.add_argument("--adapter-dir", default="outputs/lora")
    parser.add_argument(
        "--prompt",
        default="A city where shadows have a life of their own.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also generate from the base model for a side-by-side check",
    )
    return parser.parse_args()


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix("<bos>")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    return tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )[0]


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    adapter_dir = Path(args.adapter_dir)
    model_name = str(adapter_dir) if adapter_dir.exists() else args.model_name

    model, tokenizer = hf_setup.load_fast_model(
        FastModel,
        model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        token=token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

    print("--- Fine-Tuned Model Output ---")
    generate(model, tokenizer, args.prompt, args.max_new_tokens)

    if args.compare_base:
        print("\n--- Base Model Output ---")
        base_model, _ = hf_setup.load_fast_model(
            FastModel,
            args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            token=token,
        )
        generate(base_model, tokenizer, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
