#!/usr/bin/env python3
"""Run inference with a Gemma 3 270M LoRA adapter trained by src/train.py."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_setup  # noqa: E402  — sanitize HF env before unsloth
import unsloth  # noqa: F401,E402  — must precede transformers / peft

import torch  # noqa: E402
from transformers import TextStreamer, set_seed  # noqa: E402
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
        action="append",
        default=None,
        help="User prompt. Repeat the flag to run several prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSONL/JSON/TXT of eval prompts (see data/eval_prompts.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write one JSON object per prompt. Defaults to outputs/compare.jsonl "
        "when --prompts-file is set.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sampling vs greedy. Default greedy so adapter vs base is comparable.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Applied to both the adapter and the base model when comparing.",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also generate from the base model with the same precision and decoding.",
    )
    return parser.parse_args()


def load_prompt_items(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            items.append(_normalize_prompt_item(raw, line_no))
        return items
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [_normalize_prompt_item(item, i) for i, item in enumerate(raw, start=1)]
        return [_normalize_prompt_item(raw, 1)]
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if line and not line.startswith("#"):
            items.append({"prompt": line, "split": None})
    if not items:
        raise ValueError(f"No prompts found in {path}")
    return items


def _normalize_prompt_item(raw, line_no: int) -> dict:
    if isinstance(raw, str):
        prompt = raw.strip()
        split = None
    elif isinstance(raw, dict):
        prompt = (raw.get("prompt") or raw.get("theme") or "").strip()
        split = raw.get("split")
    else:
        raise ValueError(f"Prompt item {line_no} must be a string or object")
    if not prompt:
        raise ValueError(f"Prompt item {line_no} is missing prompt text")
    return {"prompt": prompt, "split": split}


def collect_eval_items(args: argparse.Namespace) -> list[dict]:
    items: list[dict] = []
    if args.prompts_file:
        items.extend(load_prompt_items(Path(args.prompts_file)))
    if args.prompt:
        items.extend({"prompt": text, "split": None} for text in args.prompt)
    if not items:
        items.append(
            {
                "prompt": "A city where shadows have a life of their own.",
                "split": None,
            }
        )
    return items


def generation_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


def generate(model, tokenizer, prompt: str, gen_kwargs: dict, stream: bool) -> str:
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
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None
    outputs = model.generate(
        **inputs,
        **gen_kwargs,
        streamer=streamer,
    )
    return tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )[0]


def load_model(model_name: str, args: argparse.Namespace, token: str | None):
    model, tokenizer = hf_setup.load_fast_model(
        FastModel,
        model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        token=token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    return model, tokenizer


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_prompts(model, tokenizer, items: list[dict], args: argparse.Namespace, label: str) -> list[str]:
    texts = []
    gen_kwargs = generation_kwargs(args)
    for index, item in enumerate(items, start=1):
        split = f" [{item['split']}]" if item.get("split") else ""
        print(f"\n=== {label} ({index}/{len(items)}){split} ===")
        print(f"Prompt: {item['prompt']}\n")
        set_seed(args.seed)
        texts.append(generate(model, tokenizer, item["prompt"], gen_kwargs, stream=True))
        print()
    return texts


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    adapter_dir = Path(args.adapter_dir)
    items = collect_eval_items(args)
    output_path = args.output
    if args.prompts_file and not output_path:
        output_path = "outputs/compare.jsonl"

    if args.compare_base and not adapter_dir.exists():
        raise SystemExit(
            f"No adapter at {adapter_dir}. Train first, or omit --compare-base."
        )

    has_adapter = adapter_dir.exists()
    if not has_adapter:
        print(f"No adapter at {adapter_dir}; using base model {args.model_name}")

    adapter_outputs: list[str] | None = None
    if has_adapter:
        model, tokenizer = load_model(str(adapter_dir), args, token)
        adapter_outputs = run_prompts(model, tokenizer, items, args, f"Adapter {adapter_dir}")
        unload(model)
    else:
        tokenizer = None

    base_outputs: list[str] | None = None
    if args.compare_base or not has_adapter:
        base_model, base_tokenizer = load_model(args.model_name, args, token)
        tokenizer = tokenizer or base_tokenizer
        base_outputs = run_prompts(
            base_model, tokenizer, items, args, f"Base {args.model_name}"
        )
        unload(base_model)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for index, item in enumerate(items):
                record = {
                    "prompt": item["prompt"],
                    "split": item.get("split"),
                    "adapter_dir": str(adapter_dir) if has_adapter else None,
                    "base_model": args.model_name,
                    "load_in_4bit": args.load_in_4bit,
                    "seed": args.seed,
                    "do_sample": args.do_sample,
                    "temperature": args.temperature if args.do_sample else None,
                    "top_p": args.top_p if args.do_sample else None,
                    "max_new_tokens": args.max_new_tokens,
                    "adapter": adapter_outputs[index] if adapter_outputs else None,
                    "base": base_outputs[index] if base_outputs else None,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(items)} comparison rows to {out}")


if __name__ == "__main__":
    main()
