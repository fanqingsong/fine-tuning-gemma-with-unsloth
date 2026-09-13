#!/usr/bin/env python3
"""Fine-tune Gemma 3 270M with Unsloth LoRA (ported from live_demo.ipynb)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    train_on_responses_only,
)

SYSTEM_PROMPT = (
    "You are a master storyteller. Write a short, imaginative story based on "
    "the user's request. The story should be concise and suitable for a general audience."
)
PREGENERATED_DATASET_NAME = "chongcht/synthetic-creative-writing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M with Unsloth")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "unsloth/gemma-3-270m-it"),
        help="Base model on Hugging Face",
    )
    parser.add_argument(
        "--dataset",
        default=PREGENERATED_DATASET_NAME,
        help="HF dataset id or local json/jsonl path",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="270M fits in memory without 4-bit; notebook trains in full precision",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Defaults to 2 * lora-r (notebook heuristic)",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="If set, overrides --max-steps for a full epoch run",
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", default="outputs/lora")
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--merged-dir", default="outputs/merged")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_training_dataset(path_or_id: str, split: str) -> Dataset:
    local = Path(path_or_id)
    if local.exists():
        suffix = local.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files=str(local), split="train")
        return load_dataset(str(local), split=split)
    return load_dataset(path_or_id, split=split)


def format_for_gemma(example: dict) -> dict:
    prompt_text = example.get("prompt") or ""
    response_text = example.get("response") or ""

    if not prompt_text and "conversations" in example:
        for message in example["conversations"]:
            if message["role"] == "user":
                prompt_text = message["content"]
            elif message["role"] in {"assistant", "model"}:
                response_text = message["content"]

    if not prompt_text and "messages" in example:
        for message in example["messages"]:
            if message["role"] == "user":
                prompt_text = message["content"]
            elif message["role"] in {"assistant", "model"}:
                response_text = message["content"]

    if not prompt_text:
        instruction = example.get("instruction") or example.get("question") or ""
        extra = example.get("input") or ""
        prompt_text = instruction if not extra else f"{instruction}\n{extra}".strip()
        response_text = (
            example.get("output") or example.get("answer") or response_text
        )

    if not prompt_text or not response_text:
        return {"conversations": None}

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
    }


def has_error_response(example: dict) -> bool:
    response = example.get("response")
    if isinstance(response, str):
        return "error" not in response.lower()
    return True


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r * 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        token=token,
    )

    dataset = load_training_dataset(args.dataset, args.dataset_split)
    if "response" in dataset.column_names:
        dataset = dataset.filter(has_error_response)

    dataset = dataset.map(format_for_gemma, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda example: example.get("conversations") is not None)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=str(output_dir),
        report_to="none",
    )
    if args.num_train_epochs is not None:
        sft_kwargs["num_train_epochs"] = args.num_train_epochs
        sft_kwargs["max_steps"] = -1
    else:
        sft_kwargs["max_steps"] = args.max_steps

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter to {output_dir}")

    if args.save_merged:
        merged_dir = Path(args.merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged_dir), tokenizer)
        print(f"Saved merged model to {merged_dir}")


if __name__ == "__main__":
    main()
