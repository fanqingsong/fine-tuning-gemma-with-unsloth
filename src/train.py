#!/usr/bin/env python3
"""
使用 Unsloth + LoRA 微调 Gemma 3 270M（逻辑来自 live_demo.ipynb）。

微调流程概览（建议按 main() 中的顺序阅读）：
  1. 加载基座模型与分词器（可选 4bit 量化以省显存）
  2. 加载并清洗训练数据，统一成 Gemma 对话格式
  3. 用 chat template 把对话转成模型看到的「一整段文本」
  4. 在注意力/MLP 等层挂上 LoRA 适配器（只训练少量新增参数）
  5. SFT（监督微调）：用 SFTTrainer 在文本上做 next-token 预测
  6. train_on_responses_only：只在 assistant 回复部分计算 loss（不训练用户问题）
  7. 保存 LoRA 权重；可选合并 LoRA 到完整模型便于部署
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 保证同目录下的 hf_setup 可被导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
# 必须在 unsloth / transformers / trl 之前：清理空的 HF 环境变量，避免下载失败
import hf_setup  # noqa: E402  — sanitize HF env before unsloth
# unsloth 会 patch 底层库以加速训练；导入顺序有要求，故放在 trl 之前
import unsloth  # noqa: F401,E402  — must precede trl / transformers / peft

from datasets import Dataset, load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth import FastModel  # noqa: E402
from unsloth.chat_templates import (  # noqa: E402
    get_chat_template,
    train_on_responses_only,
)

# 系统提示：定义模型在对话中的「角色」，会写入每条训练样本
SYSTEM_PROMPT = (
    "You are a master storyteller. Write a short, imaginative story based on "
    "the user's request. The story should be concise and suitable for a general audience."
)
# 默认 Hugging Face 上的示例数据集（也可换成本地 jsonl，见 --dataset）
PREGENERATED_DATASET_NAME = "chongcht/synthetic-creative-writing"


def parse_args() -> argparse.Namespace:
    """命令行超参数：模型、数据、LoRA、优化器与训练步数等。"""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M with Unsloth")
    parser.add_argument(
        "--model-name",
        default=hf_setup.model_name_from_env(),
        help="Base model on Hugging Face",
    )
    parser.add_argument(
        "--dataset",
        default=PREGENERATED_DATASET_NAME,
        help="HF dataset id or local json/jsonl path",
    )
    parser.add_argument("--dataset-split", default="train")
    # 单条样本最大 token 数；过长会被截断，过短浪费算力
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="270M fits in memory without 4-bit; notebook trains in full precision",
    )
    # LoRA 秩 r：越大可表达能力越强，但可训练参数和显存也更多
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Defaults to 2 * lora-r (notebook heuristic)",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    # 梯度累积：有效 batch = batch_size × accumulation_steps（显存不够时增大此项）
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    # 固定训练步数；若指定 num-train-epochs 则按 epoch 跑满整个数据集
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
    # 合并 LoRA 到基座权重，得到单目录完整模型（体积大，推理更简单）
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--merged-dir", default="outputs/merged")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_training_dataset(path_or_id: str, split: str) -> Dataset:
    """
    加载训练集：本地 .json/.jsonl，或 Hugging Face dataset id。
    返回 Hugging Face Dataset，后续用 .map / .filter 做预处理。
    """
    local = Path(path_or_id)
    if local.exists():
        suffix = local.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files=str(local), split="train")
        return load_dataset(str(local), split=split)
    return load_dataset(path_or_id, split=split)


def format_for_gemma(example: dict) -> dict:
    """
    把各种常见字段名统一成 Gemma3 需要的 conversations 列表：
      system / user / assistant 三轮消息。

    若缺少 prompt 或 response，返回 conversations=None，后续会被 filter 掉。
    """
    prompt_text = example.get("prompt") or ""
    response_text = example.get("response") or ""

    # ShareGPT 风格：conversations 里 role 为 user / assistant
    if not prompt_text and "conversations" in example:
        for message in example["conversations"]:
            if message["role"] == "user":
                prompt_text = message["content"]
            elif message["role"] in {"assistant", "model"}:
                response_text = message["content"]

    # OpenAI 风格：messages 数组
    if not prompt_text and "messages" in example:
        for message in example["messages"]:
            if message["role"] == "user":
                prompt_text = message["content"]
            elif message["role"] in {"assistant", "model"}:
                response_text = message["content"]

    # Alpaca 风格：instruction + input -> 用户侧；output -> 模型要学的回复
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
    """过滤合成数据里标记为 error 的坏样本（response 字符串含 error 则丢弃）。"""
    response = example.get("response")
    if isinstance(response, str):
        return "error" not in response.lower()
    return True


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    # LoRA 缩放系数，常用经验：alpha = 2 * r
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r * 2

    # ---------- 1. 加载基座模型 ----------
    # full_finetuning=False：后面只加 LoRA，不更新全部 270M 参数
    model, tokenizer = hf_setup.load_fast_model(
        FastModel,
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        token=token,
    )

    # ---------- 2. 数据集：加载 -> 清洗 -> 统一格式 ----------
    dataset = load_training_dataset(args.dataset, args.dataset_split)
    if "response" in dataset.column_names:
        dataset = dataset.filter(has_error_response)

    dataset = dataset.map(format_for_gemma, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda example: example.get("conversations") is not None)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Gemma3 专用 chat template（控制 <start_of_turn> 等特殊 token）
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

    def formatting_prompts_func(examples):
        """
        SFT 需要一列纯文本 `text`：把多轮对话渲染成模型训练时看到的字符串。
        add_generation_prompt=False：assistant 回复已在模板里，用于 teacher forcing。
        """
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

    # ---------- 3. LoRA：在指定线性层旁路低秩矩阵，只训练这些增量权重 ----------
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
        use_gradient_checkpointing="unsloth",  # 用算力换显存
        random_state=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 4. SFT 训练配置 ----------
    # 本质：对 text 做因果语言建模，预测下一个 token
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

    # ---------- 5. 只在 model 回复段算 loss ----------
    # 用户/系统 turn 的 token 不参与 loss，避免模型「学会复读问题」
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    trainer.train()

    # LoRA adapter + tokenizer 配置（推理时用同一 base model + 加载此目录）
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
