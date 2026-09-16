# Fine-tune Gemma 3 270M with Unsloth

Python scripts for LoRA fine-tuning of **Gemma 3 270M Instruct**, packaged with Docker Compose. Training and inference follow [thomas-chong/fine-tuning-gemma-with-unsloth](https://github.com/thomas-chong/fine-tuning-gemma-with-unsloth) (`live_demo.ipynb`): creative-writing adapters on `unsloth/gemma-3-270m-it`.

## Prerequisites

- Docker Engine + Compose v2 (`docker compose`)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Gemma 3 270M LoRA is intended for a consumer GPU (the original demo targets a free Colab T4)

## Setup

```bash
cp .env.example .env
# Edit .env: set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is unreachable.
# Do not leave HF_ENDPOINT empty. Optional: HF_TOKEN, MODEL_NAME.
# GOOGLE_API_KEY is only needed if you regenerate stories with src/generate_data.py.
```

Build the image (based on the official `unsloth/unsloth` image, pulled via a China mirror):

```bash
docker compose build
```

## Train

Default run (Gemma 3 270M Instruct, Hub dataset `chongcht/synthetic-creative-writing`, 100 steps):

```bash
docker compose run --rm train
```

Train on the bundled local stories (no Hub download, no `GOOGLE_API_KEY`):

```bash
docker compose run --rm train --dataset data/synthetic-creative-writing.jsonl
```

`data/synthetic-creative-writing.jsonl` is one short story per theme in `data/story_themes.json`, in the same `prompt` / `response` jsonl format as `src/generate_data.py`. Use it when the Hub dataset is unreachable or you do not have a Gemini key.

Smoke test with the tiny sample file:

```bash
docker compose run --rm train --dataset data/sample.jsonl --max-steps 10 --max-samples 3
```

Notebook-aligned flags (Hub dataset):

```bash
docker compose run --rm train \
  --model-name unsloth/gemma-3-270m-it \
  --dataset chongcht/synthetic-creative-writing \
  --max-steps 100 \
  --lora-r 16 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --output-dir outputs/lora
```

Same flags with the local jsonl:

```bash
docker compose run --rm train \
  --model-name unsloth/gemma-3-270m-it \
  --dataset data/synthetic-creative-writing.jsonl \
  --max-steps 100 \
  --lora-r 16 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --output-dir outputs/lora
```

Full epoch instead of a step cap (add `--dataset data/synthetic-creative-writing.jsonl` to stay on the local file):

```bash
docker compose run --rm train --dataset data/synthetic-creative-writing.jsonl --num-train-epochs 1
```

Save a merged 16-bit model after training:

```bash
docker compose run --rm train --save-merged --merged-dir outputs/merged
```

The LoRA adapter is written to `outputs/lora`. Training uses full precision (`--no-load-in-4bit`) because 270M does not need QLoRA; pass `--load-in-4bit` if you want quantization anyway.

## Infer

```bash
docker compose run --rm infer
```

Custom prompt (same default theme as the notebook). Repeat `--prompt` to run several:

```bash
docker compose run --rm infer --prompt "A city where shadows have a life of their own." --max-new-tokens 1024
```

If `outputs/lora` exists, it is loaded automatically. Otherwise the base model is used.

## Compare adapter vs base

`--compare-base` runs the **same** prompts on the LoRA adapter and the base model, with matching precision (`--load-in-4bit` / `--no-load-in-4bit`) and matching decoding (greedy by default, `--seed 3407`). Training loss is not a quality score; read the paired stories.

Single prompt:

```bash
docker compose run --rm infer --compare-base \
  --prompt "A city where shadows have a life of their own."
```

Batch file (bundled mix of training themes and held-out themes). Results are written to `outputs/compare.jsonl` unless you pass `--output`:

```bash
docker compose run --rm infer --compare-base --prompts-file data/eval_prompts.jsonl
```

Each JSONL row has `prompt`, `split` (`in_distribution` or `held_out`), `adapter`, and `base`. Judge by whether the adapter writes a concise on-theme story (not an outline or refusal), stays short, and still works on `held_out` prompts instead of echoing the training set.

`--prompts-file` accepts:

- `.jsonl` — `{"prompt": "...", "split": "held_out"}` per line (`split` is optional)
- `.json` — a string list or a list of the same objects
- `.txt` — one prompt per line (`#` comments allowed)

Sampling (off by default so runs are comparable):

```bash
docker compose run --rm infer --compare-base --prompts-file data/eval_prompts.jsonl \
  --do-sample --temperature 0.7 --seed 3407
```

Use `--load-in-4bit` only if you want **both** models quantized; 270M is intended to run in full precision.

## Dataset: local stories vs Gemini

You do **not** need `GOOGLE_API_KEY` to train. Prefer this order:

1. **Bundled local file** — `data/synthetic-creative-writing.jsonl` (same jsonl schema as `src/generate_data.py`: one `{"prompt", "response"}` object per line).
2. **Hub dataset** — `chongcht/synthetic-creative-writing` (default `--dataset` if you omit the flag).
3. **Gemini rebuild** — only if you want new stories from `data/story_themes.json`.

```bash
# requires GOOGLE_API_KEY in .env
docker compose run --rm --entrypoint python train src/generate_data.py
```

That overwrites `data/synthetic-creative-writing.jsonl` (default `--multiplier 8`, so each theme is repeated). Then train as above with `--dataset data/synthetic-creative-writing.jsonl`.

## Local run (no Docker)

Install Unsloth per the [official install guide](https://docs.unsloth.ai/get-started/installing-+-updating), then:

```bash
python src/train.py --dataset data/synthetic-creative-writing.jsonl --max-steps 100
python src/infer.py --prompt "A robot who discovers music for the first time."
python src/infer.py --compare-base --prompts-file data/eval_prompts.jsonl
```

## Dataset formats

Hugging Face datasets or local `.json` / `.jsonl` files are supported. The bundled `data/synthetic-creative-writing.jsonl` and `data/sample.jsonl` use `prompt` / `response`:

```json
{"prompt": "A city where shadows have a life of their own.", "response": "..."}
```

Also accepted:

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

```json
{"instruction": "...", "input": "", "output": "..."}
```

Each example is wrapped with the notebook's storyteller system prompt and the Gemma 3 chat template (`gemma3`).
