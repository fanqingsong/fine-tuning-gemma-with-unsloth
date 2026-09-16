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
```

Build the image (based on the official `unsloth/unsloth` image, pulled via a China mirror):

```bash
docker compose build
```

## Train

Default run (Gemma 3 270M Instruct, `chongcht/synthetic-creative-writing`, 100 steps):

```bash
docker compose run --rm train
```

Smoke test with the bundled sample dataset:

```bash
docker compose run --rm train --dataset data/sample.jsonl --max-steps 10 --max-samples 3
```

Notebook-aligned flags:

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

Full epoch instead of a step cap:

```bash
docker compose run --rm train --num-train-epochs 1
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

Custom prompt (same default theme as the notebook):

```bash
docker compose run --rm infer --prompt "A city where shadows have a life of their own." --max-new-tokens 1024
```

Compare the adapter against the base model:

```bash
docker compose run --rm infer --compare-base
```

If `outputs/lora` exists, it is loaded automatically. Otherwise the base model is used.

## Optional: regenerate synthetic stories

The notebook can call the Gemini Batch API. This project defaults to the public pre-generated Hub dataset. To rebuild from `data/story_themes.json`:

```bash
# requires GOOGLE_API_KEY
docker compose run --rm --entrypoint python train src/generate_data.py
```

Then train on the local file:

```bash
docker compose run --rm train --dataset data/synthetic-creative-writing.jsonl
```

## Local run (no Docker)

Install Unsloth per the [official install guide](https://docs.unsloth.ai/get-started/installing-+-updating), then:

```bash
python src/train.py --dataset data/sample.jsonl --max-steps 10
python src/infer.py --prompt "A robot who discovers music for the first time."
```

## Dataset formats

Hugging Face datasets or local `.json` / `.jsonl` files are supported. The original notebook uses `prompt` / `response`:

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
