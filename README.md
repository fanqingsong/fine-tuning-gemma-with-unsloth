# Fine-tune Gemma 3 with Unsloth

Python scripts for conversational QLoRA fine-tuning of Gemma 3, packaged with Docker Compose. The training flow follows the official Unsloth Gemma 3 (4B) notebook.

## Prerequisites

- Docker Engine + Compose v2 (`docker compose`)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Enough VRAM for the chosen model (Gemma 3 4B QLoRA typically needs about 8–12 GB)

## Setup

```bash
cp .env.example .env
# Optional: set HF_TOKEN, HF_ENDPOINT (e.g. https://hf-mirror.com), MODEL_NAME
```

Build the image (based on the official `unsloth/unsloth` image, pulled via a China mirror):

```bash
docker compose build
```

## Train

Default run (Gemma 3 4B Instruct, FineTome-100k, 60 steps):

```bash
docker compose run --rm train
```

Smoke test with the bundled sample dataset:

```bash
docker compose run --rm train --dataset data/sample.jsonl --max-steps 10 --max-samples 3
```

Common flags:

```bash
docker compose run --rm train \
  --model-name unsloth/gemma-3-4b-it \
  --dataset mlabonne/FineTome-100k \
  --max-steps 60 \
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

The LoRA adapter is written to `outputs/lora`.

## Infer

```bash
docker compose run --rm infer
```

Custom prompt:

```bash
docker compose run --rm infer --prompt "用一句话解释 QLoRA" --max-new-tokens 128
```

If `outputs/lora` exists, it is loaded automatically. Otherwise the base model is used.

## Local run (no Docker)

Install Unsloth per the [official install guide](https://docs.unsloth.ai/get-started/installing-+-updating), then:

```bash
python src/train.py --dataset data/sample.jsonl --max-steps 10
python src/infer.py --prompt "Hello"
```

## Dataset formats

Hugging Face datasets or local `.json` / `.jsonl` files are supported:

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

```json
{"instruction": "...", "input": "", "output": "..."}
```

`messages` is accepted as an alias of `conversations`.
