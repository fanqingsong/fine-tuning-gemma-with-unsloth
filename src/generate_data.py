#!/usr/bin/env python3
"""Generate synthetic creative-writing data via the Gemini Batch API.

Ported from live_demo.ipynb in thomas-chong/fine-tuning-gemma-with-unsloth.
Default training uses the public pre-generated dataset; run this only if you
want to rebuild stories from data/story_themes.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a master storyteller. Write a short, imaginative story based on "
    "the user's request. The story should be concise and suitable for a general "
    "audience. The story should not exceed 2048 tokens."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stories with Gemini Batch API")
    parser.add_argument("--themes", default="data/story_themes.json")
    parser.add_argument("--output", default="data/synthetic-creative-writing.jsonl")
    parser.add_argument("--multiplier", type=int, default=8)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--requests-file", default="outputs/synthetic_story_requests.jsonl")
    return parser.parse_args()


def create_batch_requests(themes: list[str], system_prompt: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i, theme in enumerate(themes):
            request = {
                "key": f"request_{i}",
                "request": {
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"parts": [{"text": theme}]}],
                    "generationConfig": {
                        "maxOutputTokens": 2048,
                        "temperature": round(random.uniform(0.7, 1.0), 2),
                    },
                },
            }
            handle.write(json.dumps(request) + "\n")
    return path


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is required to generate a new dataset")

    from google import genai
    from google.genai import types

    themes = json.loads(Path(args.themes).read_text(encoding="utf-8"))
    expanded = themes * args.multiplier
    print(f"Loaded {len(themes)} themes; generating {len(expanded)} stories")

    requests_file = create_batch_requests(expanded, SYSTEM_PROMPT, Path(args.requests_file))
    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(
        file=str(requests_file),
        config=types.UploadFileConfig(display_name="my-batch-requests", mime_type="jsonl"),
    )
    batch_job = client.batches.create(model=args.model, src=uploaded_file.name)
    print(f"Batch job created: {batch_job.name}. Polling for results...")

    while True:
        batch_job = client.batches.get(name=batch_job.name)
        if batch_job.state.name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            break
        print(f"Job not finished. Current state: {batch_job.state.name}. Waiting 30 seconds...")
        time.sleep(30)

    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        error = getattr(batch_job, "error", None)
        raise SystemExit(f"Batch job did not succeed: {batch_job.state.name} {error}")

    file_content = client.files.download(file=batch_job.dest.file_name).decode("utf-8")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for line in file_content.splitlines():
            if not line:
                continue
            parsed = json.loads(line)
            try:
                request_index = int(parsed.get("key", "").split("_")[1])
                prompt = themes[request_index % len(themes)]
            except (IndexError, ValueError):
                prompt = "Unknown prompt"
            try:
                response_text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                response_text = "Error: Could not parse response."
            handle.write(json.dumps({"prompt": prompt, "response": response_text}, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} examples to {output_path}")


if __name__ == "__main__":
    main()
