"""Hugging Face env sanitization. Import before unsloth / transformers / trl."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_NAME = "unsloth/gemma-3-270m-it"

# An empty HF_ENDPOINT makes huggingface_hub request "" instead of huggingface.co.
# Unsloth then fails with "No config file found".
for _key in ("HF_ENDPOINT", "HF_TOKEN"):
    _value = os.environ.get(_key)
    if _value is not None and not str(_value).strip():
        os.environ.pop(_key, None)


def model_name_from_env() -> str:
    return (os.environ.get("MODEL_NAME") or "").strip() or DEFAULT_MODEL_NAME


def load_fast_model(fast_model_cls, model_name: str, **kwargs):
    try:
        return fast_model_cls.from_pretrained(model_name=model_name, **kwargs)
    except Exception as exc:
        endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
        local = Path(model_name)
        extra = ""
        if local.exists():
            extra = (
                f" Path exists; config.json="
                f"{'yes' if (local / 'config.json').is_file() else 'no'}."
            )
        raise RuntimeError(
            f"Failed to load model {model_name!r} (HF hub={endpoint}).{extra} "
            "If Hugging Face is blocked, set HF_ENDPOINT=https://hf-mirror.com "
            "in .env (do not leave it empty), or set UNSLOTH_USE_MODELSCOPE=1 "
            "and install modelscope. Gated Gemma weights need a valid HF_TOKEN."
        ) from exc
