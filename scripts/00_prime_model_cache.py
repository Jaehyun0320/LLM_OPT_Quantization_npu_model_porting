#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL_IDS = [
    "google/gemma-4-E2B",
    "google/gemma-3-270m-it",
    "google/gemma-3-270m",
    "google/gemma-3-1b-it",
]


def load_tokenizer(model_id):
    if model_id == "google/gemma-4-E2B":
        return AutoTokenizer.from_pretrained(
            model_id,
            extra_special_tokens={"video_token": "<|video|>"},
        )
    return AutoTokenizer.from_pretrained(model_id)


def parse_model_ids(raw_model_ids):
    return [item.strip() for item in raw_model_ids.split(",") if item.strip()]


def safe_model_slug(model_id):
    return model_id.replace("/", "__").replace(":", "_")


def prime_model_cache(model_id, local_dir_root):
    local_dir = None
    if local_dir_root is not None:
        local_dir = Path(local_dir_root) / safe_model_slug(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir) if local_dir is not None else None,
    )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = load_tokenizer(model_id)

    return {
        "model_id": model_id,
        "snapshot_path": snapshot_path,
        "local_dir": str(local_dir) if local_dir is not None else None,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_length": len(tokenizer),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-ids",
        type=str,
        default=",".join(DEFAULT_MODEL_IDS),
        help="Comma-separated Hugging Face model ids to pre-download.",
    )
    parser.add_argument(
        "--local-dir-root",
        type=str,
        default=None,
        help=(
            "Optional directory for explicit snapshots. If omitted, the normal "
            "Hugging Face cache is used."
        ),
    )
    parser.add_argument("--output", type=str, default="results/cache_prime.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    errors = []

    for model_id in parse_model_ids(args.model_ids):
        try:
            rows.append(
                {
                    "success": True,
                    **prime_model_cache(
                        model_id=model_id,
                        local_dir_root=args.local_dir_root,
                    ),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "success": False,
                    "model_id": model_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "success": len(errors) == 0,
        "models": rows,
        "errors": errors,
        "note": (
            "This primes Hugging Face's on-disk cache. It does not keep a loaded "
            "GPU model alive across scripts."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
