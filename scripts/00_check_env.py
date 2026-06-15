#!/usr/bin/env python3
"""Check local environment for the Gemma optimization assignment.

This script should be cheap to run. By default it only inspects installed
packages and hardware. Use --check-model-access when you also want to verify
that the Hugging Face checkpoint metadata can be reached.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "google/gemma-4-E2B"
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "env.json"

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "optimum",
    "onnx",
    "onnxruntime",
    "datasets",
    "evaluate",
    "pandas",
    "numpy",
    "psutil",
    "sentencepiece",
    "safetensors",
]


def package_status(package_name: str) -> dict[str, Any]:
    """Return import/version status for a Python package."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return {"installed": True, "version": version, "error": None}
    except Exception as exc:
        return {"installed": False, "version": None, "error": repr(exc)}


def collect_python_info() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
    }


def collect_system_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def collect_memory_info() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_mb": round(vm.total / 1024**2, 2),
            "available_mb": round(vm.available / 1024**2, 2),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def collect_torch_info() -> dict[str, Any]:
    status: dict[str, Any] = package_status("torch")
    if not status["installed"]:
        return status

    import torch

    devices: list[dict[str, Any]] = []

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "type": "cuda",
                    "index": idx,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / 1024**2, 2),
                    "capability": list(props.major_minor)
                    if hasattr(props, "major_minor")
                    else [props.major, props.minor],
                }
            )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append({"type": "mps", "index": 0, "name": "Apple Metal"})

    if not devices:
        devices.append({"type": "cpu", "index": 0, "name": platform.processor()})

    status.update(
        {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "mps_available": hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available(),
            "selected_device": devices[0]["type"],
            "devices": devices,
        }
    )
    return status


def check_model_access(model_id: str) -> dict[str, Any]:
    """Check model config access without downloading model weights."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return {
            "accessible": True,
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "error": None,
        }
    except Exception as exc:
        return {"accessible": False, "error": repr(exc)}


def build_report(check_model: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "python": collect_python_info(),
        "system": collect_system_info(),
        "memory": collect_memory_info(),
        "torch": collect_torch_info(),
        "packages": {name: package_status(name) for name in REQUIRED_PACKAGES},
        "environment_variables": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_TOKEN_SET": bool(os.environ.get("HF_TOKEN")),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }

    if check_model:
        report["model_access"] = check_model_access(MODEL_ID)

    missing = [
        name
        for name, status in report["packages"].items()
        if not status.get("installed")
    ]
    report["summary"] = {
        "missing_packages": missing,
        "ready_for_basic_scripts": len(missing) == 0 and report["torch"]["installed"],
        "notes": [
            "Run with --check-model-access to verify Hugging Face config access.",
            "Full model loading is intentionally left to 01_baseline.py.",
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-model-access",
        action="store_true",
        help="Verify Hugging Face config access without downloading model weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Where to write the JSON report. Default: {OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(check_model=args.check_model_access)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    summary = report["summary"]
    print(f"Wrote environment report to {args.output}")
    print(f"Selected device: {report['torch'].get('selected_device')}")
    print(f"Missing packages: {summary['missing_packages'] or 'none'}")


if __name__ == "__main__":
    main()
