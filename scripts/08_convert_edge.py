import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_for_module(module_name):
    return [sys.executable, "-m", module_name]


def run_command(command, timeout_sec):
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "command": command,
            "started_at": started_at,
            "returncode": completed.returncode,
            "success": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "started_at": started_at,
            "returncode": None,
            "success": False,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "error": f"Command timed out after {timeout_sec} seconds.",
        }


def collect_environment():
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": platform.system() == "Darwin"
        and platform.machine() == "arm64",
        "mlx_lm_module_found": importlib.util.find_spec("mlx_lm") is not None,
        "mlx_module_found": importlib.util.find_spec("mlx") is not None,
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
    }


def build_convert_command(args):
    command = command_for_module("mlx_lm.convert")
    command.extend(["--hf-path", args.model_id])
    command.extend(["--mlx-path", args.mlx_path])

    if args.quantize:
        command.append("-q")

    if args.revision:
        command.extend(["--revision", args.revision])

    return command


def build_generate_command(args):
    command = command_for_module("mlx_lm.generate")
    command.extend(["--model", args.mlx_path])
    command.extend(["--prompt", args.prompt])
    command.extend(["--max-tokens", str(args.max_tokens)])
    return command


def inspect_output_dir(path):
    output_dir = Path(path)
    if not output_dir.exists():
        return {
            "exists": False,
            "path": str(output_dir),
            "files": [],
        }

    files = []
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            files.append(
                {
                    "path": str(item),
                    "size_mb": round(item.stat().st_size / 1024**2, 4),
                }
            )

    return {
        "exists": True,
        "path": str(output_dir),
        "files": files,
        "total_size_mb": round(sum(row["size_mb"] for row in files), 4),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B")
    parser.add_argument("--edge-format", type=str, default="mlx", choices=["mlx"])
    parser.add_argument("--mlx-path", type=str, default="models/gemma4_e2b_mlx_4bit")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Deep learning is")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--no-quantize", action="store_false", dest="quantize")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--convert-timeout-sec", type=int, default=7200)
    parser.add_argument("--generate-timeout-sec", type=int, default=600)
    parser.add_argument("--output", type=str, default="results/edge_mlx_conversion.json")
    return parser.parse_args()


def main():
    args = parse_args()
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "edge_format": args.edge_format,
        "mlx_path": args.mlx_path,
        "quantize": args.quantize,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "environment": collect_environment(),
        "convert": None,
        "generate": None,
        "output_dir": None,
        "success": False,
        "notes": [],
    }

    if not result["environment"]["is_apple_silicon"]:
        result["notes"].append(
            "Apple MLX conversion should be run on an Apple Silicon Mac "
            "(Darwin arm64)."
        )

    if not result["environment"]["mlx_lm_module_found"]:
        result["notes"].append(
            "mlx-lm is not installed. Install it with: python3 -m pip install mlx-lm"
        )

    convert_command = build_convert_command(args)
    result["convert"] = run_command(
        command=convert_command,
        timeout_sec=args.convert_timeout_sec,
    )
    result["output_dir"] = inspect_output_dir(args.mlx_path)

    if result["convert"]["success"] and not args.skip_generate:
        generate_command = build_generate_command(args)
        result["generate"] = run_command(
            command=generate_command,
            timeout_sec=args.generate_timeout_sec,
        )
    elif args.skip_generate:
        result["generate"] = {
            "success": None,
            "skipped": True,
            "reason": "--skip-generate was set.",
        }
    else:
        result["generate"] = {
            "success": None,
            "skipped": True,
            "reason": "Conversion failed, so generation was not attempted.",
        }

    result["success"] = bool(
        result["convert"]
        and result["convert"]["success"]
        and (
            args.skip_generate
            or (result["generate"] and result["generate"].get("success") is True)
        )
    )

    if not result["success"]:
        result["notes"].append(
            "Keep this JSON for the Conversion Notes section. MLX conversion can "
            "fail if the model architecture is unsupported by the installed "
            "mlx-lm version, if the gated Hugging Face token is unavailable, or "
            "if the local Apple Silicon memory/Metal runtime is insufficient."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Wrote edge conversion result to {args.output}")
    print(f"Success: {result['success']}")


if __name__ == "__main__":
    main()
