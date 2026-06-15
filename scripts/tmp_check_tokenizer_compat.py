import json
from pathlib import Path

from transformers import AutoTokenizer


TARGET_ID = "google/gemma-4-E2B"
CANDIDATE_IDS = [
    "google/gemma-3-270m-it",
    "google/gemma-3-270m",
    "google/gemma-3-1b-it",
]

PROBE_TEXTS = [
    "Explain quantization in one sentence:",
    "Why does KV-cache improve autoregressive decoding?",
    "A real-time chatbot should optimize for",
]


def load_tokenizer(model_id):
    if model_id == TARGET_ID:
        return AutoTokenizer.from_pretrained(
            model_id,
            extra_special_tokens={"video_token": "<|video|>"},
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


def tokenizer_summary(tokenizer):
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "len": len(tokenizer),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
    }


def encode_samples(tokenizer):
    return {text: tokenizer.encode(text) for text in PROBE_TEXTS}


def compare_tokenizers(target, candidate):
    target_vocab = target.get_vocab()
    candidate_vocab = candidate.get_vocab()
    target_encodings = encode_samples(target)
    candidate_encodings = encode_samples(candidate)

    return {
        "same_vocab": target_vocab == candidate_vocab,
        "same_vocab_size": len(target_vocab) == len(candidate_vocab),
        "same_probe_encodings": target_encodings == candidate_encodings,
        "target_summary": tokenizer_summary(target),
        "candidate_summary": tokenizer_summary(candidate),
        "target_probe_encodings": target_encodings,
        "candidate_probe_encodings": candidate_encodings,
    }


def main():
    result = {
        "target_id": TARGET_ID,
        "candidates": {},
    }

    try:
        target_tokenizer = load_tokenizer(TARGET_ID)
    except Exception as exc:
        result["target_error"] = repr(exc)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    for candidate_id in CANDIDATE_IDS:
        try:
            candidate_tokenizer = load_tokenizer(candidate_id)
            result["candidates"][candidate_id] = compare_tokenizers(
                target_tokenizer,
                candidate_tokenizer,
            )
        except Exception as exc:
            result["candidates"][candidate_id] = {"error": repr(exc)}

    output_path = Path("results/tokenizer_compat.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
