# Quantization / Optimization Report

## Summary
- Local tiny-gpt2 runs were smoke tests only, not final performance evidence.
- INT8/INT4 experiments target CUDA because the implementation uses bitsandbytes.
- Mixed precision is approximated by keeping selected sensitive modules in fp16/bf16 while quantizing the rest of the model.
- Speculative decoding used Gemma-family assistants with tokenizer-aware assisted decoding.

## Local Smoke Test
The tiny-gpt2 run verified that the benchmark pipeline could load a model, generate text, measure latency, and write results. On Apple MPS, latency was higher than CPU because the model is very small and autoregressive decoding pays per-token dispatch overhead. Final latency, memory, and quality comparisons should be run on the target Gemma checkpoint in CUDA.

## Mixed Precision Strategy
The implementation uses bitsandbytes INT8 or INT4 quantization, with optional skip policies for sensitive components. Attention projections, MLP projections, embeddings, or the LM head can be kept in fp16/bf16 while the rest is quantized. A true fine-grained INT4/INT8/fp16 mixture would require custom module replacement or another backend, so it is left as a production extension.

## Speculative Decoding Assistants
I tested three assistant candidates for speculative decoding with `google/gemma-4-E2B`: `google/gemma-3-270m-it`, `google/gemma-3-270m`, and `google/gemma-3-1b-it`. The benchmark prompts encoded identically across target and assistant tokenizers, making them practical candidates for the plain-text benchmark.

The full vocabularies were not identical. Gemma 4 E2B had 262,144 tokenizer entries, while the Gemma 3 assistants had 262,145. A full comparison also found special/reserved token differences and 6,187 shared token strings mapped to different IDs.

Because of this, I used the Transformers assisted decoding path with both `tokenizer` and `assistant_tokenizer` passed explicitly. This avoids assuming that the same token ID has the same meaning in both tokenizers and makes accept/reject verification safer for the tested prompts.

## Limitations
Special tokens, tool-use tokens, chat-template tokens, and multimodal tokens still require separate compatibility checks. Speculative decoding results should therefore be reported per assistant, including latency, memory, accepted draft tokens, rejected draft tokens, and reject rate.
