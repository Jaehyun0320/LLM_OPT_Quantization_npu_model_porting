Quantization report

01 baseline
The tiny-gpt2 run was used only as a local smoke test. On Apple MPS, latency was higher than CPU because the model is extremely small and autoregressive decoding incurs per-token dispatch overhead. Final quantitative comparison should be performed on the target Gemma checkpoint in a CUDA environment.

about mixed precision
In this implementation, mixed precision is approximated through quantization skip policies: the model is loaded in INT4 or INT8 while selected sensitive modules are kept in fp16/bf16. A finer-grained INT4/INT8/fp16 policy would require custom module replacement or another quantization backend, and is left as a production extension.

about assistant model
The candidate Gemma 3 tokenizers produced identical encodings for the benchmark prompts, although the full tokenizer vocabularies differed by one entry due to an extra tokenizer item. I therefore treated them as practical speculative-decoding candidates but validated prompt encodings before use.

I tested three Gemma-family assistant models for speculative decoding with `google/gemma-4-E2B`: `gemma-3-270m-it`, `gemma-3-270m`, and `gemma-3-1b-it`. The plain-text benchmark prompts encoded identically across the target and assistant tokenizers, but the full vocabularies were not identical: Gemma 4 E2B had 262,144 tokenizer entries, while the Gemma 3 assistants had 262,145, with several special/reserved token differences and 6,187 shared token strings mapped to different IDs.

Because of this, I used the Transformers assisted decoding path with both `tokenizer` and `assistant_tokenizer` passed explicitly, instead of assuming shared token IDs. This mitigates incorrect accept/reject verification across non-identical token spaces for the plain-text benchmark used here. However, prompts containing special tokens, tool-use tokens, chat-template tokens, or multimodal tokens still require separate validation. Therefore, I report latency, memory, accepted/rejected draft tokens, and reject rate separately for each assistant model.


어려웠던 점 : speculative decoding을 위한 assistant-model을 찾을 때 tokenizer와 호환되는 것을 찾아야 했던 것


onnx / edge format report

For production deployment, a static preallocated KV cache with a fixed maximum sequence length could simplify runtime memory planning and avoid dynamic cache growth. However, it requires explicit cache_position handling and cache update/scatter logic, and may require model/runtime-specific support. In this assignment, I first attempted the Hugging Face-style dynamic past/present KV-cache export because it matches the model forward API more directly.

mlx conversion
잘못된 호출:
python3 -m mlx_lm.convert --model google/gemma-4-E2B ...

맞는 호출:
python3 -m mlx_lm.convert --hf-path google/gemma-4-E2B ...
mlx-lm 0.29.1 버전에서 허깅페이스 모델 경오를 넘길 때 --model이 아니라 --hf-path로 받음