For production deployment, a static preallocated KV cache with a fixed maximum sequence length could simplify runtime memory planning and avoid dynamic cache growth. However, it requires explicit cache_position handling and cache update/scatter logic, and may require model/runtime-specific support. In this assignment, I first attempted the Hugging Face-style dynamic past/present KV-cache export because it matches the model forward API more directly.

The local tiny-gpt2 ONNX smoke export failed inside PyTorch's legacy ONNX tracer while tracing the Transformers causal-mask path. The failure occurred in the vmap/functorch-based mask generation code, not during model loading or due to model size. This indicates an exporter compatibility issue in the local torch/transformers stack rather than an out-of-memory failure.

After removing attention_mask from the tiny-gpt2 no-cache ONNX wrapper, the export still failed. The failure moved from a padding-mask path to the generic causal-mask path inside Transformers. This shows that the local failure was not caused by the attention_mask input itself, but by the legacy PyTorch ONNX tracer being unable to trace the current Transformers vmap/functorch-based causal mask implementation.

Gemma4/tiny-gpt2 HF export는 Transformers masking_utils / vmap / functorch 기반 causal mask 경로 때문에 실패했지만, toy causal LM으로 ONNX export와 ORT validation pipeline 자체는 성공적으로 검증했다.
Gemma4/tiny-gpt2 export 실패는 ONNX Runtime이나 attention_mask 개념 자체의 문제가 아니라,
Hugging Face Transformers 모델 내부의 복잡한 causal-mask 생성 경로와 legacy ONNX exporter의 호환성 문제다.
toy model을 통해 ONNX export + ORT validation pipeline 자체는 정상임을 확인했다.

2. Apple mlx format
처음 mlx conversion에 실패했는데 
잘못된 호출:
python3 -m mlx_lm.convert --model google/gemma-4-E2B ...

맞는 호출:
python3 -m mlx_lm.convert --hf-path google/gemma-4-E2B ...
mlx-lm 0.29.1 버전에서 허깅페이스 모델 경오를 넘길 때 --model이 아니라 --hf-path로 받음

그러고 다시 시도했는데 
I selected Apple MLX as the edge/mobile target and attempted to convert google/gemma-4-E2B using mlx-lm. The Hugging Face gated download succeeded, but conversion failed after model loading because the installed mlx-lm version did not provide a Gemma 4 model implementation. The error was "Model type gemma4 not supported" and "No module named mlx_lm.models.gemma4". The installed mlx-lm package included Gemma, Gemma 2, and Gemma 3 backends, but not Gemma 4. A production path would require either upgrading to an mlx-lm version with Gemma 4 support, implementing the Gemma 4 architecture in MLX, or choosing another supported edge runtime.

HF download는 됨
mlx-lm convert를 시도함
gemma4 model type unsupported로 실패함
설치된 mlx-lm에는 gemma/gemma2/gemma3는 있지만 gemma4 backend가 없음

gemma-3-270m으로 smoke test해서 성공함
MLX toolchain itself works on this Apple Silicon environment.
The blocker is Gemma 4 E2B architecture support in the installed mlx-lm version.
The Gemma 3 MLX run was used only to verify that the MLX conversion and generation toolchain worked locally. It was not used as the required 1e-3 PyTorch-vs-exported logits validation, because the assignment target was Gemma 4 E2B and the smoke model was quantized to 4-bit.
