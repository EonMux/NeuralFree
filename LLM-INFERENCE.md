# NeuralFree GGUF LLM Inference

NeuralFree includes an engine for running `*.gguf` models, in the package `com.eonmux.neuralfree.gguf`. It needs no native llama.cpp library. It follows llama.cpp's model definitions, quantization formats, tokenizers, chat templates and sampler chain, and has two forward paths:
- a CPU path, `GGUFModel`;
- a GPU path, `GGUFModelGPU`, on OpenCL, CUDA or Vulkan.

Both paths are driven by the same resolved `ModelSpec`. The command-line front end is `com.eonmux.neuralfree.demo.LLMDemo`, run through the `runLLMDemo` Gradle task.

Related pages:
- [LLAMACPP-PARITY.md](LLAMACPP-PARITY.md): the per-architecture status.
- [GAPS.md](GAPS.md): what the engine refuses.

**Verification status.** The engine was verified by:
- compiling the Java sources;
- compiling every GPU kernel: clang (OpenCL C 1.2), nvcc (PTX), and glslc with spirv-val (SPIR-V);
- tracing the code against llama.cpp.

It has not been validated by running models on hardware.

## Contents

1. [Quick start](#quick-start)
2. [Selecting a model](#selecting-a-model)
3. [Backends](#backends)
4. [Command-line flags and system properties](#command-line-flags-and-system-properties)
5. [Programmatic API](#programmatic-api)
6. [Weight types](#weight-types)
7. [Tokenizers and chat templates](#tokenizers-and-chat-templates)
8. [Sampling](#sampling)
9. [Memory, context and weight streaming](#memory-context-and-weight-streaming)
10. [Other model modes](#other-model-modes)

## Quick start

```bash
# One prompt on the best GPU backend
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --prompt 'Explain recursion' --temp 0"

# Console chat (no --prompt)
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf"

# CPU engine
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --prompt 'Hi' --cpu"

# Larger JVM heap (default 8192 MB)
./gradlew runLLMDemo -Pheap=24576 -Pargs="--model /path/to/model.gguf --prompt 'Hi'"

# Engine properties: -D and -P neuralfree.* properties are forwarded to the JVM
./gradlew runLLMDemo -Pneuralfree.kv.type=q8_0 -Pargs="--model /path/to/model.gguf"
```

`runLLMDemo`:
- passes `-Pargs` as the program arguments, with single or double quotes grouping words;
- points `java.io.tmpdir`, `TMPDIR` and `CUDA_CACHE_PATH` at `build/tmp`;
- logs at INFO to the console through `src/main/config/log4j2.xml` ([USAGE.md](USAGE.md#logging)).

## Selecting a model

The model path is taken from the first of these that is set:
1. `--model <path>`, `-m <path>`, or the first argument that is not a flag;
2. the system property `-Dneuralfree.model=<path>` (with Gradle, `-Pneuralfree.model=<path>`);
3. the environment variable `NEURALFREE_MODEL`.

When none of them is set, or the file does not exist, `LLMDemo` prints these options and exits with status 2. Split models (`*-00001-of-0000N.gguf`) load as one model.

## Backends

| Choice | How | Behaviour |
|---|---|---|
| GPU, automatic | default, `--gpu`, `--backend auto`, `neuralfree.backend=auto` | Device-first selection over Vulkan, CUDA and OpenCL (below) |
| One GPU API | `--backend vulkan`, `--backend cuda`, `--backend opencl` | Only that API is enumerated; if it cannot run the LLM kernels, the error is `<API> backend unavailable: <reason>` |
| CPU | `--cpu`, `--backend cpu`, `neuralfree.backend=cpu` | The CPU engine (`GGUFModel`) |

GPU backends are created with `HWAcceleration.forMode(mode).createBackend(mode, KernelLibraryRequirement.of(KernelLibraries.LLM))`. Only a backend that can serve the LLM kernel library qualifies.
- **Kernel sources.** The LLM kernel library lives under `src/main/resources`: `kernels/llm` holds the dual-dialect OpenCL C and CUDA sources in `manifest.txt` order, and `shaders/llm` the GLSL compute shaders. Both start from the dialect headers they share with the training framework's `train` library: `kernels/common/dialect.h` (the manifest's `[shared] common/dialect.h` entry) and `shaders/common/nf_base.glsl` (included by `shaders/llm/common/nf_common.glsl`). The Gradle tasks `compileCudaKernels` and `compileShaders` build both libraries; `-Pneuralfree.gpu.libraries=llm` restricts the kernel tasks to the LLM library for a faster build; the train library's CUDA modules and SPIR-V are then missing, so CUDA and Vulkan report that library unavailable and the training framework takes its CPU path there.
- **OpenCL.** The kernel sources are compiled at run time with `-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros -DNF_FA_SMEM_FLOATS=<n>`, plus `-DUSE_SUBGROUPS=1` on devices with subgroup reductions (`OpenClKernelLibrary`). Program binaries are cached in `~/.cache/neuralfree/opencl`.
- **CUDA.** Uses the CUDA modules built by `compileCudaKernels`. The library is unavailable when:
  - the kernels were not built (no nvcc at build time);
  - no SASS or PTX image fits the device architecture;
  - the driver is too old for the images;
  - a kernel of the library's inventory is missing from the build.
- **CUDA memory.** Device buffers come from a stream-ordered caching pool (`CudaComputeBackend`, `CudaBufferPool`). A released buffer's block is kept for reuse instead of being returned to the driver. When an allocation runs out of memory, the backend frees its pool and retries, then frees the pools of the other backends on the same device and retries again. The free VRAM the backend reports counts pooled bytes as free. Closing a `GGUFModelGPU` that runs on a backend it did not create returns the pooled blocks to the driver (`CudaComputeBackend.trimPool()`), so other processes see the memory free; a backend the model created is closed with the model.
- **Vulkan.** Uses the SPIR-V built by `compileShaders`. The device must enable 8-bit storage, 16-bit storage and timeline semaphores. Without a shader compiler (glslc, or glslangValidator) or spirv-dis at build time, or when a kernel of the library's inventory is missing from the shader index, the library is unavailable. Pipelines are cached in `~/.cache/neuralfree/vulkan/`.
- **Sharing a backend with the training framework.** A backend can serve both kernel libraries. The framework's tensor ops serialize on the backend's lock (`BackendResources.lock`), which the GGUF engine does not take, so one backend instance must not run a model and tensor ops from different threads at the same time (`TrainGpu`).

AUTO picks the physical device first, then the API (default order `vulkan,cuda,opencl`). [USAGE.md](USAGE.md#compute-backends) describes the policy and its properties.

When no GPU backend can be created, the demo continues on the CPU. `LLMModelFactory.open` also falls back to the CPU engine when the GPU load fails with a runtime or linkage error, and logs the reason. A missing or unreadable file and an I/O error while reading it are thrown without a fallback.

## Command-line flags and system properties

Every `LLMDemo` flag is listed, with its default, in [USAGE.md: LLMDemo command-line reference](USAGE.md#llmdemo-command-line-reference). Every `neuralfree.*` system property is in [USAGE.md: System properties](USAGE.md#system-properties).

## Programmatic API

`LLMModelFactory` opens a model on the configured backend, with a CPU fallback. `GenerationLoop` runs llama.cpp's completion loop over any `EngineModel`:

```java
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;

EngineConfig cfg = EngineConfig.fromSystemProperties().toBuilder()
        .backend(EngineConfig.BackendMode.AUTO)
        .build();
try (EngineModel model = LLMModelFactory.open("/path/to/model.gguf", cfg)) {
    GGUFTokenizer tok = model.tokenizer();
    int[] prompt = tok.encode(tok.formatChat("You are a helpful assistant.", "Hello", false));

    GenerationRequest request = GenerationRequest.builder()
            .maxNewTokens(256)
            .sampler(SamplerConfig.builder().temperature(0.7f).seed(42).build())
            .stopStrings("</answer>")
            .build();
    GenerationResult result = GenerationLoop.forModel(model, LLMModelFactory.effectiveConfig(model, cfg))
            .generate(prompt, request, (id, text) -> System.out.print(text));
    System.out.println("\n" + result.stopReason() + ", " + result.tokenCount() + " tokens");
}
```

Other entry points:
- **Speculative decoding.** `LLMModelFactory.openSpeculative(targetPath, draftPath, cfg)`, or `LLMModelFactory.speculate(model, cfg)` with `cfg.draftModelPath()` or `cfg.mtp()` set, returns a `SpeculativeSession`. Its `generate` has the same signature. `EngineConfig.Builder.draftType(EngineConfig.DraftType)` forces how a draft file drafts (`AUTO`, `MTP`, `MODEL`), as `--spec-type` does. Before planning, `LLMModelFactory.open` reads the configured draft file's metadata to size the DeepSeek-V4 compressor rollback snapshots (`n_rs_seq`); a draft file that cannot be read or bound gets 0 snapshots and a warning, and the target still opens. `MemoryConfig.Builder.rollbackSnapshots(int)` overrides the derived value.
- **Inspection.** `LLMModelFactory.inspect(path, memoryConfig)` resolves the `ModelSpec` without loading weights. `LLMModelFactory.memoryPlan(model)` returns the fitted plan.
- **Caller-owned backend.** `LLMModelFactory.open(path, cfg, backend)` opens the model on a backend the caller created and keeps owning, with the same CPU fallback. A device filter in `cfg` has no effect then.
- **Direct construction.** `new GGUFModelGPU(path)`, `new GGUFModelGPU(path, backend)` and `new GGUFModelGPU(path, backend, cfg)` open on the GPU; `new GGUFModel(path)` and `new GGUFModel(path, cfg)` open on the CPU. They do not fall back. A model given a backend does not close it.
- **`LLMModel` methods** (both engines):
  - `generate(int[], GenerationRequest, TokenCallback)`;
  - `encode(int[])` and `encode(int[], int[] tokenTypes)` for pooled embeddings, and `encodeTokens(int[])` for per-token embeddings;
  - `classify(int[])` and `classLabels()`;
  - `rerank(query, document)`;
  - `generateDiffusion(int[], DiffusionConfig, TokenCallback)`;
  - `decodeAudioCodes(int[])`;
  - `supportClass()`, `spec()`, and metadata accessors (`architecture()`, `contextLength()`, `embeddingLength()`, `layerCount()`, `vocabSize()`, ...).
- **Legacy positional `generate`.** The overloads `generate(int[] tokens, int maxNewTokens, float temperature, float topP, ...)` remain for older callers.

## Weight types

| Path | Native types | Other types |
|---|---|---|
| GPU (`GpuOps.supportsWeightType`, `MoeKernels`, `GemmKernels.TILED_TYPES`) | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0 | Dequantized to F32 on the CPU at upload (`GpuWeightUploader.TypePolicy`) |
| CPU (`CpuMatVec`) | The same set except Q8_K | See below |

Details:
- **Shared kernel coverage.** Matrix-vector, tiled GEMM (prompt batches) and MoE expert kernels cover the same 29 GPU types.
- **Upload type is decided per tensor.** Each tensor keeps its stored type when the backend has its kernel, so mixed-quantization files keep their per-tensor types.
- **Large tensors.** Tensors above the device's allocation limit are split into row shards.
- **CPU refusals.** The CPU engine refuses matmul weights of the types ggml-cpu has no vec-dot for: Q8_1, Q8_K, the integer types and F64.
- **Removed types.** Q4_2 and Q4_3 are rejected everywhere.

## Tokenizers and chat templates

| `tokenizer.ggml.model` | Tokenizer |
|---|---|
| `llama` | SentencePiece (`SpmTokenizer`) |
| `gpt2`, `hybriddna`, `whitespace` | Byte-level BPE (`BpeTokenizer`) with the pre-tokenizer named by `tokenizer.ggml.pre` (`PreTokenizerRegistry`) |
| `gemma4` | `Gemma4Tokenizer` |
| `t5` | Unigram (`UgmTokenizer`), including `precompiled_charsmap` normalization |
| `bert` | WordPiece (`WpmTokenizer`) |
| `rwkv` | `RwkvTokenizer` |
| `plamo2`, `plamo3` | `Plamo2Tokenizer`, `Plamo3Tokenizer` |
| `none`, `no_vocab` | `NoVocabTokenizer` (models without a vocabulary, such as the WavTokenizer decoder) |

Chat templates are rendered from `tokenizer.chat_template` by the built-in Jinja engine (`gguf/jinja`, behind `MiniJinja`). Use `--raw` to skip the template. `ReasoningFilter` splits reasoning and tool-call channels from the answer; `--content-only` prints the answer alone.

## Sampling

`Sampler` runs llama.cpp's default chain:

```
logit bias -> penalties -> top-k -> typical-p -> top-p -> min-p -> temperature -> draw
```

Notes:
- **Engine defaults** (`SamplerConfig.defaults()`) are llama.cpp's: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, typical-p 1, penalties off, penalty window 64, random seed.
- **Demo defaults.** `LLMDemo` uses temperature 0.6 and a window of 256 where no property sets them.
- **Greedy decoding.** A temperature of 0 or below, or `neuralfree.greedy=true`, selects the highest logit.
- **Device argmax.** When the chain is a plain argmax, the GPU path decodes on the device and reads back only the token id.

## Memory, context and weight streaming

- **Context fit.** With `--ctx 0` (the default), `ContextFitter` picks `n_ctx_train` when its memory plan fits, else the largest multiple of 256 that fits. The plan counts the KV caches, sliding-window rings, recurrent states, side caches, snapshot shadows, the NextN draft context and a configured draft model. A context above `n_ctx_train` is accepted with a warning. On a GPU target the draft file's weights are reserved for the route it actually drafts with (configured draft type against the actual target): a NextN file that drafts as a separate model reserves its trunk, while an MTP draft file reserves only what its NextN drafter reads.
- **Weight streaming (GPU).** With an automatic context, every weight stays resident when the remaining VRAM still fits at least 4096 tokens (or `n_ctx_train`, when that is shorter). Otherwise:
  - the trunk layers are streamed through VRAM slots (`--vram-depth`) from a host cache (`--ram-cache-mb`);
  - the automatic context is capped at that floor (4096, or `n_ctx_train` when shorter).
  
  A user context is honoured as given. `--vram-budget` prints the plan and exits.
- **Off-device weights.** Tensors only the host heads read are not uploaded. NextN/MTP blocks are placed on the device with `--mtp` or `neuralfree.nextn.memory`, or when an MTP drafter attaches to a model opened without them.
- **KV cache types, rings, prefix reuse, context shift, rollback and snapshots** are described in [LLAMACPP-PARITY.md](LLAMACPP-PARITY.md#cache-features).

## Other model modes

| Mode | Models | Output |
|---|---|---|
| `embed` | EMBED models and pooled models | Pooled embedding (dimension, norm, leading values) or per-token rows |
| `classify` | Classifiers (`RANK` pooling) and decision heads (clef, vllm-decision2) | Scores with the model's class labels |
| `rerank` | `RANK` pooling with `--document` | One relevance score |
| `diffusion` | dream, llada, llada-moe, rnd1 | Generated text |
| `audio` | wavtokenizer-dec, pockettts, qwen3tts | Spectral features, hidden rows, or codec tokens |

Encoder-decoder models (`t5`) generate through the normal loop: the prompt is encoded, then decoded from the decoder start token.
