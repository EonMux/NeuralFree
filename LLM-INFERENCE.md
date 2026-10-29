# NeuralFree GGUF LLM Inference

NeuralFree includes an engine for running `*.gguf` models and Hugging Face checkpoints, in the package `com.eonmux.neuralfree.gguf`. It needs no native llama.cpp library. It follows llama.cpp's model definitions, quantization formats, tokenizers, chat templates and sampler chain, and has two forward paths:
- a CPU path, `GGUFModel`;
- a GPU path, `GGUFModelGPU`, on OpenCL, CUDA or Vulkan.

Both paths are driven by the same resolved `ModelSpec`. A Hugging Face checkpoint (safetensors, no Python, no conversion step) is parsed into the same view a GGUF file produces and runs through the same code. The command-line front end is `com.eonmux.neuralfree.demo.LLMDemo`, run through the `runLLMDemo` Gradle task.

Related pages:
- [USAGE.md](USAGE.md#hugging-face-checkpoints): HF flags, materialization, quantization formats, supported architectures.
- [GAPS.md](GAPS.md): what the engine refuses.

**Verification status.** The engine was verified by:
- compiling the Java sources;
- compiling every GPU kernel: clang (OpenCL C 1.2), nvcc (PTX), and glslc with spirv-val (SPIR-V);
- tracing the code against llama.cpp.

It has not been validated by running models on hardware.

## Contents

1. [Quick start](#quick-start)
2. [Selecting a model](#selecting-a-model)
3. [Model sources](#model-sources)
   - [The ModelSource seam](#the-modelsource-seam)
   - [The HF pipeline order](#the-hf-pipeline-order)
4. [Backends](#backends)
5. [Command-line flags and system properties](#command-line-flags-and-system-properties)
6. [Programmatic API](#programmatic-api)
7. [Weight types](#weight-types)
   - [NF extension weight types](#nf-extension-weight-types)
8. [Tokenizers and chat templates](#tokenizers-and-chat-templates)
9. [Sampling](#sampling)
10. [Memory, context and weight streaming](#memory-context-and-weight-streaming)
11. [Other model modes](#other-model-modes)

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

The path is detected by content, not by name ([USAGE.md](USAGE.md#hugging-face-checkpoints)):
- a file that starts with the `GGUF` magic is a GGUF file, split or not;
- a `*.safetensors` file is an HF checkpoint of one shard: its parent directory must hold `config.json` (or `params.json`, the Mistral-native form) and only that file's tensors are used;
- a directory with safetensors files and a `config.json` (or the Mistral-native `params.json` with `consolidated*.safetensors`) is an HF checkpoint;
- a directory without safetensors files that holds exactly one `.gguf`, or one split set, opens that GGUF;
- a directory with `adapter_config.json` is a PEFT adapter and is refused as a model: it attaches with `--lora` to its base model;
- a directory with only pickle files (`pytorch_model*.bin`, `*.pt`, `*.pth`) is a torch pickle checkpoint, which loads through a restricted unpickler (`TorchCheckpointFactory` in `CheckpointContainers`); unsafe opcodes and non-tensor payloads are refused by name at load;
- `--hf-repo <user>/<model>[:quant][@revision]` (short `-hf`, with `--hf-file` to name a file) downloads from the Hugging Face Hub and opens the download; it excludes `--model` ([USAGE.md](USAGE.md#hugging-face-hub-download)).

## Model sources

### The ModelSource seam

Every consumer of a model sees one interface, `com.eonmux.neuralfree.gguf.ModelSource`, whether the bytes come from a GGUF file, a split GGUF or a Hugging Face checkpoint:

- metadata as an unmodifiable map over the GGUF value model, with the type of every key (`getValueType`, `getArrayElementType`) reachable through name;
- the tensor list (`getTensors`): GGUF tensor names, `ne` dimensions, `GGMLType`, and the virtual offset each tensor holds in one address space;
- `createMappedStore(use)`: the tensors' bytes by virtual offset, mmapped through `DataSegment`s of the underlying files (GGUF shards, safetensors files, produced ranges) ([USAGE.md](USAGE.md#materialization-of-derived-tensors));
- `openRegions(mode)`: the same bytes as CPU `TensorRegions` for the CPU engine.

`ModelSources.open(path, options)` creates the source for a GGUF file or an HF checkpoint, with the options of [USAGE.md](USAGE.md#flags-and-properties) (`EngineConfig` carries them as `sourceOptions()`, so the engine's load path uses the same configured source). `GGUFFileParser` implements the interface for files; `HfModelSource` builds the HF view. `ModelSources.open(path)` applies the properties; an overloaded form takes an explicit `target` model that a draft checkpoint serves. `ModelSources.openAdapter` opens a PEFT adapter against its base. Incoming tensor info is validated once for all three producers (GGUF, HF, export), with each producer's own message prefix.

### The HF pipeline order

`HfModelSource` assembles the HF view the way the llama.cpp converter does, so that the output equals `convert_hf_to_gguf.py`'s. The steps run in this order, each reading the previous's results:

1. **Config fixup.** `config.json` is read, `text_config` is merged (it wins), sub-config renames apply, `--hf-overrides` values are merged, and `generation_config.json` is taken raw (then mapped by the system properties, see below). `architectures[0]` names the entry of the arch registry, whose static tables and hook classes resolve the architecture.
2. **Tokenizer location.** The tokenizer files (also `--hf-tokenizer`) overlay the checkpoint's: `tokenizer.json`, `tokenizer.model` (SentencePiece), `*.tiktoken`, `tekken.json`, `rwkv_vocab_v*.txt`, `tokenizer.jsonl` (PLaMo), WordPiece and Unigram installations.
3. **Vocabulary.** The converter's per-family vocab writers build the `tokenizer.*` keys, so the token count is known before tensors are claimed; the model's generation defaults come from now-writable `general.sampling.*` and `neuralfree.sampling.*` keys.
4. **Tensor hooks.** Per-architecture, they map HF tensor names to GGUF names and shapes, apply transforms (permutation, row de-interleaving, expert reshaping, MLA splits), classify quantized weights into the repack targets of [USAGE.md](USAGE.md#supported-quantization-formats), and drop what the converter drops (recorded under `neuralfree.hf.dropped_prefixes`).
5. **Metadata hook.** The finished metadata (architecture, hparams, chat template, quantization types) is written through the GGUF key table; the hook reads small tensor values where a key depends on them.

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
- **Speculative decoding.** `LLMModelFactory.openSpeculative(targetPath, draftPath, cfg)`, or `LLMModelFactory.speculate(model, cfg)` with `cfg.draftModelPath()` or `cfg.mtp()` set, returns a `SpeculativeSession`. Its `generate` has the same signature. `EngineConfig.Builder.draftType(EngineConfig.DraftType)` forces how drafting runs (`AUTO`, `MTP`, `MODEL`, or the n-gram and suffix drafters, as `--spec-type` does). The verify mode is programmatic only: `SpeculativeDecoder.Builder.verifyMode(...)` takes `match`, `rejection`, `block`, `fly` or `synthetic`; no flag exposes it. Before planning, `LLMModelFactory.open` reads the configured draft file's metadata to size the DeepSeek-V4 compressor rollback snapshots (`n_rs_seq`); a draft file that cannot be read or bound gets 0 snapshots and a warning, and the target still opens. `MemoryConfig.Builder.rollbackSnapshots(int)` overrides the derived value.
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
- **SequenceRun** (`engine/SequenceRun`): the per-sequence token pipeline the loop is built from — `over(...)` to open, `next(float[])` per step, then `accept`, `offer`, `stop` and `result` — for callers that step the generation themselves.
- **Chat requests** (`gguf/chat`): `ChatRequest`, `ChatMessages`, `ChatTool`, `ChatTemplateSource`, `ChatFormatter` and `ToolChoice` model a chat completion programmatically, independent of the demo.
- **Grammar constraints** (`gguf/grammar`): a `ConstraintSpec` of kind `GBNF`, `JSON_SCHEMA`, `JSON_OBJECT`, `REGEX` or `CHOICE` is compiled by the GBNF engine (`GbnfBuilder`, `GbnfParser`, `Grammar`, `GrammarLimits`, `JsonSchemaToGbnf`, `RegexToGbnf`) and runs as the sampler chain's `GrammarStage`; structural-tag specs parse no input yet.
- **Speculative drafters without a draft file.** `NgramDrafter` (prompt-lookup and map variants), `SuffixDrafter` and `DrafterChain` draft from the prompt and the output; `SpeculativeDecoder.Builder.verifyMode(...)` selects how drafts are verified (`match`, `rejection`, `block`, `fly`, `synthetic`).
- **Importance matrices** (`quant/write/ImatrixFile`): read and write llama.cpp imatrix files, GGUF and legacy `.dat`.
- **Control vectors** (`ops/heads/ControlVector`, with `ControlVectorExtra` and `CpuControlVectorExtra`): apply a control vector to the hidden states at inference.
- **Torch pickle checkpoints** (`hf/torch`): `TorchCheckpointFactory`, a built-in `CheckpointContainers` factory, reads `pytorch_model*.bin`/`.pt`/`.pth` through a restricted unpickler (`RestrictedUnpickler`).

## Weight types

| Path | Native types | Other types |
|---|---|---|
| GPU (`GpuOps.supportsWeightType`, `MoeKernels`, `GemmKernels.TILED_TYPES`) | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0, F8_E4M3, F8_E5M2, F8_E4M3_B128, MXFP8, MXFP6_E3M2, MXFP6_E2M3, I8_B128, NF4, FP4_BNB | Dequantized to F32 on the CPU at upload (`GpuWeightUploader.TypePolicy`) |
| CPU (`CpuMatVec`) | The same set except Q8_K | See below |

Details:
- **Shared kernel coverage.** Matrix-vector, tiled GEMM (prompt batches) and MoE expert kernels cover the same 38 GPU types.
- **Upload type is decided per tensor.** Each tensor keeps its stored type when the backend has its kernel, so mixed-quantization files keep their per-tensor types.
- **Large tensors.** Tensors above the device's allocation limit are split into row shards.
- **CPU refusals.** The CPU engine refuses matmul weights of the types ggml-cpu has no vec-dot for: Q8_1, Q8_K, the integer types and F64.
- **Removed types.** Q4_2 and Q4_3 are rejected everywhere.

### NF extension weight types

Nine weight types beyond ggml's ids (ids 1000-1008, `GGMLType` validity unchanged elsewhere) carry the quantization formats Hugging Face checkpoints use and llama.cpp has no type for. Every 2-D/3-D weight of one needs its inner dimension to be a whole number of blocks; a tensor that cannot be is served by its fallback instead ([USAGE.md](USAGE.md#supported-quantization-formats)).

| Type | id | Block | Layout | Value |
|---|---|---|---|---|
| `F8_E4M3` | 1000 | 32 elems / 32 bytes | `u8 qs[32]` E4M3FN codes | `e4m3(qs[j])`, times the tensor's or expert's `.scale` |
| `F8_E4M3_B128` | 1001 | 128 / 132 | `f32 d; u8 qs[128]` | `d * e4m3(qs[j])` |
| `MXFP8` | 1002 | 32 / 33 | `u8 e (E8M0); u8 qs[32]` | `2^(e-127) * e4m3(qs[j])` |
| `I8_B128` | 1003 | 128 / 132 | `f32 d; i8 qs[128]` | `d * qs[j]` |
| `NF4` | 1004 | 64 / 36 | `f32 absmax; u8 qs[32]` (two nibbles per byte) | `NF4_CODE[q] * absmax` |
| `FP4_BNB` | 1005 | 64 / 36 | as NF4 | `FP4_BNB_CODE[q] * absmax` |
| `F8_E5M2` | 1006 | 32 / 32 | `u8 qs[32]` E5M2 codes | `e5m2(qs[j])` (IEEE half high bytes), times the `.scale` |
| `MXFP6_E3M2` | 1007 | 32 / 25 | `u8 e (E8M0); u8 qs[24]`, 32 packed 6-bit codes | `2^(e-127) * e3m2(qs[j])` |
| `MXFP6_E2M3` | 1008 | 32 / 25 | as MXFP6_E3M2, `e2m3` codes | `2^(e-127) * e2m3(qs[j])` |

Their codebooks are the output of bitsandbytes' `get_4bit_type`, normalized by the maximum magnitude. A checkpoint's `quant_map` is accepted when within 1e-4 of the codebook; a differing map makes the repack approximate, and `--hf-quant-exact` dequantizes through the stored map to F32 instead.

Kernels:
- **CPU** (`CpuBlockDecoders`, `CpuMatVec`): one vec-dot per type in the block quantizers, decoding the block and using Q8_0 activations, as ggml's `vec_dot_type` rule gives every block type.
- **GPU** (`kernels/llm/common/quant_decode_ext.h`, `matvec_extquant.cl`, `matmul_tiled.cl`, `moe_matvec.cl`): one decode triple per type in the `IQ_ALL_KERNELS` contract of the other block types, so matrix-vector, tiled GEMM (8 kernels), MoE matvec and the value-expert kernels cover them like any ggml block type. The scales never leave the block bytes, so no kernel family is excluded.
- **Vulkan** (`shaders/llm/common/quant_blocks.glsl`, `quant_ext.glsl`): the GLSL twins, with their rows in `shaders/llm/variants/*.tsv`.
- **GPU activations.** On the GPU the extension types take F32 activations, as every NF block type does on the GPU.

A source that uses one of these types sets the `neuralfree.extension_types` marker, which `GgufWriter` keeps in `--export-mode native` and converts away in the `compat` modes. A GGUF file carrying extension tensors without the marker is refused. GGUF files that do not use extension types are unaffected.

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

The chain is built from twelve stage kinds (`engine/sampling/StageKind`): `PENALTIES`, `DRY`, `TOP_N_SIGMA`, `TOP_K`, `TYPICAL_P`, `TOP_P`, `MIN_P`, `XTC`, `TEMPERATURE` (dynamic temperature), `INFILL`, `ADAPTIVE_P` and `MIROSTAT`. `Sampler.buildChain` lays them out in that default order — llama.cpp's chain — with llama.cpp's special case that a mirostat configuration replaces the whole chain with temperature + mirostat. The `neuralfree.samplers` property overrides the order with a semicolon-separated list of stage names (`StageKind.parseOrder`, e.g. `top_k;top_p;min_p`). `Sampler.create` appends the stages a request carries, such as the `GrammarStage` of a constraint (below).

Notes:
- **Engine defaults** (`SamplerConfig.defaults()`) are llama.cpp's: temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, typical-p 1, penalties off, penalty window 64, random seed.
- **Model defaults.** The `general.sampling.*` keys of the file (a GGUF's, or the `generation_config.json` values an HF checkpoint writes through them) provide defaults for every setting the caller did not set. `--model-sampling-defaults false` turns this off; [USAGE.md](USAGE.md#model-sampling-defaults) describes the mapping and precedence.
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
- **KV cache types, rings, prefix reuse, context shift, rollback and snapshots.** The KV cache has nine row types: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0` and `q5_1` (`KvDType`; there is no FP8 cache type — an FP8 checkpoint's KV scales are carried but attention does not consume them, see [GAPS.md](GAPS.md)). Quantized rows take an optional Hadamard rotation. Sliding-window layers keep ring caches; prefix reuse, context shift, rollback and recurrent-state snapshots are engine features ([USAGE.md](USAGE.md#memory-memoryconfigfromproperties) lists their properties, [GAPS.md](GAPS.md) the refusals).

### Paged KV memory

`gguf/memory/paged` keeps the rows of a KV group in reference-counted pages (`PagePool`, `PageList`, `PagedSequences`, `PagedStep`, `PageCopy`, `RowParts`) instead of one buffer per sequence. It is property-gated, not the engine default: `neuralfree.kv.paged=true` forces it for a single sequence and `neuralfree.parallel > 1` selects it; by default `ModelMemory.sequences()` returns `SingleSequenceMemory`. `neuralfree.kv.page.size` (default 16, a power of two in [1, 256]) sizes the pages, and `neuralfree.kv.unified` (default true) shares the whole context between the sequences.

On the CPU, `ForwardBatch` batches over the paged memory: `CpuForwardBackend.forwardBatch` splits the batch into one-sequence ubatches (llama.cpp `split_seq`), reserves each ubatch's pages all-or-nothing (`PagedSequences.prepare`, `MemoryFullException` before anything changed), stores K/V at the step's `cellIdx` rows, attends through the step's page tables and commits. A batch over several sequences computes the same logits as the same tokens run one sequence at a time (`CpuForwardBatchTest`). The GPU path has no paging yet ([GAPS.md](GAPS.md) lists what refuses).

The batch engine adds cross-request prefix reuse over this memory: retired sequences stay cached with their pages, an extending prompt adopts the cached sequence in place and prefills only its tail (`BatchEngine`, `PrefixCache`; [USAGE.md](USAGE.md#prefix-cache) has the user view, `BatchPrefixCacheTest` the parity runs).

### Continuous batching

`gguf/engine/batch` runs several requests over one open model: `BatchEngine` admits submitted prompts FIFO into the paged memory's sequence slots (a free slot plus enough free cells; the per-ubatch reservation stays all-or-nothing) and every `pump()` runs one step — one prompt prefill, or one decode row per running sequence through `forwardBatch`. Each sequence runs its own `SequenceRun` pipeline (sample, stop, route), so a batch computes the same tokens and texts as the same requests run one at a time (`BatchEngineTest`). A sequence outgrowing the context stops with `StopReason.CONTEXT_FULL` (no context shift in batch mode); a cancellation polls per step. The HTTP server drives this engine (`server/`, see [USAGE.md](USAGE.md#http-server)).

## Other model modes

| Mode | Models | Output |
|---|---|---|
| `embed` | EMBED models and pooled models | Pooled embedding (dimension, norm, leading values) or per-token rows |
| `classify` | Classifiers (`RANK` pooling) and decision heads (clef, vllm-decision2) | Scores with the model's class labels |
| `rerank` | `RANK` pooling with `--document` | One relevance score |
| `diffusion` | dream, llada, llada-moe, rnd1 | Generated text |
| `audio` | wavtokenizer-dec, pockettts, qwen3tts | Spectral features, hidden rows, or codec tokens |

Encoder-decoder models (`t5`) generate through the normal loop: the prompt is encoded, then decoded from the decoder start token.
