# NeuralFree Usage Guide

This guide covers:
- building NeuralFree, including the optional GPU kernel tools and the two kernel libraries;
- choosing a compute backend for the GGUF engine and for the training framework;
- how the training framework uses a GPU backend: locking, residency, ordering and numerical tolerances;
- the complete `LLMDemo` command line and the `neuralfree.*` system properties;
- loading Hugging Face checkpoints: the flags and properties, materialization, activation quantization, the supported quantization formats and architectures.

[TRAINING.md](TRAINING.md) is the guide to the training framework's API: tensors, autograd, modules, optimizers, losses, trainers, checkpoints, distributed training, reinforcement learning and quantization. [LLM-INFERENCE.md](LLM-INFERENCE.md) describes the GGUF engine itself. The runnable demos in `src/main/java/com/eonmux/neuralfree/demo/` (`MLPDemo`, `CNNDemo`, `GNNDemo`, `TransformerDemo`, `AutogradDemo`, `InferenceDemo`, `GpuMatmulDemo`, `LLMDemo`) show complete programs.

## Contents

1. [Build](#build)
   - [The neuralfree command](#the-neuralfree-command)
   - [OpenAI client commands](#openai-client-commands)
   - [GPU kernel tools (all optional)](#gpu-kernel-tools-all-optional)
   - [Kernel libraries](#kernel-libraries)
   - [Kernel inventories](#kernel-inventories)
   - [GPU kernel build cache](#gpu-kernel-build-cache)
   - [Logging](#logging)
2. [Compute backends](#compute-backends)
   - [Tensor-core flash attention](#tensor-core-flash-attention)
   - [Fast GEMM for prefill and batched decode](#fast-gemm-for-prefill-and-batched-decode)
   - [Fused MoE grouped GEMM](#fused-moe-grouped-gemm)
   - [Decode-step command replay](#decode-step-command-replay)
   - [GPU event ownership](#gpu-event-ownership)
   - [Fused decode kernels](#fused-decode-kernels)
   - [Integer-dot (Q8_1) decode matvec](#integer-dot-q8_1-decode-matvec)
   - [Certified exact-argmax LM head](#certified-exact-argmax-lm-head)
   - [Lossless 12-bit weights (W12)](#lossless-12-bit-weights-w12)
   - [GPU weight placement](#gpu-weight-placement)
   - [Multi-GPU layer split](#multi-gpu-layer-split)
   - [Tensor-parallel row split (`--split-mode row`)](#tensor-parallel-row-split---split-mode-row)
   - [Tensor parallelism (`--split-mode tensor`)](#tensor-parallelism---split-mode-tensor)
   - [Expert-parallel load balancing (EPLB)](#expert-parallel-load-balancing-eplb)
   - [Context parallelism (`--split-mode context`)](#context-parallelism---split-mode-context)
   - [Models larger than RAM](#models-larger-than-ram)
   - [AMD GPUs: the HIP backend](#amd-gpus-the-hip-backend)
   - [Metal backend (Apple GPUs)](#metal-backend-apple-gpus)
   - [Other accelerator backends (provider jars)](#other-accelerator-backends-provider-jars)
   - [CPU SIMD kernels and weight repacking](#cpu-simd-kernels-and-weight-repacking)
3. [The training framework on a GPU backend](#the-training-framework-on-a-gpu-backend)
4. [LLMDemo command-line reference](#llmdemo-command-line-reference)
   - [Fused int8 CPU attention and experts](#fused-int8-cpu-attention-and-experts)
   - [Model, prompt and output](#model-prompt-and-output)
   - [Interactive console](#interactive-console)
   - [Backend and weights](#backend-and-weights)
   - [Memory](#memory)
   - [RoPE and long context](#rope-and-long-context)
   - [Sampling](#sampling)
   - [Speculative decoding](#speculative-decoding)
   - [LoRA adapters](#lora-adapters)
   - [Control-vector flags](#control-vector-flags)
   - [Diffusion (`--mode diffusion`)](#diffusion---mode-diffusion)
   - [Embeddings, classifiers and audio](#embeddings-classifiers-and-audio)
   - [Speech synthesis](#speech-synthesis)
   - [Speech to text](#speech-to-text)
   - [Pooling type override](#pooling-type-override)
5. [Hugging Face checkpoints](#hugging-face-checkpoints)
   - [Flags and properties](#flags-and-properties)
   - [Materialization of derived tensors](#materialization-of-derived-tensors)
   - [Activation quantization](#activation-quantization)
   - [vLLM online quantization (`--quant-online`, `--quantization-config`)](#vllm-online-quantization---quant-online---quantization-config)
   - [Model sampling defaults](#model-sampling-defaults)
   - [Supported quantization formats](#supported-quantization-formats)
   - [Rejected formats](#rejected-formats)
   - [Exporting to GGUF](#exporting-to-gguf)
   - [Supported architectures](#supported-architectures)
   - [vLLM architecture names](#vllm-architecture-names)
   - [Loading a model from object storage](#loading-a-model-from-object-storage)
   - [vLLM state-dict checkpoints](#vllm-state-dict-checkpoints)
6. [Hugging Face Hub download](#hugging-face-hub-download)
7. [nftool](#nftool)
   - [nftool bench](#nftool-bench)
   - [Serving and offline benchmarks](#serving-and-offline-benchmarks)
   - [nftool embedding](#nftool-embedding)
   - [nftool retrieval](#nftool-retrieval)
   - [nftool batched-bench](#nftool-batched-bench)
   - [nftool pack-experts](#nftool-pack-experts)
   - [Batched and parallel examples](#batched-and-parallel-examples)
   - [nftool tokenize](#nftool-tokenize)
   - [nftool dump](#nftool-dump)
   - [nftool hash](#nftool-hash)
   - [nftool set-metadata](#nftool-set-metadata)
   - [nftool new-metadata](#nftool-new-metadata)
   - [nftool convert-endian](#nftool-convert-endian)
   - [nftool results, debug and eval-callback](#nftool-results-debug-and-eval-callback)
   - [nftool eval](#nftool-eval)
   - [nftool finetune](#nftool-finetune)
   - [nftool split](#nftool-split)
   - [nftool merge](#nftool-merge)
   - [nftool lookup-create, lookup-merge and lookup-stats](#nftool-lookup-create-lookup-merge-and-lookup-stats)
   - [nftool fit-params](#nftool-fit-params)
   - [nftool slice](#nftool-slice)
   - [nftool perplexity](#nftool-perplexity)
   - [nftool passkey](#nftool-passkey)
   - [nftool quantize](#nftool-quantize)
   - [nftool export](#nftool-export)
   - [nftool export-lora](#nftool-export-lora)
8. [HTTP server](#http-server)
   - [Beam search](#beam-search)
   - [Prompt embeddings](#prompt-embeddings)
   - [Score, classify and pooling](#score-classify-and-pooling)
   - [Batched pooling beside generation](#batched-pooling-beside-generation)
   - [CLIP image and text embeddings](#clip-image-and-text-embeddings)
   - [Media embeddings from vision-language models](#media-embeddings-from-vision-language-models)
   - [jina_vl multimodal reranker](#jina_vl-multimodal-reranker)
   - [CLIP and SigLIP dual encoders](#clip-and-siglip-dual-encoders)
   - [Cohere embed API](#cohere-embed-api)
   - [Decision models](#decision-models)
   - [Cohere Chat v2](#cohere-chat-v2)
   - [Tool calls](#tool-calls)
   - [Server tools and MCP](#server-tools-and-mcp)
   - [Web UI](#web-ui)
   - [API keys](#api-keys)
   - [OpenAI Responses](#openai-responses)
   - [SageMaker hosting endpoints](#sagemaker-hosting-endpoints)
   - [Vertex AI prediction](#vertex-ai-prediction)
   - [Launching the server](#launching-the-server)
   - [Image and audio input (`--mmproj`)](#image-and-audio-input---mmproj)
   - [Video input](#video-input)
   - [Batch-invariant mode](#batch-invariant-mode)
   - [Media cache keys (`uuid`) and the encoded-media cache](#media-cache-keys-uuid-and-the-encoded-media-cache)
   - [Multi-model router](#multi-model-router)
   - [Data-parallel serving](#data-parallel-serving)
   - [Distributed serving control](#distributed-serving-control)
   - [Disaggregated prefill (`--kv-transfer-config`, `kv_transfer_params`)](#disaggregated-prefill---kv-transfer-config-kv_transfer_params)
   - [Per-request LoRA adapters](#per-request-lora-adapters)
   - [Slots](#slots)
   - [Choosing a chat template](#choosing-a-chat-template)
   - [Chat template capabilities](#chat-template-capabilities)
   - [Per-request chat template and reasoning output](#per-request-chat-template-and-reasoning-output)
   - [Reasoning budget](#reasoning-budget)
   - [Offline batch runner](#offline-batch-runner)
   - [Batched chat completions](#batched-chat-completions)
   - [Scale-out serving](#scale-out-serving)
   - [Watermarking generated text](#watermarking-generated-text)
   - [gRPC API](#grpc-api)
   - [Operational endpoints](#operational-endpoints)
   - [Resumable streams](#resumable-streams)
   - [Request tracing](#request-tracing)
   - [Sleep, pause and checkpoint](#sleep-pause-and-checkpoint)
9. [Anthropic Messages](#anthropic-messages)
10. [Reasoning and tool-call parsers](#reasoning-and-tool-call-parsers)
   - [Live weight updates](#live-weight-updates)
   - [Template-derived parsing (the autoparser)](#template-derived-parsing-the-autoparser)
   - [Content fallback (`nemotron_v3`, `granite_thinking_parser`)](#content-fallback-nemotron_v3-granite_thinking_parser)
   - [Request template variables](#request-template-variables)
   - [`hf` / `beam` parsers (vLLM response templates)](#hf--beam-parsers-vllm-response-templates)
11. [Grammar-constrained generation](#grammar-constrained-generation)
   - [JSON schema coverage](#json-schema-coverage)
   - [Structured-output JSON options](#structured-output-json-options)
   - [Lark grammars](#lark-grammars)
   - [Regex constraints](#regex-constraints)
   - [Structural tags: the xgrammar `format` object](#structural-tags-the-xgrammar-format-object)
12. [Prefix cache](#prefix-cache)
   - [Context shift](#context-shift)
   - [Self-extend (group attention) and position division](#self-extend-group-attention-and-position-division)
   - [KV chunk reuse (`--cache-reuse`, `n_cache_reuse`)](#kv-chunk-reuse---cache-reuse-n_cache_reuse)
   - [Context checkpoints (prefix reuse for recurrent and sliding-window models)](#context-checkpoints-prefix-reuse-for-recurrent-and-sliding-window-models)
13. [Paged KV memory](#paged-kv-memory)
14. [Media prompts](#media-prompts)
   - [Media from the command line](#media-from-the-command-line)
   - [Vision encoders (mmproj files)](#vision-encoders-mmproj-files)
   - [Audio input](#audio-input)
15. [Control vectors](#control-vectors)
16. [imatrix](#imatrix)
17. [Session files](#session-files)
   - [Prompt cache](#prompt-cache)
18. [System properties](#system-properties)
   - [Engine (`EngineConfig.fromSystemProperties`)](#engine-engineconfigfromsystemproperties)
   - [Server (`EngineLifecycle`, `ServerOptions.fromSystemProperties`, `ApiKeys`)](#server-enginelifecycle-serveroptionsfromsystemproperties-apikeys)
   - [Hugging Face checkpoints (`ModelSourceOptions.fromProperties`)](#hugging-face-checkpoints-modelsourceoptionsfromproperties)
   - [Memory (`MemoryConfig.fromProperties`)](#memory-memoryconfigfromproperties)
   - [Sampling (`SamplerConfig.fromProperties`)](#sampling-samplerconfigfromproperties)
   - [Loading, CPU and diagnostics](#loading-cpu-and-diagnostics)
   - [Backend selection and tuning](#backend-selection-and-tuning)
19. [Troubleshooting](#troubleshooting)

## Build

Requirements:
- **JDK 11 or newer.** Sources compile with `--release 11`.
- **The Gradle wrapper**, `./gradlew`, which downloads Gradle 8.14.4.

| Task | What it does |
|---|---|
| `./gradlew assemble` | Compiles, builds the GPU kernels the available tools allow, and packages the jar |
| `./gradlew build` | `assemble`, then the unit tests |
| `./gradlew test` | Runs the unit tests in `src/test/java`; tests that need a GPU API the machine lacks are skipped |
| `./gradlew runApp` | `NeuralFree.main`: creates the training-framework backend for `neuralfree.backend` (default `auto`), logs whether the train kernels run on it, and checks a buffer round trip |
| `./gradlew runDemos` | The framework demos (`DemoRunner`) on one shared backend: `AUTO` over OpenCL, CUDA and Vulkan by default |
| `./gradlew runLLMDemo -Pargs="..."` | The GGUF command line (`LLMDemo`); `-Pheap=<MB>` sets the heap (default 8192) |
| `./gradlew runBatchedDemo -Pargs="..."`, `runParallelDemo` | The [batched and parallel examples](#batched-and-parallel-examples) |
| `./gradlew verifyGpuKernels` | Builds every GPU kernel artifact, syntax-checks the OpenCL program of every kernel library with clang, and checks that CUDA, HIP and Vulkan cover each library's kernel inventory (HIP against its portable kernels: the `[cuda]` mma sources are PTX-only and compile on no HIP pass) |
| `./gradlew compileHipKernels` | Compiles every kernel library to HIP code objects with hipcc (`build/generated/gpu/hip/kernels/<lib>/hip/`): one multi-target `.hsaco` per kernel source, with `index.tsv` (kernel name → code object resource) and `build.properties` (`hipcc.version`, `hip.archs`, `kernel.format`, and `fa.smem.floats` for the LLM library). This is what the HIP backend loads from `kernels/<lib>/hip` on the classpath. |
| `./gradlew buildHipShim` | Builds the HIP JNI shim `libhipshim.so` from `native/hip/hipshim.c` (needs hipcc and a JDK's JNI headers) into `build/generated/gpu/hip-shim/`; point the engine at it with `-Dneuralfree.hip.shim.path=<file>` (or install it on the loader path as `libhipshim.so`) |
| `./gradlew inventoryCheck` | Checks every kernel library's inventory against its kernel sources; `build` runs it before packaging the resources |
| `./gradlew updateKernelInventory` | Regenerates the checked-in inventory of every kernel library from its kernel sources (needs clang or nvcc) |
| `./gradlew installDist` | Builds the installable distribution in `build/install/neuralfree/` ([The neuralfree command](#the-neuralfree-command)) |
| `./gradlew publishToMavenLocal` | Installs `com.eonmux:neuralfree:1.0.0-SNAPSHOT` in the local Maven repository; the POM carries the JCuda and LWJGL natives of the build platform |

**Choosing the demos' backend.** `DemoRunner` takes `--backend auto|opencl|cuda|vulkan|hip`; without it the `neuralfree.backend` property decides, and `auto` is the default. `runDemos` passes program arguments with Gradle's `--args`, and every Gradle run task forwards `neuralfree.*` properties to the program's JVM:

```bash
./gradlew runDemos --args='--backend cuda'
./gradlew runDemos -Pneuralfree.backend=vulkan
```

`cpu` is rejected for the framework: its tensors live on a GPU backend, and each op takes its CPU path when the train kernels are unavailable there. On an invalid value `DemoRunner` prints the usage and exits with status 2.

### The neuralfree command

`./gradlew installDist` builds an installable distribution in `build/install/neuralfree/`: the start scripts `bin/neuralfree` and `bin/neuralfree.bat`, the jars in `lib/`, and the HTTP server's packaged jar in `server/`. `./gradlew distZip` and `distTar` build the archives. Every program is a subcommand of the one entry point, as in llama.cpp's app binary and vLLM's CLI:

| Command | Runs |
|---|---|
| `serve` (`server`) | The HTTP server ([Launching the server](#launching-the-server)) |
| `run-batch` | The [offline batch runner](#offline-batch-runner) over the server jar, as `vllm run-batch` |
| `cli` (`client`) | The LLMDemo command line, with its options unchanged |
| `chat`, `complete` | Stream chat or text completions from a running OpenAI-compatible server, as `vllm chat` and `vllm complete` ([OpenAI client commands](#openai-client-commands)) |
| `bench` | [`nftool bench`](#nftool-bench), llama-bench's sweeps (the CLI's single-run profile stays `cli --benchmark`) |
| `download` (`get`) | `nftool download` |
| `cache-list` | The models in the hub cache |
| `rpc-server` | The GPUs of this machine served to a remote layer split ([Remote compute servers](#remote-compute-servers-layer-split-offload)) |
| `bench-serve`, `bench-throughput`, `bench-latency`, `bench-sweep`, `bench-startup` | [vLLM's serving, offline, latency, sweep and startup benchmarks](#nftool-bench-serve) |
| `eval` | [`nftool eval`](#nftool-eval), task accuracy of a running server |
| `quantize`, `perplexity`, `finetune`, `imatrix`, `cvector`, `export`, `passkey` | The [nftool](#nftool) subcommands of that name ([`finetune`](#nftool-finetune) trains a GGUF model on a text corpus) |
| `collect-env` | The environment facts a bug report needs |
| `version`, `help [all]` | The version; the commands (`all` adds the further nftool commands and `train`) |
| `train` | The training framework's `NeuralFree.main` |

A command may also be spelled as a flag (`neuralfree --version`). `help all` also lists the further nftool commands (`slice`, `passkey`, `results`, `debug`, `eval-callback`, `batched-bench`, `split`, `merge`, `export-lora`, `tokenize`, `dump`, `hash`, `set-metadata`, `new-metadata`, `convert-endian`, `embedding`, `retrieval`, `lookup-create`, `lookup-merge`, `lookup-stats`, `pack-experts`, `fit-params`); each runs as its nftool subcommand, and an unknown command prints the commands and exits with status 2. The jar runs the same way (`java -jar neuralfree-<version>.jar <command>`), and so does Gradle: `./gradlew run -Pargs="--list-devices"`.

Information flags, on the launcher, on `cli` and (for `--list-devices`) on `serve` (llama.cpp's common arguments):

| Flag | Prints |
|---|---|
| `--version` | The version: the `neuralfree.version` property, else the jar manifest's `Implementation-Version`, else `development` |
| `--list-devices` | The devices OpenCL, CUDA, Vulkan and HIP enumerate, with their total memory, one `<API> <index>: <name> (<MiB> MiB)` line each; the index is the 0-based position in that API's list, the number `-dev` and `-devd` select it by (pick the API with `--backend`) |
| `-cl`, `--cache-list` | The hub cache's models, one `<n>. <user>/<model>:<TAG>` line per cached repository and quant tag (also on `nftool download`) |

`neuralfree serve` starts the server jar in a child JVM and waits for it; Ctrl-C stops it. It looks for the jar in the `neuralfree.cli.server.jar` property, then the `NEURALFREE_SERVER_JAR` environment variable, then the distribution's `server/` directory, then the source checkout's `server/build/`, and without one names every place it searched. Its `-D<name>=<value>` arguments become JVM properties of the server, and the rest are the server's own flags:

```console
neuralfree serve -m /models/m.gguf --port 8081
neuralfree serve -Dneuralfree.server.model=/models/m.gguf -Dquarkus.http.port=8081
```

The server launch also takes `-dev` / `-device` / `--device` with the same selectors as `cli`, mapped onto `neuralfree.gpu.devices`, `neuralfree.device`, `neuralfree.split.mode=layer` or `neuralfree.backend=cpu`, and refuses the same lists `cli` refuses; `neuralfree serve --list-devices` prints the device report and exits, as `llama-server --list-devices` does.

`neuralfree collect-env` prints the version, the Java runtime, the operating system, the devices, the hub cache, where the server jar resolves and the engine's environment variables; `HF_TOKEN` and `NEURALFREE_API_KEY` show only whether they are set.

### OpenAI client commands

`neuralfree chat` and `neuralfree complete` stream answers from a running OpenAI-compatible server — one started
by `neuralfree serve`, llama-server, vLLM or any OpenAI-compatible endpoint — as `vllm chat` and `vllm complete`
do. They are also `nftool chat` and `nftool complete`. Nothing is loaded locally.

Both take the OpenAI query options:

| Flag | Meaning | Default |
|---|---|---|
| `--url URL` | The OpenAI API base URL. Endpoints are appended to it, so `http://host:8000/v1` posts to `/v1/chat/completions` and `/v1/completions`, and a server behind an `--api-prefix` takes the prefixed base | `http://localhost:8000/v1` |
| `--model-name NAME` | The `model` of every request | The first model the server's `/models` lists |
| `--api-key KEY` | The bearer token | `OPENAI_API_KEY`, else `NEURALFREE_API_KEY`, else `EMPTY` |
| `--stats` | Print `TTFT: <ms> ms` and `TPS: <n> tokens/s (<n> tokens in <s>s)` after each answer; the request asks for `stream_options.include_usage` | off |
| `-h`, `--help` | The command's usage | |

`chat` also takes `--system-prompt TEXT` (the system message that opens the conversation) and
`-q/--quick MESSAGE`; `complete` takes `--max-tokens N` and `-q/--quick PROMPT`. A flag of the other command, an
unknown option and a positional argument are refused by name.

Without `--quick`, the command reads one line at a time from standard input until the end of input, as the
reference's `input("> ")` loop does. In `chat`, every line is a user message and each answer is appended to the
conversation as the assistant's content; in `complete`, every line is one prompt. Reasoning streams between
`<think>` and `</think>`, and the generated content follows. `-q/--quick` sends one message or prompt, prints the
answer and exits.

```console
neuralfree chat
neuralfree chat -q "Summarize the TCP handshake" --stats
neuralfree chat --url http://127.0.0.1:8081/v1 --model-name my-model --system-prompt "Be terse"
neuralfree complete --max-tokens 64 -q "The capital of France is"
printf 'first prompt\nsecond prompt\n' | neuralfree complete
```

The default `--url` is vLLM's. `neuralfree serve` binds `8080` by default, so a server it started takes
`--url http://localhost:8080/v1` (or `neuralfree serve --port 8000`).

The answer prints as it streams. A request the server refuses prints the server's message on standard error and
exits 1; an invalid command line prints the usage and exits 2. In the launcher's command table, `chat` and
`complete` sit with `cli` as the clients of a running server, and both run without the vector-module relaunch the
in-process engine commands make.

### GPU kernel tools (all optional)

OpenCL needs no build tool: the driver compiles the kernel sources at run time. The CUDA modules and the Vulkan SPIR-V of every kernel library are built by `gradle/gpu-kernels.gradle` during `processResources`. Each tool is found in this order:
1. its `-P` property;
2. the listed directories;
3. `PATH`.

| Tool | Used by | Found in | When it is missing |
|---|---|---|---|
| `nvcc` | `compileCudaKernels`; `inventoryCheck` and `updateKernelInventory` when clang is missing | `-Pnvcc`, `$CUDA_HOME/bin`, `$CUDA_PATH/bin`, `/opt/cuda/bin`, `/usr/local/cuda/bin` | Warning; no CUDA modules are built. The CUDA backend reports both kernel libraries unavailable. |
| `hipcc` | `compileHipKernels`, `buildHipShim` | `-Pneuralfree.hipcc`, `$ROCM_PATH/bin`, `$ROCM_HOME/bin`, `$HIP_PATH/bin`, `/opt/rocm/bin`, `/usr/bin`, `/usr/local/bin` | Warning; no HIP code objects and no shim are built. The HIP backend reports both kernel libraries unavailable, like the CUDA backend without nvcc. |
| `glslc` (fallback `glslangValidator`) | `compileShaders`, `compileTestShaders` | `-Pneuralfree.glslc`, `$VULKAN_SDK/bin`, `/usr/bin`, `/usr/local/bin` | Warning; no SPIR-V is built, so the Vulkan backend has no kernels |
| `spirv-dis` | Shader argument signatures | next to the shader compiler, then the directories above | Warning; no Vulkan kernels of either library are built |
| `spirv-opt` | Shader optimization | as above | Warning; the SPIR-V ships unoptimized |
| `spirv-val` | SPIR-V validation | as above | Warning; modules are not validated |
| `clang` | `checkOpenCLKernels` (OpenCL C 1.2 syntax check), `inventoryCheck`, `updateKernelInventory` | `-Pneuralfree.clang`, `/usr/bin`, `/usr/local/bin` | Warning; the syntax check is skipped, and the inventory tasks preprocess with nvcc instead, without checking the OpenCL-only kernels. With neither clang nor nvcc, see [Kernel inventories](#kernel-inventories). |

A backend that cannot run a kernel library still works for everything else. For the GGUF engine, AUTO skips it and uses the next backend. For the training framework, AUTO prefers a backend that runs the `train` kernels; on one that does not, every tensor op takes its CPU path.

Build properties:

| Property | Meaning | Default |
|---|---|---|
| `-Pneuralfree.requireCuda=true` | Fail instead of warning when nvcc is missing | false |
| `-Pneuralfree.requireShaders=true` | Fail instead of warning when the shader compiler or spirv-dis is missing | false |
| `-Pneuralfree.cuda.archs=<list>` | SM list, intersected with what nvcc supports. Builds SASS for each, plus PTX for the lowest and the highest. | `75,80,86,89,90,100,120` |
| `-Pneuralfree.cuda.ptxOnly=true` | Emit PTX for the lowest arch only, as a quick compile check | false |
| `-Pneuralfree.hip.archs=<list>` | Comma-separated gfx targets; every unit compiles once per target (`--offload-arch=<target>`), producing one multi-target code object. Requested targets that hipcc's supported-target listing excludes are dropped with a warning; a listing that cannot be read passes every target through. | `gfx906,gfx908,gfx90a,gfx942,gfx1030,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201` (the llama.cpp HIP build's set) |
| `-Pneuralfree.requireHip=true` | Fail instead of warning when hipcc is missing | false |
| `-Pneuralfree.hipcc=<path>` | hipcc location, searched before the default directories | none |
| `-Pnvcc=<path>`, `-Pneuralfree.glslc=<path>`, `-Pneuralfree.clang=<path>` | Tool locations, searched before the default directories | none |
| `-Pnvcc.ccbin=<path>` | Host compiler passed to nvcc as `-ccbin` | none |
| `-Pneuralfree.gpu.libraries=<ids>` | Comma-separated kernel library ids (`llm`, `train`) that the kernel tasks build and check. For developer speed only: the other libraries' CUDA and Vulkan artifacts are then missing from the build, and the build warns. An unknown id fails the build. | all |

### Kernel libraries

The GPU kernels are organized in two kernel libraries, each built from one set of dual-dialect sources into OpenCL C (compiled by the driver at run time), CUDA modules and Vulkan SPIR-V:

| Library | Serves | Sources | Build profile |
|---|---|---|---|
| `llm` | The GGUF engine | `kernels/llm`, `shaders/llm` | Fast-math CUDA (`-ftz=true -prec-div=false -prec-sqrt=false -fmad=true`), subgroup variants |
| `train` | Tensors, training and classical ML | `kernels/train`, `shaders/train` | Precise CUDA math (`-ftz=false -prec-div=true -prec-sqrt=true -fmad=false`), OpenCL `-cl-std=CL1.2` with no relaxed-math option, no subgroup variants (reductions use a fixed-order tree on every backend) |

Paths are under `src/main/resources`. `kernels/<lib>/manifest.txt` lists a library's sources in build order. A `[cuda]` entry builds only into the CUDA modules: the OpenCL program leaves it out, it needs no Vulkan variant row, and the Vulkan backend does not require its kernels. An `[opencl]` entry builds only into the OpenCL program, with no CUDA build, no Vulkan variant and no inventory row; the four such files hold the Intel subgroup forms below. The headers both libraries share are in `kernels/common` (for example `common/dialect.h`, listed as a `[shared]` manifest entry) and `shaders/common` (`nf_base.glsl`); `shaders/test` holds the backend test shaders. Every other compute shader belongs to a kernel library, and `compileShaders` rejects one outside them. The generated artifacts go to `build/generated/gpu/{cuda,spv,spv-test,check}` and are merged into the resources.

On an OpenCL device with subgroup reductions and `cl_intel_required_subgroup_size`, the `llm` program pins its matvec reductions to the largest subgroup size the device supports (`-DNF_REQD_SG_SIZE=<size>`, program cache key suffix `@<size>`). Every `{T}_MatVec` / `{T}_MatVecN` pair then reduces over one width, so their outputs are bitwise equal, and the matvec reductions, which are bandwidth-bound, run at the device's widest width (on the Arrow Lake iGPU a 9-row step of a Qwen3 0.6B Q8_0 model takes 669 ms at width 32 against 760 at 16 and 1097 at 8). A device without the extension keeps the compiler's per-kernel choice.

On an OpenCL device that also accepts a required subgroup size of 16 (the Intel GPUs), the `llm` program is built with `-DNF_SG_ROW_SIZE=16` (cache key suffix `+sgr16`), which compiles the Intel subgroup forms of the `[opencl]` sources. The dispatcher uses each form where the program holds it and keeps the portable kernel everywhere else:

| Source | Kernels | Replaces | Arrow Lake iGPU |
|---|---|---|---|
| `matvec_sg.cl` | `{T}_MatVecSg_Q8`, `{T}_MatVecAddSg_Q8`, `{T}_MatVecFusedGateUpSg_Q8` (Q8_0, Q4_K, Q6_K) | the one-token integer-dot matvecs | weights stream at 41-61 GB/s against 5-12 |
| `matmul_mmq_sg.cl` | `{T}_MatMulSg_MMQ` (Q8_0, Q4_K, Q6_K) | `{T}_MatMul_MMQ` outside split-K | 0.8-2.0 TFLOPS at 512 tokens, 4-6x the portable kernel |
| `flash_decode_sg.cl` | `LLM_FlashDecodeSg_<KV>`, `LLM_FlashDecodePagedSg_<KV>` (F32, F16, BF16; head dim a multiple of 8 up to 128) | the decode flash attention | 38 us per layer on Qwen3 0.6B against 253 |
| `flash_prefill_sg.cl` | `LLM_FlashAttnSg_<KV>`, `LLM_FlashAttnPagedSg_<KV>` (same caches and head dims) | the tiled prefill flash attention | 2.6 ms per layer at 512 tokens on Qwen3 0.6B against 417 |

They keep their state in registers across 16-lane subgroups instead of staging it in local memory, read the weights and cache rows with 16-byte loads, and share activations through subgroup shuffles. Their float summation order differs from the portable kernels', so `-Dneuralfree.batch.invariant=true` keeps the portable matvec and GEMM forms. The attention forms are chosen per layer for the dense and the paged plans together, so a step's result does not depend on which plan runs it.

What each API needs to run a library:

| API | `llm` | `train` |
|---|---|---|
| OpenCL | An OpenCL 1.2 device | An OpenCL 1.2 device |
| CUDA | Modules built with nvcc covering the library's inventory; a SASS image for the device's major architecture (minor not above the device's) or a PTX image of an architecture not above it; a driver new enough for those images | The same |
| Vulkan | SPIR-V built with glslc and spirv-dis covering the inventory; a Vulkan 1.2 device with 8-bit storage, 16-bit storage and timeline semaphores | SPIR-V covering the inventory; a Vulkan 1.2 device with timeline semaphores (sub-word data is accessed as 32-bit words) |
| HIP | Code objects built with hipcc covering the library's portable inventory, built for the device's processor (`gcnArchName` among `hip.archs`, capability suffixes ignored), and `libhipshim` loadable (on the loader path or via `-Dneuralfree.hip.shim.path`) | The same |

On every API the `train` library also needs work-groups of at least 256 items, and every kernel it lists must load. OpenCL programs are cached in `~/.cache/neuralfree/opencl` and Vulkan pipelines in `~/.cache/neuralfree/vulkan/`.

### Kernel inventories

`src/main/resources/kernels/llm/inventory.txt` and `src/main/resources/kernels/train/inventory.txt` list every kernel that CUDA and Vulkan must provide for each library, with its argument layout. They are generated from the kernel sources and checked in. A library's CUDA and Vulkan sides are available only when their built kernels cover its inventory.
- **`inventoryCheck`** runs before `processResources`, so `build` runs it. With clang, or nvcc's preprocessor when clang is missing, it checks the kernels' argument limits and that the shader-variant rows match the kernels, regenerates each inventory in memory, and fails the build when a checked-in file differs.
- **Without clang and nvcc** it checks only each file's format, that each kernel's source is an untagged or `[cuda]` `.cl` entry of the library's `manifest.txt`, and that the shader-variant rows match the listed kernels. The build warns that the inventories' freshness was not checked and ships the files as checked in. The Vulkan backend still runs both libraries from them.
- **After a kernel change** that adds, removes or renames a kernel, or changes its source file or arguments, run `./gradlew updateKernelInventory` and commit the regenerated file. Until then, `inventoryCheck`, and with it the build, fails whenever clang or nvcc is present, and names the regeneration command.

### GPU kernel build cache

Gradle's build cache is enabled for this project (`org.gradle.caching=true` in `gradle.properties`), and the tasks
that produce GPU kernel artifacts are cacheable: `compileCudaKernels` (CUDA fatbins/PTX), `compileHipKernels` (HIP
code objects), `compileShaders` and `compileTestShaders` (Vulkan SPIR-V) and `buildHipShim` (`libhipshim.so`). Each
task fingerprints the sources and headers it reads, the shader includes and variant tables, the per-library flags
and defines, the arch selection, `-Pnvcc.ccbin`, `-Pneuralfree.cuda.ptxOnly`, the selected kernel libraries and the
toolchain (executable name and `--version` output, with absolute paths removed). Paths are relative, so a second
checkout of the same commit at another path reuses the artifacts instead of recompiling them:

```bash
# first checkout
./gradlew processResources --offline --console=plain

# second checkout of the same commit, same toolchain
./gradlew processResources --offline --console=plain
# > Task :compileCudaKernels FROM-CACHE
# > Task :compileHipKernels FROM-CACHE
# > Task :compileShaders FROM-CACHE
# > Task :compileTestShaders FROM-CACHE
# > Task :buildHipShim FROM-CACHE
```

Changing a kernel or shader source, a shared header, a library's flags, the arch list (`-Pneuralfree.cuda.archs`,
`-Pneuralfree.hip.archs`), `-Pneuralfree.cuda.ptxOnly` or the toolchain produces a new key and the task runs again.
Entries live in the local Gradle cache (`~/.gradle/caches/build-cache-1`); `--no-build-cache` bypasses them for one
run and `./gradlew clean` does not remove them. HIP artifacts are only built and cached when `hipcc` is available.
`checkOpenCLKernels` and `inventoryCheck` are verification tasks and stay uncached, so they always check the current
sources.

### Logging

NeuralFree logs through the Log4j 2 API; `log4j-core` is its runtime implementation.
- **The Gradle run tasks** (`runApp`, `runDemos`, `runLLMDemo`) pass `-Dlog4j2.configurationFile=src/main/config/log4j2.xml`, which logs INFO and above to the console.
- **The library jar ships no logging configuration.** An application configures Log4j itself; without a configuration, Log4j's default prints errors only. To use the run tasks' configuration with `java`, pass `-Dlog4j2.configurationFile=<checkout>/src/main/config/log4j2.xml`.
- **The tests** log INFO and above to the console through `src/test/resources/log4j2-test.xml`.

## Compute backends

NeuralFree drives GPUs through four APIs behind one `ComputeBackend` interface: OpenCL, CUDA, Vulkan and HIP (AMD ROCm; [below](#amd-gpus-the-hip-backend)). `Config.ACCELERATION_MODE` names the choice:

| Mode | Enumerates | Picks |
|---|---|---|
| `AUTO` | Every API | Device first, then API (below) |
| `VULKAN` | Vulkan only | The best Vulkan device |
| `CUDA` | CUDA only | The best CUDA device |
| `OPENCL` | OpenCL only | The best OpenCL device |
| `HIP` | HIP only | The best HIP device |

```java
import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

// Every API, device-first selection
HWAcceleration hw = HWAcceleration.forMode(Config.ACCELERATION_MODE.AUTO);
ComputeBackend auto = hw.createBackend(Config.ACCELERATION_MODE.AUTO);

// Force one API
HWAcceleration vkHw = HWAcceleration.forMode(Config.ACCELERATION_MODE.VULKAN);
ComputeBackend vulkan = vkHw.createBackend(Config.ACCELERATION_MODE.VULKAN);

// ... use the backends ...
hw.shutdown();     // closes the backends this instance created, then releases its devices
vkHw.shutdown();
```

`new HWAcceleration()` enumerates OpenCL only, unless the static flags `HWAcceleration.enableCuda` and `HWAcceleration.enableVulkan` are set before construction. `HWAcceleration.forMode(mode)` ignores those flags, and `new HWAcceleration(Set<BackendType>)` enumerates exactly the APIs given. `createBackend` throws `NoBackendAvailableException` when no backend can be created. Device memory is not managed by the JVM: `shutdown()` closes every backend the instance handed out.

**AUTO is device-first** (`BackendSelector`):
1. **Target device.** The target is the first device, in any API, whose name contains `neuralfree.device` (case-insensitive). If none is named, it is the OpenCL ranking's winner (discrete before integrated, then measured), else Vulkan's, else CUDA's.
2. **API order.** Each API in `neuralfree.backend.order` (default `metal,vulkan,cuda,opencl,hip`) is tried on the target device. The first one that is accepted wins. With the default order, the target vendor's native API goes first: CUDA on an NVIDIA GPU, where it measured 4-5x the prefill and 1.6-2x the generation speed of Vulkan and OpenCL on an RTX 5080 Laptop, and OpenCL on an Intel GPU, whose program carries the Intel subgroup forms (on the Arrow Lake iGPU 38x the Qwen3 0.6B prefill and 1.8-2.5x the generation speed of Vulkan). A device whose native API is not installed or not accepted falls through to the rest of the order. An order set through the property is followed as written. The views one GPU offers through several APIs are recognised as one device.
3. **Fallback.** If no API runs the target, each API's own winner is tried, discrete first.

The per-API properties `neuralfree.opencl.device`, `neuralfree.cuda.device`, `neuralfree.vulkan.device` and `neuralfree.hip.device` override `neuralfree.device` within one API. `neuralfree.backend.order=cuda,opencl`, for example, keeps AUTO off Vulkan.

**Requirements.** `createBackend(mode, requirement)` adds a condition the backend must meet: AUTO skips, closes and logs every backend the requirement rejects, and an explicit mode fails with `<API> backend unavailable: <reason>`.

**LLM inference** uses the requirement `KernelLibraryRequirement.of(KernelLibraries.LLM)`, so AUTO never selects a backend that cannot serve the LLM kernels. Choose the LLM backend with `--backend auto|opencl|cuda|vulkan|hip|cpu`, `--cpu`, `--device <substr>` or a multi-GPU `--device <d1,d2>`, or with the `neuralfree.backend` and `neuralfree.device` properties. If the GPU load fails, the model opens on the CPU engine.

**The training framework** runs its GPU kernels on all three APIs: its tensor, layer, loss, optimizer and classical-ML ops dispatch to the `train` kernel library. Create its backend with `NeuralFree.createTrainingBackend(hw, mode)`:
- **`AUTO`** selects, device first, a backend on which the train library is available (`TrainKernels.requirement()`) and its context opens, which also checks the work-group size and prewarms every kernel. When a device was found but no API serves the library on it, a WARN gives the reason for each API, and the backend AUTO picks without the requirement is returned; tensor ops on it take their CPU path.
- **An explicit API** creates that API's backend. When the library is unavailable there, the same WARN is logged and the backend is still returned.
- `NeuralFree.describeTrainKernels(backend)` returns `available` or `unavailable (tensor ops take their CPU path): <reason>`. `runApp` and the demos print it.

```java
import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.NeuralFree;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

Config.ACCELERATION_MODE mode = NeuralFree.backendFromProperty();   // -Dneuralfree.backend, default AUTO
HWAcceleration hw = HWAcceleration.forMode(mode);
try {
    ComputeBackend backend = NeuralFree.createTrainingBackend(hw, mode);
    System.out.println("Train kernels: " + NeuralFree.describeTrainKernels(backend));
    // ... build and train models on backend ...
} finally {
    hw.shutdown();
}
```

### Tensor-core flash attention

On a CUDA device of compute capability 8.0 or newer, prefill attention (prompts, chunked prefill steps, batched rows and speculative verification) runs on the tensor cores through `LLM_FlashAttnMma_F16` and `LLM_FlashAttnMmaPaged_F16`: Q K^T and P V use `mma.sync` m16n8k16 F16 fragments with F32 accumulation, the mechanism of llama.cpp's `fattn-mma-f16`. `GpuAttentionMixer` picks it when all of these hold, and otherwise runs the scalar tiled kernels:

- the K and V caches are F16;
- both head dimensions are multiples of 16, at most 128;
- the layer uses standard attention (MLA keeps its own path).

Split-K over long contexts, sinks, softcap, ALiBi, the T5 relative bias, sliding windows, causal and bidirectional masks and paged caches behave as in the scalar kernels; the outputs differ only by the F16 rounding of the staged Q, K and V tiles. Single-row decode steps run it too, as a one-row prefill (48 against 65 us per layer on the scalar decode kernel at 256 keys, 32 query and 8 KV heads of 128), except on CUDA and HIP past 128 live keys where a KV head serves at most two query heads: those steps run the split-K decode kernel's warp-shuffle form (128 tokens generated after 512 cached on the RTX 5080 Laptop: 320 against 277 t/s on Qwen3-Embedding-0.6B, 201 against 187 on Qwen3-VL-Embedding-2B; with four query heads per KV head the one-row prefill stays ahead at every depth). The paged kernel's single-row steps keep the paged decode kernel; on CUDA with an F16 or BF16 cache that is the grouped-query `LLM_FlashDecodePagedGqa_F16` / `_BF16`, which reads each K and V row once for all query heads of a KV group (up to four) and splits the keys by the step's sequence count (16 sequences at 129 keys on Qwen3-Embedding-4B: 16 against 143 us per layer). On the RTX 5080 Laptop, a 512-token prefill over 1024 cached keys at head dimension 128 takes 0.38 ms against 1.84 ms on the scalar kernel. OpenCL and Vulkan run the scalar kernels.

### Fast GEMM for prefill and batched decode

A projection of a few tokens leaves the matvec family for the fast GEMM lane of its weight type at a row count
measured per device kind: the backend together with the PCI vendor, whether the device is integrated or discrete,
and on CUDA the int8 tensor band (sm_80 or newer). One GPU's measured crossover therefore never governs another
device of the same API. The measured kinds (`SmallBatchCrossover`):

| Device kind | Fast lane from rows | Float `<T>_MatVecN` up to rows | Int8 multi-column matvec up to rows |
|---|---|---|---|
| CUDA, NVIDIA discrete, sm_80+ | Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, IQ4_XS: 17, past the counted matvec (MMQ); F16: 4 (MMA) | none | Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, IQ4_XS: 16 |
| Vulkan, NVIDIA discrete | Q8_0: 4 (cooperative matrix); F16: 4 (MMA) | none | none |
| OpenCL, NVIDIA discrete | Q8_0, F16: none; Q4_0: 9 (MMQ, portable integer arithmetic); Q4_K, Q5_K, Q6_K, IQ4_XS: 17 below 8M elements, 16 on 8M-element projections and up (MMQ; the counted matvec serves through 15) | Q8_0 and F16: 16 | Q8_0: 16; Q4_K, Q5_K, Q6_K, IQ4_XS: 16 below 8M elements and 15 on 8M-element projections and up; Q4_0: 15 on 8M-element projections and up, 8 below |
| Vulkan, Intel integrated | Q8_0: 4 (cooperative matrix); F16: none | none | none |
| OpenCL, Intel integrated | Q8_0: 4 (MMQ); F16: none | F16: 16 | Q8_0: 3 |

The "Fast lane from rows" numbers for the measured quantized types are where the GEMM lane takes over from the
integer-dot matvec: the matvec's exact-token-count forms hold the rows below it where they measured faster. On
CUDA (NVIDIA discrete, sm_80+) the integer-dot lane runs its tensor-core form `<T>_MatVecMma{8,16}_Q8` where
`MatVecMmaTuning` measured it faster: Q4_K from 4 rows, Q6_K from 8, and Q8_0 from 9 rows on projections of 2048 rows
and up, from 10 on narrower ones and from 11 where the reduction is more than twice the row count; the counted forms
keep the rows below, and other types keep them through 16; on OpenCL NVIDIA they win through 15 on
the 8M-element shapes, while a 16-row step stays with the lane (its tile wins 9728x2560, 19456x2560 and the head
at 16 rows). The single counted form and its paired twin are chosen per launch: on OpenCL NVIDIA, Q4_K, Q5_K and
IQ4_XS take the single form at 9 or more tokens on launches of 9728 rows and up (the paired form wins the
transposed 2560-row launches), and every other measured shape pairs. Below 8M weight elements on the OpenCL NVIDIA
kind the matvec lane's evidence does not reach, so the GEMM lane keeps the crossover the defaults give.

A device of another vendor or class, or a type a kind does not list, keeps the defaults: the fast lane from 4 rows,
the int8 multi-column matvec below it (up to 8 rows without a fast lane), no float multi-column matvec, and the exact
tiled GEMM from 8 rows (`TILED_MIN_TOKENS`) where no fast lane runs. Rows below the lanes take the batched matvec.
If a kernel is unavailable the projection falls back automatically: fast GEMM → tiled GEMM → batched matvec. Steps narrower than the device's
token tile run a twin of the plain kernel on a shared record tile of 64 rows (256 work-items), chosen per launch by
`MmqConfig.twinTokens`: steps of at most 16 tokens (such as a speculative verify) run the 16-token twins
`<T>_MatMul_MMQ_S` and `<T>_MatMul_W4A8_S`, split along K into `<T>_MatMul_MMQ_SKSplit` planes when their grid would
not fill about 1.3 waves of the device; steps of 17 to 64 tokens run the 64-token twins `<T>_MatMul_MMQ_S64` and
`<T>_MatMul_W4A8_S64`, and every grid of up to six waves splits along K into up to eight `<T>_MatMul_MMQ_SKSplit64`
planes. On a reduction shorter than the split-K threshold (K below 2048, no split to fill the grid) the
16-token twin serves up to 48 tokens, where the 64-token tile would leave a quarter of its token columns empty.
Wider steps run the plain tile. The weight stream is a software pipeline, so the next tile's loads overlap the
current tile's dot products, and each 4-bit word decodes in two ALU ops. On an RTX 5080 Laptop (CUDA), `nftool batched-bench -c 4096 -npp 1 -ntg 128` decodes Qwen3-Embedding-4B
at an 8-row step of 11.0 ms in Q8_0 and 7.3 ms in Q4_K_M, and a 16-row step of 13.7 and 10.4 ms
(`llama-batched-bench -ngl 99 -fa on`: 9.6 and 6.7 ms at 8 rows, 10.3 and 8.1 at 16); a 512-token prefill (`nftool bench -p 512`) runs 2894 t/s in Q8_0 (`llama-bench` 9639) and
2758 t/s in Q4_K_M (llama.cpp 9140).

| Weights | Fast lane | Where | Rounding vs. the exact tiled GEMM |
|---|---|---|---|
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, I8_B128 | int8 MMQ GEMM (`<T>_MatMul_MMQ`) | every backend: int8 tensor cores (`mma.m16n8k32` / `m16n8k16` s8) on CUDA sm_80+, dp4a or scalar int dots elsewhere | activations are rounded to 8-bit codes per 32-element (tensor) or 16-element chunk, as in llama.cpp `mmq.cuh`; with an `--act-quant checkpoint` int8 scheme they are the checkpoint's own codes |
| same types | cooperative-matrix GEMM (`<T>_MatMul_CM`) | Vulkan devices with `VK_KHR_cooperative_matrix` (default `auto` mode) | weights and activations are staged as fp16 tiles, as in llama.cpp `mul_mm_cm2.comp` |
| F16 | tensor-core GEMM (`F16_MatMul_MMA`) | CUDA sm_80+, Vulkan with cooperative matrices | activations rounded to fp16; fp32 accumulation |
| BF16 | tensor-core GEMM (`BF16_MatMul_MMA`, bf16 `mma`) | CUDA sm_80+ (Vulkan keeps the tiled GEMM for BF16) | activations rounded to bf16; fp32 accumulation |
| F8_E4M3, F8_E4M3_B32, F8_E4M3_B128, MXFP8 | E4M3 GEMM over the checkpoint's own activation codes (`<T>_MatMul_MMQ`) | CUDA sm_89+ (`mma.sync` `e4m3.e4m3.f32`); elsewhere the exact tiled GEMM | activations staged as E4M3 codes with one scale per 32-element chunk; fp32 accumulation |
| MXFP4, NVFP4, IQ4_NL, IQ4_XS | E4M3 GEMM over E4M3-mapped 4-bit weight codes (`<T>_MatMul_W4A8`) | CUDA sm_89+ | weight codes are E4M3 (exact for the E2M1 codebooks, one E4M3 rounding for the IQ4 codebooks and the NVFP4 per-16 weight scales); activations staged as E4M3 |
| any quantized type with `INT8_STATIC_ASYM` | int8 MMQ GEMM with the zero-point term | as the int8 MMQ GEMM | the staged codes carry the checkpoint's zero point, which the GEMM subtracts per weight row |

The i-quant types run the same MMQ GEMM llama.cpp's `mmq.cu` serves them with, decoding the
i-quant grids from the tables the tiled GEMM uses. TQ1_0/TQ2_0 and the NeuralFree extension types
keep their tiled kernels (llama.cpp's MMQ does not serve them), except the E4M3 types and the E4M3 lanes below.

Before the MMQ GEMM, one `LLM_Quant_MMQ_Q8_1` launch quantizes every token row of the projection
once (llama.cpp `quantize_mmq_q8_1`), unless the checkpoint's activation scheme stages its own int8 codes
([activation quantization](#activation-quantization)), and every row tile reads those 8-bit codes. On CUDA sm_80+
the tensor tile is 128 rows × 128 tokens over a depth-3 software pipeline of 32-deep stages, so each weight word
decodes once per 128-token tile (four times at a 512-token projection); other devices use 16 × 16 tiles, following
llama.cpp's per-architecture MMQ configuration.

When a checkpoint carries an activation quantization scheme (`<base>.input_scale`, `<base>.input_zero_point`, as
vLLM's W8A8 and W4A8 checkpoints do), a projection no longer fake-quantizes its input to floats before the GEMM: one
staging kernel applies the scheme directly into the record layout the staged GEMMs read, and the GEMM multiplies the
checkpoint's own codes. `LLM_Quant_MMQ_Fp8Static`, `..._Fp8Row`, `..._Fp8Group128`, `..._Fp8Mx`, `..._Fp4Mx` and
`..._Fp4Nv` write E4M3 codes (`FP8_STATIC`, `FP8_TOKEN` with its amax bound, `FP8_GROUP128`, `MXFP8`, `MXFP4` and
`NVFP4`), `LLM_Quant_MMQ_Int8Static`, `..._Int8Row`, `..._Int8Group128` and `..._Int8Asym` write int8 codes
(`INT8_STATIC`, `INT8_TOKEN`, `INT8_GROUP128`, `INT8_STATIC_ASYM`), and `LLM_ActFakeQuant` is skipped for that
projection. The asymmetric form is computed in int8: the staging pass records the zero point, and the GEMM subtracts
`input_zero_point * sum_k W[r][k]` per stage from the weight code sums it reduces per row.

The E4M3 lanes need FP8 tensor cores, that is CUDA compute capability 8.9 and above. Below sm_89, on OpenCL and on
Vulkan — whose GLSL has no float8 component type for cooperative matrices — the projection keeps the exact tiled
GEMM with the fake-quantized input, exactly as vLLM falls back to its emulated GEMM when the cutlass FP8 kernels do
not apply; `-Dneuralfree.gemm.mmq=off` forces that form everywhere, for the E4M3 lanes as well. Tall-skinny 4-bit
launches split along K into `<T>_MatMul_W4A8_KSplit` planes like the int8 MMQ lane.

The staged values equal the fake-quantized activations of `FP8_STATIC`, `FP8_TOKEN`, `FP8_GROUP128`, `MXFP8` and
`MXFP4` exactly. `NVFP4` carries a scale per 16-element activation sub-block, which one E4M3 MMA scale per 32-element
chunk cannot express, so its staged value is the scheme's value rounded once to an E4M3 code of the chunk maximum (at
most half an E4M3 step, 6.25% of that maximum). The 4-bit weight codebooks are mapped to E4M3: exact for the E2M1
codebooks of MXFP4 and NVFP4, one E4M3 rounding for the IQ4 codebooks (at most 5.3%) and for the NVFP4 per-16 weight
scales.

To measure the lanes against the fake-quantization path:

    NEURALFREE_BENCH_FP8=1 ./gradlew :test --tests 'com.eonmux.neuralfree.gguf.ops.gemm.Fp8GemmBenchmarkTest'

On an RTX 5080 Laptop (sm_120), M = K = 4096: `F8_E4M3_B128` runs 2.07x (16 tokens) and 3.05x (512 tokens) faster
than the tiled GEMM over fake-quantized activations (13.9 TFLOP/s at 512 tokens), and `MXFP4` through
`MXFP4_MatMul_W4A8` 2.75x and 3.94x (17.7 TFLOP/s), the staging pass included.

A kernel the backend cannot build or prepare is skipped, the next form serves the product, and the
skip is logged once at info. The dispatch geometry is in `MmqGemmPlan`, `CmGemmPlan` and
`MmaGemmPlan`.

#### Choosing the quantized-weight lane: `-Dneuralfree.gemm.mmq`

| Value | Effect |
|---|---|
| `auto` (default) | int8 MMQ GEMM, or the cooperative-matrix form on Vulkan devices that support it |
| `on` | int8 MMQ GEMM on every device, also where `auto` would pick the cooperative-matrix form (llama.cpp `GGML_CUDA_FORCE_MMQ`) |
| `off` | the exact tiled GEMM for these weight types, for workloads that do not accept the 8-bit activation rounding (llama.cpp `GGML_CUDA_FORCE_CUBLAS`) |

Any other value fails at model load with an error naming `neuralfree.gemm.mmq`, the value and the
allowed values. The switch does not affect F16/BF16 weights.

#### Split-K for tall-skinny projections

F16/BF16 (`<T>_MatMul_MMA_KSplit`), int8 MMQ (`<T>_MatMul_MMQ_KSplit`) and cooperative-matrix
(`<T>_MatMul_CM_KSplit`) projections with K ≥ 2048 whose tile grid fills less than about two
thirds of the GPU's compute units are split along K into up to 8 slices, using llama.cpp's
Vulkan split-K heuristic. Quantized-weight slices always cover whole weight blocks. The slice
partials are then summed in fp32 by `LLM_SplitK_Reduce`. Results differ from the unsplit kernel
only by fp32 addition order: the MMQ slices read the same staged 8-bit codes as the unsplit
kernel, and the CM slices the same fp32 activations.

Split-K needs the device's compute-unit count. A device that does not report one — a Vulkan driver without
`VK_NV_shader_sm_builtins` or `VK_AMD_shader_core_properties2`, such as Intel integrated GPUs — runs every
projection unsplit, as llama.cpp's `ggml_vk_guess_split_k` does, instead of failing. The same applies to any
other heuristic that would otherwise size work by the count.

#### Measured speedups

On an RTX 5080 Laptop GPU (sm_120), against the tiled GEMM at N = 64–512 tokens:

- int8 MMQ (staging pass plus GEMM, 64-token tensor tile): 2.3–3.1x
- F16/BF16 MMA: 2.1–2.7x
- split-K on tall-skinny launches (8–32 tokens): 1.8–7.4x over the plain kernel

To reproduce:

    NEURALFREE_BENCH_GEMM=1 ./gradlew :test --tests 'com.eonmux.neuralfree.gguf.ops.gemm.CudaFastGemmBenchmarkTest'

### Fused MoE grouped GEMM

On a routed mixture-of-experts layer with more than one token per step, the expert products run as one expert-sorted tiled GEMM, as vLLM's fused MoE does. `LLM_MoE_Bucketize` sorts the `(token, slot)` pairs by expert. `LLM_MoE_AlignBlocks` lays the sorted rows out in whole 32-row tiles per expert (vLLM's `moe_align_block_size`). `{T}_GroupedGemm` then computes each expert's gate, up and down products in 64x32 tiles, the geometry of the dense prefill GEMM. Expert biases, activations, post-expert norms and the weighted sum are unchanged.

With the MoE expert cache on (`--moe-cache`), multi-token steps of a cached layer use the same sorted, aligned GEMM over the cache's resident experts: `{T}_GroupedGemmInd` reads each tile's bank slot from the layer's expert map. Tiles of experts the cache does not hold are skipped whole, and their pairs are computed on the host (or by the prefill prefetch) as before.

The grouped GEMM runs on tensor cores (`{T}_GroupedGemmMma`, `{T}_GroupedGemmMmaInd` for the cache) in two cases, with fp32 accumulation:
- **CUDA, compute capability 8.0 or higher** (`mma.sync`). Weights and activations enter the tensor cores as fp16, or as bfloat16 for BF16 experts so that their weights stay exact.
- **Vulkan with `VK_KHR_cooperative_matrix`.** The device needs 16x16x16 fp16 x fp16 -> fp32 subgroup matrices, `shaderFloat16`, `computeFullSubgroups` and `vulkanMemoryModel`. Operands are fp16.

F32 experts, BF16 experts on Vulkan, and all other devices use the scalar kernels. The tensor-core kernel is chosen when the device reports the capability and the kernel prepares; otherwise the layer logs the fallback once and uses the scalar kernel.

Steps of more than 2048 `(token, slot)` pairs sort on many workgroups (`LLM_MoE_BucketCount`, `LLM_MoE_BucketScan`, `LLM_MoE_BucketScatter`), and the tile layout is built on many workgroups as well. Both write exactly what the single-workgroup sort writes, so results are deterministic and do not depend on the batch split.

The same sorted, aligned GEMM serves two other per-row expert products on multi-token steps:
- the value experts of MoVA attention;
- the per-head absorption (`attn_k_b`) and expansion (`attn_v_b`) of latent-cache MLA, where each attention head acts as an expert.

It covers every expert weight type of the id-matvec kernels (F32, F16, BF16 and every GPU quant type) and every expert shard layout. A layer uses it when all of these hold:
- the backend has the kernels;
- the device runs the 256-work-item tiles and the tile grid fits its workgroup counts;
- every reduction length is a multiple of the weight type's block size.

Otherwise batched steps keep `{T}_BatchMatMulId` (`{T}_BatchMatMulIdInd` on the expert-cache path), or the per-head `{T}_MatVecId` for MLA. Single-token decode steps keep the `{T}_MatVecId*` kernels. There is nothing to configure: the choice is made per layer at bind time, and the results match the per-pair path within float reduction order (and fp16/bf16 operand rounding on tensor cores).

### Decode-step command replay

Single-token decode on the CUDA, Vulkan, HIP and OpenCL backends can record its transformer trunk once and
replay it as one device submission instead of launching every kernel individually. This matters most for
small and MoE models, where launch overhead dominates decode. It is on by default; turn it off with:

```
-Dneuralfree.gpu.decodeReplay=false
```

Any value other than `true` or `false` is rejected at model open with an error naming the property.
`-Dneuralfree.gpu.graphWarmupSteps=N` (default 2) sets how many consecutive decode steps of the same shape run
directly before the trunk is recorded, as Avalon's warmup does; 0 records the first eligible step, and a negative or
non-numeric value is rejected naming the property. `-Dneuralfree.gpu.chainReplay` (default on for CUDA, off
elsewhere) extends the capture into speculative decoding: a rollback that only rewinds host bookkeeping keeps the
drafter's captured decode, and the multi-row verify step records its own trunk. Output is the same token for token
either way. On OpenCL the capture is a `cl_command_buffer_khr`, which
requires the device's `cl_khr_command_buffer` extension; a device without it reports no capture support and
decode runs directly there, so both properties have no effect on that device. HIP records the stream with
`hipStreamBeginCapture`.

How it behaves:
- After load, a prefill, a rollback/reset, a context shift or a hidden-tap change, the warmup count restarts;
  once that many decode steps of one shape have run directly, the next records the trunk's dispatches and submits
  that capture once.
- Following decode steps replay the capture as a single submission. On CUDA and HIP, per-token scalar
  arguments (key counts, query positions) and the grid are refreshed through `cuGraphExecKernelNodeSetParams`
  and `hipGraphExecKernelNodeSetParams`; on Vulkan the pre-recorded command buffer is resubmitted; on OpenCL
  the recorded command buffer is resubmitted with `clEnqueueCommandBufferKHR`, and per-token scalar arguments
  are replaced in place with `clUpdateMutableCommandsKHR` when the device offers
  `cl_khr_command_buffer_mutable_dispatch`, otherwise the step is captured again.
- A replay does not run the step's dispatches again. The capture records each kernel's arguments and how every
  per-step value (key counts, positions, split layout) derives from the token position; a step at the next position
  rewrites only the nodes whose values changed and submits the graph once. CUDA instantiates the capture twice and
  patches the idle copy for the next step while the current one runs, so a replayed step costs 0.03-0.08 ms of host
  time; HIP and OpenCL patch the one instance before submitting.
- A step the capture cannot express is never replayed: a position that does not follow the last, a changed command
  pattern, or a flash-decode split count other than the captured one (the 2-to-3 split boundary at 65 keys)
  records the trunk again at the new layout.
- Measured on the RTX 5080 Laptop (CUDA, best of three 48-token runs): the pplx-embed 0.6B F16 model decodes at
  266 tokens/s replayed against 223 directly, a small synthetic model at 2723-3041 against 1632, and a Qwen3 4B
  Q8_0 model, whose decode is bound by its kernels, at 9.6 against 9.3 ms per step. The first 3-12 steps after each
  capture cost about 2 ms more while the new graph warms up. With decode replay on, `chainReplay` wins on every
  measured pair (32.9 against 31.7 tokens/s for the 4B target with a 0.6B drafter). NVIDIA's OpenCL offers no
  `cl_khr_command_buffer`, so decode there always runs directly.
- A step whose trunk cannot be captured or no longer matches runs directly: a hidden tap that wants
  layer outputs, a host transfer inside the trunk, an MoE layer whose experts come from the host expert cache, a different dispatch sequence, Vulkan push
  constants or OpenCL scalar arguments changed by per-step scalars where mutable dispatch is unavailable. After two mismatched replays or three aborted
  captures, replay stops trying until the step shape changes again.
- With `chainReplay` on (CUDA), a multi-row decode step of 2 to 16 rows (a speculative verify, a small decode
  batch) is captured once per row count and replayed the same way: after two verified body steps the step replays
  from its start position alone, every per-step value of its attention (key count, split layout, the combine grids)
  derived from that position and checked against the capture. A row count other than the captured one records the
  step again, and a position that does not follow the last runs one body step without dropping the capture. A paged
  decode step whose every row is its own sequence (continuous batching's decode shape, as `forwardBatch` over a paged
  memory runs it) keeps one capture across consecutive steps with the same row count, adapter slots and mask rule,
  and replays it from the step's largest key count alone: the paged decode attention's split layout, grid and merge
  counts derive from that count, and the page tables, per-sequence key counts and positions come from device buffers
  the step uploads before the trunk. A step that switches the attention between its direct and split forms records
  the trunk again, and so does any call of another kind. On the RTX 5080 Laptop this takes Qwen3-Embedding-0.6B Q8_0
  from 3.4 to 2.4 ms per step at 1 row and from 5.0 to 3.9 ms at 16 rows (`nftool batched-bench -c 4096 -npp 1
  -ntg 128`). A tree-masked verify step captures its trunk once per shape and replays it through the body. A paged step with a multi-row segment, a recurrent
  target's tree round, and context-parallel and dual-chunk steps always run directly. Output is the same token for
  token as running the step directly.
- Where the layer runs [tensor-core flash attention](#tensor-core-flash-attention), a one-row decode step runs
  that kernel as a one-row prefill instead of the scalar split-K decode kernel, which measured slower even at one row
  (48 against 65 us per layer at 256 keys, 32 query and 8 KV heads of 128). On CUDA and HIP, a layer whose KV heads
  serve at most two query heads switches to the decode kernel's warp-shuffle form past 128 live keys, and the replay
  captures again once where the route changes. Other layers keep the scalar decode kernel.
- Prefill steps always run directly. A `forwardBatch` whose rows are one-per-sequence decodes over the paged memory
  uses the step replay above; any batched call with a multi-row segment runs directly.

#### Graph capture and kernel batching (Java API)

The decode replay is built on a capture API that Java callers can use directly. A `ComputeQueue` can record a sequence of device commands once and replay it with a single submission: a CUDA or HIP graph (kernels, device copies and fills; replays refresh kernel parameters in place), a Vulkan command buffer (kernel dispatches only; push constants are frozen, so per-step values must flow through buffers), or an OpenCL command buffer (kernel dispatches, device copies and fills; per-step scalar arguments are updated in place where `cl_khr_command_buffer_mutable_dispatch` is available, otherwise a changed argument needs a new capture). The CUDA backend can also submit a set of independent small kernels as one submission; OpenCL reports kernel batching as unsupported.

Capture one step of a fixed shape and replay it:

```java
GpuGraphCache cache = new GpuGraphCache(8);          // graphs kept at once
GpuGraph graph = cache.getOrCapture("decode tokens=1", GpuGraphCache.captured(queue, () -> {
    ops.enqueue(KERNEL_A, globalA, localA, argsA);  // kernels, device copies and fills
    ops.enqueue(KERNEL_B, globalB, localB, argsB);
}));
graph.launch(queue);                                 // one submission per step
```

The step's body runs only on the first call for a shape key; later steps of that shape replay the stored graph. A replay reads the current contents of the captured buffers, so keep every value that changes between replays in device memory and write it before the launch. Captured argument values, buffer addresses and work sizes are frozen; a different shape or different arguments need a different key/graph. Every buffer a graph references must stay open until the graph is closed. LRU eviction closes the least recently used graph when the cache exceeds its capacity.

While a queue is capturing, host writes and reads, markers, synchronizations and waits on events of other queues are rejected with `IllegalStateException`; device copies, fills and kernel dispatches are recorded. Ending a failed step through `GpuGraphCache.captured` aborts the capture, so the queue is never left capturing. A CUDA or HIP graph may be launched on any open queue of the backend, and a launch may wait on events of other queues of the same backend; a Vulkan graph or an OpenCL command buffer runs only on the queue that captured it.

Batch independent small kernels into one submission:

```java
try (KernelTaskBatch batch = queue.beginKernelBatch()) {
    batch.add(kernelA, new long[]{n}, new long[]{local}, new Object[]{bufA, n});
    batch.add(kernelB, new long[]{n}, new long[]{local}, new Object[]{bufB, n});
    batch.launch(queue);                             // one launch for the whole batch
}
```

Tasks run in add order, exactly like the same launches on the in-order queue. A batch keeps its own copy of each task's arguments, so it does not disturb the kernel handles, and every buffer a task references must stay open until the batch is closed. When a launch has the same task count, kernels and work sizes as the previous one, the backend updates only the node arguments and reuses the instantiated graph (`graphReuses()`); otherwise it rebuilds. `supportsGraphCapture()`, `supportsKernelBatching()` and `isCapturing()` let a caller feature-detect and guard the paths.

### GPU event ownership

Events returned by `ComputeQueue.enqueueKernel`, `enqueueReadBuffer`, `enqueueWriteBuffer`,
`enqueueMarker` and `enqueueKernelPreBoundWithEvent` are caller-owned: waiting for the command or
finishing the queue does not release them. Close every event, preferably with try-with-resources:

```java
try (GpuEvent read = queue.enqueueReadBuffer(buffer, 0, bytes, host)) {
    read.awaitCompletion();
}
```

Closing an event releases its driver handle without cancelling the command or waiting for it, so an
event may be closed as soon as nothing else needs to wait on it. An unreleased OpenCL event keeps
its command queue, and therefore its context and device memory, alive after the backend is closed.
Prefer `enqueueKernelNoEvent` when a dispatch's completion is not needed.

As a safety net, `OclComputeBackend.close()` releases every event its queues still have open and
logs one WARN naming the device with the count; `openEvents()` reports the live count. The CUDA
event pool likewise destroys outstanding `CUevent`s and warns at close. Backend shutdown always
releases events before queues, kernels and the context.

### Fused decode kernels

A layer's decode path launches fused kernels in place of kernel pairs that read and write the same rows, as llama.cpp's backends fuse adjacent graph nodes:

| Kernel | Replaces | Where it is selected |
|---|---|---|
| `LLM_RmsNormRope` | `LLM_RowNorm` (RMS) then `LLM_RoPE_Ext` on the same rows | `GpuAttentionMixer` for a layer whose Q/K norms are per-head RMS norms applied before RoPE |
| `LLM_GdnNormGate` | `LLM_RowNorm` (RMS) then `LLM_GLU` (SWIGLU) | `GpuGdnMixer` output of every gated DeltaNet layer |
| `LLM_HcPostNorm` | `LLM_HC_Post` (expand) then the leading `LLM_RowNorm` of the consuming site | `GpuHcExtra.afterMixer`; both norm scopes (per stream row with gamma, per token without) are handled |
| `LLM_NormRopeStore` | The per-head RMS norm, `LLM_RoPE_Ext` and the K row store into the KV cache (llama.cpp's `rms_norm + mul + rope + view + set_rows`) | `GpuAttentionMixer` for an F32 or F16 device KV cache whose row is the K projection's width, with no K rotation or per-row scales |
| `LLM_RopeStore` | `LLM_RoPE_Ext` then the K row store, for layers without a QK norm (llama.cpp's `rope + view + set_rows`) | `GpuAttentionMixer`; the V row keeps its own store |
| `{T}_MatVecFusedGateUp[Gelu][_Q8]` | The gate and up matvecs and `LLM_GLU` | `GpuDenseFfn` for single-token decode of a plain parallel-GLU feed-forward |
| `LLM_ResidualMixNorm` | `LLM_ResidualMix` then `LLM_RowNorm` of the sum | The layer executor, where a layer's feed-forward residual feeds the next layer's attention norm of the same width: no post-norm, residual scale or extra stream operations on either layer, and an ungrouped row norm |

One persistent multi-task kernel (`LLM_MultiTask`) also runs up to four small tasks of a layer in one launch: a gated DeltaNet layer's four input projections on a single-token step, its L2 norm with the gate preparation, and an attention layer's Q norm and rotation, K norm and rotation, and V norm. Its launches go through the same prepared-dispatch path as every other kernel, so decode-step replay captures them.

**Selecting the fused path.** The selection is read when a model's GPU context binds its `GpuOps`, from the system
property `neuralfree.gpu.fused`:

- `auto` (default): a fused kernel runs when the bound backend provides it, otherwise the unfused pair runs.
- `on`: the fused kernels are required; binding a model on a backend that lacks one fails naming the kernel and the
  backend, instead of silently running the unfused path.
- `off`: only the unfused kernels run, for an exact comparison against the fused path.

Example: `-Dneuralfree.gpu.fused=off` (same convention as `neuralfree.matvec.int8`).

**What runs fused.** Any layer the engine already dispatches through `GpuOps`; no model file, tensor or flag
changes. A gated DeltaNet layer fuses its output norm+gate; its L2 norm and gate preparation, and its four input
projections on a single-token step, become one persistent launch each where that kernel exists. An attention layer
with per-head RMS Q/K norms before RoPE fuses the two norms+rotations with the V norm into one launch. A
hyper-connection layer fuses the expansion with the consuming site's leading norm. Layers whose shapes the fused
kernels do not cover (norms after RoPE, LayerNorm Q/K norms, non-F32 projections, LoRA deltas) keep the unfused
kernels.

**Coverage.** Every fused kernel, the multi-task kernel included, builds for OpenCL, CUDA and HIP from the shared kernel sources, with a GLSL twin for Vulkan.

**Arithmetic.** Every fused kernel and every multi-task body computes what the pair it replaces computes, with the
same operand layout and the same reduction; the only difference is float summation order where the launch geometry
differs. `FusedKernelsGpuTest` compares them value by value on a device.

**Effect.** A Qwen3.5 Gated DeltaNet layer goes from 11 kernel launches to 6, a Qwen3 attention layer from 12 to 8, and a dense feed-forward from 4 to 2. On the RTX 5080 Laptop with decode-step replay, Qwen3.5-9B Q4_K_S decodes 13% faster fused on CUDA and 52% faster on OpenCL; Qwen3-4B, which has no DeltaNet layers, is unchanged within the measurement spread.

### Integer-dot (Q8_1) decode matvec

GPU decode projections can run a Q8_1-activation integer-dot matvec instead of the F32-activation
matvec. The projection input is quantized to Q8_1 blocks (`LLM_QuantizeQ8_1`). One quantized row serves every projection of
the layer step that reads it in the attention, dense feed-forward, Gated DeltaNet, KDA, Lightning, MLA, DSA-MLA,
DeepSeek-V4, MSA, QSA, Step5 and Zamba2 layers; elsewhere each projection quantizes its own. The weight codes are combined with the activation codes by packed 4x8-bit
integer dot products (Vulkan `GL_EXT_integer_dot_product`, CUDA `__dp4a`, portable integer
arithmetic on OpenCL), as llama.cpp's MMVQ path does; where it measured faster, one workgroup computes two output rows
from each read of the quantized row (`{T}_MatVec2_Q8`, `{T}_MatVec2FusedGateUp_Q8`): every device kind except Q4_K, Q5_K,
Q8_0 and Q8_K on CUDA, Q8_0 and Q8_K on the NVIDIA OpenCL device, Q8_K on the NVIDIA Vulkan device and Q8_0, Q5_K
and Q6_K on the Intel OpenCL iGPU, which keep the single-row form. Decode output then differs from the F32 path
only by the 8-bit activation quantization error.

Supported weight types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ3_K, IQ3_KS, IQ1_S and IQ1_M, in the plain and the
fused gate/up projections. A step of 2 to 8 tokens (a short speculative verify, a small decode batch) runs the
exact-token-count form `{T}_MatVecN<n>_Q8` of a plain projection, which compiles the step's own token count
into the kernel: it reads each weight row once for all of the step's tokens and gives each token the single-row
kernel's result bit for bit, and where the two-row form measured faster it runs `{T}_MatVec2N<n>_Q8`, one
workgroup per two weight rows over the same activation rows with the identical results. Q8_0 instantiates the
counted forms to 16 tokens, and its measured entries keep the GEMM lanes out of the whole range on CUDA sm_80+
and on the NVIDIA OpenCL kind. A weight with a [fast GEMM lane](#fast-gemm-for-prefill-and-batched-decode)
leaves the multi-column form where the GEMM's flat cost beats the matvec's per-row integer dots, measured per
kind and type (`SmallBatchCrossover`): on CUDA sm_80+ the counted forms hold the rows through 8 tokens on Q4_K
and Q6_K and on the 4B-class Q4_0, Q5_K and IQ4_XS projections, whose 0.6B-class projections keep the matvec
through 16 (kernel medians at 2 tokens, 4096x1024: Q4_0 10.3us paired against the MMQ small tile's 45.8, Q4_K
8.8 against 32.6, IQ4_XS 10.6 against 57.2; 19456x2560: Q4_0 36.5 against 234.8, Q4_K 52.0 against 215.7,
IQ4_XS 67.0 against 343.0), and on the NVIDIA OpenCL kind the counted forms hold 2..8 on the measured types
(19456x2560 at 2 tokens: Q4_K 114.5us paired against the MMQ lane's 275.6) while the lane keeps 9..16. A step
above a type's counted range runs the runtime-count `{T}_MatVecN_Q8` where its entry serves it (Q8_0 to 16
tokens); a weight without a measured counted win keeps the previous dispatch. Longer steps and prefill run the
GEMMs. Q8_0 weights meet the Q8_1 row as ggml's `vec_dot_q8_0_q8_1` does.

Selection: `-Dneuralfree.matvec.int8=auto|on|off`

- `auto` (default): use the lane where the device reports integer dot product support and the kernel
  is available, unless the weight type measured slower with the lane than with the F32 matvec on that
  kind of device (`Int8MatvecTuning`, keyed like the crossover above; measured on the RTX 5080 Laptop
  over CUDA, OpenCL and Vulkan and on the Intel Arrow Lake iGPU over Vulkan and OpenCL, over 0.6B to 8B
  decoder projection shapes):

  | Device kind | Never the lane | The lane from 4M weight elements (rows times columns) up |
  |---|---|---|
  | CUDA, NVIDIA discrete, sm_80+ | Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL | MXFP4, NVFP4, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_XS |
  | OpenCL, NVIDIA discrete | MXFP4, NVFP4, IQ4_NL, IQ4_XS | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K |
  | Vulkan, NVIDIA discrete | | every supported type |
  | Vulkan, Intel integrated | Q4_0 | |
  | OpenCL, Intel integrated | Q4_1, Q5_0, Q5_1, Q8_0, Q3_K, MXFP4, IQ4_NL, IQ4_XS | Q4_0, Q2_K, Q4_K, Q5_K, Q6_K, Q8_K |

  The NVIDIA OpenCL device reports no packed dot product, so its row only applies under `on`; the
  multi-column forms below serve Q8_0, Q4_0, Q4_K, Q5_K, Q6_K and IQ4_XS there under `auto` through the portable
  integer path, where they measured faster than the float matvec. A device
  whose kind has no row, and a type its row does not list, keep the feature decision. The second column's
  types lose the lane's fixed quantizer cost on the 1024-column projections of a 0.6B-class model (on the
  iGPU about 100-150 us per launch, against 14-20 us on the RTX 5080) and gain from 4096 columns up. Q8_0
  on the OpenCL iGPU won up to 1.11x on single projections, but the whole 4B decoder step measured
  0.75-0.85x with the lane, so it never takes it there. The choice is per projection, so a mixed model
  keeps the lane for its other types and sizes. Elsewhere the exact F32 matvec runs.
- `on`: use the lane whenever the kernels can run, for every supported type, including the portable
  OpenCL integer path on devices without a packed dot instruction.
- `off`: never use the lane.

### Certified exact-argmax LM head

Greedy decoding and greedy speculative verification only need the argmax of each logits row, not the row. The
certified head decides each token without reading the whole head at its stored precision: an INT8 copy of the head
rows (per-row absmax scale) screens every vocabulary row, and a per-row error bound times the hidden state's norm (a
Cauchy-Schwarz bound on the quantization error plus a worst-case summation term) keeps every row whose logit can
still reach the maximum. Those candidates are rescored with the head's own exact arithmetic, the first index winning
ties. A row with more candidates than the cap, or none, takes a full exact scan, decided on the device without a host
round trip, so decode-step replay is unaffected. The token always equals the argmax of the exact logits row.

`neuralfree.argmax.head` picks the mode:

- `auto` (the default) uses it where it measured faster end to end: on OpenCL, for Q8_0 heads of at least 300
  million weights (a 4B-class head with a 150k vocabulary decodes about 1.5-2% faster; smaller heads lose to the
  extra dispatches).
- `on` uses it wherever the head is eligible.
- `off` keeps the exact readout.

`int8` is accepted as `on`, and `neuralfree.draft.argmax.head` as an alias of the property.
`neuralfree.argmax.kmax=N` (default 128, alias `neuralfree.draft.argmax.kmax`) is the candidate cap; the kernels
clamp it to 256, and a row over the cap takes the full scan. On real 150k-vocabulary heads the widest row measured 32
candidates and no row fell back.

- Eligible heads: on the CPU, F16 and BF16; on OpenCL, CUDA, HIP, Vulkan and Metal, F16, BF16 and Q8_0 (rows a
  multiple of 32 long). A Q8_0 head on CUDA or Vulkan decodes through the integer-dot matvec and keeps the exact
  readout. The head must be plain: no output bias, logit scale vector or multiplier, softcap, sigmoid cap, masked
  range, vocabulary split, input transform, Hadamard rotation, per-tensor scale or LoRA deltas. A call whose
  projection runs a fast, tiled or integer-dot matvec keeps the exact readout for that call.
- Sampling, logprobs and every other consumer of the logits keep the exact readout; an unknown mode is refused by
  name.

No llama.cpp or vLLM option corresponds to this lane.

### Lossless 12-bit weights (W12)

`--w12` (`neuralfree.weights.w12`, default off; `--no-w12` turns it off) stores the target's BF16-representable
projection weights on the GPU as W12 rows, about 12 bits per weight, and decodes them in registers inside the matvec.
Every weight decodes to exactly its BF16 value, so the logits and tokens are those of the BF16 weights. The output
head joins with `--w12-head` (`neuralfree.weights.w12.head`, which needs `--w12`). `--w12-f16 <refuse|round>`
(`neuralfree.weights.w12.f16`, default `refuse`) decides what happens to an F16 tensor holding a value BF16 cannot
represent: `refuse` keeps W12 lossless and refuses the load, `round` rounds every such weight to the nearest BF16
value (it loses its 3 low mantissa bits, so the output can differ from the F16 model's) and the load logs once how
many weights moved and the largest relative change. The server takes the same switches, and `neuralfree.draft.w12` / `neuralfree.draft.w12.head` are accepted as aliases (the new names win). GGUF
files never store W12: it is the form the weights take on the device, and a file tensor of type W12 is refused by name
on read and on write.

- **The row record.** Each row picks the 16-binade exponent window that covers most of its weights (ties to the
  highest) and stores per weight the sign, a 4-bit exponent within the window and the 7 mantissa bits; the row's
  scale `2^(E-127)` multiplies its dot product once. Weights outside the window, zeros, subnormals, infinities and
  NaNs sit in the planes as the placeholder `2^-15` and are corrected exactly from a per-row side table
  (`acc += x * w - x * 2^-15`, both products exact in fp32), with no limit on their count. A row is a Q region (one
  byte per weight: sign, exponent, top three mantissa bits) and a B region (the low four mantissa bits), so the
  matvec streams 1.5 bytes per weight. Real model projections measure 12.04-12.10 bits per weight with 0.01% escapes.
- **Eligible:** tensors consumed by a matmul, stored as BF16, or as F16 whose values are all BF16-representable, whose
  width is a whole number of 64-weight blocks. The upload re-encodes each tensor, checks the round trip bit for bit
  and frees the BF16 or F16 copy. Refused by name: an F16 tensor holding a value BF16 cannot represent (naming the
  tensor, the first such element and `neuralfree.weights.w12.f16`) unless `--w12-f16 round`, `--w12-head` without
  `--w12`, and `--w12` on a model with no eligible tensor. W12 runs only on the GPU, so a refusal fails the load
  instead of falling back to the CPU engine.
- **Steps:** decode and verify steps up to 64 rows run the W12 row kernels on OpenCL, CUDA, HIP, Vulkan and Metal; a
  wider prefill decodes the layer into a BF16 scratch once and runs the BF16 GEMM.
- **The 8-bit view.** The Q regions alone are an 8-bit view of the weights (every in-band weight read at the midpoint
  of its dropped bits, the escapes above the window corrected exactly): the row kernels can stream only them, at one
  byte per weight. [Two-level verification](#two-level-verification-neuralfreedrafttlv) runs its quantized round
  over this view.

Measured on the RTX 5080 Laptop, the W12 decode matvec takes 0.51x (a 32768 x 1024 head) to 0.88x the BF16 time on
CUDA at four of the five measured shapes, 1.11x at 9728 x 2560; on OpenCL it is 2.9x faster at the head shape and
1.2-1.3x slower at the small, launch-bound attention shapes. A 96-token prefill through the BF16 scratch runs within
about 14% of plain BF16, and Vulkan on the Intel integrated GPU is slower. End to end the picture differs by model:
a small synthetic llama decodes at 2055 tokens/s with W12 against 1753 without it on CUDA, and the pplx-embed 0.6B
F16 model, whose weights need `--w12-f16 round`, at 203.7 against 194.7 on CUDA and 207.9 against 226.6 on OpenCL. No llama.cpp or vLLM option corresponds to
W12.

### GPU weight placement

A GPU load places the weights from the free device memory at load time, as llama.cpp's `--fit` does. Without an explicit split it tries, in order:

1. **Every weight resident**, with the largest context that then fits. An automatic context (`neuralfree.ctx=0`) must reach 4096 tokens, or the training context when that is shorter; otherwise the next placement is tried.
2. **A hybrid split** (`neuralfree.n.gpu.layers=auto`, the default): the largest suffix of trunk layers whose weights and KV fit stays on the device and the earlier layers run on the host ([below](#hybrid-cpu-and-gpu-execution)). Models that cannot split are skipped to the next placement.
3. **Streaming.** The expert tensors of MoE models (`*_exps.weight`, `*_chexps.weight`) stream from the model file first while the dense tensors stay on the device, and the experts of the first layers come back onto the device while they fit. Then the smallest prefix of whole trunk layers streams, the rest staying resident. Last, everything streams except the model-level tensors, with the largest context that fits beside the streaming slots.

The chosen placement is logged at load, for example:

```
qwen3next: the dense tensors of all 4 trunk weight layers stay on the device (0.6 MiB), the expert tensors of 3
layer(s) stream from the host (0.3 MiB through 0.2 MiB of slots), the experts of the first 1 layer(s) stay on the
device, context 256
```

A context set with `neuralfree.ctx` is honored as given: the fit moves weights to the host until it fits, and refuses the load naming `neuralfree.ctx` when even the streamed placement cannot hold it. Streamed weights are read from the model file (mmap) through the streaming slots; `neuralfree.vram.depth` sets how many slots are used and `neuralfree.ram.cache.mb` the host RAM cache. Models evaluated in a single step (encoder-decoder sources, the WavTokenizer decoder) keep every weight resident.

#### Hybrid CPU and GPU execution

A split keeps the last layers on the device and runs the earlier ones on the CPU engine, as llama.cpp's `-ngl` does. Each side holds the KV cache of its own layers, and the residual stream crosses between them once per forward step, so host layers compute at RAM bandwidth instead of streaming their weights over the bus every token.

- `neuralfree.n.gpu.layers` (`--gpu-layers`, `-ngl`): `auto`, `all`, or a count. The count includes the output head as in llama.cpp, so `-ngl 8` puts 7 trunk layers on the device. A count past the model's layers offloads the model whole, as llama.cpp's `-ngl 99` does: it is clamped to every trunk layer plus the output head, like `all`, and does not run the `auto` memory search. `-ngl 0` is refused; open the model with `neuralfree.backend=cpu` instead. The context fit charges only the offloaded layers' KV, so a split fits a larger context than a full offload.
- `neuralfree.override.tensor` (`--override-tensor`, `-ot`): comma-separated `pattern=target` entries, the target one of `cpu`, `gpu`, `opencl`, `cuda`, `vulkan`, `hip`, `metal` or an accelerator API (`sycl`, `xpu`, `cann`, `hexagon`, `openvino`, `webgpu`, `musa`, `zdnn`, `zendnn`). Every target but `cpu` keeps the matched tensors on the device side, on the devices the load selected: the name states the placement, not a particular device. Patterns are regular expressions matched against stored tensor names (`blk.5.attn_k.weight`), and the first match wins. A layer computes as one block on one side, so a `cpu` match keeps the tensor's whole layer on the host, together with every earlier layer: the host layers are always a prefix. A `cpu` match on a routed expert tensor (`ffn_(up|down|gate|gate_up)_exps`) is the exception: the experts stay in RAM beside their device layer, as `--cpu-moe` keeps them. A `gpu` match shields its tensor from later rules. llama.cpp pattern strings work verbatim, for example `-ot "blk\.[0-3]\.ffn_down_exps=CPU"`.
- `neuralfree.cpu.moe` (`--cpu-moe`) and `neuralfree.n.cpu.moe` (`--n-cpu-moe <n>`): keep the expert weights of every MoE layer, or of the first `n`, on the host. The layer stays on the device; at its FFN the selected experts' input rows cross to the host, the CPU MoE block computes them, and the result crosses back. Two small activation transfers replace the per-token streaming of the expert weights.
- `neuralfree.n-cpu-ffn` (`--n-cpu-ffn`, `-ncffn <n>`): keep the dense FFN weights of the first `n` layers in RAM (dense models).
- `neuralfree.kv.offload` (`--kv-offload`, `-kvo`; `--no-kv-offload`, `-nkvo`): with `false`, every device layer keeps its KV cache in host RAM, as llama.cpp's `-nkvo` does. The layer's projections, norms, RoPE, FFN and output projection run on the device; only the Q/K/V rows of a step cross to the host, the CPU attention core writes the K/V rows into the host cache and attends, and the attention rows cross back for the output projection. The device memory of the load allocates no KV cache and the context fit charges none, so a full offload stays a full offload at a long context. It combines with a host prefix (`-ngl <count>`) and with `--cpu-moe`. It is refused at load for paged or multi-sequence memory, the multi-GPU layer split, NextN/MTP self-speculation, encoder-decoder and audio-decoder models, and non-softmax KV caches (MLA, sparse and parallel-attention layers); context shift over a host cache is refused by name.
- `neuralfree.moe.cache` (Avalon's `--moe-cache`) and `neuralfree.moe.cache.mb` (`--moe-cache-mib <n>`): host the routed experts as `--cpu-moe` does and keep the hot ones in a VRAM cache ([MoE expert cache](#moe-expert-cache)).
- `neuralfree.fit` (`--fit`): keep the experts in RAM when the dense weights alone then fit the usable device memory, leaving the rest to the context. When even the dense weights do not fit, the load keeps the usual placement and logs why.

A layer's experts host only as a whole set (gate, up or the fused gate-up, and down); a rule that hosts part of a set is refused at load, naming the layer. The hosted experts of latent or chunk expert sets, hash routing, biased-logit selection and weight-before-FFN experts compute through the host FFN block without the cache, and an expert cache on a backend that cannot run the expert type is refused by name, as is a LoRA adapter on a hosted expert tensor. The split pays off from two tokens per step; a single-token step computes every hosted expert on the host.

Model-level tensors (embedding, output head, norms) and the NextN/MTP blocks always stay on the device. A split is refused by name, before anything is placed, for encoder-decoder and audio-decoder models, models without memory, looped or hierarchical layer schedules, paged or multi-sequence memory (`neuralfree.parallel > 1`, `neuralfree.kv.paged`), and LoRA adapters. A hybrid split runs on one device; [multi-GPU layer splits](#multi-gpu-layer-split) are a separate placement.

##### MoE expert cache

On a mixture-of-experts model the routed expert stacks can stay in host RAM while the layer runs on the GPU; a
device cache holds the hot experts in VRAM, the GPU computes the cached experts through the per-layer indirection
kernels, and the CPU computes the misses at the same time.

Properties (llama.cpp / Avalon spellings in parentheses):

- `neuralfree.moe.cache` (`--moe-cache`, Avalon's spelling): `off` (default) | `on` | `auto` | a budget in MiB.
  `on`, `auto` or a MiB value hosts the routed expert stacks in RAM — the same placement as
  `neuralfree.cpu.moe` (`--cpu-moe`) — and serves their routed experts from the cache. `auto` and `on` use the
  VRAM left after loading as the budget.
- `neuralfree.moe.cache.mb` (llama.cpp `--moe-cache-mib`): the cache budget in MiB. It sets the same budget as
  `neuralfree.moe.cache` and wins when both give one; on its own it enables the cache with that budget
  (an explicit budget is clamped to the VRAM left after loading).
- `neuralfree.moe.cache.ranking` (`--moe-cache-ranking`): the `moe-cache-ranking.bin` file (the Avalon/Strata
  format) the cache preloads at open, hottest first; unset reads `moe-cache-ranking.bin` next to the model when one
  exists, and `off` disables the preload.
- `neuralfree.moe.cache.ranking.save`: a ranking file the run writes when the model closes. Every routed row counts
  into a per-layer, per-expert heat profile while the model runs, on the CPU and on the GPU (where the router's
  selections are read back asynchronously); at close the counts merge with that file's heats for the same model shape
  and the file is replaced atomically, hottest first. The profile is kept with or without the cache, so pointing the
  save at the ranking beside the model warms the next run's preload. A ranking that cannot be read or written is
  logged and skipped; it never fails the load or the close.
- `neuralfree.moe.cache.tail.skip` (`--moe-cache-tail-skip <n|off>`; Avalon's `GGML_CUDA_MOE_TAIL_SKIP`): an
  approximate routing rule. An expert the cache does not hold, which every token of the step routes only at 0-based
  top-k position `n` or later (`n` from 1 to 9), is not computed on the CPU or copied to the GPU; its output rows stay
  zero and the other experts keep their router weights, without renormalization. `off` or `0` (the default) keeps
  every miss; Avalon uses rank 7. The rule changes the output, needs `neuralfree.moe.cache`, applies to the main
  model's layers only (never a NextN/MTP layer), and never prunes a prefill step that the expert prefetch serves.

Behavior:

- Hosting follows the one placement path: `neuralfree.moe.cache` / `neuralfree.moe.cache.mb` host the routed
  experts exactly like `neuralfree.cpu.moe`, and combine with `neuralfree.n.cpu.moe`, `neuralfree.n-cpu-moe`,
  `neuralfree.override.tensor` (`-ot`) and `neuralfree.fit` (`--fit`). A layer with only part of its expert set
  in RAM is refused by name.
- Residency is a segmented LRU (probationary/protected, llama.cpp `moe_cache_slru`) over banks shared by the
  layers of one expert layout. A long prefill (over 32 tokens) only accumulates a routing histogram — it never
  evicts the experts a later generation needs — and the first small step after it admits the hottest experts.
  A preloaded ranking fills the slots before the first step. On steps of two or more tokens the split dispatcher
  (Avalon's fitted cost model) copies a missing expert from the bus into the cache when that ends the MoE window
  sooner than computing its rows on the CPU; one-token steps never split.
- The cache logs its open line (`layers`, `groups`, `slots`, MiB of banks, ranking preload) and, on close, its
  stats: hits, misses, hit rate and MiB filled. `neuralfree.moe.cache=off` keeps every weight placed as without
  the cache.
- Refused by name: the cache together with a hybrid `-ngl` split whose trunk layers compute on the host;
  LoRA deltas on the routed experts of a hosted layer (the adapter attachment reports the unbound product);
  hosted latent/chunked/hash-routed/biased-selection/weight-before-FFN layers and architectures that do not
  route on the normalized FFN input (under a `-ngl` split these layers keep their CPU MoE block through the
  host-expert FFN decorator instead); a routed expert tensor whose stored type the backend cannot run on the
  device (clear `neuralfree.moe.cache` / `neuralfree.moe.cache.mb` to compute every pair on the host).

**Prefill expert prefetch.** With `neuralfree.moe.cache` on, prefill-sized steps no longer compute the cache's misses on the CPU. A step is prefill-sized when it has more than one token and at least 2 routed pairs per expert.

- The cache lends whole layer stacks of its VRAM slots ("zones") for the duration of the prefill.
- Experts the cache already holds are read in place.
- All other experts are uploaded into the zones. The next layer's experts are copied while the current layer computes.
- The first decode step gives the zones back, so generation keeps the cache's full capacity.

| Property | Default | Meaning |
|---|---|---|
| `neuralfree.moe.cache.prefetch` | `2` | Layer stacks (zones) a prefill borrows from the cache: one computing, one being copied. `0` or `off` disables the prefetch, and the misses are computed on the host. A cache group whose slots hold fewer than one layer stack (`n_expert` slots) keeps the host misses. |
| `neuralfree.moe.cache.hosted.rows` | `0` | Steps routing at most this many rows (tokens × experts used) keep the host misses even when prefill-sized (Avalon `GGML_MOE_HOSTED_MAX_ROWS`). Raise it when the CPU computes misses faster than PCIe copies whole experts. |
| `neuralfree.moe.cache.tail.skip` | `off` | Rank `n` (1–9) of the approximate tail-skip rule: an uncached expert that every token routes only at top-k position `n` or later is dropped (zero output, no renormalization). `off` or `0` keeps every miss. Requires the cache. |
| `neuralfree.moe.cache.pool` | `false` | Pool the expert caches of a layer split's GPUs (`--moe-cache-pool`); needs the cache on and `neuralfree.split.mode=layer`, and is refused beside per-layer LoRA ([below](#pooled-expert-cache-over-several-gpus)) |

Example:

```
-Dneuralfree.moe.cache=on -Dneuralfree.moe.cache.prefetch=2
```

On close, the log line `MoE expert prefetch closed: ...` reports:
- prefetched steps and lends
- experts copied ahead and at step time
- experts read in place
- MiB copied

Invalid values (negative, or not a number) are refused with an error that names the property.

A large MoE model on an 8 GB device, with the dense parts and the KV on the GPU and every expert in RAM:

```
-Dneuralfree.backend=opencl -Dneuralfree.n.gpu.layers=auto -Dneuralfree.cpu.moe=true
```

##### Pooled expert cache over several GPUs

On a layer split (`neuralfree.split.mode=layer` over two or more GPUs) the expert cache can pool the GPUs' VRAM:
the routed hosted MoE layers are divided between the GPUs' caches, each GPU taking a run of layers whose expert
bytes match its share of the VRAM left after its weights and KV caches, and each layer's hosted MoE block runs on
the GPU whose cache serves it (Avalon's `GGML_MOE_CACHE_LAYER_DEVICE=1`, llama.cpp's per-backend
`llama_moe_cache`).

- `neuralfree.moe.cache.pool` (`--moe-cache-pool`): `false` (the default) keeps each GPU's cache for the layers
  of its own span; `true` pools the caches as described. It needs `neuralfree.moe.cache` (or
  `neuralfree.moe.cache.mb`) and a layer split: it is refused by name with the cache off, outside
  `neuralfree.split.mode=layer`, and beside `neuralfree.lora` / `neuralfree.lora.enable`, whose adapters bind to
  the layer's own stage.
- The layers are taken in layer order; a layer goes to the GPU whose free-VRAM share its middle falls in.
  `neuralfree.moe.cache.mb`, the prefill prefetch (`neuralfree.moe.cache.prefetch`), the hosted tail-skip rank
  (`neuralfree.moe.cache.tail.skip`) and the ranking file (`neuralfree.moe.cache.ranking`) apply to the cache of
  each GPU.
- The layer's block weights (the router, the expert scales and biases, the post-expert norm and the shared expert)
  upload to the GPU that caches the layer, while its attention, KV cache and the norms the executor applies around
  the block stay on the stage that runs its span. The normalized FFN input crosses to the owner GPU and the FFN
  output crosses back, as the residual stream does between spans. A layer whose block the hosted path cannot serve,
  or whose residual stream is wider than the FFN input, keeps its block on its own stage; the load log names it.
- The load log line `neuralfree.moe.cache.pool=true assigns the hosted MoE cache over N GPU(s): layers 0..3 on
  opencl1 '...'` lists the assignment. Fewer than two selected GPUs load on one device with the ordinary cache.

Example: two GPUs, three layers on the first GPU's span and one on the second, with the caches pooled:

    --device 0,1 --tensor-split 4,1 --moe-cache on --moe-cache-pool

#### Host-mapped weights (UVA zero-copy)

`neuralfree.cpu.offload.mb` keeps up to that many MiB of device-resident trunk weights in page-locked host memory
that the device reads in place, instead of VRAM: CUDA mapped host memory (`cuMemHostAlloc` +
`cuMemHostGetDevicePointer`) or Vulkan host-visible memory (device-local when the heap offers it). Nothing is
copied per forward pass; the interconnect carries each read. This is vLLM's UVA offload backend
(`cpu_offload_gb`) and needs a CUDA or Vulkan device: on another backend, beside a layer split (a
`split.mode=layer` load or an explicit hybrid `-ngl` load) and beside a row/tensor split over two or more devices,
the setting is refused by name.

```properties
# Keep 2 GiB of trunk weights readable from host memory.
neuralfree.cpu.offload.mb=2048
# Only the routed expert tensors: the attention and dense feed-forward of every layer stay in VRAM.
neuralfree.cpu.offload.params=ffn_gate_exps,ffn_up_exps,ffn_down_exps
```

`neuralfree.cpu.offload.params` takes comma-separated parameter name segments (vLLM's `cpu_offload_params`). A
segment matches whole dotted parts of the stored tensor name: `ffn_gate_exps` matches
`blk.3.ffn_gate_exps.weight`, `expert` does not, and `ffn_gate_exps` does not match `ffn_gate_exps_scale`.
Empty (the default) offloads tensors of any kind in weight-layer and file order until the next tensor would
exceed the budget. A segment that names no device tensor is refused by the property name. The offloaded tensors
are read once at load into page-locked host RAM, which they occupy instead of VRAM.

#### Selective streaming

`neuralfree.stream.params` restricts which parameters of a streamed layer go through the streaming slots; the
other tensors of those layers stay resident. It is vLLM's prefetch `offload_params`, and needs no other setting:

```properties
# Stream only the routed expert stacks of the layers the placement streams.
neuralfree.stream.params=ffn_gate_exps,ffn_up_exps,ffn_down_exps
```

A segment follows the same whole-dotted-part rule as above; empty (the default) streams every tensor of a
streamed layer. A segment that names no device tensor is refused by name. Setting a selection also suppresses the
automatic whole-layer suffix offload: an automatic `-ngl` load then streams the selected tensors instead of
moving whole layers to the host, which streams nothing. A selection beside an explicit hybrid offload is refused
by name.

#### Free VRAM sources

The GPU placement sizes its VRAM budget from the most exact free-memory figure the platform offers:

- **AMD on Linux** — the budget reads the amdgpu driver counters
  `/sys/bus/pci/devices/<pci-bus-id>/mem_info_vram_total` and `mem_info_vram_used`. They count every process, and are
  preferred over `hipMemGetInfo`, which on ROCm can report noticeably less free VRAM than the card actually has.
- **AMD on Windows** — the budget reads WDDM's adapter segment statistics through `D3DKMTQueryStatistics`: the local
  (non-aperture) segments' commit limit minus their resident bytes. This counts the desktop, browsers and every other
  process that `hipMemGetInfo` and an OpenCL device size ignore. When the statistics cannot be read, 3 GiB of the
  runtime's figure is kept aside for those processes and the result is treated as an estimate.
- **Vulkan with `VK_EXT_memory_budget`** — the driver's own budget is used, since it already accounts for other
  processes.
- **NVIDIA** — `cuMemGetInfo` or the Vulkan budget, with `nvidia-smi` as the fallback for other APIs.
- **Integrated GPUs** — the runtime's figure is used: their VRAM counters report only the BIOS carve-out, not the
  shared system memory the driver budgets.

No configuration is required. A device whose VRAM is completely full reports zero free bytes, not an estimate of the whole card.

### Multi-GPU layer split

A model can run over several GPUs, each holding a contiguous span of layers and their KV caches, as llama.cpp's `--split-mode layer` and vLLM's pipeline parallelism do. The embedding and the output head run on the main GPU, and the activations cross from one GPU to the next through the host. The GPUs are driven through the backend APIs, so a split may mix vendors.

| Flag | Property | Meaning | Default |
|---|---|---|---|
| `--split-mode`, `-sm <m>` | `neuralfree.split.mode` | `none`, `layer`, `row` ([below](#tensor-parallel-row-split---split-mode-row)), `tensor` ([below](#tensor-parallelism---split-mode-tensor)) or `context` ([below](#context-parallelism---split-mode-context)) | `none`; `layer` when `--device` lists two or more GPUs |
| `--device`, `-dev <d1,d2,..>` | `neuralfree.gpu.devices` | The GPUs of the split in pipeline order, comma-separated: a 0-based index in the backend's device list, as `--list-devices` prints it, or a device-name substring (`GeForce`). One index picks that GPU for a one-device load, in either split mode; one name substring picks it across the APIs (`neuralfree.device`); `none` runs the CPU engine | every GPU the backend reaches, discrete before integrated, one view per GPU |
| `--tensor-split`, `-ts <n0,n1,..>` | `neuralfree.tensor.split` | Positive shares, one per device, separated by `,` or `/` (`3,1` puts three quarters of the GPU layers on the first GPU). When omitted, each GPU's usable free memory sizes its share, as llama.cpp's default, and the automatic fit rewrites the shares from the layers each GPU holds | each GPU's usable free memory |
| `--rpc <[tls://]host:port[,..]>` | `neuralfree.rpc` | Remote GPUs served by `nftool rpc-server`, appended after the local devices of a layer split; the key comes from `NEURALFREE_RPC_KEY` ([Remote compute servers](#remote-compute-servers-layer-split-offload)) | none |
| `--main-gpu`, `-mg <i>` | `neuralfree.main.gpu` | Index into the device list of the GPU holding the embeddings and the output head. With `--split-mode none`, the one GPU the model runs on: entry `i` of `--device`, or GPU `i` of the devices a split would select | `0` |
| | `neuralfree.gpu.vram.limit` | MiB of device memory a GPU load may allocate on each of its devices (weights, caches, scratch, streaming slots, adapters, the MoE expert cache and the drafter's memory together). A load whose weights then do not fit streams them, as on a smaller GPU | `0` (no limit) |

The flags override the properties. The same flags launch the server (`neuralfree serve` or the server jar), and vLLM's `-pp N`/`--pipeline-parallel-size N` is its spelling of an N-GPU layer split. Combinations that cannot run are refused, naming the flag:
- `--tensor-split` without a layer or row split, and with `--split-mode tensor`;
- a share count that differs from the device count;
- a `--main-gpu` beyond the device list;
- a share of 0 (drop that GPU from `--device` instead);
- a layer, row or tensor split on the CPU backend (`--cpu`, `--device none`).

Each GPU fits its own span: its weights, its layers' KV caches and its scratch. The context is the largest every GPU fits; `neuralfree.ctx`, the KV quantization and `neuralfree.ubatch` apply per GPU. A GPU whose span does not fit beside the context streams the first layers of its span from the host, as a one-GPU load does ([models larger than RAM](#models-larger-than-ram)). The host cache of streamed weights is shared between the streaming GPUs in proportion to what each streams. With fewer than two devices selected, the model loads on one; a single listed index loads on that device.

**Automatic host fit.** With `--gpu-layers auto` (the default), the split fits as a one-GPU load does, and it moves layers to the host before it streams any:
- The split fills the GPUs back to front at the chosen context: the last GPU takes the largest run of trailing trunk layers it holds resident, then the GPU before it, and so on. The layers left in front of every GPU run on the host. The per-GPU counts become the shares, so the fit rewrites `--tensor-split`; an explicit `--tensor-split` is kept instead.
- Before a GPU gives up a whole layer, it moves the routed experts of the leading layers of its span to RAM, stage by stage as each GPU fits its own span, as llama.cpp's fit places the overflowing MoE tensors in the CPU buffer. This needs no `--fit` under `-ngl auto`.
- The load streams only when no layout keeps every span resident: with `--gpu-layers all`, or when a GPU cannot hold even one layer at the context. This is the same rule as a one-GPU `-ngl auto` load, which also offloads layers to the host before it streams.
- The load log and the model summary list the host layers and the routed experts kept in RAM.
- `--gpu-layers all` never moves layers to the host.
- A setting that already keeps layers or experts on the host (below), the MoE expert cache, LoRA adapters and `neuralfree.parallel` > 1 turn the automatic host fit off.

**Host placement.** The settings that keep tensors on the host apply per GPU, as on one GPU:
- `--gpu-layers <count>`, an `--override-tensor` match on a layer's tensor, and `--n-cpu-ffn` keep the first layers on the host (up to the highest such layer). The host runs them, and the remaining layers are spread over the GPUs. Every GPU needs at least one layer: raise `--gpu-layers` otherwise.
- `--cpu-moe`, `--n-cpu-moe` and an `--override-tensor` match on expert tensors keep those routed experts in RAM, beside their GPU layer.
- `--fit` decides per GPU for an explicit `--gpu-layers <count>` or `all` layout: a GPU whose dense weights fit its VRAM keeps its routed experts in RAM. The automatic `-ngl auto` fit already moves the experts of the layers it overflows.
- The MoE expert cache (`neuralfree.moe.cache`) runs on every GPU over the routed layers of its own span. A `neuralfree.moe.cache.mb` budget applies to each GPU.
- A split with host placement or the MoE cache keeps every GPU's weights resident.
- A KV-sharing layer and the layer owning its cache must sit on the same side of the host boundary.

**Adapters.** LoRA adapters (`--lora`, `--lora-scaled`, `neuralfree.lora`) apply across a split: each GPU applies the adapter pairs of its own layers, and the main GPU those of the token table and the output head. Control vectors (`--control-vector`, `--control-vector-scaled`) apply the same way: each GPU applies the vectors of its own layers, and the host those of the host layers. Their uploads are held back from each GPU's budget.

**Speculative decoding.**
- A draft model (`neuralfree.draft`) runs on the main GPU, whose budget holds its memory.
- The model's own NextN / MTP blocks (`neuralfree.mtp`) run on the GPU of the last layer, which also holds their draft caches. When that GPU is not the main one, it also holds a copy of the token table, the output head and the output norm, which the MTP head reads.
- Blocks left off at load (`neuralfree.mtp` unset) are placed on that GPU when the drafter attaches, if its free VRAM holds them.

`neuralfree.diag` reports every stage with its device (`[DIAG] step=1 device=opencl1 'Intel(R) Graphics' layer-12 ...`). The decode profiler (`--benchmark`) lists the stage times of each device with the layers it ran (`on opencl0 'NVIDIA GeForce RTX 5080' (layers 0-13):`). The model summary printed at load lists each stage's device, layers, streaming and MoE expert cache, and the host layers.

A split is refused, naming the setting, for:
- encoder-decoder models;
- looped layer schedules;
- switched-LoRA architectures;
- the `gemma4-assistant` drafter, which reads the target's KV caches on one device;
- the MoE expert cache together with host layers.

LoRA adapters and `neuralfree.parallel` > 1 are refused together with host layers, as on one GPU. A refused split does not fall back to the CPU engine.

Three quarters of the layers on the discrete GPU, the rest on the integrated one:

    --device GeForce,Intel --tensor-split 3,1

or with properties:

    -Dneuralfree.split.mode=layer -Dneuralfree.gpu.devices=GeForce,Intel -Dneuralfree.tensor.split=3,1

On a 48-layer model, the first four layers on the host (`--gpu-layers` counts the output head) and every routed expert in RAM, the other layers over two GPUs:

    --device 0,1 --gpu-layers 45 --cpu-moe

A 32k context over two GPUs whose memory cannot hold its KV caches: the split fills the GPUs back to front at that context, moving the overflowing routed experts to RAM and the first layers and their caches to the host on its own:

    --device 0,1 -c 32768

The model on the second GPU only:

    --split-mode none --device 0,1 --main-gpu 1

#### Remote compute servers (layer-split offload)

A layer split can run a contiguous span of trunk layers on a GPU on another machine. On that machine, start a server:

```
export NEURALFREE_RPC_KEY="$(cat /run/secrets/nf-rpc-key)"
nftool rpc-server --host 0.0.0.0 --port 50052 --backend opencl --device 0
```

`rpc-server` (also reachable as `neuralfree rpc-server`) serves the GPUs of the machine over TCP. Its options are `--host`/`-H` (default `127.0.0.1`), `--port`/`-p` (default `50052`), `--backend`/`-b` (`auto | opencl | cuda | vulkan | hip | metal`; the CPU engine is refused), `--device`/`-d` (comma-separated device indices or name substrings, default every device of the backend), `--cache`/`-c [dir]`, `--tls-cert`/`--tls-key` and `--key`. The shared key comes from `NEURALFREE_RPC_KEY` or `--key`; the server refuses to start without one. Prefer the environment variable: a `--key` value, like an inline `@key`, is visible in `ps` and shell history. Keep the default loopback bind unless a remote client is intended, and then keep the key secret.

`--cache`/`-c [dir]` keeps received weights on disk keyed by their content hash, so a reconnecting client (after a crash, a restart, or from a second process) skips the upload: the client probes by hash per 1 MiB chunk and transfers only the misses. Without a directory the cache lives under `$XDG_CACHE_HOME` (or `~/.cache`) in `neuralfree/rpc`.

On the client, add the server to the layer split:

```
export NEURALFREE_RPC_KEY="$(cat /run/secrets/nf-rpc-key)"
neuralfree serve -m model.gguf --device 0 --rpc 10.0.0.5:50052
```

or with properties:

```
-Dneuralfree.split.mode=layer \
-Dneuralfree.rpc=10.0.0.5:50052
```

`neuralfree.rpc` (llama.cpp `--rpc`) takes comma-separated `[tls://]host:port[@key]` entries (an IPv6 host in brackets, `tls://[::1]:50052`); an entry without an inline key reads `NEURALFREE_RPC_KEY`. Each device a server serves joins the client's device list as `rpc<i>:<d>`. Remote devices are appended after the local devices in configuration order and take a layer span by the same `--tensor-split` shares. With no usable local GPU, list at least two servers to form a remote-only cluster. The property requires `neuralfree.split.mode=layer`; the CPU engine (`--device none`), other split modes, and a total of fewer than two devices are refused by name.

Connections are authenticated with a challenge/response HMAC-SHA256 handshake (the key is never sent), and each connection can only touch the buffers and kernels it created; its buffers and caller-owned kernel clones are released when it disconnects. Without TLS the payloads are not encrypted, so run a plain server on a trusted network only.

For TLS, prefix the client's endpoint with `tls://` and give the server a PEM certificate chain and an unencrypted PKCS#8 key: `--tls-cert`/`--tls-key`, or `neuralfree.rpc.tls.cert`/`neuralfree.rpc.tls.key` (`NEURALFREE_RPC_TLS_CERT`/`NEURALFREE_RPC_TLS_KEY`). A PKCS#1 or SEC1 key is refused with the `openssl pkcs8 -topk8 -nocrypt` command that converts it. The client verifies the server against a PKCS12 or JKS trust store named by `neuralfree.rpc.tls.truststore` (its password in `NEURALFREE_RPC_TLS_TRUSTSTORE_PASSWORD` or `neuralfree.rpc.tls.truststore.password`), or the JVM's default store.

`neuralfree.gpu.devices` may name remote devices beside local ones: `0,rpc0` splits over local device 0 and every device of server 0, `rpc1` alone leaves server 0 out. A selector that names nothing is refused with the full device list; without selectors the split takes every local device and every server. Commands that need no answer are batched into one round trip per decode step (`neuralfree.rpc.pipeline`, default `true`); `false` sends each command in its own round trip.

### Tensor-parallel row split (`--split-mode row`)

On a host with more than one GPU, NeuralFree can put every eligible dense feed-forward on all GPUs at once instead of
giving each GPU a contiguous span of layers. This is vLLM's MLP tensor parallelism (`tensor_parallel_size` = the
number of devices) and llama.cpp's `--split-mode row`.

    --device 0,1 --split-mode row

or with properties:

    -Dneuralfree.split.mode=row -Dneuralfree.gpu.devices=0,1

- `--split-mode row` / `-sm row` / `neuralfree.split.mode=row` selects the mode. To split the attention heads as
  well, use [`--split-mode tensor`](#tensor-parallelism---split-mode-tensor).
- The FFN's hidden width is divided over the ranks: evenly by default, or in llama.cpp's `--tensor-split`
  proportions (one share per device of `--device`). Each boundary between ranks is rounded down to a whole weight
  block and the last rank takes the remainder. Rank 0 is the model's own device (the GPU of
  `--main-gpu` / `neuralfree.main.gpu`, or the default device); the other GPUs are the peers. Every rank holds its
  rows of `ffn_gate` / `ffn_up` (and both halves of a fused gate-and-up), the matching slice of the hidden-width
  biases and mpt activation scales, and the matching column slice of `ffn_down`. Each rank computes its partial
  output and a device-to-device all-reduce sums the partials; the output-width down bias is added once, on rank 0.
- Attention, the norms, the embedding, the output head and the NextN/MTP blocks stay whole on the model's device, as
  do trunk layers whose FFN cannot be split (a routed-MoE FFN, `ffn_sub_norm`, gemma3n activation sparsity, a
  hidden width that does not divide evenly, a PRISM/Hadamard transform on the split projections, or a non-trunk
  stack). The load log lists which layers were kept whole and why.
- `--device 0,1` / `neuralfree.gpu.devices` selects the GPUs (indices in enumeration order or name substrings); the
  model's device is never duplicated as a peer. With no list, every device the backend enumerates participates,
  discrete before integrated. A host that offers only one device runs the model on that one device, exactly like
  `--split-mode none` (`tensor_parallel_size=1`).
- The mode refuses, naming the setting: `--n-cpu-ffn` (`neuralfree.n-cpu-ffn`), a `-ot` rule that keeps a split FFN
  tensor in host RAM, a LoRA adapter that targets a split projection (its whole weight never exists), a load whose
  automatic placement leaves trunk layers in RAM, and any load the placement would stream through the device. The
  split ranks hold their shards resident.
- `--main-gpu` still picks the GPU that holds the embedding and the output head (rank 0); it must index the device
  list.

This matches vLLM's MLP tensor parallelism. llama.cpp's `--split-mode row` also splits the attention and output projections; here they stay whole on the model's device, and `--split-mode tensor` splits the attention heads. The collective has been tested on one physical GPU (several queues of one device and host staging); a run over two physical GPUs with device peer copies has not been measured.

### Tensor parallelism (`--split-mode tensor`)

`--split-mode tensor` (`-sm tensor`, `neuralfree.split.mode=tensor`) is vLLM's tensor parallelism over the GPUs of
`--device` (`tensor_parallel_size` = the number of devices):

    --device 0,1 --split-mode tensor

The server takes the same split as `-sm tensor`/`--split-mode tensor`, and vLLM's `-tp N`/`--tensor-parallel-size N` sets it over N devices; `-ts` and `-mg` apply there too.

Every trunk layer splits its attention and its feed-forward, and the vocabulary splits with them:

- **Attention is column-parallel.** Each rank holds its share of the query and KV heads: their Q/K/V rows (or fused
  QKV segments), per-head norms and vectors, and the matching column block of the output projection. Each rank keeps
  its own KV cache for its heads. The ranks' attention outputs are summed by an all-reduce; the output bias is added
  once, on rank 0.
- **The dense feed-forward** splits as in [row mode](#tensor-parallel-row-split---split-mode-row), evenly.
- **Routed experts** split two ways. By default each rank holds an intermediate-width slice of every expert (vLLM's
  fused MoE), at whole device-block widths, and the rank partials are summed by the all-reduce. With
  `neuralfree.expert.parallel=true` (vLLM's `--enable-expert-parallel`) each rank instead holds
  `n_expert / ranks` contiguous whole experts at the full width; every rank still runs the replicated router, computes
  only its own experts' pairs, zeroes the others and the all-reduce sums them. A shared expert always splits like a
  dense FFN.
- **The vocabulary is parallel** (vLLM's `VocabParallelEmbedding` / `ParallelLMHead`): each rank holds a contiguous
  slice of the token table and of the LM head, the lookup all-reduces the rank rows and the rank logits slices are
  gathered into the full row on the model's device.
- **Paged multi-sequence serving works.** `neuralfree.parallel > 1` or `neuralfree.kv.paged=true` gives every rank
  its own paged KV cache; each step's page tables, cell rows, `kv_pos` mirror and page copies are installed on every
  rank, so `forwardBatch`, the batch engine and prefix-cache sharing run unchanged (a rank whose page geometry
  differs from the model device's is refused at load).
- **Context shift works on every rank.** The contiguous shift moves each rank's rows and re-rotates its keys; the
  per-sequence shift applies the model device's planned row moves and per-row RoPE deltas to every rank cache.
- The heads, neurons and (with `neuralfree.expert.parallel=true`) experts divide evenly, so `--tensor-split` is
  refused; use `--split-mode row` for proportional feed-forward shards. Head counts must divide by the rank count, a
  GQA/MQA model needs at least as many KV heads as ranks, and expert parallelism needs no expert offload
  (`--cpu-moe`, `--n-cpu-moe`, `-ot` on expert tensors, `neuralfree.moe.cache`).
- The load checks that each GPU can hold its own shards and caches, and sessions save and restore every rank's cache.
- **Refused by name:** MLA and sparse MLA attention, mixers other than attention that keep a cache, shared KV,
  value experts or embeddings, low-rank Q, M-RoPE, transforms built for the full width, head counts that do not
  divide, encoder-decoder models, loaded NextN/MTP blocks, grovemoe chunk experts, zero experts, per-expert biases
  and a post-expert norm (their per-rank partials are not the whole output), and a paged layout the rank caches
  cannot address. The refusals of row mode apply as well.

A host with one GPU runs the model on that device, as `--split-mode none` does. The equivalent of vLLM's
`--enable-expert-parallel` is the system property `neuralfree.expert.parallel=true` (it requires
`neuralfree.split.mode=tensor`).

### Expert-parallel load balancing (EPLB)

With `--split-mode tensor` and `neuralfree.expert.parallel=true` (vLLM's `--enable-expert-parallel`), each rank
holds whole experts at the full intermediate width; by default a rank's experts are one contiguous block.
`neuralfree.eplb=on` (vLLM's `--enable-eplb`) balances that division by the experts' routed-row load, adds
redundant replicas, and can re-balance while the model runs. The router, the per-expert scales and the shared
expert are unchanged: every rank still runs the replicated router, computes only its own experts' pairs, zeroes
the others and the all-reduce sums the partials, exactly as the contiguous placement does — greedy outputs are
identical.

- `neuralfree.eplb.profile=<file>`: the expert-load profile read at load, a version 2 `moe-cache-ranking.bin` (the
  expert cache's ranking format): routed rows per (layer, expert). Layers the file does not count keep their
  contiguous blocks.
- `neuralfree.eplb.save=<file>`: records the routed rows counted while this model runs and writes them at close,
  merged with the file's current heats of the same expert count. Pointing both keys at one file alternates
  recording runs and balanced runs.
- `neuralfree.eplb.policy=default|hierarchical`: the placement policy. `default` is vLLM's weighted packing
  (heaviest expert into the lightest rank with a free slot); `hierarchical` packs expert groups into nodes first
  (`neuralfree.eplb.num-nodes`, multi-node inference) and then spreads the node's slots over the ranks. With
  `neuralfree.eplb.num-groups=0` (the default) each layer's router groups are used; a count that the experts do
  not divide, or `num-nodes` other than 1 with the default policy, is refused by name.
- `neuralfree.eplb.num-redundant-experts=<n>` (vLLM's `num_redundant_experts`): adds n redundant expert slots
  across the ranks. The physical slot count must divide over the ranks. Every slot uploads as its own bank row and
  a routed pair is computed by exactly one replica: the device maps the pair's logical expert to one of its
  physical replicas with the reference's token-index hash (`eplb_map_and_record`), so replicas even out the hot
  experts without changing the outputs.
- With runtime rebalancing on, `neuralfree.eplb=on` needs no profile or save file: the placement starts from the
  contiguous blocks, with any redundant replicas spread from zero load, and the first boundary re-balances it.

Runtime rebalancing (`neuralfree.eplb.rebalance=on`, vLLM's `EplbState.step`): every forward pass counts one step,
the routed rows of the last `neuralfree.eplb.window-size` steps (default 8) before each boundary are collected,
and every `neuralfree.eplb.step-interval` steps (default 1000) the layers are re-balanced from that window's live
load. Experts that keep their rank keep their slot; the changed layouts migrate in place across the ranks' expert
banks (weights, scales and replica maps) between steps. The per-rank balance is logged at every boundary as the
reference's figures, e.g. `avg_tokens=101.00, max_tokens=101, balancedness=1.0000`. Like the rest of the
expert-parallel path this needs resident experts (no `--cpu-moe`, `--n-cpu-moe`, `-ot` on expert tensors or
`neuralfree.moe.cache`), a trunk-only MoE, no zero experts, no per-expert biases and no post-expert norm.

The CLI and the server take vLLM's flags: `--enable-expert-parallel` (`-ep`), `--enable-eplb` / `--no-eplb`,
`--eplb-profile <path|off>`, `--eplb-save <path|off>`, `--eplb-policy <policy>` and `--eplb-config <json>`. As in
vLLM, `--enable-eplb` turns on runtime rebalancing too; the static, profile-only placement is
`-Dneuralfree.eplb=on` with a profile and without `neuralfree.eplb.rebalance`. `--no-eplb` also clears the
redundant experts and the rebalancing. `--eplb-config` is vLLM's `EPLBConfig` object:

| Key | Maps to |
|---|---|
| `policy` | `neuralfree.eplb.policy` |
| `num_redundant_experts` | `neuralfree.eplb.num-redundant-experts` |
| `window_size` | `neuralfree.eplb.window-size` |
| `step_interval` | `neuralfree.eplb.step-interval` |
| `log_balancedness`, `log_balancedness_interval` | accepted; the balance is logged at every boundary |
| `use_async: false`, `enable_migration_batching: false`, `communicator: null` | accepted; the experts migrate between steps inside the process |

Every other key or value, including asynchronous migration, is refused by name, and a JSON null counts as unset.
The expert-group and node counts of the hierarchical policy are `-Dneuralfree.eplb.num-groups` and
`-Dneuralfree.eplb.num-nodes`.

Refused by name: `neuralfree.eplb=on` without expert parallelism, or with none of a profile, a save path and
rebalancing; a profile naming an expert its layer does not have; `neuralfree.eplb.num-nodes` beside the default
policy; a step interval or window size below 1; rebalancing or redundant experts beside `neuralfree.eplb=off`;
and `neuralfree.eplb.save` beside `neuralfree.moe.cache.ranking.save` (one forward path records one profile).

### Context parallelism (`--split-mode context`)

NeuralFree can divide a long sequence's context between several GPUs instead of dividing the attention heads, as
vLLM's decode context parallelism does (`decode_context_parallel_size`).

Set the property:

```
neuralfree.split.mode=context
```

or pass the flag, naming two or more GPUs:

```
-sm context -dev 0,1
```

(`-dev` alone selects the layer split, so name `-sm context` explicitly.)

Every rank keeps every query and KV head and a whole copy of the layer's attention weights. The context positions
are divided into contiguous blocks, block boundaries at `rank * n_ctx / ranks`: rank `r` stores only its block's KV
rows, attends over `[rank * n_ctx / ranks, (rank + 1) * n_ctx / ranks)`, and the other positions are hidden from it.
Each rank produces the normalized attention over its block and a log-sum-exp; the ranks' results are then merged:

```
M        = max_r lse_r
out[d]   = sum_r e^(lse_r - M) * out_r[d] / sum_r e^(lse_r - M)
```

so the layer output equals the single-device attention over the whole sequence. The merge is vLLM's
`cp_lse_ag_out_ar` (`vllm/v1/attention/ops/dcp.py`). A step before a rank's block starts produces an empty
partition (`-inf` log-sum-exp) that the merge masks.

Requirements and refusals, each named at load:

- Two or more GPUs (`neuralfree.gpu.devices` / `--device`), the same requirement as the tensor split. A machine
  with one usable GPU runs the normal single-device load, as vLLM does with size 1.
- A context of at least one position per rank.
- The context shift (server slot shift, contiguous or per sequence) is refused: a shift that moves positions across
  a block or page owner cannot move the stored rows between rank caches.
- Encoder-decoder models and NextN / MTP blocks are refused; they need caches the split does not cover.
- `--tensor-split` / `neuralfree.tensor.split` is refused; the positions divide evenly.
- `neuralfree.backend=cpu` is refused.
- A layer whose attention cannot produce one partial softmax per context block keeps its whole-device mixer and
  cache on the main device: attention sinks, dual and dual-chunk attention, MLA / MSA / DSV4 and the other
  non-standard attention graphs, shared-KV readers, and projections with a PRISM input transform. The load logs
  each such layer as "attention kept whole on the main device". A model with no eligible attention layer is
  refused with the first disqualifier in the message.

**Paged multi-sequence memory.** With `neuralfree.kv.paged` (and `neuralfree.parallel` above 1) the split divides
pages instead of contiguous blocks, as vLLM's `dcp_kv_cache_interleave_size` does with the interleave at one page: a
page belongs to rank `(first stored position / page size) % ranks`. Each step hands every rank its own page tables,
slot counts and position rows over the pages it owns, and a row landing on another rank's page is stored only there
while its query still attends over every rank's keys; the log-sum-exp merge above combines the ranks. Chunked
prefill, batched decode, prefix reuse, forks with copy-on-write and preemption all run under the split. Each rank's
pool holds `ceil(pages / ranks) + parallel` pages. Refused by name under a paged split: media and embedding segments,
KV-chunk reuse, page imports at shifted positions, and session save and load.

Memory: every rank holds the whole attention weights of the split layers and its own block's KV cache only — about
`1/ranks` of the context rows per rank — and rank 0 holds the rest of the model. A peer stops after its attention
partition: its output projection is never computed.

**Prefill context parallelism.** `-Dneuralfree.prefill_context_parallel_size=N` (vLLM's
`prefill_context_parallel_size`), with N equal to the context split's rank count, divides each prefill step's
tokens between the ranks as vLLM's DualChunkSwap does: the step is cut into `2N` chunks and rank `r` takes chunks
`r` and `2N - 1 - r`, which balances the causal attention work. Each rank runs the norms, projections, RoPE (at the
true positions) and FFN of its own rows only; the queries are gathered to every rank, each rank attends them over its
own context block, the log-sum-exp merge combines the blocks, and the merged rows return to their ranks. The step is
restored to its original row order before the output head, and decode steps use the decode split above. Decode-step
replay is off while the prefill split is on.

Over paged memory, the batch engine's case, each request's segment of a batched prefill step is cut into its own
chunk pairs, every rank packs its rows over its own translated page tables, and each rank attends the gathered
queries over the pages it owns. A decode row that shares the step with prefills is replicated onto every rank and
keeps the decode split.

Refused by name: a size other than the rank count, media and embedding segments (several rows at one base
position), a sequence whose next position skips its page's next slots, layers whose attention stays whole on the main
device, recurrent, MLA and other non-standard mixers, MoE feed-forward layers, per-layer and n-gram embeddings and
the other embedding extras, switched LoRA, hidden-state taps during the prefill, and attention layers whose cache or
stages the exchange does not carry: a cache with per-row scales or vendor planes (the TurboQuant types among them),
static sink tokens, Inkling stages, and a layer without a KV cache or Q/K/V projections. A quantized cache with the
Hadamard rotation runs. A refused load releases the ranks it had set up. The gain needs ranks on
separate devices: two ranks sharing one GPU run one after the other and only add the exchanges.

**vLLM's launch flags.** `-dcp N` / `--decode-context-parallel-size N` is `-sm context` over N devices (the first N
device indices, or the N devices `--device` lists) on the CLI and the server; a size of 1 changes nothing.
`--dcp-kv-cache-interleave-size N` names the paged split's page interleave and accepts only N equal to the page
size. `-pcp N` / `--prefill-context-parallel-size N` sets `neuralfree.prefill_context_parallel_size=N` on the CLI
and the server. `-dcp` beside `-tp` or `-pp` above 1 (vLLM
nests its decode context ranks inside the tensor ranks; this engine runs one split mode), a `-pcp` other than the
context split's rank count, and an interleave other than the page size are refused by name.

### Models larger than RAM

A GGUF whose tensor data exceeds the host's physical RAM runs with its weights in the file: the OS page cache backs the `mmap` mapping, so the process never holds the model. Two caches would compete with those pages, so they size or turn themselves off, as Avalon's server does:

- The host cache of streamed weights (`neuralfree.ram.cache.mb`, `--ram-cache-mb`) is off unless set to a size. On any model the automatic size is half the maximum heap, capped at half the RAM the model leaves free, and an explicit size above the physical RAM warns, since it evicts the model's pages.
- The prompt cache (`neuralfree.prefix.reuse`) is off unless set, since every cached sequence holds pages the model needs; `true` forces it on.
- Reading without `mmap` (`neuralfree.disable.mmap=true`, or `neuralfree.tensor.read=read|direct`) warns at load: every tensor read then allocates heap beside the model instead of sharing the page cache.

The decisions compare against the host's total physical RAM; a host that does not report it keeps the heap-based defaults.

### AMD GPUs: the HIP backend

HIP (ROCm) is the AMD counterpart of the CUDA backend. Select it with `--backend hip` or `-Dneuralfree.backend=hip`. AUTO tries it last (`neuralfree.backend.order`); on an RDNA3 or CDNA GPU that also enumerates through Vulkan or OpenCL, put `hip` first in the order to prefer it.

It needs:

1. The ROCm runtime, version 5.5 or newer.
2. The JNI shim, built once with `make -C native/hip` (needs `hipcc` and a JDK). Put `libhipshim.so` on the library path, or name it with `-Dneuralfree.hip.shim.path=/absolute/path/libhipshim.so`.
3. The LLM code objects built by `./gradlew compileHipKernels` for the GPU's `gfx` target (`kernels/llm/hip`). Without them the backend starts but reports the LLM kernel library unavailable, and AUTO moves on to the next API.

The GPU's processor name (`gfxNNN`) picks the instruction paths:

- CDNA (Instinct MI100 and newer, gfx908+) and RDNA3 and newer (gfx1100+) run the F16/BF16 matrix-core GEMM, through MFMA and WMMA respectively.
- RDNA2 (gfx1030) and Vega20 (gfx906) use packed `dp4a` integer dot products for the int8 GEMM.
- The wavefront size (64 on CDNA and GCN, 32 on RDNA) tunes the reductions and flash attention.

**MMQ tiles.** The int8 MMQ GEMM (`<T>_MatMul_MMQ`, `<T>_MatMul_MMQ_KSplit` and their W4A8 twins) picks its workgroup tile from the processor family, following llama.cpp's `mmq-config` tables. The host plans the launch with the same tile, so no configuration is needed:

| processor family | gfx targets | stage depth `TK` | rows `TM` | tokens `TN` | rows/work-item `RM` | workgroup |
|---|---|---|---|---|---|---|
| RDNA3 | gfx1100-gfx1103 | 16 | 64 | 128 | 4 | 256 |
| CDNA | gfx908, gfx90a, gfx940-gfx942, gfx950, gfx1250 | 16 | 128 | 64 | 8 | 256 |
| other HIP, OpenCL, Vulkan | — | 16 | 64 | 16 | 4 | 256 |
| CUDA sm_80+ | — | 32 | 64 | 64 | 4 | 256 |

HIP tiles are compiled into the code object per `--offload-arch` target; a device whose gfx target is not among the built `hip.archs` is refused by the HIP kernel library, so the kernel and host always agree on the tile.

The OpenCL backend compiles its kernels at run time, so a tile can be forced for testing or benchmarking:

```
-Dneuralfree.mmq.tile=rdna3
```

Accepted values are `packed-dot`, `int8-tensor`, `rdna3` and `cdna`. Any other value fails with an `IllegalArgumentException` that names the property and the accepted values. The property affects only the OpenCL backend; CUDA chooses its tile from the compute capability and HIP from the build's gfx targets, so both ignore it. CUDA, OpenCL and Vulkan defaults are unchanged when the property is unset.

A layer split (`neuralfree.split.mode=layer`) may mix a HIP device with OpenCL, CUDA or Vulkan devices; one GPU seen through two APIs counts once. Free VRAM comes from the amdgpu counters on Linux and WDDM's segment statistics on Windows ([Free VRAM sources](#free-vram-sources)). `neuralfree.hip.device` picks a device by name within HIP.

The backend has been tested against a simulated device only; no AMD GPU has run it yet.

### Metal backend (Apple GPUs)

The Metal backend drives Apple GPUs natively, without MoltenVK: a JNI bridge to Apple's Metal framework plus Metal Shading Language (MSL) kernel libraries. It is available on macOS only; on every other host the API is enumerated but reports why it is unavailable, and AUTO ignores it. The LLM and training kernels are generated as MSL from the Vulkan shaders at build time ([Metal kernel libraries](#metal-kernel-libraries)), so on macOS AUTO prefers Metal for a model. The generated kernels have not yet been compiled or run on Apple hardware; pass `--backend vulkan` (MoltenVK) to keep the previous path.

Build the bridge once on the machine that will run the engine:

```
make -C native/metal
# or: ./gradlew buildMetalShim   (writes build/generated/gpu/metal-shim/libmetalshim.dylib)

export DYLD_LIBRARY_PATH="$PWD/native/metal:$DYLD_LIBRARY_PATH"   # if not installed system-wide
# or point at it explicitly:
java -Dneuralfree.metal.shim.path=/path/to/libmetalshim.dylib ...
```

Select it:

```
--backend metal                         # neuralfree
-Dneuralfree.backend=metal             # engine property
-Dneuralfree.backend.order=metal,vulkan,cuda,opencl,hip   # AUTO order (the default; on NVIDIA, CUDA goes first)
-Dneuralfree.metal.device="M3"         # first Metal device whose name contains this text
```

AUTO enumerates Metal on macOS and prefers it for the device it shares with the Vulkan view
(MoltenVK), so an Apple GPU is driven natively by default. `--list-devices` lists Metal devices with
their recommended working-set size.

Kernels are MSL files indexed like the Vulkan shaders: `shaders/<library>/metal/index.properties`
(`<name>.buffers`, `<name>.ro`, `<name>.push`, `<name>.sg`) plus `<name>.metal`. A kernel's buffers bind at MSL
buffer indices `0..n-1` and its push-constant scalars are packed into one constant buffer at `[[buffer(n)]]`. Metal
allocations are shared-storage buffers, so weights can also be host-mapped without a copy
(`-Dneuralfree.gpu.hostMappedWeights` paths).

#### Metal kernel libraries

The Metal versions of NeuralFree's LLM and training kernels are Metal Shading Language sources generated
at build time from the same Vulkan GLSL (via the SPIR-V the Vulkan build already produces) with
SPIRV-Cross, so they cannot drift from the Vulkan kernels; the buffers and push constants bind exactly as
the Metal backend launches them.

Building:

```
./gradlew build
```

`compileMetalShaders` runs as part of the normal build. It needs the Vulkan shaders to compile (`glslc` and
`spirv-dis`, as before) and, the first time only, Maven Central access for the build-only
`org.lwjgl:lwjgl-spvc` dependency and its native library; afterwards the artifacts are in the Gradle cache
and offline builds work. When SPIRV-Cross cannot be resolved the task warns and produces nothing, which
keeps the Linux build working: the Metal library then reports "Metal shaders were not built" and AUTO
ignores it. Add `-Pneuralfree.requireShaders=true` to make a missing tool fail the build instead.

What the build produces (under `build/generated/gpu/metal`, merged into the application resources as
`shaders/<library>/metal/`):

- `<Kernel>.metal` for every kernel of `kernels/<library>/inventory.txt` that the shader build provides
  (all 1,463 LLM and 85 train kernels, including the subgroup variants, which are translated from the
  `__sg` module because Metal SIMD groups are always available);
- `index.properties` with the same `.buffers`, `.ro`, `.push` and `.sg` entries the Vulkan index uses.

Argument contract of a generated kernel: storage buffer with binding `b` is `[[buffer(b)]]` for
`b` in `0..n-1`, and the push-constant scalars are packed into one constant buffer at `[[buffer(n)]]`.
Specialisation constants 0-2 carry the dispatch's local size and id 3 carries `NF_FA_SMEM_FLOATS`; the
backend supplies them per dispatch (function constants, or the `SPIRV_CROSS_CONSTANT_ID_*` macros when
SPIRV-Cross had to fold a constant into a constant expression, which recompiles the library for that
work-group size). Subgroup reductions use Metal `simd_*` operations.

Verify the generated artifacts:

```
./gradlew verifyGpuKernels
```

It now checks Metal coverage alongside CUDA, HIP and Vulkan: every portable inventory kernel must be in
the Metal index, every indexed kernel must have its `.metal` file and `kernel void` declaration, and the
Metal signature entries must match the SPIR-V index. The JUnit suite additionally parses every generated
index and MSL file against the inventory and the backend's binding contract.

The MSL is compiled by the Metal framework on the machine that runs the engine. Compiling and running it
requires macOS with an Apple GPU; the Linux development host can generate and validate the sources but not
compile them.

### Other accelerator backends (provider jars)

NeuralFree drives OpenCL, CUDA, Vulkan and HIP itself. The other accelerator APIs run through a
**backend provider**: a jar on the application class path that declares an implementation of
`com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorBackendProvider` in
`META-INF/services/com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorBackendProvider`
(vLLM's `vllm.platform_plugins` entry points play the same role; llama.cpp builds these backends into its
own tree). NeuralFree ships no provider: the registry, selection, ranking, listing and refusals below are in
place, and running a model on one of these APIs needs a provider jar with its vendor binding and a kernel library
for that API. The provider reports its devices as
`AcceleratorDevice`s — name, identity, device class, memory, compute units and clock — and opens
one `ComputeBackend` per device on demand.

Backend names, accepted by `--backend`, `-Dneuralfree.backend`, the `neuralfree.backend` property
and `-Dneuralfree.backend.order`, case-insensitively:

| API | names |
| --- | --- |
| SYCL / Level Zero | `sycl`, `level-zero`, `level_zero`, `levelzero` |
| Intel XPU | `xpu` |
| Huawei CANN | `cann` |
| Qualcomm Hexagon | `hexagon` |
| Intel OpenVINO | `openvino` |
| WebGPU | `webgpu`, `web-gpu` |
| Moore Threads MUSA | `musa` |
| IBM Z zDNN | `zdnn`, `z-dnn` |
| AMD ZenDNN | `zendnn`, `zen-dnn` |

- **Explicit mode:** `--backend sycl` in the CLI and demo surfaces (or `-Dneuralfree.backend=sycl`,
  which the server reads as well) runs on the fastest device that API's provider reported, ranked
  discrete first and then by compute units times clock. `-Dneuralfree.sycl.device=<substring>`
  (likewise `xpu`, `cann`, ...) forces a device; `-Dneuralfree.device` is the fallback filter.
- **AUTO:** installed providers join the API order after Vulkan, CUDA, OpenCL and HIP; override
  the order with `-Dneuralfree.backend.order=sycl,cuda,opencl`. AUTO still settles on one physical
  device first and recognises an accelerator's view of it by PCI id, UUID or device name.
- **No provider installed:** the engine refuses the mode by name, for example
  `SYCL backend requested but no SYCL devices available. Install the vendor runtime and a SYCL
  backend provider jar (...AcceleratorBackendProvider in META-INF/services). Cause: SYCL: no SYCL
  backend provider is installed; ...`. AUTO reports the same reason for each accelerator API.
- **Device listing:** `--list-devices` prints each provider's devices under its API label, numbered
  from 0 with the memory the provider reports, e.g. `  SYCL 0: Intel Arc B (8192 MiB)`.
- **Kernels:** a provider whose backend runs the engine's kernels also registers a
  `com.eonmux.neuralfree.acceleration.kernels.KernelLibrary` for its API in
  `KernelLibraryRegistry.global()`; without one, backend selection reports that the `llm` library
  is unavailable on that API and AUTO falls back to a built-in backend.
- **TPU:** vLLM's TPU platform runs through JAX/torch_xla, which this engine cannot drive;
  `--backend tpu` is refused with that reason instead of being silently accepted.
- A second provider for the same API, or a provider that fails to enumerate, is reported and the
  engine keeps starting; the failing API simply offers no devices.

### CPU SIMD kernels and weight repacking

On Java 17+ the CPU engine uses hand-written dot-product kernels built on the JDK Vector API (`jdk.incubator.vector`) when that module is resolved, and the scalar kernels otherwise (Java 11, no module, or `neuralfree.cpu.simd=false`). The kernels are bit-exact with the scalar ones, so the choice changes speed only. Vector kernels exist for the legacy types `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` and `Q8_0`, the K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K` and `Q6_K`, `F32`, `F16` and `BF16` weights (F32 activations), the ternary `TQ1_0` and `TQ2_0`, the 16-entry codebook types `IQ4_NL`, `IQ4_XS` and `MXFP4` (a `selectFrom` codebook lookup, llama.cpp's `vpshufb`), the newer low-bit types `Q1_0`, `Q2_0`, `NF4`, `FP4_BNB` and `I8_B128`, and the repacked `Q4_0x8`, `Q8_0x4` and `Q4_Kx8` layouts. Where an int8 product fits a short, the kernels multiply in 16-bit lanes, sixteen products per instruction. `NVFP4`, the F8/MXFP8/MXFP6 extension types, the IQ grid-codebook types and every other weight type keep their scalar kernel: their scalar path measured at least as fast, or the decode needs a lane-crossing permute or a grid lookup the portable Vector API cannot express efficiently.

The kernels are chosen by measurement, not by type name. On a 16 x 4096 matvec the Vector API kernels are faster by these factors: `Q1_0` 2.8-3.5x, `Q2_0` 2.2-3.1x, `I8_B128` 1.8-2.1x, `MXFP4` 1.20x, `NF4` 1.21x, `FP4_BNB` 1.21x. On the CPU backend with Jina Embeddings v5 Small Retrieval (596M parameters, 64-token prompt, 8 generated tokens, three repetitions) `Q1_0` moves from 16.28 to 52.59 prefill tok/s and 3.51 to 5.35 decode tok/s; `Q2_0` decode moves from 1.80 to 2.30 tok/s. On a 2560 x 2560 x 64 product `IQ4_XS` runs 4.4x and `F16` 3.7x faster than their scalar kernels. The vector forms of `NVFP4`, the F8/MXFP8/MXFP6 types and every IQ grid-codebook type measured slower and are not built.

**Tiled prefill GEMM.** A product of several tokens is a tiled GEMM: each kernel multiplies a register tile of weight rows by eight token rows, so every decoded weight block serves eight tokens and every widened activation block serves the rows of the tile, and the output tiles are spread over the CPU pool. The legacy 32-element block types use a two-row tile, the K-quant and float types one row (their super-block unpack dominates). The results are bit-exact with the per-token row loop, so tiling changes speed only. With 8 threads, Qwen3-4B Q4_0 prefill moves from 3.08 to 8.45 tok/s at 64 tokens and from 2.92 to 7.11 at 512, and MiniCPM3-4B Q4_K_M from 6.49 to 11.47 and 5.15 to 7.64; decode gains 1.1-2.1x because single-token dots use the same int16 product path. CPU decode is close to llama.cpp's on Q4_0 and Q8_0, but prefill stays 5.6-7.3x behind it at 512 tokens: the portable Vector API has no 8-bit multiply-add, so the int16 path costs about two instructions per multiply-accumulate against llama.cpp's one.

Resolve the module with the JVM option `--add-modules=jdk.incubator.vector`. On Java 17 or later every packaged launcher resolves it without the caller passing the option; on Java 11 and earlier the option is never added (the module does not exist there) and the scalar kernels run.

- `bin/neuralfree`, `bin/neuralfree.bat` and `java -jar neuralfree-<version>.jar`: for the commands that run the engine in the launcher's JVM (`cli`, `bench`, the nftool tools, `train`, `download`) the launcher re-executes itself once with the option (`JvmVectorModule`), keeping the start-up JVM's heap and `--add-opens` options. The child carries `-Dneuralfree.cpu.vector.relaunched=true`, so it never relaunches again. Commands that do not load the CPU kernels (`version`, `help`, `cache-list`, `collect-env`, `--version`, `--list-devices`, `--cache-list`) run in the original JVM.
- `neuralfree serve` and `neuralfree run-batch`: the server's child JVM is started with the option (`ServeCommand`).
- The Gradle test and run tasks (`./gradlew test`, `run`, `runLLMDemo`, `runNfTool`, the demos): their JVMs are started with the option.

`neuralfree.cpu.simd=false` skips the relaunch and keeps the scalar kernels, and a direct `java -jar server/<server runner jar>` launch (not a distribution start script) still takes `--add-modules=jdk.incubator.vector` on its command line itself.

Q4_0, Q8_0 and Q4_K matmul weights can be repacked into llama.cpp's interleaved `block_q4_0x8`, `block_q8_0x4` and `block_q4_Kx8` layouts. A repacked weight is copied once into the Java heap when its first matmul asks for it (not when the tensor is mapped), and every later product runs over the copy; the mapped original is not read again, so its file-backed pages stay reclaimable. Lookup tables (`token_embd`, position and token-type embeddings) are never copied: they are read row by row, so a copy would only duplicate them.

`neuralfree.cpu.repack` selects the types: `auto` (default) repacks only the types whose repacked kernels measured faster than the plain Vector API kernels, `on` repacks every covered type, and `off` keeps every weight in place; any other value is rejected by name. On the CPU backend no repacked kernel wins both the one-token and the batched product (`Q4_0x8` wins only the one-token product; `Q8_0x4` and `Q4_Kx8` are at parity or slower), so `auto` currently selects none. A `Q4_K` weight repacks only when its column count is a multiple of 2048, so common 2560- and 3072-column models keep `Q4_K` in place even with `on`.

Copies are never unbounded. The budget is the free heap measured when the first copy is asked for minus a reserve of 25% of it (at least 128 MiB) for the arrays the forward pass allocates later; the KV cache and the model's scratch are allocated before the first product, so the measured headroom already excludes them. A weight whose copy does not fit stays on the plain kernels, and an allocation that fails is dropped the same way, so repacking never fails or endangers a load. `neuralfree.cpu.repack.budget` caps the copies explicitly in bytes. When the weights close, one summary line reports the weights and bytes repacked versus kept in place and why, for example `CPU repack (on): repacked 19 weight(s), 256.3 MiB; kept in place 234 weight(s), 3818.7 MiB (heap headroom 258.1 MiB; lookup table 1, over budget 233)`.

Both properties are also accepted on the `serve` command line as `-D<name>=<value>`; a value other than the ones listed in the [property table](#loading-cpu-and-diagnostics) is rejected with an error naming the property.

## The training framework on a GPU backend

**Contexts.** The train library keeps one context per backend (`TrainGpu`), opened, prewarmed and checked the first time an op needs it. The availability result is remembered with its reason, which is logged once at WARN. `TrainGpu.availability(backend)` and `TrainKernels.availability(backend)` return it. A backend's `close()` closes its context first, while the device is still alive.

**Residency.** A tensor lives on one backend, and so do its gradient and every result computed from it. Ops on two tensors require one backend and throw `IllegalArgumentException` otherwise. `Module.to(backend)` moves a module's parameters (in place) and buffers to another backend through the host and drops their gradients. When the train library is available, an op runs on the device and nothing is read back. When it is unavailable, the op computes its CPU reference instead.

**Locking.** Every enqueue that tensor code issues on a backend's `computeQueue()` or `transferQueue()` (kernel launches, fills and transfers) runs under that backend's lock, `BackendResources.lock(backend)`. One backend can therefore serve tensor ops from several threads. Two rules follow:
- **Code that holds one backend's lock never transfers on another.** `PipelineParallel` runs each stage on its own worker thread; stages that share a backend are serialized by that backend's lock, and a tensor that moves between stages on different backends is copied through the host.
- **The GGUF engine does not take this lock.** A backend instance must not be used by the GGUF engine and by tensor ops from different threads at the same time. Give each its own backend, or use them from one thread.

**Ordering.** Train kernels are enqueued only on `backend.computeQueue()`, without waiting for each op. Host transfers of tensor storage are ordered against pending train work (`TrainGpu.beforeHostRead`, `TrainGpu.afterHostWrite`), so a readback always sees the results of every op issued before it. The host waits only at a readback. An index out of range in a device gather, scatter or embedding lookup is recorded on the device and thrown at the next host synchronization, as `IndexOutOfBoundsException` with the message the CPU path gives. On the CPU path the same exception is thrown by the op itself.

**Host readbacks.** Tensor math stays on the device. Data crosses to the host only for these reasons:
- reproducible random streams: masks and noise drawn from a seeded `java.util.Random`, uploaded once;
- values Java control flow consumes: a logged loss or metric, an early-stopping decision, a sampled action or token, routing indices, a convergence test;
- I/O: state dicts, checkpoints, model export, collectives, and moving a tensor between backends;
- sequential factorizations: Cholesky, triangular solves and Jacobi iterations;
- algorithms without a device form: tree ensembles, hidden Markov models and the Gaussian splatting rasterizer;
- graph CSR metadata, built once per graph and cached on the device;
- FLOAT64 conversions, since the library does not require fp64 device support.

A tensor may hold at most `2^31-1` elements; larger tensors are rejected with `IllegalArgumentException`.

**Numerical contract.** Every device op uses the formula and constants of its CPU reference. Sequential loops run in ascending index order, work-group reductions use a fixed 256-wide tree, and the choice between kernel variants depends only on the shapes, so a result never depends on the device. Ties and NaN follow the CPU code. Device results agree with the CPU reference within one of these tolerance classes:

| Class | Ops | Bound |
|---|---|---|
| Bit-identical | Data movement (fill, copy, casts, gather, scatter overwrite, index-select, where), neg, abs, round, floor, max/min, compare, arg-reductions, max-pool and its backward | Exact |
| Single-rounding | add, sub, mul, axpby, scale, mask-scale, scalar add and multiply | Exact for normal operands and results; a value below `FLT_MIN` may flush to a signed zero |
| Division | div, sqrt, average pools, mean, segment mean | `<= 4 ulp(ref)` or `<= FLT_MIN` |
| Transcendental | exp, log, tanh, sigmoid, gelu, silu, elu, softplus, mish, pow, their backwards, softmax, log-softmax | `<= 1e-5 * max(1, \|ref\|)`; NaN and infinities match exactly |
| Reduction | Reductions, GEMM, convolutions, norms, losses, optimizer steps | `<= 1e-5 * sqrt(K) * sum\|terms\| + K * FLT_MIN` against a double-precision reference, `K` the reduction length |

A composition inherits the class of its last operation. The full contract is the package documentation of `com.eonmux.neuralfree.tensor.gpu`.

**Backends you manage yourself.** A backend created through `HWAcceleration` is closed by `shutdown()`, and closing any backend closes its train context. To free a context earlier while the backend stays open, call `TrainGpu.release(backend)` (or the equivalent `GpuTensorOps.clearCache(backend)`, `GpuMatmul.clearCache(backend)`, `GpuConvOps.clearCache(backend)` or `GpuOptimizerOps.clearCache(backend)`); `TrainGpu.releaseAll()` and the dispatchers' `clearCache()` release every backend's context. The next op reopens the context, for example after `KernelLibraries.register(config)` replaced the library's registration.

## LLMDemo command-line reference

Every flag below is parsed by `LLMDemoOptions`. `--help` prints the same list.

### Fused int8 CPU attention and experts

On the CPU backend two fused int8 kernels run where the model's data is already int8.

**Attention.** A cache whose K and V are both `int8_per_token_head` (see
[Per-token-head KV caches](#per-token-head-kv-caches)) is attended directly from its stored codes: each
(token, head) query is quantized once per step to the same symmetric int8 format the cache uses
(`max(absmax / 127, 1e-6)`, round half to even), keys are scored with the exact int8 dot product scaled by the
query and the row's stored scale, and each value row is dequantized inside the accumulation instead of into a
float buffer. The generic path dequantizes every K and V row before scoring, so the fused path differs only by the
query quantization. On the parity fixtures the outputs agree with the dequantizing path to within `1e-2` (queries
already on the int8 grid agree to `2e-5`), and end-to-end logits to within `5e-3`. Learned static sink tokens,
per-KV-head FP8 checkpoint scales and mixed cache dtypes keep the generic path, as does `neuralfree.cpu.int8=false`.

**Experts.** A MoE layer whose routed experts are stored as int8 blocks (`Q8_0` or `I8_B128`) runs one grouped
pass over the step's expert pairs: the gate and up products apply each pair's expert bias and the gated (or
ungated) activation in the product's tile epilogue, the down product applies its expert bias the same way, and the
final per-token weighted sum is fused into the accumulation. Every product is the same int8 vec-dot over the same
quantized row as the generic per-pair path, so the results are bit-for-bit equal. Fused `ffn_gate_up_exps`
tensors, LoRA projection deltas, a post-expert norm, llama4's weighted expert input and media/vision rows keep the
generic path.

Both paths use the Vector API int8 kernels when `jdk.incubator.vector` is resolved and the scalar reference
otherwise; the two produce identical results, so the choice changes speed only. AMX tile instructions are not
reachable from Java, so the fused kernels do not use them and no emulation is attempted.

Add to the property table:

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.cpu.int8` | `false` keeps the generic CPU kernels: attention dequantizes every int8 K/V row before scoring and the MoE runs its experts product by product; `true` runs the fused int8 attention over an `int8_per_token_head` cache and the fused int8 expert pass over `Q8_0`/`I8_B128` expert weights | true |

### Model, prompt and output

| Flag | Meaning | Default |
|---|---|---|
| `--model`, `-m <path>` | GGUF file, Hugging Face checkpoint (directory or `.safetensors` file) or object-store URI (`s3://`, `gs://`, `az://`, [Loading a model from object storage](#loading-a-model-from-object-storage)); see [Hugging Face checkpoints](#hugging-face-checkpoints) and [LLM-INFERENCE.md](LLM-INFERENCE.md#selecting-a-model) | see [LLM-INFERENCE.md](LLM-INFERENCE.md#selecting-a-model) |
| `--hf-repo`, `-hf <user>/<model>[:quant][@revision]` | Download a model from the Hugging Face Hub and open it; the quant defaults to `Q4_K_M`, then `Q8_0`. `HF_HUB_OFFLINE=1` uses the cache only. Excludes `--model` ([Hugging Face Hub download](#hugging-face-hub-download)) | none |
| `--hf-file`, `-hff <name>` | File inside the `--hf-repo` repository, replacing the default choice | none |
| `--prompt`, `-p <text>` | Run one prompt instead of the console chat | console chat |
| `--file`, `-f <path>` | Read the prompt from a file, one trailing newline dropped | |
| `--system`, `-sys <text>` | System prompt | `You are a helpful assistant.` |
| `--system-prompt-file`, `-sysf <path>` | Read the system prompt from a file | |
| `--chat-template <value>` | Replace the model's chat template with Jinja source, a built-in name (`chatml`, `llama3`, `mistral-v7`, ...) or a template file path ([Choosing a chat template](#choosing-a-chat-template)) | the model's |
| `--chat-template-kwargs <json>` | Default chat template variables, a JSON object (anything else is refused); repeatable, later keys win; applied after `--thinking` | |
| `--chat-template-file <path>` | Read the replacement template from a UTF-8 file (at most 4 MiB); excludes `--chat-template` | |
| `--jinja` | Render chat prompts through the model's Jinja template (the default, llama.cpp's flag) | on |
| `--no-jinja` | llama.cpp's non-Jinja route: the model's template is replaced by the built-in template its text detects ([Choosing a chat template](#choosing-a-chat-template)); a template none is detected from is refused at startup | |
| `--raw` | Use the prompt verbatim, without the chat template | off |
| `--thinking` | Enable the chat template's thinking mode | off |
| `-rea`, `--reasoning <on\|off\|auto>` | Thinking mode. `on` (also `enabled`, `true`, `1`) and `off` (also `disabled`, `false`, `0`) bind `enable_thinking`; `on` equals `--thinking`. `auto` (also `-1`) leaves `enable_thinking` to the template. Any other value is refused by name | off |
| `--reasoning-preserve`, `--no-reasoning-preserve` | The `preserve_reasoning` template variable: keep or drop the reasoning of earlier assistant turns, for templates that support it. Without either flag it is `true`, as in llama.cpp | keep |
| `--reasoning-effort <level>` | Set the chat template's `reasoning_effort` (`low`, `medium`, `high`, ...). `default` removes it, keeping the template's own | the template's |
| `--max-tokens`, `--n-predict`, `-n <n>` | Maximum new tokens; `-1` = until end of generation (0 is rejected) | 2560 |
| `--non-interactive` | Without `--prompt`, read the prompt from standard input | off |
| `--mode <m>` | `generate`, `embed`, `classify`, `rerank`, `diffusion` or `audio` | from the model |
| `--embed` | Same as `--mode embed` | off |
| `--show-all-channels` | Print reasoning and tool-call channels with headers | on |
| `--content-only` | Print only the answer channel | off |
| `--reasoning-parser <name>` | Reasoning delimiters by vLLM `--reasoning-parser` name, as `neuralfree.reasoning.parser` ([Reasoning and tool-call parsers](#reasoning-and-tool-call-parsers)) | `auto`: the chat template's |
| `--debug-prompt` | Print the formatted prompt and its token ids | off |
| `--vram-budget`, `--memory-report` | Print the memory plan and device memory, then exit | off |
| `--benchmark` | GPU decode profiling benchmark | off |
| `--bench-streaming` | GPU compute and transfer overlap benchmark | off |
| `--stats-file <path>` | Append the session's statistics to `<path>` when the session ends | none |
| `--help`, `-h` | Print the usage | |

`--benchmark` and `--bench-streaming` exclude each other. Without `--mode`, the mode is chosen from the model:
- a decision head selects `classify`;
- `RANK` pooling selects `rerank` when `--document` is given, else `classify`;
- other pooled models select `embed`;
- otherwise the support class decides: EMBED, CLS, DIFF, AUDIO, or generation.

### Interactive console

Without `-p`, `LLMDemo` opens an interactive chat. The commands follow llama-cli:

| Command | Effect |
|---|---|
| `/exit` (also `/quit`, `quit`, `exit`) | end the session |
| `/regen` | drop the last answer and generate it again from the same history |
| `/clear` (also `/reset`, `clear`, `reset`) | clear the history, pending files and the KV cache |
| `/read <file>` | add a text file to the next message as a `--- File: <name> ---` block |
| `/glob <pattern>` | add every text file matching the pattern (`*`, `?`, `**`, `[a-z]`, `[!x]`; at most 100 files) |
| `/image <file>`, `/audio <file>`, `/video <file>` | attach media to the next message; only in client mode against a server whose `/props` reports the modality, refused by name otherwise |

- **Ctrl+C** stops the answer being generated and keeps the partial text. At the prompt, or pressed a second time, it exits with code 130.
- **Line continuation:** end a line with `\` to continue on the next line, or with `/` to submit without a newline. With `-mli/--multiline-input`, lines keep joining until a line ends in `/` or `\`.
- **Transcript:** `-o/--output <file>` writes the conversation (user turns, attached files, answers with their reasoning) in llama-cli's output-file format. It also works with a single `-p` prompt.
- **Other chat flags:** `-st/--single-turn` ends after the first answer. `-sys/--system-prompt` is an alias of `--system`. `-cnv` forces the chat. `-no-cnv` turns the chat template off, completing `-p` once, or with `-i` running the completion mode.

**Completion mode** (llama-completion `-i`).

```
LLMDemo model.gguf -i -p "Transcript of a dialog.\nUser:" -r "User:" --in-prefix " " --in-suffix "Assistant:"
```

- Raw completion with no chat template, over one growing context.
- The prompt is completed first. Control returns to you when a `-r` reverse prompt appears, at the end of generation (which inserts the first reverse prompt), when the `-n` token budget runs out, or on Ctrl+C.
- Each input enters the context as `--in-prefix` + input + `--in-suffix`. `--in-prefix-bos` adds a BOS before each input.
- `-if/--interactive-first` waits for input before generating anything. An empty line passes control back without adding tokens.
- Escapes (`\n \t \' \" \\ \xHH`) in the prompt, prefix, suffix and input are processed unless `--no-escape` is given.
- When the context is full, its older half (after the BOS) shifts out.

**Client mode** (llama-cli `--server-base`).

```
LLMDemo --server-base http://localhost:8080 [--api-key KEY] [-p "first turn" --image cat.png]
```

- Chats with a running NeuralFree, llama.cpp or OpenAI-compatible server. Nothing is loaded locally.
- The CLI waits for `/health` (riding out a 503 while the model loads), takes the only model or asks you to pick one from `/v1/models`, and reads `/props` for the model name and modalities.
- Answers stream from `/v1/chat/completions`, with the reasoning shown and the server's prompt/generation speed printed.
- Sampling flags you give (`--temp`, `--top-k`, `--top-p`, `--min-p`, `--typical`, the penalties, `--seed`, `-n`, `-r`, `--ignore-eos`) are sent with the request. `--thinking`, `--reasoning-effort` and `--chat-template-kwargs` are sent as `chat_template_kwargs`. Settings you do not give stay the server's.
- `--api-key` (or `NEURALFREE_API_KEY`) is sent as a bearer token.
- `--image <file>` / `--audio <file>` attach media to the `-p` turn.
- Flags that configure a local model (`-m`, `-c`, `--gpu`, `--draft`, `--lora`, `--raw`, …) are refused by name.

### Backend and weights

| Flag | Meaning | Default |
|---|---|---|
| `--gpu` | GPU, automatic backend | on |
| `--cpu` | CPU engine | off |
| `--backend <type>` | `auto`, `opencl`, `cuda`, `vulkan`, `hip`, `metal` ([Metal](#metal-backend-apple-gpus)) or `cpu` | `auto` |
| `--device`, `-device`, `-dev <d1,d2,..>` | GPUs by index or name substring; one device picks the GPU, two or more split the model by layer ([multi-GPU layer split](#multi-gpu-layer-split)); `none` = CPU engine | none |
| `--split-mode`, `-sm <none\|layer\|row\|tensor\|context>` | Multi-GPU split mode ([row](#tensor-parallel-row-split---split-mode-row), [tensor](#tensor-parallelism---split-mode-tensor), [context](#context-parallelism---split-mode-context)) | `none` (`layer` with a multi-device `--device`) |
| `--tensor-split`, `-ts <n0,n1,..>` | Share of each GPU of `--device` (its layers under `layer`, its feed-forward shard under `row`), separated by `,` or `/` | equal |
| `--rpc <[tls://]host:port[,..]>` | Remote GPUs of `nftool rpc-server` appended to a layer split ([Remote compute servers](#remote-compute-servers-layer-split-offload)) | none |
| `--main-gpu`, `-mg <i>` | Index into `--device` of the GPU holding the embedding and the output head | 0 |
| `--threads`, `-t <n>` | CPU engine worker threads | available processors |
| `-fa`, `--flash-attn <on\|auto>` | Flash attention. Attention always runs the fused flash kernels, so `on` and `auto` (llama.cpp's `enabled`, `true`, `1`, `-1` too) are accepted, and `off` is refused by name | on |
| `--ram-cache-mb <n>` | Host cache of streamed layers in MiB; `-1` = automatic ([models larger than RAM](#models-larger-than-ram)), `0` = none | -1 |
| `--vram-depth <n>` | Streamed-layer slots in VRAM; `0` = from the VRAM budget | 0 |
| `--gpu-layers`, `-ngl`, `--n-gpu-layers <n>` | Layers on the device, `auto` or `all` ([hybrid execution](#hybrid-cpu-and-gpu-execution)) | `auto` |
| `--override-tensor`, `-ot <pattern=target,...>` | Tensor placement overrides; target `cpu`, `gpu`, a GPU backend name or an accelerator API name; `cpu` keeps the matched layer on the host | none |
| `--cpu-moe` | Keep every MoE layer's expert weights on the host | off |
| `--n-cpu-moe <n>` | Keep the expert weights of the first `n` MoE layers on the host | 0 |
| `--cpu-offload-mb <n>` | Trunk weights kept in host-mapped memory the device reads in place (UVA zero-copy, vLLM `cpu_offload_gb`) ([Host-mapped weights](#host-mapped-weights-uva-zero-copy)); `0` = off | 0 |
| `--cpu-offload-params <seg,..>` | Parameter segments the host-mapped offload takes, in weight-layer order up to the budget; empty takes any tensor | any |
| `--stream-params <seg,..>` | Parameter segments of a streamed layer that stream; the rest stay resident ([Selective streaming](#selective-streaming)) | every tensor |
| `--no-kv-offload`, `-nkvo` | Keep every layer's KV cache in host RAM: its projections, norms and FFN run on the device, its attention on the host over that cache ([hybrid execution](#hybrid-cpu-and-gpu-execution)) | off |
| `--kv-offload`, `-kvo` | Keep the KV caches on the device | on |
| `--moe-cache <off\|on\|auto\|n>`, `--moe-cache-mib <n>` | Host the routed experts and serve the hot ones from a VRAM cache ([MoE expert cache](#moe-expert-cache)) | off |
| `--moe-cache-ranking <path\|off>` | The `moe-cache-ranking.bin` to preload, or `off` | `moe-cache-ranking.bin` beside the model |
| `--moe-cache-tail-skip <n\|off>` | Drop uncached experts that every token routes only at top-k position `n` (1–9) or later; approximate | `off` |
| `--moe-cache-pool` | On a layer split over two or more GPUs, divide the hosted MoE layers between the GPUs' expert caches by free VRAM ([Pooled expert cache over several GPUs](#pooled-expert-cache-over-several-gpus)) | off |
| `--no-mmap` | Read tensors into memory instead of mapping the file (sets `neuralfree.disable.mmap=true`) | mmap |
| `-lm`, `--load-mode <m>` | How tensors load: `mmap`, `none` (read into buffers), `mlock`, `mmap+mlock` or `dio` (direct reads) ([NUMA placement and memory locking](#numa-placement-and-memory-locking)) | mmap |
| `--mlock` | Pin the model's tensor pages in RAM; with `--no-mmap`, pin the read buffers | off |
| `--numa <s>` | Place the CPU worker threads: `distribute`, `isolate`, `numactl` or `disabled` | disabled |

### Memory

| Flag | Meaning | Default |
|---|---|---|
| `--ctx`, `-c`, `--ctx-size`, `--max-context <n>` | Context length; `0` = fit to the budget | 0 |
| `--parallel`, `-np <n>` | Sequences the memory holds, 1-256; above 1 selects the [paged KV memory](#paged-kv-memory) | 1 |
| `--kv-type <t>` | KV cache type of K and V: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, `fp8_e4m3` (alias `fp8`), `fp8_e5m2`, `fp8_per_token_head`, `int8_per_token_head`, `int4_per_token_head`, and vLLM's vendor formats `nvfp4`, `nvfp4_4over6`, `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`, `fp8_ds_mla`, `nvfp4_ds_mla`, `fp8_inc` ([FP8 KV cache](#fp8-kv-cache), [Per-token-head KV caches](#per-token-head-kv-caches), [KV cache vendor formats](#kv-cache-vendor-formats)) | `f16` |
| `--kv-type-k`, `--cache-type-k`, `-ctk <t>` | K type only | `f16` |
| `--kv-type-v`, `--cache-type-v`, `-ctv <t>` | V type only | `f16` |
| `--ubatch`, `-ub`, `--ubatch-size <n>` | Most tokens per forward step; a prompt splits into equal-size ubatches up to this size | 512 (`embed`, `classify` and `rerank`: max(2048, context) unless given) |
| `--swa-full` | Full-size caches for sliding-window layers | off |
| `--mla-cache <m>` | `auto`, `latent` or `decompressed` | `auto` |
| `--no-prefix-reuse` | Re-evaluate the whole prompt every turn | reuse on |
| `--context-shift` / `--no-context-shift` | Drop old positions when the context is full | off |
| `--grp-attn-n, -gan <n>` / `--grp-attn-w, -gaw <n>` | Self-extend: group factor and span ([self-extend](#self-extend-group-attention-and-position-division)) | 1 / 512 |
| `--no-kv-rotation` | Store quantized KV rows without Hadamard rotation | rotation on |
| `--prompt-cache <file>` | Load the session file at start and save the state back after the first turn ([Prompt cache](#prompt-cache)) | none |
| `--prompt-cache-all` | Save every turn's prompt and output, not the first turn's alone; also runs the completion REPL (`-i`, `-if`), saving each turn | off |
| `--prompt-cache-ro` | Load the prompt cache but never write it | off |

The context fit of a CPU load measures the memory it may plan against as the free JVM heap less a reserve for loading and per-call allocations (an eighth of what is free, at least 64 MiB). When that reading leaves nothing, the heap is collected once and measured again, so unreachable objects do not refuse a load that fits once they are reclaimed, as vLLM collects before it measures the free memory of its profile run.

#### NUMA placement and memory locking

On a multi-socket host, NeuralFree can place its CPU thread pool on NUMA nodes and pin the model's tensor pages in RAM.

`--numa <distribute|isolate|numactl>` (property `-Dneuralfree.numa=...`):

| value | placement |
|---|---|
| `distribute` | worker `N` runs on the CPUs of NUMA node `N % node count` |
| `isolate` | every worker runs on the NUMA node the process started on |
| `numactl` | every worker keeps the CPU set inherited from `numactl` |
| `disabled` / `none` / `off` (default) | threads run wherever the OS schedules them |

The strategy is read when the CPU pool starts (the first CPU computation), so set it before the model runs. On a host with one NUMA node, or without the Linux sysfs topology, it logs that it has no effect. When placement is active, the model's memory mapping is also advised `POSIX_MADV_RANDOM`, and a warning is logged when `/proc/sys/kernel/numa_balancing` is on, since automatic balancing has been observed to hurt CPU inference.

Tensor loads and RAM pinning (`-Dneuralfree.tensor.read=...`):

| flag | mode | behaviour |
|---|---|---|
| (default) | `mmap` | tensors are memory-mapped, backed by the page cache |
| `-lm none` / `--no-mmap` | `read` | tensors are read into process buffers |
| `-lm mmap` | `mmap` | as the default |
| `-lm mlock` | `mlock` | tensors are read into process buffers and those buffers are pinned in RAM |
| `-lm mmap+mlock` | `mmap+mlock` | tensors are mapped and the mapping is pinned in RAM |
| `-lm dio` | `direct` | sector-aligned direct reads |
| `--mlock` | `mmap+mlock` | pin the model's tensor pages; combined with `--no-mmap` it becomes `mlock` |

`--mlock` forces the kernel to keep the model's tensor pages in RAM, so they are neither swapped nor compressed. It needs a sufficient locked-memory limit: when the kernel refuses, a warning names `RLIMIT_MEMLOCK` (`ulimit -l`) and inference continues with pageable memory. `dio` bypasses the page cache and lazily read tensors are fetched row by row, so neither has resident bytes to pin: `--mlock` with those modes is refused by name.

The `nftool bench` load-mode sweep (`-lm`) accepts all six modes, and `--numa` sets the placement before the pool starts. The server accepts the same flags:

```
neuralfree --model model.gguf --numa distribute --mlock -t 32
neuralfree serve -m model.gguf --numa isolate --mlock
```

#### FP8 KV cache

Besides llama.cpp's cache types, the cache can store FP8, under vLLM's `--kv-cache-dtype` names: `fp8_e4m3` (or the alias `fp8`) and `fp8_e5m2`. One OCP FP8 code is stored per value, half the bytes of F16.

A quantized checkpoint's per-layer scales `blk.N.attn_k.kv_scale` and `blk.N.attn_v.kv_scale` (one value, or one per KV head) drive the quantization, as in vLLM: the store writes the code of `value / kv_scale`, rounded to nearest even and saturating, and attention multiplies the decoded rows by the scale again. Without those tensors the scale is 1.0. A cache that is not FP8 ignores the checkpoint scales, as vLLM enforces.

K and V may use different types (mixed flash kernels). Every cache type works with paging, split-K decode, the context shift (the RoPE rotation is linear, so it moves the stored, scale-divided codes exactly) and the prefix cache. Examples: `-ctk fp8 -ctv fp8` uses both scales; `-ctk fp8_e4m3 -ctv f16` keeps an exact, unscaled V cache. The draft flags `-ctkd`/`-ctvd` take the same names; a draft cache has no checkpoint scales.

#### Per-token-head KV caches

Under vLLM's `CacheDType` names, `fp8_per_token_head`, `int8_per_token_head` and `int4_per_token_head` store a
dynamic scale for every (token, KV head) row: at store time the writer computes
`max(absmax / qmax, 1e-6)` with qmax 448 (FP8 E4M3FN), 127 (INT8) or 7 (INT4), stores the scale beside the row and
writes the codes of `value / scale`; attention multiplies each decoded head by its row's scale. FP8 stores one
E4M3FN code per value, INT8 one signed byte, INT4 two signed nibbles per byte (the first value in the low nibble).
The checkpoint's `blk.N.attn_k.kv_scale` / `blk.N.attn_v.kv_scale` tensors are not used by these types.

The scales are part of the cached row. Page copies, the context shift (the RoPE rotation re-encodes the rotated
codes under the unchanged row scale), the prefix cache and the session and slot files carry them with the codes, so
a cache saved on the CPU restores on the device and back bit for bit.

K and V must both be per-token-head or neither: attention reads both sides' scales from one buffer. A mixed pair
such as `-ctk int8_per_token_head -ctv f16` is refused with an error naming both types. The two sides may still
use different per-token-head types (for example FP8 K with INT4 V). These caches work on the CPU and the GPU flash
paths, including paging, split-K decode, the context shift and the prefix cache.

The MLA latent/K/V caches, cross-attention caches, the DSA and MSA index caches and the DeepSeek-V4 raw K cache
reject these types. A draft cache on a GPU also rejects them and runs on the host.

Example: `-ctk fp8_per_token_head -ctv fp8_per_token_head`.

#### KV cache vendor formats

`--kv-type` (engine property `neuralfree.kv.type`) accepts these names besides the llama.cpp and FP8 types:

```
nvfp4              NVFP4: E2M1 values, one E4M3 scale per 16 values
nvfp4_4over6       NVFP4 with the per-group scale searched between max/6 and max/4
turboquant_k8v4    FP8 E4M3 keys + 4-bit values
turboquant_4bit_nc 4-bit Lloyd-Max keys + 4-bit values, norm correction
turboquant_k3v4_nc 3-bit Lloyd-Max keys + 4-bit values, norm correction
turboquant_3bit_nc 3-bit Lloyd-Max keys + 3-bit values, norm correction
fp8_ds_mla         DeepSeek MLA latent rows: 512 E4M3 NoPE + 64 BF16 RoPE
nvfp4_ds_mla       DeepSeek MLA latent rows: 512 E2M1 NoPE + 64 E4M3 RoPE
fp8_inc            Intel Gaudi's name for fp8_e4m3 (same cache)
```

Examples:

```bash
# NVFP4 on both caches; blk.N.attn_k/attn_v.kv_scale is applied around the store as for FP8
neuralfree --model model.gguf --kv-type nvfp4 --ctx-size 4096

# a TurboQuant preset sets both the key and the value stream
neuralfree --model model.gguf --kv-type turboquant_k8v4 --ctx-size 4096

# a DeepSeek MLA latent cache
neuralfree --model deepseek.gguf --kv-type fp8_ds_mla --ctx-size 4096
```

Rules:

- A TurboQuant preset names a key stream and a value stream together. It is accepted only by `--kv-type` (or `neuralfree.kv.type`), never by the per-side flags `--kv-type-k` / `--kv-type-v` / `-ctk` / `-ctv`; the per-side flags still accept individual types, and a key stream must pair with a value stream.
- NVFP4 uses the layer's checkpoint KV scale exactly like FP8: the cache stores `value / k_scale` (or `v_scale`) and attention multiplies it back; a missing tensor means 1.0. TurboQuant and the MLA latent rows carry their scales inside the cache and ignore the checkpoint scales.
- NVFP4 needs head widths that are multiples of 16. TurboQuant Lloyd-Max keys need a power-of-two head width (the Hadamard rotation); every TurboQuant stream packs its codes in groups of 8, so all four presets need head widths that are multiples of 8 (`turboquant_k8v4` included, through its value stream).
- `fp8_ds_mla` and `nvfp4_ds_mla` fill the single MLA latent row of exactly 576 values (`kv_lora_rank` 512 plus 64 RoPE); they are refused on per-head and cross-attention caches, and on MLA models with another latent size.
- Vendor formats are refused by name on the DSA indexer, MSA, DeepSeek-V4 raw and T5 cross-attention caches, whose stores have their own formats.
- `q_scale` and `prob_scale` (`blk.N.attn_q.q_scale`, `blk.N.attn_prob_scale`) are not consumed by any of these formats. They stay claimed-but-unused: none of the formats needs FP8 queries or probabilities, and vLLM's TurboQuant decode keeps queries in high precision.

Storage (values per token row, plus side planes):

- NVFP4: 0.5 byte per value, one E4M3 scale per 16 values (`0.5 + 1/16` bytes per value).
- TurboQuant keys: 1 byte per value (`k8v4`) or 3/4 bit per value with one F16 norm per KV head; values: 3/4 bit per value with one F16 scale and zero per KV head.
- `fp8_ds_mla`: 512 bytes of NoPE codes + 128 bytes of RoPE + a 16-byte tile-scale plane (vLLM's record is 656 bytes with the scales inline).
- `nvfp4_ds_mla`: 256 bytes of packed NoPE + 64 bytes of RoPE + a 32-byte scale plane (vLLM's record is 352 bytes with the scales inline).

Backends:

- The CPU backend implements all of them end to end, including context shift, session files, page copies and slot files.
- The GPU backends keep them in device memory, the default placement (`neuralfree.kv.offload=true`): `nvfp4`, `nvfp4_4over6`, `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`, `fp8_ds_mla` and `nvfp4_ds_mla`, mixed K/V pairs included. The packed codes and their per-group, per-tile and per-head scales, the TurboQuant centroids and the DeepSeek latent scales live in one side buffer per layer that the store kernels fill and the flash kernels read; state transfer, session files and paged block images move it with the rows. A TurboQuant MSE key cache is scored by rotating the query with the same Hadamard matrix. `fp8_ds_mla` and `nvfp4_ds_mla` keep vLLM's 576-value latent row (`kv_lora_rank` 512 + 64 RoPE values, V read from the same row), so an absorbed-MLA model keeps its latent cache on the device.
- On the GPU, context shift and the sparse-attention side caches (DSA, MSA and the DFlash draft cache) refuse a vendor type by name, as does dual-chunk attention over a TurboQuant MSE key cache. `--no-kv-offload` (`neuralfree.kv.offload=false`) keeps the caches in host RAM and decodes through the host path of the same format.

### RoPE and long context

These flags override the model's stored RoPE parameters. They write the `%s.rope.*` metadata keys before the model loads, so the model is cached and served under a separate prefix-cache namespace.

| Flag | Property | Meaning |
|---|---|---|
| `--rope-scaling <t>` | `neuralfree.rope.scaling` | `none`, `linear`, `long`, `yarn`, `dynamic` or `ntk` (`rope.scaling.type`) |
| `--rope-scale <n>` | `neuralfree.rope.scale` | Context scaling factor: the context expands by `n` (`rope.scaling.factor = n`) |
| `--rope-freq-scale <n>` | `neuralfree.rope.freq.scale` | Frequency scale: the context expands by `1/n` (`rope.scaling.factor = 1/n`), as llama.cpp's `rope_freq_scale` |
| `--rope-freq-base <f>` | `neuralfree.rope.freq.base` | RoPE base frequency (`rope.freq_base`) |
| `--yarn-orig-ctx <n>` | `neuralfree.yarn.orig.ctx` | Trained context length (`rope.scaling.original_context_length`); `0` keeps the model's |
| `--yarn-ext-factor <f>`, `--yarn-attn-factor <f>`, `--yarn-beta-fast <f>`, `--yarn-beta-slow <f>` | `neuralfree.yarn.ext.factor`, `.attn.factor`, `.beta.fast`, `.beta.slow` | YaRN extrapolation mix, attention magnitude and correction ranges |

`--rope-scale` and `--rope-freq-scale` must be positive.

**Fixed and mixed NTK scaling.** vLLM's `ntk` rope type keeps the trained base and divides every pair's frequency by a factor: the scaling factor itself in the fixed form, and a divisor that grows per pair with `mixed_b` (`{arch}.rope.scaling.mixed_b`) in the mixed form. The divisors are computed per layer while the RoPE spec is resolved.

**Dynamic NTK scaling.** vLLM's `dynamic` rope type scales the RoPE base rather than the angles, in both of its forms:

- `rope_scaling: {"rope_type": "dynamic", "factor": F}` — the base becomes
  `theta * ((F * context_length / trained_length) - (F - 1))^(rotary_dim / (rotary_dim - 2))`.
  `trained_length` is `rope_scaling.max_trained_positions` (or `rope_parameters.max_trained_positions`,
  or `original_max_position_embeddings`) when the config states one, otherwise the declared context length.
- `rope_scaling: {"rope_type": "dynamic", "alpha": A}` — the base becomes
  `theta * A^(rotary_dim / (rotary_dim - 2))` (the HunYuan/XDRoPE form). When both are present `alpha` wins, as in
  vLLM.

`context_length` is `max_position_embeddings` and the rotary width is the model's own `n_rot` (or the linear-attention
head width), so partial-rotary and sliding-window layers scale with their own width. The effective base is computed
per layer while the RoPE spec is resolved; `freq_scale` stays 1 because NTK scaling changes the base, not the angles.
This works for HF checkpoints served directly and for GGUFs that carry `rope.scaling.type=dynamic`.

On the command line, for a model trained at 32768 positions with a factor of 4: `--rope-scaling dynamic --rope-scale 4 --yarn-orig-ctx 32768`. The factor is evaluated once at the declared context length, as vLLM's cache does, not per token as Hugging Face's formula does. `rope.scaling.type=dynamic` is a NeuralFree metadata value that stock llama.cpp rejects; a GGUF meant for llama.cpp keeps a pre-scaled `rope.freq_base` instead.

**Dual-chunk attention.** Qwen2.5-1M's `dual_chunk_attention_config` (vLLM's `DualChunkFlashAttentionBackend`) loads from the Hugging Face config (defaults `chunk_size` 8192, `local_size` 1024, `original_max_position_embeddings` 32768) and is recorded as `{arch}.attention.dual_chunk.*`. With `cl = chunk_size - local_size`, keys rotate at `p % cl` and each query scores three key ranges with its own rotation: its chunk causally (intra), the previous chunk (succ) and everything before it (inter); the three partial softmaxes merge into one. Prefill, chunked prefill and decode follow vLLM's ranges, and logits scale by `max(1, 0.1 * ln(seqLen / original_max_position_embeddings) + 1)` squared, as vLLM applies it to keys and queries. It runs on the CPU and GPU backends, over contiguous and paged caches and a rotated quantized KV cache. Refused by name: vLLM's vertical-slash sparse attention (`sparse_attention_enabled: true` or a `sparse_attention_config`, whose per-head table a GGUF cannot carry), context shift, a host KV offload, a shared KV cache, a sliding window, and attention variants the three-stage merge cannot express (sinks, softcap, NoPE, Q/K norm after RoPE, non-causal layers).

These RoPE forms are refused at load, naming the configuration key, instead of loading with wrong positions:

- The Fourier position embedding keys `use_fope`, `fope_sep_head`, `num_inv_freq` and `fope_init_factor` on any
  architecture but InternS1-Pro ([InternS1-Pro](#hugging-face-checkpoints) carries the learned coefficients FoPE needs).
- A `dynamic` config with neither `alpha` nor `factor` is rejected.

### Sampling

The demo's defaults apply only where no `neuralfree.*` sampler property is set. The engine defaults, which match llama.cpp, are listed under [Sampling](LLM-INFERENCE.md#sampling). The model's own defaults — the `general.sampling.*` keys of a GGUF, or `generation_config.json` of a Hugging Face checkpoint — are applied next, for every setting the caller did not give ([Model sampling defaults](#model-sampling-defaults)).

| Flag | Meaning | Demo default |
|---|---|---|
| `--temp`, `--temperature <f>` | Temperature; `0` = greedy | 0.6 |
| `--top-k <n>` | Top-k; `0` = off | 40 |
| `--top-p <f>` | Top-p | 0.95 |
| `--min-p <f>` | Min-p | 0.05 |
| `--typical`, `--typical-p <f>` | Locally typical sampling; `1` = off | 1 |
| `--repeat-penalty <f>` | Repetition penalty; `1` = off | 1.0 |
| `--presence-penalty <f>` | Presence penalty; `0` = off | 0 |
| `--frequency-penalty <f>` | Frequency penalty; `0` = off | 0 |
| `--repeat-last-n <n>` | Penalty window; `0` = off, `-1` = whole context | 256 |
| `--repeat-window <n>` | Penalty window; `0` = whole context | |
| `--seed`, `-s <n>` | Sampler seed; `-1` = random | -1 |
| `--ignore-eos` | Keep generating past end-of-generation tokens | off |
| `--reverse-prompt`, `-r <text>` | Stop when the output contains `<text>`; repeatable | none |
| `--min-keep <n>` | Fewest candidates top-k, top-p, min-p and typical-p keep | llama.cpp's |
| `--top-n-sigma`, `--top-nsigma <f>` | Top-n-sigma logit mask; `<= 0` = off | off |
| `--xtc-probability <f>`, `--xtc-threshold <f>` | XTC trimming; probability 0 or a threshold above 0.5 = off | off |
| `--dry-multiplier <f>`, `--dry-base <f>`, `--dry-allowed-length <n>`, `--dry-penalty-last-n <n>` | DRY repetition penalty; multiplier 0 = off | off |
| `--dry-sequence-breaker <s>` | DRY restart sequence; repeatable. The first one replaces the defaults (`\n`, `:`, `"`, `*`), and `none` leaves none | llama.cpp's |
| `--mirostat <0\|1\|2>`, `--mirostat-ent <f>`, `--mirostat-lr <f>` | Mirostat 1 or 2, replacing the sampler chain; target entropy and learning rate | off |
| `--dynatemp-range <f>`, `--dynatemp-exp <f>` | Dynamic temperature; range 0 = off | off |
| `--samplers <name;name;...>` | The sampler chain order by llama.cpp's sampler names (`penalties`, `dry`, `top_n_sigma`, `top_k`, `typ_p`, `top_p`, `min_p`, `xtc`, `temperature`, `infill`, `adaptive_p`); kebab-case and the `nucleus`, `temp` and `typ` aliases work too | llama.cpp's |
| `--sampler-seq`, `--sampling-seq <chars>` | The sampler chain order, one character per sampler (llama.cpp): `d` dry, `k` top_k, `y` typ_p, `p` top_p, `s` top_n_sigma, `m` min_p, `t` temperature, `x` xtc, `i` infill, `e` penalties, `a` adaptive_p | llama.cpp's |
| `--adaptive-target <f>` | Adaptive-p target probability in [0, 1]; negative = off | -1 |
| `--adaptive-decay <f>` | Adaptive-p decay of the target adaptation, in [0, 0.99] | 0.9 |
| `--reasoning-budget <n>` | Tokens a reasoning block may take before its end tag is forced; `-1` = unlimited, `0` = end at once. Needs a reasoning parser with tags (the chat template's, or `--reasoning-parser`) | -1 |
| `--reasoning-budget-message <text>` | Text forced ahead of the end tag when the budget runs out | none |
| `--logit-bias`, `-l <id(+\|-)bias>` | Add a bias to a token's logit before the chain, e.g. `-l 15043+1`; repeatable. `-100` and below ban the token | none |
| `--grammar <gbnf>`, `--grammar-file <path>` | Constrain the output to a GBNF grammar ([Grammar-constrained generation](#grammar-constrained-generation)) | none |
| `--json-schema`, `-j <json>`, `--json-schema-file`, `-jf <path>` | Constrain the output to a JSON schema's instances; the last constraint flag wins | none |

In client mode (`--server-base`) these flags travel as the request's `samplers`, `adaptive_target`,
`adaptive_decay`, `reasoning_budget_tokens` and `reasoning_budget_message` fields. The modes without an
autoregressive token loop (`--mode embed|classify|rerank|diffusion|audio`) refuse `--grammar`/`--json-schema` and
`--reasoning-budget` by name.

#### Custom logits processors

A custom logits processor adjusts the next-token candidate set before the configured sampling stages (penalties,
top-k, top-p, min-p, temperature) and can be installed server-wide and activated per request, like vLLM's
`LogitsProcessor`.

**Writing one.** Implement `com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorFactory`. `create` is called
once per generation with the request's `vllm_xargs` arguments; return `null` to leave that request alone and a
fresh `SamplerStage` to apply. `validate` runs at request admission, so throwing `IllegalArgumentException` there
answers a 400 naming the argument. Factories are shared across requests and must be stateless and thread-safe; a
stage that keeps state across accepted tokens overrides `saveState`/`restoreState` (needed to roll back under
speculative decoding).

```java
package com.example;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.sampling.*;

/** Keeps only vllm_xargs.target_token, for the requests that name it. */
public final class TargetTokenFactory implements LogitsProcessorFactory {
    @Override
    public String name() {
        return "target_token";
    }

    @Override
    public void validate(LogitsProcessorArgs args) {
        if (args.has("target_token")) {
            args.integer("target_token"); // refuses a non-integer by name
        }
    }

    @Override
    public SamplerStage create(LogitsProcessorArgs args) {
        if (!args.has("target_token")) {
            return null;
        }
        int target = args.integer("target_token");
        return new SamplerStage() {
            @Override
            public void apply(Sampler.Candidates candidates) {
                for (int i = 0; i < candidates.size(); i++) {
                    if (candidates.id(i) != target) {
                        candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                    }
                }
                candidates.setSorted(false);
            }
        };
    }
}
```

**Installing server-wide.** Either name the class on the command line:

```bash
neuralfree serve model.gguf --logits-processors com.example.TargetTokenFactory another.Factory
```

or package it in a plugin jar: implement
`com.eonmux.neuralfree.gguf.plugin.LogitsProcessorProvider` (returning your factories from
`logitsProcessorFactories()`), list it in
`META-INF/services/com.eonmux.neuralfree.gguf.plugin.LogitsProcessorProvider`, and put the jar on the application
class path. Provider factories are always loaded, as vLLM's entry-point plugins are. The same classes can be set
with the `neuralfree.logits.processors` system property (comma-separated). A processor name or alias registered
twice is rejected with both sources named.

**Activating per request.** Send a `vllm_xargs` object in the OpenAI completions/chat (or Anthropic Messages)
body. Values are strings, booleans, integers, numbers or flat arrays of those; each factory sees them and decides
whether it applies:

```bash
curl http://localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "model.gguf", "prompt": "The capital of France is",
  "max_tokens": 8, "temperature": 0.8, "top_k": 1,
  "vllm_xargs": {"target_token": 67}
}'
```

A malformed `vllm_xargs` (not an object, a nested object/array, a null value, a wrong type) answers 400 naming
`vllm_xargs.<argument>`. The arguments are carried on
the request (`GenerationRequest.logitsProcessorArgs()`) and re-validated when the engine builds the sampler, so
callers of the Java API can set them with
`GenerationRequest.builder().logitsProcessorArgs(LogitsProcessorArgs.of(map))`.

Only the operator installs processors (the flag, the property or a jar on the class path); a request's `vllm_xargs` passes arguments to the installed ones and cannot load a class.

### Speculative decoding

| Flag | Meaning | Default |
|---|---|---|
| `--draft`, `-md`, `--model-draft`, `--spec-draft-model <gguf>` | Drafter: a small model, an MTP draft file, EAGLE-1, eagle3, gemma4-assistant or dflash | none |
| `--mtp` | Draft with the target's own NextN/MTP blocks (excludes `--draft`) | off |
| `--spec-type <t>` | How drafting runs: a comma-separated list of drafter types (see **Drafter types** below); repeated flags append | `auto` |
| `--draft-max`, `--draft-n-max`, `--spec-draft-n-max <n>` | Maximum drafted tokens per round | 3 |
| `--draft-min`, `--draft-n-min`, `--spec-draft-n-min <n>` | Minimum draft length that is verified | 0 |
| `--draft-p-min`, `--spec-draft-p-min <p>` | Stop drafting below this probability | 0 |
| `--spec-draft-p-split <p>`, `--draft-p-split` | Probability above which a secondary draft candidate splits a branch into a tree (needs a draft model and `-np > 1`) | 0.1 |
| `--spec-draft-tree`, `--no-spec-draft-tree` | A dflash drafter proposes a draft tree verified in one masked forward ([draft-tree verification](#single-pass-draft-tree-verification)) | off |
| `--spec-draft-tree-nodes`, `--spec-tree-nodes <n>` | Fixed tree verify width, at most 63 and `ubatch - 1` | 16 |
| `--spec-draft-tree-temp`, `--spec-tree-temp <t>` | Temperature of the tree builder's path scores | 1.0 |
| `--kv-type-draft <t>` | KV cache type of the draft contexts, K and V: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, `fp8_e4m3` (alias `fp8`), `fp8_e5m2`, `fp8_per_token_head`, `int8_per_token_head`, `int4_per_token_head` (on a GPU the draft cache runs on the host for these) | `f16` |
| `--kv-type-k-draft`, `--cache-type-k-draft`, `--spec-draft-type-k`, `-ctkd <t>` | Draft cache type of K only | `f16` |
| `--kv-type-v-draft`, `--cache-type-v-draft`, `--spec-draft-type-v`, `-ctvd <t>` | Draft cache type of V only (an MLA draft context needs K = V) | `f16` |
| `--spec-draft-ubatch-size`, `-ubd`, `--ubatch-size-draft`, `--spec-draft-n-ubatch <n>` | Tokens per forward step of a draft context; 0 keeps the target's `--ubatch` | 0 |
| `--spec-draft-device`, `-devd`, `--device-draft <d1,d2,..>` | The GPUs of the draft model's own load, like `--device` but for the draft: device indexes, name substrings, or `none` alone to keep the draft off every GPU (a host draft); unset keeps the draft beside the target's weights on the target's backend | the target's `--device` |
| `--spec-draft-ngl`, `-ngld`, `--gpu-layers-draft`, `--n-gpu-layers-draft <n>` | Layers of the draft model on its device: `auto`, `all` (llama.cpp's `-1` / `-2` markers work), or a count that includes the output head like `-ngl` | `auto` |
| `--spec-draft-override-tensor`, `-otd`, `--override-tensor-draft <pattern=target,...>` | Tensor placement overrides of the draft model's own load, same rules as `-ot`; they replace the target's for the draft load, never inherit it | none |
| `--spec-draft-cpu-moe`, `-cmoed`, `--cpu-moe-draft` | Keep every MoE layer's expert weights of the draft model on the host (the draft's own `--cpu-moe`) | off |
| `--spec-draft-n-cpu-moe`, `-ncmoed`, `--spec-draft-ncmoe`, `--n-cpu-moe-draft <n>` | Keep the expert weights of the first `n` MoE layers of the draft model on the host | 0 |
| `--spec-draft-threads`, `-td`, `--threads-draft <n>` | CPU thread pool size the draft model's host tensors run on. The engine runs the target's and the draft's host tensors on one pool, so `-tbd` (default `-td`) must agree with `-td` and with `-t` | the target's `--threads` |
| `--spec-draft-threads-batch`, `-tbd`, `--threads-batch-draft <n>` | The draft's batch threads; sizes the same one pool as `-td` and must agree with it | `-td` |

**When speculation pays.** A round costs one verify forward of `k + 1` rows plus `k` drafter decodes and emits the
confirmed drafts plus one token, so it beats plain decoding only when the drafter is confirmed often enough. On the
RTX 5080 Laptop with a Qwen3 4B Q8_0 target and a 0.6B Q8_0 drafter at `--draft-max 8`, a 9-row verify costs about
1.3 plain decode steps of device time on CUDA and 2.5 on OpenCL, and a drafter decode about 0.3 of a target step on
both, so a round costs about 3.7 target steps on CUDA and 5.0 on OpenCL. Emitting `a·8 + 1` tokens, the pair needs
about 33% of drafted tokens confirmed to break even on CUDA and about 50% on OpenCL; it reaches 20-37%, so the
chain is at best even with plain decoding on CUDA and slower on OpenCL for this pair. Short drafts cost less per
round: on CUDA a 2-row or 4-row verify of the Q8_0 target costs about 1.05x a decode step. The drafter picks its
token from 10 candidates read back from the device, not the full logits row, under greedy drafting over the
target's vocabulary.

**Draft model placement.** A separate draft model (`--draft` with `--spec-type auto` or `draft-simple`) carries its
own placement settings, as llama.cpp's do (common/speculative.cpp): `-devd` names the draft's own GPUs — one
selector a one-GPU draft load, two or more a layer split, `none` alone a host draft; unset keeps the draft beside
the target's weights on the target's backend. `-ngld`, `-otd`, `-cmoed` and `-ncmoed` are the draft load's `-ngl`,
`-ot`, `--cpu-moe` and `--n-cpu-moe`: the count includes the output head, an override's `cpu` match keeps the
matched tensor's layer on the host (a routed expert stack stays in RAM beside its device layer), and the values
replace — never inherit — the target's. A draft on its own devices reserves nothing beside the target's context
fit, so `-devd` frees target VRAM for context. Every placement flag needs `--draft` and a route that loads a model
of its own: `draft-mtp` and the hidden-fed drafters refuse them by name
(`neuralfree.draft.gpu.devices`, `neuralfree.draft.n.gpu.layers`, `neuralfree.draft.override.tensor`,
`neuralfree.draft.cpu.moe`, `neuralfree.draft.n.cpu.moe`), pointing at `draft-simple` for a placeable draft. The
draft thread flags `-td`/`-tbd` size the one CPU pool `-t` (`neuralfree.cpu.threads`) sizes — llama.cpp keeps two
pools, the engine runs one — so disagreeing flags are refused naming them.

**Drafter types.** `neuralfree.draft.type` (CLI `--spec-type`, server `--spec-type`) takes a comma-separated list. On the CLI,
repeated flags append. The first flag replaces the property's list.

| Type | Drafts from | Needs |
|------|-------------|-------|
| `auto` (default) | the draft file, with the route detected from it | `--draft` / `neuralfree.draft` |
| `draft-mtp` | the NextN/MTP blocks of the draft file, or of the target without a file | MTP blocks |
| `draft-simple` | a separate draft model | `--draft` |
| `ngram-simple` | the last n-gram looked up in the history | nothing |
| `ngram-map-k` (alias `ngram-map`) | a key n-gram map of the history | nothing |
| `ngram-map-k4v` | a key n-gram map with four values per key | nothing |
| `ngram-mod` | a hashed n-gram table | nothing |
| `ngram-cache` | n-gram caches of the context, a static file and a dynamic file | nothing |
| `suffix` | a suffix tree of the history (vLLM suffix decoding) | nothing |

Several types chain in llama.cpp's order: the lookup types in the order of the table, then the draft-file type.
For each round, the first drafter with a proposal drafts. `neuralfree.draft.max` caps every drafter. A lookup
drafter also stops at its own m-gram size, and the draft-file drafter stops at its own limit (DFlash block, EAGLE3
depth).

Example: `--spec-type ngram-simple,suffix --draft small.gguf --spec-type draft-simple`.

**EAGLE-1 drafters.** A standalone EAGLE-1 drafter (SafeAILab EAGLE, vLLM's `"method": "eagle"`:
`EagleLlamaForCausalLM`, `EagleMistralForCausalLM`, `EagleForCausalLM`, `EagleModel`, `LlamaEagleForCausalLM`)
runs under `auto` beside its target: `--draft` takes the Hugging Face checkpoint directory directly, assembled
at load with the target, from which the drafter takes its vocabulary and tokenizer, or a GGUF of it
(`eagle`-architecture, with `eagle.target_layers` naming the target's last layer and `eagle.target_hidden_size`). The
drafter must have exactly one decoder layer and the target's width and vocabulary, and both the llama shape (only
the layer's `ffn_norm`) and the mistral shape (input and final norms) load. `--draft-max` bounds a proposal and
`--draft-p-min` stops the greedy draft below that probability. Forcing `draft-mtp` or `draft-simple` on an EAGLE
file is refused by name. The EAGLE row runs GQA attention only; drafters for MLA targets load through
their own routes (`EagleDeepSeekMTPModel` over a DeepSeek target as a DeepSeek MTP-shaped EAGLE draft,
`EagleMistralLarge3ForCausalLM` as a concat-fc drafter, both below).

**EAGLE-3 drafters.** A Hugging Face EAGLE-3 draft directory (`--draft`) or a GGUF of it (`eagle3` architecture)
runs under `auto` beside its target. The route serves vLLM's `llama_eagle3` layouts and the Qwen3 EAGLE-3 layout
of `qwen3_eagle3` (the `Eagle3Qwen3ForCausalLM` and `PeagleQwen3ForCausalLM` checkpoints): Qwen3's per-head Q/K
RMS norms, the Q/K/V projection biases of `attention_bias` checkpoints, the whole-input `input_norm` of
`norm_before_fc`, the per-tap `fc_norm.<part>` norms and the draft `logit_scale` all load, and the Q/K projections
take the same permutation as every EAGLE-3 draft. The target taps are the configuration's
`eagle_aux_hidden_state_layer_ids` (at the root or under `eagle_config`); a draft that names none fuses vLLM's
default `[2, n // 2, n - 3]` of the target, the same three layers llama.cpp picks. `draft_vocab_size` defaults to
the draft's `vocab_size`, and a `d2t` map is applied when the checkpoint carries a reduced draft vocabulary. The
checkpoint must agree with itself: the `fc_norm` flag and the `fc_norm.<part>.weight` tensors are both required or
both absent, and the declared aux-layer count must be three. A draft with `use_aux_hidden_state: false`,
`norm_output: true`, a `sliding_attention` first layer, a non-three aux or tap count, or an `fc_norm`/tensor
mismatch is refused by name. The DeepSeek EAGLE-3 drafts of vLLM's `deepseek_eagle3`
(`Eagle3DeepseekV2ForCausalLM`, and its `Eagle3DeepseekV3ForCausalLM` alias) also load: their single decoder layer
runs DeepSeek's multi-head latent attention over the target fusion, so the low-rank query path
(`q_a_proj`/`q_a_layernorm`/`q_b_proj`, or a plain `q_proj` without `q_lora_rank`), the latent KV path
(`kv_a_proj_with_mqa`/`kv_a_layernorm`) and the `kv_b_proj` split into the absorbed per-head `attn_k_b`/`attn_v_b`
all convert, and the draft carries `kv_lora_rank`, the decompressed key/value head widths, `rope.dimension_count`
and the YaRN log multiplier. The draft still reads the target's three taps and shares the target tokenizer, and
`--spec-type auto` runs it beside a DeepSeek MLA target with lossless verification.

**DeepSeek MTP-shaped EAGLE drafts.** `--draft, -md` also takes vLLM's `EagleDeepSeekMTPModel` checkpoints — the fused drafter `deepseek_eagle.EagleDeepseekV3ForCausalLM` runs over DeepSeek V2/V3 MLA targets. The checkpoint's `config.json` is what selects the conversion: one with the MLA geometry (`kv_lora_rank`, or `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim`) converts through the DeepSeek mapping, and one without it keeps the llama/mistral EAGLE-1 mapping. The draft holds exactly one decoder layer, keeps its own LM head (no `tie_word_embeddings`), and must have as many KV heads as attention heads; its routed experts (DeepSeek MoE) are supported. The draft's embedding, LM head and final norm are read from the file when present and otherwise taken from the target. The default `--spec-type auto` picks the eagle route; a Hugging Face target directory works the same way, since it is read for its `config.json` and tokenizer only.

**Concat-fc drafters.** vLLM's further drafters fuse `fc([embedding | target hidden])` without EAGLE-3's encoder norms and run the family's decoder layer: `EagleMistralLarge3ForCausalLM`, `EagleLlama4ForCausalLM`, `EagleMiniCPMForCausalLM` and `EagleCohereForCausalLM` convert from their Hugging Face directories and load as the `eagle-mistral-large3`, `eagle-llama4`, `eagle-minicpm` and `eagle-cohere` GGUF architectures. Each reads the target's final layer, applies the target's own output norm to the tapped hidden state (vLLM feeds the post-norm `hidden_states`), shares the target's tokenizer and vocabulary, and drafts under `--spec-type auto` with lossless verification. The families differ in what their files describe: Mistral Large 3's MLA attention with its `llama_4_scaling` query temperature; Llama 4's chunked local attention, `no_rope_layers` RoPE pattern and weightless per-head Q/K norm (with the 8192-token NoPE temperature); MiniCPM's `scale_emb`, `scale_depth` / `mup_denominator` residual, `input_norm1`/`input_norm2` fusion norms and `dim_model_base / hidden_size` output scale; Cohere's parallel residual, single LayerNorm-or-RMS attention norm, LayerNorm final norm and `logit_scale`. A checkpoint with a reduced draft vocabulary (no `d2t`), a Llama 4 yoco or mixture-of-experts layer, a MiniCPM expert or non-silu feed-forward, or a target output norm in groups is refused by name.

**Drafter settings.** The n-gram sizes default to llama.cpp's.

| Property | CLI flag | Default |
|----------|----------|---------|
| `neuralfree.draft.ngram.simple.size.n` / `.size.m` | `--spec-ngram-simple-size-n` (alias `--ngram-prompt-lookup-max`) / `-size-m` | 12 / 48 |
| `neuralfree.draft.ngram.simple.size.n.min` | `--spec-ngram-simple-size-n-min` (alias `--ngram-prompt-lookup-min`) | `size.n` |
| `neuralfree.draft.ngram.map.k.size.n` / `.size.m` / `.min.hits` | `--spec-ngram-map-k-size-n` / `-size-m` / `-min-hits` | 12 / 48 / 1 |
| `neuralfree.draft.ngram.map.k4v.size.n` / `.size.m` / `.min.hits` | `--spec-ngram-map-k4v-size-n` / `-size-m` / `-min-hits` | 12 / 48 / 1 |
| `neuralfree.draft.ngram.mod.n.match` / `.n.max` / `.n.min` | `--spec-ngram-mod-n-match` / `-n-max` / `-n-min` | 24 / 64 / 48 |
| `neuralfree.draft.lookup.cache.static` | `-lcs`, `--lookup-cache-static` | none |
| `neuralfree.draft.lookup.cache.dynamic` | `-lcd`, `--lookup-cache-dynamic` | none |
| `neuralfree.draft.suffix.max.tree.depth` | `--spec-suffix-max-tree-depth` | 64 |
| `neuralfree.draft.suffix.max.spec.factor` | `--spec-suffix-max-spec-factor` | 1.0 |
| `neuralfree.draft.suffix.max.spec.offset` | `--spec-suffix-max-spec-offset` | 0.0 |
| `neuralfree.draft.suffix.min.token.prob` | `--spec-suffix-min-token-prob` | 0.1 |
| `neuralfree.draft.suffix.max.cached.requests` | `--spec-suffix-max-cached-requests` | 10000 (0 turns the request tree off) |
| `neuralfree.draft.gpu.devices` | `--spec-draft-device`, `-devd`, `--device-draft` | unset: the draft follows the target's devices |
| `neuralfree.draft.n.gpu.layers` | `--spec-draft-ngl`, `-ngld`, `--gpu-layers-draft`, `--n-gpu-layers-draft` | `auto` |
| `neuralfree.draft.override.tensor` | `--spec-draft-override-tensor`, `-otd`, `--override-tensor-draft` | none (replaces the target's for the draft load) |
| `neuralfree.draft.cpu.moe` | `--spec-draft-cpu-moe`, `-cmoed`, `--cpu-moe-draft` | off |
| `neuralfree.draft.n.cpu.moe` | `--spec-draft-n-cpu-moe`, `-ncmoed`, `--spec-draft-ncmoe`, `--n-cpu-moe-draft` | 0 |

Size ranges:
- n-gram sizes are 1..64. A single `ngram-simple` size needs at least 2.
- m-gram sizes are 1..128.
- `ngram-simple`'s `size.n.min` must not exceed `size.n`. The other types have no `size.n.min`.

`neuralfree.draft.max` (default 3) caps every drafter, so a lookup type drafts at most 3 tokens unless you raise it.
llama.cpp lets its n-gram types draft their full m-gram; for that, set `--spec-draft-n-max` to at least the m-gram size.

**Prompt lookup.** `ngram-simple` looks up one n-gram size or a range of sizes:
- With one size (`size.n.min` unset or equal to `size.n`), it is llama.cpp's `ngram-simple`. The pattern is the last
  `size.n` tokens, the sampled token included. The drafter searches backwards for the pattern's most recent earlier
  occurrence and drafts the `size.m` tokens after it, as long as at least `size.n` of them are present.
- With a range (`size.n.min < size.n`), it is vLLM's `ngram` method (`prompt_lookup_min` / `prompt_lookup_max`).
  The drafter takes the longest suffix of the context, between `size.n.min` and `size.n` tokens, that occurs earlier.
  It uses that suffix's earliest occurrence and drafts up to `size.m` tokens after it, ending at the end of the context.

The range drafts where one large size finds nothing: `--spec-ngram-simple-size-n 12` alone needs a 12-token repeat,
while `--ngram-prompt-lookup-min 2 --ngram-prompt-lookup-max 12` falls back to shorter matches. `ngram-mod` keeps a
chain that reaches `n.max` tokens even when `n.max` is below `n.min`, as llama.cpp does. A chain that breaks off
early is dropped when it is shorter than `n.min`.

`--spec-ngram-simple-min-hits` is refused, because llama.cpp reads min hits for the map types only.

Cache files:
- They use llama.cpp's `common_ngram_cache_save` layout, so llama.cpp's files can be read and written.
- They need `ngram-cache` in the type list.
- The static file must exist.
- The dynamic file is read if it exists. When the session closes, it is rewritten with that run's n-grams added.
- `nftool lookup-create`, `lookup-merge` and `lookup-stats` build, merge and measure the files ([nftool lookup tools](#nftool-lookup-create-lookup-merge-and-lookup-stats)).

**Suffix decoding** (`--spec-type suffix`, vLLM's and Arctic Inference's suffix decoding with tree speculation off). Each round the drafter matches every suffix of the recent context (the last `maxTreeDepth` tokens plus the sampled token) against two trees: one built from the generation's own tokens and one from the shared n-gram corpus, whose sequences are searched in full. A match drafts the chain of its most frequent continuations. Each token's probability is the running product of the continuation frequencies, and the chain stops once that falls below `minTokenProb`. A match of `L` tokens drafts at most `floor(L * maxSpecFactor + maxSpecOffset)` tokens. The draft with the highest summed probability wins, a tie goes to the longer match, and equal continuation counts pick the lower token id.

Beside the generation's own tree and the corpus tree, the drafter keeps a request tree of past responses, as Arctic
Inference's `SuffixDecodingCache` does (vLLM `suffix_decoding_max_cached_requests`):
- The tree holds each generation's response: the sampled tokens, the accepted draft tokens and the tokens committed
  between drafts. Prompts are not cached; they live in the generation's own tree.
- A new generation starts a new request. Once more than `neuralfree.draft.suffix.max.cached.requests` requests are
  cached, the oldest are evicted first, never the request just started.
- The draft with the highest score among the three trees wins. A tie keeps the generation's own tree.
- `0` turns the request tree off.

**Drafting over a reused prompt prefix.** With prompt-prefix reuse on (the default), a second generation that shares a prompt prefix with the
previous one keeps the target's stored rows and evaluates only the new tail. The drafter is brought
to that prefix before the tail is evaluated, so drafting continues from the same state a full prefill
builds:

- EAGLE-1, EAGLE-3 and the target's NextN/MTP heads keep their own decoder rows for the reused
  positions and re-evaluate the last stored position, which supplies the hidden row the first new row
  pairs with.
- A dflash drafter keeps its cached draft rows for the reused positions and drops the rows after them.
- gemma4-assistant keeps no cache of its own, reads the target's rows, and starts each generation
  clean.
- A separate draft model (`neuralfree.draft`) prefills the shared prefix into its own cache; the
  lookup and suffix drafters read the stored tokens directly.

If a drafter's memory cannot drop the rows after the shared prefix, the target evaluates the whole
prefix again through the drafter, so the second generation's greedy tokens and its accepted drafts
stay exactly those of a fresh session over the same prompt. No drafting is skipped and no error is
raised.

**Verification.** How drafted tokens are verified:

| Property | CLI flag | Values |
|----------|----------|--------|
| `neuralfree.draft.verify` | `--spec-verify`, `--rejection-sample-method` | `match` (default), `rejection`, `block`, `fly` (lossy), `synthetic` |
| (same) | `--spec-draft-sampling` | `greedy` = `match`, `probabilistic` = `rejection` |
| `neuralfree.draft.fly.window` | `--spec-fly-window` | 0, meaning `min(6, k-1)` |
| `neuralfree.draft.fly.entropy.threshold` | `--spec-fly-entropy-threshold` | 0.3 |
| `neuralfree.draft.fly.entropy.top.k` | `--spec-fly-entropy-top-k` | 3 |
| `neuralfree.draft.synth.len` | `--spec-synth-len` | mean accepted length, at least 1 |
| `neuralfree.draft.synth.rates` | `--spec-synth-rates` | per-position acceptance, non-increasing, e.g. `0.9,0.6,0.3` |

Synthetic acceptance:
- A synthetic length or rate list selects `synthetic` verification when no mode is set. Combining it with another
  mode is refused.
- It is for benchmarking and also needs `-Dneuralfree.spec.synthetic=true`.

**Device-resident speculation.** On a GPU target a round stays on the device, as llama.cpp's backend sampling
(`-bs`, `--backend-sampling`; `--spec-draft-backend-sampling` for the draft alone) and Avalon's verification argmax
keep it:

- *Verification.* When a round's `match` verification needs only the target's selection (no other verifier, no
  logprobs requested), the verified rows are not read back. A greedy chain selects every row's argmax on the device;
  a stochastic chain samples every row over the device's top-k candidates, with temperature, top-k, top-p, min-p and
  the seeded draw, drawing the same tokens the host chain draws. A chain whose stages can keep a token outside those
  candidates (penalties, DRY, a positive `logit_bias`, a grammar or another masking or forcing stage, typical
  sampling, or a whole-row stage before the chain's top-k) reads the rows back, and the log names the reason. The
  `rejection`, `block`, `fly` and `synthetic` verifiers read the rows back. The tokens are the same either way.
- *Drafting.* A `dflash` draft (DFlash, DFlash2, DSpark, LiLiCorr) of a GPU target drafts on the target's device: its encoder,
  decoder stack and KV cache live there beside the target, and each readout reads back only what it uses (a plain
  DFlash draft its drafted ids, plus the top-10 candidates and their softmax denominators under `p_min`; a DFlash2
  draft its selector's chosen ids, plus their probabilities under `p_min`, with the dynamic convolution and the
  selector lattice on the device; a DSpark
  draft its argmax chain and the `t_embd` rows its confidence head reads; a LiLiCorr draft its top candidates,
  denominators and `t_embd` rows). The target's feature rows reach the drafter without a host round trip when the
  target streams them on the drafter's device. A quantized draft cache (`-ctkd` / `-ctvd` `q8_0` and the other block
  types) stores on the device with the Hadamard rotation. The drafter falls back to the host, logging the reason,
  for the DeepSeek-V4 DSpark backbone, a quantized draft cache whose head size is not a multiple of 32, a
  non-ascending `d2t` (the DFlash2 selector then reads out on the host), a hyper-connection tap,
  `neuralfree.split.mode=layer`, a hybrid target, or a draft tensor type or kernel the device does not take. The
  context fit counts the device drafter's matrices, cache, tables and readout buffers in the target's device budget.

| Property | CLI flag | Default |
|----------|----------|---------|
| `neuralfree.draft.verify.device` | `-bs, --backend-sampling`, `--no-backend-sampling` | `true`; `false` reads the verified rows back and selects on the host |
| `neuralfree.draft.dflash.device` | `-bs`, `--no-backend-sampling`, or `--[no-]spec-draft-backend-sampling` for the draft alone | `true`; `false` drafts a dflash draft on the host |
| `neuralfree.sampling.device` | `-bs, --backend-sampling`, `--no-backend-sampling` | `true`; `false` samples every generation step on the host |

`-bs` / `--backend-sampling` and `--no-backend-sampling` set all three properties at once and take no value, as in
llama.cpp; `--[no-]spec-draft-backend-sampling` sets the draft row alone. `neuralfree-server` takes the same
switches.

**On-device sampling.** With `neuralfree.sampling.device`, a GPU generation step whose sampler chain selects inside a top-k cone does not read the logits row back: the device computes the row's best candidates (`LLM_Row_TopK`, at most 64) and the host chain selects over them, which is exactly the token it selects over the whole row (llama.cpp's backend sampling). The single-sequence loop and the batch engine use the same cones; speculative verification keeps its own. A chain fits in three shapes:

- a truncated chain whose top-k stage runs before every whole-row stage (top-p, min-p, typical, top-n-sigma, XTC); banned tokens and non-positive `logit_bias` entries widen the cone by one slot each;
- a greedy chain (`top_k` 0, or a top-k of 1) whose only edits are bans (`tokenizer.ggml.suppress_tokens`, the end-of-generation tokens under ignore-eos, `-100` entries) and non-positive `logit_bias`: the winner lies among the largest *n + 1* raw logits for *n* edited tokens (Avalon's `greedy_argmax`), so banning the top token costs one extra candidate instead of a whole-row readback;
- a truncated chain whose pre-top-k stages edit the row (repetition, presence and frequency penalties, DRY, banned tokens and any `logit_bias`, positive included): `LLM_LogitsPrefilter` (Vulkan `penalties.comp`) applies those edits to the whole row on the device in the chain's own order, with llama.cpp's semantics (`-inf` for a ban or a bias of `-100` and below, `l = (l <= 0 ? l * repeat : l / repeat) - (count * frequency + presence)` over each sequence's window, DRY's restart-sequence matching with `multiplier * base^(repeatLength - allowed)`), the readout returns `max(top_k, min_keep)` candidates, and the host chain skips the stages the device ran. Each sequence uploads its own window and counts per step, as llama.cpp's backend penalties sampler feeds its graph. A penalty window over 16384 distinct tokens, a DRY window over 256 tokens, or a DRY breaker set over 8192 restart sequences or 16384 tail ids keeps the host path.

The step reads whole rows back for a greedy chain with penalties, DRY or a positive bias, mirostat, adaptive-p, infill, a whole-row stage or a dynamic temperature range ahead of the chain's top-k, a grammar, `allowed_token_ids`, bad words, the min-tokens ban or a reasoning budget (llama.cpp disables backend sampling under a grammar or reasoning budget too), a logprobs request, and a cone wider than 64. Speculative verification reads its verified rows back when the chain carries penalties, DRY or a positive bias. In the batch engine one such row, a pooling request, a media chunk that completes a prompt or a multi-row speculative verification makes the whole step read back. `BatchStats.deviceSampledRows()` and the Prometheus counter `neuralfree_tokens_predicted_device_total` count the tokens selected over a device readback. The generated tokens are identical on either path, for the same seed.

**vLLM speculative config.** `--speculative-config '<json>'` takes vLLM's `SpeculativeConfig` object on the CLI
and on the server. Each key is applied as the flag below, so it sets the same properties with the same checks, and a
later flag overrides a key. A JSON `null` counts as an absent key, and absent keys keep the configuration.

| Key | Flag |
|-----|------|
| `method` | `--spec-type`: `ngram` (also `[ngram]`) → `ngram-simple`; `suffix` → `suffix`; `draft_model` → `draft-simple`; `mtp` and vLLM's model-specific MTP names (`deepseek_mtp`, `qwen3_next_mtp`, …) → `draft-mtp`; `eagle`, `eagle3`, `dflash` → `auto` (the draft file decides the route) |
| `model` | `--spec-draft-model`. Needed by `draft_model`, `eagle`, `eagle3` and `dflash`, and refused beside `ngram` and `suffix`. Without a method, `"ngram"` selects `ngram` and any other model selects `auto` |
| `draft_lm_head_quantization` | `--draft-head-prec`: `fp8` or `nvfp4` for the [reduced draft head](#reduced-draft-head-dflash-drafters) |
| `num_speculative_tokens` | `--spec-draft-n-max`; for `suffix` it defaults to the tree depth, as in vLLM |
| `prompt_lookup_max` / `prompt_lookup_min` | `--spec-ngram-simple-size-n` / `--spec-ngram-simple-size-n-min`; one of them alone sets both, as vLLM resolves them |
| `suffix_decoding_max_tree_depth`, `suffix_decoding_max_cached_requests`, `suffix_decoding_max_spec_factor`, `suffix_decoding_min_token_prob` | `--spec-suffix-max-tree-depth`, `-max-cached-requests`, `-max-spec-factor`, `-min-token-prob` |
| `rejection_sample_method` | `--rejection-sample-method` (`standard` = `rejection`) |
| `use_heterogeneous_vocab` | `--spec-draft-heterogeneous-vocab` ([below](#speculative-decoding)): a `draft_model` draft with a different tokenizer drafts over the vocabularies' intersection |
| `synthetic_acceptance_rates` / `synthetic_acceptance_length` | `--spec-synth-rates` / `--spec-synth-len` |
| `fly_window_size`, `fly_entropy_threshold`, `fly_entropy_top_k` | `--spec-fly-window`, `--spec-fly-entropy-threshold`, `--spec-fly-entropy-top-k` |

Example: `--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'`.

Refusals name the key:
- `draft_tensor_parallel_size`, because the draft model runs in the target's engine without tensor parallelism.
- `tensor_parallel_size`, which vLLM refuses too.
- The methods `ngram_gpu`, `custom_class`, `dspark` and `extract_hidden_states`.
- Every other key. Its refusal lists the supported keys.

A value the flag refuses is reported as `--speculative-config key=value: <reason>`. On the server, keys whose flag the
server refuses (the MTP route `draft-mtp`) are refused the same way.

**Heterogeneous vocabularies.** A draft model does not have to share the target's vocabulary. Enable token-level intersection drafting when the draft tokenizer differs from the target's:

```bash
neuralfree --model target.gguf --draft small-draft.gguf \
  --spec-type draft-simple --draft-max 4 --spec-draft-heterogeneous-vocab
```

The server takes the same switch, and vLLM's `--speculative-config` key maps to it:

```bash
neuralfree serve --model target.gguf --model-draft small-draft.gguf \
  --spec-draft-heterogeneous-vocab

# equivalent JSON form (vLLM spelling)
--speculative-config '{"method": "draft_model", "model": "small-draft.gguf",
                       "use_heterogeneous_vocab": true}'
```

How it works: the engine intersects the two tokenizers by token text (`Ġ` and `▁` word-initial space markers are normalized to a space), feeds the target's tokens to the draft model through that map (tokens outside the intersection become the draft's unknown token), masks the draft logits so it can only propose tokens the target also has, and translates every proposal back to the target's token id. Greedy verification then keeps the run lossless: under greedy decoding the output is exactly the target-only output.

Notes and refusals:
- Greedy drafting only, as in vLLM. Combining it with `--spec-draft-sample-method probabilistic` is refused by name.
- It applies to a separate draft model (`--spec-type draft-simple` / `auto` with a plain draft LM); an EAGLE/DFlash/MTP draft file is refused at load.
- A draft model without an unknown or end-of-sequence token is refused: tokens outside the intersection need a fallback.
- The launch property is `neuralfree.draft.heterogeneous.vocab=true` (`false` by default). Without it, the previous llama.cpp compatibility rules still apply and mismatched vocabularies are rejected with the differing token named.

Every draft context (a separate draft model, an eagle3 drafter, the dflash drafter's host cache, and the NextN/MTP caches of `--mtp` or an MTP draft file) keeps its KV rows in `f16` and steps with the target's `--ubatch` unless the draft-context settings say otherwise. Quantized draft rows follow the main cache's rules: whole 32-value blocks, Hadamard rotation only for the block types. At long context the draft cache is real memory: `-ubd 256 -ctkd q8_0 -ctvd q8_0` is how a 256k-token context with MTP fits a budget, the same knobs llama.cpp and Avalon expose. An MLA draft context needs equal K and V types, and an invalid type or a negative ubatch is refused naming the flag or property.

The batch engine behind the [HTTP server](#http-server) runs every drafter that does not read the target's hidden states, with drafter state per running sequence (vLLM's batched proposers, llama.cpp server's per-slot speculative state):

- **Lookup types.** `ngram-simple` (`prompt-lookup`), `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` and `suffix` each keep one state per sequence over that sequence's own text tokens. A finished sequence's state is dropped before its slot serves the next one; the n-grams it learned are kept for the dynamic cache. `neuralfree.draft.lookup.cache.static` is read once and shared; `neuralfree.draft.lookup.cache.dynamic` is written when the engine closes, with every sequence's n-grams added.
- **Draft model.** `neuralfree.draft` (with `neuralfree.draft.type` `auto` or `draft-simple`) keeps its draft KV in paged memory of its own, one draft sequence per slot (`neuralfree.draft.kv.typeK/typeV`, `neuralfree.draft.ubatch`). Each step drafts every decoding sequence in the same forward batches, stopping below `neuralfree.draft.p.min`. With `neuralfree.draft.sample.method=probabilistic` (vLLM's `draft_sample_method`), a sampling request's draft tokens are drawn at the request's temperature and verified with their draft probabilities; the default `greedy` drafts the argmax. Probabilistic drafting needs a draft model and runs in the batch engine only.
- **Chains.** Several types chain in llama.cpp's order (the lookups first, the draft model last); the first non-empty proposal wins.
- **Verification.** Proposals are drafted once per step and verified inside the same batched step, next to the other sequences' decodes and prefill chunks. Every `neuralfree.draft.verify` mode applies per sequence, under that request's own sampling parameters, grammar and penalties: `match`, `rejection` (vLLM's `RejectionSampler`), `block`, `fly` and `synthetic` (with `-Dneuralfree.spec.synthetic=true`). `match`, `rejection` and `block` are lossless.
- **Refusals.** A request whose chain ends in adaptive-p is refused at submit beside a mode other than `match` ([Constraints](#speculative-decoding)); mirostat runs in every mode. The batch engine refuses by name the drafters that read the target's hidden states of one sequence: `neuralfree.mtp`, `draft-mtp`, and EAGLE-3, gemma4-assistant, DFlash and MTP draft files. These run on the single-sequence path.

The server takes:
- `--spec-type` (repeatable, appends): the lookup types, `draft-simple` and `auto`;
- `-md` / `--model-draft` / `--spec-draft-model`;
- `-ctkd` / `-ctvd` (also `--cache-type-k/v-draft`, `--spec-draft-type-k/v`);
- `-ubd` (also `--spec-draft-n-ubatch`, `--spec-draft-ubatch-size`);
- `-ngld` / `-devd` / `-otd` / `-ncmoed` (also `--spec-draft-ngl`, `--gpu-layers-draft`, `--n-gpu-layers-draft`,
  `--spec-draft-device`, `--device-draft`, `--spec-draft-override-tensor`, `--override-tensor-draft`,
  `--spec-draft-n-cpu-moe`, `--spec-draft-ncmoe`, `--n-cpu-moe-draft`): the placement of the separate draft
  model's own load (a single draft device, a host `none` draft);
- `-cmoed` (also `--spec-draft-cpu-moe`, `--cpu-moe-draft`): the draft's MoE experts on the host;
- `-td` / `-tbd` (also `--spec-draft-threads`, `--threads-draft`, `--spec-draft-threads-batch`,
  `--threads-batch-draft`): size `neuralfree.cpu.threads`, the one pool the target's and the draft's host tensors
  share, so both must agree;
- `--spec-draft-p-min` (`--draft-p-min`);
- `--spec-draft-n-max` (also `--draft-max`, `--draft-n-max`) and `--spec-draft-n-min` (also `--draft-min`, `--draft-n-min`);
- `--spec-ngram-simple-size-n` (also `--ngram-prompt-lookup-max`, 2..64), `--spec-ngram-simple-size-n-min` (also `--ngram-prompt-lookup-min`, 1..64; below `-size-n` each sequence drafts with vLLM's range search) and `--spec-ngram-simple-size-m` (1..128);
- `--spec-ngram-map-k[4v]-size-n/-size-m/-min-hits` and `--spec-ngram-mod-n-match/-n-max/-n-min`;
- `-lcs` / `-lcd`;
- `--spec-suffix-max-tree-depth`, `--spec-suffix-max-spec-factor`, `--spec-suffix-max-spec-offset`, `--spec-suffix-min-token-prob`;
- `--spec-verify` (`--rejection-sample-method`; `--spec-draft-sampling greedy|probabilistic` = `match|rejection`);
- `--spec-fly-window`, `--spec-fly-entropy-threshold`, `--spec-fly-entropy-top-k`;
- `--spec-synth-len` / `--spec-synth-rates`;
- `--spec-draft-sample-method greedy|probabilistic`;
- `--speculative-config` with the keys of those flags.

Every value is checked at launch naming its flag; contradicting n-gram settings are refused once the command line is read. `--spec-type draft-mtp` and the draft download flags (`-hfd`, `-hfrd` and their long spellings) are refused at launch naming the flag: the server's draft model (`-md`) loads from a local file, so a Hugging Face draft download runs in the single-sequence CLI. `--spec-ngram-simple-min-hits` is refused as llama.cpp's map-only setting.

In code, `BatchEngine.forModel(model, config, drafter, spec)` takes any `BatchDrafter`, built with `BatchSpec.of(draftMax, draftMin, options)`. `BatchStats` counts `specRounds()`, `specDrafted()` and `specAccepted()`.

A Hugging Face `MiMoForCausalLM` checkpoint keeps its `model.mtp_layers.*` weights when it is converted or loaded (`--hf-no-mtp` drops them). The result is a `qwen2` model whose `nextn_predict_layers` counts the MTP layers: MTP layer `i` becomes the appended block `n_hidden_layers + i`, and its `token_layernorm`, `hidden_layernorm`, `input_proj` and `final_layernorm` become that block's `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj` and `nextn.shared_head_norm`, with `input_proj` stored in `[embed | hidden]` order. `--mtp` then drafts one token per step with the head, reading the model's own token table and LM head; a checkpoint that declares more than one MTP layer is refused at engine setup, naming the block count. `MiMoMTPModel`, vLLM's name for the same checkpoint served as its own MTP draft, converts the same way; the GGUF drafts through those blocks with `--mtp` or as an MTP draft file (`--draft`). vLLM's rewritten draft names `Qwen3OmniDSparkModel` (for `Qwen3DSparkModel`) and `DFlashMuseGlimmerAssistantModel` (for `MuseGlimmerAssistantModel`) convert through the same rows as the names they alias.

The converter folds the NextN/MTP head the same way for EXAONE 4.5, EXAONE-MoE, BailingMoe V2.5, Dots3-Note and Nemotron-H (dense and MoE) checkpoints: the head's layers become appended blocks counted by `nextn_predict_layers`, its shared projection and norms go into every appended block, and a shared-layer head (`mtp_share_layers`) is replicated into each. The file then drafts with `--mtp`. A head with several NextN blocks drafts up to `--draft-max` tokens per step, one block after another, rather than one token per block. Nemotron-H takes at most one NextN layer and refuses more by name.

An MTP draft file of a qwen4exp target may hold only the NextN/MTP block, as Unsloth's and Avalon's MTP sidecars do. A block whose `attention.compress_ratios` entry is 0 drafts with the last trunk layer's compress ratio, so its attention runs over the QSA indexer like the trunk's. A shared sidecar without `token_embd.weight` or `output.weight` drafts with the target's token embeddings and LM head (llama.cpp's `llama_model_share_mtp_io`); a sidecar that carries its own keeps them.

An MTP draft file may carry its own LM head cut to its most frequent tokens (FR-Spec), with a `d2t` table naming the target token of every head row, as Avalon's `mtp-Qwen3.8-Flash-Next-*-FR65k.gguf` does (its head covers 65k of the target's tokens). `draft-mtp` detects the cut head from the file: drafting picks over the reduced rows and proposes the mapped target token, so nothing changes on the target side. The file is refused, naming the field, when `d2t` does not have one entry per head row, when an entry is not a target token id, or when the head is smaller than the target vocabulary without a `d2t` table. Only draft files are cut; the target's own NextN head (`--mtp`) never is.

**Shared embeddings and LM head (qwen4exp).** An MTP-only qwen4exp draft file need not carry `token_embd.weight` or its own LM head: a missing one is read from the target, as llama.cpp's `llama_model_share_mtp_io` does, and a file that ships its own keeps it. The file must still be an MTP draft of the target (same architecture, trunk depth and schedule, `n_embd` and vocabulary). A reduced (FR-Spec) head is used with its `d2t` table, so a file carrying `d2t` without its own `output.weight` is refused, naming the tensor.

**MTP drafter cost.** The NextN/MTP drafter reads the target's hidden rows where the target runs, as llama.cpp's device feed does. On a GPU load the rows stay on the device with one device or a layer split: a forward call copies its rows device-side (one copy per ubatch), captured on the last trunk layer's device and received by the head's device — a device-to-device copy when they share a device, a peer copy when the backend supports one, and a staged host copy when it does not. The drafter builds its pairs, applies the target's output norm and keeps its pending row there, so a prompt never moves its rows to the host. `-Dneuralfree.draft.mtp.device.rows=false` (llama.cpp's `LLAMA_MTP_H_DEVICE=0`) restores the host feed; the CPU and hybrid backends always use it.

A forward call's own rows never feed a draft head output, so its prompt batches run only the NextN block's cache writes: the K/V projections with their norms, RoPE, indexer keys and pool writes, rotations and stores, for the softmax-attention, MLA, QSA/MSA and DeepSeek-V4 mixers alike. The query path, attention, output projection, FFN and head do not run, the cache rows come out bit-identical, and a prompt ubatch costs measurably fewer kernels. A small call (a verification, a decode fallback) does not decode into the draft context right away. The drafter keeps its pairs, drops the ones a rejection discards, and the next draft decodes them in its first pass, one run of the block per drafted token instead of two. `NextnDraftModel.toString()` names the feed it bound (`device rows` on the device feed).

A forward call's own rows never feed a draft head output, so its prompt batches run only the NextN block's attention part: the draft caches are written, the FFN and the head are skipped, and the cache rows come out bit-identical. A small call (a verification, a decode fallback) does not decode into the draft context right away. The drafter keeps its pairs, drops the ones a rejection discards, and the next draft decodes them in its first pass, one run of the block per drafted token instead of two. `NextnDraftModel.toString()` names the feed it bound (`device rows` on the device feed).

**Constraints.** A grammar constraint (`--grammar`, `--grammar-file`, `-j/--json-schema`, `-jf/--json-schema-file`, or the server's
`grammar`/`json_schema`/`guided_*` fields) composes with speculative decoding (`--draft`, `--mtp`, n-gram drafting,
or the server's batched drafters) in every verification mode, as vLLM's structured output does with spec
decode:

- Each proposal is cut before its first token the grammar cannot accept after the tokens before it (vLLM's
  `validate_tokens`). Only the accepted prefix is verified; a proposal cut below `--spec-draft-n-min` decodes one
  token plainly. The batch engine behind the HTTP server cuts its proposals the same way, per sequence, so a
  constrained request verifies no row its grammar could never confirm; its `neuralfree.spec.drafted.tokens` counter
  counts the verified tokens.
- Every verify row is scored under the grammar, penalty and request-stage state its draft prefix leaves. A rejected
  draft token is rolled back out of that state, so the block and FLy verifiers, which read all rows before
  confirming, also emit only grammar-valid text.
- The logit bias, `min_tokens`, bad words, allowed tokens and the reasoning budget apply on the speculative path
  exactly as on the plain one. A greedy constrained generation gives the same tokens with or without a drafter.
- Penalties and DRY see the draft tokens too, so the whole request chain scores every verify row.
- Logprobs (`logprobs`, `top_logprobs`, `logprob_token_ids`) cover every emitted token in every verify mode:
  confirmed draft tokens, rejection replacements and bonus tokens, each from the target's row at its position,
  scored under the request's logprobs mode.
- Mirostat runs in every verification mode. Its `mu` adapts on each accepted token's surprise in the candidate set
  of the row it came from, so plain decoding keeps llama.cpp's trajectory. A block or FLy row read ahead of a draft
  token that is then rejected leaves no trace, since the state rolls back with the grammar and the penalty history.
  A token outside mirostat's kept set leaves `mu` unchanged. The synthetic verifier confirms draft tokens without
  scoring their rows, so those tokens do not adapt `mu`.
- Adaptive-p folds only its own draw into its moving average, so it runs only under `match` verification; any other
  mode refuses the request, naming `adaptive_p`.
- Synthetic verification never confirms a draft token the request's chain rules out (a grammar mask, a ban, bad
  words).
- With `neuralfree.context.shift=true`, a speculative generation that fills the context shifts it as plain
  generation does and keeps drafting; without it, the generation stops with `StopReason.CONTEXT_FULL`. A drafter
  that reads the target's hidden rows (EAGLE-1, EAGLE-3, gemma4-assistant, DFlash and the target's NextN/MTP head)
  keeps the rows of the positions the shift kept through `DraftModel.beginGeneration`, and the target re-evaluates
  the moved window through it, feeding the drafter the hidden rows of the shifted positions, as llama.cpp's
  speculative path re-runs its draft context after a shift. A drafter that reads only the target's tokens, or runs
  a model of its own, resets and aligns itself from the target's moved tokens at its next draft.
- A Java-API `SamplerStage` that records accepted tokens must override `saveState` and `restoreState` for the `block`
  and `fly` modes, which feed draft tokens ahead of confirmation; a stage that records nothing needs neither.

#### Lookahead (Jacobi) decoding

Lookahead decoding drafts with the target itself: it keeps a window of Jacobi guesses, refines them in a couple of
parallel iterations, and verifies the n-grams it has already observed against the target in the same forward batch.
It replaces speculative decoding rather than chaining with it, and runs over a paged, unified KV cache whose `-np`
holds `W + G + 1` sequences. The console runs it per generation, and `neuralfree serve` runs one lookahead group per
generation request, so every served generation request uses it.

| Flag | Meaning | Default |
|---|---|---|
| `--lookahead` | Lookahead (Jacobi) decoding with the default sizes; needs `-np 31` | off |
| `--lookahead-window <n>` | Jacobi window `W`, the guesses kept per step (turns decoding on) | 15 |
| `--lookahead-ngram-size <n>` | Observed n-gram size `N`, at least 3 (turns decoding on) | 5 |
| `--lookahead-max-ngrams <n>` | Verification n-grams `G` kept per first token (turns decoding on) | 15 |
| `--no-lookahead` | Turns lookahead decoding off | |

The same settings are properties: `neuralfree.lookahead`, `neuralfree.lookahead.window`,
`neuralfree.lookahead.ngram.size` and `neuralfree.lookahead.max.ngrams`.

```bash
neuralfree model.gguf -np 31 --lookahead -p "The capital of France is"
neuralfree serve -m model.gguf -Dneuralfree.lookahead=true -Dneuralfree.parallel=31 \
  -Dneuralfree.kv.unified=true
```

Requirements and refusals (each names the setting that is wrong):

- The memory must be paged and unified (`neuralfree.parallel = W + G + 1`, `neuralfree.kv.unified = true`). Context
  shift (`neuralfree.context.shift`), recurrent state, compressed caches and sliding-window memories are refused,
  because they cannot rewind the rows of a step.
- Lookahead decoding and speculative decoding exclude each other: `--lookahead` with `--draft`, `--mtp` or a
  `--spec-type` lookup drafter is refused.
- The server refuses, by name, a request with media chunks or prompt-embedding rows, self-extend
  (`grp_attn_n`/`grp_attn_w`), `n_keep`/`n_discard` or per-request LoRA adapters. The console refuses media runs and
  `--prompt-cache` with lookahead.
- Lookahead keeps no reusable prompt prefix: the prefix cache is off and slot save/restore/erase refuse. Every request
  prefills its prompt into its own sequence group.
- `W + G + 1` sequences each need room for the prompt plus the window; a step that does not fit the page pools ends
  the generation with a context-full stop instead of failing. Sizing `-c` for `W + G + 1` sequences is the
  operator's job; small contexts leave too few pages.
- The server publishes `neuralfree_lookahead_rounds_total`, `neuralfree_lookahead_emitted_tokens_total` and
  `neuralfree_lookahead_accepted_tokens_total`.

The Java API is `LookaheadOptions` (settings), `LookaheadDecoder.forModel(model, config)` and
`LookaheadDecoder.generate(prompt, request, callback)`; `LookaheadDecoder.lastStats()` reports the last generation's
steps and accepted tokens, and `BatchStats.lookaheadRounds()/lookaheadEmitted()/lookaheadAccepted()` report a server's.
`EngineConfig.builder().lookaheadOptions(...)` enables it from code.

#### Multi-branch (tree) speculative drafting

With a separate draft model over a paged, unified KV cache of `-np` sequences, the draft model drafts a tree instead
of one chain: each draft sequence evaluates one branch per depth, and a branch splits into a new sequence when a
secondary draft candidate's probability exceeds `--spec-draft-p-split`. The target verifies the tree by sampling once
per depth and accepting the drafted tokens its samples equal (llama.cpp's
`common_sampler_sample_and_accept_n`), so greedy and sampled output equal the target-only loop token for token.

| Flag | Meaning | Default |
|---|---|---|
| `--spec-draft-p-split <p>` (alias `--draft-p-split`) | Probability above which a secondary candidate splits a branch | 0.1 |

The property is `neuralfree.draft.p.split`. A value of 1 keeps every round linear.

```bash
neuralfree model.gguf --draft draft.gguf -np 4 -p "The capital of France is"
```

Requirements and refusals (each names the setting that is wrong):

- A separate draft model (`--draft`, `--spec-type draft-simple`) and more than one sequence
  (`neuralfree.parallel > 1`; that turns the memory paged), with `neuralfree.kv.unified=true` (llama.cpp requires
  `--kv-unified` with `--parallel > 1`). The draft model's context is opened with the target's sequence count.
- `--spec-draft-p-split 1` drafts one chain per round; the draft model's top-10 candidate probabilities decide splits
  otherwise.
- Refused by name: a verify mode other than `match`, per-request adapters, `logprobs` / `logprob_token_ids`,
  compressed caches, sliding-window memory and the context shift.
- The target verifies the whole tree in one masked forward over sequence 0 at the fixed width of
  `--spec-draft-tree-nodes` ([single-pass draft-tree verification](#single-pass-draft-tree-verification)), and the
  accepted branch's KV rows move onto their positions.

#### Reduced draft head (dflash drafters)

A dflash drafter reads out its candidates through a head as wide as the vocabulary. `--draft-head-vocab <n>`
(`neuralfree.draft.head.vocab`) gives it a derived head of `n` rows instead: the rows of the head the drafter would
use (its own `output.weight`, else the target's head, else the tied `token_embd.weight`), mapped back to token ids by
a synthesized `d2t` table. The target still verifies with its own full head, so the generated tokens are the same;
only the drafter's proposals can change.

- The kept rows are the draft's special ids (unknown, BOS, EOS, EOT, EOM, SEP, PAD, CLS, mask, the FIM tokens) and
  every `neuralfree.tokenizer.extra_eog_ids` entry, then `--draft-head-keep <id,id,..>`
  (`neuralfree.draft.head.keep`), then the ranking of `--draft-head-freq <file>` (`neuralfree.draft.head.freq`:
  JSON `{"ranked_ids": [..]}`, a bare JSON array, or `<id> <count>` lines, most frequent first), then ascending ids.
  The table is stored ascending, so the device's greedy readout keeps its fast path.
- `--draft-head-prec <t>` (`neuralfree.draft.head.prec`, vLLM's `draft_lm_head_quantization`) stores the kept rows as
  `fp8` (E4M3 with per-row block scales), `nvfp4` (E2M1 codes with an E4M3 scale per 16 values and one F32 global
  scale for the tensor, as vLLM's NVFP4 quantization) or `q8_0`; without it they are copied in the head's own type.
  A borrowed head folds its output scale into the kept rows.
- `--draft-prec <t>` (`neuralfree.draft.prec`) stores the drafter's own projection matrices (encoder, per-block
  attention and FFN, DFlash2 convolutions, selector hidden layer) as `fp8`, `nvfp4` or `q8_0`; tensors already
  quantized stay as they are.
- DSpark's Markov table gathers its candidate columns at the kept ids. The memory fit plans the derived tensors from
  metadata.

Refused by name: a size that removes nothing or cannot hold the mandatory ids, ranking entries outside the
vocabulary or malformed, a draft that ships its own `d2t`, the DeepSeek-V4 DSpark draft, a borrowed head without the
target, a borrowed head with a PRISM input Hadamard, `fp8` rows that are not whole 32- or 128-value blocks, and `nvfp4`
rows that are not whole 64-value blocks (the DFlash2 convolutions and DSpark's Markov tables under `--draft-prec`).

#### Single-pass draft-tree verification

With `--spec-draft-tree` (`neuralfree.draft.tree=true`; `--no-spec-draft-tree` turns it off), a `dflash` drafter
(DFlash, DFlash2 or DSpark) proposes a draft tree instead of one chain. One draft-block forward ranks every slot's top
candidates, and the drafted nodes are chosen best-first by path score: the DFlash2 selector lattice's
path-conditional log-softmax, DSpark's Markov-biased top-k under the conditioning token, or the plain row top-k for
DFlash. Siblings carry distinct tokens. `--spec-draft-tree-temp` / `--spec-tree-temp` (`neuralfree.draft.tree.temp`,
default 1.0) scales the path scores: below 1 the tree concentrates on the best chain, above 1 it spreads; the ranking
stays deterministic.

The target verifies the tree in one forward over its own sequence: the pending token and the tree's nodes are the
step's rows, node `i` stored at KV slot `seq_len + i` and roped at `seq_len + depth`, and the attention runs under a
tree mask, so each node sees the committed prefix, its ancestors and itself. The mask runs in the CPU attention and in
the flash prefill, decode and tensor-core kernels on OpenCL, CUDA, HIP, Vulkan and Metal. The verify width is fixed:
`--spec-draft-tree-nodes` / `--spec-tree-nodes` (`neuralfree.draft.tree.nodes`, default 16, at most 63, and at most
`ubatch - 1`) rows of nodes every round, a shorter tree padded. The round accepts along the tree with one sampler draw
per emitted token, as the chain rounds do, and keeps the longest accepted path plus the target's bonus token: the
accepted nodes' KV rows move from their verify slots onto the consecutive positions they were roped at (every cache
type, quantized ones included), the other rows leave, and the drafter takes the accepted rows' hidden states from the
verify forward. One target forward per round. The CPU runs it on contiguous and paged memory; a GPU backend runs it
on paged memory only (`neuralfree.parallel > 1` or `neuralfree.kv.paged=true`) and refuses a contiguous cache by name.

A separate draft model (`--spec-draft-model`) verifies its branching proposals (see
[multi-branch drafting](#multi-branch-tree-speculative-drafting)) through the same single pass at the same width.

M-RoPE targets rope every node at its depth position in each section. Requirements: `neuralfree.draft.verify=match`
(the default) and a `neuralfree.ubatch` of at least 2. Refused by name: compressed or sliding-window memories, the context shift, per-request LoRA, logprobs
capture, and a backend without the batched forward. A request whose sampler constrains its output (a grammar,
`min_tokens`, bad words, allowed tokens) narrows the proposal to one chain, which verifies as the chain round does.
Under greedy verification on CUDA, OpenCL and Vulkan the accept walk runs on the device (`LLM_TreeAccept`): only the accepted
length, the path and the two greedy ids come back instead of the verify rows' logits; sampled verification walks on
the host. With `neuralfree.gpu.chainReplay` on, the fixed-width verify step is captured once and replayed for every
round of a generation, the commit's row moves dispatching outside the capture
([decode-step command replay](#decode-step-command-replay)).

The masked forward carries the full fixed width every round, so the chain stays faster unless the tree lifts the
accepted length. On the RTX 5080 Laptop with a Qwen3 4B Q8_0 target and a 0.6B Q8_0 drafter (48 tokens), plain
decoding runs at 102 tokens/s on CUDA, the chain at 32 (20% of drafted tokens confirmed) and the tree at 22 at
width 16 or 32 (12%); on OpenCL 67, 29 and 14. With a drafter forced to agree, the chain still runs about twice as
fast as the tree on both backends, which is why the tree stays off by default; `TreeVerifyBenchmarkTest` (`-Dneuralfree.bench.tree=1`, `-Dneuralfree.bench.backend=CUDA`
or `OPENCL`) prints plain, chain and tree tokens per second with the proposed and confirmed tokens per round.

A target with recurrent layers (Gated DeltaNet, KDA, Mamba-1, Mamba-2, PLaMo-2, alone or in a hybrid with attention or
MLA) verifies the tree
in the same single forward: each recurrent layer steps one state per tree node, seeded from the node's parent, and
leaves the committed state untouched; the commit then replays the accepted path's captured layer inputs from the
committed state, so the state after a round is the state after exactly the accepted tokens.
`neuralfree.draft.tree.recurrent=false` (default `true`) refuses this. Refused by name over recurrent layers:
two-level verification, a drafter whose tree is verified per depth, and on a GPU a memory other than paged or a
non-linear layer schedule. A separate draft model drafts its branching tree for such a target the same way.

#### Batched draft-tree verification

In the batch engine (the server's continuous batching, `-np` > 1) `neuralfree.draft.tree=true` drafts one tree per
decoding request and verifies every request's tree in one masked forward of the step, prompt chunks of new requests
included. The step's verify rows are shared out over the requests' trees, best path scores first, so cutting a tree
only shortens its round and never changes a token.

| Property | Default | Meaning |
|---|---|---|
| `neuralfree.draft.tree.rows` | batch size times `--spec-draft-tree-nodes` | The step's verify rows: a number, or tiers by batch size, `bs:T` for a total or `bs:xF` for a factor of the batch size (`4:8;16:24`, `8:x2.5`); clamped between the batch size and 64 rows above the full-width total |
| `neuralfree.draft.tree.budget` | `full` | How the rows are shared: `full` verifies every drafted node, `topt` the strongest nodes over all requests, `thresh` the nodes whose score clears `neuralfree.draft.tree.thresh.gamma` (1.0) and `neuralfree.draft.tree.thresh.lambda` (0), `fixed` the same count for every request (`neuralfree.draft.tree.fixed`, default 16) |
| `neuralfree.draft.tree.mean` | | The `topt` total without `rows`: the mean accepted length times the batch size |
| `neuralfree.draft.tree.min` | 1 | Nodes every request keeps after the cut (at least 1) |
| `neuralfree.draft.tree.branches` | 4 | Branches a draft model splits each request's tree into |

A malformed value is refused naming its property. Refused by name: DSA, MSA, Step5 and DeepSeek-V4 attention layers
and recurrent layers (a recurrent or hybrid target verifies trees one sequence at a time, see
[single-pass draft-tree verification](#single-pass-draft-tree-verification)); MLA layers are covered. On a tiny
synthetic model the tree beats plain decoding at 4 requests on the CPU and OpenCL and trails chain speculation at 8
and 16; `-Dneuralfree.bench.batch.tree=1` runs that comparison.

#### Two-level verification (`neuralfree.draft.tlv`)

`-Dneuralfree.draft.tlv=true` (SpecPilot's two-level verification) verifies each draft tree twice. First a quantized
copy of the target (its projection matrices restated as `Q6_K` rows at `neuralfree.draft.tlv.bits=6`, the default,
or `Q8_0` rows at `8`) runs the tree-mask verify with plain argmax and commits nothing: its accepted path is a chain
and its next argmax a bonus token. The chain replays into the drafter, the drafter drafts a second, smaller tree
anchored at the bonus, and the exact target verifies chain, bonus and second tree in one tree-mask pass and commits.
Every emitted token is the exact target's greedy choice; the quantized copy only proposes. The copy's bytes count in
the context fit and the device placement budget.

Requirements, each refused by name: a target whose projection matrices are stored as F32, F16 or BF16 with rows of
whole copy blocks and more bytes than the copy (an already quantized target, or a routed-expert target, gains
nothing); a tree drafter that verifies in one pass (`neuralfree.draft.tree=true` with a dflash drafter); greedy
decoding (a grammar-constrained greedy request works, `temperature > 0` is refused); and a load the copy can shadow:
no layer or tensor split, no streamed or host-mapped weights, no MoE expert cache or host experts, no EPLB, and no
LoRA adapters or control vectors. No llama.cpp or vLLM option corresponds to it.

On a target that runs [W12](#lossless-12-bit-weights-w12) (`--w12`), the quantized round reads W12's nested 8-bit
view of the target's own weights instead of a copy: no second weight set, no second backend, and no view bytes in the
context fit. That view is fixed at 8 bits, so `neuralfree.draft.tlv.bits=6` with W12 is refused by name.

Measured on the RTX 5080 Laptop (greedy, best of three 48-token runs), TLV does not beat the single-pass tree verify,
and neither beats plain decoding, at the acceptance the available drafters reach:

| Pair | plain | chain | tree, width 32 | TLV, width 32 |
|---|---|---|---|---|
| pplx-embed 0.6B F16 target + 0.6B F16 draft, CUDA | 194.7 | 101.8 | 22.4 | 21.6 (copy) |
| same, OpenCL | 226.6 | 77.3 | 15.0 | 15.8 (copy) |
| same with W12 (`--w12-f16 round`), CUDA | 203.7 | | 21.4 | 21.5 (W12 view) |
| same with W12 (`--w12-f16 round`), OpenCL | 207.9 | | 14.0 | 13.9 (W12 view) |

TLV commits about twice the tree's tokens per exact verify (3.1 against 1.6 on the pplx pair), but the quantized
round, the replay and the second draft cost more than that saves at 3-27% acceptance, so keep it off unless a drafter
confirms far more of each tree. `TlvBenchmarkTest` (`-Dneuralfree.bench.tlv=1`, `-Dneuralfree.bench.backend=CUDA` or
`OPENCL`) reproduces the numbers.

### LoRA adapters

| Flag | Meaning |
|---|---|
| `--lora <gguf[:scale]|PEFT dir>` | Apply adapters, scale 1 unless given; repeatable. A PEFT directory (`adapter_config.json` plus `adapter_model.safetensors`) attaches to a GGUF or Hugging Face base model |
| `--lora-scaled <gguf>:<scale>[,...]` or `--lora-scaled <gguf> <scale>` | Apply adapters with explicit scales; repeatable |
| `--lora-base-config <dir>` | An HF directory whose config the adapter's GGUF base takes its hyperparameters from |
| `--hf-base-dir <dir>` | Local base model a decision-LoRA checkpoint (PEFT checkpoints trained with llama.cpp's `lev` conversion) merges at load |

Adapters given on the command line replace the `neuralfree.lora` list.

**Vision-encoder and connector adapters.** An adapter may also carry the vision tower's and the projector's tensors (`v.blk.<n>...`, `mm.<n>.weight`), as vLLM's `enable_tower_connector_lora`. A target absent from the text model is resolved against the projector file given by `--mmproj`, so one `--lora` adapter updates the text forward pass and the vision encoder together; a tower-only adapter may carry the projector's `clip` architecture instead of the text model's, while an adapter that mixes text and vision targets keeps the text architecture. The deltas apply after each tower, MoE-router and connector product, before its bias, on both the CPU and OpenCL vision encoders. The server applies the startup `--lora` adapters to the media path, and a request's own adapter (`--lora-modules`, `/v1/load_lora_adapter`; selected by `"model": "<adapter>"` or `"lora": [{"id": N, "scale": S}]` on `/v1/chat/completions`, `/v1/completions` with `multimodal_data`, `/v1/embeddings`, `/v1/rerank` and the Anthropic messages endpoint) applies its vision-tower and connector pairs while that request's images and videos encode. The adapter's identity scopes the encoded-media cache and the prompt's media-chunk identities, so the same image under two adapters encodes and caches twice. A PEFT adapter directory with LoRA on `vision_tower.*`, `multi_modal_projector.*`, `merger.*` or `visual.*` modules produces vision pairs against the base's projector conversion, as `nftool export --mmproj` plans those modules; the base still needs its Hugging Face config (`--lora-base-config` for a GGUF base). A target that names no tensor of either model is refused by name at load.

### Control-vector flags

| Flag | Meaning |
|---|---|
| `--control-vector <gguf[:scale]>[,...]` | Add control vectors, scale 1 unless given; repeatable ([Control vectors](#control-vectors)) |
| `--control-vector-scaled <gguf>:<scale>[,...]` or `--control-vector-scaled <gguf> <scale>` | Add control vectors with explicit scales; repeatable |
| `--control-vector-layer-range <start> <end>` | The 1-based decoder layers the vectors apply to, inclusive |

Vectors given on the command line replace the `neuralfree.control-vector` list.

### Diffusion (`--mode diffusion`)

| Flag | Meaning | Default |
|---|---|---|
| `--diffusion-steps <n>` | Denoising steps | 128 |
| `--diffusion-eps <f>` | Timestep schedule epsilon (Dream); excludes `--block-length` | 0.001 |
| `--block-length`, `--diffusion-block-length <n>` | Block schedule with blocks of `<n>` tokens (LLaDA) | |
| `--algorithm`, `--diffusion-algorithm <0-4>` | origin, entropy, margin, random, confidence | 4 |
| `--diffusion-alg-temp <f>` | Confidence temperature | |
| `--cfg`, `--diffusion-cfg-scale <f>` | Classifier-free guidance scale; `0` = off | 0 |
| `--diffusion-add-gumbel-noise` | Gumbel noise on the logits | off |

### Embeddings, classifiers and audio

| Flag | Meaning | Default |
|---|---|---|
| `--embed-preview <n>` | Leading embedding dimensions to print | 8 |
| `--token-types <csv>` | Segment id per token (BERT token types) | |
| `--document <text>` | Document scored against the prompt (`--mode rerank`) | |
| `--decision-order <csv>` | Decision code per token (clef, vllm-decision2) | |
| `--audio-codes <csv>` | WavTokenizer codes | parsed from the prompt |
| `--audio-out <file>` | A `.wav` name writes the synthesized waveform ([Speech synthesis](#speech-synthesis)); any other name writes the audio features or hidden rows as float32 little-endian | |
| `--audio-rate <hz>`, `--audio-fft <n>`, `--audio-hop <n>` | The speech geometry, filling or overriding what the model declares | the model's |

After each generation, the demo prints timing statistics. With `--stats-file <path>` it also appends the session's statistics (model, backend, memory plan, and the timing, prompt and response of each generation) to `<path>` when the session ends; a relative path is resolved against the working directory, which is the project directory under `runLLMDemo`. Without the flag nothing is written.

### Speech synthesis

A `wavtokenizer-dec` model turns audio codec codes into a waveform (`--mode audio`). The model declares its geometry with the GGUF keys `audio.sample_rate`, `audio.n_fft` and `audio.hop_length` (all three or none); one code is one spectral frame. The codes come from `--audio-codes` or the prompt:

```bash
LLMDemo -m wavtokenizer.gguf --mode audio --audio-codes "12,44,7" --audio-out speech.wav
# a model that declares no geometry
LLMDemo -m wavtokenizer.gguf --mode audio --audio-codes "12,44,7" --audio-out speech.wav \
  --audio-rate 24000 --audio-fft 1920 --audio-hop 320
```

The server serves the same model on llama.cpp's `/tts` and OpenAI's `/v1/audio/speech`:

```bash
curl -s localhost:8080/tts -H 'Content-Type: application/json' -d '{"codes": [12, 44, 7]}' -o speech.wav
curl -s localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "12, 44, 7", "response_format": "pcm"}' -o speech.pcm
curl -sN localhost:8080/tts -H 'Content-Type: application/json' \
  -d '{"input": "Hello.", "stream": true}' -o speech.wav
curl -sN localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hello.", "stream": true, "stream_format": "sse"}'
```

- `response_format` is `wav` (the default; 16-bit PCM mono) or `pcm` (raw little-endian int16).
- `stream: true` delivers the audio as the decoders produce it instead of waiting for the whole utterance:
  `wav` writes the canonical 44-byte header with the unknown-length sentinel (`0xFFFFFFFF`) once and appends
  16-bit PCM runs as each decoder window is decoded (one hop per spectral frame); `pcm` streams the raw samples
  with no header. A client that disconnects cancels the generation.
- `stream_format` selects the streamed envelope: `audio` (the default) is the raw byte stream, `sse` is
  OpenAI's server-sent events — `speech.audio.delta` events whose `audio` field is the base64 of the same bytes,
  then a final `speech.audio.done` — and requires `stream: true`. Any other value is refused by name.
- On a codec-decoder model, input text that is not integer codes answers 501 `no speech-generator backbone`;
  text needs a speech-generator backbone (qwen3tts, pockettts or lfm3a, below). A model without a speech decoder
  answers 501 `does not support audio generation`.
- In the engine API, `LLMModel.speechFormat()` returns the declared geometry (null without the keys),
  `LLMModel.requireSpeechFormat()` the same object or the refusal a synthesis would raise,
  `LLMModel.synthesizeSpeech(int[] codes)` the `AudioClip`, and
  `LLMModel.synthesizeSpeech(int[] codes, Consumer<AudioClip>)` the incremental form whose clips concatenate to
  it.
- `SopranoVocoder.open(path)` loads a soprano generator GGUF (`clip.gen.audio.projector_type = soprano`, 32 kHz,
  512 samples per frame) as a library: `spectrum(frames)` returns the per-frame polar spectra and
  `decode(frames)` the PCM, trimmed as llama.cpp's `decode_spectrum` does.

#### Text to speech (qwen3tts)

A `qwen3tts` backbone speaks text through its generator file: a clip GGUF holding the code predictor and the code2wav DAC decoder, passed as `--mmproj` (also `-Dneuralfree.server.mmproj`). Serve the pair:

```bash
neuralfree serve -m qwen3-tts.gguf --mmproj qwen3-tts-mmproj.gguf --port 8080
curl -s localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hello from NeuralFree.", "lang": "english"}' -o speech.wav
curl -s localhost:8080/tts -H 'Content-Type: application/json' \
  -d '{"input": "Hello.", "response_format": "pcm", "max_tokens": 128}' -o speech.pcm
```

- `lang` is llama.cpp's ISO code (`zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru`) or the language name; blank speaks `english`. A language with no `<|codec_language_<name>|>` token answers 501 naming it.
- `voice` names a speaker preset; this pipeline has none, so it answers 501 naming the preset. A speaker reference (`speaker_ref` file upload or `speaker_ref_b64`) clones a voice when the generator file carries the speaker encoder; without one it answers 501 naming `speaker_ref` ([Speech synthesis with a cloned voice](#speech-synthesis-with-a-cloned-voice)).
- `seed` (default 0), `top_k` (default 50), `top_p` (default 1) and `max_tokens` / `n_predict` (default 512 frames) match llama.cpp's `/tts` fields. `response_format` is `wav` or `pcm`; `stream: true` streams the code2wav windows as they are decoded and `stream_format: "sse"` wraps them in `speech.audio.delta` / `speech.audio.done` events. A streamed request the generator refuses (an unknown language, a `voice` preset) answers with its 501 before the stream starts.
- The CLI takes the same pair: `LLMDemo -m qwen3-tts.gguf --mmproj qwen3-tts-mmproj.gguf --mode audio --prompt "Hello from NeuralFree." --audio-lang en --audio-out speech.wav`.
- Both files convert from a Hugging Face `Qwen3TTSForConditionalGeneration` checkpoint: `nftool export <checkpoint> qwen3-tts.gguf` writes the backbone and `nftool export --mmproj <checkpoint> qwen3-tts-mmproj.gguf` the generator, as llama.cpp's `convert_hf_to_gguf.py --mmproj` does. The generator takes the code predictor from `talker.code_predictor.*` and `talker.model.codec_embedding` (its 15 codebook and LM-head pages stacked), and the code2wav decoder from `speech_tokenizer/` (RVQ codebooks folded from their EMA statistics, weight-norm pairs and the SnakeBeta parameters folded). The speaker-encoder tensors are not converted: no voice cloning is served.
- The backbone runs on any engine backend (CPU, OpenCL, CUDA, Vulkan, HIP) and produces the same codes and PCM as the CPU engine. On a GPU engine the code predictor and the code2wav decoder run on the same device (OpenCL and CUDA kernels, Vulkan shaders), with the generator weights uploaded once per generator file; per-codebook sampling stays on the host, so the codes match the CPU pipeline exactly. On the CPU engine both run as host graphs.

#### Text to speech (pockettts)

A `pockettts` backbone speaks text through its generator file: a clip GGUF holding the flow-matching network and the streaming mimi decoder, passed as `--mmproj` (also `-Dneuralfree.server.mmproj`). Serve the pair:

```bash
neuralfree serve -m pocket-tts.gguf --mmproj pocket-tts-mmproj.gguf --port 8080
curl -s localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hello from NeuralFree."}' -o speech.wav
curl -s localhost:8080/tts -H 'Content-Type: application/json' \
  -d '{"input": "Hello.", "response_format": "pcm", "max_tokens": 128}' -o speech.pcm
```

- The audio is 24 kHz 16-bit PCM mono: `response_format` is `wav` (the default) or `pcm`; `stream: true` streams the mimi decoder's windows as they are decoded, and `stream_format: "sse"` wraps them in OpenAI's speech events.
- `lang` is checked against the pack: English packs speak `english` / `en` and refuse any other language by name (501); `french_24l` speaks `french` / `fr`. Blank speaks the pack's language.
- `voice` names no preset in this pipeline and answers 501 naming the preset. A speaker reference (`speaker_ref` file upload or `speaker_ref_b64`) clones a voice when the generator file carries the mimi speaker encoder; without one it answers 501 naming `speaker_ref` ([Speech synthesis with a cloned voice](#speech-synthesis-with-a-cloned-voice)).
- `seed` (default 0) seeds the flow noise; `max_tokens` / `n_predict` caps frames (default 512; 12.5 frames per second). `top_k` / `top_p` are accepted but do not apply: the decoder samples continuous latents, not tokens.
- The reference's per-variant settings apply: `english` 0.3 temperature, `english_2026-01` 0.7 with short-prompt padding, `english_2026-04` 0.3, `french_24l` 0.7 keeping eight frames after the end-of-speech probe; an unknown variant uses 0.7 and the prompt-length tail guess (five frames up to six tokens, three above).
- The CLI takes the same pair: `LLMDemo -m pocket-tts.gguf --mmproj pocket-tts-mmproj.gguf --mode audio --prompt "Hello from NeuralFree." --audio-lang en --audio-out speech.wav`.
- Preparing the files: `nftool export <checkpoint> pocket-tts.gguf` converts the Hugging Face `PocketTTSModel` backbone, and `nftool export --mmproj <checkpoint> pocket-tts-mmproj.gguf` writes the generator file (llama.cpp's `convert_hf_to_gguf.py --mmproj` produces an equivalent one). The generator declares `clip.gen.audio.projector_type = pockettts_gen` and its variant comes from the checkpoint directory name.
- Prompts longer than 50 tokens are split on sentence punctuation, then on weaker punctuation; each chunk re-prompts the backbone with `<|audio_bos|>` and restarts the decoder, so chunk audio is not continuous with the previous chunk. Latents are decoded in windows of eight frames. The flow and mimi weights are read on the host; the backbone runs on the engine's configured device.
- On a GPU engine the flow-matching network and the mimi decoder run on the backbone's device (OpenCL and CUDA kernels, Vulkan shaders), with the generator weights uploaded once per generator file; the flow noise, the latency draw, token sampling and the end-of-speech check stay on the host. On the CPU engine both run as host graphs.

#### Text to speech (lfm3a)

An `lfm3a` backbone speaks text through its generator file: a clip GGUF with `clip.gen.audio.projector_type = lfm3a_gen` holding the MaskGIT depthformer and the liquid-codec decoder, passed as `--mmproj` (also `-Dneuralfree.server.mmproj`). Serve the pair:

```bash
neuralfree serve -m lfm3.gguf --mmproj lfm3a-generator.gguf --port 8080
curl -s localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hello from NeuralFree.", "lang": "en", "voice": "female"}' -o speech.wav
curl -s localhost:8080/tts -H 'Content-Type: application/json' \
  -d '{"input": "Hello.", "response_format": "pcm", "max_tokens": 128}' -o speech.pcm
```

- The utterance (`input`, or its `prompt` alias), `lang` and `voice` render the model's `tokenizer.chat_template.tts` template: the system message becomes `Perform TTS in <language> using a <voice> voice.` and the prompt ends at `<|audio_start|>`. An unknown language keeps its raw code and an off-list voice falls back to `female`, as llama.cpp's template does.
- `seed` (default 0) and `max_tokens` / `n_predict` (default 512 frames) are the request's hands on the run. The temperature, top-k and repetition penalty come from the generator file's `clip.gen.audio.sampling.temp` / `top_k` / `penalty_repeat` / `penalty_last_n` keys (the model card's defaults: 0.4, 0, 1.2, 64), so the request's `top_k` / `top_p` are not applied on this pipeline.
- `response_format` is `wav` (the default; 16-bit PCM mono at 24 kHz) or `pcm` (raw little-endian int16). `stream: true` streams one clip per codec frame as the liquid codec decodes it, and `stream_format: "sse"` wraps the clips in OpenAI's speech events. A speaker reference (`speaker_ref` / `speaker_ref_b64`) is refused with 400 naming the field.
- A backbone without `neuralfree.server.mmproj` answers 501: `speech generation needs the generator file: set neuralfree.server.mmproj (llama.cpp --mmproj) to the lfm3a MaskGIT depthformer and liquid-codec GGUF`. Without a model the route answers 501 `no model loaded: speech generation needs a qwen3tts or lfm3a backbone`; on a model with no speech backbone, 501 `this server has no speech-generator backbone: the loaded <arch> model cannot turn text into audio; a qwen3tts or lfm3a backbone with its generator file does`.
- The CLI takes the same pair: `LLMDemo -m lfm3.gguf --mmproj lfm3a-generator.gguf --mode audio --prompt "Hello from NeuralFree." --audio-lang en --voice female --audio-out speech.wav`.
- The generator GGUF converts from an `Lfm3AudioForConditionalGeneration` checkpoint, or from the audio half of an `Lfm3OmniForConditionalGeneration` checkpoint whose codec ships, with `nftool export --mmproj <checkpoint> lfm3a-gen.gguf`, as llama.cpp's `convert_hf_to_gguf.py --mmproj` does (see [Converting vision/audio projector towers](#converting-visionaudio-projector-towers---mmproj)).
- On a GPU engine the MaskGIT depthformer and the liquid-codec decoder run on the backbone's device (OpenCL and CUDA kernels, Vulkan shaders), with the generator weights uploaded once per generator file; codebook sampling and the iSTFT stay on the host. On the CPU engine both run as host graphs.

On chat completions, `modalities` (or OpenAI's `output_modalities`) naming `audio` answers with speech:
`["audio"]` speaks a fresh turn, `["text","audio"]` interleaves text and audio. `message.content` carries the
turn's text (`null` without the `text` modality) and `message.audio` the OpenAI audio object (`id`, base64
`data`, `expires_at`, `transcript`, `format` `wav` or `pcm16`):

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Say hi"}],
  "modalities": ["text", "audio"], "max_tokens": 128}'
```

- The stream sends the role chunk, the text deltas, the waveform as one `delta.audio` chunk and a final `stop`
  chunk (with `usage` when `stream_options.include_usage` asks), then `data: [DONE]`.
- `audio: {"voice": ..., "format": ...}` binds the template voice and picks `wav` or `pcm16`; `lang` is the
  alias for the language variable. Unknown modalities, both modality fields together, `n > 1`, unsupported
  formats, conflicting voices and server tool execution beside audio refuse by name.
- A model without the omni pipeline answers 400 naming the missing piece; qwen3tts speaks through
  `/v1/audio/speech`.
- The Responses and Anthropic bodies answer in text: the reference serializes audio in neither format.

#### Speech synthesis with a cloned voice

A qwen3tts or pockettts generator file that carries a speaker encoder can clone the voice of a reference recording.
The reference conditions the backbone; it does not change the model or the voice of the text.

Server (`POST /v1/audio/speech`, and llama.cpp's `POST /tts`):

- JSON body: add `"speaker_ref_b64"` with the base64 bytes of a WAV, AIFF, AU or (when ffmpeg is available) MP3,
  FLAC, OGG or M4A file.
- Multipart form: upload the recording as the `speaker_ref` file field; the other request fields (`input`, `lang`,
  `seed`, `top_k`, `top_p`, `max_tokens`, `response_format`, ...) are form fields. If both are present, the uploaded
  file wins.
- A JSON `speaker_ref` string is refused: upload the file, or send its bytes as `speaker_ref_b64`.
- The bytes go through the server's audio decoder (the same formats as the transcription endpoints). An empty
  reference, one over 25 MiB, one longer than 120 seconds, or one in a format the decoder cannot read is a 400
  naming `speaker_ref`; a malformed JSON field is a 400 naming `speaker_ref_b64`.
- A generator file without a speaker encoder answers 501 naming `speaker_ref`; a request on the audio-code path
  (`"codes"`) is refused too. `response_format` stays `wav` (default) or `pcm`. `voice` requests a named preset:
  neither generator has presets, so the message points at `speaker_ref`.

Example:

```bash
curl -s http://localhost:8080/v1/audio/speech \
  -F input='Hello from a cloned voice.' \
  -F speaker_ref=@reference.wav \
  -F response_format=wav --output speech.wav
```

CLI (llama.cpp's helper shape, through the unified launcher):

```bash
neuralfree cli --model qwen3tts-backbone.gguf --mmproj qwen3tts-gen.gguf \
  --mode audio --prompt 'Hello from a cloned voice.' \
  --audio-speaker reference.wav --audio-out speech.wav
```

`--audio-speaker` is accepted by the qwen3tts and pockettts audio modes; lfm3a has no encoder and refuses it by
name. Pockettts truncates a reference to its first 30 seconds (the reference preprocessor's limit); qwen3tts uses
the whole clip. The encoders run on the host, so the voice path is identical for CPU and device backbones.

Conversion (`nftool ... --mmproj`): the qwen3tts voice-clone checkpoints keep `speaker_encoder.*` and the file
declares `clip.has_audio_encoder = true` plus `clip.audio.projector_type = qwen3tts_spkenc`; pockettts checkpoints
keep the mimi encoder, its transformer and `flow_lm.speaker_proj_weight` and declare `pockettts_spkenc`. A qwen3tts
CustomVoice/VoiceDesign checkpoint (no speaker encoder) exports the generator alone, and a checkpoint with only part
of an encoder has those tensors dropped, so the file never declares an encoder it does not hold.

### Speech to text

`POST /v1/audio/transcriptions` transcribes an uploaded clip and `POST /v1/audio/translations` translates it;
llama.cpp's unversioned `POST /audio/transcriptions` is the transcription alias. Both take
`multipart/form-data`, the clip in `file` (at most 25 MiB), and the OpenAI form fields:

| Field | Meaning |
|---|---|
| `model` | The served model to use; omitted, the single model serves (required in router mode). |
| `prompt` | Context or instruction for the model (llama.cpp's ASR prompt). |
| `language` | The spoken language (ISO-639-1): a hint for a transcription, the source language of a translation. |
| `to_language` | The translation target, English when omitted; refused on transcriptions. |
| `response_format` | `json` (default; `{"text": "..."}`) or `text` (bare `text/plain`). |
| `stream` | `true` streams server-sent events (below). |
| `temperature`, `max_tokens` / `max_completion_tokens`, `seed`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty` | The generation settings; temperature 0 and a 448-token answer when omitted. |

Any other field, and any response format other than `json` or `text`, is refused by name.

```bash
curl -s localhost:8080/v1/audio/transcriptions -F file=@speech.wav -F language=en
# {"text":"..."}
curl -s localhost:8080/v1/audio/translations -F file=@speech.wav -F language=fr -F to_language=en
curl -s localhost:8080/v1/audio/transcriptions -F file=@speech.wav -F response_format=text
curl -sN localhost:8080/v1/audio/transcriptions -F file=@speech.wav -F stream=true
```

The decoder reads WAV, AIFF and AU directly, and MP3, FLAC, OGG and M4A through an `ffmpeg` on the PATH
(they are refused by name when it is missing).

Streaming answers vLLM's speech-to-text chunks, one `data:` frame per delta:

```json
{"id":"trsc-...","object":"transcription.chunk","created":1730000000,"model":"...","choices":[{"delta":{"content":"..."},"finish_reason":null}]}
```

A translation's `object` is `translation.chunk` and its id starts `trsl-`; the last chunk carries the
`finish_reason` with an empty delta, then `data: [DONE]` ends the stream. The deltas joined are the
non-streaming answer.

These routes are the existing audio input path: the clip becomes one `audio_url` part beside an
instruction, so they need a model that accepts audio and an audio projector (`--mmproj`, or
`neuralfree.server.mmproj`); without one the request is refused naming the audio input.

#### Realtime transcription

`ws://host/v1/realtime` streams audio into a session (vLLM's realtime transcription protocol). The
connection opens with `session.created`; the client sends base64 PCM16 little-endian mono at 16 kHz:

```json
{"type": "session.update", "model": "<served model>"}
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}
{"type": "input_audio_buffer.commit"}
```

`session.update` is optional when the server serves one model. A commit transcribes the buffered clip as
`transcription.delta` events (`{"type":"transcription.delta","delta":"..."}`) and closes with
`transcription.done` carrying the full `text`. `input_audio_buffer.clear` drops the buffer;
`"final": true` on a commit transcribes the clip and then refuses further audio for the connection.
A malformed event answers `{"type":"error","error":"...","code":"..."}` and keeps the session open. The
WebSocket upgrade carries the same API key as the HTTP routes (`Authorization: Bearer <key>` or
`x-api-key`).

An appended message may carry up to 1 MiB of base64; the buffered clip is capped at 25 MiB of PCM16.

#### Batch records

`neuralfree run-batch` routes records whose `url` is `/v1/audio/transcriptions` or
`/v1/audio/translations` through the same mapping. The body carries the multipart fields as JSON and
vLLM's `file_url` (an http(s) URL or a data URL) in place of `file`:

```json
{"custom_id":"a","method":"POST","url":"/v1/audio/transcriptions","body":{"file_url":"https://example.com/speech.wav","language":"en"}}
{"custom_id":"b","method":"POST","url":"/v1/audio/translations","body":{"file_url":"data:audio/wav;base64,...","to_language":"en"}}
```

`file` is refused; `stream` and a `response_format` other than `json` are refused in batch. The
transcript lands under `response.body` as `{"text": "..."}` like any other successful record.

### Pooling type override

The pooling a model's embeddings use comes from the GGUF's `<arch>.pooling_type` unless an override replaces it.
The override uses llama.cpp's `--pooling` names (vLLM's pooler `pooling_type` names are also accepted). Every
embedding consumer uses it: `LLMDemo --mode embed/classify/rerank` and AUTO mode, the server's `/v1/embeddings`,
`/pooling`, `/score` and `/rerank`, and `nftool embedding` / `retrieval`.

| Setting | Values | Notes |
|---|---|---|
| `-Dneuralfree.pooling=<t>` | `none \| mean \| cls \| last \| rank` (case-insensitive; vLLM's `ALL` = `none`) | Unset: the GGUF's pooling. `rank` needs a classifier head (`cls.weight` or `cls.output`) and refuses the load by name otherwise. vLLM's `STEP` is refused. |
| `--pooling <t>` | as above | llama.cpp's flag; LLMDemo, the server launcher, `nftool embedding`, `nftool retrieval`. |
| `--override-pooler-config <json>`, `--pooler-config <json>` | `{"pooling_type": "MEAN"}`, `{"seq_pooling_type": "CLS"}`, `{"tok_pooling_type": "ALL"}` | vLLM's spellings (server launcher). `pooling_type` beside either split type refuses, as in vLLM; the two split types together refuse (the engine runs one pooling). Any other key (`use_activation`, `dimensions`, ...) refuses by name. `{}` keeps the defaults. |

With `--pooling mean|cls|last`, a generative model embeds as llama.cpp's "embeddings of any model" do. AUTO mode in
LLMDemo then picks `embed`.

## Hugging Face checkpoints

The engine opens a Hugging Face checkpoint wherever it opens a GGUF file. Format detection (`ModelSources.detect`) is content-based:

1. Regular files: a file whose first 4 bytes are `GGUF`, else a file ending in `.safetensors`. An HF source file is a single `*.safetensors` whose parent directory holds `config.json`, `params.json` (Mistral native) or a checkpoint-detector file; only that file's tensors are used. A file the parent index lists as a shard is refused with a message to pass the directory instead.
2. A directory containing `config.json` plus `*.safetensors` files — or `params.json` plus `consolidated*.safetensors` — is an HF checkpoint. Line endings, dtype names, index files (`*.safetensors.index.json`) and per-shard files are all handled; encoder-decoder, diffusion, embedding, reranker, audio and drafter architectures load like their GGUF counterparts.
3. A directory without safetensors files that holds exactly one `.gguf` file, or one split set `*-00001-of-NNNNN.gguf`, opens that GGUF (the `gguf` load format vLLM and SGLang use).
4. A directory with `adapter_config.json` and `adapter_model.safetensors` is a PEFT adapter: it attaches to a base model through `--lora` (`ModelSources.openAdapter`), and as a model it is refused with a message saying so.
5. Only `pytorch_model*.bin`, `*.pt` or `*.pth` files: a torch pickle checkpoint, which loads through a restricted unpickler (`TorchCheckpointFactory` in `CheckpointContainers`); unsafe opcodes and non-tensor payloads are refused by name at load.

A parsed HF checkpoint becomes the same `ModelSource` view a GGUF file produces ([LLM-INFERENCE.md](LLM-INFERENCE.md#model-sources)), so `--backend`, `--ctx`, `--kv-type`, `--mtp`, `--lora`, `--draft` and the sampler flags mean the same thing on both paths. On disk it needs nothing beside the checkpoint: the derived tensor cache below points at `~/.cache/neuralfree/hf` and is created only when derivations spill.

### Flags and properties

These flags set the matching `ModelSourceOptions` or `EngineConfig` fields. An enum value is parsed case-insensitively, with `-` and `_` equivalent; an unknown value is rejected naming the property. Property defaults apply when the flag is absent.

| Flag | Property | Default | Meaning |
|---|---|---|---|
| `--hf-materialize auto\|memory\|disk\|none` | `neuralfree.hf.materialize` | `auto` | Where derived tensors live ([Materialization](#materialization-of-derived-tensors)) |
| `--hf-memory-max <n[K\|M\|G]>` | `neuralfree.hf.materialize.memory.max` | `2G` | Largest total of derived bytes `auto` keeps in direct memory |
| `--hf-cache-dir <dir>` | `neuralfree.hf.cache.dir` | `~/.cache/neuralfree/hf` | Parent of the per-process directories of derived tensors written to disk |
| `--hf-outtype source\|auto\|f32\|f16\|bf16\|q8_0` | `neuralfree.hf.outtype` | `source` | Type of the view's float tensors. `auto` reproduces the converter's `--outtype auto`; the other values, except `source`, reproduce the converter's fixed outtypes; `q8_0` quantizes through the online quantizer |
| `--quant-online none\|q8_0\|q4_0\|q4_1\|q5_0\|q5_1\|iq4_nl` or a [vLLM method](#vllm-online-quantization---quant-online---quantization-config) | `neuralfree.quant.online` | `none` | Online quantization of float matmul weights with llama-quantize eligibility (norms, position and patch embeddings, convolutions, relative-position biases, gates and codebooks stay float, as llama-quantize keeps them; LIMITE's attention gates and dense connection mixers, GLM5-next's hc/indexer/MLA/SSM projections and the vllm-decision2 decision head stay float on those architectures only) |
| `--leave-output-tensor` | `neuralfree.quant.online.output` | `true` | Keep `output.weight` unquantized under `--quant-online`, as llama-quantize's `--leave-output-tensor`; the property's default `true` quantizes it |
| `--hf-quant-exact` | `neuralfree.hf.quant.exact` | off | Prefer bit-exact weight types over close repacks ([Supported quantization formats](#supported-quantization-formats)) |
| `--hf-no-mtp` | `neuralfree.hf.mtp` | true | The flag keeps the model's MTP/NextN layers out (the converter's `no_mtp`); the property set to `true` includes them again |
| `--hf-draft none\|mtp\|no-nextn\|dspark` | `neuralfree.hf.draft` | `none` | The converter's draft selection ([draft selection](#draft-selection---mtp---no-nextn---dspark)); `--hf-no-mtp` is the `no-nextn` shorthand, and `neuralfree.hf.mtp=false` conflicts with `mtp` and `dspark` |
| `--hf-st-dense true\|false` | `neuralfree.hf.st.dense` | `true` | Include sentence-transformers `2_Dense`/`3_Dense` heads; `false` reproduces the converter's default off |
| `--hf-overrides <json>` | `neuralfree.hf.overrides` | none | JSON object merged into `config.json` before conversion (vLLM's `hf_overrides`) |
| `--hf-tokenizer <dir\|file.gguf>` | `neuralfree.hf.tokenizer` | none | Tokenizer files from another directory, or the `tokenizer.*` keys of a GGUF file, replacing the model's |
| `--act-quant weight-only\|checkpoint` | `neuralfree.act.quant` | `weight-only` | Activation quantization ([Activation quantization](#activation-quantization)) |
| `--model-sampling-defaults true\|false` | `neuralfree.sampling.model.defaults` | `true` | Apply the model's sampling defaults ([Model sampling defaults](#model-sampling-defaults)) |
| `--lora-base-config <dir>` | `neuralfree.lora.base.config` | none | HF config of a LoRA adapter's base when the base is a GGUF file |
| `--hf-base-dir <dir>` | `neuralfree.hf.base.dir` | none | Local copy of the base model a decision-LoRA checkpoint merges at load; without it, its `base_model_name_or_path` is used when that names an existing local directory |
| `--export-gguf <out.gguf>` | — | none | Write the model's GGUF view to `<out.gguf>` and exit without generating ([Exporting to GGUF](#exporting-to-gguf)) |
| `--export-mode native\|compat\|compat-exact` | — | `native` | With `--export-gguf`: keep the extension tensor types; convert them to llama.cpp types; or write every lossy or approximate value as F32 |
| `--export-overwrite` | — | off | Replace an existing output file |

**Draft selection.** The same draft selection is available wherever a Hugging Face checkpoint is opened, through the system property `neuralfree.hf.draft` (`none` by default, or `mtp`, `no-nextn`, `dspark`) and the demos' `--hf-draft <mode>`; `--hf-no-mtp` remains the `no-nextn` shorthand. MTP export is supported by the converters of DeepSeek-V3.2 and DeepSeek-V4, the Dots3-Note family, GLM-4 MoE / GLM-4 MoE Lite / GLM-MoE-DSA / GLM-5 Next, Bailing-MoE V3, HY-V3, Mellum, MiMo V2 (including the vLLM `MiMoForCausalLM` alias), Step 3.5/3.7, Nemotron-H (MoE builds with one MTP block), Qwen3-Next, Qwen3.5 / Qwen3.5 MoE, Qwen4-Exp and Qwen4-Exp's vLLM aliases. `--mtp` on a Nemotron-H or Mellum checkpoint without an MTP head is refused, as the converter does. The DSpark drafts read the target checkpoint itself, so no separate target-model directory is needed; the resulting `dflash` GGUF carries the draft's block protocol keys (`dflash.block_size`, `dflash.target_layers`, `dflash.sample_from_anchor`, the confidence-head flag) and loads as a DFLASH drafter.

### Materialization of derived tensors

Some tensors of an HF checkpoint are "derived": the loader transforms the checkpoint's bytes into GGUF names, shapes and layouts (name mapping, `permute`d attention shapes, fused `ffn_gate_up`, MLA `kv_b` splits, `quantize_*` repacks for `--quant-online` and `--hf-outtype q8_0`). A direct tensor — one that only maps safetensors bytes into a GGUF name and shape — is always served straight from the mmapped files and costs nothing.

The mode where the derived bytes go is `--hf-materialize`; `auto` decides per reader and per tensor, and logs each decision once at INFO with byte totals:

| How the loader reads | Row-local producer (e.g. a plain concat or stack) | Non-row-local producer (e.g. a K-major GPTQ repack) |
|---|---|---|
| One pass for GPU upload (`RESIDENT_UPLOAD`) | Not materialized: produced on each read | Whole-tensor window: produced once into a buffer kept until the next tensor is read — in memory when it fits `--hf-memory-max`, else to disk |
| Streamed layer re-reads per token (`STREAMING`) | To disk | To disk |
| CPU reads that keep the regions open (`openRegions`) | In memory when the total derived bytes fit `--hf-memory-max`, else to disk | Same |

An explicit `memory` or `disk` applies to every derived tensor. An explicit `none` produces on every read, keeps the whole-tensor window for non-row-local producers, and is not valid for `openRegions` (which would keep everything produced). In every mode the bytes read for one tensor are bounded by about 1.1 × that tensor's original source bytes.

The disk mode writes each derived tensor once into a per-process sidecar directory `<cache-dir>/run-<pid>-<random>/`:

- created with `rwx------` where POSIX attributes apply; it holds a lock file held for the life of the source;
- sidecar files are opened `CREATE_NEW` with `rw-------` and `DELETE_ON_CLOSE`, and the directory is deleted when the source closes; stale `run-*` directories of dead processes are cleaned up when the first sidecar is opened. Nothing is reused across runs.
- **Free-space requirement.** Before a tensor is written, the process checks the file store's usable space covers it and refuses with `IOException("materializing <tensor> needs <n> bytes in <dir>; free space or pass --hf-cache-dir")` — free space on the device, or point `--hf-cache-dir` at a device that has it.

`memory` keeps produced tensors in direct (off-heap) memory, so it is bounded by `--hf-memory-max` under `auto` and by the JVM heap when set explicitly.

### Activation quantization

`--act-quant` chooses how a checkpoint whose weights carry activation-quantization parameters behaves:

- **`weight-only` (the default).** The weights are used as stored, and the activation-quantization carrier is ignored: plain `W · A16` semantics, as llama.cpp has.
- **`checkpoint`.** The input of every projection whose checkpoint carries a scheme is *fake-quantized* — quantized and immediately dequantized per the checkpoint's scheme — and the result feeds the unchanged weight kernel (`y = W · x̂`). One kernel per backend serves all weight types and every kernel family.

Under `checkpoint`, numerics agree with vLLM's integer and FP8 GEMMs: the quantizer's codes are reproduced operation for operation (scales, rounding, saturation and the 128-value groups), and the products of the coded weights and codes are f32-exact, so results agree up to f32 accumulation rounding.

Under `checkpoint`, a projection of 8 or more tokens whose checkpoint scheme is symmetric int8 (`INT8_TOKEN`, `INT8_STATIC` or `INT8_GROUP128`) runs the int8 MMQ GEMM on the checkpoint's own activation codes: `LLM_Quant_MMQ_Int8Static` / `LLM_Quant_MMQ_Int8Row` / `LLM_Quant_MMQ_Int8Group128` quantize each token row once per projection into the 8-bit records the GEMM reads, with the scheme's scale (the static `input_scale`, the per-token `amax / 127`, or the per-128-run `max(amax, 1e-10) / 127`) and the scheme's rounding, and the fake-quantization pass is skipped. The weight stays in its stored codes, so an int8 W8A8 checkpoint (`I8_B128`, one f32 scale per 128 values) is a true int8×int8 GEMM (`mma.m16n8k32` / `m16n8k16` s8 on CUDA sm_80+, dp4a or scalar integer dots elsewhere, cooperative matrices on Vulkan), and the 4- and 5-bit weight types run the same W4A8 integer GEMM. Other schemes (asymmetric int8, FP8, FP4 and MX), weights with per-expert activation parameters, weights with an act-order permutation, and backends without the staging kernels keep the fake-quantization path: the numerics stay vLLM's, but there the product is not computed in int8.

The checkpoint's scheme for each weight, its static parameters and model-level bound (`neuralfree.act_quant.version`, `.weight_names`, `.schemes`, `.scale_ub`, and sibling `.input_scale` / `.input_zero_point` tensors) are read from the parsed view on both paths. Where the checkpoint quantizes its *output* activations, the load reports it and refuses under `checkpoint` with a named GgufFormatException — the reference's output activation quantization is not emulated; use `weight-only`. It is carried and dropped with one WARN under `weight-only`.

### vLLM online quantization (`--quant-online`, `--quantization-config`)

`--quant-online <t>` (property `neuralfree.quant.online`) also accepts vLLM's online quantization methods. They quantize the float weights of a Hugging Face checkpoint while it loads:

| Value | Weights quantized | Stored as | Scales |
|---|---|---|---|
| `fp8_per_tensor` (alias `fp8`) | linear layers and routed experts | `F8_E4M3_B128`, or `F8_E4M3_B32` for rows of no whole 128-value block | one per tensor (per expert), shared by a fused projection |
| `fp8_per_block` | linear layers and routed experts | `F8_E4M3_B128`, or `F8_E4M3_B32` for rows of no whole 128-value block | one per 128x128 tile (per expert), tiled over a fused projection's rows |
| `fp8_per_channel` (alias `ptpc_fp8`) | linear layers and routed experts | `F8_E4M3_B128`, or `F8_E4M3_B32` for rows of no whole 128-value block | one per output channel |
| `mxfp8` | linear layers and routed experts | `MXFP8` | one power-of-two (E8M0) per 32 values |
| `mxfp4` | linear layers and routed experts | `MXFP4` | one power-of-two (E8M0) per 32 values |
| `int8_per_channel_weight_only`, `experts_int8` | routed experts only | `I8_B128` | one per output channel |
| `nvfp4_per_token` (alias `nvfp4_online`) | routed experts only | `NVFP4` | E4M3 per 16 values, plus one global scale per expert over gate and up |
| `online` | the layers `--quantization-config` selects | per layer | per method |

The scale rules are vLLM's, so the weights quantize to the same codes vLLM computes. Like vLLM, these methods never quantize embeddings or LM heads. The MoE-only methods leave every linear layer unquantized, shared experts included.

**Row lengths.** An FP8 row whose length is a multiple of 32 but not of 128 (Qwen2-MoE's 192- and 160-wide down projections, for example) is stored as `F8_E4M3_B32`. Each run of 32 values carries the scale of the tensor, 128x128 tile or channel it lies in, so the codes stay vLLM's. A weight whose row length is not a whole number of the target's blocks stays in its stored type: 32 values for FP8, MXFP8 and MXFP4, 128 for int8, and 64 for NVFP4.

**MLA and gated-delta-net weights.** The vLLM methods quantize every weight vLLM quantizes. That includes the MLA `attn_kv_a_mqa`, `attn_k_b` and `attn_v_b`, and Qwen3.5's `ssm_beta`. The llama-quantize types (`q8_0`, ...) quantize them as well; llama-quantize keeps the MLA projections and `ssm_beta` float only on GLM5-next. NeuralFree stores `attn_k_b` transposed, one row per column of vLLM's `kv_b_proj`:

- Under `fp8_per_tensor` and `fp8_per_block`, those rows are quantized with the scales of the `kv_b_proj` tiles they cross.
- Under the other methods (per-channel FP8, int8, NVFP4, MXFP8/MXFP4 and the llama-quantize block types), `attn_k_b` holds the key rows of the quantized `kv_b_proj` dequantized back to the checkpoint's float type.

**Fused projections.** vLLM loads some projections into one fused weight and quantizes it with one set of scales. NeuralFree plans the members as separate tensors but gives them the same scales:

| vLLM layer | Members |
|---|---|
| `qkv_proj` | `attn_q`, `attn_k`, `attn_v` |
| `gate_up_proj` (the shared expert's included) | `ffn_gate`, `ffn_up` |
| `w13` (routed experts) | `ffn_gate_exps`, `ffn_up_exps` |
| `fused_qkv_a_proj` (MLA) | `attn_q_a`, `attn_kv_a_mqa` |
| `kv_b_proj` (MLA) | `attn_k_b`, `attn_v_b` |
| `in_proj_qkvz` (Qwen3.5, Qwen3-Next) | `attn_qkv`, `attn_gate` |
| `in_proj_ba` (Qwen3.5) | `ssm_beta`, `ssm_alpha` |

How each method shares scales across a fused projection:

- `fp8_per_tensor` uses the maximum over all members (per expert for `w13`).
- `fp8_per_block` tiles the rows in the order vLLM loads them. Each routed expert's gate and up start on a fresh 128-row tile, as vLLM pads them. Rows the converter reorders are tiled in the checkpoint's order: the rope permutation of LLaMA-family q and k, Qwen3.5's regrouped value heads, and Qwen3-Next's split `in_proj_qkvz`. A reorder that cannot be traced back to the checkpoint rows is tiled in the converted order, with a warning.
- `nvfp4_per_token` takes each expert's global scale over gate and up together.

The codes are bit-identical to vLLM's. A `kv_b_proj` with only one of its two parts quantized stays in its stored type, with a warning.

`--quant-online fp8` is vLLM's deprecated `--quantization fp8` for unquantized checkpoints, which means `fp8_per_tensor`.

#### Quantization config (`--quantization-config`)

`--quantization-config <json>` (property `neuralfree.quant.config`) is vLLM's `--quantization-config`. It is a JSON object with these members:

| Member | Meaning |
|---|---|
| `linear` | The method of the linear layers (attention, dense MLP and shared-expert projections). Give a shorthand above (its linear spec), a weight name (`fp8_per_tensor_static`, `fp8_per_block_static`, `fp8_per_channel_static`, `mxfp8`), or `{"weight": ...}`. A null weight leaves the layers unquantized. |
| `moe` | The method of the routed experts, in the same forms. It also accepts `int8_per_channel_static`, `int8_per_channel_weight_only` and `nvfp4_per_token`. |
| `ignore` | Layers left unquantized. Each entry is an exact layer name, a `re:` regular expression matched from the start of the name, or an fnmatch pattern (`*.o_proj`). |
| `targets` | `{pattern: shorthand}` per layer, with patterns as in `ignore`. Layers that match no pattern stay unquantized. It cannot be combined with `linear` or `moe`. |

Layers are named as vLLM names them:

- `model.layers.0.self_attn.qkv_proj` for q/k/v;
- `….mlp.gate_up_proj` for gate/up;
- `model.layers.0.mlp.experts` for the routed experts.

A pattern may match a fused layer's own name or each of its shards (`*.q_proj`, `*.k_proj` and `*.v_proj`).

The names start from the checkpoint's tensor names, after the renames vLLM's model class applies (its `hf_to_vllm_mapper`). A Qwen2.5-VL checkpoint's `model.layers.0.mlp.down_proj` is therefore the layer `language_model.model.layers.0.mlp.down_proj`. An InternLM2 `feed_forward.w1` is a shard of `feed_forward.gate_up_proj`. A weight vLLM's model drops keeps its checkpoint name.

The renames come from the architecture in the checkpoint's `config.json` (`architectures[0]`). They are generated from vLLM's sources into `src/main/resources/vllm/weights_mappers.tsv`; regenerate them with `./gradlew updateVllmTables -Pvllm=<vLLM checkout>`.

Cosmos3Edge builds its renames with code the table cannot hold, so `ignore` and `targets` are refused for it.

The following are refused with an error naming the tensor, as in vLLM:

- shards of one fused layer that match differently;
- a layer in both `ignore` and `targets`;
- a layer matched by several targets;
- a target whose shorthand has no spec for the layer kind;
- a target that matches an embedding.

NeuralFree quantizes weights only, so `activation` overrides are refused.

How the config combines with `--quant-online`:

- With a shorthand (`--quant-online fp8_per_block`), the members the config sets replace the shorthand's. An empty `ignore` or `targets` counts as unset.
- With `--quant-online online`, or with no `--quant-online` at all, the config alone decides.
- With a llama-quantize type (`q4_0`, ...), the config is refused.

**Checkpoint configurations.** A checkpoint whose `quantization_config.quant_method` names a shorthand above (for example `{"quant_method": "fp8_per_block"}`) is quantized at load without any flag, as vLLM does. One with `"quant_method": "online"` quantizes as its own `linear`, `moe`, `ignore` and `targets` select. A `--quantization-config` merges over it, and an explicit `--quant-online` method replaces it.

`mxfp4` and `mxfp8` are also checkpoint formats. On a checkpoint stored as MXFP4 (or MXFP8), `--quant-online mxfp4` (or `mxfp8`) without a `--quantization-config` names the checkpoint's own quantization and adds none. Already-quantized weights are never requantized. With an online method, only the modules a checkpoint stores as floats are quantized.

```
neuralfree --model ./Qwen3-8B --quant-online fp8_per_block
neuralfree --model ./Mixtral-8x7B --quant-online experts_int8
neuralfree --model ./Qwen3-30B-A3B --quant-online mxfp4
neuralfree --model ./Qwen3-30B-A3B --quantization-config '{"linear": "fp8_per_block", "moe": "mxfp4", "ignore": ["*.o_proj"]}'
neuralfree --model ./Qwen3-30B-A3B --quantization-config '{"targets": {"*.self_attn.*": "fp8_per_tensor", "re:.*\\.experts$": "mxfp4"}}'
neuralfree --model ./Qwen2.5-VL-7B-Instruct --quantization-config '{"linear": "fp8_per_block", "ignore": ["re:language_model\\.model\\.layers\\.0\\..*"]}'
```

### Model sampling defaults

A model can carry sampling defaults, and the engine applies them to every generation path (completion, speculative decoding and pooled modes) for every setting the caller did not give: llama.cpp does this for the `general.sampling.*` keys and vLLM for `generation_config`:

| Model key | Where it comes from | Defaults it provides |
|---|---|---|
| `general.sampling.temp`, `.top_k`, `.top_p`, `.min_p`, `.penalty_repeat`, `.penalty_last_n` | GGUF metadata | `temperature`, `top-k`, `top-p`, `min-p`, `repeat-penalty`, `repeat-last-n` |
| `neuralfree.sampling.presence_penalty`, `.frequency_penalty` | GGUF metadata | `presence-penalty`, `frequency-penalty` |
| `neuralfree.sampling.max_new_tokens` | GGUF metadata | `--max-tokens` |
| `generation_config.json` | Hugging Face checkpoint; `top_k`, `top_p`, `min_p`, `temperature` map like above, `repetition_penalty` → `penalty_repeat` | the same knobs |

Precedence: an explicit caller setting (CLI flag, builder setter or `neuralfree.*` property) wins; otherwise the model's value; otherwise the engine's default. `max_new_tokens` unwritten with no model default stays 512. Keys with no NF sampler stage (`mirostat*`, `xtc_*`, `sequence`) are logged once as ignored. `do_sample` is not carried, as vLLM also ignores it for defaults.

`--model-sampling-defaults false` (or `neuralfree.sampling.model.defaults=false`) turns the feature off: only the caller's settings and the engine defaults apply. It is on by default.

### Supported quantization formats

Quantized checkpoints are repacked at load into matching weight types; nothing on disk changes. Exactness classes:

- **(a)** exact — every weight value survives the repack bit-for-bit at the output of the weight kernel.
- **(a~)** exact except that an fp32/bf16 scale is rounded to fp16, or an fp16 fallback represents `-z*s` approximately. `--hf-quant-exact` switches every `(a~)` row to its exact alternative.
- **float fallback** — the checkpoint's own float dtype (the dtype of `weight_scale`/`scales`; else BF16), or F32 under `--hf-quant-exact`. Every fallback logs once per tensor group at WARN with the reason.

| Checkpoint format | Condition | Target | Class | Alternative |
|---|---|---|---|---|
| GPTQ / compressed-tensors WNA16 / AutoRound symmetric 4-bit | group % 32 == 0, or channel scales | Q4_0 | a (fp16 scales); CT bf16 scales exact when representable in fp16, else a~ | float fallback |
| same, symmetric 2-bit | group % 64 == 0 | Q2_0 | a | Q8_0 (exact) |
| same, symmetric 5-bit (CT) | group % 32 == 0, packing validated | Q5_0 | a | Q8_0 |
| same, symmetric 3/6/7-bit | group % 32 == 0 | Q8_0 | a | float fallback |
| same, symmetric 8-bit | group % 32 == 0 | Q8_0 | a | float fallback |
| asymmetric 4-bit (GPTQ v1 or v2, AWQ, CT `weight_zero_point`, HQQ) | group % 32 == 0 | Q4_1 | a~ | `--hf-quant-exact` → Q8_0 (exact) |
| asymmetric 5-bit (CT) | group % 32 == 0 | Q5_1 | a~ | `--hf-quant-exact` → Q8_0 |
| HQQ 8/3/2/1-bit | group % 32 == 0, integer zero point | Q8_0 | a | non-integer zero point: float fallback |
| GPTQ / CT act-order (`g_idx` not dense) | 2-D weights on scaled bases | permuted target plus a `.input_perm` sibling tensor | a | 3-D experts or other bases: float fallback in original order |
| MLX 4-bit | group % 32 == 0 | Q4_1 | a (exact) when bf16 scale → fp16 is exact, else a~ | `--hf-quant-exact` → F32 |
| MLX other bit widths | — | float fallback (`q*scale + bias`) | a~ | `--hf-quant-exact` → F32 |
| BitNet | — | TQ2_0 with `.scale = 1/weight_scale` | a on scaled bases; a~ elsewhere (scale rounded to fp16) | F32 |
| W8A8 int8 (CT int-quantized, SGLang `w8a8_int8`, Quark/AutoRound int8, bnb int8, SGLang `blockwise_int8`) | K % 128 == 0 | I8_B128 (per-channel or per-128-block scales carried in the block) | a | K % 32 == 0: Q8_0 (a~); `--hf-quant-exact` → F32 |
| FP8 E4M3, per-tensor or per-expert scale, on scaled bases | — | `F8_E4M3FN` (ggml's `F8_E4M3`, type 43, zero-copy) + `.scale` | a | — |
| FP8 E4M3, block 128×128 (`weight_scale_inv`), UE8M0 scales, per-channel, or DeepSeek-V4's `.scale` | K % 128 == 0 | F8_E4M3_B128 | a | float fallback |
| FP8 E5M2, per-tensor scale on scaled bases | — | F16 (code shifted) + `.scale` | a | float fallback |
| FP8 E4M3FNUZ / E5M2FNUZ | — | F32 | a | — |
| MXFP8 (CT `W8A8Mxfp8`, ModelOpt, `mxfp8`) | K % 32 == 0 | MXFP8 | a | — |
| MXFP6 E2M3 / E3M2 (Quark OCP-MX) | K % 32 == 0 | MXFP6_E2M3 / MXFP6_E3M2 (zero-copy codes, the E8M0 scale as the block header; a COMPAT export writes BF16) | a | an E8M0 exponent above 252 (E2M3) or 250 (E3M2), whose values overflow f32, or the NaN byte 0xFF: rejected, naming tensor, row and block |
| NVFP4 (CT, ModelOpt, `petit_nvfp4`, `modelopt_fp4`, fp_quant `nvfp4`, Quark two-level `weight`/`input_tensors`) | K % 64 == 0 | NVFP4 with UE4M3 sub-scales plus the global `.scale` | a | other bases: float fallback |
| torchao `Int8WeightOnlyConfig`, `Int8DynamicActivationInt8WeightConfig` | per-tensor, per-row or per-group scales; per-row symmetric activations for the dynamic config | I8_B128 (K % 128 == 0) or Q8_0, as W8A8 int8 | a | per-tensor, asymmetric or reduced-range dynamic activations load weight-only with a warning |
| torchao `Int4WeightOnlyConfig` (PLAIN, PRESHUFFLED, PLAIN_INT32, TILE_PACKED_TO_4D with ntile 8 or 16), `IntxWeightOnlyConfig` and `Int8DynamicActivationIntxWeightConfig` (int4) | group `% 32 == 0`; preshuffled needs N % 16 and K % 16 (bf16) or K % 32 (FP8 activations) | Q4_1 (`q*s + z` with offset codes) | a~ | `--hf-quant-exact` → Q8_0 |
| torchao `Float8DynamicActivationInt4WeightConfig`, `Int8StaticActivationInt8WeightConfig`, `IntxWeightOnlyConfig` and `Int8DynamicActivationIntxWeightConfig` (int1, int2, int3, int5, int6, int7, int8) | int4 group 128 with per-token FP8 activations; int1..int8 byte codes with per-channel or per-group scales and zero points, static (`act_quant_scale`, `act_quant_zero_point`) or per-token activations | Q4_1 + FP8 activations; I8_B128 (K % 128 == 0) or Q8_0, else the exact F32 fallback, as W8A8 int8 | a | codes or zero points that do not fit an 8-bit block: the exact F32 fallback; reduced-range static activations load weight-only with a warning |
| torchao `act_pre_scale` (`Int8Tensor`, `Int4Tensor`, `Int4TilePackedTo4dTensor`, `IntxUnpackedToInt8Tensor`), with or without activation quantization | one value per input column, or per expert of a stacked tensor | applied to the activation before its fake-quantization and the act-order gather (a runtime `<base>.input_pre_scale` F32 sibling); folded exactly into an F32 weight when the weight is weight-only | a | per-expert pre-scale on the dense GPU projection: refused by name; a weight outside a `blk.N` projection slot with both an activation scheme and a pre-scale: refused by name |
| torchao `Float8WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig` | E4M3/E5M2 weights; per-tensor, per-row or 1×group scales; per-row activations | F8_E4M3FN with `.scale`, F8_E4M3_B128, F16 (E5M2) or the float fallback, as FP8 | a | per-tensor dynamic activations load weight-only with a warning |
| torchao `MXDynamicActivationMXWeightConfig` | `float4_e2m1fn_x2` or `float8_e4m3fn` weights, `block_size` 32 | MXFP4 or MXFP8 with unswizzled E8M0 scales | a | E5M2 is refused by name |
| torchao `NVFP4DynamicActivationNVFP4WeightConfig` | swizzled or plain E4M3 block scales, block 16 | NVFP4 with the global weight multiplier and static activation scale | a | dynamic per-tensor activations load weight-only with a warning |
| MXFP4 (CT, Quark, `quark_mxfp4`, gpt-oss `*_blocks`/`*_scales`, DeepSeek-V4 experts, fp_quant `mxfp4`) | K % 32 == 0 | MXFP4 with E8M0 scales; gpt-oss gate/up rows de-interleaved | a | — |
| bnb NF4 / FP4 (`bitsandbytes`), including double quant | `K % 64 == 0`, blocksize a multiple of 64 | NF4 / FP4_BNB with the absmax replicated per 64-run | a (a~ when the checkpoint's `quant_map` differs from the codebook within 1e-4) | rows-crossing layouts or blocksize 32: float fallback; different `quant_map`: F32 under `--hf-quant-exact` |
| Hadamard input transforms (CT `transform_config`, fp_quant) | signed Sylvester transform, verified arch | the PRISM input-transform keys (block size, sign mode); carried through to the GPU | a | non-Hadamard type or unverified arch: named error |
| Unquantized F32/F16/BF16/F64 | — | the outtype chosen by `--hf-outtype` | a | — |

New weight types behind the ten extension ids (`F8_E4M3`, `F8_E4M3_B128`, `F8_E4M3_B32`, `F8_E5M2`, `MXFP8`, `MXFP6_E3M2`, `MXFP6_E2M3`, `I8_B128`, `NF4`, `FP4_BNB`) run natively on the CPU and the GPU, like the extension type details under [LLM-INFERENCE.md: Weight types](LLM-INFERENCE.md#weight-types).

ggml's own per-element FP8 type (`GGML_TYPE_F8_E4M3`, id 43, `F8_E4M3FN` here) loads too: one E4M3FN code per element, with the weight's `.scale` tensor applied as an output multiplier. Rows of any length compute on the CPU; on the GPU, rows that are multiples of 32 run the FP8 kernels (the hardware FP8 path on CUDA sm_89 and newer) and other widths are decoded to F32 at upload. `--hf-outtype auto` keeps per-tensor and per-expert FP8 E4M3 weights exactly as type 43 with a `<name>.scale` sibling, the files llama.cpp's converter writes, and sets `general.file_type` to `MOSTLY_F8_E4M3` (42); `f16`, `f32`, `bf16` and `q8_0` dequantize them instead. A COMPAT export (`GgufWriter`) writes these weights as type 43 unchanged, so the file loads in llama.cpp without conversion.

### Rejected formats

These are refused at load with a named error (`GgufFormatException` for a config value, `IOException` for a container problem), each saying what it is and why:

- `checkpoint_format: marlin` — vLLM's Marlin repacking is a runtime format, not a checkpoint layout;
- compressed-tensors `sparsity_config` (2:4 and bitmask sparsity) — sparse matmul runtime feature;
- `quant_method`: `aqlm`, `eetq`, `vptq`, `spqr`, `higgs`, `quanto` — vLLM rejects, deprecates or has removed each;
- `quant_method: torchao` — the listed configurations above load, including mslk's preshuffled int4 (the exact inverse of CUTLASS's reordering), XPU's `Int4PlainInt32Tensor`, `IntxUnpackedToInt8Tensor` at int1 through int8 (int4 packs into nibbles; the other widths keep their int8 byte codes through the int8 channel scheme), ROCm's 16-tile int4 packing, and `act_pre_scale` (applied to the activation before its quantization, or folded exactly into an F32 weight when the weight is weight-only). Refused by name: any other `quant_type`, an `intx_packing_format` other than `UNPACKED_TO_INT8`, NPU's `Int4PlainInt32Tensor` (float zero point), a per-expert `act_pre_scale` on the dense GPU projection (that path has no expert index), a weight that the runtime gives no `blk.N` projection slot but that carries both an activation scheme and an `act_pre_scale`, and a preshuffled shape off the 16-row atom;
- Quark multi-entry specs other than the NVFP4 two-level form (fp4 `per_group` with group size 16, a per-tensor `fp8_e4m3` global scale and a dynamic fp4 input entry, as vLLM's `_is_nvfp4`), which loads from `weight`, `weight_scale`, `weight_scale_2` and `input_scale_2`; the refusal names `weight` and `input_tensors`;
- compressed-tensors input transforms at `output`, `k_cache` or `q_attn` locations, and non-Hadamard transform types — NF only carries input transforms it can verify as signed Sylvester Hadamard ([Supported quantization formats](#supported-quantization-formats));
- fp_quant with `transform_init: gsr` — its transform is not a verifiable Hadamard;
- bnb checkpoints with `bnb_4bit_quant_storage` other than uint8 — a quantized-storage dtype NF cannot decode as written;
- `quant_method` values not among those the loader recognizes — always with the value named.

### Exporting to GGUF

`--export-gguf <out.gguf> [--export-mode <m>] [--export-overwrite]` writes the current model's GGUF view — the same metadata keys, tensor names, shapes and types the loader produced — and exits without running generation:

| Mode | Behaviour |
|---|---|
| `native` | Keep extension weight types and `neuralfree.*` keys, and mark the file `neuralfree.extension_types = true` |
| `compat` | Convert extension tensors to llama.cpp types through the compatibility converter, drop the extension marker; fails with a named error when a source keeps `.input_perm` or `.input_zero_point` tensors, which only serve NF |
| `compat-exact` | Like `compat`, but every lossy or approximate converted tensor is written as F32 instead |

The written file is a normal llama.cpp GGUF in the `compat` modes; `native` files are readable by NeuralFree and by anything that ignores unknown keys and types (as llama.cpp does).

### Supported architectures

An HF checkpoint is recognized by its `config.json` `architectures[0]` name. The names below — 305 of them, over 158 concrete GGUF architectures and 8 resolved at load time — are the TEXT rows of `src/main/resources/hf/archs.tsv` (generated from llama.cpp's converter registrations; the llama.cpp commit is recorded in the file header). They are grouped by the GGUF architecture each resolves to. The MMPROJ rows of the same file (the vision and audio tower classes) are not loadable model classes; `nftool export --mmproj` converts the ones listed under [Converting vision/audio projector towers](#converting-visionaudio-projector-towers---mmproj). A name absent below is refused with a message naming it.

| GGUF architecture | Names | `architectures[0]` values |
|---|---|---|
| `-` | 8 | `GraniteSpeechForConditionalGeneration`, `GraniteSpeechPlusForConditionalGeneration`, `MiniCPMV4_7ForConditionalGeneration`, `Ministral3ForCausalLM`, `Ministral3Model`, `Mistral3ForConditionalGeneration`, `Mistral3Model`, `NomicBertModel` |
| `afmoe` | 1 | `AfmoeForCausalLM` |
| `apertus` | 1 | `ApertusForCausalLM` |
| `arcee` | 1 | `ArceeForCausalLM` |
| `arctic` | 1 | `ArcticForCausalLM` |
| `arwkv7` | 1 | `RwkvHybridForCausalLM` |
| `baichuan` | 2 | `BaichuanForCausalLM`, `BaiChuanForCausalLM` |
| `bailingmoe` | 1 | `BailingMoeForCausalLM` |
| `bailingmoe2` | 3 | `BailingMoeV2ForCausalLM`, `modeling_sarvam_moe.SarvamMoEForCausalLM`, `SarvamMoEForCausalLM` |
| `bailingmoe3` | 2 | `BailingMoeV3ForCausalLM`, `BailingMoeV3VLForConditionalGeneration` |
| `beam` | 1 | `BeamForCausalLM` |
| `bert` | 11 | `BertForMaskedLM`, `BertForSequenceClassification`, `BertModel`, `CamembertModel`, `DistilBertForMaskedLM`, `DistilBertForSequenceClassification`, `DistilBertModel`, `RobertaForSequenceClassification`, `RobertaModel`, `XLMRobertaForSequenceClassification`, `XLMRobertaModel` |
| `bitnet` | 2 | `BitnetForCausalLM`, `BitNetForCausalLM` |
| `bloom` | 2 | `BloomForCausalLM`, `BloomModel` |
| `chameleon` | 2 | `ChameleonForCausalLM`, `ChameleonForConditionalGeneration` |
| `chatglm` | 3 | `ChatGLMForConditionalGeneration`, `ChatGLMModel`, `GlmForCausalLM` |
| `clef` | 1 | `ClefModel` |
| `codeshell` | 1 | `CodeShellForCausalLM` |
| `cogvlm` | 1 | `CogVLMForCausalLM` |
| `cohere2` | 1 | `Cohere2ForCausalLM` |
| `cohere2moe` | 1 | `Cohere2MoeForCausalLM` |
| `command-r` | 1 | `CohereForCausalLM` |
| `dbrx` | 1 | `DbrxForCausalLM` |
| `deci` | 1 | `DeciLMForCausalLM` |
| `deepseek` | 1 | `DeepseekForCausalLM` |
| `deepseek2` | 9 | `DeepseekOCRForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `Glm4MoeLiteForCausalLM`, `KimiK25ForConditionalGeneration`, `KimiVLForConditionalGeneration`, `UnlimitedOCRForCausalLM`, `YoutuForCausalLM`, `YoutuVLForConditionalGeneration` |
| `deepseek32` | 1 | `DeepseekV32ForCausalLM` |
| `deepseek4` | 1 | `DeepseekV4ForCausalLM` |
| `deepseek41` | 1 | `DeepseekV41ForCausalLM` |
| `dflash` | 11 | `DeepseekV4DSparkModel`, `DFlash2DraftModel`, `DFlashDraftModel`, `DSparkDraftModel`, `DSparkSpeculator`, `Gemma4DSparkModel`, `Lfm2DSparkDraftModel`, `LiLiCorrDraftModel`, `LingDSparkModel`, `MuseGlimmerAssistantModel`, `Qwen3DSparkModel` |
| `dots1` | 1 | `Dots1ForCausalLM` |
| `dots3note` | 3 | `Dots3NoteForCausalLM`, `Dots3NoteForConditionalGeneration`, `Dots3NoteTextForCausalLM` |
| `dream` | 1 | `DreamModel` |
| `ernie4_5` | 2 | `Ernie4_5_ForCausalLM`, `Ernie4_5ForCausalLM` |
| `ernie4_5-moe` | 1 | `Ernie4_5_MoeForCausalLM` |
| `eurobert` | 2 | `EuroBertModel`, `JinaEmbeddingsV5Model` |
| `exaone` | 1 | `ExaoneForCausalLM` |
| `exaone4` | 2 | `Exaone4_5_ForConditionalGeneration`, `Exaone4ForCausalLM` |
| `exaone-moe` | 2 | `ExaoneMoeForCausalLM`, `ExaoneMoEForCausalLM` |
| `falcon` | 2 | `FalconForCausalLM`, `RWForCausalLM` |
| `falcon-h1` | 1 | `FalconH1ForCausalLM` |
| `flint` | 1 | `FlintForCausalLM` |
| `gemma` | 1 | `GemmaForCausalLM` |
| `gemma2` | 1 | `Gemma2ForCausalLM` |
| `gemma3` | 2 | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` |
| `gemma3n` | 2 | `Gemma3nForCausalLM`, `Gemma3nForConditionalGeneration` |
| `gemma4` | 3 | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration` |
| `gemma4-assistant` | 2 | `Gemma4AssistantForCausalLM`, `Gemma4UnifiedAssistantForCausalLM` |
| `gemma-embedding` | 1 | `Gemma3TextModel` |
| `gemma-embedding2` | 1 | `EmbeddingGemma2Model` |
| `glm4` | 3 | `Glm4ForCausalLM`, `Glm4vForConditionalGeneration`, `GlmOcrForConditionalGeneration` |
| `glm4moe` | 3 | `Glm4MoeForCausalLM`, `Glm4vMoeForConditionalGeneration`, `SolarOpenForCausalLM` |
| `glm5-next` | 1 | `Glm5NextForConditionalGeneration` |
| `glm-dsa` | 1 | `GlmMoeDsaForCausalLM` |
| `gpt2` | 2 | `GPT2LMHeadModel`, `RuGPT3XLForCausalLM` |
| `gptneox` | 1 | `GPTNeoXForCausalLM` |
| `gpt-oss` | 1 | `GptOssForCausalLM` |
| `granite` | 1 | `GraniteForCausalLM` |
| `granitehybrid` | 2 | `BambaForCausalLM`, `GraniteMoeHybridForCausalLM` |
| `granitemoe` | 2 | `GraniteMoeForCausalLM`, `GraniteMoeSharedForCausalLM` |
| `granite_swa` | 2 | `GraniteMoeSWAForCausalLM`, `GraniteSWAForCausalLM` |
| `graniteswitch` | 1 | `GraniteSwitchForCausalLM` |
| `grok` | 2 | `Grok1ForCausalLM`, `GrokForCausalLM` |
| `grovemoe` | 2 | `GroveMoeForCausalLM`, `modeling_grove_moe.GroveMoeForCausalLM` |
| `hrm_text` | 1 | `HrmTextForCausalLM` |
| `hunyuan-dense` | 1 | `HunYuanDenseV1ForCausalLM` |
| `hunyuan-moe` | 1 | `HunYuanMoEV1ForCausalLM` |
| `hunyuan_vl` | 1 | `HunYuanVLForConditionalGeneration` |
| `hy_v3` | 1 | `HYV3ForCausalLM` |
| `hy_v4` | 1 | `HYV4ForCausalLM` |
| `internlm2` | 1 | `InternLM2ForCausalLM` |
| `jais` | 1 | `JAISLMHeadModel` |
| `jais2` | 1 | `Jais2ForCausalLM` |
| `jamba` | 1 | `JambaForCausalLM` |
| `jina-bert-v2` | 2 | `JinaBertForMaskedLM`, `JinaBertModel` |
| `k2-horizon` | 1 | `K2HorizonForCausalLM` |
| `kimi-k3` | 1 | `KimiK3ForConditionalGeneration` |
| `kimi-linear` | 2 | `KimiLinearForCausalLM`, `KimiLinearModel` |
| `kolibri1` | 1 | `Kolibri1ForCausalLM` |
| `laguna` | 1 | `LagunaForCausalLM` |
| `lfm2` | 8 | `D1Model`, `D1OmniModel`, `Lfm25AudioTokenizer`, `Lfm2BidirectionalForMaskedLM`, `Lfm2BidirectionalModel`, `Lfm2ForCausalLM`, `LFM2ForCausalLM`, `Lfm2Model` |
| `lfm2moe` | 1 | `Lfm2MoeForCausalLM` |
| `lfm3` | 4 | `Lfm3D1Model`, `Lfm3ForCausalLM`, `Lfm3OmniForConditionalGeneration`, `Lfm3VlForConditionalGeneration` |
| `limite` | 1 | `LimiteForCausalLM` |
| `llada` | 1 | `LLaDAModelLM` |
| `llada-moe` | 2 | `LLaDAMoEModel`, `LLaDAMoEModelLM` |
| `llama` | 17 | `Eagle3DraftModel`, `Eagle3LlamaForCausalLM`, `Eagle3Speculator`, `InternLM3ForCausalLM`, `IQuestCoderForCausalLM`, `JanusForConditionalGeneration`, `LlamaForCausalLM`, `LLaMAForCausalLM`, `LlamaForCausalLMEagle3`, `LlamaModel`, `LlavaForConditionalGeneration`, `MistralForCausalLM`, `MixtralForCausalLM`, `Sarashina2VisionForCausalLM`, `UltravoxModel`, `VLlama3ForCausalLM`, `VoxtralForConditionalGeneration` |
| `llama4` | 2 | `Llama4ForCausalLM`, `Llama4ForConditionalGeneration` |
| `llama-embed` | 1 | `LlamaBidirectionalModel` |
| `maincoder` | 1 | `MaincoderForCausalLM` |
| `mamba` | 3 | `FalconMambaForCausalLM`, `MambaForCausalLM`, `MambaLMHeadModel` |
| `mamba2` | 1 | `Mamba2ForCausalLM` |
| `maple` | 1 | `MapleForCausalLM` |
| `mellum` | 1 | `MellumForCausalLM` |
| `mimo2` | 2 | `MiMoV2FlashForCausalLM`, `MiMoV2ForCausalLM` |
| `minicpm` | 1 | `MiniCPMForCausalLM` |
| `minicpm3` | 1 | `MiniCPM3ForCausalLM` |
| `minimax-01` | 2 | `MiniMaxM1ForCausalLM`, `MiniMaxText01ForCausalLM` |
| `minimax-m2` | 1 | `MiniMaxM2ForCausalLM` |
| `minimax-m3` | 2 | `MiniMaxM3SparseForCausalLM`, `MiniMaxM3SparseForConditionalGeneration` |
| `mistral3` | 1 | `VoxtralRealtimeForConditionalGeneration` |
| `mistral-large4` | 1 | `MistralLarge4ForConditionalGeneration` |
| `modern-bert` | 4 | `ModernBertDecisionModel`, `ModernBertForMaskedLM`, `ModernBertForSequenceClassification`, `ModernBertModel` |
| `mpt` | 1 | `MPTForCausalLM` |
| `muse-glimmer` | 1 | `MuseGlimmerForConditionalGeneration` |
| `muse-spark` | 1 | `MuseSparkForConditionalGeneration` |
| `nanbeige` | 1 | `NanbeigeForCausalLM` |
| `nemotron` | 1 | `NemotronForCausalLM` |
| `nemotron_h` | 1 | `NemotronHForCausalLM` |
| `nemotron_h_moe` | 1 | `NemotronHPuzzleForCausalLM` |
| `neo-bert` | 3 | `NeoBERT`, `NeoBERTForSequenceClassification`, `NeoBERTLMHead` |
| `olmo` | 2 | `OlmoForCausalLM`, `OLMoForCausalLM` |
| `olmo2` | 2 | `Olmo2ForCausalLM`, `Olmo3ForCausalLM` |
| `olmoe` | 1 | `OlmoeForCausalLM` |
| `openelm` | 1 | `OpenELMForCausalLM` |
| `orion` | 1 | `OrionForCausalLM` |
| `paddleocr` | 1 | `PaddleOCRVLForConditionalGeneration` |
| `pangu-embedded` | 1 | `PanguEmbeddedForCausalLM` |
| `phi2` | 1 | `PhiForCausalLM` |
| `phi3` | 2 | `Phi3ForCausalLM`, `Phi4ForCausalLMV` |
| `phimoe` | 1 | `PhiMoEForCausalLM` |
| `plamo` | 1 | `PlamoForCausalLM` |
| `plamo2` | 2 | `Plamo2ForCausalLM`, `PLaMo2ForCausalLM` |
| `plamo3` | 2 | `Plamo3ForCausalLM`, `PLaMo3ForCausalLM` |
| `plm` | 1 | `PLMForCausalLM` |
| `pockettts` | 1 | `PocketTTSModel` |
| `qwen` | 1 | `QWenLMHeadModel` |
| `qwen2` | 6 | `AudioFlamingo3ForConditionalGeneration`, `DotsOCRForCausalLM`, `KORMoForCausalLM`, `Qwen2AudioForConditionalGeneration`, `Qwen2ForCausalLM`, `Qwen2Model` |
| `qwen2moe` | 1 | `Qwen2MoeForCausalLM` |
| `qwen2vl` | 4 | `Qwen2_5OmniModel`, `Qwen2_5_VLForConditionalGeneration`, `Qwen2VLForConditionalGeneration`, `Qwen2VLModel` |
| `qwen3` | 4 | `Qwen3ForCausalLM`, `Qwen3Model`, `SopranoTextModel`, `StepVLForConditionalGeneration` |
| `qwen35` | 13 | `H2OLightningModel`, `KevModel`, `LevModel`, `MiniCPMOcrForConditionalGeneration`, `MiniCPMOCRForSequenceClassification`, `MiniCPMV4_6ForConditionalGeneration`, `NimbleModel`, `OpenJevModel`, `PplxDeciderModel`, `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5TextModel`, `VllmD3Model` |
| `qwen35moe` | 2 | `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` |
| `qwen3moe` | 1 | `Qwen3MoeForCausalLM` |
| `qwen3next` | 1 | `Qwen3NextForCausalLM` |
| `qwen3tts` | 1 | `Qwen3TTSForConditionalGeneration` |
| `qwen3vl` | 2 | `Qwen3ASRForConditionalGeneration`, `Qwen3VLForConditionalGeneration` |
| `qwen3vlmoe` | 2 | `Qwen3OmniMoeForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration` |
| `qwen4exp` | 2 | `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration` |
| `refact` | 1 | `GPTRefactForCausalLM` |
| `rnd1` | 1 | `RND1` |
| `rwkv6` | 1 | `Rwkv6ForCausalLM` |
| `rwkv6qwen2` | 1 | `RWKV6Qwen2ForCausalLM` |
| `rwkv7` | 2 | `Rwkv7ForCausalLM`, `RWKV7ForCausalLM` |
| `seed_oss` | 1 | `SeedOssForCausalLM` |
| `sensenova_u1` | 1 | `NEOChatModel` |
| `smallthinker` | 1 | `SmallThinkerForCausalLM` |
| `smollm3` | 1 | `SmolLM3ForCausalLM` |
| `spark2_5` | 1 | `Spark2_5ForCausalLM` |
| `stablelm` | 3 | `LlavaStableLMEpochForCausalLM`, `StableLMEpochForCausalLM`, `StableLmForCausalLM` |
| `starcoder` | 1 | `GPTBigCodeForCausalLM` |
| `starcoder2` | 1 | `Starcoder2ForCausalLM` |
| `step35` | 3 | `Step3p5ForCausalLM`, `Step3p5MTP`, `Step3p7ForConditionalGeneration` |
| `t5` | 5 | `MT5ForConditionalGeneration`, `T5ForConditionalGeneration`, `T5WithLMHeadModel`, `UMT5ForConditionalGeneration`, `UMT5Model` |
| `t5encoder` | 1 | `T5EncoderModel` |
| `talkie` | 1 | `TalkieForCausalLM` |
| `vllm-decision2` | 1 | `VllmDecision2Model` |
| `wavtokenizer-dec` | 1 | `WavTokenizerDec` |
| `xverse` | 1 | `XverseForCausalLM` |

Notes on recent rows: LFM3 (`Lfm3VlForConditionalGeneration` and `Lfm3OmniForConditionalGeneration` convert their text side; the LFM3-VL vision tower and the LFM3 audio and Omni projectors convert with `--mmproj`) and `Lfm3D1Model` d1 checkpoints record non-causal attention, the decision type and the calibration temperature. Mistral Large 4 checkpoints whose `quantization` block uses `qscheme_weight = BLOCK` are dequantized with the declared block shape. A DeepSeek-V4.1 checkpoint that declares `engram_layer_ids` converts its Engram tables: `per_layer_token_embd` holds every Engram layer's rows in layer order, dequantized from the checkpoint's E4M3 rows (one E8M0 scale per 32 elements) to BF16 (F32 with `--outtype f32`); the `deepseek41.ple.*` keys carry the bucket moduli, row offsets and the numpy-PCG64 hash multipliers, reproduced bit for bit; `deepseek41.engram.token_map` and `deepseek41.engram.pad_token_id` carry the normalizer's compressed token ids and the pad id. The token map needs the checkpoint's `tokenizer.json`; an Engram checkpoint without one is refused by name. The DeepSeek-V4/V4.1 vision towers and the DeepSeek-OCR, OCR2 and Unlimited-OCR SAM+CLIP towers convert with `--mmproj`.

**DeepSeek-V4.1 (`deepseek41`)** runs on the CPU backend and on the GPU backends (OpenCL, CUDA, Vulkan), over contiguous and paged caches: the compressed KV cache with its source-layer sharing, the indexer with the two-level candidate selection, the Engram n-gram embeddings (the n-gram hash runs on the host, the gather, projection, norms and gated add on the device) and the lagging hyper-connection mix. NextN/MTP drafting over a V4.1 target is refused by name: its draft is the separate DSpark DFLASH model (`--draft`). Beyond the DeepSeek-V4 fields the file carries `deepseek41.attention.compress_ratios` (only `0` window-only, `2` selected and `1` dense), `attention.kv_source_layers` and `attention.indexer.source_layers` (compressed caches and indexer top-k are shared across layers: only source layers keep a compressed cache and run the compressor), `attention.indexer.top_k > 0`, optionally the two-level candidate selection (`indexer.candidate_source_layer`, `candidate_top_k_blocks`, `candidate_block_size`, all three), and for Engram models `ple.layers` with `ple.ngram_size`, `ple.heads_per_ngram`, `embedding_length_per_layer_input`, `ple.layer_multipliers`, `ple.head_offsets`, `ple.head_vocab_sizes`, `engram.pad_token_id` and `engram.token_map`. An inconsistent file is refused naming the key: a source layer cannot have ratio 0, every compressed layer needs a KV source and an indexer source at or before it with its ratio, an indexer source cannot precede its KV source, and the Engram arrays must have their exact lengths and span `n_embd`.

**Limite (`limite`).** A pre-norm transformer with token value embeddings added to V (`value_embd`, optionally gated per KV head by `attn_ve_gate`), exclusive self attention (`attn_xsa_alpha`, `attention.xsa_epsilon`) that removes each head's projection on the token's own value, an optional per-head output gate (`attn_gate`), learned residual scales (`attn_resid_scale`, `ffn_resid_scale`), MUDD dense connections over earlier layers' inputs (`mudd.tap_count`, `mudd.feed_forward_length`, `mudd.tap_indices`, `mudd_down`, `mudd_up`) and a sigmoid logit cap (`final_logit_sigmoid_capping`). Full-attention layers have no positional encoding; the sliding-window layers rotate. It runs on the CPU and on the GPU backends, where the value embeddings, the exclusive self attention, the residual scales and the MUDD mixes are device kernels (`LLM_AttnValueEmbd`, `LLM_AttnXsa`, `LLM_MuddMix`) for contiguous prefill, decode and paged multi-sequence batches. Two GPU splits refuse by name: `--split-mode tensor` does not split the value embeddings or the exclusive self attention, and `--split-mode layer` does not split a model with MUDD connections, since each mix reads earlier layers' inputs that only their own device holds.

**Mistral Large 4 (`mistral-large4`).** NeuralFree runs Mistral Large 4 GGUFs (`mistral-large4`, produced by llama.cpp's `conversion/mistral4.py`) on the CPU and on the GPU backends.

The DSA sparse-MLA mixer supports the architecture's attention form:

- direct `attn_q` (the model has no q-LoRA rank), with the indexer query built from the layer input;
- SSMax per-head query temperatures: `attn_ssmax` scales the query of every head by `s[h] * log(pos + 1) + b[h]`, and `indexer.ssmax` scales the indexer queries the same way before their rotation;
- indexer queries from the layer-normed input (`indexer.q_norm` weight and bias over the attention-normed input, then `indexer.attn_q_b`);
- LeRoPE: the learned `rope_freqs` factors are applied to the main and the indexer rotations;
- mixed rope patterns (`attention.rope_pattern`): NoPE layers rotate nothing, absorb the whole query head and zero-pad the rope dims of their query and latent row while keeping the roped cache layout;
- KV sharing (`attention.kv_source_map`): a layer without its own KV cache reads the latent rows of its source layer and keeps its own indexer keys; it must point at an earlier layer with the same cache width.

The tensors `attn_ssmax.weight`/`.bias`, `indexer.ssmax.weight`/`.bias` and `indexer.q_norm.weight`/`.bias` are required; a file carrying them without the corresponding spec flag is refused by name instead of silently ignoring them. Shared-KV layers need no `attn_kv_a_mqa`/`attn_kv_a_norm`; if the file does carry them, their shapes are still checked.

### vLLM architecture names

Hugging Face checkpoints whose `config.json` names one of vLLM's architecture classes load directly:

| `architectures` | Loads as | Notes |
|---|---|---|
| `MistralModel`, `Gemma2Model`, `DeepseekV3BidirectionalModel` | llama / gemma2 / deepseek2 | Embedding checkpoints; tensor names with or without the `model.` prefix. Pooling comes from `modules.json` / `embedding_config.json`, else last-token. `is_causal: false` makes attention bidirectional. |
| `TeleChat2ForCausalLM` | llama | `transformer.h.*` names and the fused `key_value` projection are converted. |
| `TeleChat3ForCausalLM` | llama | `telechat3-yarn` RoPE scaling with its own attention factor. |
| `TeleFLMForCausalLM` | granite | muP `input_mult` / `output_mult` / `mup_scale_factor` become the embedding and logit scales; `rope_type: llama3` is not supported. |
| `GPTJForCausalLM` | gptj | GPT-J has no llama.cpp converter, so this is NeuralFree's own `gptj` layout: `gelu_new`, partial adjacent-pair rotary over the first `rotary_dim` dims, parallel residual, biased MLP and an untied biased output head. `tie_word_embeddings: true`, a non-`gelu_new` activation, `rotary: false`, an invalid `rotary_dim` and RoPE scaling are refused by name. |
| `OPTForCausalLM` | opt | OPT has no llama.cpp converter, so this is NeuralFree's own `opt` layout: learned position embeddings offset by two padding rows, separate biased q/k/v projections, an ungated ReLU MLP, `project_in`/`project_out` when `word_embed_proj_dim != hidden_size`, and pre-LN blocks or the 350M post-LN placement. A non-`relu` activation, non-affine LayerNorms and a checkpoint whose final norm or projection pair disagrees with its configuration are refused by name. |
| `Rnj1ForCausalLM` | rnj1 | NeuralFree's own `rnj1` layout: Gemma-style sandwich norms, per-head Q/K RMS norms, `sqrt(hidden_size)` embedding scale and gated `gelu_pytorch_tanh` MLP, with `chunked_attention` layers that see their own chunk and the one before it. |
| `AXK1ForCausalLM` | axk1 | NeuralFree's own `axk1` layout over the DeepSeek-2 MLA/MoE graph: optional Q LoRA, the grouped sigmoid router, `llama_4_scaling` of the queries, and a post-MLP norm on the sparse layers only. |
| `OlmoHybridForCausalLM` | olmo-hybrid | NeuralFree's own `olmo-hybrid` layout: Gated DeltaNet linear-attention layers alternating with OLMo2 post-norm attention layers and a dense SwiGLU MLP; released checkpoints run attention without RoPE. |
| `Zamba2ForCausalLM` | zamba2 | NeuralFree's own `zamba2` layout: Mamba-2 layers interleaved with hybrid layers that run one of `num_mem_blocks` shared transformer blocks, stored once and reused at several depths with a per-use LoRA adapter and KV cache. |
| `LongcatFlashForCausalLM`, `LongCatFlashMTPModel` | longcat-flash | NeuralFree's own `longcat-flash` layout: MLA attention, two physical blocks per layer, and a shortcut MoE with zero-compute experts; the MTP layer becomes the NextN draft block. |
| `LongcatFlashNgramForCausalLM` | longcat-flash-ngram | The `longcat-flash` layout with the fused n-gram input embedding; these checkpoints carry no MTP layer. |
| `AliceAIForCausalLM` | alice-ai | NeuralFree's own `alice-ai` layout: KDA linear-attention layers with low-rank decay and output gates alternating with Qwen3-Next gated attention, a sigmoid-routed MoE with a gated shared expert, and block-grouped cross-layer attention residuals. |
| `BailingMoeV2_5ForCausalLM` | bailingmoe-v25 | NeuralFree's own `bailingmoe-v25` layout: lightning linear-attention layers alternating with MLA attention, leading dense layers, then routed and shared experts. MTP layers are not loaded. |
| `InklingForCausalLM`, `InklingForConditionalGeneration` | inkling | NeuralFree's own `inkling` layout: NoPE attention with a per-query relative-logit bias, per-head Q/K RMS norms, K/V and residual-stream short convolutions, a SwiGLU MLP and a muP logit divisor. Only the text backbone converts; the conditional-generation checkpoint's multimodal towers are dropped. |
| `DoryForCausalLM` | dory | NeuralFree's own `dory` layout from vLLM `dory.py`: nGPT justnorm residual helpers, alternating attention-only and MLP-only blocks, slerp residuals with per-layer `attn_alpha`/`mlp_alpha`, and the block `[n_input_layers, n_input_layers + n_recurrent_layers)` repeated `n_recurrent_loops` times with one KV cache per pass. Learned `sqk`, `s_gate`, `suv_gate`, `suv_up` and `lm_head.logit_scale` are folded into the projections and `output_logit_scale` at conversion. Tied embeddings, attention or MLP biases, a non-silu activation, a `recurrent_kv_cache_mode` other than `per_loop`, invalid layer types or repeat ranges, and RoPE scaling or partial rotation are refused by name. |
| `IQuestLoopCoderForCausalLM` | iquest-loopcoder | NeuralFree's own `iquest-loopcoder` layout from vLLM `iquest_loopcoder.py`: a Llama block repeated `loop_num` times, where pass 0 writes the global caches and every later pass gates a read-only full causal attention over pass 0's cache against a sliding-window attention over its own per-pass cache, with the per-head gate `sigmoid(gate_projections.<l>.gate_proj(q_head) + bias_head)`; a final RMS norm precedes the head. A non-silu activation, `is_causal: false`, attention or MLP biases, `loop_num < 1`, `loop_window_size <= 0`, an invalid head geometry and RoPE scaling are refused by name. |
| `InternS1ProForConditionalGeneration` | interns1-pro | NeuralFree's own `interns1-pro` layout from vLLM `interns1_pro.py`: the Qwen3-VL MoE text backbone with a grouped router and the Fourier position embedding (FoPE); the vision tower converts through `--mmproj` as Qwen3-VL's, and the time-series tower is refused. |
| `SolarForCausalLM` | llama | The base Solar checkpoints are a plain Llama decoder, so the native layout is used. The Solar-pro graph additions are refused by name: a non-empty `bskcn_1`…`bskcn_4` block-skip list and a `logit_scale` other than 1. |
| `HyperCLOVAXForCausalLM` | hyperclovax | HyperCLOVAX has no llama.cpp converter, so this is NeuralFree's own `hyperclovax` layout: NEOX rotary, a post norm after each sub-layer output (sandwich norms), a scaled residual and the MuP multipliers (`attention_multiplier`, `embedding_multiplier`, `residual_multiplier`, `logits_scaling`). A non-silu `hidden_act`, `use_post_norm: false`, `attention_bias: true`, non-positive multipliers and RoPE scaling are refused by name. |
| `DomynEdgeForCausalLM` | domyn-edge | DomynEdge has no llama.cpp converter, so this is NeuralFree's own `domyn-edge` layout: sandwich norms, per-head RMS norms on Q and K (Q's norm is weightless in the checkpoint), a per-layer head width, a per-layer RoPE switch and one shared causal sliding window whose width counts the query itself. Non-causal or differing windows, differing head sizes inside one layer class, `position_layers` values other than `rope`/`nope` and `attention_bias: true` are refused by name. |
| `Param2MoEForCausalLM` | dots1 | The `dots1` layout matches Param2MoE's dense lead, sigmoid router with a per-expert correction bias, normalized top-k, routed weight scale and shared expert. The checkpoint's fused `query_key_value` is split, `word_embeddings` becomes the token table (which also fills `output.weight` when the head is tied) and the router bias is zero-meaned at conversion, as vLLM does. Grouped routing (`n_group`/`topk_group` above 1), NextN heads, biased attention projections, `use_qk_norm: false`, a non-silu activation and a partial rotary width are refused by name. |
| `Step1ForCausalLM` | step1 | NeuralFree's own `step1` layout: no rotary position, vLLM's square-root ALiBi with the fixed maximum bias 8, RMSNorm and a gated SiLU FFN. q/k/v, the output projection and the MLP carry biases when `attention_bias` / `mlp_bias` set them. A non-`silu` activation is refused by name. |
| `Step3p5MTP` | step35 | vLLM's draft-model name of the Step3.5 checkpoint: the same layout, whose MTP layers are the appended NextN blocks drafted with `--mtp`. |
| `SarvamMLAForCausalLM` | deepseek2 | DeepSeek-V2-style MLA with routed and one shared expert. The released checkpoints omit the routing keys; one expert group, sigmoid scoring, top-k renormalization, one shared expert, a dense lead and a routing scale of 2.5 are assumed. The router bias is centered before it is written. `moe_layer_freq` other than 1, zero shared experts and a non-silu activation are refused by name. |
| `PanguProMoEV2ForCausalLM`, `PanguUltraMoEForCausalLM` | deepseek2 or pangu-moe | Trunk selection is unchanged (MLA geometry loads as `deepseek2`, otherwise `pangu-moe`). `sandwich_norm` maps `post_attention_layernorm`, `pre_mlp_layernorm` and `post_mlp_layernorm` onto the SANDWICH branch norms. `interleaved_sliding_window` (integer or per-layer list) becomes the per-layer sliding window, so layers may name different window sizes (0 = full attention). `param_sink_number > 0` on the non-MLA trunk loads vLLM's static sink attention: `qk_nope_dim` + `qk_rope_dim` key heads, `v_channels` value heads, the `k_layernorm` K norm and the learned `param_sink_key`/`param_sink_value` tokens become `attn_k_norm`, `attn_sink_k` and `attn_sink_v`, with NORM rotary over the first `qk_rope_dim` values. Non-causal sink configurations, missing sink geometry and non-integer `interleaved_sliding_window` entries are refused by name. |
| `Step3TextForCausalLM` | step3text | NeuralFree's own `step3text` layout from vLLM `step3_text.py`: a shared query projection `attn_q_a` RMS-normalized by `attn_q_a_norm` and widened per head by `attn_q_b` before NEOX RoPE, a single key/value head, a dense lead and routed-plus-shared-expert layers (`ffn_gate_inp`, `ffn_*_exps`, `ffn_*_shexp`). Softmax routing, renormalized only when `norm_expert_weight` is set. A non-`silu` activation or more than one key/value head is refused by name. |
| `Step5ForCausalLM`, `Step5MTP` | step5 | Converts to NeuralFree's own `step5` layout (`--mtp` adds the draft block, and an MTP-only export keeps it alone). |
| `MiMoForCausalLM`, `MiMoMTPModel` | qwen2 | The multi-token-prediction layers (`model.mtp_layers.*`) become the appended NextN blocks ([MiMo MTP](#speculative-decoding)); `--hf-no-mtp` drops them. `MiMoMTPModel` is vLLM's name for the same checkpoint served as its own MTP draft. |
| `MistralLarge3ForCausalLM` | deepseek2 | Native Mistral tensor names and FP8 `qscale_*` scales are accepted; `llama_4_scaling` sets the attention temperature. |
| `Glm5NextForCausalLM`, `MuseGlimmerForCausalLM` | as their `*ForConditionalGeneration` | Text-only names of the same models. |
| `ClefFlashForJointSchema` | clef | The joint head config comes from `joint_head_config.json` or an inline `joint_head_config`. |
| `D3ForSequenceClassification` | as `VllmD3Model` | The decision config comes from `decision_config.json` or an inline `decision_config`. |

**GPT-J (`gptj`).** `GPTJForCausalLM` checkpoints load directly from Hugging Face. The exported GGUF layout is `token_embd`, `output_norm.{weight,bias}`, `output.{weight,bias}` and, per block, `attn_norm.{weight,bias}`, `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_output.weight`, `ffn_up.{weight,bias}` and `ffn_down.{weight,bias}`; `gptj.rope.dimension_count` carries `rotary_dim` and `gptj.attention.layer_norm_epsilon` the LayerNorm epsilon. The attention projections carry no bias; the MLP and the output head do. `nftool export` writes this layout and the exported file loads back through the GGUF path. `GPTJForCausalLM` appears in `nftool export --print-supported-models`.

**OPT (`opt`).** `OPTForCausalLM` checkpoints load directly from Hugging Face. The exported GGUF layout is `token_embd.weight`, `position_embd.weight`, `project_in.weight`/`project_out.weight` when the word width differs from `hidden_size`, `output_norm.{weight,bias}` when the decoder has a final LayerNorm, `output.weight` (or the tied token table), and per block `attn_norm.{weight,bias}`/`ffn_norm.{weight,bias}` (`attn_output_norm.{weight,bias}`/`layer_output_norm.{weight,bias}` for the 350M post-LN variant), `attn_q|attn_k|attn_v|attn_output.{weight,bias}`, `ffn_up.{weight,bias}` and `ffn_down.{weight,bias}`. Sequence positions are looked up in `position_embd` with OPT's `+2` offset and are bounds-checked against the table, including its two padding rows. `opt.context_length` carries `max_position_embeddings` and `opt.attention.layer_norm_epsilon` the LayerNorm epsilon. The causal-mask buffers stored by older checkpoints are dropped. `nftool export` writes this layout and the exported file loads back through the GGUF path. `OPTForCausalLM` appears in `nftool export --print-supported-models`.

**RNJ-1 (`rnj1`).** Layers named `chunked_attention` in `layer_types` attend within chunks of `sliding_window / 2` positions: a query sees its own aligned chunk and the one immediately before it, and the layer uses the RoPE base under `rope_parameters.chunked_attention` (else the global one). `full_attention` layers see the whole context with the global base. Embeddings are scaled by `sqrt(hidden_size)`, the attention scale is `query_pre_attn_scalar ** -0.5`, every sub-layer has a sandwich norm, and `final_logit_softcapping` applies when present; the checkpoint's zero-centered norms are shifted to plain RMS scales at conversion. Refused by name: `attention_bias`, an activation other than `gelu_pytorch_tanh`, `attn_logit_softcapping`, a missing or non-list `layer_types` or an unknown layer type, a `sliding_window` below 2 with chunked layers, RoPE scaling other than the default, and non-positive rotary bases or head dimensions.

**AXK-1 (`axk1`).** Multi-head latent attention with optional Q LoRA (without `q_lora_rank` the plain `q_proj` is stored as `attn_q`), NORM-style rotary embeddings over `qk_rope_head_dim` and the DeepSeek grouped router (`scoring_func` defaults to `sigmoid`). With `llama_4_scaling`, queries are multiplied by `1 + beta * log(1 + floor(position / original_max_position_embeddings))`. The sparse layers are the block after `first_k_dense_replace`, or follow `mlp_layer_types` when it is a dense lead followed by sparse entries. The checkpoint stores `post_mlp_layernorm` on every layer but only sparse layers apply it, so conversion writes it as `blk.N.post_ffw_norm.weight` there and drops the dense layers' copies. Refused by name: `attention_bias`, an activation other than `silu`/`swish`, a zero `qk_nope_head_dim` or `qk_rope_head_dim`, an invalid `n_group`, `topk_group` or `num_experts_per_tok`, `moe_layer_freq` other than 1, interleaved or all-dense `mlp_layer_types`, and a malformed `llama_4_scaling`.

**OLMo Hybrid (`olmo-hybrid`).** The Gated DeltaNet layers store `attn_qkv` as `[q|k|v]`, `ssm_conv1d`, `ssm_a` (`-exp(A_log)`), `ssm_dt.bias`, `ssm_beta`, `ssm_alpha`, `ssm_norm` and `ssm_out`; two keys describe the delta rule, `ssm.norm_eps` (the output-norm epsilon, 1e-5) and `ssm.beta_scale` (`linear_allow_neg_eigval`). The attention layers carry `post_attention_norm` and `post_ffw_norm`, their Q/K norms span the whole Q/K projection, and they rotate only when `attention.rope_pattern` and `rope.dimension_count` enable it. Decode runs incrementally on the recurrent state.

**Zamba2 (`zamba2`).** A hybrid layer runs its shared block on `[h | original embedding]` (a `2 * n_embd` row): RMSNorm, attention over the shared `shared_blk` weights (whole-head NEOX RoPE when `use_mem_rope`, scale `(attention_head_dim / 2) ** -0.5`), RMSNorm, a `GELU(gate) * up` MLP, then `shared_linear`; its output is added to the Mamba-2 input. Each use of a shared block adds its own rank-`adapter_rank` adapters to the q/k/v and gate/up projections and keeps its own KV cache, and the Mamba-2 mixer runs over the layer's own recurrent state. Prefix reuse works through state snapshots and session files; context shift and cache-write-only steps are refused by name, since every layer keeps recurrent state. Multi-device splits, GQA checkpoints (`num_key_value_heads != num_attention_heads`), `add_bias_linear` and activations other than GELU are refused by name. `time_step_limit` is not applied, as in vLLM's forward.

**Step5 (`step5`).** GQA attention with per-head Q/K RMS norms, a per-head sigmoid output gate and partial NeoX RoPE, restricted by a sparse region indexer: the sequence is cut into 8-token regions, a 256-wide proxy head scores each closed region's summary with weighted-ReLU scores, and each query attends to the top-k closed regions plus the open one. Feed-forward layers are dense SwiGLU or sigmoid-routed experts with a shared expert, and some blocks use a sliding window. `--mtp` drafts with the model's NextN block; the target keeps one state snapshot per drafted token so the region summaries roll back across a rejected draft. Step5 runs on the CPU and on the OpenCL, CUDA, HIP and Vulkan backends, which share one sparse-attention kernel source (`kernels/llm/sparse/step5.cl`) and its GLSL twins: the indexer norms and RoPE, the BF16 and FP8-E4M3 roundings, the open-region ring and closed-region summaries, the region scorer, the top-k selection and the gated sparse or sliding-window gather, for every KV cache type without per-token-head scales. The CPU engine also serves paged multi-sequence batches, each sequence with its own summary and ring state; non-causal pooling batches are refused by name, because the summaries and the gather are causal. A contiguous Step5 sequence can be context-shifted when the memory is planned with `-Dneuralfree.context.shift=true`: the cached keys and the open-region ring are re-rotated and the closed-region summaries rebuilt at their new positions. That plan reserves `nCtx * 2 * proxyDim` floats per full-attention layer for the ring; paged serving does not use it.

**LongCat-Flash (`longcat-flash`, `longcat-flash-ngram`).** Each logical layer is two physical blocks, as vLLM's `FlashDecoderLayer` runs two attention and two MLP branches: the first block's dense MLP feeds the residual while its routed MoE output is carried to the second block, which adds it after its own dense MLP. The router is `real + zero` experts wide (`expert_zero_count`): top-k runs over the softmax probabilities plus `e_score_correction_bias`, the weights are the unbiased probabilities times `routed_scaling_factor`, and a zero-compute expert passes its input through scaled by its weight. Attention is the DeepSeek-V2 absorbed MLA path, with `mla_scale_q_lora` / `mla_scale_kv_lora` folded into the LoRA norms at conversion. The n-gram variant reads `ngram_vocab_size_ratio`, `emb_neighbor_num` and `emb_split_num` and adds the projected hash-table embeddings of the preceding tokens to the word embedding, averaged over the embedders. The MTP layer converts to the NextN block (`--no-mtp` drops it). A `--split-mode layer` boundary, and the hybrid host/device boundary, must fall between the two blocks of a pair.

**AliceAI (`alice-ai`).** KDA layers (separate q/k/v depthwise convolutions, a low-rank decay gate, a low-rank output gate and a gated output norm) alternate with Qwen3-Next gated attention layers (a per-head `[q|gate]` projection, per-head Q/K norms and partial rotary). Every layer has a sigmoid-routed MoE with a router bias and a gated shared expert. Layers are grouped into blocks of `block_attn_res_block_size`, and learned scores (`attn_res_score`, `ffn_res_score`, `output_res_score`) mix each block's residual stream across layers. Head counts differ between the KDA and attention layers, so they are stored per layer. The Gemma-style zero-centered norms are shifted by one at conversion. The checkpoint's `mtp.*` block is dropped, as vLLM ignores it. Refused by name: an activation other than `silu`, `kda_allow_negative_eigenvalues: true`, a router score function other than sigmoid, `router_bias_correction: false`, unequal KDA key/value geometry, latent routed experts, per-expert instead of fused expert tensors, and an unknown `layer_types` entry. Tensor parallelism does not split it.

**Bailing MoE V2.5 (`bailingmoe-v25`).** Lightning linear-attention layers alternate with MLA attention layers, as `attention.recurrent_layers` records. A lightning layer stores its fused projection head-interleaved (`[h][q | k | v]`) with an output gate, a gated group output norm and optional Q/K norms; `ssm.group_norm_size`, `ssm.qk_norm`, `ssm.linear_silu`, `ssm.q_scale` and `ssm.rope_dimension_count` describe the steps before the recurrence, whose state is `head_dim² × n_head` floats per layer. An MLA layer is the DeepSeek-V2 absorbed path. The first `leading_dense_block_count` layers use a dense feed-forward and the rest routed experts with an optional shared expert and router bias. It runs on the CPU, OpenCL and CUDA engines; the MTP draft layers are refused when requested.

**Inkling (`inkling`).** The converter splits vLLM's fused `attn.qkvr` projection into `attn_q`, `attn_k`, `attn_v` and `attn_rel` (rows `[Q | K | V | R]`) and its interleaved `gate_up_proj` into `ffn_gate` and `ffn_up`. Attention has no RoPE: a per-query relative-logit bias (`attn_rel_proj`, `rel_extent` wide, the sliding window on a local layer) positions the keys, the normed queries take a length-dependent log scaling (`log_scaling_n_floor`, `log_scaling_alpha`), and short convolutions (`attn_sconv_k`, `attn_sconv_v`, `attn_sconv`, `ffn_sconv`) mix K, V and the residual stream over the last tokens, each keeping its own state. Head geometry is per layer, the logits are divided by the muP `logit_scale`, and a checkpoint that pads its vocabulary has the padded logit rows masked. MoE layers, MTP heads and context shift are refused by name. Its attention runs the unfused norm, rotation and K-store steps.

**Dory (`dory`).** The trunk is `n_input_layers` plain blocks, then the block `[n_input_layers, n_input_layers + n_recurrent_layers)` run `n_recurrent_loops` times, then `n_output_layers` plain blocks. Every pass is a slot with its own KV cache; a pass starts from `justNorm(state + input)` and ends by writing `state = justNorm(x)`. `layer_types` selects each physical block's attention form (`full_attention`, `sliding_attention` or `mlp`), recorded as `dory.attention.sliding_window_pattern` and `dory.mlp_layer_pattern`; the sliding window is `dory.attention.sliding_window` and the SWA layers rotate with `dory.rope.freq_base_swa`. The repeat geometry is `dory.recurrent_block.{num_loops,n_input_layers,n_recurrent_layers,n_output_layers}`, the pass-boundary justnorm is `dory.recurrent_block.justnorm` with `dory.recurrent_block.justnorm_epsilon`, and the slerp residuals read `dory.slerp_alpha_scale` with the per-block `blk.N.attn_slerp_alpha.weight` / `blk.N.ffn_slerp_alpha.weight`. Attention blocks carry `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_gate` and `attn_q_norm.weight` (the weightless K justnorm has no tensor); MLP blocks carry `ffn_gate`, `ffn_up` and `ffn_down`. There is no final output norm, and `output_logit_scale.weight` scales the untied head's logits. Paged memory and prefix reuse keep one cache per pass (verified through the batch engine on the CPU and OpenCL); a session file carries every pass's rows, and a paged memory refuses the session file by name. Refused by name: `tie_word_embeddings`, `attention_bias`, `mlp_bias`, a non-`silu` activation, `recurrent_kv_cache_mode` other than `per_loop` (`last_loop` is refused explicitly), unknown `layer_types`, repeat ranges that do not cover the trunk or an empty block, a non-positive window on a sliding layer, non-positive `initializer_range`, `ngpt_alpha_init` or `attention_output_gate_logit_scale_init`, and RoPE scaling, partial rotation or a missing `rope_parameters` entry for a used attention type.

**InternS1-Pro (`interns1-pro`).** The text backbone is Qwen3-VL MoE's. `router_n_groups` becomes the router's group count, with every group contributing its share of the selected experts (`expert_per_group_top_k`). Attention rotates with the Fourier position embedding of vLLM's decoder layer: besides the NeoX rotation, each head projects its frequency components through the learned `rope_cos_coef` and `rope_sin_coef` matrices (kept F32 under every output type, since they are the rotation's own arithmetic). FoPE is enabled when the rope parameters carry `use_fope`, `fope_sep_head`, `num_inv_freq` or `fope_init_factor`, and is recorded as `{arch}.rope.fope.*`; it runs on the CPU and GPU backends and refuses context shift, per-pair frequency factors (NTK or `rope_freqs`) and a rope type other than NeoX by name. A checkpoint with `decoder_sparse_step != 1` or non-empty `mlp_only_layers` is refused, since the Qwen3-VL MoE text graph has no dense blocks, and the time-series tower's tensors are refused by name.

**IQuestLoopCoder (`iquest-loopcoder`).** `loop_num` is recorded as `iquest-loopcoder.recurrent_block.num_loops`, the whole `num_hidden_layers` trunk is the repeated block (`n_input_layers` and `n_output_layers` are 0), and `loop_window_size` is `iquest-loopcoder.recurrent_block.local_window` with `dual_attention` true and `justnorm` false. Every pass keeps its own cache. Pass 0 attends its own cache causally; every later pass projects the same Q/K/V and mixes a read-only attention over the pass-0 cache of the same physical layer (full causal window) with an attention over its own cache (window `loop_window_size`), gated per head by `sigmoid(blk.N.attn_loop_gate.weight · q_head + blk.N.attn_loop_gate.bias)`. The tuning knobs are therefore the two attention launches and the gate projection: `blk.N.attn_loop_gate.weight` is `[n_heads][head_dim]` and its bias holds one value per head. The block is otherwise a Llama block (input/post-attention RMS norms, GQA with NEOX RoPE, SwiGLU FFN, residual adds) followed by `output_norm.weight`; the head may be tied. Session save/restore and the context shift cover every pass's cache; the dual attention refuses paged memory and a host KV cache (`-nkvo`) by name. Refused by name: a non-`silu` activation, `is_causal: false`, `attention_bias`, `mlp_bias`, `loop_num < 1`, `loop_window_size <= 0`, a non-positive `rms_norm_eps`, `head_dim * num_attention_heads != hidden_size`, and RoPE scaling or partial rotation.

**HyperCLOVAX (`hyperclovax`).** `HyperCLOVAXForCausalLM` checkpoints load directly from Hugging Face. The exported GGUF layout is `token_embd`, `output_norm` and `output` (or a tied head) and, per block, `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output` (with optional biases on the output projection), `post_attention_norm`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down` (optional biases) and `post_ffw_norm`. `hyperclovax.attention.scale` carries `attention_multiplier`, `hyperclovax.embedding_scale` and `hyperclovax.residual_scale` the MuP factors, and `hyperclovax.logit_scale` the factor applied to the logits (`logits_scaling * logit_scale`). `nftool export` writes this layout and the exported file loads back through the GGUF path.

**DomynEdge (`domyn-edge`).** `DomynEdgeForCausalLM` checkpoints load directly from Hugging Face. The exported GGUF layout is `token_embd`, `output_norm` and `output` (or a tied head) and, per block, `pre_attention_norm` becomes `attn_norm`, `attention.q_proj`/`k_proj`/`v_proj` become `attn_q`/`attn_k`/`attn_v`, `attention.k_norm` becomes `attn_k_norm` (Q has no norm weight, so no `attn_q_norm` is written), `attention.o_proj` becomes `attn_output`, `post_attention_norm` stays, and `pre_feedforward_norm`, `feedforward.gate_proj`/`up_proj`/`down_proj`, `post_feedforward_norm` become `ffn_norm`, `ffn_gate`/`ffn_up`/`ffn_down`, `post_ffw_norm`. The per-layer head widths travel in `attention.key_length`/`attention.value_length` and their `_swa` counterparts, the rotary width in `rope.dimension_count` (and `_swa`), the shared window in `attention.sliding_window`, and the layer switches in `attention.sliding_window_pattern` and `attention.rope_pattern`. `nftool export` writes this layout and the exported file loads back through the GGUF path.

**Solar (`llama`).** `SolarForCausalLM` checkpoints whose `bskcn_*` lists are empty and whose `logit_scale` is absent or 1 load through the ordinary Llama conversion, with the Llama tensor names and the Q/K permutation the NORM rotary needs.

**Param2MoE (`dots1`).** `Param2MoEForCausalLM` checkpoints load directly from Hugging Face. `model.word_embeddings` becomes `token_embd`, `model.norm` becomes `output_norm`, the fused `attention.query_key_value` is split into `attn_q`, `attn_k` and `attn_v`, `attention.query_layernorm`/`key_layernorm` become `attn_q_norm`/`attn_k_norm`, `attention.dense` becomes `attn_output`, `mlp.gate.expert_bias` becomes `exp_probs_b.bias` (zero-meaned at conversion), `mlp.gate` becomes `ffn_gate_inp`, the per-expert projections are stacked into `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps`, and `mlp.shared_experts.*` becomes `ffn_gate_shexp`, `ffn_up_shexp` and `ffn_down_shexp`. Because the published checkpoints tie the head, the token table is also written as `output.weight`, which the `dots1` layout requires. `dots1.leading_dense_block_count`, `dots1.expert_shared_count`, `dots1.expert_weights_scale` (from `routed_scaling_factor`), `dots1.expert_weights_norm` (from `norm_topk_prob`) and `dots1.expert_gating_func` carry the routing; `dots1.rope.dimension_count` carries the whole head. `nftool export` writes this layout and the exported file loads back through the GGUF path.

**Step1 (`step1`).** `Step1ForCausalLM` checkpoints load directly from Hugging Face. The exported GGUF layout is `token_embd`, `output_norm.weight`, `output.weight` and, per block, `attn_norm.weight`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_norm.weight` and the gated `ffn_gate`/`ffn_up`/`ffn_down`, each with an optional `.bias`. `step1.attention.max_alibi_bias` carries the fixed 8; the square-root ALiBi distance is the `step1` architecture's own. `nftool export` writes this layout and the exported file loads back through the GGUF path.

**Sarvam MLA (`deepseek2`).** `SarvamMLAForCausalLM` checkpoints use the DeepSeek-V2 layout (`attn_q_a*`, `attn_kv_a_mqa`, split `attn_k_b`/`attn_v_b`, `ffn_gate_inp`, `ffn_*_exps`, `ffn_*_shexp`). The converter centers `e_score_correction_bias` (written as `exp_probs_b.bias`) before writing it and supplies the routing defaults above.

**openPangu MoE (`deepseek2` or `pangu-moe`).** `PanguUltraMoEForCausalLM` and `PanguProMoEV2ForCausalLM` checkpoints load from Hugging Face; the MLA geometries export as `deepseek2`, the standard ones as `pangu-moe` with `attn_q/k/v` (or `attn_qkv`), `attn_output`, `ffn_norm`, `ffn_gate_inp`, `ffn_exp_probs_b.bias`, `ffn_*_exps` and `ffn_*_shexp`, plus the dense lead's `ffn_gate/up/down`. Router biases map from `e_score_correction_bias`; `norm_topk_prob` and `routed_scaling_factor` are carried into `expert_weights_norm` / `expert_weights_scale`. The static sink attention (`param_sink_number > 0`) is exported with `attn_k_norm`, `attn_sink_k` and optional `attn_sink_v`; the sink keys are normalized by the K norm at load time and are not rotated. The per-layer windows are written as `attention.sliding_window` (largest), `attention.sliding_window_pattern` and, when the sizes differ, `attention.sliding_window_per_layer` (one entry per block, 0 = full attention). `rope.type = norm` selects the consecutive-pair rotary of this attention.

### Loading a model from object storage

A model path may be an object-store URI instead of a local file or directory. This works anywhere a model
path is taken: the CLI `-m/--model`, the server's `-m/--model` (or its positional model argument), batch
runs, and `--export-gguf`.

```
neuralfree --model s3://my-bucket/models/llama-3.1-8b --ctx-size 8192
neuralfree serve -m gs://my-bucket/models/qwen2.5-7b --port 8000
neuralfree serve az://my-container/models/mistral-7b
```

Supported locations:

| URI | Store |
|-----|-------|
| `s3://<bucket>/<prefix>` | Amazon S3 and S3-compatible stores (MinIO, Ceph, R2 with the S3 API) |
| `gs://<bucket>/<prefix>` | Google Cloud Storage |
| `az://<container>/<prefix>`, `azure://<container>/<prefix>` | Azure Blob Storage |

`<prefix>` names the checkpoint directory (the directory holding `config.json`, the tokenizer files and the
`*.safetensors` / `*.gguf` / `*.bin` weights). A prefix that is one object's full key loads that single file,
so a single GGUF or safetensors file can be loaded directly.

The checkpoint is mirrored into `neuralfree.hf.cache.dir` (default `~/.cache/neuralfree/hf`) under
`remote/<sha256 of the URI>`. The first open downloads the model files; later opens reuse the mirror and
download only objects whose size or entity tag changed, and remove files the store no longer lists. The mirror
keeps configurations, tokenizers, indexes and weights (`*.json`, `*.safetensors`, `*.gguf`, `*.bin`, `*.pt`,
`*.pth`, `*.model`, `*.txt`, `*.tiktoken`, `*.jinja`, `*.spm`, `*.vocab`, `*.merges`), and skips `original/`
and `.git/`.

The number of parallel downloads is set with the `neuralfree.remote.concurrency` system property
(default 8, accepted range 1-256), for example `-Dneuralfree.remote.concurrency=16`.

#### S3 credentials and endpoint

| Setting | System property | Environment |
|---------|-----------------|-------------|
| Endpoint | `neuralfree.remote.s3.endpoint` | `AWS_ENDPOINT_URL`, then `S3_ENDPOINT_URL` |
| Region | `neuralfree.remote.s3.region` | `AWS_REGION`, then `AWS_DEFAULT_REGION` |
| Access key | - | `AWS_ACCESS_KEY_ID`, then `S3_ACCESS_KEY_ID` |
| Secret key | - | `AWS_SECRET_ACCESS_KEY`, then `S3_SECRET_ACCESS_KEY` |
| Session token | - | `AWS_SESSION_TOKEN` |

Without an endpoint the client uses `https://s3.<region>.amazonaws.com` (default region `us-east-1`); with an
endpoint it uses path-style requests, which is what MinIO and other S3-compatible stores expect. Without
access keys the requests are anonymous, which public buckets accept.

#### GCS credentials

| Setting | System property | Environment |
|---------|-----------------|-------------|
| Endpoint | `neuralfree.remote.gcs.endpoint` | - (default `https://storage.googleapis.com`) |
| Bearer token | `neuralfree.remote.gcs.token` | `GCS_ACCESS_TOKEN`, then `GOOGLE_OAUTH_ACCESS_TOKEN` |

Without a token the requests are anonymous, which public buckets accept.

#### Azure credentials

| Setting | System property | Environment |
|---------|-----------------|-------------|
| Endpoint | `neuralfree.remote.azure.endpoint` | `AZURE_STORAGE_ENDPOINT` |
| Account | `neuralfree.remote.azure.account` | `AZURE_STORAGE_ACCOUNT` |
| Account key | - | `AZURE_STORAGE_KEY`, then `AZURE_STORAGE_ACCESS_KEY` |
| SAS token | `neuralfree.remote.azure.sas` | `AZURE_STORAGE_SAS_TOKEN` |

Without an endpoint the client uses `https://<account>.blob.core.windows.net`. A SAS token, an account key
(SharedKey signing), or neither (anonymous public container) authenticates the requests.

### vLLM state-dict checkpoints

Two vLLM load formats store vLLM's own module state dicts instead of Hugging Face tensors. Both load directly, from a local path or an object-store URI, and their tensors are mapped back to the Hugging Face names the converter reads; the directory needs the model's `config.json` (or `params.json`).

- **Pre-sharded checkpoints** (vLLM's `sharded_state` and `runai_streamer_sharded`): `model-rank-<rank>-part-<part>.safetensors`, or `model-pp-<pp>-tp-<tp>-part-<part>.safetensors` with pipeline stages. The ranks are reassembled by each parameter's vLLM layer type: column-parallel weights join along dim 0 and row-parallel along dim 1, vocabulary-parallel embeddings and heads join and are trimmed of vLLM's padding to `vocab_size`, and replicated parameters are taken once after a byte-for-byte check across ranks.
- **CoreWeave tensorizer** files (`*.tensors`, vLLM's `tensorizer` format, one file per rank): data versions 1 to 5, every length and offset checked against the file size before it is read. Encrypted files decrypt with the 32-byte key in `NEURALFREE_TENSORIZER_KEY` (64 hex digits) or the file `NEURALFREE_TENSORIZER_KEY_FILE` names; every chunk's Poly1305 tag is verified before its data is used, and an encrypted file without a key is refused by name.

vLLM's fused modules are split with the configuration's head counts and sizes: `qkv_proj` into q, k and v (with KV-head replication across ranks), `gate_up_proj` into gate and up, and packed `FusedMoE` experts (`w13`, `w2`) into per-expert tensors under the converter's names. Architectures whose Hugging Face checkpoint is itself fused keep the fused tensor. vLLM's tied-weight filtering is reversed when `tie_word_embeddings` is set.

Refused by name: quantized `FusedMoE` tensors (`w13_weight_scale` and their siblings), GPT-2 `Conv1D` state dicts, the packed or interleaved expert layouts of GPT-OSS, DBRX, Grok and Gemma 4, and expert-parallel state dicts (each rank holds a subset of experts rather than a slice of every one).

## Hugging Face Hub download

`--hf-repo <user>/<model>[:quant][@revision]` (short `-hf`) downloads a model from the Hugging Face Hub and opens it; it excludes `--model`. The `:quant` defaults to `Q4_K_M`, then `Q8_0`; `@revision` names a branch or commit. `--hf-file <name>` (short `-hff`) picks a concrete file in the repository instead of the default choice. `HF_HUB_OFFLINE=1` resolves from the cache only and refuses to download.

**Hub token.** Every Hub request (`-hf`, `nftool download`, `nftool slice`, `nftool bench -hft` and the bench-serve Hub datasets) authenticates with a token resolved in one place, in huggingface_hub's precedence: an explicit `--hf-token`/`-hft` first, then the `HF_TOKEN` environment variable, then the file `HF_TOKEN_PATH` names, then `HF_HOME/token` — where `HF_HOME` defaults to `$XDG_CACHE_HOME/huggingface` and then `~/.cache/huggingface` — which is the file `hf auth login` (or `huggingface-cli login`) writes, so a login token reaches gated models without exporting `HF_TOKEN`. A token is read with its whitespace trimmed and is sent only in the `Authorization` header, never printed or logged. The first source that carries a value decides: a value that is not a valid Hub token form means anonymous access rather than a silent fall-through, and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (also `on`, `yes`, `true`, case-insensitive) turns the environment and file sources off, so only an explicitly passed token is sent then.

The cache root follows llama.cpp's environment precedence (`HubCache`): `LLAMA_CACHE`, then `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, and finally `~/.cache/huggingface/hub`. `nftool download` fills the same cache (below).

## nftool

`demo/tools/NfTool` is the maintenance CLI. Thirty-nine subcommands are implemented:

| Subcommand | Flags | Meaning |
|---|---|---|
| `export` | see [below](#nftool-export) | Convert a checkpoint, GGUF model, projector or PEFT adapter to GGUF, as llama.cpp's `convert_hf_to_gguf.py` and `convert_lora_to_gguf.py` |
| `download` | the `--hf-repo` spec above | Download from the Hugging Face Hub into the cache root |
| `imatrix` | see [below](#imatrix) | Collect importance-matrix statistics over calibration text, as llama.cpp's `imatrix` |
| `cvector` | `--positive-file <fname>`, `--negative-file <fname>`, `--out-file <fname>`, `--method pca\|mean`, `--model-hint <arch>`, `--pca-batch <n>`, `--pca-iter <n>`, `--overwrite` | Train a control vector from prompt pairs over one loaded engine (below) |
| `perplexity` | see [below](#nftool-perplexity) | Perplexity, KL divergence and the multiple-choice benchmarks, as `llama-perplexity` |
| `passkey` | see [below](#nftool-passkey) | The long-context pass-key retrieval test of llama.cpp's passkey example |
| `quantize` | see [below](#nftool-quantize) | Requantize a GGUF file offline, as `llama-quantize` |
| `export-lora` | see [below](#nftool-export-lora) | Merge LoRA adapters into a model, or convert a PEFT adapter to GGUF |
| `bench` | see [below](#nftool-bench) | Prompt-processing and generation sweeps with statistics, as `llama-bench` |
| `embedding` | see [below](#nftool-embedding) | Embed prompts and print the vectors, as `llama-embedding` |
| `retrieval` | see [below](#nftool-retrieval) | Search chunked files for queries by embedding similarity, as `llama-retrieval` |
| `batched-bench` | see [below](#nftool-batched-bench) | Multi-sequence prompt and generation throughput, as llama.cpp's `batched-bench` |
| `pack-experts` | see [below](#nftool-pack-experts) | Write a MoE model in Avalon's packed-expert layout |
| `split`, `merge` | see [below](#nftool-split) | Cut a GGUF into llama.cpp shard files, or join them back, as `llama-gguf-split` |
| `lookup-create`, `lookup-merge`, `lookup-stats` | see [below](#nftool-lookup-create-lookup-merge-and-lookup-stats) | Build, merge and replay the n-gram cache files of the `ngram-cache` drafter, as llama.cpp's lookup examples |
| `fit-params` | see [below](#nftool-fit-params) | Fit the context and layer placement to free device memory and print the arguments, as `llama-fit-params` |
| `slice` | see [below](#nftool-slice) | Build a small test slice of a Hugging Face safetensors model from range requests, without downloading its weights |
| `chat`, `complete` | see [OpenAI client commands](#openai-client-commands) | Stream from a running OpenAI-compatible server, as `neuralfree chat` and `neuralfree complete` |
| `dump`, `hash` | see [below](#nftool-dump) | Print a GGUF file's metadata and tensors, or hash its tensor data (llama.cpp's `gguf-dump`, `llama-gguf-hash`) |
| `set-metadata`, `new-metadata`, `convert-endian` | see [below](#nftool-set-metadata) | Edit one scalar metadata value in place, copy a file with metadata changed, or swap its byte order |
| `results`, `debug`, `eval-callback` | see [below](#nftool-results-debug-and-eval-callback) | Check logits against stored results, dump logits or embeddings, and print every forward stage |
| `tokenize` | see [below](#nftool-tokenize) | Tokenize a text on the model's vocabulary, as `llama-tokenize` |
| `bench-serve`, `bench-throughput`, `bench-latency`, `bench-sweep`, `bench-startup` | see [below](#nftool-bench-serve) | vLLM's serving, offline throughput, latency, parameter-sweep and startup benchmarks |
| `eval` | see [below](#nftool-eval) | Task accuracy of a running server on GSM8K or AIME, with regex, LLM or script graders (llama.cpp's `llama-eval.py`) |
| `finetune` | see [below](#nftool-finetune) | Fine-tune an F32, F16 or BF16 GGUF model on a text corpus on the CPU or GPU and write the trained GGUF, as `llama-finetune` |
| `rpc-server` | see [Remote compute servers](#remote-compute-servers-layer-split-offload) | Serve this machine's GPUs to a remote layer split, as llama.cpp's `rpc-server` |

A name that is not one of these subcommands is refused with the list of subcommands.

### nftool bench

`nftool bench <model> [options]` (also `neuralfree bench`) benchmarks a model as llama.cpp's llama-bench does: prompt processing (pp), token generation (tg), combined (pg) and depth (d) sweeps over grids of engine settings, repeated, with statistics and machine-readable output.

```console
./gradlew runNfTool -Pargs="bench model.gguf -ngl 0 -p 512,2048 -n 128 -r 5"
./gradlew runNfTool -Pargs="bench model.gguf -p 0 -n 64,128,256 -ctk q8_0 -o csv"
./gradlew runNfTool -Pargs="bench model.gguf -pg 512,128 -d 256 -o json -oe sql"
```

Every parameter takes a comma-separated list or a range (`first-last`, `first-last+step`, `first-last*mult`), and the cross product runs in an order that reuses each model load: runs that share an engine configuration (model, `-ngl`, `-ncmoe`, `-sm`, `-lm`, `-mg`, `-dev`, `-ts`, `-ot`, `-embd`, `-ub`, `-ctk`, `-ctv`, `-nkvo`) open the model once, with a context sized for the group's largest shape.

| Flag | Meaning | Default |
|---|---|---|
| `-p`, `--n-prompt N` | Prompt-processing tokens | 512 |
| `-n`, `--n-gen N` | Generated tokens | 128 |
| `-pg PP,TG` | One test that does both | |
| `-d`, `--n-depth N` | Tokens prefilled, untimed, before each run | 0 |
| `-b`, `--batch-size N`; `-ub`, `--ubatch-size N` | Batch and prefill chunk; `-b` must be at least `-ub` | 2048, 512 |
| `-ctk`, `-ctv TYPE` | KV cache types: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, fp8_e4m3, fp8_e5m2, fp8_per_token_head, int8_per_token_head, int4_per_token_head | f16 |
| `-ngl N` | 0 runs the CPU engine; any other value runs the GPU engine, or the CPU when no GPU is usable | -1 |
| `-sm`, `--split-mode MODE` | `none` runs one GPU; `layer` spreads contiguous layers and their KV caches over the GPUs; `row` shards every dense feed-forward over them; `tensor` also shards the attention heads | `none` |
| `-mg`, `--main-gpu I` | Index of the GPU that holds the embedding and the output head; with split mode `none`, the one GPU the load runs on | 0 |
| `-ts`, `--tensor-split S0/S1/...` | Share of each GPU (its layers under `-sm layer`, its feed-forward shard under `-sm row`), `/` or `;` separated; each comma-separated member is another test. Given without `-sm`, it runs a layer split | equal |
| `-nkvo`, `--no-kv-offload 0\|1` | `1` keeps the KV caches in host RAM, `0` keeps them on the device | 0 |
| `-t`, `--threads N` | CPU pool size, set before the pool starts; one value only | machine |
| `-ncmoe`, `--n-cpu-moe N` | The first N MoE layers keep their expert weights in host RAM | 0 |
| `-lm`, `--load-mode M` | Tensor reads: `auto`, `none`, `mmap` or `dio` (`none` reads into process buffers, `dio` uses direct I/O); the `mlock` modes are refused | `auto` |
| `-dev`, `--device D0/D1,...` | GPUs to use, by index or name substring; `none` = no offload (the CPU engine), `auto` = the configured devices; each comma-separated member is another test, and a member listing two or more devices splits by layer unless `-sm` says otherwise | `auto` |
| `-ot`, `--override-tensor P=BT;...` | Tensor placement overrides by pattern and buffer type; `CPU`, `CUDA*`, `ROCm*`/`HIP*`, `Metal*`/`MTL*`, `Vulkan*`, `OpenCL*` and the accelerator APIs' buffers (`SYCL*`, `XPU*`, `CANN*`, `Hexagon*`, `OpenVINO*`, `WebGPU*`, `MUSA*`, `zDNN*`, `ZenDNN*`) map onto host and backend placements, and a buffer type with no counterpart (`CUDA_Host`, `MTL0_Private`) is refused; each comma-separated group is another test | none |
| `-embd`, `--embeddings 0\|1` | `1` times the model's embedding pass over the prompt (the rows the pooling head reads) instead of the logits pass; embedding models without a decoder loop can bench their prompt shapes | 0 |
| `-hf`, `-hfr`, `--hf-repo U/M[:Q]`; `-hff`, `--hf-file F`; `-hft`, `--hf-token T` | A Hugging Face Hub model resolved (or read from the cache) into a model path; `-hff` names the file and overrides the quant tag, `-hft` supplies the token (default: `HF_TOKEN`, then the `hf auth login` token file; see [Hub token](#hugging-face-hub-download)) | unused |
| `-r`, `--repetitions N` | Timed repetitions per test; rows report `avg ± stddev` | 5 |
| `-o`, `--output FMT`; `-oe`, `--output-err FMT` | `md`, `csv`, `json`, `jsonl`, `sql` or `none`, on stdout and stderr | `md`, `none` |
| `--progress`, `--no-warmup`, `--delay S` | Progress lines on stderr; skip the warmup pass; pause before each test | |

Rows carry the model facts (file, type, size, parameter count), the run's settings — including `n_threads`, `n_cpu_moe`, `split_mode`, `main_gpu`, `no_kv_offload`, `devices`, `tensor_split`, `tensor_buft_overrides`, `load_mode` and `embeddings` in `csv`, `json`, `jsonl` and `sql`, and as `threads`, `n_cpu_moe`, `sm`, `main_gpu`, `nkvo`, `dev`, `ts`, `ot`, `lm` and `embd` in the Markdown table when they vary — and its statistics; json and jsonl add the raw `samples_ns` and `samples_ts`. A parameter that is not swept takes its value from the `-Dneuralfree.*` properties. Prompt tokens come from a fixed-seed generator, so a sweep is reproducible. `-fa`/`--flash-attn` (attention always runs the fused kernels, so `off` cannot be honored), `-nopo`/`--no-op-offload` (operations follow their tensors, so there is no separate host-operation offload), `-fitt`/`--fit-target` (the load has no per-device reserve margin; `neuralfree.gpu.vram.limit` caps the allocation instead) and `-fitc`/`--fit-ctx` (there is no minimum-context floor for the fit) are refused by name, as is every other llama-bench parameter the bench does not sweep.

### Serving and offline benchmarks

NeuralFree ships the serving and offline benchmark harness of `vllm bench serve|throughput|latency|sweep|startup`. The commands measure a running OpenAI-compatible HTTP server — NeuralFree's own `neuralfree serve` or any other (vLLM included) — so the numbers are end-to-end server numbers. Run them as `neuralfree bench-serve ...` or `nftool bench-serve ...`; `nftool <command> --help` prints each surface.

Common server flags: `--host` (default `127.0.0.1`), `--port` (default `8000`), `--base-url`, `--endpoint` (`/v1/completions` by default, or `/v1/chat/completions`), `--model` (default: the server's first `/v1/models` entry), `--served-model-name`, `--header KEY=VALUE`, `--insecure`. Prompt lengths come from the server's `/tokenize`, vocabulary and special ids from `/tokenizer_info`, and generated token counts from the stream's usage chunk (`stream_options.include_usage`).

#### nftool bench-serve

Sends a dataset at a controlled request rate and reports throughputs plus TTFT, TPOT (first token excluded), ITL and end-to-end percentiles.

```
neuralfree bench-serve --base-url http://127.0.0.1:8080 \
    --dataset-name random --num-prompts 200 \
    --random-input-len 1024 --random-output-len 128 \
    --request-rate 8 --burstiness 1.0 --max-concurrency 32 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result
```

Datasets (`--dataset-name`):
- `random` (default): token-id prompts of `--random-input-len` and `--random-output-len`, spread by `--random-range-ratio` (a float, or `{"input":0.2,"output":0.5}`) with `--random-prefix-len` shared prefix tokens; `ignore_eos` is on, as the reference defaults for this dataset.
- `sharegpt`: a ShareGPT JSON file at `--dataset-path`; the first turn is the prompt and the second the output length (`--sharegpt-output-len` fixes it).
- `custom`: a JSONL file of `prompt` records with `output_tokens` (required when `--custom-output-len` is `-1`; otherwise that flag, default 256, fixes the length).
- `burstgpt`: a BurstGPT CSV log at `--dataset-path`; GPT-4 rows with a positive response length replay their length distribution.
- `prefix_repetition`: `--prefix-repetition-num-prefixes` shared prefixes of `--prefix-repetition-prefix-len` tokens plus a fresh suffix of `--prefix-repetition-suffix-len` tokens per request, to stress the prefix cache.

File datasets are pruned by the reference's bounds (prompt 4..1024 tokens, prompt+output at most 2048, fixed outputs skip the minimum), shuffled unless `--disable-shuffle`, and oversampled to `--num-prompts` unless `--no-oversample`. `--seed` makes the sampling reproducible; `--input-len`/`--output-len` map onto the dataset-specific lengths.

Traffic: `--request-rate` (requests/second, `inf` sends everything at once, default `inf`), `--burstiness` (gamma shape; `1` is a Poisson process, `inf` constant gaps), `--max-concurrency`, and the ramp-up flags `--ramp-up-strategy linear|exponential`, `--ramp-up-start-rps`, `--ramp-up-end-rps`. A ramp-up cannot be combined with `--request-rate`. `--ready-check-timeout-sec` makes the first request a readiness probe (skipped by default), and `--num-warmups` sends unmeasured requests first.

Metrics: `--percentile-metrics` selects from `ttft,tpot,itl,e2el,client_queue_time,e2el_including_client_queue` (default `ttft,tpot,itl`) and `--metric-percentiles` selects percentile points (default `99`). `--goodput KEY:VALUE` sets service-level objectives in milliseconds for `ttft`, `tpot` and/or `e2el` and adds request goodput. The report prints the reference's layout; `--save-result`, `--append-result`, `--save-detailed`, `--result-dir`, `--result-filename`, `--metadata KEY=VALUE`, `--label` and `--request-id-prefix` control the JSON result file (`{label}-{rate}qps[-concurrencyN]-{model}-{timestamp}.json`).

Sampling flags apply to every request: `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--frequency-penalty`, `--presence-penalty`, `--repetition-penalty`, `--logprobs` (completions), `--extra-body <json>`, and LoRA selection with `--lora-modules <name...>` and `--lora-assignment random|round-robin`.

##### Serving benchmark datasets

`nftool bench-serve` (and `bench-throughput`, `bench-sweep`) now accept the reference's remaining
`--dataset-name` values:

| Dataset | `--dataset-path` | Endpoint | Notes |
|---|---|---|---|
| `random-mm` | none | `/v1/chat/completions` | Synthetic text plus random-pixel images; `--random-mm-base-items-per-request`, `--random-mm-num-mm-items-range-ratio`, `--random-mm-limit-mm-per-prompt '{"image": 3}'`, `--random-mm-bucket-config '{(256, 256, 1): 0.5}'`. Image buckets only (video is refused). |
| `random-rerank` | none | `/rerank` | Synthetic query/documents; `--random-batch-size`, `--no-reranker`. |
| `custom_image` | JSONL file | `/v1/chat/completions` | Records with `prompt` + `image_files`, or an interleaved `content` array. Local images become base64 data URLs. |
| `custom_audio` | JSONL file | `/v1/audio/transcriptions` | Records with `audio` and optional `prompt`; `--custom-output-len -1` reads `output_tokens`. |
| `asr` | directory or row file | `/v1/audio/transcriptions` | Records with `audio`; `--asr-min-audio-len-sec`, `--asr-max-audio-len-sec`; output default 1024. |
| `spec_bench` | JSONL file | generation | `turns[0]` is the prompt; `--spec-bench-category`, `--spec-bench-output-len`. |
| `speed_bench` | directory | generation | Reads `<subset>.jsonl`; `--speed-bench-dataset-subset`, `--speed-bench-category`, `--speed-bench-output-len`. |
| `timed_trace` | JSONL file | `/v1/completions` | `--timed-trace-chunk-hash-size`, `--timed-trace-sec-multiplier`, `--timed-trace-label-*`; `--self-timed` schedules by the trace timestamps. |
| `blazedit` | directory or row file | generation | `code` + `change_request` + `norm_distance`; `--blazedit-min-distance`, `--blazedit-max-distance`; output default 4000. |
| `hf` | directory, row file or Hub repository id | generation/chat | vLLM's Hugging Face datasets. `--dataset-path` is a local file/directory or a repository id (`openai/gsm8k`); `--hf-name` names the repository when `--dataset-path` is absent (a local path wins). Local directories of `.jsonl`, `.json`, `.csv`, `.tsv` or `.parquet` files; Hub repositories are downloaded through NeuralFree's Hub client into the Hub cache (`HF_TOKEN`, `HF_HUB_OFFLINE`). The row schema selects the sampler (conversations, turns, messages, audio, code edits, interleaved images, prompt, question). `--hf-subset`, `--hf-split`, `--hf-output-len`; `--no-stream` is accepted (data files are always read completely). See [Hub datasets and client-side chat templates](#hub-datasets-and-client-side-chat-templates). |

The endpoint is inferred from the dataset when `--endpoint` is omitted, and an explicit endpoint
that the dataset cannot use is refused by name. Multimodal samples always reach the chat endpoint
as one user message; `--enable-multimodal-chat` is accepted for the image datasets.

##### Local tokenizer

Prompt lengths no longer require the server's `POST /tokenize`:

    nftool bench-serve --tokenizer /models/model.gguf --dataset-name custom --dataset-path prompts.jsonl
    nftool bench-serve --tokenizer /models/hf-checkpoint --tokenizer-mode hf --dataset-name hf --dataset-path rows/

`--tokenizer` accepts a GGUF file or a Hugging Face checkpoint directory and loads it with
NeuralFree's own tokenizer loader (metadata and tokenizer files only; model weights are not read).
`--tokenizer-mode` is `auto` (default) or `hf`. `--skip-tokenizer-init` initializes no tokenizer;
the reference's tokenizer-free datasets (`custom`, `custom_image`, `custom_audio`) then count one
token per prompt and one output token where no length is fixed.

##### Hub datasets and client-side chat templates

`--dataset-path` may name a Hugging Face dataset repository instead of a local path. The harness
downloads only the selected data files (`.parquet`, `.jsonl`, `.json`, `.csv`, `.tsv`) through the
same Hub client the model downloader uses, into the standard cache (`datasets--user--name`), and
reads them locally:

    nftool bench-serve --dataset-name hf --dataset-path openai/gsm8k
    nftool bench-serve --dataset-name hf --hf-name likaixin/InstructCoder --hf-split train

`--hf-subset` selects a configuration directory and `--hf-split` the split-suffixed files. Known
vLLM repositories (GSM8K, HumanEval, MT-Bench, InstructCoder, ShareGPT, the MLPerf data, AIMO and
the blazedit repositories) default to the same subset and split as the reference; an unknown
repository defaults to its single top-level directory and all of its data files. The Hub token (`HF_TOKEN`, else the `hf auth login` file)
authenticates the download and `HF_HUB_OFFLINE=1` reads an already-populated cache, refusing when
the repository is not there. `--no-stream` is accepted because the data files are always read
completely. `--trust-remote-code` stays refused by name: no remote code runs.

Parquet files are read directly, including nested `LIST`/`STRUCT` columns, uncompressed, SNAPPY and
GZIP compression, and PLAIN, dictionary, delta and RLE/bit-packed encodings. A file that uses a
MAP or nested list, an external column file, or ZSTD/LZ4/Brotli compression is refused by name.

Where the reference renders a dataset's prompt on the client, so does the harness, through
NeuralFree's Jinja chat-template engine. The template comes from `--tokenizer` (a GGUF file or a
Hugging Face checkpoint directory); without it, the server's `chat_template` reported by
`GET /tokenizer_info` is used. Rendered: `hf` turns (MT-Bench), instruction/input
(InstructCoder), messages, MLPerf system-plus-question, blazedit, the GSM8K question and plain
prompt rows; `custom`, `spec_bench` and `speed_bench`; and the chat form of the other text
datasets. Left raw: ShareGPT conversations, AIMO problems, image rows and audio rows.
`--skip-chat-template` sends the raw prompts, and `--chat-template-kwargs '{"key": value}'` binds
variables into every render (it needs a tokenizer, so it is refused with
`--skip-tokenizer-init`).

##### Profiling

`--profile` brackets the measured run with the server's profile window:

    nftool bench-serve --dataset-name random --num-prompts 200 --profile

The harness prints `Starting profiler...`, `Profiler started`, then after the run
`Stopping profiler...` and `Profiler stopped`. A server that refuses either call prints
`Profiler start failed: ...` or `Profiler stop failed: ...` on stderr and the benchmark still
reports its numbers. The server must run with `--profiler-config`, as vLLM's does: the profile
window answers 404 without it, so the harness prints the start failure and produces no recording.
NeuralFree's server writes a Java Flight Recorder recording for the window into the directory named by
the config's `torch_profiler_dir` (or `neuralfree.server.profile.dir`). The `.jfr` file can be opened
with JDK Mission Control or any JFR viewer. The routes are protected by the server's API key.

##### Plots

`--plot-timeline` writes `<result-file>.timeline.html`, one row per request with the TTFT and ITL
segments; ITL is coloured green below the lower threshold, orange up to the higher one and red
above it. `--timeline-itl-thresholds 25,50` sets exactly two non-negative millisecond thresholds
(lower first; the default and the reference's values are `25,50`). `--plot-dataset-stats` writes
`<result-file>.dataset_stats.svg` with the prompt, output and total token histograms and a stacked
per-request chart. Both files are self-contained HTML/SVG with no external assets and are written
even without `--save-result`; their stem is the result file name, so `--result-dir`,
`--result-filename` and `--label` place them. The harness prints `Timeline plot saved to: ...` and
`Dataset statistics plot saved to: ...`.

##### Server metrics, speculative decoding and RTFx

Every run snapshots the server's Prometheus endpoint before and after the measured workload. When
the server publishes speculative-decoding counters (`vllm:spec_decode_*` or NeuralFree's
`neuralfree_spec_*`), the result JSON carries the reference's keys:

- `spec_decode_acceptance_rate` (percent of drafted tokens accepted)
- `spec_decode_acceptance_length` (mean accepted tokens per verify round)
- `spec_decode_num_drafts`, `spec_decode_draft_tokens`, `spec_decode_accepted_tokens`
- `spec_decode_per_position_acceptance_rates`

Audio runs sum their input durations, so `rtfx` is the inverse real-time factor
(`audio seconds / benchmark duration`) and the report prints `RTFx (Inverse Real-Time Factor)` when
it is positive; the value is no longer a placeholder. Rerank runs report the reference's pooling
metrics (`input_sequence_throughput`, end-to-end latency percentiles) instead of generation metrics.

##### Probes and beam search

- `--probe-request-rate <rps>` sends single-token text probes alongside the main workload, outside
  `--max-concurrency`, and reports `probe_completed`, `probe_failed`, `probe_median_e2el_ms`,
  `probe_p99_e2el_ms` and `probe_max_e2el_ms`.
- `--use-beam-search` adds `use_beam_search: true` to completions/chat requests. Both flags are
  refused by name for the rerank and audio endpoints, which cannot carry them.

Unsupported reference options are refused by name, never ignored: other `--backend` values and
`--trust-remote-code` (no remote code runs; the harness downloads the repository's data files and
reads them locally). The synthetic datasets send token ids and therefore run on
`/v1/completions`; the chat endpoint takes the text datasets.

#### nftool bench-throughput

The server form of `vllm bench throughput`: the whole workload leaves at once with no rate and no concurrency cap, and the report gives request, output-token and total-token throughput plus latency statistics. It accepts the dataset and result flags of `bench-serve`; the traffic flags (`--request-rate`, `--burstiness`, `--max-concurrency`, the ramp-up flags, `--num-warmups`, `--ready-check-timeout-sec`, `--goodput`) belong to `bench-serve` and are refused by name.

#### nftool bench-latency

The single-batch latency benchmark: `--num-iters-warmup` (default 10) warmup batches and `--num-iters` (default 30) measured batches of `--batch-size` (default 8) token-id prompts of `--input-len` (default 32) tokens asking `--output-len` (default 128) tokens each. It prints the average and the 10/25/50/75/90/99th percentile latencies, and `--output-json` saves the latencies and percentiles. `--use-beam-search` and `--disable-detokenize` are refused by name.

#### nftool bench-sweep

Runs a parameter sweep over server and benchmark records, as `vllm bench sweep serve` does:

```
neuralfree bench-sweep \
    --serve-cmd "neuralfree serve model.gguf --port 8080" \
    --bench-cmd "bench-serve --dataset-name random --num-prompts 100" \
    --serve-params serve.json --bench-params bench.json \
    --link-vars max-model-len=max_tokens \
    --num-runs 3 --warmup-num-prompts 10 -o results -e my-experiment
```

`--serve-params` and `--bench-params` are JSON files: a list of parameter records, or an object of benchmark name to record. Each record's keys are flags and replace the command line's values (booleans become `--flag`/`--no-flag`, objects become JSON). For every serve record the sweep starts the given server, waits for its `/health` (derived from the command's `--host`/`--port`, default `127.0.0.1:8080`, capped by `--server-ready-timeout`), runs every linked-valid benchmark record `--num-runs` times, writes each run's own result plus the record parameters under `run=N.json` and the combination's `summary.json`, and finally a `summary.csv` over every run. `--continue-on-error` records failures as `*.failure.json`, `--dry-run` prints the plan, `--resume` skips runs already on disk, and `--show-stdout` keeps the server output on the console.

#### nftool bench-startup

Times how long a server command takes to answer `/health`, once per iteration: `--num-iters-cold` (default 3) cold iterations, `--num-iters-warmup` (default 1) unmeasured ones, then `--num-iters-warm` (default 3) measured ones. It prints the average and 10/25/50/75/90/99th percentile startup time of both phases and `--output-json` saves them (`avg_cold_startup_time`, `cold_startup_times`, `cold_startup_percentiles`, and the `warm_*` equivalents). `--serve-cmd` is required; `--health-url`, `--log-file`, `--show-stdout` and `--server-ready-timeout` tune the run. The reference's compilation-time metrics need in-process engine timings and are not reported.

### nftool embedding

`nftool embedding <model> [options]` works like llama.cpp's `llama-embedding`. It embeds one or more prompts and prints the vectors.

| Flag | Meaning |
|---|---|
| `--prompt, -p <text>` | The prompts; every `--embd-separator` occurrence starts a new prompt |
| `--file, -f <fname>` | Read the prompts from a file (one trailing newline dropped); the last of `-p`/`-f` wins |
| `--escape, -e` / `--no-escape` | Process the `\n \r \t \' \" \\` escapes of the prompt (default on) |
| `--embd-normalize <n>` | `-1` none, `0` max absolute scaled to int16, `1` taxicab, `2` Euclidean (default), `>2` p-norm |
| `--embd-output-format <f>` | empty: preview; `array`: `[[...],[...]]`; `json`: OpenAI-style list; `json+`: json plus the `cosineSimilarity` matrix; `raw`: one whitespace-delimited vector per line |
| `--embd-separator <s>` | Prompt separator (default a newline) |
| `--cls-separator <s>` | Separator of a reranker's query/document pair (default a tab) |
| `--verbose-prompt` | Print every prompt's tokens (`id -> 'piece'`) before embedding |
| `--parallel, -np <n>` | Memory sequences per batch, 1..256. `embedding` default: every prompt (up to 256) in one unified context, as llama.cpp's embedding tool does for `-np 1`. `retrieval` default: 1. `-np 1` given explicitly runs one prompt per forward. |
| `--batch-size, -b <n>` | Max tokens per batch (default 2048, raised to the context). A prompt or chunk longer than it refuses: `number of tokens in input line ... exceeds batch size`. |
| `--ubatch-size, -ub <n>` | Tokens per forward step. A non-causal model takes the batch size unless this is given (llama.cpp sets `n_ubatch = n_batch`). |
| `--ctx-size, -c <n>` | Context length (default: the model's). |
| `--pooling <t>` | The pooling override above. |
| `--gpu`, `--cpu`, `--backend <name>`, `-ngl <n\|auto\|all>`, `-ot`, `-t`, `-fa`, `--kv-type`, ... | LLMDemo's load flags. The tools run on the CPU engine unless `--gpu`, a non-cpu `--backend`, `-ngl` above 0 or `-Dneuralfree.backend` asks for a GPU. |

What the default preview prints depends on the model's pooling:
- Pooled models print up to 16 values per prompt when there are several prompts (every value for a single prompt), then the cosine similarity matrix.
- Pooling NONE prints one row per token (first and last three values).
- RANK models print `rerank score <i>: <score>`, with the class label when the head has several outputs.

For a RANK model, a prompt holding the `--cls-separator` is scored as a query/document pair. The pair fills the model's `tokenizer.chat_template.rerank` prompt when it has one; otherwise the pieces are joined with the EOS/SEP tokens the vocabulary adds.

Results go to stdout; the run summary, the last-token-not-SEP/EOS warning and the verbose dump go to stderr. Prompts (or chunks) embed in llama.cpp's batches: a batch closes when the next prompt would pass `--batch-size`
tokens or the memory's sequence count. Each batch runs as one multi-sequence forward over a paged memory, and the
vectors match the one-prompt-per-forward run.

For example, eight sequences per batch on the GPU: `nftool embedding qwen3-0.6b.gguf --pooling last -np 8 -ngl all -p "first<#>second" --embd-separator "<#>"`

```console
./gradlew runNfTool -Pargs="embedding bge-small.gguf -p 'cats purr<#>dogs bark' --embd-separator <#> --embd-output-format json+"
```

### nftool retrieval

`nftool retrieval <model> [options]` works like llama.cpp's `llama-retrieval`. It chunks text files, embeds the chunks, then answers queries read from stdin (one per line, until the input ends) with the most similar chunks by cosine similarity.

| Flag | Meaning |
|---|---|
| `--context-file <f>[,<f>...]` | Files to search; comma-separated and repeatable; at least one required |
| `--chunk-size <n>` | A chunk closes once it passes this many bytes, at a separator (default 64) |
| `--chunk-separator <s>` | Chunk boundary (default a newline) |
| `--top-k <n>` | Chunks printed per query (default 40) |
| `--verbose-prompt` | Print every chunk's tokens before embedding |
| `--parallel`, `--batch-size`, `--ubatch-size`, `--ctx-size`, `--pooling`, load flags | As for [`nftool embedding`](#nftool-embedding); `--parallel` defaults to 1 |

How chunks and queries are handled:
- Chunks follow llama.cpp's `chunk_file`: sizes and `filepos` count UTF-8 bytes, a final undersized chunk joins the previous one, and text after the last separator is not indexed.
- Chunks embed with a trailing EOS and are normalized Euclidean.
- Each query prints `filename`, `filepos`, `similarity` and `textdata` per hit.
- The model must produce pooled embeddings; pooling NONE is refused.

```console
echo "how do I build it" | ./gradlew runNfTool -Pargs="retrieval bge-small.gguf --context-file README.md,USAGE.md --chunk-size 100 --top-k 3"
```

### nftool batched-bench

llama.cpp's `batched-bench`. Every cell of the `-npp x -ntg x -npl` grid prefills `pl` random prompts of `pp` tokens, then runs `tg` decode steps of `pl` rows each over the paged memory. It reports the prompt-phase, generation-phase and total times and throughputs. The model opens with LLMDemo's [backend and weight flags](#backend-and-weights) and KV cache types:

- By default (or with `neuralfree.backend`) it opens on the best GPU backend, and every batch runs through the paged device memory's page tables.
- `--cpu` or `--backend cpu` runs the CPU engine.
- The model opens on the CPU engine instead of the GPU only for a reason the GPU cannot change: no usable GPU device
  or driver, the LLM kernel library unavailable on the device, a model that does not fit the device memory, no
  context that fits it, an attention geometry the GPU flash kernels do not serve (head dims that are not positive
  multiples of 4), or an architecture, weight type or layer feature the GPU engine does not implement. The
  warning names the reason, the startup line shows `Backend: cpu (<reason>)`, and `GET /props` and
  `GET /server-info` carry it as `gpu_fallback_reason` beside `backend`.
- Any other GPU load failure fails the load with the GPU error, and so does a named refusal of a layer split, `--w12`
  or an explicit `neuralfree.gpu.chainReplay=true`, settings the CPU engine would ignore.

The header's `backend = ...` names the backend the model opened on.

```
./gradlew runNfTool -Pargs="batched-bench model.gguf -c 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8"
```

| Option | Meaning |
|---|---|
| `-npp`, `-ntg`, `-npl` | comma-separated prompt, generated and parallel counts (required; repeatable, they append; `-npp` and `-ntg` also take 0) |
| `-pps` | prefill one prompt and share it across the sequences (page copy); `n_kv = pp + pl*tg`, `S_PP = pp/T_PP` |
| `-tgs` | decode the sequences one after another instead of one step for all |
| `-c`, `--ctx` | context the sequences share (`n_kv_max`); default 0, the model's trained context |
| `-ub`, `--ubatch` | tokens per forward step (default 512) |
| `-s`, `--seed` | seed of the random tokens (default 1) |
| `--output-format md\|jsonl` | the Markdown table (default) or one JSON object per cell |
| `--backend`, `--cpu`/`--gpu`, `--device`, `-ngl`, `-ot`, `--cpu-moe`, `--n-cpu-moe`, `-t` | where the model loads and computes, as for [LLMDemo](#backend-and-weights) (default: the best GPU backend, `-ngl auto`) |
| `-ctk`, `-ctv`, `--kv-type` | KV cache types (default `f16`) |
| `-fa on\|auto` | flash attention, which is always on; `-fa off` is refused |

A cell is skipped, with a note on stderr, when its `n_kv` exceeds `n_kv_max`, when `pp + tg` exceeds one sequence's cap, or when the paged pools run out during the cell. The pools allocate whole pages (`neuralfree.kv.page.size`, default 16 rows), so partial pages can exhaust them below `n_kv_max`.

`-npp 0` measures generation only: no prompt phase runs, the decode steps start at position 0, and the cell prints ~0 s for `T_PP s` and 0.00 for `S_PP t/s` (the cell carries no prompt tokens), as llama.cpp's batched-bench does.

A layer count below the model's layers (`-ngl <n>`) asks for a hybrid CPU+GPU split. Paged memory does not run over a split: the GPU load is refused, logged, and the bench falls back to the CPU engine. Use `-ngl all` or `-ngl auto`.

### nftool pack-experts

NeuralFree loads MoE models in Avalon's packed-expert layout directly. In such a model each MoE layer stores all its experts in one tensor, `blk.N.ffn_exps_packed`, described by the metadata key `avalon.experts_packed.blk.N`. Avalon's `scripts/pack-experts.py` writes these files, for example the Qwen3.8-Flash-Next-Avalon splits.

Open a packed model like any other GGUF model, single file or the first file of a split; no flag is needed. On load NeuralFree presents the usual `ffn_up_exps`, `ffn_gate_exps` and `ffn_down_exps` tensors, read in place from the packed tensor, so every backend, `--no-mmap` and the other read modes, `nftool export` and `nftool quantize` work unchanged. A packed file with a malformed layout is refused at open, with the layer and the key or tensor named:

- a packed tensor without its metadata key, or a key without its tensor
- a key that is not an array of 14 UINT64 values
- a stride that does not match the tensor size
- a role that does not fit its entry
- a standard expert tensor present beside the packed one

To write the packed layout:

```
nftool pack-experts <model> <out.gguf> [--overwrite] [--split-max-tensors <n>] [--split-max-size <n>]
```

- Every layer that has all three `blk.N.ffn_{up,gate,down}_exps.weight` tensors is packed into one I8 `blk.N.ffn_exps_packed` tensor, byte for byte as `pack-experts.py` writes it. The tool prints each layer's entry layout.
- All other tensors and keys are copied unchanged.
- `--split-max-tensors` and `--split-max-size` (bytes, or with a `k`/`m`/`g` suffix) write llama.cpp-named shards.
- A model with no routed experts, or a layer with only some of the three expert tensors, is refused by name.

To turn a packed model back into the standard layout, use `nftool export <packed model> <out.gguf>`.

### Batched and parallel examples

`BatchedDemo` and `ParallelDemo` run `BatchEngine` over the paged memory of the backend the model opens on. They take LLMDemo's [backend and weight flags](#backend-and-weights) (`--backend`, `--cpu`, `--device`, `-ngl`, `-ot`, `--cpu-moe`, `-t`), the KV cache types (`-ctk`, `-ctv`, `--kv-type`) and `-fa on|auto`.

- By default (or with `neuralfree.backend`) they open on the best GPU backend; `--cpu` runs the CPU engine.
- The header line names the backend the model opened on.

`BatchedDemo` (llama.cpp `examples/batched`) continues one prompt in `-np` parallel streams:

```
./gradlew runBatchedDemo -Pargs="model.gguf -p 'Hello my name is' -n 32 -np 4"
```

`-n` counts the whole sequence, prompt included. Each stream gets `n - P` tokens. The run is refused when `n_kv_req = P + (n - P) * np` exceeds `-c`. Further options: `-c`, `-ub`, `-s` (one seed for every chain, -1 random), `--temp`, `--top-k`, `--top-p`.

`ParallelDemo` (llama.cpp `examples/parallel`) simulates `-np` clients serving `-ns` requests:

```
./gradlew runParallelDemo -Pargs="model.gguf -np 4 -ns 16 -n 128 --junk 4"
```

- **Requests:** each request is the system dialog, `[0, max(1, junk))` filler question/answer blocks, and a question from the built-in list, or from the newline-separated `-p <text>` / `-f <file>` lines.
- **Batching:** `-cb` (the default) admits a request as soon as a client finishes. `-nocb` waits until every client has finished.
- **Seeds:** a non-negative `-s` seeds every client alike. A negative one gives client `i` the 32-bit seed `s - i`, where `0xFFFFFFFF` means random, as in llama.cpp.
- **Stopping:** responses stop at `User:`, end-of-generation or `-n`.
- **Output:** each request prints its client line, input and response. The totals report the prompt, generated and average throughput, the prefix-cache reuse and the preemptions.
- **`-pps`:** refused. The engine's prefix cache shares the system prompt's pages once a request holding it has retired.

### nftool tokenize

`nftool tokenize <model> [options]` works like llama.cpp's `llama-tokenize`. It tokenizes one
text on the model's vocabulary only — the weights are never read, as llama.cpp loads it with
`vocab_only`.

| Flag | Meaning |
|---|---|
| `--prompt, -p <text>` | The text to tokenize |
| `--file, -f <fname>` | Read the text from a file, its content kept verbatim: one trailing newline is not dropped, because a trailing newline is itself a token. An explicit `--file` answers any `--prompt`, whatever the order |
| `--stdin` | Read the text from standard input (takes precedence over `-f`/`-p`); it is read after the model opens, so a bad model exits before the read |
| `--ids` | Print the ids as a Python-parseable list like `[1, 2, 3]` |
| `--show-count` | Print `Total number of tokens: N` after the tokens |
| `--no-bos` | Do not add a BOS token, even when the vocabulary normally uses one |
| `--no-parse-special` | Do not parse special tokens (chat, tool, etc) in the text |
| `--escape, -e` / `--no-escape` | Process the `\n \r \t \' \" \\` escapes of the text, stdin included (default on) |
| `--model, -m <path>` | The model as a flag, instead of the positional path |

Without `--ids`, every token prints as `%6d -> 'piece'`, the piece rendered as
`common_token_to_piece` renders it (control tokens show their text, SPM pieces their unescaped
space). The empty `-p ""` refuses with `tokenize needs one of --stdin, --file or --prompt`
(llama.cpp's `must specify one of` check); an explicitly given `--file` or `--stdin` passes even
when empty, and empty stdin then tokenizes to its special tokens alone.

Results go to stdout; the usage and errors go to stderr.

```console
./gradlew runNfTool -Pargs="tokenize model.gguf -f prompt.txt --ids"
cat prompt.txt | nftool tokenize -m model.gguf --stdin --show-count
```

### nftool dump

`nftool dump <model.gguf> [options]` prints the metadata and tensor table of a GGUF file, as
llama.cpp's `gguf_dump.py` does. The weights are not read.

| Flag | Meaning |
|---|---|
| `--no-tensors` | Stop after the key/value store |
| `--json` | Print the report as JSON instead of text |
| `--json-array` | Include full array values in the JSON report (needs `--json`) |
| `--data-offset` | Print the file offset of the tensor data and nothing else |
| `--data-alignment` | Print the data alignment and nothing else |

The text report starts with the file's byte order and the key/value count, then lists every
key/value pair as `<index>: <type> | <offset> | <key> = <value>`, and every tensor as
`<index>: <elements> | <shape> | <type> | <name>`. As in the reference, the version, tensor
count and key/value count head the list as `GGUF.version`, `GGUF.tensor_count` and
`GGUF.kv_count`. The JSON report carries the same facts, with `--json-array` adding every array
element. `--markdown` is not provided: `--json` carries the same content for a renderer.

A big-endian file is refused by the reader; convert it first with `nftool convert-endian`.

```console
./gradlew runNfTool -Pargs="dump model.gguf"
./gradlew runNfTool -Pargs="dump model.gguf --json --json-array"
nftool dump model.gguf --data-offset
```

### nftool hash

`nftool hash [options] <model.gguf>` hashes the tensor bytes of a GGUF file, as llama.cpp's
`gguf-hash` does. Metadata is not hashed.

| Flag | Meaning |
|---|---|
| `--xxh64` | xxHash64 per tensor and for the model (the default) |
| `--sha1` | SHA-1 per tensor and for the model |
| `--sha256` | SHA-256 per tensor and for the model |
| `--uuid` | The model's UUIDv5 in llama.cpp's namespace |
| `--all` | `--xxh64`, `--sha1` and `--sha256` together |
| `--no-layer` | Omit the per-tensor lines |
| `-c, --check <file>` | Verify against a manifest, line by line |

Every selected algorithm prints one line per tensor (`<type> <hash> <file>:<tensor>`) and one
for the model (`<type> <hash> <file>`), in tensor-table order. The UUID is one model-wide
identifier, not a per-tensor hash, so `--all` leaves it out.

A manifest holds `<type> <hash> <target>` lines, as generated by redirecting the tool's output.
`--check` prints each line followed by `Ok`, `Mismatch` or `Not Found`, then
`Verification results for <manifest> - Success`, `- Failure` or `- Manifest Missing Entry`.
Without an algorithm flag the check uses the most secure type the manifest holds (sha256, then
sha1, xxh64, uuid). The exit codes match the reference: 0 success, 1 a hash mismatch or a
per-tensor failure, 2 a mismatch, 3 no entry for the model, 4 no known type in the manifest,
5 the manifest cannot be read.

```console
nftool hash model.gguf > model.gguf.hash
nftool hash --all model.gguf
nftool hash --check model.gguf.hash model.gguf
```

### nftool set-metadata

`nftool set-metadata <model.gguf> <key> <value> [--dry-run]` changes one scalar metadata value
in place, as llama.cpp's `gguf_set_metadata.py` does. The key keeps its stored GGUF type and
byte width, and the value is parsed as that type (the UINT types unsigned). `--dry-run` prints
the planned change and writes nothing.

A key the file does not have is refused, because the tool changes values rather than adding
keys; STRING and ARRAY values are refused because their length is part of the file layout;
`general.alignment` is refused because the tensor offsets depend on it. Use
`nftool new-metadata` to add keys or change those values.

```console
nftool set-metadata model.gguf general.name "My model" --dry-run
nftool set-metadata model.gguf llama.context_length 8192
```

### nftool new-metadata

`nftool new-metadata <in.gguf> <out.gguf> [options]` copies a GGUF file with metadata added,
changed or removed, as llama.cpp's `gguf_new_metadata.py` does. The tensor table and every
tensor byte are copied unchanged; only the key/value section differs. Flags apply in
command-line order, so a later value of a key wins.

| Flag | Meaning |
|---|---|
| `--set <key=type:value>` | Set or add one scalar value; type is `str`, `bool`, `int`, `float`, `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int64` or `float64` |
| `--metadata <file>` | Set values from one JSON object file; arrays and every GGUF type |
| `--remove <key>` | Remove a metadata key |
| `--general-name <name>` | Set `general.name` |
| `--general-description <text>` | Set `general.description` |
| `--pre-tokenizer <name>` | Set `tokenizer.ggml.pre` |
| `--chat-template <text\|JSON>` | Set the chat template; a JSON list of `{name, template}` objects writes named templates too |
| `--chat-template-file <file>` | Set the chat template from a Jinja file |
| `--chat-template-config <file>` | Set the chat template from a `tokenizer_config.json` |
| `--special-token <name> <token>` | Set a special token id by its token text, resolved in the model's own vocabulary |
| `--special-token-by-id <name> <id>` | Set a special token id directly |
| `--overwrite` | Replace an existing output |
| `--dry-run` | Print the files the rewrite would produce; write nothing |

Setting a chat template replaces the file's whole chat template family: the old
`tokenizer.chat_template`, `tokenizer.chat_template.*` and `tokenizer.chat_templates` keys are
removed before the new ones are written. Split models are refused: merge them first with
`nftool merge`.

```console
nftool new-metadata in.gguf out.gguf --general-name "My model" --chat-template-file chat.jinja
nftool new-metadata in.gguf out.gguf --set llama.context_length=uint32:8192 --remove general.description
```

### nftool convert-endian

`nftool convert-endian <model.gguf> <big|little|native> [options]` rewrites a GGUF file in the
other byte order, as llama.cpp's `gguf_convert_endian.py` does. The current order is detected
from the file's version field; `native` names the host's own order. Every multi-byte field of
the header, the key/value store and the tensor infos is written in the target order, and every
multi-byte tensor block field is byte-swapped.

| Flag | Meaning |
|---|---|
| `--out <file>` | Write the converted file to `<file>` instead of converting in place |
| `--overwrite` | Replace an existing `--out` file |
| `--dry-run` | Validate the tensor types and write nothing |

Without `--out` the file is converted in place through a temporary twin and an atomic rename. A
file already in the target order is left untouched. Types whose block layout is defined only
little-endian are refused by tensor name. NeuralFree itself reads little-endian files, so a
big-endian result is for a consumer that reads that order.

```console
nftool convert-endian model.gguf big --out model-be.gguf
nftool convert-endian model-be.gguf little --overwrite
```

### nftool results, debug and eval-callback

Three tools compare a run against stored results, dump logits or embeddings, and print the forward stages of a model.
They are `nftool` subcommands and are also reachable as `neuralfree results`, `neuralfree debug` and
`neuralfree eval-callback` (`neuralfree help all` lists them).

#### results: store or check logits

Evaluates a prompt and writes the logits of every position to a GGUF file, or checks a run against such a file.

```sh
nftool results -m model.gguf -p "People die when they are killed." -o results.gguf
nftool results -m model.gguf -p "People die when they are killed." -o results.gguf --check
```

Options:

| Option | Meaning |
|---|---|
| `-m, --model <path>` | the model (also accepted as a positional argument) |
| `-p, --prompt <text>` | the prompt to evaluate |
| `-f, --file <fname>` | read the prompt from a file (one trailing newline dropped) |
| `-o, --output, --output-file <fname>` | the results file to write, or to check against (required) |
| `--check` | compare the run against the file instead of writing it |
| `-c, --ctx-size <n>` | the context the prompt must fit |
| `-b, --batch-size <n>` | forward ubatch of the engine |

The file is llama.cpp's results format: a `path_model` string key, a `tokens` tensor of GGML type I32 and a `logits`
tensor of type F32 holding `n_tokens x n_vocab` values token-major, so `llama-results` and this tool read each other's
files. `--check` prints `nftool results: NMSE=<value>` and then `OK` while the normalized mean squared error
(`mse(stored, computed) / mse(stored, 0)`) is at most `1e-6`, else `FAIL` with exit status 1. A check refuses a file
whose `path_model` names another model, whose tensors have another type or shape, or whose tokens differ from the
prompt's before it compares any logit.

#### debug: dump logits/embeddings or trace stages

```sh
nftool debug -m model.gguf -p "Hello my name is" --save-logits --logits-output-dir data
nftool debug -m model.gguf -p "Hello my name is" --embedding --save-logits --embd-normalize 2
nftool debug -m model.gguf -p "Hello my name is" --tensor-filter mixer --tensor-filter logits
```

Options:

| Option | Meaning |
|---|---|
| `-m, --model <path>` | the model |
| `-p, --prompt <text>` / `-f, --file <fname>` | the prompt |
| `--save-logits` | save files instead of printing the stage trace |
| `--logits-output-dir <dir>` | directory of the saved files, default `data` |
| `--embedding, --embeddings` | save embeddings instead of logits (needs `--save-logits`) |
| `--embd-normalize <n>` | -1 none, 0 max absolute, 1 taxicab, 2 euclidean (default), >2 p-norm |
| `--tensor-filter <regex>` | print only stages whose label starts with the pattern; repeatable |
| `--abort-on-nan` | stop with a message on the first stage that sums to NaN |
| `--steps <n>` | forward steps to trace, default 0 (every step) |
| `-c, --ctx-size <n>` / `-b, --batch-size <n>` | context and forward ubatch |
| `--escape, -e` / `--no-escape` | process the prompt's `\n \r \t \' \" \\` escapes (default on) |

The saved files are the reference's:

```
data/llamacpp-<model>.bin             raw little-endian floats
data/llamacpp-<model>.txt             one "index: value" line per float
data/llamacpp-<model>-prompt.txt      the prompt, its token count and its token ids
data/llamacpp-<model>-tokens.bin      the token ids, little-endian int32
```

`--embedding` names them `llamacpp-<model>-embeddings.*`: a pooled model saves its single pooled row, a model that
pools `NONE` one row per token, and a generative model without a per-token embedding output saves its final hidden
rows. Without `--save-logits` the tool prints the tokenized prompt and the forward-stage trace.

#### eval-callback: print every forward stage

```sh
nftool eval-callback -m model.gguf -p "Hello my name is"
nftool eval-callback -m model.gguf -p "Hello my name is" --tensor-filter mixer-0
```

Options: `-m, --model`, `-p, --prompt`, `-f, --file`, `--tensor-filter <regex>` (repeatable), `--abort-on-nan`,
`--steps <n>`, `-c, --ctx-size`, `-b, --batch-size`, `--escape/-e | --no-escape`, `--help/-h`.

The tool prints `number of input tokens = N` and each id, then every forward stage's result table. Stage labels are
`embedding`, `output_norm` and `logits` for the model-level stages and `<stage>-<layer>` for the layer stages
(`attn_norm-0`, `mixer-0`, `mixer_residual-0`, `ffn_norm-0`, `ffn-0`, `ffn_residual-0`, `layer-0`, ...), the labels
the engine's diagnostics use. A filter pattern is anchored at the start of the label, so `mixer` selects every mixer
stage and `mixer-1` one layer's. Each table prints its first and last three entries per truncated dimension and the
stage's sum, as llama.cpp's `common_debug_print_tensor` does.

The trace reads the GPU execution stages (the engine's dispatch boundary, the counterpart of llama.cpp's graph-node
callback). A model that opens on the CPU backend has no stage trace and is refused by name; `nftool debug
--save-logits` still dumps logits and embeddings on the CPU.

### nftool eval

`nftool eval` measures how often a running OpenAI-compatible server answers a math suite correctly. It is the NeuralFree counterpart of llama.cpp's `llama-eval.py`: it loads GSM8K or an AIME suite, shuffles it into cases, sends each prompt to one of one or more servers, grades the reply, and saves the verdicts as a state JSON file with a readable HTML report beside it.

```bash
# GSM8K, first 200 cases, regex grader, 32 requests in flight
nftool eval --server http://localhost:8033 --dataset gsm8k --n_cases 200 \
  --grader-type regex --threads 32 --model my-model

# AIME 2025, two servers, 16 requests each, LLM extraction grader on the first server
nftool eval --server http://a:8033,http://b:8033 --server-name a,b --threads 16,16 \
  --dataset aime2025 --grader-type llm --model my-model --output aime2025-state.json

# Authenticated servers (one key for all, or one per --server; $LLAMA_API_KEY is the fallback)
nftool eval --server http://a:8033,http://b:8033 --threads 8,8 --api-key sk-one,sk-two \
  --dataset gsm8k --grader-type regex --model my-model

# Resume the cases a former run could not answer (server errors and other non-ok cases)
nftool eval --server http://localhost:8033 --dataset gsm8k --grader-type regex \
  --model my-model --resume
```

Suites: `gsm8k` (default `openai/gsm8k`, `main`/`test`), `aime` (`AI-MO/aimo-validation-aime`), `aime2025` (`opencompass/AIME2025`, both parts) and `aime2026` (`MathArena/aime_2026`). `--dataset-source` replaces the canonical location with a local `.jsonl`, `.json`, `.csv`, `.tsv` or `.parquet` file, a dataset directory, or another Hub repository id. Rows are read with the same loader the serving benchmarks use, so the Hugging Face cache and `HF_HUB_OFFLINE` apply.

Graders: `regex` extracts the last number of the reply (last `\boxed{...}` first for AIME), grades only the last ten lines, and compares the answer exactly with the gold answer; `llm` asks a chat model (`--grader-server`, `--grader-model`, `--grader-api-key`, defaulting to the first server and `--model`) to extract the answer and compares case-insensitively; `cli` runs `--grader-script --answer <reply> --expected <gold>` and treats exit status 0 as correct. `--grader-type exec` and the code suites are refused by name because they execute generated code, which this tool does not do.

`--n_cases` may exceed the suite size: each pass is reshuffled from `--seed` and the report adds pass@1/5/10/n. Every case's prompt is sent with `n_predict` (and `max_completion_tokens` when bounded), `temperature`, `top_k`, `top_p`, `min_p` and `reasoning_effort` when given. Failed or empty requests are resent `--retries` times with a doubling `--retry-delay`; a case that never answers records `server_error` and does not count toward accuracy. `--verbose` prints each reply and answer. The result is `<output>.json` (the reference's state layout) plus `<output>.json.html`; the state is rewritten after every case, so an interrupted run continues with `--resume`.

### nftool finetune

Trains a GGUF model on a text corpus and writes the trained weights to a new GGUF file, the job of llama.cpp's
`llama-finetune`. `neuralfree finetune ...` runs the same tool.

```bash
nftool finetune -m model-f32.gguf -f corpus.txt -o trained.gguf \
    -c 128 -epochs 3 -lr0 1e-4 -lr-min 1e-5 -decay-epochs 10 -wd 0.1 --val-split 0.05
```

The corpus is tokenized once and cut into windows of `-c` tokens that start half a context apart; a window
predicts its own next token at every position but the last. The last `--val-split` fraction of the windows is
scored after every epoch and never trained on. After every epoch the tool prints the learning rate, the training
loss and accuracy and, when the split left validation windows, the validation loss and accuracy.

| flag | meaning |
|---|---|
| `-m, --model <fn>` | the F32, F16 or BF16 GGUF model to train |
| `-f, --file <fn>` | the training corpus (UTF-8 text) |
| `-o, --output <fn>` | the GGUF file the trained weights are written to; an existing file is refused |
| `-c, --ctx-size <n>` | the context of a training window (default 128) |
| `-b, --batch-size <n>` | must equal `-c`: one step trains one whole window |
| `-ngl, --gpu-layers <n>` | `0` (default) trains on the CPU; a count covering every trunk layer trains on the GPU (`neuralfree.backend` selects it); a partial count is refused |
| `-opt, --optimizer <adamw\|sgd>` | the update rule (default `adamw`) |
| `-lr0, --learning-rate <f>` | the learning rate of the first epoch (default `1e-4`) |
| `-lr-min, --learning-rate-min <f>` | the floor of the schedule (default `1e-5`) |
| `-decay-epochs <f>` | the epochs the rate decays over, reaching `-lr-min` (default 10) |
| `-wd, --weight-decay <f>` | weight decay (default 0.1) |
| `-epochs, --epochs <n>` | passes over the corpus (default 1) |
| `--val-split <f>` | the fraction of windows that validate (default 0.05) |

The model may be stored as F32, F16 or BF16: the run keeps float32 master weights and the output writes each trained
tensor back in its own stored type, so an F16 or BF16 model keeps its precision and size. The model must belong to the
supported decoder family: causal GQA/MHA attention with RoPE, an optional standard sliding window, attention and final
logit soft-caps, an embedding scale, a dense gated or ungated feed-forward network or a softmax-routed mixture of
experts, per-head Q/K norms with their biases, and pre-norms or sandwich post-norms. A model outside it — a quantized
model, a recurrent or otherwise unsupported architecture, group-limited or shared-expert routing, a sigmoid logit cap,
a context longer than the model was trained with — is refused with the architecture, layer and field that place it
outside. The output keeps the input's metadata, tokenizer, stored tensor types and every tensor the run did not step;
only the trained tensors' values change.

`-ngl 0` runs the host trainer on the engine's CPU pool, which `-Dneuralfree.cpu.threads` sizes. A count covering every
trunk layer runs the forward and backward pass on the train kernels of the GPU backend `neuralfree.backend` selects; one
window runs on one backend, so a partial count is refused. The GPU trainer runs the SWIGLU, GEGLU_TANH and REGLU gates
and the SILU, GELU_TANH and RELU activations; GELU's exact and quick forms are refused by name (train on the CPU with
`-ngl 0`).

### nftool split

Split one GGUF into llama.cpp shard files named `<prefix>-00001-of-0000N.gguf` derived from the output path.

```text
nftool split [--split] [--split-max-tensors <n>] [--split-max-size <n>]
             [--no-tensor-first-split] [--dry-run] [--overwrite]
             GGUF_IN GGUF_OUT
```

- `--split` — the operation of this command, as gguf-split names it.
- `--split-max-tensors <n>` — largest tensor count of one shard (default 128).
- `--split-max-size <n>` — largest shard size in bytes, e.g. `2000m` (`k`/`m`/`g` are powers of 1024); exclusive of `--split-max-tensors`.
- `--no-tensor-first-split` — the first shard carries only the metadata and the `split.*` keys, no tensors.
- `--dry-run` — print the files, their tensor counts and bytes; write nothing.
- `--overwrite` — replace an existing output (or every derived shard).

A source that fits one shard writes `GGUF_OUT` itself as a plain GGUF. The input may be the first file of a split, which is merged before it is cut again. A single tensor larger than `--split-max-size` is an error naming the tensor.

### nftool merge

Join the shards of a split back into one plain GGUF. `GGUF_IN` is the split's first file, named `<prefix>-00001-of-0000N.gguf`.

```text
nftool merge [--merge] [--dry-run] [--delete-splits] [--overwrite] GGUF_IN GGUF_OUT
```

- `--merge` — the operation of this command, as gguf-split names it.
- `--dry-run` — print the file, its tensor count and bytes; write nothing.
- `--delete-splits` — delete every shard file once the merged file is complete; a shard that cannot be deleted is a named error and every shard stays in place.
- `--overwrite` — replace an existing output.

The merged file keeps the split's metadata without the `split.*` keys, so it is a plain GGUF that no longer merges. Naming a shard file as the output is refused.

### nftool lookup-create, lookup-merge and lookup-stats

The `ngram-cache` drafter reads the same two files llama.cpp's lookup examples use. Build and inspect them with
three `nftool` subcommands; the model is read for its tokenizer (and, for stats, its training context) only, never
its weights.

```sh
# Build a static cache from a prompt's 2-grams (LLAMA_NGRAM_STATIC).
nftool lookup-create <model> -p "<prompt>" -lcs prompt-cache.bin
nftool lookup-create <model> -f corpus.txt -lcs corpus.bin

# Merge cache files into the last path; equal n-gram/token counts are summed.
nftool lookup-merge part1.bin part2.bin merged.bin

# Replay the prompt as consecutive contexts and report drafted/accepted tokens.
nftool lookup-stats <model> -p "<prompt>" -c 512 -lcs prompt-cache.bin -lcd generations.bin
```

`lookup-create` requires `-lcs/--lookup-cache-static <file>`; `-p/--prompt` or `-f/--file` supplies the text (a
file's single trailing newline is dropped, as llama.cpp's `read_file` does). `lookup-merge` needs at least one
input and an output; with one input it rewrites the same counts. `lookup-stats` takes `-c/--ctx-size` (default:
the model's training context), optionally `-lcs` and `-lcd` (a missing dynamic file starts empty), and
`--draft-max` (alias `--draft-n-max`, default 3); it writes neither cache file. All three also accept the model as
`-m/--model <file>`.

Feed the files to a generation with the same flags the engine takes, for example:

```sh
neuralfree cli <model> --spec-type ngram-cache -lcs prompt-cache.bin -lcd generations.bin
```

The static cache must exist and is read only; the dynamic cache is read when present and rewritten with the run's
n-grams when the session closes.

### nftool fit-params

Fit the parameters of a device load to the free device memory and print them, as llama.cpp's
`llama-fit-params` does. Nothing is uploaded; the fitted arguments print to stdout and a load with
them runs as fitted.

```
nftool fit-params <model> [options]

  -m, --model <path>         the model file (or give it as the only argument)
  --fit-target, -fitt <MiB>  free device memory to leave beside the load (default 1024)
  --fit-ctx, -fitc <n>       smallest context the fit may reduce to (default 4096)
  --fit-print, -fitp         print the memory estimate beside the fitted arguments
  -c, --ctx-size <n>         context the load runs with (kept, not fitted)
  -ub, --ubatch-size <n>     logical batch of the compute buffers
```

The fit plans the load the engine would run: the context (`-c`), the layer placement (`-ngl`),
the expert overflow (`--cpu-moe` / `-ot`) and `-nkvo`. The load flags of the demos (`--backend`,
`--device`, the KV types, ...) shape the fit as they shape the load. The flags that fix the
placement the fit has to choose itself are refused: `-ngl`, `-ot`, `--cpu-moe`,
`--n-cpu-moe`, `--n-cpu-ffn`, `--fit`.

Output: one `fit-params <model>: <MODE> placement, ...` line, the memory estimate with
`--fit-print`, and the fitted arguments line last, e.g.

```
fit-params qwen3-30b-q4.gguf: HYBRID placement, 28 of the trunk weight layers keep their weights on the device
-c 8192 -ngl 29
```

A fitted load on every trunk layer prints `-ngl all`; an expert overflow prints `--cpu-moe` or the
overflowing layers as `-ot 'blk\.N\.ffn_(up|down|gate|gate_up)_(ch|)exps=cpu,...'`; host KV caches
add `-nkvo`.

`--vram-budget` / `--memory-report` of the demos end with the same line (`Fitted arguments: ...`)
for the load they describe; a load that needed no fitting prints none.

### nftool slice

`nftool slice <user>/<model>[@revision]` builds a small test slice of a safetensors model without downloading its
weights: it reads each shard's safetensors header with HTTP range requests, keeps the requested layers and experts
(renumbered to `0..n-1`) and fetches only those tensors. The slice is written as one `model.safetensors`,
`config.json` is patched to the new layer, expert and vocab counts, the small auxiliary repository files
(tokenizer, chat template, README, ...) are copied, and `.slice_meta.json` records what was read, kept and changed.
It is a developer and test tool for converter and loader fixtures; it is online only.

```
nftool slice Qwen/Qwen3-30B-A3B --num-layers 2 --num-experts 8 --dry-run
nftool slice Qwen/Qwen3-30B-A3B --keep-layers 0,3 --output ../models/qwen3moe-slice
nftool slice Qwen/Qwen3-0.6B --inspect
```

options:

```
  -o, --output <dir>       output directory (default ./tmp/hf_slices/<user>--<model>)
  --revision <rev>         branch or commit, instead of <user>/<model>@<rev>
  --num-layers <n>         keep the first n layers (default 2)
  --keep-layers <i,j,...>  keep exactly these layer indices (overrides --num-layers)
  --num-experts <n>        keep the first n experts of each expert layer (default 8)
  --vocab-size <n>         row-slice vocab-sized tensors to n rows (default 0: off)
  --max-size <size>        refuse a slice over this size (default 1G; K, M, G, T suffixes)
  --jobs <n>               concurrent range requests (default 8)
  --inspect                print the config analysis, tensor families and plan; write nothing
  --dry-run                plan and report the size; write nothing
  --overwrite              replace the files of a non-empty output directory
  --hf-token <token>       access token (default: HF_TOKEN, else the token file
                           HF_TOKEN_PATH names, else HF_HOME/token, as `hf auth login`
                           writes it)
```

Read `--inspect` before trusting a slice: the heuristics cover common layouts only. A family is recognized as the
model's layers only when its index count matches a layer count in `config.json` (or, without a config, when its
container is `layers`/`blocks`/`h` and it has at least four indices), so a model with unusual keys gets a warning
and no layer slicing. The MTP/nextn tail is kept, a dense-leading model keeps one MoE layer, stacked expert and
router tensors are row-sliced only when their first dimension is the declared expert count, and `--vocab-size`
does not slice the tokenizer, so a converted model's vocabulary will not match it. A quant tag
(`user/model:Q4_K_M`) is refused because the tool reads safetensors weights. The Hub token (`--hf-token`, `HF_TOKEN` or the `hf auth login` file) is sent
in the `Authorization` header and never printed.

### nftool perplexity

`nftool perplexity <model> [options]` is llama.cpp's `llama-perplexity`. One mode runs per invocation:

- **Perplexity** (default), over the data file in `--ctx-size` chunks. Every chunk runs from position 0 with the vocabulary's BOS first, and the second half of each chunk is scored. Each `[chunk]value,` line is the running perplexity, and the last line on stderr is `nftool perplexity: final estimate: PPL = <ppl> +/- <error>`.
- **`--ppl-stride <n>`** scores the trailing `n` positions of every chunk instead (llama.cpp's perplexity_v2); the context is widened by `n/2` at load, as llama.cpp does.
- **`--save-all-logits <fname>`** stores the run's log-probabilities in llama.cpp's `_logits_` format, so `llama-perplexity --kl-divergence-base` reads this tool's file and the other way round.
- **`--kl-divergence --kl-divergence-base <fname>`** compares the model with stored base log-probabilities. The per-chunk table prints PPL(Q), ln(PPL(Q)/PPL(base)), the KL divergence, the RMS of Δp and the top-token agreement; past a hundred rows the perplexity, KL-divergence and token-probability statistics follow.
- **`--hellaswag`, `--winogrande`, `--multiple-choice`** score a data file in the format of llama.cpp's validation sets (6-line HellaSwag tasks, the WinoGrande CSV, the binary multiple-choice format). `--hellaswag-tasks <n>` (and the other modes' `-tasks` flags) limits the tasks, and the seeded selection picks the tasks llama.cpp picks. HellaSwag and multiple-choice print the running accuracy with its Wilson interval, WinoGrande the per-task scores.

Options: `-f`, `--file <fname>` (the data file; a later one replaces an earlier one), `-p`, `--prompt <text>`, `-c`, `--ctx-size <n>` (default 512), `-b`, `--batch-size <n>` (the engine ubatch), `--chunks <n>` (-1 = all), `--ppl-output-type 0|1` (0: `[chunk]ppl,` per chunk; 1: one line per chunk) and `--help`. The modes and `--ppl-stride` exclude each other, and a prompt shorter than twice the context is refused with the counts.

### nftool passkey

`nftool passkey <model> [options]` is llama.cpp's passkey test. A pass key (a number drawn from the seed) hides at position `--pos` among `--junk` filler sentences. The prompt is prefilled in `--batch-size` batches, and once the context fills it shifts (`--keep` positions kept, one batch dropped per shift). The greedy answer to "What is the pass key?" (16 tokens) is printed with the key and `nftool passkey: passkey recovered: true|false`.

Options: `--junk <n>` (default 250), `--pos <n>` (default drawn from the seed), `--keep <n>` (default 32), `--batch-size <n>` (default 2048), `--ctx-size <n>` (default the training context plus `--keep`; required for files whose metadata has no training context) and `--seed <n>` (default 1).

### nftool quantize

`nftool quantize [flags] <model.gguf> [<out.gguf>] <type> [<nthreads>]` is llama.cpp's `llama-quantize`. The type is a recipe below, by name (case-insensitive) or by its `LLAMA_FTYPE_*` number. Without an output path the file is written to `ggml-model-<type>.gguf` beside the input, replacing an existing one. Flags come before the file arguments:

| Flag | Meaning |
|---|---|
| `--allow-requantize` | Allow requantizing tensors that are already quantized (reduces quality); without it such a tensor is refused |
| `--leave-output-tensor` | Keep `output.weight` as it is |
| `--pure` | Quantize every tensor to the recipe's base type, without the per-tensor mixture |
| `--imatrix <file>` | Importance matrix (GGUF or the legacy `.dat`) for weighted quantization |
| `--include-weights <name>`, `--exclude-weights <name>` | Use the matrix only for, or never for, tensors whose name contains `name` |
| `--output-tensor-type <type>`, `--token-embedding-type <type>` | Tensor type of `output.weight`, of `token_embd.weight` |
| `--tensor-type <regex>=<type>` | Tensor type of the tensors whose name matches; repeatable |
| `--tensor-type-file <file>` | Whitespace-separated `name=type` pairs |
| `--prune-layers <L0,L1,...>` | Drop the given blocks and renumber the rest (updates `<arch>.block_count`) |
| `--keep-split` | One output shard per input shard, with the input's split layout |
| `--override-kv <key>=<type>:<value>` | Set a metadata key in the output; repeatable |
| `--dry-run` | Print the per-tensor and total sizes, write nothing |
| `--max-buffer-size <MiB>` | Validated; rows are quantized one at a time, so the working set is one row already |

Recipes, with llama-quantize's per-tensor mixture rules unless `--pure`: `Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `F16`, `BF16`, `F32`, `COPY` (every tensor unchanged), `Q2_K`, `Q2_K_S`, `Q3_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_K`, `Q4_K_S`, `Q4_K_M`, `Q5_K`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `IQ4_NL`, `IQ4_XS`, `TQ1_0`, `TQ2_0`, `MXFP4_MOE` (MoE expert tensors as MXFP4, the rest as Q8_0), and the grid-search recipes `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S` and `IQ3_M`. The grid-search types are encoded the way llama.cpp's ggml-quants.c encodes them, block for block. They are mixed with the reference's rules: for example, `attn_v` goes to Q4_K when there are at least 4 query heads per KV head, `attn_output` goes to IQ2_XXS under the IQ1 recipes, and the middle layers' `ffn_gate`/`ffn_up` go to IQ3_XXS under `IQ3_XS`.

Tensors that land on IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S or IQ3_XXS, and the Q2_K tensors of a `Q2_K_S` file, need `--imatrix`. `token_embd.weight` and `output.weight` are exempt. A tensor whose matrix entry is missing is refused. IQ1_S, IQ2_XXS and IQ2_XS cannot be encoded without weights at all, so a `--tensor-type`, `--token-embedding-type` or `--output-tensor-type` that gives one of them to a tensor without a matrix entry is refused by tensor name. The `IQ3_S` and `IQ3_M` recipes run without an imatrix. The output keeps the input's metadata without the `split.*` keys, sets `general.quantization_version` to 2 and `general.file_type` to the recipe number, and records `quantize.imatrix.file`, `quantize.imatrix.dataset`, `quantize.imatrix.entries_count` and `quantize.imatrix.chunks_count` when a matrix was used.

```
nftool quantize model-f32.gguf Q4_K_M
nftool quantize --imatrix imatrix.gguf model-f16.gguf out.gguf Q4_K_M 8
nftool quantize --imatrix imatrix.gguf model-f16.gguf out-iq2.gguf IQ2_XXS
nftool quantize model-f16.gguf out-iq3.gguf IQ3_S
nftool quantize --dry-run model-f32.gguf Q4_K_M
nftool quantize --keep-split --prune-layers 1,2 model.gguf Q8_0
```

### nftool export

`nftool export <model> <out.gguf> [options]` converts any model NeuralFree opens to GGUF: a Hugging Face checkpoint, a GGUF file or split, or a PEFT LoRA adapter. It takes the flags of llama.cpp's `convert_hf_to_gguf.py` and `convert_lora_to_gguf.py`.

| Flag | Meaning |
|---|---|
| `--outtype <t>` | Float tensor type of a converted checkpoint: `f32`, `f16`, `bf16`, `q8_0`, `tq1_0`, `tq2_0`, `auto` or `source` (default `source`, which keeps each tensor's stored type; llama.cpp's default is `auto`). Under `tq1_0`/`tq2_0`, token embeddings, `output`, `altup_router` and `laurel` weights stay F16, and rows that are not whole 256-element blocks fall back to F16, as in the converter. Refused for a source already stored as GGUF. |
| `--mmproj` | Export the vision/audio projector of a Hugging Face checkpoint. The output path is used as given. |
| `--mtp` | Export only the NextN / MTP draft head as its own GGUF: the NextN blocks plus the embedding, final norm and output head the head shares with the target. |
| `--no-nextn`, `--no-mtp` | Export the target without its NextN / MTP tensors (and without `*.nextn_predict_layers`). Pair with `--mtp` on a second run to publish the target and the draft as two files. |
| `--dspark` | Export the checkpoint's DSpark draft as a `dflash` GGUF. Supported for `DeepseekV4ForCausalLM`, `DeepseekV41ForCausalLM` (from the `mtp.*` tensors) and `MistralLarge4ForConditionalGeneration` (from the `mtp_dspark.*` tensors); any other target is refused. |
| `--lora` | `<model>` is a PEFT adapter directory (`adapter_config.json` plus `adapter_model.safetensors`). The output is a GGUF LoRA adapter. `--outtype` takes `f32`, `f16`, `bf16`, `q8_0`, `auto` or `source`. |
| `--base <path>` | Base model of a `--lora` export: an HF directory with at least its `config.json` (weights optional), or a GGUF model. Without it, the adapter's `base_model_name_or_path` is used when it names a local directory. The Hugging Face hub is not queried. |
| `--metadata <file>` | Authorship metadata overrides from one JSON object, with the converter's keys and types: `general.name`, `author`, `version`, `organization`, `finetune`, `basename`, `description`, `quantized_by`, `size_label`, `license.name`, `license.link`, `url`, `doi`, `uuid`, `repo_url`, `source.*`, the `general.sampling.*` keys, `general.tags`, `general.languages`. `general.base_models` / `general.datasets` are lists of objects, written as `general.base_model.count` plus `general.base_model.<i>.<field>` and replacing the model's existing list. Any other key is written with the GGUF type of its JSON value. Invalid values are refused, naming the key. |
| `--model-name <name>` | The exported `general.name`; wins over `--metadata`. |
| `--dry-run` | Print the files the export would write (`path: n_tensors = N, total_size = B`) and write nothing. |
| `--print-supported-models` | List the convertible architectures (`TEXT models:` / `MMPROJ models:`) and exit. |
| `--split-max-tensors <n>`, `--split-max-size <n>[k\|m\|g]` | Split into llama.cpp shard files `<name>-0000i-of-0000n.gguf`. |
| `--no-tensor-first-split` | The first shard holds only metadata, with every tensor in the following shards; this always writes at least two files. |
| `--mode`, `--overwrite`, `--vocab-only` | Export mode (`native`, `compat`, `compat-exact`), replacing existing files, tokenizer-only export. |

The output file name may contain `{ftype}` / `{outtype}` (lower case) or `{FTYPE}` / `{OUTTYPE}` (upper case). They are replaced with the exported file type, e.g. `nftool export ./Qwen3-8B ./qwen3-8b-{ftype}.gguf --outtype q8_0` writes `qwen3-8b-q8_0.gguf`. With an explicit `--outtype`, `general.file_type` is that type, as in the converter.

Examples:

```
nftool export ./Llama-3.2-1B ./llama-tq2.gguf --outtype tq2_0
nftool export ./my-lora ./my-lora.gguf --lora --base ./Llama-3.2-1B --outtype f16
nftool export ./model ./model.gguf --metadata meta.json --model-name "My Model" --split-max-size 2g --dry-run
```

`nftool export-lora --peft` runs the same adapter conversion with the default `--outtype`.

#### Draft selection (`--mtp`, `--no-nextn`, `--dspark`)

```sh
# the target without its NextN head, then the head alone
nftool export DeepSeek-V4-Flash/ deepseek-v4.gguf --no-nextn
nftool export DeepSeek-V4-Flash/ deepseek-v4-mtp.gguf --mtp

# DeepSeek-V4.1 / Mistral Large 4 DSpark drafts, written as DFLASH drafters
nftool export DeepSeek-V4.1-Flash/ deepseek-v4.1-dspark.gguf --dspark
nftool export Mistral-Large-4/ mistral-large-4-dspark.gguf --dspark
```

The flags apply to a converted Hugging Face checkpoint, not to a GGUF source, and cannot be combined with
`--mmproj` or `--lora`. `--mtp` and `--no-nextn` are refused for an architecture whose converter has no MTP export
(`supports_mtp_export`), as `convert_hf_to_gguf.py` does; `--dspark` is refused for every other target with the
converter's message. The DSpark draft is selected from the target's `architectures[0]` (or
`text_config.architectures[0]`); the checkpoint never names the draft.

#### Converting vision/audio projector towers (`--mmproj`)

`nftool export --mmproj <checkpoint> <out.gguf>` converts the vision or audio tower of a multimodal Hugging Face
checkpoint into an mmproj GGUF, the equivalent of llama.cpp's `convert_hf_to_gguf.py --mmproj`. The output has
`general.type=mmproj`, the `clip.*` encoder hyperparameters, the tower and projector tensors, and no vocabulary.
`--print-supported-models` lists the convertible projector architectures under `MMPROJ models:`.

Vision families:

| Family | Hugging Face architectures |
| --- | --- |
| Pixtral / LLaVA / Mistral 3 | `LlavaForConditionalGeneration`, `Mistral3ForConditionalGeneration`, and Mistral-native checkpoints (`params.json` without `config.json`) |
| Mistral 3 embedding tower | `Mistral3Model` (requires a headless text model, `is_causal: false`) |
| LightOnOCR | `LightOnOCRForConditionalGeneration` |
| Qwen2-VL / Qwen2.5-VL / Sarashina 2 | `Qwen2VLForConditionalGeneration`, `Qwen2VLModel`, `Qwen2_5_VLForConditionalGeneration`, `Sarashina2VisionForCausalLM` |
| Qwen3-VL | `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`, `OpenJevModel`, `InternS1ProForConditionalGeneration` |
| GLM-4V / GLM-OCR / GLM-5.3-Flash | `Glm4vForConditionalGeneration`, `Glm4vMoeForConditionalGeneration`, `GlmOcrForConditionalGeneration`, `Glm5NextForConditionalGeneration` |
| Gemma 3 | `Gemma3ForConditionalGeneration` |
| Idefics3 / SmolVLM | `Idefics3ForConditionalGeneration`, `SmolVLMForConditionalGeneration` |
| InternVL | `InternVisionModel` |
| Llama 4 | `Llama4ForConditionalGeneration` |
| Kimi-VL | `KimiVLForConditionalGeneration` |
| DeepSeek-OCR | `DeepseekOCRForCausalLM`, `DeepseekOCR2ForCausalLM`, `UnlimitedOCRForCausalLM` |
| DeepSeek-V4 | `DeepseekV4ForCausalLM`, `DeepseekV41ForCausalLM` |
| MiniCPM-V | `MiniCPMV4_6ForConditionalGeneration`, `MiniCPMV4_7ForConditionalGeneration`, `MiniCPMOcrForConditionalGeneration`, `MiniCPMOCRForSequenceClassification` |
| LFM | `Lfm2VlForConditionalGeneration`, `Lfm3VlForConditionalGeneration` |
| Granite / Nemotron | `Granite4VisionForConditionalGeneration`, `NemotronH_Nano_VL_V2`, `RADIOModel` |
| Bailing | `BailingMoeV3VLForConditionalGeneration` |
| Qwen3-VL derivatives | `ClefModel`, `H2OLightningModel`, `Qwen4ExpForConditionalGeneration`, `PplxDeciderModel`, `VllmD3Model`, `Exaone4_5_ForConditionalGeneration` |
| CogVLM / Cohere | `CogVLMForCausalLM`, `Cohere2VisionForConditionalGeneration` |
| OCR towers | `DotsOCRForCausalLM`, `PaddleOCRVisionModel` |
| Janus / Youtu / Step / Hunyuan | `JanusForConditionalGeneration`, `YoutuVLForConditionalGeneration`, `StepVLForConditionalGeneration`, `Step3p7ForConditionalGeneration`, `HunYuanVLForConditionalGeneration` |
| SenseNova | `NEOVisionModel` |
| Kimi / GLM | `KimiK25ForConditionalGeneration`, `Glm5vForConditionalGeneration` |
| MiniMax | `MiniMaxM3SparseForConditionalGeneration` |
| Muse | `MuseGlimmerForConditionalGeneration`, `MuseSparkForConditionalGeneration` |
| Phi-4 | `Phi4ForCausalLMV` |
| dots3-note | `Dots3NoteForCausalLM`, `Dots3NoteForConditionalGeneration` |

Audio families:

| Family | Hugging Face architectures |
| --- | --- |
| Qwen2-Audio | `Qwen2AudioForConditionalGeneration` |
| Ultravox | `UltravoxModel` |
| Voxtral | `VoxtralForConditionalGeneration` |
| Audio-Flamingo-3 | `AudioFlamingo3ForConditionalGeneration` |
| MERaLiON-2 | `MERaLiON2ForConditionalGeneration` |
| GLM-ASR | `GlmasrModel` |
| LFM2-Audio | `Lfm2AudioForConditionalGeneration` |
| Granite Speech | `GraniteSpeechForConditionalGeneration`, `GraniteSpeechPlusForConditionalGeneration` |
| Voxtral Realtime | `VoxtralRealtimeForConditionalGeneration` |
| Qwen3-ASR | `Qwen3ASRForConditionalGeneration` |

Vision and audio in one file:

| Family | Hugging Face architectures |
| --- | --- |
| Qwen2.5-Omni / Qwen3-Omni | `Qwen2_5OmniModel`, `Qwen3OmniMoeForConditionalGeneration` |
| Gemma 3n / Gemma 4 | `Gemma3nForConditionalGeneration`, `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`, `EmbeddingGemma2Model` |
| MiMo-V2 | `MiMoV2ForCausalLM` (needs the checkpoint's side tokenizer) |
| D1-Omni / LFM3-Omni | `D1OmniModel`, `Lfm3OmniForConditionalGeneration` |
| Nemotron Omni | `NemotronH_Omni_Reasoning_V3` |

Speech generators (`clip.has_gen_audio_encoder` and the `clip.gen.audio.*` keys):

| Family | Hugging Face architectures |
| --- | --- |
| Qwen3-TTS | `Qwen3TTSForConditionalGeneration` |
| PocketTTS | `PocketTTSModel` |
| Soprano | `SopranoModel` (weights from `decoder.pth` beside the checkpoint; its ISTFT window must be the periodic Hann window) |
| LFM3-Audio | `Lfm3AudioForConditionalGeneration` (audio encoder, depthformer and liquid codec) |

Muse-Spark, Nemotron Nano V2 VL, dots3-note and the vision-and-audio families write both encoder types into one file (`clip.has_vision_encoder` and `clip.has_audio_encoder`, with `clip.vision.projector_type` and `clip.audio.projector_type`) and number their tensor blocks with the converter's 128-block map. Nemotron Omni follows llama.cpp: its file declares the vision encoder only (`clip.projector_type = nemotron_omni`) while the Parakeet audio weights are still written under `a.*`. Every other row writes `clip.projector_type`. A Mistral-native checkpoint's `[IMG_BREAK]` row may be a `-1` placeholder; it is then resolved from the bundled `tekken.json` or `tokenizer.json`. A checkpoint missing a required side file (Soprano's `decoder.pth`, MiMo's tokenizer) or carrying constants the converter hard-codes differently is refused with the key or file named.

The normalization colors come from `preprocessor_config.json` / `processor_config.json` (with `media_proc_cfg` and
`image_processor` hoisted, as the converter does). `--mmproj` on a text-only checkpoint fails with "has no
projector conversion"; a projector architecture NeuralFree does not convert fails naming its llama.cpp converter
class and source file.

### nftool export-lora

`nftool export-lora <base-model> <out.gguf> [options]` merges LoRA adapters into a base model and writes the merged GGUF, or converts a PEFT adapter directory into a GGUF adapter file: llama.cpp's `export-lora` and `convert_lora_to_gguf.py` in one command.

| Flag | Meaning |
|---|---|
| `--lora <adapter>` | Merge this adapter at scale 1: a GGUF adapter file or a PEFT directory (`adapter_config.json` with `adapter_model.safetensors`). Repeatable |
| `--lora-scaled <adapter> <scale>` | Merge this adapter at the given scale. Repeatable; the deltas sum in the order given |
| `--peft <dir>` | Convert this PEFT directory to a GGUF adapter file instead; the base model fixes the architecture's tensor names. Excludes `--lora` |
| `--overwrite` | Replace an existing output; without it an existing file is refused before anything is written |

A merge adds `scale * alpha / rank * B·A` to each adapted tensor (a plain scale when the adapter has no alpha), in F32: a quantized base tensor is dequantized and the merged tensor is written as F16, while an F32 base tensor stays F32. Untouched tensors are copied byte for byte, the base metadata is kept, and `general.file_type` becomes 1 (mostly F16). Several adapters must adapt the same tensors; merge them one at a time otherwise. A split base model, an adapter of another architecture, factor shapes that do not fit the base tensor, an adapter tensor the base lacks, and an aLoRA (its delta starts at its invocation tokens, which a merged copy cannot express) are refused by name. A PEFT adapter on a GGUF base needs the base's Hugging Face `config.json`, from `-Dneuralfree.lora.base.config=<dir>` or the adapter's `base_model_name_or_path` when that is a readable directory, as the runtime adapter load does.

```
nftool export-lora base.gguf merged.gguf --lora adapter.gguf
nftool export-lora base.gguf merged.gguf --lora-scaled a.gguf 0.5 --lora-scaled b.gguf 1.5 --overwrite
nftool export-lora hf-base-dir adapter.gguf --peft peft-adapter-dir
```

## HTTP server

`server/` serves the OpenAI-compatible surface over the batch engine, with the llama.cpp and vLLM spellings of the shared endpoints:

| Endpoint | Serves |
|----------|--------|
| `GET /v1/models` | The served model, with `max_model_len` |
| `POST /v1/completions` | Text completions; `stream: true` sends server-sent events ending in `data: [DONE]` |
| `POST /v1/chat/completions` | Chat completions through the model's template, with tools, streaming and speech output |
| `POST /v1/embeddings` | Pooled vectors from an embedding or pooling model, in OpenAI's body |
| `POST /embeddings` (`/embedding`) | llama.cpp's native embeddings: per-token rows without pooling |
| `POST /v1/rerank` (`/rerank`, `/v1/reranking`, `/reranking`) | Documents scored against a query by a rank model |
| `POST /score` (`/v1/score`), `POST /classify`, `POST /pooling` | vLLM's scoring, classification and raw pooling ([below](#score-classify-and-pooling)) |
| `POST /generative_scoring` | Label-token probabilities of items read from a generative model's next-token distribution |
| `POST /v1/systemone`, `POST /v1/decisions` | Structured answers from a decision model, as llama.cpp's TypeSafe API and vLLM's Decisions API ([Decision models](#decision-models)) |
| `POST /v1/chat/completions/batch` | Several conversations in one request ([below](#batched-chat-completions)) |
| `POST /tokenize`, `POST /detokenize` | Token ids of text or chat messages, and text of token ids |
| `POST /v1/watermark/detect` | Scores a token sequence or text against the server's watermark key ([Watermarking generated text](#watermarking-generated-text)) |
| `POST /apply-template` | A chat body rendered through the template, as `prompt` |
| `GET /health` (`/v1/health`), `GET /props` | Load state; serving setup (model, context, slots, chat template) |
| `GET /ready` (`/readyz`), `/ping`, `GET /version`, `GET /load`, `GET /tokenizer_info`, `POST /start_profile`, `POST /stop_profile` | Probes, server state and operations ([below](#operational-endpoints)); the profile window is **served only with `--profiler-config`** |
| `GET /server_info`, `POST /reset_prefix_cache`, `POST /reset_mm_cache`, `POST /reset_encoder_cache`, `POST /abort_requests` | vLLM's development routes; **served only with `--dev-mode`** ([Development endpoints](#development-endpoints)) |
| `POST /v1/messages` (+ `/count_tokens`) | Anthropic's Messages API ([below](#anthropic-messages)) |
| `GET /v1/stream`, `DELETE /v1/stream`, `POST /v1/streams/lookup` | Reattach to, stop or look up a stream recorded under `X-Conversation-Id` ([Resumable streams](#resumable-streams)) |
| `GET /v1/kv_transfer/status` | The KV connector's role, address and transfer counters ([disaggregated prefill](#disaggregated-prefill---kv-transfer-config-kv_transfer_params)) |
| `POST /cohere/v2/chat`, `POST /v2/embed`, `POST /v2/rerank` | Cohere's v2 chat, embed and rerank APIs ([Cohere Chat v2](#cohere-chat-v2)) |
| `POST /invocations` | SageMaker's single inference route; the body's fields choose the endpoint that answers ([SageMaker hosting endpoints](#sagemaker-hosting-endpoints)) |
| `POST /predict` (`AIP_PREDICT_ROUTE`) | Vertex AI prediction when `AIP_MODE=PREDICTION` ([Vertex AI prediction](#vertex-ai-prediction)) |
| `POST /v1/responses` (`/responses`), `GET`/`DELETE /v1/responses/{id}`, `GET .../input_items`, `POST .../cancel` | OpenAI's Responses API ([below](#openai-responses)) |
| `POST /v1/responses/input_tokens`, `POST /v1/chat/completions/input_tokens` (and unversioned) | The prompt token count of a Responses or chat request |
| `POST /v1/chat/completions/render`, `/v1/completions/render`, `/v1/messages/render`, `/v1/responses/render`; `POST /v1/chat/completions/derender`, `/v1/completions/derender` | A request rendered to token ids and sampling parameters, and a token-in generate response turned back into the endpoint's answer ([Scale-out serving](#scale-out-serving)) |
| `POST /inference/v1/generate`, `POST /inference/v1/abort_requests` | Token-in/token-out generation for a separate front end, and its cancel route ([Scale-out serving](#scale-out-serving)) |
| `POST /completion` (`/completions`), `POST /infill` | llama.cpp's native completion and fill-in-the-middle ([below](#http-server)) |
| `POST /v1/chat/completions/control` | Ends the reasoning of a live chat completion ([below](#reasoning-budget)) |
| `GET /slots`, `POST /slots/{id}?action=save\|restore` | llama.cpp's slots ([below](#slots)) |
| `POST /sleep`, `POST /wake_up`, `GET /is_sleeping`, `POST /release_kv_cache_memory`, `POST /pause`, `POST /resume`, `GET /is_paused` | vLLM's sleep mode and pause routes; **served only with `--dev-mode`**, and `/sleep`, `/wake_up` and `/release_kv_cache_memory` also need `--enable-sleep-mode` ([Sleep, pause and checkpoint](#sleep-pause-and-checkpoint)) |
| `POST /checkpoint/prepare\|restore`, `GET /checkpoint/status` | vLLM's sleep-mode checkpoint window; **served only with `--enable-sleep-mode`**, not gated by `--dev-mode` ([Sleep, pause and checkpoint](#sleep-pause-and-checkpoint)) |
| `POST /init_weight_transfer_engine`, `/start_weight_update`, `POST /collective_rpc`, `/update_weights`, `/finish_weight_update`, `/update_weight_version`, `/shutdown_weight_transfer_engine`, `GET /weight_info` | vLLM's live weight updates; **served only with `--dev-mode`** ([Live weight updates](#live-weight-updates)) |
| `POST /tts`, `POST /v1/audio/speech` | Speech from audio codec codes or text ([Speech synthesis](#speech-synthesis)) |
| `POST /v1/audio/transcriptions` (+ `/audio/transcriptions`), `POST /v1/audio/translations` | Speech to text and translation ([Speech to text](#speech-to-text)) |
| `WS /v1/realtime` | Streaming speech-to-text WebSocket ([Realtime transcription](#realtime-transcription)) |
| `POST /models/load`, `POST /models/unload`, `POST`/`DELETE /models`, `GET /models/sse` | Router mode's model management ([Multi-model router](#multi-model-router)) |
| `GET /` (`/index.html`, `/ui`), `GET /ui/{asset}` | The bundled chat UI ([below](#web-ui)) |

`./gradlew :server:build` produces the uber-jar under `server/build/`. Start it with llama-server's flags ([Launching the server](#launching-the-server)) or with the model path property (`-Dneuralfree.server.model=<gguf path>`); every other setting reads its `neuralfree.*` property. Without a model the server starts unloaded: `/health` reports the error and every model request refuses with 503.

**Metrics.** `/metrics` serves Prometheus text with the JVM meters and the engine's own. The engine meters follow llama.cpp's set under a `neuralfree_` prefix: `prompt_tokens_total` (cached tokens excluded), `prompt_tokens_cached_total`, `prompt_seconds_total`, `tokens_predicted_total`, `tokens_predicted_seconds_total`, `decode_total`, `tokens_max`, `busy_slots_per_decode`, `requests_processing` and `requests_deferred`. Following vLLM, they add `preemptions_total`, `recomputed_tokens_total`, `kv_cache_usage` (the share of KV rows in use, with cached prefixes counted until evicted), `request_success_total{finished_reason="stop"|"length"|"abort"}` and `request_failures_total`, and for speculative decoding `spec_rounds_total`, `spec_drafted_tokens_total` and `spec_accepted_tokens_total` (the acceptance rate is `rate(accepted) / rate(drafted)`), and for multimodal requests `encoder_cache_rows` (rows of shared media embeddings the encoder cache holds), `encoder_cache_hits_total` (media chunks whose embeddings were already cached) and `encoder_cache_evictions_total`. They read the engine's state after its latest step, read zero before a model loads, and a scrape never loads one. Throughput is `rate()` over the counters.

**Model names.** `-Dneuralfree.server.model.name=<name>[,<name>...]` sets the names the server answers to, like vLLM's `--served-model-name` and llama.cpp's `--alias`. `/v1/models` lists each name with the model path as its `root`, and `/props` reports the first name as `model_alias`. A response carries the `model` its request named when that name is served, and the first name otherwise. With names configured, a request that names any other model is refused with 404, as in vLLM. Without them the model path is the only name and any requested name is accepted, as in llama.cpp.

One `EngineService` opens the model and pumps its `BatchEngine` on one thread; requests queue FIFO and admitted sequences decode together over the paged memory (`neuralfree.parallel` slots, [scheduling](#paged-kv-memory)). A client that disconnects cancels its sequences, whether it was streaming or waiting for a whole response. A request that runs longer than `neuralfree.server.request.timeout` (seconds, default 1800) is cancelled and refused with 503, and an embedding or rerank call that waits that long for an idle engine is refused too. `/models` and `/chat/completions` answer as llama.cpp's unversioned aliases of the OpenAI routes.

**Native completion.** `POST /completion` and `/completions` serve llama.cpp's native completion; its OpenAI form stays at `/v1/completions`. A `prompt` is a string, an array of token ids, an array that mixes strings and token ids (one prompt, with BOS only before a leading string), or `{"prompt_string": ...}`. An array that holds no token id at all is several prompts. `n_predict` sets the budget (-1 is unlimited), `n_probs` adds `completion_probabilities` (each with `id`, `token`, `bytes`, `logprob`, `top_logprobs`), and `response_fields` picks nested fields by `a/b` paths. Every sampler and grammar field above applies too. One prompt answers one object with `content`, `tokens`, `id_slot`, `stop_type` (`eos`, `word`, `limit`), `stopping_word`, `tokens_predicted`, `tokens_evaluated`, `tokens_cached`, `generation_settings` and `timings`; several prompts answer an array of these. A stream sends one `data:` chunk per generated token that released text (`content`, `tokens`, `stop: false`, `tokens_predicted`) and ends each prompt with its final object, with no `[DONE]`. Text held back for a possible stop string arrives in the chunk of the token that releases it. `post_sampling_probs` is refused, because the probabilities are the model's own, taken before the sampler chain.

On `/completion`, `"post_sampling_probs": true` with `n_probs` reports the sampler chain's probabilities: each entry
carries `prob` instead of `logprob` and `top_probs` instead of `top_logprobs`, tokens the chain removed are left out,
and `generation_settings.post_sampling_probs` echoes the setting. Without the flag the probabilities are the model's
own, over the whole vocabulary. `"return_tokens": true` lists the generated token ids in the final object's `tokens`;
without it `tokens` is empty there, while each streamed chunk always carries its token.

**Embeddings.** `/v1/embeddings` tokenizes text with the vocabulary's special tokens (a BERT model's CLS and SEP; `add_special_tokens: false` leaves them out) and returns L2-normalized vectors, as llama.cpp and vLLM do. `embd_normalize` picks llama.cpp's mode (-1 none, 0 max absolute in the int16 range, 1 taxicab, 2 euclidean, larger for a p-norm), and `use_activation` or `normalize` set to false return the raw pooled vector. `dimensions` keeps the first n values and normalizes after the cut, vLLM's matryoshka truncation. `truncate_prompt_tokens` keeps the last n tokens (-1 for the context length). `encoding_format: "base64"` writes float32 bytes, little-endian unless `endianness` is `big`. A model without pooling is refused there. Its per-token rows come from llama.cpp's native `POST /embeddings` (and `/embedding`), which takes `input` or `content` in llama.cpp's prompt forms and answers an array of `{"index", "embedding"}`: `[vector]` for a pooled model, one unnormalized row per token otherwise.

**Chat-message embeddings.** `/v1/embeddings`, `/pooling` and `/classify` take chat messages wherever `input` is allowed: `messages` holds one conversation or an array of conversations, and `input` may hold one conversation (`[{...}]`) or several (`[[{...}], [{...}]]`). Each conversation renders through the model's chat template (`add_generation_prompt`, `continue_final_message` and `chat_template_kwargs` apply) and embeds once.

**Infill.** `POST /infill` answers like `/completion`, but over llama.cpp's fill-in-the-middle prompt. `input_prefix` and `input_suffix` are required (a string, or an array mixing strings and token ids). `input_extra` is an optional array of `{"filename", "text"}` context chunks, and `prompt` is an optional string appended to the prefix. The prompt follows the repo-level FIM layout: `[FIM_REP]` and `[FIM_SEP]` file headers when the vocabulary has those tokens, then `[FIM_PRE]prefix prompt[FIM_SUF]suffix[FIM_MID]`. The prefix keeps its last three quarters of a 2048-token batch, the suffix fills the remaining quarter, and the extra context gets what the context has left after the batch and twice `n_predict`. `-Dneuralfree.server.spm.infill=true` puts the suffix before the prefix, like llama.cpp's `--spm-infill`. A model without FIM prefix, suffix and middle tokens is refused with 501.

**Sampling.** `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `repeat_penalty` (`repetition_penalty`), `presence_penalty`, `frequency_penalty`, `seed`, `stop`, `max_tokens` (`max_completion_tokens`), `min_tokens`, `ignore_eos` and `bad_words` map onto `SamplerConfig` and the request stages, and so do llama.cpp's sampler fields: `dynatemp_range`, `dynatemp_exponent`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `repeat_last_n`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers` (at most 64 distinct), `mirostat`, `mirostat_tau`, `mirostat_eta`, `adaptive_target`, `adaptive_decay`, `min_keep` and `samplers`. `samplers` is an array of stage names (`top-k`, `topk`, `nucleus` and llama.cpp's other spellings) or a string of stage characters (`dkypsmtxiea`). Only keys that are present and not null change the llama.cpp defaults, and an out-of-range value or an unknown sampler is refused with 400. `logit_bias` maps onto the sampler's first stage (`BiasStage`): token id to an additive bias in [-100, 100], where -100 and below ban the token.

**Choices and prompts.** `n` (1 to 128) decodes that many choices side by side; with a `seed`, choice `i` samples with `seed + i` (vLLM's parallel sampling), without one each choice draws its own. A completion's `prompt` takes every OpenAI form — a string, an array of strings, an array of token ids, or an array of token-id arrays (ids outside the vocabulary refuse) — and choice `p * n + i` answers prompt `p`; `echo` prepends the prompt. Usage counts each prompt once and sums the choices' tokens. In a stream, each chat choice opens with its own `role` delta; the last finish chunk carries the usage, or under `stream_options.include_usage` a final chunk with empty `choices` does.

**Logprobs.** Chat takes `logprobs: true` with `top_logprobs` (0 to 20) and answers `logprobs.content` entries (`token`, `logprob`, `bytes`, `top_logprobs`); completions take an integer `logprobs` and answer the legacy `tokens`, `token_logprobs`, `top_logprobs` and `text_offset` arrays; llama.cpp's `n_probs` selects the count on either. A stream carries each token's entry on its chunk.

**vLLM request extras.** `stop_token_ids` (token ids inside the vocabulary) end a choice like an
end-of-generation token: the token itself stays out of the text and the finish reason is `stop`; they stop even
under `ignore_eos`. `include_stop_str_in_output: true` keeps the matched `stop` string in the text instead of
cutting before it. `allowed_token_ids` confines sampling to those ids (every other id's logit is masked before the
chain; an empty or out-of-vocabulary list refuses). `skip_special_tokens` defaults to true (control and
end-of-generation tokens contribute no text, as llama.cpp's `token_to_piece` with `special = false`); false
renders their texts, and `spaces_between_special_tokens` (default true) puts a space between consecutive special
tokens' texts. `truncate_prompt_tokens` bounds a prompt to its first n tokens; `truncation_side: "left"` keeps the
last n instead (on the pooling routes `left` is the default, as in vLLM, and `right` keeps the first n) (values below 1 refuse, and the echo of a truncated prompt follows the kept tokens).
`prompt_logprobs` answers each prompt token's log probabilities — the raw distribution's log-softmax, the prompt
token leading with its exact vocabulary rank: `0` reports the prompt token alone, `n` adds its `n` best
alternatives, `-1` reports every vocabulary token, and with `echo` and no explicit value the `logprobs` count
answers. The first prompt token has a null entry (no row predicts it), the field sits on the completion choice and
at the top of the chat response, and a stream with alternatives or `-1` refuses. `return_token_ids: true` adds
`token_ids` (the generated ids) and `prompt_token_ids` — on the choice for completions, on the choice and the
response for chat; a stream carries the prompt ids on the first chunk of a choice and one delta id per text chunk.
`return_tokens_as_token_ids: true` renders every logprob token as `token_id:{id}`. `priority` (vLLM's request field, any 32-bit
integer) orders the batch engine under the `priority` scheduling policy, the lowest value first and ties by arrival;
under the default `fcfs` policy any value other than `0` is refused with a 400 naming `priority`. And `cache_salt` (vLLM's rules: a non-empty string of at most 128 characters
without `@`, `/`, `\` or NUL; `null` means none) is hashed into the first block of the prefix cache's hash chain, so
only requests of the same salt share cached blocks. Unsalted requests form a partition of their own. It is accepted
on completions, chat, Responses and Anthropic `/v1/messages`.
Usage always reports `prompt_tokens_details.cached_tokens`, the prompt tokens the prefix cache served; under
`stream_options`, `continuous_usage_stats: true` (with `include_usage`) puts the running usage on every chunk
instead of the last one. An `X-Request-Id` header names the response id (`cmpl-<id>` / `chatcmpl-<id>`), like
vLLM's request id.

Every completion and chat choice carries vLLM's `stop_reason`: the matched `stop` string, or the `stop_token_ids`
id the choice ended on, and null for an end-of-generation token or a length end (the vocabulary's EOS token counts
as a stop token id only under `ignore_eos`). A stream reports it on each choice's finish chunk. `echo: true` with
`logprobs` on `/v1/completions` merges the prompt's entries before the generated ones in the `tokens` /
`token_logprobs` / `top_logprobs` / `text_offset` arrays, as vLLM does: the first prompt token has a null logprob
and null alternatives (no row predicts it), the offsets run from 0 over the echoed text, and a stream carries the
prompt's entries on its echo chunk. `best_of` (at least `n`, at most 128) generates that many candidates per prompt
and returns the best `n` by cumulative log probability; usage counts the returned choices alone, and a stream with
`best_of` above `n` refuses. `logprob_token_ids` (at most 128 vocabulary ids, needs `logprobs`) makes every
generated token's alternatives list exactly the sampled token followed by those ids, past the `top_logprobs`
count (the speculative path included); an id the sampler chain masked out reports -9999.0. `prompt_logprobs` works on the single-sequence
generation loop as well as the batch engine; it refuses for `prompt_embeds` and encoder-decoder models.
`vllm_xargs` carries the arguments of [custom logits processors](#custom-logits-processors); `kv_transfer_params` runs a [disaggregated prefill](#disaggregated-prefill---kv-transfer-config-kv_transfer_params) transfer.

**Logprobs: what they score.** By default logprobs score the model's own distribution: the raw logits row, before
penalties, logit bias, temperature or truncation. This is vLLM's default `raw_logprobs`, and it matches llama.cpp's
pre-sampling probabilities. `--logprobs-mode` (property `neuralfree.server.logprobs.mode`) picks another mode:

| Mode | Scores |
|------|--------|
| `raw_logprobs` (default) | log-softmax of the model's logits |
| `raw_logits` | the model's logits |
| `processed_logprobs` | log-softmax of the sampler chain's candidates. Greedy: after penalties and bias. Sampled: after temperature and top-k/top-p/min-p; tokens the chain removed are `-inf` (reported as `-9999.0`) |
| `processed_logits` | the chain's candidate logits, on the same rule |

Prompt logprobs have no sampler chain: the logits modes report raw logits for them, and the logprobs modes the raw
log-softmax. In Java, `GenerationRequest.builder().logprobsMode(LogprobsMode.PROCESSED_LOGPROBS)`. Every token
speculative decoding emits carries the score plain decoding would give it: the target's row at that token's verify
position, in every verify mode.

More on logprobs requests:
- `logprob_token_ids` lists repeated ids once; chat lists them under `top_logprobs`, completions in the
  `top_logprobs` dict. An empty list asks for nothing.
- With `echo`, each `top_logprobs` dict holds the token itself plus the `logprobs` most likely tokens. On chat,
  `echo` fills an absent `prompt_logprobs` with `top_logprobs` (default 0), so an echoed chat response carries
  `prompt_logprobs`.
- Streaming is refused only for an explicit `prompt_logprobs` above 0 or equal to -1; `prompt_logprobs: 0` and the
  echo default both stream.
- Each `prompt_logprobs` entry lists the prompt token first, then the top-n tokens; a top-n token that is the prompt
  token itself is listed once. With `-1` the entry lists the prompt token, then every vocabulary token, best first.

Prompt token-id scoring (Java API) scores chosen ids at every prompt position after a start:

```java
GenerationRequest request = GenerationRequest.builder()
        .promptLogprobStart(4)                                 // positions 0..4 are not scored
        .promptLogprobTokenIds(new int[][]{{17, 5}, {6}, ...}) // one row per position 5..len-1
        .build();
float[][] scores = engine.submit(prompt, request, sink).result().promptTokenIdLogprobs();
```

There must be exactly `prompt.length - 1 - start` rows; shorter rows are padded with `-1`, and a pad scores `-inf`. A
start needs `promptLogprobs` or `promptLogprobTokenIds`, and with `promptLogprobs` the positions up to and including
the start are `null`. A prompt of embedding rows (`prompt_embeds`) refuses both by name, since it carries no token id
per row.

The same token controls are accepted by llama.cpp's `POST /completion`.

**Repetition detection.** `repetition_detection` ends a choice early when the output starts looping:

```json
{"repetition_detection": {"max_pattern_size": 8, "min_pattern_size": 1, "min_count": 3}}
```

After each generated token, the server checks whether the output ends with the same pattern of
`min_pattern_size`..`max_pattern_size` tokens repeated `min_count` times in a row (for example `abcabcabc`). If it
does, the choice ends. Each member is an integer and defaults to `0`:

- `max_pattern_size`: the longest pattern checked. `0` turns detection off.
- `min_pattern_size`: the shortest pattern checked. `0` means 1. It must not exceed `max_pattern_size`.
- `min_count`: how many back-to-back repeats end generation. It must be at least 2 when detection is on.

The check starts only once `min_tokens` tokens have been generated. The repeated tokens stay in the output. When
the same token also hits an end-of-generation or stop token, or the `max_tokens` budget, that reason is reported
instead. A stop string completed by the same token is reported as a stop string.

A repetition stop is reported as:

- `finish_reason: "repetition"` on the OpenAI endpoints, plain and streamed;
- `stop_type: "limit"` on llama.cpp's `/completion`;
- the `finished_reason="repetition"` label of the `neuralfree.request.success` counter.

From Java, set it with `GenerationRequest.builder().repetitionDetection(RepetitionDetection.of(max, min, count))`;
the result's `stopReason()` is `StopReason.REPETITION`.

**Output constraints.** One of `grammar`, `json_schema`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `guided_structural_tag` / `structural_tag`, `response_format` (`text`, `json_object`, `json_schema`, `structural_tag` ([format](#structural-tags-the-xgrammar-format-object))) or vLLM's `structured_outputs` (`json`, `regex`, `choice`, `grammar`, `json_object`, `structural_tag`) constrains the output ([Grammar-constrained generation](#grammar-constrained-generation)); two at once refuse. A regex matches the raw output, with or without `^`/`$` anchors. In chat completions the constraint holds the answer rather than the reasoning before it. A thinking model whose prompt leaves `<think>` open writes freely until its end tag, and the constraint starts after that tag and any whitespace. gpt-oss is constrained after its `<|channel|>final` header (a `<|constrain|>` type may sit in it), and Muse after the header of its message to the user. Other models are constrained from the first token, as are text completions.

**Chat.** `messages` (string or text-part content, assistant `tool_calls` and `reasoning` / `reasoning_content`, `tool` results), `add_generation_prompt`, `continue_final_message`, `enable_thinking`, `reasoning_effort` and `chat_template_kwargs` reach the template. Reasoning output streams as `reasoning_content` deltas and lands in `message.reasoning_content`. Image and audio parts refuse by name: the text chat path has no media encoder.

Finish reasons: `stop` (end of generation, stop string, cancellation), `length` (token budget, full context) and `tool_calls`.

### Beam search

Set `use_beam_search` to `true` on a chat completion or text completion to answer with beam search instead
of sampling. `beam_width` is the request's `n`, and `length_penalty` (number, default `1.0`) shapes the
ranking: the score of a beam is its cumulative log probability divided by its generated-token count raised
to the penalty. Higher values favour longer beams; `0` ranks by the bare cumulative log probability.

The response is one final (non-streaming) body: chat choices carry each ranked beam as a message, completion
choices each ranked beam as `text`, ordered best-first with `index` 0..n-1. `logprobs` render each step's
top entries as usual, and `usage.prompt_tokens_details.cached_tokens` counts the prefix-cache rows the
search's per-step submissions reused.

The search resubmits each beam's full token sequence every step, so a beam shares the prompt's cached KV
pages; the beams grow apart after the first step.

Not supported beside `use_beam_search` (the request is refused, naming the field): `stream`,
`response_format`/grammar constraints, audio modalities, server tool execution, `best_of` greater than `n`,
media content parts, and on completions `echo` with `logprobs`. Request features the engine cannot serve
under the search — `stop` strings, `bad_words`, `logit_bias`, `min_tokens`, prompt logprobs, repetition
detection, reasoning budgets, `grp_attn_n` — are refused the same way. Encoder-decoder models refuse the
search entirely. Batch requests and the Responses API do not accept `use_beam_search`.

### Prompt embeddings

Requests may carry precomputed embedding rows instead of text or token ids (vLLM's `prompt_embeds`). The rows replace the token-table lookup at the model's input; the model's position additions and input norm still apply, so a row holding a token-embedding row produces exactly that token's continuation. Row width must equal the model's embedding width (`n_embd`); anything else is a 400 naming the field. Embedding rows run on the CPU and GPU backends, mixed with token rows in one batch. On a model without rotary positions (`mamba2` and the other rope-less architectures) they take plain sequence positions, one per row, as llama.cpp's `set_position_normal` does; an M-RoPE position grid is refused by name there, since the model reads one position section where the grid carries four. A model whose blocks read the step's token ids refuses embedding rows by name.

**Completions.** `prompt_embeds` is an array of finite numbers — one row per array element, in JSON:

- a flat row `[0.1, 0.2, ...]` — one prompt of one row;
- a list of rows `[[...], [...]]` — one prompt, one row per inner array;
- a list of prompts `[[[...]], [[...]]]` — one prompt per entry, as vLLM's `prompt_embeds` list of tensors.

`prompt`, `prompt_embeds`, or both may be given; with both, the prompts run side by side (vLLM's order: the embeds prompts first). `n` choices apply per prompt as usual. `echo` and `truncate_prompt_tokens` with `prompt_embeds` are a 400 (there is no prompt text to echo or cut), and `usage.prompt_tokens` counts the rows.

**Chat parts.** A message content part of type `image_embeds` or `audio_embeds` carries precomputed rows that the prompt splices where the part sits (vLLM's `<##PROMPT_EMBEDS##>` placeholder):

```json
{"messages": [{"role": "user", "content": [
  {"type": "text", "text": "look "},
  {"type": "image_embeds", "image_embeds": [[...], [...]]},
  {"type": "text", "text": " now"}]}]}
```

The part's payload is a flat row or a list of rows (each of the model's embedding width). The text around the parts tokenizes segment by segment; `usage.prompt_tokens` counts the prompt's tokens plus the rows. Base64 tensors and map payloads are refused with a 400 naming the part. `truncate_prompt_tokens` cannot be combined with these parts.

**Java API.**

- `EmbedsPrompt.ofRows(float[][])` / `EmbedsPrompt.of(int[][] tokenRuns, float[][][] embedsRuns)` — the immutable prompt (rows validated: non-empty, one finite width).
- `GenerationLoop.generate(EmbedsPrompt, GenerationRequest, TokenCallback)` — single-sequence generation.
- `BatchEngine.submit(EmbedsPrompt, GenerationRequest, TokenCallback)` — batched generation; row sequences share the prefix cache, each row keyed by the digest of its float bytes.
- `LLMModel.generateEmbeds(float[][] promptEmbeds, GenerationRequest, TokenCallback)` — the model-level API (GGUFModel, GGUFModelGPU).

### Score, classify and pooling

`POST /score` (also `/v1/score`) scores two sides against each other. The sides are `queries` and `documents`, `queries` and `items`, `data_1` and `data_2`, or `text_1` and `text_2`, each a string or an array of strings. A sequence classifier (a RANK model or a pooled classifier head) scores every query-document pair through its classifier, as `/v1/rerank` does; an embedding model answers the cosine similarity of the pooled vectors. One query is scored against every document; otherwise the sides must have the same length. `max_tokens_per_query` and `max_tokens_per_doc` (0 keeps all) keep the first tokens of each side.

```json
{"queries": "What is NeuralFree?", "documents": ["An inference engine", "A recipe"], "max_tokens_per_doc": 64}
```

The answer is a list of `{"index", "object": "score", "score"}` entries with `usage`.

`POST /classify` classifies each input with a sequence classifier. The input is `input` (a string, strings, token ids or token-id arrays) or chat `messages` rendered through the template. Each entry answers `probs`, the best `label` (from `classifier.output_labels`, else the index) and `num_classes`. A one-output head (a reward or relevance score) reports its raw score as `probs` with label `"0"`, since a softmax over one output would flatten it.

`POST /pooling` runs one pooling task and answers the raw pooled output. `task` is `embed` (the pooled vector; the default of an embedding model), `token_embed` (one row per token; needs a model that pools by `none`) `classify` (a sequence classifier's probabilities, the default of a sequence classifier or decision model; a decision head answers its raw scores) or `token_classify` (one score row per token, `data` an array of arrays or base64; the default of a token classifier, and answered by sequence classifiers too, as vLLM's `for_seq_cls` dispatch does). It takes `/classify`'s input forms plus `encoding_format` (`float` or `base64`), `embed_dtype`, `endianness`, `use_activation`/`normalize` and `embd_normalize`. `dimensions` is refused: the output follows the model's own pooling.

`POST /generative_scoring` scores items with a generative model through its next-token distribution. It takes `query` (text or token ids), `items` (texts or token-id arrays), `label_token_ids`, `apply_softmax` (normalize over the label ids, default true; false gives the full-vocabulary probabilities), `item_first` (put the items before the query, default false) and `add_special_tokens` (default true, for the query only). Each item answers the probability of the first label token after its prompt, and prompts are cut to the context minus one token.

A model that cannot serve one of these is refused by name: a generative model has no pooled output, and `/pooling` and `/classify` name the pooling kind the model carries. Pooling tasks other than `embed`, `classify`, `token_classify` and `token_embed` (such as `plugin`) are refused by name. A pooling request may name a per-request adapter ([Per-request LoRA adapters](#per-request-lora-adapters)); media inputs pool under it too.

A score or rerank request takes vLLM's scoring fields explicitly. `instruction` is a task instruction applied
through the model's `tokenizer.chat_template.rerank` prompt, which may place it with an `{instruction}`
placeholder beside `{query}` and `{document}`; the placeholder stays as written when no instruction is given.
`chat_template_kwargs` carries the same instruction (an explicit `chat_template_kwargs["instruction"]` wins over
the top-level field); any other kwarg, and the reserved `chat_template`, `tools` and `tokenize` keys, are
refused by name. `instruction` is refused by name when the model has no place for it: the rerank prompt is
missing, holds no `{instruction}` placeholder, or the model is an embedding model scoring by cosine, which
reads no instruction.

`max_tokens_per_query` and `max_tokens_per_doc` (0 keeps all) keep the first tokens of each side, on `/rerank`
as on `/score`; a value at or past the model's context refuses. `truncate_prompt_tokens` (-1 for the context)
cuts the composed pair instead, keeping its last tokens, or its first under `"truncation_side": "right"`;
beside an image or audio side it refuses by name, since media rows keep their length. A negative `top_n`
refuses. `request_id` names the response: every pooling route (`/v1/embeddings`, `/score`, `/rerank`,
`/classify`, `/pooling`) echoes it after the id prefix (`"score-my-id"`), as vLLM does.

Every pooling request validates `priority` (any value but 0 refuses under the fcfs scheduling policy) and
`cache_salt` (a non-empty string of at most 128 characters without `@`, `/`, `\` or NUL), and refuses `padding`
(other than `do_not_pad`) and a non-empty `mm_processor_kwargs` by name, whose engine effects this server does
not run. The removed `normalize` refuses with vLLM's message on `/v1/embeddings`, `/classify` and `/pooling`
(`normalize` was removed; use `use_activation`), and `task` values `score` and `encode` refuse with vLLM's
removal messages (`/score` and `/rerank`, whose protocol carries no task, ignore them).

`use_activation` applies or skips the classifier's final activation per request, as vLLM serves it:

- On `/score` and `/rerank`, a cross-encoder pair answers the model's score by default; `false` answers the raw head output (before the head's own softmax on the qwen3 and qwen3vl rerankers). An embedding model scoring by cosine has no activation, so the field changes nothing there.
- On `/classify`, `false` answers the raw classifier scores, which do not sum to one, instead of the softmaxed `probs`.
- A value other than `true` or `false` is refused by name.

Every pooling route (`/v1/embeddings`, `/score`, `/rerank`, `/classify`, `/pooling` and Cohere's `/v2/embed`) names its response from the `X-Request-Id` header first and the body's `request_id` second, as the generation routes do. Cohere's protocol has no `request_id`, so there the header alone names it.

`padding` other than `do_not_pad` is refused because the pooling forward reads no attention mask: padding every prompt to the context would put the pad tokens into every attention row and change the pooled result.

**Classifier heads.** The GGUF key `{arch}.classifier.scope` says where the classifier chain (`cls`, the activation, `cls.norm`, `cls.output`) runs: `token` (NONE pooling; every token is scored) or `sequence` (a pooled type; the pooled row is scored). A file without the key keeps llama.cpp's behaviour, a classifier on RANK pooling only. `nftool export` converts these Hugging Face heads:

- Token classifiers: `BertForTokenClassification`, `RobertaForTokenClassification`, `XLMRobertaForTokenClassification`, `ModernBertForTokenClassification`.
- Sequence classifiers: `BertForSequenceClassification`, `RobertaForSequenceClassification`, `XLMRobertaForSequenceClassification`, `ModernBertForSequenceClassification` (the CLS row, or the configured `classifier_pooling` mean), `GPT2ForSequenceClassification` and `JambaForSequenceClassification` (the last row).
- Reward models: `Qwen2ForRewardModel` and `InternLM2ForRewardModel` give one raw score per token, the last row being the sequence reward; `Qwen2ForProcessRewardModel` gives one row of step scores per token, and clients pick the step rows (vLLM's `step_tag_id` filtering is not applied).

#### Late-interaction models (ColBERT)

`HF_ColBERT`, `ColBERTModernBertModel`, `ColBERTJinaRobertaModel` and `ColBERTLfm2Model` answer per-token
embeddings and score query/document pairs by MaxSim over the L2-normalized token rows.

- `POST /pooling` with `"task": "token_embed"` returns one L2-normalized vector per token
  (`{"data": [{"data": [[...], ...]}]}`). The pooled `embed` task refuses: a late-interaction model has no
  pooled embedding.
- `POST /score` takes `{"queries": "...", "documents": ["...", ...]}` and returns the MaxSim score of each
  document against the query.
- `POST /rerank` takes `{"query": "...", "documents": [...], "top_n": n}` and ranks by the same MaxSim;
  `instruction` refuses because instructions need a cross-encoder.
- `POST /score` and `POST /rerank` answer identical scores for identical inputs.

#### Sparse embeddings (SPLADE)

`BertSpladeSparseEmbeddingModel` (and any model carrying `classifier.sparse` metadata) embeds text into one
non-negative weight per vocabulary entry: the masked-LM head's scores go through `log1p(relu(x))` and reduce
over the token rows by `classifier.sparse` — `log1p_relu_max` (vLLM's default) or `log1p_relu_sum`. The leading
CLS row and trailing SEP row drop first (`remove_cls_sep`).

- The model answers the `embed` task by default on `/pooling`, and `/v1/embeddings` and `/v2/embed` answer the
  same way: one weight per vocabulary entry.
- `token_embed` answers the plain embedding rows (one L2-normalized vector per token), as vLLM's SPLADE pooler
  does.
- The weights stay raw and are never normalized, so scores stay comparable across queries. `embd_normalize` and
  `dimensions` answer 400 naming the field: normalization would rescale the weights and cutting the vector would
  misalign the vocabulary.
- `token_classify` answers 400: the per-token classifier scores reduce into the sparse vector.

#### Process-reward step filtering

`Qwen2ForProcessRewardModel` scores every token; a `token_classify` request can select the reward steps:

- `"step_tag_id": 151651` keeps only the rows whose token id matches.
- `"returned_token_ids": [0, 1]` keeps only those classifier output columns (vLLM's vocabulary-dimension
  indices, such as a model's good/bad logits).

Both apply only to `token_classify`; a request naming them with another task, on a model that is not a
per-token classifier, or a value that does not fit answers 400 naming the field.

#### Pooler metadata for classifier and embedding heads

- `classifier.pooler_activation` applies the final activation to classifier scores: `identity`, `sigmoid`,
  `softmax` or `tanh`. It resolves from the checkpoint's `problem_type` (regression = identity, multi-label =
  sigmoid, single-label = sigmoid for one output else softmax) and otherwise from the sentence-transformers
  `activation_fn` / `sbert_ce_default_activation_function`; unknown values refuse by name.
- `pooling.normalize` L2-normalizes embedding rows (vLLM's `PoolerNormalize`): the pooled vector, or every
  per-token row on a per-token embedding model.
- `classifier.token_direct` marks a sequence classifier whose `token_classify` runs only the final projection
  over every row, as vLLM's `for_seq_cls` does.

#### Sequence-classification token scores

For a sequence-classification model (`ModernBertForSequenceClassification` and the other
`*ForSequenceClassification` loaders), `token_classify` returns the final classifier's scores over every row
(vLLM's `for_seq_cls`), not the checkpoint's masked-LM prediction-head projection.

#### Multi-output embedders (BGE-M3) and sparse responses

`BgeM3EmbeddingModel` converts to a single GGUF with three heads: the CLS-pooled dense embedding,
the `colbert_linear` per-token projection (`cls.colbert`), and the `sparse_linear` per-token score
projection (`cls.sparse`). `/pooling` answers:

- `token` omitted or `embed`: the dense pooled vector.
- `token_embed`: one L2-normalized ColBERT row per token, minus the leading BOS row (vLLM's
  token-embed filter leaves the EOS row in place).
- `token_classify`: a `{token_id: weight}` object of the ReLU sparse weights, without the BOS and
  EOS rows; a repeated token id keeps its largest weight.
- `embed&token_classify`: one dense array holding the pooled vector followed by the flattened
  per-token scores.

A SPLADE sparse model answers `embed` on `/pooling` with a `{token_id: weight}` object of the
vocabulary's non-zero weights; `encoding_format: base64` is refused there. All other routes
(`/v1/embeddings`, `/v2/embed`, llama.cpp's `/embeddings` and `/embedding`) keep their dense float
arrays.

New metadata keys: `classifier.bos_token` and `classifier.eos_token` give the boundary ids the
per-token outputs filter at the prompt's ends; `classifier.step_tag_id` is the process-reward step
filter applied when a `token_classify` request omits `step_tag_id` (vLLM writes `151651` for
`Qwen2ForProcessRewardModel`).

`Qwen3ASRForcedAlignerForTokenClassification` converts its `thinker` text backbone with
`thinker.lm_head` as the per-token classifier (`classify_num` time bins, softmax scores by default);
`audio_tower`, `talker` and `code2wav` are not text weights and are dropped.

#### Model names for these heads

Accepted converter names include `HF_ColBERT`, `ColBERTModernBertModel`, `ColBERTJinaRobertaModel`,
`ColBERTLfm2Model`, `BertSpladeSparseEmbeddingModel`, `BertForTokenClassification`,
`ModernBertForTokenClassification`, `RobertaForTokenClassification`,
`XLMRobertaForTokenClassification`, `Qwen2ForProcessRewardModel`, `Qwen2ForRewardModel`,
`InternLM2ForRewardModel`, `LlamaBidirectionalForSequenceClassification`,
`GPT2ForSequenceClassification`, `JambaForSequenceClassification`, `BgeM3EmbeddingModel` and
`Qwen3ASRForcedAlignerForTokenClassification`.

`GteNewForSequenceClassification`, `OpenAIPrivacyFilterForTokenClassification`,
`LlamaNemotronVLForSequenceClassification` and `ClefOmniForJointSchema` are recognized and refused
by name with the missing backbone.

### Batched pooling beside generation

Embedding, rerank, score, classify and pooling requests are scheduled together with generations
instead of waiting for them. Over a paged memory (`--parallel` above 1, or `neuralfree.kv.paged`),
every pooling request joins the batch engine the generations run on. It prefills beside the running
sequences under the same `neuralfree.max.batched.tokens` budget, as vLLM schedules its pooling
requests and llama.cpp runs embedding tasks in slots of the shared batch. It never clears the prefix
cache: its committed rows feed the cache as a generation's do. A request with several inputs submits
them as one batch, so `/v1/embeddings` with an array of texts runs all its prompts in the engine's
steps.

Whether a pooling request reuses a cached prefix depends on its output:
- An output that reads only the last computed row (LAST pooling, or the RANK classifier over causal
  attention) reuses a cached prefix like a generation prompt.
- An output that pools every row (MEAN, CLS, per-token embeddings, decision scores) prefills every
  row itself, because reused rows' hidden states are not its own forward's.

A pooling-only model with pooling heads (an embedding or classifier model over a paged memory) opens
the batch engine for pooling requests alone, and a generation request to it is refused by name. A
bidirectional model's pooling prompt prefills whole, because its rows attend each other, so it must
fit one `neuralfree.ubatch`. Such an engine runs no prefix cache, as vLLM runs none for its
bidirectional pooling models.

A model whose memory is not paged (the memoryless BERT and CLEF families, or `--parallel 1`) pools
one input per forward on the engine's thread, once nothing else runs. `/generative_scoring` also
keeps its one-shot forwards.

### CLIP image and text embeddings

A CLIP two-tower GGUF (arch `clip` with `clip.has_text_encoder` / `clip.has_vision_encoder` and no `clip.projector_type`, as llama.cpp's `convert_image_encoder_to_gguf.py` writes it) serves as the model itself, like vLLM's `CLIPEmbeddingModel`. It answers the pooling routes only: `/v1/embeddings`, `/pooling`, `/score` and `/rerank`. Do not set `neuralfree.server.mmproj` for it.

The legacy GGUF stores CLIP's vocabulary without its BPE merges. Point `neuralfree.hf.tokenizer` at the checkpoint's tokenizer directory, which holds `tokenizer.json`, or `vocab.json` + `merges.txt`:

```
java -Dneuralfree.server.model=clip-vit-base-patch32.gguf \
     -Dneuralfree.hf.tokenizer=/models/clip-vit-base-patch32 -jar server.jar
```

Pooling comes from `clip.pooling_type` and defaults to `last`:
- `last` or `cls`: the text's last or first token, and the image's class token.
- `mean`: all tokens, or all image patches.

The text side is limited to `clip.text.context_length` tokens (77 for OpenAI CLIP).

**Text embeddings** use the usual `input` forms:

```json
{"model": "clip", "input": ["a photo of a cat", "a photo of a dog"]}
```

**Image embeddings** use vLLM's chat form: one conversation per image, holding a single `image_url` part and no text. The URL can be data, http(s) under the media connector's domain rules, or file under the local media root.

```json
{"model": "clip", "messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}]}]}
```

Text and image vectors share one embedding space. CLIP accepts text-only or image-only inputs, never both in one input, and at most one image per input.

**Score and rerank** take a string or `{"content": [parts]}` on every side:

```json
{"query": "a photo of a cat",
 "documents": [{"content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}]},
               "a photo of a dog"],
 "return_documents": true}
```

Embedding models score by the cosine similarity of the pooled vectors. With `return_documents`, rerank returns `{"text": ...}` for a text document and `{"multi_modal": [parts]}` for a content document. A RANK cross-encoder scores text pairs, and media pairs through a projector ([Media embeddings from vision-language models](#media-embeddings-from-vision-language-models)).

### Media embeddings from vision-language models

When the server runs a model with a media projector (`-Dneuralfree.server.mmproj=mmproj.gguf`), the pooling routes accept chat conversations that contain media:

- **Routes:** `/v1/embeddings`, `/pooling` (task `embed`), `/score` and `/rerank`.
- **Parts:** `image_url`, `input_audio` and `audio_url`, mixed freely with `text` parts.
- **How a conversation pools:** it is rendered through the model's chat template with media markers, then pooled as one sequence. The media rows sit where the projector places them.
- **Requirement:** the KV memory must be paged (`neuralfree.parallel > 1` or `neuralfree.kv.paged=true`).
- **Refusals:** without a projector, media parts are refused by name. Tasks that read only text (`classify`, `token_classify`, `token_embed`) refuse them too.

```bash
curl localhost:8080/v1/embeddings -d '{"messages": [[{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "a photo"}]}]]}'
```

Other pooling request changes:

- `/pooling` now accepts `"task": "embed"` on an embedding model.
- `truncate_prompt_tokens` now takes a `truncation_side`:
  - `left` (the default) keeps the last tokens.
  - `right` keeps the first tokens.

### jina_vl multimodal reranker

`jinaai/jina-reranker-m0` (`JinaVLForRanking`) loads straight from its Hugging Face directory.

- **How it scores:** the language model scores the prompt `**Document**:\n{document}\n**Query**:\n{query}` followed by the score token. The two-layer `score` head is applied with ReLU, and the result is reported as the relevance.
- **Vision tower:** convert the checkpoint's vision tower to an mmproj with llama.cpp's Qwen2-VL converter, then pass it with `-Dneuralfree.server.mmproj=...`.
- **Media pairs:** `/score` and `/rerank` take image entries (`{"content": [{"type": "image_url", ...}]}`) on either side. Each pair is scored as one prompt.

### CLIP and SigLIP dual encoders

Serve a two-tower CLIP GGUF, or a Hugging Face `CLIPModel` / `SiglipModel` checkpoint directory, as the model:

```bash
neuralfree serve -Dneuralfree.server.model=./siglip-base-patch16-224
```

- **Inputs:** text through `input`; images through `messages` with one `image_url` part. Both embed into one space.
- **Routes:** `/v1/embeddings`, `/pooling`, `/score`, `/rerank` and `/v2/embed`.
- **Scores:**
  - CLIP: the cosine similarity of the pair.
  - SigLIP: SigLIP's pair probability, `sigmoid(cos · exp(logit_scale) + logit_bias)`.
- **SigLIP tokenizer:** read from the checkpoint's `spiece.model`. A checkpoint without one needs `-Dneuralfree.hf.tokenizer=<dir>` for text inputs.
- **Device:** the towers' transformer blocks run on the GPU and follow the engine backend, or `-Dneuralfree.mmproj.device=<selector>` to choose a device.
  - `-Dneuralfree.mmproj.offload=false` keeps them on the CPU.
  - If the device cannot run the towers, they fall back to the CPU and the reason is logged.

### Cohere embed API

`POST /v2/embed` serves Cohere's embed API.

**Inputs.** Give exactly one of:
- `texts`: strings.
- `images`: image URLs or data URIs.
- `inputs`: objects whose `content` holds `text` and `image_url` parts.

**Options:**

| Field | Values |
|---|---|
| `truncate` | `END` (default; keep the first `max_tokens`), `START` (keep the last), or `NONE` (refuse inputs longer than `max_tokens`) |
| `output_dimension` | Keep the leading values of the normalized embedding |
| `embedding_types` | Any of `float` (default), `binary`, `ubinary`, `base64` |

Image and mixed inputs embed through the chat template, on a dual encoder or on a projector model. `input_type` is refused, because served models define no task instructions. `model` may name a per-request adapter, as on the other pooling routes.

```bash
curl localhost:8080/v2/embed -d '{"texts": ["hello"], "embedding_types": ["float", "binary"]}'
```

### Decision models

A decision model answers structured questions about a state instead of generating text. Two routes serve it, one per reference API, over every decision family the engine reads:

| `decision.type` | Read | Notes |
|---|---|---|
| `vllm-decision2` | one prompt per question, per-option scores | strict payloads, entropy confidence |
| `clef` | all questions as one joint prompt | `joint: true` is accepted, images supported |
| `openjev` | letter logits at the last token | letters A-Z/a-z, true-first noul |
| `lev` | code logits at the last token | two option orders for a choice, nine-point noul rating, small/mid/large temperatures |
| `nimble` | code logits at the last token | the prompt lists every question of the request |
| `pplx-decider` | code logits at the last token | canonical legends, images supported |
| `vllm-d3` | code logits at the last token | null state, default question, entropy confidence, canonical legends, images and video |
| `laya` | modern-bert question-type columns at the option markers | `decision.max_head_tokens` truncation |
| `lfm2-d1` | answer-code logits at the last token | per-type yes/no, digit and code labels, images supported |
| `h2o-lightning` | option scores at the markers (`decision.labels`) | a noul question also reads its criteria under the yes/no aliases, canonical legends, images supported |
| `openbmb-minicpm-ocr-jev` | one pooled read over exactly two page images | the fixed question sits in the template; choice and score, at most 2 options |
| `lfm2-d1-omni` | choice columns at the mask markers, the prompt cut per piece | 16384-token cap with 896 image and 15360 audio budgets, the state takes the rest; a media read answers uncalibrated |
| `lfm3-d1-omni` | answer-code logits at the last prompt token after `<think></think>` | letters for a choice, digits for a score, the best of the yes/no group for noul; a score answers its median level and the confidence is the answer's own probability; a spoken question (`input_audio`) is heard after the state's media |
| `lfm3-d1` | as `lfm3-d1-omni`, without the audio path | LFM3 vision-language checkpoints; images and `video_url` state parts; a spoken question (`input_audio` instructions) is refused by name |
| `kev` | dot product of the last prompt token's hidden state with each option's `<|box_end|>` token's (first half against second half, scaled by `sqrt(n_embd / 2)`) | text only: the state, instructions and options flatten from JSON to labelled text; refused at load without a single `<|box_end|>` token |
| `granite-switch` | answer-code logits at the last token, as `lev`, run through the checkpoint's in-graph switched LoRA | codes A-Z then AA-ZZ up to 255 options, true-first noul |

The routes:

**POST /v1/systemone** — TypeSafe-compatible decisions over a decision model (`vllm-decision2` or `clef`). Body: `state` (text, object or array), `questions` (an object keyed by question id; each with `instructions` and `type`: `choice` with a `criteria` object of `{key: description}` entries, `score` with a `criteria` array of 2 to 10 level descriptions, or `noul` with an optional `criteria` object naming only `false`/`true`), optional `files`/`images` (data-URL arrays; refused on a server without `--mmproj`) and `videos` (never supported). Optional `model` selects a served model or adapter. Answer: `{"model", "answers": {"<id>": {"type", ...}}, "usage": {"input_tokens", "output_tokens": 0}}` — choice answers carry the best key, per-key `probabilities` and a `confidence`; score answers add the expected `score` and a `legend` of level descriptions; noul answers carry the probability of `true`. Temperatures and score offsets come from the model's `decision.temperature.*` and `decision.score_bias.*` metadata.

**POST /v1/decisions** — vLLM's Decisions API over the same models. Body: `input` (text, or an array of `user` messages whose content is text or `input_text` parts; media as below), `questions` (an array of at most 64: `{"type": "predicate"}`, `{"type": "choice", "choices": [{"value", "description?"}]}` with 2 to 26 distinct string or boolean values, or `{"type": "score", "levels": [{"label", "description?"}]}` with 2 to 10 levels; each question may name itself with `name`), optional `joint` (accepted by the joint `clef` head, refused by the heads that read each question on their own), `safety_identifier` and `cache_salt`. Answer: `{"model", "answers": [...], "usage": {...}}` where each answer echoes `name` and, per type, `probability` (of true), `choice` with per-value `probabilities` and `confidence`, or `score` with per-level `probabilities` and `confidence`. Values and labels echo the request in the request's own order.

Request media goes through the server's projector (`--mmproj`): `files`/`images` take data URLs
(`data:image/...`, `data:audio/...`, `data:video/...;base64,...`), and a message `state` gives up its `image_url` and
`input_audio` parts; at most 8 items. A server without a projector, or a family without an image slot, answers
llama.cpp's 501; `videos` and non-data URLs answer 400. `joint: true` is accepted by the joint (clef) head and refused
with vLLM's message by the heads that read each question on its own. Decision types outside the table answer 501 naming
the type.

### Cohere Chat v2

`POST /cohere/v2/chat` takes Cohere's v2 chat body and answers Cohere's message shape. The request maps onto the same
chat pipeline as `/v1/chat/completions`: `system`/`developer` messages, user text and `image_url` parts, assistant
`text` and `thinking` blocks (with `tool_plan`) and `tool_calls`, tool messages, `tools` with `strict_tools`,
`tool_choice` (`REQUIRED`/`NONE`), `documents`, `response_format` (`text`, `json_object`, `json_schema`), `thinking`
(`enabled`/`disabled` with `token_budget`), and the sampling fields (`max_tokens`, `stop_sequences`, `temperature`,
`seed`, `k`, `p`, `frequency_penalty`, `presence_penalty`, `logprobs`, `priority`, `chat_template_kwargs`).

```sh
curl -s http://127.0.0.1:8080/cohere/v2/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"command-r","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The response carries `finish_reason` (`COMPLETE`, `STOP_SEQUENCE`, `MAX_TOKENS`, `TOOL_CALL`, `ERROR`), a message of
`thinking` and `text` blocks with any `tool_calls`, and `usage` in both billed units and tokens. With
`"stream": true` the answer is Cohere's SSE sequence — `message-start`, `content-start`/`content-delta`/
`content-end`, `tool-call-start`/`tool-call-delta`/`tool-call-end`, `message-end`, then `data: [DONE]` — with bare
`data:` frames, as Cohere clients demux on each event's `type`.

Cohere's embedding (`POST /v2/embed`) and rerank (`POST /v2/rerank`) endpoints are also served. `safety_mode`,
`citation_options` and message `citations` are refused with 400
naming the field, since this server has no Cohere safety or citation renderer. `kv_transfer_params` runs a
[disaggregated prefill](#disaggregated-prefill---kv-transfer-config-kv_transfer_params) transfer and returns on the `message-end` event. `watermarking` is the per-request
opt-in of [Watermarking generated text](#watermarking-generated-text).

### Tool calls

`tools` (`{"type": "function", "function": {name, description, parameters}}`) render through the template, and `tool_choice` picks how calls come back ([parsers](#reasoning-and-tool-call-parsers)):

| `tool_choice` | Effect |
|---------------|--------|
| `auto` (default) | The configured tool parser reads calls out of the output |
| `none` | The tools still render; nothing is parsed |
| `required` | A grammar forces a JSON array of `{name, parameters}` calls of the declared tools (one when `parallel_tool_calls` is false), as vLLM does |
| `{"type": "function", "function": {"name": ...}}` | A grammar forces the named function's `parameters` schema |

The forcing grammars start after the reasoning when the rendered prompt leaves a thinking block open (a lazy grammar triggered by the reasoning end tag). Calls answer as `message.tool_calls` (`id`, `type: function`, `function.name`, `function.arguments` as JSON text) with `content` null when the output was only calls, and stream as `delta.tool_calls` — the first delta of a call carries its `index`, `id`, `type` and `name`, the following ones its argument text. The finish reason is `tool_calls`, except for a named function, which finishes with `stop` as OpenAI's does. A forcing choice refuses alongside another output constraint, `required` or a named function refuses without `tools`, and a named function must be declared.

The forcing grammars write the calls in the model's own syntax, the one its tool parser reads: Hermes' `<tool_call>` JSON, Qwen3-Coder's `<function=...>` parameters, GLM's `<arg_key>`/`<arg_value>` pairs, DeepSeek's sectioned calls, Kimi's `functions.NAME:IDX`, Mistral's `[TOOL_CALLS]`, the pythonic `[f(k=v)]` list and the rest of the registry. Under `auto`, a lazy grammar can also hold a call to that syntax from its opener on, so a call that starts cannot come out malformed. `-Dneuralfree.tool.strict` (vLLM's `--tool-strict-level`) sets how far the grammars go, and each tool's OpenAI `strict` flag refines it:

| `neuralfree.tool.strict` | Grammar under `tool_choice: auto` | Argument schemas enforced |
|---|---|---|
| `auto` (default) | Only when a tool declares `"strict": true` | Under a forcing `tool_choice`, every tool except those declaring `"strict": false`; under `auto`, only strict tools |
| `function` | Always | As `auto` |
| `parameter` | Always | Every tool, as if `"strict": true` |

Every registered tool format has a forcing grammar and a lazy one, in its own envelope:

- JSON-object formats: `hermes` / `ernie45` (`<tool_call>{"name", "arguments"}</tool_call>`), `longcat` (`<longcat_tool_call>…`), `internlm` (`<|action_start|><|plugin|>…<|action_end|>`), `llama3_json` (`<|python_tag|>{"name", "parameters"}`), `granite-20b-fc` (`<function_call>{"name", "arguments"}`), `gigachat3` (`function call<|role_sep|>\n{json}` or `<|function_call|>{json}`, optionally after `<|message_sep|>`), `inkling` (`<|message_model|><|content_invoke_tool_json|>{"name", "args"}<|end_message|>`), `cohere_command3` / `cohere_command4` (`<|START_ACTION|>[{"tool_call_id", "tool_name", "parameters"}]<|END_ACTION|>`).
- JSON-array formats: `jamba`, `granite`, `phi4_mini_json`, `xlam`, `mistral`, and `apertus` (`<|tools_prefix|>[{"f": {args}}]<|tools_suffix|>`).
- XML function/parameter formats: `qwen3_coder` (`<function=f><parameter=k>v</parameter></function>`), `mimo`, `seed_oss`, `dots` (`<dots_function_call><invoke name="f"><parameter name="k">v</parameter></invoke></dots_function_call>`), `minimax_m2`, `minimax_m3` (namespace-tagged `<invoke name="f">` with one element per argument), `minicpm5` (`<function name="f"><param name="k">v</param></function>`), `glm45` / `glm_4_7` (`<arg_key>`/`<arg_value>`), `k2_horizon` (`<ifm|tool_calls>` sections), `step3` (StepTML invokes), `plamo3` (request blocks), `functiongemma` (`<start_function_call>call:f{k:<escape>v<escape>}<end_function_call>`), `gemma4` (`<|tool_call>call:f{k:<|"|>v<|"|>}<tool_call|>`), and the DSML formats `deepseek_v32`, `deepseek_v4`, `deepseek_v41`.
- Sectioned formats: `deepseek_v3` / `deepseek_v31`, `kimi_k2`, and HY `hy_v3` / `hy_v4` (with the `<tool_sep>` token and the template's or vocabulary's structural-token suffix).
- Pythonic formats: `pythonic`, `llama4_pythonic`, `lfm2`, `olmo3`.
- `functionary_v3_2` (`>>>f\n{json}` recipients; under `auto` a `>>>all` reply may precede the calls).

Some formats follow their templates' finer rules:

- **`minimax_m3`** writes an object or array argument in the template's recursive element form, not as JSON text: object properties become child elements and array elements `<item>` elements, nested to any depth. The parser still reads an element whose value is JSON text.
- **`k2_horizon`** takes the call body from the template's `tool_call_format` variable (`chat_template_kwargs`): `xml` (the default, also `auto`) writes `<ifm|arg_key>`/`<ifm|arg_value>` pairs, `xml_typed` adds the optional `<ifm|arg_type>` tag before each value, and `json` writes `{"name", "arguments"}`. Any other value is refused with 400 naming `chat_template_kwargs.tool_call_format`.
- **`gemma4`** enforces a string parameter's `enum`/`const`, `pattern` and `minLength`/`maxLength` inside the `<|"|>` delimiters. A constraint the delimiters cannot carry safely (an enum value containing `<|"|>`, or a pattern that could write the delimiter's first character) leaves that value free, so the envelope is never broken.
- **`apertus`** accepts an output that ends right after the call array, without the closing `<|tools_suffix|>`, as its parser does.

With strict tool calling, the `deepseek_v32`, `deepseek_v4`, `deepseek_v41` and `mimo` tool parsers constrain calls exactly as xgrammar's builtin `deepseek_v3_2`, `deepseek_v4`, `deepseek_v4_1` and `mimo` structural tags do: the parameters are the tool's schema in the `deepseek_xml` / `deepseek_v4_1_xml` / `qwen_xml` style, a forced DSML call opens with xgrammar's blank line, and under `auto` the call grammar starts at any opener the parser recognizes. The `kimi_k3`, `plamo3`, `hy_v4` and `glm_4_7` parsers match vLLM's own builtin structural-tag builders the same way: Kimi K3 holds the output to its XTML channels — an optional response channel whose text holds none of the reserved markers, the tools channel of `call` tags with one `argument` tag per strict parameter (an enum pinned, a length-bounded string by its bound, a string raw, any other type JSON per the property schema, a union-typed property permissive) and the optional `<|close|>message<|sep|>` closer; PLaMo 3 to its requests block, whose closer may carry the end-of-turn tag; HY v4 to `<arg_key>K</arg_key><arg_value>V</arg_value>` pairs with the required keys in declared order, the optional ones after them in any order, and free values; GLM 4.7 to `arg_key`/`arg_value` pairs whose strict tools pin the schema in the `glm_xml` style and whose loose tools pin the keys, an enum value or a bare JSON scalar, leaving other values free. Under `auto` each grammar starts at any opener its parser recognizes. `glm_4_7` is its own parser name beside `glm45`.

A tool whose arguments are not enforced still has its call envelope constrained: the function and parameter names come from the request, and only the argument values stay free. A `strict` that is not a boolean is refused with 400 naming `tools[i].function.strict`.

Harmony-family models read calls out of their message headers instead of a tool parser. gpt-oss calls arrive through the `to=functions.NAME` recipient with the message body as arguments; Muse Spark / Glimmer calls are the `<atem:function_calls>` block of a message addressed to a tool, each `<atem:invoke>` one call whose `<atem:parameter>` values are typed by the tool's schema. Other recipients (gpt-oss built-ins such as `browser.search` or `python`) are dropped. Under `required` or a named function a grammar in the model's dialect lets it write its reasoning and preamble messages first and then holds the turn to a call of the declared tools, as llama.cpp's gpt-oss and Muse formats do.

### Server tools and MCP

A server started with `--tools` or `--mcp-servers-config` runs tools itself, as llama.cpp's server tools do. **These tools read files, write files and run shell commands on the host. Enable them only where every client is trusted, and set an [API key](#api-keys).**

- **Built-ins** (`--tools NAME,...` or `--tools all`; `--agent` is `--tools all`): `read_file`, `file_glob_search`, `grep_search`, `exec_shell_command` (runs `sh -c`), `write_file`, `edit_file` and `get_info`.
- **MCP servers** (`--mcp-servers-config FILE` or `--mcp-servers-json JSON`): stdio MCP servers from the Cursor-style `mcpServers` document; each entry takes `command`, `args`, `env`, `cwd` and `timeout_ms`. Each server is spawned once at startup to list its tools and again for each call; a server whose spawn failed is skipped for 5 seconds. Its tools are named `<server>_<tool>`.
- `--tools-runtime` is refused: the tools run in the server process, not in a container or over ssh.

**Endpoints.** `GET /tools` lists every tool: the built-ins as `type: "server"` and the MCP tools as `type: "mcp"`, each with `display_name`, `permissions.write`, `uses_cwd` and its OpenAI function `definition`. `POST /tools` with `{"tool": NAME, "params": {...}}` runs one and answers its result object: `plain_text_response` for text, `error` for a failure, and the tool's own fields (the write tools' `result`, `path`, `bytes` and `edits_applied`; `get_info`'s `os` and `cwd`; the search tools' `entries` and `base`). `stream: true` on `exec_shell_command` answers server-sent `{"chunk": ...}` frames and then `{"done": true}`. The `X-Tool-Cwd` header sets the working directory; a `cwd` in the body is dropped, so the model cannot choose where tools run. An unknown tool, or a non-streaming tool asked to stream, answers 404, a malformed body 400, and a server started without tools 403.

**Agent loop.** A chat completion whose `tools` list holds `{"type": "server", "tool": NAME}` or `{"type": "mcp", "tool": "<server>_<tool>"}` entries (each with an optional `definition` replacing the registry's) has the server run the model's calls of those tools, feed each result back as a `tool` message and generate the next round. The loop ends when a round makes no such call, when a round calls a client function (those calls return to the caller as usual), or after `neuralfree.server.tools.max.rounds` rounds (default 10), which leaves the pending calls in the response. The response is the last round's body, with every executed call under `server_tool_calls` (`id`, `name`, `arguments`, `result`) and the usage summed over the rounds. The loop answers one choice, does not stream and takes no `best_of`; each is refused by name. A `cwd`, `runtime` or `resp_type` in a call's arguments is dropped.

**Startup warnings and the Host guard.** A server with server tools or MCP servers logs `security: server tools (experimental), MCP servers (experimental) enabled - do not expose to untrusted environments` at startup, naming exactly what is on (as llama.cpp does), and logs a second warning when no [API key](#api-keys) is configured. A hostile web page that rebinds its own domain to `127.0.0.1` is same-origin to the browser, so while tools or MCP servers are enabled the server checks the `Host` header of the tool requests: `GET /tools`, `POST /tools` and a chat, Responses or Anthropic request whose `tools` list holds a `server` or `mcp` entry must carry a `Host` of `localhost`, a `127.x.x.x` literal, `[::1]`, the configured `--host` or a name from `--allowed-hosts` (comma-separated host names or IP literals, without schemes, ports or paths; `*` disables the check). A browser's `Origin`, when present, must be an `http(s)` origin naming one of those hosts: the server answers CORS for any origin by default, so without this a page on another site could call a local server's tools directly. Non-browser clients send no `Origin` and are unaffected. Any other `Host` or `Origin` answers 403 with the OpenAI error envelope naming `neuralfree.server.allowed.hosts`. Other surfaces are untouched: clients behind proxies keep reaching `/v1/models`, `/v1/completions` and tool-free chat requests, non-JSON content types keep their 415, and forwarded headers (`X-Forwarded-Host`, `Forwarded`) are never trusted. A UI or proxy that reaches the server under another name needs that name in `--allowed-hosts`.

### Web UI

The server jar bundles a chat UI. Open the server's address in a browser (`http://127.0.0.1:8080/` by default), the same address OpenAI clients use. The UI ships inside the jar, with no separate web server or build step.

- **Chat.** Answers stream over the OpenAI chat endpoint. A model's `reasoning_content` shows as a collapsible *Thinking* block, and *Stop* aborts an answer mid-stream.
- **Server state.** The model selector reads `/v1/models`, the server panel reads `/props` (model, context, slots, vision and audio support, sleeping state), and the slots panel reads `/slots`, refreshing while an answer streams.
- **Conversations.** They are kept in the browser's local storage, with search, rename, delete, and JSON export and import.
- **Attachments.** Images (PNG, JPEG, GIF, WebP, SVG; up to 4 MB) go to vision models as OpenAI `image_url` parts. Text and source files (up to 200 KB) are inlined into the prompt. Files can be picked, dropped or pasted.
- **Markdown.** Answers render headings, lists, tables, links and code blocks with a *Copy* button. HTML in model output is always escaped, and links and images render only for `http(s)`, `mailto`, same-site paths and fragments.
- **Settings.** The API key (sent as `Authorization: Bearer`), the theme (dark, light or system), a system prompt applied to every chat, and temperature, top-p, max-tokens and seed defaults. An empty sampling field leaves the parameter out, so the server default applies.

The document answers at `/`, `/index.html` and `/ui`; its assets live under `/ui/<name>`. These routes stay open when [API keys](#api-keys) are configured, as llama.cpp's frontend does, while every endpoint the UI calls still requires the key; enter it once in *Settings*.

Serving options:

- **Cache busting.** The document references each asset with its content hash, `/ui/app.js?v=<12 hex chars>` (SHA-256, computed once from the asset bytes at startup). A request whose version names the served bytes answers `Cache-Control: public, max-age=31536000, immutable`; unversioned, stale, PWA (`manifest.webmanifest`, `sw.js`, `version.json`, `build.json`) and document requests answer `no-cache` with a strong ETag and 304 of a matching `If-None-Match`, and the document carries llama.cpp's `Cross-Origin-Embedder-Policy`/`Cross-Origin-Opener-Policy` isolation headers. A changed asset changes its URL and the document's ETag, so a server upgrade is picked up on the next load, as llama.cpp's content-hashed asset names are.
- **`--ui`, `--webui`, `--no-ui`, `--no-webui`.** The UI serves by default (llama.cpp's `--ui`). `--no-ui` turns it off: the UI routes answer 404 and `/`, `/index.html`, `/ui` and `/ui/<asset>` need the API key like every endpoint.
- **`--path DIR`.** Serves the UI's static files from DIR instead of the bundle (llama.cpp's `--path`). Files answer at their directory-relative paths, an `index.html` serves for its directory, and a path that resolves outside DIR — including through a symlink — is refused with 404. The caching and ETag rules are the bundle's.
- **`--ui-config JSON`, `--ui-config-file FILE`.** Default UI settings as JSON, any JSON value (llama.cpp's `--ui-config`/`--webui-config`); the file form reads it. Bad JSON fails the launch naming the flag. `/props` reports it as `ui_settings`, and the bundled UI applies each key as the default of the matching setting (`apiKey`, `theme`, `defaultSystem`, `model`, `temperature`, `topP`, `maxTokens`, `seed`) beneath the settings its users store in the browser, which keep winning.
- **`--api-prefix PREFIX`.** Serves every route — API, UI, probes and `/metrics` — under the prefix, as llama.cpp does; `--api-prefix api`, `/api` and `/api/` all canonicalize to `/api`, bare paths answer a named 404, and a prefix is one or more simple path segments (`..`, spaces and other unusable segments refuse the launch). The API-key open paths apply after the prefix is stripped, so the UI stays open under a prefix and a `/ui/../v1/models`-style path never opens an API route.
- **CORS.** llama.cpp's flags `--cors-origins ORIGINS`, `--cors-methods METHODS`, `--cors-headers HEADERS`, `--cors-credentials` / `--no-cors-credentials`, with llama.cpp's defaults: origins `*`, methods `GET, POST, DELETE, OPTIONS`, headers `*`, credentials on. vLLM's spellings are aliases: `--allowed-origins`, `--allowed-methods` and `--allowed-headers` take a JSON list of strings (`--allowed-origins '["http://ui.example"]'`) and `--allow-credentials` is `--cors-credentials`; every value passes the same validation whichever spelling names it, and a malformed or empty list refuses the launch naming the flag. With credentials and the wildcard origin the server echoes the request's `Origin` back, so credentialed calls work from any origin; without credentials the wildcard answers the literal `*`. The special origin value `localhost` allows localhost origins only. Every OPTIONS request answers the empty preflight response (`Access-Control-Allow-Origin`, `-Allow-Credentials`, `-Allow-Methods`, `-Allow-Headers`) before authentication and routing, as llama.cpp's pre-routing handler does. Empty origins, control characters and non-token methods or headers refuse the launch naming the flag. The server logs llama.cpp's warning at startup when the CORS policy admits every origin and no API key is configured. A 401 from the API-key guard — including `/metrics`, whose guard runs before Quarkus's CORS filter — carries the configured origin headers, so a browser on another origin can read the refusal. Open CORS lets other sites read the API from a visitor's browser; [server tools and MCP](#server-tools-and-mcp) additionally require a trusted `Origin`, so it never lets them drive the tools.
- **`--reuse-port`.** Allows several sockets to bind the same port (llama.cpp's `--reuse-port`).

### API keys

Keys come from the `NEURALFREE_API_KEY` environment variable and the `neuralfree.server.api.key` property (comma-separated each), and from the file named by `neuralfree.server.api.key.file` (one key per line; blank lines and `#` comments skipped) — llama.cpp's `--api-key` / `--api-key-file` and vLLM's `--api-key`. With any key configured every endpoint requires `Authorization: Bearer <key>` or Anthropic's `x-api-key: <key>`, compared in constant time; a missing or wrong key answers 401 with the OpenAI error envelope. `/health`, `/v1/health`, `/ready`, `/readyz`, `/ping`, the Quarkus `/q/` probes, the [web UI](#web-ui) routes and CORS preflight requests stay open. `/metrics` needs the key, as in llama.cpp. A blank key or an unreadable key file fails the server at startup.

#### Keeping secrets off the command line

Prefer environment variables to secrets typed on the command line. For [remote compute servers](#remote-compute-servers-layer-split-offload), prefer `NEURALFREE_RPC_KEY` over an inline key. `neuralfree.rpc` entries may carry the key as
`host:port@key`, but an inline key is visible in `ps`, `/proc/<pid>/cmdline` and shell history for as long as the
command runs. Write `host:port` and export the key instead:

```sh
export NEURALFREE_RPC_KEY="$(cat /run/secrets/nf-rpc-key)"
neuralfree serve -m model.gguf --device 0 --rpc 10.0.0.5:50052
```

The same rule applies to `--api-key`, to an inline watermark `key` and to a KV transfer `auth_token`: use
`NEURALFREE_API_KEY`, `NEURALFREE_WATERMARK_KEY` (or `NEURALFREE_WATERMARK_KEY_FILE`) and
`NEURALFREE_KV_TRANSFER_TOKEN` (or `NEURALFREE_KV_TRANSFER_TOKEN_FILE`) rather than a value typed on the command line.

The server never reports a secret setting. The `neuralfree_env` field of the server-info responses leaves out every `neuralfree.*`
setting whose name contains `key`, `secret`, `password` or `token`, and the settings whose values embed a secret
under a neutral name: `neuralfree.rpc` (inline keys), `neuralfree.watermark.config` (the JSON `key`),
`neuralfree.server.mcp.json` (MCP server `env` credentials), `neuralfree.server.kv.transfer.config` (the JSON
`auth_token`) and `neuralfree.remote.azure.sas`. `collect-env`
likewise prints only whether `NEURALFREE_API_KEY`, `NEURALFREE_WATERMARK_KEY`, `NEURALFREE_RPC_KEY`,
`NEURALFREE_KV_TRANSFER_TOKEN` and `HF_TOKEN` are set. When `neuralfree serve` relaunches the server, every secret setting (including the values of `--api-key`,
`--rpc host:port@key`, `--watermark-config` and `--mcp-servers-json`) is handed to the server in an owner-only
(`0600`) temporary properties file named by `NEURALFREE_SECRET_PROPERTIES`, not on the child's command line; the
server reads and deletes that file at startup. Error messages that name an RPC entry show only its endpoint
(`host:port@...`).

### OpenAI Responses

`POST /v1/responses` (and llama.cpp's `/responses`) serves OpenAI's Responses API, as llama.cpp and vLLM do, by converting the request into a chat-completions body and running it through the chat path. Every sampler, grammar, tool and reasoning setting therefore behaves as on `/v1/chat/completions`.

- **Request.** `input` is a string or an array of items: messages (`user`, `system`, `developer` with `input_text` and `input_image` parts; `assistant` with `output_text` and `refusal` parts), `function_call`, `function_call_output` and `reasoning` items. Consecutive assistant items merge into one assistant turn. `instructions` become the system message. `tools` of type `function` become chat tools; built-in tool types such as `web_search_preview` have no local equivalent and are skipped. `tool_choice` takes `auto`, `none`, `required` or `{"type": "function", "name": ...}`. `max_output_tokens` is the budget, `reasoning.effort` sets the template's effort, and `text.format` (`text`, `json_object`, `json_schema` with `schema`, `structural_tag` with `format`) constrains the answer the way `response_format` does. `top_logprobs`, or `include: ["message.output_text.logprobs"]`, adds logprobs to the output text, and `extra_params` merges any further chat field (`min_p`, `ignore_eos`, `grammar` and the rest). `input_file`, `input_audio`, `item_reference`, `conversation` and stored `prompt` templates are refused by name.
- **Response.** The Response object carries `reasoning` items (`reasoning_text` content), `message` items (`output_text` with its logprobs) and `function_call` items (`call_id`, `name`, `arguments`). The final output follows the tool parser's reading of the whole text, as the chat route's message does, so a call the finished text turned into content, or arguments completed after the last delta, appear as they end. `status` is `completed`, `incomplete` with `incomplete_details.reason` `max_output_tokens`, `cancelled` or `failed`. `usage` holds `input_tokens` (with `cached_tokens` from the prefix cache), `output_tokens` (with `reasoning_tokens`, the tokens inside the reasoning block) and `total_tokens`.
- **Stream.** `stream: true` sends `response.created` and `response.in_progress`, then for each item `response.output_item.added`, its deltas (`response.reasoning_text.delta`, `response.content_part.added` and `response.output_text.delta`, `response.function_call_arguments.delta`), their `.done` events and `response.output_item.done`, and ends with `response.completed`, `response.incomplete` (also for a cancelled response) or `response.failed`. Each event carries its `sequence_number`, and the last event carries the whole response.
- **Store.** `-Dneuralfree.server.responses.store=<count>` keeps up to that many responses, evicting the least recently used. It is off by default, as in vLLM, and with it off `store: true` keeps nothing. A stored response can be fetched with `GET /v1/responses/{id}`, its input listed with `GET /v1/responses/{id}/input_items` (`limit`, `order`, `after`) and removed with `DELETE`. `previous_response_id` continues its conversation, replaying the earlier input and output items but not the earlier `instructions`.
- **Background.** `background: true` needs the store. It answers at once with a `queued` response and runs the generation detached from the client. `GET /v1/responses/{id}` shows its progress, `GET ...?stream=true&starting_after=<n>` replays its events from that sequence number, and `POST /v1/responses/{id}/cancel` stops it and answers the cancelled response (it waits for the current decode step, at most the request budget). Only a background response can be cancelled.

`POST /v1/responses/input_tokens` and `POST /v1/chat/completions/input_tokens` (and their unversioned forms) return `{"object": "response.input_tokens", "input_tokens": n}` for the rendered prompt.

### SageMaker hosting endpoints

`POST /invocations` accepts any supported inference body and answers exactly as the endpoint its fields name, so one
SageMaker endpoint serves every task the model supports. The type is chosen from the body, in vLLM's priority order:

| Body carries | Answers as |
| --- | --- |
| `messages` | `POST /v1/chat/completions` |
| `prompt` | `POST /v1/completions` |
| `query` + `documents` | `POST /v1/rerank` |
| `queries`/`documents`, `queries`/`items`, `data_1`/`data_2`, or `text_1`/`text_2` | `POST /score` |
| `input` + `task` | `POST /pooling` |
| `input` on a sequence-classifier model | `POST /classify` |
| `input` otherwise | `POST /v1/embeddings` |

`GET /ping` and `POST /ping` answer the readiness probe as `/health` does: an empty 200 while a model is open, 503
with the error envelope before. `model` selects a routed model or a per-request LoRA adapter as on every route.

```sh
curl -s http://127.0.0.1:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

A body that names none of the types is refused with 400 and vLLM's message listing the expected request types.

### Vertex AI prediction

Set the standard Vertex AI prediction variables. When `AIP_MODE=PREDICTION`, the server registers the prediction and
health routes on the HTTP router:

| Variable | Meaning |
| --- | --- |
| `AIP_MODE` | `PREDICTION` enables the protocol; anything else leaves it off |
| `AIP_PREDICT_ROUTE` | prediction path, default `/predict` |
| `AIP_HEALTH_ROUTE` | optional health path, answered as `/health` (unset keeps only `/health`) |
| `AIP_HTTP_PORT` | overrides the server's `--port` in prediction mode |

`POST /predict` takes `{"instances": [...]}`, up to 128 instances. Each instance names the endpoint that answers it
with `@requestFormat` — either a served path (`/v1/chat/completions`) or llama.cpp's camelCase alias
(`chatCompletions`) — and carries that endpoint's body without the format field. The answer is
`{"predictions": [...]}`, one entry per instance in request order; an instance that fails contributes its error
envelope as its prediction, and a streaming instance (`"stream": true`) is refused by name because a prediction
cannot carry an SSE body.

```sh
curl -s http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[
        {"@requestFormat":"chatCompletions",
         "messages":[{"role":"user","content":"Hello"}],"max_tokens":64},
        {"@requestFormat":"/v1/embeddings","input":["a sentence"]}]}'
```

### Launching the server

The server jar takes llama-server's flags, with vLLM's spellings beside them. Each flag sets the `neuralfree.*` or `quarkus.*` property a `-D` launch would set, so both launches configure the same settings, and a flag overrides a property set before it.

```console
java -jar server/build/<server jar> [flags] [model]
```

The model is the first unnamed argument (as `vllm serve <model>` takes it) or `-m`/`--model PATH`.

| Flag | Property | Meaning |
|---|---|---|
| `-m`, `--model PATH` | `neuralfree.server.model` | Model to load |
| `--host HOST` | `quarkus.http.host` | Interface to bind (default 127.0.0.1) |
| `--port PORT` | `quarkus.http.port` | Port to bind, 1-65535 (default 8080) |
| `-sm`, `--split-mode MODE` | `neuralfree.split.mode` | Multi-GPU split: `none` (one GPU), `layer` (each GPU holds a contiguous span of layers and their KV caches, vLLM pipeline parallelism), `row` (feed-forward shards joined by an all-reduce) `tensor` (vLLM tensor parallelism: the attention heads, their KV caches and the feed-forward divided over the ranks); `context` ([context parallelism](#context-parallelism---split-mode-context)) runs over the paged memory the batch engine drives, sharding its page pools ([multi-GPU layer split](#multi-gpu-layer-split), [row](#tensor-parallel-row-split---split-mode-row), [tensor](#tensor-parallelism---split-mode-tensor)); a `--device` list of two or more GPUs splits by layer |
| `-ts`, `--tensor-split SHARES` | `neuralfree.tensor.split` | Positive share of each GPU, comma-separated and one per device, e.g. `3,1` (layer spans, or feed-forward shards under `row`); refused under `tensor` mode, which divides evenly, and without a layer or row split ([multi-GPU layer split](#multi-gpu-layer-split)) |
| `-mg`, `--main-gpu INDEX` | `neuralfree.main.gpu` | Index into `--device` of the GPU holding the embedding and the output head; with `--split-mode none`, the one GPU the model runs on |
| `-tp`, `--tensor-parallel-size N` | `neuralfree.split.mode`, `neuralfree.gpu.devices` | vLLM's tensor parallelism over N GPUs: `--split-mode tensor` over the first N device indices, or over the N devices `--device` lists; `1` is vLLM's default and changes nothing. Refused with `-pp` above 1, with `--rpc`, and beside a contradicting `--split-mode` |
| `-pp`, `--pipeline-parallel-size N` | `neuralfree.split.mode`, `neuralfree.gpu.devices` | vLLM's pipeline parallelism: an N-GPU layer split, over the first N device indices or the N devices `--device` lists; `1` changes nothing. Refused with `-tp` above 1 and with `--rpc` |
| `-dcp`, `--decode-context-parallel-size N` | `neuralfree.split.mode`, `neuralfree.gpu.devices` | vLLM's decode context parallelism: the [context split](#context-parallelism---split-mode-context) over N GPUs, the first N device indices or the N devices `--device` lists; the ranks shard the paged KV pools by whole pages. `1` changes nothing. Refused beside `-tp` or `-pp` above 1 (one split mode runs at a time), with `--rpc`, and beside a different `--split-mode` |
| `-pcp`, `--prefill-context-parallel-size N` | `neuralfree.prefill_context_parallel_size` | vLLM's prefill context parallelism beside the context split: each prefill's rows are divided between the ranks as DualChunkSwap chunk pairs over their page shards ([context parallelism](#context-parallelism---split-mode-context)); needs the same rank count as `-dcp` |
| `--dcp-kv-cache-interleave-size N` | — | vLLM's `dcp_kv_cache_interleave_size`: the context split interleaves whole pages, so N must equal the page size (`neuralfree.kv.page.size`, default 16); a contiguous memory, another split, or another value is refused by name |
| `--enable-expert-parallel`, `-ep` | `neuralfree.expert.parallel` | Tensor split only: each rank holds whole experts instead of a slice of every expert ([tensor parallelism](#tensor-parallelism---split-mode-tensor)); requires `--split-mode tensor` or `-tp N` |
| `--enable-eplb`, `--no-eplb` | `neuralfree.eplb` | Pack the experts of expert parallelism over the ranks by their recorded routed-row load ([EPLB](#expert-parallel-load-balancing-eplb)); needs `--enable-expert-parallel`; as in vLLM it also rebalances at runtime |
| `--eplb-profile PATH\|off` | `neuralfree.eplb.profile` | The expert-load profile read at load, a `moe-cache-ranking.bin` |
| `--eplb-save PATH\|off` | `neuralfree.eplb.save` | Write the routed rows counted while the model runs to this file |
| `--eplb-policy POLICY` | `neuralfree.eplb.policy` | Expert placement policy: `default` (vLLM's weighted packing) or `hierarchical` (groups to nodes, then ranks) |
| `--eplb-config JSON` | `neuralfree.eplb.*` | vLLM's `EPLBConfig` object: `policy`, `num_redundant_experts`, `window_size` and `step_interval` map onto their `neuralfree.eplb.*` settings, and asynchronous migration refuses by name ([EPLB](#expert-parallel-load-balancing-eplb)) |
| `--batch-invariant` | `neuralfree.batch.invariant` | Logits independent of the batch a request runs in ([batch-invariant mode](#batch-invariant-mode)); also `NEURALFREE_BATCH_INVARIANT=1` |
| `--kv-transfer-config JSON` | `neuralfree.server.kv.transfer.config` | vLLM's KV transfer configuration: this instance's side of a [disaggregated prefill](#disaggregated-prefill---kv-transfer-config-kv_transfer_params) |
| `--cache-reuse N` | `neuralfree.cache-reuse` | Default `n_cache_reuse`: re-append cached runs of at least N rows past the first prompt mismatch ([KV chunk reuse](#kv-chunk-reuse---cache-reuse-n_cache_reuse)); 0 (default) is off |
| `-gan`, `--grp-attn-n N` / `-gaw`, `--grp-attn-w N` | `neuralfree.grp-attn-n`, `neuralfree.grp-attn-w` | Default `grp_attn_n` / `grp_attn_w` ([self-extend](#self-extend-group-attention-and-position-division)); 1 and 512 by default |
| `-a`, `--alias NAMES`; `--served-model-name NAMES` | `neuralfree.server.model.name` | Comma-separated names to serve ([model names](#http-server)) |
| `--api-key KEY` | `neuralfree.server.api.key` | Require this bearer key; repeatable ([API keys](#api-keys)) |
| `--api-key-file FILE` | `neuralfree.server.api.key.file` | One key per line, `#` comments |
| `-c`, `--ctx-size N`; `--max-model-len N` | `neuralfree.ctx` | Context length; 0 lets the fit pick |
| `-np`, `--parallel N` | `neuralfree.parallel` | Sequence slots, 1-256 |
| `--logprobs-mode MODE` | `neuralfree.server.logprobs.mode` | What logprobs score: `raw_logprobs` (default), `raw_logits`, `processed_logprobs`, `processed_logits` (vLLM's flag) |
| `--scheduling-policy fcfs\|priority` | `neuralfree.scheduling.policy` | Batch engine order: `fcfs` by arrival (request `priority` 0 only), `priority` by the request's `priority`, lowest first ([scheduling](#paged-kv-memory)) |
| `--chat-template VALUE` | `neuralfree.chat.template` | Jinja source, built-in name or template file path replacing the model's ([Choosing a chat template](#choosing-a-chat-template)) |
| `--chat-template-kwargs JSON`, `--default-chat-template-kwargs JSON` | `neuralfree.chat.template.kwargs` | Default template variables for every chat-rendering route; a request's own variables win |
| `--chat-template-content-format auto\|string\|openai` | `neuralfree.chat.template.content.format` | The content form the template receives ([Chat template capabilities](#chat-template-capabilities)) |
| `--reasoning-preserve`, `--no-reasoning-preserve` | `neuralfree.chat.reasoning.preserve` | The default of `preserve_reasoning` for templates that support it; `true` without either flag, as in llama.cpp |
| `--chat-template-file FILE` | `neuralfree.chat.template.file` | Template file, UTF-8; excludes `--chat-template` |
| `--lora PATH`; `--lora-scaled PATH SCALE` | `neuralfree.lora` | LoRA adapter, optionally scaled; repeatable |
| `--ssl-key-file FILE`, `--ssl-cert-file FILE` (`--ssl-keyfile`, `--ssl-certfile`) | `quarkus.http.ssl.certificate.key-files`, `.files` | TLS private key and certificate chain, PEM; both are required |
| `--slot-save-path DIR` | `neuralfree.server.slot.save.path` | Turns on slot save, restore and erase ([Slots](#slots)) |
| `--dev-mode` | `neuralfree.server.dev.mode` | Serve vLLM's development and RL control routes ([Development endpoints](#development-endpoints)); off by default and the routes answer 404 while off. `VLLM_SERVER_DEV_MODE=1` enables it too; use only on trusted deployments |
| `--grpc` | `neuralfree.server.grpc` | Serve the gRPC API on the HTTP port ([gRPC API](#grpc-api)); off by default, and with API keys every call needs `authorization: Bearer <key>` |
| `--enable-elastic-ep` | `neuralfree.server.elastic.ep` | Serve `/scale_elastic_ep` and `/is_scaling_elastic_ep` ([Distributed serving control](#distributed-serving-control)); off answers 404 |
| `--enable-fault-tolerance` | `neuralfree.server.fault.tolerance` | Serve `/v1/fault_tolerance/status` and `/v1/fault_tolerance/apply` ([Distributed serving control](#distributed-serving-control)); off answers 404 |
| `--enable-sleep-mode` | `neuralfree.server.sleep.mode` | vLLM's sleep-mode switch: serves the checkpoint window and lets `/sleep`, `/wake_up` and `/release_kv_cache_memory` release memory ([Sleep, pause and checkpoint](#sleep-pause-and-checkpoint)); off by default |
| `--profiler-config JSON` | `neuralfree.server.profiler.config` | vLLM's ProfilerConfig object, e.g. `'{"profiler": "torch", "torch_profiler_dir": "/tmp/profile"}'`; `torch` (the JFR window's counterpart) is accepted, vLLM's `cuda` and `proton` and any other field refuse by name; the profile window is served only with a profiler ([Operational endpoints](#operational-endpoints)) |
| `--watermark-config JSON` | `neuralfree.watermark.config` | vLLM's watermark configuration: the secret `key`, `algorithm`, `alpha`, `context_width` and the deduplication fields ([Watermarking generated text](#watermarking-generated-text)); off by default |
| `--tools NAME,...\|all`, `-ag, --agent` | `neuralfree.server.tools` | Built-in server tools to enable ([Server tools and MCP](#server-tools-and-mcp)) |
| `--mcp-servers-config FILE`, `--mcp-servers-json JSON` | `neuralfree.server.mcp.config`, `neuralfree.server.mcp.json` | Stdio MCP servers whose tools the server runs |
| `--allowed-hosts HOSTS` | `neuralfree.server.allowed.hosts` | Trusted `Host` names for server tools and MCP besides loopback and `--host`, comma-separated; `*` disables the check ([Server tools and MCP](#server-tools-and-mcp)) |
| `--ui`, `--webui`; `--no-ui`, `--no-webui` | `neuralfree.server.webui` | The bundled chat UI at `/` ([Web UI](#web-ui)); on by default, off closes its routes and keys them |
| `--path DIR` | `neuralfree.server.ui.path` | Serve the UI's static files from DIR instead of the bundle ([Web UI](#web-ui)) |
| `--ui-config JSON`, `--webui-config JSON`; `--ui-config-file FILE`, `--webui-config-file FILE` | `neuralfree.server.ui.config` | The UI's default settings as JSON ([Web UI](#web-ui)); bad JSON fails the launch |
| `--api-prefix PREFIX` | `neuralfree.server.api.prefix` | Serve every route under the prefix ([Web UI](#web-ui)) |
| `--cors-origins ORIGINS`; `--cors-methods METHODS`; `--cors-headers HEADERS`; `--cors-credentials`, `--no-cors-credentials`; vLLM's `--allowed-origins` / `--allowed-methods` / `--allowed-headers` JSON lists and `--allow-credentials` | `quarkus.http.cors.*` | Browser CORS with llama.cpp's defaults, vLLM's spellings as aliases ([Web UI](#web-ui)) |
| `--reuse-port` | `quarkus.http.so-reuse-port` | Allow several sockets to bind the same port |
| `--models-dir DIR` | `neuralfree.server.models.dir` | Turns on the multi-model router over the GGUF files in `DIR`, one model per file ([Multi-model router](#multi-model-router)); the directory must exist |
| `--models-preset FILE` | `neuralfree.server.models.preset` | Turns on the router over the preset INI `FILE` (format below); the file must exist |
| `--models-max N` | `neuralfree.server.models.max` | How many models may be open at once; `0` means no limit (default `4`) |
| `--models-memory-margin N` | `neuralfree.server.models.memory.margin` | MiB of device memory kept free: idle models unload least-recently-used first until the next one fits; `0` disables it (default `0`) |
| `--models-autoload`, `--no-models-autoload` | `neuralfree.server.models.autoload` | Whether a request naming an unloaded model loads it (default on) |
| `-cram, --cache-ram N` | `neuralfree.cache.ram` | MiB of host RAM for prefix-cache blocks the KV pools evict; `0` off (default), `-1` no limit |
| `-nkvo`, `--no-kv-offload` | `neuralfree.kv.offload` | Keep every layer's KV cache in host RAM: its projections and FFN run on the device, its attention on the host over that cache ([hybrid execution](#hybrid-cpu-and-gpu-execution)) |
| `-kvo`, `--kv-offload` | `neuralfree.kv.offload` | Keep the KV caches on the device (the default) |
| `-cdisk, --cache-disk N` | `neuralfree.cache.disk.mib` | MiB of disk for blocks the host-RAM tier drops; `0` off (default), `-1` no limit; needs `--cache-ram` and `--cache-disk-dir` ([offload tiers](#prefix-cache)) |
| `--cache-disk-dir DIR` | `neuralfree.cache.disk.dir` | The directory of the disk tier's block files, created when missing |
| `--trust-request-chat-template` | `neuralfree.server.trust.request.chat.template` | Lets requests supply their own `chat_template` (vLLM's flag; default off, such a request is refused with 400) |
| `--kv-events-config JSON` | `neuralfree.kv.events` | vLLM's `KVEventsConfig`: `{"enable_kv_cache_events": true}` publishes the KV cache events on vLLM's ZMQ transport (`tcp://*:5557`). `"publisher"` takes `zmq` (the default when enabled), `null`, `log` or a `KvEventPublisher` class name. `endpoint`, `replay_endpoint`, `buffer_steps`, `hwm`, `max_queue_size` and `topic` configure the ZMQ transport, and are refused by name next to any other publisher ([Prefix cache](#prefix-cache)); default off |
| `--prefix-caching-hash-algo NAME` | `neuralfree.prefix.hash.algo` | The prefix cache's block-hash algorithm (vLLM's `--prefix-caching-hash-algo`): `sha256` (default), `sha256_cbor`, `xxhash` or `xxhash_cbor`. The published `BlockStored` hashes equal vLLM's for each ([Prefix cache](#prefix-cache)); any other name is refused |
| `--pooling none\|mean\|cls\|last\|rank`; `--override-pooler-config JSON` (alias `--pooler-config`) | `neuralfree.pooling` | The pooling of the embedding routes over the GGUF's ([Pooling type override](#pooling-type-override)) |
| `-dp`, `--data-parallel-size N`; `--dp-size N` | `neuralfree.server.replicas` (with a rank or local size: `neuralfree.server.dp.size`) | Data-parallel size: `N` replicas behind this server ([Data-parallel serving](#data-parallel-serving)) |
| `-dpn`, `--data-parallel-rank R`; `-dpe`, `--data-parallel-external-lb` | `neuralfree.server.dp.rank` | Serve rank `R` alone behind an external load balancer (vLLM's external-LB mode); `R` must be below the size |
| `-dpl`, `--data-parallel-size-local N`; `-dpr`, `--data-parallel-start-rank S`; `-dph`, `--data-parallel-hybrid-lb` | `neuralfree.server.replicas`, `neuralfree.server.dp.rank` | Serve `N` consecutive ranks from `S` (default 0) here (vLLM's hybrid mode) |
| `--data-parallel-devices LIST`; `--dp-devices LIST` | `neuralfree.server.devices` | Comma-separated device names, one replica each |
| `--load-balance-method NAME` | `neuralfree.server.load.balance` | `shortest_queue` (default), `round_robin` or `minimum_tokens` (SGLang's; also `auto`, `total_requests`, `total_tokens`) |
| `--jinja`, `--no-jinja` | `neuralfree.chat.jinja` | Render prompts through the model's Jinja template (default), or llama.cpp's non-Jinja route ([Choosing a chat template](#choosing-a-chat-template)); the later flag wins |
| `--metrics` | — | Accepted: the `/metrics` endpoint is always served |
| `--otlp-traces-endpoint URL` | `neuralfree.server.otlp.traces.endpoint` | Export one OpenTelemetry `llm_request` span per generation request to this OTLP endpoint ([Request tracing](#request-tracing)); off when unset |
| `-h`, `--help` | — | The flag text |

With TLS the server serves HTTPS on `--port` and plain HTTP is off, as llama-server does. Engine settings beyond these flags travel through their `neuralfree.*` properties, as in any launch. An unknown flag, a missing value and a value that cannot work are refused by name before the server starts; `--lora-modules` is refused because adapters are fixed at launch. The template override replaces the model's `tokenizer.chat_template` everywhere a tokenizer is built: prompts, tool routing and `/props`. The data-parallel flags are resolved together, as vLLM resolves them, and contradicting combinations are refused by name. For example, a rank at or past the size, `--data-parallel-external-lb` without a rank, a rank together with `--data-parallel-start-rank` or a local size other than 1, `--data-parallel-hybrid-lb` without a local size, or a device list whose length differs from the replicas placed here. vLLM's cross-process coordination flags (`--data-parallel-address`, `--data-parallel-rpc-port`, `--data-parallel-backend`, `--data-parallel-multi-port-external-lb`) are refused, because the replicas decode independently and need no coordination.

### Image and audio input (`--mmproj`)

Start the server with a multimodal projector beside the text model to accept image and audio input:

    neuralfree-server -m model.gguf --mmproj mmproj.gguf

| Flag | Meaning |
|---|---|
| `-mm, --mmproj FILE` | The projector GGUF (vision and/or audio encoder). Its output width must equal the model's embedding width. `--no-mmproj` turns media input off. |
| `--media-path DIR` / `--allowed-local-media-path DIR` | The directory `file://` media URLs may read from. `file:///abs/path` must lie inside it; `file://name.png` is relative to it. Without the flag, file URLs are refused. |
| `--allowed-media-domains HOSTS` | Comma-separated hosts that http(s) media URLs may name (exact match, re-checked on every redirect). Default: any host. Set it on servers reachable by untrusted clients. |
| `--limit-mm-per-prompt JSON` | The most items per prompt, e.g. `{"image": 2, "audio": 1}` (`{"image": {"count": 2}}` also works). Unset modalities allow 999; 0 turns a modality off. |
| `--image-min-tokens N` / `--image-max-tokens N` | Token budget per image for dynamic-resolution projectors. |
| `--video-fps N` | Sampling rate of projectors without a recommended rate (default 4; 0 keeps the source rate). Property: `neuralfree.server.video.fps`. |
| `--video-timestamp-interval MS` | Timestamp spacing of the default video template (default 5000; 0 stamps none). Property: `neuralfree.server.video.timestamp.interval`. |
| `--video-ffmpeg-dir DIR` | Directory holding the `ffmpeg` and `ffprobe` binaries (default: the `PATH`). Property: `neuralfree.server.video.ffmpeg.dir`. |
| `--video-pruning-rate Q` / `--video-pruning-method evs\|vidcom2` | vLLM's video token pruning: the fraction of each video's rows to drop (default 0) and the algorithm (default `evs`). Properties: `neuralfree.server.video.pruning.rate`, `neuralfree.server.video.pruning.method`. |

The matching system properties are `neuralfree.server.mmproj`, `neuralfree.server.media.{path,domains,limits,timeout,max.bytes,redirects}` (timeout in seconds, default 10; payload limit default 25 MiB; redirects default true) and `neuralfree.server.image.{min,max}.tokens`.

Media prompts run on the CPU and GPU backends alike. An embedding row (a media row or a `prompt_embeds` row) carries no token id, so models whose layers read token ids handle it as llama.cpp does:
- **Gemma 3n, Gemma 4:** the per-layer input is the padding token's (id 0) row of `per_layer_token_embd`; the projection and scales then run as for tokens.
- **qwen4exp:** the n-gram per-layer embeddings hash `ple.image_token_id`, or EOS when that is 0. Media rows sharing one M-RoPE position take each other as predecessors in prompt order.
- **deepseek4:** hash-routed layers route embedding rows through the router, with `exp_probs_b_vl` or no selection bias. Other layers use `exp_probs_b_vl` for them when present.
- **Refused by name:** models with switched-LoRA routing or value embeddings refuse media and `prompt_embeds` rows.

**Chat completions.** Content parts `image_url` (`{"url": ...}` or a bare URL string), `input_audio` (`{"data": base64, "format": "wav"}`) and `audio_url` take data URLs (base64 only), http(s) URLs, `file://` URLs inside the media path, or raw base64:

    {"messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this picture?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}]}]}

Video parts (`video_url`, `input_video`) are described under [Video input](#video-input).

**Responses API.** Use `{"type": "input_image", "image_url": "..."}` and `{"type": "input_audio", "input_audio": {"data": ..., "format": ...}}`.

**Anthropic Messages API.** Use `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ...}}` or `{"type": "image", "source": {"type": "url", "url": ...}}`.

**Raw completions.** On `/completion`, `/completions` and `/v1/completions`, a prompt may be `{"prompt_string": "Describe <__media__> briefly", "multimodal_data": ["<base64>"]}`: one marker per entry, in order. Images and audio are told apart by their bytes.

**Token counts.** These include the media rows, matching the generation's usage: `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens`, and `/tokenize` with `messages`. In `/tokenize`, each media row is listed as token id `-1` (piece `null`). `/apply-template` renders media parts as the `media_marker` that `/props` reports. `/props` also reports `modalities: {vision, audio, video}`.

### Video input

NeuralFree accepts video content parts when the projector has a vision encoder. Frames are decoded by the
external `ffmpeg` and `ffprobe` binaries, which must be on the `PATH` or in the directory named by
`--video-ffmpeg-dir DIR` (`neuralfree.server.video.ffmpeg.dir`). A missing or unreadable binary, an empty or
non-video payload, an oversized video and a decoder timeout are refused with the request field that carried the
video.

Chat requests take the same two shapes vLLM and llama.cpp use:

```json
{"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
{"type": "input_video", "input_video": {"data": "<base64 MP4>", "format": "mp4"}}
```

A `video_url` accepts an http(s) or data URL (subject to `--allowed-media-domains` / `--allowed-local-media-path`);
`input_video` accepts inline base64 `data`, or `url`. A part may also carry a `uuid`; the first request may then
send the data and later requests may repeat the `uuid` alone, and the decoded frames and encoded rows come from
the media cache. Raw `/completion` prompts accept video base64 entries in `multimodal_data`: the container is
sniffed, so MP4/MOV, Matroska/WebM, AVI, FLV and MPEG-TS are recognized as video before the audio and image sniffs
run. Per-prompt accounting uses `--limit-mm-per-prompt` (`{"video": N}`; 0 turns the modality off), and `/props`
reports `"modalities": {"video": true}` for a projector with a vision encoder.

Sampling follows the model's reference processor. The LFM3-VL and Qwen-VL families recommend 2 fps with a
`[4, 768]` frame clamp (LFM3-Omni clamps at 384 frames); the other projectors sample at
`--video-fps N` fps (default 4), and 0 keeps the source rate. The frame count is the source's frames replayed at
the target rate, clamped to the family's range and the source's frames, and the sampled indices are
`round(linspace(0, n-1, count))`. A request can override this per prompt through the video loader options:

**Qwen2-VL sampling.** Qwen2-VL and Qwen2.5-VL follow vLLM's `Qwen2VLVideoBackend` frame for frame. The frame count is `frames / source_fps * fps` (the request's `fps` capped at 30, or the family's 2 fps), clamped to the family's `[4, 768]` range and to the source's frames, then floored to a multiple of the temporal patch size (2); a one-frame video cannot fill one patch and is refused by name. The sampled source indices are `torch.arange(0, frames, frames / count)`, evaluated in float32 and clamped to the last frame, so the tail can repeat the last frame. `num_frames` is ignored, as for the other fps-driven Qwen loaders. Every other family keeps `round(linspace(0, n - 1, count))`.

**Sources without a frame rate.** A container whose frame rate ffprobe reports as 0 or `N/A` (a variable or unknown rate) is resolved per family. Qwen3-VL assumes a 24 fps source, as its vLLM loader does: the frame count becomes `int(frames / 24 * fps)` and the `<X.X seconds>` group stamps are the sampled frames' times at 24 fps. Every other family refuses the video with a client error naming the request field: Qwen2-VL and Qwen2.5-VL need a known rate (vLLM's loader raises), and the LFM3-VL and generic families follow llama.cpp's helper, which rejects a probe without one. A source without a rate is accepted only when ffprobe still reports its frame count (`nb_frames`); a frame count cannot be derived from the duration without a rate.

```json
--media-io-kwargs '{"video": {"num_frames": 32, "fps": 2, "max_duration": 120}}'
```

`num_frames` takes an explicit count, `fps` the sampling rate, and `max_duration` the longest accepted video in
seconds (300 by default; 0 disables the check). Configured video options are part of the media cache key.

The prompt layout is per family. The default template (llama.cpp's) opens with `Video:` and stamps
`[XmY.YYs]` text after the frames that cross `--video-timestamp-interval MS` (default 5000; 0 stamps none). The
group template (LFM3-VL and Qwen3-VL) stamps `<X.X seconds>` before every temporal group, the midpoint time of the
group's source frames. The Qwen-VL families with a temporal patch kernel (Qwen2-VL, Qwen2.5-VL, Qwen3-VL,
Ling3-VL, LFM3-VL, Nemotron-Omni) merge adjacent frames two at a time into one encoded chunk, as llama.cpp's
`build_inp_temporal_merge` and vLLM's Conv3d patch embedding do.

Video token pruning is vLLM's. `--video-pruning-rate Q` (a fraction in `[0, 1)`, `0` by default) drops `Q` of
every video's embedding rows, keeping one frame's rows at least under EVS and one row per frame under VidCom2;
`--video-pruning-method evs|vidcom2` picks the algorithm (EVS by default). The kept tokens keep their M-RoPE grid
positions, and fully pruned frames leave their timestamp text in place. Pruning runs over single-chunk frames, so
a projector whose frame layout slices one frame over several tiles refuses a non-zero rate by name.

**Frame size and count.** Frames are decoded at the size the projector's image preprocessor recommends (for LFM3-VL and LFM3-Omni, the reference video processor's size within its 4096 to 25165824 pixel bounds), through the reference's `format=rgb24,scale=…:flags=bicubic+accurate_rnd+full_chroma_int` filter, so they match llama.cpp's mtmd helper byte for byte. The Qwen-VL and LFM3-VL loaders sample at the request's `fps` (or the family's recommended rate) and ignore `num_frames`, as vLLM's Qwen loaders do; other families take the smaller of `num_frames` and `floor(duration * fps)`, as vLLM's generic video backend does.

**Soundtracks (LFM3-Omni).** With an LFM3-Omni projector (an LFM3-VL vision encoder beside an LFM3-Audio tower), the video's audio track is decoded once at the audio tower's sample rate, split over each temporal group's span, encoded by the audio tower and inserted after that group's frames, as llama.cpp's `MTMD_VIDEO_TEMPLATE_LFM3_OMNI` does. A video without an audio track carries no soundtrack.

**Decoder safety.** Every `ffmpeg` and `ffprobe` run reads client-supplied media through one input guard: a staged copy is read through the `file` protocol, a piped payload through the `pipe` protocol, and the demuxers are whitelisted per modality (`-protocol_whitelist`, `-format_whitelist`). The video whitelist holds the containers above; the audio whitelist holds WAV, AIFF, AU, MP3, FLAC, OGG, the MP4/M4A family, Matroska and WebM; the image whitelist holds only the WebP and image-pipe demuxers. An HLS playlist or ffconcat script posing as any kind of media therefore cannot make ffmpeg open other files or URLs, and an ffmpeg decode over a staged file also passes `-nostdin`. A video decode whose kept frames (at the decoded size) would take more than a quarter of the JVM heap is refused, naming the field, and so is a soundtrack whose buffered span (from the container's start time to the end of the video) would; the soundtrack decode stops at that span.

### Batch-invariant mode

`--batch-invariant` makes a request's output independent of what else is batched beside it: the logits of a sequence are bit-identical whether it runs alone or in a batch of 2, 4, 8, ... with other requests. It is vLLM's `VLLM_BATCH_INVARIANT` behaviour.

Enable it with the server flag, the system property, or the environment variable:

```
neuralfree serve --batch-invariant -m model.gguf
neuralfree-server -m model.gguf --batch-invariant
java -Dneuralfree.batch.invariant=true -jar server.jar -m model.gguf
NEURALFREE_BATCH_INVARIANT=1 neuralfree-server -m model.gguf
```

The property accepts `true`/`false` (and, like vLLM's environment variable, `1`/`0`); an unknown value is refused naming the property. The mode is process-wide and is read once.

While it is on, the engine pins every reduction whose order would otherwise depend on the number of rows in a step:

- projections use one lane per weight type at every row count: a type with a fast GEMM lane (the int8 MMQ GEMM, its cooperative-matrix form, or the tensor-core MMA GEMM) always uses that lane and never its split-K twins; a type without one always uses the float matvec family (never the exact tiled GEMM, the two-row matvec or the Q8_1 single-token matvec lane);
- flash attention runs one decode launch per query row on both dense and paged caches, so each row's key splits come from its own live key count rather than from the batch's longest sequence;
- MLA (DeepSeek2) keeps its per-head absorbed absorption and expansion products at every row count and runs one decode launch per query row on the unpaged and the paged cache;
- routed experts always use the sorted per-pair product (never the one-token matvec path or the grouped expert GEMM); when the expert cache is configured, every routed pair is computed on the host, because a cached device product and a host miss differ in their accumulation order (the cache then provides no speedup);
- the CPU attention key-split count is derived from one query row's work units;
- sampling is already per request: each generation owns a sampler and its own seeded generator, so the chain never reads the batch and no separate mode is required.

Cost, measured on an NVIDIA RTX 5080 Laptop GPU over OpenCL (`nftool bench`, three repetitions; the prompt and generation token rates are tokens per second):

| model | test | mode off | mode on | change |
|---|---|---:|---:|---:|
| Qwen3-Embedding-0.6B Q8_0 | pp128 | 2330.8 | 1017.7 | 2.3x slower |
| Qwen3-Embedding-0.6B Q8_0 | tg32 | 183.5 | 59.0 | 3.1x slower |
| jina-v5-small-retrieval F16 | pp64 | 1933.6 | 482.3 | 4.0x slower |
| jina-v5-small-retrieval F16 | tg16 | 211.4 | 207.8 | unchanged |

On the CPU engine the mode changes only the attention split rule and the measured rates stay within run-to-run noise (Qwen3-Embedding-0.6B Q8_0: pp64 63.6 vs 69.9, tg16 19.1 vs 17.6 tokens per second). A Mixture-of-Experts model with the expert cache configured pays an additional cost: under the mode the cache never serves a product, so every routed pair is computed on the host. Use the mode when exact cross-batch reproducibility matters more than throughput.

### Media cache keys (`uuid`) and the encoded-media cache

The server caches media at two layers, both keyed by one media identity (the part's `uuid`, else the SHA-256 of its
bytes):
- **Encoded media:** the embedding rows the vision and audio encoders produced. A hit skips decoding, preprocessing
  and the encoder.
- **Processed media:** the decoded and preprocessed image tiles and the decoded audio. A hit skips the decode when the
  encoding has been evicted or reset.

A token count followed by the generation of the same request encodes its media once. Beside these, the paged prefix
cache reuses the KV of earlier media ([Prefix cache](#prefix-cache)), and with data-parallel replicas a media prompt
is routed to the replica that holds its prefix.

**Item keys.** Every media item has one key, vLLM's `mm_hash`, and the media caches and the prefix cache's block keys
all use it. The key is the part's `uuid` when it has one, otherwise the SHA-256 of the item's bytes. The
`--media-io-kwargs` options are hash factors: with options set, the key is a SHA-256 over the uuid (or the digest)
and the options, as in vLLM. KV cache events publish the key as the media extra key's hash, so a uuid appears there
verbatim.

A media content part may carry vLLM's `uuid`: your own cache key for the item. While the cache holds that uuid, the
part's URL or data is not loaded at all, and a later request may send the uuid alone:

    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}, "uuid": "cat-1"}
    {"type": "image_url", "image_url": null, "uuid": "cat-1"}

- `uuid` works on chat `image_url`, `input_audio` (empty `data` allowed beside a uuid) and `audio_url` parts, and on
  the Responses API's `input_image` and `input_audio` parts.
- The uuid names the item: as in vLLM, a uuid the cache holds serves its cached encoding even when other bytes come
  with it. Use one uuid per distinct item.
- A uuid sent alone that the cache does not hold (never sent, evicted, or dropped by a reset or sleep) is refused
  with HTTP 400 naming the part; send the item with its data again.
- `uuid` must be a non-blank string (a number is taken as its text). It is refused on `image_embeds` /
  `audio_embeds` parts, which skip the encoder; such a part is keyed as one item by a SHA-256 over all its rows.

**Encoding skipped on a prefix hit.** The server remembers the token layout of every item it has encoded. When an
item's encoding is no longer cached but its layout is known (after eviction, a reset or a sleep), the prompt is built
without encoding it, and the item is encoded only when the engine prefills its rows. If the prefix cache already
holds the item's rows (same key at the same place), the item is never encoded. A deferred encode that fails (for
example, a uuid reused for an item of another size) fails only its own request.

| Flag | Meaning |
|---|---|
| `--mm-processor-cache-gb GIB` | The budget of each layer in GiB (fractions allowed), as vLLM's flag sizes each processor-cache copy. Default 4, as in vLLM, but at most an eighth of the JVM heap per layer, since both live on the heap. Least recently used items are evicted first; an item larger than the budget is not cached; `0` turns the cache off. Property: `neuralfree.server.media.cache.gb`. |
| `--media-io-kwargs JSON` | Media loader options, vLLM's JSON form keyed by modality. Image: `image_mode` (`"RGB"`, default, or `"L"` for 8-bit grayscale) and `rgba_background_color` (`[r, g, b]`, each 0–255, default white: the colour transparent pixels are composited over in RGB mode). Audio takes no options. Video takes `num_frames`, `fps` and `max_duration` (seconds, default 300; 0 disables the check); see [Video input](#video-input). Any other key or value is refused at launch, by name. The options are hash factors of every image key. Property: `neuralfree.server.media.io.kwargs`. |

Example: `--media-io-kwargs '{"image": {"image_mode": "RGB", "rgba_background_color": [0, 0, 0]}}'`. A request's own
`media_io_kwargs` is refused with HTTP 400 naming the field, as in vLLM without `--trust-request-mm-kwargs` (not
offered here); an empty object or `null` is accepted.

`POST /reset_mm_cache` drops both layers, and `POST /reset_encoder_cache` drops the encodings alone (the projector's
encoded layer and every replica's batch encoder cache; the processed media stay). Both are vLLM's development routes: they need `--dev-mode` (`neuralfree.server.dev.mode`, or vLLM's `VLLM_SERVER_DEV_MODE=1`) and answer 404 without it. They answer an
empty 200, also on a server without a projector, and refuse any body key or query parameter by name. Both layers are also dropped by a sleep
of level 1 or more, by `POST /release_kv_cache_memory` and when a checkpoint preparation starts, as in vLLM. `/metrics` counts the
lookups and hits as `neuralfree_mm_cache_queries_total` and `neuralfree_mm_cache_hits_total` (vLLM's `mm_cache_queries` /
`mm_cache_hits`) and the bytes held as `neuralfree_mm_cache_usage_bytes`. The same lookups and hits are also published under vLLM's names,
`vllm:mm_cache_queries_total` and `vllm:mm_cache_hits_total`, labelled `model_name` (the served model name) and
`engine="0"`; the hit rate is `rate(hits) / rate(queries)`. The counters run from the projector's opening, and a cache
reset leaves them unchanged.

**Projector memory during sleep.** The projector follows the model weights. A sleep of level 2 (`POST /sleep?level=2`,
or the idle sleep of `--sleep-idle-seconds`) releases the projector's weights and encoders once every replica has
released its model. `POST /wake_up` reopens the projector from its file after the models. A reopen failure fails the
wake with its message. The request that wakes an idle-sleeping server reopens it as well. While asleep, `/props`
still reports `modalities` and `media_marker`.

### Multi-model router

Serve several models from one server and route every request by its `model` name, as llama.cpp's router server
does. Turn the mode on by naming a model source; the single-model properties (`-Dneuralfree.server.model`,
`-Dneuralfree.server.model.name`) refuse beside it.

| Property | Meaning | Default |
| --- | --- | --- |
| `neuralfree.server.models.dir` | Directory of GGUF files to serve; one model per file, named after the file (a split set `<name>-00001-of-000NN.gguf` serves under `name` from its first shard) | off |
| `neuralfree.server.models.preset` | INI file of model presets, one `[section]` per model (below) | off |
| `neuralfree.server.models.max` | How many models may be open at once; 0 for no limit | 4 |
| `neuralfree.server.models.memory.margin` | MiB of device memory to keep free: an idle model is unloaded (least recently used first) until the next load fits what the open engines hold against the devices' total; 0 disables it; without GPU devices it has no effect | 0 |
| `neuralfree.server.models.autoload` | Whether a request naming an unloaded model loads it | true |

Every model serves in its own engine on the server's settings (`neuralfree.backend`, `neuralfree.server.replicas`,
...). A preset tunes one model alone:

```ini
# models.ini
[gemma-4b]
path = gemma-4b-q4.gguf      ; required; relative paths resolve beside the INI file
alias = gemma                 ; repeatable: another name that routes here
tag = vision                  ; repeatable: informational
n_ctx = 8192                  ; the model's own context (else the server's)
parallel = 2                  ; the model's own parallel sequences
ngl = 999                     ; GPU layers for this model
device = cuda0                ; device filter for this model
load_on_startup = true        ; load when the server starts
```

Unknown keys, malformed lines and missing paths are refused, naming the section, key and line.

**Routing.** POST endpoints (`/v1/chat/completions`, `/v1/completions`, `/completion`, `/infill`, `/v1/embeddings`,
`/score`, `/rerank`, `/classify`, `/pooling`, `/v1/messages`, `/v2/embed`, `/tokenize`, `/detokenize`,
`/apply-template`, ...) take the model from the body's `"model"` field; `GET /props` takes it from the
`?model=` query parameter (URL-encoded), with `?autoload=true|false` overriding the autoload per request. An
unknown name answers 404; a name with no model field answers 400; a loaded model serves; an unloaded one loads
first (autoload on) or refuses with 400 (autoload off, or `?autoload=false`). The response carries the routed
model's name. Requests are served on that model's engine, and naming it keeps it loaded: `models.max` evicts the
least recently used model that serves no request — a model with work in flight never gives its slot, and a load
that cannot free one waits within the request budget and refuses with 503 past it.

**Management.**

- `GET /v1/models` (also `GET /models`): the registry — per model the `id`, `aliases`, `tags`, `source`
  (`models_dir`, `preset` or `registered`), `can_remove` (registered models only), `context_length` once the
  engine runs, and the `status`: `unloaded` (with `failed: true` and the `error` after a failed load), `loading`,
  `loaded` or `sleeping`. `?reload=true` re-reads the directory and the preset: new files join, a removed or
  replaced one unloads and leaves.
- `POST /models/load` `{"model": name}`: opens the model's engine; unknown 404, already running (or sleeping) 400.
- `POST /models/unload` `{"model": name}`: closes it; unknown 404, not running 400.
- `POST /models` `{"model": "<gguf path>", "alias": [...], "tags": [...]}`: registers a local GGUF under its file
  name for routing and later removal; the path must be a readable `.gguf` file, and the name and aliases must be
  free.
- `DELETE /models?model=<name>`: removes a registered model (unloading it first, keeping its files); a model a
  directory or preset declares refuses — the server configuration keeps it.
- `GET /models/sse`: one `data:` frame per state change,
  `{"model": name, "event": "model_status", "data": {"status": ...}}`, a `: ping` comment between the events.

`GET /props` without `?model=` answers the router's own setup (`"role": "router"`, `max_instances`,
`models_autoload`); `GET /health` answers while the router runs.

The routes merged after the router follow its model rules too. POST routes take the model from the body's
`"model"` field, including `/tts` and `/v1/audio/speech` and the Responses API's `/v1/responses` (and
`/responses`). GET, DELETE and control routes take it from the URL-encoded `?model=` query parameter, with
`?autoload=true|false` overriding the autoload per request: `/props`, `/slots` and `/slots/{id}`,
`/lora-adapters` and the adapter load/unload routes, `/sleep`, `/wake_up`, `/is_sleeping`, `/pause`, `/resume`,
`/is_paused`, `/release_kv_cache_memory`, `/checkpoint/prepare|restore|status`, `/server_info`,
`/tokenizer_info`, `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache`, `/start_profile`,
`/stop_profile`, and `GET`/`DELETE` `/v1/responses/{id}` with its `input_items` and `cancel` routes. A stored
response read without a model is found on the model whose store holds its id. A missing name answers 400 and an
unknown one 404, as elsewhere in router mode.

The development routes in that list (`/server_info`, `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache`, `/sleep`, `/wake_up`, `/is_sleeping`, `/pause`, `/resume`, `/is_paused`, `/release_kv_cache_memory`, `/collective_rpc` and the weight-transfer routes) follow the same `?model=` rule but need `--dev-mode` first: with the switch off they answer 404 before the model query is read. The sleep-mode and profiler gates run before the model query as well: without `--enable-sleep-mode` the sleep operations and the checkpoint window refuse for every routed model, and without `--profiler-config` the profile window answers 404 for every routed model.

The server-global routes stay router-scoped: `/version`, `/ready`/`/readyz` and `/ping` (up while the router
serves, as is the Kubernetes readiness probe `/q/health/ready`), `/load` (summed over the open models),
`/abort_requests` (aborts the live generations of every open model), `/server_info` without a model (the
router's configuration), and `/tools` (the launch's tool registry, as llama.cpp's router serves it).

The bundled web UI follows the selected model: its Server panel reads `/props?model=` and its Slots panel reads
`/slots?model=` (in router mode with `&autoload=false`, so viewing a panel does not load a model), and the slots
panel shows the server's refusal message when the model cannot answer. In router mode the model selector labels
each entry with its `/v1/models` status (`unloaded`, `loading`, `loaded`, `sleeping`, `failed`). A Load button
appears for the selected unloaded or failed model and calls `POST /models/load`.

### Data-parallel serving

The server can hold several replicas of the model behind one API, as SGLang's and vLLM's data parallelism do. Each replica is an independent single-device engine with its own model copy, batch engine and pump thread, so the replicas decode in parallel without collectives.

- `neuralfree.server.replicas=N` (`-dp N`) opens `N` replicas (default 1).
- `neuralfree.server.devices="name a,name b"` (`--data-parallel-devices`) opens one replica per entry, on the device whose name contains it (the substring match of `neuralfree.device`). The list implies the replica count, and an explicit `neuralfree.server.replicas` must match it. A count below 1 or an empty entry is refused by name.

```console
java -jar server/build/<server jar> -m model.gguf --data-parallel-devices "NVIDIA GeForce RTX 5080,AMD Radeon 7900 XTX"
```

#### Ranks across several servers

A deployment can also spread its data-parallel ranks over several servers behind an external load balancer, as vLLM's external-LB and hybrid modes do. Replica `i` of a server serves global rank `neuralfree.server.dp.rank + i`, out of `neuralfree.server.dp.size` ranks in total.

- **Internal** (the default): `-dp N` serves every rank on this server.
- **External**: `-dp N --data-parallel-rank R` serves rank `R` alone. Start one such server per rank and point the balancer at them.
- **Hybrid**: `-dp N --data-parallel-size-local L --data-parallel-start-rank S` serves ranks `S` to `S+L-1` here, balanced internally, with the other ranks on other servers.

```console
# four ranks on four servers, one per GPU host
java -jar server/build/<server jar> -m model.gguf -dp 4 --data-parallel-rank 0 --port 8000
java -jar server/build/<server jar> -m model.gguf -dp 4 --data-parallel-rank 1 --port 8001
...
```

The ranks must lie inside `[0, size)`; a layout past the size is refused by name.

#### Routing

`neuralfree.server.load.balance` (`--load-balance-method`) picks how a request chooses a local replica, with SGLang's method names:

- `shortest_queue` (the default; also `auto` and `total_requests`): the replica with the least queued plus running work. Ties between idle replicas go round robin, except that a replica that recently served a prompt the request extends wins the tie, so its prefix cache reuses the rows. Each replica remembers the first 64 tokens of its last 64 routed prompts.
- `round_robin`: each request to the next replica in turn, whatever the load.
- `minimum_tokens` (also `total_tokens`): the replica with the fewest prompt tokens in flight (submitted and not yet finished). Ties go to the fewest requests in flight, then round robin.

An unknown name is refused by name.

#### Reporting

`GET /health`, `GET /props` and `GET /server_info` carry the layout:

- `data_parallel_size`: the global size; 0 while no model is loaded.
- `data_parallel_size_local`: the replicas on this server.
- Once a model is loaded, also:
  - `data_parallel_rank`: the global rank of the first local replica.
  - `data_parallel_external_lb`: true when other servers hold the other ranks.
  - `data_parallel_mode`: `internal`, `external` or `hybrid`.
  - `load_balance_method`.

`/health` is `ok` while every pump thread runs. Past one replica, `/props` sums `total_slots` and lists `replicas`, each with its `replica` index, global `rank`, `device`, `model_path`, `n_ctx`, `n_slots`, `queued`, `running`, `kv_cache_usage` and `prompt_tokens_in_flight`.

The `/metrics` series read the aggregate over the replicas. `neuralfree_dp_replicas` reports the local replica count and `neuralfree_dp_size` the global size. Each replica also has its own series, tagged with its pool index and global rank (`{replica="0",dp_rank="2"}`):

- `neuralfree_replica_requests_processing`
- `neuralfree_replica_requests_deferred`
- `neuralfree_replica_kv_cache_usage`
- `neuralfree_replica_prompt_tokens_in_flight`

They appear once the model opens.

### Distributed serving control

The replica pool a server opens with `--data-parallel-size` can be resized and repaired at runtime. Both control surfaces are opt-in; while a switch is off its routes answer 404 as an unregistered route does.

```
--enable-elastic-ep          serve the elastic-scale routes (neuralfree.server.elastic.ep)
--enable-fault-tolerance     serve the fault-tolerance routes (neuralfree.server.fault.tolerance)
```

#### Elastic data-parallel scaling

`POST /scale_elastic_ep` resizes the pool. A grow opens the new replicas before routing to them; a shrink stops routing to the removed replicas, waits up to `drain_timeout` seconds for their requests, then closes them. The global data-parallel size follows because the server holds every rank.

```bash
curl -X POST localhost:8080/scale_elastic_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size": 4, "drain_timeout": 120}'
# 200 {"message": "Scaled to 4 data parallel engines"}
# 400 an unusable body, naming the field
# 408 the removed replicas did not drain within drain_timeout
# 500 the scale failed; the pool is left as it was
```

`POST /is_scaling_elastic_ep` reports the lifecycle, as vLLM's external elastic EP status:

```json
{"is_scaling_elastic_ep": false, "phase": "completed", "epoch": "1",
 "error": null, "requested_data_parallel_size": 4}
```

`phase` is one of `idle`, `preparing`, `committing`, `completed` or `failed`; `is_scaling_elastic_ep` is true while a scale is `preparing` or `committing`. A server that holds only part of the deployment (`--data-parallel-rank`, `--data-parallel-size-local`, or a layout whose start rank is not 0) refuses the scale by name: its external balancer owns the topology. After a fault-tolerance `scale_down`, elastic scaling stays off.

#### Fault tolerance

`GET /v1/fault_tolerance/status` reports the pool in vLLM's engine-status shape. Each entry names its data-parallel rank, its health (`healthy`, `unhealthy`, `dead`), the failure that made it unhealthy (`fault_info`), and the state of the last instruction (`ft_state`, `last_ft_request_id`, `ft_error`).

```json
{"schema_version": 1, "total_engines": 2, "engines": [
  {"id": 0, "status": "healthy"},
  {"id": 1, "status": "unhealthy", "fault_info": "the engine did not wake: ..."}
]}
```

`POST /v1/fault_tolerance/apply` validates the body and answers 202 before the work runs; poll the status for the outcome.

```bash
curl -X POST localhost:8080/v1/fault_tolerance/apply \
  -H 'Content-Type: application/json' \
  -d '{"instruction": "retry", "request_id": "round-1"}'
# 202 {"message": "Request accepted; poll /v1/fault_tolerance/status for updates.",
#      "request_id": "round-1"}
```

- `retry` recovers every unhealthy or dead replica: it aborts its live requests, releases its engine and model, re-opens the model from the served path and clears the failure. A replica whose pump thread stopped is replaced in place. A retry on an all-healthy pool records the reason instead.
- `scale_down` takes `{"removed_dp_ranks": [1, 2]}` and drops those ranks from the pool. Only failed ranks may be removed, at least one replica must remain, and the deployment's global size keeps the removed slots (the remaining replicas keep their own ranks). After a `scale_down`, elastic scaling is disabled for the process.

Both routes and the switches are refused by name: an unknown instruction is a 400 `Invalid instruction: '…'.`, a missing `request_id` a 400 naming it.

#### KV event sources

`GET /kv_event_sources` reports the KV cache event publisher of every served replica keyed by data-parallel rank, as vLLM's instrumentator route does. It is empty when nothing is loaded or `--kv-events-config` publishes nothing. A ZMQ publisher reports its resolved transport fields with the rank's port offset (`endpoint`, `replay_endpoint`, `buffer_steps`, `hwm`, `max_queue_size`, `topic`); the engine's `log` and class-name publishers report `publisher` and `enable_kv_cache_events` only.

```json
{"0": {"enable_kv_cache_events": true, "publisher": "zmq",
       "endpoint": "tcp://*:5557", "replay_endpoint": null,
       "buffer_steps": 10000, "hwm": 100000, "max_queue_size": 100000, "topic": ""},
 "1": {"enable_kv_cache_events": true, "publisher": "zmq",
       "endpoint": "tcp://*:5558", "replay_endpoint": null,
       "buffer_steps": 10000, "hwm": 100000, "max_queue_size": 100000, "topic": ""}}
```

### Disaggregated prefill (`--kv-transfer-config`, `kv_transfer_params`)

vLLM's disaggregated prefill splits one request over two instances: a prefill instance runs the prompt and hands its
KV cache to a decode instance, which generates from it. Each instance names its side with `--kv-transfer-config`
(`neuralfree.server.kv.transfer.config`), vLLM's `KVTransferConfig` JSON:

| Field | Meaning | Default |
|---|---|---|
| `kv_connector` | `TcpKvConnector` (a pull over TCP from the prefill instance) or `FileKvConnector` (a directory both instances mount) | required |
| `kv_role` | `kv_producer` (prefill), `kv_consumer` (decode) or `kv_both` | required with a connector |
| `engine_id` | This instance's id, checked by the peer on every pull | a fresh UUID |
| `kv_ip`, `kv_port` | The address a producer binds and advertises; port 0 picks an ephemeral one | `127.0.0.1`, 14579 |
| `kv_buffer_size` | Bytes of transferred chains the producer keeps in memory | 1 GiB |
| `kv_load_failure_policy` | `fail` refuses a request whose pull fails; `recompute` prefills it locally instead | `fail` |
| `kv_connector_extra_config` | `TcpKvConnector`: `auth_token`, `connect_timeout_ms`, `io_timeout_ms`, `server_threads`, `kv_lease_seconds` (300), `max_payload_bytes`; `FileKvConnector`: `storage_path`, `max_payload_bytes` | |

Unknown or mistyped fields are refused by name. NIXL, Mooncake, LMCache and the multi- and offloading connectors are
refused by name with the native library or service they need, as are `kv_connector_module_path` and
`enable_permute_local_kv`.

A router sends the request to the prefill instance with `"kv_transfer_params": {"do_remote_decode": true}`. The
prefill instance runs the prompt with `max_tokens` forced to 1, stores the sequence's cached chain, and answers with
`kv_transfer_params` naming its connector address and a `transfer_id`. The router then sends the same request to the
decode instance with those parameters and `"do_remote_prefill": true`. The decode instance pulls the chain into the
prefix cache of the replica that will run the request, so the prompt adopts every prefilled row and computes only its
last token; its first sampled token is generated locally, as vLLM's decode instance does. A streamed response carries
`kv_transfer_params` on its finish chunk.

```bash
neuralfree serve -m model.gguf --port 8100 --kv-transfer-config \
  '{"kv_connector":"TcpKvConnector","kv_role":"kv_producer","kv_ip":"127.0.0.1","kv_port":14579}'
neuralfree serve -m model.gguf --port 8200 --kv-transfer-config \
  '{"kv_connector":"TcpKvConnector","kv_role":"kv_consumer"}'
```

The field rides `/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/cohere/v2/chat` and the scale-out
generate endpoint; llama.cpp's `/completion` and `/infill` refuse it and point at the OpenAI routes. The chain travels
in the slot-file format with its block layout, so a decode instance whose cache layout differs, or whose prompt the
chain does not start, refuses it naming the field that differs.

Refused by name: a request naming both sides or neither, `remote_block_ids` (this connector moves one chain under one
`transfer_id`), more than one prompt or `n`/`best_of` above 1, `use_beam_search` (beam search does not run on the
batch engine the transfer reads), and media or prompt-embedding prompts (a cached block holding media rows cannot leave
its memory). Both instances must run the same model with the same page size, KV group layout and cache types.

`GET /v1/kv_transfer/status` reports the instance's role, connector address and transfer counters (`enabled: false`
when no transfer is configured).

#### Securing the KV transfer

A prefill and decode pair that exchange KV over the network share a token. Give it in
`kv_connector_extra_config.auth_token`, or keep it off the command line and out of the properties:

```bash
export NEURALFREE_KV_TRANSFER_TOKEN='a long random value'          # the token itself, or
export NEURALFREE_KV_TRANSFER_TOKEN_FILE=/etc/neuralfree/kv.token  # a file holding it (trimmed)
neuralfree serve -m model.gguf --kv-transfer-config '{"kv_connector":"TcpKvConnector","kv_role":"kv_both"}'
```

The variable wins over the file. A token given twice (an environment source beside a JSON `auth_token`), a blank
variable, and an unreadable or empty file refuse the start by name, and no message repeats the token. Pulls compare
the token in constant time. A producer whose `kv_ip` is not a loopback address refuses to start without a token, since
an unauthenticated listener would hand every stored chain (other requests' prompt KV) to any peer that reaches the
port; bind `kv_ip` to `127.0.0.1` on a single host.

#### Restricting where a decode instance pulls from

A decode instance pulls only from the peers `kv_connector_extra_config.allowed_remote_hosts` allows: host names, IP
literals or CIDR ranges, each with an optional `:port` (an IPv6 literal with a port in brackets), as a JSON array or
one comma-separated string:

```json
{"kv_connector": "TcpKvConnector", "kv_role": "kv_consumer",
 "kv_connector_extra_config": {"allowed_remote_hosts": ["10.0.0.0/8", "[::1]:14579"]}}
```

Without the key, the allowed peers are the loopback addresses and the instance's own `kv_ip`, enough for a pair on
one host; an explicit list replaces that default, and an empty list allows nothing. A request whose
`remote_host:remote_port` is not allowed answers 400 naming `allowed_remote_hosts` before any connection, whatever the
`kv_load_failure_policy`. The host is resolved once and the connection uses the checked address, so DNS cannot
redirect it. `FileKvConnector` validates the key but connects nowhere; it names each file after the URL-safe base64
of the transfer id, so no transfer id reaches outside `storage_path`. Transfer ids the server generates are random
UUIDs.

### Per-request LoRA adapters

NeuralFree serves LoRA adapters per request, as vLLM's multi-LoRA serving and llama.cpp server's per-request `lora` list do. Requests that use different adapters, and requests on the base model, run in the same batched steps on CPU and GPU. Generation, embeddings, rerank, classification and scoring all take the same selection.

**Loading adapters.**
- Startup adapters: `--lora FILE` / `--lora-scaled FILE SCALE` (property `neuralfree.lora`). They are listed first in `/lora-adapters`, named after their file (without `.gguf`), and run at their startup scale for every request that selects nothing. A request can rescale or disable them. They cannot be unloaded or replaced at runtime (400); set their scale to 0 instead.
- Per-request modules: `--lora-modules name=path [name=path ...]` (or vLLM's JSON form `'{"name": "sql", "path": "/adapters/sql"}'`; property `neuralfree.server.lora.modules`). Each adapter is a GGUF LoRA file or a PEFT directory, validated against the model.
- At runtime: start the server with `-Dneuralfree.server.lora.runtime.updating=true` (or `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`). Then:
  - `POST /v1/load_lora_adapter {"lora_name": "sql", "lora_path": "/adapters/sql"}` loads an adapter; add `"load_inplace": true` to replace one.
  - `POST /v1/unload_lora_adapter {"lora_name": "sql"}` removes it. An adapter still used by running requests is refused with 409.
- Limits:
  - `--max-loras N` (`neuralfree.lora.max`, default 4): adapters resident, and distinct adapter/scale combinations running at once. A request whose combination finds every slot busy waits; other requests keep being admitted. A request that disables every adapter takes no slot.
  - `--max-lora-rank N` (`neuralfree.lora.max.rank`, default 64): the highest rank a per-request load accepts; a higher rank is refused by name. A startup adapter of a higher rank still loads, since `--lora` checks no rank, and requests can rescale it (see GPU memory below).
  - `--max-cpu-loras N` (`neuralfree.lora.cache`, default 8): unloaded adapters kept in the host cache.
- GPU memory: `--enable-lora` (`neuralfree.lora.enable=true`) reserves, when the model loads, the device memory of `--max-loras` slots of rank `--max-lora-rank` over every adaptable projection, as vLLM does. A load that cannot fit the reservation fails then, so a request never runs out of memory when it opens a slot. The server turns it on by itself when it serves adapters per request (`--lora`, `--lora-modules` or runtime updating). Without it, a GPU model refuses per-request adapter loads by naming the property. The CPU needs no reservation. A request that only rescales or disables startup adapters opens no slot copy: its slot runs the resident startup factors at the request's scale, so it needs no reservation and also works for a startup adapter above `--max-lora-rank`. A request that selects startup adapters beside per-request ones, or drops some startup adapters while rescaling others, uploads a slot copy and is bounded by the reservation.

**Selecting adapters per request.**
- vLLM style: set `model` to the adapter's name, e.g. `{"model": "sql", ...}`. The adapter runs alone at its default scale (1 for a per-request adapter), and responses report that model name.
- llama.cpp style: `"lora": [{"id": 0, "scale": 0.5}]`. Listed adapters run at their scale; unlisted ones, startup adapters included, run at 0; a missing scale counts as 0. An empty list means "no override".
- Using both forms in one request is refused with 400.
- Both forms work on `/completion`, `/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/embeddings`, `/embeddings`, `/v1/rerank` (and its aliases), `/classify`, `/score`, `/pooling` and `/generative_scoring`.
- `GET /lora-adapters` lists `[{"id", "path", "scale", "task_name", "prompt_prefix", "name"}]`, startup adapters first. `task_name` and `prompt_prefix` come from the adapter's PEFT metadata and are empty when it has none. An activated LoRA also lists `alora_invocation_string` and `alora_invocation_tokens`. Ids are load-order positions, so unloading an adapter renumbers the later ones.
- `POST /lora-adapters [{"id": 0, "scale": 1.0}]` sets the global scales used by requests that select nothing. Unlisted adapters go to 0. Startup adapters start at their startup scale and per-request ones at 0. A full sleep's wake keeps the global scales.
- `GET /v1/models` lists every adapter with `root` set to its path and `parent` set to the base model.

**Activated LoRA (aLoRA).** An adapter with invocation tokens (PEFT `alora_invocation_tokens`, GGUF `adapter.alora.invocation_tokens`) applies only from the last occurrence of its invocation sequence in the prompt on. The rows before it run without the adapter. If the prompt does not contain the sequence, the whole request runs without it. The sequence is searched among the prompt's text tokens, and a media chunk counts for the rows it occupies, as vLLM counts a media placeholder in `prompt_token_ids`. So in an image prompt the rows before the sequence, image rows included, run without the adapter. The following are refused with 400:
- an aLoRA selected together with other adapters, or loaded with `--lora` beside other startup adapters;
- an aLoRA on a pooling request, since a pooling pass runs every row under one adapter.

Speculative decoding serves adapters: the target's verify rows run under the request's selection, and a startup aLoRA's prefill splits at its invocation row. The drafter runs without adapters, as vLLM's drafter does not use the target's LoRA. `export-lora` refuses to merge an aLoRA.

**Notes.**
- The prefix cache only reuses rows computed under the same adapters at the same scales, and for an aLoRA, the same invocation start. The single-sequence loop and the speculative decoder apply the same rule.
- A full sleep's wake reloads the served adapters.
- Hybrid CPU+GPU models (`--n-gpu-layers`) and layer-split multi-device models serve per-request adapters too: each device and the host layers apply the deltas of their own layers. They have no paged memory, so their requests run one at a time.
- The single-sequence `GenerationLoop` (CLI generation, hybrid and split models) runs per-request selections, including aLoRA, and so does its speculative decoding.
- The audio decoder runs no per-request adapters; selecting one there is refused with 400.
- A CLIP dual encoder ([CLIP image and text embeddings](#clip-image-and-text-embeddings)) serves no adapters: the server refuses to start with `--lora-modules` or runtime updating beside one.

### Slots

`GET /slots` lists every slot of the pool with llama.cpp's slot status. Every slot reports `id`, `n_ctx`, `speculative` and `is_processing`. A slot's running request, or else its last one, adds `id_task`, `n_prompt_tokens`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`, `params` (the sampler parameters under llama.cpp's names, `samplers` included), `next_token` (`has_next_token`, `n_remain`, `n_decoded`), `prompt` and `generated`. A busy engine answers between steps. `?fail_on_no_slot=1` answers `503 {"error": {"message": "no slot available"}}` while no slot is idle.

Start the server with `--slot-save-path DIR` (`neuralfree.server.slot.save.path`) to turn on the slot actions; without it every action answers 501.

| Request | Answer |
|---|---|
| `POST /slots/{id}?action=save` with `{"filename": "a.state"}` | `id_slot`, `filename`, `n_saved` (tokens), `n_written` (bytes), `timings.save_ms` |
| `POST /slots/{id}?action=restore` with `{"filename": "a.state"}` | `id_slot`, `filename`, `n_restored`, `n_read`, `timings.restore_ms` |
| `POST /slots/{id}?action=erase` | `id_slot`, `n_erased` |

- A slot's state is the prompt and output its last request left in the prefix cache. Save writes those cached rows under `DIR`. Restore puts them back into the prefix cache, so the next request whose prompt starts with the restored tokens reuses them.
- Erase drops the slot's state and evicts the cached blocks no other slot uses.
- Files are portable between the CPU and GPU backends when the page size and cache layout match.
- An action waits until the engine is idle.

| Case | Status | Message |
|---|---|---|
| Unknown slot id | 400 | `Invalid slot ID` |
| Unknown action | 400 | `Invalid action` |
| Unreadable, foreign or mismatched restore file | 400 | `Unable to restore slot: ...` |
| No prefix cache (`neuralfree.prefix.reuse` off, a sliding-window model, a bidirectional model or an encoder-decoder), or a recurrent or compressed memory | 501 | names the cause |

### Choosing a chat template

**Choosing a chat template.** `--chat-template VALUE` (server and CLI; property `neuralfree.chat.template`) replaces
the model's `tokenizer.chat_template`. VALUE is one of:
- a built-in name: `chatml`, the mistral family (`mistral-v1`, `mistral-v3`, `mistral-v3-tekken`, `mistral-v7`,
  `mistral-v7-tekken`), and llama.cpp's legacy templates `llama2`, `llama2-sys`, `llama2-sys-bos`,
  `llama2-sys-strip`, `phi3`, `phi4`, `falcon3`, `zephyr`, `monarch`, `gemma`, `orion`, `openchat`, `vicuna`,
  `vicuna-orca`, `deepseek`, `deepseek2`, `deepseek3`, `deepseek-ocr`, `command-r`, `llama3`, `chatglm3`,
  `chatglm4`, `glmedge`, `minicpm`, `exaone3`, `exaone4`, `exaone-moe`, `rwkv-world`, `granite`, `granite-4.0`,
  `granite-4.1`, `gigachat`, `megrez`, `yandex`, `bailing`, `bailing-think`, `bailing2`, `llama4`, `smolvlm`,
  `hunyuan-moe`, `gpt-oss`, `hunyuan-dense`, `hunyuan-vl`, `kimi-k2`, `seed_oss`, `grok-2`, `pangu-embedded`,
  `solar-open`. These render the same text as llama.cpp's built-in C++ formatter for each name.
- Jinja source: any value containing `{`, `}` or a newline.
- a path to a template file: any other value, as in vLLM. The file may hold Jinja source or a built-in name.

A value that is none of these is refused at startup. `--chat-template-file PATH` always reads a file.

**The non-Jinja route.** `--no-jinja` (server and CLI; property `neuralfree.chat.jinja=false`) renders llama.cpp's
non-Jinja route: the template's text — the `--chat-template`/`--chat-template-file` override, or the model's
`tokenizer.chat_template` — is read as one of llama.cpp's known templates (`llm_chat_detect_template`,
src/llama-chat.cpp:89-240) and replaced by that built-in's source. `GET /props` then reports the built-in as
`chat_template`. The guess runs llama.cpp's substring checks in their order, so a model template llama.cpp detects,
NeuralFree detects; the checks are heuristics, not a Jinja parser, and the templates llama.cpp misses stay missed.
A template no style fits is refused at startup with its first 80 characters quoted — llama.cpp's refusal without
`--jinja` (for an override at flag parse, for the model's own template when the model loads). The override still
replaces the model's template, and the tool-use fallback does not apply to it, as in llama.cpp. One name is
detection-only: `dots1` is guessed from a template carrying `<|endofuserprompt|>` and renders like llama.cpp's
DOTS1 formatter, but `--chat-template dots1` is refused like any unknown name (llama.cpp's name table has no
`dots1` entry either). Two names are never guessed: `mistral-v7-tekken` and `deepseek-ocr` have no detection
checks, as in llama.cpp.

The templates whose llama.cpp formatters upper-case roles (`openchat`, `bailing`, `bailing-think`, `bailing2`)
do it byte-wise, as C's `toupper` does: ASCII letters change, every other character is left alone (`ïdea` renders
`ïDEA`, where a Unicode upper would give `ÏDEA`).

**Default template variables.** `--chat-template-kwargs JSON` (llama.cpp spelling) or
`--default-chat-template-kwargs JSON` (vLLM spelling) takes a JSON object of template variables, for example
`--chat-template-kwargs '{"enable_thinking": false}'`. The flags can be repeated and mixed; later keys win. The
property is `neuralfree.chat.template.kwargs`. The defaults apply to every route that renders a chat: chat
completions, Responses, input-token counts, `/apply-template`, `/tokenize` with messages, chat-form embeddings and
pooling, and Anthropic messages.

A request's own variables win over the defaults. That covers the request's `chat_template_kwargs`, the top-level
`enable_thinking` and `reasoning_effort`, and Anthropic's `thinking`. A request kwarg that is `null`, or the string
`"auto"`, counts as unset, so the default applies (vLLM's merge rule).

**CLI.** `--chat-template-kwargs JSON` binds the same variables in interactive chat, single-prompt and diffusion
runs. These variables are applied after `--thinking`'s `enable_thinking`, so the JSON can override it. In
`--server-base` mode they are sent as the request's `chat_template_kwargs`.

### Chat template capabilities

When NeuralFree loads a model, it probes the chat template the way llama.cpp does: it renders small test conversations and records which inputs the template reads. `GET /props` reports the result as `chat_template_caps`:

```json
"chat_template_caps": {
  "supports_object_arguments": true, "supports_parallel_tool_calls": true,
  "supports_preserve_reasoning": false, "supports_reasoning_effort": false,
  "supports_string_content": true, "supports_system_role": true, "supports_tool_calls": true,
  "supports_tools": true, "supports_typed_content": false, "reasoning_effort_levels": []
}
```

Messages reach the template in the forms it reads:

- **Content.** A template that reads only strings gets typed text parts joined with newlines. A template that iterates content parts gets a string wrapped as one `{"type": "text", "text": ...}` part. A template that reads both gets each message as you sent it. Media parts always become one string that carries the media markers.
- **Tool-call arguments.** A template that reads arguments as an object (for example through `| tojson` or `.items()`) gets an object. A template that prints them gets JSON text: text you sent stays as written, and an object you sent becomes compact JSON. Arguments that are not a JSON object are refused with 400.
- **System role.** A template that does not read the system message gets the system prompt merged into the next message.
- **`parallel_tool_calls`.** Without the field, a request allows several tool calls only when the template renders every call of a message (`supports_parallel_tool_calls`), as llama.cpp does. A template that renders only the first call (for example Llama 3.1) defaults to one call. So does a template that renders no tool calls at all, including the built-in ChatML fallback. A request that sets the field is served as sent: `false` holds the output to one call (the parsers keep the first, forcing grammars pin `maxItems: 1`). Anthropic's `tool_choice.disable_parallel_tool_use: true` is the explicit `false`, `false` the explicit `true`, and an absent switch takes the template default, as vLLM's Anthropic endpoint maps it.
- **A request's own template.** A request with a `chat_template` override is converted and defaulted by that template's capabilities, not the model's. `POST /apply-template` returns the capabilities of the template that rendered the prompt as `chat_template_caps`.
- **`tool_reference` parts.** A content part `{"type": "tool_reference", "name": "<tool>"}` reaches a template that reads typed content as written. Where the content joins to a string, the part is skipped with a warning. A part without a string `name` is refused with 400.
- **DeepSeek V4.1.** For a template that names the `｜DSML｜ calls>` block, a message with images or audio keeps each part as its own text block, with the media marker as a block, and the template joins them. Extra keys on tool calls (for example `namespace`) reach the template. As with every template, extra message keys (for example `task`) are passed through.
- **StepFun.** For StepFun templates, each text part is trimmed before the parts join, along with string content and `reasoning_content`.
- **Tools the template ignores.** When a request offers tools the template does not describe, the server logs a warning once per template.

Launch flags:

| Flag | Effect |
|---|---|
| `--chat-template-content-format auto\|string\|openai` | vLLM's override of the detected content format: `string` always sends strings, `openai` always sends typed parts (`neuralfree.chat.template.content.format`). |
| `--reasoning-preserve` | `preserve_reasoning` is `true`: templates that support it keep the reasoning of earlier assistant turns in the prompt (`neuralfree.chat.reasoning.preserve`). This is also the default without either flag, as in llama.cpp. |
| `--no-reasoning-preserve` | `preserve_reasoning` is `false`: earlier reasoning is dropped from the prompt. |

A request can set `"chat_template_kwargs": {"preserve_reasoning": true}` itself; the value must be a boolean. Like llama.cpp, it sets `preserve_thinking` to the value and `clear_thinking`, `truncate_history_thinking` and `drop_thinking` to its opposite. A kwarg sent explicitly still overrides any one of these. `reasoning_effort` is also bound as `reasoning_strength`.

### Per-request chat template and reasoning output

`POST /v1/chat/completions` (and `POST /v1/responses` through `extra_params`) takes the per-request controls of vLLM and llama.cpp:

- `chat_template` (string): a Jinja template or a built-in template name (for example `"gemma"`, never a file path) rendered for this request in place of the model's own, probed for system-role support the same way. It is honored only on a server launched with `--trust-request-chat-template` (`-Dneuralfree.server.trust.request.chat.template=true`), as in vLLM. Without the flag (the default), a request carrying a non-empty `chat_template`, or a `chat_template` key inside `chat_template_kwargs`, is refused with 400 naming the field and the flag. The same rule applies on every route that renders a chat: chat completions, `/v1/responses`, the input-token counts, `/apply-template`, `/tokenize` with messages, and chat-message embeddings and pooling. A template that does not parse, or that fails on this conversation, is refused with 400 naming `chat_template`. An empty string keeps the model's template.
- `documents` (array of objects): bound into the template as the `documents` variable, for retrieval templates (vLLM's RAG field). Each document must be a JSON object.
- `echo` (boolean, default false): prepends the last message's content to the answer when its role is the one generation continues: the assistant role under a generation prompt, otherwise the last message's own role (vLLM's chat echo). On a stream the echoed text arrives in its own delta right after the role delta.
- `include_reasoning` (boolean, default true): when false, neither `reasoning` nor `reasoning_content` reaches the response, in the message or as stream deltas. An output constraint then holds from the first token, as in vLLM.
- `reasoning_format` (llama.cpp; also on `/v1/completions` and `/completion`): `auto` and `deepseek` keep the configured reasoning split into `reasoning_content`. `deepseek-legacy` splits like `deepseek` except on a stream, where the reasoning stays inline in the content. `none` leaves the reasoning inline in the content:
  - With thinking tags, `none` returns the raw generated text with the tags included.
  - On a harmony model (gpt-oss), the output is still parsed as llama.cpp parses it, so tool calls still come back as `tool_calls`. Each analysis message stays in the content with its header and terminator (`<|channel|>analysis<|message|>...<|end|>`), while commentary and final messages contribute only their text. For example, `<|channel|>analysis<|message|>Thinking<|end|><|start|>assistant<|channel|>final<|message|>Hi` answers `<|channel|>analysis<|message|>Thinking<|end|>Hi`.
  - On a Muse model, a `to=self` message contributes its text to the content.

  The format also selects the routing of the output constraint gate and the tool-call grammar. An unknown name is refused with 400.
- **Reasoning field names.** Responses carry the reasoning under both vLLM's `reasoning` and llama.cpp's `reasoning_content`, in the assembled message and in every stream delta. An assistant message in the request may carry it under either name; both reach the template with the same text, so templates written for either name render the history, and `reasoning` wins when a message has both. A non-string `reasoning` (vLLM's prompt-embedding form) is refused with 400 naming the field.

### Reasoning budget

Models that wrap their thinking in tags (`<think>...</think>`, the gpt-oss `analysis` channel) can be capped, as llama.cpp's reasoning-budget sampler and vLLM's `thinking_token_budget` do. The engine counts the tokens generated inside a reasoning block and forces the end tag when the budget runs out. Prompt tokens count too, so a generation prompt that leaves reasoning open starts with its budget armed. A new start tag after a block ends, naturally or by force, gets a fresh budget.

Request fields (all optional; a malformed value is refused with 400 naming the field):

- `reasoning_budget_tokens` (llama.cpp; also on `POST /completion`) or `thinking_token_budget` (vLLM): the token budget of one reasoning block; `-1`, the default, disables it. When it runs out, the engine injects the budget message, if any, and then forces the first end tag token by token.
- `reasoning_budget_message`: text injected ahead of the forced end tag, such as `"Time is up, conclude."`.
- `reasoning_control: true`: arms the machinery without a budget, so the reasoning can be ended at runtime.
- `reasoning_budget_start_tag` and `reasoning_budget_end_tags` (alias `reasoning_budget_end_tag`; the first entry is the forced one): override the tags, which otherwise come from the model's reasoning parser. A budget on a model with no reasoning parser and no explicit tags is refused by name.
- On `POST /v1/messages`, `thinking: {"type": "enabled", "budget_tokens": N}` sets the budget.

The forced tokens stream like any output: the end tag routes through the reasoning parser, the answer continues as content, and `reasoning_tokens` counts them.

`POST /v1/chat/completions/control` with `{"id": "<completion id>", "action": "reasoning_end"}` ends the reasoning of a live chat completion: the budget message and the end tag are forced from the next token, and the answer streams on. The completion must be armed with `reasoning_control: true` or a budget. The id is the response's `chatcmpl-...` id, which a stream carries on every chunk; with `n > 1` every running choice is ended. A missing id, an unknown action, an id with no live completion, or a completion without reasoning control is refused with 400.

For a parser whose reasoning opens at the output's first token without a start tag, the budget counts from the first
generated token. Examples are `nemotron_v3` and `granite_thinking_parser` with thinking on, and a prompt that leaves
the block open. A start tag in the prompt keeps its own count, and a prompt that closed the block leaves the budget
idle.

### Offline batch runner

`neuralfree run-batch` runs a file of OpenAI Batch requests through the engine without the HTTP server, as `vllm run-batch` does. It starts the server jar's `BatchRunner` in a child JVM. It finds the jar the same way `neuralfree serve` does, and its `-D<name>=<value>` arguments become JVM properties of the runner. Each line of the input is one record:

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128}}
```

```console
neuralfree run-batch -i input.jsonl -o output.jsonl -m model.gguf -Dneuralfree.parallel=2
neuralfree run-batch -i https://example.com/batch/input.jsonl -o https://example.com/batch/output.jsonl -m model.gguf --output-tmp-dir /scratch
java -cp server/build/<server jar> com.eonmux.neuralfree.server.BatchRunner -i input.jsonl -o output.jsonl -m model.gguf
```

| Flag | Meaning |
|---|---|
| `-i`, `--input-file PATH\|URL` | The input JSONL (required): a local file or pipe (for example `/dev/stdin`), or an `http`/`https` URL read with GET |
| `-o`, `--output-file PATH\|URL` | The output JSONL (required; never the input file): a local file, or an `http`/`https` URL the finished file is uploaded to with one PUT |
| `-m`, `--model PATH` | The model; defaults to `neuralfree.server.model` |
| `--max-inflight N` | The most jobs one wave queues at the engine (default 1024) |
| `--output-tmp-dir DIR` | Where a URL input is downloaded and a URL output is staged before upload; it must be a writable directory (default: the system temporary directory) |
| `--enable-metrics` | Serve Prometheus metrics while the batch runs |
| `--host HOST` | The metrics server's address (default `127.0.0.1`); `--url HOST` is the deprecated spelling, and `--host` wins when both are given |
| `--port PORT` | The metrics server's port (default 8000; 0 picks a free port) |

Options also accept the `--name=value` spelling. `--host`, `--url` and `--port` without `--enable-metrics` are refused.

The engine and serving settings come from their `neuralfree.*` system properties, as in a server launch; generation needs the paged KV cache (for example `-Dneuralfree.parallel=2`).

- **Endpoints.** `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings` match exactly, `/score` and
  `/rerank` by suffix (so `/v1/score` routes too), and `method` must be `POST`. Blank lines are skipped.  `/v1/audio/transcriptions` and `/v1/audio/translations` route too and take vLLM's batch record body: the
  multipart fields as JSON and `file_url` (an http(s) URL or a data URL) in place of `file`
  ([Batch records](#batch-records)). An endpoint the server does not serve refuses by name in an error
  record, listing the supported endpoints.
- **Validation.** The whole file is read before the model loads. A record that is not a JSON object, or that lacks `custom_id`, `method`, `url` or a `body` object, stops the run with `Invalid request on line N: ...`.
- **Errors.** A record that fails while running (an unsupported URL, another method, `stream: true`, a refused request) writes an error record and the run goes on: `{"id": "batch_req-...", "custom_id": ..., "response": {"status_code": 400, "request_id": "req-...", "body": null}, "error": {"message": ..., "type": ...}}`.
- **Output.** A successful record carries the endpoint's own response body under `response.body` with `status_code` 200 and `error: null`. Records keep the input order. Chat and completions records share the engine's waves, so the file decodes continuously; embeddings and rerank records run between waves. A request that cannot finish inside `neuralfree.server.request.timeout` stops the run, and the records before it stay written (to a local output; a URL output is uploaded only after a complete run).
- **Remote files.** Only `http` and `https` URLs are accepted; any other `scheme://` (`file://`, `s3://`, `ftp://`) is refused, naming the option. A URL input is downloaded before validation and deleted after the run; a piped input is staged the same way, so it can be validated and then run. A URL output is written to a staged `tmp_batch_output_*.jsonl` file and uploaded with PUT once every record is written; no `Content-Type` is set, so pre-signed upload URLs work.
  - **Retries.** Connection failures, timeouts and `408`, `429` or `5xx` answers are retried up to 5 times, 5 seconds apart. Any other non-`2xx` answer fails at once, with the status and the start of the response body.
  - **Failed upload.** The staged file is kept, and the error names its path.
  - **Logging.** Messages show URLs without their query string, so signatures stay out of the log.
- **Metrics.** With `--enable-metrics` the runner serves the server's `neuralfree_*` engine metrics, the per-replica series and the `vllm:mm_cache_*` counters (registered once the model loads) (see [Operational endpoints](#operational-endpoints)) and the JVM's in the Prometheus text format on every path of `http://HOST:PORT/` (for example `/metrics`) from start-up until the run ends. The engine metrics read zero until the model loads. Only GET and HEAD are answered. When API keys are configured (`NEURALFREE_API_KEY`, `neuralfree.server.api.key`, `neuralfree.server.api.key.file`), a scrape must present one as `Authorization: Bearer <key>` or `x-api-key`, otherwise it gets 401. A port that cannot be bound stops the run, naming the address.

### Batched chat completions

`POST /v1/chat/completions/batch` (vLLM's batch chat) takes `messages` as a list of conversations, each a normal `messages` array, and applies every other field (`temperature`, `max_tokens`, `logprobs`, `response_format` and the rest) to each of them. The response is one `chat.completion` whose `choices[i]` answers conversation `i`, generated together in one batch, with `usage` summed over the conversations. `stream`, `tools`, a `tool_choice` other than `none` and `n > 1` are refused by name, as vLLM refuses them, and a malformed conversation is refused with `messages[i]: ...`.

### Scale-out serving

These routes let a prompt-serving front end and a pure generation engine run on separate hosts, as
vLLM's scale-out server does.

#### Render a prompt

- `POST /v1/chat/completions/render`
- `POST /v1/completions/render` (answers an array, one entry per prompt)
- `POST /v1/messages/render`
- `POST /v1/responses/render`

The body is the body you would send to the matching generation endpoint. The answer is the
rendered prompt and its sampling parameters:

```json
{
  "request_id": "chatcmpl-...",
  "token_ids": [1, 9038, 2501],
  "sampling_params": {"n": 1, "max_tokens": 256, "temperature": 0.7,
                      "stop": ["</s>"], "detokenize": true, "output_kind": "final_only"},
  "model": "my-model",
  "stream": false,
  "return_token_ids": false,
  "parse_context": {"tools": [], "tool_choice": "none", "parallel_tool_calls": true,
                    "include_reasoning": true, "chat_template_kwargs": {},
                    "history_tool_call_cnt": 0}
}
```

`sampling_params` uses vLLM's names (`n`, `max_tokens`, `min_tokens`, `temperature`, `top_p`,
`top_k`, `min_p`, `typical_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`,
`seed`, `stop`, `stop_token_ids`, `ignore_eos`, `logprobs`, `logprob_token_ids`,
`prompt_logprobs`, `skip_special_tokens`, `spaces_between_special_tokens`,
`include_stop_str_in_output`, `logit_bias`, `allowed_token_ids`, `bad_words`,
`repetition_detection`, `structured_outputs`/`guided_*`/`response_format`, `thinking_token_budget`,
`priority`, `cache_salt`, `detokenize`, `output_kind`) plus llama.cpp's sampler extensions
(`dry_*`, `xtc_*`, `mirostat*`, `samplers`, ...). A later `/inference/v1/generate` runs the answer
unchanged.

Media prompts render too ([below](#render-a-multimodal-prompt)). `best_of` greater than `n` is refused (vLLM's sampling parameters carry no
candidate count). `/v1/responses/render` is stateless: `previous_response_id`, `conversation` and a
stored `prompt` are refused by name.

#### Generate from tokens

`POST /inference/v1/generate`

```json
{
  "request_id": "gen-1",
  "token_ids": [1, 9038, 2501],
  "sampling_params": {"max_tokens": 32, "temperature": 0.7, "logprobs": 2},
  "model": "my-model",
  "output_mode": "tokens",
  "stream": false,
  "stream_options": {"include_usage": true, "continuous_usage_stats": false},
  "return_token_ids": false,
  "cache_salt": "tenant-a",
  "priority": 0
}
```

- `token_ids` and `sampling_params` are required; `sampling_params` is required and its
  unsupported fields are refused by name.
- `output_mode` is `tokens` (ids only) or `text` (ids plus decoded text). `stream: true` answers
  server-sent events with `token_ids` (or `text`) deltas, a finish chunk, the optional usage chunk
  and `data: [DONE]`; `return_token_ids` puts the prompt ids on the first chunk.
- `logprobs` is vLLM's count of reported tokens per step; tokens mode answers
  `{"content": [{token_id, logprob, rank, top_logprobs}]}` and text mode the OpenAI chat form.
- `structured_outputs`, `guided_*`, `response_format`, `grammar` and `json_schema` in
  `sampling_params` constrain the output; `reasoning_ended` says whether the prompt already closed
  its reasoning so the constraint applies from the first token.
- `features` and `content_parts` carry a media prompt ([below](#generate-from-features-or-content-parts)) and
  `token_offsets` the prompt's character spans; `kv_transfer_params` runs a [disaggregated prefill](#disaggregated-prefill---kv-transfer-config-kv_transfer_params)
  transfer, and `ec_transfer_params` is refused by name.
- `POST /inference/v1/abort_requests` with `{"request_ids": ["gen-1"]}` cancels live generate
  requests and answers an empty 200.

#### Derender a generate response

- `POST /v1/chat/completions/derender`
- `POST /v1/completions/derender`

Non-streamed chat request:

```json
{
  "model": "my-model",
  "generate_response": {"request_id": "gen-1", "output_mode": "tokens",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "token_ids": [9038, 2501]}]},
  "prompt_tokens": 3,
  "prompt_token_ids": [1, 9038, 2501],
  "chat_request": {"messages": [{"role": "user", "content": "hello"}]}
}
```

The answer is an ordinary `chat.completion` (message content, reasoning, tool calls, usage). The
completions form takes `generate_responses` (an array), `prompt_tokens` and `prompt_token_ids`
arrays parallel to it, and answers a `text_completion`. `prompt_token_ids` seeds detokenization so
the first output token keeps its leading space. Supply the original `chat_request` for tool and
reasoning parsing; without it the answer is the raw detokenized text.

`stream: true` takes the stateless per-chunk form: each call carries one `generate_chunk` from
`/inference/v1/generate` and the `stream_state` the previous call returned; the answer is
`{"chunk": <OpenAI stream chunk>, "stream_state": {...}}`. Carry the state unchanged between calls.

#### Token counts

`POST /v1/chat/completions/input_tokens` (and `/chat/completions/input_tokens`) and
`POST /v1/responses/input_tokens` (and `/responses/input_tokens`) answer
`{"object": "response.input_tokens", "input_tokens": n}` over the rendered prompt, media rows
included; `POST /v1/messages/count_tokens` answers Anthropic's `{"input_tokens": n}`.

A render server and a generate server can also pass media prompts between them: the media travels beside the tokens.

#### Render a multimodal prompt

`POST /v1/chat/completions/render`, `/v1/completions/render`, `/v1/messages/render` and `/v1/responses/render` accept
the same media inputs as their generation endpoints: chat `image_url`, `input_audio`, `audio_url`, `video_url`,
`input_video`, `image_embeds` and `audio_embeds` content parts, Anthropic image blocks, and llama.cpp's
`multimodal_data` on a completions prompt. The router needs a projector (`--mmproj`) for image, audio and video
input; embedding parts need none.

The answer is the usual `GenerateRequest` plus two fields:

```json
{
  "request_id": "chatcmpl-...",
  "token_ids": [1, 9038, 0, 0, 0, 0, 2501],
  "features": {
    "mm_hashes": {"image": ["image:content:9f2c..."]},
    "mm_placeholders": {"image": [{"offset": 2, "length": 4}]},
    "kwargs_data": {"image": ["iVBORw0KGgo..."]}
  },
  "token_offsets": null,
  "sampling_params": {"n": 1, "max_tokens": 256, "...": "..."},
  "model": "my-model",
  "stream": false,
  "return_token_ids": false
}
```

- `token_ids` is the fully rendered prompt: one id per KV row, so its length is the prompt token count. Every media
  row is written as token id `0`; a generate server replaces those rows with the item's embeddings.
- `features` is null for a text prompt. Its three members are parallel per modality:
  - `mm_hashes` holds each item's key handle: `image:content:<sha256>`, `image:uuid:<uuid>` or, for a precomputed
    embedding part, `embeddings:rows:<sha256>`. The handle is the identity the projector's encoded-media cache and
    the prefix cache share.
  - `mm_placeholders` holds one `{"offset", "length"}` row span per item. The span covers everything the item
    contributes to the prompt — its marker text and its embedding rows for an image, audio or video part, its rows
    for an embedding part. Spans are ordered and do not overlap.
  - `kwargs_data` holds one entry per item: base64 of the media source (the image, audio or video bytes the render
    server loaded) for a `content`- or `uuid`-keyed item, base64 of the embedding rows for a `rows`-keyed item, or
    null for an item sent by `uuid` alone whose bytes the rendering server did not hold (the receiving server must
    then find it in its own media cache). The field is omitted when every entry is null.
- `token_offsets` is null for a media prompt: a media row has no character span in the rendered text.

The same media may also be sent as `content_parts` to the generate endpoint instead of `features` (see below).

#### Generate from features or content parts

`POST /inference/v1/generate` accepts a multimodal prompt in either of two shapes.

**Features** — send a render answer unchanged (plus `sampling_params`). The prompt is the expanded `token_ids`; each
`mm_placeholders` span is replaced by its item, encoded through the media cache keyed by the item's `mm_hashes`
handle:

```json
{
  "token_ids": [1, 9038, 0, 0, 0, 0, 2501],
  "features": {
    "mm_hashes": {"image": ["image:content:9f2c..."]},
    "mm_placeholders": {"image": [{"offset": 2, "length": 4}]},
    "kwargs_data": {"image": ["iVBORw0KGgo..."]}
  },
  "sampling_params": {"max_tokens": 32, "temperature": 0.0},
  "return_token_ids": true
}
```

On the server that rendered the prompt the encoded-media cache already holds the item under that hash, so the
generate step splices the cached rows without decoding or encoding again. On another server, `kwargs_data`
re-encodes the item; a null entry with no cache entry is a 400 naming the item. Both servers must run the same model
and projector for the rows to match. With `return_token_ids`, the answer carries the expanded `prompt_token_ids`
and the `mm_placeholders` of the prompt (on the first stream chunk when `stream` is true).

**Content parts** — send the *collapsed* prompt (one `-1` token in place of each media item, its other tokens
unchanged) and the raw parts in order:

```json
{
  "token_ids": [1, 9038, -1, 2501],
  "content_parts": [{"type": "image_url", "url": "data:image/png;base64,...", "uuid": "optional"}],
  "sampling_params": {"max_tokens": 32},
  "return_token_ids": true
}
```

Each `-1` is replaced by the next part's marker text and embeddings, loaded through the same media connector and
media cache as the chat routes; `prompt_token_ids` in the answer is the expanded prompt. `url` and `uuid` follow
the chat media rules; a cached `uuid` may be sent without a URL. `features` and `content_parts` are mutually
exclusive.

#### Token offsets

Set `"return_token_offsets": true` on any render request. The answer's `token_offsets` is then an array with one
`[start, end]` character span per token, relative to the rendered prompt text (for chat, messages and responses the
templated prompt; for completions the prompt string). A token with no source span, such as an inserted BOS, reports
`[0, 0]`. Offsets are computed over the whole prompt and cut with `truncate_prompt_tokens`, so a left-truncated
prompt keeps the offsets of the tokens it kept. Without the flag, or for a media prompt, `token_offsets` is null.

`/inference/v1/generate` accepts `token_offsets` beside `token_ids`: when present it must hold one ordered,
non-negative `[start, end]` pair per prompt token. They describe the prompt and are not needed by generation.

#### Derender from generate (`output_mode`)

`"output_mode": "derender"` answers the assistant message a coupled chat endpoint would, without a separate
`/derender` call:

```json
{
  "token_ids": [1, 9038, 2501],
  "sampling_params": {"max_tokens": 32, "temperature": 0.0},
  "output_mode": "derender",
  "parse_context": {"tools": [], "tool_choice": "none", "parallel_tool_calls": true,
                    "include_reasoning": true, "chat_template_kwargs": {}}
}
```

Each choice carries `message` (`role`, `content`, `reasoning_content`/`reasoning`, `tool_calls`) beside
`finish_reason`, `stop_reason` and `token_ids`. `parse_context` is the one `/render` returned; a server with a tool
or reasoning parser requires it and returns a 400 without one, and an unknown `parse_context` field is refused by
name at this level (it is ignored at the other levels). `derender` needs `sampling_params.detokenize` true and does
not support `stream: true`. With `include_reasoning: false`, `token_ids` and `logprobs` are omitted so the hidden
reasoning is not exposed. `logprobs` follow the chat shape (decoded token strings and `bytes`).

#### Scale-out refusals

- Image, audio and video parts need a projector; without one they are refused with llama.cpp's `--mmproj` hint.
- `prompt_embeds` on `/v1/completions/render` is still refused: an embedding-only prompt has no token ids to render.
  Chat `image_embeds`/`audio_embeds` parts are supported and travel in `features`.
- `features` and `content_parts` together are refused; so are malformed handles, mismatched parallel arrays,
  overlapping spans, and entries missing from `features.kwargs_data` when the media cache does not hold them.
- `ec_transfer_params`, `features.mm_metadata` and `mm_payload: "none"` are not supported by this build.

### Watermarking generated text

The server can mark generated tokens with vLLM's Gumbel-max text watermark. Start it with the secret key and, optionally,
the algorithm and its settings:

```bash
export NEURALFREE_WATERMARK_KEY=123456789   # the secret key; see below
neuralfree serve model.gguf --watermark-config '{"context_width": 4}' --port 8080
```

The flag sets `neuralfree.watermark.config`, so `-Dneuralfree.watermark.config='...'` or a Quarkus `application.properties`
entry works the same way. The JSON fields are vLLM's:

| Field | Meaning | Default |
|---|---|---|
| `key` | Secret 64-bit unsigned key (number or decimal string); prefer `NEURALFREE_WATERMARK_KEY` ([below](#where-the-key-comes-from)) | — |
| `algorithm` | `gumbel` (one key) or `dual_key_gumbel` (two routed keys) | `gumbel` |
| `alpha` | Dual-key weight of key B, in `[0, 1]`; key A takes `1 - alpha` | `0.1` |
| `context_width` | Prior tokens the pseudorandom function keys on | `4` |
| `deduplicate_contexts` | Repeated-context search: `none`, `single_turn` (generated tokens) or `all` (prompt too) | `single_turn` |
| `deduplicate_contexts_max_history` | Most recent positions searched, or `null` for the whole scope | `8192` |
| `prf` | Pseudorandom function; only `philox` | `philox` |
| `allow_target_only_watermarking` | Draft-stage opt-out under speculative decoding | `false` |

Once configured, every text generation route watermarks: OpenAI chat/completions and Responses, Cohere Chat v2,
llama.cpp `/completion`, Anthropic Messages, batch runs and scale-out. Greedy decoding (temperature 0) never
watermarks, because there is no distribution to shift. A chain that selects tokens itself (mirostat, adaptive-p)
refuses a watermark by name.

Requests opt out (or back in) with vLLM's sampling field; an absent field inherits the engine configuration:

```json
{"model": "model.gguf", "messages": [{"role": "user", "content": "hello"}], "watermarking": false}
```

Asking for `"watermarking": true` on a server started without `--watermark-config` is a 400 reporting the missing
configuration. Audio output refuses `"watermarking": true`: the speech sampler draws codec frames, not text.

Detection is served by `POST /v1/watermark/detect`, using the engine's key, algorithm, `alpha` and context width:

```bash
curl -s http://localhost:8080/v1/watermark/detect \
  -H 'Content-Type: application/json' \
  -d '{"token_ids": [1, 2, 3, 4], "p_value_threshold": 0.01, "deduplicate_contexts": true}'
```

The body names exactly one sequence: `token_ids` (the ids generation produced) or `text` (tokenized without a BOS).
`p_value_threshold` is in `(0, 1)` and defaults to `0.01`; `deduplicate_contexts` defaults to `true`, scoring a repeated
context once. The answer is:

```json
{"score": 41.2, "p_value": 0.0002, "num_scored_tokens": 32,
 "is_watermarked": true, "algorithm": "gumbel", "context_width": 4}
```

`p_value` is the Gamma survival probability of the score under unwatermarked text, so `is_watermarked` is
`p_value <= p_value_threshold`. A server started without a watermark refuses detection with 400 naming the missing
configuration. Detection needs the same secret key as generation; a different key does not detect the text.

Speculative decoding keeps the watermark ([below](#watermarking-under-speculative-decoding)).

#### Where the key comes from

A watermark is configured with vLLM's JSON fields through `--watermark-config` (or the
`neuralfree.watermark.config` property). Keep the secret 64-bit key out of that JSON: the flag
appears in `ps`, `/proc/<pid>/cmdline` and shell history. Put it in the environment instead:

```sh
export NEURALFREE_WATERMARK_KEY=18446744073709551615
neuralfree serve --model model.gguf \
  --watermark-config '{"algorithm": "dual_key_gumbel", "context_width": 4}'
```

Or point `NEURALFREE_WATERMARK_KEY_FILE` at a file whose trimmed contents are the same unsigned
64-bit decimal:

```sh
printf '%s\n' 18446744073709551615 > /run/secrets/watermark-key
export NEURALFREE_WATERMARK_KEY_FILE=/run/secrets/watermark-key
```

`NEURALFREE_WATERMARK_KEY` takes precedence over `NEURALFREE_WATERMARK_KEY_FILE`. Either one
alone turns watermarking on at the default settings, so the JSON can carry just the fields that
differ from them. The environment is inherited by the processes the server starts, so the key
never appears on a command line, the CPU-SIMD relaunch that re-executes the JVM with
`--add-modules=jdk.incubator.vector` included.

vLLM's JSON `key` field is still accepted when no environment source is set. Giving an
environment source and the JSON `key` together is refused by name rather than silently choosing
one. When the key does come from the JSON (`--watermark-config` or a
`-Dneuralfree.watermark.config` property), the server logs one warning, without the value,
recommending the environment form.

The key is never reported: `/server_info` omits the watermark configuration, `/props` carries no
settings, `collect-env` shows only whether `NEURALFREE_WATERMARK_KEY` is set, and the
configuration's `toString`, logs and error messages leave out the key and the raw JSON.

#### Watermarking under speculative decoding

A watermark follows the request into speculative decoding with vLLM's draft/target split. Start a speculative
server with a dual-key watermark, or enable a single-key one together with the target-only opt-in:

```bash
export NEURALFREE_WATERMARK_KEY=123456789   # the secret key ([Where the key comes from](#where-the-key-comes-from))
neuralfree serve model.gguf --draft-model draft.gguf --draft-max 4 \
  --watermark-config '{"algorithm": "dual_key_gumbel", "context_width": 4}'

neuralfree serve model.gguf --draft-model draft.gguf --draft-max 4 \
  --watermark-config '{"allow_target_only_watermarking": true}'
```

The same settings reach the Java API through a speculative session:

```java
EngineConfig cfg = EngineConfig.builder()
        .draftModelPath("draft.gguf").draftMax(4)
        .speculativeOptions(SpeculativeOptions.builder().verifyMode(VerifyMode.REJECTION).build())
        .build();
WatermarkConfig watermark = WatermarkConfig.builder(123456789L)
        .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).contextWidth(4).build();
GenerationRequest request = GenerationRequest.builder().maxNewTokens(256)
        .sampler(SamplerConfig.builder().temperature(0.8f).seed(7).watermark(watermark).build())
        .build();
try (SpeculativeSession s = LLMModelFactory.speculate(model, cfg)) {
    s.generate(prompt, request, null);
}
```

How the tokens are marked:

- **Dual key (`dual_key_gumbel`).** The drafter draws its proposals under key A; the target verifies by standard
  rejection sampling, so a confirmed proposal keeps key A while a replacement and the round's bonus token are
  drawn under key B. `alpha` (default `0.1`) weights key B in the detector's fused score.
- **Single key with `allow_target_only_watermarking`.** The draft tokens are ordinary samples; the target's own
  draws (accepted drafts are not re-keyed, replacements and bonuses are) carry the single key.
- **Greedy decoding.** Temperature 0 never watermarks, with or without speculation.

Requirements, refused by name otherwise:

- The verify mode must be `neuralfree.draft.verify=rejection` (vLLM's `rejection_sample_method='standard'`).
- The drafter must be able to sample under the draft key: a separate draft model, the MTP/NextN head, EAGLE 1/3/fc,
  gemma4-assistant or DFlash. Lookup and suffix drafters replay stored tokens and refuse.
- A single-key `gumbel` configuration needs `allow_target_only_watermarking=true`; a dual-key configuration does
  not.
- DFlash2 and LiLiCorr refuse watermarking by name, as in vLLM.
- A separate draft model whose vocabulary is mapped onto the target's refuses: its keyed draws would not be the target ids the detector scores. Drafters with a `d2t` table (NextN/MTP, EAGLE 3, EAGLE fc) scatter their logits to the target vocabulary and are supported.

Under a DFlash drafter the watermark forces the host readout of the block (the keyed draw needs the logits); the
target keeps its device paths, and the draft still drafts on the device when its readout is the host's.

Detection is unchanged: `POST /v1/watermark/detect` with the engine's key, algorithm, `alpha` and context width
scores the produced text exactly as for a non-speculative generation.

### gRPC API

The server speaks gRPC only when launched with `--grpc` (`neuralfree.server.grpc=true`), as vLLM serves it only with `vllm serve --grpc`. Without the flag the gRPC surface is off: every gRPC call — generation, server reflection and the gRPC health service — answers `UNIMPLEMENTED`, whatever port it reaches. With the flag, the server speaks gRPC on the same host and port as HTTP. The schema is `server/src/main/proto/neuralfree.proto` (protobuf package `neuralfree`, Java package `com.eonmux.neuralfree.grpc`); the generated Java classes ship in the server jar. Server reflection and the standard `grpc.health.v1.Health` service are enabled, so `grpcurl` can discover the API without the `.proto` file.

Two services are served:

- `neuralfree.Inference` — `Generate` (one response with every requested sequence) and `GenerateStream` (server streaming: the first message carries the prompt info, the next ones carry one generated piece each, and each sequence's final message carries its finish info).
- `neuralfree.Control` — `GetServerInfo`, `GetModelInfo`, `Abort`, `LoadLora`, `UnloadLora`, `ListLoras`, `GetKvEventSources`, `PauseGeneration`, `ResumeGeneration`, `IsPaused`, `Sleep`, `WakeUp`, `IsSleeping`.

A `GenerateRequest` carries `request_id` (used by `Abort` and duplicate detection), `model`, a `text` or `token_ids` prompt, `temperature`, `sampling` (`num_sequences`, `top_k`, `top_p`, `min_p`, `typical_p`, `min_keep`, `seed`), `decoding` (`presence_penalty`, `frequency_penalty`, `repetition_penalty`, `penalty_last_n`, `logit_bias`, `allowed_token_ids`, `bad_words`, and one structured-output constraint: `json`, `regex`, `choice`, `grammar`, `json_object` or `structural_tag`), `stopping` (`max_new_tokens`, `min_new_tokens`, `stop_token_ids`, `stop_strings`, `include_stop_strings`, `ignore_eos`, `thinking_token_budget`), `response` (`prompt_token_ids`, `prompt_logprobs`, `prompt_candidates`, `output_text`, `output_token_ids`, `output_logprobs`, `output_candidates`, `skip_special_tokens`), `kv` (`cache_salt`, `bypass_prefix_cache`), `truncate_prompt_tokens`, `priority` and `lora_name`. Omitted fields keep the server or model defaults; `max_new_tokens: 0` keeps the server default, `num_sequences: 0` means one, and `truncate_prompt_tokens: 0` keeps the whole prompt (the leading N tokens are kept, vLLM's right truncation side).

Each `SequenceOutput` carries its sequence `index`, the new `text`, `num_tokens`, and — when asked for — `token_ids` and per-token `logprobs` with the alternatives in `candidates`. The final output of a sequence carries `finish_info`: `LENGTH`, `STOP` (with `stop_token_id`, `eos_token_id` or `stop_string`) or `ABORTED`, plus `num_output_tokens` and `num_cached_tokens`. `PromptInfo` reports `num_prompt_tokens`, the requested `token_ids`, the requested per-position `logprobs`, and `num_cached_tokens`.

Refusals use the gRPC status that mirrors the HTTP answer and name the field or switch to change: `INVALID_ARGUMENT` for a malformed or unsupported setting, `NOT_FOUND` for an unknown model or LoRA adapter, `UNAVAILABLE` for an unloaded, sleeping or paused engine, `FAILED_PRECONDITION` for a conflict, and `UNIMPLEMENTED` for a refused capability or a call while the gRPC surface is off.

#### Authentication

When the server has API keys (`--api-key`, `--api-key-file`, `NEURALFREE_API_KEY` or `neuralfree.server.api.key`), every gRPC call must present one in its `authorization: Bearer <key>` metadata; the Anthropic-style `x-api-key: <key>` metadata is also accepted. The check uses the same key set and constant-time comparison as the HTTP `ApiKeyFilter`, so one launch's keys protect both transports, and a missing or wrong key answers `UNAUTHENTICATED`. Server reflection and the gRPC health service are covered like every other call, with one exception: the standard `grpc.health.v1.Health/Check` stays open while the HTTP `/health` probe is open, so a probe that works over HTTP works over gRPC.

#### Control gates

The `Control` calls apply exactly the switches their HTTP routes apply:

- `GetServerInfo`, `Abort`, `PauseGeneration`, `ResumeGeneration`, `IsPaused`, `Sleep`, `WakeUp` and `IsSleeping` are vLLM's development routes. Without `--dev-mode` (`neuralfree.server.dev.mode`) they answer `NOT_FOUND`, as those HTTP routes answer 404.
- `Sleep` and `WakeUp` additionally need sleep mode (`--enable-sleep-mode`, `neuralfree.server.sleep.mode`); without it they answer `INVALID_ARGUMENT`, the HTTP routes' 400 sleep-mode refusal.
- `LoadLora` and `UnloadLora` need runtime adapter updating (`neuralfree.server.lora.runtime.updating=true`, vLLM's `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`); without it they answer `NOT_FOUND`.
- `GetModelInfo`, `ListLoras`, `GetKvEventSources` and the `Inference` calls are not development routes and stay reachable.

In the multi-model router mode, `model` selects the engine exactly as the HTTP routes' `?model=` does; the control calls accept a `model` too.

```bash
# --grpc is on; server reflection is on, so no proto file is needed
grpcurl -plaintext -d '{"text":"Hello","stopping":{"max_new_tokens":32}}' \
  127.0.0.1:8080 neuralfree.Inference/Generate

grpcurl -plaintext -H 'authorization: Bearer sk-secret' \
  -d '{"request_id":"r1","text":"Hello","stopping":{"max_new_tokens":32}}' \
  127.0.0.1:8080 neuralfree.Inference/GenerateStream

grpcurl -plaintext -H 'authorization: Bearer sk-secret' \
  -d '{}' 127.0.0.1:8080 neuralfree.Control/GetServerInfo
grpcurl -plaintext -H 'authorization: Bearer sk-secret' \
  -d '{"model":"qwen"}' 127.0.0.1:8080 neuralfree.Control/GetModelInfo
grpcurl -plaintext -H 'authorization: Bearer sk-secret' \
  -d '{"request_ids":["r1"]}' 127.0.0.1:8080 neuralfree.Control/Abort
```

```python
# python -m grpc_tools.protoc -I server/src/main/proto --python_out=. --grpc_python_out=. \
#   server/src/main/proto/neuralfree.proto
import grpc
from neuralfree_pb2 import GenerateRequest, ResponseOptions, StoppingCriteria
from neuralfree_pb2_grpc import InferenceStub

# the key metadata every call carries when the server has API keys
metadata = (("authorization", "Bearer sk-secret"),)

with grpc.insecure_channel("127.0.0.1:8080") as channel:
    stub = InferenceStub(channel)
    reply = stub.Generate(GenerateRequest(
        text="Hello",
        stopping=StoppingCriteria(max_new_tokens=32),
        response=ResponseOptions(output_token_ids=True)), metadata=metadata)
    print(reply.outputs[0].text)

    for message in stub.GenerateStream(GenerateRequest(
            text="Hello", stopping=StoppingCriteria(max_new_tokens=32)), metadata=metadata):
        for output in message.outputs:
            print(output.text, end="", flush=True)
```

With `--grpc` off no gRPC call is served. With it on, the API key (when configured) protects every call; bind to loopback or put the port behind a gateway when the server is exposed.

### Operational endpoints

| Endpoint | Answer |
|---|---|
| `GET /health`, `GET /v1/health` | llama.cpp semantics: `200 {"status":"ok"}` while a model is open, `503` with `{"error":{"message":…,"type":"unavailable_error"}}` before. |
| `GET /ready`, `GET /readyz` | Empty `200` once the model is open, `503` with the error envelope before. |
| `GET /ping`, `POST /ping` | SageMaker probe; answers as `/health` does, with an empty body. |
| `GET /q/health/ready` | MicroProfile readiness: `UP` with the `engine-model` check once a model is open, `DOWN` (503) before. `/q/health/live` stays the process liveness. |
| `GET /version` | `{"version": "<build version>"}`. |
| `GET /server_info` | Configuration and state: `version`, `loaded`, `model_path`, `model_alias`, `served_names`, `model_arch`, `context_length`, `slots`, `backend`, `prefix_reuse`, `request_timeout_seconds`, `spm_infill`, `response_store`, and `neuralfree_env` (the `neuralfree.*` system properties, leaving out any whose name contains key/secret/password/token). `config_format=json` is accepted; `text` is refused with 400. Needs `--dev-mode`. |
| `GET /load` | `{"server_load": n}`: requests in flight (queued, waiting, running or preempted generations, plus pooling calls waiting on the model). |
| `GET /tokenizer_info` | `tokenizer_class`, `tokenizer_pre`, `vocab_size`, `add_bos_token`, `add_eos_token`, `bos_token`/`eos_token`/`pad_token`/`unk_token` (text) and their `*_token_id`, `stop_token_ids`, `model_max_length`, `chat_template`. 503 while unloaded. |
| `POST /reset_prefix_cache` | Drops every cached prefix: `{"success": true}`, or `{"success": false}` while requests are live (retry later). `reset_running_requests=true` and `reset_external=true` are refused with 400. Needs `--dev-mode`. |
| `POST /abort_requests` | Aborts every live generation and answers `{"aborted": n}`; aborted clients get `finish_reason: "abort"`. The body may be empty or `{"abort_all": true|false}`; any other field is refused with 400. Needs `--dev-mode`. |
| `POST /start_profile`, `POST /stop_profile` | Profile window; **served only with `--profiler-config`**, as vLLM registers it only with a profiler, and answers 404 without one. `/start_profile` (empty body) opens it and starts a Java Flight Recorder recording; a second start answers 400. `/stop_profile` closes it, writes `profile-<timestamp>.jfr` into the configured directory (`torch_profiler_dir`, or `neuralfree.server.profile.dir`) and reports `duration_seconds`, `prompt_tokens`, `prompt_tokens_cached`, `prompt_seconds`, `generated_tokens`, `generated_tokens_per_second`, `decode_seconds`, `decode_steps`, `preemptions`, `recomputed_tokens` and the recording's absolute `recording` path; with no window open it answers 400. vLLM's override fields (`profile_prefix`, `delay_iterations`, `max_iterations`, `profiler_kwargs`) are refused with 400. The routes are API-key protected. |

When API keys are configured, `/health`, `/v1/health`, `/ready`, `/readyz`, `/ping` and `/q/*` need no key; every other endpoint above does.

#### Development endpoints

vLLM registers its development and RL control routes only when `VLLM_SERVER_DEV_MODE` is set. This server gates the same routes behind one switch, off by default:

- `--dev-mode` sets `neuralfree.server.dev.mode=true`; `-Dneuralfree.server.dev.mode=true` works where flags do not.
- vLLM's `VLLM_SERVER_DEV_MODE` is honoured when the property is unset: `0` turns the routes off, any other integer turns them on.
- With the switch on, the server logs vLLM's warning once at startup: `SECURITY WARNING: Development endpoints are enabled! This should NOT be used in production!`
- With it off, every gated route answers 404 with the error envelope (type `NotFoundError`) naming the switch, exactly as an unregistered vLLM route does, and is not listed anywhere.

Gated: `GET /server_info`, `POST /collective_rpc`, `POST /reset_prefix_cache`, `POST /reset_mm_cache`, `POST /reset_encoder_cache`, `POST /abort_requests`, `POST /sleep`, `POST /wake_up`, `GET /is_sleeping`, `POST /release_kv_cache_memory`, `POST /pause`, `POST /resume`, `GET /is_paused`, and the live-weight routes `POST /init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/update_weight_version`, `/shutdown_weight_transfer_engine`, `GET /weight_info`.

Not gated by the development switch, because vLLM serves them outside its dev mode: `GET /version`, `GET /load`, `GET /tokenizer_info`, the probes (`/health`, `/ready`, `/readyz`, `/ping`), the profile window and the checkpoint window. The profile window has its own gate: `/start_profile` and `/stop_profile` answer 404 until `--profiler-config` names a profiler. The checkpoint window has its own gate too: `/checkpoint/prepare|restore|status` answer 404 until `--enable-sleep-mode` is on. llama.cpp's own gates are unchanged: the slot save/restore actions need `--slot-save-path`, and `GET /props`, `/metrics` and the completion routes stay on.

In router mode the gate runs before the model query, so a gated route answers 404 with `--dev-mode` off whether or not `?model=` is given; with the switch on it follows the model like every other routed route.

### Resumable streams

A streamed `/v1/completions` or `/v1/chat/completions` request that carries an `X-Conversation-Id` header records
its server-sent-event bytes under that id, as llama.cpp's resumable streams do. The generation then keeps running
when the client loses its connection, and the client reattaches to it:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -H 'X-Conversation-Id: chat-17' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "stream": true, "max_tokens": 200}'
# the connection drops; reattach from byte 0, or from any offset the server still holds
curl -N 'http://127.0.0.1:8080/v1/stream?conv_id=chat-17&from=0'
```

- `GET /v1/stream?conv_id=<id>&from=N` replays the recorded bytes from byte offset `N` (0 without it), then streams
  the live bytes until the generation finishes. An unknown id answers 404, a missing one 400, and a `from` below what
  the buffer still holds 400 with a restart hint.
- `DELETE /v1/stream?conv_id=<id>` is the explicit stop: it cancels the generation, drops the recorded bytes and
  answers 204. Repeating it is harmless.
- `POST /v1/streams/lookup` with `{"conversation_ids": ["chat-17"]}` answers an array of
  `{conversation_id, is_done, total_bytes, started_at, completed_at}` for each held session matching an id, the exact
  id or any `<id>::<model>` variant. Ids the server does not hold are absent, so the route never lists sessions the
  caller did not name.

One conversation holds one live stream: a new stream on the same id replaces, and cancels, the previous one. A session
keeps up to 4 MiB and is dropped 300 s after it finishes. The id travels in the `conv_id` query string rather than the
path because it may embed a model name containing slashes. Without the header a stream behaves as before: a client
that goes away cancels its generation.

The conversation id is the only handle on a session, so anyone who can reach the server, holding an API key when keys
are configured, can replay or stop a stream whose id they know. Use unguessable ids, such as random UUIDs.

### Request tracing

Start the server with `--otlp-traces-endpoint` and every generation request — chat completions, completions, responses, Anthropic messages, batch items — emits one `llm_request` span to that OTLP endpoint when the request finishes:

```sh
./gradlew :server:run --args='-m model.gguf --otlp-traces-endpoint http://collector.example:4318'
```

The equivalent system property works where flags do not reach (containers, `MANIFEST` launches):

```sh
java -Dneuralfree.server.otlp.traces.endpoint=http://collector.example:4318 -jar server.jar -m model.gguf
```

The flag/property enables the embedded OpenTelemetry SDK (`quarkus.otel.sdk.disabled=false`, otherwise left off so tracing costs nothing) and points its OTLP exporter at the value. The export protocol follows the standard environment variable — `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (`grpc`, the default, or `http/protobuf`); the standard `OTEL_EXPORTER_OTLP_*` variables configure headers, compression and timeouts as usual.

Spans join the caller's trace: a request carrying W3C `traceparent`/`tracestate` headers produces a span that is that trace's child, so a traced frontend shows its LLM calls inline. Without those headers the span roots itself.

Each span carries the engine attributes (`gen_ai.*`, vLLM's set):

| Attribute | Meaning |
|---|---|
| `gen_ai.usage.prompt_tokens` | prompt tokens, refused choices included |
| `gen_ai.usage.completion_tokens` | generated tokens over all choices |
| `gen_ai.request.id` | the response's id (`chatcmpl-…`, `msg_…`, `resp_…`) |
| `gen_ai.response.model` | the served model |
| `gen_ai.request.n` | number of choices |
| `gen_ai.request.max_tokens`, `gen_ai.request.top_p`, `gen_ai.request.temperature` | set only when the request named them |
| `gen_ai.latency.time_in_queue` | arrival to first admission (seconds) |
| `gen_ai.latency.time_to_first_token` | arrival to first token |
| `gen_ai.latency.e2e` | arrival to the last choice's terminal state |
| `gen_ai.latency.time_in_model_prefill` / `_decode` / `_inference` | admission→first token, first token→end, admission→end |

A failed or refused request marks the span `ERROR` and records the exception; a served request records none. A request that arrives with trace headers while tracing is off warns once: `Received a request with trace context but tracing is disabled; start the server with --otlp-traces-endpoint to export request spans`.

### Sleep, pause and checkpoint

The server can release the model's memory while it is idle and restore it without restarting, as vLLM's sleep mode and llama.cpp's idle sleep do. Unknown levels, modes and tags, and a `clear_cache` that is not a boolean, are refused with 400 naming the field. The sleep, release and pause routes below are vLLM's development routes: they are served only with `--dev-mode` (property `neuralfree.server.dev.mode`, vLLM's `VLLM_SERVER_DEV_MODE`) and answer 404 without it. The checkpoint window is vLLM's sleep-mode route, not a development one, but it needs sleep mode itself: without `--enable-sleep-mode` it answers 404 as an unregistered vLLM route.

| Route | Effect |
|---|---|
| `POST /sleep?level=N&mode=M` | Level 0 pauses scheduling, level 1 also drops the KV cache contents, level 2 also releases the model (weights, KV cache and scratch; the mapped file closes). `mode` treats running requests: `abort` (default) cancels them, `wait` lets them finish, `keep` freezes them (at level 1 they are preempted and recompute after the wake; level 2 refuses `keep`). A sleeping engine refuses generation with 503. |
| `POST /wake_up?tags=...` | Restores a slept engine, re-opening the model from its path. `weights` or `kv` restore the memory; `scheduling` alone ends only a level-0 sleep. |
| `GET /is_sleeping` | `{"is_sleeping": true\|false}` |
| `POST /release_kv_cache_memory` | Drops the KV cache contents, every cached prefix included, of an idle engine; a busy engine gets 409. |
| `POST /pause?mode=M&clear_cache=B` | Pauses generation: `abort` (default) cancels in-flight requests, `wait` drains them, `keep` accepts and queues new requests but schedules nothing. Under `abort` and `wait`, new generation requests get 503 naming `/resume`. `clear_cache` (default true) drops the KV cache after the drain. |
| `POST /resume`, `GET /is_paused` | Serves again; reports the pause. |
| `POST /checkpoint/prepare?mode=M`, `POST /checkpoint/restore`, `GET /checkpoint/status` | A capture window: prepare drains per `mode`, releases the transient KV state and holds new work out until the restore. The status is `running`, `preparing`, `prepared`, `restoring` or `failed` (with `error`), answered with 200 once prepared, 503 before and 500 after a failure. |

Sleep mode is vLLM's `--enable-sleep-mode` (property `neuralfree.server.sleep.mode`, off by default). Without it, `/checkpoint/prepare`, `/checkpoint/restore` and `/checkpoint/status` answer 404 as an unregistered vLLM route, and `/sleep`, `/wake_up` and `/release_kv_cache_memory` answer 400 naming the flag; the dev-mode switch is still required for the latter three. `/is_sleeping`, `/pause`, `/resume` and `/is_paused` are not sleep-mode operations and keep their dev-mode gate alone. llama.cpp's idle sleep (`--sleep-idle-seconds`) is a serving option of its own and does not need the switch.

`-Dneuralfree.server.sleep.idle.seconds=N` (llama.cpp's `--sleep-idle-seconds`; negative, the default, never sleeps) releases the whole model after `N` seconds without work. The next request, whether generation, tokenize or embedding, wakes it and is served once the model is back. `GET /props` and `GET /health` answer while it sleeps and report `"is_sleeping": true`. An explicit `/sleep` does not wake on requests; end it with `/wake_up`. A wake starts with an empty KV cache, so cached prefixes are computed again.

## Anthropic Messages

`server/` also serves Anthropic's Messages API (`AnthropicResource`, llama.cpp's `post_anthropic_messages`). `POST /v1/messages` renders the `system` prompt (string or text blocks) and the messages through the model's chat template: `user` content takes text and `tool_result` blocks (which become tool messages ahead of the user's text), `assistant` content takes text, `thinking` and `tool_use` blocks. `tools` (`name`, `description`, `input_schema`) and `tool_choice` (`auto`, `any`, `tool` with its `name`, `none`; `disable_parallel_tool_use`) route like the OpenAI tool choices; Anthropic's server tools (a `type` such as `web_search_20250305`) refuse by name. `thinking` (`{"type": "enabled"}` or `"disabled"`) sets the template's thinking switch. `max_tokens` (required, as Anthropic documents), `temperature`, `top_p`, `top_k` and `stop_sequences` apply.

The answer is the Anthropic body: `thinking`, `text` and `tool_use` content blocks (`id`, `name`, `input`), `stop_reason` `end_turn`/`max_tokens`/`stop_sequence` (with the matched `stop_sequence` named)/`tool_use`, and `usage`: `cache_read_input_tokens` counts the prompt tokens the prefix cache served (the OpenAI routes' `prompt_tokens_details.cached_tokens`), `input_tokens` the rest of the prompt, `cache_creation_input_tokens` is 0 (the prefix cache stores blocks automatically, with no explicit write), and `output_tokens`. A stream sends `message_start` once the prompt has been evaluated, so its usage holds the cache read, and `message_delta` carries the final usage in full. `cache_control` on content blocks, system blocks and tools accepts `{"type": "ephemeral"}` with an optional `ttl` of `"5m"` or `"1h"` and has no further effect; any other value is refused by name. With `stream: true` the stream carries Anthropic's event sequence: `message_start`, then per block `content_block_start`, `content_block_delta` (`thinking_delta`, `text_delta` or `input_json_delta`) and `content_block_stop`, then `message_delta` and `message_stop`. `POST /v1/messages/count_tokens` returns the rendered prompt's token count. Image and document blocks refuse by name.

## Reasoning and tool-call parsers

The engine splits reasoning from content with a reasoning parser and reads tool calls with a tool parser; both default to `auto`, which derives them from the chat template, and both take vLLM's parser names (case-insensitive, `-` and `_` alike).

| Property | Flag | Values |
|----------|------|--------|
| `neuralfree.reasoning.parser` | `--reasoning-parser` | `auto`, `none`, or a reasoning parser name |
| `neuralfree.tool.parser` | `--tool-call-parser` (server) | `auto`, `none`, `autoparser` ([template-derived](#template-derived-parsing-the-autoparser)), or a tool parser name |

The server takes vLLM's `--enable-auto-tool-choice` beside them; as in vLLM it needs `--tool-call-parser`, and the launch is refused without it. An unknown parser name is refused naming the flag.

Reasoning parsers (aliases in parentheses): `deepseek_r1` (`step5`), `deepseek_v3` (`deepseek_v31`), `deepseek_v4` (`deepseek_v41`, `deepseek_v32`), `qwen3` (`qwen3_5`, `qwen3_coder`), `glm45` (`glm47`, `glm4_moe`), `kimi_k2`, `kimi_k3`, `k2_horizon`, `minimax_m2`, `minimax_m2_append_think`, `minimax_m3`, `ernie45`, `hunyuan_a13b`, `hy_v3`, `hy_v4`, `olmo3`, `step3` (`step3p5`), `nemotron_v3`, `granite_thinking_parser`, `minicpmv`, `plamo3`, `inkling`, `mimo`, `poolside_v1`, `holo2`, `ling3`, `seed_oss`, `mistral`, `gemma4`, `granite`, `cohere_command3` (`cohere_command4`), `openai_gptoss` (`gpt_oss`, `harmony`), `muse` (`muse_glimmer`, `musespark1x`) and `none`.

Some reasoning parsers read the request's chat template variables, as vLLM's do. The variables are the top-level `enable_thinking` and `reasoning_effort`, with `chat_template_kwargs` taking precedence over them, as in the template:

| Parser | Request switch | Behavior |
|---|---|---|
| `hy_v3`, `hy_v4` | `reasoning_effort` (unset: `no_think` for `hy_v3`, `high` for `hy_v4`) | `no_think` makes everything content. Otherwise the reasoning runs from the output's start to `</think{S}>`, where `S` is the checkpoint's structural-token suffix read from the vocabulary (`<think:6124c78e>`). |
| `nemotron_v3`, `granite_thinking_parser` | `enable_thinking` (default true), `force_nonempty_content` | Qwen3 tags; a `<tool_call>` also ends the reasoning, and a stray `</think>` is dropped. With thinking on, the reasoning runs from the output's start. With `enable_thinking: false` no reasoning is parsed at all: the whole output is content, and a `<think>` in it stays text. With `force_nonempty_content: true`, an answer without content gets its reasoning again as the content. |
| `minicpmv` | `enable_thinking` (default true) | Qwen3 tags, with `<|im_end|>` and the reserved markers 12–15 removed. The tokenizer's `\n` escapes are recovered as newlines, except in code, LaTeX and tool calls. With `enable_thinking: false`, a `<think>...</think>` block is discarded. |
| `plamo3` | `enable_thinking` (default true) | `<|plamo:begin_think:plamo|>` / `<|plamo:end_think:plamo|>`, with the reasoning running from the output's start. The end of turn and the tool-request section also end it, and a later start tag opens a new block. `enable_thinking: false` makes everything content. |

A wrongly typed switch, such as a non-boolean `enable_thinking` or a non-string `reasoning_effort`, is a 400 naming the variable. A parser that opens reasoning at the output's start still starts in the content when the rendered prompt ends with a closed block (a template's empty `<think></think>`).

`inkling` parses Inkling's typed blocks:
- a `<|content_thinking|>` block is reasoning;
- text, tool-text and tool-error blocks are content, without their markers;
- the `<|message_model|>` header, including an optional function name, is dropped;
- a `<|content_invoke_tool_json|>` block passes to the content whole, for the `inkling` tool parser.

`minimax_m2_append_think` keeps the whole output as content, prefixed with the `<think>` the prompt opened.

With `auto`, Hunyuan templates (`'<think{}>'.format(HYTK)`) select their suffixed think tags, PLaMo 3 templates select the PLaMo tags, and Inkling templates select the block layout. An auto-detected `hy_v3`/`hy_v4` tool parser also reads the template's `HYTK` suffix. Selected by name (`neuralfree.tool.parser=hy_v3`), they read the Hunyuan structural-token suffix from the model's vocabulary, as under `auto`.

`plamo3`: a control marker that the end of generation cut off is dropped from the content. Examples are `<|plamo:end_` and any
other prefix of a `<|plamo:` marker.

Tool parsers: `hermes` (`qwen2_5`, `qwen3_json`, `granite4`), `ernie45`, `longcat`, `qwen3_coder` (`qwen3_xml`, `qwen3`, `qwen3_5`, `nemotron_v3`, `step3p5`, `step5`), `mimo` (the qwen3-coder syntax with every value kept as its text), `seed_oss`, `minimax_m2` (`minimax`), `minimax_m3`, `glm45` (`glm47`, `glm4_moe`, `ling3`, `poolside_v1`), `llama3_json` (`llama4_json`), `mistral` (`ministral3`), `deepseek_v3` (`deepseek_r1`), `deepseek_v31`, `deepseek_v32`, `deepseek_v4`, `deepseek_v41`, `kimi_k2`, `kimi_k3`, `k2_horizon`, `hy_v3`, `hy_v4`, `internlm` (`internlm2`), `jamba` (`apriel`, `hunyuan_a13b`), `granite` (`granite_3`), `granite-20b-fc`, `phi4_mini_json` (`firefunction`, `phi4mini`), `xlam`, `pythonic`, `llama4_pythonic`, `lfm2` (`lfm2_5`), `olmo3`, `dots`, `gigachat3`, `minicpm5`, `step3`, `plamo3`, `apertus`, `functiongemma`, `gemma4`, `inkling`, `cohere_command3`, `cohere_command4` and `functionary_v3_2`. The DeepSeek V3.2/V4/V4.1 formats read DSML blocks (`<｜DSML｜invoke name="f">` with `<｜DSML｜parameter name="k" string="true">` values, kept as text when `string="true"` and read as JSON otherwise). `hy_v3`/`hy_v4` take the Hunyuan structural-token suffix (`<tool_calls:6124c78e>` style) from the chat template: `auto` detects them with their suffix, while an explicit name selects the unsuffixed markers. `auto` detects the format from the template's call markers; a template it does not recognize parses no calls.

The tool parsers stream: text that may begin a call is held back until it either becomes a call (its name, then its argument text as it grows) or turns out to be content. An unknown name refuses at configuration with the registered names listed. More parsers plug in through `ToolParserProvider` and `ReasoningParserProvider` (`ServiceLoader`); a name that collides with a registered one refuses.

### Live weight updates

vLLM's in-process weight transfer engine: a training loop pushes new tensor contents into the served model without a reload, then labels the weights with a version. Updates write over the resident device (or CPU) buffers in place; a tensor the load did not keep resident (a streamed trunk, host-placed or host-mapped layers, MoE-cached experts) or a split or tensor-parallel model refuses by name. An update while requests run answers 503 naming `/pause`.

The routes are vLLM's development RL routes: they are served only with `--dev-mode` (property `neuralfree.server.dev.mode`, vLLM's `VLLM_SERVER_DEV_MODE`) and answer 404 without it. They work against any loaded model; examples use curl against a local server started with `--dev-mode`.

Initialize the transfer engine (the only backend is `inprocess`; the blank body is fine):

```sh
curl -X POST localhost:8080/init_weight_transfer_engine -d '{"backend": "inprocess"}'
# {"message": "Weight transfer initialized", "backend": "inprocess", "weight_version": "default"}
```

Run an update round — pause or wait for in-flight completions first, then start, apply, finish:

```sh
curl -X POST localhost:8080/start_weight_update
curl -X POST localhost:8080/update_weights -d '[{
  "name": "output.weight", "dtype": "F32", "ne": [64, 320],
  "data_b64": "<base64 little-endian payload>"
}]'
curl -X POST localhost:8080/finish_weight_update -d '{"weight_version": "rl-step-9"}'
```

`update_weights` also takes a single tensor object instead of a list, and accepts tensors between steps without a round. `dtype` is the ggml type name of the stored tensor (`F32`, `F16`, `BF16`, ...); `ne` is the ggml shape with `ne[0]` first and the payload row-major little-endian. Unknown tensors, shapes, dtypes or fields are refused with HTTP 400 naming the offender.

`update_weights` takes vLLM's embedded wire form too: the tensor object or list under `update_info` (`{"update_info": [...]}`). A body that names any member beside `update_info` in that wrapper is a 400 naming it.

Label the served weights and read them back:

```sh
curl -X POST localhost:8080/update_weight_version -d '{"new_version": "rl-step-10"}'
curl localhost:8080/weight_info   # {"weight_version": "rl-step-10", "backend": "inprocess", "initialized": true}
```

Shut the transfer engine down when a training loop ends (`shutdown_weight_transfer_engine`, idempotent). The version label survives shutdown and re-init; a level-2 sleep releases the weights, so initialize the transfer engine again after waking.

Only the `inprocess` backend is served (not vLLM's NCCL, IPC or other cross-process transports), payloads travel as JSON under the server's ~16 MiB parse limit (send a whole model tensor by tensor), and draft-model updates and `/weight_checker` are not served.

#### Collective RPC

`POST /collective_rpc` is vLLM's development RPC route: one call drives an engine method on every replica and
answers one result per replica, in pool order. It is a development route like the weight routes and answers 404
without `--dev-mode`.

The body carries the `method` (required), its positional `args` and keyword `kwargs` (strings only, as vLLM's route
accepts), and an optional `timeout` in seconds that bounds each replica's wait. A payload travels as one positional
argument or the named keyword, not both.

| Method | Payload | Result per replica |
|---|---|---|
| `init_weight_transfer_engine` | optional `init_info`: a serialized init-info object (`{"backend": "inprocess"}`) or a bare backend name | null |
| `shutdown_weight_transfer_engine` | none | null |
| `start_weight_update` | none | null |
| `update_weights` | `update_info`: the serialized tensor object or list (the same JSON as the `/update_weights` body) | null |
| `finish_weight_update` | optional `weight_version`: a bare version or `{"weight_version": ...}` | null |
| `update_weight_version` | required `new_version`: a bare version or `{"new_version": ...}` | null |
| `get_weight_version` | none | the version that replica reports |
| `reset_prefix_cache` | none | whether that replica cleared its cache |

```sh
curl -X POST localhost:8080/collective_rpc -d '{"method": "init_weight_transfer_engine", "args": ["inprocess"]}'
curl -X POST localhost:8080/collective_rpc -d '{"method": "start_weight_update"}'
curl -X POST localhost:8080/collective_rpc -d '{"method": "update_weights", "kwargs": {"update_info":
  "[{\"name\": \"output.weight\", \"dtype\": \"F32\", \"ne\": [64, 320], \"data_b64\": \"...\"}]"}}'
curl -X POST localhost:8080/collective_rpc -d '{"method": "finish_weight_update", "kwargs": {"weight_version": "rl-step-9"}}'
curl -X POST localhost:8080/collective_rpc -d '{"method": "get_weight_version"}'
# {"results": ["rl-step-9"]}
```

An unknown method is a 400 listing the supported set; an argument the method does not take is a 400 naming it. An
uninitialized transfer engine or a replica running requests is a 503 (the latter naming `/pause`), and a placement
that cannot take an update is a 409. `get_weight_version` reports each replica's own value (a replica without an
initialized updater reports the pool's committed label), and `reset_prefix_cache` each replica's cleared flag. In
router mode the call takes `?model=` (and `?autoload=`), like the weight routes.

### Template-derived parsing (the autoparser)

A chat template whose tool-call syntax no hand-written parser recognizes is read by the **autoparser**. It derives
the format from the template itself, as llama.cpp's differential autoparser does: it renders probe conversations
with the template, diffs the outputs, and reads the role markers, the reasoning block, the content wrapper and every
tool-call marker (section, per-call, function, argument and call-id) out of the differences.

Selection:

- `-Dneuralfree.tool.parser=autoparser` selects it explicitly. It needs a
  loaded model and refuses by name when the template does not parse or renders no tool-call syntax.
- `auto` keeps the hand-written parsers first and offers the template to the autoparser only when none of them
  matches, so models with a fixed parser keep it.
- With `--reasoning-parser auto`, a template whose reasoning markers the built-in detection does not know gets the
  derived reasoning tags, including a delimiter-only format whose reasoning opens at the first token. A detected
  tool-call opener interrupts an unfinished reasoning block.

The same derived definition drives generation. `tool_choice: required` or a named function constrains the output to
a call of the declared tools in the template's own markers, each tool's parameters held to its JSON schema; under
`auto`, a lazy grammar can start at the call opener when the strictness asks for one. The parser reads calls back
with the same definition, strips the content wrapper when the template has one, keeps a call id the template carries
between the function name and the arguments, and streams partial calls with argument text that is always a prefix of
the final arguments.

- **Structured output.** A `response_format` (`json_schema` or `json_object`) on a request whose template the
  autoparser reads constrains the reply to the schema's value, bare or inside a `` ```json `` fence, as llama.cpp
  builds it; reasoning stays outside the constraint.
- **Continuing the last message.** A continued final assistant message renders as the conversation before it,
  the generation prompt cut at the derived reasoning opener, the reasoning text, then (unless only the reasoning
  is continued) the reasoning closer and the content. A per-request `chat_template` override is analysed on its own.
  Messages with media parts fall back to the template's own render.
- **Marker tokens.** Every derived marker that is one vocabulary token is kept whole by the tokenizer when special
  tokens are parsed, so the markers the template writes come back as the same single tokens.
- **Template workarounds.** The fixes llama.cpp's autoparser applies to known templates are applied with the same
  trigger fragments: old Qwen/DeepSeek thinking, Granite 3.3, Cohere Command R+, Functionary, DeepSeek-R1-Distill,
  Nemotron Nano v2, Fireworks v2, Solar Open, Apriel 1.6, the JSON `{name, parameters}` OpenAI wrapper, Laguna
  (with its `</assistant>` stop) and Bailing V3. Call ids are read and written before the function name and after
  the arguments as the template places them.

Analyses are cached per template source. The analysis is best-effort: a probe a template fails leaves that
structure empty rather than failing the request.

### Content fallback (`nemotron_v3`, `granite_thinking_parser`)

The swap happens with `force_nonempty_content: true` (vLLM `_should_force_content`). With `enable_thinking: false`,
the parser reads no reasoning, so the whole output is already content.

- **Streaming:** the reasoning streams as reasoning, then the same text is released again as content.
- **Non-streamed:** in a chat completion, Anthropic message, Responses output or console transcript, the reasoning is
  the `content` only, with no `reasoning_content` and no reasoning item.
- **`granite_thinking_parser`:** swaps only an answer with no content at all, and drops the newlines the content starts
  with.

### Request template variables

`GenerationRequest.Builder#templateVariables(Map)` sets the chat template variables the reasoning parser picks its
per-request configuration from: `enable_thinking`, `reasoning_effort` and any `chat_template_kwargs` entry. The map is
copied, and empty names are refused. `GenerationLoop` and `SpeculativeDecoder` resolve the configured filter for them.

The CLI renders its chat prompts with the variables below, and passes them on the request:
- `--thinking` sets `enable_thinking: true`;
- `--reasoning-effort` sets `reasoning_effort`;
- `--chat-template-kwargs` sets each member of its JSON object, over `--thinking`.

### `hf` / `beam` parsers (vLLM response templates)

**Selecting them.** Set `neuralfree.reasoning.parser` and/or `neuralfree.tool.parser` to `hf` or `beam`. One parser
reads the reasoning and the tool calls alike. Beside it, the other parser must be unset (`auto`, `none`, `identity`)
or the same name. Any other combination is refused by `EngineConfig` with "... cannot be combined", and at request
routing.

With the tool parser `hf`/`beam` and the reasoning parser `auto` (the default), no reasoning parser runs, as in vLLM:
- the template still routes the output;
- the thinking region's text streams as content, without its delimiters;
- no output constraint waits for the reasoning to end.

Set the reasoning parser to the same name to parse the reasoning. An explicit `none` keeps the raw output.

**Template source.** The template is the model's response template:
- the GGUF string key `tokenizer.response_template`, holding the template as JSON;
- or, for an HF model directory, the `response_template` object in `tokenizer_config.json`, carried under the same key.
  A non-object value there is logged and ignored.

The template is parsed and validated once per model, when the parser is built at load. Every request reuses it.

A model without a template is refused with "--reasoning-parser hf requires `response_template` metadata ...". A
template without the field a parser needs is refused too: `thinking` for the reasoning parser, `tool_calls` for the
tool parser. Invalid templates are refused with vLLM's validation messages. The server runs these checks when it opens
the model, as vLLM validates the tokenizer's template at startup, and again when a sleeping replica reopens it, so a
named parser the model cannot serve stops the launch instead of failing the first request.

**Template format.** This is vLLM's format:
- `start_anchor` / `start_anchor_pattern`, `defaults` and `fields`;
- `open` / `close` as a literal or a list of literals, or `open_pattern` / `close_pattern` as a Python regex
  (including `(?P<name>…)` and `(?<name>…)` groups, `(?P=name)`, `\g<name>` and `\g<n>` back references, and `{,}`);
- `repeats`;
- `content` parsers: `text`, `int`, `float`, `bool`, `json` (`string_delims`, `unquoted_keys`, `allow_non_json`),
  `xml-inline`, `kv-lines`;
- `transform` templates with whole-string `{placeholder}` values.

Serving keeps the `thinking`, `content` and `tool_calls` fields. It adds an implicit `content` field when every field
opens explicitly.

**Parsing.** The reasoning filter streams the thinking region as reasoning and the content region as content. It
passes a tool region whole to the content, where the tool parser reads its calls. The arguments are coerced to the
types in the tools' schemas, and a malformed or unfinished region is dropped, with vLLM's warning in the log.

`enable_thinking: false` tells an output constraint that the reasoning is over. Otherwise the constraint waits for the
template's thinking closer. A repeating or regex closer cannot be gated and is refused.

**The `beam` grammar.** `beam` holds each chat's output to a grammar derived from the template, as vLLM's
`BeamParser` does. The grammar is an xgrammar structural tag in its format form, and it replaces the request's own
output constraint:

- **Tools** (`tool_choice` auto, required or named):
  - Calls follow the template's tool-call format: its opener, the function name, and arguments in the field's
    `content` parser, written sorted by name.
  - A named choice allows only that function. A required or named choice allows no answer beside the calls.
    `parallel_tool_calls: false` allows a single call.
  - A strict tool's arguments follow its schema where the template's format can spell it. A tool is strict when it
    has `strict: true`, or always at `neuralfree.tool.strict=parameter`. Other arguments are free.
  - When the model may answer, a call may also name an undeclared function with free arguments.
  - With an automatic choice, no strict tool and the `auto` strict level, the calls are free.
- **`tool_choice: none`** beside tools, when the template has a `tool_calls` field: the answer alone, so the model
  cannot call the tools the prompt still lists.
- **An output constraint** (`response_format`, a guided grammar or regex) inside a content field the template opens
  explicitly:
  - The output is the field's opener, then the constrained answer.
  - JSON travels as `json_schema` (`json_object` as `{"type": "object"}`), a choice as its strings, and a dot-all
    regex as `regex`. Any other grammar, and JSON in the compact whitespace mode, travels as its compiled GBNF.
  - An output constraint stays the request's own and holds the output alone in three cases: the template answers
    without an opener, the constraint is a structural tag, or the constraint has a trigger or gate.
- **Otherwise** the output follows the template's framing alone: an optional thinking region, then the answer and/or
  the calls. Set `neuralfree.beam.always.constrain=false` (default `true`; vLLM's
  `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT`) to leave such chats unconstrained.

Where the grammar starts:
- **The prompt ends at the template's start anchor.** The grammar covers the output from its first token. It opens
  with non-empty reasoning unless the request sets `enable_thinking: false`.
- **Elsewhere.** The grammar follows the reasoning like any output constraint.
- **The prompt leaves the content field open.** Only the rest of a free answer is held.
- **The prompt leaves the thinking field open.** The grammar needs the reasoning gate, so it needs a reasoning parser.

How `^` and lookaheads count:
- A `^` in an opener pattern matches only at the message start.
- A final lookahead in an `xml-inline` value end reads as the next parameter's opener.
- The closer's completion and the message-start anchors use neither, as in vLLM.

**Refusals.** Each refusal names the request parameter, and the server answers with a 400. `beam` refuses a tool
guarantee only when its template cannot express it. An output constraint is never lost: without a derived grammar,
it holds the output itself.

| Request | `hf` | `beam` |
|---------|------|--------|
| Strict tools (`tools`) | Refused | Enforced; refused when the template cannot frame the calls, or beside a structured output answered without an opener |
| `tool_choice` required or named | Refused | Enforced; refused when the template cannot frame the calls |
| An explicit `parallel_tool_calls: false` | Refused | Enforced; refused when a repeating tool field cannot be framed |
| `tool_choice: none` beside tools, with a template `tool_calls` field | Accepted (no parsing) | Enforced; refused when the template cannot frame the answer |
| `response_format` with an explicit content opener | Accepted | Enforced inside the answer; a structural tag holds the output itself |
| A required or named `tool_choice` together with `response_format` | Refused (`tool_choice`) | Refused (`response_format`) |
| A tool or parameter name the template's format cannot encode | — | Refused (`tools`) |

The refusals read "response_template parsing (beam) cannot guarantee ... for this request; request parameter
'...'". Only the client's explicit `parallel_tool_calls: false` asks for the one-call guarantee, as vLLM refuses only `parallel_tool_calls is False`; a template's one-call default never refuses and keeps the first call.

## Grammar-constrained generation

`gguf/grammar` constrains sampling to a grammar: a `GbnfBuilder`/`GbnfParser` pair runs GBNF grammars (`Grammar`, `GrammarLimits`), and a `ConstraintSpec` selects the kind — `GBNF`, `JSON_SCHEMA` (compiled by `JsonSchemaToGbnf`), `JSON_OBJECT`, `REGEX` (compiled by `RegexToGbnf`), `CHOICE`, `STRUCTURAL_TAG` (vLLM's payload in either form: the legacy `structures` with `triggers`, which `StructuralTag` compiles into each structure's begin literal, schema grammar and end literal, dispatched by the triggers; or xgrammar's `format` object, which `StructuralTagFormat` compiles: `const_string`, `regex`, `grammar`, `json_schema` (with `excludes`), `any_text`, `sequence`, `or`, `optional`, `star`, `plus`, `repeat`, `tag`, `triggered_tags`, `tags_with_separator` and `dispatch`. The token-level formats, `qwen_xml_parameter` and a non-`json` `json_schema` style are refused by name) — and the constraint runs as the `GrammarStage` of the sampler chain (`Sampler.create(config, bannedTokens, extra, vocab)`), which masks every token the grammar cannot accept next and advances on each accepted token; a completed rule admits the end-of-generation tokens again. As in llama.cpp, a token matches by its piece rendered with special tokens shown, so the literal `"<|channel|>"` matches that control token, and a token reference matches by id: `<[42]>`, or `<|channel|>` written as a token name, which the batch engine resolves through the model's vocabulary (the name must tokenize to exactly one token); `!<...>` matches any token but that one.

A spec can be trigger-gated (`ConstraintSpec.withTriggers(GrammarTriggers)`, llama.cpp's lazy grammars): generation runs unconstrained until a trigger fires — a word, a regex whose first non-empty capture group (or the match) starts the constrained text, a regex the whole output must match, or a token id — and the grammar then constrains from the trigger start on, replaying the tokens that overlapped it (`TriggeredGrammarStage`). `LAZY` triggers hold the grammar to the end of generation; `DISPATCH` triggers (a structural tag's) return to free text after each completed structure and wait for the next trigger. While waiting for a word trigger, the stage masks tokens that would write a trigger's beginning followed by text the grammar refuses. The engine routes it through `BatchEngine.submit(..., constraint)` (compiled at submit, so an invalid spec throws to the caller) and the server exposes it in the llama.cpp, vLLM and OpenAI spellings ([HTTP server](#http-server)).

**Composition with the sampler chain.** The constraint head masks the candidates before every configured stage
of the sampler chain runs. It applies the grammar (JSON schema, regex, choice, structural tag) first, then the
request's `allowed_token_ids`, `bad_words` and the `min_tokens` ban, in vLLM's pre-sampling order: vLLM applies
the structured-output bitmask, `allowed_token_ids` and bad words to the logits before any sampling stage, and
llama.cpp applies its grammar first on the resample. So top-k, typical-p, top-p, min-p, XTC and top-n-sigma
operate on the masked set. At any temperature the chain keeps at least one token the grammar admits, an allowed
id survives a truncation that would otherwise cut it, and a `min_tokens` ban that would leave nothing selectable
yields so the generation can stop. The reasoning budget's end-tag force keeps the final say, applied behind the
truncating stages. This holds on every sampling path: the single-sequence loop, the batch engine's decodes, the
speculative verifiers and the `beam` parser's derived tag.

A masked set that empties ends the generation with `ConstraintDeadEndException`, and the server answers a 400,
`the constraint head admits no candidate ...`. That happens when the text so far leaves no parse any vocabulary
token continues (such as `root ::= <[999]>` beyond the vocabulary), or when the request's masks ban every id (an
`allowed_token_ids` set whose single token `bad_words` bans).

### JSON schema coverage

`response_format` (`json_schema`), `guided_json` and `structured_outputs.json` compile the schema to a grammar. On top of llama.cpp's converter, these keywords are supported:

**String formats**

| Format | Grammar |
|---|---|
| `date`, `time`, `date-time` | RFC 3339 |
| `uuid` | 8-4-4-4-12 hex |
| `email` | RFC 5321 mailbox: dot-atom or quoted local part; hostname or address-literal domain |
| `duration` | RFC 3339 Appendix A (`P1Y2M`, `PT5M`, `P3W`) |
| `ipv4` | dotted quad, no leading zeros |
| `ipv6` | RFC 3986 `IPv6address`, including `::` compression and an embedded IPv4 address |
| `hostname` | RFC 1123 labels of 1-63 characters |
| `uri`, `uri-reference` | RFC 3986 |
| `uri-template` | RFC 6570 |
| `json-pointer` | RFC 6901 |
| `relative-json-pointer` | relative JSON pointer |

Any other format is an annotation only: the value may be any string.

**Number and integer bounds**

`minimum`, `maximum`, `exclusiveMinimum` and `exclusiveMaximum` bound both `number` and `integer`.

- The exclusive keywords take a number, or the draft-4 boolean that makes `minimum`/`maximum` exclusive.
- When an inclusive and an exclusive bound are both given, the tighter one applies.
- A bounded number is written as `-?I(.F)?`: no exponent, no `-0`, and at most 16 fraction digits. It is generated within the bounds exactly.
- Integer bounds may be fractional (they round inwards) and of any magnitude.
- Bounds that no value satisfies are refused.

**Property counts**

`minProperties` and `maxProperties` count the required, optional, additional and pattern members together. A count the object cannot reach is refused.

**patternProperties**

- Keys matching a pattern take its schema. Patterns are searched, as in JSON Schema: `"id"` matches `user_id`, while `"^x-"` must match at the start.
- An object with `patternProperties` and no `additionalProperties` is closed to all other keys.
- These are refused:
  - patterns with different schemas;
  - `patternProperties` together with `additionalProperties`;
  - a declared property that also matches a pattern;
  - an edge anchor that binds only one alternative (`^a|b`).

**propertyNames**

- The schema every key satisfies. It accepts `pattern`, `format`, `minLength`/`maxLength`, `const`, `enum`, `$ref` and `anyOf` of these.
- A non-string key schema allows only the empty object.
- `propertyNames` cannot be combined with `properties` or `patternProperties`.

**Key patterns**

Key patterns (`patternProperties`, `propertyNames.pattern`) read the full regex dialect of [Regex constraints](#regex-constraints) (shorthand classes, named groups, lazy quantifiers, `{,n}`), searched rather than fully matched, and the key characters they match are written JSON-encoded. A property's `pattern` keeps llama.cpp's subset. A key pattern the dialect cannot express is refused by construct rather than relaxed.

**$ref**

References resolve inside the schema document:

- JSON pointers (`#/$defs/a~1b`, percent-encoded names);
- embedded `$id` resources, absolute or relative to the base URI in scope;
- `$anchor` names and draft-6/7 `#name` ids.

Recursive references are supported. Remote schemas are never fetched: a `$ref` to anything outside the document is refused.

**Boolean schemas**

- `true` allows any value.
- `false` removes a property, pattern or `anyOf` alternative.
- `items: false` ends an array after its `prefixItems`.
- `propertyNames: false` allows only `{}`.
- Wherever a value is still required, `false` is refused.

A schema the grammar cannot express exactly is refused with HTTP 400 and a message naming the keyword. It is never silently loosened.

### Structured-output JSON options

`structured_outputs` accepts three JSON options beside a `json` or `json_object` constraint, as vLLM does:

```json
{
  "structured_outputs": {
    "json": {"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"]},
    "disable_any_whitespace": true,
    "disable_additional_properties": true
  }
}
```

- `disable_any_whitespace` (boolean, default `false`) — write compact JSON: no spaces, newlines or tabs between
  tokens. `{"a":true}`, never `{"a": true}`.
- `whitespace_pattern` (ECMAScript regex string, default absent) — the syntactic whitespace between JSON tokens
  matches this regex instead of the default flexible rule. For example `"[ ]?"` admits at most one space at each
  token boundary and no newlines; `"[\\n\\t ]*"` admits runs of newlines, tabs and spaces. This is the pattern
  outlines and llguidance apply to their JSON regexes. When both `whitespace_pattern` and `disable_any_whitespace`
  are given, the explicit pattern wins.
- `disable_additional_properties` (boolean, default `false`) — close every schema object that declares
  `properties` or `patternProperties` and no `additionalProperties`, at any depth (including `$defs`,
  `definitions`, `allOf`/`anyOf`/`oneOf` and `items`). Values under `const`, `enum`, `default` and `examples` are
  literals, not schemas, and are never rewritten; an explicit `additionalProperties` keeps the value it was given.
  The engine already emits only the declared properties when `additionalProperties` is omitted, so on such schemas
  the option is a compatibility no-op (it matters for backends whose default is open).

The options are refused with HTTP 400 naming the field when they are given with another constraint kind (`regex`,
`choice`, `grammar`, `structural_tag`), when their value has the wrong JSON type, when `whitespace_pattern` is empty
or an invalid/unsupported regex, and when they appear without a `json` or `json_object` constraint.

Examples:

```bash
curl localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "value:", "max_tokens": 64, "temperature": 0,
  "structured_outputs": {"json": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                          "disable_any_whitespace": true}
}'
```

### Lark grammars

These fields take a grammar in either GBNF or Lark syntax:

- `grammar` (llama.cpp `/completion` and the OpenAI routes)
- `guided_grammar`
- `structured_outputs.grammar`
- the CLI's `--grammar` / `--grammar-file`

NeuralFree tells the two syntaxes apart the way vLLM does: a grammar with no line containing `::=` (outside a `#` or `//` comment) is Lark. Lark grammars compile to the engine's own grammar runtime and follow the dialect of vLLM's xgrammar backend.

A grammar that starts with llama.cpp's `%llguidance` header is Lark as well. The header is written `%llguidance {}`, optionally with llguidance's options `{"no_forcing": true}` and/or `{"allow_invalid_utf8": true}`; NeuralFree validates the declaration the way llguidance does and compiles the rules after it unchanged. This is the form llama.cpp's llguidance backend and llguidance's `gbnf_to_lark` script produce, so their output can be sent as-is. A malformed declaration is a 400 naming `%llguidance`.

In addition to `~ n` and `~ n..m`, the llguidance brace forms `{n}`, `{n,}`, `{,m}` and `{n,m}` are accepted (`/[0-9]/{0,15}`); they mean the same as the `~` forms. llguidance's nested `%lark` grammars and `%regex` substring constraints are not supported and are refused by name.

```json
{"prompt": "query:", "structured_outputs": {"grammar":
  "start: \"SELECT \" column \" FROM \" NAME\ncolumn: NAME | \"*\"\nNAME: /[a-z_]+/"}}
```

Supported:

- **Definitions.** Rules (`name: ...`, with the `?`/`!` modifiers) and terminals (`NAME: ...`). The grammar must define a `start` rule.
- **Combinators.** Alternatives `|` (a following line may continue them with `|`), aliases `-> name` (they don't change the language), groups `( )`, optional `[ ]`, and `? * +`, `~ n`, `~ n..m`. A count above the engine's repetition limit (10000) is accepted, as xgrammar accepts it: an exact `~ n` is written out copy by copy, and an over-limit upper bound in `~ n..m` falls back to unbounded.
- **Strings.** Double-quoted strings with JSON escapes. The `i` flag makes them case-insensitive for ASCII only.
- **Ranges.** `"a".."z"`.
- **Regular expressions.** `/.../` with the flags `i` (ASCII case folding), `s` (`.` matches a newline) and `u` (accepted, no effect). Without `s`, `.` matches anything but `\n`.
- **Common terminals.** `%import common.NAME [-> ALIAS]` and `%import common (A, B)` for DIGIT, HEXDIGIT, INT, SIGNED_INT, DECIMAL, FLOAT, SIGNED_FLOAT, NUMBER, SIGNED_NUMBER, ESCAPED_STRING, LETTER, LCASE_LETTER, UCASE_LETTER, WORD, CNAME, WS_INLINE, WS, CR, LF, NEWLINE, C_COMMENT, CPP_COMMENT, SH_COMMENT and SQL_COMMENT.
- **`%ignore`.** Inside rules, the ignored text may follow every token. It never splits a terminal and never comes before the first token.
- **Inline schemas.** `%json { ...schema... }` inside rules, compiled like `json_schema`.
- **Tokens.** Token sets `<[42]>`, `<[1-5,7]>` and `<[^1,2]>`, and named special tokens such as `<|im_end|>`, resolved through the model's vocabulary.
- **Comments.** They start with `//` or `#`.
- **Left recursion.** Directly and indirectly left-recursive rules (`sum: sum "+" product | product`, `a: b "x"`, `b: a "y"`) are rewritten to an equivalent form automatically. A rule whose every alternative starts with itself (directly, or through the other rules of its cycle) is accepted, as xgrammar accepts it: it matches nothing.
- **Budgets.** The rule attributes `[max_tokens=n, max_chars=m]` (`word[max_tokens=8, max_chars=20]: /[a-z]+/`) stop the rule's text once its budget is spent, as xgrammar's do. GBNF rules take the same attributes: `word[max_tokens=8, max_chars=20] ::= [a-z]+`.
- **Stops.** The rule attribute `[stop="s"]` makes the rule's text end at the stop string `s`: the compiled rule is the body with no occurrence of `s` inside, followed by `s` itself (`t[stop="!"]: /[^!]*/` matches `abc!` and refuses `abc!x`), enforced by the grammar runtime rather than a post-filter. A budget attribute on the same rule spans the body and the stop together. The body holds what a terminal holds — literals, ranges, regular expressions, terminal references, and the combinators over them; a rule reference, `%json`, tokens and the `i` regular-expression flag are refused by name, as xgrammar refuses them in its stop rules. The value is one double-quoted string: an empty stop, a second `stop`, or a non-string value is refused.

Refused with a 400 that names the construct and its line and column:

- rule or terminal priorities (`rule.2:`)
- templates and terminal intersection `&`
- attributes on terminals, and rule attributes other than `max_tokens`, `max_chars` and `stop` (`lazy` among them)
- directives other than `%import` and `%ignore`
- named grammar references `@name`
- single-quoted strings
- unknown names, duplicate definitions, and a terminal that references a rule or itself
- `%json` or tokens inside terminals
- the regex flags `m`, `x` and `l`, and regex constructs with no grammar equivalent (lookarounds, backreferences)
- left recursion hidden behind an optional leading item
- a stop whose body cannot avoid the stop string (the rule would match nothing)

In a structural tag (`response_format.type: "structural_tag"`), a `{"type": "grammar"}` node compiles EBNF by default. Add `"syntax": "lark"` to give it a Lark grammar: the whole dialect above, including the rule attributes `[max_tokens=n]`, `[max_chars=m]` and `[stop="s"]`.

### Regex constraints

`guided_regex`, `structured_outputs.regex` and `ConstraintSpec.regex(pattern, dotAll)` constrain the output to the strings the whole regex matches. This is vLLM's full-match semantics: `\d+` accepts `123` but not `a123`, and the pattern needs no `^…$`. The syntax is ECMAScript, plus the Python `re` forms vLLM clients commonly send:

| Construct | Supported forms |
|---|---|
| Literals and escapes | plain characters; `\t \n \r \f \v \0`; `\xHH`; `\uHHHH` (a surrogate pair is one code point); `\u{H…}`; `\UHHHHHHHH`; `\cX`; any escaped punctuation (`\. \/ \- \$ \\ …`) |
| Shorthand classes | `\d` = `[0-9]`, `\w` = `[A-Za-z0-9_]`, `\s` = ECMAScript whitespace and line terminators; `\D \W \S` are their complements; all six also work inside `[…]` |
| Character classes | `[abc]`, `[a-z]`, `[^…]`, `[\d\s-]`, `[a\D]`, `[\b]` (backspace) |
| Any character | `.` (any character except CR and LF; dotAll makes it match everything) |
| Groups | `( … )`, `(?: … )`, `(?<name> … )`, `(?P<name> … )` (groups only group; captures are not reported) |
| Alternation | `a|b`, empty alternatives |
| Quantifiers | `* + ?`, `{m}`, `{m,}`, `{m,n}`, `{,n}`, and their lazy forms (`*?`, `{m,n}?`, …), which accept the same strings |
| Anchors | `^` at the start and `$` at the end of a top-level alternative (`^yes$|^no$`) |

A `{` that starts no quantifier, or a lone `]` or `}`, matches itself. These are refused with HTTP 400, and the message names the regex constraint and the construct (for example `regex constraint \bfoo uses an unsupported construct: word boundary assertion: \b`):

- lookahead and lookbehind assertions;
- backreferences (`\1`, `\k<n>`, `(?P=n)`);
- word boundaries `\b` and `\B`;
- inline flag groups (`(?i)`), atomic groups and comment groups;
- possessive quantifiers;
- Unicode property escapes (`\p{…}`);
- octal escapes and unknown letter escapes (`\q`);
- anchors inside the pattern;
- a `]` first in a class (write it as `\]`).

A malformed pattern is also a 400 (`… is not a valid regex: unbalanced parentheses`).

A JSON-schema `pattern` keeps llama.cpp's converter behavior: it must be anchored `^…$` and stay within llama.cpp's subset. A pattern outside the subset (for example one that uses `\w`) makes that property accept any string, and the conversion records a warning.

### Structural tags: the xgrammar `format` object

`structural_tag` constraints accept both xgrammar shapes: the legacy `{"structures": [...], "triggers": [...]}` and the modern format object. You can send either through `guided_structural_tag`, `structural_tag`, `structured_outputs.structural_tag`, or `response_format: {"type": "structural_tag", ...}`.

```json
{"response_format": {"type": "structural_tag", "format": {
  "type": "sequence", "elements": [
    {"type": "tag", "begin": "", "content": {"type": "any_text"}, "end": "</think>"},
    {"type": "triggered_tags", "triggers": ["<tool_call>"], "excludes": ["<think>"],
     "tags": [{"begin": "<tool_call>", "content": {"type": "json_schema", "json_schema": {"type": "object"}},
               "end": "</tool_call>"}]}]}}}
```

Supported format types:

| Type | Matches |
|---|---|
| `const_string` (`value`) | exactly the string |
| `json_schema` (`json_schema`: object or `true`; `style`, `any_order`, `max_whitespace_cnt`) | JSON for the schema, in the request's whitespace mode, or the XML parameter style below |
| `any_text` (`excludes`, `max_chars`, `max_tokens`) | any text holding none of the excluded strings, within its budgets |
| `grammar` (`grammar`, `syntax`: `ebnf` (default) or `lark`) | the grammar |
| `regex` (`pattern`) | the regex |
| `token` (`token`: id or token string) | exactly that token |
| `exclude_token` / `any_tokens` (`exclude_tokens`, may be empty; `any_tokens.max_tokens`) | one / any number of tokens outside the set |
| `sequence` / `or` (`elements`) | the elements in order / any one element |
| `tag` (`begin`, `content`, `end`) | begin, content, end; `begin` may be a token object; `end` may be a string, a list of strings, or a token object |
| `optional`, `plus`, `star` (`content`) | 0–1, 1+, 0+ repetitions |
| `repeat` (`min`, `max`; `-1` = unbounded) | between min and max repetitions |
| `tags_with_separator` (`tags`, `separator`, `at_least_one`, `stop_after_first`) | tags joined by the separator, no other text |
| `triggered_tags` (`triggers`, `tags`, `at_least_one`, `stop_after_first`, `excludes`) | free text; a trigger opens the tag whose `begin` it prefixes |
| `dispatch` (`rules`: `[[pattern, format], ...]`, `loop`, `excludes`) | free text; a pattern is followed by its format |
| `token_triggered_tags` / `token_dispatch` | the same, triggered by token ids (`trigger_tokens`, `exclude_tokens`) |

Behavior notes:
- A node without `type` infers it from its members.
- Token strings (`"<|tool|>"`) resolve through the model vocabulary to exactly one token.
- Free text never writes the `excludes` strings. Inside a `tag`, an `any_text` or `triggered_tags` also never writes the tag's end strings.
- A top-level dispatching format runs as a lazy grammar: free text is unconstrained until a trigger fires. When nested, or with `at_least_one`, it is part of the grammar itself.

- A dispatching format nested in another fires at the earliest trigger occurrence, as xgrammar does: with trigger `aba`, `ababa…` dispatches at the first `aba`.

**`json_schema` styles.** The default `"style": "json"` is JSON. The XML styles write the top-level object as tagged parameters, as the model's chat template does:

| Style | Parameter |
|---|---|
| `qwen_xml` | `<parameter=NAME>VALUE</parameter>` |
| `minimax_xml` | `<parameter name="NAME">VALUE</parameter>` |
| `deepseek_xml` | `<｜DSML｜parameter name="NAME" string="true\|false">VALUE</｜DSML｜parameter>` |
| `deepseek_v4_1_xml` | as `deepseek_xml`, with `<｜DSML｜ parameter` |
| `glm_xml` | `<arg_key>NAME</arg_key><arg_value>VALUE</arg_value>` |

- Properties come in the order the schema declares them, and required ones always appear. Whitespace may precede each parameter.
- A plain string value is raw text up to the closing tag. A string with `pattern`, `format`, length bounds, `const` or `enum` is its raw, unquoted text. Every other value is JSON.
- `additionalProperties` keys match `[a-zA-Z_][a-zA-Z0-9_]*`.
- `{"type": "qwen_xml_parameter", "json_schema": ...}` is the `qwen_xml` style.

**`any_order: true`** lets an object's properties come in any order, each at most once; `minProperties` / `maxProperties` and the required properties still bound the count. **`max_whitespace_cnt: n`** (1–10000) bounds every whitespace run of the schema to `n` characters.

The following are refused with a 400 that names the field:
- the styles `minimax_m3_xml`, `cohere_xml` and `kimi_k3_xml`, and `patternProperties` / `propertyNames` on the top-level object of an XML style (parameter names are raw text no JSON key pattern describes)
- `json_schema` with `excludes`
- a tag that no trigger opens, or that several triggers open

The Responses API takes the same payload as `text.format` of type `structural_tag`: `{"text": {"format": {"type": "structural_tag", "format": {...}}}}`.

GBNF grammars also accept token sets: `<[1],[2]>` matches token 1 or 2, `!<[1],[2]>` matches any token except 1 and 2.

## Prefix cache

NeuralFree reuses the KV rows of prompts it has already computed. Reused rows were computed from the same tokens, so the output is identical to an uncached run (`BatchPrefixCacheTest`).

- **Batch engine** (serving: `neuralfree.parallel` above 1 or `neuralfree.kv.paged=true`).
  - The prefix cache is a block pool, as in vLLM. A block is one KV page of `neuralfree.kv.page.size` positions (default 16), keyed by a hash chain over its parent's hash and its rows.
  - The block hashes are vLLM's `hash_block_tokens` values, computed with the algorithm `neuralfree.prefix.hash.algo` selects (server: `--prefix-caching-hash-algo`, as in vLLM):
    - `sha256` (the default) over vLLM's pickle serialization, or `sha256_cbor` over canonical CBOR, the cross-language form;
    - `xxhash` and `xxhash_cbor`: XXH3's 128-bit digest over the same two serializations.

    A chain's first block hashes vLLM's `NONE_HASH` seed: the `PYTHONHASHSEED` environment variable when set, otherwise vLLM's fixed `vllm-none-hash` for the SHA-256 forms and fresh random bytes for the XXH3 ones (with a warning; set `PYTHONHASHSEED` to a shared value to reuse the cache across instances). The first block's extra keys hash in vLLM's tagged shapes (`("lora", identity)`, `("mm", identifier, offset)`, `("cache_salt", salt)`, `("prompt_embeds", digest)`), so a router that recomputes vLLM's hash from a block's token ids and published extra keys gets the published value.
  - A text row keys by its token. A media row (image or audio through the mmproj) keys by its item's key (the part's `uuid`, otherwise the SHA-256 of its bytes, with `--media-io-kwargs` as hash factors; see [Media cache keys](#media-cache-keys-uuid-and-the-encoded-media-cache)) together with its run index, layout, row index and mask rule, as vLLM's `mm_hash` extra keys do. So the same media at the same positions hits the cache and other media never does. An `image_embeds` / `audio_embeds` part is one item keyed by a SHA-256 over its rows, and a `prompt_embeds` row keys by the SHA-256 of its float32 bytes.
  - A reuse never ends inside a media item: a partly matched item prefills whole. A prompt that differs only from an image on still shares the text before it, and a text-only prompt shares the text prefix of a media prompt. An item whose rows a hit covers is not encoded when the encoder cache no longer holds it but its layout is known.
  - Running requests add their full blocks as they compute them, prompt and answer alike. A finished or preempted request also leaves its trailing partial block.
  - A new prompt walks the chain over whole blocks, whichever earlier requests computed them. It then shares the longest token prefix of one more cached block; that page is copied on the first write.
  - So a prompt reuses any cached prefix at token granularity, including a conversation's next turn, which reuses the previous answer.
  - While an earlier request with the same prompt still runs, the new one forks off its committed pages instead, if that reuses more. Concurrent requests with a common prompt, including the `n` choices of one request, prefill it once. Forks share text, media and prompt-embedding prompts alike; a fork never ends inside a media item. Over a memory with context checkpoints nothing forks off a running request.
  - A preempted request adopts the cached blocks of its own history when it is re-admitted, so its recompute replays only the rest.
  - The last prompt token always prefills, and `BatchSequence.reusedPrefix()` reports the reused token count.
- **Single-sequence loop** (CLI, GPU backends). A generation keeps the longest common prefix of the stored sequence and the new prompt, short of the last prompt token, when the memory can drop the rest, as llama.cpp checks before reusing a cached prefix; otherwise it prefills from scratch.
- **Speculative decoding.** The target keeps its reusable prefix as above, and the drafter realigns with the target's stored tokens at its next draft; the output is unchanged, and only the acceptance rate can dip on the first generation after a switch.
- **Memory.**
  - Cached blocks hold KV pages and count toward `neuralfree_kv_cache_usage`.
  - Blocks no running request uses are free. Under memory pressure they are evicted least recently used first, a history's tail before its prefix, and only then does preemption run.
  - `neuralfree.prefix.cache.budget` caps the tokens held by free blocks (0, the default, leaves the KV pools as the only bound).
  - **Offload tiers** (llama.cpp's `--cache-ram`, vLLM's CPU and filesystem KV offloading). A block the KV pools evict moves to host RAM (`-cram N`, `neuralfree.cache.ram`), and a block the host tier's budget drops moves on to disk (`-cdisk N`, `neuralfree.cache.disk.mib`), one file per block under `--cache-disk-dir DIR` (`neuralfree.cache.disk.dir`). `N` is MiB; `0` keeps a tier off (the default) and `-1` leaves it unbounded. The disk tier needs the host tier on and its directory set.
    - A prompt walks the pools, then host RAM, then the disk. A block found in a lower tier is copied back into fresh pages before the prompt adopts it, so the output is the same as the first run's.
    - The disk directory is scanned when the engine starts, so blocks an earlier run left are served again. Each file is checked against its whole token chain, the request's extra keys (adapters, cache salt), the block layout (page size, cache types and widths) and its own name. A file from another model or layout only misses, and is left alone; a truncated or corrupted file is deleted.
    - Disk reads and writes run off the engine thread, as vLLM's filesystem tier does. A spill joins a bounded write-behind queue, so the engine never waits for the disk and a full queue drops that spill instead of stalling decoding. The block stays promotable from its in-memory host image until its file is durable, and the storage-medium `BlockStored` is published only then.
    - A prompt whose prefix sits on the disk is prefetched when it reaches the head of the admission queue: the blocks load on background workers and the request is admitted once they are ready, so a slow disk delays that request but not the running decodes. A promotion that arrives before a read finished uses what is loaded and computes the remaining rows again.
    - A failed read or write never fails a request; it only misses or drops that copy. Shutdown lets the admitted writes finish for a bounded time, then abandons the rest, so no writer outlives the engine.
    - Both tiers need the paged prefix cache over a memory whose cached blocks are a sequence's whole state (no recurrent, DSV4 compressed, sparse side or cross-attention caches), and refuse by name otherwise.
    - A slot whose blocks sit in a lower tier still saves them ([Slots](#slots)). llama.cpp's `--cache-idle-slots` has no flag: a finished request's blocks are always kept, and these tiers decide where they wait.
  - A memory reset or `POST /reset_prefix_cache` drops every cached block.
- **KV cache events.**
  - `neuralfree.kv.events` (default `none`) publishes the pool's events after every engine step as vLLM's `KVEventBatch`: `BlockStored`, `BlockRemoved` and `AllBlocksCleared`. A `BlockStored` carries:
    - the block hashes: the selected algorithm's digests, published as their unsigned leading 64 bits like vLLM's integers, or as the raw digest bytes when the environment variable `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` is `0`; and the parent block hash;
    - the token ids (media and prompt-embedding rows carry -1), the block size and the medium: `GPU` or `CPU` in the KV pools, `CPU` in the host-RAM tier, `STORAGE` in the disk tier (vLLM's names);
    - `lora_name`, the per-request adapters the blocks were computed under (null for the base model), and `lora_id`, a stable id of the adapters' content identity (vLLM's deprecated `lora_int_id`);
    - `kv_cache_spec_kind` and `kv_cache_spec_sliding_window`, the KV cache group the blocks' pages live in (`full_attention`, `sliding_window`, `sliding_window_mla`, `chunked_local_attention`, `mla_attention`). Both are omitted when the groups disagree, and the window is set only for a sliding-window kind;
    - per block, its extra keys in vLLM's order, or null: the adapters' content identity on a chain's first block computed under per-request adapters, each media item it covers as `(hash, offset)` with the item's first row relative to the block start, the cache salt on a chain's first block, and the SHA-256 of its prompt-embedding rows.
  - **Reuse reports** (vLLM's `kv_cache_report_mode`): by default (`incremental`) a `BlockStored` is published only when a block is stored. A request whose `vllm_xargs` object sets `"kv_cache_report_mode": "full"` also publishes a `BlockStored` for every whole cached block its prompt reuses from the replica's prefix cache, carrying the same hashes, tokens, extra keys, adapters and cache spec as the block's original store and starting at the chain's first block (no parent). A gateway that subscribes after the blocks were first stored learns from these events that the replica holds the prefix. As in vLLM, only the string `full` selects the mode; any other value, including a non-string, leaves the incremental default. The boundary block a cache hit ends inside is not reported, since it is not a whole cached block.
  - A partial block is stored alone, with `block_size` set to its token count.
  - Values: `zmq` publishes on vLLM's ZMQ transport with vLLM's defaults. `log` writes each batch at INFO to the logger `com.eonmux.neuralfree.kvevents`. A class name loads a `KvEventPublisher` with a public no-argument constructor. The value may also be vLLM's `KVEventsConfig` JSON itself, as `--kv-events-config` takes it. In-process code passes `KvEventPublishers.listener(...)` to `BatchEngine.forModel(model, config, publisher)`.
  - **ZMQ transport** (vLLM's `ZmqEventPublisher`, wire-compatible with vLLM's KV-aware routers such as the llm-d and Dynamo indexers):
    - The PUB socket sends each batch as `[topic, sequence, payload]`. The sequence is an 8-byte big-endian counter from 0. The payload is the batch, msgpack-encoded exactly as vLLM's msgspec encoder writes it, with block hashes as unsigned 64-bit integers.
    - `endpoint` (default `tcp://*:5557`): a wildcard host (`*`, `[::]`) binds, any other host connects. `tcp://*:0` binds an ephemeral port on the node IP (`neuralfree.host.ip` when set, the equivalent of `VLLM_HOST_IP`, else the primary interface's address, as vLLM's `get_ip` picks it) and reports the bound address as `tcp://<ip>:<port>`. A wildcard ephemeral bind without a node IP is refused by name.
    - `replay_endpoint` (default none): a ROUTER socket that serves the last `buffer_steps` (default 10000) batches. A DEALER or REQ client sends the first sequence it missed as 8 big-endian bytes. It receives `["", topic, sequence, payload]` for every buffered batch from there on, then the end marker `["", "", -1, ""]`.
    - `hwm` (default 100000; 0 means no limit) is the PUB socket's high-water mark.
    - `max_queue_size` (default 100000; 0 means no bound) caps the batches queued for the publisher thread. The engine waits while the queue is full.
    - `topic` (default empty) is the topic frame; subscribers match it as a prefix.
    - Only `tcp://host:port` endpoints are accepted. `ipc://` and `inproc://` are refused by name.
  - On the server, each data-parallel replica stamps its global rank as `data_parallel_rank`. Under `zmq`, each replica has its own publisher, with the ports of `endpoint` and `replay_endpoint` raised by its rank (port 0 is left as is), as vLLM's per-rank publishers do.
- **Metrics.** `neuralfree_prompt_tokens_cached_total` counts reused prompt tokens, and `neuralfree_prefix_cache_hit_rate` their share of all prompt tokens (vLLM's `gpu_prefix_cache_hit_rate`).

`neuralfree.prefix.reuse=false` (default `true`) turns reuse off everywhere, and the engine then caches nothing. Recurrent and sliding-window (SWA) memories reuse prefixes through [context checkpoints](#context-checkpoints-prefix-reuse-for-recurrent-and-sliding-window-models) only, since their state cannot be rebuilt from KV pages alone.

### Context shift

A decoding sequence whose next token has no position left either stops with `CONTEXT_FULL` or shifts its context: it drops the oldest positions after the kept prefix and moves every later position down, so generation continues in the same slot, as llama.cpp's server shifts its slots. The single-sequence loop and the batch engine behind the server both shift when `neuralfree.context.shift=true` (server: `--context-shift`, off with `--no-context-shift`).

The generation routes (`/v1/completions`, `/v1/chat/completions`, llama.cpp's `/completion`) take llama.cpp's counts:

- `n_keep` (default 0): the prompt tokens every shift keeps. A leading BOS is kept on top, `n_keep` never passes the `n_ctx - 4` margin, and `-1` keeps the whole prompt.
- `n_discard` (default 0): the tokens one shift discards after `n_keep`; 0 discards half of what the context holds beyond the kept tokens.

A shift sets `truncated` in the `/completion` final body, and `BatchStats.contextShifts` counts an engine's shifts. A shifted sequence's fed tokens compact with the moved rows, so a preemption's recompute replays the compacted history; its cached blocks above the kept prefix leave the prefix cache, and it no longer donates rows to a fork.

Limits, each named:

- The shift serves the text prompts of a decoder-only model. `n_keep` / `n_discard` on a media prompt, `prompt_embeds` or an encoder-decoder is refused with a 400 at submit; without the fields those sequences stop with `CONTEXT_FULL`.
- The batch engine shifts on the CPU backend and on the GPU backends (OpenCL, CUDA, Vulkan) over their paged memory. On a GPU the pages move on the device and the moved keys are re-rotated there (`LLM_KV_RopeShift_*`; quantized K goes through the kernel's dequantize, rotate and requantize path), so the shifted sequence computes what the CPU backend's shift computes. The single-sequence loop shifts on the CPU and on every GPU backend. A hybrid CPU+GPU layer split runs no paged memory and the multi-GPU layer split shifts no paged sequence, so on those loads the batch engine turns the shift off when it is built, logging why, and a full context stops with `CONTEXT_FULL`.
- A model whose layers cannot re-rotate their keys refuses the shift when the engine is built, logging why: recurrent layers (Mamba, RWKV, GDN, KDA), a K cache whose geometry the shift kernels cannot address, and the models llama.cpp refuses (step35, several positions per token). A full context then stops with `CONTEXT_FULL`.
- A DeepSeek-V4 compressed cache refuses the move (its rollback snapshots cover tail drops only), so the sequence stops with `CONTEXT_FULL`, as llama.cpp's compressor does.
- A shift whose rows another running sequence still shares fails that sequence alone when the pools cannot supply a copy page, as llama.cpp's server refuses a shared-prompt shift.

### Self-extend (group attention) and position division

A generation request may ask the engine to compress its stored positions as it grows, llama.cpp's self-extend
(`--grp-attn-n` / `--grp-attn-w`):

- `grp_attn_n` — the group-attention factor (default `1`, which turns self-extend off).
- `grp_attn_w` — the positions one group spans (default `512`); it must be a multiple of `grp_attn_n`.

The launch flags `--grp-attn-n <n>` (`-gan`) and `--grp-attn-w <n>` (`-gaw`) run self-extend on the console and set
these fields' defaults on the server; a request naming its own value overrides the server's default. Values below their
minimums are refused by name on both surfaces.

Both fields ride `/v1/completions`, `/v1/chat/completions` and llama.cpp's `/completion`. With `grp_attn_n > 1`, every
`grp_attn_w` stored positions are grouped and their positions divided by the factor (llama.cpp's
`llama_memory_seq_div`, the outer shifts being `llama_memory_seq_add`), so the sequence attends at an effective
context of `nCtx / grp_attn_n` while it keeps filling the context's rows — generation no longer stops at the position
where a plain request would reach `n_ctx`. The engine applies the group loop before each step, exactly as llama.cpp's
completion loop does; `BatchStats.grpAttnEdits` counts the compressions.

```
curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "local", "prompt": "...",
  "max_tokens": 512, "grp_attn_n": 4, "grp_attn_w": 4096
}'
```

The request is refused with a 400 naming the fields when:

- `grp_attn_w` is not a multiple of `grp_attn_n`, or either value is out of range;
- it also sets `n_keep`/`n_discard` (one sequence cannot both compress its positions and drop windows of them);
- the prompt carries media chunks or prompt embeddings, or the model is an encoder-decoder (those rows share
  M-RoPE positions or start at a fixed decoder row);
- the memory restores context checkpoints (recurrent state or a sliding window), whose state a compression would
  invalidate;
- the backend serves no per-sequence position edits (the tensor-parallel split refuses them).

Library users get the same primitives directly: `SequenceMemory.div(seq, p0, p1, d)` divides a sequence's stored
positions and re-rotates the keys by each row's own shift, `SequenceMemory.add(...)` moves them, and
`ForwardBackend.addSequence`/`divSequence` run both over a paged memory. The contiguous single-sequence memory
refuses `div` by name (`divRefusal()`), since it has no per-row positions.

### KV chunk reuse (`--cache-reuse`, `n_cache_reuse`)

The server accepts llama.cpp's `n_cache_reuse` request field and the `--cache-reuse <n>` launch flag
(`neuralfree.cache-reuse`; default 0, off). With `n_cache_reuse = n > 0`, a prompt whose cached prefix diverges
re-appends cached runs of at least `n` rows past the first mismatch instead of prefilling them: the rows are copied to
the prompt's own positions with their keys re-rotated to the new positions, and the prefill skips the re-appended rows.
Chunks never shift forward and never cover the row producing the next token.

As in llama.cpp this trades a little accuracy for the saved prefill: a re-appended row keeps the deeper-layer state its
donor context produced, which is exact where the donor prefix is the prompt's own (the reused common prefix). The reuse
is skipped, and plain prefix reuse runs instead, when the memory cannot shift KV, the engine runs without its prefix
cache, or the pages the run needs are not all free.

- Launch (server): `--cache-reuse <n>`, the default of the `n_cache_reuse` request field.
- Request: `"n_cache_reuse": <n>`, per request; 0 keeps the chunk reuse off.

### Context checkpoints (prefix reuse for recurrent and sliding-window models)

The batch engine (`neuralfree.parallel > 1` or `neuralfree.kv.paged=true`) reuses a cached prompt prefix by sharing its KV pages. Some models need more than KV pages to resume at a prefix:

- **Recurrent state** (Mamba, RWKV, and the GDN or convolution layers of hybrid models such as Qwen3-Next and LFM) cannot be rewound.
- **Sliding-window layers** (Gemma 2/3, gpt-oss, Mistral SWA) drop the rows that fall out of their window.
- **DSV4 compressed caches** (DeepSeek-V4) accumulate compressor state across positions.

For these models the prefix cache keeps **context checkpoints**: the recurrent state, the DSV4 compressor streams and rollback planes, and the window pages at an exact row count, as llama.cpp's server does with `--ctx-checkpoints`. A cached prefix is reused only up to the longest checkpoint it contains.

Checkpoints are taken:

- when a request finishes or is preempted, at the rows it computed, so the next turn of a conversation resumes right after the previous answer;
- during a prompt's prefill, text or media:
  - shortly before the prompt's end (the last page boundary before the final token), so a repeated prompt re-runs only its last few tokens;
  - every `neuralfree.checkpoint.min.step` tokens;
  - at the chat message boundaries the server names: the rendered prompt's last message start (llama.cpp's message spans), so a conversation branched at the last message restores there and re-runs only its own tail. A prompt-embedding chunk is cut at the checkpoint row; a media chunk the row falls inside runs whole with the checkpoint at its end.

| Property | CLI flag | Default | Meaning |
|---|---|---|---|
| `neuralfree.ctx.checkpoints` | `--ctx-checkpoints <n>`, `--swa-checkpoints <n>`, `-ctxcp <n>` (server and llmdemo) | `32` | Checkpoints kept per sequence slot; beyond the limit, checkpoints within `neuralfree.checkpoint.min.step` rows of the kept checkpoint before them leave first and the oldest created one last (llama.cpp's eviction). `0` turns prefix reuse off for recurrent, sliding-window and DSV4 memories. |
| `neuralfree.checkpoint.min.step` | `--checkpoint-min-step <n>`, `-cms <n>` | `8192` | Token spacing of the checkpoints a long prompt's prefill takes on its way, rounded up to a KV page. `0` sets no minimum: a checkpoint at the end of every prefill chunk. |
| `neuralfree.swa.checkpoint.reserve` | — | `0` | Pages of every sliding-window group's pool reserved for checkpoint windows (llama.cpp keeps its SWA checkpoints as host copies, which need no pool room). The reserve makes a one-slot SWA memory's mid-prefill checkpoints survive beside the running sequence. |

Examples:

```bash
# keep 8 checkpoints per slot, one every 2048 prompt tokens
java -Dneuralfree.parallel=4 -Dneuralfree.ctx.checkpoints=8 -Dneuralfree.checkpoint.min.step=2048 ...
# CLI
llmdemo --parallel 4 --ctx-checkpoints 8 --checkpoint-min-step 2048 ...
# server, with 12 reserved pages per sliding-window group for checkpoint windows
java -Dneuralfree.swa.checkpoint.reserve=12 -jar server/build/<server jar> -m model.gguf -np 4 --ctx-checkpoints 8 --checkpoint-min-step 2048
```

Notes:

- Full-attention models are unaffected: their prefix cache reuses any shared prefix as before.
- **Memory cost.**
  - A recurrent checkpoint costs one copy of a sequence's recurrent state (on the GPU a read-back of the sequence's device state rows), held in host memory.
  - A DSV4 checkpoint costs a copy of the sequence's compressed streams and compressor planes, held in host memory.
  - A sliding-window checkpoint shares pages with the window and costs no copy, but it keeps window pages that would otherwise be reused; checkpoints give their window pages back whenever a running request needs them, and `neuralfree.swa.checkpoint.reserve` keeps them alive under load.
  - A checkpoint taken at a message boundary or mid-page pins the page it ends in, so the sequence's next write copies the page (one page copy per boundary checkpoint).
- Over recurrent, sliding-window or DSV4 memories, a request does not share the prompt of a request that is still running; it reuses only cached checkpoints.
- Batched generation over a recurrent model needs one recurrent state row per sequence: the paged memory on either backend. The contiguous CPU and GPU memories refuse it by name. The GPU keeps the DSV4 prefix cache off (no host copies of the compressed streams); the paged CPU memory reuses DSV4 prefixes at checkpoints.
- Speculative decoding (`neuralfree.draft.type`) is refused over recurrent models in the batch engine, because a rejected draft cannot rewind the recurrent state.
- Checkpoints emit no KV cache events of their own: vLLM's KV cache events are `BlockStored`, `BlockRemoved` and `AllBlocksCleared`, and llama.cpp has none. A checkpoint that leaves with its block surfaces as that block's `BlockRemoved`.

## Paged KV memory

`gguf/memory/paged` keeps the rows of a KV group in reference-counted pages instead of one buffer per sequence: `PagePool`, `PageList`, `PagedSequences`, `PagedStep`, `PageCopy` and `RowParts`. It is property-gated, not the engine default: `neuralfree.kv.paged=true` forces it for a single sequence, and `neuralfree.parallel > 1` selects it; `neuralfree.kv.page.size` (default 16, a power of two in [1, 256]) sizes the pages and `neuralfree.kv.unified` (default true) shares the whole context between the sequences. Without the properties, `ModelMemory.sequences()` returns `SingleSequenceMemory`. On the CPU path, `ForwardBatch` runs a batch of sequences over the paged memory in one trunk pass: ubatches of at most `neuralfree.ubatch` rows that mix sequences (llama.cpp's `split_simple` over its unified cache, vLLM's one flattened batch per step), all-or-nothing page reservation per ubatch, K/V stored at the step's `cellIdx` rows and attention through each sequence's own page table, so N concurrent decodes cost one matmul pass. A batch returns the same logits as running its sequences one at a time ([LLM-INFERENCE.md](LLM-INFERENCE.md#paged-kv-memory) has the details).

Recurrent and hybrid models batch too. Every CPU token mixer has a multi-sequence step: softmax attention, MLA (latent and decompressed caches), Mamba-1/2 and PLaMo-2 selective scans, the Qwen3-Next and Qwen3.5 gated delta nets, KDA, lightning attention, the RWKV-6/7 time mixes and the LFM2 short convolutions. The paged memory keeps one recurrent-state row per sequence, and a slot whose sequence is evicted or preempted starts its next generation from a fresh state. Over such a memory a batch splits as llama.cpp splits it: each ubatch takes as many tokens of every sequence as fit (`split_equal`), or one sequence per ubatch when every row is an output (`split_seq`). The layer extras that transform whole rows (hyper-connections, attention-residual stacks, Gemma 3n AltUp and Laurel, per-layer embeddings including the n-gram variant, control vectors, embedding skip, alpha/beta residuals, loop norms, and limite's MUDD dense connections and learned residual scales) run on every layout. On the GPU, the Gemma 3n AltUp, Laurel and per-layer-embedding extras run batched steps as well, so Gemma 3n serves over the paged memory on both backends. A whole-memory snapshot is refused on a paged memory, because restoring it would rewind every sequence at once. The sparse mixers (DSA, MSA, QSA) batch with every sequence selecting among its own cached rows, DeepSeek-V4 batches causal steps over per-sequence compressed caches and compressor states (a sequence drops its tail only within the compressor's rollback snapshots, and copies take whole sequences), and T5 encoder-decoder models batch with one encoding per sequence, the batch engine encoding each request's prompt as it admits it; only switched-LoRA routing (its router is position-addressed) still refuses a batched step by name. On the GPU, DSA sparse attention, DeepSeek-V4 compressed attention (every sequence with its own compressed streams) and T5 encoder-decoders batch over the paged memory as well, so the batch engine serves GPU encoder-decoders; the GPU MSA and QSA mixers still run one sequence and refuse a batched step by name.

**On the GPU.** The paged memory and the batch engine run on the GPU backends too. With `neuralfree.parallel` above 1 (or `neuralfree.kv.paged=true`) on a GPU backend, the device memory holds per-group page tables. `BatchEngine` then decodes all running sequences in one GPU step, and attention dispatches the paged flash kernels: one launch per decode batch and one per multi-token prompt segment. The paged kernels run on OpenCL, CUDA and Vulkan; their arguments fit Vulkan's 128-byte push-constant block. Prefix reuse, copy-on-write pages (the copy runs on the device) and preemption work as on the CPU. The server opens its model through the same factory as the CLI, so `neuralfree.backend` (default `auto`) decides where the HTTP API runs.

A batched GPU step covers softmax attention, cross-attention, MLA, DSA and DeepSeek-V4 layers and the recurrent mixers (Mamba-1/2 selective scans, the gated delta nets, KDA, lightning attention, the RWKV-6/7 time mixes and the LFM2 short convolutions, each sequence keeping its state row in its paged slot), and serves the same requests as the CPU one:

- **Embedding rows.** `prompt_embeds`, `image_embeds` / `audio_embeds` parts and projected media rows batch together with other sequences' token rows, even within one ubatch. A media chunk's rows take their explicit positions (M-RoPE sections included), and a bidirectional media chunk runs non-causally on the standard attention layers. A batch carrying explicit M-RoPE sections on a model whose RoPE reads one section per token (anything but `MROPE`, `IMROPE` or `VISION`) is refused before either backend uploads or evaluates anything, with the same message on both: `the batch carries 4 rope sections, the model's NONE rope reads 1`.
- **Pooling and scoring.** Embeddings, pooling, rerank, scoring and classification evaluate the prompt in a sequence of its own, whose pages are freed afterwards.
- **Hidden rows and taps** follow every row of a multi-sequence batch.
- **Adapters.** Rows under different per-request adapters batch together, each applying its own slot's deltas ([Per-request LoRA adapters](#per-request-lora-adapters)).
- **Models that read token ids.**
  - Per-layer token embeddings (Gemma 3n, Gemma 4) give an embedding row the padding token's (id 0) per-layer row.
  - qwen4exp's n-gram per-layer embeddings hash an embedding row as `ple.image_token_id`, or EOS when that is 0. The row's predecessors come from the ubatch order and the sequence's recorded history, which stores the same stand-in id. The n-gram drafter still drafts from text tokens only.
  - deepseek4's routed experts send an embedding row through the router instead of the hash table (`ffn_gate_tid2eid`), with `exp_probs_b_vl` as the selection bias, or none without it. On non-hash layers `exp_probs_b_vl` replaces `exp_probs_b` when the file has it. Each row of a mixed ubatch routes by its own kind.
  - Switched-LoRA routing (granite-switch) and value embeddings (limite) refuse embedding-row inputs by name on every backend.

The MSA and QSA sparse mixers have no multi-sequence step on the device yet and refuse a batched GPU step by name. Every refusal reaches an HTTP client as a 400 naming the field and the backend.

The batch engine schedules over the shared pools as vLLM's scheduler does. A queued prompt is admitted when a slot is free and every KV group has the free pages its prompt needs beside the admitted prompts still waiting for their prefill, counted as vLLM's `allocate_slots` counts them: whole pages of a reused prefix (a prefix-cache hit, or a fork off a running request with the same prompt) cost nothing, a shared partial last page costs one copy-on-write page, and a sliding-window group counts at most its window's pages. Free cached blocks are evicted first to make room. Requests that cannot hold their pages together, such as the `n`/`best_of` candidates of a prompt that nearly fills the pool, run one after the other, the later ones reusing the earlier ones' cached prefix. Running out of pages is recoverable: a step that finds the pools full rolls its scheduled rows back, then evicts a cached prefix or preempts a running sequence (the scheduling policy picks which) before the work retries. A preempted sequence keeps its sampler, grammar and streamed text; it is re-admitted once a running sequence retires, prefills its prompt and generated tokens again, and continues, so its result is the one it would have produced alone. A request that cannot fit even with every other sequence's memory freed fails by name.

The scheduling policy (`neuralfree.scheduling.policy`, or `--scheduling-policy` on the server; vLLM's `--scheduling-policy`) decides the order:

| Policy | Admission | Preemption under memory pressure | Request `priority` |
|---|---|---|---|
| `fcfs` (default) | Arrival order; a preempted sequence comes back ahead of the queue | The youngest running sequence | `0` only; any other value is refused with a 400 naming `priority` |
| `priority` | The lowest `priority` value first, ties by arrival. Preempted sequences and queued requests compete in one order, so a waiting request with a lower value is admitted before a preempted one with a higher value. The first request in the order waits until it fits, and nothing behind it skips ahead | The running sequence with the highest `priority` value, ties broken by latest arrival | Any 32-bit integer |

For example, start the server with `-m model.gguf -np 4 --scheduling-policy priority`, then send `"priority": -1` on urgent requests and `"priority": 10` on background ones. In Java, `BatchSequence.preemptions()` reports how often a request was preempted (vLLM's `num_preemptions`).

Every step spends a token budget, mixing prefill chunks with the decodes of the sequences around them (vLLM V1's `enable_chunked_prefill` and `max_num_batched_tokens`; llama.cpp serves its slots' decode tokens and a prompt chunk from one batch the same way). The step schedules the running sequences first come, first served, in admission order: one row per decoding sequence, and the rest of the budget to prefill chunks. A sequence admitted before a long prompt keeps decoding every step, so the long prompt no longer stalls the batch for its whole prefill. Preempted sequences recompute their history through the same chunks, and the results never change: a chunked sequence samples from the same logits rows as an unchunked one.

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.chunked.prefill` | Split prompt prefills into per-step chunks beside the decodes. `false` runs the oldest waiting prefill alone, its whole prompt tail in one forward, while every decode waits | true |
| `neuralfree.scheduling.policy` | The order the batch engine admits and preempts requests in (vLLM's `--scheduling-policy`): `fcfs` serves by arrival and takes `priority` 0 only; `priority` serves the lowest `priority` value first, ties by arrival, and preempts the highest value first | fcfs |
| `neuralfree.max.batched.tokens` | The token budget one step shares between prefill chunk rows and decode rows; `0` uses `neuralfree.ubatch` | 0 |
| `neuralfree.long.prefill.tokens` | The most rows one request's prefill chunk takes per step while another request runs or waits (vLLM's `long_prefill_token_threshold`); `0` sets no cap | 0 |
| `neuralfree.max.encoder.tokens` | The media rows one step schedules across all its media prefills (vLLM's `max_num_encoder_input_tokens`, applied as its scheduler's `encoder_compute_budget`). Media chunks run whole, so a chunk that does not fit what is left waits for a later step, along with everything after it in its prompt. A chunk larger than the whole budget runs on a step with no other media and uses all of that step's budget. Text chunks take no encoder budget. `0` uses the step's token budget | 0 |
| `neuralfree.encoder.cache.rows` | Rows of encoder output the batch engine's encoder cache holds (vLLM's `encoder_cache_size` and `EncoderCacheManager`). Requests that carry the same image or audio share one copy of its embedding rows. An entry stays while any running request references it; once released it stays cached until a new medium needs the space, oldest first. A medium larger than the cache keeps its own copy. `0` uses the step's token budget; `-1` turns the cache off | 0 |

A long multimodal prompt cannot hold up the batch: each step prefills at most `neuralfree.max.encoder.tokens` media rows, and the running sequences keep decoding beside it. Results are unchanged; a prompt whose media spans several steps generates the same tokens as one prefilled in one step. Media are identified by the SHA-256 of their bytes plus the chunk's place within the medium (a sliced image's overview and slices, an audio clip's 30 s windows), so the same image sent by several concurrent requests is stored once.

For example, four decoding sequences beside a 32k-token prompt with a 2048-token budget: `-Dneuralfree.parallel=4 -Dneuralfree.max.batched.tokens=2048 -Dneuralfree.long.prefill.tokens=512`.

**Prompt ubatches of equal size.** A prompt prefills in ubatches of equal size rather than in greedy `--ubatch` chunks with a short remainder at the end. The rows up to the next forced break — the end of the prompt, a context checkpoint, a message boundary — divide into as few ubatches as `--ubatch` allows, equal up to one token: 19k prompt tokens at `-ub 8192` run as 6338 / 6338 / 6338 where greedy filling gave 8192 / 8192 / 2630, and a 45-token prompt at `-ub 8` takes 8 / 8 / 8 / 7 / 7 / 7 instead of 8 / 8 / 8 / 8 / 8 / 5. Every ubatch pays the same fixed cost — the experts a streaming model uploads or prefetches per ubatch, one queue submission — and the cost of a ubatch, `max(compute ~ tokens, upload)`, is convex in its size, so the trailing ubatch that was short but not cheaper disappears. With chunked prefill on, each step's chunk is a whole number of the run's ubatches, so no step holds a short trailing ubatch; with `neuralfree.chunked.prefill=false` the whole tail runs as one batch that splits the same way. The sizes follow `--ubatch` and `neuralfree.max.batched.tokens` as before, and a batch of several requests keeps llama.cpp's `split_simple` / `split_equal` row grouping.

## Media prompts

The batch engine accepts prompts that interleave text tokens with media embedding rows, the way llama.cpp's mtmd path feeds the decoder. The embeddings come from the caller: the output of a vision or audio encoder, projected to the model's input width `n_embd_inp`. A `MediaPrompt` is built from chunks and submitted to a `BatchEngine`:

```java
MediaPrompt prompt = MediaPrompt.builder()
    .addText(tokenizer.encode("Describe the image:", false, true))
    .addMedia(MediaRopePositions.image2d(5, 24, 24), imageRows, /* nonCausal */ false)
    .addText(tokenizer.encode(" What is in it?", false, true))
    .build();
BatchSequence handle = engine.submit(prompt, request, callback);
```

- A media chunk takes one KV row per embedding row but advances the sequence's position cursor by its `nPositions()`: `max(nx, ny)` for an image grid, one per row otherwise (llama.cpp's `mtmd_input_chunk_get_n_pos`). On an M-RoPE model the chunk's rows rotate with the grid positions (the temporal base shared, the grid coordinates in the height and width sections), and the shared base position lets every row of the chunk attend the whole chunk.
- `nonCausal = true` runs the chunk's rows without the causal mask (llama.cpp's `llama_set_causal_attn(false)`), for models whose media chunks attend bidirectionally.
- `MediaRopePositions.image2d` and `audio1d` need a multi-section RoPE model (`MROPE`/`IMROPE`, such as qwen2vl); `normal` positions work on any model.
- Media prompts need batched generation (`neuralfree.parallel` above 1 or `neuralfree.kv.paged=true`) on the CPU backend. They share the prefix cache ([Prefix cache](#prefix-cache)), and every row counts against the context: a prompt with more rows than `neuralfree.ctx` throws `ContextFullException` at submit.
- With chunked prefill a media chunk is never split across steps: each chunk runs as its own forward batch, while decode rows and text chunks keep sharing the step's token budget. A preempted media sequence recomputes chunk by chunk.
- `ForwardBackend.forwardBatch` on the CPU paged path also takes single-section embedding batches directly (`ForwardBatch` embedding rows, explicit `ropePositions`, `nonCausal`). A batch the running backend cannot serve (embedding rows or sections on the GPU path, a wrong width or section count) is refused with a message naming the field.

### Media from the command line

A multimodal model encodes images, videos and audio through a separate projector GGUF. The demo takes it
with the llama.cpp flags and runs the media turns locally:

- `--mmproj <file>` — the projector of a multimodal model in the generate mode; in `--mode audio` the
  same flag stays the speech generator of qwen3tts, pockettts and lfm3a.
- `--mmproj-url <url>`, `-mmu` — the projector downloaded once into the hub cache (http or https).
- `--mmproj-auto` (the default) — when no projector was given, take the `mmproj*.gguf` beside the model
  file, or the sidecar of the `-hf` repository; `--no-mmproj` and `--no-mmproj-auto` turn the projector
  and the search off.
- `--mmproj-offload` (the default) / `--no-mmproj-offload` — run the vision encoder on the model's
  device or on the CPU.
- `-mmdev <device>`, `--mmproj-device` — the device of the vision encoder (e.g. `OpenCL:0`, `cuda:0`);
  `none` forces the CPU, and exactly one device may be named.
- `--mm-encoder-tp-mode weights|data`, `--mmproj-devices <list>` — vLLM's encoder parallel mode: `data` spreads
  an encode batch over the listed devices ([Vision encoder scale-out](#vision-encoder-scale-out)).
- `--ec-transfer-config <json>` — vLLM's encoder-cache connector: publish or load encoder outputs in a shared
  directory ([Sharing encoder output between servers](#sharing-encoder-output-between-servers-encoder-prefill-decode-disaggregation)).
- `--image`, `--audio`, `--video <file[,file...]>` — media files of the `-p` turn (repeatable, each flag
  a comma-separated list); in the chat the same files stage with `/image`, `/audio` and `/video`, each
  becoming a media marker the projector fills.
- `--image-min-tokens <n>`, `--image-max-tokens <n>` — the token budget of one image (0 = the model's
  default).

Media turns run the batch engine, so the run enables paged memory (`neuralfree.kv.paged=true`) when it
is not configured. Media rides the chat template (`-p` or the REPL) and refuses `--raw`, `-no-cnv`, the
completion REPL, speculative decoding and `--prompt-cache` by name; `--no-mmproj` releases the text
generators again. A media file with no projector anywhere refuses at startup naming the flags. With
`--server-base` the files are sent to the server as content parts instead, which refuses a modality it
cannot serve.

The server takes the same sources as launch flags: `--mmproj` (existing), `--mmproj-url` (downloaded
once into the hub cache at startup), and `--mmproj-auto` (the default; the mmproj GGUF beside the model
file, `--no-mmproj-auto` turns the search off, `--no-mmproj` turns media input off entirely).

### Vision encoders (mmproj files)

A projector GGUF (llama.cpp's mmproj file, `general.architecture=clip`) produces the embedding rows of an image. `MmprojModel.open(path)` opens it beside the text model; the projector type is resolved and the tensors are checked before any weight loads, so an unusable file fails at open naming the key or tensor.

```java
try (MmprojModel mm = MmprojModel.open("mmproj-model-f16.gguf");
     VisionEncoder encoder = mm.openVisionEncoder()) {
    float[][] imageRows = encoder.encode(ClipImage.of(width, height, rgb));
    // imageRows feed MediaPrompt.Builder.addMedia(...)
}
```

- `caps()` reports `inpVision()`, `inpAudio()` and `genAudio()` without loading weights, and `memoryUsageBytes()` the projector's weight bytes. `visionHparams()` and `audioHparams()` expose the parsed `clip.*` metadata.
- `encode` takes an image that is already resized and normalized: interleaved RGB floats at `(y * width + x) * 3 + c`. It returns one row per output token at the text model's input width.
- Image bytes are sniffed before decoding: PNG, JPEG, BMP and GIF decode in process, and WebP decodes through a guarded `ffmpeg` pipe; any other format is refused by name.
- Runnable projector types:
  - `mlp` (LLaVA: CLIP ViT, the LLaVA MLP projector). The tower stops at the deepest `clip.vision.feature_layer`, or the second-to-last block without the key. Several feature layers stack along the channels (Granite Vision 3).
  - `mlp_norm` (as `mlp`, with a layer norm before and after the MLP).
  - `gemma3` (Gemma 3: SigLIP ViT with the average-pool projector).
  - `qwen2vl_merger` (Qwen2-VL: 2D M-RoPE encoder over both Conv3d patch frames, pixel-shuffle merger).
  - `qwen2.5vl_merger` (Qwen2.5-VL: the Qwen2-VL graph with window attention on the layers `clip.vision.wa_layer_indexes` lists).
  - `qwen3vl_merger` (Qwen3-VL: both patch frames, position grid resized with align corners, 2D M-RoPE, the merger plus the deepstack mergers of the blocks that carry them).
  - `resampler` (MiniCPM-V: SigLIP ViT with bucketed position rows and the perceiver resampler, all versions).
  - `minicpmv4_6` (MiniCPM-V 4.6: the bucketed-position ViT with the window-attention ViT merger inserted after block `insert_layer_id`, then the final 2x2 neighbor-concatenation merger).
  - `minicpmv4_7` (MiniCPM-V 4.7: the MiniCPM-V 4.6 graph whose overview and slices share one 2D canvas, with the separator rows from the file's `v.tok_embd_sep`).
  - `minicpm_ocr_1_0` (MiniCPM-OCR 1.0: the MiniCPM-V 4.6 graph with slices aligned to the merged patch).
  - `internvl` (InternVL: InternViT with layer scale, pixel shuffle, MLP).
  - `pixtral` (Pixtral: RMS ViT with 2D RoPE, the optional Mistral-3 patch merger, MLP, `[IMG_BREAK]` rows).
  - `lightonocr` (LightOnOCR: the Pixtral graph without `[IMG_BREAK]` rows).
  - `idefics3` (Idefics3 / SmolVLM: SigLIP ViT, pixel shuffle, linear projection).
  - `llama4` (Llama 4: unfold-linear patches with 2D RoPE, pixel shuffle, two-layer MLP).
  - `phi4` (Phi-4 multimodal: SigLIP ViT with the position grid resized to the image, MLP).
  - `ldp` (MobileVLM: CLIP ViT, MLP, then depthwise-conv squeeze-excite blocks, the second at stride 2).
  - `ldpv2` (MobileVLM V2: CLIP ViT, GELU MLP, 2x2 average pool, depthwise-conv position encoding with residual).
  - `adapter` (GLM-Edge: CLIP ViT, 2x2 conv, SwiGLU, begin/end-of-image rows).
  - `cogvlm` (CogVLM: post-norm ViT, SwiGLU projector, begin/end-of-image rows).
  - `glm4v` (GLM-4.1V, GLM-4.5V and GLM-OCR: two patch convolutions, bicubic-resized position rows, RMS ViT with 2D M-RoPE, conv patch merger, gated FFN).
  - `youtuvl` (Youtu-VL: SigLIP2 ViT with 2D M-RoPE and window attention, 2x2 merger).
  - `granite4_vision` (Granite Vision 4.1: SigLIP ViT, one WindowQFormer projector per feature layer; anyres: every image takes the grid pinpoint with the most effective resolution, so the file must carry `clip.vision.image_grid_pinpoints` and is refused by name without it. One chunk holds the overview's `s²` rows, then the tile canvas cropped to the original aspect ratio with a newline row after each canvas row, `s² + h·(w+1)` rows, with `s = (image_size/patch/window_side)·query_side`, so 384 px at patch 16, window 8, query 4 gives 144 rows per tile).
  - `lfm2` (LFM2-VL: SigLIP ViT with resized positions, pixel unshuffle, optional layer norm, GELU MLP).
  - `d1omni_v` (d1-omni: the LFM2-VL graph with exact GELU).
  - `janus_pro` (Janus-Pro: SigLIP ViT, MLP).
  - `cohere2v` (Cohere2 Vision: SigLIP ViT, pixel unshuffle, SwiGLU projector).
  - `paddleocr` (PaddleOCR-VL: SigLIP ViT with resized positions and 2D M-RoPE, layer norm, pixel unshuffle, MLP).
  - `kimivl` (Kimi-VL: resized positions, 2D RoPE, pixel unshuffle, chunk layer norm, GELU MLP).
  - `kimik25` (Kimi K2.5: the Kimi-VL graph whose `[n_embd, W, H]` position grid is resized with the bicubic filter).
  - `dots_ocr` (dots.ocr: RMS ViT with 2D M-RoPE, dense or mixture-of-experts blocks, layer norm, pixel unshuffle, GELU MLP).
  - `dots3note_v` (dots3 Note: the dots.ocr graph; its mixture-of-experts blocks run SiLU experts and need `clip.vision.expert_used_count`).
  - `hunyuanvl` (HunyuanVL: resampled positions, the conv perceiver with a newline column and begin/end rows).
  - `yasa2` (YASA2 ConvNeXt: the image stretched to `clip.vision.image_size`, a fixed 64 rows from an 8x8 pooled grid).
  - `glm5v` (GLM-5V: sides multiples of `2p`, `(W/2p)·(H/2p)` rows; the file must carry `clip.vision.swiglu_clamp`).
  - `lfm3vl` (LFM3-VL: an even patch grid per side, `(W/2p)·(H/2p)` rows; `clip.vision.spatial_merge_size` must be 2 and `clip.vision.attention.head_count_kv` is required).
  - `ling3vl` (Ling 3 VL: sides multiples of `2p`, `(W/2p)·(H/2p)` rows; the patch bias is optional).
  - `minimax_m3` (MiniMax M3: a patch grid that is a multiple of `m` = `clip.vision.spatial_merge_size` (default 2) per side, `(W/(p·m))·(H/(p·m))` rows, capped at 576 by preprocessing; the head width must hold three rotated thirds).
  - `muse-glimmer` (Muse Glimmer: a patch grid that is a multiple of the merge size `m` (default 2), `(W/(p·m))·(H/(p·m))` rows; attention windows of `s x s` patches, `s` the side of the learned position grid, with every fourth block and the last attending globally).
  - `nemotron_v2_vl` (Nemotron Nano V2 VL: the `clip.vision.image_size` square, `((image_size/p)/s)²` rows with `s` = `clip.vision.projector.scale_factor` above 1).
  - `nemotron_omni` (Nemotron Omni: any patch-multiple size with the learned positions resized to the grid, `(W/p)·(H/p)/s²` rows; `clip.vision.projector.scale_factor` is required).
  - `exaone4_5` (EXAONE 4.5: sides multiples of `2p`, `(W/2p)·(H/2p)` rows; `clip.vision.n_wa_pattern` enables window attention over `clip.vision.window_size`-pixel windows (112 when absent); `clip.vision.attention.head_count_kv` is required).
  - `sensenova_u1` (SenseNova-U1: bicubic dynamic resolution to multiples of 32 px, 16 to 4096 merged tokens, `(W/32)·(H/32)` non-causal rows between `<img>` and `</img>`; the M-RoPE grid counts rows and columns from 0 and the whole image takes one temporal position).
  - `step3vl` (Step3-VL: the longest edge capped at 3024, larger images cut into 504-px window crops of 81 rows, each between `<patch_start>` and `<patch_end>` with `<patch_newline>` after each crop row, then a 728-px overview of 169 rows between `<im_start>` and `<im_end>`).
  - `deepseek4v` (DeepSeek-V4 vision: the largest patch-aligned size whose block fits 384 tokens, padded with gray; one non-causal block of 0 to 3 lead pads, `[START]`, the aligner grid's rows each ending in a newline with adjacent row pairs interleaved column-wise, even-keeping pads and `[END]`. Each prompt place takes the lead pads that start the image on a 4-token boundary, so the image encodes once and each place is its own chunk).
  - `deepseek41v` (DeepSeek-V4.1 vision: the block grid plan with a cap of 1024 tokens, the image fitted and centered on gray, samples rounded to bf16 as the reference does; `[START] (row [NEWLINE])×h [END]` = `h·(w+1)+2` causal rows, no lead pads).
  - `deepseekocr` (DeepSeek-OCR: SAM plus CLIP over 16-px patches; a small image is only the 1024 global view, `16·17+1 = 273` rows; a larger one adds rows of 640-px tiles, a row of `g` tiles fusing into one chunk of `(10·g+1)·10` rows; the image is followed by `"\n"`).
  - `deepseekocr2` (DeepSeek-OCR 2: 768-px tiles of 144 query rows, then the 1024 global view of 256 query rows plus the view separator, followed by `"\n"`).
  - `gemma4v` (Gemma 4: RMS ViT with the NEOX 2D RoPE and clippable linears, the pooled projector with the `sqrt(n_embd)` scale).
  - `gemma4uv` (Gemma 4 Unified: the blockless patch embedder, a weightless RMS norm and the input projection).
  - `mimovl` (MiMo-VL and MiMo-V2.5 vision: two summed patch convolutions in 2x2 merge-group order, RMS blocks with per-head attention sinks and full, row- or column-window attention, qwen 2D M-RoPE, SwiGLU FFN; every 2x2 group merges into one row).
  - `gemma3nv` (Gemma 3n vision: the MobileNetV5 convolutional encoder, `image_size / patch_size` rows through the soft-token embedder).

  Every runnable type above also runs on the GPU encoder. The other projector types of llama.cpp are refused by name: `qwen3tts_spkenc`, `qwen3tts_gen`, `pockettts_spkenc`, `pockettts_gen`, `soprano` and `lfm3a_gen`. The bare `qwen2.5o` is refused too, because it names no single encoder; a `qwen2.5o` file still opens and runs each encoder per modality, vision as `qwen2.5vl_merger` and audio as `qwen2a`. A plugin jar can add types through `ProjectorProvider`.
- Image sizes per type. `encode` takes one tile or slice; tiling, slicing and the resize to the family's pixel budget happen before it.
  - CLIP, SigLIP, Gemma3, Idefics3, Janus-Pro, Cohere2V, MobileVLM, GLM-Edge, CogVLM and Granite Vision 4.1 need the exact `clip.vision.image_size` grid (Cohere2V: one square tile).
  - InternVL and Llama 4 also take `image_size` tiles: their dynamic-resolution tiles and the overview.
  - The qwen2 encoders take any rectangle whose patch grid is a multiple of `clip.vision.spatial_merge_size` on each side. Qwen3-VL takes any rectangle with an even patch grid on each side (it merges 2x2 patches). GLM-4V and Youtu-VL take any rectangle whose patch grid is a multiple of 2 on each side.
  - PaddleOCR-VL takes a patch grid that is a multiple of 2 on each side. dots.ocr and dots3-note take a multiple of their merge size on each side.
  - Pixtral and LightOnOCR take the same kind of rectangle. Without a patch merger (`spatial_merge_size` 1) they take any patch-multiple rectangle.
  - Kimi-VL, LFM2 and d1-omni take any rectangle whose sides are multiples of `clip.vision.patch_size`; the pixel unshuffle zero-pads the grid to a multiple of the scale factor.
  - HunyuanVL takes any patch-multiple rectangle; a trailing patch row or column that does not fill a merge window is dropped, as the reference conv drops it.
  - MiniCPM-V and Phi-4 take any rectangle whose sides are multiples of `clip.vision.patch_size`: each llava-uhd slice of MiniCPM-V, and the pixel-budget resize of Phi-4.
  - Granite Vision 4.1 anyres images go through `encodeAnyres(overview, tiles, gridX, gridY, origWidth, origHeight)`. It takes the overview tile, then the `gridX x gridY` tiles in raster order, each an `image_size` square, and the original image size.
- Image preprocessing per type (the `gguf/image` package, as llama.cpp tools/mtmd prepares images):
  - Idefics3 / SmolVLM (`idefics3`): with `clip.vision.preproc_image_size` set and no token budget, the longest side resizes to that edge, keeping the aspect ratio. The short side is truncated, then rounded up to even. The image then stretches up to whole `clip.vision.image_size` tiles. The tiles encode in row-major order, followed by an overview tile (the tiled image shrunk to one tile). Files without the key, and runs with `--image-min-tokens`/`--image-max-tokens`, use the llava-uhd slicing over a resize aligned to `image_size`. Resizes use Lanczos.
  - **LFM2 / d1-omni:** smart resize to 64–256 merged tokens. Above twice the pixel cap, the image stretches to a grid of `clip.vision.preproc_tile_size` (512) tiles; the grid has `preproc_min_tiles`..`preproc_max_tiles` tiles (2..10) and is the one closest to the aspect ratio. The thumbnail goes last. Resizes use torchvision bilinear or bicubic, or Lanczos, per `clip.vision.image_resize_algo`.
  - **Step3-VL:**
    - The longest edge comes from `preproc_image_size`, default 3024.
    - An `image_size` overview.
    - Images larger or longer than 1.5:1 add 504 windows, encoded before the overview.
  - **LightOnOCR:** longest edge = `image_size` unless `preproc_image_size` is set.
  - **GLM-5V:** ceil-aligned canvas inside the GGUF pixel limits, image at top-left on black.
  - **Youtu-VL:** stretched; 1–62500 tokens.
  - **LFM3-VL:** stretched smart resize; the GGUF pixel limits are required, max capped at 4096 tokens.
  - **Muse Glimmer:** Lanczos; up to 4096 tokens.
  - **DeepSeek-OCR / OCR 2:**
    - A 1024 global view, padded gray, encoded last.
    - Images above the tile size (640 / 768) add 2..9 / 2..6 tiles.
    - OCR v1 fuses each tile row into one image.
  - **Gemma 4:** bicubic, no padding, 70–1120 tokens; gemma4uv folds the merge into its patch convolution over `p·m` pixels.
  - **Kimi K2.5:** bicubic within `clip.vision.image_min_pixels` / `image_max_pixels`, default 8–1024 tokens; the stored position grid is resized to the patch grid.
  - **MiMo-VL:** bicubic smart resize within the required `clip.vision.image_min_pixels` / `image_max_pixels`, sides multiples of `2p`.
  - **Gemma 3n:** bilinear resize to the `clip.vision.image_size` square.
  - **MiniMax-M3:** dynamic size, 8–576 tokens.
  - **MiniCPM-V 4.6/4.7 and MiniCPM-OCR:** stretched slices aligned to 4 patches (OCR: the merged patch), Python rounding. minicpmv4_7 lays the overview and slices on one canvas with M-RoPE canvas positions, so it needs an M-RoPE text model.
  - **Llama 4:** bilinear; grids up to 3x3 tiles; overview last.
  - **GLM-Edge:** black pad.
  - **Janus-Pro:** gray pad.
- Slice prompt templates (`MediaMarkers.tiledLayout`, as llama.cpp `mtmd.cpp` lays a sliced image out): the image id and the overview when the overview comes first, the slices start, each slice framed by its start (fixed, or formatted with its row and column) and end with a separator between columns, a row end between rows (or after every row), the slices end, then the overview when it comes last:
  - **idefics3:** `<fake_token_around_image><row_r_col_c>` per tile, `\n` per row, then `\n\n<fake_token_around_image><global-img>` overview `<fake_token_around_image>`.
  - **MiniCPM-V 2.5 and 2.6 templates.** MiniCPM-V 4.6, 4.7 and OCR 1.0 add `<image_id>N</image_id>`.
  - **llama4:** `<|tile_x_separator|>` between columns, `<|tile_y_separator|>` after every row, then `<|image|>` overview, all inside `<|image_start|>`/`<|image_end|>`.
  - **step3vl:** `<patch_start>`/`<patch_end>`, `<patch_newline>`, overview in `<im_start>`/`<im_end>` last.
  - **lfm2:** `<|img_row_r_col_c|>`, then `<|img_thumbnail|>`.
  - **cohere2v:** `<|IMG_LINE_BREAK|>`.
  - **Frames for the new families:**
    - gemma4 `<|image>`/`<image|>`
    - lightonocr `<|im_start|>`/`<|im_end|>`
    - minimax_m3 `]<]start of image[>[`
    - muse-glimmer `<|image_start|>`
    - lfm3vl / youtuvl `<|vision_start|>`; mimovl `<|vision_start|>`/`<|vision_end|>`
    - gemma3nv `<start_of_image>`/`<end_of_image>`
    - glm5v `<|begin_of_image|>`
    - dots3note_v `<|img|>`
    - deepseek-ocr `\n` end
    - d1omni_v and glm_edge / cogvlm / janus_pro: none
  - **Non-causal decode:** gemma4v image rows decode non-causally unless the projector width is 1536 or 2560. gemma4uv and sensenova_u1 always decode non-causally.
- Output rows per image (`px`, `py` the patch grid, `s` the merge size). These are also the image token counts the prompt reserves per prepared tile, as llama.cpp's `clip_n_output_tokens` counts them, including for the types below whose encoder is not runnable yet:
  - CLIP (`mlp`, `mlp_norm`): one row per patch.
  - MiniCPM-V: `clip.minicpmv_query_num` rows per slice. Without the key, 64 for versions 3, 4, 5, 6 and 100045, and 96 otherwise.
  - Pixtral: `(px/s)·(py/s)` token rows plus one `[IMG_BREAK]` row after every token row but the last. LightOnOCR: `(px/s)·(py/s)` rows.
  - InternVL, Idefics3, Llama 4, Gemma 3, Cohere2V, Nemotron v2 VL and Nemotron Omni: `px·py / s²` rows (`(n/s)²` for an `n`x`n` grid).
  - The qwen mergers: `(px/2)·(py/2)` rows. Qwen3-VL rows are `(1 + d)` times the merger width, `d` the number of deepstack blocks.
  - PaddleOCR-VL, Kimi-VL, Kimi K2.5, dots, LFM2 and d1-omni: `⌈px/s⌉·⌈py/s⌉` rows.
  - HunyuanVL: `(⌊px/s⌋ + 1)·⌊py/s⌋ + 2` rows (a newline row per token row, then the begin and end rows).
  - Janus-Pro: one row per patch.
  - Gemma 3n: `image_size / patch_size` rows for any tile.
  - LDP and LDPv2: `px·py / 4` rows. GLM-Edge (`adapter`): `px·py / 4 + 2`, CogVLM: one row per patch `+ 2` (their BOI and EOI rows).
  - GLM-4V and Youtu-VL: one row per 2x2 patch group (`px·py / 4`).
  - Granite Vision 4.1: `(s/w·q)²` rows per tile for an `s`x`s` patch grid, where `w` is `clip.vision.projector.window_side` and `q` is `clip.vision.projector.query_side`. An anyres image adds `h·(v+1)` rows after the overview's tokens, where `h` and `v` are the unpadded tile-grid height and width in tokens and each row ends in a newline row.
  - DeepSeek-OCR: `n = px·py/16`; the global view gives `h(h+1)+1` with `h = ⌊√n⌋`; a fused row of `g` tiles of `t` tokens per side gives `(t·g+1)·t`.
  - DeepSeek-OCR 2: `n`, plus 1 for the global view.
  - Step3-VL: `(w/(p·s))·(h/(p·s))` with `s` = 4.
  - Gemma 4: `px·py/9` (`(px/3)·(py/3)` for gemma4uv's merged patches).
  - MiniCPM-V 4.6 / 4.7 and MiniCPM-OCR 1.0: `(px/m)·(py/m)` per tile, `m` the merge (default 4).
  - MiMo-VL: `(px/2)·(py/2)`.
- The merge size:
  - `clip.vision.projector.scale_factor` for gemma3 (default 4), gemma3nv (default 1), idefics3, internvl, llama4, kimivl, lfm2, d1omni_v and nemotron_v2_vl. It is required for cohere2v, nemotron_omni and dots_ocr; a file without it is refused, naming the key.
  - `clip.vision.spatial_merge_size` for the qwen mergers (default 2), pixtral, lightonocr, glm4v, youtuvl and hunyuanvl (default 2 for glm4v, youtuvl and hunyuanvl). It is required for dots3note_v.
  - PaddleOCR-VL always merges 2x2.
  - step3vl: default 4.
  - gemma4v / gemma4uv: default 3.
  - minicpm_ocr_1_0: default 4; must be 2 or 4.
  - lfm3vl: `spatial_merge_size` required, must be 2.
  - glm5v, minimax_m3, muse-glimmer: `spatial_merge_size`, default 2.
- Family keys:
  - Youtu-VL requires `clip.vision.wa_layer_indexes` (the full-attention layers) and `clip.vision.window_size`.
  - Granite Vision 4.1 requires `clip.vision.projector.spatial_offsets` (one per feature layer, -1 for the average pool), `clip.vision.projector.query_side` and `clip.vision.projector.window_side`.
- dots3note_v needs `clip.vision.expert_used_count` for its mixture-of-experts blocks, which run SiLU experts.
- **Where the encoder runs.** `openVisionEncoder()` puts the encoder on a GPU by default, the way llama.cpp's `--mmproj-offload` does. `device()` reports where it runs: `cpu`, or the API and device name, such as `opencl 'NVIDIA GeForce RTX 5080 Laptop GPU'`.
  - **Choosing the device.** Without a device selector, the encoder runs where `neuralfree.backend` and `neuralfree.device` point. Under `auto`, a host without a usable GPU runs it on the CPU.
  - **Keeping it on the CPU.** `-Dneuralfree.mmproj.offload=false` (llama.cpp `--no-mmproj-offload`) or `-Dneuralfree.mmproj.device=none` keeps the encoder on the CPU. So does `neuralfree.backend=cpu`, unless a device is named.
  - **Naming one device.** `-Dneuralfree.mmproj.device=<index|name substring>` (llama.cpp `--mmproj-device`) names one device, matched as `neuralfree.gpu.devices` matches. Naming several, a device that does not exist, or a device that cannot run the LLM kernels is refused by name.
  - **In code.** The same choices are available without properties: `openVisionEncoder(MmprojOptions.onDevice("1"))`, `MmprojOptions.cpu()`, and `MmprojOptions.defaults().withBackend(BackendMode.CUDA)`. `openVisionEncoder(ComputeBackend)` shares a backend you keep, such as the text model's. `openCpuVisionEncoder()` always runs on the CPU.
- **What runs on the GPU.** Every runnable vision type runs on the GPU encoder. The CPU and GPU encoders run one shared graph per family, so their rows agree within F32 rounding. The GPU graph covers:
  - the patch convolution as a GEMM, with the learned or interpolated position rows;
  - layer and RMS norms, Q/K norms and layer scales;
  - bidirectional flash attention, one launch per window on windowed layers (Qwen2.5-VL, Youtu-VL), and the attention of the MiniCPM-V resampler and the Granite Vision QFormer;
  - the Qwen 2D M-RoPE and the split 2D RoPE (Pixtral, Llama 4, Kimi-VL);
  - the GELU, GELU_QUICK, SiLU and gated FFNs, and the dots3note_v mixture-of-experts blocks;
  - the projectors, including pixel-shuffle merges, stride and depthwise convolutions;
  - the convolutional encoders: the yasa2 ConvNeXt (depthwise 7x7 convolutions, response norms, the 8x8 pool), the gemma3nv MobileNetV5 (depthwise and stride convolutions, the multi-scale fusion adapter, the multi-query attention blocks), and the DeepSeek-OCR SAM backbone (its windowed attention with the decomposed relative-position bias).

  Small steps between device calls run on the host: the MobileNetV5 fusion adapter's upscale and channel concat, squeeze-excite gates, begin/end, break and newline rows, and deepstack splicing.

  When the device cannot run a file's graph, a warning names the reason and the encoder runs on the CPU. The reasons are: a head width that is not a multiple of 4; an attention with no tiled flash-attention plan; a mixture-of-experts block that routes more than 32 of its experts or has more than 1024; or an expert tensor type with no GPU id-matmul kernel.
- **Lifetime.** A GPU encoder owns its device weights, and its backend when it opened the backend itself; close the encoder after use. The CPU encoder spreads attention over the CPU worker pool.
- The Qwen2.5-VL window defaults to 112 pixels when `clip.vision.window_size` is absent, as in clip.cpp.

#### Vision encoder scale-out

The vision encoder can run as a data-parallel rank group (vLLM's `--mm-encoder-tp-mode`): every rank hosts the full
encoder weights and the images of one encode batch are spread over the ranks by pixel count, so the tiles of a
dynamic-resolution or canvas image encode in parallel. The rows are exactly those of the single-device encoder,
returned in prompt order.

```sh
neuralfree serve -m model.gguf --mmproj mmproj.gguf \
  --mm-encoder-tp-mode data --mmproj-devices 0,1
```

- `--mm-encoder-tp-mode weights|data` (`neuralfree.mmproj.tp.mode`): `weights` (default) runs the encoder whole on one
  device; `data` spreads a batch over the rank group.
- `--mmproj-devices 0,1` (`neuralfree.mmproj.devices`): the rank group, one selector per rank (an index into the
  device enumeration or a device-name substring, as `--device` takes them). Without a group, `data` mode uses the
  engine's own devices (`--device`) when it has some; the encoder is then one rank, like vLLM at
  `tensor_parallel_size=1`.
- A rank group of two or more devices under `weights` mode is refused by name: NeuralFree runs the encoder graphs
  whole on one device, so it does not shard the encoder weights across ranks.
- A model a rank's device cannot run falls back to one whole encoder on the first rank, with the reason logged.
- The CLIP dual encoder's towers always run whole on one device and refuse a rank group.

#### Sharing encoder output between servers (encoder-prefill-decode disaggregation)

An encoder instance can publish the encoder output of every media item, and prefill/decode instances can load it
instead of running the vision encoder themselves. The store is a shared directory, keyed by the item's `mm_hash`
(the client's `uuid`, else the content digest) — the same key the media cache and the prefix cache use.

```sh
# encoder instance
neuralfree serve -m model.gguf --mmproj mmproj.gguf --ec-transfer-config \
  '{"ec_connector": "ECExampleConnector", "ec_role": "ec_producer",
    "ec_connector_extra_config": {"shared_storage_path": "/shared/nf-ec"}}'

# prefill / decode instance
neuralfree serve -m model.gguf --mmproj mmproj.gguf --ec-transfer-config \
  '{"ec_connector": "ECExampleConnector", "ec_role": "ec_consumer",
    "ec_connector_extra_config": {"shared_storage_path": "/shared/nf-ec"}}'
```

or with the properties `-Dneuralfree.mm.encoder.cache.dir=/shared/nf-ec -Dneuralfree.mm.encoder.cache.role=ec_consumer`
(`ec_producer`, `ec_consumer` or `ec_both`).

- `ec_connector`: `ECExampleConnector` (vLLM's shared-filesystem connector) or `file`. Other connectors
  (`ECMooncakeConnector`, `ECCPUConnector`) are refused by name; so is any other field.
- A consumer loads an item before it reads the item's bytes, decodes or preprocesses it, so a media part may be sent
  as its `uuid` alone once the producer published it. Without an entry, a uuid sent alone is refused as before.
- A producer saves every item it encodes (replace on re-encode); the directory is created when missing and entries are
  written atomically.
- `/metrics` reports `neuralfree_mm_encoder_cache_loads` and `neuralfree_mm_encoder_cache_saves`.

### Audio input

Audio-capable mmproj files encode speech or music into embedding rows for the language model. Open the mmproj GGUF with `MmprojModel`, then bind its audio encoder:

```java
try (MmprojModel mm = MmprojModel.open("mmproj-model.gguf");
     CpuAudioEncoder audio = mm.openAudioEncoder()) {
    AudioClip clip = AudioDecoder.decode(Files.readAllBytes(path), path.getFileName().toString());
    List<float[][]> chunks = audio.embed(clip);   // one float[rows][n_embd] per encoder chunk
    int tokens = audio.nTokens(clip);             // rows the prompt must reserve
}
```

Supported audio projectors (`clip.projector_type`, or `clip.audio.projector_type` in files with both vision and audio):

| Projector | Model families |
|---|---|
| `ultravox` | Ultravox |
| `qwen2a` | Qwen2-Audio |
| `voxtral` | Voxtral |
| `musicflamingo` | Audio Flamingo 3 / Music Flamingo |
| `meralion` | MERaLiON 2 |
| `glma` | GLM-ASR |
| `qwen3a` | Qwen3-Omni, Qwen3-ASR |
| `lfm2a` | LFM2-Audio, LFM2.5-Audio |
| `d1omni_a` | d1-omni |
| `lfm3a` | LFM3-Audio, LFM3-Omni |
| `gemma4a` | Gemma 4 (audio encoder) |
| `gemma4ua` | Gemma 4 Unified (raw-waveform audio) |
| `parakeet` | Nemotron Omni (Parakeet encoder) |
| `granite_speech` | Granite Speech 3.3 / 4.0, Granite Speech 4.1 Plus |
| `voxtral_rt` | Voxtral Mini Realtime |
| `mimo_audio` | MiMo-V2.5 |
| `muse-spark-audio` | Muse Spark |
| `dots3note_a` | dots3 Note |
| `gemma3na` | Gemma 3n (USM audio encoder) |

- A `qwen2.5o` file (Qwen2.5-Omni, one audio and one vision encoder) opens too: the bare `qwen2.5o` name is refused in the projector registry, and the file's audio encoder runs through the `qwen2a` row above (its vision encoder through `qwen2.5vl_merger`).

- The encoder resamples any input rate to the projector's rate: 24 kHz for `mimo_audio` and `muse-spark-audio`, 16 kHz for the others.
- Each projector uses the log-mel preprocessing of its llama.cpp counterpart:
  - The whisper family cuts 30-second chunks of 400-sample windows with a 160-sample hop.
  - `gemma4a` also cuts 30-second chunks, and `dots3note_a` cuts 60-second chunks.
  - `d1omni_a` encodes the first 30 seconds.
  - `parakeet` reads its filterbank and window from the file (`a.mel_filters`, `a.window`).
  - `gemma4ua` and `muse-spark-audio` take the raw waveform.
  - The other projectors encode the whole clip in one chunk.
- `voxtral_rt` encodes the clip as the first call of a stream: two zero context frames, positions from 0 and the clip padded to whole 80 ms rows. The stream prefix (BOS and padding tokens, leading and trailing silence) is not added.
- `mimo_audio` adds the complete output of encoder block 3 as its skip connection, as the reference MiMo audio tokenizer does. llama.cpp adds that block's output before its FFN, so outputs differ from llama.cpp's.
- `gemma3na` gives 188 rows per 30-second window between `<start_of_audio>` and `<end_of_audio>`. A shorter window is padded to 188 rows with the audio embedding's hard-token row, and longer clips continue in further windows (HF and vLLM truncate at 30 s). It runs on the CPU, with eps 1e-6 on every RMS norm (HF's `rms_norm_eps`; the file's 1e-5 is ignored for this type) and HF's fixed config defaults built in.
- A Gemma 3n mmproj file (gemma3na audio beside gemma3nv vision) serves both from the same weights. A file with one runnable and one unrunnable encoder opens, `caps()` lists both, and `supportsVision()` / `supportsAudio()` list only the encoders that run.
- Any other audio projector is refused by name when the file opens.
- mmproj weights may be F32, F16, BF16 or quantized.
- One mmproj file that holds both a vision and an audio encoder opens once and serves `openVisionEncoder()` and `openAudioEncoder()` from the same weights.

Audio formats:

- WAV, AIFF and AU decode natively: 8/16/24/32-bit integer or 32-bit float PCM, any channel count, mixed to mono.
- MP3, FLAC, OGG and M4A decode through `ffmpeg`, which must be on the `PATH`. Without it they are refused with a message naming the format. Convert them to WAV (for example `ffmpeg -i input.mp3 -ac 1 -ar 16000 input.wav`) to decode without ffmpeg.
- Those `ffmpeg` decodes share the media input guard: the bytes are read on standard input through the `pipe` protocol and the audio demuxers are whitelisted, so an HLS playlist or ffconcat script posing as audio cannot make `ffmpeg` open another file or URL.
- A single decode is capped at one hour of 48 kHz mono audio.
- The encoder runs on the CPU.

## Control vectors

`ops/heads/ControlVector` (with `ControlVectorExtra` and `CpuControlVectorExtra`) applies a control vector to the hidden states at inference. `nftool cvector` trains one (`CvectorGenerator`, llama.cpp's cvector-generator): it opens the model on the CPU engine, tokenizes every positive/negative prompt pair with the vocabulary's BOS policy, pads both sides to one length with the pad token llama.cpp derives (the last token of tokenizing a single space), and per pair prefills the positive side from position 0, switches the tap side, prefills the negative side and keeps the per-layer differences (the rows whose every value stays within 1e-6 drop, as llama.cpp's `filter_nonzero_rows` drops them). The tracked layers are all decoder layers except the last. The directions reduce by PCA power iteration or the row mean, and the export writes the llama.cpp `controlvector` GGUF form (`direction.<il>` tensors, `controlvector.model_hint` from `--model-hint` or the model's architecture).

Control vectors in that form apply at inference on the CPU and the GPU, as llama.cpp's `--control-vector` does: `--control-vector`, `--control-vector-scaled` and `--control-vector-layer-range` ([flags](#control-vector-flags)), or the properties `neuralfree.control-vector=<gguf>[:scale][,...]` and `neuralfree.control-vector-layer-range=<start>:<end>`, which the server reads too. The scaled rows of every listed vector sum, and each controlled layer adds the sum to its output (`x += scale * direction`), the placement llama.cpp uses. Layer 0 and the top layer never receive an add. Vectors are fixed for the engine's lifetime; the server has no endpoint that adds or removes them.

Control-vector files must match the model exactly: every `direction.<il>` row must be `n_embd` floats wide, as llama.cpp's `llama_adapter_cvec::apply` requires. A file whose rows are narrower or wider than the model is refused when the engine opens, with a message naming the file, its width, the model's width and the layer range.

The vectors also apply on the paged multi-sequence path: batched prefill and decode on the GPU produce the same logits as the same batch on the CPU.

## imatrix

`nftool imatrix <model> [options]` collects the importance matrix for quantization over a calibration corpus, as llama.cpp's `imatrix` tool does. With `-ngl 0` (the default) the model runs on the CPU engine, and the collector (`cpu/quant/ImatrixCollector`) taps the CPU matmul inputs (`CpuMatVec.observeGlobally`). With any other `-ngl` it runs on the GPU engine: the device matmul tap (`gpu/forward/GpuMatmulTap`) reads the input of every collected product back to the host, which serializes the forward pass. Layers kept on the host are collected through the CPU tap. If no GPU backend can run the model, the run falls back to the CPU engine and says so.

Both engines collect the same products:
- The projections.
- The per-head and per-group products of stacked 3-D weights, one matrix per head or group: absorbed-MLA `attn_k_b`/`attn_v_b`, sparse-MLA and QSA attention, and the grouped DeepSeek-V4 `attn_output_a`.
- The routed experts, one input row per routed (token, slot) pair: the FFN experts and the routed value experts.

Routed experts kept in host memory (`--cpu-moe`, `--n-cpu-moe`, `-ot`, the MoE expert cache) are collected through the CPU tap. While a collection wants them, every routed pair of those layers is computed on the host, bypassing the expert cache's device slots and its prefetch.

Which products count follows llama.cpp's ask (`collect_imatrix`):
- Every routed product (`MUL_MAT_ID`) is collected.
- A dense product (`MUL_MAT`) is collected only over at least 16 input rows. Single-token and other small decodes add only their routed experts, and so does a final sub-batch below 16 rows when `--batch-size` splits a window.
- Of the dense products, only these names are kept:
  - the block tensors (`blk.*`);
  - `output.weight`, `token_embd.weight` and `nextn.post_projection.weight` under `--process-output`;
  - `nextn.pre_projection.weight` under `--nextn`.

The calibration files are tokenized once into one sequence and run as consecutive whole-context windows of `--ctx-size` tokens, each decoded from position 0 with its first token replaced by BOS when the vocabulary adds one. The corpus must tokenize to at least twice the window. Each decode runs `--batch-size / --ctx-size` windows as parallel sequences over paged memory. A batch below the window splits each window into `ceil(ctx / batch)` decodes.

Per weight tensor the file holds, under the stored GGUF names, with one matrix per expert of a MoE tensor and per head or group of a stacked 3-D weight:
- the column sums of the squared activations (`<tensor>.in_sum2`);
- the plain column sums (`<tensor>.in_sum`);
- the activation-row counts (`<tensor>.counts`);
- the twelve statistics of `imatrix.stats_schema` (`<tensor>.stats`: sum_sq, mean, elements, std_deviation, skewness, kurtosis, gain, h_norm, l2_dist, cossim, pearson, covariance).

| Option | Meaning | Default |
|---|---|---|
| `--file <fname>` | Calibration text; repeatable | |
| `--in-file <fname>` | A matrix to combine in, GGUF or legacy `.dat` (its sums and counts scaled by the chunk size, as llama.cpp does); repeatable. Without `--file` the tool only combines and saves | |
| `--out-file <fname>` | Output path, replaced when it exists | `imatrix.gguf` |
| `--output-format gguf\|dat` | The GGUF form or the legacy pre-GGUF binary | `gguf` |
| `--ctx-size <n>` | Window length in tokens; a warning above the trained context | 512 |
| `-b`, `--batch-size <n>` | Tokens per decode: below `--ctx-size` or a multiple of it. A multiple runs `n / ctx-size` windows per decode as parallel sequences (at most 256). Below 16, only the routed experts are collected | 2048 |
| `-ngl`, `--n-gpu-layers <n\|auto\|all>` | 0 runs the CPU engine; any other value runs the GPU engine with that many layers on the device, a count past the model's layers offloading it whole as llama.cpp does | 0 |
| `-ot`, `--override-tensor <pattern=cpu\|gpu,...>` | GPU engine: keep the matching tensors in host or device memory, as `nftool chat` takes it | |
| `--cpu-moe`, `--n-cpu-moe <n>` | GPU engine: keep the routed experts of every layer, or of the first `n` layers, in host memory; they are collected through the CPU tap | off |
| `--moe-cache <off\|on\|auto\|MiB>`, `--moe-cache-mib <n>` | GPU engine: the MoE expert cache, as `nftool chat` takes it; its layers are collected through the CPU tap | |
| `--chunks <n>` | At most `n` windows, `-1` for all | -1 |
| `--chunk <n>`, `--from-chunk <n>` | Skip the first `n` windows of the corpus | 0 |
| `--output-frequency <n>` | Rewrite the output each time the window count crosses a multiple of `n` | 10 |
| `--save-frequency <n>` | Also keep `<out-file>.at_<chunk>` each time the window count crosses a multiple of `n`; 0 is off | 0 |
| `--process-output` | Also collect `output.weight`, `token_embd.weight` and `nextn.post_projection.weight` | off |
| `--nextn` | Also collect the model's NextN/MTP head: after every decode it is replayed over the trunk's hidden rows, pairing each token with the previous position's hidden row (`n_ctx - 1` rows per window). The replay follows the same 16-row rule. Needs `--batch-size` at most `--ctx-size`. A model without NextN layers only gets a warning; multi-layer NextN heads are refused | off |
| `-md`, `--model-draft <fname>` | Collect the NextN head of an MTP draft file of the model; implies `--nextn`. Refused when the model already includes NextN layers | |
| `--ppl`, `--no-ppl` | Report the perplexity over the second half of every window (`[n]<ppl>,` per window), ending with `Final estimate: PPL = <ppl> +/- <error>` | on |
| `--show-statistics` | Print the statistics of the first `--in-file` and exit without a model: per tensor mean, standard deviation, skew, kurtosis, normalized entropy and ∑E[A²]; per layer gain, L2 distance (files with `.in_sum`), cosine similarity, Pearson correlation and covariance. NextN blocks are labelled `mtp0`, `mtp1`, … | |
| `--parse-special` | Parse special tokens in the calibration text | off |

`imatrix.chunk_count` starts at the chunk count of the combined `--in-file` matrices and grows by one per window, so an extended run continues the previous accounting. `imatrix.n_layer_nextn` records the NextN blocks the run replayed, or else the first non-zero count among the `--in-file` matrices; a different count in a later file gets a warning. When combining, `.in_sum` is kept only if every contribution has it, as llama.cpp does. `ImatrixFile` reads both llama.cpp forms, so a matrix collected by either tool can be combined into a run, and `nftool quantize --imatrix` accepts either.

## Session files

`ModelMemory.saveSession(path)` / `loadSession(path)` save and restore a single-sequence memory's whole state, llama.cpp `llama_state_save_file`/`llama_state_load_file` style (`memory/SessionFile`): the stored tokens with their reuse frontier and wrap flag, every KV group's position mirror, the K/V rows of every owning cache layer and the live recurrent state tensors. A load checks the whole plan identity first (context, ubatch, group and layer geometry, cache dtypes, head shapes, tensor names), resets the memory and writes everything back, so the next decode continues the saved sequence with the same logits. A file saved on the CPU loads on the GPU and the other way round, bit-exact; a mismatch is refused by naming the field. Paged memories, DeepSeek-V4 compressed caches, indexer and MSA side caches, cross-attention caches and layers that share another layer's KV are refused with `session save/load refused: ...`; the server's slot files ([Slots](#slots)) cover the paged memory. Snapshot shadows, compressor rollback planes and the sampler chain stay runtime-only.

`GGUFModel.saveSession(Path)` / `loadSession(Path)`, on every `EngineModel` including the GPU models, move the same file for a loaded model: the model continues the saved sequence exactly, and its prefix reuse skips a matching prompt's shared tokens.

### Prompt cache

`neuralfree cli` takes llama.cpp's `--prompt-cache` flags over the session file:

```bash
neuralfree cli --prompt-cache turns.session -m model.gguf -p "..."          # one-shot
neuralfree cli --prompt-cache turns.session -m model.gguf                   # console chat
neuralfree cli --prompt-cache turns.session --prompt-cache-all -m model.gguf -i -p "Once upon a time"
```

- `--prompt-cache <file>`: at start, an existing non-empty file loads into the model's memory, and the next prompts reuse the tokens they share with the saved sequence instead of prefilling them. An absent or empty file starts a new session. The state saves back to the file after the first turn's generation.
- `--prompt-cache-all`: every turn's prompt and generated tokens are saved, not the first turn's alone. It also runs the completion REPL (`-i`, `-if`), which saves each of its turns.
- `--prompt-cache-ro`: load but never write; refused without `--prompt-cache`.

The flags need a local decoder-only model in the generate mode: `--mode embed`, `--mode classify`, `--mode rerank`, `--mode diffusion`, `--mode audio`, a model whose AUTO mode resolves to one of them, `--benchmark`/`--bench-streaming`, `--vram-budget`, `--export-gguf`, an encoder-decoder, a paged memory (`-np` above 1) and client mode (`--server-base`) are refused by name. `--prompt-cache-all` needs `--prompt-cache`, and an empty path or a directory in place of the file is refused. A file saved on the GPU loads on the CPU and back unchanged, and a saved turn reloads into a fresh model with the same continuation.

## System properties

Every `neuralfree.*` system property the code reads is listed below. Every Gradle run task (`runApp`, `runDemos`, `runLLMDemo`) forwards `-Pneuralfree.x=y` and `-Dneuralfree.x=y` to the program's JVM; pass them to your own JVM as `-Dneuralfree.x=y`. The build properties are under [Build](#build). The demo's command-line flags take precedence over the engine properties they correspond to.

### Engine (`EngineConfig.fromSystemProperties`)

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.backend` | `auto`, `opencl`, `cuda`, `vulkan`, `hip`, `metal` or `cpu`. Also chooses the training framework's API for `runApp` and `runDemos` (`NeuralFree.backendFromProperty`), where `cpu` is rejected. | `auto` |
| `neuralfree.device` | Device-name substring for every API (case-insensitive) | none |
| `neuralfree.ram.cache.mb` | Host cache of streamed weights in MiB; `-1` = automatic ([models larger than RAM](#models-larger-than-ram)), `0` = none | -1 |
| `neuralfree.vram.depth` | Streamed layers resident in VRAM; `0` = from the VRAM budget | 0 |
| `neuralfree.draft` | Draft GGUF for speculative decoding | none |
| `neuralfree.draft.max` | Maximum drafted tokens per round | 3 |
| `neuralfree.draft.min` | Minimum verified draft length | 0 |
| `neuralfree.draft.p.min` | Draft probability floor | 0 |
| `neuralfree.draft.p.split` | Secondary-candidate probability that splits a draft branch into a tree | 0.1 |
| `neuralfree.draft.sample.method` | How the batch engine's draft model picks draft tokens (vLLM's `draft_sample_method`, `--spec-draft-sample-method`): `greedy` drafts the argmax, `probabilistic` draws at the request's temperature and verifies with the draft probabilities; needs a draft model | `greedy` |
| `neuralfree.draft.heterogeneous.vocab` | A separate draft model with a different tokenizer drafts over the two vocabularies' intersection by token text and greedy verification keeps the output lossless (vLLM's `use_heterogeneous_vocab`, `--spec-draft-heterogeneous-vocab`) | false |
| `neuralfree.draft.kv.type` | KV cache type of the draft contexts, K and V, as `--kv-type-draft` | `f16` |
| `neuralfree.draft.kv.typeK`, `neuralfree.draft.kv.typeV` | Draft cache type of K, of V, as `-ctkd`, `-ctvd` | `f16` |
| `neuralfree.draft.ubatch` | Tokens per forward step of a draft context; 0 keeps the target's ubatch | 0 |
| `neuralfree.draft.type` | The drafting kind, as `--spec-type`; the batch engine runs every type but `draft-mtp` ([Speculative decoding](#speculative-decoding)) | `auto` |
| `neuralfree.draft.verify.device` | `match` verification selects on the GPU target's device: the greedy ids, or the draw over the device's top candidates (llama.cpp `-bs`) | true |
| `neuralfree.draft.dflash.device` | A dflash draft drafts on the GPU target's device, as `--[no-]spec-draft-backend-sampling` | true |
| `neuralfree.sampling.device` | A generation step whose chain keeps inside the device's top-k candidates draws over them on the device (llama.cpp `-bs`) | true |
| `neuralfree.mtp` | Self-speculation with the model's NextN/MTP blocks | false |
| `neuralfree.draft.mtp.device.rows` | The NextN/MTP drafter captures the target's hidden rows on the device (one-device GPU loads); `false` copies them to the host (llama.cpp's `LLAMA_MTP_H_DEVICE=0`) | true |
| `neuralfree.lora` | Adapters as `path[:scale][,path[:scale]...]` | none |
| `neuralfree.lora.max` | Adapters a registry keeps resident | 4 |
| `neuralfree.lora.max.rank` | Largest adapter rank a registry loads | 64 |
| `neuralfree.lora.cache` | Unloaded adapters a registry keeps for a quick reload; 0 keeps none, so an unloaded or evicted adapter is read again from its file | 8 |
| `neuralfree.mmproj.offload` | `true` or `false`: run the vision encoder on a GPU (llama.cpp `--mmproj-offload` / `--no-mmproj-offload`) | `true` |
| `neuralfree.mmproj.device` | The vision encoder's device: an index in enumeration order or a device-name substring, or `none` for the CPU (llama.cpp `--mmproj-device`); one device only | the `neuralfree.backend` / `neuralfree.device` choice |
| `neuralfree.n.gpu.layers` | Layers on the device: `auto`, `all` or a count that includes the output head ([hybrid execution](#hybrid-cpu-and-gpu-execution)) | `auto` |
| `neuralfree.kv.offload` | Whether a GPU load keeps the KV caches on the device (llama.cpp `offload_kqv`); `false` runs the layers on the device with their attention on the host over a host-resident cache ([hybrid execution](#hybrid-cpu-and-gpu-execution)) | `true` |
| `neuralfree.override.tensor` | Tensor placement overrides, `pattern=target[,...]` | none |
| `neuralfree.cpu.moe` | Keep every MoE layer's expert weights on the host | false |
| `neuralfree.n.cpu.moe` | Keep the expert weights of the first `n` MoE layers on the host | 0 |
| `neuralfree.control-vector` | Control vectors as `path[:scale][,path[:scale]...]` ([Control vectors](#control-vectors)) | none |
| `neuralfree.control-vector-layer-range` | `start:end`, the 1-based decoder layers the vectors apply to | every layer |
| `neuralfree.chat.template`, `neuralfree.chat.template.file` | A chat template (source or built-in name), or a file holding one, replacing the model's | the model's |
| `neuralfree.chat.jinja` | `false` renders llama.cpp's non-Jinja route: the template's text is replaced by the built-in its text detects, and a template none is detected from is refused ([Choosing a chat template](#choosing-a-chat-template)) | `true` |
| `neuralfree.reasoning.parser` | Reasoning parser by vLLM name, or `auto` for the chat template's | `auto` |
| `neuralfree.tool.parser` | Tool-call parser by vLLM name, `auto` to detect it from the chat template, `none` to leave calls in the content | `auto` |
| `neuralfree.tool.strict` | How far the tool-call grammars constrain: `auto`, `function` or `parameter` ([Tool calls](#tool-calls)) | `auto` |

### Server (`EngineLifecycle`, `ServerOptions.fromSystemProperties`, `ApiKeys`)

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.server.model` | The GGUF or checkpoint the server opens | none (unloaded) |
| `neuralfree.server.model.name` | Served model names, comma-separated ([model names](#http-server)) | the model path |
| `neuralfree.server.request.timeout` | Seconds a request may run before it is cancelled with 503, at most 31536000 (a year) | 1800 |
| `neuralfree.server.spm.infill` | `/infill` puts the suffix before the prefix | false |
| `neuralfree.server.responses.store` | Responses API responses kept for retrieval, `previous_response_id` and background mode | 0 (off) |
| `neuralfree.server.api.key` | API keys, comma-separated ([API keys](#api-keys)) | none |
| `neuralfree.server.api.key.file` | A file of API keys, one per line | none |
| `neuralfree.server.slot.save.path` | The directory slot save and restore use ([Slots](#slots)) | none (off) |
| `neuralfree.server.dev.mode` | Serve vLLM's development and RL control routes (`--dev-mode`, [Development endpoints](#development-endpoints)); unset reads vLLM's `VLLM_SERVER_DEV_MODE` | false |
| `neuralfree.server.grpc` | Serve the gRPC `Inference` and `Control` services, reflection and gRPC health (`--grpc`, [gRPC API](#grpc-api)) | false |
| `neuralfree.server.elastic.ep` | Serve the elastic data-parallel scaling routes (`--enable-elastic-ep`) | false |
| `neuralfree.server.fault.tolerance` | Serve the fault-tolerance routes (`--enable-fault-tolerance`) | false |
| `neuralfree.server.sleep.mode` | Serve the checkpoint window and let the sleep operations release memory (`--enable-sleep-mode`, [Sleep, pause and checkpoint](#sleep-pause-and-checkpoint)) | false |
| `neuralfree.server.profiler.config` | vLLM's ProfilerConfig JSON: `profiler` and `torch_profiler_dir` (`--profiler-config`); no profiler turns the profile window off | none |
| `neuralfree.watermark.config` | vLLM's watermark configuration JSON (`--watermark-config`); every text route watermarks unless a request sets `watermarking: false` | none (off) |
| `neuralfree.server.profile.dir` | The profile window's trace directory when the profiler config names no `torch_profiler_dir` | none |
| `neuralfree.server.tools` | Built-in server tools, comma-separated or `all` ([Server tools and MCP](#server-tools-and-mcp)) | none (off) |
| `neuralfree.server.mcp.config`, `neuralfree.server.mcp.json` | The `mcpServers` file, or the document inline | none |
| `neuralfree.server.tools.max.rounds` | Rounds the agent loop runs before returning the pending calls | 10 |
| `neuralfree.server.allowed.hosts` | Trusted `Host` names of server-tool and MCP requests, comma-separated host names or IP literals; `*` disables the guard ([Server tools and MCP](#server-tools-and-mcp)) | none (loopback and the bind host only) |
| `neuralfree.server.webui` | Serve the bundled chat UI ([Web UI](#web-ui)) | true |
| `neuralfree.server.ui.path` | The directory the UI's static files serve from instead of the bundle ([Web UI](#web-ui)) | none (bundle) |
| `neuralfree.server.ui.config` | The UI's default settings, any JSON value, reported by `/props` as `ui_settings` ([Web UI](#web-ui)) | none |
| `neuralfree.server.api.prefix` | The path prefix every route serves under ([Web UI](#web-ui)) | none |
| `neuralfree.server.models.dir`, `neuralfree.server.models.preset` | Router mode's model sources: a GGUF directory and an INI preset file ([Multi-model router](#multi-model-router)) | off |
| `neuralfree.server.models.max`, `neuralfree.server.models.memory.margin`, `neuralfree.server.models.autoload` | Router mode's open-model limit, free-memory margin in MiB, and loading on first mention | 4, 0, true |
| `neuralfree.server.sleep.idle.seconds` | Seconds without work before the model is released ([sleep](#sleep-pause-and-checkpoint)); negative never sleeps | -1 |

### Hugging Face checkpoints (`ModelSourceOptions.fromProperties`)

GGUF files ignore these; Hugging Face checkpoints read them at open. The meanings and defaults are the [flags table](#flags-and-properties)'s; each is the matching property of `ModelSourceOptions` or `EngineConfig`.

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.hf.materialize` | `auto`, `memory`, `disk` or `none` | `auto` |
| `neuralfree.hf.materialize.memory.max` | Direct-memory budget of `auto`, as `<n>[K|M|G]` bytes | `2G` |
| `neuralfree.hf.cache.dir` | Parent of the per-process disk-sidecar directories | `~/.cache/neuralfree/hf` |
| `neuralfree.hf.outtype` | `source`, `auto`, `f32`, `f16`, `bf16` or `q8_0` | `source` |
| `neuralfree.quant.online` | `none`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1` or `iq4_nl` | `none` |
| `neuralfree.hf.quant.exact` | Prefer bit-exact weight types over close repacks | false |
| `neuralfree.hf.mtp` | Keep the model's MTP/NextN layers | true |
| `neuralfree.hf.draft` | The converter's draft selection: `none`, `mtp` (the NextN head alone), `no-nextn` (the target without it) or `dspark` (the DSpark draft as a `dflash` GGUF) | `none` |
| `neuralfree.hf.st.dense` | Include sentence-transformers dense heads | true |
| `neuralfree.hf.overrides` | JSON object merged into `config.json` before conversion | none |
| `neuralfree.hf.tokenizer` | Tokenizer directory, or GGUF file, replacing the model's tokenizer files | none |
| `neuralfree.lora.base.config` | HF config of a LoRA adapter's GGUF base | none |
| `neuralfree.hf.base.dir` | Local base model a decision-LoRA checkpoint merges | none |
| `neuralfree.act.quant` | Activation quantization: `weight-only` or `checkpoint` | `weight-only` |
| `neuralfree.sampling.model.defaults` | Apply the model's sampling defaults | true |

### Memory (`MemoryConfig.fromProperties`)

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.ctx` (alias `neuralfree.max.context`) | Context length; `0` = `min(n_ctx_train, largest fitting)` | 0 |
| `neuralfree.kv.type` | KV cache type of K and V | `f16` |
| `neuralfree.kv.typeK`, `neuralfree.kv.typeV` | K or V type; overrides `neuralfree.kv.type` | `f16` |
| `neuralfree.ubatch` | Tokens per forward step | 512 |
| `neuralfree.swa.full` | Full-size sliding-window caches | false |
| `neuralfree.mla.cache` | MLA (deepseek2 family) cache: `latent` keeps the compressed latent row and decompresses it per step through `attn_k_b` / `attn_v_b`; `decompressed` stores per-head K and V rows, on CPU and GPU for both the fused `attn_kv_b` and the per-head `attn_k_b` / `attn_v_b` conversions; `auto` picks `latent` for absorbed conversions and `decompressed` otherwise (`--mla-cache`). A LoRA delta on `attn_kv_b` is refused on an absorbed conversion, which never multiplies `attn_kv_b` | `auto` |
| `neuralfree.prefix.reuse` | Keep the common prompt prefix between generations; off by default beside a model larger than RAM, `true` forces it | true |
| `neuralfree.cache.ram` | MiB of host RAM the prefix cache keeps evicted blocks in (llama.cpp's `--cache-ram`, vLLM's CPU KV offloading): a later prompt with the same prefix brings them back instead of prefilling again, and KV events report them with medium `CPU`. Needs the paged prefix cache over a memory whose blocks are a sequence's whole state (no recurrent, DSV4 compressed, sparse side or cross-attention caches); refused by name otherwise. `-1` = no limit | 0 |
| `neuralfree.cache.disk.mib` | MiB of disk the prefix cache keeps the host tier's dropped blocks in, one file per block; KV events report them with medium `STORAGE`. Needs `neuralfree.cache.ram` on and `neuralfree.cache.disk.dir`. `-1` = no limit | 0 |
| `neuralfree.cache.disk.dir` | The disk tier's directory; blocks left there by an earlier run are served again | none |
| `neuralfree.prefix.cache.budget` | Tokens the batch engine's free cached blocks hold together; 0 = bounded by the KV pools ([Prefix cache](#prefix-cache)) | 0 |
| `neuralfree.prefix.hash.algo` | The block-hash algorithm of the batch engine's prefix cache (vLLM's `--prefix-caching-hash-algo`): `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor` ([Prefix cache](#prefix-cache)) | `sha256` |
| `neuralfree.host.ip` | The node IP an ephemeral KV-events ZMQ bind (`tcp://*:0`) binds and reports (vLLM's `VLLM_HOST_IP`); unset uses the primary interface's address ([Prefix cache](#prefix-cache)) | unset |
| `neuralfree.kv.events` | KV cache event publisher: `none`, `log` or a `KvEventPublisher` class name ([Prefix cache](#prefix-cache)) | none |
| `neuralfree.kv.rotation` | Hadamard rotation of quantized KV rows | true |
| `neuralfree.context.shift` | Shift the context when full instead of stopping | false |
| `neuralfree.state.snapshots` | Reserve recurrent-state snapshot buffers | true when speculative decoding is configured |
| `neuralfree.nextn.memory` | Place the model's NextN/MTP blocks and their caches in the fit | true when `neuralfree.mtp` is set |
| `neuralfree.parallel` | Sequences the memory holds at once (llama.cpp's `-np`), 1-256; above 1 selects the paged KV memory ([Paged KV memory](#paged-kv-memory)) | 1 |
| `neuralfree.kv.paged` | Force the paged KV memory for a single sequence ([Paged KV memory](#paged-kv-memory)); property-gated, not the engine default | false |
| `neuralfree.kv.page.size` | Page size of the paged KV pools, a power of two in [1, 256] | 16 |
| `neuralfree.kv.unified` | Share the whole context between the `neuralfree.parallel` sequences instead of one pool per sequence | true |
| `neuralfree.chunked.prefill`, `neuralfree.max.batched.tokens`, `neuralfree.long.prefill.tokens` | The batch engine's chunked prefill and step budget ([Paged KV memory](#paged-kv-memory)) | true, 0, 0 |

### Sampling (`SamplerConfig.fromProperties`)

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.temp` | Temperature | 0.8 |
| `neuralfree.top.k` | Top-k | 40 |
| `neuralfree.top.p` | Top-p | 0.95 |
| `neuralfree.min.p` | Min-p | 0.05 |
| `neuralfree.typical.p` | Typical-p | 1 |
| `neuralfree.repeat.penalty` | Repetition penalty | 1 |
| `neuralfree.presence.penalty` | Presence penalty | 0 |
| `neuralfree.frequency.penalty` | Frequency penalty | 0 |
| `neuralfree.repeat.last.n` | Penalty window; `-1` = whole context | 64 |
| `neuralfree.seed` | Seed; `-1` = random | -1 |
| `neuralfree.greedy` | Argmax decoding | false |
| `neuralfree.samplers` | Semicolon-separated list of stages overriding the default chain order, e.g. `top_k;top_p;min_p` (`StageKind.parseOrder`); an unknown stage is rejected naming the property | llama.cpp's order |
| `neuralfree.min.keep` | Minimum keep count shared by top-k, top-p, min-p and typical-p | llama.cpp's |
| `neuralfree.dry.multiplier` | DRY repetition strength; `0` disables the stage | 0 |
| `neuralfree.dry.base` | DRY exponential-penalty base; below 1 disables the stage | llama.cpp's |
| `neuralfree.dry.allowed.length` | DRY length the penalty starts at | llama.cpp's |
| `neuralfree.dry.penalty.last.n` | DRY window of accepted tokens; `0` disables the stage | 0 |
| `neuralfree.dry.sequence.breakers` | DRY restart sequences | llama.cpp's |
| `neuralfree.top.n.sigma` | Top-n-sigma; `0` and below disables the stage | 0 |
| `neuralfree.xtc.probability` | XTC trim probability; `0` disables the stage | 0 |
| `neuralfree.xtc.threshold` | XTC probability threshold; above 0.5 disables the stage | llama.cpp's |
| `neuralfree.mirostat` | Mirostat version: `0` off, `1` or `2` | 0 |
| `neuralfree.mirostat.tau` | Mirostat target entropy | llama.cpp's |
| `neuralfree.mirostat.eta` | Mirostat learning rate | llama.cpp's |
| `neuralfree.dynatemp.range` | Dynamic-temperature half-width; `0` keeps the plain scaling | 0 |
| `neuralfree.dynatemp.exponent` | Dynamic-temperature entropy exponent | llama.cpp's |
| `neuralfree.adaptive.target` | adaptive-p's target probability; negative disables the adaptation | off |
| `neuralfree.adaptive.decay` | adaptive-p's EMA decay | llama.cpp's |

### Loading, CPU and diagnostics

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.model` | Model path for `LLMDemo` | none |
| `neuralfree.disable.mmap` | Read tensors instead of mapping them | false |
| `neuralfree.rope.scaling` | RoPE scaling type override: `none`, `linear`, `long`, `yarn`, `dynamic` ([RoPE and long context](#rope-and-long-context)) | unset |
| `neuralfree.rope.scale` | RoPE context scaling factor (`rope.scaling.factor`), as `--rope-scale` | unset |
| `neuralfree.rope.freq.scale` | RoPE frequency scale; sets `rope.scaling.factor` to its reciprocal, as `--rope-freq-scale` | unset |
| `neuralfree.rope.freq.base` | RoPE base frequency override | unset |
| `neuralfree.yarn.orig.ctx`, `neuralfree.yarn.ext.factor`, `neuralfree.yarn.attn.factor`, `neuralfree.yarn.beta.fast`, `neuralfree.yarn.beta.slow` | YaRN overrides, as the `--yarn-*` flags | unset |
| `neuralfree.cpu.threads` | CPU engine worker threads | available processors |
| `neuralfree.cpu.simd` | `false` keeps the scalar CPU kernels; `true` selects the Vector API kernels when `jdk.incubator.vector` is resolved | true |
| `neuralfree.cpu.repack` | Type selection for CPU weight repacking: `auto` repacks only the types whose repacked kernels measured faster than the plain ones (currently none), `on` every covered type (`Q4_0`, `Q8_0`, `Q4_K`), `off` none; any other value is rejected by name | auto |
| `neuralfree.cpu.repack.budget` | Byte cap on the repacked copies; unset measures the free-heap headroom automatically | unset |
| `neuralfree.stream.sync` | Force synchronous transfers for streamed weights | false |
| `neuralfree.diag` | GPU stage diagnostics | false |
| `neuralfree.diag.dump` | Append full-precision diagnostic values to the dump file | false |
| `neuralfree.diag.dumpfile` | Dump file | `<java.io.tmpdir>/neuralfree_diag_dump.txt` |
| `neuralfree.diag.steps` | Steps the diagnostics report | 1 |

### Backend selection and tuning

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.backend.order` | AUTO's API preference; when unset, the target vendor's native API (CUDA on NVIDIA, OpenCL on Intel) goes first | `metal,vulkan,cuda,opencl,hip` |
| `neuralfree.opencl.device`, `neuralfree.cuda.device`, `neuralfree.vulkan.device`, `neuralfree.hip.device` | Per-API device-name substring; takes precedence over `neuralfree.device` for that API | none |
| `neuralfree.opencl.benchmark.disable` | Rank OpenCL devices on published attributes only | false |
| `neuralfree.opencl.disable` | Skip OpenCL enumeration | false |
| `neuralfree.opencl.skip.probe` | Skip the subprocess probe for crashing OpenCL platforms | false |
| `neuralfree.opencl.binaryCache` | OpenCL program binary cache (`~/.cache/neuralfree/opencl`, capped at 1 GiB) | true |
| `neuralfree.gpu.profile` | Per-command profiling on OpenCL queues | false |
| `neuralfree.gpu.decodeReplay` | Replay the single-token decode trunk as one CUDA graph / HIP graph / Vulkan or OpenCL command-buffer submission ([decode-step command replay](#decode-step-command-replay)) | true |
| `neuralfree.gpu.chainReplay` | Extend decode replay into speculative rounds: keep the drafter's capture across rollbacks and record the verify step ([decode-step command replay](#decode-step-command-replay)) | true on CUDA, false elsewhere |
| `neuralfree.gpu.graphWarmupSteps` | Consecutive same-shape decode steps that run directly before the trunk is recorded for replay ([decode-step command replay](#decode-step-command-replay)) | 2 |
| `neuralfree.gpu.patchedReplay` | Let a verified decode or paged decode capture replay without re-running its dispatches, patching the per-step values in place; `false` keeps every capture on the body replay ([decode-step command replay](#decode-step-command-replay)) | true |
| `neuralfree.matvec.int8` | Single-token GPU projections through the Q8_1 integer-dot matvec: `auto` where the device reports packed dot products, `on` wherever the kernels run, `off` for the F32 matvec ([integer-dot decode matvec](#integer-dot-q8_1-decode-matvec)) | `auto` |
| `neuralfree.gpu.fused` | Fused GPU decode kernels: `auto` runs a fused kernel where the bound backend provides it, `on` requires them (binding fails naming the missing kernel and backend), `off` runs only the unfused pairs | auto |
| `neuralfree.gpu.stagingMb` | CUDA and Vulkan staging ring size in MiB (CUDA 1-1024, Vulkan 1-2047) | upload 64, readback 16 |
| `neuralfree.vulkan.pipelineCache` | On-disk Vulkan pipeline cache (`~/.cache/neuralfree/vulkan/`) | true |
| `neuralfree.vulkan.validation` | Enable `VK_LAYER_KHRONOS_validation` | false |
| `neuralfree.vulkan.blockMb` | Vulkan device-memory block size in MiB | 256 |
| `neuralfree.hip.shim.path` | Absolute path of `libhipshim.so`, the HIP backend's JNI shim; unset loads `hipshim` from the library path | none |
| `neuralfree.vulkan.dispatchesPerSubmit` | Dispatches recorded per Vulkan command buffer | 256 |

## Troubleshooting

- **No backend.** `NoBackendAvailableException` lists the reason for every API that was tried. For LLM inference, the log names the backends skipped for the LLM kernel library. The usual causes are CUDA kernels or Vulkan shaders that were not built; see [GPU kernel tools](#gpu-kernel-tools-all-optional).
- **Training runs on the CPU.** When the train kernels are unavailable on the chosen backend, `createTrainingBackend` logs a WARN with the reason for each API, and `describeTrainKernels` reports `unavailable (tensor ops take their CPU path): <reason>`. Build the missing CUDA modules or SPIR-V, or choose another API with `--backend` or `neuralfree.backend`.
- **Device memory.** For LLM inference:
  - lower `--ctx`;
  - use a quantized KV cache (`--kv-type q8_0`);
  - let the trunk stream (automatic context);
  - run `--vram-budget` to see the plan.

  In the training framework, tensors, modules, optimizers and data loaders hold device memory until they are closed; see [Tensor ownership](TRAINING.md#tensor-ownership).
- **OpenCL device that cannot open.** A device whose context cannot be created during enumeration (for example a GPU whose memory is exhausted by another process) is skipped, and the driver status, such as `CL_OUT_OF_RESOURCES`, is reported as the OpenCL initialization failure; the other backends are still tried.
- **Heap.** Large models with a host cache need a larger heap: `./gradlew runLLMDemo -Pheap=24576 ...`.
- **Temporary space.** The NVIDIA OpenCL compiler needs temporary space. `runLLMDemo` points `TMPDIR` and `CUDA_CACHE_PATH` at `build/tmp`.
- **Kernel caches.** To rebuild stale caches after a driver change, delete `~/.cache/neuralfree/opencl` and `~/.cache/neuralfree/vulkan`. They are also disabled by `neuralfree.opencl.binaryCache=false` and `neuralfree.vulkan.pipelineCache=false`.
- **No log output.** Outside the Gradle run tasks, Log4j prints errors only until it is given a configuration; see [Logging](#logging).
- **Windows GPU timeouts.** Windows' timeout detection and recovery (TDR) resets a GPU whose work runs longer than `TdrDelay` seconds, two by default. The application then sees a device-lost or out-of-resources error, and the screen may flicker while the driver recovers. Long dispatches on a slow or busy GPU, such as a large prompt on an integrated GPU, can exceed it. To raise the delay, set `TdrDelay` from an elevated command prompt and restart Windows:

  ```bat
  reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
  ```

  To restore the default, delete the value and restart:

  ```bat
  reg delete "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /f
  ```

  Do not set `TdrLevel` to 0: that disables recovery, so a hung GPU freezes the whole system. Microsoft documents these values in [TDR registry keys](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys).
