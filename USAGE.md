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
   - [GPU kernel tools (all optional)](#gpu-kernel-tools-all-optional)
   - [Kernel libraries](#kernel-libraries)
   - [Kernel inventories](#kernel-inventories)
   - [Logging](#logging)
2. [Compute backends](#compute-backends)
3. [The training framework on a GPU backend](#the-training-framework-on-a-gpu-backend)
4. [LLMDemo command-line reference](#llmdemo-command-line-reference)
5. [Hugging Face checkpoints](#hugging-face-checkpoints)
   - [Flags and properties](#flags-and-properties)
   - [Materialization of derived tensors](#materialization-of-derived-tensors)
   - [Activation quantization](#activation-quantization)
   - [Model sampling defaults](#model-sampling-defaults)
   - [Supported quantization formats](#supported-quantization-formats)
   - [Rejected formats](#rejected-formats)
   - [Supported architectures](#supported-architectures)
6. [Hugging Face Hub download](#hugging-face-hub-download)
7. [nftool](#nftool)
8. [HTTP server](#http-server)
   - [Tool calls](#tool-calls)
   - [API keys](#api-keys)
   - [OpenAI Responses](#openai-responses)
9. [Anthropic Messages](#anthropic-messages)
10. [Reasoning and tool-call parsers](#reasoning-and-tool-call-parsers)
11. [Grammar-constrained generation](#grammar-constrained-generation)
12. [Prefix cache](#prefix-cache)
13. [Paged KV memory](#paged-kv-memory)
14. [Control vectors](#control-vectors)
15. [imatrix](#imatrix)
16. [Session files](#session-files)
17. [System properties](#system-properties)
18. [Troubleshooting](#troubleshooting)

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
| `./gradlew verifyGpuKernels` | Builds every GPU kernel artifact, syntax-checks the OpenCL program of every kernel library with clang, and checks that CUDA and Vulkan cover each library's kernel inventory |
| `./gradlew inventoryCheck` | Checks every kernel library's inventory against its kernel sources; `build` runs it before packaging the resources |
| `./gradlew updateKernelInventory` | Regenerates the checked-in inventory of every kernel library from its kernel sources (needs clang or nvcc) |
| `./gradlew publishToMavenLocal` | Installs `com.eonmux:neuralfree:1.0.0-SNAPSHOT` in the local Maven repository; the POM carries the JCuda and LWJGL natives of the build platform |

**Choosing the demos' backend.** `DemoRunner` takes `--backend auto|opencl|cuda|vulkan`; without it the `neuralfree.backend` property decides, and `auto` is the default. `runDemos` passes program arguments with Gradle's `--args`, and every Gradle run task forwards `neuralfree.*` properties to the program's JVM:

```bash
./gradlew runDemos --args='--backend cuda'
./gradlew runDemos -Pneuralfree.backend=vulkan
```

`cpu` is rejected for the framework: its tensors live on a GPU backend, and each op takes its CPU path when the train kernels are unavailable there. On an invalid value `DemoRunner` prints the usage and exits with status 2.

### GPU kernel tools (all optional)

OpenCL needs no build tool: the driver compiles the kernel sources at run time. The CUDA modules and the Vulkan SPIR-V of every kernel library are built by `gradle/gpu-kernels.gradle` during `processResources`. Each tool is found in this order:
1. its `-P` property;
2. the listed directories;
3. `PATH`.

| Tool | Used by | Found in | When it is missing |
|---|---|---|---|
| `nvcc` | `compileCudaKernels`; `inventoryCheck` and `updateKernelInventory` when clang is missing | `-Pnvcc`, `$CUDA_HOME/bin`, `$CUDA_PATH/bin`, `/opt/cuda/bin`, `/usr/local/cuda/bin` | Warning; no CUDA modules are built. The CUDA backend reports both kernel libraries unavailable. |
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
| `-Pnvcc=<path>`, `-Pneuralfree.glslc=<path>`, `-Pneuralfree.clang=<path>` | Tool locations, searched before the default directories | none |
| `-Pnvcc.ccbin=<path>` | Host compiler passed to nvcc as `-ccbin` | none |
| `-Pneuralfree.gpu.libraries=<ids>` | Comma-separated kernel library ids (`llm`, `train`) that the kernel tasks build and check. For developer speed only: the other libraries' CUDA and Vulkan artifacts are then missing from the build, and the build warns. An unknown id fails the build. | all |

### Kernel libraries

The GPU kernels are organized in two kernel libraries, each built from one set of dual-dialect sources into OpenCL C (compiled by the driver at run time), CUDA modules and Vulkan SPIR-V:

| Library | Serves | Sources | Build profile |
|---|---|---|---|
| `llm` | The GGUF engine | `kernels/llm`, `shaders/llm` | Fast-math CUDA (`-ftz=true -prec-div=false -prec-sqrt=false -fmad=true`), subgroup variants |
| `train` | Tensors, training and classical ML | `kernels/train`, `shaders/train` | Precise CUDA math (`-ftz=false -prec-div=true -prec-sqrt=true -fmad=false`), OpenCL `-cl-std=CL1.2` with no relaxed-math option, no subgroup variants (reductions use a fixed-order tree on every backend) |

Paths are under `src/main/resources`. `kernels/<lib>/manifest.txt` lists a library's sources in build order. The headers both libraries share are in `kernels/common` (for example `common/dialect.h`, listed as a `[shared]` manifest entry) and `shaders/common` (`nf_base.glsl`); `shaders/test` holds the backend test shaders. Every other compute shader belongs to a kernel library, and `compileShaders` rejects one outside them. The generated artifacts go to `build/generated/gpu/{cuda,spv,spv-test,check}` and are merged into the resources.

What each API needs to run a library:

| API | `llm` | `train` |
|---|---|---|
| OpenCL | An OpenCL 1.2 device | An OpenCL 1.2 device |
| CUDA | Modules built with nvcc covering the library's inventory; a SASS image for the device's major architecture (minor not above the device's) or a PTX image of an architecture not above it; a driver new enough for those images | The same |
| Vulkan | SPIR-V built with glslc and spirv-dis covering the inventory; a Vulkan 1.2 device with 8-bit storage, 16-bit storage and timeline semaphores | SPIR-V covering the inventory; a Vulkan 1.2 device with timeline semaphores (sub-word data is accessed as 32-bit words) |

On every API the `train` library also needs work-groups of at least 256 items, and every kernel it lists must load. OpenCL programs are cached in `~/.cache/neuralfree/opencl` and Vulkan pipelines in `~/.cache/neuralfree/vulkan/`.

### Kernel inventories

`src/main/resources/kernels/llm/inventory.txt` and `src/main/resources/kernels/train/inventory.txt` list every kernel that CUDA and Vulkan must provide for each library, with its argument layout. They are generated from the kernel sources and checked in. A library's CUDA and Vulkan sides are available only when their built kernels cover its inventory.
- **`inventoryCheck`** runs before `processResources`, so `build` runs it. With clang, or nvcc's preprocessor when clang is missing, it checks the kernels' argument limits and that the shader-variant rows match the kernels, regenerates each inventory in memory, and fails the build when a checked-in file differs.
- **Without clang and nvcc** it checks only each file's format, that each kernel's source is an untagged `.cl` entry of the library's `manifest.txt`, and that the shader-variant rows match the listed kernels. The build warns that the inventories' freshness was not checked and ships the files as checked in. The Vulkan backend still runs both libraries from them.
- **After a kernel change** that adds, removes or renames a kernel, or changes its source file or arguments, run `./gradlew updateKernelInventory` and commit the regenerated file. Until then, `inventoryCheck`, and with it the build, fails whenever clang or nvcc is present, and names the regeneration command.

### Logging

NeuralFree logs through the Log4j 2 API; `log4j-core` is its runtime implementation.
- **The Gradle run tasks** (`runApp`, `runDemos`, `runLLMDemo`) pass `-Dlog4j2.configurationFile=src/main/config/log4j2.xml`, which logs INFO and above to the console.
- **The library jar ships no logging configuration.** An application configures Log4j itself; without a configuration, Log4j's default prints errors only. To use the run tasks' configuration with `java`, pass `-Dlog4j2.configurationFile=<checkout>/src/main/config/log4j2.xml`.
- **The tests** log INFO and above to the console through `src/test/resources/log4j2-test.xml`.

## Compute backends

NeuralFree drives GPUs through three APIs behind one `ComputeBackend` interface: OpenCL, CUDA and Vulkan. `Config.ACCELERATION_MODE` names the choice:

| Mode | Enumerates | Picks |
|---|---|---|
| `AUTO` | Every API | Device first, then API (below) |
| `VULKAN` | Vulkan only | The best Vulkan device |
| `CUDA` | CUDA only | The best CUDA device |
| `OPENCL` | OpenCL only | The best OpenCL device |

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
2. **API order.** Each API in `neuralfree.backend.order` (default `vulkan,cuda,opencl`) is tried on the target device. The first one that is accepted wins. The views one GPU offers through several APIs are recognised as one device.
3. **Fallback.** If no API runs the target, each API's own winner is tried, discrete first.

The per-API properties `neuralfree.opencl.device`, `neuralfree.cuda.device` and `neuralfree.vulkan.device` override `neuralfree.device` within one API. `neuralfree.backend.order=cuda,opencl`, for example, keeps AUTO off Vulkan.

**Requirements.** `createBackend(mode, requirement)` adds a condition the backend must meet: AUTO skips, closes and logs every backend the requirement rejects, and an explicit mode fails with `<API> backend unavailable: <reason>`.

**LLM inference** uses the requirement `KernelLibraryRequirement.of(KernelLibraries.LLM)`, so AUTO never selects a backend that cannot serve the LLM kernels. Choose the LLM backend with `--backend auto|opencl|cuda|vulkan|cpu`, `--cpu` or `--device <substr>`, or with the `neuralfree.backend` and `neuralfree.device` properties. If the GPU load fails, the model opens on the CPU engine.

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

### Model, prompt and output

| Flag | Meaning | Default |
|---|---|---|
| `--model`, `-m <path>` | GGUF file or Hugging Face checkpoint (directory or `.safetensors` file); see [Hugging Face checkpoints](#hugging-face-checkpoints) and [LLM-INFERENCE.md](LLM-INFERENCE.md#selecting-a-model) | see [LLM-INFERENCE.md](LLM-INFERENCE.md#selecting-a-model) |
| `--hf-repo`, `-hf <user>/<model>[:quant][@revision]` | Download a model from the Hugging Face Hub and open it; the quant defaults to `Q4_K_M`, then `Q8_0`. `HF_HUB_OFFLINE=1` uses the cache only. Excludes `--model` ([Hugging Face Hub download](#hugging-face-hub-download)) | none |
| `--hf-file`, `-hff <name>` | File inside the `--hf-repo` repository, replacing the default choice | none |
| `--prompt`, `-p <text>` | Run one prompt instead of the console chat | console chat |
| `--system <text>` | System prompt | `You are a helpful assistant.` |
| `--raw` | Use the prompt verbatim, without the chat template | off |
| `--thinking` | Enable the chat template's thinking mode | off |
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

### Backend and weights

| Flag | Meaning | Default |
|---|---|---|
| `--gpu` | GPU, automatic backend | on |
| `--cpu` | CPU engine | off |
| `--backend <type>` | `auto`, `opencl`, `cuda`, `vulkan` or `cpu` | `auto` |
| `--device`, `-device`, `-dev <substr>` | Use the device whose name contains `<substr>` | none |
| `--ram-cache-mb <n>` | Host cache of streamed layers in MiB; `-1` = half the maximum heap, `0` = none | -1 |
| `--vram-depth <n>` | Streamed-layer slots in VRAM; `0` = from the VRAM budget | 0 |
| `--no-mmap` | Read tensors into memory instead of mapping the file (sets `neuralfree.disable.mmap=true`) | mmap |

### Memory

| Flag | Meaning | Default |
|---|---|---|
| `--ctx`, `-c`, `--ctx-size`, `--max-context <n>` | Context length; `0` = fit to the budget | 0 |
| `--kv-type <t>` | KV cache type of K and V: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` | `f16` |
| `--kv-type-k`, `--cache-type-k`, `-ctk <t>` | K type only | `f16` |
| `--kv-type-v`, `--cache-type-v`, `-ctv <t>` | V type only | `f16` |
| `--ubatch`, `-ub`, `--ubatch-size <n>` | Tokens per forward step | 512 (`embed`, `classify` and `rerank`: max(2048, context) unless given) |
| `--swa-full` | Full-size caches for sliding-window layers | off |
| `--mla-cache <m>` | `auto`, `latent` or `decompressed` | `auto` |
| `--no-prefix-reuse` | Re-evaluate the whole prompt every turn | reuse on |
| `--context-shift` / `--no-context-shift` | Drop old positions when the context is full | off |
| `--no-kv-rotation` | Store quantized KV rows without Hadamard rotation | rotation on |

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

### Speculative decoding

| Flag | Meaning | Default |
|---|---|---|
| `--draft`, `-md`, `--model-draft <gguf>` | Drafter: a small model, an MTP draft file, eagle3, gemma4-assistant or dflash | none |
| `--mtp` | Draft with the target's own NextN/MTP blocks (excludes `--draft`) | off |
| `--spec-type <t>` | How drafting runs: `auto` (detected from the file); `draft-mtp` (alias `mtp`: the file's NextN/MTP blocks; without `--draft`, the target's own); `draft-simple` (aliases `simple`, `model`: a separate draft model); `ngram-simple` (alias `prompt-lookup`); `ngram-map` (aliases `ngram-map-k`, `map-k`, `ngram-map-k4v`, `map-k4v`, `ngram-mod`, `mod`); `ngram-cache`; `suffix` — the n-gram and suffix drafters draft from the prompt and the output and need no draft file | `auto` |
| `--draft-max`, `--draft-n-max`, `--spec-draft-n-max <n>` | Maximum drafted tokens per round | 3 |
| `--draft-min`, `--draft-n-min`, `--spec-draft-n-min <n>` | Minimum draft length that is verified | 0 |
| `--draft-p-min`, `--spec-draft-p-min <p>` | Stop drafting below this probability | 0 |

How drafted tokens are verified is programmatic only: `SpeculativeDecoder.Builder.verifyMode(...)` selects the mode (`match`, `rejection`, `block`, `fly`, `synthetic`). No command-line flag exposes it.

### LoRA adapters

| Flag | Meaning |
|---|---|
| `--lora <gguf[:scale]|PEFT dir>` | Apply adapters, scale 1 unless given; repeatable. A PEFT directory (`adapter_config.json` plus `adapter_model.safetensors`) attaches to a GGUF or Hugging Face base model |
| `--lora-scaled <gguf>:<scale>[,...]` or `--lora-scaled <gguf> <scale>` | Apply adapters with explicit scales; repeatable |
| `--lora-base-config <dir>` | An HF directory whose config the adapter's GGUF base takes its hyperparameters from |
| `--hf-base-dir <dir>` | Local base model a decision-LoRA checkpoint (PEFT checkpoints trained with llama.cpp's `lev` conversion) merges at load |

Adapters given on the command line replace the `neuralfree.lora` list.

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
| `--audio-out <file>` | Write audio features or hidden rows as float32 little-endian | |

After each generation, the demo prints timing statistics. With `--stats-file <path>` it also appends the session's statistics (model, backend, memory plan, and the timing, prompt and response of each generation) to `<path>` when the session ends; a relative path is resolved against the working directory, which is the project directory under `runLLMDemo`. Without the flag nothing is written.

## Hugging Face checkpoints

The engine opens a Hugging Face checkpoint wherever it opens a GGUF file. Format detection (`ModelSources.detect`) is content-based:

1. Regular files: a file whose first 4 bytes are `GGUF`, else a file ending in `.safetensors`. An HF source file is a single `*.safetensors` whose parent directory holds `config.json`, `params.json` (Mistral native) or a checkpoint-detector file; only that file's tensors are used. A file the parent index lists as a shard is refused with a message to pass the directory instead.
2. A directory containing `config.json` plus `*.safetensors` files — or `params.json` plus `consolidated*.safetensors` — is an HF checkpoint. Line endings, dtype names, index files (`*.safetensors.index.json`) and per-shard files are all handled; encoder-decoder, diffusion, embedding, reranker, audio and drafter architectures load like their GGUF counterparts.
3. A directory without safetensors files that holds exactly one `.gguf` file, or one split set `*-00001-of-NNNNN.gguf`, opens that GGUF (the `gguf` load format vLLM and SGLang use).
4. A directory with `adapter_config.json` and `adapter_model.safetensors` is a PEFT adapter: it attaches to a base model through `--lora` (`ModelSources.openAdapter`), and as a model it is refused with a message saying so.
5. Only `pytorch_model*.bin`, `*.pt` or `*.pth` files: a torch pickle checkpoint, which loads through a restricted unpickler (`TorchCheckpointFactory` in `CheckpointContainers`); unsafe opcodes and non-tensor payloads are refused by name at load ([GAPS.md](GAPS.md)).

A parsed HF checkpoint becomes the same `ModelSource` view a GGUF file produces ([LLM-INFERENCE.md](LLM-INFERENCE.md#model-sources)), so `--backend`, `--ctx`, `--kv-type`, `--mtp`, `--lora`, `--draft` and the sampler flags mean the same thing on both paths. On disk it needs nothing beside the checkpoint: the derived tensor cache below points at `~/.cache/neuralfree/hf` and is created only when derivations spill.

### Flags and properties

These flags set the matching `ModelSourceOptions` or `EngineConfig` fields. An enum value is parsed case-insensitively, with `-` and `_` equivalent; an unknown value is rejected naming the property. Property defaults apply when the flag is absent.

| Flag | Property | Default | Meaning |
|---|---|---|---|
| `--hf-materialize auto\|memory\|disk\|none` | `neuralfree.hf.materialize` | `auto` | Where derived tensors live ([Materialization](#materialization-of-derived-tensors)) |
| `--hf-memory-max <n[K\|M\|G]>` | `neuralfree.hf.materialize.memory.max` | `2G` | Largest total of derived bytes `auto` keeps in direct memory |
| `--hf-cache-dir <dir>` | `neuralfree.hf.cache.dir` | `~/.cache/neuralfree/hf` | Parent of the per-process directories of derived tensors written to disk |
| `--hf-outtype source\|auto\|f32\|f16\|bf16\|q8_0` | `neuralfree.hf.outtype` | `source` | Type of the view's float tensors. `auto` reproduces the converter's `--outtype auto`; the other values, except `source`, reproduce the converter's fixed outtypes; `q8_0` quantizes through the online quantizer |
| `--quant-online none\|q8_0\|q4_0\|q4_1\|q5_0\|q5_1\|iq4_nl` | `neuralfree.quant.online` | `none` | Online quantization of float matmul weights with llama-quantize eligibility |
| `--hf-quant-exact` | `neuralfree.hf.quant.exact` | off | Prefer bit-exact weight types over close repacks ([Supported quantization formats](#supported-quantization-formats)) |
| `--hf-no-mtp` | `neuralfree.hf.mtp` | true | The flag keeps the model's MTP/NextN layers out (the converter's `no_mtp`); the property set to `true` includes them again |
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

Under `checkpoint`, numerics agree with vLLM's integer and FP8 GEMMs: the quantizer's codes are reproduced operation for operation (scales, rounding, saturation and the 128-value groups), and the products of the coded weights and codes are f32-exact, so results agree up to f32 accumulation rounding. **No speedup is claimed**: this is an accuracy/reproduction mode, and no int8 or FP8 tensor-core kernels exist (`GAPS.md`).

The checkpoint's scheme for each weight, its static parameters and model-level bound (`neuralfree.act_quant.version`, `.weight_names`, `.schemes`, `.scale_ub`, and sibling `.input_scale` / `.input_zero_point` tensors) are read from the parsed view on both paths. Where the checkpoint quantizes its *output* activations, the load reports it and refuses under `checkpoint` with a named GgufFormatException — the reference's output activation quantization is not emulated; use `weight-only`. It is carried and dropped with one WARN under `weight-only`.

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
| FP8 E4M3, per-tensor or per-expert scale, on scaled bases | K % 32 == 0 | F8_E4M3 (zero-copy) + `.scale` | a | — |
| FP8 E4M3, block 128×128 (`weight_scale_inv`), UE8M0 scales, per-channel, or DeepSeek-V4's `.scale` | K % 128 == 0 | F8_E4M3_B128 | a | float fallback |
| FP8 E5M2, per-tensor scale on scaled bases | — | F16 (code shifted) + `.scale` | a | float fallback |
| FP8 E4M3FNUZ / E5M2FNUZ | — | F32 | a | — |
| MXFP8 (CT `W8A8Mxfp8`, ModelOpt, `mxfp8`) | K % 32 == 0 | MXFP8 | a | — |
| MXFP6 E2M3 / E3M2 (Quark OCP-MX) | every block scale in bf16 range | BF16 | a | larger scales: rejected, naming tensor and block |
| NVFP4 (CT, ModelOpt, `petit_nvfp4`, `modelopt_fp4`, fp_quant `nvfp4`) | K % 64 == 0 | NVFP4 with UE4M3 sub-scales plus the global `.scale` | a | other bases: float fallback |
| MXFP4 (CT, Quark, `quark_mxfp4`, gpt-oss `*_blocks`/`*_scales`, DeepSeek-V4 experts, fp_quant `mxfp4`) | K % 32 == 0 | MXFP4 with E8M0 scales; gpt-oss gate/up rows de-interleaved | a | — |
| bnb NF4 / FP4 (`bitsandbytes`), including double quant | `K % 64 == 0`, blocksize a multiple of 64 | NF4 / FP4_BNB with the absmax replicated per 64-run | a (a~ when the checkpoint's `quant_map` differs from the codebook within 1e-4) | rows-crossing layouts or blocksize 32: float fallback; different `quant_map`: F32 under `--hf-quant-exact` |
| Hadamard input transforms (CT `transform_config`, fp_quant) | signed Sylvester transform, verified arch | the PRISM input-transform keys (block size, sign mode); carried through to the GPU | a | non-Hadamard type or unverified arch: named error |
| Unquantized F32/F16/BF16/F64 | — | the outtype chosen by `--hf-outtype` | a | — |

New weight types behind the six extension ids (`F8_E4M3`, `F8_E4M3_B128`, `MXFP8`, `I8_B128`, `NF4`, `FP4_BNB`) run natively on the CPU and the GPU, like the extension type details under [LLM-INFERENCE.md: Weight types](LLM-INFERENCE.md#weight-types).

### Rejected formats

These are refused at load with a named error (`GgufFormatException` for a config value, `IOException` for a container problem), each saying what it is and why:

- `checkpoint_format: marlin` — vLLM's Marlin repacking is a runtime format, not a checkpoint layout;
- compressed-tensors `sparsity_config` (2:4 and bitmask sparsity) — sparse matmul runtime feature;
- `quant_method`: `torchao`, `aqlm`, `eetq`, `vptq`, `spqr`, `higgs`, `quanto` — vLLM rejects, deprecates or has removed each; details in [GAPS.md](GAPS.md);
- Quark two-level weight specs (a `weight` given as a list) — layout only defined by the Quark runtime;
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

An HF checkpoint is recognized by its `config.json` `architectures[0]` name. The names below — 295 of them, over 154 concrete GGUF architectures and 8 resolved at load time — are the TEXT rows of `src/main/resources/hf/archs.tsv` (generated from llama.cpp's converter registrations; the llama.cpp commit is recorded in the file header). They are grouped by the GGUF architecture each resolves to. The MMPROJ rows of the same file (the vision and audio tower classes) are not loadable model classes (`GAPS.md`). A name absent below is refused with a message naming it.

| GGUF architecture | Names | `architectures[0]` values |
|---|---|---|
| `-` | 8 | `GraniteSpeechForConditionalGeneration`, `GraniteSpeechPlusForConditionalGeneration`, `MiniCPMV4_7ForConditionalGeneration`, `Ministral3ForCausalLM`, `Ministral3Model`, `Mistral3ForConditionalGeneration`, `Mistral3Model`, `NomicBertModel` — resolved (or refused) at load time from the rest of the config; their rows carry no fixed GGUF architecture |
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
| `llada` | 1 | `LLaDAModelLM` |
| `llada-moe` | 2 | `LLaDAMoEModel`, `LLaDAMoEModelLM` |
| `llama` | 17 | `Eagle3DraftModel`, `Eagle3LlamaForCausalLM`, `Eagle3Speculator`, `InternLM3ForCausalLM`, `IQuestCoderForCausalLM`, `JanusForConditionalGeneration`, `LlamaForCausalLM`, `LLaMAForCausalLM`, `LlamaForCausalLMEagle3`, `LlamaModel`, `LlavaForConditionalGeneration`, `MistralForCausalLM`, `MixtralForCausalLM`, `Sarashina2VisionForCausalLM`, `UltravoxModel`, `VLlama3ForCausalLM`, `VoxtralForConditionalGeneration` |
| `llama4` | 2 | `Llama4ForCausalLM`, `Llama4ForConditionalGeneration` |
| `llama-embed` | 1 | `LlamaBidirectionalModel` |
| `maincoder` | 1 | `MaincoderForCausalLM` |
| `mamba2` | 1 | `Mamba2ForCausalLM` |
| `mamba` | 3 | `FalconMambaForCausalLM`, `MambaForCausalLM`, `MambaLMHeadModel` |
| `maple` | 1 | `MapleForCausalLM` |
| `mellum` | 1 | `MellumForCausalLM` |
| `mimo2` | 2 | `MiMoV2FlashForCausalLM`, `MiMoV2ForCausalLM` |
| `minicpm` | 1 | `MiniCPMForCausalLM` |
| `minicpm3` | 1 | `MiniCPM3ForCausalLM` |
| `minimax-01` | 2 | `MiniMaxM1ForCausalLM`, `MiniMaxText01ForCausalLM` |
| `minimax-m2` | 1 | `MiniMaxM2ForCausalLM` |
| `minimax-m3` | 2 | `MiniMaxM3SparseForCausalLM`, `MiniMaxM3SparseForConditionalGeneration` |
| `mistral3` | 1 | `VoxtralRealtimeForConditionalGeneration` |
| `modern-bert` | 4 | `ModernBertDecisionModel`, `ModernBertForMaskedLM`, `ModernBertForSequenceClassification`, `ModernBertModel` |
| `mpt` | 1 | `MPTForCausalLM` |
| `muse-glimmer` | 1 | `MuseGlimmerForConditionalGeneration` |
| `muse-spark` | 1 | `MuseSparkForConditionalGeneration` |
| `nanbeige` | 1 | `NanbeigeForCausalLM` |
| `nemotron` | 1 | `NemotronForCausalLM` |
| `nemotron_h` | 1 | `NemotronHForCausalLM` |
| `nemotron_h_moe` | 1 | `NemotronHPuzzleForCausalLM` |
| `neo-bert` | 3 | `NeoBERT`, `NeoBERTForSequenceClassification`, `NeoBERTLMHead` |
| `olmo2` | 2 | `Olmo2ForCausalLM`, `Olmo3ForCausalLM` |
| `olmo` | 2 | `OlmoForCausalLM`, `OLMoForCausalLM` |
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
| `qwen3` | 3 | `Qwen3ForCausalLM`, `Qwen3Model`, `StepVLForConditionalGeneration` |
| `qwen35` | 11 | `H2OLightningModel`, `KevModel`, `LevModel`, `MiniCPMV4_6ForConditionalGeneration`, `NimbleModel`, `OpenJevModel`, `PplxDeciderModel`, `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5TextModel`, `VllmD3Model` |
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
| `step35` | 2 | `Step3p5ForCausalLM`, `Step3p7ForConditionalGeneration` |
| `t5` | 5 | `MT5ForConditionalGeneration`, `T5ForConditionalGeneration`, `T5WithLMHeadModel`, `UMT5ForConditionalGeneration`, `UMT5Model` |
| `t5encoder` | 1 | `T5EncoderModel` |
| `talkie` | 1 | `TalkieForCausalLM` |
| `vllm-decision2` | 1 | `VllmDecision2Model` |
| `wavtokenizer-dec` | 1 | `WavTokenizerDec` |
| `xverse` | 1 | `XverseForCausalLM` |

## Hugging Face Hub download

`--hf-repo <user>/<model>[:quant][@revision]` (short `-hf`) downloads a model from the Hugging Face Hub and opens it; it excludes `--model`. The `:quant` defaults to `Q4_K_M`, then `Q8_0`; `@revision` names a branch or commit. `--hf-file <name>` (short `-hff`) picks a concrete file in the repository instead of the default choice. `HF_HUB_OFFLINE=1` resolves from the cache only and refuses to download. Requests authenticate with the `HF_TOKEN` environment variable.

The cache root follows llama.cpp's environment precedence (`HubCache`): `LLAMA_CACHE`, then `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, and finally `~/.cache/huggingface/hub`. `nftool download` fills the same cache (below).

## nftool

`demo/tools/NfTool` is the maintenance CLI. Four subcommands are implemented:

| Subcommand | Flags | Meaning |
|---|---|---|
| `export` | `--mode native\|compat\|compat-exact`, `--overwrite`, `--vocab-only`, `--split-max-tensors <n>`, `--split-max-size <n[K\|M\|G]>` | Write a model's GGUF view, as `--export-gguf` does |
| `download` | the `--hf-repo` spec above | Download from the Hugging Face Hub into the cache root |
| `imatrix` | `--file <fname>` (repeatable), `--out-file <fname>`, `--chunk-size <n>` | Collect importance-matrix statistics over calibration text (below) |
| `cvector` | `--positive-file <fname>`, `--negative-file <fname>`, `--out-file <fname>`, `--method pca\|mean`, `--model-hint <arch>`, `--pca-batch <n>`, `--pca-iter <n>`, `--overwrite` | Train a control vector from prompt pairs over one loaded engine (below) |

Every other listed subcommand (`quantize`, `perplexity`, `results`, `bench`, `batched-bench`, `split`, `merge`, `export-lora`, `tokenize`, `dump`, `hash`, `embedding`, `retrieval`) refuses with a `not implemented yet` message naming the subcommand.

## HTTP server

`server/` serves the OpenAI-compatible surface over the batch engine, with the llama.cpp and vLLM spellings of the shared endpoints:

| Endpoint | Serves |
|----------|--------|
| `GET /v1/models` | The served model, with `max_model_len` |
| `POST /v1/completions` | Text completions; `stream: true` sends server-sent events ending in `data: [DONE]` |
| `POST /v1/chat/completions` | Chat completions through the model's template, with tools and streaming |
| `POST /v1/embeddings` | Pooled vectors from an embedding or pooling model, in OpenAI's body |
| `POST /embeddings` (`/embedding`) | llama.cpp's native embeddings: per-token rows without pooling |
| `POST /v1/rerank` (`/rerank`, `/v1/reranking`, `/reranking`) | Documents scored against a query by a rank model |
| `POST /tokenize`, `POST /detokenize` | Token ids of text or chat messages, and text of token ids |
| `POST /apply-template` | A chat body rendered through the template, as `prompt` |
| `GET /health` (`/v1/health`), `GET /props` | Load state; serving setup (model, context, slots, chat template) |
| `POST /v1/messages` (+ `/count_tokens`) | Anthropic's Messages API ([below](#anthropic-messages)) |
| `POST /v1/responses` (`/responses`), `GET`/`DELETE /v1/responses/{id}`, `GET .../input_items`, `POST .../cancel` | OpenAI's Responses API ([below](#openai-responses)) |
| `POST /v1/responses/input_tokens`, `POST /v1/chat/completions/input_tokens` (and unversioned) | The prompt token count of a Responses or chat request |
| `POST /completion` (`/completions`), `POST /infill` | llama.cpp's native completion and fill-in-the-middle ([below](#http-server)) |

Start it with the model path property (`-Dneuralfree.server.model=<gguf path>`; every other setting reads its `neuralfree.*` property) after `./gradlew :server:build` — the build produces the uber-jar under `server/build/`. Without the property the server starts unloaded: `/health` reports the error and every model request refuses with 503.

**Metrics.** `/metrics` serves Prometheus text with the JVM meters and the engine's own. The engine meters follow llama.cpp's set under a `neuralfree_` prefix: `prompt_tokens_total` (cached tokens excluded), `prompt_tokens_cached_total`, `prompt_seconds_total`, `tokens_predicted_total`, `tokens_predicted_seconds_total`, `decode_total`, `tokens_max`, `busy_slots_per_decode`, `requests_processing` and `requests_deferred`. Following vLLM, they add `preemptions_total`, `recomputed_tokens_total`, `kv_cache_usage` (the share of KV rows in use, with cached prefixes counted until evicted), `request_success_total{finished_reason="stop"|"length"|"abort"}` and `request_failures_total`. They read the engine's state after its latest step, read zero before a model loads, and a scrape never loads one. Throughput is `rate()` over the counters.

**Model names.** `-Dneuralfree.server.model.name=<name>[,<name>...]` sets the names the server answers to, like vLLM's `--served-model-name` and llama.cpp's `--alias`. `/v1/models` lists each name with the model path as its `root`, and `/props` reports the first name as `model_alias`. A response carries the `model` its request named when that name is served, and the first name otherwise. With names configured, a request that names any other model is refused with 404, as in vLLM. Without them the model path is the only name and any requested name is accepted, as in llama.cpp.

One `EngineService` opens the model and pumps its `BatchEngine` on one thread; requests queue FIFO and admitted sequences decode together over the paged memory (`neuralfree.parallel` slots, [scheduling](#paged-kv-memory)). A client that disconnects cancels its sequences, whether it was streaming or waiting for a whole response. A request that runs longer than `neuralfree.server.request.timeout` (seconds, default 1800) is cancelled and refused with 503, and an embedding or rerank call that waits that long for an idle engine is refused too. `/models` and `/chat/completions` answer as llama.cpp's unversioned aliases of the OpenAI routes.

**Native completion.** `POST /completion` and `/completions` serve llama.cpp's native completion; its OpenAI form stays at `/v1/completions`. A `prompt` is a string, an array of token ids, an array that mixes strings and token ids (one prompt, with BOS only before a leading string), or `{"prompt_string": ...}`. An array that holds no token id at all is several prompts. `n_predict` sets the budget (-1 is unlimited), `n_probs` adds `completion_probabilities` (each with `id`, `token`, `bytes`, `logprob`, `top_logprobs`), and `response_fields` picks nested fields by `a/b` paths. Every sampler and grammar field above applies too. One prompt answers one object with `content`, `tokens`, `id_slot`, `stop_type` (`eos`, `word`, `limit`), `stopping_word`, `tokens_predicted`, `tokens_evaluated`, `tokens_cached`, `generation_settings` and `timings`; several prompts answer an array of these. A stream sends one `data:` chunk per generated token that released text (`content`, `tokens`, `stop: false`, `tokens_predicted`) and ends each prompt with its final object, with no `[DONE]`. Text held back for a possible stop string arrives in the chunk of the token that releases it. `post_sampling_probs` is refused, because the probabilities are the model's own, taken before the sampler chain.

**Embeddings.** `/v1/embeddings` tokenizes text with the vocabulary's special tokens (a BERT model's CLS and SEP; `add_special_tokens: false` leaves them out) and returns L2-normalized vectors, as llama.cpp and vLLM do. `embd_normalize` picks llama.cpp's mode (-1 none, 0 max absolute in the int16 range, 1 taxicab, 2 euclidean, larger for a p-norm), and `use_activation` or `normalize` set to false return the raw pooled vector. `dimensions` keeps the first n values and normalizes after the cut, vLLM's matryoshka truncation. `truncate_prompt_tokens` keeps the last n tokens (-1 for the context length). `encoding_format: "base64"` writes float32 bytes, little-endian unless `endianness` is `big`. A model without pooling is refused there. Its per-token rows come from llama.cpp's native `POST /embeddings` (and `/embedding`), which takes `input` or `content` in llama.cpp's prompt forms and answers an array of `{"index", "embedding"}`: `[vector]` for a pooled model, one unnormalized row per token otherwise.

**Infill.** `POST /infill` answers like `/completion`, but over llama.cpp's fill-in-the-middle prompt. `input_prefix` and `input_suffix` are required (a string, or an array mixing strings and token ids). `input_extra` is an optional array of `{"filename", "text"}` context chunks, and `prompt` is an optional string appended to the prefix. The prompt follows the repo-level FIM layout: `[FIM_REP]` and `[FIM_SEP]` file headers when the vocabulary has those tokens, then `[FIM_PRE]prefix prompt[FIM_SUF]suffix[FIM_MID]`. The prefix keeps its last three quarters of a 2048-token batch, the suffix fills the remaining quarter, and the extra context gets what the context has left after the batch and twice `n_predict`. `-Dneuralfree.server.spm.infill=true` puts the suffix before the prefix, like llama.cpp's `--spm-infill`. A model without FIM prefix, suffix and middle tokens is refused with 501.

**Sampling.** `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `repeat_penalty` (`repetition_penalty`), `presence_penalty`, `frequency_penalty`, `seed`, `stop`, `max_tokens` (`max_completion_tokens`), `min_tokens`, `ignore_eos` and `bad_words` map onto `SamplerConfig` and the request stages, and so do llama.cpp's sampler fields: `dynatemp_range`, `dynatemp_exponent`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `repeat_last_n`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers` (at most 64 distinct), `mirostat`, `mirostat_tau`, `mirostat_eta`, `adaptive_target`, `adaptive_decay`, `min_keep` and `samplers`. `samplers` is an array of stage names (`top-k`, `topk`, `nucleus` and llama.cpp's other spellings) or a string of stage characters (`dkypsmtxiea`). Only keys that are present and not null change the llama.cpp defaults, and an out-of-range value or an unknown sampler is refused with 400. `logit_bias` maps onto the sampler's first stage (`BiasStage`): token id to an additive bias in [-100, 100], where -100 and below ban the token.

**Choices and prompts.** `n` (1 to 128) decodes that many choices side by side; with a `seed`, choice `i` samples with `seed + i` (vLLM's parallel sampling), without one each choice draws its own. A completion's `prompt` takes every OpenAI form — a string, an array of strings, an array of token ids, or an array of token-id arrays (ids outside the vocabulary refuse) — and choice `p * n + i` answers prompt `p`; `echo` prepends the prompt. Usage counts each prompt once and sums the choices' tokens. In a stream, each chat choice opens with its own `role` delta; the last finish chunk carries the usage, or under `stream_options.include_usage` a final chunk with empty `choices` does.

**Logprobs.** Chat takes `logprobs: true` with `top_logprobs` (0 to 20) and answers `logprobs.content` entries (`token`, `logprob`, `bytes`, `top_logprobs`); completions take an integer `logprobs` and answer the legacy `tokens`, `token_logprobs`, `top_logprobs` and `text_offset` arrays; llama.cpp's `n_probs` selects the count on either. A stream carries each token's entry on its chunk.

**Output constraints.** One of `grammar`, `json_schema`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `guided_structural_tag` / `structural_tag`, `response_format` (`text`, `json_object`, `json_schema`) or vLLM's `structured_outputs` (`json`, `regex`, `choice`, `grammar`, `json_object`, `structural_tag`) constrains the output ([Grammar-constrained generation](#grammar-constrained-generation)); two at once refuse. A regex matches the raw output, with or without `^`/`$` anchors. In chat completions the constraint holds the answer rather than the reasoning before it. A thinking model whose prompt leaves `<think>` open writes freely until its end tag, and the constraint starts after that tag and any whitespace. gpt-oss is constrained after its `<|channel|>final` header (a `<|constrain|>` type may sit in it), and Muse after the header of its message to the user. Other models are constrained from the first token, as are text completions.

**Chat.** `messages` (string or text-part content, assistant `tool_calls` and `reasoning_content`, `tool` results), `add_generation_prompt`, `continue_final_message`, `enable_thinking`, `reasoning_effort` and `chat_template_kwargs` reach the template. Reasoning output streams as `reasoning_content` deltas and lands in `message.reasoning_content`. Image and audio parts refuse by name: the text chat path has no media encoder.

Finish reasons: `stop` (end of generation, stop string, cancellation), `length` (token budget, full context) and `tool_calls`.

### Tool calls

`tools` (`{"type": "function", "function": {name, description, parameters}}`) render through the template, and `tool_choice` picks how calls come back ([parsers](#reasoning-and-tool-call-parsers)):

| `tool_choice` | Effect |
|---------------|--------|
| `auto` (default) | The configured tool parser reads calls out of the output |
| `none` | The tools still render; nothing is parsed |
| `required` | A grammar forces a JSON array of `{name, parameters}` calls of the declared tools (one when `parallel_tool_calls` is false), as vLLM does |
| `{"type": "function", "function": {"name": ...}}` | A grammar forces the named function's `parameters` schema |

The forcing grammars start after the reasoning when the rendered prompt leaves a thinking block open (a lazy grammar triggered by the reasoning end tag). Calls answer as `message.tool_calls` (`id`, `type: function`, `function.name`, `function.arguments` as JSON text) with `content` null when the output was only calls, and stream as `delta.tool_calls` — the first delta of a call carries its `index`, `id`, `type` and `name`, the following ones its argument text. The finish reason is `tool_calls`, except for a named function, which finishes with `stop` as OpenAI's does. A forcing choice refuses alongside another output constraint, `required` or a named function refuses without `tools`, and a named function must be declared.

Harmony-family models read calls out of their message headers instead of a tool parser. gpt-oss calls arrive through the `to=functions.NAME` recipient with the message body as arguments; Muse Spark / Glimmer calls are the `<atem:function_calls>` block of a message addressed to a tool, each `<atem:invoke>` one call whose `<atem:parameter>` values are typed by the tool's schema. Other recipients (gpt-oss built-ins such as `browser.search` or `python`) are dropped. Under `required` or a named function a grammar in the model's dialect lets it write its reasoning and preamble messages first and then holds the turn to a call of the declared tools, as llama.cpp's gpt-oss and Muse formats do.

### API keys

Keys come from the `NEURALFREE_API_KEY` environment variable and the `neuralfree.server.api.key` property (comma-separated each), and from the file named by `neuralfree.server.api.key.file` (one key per line; blank lines and `#` comments skipped) — llama.cpp's `--api-key` / `--api-key-file` and vLLM's `--api-key`. With any key configured every endpoint requires `Authorization: Bearer <key>` or Anthropic's `x-api-key: <key>`, compared in constant time; a missing or wrong key answers 401 with the OpenAI error envelope. `/health`, `/v1/health`, the Quarkus `/q/` probes and CORS preflight requests stay open. `/metrics` needs the key, as in llama.cpp. A blank key or an unreadable key file fails the server at startup.

### OpenAI Responses

`POST /v1/responses` (and llama.cpp's `/responses`) serves OpenAI's Responses API, as llama.cpp and vLLM do, by converting the request into a chat-completions body and running it through the chat path. Every sampler, grammar, tool and reasoning setting therefore behaves as on `/v1/chat/completions`.

- **Request.** `input` is a string or an array of items: messages (`user`, `system`, `developer` with `input_text` and `input_image` parts; `assistant` with `output_text` and `refusal` parts), `function_call`, `function_call_output` and `reasoning` items. Consecutive assistant items merge into one assistant turn. `instructions` become the system message. `tools` of type `function` become chat tools; built-in tool types such as `web_search_preview` have no local equivalent and are skipped. `tool_choice` takes `auto`, `none`, `required` or `{"type": "function", "name": ...}`. `max_output_tokens` is the budget, `reasoning.effort` sets the template's effort, and `text.format` (`text`, `json_object`, `json_schema` with `schema`) constrains the answer the way `response_format` does. `top_logprobs`, or `include: ["message.output_text.logprobs"]`, adds logprobs to the output text, and `extra_params` merges any further chat field (`min_p`, `ignore_eos`, `grammar` and the rest). `input_file`, `input_audio`, `item_reference`, `conversation` and stored `prompt` templates are refused by name.
- **Response.** The Response object carries `reasoning` items (`reasoning_text` content), `message` items (`output_text` with its logprobs) and `function_call` items (`call_id`, `name`, `arguments`). The final output follows the tool parser's reading of the whole text, as the chat route's message does, so a call the finished text turned into content, or arguments completed after the last delta, appear as they end. `status` is `completed`, `incomplete` with `incomplete_details.reason` `max_output_tokens`, `cancelled` or `failed`. `usage` holds `input_tokens` (with `cached_tokens` from the prefix cache), `output_tokens` (with `reasoning_tokens`, the tokens inside the reasoning block) and `total_tokens`.
- **Stream.** `stream: true` sends `response.created` and `response.in_progress`, then for each item `response.output_item.added`, its deltas (`response.reasoning_text.delta`, `response.content_part.added` and `response.output_text.delta`, `response.function_call_arguments.delta`), their `.done` events and `response.output_item.done`, and ends with `response.completed`, `response.incomplete` (also for a cancelled response) or `response.failed`. Each event carries its `sequence_number`, and the last event carries the whole response.
- **Store.** `-Dneuralfree.server.responses.store=<count>` keeps up to that many responses, evicting the least recently used. It is off by default, as in vLLM, and with it off `store: true` keeps nothing. A stored response can be fetched with `GET /v1/responses/{id}`, its input listed with `GET /v1/responses/{id}/input_items` (`limit`, `order`, `after`) and removed with `DELETE`. `previous_response_id` continues its conversation, replaying the earlier input and output items but not the earlier `instructions`.
- **Background.** `background: true` needs the store. It answers at once with a `queued` response and runs the generation detached from the client. `GET /v1/responses/{id}` shows its progress, `GET ...?stream=true&starting_after=<n>` replays its events from that sequence number, and `POST /v1/responses/{id}/cancel` stops it and answers the cancelled response (it waits for the current decode step, at most the request budget). Only a background response can be cancelled.

`POST /v1/responses/input_tokens` and `POST /v1/chat/completions/input_tokens` (and their unversioned forms) return `{"object": "response.input_tokens", "input_tokens": n}` for the rendered prompt.

## Anthropic Messages

`server/` also serves Anthropic's Messages API (`AnthropicResource`, llama.cpp's `post_anthropic_messages`). `POST /v1/messages` renders the `system` prompt (string or text blocks) and the messages through the model's chat template: `user` content takes text and `tool_result` blocks (which become tool messages ahead of the user's text), `assistant` content takes text, `thinking` and `tool_use` blocks. `tools` (`name`, `description`, `input_schema`) and `tool_choice` (`auto`, `any`, `tool` with its `name`, `none`; `disable_parallel_tool_use`) route like the OpenAI tool choices; Anthropic's server tools (a `type` such as `web_search_20250305`) refuse by name. `thinking` (`{"type": "enabled"}` or `"disabled"`) sets the template's thinking switch. `max_tokens` (required, as Anthropic documents), `temperature`, `top_p`, `top_k` and `stop_sequences` apply.

The answer is the Anthropic body: `thinking`, `text` and `tool_use` content blocks (`id`, `name`, `input`), `stop_reason` `end_turn`/`max_tokens`/`stop_sequence` (with the matched `stop_sequence` named)/`tool_use`, and `usage.input_tokens`/`output_tokens`. With `stream: true` the stream carries Anthropic's event sequence: `message_start`, then per block `content_block_start`, `content_block_delta` (`thinking_delta`, `text_delta` or `input_json_delta`) and `content_block_stop`, then `message_delta` and `message_stop`. `POST /v1/messages/count_tokens` returns the rendered prompt's token count. Image and document blocks refuse by name.

## Reasoning and tool-call parsers

The engine splits reasoning from content with a reasoning parser and reads tool calls with a tool parser; both default to `auto`, which derives them from the chat template, and both take vLLM's parser names (case-insensitive, `-` and `_` alike).

| Property | Flag | Values |
|----------|------|--------|
| `neuralfree.reasoning.parser` | `--reasoning-parser` | `auto`, `none`, or a reasoning parser name |
| `neuralfree.tool.parser` | — | `auto`, `none`, or a tool parser name |

Reasoning parsers (aliases in parentheses): `deepseek_r1` (`step5`), `deepseek_v3` (`deepseek_v31`), `deepseek_v4` (`deepseek_v41`, `deepseek_v32`), `qwen3` (`qwen3_5`, `qwen3_coder`), `glm45` (`glm47`, `glm4_moe`), `kimi_k2`, `kimi_k3`, `k2_horizon`, `minimax_m2` (`minimax_m2_append_think`), `minimax_m3`, `ernie45`, `hunyuan_a13b` (`hy_v3`, `hy_v4`), `olmo3`, `step3` (`step3p5`), `nemotron_v3`, `mimo`, `poolside_v1`, `holo2`, `ling3`, `seed_oss`, `mistral`, `gemma4`, `granite`, `cohere_command3` (`cohere_command4`), `openai_gptoss` (`gpt_oss`, `harmony`), `muse` (`muse_glimmer`, `musespark1x`) and `none`.

Tool parsers: `hermes` (`qwen2_5`, `qwen3_json`), `ernie45`, `longcat`, `qwen3_coder` (`qwen3_xml`, `qwen3`, `qwen3_5`, `nemotron_v3`), `seed_oss`, `minimax_m2` (`minimax`), `glm45` (`glm47`, `glm4_moe`), `llama3_json` (`llama4_json`), `mistral`, `deepseek_v3` (`deepseek_r1`), `deepseek_v31`, `kimi_k2`, `internlm` (`internlm2`), `jamba` (`apriel`, `hunyuan_a13b`), `granite` (`granite_3`), `phi4_mini_json` (`firefunction`, `phi4mini`), `xlam`, `pythonic`, `llama4_pythonic`, `lfm2` (`lfm2_5`) and `olmo3`. `auto` detects the format from the template's call markers; a template it does not recognize parses no calls.

The tool parsers stream: text that may begin a call is held back until it either becomes a call (its name, then its argument text as it grows) or turns out to be content. An unknown name refuses at configuration with the registered names listed. More parsers plug in through `ToolParserProvider` and `ReasoningParserProvider` (`ServiceLoader`); a name that collides with a registered one refuses.

## Grammar-constrained generation

`gguf/grammar` constrains sampling to a grammar: a `GbnfBuilder`/`GbnfParser` pair runs GBNF grammars (`Grammar`, `GrammarLimits`), and a `ConstraintSpec` selects the kind — `GBNF`, `JSON_SCHEMA` (compiled by `JsonSchemaToGbnf`), `JSON_OBJECT`, `REGEX` (compiled by `RegexToGbnf`), `CHOICE`, `STRUCTURAL_TAG` (the vLLM legacy payload: `StructuralTag` compiles each structure's begin literal, schema grammar and end literal into one alternated grammar, dispatched by the payload's triggers) — and the constraint runs as the `GrammarStage` of the sampler chain (`Sampler.create(config, bannedTokens, extra, vocab)`), which masks every token the grammar cannot accept next and advances on each accepted token; a completed rule admits the end-of-generation tokens again. As in llama.cpp, a token matches by its piece rendered with special tokens shown, so the literal `"<|channel|>"` matches that control token, and a token reference matches by id: `<[42]>`, or `<|channel|>` written as a token name, which the batch engine resolves through the model's vocabulary (the name must tokenize to exactly one token); `!<...>` matches any token but that one.

A spec can be trigger-gated (`ConstraintSpec.withTriggers(GrammarTriggers)`, llama.cpp's lazy grammars): generation runs unconstrained until a trigger fires — a word, a regex whose first non-empty capture group (or the match) starts the constrained text, a regex the whole output must match, or a token id — and the grammar then constrains from the trigger start on, replaying the tokens that overlapped it (`TriggeredGrammarStage`). `LAZY` triggers hold the grammar to the end of generation; `DISPATCH` triggers (a structural tag's) return to free text after each completed structure and wait for the next trigger. While waiting for a word trigger, the stage masks tokens that would write a trigger's beginning followed by text the grammar refuses. The engine routes it through `BatchEngine.submit(..., constraint)` (compiled at submit, so an invalid spec throws to the caller) and the server exposes it in the llama.cpp, vLLM and OpenAI spellings ([HTTP server](#http-server)).

## Prefix cache

The batch engine reuses computed prompts: a retired sequence stays in the paged memory with its pages, and `BatchEngine` indexes those sequences by their exact token arrays (`PrefixCache`, bounded by the slot count). When a submitted prompt extends a cached prefix, the engine adopts the cached sequence in place — its slot, its pages, its divergent tail cut at the reused length — and prefills only the prompt's tail (at least the last prompt token always prefills, so its logits start the generation); `BatchSequence.reusedPrefix()` reports the copied token count. Cache entries match exact tokens, ties resolve to the most recently used, and a new generation for a slot evicts its cached sequence. The route follows `neuralfree.prefix.reuse` (default true) and is refused for windowed (SWA) memories, whose trim rules make a cached prefix's rows stale. Same tokens compute the same pages, so a reused run produces the same tokens and text as the uncached run (`BatchPrefixCacheTest`). Cached sequences hold pages, so admission evicts them least recently used first when a new prompt needs the cells.

## Paged KV memory

`gguf/memory/paged` keeps the rows of a KV group in reference-counted pages instead of one buffer per sequence: `PagePool`, `PageList`, `PagedSequences`, `PagedStep`, `PageCopy` and `RowParts`. It is property-gated, not the engine default: `neuralfree.kv.paged=true` forces it for a single sequence, and `neuralfree.parallel > 1` selects it; `neuralfree.kv.page.size` (default 16, a power of two in [1, 256]) sizes the pages and `neuralfree.kv.unified` (default true) shares the whole context between the sequences. Without the properties, `ModelMemory.sequences()` returns `SingleSequenceMemory`. On the CPU path, `ForwardBatch` runs a batch of sequences over the paged memory — one-sequence ubatches (llama.cpp's `split_seq`), all-or-nothing page reservation per ubatch, K/V stored at the step's `cellIdx` rows and attention through the step's page tables ([LLM-INFERENCE.md](LLM-INFERENCE.md#paged-kv-memory) has the details; the GPU path refuses).

The batch engine schedules over the shared pools as vLLM's scheduler does. A queued prompt is admitted when a slot is free and the free cells cover the part of the prompt it does not reuse plus the admitted prompts still waiting for their prefill. Running out of pages is recoverable: a prefill that finds the pools full rolls back and queues again, and a decode step that finds them full rolls back, then evicts a cached prefix or preempts the youngest running sequence. A preempted sequence keeps its sampler, grammar and streamed text; once a running sequence retires it is re-admitted ahead of the queue, prefills its prompt and generated tokens again, and continues, so its result is the one it would have produced alone. A request that cannot fit even with every other sequence's memory freed fails by name.

## Control vectors

`ops/heads/ControlVector` (with `ControlVectorExtra` and `CpuControlVectorExtra`) applies a control vector to the hidden states at inference. `nftool cvector` trains one (`CvectorGenerator`, llama.cpp's cvector-generator): it opens the model on the CPU engine, tokenizes every positive/negative prompt pair with the vocabulary's BOS policy, pads both sides to one length with the pad token llama.cpp derives (the last token of tokenizing a single space), and per pair prefills the positive side from position 0, switches the tap side, prefills the negative side and keeps the per-layer differences (the rows whose every value stays within 1e-6 drop, as llama.cpp's `filter_nonzero_rows` drops them). The tracked layers are all decoder layers except the last. The directions reduce by PCA power iteration or the row mean, and the export writes the llama.cpp `controlvector` GGUF form (`direction.<il>` tensors, `controlvector.model_hint` from `--model-hint` or the model's architecture).

## imatrix

`nftool imatrix` collects importance-matrix statistics over a calibration run (llama.cpp's imatrix tool): it opens the model on the CPU engine, installs the global matmul-input tap (`CpuMatVec.observeGlobally` with `cpu/quant/ImatrixCollector`), prefills every non-empty line of every `--file` from position 0 (cache reset between lines) and writes the statistics through `quant/write/ImatrixFile.writeGguf` — per weight tensor the column sums of the squared activations (`<tensor>.in_sum2`) and the activation-row counts (`<tensor>.counts`), one matrix per expert of a MoE tensor, keyed by the stored GGUF names. The chunk counters record llama.cpp's accounting: `--chunk-size` (default 512) and `chunk_count` as `ceil(tokens / chunk-size)` over the lines. `ImatrixFile` also reads both llama.cpp forms, GGUF and legacy `.dat`, so a matrix either tool collected combines into a run.

## Session files

`HeapModelMemory.saveSession(path)` / `loadSession(path)` save and restore a decoding sequence, llama.cpp `llama_state_save_file`/`llama_state_load_file` style (`memory/cpu/SessionFile`): the stored tokens with their reuse frontier and wrap flag, every KV group's position mirror, the K/V row values of every owning cache layer and the live recurrent state tensors. A load checks the whole plan identity first (context, ubatch, group and layer geometry, cache dtypes, head shapes, tensor names), resets the memory and writes everything back, so the next decode continues the saved sequence with the same logits. The scope is the single-sequence CPU path: a paged memory, DSV4 compressed caches, the sparse side caches and target-model caches refuse by name ([GAPS.md](GAPS.md)). Snapshot shadows, compressor rollback planes and the sampler chain stay runtime-only.

## System properties

Every `neuralfree.*` system property the code reads is listed below. Every Gradle run task (`runApp`, `runDemos`, `runLLMDemo`) forwards `-Pneuralfree.x=y` and `-Dneuralfree.x=y` to the program's JVM; pass them to your own JVM as `-Dneuralfree.x=y`. The build properties are under [Build](#build). The demo's command-line flags take precedence over the engine properties they correspond to.

### Engine (`EngineConfig.fromSystemProperties`)

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.backend` | `auto`, `opencl`, `cuda`, `vulkan` or `cpu`. Also chooses the training framework's API for `runApp` and `runDemos` (`NeuralFree.backendFromProperty`), where `cpu` is rejected. | `auto` |
| `neuralfree.device` | Device-name substring for every API (case-insensitive) | none |
| `neuralfree.ram.cache.mb` | Host cache of streamed weights in MiB; `-1` = half the maximum heap, `0` = none | -1 |
| `neuralfree.vram.depth` | Streamed layers resident in VRAM; `0` = from the VRAM budget | 0 |
| `neuralfree.draft` | Draft GGUF for speculative decoding | none |
| `neuralfree.draft.max` | Maximum drafted tokens per round | 3 |
| `neuralfree.draft.min` | Minimum verified draft length | 0 |
| `neuralfree.draft.p.min` | Draft probability floor | 0 |
| `neuralfree.draft.type` | `auto`, `draft-mtp` or `draft-simple`, as `--spec-type` | `auto` |
| `neuralfree.mtp` | Self-speculation with the model's NextN/MTP blocks | false |
| `neuralfree.lora` | Adapters as `path[:scale][,path[:scale]...]` | none |
| `neuralfree.reasoning.parser` | Reasoning parser by vLLM name, or `auto` for the chat template's | `auto` |
| `neuralfree.tool.parser` | Tool-call parser by vLLM name, `auto` to detect it from the chat template, `none` to leave calls in the content | `auto` |

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
| `neuralfree.mla.cache` | `auto`, `latent` or `decompressed` | `auto` |
| `neuralfree.prefix.reuse` | Keep the common prompt prefix between generations | true |
| `neuralfree.kv.rotation` | Hadamard rotation of quantized KV rows | true |
| `neuralfree.context.shift` | Shift the context when full instead of stopping | false |
| `neuralfree.state.snapshots` | Reserve recurrent-state snapshot buffers | true when speculative decoding is configured |
| `neuralfree.nextn.memory` | Place the model's NextN/MTP blocks and their caches in the fit | true when `neuralfree.mtp` is set |
| `neuralfree.parallel` | Sequences the memory holds at once (llama.cpp's `-np`), 1-256; above 1 selects the paged KV memory ([Paged KV memory](#paged-kv-memory)) | 1 |
| `neuralfree.kv.paged` | Force the paged KV memory for a single sequence ([Paged KV memory](#paged-kv-memory)); property-gated, not the engine default | false |
| `neuralfree.kv.page.size` | Page size of the paged KV pools, a power of two in [1, 256] | 16 |
| `neuralfree.kv.unified` | Share the whole context between the `neuralfree.parallel` sequences instead of one pool per sequence | true |

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
| `neuralfree.cpu.threads` | CPU engine worker threads | available processors |
| `neuralfree.stream.sync` | Force synchronous transfers for streamed weights | false |
| `neuralfree.diag` | GPU stage diagnostics | false |
| `neuralfree.diag.dump` | Append full-precision diagnostic values to the dump file | false |
| `neuralfree.diag.dumpfile` | Dump file | `<java.io.tmpdir>/neuralfree_diag_dump.txt` |
| `neuralfree.diag.steps` | Steps the diagnostics report | 1 |

### Backend selection and tuning

| Property | Meaning | Default |
|---|---|---|
| `neuralfree.backend.order` | AUTO's API preference | `vulkan,cuda,opencl` |
| `neuralfree.opencl.device`, `neuralfree.cuda.device`, `neuralfree.vulkan.device` | Per-API device-name substring; takes precedence over `neuralfree.device` for that API | none |
| `neuralfree.opencl.benchmark.disable` | Rank OpenCL devices on published attributes only | false |
| `neuralfree.opencl.disable` | Skip OpenCL enumeration | false |
| `neuralfree.opencl.skip.probe` | Skip the subprocess probe for crashing OpenCL platforms | false |
| `neuralfree.opencl.binaryCache` | OpenCL program binary cache (`~/.cache/neuralfree/opencl`, capped at 1 GiB) | true |
| `neuralfree.gpu.profile` | Per-command profiling on OpenCL queues | false |
| `neuralfree.gpu.stagingMb` | CUDA and Vulkan staging ring size in MiB (CUDA 1-1024, Vulkan 1-2047) | upload 64, readback 16 |
| `neuralfree.vulkan.pipelineCache` | On-disk Vulkan pipeline cache (`~/.cache/neuralfree/vulkan/`) | true |
| `neuralfree.vulkan.validation` | Enable `VK_LAYER_KHRONOS_validation` | false |
| `neuralfree.vulkan.blockMb` | Vulkan device-memory block size in MiB | 256 |
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
