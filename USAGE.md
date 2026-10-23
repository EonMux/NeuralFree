# NeuralFree Usage Guide

This guide covers:
- building NeuralFree, including the optional GPU kernel tools and the two kernel libraries;
- choosing a compute backend for the GGUF engine and for the training framework;
- how the training framework uses a GPU backend: locking, residency, ordering and numerical tolerances;
- the complete `LLMDemo` command line and the `neuralfree.*` system properties.

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
5. [System properties](#system-properties)
6. [Troubleshooting](#troubleshooting)

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
| `--model`, `-m <path>` | GGUF model | see [LLM-INFERENCE.md](LLM-INFERENCE.md#selecting-a-model) |
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

The demo's defaults apply only where no `neuralfree.*` sampler property is set. The engine defaults, which match llama.cpp, are listed under [Sampling](LLM-INFERENCE.md#sampling).

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
| `--spec-type <t>` | How the draft file drafts: `auto` (detected from the file), `draft-mtp` (alias `mtp`: its NextN/MTP blocks; without `--draft`, the target's own) or `draft-simple` (aliases `simple`, `model`: a separate draft model) | `auto` |
| `--draft-max`, `--draft-n-max`, `--spec-draft-n-max <n>` | Maximum drafted tokens per round | 3 |
| `--draft-min`, `--draft-n-min`, `--spec-draft-n-min <n>` | Minimum draft length that is verified | 0 |
| `--draft-p-min`, `--spec-draft-p-min <p>` | Stop drafting below this probability | 0 |

### LoRA adapters

| Flag | Meaning |
|---|---|
| `--lora <gguf>[:scale][,...]` | Apply adapters, scale 1 unless given; repeatable |
| `--lora-scaled <gguf>:<scale>[,...]` or `--lora-scaled <gguf> <scale>` | Apply adapters with explicit scales; repeatable |

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
