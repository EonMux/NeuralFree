# NeuralFree Quick Start

This guide takes a fresh clone to a running GGUF model and a trained model, in seven steps. The demos in `src/main/java/com/eonmux/neuralfree/demo/` are complete programs that use the same API.

1. [Install the prerequisites](#1-install-the-prerequisites)
2. [Clone and build](#2-clone-and-build)
3. [Check the GPU backend](#3-check-the-gpu-backend)
4. [Run the training demos](#4-run-the-training-demos)
5. [Run a GGUF model](#5-run-a-gguf-model)
6. [Use NeuralFree from Java](#6-use-neuralfree-from-java)
7. [Troubleshooting](#7-troubleshooting)

## 1. Install the prerequisites

**Required:**
- **JDK 11 or newer** (`java -version`). The Gradle wrapper, `./gradlew`, downloads Gradle 8.14.4 on first use, so Gradle need not be installed.
- **Git**, to clone the repository.

**A GPU driver for at least one API:**

| API | Needed by both parts | Also needed by the GGUF kernels (`llm`) | Also needed by the training kernels (`train`) |
|---|---|---|---|
| OpenCL | OpenCL 1.2 or newer on a GPU | Nothing more | Work-groups of at least 256 items |
| Vulkan | Vulkan 1.2 with timeline semaphores and a compute queue | 8-bit and 16-bit storage | Work-groups of at least 256 items |
| CUDA | An NVIDIA driver | A device of compute capability 7.5 or newer (with the default kernel build), and a driver new enough for the built modules | The same as the GGUF kernels |

The GGUF engine also runs on the CPU, without any GPU. The training framework needs a GPU backend of one of these APIs; on a backend that cannot run the `train` kernels, each op computes on the CPU.

**Optional build tools.** OpenCL needs none, because the driver compiles the kernel sources at run time. CUDA and Vulkan run kernels built ahead of time:
- `nvcc` from the CUDA Toolkit builds the CUDA kernels;
- `glslc` (or `glslangValidator`) with `spirv-dis` from the Vulkan SDK builds the Vulkan kernels; `spirv-opt` and `spirv-val` optimize and validate them when present;
- `clang` syntax-checks the OpenCL kernels and checks the kernel inventories against the kernel sources.

What happens without them:

| Missing | Effect |
|---|---|
| `nvcc` | The build warns, and the CUDA backend reports both kernel libraries unavailable |
| `glslc` or `spirv-dis` | The build warns, and the Vulkan backend reports both kernel libraries unavailable |
| `clang` | The OpenCL syntax check is skipped; the inventory checks use nvcc's preprocessor instead |
| `clang` and `nvcc` | The build warns that the checked-in kernel inventories were not checked against the kernel sources; OpenCL and Vulkan still run both libraries |

When a backend reports a kernel library unavailable:
- **the GGUF engine's AUTO** skips that backend and uses the next one;
- **the training framework's AUTO** prefers a backend that runs the `train` kernels; on a backend that does not, every op takes its CPU path.

Tools are found through their `-P` property, then the usual install directories (`$CUDA_HOME/bin`, `$VULKAN_SDK/bin`, ...), then `PATH`; [USAGE.md](USAGE.md#gpu-kernel-tools-all-optional) lists them. To make a missing tool fail the build instead of warning, pass `-Pneuralfree.requireCuda=true` or `-Pneuralfree.requireShaders=true`.

## 2. Clone and build

```bash
git clone https://github.com/EonMux/NeuralFree.git
cd NeuralFree

./gradlew assemble
```

`assemble` compiles the Java sources, builds the CUDA modules and SPIR-V that the installed tools allow, and packages the jar under `build/libs`. A warning names each missing tool and the backend it affects.

`./gradlew build` does the same and also runs the unit tests. Tests that need a GPU API the machine lacks are skipped. The run tasks below build what they need, so `assemble` is optional before them.

## 3. Check the GPU backend

```bash
./gradlew runApp
```

`runApp` creates the training framework's backend, logs the API and device it chose and whether the `train` kernels run there, and checks a buffer round trip on the device:

```text
ComputeBackend created - API: VULKAN, Device: <your GPU>, VRAM: <n> MB, Compute Units: <n>
Train kernels: available
...
Buffer round-trip test PASSED - data verified correctly
```

`Train kernels: unavailable (tensor ops take their CPU path): <reason>` means the backend works but the training ops compute on the CPU; the reason names what is missing, usually kernels the build could not produce (see [Troubleshooting](#7-troubleshooting)). `No GPU backend available` means no API found a usable GPU.

AUTO picks the device first, then tries Vulkan, CUDA and OpenCL on it, in that order. To choose:

```bash
./gradlew runApp -Pneuralfree.backend=opencl          # auto, opencl, cuda or vulkan
./gradlew runApp -Pneuralfree.device=4090             # the GPU whose name contains "4090"
./gradlew runApp -Pneuralfree.backend.order=cuda,opencl,vulkan
```

The Gradle run tasks forward every `neuralfree.*` property given with `-P` or `-D` to the program's JVM.

## 4. Run the training demos

```bash
./gradlew runDemos
./gradlew runDemos --args="--backend cuda"     # auto, opencl, cuda or vulkan
```

`runDemos` runs the MLP, CNN, GPU matmul, transformer, GNN, autograd and inference demos on one shared backend, and prints whether the `train` kernels run on it. `--backend` takes precedence over `-Pneuralfree.backend`. `cpu` is refused, because the framework's tensors live on a GPU backend.

## 5. Run a GGUF model

`runLLMDemo` runs any GGUF model that llama.cpp runs, for example one downloaded from Hugging Face. It needs no native library. The model path is taken from the first of these that is set:
1. `--model <path>` (or `-m <path>`);
2. `-Pneuralfree.model=<path>`;
3. the `NEURALFREE_MODEL` environment variable.

Without any of them, `runLLMDemo` prints these options and exits with status 2.

```bash
# One prompt, greedy, on the automatically chosen GPU backend
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --prompt 'Explain recursion' --temp 0"

# Console chat (no --prompt), with a 16 GiB heap instead of the default 8 GiB
./gradlew runLLMDemo -Pheap=16384 -Pargs="--model /path/to/model.gguf"

# Force a backend, or run on the CPU
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --prompt 'Hi' --backend vulkan"
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --prompt 'Hi' --cpu"

# 8-bit KV cache and a fixed context
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --kv-type q8_0 --ctx 16384"

# Self-speculation with the model's MTP heads
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --mtp --draft-max 3"

# Pooled embedding from an embedding model
./gradlew runLLMDemo -Pargs="--model /path/to/embedder.gguf --embed --prompt 'text to embed'"

# Print the memory plan and exit
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --vram-budget"

# Every flag
./gradlew runLLMDemo -Pargs="--help"
```

`-Pargs` holds the program arguments; single or double quotes group words. The GGUF engine chooses AUTO among the backends that run the `llm` kernels, and continues on the CPU when no GPU backend can be created. After each generation the demo prints its timing statistics; `--stats-file <path>` also appends the session's statistics to `<path>`. Every flag and `neuralfree.*` property is listed in [USAGE.md](USAGE.md#llmdemo-command-line-reference).

## 6. Use NeuralFree from Java

To use NeuralFree from another Gradle or Maven project on the same machine, install it in the local Maven repository:

```bash
./gradlew publishToMavenLocal
```

and depend on `com.eonmux:neuralfree:1.0.0-SNAPSHOT` (with `mavenLocal()` among the Gradle repositories). The published POM carries the JCuda and LWJGL natives of the platform it was built on.

Two settings help when you run your own program with `java`:
- **Logging.** The library logs through the Log4j 2 API and ships no logging configuration, so Log4j's default prints errors only. Pass a configuration, for example `-Dlog4j2.configurationFile=src/main/config/log4j2.xml` (INFO to the console, as the run tasks use).
- **`--add-opens=java.base/java.util=ALL-UNNAMED`** lets abacus-util, a dependency, take its fast path through `java.util.ArrayList` internals; without it, abacus-util falls back quietly.

The snippets below are consecutive parts of one program: each uses the variables the earlier ones define, such as `backend` and `model`. Variables the comments describe, such as `inputs` and `targets`, are your data.

### Choose a backend

`HWAcceleration.forMode(mode)` enumerates the GPU APIs that `mode` allows, and `NeuralFree.createTrainingBackend(hw, mode)` returns the backend the training framework runs on. `hw.shutdown()` closes every backend the instance created and releases the devices.

```java
import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.NeuralFree;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

HWAcceleration hw = HWAcceleration.forMode(Config.ACCELERATION_MODE.AUTO);

// AUTO among the backends that run the train kernels
ComputeBackend backend = NeuralFree.createTrainingBackend(hw, Config.ACCELERATION_MODE.AUTO);
System.out.println(NeuralFree.describeTrainKernels(backend));   // "available", or why the ops run on the CPU

// ... the snippets below use the backend; when the program is done with it:
// hw.shutdown();
```

- **`AUTO`** picks the device first, then tries Vulkan, CUDA and OpenCL on it. With `AUTO`, `createTrainingBackend` selects a backend on which the `train` kernels run; when none does, it logs a warning with the reason per API and returns the backend plain AUTO picks.
- **`OPENCL`, `CUDA` or `VULKAN`** forces one API; the same warning is logged when the `train` kernels do not run there.
- **`hw.createBackend(mode)`** returns a backend without the `train` requirement.
- `createTrainingBackend` throws `NoBackendAvailableException` when no GPU backend can be created at all.

[USAGE.md](USAGE.md#compute-backends) has the details.

### Tensors

```java
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

Tensor a = Tensor.randn(backend, Shape.of(2, 3));
Tensor b = Tensor.fromArray(backend, new float[][]{{1, 2, 3}, {4, 5, 6}});

Tensor sum  = TensorOps.add(a, b);
Tensor prod = TensorOps.matmul(a, b.transpose());   // (2x3)(3x2) -> (2x2)
Tensor total = TensorOps.sum(sum);                 // scalar; TensorOps.sum(t, dims...) reduces dims

int[] dims = sum.shape().dims();                    // {2, 3}
long count = sum.shape().numElements();             // 6
float[] host = prod.toFloatArray();
```

Ops run on the tensors' backend, and data stays on the device until it is read, for example with `toFloatArray()`. The garbage collector does not free device memory: close a tensor (`close()`, or try-with-resources) when it is no longer needed.

### Autograd

An op records a node when grad mode is on and one of its inputs requires grad. Gradients reach the leaves: parameters, and tensors made with `withRequiresGrad(true)`.

```java
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import java.util.List;

Tensor values = Tensor.fromArray(backend, new float[]{1, 2, 3});
Tensor x = values.withRequiresGrad(true);           // a leaf over the same storage
values.close();

Tensor y = TensorOps.sum(TensorOps.mul(x, x));      // y = sum(x * x), recorded

// dy/dx = 2x, returned without touching x.grad(); retainGraph keeps the graph for the backward below
List<Tensor> dydx = Autograd.grad(y, List.of(x), /*gradOutput*/ null, /*retainGraph*/ true, /*createGraph*/ false);

y.backward();                                       // x.grad() = 2x; the graph is released

try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
    Tensor doubled = TensorOps.mul(x, 2f);          // not recorded: doubled.requiresGrad() == false
    doubled.close();
}
```

- **A second backward** through a released graph throws `IllegalStateException`. `ComputationGraph.current().backward(y, true)` keeps the graph instead.
- **With `createGraph = true`**, `Autograd.grad` records the gradient computation, so the result can be differentiated again (second-order gradients, gradient penalties).

[TRAINING.md](TRAINING.md#autograd) covers backward passes, grad mode, step scopes and gradient checkpointing in full.

### Define and train a model

Layer constructors take dimensions only. `initialize(backend, random)` allocates each layer's parameters on the device. A `Module` registers its layers with `registerModule`.

```java
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.training.EarlyStopping;
import com.eonmux.neuralfree.training.ModelCheckpoint;
import com.eonmux.neuralfree.training.Trainer;
import com.eonmux.neuralfree.training.TrainerConfig;
import com.eonmux.neuralfree.training.TrainingResult;
import java.util.List;
import java.util.Random;

public class MLP extends Module {
    private Linear fc1;
    private Linear fc2;

    public void initialize(ComputeBackend backend, Random random, int in, int hidden, int out) {
        fc1 = registerModule("fc1", new Linear(in, hidden));
        fc1.initialize(backend, random);
        fc2 = registerModule("fc2", new Linear(hidden, out));
        fc2.initialize(backend, random);
    }

    @Override
    public Tensor forward(Tensor x) {
        return fc2.forward(TensorOps.relu(fc1.forward(x)));
    }
}

MLP model = new MLP();
model.initialize(backend, new Random(42), 784, 256, 10);

// inputs, targets (and valInputs, valTargets): Tensor[] of samples;
// a target is a one-hot FLOAT32 vector or an INT32 class index
DataLoader<TensorDataset.Sample> train = DataLoader.builder(new TensorDataset(inputs, targets))
        .batchSize(64).shuffle(true).build();
DataLoader<TensorDataset.Sample> val = DataLoader.builder(new TensorDataset(valInputs, valTargets))
        .batchSize(64).build();

Adam optimizer = new Adam(model.parameters(), 1e-3f);
ModelCheckpoint checkpoint = new ModelCheckpoint("checkpoints/");
Trainer trainer = new Trainer(model, optimizer, new CrossEntropyLoss(), train, val,
        List.of(new EarlyStopping(/*patience*/ 5), checkpoint),
        /*lrScheduler*/ null, TrainerConfig.defaults());
TrainingResult result = trainer.train(/*epochs*/ 10);
checkpoint.restoreBest();
```

- **`EarlyStopping(5)`** stops training after 5 epochs in a row without a lower validation loss. Without a validation loader it never stops training and logs a warning. `EarlyStopping.builder()` can monitor the training loss instead (`monitor("loss")`) and put the best epoch's weights back when training ends (`restoreBestWeights(true)`).
- **`ModelCheckpoint("checkpoints/")`** writes the first epoch, and each later epoch whose monitored loss is lower than the best so far, to `checkpoints/checkpoint-epochNNNN.nfc`. The monitored loss is the validation loss, or the training loss when no validation ran. `restoreBest()` loads the best one back into the model.
- **`TrainerConfig.builder()`** sets gradient accumulation, gradient clipping, the evaluation interval, the trainer's progress log and a periodic checkpoint interval. By default (`verbose`, `logEveryNSteps` 10) the trainer logs the loss and learning rate at INFO every 10 steps, and the losses at the end of every epoch. A `ProgressLogger` callback prints progress to standard output instead. `checkpointEveryNEpochs(n)` makes the `ModelCheckpoint` callback also save every `n`-th epoch; the trainer refuses that setting without one.
- **For mixed precision**, build a `MixedPrecisionTrainer` around the same trainer: `MixedPrecisionTrainer.builder().baseTrainer(trainer).build().train(10)`. It emulates FP16 inputs and scales the loss dynamically.

To write the loop by hand instead, open a `StepScope` per step. It discards whatever the step recorded but did not backpropagate:

```java
CrossEntropyLoss lossFn = new CrossEntropyLoss();

// batchInputs, batchTargets: one batch, with the batch as dimension 0
try (ComputationGraph.StepScope step = ComputationGraph.current().step();
     Tensor loss = lossFn.compute(model.forward(batchInputs), batchTargets)) {
    loss.backward();        // accumulates into each parameter's gradient and releases the graph
    optimizer.step();
    optimizer.zeroGrad();
}
```

When training ends, close what you created: the loaders, `optimizer.close()`, `model.close()`, then `hw.shutdown()`. [TRAINING.md](TRAINING.md#training) describes the trainers, callbacks and accumulation in full.

### Save and load

```java
import com.eonmux.neuralfree.io.ModelSerializer;

ModelSerializer.save(model, "model.nf");                  // or save(model, optimizer, path)

MLP reloaded = new MLP();
reloaded.initialize(backend, new Random(0), 784, 256, 10);
ModelSerializer.load("model.nf", reloaded, backend);      // or load(path, model, optimizer, backend)
```

The file holds every parameter and persistent buffer (for example, BatchNorm running statistics). It also holds the optimizer state when an optimizer is passed.

### Run a GGUF model

`LLMModelFactory.open` opens a model on the configured backend, with a CPU fallback, and `GenerationLoop` runs llama.cpp's completion loop over it. The engine creates and closes its own backend here, so this part needs none of the code above.

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
        .backend(EngineConfig.BackendMode.AUTO)        // or CPU, OPENCL, CUDA, VULKAN
        .build();
try (EngineModel llm = LLMModelFactory.open("/path/to/model.gguf", cfg)) {
    GGUFTokenizer tok = llm.tokenizer();
    int[] prompt = tok.encode(tok.formatChat("You are a helpful assistant.", "Hello", false));

    GenerationRequest request = GenerationRequest.builder()
            .maxNewTokens(256)
            .sampler(SamplerConfig.builder().temperature(0.7f).seed(42).build())
            .build();
    GenerationResult generation = GenerationLoop.forModel(llm, LLMModelFactory.effectiveConfig(llm, cfg))
            .generate(prompt, request, (id, text) -> System.out.print(text));
    System.out.println("\n" + generation.stopReason() + ", " + generation.tokenCount() + " tokens");
}
```

`LLMModelFactory.open` throws `IOException` for a missing or unreadable file. [LLM-INFERENCE.md](LLM-INFERENCE.md#programmatic-api) describes speculative decoding, embeddings, classification and the other entry points.

## 7. Troubleshooting

- **No GPU at all.** The GGUF engine runs on the CPU: pass `--cpu`, or let AUTO fall back. The training framework needs a GPU backend: `runApp` logs `No GPU backend available`, and `createTrainingBackend` throws `NoBackendAvailableException`, which lists the reason for every API it tried. Install or update the GPU driver of at least one API.
- **Training runs on the CPU.** `Train kernels: unavailable (tensor ops take their CPU path): <reason>` usually means the build produced no CUDA modules or SPIR-V for the chosen API. Install `nvcc` or the Vulkan SDK and run `./gradlew assemble` again, or choose another API with `--backend` or `-Pneuralfree.backend`. OpenCL needs no build tool.
- **Missing toolchains.** Each missing tool prints a warning naming the backend it affects; [USAGE.md](USAGE.md#gpu-kernel-tools-all-optional) lists where each tool is searched and the `-P` properties that point to it. `-Pneuralfree.cuda.archs=<list>` limits the CUDA architectures nvcc builds for (default `75,80,86,89,90,100,120`).
- **Vulkan device features.** The `train` kernels need Vulkan 1.2 with timeline semaphores; the GGUF kernels also need 8-bit and 16-bit storage. `vulkaninfo` (from the Vulkan SDK) lists what a device offers. AUTO skips a device without them. With `--backend vulkan`, `runLLMDemo` prints `GPU initialization failed: Vulkan backend unavailable: <reason>` and continues on the CPU. `-Dneuralfree.vulkan.validation=true` enables the Khronos validation layer when it is installed.
- **Device memory.** For the GGUF engine, lower `--ctx`, use a quantized KV cache (`--kv-type q8_0`), or run `--vram-budget` to see the plan. In the training framework, tensors, modules, optimizers and data loaders hold device memory until they are closed.
- **Heap.** Large models with a host cache need a larger heap: `./gradlew runLLMDemo -Pheap=24576 ...`.
- **Windows GPU timeouts.** Windows resets a GPU whose work runs longer than its timeout detection delay, two seconds by default, and long kernels on a slow or busy GPU can exceed it. [USAGE.md](USAGE.md#troubleshooting) shows how to raise the delay.
- **No log output.** Run through the Gradle tasks, or pass `-Dlog4j2.configurationFile=src/main/config/log4j2.xml` to `java` (see [step 6](#6-use-neuralfree-from-java)).

[USAGE.md](USAGE.md#troubleshooting) covers more cases, and [GAPS.md](GAPS.md) lists the known limitations.

## Next steps

- [TRAINING.md](TRAINING.md): the training framework's API in full.
- [USAGE.md](USAGE.md): the build, backends, how the framework uses a GPU backend, and the complete LLMDemo reference.
- [LLM-INFERENCE.md](LLM-INFERENCE.md): the GGUF engine and its Java API.
- [LLAMACPP-PARITY.md](LLAMACPP-PARITY.md): which architectures run and how.
- [GAPS.md](GAPS.md): known limitations.
- [LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md): the licence terms, including when a commercial licence is required (above USD 20 million in revenue).
