# NeuralFree

NeuralFree is a hardware-accelerated deep learning library written in Java. It runs GGUF language models and trains models of its own on OpenCL, CUDA and Vulkan GPUs, with CPU fallbacks, without Python or native ML libraries.

It has two parts:

1. **A GGUF inference engine** (`com.eonmux.neuralfree.gguf`). It loads `*.gguf` models and runs them on a GPU or the CPU. It follows [llama.cpp](https://github.com/ggml-org/llama.cpp) for:
   - architectures (every architecture name llama.cpp defines);
   - quantization formats;
   - tokenizers and chat templates;
   - sampling and cache behaviour.

   The `runLLMDemo` task is its command line; [LLM-INFERENCE.md](LLM-INFERENCE.md) describes the engine and its Java API.
2. **A training framework.** It provides:
   - N-dimensional tensors, reverse-mode autograd and a module system;
   - optimizers, a data pipeline, `Trainer` and `MixedPrecisionTrainer`;
   - classical ML, reinforcement learning, meta-learning and distributed training on the same tensors;
   - the `train` GPU kernel library, built from one source set for OpenCL, CUDA and Vulkan, behind the `ComputeBackend` interface. On a backend that cannot run it, each op computes on the CPU.

## Status

The GGUF engine covers every architecture name llama.cpp defines (164 entries, including `(unknown)`). [LLAMACPP-PARITY.md](LLAMACPP-PARITY.md) gives the class and cache of each one. Of these:
- 136 are generative models;
- the rest are embedding, classifier, diffusion, drafter, audio and encoder-decoder models;
- three are refused: `clip`, `gptj` and `(unknown)`.

**How the engine was verified:**
- the Java sources compile;
- the GPU kernels compile with clang (OpenCL C 1.2), nvcc (PTX) and glslc with spirv-val (SPIR-V);
- the code was traced against the llama.cpp sources.

It has not been validated by running models on GPU hardware. The limitations the code refuses explicitly are listed in [GAPS.md](GAPS.md).

## Features

### Training framework
- **Backends.** OpenCL, CUDA and Vulkan behind `ComputeBackend`, with automatic device-first selection. Tensor, layer, loss, optimizer and classical-ML ops run the same `train` kernels on all three APIs, and tensors stay on the device between ops. Device results match the CPU reference within documented tolerance classes. On a backend where the `train` library is unavailable, each op takes its CPU reference path.
- **Tensors.** N-dimensional with broadcasting and strided views (reshape, transpose, permute, slice, expand). Ops compute in FLOAT32. INT32, INT64 and BOOL tensors carry indices, class targets and masks. `Tensor.to` converts FLOAT32 to and from FLOAT16, BFLOAT16, FLOAT64, INT32 and BOOL.
- **Autograd.** Reverse-mode, recorded per thread, with:
  - `noGrad` and `enableGrad` scopes;
  - `backward` with an optional seed and `retainGraph`;
  - `Autograd.grad`, the equivalent of `torch.autograd.grad`;
  - create-graph passes for higher-order gradients;
  - gradient checking and activation checkpointing.
- **Layers:**
  - Linear, Conv1d, Conv2d, ConvTranspose2d;
  - max, average and adaptive pooling;
  - Dropout, Embedding, Flatten;
  - BatchNorm1d/2d, LayerNorm, GroupNorm, RMSNorm.
- **Activations:** ReLU, LeakyReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax, ELU, Mish, GLU, GeGLU.
- **Losses:** cross-entropy, BCE, BCE-with-logits, MSE, L1, smooth L1, Huber, focal, KL divergence, contrastive, triplet margin.
- **Architectures:**
  - transformers (multi-head attention, RoPE, ALiBi, KV cache, mixture of experts);
  - ResNet 18-152 and UNet;
  - RNN, GRU, LSTM and state-space models;
  - graph networks (GCN, GAT, GraphSAGE, GIN);
  - VAE, GAN training, DDPM/DDIM diffusion;
  - multimodal encoders;
  - Neural ODE, NeRF and Gaussian splatting.
- **Training:**
  - SGD, Adam, AdamW, Adagrad, RMSprop and LAMB, stepping on the device;
  - step, cosine, one-cycle, warm-up and plateau schedulers;
  - `Trainer`, with gradient accumulation, gradient clipping, callbacks, early stopping and on-disk checkpoints;
  - mixed precision (FP16 or BF16 emulation with dynamic loss scaling);
  - INT8, INT4 and FP8 quantization and quantization-aware training;
  - data-parallel, pipeline-parallel and federated (FedAvg, FedProx) training.
- **Model I/O:** native serialization of model and optimizer state, and ONNX export.
- **Other ML:**
  - KMeans, PCA, SVM, Gaussian processes, logistic regression, perceptrons and forecasting units on the `train` kernels;
  - random forests, gradient-boosted trees and HMMs on the CPU;
  - DQN, A2C, PPO, SAC and RLHF;
  - MAE, CLIP, DINO;
  - MAML, NAS, knowledge distillation.

### GGUF inference engine
- **Model files and types:**
  - single-file and split (`-0000N-of-0000M`) GGUF files;
  - 29 weight types run natively on the GPU, the others are dequantized at upload;
  - the CPU engine runs the same types except Q8_K.
- **Model classes:**
  - decoder-only text models, including mixture-of-experts, MLA, sparse-attention (DSA/MSA), recurrent (Mamba, RWKV) and hybrid linear-attention models;
  - T5 encoder-decoder;
  - embedding, classifier and reranker models;
  - masked-diffusion models;
  - audio decoders.
- **Caches:**
  - nine KV cache types (`f32` to `q5_1`), with Hadamard rotation of quantized rows;
  - sliding-window rings;
  - prefix reuse, context shift, rollback and recurrent-state snapshots.
- **Speculative decoding:**
  - built-in NextN/MTP heads;
  - MTP draft files;
  - separate draft models;
  - EAGLE-3, Gemma 4 assistant and DFlash drafters.
- **LoRA adapters** on the CPU and the GPU.
- **llama.cpp tokenizers** (SentencePiece, BPE with 63 pre-tokenizers, Unigram, WordPiece, RWKV, PLaMo) and Jinja chat templates.
- **llama.cpp's sampler chain** and defaults.
- **Context sizing:** the context is fitted to device memory automatically, and layer weights stream from host memory when VRAM is short.

## Requirements

- **JDK 11 or newer.** The Gradle wrapper (`./gradlew`) downloads Gradle 8.14.4.
- **A GPU driver for at least one API.** The GGUF engine also runs on the CPU without one. The training framework needs a backend of some API. On a backend that cannot run its kernels, it computes on the CPU.

  | API | The backend needs | The GGUF kernels (`llm`) also need | The framework kernels (`train`) also need |
  |---|---|---|---|
  | OpenCL | OpenCL 1.2 or newer | Nothing more | Work-groups of at least 256 items |
  | Vulkan | Vulkan 1.2 with timeline semaphores and a compute queue | 8-bit and 16-bit storage | Work-groups of at least 256 items |
  | CUDA | An NVIDIA driver | A device and driver the built modules cover; the default kernel build targets compute capability 7.5 and newer | The same as the GGUF kernels |

- **Optional build tools:**
  - `nvcc` from the CUDA Toolkit, for the CUDA kernels;
  - `glslc` (or `glslangValidator`) with `spirv-dis`, `spirv-opt` and `spirv-val` from the Vulkan SDK, for the Vulkan kernels;
  - `clang`, for the OpenCL syntax check and to check and regenerate the kernel inventories (nvcc's preprocessor stands in when clang is missing).

### Without the build tools

OpenCL needs no build tool, because the driver compiles the kernel sources at run time. Each build tool serves both kernel libraries, `llm` and `train`:
- **Without `nvcc`**, the build warns and builds no CUDA kernels. The CUDA backend then reports both libraries unavailable.
- **Without a shader compiler or `spirv-dis`**, the build warns and builds no Vulkan kernels. The Vulkan backend then reports both libraries unavailable.
- **Without `clang`**, the OpenCL syntax check is skipped and the inventory tasks use nvcc's preprocessor.
- **Without `clang` and `nvcc`**, the build warns that the inventories' freshness was not checked. Both inventories (`kernels/llm/inventory.txt` and `kernels/train/inventory.txt`) are checked in, so the Vulkan backend still runs both libraries from the shaders the build compiled.

When a backend cannot run a library:
- for the GGUF engine, AUTO skips that backend and uses the next one;
- for the training framework, AUTO prefers a backend that runs the `train` kernels; on a backend that does not, every op takes its CPU path.

To make a missing tool fail the build instead of warning, pass `-Pneuralfree.requireCuda=true` or `-Pneuralfree.requireShaders=true`. See [USAGE.md](USAGE.md#gpu-kernel-tools-all-optional).

## Quick start

[QUICK-START.md](QUICK-START.md) walks through these steps in detail, with Java examples for both parts.

```bash
git clone https://github.com/EonMux/NeuralFree.git
cd NeuralFree

./gradlew assemble    # compile and build the GPU kernels the installed tools allow
./gradlew runApp      # backend check: picks a GPU backend and verifies a buffer round trip

# Training framework: the demo suite (MLP, CNN, GPU matmul, transformer, GNN, autograd, inference)
./gradlew runDemos
./gradlew runDemos --args="--backend vulkan"

# GGUF engine: one prompt on the best GPU backend, a console chat, and the CPU engine
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --prompt 'Explain recursion' --temp 0"
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf"
./gradlew runLLMDemo -Pargs="--model /path/to/model.gguf --prompt 'Hi' --cpu"
```

`./gradlew build` also runs the unit tests; tests that need a GPU API the machine lacks are skipped.

**Backend selection.** AUTO chooses the device first, then tries Vulkan, CUDA and OpenCL on it, in that order by default. The property `neuralfree.device=<name substring>` picks the GPU, and `neuralfree.backend.order` changes the API order. The Gradle run tasks forward every `neuralfree.*` property given with `-P` or `-D` to the JVM.
- **`runApp` and `runDemos`** choose AUTO among the backends that run the `train` kernels. To force an API, pass `-Pneuralfree.backend=opencl|cuda|vulkan`. `runDemos` also takes `--args="--backend auto|opencl|cuda|vulkan"`, which overrides the property.
- **The GGUF engine** chooses AUTO among the backends that run the `llm` kernels. To force an API, pass `--backend opencl|cuda|vulkan|cpu`.
- **In code**, call `NeuralFree.createTrainingBackend(hw, mode)` for the framework. `NeuralFree.describeTrainKernels(backend)` reports whether the `train` kernels run on a backend, and why not when they do not.

**Logging.** The run tasks log at INFO to the console through `src/main/config/log4j2.xml`. The library jar ships no logging configuration; an application that uses NeuralFree configures Log4j 2 itself, for example with `-Dlog4j2.configurationFile=<file>`.

## Example: define and train a model

Layer constructors take dimensions only. `initialize(backend, random)` allocates each layer's parameters on the device, as in `demo/MLPDemo.java`.

```java
import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.NeuralFree;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Trainer;
import com.eonmux.neuralfree.training.TrainerConfig;

import java.util.List;
import java.util.Random;

public class SimpleClassifier extends Module {
    private Linear fc1;
    private Linear fc2;

    public void initialize(ComputeBackend backend, Random random, int inputDim, int numClasses) {
        fc1 = registerModule("fc1", new Linear(inputDim, 256));
        fc1.initialize(backend, random);
        fc2 = registerModule("fc2", new Linear(256, numClasses));
        fc2.initialize(backend, random);
    }

    @Override
    public Tensor forward(Tensor input) {
        return fc2.forward(TensorOps.relu(fc1.forward(input)));
    }
}

// AUTO among the backends that run the train kernels (OpenCL, CUDA or Vulkan)
HWAcceleration hw = HWAcceleration.forMode(Config.ACCELERATION_MODE.AUTO);
ComputeBackend backend = NeuralFree.createTrainingBackend(hw, Config.ACCELERATION_MODE.AUTO);

SimpleClassifier model = new SimpleClassifier();
model.initialize(backend, new Random(42), 784, 10);

// inputs, targets: Tensor[] of samples; a target is a one-hot FLOAT32 vector or an INT32 class index
DataLoader<TensorDataset.Sample> trainLoader = DataLoader.builder(new TensorDataset(inputs, targets))
        .batchSize(64).shuffle(true).build();
Adam optimizer = new Adam(model.parameters(), 1e-3f);
Trainer trainer = new Trainer(model, optimizer, new CrossEntropyLoss(),
        trainLoader, /*valLoader*/ null, List.of(), /*lrScheduler*/ null, TrainerConfig.defaults());
trainer.train(/*epochs*/ 10);

trainLoader.close();
optimizer.close();
model.close();
hw.shutdown();   // closes the backend and releases the devices
```

## Source layout

```
com.eonmux.neuralfree
├── acceleration/     GPU backends (backend/ocl, backend/cuda, backend/vulkan), device selection,
│                     kernel libraries, memory pool, execution pipeline, kernel cache
├── tensor/           Tensor, Shape, TensorOps and the device dispatchers; tensor/gpu holds the
│                     train-library runtime (TrainGpu, TrainOps)
├── autograd/         Reverse-mode differentiation, gradient checking, checkpointing
├── nn/               layers, activations, norm, loss, transformer, recurrent, vision, graph,
│                     generative, multimodal, selfsupervised, structured
├── optim/            Optimizers and learning-rate schedulers
├── training/         Trainer, callbacks, checkpoints
├── data/             DataLoader, datasets, samplers, transforms
├── inference/        InferenceEngine, batching, paged attention, speculative decoding
├── precision/        Mixed precision, quantizer, QAT
├── distributed/      Data-parallel, pipeline-parallel and federated training
├── ml/  rl/  meta/   Classical ML, reinforcement learning, meta-learning
├── ai/               A standalone dense network (ai.v2) trained on the train kernels through
│                     acceleration/pipeline, with its activation and cost functions
├── io/               Model serialization, ONNX export
├── gguf/             GGUF engine: arch, hparams, spec, tokenizer, jinja, chat, quant, memory,
│                     ops, cpu, gpu, engine
└── demo/             Runnable demos (DemoRunner, MLPDemo, CNNDemo, GNNDemo, TransformerDemo,
                      AutogradDemo, InferenceDemo, GpuMatmulDemo, LLMDemo)
```

### GPU kernel libraries

The kernels are organized as two libraries, `llm` (the GGUF engine) and `train` (the training framework and classical ML), registered in `KernelLibraries`. `gradle/gpu-kernels.gradle` builds both. Each library has:
- OpenCL C sources that also compile as CUDA;
- GLSL twins of those kernels for Vulkan;
- a `manifest.txt` that orders its sources;
- a checked-in `inventory.txt` of the kernels CUDA and Vulkan must provide.

```
src/main/resources
├── kernels/
│   ├── common/dialect.h     OpenCL C / CUDA dialect header shared by both libraries
│   ├── llm/                 GGUF engine kernels (*.cl, common/*.h), manifest.txt, inventory.txt
│   └── train/               framework kernels (*.cl, ml/ for classical ML, common/train.h), manifest.txt,
│                            inventory.txt
└── shaders/
    ├── common/nf_base.glsl  GLSL base include shared by both libraries
    ├── llm/                 GLSL twins of the GGUF kernels (common/*.glsl), variants/*.tsv
    ├── train/               GLSL twins of the framework kernels (ml/, common/train.glsl), variants/*.tsv
    └── test/                backend test shaders
```

The build writes the CUDA modules and the SPIR-V under `build/generated/gpu`, and `processResources` merges them into the classpath as `kernels/<lib>/cuda` and `shaders/<lib>`. `-Pneuralfree.gpu.libraries=llm` (or `train`) restricts the kernel tasks to one library, to speed up development builds; the other library's CUDA and Vulkan kernels are then missing from that build.

## Documentation

- [QUICK-START.md](QUICK-START.md): step-by-step first use, from a fresh clone to running a GGUF model and training a model in Java.
- [USAGE.md](USAGE.md):
  - the build, the GPU kernel tools and the kernel libraries;
  - backend selection;
  - how the training framework uses a GPU backend;
  - every `LLMDemo` flag and `neuralfree.*` property;
  - troubleshooting.
- [TRAINING.md](TRAINING.md): the training framework's API:
  - [Tensors](TRAINING.md#tensors) and [Autograd](TRAINING.md#autograd);
  - [Parameters](TRAINING.md#parameters) and [Modules and layers](TRAINING.md#modules-and-layers);
  - [Optimizers and schedulers](TRAINING.md#optimizers-and-schedulers), [Loss functions](TRAINING.md#loss-functions) and the [Data pipeline](TRAINING.md#data-pipeline);
  - [Training](TRAINING.md#training), [Mixed precision](TRAINING.md#mixed-precision) and [Checkpoints and serialization](TRAINING.md#checkpoints-and-serialization);
  - [Distributed training](TRAINING.md#distributed-training), [Reinforcement learning](TRAINING.md#reinforcement-learning) and [Meta-learning and distillation](TRAINING.md#meta-learning-and-distillation);
  - [Inference, quantization and export](TRAINING.md#inference-quantization-and-export) and [Classical ML](TRAINING.md#classical-ml-and-other-packages).
- [LLM-INFERENCE.md](LLM-INFERENCE.md): the GGUF engine, covering model selection, backends, the programmatic API, weight types, tokenizers, sampling, memory and model modes.
- [LLAMACPP-PARITY.md](LLAMACPP-PARITY.md): the per-architecture matrix and the cache, speculative, LoRA, encoder-decoder, diffusion and audio support compared with llama.cpp.
- [GAPS.md](GAPS.md): known limitations and defects.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md): how to contribute and how to report a vulnerability.
- [LICENSE](LICENSE), [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md), [NOTICE](NOTICE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md): the licence terms.

## Contributing

Bug reports, fixes and improvements are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) describes the build, the code style, the tests, the pull request process and the licence terms that apply to contributions.

## Security

Report vulnerabilities privately through a GitHub security advisory, not in a public issue. [SECURITY.md](SECURITY.md) describes the process.

## License

Copyright (c) 2017-2026 EONMUX LTD and Amir Farhad Eslampanah.

NeuralFree is licensed under the [NeuralFree License](LICENSE): the Apache License, Version 2.0, with an Additional Commercial Use Condition.
- **Apache 2.0 terms for most users.** Individuals, students, researchers and educators, and companies and projects below the USD 20 million revenue threshold, can use, modify and redistribute NeuralFree under the Apache License 2.0 terms, including in commercial products. Non-commercial research, education and personal use is not restricted.
- **Commercial licence above USD 20 million.** A commercial licence from EONMUX LTD is required when either of these applies:
  - your organization, together with its affiliates, had more than USD 20 million in annual gross revenue, and you use NeuralFree, or software derived from it, for any commercial activity;
  - you use NeuralFree, or software derived from it, in or for the benefit of a product, service, model or other project that generated more than USD 20 million in gross revenue in any twelve-month period.

  [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) summarizes the condition; the [LICENSE](LICENSE) file is the binding text. Request a commercial licence through [github.com/EonMux](https://github.com/EonMux).
- **Redistribution.** Redistributions must keep the complete LICENSE file, including the commercial condition, and the [NOTICE](NOTICE) file.
- **Third-party code.** Parts of NeuralFree are derived from llama.cpp/ggml (MIT License) and the other projects listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Those parts remain under their original licences, which the commercial condition does not restrict.

## Acknowledgements

NeuralFree's GGUF engine follows [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) by the ggml authors: its architectures, quantization formats and tables, tokenizers, chat handling and sampler are ported from them. It also includes code derived from:
- the [YaRN](https://github.com/jquesnelle/yarn) reference implementation by Jeffrey Quesnelle and Bowen Peng (RoPE scaling);
- the [JCuda](https://github.com/jcuda/jcuda-samples) and [JOCL](https://github.com/gpu/JOCLSamples) samples by Marco Hutter (device queries);
- [Jinja2](https://github.com/pallets/jinja) by Pallets (the `indent` filter);
- OpenAI's [gpt-oss](https://github.com/openai/gpt-oss) chat template (the harmony prompt fallback).

The [Gradle Wrapper](https://github.com/gradle/gradle) is distributed unmodified. [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) gives the copyright notices and licence terms of all of these.

At run time NeuralFree uses [JOCL](http://www.jocl.org/), [JCuda](http://www.jcuda.org/), [LWJGL](https://www.lwjgl.org/), [Apache Log4j 2](https://logging.apache.org/log4j/2.x/), [abacus-util](https://github.com/landawn/AbacusUtil) and [Brownies Collections](http://www.magicwerk.org/page-collections-download.html), each under its own licence.
