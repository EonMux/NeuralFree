# NeuralFree: Known Limitations

This page lists only limitations that can be checked against the code. Each item names the class, or the file and line, that refuses the operation or holds the defect. For what the GGUF engine does support, see [LLM-INFERENCE.md](LLM-INFERENCE.md) and [LLAMACPP-PARITY.md](LLAMACPP-PARITY.md); for the training framework, see [TRAINING.md](TRAINING.md) and [USAGE.md](USAGE.md#the-training-framework-on-a-gpu-backend).

## GGUF engine: out of scope

- **Vision and audio projectors.** `clip` (the `mmproj` projector files) is rejected at load:
  > architecture 'clip' is a vision/audio projector without a text model ... load the language model GGUF instead

  (`ArchRegistry`). The text part of a vision-language model runs on text tokens only. No image or audio encoder feeds it.
- **gptj.** Rejected because llama.cpp has no model implementation for it (`ArchRegistry`).
- **Unknown architectures.** A `general.architecture` that names no llama.cpp architecture is rejected (`HparamsLoader`, `ModelSpecBuilder.resolveEntry`).
- **Waveform synthesis.** `wavtokenizer-dec` returns spectral features. The iSTFT that turns them into audio samples is not part of the engine (`LLMDemo` audio mode).

## GGUF engine: speculative decoding

- **MTP drafting needs a draft file of the target's own model.** The NextN blocks of a draft file draft only for the model they were trained with. The file must match the target in all of the following:
  - architecture;
  - trunk depth;
  - trunk layer schedule;
  - `n_embd`;
  - vocabulary size.

  The check is `ModelSpecBuilder.mtpDraftMismatch`. When it fails:
  - under `--spec-type draft-mtp` the file is refused;
  - under `auto`, a file that holds only MTP blocks is refused, and a full model with NextN blocks drafts as a separate draft model instead (`LLMModelFactory.draftRoute`).

  llama.cpp checks only that the draft's and the target's `n_embd_out` agree (`common/speculative.cpp`, `common_speculative_impl_draft_mtp`), so it accepts the MTP blocks of any model with the same width.
- **Archs with NextN tensors but no MTP graph.** These architectures carry NextN tensors, but llama.cpp defines no MTP draft graph for them:
  - glm4, nemotron_h, exaone4, exaone-moe, bailingmoe2, dots3note, mistral4;
  - clef, which is a classifier.

  Drafting with their blocks is refused: "has no NextN/MTP draft graph in llama.cpp" (`NextnDraftModel`).
- **Several NextN blocks.** A file with several NextN blocks drafts only on hy_v3, mimo2 and step35, whose heads chain. Elsewhere it is refused, as llama.cpp asserts `n_layer_nextn == 1` (`NextnDraftModel`).
- **MTP-only files.** An MTP-only file cannot be opened as a model of its own; it is accepted only as the draft of its target (`ModelSpecBuilder.refuseMtpOnly`).
- **Targets that cannot be speculated on.** Speculation is refused for:
  - targets of class EMBED, CLS, NA, DIFF, ENC_DEC or DRAFT, which have no decoder-only token loop;
  - targets without model memory.

  The checks are in `LLMModelFactory` and `SpeculativeDecoder`.
- **Draft models.** A separate draft model must be a GEN model with memory and a vocabulary compatible with the target's (`ModelDrafter`).
- **One drafter per target.** A target serves one drafter at a time, because the target's forward backend (`ForwardBackend`, one per loaded model) has one hidden-state tap (`HiddenTapSlot`).
- **Separate draft models on DeepSeek-V4.** llama.cpp falls back to speculative state checkpoints when a draft is longer than the target's `n_rs_seq` (`tools/server/server-context.cpp`). NeuralFree has no checkpoint path for DeepSeek-V4 compressor state, and a separate draft model gives the target no rollback snapshots, so on a DeepSeek-V4 target it drafts nothing (`SpeculativeDecoder`).

## GGUF engine: memory

- **Context shift is refused** (`DeviceStepInputs.SequenceRules.shiftBlocker`, `ModelSpec.contextShiftRefusal`) for:
  - recurrent state;
  - DSA indexer, MSA index-key and DeepSeek-V4 compressed caches;
  - layers that read a target model's cache;
  - several positions per token: M-RoPE sections, `n_pos_per_embd > 1`, and sensenova_u1;
  - step35, which has per-layer RoPE dims.

  It is also refused when a sliding-window ring has already overwritten kept positions that the next query attends (`ringShiftBlocker`). In every case, nothing moves.
- **Rollback is refused** (`SequenceRules.rollbackBlocker`) for:
  - recurrent state, which rolls back through snapshots;
  - DeepSeek-V4 compressor state beyond its rollback snapshots (`CompressorRollback`): more than `n_rs_seq` positions (`MemoryConfig.rollbackSnapshots`), a second rollback before the next step, or more positions than the last step stored;
  - a ring that no longer holds the window of the next query.
- **Snapshots are refused** (`SequenceRules.snapshotBlocker`):
  - with DeepSeek-V4 compressed caches;
  - when the memory was planned without snapshot buffers: `neuralfree.state.snapshots=false`, or the default without a configured drafter.
- **K and V types.** MLA models and DeepSeek-V4 need equal K and V cache types (`MemoryPlanner.checkKvTypes`).
- **Block types need whole blocks.** Quantized cache types need every head to be whole 32-value blocks (`KvDType`).
- **MLA restrictions** (`MlaBlocks.unsupportedCommon`):
  - on the GPU, a decompressed MLA cache over absorbed `attn_k_b` / `attn_v_b` weights has no kernel (`GpuMlaMixer.unsupported`);
  - MLA without a KV cache is refused;
  - an MLA layer with a position bias is refused;
  - an MLA layer with a Q-interleaved gate is refused;
  - per-head MLA products take plain weights, so a PRISM Hadamard or a tensor scale on them is refused (`GpuMlaMixer`, `CpuMlaMixer`).

## GGUF engine: weights and devices

- **CPU weight types.** The CPU engine has no matmul kernel for Q8_1, Q8_K, the integer types or F64 weights, matching ggml-cpu (`CpuActivationQuantizer`, `CpuMatVec`). The GPU path covers Q8_K natively. It dequantizes the other types to F32 at upload.
- **Removed types.** Q4_2 and Q4_3 were removed from ggml and are rejected (`CpuBlockDecoders`).
- **Device work-group limits.** Some kernels need work-group sizes or dispatch grids the device must allow; the load fails with the limit named. This applies to:
  - the matvec work-group multiple (`GpuOps`);
  - the MoE and value-expert grids (`GpuMoeFfn`, `GpuValueExperts`);
  - OpenCL's `CL_KERNEL_WORK_GROUP_SIZE` and local-memory limits (`OclKernelHandle`, `OclComputeBackend`).
- **Sharded embedding tables.** An embedding table split into row shards is looked up with host-side token ids; device-side ids cannot select the shard (`GpuOps`).
- **WavTokenizer posnet attention.** On the GPU, the posnet attention width must be a multiple of 4 within the flash-attention tile (`GpuWavTokenizer`).
- **Backend availability.** Each GPU backend reports the LLM kernel library unavailable, and AUTO skips it, when:
  - CUDA: the CUDA kernels were not built (no nvcc at build time), no image matches the device, the driver is too old, or a kernel of the inventory is missing (`CudaKernelLibrary`);
  - Vulkan: the LLM shaders were not built (no shader compiler, glslc or glslangValidator, or no spirv-dis at build time), a kernel of the inventory is missing, or the device lacks 8-bit storage, 16-bit storage or timeline semaphores (`VulkanKernelLibrary`, `KernelLibraries`).

## GGUF engine: LoRA

- **Activated LoRA** (`adapter.alora.invocation_tokens`) is refused, because its per-token mask is not carried by the forward passes (`LoraAdapterLoader`).
- **Whole attachment fails** (`LoraAttachment`) when an adapter pair targets:
  - a tensor no bound block applies deltas to;
  - a block that refuses deltas: a PRISM Hadamard-rotated input, or the absorbed MLA `attn_kv_b`.

## GGUF engine: model modes

- **`encode`** needs an embedding model or a pooling type (`EncodeGate`).
- **Reranking** needs `RANK` pooling. A model whose rerank prompt template takes text must be reranked from text, not tokens (`EncodeGate`, `ClassifierHead`).
- **Decision models** score with `classify` only (`EncodeGate`).
- **Masked diffusion** (`generateDiffusion`) is for DIFF models only (`DiffusionGenerator`).
- **Audio-code decoding** (`decodeAudioCodes`) is for `wavtokenizer-dec` only (`GGUFModel`, `GGUFModelGPU`).
- **Draft-class files** (eagle3, gemma4-assistant, dflash) load only beside a target, as drafters (`GGUFModel`, `GGUFModelGPU`, `DraftModelLoader`).
- **Models without a vocabulary** (`tokenizer.ggml.model = none`) cannot tokenize text (`NoVocabTokenizer`).
- **Legacy `generate`.** The positional `LLMModel.generate(...)` overloads refuse request features they cannot honour (`LLMModel`). Use `generate(int[], GenerationRequest, TokenCallback)`.

## Training framework: device execution

The training framework (tensors, autograd, `nn`, optimizers, trainers) and the classical-ML classes run on the `train` kernel library of the tensors' backend: OpenCL, CUDA or Vulkan. The numerical contract and the host-path rules are documented in `tensor/gpu/package-info.java`.

- **Library availability.** `TrainGpu` reports the train library unavailable on a backend when:
  - the kernel library itself is unavailable: CUDA kernels not built, no image for the device, the driver too old, or a kernel of the train inventory missing from the build (`CudaKernelLibrary`); Vulkan shaders not built, no timeline-semaphore support, or a kernel of the train inventory missing from the build (`VulkanKernelLibrary`, `KernelLibraries.TRAIN_CONFIG`); OpenCL sources that cannot be assembled (`OpenClKernelLibrary`);
  - the device cannot run work-groups of 256 items (`TrainGpu.LINEAR_LOCAL`);
  - a kernel of the library fails to load or prepare.

  The reason is logged once at WARN and returned by `TrainGpu.availability(backend)`.
- **CPU fallback on such a backend.** Each tensor dispatcher (`tensor/Gpu*Ops`) and each `ml` class then runs its CPU reference: the operands are read to the host and the result is uploaded, for every operation.
- **No CPU path.** `PipelineScheduler` (`acceleration/pipeline`) and `ai.v2.learning.BackPropagate` throw `UnsupportedKernelException` when the train library is unavailable on their backend.
- **Device results are not always bit-identical to the CPU.** Data movement, comparisons, arg-reductions and max-pool are bit-identical. Single-rounding arithmetic may flush values below `FLT_MIN`. Division, transcendental and reduction results agree within the tolerance classes of `tensor/gpu/package-info.java`, because Vulkan without `VK_KHR_shader_float_controls` and OpenCL 1.2 allow inexact division and square root, looser transcendentals and flushed denormals.
- **Element limit.** The kernel ABI addresses at most `2^31-1` elements per tensor; larger tensors are rejected with `IllegalArgumentException`.
- **Deferred index errors.** On a device, an out-of-range gather, scatter or embedding index is recorded by the kernel and thrown as `IndexOutOfBoundsException` at the backend's next host synchronization, not by the call that launched it (`TrainGpu`, `nn/layers/Embedding.java`). The CPU path throws at once.
- **Sharing a backend with the GGUF engine.** Tensor ops serialize on the backend's lock (`BackendResources.lock`); the LLM engine does not take it. One backend instance must not be used by the LLM engine and by tensor ops from different threads at the same time (`TrainGpu`).

## Training framework: host readbacks

Tensor math stays on the device. Data crosses to the host only in these categories (`tensor/gpu/package-info.java`); each is a synchronization point.

- **Reproducible random streams.** Masks and noise are drawn from a `java.util.Random` on the host, so equal seeds give equal values on every backend, and uploaded once. Sites:
  - `Tensor.rand`, `Tensor.randn`;
  - dropout masks: `Dropout`, the attention dropout of `GATConv`, the embedding dropout of `TextEncoder` and the attention-output dropout of `CrossAttentionFusion` (`nn/multimodal`);
  - `VAE` reparameterization noise, `DDPMTrainer` and `DDIMSampler` noise, the `GANTrainer` latent noise and gradient-penalty mixing weights, `SAC` policy noise;
  - `MaskedAutoencoder` and `MaskedLanguageModel` masks, `ContrastiveLearner` augmentations.
- **Java control decisions.** A value that Java control flow consumes is read back as the smallest tensor that carries it:
  - the logged loss and metric scalars of `Trainer` and `MixedPrecisionTrainer`, the early-stopping and checkpoint decisions built on them, and the gradient-overflow flag of `MixedPrecisionTrainer`;
  - sampled actions, action probabilities and values in `rl` (A2C, PPO, DQN, SAC, RLHFTrainer), and the DQN TD errors under prioritized replay;
  - sampled tokens in `AutoregressiveGenerator` and `inference/SpeculativeDecoder`;
  - the top-k expert indices of `MixtureOfExperts`, which select the expert modules that run, the node selection scores of `TopKPool`, and the routing decision of `ModalityRouter.route`;
  - the per-iteration convergence scalar of `KMeans` (centroid movement) and `LogisticRegression`;
  - the `num_batches_tracked` counter of a BatchNorm layer with a null momentum (cumulative average), one 4-byte read per training forward pass (`nn/norm/AbstractBatchNorm.java`).
- **Cross-process and serialization I/O.** `Module.stateDict`/`loadStateDict`, `io/ModelSerializer`, `io/ONNXExporter`, `training/ModelCheckpoint`, the collectives of `distributed/TCPBackend` and `distributed/SharedMemoryBackend`, `Module.to(ComputeBackend)`, and moving a tensor to another backend (`DataLoader`, `PipelineParallel` stages on different backends).
- **Sequential factorizations.** The Cholesky factorization and triangular solves of `GaussianProcess`, the Jacobi eigen-iteration of `PCA` and the SMO pair selection of `SVM` run on the host. The kernel, covariance and prediction products feeding them run on the device.
- **CPU-only algorithms.** `RandomForest`, `GradientBoostedTrees` and `HiddenMarkovModel` (`ml`) and the per-pixel compositing of `GaussianSplatting` (`nn/structured`).
- **Graph structure metadata.** CSR offsets and degrees are built once per graph from its integer edge index and cached as device tensors (`nn/graph/GraphData.java`, `MessagePassing`).
- **FLOAT64.** Conversions to and from FLOAT64 run on the host, since the library does not require fp64 device support (`Tensor.to`).

## Training framework: autograd

- **Create-graph passes.** `Autograd.grad(..., createGraph = true)` and a create-graph `backward` throw `UnsupportedOperationException`, naming the operation, when a node they would run cannot record its own backward (`GradientFunction.supportsCreateGraph`). Two gradient functions cannot:
  - a checkpointed segment (`CheckpointFunction`);
  - the Gaussian splatting rasterizer (`nn/structured/RasterizeGrad.java`).

  Second-order uses (second-order `MAML`, the WGAN-GP gradient penalty of `GANTrainer`) therefore fail on models that contain either.
- **`Autograd.grad` and checkpointing.** `Autograd.grad` cannot run through a checkpointed segment, because the segment's recompute writes leaf gradient slots; use `backward` instead (`ComputationGraph`).
- **Auxiliary losses inside a checkpointed segment get no gradient.** The checkpointed forward pass runs under `noGrad`, so a module's auxiliary term (`Module.setAuxiliaryLoss`, such as the `MixtureOfExperts` load-balancing term or the `VAE` KL term) is a constant. The term its recompute records is discarded with the recompute's nodes.

## Training framework: modules, losses and tensors

- **Tensor factories for other dtypes.** `Tensor.ones`, `rand`, `randn` and `eye` create FLOAT32 tensors only: another floating-point dtype throws `UnsupportedOperationException`, and an integer or boolean dtype throws `IllegalArgumentException` (`tensor/Tensor.java`).
- **`Tensor.to(DType)`** converts FLOAT32 to and from FLOAT64, INT32, BOOL, FLOAT16 and BFLOAT16, and INT64 to INT32. Every other pair throws `UnsupportedOperationException`.
- **State dicts hold `float[]` values.** `Module.stateDict` returns every entry as `float[]`: FLOAT64 values are rounded to float, and integers are exact only up to `2^24`. `loadStateDict(map)` ignores entries whose value is not a `float[]`, and `loadStateDict(map, true)` refuses them. An INT8, INT16 or INT64 parameter cannot be loaded from a `float[]` and throws `UnsupportedOperationException` (`nn/Module.java`).
- **BatchNorm batch counter.** `num_batches_tracked` is a FLOAT32 scalar, not PyTorch's int64, because the tensor layer has no integer in-place add. It counts exactly up to `2^24` batches and then stays there (`nn/norm/AbstractBatchNorm.java`).
- **Integer class targets.** `CrossEntropyLoss` and `FocalLoss` take class-index targets as INT32, INT64 or FLOAT32 (rounded to the nearest integer). The class values pass through FLOAT32, so INT32 and INT64 targets are refused for predictions of more than `2^24` classes, and the ignore index must lie in `(-2^24, 2^24)` (`nn/loss/LossTensors.java`).
- **`KLDivLoss` defaults to `batchmean`.** `new KLDivLoss()` divides the summed divergence by the batch size, the mathematical KL divergence per sample. PyTorch's default reduction is `mean`, which divides by the element count. `new KLDivLoss(LossFunction.Reduction.MEAN)` gives PyTorch's default.
- **Max-pool padding.** `ConvOps.poolOutputSize` accepts a padding up to half the effective window `dilation * (kernel - 1) + 1`. PyTorch's `max_pool2d` limits it to half the kernel size, so a dilated max-pool may accept a padding that PyTorch rejects. Average pooling has no dilation, so its rule is the same as PyTorch's.
- **GELU is the tanh approximation.** `TensorOps.gelu` and the `GELU` module compute `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`. PyTorch's default exact (erf) GELU is not available.
- **No 3-D convolution.** `nn/layers` has `Conv1d`, `Conv2d` and `ConvTranspose2d`, but no `Conv3d`.
- **`NeuralODE` takes fixed steps.** Every solver (Euler, RK4, Dormand-Prince) advances over the fixed time grid. `rtol` and `atol` are validated and kept, but do not control the step size (`nn/structured/NeuralODE.java`).

## Training framework: distributed training

- **`TCPBackend` is unauthenticated and unencrypted.** Any host that can reach a rank's port can join as a rank, read the exchanged values and inject its own (`distributed/TCPBackend.java`). Run it only on a trusted, private network with the rank ports firewalled from other hosts.
- **Collectives cross the host.** `TCPBackend` and `SharedMemoryBackend` move values through host memory; there is no device-to-device (NCCL-style) path. In-place results (`allReduce`, `broadcast`) need FLOAT32 tensors. `TCPBackend.allReduce` reduces on rank 0 and sends the result back.
- **`PipelineParallel` stages on different backends** exchange activations through the host. Stages that share a backend are serialized by that backend's lock.
- **`FederatedTrainer` simulates its clients with one optimizer.** The optimizer's internal state (moments, step counts) carries from one client to the next. Local training uses the mean-squared-error loss. Clients are selected with an unseeded `java.util.Random`, so client selection differs from run to run.
- **`FederatedMetaLearner`** reports the global model's evaluation loss per round as `FederatedMetaResult.roundAccuracies()`.

Java paths in the training sections are relative to `src/main/java/com/eonmux/neuralfree/`.
