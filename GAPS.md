# NeuralFree: Known Limitations

This page lists only limitations that can be checked against the code. Each item names the class, or the file and line, that refuses the operation or holds the defect. For what the GGUF engine does support, see [LLM-INFERENCE.md](LLM-INFERENCE.md); for the training framework, see [TRAINING.md](TRAINING.md) and [USAGE.md](USAGE.md#the-training-framework-on-a-gpu-backend).

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
- **Paged KV memory runs on the CPU path; the GPU path has no paging.** The paged form (`memory/paged`: `PagePool`, `PageList`, `PagedSequences`, `PagedStep`, `PageCopy`, `RowParts`) runs when `neuralfree.kv.paged=true` or `neuralfree.parallel > 1`; by default `ModelMemory.sequences()` returns `SingleSequenceMemory`. On the CPU, `CpuForwardBackend.forwardBatch` splits a batch into one-sequence ubatches (llama.cpp `split_seq`), reserves each ubatch's pages all-or-nothing (`PagedSequences.prepare`), stores K/V at the step's `cellIdx` rows, attends through the step's page tables (`CpuAttentionCore.attendRows`' paged form) and commits; a batch over several sequences computes the same logits as the same tokens run one sequence at a time (`CpuForwardBatchTest`). Batched steps over embedding rows or explicit RoPE sections refuse (`forwardBatch`).
- **The GPU paged flash-attention kernels have no caller.** `AttnKernels` lists `FLASH_DECODE_PAGED_*` and `FLASH_ATTN_PAGED_*` (OpenCL and GLSL sources, in the `kernels/llm` inventory), but no mixer dispatches them: `GpuAttentionMixer` reads the base kernels, and `GpuForwardBackend` does not override `forwardBatch`, so its memory stays single-sequence. The host-side paged attention (`CpuAttentionCore.attendRows` with a page table) is wired; the GPU kernels wait for the device page tables of the GPU paging epic.

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
- **Structural tags compile without trigger gating.** `StructuralTag` compiles the vLLM legacy payload's structures (begin literal, schema grammar, end literal) into one alternated grammar, but a non-empty `triggers` array refuses: trigger-gated free text around the structures is not compiled — the grammar requires a structure outright. The modern xgrammar `format` object refuses too (pending).

## HF checkpoint loading

Hugging Face checkpoints open through the same engine ([USAGE.md](USAGE.md#hugging-face-checkpoints)). These parts are refused or carried but unused:

- **Vision/audio tower conversion.** An HF checkpoint's MMPROJ tower (vision or audio encoder) is not converted: the generated `HfArchTableMmproj` rows and `ModelSources.openProjector` reserve the seam, but the tower hook classes (67 converter classes) are not ported, because the engine has no vision or audio encoder to feed. The tower tensors are dropped and what was dropped is recorded (`neuralfree.hf.dropped_prefixes`); the checkpoint's text side loads and runs. The MMPROJ epic owns the conversion.
- **FP8 KV-cache scales are carried but unused.** A checkpoint's `kv_scale`, `q_scale` and attention-prob scales are loaded as the per-layer tensors `blk.N.attn_k.kv_scale`, `blk.N.attn_v.kv_scale`, `blk.N.attn_q.q_scale` and `blk.N.attn_prob_scale`, but attention does not consume them: the FP8 KV cache is a cache-dtype feature, not a loading feature (`KvScaleCollector` emits the tensors; the cache-side consumer is not written yet).
- **Pickle checkpoints load, but only torch's own payload.** `pytorch_model*.bin`, `*.pt` and `*.pth` are read by `TorchCheckpointFactory`, a built-in `CheckpointContainers` factory, through `RestrictedUnpickler`, a pickle-bytecode interpreter that accepts opcodes and globals by allow-list. Everything else is refused with a named error at load: a `STACK_GLOBAL` or `REDUCE` outside the allow-list (torch's `_rebuild_tensor_v2` and the standard dict/list primitives are what is on it), a `BUILD` whose state is not a dict or an item list, a `BINPERSID` whose persistent id does not resolve to tensor storage, and pickles past the size, opcode-count, stack and string limits. Object graphs, custom classes and non-tensor payloads are therefore refused, not silently loaded.
- **torchao and Ascend ModelSlim checkpoints are rejected.** Their tensor layouts are defined in packages not on disk, so no layout can be specified without guessing; both are refused with a named error. The registry-based online quantizer (`OnlineQuantizers`) is the seam these formats attach through later.
- **Other refused quantization formats.** `checkpoint_format: marlin`, compressed-tensors `sparsity_config`, the `quant_method`s `aqlm`, `eetq`, `vptq`, `spqr`, `higgs` and `quanto`, and Quark two-level weight specs (see [USAGE.md](USAGE.md#rejected-formats) for the reasons): vLLM rejects, deprecates or has removed each, and NF matches. The same refusal covers compressed-tensors transforms outside the input location (`output`, `k_cache`, `q_attn`), non-Hadamard transform types, and fp_quant's `transform_init: gsr`.
- **K-quant, IQ and FP8 online quantization.** `--quant-online` quantizes float matmul weights into Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL only. Online quantization into K-quants, IQ types, FP8 and TQ/MXFP4 has no quantizer registered yet (`OnlineQuantizers` is the seam); a float tensor is served at full precision or F32-fallbacked when its target type is not among these.
- **Tensor-core W8A8/FP8 kernels.** `--act-quant checkpoint` fake-quantizes input activations and runs the unchanged weight kernel, so it computes exactly (no speedup). Real int8/FP8 Tensor Core GEMM kernels, and real int8 GEMM speedups for the extension types, are not implemented (`ActQuantScheme` has the schemes; no such kernels exist).
- **vLLM draft classes.** As HF draft checkpoints, NF loads EAGLE-3 and its vLLM alias classes with the Llama-EAGLE3 layout, the EAGLE-1 heads (`EagleForCausalLM`, `EagleModel`, `EagleLlamaModel`, SGLang's `LlamaEagleForCausalLM` alias, and `EagleDeepSeekMTPModel`, through `ArchTableDraft` and `HfEagle1Hooks`), the DFlash/DSpark drafters and the Gemma-4 assistant. The remaining `Eagle3*`/`Peagle*`/`Eagle3Deepseek*` fusion classes (for example `Eagle3MiniMaxM2ForCausalLM`, `PeagleLlamaForCausalLM`, `PEagleDraftModel`), `DFlash2BeamDraftModel`, `DSparkV41DraftModel`, `Qwen3OmniDSparkModel`, `K3DSparkModel`, `DFlashLagunaForCausalLM` and standalone MTP checkpoints are refused: the draft-class table (`HfArchTableDrafts`) has no rows for them.

- **Speculative verify modes are programmatic only.** How drafted tokens are checked — `match`, `rejection`, `block`, `fly` or `synthetic` — is set through `SpeculativeDecoder.Builder.verifyMode(...)`; no command-line flag exposes it (`--spec-type` selects the drafter, never the verifier).

## Demo tools

- **nftool implements four subcommands.** `demo/tools/NfTool` runs `export` (`--mode native|compat|compat-exact`, `--overwrite`, `--vocab-only`, `--split-max-tensors`, `--split-max-size`), `download`, `imatrix` (`--file`, `--out-file`, `--chunk-size`; the CPU-engine collector `cpu/quant/ImatrixCollector` over `CpuMatVec.observeGlobally`, writing through `quant/write/ImatrixFile.writeGguf`) and `cvector` (the prefill loop over one loaded CPU engine, `CvectorGenerator.generate`); every other listed subcommand (`quantize`, `perplexity`, `results`, `bench`, `batched-bench`, `split`, `merge`, `export-lora`, `tokenize`, `dump`, `hash`, `embedding`, `retrieval`) refuses with `not implemented yet`. Both collection tools open their model on the CPU engine by design: the imatrix tap reads the CPU matmul inputs and the cvector tap reads the CPU hidden stream; the GPU event-tap route of `CvectorGenerator.EventTap` has no driver yet.
- **The session file covers the single-sequence CPU path.** `HeapModelMemory.saveSession`/`loadSession` (`memory/cpu/SessionFile`) serialize the tokens, the KV-group position mirrors, the owning cache layers' K/V rows and the live recurrent state. A paged memory (its page lists and sequences), a plan with DSV4 compressed caches, sparse side caches (DSA indexer, MSA, T5 cross-attention) and caches reading a target model refuse by name; snapshot shadows, compressor rollback planes and the sampler chain stay runtime-only.
- **The prefix cache keeps retired sequences only, within the slot budget.** `PrefixCache` indexes sequences the batch engine retired (bounded by `neuralfree.parallel`: every cached prefix occupies one memory slot), matches exact token arrays, and adopts a match in place. A live request's pages are never shared into a second concurrent request (two submissions with the same running prefix each prefill their own pages), there is no independent cache budget above the slot count, and eviction is LRU over retired sequences alone (`BatchEngine`).
- **Streamed logprobs stop at the sampled chain.** `/v1/completions` and `/v1/chat/completions` carry the OpenAI `logprobs.content` shape on the non-streamed responses and per chunk on a stream (`OpenAiResource`, through the `ReasoningFilter.Sink` capture overload). Speculation reports the capture for the default `MATCH` verify mode and the plain decode steps (`SpeculativeDecoder`); the lossless verifier modes (`rejection`, `block`, `fly`) emit their tokens through the verifier's round, which reports no scores yet.
- **The server serves the OpenAI and Anthropic surfaces; the other adapters are pending.** `server/` (Quarkus, Java 21, bound to `127.0.0.1:8080`) serves `GET /v1/models`, `POST /v1/completions` and `POST /v1/chat/completions` with SSE streaming and streamed logprobs, `POST /v1/embeddings`, `POST /v1/rerank`, the Anthropic `POST /v1/messages` and `/v1/messages/count_tokens` (`AnthropicResource`; non-text content blocks refuse by name), and the llama.cpp-native `GET /health` and `GET /props` (`OpenAiResource`, `LlamaCppResource` over `EngineService` and the `gguf/engine/batch` engine). Pending: tool-call output parsing (the `ToolParserProvider` has no consumer), the WebSocket and gRPC adapters of `package-info.java`, and authentication, CORS and TLS.
- **The plugin SPI is partially consumed.** `gguf/plugin` carries five provider interfaces (`ArchProvider`, `HfArchProvider`, `ProjectorProvider`, `ToolParserProvider`, `ReasoningParserProvider`) loaded through `Plugins.load` from `META-INF/services` on the application class path. `ArchProvider` and `HfArchProvider` are merged into `ArchRegistry` and `HfArchRegistry`; the projector, tool-parser and reasoning-parser providers have no consumer registry yet, so registering them does nothing.

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
