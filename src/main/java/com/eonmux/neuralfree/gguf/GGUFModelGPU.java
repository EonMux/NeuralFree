package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.DiffusionGenerator;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.QuantizedView;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.PoolingOperator;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuProfiler;
import com.eonmux.neuralfree.gguf.gpu.diag.ObserverSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.pipeline.SplitRequest;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuSplitSession;
import com.eonmux.neuralfree.gguf.gpu.parallel.EplbRebalancer;
import com.eonmux.neuralfree.gguf.gpu.parallel.EplbSession;
import com.eonmux.neuralfree.gguf.gpu.parallel.TensorParallelGroup;
import com.eonmux.neuralfree.gguf.gpu.parallel.FfnRowSplitPlan;
import com.eonmux.neuralfree.gguf.gpu.parallel.TpAttentionPlan;
import com.eonmux.neuralfree.gguf.gpu.parallel.TpMoePlan;
import com.eonmux.neuralfree.gguf.gpu.parallel.TpVocabPlan;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUpdater;
import com.eonmux.neuralfree.gguf.gpu.weights.HostMappedPlan;
import com.eonmux.neuralfree.gguf.gpu.weights.HostMappedWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.HostPlacement;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertPrefetch;
import com.eonmux.neuralfree.gguf.gpu.weights.PlacementFitter;
import com.eonmux.neuralfree.gguf.gpu.weights.StreamingLayerWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hybrid.HybridForwardBackend;
import com.eonmux.neuralfree.gguf.hybrid.HybridMemory;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.audio.GpuWavTokenizer;
import com.eonmux.neuralfree.gguf.ops.encdec.GpuEncDecBackend;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.IntUnaryOperator;
import java.util.function.Predicate;

/**
 * A GGUF model on a GPU backend (OpenCL, CUDA, Vulkan, HIP or Metal), assembled from the spec-driven engine:
 * <ol>
 *   <li>the model's {@link ModelSource} ({@link ModelSources} with the configuration's
 *       {@link EngineConfig#sourceOptions()}), its metadata ({@link GgufKv}) and the resolved {@link ModelSpec};</li>
 *   <li>the LLM kernels and the weight upload plan ({@link GpuWeightUploader}, PRISM transforms included);</li>
 *   <li>the memory plan ({@link MemoryPlanner} with {@link ContextFitter} against the free VRAM), the device memory
 *       ({@link DeviceModelMemory}) and the weight placement ({@link PlacementFitter}): every weight resident, the
 *       placement that fits across the device and the host streamed through a {@link StreamingLayerWeightSource}, or
 *       a hybrid CPU+GPU split ({@code com.eonmux.neuralfree.gguf.hybrid}, llama.cpp {@code -ngl}): the last layers
 *       on the device, the earlier ones on the host, each side with its KV on its own device and the activations
 *       crossing at the split ({@code HybridForwardBackend});</li>
 *   <li>the {@link GpuForwardContext}, the block registry of every family ({@link GpuFactoryCatalog}) and the
 *       forward backend ({@link GpuForwardBackend}, or {@link GpuEncDecBackend} for encoder-decoder models); or,
 *       with {@code neuralfree.split.mode=layer}, the split session ({@link GpuSplitSession}) that runs the model
 *       over several GPUs, each holding a contiguous span of layers.</li>
 * </ol>
 * Generation runs through {@link GenerationLoop}, masked diffusion through {@link DiffusionGenerator}, embeddings,
 * classification and reranking through the host heads over {@link ForwardBackend#hiddenRows}, and audio codes through
 * {@link GpuWavTokenizer}. Not thread-safe.
 *
 * <p>Weight placement ({@link GpuLoadPlacement}): with an automatic context, every weight stays resident when the
 * largest context that then fits the remaining VRAM is at least {@value GpuLoadPlacement#AUTO_CONTEXT_FLOOR} tokens
 * (or the training context when shorter);
 * otherwise {@link PlacementFitter} fits the weights across the device and the host from the free VRAM: the expert
 * tensors that do not fit stream first while their layers' dense tensors stay resident, then whole trunk layers, and
 * the automatic context is capped at that floor so the rest of the VRAM holds streaming slots. A user context
 * ({@code neuralfree.ctx}) is honoured as given. The context is fitted beside the memory allocated after loading: the
 * NextN / MTP draft caches ({@link MemoryConfig#nextnMemory()}) and a configured draft model
 * ({@link LLMModelFactory#draftCompanion}); that memory stays out of the budget of the streaming slots.
 * Encoder-decoder models and the WavTokenizer decoder keep every weight resident and evaluate their whole input in one
 * step, so their ubatch is raised to the context ({@link MemoryConfig#forModel}).</p>
 *
 * <p>Weights off the device ({@link WeightPlacement}): the tensors only the host heads read are never uploaded nor
 * charged to the fit, and the NextN / MTP blocks are placed only when MTP is loaded, as llama.cpp loads them only with
 * {@code load_mtp}: when {@link EngineConfig#mtp()} or {@link MemoryConfig#nextnMemory()} asks for the model's own
 * NextN draft context. An MTP drafter that attaches to a model opened without it places them on demand
 * ({@link #ensureNextnWeights()}).</p>
 */
public final class GGUFModelGPU implements EngineModel {

    private static final Logger LOGGER = LogManager.getLogger(GGUFModelGPU.class);

    private static final double MIB = 1024.0 * 1024.0;

    final ModelSource source;
    final GGUFTokenizer tokenizer;
    final String modelName;
    /** The path the model was loaded from; its expert ranking session looks for the ranking file next to it. */
    private final String modelPath;
    private final SamplingDefaults samplingDefaults;
    final ObserverSlot observer = new ObserverSlot();

    EngineConfig config;
    ModelSpec spec;
    ModelSpec forwardSpec;
    MemoryPlan plan;
    VRAMBudget.FreeVram freeAtLoad;
    long allocatedAtLoad;
    HWAcceleration ownedAcceleration;
    ComputeBackend backend;
    ComputeQueue queue;
    /** The split session when the model runs over several GPUs ({@code neuralfree.split.mode=layer}); it owns the
     * backends, the per-device memory and the forward backend. */
    GpuSplitSession split;
    /** The tensor-parallel group when the model runs in row or tensor mode ({@code neuralfree.split.mode=row} or
     * {@code =tensor}); it owns the peer backends, the rank stacks, the shard weights and the per-rank caches. */
    TensorParallelGroup parallelGroup;
    /** The split of the row/tensor mode: the eligible trunk FFN layers and their per-rank tensors. */
    FfnRowSplitPlan rowSplitPlan;
    /** The attention split of the tensor mode, or null in row mode. */
    TpAttentionPlan attentionPlan;
    /** The vocabulary split of the tensor mode, or null in row mode. */
    TpVocabPlan vocabPlan;
    /** The expert split of the tensor mode, or null when the model has no routed experts or in row mode. */
    TpMoePlan moePlan;
    LlmKernels kernels;
    TensorCatalog catalog;
    InputTransformLoader.Result transforms;
    WeightStore store;
    /** The store the streamed trunk is read from, opened for {@link StoreUse#STREAMING} when streaming is chosen. */
    WeightStore streamingStore;
    /** Trunk tensors kept in host-mapped memory ({@code neuralfree.cpu.offload.mb}); empty when off. */
    HostMappedPlan hostMappedPlan = HostMappedPlan.empty();
    /** The host-mapped tensors over the streamed or resident ones; null when nothing is host-mapped. */
    HostMappedWeightSource hostMapped;
    GpuPlaceholders placeholders;
    GpuScratch scratch;
    GpuOps ops;
    MoeCacheRanking ranking;
    /** The expert-parallel load balancer, or null when {@code neuralfree.eplb} is off. */
    EplbSession eplb;
    /** The runtime expert rearrangement, or null when {@code neuralfree.eplb.rebalance} is off. */
    EplbRebalancer eplbRebalancer;
    DeviceModelMemory memory;
    GpuWeightUploader uploader;
    /** The tensors placed on the device; replaced when the NextN blocks are placed on demand. */
    WeightPlacement weightPlacement;
    StreamingLayerWeightSource streaming;
    List<LayerWeightSet> streamedSets = Collections.emptyList();
    /** The MoE layers whose routed experts the placement keeps in RAM. */
    GpuHostedLayers hostedLayers = GpuHostedLayers.NONE;
    GpuForwardContext ctx;
    GpuBlockRegistry registry;
    SwitchedLoraExtra.Router router;
    ForwardBackend rawBackend;
    ForwardBackend forward;
    LoraAttachment loras;
    /**
     * The per-request adapters the LoRA attachment serves (the split session's for a split model); null for
     * wavtokenizer-dec.
     */
    LoraRegistry loraRegistry;
    /** The control vectors and LoRA adapters, loaded before placement. */
    GpuAdapters adapters;
    /** Device bytes still to be allocated after placement (the adapter uploads), held back from the VRAM budget. */
    long pendingDeviceBytes;
    /** The configured draft model's memory, fitted beside the context ({@link LLMModelFactory#draftCompanion}). */
    ContextFitter.Companion drafter = ContextFitter.Companion.NONE;
    /**
     * Device bytes the context fit held for memory a drafter allocates after loading (the NextN draft caches and the
     * draft model's memory, {@link ContextFitter#reservedBytes}), held back from the VRAM budget of the streaming
     * slots.
     */
    long drafterDeviceBytes;
    /** The bytes the load may allocate on its device ({@code neuralfree.gpu.vram.limit}), 0 for no limit. */
    private long vramLimitBytes;

    CpuWeights cpuWeights;
    CpuScratch cpuScratch;
    CpuForwardContext cpuContext;
    LayerOffload hybridSplit;
    /** The context and weight placement the load resolved ({@link GpuLoadPlacement}); null for a split load. */
    GpuLoadPlacement.Placement placement;
    HybridMemory hybridMemory;
    MoeExpertCache moeCache;
    /** The prefill expert prefetch over the cache; null without a cache or when the configuration disables it. */
    MoeExpertPrefetch moePrefetch;
    /** Trunk layers with routed experts, for the MoE expert cache. */
    List<LayerSpec> moeCacheLayers = Collections.emptyList();
    /** The resolved cache settings of the load: {@code neuralfree.moe.cache} and {@code .mb} as one configuration. */
    MoeCacheConfig moeCacheSettings = MoeCacheConfig.off();
    /** True when the configuration enables the cache and the model has routed MoE layers. */
    boolean moeCacheConfigured;
    /** The host-head tasks: embeddings, classification, reranking, logits and the audio decoder. */
    final GpuModelTasks tasks;
    private GenerationLoop generationLoop;
    private DiffusionGenerator diffusion;
    private boolean showAllChannels;
    private boolean closed;

    /**
     * Loads a model on the backend {@link EngineConfig#fromSystemProperties()} selects; the model owns that backend
     * and releases it in {@link #close()}.
     */
    public GGUFModelGPU(String path) throws IOException {
        this(path, null);
    }

    /**
     * Loads a model on {@code sharedBackend}, which stays owned by the caller, or, when it is null, on the backend
     * {@link EngineConfig#fromSystemProperties()} selects (then owned by the model).
     */
    public GGUFModelGPU(String path, ComputeBackend sharedBackend) throws IOException {
        this(path, sharedBackend, EngineConfig.fromSystemProperties());
    }

    /**
     * Loads a model with an engine configuration. When {@code sharedBackend} is null the model creates the backend of
     * {@code cfg.backend()} that can run the LLM kernels and releases it in {@link #close()}. When loading fails after
     * the file is opened, everything acquired so far is released before the exception leaves.
     *
     * @throws IllegalArgumentException          when {@code cfg} selects the CPU and no backend is given
     * @throws UnsupportedOperationException      for an unsupported architecture or a draft model (drafters load
     *                                           against their target through {@code DraftModelLoader})
     * @throws ContextFitter.ContextBudgetException when no allowed context fits the VRAM
     * @throws GpuOutOfMemoryException            when the weights do not fit even when streamed
     */
    public GGUFModelGPU(String path, ComputeBackend sharedBackend, EngineConfig cfg) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(cfg, "cfg");
        if (sharedBackend == null && !cfg.backend().usesGpu()) {
            throw new IllegalArgumentException("backend mode " + cfg.backend() + " runs on the CPU; load the model with"
                    + " GGUFModel, or pass a GPU backend");
        }
        this.tasks = new GpuModelTasks(this);
        this.source = ModelSources.open(path, cfg.sourceOptions(), null, cfg.activationQuant());
        this.modelPath = path;
        try {
            this.tokenizer = new GGUFTokenizer(source);
            String name = source.getString("general.name");
            this.modelName = name != null ? name : source.getString("general.architecture");
            GgufKv kv = GgufKv.of(source);
            this.samplingDefaults = SamplingDefaults.read(kv);
            load(sharedBackend, cfg, kv);
        } catch (Throwable failure) {
            releaseAfterFailedLoad(failure);
            throw failure;
        }
    }

    // ============================================================================================
    // Construction
    // ============================================================================================

    private void load(ComputeBackend sharedBackend, EngineConfig cfg, GgufKv kv) throws IOException {
        this.vramLimitBytes = cfg.gpuVramLimitBytes();
        this.catalog = TensorCatalog.of(source);
        ModelSpec initial = ModelSpecBuilder.build(kv, catalog, cfg.memory(), 0, cfg.pooling());
        if (initial.supportClass() == SupportClass.DRAFT) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " is a draft model; load it against"
                    + " its target model through DraftModelLoader");
        }
        if (initial.arch() == Arch.WAVTOKENIZER_DEC) {
            this.tasks.setSpeechFormat(SpeechFormat.ofGguf(kv));
        }
        GpuFactoryCatalog.checkRouting(initial);
        this.transforms = InputTransformLoader.load(source, kv, catalog, cfg.activationQuant());
        this.ranking = MoeCacheRanking.open(Paths.get(modelPath), MoeCacheRanking.maxExperts(initial),
                cfg.moeCache().ranking());
        this.eplb = EplbSession.open(cfg.eplb(), initial);
        if (eplb != null) {
            LOGGER.info("{}: expert load balancer {}", modelName, eplb);
        }

        List<LoraAttachment.Binding> loraBindings = GpuAdapters.readLoras(initial, cfg, this::cpuWeights);
        if (cfg.split()) {
            GpuSplitPlanner.refuseOffloadWithMultipleDevices(initial, cfg, null);
            this.split = GpuSplitSession.open(SplitRequest.builder(source, catalog, transforms, kv, cfg, initial,
                            this::cpuWeights)
                    .modelName(modelName)
                    .ranking(ranking)
                    .loras(loraBindings)
                    .controlVectors(GpuAdapters.readControlVectors(modelName, initial, cfg))
                    .drafter((b, q, k) -> LLMModelFactory.draftCompanion(cfg, initial, false,
                            (draft, placed) -> GpuWeightUploader.plannedBytesOf(b, q, k, draft, placed)))
                    .build());
        }
        if (split != null) {
            finishSplitLoad(cfg);
            return;
        }

        GpuModelLoad.load(this, sharedBackend, cfg, kv, initial, loraBindings);
    }

    /**
     * Whether a load places the model's NextN / MTP blocks: MTP drafting is configured ({@link EngineConfig#mtp()},
     * llama.cpp's {@code DRAFT_MTP} speculative type that sets {@code load_mtp}, common/common.cpp:1710) or the
     * context fit holds the NextN draft context ({@link MemoryConfig#nextnMemory()}).
     */
    static boolean loadsNextn(EngineConfig cfg, MemoryConfig mem) {
        return cfg.mtp() || mem.nextnMemory();
    }

    /** The unified settings of the MoE expert cache ({@link HostPlacement#moeCacheSettings}). */
    static MoeCacheConfig resolveMoeCache(EngineConfig cfg) {
        return HostPlacement.moeCacheSettings(cfg);
    }

    /** Wires the model's fields to the split session the load built ({@link #split}). */
    private void finishSplitLoad(EngineConfig cfg) {
        this.spec = split.spec();
        this.forwardSpec = split.forwardSpec();
        this.plan = split.plan();
        this.config = split.config();
        this.backend = split.backend();
        this.queue = backend.computeQueue();
        this.forward = split.forwardBackend();
        if (spec.arch() != Arch.WAVTOKENIZER_DEC) {
            long[] reserve = split.slotReserve();
            this.loraRegistry = GpuAdapters.perRequest(split.loras(), split.stageKernels(), cfg, this::cpuWeights,
                    reserve, split.slotUbatch());
        }
        LOGGER.info("{} ({}): {}=layer over {} device(s), main device {}, context {}, ubatch {}", modelName,
                spec.arch().ggufName(), EngineConfig.PROP_SPLIT_MODE, GpuSplitPlanner.splitDeviceCount(cfg),
                split.mainDeviceName(), plan.nCtx(), plan.ubatch());
    }

    /** The refusal of an operation a split model cannot serve: {@code neuralfree.split.mode=layer} and why. */
    private static UnsupportedOperationException splitRefusal(String reason) {
        return new UnsupportedOperationException(EngineConfig.PROP_SPLIT_MODE + "="
                + EngineConfig.SplitMode.LAYER.llamaName() + " " + reason);
    }

    /**
     * Device bytes left for this model: the free VRAM measured when the model began loading (before its buffers, as
     * {@link VRAMBudget#queryFreeVram} requires) less everything the backend has allocated since and the allocations
     * still to come: the adapter uploads ({@link #pendingDeviceBytes}) and, once the context is chosen, the drafter
     * memory the fit held beside it ({@link #drafterDeviceBytes}), capped by {@code neuralfree.gpu.vram.limit}
     * ({@link VRAMBudget#limitedBytes}).
     */
    long usableVram() {
        long allocatedSince = Math.max(0L, backend.getAllocatedBytes() - allocatedAtLoad);
        long pending = Math.addExact(pendingDeviceBytes, drafterDeviceBytes);
        long budget = VRAMBudget.memoryBudgetBytes(backend.capabilities().getTotalVramBytes(), freeAtLoad, 0L,
                allocatedSince, pending);
        return VRAMBudget.limitedBytes(budget, vramLimitBytes, allocatedSince, pending);
    }

    /**
     * Device bytes of a drafter file's weights on this model's device: the tensors {@code placed} its drafter places
     * there for the route it drafts with (never an MTP draft file's trunk), planned with this model's kernel
     * type policy as the drafter uploads them (a draft model's own load, or the eagle3 / gemma4-assistant / MTP draft
     * file weights {@code LLMModelFactory} uploads; {@link GpuWeightUploader#plannedBytesOf}).
     */
    long draftDeviceWeightBytes(ModelSource draft, Predicate<GGUFTensorInfo> placed) throws IOException {
        return GpuWeightUploader.plannedBytesOf(backend, queue, kernels, draft, placed);
    }

    private void releaseAfterFailedLoad(Throwable failure) {
        try {
            releaseAll();
        } catch (Throwable t) {
            if (t != failure) {
                failure.addSuppressed(t);
            }
        }
    }

    // ============================================================================================
    // Engine API
    // ============================================================================================

    @Override
    public ModelSpec spec() {
        return spec;
    }

    /** The sampling defaults of the model's metadata ({@link SamplingDefaults#read}). */
    @Override
    public SamplingDefaults modelSamplingDefaults() {
        return samplingDefaults;
    }

    /** The registry of the per-request adapters; null for wavtokenizer-dec. */
    @Override
    public LoraRegistry loraRegistry() {
        return loraRegistry;
    }

    /**
     * The backend of the forward passes: the decorated GPU backend (switched-LoRA routing, qwen3tts logits layout).
     *
     * @throws UnsupportedOperationException for the WavTokenizer decoder, which has no token forward pass
     */
    @Override
    public ForwardBackend forwardBackend() {
        ensureOpen();
        if (forward == null) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " decodes audio codes"
                    + " (decodeAudioCodes) and has no token forward pass");
        }
        return forward;
    }

    /**
     * The GPU context of the model, for drafters that run on the target's device and kernels (eagle3, an MTP draft
     * file); for a split model the context of the main device, which holds the embedding and the output head.
     */
    public GpuForwardContext forwardContext() {
        ensureOpen();
        return split != null ? split.forwardContext() : ctx;
    }

    /** The session file ({@code EngineModel.saveSession}) over the device memory: {@code SessionFile.save}. */
    @Override
    public void saveSession(java.nio.file.Path out) throws IOException {
        if (parallelGroup != null && attentionPlan != null) {
            parallelGroup.saveSession(sessionMemory("save"), out);
            return;
        }
        sessionMemory("save").saveSession(out);
    }

    /** The session file ({@code EngineModel.loadSession}) over the device memory: {@code SessionFile.load}. */
    @Override
    public void loadSession(java.nio.file.Path in) throws IOException {
        if (parallelGroup != null && attentionPlan != null) {
            parallelGroup.loadSession(sessionMemory("load"), in);
            return;
        }
        sessionMemory("load").loadSession(in);
    }

    /** The memory a session action moves, refused by name for a model without a token forward pass. */
    private com.eonmux.neuralfree.gguf.memory.ModelMemory sessionMemory(String action) {
        com.eonmux.neuralfree.gguf.memory.ModelMemory memory = forwardBackend().memory();
        if (memory == null) {
            throw new UnsupportedOperationException("session " + action + " refused: " + architecture()
                    + " holds no KV cache memory");
        }
        return memory;
    }

    /**
     * The GPU context that holds the model's own NextN / MTP blocks, for an MTP drafter over them: the model's
     * context, or for a split model the context of the device that runs the last trunk layer.
     */
    public GpuForwardContext nextnContext() {
        ensureOpen();
        return split != null ? split.nextnContext() : ctx;
    }

    /** True when the model runs over several GPUs ({@code neuralfree.split.mode=layer}). */
    public boolean isSplit() {
        return split != null;
    }

    /** True when the load keeps its device layers' KV caches in host RAM ({@code neuralfree.kv.offload=false}). */
    public boolean kvOnHost() {
        ensureOpen();
        return hybridSplit != null && hybridSplit.hostKv();
    }

    /**
     * The arguments that reproduce this load's placement ({@code -c}, {@code -ngl}, the expert overflow as
     * {@code --cpu-moe} or {@code -ot}, and {@code -nkvo}), as {@link FitParams#arguments()} prints them: the
     * context and placement this load fitted, reusable as llama.cpp's {@code llama_params_fit} output is. Null
     * when the load was not placed by the single-device fit: a multi-device layer split
     * ({@code neuralfree.split.mode=layer}) spreads its layers over the devices by shares that only
     * {@code --tensor-split} repeats, so its placement has no {@code -ngl}-only argument line.
     */
    public String fittedArguments() {
        ensureOpen();
        return placement == null || split != null ? null : FitParams.argumentsOf(placement);
    }

    /** Devices the forward pass runs on: the stages of a split, else one ({@code GpuBenchmarks}). */
    public int deviceCount() {
        return split != null ? split.deviceCount() : 1;
    }

    /** The execution-observer slot of device {@code d} the benchmark profilers are swapped into. */
    ObserverSlot observerSlot(int d) {
        return split != null ? split.observer(d) : observer;
    }

    /**
     * Makes {@code observer} the base observer of device {@code d}'s forward stages, replacing the diagnostics
     * installed from the system properties; {@link ExecutionObserver#NONE} removes it. The stage trace of
     * {@code nftool debug} and {@code nftool eval-callback} installs one per device.
     */
    public void setExecutionObserver(int d, ExecutionObserver observer) {
        ensureOpen();
        observerSlot(d).setBase(Objects.requireNonNull(observer, "observer"));
    }

    /** The base observer of device {@code d}'s forward stages ({@link ExecutionObserver#NONE} when none is set). */
    public ExecutionObserver executionObserver(int d) {
        ensureOpen();
        return observerSlot(d).base();
    }

    /** The compute queue every kernel and transfer of device {@code d} runs on. */
    public ComputeQueue computeQueue(int d) {
        return split != null ? split.queue(d) : queue;
    }

    /** The name of device {@code d}, as the diagnostics and the profiler report it. */
    public String deviceName(int d) {
        return split != null ? split.deviceName(d) : deviceName();
    }

    /** The streaming source of device {@code d}, or null when its weights are resident. */
    StreamingLayerWeightSource streamingOf(int d) {
        return split != null ? split.streaming(d) : streaming;
    }

    /** The host-mapped tier of the load (the tensors {@code neuralfree.cpu.offload.mb} took), or null. */
    HostMappedWeightSource hostMappedWeights() {
        return hostMapped;
    }

    /** The MoE expert cache of device {@code d}, or null when its layers keep their experts on the device. */
    MoeExpertCache moeExpertCacheOf(int d) {
        return split != null ? split.moeExpertCache(d) : moeCache;
    }

    /** The streamed sets of device {@code d}, empty when its weights are resident. */
    List<LayerWeightSet> streamedSetsOf(int d) {
        return split != null ? split.streamedSets(d) : streamedSets;
    }

    /** The first trunk layers that run on the host, {@code [0, n)}: of the split, or of the hybrid one-device load. */
    int hostLayerCount() {
        if (split != null) {
            return split.hostLayers();
        }
        return hybridSplit != null ? hybridSplit.firstGpuLayer() : 0;
    }

    /** The first trunk layer device {@code d} runs. */
    int firstLayerOf(int d) {
        return split != null ? split.firstLayer(d) : hostLayerCount();
    }

    /** The trunk layers device {@code d} runs. */
    int layerCountOf(int d) {
        return split != null ? split.layerCount(d) : spec.nLayer() - hostLayerCount();
    }

    /** The backend type and device name of the one-device load. */
    String deviceName() {
        return backend.backendType().toString().toLowerCase(Locale.ROOT) + " '"
                + backend.capabilities().getDeviceName() + "'";
    }

    /** The block registry the model was bound with; a split model binds every device with the same one. */
    public GpuBlockRegistry blockRegistry() {
        ensureOpen();
        return split != null ? split.blockRegistry() : registry;
    }

    /**
     * The resident device tensors: every tensor placed on the device ({@link WeightPlacement}) except the streamed
     * trunk layers. Model-level tensors and encoder layers are always resident; the NextN / MTP blocks are resident
     * once placed (at load with MTP configured, or by {@link #ensureNextnWeights()}); the tensors only the host heads
     * read never are.
     *
     * @throws UnsupportedOperationException for a split model, whose device tensors live on several backends
     */
    public GpuWeights gpuWeights() {
        ensureOpen();
        if (split != null) {
            throw splitRefusal("spreads the device tensors over several backends");
        }
        return uploader;
    }

    /**
     * Live weight updates over the resident device tensors (vLLM's in-process weight transfer engine); see
     * {@link GpuWeightUpdater}.
     *
     * @throws UnsupportedOperationException for a split or tensor-parallel model, whose tensors live on several
     *                                       backends or per-rank shards an update cannot address
     */
    @Override
    public WeightUpdater weightUpdater() {
        ensureOpen();
        if (split != null) {
            throw splitRefusal("spreads the device tensors over several backends, so a live weight update cannot"
                    + " address them");
        }
        if (parallelGroup != null) {
            throw new UnsupportedOperationException(modelName + " runs in " + EngineConfig.PROP_SPLIT_MODE
                    + " mode over several devices; its weights are per-rank shards a live update cannot address");
        }
        return new GpuWeightUpdater(uploader, modelName);
    }

    /**
     * Places the NextN / MTP blocks on the device for an MTP drafter over this model's own blocks; a no-op when they
     * are resident or the model has none. A model opened without MTP ({@link #loadsNextn}) left them and their draft
     * caches out of its context fit, so the blocks are uploaded now, when the VRAM left after loading holds them and
     * the caches the drafter then allocates ({@link MemoryPlanner#planNextn(ModelSpec, MemoryPlan, int)}), and their
     * weight kernels prepared.
     *
     * A split model places them on the device of the last trunk layer ({@link GpuSplitSession#ensureNextnWeights}).
     *
     * @throws GpuOutOfMemoryException when they do not fit the usable VRAM; the model then has to be opened with MTP
     *                                 configured, so the context is fitted beside them
     */
    public void ensureNextnWeights() {
        ensureOpen();
        if (split != null) {
            split.ensureNextnWeights();
            return;
        }
        WeightPlacement current = weightPlacement;
        if (!current.hasNextn() || current.nextnOnDevice()) {
            return;
        }
        long bytes = uploader.plannedBytes(t -> current.isNextn(t) && uploader.find(t.name()) == null);
        long caches = plan.kind() == MemoryKind.NONE || spec.nLayer() == 0 ? 0L
                : MemoryPlanner.nextnBytes(spec, config.memory(), plan.nCtx(), plan.ubatch());
        long needed = Math.addExact(bytes, caches);
        long usable = usableVram();
        if (needed > usable) {
            throw new GpuOutOfMemoryException(spec.arch().ggufName() + ": the " + spec.nextnLayers().size()
                    + " NextN block(s) were left off the device because MTP was not configured when the model was"
                    + " opened, and their " + mib(bytes) + " MiB of weights and " + mib(caches) + " MiB of draft"
                    + " caches do not fit the " + mib(Math.max(0L, usable)) + " MiB of usable VRAM left; open the"
                    + " model with " + EngineConfig.PROP_MTP + "=true so its context is fitted beside them", needed,
                    usable, backend.backendType().name());
        }
        uploader.uploadResident(t -> !current.isNextn(t));
        WeightKernelPrewarm.prewarm(uploader.plans(current::isNextn), ops, kernels);
        this.weightPlacement = current.withNextnOnDevice();
        LOGGER.info("{}: {} NextN block(s) placed on {} for MTP drafting ({} MiB)", modelName,
                spec.nextnLayers().size(), backend.backendType(), mib(bytes));
    }

    /**
     * Host access to the model's tensors (heads, drafters, adapters), opened on first use and released by
     * {@link #close()}.
     */
    public CpuWeights cpuWeights() throws IOException {
        ensureOpen();
        if (cpuWeights == null) {
            cpuWeights = CpuWeights.open(source, catalog, transforms);
        }
        return cpuWeights;
    }

    /**
     * The MoE expert cache of the model, or null when none is configured; a split model holds one per device
     * ({@link GpuSplitSession#moeExpertCache}).
     */
    public MoeExpertCache moeExpertCache() {
        return moeCache;
    }

    /** The prefill expert prefetch pool of the model, or null when it is disabled. */
    public MoeExpertPrefetch moeExpertPrefetch() {
        return moePrefetch;
    }

    /** The effective engine configuration (its memory settings carry the planned ubatch). */
    public EngineConfig engineConfig() {
        return config;
    }

    /** The memory plan of the session. */
    public MemoryPlan memoryPlan() {
        return plan;
    }

    /** The backend the model runs on (owned by the caller when it was passed in). */
    public ComputeBackend backend() {
        return backend;
    }

    /** True while the model is open; every forward pass runs on the GPU backend. */
    public boolean isGpuEnabled() {
        return !closed;
    }

    // ============================================================================================
    // Generation
    // ============================================================================================

    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP) throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP, 1f, 0, 0f, 0, null);
    }

    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP, TokenCallback callback)
            throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP, 1f, 0, 0f, 0, callback);
    }

    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP, float repetitionPenalty,
                          int repeatWindow, TokenCallback callback) throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP, repetitionPenalty, repeatWindow, 0f, 0, callback);
    }

    /**
     * Generates up to {@code maxNewTokens} tokens after {@code tokens}, stopping at an end-of-generation token, with
     * the sampler {@link EngineConfig#legacySampler} maps these parameters to over the configured sampler defaults
     * (the same chain the CPU engine builds). {@code temperature <= 0} decodes greedily; {@code topP == 1},
     * {@code minP == 0}, {@code topK <= 0} and {@code repetitionPenalty == 1} disable their samplers;
     * {@code repeatWindow} 0 or -1 penalizes the whole context.
     *
     * @return the prompt followed by the generated tokens; for an encoder-decoder model only the generated tokens
     * @throws IllegalArgumentException for empty tokens, a negative token budget or out-of-range sampling values
     */
    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP, float repetitionPenalty,
                          int repeatWindow, float minP, int topK, TokenCallback callback) throws IOException {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("Input tokens must not be empty");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be >= 0, got " + maxNewTokens);
        }
        if (maxNewTokens == 0) {
            return isEncoderDecoder() ? new int[0] : tokens.clone();
        }
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(config.legacySampler(temperature, topP, repetitionPenalty, repeatWindow, minP, topK))
                .build();
        GenerationResult result = generate(tokens, request, callback);
        return isEncoderDecoder() ? result.tokens() : result.withPrompt(tokens);
    }

    /**
     * Generates through the {@link GenerationLoop} of the model; the reasoning filter of the chat template routes
     * content to {@code callback}, or every channel after {@code setShowAllChannels(true)}.
     */
    @Override
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        GenerationLoop loop = generationLoop();
        if (!showAllChannels) {
            return loop.generate(prompt, request, callback);
        }
        ReasoningFilter.Sink sink = (channel, tokenId, text) -> {
            if (callback != null) {
                callback.onToken(tokenId, text);
            }
        };
        return loop.generate(prompt, request, sink, null);
    }

    @Override
    public GenerationResult generateEmbeds(float[][] promptEmbeds, GenerationRequest request, TokenCallback callback) {
        return generationLoop().generate(EmbedsPrompt.ofRows(promptEmbeds), request, callback);
    }

    @Override
    public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP)
            throws IOException {
        return generate(tokenizer.encode(prompt), maxNewTokens, temperature, topP);
    }

    @Override
    public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                    TokenCallback callback) throws IOException {
        return generate(tokenizer.encode(prompt), maxNewTokens, temperature, topP, callback);
    }

    @Override
    public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                    float repetitionPenalty, int repeatWindow, TokenCallback callback)
            throws IOException {
        return generate(tokenizer.encode(prompt), maxNewTokens, temperature, topP, repetitionPenalty, repeatWindow,
                callback);
    }

    /** When true, generation streams every reasoning-filter channel (reasoning and tool calls), not only content. */
    public void setShowAllChannels(boolean show) {
        this.showAllChannels = show;
    }

    /**
     * Masked-diffusion generation with {@link DiffusionGenerator}; the canvas defaults to the planned ubatch.
     *
     * @throws UnsupportedOperationException when the model is not a masked-diffusion model
     */
    @Override
    public int[] generateDiffusion(int[] prompt, DiffusionConfig diffusionConfig, TokenCallback callback) {
        ensureOpen();
        if (diffusion == null) {
            diffusion = DiffusionGenerator.forModel(this, config);
        }
        return diffusion.generate(prompt, diffusionConfig, callback);
    }

    private GenerationLoop generationLoop() {
        ensureOpen();
        if (generationLoop == null) {
            generationLoop = GenerationLoop.forModel(this, config);
        }
        return generationLoop;
    }

    @Override
    public void resetKVCache() {
        ensureOpen();
        if (forward != null) {
            forward.reset();
        }
    }

    // ============================================================================================
    // Embeddings, classification, reranking, audio
    // ============================================================================================

    @Override
    public float[] encode(int[] tokens) throws IOException {
        return tasks.encode(tokens, null);
    }

    @Override
    public float[] encode(int[] tokens, int[] tokenTypes) throws IOException {
        return tasks.encode(tokens, tokenTypes);
    }

    @Override
    public float[] encodeMedia(MediaPrompt prompt) throws IOException {
        return tasks.encodeMedia(prompt);
    }

    @Override
    public float[][] encodeTokens(int[] tokens) throws IOException {
        return tasks.encodeTokens(tokens);
    }

    @Override
    public float[][] encodeBatch(int[][] sequences) throws IOException {
        return tasks.encodeBatch(sequences);
    }

    @Override
    public float[] classify(int[] tokens) throws IOException {
        return classify(tokens, null);
    }

    /** Classifier output of one input: decision-head scores, or the sequence classifier scores. */
    @Override
    public float[] classify(int[] tokens, int[] decisionOrder) throws IOException {
        return tasks.classify(tokens, decisionOrder);
    }

    @Override
    public float[][] tokenClassify(int[] tokens) throws IOException {
        return tasks.tokenClassify(tokens);
    }

    @Override
    public String[] classLabels() {
        return tasks.classLabels();
    }

    @Override
    public ClassifierOutput classifyOutput(int[] tokens) throws IOException {
        return tasks.classifyOutput(tokens, true);
    }

    @Override
    public ClassifierOutput classifyOutput(int[] tokens, boolean activation) throws IOException {
        return tasks.classifyOutput(tokens, activation);
    }

    @Override
    public ClassifierOutput classifyOutputMedia(MediaPrompt prompt) throws IOException {
        return tasks.classifyOutputMedia(prompt, true);
    }

    @Override
    public ClassifierOutput classifyOutputMedia(MediaPrompt prompt, boolean activation) throws IOException {
        return tasks.classifyOutputMedia(prompt, activation);
    }

    @Override
    public float[] nextLogits(int[] tokens) throws IOException {
        return tasks.nextLogits(tokens);
    }

    @Override
    public float[] nextLogits(MediaPrompt prompt) throws IOException {
        return tasks.nextLogits(prompt);
    }

    @Override
    public float[] classifyMedia(MediaPrompt prompt, int[] order) throws IOException {
        return tasks.classifyMedia(prompt, order);
    }

    @Override
    public float rerank(int[] query, int[] document) throws IOException {
        return tasks.rerank(query, document);
    }

    @Override
    public float rerank(String query, String document) throws IOException {
        return tasks.rerank(query, document);
    }

    @Override
    public float[][] decodeAudioCodes(int[] codes) {
        return tasks.decodeAudioCodes(codes);
    }

    @Override
    public SpeechFormat speechFormat() {
        return tasks.speechFormat();
    }

    @Override
    public SpeechFormat requireSpeechFormat() {
        return tasks.requireSpeechFormat();
    }

    @Override
    public AudioClip synthesizeSpeech(int[] codes) {
        return tasks.synthesizeSpeech(codes);
    }

    @Override
    public void synthesizeSpeech(int[] codes, Consumer<AudioClip> sink) {
        tasks.synthesizeSpeech(codes, sink);
    }

    @Override
    public PoolingOperator poolingOperator() {
        return tasks.poolingOperator();
    }

    // ============================================================================================
    // Model properties (from the spec; per-layer values of layer 0)
    // ============================================================================================

    @Override
    public GGUFTokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public String architecture() {
        return spec.arch().ggufName();
    }

    @Override
    public String name() {
        return modelName;
    }

    @Override
    public int vocabSize() {
        return spec.hparams().nVocab();
    }

    @Override
    public int contextLength() {
        return spec.hparams().nCtxTrain();
    }

    /** The planned context of the memory, or the most tokens one pass takes for models without memory. */
    @Override
    public int kvCacheContext() {
        return plan.kind() == MemoryKind.NONE ? plan.ubatch() : plan.nCtx();
    }

    /**
     * Width of an embedding output row: the {@code n_embd_out} of the head projection or the embedding output when
     * the model has one ({@link com.eonmux.neuralfree.gguf.spec.HeadSpec#embeddingWidth}), else {@code n_embd}, as
     * llama.cpp's {@code llama_model_n_embd_out}.
     */
    @Override
    public int embeddingLength() {
        int nEmbd = spec.hparams().nEmbd();
        return spec.head() != null ? spec.head().embeddingWidth(nEmbd) : nEmbd;
    }

    @Override
    public int layerCount() {
        return spec.nLayer();
    }

    @Override
    public int headCount() {
        return layer0(spec.hparams()::nHead);
    }

    @Override
    public int headCountKv() {
        return layer0(spec.hparams()::nHeadKv);
    }

    @Override
    public int feedForwardLength() {
        return layer0(spec.hparams()::nFf);
    }

    /** Epsilon of the block norm of layer 0 (RMSNorm or LayerNorm, as the architecture uses). */
    @Override
    public float rmsNormEps() {
        List<LayerSpec> first = !spec.layers().isEmpty() ? spec.layers() : spec.encoderLayers();
        return first.isEmpty() ? spec.hparams().normRmsEps() : first.get(0).norm().eps();
    }

    @Override
    public float ropeFreqBase() {
        return spec.hparams().nLayer() > 0 ? spec.hparams().ropeFreqBase(0) : spec.hparams().ropeFreqBase();
    }

    @Override
    public int headDim() {
        return layer0(spec.hparams()::headK);
    }

    @Override
    public int kvDim() {
        return layer0(spec.hparams()::nEmbdKGqa);
    }

    private int layer0(IntUnaryOperator perLayer) {
        return spec.hparams().nLayer() > 0 ? perLayer.applyAsInt(0) : 0;
    }

    @Override
    public boolean isEncoder() {
        return EncodeGate.isEncoder(spec);
    }

    @Override
    public boolean isEncoderDecoder() {
        return spec.supportClass() == SupportClass.ENC_DEC;
    }

    @Override
    public PoolingType poolingType() {
        return spec.head().pooling().toPoolingType();
    }

    @Override
    public void printInfo() {
        System.out.println("=== GGUF Model Info (GPU) ===");
        System.out.println("Architecture: " + architecture() + " (" + spec.supportClass() + ")");
        System.out.println("Name: " + modelName);
        System.out.println("Vocab size: " + vocabSize());
        System.out.println("Context length: " + contextLength() + " (planned: " + plan.nCtx() + ", ubatch "
                + plan.ubatch() + ")");
        System.out.println("Embedding length: " + embeddingLength());
        System.out.println("Layers: " + layerCount() + (spec.nextnLayers().isEmpty() ? ""
                : " (+" + spec.nextnLayers().size() + " NextN)") + (spec.encoderLayers().isEmpty() ? ""
                : " (+" + spec.encoderLayers().size() + " encoder)"));
        System.out.println("Heads: " + headCount() + ", KV heads: " + headCountKv() + ", head dim: " + headDim());
        System.out.println("FFN length: " + feedForwardLength());
        System.out.println("Norm eps: " + rmsNormEps());
        System.out.println("RoPE freq base: " + ropeFreqBase());
        System.out.println("Memory: " + plan.kind() + ", " + mib(plan.totalBytes())
                + " MiB");
        System.out.println("Backend: " + backend.backendType() + " on " + backend.capabilities().getDeviceName());
        if (split != null) {
            System.out.println("Split: " + EngineConfig.PROP_SPLIT_MODE + "=layer over " + split.deviceCount()
                    + " devices, main " + split.mainDeviceName());
            for (int d = 0; d < split.deviceCount(); d++) {
                System.out.println("  Stage " + d + ": " + split.describe(d));
            }
            if (split.hostLayers() > 0) {
                System.out.println("Host layers: 0.." + (split.hostLayers() - 1) + " run on the CPU");
            }
            if (split.loras() != null) {
                System.out.println("LoRA deltas: " + split.loras().deltaCount() + " across the stages");
            }
            System.out.println("Tensors: " + source.getTensorCount());
            return;
        }
        System.out.println("Resident weights: " + mib(uploader.residentBytes())
                + " MiB in " + uploader.residentCount() + " tensors");
        if (streaming != null) {
            System.out.println("Streamed layers: " + streaming.streamedLayers() + " through " + streaming.depth()
                    + " slots (" + mib(streaming.deviceBytes()) + " MiB, "
                    + (streaming.asyncTransfers() ? "async transfers" : "transfers on the compute queue") + ")");
        }
        if (moeCache != null) {
            System.out.println("MoE expert cache: " + moeCache);
        }
        if (moePrefetch != null) {
            System.out.println("MoE expert prefetch: " + moePrefetch);
        }
        System.out.println("Tensors: " + source.getTensorCount());
    }

    // ============================================================================================
    // Benchmarks
    // ============================================================================================

    /**
     * Decodes {@code tokenId} at {@code position} under a {@link GpuProfiler} and prints the per-stage times; the
     * memory is reset and filled with {@code position} copies of the token first
     * ({@code GpuBenchmarks.profileDecodeToken}).
     *
     * @param quiet true to time without printing (warm-up)
     * @throws UnsupportedOperationException when the model has no decoder memory
     */
    public void profileDecodeToken(int tokenId, int position, boolean quiet) {
        benchmarks().profileDecodeToken(tokenId, position, quiet);
    }

    /**
     * Profiles the decode at several positions, then times unobserved decode throughput. Call it before
     * {@code generate}; it leaves the memory empty ({@code GpuBenchmarks.runBenchmark}).
     */
    public void runBenchmark(int tokenId) {
        benchmarks().runBenchmark(tokenId);
    }

    /**
     * Compares the per-token weight transfer of a streamed model with its streamed decode
     * ({@code GpuBenchmarks.benchmarkStreamingOverlap}).
     */
    public void benchmarkStreamingOverlap(int tokenId) {
        benchmarks().benchmarkStreamingOverlap(tokenId);
    }

    private GpuBenchmarks benchmarks() {
        ensureOpen();
        return new GpuBenchmarks(this);
    }

    /** {@code GpuBenchmarks} access: the decoder forward pass, refused for a model without decode memory. */
    ForwardBackend requireDecoder(String what) {
        ensureOpen();
        if (!hasDecoderMemory()) {
            throw new UnsupportedOperationException(what + " needs a decoder with memory; " + architecture()
                    + " (" + spec.supportClass() + ") has none");
        }
        return forward;
    }

    /** True when the model has a decoder forward pass with memory (a decode step to benchmark). */
    boolean hasDecoderMemory() {
        if (split != null) {
            return forward != null && plan.kind() != MemoryKind.NONE && !isEncoderDecoder();
        }
        return forward != null && ctx.hasMemory() && !isEncoderDecoder();
    }

    // ============================================================================================
    // Lifecycle
    // ============================================================================================

    /**
     * Opens the quantized target view of two-level verification ({@code neuralfree.draft.tlv}) over this model's
     * memory. When this model streams its projection matrices as W12 rows ({@code neuralfree.weights.w12}), the
     * view is the nested 8-bit Q view: the Q round runs this model's own backend inside a Q-view scope, which
     * reads the nested 8-bit view of the same code bytes — no copy, no second weight upload or backend, no
     * added memory. Otherwise a {@link QCopySource} restates this model's projection matrices at the copy's
     * bits, a second weight upload and forward context run them on this model's device beside its own passes
     * through the same block registry, and the returned view owns the store, the upload and the backend it
     * built. Both shapes read and write this model's KV cache, so a Q round's rows sit exactly where the exact
     * round's would.
     *
     * @throws UnsupportedOperationException naming {@code neuralfree.draft.tlv} for the load shapes neither view
     *                                       can shadow (a tensor or pipeline split, streamed or host-mapped
     *                                       weights, a MoE expert cache or hosted layers, expert-parallel
     *       rebalancing, LoRA adapters or control vectors — the view's passes would run a different graph than
     *       the target's), naming {@code neuralfree.draft.tlv.bits} on a W12 target for any width but 8 (the
     *       nested view is fixed 8-bit), and for the non-W12 targets {@link QCopySource} refuses (routed
     *       experts, no matrix the copy undercuts)
     */
    @Override
    public QuantizedView openQuantizedView(int bits) throws IOException {
        ensureOpen();
        return GpuQuantizedViews.open(this, bits);
    }

    /**
     * The nested 8-bit Q view of a W12 target: the Q round runs this model's own backend inside a
     * {@link GpuOps#openW12QView} scope, which routes every projection of a W12 weight through the nested 8-bit
     * view of its code bytes — no copy is built, no second weight upload or backend exists, and the view adds
     * no memory. The scope opens per Q round and closes before the exact round reads the same weights. The
     * nested view is fixed 8-bit, so the copy's 6-bit width does not apply.
     *
     * @throws UnsupportedOperationException naming {@code neuralfree.draft.tlv.bits} for any width but 8
     */

    /**
     * Releases everything through its owner, in reverse order of acquisition: the audio decoder and the forward
     * backend (blocks and heads), the streaming slots, the memory, the dispatch helpers, scratch and placeholders,
     * the resident weights, then returns a shared CUDA backend's pooled device memory to the driver
     * ({@link #trimSharedCudaPool}), then the host weights, the mapped file and the source, and last the backend when
     * the model created it. Idempotent.
     */
    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        releaseAll();
    }

    private void releaseAll() throws IOException {
        Throwable first = queue != null ? closeStep(queue::finish, null) : null;
        first = closeStep(tasks, first);
        first = closeStep(loras, first);
        first = closeStep(split, first);
        // the forward backend owns the bound FFN blocks, which read the row split's rank contexts and shard buffers
        first = closeStep(rawBackend, first);
        first = closeStep(parallelGroup, first);
        parallelGroup = null;
        rowSplitPlan = null;
        attentionPlan = null;
        vocabPlan = null;
        moePlan = null;
        first = closeStep(hostMapped, first);
        first = closeStep(streaming, first);
        first = closeStep(streamingStore, first);
        first = closeStep(moePrefetch, first);
        first = closeStep(moeCache, first);
        first = closeStep(hybridMemory != null ? hybridMemory : memory, first);
        first = closeStep(ops, first);
        first = closeStep(scratch, first);
        first = closeStep(placeholders, first);
        first = closeStep(uploader, first);
        first = closeStep(eplb, first);
        eplb = null;
        eplbRebalancer = null;
        first = closeStep(ranking, first);
        first = closeStep(this::trimSharedCudaPool, first);
        first = closeStep(cpuWeights, first);
        first = closeStep(store, first);
        first = closeStep(source::close, first);
        if (ownedAcceleration != null) {
            first = closeStep(ownedAcceleration::shutdown, first);
        }
        if (first instanceof IOException) {
            throw (IOException) first;
        }
        if (first instanceof RuntimeException) {
            throw (RuntimeException) first;
        }
        if (first instanceof Error) {
            throw (Error) first;
        }
        if (first != null) {
            throw new IOException("closing " + modelName + " failed", first);
        }
    }

    /**
     * Returns the blocks pooled by a shared CUDA backend, which hold the device memory of this model's released
     * buffers, to the driver ({@link CudaComputeBackend#trimPool}), so other processes and APIs on the device see it
     * free once the model is closed. Trimming frees only pooled blocks (including those other users of the backend
     * released), so buffers they still hold are unaffected. Does nothing for another backend type, for a backend this
     * model created (closing it frees its pool) and for a shared backend that is already closed.
     */
    private void trimSharedCudaPool() {
        if (ownedAcceleration != null || !(backend instanceof CudaComputeBackend)) {
            return;
        }
        CudaComputeBackend cuda = (CudaComputeBackend) backend;
        if (cuda.isClosed()) {
            return;
        }
        long freed = cuda.trimPool();
        LOGGER.debug("{}: returned {} MiB of pooled CUDA memory to the driver on close", modelName, mib(freed));
    }

    private static Throwable closeStep(AutoCloseable c, Throwable first) {
        if (c == null) {
            return first;
        }
        try {
            c.close();
            return first;
        } catch (Throwable t) {
            if (first == null) {
                return t;
            }
            if (t != first) {
                first.addSuppressed(t);
            }
            return first;
        }
    }

    void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("model " + modelName + " is closed");
        }
    }

    /** Mebibytes with one decimal. */
    static String mib(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }

    @Override
    public String toString() {
        return "GGUFModelGPU{" + modelName + ", " + (spec != null ? spec.arch().ggufName() : "?") + ", "
                + (backend != null ? backend.backendType() : "no backend") + '}';
    }
}
