package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.DiffusionGenerator;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuDiagnostics;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuProfiler;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.StreamingLayerWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.audio.GpuWavTokenizer;
import com.eonmux.neuralfree.gguf.ops.audio.TtsBackbone;
import com.eonmux.neuralfree.gguf.ops.encdec.GpuEncDecBackend;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionBody;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.ops.heads.GpuLoraDeltas;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.PoolingHead;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraBackend;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.quant.PrismHadamardLoader;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.function.IntUnaryOperator;
import java.util.function.Predicate;

/**
 * A GGUF model on a GPU backend (OpenCL, CUDA or Vulkan), assembled from the spec-driven engine:
 * <ol>
 *   <li>the file ({@link GGUFFileParser}), its metadata ({@link GgufKv}) and the resolved {@link ModelSpec};</li>
 *   <li>the LLM kernels and the weight upload plan ({@link GpuWeightUploader}, PRISM transforms included);</li>
 *   <li>the memory plan ({@link MemoryPlanner} with {@link ContextFitter} against the free VRAM), the device memory
 *       ({@link DeviceModelMemory}) and the weight placement: every weight resident, or the trunk layers streamed
 *       through a {@link StreamingLayerWeightSource} when they do not fit;</li>
 *   <li>the {@link GpuForwardContext}, the block registry of every family ({@link GpuFactoryCatalog}) and the
 *       forward backend ({@link GpuForwardBackend}, or {@link GpuEncDecBackend} for encoder-decoder models).</li>
 * </ol>
 * Generation runs through {@link GenerationLoop}, masked diffusion through {@link DiffusionGenerator}, embeddings,
 * classification and reranking through the host heads over {@link ForwardBackend#hiddenRows}, and audio codes through
 * {@link GpuWavTokenizer}. Not thread-safe.
 *
 * <p>Weight placement: with an automatic context, every weight stays resident when the largest context that then fits
 * the remaining VRAM is at least {@value #AUTO_CONTEXT_FLOOR} tokens (or the training context when shorter);
 * otherwise the trunk layers are streamed and the automatic context is capped at that floor so the rest of the VRAM
 * holds streaming slots. A user context ({@code neuralfree.ctx}) is honoured as given. The context is fitted beside the
 * memory allocated after loading: the NextN / MTP draft caches ({@link MemoryConfig#nextnMemory()}) and a configured
 * draft model ({@link LLMModelFactory#draftCompanion}); that memory stays out of the budget of the streaming slots.
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

    /** Smallest automatic context that keeps every weight resident before the trunk is streamed. */
    static final int AUTO_CONTEXT_FLOOR = 4096;
    /** Streaming slots reserved before the context is fitted, so uploads overlap compute. */
    static final int MIN_STREAMING_SLOTS = 2;
    private static final double MIB = 1024.0 * 1024.0;

    private final GGUFFileParser parser;
    private final GGUFTokenizer tokenizer;
    private final String modelName;
    private final ObserverSlot observer = new ObserverSlot();

    private EngineConfig config;
    private ModelSpec spec;
    private ModelSpec forwardSpec;
    private MemoryPlan plan;
    private VRAMBudget.FreeVram freeAtLoad;
    private long allocatedAtLoad;
    private HWAcceleration ownedAcceleration;
    private ComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;
    private TensorCatalog catalog;
    private PrismHadamardLoader.Result prism;
    private WeightStore store;
    private GpuPlaceholders placeholders;
    private GpuScratch scratch;
    private GpuOps ops;
    private DeviceModelMemory memory;
    private GpuWeightUploader uploader;
    /** The tensors placed on the device; replaced when the NextN blocks are placed on demand. */
    private WeightPlacement weightPlacement;
    private StreamingLayerWeightSource streaming;
    private List<LayerWeightSet> streamedSets = Collections.emptyList();
    private GpuForwardContext ctx;
    private GpuBlockRegistry registry;
    private SwitchedLoraExtra.Router router;
    private ForwardBackend rawBackend;
    private ForwardBackend forward;
    private LoraAttachment loras;
    /** Adapters loaded and validated before placement, attached after the forward blocks are bound. */
    private List<LoraAttachment.Binding> loraBindings = Collections.emptyList();
    /** Device bytes still to be allocated after placement (the adapter uploads), held back from the VRAM budget. */
    private long pendingDeviceBytes;
    /** The configured draft model's memory, fitted beside the context ({@link LLMModelFactory#draftCompanion}). */
    private ContextFitter.Companion drafter = ContextFitter.Companion.NONE;
    /**
     * Device bytes the context fit held for memory a drafter allocates after loading (the NextN draft caches and the
     * draft model's memory, {@link ContextFitter#reservedBytes}), held back from the VRAM budget of the streaming
     * slots.
     */
    private long drafterDeviceBytes;

    private CpuWeights cpuWeights;
    private PoolingHead poolingHead;
    private DecisionScorer decisionScorer;
    private GpuWavTokenizer wavTokenizer;
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
        this.parser = new GGUFFileParser(path);
        try {
            this.tokenizer = new GGUFTokenizer(parser);
            String name = parser.getString("general.name");
            this.modelName = name != null ? name : parser.getString("general.architecture");
            load(sharedBackend, cfg);
        } catch (Throwable failure) {
            releaseAfterFailedLoad(failure);
            throw failure;
        }
    }

    // ============================================================================================
    // Construction
    // ============================================================================================

    private void load(ComputeBackend sharedBackend, EngineConfig cfg) throws IOException {
        GgufKv kv = GgufKv.of(parser);
        this.catalog = TensorCatalog.of(parser);
        ModelSpec initial = ModelSpecBuilder.build(kv, catalog, cfg.memory());
        if (initial.supportClass() == SupportClass.DRAFT) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " is a draft model; load it against"
                    + " its target model through DraftModelLoader");
        }
        GpuFactoryCatalog.checkRouting(initial);
        this.prism = PrismHadamardLoader.load(kv, n -> {
            TensorRef r = catalog.get(n);
            return r != null ? r.ne() : null;
        }, kv.u32(GgufKeys.SSM_TIME_STEP_RANK, 0), kv.u32(GgufKeys.SSM_GROUP_COUNT, 0));

        openBackend(sharedBackend, cfg);
        this.queue = backend.computeQueue();
        this.kernels = LlmKernels.open(backend);
        this.freeAtLoad = VRAMBudget.queryFreeVram(backend);
        this.allocatedAtLoad = backend.getAllocatedBytes();
        MemoryConfig mem = cfg.memory().forModel(initial);
        this.drafter = LLMModelFactory.draftCompanion(cfg, initial, false, this::draftDeviceWeightBytes);
        this.placeholders = GpuPlaceholders.allocate(backend, queue);
        this.scratch = GpuScratch.forModel(backend, initial, mem.ubatch());
        this.store = parser.createMappedStore();
        this.uploader = new GpuWeightUploader(backend, queue, parser.getTensors(), store, initial.arch().ggufName(),
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels))
                        .prism(prism)
                        .build());
        this.weightPlacement = WeightPlacement.of(initial, loadsNextn(cfg, mem));
        loadLoras(initial, cfg, mem);

        Placement placement = place(initial, mem);
        int derived = mem.autoContext() ? initial.hparams().nCtxTrain() : mem.nCtx();
        this.spec = placement.nCtx == derived ? initial : ModelSpecBuilder.build(kv, catalog, mem, placement.nCtx);
        this.forwardSpec = DecisionBody.forwardSpec(spec);
        this.plan = MemoryPlanner.planForContext(spec, mem, placement.nCtx);
        this.config = cfg.toBuilder().memory(mem.toBuilder().ubatch(plan.ubatch()).build()).build();
        if (plan.ubatch() < scratch.ubatch()) {
            GpuScratch larger = scratch;
            scratch = null;
            larger.close();
            scratch = GpuScratch.forModel(backend, spec, plan);
        }
        this.ops = new GpuOps(queue, kernels, scratch, placeholders);
        if (plan.kind() != MemoryKind.NONE) {
            this.memory = DeviceModelMemory.allocate(backend, queue, plan, spec.hparams().mropeTextPositions(), null);
            this.drafterDeviceBytes = ContextFitter.reservedBytes(spec, mem, plan, drafter);
        }
        uploadWeights(placement.streamed);
        bindForward();
        attachLoras();
        LOGGER.info("{} ({}) on {} {}: context {}, ubatch {}, memory {} MiB, resident weights {} MiB{}",
                modelName, spec.arch().ggufName(), backend.backendType(), backend.capabilities().getDeviceName(),
                plan.nCtx(), plan.ubatch(), mib(plan.totalBytes()), mib(uploader.residentBytes()),
                streaming == null ? "" : ", " + streaming.streamedLayers() + " layers streamed through "
                        + streaming.depth() + " slots (" + mib(streaming.deviceBytes()) + " MiB)");
        if (weightPlacement.hasNextn() && !weightPlacement.nextnOnDevice()) {
            LOGGER.info("{}: {} NextN block(s) stay off the device ({} MiB); set {}=true to draft with them",
                    modelName, spec.nextnLayers().size(), mib(uploader.plannedBytes(weightPlacement::isNextn)),
                    EngineConfig.PROP_MTP);
        }
    }

    /**
     * Whether a load places the model's NextN / MTP blocks: MTP drafting is configured ({@link EngineConfig#mtp()},
     * llama.cpp's {@code DRAFT_MTP} speculative type that sets {@code load_mtp}, common/common.cpp:1710) or the
     * context fit holds the NextN draft context ({@link MemoryConfig#nextnMemory()}).
     */
    static boolean loadsNextn(EngineConfig cfg, MemoryConfig mem) {
        return cfg.mtp() || mem.nextnMemory();
    }

    private void openBackend(ComputeBackend sharedBackend, EngineConfig cfg) {
        if (sharedBackend != null) {
            this.backend = sharedBackend;
            return;
        }
        HWAcceleration acceleration = HWAcceleration.forMode(cfg.backend().toAccelerationMode());
        this.ownedAcceleration = acceleration;
        this.backend = acceleration.createBackend(cfg.backend().toAccelerationMode(), LlmKernels.requirement());
    }

    /**
     * Device bytes left for this model: the free VRAM measured when the model began loading (before its buffers, as
     * {@link VRAMBudget#queryFreeVram} requires) less everything the backend has allocated since and the allocations
     * still to come: the adapter uploads ({@link #pendingDeviceBytes}) and, once the context is chosen, the drafter
     * memory the fit held beside it ({@link #drafterDeviceBytes}).
     */
    private long usableVram() {
        long allocatedSince = Math.max(0L, backend.getAllocatedBytes() - allocatedAtLoad);
        return VRAMBudget.memoryBudgetBytes(backend.capabilities().getTotalVramBytes(), freeAtLoad, 0L,
                allocatedSince, Math.addExact(pendingDeviceBytes, drafterDeviceBytes));
    }

    /**
     * Device bytes of a drafter file's weights on this model's device: the tensors {@code placed} its drafter places
     * there for the route it drafts with (never an MTP draft file's trunk), planned with this model's kernel
     * type policy as the drafter uploads them (a draft model's own load, or the eagle3 / gemma4-assistant / MTP draft
     * file weights {@code LLMModelFactory} uploads). The PRISM transforms of a draft file attach to its device tensors
     * without changing their plans, so the planner needs none. Plans only; nothing is allocated.
     */
    private long draftDeviceWeightBytes(GGUFFileParser draft, Predicate<GGUFTensorInfo> placed) throws IOException {
        try (WeightStore draftStore = draft.createMappedStore()) {
            GpuWeightUploader planner = new GpuWeightUploader(backend, queue, draft.getTensors(), draftStore,
                    draft.getString("general.architecture"),
                    GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
            try {
                return planner.plannedBytes(placed);
            } finally {
                planner.close();
            }
        }
    }

    /**
     * The context and whether the trunk is streamed, measured against the VRAM left after the kernels, placeholders
     * and scratch are allocated; the weights are those placed on the device ({@link #weightPlacement}), and every
     * context fit also holds the NextN draft caches and the configured drafter's memory ({@link #drafter}) beside the
     * context.
     */
    private Placement place(ModelSpec initial, MemoryConfig mem) {
        long weightBytes = uploader.plannedBytes(weightPlacement::onDevice);
        long usable = usableVram();
        int train = initial.hparams().nCtxTrain();
        int floor = train > 0 ? Math.min(train, AUTO_CONTEXT_FLOOR) : AUTO_CONTEXT_FLOOR;
        boolean memoryless = initial.memoryKind() == MemoryKind.NONE;
        int residentCtx;
        if (memoryless) {
            residentCtx = mem.memorylessContext(initial);
            if (weightBytes <= usable) {
                return new Placement(residentCtx, false);
            }
        } else {
            residentCtx = fitOrZero(initial, mem, usable - weightBytes, drafter);
            if (residentCtx > 0 && (!mem.autoContext() || residentCtx >= floor)) {
                return new Placement(residentCtx, false);
            }
        }
        ModelSpec run = DecisionBody.forwardSpec(initial);
        List<LayerWeightSet> sets = MemoryConfig.singleStepInput(initial) ? Collections.<LayerWeightSet>emptyList()
                : trunkSets(run);
        if (sets.isEmpty()) {
            if (!memoryless && residentCtx > 0) {
                return new Placement(residentCtx, false);
            }
            if (!memoryless && weightBytes <= usable) {
                return new Placement(ContextFitter.fit(initial, mem, usable - weightBytes, drafter), false);
            }
            throw outOfMemory(initial, weightBytes, usable);
        }
        long base = weightBytes - streamedBytes(sets);
        long slots = StreamingLayerWeightSource.plannedBytes(run, sets, Math.min(MIN_STREAMING_SLOTS, sets.size()));
        if (base + slots > usable) {
            throw outOfMemory(initial, weightBytes, usable);
        }
        int nCtx = residentCtx;
        if (!memoryless) {
            nCtx = ContextFitter.fit(initial, mem, usable - base - slots, drafter);
            if (mem.autoContext()) {
                nCtx = Math.min(nCtx, floor);
            }
        }
        LOGGER.info("{}: {} MiB of weights leave too little of the {} MiB of usable VRAM; streaming {} trunk layers"
                + " ({} MiB)", initial.arch().ggufName(), mib(weightBytes), mib(usable), sets.size(),
                mib(streamedBytes(sets)));
        return new Placement(nCtx, true);
    }

    private static int fitOrZero(ModelSpec spec, MemoryConfig mem, long budget, ContextFitter.Companion companion) {
        if (budget < 0) {
            return 0;
        }
        try {
            return ContextFitter.fit(spec, mem, budget, companion);
        } catch (ContextFitter.ContextBudgetException e) {
            return 0;
        }
    }

    private GpuOutOfMemoryException outOfMemory(ModelSpec s, long weightBytes, long usable) {
        return new GpuOutOfMemoryException(s.arch().ggufName() + ": " + mib(weightBytes) + " MiB of weights do not fit"
                + " the " + mib(usable) + " MiB of usable VRAM" + (MemoryConfig.singleStepInput(s)
                ? " and this model keeps every weight resident" : ", even with the trunk layers streamed"),
                weightBytes, usable,
                backend.backendType().name());
    }

    /** Streamed sets of the trunk, with the uploader's device types and the PRISM input transforms. */
    private List<LayerWeightSet> trunkSets(ModelSpec s) {
        LayerWeightSet.Options options = LayerWeightSet.Options.builder(t -> uploader.deviceType(t.name()))
                .maxShardBytes(LayerWeightSet.Options.maxShardBytes(backend.capabilities()))
                .hadamard(prism.forward())
                .build();
        return LayerWeightSet.forTrunk(s, options);
    }

    private static long streamedBytes(List<LayerWeightSet> sets) {
        long sum = 0;
        for (LayerWeightSet s : sets) {
            sum += s.uploadBytes();
        }
        return sum;
    }

    /**
     * Uploads the resident weights, every tensor placed on the device ({@link #weightPlacement}) but the streamed
     * trunk, then opens the streaming source over the usable VRAM that is left after the drafter memory the context
     * fit reserved (its depth from {@code neuralfree.vram.depth}, else the largest that fits).
     */
    private void uploadWeights(boolean streamed) {
        WeightPlacement placed = weightPlacement;
        if (!streamed) {
            uploader.uploadResident(t -> !placed.onDevice(t));
            return;
        }
        List<LayerWeightSet> sets = trunkSets(forwardSpec);
        Set<String> trunk = new HashSet<>(LayerWeightSet.allNames(sets));
        uploader.uploadResident(t -> !placed.onDevice(t) || trunk.contains(t.name()));
        this.streaming = StreamingLayerWeightSource.open(backend, queue, uploader, store, parser.getTensors(),
                forwardSpec, sets, config, usableVram());
        this.streamedSets = sets;
    }

    /** Builds the context, the registry and the forward backend, and prewarms every kernel they launch. */
    private void bindForward() {
        GpuDiagnostics.Config diag = GpuDiagnostics.Config.fromSystemProperties();
        if (diag.enabled()) {
            observer.setBase(new GpuDiagnostics(queue, diag));
        }
        GpuForwardContext.Builder b = GpuForwardContext.builder(kernels, ops, uploader, forwardSpec)
                .observer(observer);
        if (streaming != null) {
            b.weightSource(streaming);
        }
        if (memory != null) {
            b.memory(memory);
        }
        this.ctx = b.build();
        this.router = GpuFactoryCatalog.usesSwitchedLora(forwardSpec) ? SwitchedLoraExtra.Router.of(spec.hparams())
                : null;
        this.registry = GpuFactoryCatalog.defaults(router);
        kernels.prewarm(GpuFactoryCatalog.kernels(forwardSpec));
        WeightKernelPrewarm.prewarm(uploader.plans(weightPlacement::onDevice), ops, kernels);
        if (spec.arch() == Arch.WAVTOKENIZER_DEC) {
            kernels.prewarm(GpuWavTokenizer.KERNELS);
            this.wavTokenizer = GpuWavTokenizer.bind(ctx);
            return;
        }
        if (spec.supportClass() == SupportClass.ENC_DEC) {
            rawBackend = new GpuEncDecBackend(ctx, registry, GpuFactoryCatalog.headPreprocessors(),
                    tokenizer.specialTokens().bos());
        } else {
            rawBackend = new GpuForwardBackend(ctx, registry, GpuFactoryCatalog.headPreprocessors());
        }
        ForwardBackend f = rawBackend;
        if (spec.arch() == Arch.QWEN3TTS) {
            f = TtsBackbone.qwen3tts(f, spec);
        }
        if (router != null) {
            f = SwitchedLoraBackend.wrap(f, router);
        }
        this.forward = f;
        if (streaming != null) {
            streaming.warmup();
        }
    }

    /**
     * Loads the configured LoRA adapters ({@link EngineConfig#loras()}, llama.cpp {@code --lora} /
     * {@code --lora-scaled}) and validates them against the base tensors before placement, so their device uploads
     * are held back from the budget the context and streaming fits use.
     *
     * @throws UnsupportedOperationException for the WavTokenizer decoder or a refused adapter
     * @throws IOException                   when an adapter file cannot be read
     */
    private void loadLoras(ModelSpec initial, EngineConfig cfg, MemoryConfig mem) throws IOException {
        List<EngineConfig.LoraSpec> specs = cfg.loras();
        if (specs.isEmpty()) {
            return;
        }
        if (initial.arch() == Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " has no token forward pass to apply"
                    + " LoRA adapters to");
        }
        List<LoraAttachment.Binding> bindings = new ArrayList<>(specs.size());
        for (EngineConfig.LoraSpec s : specs) {
            bindings.add(LoraAttachment.Binding.load(s.path(), s.scale(), cpuWeights()));
        }
        this.loraBindings = Collections.unmodifiableList(bindings);
        this.pendingDeviceBytes = LoraAttachment.gpuDeviceBytes(bindings, mem.ubatch());
        LOGGER.info("{}: {} LoRA adapter(s) loaded, {} MiB of device uploads reserved", modelName, bindings.size(),
                mib(pendingDeviceBytes));
    }

    /**
     * Attaches the loaded LoRA adapters ({@link #loadLoras}) to the bound forward blocks: their pairs are uploaded and
     * registered in the context's projection deltas, which the blocks add after their products
     * (llama-graph.cpp:1629-1749, 2694-2709).
     *
     * @throws UnsupportedOperationException for a refused adapted product
     * @throws IllegalStateException         when an adapter tensor targets a product the model does not run
     */
    private void attachLoras() {
        if (loraBindings.isEmpty()) {
            return;
        }
        kernels.prewarm(GpuLoraDeltas.KERNELS);
        this.loras = LoraAttachment.gpu(loraBindings, ops, ctx.projectionDeltas());
        this.pendingDeviceBytes = 0L;
        LOGGER.info("{}: {} LoRA adapter(s) attached, {} deltas: {}", modelName, loraBindings.size(),
                loras.deltaCount(), loraBindings);
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

    /** The GPU context of the model, for drafters that run on the target's device and kernels (MTP). */
    public GpuForwardContext forwardContext() {
        ensureOpen();
        return ctx;
    }

    /** The block registry the model was bound with. */
    public GpuBlockRegistry blockRegistry() {
        ensureOpen();
        return registry;
    }

    /**
     * The resident device tensors: every tensor placed on the device ({@link WeightPlacement}) except the streamed
     * trunk layers. Model-level tensors and encoder layers are always resident; the NextN / MTP blocks are resident
     * once placed (at load with MTP configured, or by {@link #ensureNextnWeights()}); the tensors only the host heads
     * read never are.
     */
    public GpuWeights gpuWeights() {
        ensureOpen();
        return uploader;
    }

    /**
     * Places the NextN / MTP blocks on the device for an MTP drafter over this model's own blocks; a no-op when they
     * are resident or the model has none. A model opened without MTP ({@link #loadsNextn}) left them and their draft
     * caches out of its context fit, so the blocks are uploaded now, when the VRAM left after loading holds them and
     * the caches the drafter then allocates ({@link MemoryPlanner#planNextn(ModelSpec, MemoryPlan, int)}), and their
     * weight kernels prepared.
     *
     * @throws GpuOutOfMemoryException when they do not fit the usable VRAM; the model then has to be opened with MTP
     *                                 configured, so the context is fitted beside them
     */
    public void ensureNextnWeights() {
        ensureOpen();
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
            cpuWeights = CpuWeights.open(parser, catalog, prism);
        }
        return cpuWeights;
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
        return encode(tokens, null);
    }

    /**
     * The pooled embedding of {@code tokens} (MEAN / CLS / LAST), the classifier scores (RANK), or the concatenated
     * per-token embeddings (NONE), through the dense projections of the pooling head. Embedding models encode, and
     * so does any model whose GGUF sets a pooling type (llama.cpp embeddings).
     *
     * @throws UnsupportedOperationException for a generative model without pooling and for decision models
     *                                       ({@link #classify})
     */
    @Override
    public float[] encode(int[] tokens, int[] tokenTypes) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkEncode(spec);
        PoolingHead head = poolingHead();
        float[][] rows = hidden(tokens, tokenTypes);
        return head.pooling() == PoolingKind.NONE ? flatten(head.poolTokens(rows)) : head.pool(rows);
    }

    /**
     * Per-token embeddings of a model with NONE pooling through the dense projections; for pockettts the backbone's
     * {@code t_embd} rows.
     *
     * @throws UnsupportedOperationException for a decision model ({@link #classify}), and when the model pools per
     *                                       sequence or has no per-token embedding output
     */
    @Override
    public float[][] encodeTokens(int[] tokens) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkEncodeTokens(spec);
        return poolingHead().poolTokens(hidden(tokens, null));
    }

    @Override
    public float[] classify(int[] tokens) throws IOException {
        return classify(tokens, null);
    }

    /**
     * Classifier output of one input: the decision-head scores (clef and vllm-decision2 one value per output row;
     * modern-bert and lfm2 decision heads {@code [token][question type]} flattened row-major), or the RANK classifier
     * scores, one per {@link #classLabels()} entry.
     *
     * @param decisionOrder one decision code per token for the clef and vllm-decision2 heads, or null
     * @throws UnsupportedOperationException when the model has no classifier or decision head
     * @throws IllegalArgumentException      for an order given to a head that reads none, or of the wrong length
     */
    public float[] classify(int[] tokens, int[] decisionOrder) throws IOException {
        requireTokens(tokens);
        DecisionScorer scorer = decisionScorer();
        if (scorer != null) {
            scorer.checkOrder(tokens.length, decisionOrder);
            return scorer.score(hidden(tokens, null), tokens, decisionOrder);
        }
        if (decisionOrder != null) {
            throw new IllegalArgumentException(spec.arch().ggufName() + " reads no decision order");
        }
        PoolingKind pooling = spec.head().pooling();
        if (pooling != PoolingKind.RANK) {
            throw new UnsupportedOperationException("classification needs a RANK classifier or a decision head; "
                    + spec.arch().ggufName() + " pools by " + pooling);
        }
        return poolingHead().classify(hidden(tokens, null)).scores();
    }

    @Override
    public String[] classLabels() {
        ensureOpen();
        if (spec.head().pooling() != PoolingKind.RANK || DecisionScorer.present(spec)) {
            return new String[0];
        }
        try {
            return poolingHead().classifier().labels();
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read the classifier tensors of " + spec.arch().ggufName(), e);
        }
    }

    /**
     * Relevance of a tokenized {@code document} for a tokenized {@code query}: the RANK classifier's first output
     * over the pair packed with the vocabulary's special tokens ({@link ClassifierHead#rerankInput(GGUFTokenizer,
     * int[], int[])}).
     *
     * @throws UnsupportedOperationException when the model's pooling is not RANK, or the model has a rerank prompt,
     *                                       which needs the text ({@link #rerank(String, String)})
     */
    @Override
    public float rerank(int[] query, int[] document) throws IOException {
        ensureOpen();
        EncodeGate.checkRerank(spec);
        return poolingHead().rerank(hidden(ClassifierHead.rerankInput(tokenizer, query, document), null));
    }

    /**
     * Relevance of {@code document} for {@code query}: the RANK classifier's first output over the model's rerank
     * input ({@link ClassifierHead#rerankInput(GGUFTokenizer, String, String)}: the
     * {@code tokenizer.chat_template.rerank} prompt when the model has one, else the packed pair), as the llama.cpp
     * server scores it.
     *
     * @throws UnsupportedOperationException when the model's pooling is not RANK
     */
    @Override
    public float rerank(String query, String document) throws IOException {
        ensureOpen();
        EncodeGate.checkRerank(spec);
        return poolingHead().rerank(hidden(ClassifierHead.rerankInput(tokenizer, query, document), null));
    }

    /**
     * WavTokenizer features of audio codes through the decoder bound at load ({@link GpuWavTokenizer}); the longest
     * code sequence is the planned ubatch.
     *
     * @throws UnsupportedOperationException when the model is not a WavTokenizer decoder
     */
    @Override
    public float[][] decodeAudioCodes(int[] codes) {
        ensureOpen();
        if (spec.arch() != Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException("audio code decoding not supported by " + spec.arch().ggufName());
        }
        return wavTokenizer.decode(codes);
    }

    /** Checks that the model is open and {@code tokens} is a non-empty input. */
    private void requireTokens(int[] tokens) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("tokens must not be empty");
        }
    }

    private float[][] hidden(int[] tokens, int[] tokenTypes) {
        requireTokens(tokens);
        ForwardBackend f = forwardBackend();
        if (f.memory() != null) {
            f.reset();
        }
        return f.hiddenRows(tokens, tokenTypes);
    }

    private PoolingHead poolingHead() throws IOException {
        if (poolingHead == null) {
            poolingHead = PoolingHead.of(spec, cpuWeights());
        }
        return poolingHead;
    }

    /** The decision head of clef, vllm-decision2 and the modern-bert / lfm2 decision models, or null. */
    private DecisionScorer decisionScorer() throws IOException {
        if (decisionScorer == null && DecisionScorer.present(spec)) {
            decisionScorer = DecisionScorer.forModel(spec, cpuWeights());
        }
        return decisionScorer;
    }

    /** Concatenates {@code rows} in order. */
    private static float[] flatten(float[][] rows) {
        int total = 0;
        for (float[] r : rows) {
            total = Math.addExact(total, r.length);
        }
        float[] out = new float[total];
        int off = 0;
        for (float[] r : rows) {
            System.arraycopy(r, 0, out, off, r.length);
            off += r.length;
        }
        return out;
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
        System.out.println("Resident weights: " + mib(uploader.residentBytes())
                + " MiB in " + uploader.residentCount() + " tensors");
        if (streaming != null) {
            System.out.println("Streamed layers: " + streaming.streamedLayers() + " through " + streaming.depth()
                    + " slots (" + mib(streaming.deviceBytes()) + " MiB, "
                    + (streaming.asyncTransfers() ? "async transfers" : "transfers on the compute queue") + ")");
        }
        System.out.println("Tensors: " + parser.getTensorCount());
    }

    // ============================================================================================
    // Benchmarks
    // ============================================================================================

    /**
     * Decodes {@code tokenId} at {@code position} under a {@link GpuProfiler} and prints the per-stage times; the
     * memory is reset and filled with {@code position} copies of the token first.
     *
     * @param quiet true to time without printing (warm-up)
     * @throws UnsupportedOperationException when the model has no decoder memory
     */
    public void profileDecodeToken(int tokenId, int position, boolean quiet) {
        ForwardBackend f = requireDecoder("profileDecodeToken");
        if (position < 0 || position >= plan.nCtx()) {
            throw new IllegalArgumentException("position " + position + " is outside the context [0, "
                    + plan.nCtx() + ")");
        }
        f.reset();
        if (position > 0) {
            f.prefill(repeated(tokenId, position), 0);
        }
        GpuProfiler profiler = new GpuProfiler(queue);
        observer.use(profiler);
        try {
            f.decode(tokenId, position);
        } finally {
            observer.restore();
        }
        if (!quiet) {
            GpuProfiler.Report r = profiler.report();
            System.out.print(r.format("decode at position " + position, r.nanos(GpuProfiler.STEP)));
        }
    }

    /**
     * Profiles the decode at several positions, then times unobserved decode throughput. Call it before
     * {@code generate}; it leaves the memory empty.
     */
    public void runBenchmark(int tokenId) {
        System.out.println("\n=== GPU DECODE BENCHMARK ===");
        System.out.println("Model: " + modelName + " (" + architecture() + "), " + layerCount() + " layers, embed "
                + embeddingLength() + ", heads " + headCount() + "/" + headCountKv() + ", vocab " + vocabSize());
        if (!hasDecoderMemory()) {
            System.out.println("  " + architecture() + " has no decoder memory; there is no decode step to time.");
            return;
        }
        for (int w = 0; w < 3 && w < plan.nCtx(); w++) {
            profileDecodeToken(tokenId, w, true);
        }
        for (int pos : new int[]{0, 10, 50, 100, 500}) {
            if (pos >= plan.nCtx()) {
                break;
            }
            profileDecodeToken(tokenId, pos, false);
            System.out.println();
        }
        int primed = Math.min(10, plan.nCtx() / 2);
        int timed = Math.min(20, plan.nCtx() - primed);
        System.out.println("=== THROUGHPUT (unobserved, " + timed + " tokens) ===");
        long ns = timeDecode(tokenId, primed, timed);
        System.out.printf(Locale.ROOT, "  %d tokens in %.1f ms = %.2f ms/token = %.1f tok/s%n", timed, ns / 1e6,
                ns / 1e6 / timed, timed * 1e9 / ns);
        System.out.println("========================================\n");
    }

    /**
     * Compares the per-token weight transfer of a streamed model with its streamed decode: the time to make every
     * streamed layer resident (transfer only) against the decode of {@code tokenId}. Models whose weights are resident,
     * or whose streamed layers all fit the slots, stream nothing per token and report that instead.
     */
    public void benchmarkStreamingOverlap(int tokenId) {
        System.out.println("\n=== STREAMING OVERLAP BENCHMARK ===");
        System.out.println("Model: " + modelName + " (" + architecture() + "), " + layerCount() + " layers");
        if (streaming == null) {
            System.out.println("  Layer weights are resident in VRAM; nothing streams per token.");
        } else if (streaming.depth() >= streaming.streamedLayers()) {
            System.out.println("  All " + streaming.streamedLayers() + " streamed layers fit the " + streaming.depth()
                    + " slots; after warm-up nothing streams per token.");
        } else if (!hasDecoderMemory() || plan.nCtx() < 4) {
            System.out.println("  " + architecture() + " has no decode step to time.");
        } else {
            long bytesPerToken = streamedBytes(streamedSets);
            queue.finish();
            long t0 = System.nanoTime();
            for (LayerWeightSet s : streamedSets) {
                streaming.await(s.weightLayer());
            }
            queue.finish();
            long transferNs = System.nanoTime() - t0;
            int warm = Math.min(2, plan.nCtx() / 2);
            int timed = Math.min(10, plan.nCtx() - warm);
            long decodeNs = timeDecode(tokenId, warm, timed) / timed;
            double transferMs = transferNs / 1e6;
            double decodeMs = decodeNs / 1e6;
            System.out.printf(Locale.ROOT, "  %-30s %10.1f MB%n", "Streamed weights per token:", bytesPerToken / MIB);
            System.out.printf(Locale.ROOT, "  %-30s %10.3f ms  (%.2f GB/s)%n", "Transfer only (per token):",
                    transferMs, transferNs > 0 ? bytesPerToken / (double) transferNs : 0.0);
            System.out.printf(Locale.ROOT, "  %-30s %10.3f ms  (%d slots, %d tokens)%n",
                    "Streamed decode (per token):", decodeMs, streaming.depth(), timed);
            System.out.printf(Locale.ROOT, "  %-30s %10.1f%%%n", "Transfer share of decode:",
                    decodeNs > 0 ? 100.0 * transferNs / decodeNs : 0.0);
            System.out.printf(Locale.ROOT, "  %-30s %10.3f ms%n", "Decode beyond transfer:",
                    Math.max(0.0, decodeMs - transferMs));
        }
        System.out.println("========================================\n");
    }

    /** Fills {@code primed} positions, then returns the nanoseconds of {@code timed} decodes; leaves memory empty. */
    private long timeDecode(int tokenId, int primed, int timed) {
        ForwardBackend f = requireDecoder("benchmark");
        f.reset();
        try {
            if (primed > 0) {
                f.prefill(repeated(tokenId, primed), 0);
            }
            queue.finish();
            long t0 = System.nanoTime();
            for (int i = 0; i < timed; i++) {
                f.decode(tokenId, primed + i);
            }
            queue.finish();
            return Math.max(1L, System.nanoTime() - t0);
        } finally {
            f.reset();
        }
    }

    private boolean hasDecoderMemory() {
        return forward != null && ctx.hasMemory() && !isEncoderDecoder();
    }

    private ForwardBackend requireDecoder(String what) {
        ensureOpen();
        if (!hasDecoderMemory()) {
            throw new UnsupportedOperationException(what + " needs a decoder with memory; " + architecture()
                    + " (" + spec.supportClass() + ") has none");
        }
        return forward;
    }

    private static int[] repeated(int token, int n) {
        int[] t = new int[n];
        Arrays.fill(t, token);
        return t;
    }

    // ============================================================================================
    // Lifecycle
    // ============================================================================================

    /**
     * Releases everything through its owner, in reverse order of acquisition: the audio decoder and the forward
     * backend (blocks and heads), the streaming slots, the memory, the dispatch helpers, scratch and placeholders,
     * the resident weights, then returns a shared CUDA backend's pooled device memory to the driver
     * ({@link #trimSharedCudaPool}), then the host weights, the mapped file and the parser, and last the backend when
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
        first = closeStep(wavTokenizer, first);
        first = closeStep(loras, first);
        first = closeStep(rawBackend, first);
        first = closeStep(streaming, first);
        first = closeStep(memory, first);
        first = closeStep(ops, first);
        first = closeStep(scratch, first);
        first = closeStep(placeholders, first);
        first = closeStep(uploader, first);
        first = closeStep(this::trimSharedCudaPool, first);
        first = closeStep(cpuWeights, first);
        first = closeStep(store, first);
        first = closeStep(parser::close, first);
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

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("model " + modelName + " is closed");
        }
    }

    /** Mebibytes with one decimal. */
    private static String mib(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }

    @Override
    public String toString() {
        return "GGUFModelGPU{" + modelName + ", " + (spec != null ? spec.arch().ggufName() : "?") + ", "
                + (backend != null ? backend.backendType() : "no backend") + '}';
    }

    // ============================================================================================
    // Collaborators
    // ============================================================================================

    /** Context length and weight placement chosen at load. */
    private static final class Placement {
        final int nCtx;
        final boolean streamed;

        Placement(int nCtx, boolean streamed) {
            this.nCtx = nCtx;
            this.streamed = streamed;
        }
    }

    /**
     * The model's execution observer: the diagnostics from {@code neuralfree.diag} (or nothing), temporarily replaced
     * by a profiler while a benchmark runs. Executors consult {@link #enabled()} per step.
     */
    private static final class ObserverSlot implements ExecutionObserver {
        private ExecutionObserver base = ExecutionObserver.NONE;
        private ExecutionObserver active = ExecutionObserver.NONE;

        void setBase(ExecutionObserver o) {
            base = Objects.requireNonNull(o, "observer");
            active = base;
        }

        void use(ExecutionObserver o) {
            active = Objects.requireNonNull(o, "observer");
        }

        void restore() {
            active = base;
        }

        @Override
        public boolean enabled() {
            return active.enabled();
        }

        @Override
        public void onStepStart(int pos0, int nTokens) {
            active.onStepStart(pos0, nTokens);
        }

        @Override
        public void onStageStart(int layer, Stage stage) {
            active.onStageStart(layer, stage);
        }

        @Override
        public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
            active.onStageEnd(layer, stage, result, nRows, width);
        }

        @Override
        public void onStepEnd(int pos0, int nTokens) {
            active.onStepEnd(pos0, nTokens);
        }
    }
}
