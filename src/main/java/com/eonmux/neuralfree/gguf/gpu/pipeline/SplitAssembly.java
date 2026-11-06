package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GpuHostedLayers;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuDiagnostics;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuLayerExecutor;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuSplitForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.PooledFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.forward.SplitHostStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.HostPlacement;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeCacheLoader;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeCachePool;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.MoePoolTensors;
import com.eonmux.neuralfree.gguf.gpu.weights.PlacementFitter;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightShards;
import com.eonmux.neuralfree.gguf.hybrid.HostLayerStage;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionBody;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.ops.heads.GpuLoraDeltas;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertHeat;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecValidator;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import java.util.function.IntPredicate;
import java.util.function.Predicate;

/**
 * The inputs and the per-stage state of one layer-split load, in the order the load acquires it
 * ({@link GpuSplitSession#open}): the stages, the layout (the first layers the host keeps and the spans of the
 * GPUs), the weight placement of every stage (the host placement rules, the host layers' tensors, a per-stage
 * {@code --fit}), the context every stage fits beside its weights (with the NextN draft caches on the stage of the
 * last layer and the drafter on the main stage), the memory of every stage and of the host layers, the MoE expert
 * cache of every stage that hosts routed experts (with {@code neuralfree.moe.cache.pool} the layers each GPU's cache
 * serves, whose blocks run there), and the split forward backend.
 *
 * <p>Under {@code -ngl auto} the layout fits as a one-GPU load does ({@code GpuLoadPlacement}): a load that does not
 * hold every span resident first moves the routed experts of the leading layers of each span to RAM, then the first
 * trunk layers to the host, and streams only when neither makes every span resident. The host layers are the leftover
 * of the {@linkplain BackFill back-to-front fill} of the GPUs at the chosen context, which sizes the spans from each
 * device's free memory and rewrites the shares rather than spreading a shared host-layer count by a fixed
 * {@code --tensor-split}; a {@code --tensor-split} the user gives is kept. Mutable during assembly, not thread-safe.</p>
 */
final class SplitAssembly {

    private static final Logger LOGGER = LogManager.getLogger(SplitAssembly.class);
    private static final double MIB = 1024.0 * 1024.0;

    final SplitRequest request;
    final EngineConfig cfg;
    final ModelSpec initial;
    final ModelSpec forwardSpec;
    final MemoryConfig mem;
    final MoeExpertHeat heat;
    final List<GpuDevices.Selected> devices;
    final int main;
    /** The stage of the last trunk layer, which holds the NextN / MTP blocks. */
    final int nextnStage;
    final boolean loadNextn;
    /**
     * The first layers the host keeps and the experts it hosts ({@code -ngl}, {@code -ot}, ...), or null; the layout
     * ({@link #layOut}) may move further first layers to the host under {@code -ngl auto}.
     */
    LayerOffload offload;
    int hostLayers;
    LayerSpans spans;
    /** True when {@code -ngl auto} moved the host layers of the layout to the host. */
    boolean autoHostLayers;
    final MoeCacheConfig moeCacheSettings;
    final boolean moeCacheConfigured;
    final List<SplitStage> stages = new ArrayList<>();
    final List<GpuForwardContext> contexts = new ArrayList<>();
    WeightPlacement placement;
    /**
     * The GPUs each hosted MoE layer's cache and block run on ({@code neuralfree.moe.cache.pool}), or null when the
     * cache serves each stage's own layers.
     */
    MoeCachePool moePool;
    /** The block tensors of each pooled layer, by weight layer; empty when the pool is off. */
    private final Map<Integer, MoePoolTensors> poolTensors = new HashMap<>();
    GpuHostedLayers hosted = GpuHostedLayers.NONE;
    ContextFitter.Companion drafter = ContextFitter.Companion.NONE;
    long[] weightBytes;
    long[] pendingBytes;
    /**
     * Device bytes one per-request adapter slot reserves on each stage under the current layout
     * ({@code neuralfree.lora.enable}); zeros without it.
     */
    long[] slotReserve;
    /** The {@code --fit} decision of every stage under the current layout, for the load log; null without a fit. */
    private String[] fitDecisions;
    /**
     * The leading trunk layers of each stage whose routed experts the automatic fit moves to the host, one count per
     * stage; null when the automatic fill chose no expert overflow.
     */
    private int[] expertOverflow;
    int nCtx;
    ModelSpec spec;
    MemoryPlan plan;
    EngineConfig effective;
    GpuBlockRegistry registry;
    HeapModelMemory hostMemory;
    private CpuWeights cpuWeights;
    private CpuForwardContext cpuContext;

    /**
     * @throws IllegalArgumentException      for a host placement the split cannot lay out (no layer left for a GPU,
     *                                       a KV-sharing pair across the host boundary), naming the setting
     * @throws UnsupportedOperationException for the MoE expert cache beside host layers, or a host placement the
     *                                       hybrid machinery refuses
     */
    SplitAssembly(SplitRequest request, List<GpuDevices.Selected> devices) {
        this.request = request;
        this.cfg = request.cfg();
        this.initial = request.initial();
        ModelSpec run = DecisionBody.forwardSpec(initial);
        this.forwardSpec = request.controlVectors() != null ? ControlVector.applied(run) : run;
        this.mem = cfg.memory().forModel(initial);
        this.heat = request.ranking() != null ? request.ranking().heat() : null;
        this.devices = devices;
        this.main = cfg.mainGpu();
        this.nextnStage = devices.size() - 1;
        this.loadNextn = cfg.mtp() || mem.nextnMemory();
        this.offload = LayerOffload.requested(initial, cfg, request.source().getTensors());
        this.hostLayers = offload != null ? offload.firstGpuLayer() : 0;
        this.spans = LayerSpans.of(forwardSpec, devices.size(), cfg.tensorSplit(), hostLayers);
        this.moeCacheSettings = HostPlacement.moeCacheSettings(cfg);
        this.moeCacheConfigured = HostPlacement.moeCacheRequested(cfg)
                && !MoeCacheLoader.routedMoeLayers(initial).isEmpty();
        if (devices.size() > 1 && initial.arch() == Arch.ZAMBA2) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + ": the shared transformer blocks are"
                    + " model-level tensors, which only the main stage holds; run the model on one device or keep"
                    + " trunk layers on the host with a single GPU");
        }
        if (HostPlacement.moeCacheRequested(cfg) && hostLayers > 0) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + ": a split that keeps trunk layers"
                    + " on the host (" + offload + ") cannot be combined with the MoE expert cache, which keeps the"
                    + " routed experts of the device layers in host RAM; clear " + EngineConfig.PROP_N_GPU_LAYERS
                    + " / " + EngineConfig.PROP_TENSOR_OVERRIDES + " / " + EngineConfig.PROP_N_CPU_FFN + ", or "
                    + MoeCacheConfig.PROP_MOE_CACHE + " / " + EngineConfig.PROP_MOE_CACHE_MB);
        }
        if (cfg.moeCachePool() && (!request.loras().isEmpty() || cfg.loraConfig().enabled())) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + ": "
                    + EngineConfig.PROP_MOE_CACHE_POOL + " moves a hosted MoE block to the GPU that caches its"
                    + " experts, but the LoRA adapters bind to the layer's own stage; run without "
                    + EngineConfig.PROP_LORA + " / "
                    + com.eonmux.neuralfree.gguf.engine.lora.LoraConfig.PROP_ENABLE + ", or without "
                    + EngineConfig.PROP_MOE_CACHE_POOL);
        }
    }

    GpuSplitSession assemble(HWAcceleration acceleration) throws IOException {
        openStages();
        SplitStage mainStage = stages.get(main);
        drafter = request.drafterOn(mainStage.backend(), mainStage.queue(), mainStage.kernels());
        layOut();
        LOGGER.info("{} ({}) split over {} device(s), main {}: {}{}", initial.arch().ggufName(),
                initial.supportClass(), devices.size(), devices.get(main).name(), spans,
                hostLayers > 0 ? ", layers 0.." + (hostLayers - 1) + " on the host"
                        + (autoHostLayers ? " (" + EngineConfig.PROP_N_GPU_LAYERS + "=auto)" : "") : "");
        if (fitDecisions != null) {
            for (String d : fitDecisions) {
                LOGGER.info("{}: {}", initial.arch().ggufName(), d);
            }
        }
        logSlotReserve();
        resolveHosted();
        int derived = mem.autoContext() ? initial.hparams().nCtxTrain() : mem.nCtx();
        spec = nCtx == derived ? initial : ModelSpecBuilder.build(request.kv(), request.catalog(), mem, nCtx,
                cfg.pooling());
        plan = MemoryPlanner.planForContext(spec, mem, nCtx);
        allocate();
        openMoeCaches();
        GpuSplitForwardBackend forward = bind();
        LoraAttachment att;
        try {
            att = attachLoras();
            for (int i = 0; i < stages.size(); i++) {
                SplitStage s = stages.get(i);
                WeightKernelPrewarm.prewarm(s.uploader().plans(owned(i)), s.ops(), s.kernels());
                if (s.streaming() != null) {
                    s.streaming().warmup();
                }
            }
        } catch (RuntimeException e) {
            RuntimeException closing = Closeables.close(forward, null);
            if (closing != null) {
                e.addSuppressed(closing);
            }
            throw e;
        }
        EngineConfig.Builder effectiveCfg = cfg.toBuilder()
                .memory(mem.toBuilder().ubatch(plan.ubatch()).build());
        if (cfg.tensorSplit().length == 0) {
            effectiveCfg.tensorSplit(derivedShares());
        }
        effective = effectiveCfg.build();
        return new GpuSplitSession(this, acceleration, forward, att);
    }

    /**
     * The per-stage layer counts as shares: the {@code --tensor-split} the automatic fit rewrites from what each
     * device holds (llama.cpp common/fit.cpp, {@code set_ngl_tensor_split_tbo}), when the user gave none.
     */
    private double[] derivedShares() {
        double[] shares = new double[stages.size()];
        for (int d = 0; d < shares.length; d++) {
            shares[d] = spans.spanLayers(d);
        }
        return shares;
    }

    // ------------------------------------------------------------------ stages and placement

    /** Opens every stage, with diagnostics naming its device when {@code neuralfree.diag} is set. */
    private void openStages() throws IOException {
        GpuDiagnostics.Config diag = GpuDiagnostics.Config.fromSystemProperties();
        for (int i = 0; i < devices.size(); i++) {
            SplitStage stage = SplitStage.open(i, devices.get(i), request.source(), request.transforms(), initial,
                    mem.ubatch(), cfg.gpuVramLimitBytes(), cfg.speculativeOptions().w12(),
                    cfg.speculativeOptions().w12Head(), cfg.speculativeOptions().w12F16());
            stages.add(stage);
            if (diag.enabled()) {
                stage.observer().setBase(new GpuDiagnostics(stage.queue(), diag, System.err, stage.name()));
            }
        }
    }

    /**
     * The device bytes still to come on every stage under the current spans: its adapter uploads, control-vector
     * rows and the {@code --max-loras} per-request adapter slots of {@link #slotReserve}.
     */
    private long[] pendingBytesOfSpans() {
        List<LoraAttachment.Binding> loras = request.loras();
        long maxAdapters = cfg.loraConfig().maxAdapters();
        long[] out = new long[stages.size()];
        for (int i = 0; i < out.length; i++) {
            int device = i;
            long loraBytes = loras.isEmpty() ? 0L
                    : LoraAttachment.gpuDeviceBytes(loras, mem.ubatch(), name -> deviceOf(name) == device);
            long vectorBytes = request.controlVectors() == null ? 0L
                    : (long) spans.spanLayers(i) * initial.hparams().nEmbd() * Float.BYTES;
            long slots = Math.multiplyExact(slotReserve[i], maxAdapters);
            out[i] = Math.addExact(Math.addExact(loraBytes, vectorBytes), slots);
        }
        return out;
    }

    /**
     * The device bytes one worst-case per-request adapter slot takes on every stage under the current layout: rank
     * {@code --max-lora-rank} over every adaptable product the stage holds, the host layers' products excluded
     * (their deltas run on the host); zeros without {@code neuralfree.lora.enable}.
     */
    private long[] slotReserveOfSpans() {
        LoraConfig lc = cfg.loraConfig();
        long[] out = new long[stages.size()];
        if (!lc.enabled()) {
            return out;
        }
        Predicate<String> hostTensor = offload != null ? offload::hostTensor : name -> false;
        for (int i = 0; i < out.length; i++) {
            int device = i;
            out[i] = LoraAttachment.slotReserveBytes(LoraAttachment.adaptableShapes(request.catalog(),
                    name -> deviceOf(name) == device && !hostTensor.test(name)), lc.maxRank(), mem.ubatch());
        }
        return out;
    }

    /** Logs the per-request adapter slots the chosen layout reserves on every stage, when they are enabled. */
    private void logSlotReserve() {
        LoraConfig lc = cfg.loraConfig();
        if (lc.enabled()) {
            LOGGER.info("{}: {} per-request LoRA slot(s) of rank {} reserved, {} bytes per slot by stage ({})",
                    initial.arch().ggufName(), lc.maxAdapters(), lc.maxRank(), Arrays.toString(slotReserve),
                    LoraConfig.PROP_ENABLE);
        }
    }

    // ------------------------------------------------------------------ layout

    /**
     * Chooses the layout of the split and fits it ({@link #fitLayout}): the layout the settings ask for, or, under
     * {@code -ngl auto}, the layout that holds every span resident. A requested layout that would stream moves the
     * routed experts of the leading layers to RAM and the first trunk layers to the host first ({@link #autoFit}, as
     * on one GPU); streaming is the last resort, kept only when no host layout fits. When no layout fits at all, the
     * requested one stands and its failure is thrown.
     */
    private void layOut() {
        if (cfg.tensorSplit().length == 0) {
            spans = LayerSpans.of(forwardSpec, devices.size(), freeMemoryShares(), hostLayers);
        }
        boolean auto = autoHostApplies();
        applyLayout(offload, spans);
        try {
            fitLayout();
        } catch (GpuOutOfMemoryException | ContextFitter.ContextBudgetException e) {
            if (!auto || !autoFit(true)) {
                throw e;
            }
            return;
        }
        if (auto && !keepsEveryStageResident()) {
            LayerOffload requestedOffload = offload;
            LayerSpans requestedSpans = spans;
            if (!autoFit(false)) {
                applyLayout(requestedOffload, requestedSpans);
                fitLayout();
            }
        }
    }

    /**
     * The shares of the requested layout when {@code --tensor-split} is not given: each device's usable bytes, as
     * llama.cpp's default split by free memory (src/llama-model.cpp:1660-1677). The automatic fit rewrites them from
     * what each device holds ({@link #fillLayout}).
     */
    private double[] freeMemoryShares() {
        double[] shares = new double[stages.size()];
        for (int i = 0; i < shares.length; i++) {
            shares[i] = Math.max(1.0, stages.get(i).usableBytes(0L));
        }
        return shares;
    }

    /**
     * The {@code -ngl auto} fit, host layers before streaming: the back-to-front fill at the chosen context
     * ({@link #fillLayout}), then the search over host prefixes with the configured shares ({@link #placeOnHost}).
     *
     * @param belowChosen whether a layout that fits only a smaller automatic context may win, when the requested
     *                    layout did not fit at all
     * @return true when a layout was fitted and kept
     */
    private boolean autoFit(boolean belowChosen) {
        return fillLayout() || placeOnHost(belowChosen);
    }

    /**
     * Fits the layout of the {@linkplain BackFill back-to-front fill}: each device takes the largest run of trailing
     * layers it holds resident at the chosen context, the routed experts of its leading layers moving to RAM first,
     * and the layers left in front stay on the host. The spans become the shares, so no shared host-layer count with
     * a fixed {@code --tensor-split} is needed. Only when the user gave no {@code --tensor-split}.
     *
     * @return true when the filled layout holds every span resident at the chosen context
     */
    private boolean fillLayout() {
        if (cfg.tensorSplit().length > 0 || initial.memoryKind() == MemoryKind.NONE) {
            return false;
        }
        int target = mem.autoContext() ? PlacementFitter.contextFloor(initial) : mem.nCtx();
        BackFill.Capacity capacity = capacity(target);
        BackFill.Layout filled = BackFill.fill(devices.size(), initial.nLayer(), capacity);
        if (filled == null) {
            return false;
        }
        LayerSpans candidate;
        LayerOffload candidateOffload;
        try {
            candidate = LayerSpans.fromCounts(forwardSpec, counts(filled), filled.firstGpuLayer());
            candidateOffload = LayerOffload.auto(initial, cfg, request.source().getTensors(),
                    initial.nLayer() - filled.firstGpuLayer());
        } catch (UnsupportedOperationException | IllegalArgumentException e) {
            return false;
        }
        int[] overflow = new int[devices.size()];
        for (int d = 0; d < overflow.length; d++) {
            int start = candidate.start(d);
            overflow[d] = capacity.expertOverflow(d, start, start + candidate.spanLayers(d));
            if (overflow[d] < 0) {
                return false;
            }
        }
        applyLayout(candidateOffload, candidate, overflow);
        try {
            fitLayout();
        } catch (GpuOutOfMemoryException | ContextFitter.ContextBudgetException e) {
            return false;
        }
        if (!keepsEveryStageResident() || !reachesChosenContext()) {
            return false;
        }
        logExpertOverflow(overflow);
        autoHostLayers = hostLayers > 0;
        LOGGER.info("{}: {}=auto fills the {} GPU(s) back to front at context {}: {}{}", initial.arch().ggufName(),
                EngineConfig.PROP_N_GPU_LAYERS, devices.size(), nCtx, spans,
                hostLayers > 0 ? ", layers 0.." + (hostLayers - 1) + " on the host" : "");
        return true;
    }

    /** The per-device layer counts of a filled layout. */
    private static int[] counts(BackFill.Layout filled) {
        int[] counts = new int[filled.devices()];
        for (int d = 0; d < counts.length; d++) {
            counts[d] = filled.layers(d);
        }
        return counts;
    }

    /** True when no stage of the current layout streams: every span and its context fit its device. */
    private boolean keepsEveryStageResident() {
        for (SplitStage stage : stages) {
            PlacementFitter.Fit fit = stage.fit();
            if (fit != null && fit.mode() != PlacementFitter.Mode.RESIDENT) {
                return false;
            }
        }
        return true;
    }

    /**
     * The capacity callback of every device at {@code nCtx} for the {@link BackFill}: its resident weights (the routed
     * experts of leading layers hostable), the context over the candidate layers, and the adapter bytes still to come.
     * It answers -1 when even hosting all the run's experts does not make it fit.
     */
    private BackFill.Capacity capacity(int nCtx) {
        Pending pending = pendingModel();
        StageBytes[] bytes = new StageBytes[stages.size()];
        for (int d = 0; d < bytes.length; d++) {
            bytes[d] = stageBytes(d);
        }
        return (device, start, end) -> overflowFor(device, start, end, nCtx, pending, bytes[device]);
    }

    /** {@link BackFill.Capacity#expertOverflow} for one device and run. */
    private int overflowFor(int device, int start, int end, int nCtx, Pending pending, StageBytes bytes) {
        SplitStage stage = stages.get(device);
        long usable = stage.usableBytes(pendingFor(device, start, end, pending));
        long context = ContextFitter.requiredLayerBytes(initial, stageMemory(device, mem), nCtx,
                il -> il >= start && il < end, companion(device));
        boolean last = device == nextnStage;
        long fixed = bytes.densePrefix[end] - bytes.densePrefix[start] + modelBytes(device, bytes)
                + (last ? bytes.nextnDense : 0L);
        long experts = bytes.expertPrefix[end] - bytes.expertPrefix[start] + (last ? bytes.nextnExperts : 0L);
        if (fixed + context > usable) {
            return -1;
        }
        if (fixed + experts + context <= usable) {
            return 0;
        }
        long excess = fixed + experts + context - usable;
        long hosted = 0L;
        int k = 0;
        while (k < end - start && hosted < excess) {
            hosted += bytes.experts[start + k];
            k++;
        }
        if (hosted >= excess) {
            return k;
        }
        return last && fixed + context <= usable ? end - start : -1;
    }

    /** Logs the routed experts the automatic fill keeps in RAM, one line per device. */
    private void logExpertOverflow(int[] overflow) {
        for (int d = 0; d < overflow.length; d++) {
            if (overflow[d] > 0) {
                LOGGER.info("{}: {}=auto keeps the routed experts of layers {}..{} of device {} in RAM, so its span"
                                + " stays resident", initial.arch().ggufName(), EngineConfig.PROP_N_GPU_LAYERS,
                        spans.start(d), spans.start(d) + overflow[d] - 1, stages.get(d).name());
            }
        }
    }

    /**
     * Whether {@code -ngl auto} may move trunk layers to the host, as on one GPU: no setting already keeps layers or
     * experts on the host ({@link LayerOffload#requested}), the MoE expert cache is not requested (it cannot run
     * beside host layers), and the model plans memory and evaluates its input step by step.
     */
    private boolean autoHostApplies() {
        return cfg.gpuLayers() == EngineConfig.GPU_LAYERS_AUTO && offload == null
                && !HostPlacement.moeCacheRequested(cfg) && initial.memoryKind() != MemoryKind.NONE
                && !MemoryConfig.singleStepInput(initial);
    }

    /**
     * True when the fitted context is the one the load asked for: the user's context ({@code neuralfree.ctx}, which a
     * fit keeps or refuses), or an automatic context of at least the streaming floor
     * ({@link PlacementFitter#contextFloor}).
     */
    private boolean reachesChosenContext() {
        return !mem.autoContext() || nCtx >= PlacementFitter.contextFloor(initial);
    }

    /**
     * The {@code -ngl auto} search over host prefixes, as a one-GPU load's: from one host layer up to all but one
     * layer per GPU, the first trunk layers on the host and the rest spread by the shares ({@code --tensor-split}, or
     * {@link #freeMemoryShares}); the first layout whose GPUs hold their spans resident at the chosen context wins.
     * A layout the split cannot lay out (a KV-sharing pair across the host boundary, a residual width change there)
     * is skipped; a setting that cannot run beside host layers ends the search.
     *
     * @param belowChosen whether a layout that fits only a smaller automatic context may win, the first such one,
     *                    when none reaches the chosen context (the requested layout did not fit at all)
     * @return true when a host layout was fitted and kept
     */
    private boolean placeOnHost(boolean belowChosen) {
        int firstFit = -1;
        for (int g = initial.nLayer() - 1; g >= devices.size(); g--) {
            HostLayout tried = tryHostLayout(g);
            if (tried == HostLayout.REFUSED) {
                break;
            }
            if (tried == HostLayout.UNFIT) {
                continue;
            }
            if (reachesChosenContext()) {
                return keptOnHost();
            }
            if (belowChosen && firstFit < 0) {
                firstFit = g;
            }
        }
        if (firstFit >= 0 && tryHostLayout(firstFit) == HostLayout.FITS) {
            return keptOnHost();
        }
        LOGGER.info("{}: {}=auto found no host layout whose GPUs hold their layers resident at the chosen context",
                initial.arch().ggufName(), EngineConfig.PROP_N_GPU_LAYERS);
        return false;
    }

    /** The outcome of fitting one host layout ({@link #tryHostLayout}). */
    private enum HostLayout {
        /** The layout fits and is the current one. */
        FITS,
        /** The layout does not fit the GPUs, or the split cannot lay it out. */
        UNFIT,
        /** No host layout can run with these settings (paged KV, the model's schedule). */
        REFUSED
    }

    /**
     * Applies and fits the layout that keeps the last {@code gpuLayers} trunk layers on the GPUs. Only a layout whose
     * stages all stay resident counts: streaming is the fit's last resort, as on one GPU.
     */
    private HostLayout tryHostLayout(int gpuLayers) {
        LayerOffload candidate;
        LayerSpans candidateSpans;
        try {
            candidate = LayerOffload.auto(initial, cfg, request.source().getTensors(), gpuLayers);
        } catch (UnsupportedOperationException e) {
            LOGGER.info("{}: {}=auto keeps no layer on the host: {}", initial.arch().ggufName(),
                    EngineConfig.PROP_N_GPU_LAYERS, e.getMessage());
            return HostLayout.REFUSED;
        } catch (IllegalArgumentException e) {
            return HostLayout.UNFIT;
        }
        try {
            candidateSpans = LayerSpans.of(forwardSpec, devices.size(), layoutShares(), candidate.firstGpuLayer());
        } catch (IllegalArgumentException e) {
            return HostLayout.UNFIT;
        }
        applyLayout(candidate, candidateSpans);
        try {
            fitLayout();
            return keepsEveryStageResident() ? HostLayout.FITS : HostLayout.UNFIT;
        } catch (GpuOutOfMemoryException | ContextFitter.ContextBudgetException e) {
            return HostLayout.UNFIT;
        }
    }

    /** The shares of a host-prefix candidate: {@code --tensor-split} when given, {@link #freeMemoryShares} else. */
    private double[] layoutShares() {
        double[] explicit = cfg.tensorSplit();
        return explicit.length > 0 ? explicit : freeMemoryShares();
    }

    /** Keeps the current host layout as the one {@code -ngl auto} chose. */
    private boolean keptOnHost() {
        autoHostLayers = true;
        LOGGER.info("{}: the {} GPU(s) do not hold the model resident at the chosen context; {}=auto keeps"
                        + " layers 0..{} on the host, context {}", initial.arch().ggufName(), devices.size(),
                EngineConfig.PROP_N_GPU_LAYERS, hostLayers - 1, nCtx);
        return true;
    }

    /** Makes {@code o} and {@code s} the layout, with the pending bytes of its spans and no stage placed yet. */
    private void applyLayout(LayerOffload o, LayerSpans s) {
        applyLayout(o, s, null);
    }

    /** As {@link #applyLayout(LayerOffload, LayerSpans)}, with the routed experts {@link #expertOverflow}. */
    private void applyLayout(LayerOffload o, LayerSpans s, int[] overflow) {
        offload = o;
        hostLayers = o != null ? o.firstGpuLayer() : 0;
        spans = s;
        expertOverflow = overflow;
        slotReserve = slotReserveOfSpans();
        pendingBytes = pendingBytesOfSpans();
        for (SplitStage stage : stages) {
            stage.unplace();
        }
    }

    /** Fits the current layout: the weight placement, every stage's device weight bytes, the placement and context. */
    private void fitLayout() {
        moePool = null;
        poolTensors.clear();
        placement = placeWeights();
        weightBytes = stageWeightBytes();
        planMoeCachePool();
        weightBytes = stageWeightBytes();
        place();
    }

    /** The device weight bytes of every stage under the current placement and pool. */
    private long[] stageWeightBytes() {
        long[] bytes = new long[stages.size()];
        for (int i = 0; i < stages.size(); i++) {
            bytes[i] = stages.get(i).uploader().plannedBytes(owned(i));
        }
        return bytes;
    }

    /**
     * Assigns the hosted routed MoE layers to the GPUs whose caches serve them ({@code neuralfree.moe.cache.pool}):
     * the layers in layer order, each GPU taking a run of experts' bytes proportional to the VRAM it has left after
     * its weights. A layer whose block the hosted path cannot serve, whose residual stream is wider than the FFN
     * input, or whose owner is its own stage keeps its block where it is; the others run it on the owner
     * ({@link PooledFfnBlock}).
     *
     * <p>Runs after the placement, so the shares see the host placement's device weights; the pool then rewrites
     * {@link #weightBytes}, and {@link #place()} fits the context beside the moved tensors.</p>
     */
    private void planMoeCachePool() {
        if (!cfg.moeCachePool() || !moeCacheConfigured || stages.size() < 2) {
            return;
        }
        List<LayerSpec> candidates = poolCandidates();
        if (candidates.isEmpty()) {
            LOGGER.info("{}: {}=true finds no hosted routed layer whose block can move to another GPU",
                    initial.arch().ggufName(), EngineConfig.PROP_MOE_CACHE_POOL);
            return;
        }
        long[] expertBytes = new long[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            expertBytes[i] = HostPlacement.hostExpertBytes(Collections.singletonList(candidates.get(i)),
                    request.catalog());
        }
        MoeCachePool planned = MoeCachePool.plan(stages.size(), candidates, expertBytes, poolShares());
        if (planned == null) {
            return;
        }
        moePool = planned;
        for (LayerSpec l : candidates) {
            poolTensors.put(l.weightLayer(), MoePoolTensors.of(l));
        }
        LOGGER.info("{}: {}=true assigns the hosted MoE cache over {} GPU(s): {}", initial.arch().ggufName(),
                EngineConfig.PROP_MOE_CACHE_POOL, stages.size(), poolDescription(planned));
    }

    /**
     * The hosted routed layers whose whole block can run on another GPU: the hosted path serves them, their residual
     * stream is the FFN input width (the block moves the FFN input rows alone), and they are trunk layers of a GPU.
     * The others keep their block on their own stage and are logged.
     */
    private List<LayerSpec> poolCandidates() {
        List<LayerSpec> hosted = HostPlacement.hostedMoeLayers(initial, request.catalog(), placement,
                il -> spans.deviceOfWeightLayer(il) >= 0);
        List<LayerSpec> candidates = new ArrayList<>(hosted.size());
        for (LayerSpec l : hosted) {
            if (l.residual().streamWidth() != initial.hparams().nEmbd()) {
                LOGGER.info("{}: layer {} keeps its hosted MoE block on its own device: its residual stream is {}"
                                + " wide, the block reads {}; {}=true moves only the blocks of the standard width",
                        initial.arch().ggufName(), l.il(), l.residual().streamWidth(),
                        initial.hparams().nEmbd(), EngineConfig.PROP_MOE_CACHE_POOL);
                continue;
            }
            try {
                GpuMoeFfn.checkSupported(initial, l);
            } catch (UnsupportedOperationException e) {
                LOGGER.info("{}: layer {} keeps its hosted MoE block on its own device: {}", initial.arch()
                        .ggufName(), l.il(), e.getMessage());
                continue;
            }
            candidates.add(l);
        }
        return candidates;
    }

    /** The VRAM each stage has left for its cache: the usable bytes beside its weights and KV caches. */
    private long[] poolShares() {
        long[] shares = new long[stages.size()];
        for (int i = 0; i < shares.length; i++) {
            shares[i] = Math.max(1L, stages.get(i).usableBytes(pendingBytes[i]) - weightBytes[i]);
        }
        return shares;
    }

    /** The pool's assignment as the load log line: the runs of layers and the stage each runs on. */
    private String poolDescription(MoeCachePool planned) {
        StringBuilder sb = new StringBuilder();
        for (MoeCachePool.Run run : planned.runs()) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append("layers ").append(run.firstIl()).append("..").append(run.lastIl()).append(" on ")
                    .append(stages.get(run.stage()).name());
        }
        return sb.toString();
    }

    /** The block tensors moved to another stage's pool, or null when the tensor is not one. */
    private MoePoolTensors poolTensorsOf(String storedName) {
        if (moePool == null) {
            return null;
        }
        int wl = ModelSpecValidator.blockIndexOf(storedName);
        return wl < 0 ? null : poolTensors.get(wl);
    }

    /**
     * The weight placement of the split: the host placement rules of the configuration ({@link HostPlacement}) with
     * the tensors of the host layers kept on the host, the routed experts the automatic fill moved to RAM
     * ({@link #expertOverflow}), and, under {@code --fit} without the automatic fill, the routed experts of every
     * stage whose dense weights fit its device alone moved to the host, stage by stage as each device fits its own
     * span.
     */
    private WeightPlacement placeWeights() {
        Predicate<String> fixedHost = offload != null ? offload::hostTensor : n -> false;
        WeightPlacement rules = HostPlacement.rulePlacement(cfg, initial, loadNextn, fixedHost);
        if (expertOverflow != null) {
            fitDecisions = null;
            return rules.withHostTensors(this::autoHostExperts);
        }
        if (!HostPlacement.fitApplies(cfg, initial)) {
            fitDecisions = null;
            return rules;
        }
        WeightPlacement candidate = HostPlacement.fitCandidate(cfg, initial, loadNextn, fixedHost);
        boolean[] fits = new boolean[stages.size()];
        fitDecisions = new String[stages.size()];
        for (int i = 0; i < stages.size(); i++) {
            SplitStage s = stages.get(i);
            int device = i;
            long dense = s.uploader().plannedBytes(t -> candidate.onDevice(t) && owns(device, t.name(), candidate));
            long usable = s.usableBytes(pendingBytes[i]);
            fits[i] = dense <= usable;
            fitDecisions[i] = EngineConfig.PROP_FIT + " on device " + s.name() + ": the dense weights take "
                    + mib(dense) + " of " + mib(usable) + " MiB, so its experts "
                    + (fits[i] ? "stay in RAM" : "stay on the device");
        }
        return rules.withHostTensors(name -> fits[deviceOf(name)] && !candidate.onDeviceName(name));
    }

    /**
     * True when the automatic fill keeps the routed experts of {@code storedName} in RAM: the expert tensors of the
     * leading {@link #expertOverflow} trunk layers of the tensor's device, and those of the NextN blocks when all the
     * device's trunk experts move.
     */
    private boolean autoHostExperts(String storedName) {
        if (expertOverflow == null || !isExpert(storedName)) {
            return false;
        }
        int wl = ModelSpecValidator.blockIndexOf(storedName);
        if (wl < 0) {
            return false;
        }
        int device = spans.deviceOfWeightLayer(wl);
        if (device < 0 || device >= expertOverflow.length) {
            return false;
        }
        if (wl >= initial.nLayer()) {
            return device == nextnStage && expertOverflow[device] >= spans.spanLayers(device);
        }
        int start = spans.start(device);
        return wl >= start && wl < start + expertOverflow[device];
    }

    /** True when the stored tensor is a routed expert stack ({@link WeightShards#isExpert}). */
    private boolean isExpert(String storedName) {
        TensorRef ref = request.catalog().get(storedName);
        return ref != null && WeightShards.isExpert(storedName, ref.rank());
    }

    /** The weight layers a tensor can belong to: the trunk and the NextN / MTP blocks. */
    private int blockLayers() {
        int size = initial.nLayer();
        for (LayerSpec l : initial.nextnLayers()) {
            size = Math.max(size, l.weightLayer() + 1);
        }
        return size;
    }

    /** True when stage {@code device} uploads the model-level tensors (the embedding, the head, the norms). */
    private boolean ownsModelLevel(int device) {
        return device == main || (device == nextnStage && loadNextn && !initial.nextnLayers().isEmpty());
    }

    /** The model-level device bytes of {@code device}, zero when it uploads none. */
    private long modelBytes(int device, StageBytes bytes) {
        return ownsModelLevel(device) ? bytes.model : 0L;
    }

    /**
     * The planned device bytes of one stage's tensors, by weight layer: dense (non-expert) and routed-expert bytes
     * with their prefix sums, the model-level bytes, and the NextN / MTP layer bytes the last stage holds.
     */
    private static final class StageBytes {

        private final long[] densePrefix;
        private final long[] expertPrefix;
        private final long[] experts;
        private final long model;
        private final long nextnDense;
        private final long nextnExperts;

        private StageBytes(long[] densePrefix, long[] expertPrefix, long[] experts, long model, long nextnDense,
                           long nextnExperts) {
            this.densePrefix = densePrefix;
            this.expertPrefix = expertPrefix;
            this.experts = experts;
            this.model = model;
            this.nextnDense = nextnDense;
            this.nextnExperts = nextnExperts;
        }
    }

    /** Groups the device bytes of stage {@code device}'s upload plans by weight layer. */
    private StageBytes stageBytes(int device) {
        WeightPlacement rules = HostPlacement.rulePlacement(cfg, initial, loadNextn, n -> false);
        int size = blockLayers();
        long[] dense = new long[size];
        long[] experts = new long[size];
        long model = 0L;
        for (GpuWeightUploader.TensorPlan plan : stages.get(device).uploader().plans().values()) {
            if (!rules.onDevice(plan.info())) {
                continue;
            }
            int wl = ModelSpecValidator.blockIndexOf(plan.name());
            if (wl < 0 || wl >= size) {
                model += plan.deviceBytes();
            } else if (isExpert(plan.name())) {
                experts[wl] += plan.deviceBytes();
            } else {
                dense[wl] += plan.deviceBytes();
            }
        }
        long nextnDense = 0L;
        long nextnExperts = 0L;
        for (int wl = initial.nLayer(); wl < size; wl++) {
            nextnDense += dense[wl];
            nextnExperts += experts[wl];
        }
        return new StageBytes(prefix(dense), prefix(experts), experts, model, nextnDense, nextnExperts);
    }

    /** The bytes still to come on a stage for any run of its span: adapters, control vectors and slot reserves. */
    private static final class Pending {

        private final long[] loraPrefix;
        private final long[] slotPrefix;
        private final long loraModel;
        private final long slotModel;
        private final long loraNextn;
        private final long slotNextn;

        private Pending(long[] loraPrefix, long[] slotPrefix, long loraModel, long slotModel, long loraNextn,
                        long slotNextn) {
            this.loraPrefix = loraPrefix;
            this.slotPrefix = slotPrefix;
            this.loraModel = loraModel;
            this.slotModel = slotModel;
            this.loraNextn = loraNextn;
            this.slotNextn = slotNextn;
        }
    }

    /**
     * The pending bytes of any run of a stage's span, by weight layer: the adapter uploads
     * ({@link LoraAttachment#gpuDeviceBytes}) and the per-request adapter slots ({@link LoraAttachment#slotReserveBytes}),
     * split into the trunk layers, the NextN blocks the last stage holds and the model-level products the main stage
     * holds, as {@link #pendingBytesOfSpans} attributes them.
     */
    private Pending pendingModel() {
        int size = blockLayers();
        long[] lora = new long[size];
        long[] slots = new long[size];
        long loraModel = 0L;
        long loraNextn = 0L;
        long slotModel = 0L;
        long slotNextn = 0L;
        int ubatch = mem.ubatch();
        for (LoraAttachment.Binding b : request.loras()) {
            for (String target : b.adapter().targets()) {
                long bytes = LoraAttachment.pairDeviceBytes(b.adapter().pair(target), ubatch);
                int wl = ModelSpecValidator.blockIndexOf(target);
                if (wl >= 0 && wl < size) {
                    lora[wl] += bytes;
                } else if (wl < 0) {
                    loraModel += bytes;
                }
            }
        }
        LoraConfig lc = cfg.loraConfig();
        if (lc.enabled()) {
            for (String name : new TreeSet<>(request.catalog().names())) {
                TensorRef t = request.catalog().get(name);
                if (t == null || !name.endsWith(".weight") || (t.rank() != 2 && t.rank() != 3)) {
                    continue;
                }
                long floats = Math.multiplyExact(Math.multiplyExact((long) lc.maxRank(), t.ne(0) + t.ne(1)),
                        t.rank() == 3 ? t.ne(2) : 1L);
                if (t.rank() != 3 || t.ne(2) <= 1) {
                    floats = Math.addExact(floats, ubatch);
                }
                long bytes = Math.multiplyExact(floats, (long) Float.BYTES);
                int wl = ModelSpecValidator.blockIndexOf(name);
                if (wl >= 0 && wl < size) {
                    slots[wl] += bytes;
                } else if (wl < 0) {
                    slotModel += bytes;
                }
            }
        }
        for (int wl = initial.nLayer(); wl < size; wl++) {
            loraNextn += lora[wl];
            slotNextn += slots[wl];
        }
        return new Pending(prefix(lora), prefix(slots), loraModel, slotModel, loraNextn, slotNextn);
    }

    /** The device bytes still to come on {@code device} for trunk layers {@code [start, end)}. */
    private long pendingFor(int device, int start, int end, Pending p) {
        long lora = p.loraPrefix[end] - p.loraPrefix[start] + (device == main ? p.loraModel : 0L)
                + (device == nextnStage ? p.loraNextn : 0L);
        long vector = Math.multiplyExact((long) (end - start) * initial.hparams().nEmbd(), (long) Float.BYTES);
        long reserve = p.slotPrefix[end] - p.slotPrefix[start] + (device == main ? p.slotModel : 0L)
                + (device == nextnStage ? p.slotNextn : 0L);
        long slots = Math.multiplyExact(reserve, cfg.loraConfig().maxAdapters());
        return Math.addExact(Math.addExact(lora, vector), slots);
    }

    /** Prefix sums of {@code values}, with a leading zero. */
    private static long[] prefix(long[] values) {
        long[] out = new long[values.length + 1];
        for (int i = 0; i < values.length; i++) {
            out[i + 1] = Math.addExact(out[i], values[i]);
        }
        return out;
    }

    /**
     * The hosted MoE layers of the chosen placement. The MoE expert cache, the hosted expert blocks and the host
     * layers read the model's tensors on the host.
     */
    private void resolveHosted() throws IOException {
        hosted = GpuHostedLayers.resolve(initial, request.catalog(), placement, offload);
        if (hosted.hasCached() || offload != null) {
            cpuWeights = request.hostWeights().get();
        }
        hosted.log(request.modelName(), request.catalog(), moeCacheConfigured ? moeCacheSettings
                : MoeCacheConfig.off());
    }

    /** The device of a tensor by name: model-level tensors on the main device, layer tensors on their span. */
    int deviceOf(String name) {
        int wl = ModelSpecValidator.blockIndexOf(name);
        return wl < 0 ? main : spans.deviceOfWeightLayer(wl);
    }

    /**
     * True when stage {@code i} uploads tensor {@code name} under {@code placed}: the tensors of its span, the
     * model-level tensors on the main stage, and, on the NextN stage when it is not the main one and the NextN blocks
     * are placed, the model-level tensors too, which the MTP head reads beside its blocks.
     */
    boolean owns(int i, String name, WeightPlacement placed) {
        MoePoolTensors block = poolTensorsOf(name);
        if (block != null && block.contains(name)) {
            return moePool.ownerOf(ModelSpecValidator.blockIndexOf(name)) == i;
        }
        return owns(spans, main, nextnStage, i, name, placed);
    }

    /** {@link #owns(int, String, WeightPlacement)} over the given spans, main stage and NextN stage. */
    static boolean owns(LayerSpans spans, int main, int nextnStage, int i, String name, WeightPlacement placed) {
        int wl = ModelSpecValidator.blockIndexOf(name);
        if (wl >= 0) {
            return spans.deviceOfWeightLayer(wl) == i;
        }
        return i == main || i == nextnStage && placed.hasNextn() && placed.nextnOnDevice();
    }

    /** The tensors stage {@code i} uploads under the current placement. */
    Predicate<GGUFTensorInfo> owned(int i) {
        WeightPlacement placed = placement;
        return t -> placed.onDevice(t) && owns(i, t.name(), placed);
    }

    /** The memory settings stage {@code i} fits with: only the NextN stage holds the NextN draft caches. */
    MemoryConfig stageMemory(int i, MemoryConfig memory) {
        return i == nextnStage || !memory.nextnMemory() ? memory : memory.toBuilder().nextnMemory(false).build();
    }

    /** The drafter stage {@code i} holds beside its caches: the main stage holds it, the others none. */
    ContextFitter.Companion companion(int i) {
        return i == main ? drafter : ContextFitter.Companion.NONE;
    }

    /** The layers of device {@code device}'s span, by memory slot. */
    IntPredicate spanFilter(int device) {
        return il -> il >= spans.start(device) && il < spans.start(device) + spans.spanLayers(device);
    }

    /**
     * Chooses every stage's placement and the context they share. A model that evaluates its input in one step, a
     * split with a host placement and a split whose routed experts the MoE expert cache serves keep every device
     * weight resident, as on one device. Otherwise each stage fits over its own span; when the stages fit different
     * contexts, they fit again at the smallest, so a stage that streamed only to reach a larger context keeps its
     * weights resident.
     */
    private void place() {
        if (MemoryConfig.singleStepInput(initial) || offload != null || moeCacheConfigured) {
            nCtx = residentContext();
            return;
        }
        int[] fitted = fitStages(mem);
        int smallest = Integer.MAX_VALUE;
        boolean differ = false;
        for (int c : fitted) {
            differ |= smallest != Integer.MAX_VALUE && c != smallest;
            smallest = Math.min(smallest, c);
        }
        nCtx = smallest;
        if (differ) {
            fitStages(mem.toBuilder().nCtx(nCtx).build());
        }
    }

    /** Fits every stage at {@code memory}; returns the context of each. */
    private int[] fitStages(MemoryConfig memory) {
        int[] out = new int[stages.size()];
        int floor = PlacementFitter.contextFloor(initial);
        WeightPlacement placed = placement;
        for (int i = 0; i < stages.size(); i++) {
            SplitStage s = stages.get(i);
            int device = i;
            PlacementFitter.Fit fit = new PlacementFitter.Builder(initial, forwardSpec, request.catalog(),
                    s.uploadOptions(request.transforms()),
                    ref -> placed.onDevice(ref.name()) && owns(device, ref.name(), placed))
                    .usableBytes(Math.max(0L, s.usableBytes(pendingBytes[i])))
                    .weightBytes(weightBytes[i])
                    .memory(stageMemory(i, memory))
                    .companion(companion(i))
                    .layers(spanFilter(i))
                    .contextFloor(floor)
                    .backendType(s.backend().backendType().name())
                    .build();
            s.placed(fit);
            out[i] = fit.nCtx();
        }
        return out;
    }

    /** The context of a load whose device weights all stay resident: the smallest per-device fit beside them. */
    private int residentContext() {
        int fit = Integer.MAX_VALUE;
        for (int i = 0; i < stages.size(); i++) {
            SplitStage s = stages.get(i);
            long usable = s.usableBytes(pendingBytes[i]);
            if (weightBytes[i] > usable) {
                throw new GpuOutOfMemoryException(initial.arch().ggufName() + ": the " + mib(weightBytes[i])
                        + " MiB of weights of device " + s.name() + " do not fit the " + mib(usable)
                        + " MiB of usable VRAM, and " + residentReason() + "; spread them with "
                        + EngineConfig.PROP_TENSOR_SPLIT + " or free device memory", weightBytes[i], usable,
                        s.backend().backendType().name());
            }
            if (initial.memoryKind() != MemoryKind.NONE) {
                fit = Math.min(fit, fitLayers(s, usable - weightBytes[i], i));
            }
        }
        return fit == Integer.MAX_VALUE ? mem.memorylessContext(initial) : fit;
    }

    /** Why the load keeps every device weight resident, for the refusal of a stage whose weights do not fit. */
    private String residentReason() {
        if (offload != null) {
            return "a host placement (" + offload + ") keeps the device layers resident, as on one GPU; keep more"
                    + " layers on the host with " + EngineConfig.PROP_N_GPU_LAYERS;
        }
        if (moeCacheConfigured) {
            return MoeCacheConfig.PROP_MOE_CACHE + " keeps the routed experts in host RAM instead of streaming the"
                    + " trunk; lower the context, or clear it to stream the trunk layers";
        }
        return "this model keeps every weight resident";
    }

    private int fitLayers(SplitStage s, long budget, int i) {
        try {
            return ContextFitter.fitLayers(initial, stageMemory(i, mem), budget, spanFilter(i), companion(i));
        } catch (ContextFitter.ContextBudgetException e) {
            throw new GpuOutOfMemoryException(initial.arch().ggufName() + ": no context fits device "
                    + s.name() + " (" + mib(budget) + " MiB left beside its weights): " + e.getMessage(),
                    e.requiredBytes(), e.budgetBytes(), s.backend().backendType().name());
        }
    }

    // ------------------------------------------------------------------ allocation

    /**
     * Sizes every stage's scratch to the plan, allocates its memory and holds the bytes its fit reserved for later
     * (the NextN draft caches, the drafter) out of its later budgets, uploads its resident weights and opens its
     * streaming source, whose slots take the VRAM the stage has left beside its adapter uploads; then the host
     * layers' scratch, heap memory and context.
     */
    private void allocate() throws IOException {
        boolean memoryless = plan.kind() == MemoryKind.NONE;
        long ramCache = ramCacheShare();
        long streamedTotal = 0L;
        for (SplitStage s : stages) {
            streamedTotal += s.streamedBytes();
        }
        for (int i = 0; i < stages.size(); i++) {
            SplitStage s = stages.get(i);
            s.fitScratch(spec, plan);
            if (!memoryless) {
                MemoryConfig stageMem = stageMemory(i, mem);
                MemoryPlan stagePlan = MemoryPlanner.planForContext(spec, stageMem, nCtx, spanFilter(i));
                s.allocateMemory(stagePlan, spec.hparams().mropeTextPositions());
                s.reserve(ContextFitter.reservedBytes(spec, stageMem, stagePlan, companion(i)));
            }
            s.uploadResident(owned(i));
            long share = streamedTotal == 0L ? 0L
                    : (long) (ramCache * ((double) s.streamedBytes() / streamedTotal));
            s.openStreaming(request.source(), forwardSpec, cfg, Math.max(0L, s.usableBytes(pendingBytes[i])), share);
            LOGGER.info("{}: {}", initial.arch().ggufName(), GpuSplitSession.describe(s, spans, weightBytes[i]));
        }
        if (offload != null) {
            CpuScratch scratch = CpuScratch.forModel(spec, plan.ubatch());
            hostMemory = HeapModelMemory.allocate(plan, spec, null, offload::hostLayer);
            CpuForwardContext.Builder b = CpuForwardContext.builder(forwardSpec, cpuWeights, scratch)
                    .memory(hostMemory);
            if (heat != null) {
                b.heat(heat);
            }
            cpuContext = b.build();
        }
    }

    /** The heap bytes of the host weight cache the streaming stages share ({@code neuralfree.ram.cache.mb}). */
    private long ramCacheShare() {
        long modelBytes = 0L;
        for (GGUFTensorInfo t : request.source().getTensors()) {
            modelBytes += t.sizeInBytes();
        }
        return WeightDataProvider.ramCacheBytes(cfg, modelBytes);
    }

    /**
     * Opens the MoE expert cache of every stage whose span has routed MoE layers, over the VRAM the stage has left
     * beside its pending uploads and its reservations; a {@code neuralfree.moe.cache.mb} budget applies per device.
     * With the pooled cache each stage's cache serves the layers {@link #moePool} assigned to it, wherever their spans
     * run.
     */
    private void openMoeCaches() {
        if (!moeCacheConfigured) {
            return;
        }
        List<LayerSpec> routed = MoeCacheLoader.routedMoeLayers(initial);
        for (int i = 0; i < stages.size(); i++) {
            SplitStage s = stages.get(i);
            List<LayerSpec> own;
            if (moePool != null) {
                own = moePool.layersOf(i);
            } else {
                own = new ArrayList<>();
                for (LayerSpec l : routed) {
                    if (spanFilter(i).test(l.weightLayer())) {
                        own.add(l);
                    }
                }
            }
            String name = request.modelName() + " on " + s.name();
            MoeExpertCache cache = MoeCacheLoader.open(moeCacheSettings, request.source().getTensors(), s.store(),
                    s.uploader(), s.backend(), s.queue(), plan.ubatch(), own,
                    Math.max(0L, s.usableBytes(pendingBytes[i])), request.ranking(), name);
            s.moeCache(cache, MoeCacheLoader.openPrefetch(moeCacheSettings, cache, s.backend(), name));
        }
    }

    // ------------------------------------------------------------------ binding

    /**
     * Binds every span executor, the head on the main device, the embedding there or, when the host keeps the first
     * layers, the host layers with the host embedding, and the split backend over them. One registry serves every
     * stage: the control vectors apply on the device of each layer, and the hosted expert layers the cached path
     * cannot serve run their CPU MoE block.
     */
    private GpuSplitForwardBackend bind() {
        ControlVector vectors = request.controlVectors();
        registry = hosted.decorate(GpuFactoryCatalog.defaults(null, vectors), cpuContext);
        if (moePool != null) {
            registry = registry.withFfnDecorator(this::poolBlock);
        }
        registry.validate(forwardSpec);
        for (SplitStage s : stages) {
            s.kernels().prewarm(GpuForwardBackend.kernels(forwardSpec));
        }
        List<GpuSplitForwardBackend.Shard> shards = new ArrayList<>(stages.size());
        List<AutoCloseable> bound = new ArrayList<>(stages.size() + 1);
        try {
            // every context exists before a block binds: a pooled FFN block binds its delegate to the owner stage.
            for (SplitStage s : stages) {
                contexts.add(context(s));
            }
            for (int i = 0; i < stages.size(); i++) {
                GpuForwardContext ctx = contexts.get(i);
                List<LayerSpec> spanLayers = forwardSpec.layers().subList(spans.start(i),
                        spans.start(i) + spans.spanLayers(i));
                GpuLayerExecutor executor = GpuLayerExecutor.forStack(ctx, registry, spanLayers);
                bound.add(executor);
                shards.add(new GpuSplitForwardBackend.Shard(ctx, executor, stages.get(i).memory(),
                        spans.start(i), spanLayers.get(0).residual().streamWidth(),
                        spanLayers.get(spanLayers.size() - 1).residual().streamWidth()));
            }
            SplitHostStage host = null;
            if (hostLayers > 0) {
                host = HostLayerStage.bind(cpuContext, CpuFactoryCatalog.defaults(null, vectors), hostMemory,
                        hostLayers);
                bound.add(host);
            }
            GpuForwardContext mainCtx = shards.get(main).context();
            GpuEmbeddingStage embedding = host != null ? null : new GpuEmbeddingStage(mainCtx);
            return new GpuSplitForwardBackend(spec, shards, main, embedding,
                    new GpuLogitsHead(mainCtx, GpuFactoryCatalog.headPreprocessors()),
                    plan.kind() == MemoryKind.NONE ? null : plan, host);
        } catch (RuntimeException e) {
            RuntimeException first = null;
            for (AutoCloseable c : bound) {
                first = Closeables.close(c, first);
            }
            if (first != null) {
                e.addSuppressed(first);
            }
            throw e;
        }
    }

    /**
     * The pooled FFN block of a hosted MoE layer whose cache lives on another stage: the kind's block, bound to the
     * owner's context, wrapped so its input and output cross the boundary. A layer of the stage itself keeps the
     * block.
     */
    private GpuFfnBlock poolBlock(LayerSpec layer, GpuFfnBlock block) {
        int owner = moePool.ownerOf(layer.il());
        if (owner < 0 || owner == spans.deviceOfWeightLayer(layer.weightLayer())) {
            return block;
        }
        return new PooledFfnBlock(block, owner, contexts.get(owner), initial.hparams().nEmbd());
    }

    /**
     * The forward context of one stage: its weights or streaming source, memory, observer, its MoE expert cache, and
     * host access to the model's tensors when routed experts compute on the host.
     */
    private GpuForwardContext context(SplitStage s) {
        GpuForwardContext.Builder b = GpuForwardContext.builder(s.kernels(), s.ops(), s.uploader(), forwardSpec)
                .memory(s.memory())
                .observer(s.observer())
                .heat(heat);
        if (s.streaming() != null) {
            b.weightSource(s.streaming());
        }
        if (s.moeCache() != null) {
            b.moeExpertCache(s.moeCache());
            b.moeExpertPrefetch(s.moePrefetch());
        }
        b.moeTailSkipRank(moeCacheSettings.tailSkipRank());
        if (hosted.hasCached()) {
            b.hostWeights(cpuWeights);
        }
        return b.build();
    }

    /**
     * Attaches the adapters over every stage's deltas, and the host layers' deltas when layers run on the host; the
     * attachment also serves the per-request adapters, so it is built without adapters too.
     */
    private LoraAttachment attachLoras() {
        List<LoraAttachment.Binding> loras = request.loras();
        List<LoraAttachment.GpuDevice> targets = new ArrayList<>(stages.size());
        for (int i = 0; i < stages.size(); i++) {
            SplitStage s = stages.get(i);
            if (!loras.isEmpty()) {
                s.kernels().prewarm(GpuLoraDeltas.KERNELS);
            }
            targets.add(new LoraAttachment.GpuDevice(s.ops(), contexts.get(i).projectionDeltas()));
        }
        LoraAttachment att = cpuContext != null ? LoraAttachment.hybrid(loras, targets, cpuContext.projectionDeltas())
                : LoraAttachment.gpu(loras, targets);
        if (!loras.isEmpty()) {
            LOGGER.info("{}: {} LoRA adapter(s) attached across {} devices{}, {} deltas: {}",
                    initial.arch().ggufName(), loras.size(), stages.size(), cpuContext != null ? " and the host" : "",
                    att.deltaCount(), loras);
        }
        return att;
    }

    /** Releases the host memory and the stages opened before a failure, adding their failures to it. */
    void release(Throwable failure) {
        RuntimeException first = Closeables.close(hostMemory, null);
        for (int i = stages.size() - 1; i >= 0; i--) {
            first = Closeables.close(stages.get(i), first);
        }
        GpuDevices.closeRemote(devices);
        if (first != null) {
            failure.addSuppressed(first);
        }
    }

    static String mib(double bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }
}
