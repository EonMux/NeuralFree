package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.diag.ObserverSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuSplitForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.StreamingLayerWeightSource;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The GPU session of a layer split: it opens one backend per device ({@link GpuDevices}), assigns each a contiguous
 * span of layers ({@link LayerSpans} sized by {@code neuralfree.tensor.split}) and opens one {@link SplitStage} per
 * device ({@link SplitAssembly}). Each stage places its own weights as a one-device load does ({@link
 * com.eonmux.neuralfree.gguf.gpu.weights.PlacementFitter} over the span's layers): resident when they and the span's
 * caches fit, otherwise with part of the span streamed from the host through a streaming source of the stage's own.
 * The host placement settings apply per stage over the hybrid CPU+GPU machinery: the first layers stay on the host
 * ({@code -ngl}, {@code -ot}, {@code --n-cpu-ffn}), the routed experts of the named layers stay in RAM
 * ({@code --cpu-moe}, {@code --n-cpu-moe}, {@code -ot}, {@code --fit} per device) and each device's MoE expert cache
 * serves the experts of its own layers, or, under {@code neuralfree.moe.cache.pool}, the routed layers its free VRAM
 * earns, wherever their spans run ({@link com.eonmux.neuralfree.gguf.gpu.weights.MoeCachePool}): such a layer's
 * hosted MoE block runs on the GPU that caches it. Under {@code -ngl auto} the layout fits as on one GPU and host layers come
 * before streaming: when the requested spans do not stay resident, the routed experts of their leading layers move to
 * RAM and the first trunk layers move to the host, each GPU taking the largest run of trailing layers it holds
 * ({@link BackFill}); the layers left in front run on the host. Only when no host layout fits does the split stream.
 * The context is the largest every stage fits; the stage of the last layer
 * also fits the NextN draft caches and holds the NextN / MTP blocks, the main stage the drafter. The session then
 * allocates each stage's memory, uploads its resident weights, binds the split forward backend
 * ({@link GpuSplitForwardBackend}) with the embedding and the output head on the main device, and attaches the LoRA
 * adapters so that every stage, and the host layers, apply the pairs of their own layers
 * ({@link LoraAttachment#gpu(List, List)}, {@link LoraAttachment#hybrid}); the same attachment serves the
 * per-request adapters within each stage's slot reservation ({@link #slotReserve}); the
 * control vectors apply on the device of each layer. With {@code neuralfree.diag} every stage reports through
 * diagnostics that name its device. The session owns the acceleration it opened; {@link #close()} releases
 * everything in reverse order.
 */
public final class GpuSplitSession implements AutoCloseable {

    private static final Logger LOGGER = LogManager.getLogger(GpuSplitSession.class);

    private final EngineConfig config;
    private final ModelSpec spec;
    private final ModelSpec forwardSpec;
    private final MemoryPlan plan;
    private final HWAcceleration acceleration;
    private final List<SplitStage> stages;
    private final List<GpuForwardContext> contexts;
    private final LayerSpans spans;
    private final int main;
    private final int nextnStage;
    private final int hostLayers;
    private final GpuBlockRegistry registry;
    private final HeapModelMemory hostMemory;
    private final GpuSplitForwardBackend forward;
    private final LoraAttachment loras;
    private final long[] slotReserve;
    private final List<GGUFTensorInfo> tensors;
    private WeightPlacement placement;
    private boolean closed;

    GpuSplitSession(SplitAssembly a, HWAcceleration acceleration, GpuSplitForwardBackend forward,
                    LoraAttachment loras) {
        this.config = a.effective;
        this.spec = a.spec;
        this.forwardSpec = a.forwardSpec;
        this.plan = a.plan;
        this.acceleration = acceleration;
        this.stages = Collections.unmodifiableList(new ArrayList<>(a.stages));
        this.contexts = Collections.unmodifiableList(new ArrayList<>(a.contexts));
        this.spans = a.spans;
        this.main = a.main;
        this.nextnStage = a.nextnStage;
        this.hostLayers = a.hostLayers;
        this.registry = a.registry;
        this.hostMemory = a.hostMemory;
        this.placement = a.placement;
        this.forward = forward;
        this.loras = loras;
        this.slotReserve = a.slotReserve.clone();
        this.tensors = a.request.source().getTensors();
    }

    /**
     * Opens the split of the requested model over the devices its configuration selects.
     *
     * @return the session, or null when fewer than two devices are selected, so the caller loads on one device (the
     *         listed device of a single {@code neuralfree.gpu.devices} index, {@link EngineConfig#picksMainGpu()})
     * @throws IllegalArgumentException      for a configuration the split cannot lay out (a main GPU beyond the last
     *                                       device, more devices than device layers, a host placement that leaves no
     *                                       layer for the GPUs), naming the setting
     * @throws UnsupportedOperationException for a model shape the split cannot run (encoder-decoder, a looped
     *                                       schedule, a switched-LoRA graph, MUDD cross-layer connections), the MoE
     *                                       expert cache beside host layers, or an adapter pair no block serves
     * @throws GpuOutOfMemoryException       when the weights of a device do not fit even streamed, or the context
     *                                       every device must fit does not fit
     * @throws IOException                   when a model store cannot be mapped
     */
    public static GpuSplitSession open(SplitRequest request) throws IOException {
        Objects.requireNonNull(request, "request");
        EngineConfig cfg = request.cfg();
        ModelSpec initial = request.initial();
        refuseUnsupported(initial);
        HWAcceleration acceleration = HWAcceleration.forMode(cfg.backend().toAccelerationMode());
        List<GpuDevices.Selected> devices;
        try {
            devices = GpuDevices.select(acceleration, cfg);
        } catch (RuntimeException e) {
            acceleration.shutdown();
            throw e;
        }
        if (devices.isEmpty()) {
            acceleration.shutdown();
            LOGGER.info("{}: {}=layer selected fewer than two GPU devices; the model loads on one device{}{}",
                    initial.arch().ggufName(), EngineConfig.PROP_SPLIT_MODE,
                    cfg.picksMainGpu() ? ", the listed " + EngineConfig.PROP_GPU_DEVICES + " entry" : "",
                    cfg.moeCachePool() ? " and " + EngineConfig.PROP_MOE_CACHE_POOL + " cannot pool its expert"
                            + " cache" : "");
            return null;
        }
        SplitAssembly a = null;
        try {
            a = new SplitAssembly(request, devices);
            return a.assemble(acceleration);
        } catch (Throwable failure) {
            if (a != null) {
                a.release(failure);
            }
            GpuDevices.closeRemote(devices);
            try {
                acceleration.shutdown();
            } catch (RuntimeException shutdownFailure) {
                failure.addSuppressed(shutdownFailure);
            }
            throw failure;
        }
    }

    /** The model shapes the split refuses before anything is opened, naming the shape. */
    private static void refuseUnsupported(ModelSpec initial) {
        String arch = initial.arch().ggufName();
        String layer = EngineConfig.PROP_SPLIT_MODE + "=" + EngineConfig.SplitMode.LAYER.llamaName();
        if (GpuFactoryCatalog.usesSwitchedLora(initial)) {
            throw new UnsupportedOperationException(arch + " routes switched-LoRA adapters through one device's"
                    + " graph; " + layer + " does not split it");
        }
        if (GpuFactoryCatalog.usesMudd(initial)) {
            throw new UnsupportedOperationException(arch + " carries the MUDD dense cross-layer connections, whose"
                    + " weighted sums read earlier layers' inputs; " + layer + " keeps each device's layers over its"
                    + " own bank and cannot place those inputs across devices");
        }
        if (initial.supportClass() == SupportClass.ENC_DEC) {
            throw new UnsupportedOperationException(arch + " is an encoder-decoder model; the split cannot place"
                    + " its cross-attention");
        }
    }

    /** One stage as a line: its device, layers, resident and streamed weights and caches. */
    static String describe(SplitStage s, LayerSpans spans, long weightBytes) {
        int first = spans.start(s.index());
        StringBuilder sb = new StringBuilder("device ").append(s.name()).append(" holds layers ").append(first)
                .append("..").append(first + spans.spanLayers(s.index()) - 1).append(" (")
                .append(SplitAssembly.mib(weightBytes - s.streamedBytes())).append(" MiB of resident weights");
        StreamingLayerWeightSource streaming = s.streaming();
        if (streaming != null) {
            sb.append(", ").append(streaming.streamedLayers()).append(" layer(s) streamed through ")
                    .append(streaming.depth()).append(" slots (")
                    .append(SplitAssembly.mib(streaming.deviceBytes())).append(" MiB)");
        }
        sb.append(", ").append(s.memory() == null ? "0.0" : SplitAssembly.mib(s.memory().plan().totalBytes()))
                .append(" MiB of caches");
        MoeExpertCache cache = s.moeCache();
        if (cache != null) {
            sb.append(", MoE expert cache ").append(cache);
        }
        return sb.append(')').toString();
    }

    // ------------------------------------------------------------------ accessors

    /** The effective engine configuration (its memory settings carry the planned ubatch). */
    public EngineConfig config() {
        return config;
    }

    /** The model spec the forward pass runs. */
    public ModelSpec spec() {
        return spec;
    }

    /** The memory plan of the whole model. */
    public MemoryPlan plan() {
        return plan;
    }

    /** The backend of the main device, which holds the embedding and the output head. */
    public ComputeBackend backend() {
        return stages.get(main).backend();
    }

    /** The split forward backend. */
    public GpuSplitForwardBackend forwardBackend() {
        return forward;
    }

    /** The name of the main device, for messages. */
    public String mainDeviceName() {
        return stages.get(main).name();
    }

    /** Index of the main device among the stages. */
    public int mainIndex() {
        return main;
    }

    /** Number of stages, one per device, in pipeline order. */
    public int deviceCount() {
        return stages.size();
    }

    /** The name of the device of stage {@code i}. */
    public String deviceName(int i) {
        return stages.get(i).name();
    }

    /** The compute queue of stage {@code i}. */
    public ComputeQueue queue(int i) {
        return stages.get(i).queue();
    }

    /** The observer slot the forward context of stage {@code i} reports to. */
    public ObserverSlot observer(int i) {
        return stages.get(i).observer();
    }

    /** The streaming source of stage {@code i}, or null when its span is resident. */
    public StreamingLayerWeightSource streaming(int i) {
        return stages.get(i).streaming();
    }

    /** The streamed sets of stage {@code i}, empty when its span is resident (unmodifiable). */
    public List<LayerWeightSet> streamedSets(int i) {
        return stages.get(i).streamed();
    }

    /** The first layer of stage {@code i}'s span. */
    public int firstLayer(int i) {
        return spans.start(i);
    }

    /** The number of layers of stage {@code i}'s span. */
    public int layerCount(int i) {
        return spans.spanLayers(i);
    }

    /** Stage {@code i} as a line: device, layers, resident and streamed weights, caches. */
    public String describe(int i) {
        SplitStage s = stages.get(i);
        return describe(s, spans, s.uploader().residentBytes() + s.streamedBytes());
    }

    /** The LoRA attachment over every stage: the configured adapters, and the per-request ones it serves. */
    public LoraAttachment loras() {
        return loras;
    }

    /** The kernels of every stage, in stage order. */
    public List<LlmKernels> stageKernels() {
        List<LlmKernels> out = new ArrayList<>(stages.size());
        for (SplitStage s : stages) {
            out.add(s.kernels());
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Device bytes one per-request adapter slot may upload on each stage, the share of the reservation the fit held
     * back ({@code neuralfree.lora.enable}); zeros without it.
     */
    public long[] slotReserve() {
        return slotReserve.clone();
    }

    /** The largest scratch ubatch of the stages, the zero-slot rows a stage's device delta allocates. */
    public int slotUbatch() {
        int ubatch = 1;
        for (SplitStage s : stages) {
            ubatch = Math.max(ubatch, s.ops().scratch().ubatch());
        }
        return ubatch;
    }

    /** The first trunk layers that run on the host, {@code [0, hostLayers())}; 0 when every layer runs on a GPU. */
    public int hostLayers() {
        return hostLayers;
    }

    /** The MoE expert cache of stage {@code i}, or null when its layers keep their experts on the device. */
    public MoeExpertCache moeExpertCache(int i) {
        return stages.get(i).moeCache();
    }

    /**
     * The forward context of the main device, which holds the embedding, the output head and every model-level
     * tensor: the context the drafters that run beside the target bind to (eagle3, an MTP draft file).
     */
    public GpuForwardContext forwardContext() {
        return contexts.get(main);
    }

    /**
     * The forward context of the stage of the last trunk layer, which holds the model's NextN / MTP blocks and,
     * when it is not the main stage, the model-level tensors the MTP head reads beside them.
     */
    public GpuForwardContext nextnContext() {
        return contexts.get(nextnStage);
    }

    /** The block registry every stage was bound with. */
    public GpuBlockRegistry blockRegistry() {
        return registry;
    }

    /**
     * Places the NextN / MTP blocks on the stage of the last trunk layer for an MTP drafter over the model's own
     * blocks, with the model-level tensors the MTP head reads when that stage is not the main one; a no-op when they
     * are placed or the model has none. A split opened without MTP left them and their draft caches out of the
     * stage's fit, so they are uploaded now when the VRAM the stage has left holds them and their caches.
     *
     * @throws GpuOutOfMemoryException when they do not fit; the model then has to be opened with MTP configured
     */
    public void ensureNextnWeights() {
        WeightPlacement current = placement;
        if (!current.hasNextn() || current.nextnOnDevice()) {
            return;
        }
        WeightPlacement next = current.withNextnOnDevice();
        SplitStage s = stages.get(nextnStage);
        Set<String> added = new HashSet<>();
        for (GGUFTensorInfo t : tensors) {
            if (next.onDevice(t) && owns(nextnStage, t.name(), next) && s.uploader().find(t.name()) == null) {
                added.add(t.name());
            }
        }
        Predicate<GGUFTensorInfo> adding = t -> added.contains(t.name());
        long bytes = s.uploader().plannedBytes(adding);
        long caches = plan.kind() == MemoryKind.NONE || spec.nLayer() == 0 ? 0L
                : MemoryPlanner.nextnBytes(spec, config.memory(), plan.nCtx(), plan.ubatch());
        long needed = Math.addExact(bytes, caches);
        long usable = s.usableBytes(0L);
        if (needed > usable) {
            throw new GpuOutOfMemoryException(spec.arch().ggufName() + ": the " + spec.nextnLayers().size()
                    + " NextN block(s) were left off device " + s.name() + " because MTP was not configured when the"
                    + " model was opened, and their " + SplitAssembly.mib(bytes) + " MiB of weights and "
                    + SplitAssembly.mib(caches) + " MiB of draft caches do not fit the "
                    + SplitAssembly.mib(Math.max(0L, usable)) + " MiB of usable VRAM left; open the model with "
                    + EngineConfig.PROP_MTP + "=true so its context is fitted beside them", needed, usable,
                    s.backend().backendType().name());
        }
        s.uploader().uploadResident(adding.negate());
        WeightKernelPrewarm.prewarm(s.uploader().plans(adding), s.ops(), s.kernels());
        placement = next;
        LOGGER.info("{}: {} NextN block(s) placed on device {} for MTP drafting ({} MiB)", spec.arch().ggufName(),
                spec.nextnLayers().size(), s.name(), SplitAssembly.mib(bytes));
    }

    /** {@link SplitAssembly#owns} over the session's spans. */
    private boolean owns(int i, String name, WeightPlacement placed) {
        return SplitAssembly.owns(spans, main, nextnStage, i, name, placed);
    }

    /** The model spec the forward blocks run (with the control-vector extras when vectors are applied). */
    public ModelSpec forwardSpec() {
        return forwardSpec;
    }

    // ------------------------------------------------------------------ lifecycle

    /**
     * Releases everything in reverse order of acquisition: the adapters' device uploads, the forward backend (span
     * executors, host layers and head), the host layers' memory, every stage (MoE expert cache, streaming, memory,
     * dispatch helpers, scratch, placeholders, uploaded weights and stores), then every backend through the
     * acceleration's shutdown. Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = null;
        if (loras != null && loras.deltaCount() > 0) {
            for (SplitStage s : stages) {
                try {
                    s.queue().finish();
                } catch (RuntimeException e) {
                    first = Closeables.chain(first, e);
                }
            }
        }
        first = Closeables.close(loras, first);
        first = Closeables.close(forward, first);
        first = Closeables.close(hostMemory, first);
        for (int i = stages.size() - 1; i >= 0; i--) {
            first = Closeables.close(stages.get(i), first);
        }
        for (int i = stages.size() - 1; i >= 0; i--) {
            try {
                stages.get(i).closeDevice();
            } catch (RuntimeException e) {
                first = Closeables.chain(first, e);
            }
        }
        try {
            acceleration.shutdown();
        } catch (RuntimeException e) {
            if (first == null) {
                first = e;
            } else {
                first.addSuppressed(e);
            }
        }
        if (first != null) {
            throw first;
        }
    }
}
