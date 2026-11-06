package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LoraSpec;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;
import com.eonmux.neuralfree.gguf.ops.heads.GpuLoraDeltas;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapterLoader;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * The adapters of a {@link GGUFModelGPU} load: the configured control vectors ({@link EngineConfig#controlVectors()},
 * llama.cpp {@code --control-vector} / {@code --control-vector-scaled} with {@code --control-vector-layer-range}) and
 * LoRA adapters ({@link EngineConfig#loras()}, llama.cpp {@code --lora} / {@code --lora-scaled}). Both are loaded and
 * validated before placement, so the device bytes they allocate later ({@link #deviceBytes()}) are held back from the
 * VRAM budget of the context and streaming fits; the LoRA pairs attach after the forward blocks are bound
 * ({@link #attachLoras}), and the same attachment serves the per-request adapters ({@link #perRequest}). With
 * {@code neuralfree.lora.enable} the load also holds back the per-request adapter slots ({@link #slotBytes()}, vLLM's
 * {@code max_loras} x {@code max_lora_rank} reservation): {@code neuralfree.lora.max} slots of one worst-case adapter
 * of rank {@code neuralfree.lora.max.rank} over every adaptable product on the device. Immutable.
 */
final class GpuAdapters {

    private static final Logger LOGGER = LogManager.getLogger(GpuAdapters.class);
    private static final double MIB = 1024.0 * 1024.0;

    /** Host access to the model's tensors, opened on first use ({@link GGUFModelGPU#cpuWeights()}). */
    interface HostWeights {
        CpuWeights get() throws IOException;
    }

    private final ControlVector controlVectors;
    private final List<LoraAttachment.Binding> loras;
    private final long deviceBytes;
    private final long slotBytes;

    private GpuAdapters(ControlVector controlVectors, List<LoraAttachment.Binding> loras, long deviceBytes,
                        long slotBytes) {
        this.controlVectors = controlVectors;
        this.loras = loras;
        this.deviceBytes = deviceBytes;
        this.slotBytes = slotBytes;
    }

    /**
     * Loads the configured control vectors and LoRA adapters of {@code cfg} against {@code initial}; the adapters are
     * validated against the base tensors {@code host} opens.
     *
     * @throws UnsupportedOperationException for the WavTokenizer decoder or a refused adapter
     * @throws IOException                   when a vector or adapter file cannot be read
     */
    static GpuAdapters load(String modelName, ModelSpec initial, EngineConfig cfg, MemoryConfig mem,
                            HostWeights host) throws IOException {
        return load(modelName, initial, cfg, mem, readLoras(initial, cfg, host), Collections.<long[]>emptyList());
    }

    /**
     * Loads the configured control vectors of {@code cfg} against {@code initial} beside the LoRA adapters
     * {@link #readLoras} already read, so a load that reads the adapters before choosing between one device and a
     * layer split reads them once. {@code slotShapes} are the adaptable products on the device
     * ({@link LoraAttachment#adaptableShapes}) the per-request slots reserve over.
     *
     * @throws UnsupportedOperationException for the WavTokenizer decoder
     * @throws IOException                   when a vector file cannot be read
     */
    static GpuAdapters load(String modelName, ModelSpec initial, EngineConfig cfg, MemoryConfig mem,
                            List<LoraAttachment.Binding> bindings, List<long[]> slotShapes) throws IOException {
        Objects.requireNonNull(bindings, "bindings");
        ControlVector vectors = readControlVectors(modelName, initial, cfg);
        long vectorBytes = vectors == null ? 0L : controlVectorBytes(initial);
        long slots = reservedSlotBytes(modelName, initial, cfg, mem.ubatch(), slotShapes);
        if (bindings.isEmpty()) {
            return new GpuAdapters(vectors, bindings, vectorBytes, slots);
        }
        long loraBytes = LoraAttachment.gpuDeviceBytes(bindings, mem.ubatch());
        LOGGER.info("{}: {} LoRA adapter(s) loaded, {} MiB of device uploads reserved", modelName, bindings.size(),
                mib(loraBytes));
        return new GpuAdapters(vectors, bindings, loraBytes, slots);
    }

    /**
     * Device bytes of the per-request adapter slots a load reserves: {@code neuralfree.lora.max} worst-case slots
     * over {@code shapes} ({@link LoraAttachment#slotReserveBytes}) when {@code neuralfree.lora.enable} is on and
     * the model runs tokens, else 0.
     */
    static long reservedSlotBytes(String modelName, ModelSpec initial, EngineConfig cfg, int ubatch,
                                  List<long[]> shapes) {
        LoraConfig lc = cfg.loraConfig();
        if (!lc.enabled() || initial.arch() == Arch.WAVTOKENIZER_DEC || shapes.isEmpty()) {
            return 0L;
        }
        long perSlot = LoraAttachment.slotReserveBytes(shapes, lc.maxRank(), ubatch);
        long total = Math.multiplyExact(perSlot, (long) lc.maxAdapters());
        LOGGER.info("{}: {} per-request LoRA slot(s) of rank {} over {} products, {} MiB of device memory reserved"
                + " ({})", modelName, lc.maxAdapters(), lc.maxRank(), shapes.size(), mib(total),
                LoraConfig.PROP_ENABLE);
        return total;
    }

    /**
     * The control vectors of {@code cfg}, or null when none is configured; their device buffers (one {@code n_embd}
     * row per trunk layer, on the device that runs the layer) are written by the trunk extras.
     *
     * @throws UnsupportedOperationException for the WavTokenizer decoder
     * @throws IOException                   when a vector file cannot be read
     */
    static ControlVector readControlVectors(String modelName, ModelSpec initial, EngineConfig cfg)
            throws IOException {
        if (cfg.controlVectors().isEmpty()) {
            return null;
        }
        if (initial.arch() == Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " has no token forward pass to apply"
                    + " control vectors to");
        }
        ControlVector vectors = ControlVector.load(initial, cfg.controlVectors(), cfg.controlVectorLayerStart(),
                cfg.controlVectorLayerEnd());
        LOGGER.info("{}: {} control vector(s) applied over layers {}..{} ({} KiB of device buffers): {}",
                modelName, vectors.list().size(), vectors.list().get(0).layerStart(),
                vectors.list().get(0).layerEnd(), controlVectorBytes(initial) / 1024, vectors.list());
        return vectors;
    }

    /** Device bytes of the control-vector buffers: one {@code n_embd} float row per trunk layer. */
    private static long controlVectorBytes(ModelSpec initial) {
        return (long) initial.nLayer() * initial.hparams().nEmbd() * Float.BYTES;
    }

    /**
     * The LoRA adapters of {@code cfg}, each loaded and validated against the base tensors {@code host} opens; empty
     * when none is configured, in which case {@code host} is not opened.
     *
     * @throws UnsupportedOperationException for the WavTokenizer decoder or a refused adapter
     * @throws IOException                   when an adapter file cannot be read
     */
    static List<LoraAttachment.Binding> readLoras(ModelSpec initial, EngineConfig cfg, HostWeights host)
            throws IOException {
        List<LoraSpec> specs = cfg.loras();
        if (specs.isEmpty()) {
            return Collections.emptyList();
        }
        if (initial.arch() == Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException(initial.arch().ggufName() + " has no token forward pass to apply"
                    + " LoRA adapters to");
        }
        List<LoraAttachment.Binding> bindings = new ArrayList<>(specs.size());
        for (LoraSpec s : specs) {
            bindings.add(LoraAttachment.Binding.load(s.path(), s.scale(), host.get(), cfg.sourceOptions()));
        }
        return Collections.unmodifiableList(bindings);
    }

    /** The configured control vectors, or null when none is configured. */
    ControlVector controlVectors() {
        return controlVectors;
    }

    /** True when LoRA adapters are loaded, so {@link #attachLoras} attaches them. */
    boolean hasLoras() {
        return !loras.isEmpty();
    }

    /**
     * Device bytes the adapters allocate after placement, held back from the VRAM budget until they are allocated:
     * the LoRA uploads when adapters are loaded, else the control-vector buffers.
     */
    long deviceBytes() {
        return deviceBytes;
    }

    /**
     * Device bytes held back for the per-request adapter slots for the model's lifetime: {@code neuralfree.lora.max}
     * times one slot's share, 0 without {@code neuralfree.lora.enable}.
     */
    long slotBytes() {
        return slotBytes;
    }

    /**
     * Attaches the loaded LoRA adapters to the bound forward blocks: their pairs are uploaded and registered in the
     * context's projection deltas, which the blocks add after their products to every row (llama-graph.cpp:1629-1749,
     * 2694-2709). The attachment holds no pair when no adapter is configured; it serves the per-request adapters of
     * {@link #perRequest} either way.
     *
     * @throws UnsupportedOperationException for a refused adapted product
     * @throws IllegalStateException         when an adapter tensor targets a product the model does not run
     */
    LoraAttachment attachLoras(String modelName, LlmKernels kernels, GpuOps ops, GpuProjectionDeltas deltas) {
        if (loras.isEmpty()) {
            return LoraAttachment.gpu(loras, ops, deltas);
        }
        kernels.prewarm(GpuLoraDeltas.KERNELS);
        LoraAttachment attached = LoraAttachment.gpu(loras, ops, deltas);
        LOGGER.info("{}: {} LoRA adapter(s) attached, {} deltas: {}", modelName, loras.size(),
                attached.deltaCount(), loras);
        return attached;
    }

    /**
     * Attaches the loaded LoRA adapters to a hybrid CPU+GPU layer split: each pair registers on the device deltas
     * when a device block serves its base tensor, else on the host deltas of the layers kept on the CPU
     * ({@link LoraAttachment#hybrid}). The attachment serves the per-request adapters of {@link #perRequest} either
     * way.
     *
     * @throws UnsupportedOperationException for a refused adapted product
     * @throws IllegalStateException         when an adapter tensor targets a product the model does not run
     */
    LoraAttachment attachHybridLoras(String modelName, LlmKernels kernels, GpuOps ops, GpuProjectionDeltas deltas,
                                     CpuProjectionDeltas host) {
        kernels.prewarm(GpuLoraDeltas.KERNELS);
        LoraAttachment attached = LoraAttachment.hybrid(loras,
                Collections.singletonList(new LoraAttachment.GpuDevice(ops, deltas)), host);
        if (!loras.isEmpty()) {
            LOGGER.info("{}: {} LoRA adapter(s) attached over the device and host layers, {} deltas: {}", modelName,
                    loras.size(), attached.deltaCount(), loras);
        }
        return attached;
    }

    /**
     * The registry of the model's per-request adapters, served by {@code attached}: a load validates the adapter
     * against the base tensors {@code host} opens and prewarms the delta kernels on every device; the slots it opens
     * upload their pairs on demand, each within its share of the reservation, {@code perSlotBytes[d]} on device
     * {@code d} with zero slots of {@code ubatch} rows. Without {@code neuralfree.lora.enable} nothing is reserved: a
     * load is refused by name, and a slot that would upload (a startup adapter at another scale) too.
     */
    static LoraRegistry perRequest(LoraAttachment attached, List<LlmKernels> kernels, EngineConfig cfg,
                                   HostWeights host, long[] perSlotBytes, int ubatch) {
        LoraConfig lc = cfg.loraConfig();
        LoraRegistry registry = LoraRegistry.of(path -> {
            if (!lc.enabled()) {
                throw new UnsupportedOperationException("per-request LoRA adapters on the GPU need "
                        + LoraConfig.PROP_ENABLE + "=true (vLLM --enable-lora), which reserves the device memory of"
                        + " their slots when the model loads; cannot load " + path);
            }
            for (LlmKernels k : kernels) {
                k.prewarm(GpuLoraDeltas.KERNELS);
            }
            return LoraAdapterLoader.load(path, host.get(), cfg.sourceOptions());
        }, lc);
        attached.reserveSlots(lc.enabled() ? perSlotBytes : new long[perSlotBytes.length], ubatch,
                lc.enabled() ? LoraConfig.PROP_ENABLE + ": " + lc.maxAdapters() + " slot(s) of rank " + lc.maxRank()
                        : LoraConfig.PROP_ENABLE + " is off, so no slot memory is reserved");
        attached.serve(registry);
        return registry;
    }

    /**
     * The share of one per-request slot in {@code totalBytes}, the reservation of {@code neuralfree.lora.max} slots.
     */
    static long perSlot(EngineConfig cfg, long totalBytes) {
        return totalBytes / cfg.loraConfig().maxAdapters();
    }

    /** Mebibytes with one decimal. */
    private static String mib(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }
}
