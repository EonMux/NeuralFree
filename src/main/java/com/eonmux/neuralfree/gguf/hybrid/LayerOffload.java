package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.TensorOverride;
import com.eonmux.neuralfree.gguf.gpu.weights.TensorPlacementOverrides;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecValidator;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The resolved CPU+GPU layer split of one GPU load (llama.cpp {@code -ngl}, common/arg.cpp:2750, with the
 * per-layer device assignment of src/llama-model.cpp:1609-1700): the last {@link #nGpuLayers()} trunk layers compute
 * on the device, the earlier ones on the host, each side with its KV cache on its own device. Model-level tensors
 * (the embedding, the output head and the norms) and the NextN / MTP blocks always stay on the device; the output
 * head is what the llama.cpp count covers with its extra layer, so a request of {@code N} keeps {@code N - 1} trunk
 * layers on the device.
 *
 * <p>Tensor overrides ({@link EngineConfig#tensorOverrides()}, llama.cpp {@code -ot}) keep matched tensors on the
 * side they name. A host match on a routed expert tensor keeps the tensor in RAM while its layer computes on the
 * device and runs its experts on the host (the hybrid expert path). Any other host match keeps the tensor's whole
 * layer on the host, because a layer of this engine computes as one block on one side, and the split grows until
 * every host-named layer is on the host: the host layers are the prefix {@code [0, firstGpuLayer)}, so a host-named
 * layer also brings its earlier layers to the host. The dense FFN weights {@code neuralfree.n-cpu-ffn} keeps on the
 * host (llama.cpp {@code --n-cpu-ffn}) are host matches of the same kind: their layers run whole on the host. The
 * expert weights the MoE settings derive
 * ({@code neuralfree.cpu.moe} / {@code neuralfree.n.cpu.moe}, llama.cpp {@code --cpu-moe} / {@code --n-cpu-moe})
 * keep their layer on the device and run its experts on the host through the hybrid expert path
 * ({@link #hostExperts(int)}); the layers that path cannot serve keep the CPU MoE block of the layer
 * ({@code GGUFModelGPU}). Immutable.</p>
 *
 * <p>With {@code neuralfree.kv.offload=false} (llama.cpp {@code -nkvo}) every device layer keeps its KV cache in host
 * RAM instead: its projections, FFN and cache writes run on the device, its attention over the host cache on the
 * host ({@link #hostKvLayer(int)}). The host layers of a layer split keep their whole layer, and so their KV, on the
 * host either way. The device memory of such a load allocates no KV cache of a host-KV layer
 * ({@link #deviceLayer(int)}), so the context fit charges only the caches that stay on the device.</p>
 */
public final class LayerOffload {

    /** Expert-weight tensors, the llama.cpp {@code LLM_FFN_EXPS_REGEX} (common/common.h:1257). */
    public static final String EXPERT_TENSORS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

    /** Per-block expert override of {@code neuralfree.n.cpu.moe} (common/common.h:1267-1273). */
    private static final String BLOCK_EXPERT_FORMAT = "blk\\.%d" + EXPERT_TENSORS_REGEX;

    /** Per-block dense FFN rule of {@code neuralfree.n-cpu-ffn} ({@code LLM_FFN_DENSE_REGEX}, common/common.h:1259). */
    private static final String BLOCK_DENSE_FFN_FORMAT = "blk\\.%d" + TensorPlacementOverrides.DENSE_FFN_REGEX;

    /** Compiled {@link #EXPERT_TENSORS_REGEX}, over stored tensor names with {@code find()} semantics. */
    private static final java.util.regex.Pattern EXPERT_TENSOR = java.util.regex.Pattern.compile(EXPERT_TENSORS_REGEX);

    private final ModelSpec spec;
    private final int firstGpuLayer;
    private final List<TensorOverride> overrides;
    private final List<TensorOverride> hostExpertOverrides;
    private final int hostExpertLimit;
    /** True when the load keeps the KV caches of its device layers in host RAM ({@code -nkvo}). */
    private final boolean hostKv;

    private LayerOffload(ModelSpec spec, int firstGpuLayer, List<TensorOverride> overrides,
                         List<TensorOverride> hostExpertOverrides, int hostExpertLimit, boolean hostKv) {
        this.spec = spec;
        this.firstGpuLayer = firstGpuLayer;
        this.overrides = List.copyOf(overrides);
        this.hostExpertOverrides = List.copyOf(hostExpertOverrides);
        this.hostExpertLimit = hostExpertLimit;
        this.hostKv = hostKv;
    }

    /**
     * The offload the configuration asks for: an exact {@code neuralfree.n.gpu.layers} split, or, for
     * {@code auto} / {@code all}, the split the tensor overrides and the MoE expert settings force. Null when the
     * load keeps every layer on the device and no setting keeps tensors or KV caches on the host; a host-KV load
     * ({@code neuralfree.kv.offload=false}) resolves a split even with every layer on the device.
     *
     * @param tensors the stored tensors, so a host override that names a non-trunk tensor (the embedding, the output
     *                head, the NextN blocks) is refused before anything is placed
     * @throws IllegalArgumentException for a layer count of 0 (nothing on the device) or one above the layers plus
     *                                  the output head
     */
    public static LayerOffload requested(ModelSpec spec, EngineConfig cfg, List<GGUFTensorInfo> tensors) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(tensors, "tensors");
        int requested = cfg.gpuLayers();
        if (requested == EngineConfig.GPU_LAYERS_AUTO || requested == EngineConfig.GPU_LAYERS_ALL) {
            return auto(spec, cfg, tensors, spec.nLayer());
        }
        if (requested == 0) {
            throw new IllegalArgumentException("a layer count of 0 (" + EngineConfig.PROP_N_GPU_LAYERS + ", or "
                    + EngineConfig.PROP_DRAFT_N_GPU_LAYERS + " for a draft model) keeps no layer on the device;"
                    + " load the model with " + EngineConfig.PROP_BACKEND + "=cpu, or keep a draft model off the"
                    + " GPUs with " + EngineConfig.PROP_DRAFT_GPU_DEVICES + "=none");
        }
        int nLayer = spec.nLayer();
        if (requested > nLayer + 1) {
            throw new IllegalArgumentException("the layer count " + requested + " (" + EngineConfig.PROP_N_GPU_LAYERS
                    + ", or " + EngineConfig.PROP_DRAFT_N_GPU_LAYERS + " for a draft model) exceeds the " + nLayer
                    + " trunk layers plus the output head of " + spec.arch().ggufName());
        }
        return build(spec, cfg, tensors, requested - 1);
    }

    /**
     * The offload that keeps at most {@code nGpuLayers} trunk layers on the device (the {@code -ngl auto} search
     * runs it once per candidate); a host tensor override grows the host part beyond the count. Returns null when
     * the result keeps every layer and every expert on the device and every KV cache there too.
     *
     * @throws IllegalArgumentException for {@code nGpuLayers} outside {@code [0, nLayer]}
     */
    public static LayerOffload auto(ModelSpec spec, EngineConfig cfg, List<GGUFTensorInfo> tensors, int nGpuLayers) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(cfg, "cfg");
        int nLayer = spec.nLayer();
        if (nGpuLayers < 0 || nGpuLayers > nLayer) {
            throw new IllegalArgumentException("nGpuLayers " + nGpuLayers + " outside [0, " + nLayer + "]");
        }
        return build(spec, cfg, tensors, nGpuLayers);
    }

    private static LayerOffload build(ModelSpec spec, EngineConfig cfg, List<GGUFTensorInfo> tensors,
                                      int nGpuLayers) {
        int nLayer = spec.nLayer();
        int firstGpu = nLayer - nGpuLayers;
        // the host layers are a prefix: a host match on a device layer pulls it and every earlier layer to the host,
        // so the split keeps one host stack and one device stack
        List<TensorOverride> hostRules = hostLayerRules(spec, cfg);
        firstGpu = Math.max(firstGpu, highestHostLayer(spec, hostRules, tensors) + 1);
        List<TensorOverride> hostExperts = hostExpertOverrides(spec, cfg);
        int expertLimit = hostExperts.isEmpty() ? 0 : hostExperts.size();
        LayerOffload offload = new LayerOffload(spec, firstGpu, hostRules, hostExperts, expertLimit,
                hostKvRequested(spec, cfg));
        if (!offload.active() && hostExperts.isEmpty() && !offload.hostKv()) {
            return null;
        }
        offload.validate(cfg);
        return offload;
    }

    /**
     * Whether this load keeps its KV caches in host RAM: {@code neuralfree.kv.offload=false} on a model that plans
     * KV caches. A model without memory has no KV to move, so the setting leaves it on its normal placement.
     */
    public static boolean hostKvRequested(ModelSpec spec, EngineConfig cfg) {
        if (cfg.kvOffload() || spec.memoryKind() == MemoryKind.NONE) {
            return false;
        }
        for (LayerSpec l : spec.layers()) {
            if (l.usesKv()) {
                return true;
            }
        }
        return false;
    }

    /**
     * The tensor rules whose host matches pull their layer to the host: the user's overrides in order, then the
     * per-layer dense FFN rules of {@code neuralfree.n-cpu-ffn} (llama.cpp {@code LLM_FFN_DENSE_REGEX} over the
     * first N blocks, common/common.h:1259-1273).
     */
    private static List<TensorOverride> hostLayerRules(ModelSpec spec, EngineConfig cfg) {
        List<TensorOverride> out = new ArrayList<>(cfg.tensorOverrides());
        int limit = Math.min(cfg.nCpuFfn(), spec.nLayer());
        for (int il = 0; il < limit; il++) {
            out.add(TensorOverride.parse(String.format(BLOCK_DENSE_FFN_FORMAT, il), "cpu"));
        }
        return out;
    }

    /**
     * Highest trunk layer any host rule names, or -1 when none does. Device-target overrides place nothing on the
     * host and are ignored here. A host match on a routed expert tensor is ignored too: the expert set stays in RAM
     * beside its device layer and computes there (the hybrid expert path), so it does not pull its layer to the
     * host.
     */
    private static int highestHostLayer(ModelSpec spec, List<TensorOverride> overrides,
                                        List<GGUFTensorInfo> tensors) {
        int highest = -1;
        for (TensorOverride o : overrides) {
            if (!o.host()) {
                continue;
            }
            for (GGUFTensorInfo t : tensors) {
                String name = t.name();
                if (!o.matches(name) || EXPERT_TENSOR.matcher(name).find()) {
                    continue;
                }
                if (!ModelSpecValidator.isTrunkTensor(name, spec.nLayer())) {
                    throw new UnsupportedOperationException("tensor override " + o + " matches '" + name
                            + "', which is not a trunk layer tensor and cannot be kept on the host");
                }
                highest = Math.max(highest, blockIndex(name));
            }
        }
        return highest;
    }

    /** The expert-weight overrides the MoE settings derive (the user's {@code -ot} entries take precedence). */
    private static List<TensorOverride> hostExpertOverrides(ModelSpec spec, EngineConfig cfg) {
        List<TensorOverride> out = new ArrayList<>();
        if (!cfg.cpuMoe() && cfg.nCpuMoe() <= 0) {
            return out;
        }
        int limit = cfg.cpuMoe() ? spec.nLayer() : Math.min(cfg.nCpuMoe(), spec.nLayer());
        for (int il = 0; il < limit; il++) {
            out.add(TensorOverride.parse(String.format(BLOCK_EXPERT_FORMAT, il), "cpu"));
        }
        return out;
    }

    /**
     * Checks the split against the model and the rest of the configuration.
     *
     * @throws UnsupportedOperationException when the model's kind or schedule cannot split, or a configured feature
     *                                       does not run over a split
     * @throws IllegalArgumentException      when the residual stream changes width at the split
     */
    private void validate(EngineConfig cfg) {
        String arch = spec.arch().ggufName();
        if (spec.supportClass() == SupportClass.ENC_DEC) {
            throw new UnsupportedOperationException(arch + " is an encoder-decoder model; its encoder and decoder"
                    + " stacks cannot split across the host and the device");
        }
        if (spec.memoryKind() == MemoryKind.NONE) {
            throw new UnsupportedOperationException(arch + " plans no memory; there is no layer split to make");
        }
        if (spec.arch() == Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException(arch + " decodes audio codes and has no token forward pass to"
                    + " split across the host and the device");
        }
        if (spec.schedule().kind() != LayerSchedule.Kind.LINEAR) {
            throw new UnsupportedOperationException(arch + " runs a " + spec.schedule().kind() + " schedule; only"
                    + " a linear schedule can split across the host and the device");
        }
        if (cfg.memory().paged()) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_PARALLEL + " > 1 or "
                    + MemoryConfig.PROP_KV_PAGED + " (paged KV memory) does not run over a hybrid CPU+GPU layer"
                    + " split");
        }
        if (hostKv) {
            validateHostKv(cfg);
        }
        if (active() && nGpuLayers() > 0) {
            checkBoundaryWidth();
        }
    }

    /** The checks of a host-KV load: one GPU's contiguous caches, plain softmax attention and no NextN drafting. */
    private void validateHostKv(EngineConfig cfg) {
        String arch = spec.arch().ggufName();
        if (cfg.split()) {
            throw new UnsupportedOperationException(EngineConfig.PROP_KV_OFFLOAD + "=false cannot run over "
                    + EngineConfig.PROP_SPLIT_MODE + "=" + EngineConfig.SplitMode.LAYER.llamaName()
                    + ", which spreads the layers over several devices; use one device, or keep the KV caches on"
                    + " the device");
        }
        if (cfg.mtp()) {
            throw new UnsupportedOperationException(EngineConfig.PROP_KV_OFFLOAD + "=false cannot serve the NextN /"
                    + " MTP blocks (" + EngineConfig.PROP_MTP + "), whose draft passes write their own KV rows;"
                    + " clear " + EngineConfig.PROP_MTP + ", or keep the KV caches on the device");
        }
        for (LayerSpec l : spec.layers()) {
            if (!l.usesKv()) {
                continue;
            }
            if (l.mixerKind() != MixerKind.ATTENTION || l.parallelAttention()) {
                throw new UnsupportedOperationException(arch + ": " + EngineConfig.PROP_KV_OFFLOAD + "=false can"
                        + " keep the KV cache of a softmax-attention layer on the host, layer " + l.il() + " runs a "
                        + l.mixerKind() + (l.parallelAttention() ? " with a parallel attention branch" : "")
                        + " cache instead");
            }
            if (l.attention() == null) {
                throw new UnsupportedOperationException(arch + ": " + EngineConfig.PROP_KV_OFFLOAD + "=false cannot"
                        + " keep the KV cache of layer " + l.il() + " on the host, which has no attention spec");
            }
            if (l.attention().biasMode() == AttentionSpec.BiasMode.T5_CAUSAL
                    || l.attention().biasMode() == AttentionSpec.BiasMode.T5_BIDIR) {
                throw new UnsupportedOperationException(arch + ": " + EngineConfig.PROP_KV_OFFLOAD + "=false cannot"
                        + " run the relative-position bias cache of layer " + l.il() + " on the host");
            }
        }
    }

    /** The residual stream must leave the host stack as wide as the device stack reads it. */
    private void checkBoundaryWidth() {
        int first = firstGpuLayer;
        int hostWidth = spec.layer(first - 1).residual().streamWidth();
        int deviceWidth = spec.layer(first).residual().streamWidth();
        if (hostWidth != deviceWidth) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": the residual stream changes width at the"
                    + " split (" + hostWidth + " -> " + deviceWidth + " between layers " + (first - 1) + " and "
                    + first + "); split at a layer boundary of one width");
        }
    }

    /** True when at least one trunk layer computes on the host. */
    public boolean active() {
        return firstGpuLayer > 0;
    }

    /** Trunk layers computing on the device: {@code [firstGpuLayer(), nLayer)}. */
    public int nGpuLayers() {
        return spec.nLayer() - firstGpuLayer;
    }

    /** First trunk layer on the device. */
    public int firstGpuLayer() {
        return firstGpuLayer;
    }

    /** True when trunk layer {@code il} computes on the host. */
    public boolean hostLayer(int il) {
        checkLayer(il);
        return il < firstGpuLayer;
    }

    /** True when this load keeps the KV caches in host RAM ({@code neuralfree.kv.offload=false}). */
    public boolean hostKv() {
        return hostKv;
    }

    /**
     * True when trunk layer {@code il} computes on the device but keeps its KV cache in host RAM and attends over
     * it on the host ({@code neuralfree.kv.offload=false}).
     */
    public boolean hostKvLayer(int il) {
        return hostKv && isTrunkLayer(il) && il >= firstGpuLayer && spec.layer(il).usesKv();
    }

    /**
     * True when trunk layer {@code il}'s caches live in the host memory: a whole host layer, or a device layer whose
     * KV is kept on the host.
     */
    public boolean hostMemoryLayer(int il) {
        return (isTrunkLayer(il) && il < firstGpuLayer) || hostKvLayer(il);
    }

    /**
     * True when the device memory allocates trunk layer {@code il}'s caches: a device layer whose KV stays on the
     * device. The rest of a host-KV device layer computes on the device with no cache there.
     */
    public boolean deviceLayer(int il) {
        return isTrunkLayer(il) && il >= firstGpuLayer && !hostKvLayer(il);
    }

    /**
     * True when a stored tensor is kept on the host: a tensor of a host layer, a tensor a host override names, or
     * an expert tensor of the MoE settings. Expert tensors of a device layer are host tensors of a layer that still
     * computes on the device ({@link #hostExperts(int)}); every other host tensor belongs to a host layer.
     */
    public boolean hostTensor(String storedName) {
        Objects.requireNonNull(storedName, "storedName");
        int il = blockIndex(storedName);
        if (il >= 0 && il < spec.nLayer() && hostLayer(il)) {
            return true;
        }
        for (TensorOverride o : overrides) {
            if (o.host() && o.matches(storedName)) {
                return true;
            }
        }
        for (TensorOverride o : hostExpertOverrides) {
            if (o.matches(storedName)) {
                return true;
            }
        }
        return false;
    }

    /**
     * True when the trunk layer keeps its expert weights on the host while it computes on the device; its FFN then
     * runs the experts on the host. Only a device layer with experts can.
     */
    public boolean hostExperts(int il) {
        checkLayer(il);
        return il >= firstGpuLayer && il < hostExpertLimit && spec.layer(il).moe() != null;
    }

    /** The derived expert overrides, for diagnostics. */
    public List<TensorOverride> hostExpertOverrides() {
        return hostExpertOverrides;
    }

    private void checkLayer(int il) {
        if (!isTrunkLayer(il)) {
            throw new IndexOutOfBoundsException("trunk layer " + il + " outside [0, " + spec.nLayer() + ")");
        }
    }

    private boolean isTrunkLayer(int il) {
        return il >= 0 && il < spec.nLayer();
    }

    /** Trunk block index of a {@code blk.N.*} or {@code model.layers.N.*} name, or -1 for another form. */
    private static int blockIndex(String storedName) {
        for (String prefix : new String[]{TensorNames.STACK_BLK + ".", TensorNames.HF_LAYER_PREFIX + "."}) {
            if (!storedName.startsWith(prefix)) {
                continue;
            }
            int dot = storedName.indexOf('.', prefix.length());
            if (dot <= prefix.length() || dot == storedName.length() - 1) {
                continue;
            }
            try {
                int il = Integer.parseInt(storedName.substring(prefix.length(), dot));
                if (il >= 0) {
                    return il;
                }
            } catch (NumberFormatException ignored) {
                return -1;
            }
        }
        return -1;
    }

    @Override
    public String toString() {
        return "LayerOffload{" + spec.arch().ggufName() + ", layers [0, " + firstGpuLayer + ") on the host, ["
                + firstGpuLayer + ", " + spec.nLayer() + ") on the device"
                + (hostExpertLimit > 0 ? ", experts of layers [0, " + hostExpertLimit + ") on the host" : "")
                + (hostKv ? ", KV caches on the host" : "")
                + '}';
    }
}
