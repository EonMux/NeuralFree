package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.Layer;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.LayerKind;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.Selection;
import com.eonmux.neuralfree.gguf.hf.config.VllmWeightsMapper;
import com.eonmux.neuralfree.gguf.hf.tensor.DtypePolicy;
import com.eonmux.neuralfree.gguf.hf.tensor.LoraFactorSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.FusedProjections;
import com.eonmux.neuralfree.gguf.quant.OnlineQuantizers;
import com.eonmux.neuralfree.gguf.quant.RequantPolicy;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

/**
 * The online requantization of one planning pass ({@link HfSourceAssembler#requant}): llama-quantize's block types
 * through {@link RequantPolicy#replace}, and vLLM's online methods through the load's {@link OnlineQuantConfig},
 * which chooses each layer's method from the checkpoint module the tensor comes from. Only plain float sources the
 * architecture forces no type on are requantized, and LoRA factors never are. The members of a fused projection
 * ({@link FusedProjections}) whose method has tensor-scoped scales are held until the pass ends and then
 * quantized together ({@link #flush}), so they share vLLM's fused scales. Not thread-safe; one per planning pass.
 */
final class HfOnlineRequant {

    private static final Logger LOG = LogManager.getLogger(HfOnlineRequant.class);

    private final HfForceType forceType;
    private final DtypePolicy policy;
    private final Arch arch;
    private final boolean quantizeOutputTensor;
    private final OnlineQuant target;
    private final OnlineQuantConfig config;
    private final Predicate<String> hasTensor;
    private final UnaryOperator<String> checkpointName;
    private final VllmWeightsMapper mapper;
    private final List<Held> held = new ArrayList<>();
    private final Map<String, Set<String>> summary = new TreeMap<>();

    /**
     * @param target    the load's {@code --quant-online} value, which a llama-quantize type applies alone
     * @param arch      the resolved model architecture, whose name rules llama-quantize gates some tensors on
     * @param quantizeOutputTensor whether {@code output.weight} is quantized (llama-quantize's
     *                  {@code quantize_output_tensor}; false is its {@code --leave-output-tensor})
     * @param config    the load's vLLM configuration ({@link OnlineQuantConfig#effective}); null for a
     *                  llama-quantize type
     * @param hasTensor whether the checkpoint stores a tensor of that name (the shards of a fused layer)
     * @param checkpointName the checkpoint name of a tensor by its converter name (the converter's renames
     *                  undone), which vLLM's mapper starts from
     * @param mapper    vLLM's {@code hf_to_vllm_mapper} of the checkpoint's architecture, which names the layers
     *                  the configuration's patterns match
     * @throws GgufFormatException when the configuration matches layer names and vLLM names the architecture's
     *                             layers with code the mapper table does not hold
     */
    HfOnlineRequant(HfForceType forceType, DtypePolicy policy, Arch arch, boolean quantizeOutputTensor,
                    OnlineQuant target, OnlineQuantConfig config, Predicate<String> hasTensor,
                    UnaryOperator<String> checkpointName, VllmWeightsMapper mapper) {
        this.forceType = Objects.requireNonNull(forceType, "forceType");
        this.policy = Objects.requireNonNull(policy, "policy");
        this.arch = Objects.requireNonNull(arch, "arch");
        this.quantizeOutputTensor = quantizeOutputTensor;
        this.target = Objects.requireNonNull(target, "target");
        this.config = config;
        this.hasTensor = Objects.requireNonNull(hasTensor, "hasTensor");
        this.checkpointName = Objects.requireNonNull(checkpointName, "checkpointName");
        Objects.requireNonNull(mapper, "mapper");
        if (mapper.unresolved() && config != null && config.matchesNames()) {
            throw new GgufFormatException("online quantization: quantization_config.targets and .ignore name vLLM's"
                    + " layers, and vLLM names the layers of " + mapper.architecture() + " with code ("
                    + mapper.unresolvedReason() + "); use the linear and moe members without ignore");
        }
        this.mapper = mapper.unresolved() ? VllmWeightsMapper.IDENTITY : mapper;
    }

    /** A fused-projection member held until {@link #flush}. */
    private static final class Held {
        final String ggufName;
        final RowSource src;
        final OnlineQuant target;
        final String group;

        Held(String ggufName, RowSource src, OnlineQuant target, String group) {
            this.ggufName = ggufName;
            this.src = src;
            this.target = target;
            this.group = group;
        }
    }

    /**
     * Adds {@code ggufName}, planned from checkpoint tensor {@code hfName}, to {@code out}: requantized when its
     * source and the load select it, held for {@link #flush} when it is a fused-projection member with
     * tensor-scoped scales, as stored otherwise.
     *
     * @throws GgufFormatException naming the tensor when the quantization config refuses its layer
     */
    void add(TensorSink out, String hfName, String ggufName, RowSource src) {
        if (src instanceof LoraFactorSource || !src.format().isFloat()
                || forceType.force(ggufName, Math.max(1, src.shape().length), src.format(), policy.outType()) != null) {
            out.add(ggufName, src);
            return;
        }
        GGMLType current = policy.typeFor(ggufName, src, null);
        if (config == null) {
            out.add(ggufName, RequantPolicy.replace(ggufName, arch, quantizeOutputTensor, src, current, target));
            return;
        }
        LayerKind kind = RequantPolicy.layerKind(ggufName);
        if (kind == null) {
            out.add(ggufName, src);
            return;
        }
        Layer layer = Layer.of(checkpointName.apply(Objects.requireNonNull(hfName, "hfName")), kind, hasTensor,
                mapper);
        Selection sel = select(ggufName, kind, layer);
        if (sel == null || !RequantPolicy.eligible(ggufName, arch, quantizeOutputTensor, src, current,
                sel.target())) {
            out.add(ggufName, src);
            return;
        }
        summary.computeIfAbsent(afterIndex(layer.name()) + " (" + sel.origin() + ")", k -> new LinkedHashSet<>())
                .add(layer.name());
        String group = FusedProjections.groupOf(ggufName);
        if (group != null && OnlineQuantizers.scanning(sel.target()) != null) {
            held.add(new Held(ggufName, src, sel.target(), group + "|" + layer.name() + "|" + sel.target()));
            return;
        }
        out.add(ggufName, RequantPolicy.quantize(ggufName, arch, quantizeOutputTensor, src, current,
                sel.target()));
    }

    private Selection select(String ggufName, LayerKind kind, Layer layer) {
        try {
            return config.select(kind, layer);
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("online quantization of " + ggufName + ": " + e.getMessage(), e);
        }
    }

    /**
     * Quantizes the held fused-projection members, each group with shared scales
     * ({@link RequantPolicy#quantizeFused}), and adds them to {@code out} in the order they arrived; then logs
     * the layers the pass quantized. A {@code kv_b} group missing a member stays as stored: its scales span both.
     */
    void flush(TensorSink out) {
        Map<String, List<Held>> groups = new LinkedHashMap<>();
        for (Held h : held) {
            groups.computeIfAbsent(h.group, k -> new ArrayList<>()).add(h);
        }
        Map<Held, RowSource> quantized = new IdentityHashMap<>();
        for (List<Held> members : groups.values()) {
            members.sort(Comparator.comparingInt(h -> FusedProjections.ordinalOf(h.ggufName)));
            List<String> names = new ArrayList<>();
            List<RowSource> srcs = new ArrayList<>();
            for (Held h : members) {
                names.add(h.ggufName);
                srcs.add(h.src);
            }
            Held first = members.get(0);
            if (FusedProjections.layoutOf(first.ggufName) == FusedProjections.Layout.MLA_KV_B
                    && members.size() != FusedProjections.sizeOf(first.ggufName)) {
                LOG.warn("online quantization leaves {} as stored: vLLM quantizes kv_b_proj whole, and the other"
                        + " part of it is not quantized", names);
                for (Held h : members) {
                    quantized.put(h, h.src);
                }
                continue;
            }
            List<RowSource> q = RequantPolicy.quantizeFused(names, srcs, first.target);
            for (int i = 0; i < members.size(); i++) {
                quantized.put(members.get(i), q.get(i));
            }
        }
        for (Held h : held) {
            out.add(h.ggufName, quantized.get(h));
        }
        held.clear();
        logSummary();
    }

    private void logSummary() {
        if (summary.isEmpty()) {
            return;
        }
        StringJoiner j = new StringJoiner("; ");
        for (Map.Entry<String, Set<String>> e : summary.entrySet()) {
            int at = e.getKey().indexOf(" (");
            j.add(e.getKey().substring(0, at) + ": " + e.getValue().size() + e.getKey().substring(at));
        }
        LOG.info("online quantization: {}", j);
        summary.clear();
    }

    /** {@code get_layer_name_after_index}: the part of {@code layer} after its last numeric component. */
    private static String afterIndex(String layer) {
        String[] parts = layer.split("\\.");
        for (int i = parts.length - 1; i >= 0; i--) {
            if (!parts[i].isEmpty() && parts[i].chars().allMatch(Character::isDigit)) {
                return String.join(".", Arrays.copyOfRange(parts, i + 1, parts.length));
            }
        }
        return layer;
    }
}
