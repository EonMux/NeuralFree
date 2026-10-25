package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * The quantized modules of a checkpoint: tensor names grouped into modules by stripping the auxiliary suffix of every
 * quantization format. A suffix matches in dotted form ({@code <module>.qweight}, and {@code <module>.weight.absmax}
 * for the bitsandbytes state attached to the weight) and, for a quantized parameter that is not a {@code .weight}
 * (fused experts), in underscore form ({@code <module>_qweight}); the longest matching suffix wins and the dotted form
 * wins a tie. Some suffixes depend on the module's format:
 * <ul>
 *   <li>{@code .scale}, {@code .zero}, {@code .meta} and the HQQ metadata tensors ({@link #HQQ_META}) are auxiliary
 *       for HQQ modules ({@link QuantFormat#INT_HQQ});</li>
 *   <li>{@code .scale} is also auxiliary for DeepSeek V4 modules (the {@code deepseek_v4_fp8} method or
 *       {@link QuantFormat#MXFP4_DSV4}) whose {@code <module>.weight} is stored as FP8 or packed bytes;</li>
 *   <li>a parameter {@code <p>} (no {@code .weight}) with a {@code <p>_scale_inv} sibling is the transformers fused
 *       FP8 expert module {@code <p>} (finegrained_fp8.py:673-680).</li>
 * </ul>
 * A name that has an auxiliary suffix marks its module as quantized; {@link #modules()} lists those modules in the
 * order of their first tensor and {@link #auxiliary()} the auxiliary names. Immutable.
 */
public final class QuantModuleIndex {

    /** Auxiliary suffixes of every format whose meaning does not depend on the module's format. */
    static final List<String> COMMON = Collections.unmodifiableList(Arrays.asList(
            // GPTQ, AWQ, MLX and bitsandbytes
            "qweight", "qzeros", "scales", "g_idx", "biases", "SCB", "weight_format", "absmax", "quant_map",
            "nested_absmax", "nested_quant_map",
            // compressed-tensors, ModelOpt and FP8
            "weight_packed", "weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx", "weight_global_scale",
            "input_global_scale", "input_scale", "input_zero_point", "output_scale", "weight_scale_inv",
            "weight_scale_2", "qscale_weight", "qscale_act",
            // HQQ and fp_quant
            "W_q", "dqweight", "act_global_scale", "forward_hadamard_matrix", "backward_hadamard_matrix"));

    /** Auxiliary suffixes that only exist in underscore form (gpt-oss MXFP4 experts: {@code <p>_blocks}). */
    static final List<String> UNDERSCORE_ONLY = Collections.singletonList("blocks");

    /** The bitsandbytes quantization-state prefix: {@code <module>.weight.quant_state.bitsandbytes__<type>}. */
    static final String BNB_QUANT_STATE = "quant_state.";

    /** HQQ module tensors that are auxiliary only for HQQ modules (HqqScheme). */
    static final List<String> HQQ_META = Collections.unmodifiableList(Arrays.asList(
            "scale", "zero", "meta", "shape", "axis", "nbits", "group_size", "view_as_float", "quant_scale",
            "quant_zero", "channel_wise", "compute_dtype", "encoded_state_dict", "offload_meta", "optimize",
            "packing", "round_zero", "stores_quant_config", "unpack_view_dtype"));

    /** The DeepSeek V4 per-block scale of an FP8 or packed weight. */
    static final String DSV4_SCALE = "scale";

    /** The scale suffix of a transformers fused FP8 expert parameter. */
    static final String FUSED_SCALE_INV = "_scale_inv";

    private final Map<String, List<String>> modules;
    private final Set<String> auxiliary;

    private QuantModuleIndex(Map<String, List<String>> modules, Set<String> auxiliary) {
        this.modules = modules;
        this.auxiliary = auxiliary;
    }

    /**
     * Groups the names of {@code in}.
     *
     * @param argsFor the quantization arguments of a module, or null when it is not quantized
     * @param dsv4    true for the {@code deepseek_v4_fp8} method, whose FP8 weights carry {@code .scale}
     */
    public static QuantModuleIndex of(HfTensorIndex in, Function<String, QuantArgs> argsFor, boolean dsv4) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(argsFor, "argsFor");
        Map<String, List<String>> modules = new LinkedHashMap<>();
        Set<String> aux = new LinkedHashSet<>();
        for (String name : in.names()) {
            String module = moduleOf(name, in, argsFor, dsv4);
            if (module != null) {
                modules.computeIfAbsent(module, k -> new ArrayList<>()).add(name);
                aux.add(name);
            }
        }
        Map<String, List<String>> frozen = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> e : modules.entrySet()) {
            frozen.put(e.getKey(), Collections.unmodifiableList(e.getValue()));
        }
        return new QuantModuleIndex(Collections.unmodifiableMap(frozen),
                Collections.unmodifiableSet(aux));
    }

    /** The module {@code name} is an auxiliary tensor of, or null when it is not auxiliary. */
    static String moduleOf(String name, HfTensorIndex in, Function<String, QuantArgs> argsFor, boolean dsv4) {
        String bnb = bnbQuantStateModule(name);
        if (bnb != null) {
            return bnb;
        }
        String dotted = null;
        int dottedLen = -1;
        for (String s : COMMON) {
            String m = strip(name, "." + s);
            if (m != null && s.length() > dottedLen) {
                dotted = m;
                dottedLen = s.length();
            }
        }
        String underscore = null;
        int underscoreLen = -1;
        for (List<String> list : Arrays.asList(COMMON, UNDERSCORE_ONLY)) {
            for (String s : list) {
                String m = strip(name, "_" + s);
                if (m != null && s.length() > underscoreLen && underscoreModule(m, argsFor)) {
                    underscore = m;
                    underscoreLen = s.length();
                }
            }
        }
        String fused = strip(name, FUSED_SCALE_INV);
        int fusedLen = FUSED_SCALE_INV.length() - 1;
        if (fused != null && fusedLen > underscoreLen && in.has(fused) && underscoreModule(fused, argsFor)) {
            underscore = fused;
            underscoreLen = fusedLen;
        }
        if (dotted != null && dottedLen >= underscoreLen) {
            String bnbWeight = strip(dotted, ".weight");
            return bnbWeight != null ? bnbWeight : dotted;
        }
        if (underscore != null) {
            return underscore;
        }
        return conditionalModule(name, in, argsFor, dsv4);
    }

    /**
     * True when {@code module} can own underscore-form tensors: a quantized parameter that is not a {@code .weight}
     * (a fused expert parameter such as {@code mlp.experts.gate_up_proj}).
     */
    private static boolean underscoreModule(String module, Function<String, QuantArgs> argsFor) {
        return !module.endsWith(".weight") && argsFor.apply(module) != null;
    }

    /** The module of a format-dependent auxiliary tensor (HQQ metadata, DeepSeek V4 {@code .scale}), or null. */
    private static String conditionalModule(String name, HfTensorIndex in, Function<String, QuantArgs> argsFor,
                                            boolean dsv4) {
        int dot = name.lastIndexOf('.');
        if (dot <= 0) {
            return null;
        }
        String module = name.substring(0, dot);
        String suffix = name.substring(dot + 1);
        if (!HQQ_META.contains(suffix)) {
            return null;
        }
        QuantArgs args = argsFor.apply(module);
        if (args == null) {
            return null;
        }
        if (args.format() == QuantFormat.INT_HQQ) {
            return module;
        }
        boolean dsv4Module = dsv4 || args.format() == QuantFormat.MXFP4_DSV4;
        if (DSV4_SCALE.equals(suffix) && dsv4Module && packedOrF8(in, module + ".weight")) {
            return module;
        }
        return null;
    }

    private static boolean packedOrF8(HfTensorIndex in, String weight) {
        if (!in.has(weight)) {
            return false;
        }
        RowSource w = in.get(weight);
        StDType t = w.format().stDType();
        if (t == null) {
            return w.format().isBlock();
        }
        switch (t) {
            case F8_E4M3:
            case F8_E5M2:
            case F8_E4M3FNUZ:
            case F8_E5M2FNUZ:
            case U8:
            case I8:
                return true;
            default:
                return false;
        }
    }

    /** {@code <module>} of {@code <module>.weight.quant_state.<x>}, or null. */
    private static String bnbQuantStateModule(String name) {
        int at = name.indexOf(".weight." + BNB_QUANT_STATE);
        if (at <= 0 || name.length() == at + ".weight.".length() + BNB_QUANT_STATE.length()) {
            return null;
        }
        return name.substring(0, at);
    }

    /** {@code name} without {@code suffix} when it ends with it and leaves a non-empty module, else null. */
    private static String strip(String name, String suffix) {
        if (name.length() > suffix.length() && name.endsWith(suffix)) {
            return name.substring(0, name.length() - suffix.length());
        }
        return null;
    }

    /** The quantized modules and their auxiliary tensors, in the order of their first tensor. Unmodifiable. */
    public Map<String, List<String>> modules() {
        return modules;
    }

    /** Every auxiliary tensor name. Unmodifiable. */
    public Set<String> auxiliary() {
        return auxiliary;
    }

    @Override
    public String toString() {
        return "QuantModuleIndex{" + modules.size() + " modules, " + auxiliary.size() + " auxiliary tensors}";
    }
}
