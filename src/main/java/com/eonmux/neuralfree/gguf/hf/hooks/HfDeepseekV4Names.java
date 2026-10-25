package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The DeepSeek-V4 native tensor name table, a port of {@code DeepseekV4Model._map_dsv4_tensor_name}
 * (LC/conversion/deepseek.py:809-878): DeepSeek-V4 checkpoints use the reference implementation's names
 * ({@code embed.weight}, {@code layers.<b>.attn.wq_a.weight}, ...), which the converter maps through this explicit
 * table instead of the tensor name map.
 */
public final class HfDeepseekV4Names {

    /** The routed expert members, {@code ffn.experts.<x>.w1|w2|w3.weight|scale} (deepseek.py:874). */
    private static final Pattern EXPERT_TAIL = Pattern.compile("ffn\\.experts\\.\\d+\\.w[123]\\.(weight|scale)");

    /** {@code layers.<b>.<tail>} (deepseek.py:820). */
    private static final Pattern LAYER = Pattern.compile("layers\\.(\\d+)\\.(.+)");

    /** The longest block number accepted, as {@code HfLoadContext.bidOf} bounds it. */
    private static final int MAX_BID_DIGITS = 9;

    private static final String WEIGHT = ".weight";
    private static final String BIAS = ".bias";

    private static final Map<String, Mapped> ROOT = rootMap();
    private static final Map<String, Mapped> LAYERS = layerMap();

    private HfDeepseekV4Names() {
    }

    /**
     * A native name table: the converter tensor of a name in block {@code bid}. Subclasses that extend
     * {@code _map_dsv4_tensor_name} (DeepseekV4DSpark's root names, deepseek.py:1005-1008) consult their own entries
     * and fall back to {@link #DEFAULT}.
     */
    @FunctionalInterface
    public interface Table {

        /** The table of {@code DeepseekV4Model} itself, {@link HfDeepseekV4Names#map}. */
        Table DEFAULT = HfDeepseekV4Names::map;

        /**
         * The mapping of {@code name} in block {@code bid} (-1 for none).
         *
         * @throws GgufFormatException when the table does not map the name
         */
        Mapped map(String name, int bid);
    }

    /** A native name's converter tensor ({@code MODEL_TENSOR} name), GGUF suffix, and whether it is model-level. */
    public static final class Mapped {
        private final String tensor;
        private final String suffix;
        private final boolean root;

        /**
         * A mapping to converter tensor {@code tensor} with GGUF suffix {@code suffix}; {@code root} for a model-level
         * tensor.
         *
         * @throws IllegalArgumentException when the tensor is empty or the suffix is not {@code .weight} or
         *                                  {@code .bias}
         */
        public Mapped(String tensor, String suffix, boolean root) {
            this.tensor = Objects.requireNonNull(tensor, "tensor");
            this.suffix = Objects.requireNonNull(suffix, "suffix");
            if (tensor.isEmpty() || !(suffix.equals(WEIGHT) || suffix.equals(BIAS))) {
                throw new IllegalArgumentException("invalid DeepSeek-V4 mapping '" + tensor + "' + '" + suffix + "'");
            }
            this.root = root;
        }

        /** The converter tensor, such as {@code ATTN_Q_A}. */
        public String tensor() {
            return tensor;
        }

        /** The GGUF name suffix, {@code .weight} or {@code .bias}. */
        public String suffix() {
            return suffix;
        }

        /** True for a model-level tensor, which takes no block number. */
        public boolean root() {
            return root;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Mapped)) {
                return false;
            }
            Mapped m = (Mapped) o;
            return tensor.equals(m.tensor) && suffix.equals(m.suffix) && root == m.root;
        }

        @Override
        public int hashCode() {
            return Objects.hash(tensor, suffix, root);
        }

        @Override
        public String toString() {
            return tensor + suffix;
        }
    }

    /**
     * {@code _map_dsv4_tensor_name(name, bid)}: the model-level table, else {@code layers.<b>.<tail>} through the layer
     * table, the routed expert members mapping to {@code FFN_GATE_EXP} as the converter's table does.
     *
     * @param name the native tensor name
     * @param bid  the block number the planner parsed from the name ({@code HfLoadContext.bidOf}), or -1 for none
     * @throws GgufFormatException {@code "Unsupported DeepSeek-V4 tensor"} when the name is in neither table, or when
     *                             the layer number of the name is not {@code bid}
     */
    public static Mapped map(String name, int bid) {
        Objects.requireNonNull(name, "name");
        Mapped root = ROOT.get(name);
        if (root != null) {
            return root;
        }
        Matcher m = LAYER.matcher(name);
        if (!m.matches()) {
            throw unsupported(name);
        }
        String digits = m.group(1);
        if (digits.length() > MAX_BID_DIGITS) {
            throw new GgufFormatException("tensor '" + name + "': block number " + digits + " is too large");
        }
        int layer = Integer.parseInt(digits);
        if (bid != layer) {
            throw new GgufFormatException("Tensor '" + name + "' parsed bid " + (bid < 0 ? "None" : bid)
                    + " but layer name has " + layer);
        }
        String tail = m.group(2);
        Mapped mapped = LAYERS.get(tail);
        if (mapped != null) {
            return mapped;
        }
        if (EXPERT_TAIL.matcher(tail).matches()) {
            return new Mapped("FFN_GATE_EXP", WEIGHT, false);
        }
        throw unsupported(name);
    }

    /**
     * The GGUF name of the native tensor {@code name} in block {@code bid}: {@link #map} formatted by {@code names}
     * ({@code _format_dsv4_tensor_name}, deepseek.py:806-807).
     *
     * @throws GgufFormatException      as {@link #map} does
     * @throws IllegalArgumentException when the architecture of {@code names} does not list the tensor
     */
    public static String ggufName(HfTensorNameMap names, String name, int bid) {
        Objects.requireNonNull(names, "names");
        Mapped m = map(name, bid);
        return names.ggufName(m.tensor(), m.root() ? -1 : bid, m.suffix());
    }

    /** The model-level table (unmodifiable), native name to mapping. */
    public static Map<String, Mapped> rootTable() {
        return ROOT;
    }

    /** The per-layer table (unmodifiable), tail after {@code layers.<b>.} to mapping. */
    public static Map<String, Mapped> layerTable() {
        return LAYERS;
    }

    private static GgufFormatException unsupported(String name) {
        return new GgufFormatException("Unsupported DeepSeek-V4 tensor '" + name + "'");
    }

    /** {@code root_map} (deepseek.py:810-817). */
    private static Map<String, Mapped> rootMap() {
        Map<String, Mapped> m = new LinkedHashMap<>();
        root(m, "embed.weight", "TOKEN_EMBD");
        root(m, "norm.weight", "OUTPUT_NORM");
        root(m, "head.weight", "OUTPUT");
        root(m, "hc_head_fn", "HC_HEAD_FN");
        root(m, "hc_head_base", "HC_HEAD_BASE");
        root(m, "hc_head_scale", "HC_HEAD_SCALE");
        return Collections.unmodifiableMap(m);
    }

    /** {@code layer_map} (deepseek.py:828-867). */
    private static Map<String, Mapped> layerMap() {
        Map<String, Mapped> m = new LinkedHashMap<>();
        layer(m, "hc_attn_fn", "HC_ATTN_FN");
        layer(m, "hc_attn_base", "HC_ATTN_BASE");
        layer(m, "hc_attn_scale", "HC_ATTN_SCALE");
        layer(m, "hc_ffn_fn", "HC_FFN_FN");
        layer(m, "hc_ffn_base", "HC_FFN_BASE");
        layer(m, "hc_ffn_scale", "HC_FFN_SCALE");
        layer(m, "attn.attn_sink", "ATTN_SINKS");
        layer(m, "attn.wq_a.weight", "ATTN_Q_A");
        layer(m, "attn.wq_b.weight", "ATTN_Q_B");
        layer(m, "attn.q_norm.weight", "ATTN_Q_A_NORM");
        layer(m, "attn.wkv.weight", "ATTN_KV");
        layer(m, "attn.kv_norm.weight", "ATTN_KV_NORM");
        layer(m, "attn.wo_a.weight", "ATTN_OUT_A");
        layer(m, "attn.wo_b.weight", "ATTN_OUT_B");
        layer(m, "attn.compressor.ape", "ATTN_COMPRESSOR_APE");
        layer(m, "attn.compressor.wkv.weight", "ATTN_COMPRESSOR_WKV");
        layer(m, "attn.compressor.wgate.weight", "ATTN_COMPRESSOR_WGATE");
        layer(m, "attn.compressor.norm.weight", "ATTN_COMPRESSOR_NORM");
        layer(m, "attn.indexer.wq_b.weight", "INDEXER_ATTN_Q_B");
        layer(m, "attn.indexer.weights_proj.weight", "INDEXER_PROJ");
        layer(m, "attn.indexer.compressor.ape", "INDEXER_COMPRESSOR_APE");
        layer(m, "attn.indexer.compressor.wkv.weight", "INDEXER_COMPRESSOR_WKV");
        layer(m, "attn.indexer.compressor.wgate.weight", "INDEXER_COMPRESSOR_WGATE");
        layer(m, "attn.indexer.compressor.norm.weight", "INDEXER_COMPRESSOR_NORM");
        layer(m, "attn_norm.weight", "ATTN_NORM");
        layer(m, "ffn_norm.weight", "FFN_NORM");
        layer(m, "ffn.gate.weight", "FFN_GATE_INP");
        m.put("ffn.gate.bias", new Mapped("FFN_EXP_PROBS_B", BIAS, false));
        m.put("ffn.gate.bias_vl", new Mapped("FFN_EXP_PROBS_B_VL", BIAS, false));
        layer(m, "ffn.gate.tid2eid", "FFN_GATE_TID2EID");
        layer(m, "ffn.shared_experts.w1.weight", "FFN_GATE_SHEXP");
        layer(m, "ffn.shared_experts.w2.weight", "FFN_DOWN_SHEXP");
        layer(m, "ffn.shared_experts.w3.weight", "FFN_UP_SHEXP");
        layer(m, "nextn.eh_proj.weight", "NEXTN_EH_PROJ");
        layer(m, "nextn.enorm.weight", "NEXTN_ENORM");
        layer(m, "nextn.hnorm.weight", "NEXTN_HNORM");
        layer(m, "nextn.shared_head_norm.weight", "NEXTN_SHARED_HEAD_NORM");
        layer(m, "nextn.embed_tokens.weight", "NEXTN_EMBED_TOKENS");
        layer(m, "nextn.shared_head_head.weight", "NEXTN_SHARED_HEAD_HEAD");
        return Collections.unmodifiableMap(m);
    }

    private static void root(Map<String, Mapped> m, String name, String tensor) {
        m.put(name, new Mapped(tensor, WEIGHT, true));
    }

    private static void layer(Map<String, Mapped> m, String tail, String tensor) {
        m.put(tail, new Mapped(tensor, WEIGHT, false));
    }
}
