package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfHadamardMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Hooks of the Qwen linear-attention (Gated DeltaNet) converter classes (LC/conversion/qwen.py):
 * {@code Qwen3NextModel} (:380-444), {@code _LinearAttentionVReorderBase} (:463-655), {@code _Qwen35MRopeMixin}
 * (:658-671), {@code Qwen3_5TextModel} (:674-690) and {@code Qwen3_5MoeTextModel} (:749-752). The MTP wiring of
 * these classes is {@link HfQwenHooks#mtpFilter}, {@link HfQwenHooks#mtpBlockCount} and
 * {@link HfQwenHooks#mtpMeta}; their vocabulary is {@link HfQwenMoeHooks#qwen2MoeVocab}.
 */
public final class HfQwenLinearHooks {

    /** The rope sections Qwen3.5 always applies when the checkpoint omits them (qwen.py:663). */
    static final List<Integer> DEFAULT_MROPE_SECTIONS = Collections.unmodifiableList(Arrays.asList(11, 11, 10, 0));
    /** The partial rotary factor of Qwen3-Next when the RoPE parameters have none (qwen.py:402). */
    static final double DEFAULT_PARTIAL_ROTARY_FACTOR = 0.25;
    /** The full-attention interval of Qwen3-Next when the configuration has none (qwen.py:399). */
    static final int DEFAULT_FULL_ATTENTION_INTERVAL = 4;

    private static final String ATTN_QKV = "ATTN_QKV";
    private static final String ATTN_GATE = "ATTN_GATE";
    private static final String WEIGHT = ".weight";
    private static final long UINT32_MAX = 0xFFFF_FFFFL;
    private static final String V_GROUPED_STATE = HfQwenLinearHooks.class.getName() + ".vGrouped";

    private HfQwenLinearHooks() {
    }

    // ------------------------------------------------------------------ Qwen3NextModel

    /**
     * {@code Qwen3NextModel.set_gguf_parameters} (qwen.py:385-402): {@link HfQwenMoeHooks#qwen2MoeMeta} and the MTP
     * mixin's {@link HfQwenHooks#mtpMeta}, then the SSM sizes ({@code linear_conv_kernel_dim},
     * {@code linear_key_head_dim}, {@code linear_num_key_heads}, {@code linear_num_value_heads},
     * {@code linear_value_head_dim * linear_num_value_heads}), the recurrent layers of {@code layer_types} (MTP layers
     * are attention layers), {@code full_attention_interval} (default 4), and the rope dimension count
     * {@code head_dim} (or {@code hidden_size / num_attention_heads}) times the partial rotary factor (default 0.25).
     *
     * @throws GgufFormatException when a required size is missing or {@code layer_types} does not have one entry per
     *                             layer
     */
    public static void qwen3NextMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfQwenMoeHooks.qwen2MoeMeta(ctx, kv, plan);
        HfQwenHooks.mtpMeta(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.SSM_CONV_KERNEL, cfg.requireInt("linear_conv_kernel_dim"));
        kv.set(GgufKeys.SSM_STATE_SIZE, cfg.requireInt("linear_key_head_dim"));
        kv.set(GgufKeys.SSM_GROUP_COUNT, cfg.requireInt("linear_num_key_heads"));
        int nV = cfg.requireInt("linear_num_value_heads");
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, nV);
        long inner = (long) cfg.requireInt("linear_value_head_dim") * nV;
        if (inner > UINT32_MAX) {
            throw new GgufFormatException(cfg.source() + ": SSM inner size " + inner + " does not fit UINT32");
        }
        kv.set(GgufKeys.SSM_INNER_SIZE, inner);
        recurrentLayers(ctx, kv);
        kv.set(GgufKeys.FULL_ATTENTION_INTERVAL,
                cfg.findInt(DEFAULT_FULL_ATTENTION_INTERVAL, "full_attention_interval"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDimensionCount(cfg));
    }

    /** {@code add_recurrent_layers} from {@code layer_types} (qwen.py:392-398). */
    private static void recurrentLayers(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        Optional<Object> layerTypes = cfg.find("layer_types");
        if (!layerTypes.isPresent()) {
            return;
        }
        if (!(layerTypes.get() instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": 'layer_types' must be an array");
        }
        List<?> types = (List<?>) layerTypes.get();
        int nLayer = cfg.requireInt("num_hidden_layers");
        if (types.size() != nLayer) {
            throw new GgufFormatException(cfg.source() + ": layer_types has " + types.size()
                    + " entries, expected num_hidden_layers (" + nLayer + ")");
        }
        List<Boolean> recurrent = new ArrayList<>(Math.max(nLayer, ctx.blockCount()));
        for (Object t : types) {
            recurrent.add("linear_attention".equals(t));
        }
        for (int i = nLayer; i < ctx.blockCount(); i++) {
            recurrent.add(false);
        }
        kv.set(GgufKeys.ATTENTION_RECURRENT_LAYERS, recurrent);
    }

    /** {@code int(rope_dim * partial_rotary_factor)} (qwen.py:400-402). */
    static long ropeDimensionCount(HfConfig cfg) {
        OptionalInt headDim = cfg.findInt("head_dim");
        long ropeDim;
        if (headDim.isPresent()) {
            ropeDim = headDim.getAsInt();
        } else {
            int heads = cfg.requireInt("num_attention_heads");
            if (heads <= 0) {
                throw new GgufFormatException(cfg.source() + ": num_attention_heads " + heads + " is not positive");
            }
            ropeDim = Math.floorDiv(cfg.requireInt("hidden_size"), heads);
        }
        Map<String, Object> rope = cfg.rope().raw();
        double factor = DEFAULT_PARTIAL_ROTARY_FACTOR;
        if (rope.containsKey("partial_rotary_factor")) {
            Object f = rope.get("partial_rotary_factor");
            if (!(f instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": rope_parameters.partial_rotary_factor must be a"
                        + " number, got " + f);
            }
            factor = ((Number) f).doubleValue();
        }
        double dim = ropeDim * factor;
        if (!(dim >= 0) || dim > UINT32_MAX) {
            throw new GgufFormatException(cfg.source() + ": rope dimension " + ropeDim + " * " + factor
                    + " is out of range");
        }
        return (long) dim;
    }

    /**
     * {@code Qwen3NextModel.modify_tensors} (qwen.py:404-444): {@code A_log} becomes {@code -exp(A_log)};
     * {@code dt_bias} is renamed {@code dt_proj.bias}; {@code conv1d} weights are squeezed; every
     * {@code norm.weight} except {@code linear_attn.norm.weight} gets +1. A fused {@code in_proj_qkvz} weight,
     * grouped per key head as {@code [q, k, v, z]}, is regrouped into {@code attn_qkv} ({@code [all q; all k;
     * all v]}) and {@code attn_gate} (all z) ({@link QkvLayouts#splitQkvz}); {@code in_proj_ba} stays fused and maps
     * to {@code ssm_ba}. Then {@link HfQwenMoeHooks#qwen2MoeTensors}.
     *
     * @throws GgufFormatException      when the value heads do not group over the key heads
     * @throws IllegalArgumentException when a fused {@code in_proj_qkvz} does not hold the configured heads
     */
    public static void qwen3NextTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        String name = hfName;
        RowSource s = src;
        if (name.endsWith(".A_log")) {
            s = ElementTransforms.negExp(s);
        } else if (name.endsWith(".dt_bias")) {
            name = HfHookSupport.renameTail(name, ".dt_bias", ".dt_proj.bias");
        } else if (name.contains("conv1d")) {
            s = ElementTransforms.squeezeConv(s);
        } else if (HfHookSupport.isNorm(name) && !name.endsWith("linear_attn.norm.weight")) {
            s = HfHookSupport.normPlusOne(s);
        }
        if (name.contains("in_proj_qkvz.weight")) {
            HfConfig cfg = ctx.config();
            int nK = cfg.requireInt("linear_num_key_heads");
            int nV = cfg.requireInt("linear_num_value_heads");
            if (nK <= 0 || nV <= 0 || nV % nK != 0) {
                throw new GgufFormatException(cfg.source() + ": " + nV + " linear-attention value heads do not group"
                        + " over " + nK + " key heads");
            }
            RowSource[] qkvz = QkvLayouts.splitQkvz(s, nK, nV, cfg.requireInt("linear_key_head_dim"),
                    cfg.requireInt("linear_value_head_dim"));
            int bid = HfHookSupport.layerOf(name);
            out.add(ctx.names().ggufName(ATTN_QKV, bid, WEIGHT), qkvz[0]);
            out.add(ctx.names().ggufName(ATTN_GATE, bid, WEIGHT), qkvz[1]);
            return;
        }
        HfQwenMoeHooks.qwen2MoeTensors(ctx, name, s, out);
    }

    // ------------------------------------------------------------------ _LinearAttentionVReorderBase

    /**
     * {@code _LinearAttentionVReorderBase.modify_tensors} (qwen.py:602-655). When the linear attention has fewer key
     * heads than value heads, the value heads are stored grouped by key head ({@code [K0: v0..v(r-1), K1: ...]}) and
     * are reordered to the tiled order the runtime broadcasts ({@code [K0 v0, K1 v0, ..., K0 v1, K1 v1, ...]}): the
     * V rows of {@code in_proj_qkv}, the rows of {@code in_proj_z} (per value head) and {@code in_proj_a}/{@code
     * in_proj_b} (one row per head), the entries of {@code A_log}/{@code dt_bias}/{@code dt_proj} (the last dimension
     * when not 1-D), the V channels of the squeezed {@code conv1d}, and the input-column groups of {@code out_proj}.
     * An {@code out_proj} folded by {@code hadamard_packing.json} keeps the grouped order, and the load records
     * {@code prism.hadamard.gdn_v_grouped} once so the runtime permutes its input instead. Quantized weights such as
     * NVFP4 follow the same row and column-group permutation ({@code _transform_nvfp4_weight}, :503-591): rows move
     * whole, and column groups move whole blocks when the head width is a whole number of blocks
     * ({@link ColumnTransforms} dequantizes otherwise). Then {@link #qwen3NextTensors}.
     *
     * @throws GgufFormatException when the value heads are not a whole multiple of the key heads, or a tensor does
     *                             not hold the value heads its name implies
     * @throws IOException         when the Hadamard manifest cannot be read
     */
    public static void vReorderTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfConfig cfg = ctx.config();
        int nK = cfg.findInt(0, "linear_num_key_heads");
        int nV = cfg.findInt(0, "linear_num_value_heads");
        RowSource s = src;
        if (nK > 0 && nV > 0 && nK != nV && hfName.contains("linear_attn.")) {
            s = reorderValueHeads(ctx, hfName, src, out, nK, nV);
        }
        qwen3NextTensors(ctx, hfName, s, out);
    }

    private static RowSource reorderValueHeads(HfLoadContext ctx, String name, RowSource src, TensorSink out,
                                               int nK, int nV) throws IOException {
        HfConfig cfg = ctx.config();
        if (nV % nK != 0) {
            throw new GgufFormatException(cfg.source() + ": " + nV + " linear-attention value heads are not a"
                    + " multiple of the " + nK + " key heads");
        }
        int hk = cfg.requireInt("linear_key_head_dim");
        int hv = cfg.requireInt("linear_value_head_dim");
        if (hk <= 0 || hv <= 0) {
            throw new GgufFormatException(cfg.source() + ": linear-attention head sizes " + hk + " (key) and " + hv
                    + " (value) must be positive");
        }
        ValueHeads v = new ValueHeads(name, nK, nV / nK);
        long qk = (long) hk * nK;
        if (name.contains(".in_proj_qkv.")) {
            return v.rows(src, 2 * qk, hv);
        }
        if (name.contains(".in_proj_z.")) {
            return v.rows(src, 0, hv);
        }
        if (name.contains(".in_proj_b.") || name.contains(".in_proj_a.")) {
            return v.rows(src, 0, 1);
        }
        if (name.contains(".A_log") || name.contains(".dt_bias") || name.contains(".dt_proj")) {
            return src.shape().length == 1 ? v.rows(src, 0, 1) : v.cols(src, 1);
        }
        if (name.contains(".conv1d")) {
            return v.rows(ElementTransforms.squeezeConv(src), 2 * qk, hv);
        }
        if (name.contains(".out_proj.")) {
            if (hadamardFolds(ctx, name)) {
                markValueHeadsGrouped(ctx, out);
                return src;
            }
            return v.cols(src, hv);
        }
        return src;
    }

    /** {@code _hadamard_folds_tensor} (qwen.py:598-600): a manifest name and {@code name} agree up to a prefix. */
    private static boolean hadamardFolds(HfLoadContext ctx, String name) throws IOException {
        for (String folded : HfHadamardMetadata.foldedNames(ctx)) {
            if (name.endsWith(folded) || folded.endsWith(name)) {
                return true;
            }
        }
        return false;
    }

    /** Records {@code prism.hadamard.gdn_v_grouped} once per load (base.py:836-838). */
    private static void markValueHeadsGrouped(HfLoadContext ctx, TensorSink out) {
        AtomicBoolean recorded = ctx.state(V_GROUPED_STATE, AtomicBoolean.class, AtomicBoolean::new);
        if (recorded.compareAndSet(false, true)) {
            out.addKv(GgufKeys.PRISM_HADAMARD_GDN_V_GROUPED, GGUFValueType.BOOL, true);
            ctx.log("Hadamard: linear-attention out_proj kept in grouped V order");
        }
    }

    /**
     * {@code _reorder_v_heads} (qwen.py:476-501) for {@code nK} key heads with {@code perK} value heads each: tiled
     * unit {@code u = v * nK + k} comes from grouped unit {@code k * perK + v}.
     */
    private static final class ValueHeads {
        private final String name;
        private final int nV;
        private final int[] order;

        ValueHeads(String name, int nK, int perK) {
            this.name = name;
            this.nV = nK * perK;
            this.order = new int[nV];
            for (int u = 0; u < nV; u++) {
                order[u] = (u % nK) * perK + u / nK;
            }
        }

        /**
         * Dim 0 from entry {@code from} on, {@code nV} heads of {@code headDim} entries, reordered; the stored rows
         * stay traceable for online quantization, which tiles in vLLM's stored order.
         */
        RowSource rows(RowSource src, long from, int headDim) {
            long[] shape = src.shape();
            long n = (long) nV * headDim;
            if (shape.length == 0 || shape[0] - from != n) {
                throw new GgufFormatException("tensor '" + name + "' of shape " + Arrays.toString(shape)
                        + " does not hold " + nV + " value heads of " + headDim + " after " + from + " entries");
            }
            return RowTransforms.permuteRowGroups(src, from, headDim, order);
        }

        /** The innermost dimension, {@code nV} groups of {@code groupWidth} columns, reordered. */
        RowSource cols(RowSource src, int groupWidth) {
            long[] shape = src.shape();
            if (shape.length < 2 || shape[shape.length - 1] != (long) nV * groupWidth) {
                throw new GgufFormatException("tensor '" + name + "' of shape " + Arrays.toString(shape)
                        + " does not hold " + nV + " value heads of " + groupWidth + " in its last dimension");
            }
            return ColumnTransforms.permuteColGroups(src, groupWidth, order);
        }
    }

    // ------------------------------------------------------------------ Qwen3.5

    /**
     * {@code _Qwen35MRopeMixin.set_gguf_parameters}' own part (qwen.py:668-671), run after its {@code super()}: the
     * rope sections {@code [11, 11, 10, 0]} when {@code rope_parameters} has no {@code mrope_section}.
     */
    public static void mropeSections(HfLoadContext ctx, HfMetadataBuilder kv) {
        if (!ctx.config().rope().raw().containsKey("mrope_section")) {
            kv.set(GgufKeys.ROPE_DIMENSION_SECTIONS, DEFAULT_MROPE_SECTIONS);
        }
    }

    /**
     * {@code Qwen3_5TextModel.set_gguf_parameters} (qwen.py:688-690 over the mixin): {@link #qwen3NextMeta},
     * {@link #mropeSections}, then the pooling type ({@link HfQwenHooks#pooling}). The inner
     * {@value HfQwenHooks#QWEN35_TEXT_MODEL} keeps no MTP layers ({@link HfQwenHooks#includeMtp}).
     */
    public static void qwen35Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        qwen3NextMeta(ctx, kv, plan);
        mropeSections(ctx, kv);
        HfQwenHooks.pooling(ctx, kv);
    }

    /** {@code Qwen3_5MoeTextModel}'s metadata (qwen.py:749-752): {@link #qwen3NextMeta}, {@link #mropeSections}. */
    public static void qwen35MoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        qwen3NextMeta(ctx, kv, plan);
        mropeSections(ctx, kv);
    }
}
