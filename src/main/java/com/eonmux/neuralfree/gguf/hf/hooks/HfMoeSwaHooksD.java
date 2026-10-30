package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;

/**
 * Hooks of the sliding-window / sink MoE converter classes, part D: {@code Step35Model}
 * (LC/conversion/step3.py:102-341, with the llama3 rope factors step3.py:300-341) and {@code MuseSparkModel}
 * (muse_spark.py:39-182, with the stored-minus-one norms, the Q/K unpermute, the router multiplier and the
 * residual gate α/β synthesis). The MimoV2/Beam classes are in {@link HfMoeSwaHooksB}, the Laguna/Maple/Mellum,
 * ExaoneMoE and SmallThinker classes in {@link HfMoeSwaHooksC}.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}.</p>
 */
public final class HfMoeSwaHooksD {

    /** The floor of the squared stream coefficient, muse_spark.py:14. */
    private static final float RESIDUAL_GATE_EPS = 1e-3f;

    /** muse_spark.py:18-24: the block norms that store a delta around one. */
    private static final List<String> OFFSET_NORM_SUFFIXES = Collections.unmodifiableList(Arrays.asList(
            "input_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
            "pre_expert_norm.weight", "post_expert_norm.weight"));

    /** The muse residual gate tensor, post-attention or post-ffn (muse_spark.py:134-149). */
    private static final java.util.regex.Pattern RESIDUAL_GATE =
            java.util.regex.Pattern.compile(".*\\w+_residual_gate\\.gate$");

    /** The tensors of step3.py:284-286 that are squeezed to their layout shape. */
    private static final java.util.Set<String> STEP35_SQUEEZE = Collections.unmodifiableSet(
            new java.util.LinkedHashSet<>(Arrays.asList(".self_attn.g_proj.weight", ".moe.gate.weight",
                    ".moe.up_proj.weight", ".moe.gate_proj.weight", ".moe.down_proj.weight")));

    private HfMoeSwaHooksD() {
    }

    // ------------------------------------------------------------------ Step35Model

    /**
     * {@code Step35Model.__init__}'s block count (step3.py:124-131): the trunk plus
     * {@code num_nextn_predict_layers} MTP layers, the trunk alone for a {@code --no-mtp} load.
     */
    public static int step35Blocks(HfLoadContext ctx) {
        return ctx.config().blockCount() + cfgNextn(ctx);
    }

    private static int cfgNextn(HfLoadContext ctx) {
        return HfQwenHooks.includeMtp(ctx) ? ctx.config().findInt(0, "num_nextn_predict_layers") : 0;
    }

    /**
     * {@code Step35Model.set_gguf_parameters} (step3.py:138-247) after the generic metadata: the per-layer head and
     * KV-head arrays over {@code layer_types} (the {@code attention_other_setting} widths on sliding layers), the
     * sliding window and pattern, the value width of {@code head_dim}, the MoE counts, the dense lead and the
     * every-n-layer step, the RMS epsilon, the per-layer SwiGLU clamps (MTP entries padded with 0.0), NextN and the
     * local RoPE theta of the {@code rope_theta} pair.
     *
     * @throws GgufFormatException when a required value is missing
     */
    public static void step35Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int blocks = ctx.blockCount();
        List<String> types = stringOf(padArray(cfg, "layer_types", blocks, "full_attention"));
        int nHeadBase = cfg.requireInt("num_attention_heads");
        int nKvBase = cfg.requireInt("num_attention_groups");
        Object otherRaw = cfg.find("attention_other_setting").orElse(null);
        int nHeadSwa = otherRaw instanceof Map && ((Map<?, ?>) otherRaw).get("num_attention_heads") instanceof Number
                ? (int) ((Number) ((Map<?, ?>) otherRaw).get("num_attention_heads")).doubleValue() : nHeadBase;
        int nKvSwa = otherRaw instanceof Map && ((Map<?, ?>) otherRaw).get("num_attention_groups") instanceof Number
                ? (int) ((Number) ((Map<?, ?>) otherRaw).get("num_attention_groups")).doubleValue() : nKvBase;
        List<Integer> heads = new ArrayList<>(blocks);
        List<Integer> kvs = new ArrayList<>(blocks);
        List<Boolean> swaPat = new ArrayList<>(blocks);
        for (int i = 0; i < blocks; i++) {
            boolean sliding = "sliding_attention".equals(types.get(i));
            heads.add(sliding ? nHeadSwa : nHeadBase);
            kvs.add(sliding ? nKvSwa : nKvBase);
            swaPat.add(sliding);
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, kvs);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, swaPat);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, cfg.requireInt("head_dim"));
        kv.set(GgufKeys.EXPERT_COUNT, cfg.requireInt("moe_num_experts"));
        kv.set(GgufKeys.EXPERT_USED_COUNT, cfg.requireInt("moe_top_k"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, cfg.requireInt("share_expert_dim"));
        Object scale = cfg.find("moe_router_scaling_factor").orElse(null);
        if (scale instanceof Number) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, ((Number) scale).doubleValue());
        }
        Object norm = cfg.find("norm_expert_weight").orElse(null);
        if (norm instanceof Boolean) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, norm);
        }
        int lead = leadingDense(cfg);
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, lead);
        kv.set(GgufKeys.MOE_EVERY_N_LAYERS, cfg.findInt(1, "moe_every_n_layer"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, cfg.findFloat(1e-5, "rms_norm_eps"));
        clampArray(cfg, kv, "swiglu_limits", GgufKeys.SWIGLU_CLAMP_EXP, blocks);
        clampArray(cfg, kv, "swiglu_limits_shared", GgufKeys.SWIGLU_CLAMP_SHEXP, blocks);
        int nextn = cfgNextn(ctx);
        if (nextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn);
        }
        Object localTheta = cfg.find("local_rope_theta").orElse(null);
        if (localTheta instanceof Number) {
            kv.set(GgufKeys.ROPE_FREQ_BASE_SWA, ((Number) localTheta).doubleValue());
        }
    }

    /** The leading dense layer of {@code moe_layers_enum} (step3.py:222-231), else 0. */
    private static int leadingDense(HfConfig cfg) {
        Object moeLayers = cfg.find("moe_layers_enum").orElse(null);
        if (moeLayers instanceof String) {
            for (String part : ((String) moeLayers).trim().split(",")) {
                if (!part.trim().isEmpty()) {
                    return Math.max(0, Integer.parseInt(part.trim()));
                }
            }
        }
        return 0;
    }

    /** {@code _pad(arr, n, default)} over the raw config list values with {@code None} entries replaced. */
    private static List<Object> padArray(HfConfig cfg, String name, int n, Object dflt) {
        Object v = cfg.find(name).orElse(null);
        List<Object> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Object o = v instanceof List && i < ((List<?>) v).size() ? ((List<?>) v).get(i) : dflt;
            out.add(o == null ? dflt : o);
        }
        return out;
    }

    private static List<String> stringOf(List<Object> raw) {
        List<String> out = new ArrayList<>(raw.size());
        for (Object v : raw) {
            out.add(String.valueOf(v));
        }
        return out;
    }

    private static List<Double> doubleListOf(List<Object> raw) {
        List<Double> out = new ArrayList<>(raw.size());
        for (Object v : raw) {
            if (!(v instanceof Number)) {
                throw new GgufFormatException("numeric entry required, got " + v);
            }
            out.add(((Number) v).doubleValue());
        }
        return out;
    }

    /** The per-layer SwiGLU clamp list of a padded config array ({@code null} entries become 0.0, step3.py:231-241). */
    private static void clampArray(HfConfig cfg, HfMetadataBuilder kv, String name, String key, int blocks) {
        kv.set(key, doubleListOf(padArray(cfg, name, blocks, 0.0)));
    }

    /**
     * {@code Step35Model.filter_tensors} (step3.py:250-278): the router bias tensor gains {@code .bias}; the
     * per-MTP-layer shared-head names lose the {@code transformer.} infix and rename {@code shared_head.output} to
     * {@code shared_head.head}.
     */
    public static final HfNameFilter STEP35_FILTER = (name, base) -> {
        String t = base.apply(name);
        if (t == null) {
            return null;
        }
        if (t.endsWith(".moe.router_bias")) {
            return t + ".bias";
        }
        return t.replace(".transformer.", ".").replace("shared_head.output", "shared_head.head");
    };

    /**
     * {@code Step35Model.modify_tensors} (step3.py:281-291): the stored-minus-one norm weights gain {@code +1}; the
     * gate/g_proj routing tensors of size-0 dims are squeezed; everything else maps.
     */
    public static void step35Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data = src;
        if (hfName.endsWith("norm.weight")) {
            data = ElementTransforms.addScalar(data, 1.0f);
        }
        for (String suffix : STEP35_SQUEEZE) {
            if (hfName.endsWith(suffix)) {
                data = ElementTransforms.squeezeConv(data);
                break;
            }
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /**
     * {@code Step35Model.generate_extra_tensors} (step3.py:300-341): the llama3 rope factors of the full-attention
     * rotary width, padded with 1.0 up to half the storage head width so the sliding-attention layers stay neutral.
     */
    public static void step35Extra(HfLoadContext ctx, TensorSink out) {
        HfConfig cfg = ctx.config();
        String ropeType = cfg.rope().global().type().orElse(null);
        if (ropeType == null || !"llama3".equalsIgnoreCase(ropeType)) {
            return;
        }
        int storage = cfg.findInt(0, "head_dim");
        if (storage <= 0) {
            storage = Math.floorDiv(cfg.requireInt("hidden_size"), cfg.requireInt("num_attention_heads"));
        }
        float[] freqs = llama3Factors(cfg, storage);
        out.add(ctx.names().ggufName("ROPE_FREQS", -1, ".weight"), new HfMoeSwaHooksB.FloatArraySource(freqs));
    }

    /**
     * The llama3 rope factors (step3.py:305-339), one per rotary pair computed from the checkpoint's rope theta pair
     * and the llama3 scaling parameters, padded to half the storage width.
     */
    private static float[] llama3Factors(HfConfig cfg, int storage) {
        double theta = cfg.findFloat(10000.0, "rope_theta");
        double factor = cfg.findFloat(8.0, "factor");
        double lowFreq = cfg.findFloat(1.0, "low_freq_factor");
        double highFreq = cfg.findFloat(4.0, "high_freq_factor");
        long oldCtx = cfg.findLong(8192L, "original_max_position_embeddings");
        double lowWavelen = oldCtx / lowFreq;
        double highWavelen = oldCtx / highFreq;
        int rotary = storage / 2;
        float[] out = new float[rotary];
        for (int i = 0; i < rotary; i++) {
            double freq = 1.0 / Math.pow(theta, 2.0 * i / (double) (2 * rotary));
            double wavelen = 2 * Math.PI / freq;
            double value;
            if (wavelen < highWavelen) {
                value = 1.0;
            } else if (wavelen > lowWavelen) {
                value = factor;
            } else {
                double s = (oldCtx / wavelen - lowFreq) / (highFreq - lowFreq);
                value = 1.0 / ((1.0 - s) / factor + s);
            }
            out[i] = (float) value;
        }
        return out;
    }

    // ------------------------------------------------------------------ MuseSparkModel

    /**
     * {@code MuseSparkModel.set_gguf_parameters} (muse_spark.py:53-83) after the generic metadata: the output
     * multiplier as logit scale, the final softcap, the post-norm epsilon, the sliding window with the
     * {@code layer_rope_theta} check, the SWA rope theta, the latent MoE sizes, the residual routing scale with
     * normalized weights and the sigmoid gating the reference hardcodes.
     */
    public static void museSparkMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        requiredDouble(kv, cfg, "output_multiplier", GgufKeys.LOGIT_SCALE);
        requiredDouble(kv, cfg, "final_logit_softcapping", GgufKeys.FINAL_LOGIT_SOFTCAPPING);
        requiredDouble(kv, cfg, "post_norm_eps", GgufKeys.POST_NORM_EPS);
        required(kv, cfg, "sliding_window", GgufKeys.ATTENTION_SLIDING_WINDOW);
        List<String> layerTypes = HfMoeSwaHooksB.stringList(cfg, "layer_types");
        double theta = cfg.findFloat(10000.0, "rope_theta");
        List<Object> thetaArr = cfg.find("layer_rope_theta").orElse(Collections.emptyList()) instanceof List
                ? (List<Object>) cfg.find("layer_rope_theta").orElse(Collections.emptyList()) : Collections.emptyList();
        List<Boolean> pattern = new ArrayList<>();
        for (String t : layerTypes) {
            pattern.add("sliding_attention".equals(t));
        }
        if (thetaArr.size() == pattern.size()) {
            for (int i = 0; i < thetaArr.size(); i++) {
                double expected = pattern.get(i) ? theta : 0.0;
                double actual = thetaArr.get(i) instanceof Number ? ((Number) thetaArr.get(i)).doubleValue() : -1;
                if (actual != expected) {
                    throw new GgufFormatException(cfg.source() + ": layer_rope_theta[" + i + "] does not match"
                            + " the layer_types entry (" + actual + " != " + expected + ")");
                }
            }
        }
        for (String t : HfMoeSwaHooksB.stringList(cfg, "mlp_layer_types")) {
            if (!"sparse".equals(t)) {
                throw new GgufFormatException(cfg.source() + ": MuseSpark layers are sparse-only, got " + t);
            }
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        kv.set(GgufKeys.ROPE_FREQ_BASE_SWA, theta);
        required(kv, cfg, "moe_hidden_size", GgufKeys.MOE_LATENT_SIZE);
        required(kv, cfg, "moe_intermediate_size", GgufKeys.EXPERT_FEED_FORWARD_LENGTH);
        requiredDouble(kv, cfg, "routed_scaling_factor", GgufKeys.EXPERT_WEIGHTS_SCALE);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        HfMoeSwaHooksA.gatingSigmoid(kv);
    }

    /**
     * {@code MuseSparkModel.modify_tensors} (muse_spark.py:85-155): the stored-minus-one norms; the Q/K weights are
     * unpermuted from the rotate-half layout and a synth Q/K norm pair carries the MetaP scale; the fused expert
     * gate/up splits over the output dim; the router weight is scaled by the MetaP multiplier; the residual gate
     * synthesizes an α/β pair; everything else maps.
     */
    public static void museSparkTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int bid = HfLoadContext.bidOf(hfName);
        for (String suffix : OFFSET_NORM_SUFFIXES) {
            if (hfName.endsWith(suffix)) {
                // the norms hold a delta around one (muse_spark.py:15-24)
                out.add(ctx.names().mapOrThrow(hfName), HfHookSupport.normPlusOne(src));
                return;
            }
        }
        if (hfName.endsWith(".self_attn.q_proj.weight")) {
            int nHead = cfg.requireInt("num_attention_heads");
            int headDim = cfg.requireInt("head_dim");
            out.add(ctx.names().mapOrThrow(hfName), RowTransforms.rotateHalfUnpermute(src, nHead));
            float[] ones = new float[headDim];
            float[] scaled = new float[headDim];
            double qkScale = cfg.findFloat(1.0, "qk_scale_factor");
            Arrays.fill(ones, 1.0f);
            Arrays.fill(scaled, (float) qkScale);
            out.add(ctx.names().ggufName("ATTN_Q_NORM", bid, ".weight"), new FmGradeSource(scaled));
            out.add(ctx.names().ggufName("ATTN_K_NORM", bid, ".weight"), new FmGradeSource(ones));
            return;
        }
        if (hfName.endsWith(".self_attn.k_proj.weight")) {
            out.add(ctx.names().mapOrThrow(hfName),
                    RowTransforms.rotateHalfUnpermute(src, cfg.requireInt("num_key_value_heads")));
            return;
        }
        if (hfName.endsWith(".mlp.experts.gate_up_proj")) {
            long[] shape = src.shape();
            long nFf = shape[shape.length - 2] / 2;
            out.add(ctx.names().mapOrThrow(hfName.replace("gate_up_proj", "gate_proj.weight")),
                    RowTransforms.sliceDim(src, shape.length - 2, 0, nFf));
            out.add(ctx.names().mapOrThrow(hfName.replace("gate_up_proj", "up_proj.weight")),
                    RowTransforms.sliceDim(src, shape.length - 2, nFf, shape[shape.length - 2] - nFf));
            return;
        }
        if (hfName.endsWith(".mlp.experts.down_proj")) {
            out.add(ctx.names().mapOrThrow(hfName + ".weight"), src);
            return;
        }
        if (hfName.endsWith(".mlp.gate.weight")) {
            Object multRaw = cfg.find("output_multiplier").orElse(null);
            if (!(multRaw instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": 'output_multiplier' must be a number, got " + multRaw);
            }
            out.add(ctx.names().mapOrThrow(hfName),
                    ElementTransforms.mulScalar(src, (float) ((Number) multRaw).doubleValue()));
            return;
        }
        java.util.regex.Matcher gate = RESIDUAL_GATE.matcher(hfName);
        if (gate.matches()) {
            boolean attn = hfName.contains("post_attention");
            Object tempRaw = cfg.find("residual_gate_temperature").orElse(null);
            if (!(tempRaw instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": 'residual_gate_temperature' must be a number, got "
                        + tempRaw);
            }
            double temperature = ((Number) tempRaw).doubleValue();
            out.add(ctx.names().ggufName(attn ? "ATTN_RESID_ALPHA" : "FFN_RESID_ALPHA", bid, ".weight"),
                    elementwise(src, museAlpha(temperature)));
            out.add(ctx.names().ggufName(attn ? "ATTN_RESID_BETA" : "FFN_RESID_BETA", bid, ".weight"),
                    elementwise(src, museBeta(temperature)));
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code required} of a config number written as a double. */
    private static void requiredDouble(HfMetadataBuilder kv, HfConfig cfg, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a number, got " + v);
        }
        kv.set(key, ((Number) v).doubleValue());
    }

    /** {@code required} of a config number written as an int. */
    private static void required(HfMetadataBuilder kv, HfConfig cfg, String name, String key) {
        Object v = cfg.find(name).orElse(null);
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a number, got " + v);
        }
        kv.set(key, ((Number) v).intValue());
    }

    /** β = σ(w / temperature) of muse_spark.py:137-148. */
    private static DoubleUnaryOperator museBeta(double temperature) {
        return w -> 1.0 / (1.0 + Math.exp(-w / temperature));
    }

    /**
     * α = √(max(σ(−w / temperature) · (1 + β), ε)), the squared coefficient clamped to ε (muse_spark.py:137-148,
     * {@code RESIDUAL_GATE_EPS}).
     */
    private static DoubleUnaryOperator museAlpha(double temperature) {
        return w -> {
            double beta = 1.0 / (1.0 + Math.exp(-w / temperature));
            double alphaSq = (1.0 / (1.0 + Math.exp(w / temperature))) * (1.0 + beta);
            return Math.sqrt(Math.max(alphaSq, RESIDUAL_GATE_EPS));
        };
    }

    /** {@code src} through an F32 elementwise map (the gates are small per-element images). */
    private static RowSource elementwise(RowSource src, DoubleUnaryOperator fn) {
        RowSource f32 = ElementTransforms.convert(src, GGMLType.F32);
        return new ElementwiseF32(f32, fn);
    }

    /** The per-element F32 map over a converted source. */
    static final class ElementwiseF32 implements RowSource {
        private final RowSource src;
        private final DoubleUnaryOperator fn;
        private final long[] shape;

        ElementwiseF32(RowSource src, DoubleUnaryOperator fn) {
            this.src = src;
            this.fn = fn;
            this.shape = src.shape();
        }

        @Override
        public String describe() {
            return src.describe() + " elementwise";
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(GGMLType.F32);
        }

        @Override
        public long[] shape() {
            return shape;
        }

        @Override
        public long rows() {
            return src.rows();
        }

        @Override
        public long rowBytes() {
            return src.rowBytes() * 4;
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            dst.order(ByteOrder.LITTLE_ENDIAN);
            int perRow = (int) (shape.length == 0 ? 1 : shape[shape.length - 1]);
            ByteBuffer in = ByteBuffer.allocate((int) (src.rowBytes() * count)).order(ByteOrder.LITTLE_ENDIAN);
            src.readRows(first, count, in);
            in.flip();
            for (long i = 0; i < (long) perRow * count; i++) {
                dst.putFloat((float) fn.applyAsDouble(in.getFloat()));
            }
        }
    }

    /** The Q/K norm rows the MetaP scale synthesizes, F32 in-memory. */
    private static final class FmGradeSource implements RowSource {
        private final float[] values;

        FmGradeSource(float[] values) {
            this.values = values;
        }

        @Override
        public String describe() {
            return "hook f32 norm(" + values.length + ")";
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(GGMLType.F32);
        }

        @Override
        public long[] shape() {
            return new long[]{values.length};
        }

        @Override
        public long rows() {
            return 1;
        }

        @Override
        public long rowBytes() {
            return 4L * values.length;
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) {
            dst.order(ByteOrder.LITTLE_ENDIAN);
            for (float v : values) {
                dst.putFloat(v);
            }
        }
    }

    /** {@code MuseSparkModel.set_vocab} (muse_spark.py:46-51): the GPT-2 vocabulary with the {@code <|eot|>} token. */
    public static Map<String, Object> museSparkVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new java.util.LinkedHashMap<>(
                HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre));
        int eot = HfMoeSwaHooksB.addedTokenId(ctx, "<|eot|>");
        if (eot >= 0) {
            md.put(GgufKeys.TOKENIZER_EOT_ID, eot);
        }
        return Collections.unmodifiableMap(md);
    }
}
