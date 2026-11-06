package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfHookSupport;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * The Kimi-K3 part of the KDA converter classes (LC/conversion/kimi_k3.py:19-345), split from
 * {@link HfKdaHooks}: the MLA lengths of the nope-only MLA, the latent MoE, the situ activation and the
 * attention-residual block size; the attention-residual-score fusions
 * {@code <x>_res_norm.weight * <x>_res_proj.weight} (kimi_k3.py:40-44, :242-267), whose element product runs in
 * f32; the {@code g_proj} layer-type resolution to {@code ssm_g} on KDA layers and {@code attn_gate} on
 * full-attention layers over the 1-based {@code full_attn_layers} list (kimi_k3.py:301-306); the
 * compressed-tensors {@code mxfp4-pack-quantized} experts, which the NeuralFree quantization rewrite repacks to
 * ggml MXFP4; the configured {@code eos_token_id} written over the tokenizer's; and the bundled
 * {@value #K3_TEMPLATE} chat template. The shared KDA passthroughs, the expert stacking and the Kimi vocabulary
 * stay in {@link HfKdaHooks}.
 */
public final class HfKdaHooksB {

    /** The bundled Kimi-K3 chat template, added when the checkpoint has none (kimi_k3.py:76-81). */
    static final String K3_TEMPLATE = "Kimi-K3.jinja";

    private static final String WEIGHT = ".weight";
    private static final String RES_STATE = "kda.resParts";

    private HfKdaHooksB() {
    }

    /** A resolver to {@link Arch#KIMI_K3}. */
    public static HfArchResolver k3Arch() {
        return HfArchResolver.of(Arch.KIMI_K3);
    }

    /**
     * {@code KimiK3Model.set_gguf_parameters} (kimi_k3.py:165-227): the MLA lengths (K3 MLA is nope-only and its
     * {@code qk_rope_head_dim} still sizes the un-absorbed part of K), the latent MoE, the situ activation and the
     * attention-residual block size.
     */
    public static void k3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfKdaHooks.oneKvHead(kv);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        HfKdaHooks.kvHeadArray(ctx, kv);
        Map<String, ?> lin = HfKdaHooks.nested(cfg, "linear_attn_config");
        kv.set(GgufKeys.SSM_CONV_KERNEL, requiredMember(lin, "short_conv_kernel_size", cfg));
        kv.set(GgufKeys.KDA_HEAD_DIM, requiredMember(lin, "head_dim", cfg));
        Optional<Double> lower = memberDouble(lin, "gate_lower_bound");
        if (lower.isPresent()) {
            kv.set(GgufKeys.KDA_GATE_LOWER_BOUND, lower.get());
        }
        int kvLoraRank = cfg.requireInt("kv_lora_rank");
        OptionalInt qLora = cfg.findInt("q_lora_rank");
        if (qLora.isPresent()) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLora.getAsInt());
        }
        int nope = cfg.requireInt("qk_nope_head_dim");
        int rope = cfg.requireInt("qk_rope_head_dim");
        if (!Boolean.TRUE.equals(cfg.find("mla_use_nope").orElse(Boolean.FALSE))) {
            throw new GgufFormatException(cfg.source() + ": K3 MLA is expected to be nope-only (mla_use_nope)");
        }
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, rope);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLoraRank + rope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, kvLoraRank);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, nope + rope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, cfg.requireInt("v_head_dim"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("num_shared_experts"));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat(1.0, "routed_scaling_factor"));
        if (cfg.findBool(Boolean.TRUE, "moe_renormalize")) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
        if (!"sigmoid".equals(cfg.findStringOr("", "moe_router_activation_func"))) {
            throw new GgufFormatException(cfg.source() + ": Kimi-K3 requires moe_router_activation_func 'sigmoid'");
        }
        kv.set(GgufKeys.EXPERT_GATING_FUNC, 2);
        cfg.findInt("routed_expert_hidden_size").ifPresent(v -> kv.set(GgufKeys.EXPERT_LATENT_LENGTH, v));
        if (!"situ".equals(cfg.findStringOr("situ", "hidden_act"))) {
            throw new GgufFormatException(cfg.source() + ": Kimi-K3 requires hidden_act 'situ'");
        }
        kv.set(GgufKeys.ACTIVATION_SITU_BETA, requiredDouble(cfg, "activation_situ_beta"));
        kv.set(GgufKeys.ACTIVATION_SITU_LINEAR_BETA, requiredDouble(cfg, "activation_situ_linear_beta"));
        kv.set(GgufKeys.ATTN_RES_BLOCK_SIZE, cfg.requireInt("attn_res_block_size"));
    }

    /**
     * {@code KimiK3Model.modify_tensors} (kimi_k3.py:269-345): the residual-score fusion, then the shared KDA
     * passthrough, the {@code g_proj} layer-type resolution and the expert stacking.
     */
    public static void k3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (fusedResidual(ctx, hfName, src, out)) {
            return;
        }
        if (hfName.endsWith(".self_attn.g_proj.weight")) {
            int bid = HfHookSupport.layerOf(hfName);
            boolean isKda = !HfKdaHooks.fullAttnLayers(ctx.config()).contains(bid + 1);
            out.add(ctx.names().ggufName(isKda ? TensorNames.SSM_G : TensorNames.ATTN_GATE, bid, WEIGHT), src);
            return;
        }
        if (hfName.contains("block_sparse_moe.experts")) {
            HfKdaHooks.experts(ctx, ctx.config().requireInt("num_experts")).offer(ctx, hfName, out);
            return;
        }
        HfKdaHooks.kdaCommon(ctx, hfName, src, out);
    }

    /**
     * {@code KimiK3Model.prepare_tensors} (kimi_k3.py:229-240): every expert group completed and no unpaired
     * residual tensor.
     */
    public static void k3Extra(HfLoadContext ctx, TensorSink out) {
        HfKdaHooks.completeExperts(ctx);
        ResParts parts = ctx.state(RES_STATE, ResParts.class, ResParts::new);
        if (!parts.pending.isEmpty()) {
            throw new GgufFormatException("Kimi-K3: unpaired attention-residual tensors "
                    + new ArrayList<>(parts.pending.keySet()) + ": every <x>_res_norm needs its <x>_res_proj");
        }
    }

    /**
     * {@code KimiK3Model.set_vocab} (kimi_k3.py:63-81): the Kimi vocabulary, the configured {@code eos_token_id}
     * written over the tokenizer's, and the bundled {@value #K3_TEMPLATE} when the checkpoint has no chat template.
     */
    public static Map<String, Object> k3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfConfig cfg = ctx.config();
        OptionalInt eos = cfg.findInt("eos_token_id");
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx)
                .withFallbackChatTemplate(HfChatTemplates.bundled(K3_TEMPLATE));
        Map<String, Object> md = HfKdaHooks.kimiVocab(ctx, in, pre);
        if (eos.isPresent()) {
            md.put(GgufKeys.TOKENIZER_EOS_ID, eos.getAsInt());
        }
        return md;
    }

    // ------------------------------------------------------------------ K3 residual-score fusions

    /**
     * Pairs {@code <x>_res_norm.weight} with {@code <x>_res_proj.weight} and emits their element product at the
     * names _try_fuse_res writes (kimi_k3.py:242-267): per-layer {@code attn_res_score} and {@code ffn_res_score}
     * and the model-level {@code output_res_score}. Consumes the first-arriving half; returns false for any other
     * tensor.
     */
    private static boolean fusedResidual(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        ResParts parts = ctx.state(RES_STATE, ResParts.class, ResParts::new);
        for (ResKind k : ResKind.values()) {
            if (!k.isHalf(hfName)) {
                continue;
            }
            String key = k.key(hfName);
            ResPart other = parts.pending.remove(key);
            if (other == null) {
                parts.pending.put(key, new ResPart(src, k.isNormHalf(hfName)));
                return true;
            }
            RowSource norm = other.norm ? other.src : src;
            RowSource proj = other.norm ? src : other.src;
            String fusedName = k.ggufName(ctx, hfName);
            ctx.log("Kimi-K3: fused " + k.prefix() + "_norm * " + k.prefix() + "_proj -> " + fusedName);
            out.add(fusedName, ResidualProduct.of(norm, proj));
            return true;
        }
        return false;
    }

    /** One residual fusion group of {@code _RES_FUSIONS} (kimi_k3.py:40-44). */
    private enum ResKind {
        SELF_ATTN_RES(TensorNames.ATTN_RES_SCORE, true),
        MLP_RES(TensorNames.FFN_RES_SCORE, true),
        OUTPUT_ATTN_RES(TensorNames.OUTPUT_RES_SCORE, false);

        private final String tensor;
        private final boolean perLayer;

        ResKind(String tensor, boolean perLayer) {
            this.tensor = tensor;
            this.perLayer = perLayer;
        }

        /** True for {@code .self_attention_res_norm.weight} and {@code .self_attention_res_proj.weight}. */
        boolean isHalf(String hfName) {
            return tail(hfName, "norm") || tail(hfName, "proj");
        }

        private boolean tail(String hfName, String kind) {
            return hfName.endsWith("." + prefix() + "_" + kind + WEIGHT);
        }

        /** True for the {@code _norm} half, false for the {@code _proj} half. */
        boolean isNormHalf(String hfName) {
            return !hfName.endsWith("_proj" + WEIGHT);
        }

        String prefix() {
            switch (this) {
                case SELF_ATTN_RES: return "self_attention_res";
                case MLP_RES: return "mlp_res";
                default: return "output_attn_res";
            }
        }

        String key(String hfName) {
            int bid = HfLoadContext.bidOf(hfName);
            return bid < 0 || !perLayer ? prefix() : prefix() + "." + bid;
        }

        String ggufName(HfLoadContext ctx, String hfName) {
            if (!perLayer) {
                return TensorNames.weight(tensor);
            }
            return ctx.names().ggufName(tensor, HfHookSupport.layerOf(hfName), WEIGHT);
        }
    }

    /** The pending halves of one load's residual groups (kimi_k3.py's {@code _res_parts}). */
    private static final class ResParts {
        final Map<String, ResPart> pending = new LinkedHashMap<>();
    }

    /** A buffered residual half. */
    private static final class ResPart {
        final RowSource src;
        final boolean norm;

        ResPart(RowSource src, boolean norm) {
            this.src = src;
            this.norm = norm;
        }
    }

    /**
     * The elementwise product of two equal-sized tensors computed in f32 (kimi_k3.py:261
     * {@code norm.float().flatten() * proj.float().flatten()}): one row at a time, chunk-local, immutable.
     */
    static final class ResidualProduct implements RowSource {
        private final RowSource norm;
        private final RowSource proj;
        private final long[] shape;

        static RowSource of(RowSource a, RowSource b) {
            long[] sa = a.shape();
            long[] sb = b.shape();
            if (!Arrays.equals(sa, sb)) {
                throw new IllegalArgumentException("fused residual tensors' shapes differ: " + a.describe() + " of "
                        + Arrays.toString(sa) + " and " + b.describe() + " of " + Arrays.toString(sb));
            }
            return new ResidualProduct(ElementTransforms.convert(a, GGMLType.F32),
                    ElementTransforms.convert(b, GGMLType.F32), sa);
        }

        private ResidualProduct(RowSource norm, RowSource proj, long[] shape) {
            this.norm = norm;
            this.proj = proj;
            this.shape = shape;
        }

        @Override public String describe() {
            return norm.describe() + " * " + proj.describe() + " (Kimi-K3 residual score)";
        }

        @Override public ElementFormat format() {
            return ElementFormat.of(GGMLType.F32);
        }

        @Override public long[] shape() { return shape.clone(); }

        @Override public long rows() { return norm.rows(); }

        @Override public long rowBytes() {
            long inner = shape.length == 0 ? 1 : shape[shape.length - 1];
            return 4 * inner;
        }

        @Override public boolean rowLocal() { return norm.rowLocal() && proj.rowLocal(); }

        @Override public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            long inner = shape.length == 0 ? 1 : shape[shape.length - 1];
            int chunk = 1 << 12;
            for (int done = 0; done < count; ) {
                int c = Math.min(chunk, count - done);
                ByteBuffer a = ByteBuffer.allocate(Math.multiplyExact(c * 4, (int) inner))
                        .order(ByteOrder.LITTLE_ENDIAN);
                ByteBuffer b = ByteBuffer.allocate(a.capacity()).order(ByteOrder.LITTLE_ENDIAN);
                norm.readRows(first + done, c, a);
                proj.readRows(first + done, c, b);
                a.flip();
                b.flip();
                int n = a.remaining() / 4;
                FloatBuffer out = dst.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                for (int i = 0; i < n; i++) {
                    out.put(a.getFloat(i * 4) * b.getFloat(i * 4));
                }
                dst.position(dst.position() + n * 4);
                done += c;
            }
        }
    }

    // ------------------------------------------------------------------ K3 small helpers

    private static int requiredMember(Map<String, ?> map, String key, HfConfig cfg) {
        Object v = map.get(key);
        if (v instanceof Integer || v instanceof Long) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException(cfg.source() + ": 'linear_attn_config." + key + "' is missing or is not"
                + " an integer: " + v);
    }

    private static Optional<Double> memberDouble(Map<String, ?> map, String key) {
        Object v = map.get(key);
        return v instanceof Integer || v instanceof Long || v instanceof Double || v instanceof Float
                ? Optional.of(((Number) v).doubleValue()) : Optional.empty();
    }

    private static double requiredDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing"));
    }
}
