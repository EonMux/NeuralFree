package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Hooks of the classic MoE converter classes, part C: {@code PhiMoeModel} (LC/conversion/phi.py:343-392),
 * {@code OlmoeModel} (olmo.py:74-124), the chunked experts of {@code GroveMoeModel} (grovemoe.py:15-109), the MoVA
 * value experts of {@code K2HorizonModel} (k2_horizon.py:17-105), the tiktoken HunYuan MoE classes (hunyuan.py:21-154,
 * 363-464) and the HYV3 NextN filters (hunyuan.py:365-464). The InternLM3, Janus/Sarashina and Qwen3-VL/Omni rows
 * continue in {@link HfMoeClassicHooksD}.
 */
public final class HfMoeClassicHooksC {

    /** The state keys the per-load hooks register their private holders under. */
    private static final String PHIMOE_STATE = HfMoeClassicHooksC.class.getName() + ".phimoe";
    private static final String OLMOE_STATE = HfMoeClassicHooksC.class.getName() + ".olmoe";
    private static final String GROVE_STATE = HfMoeClassicHooksC.class.getName() + ".grove";
    private static final String GROVE_CHUNK_STATE = HfMoeClassicHooksC.class.getName() + ".groveChunk";
    private static final String K2_STATE = HfMoeClassicHooksC.class.getName() + ".k2";
    private static final String K2_Mova_STATE = HfMoeClassicHooksC.class.getName() + ".k2Mova";
    private static final String HUNYUAN_STATE = HfMoeClassicHooksC.class.getName() + ".hunyuan";
    private static final String HYV3_STATE = HfMoeClassicHooksC.class.getName() + ".hyv3";

    private HfMoeClassicHooksC() {
    }

    // ------------------------------------------------------------------ PhiMoeModel, OlmoeModel

    /**
     * {@code PhiMoeModel.set_gguf_parameters} (phi.py:348-351) after {@code Phi3MiniModel.set_gguf_parameters}: the
     * used and total expert counts.
     */
    public static void phiMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfBaseDenseHooks.phi3Meta(ctx, kv, plan);
        kv.set(GgufKeys.EXPERT_USED_COUNT, ctx.config().requireInt("num_experts_per_tok", "num_experts_per_token"));
        kv.set(GgufKeys.EXPERT_COUNT, ctx.config().requireInt("num_local_experts", "num_experts"));
    }

    /** {@code PhiMoeModel.modify_tensors} (phi.py:353-383): {@code block_sparse_moe.experts.<x>.w{1,2,3}} stack. */
    public static void phiMoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("block_sparse_moe.experts")) {
            if (!HfMoeClassicHooksA.stackedExperts(ctx, PHIMOE_STATE,
                    ctx.config().requireInt("num_local_experts", "num_experts"),
                    HfMoeClassicHooksA.MIXTRAL_PARTS).offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not a PhiMoE expert weight"
                        + " (block_sparse_moe.experts.<x>.w1|w2|w3)");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code PhiMoeModel.prepare_tensors} (phi.py:385-392): leftover experts are invalid. */
    public static void phiMoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, PHIMOE_STATE, "block_sparse_moe.experts");
    }

    /**
     * {@code OlmoeModel.set_gguf_parameters} (olmo.py:77-79) after the generic metadata: the RMS epsilon set to the
     * constant 1e-5, overriding the generic value as the converter's later write does.
     */
    public static void olmoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, 1e-5);
    }

    /** {@code OlmoeModel.modify_tensors} (olmo.py:84-114, the Qwen2Moe copy without fused experts): per-expert
     * {@code mlp.experts.<x>.down|gate|up_proj} stack under the merged names. */
    public static void olmoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("experts")) {
            HfHookSupport.ExpertCollector experts = HfMoeClassicHooksA.stackedExperts(ctx, OLMOE_STATE,
                    ctx.config().requireInt("num_local_experts", "num_experts"), HfMoeClassicHooksA.GATE_UP_DOWN);
            if (!experts.offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code OlmoeModel.prepare_tensors} (olmo.py:117-124): leftover experts are invalid. */
    public static void olmoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, OLMOE_STATE, "experts");
    }

    // ------------------------------------------------------------------ GroveMoeModel

    /**
     * {@code GroveMoeModel.set_gguf_parameters} (grovemoe.py:18-28) after the generic metadata: the optional expert
     * feed-forward length, the chunk width ({@code head_dim} or the 128 default), the experts-per-group 2 and the
     * expert group scale 0.05, all constants of the modeling file the converter cites.
     */
    public static void groveMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        OptionalInt ff = cfg.findInt("moe_intermediate_size");
        if (ff.isPresent()) {
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, ff.getAsInt());
        }
        kv.set(GgufKeys.EXPERT_CHUNK_FEED_FORWARD_LENGTH, cfg.findInt(128, "head_dim"));
        kv.set(GgufKeys.EXPERTS_PER_GROUP, 2);
        kv.set(GgufKeys.EXPERT_GROUP_SCALE, 0.05);
    }

    /**
     * {@code GroveMoeModel.modify_tensors} (grovemoe.py:33-94): the unused {@code .expert_bias} is dropped, the
     * {@code chunk_experts.<x>.<part>} tensors of every layer stack into {@code n_experts // experts_per_group}
     * chunk experts after the plain {@code mlp.experts} group; everything else maps directly.
     */
    public static void groveMoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".expert_bias")) {
            out.drop(hfName, "unused router bias (grovemoe.py:34-36)");
            return;
        }
        if (hfName.contains("chunk_experts")) {
            offerStacked(ctx, GROVE_CHUNK_STATE, hfName, src, out, chunkExperts(ctx));
            return;
        }
        if (hfName.contains("experts")) {
            offerStacked(ctx, GROVE_STATE, hfName, src, out, rowExperts(ctx));
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    private static void offerStacked(HfLoadContext ctx, String state, String hfName, RowSource src, TensorSink out,
                                     HfHookSupport.ExpertCollector collector) {
        if (!collector.offer(ctx, hfName, out)) {
            throw new GgufFormatException("tensor '" + hfName + "' is not a GroveMoe expert weight"
                    + " (mlp.experts.<x>.down_proj|gate_proj|up_proj)");
        }
    }

    private static HfHookSupport.ExpertCollector rowExperts(HfLoadContext ctx) {
        return HfMoeClassicHooksA.stackedExperts(ctx, GROVE_STATE,
                ctx.config().requireInt("num_local_experts", "num_experts"), HfMoeClassicHooksA.GATE_UP_DOWN);
    }

    /** The chunk-expert count {@code num_local_experts // experts_per_group} (grovemoe.py:40). */
    private static HfHookSupport.ExpertCollector chunkExperts(HfLoadContext ctx) {
        int n = ctx.config().requireInt("num_local_experts", "num_experts");
        if (n % 2 != 0 || n < 2) {
            throw new GgufFormatException(ctx.config().source() + ": num_local_experts " + n + " does not split into"
                    + " experts_per_group 2 chunk experts");
        }
        return HfMoeClassicHooksA.stackedExperts(ctx, GROVE_CHUNK_STATE, n / 2, HfMoeClassicHooksA.GATE_UP_DOWN);
    }

    /** {@code GroveMoeModel.prepare_tensors} (grovemoe.py:96-109): leftover chunk experts and experts are invalid. */
    public static void groveMoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, GROVE_STATE, "experts");
        HfMoeClassicHooksA.requireOrphans(ctx, GROVE_CHUNK_STATE, "chunk_experts");
    }

    // ------------------------------------------------------------------ K2HorizonModel

    /**
     * {@code K2HorizonModel.set_gguf_parameters} (k2_horizon.py:22-61): the group-norm group count, the rope head
     * dimension, the router MoE keys from the config (the leading dense count as the prefix of {@code
     * mlp_only_layers} unless {@code num_dense_layers} names it), the MoVA value-expert counts, and the check that
     * {@code attention_gate_func} is absent or {@code softplus}.
     */
    public static void k2HorizonMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ATTENTION_GROUPNORM_GROUPS, cfg.findInt(1, "layernorm_num_groups"));
        OptionalInt ropeHead = cfg.findInt("rope_head_dim");
        if (ropeHead.isPresent()) {
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeHead.getAsInt());
        }
        if (cfg.findInt(0, "num_experts") > 0) {
            int nFfExp = cfg.requireInt("moe_intermediate_size");
            int nShared = cfg.findInt(0, "num_shared_experts");
            OptionalInt explicitDense = cfg.findInt("num_dense_layers");
            int nDense = explicitDense.orElseGet(() -> leadingDense(cfg));
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, nFfExp);
            kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, nDense);
            kv.set(GgufKeys.MOE_EVERY_N_LAYERS, cfg.findInt(1, "decoder_sparse_step"));
            kv.set(GgufKeys.EXPERT_SHARED_COUNT, nShared);
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_topk_prob"));
            if (nShared > 0) {
                kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, nFfExp * nShared);
            }
            OptionalDouble scale = cfg.findFloat("router_scaling_factor");
            if (scale.isPresent()) {
                kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, scale.getAsDouble());
            }
        }
        int nValueExpert = cfg.findInt(0, "mova_num_experts");
        int nValueUsed = cfg.findInt(0, "mova_num_experts_per_tok");
        if (nValueExpert > 0 && nValueUsed > 0) {
            if (nValueUsed > nValueExpert) {
                throw new GgufFormatException(cfg.source() + ": mova_num_experts_per_tok <= mova_num_experts");
            }
            kv.set(GgufKeys.ATTENTION_VALUE_EXPERT_COUNT, nValueExpert);
            kv.set(GgufKeys.ATTENTION_VALUE_EXPERT_USED_COUNT, nValueUsed);
        }
        String gateFunc = cfg.findString("attention_gate_func").orElse(null);
        if (gateFunc != null && !gateFunc.equals("softplus")) {
            throw new GgufFormatException("Unsupported attention_gate_func: '" + gateFunc + "'");
        }
    }

    /** The prefix of {@code mlp_only_layers}: dense layers 0..n until the first MoE layer (k2_horizon.py:37-40). */
    private static int leadingDense(HfConfig cfg) {
        List<Integer> mlpOnly = cfg.findIntList(Collections.emptyList(), "mlp_only_layers");
        int n = 0;
        while (mlpOnly.contains(n)) {
            n++;
        }
        return n;
    }

    /**
     * {@code K2HorizonModel.modify_tensors} (k2_horizon.py:63-97): the router bias {@code .mlp.gate.bias} becomes
     * {@code FFN_EXP_PROBS_B} directly; the per-expert {@code mlp.experts.<x>.(down|gate|up)_proj.weight} stack; the
     * MoVA {@code self_attn.v_experts.<x>.weight} rows stack into the value-expert tensor; everything else maps.
     */
    public static void k2HorizonTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".mlp.gate.bias")) {
            int bid = HfHookSupport.layerOf(hfName);
            out.add(ctx.names().ggufName("FFN_EXP_PROBS_B", bid, ".bias"), src);
            return;
        }
        if (hfName.matches("model\\.layers\\.\\d+\\.mlp\\.experts\\.\\d+\\.(down|gate|up)_proj\\.weight")) {
            if (!HfMoeClassicHooksA.stackedExperts(ctx, K2_STATE, ctx.config().requireInt("num_experts"),
                    HfMoeClassicHooksA.GATE_UP_DOWN).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        if (hfName.matches("model\\.layers\\.\\d+\\.self_attn\\.v_experts\\.\\d+\\.weight")) {
            if (!valueExperts(ctx).offer(ctx, hfName, src, out)) {
                throw new GgufFormatException("Unprocessed value experts: ['" + hfName + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** The per-load MoVA value-expert stacker of k2_horizon.py:75-97: rows {@code v_experts.<x>.weight} stack dim 0
     * under the merged {@code v_experts.weight} name, planned when the layer's last member arrives. */
    private static VExpertStack valueExperts(HfLoadContext ctx) {
        int nExperts = ctx.config().requireInt("mova_num_experts");
        if (nExperts < 1) {
            throw new GgufFormatException(ctx.config().source() + ": 'mova_num_experts' must be positive");
        }
        return ctx.state(K2_Mova_STATE, VExpertStack.class, () -> new VExpertStack(nExperts));
    }

    /** The MoVA collector of k2_horizon.py:83-96; each layer's rows arrive once (planned tensors are unique). */
    private static final class VExpertStack {
        private final int nExperts;
        private final Map<Long, List<RowSource>> rows = new LinkedHashMap<>();

        VExpertStack(int nExperts) {
            this.nExperts = nExperts;
        }

        boolean offer(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
            long bid = HfLoadContext.bidOf(hfName);
            if (bid < 0 || hfName.contains("mlp.experts")) {
                return false;
            }
            List<RowSource> list = rows.computeIfAbsent(bid, k -> new ArrayList<>());
            if (list.size() >= nExperts) {
                return false;
            }
            list.add(src);
            if (list.size() == nExperts) {
                rows.remove(bid);
                out.add(ctx.names().mapOrThrow("model.layers." + bid + ".self_attn.v_experts.weight"),
                        ExpertTransforms.stack(list));
            }
            return true;
        }

        /** All planned experts were emitted (k2_horizon.py:98-105). */
        void requireComplete() {
            if (!rows.isEmpty()) {
                throw new GgufFormatException("Unprocessed value experts: " + rows.keySet());
            }
        }
    }

    /** {@code K2HorizonModel.prepare_tensors} (k2_horizon.py:98-105): leftover experts are invalid. */
    public static void k2HorizonExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, K2_STATE, "mlp.experts");
        ctx.state(K2_Mova_STATE, VExpertStack.class,
                () -> new VExpertStack(ctx.config().findInt(1, "mova_num_experts"))).requireComplete();
    }

    // ------------------------------------------------------------------ HunYuan V1 MoE and HYV3

    /**
     * {@code HunYuanMoEModel.set_gguf_parameters} (hunyuan.py:75-110) after the generic metadata: the shared width
     * ({@code intermediate_size}), the uniform per-layer {@code moe_intermediate_size}/{@code moe_topk}/
     * {@code num_shared_expert} lists, and the NTK-alpha rope when the type is dynamic. The uniformity and the
     * dynamic assumptions raise exactly like the converter's asserts.
     */
    public static void hunYuanMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int sharedFf = cfg.requireInt("intermediate_size");
        int moeFf = uniformFirst(cfg, "moe_intermediate_size");
        int moeTopk = uniformFirst(cfg, "moe_topk");
        int sharedCount = uniformFirst(cfg, "num_shared_expert");
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, sharedFf);
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, moeFf);
        kv.set(GgufKeys.EXPERT_USED_COUNT, moeTopk);
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, sharedCount);
        String ropeType = cfg.rope().global().type().orElse(null);
        if (!"dynamic".equals(ropeType)) {
            return;
        }
        double alpha = ropeGlobalNumber(cfg, "alpha", 1000.0);
        double base = ropeGlobalNumber(cfg, "rope_theta", 10000.0);
        int dim = Math.floorDiv(cfg.requireInt("hidden_size"), cfg.requireInt("num_attention_heads"));
        double scaledBase = base * Math.pow(alpha, (double) dim / (dim - 2));
        kv.set(GgufKeys.ROPE_FREQ_BASE, scaledBase);
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
        kv.set(GgufKeys.ROPE_SCALING_FACTOR, 1.0);
        kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, 256 * 1024);
        kv.set(GgufKeys.CONTEXT_LENGTH, 256 * 1024);
        Object maxPos = cfg.flat().get("max_position_embeddings");
        int pos = maxPos instanceof Number ? (int) ((Number) maxPos).doubleValue() : 0;
        if (alpha != 1000.0 || base != 10000.0 || dim != 128
                || (pos != 32 * 1024 && pos != 256 * 1024)) {
            throw new GgufFormatException("HunYuan dynamic RoPE scaling assumptions changed, please update the"
                    + " logic or context length manually");
        }
    }

    /** A global RoPE parameter read as a number, with the converter's default when absent. */
    private static double ropeGlobalNumber(HfConfig cfg, String key, double dflt) {
        Object v = cfg.rope().global().get(key).orElse(null);
        if (v == null) {
            return dflt;
        }
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": rope " + key + " must be a number, got "
                    + v.getClass().getSimpleName());
        }
        return ((Number) v).doubleValue();
    }

    /** One uniform per-layer list read as an integer: the first entry, after the uniformity assert of hunyuan.py. */
    private static int uniformFirst(HfConfig cfg, String key) {
        Optional<Object> v = cfg.find(key);
        if (!v.isPresent() || !(v.get() instanceof List) || ((List<?>) v.get()).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a non-empty per-layer list");
        }
        List<?> list = (List<?>) v.get();
        double first;
        Object at0 = list.get(0);
        if (!(at0 instanceof Number) || at0 instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must hold per-layer numbers");
        }
        first = ((Number) at0).doubleValue();
        for (Object entry : list) {
            if (!(entry instanceof Number) || entry instanceof Boolean
                    || ((Number) entry).doubleValue() != first) {
                throw new GgufFormatException(cfg.source() + ": '" + key + "' values are not uniform");
            }
        }
        long l = (long) first;
        if (l < 0 || l > Integer.MAX_VALUE) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' value " + l + " is outside int range");
        }
        return (int) l;
    }

    /**
     * {@code HunYuanMoEModel.modify_tensors} (hunyuan.py:114-147): the tied {@code lm_head.weight} dropped when
     * {@code tie_word_embeddings}; per-expert {@code mlp.experts.<x>.<part>} stack under the merged names.
     */
    public static void hunYuanMoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (hfName.equals("lm_head.weight") && cfg.findBool(false, "tie_word_embeddings")) {
            out.drop(hfName, "tied output layer (hunyuan.py:115-118)");
            return;
        }
        if (hfName.contains("mlp.experts")) {
            if (!HfMoeClassicHooksA.stackedExperts(ctx, HUNYUAN_STATE,
                    ctx.config().requireInt("num_local_experts", "num_experts"),
                    HfMoeClassicHooksA.GATE_UP_DOWN).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code HunYuanMoEModel.prepare_tensors} (hunyuan.py:149-154): leftover experts are invalid. */
    public static void hunYuanMoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, HUNYUAN_STATE, "mlp.experts");
    }

    /**
     * {@code HunYuanMoEModel.set_vocab} (hunyuan.py:24-73) ported to the tiktoken GPT-2 writer: the merge rebuild of
     * {@link HfVocabWriters#qwen}, then the fixed BOS {@code <|bos|>} id 127959 that overwrites the ruler mis-read
     * from config.json.
     */
    public static Map<String, Object> hunYuanMoeVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.qwen(HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_BOS_ID, 127959);
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ HYV3Model

    /**
     * {@code HYV3Model.set_gguf_parameters} (hunyuan.py:390-403) after the generic metadata: the MoE expert width,
     * the shared width {@code moe_intermediate_size * num_shared_experts} (default 1), the gated weight norm (default
     * true) and scale (default 1.0), the sigmoid router, and the NextN count exported to the model when
     * {@code num_nextn_predict_layers} is set.
     */
    public static void hyV3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int mFf = cfg.requireInt("moe_intermediate_size");
        int nShared = cfg.findInt(1, "num_shared_experts");
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, mFf);
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, mFf * nShared);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(true, "route_norm"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat(1.0f, "router_scaling_factor"));
        kv.set(GgufKeys.EXPERT_GATING_FUNC, 2); // gguf.ExpertGatingFuncType.SIGMOID
        OptionalInt nextn = cfg.findInt("num_nextn_predict_layers");
        if (nextn.isPresent() && nextn.getAsInt() > 0 && ctx.options().includeMtp()) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn.getAsInt());
        }
    }

    /**
     * {@code HYV3Model.filter_tensors} (hunyuan.py:405-430): the appended MTP block (layer ids >= the main count)
     * is dropped under {@code --no-mtp} ({@code includeMtp = false}), renamed from {@code final_layernorm} to
     * {@code shared_head.norm}, and kept under everything else. {@code --mtp-only} reorders which conventions exist:
     * it is not part of the model source options, so the MTP plus shared-embedding subset is not reconstructed here
     * and only the full drop applies.
     */
    public static HfNameFilter hyV3Filter(HfLoadContext ctx) {
        int nMain = ctx.config().requireInt("num_hidden_layers");
        boolean includeMtp = ctx.options().includeMtp();
        return (name, base) -> {
            int bid = HfLoadContext.bidOf(name);
            boolean mtpLayer = bid >= nMain && name.startsWith("model.layers.");
            if (mtpLayer && !includeMtp) {
                return null;
            }
            String renamed = mtpLayer ? name.replace(".final_layernorm.", ".shared_head.norm.") : name;
            return base.apply(renamed);
        };
    }

    /** The NextN extended block count, honored only when MTP layers are kept (hunyuan.py:374-381). */
    public static int hyV3BlockCount(HfLoadContext ctx) {
        int n = ctx.config().requireInt("num_hidden_layers");
        OptionalInt nextn = ctx.config().findInt("num_nextn_predict_layers");
        if (nextn.isPresent() && nextn.getAsInt() > 0 && ctx.options().includeMtp()) {
            return n + nextn.getAsInt();
        }
        return n;
    }

    /**
     * {@code HYV3Model.modify_tensors} (hunyuan.py:434-457): expert-first stacked naming, everything else mapped via
     * {@link HfTensorHook#DEFAULT}.
     */
    public static void hyV3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains(".mlp.experts.") && hfName.startsWith("model.layers.")) {
            if (!HfMoeClassicHooksA.stackedExperts(ctx, HYV3_STATE,
                    ctx.config().requireInt("num_local_experts", "num_experts"),
                    HfMoeClassicHooksA.GATE_UP_DOWN).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: ['" + hfName + "']");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code HYV3Model.prepare_tensors} (hunyuan.py:459-464): leftover experts are invalid. */
    public static void hyV3Extra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, HYV3_STATE, "mlp.experts");
    }
}
