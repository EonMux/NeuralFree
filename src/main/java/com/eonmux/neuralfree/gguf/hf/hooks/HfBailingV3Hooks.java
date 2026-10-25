package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The hooks of {@code BailingMoeV3Model} and its VL text variant (LC/conversion/bailingmoe3.py:19-232), whose
 * architecture the converter calls KDA + MLA: the per-layer KV head array from {@code layer_group_size};
 * {@code A_log} to {@code exp(A_log)} — NOT negated — reshaped to a column; the KDA conv reshape; {@code dt_bias}
 * to {@code dt_proj.bias}; {@code f_proj} to {@code ssm_f_a} and {@code g_proj} to {@code attn_gate} or
 * {@code ssm_g_a} chosen by {@code is_full_attention}; the per-layer expert stacking and the MLA {@code kv_b}
 * split. The kept MTP layers plan under their own layer numbers after the main layers; the VL variant validates
 * and writes the 4-section {@code mrope_section}. Vocabulary: GPT-2 BPE.
 */
public final class HfBailingV3Hooks {

    /** The per-expert parts merged in the LC order (bailingmoe3.py:165). */
    static final List<String> EXPERT_PARTS =
            Collections.unmodifiableList(Arrays.asList("down_proj", "gate_proj", "up_proj"));

    private static final String EXPERTS_STATE = "bailing3.experts";
    private static final Pattern MODEL_LAYER = Pattern.compile("model\\.layers\\.(\\d+)\\.");
    private static final String WEIGHT = ".weight";

    private HfBailingV3Hooks() {
    }

    /** A resolver to {@link Arch#BAILINGMOE3}. */
    public static HfArchResolver arch() {
        return HfArchResolver.of(Arch.BAILINGMOE3);
    }

    /**
     * The block count (bailingmoe3.py:26-33): {@code num_hidden_layers} plus {@code num_nextn_predict_layers} when
     * the MTP layers are kept (the converter's {@code --no-mtp} zeroes them).
     */
    public static int blockCount(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        long base = cfg.requireInt("num_hidden_layers");
        long nextn = ctx.options().includeMtp() ? cfg.findInt(0, "num_nextn_predict_layers") : 0;
        if (base + nextn > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": " + base + " layers and " + nextn
                    + " MTP layers exceed " + HfConfig.MAX_LAYERS);
        }
        return (int) (base + nextn);
    }

    /**
     * {@code BailingMoeV3Model.filter_tensors} (bailingmoe3.py:112-131): {@code .expert_bias} gains its
     * {@code .bias} tail before the inherited filters; MTP layers are dropped when the MTP layers are not kept.
     */
    public static HfNameFilter filter(HfLoadContext ctx) {
        boolean keep = ctx.options().includeMtp();
        int main = ctx.config().requireInt("num_hidden_layers");
        return (name, base) -> {
            String t = base.apply(expertBiasBias(name));
            if (t == null) {
                return null;
            }
            Matcher m = MODEL_LAYER.matcher(t);
            if (m != null && Integer.parseInt(m.group(1)) >= main && !keep) {
                return null;
            }
            return t;
        };
    }

    /** {@code bailingmoe3.py:114-116}: the router bias renamed to map into the manager's name shape. */
    static String expertBiasBias(String name) {
        return name.endsWith(".expert_bias")
                ? HfHookSupport.renameTail(name, ".expert_bias", ".expert_bias.bias") : name;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code BailingMoeV3Model.set_gguf_parameters} (bailingmoe3.py:46-97). It refuses KDA LoRA projections,
     * non-safe gates and element-wise attention gates, and forces the MLA KV head count to 1.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        if (!truthy(cfg.find("no_kda_lora"))) {
            throw new GgufFormatException(cfg.source() + ": BailingMoeV3 KDA LoRA projections are not supported;"
                    + " config 'no_kda_lora' must be true");
        }
        if (!truthy(cfg.find("kda_safe_gate"))) {
            throw new GgufFormatException(cfg.source() + ": BailingMoeV3 non-safe KDA gates are not supported;"
                    + " config 'kda_safe_gate' must be true");
        }
        if (!"head_wise".equals(cfg.findStringOr("", "gated_attention_proj_granularity_type"))) {
            throw new GgufFormatException(cfg.source() + ": BailingMoeV3 requires head-wise attention gates"
                    + " (gated_attention_proj_granularity_type 'head_wise')");
        }
        HfGenericMetadata.apply(ctx, kv);
        oneKvHead(kv);
        kvHeadPerLayer(ctx, kv);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.SSM_CONV_KERNEL, cfg.requireInt("short_conv_kernel_size"));
        kv.set(GgufKeys.KDA_HEAD_DIM, cfg.requireInt("head_dim"));
        kv.set(GgufKeys.KDA_SAFE_GATE, true);
        kv.set(GgufKeys.KDA_GATE_LOWER_BOUND, requiredDouble(cfg, "kda_lower_bound"));
        int kvLora = cfg.requireInt("kv_lora_rank");
        OptionalInt qLora = cfg.findInt("q_lora_rank");
        if (qLora.isPresent()) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLora.getAsInt());
        }
        int nope = cfg.requireInt("qk_nope_head_dim");
        int rope = cfg.requireInt("qk_rope_head_dim");
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, rope);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLora + rope);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, nope + rope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, cfg.requireInt("v_head_dim"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, cfg.requireInt("moe_shared_expert_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.findInt(1, "num_shared_experts"));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, requiredDouble(cfg, "routed_scaling_factor"));
        if (truthy(cfg.find("norm_topk_prob"))) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
        List<Double> clampExp = clampList(ctx, "expert_swiglu_limit_list");
        if (clampExp != null) {
            kv.set(GgufKeys.SWIGLU_CLAMP_EXP, clampExp);
        }
        List<Double> clampShexp = clampList(ctx, "share_expert_swiglu_limit_list");
        if (clampShexp != null) {
            kv.set(GgufKeys.SWIGLU_CLAMP_SHEXP, clampShexp);
        }
        long nextn = ctx.options().includeMtp() ? cfg.findInt(0, "num_nextn_predict_layers") : 0;
        if (nextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, (int) nextn);
        }
    }

    /**
     * The VL variant's own metadata (bailingmoe3.py:211-222): {@code mrope_section} counts rope pairs, so its first
     * three entries must add up to {@code qk_rope_head_dim / 2}; sections become {@code [t, h, w, 0]}.
     */
    public static void vlMeta(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        List<?> mrope = cfg.find("mrope_section").orElse(null) instanceof List
                ? (List<?>) cfg.find("mrope_section").get() : null;
        if (mrope == null || mrope.size() < 3) {
            throw new GgufFormatException(cfg.source() + ": BailingMoeV3VL requires mrope_section in the config");
        }
        int sum = 0;
        List<Integer> sections = new ArrayList<>(4);
        for (int i = 0; i < 3; i++) {
            int s = intOf(cfg, mrope.get(i), "mrope_section[" + i + "]");
            sum += s;
            sections.add(s);
        }
        int ropePairs = cfg.requireInt("qk_rope_head_dim");
        if (2L * sum != ropePairs) {
            throw new GgufFormatException(cfg.source() + ": mrope_section " + sections + " counts rope pairs and"
                    + " must sum to qk_rope_head_dim / 2 = " + (ropePairs / 2));
        }
        sections.add(0);
        kv.set(GgufKeys.ROPE_DIMENSION_SECTIONS, sections);
    }

    /** {@code BailingMoeV3Model.is_full_attention} (bailingmoe3.py:41-44). */
    static boolean isFullAttention(HfConfig cfg, int bid) {
        int nLayer = cfg.requireInt("num_hidden_layers");
        int group = cfg.requireInt("layer_group_size");
        return bid >= nLayer || (bid + 1) % group == 0 || bid >= nLayer / group * group;
    }

    /** {@code bailingmoe3.py:57}: one KV head on full-attention layers, none for the KDA layers. */
    private static void kvHeadPerLayer(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        List<Integer> arr = new ArrayList<>(ctx.blockCount());
        for (int il = 0; il < ctx.blockCount(); il++) {
            arr.add(isFullAttention(cfg, il) ? 1 : 0);
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, arr);
    }

    /**
     * {@code clamp_limits} (bailingmoe3.py:84-89): the first {@code block_count} entries, {@code null}s folded to
     * 0.0 and the list padded to the block count; null when the configuration has no such list.
     */
    private static List<Double> clampList(HfLoadContext ctx, String key) {
        List<?> raw = ctx.config().find(key).orElse(null) instanceof List
                ? (List<?>) ctx.config().find(key).get() : null;
        if (raw == null) {
            return null;
        }
        List<Double> out = new ArrayList<>(ctx.blockCount());
        for (int i = 0; i < ctx.blockCount(); i++) {
            if (i < raw.size() && raw.get(i) != null) {
                Object v = raw.get(i);
                if (!(v instanceof Number) || v instanceof Boolean) {
                    throw new GgufFormatException(ctx.config().source() + ": '" + key + "[" + i + "]' must be a"
                            + " number, got " + v);
                }
                out.add(((Number) v).doubleValue());
            } else {
                out.add(0.0);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ tensors

    /** {@code BailingMoeV3Model.modify_tensors} (bailingmoe3.py:133-188). */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        String name = hfName;
        RowSource s = src;
        if (name.endsWith(".q_conv1d.weight") || name.endsWith(".k_conv1d.weight")
                || name.endsWith(".v_conv1d.weight")) {
            s = HfKdaHooks.convWeight(s, name);
        } else if (name.endsWith(".A_log")) {
            // exp, NOT negated (bailingmoe3.py:139-140); the (-1, 1) reshape follows the converter
            RowSource e = ElementTransforms.exp(s);
            long n = 1;
            for (long d : e.shape()) {
                n = Math.multiplyExact(n, d);
            }
            s = ElementTransforms.reshape(e, n, 1);
        } else if (name.endsWith(".dt_bias")) {
            name = HfHookSupport.renameTail(name, ".dt_bias", ".dt_proj.bias");
        } else if (name.endsWith(".attention.f_proj.weight")) {
            int bid = HfHookSupport.layerOf(name);
            if (isFullAttention(cfg, bid)) {
                throw new GgufFormatException("unexpected f_proj on full-attention layer " + bid
                        + " (tensor '" + name + "')");
            }
            out.add(ctx.names().ggufName(TensorNames.SSM_F_A, bid, WEIGHT), s);
            return;
        } else if (name.endsWith(".attention.g_proj.weight")) {
            int bid = HfHookSupport.layerOf(name);
            String tensor = isFullAttention(cfg, bid) ? TensorNames.ATTN_GATE : TensorNames.SSM_G_A;
            out.add(ctx.names().ggufName(tensor, bid, WEIGHT), s);
            return;
        } else if (name.contains(".mlp.experts.")) {
            expert(ctx, cfg.requireInt("num_experts")).offer(ctx, name, out);
            return;
        } else if (name.endsWith(".attention.kv_b_proj.weight")) {
            splitKvB(ctx, name, s, out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, name, s, out);
    }

    /**
     * {@code bailingmoe3.py:174-186}: {@code attention.kv_b_proj} of shape {@code [n_head*(nope+v), kv_lora]} to
     * {@code attn_k_b [n_head, kv_lora, nope]} (transposed per head) and {@code attn_v_b [n_head, v, kv_lora]}.
     */
    private static void splitKvB(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape " + Arrays.toString(shape)
                    + "; kv_b_proj must have rank 2");
        }
        int nHead = cfg.requireInt("num_attention_heads");
        int vHead = cfg.requireInt("v_head_dim");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, qkNope, vHead, Math.toIntExact(shape[1]));
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new GgufFormatException("tensor '" + hfName + "' does not split into " + nHead + " heads of "
                    + qkNope + " key and " + vHead + " value rows: " + e.getMessage(), e);
        }
        int bid = HfHookSupport.layerOf(hfName);
        out.add(ctx.names().ggufName(TensorNames.ATTN_K_B, bid, WEIGHT), kvB[0]);
        out.add(ctx.names().ggufName(TensorNames.ATTN_V_B, bid, WEIGHT), kvB[1]);
    }

    /** {@code bailingmoe3.py:156-172}: one collector per load buffers the experts of each layer. */
    private static HfHookSupport.ExpertCollector expert(HfLoadContext ctx, int nExperts) {
        Experts e = ctx.state(EXPERTS_STATE, Experts.class, Experts::new);
        synchronized (e) {
            if (e.collector == null) {
                e.collector = new HfHookSupport.ExpertCollector(nExperts, EXPERT_PARTS,
                        HfHookSupport.MergedName.SAME_PREFIX);
            }
            return e.collector;
        }
    }

    /** The per-load expert collector of BailingMoeV3. */
    private static final class Experts {
        HfHookSupport.ExpertCollector collector;
    }

    /** {@code bailingmoe3.py:190-196}: every expert group must complete. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        Experts e = ctx.state(EXPERTS_STATE, Experts.class, Experts::new);
        synchronized (e) {
            if (e.collector != null) {
                e.collector.requireComplete();
            }
        }
    }

    // ------------------------------------------------------------------ vocabulary and helpers

    /** {@code BailingMoeV3Model.set_vocab} (bailingmoe3.py:38-39). */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    private static void oneKvHead(HfMetadataBuilder kv) {
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
    }

    /** Python truthiness of the configuration value. */
    private static boolean truthy(Optional<Object> v) {
        if (!v.isPresent()) {
            return false;
        }
        Object o = v.get();
        if (o instanceof Boolean) {
            return (Boolean) o;
        }
        if (o instanceof Number) {
            return ((Number) o).doubleValue() != 0.0;
        }
        if (o instanceof String) {
            return !((String) o).isEmpty();
        }
        return true;
    }

    private static int intOf(HfConfig cfg, Object v, String where) {
        if (v instanceof Integer || v instanceof Long) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException(cfg.source() + ": '" + where + "' must be an integer, got " + v);
    }

    private static double requiredDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing"));
    }
}
