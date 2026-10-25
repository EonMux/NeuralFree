package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * The draft hooks of Appendix A table HK-SG part B: LC/conversion/deepseek.py {@code DeepseekV4DSparkModel}
 * (:929-1056) on the HK-B3 DeepSeek-V4 hooks, LC/conversion/nemotron.py {@code LiLiCorrDraftModel} (:19-117),
 * LC/conversion/gemma.py {@code Gemma4DSparkModel} (:836-950) and {@code Gemma4AssistantModel} (:957-961) on the
 * HK-B2 Gemma 4 hooks. DFlash, DSpark and MuseGlimmerAssistant are {@link HfDraftHooksA}; the shared target, mask
 * and {@code dflash_config} helpers live in {@link HfDraftHooks}.
 */
public final class HfDraftHooksB {

    private static final String WEIGHT = HfDraftHooks.WEIGHT;
    private static final String DFLASH_CONFIG = HfDraftHooks.DFLASH_CONFIG;
    private static final String ATTENTION_POST_NORM = "ATTN_POST_NORM";
    private static final String FFN_NORM = "FFN_NORM";
    private static final String RELATIVE_SLOT_BIAS = "model.lilicorr.relative_slot_bias";
    private static final String SAME_SLOT_BIAS = "model.lilicorr.same_slot_bias";
    private static final String LILICORR_PREFIX = "model.lilicorr.";
    private static final String IN_PROJ = "attn.in_proj";

    private HfDraftHooksB() {
    }

    // ------------------------------------------------------------------ DeepseekV4DSparkModel

    /** The native names of an MTP stage of a DSpark draft, mapped to their DFLASH tensors (deepseek.py:931-938). */
    private static final Map<String, HfDeepseekV4Names.Mapped> DSPARK_ROOT;
    static {
        Map<String, HfDeepseekV4Names.Mapped> m = new LinkedHashMap<>();
        m.put("main_proj.weight", new HfDeepseekV4Names.Mapped("FC", WEIGHT, true));
        m.put("main_norm.weight", new HfDeepseekV4Names.Mapped("ENC_OUTPUT_NORM", WEIGHT, true));
        m.put("markov_head.markov_w1.weight", new HfDeepseekV4Names.Mapped("DSPARK_MARKOV_W1", WEIGHT, true));
        m.put("markov_head.markov_w2.weight", new HfDeepseekV4Names.Mapped("DSPARK_MARKOV_W2", WEIGHT, true));
        m.put("confidence_head.proj.weight", new HfDeepseekV4Names.Mapped("DSPARK_CONF_PROJ", WEIGHT, true));
        DSPARK_ROOT = Collections.unmodifiableMap(m);
    }

    /** The names whose DSV4 model-level table the DSpark re-key keeps (deepseek.py:1004-1010). */
    private static final Set<String> DSV4_ROOT_NAMES = Set.of("main_proj.scale", "norm.weight", "hc_head_fn",
            "hc_head_base", "hc_head_scale");
    /** The table of the DSpark subclass: its root map first, then the DeepSeek-V4 native names. */
    private static final HfDeepseekV4Names.Table DSPARK_TABLE = (name, bid) -> {
        HfDeepseekV4Names.Mapped m = DSPARK_ROOT.get(name);
        return m != null ? m : HfDeepseekV4Names.Table.DEFAULT.map(name, bid);
    };

    /**
     * {@code DeepseekV4DSparkModel.__init__} (deepseek.py:961-967): a draft has no compression and no hash routing.
     */
    public static HfConfig dsv4DsparkFixup(HfConfig cfg) {
        Map<String, Object> f = new LinkedHashMap<>();
        f.put("compress_ratios", Collections.emptyList());
        f.put("num_hash_layers", 0);
        return cfg.withFlat(f);
    }

    /**
     * {@code DeepseekV4DSparkModel.__init__}'s block count (deepseek.py:957-963): one plus the largest
     * {@code mtp.<stage>.} index of the checkpoint (as {code filter_tensors} re-keys it).
     */
    public static int dsv4DsparkBlockCount(HfLoadContext ctx) {
        int max = -1;
        for (String name : ctx.index().names()) {
            if (!name.startsWith("mtp.")) {
                continue;
            }
            String[] parts = name.split("\\.", 3);
            if (parts.length < 3 || parts[1].isEmpty() || !isDigits(parts[1])) {
                throw new GgufFormatException("Unexpected DSpark tensor '" + name + "'");
            }
            max = Math.max(max, Integer.parseInt(parts[1]));
        }
        if (max < 0) {
            throw new GgufFormatException(ctx.config().source() + ": a DeepSeek-V4 DSpark draft has no mtp.<stage>"
                    + " tensor");
        }
        long total = (long) max + 1;
        if (total > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(ctx.config().source() + ": block count " + total + " is outside [0, "
                    + HfConfig.MAX_LAYERS + "]");
        }
        return (int) total;
    }

    /** {@code tensor_force_quant} of the parent {@code DeepseekV4Model} (deepseek.py:894-908). */
    public static GGMLType dsv4DsparkForceType(String ggufName, int nDims, ElementFormat source,
            ModelSourceOptions.OutType outType) {
        return HfDeepseekV4Hooks.forceType(ggufName, nDims, source, outType);
    }

    /**
     * {@code DeepseekV4DSparkModel.filter_tensors} (deepseek.py:1040-1043): only {@code mtp.*} names survive;
     * {@code mtp.<stage>.<rest>} is rekeyed to phrases of the DSV4 native name space and passed to the parent's
     * filter.
     */
    public static HfNameFilter dsv4DsparkFilter(HfLoadContext ctx) {
        HfNameFilter dsv4 = HfDeepseekV4Hooks.filter(ctx);
        return (name, base) -> {
            if (!name.startsWith("mtp.")) {
                return null;
            }
            String[] parts = name.split("\\.", 3);
            if (parts.length < 3 || parts[2].isEmpty() || !isDigits(parts[1])) {
                throw new GgufFormatException("Unexpected DSpark tensor '" + name + "'");
            }
            String rest = parts[2];
            String rekeyed = DSPARK_ROOT.containsKey(rest) || DSV4_ROOT_NAMES.contains(rest) ? rest
                    : "layers." + parts[1] + "." + rest;
            return dsv4.filter(rekeyed, base);
        };
    }

    /**
     * {@code DeepseekV4DSparkModel.modify_tensors} (deepseek.py:1013-1017): the mtmd-only hash-gate bias of a DSV4
     * layer is not part of the DFLASH arch, everything else goes through the parent's native-name mapping.
     */
    public static void dsv4DsparkTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".ffn.gate.bias_vl")) {
            out.drop(hfName, "the mtmd-only gate bias variant is not part of the DFLASH arch");
            return;
        }
        HfDeepseekV4Hooks.tensors(ctx, hfName, src, out, DSPARK_TABLE);
    }

    /**
     * {@code DeepseekV4DSparkModel.set_gguf_parameters} (deepseek.py:1082-1089):
     * the parent's DeepSeek-V4 keys, then the draft's {@code dspark_block_size} and {@code target_layers}
     * (shifted by one).
     */
    public static void dsv4DsparkMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfDeepseekV4Hooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.DFLASH_BLOCK_SIZE, cfg.requireInt("dspark_block_size"));
        kv.set(GgufKeys.TARGET_LAYERS, shiftByOne(ctx, requireIntList(cfg, "dspark_target_layer_ids"),
                "dspark_target_layer_ids"));
    }

    /**
     * {@code DeepseekV4DSparkModel.set_vocab} (deepseek.py:1067-1080): the parent's GPT-2 vocabulary over the
     * target's tokenizer files (holds the bundled DeepSeek-V4 template as its fallback), then the draft's
     * {@code dspark_noise_token_id} as the mask.
     */
    public static Map<String, Object> dsv4DsparkVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        ctx.target();
        HfTokenizerFiles files = HfDraftHooks.targetTokenizer(ctx);
        HfVocabWriters.Input in = HfVocabWriters.Input.of(ctx.config(), files, PreTokenizerCanonicalizer::toSpec)
                .withGgufArch(ctx.arch().ggufName())
                .withFallbackChatTemplate(HfChatTemplates.bundled(
                        HfDeepseekV4Hooks.defaultTemplate(ctx.config().dir())));
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.gpt2(in, pre));
        md.put(GgufKeys.TOKENIZER_MASK_ID, ctx.config().requireInt("dspark_noise_token_id"));
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code dspark_target_layer_ids} plus one, the converter's target layer indices.
     */
    private static List<Integer> shiftByOne(HfLoadContext ctx, List<Integer> ids, String where) {
        List<Integer> out = new ArrayList<>(ids.size());
        for (int id : ids) {
            if (id < -1 || id >= HfConfig.MAX_LAYERS) {
                throw new GgufFormatException(ctx.config().source() + ": " + where + " entry " + id
                        + " is outside [0, " + HfConfig.MAX_LAYERS + ")");
            }
            out.add(id + 1);
        }
        return Collections.unmodifiableList(out);
    }

    /** A required {@code List<Integer>} key of the configuration (the draft's {@code target_layer_ids}). */
    private static List<Integer> requireIntList(HfConfig cfg, String key) {
        return cfg.findIntList(key).orElseThrow(() -> new GgufFormatException(
                cfg.source() + ": required key '" + key + "' is missing"));
    }

    private static boolean isDigits(String s) {
        if (s.isEmpty()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------ LiLiCorrDraftModel

    /** {@code LiLiCorrDraftModel.__init__}'s {@code dflash_config} selector cut (nemotron.py:56-57). */
    public static HfConfig lilicorrFixup(HfConfig cfg) {
        Map<String, Object> dc = HfDraftHooks.dflashConfig(cfg);
        if (dc.isEmpty()) {
            return cfg;
        }
        Map<String, Object> merged = new LinkedHashMap<>(dc);
        merged.put("selector_rank", 0);
        merged.put("selector_top_k", 0);
        return cfg.withFlat(Collections.singletonMap(DFLASH_CONFIG, merged));
    }

    /** The illicorr layers: {@code dflash_config.lilicorr_num_layers}, required. */
    private static int lilicorrLayers(HfConfig cfg) {
        Object v = HfDraftHooks.dflashConfig(cfg).get("lilicorr_num_layers");
        if (v == null) {
            throw new GgufFormatException(cfg.source() + ": LiLiCorr requires "
                    + DFLASH_CONFIG + ".lilicorr_num_layers");
        }
        return (int) HfDraftHooks.longOf(cfg, DFLASH_CONFIG + ".lilicorr_num_layers", v);
    }

    /**
     * {@code LiLiCorrDraftModel.__init__} (nemotron.py:36, :85): the tensor map also covers the lilicorr layers and
     * the feature block, so the name map extends {@code num_layers + 1} blocks past the drafter's own.
     */
    public static int lilicorrBlockCount(HfConfig cfg) {
        int base = cfg.blockCount();
        int layers = lilicorrLayers(cfg);
        long total = (long) base + layers + 1;
        if (total > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": block count " + total + " is outside [0, "
                    + HfConfig.MAX_LAYERS + "]");
        }
        return (int) total;
    }

    /**
     * {@code LiLiCorrDraftModel.tensor_force_quant} (nemotron.py:89-94): every LiLiCorr correction tensor is
     * computed against the float32 biases, so the GGUF tensors stay F32.
     */
    public static GGMLType lilicorrForceType(String ggufName, int nDims, ElementFormat source,
                                             ModelSourceOptions.OutType outType) {
        return ggufName.startsWith("lilicorr.") ? GGMLType.F32 : null;
    }

    /**
     * {@code LiLiCorrDraftModel.map_tensor_name} plus the {@code __init__} renames (nemotron.py:24-57, :86-95):
     * the slot and rank embeddings gain a {@code .weight}; an {@code in_proj_<suffix>} becomes
     * {@code in_proj.<suffix>}; layer numbers past {@code block_count} are re-numbered with the base block count
     * (except the attention input projections, which keep their local layer number); {@code feature_norm} and
     * {@code feature_mlp} become the feature block's norms.
     */
    public static HfNameFilter lilicorrFilter(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        int base = cfg.blockCount();
        int layers = lilicorrLayers(cfg);
        HfNameFilter dflash = HfDraftHooksA.dflashFilter(ctx);
        return (name, base2) -> {
            String t = name;
            if (t.startsWith(LILICORR_PREFIX)) {
                if (t.startsWith(LILICORR_PREFIX + "layers.")) {
                    String[] parts = t.split("\\.", 4);
                    if (parts.length == 4 && isDigits(parts[2])) {
                        int i = Integer.parseInt(parts[2]);
                        String rest = parts.length == 4 ? "" : "." + parts[3];
                        if (rest.startsWith("." + IN_PROJ)) {
                            int dot = rest.indexOf('.');
                            String tail = rest.substring(dot + 1);
                            if (tail.equals(IN_PROJ + "_weight") || tail.equals(IN_PROJ + "_bias")) {
                                t = "model.lilicorr.layers." + i + "." + IN_PROJ
                                        + tail.substring(tail.lastIndexOf('_'));
                                parts = t.split("\\.", 4);
                            }
                        } else if (i < 0 || i > HfConfig.MAX_LAYERS - base) {
                            throw new GgufFormatException("tensor '" + t + "': layer " + i + " exceeds the "
                                    + (HfConfig.MAX_LAYERS - base) + "-layer limit");
                        } else {
                            t = "model.lilicorr.layers." + (base + i) + "." + parts[3];
                        }
                    }
                } else if (t.startsWith(LILICORR_PREFIX + "feature_norm.")) {
                    t = LILICORR_PREFIX + "feature_mlp." + (base + layers) + ".norm"
                            + t.substring((LILICORR_PREFIX + "feature_norm.").length());
                } else if (t.startsWith(LILICORR_PREFIX + "feature_mlp.")) {
                    t = LILICORR_PREFIX + "feature_mlp." + (base + layers) + "."
                            + t.substring((LILICORR_PREFIX + "feature_mlp.").length());
                } else if (t.equals(LILICORR_PREFIX + "slot_embedding") || t.equals(LILICORR_PREFIX + "rank_embedding")) {
                    t = t + WEIGHT;
                }
            }
            return dflash.filter(t, base2);
        };
    }

    /**
     * {@code LiLiCorrDraftModel.modify_tensors} (nemotron.py:97-111): the two slot biases go into the synthesized
     * attention bias (see {@link #lilicorrExtra}), lilicorr names map as they are, and the DFlash parent handles
     * the rest.
     */
    public static void lilicorrTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals(RELATIVE_SLOT_BIAS) || hfName.equals(SAME_SLOT_BIAS)) {
            out.consume(hfName);
            return;
        }
        if (hfName.startsWith(LILICORR_PREFIX)) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        HfDraftHooksA.dflashTensors(ctx, hfName, src, out);
    }

    /**
     * {@code LiLiCorrDraftModel.set_gguf_parameters} (nemotron.py:81-90): the DFlash metadata then the LiLiCorr
     * head geometry.
     */
    public static void lilicorrMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfDraftHooksA.dflashMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        Map<String, Object> dc = HfDraftHooks.dflashConfig(cfg);
        kv.set(GgufKeys.LILICORR_HIDDEN_SIZE, dcInt(cfg, dc, "lilicorr_hidden_size"));
        kv.set(GgufKeys.LILICORR_CANDIDATE_TOPK, dcInt(cfg, dc, "lilicorr_candidate_topk"));
        kv.set(GgufKeys.LILICORR_NUM_HEADS, dcInt(cfg, dc, "lilicorr_num_heads"));
        kv.set(GgufKeys.LILICORR_NUM_LAYERS, dcInt(cfg, dc, "lilicorr_num_layers"));
        kv.set(GgufKeys.LILICORR_FACTOR_DIM, dcInt(cfg, dc, "lilicorr_factor_dim"));
        kv.set(GgufKeys.LILICORR_VECTOR_EPS, dcFloat(cfg, dc, "lilicorr_vector_eps"));
        kv.set(GgufKeys.LILICORR_LOGIT_SCALE, dcFloat(cfg, dc, "lilicorr_logit_scale"));
        int mlpHidden = layers0UpCols(ctx, plan);
        kv.set(GgufKeys.LILICORR_MLP_HIDDEN_SIZE, mlpHidden);
    }

    private static int dcInt(HfConfig cfg, Map<String, Object> dc, String key) {
        Object v = dc.get(key);
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": " + DFLASH_CONFIG + "." + key + " must be an integer");
        }
        return (int) HfDraftHooks.longOf(cfg, DFLASH_CONFIG + "." + key, v);
    }

    private static double dcFloat(HfConfig cfg, Map<String, Object> dc, String key) {
        Object v = dc.get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": " + DFLASH_CONFIG + "." + key + " must be a number");
        }
        return ((Number) v).doubleValue();
    }

    /**
     * {@code mlp_hidden_size} (nemotron.py:35): dim 0 of the first lilicorr layer's {@code mlp.up_proj}, i.e. the
     * inner GGUF dimension of the re-numbered FFN_UP tensor of the layer at block {@code block_count}.
     */
    private static int layers0UpCols(HfLoadContext ctx, HfPlanView plan) throws IOException {
        int bid = ctx.config().blockCount();
        String ggufName = ctx.names().ggufName("FFN_UP", bid, WEIGHT);
        long[] ne = plan.ne(ggufName);
        return Math.toIntExact(ne.length > 1 ? ne[1] : 0);
    }

    /** {@code LiLiCorrDraftModel.set_vocab} (nemotron.py:59-77): the target vocabulary and the hard mask token. */
    public static Map<String, Object> lilicorrVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfDraftHooks.targetVocab(ctx, pre,
                (c, p) -> HfQwenHooks.qwen3Vocab(c, p)));
        int maskId = HfDraftHooks.maskTokenId(ctx.config());
        if (maskId < 0) {
            throw new GgufFormatException(ctx.config().source() + ": LiLiCorr requires "
                    + DFLASH_CONFIG + ".mask_token_id");
        }
        md.put(GgufKeys.TOKENIZER_MASK_ID, maskId);
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code LiLiCorrDraftModel.__init__}' attention bias (nemotron.py:38-52): the dense [heads, nodes, nodes] F32
     * correction head from the relative slot bias (a per-head array of {@code 2 * block_size - 1} values) plus the
     * same-slot bias, applied per token pair {@code (q, k)} at the slot distance of their block-relative positions.
     */
    public static void lilicorrExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        Map<String, Object> dc = HfDraftHooks.dflashConfig(cfg);
        int block = blockSize(cfg, dc);
        int topK = dcInt(cfg, dc, "lilicorr_candidate_topk");
        int heads = dcInt(cfg, dc, "lilicorr_num_heads");
        int nodes = (block - 1) * topK;
        int slots = 2 * block - 1;
        HfTensorIndex index = ctx.index();
        if (!index.has(RELATIVE_SLOT_BIAS) || !index.has(SAME_SLOT_BIAS)) {
            throw new GgufFormatException(cfg.source() + ": the LiLiCorr draft requires " + RELATIVE_SLOT_BIAS
                    + " and " + SAME_SLOT_BIAS);
        }
        float[] rel = HfDraftHooks.rowFloats(index.get(RELATIVE_SLOT_BIAS), RELATIVE_SLOT_BIAS);
        if (rel.length != heads * slots) {
            throw new GgufFormatException(cfg.source() + ": " + RELATIVE_SLOT_BIAS + " holds " + rel.length
                    + " values, expected " + heads + " heads x " + slots + " slots");
        }
        float[] same = HfDraftHooks.rowFloats(index.get(SAME_SLOT_BIAS), SAME_SLOT_BIAS);
        if (same.length != heads && same.length != 1) {
            throw new GgufFormatException(cfg.source() + ": " + SAME_SLOT_BIAS + " holds " + same.length
                    + " values, expected " + heads);
        }
        float[] bias = new float[Math.toIntExact(Math.multiplyExact((long) heads, (long) nodes * nodes))];
        int nodeSpan = nodes;
        for (int h = 0; h < heads; h++) {
            float sameValue = same.length == 1 ? same[0] : same[h];
            int biasRow = h * nodeSpan * nodeSpan;
            for (int q = 0; q < nodes; q++) {
                int qSlot = q / topK;
                for (int k = 0; k < nodes; k++) {
                    int kSlot = k / topK;
                    float v = rel[h * slots + qSlot - kSlot + block - 1];
                    if (qSlot == kSlot) {
                        v += sameValue;
                    }
                    bias[biasRow + q * nodeSpan + k] = v;
                }
            }
        }
        out.add(ctx.names().ggufName("LILICORR_ATTN_BIAS", -1, ".bias"),
                RopeTensors.fromFloats(new long[]{heads, (long) nodes, (long) nodes}, bias));
    }

    /** The {@code block_size} of the draft: the shared helper over the LiLiCorr {@code dflash_config}. */
    private static int blockSize(HfConfig cfg, Map<String, Object> dc) {
        return HfDraftHooksA.blockSize(cfg, dc);
    }

    // ------------------------------------------------------------------ Gemma4DSparkModel

    /**
     * {@code Gemma4DSparkModel.__init__} (gemma.py:839-869): the current DSpark draft keeps its validations and
     * the effective rope parameters of the Gemma 4 text config defaults of the absent rope block.
     */
    public static HfConfig gemma4DsparkFixup(HfConfig cfg) {
        if (!cfg.findBool(false, "attention_k_eq_v")) {
            throw new GgufFormatException(cfg.source() + ": attention_k_eq_v must be true for a Gemma4 DSpark draft");
        }
        List<Boolean> swa = allFullAttention(cfg);
        if (!swa.isEmpty() && swa.contains(Boolean.TRUE)) {
            throw new GgufFormatException(cfg.source() + ": Gemma4 DSpark requires uniform full_attention layers");
        }
        String act = cfg.findStringOr("gelu_pytorch_tanh", "hidden_activation");
        if (!"gelu_pytorch_tanh".equals(act)) {
            throw new GgufFormatException(cfg.source() + ": Gemma4 DSpark requires hidden_activation gelu_pytorch_tanh");
        }
        if (cfg.findBool(false, "attention_bias") || cfg.findBool(false, "enable_moe_block")) {
            throw new GgufFormatException(cfg.source() + ": Gemma4 DSpark attention bias and MoE are not supported");
        }
        int vocab = cfg.requireInt("vocab_size");
        int draftVocab = cfg.findInt(0, "draft_vocab_size");
        if (draftVocab != 0 && draftVocab != vocab) {
            throw new GgufFormatException(cfg.source() + ": Gemma4 DSpark requires a full draft vocabulary");
        }
        String markov = stringOf(HfDraftHooks.dflashConfig(cfg), cfg, "markov_head_type", "vanilla");
        if (!"vanilla".equals(markov)) {
            throw new GgufFormatException(cfg.source() + ": Gemma4 DSpark requires a vanilla Markov head");
        }
        return ropeDefaults(cfg);
    }

    /** The Gemma 4 rope: the proportional factors of a {@code rope_parameters} that misses its block. */
    private static HfConfig ropeDefaults(HfConfig cfg) {
        boolean present = cfg.raw().has("rope_parameters") || cfg.raw().has("rope_scaling");
        if (present) {
            String type = cfg.rope().global().type().orElse("");
            if (!"default".equals(type) && !"proportional".equals(type)) {
                throw new GgufFormatException(cfg.source() + ": rope type '" + type
                        + "'; Gemma4 DSpark requires default or proportional RoPE");
            }
            return cfg;
        }
        Map<String, Object> full = new LinkedHashMap<>();
        full.put("rope_type", "proportional");
        full.put("partial_rotary_factor", 0.25);
        full.put("rope_theta", 1000000.0);
        return cfg.withFlat(Collections.singletonMap("rope_parameters",
                Collections.singletonMap("full_attention", full)));
    }

    private static List<Boolean> allFullAttention(HfConfig cfg) {
        Object v = cfg.flat().get("layer_types");
        if (!(v instanceof List)) {
            return Collections.emptyList();
        }
        List<Boolean> out = new ArrayList<>(((List<?>) v).size());
        for (Object t : (List<?>) v) {
            out.add(!("full_attention".equals(t)));
        }
        return out;
    }

    private static String stringOf(Map<String, Object> dc, HfConfig cfg, String key, String dflt) {
        Object v = dc.get(key);
        if (v == null) {
            v = cfg.flat().get(key);
        }
        return v == null ? dflt : String.valueOf(v);
    }

    /**
     * {@code Gemma4DSparkModel.filter_tensors} (gemma.py:903-907): a missing {@code model.} prefix, a
     * {@code .layer_scalar} weight suffix, the confidence head rename, then the DFlash filter.
     */
    public static HfNameFilter gemma4DsparkFilter(HfLoadContext ctx) {
        HfNameFilter dflash = HfDraftHooksA.dflashFilter(ctx);
        return (name, base) -> {
            String t = HfDraftHooksA.dflashRename(name);
            if (t.endsWith(".layer_scalar")) {
                t = t + WEIGHT;
            }
            if (t.startsWith("model.")) {
                t = t.replace("model.confidence_proj.", "model.confidence_head.proj.");
            }
            return dflash.filter(t, base);
        };
    }

    /**
     * {@code Gemma4DSparkModel.set_gguf_parameters} (gemma.py:871-901): the DFlash metadata, then the Gemma 4
     * global head geometry, which the DSpark draft tied to the target's, the identity attention scale, the
     * activation, the sample-from-anchor flag and the confidence-head probe.
     */
    public static void gemma4DsparkMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfDraftHooksA.dflashMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        int headDim = cfg.requireInt("global_head_dim");
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.requireInt("num_global_key_value_heads"));
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, headDim);
        kv.set(GgufKeys.EMBEDDING_SCALE, Math.sqrt(cfg.requireInt("hidden_size")));
        kv.set(GgufKeys.ATTENTION_SCALE, 1.0);
        kv.set(GgufKeys.HIDDEN_ACT, "gelu_pytorch_tanh");
        kv.set(GgufKeys.SAMPLE_FROM_ANCHOR, cfg.findBool(true, "sample_from_anchor"));
        kv.set(GgufKeys.HAS_CONFIDENCE_HEAD, anyConfidence(ctx));
        Map<String, Object> dc = HfDraftHooks.dflashConfig(cfg);
        if (HfDraftHooks.truthy(dc.get("use_swa"))) {
            Object window = dc.get("swa_window_size");
            if (window == null || HfDraftHooks.doubleOf(cfg, DFLASH_CONFIG + ".swa_window_size", window) <= 0) {
                throw new GgufFormatException(cfg.source() + ": Gemma4 DSpark swa_window_size must be positive");
            }
        }
    }

    private static boolean anyConfidence(HfLoadContext ctx) {
        for (String name : ctx.index().names()) {
            if (name.contains("confidence_head.proj")) {
                return true;
            }
        }
        return false;
    }

    /**
     * {@code Gemma4DSparkModel.modify_tensors} (gemma.py:908-917): the Gemma layer-norm names map to their qwen
     * counterpart tensors, the rest through the DFlash step.
     */
    public static void gemma4DsparkTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        if (name.endsWith(".post_attention_layernorm.weight") || name.endsWith(".post_attention_layernorm")) {
            name = ctx.names().ggufName(ATTENTION_POST_NORM, HfHookSupport.layerOf(ctx.names().ggufName(
                    ATTENTION_POST_NORM, HfHookSupport.layerOf(name), "")),
                    name.endsWith(WEIGHT) ? WEIGHT : "");
        } else if (name.endsWith(".pre_feedforward_layernorm.weight") || name.endsWith(".pre_feedforward_layernorm")) {
            name = ctx.names().ggufName(FFN_NORM, HfHookSupport.layerOf(ctx.names().ggufName(FFN_NORM,
                    HfHookSupport.layerOf(name), "")), name.endsWith(WEIGHT) ? WEIGHT : "");
        }
        HfDraftHooksA.dflashTensors(ctx, name, src, out);
    }

    /**
     * {@code Gemma4DSparkModel.generate_extra_tensors} (gemma.py:922-940): the proportional-rope frequency factors
     * holding the unrotated RoPE identifiers in place.
     */
    public static void gemma4DsparkExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        if (!"proportional".equals(cfg.rope().global().type().orElse(""))) {
            return;
        }
        int headDim = cfg.requireInt("global_head_dim");
        double fraction = cfg.rope().global().partialRotaryFactor().orElse(0.25);
        if (!(fraction > 0 && fraction <= 1.0) || headDim * fraction != 2.0 * Math.floor(headDim * fraction / 2.0)) {
            throw new GgufFormatException(cfg.source() + ": Gemma4 DSpark rotary dimension count must be positive and even");
        }
        out.add(ctx.names().ggufName("ROPE_FREQS", -1, WEIGHT), RopeTensors.proportionalFreqs(headDim, fraction));
    }

    /**
     * {@code Gemma4DSparkModel.set_vocab} (gemma.py:846-852): the DFlash vocab dispatch, then the mask token of
     * {@code dflash_config.mask_token_id} (else head default). A draft without either fails.
     */
    public static Map<String, Object> gemma4DsparkVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfDraftHooksA.dflashVocab(ctx, pre);
    }

    // ------------------------------------------------------------------ Gemma4AssistantModel

    /**
     * {@code Gemma4AssistantModel.set_gguf_parameters} (gemma.py:954-957):
     * the parent's Gemma 4 keys, then the output width (the backbone's hidden size) and the assistant's NextN
     * block count (which is the drafter block count).
     */
    public static void gemma4AssistantMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfGemma4Hooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.EMBEDDING_LENGTH_OUT, cfg.requireInt("backbone_hidden_size"));
        kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, ctx.blockCount());
    }

    /**
     * {@code Gemma4AssistantModel.modify_tensors} (gemma.py:943-948, 958-961): the class keeps the Gemma 4 parent's
     * tensor mapping and only its filter removes the {@code masked_embedding} tensors it does not use.
     */
    public static void gemma4AssistantTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfGemma4Hooks.tensors(ctx, hfName, src, out);
    }

    /**
     * {@code Gemma4AssistantModel.filter_tensors} (gemma.py:943-948): the masked embedding it does not use is
     * dropped; the rest through the parent Gemma 4 filter.
     */
    public static String gemma4AssistantFilter(String name, UnaryOperator<String> base) {
        if (name.contains("masked_embedding")) {
            return null;
        }
        return HfGemma4Hooks.filter(name, base);
    }
}
