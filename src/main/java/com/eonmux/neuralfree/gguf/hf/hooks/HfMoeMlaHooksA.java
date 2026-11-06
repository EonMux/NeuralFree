package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hooks of the DeepseekV2-deriving MLA/DSA converter classes, part A: {@code DeepseekV32Model}
 * (LC/conversion/deepseek.py:466-523), {@code Glm4MoeLiteModel} (glm.py:249-314), {@code GlmMoeDsaModel}
 * (glm.py:318-383) and {@code Dots3NoteModel} (dots3.py:19-198). The classes compose the {@link HfDeepseekHooks} of
 * their base {@code DeepseekV2Model} and add what their methods layer on it: the appended-NextN block, the kept MTP
 * layers, the GLM vocabulary, the DSA indexer and the per-layer-kind geometry of the sliding-window MLA.
 *
 * <p>The standalone text classes live in {@link HfMoeMlaHooksB}, {@code HYV4Model} in {@link HfMoeMlaHooksC} and the
 * native Mistral MoE format in {@link HfMistralMoeHooks}. Every method is a public static hook reference; the entry
 * registers them.</p>
 */
public final class HfMoeMlaHooksA {

    /** {@code re.match(r"model.layers.(\d+)", name)} for the MTP name checks. */
    private static final Pattern MODEL_LAYER = Pattern.compile("model\\.layers\\.(\\d+)");

    /** The ambient sliding-window head dimension (dots3.py:61). */
    private static final int DEFAULT_SWA_HEAD_DIM = 256;

    /** Solar-Open-style added-vocabulary end-of-turn special of the Dots3-Note tokenizer (dots3.py:84). */
    static final String DOTS3_EOT = "<|endofassistant|>";

    /** The supported values the class restricts its configuration to, per dots3.py:46-65. */
    private static final Set<String> RMS_ONLY = Set.of("RMSNorm");
    private static final Set<String> HEADWISE = Set.of("headwise");

    private HfMoeMlaHooksA() {
    }

    // ------------------------------------------------------------------ shared composition helpers

    /** The number of NextN layers the configuration declares, 0 when it has none. */
    static int nextnLayers(HfConfig cfg) {
        return cfg.findInt(0, "num_nextn_predict_layers");
    }

    /** True when {@code name} is a layer tensor past {@code mainLayers} ({@code model.layers.<n>.} with n >= main). */
    static boolean isMtpLayer(String name, int mainLayers) {
        Matcher m = MODEL_LAYER.matcher(name);
        return m.lookingAt() && Integer.parseInt(m.group(1)) >= mainLayers;
    }

    /**
     * A load name filter of an MTP-keeping DeepseekV2 subclass (DeepseekV32, Glm4MoeLite and GlmMoeDsa share the
     * shape of their {@code filter_tensors}, deepseek.py:483-504 / glm.py:284-300 / glm.py:339-360): the base
     * {@link HfDeepseekHooks#filter} first, then the appended NextN block dropped from a view that excludes
     * multi-token prediction.
     */
    public static Function<HfLoadContext, HfNameFilter> mtpKeepFilter() {
        return ctx -> {
            int main = ctx.config().requireInt("num_hidden_layers");
            boolean keep = ctx.options().includeMtp();
            return (name, base) -> {
                String n = HfDeepseekHooks.filter(name, base);
                if (n == null || (isMtpLayer(n, main) && !keep)) {
                    return null;
                }
                return n;
            };
        };
    }

    /** The GPT-2 BPE vocabulary of the classes that keep {@code TextModel.set_vocab}. */
    public static Map<String, Object> gpt2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /** The GLM vocabulary ({@code _set_vocab_glm}) of the GLM classes. */
    public static Map<String, Object> glmVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.glm(HfHookSupport.vocabInput(ctx), pre);
    }

    /** The expert completeness check of {@code DeepseekV2Model.prepare_tensors} (deepseek.py:454-462). */
    public static void deepseekExtra(HfLoadContext ctx, TensorSink out) {
        HfDeepseekHooks.extra(ctx, out);
    }

    /** The partial rotary factor of {@code cfg}'s RoPE parameters, or {@code dflt}. */
    static double partialRotary(HfConfig cfg, double dflt) {
        Object v = cfg.rope().raw().get("partial_rotary_factor");
        if (v instanceof Number && !(v instanceof Boolean)) {
            return ((Number) v).doubleValue();
        }
        OptionalDouble f = cfg.findFloat("partial_rotary_factor", "rope_pct", "rotary_pct");
        return f.isPresent() ? f.getAsDouble() : dflt;
    }

    // ------------------------------------------------------------------ DeepseekV32Model

    /**
     * {@code DeepseekV32Model.set_gguf_parameters} (deepseek.py:512-523) after the base metadata: the NextN layer
     * count and the DSA indexer parameters (head count, key length, top-k).
     */
    public static void deepseek32Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        OptionalInt nextn = cfg.findInt("num_nextn_predict_layers");
        if (ctx.options().includeMtp() && nextn.isPresent()) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn.getAsInt());
        }
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, cfg.requireInt("index_n_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, cfg.requireInt("index_head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, cfg.requireInt("index_topk"));
    }

    /** The block count of the view: the NextN block appended when MTP is included (deepseek.py:472-477). */
    public static int deepseek32BlockCount(HfLoadContext ctx) {
        return ctx.options().includeMtp()
                ? deepseek32ConfigBlockCount(ctx.config())
                : ctx.config().requireInt("num_hidden_layers");
    }

    /** The block count of the configuration alone. */
    public static int deepseek32ConfigBlockCount(HfConfig cfg) {
        return cfg.requireInt("num_hidden_layers") + nextnLayers(cfg);
    }

    /** {@code DeepseekV32Model.modify_tensors} through the base with MTP layers kept. */
    public static void deepseek32Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfDeepseekHooks.tensors(ctx, hfName, src, out, false, true);
    }

    /**
     * {@code DeepseekV32Model.set_vocab} (deepseek.py:506-510): asserts the tokenizer's {@code add_bos_token}, then
     * the GPT-2 BPE vocabulary.
     *
     * @throws GgufFormatException with the converter's message when tokenizer_config adds no BOS token
     */
    public static Map<String, Object> deepseek32Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Object bos = ctx.tokenizerFiles().readTokenizerConfig().raw("add_bos_token");
        if (Boolean.FALSE.equals(bos)) {
            throw new GgufFormatException("Change value of add_bos_token to true in tokenizer_config.json file.");
        }
        return gpt2Vocab(ctx, pre);
    }

    // ------------------------------------------------------------------ Glm4MoeLiteModel

    /**
     * {@code Glm4MoeLiteModel.set_gguf_parameters} (glm.py:272-278) after the base metadata: the NextN layer count,
     * unless the class skips MTP ({@code skip_mtp}, glm.py:262-263).
     */
    public static void glm4MoeLiteMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
        if (ctx.options().includeMtp() && !glm4MoeLiteSkipMtp(ctx.config())) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextnLayers(ctx.config()));
        }
    }

    /** {@code skip_mtp = self.no_mtp or num_nextn_predict_layers == 0} (glm.py:262-263). */
    static boolean glm4MoeLiteSkipMtp(HfConfig cfg) {
        return nextnLayers(cfg) == 0;
    }

    /** The block count of the view: the NextN block unless the class skips it or the view drops MTP. */
    public static int glm4MoeLiteBlockCount(HfLoadContext ctx) {
        if (!ctx.options().includeMtp() || glm4MoeLiteSkipMtp(ctx.config())) {
            return ctx.config().requireInt("num_hidden_layers");
        }
        return ctx.config().requireInt("num_hidden_layers") + nextnLayers(ctx.config());
    }

    /** {@code Glm4MoeLiteModel.modify_tensors} through the base with MTP layers kept and experts stacked. */
    public static void glm4MoeLiteTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfDeepseekHooks.tensors(ctx, hfName, src, out, false, true);
    }

    // ------------------------------------------------------------------ GlmMoeDsaModel

    /**
     * {@code GlmMoeDsaModel.set_gguf_parameters} (glm.py:365-383) after the base metadata: the RoPE width over
     * {@code partial_rotary_factor} (default 1.0), the NextN layer count and the DSA indexer parameters with the
     * per-layer {@code indexer_types}, {@code full} for the layers that own an indexer.
     */
    public static void glmMoeDsaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (cfg.requireInt("qk_rope_head_dim") * partialRotary(cfg, 1.0)));
        OptionalInt nextn = cfg.findInt("num_nextn_predict_layers");
        if (ctx.options().includeMtp() && nextn.isPresent()) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn.getAsInt());
        }
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, cfg.requireInt("index_n_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, cfg.requireInt("index_head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, cfg.requireInt("index_topk"));
        Object indexerTypes = cfg.flat().get("indexer_types");
        if (indexerTypes instanceof List) {
            List<Boolean> full = new ArrayList<>();
            for (Object t : (List<?>) indexerTypes) {
                full.add("full".equals(t));
            }
            kv.set(GgufKeys.ATTENTION_INDEXER_TYPES, full);
        }
    }

    /** {@link #deepseek32BlockCount}: the same appended-NextN count (glm.py:328-333). */
    public static int glmMoeDsaBlockCount(HfLoadContext ctx) {
        return deepseek32BlockCount(ctx);
    }

    /** {@link HfDeepseekHooks#tensors}: the base MTP-keeping, expert-stacking path. */
    public static void glmMoeDsaTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfDeepseekHooks.tensors(ctx, hfName, src, out, false, true);
    }

    // ------------------------------------------------------------------ Dots3NoteModel

    /** True when block {@code bid} is a sliding-window kind layer ({@code _is_swa_layer}, dots3.py:69-73). */
    public static boolean isSwaLayer(HfLoadContext ctx, int bid) {
        return bid >= ctx.config().requireInt("num_hidden_layers") || isSwaLayer(ctx.config(), bid);
    }

    private static boolean isSwaLayer(HfConfig cfg, int bid) {
        Object types = cfg.flat().get("layer_types");
        return types instanceof List && bid >= 0 && bid < ((List<?>) types).size()
                && "sliding_attention".equals(((List<?>) types).get(bid));
    }

    /**
     * {@code Dots3NoteModel.__init__} (dots3.py:31-67): the configuration the class validates, and the per-layer
     * attention head counts over the blocks with the MTP block a sliding-window one, plus the removal of
     * {@code head_dim} ({@code set_gguf_parameters}, dots3.py:109-122).
     *
     * @throws GgufFormatException on every unsupported configuration variant the converter rejects
     */
    public static HfConfig dots3Fixup(HfConfig cfg) {
        Object layerTypes = cfg.flat().get("layer_types");
        int layers = cfg.requireInt("num_hidden_layers");
        if (!(layerTypes instanceof List) || ((List<?>) layerTypes).size() < layers) {
            throw new GgufFormatException(cfg.source() + ": layer_types is shorter than num_hidden_layers");
        }
        check(cfg, !truthy(cfg, "use_dsa", true), "dots3-note conversion requires use_dsa=true");
        Object norm = cfg.flat().get("normalization");
        Object finalNorm = cfg.flat().get("final_norm");
        check(cfg, norm != null && !RMS_ONLY.contains(norm) || finalNorm != null && !RMS_ONLY.contains(finalNorm),
                "dots3-note conversion only supports RMSNorm");
        check(cfg, !truthy(cfg, "k_rope_only_layernorm", true),
                "dots3-note conversion requires k_rope_only_layernorm=true");
        Object topk = cfg.find("topk_method").isPresent() ? cfg.flat().get("topk_method") : "noaux_tc";
        check(cfg, !"noaux_tc".equals(topk) || !"sigmoid".equals(cfg.find("scoring_func").orElse(null)),
                "dots3-note conversion only supports noaux_tc/sigmoid expert gating");
        check(cfg, cfg.findInt(1, "n_group") != 1 || cfg.findInt(1, "topk_group") != 1,
                "dots3-note conversion does not support grouped expert routing");
        check(cfg, truthy(cfg, "use_dynamic_rsf", false) || truthy(cfg, "moe_gating_fp32", false),
                "dots3-note conversion does not support use_dynamic_rsf/moe_gating_fp32");
        Object gate = cfg.find("attention_gate_type").orElse("headwise");
        Object swaGate = cfg.find("swa_attention_gate_type").orElse("headwise");
        check(cfg, !HEADWISE.contains(gate), "dots3-note conversion only supports headwise attention gate, got"
                + " attention_gate_type=" + gate);
        check(cfg, !HEADWISE.contains(swaGate), "dots3-note conversion only supports headwise attention gate, got"
                + " swa_attention_gate_type=" + swaGate);
        int swaNope = cfg.requireInt("swa_qk_nope_head_dim");
        int swaRope = cfg.requireInt("swa_qk_rope_head_dim");
        check(cfg, swaNope + swaRope != cfg.findInt(DEFAULT_SWA_HEAD_DIM, "swa_head_dim"),
                "swa_head_dim must equal swa_qk_nope_head_dim + swa_qk_rope_head_dim");
        check(cfg, swaRope != cfg.requireInt("qk_rope_head_dim"),
                "swa_qk_rope_head_dim must match qk_rope_head_dim");
        check(cfg, !Integer.valueOf(cfg.requireInt("q_lora_rank")).equals(cfg.findInt("swa_q_lora_rank")),
                "dots3-note conversion assumes a shared q_lora_rank for both layer kinds");
        List<Long> heads = new ArrayList<>();
        for (int bid = 0; bid < layers; bid++) {
            heads.add(isSwaLayer(cfg, bid) ? (long) cfg.requireInt("swa_num_attention_heads")
                    : (long) cfg.requireInt("num_attention_heads"));
        }
        heads.add((long) cfg.requireInt("swa_num_attention_heads"));
        Map<String, Object> fix = new LinkedHashMap<>();
        fix.put("num_attention_heads", heads);
        fix.put("head_dim", null);
        return cfg.withFlat(fix);
    }

    /** Python truthiness of an optional value under {@code key}. */
    private static boolean truthy(HfConfig cfg, String key, boolean dflt) {
        Object o = cfg.flat().get(key);
        if (o == null) {
            return dflt;
        }
        if (o instanceof Boolean) {
            return (Boolean) o;
        }
        if (o instanceof Number) {
            return ((Number) o).doubleValue() != 0;
        }
        if (o instanceof String) {
            return !((String) o).isEmpty();
        }
        if (o instanceof Collection) {
            return !((Collection<?>) o).isEmpty();
        }
        return o instanceof Map && !((Map<?, ?>) o).isEmpty();
    }

    private static void check(HfConfig cfg, boolean invalid, String message) {
        if (invalid) {
            throw new GgufFormatException(cfg.source() + ": " + message);
        }
    }

    /**
     * {@code Dots3NoteModel.set_gguf_parameters} (dots3.py:109-144) after the base metadata: the sliding-window
     * geometry, the per-layer kinds of the sliding window and of the DSA indexer, and the indexer parameters. The
     * MTP block uses the sliding-attention MLA, so its indexer type is {@code false}.
     */
    public static void dots3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        int swaKvLora = cfg.requireInt("swa_kv_lora_rank");
        int swaRope = cfg.requireInt("swa_qk_rope_head_dim");
        int nLayer = cfg.requireInt("num_hidden_layers");
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window_size"));
        List<Boolean> swa = new ArrayList<>();
        List<Boolean> indexerFull = new ArrayList<>();
        for (int bid = 0; bid < ctx.blockCount(); bid++) {
            // the NextN/MTP block uses the sliding-attention MLA (dots3.py:69-73, dots3note.cpp:75)
            boolean s = isSwaLayer(ctx, bid);
            swa.add(s);
            if (bid < nLayer) {
                indexerFull.add(!s);
            }
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, swa);
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK_SWA, swaKvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_SWA, swaKvLora + swaRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_SWA, swaKvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA_SWA, cfg.requireInt("swa_qk_nope_head_dim") + swaRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA_SWA, cfg.requireInt("swa_v_head_dim"));
        if (ctx.blockCount() > nLayer) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, ctx.blockCount() - nLayer);
        }
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, cfg.requireInt("index_n_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, cfg.requireInt("index_head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, cfg.requireInt("index_topk"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TYPES, indexerFull);
    }

    /** The block count of the configuration alone: a candidate index fixes the count with {@link #dots3BlockCount}. */
    public static int dots3ConfigBlockCount(HfConfig cfg) {
        return cfg.requireInt("num_hidden_layers");
    }

    /** The MTP detection of {@link #dots3BlockCount}: the MTP embedding tensor or layers past the trunk. */
    private static boolean dots3MtpIn(HfTensorIndex index, int layers) {
        for (String name : index.names()) {
            if (name.equals("model.mtp.embed_tokens.weight") || isMtpLayer(name, layers)) {
                return true;
            }
        }
        return false;
    }

    /**
     * The indexed block count: one more block when the checkpoint carries MTP tensors (dots3.py:36-40, the class
     * detects the MTP block from the model weights).
     */
    public static int dots3BlockCount(HfLoadContext ctx) {
        int layers = ctx.config().requireInt("num_hidden_layers");
        return dots3MtpIn(ctx.index(), layers) ? layers + 1 : layers;
    }

    /**
     * The load name filter of {@code Dots3NoteModel.filter_tensors} (dots3.py:88-107): the DeepseekV2 base, the
     * vision and audio encoder drops, the MTP-prefixed name drop and the appended NextN block dropped from a view
     * that excludes multi-token prediction.
     */
    public static Function<HfLoadContext, HfNameFilter> dots3Filter() {
        return ctx -> {
            int main = ctx.config().requireInt("num_hidden_layers");
            boolean keep = ctx.options().includeMtp();
            return (name, base) -> {
                if (name.startsWith("vision_encoder.") || name.startsWith("audio_encoder.")) {
                    return null;
                }
                String n = HfDeepseekHooks.filter(name, base);
                if (n == null || n.startsWith("model.mtp.") || (isMtpLayer(n, main) && !keep)) {
                    return null;
                }
                return n;
            };
        };
    }

    /** The GPT-2 BPE vocabulary with the {@code eot} special of {@code Dots3NoteModel.set_vocab} (dots3.py:75-85). */
    public static Map<String, Object> dots3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre));
        int nVocab = HfHookSupport.embeddingRows(ctx).orElseThrow(() -> new GgufFormatException(
                "a Dots3-Note checkpoint has no token embedding to size its vocabulary"));
        HfSpecialVocab sv = HfSpecialVocab.load(files, files.readTokenizerJson(), ctx.config().raw(), true, nVocab);
        sv = sv.withSpecialToken("eot", HfMoeMlaHooksB.addedVocabId(files, DOTS3_EOT, nVocab));
        sv.writeTo(md);
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code Dots3NoteModel.modify_tensors} (dots3.py:158-198): the MTP token embedding moves into the NextN block,
     * the activation rescale {@code sqrt(n_embd / lora_rank)} folds into the {@code q_a}/{@code kv_a} layernorms
     * when configured, {@code kv_b_proj} splits into {@code k_b_proj} and {@code v_b_proj} with the per-kind
     * geometry, and everything else goes to the base path.
     */
    public static void dots3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        if (name.equals("model.mtp.embed_tokens.weight")) {
            name = "model.layers." + ctx.config().requireInt("num_hidden_layers") + ".embed_tokens.weight";
        }
        HfConfig cfg = ctx.config();
        int bid = ctx.bid(name);
        if (cfg.flat().get("apply_mla_qkv_lora_rescale") instanceof Boolean
                && (Boolean) cfg.flat().get("apply_mla_qkv_lora_rescale") && bid >= 0) {
            if (name.endsWith("q_a_layernorm.weight")) {
                src = ElementTransforms.mulScalar(src, scale(cfg, hidden(cfg), cfg.requireInt("q_lora_rank")));
            } else if (name.endsWith("kv_a_layernorm.weight")) {
                int rank = isSwaLayer(ctx, bid) ? cfg.requireInt("swa_kv_lora_rank") : cfg.requireInt("kv_lora_rank");
                src = ElementTransforms.mulScalar(src, scale(cfg, hidden(cfg), rank));
            }
        }
        if (name.endsWith("kv_b_proj.weight")) {
            splitKvBPerKind(ctx, name, src, out);
            return;
        }
        HfDeepseekHooks.tensors(ctx, name, src, out, false, true);
    }

    private static int hidden(HfConfig cfg) {
        return cfg.requireInt("hidden_size");
    }

    /** {@code math.sqrt(n_embd / lora_rank)} of the rescale. */
    private static float scale(HfConfig cfg, int hidden, int rank) {
        if (rank <= 0) {
            throw new GgufFormatException(cfg.source() + ": the LoRA rank must be positive, got " + rank);
        }
        return (float) Math.sqrt((double) hidden / rank);
    }

    /** The per-kind {@code kv_b_proj} split (dots3.py:174-195). */
    private static void splitKvBPerKind(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        int bid = ctx.bid(hfName);
        if (bid < 0) {
            throw new GgufFormatException("tensor '" + hfName + "' has no layer number");
        }
        HfConfig cfg = ctx.config();
        int nHead;
        int qkNope;
        int vHead;
        if (isSwaLayer(ctx, bid)) {
            nHead = cfg.requireInt("swa_num_attention_heads");
            qkNope = cfg.requireInt("swa_qk_nope_head_dim");
            vHead = cfg.requireInt("swa_v_head_dim");
        } else {
            nHead = cfg.requireInt("num_attention_heads");
            qkNope = cfg.requireInt("qk_nope_head_dim");
            vHead = cfg.requireInt("v_head_dim");
        }
        if (nHead <= 0 || qkNope <= 0 || vHead <= 0) {
            throw new GgufFormatException(cfg.source() + ": kv_b of block " + bid + " has nonpositive head geometry "
                    + nHead + "/" + qkNope + "/" + vHead);
        }
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, qkNope, vHead, Math.toIntExact(src.shape()[src.shape().length - 1]));
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("tensor '" + hfName + "' does not split into " + nHead + " heads of "
                    + qkNope + " key and " + vHead + " value rows: " + e.getMessage(), e);
        }
        out.add(ctx.names().mapOrThrow(hfName.replace("kv_b_proj", "k_b_proj")), kvB[0]);
        out.add(ctx.names().mapOrThrow(hfName.replace("kv_b_proj", "v_b_proj")), kvB[1]);
    }

    /**
     * The class's FP32 routing load-bearing forcing (dots3.py:317-323, the class's {@code tensor_force_quant}): the
     * vision MoE routing and the convolution tensors are F32.
     */
    public static GGMLType dots3ForceType(String ggufName, int nDims, ElementFormat source,
                                          ModelSourceOptions.OutType outType) {
        if (ggufName.contains(".ffn_gate_inp.") || ggufName.contains(".exp_probs_b.")
                || ggufName.contains(".conv2d") || ggufName.contains("a.conv_out")) {
            return GGMLType.F32;
        }
        return null;
    }
}
