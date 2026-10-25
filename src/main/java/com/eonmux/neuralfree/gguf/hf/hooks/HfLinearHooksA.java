package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorNameMap;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
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
import java.util.Optional;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The linear-attention hybrid classes of minimax.py and glm.py that no other pack builds on:
 * {@code MiniMaxText01Model} (LC/conversion/minimax.py:16-119) and {@code Glm5NextModel} (glm.py:409-598). Glm5Next
 * is the KDA + nope-MLA + DSA kpool + mHC architecture; MiniMax-Text-01 carries the zero-embedding token
 * suppression, the residual scale and the interleaved {@code block_sparse_moe} experts. Both compose the KDA
 * passthroughs ({@code -exp(A_log)}, the conv reshape, {@code dt_bias} to {@code dt_proj.bias}) and the MLA
 * {@code kv_b} split from {@link HfKdaHooks}.
 */
public final class HfLinearHooksA {

    /** The bundled MiniMax-M1 chat template, the file-template fallback (minimax.py:53-60). */
    static final String MINIMAX_TEMPLATE = "MiniMax-M1.jinja";

    private static final String MINIMAX_EXPERTS_STATE = "minimax01.experts";
    private static final String GLM_EXPERTS_STATE = "glm5next.experts";

    /** The MiniMax expert parts and their merged-name rule (minimax.py:100-114). */
    private static final List<String> MINIMAX_EXPERT_PARTS =
            Collections.unmodifiableList(Arrays.asList("w1", "w2", "w3"));
    private static final List<String> GLM_EXPERT_PARTS =
            Collections.unmodifiableList(Arrays.asList("down_proj", "gate_proj", "up_proj"));

    /** The exact F32 keys of {@code Glm5NextModel.tensor_force_quant} (glm.py:575-580). */
    private static final String[] EXACT_KEYS = {"hc_attn_", "hc_ffn_", "indexer_compressor_", "ssm_a", "ssm_dt",
            "exp_probs_b"};

    /** glm.py:569: {@code .hc_(attn|ffn)_(fn|base|scale)} and {@code .index_kpool_compress_(ape|gate)}. */
    private static final Pattern HC_UNWEIGHTED =
            Pattern.compile("\\.(hc_(?:attn|ffn)_(?:fn|base|scale)|index_kpool_compress_(?:ape|gate))$");

    private HfLinearHooksA() {
    }

    private static final Pattern MODEL_LAYER = Pattern.compile("model\\.layers\\.(\\d+)\\.");

    /** A resolver to {@link Arch#MINIMAX_01}. */
    public static HfArchResolver minimaxArch() {
        return HfArchResolver.of(Arch.MINIMAX_01);
    }

    /** A resolver to {@link Arch#GLM5_NEXT}. */
    public static HfArchResolver glmArch() {
        return HfArchResolver.of(Arch.GLM5_NEXT);
    }

    // ------------------------------------------------------------------ MiniMaxText01Model

    /**
     * {@code MiniMaxText01Model.set_gguf_parameters} (minimax.py:62-82): the generic keys, the zero-embedding
     * suppression, the residual scale (the common layernorm alpha; the betas, all 1.0, are not stored) and the
     * partial-RoPE dimension count {@code rotary_dim}.
     */
    public static void minimaxMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        List<Integer> suppress = suppressTokens(ctx);
        if (!suppress.isEmpty()) {
            ctx.log("Suppressing tokens with zero embeddings " + suppress);
            kv.set(GgufKeys.TOKENIZER_SUPPRESS_TOKENS, suppress);
        }
        double fullAlpha = requiredDouble(cfg, "layernorm_full_attention_alpha");
        double fullBeta = requiredDouble(cfg, "layernorm_full_attention_beta");
        double linAlpha = requiredDouble(cfg, "layernorm_linear_attention_alpha");
        double linBeta = requiredDouble(cfg, "layernorm_linear_attention_beta");
        double mlpAlpha = requiredDouble(cfg, "layernorm_mlp_alpha");
        double mlpBeta = requiredDouble(cfg, "layernorm_mlp_beta");
        if (fullAlpha != linAlpha || fullAlpha != mlpAlpha) {
            throw new GgufFormatException(cfg.source() + ": the three layernorm alphas differ"
                    + " (assertion of minimax.py:76)");
        }
        if (fullBeta != linBeta || fullBeta != mlpBeta || fullBeta != 1.0) {
            throw new GgufFormatException(cfg.source() + ": the three layernorm betas differ from 1.0"
                    + " (assertion of minimax.py:77)");
        }
        kv.set(GgufKeys.RESIDUAL_SCALE, fullAlpha);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, cfg.requireInt("rotary_dim"));
    }

    /**
     * {@code _get_suppress_tokens} (minimax.py:19-46): the added tokens whose embedding rows are entirely zero; a
     * token is <em>added</em> when its id is past the base vocabulary the tokenizer ships.
     *
     * @throws IOException when the embedding cannot be read
     */
    static List<Integer> suppressTokens(HfLoadContext ctx) throws IOException {
        int base = baseVocabSize(ctx);
        OptionalInt rows = HfHookSupport.embeddingRows(ctx);
        if (base <= 0 || !rows.isPresent() || rows.getAsInt() <= base) {
            return Collections.emptyList();
        }
        HfTensorIndex index = ctx.index();
        HfTensorNameMap names = ctx.names();
        RowSource src = null;
        String embd = null;
        for (String name : index.names()) {
            Optional<HfTensorNameMap.Mapped> m = names.map(name, ".weight", ".bias");
            if (m.isPresent() && TensorNames.weight(TensorNames.TOKEN_EMBD).equals(m.get().ggufName())) {
                embd = name;
                src = index.get(name);
                break;
            }
        }
        if (src == null) {
            return Collections.emptyList();
        }
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("token embedding '" + embd + "' must have rank 2, got "
                    + Arrays.toString(shape));
        }
        int hidden = (int) shape[1];
        long vocab = shape[0];
        List<Integer> out = new ArrayList<>();
        ByteBuffer row = ByteBuffer.allocate(hidden * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (long r = base; r < vocab; r++) {
            row.clear();
            src.readRows(r, 1, row);
            row.flip();
            boolean zero = true;
            for (int i = 0; i < hidden; i++) {
                if (row.getFloat() != 0f) {
                    zero = false;
                    break;
                }
            }
            if (zero) {
                out.add((int) r);
            }
        }
        return out;
    }

    /**
     * The base (non-added) token count of the tokenizer files: the {@code tokenizer.ggml.tokens} of a GGUF source,
     * the BPE model vocab of tokenizer.json, or the largest id of vocab.json / 0.
     */
    private static int baseVocabSize(HfLoadContext ctx) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            Object tokens = files.ggufTokenizerKeys().get(GgufKeys.TOKENIZER_LIST);
            return tokens instanceof List ? ((List<?>) tokens).size() : 0;
        }
        HfTokenizerJson tj = files.readTokenizerJson();
        if (tj != null) {
            return tj.model().vocabSize();
        }
        Map<String, Integer> vocabJson = files.readVocabJson();
        if (vocabJson == null) {
            return 0;
        }
        int max = -1;
        for (Integer v : vocabJson.values()) {
            if (v != null && v > max) {
                max = v;
            }
        }
        return max + 1;
    }

    /**
     * {@code MiniMaxText01Model.modify_tensors} (minimax.py:86-119): the {@code block_sparse_moe.experts.<x>.wN}
     * tensors of a layer are stacked and mapped under theconverter's {@code layers.<b>.feed_forward.experts.<part>}
     * name; every other tensor maps directly.
     */
    public static void minimaxTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("block_sparse_moe.experts")) {
            experts(ctx, ctx.config().requireInt("num_local_experts")).offer(ctx, hfName, out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code MiniMaxText01Model.prepare_tensors}: no leftover expert group. */
    public static void minimaxExtra(HfLoadContext ctx, TensorSink out) {
        completeExperts(ctx);
    }

    /**
     * {@code MiniMaxText01Model.set_vocab} (minimax.py:48-60): GPT-2 BPE, with the checkpoint's
     * {@code chat_template.jinja} or the bundled {@value #MINIMAX_TEMPLATE} overriding the template.
     */
    public static Map<String, Object> minimaxVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx)
                .withFallbackChatTemplate(HfChatTemplates.bundled(MINIMAX_TEMPLATE)), pre);
    }

    // ------------------------------------------------------------------ Glm5NextModel

    /** The block count with the MTP layers appended (glm.py:417-427). */
    public static int blockCount(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        long base = cfg.blockCount();
        long nextn = ctx.options().includeMtp() ? cfg.findInt(0, "num_nextn_predict_layers") : 0;
        if (base + nextn > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": " + base + " layers and " + nextn
                    + " MTP layers exceed " + HfConfig.MAX_LAYERS);
        }
        return (int) (base + nextn);
    }

    /**
     * {@code Glm5NextModel.filter_tensors} (glm.py:438-455): MTP layers are dropped when not kept.
     */
    public static HfNameFilter filter(HfLoadContext ctx) {
        boolean keep = ctx.options().includeMtp();
        int main = ctx.config().requireInt("num_hidden_layers");
        return (name, base) -> {
            String t = base.apply(name);
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

    /**
     * {@code Glm5NextModel.set_gguf_parameters} (glm.py:457-516): hybrid KV head array with attention layer's 1 per
     * MTP block and a 1e-6 epsilon, nope-only MLA, the kpool indexer, hyperconnections, and the uniform swiglu
     * clamp array.
     *
     * @throws GgufFormatException when a required hyperparameter is missing or a format check fails
     */
    public static void glmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        Object raw = cfg.find("layer_types").orElse(null);
        if (!(raw instanceof List) || ((List<?>) raw).size() != cfg.requireInt("num_hidden_layers")) {
            throw new GgufFormatException(cfg.source() + ": 'layer_types' must have one entry per hidden layer");
        }
        List<Object> layerTypes = new ArrayList<>((List<?>) raw);
        List<Integer> nKv = new ArrayList<>(ctx.blockCount());
        for (Object t : layerTypes) {
            nKv.add("linear_attention".equals(t) ? 0 : 1);
        }
        for (int i = layerTypes.size(); i < ctx.blockCount(); i++) {
            nKv.add(1);
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, nKv);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, 1e-6);
        long nextn = ctx.options().includeMtp() ? cfg.findInt(0, "num_nextn_predict_layers") : 0;
        if (nextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, (int) nextn);
        }
        Map<String, ?> lin = nested(cfg, "linear_attn_config");
        int nHead = cfg.requireInt("num_attention_heads");
        Object linHeads = lin.get("num_heads");
        if (!(linHeads instanceof Integer) || (Integer) linHeads != nHead) {
            throw new GgufFormatException(cfg.source() + ": linear_attn_config.num_heads must equal"
                    + " num_attention_heads (" + nHead + ")");
        }
        kv.set(GgufKeys.SSM_CONV_KERNEL, requiredMember(lin, "short_conv_kernel_size", cfg));
        kv.set(GgufKeys.KDA_HEAD_DIM, requiredMember(lin, "head_dim", cfg));
        Optional<Object> lower = optionalMember(lin, "gate_lower_bound");
        if (lower.isPresent()) {
            kv.set(GgufKeys.KDA_GATE_LOWER_BOUND, ((Number) lower.get()).doubleValue());
        }
        if (!truthy(cfg.find("mla_use_nope")) || cfg.requireInt("qk_rope_head_dim") != 0) {
            throw new GgufFormatException(cfg.source() + ": expected nope-only MLA"
                    + " (mla_use_nope and qk_rope_head_dim == 0)");
        }
        int kvLora = cfg.requireInt("kv_lora_rank");
        kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, cfg.requireInt("q_lora_rank"));
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, cfg.requireInt("qk_rope_head_dim"));
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLora);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, kvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, cfg.requireInt("qk_nope_head_dim"));
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, cfg.requireInt("v_head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, cfg.requireInt("index_n_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, cfg.requireInt("index_head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, cfg.requireInt("index_topk"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KPOOL, cfg.requireInt("index_kpool"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KPOOL_SELECT_TAIL, cfg.findBool(Boolean.TRUE, "index_kpool_always_select_tail"));
        if (cfg.find("indexer_types").isPresent()) {
            kv.set(GgufKeys.ATTENTION_INDEXER_TYPES, indexerTypes(cfg));
        }
        if (!truthy(cfg.find("mhc"))) {
            throw new GgufFormatException(cfg.source() + ": expected hyper-connections (config 'mhc')");
        }
        kv.set(GgufKeys.HYPER_CONNECTION_COUNT, cfg.requireInt("hc_mult"));
        kv.set(GgufKeys.HYPER_CONNECTION_SINKHORN_ITERATIONS, cfg.requireInt("hc_sinkhorn_iters"));
        kv.set(GgufKeys.HYPER_CONNECTION_EPSILON, requiredDouble(cfg, "hc_eps"));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("n_shared_experts"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, requiredDouble(cfg, "routed_scaling_factor"));
        if (truthy(cfg.find("norm_topk_prob"))) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
        Optional<Object> limit = cfg.find("swiglu_limit");
        if (limit.isPresent()) {
            double v = ((Number) limit.get()).doubleValue();
            List<Double> exp = new ArrayList<>(ctx.blockCount());
            List<Double> shexp = new ArrayList<>(ctx.blockCount());
            for (int i = 0; i < ctx.blockCount(); i++) {
                exp.add(v);
                shexp.add(v);
            }
            kv.set(GgufKeys.SWIGLU_CLAMP_EXP, exp);
            kv.set(GgufKeys.SWIGLU_CLAMP_SHEXP, shexp);
        }
    }

    private static List<Boolean> indexerTypes(HfConfig cfg) {
        Object v = cfg.find("indexer_types").get();
        if (!(v instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": 'indexer_types' must be an array");
        }
        List<Boolean> out = new ArrayList<>();
        for (Object t : (List<?>) v) {
            out.add("full".equals(t));
        }
        return out;
    }

    /** {@code Glm5NextModel.modify_tensors} (glm.py:518-573). */
    public static void glmTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (cfg.findBool(Boolean.FALSE, "tie_word_embeddings") && hfName.equals("lm_head.weight")) {
            out.drop(hfName, "tied output layer; token_embd.weight serves as the output");
            return;
        }
        String name = hfName;
        RowSource s = src;
        if (name.contains(".mlp.experts.")) {
            experts(ctx, cfg.requireInt("n_routed_experts")).offer(ctx, name, out);
            return;
        }
        if (name.endsWith("kv_b_proj.weight")) {
            splitKvB(ctx, name, s, out);
            return;
        }
        if (name.endsWith(".q_conv1d.weight") || name.endsWith(".k_conv1d.weight")
                || name.endsWith(".v_conv1d.weight")) {
            s = HfKdaHooks.convWeight(s, name);
        } else if (name.endsWith(".A_log")) {
            s = HfKdaHooks.aLog(ctx, s);
        } else if (name.endsWith(".dt_bias")) {
            name = HfHookSupport.renameTail(name, ".dt_bias", ".dt_proj.bias");
        } else if (HC_UNWEIGHTED.matcher(name).find()) {
            out.add(ctx.names().mapOrThrow(name) + ".weight", s);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, name, s, out);
    }

    /** {@code Glm5NextModel.tensor_force_quant} (glm.py:575-580). */
    public static GGMLType glmForceType(String ggufName, int nDims, ElementFormat source,
                                        com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType outType) {
        if (ggufName.startsWith("blk.") || ggufName.startsWith("output_hc")) {
            for (String k : EXACT_KEYS) {
                if (ggufName.contains(k)) {
                    return GGMLType.F32;
                }
            }
        }
        return null;
    }

    /** Glm5NextModel.prepare_tensors' expert-completeness check. */
    public static void glmExtra(HfLoadContext ctx, TensorSink out) {
        completeExperts(ctx);
    }

    /** {@code Glm5NextModel.set_vocab} (glm.py:429-431): the GLM ({@code _set_vocab_glm}) vocabulary. */
    public static Map<String, Object> glmVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.glm(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ shared pieces

    private static void completeExperts(HfLoadContext ctx) {
        Collectors c = ctx.state(ctx.arch().ggufName() + ".experts", Collectors.class, Collectors::new);
        synchronized (c) {
            if (c.collector != null) {
                c.collector.requireComplete();
            }
        }
    }

    private static HfHookSupport.ExpertCollector experts(HfLoadContext ctx, int nExperts) {
        Collectors c = ctx.state(ctx.arch().ggufName() + ".experts", Collectors.class, Collectors::new);
        synchronized (c) {
            if (c.collector == null) {
                c.collector = new HfHookSupport.ExpertCollector(nExperts,
                        ctx.arch() == Arch.MINIMAX_01 ? MINIMAX_EXPERT_PARTS : GLM_EXPERT_PARTS,
                        ctx.arch() == Arch.MINIMAX_01 ? MINIMAX_MERGED : HfHookSupport.MergedName.SAME_PREFIX);
            }
            return c.collector;
        }
    }

    private static HfHookSupport.MergedName MINIMAX_MERGED =
            (prefix, part, tail) -> "layers." + HfLoadContext.bidOf(prefix) + ".feed_forward.experts."
                    + part + tail;

    private static final class Collectors {
        HfHookSupport.ExpertCollector collector;
    }

    /**
     * Glm5Next's {@code kv_b_proj} split: a {@code [n_head*(nope+v), kv_lora]} tensor to
     * {@code attn_k_b [n_head, kv_lora, nope]} and {@code attn_v_b [n_head, v, kv_lora]}.
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
        out.add(ctx.names().mapOrThrow(hfName.replace("kv_b_proj", "k_b_proj")), kvB[0]);
        out.add(ctx.names().mapOrThrow(hfName.replace("kv_b_proj", "v_b_proj")), kvB[1]);
    }

    private static Optional<Object> optionalMember(Map<String, ?> map, String key) {
        Object v = map.get(key);
        return v instanceof Number && !(v instanceof Boolean) ? Optional.of(v) : Optional.empty();
    }

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

    private static double requiredDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing"));
    }

    private static int requiredMember(Map<String, ?> map, String key, HfConfig cfg) {
        Object v = map.get(key);
        if (v instanceof Integer || v instanceof Long) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException(cfg.source() + ": 'linear_attn_config." + key + "' is missing or is not"
                + " an integer: " + v);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, ?> nested(HfConfig cfg, String key) {
        Object v = cfg.find(key).orElse(null);
        if (!(v instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing"
                    + (v == null ? "" : " or is not an object"));
        }
        return (Map<String, ?>) v;
    }
}
