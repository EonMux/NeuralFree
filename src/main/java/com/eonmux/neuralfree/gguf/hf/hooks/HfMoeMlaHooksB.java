package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

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
 * Hooks of the standalone MLA converter classes, part B: {@code Glm4MoeModel} (LC/conversion/glm.py:113-245), its
 * subclass {@code SolarOpenModel} (glm.py:387-405), {@code MiniCPM3Model} (minicpm.py:66-114), {@code
 * MiniMaxM3Model} (minimax.py:169-206) and {@code PLMModel} (plm.py:8-24). The DeepseekV2-composing classes live in
 * {@link HfMoeMlaHooksA}, the native Mistral MoE format in {@link HfMistralMoeHooks}; every method is a public
 * static hook reference the entries register.
 */
public final class HfMoeMlaHooksB {

    /** The merged expert parts, in the order the GLM MoE converter writes them (glm.py:219). */
    static final List<String> GLM_PARTS = List.of("down_proj", "gate_proj", "up_proj");

    private static final String GLM_EXPERTS_STATE = HfMoeMlaHooksB.class.getName() + ".glmExperts";

    /** Solar-Open's added vocabulary specials (glm.py:400-403, read byte-exact). */
    static final String SOLAR_EOS = "<|endofassistant|>";
    static final String SOLAR_EOT = "<|endofassistant|>";
    static final String SOLAR_UNK = "<unk>";
    static final String SOLAR_BOS = "<|startoftext|>";

    /** Default partial_rotary_factor of Glm4Moe (glm.py:158). */
    static final double GLM4_MOE_PARTIAL_ROTARY_DEFAULT = 0.5;

    private HfMoeMlaHooksB() {
    }

    // ------------------------------------------------------------------ Glm4MoeModel (and SolarOpenModel)

    /**
     * {@code Glm4MoeModel.set_gguf_parameters} (glm.py:151-184) after the generic metadata: the RoPE width over
     * {@code partial_rotary_factor} (default 0.5), the routed expert counts, the sigmoid gating function, the weight
     * scale and norm, and the NextN layer count.
     */
    public static void glm4MoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int ropeDim;
        if (cfg.findInt("head_dim").isPresent()) {
            ropeDim = cfg.requireInt("head_dim");
        } else {
            ropeDim = Math.floorDiv(cfg.requireInt("hidden_size"), cfg.requireInt("num_attention_heads"));
        }
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (ropeDim * HfMoeMlaHooksA.partialRotary(cfg,
                GLM4_MOE_PARTIAL_ROTARY_DEFAULT)));
        OptionalInt routed = cfg.findInt("n_routed_experts");
        if (routed.isPresent()) {
            kv.set(GgufKeys.EXPERT_COUNT, routed.getAsInt());
        }
        OptionalInt moeFf = cfg.findInt("moe_intermediate_size");
        if (moeFf.isPresent()) {
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, moeFf.getAsInt());
        }
        OptionalInt shared = cfg.findInt("n_shared_experts");
        if (shared.isPresent()) {
            kv.set(GgufKeys.EXPERT_SHARED_COUNT, shared.getAsInt());
        }
        OptionalInt lead = cfg.findInt("first_k_dense_replace");
        if (lead.isPresent()) {
            kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, lead.getAsInt());
        }
        kv.set(GgufKeys.EXPERT_GATING_FUNC, 2);
        Optional<Object> scale = cfg.find("routed_scaling_factor");
        if (scale.isPresent() && scale.get() instanceof Number && !(scale.get() instanceof Boolean)) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, ((Number) scale.get()).doubleValue());
        }
        if (truthyAndPresent(cfg, "norm_topk_prob")) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
        OptionalInt nextn = cfg.findInt("num_nextn_predict_layers");
        if (ctx.options().includeMtp() && nextn.isPresent()) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn.getAsInt());
        }
    }

    /** Python truthiness that requires the value to be present and non-false. */
    private static boolean truthyAndPresent(HfConfig cfg, String key) {
        Object o = cfg.flat().get(key);
        if (o instanceof Boolean) {
            return (Boolean) o;
        }
        if (o instanceof Number) {
            return ((Number) o).doubleValue() != 0;
        }
        if (o instanceof String) {
            return !((String) o).isEmpty();
        }
        return o != null && !(o instanceof List) || o instanceof List && !((List<?>) o).isEmpty();
    }

    /**
     * {@code Glm4MoeModel.modify_tensors} (glm.py:201-237): the main token embedding (not the layer-specific NextN
     * one) is planned under {@code token_embd.weight}; the routed experts of a layer are stacked into one tensor per
     * part ({@link #glm4MoeExtra} fails on leftovers); everything else maps directly.
     *
     * @throws GgufFormatException when the expert count is missing, an expert name is not a member of a complete
     *                             expert set, or a name does not map
     */
    public static void glm4MoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals("model.embed_tokens.weight")) {
            out.add(ctx.names().mapOrThrow("token_embd.weight"), src);
            return;
        }
        if (hfName.contains("mlp.experts")) {
            if (!glmExperts(ctx).offer(ctx, hfName, out)) {
                throw new GgufFormatException("Unprocessed experts: tensor '" + hfName + "' is not an expert member"
                        + " <prefix>.experts.<x>.<part> with a part in " + GLM_PARTS);
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code Glm4MoeModel.prepare_tensors} (glm.py:238-245): leftover experts are invalid. */
    public static void glm4MoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, GLM_EXPERTS_STATE, "mlp.experts");
    }

    /** The expert collector of glm.py:208-235, one per load. */
    private static HfHookSupport.ExpertCollector glmExperts(HfLoadContext ctx) {
        return HfMoeClassicHooksA.stackedExperts(ctx, GLM_EXPERTS_STATE,
                ctx.config().requireInt("n_routed_experts"), GLM_PARTS);
    }

    /** The block count: the NextN layers appended when the view keeps MTP (glm.py:118-122). */
    public static int glm4MoeBlockCount(HfLoadContext ctx) {
        return ctx.options().includeMtp()
                ? ctx.config().requireInt("num_hidden_layers") + HfMoeMlaHooksA.nextnLayers(ctx.config())
                : ctx.config().requireInt("num_hidden_layers");
    }

    /** {@code SolarOpenModel.set_vocab} (glm.py:391-404): the gpt2 base with the added-vocabulary specials. */
    public static Map<String, Object> solarOpenVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(
                HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre));
        HfTokenizerFiles files = ctx.tokenizerFiles();
        int nVocab = HfHookSupport.embeddingRows(ctx).orElseThrow(() -> new GgufFormatException(
                "a Solar-Open checkpoint has no token embedding to size its vocabulary"));
        HfSpecialVocab sv = HfSpecialVocab.load(files, files.readTokenizerJson(), ctx.config().raw(), true, nVocab);
        sv = sv.withSpecialToken("eos", addedVocabId(files, SOLAR_EOS, nVocab));
        sv = sv.withSpecialToken("eot", addedVocabId(files, SOLAR_EOT, nVocab));
        sv = sv.withSpecialToken("unk", addedVocabId(files, SOLAR_UNK, nVocab));
        sv = sv.withSpecialToken("bos", addedVocabId(files, SOLAR_BOS, nVocab));
        sv.writeTo(md);
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ MiniCPM3Model

    /**
     * {@code MiniCPM3Model.set_gguf_parameters} (minicpm.py:69-85): the generic keys cover everything the class
     * writes, in table order.
     */
    public static void miniCpm3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
    }

    /**
     * {@code MiniCPM3Model.generate_extra_tensors} (minicpm.py:87-101): the long and short LongRoPE factors from the
     * configuration's rope parameters, both present or neither ({@link RopeTensors#longropeFactors}).
     */
    public static void miniCpm3Extra(HfLoadContext ctx, TensorSink out) throws IOException {
        RopeTensors.longropeFactors(ctx.config(), RopeTensors.Factors.LONG)
                .ifPresent(f -> out.add(ctx.names().ggufName("ROPE_FACTORS_LONG", -1, ".weight"), f));
        RopeTensors.longropeFactors(ctx.config(), RopeTensors.Factors.SHORT)
                .ifPresent(f -> out.add(ctx.names().ggufName("ROPE_FACTORS_SHORT", -1, ".weight"), f));
    }

    /** {@code MiniCPM3Model.set_vocab} ({@code _set_vocab_sentencepiece}, minicpm.py:102-104). */
    public static Map<String, Object> miniCpm3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ MiniMaxM3Model

    /**
     * {@code MiniMaxM3Model.set_gguf_parameters} (minimax.py:177-199) after the MiniMax M2 metadata: the shared
     * experts, the weights scale, the top-k norm, the block-sparse indexer parameters of
     * {@code sparse_attention_config} and the leading dense block count derived from {@code moe_layer_freq}.
     */
    public static void miniMaxM3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfMiniMaxHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        OptionalInt shared = cfg.findInt("n_shared_experts");
        if (shared.isPresent()) {
            kv.set(GgufKeys.EXPERT_SHARED_COUNT, shared.getAsInt());
        }
        OptionalDouble scale = cfg.findFloat("routed_scaling_factor");
        if (scale.isPresent()) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, scale.getAsDouble());
        }
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        Map<String, Object> sac = subConfig(cfg, "sparse_attention_config");
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, subInt(cfg, sac, "sparse_num_index_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, subInt(cfg, sac, "sparse_index_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, subInt(cfg, sac, "sparse_topk_blocks"));
        kv.set(GgufKeys.ATTENTION_INDEXER_BLOCK_SIZE, subInt(cfg, sac, "sparse_block_size"));
        kv.set(GgufKeys.ATTENTION_INDEXER_LOCAL_BLOCKS, subInt(cfg, sac, "sparse_local_block"));
        Object freq = cfg.flat().get("moe_layer_freq");
        if (freq instanceof List) {
            int lead = 0;
            for (Object v : (List<?>) freq) {
                if (!(v instanceof Number) || ((Number) v).doubleValue() != 0) {
                    break;
                }
                lead++;
            }
            kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, lead);
        }
    }

    /**
     * {@code MiniMaxM3Model.modify_tensors} (minimax.py:200-206): the Gemma-style {@code 1 + w} norm weights have
     * the +1 baked in, then the MiniMax M2 expert stacking and default mapping.
     */
    public static void miniMaxM3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource src2 = hfName.endsWith("norm.weight") ? ElementTransforms.addScalar(src, 1f) : src;
        HfMiniMaxHooks.tensors(ctx, hfName, src2, out);
    }

    /** {@code MiniMaxM3Model.tensor_force_quant} (minimax.py:172-175): the DSA indexer tensors are F32. */
    public static GGMLType miniMaxM3ForceType(String ggufName, int nDims, ElementFormat source,
                                              ModelSourceOptions.OutType outType) {
        return ggufName.contains(".indexer.") ? GGMLType.F32 : null;
    }

    // ------------------------------------------------------------------ PLMModel

    /**
     * {@code PLMModel.set_gguf_parameters} (plm.py:14-21) after the generic metadata: the MLA-lite keys, the
     * vocabulary size, the KV LoRA rank, key and value lengths, and the RoPE width over {@code qk_rope_head_dim}.
     */
    public static void plmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, cfg.requireInt("kv_lora_rank"));
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, cfg.requireInt("qk_nope_head_dim") + cfg.requireInt("qk_rope_head_dim"));
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, cfg.requireInt("v_head_dim"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, cfg.requireInt("qk_rope_head_dim"));
    }

    /** The GPT-2 vocabulary of PLM ({@code _set_vocab_gpt2}, plm.py:11-12). */
    public static Map<String, Object> plmVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ helpers

    /** The id of {@code text} in the checkpoint's added vocabulary ({@code tokenizer.get_added_vocab()}). */
    static Integer addedVocabId(HfTokenizerFiles files, String text, int nVocab) throws IOException {
        Map<String, Integer> added = files.readAddedTokens();
        Integer id = added.get(text);
        if (id == null) {
            HfTokenizerJson tj = files.readTokenizerJson();
            if (tj != null) {
                for (HfTokenizerJson.AddedToken t : tj.addedTokens()) {
                    if (t.content().equals(text)) {
                        return t.id();
                    }
                }
            }
            throw new GgufFormatException("AddedVocabulary has no token '" + text + "' (the Solar-Open tokenizer"
                    + " defines it in its added vocabulary)");
        }
        return id;
    }

    /** A nested configuration object ({@code self.find_hparam} over an object key). */
    static Map<String, Object> subConfig(HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be an object, got "
                    + (v == null ? "nothing" : v.getClass().getSimpleName()));
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
            out.put(String.valueOf(e.getKey()), e.getValue());
        }
        return out;
    }

    /** An integer of a nested configuration object. */
    static int subInt(HfConfig cfg, Map<String, Object> sub, String key) {
        Object v = sub.get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be an integer, got " + v);
        }
        double d = ((Number) v).doubleValue();
        return Math.toIntExact(Math.round(d));
    }
}
