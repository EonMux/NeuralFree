package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * The draft hooks, part A (LC/conversion/qwen.py {@code DFlashModel} :757-1029 with its
 * {@code DSparkModel} subclass :958-1046, and LC/conversion/muse_glimmer.py {@code MuseGlimmerAssistantModel}
 * :137-163). The shared target, mask and {@code dflash_config} helpers live in {@link HfDraftHooks}; the
 * DeepSeek-V4, LiLiCorr and Gemma 4 drafts are {@link HfDraftHooksB}.
 */
public final class HfDraftHooksA {

    private static final String WEIGHT = HfDraftHooks.WEIGHT;
    private static final String EMBED_TOKENS = "model.embed_tokens.weight";
    private static final String LM_HEAD = "model.lm_head.weight";
    private static final String D2T = "model.d2t";
    private static final String T2D = "t2d";
    private static final String D2T_STATE = HfDraftHooksA.class.getName() + ".d2t";
    private static final String DFLASH_CONFIG = HfDraftHooks.DFLASH_CONFIG;
    /** The suffixes permuted when a draft checkpoint stores its Q/K per rotate-half (interleaved) rope. */
    private static final List<String> ROPE_PERMUTE_TAILS = List.of("self_attn.q_proj.weight",
            "self_attn.k_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight");

    private HfDraftHooksA() {
    }

    // ------------------------------------------------------------------ shared head geometry

    /**
     * {@code head_dim} of the configuration, else {@code hidden_size / num_attention_heads}. Used for the draft rope
     * permutation (qwen.py:796, :945) and M-RoPE sections.
     */
    static int headDimOf(HfConfig cfg) {
        OptionalInt hd = cfg.findInt("head_dim");
        if (hd.isPresent() && hd.getAsInt() > 0) {
            return hd.getAsInt();
        }
        int hidden = cfg.requireInt("hidden_size");
        int heads = cfg.requireInt("num_attention_heads", "n_head", "n_heads");
        if (heads <= 0 || hidden % heads != 0) {
            throw new GgufFormatException(cfg.source() + ": hidden_size " + hidden + " is not divisible by the head count " + heads);
        }
        return hidden / heads;
    }

    /** The {@code block_size} of the draft (qwen.py:801): {@code dflash_config.block_size}, else the flat
     * {@code block_size}, else 16. Shared with the LiLiCorr draft. */
    static int blockSize(HfConfig cfg, Map<String, Object> dc) {
        Object v = dc.get("block_size");
        if (v != null && HfDraftHooks.doubleOf(cfg, HfDraftHooks.DFLASH_CONFIG + ".block_size", v) != 0.0) {
            return (int) HfDraftHooks.longOf(cfg, HfDraftHooks.DFLASH_CONFIG + ".block_size", v);
        }
        return cfg.findInt(16, "block_size");
    }

    // ------------------------------------------------------------------ DFlashModel

    /**
     * {@code DFlashModel.set_gguf_parameters} (qwen.py:791-856) over the base Qwen chain: the rope dimension of a
     * partial rotation, the attention value scale, the block size and DFlash2 convolution/selector geometry, the
     * logit scale, softcapping and embedding scale, the shifted target layers, the sliding window, the causal flag
     * and the degenerate M-RoPE sections of an M-RoPE target.
     */
    public static void dflashMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenHooks.qwen2Meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        Map<String, Object> dc = HfDraftHooks.dflashConfig(cfg);

        OptionalDouble partial = cfg.rope().global().partialRotaryFactor();
        if (partial.isPresent()) {
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (headDimOf(cfg) * partial.getAsDouble()));
        }
        Object valueScale = firstThat(cfg, dc, "attention_value_scale");
        if (valueScale != null) {
            kv.set(GgufKeys.ATTENTION_VALUE_SCALE,
                    HfDraftHooks.doubleOf(cfg, DFLASH_CONFIG + ".attention_value_scale", valueScale));
        }
        kv.set(GgufKeys.DFLASH_BLOCK_SIZE, blockSize(cfg, dc));
        if (dc.containsKey("conv_kernel_size")) {
            kv.set(GgufKeys.DFLASH_CONV_KERNEL_SIZE, intOf(cfg, dc, "conv_kernel_size"));
            kv.set(GgufKeys.DFLASH_CONV_GROUP_SIZE, intOf(cfg, dc, "conv_group_size"));
            kv.set(GgufKeys.DFLASH_SELECTOR_RANK, intOf(cfg, dc, "selector_rank"));
            kv.set(GgufKeys.DFLASH_SELECTOR_TOP_K, intOf(cfg, dc, "selector_top_k"));
        }
        Object multiplier = firstThat(cfg, dc, "output_multiplier");
        if (multiplier != null) {
            kv.set(GgufKeys.LOGIT_SCALE, HfDraftHooks.doubleOf(cfg, "output_multiplier", multiplier));
        }
        Object softcap = firstThat(cfg, dc, "final_logit_softcapping");
        if (softcap != null) {
            double v = HfDraftHooks.doubleOf(cfg, "final_logit_softcapping", softcap);
            if (v > 0) {
                kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, v);
            }
        }
        Object scale = firstThat(cfg, dc, "input_embedding_scale");
        if (scale == null && ctx.targetPath().isPresent()) {
            HfDraftHooks.Target t = HfDraftHooks.target(ctx);
            if (t.config != null && t.config.architecture().startsWith("Gemma")) {
                scale = (float) Math.sqrt(t.config.requireInt("hidden_size"));
            }
        }
        if (scale != null) {
            kv.set(GgufKeys.EMBEDDING_SCALE, HfDraftHooks.doubleOf(cfg, "input_embedding_scale", scale));
        }
        Optional<List<Integer>> targets = cfg.findIntList("target_layer_ids");
        if (targets.isPresent() && !targets.get().isEmpty()) {
            kv.set(GgufKeys.TARGET_LAYERS, shiftByOne(ctx, targets.get(), "target_layer_ids"));
        }
        boolean useSliding = cfg.findBool(false, "use_sliding_window") || HfDraftHooks.truthy(dc.get("use_swa"));
        Integer swa = windowOf(cfg, dc);
        if (useSliding && swa != null) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, swa);
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                    swaPattern(ctx, cfg, dc, HfDraftHooks.truthy(dc.get("use_swa"))));
        }
        Optional<Boolean> causal = causalOf(cfg, dc);
        if (causal.isPresent()) {
            kv.set(GgufKeys.ATTENTION_CAUSAL, causal.get());
        }
        if (HfDraftHooks.targetUsesMrope(ctx)) {
            kv.set(GgufKeys.ROPE_DIMENSION_SECTIONS, Arrays.asList(headDimOf(cfg) / 2, 0, 0, 0));
        }
    }

    /** {@code dflash_config.get(name, hparams.get(name))}: the first value present and truthy. */
    private static Object firstThat(HfConfig cfg, Map<String, Object> dc, String name) {
        Object v = dc.get(name);
        if (v != null && HfDraftHooks.truthy(v)) {
            return v;
        }
        return cfg.flat().get(name);
    }

    /** An int of {@code dflash_config}, required when the key is there. */
    private static int intOf(HfConfig cfg, Map<String, Object> dc, String key) {
        Object v = dc.get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": " + DFLASH_CONFIG + "." + key + " must be an integer");
        }
        return (int) HfDraftHooks.longOf(cfg, DFLASH_CONFIG + "." + key, v);
    }

    /** {@code dflash_config.swa_window_size or hparams.sliding_window}, null when neither has one. */
    private static Integer windowOf(HfConfig cfg, Map<String, Object> dc) {
        Object v = dc.get("swa_window_size");
        if (v != null && HfDraftHooks.doubleOf(cfg, DFLASH_CONFIG + ".swa_window_size", v) != 0.0) {
            return (int) HfDraftHooks.longOf(cfg, DFLASH_CONFIG + ".swa_window_size", v);
        }
        OptionalInt hp = cfg.findInt("sliding_window");
        return hp.isPresent() ? hp.getAsInt() : null;
    }

    /** qwen.py:841-842: true on every block under {@code use_swa}, else the sliding_attention layer types. */
    private static List<Boolean> swaPattern(HfLoadContext ctx, HfConfig cfg, Map<String, Object> dc,
                                            boolean useSwa) {
        List<Boolean> out = new ArrayList<>(ctx.blockCount());
        if (useSwa) {
            for (int i = 0; i < ctx.blockCount(); i++) {
                out.add(Boolean.TRUE);
            }
            return out;
        }
        Object layerTypes = cfg.flat().get("layer_types");
        if (layerTypes instanceof List) {
            for (Object t : (List<?>) layerTypes) {
                out.add("sliding_attention".equals(t));
            }
        }
        return out;
    }

    /** hparams.is_causal, else dflash_config.causal (qwen.py:846-850). */
    private static Optional<Boolean> causalOf(HfConfig cfg, Map<String, Object> dc) {
        Optional<Boolean> h = cfg.findBool("is_causal");
        if (h.isPresent()) {
            return h;
        }
        Object v = dc.get("causal");
        if (v instanceof Boolean) {
            return Optional.of((Boolean) v);
        }
        if (v instanceof Number) {
            return Optional.of(((Number) v).doubleValue() != 0);
        }
        return Optional.empty();
    }

    /** {@code target_layer_ids} of the draft shifted by one. */
    private static List<Integer> shiftByOne(HfLoadContext ctx, List<Integer> ids, String where) {
        List<Integer> out = new ArrayList<>(ids.size());
        for (int id : ids) {
            if (id < 0 || id > HfConfig.MAX_LAYERS) {
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

    // ------------------------------------------------------------------ the rest of DFlash

    /**
     * {@code DFlashModel.filter_tensors} (qwen.py:923-930): a name without a {@code model.} prefix gets one; a sink
     * tensor without a {@code .weight} suffix gets one.
     */
    public static String dflashRename(String name) {
        String t = name.startsWith("model.") ? name : "model." + name;
        if (t.contains("sink") && !t.endsWith(WEIGHT)) {
            t = t + WEIGHT;
        }
        return t;
    }

    public static HfNameFilter dflashFilter(HfLoadContext ctx) {
        return (name, base) -> base.apply(dflashRename(name));
    }

    /** The mapped name of a checkpoint tensor: the DFlash rename, then the codebook suffixes (qwen.py:949-953). */
    private static String dflashName(HfLoadContext ctx, String hfName) {
        String t = dflashRename(hfName);
        if (t.equals("model.candidate_selector.predecessor_codebook")
                || t.equals("model.candidate_selector.successor_codebook")) {
            return t + WEIGHT;
        }
        return t;
    }

    /** qwen.py:943-947: the non-NeoX rope permutation of the q/k projections and norms. */
    static RowSource dflashRopePermute(HfLoadContext ctx, String hfName, RowSource src) {
        int headDim = headDimOf(ctx.config());
        long d0 = src.shape().length == 0 ? 0 : src.shape()[0];
        if (d0 % headDim != 0) {
            throw new GgufFormatException("tensor '" + hfName + "' of shape " + Arrays.toString(src.shape())
                    + " cannot be permuted per head dim " + headDim);
        }
        return RowTransforms.normalToNeoxPartial(src, (int) (d0 / headDim), headDim, headDim);
    }

    /** Whether the draft stores rotate-half rope rows: {@code rope_is_neox_style} not named true. */
    private static boolean interleavedRope(HfLoadContext ctx) {
        Boolean neox = ctx.config().findBool("rope_is_neox_style").orElse(Boolean.TRUE);
        return !neox;
    }

    /**
     * {@code DFlashModel.modify_tensors} (qwen.py:939-955): the token embedding and, when the draft does not hold
     * one, the output head come from the target under the mask patch (planned by {@link #dflashExtra}); the
     * interleaved rope checkpoints permute the q/k projections and norms; the candidate-selector codebooks are
     * weights.
     */
    public static void dflashTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals(EMBED_TOKENS) || hfName.equals(LM_HEAD)) {
            if (HfDraftHooks.maskOf(ctx) != null) {
                out.consume(hfName);   // planned in dflashExtra from the target or the patched rows
                return;
            }
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        String name = dflashName(ctx, hfName);
        RowSource s = src;
        for (String tail : ROPE_PERMUTE_TAILS) {
            if (name.endsWith(tail) && interleavedRope(ctx)) {
                // Interleaved-rope checkpoints (rope_is_neox_style = false): per head, even entries then odd.
                s = dflashRopePermute(ctx, name, src);
                break;
            }
        }
        HfTensorHook.DEFAULT.apply(ctx, name, s, out);
    }

    /**
     * {@code DFlashModel.generate_extra_tensors}' mask patch (qwen.py:858-912):
     * <ul>
     *   <li>{@code mask_embedding.pt} is a pickle, which NF never unpickles: the {@code mask_embedding.safetensors}
     *       sibling provides the mask tokens, else {@link com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers#forFile}
     *       is asked for the {@code .pt}, else the pickle error is thrown; a checkpoint without a mask plans its own
     *       embeddings;</li>
     *   <li>the token embedding becomes the masked token's patched table (the draft's when it holds
     *       {@code model.embed_tokens.weight}, the target's otherwise), with the shape the draft asserts;</li>
     *   <li>when the draft holds no {@code model.lm_head.weight}, the target's output head becomes it (the tied
     *       target uses its token embedding).</li>
     * </ul>
     */
    public static void dflashExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        HfDraftHooks.Mask mask = HfDraftHooks.maskOf(ctx);
        if (mask == null) {
            return;
        }
        HfTensorIndex index = ctx.index();
        RowSource embeddings = index.has(EMBED_TOKENS) ? index.get(EMBED_TOKENS)
                : HfDraftHooks.targetTensor(ctx, EMBED_TOKENS);
        long[] shape = embeddings.shape();
        if (shape.length != 2 || shape[0] != ctx.config().requireInt("vocab_size")
                || shape[1] != ctx.config().requireInt("hidden_size")) {
            throw new GgufFormatException("Target token embedding shape " + Arrays.toString(shape)
                    + " does not match the DFlash draft (vocab_size " + ctx.config().requireInt("vocab_size")
                    + ", hidden_size " + ctx.config().requireInt("hidden_size") + ")");
        }
        out.add(ctx.names().mapOrThrow(EMBED_TOKENS), HfDraftHooks.patch(embeddings, mask));
        if (!index.has(LM_HEAD)) {
            boolean tied = HfDraftHooks.target(ctx).config.findBool(false, "tie_word_embeddings");
            String headName = tied ? "model.embed_tokens.weight" : "lm_head.weight";
            out.add(ctx.names().mapOrThrow(LM_HEAD), HfDraftHooks.targetTensor(ctx, headName));
        }
    }

    /**
     * {@code DFlashModel.set_vocab} (qwen.py:760-789): the target class's vocabulary (the draft family's inherited
     * writer when the dispatch would recurse), plus {@code tokenizer.ggml.mask_token_id} when
     * {@code dflash_config.mask_token_id} is present.
     */
    public static Map<String, Object> dflashVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfDraftHooks.targetVocab(ctx, pre, HfDraftHooksA::qwen3Fallback));
        int maskId = HfDraftHooks.maskTokenId(ctx.config());
        if (maskId >= 0) {
            md.put(GgufKeys.TOKENIZER_MASK_ID, maskId);
        }
        return Collections.unmodifiableMap(md);
    }

    private static Map<String, Object> qwen3Fallback(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfQwenHooks.qwen3Vocab(ctx, pre);
    }

    // ------------------------------------------------------------------ DSparkModel

    /** {@code DSparkModel.filter_tensors} (qwen.py:1012-1016): the target-decoder horizon {@code t2d} is unused. */
    public static HfNameFilter dsparkFilter(HfLoadContext ctx) {
        HfNameFilter dflash = dflashFilter(ctx);
        return (name, base) -> name.equals(T2D) ? null : dflash.filter(name, base);
    }

    /** The {@code d2t} table of one load, read from "model.d2t". */
    private static final class D2t {
        RowSource value;
    }

    private static D2t d2tOf(HfLoadContext ctx) {
        return ctx.state(D2T_STATE, D2t.class, D2t::new);
    }

    /**
     * {@code DSparkModel.modify_tensors} (qwen.py:1018-1026): "model.d2t" is held for {@link #dsparkExtra}; the
     * output head is dropped when the draft keeps the full target vocabulary (the head is tied).
     */
    public static void dsparkTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals(D2T)) {
            d2tOf(ctx).value = src;
            out.consume(hfName);
            return;
        }
        int nVocabDraft = ctx.config().findInt("draft_vocab_size")
                .orElseGet(() -> ctx.config().requireInt("vocab_size"));
        if (hfName.endsWith("lm_head.weight") && nVocabDraft == ctx.config().requireInt("vocab_size")) {
            out.drop(hfName, "DSpark keeps the full target vocabulary: the output head is the shared embedding");
            return;
        }
        dflashTensors(ctx, hfName, src, out);
    }

    /**
     * {@code DSparkModel.prepare_tensors}' {@code d2t} (qwen.py:1028-1046), after the DFlash mask patch: a draft
     * with {@code draft_vocab_size < vocab_size} requires the table; its ids plus their index become I64 absolute
     * target token ids, in-range and unique, EAGLE-3's checks with the DSpark bounds.
     */
    public static void dsparkExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        dflashExtra(ctx, out);
        HfConfig cfg = ctx.config();
        int nVocab = cfg.requireInt("vocab_size");
        int nVocabDraft = cfg.findInt("draft_vocab_size").orElse(nVocab);
        RowSource held = d2tOf(ctx).value;
        if (held == null) {
            if (nVocabDraft < nVocab) {
                throw new GgufFormatException("draft_vocab_size " + nVocabDraft + " < vocab_size " + nVocab
                        + " but no d2t table found");
            }
            return;
        }
        long[] ids = HfEagle3Hooks.readIntegers(held, nVocabDraft);
        for (int i = 0; i < ids.length; i++) {
            long abs = ids[i] > Long.MAX_VALUE - i ? Long.MAX_VALUE : ids[i] + i;
            if (abs < 0 || abs >= nVocab) {
                throw new GgufFormatException("d2t target ids out of range for target vocab size " + nVocab
                        + " (entry " + i + " maps to " + abs + ")");
            }
            ids[i] = abs;
        }
        BitSet seen = new BitSet(nVocab);
        for (long id : ids) {
            int v = (int) id;
            if (seen.get(v)) {
                throw new GgufFormatException("d2t contains duplicate target ids (" + v + ")");
            }
            seen.set(v);
        }
        out.add(ctx.names().ggufName("D2T", -1), RopeTensors.fromLongs(new long[]{ids.length}, ids));
    }

    /** {@code DSparkModel.set_vocab}: the DFlash vocabulary dispatch with the draft's mask token. */
    public static Map<String, Object> dsparkVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return dflashVocab(ctx, pre);
    }

    // ------------------------------------------------------------------ MuseGlimmerAssistantModel

    /**
     * {@code MuseGlimmerAssistantModel.set_vocab} (muse_glimmer.py:141-156): the target's class owns the
     * vocabulary, with the plain GPT-2 writer for a MuseGlimmerAssistant target.
     */
    public static Map<String, Object> museGlimmerAssistantVocab(HfLoadContext ctx, PreNameResolver pre)
            throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfDraftHooks.targetVocab(ctx, pre, (c, p) ->
                HfVocabWriters.gpt2(HfHookSupport.vocabInput(c), p)));
        int mask = maskTokenIdFromRoot(ctx.config());
        if (mask >= 0) {
            md.put(GgufKeys.TOKENIZER_MASK_ID, mask);
        }
        return Collections.unmodifiableMap(md);
    }

    /** {@code hparams.mask_token_id} of the assistant, or -1. */
    private static int maskTokenIdFromRoot(HfConfig cfg) {
        Object v = cfg.flat().get("mask_token_id");
        return v == null ? -1 : (int) HfDraftHooks.longOf(cfg, "mask_token_id", v);
    }

    /**
     * {@code MuseGlimmerAssistantModel.set_gguf_parameters} (muse_glimmer.py:158-166): the block size, the
     * shifted target layers and the sliding window when the layer types name one.
     */
    public static void museGlimmerAssistantMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.DFLASH_BLOCK_SIZE, cfg.requireInt("block_size"));
        kv.set(GgufKeys.TARGET_LAYERS, shiftByOne(ctx, requireIntList(cfg, "target_layer_ids"), "target_layer_ids"));
        OptionalInt window = cfg.findInt("sliding_window");
        Object layerTypes = cfg.flat().get("layer_types");
        if (window.isPresent() && window.getAsInt() > 0 && layerTypes instanceof List) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, window.getAsInt());
            List<Boolean> pattern = new ArrayList<>(((List<?>) layerTypes).size());
            for (Object t : (List<?>) layerTypes) {
                pattern.add("sliding_attention".equals(t));
            }
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        }
    }

    /**
     * {@code MuseGlimmerAssistantModel.modify_tensors} (muse_glimmer.py:168-170): DFlash defaults to NEOX
     * (rotate-half) rope, so the Hugging Face q/k rows map as they are.
     */
    public static void museGlimmerAssistantTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }
}
