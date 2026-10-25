package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * The hooks of {@code Gemma3Model} (LC/conversion/gemma.py:128-183): Gemma 3 text models, whose RMS norms store
 * {@code weight - 1}, whose token embedding carries rows past the vocabulary, and whose metadata fills the
 * converter's defaults. {@link HfGemma4Hooks} composes these hooks as {@code Gemma4Model} inherits the class.
 */
public final class HfGemmaHooks {

    private static final int DEFAULT_CONTEXT = 131072;
    private static final int DEFAULT_HEADS = 8;
    private static final int DEFAULT_KV_HEADS = 4;
    private static final int DEFAULT_HEAD_DIM = 256;
    private static final double DEFAULT_RMS_EPS = 1e-6;
    private static final double DEFAULT_THETA = 1_000_000.0;
    private static final String EMBED_TOKENS = "embed_tokens.weight";

    private HfGemmaHooks() {
    }

    /**
     * {@code Gemma3Model.set_vocab} (LC/conversion/gemma.py:134-139): the SentencePiece vocabulary with
     * {@code add_space_prefix} false when the checkpoint has {@code tokenizer.model}, else the GPT-2 vocabulary.
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        if (files.file(HfTokenizerFiles.TOKENIZER_MODEL) == null) {
            return HfVocabWriters.gpt2(in, pre);
        }
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.sentencePiece(in, pre));
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, false);
        return Collections.unmodifiableMap(md);
    }

    /**
     * {@code Gemma3Model.set_gguf_parameters} (LC/conversion/gemma.py:141-158): the generic metadata, then the
     * converter's defaults for values the configuration may omit (context 131072, 8 heads, RMS epsilon 1e-6, head
     * width 256, 4 KV heads), the global RoPE base of {@code full_attention} (default 1e6; the local base is the
     * generic {@code rope.freq_base_swa}), the final logit soft-capping when set, and the sliding window unless
     * {@code sliding_window_pattern} is 1.
     *
     * @throws GgufFormatException when {@code attn_logit_softcapping} is set, which Gemma 3 removed, or the sliding
     *                             window is required and missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.findInt(DEFAULT_CONTEXT, "max_position_embeddings"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.findInt(DEFAULT_HEADS, "num_attention_heads"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, cfg.findFloat(DEFAULT_RMS_EPS, "rms_norm_eps"));
        int headDim = cfg.findInt(DEFAULT_HEAD_DIM, "head_dim");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        kv.set(GgufKeys.ROPE_FREQ_BASE, cfg.rope().global().theta().orElse(DEFAULT_THETA));
        if (cfg.find("attn_logit_softcapping").isPresent()) {
            throw new GgufFormatException(cfg.source() + ": attn_logit_softcapping is set, but Gemma 3 has no"
                    + " attention logit soft-capping");
        }
        OptionalDouble finalSoftcap = cfg.findFloat("final_logit_softcapping");
        if (finalSoftcap.isPresent() && finalSoftcap.getAsDouble() != 0) {
            kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, finalSoftcap.getAsDouble());
        }
        if (!isOne(cfg.flat().get("sliding_window_pattern"))) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.findInt(DEFAULT_KV_HEADS, "num_key_value_heads"));
    }

    /** Python {@code v == 1} for a JSON value. */
    private static boolean isOne(Object v) {
        return v instanceof Boolean ? (Boolean) v : v instanceof Number && ((Number) v).doubleValue() == 1.0;
    }

    /**
     * {@code Gemma3Model.modify_tensors} (LC/conversion/gemma.py:160-183): the token embedding cut to the vocabulary,
     * every {@code *norm.weight} (the q/k norms included) shifted by +1 ({@code norm_shift}, gemma.py:131-132), then
     * the mapped tensor.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        plan(ctx, hfName, src, out, true);
    }

    /**
     * The Gemma 3 tensor step with or without the norm shift: {@code Gemma4Model} inherits {@code modify_tensors}
     * with a zero {@code norm_shift} (LC/conversion/gemma.py:637-639).
     */
    static void plan(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, boolean normShift)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource s = truncateEmbedding(ctx, hfName, src);
        if (normShift && HfHookSupport.isNorm(hfName)) {
            s = HfHookSupport.normPlusOne(s);
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, s, out);
    }

    /**
     * The out-of-vocabulary rows of {@code embed_tokens.weight} removed ({@code data_torch[:n_vocab_real]}): the
     * embedding keeps the first {@link HfLoadContext#vocabTokenCount()} rows, the length of the token list the
     * vocabulary step produced; a shorter embedding is kept whole.
     *
     * @throws IllegalStateException when the vocabulary step has not run
     */
    static RowSource truncateEmbedding(HfLoadContext ctx, String hfName, RowSource src) {
        if (!hfName.contains(EMBED_TOKENS)) {
            return src;
        }
        OptionalInt count = ctx.vocabTokenCount();
        if (!count.isPresent()) {
            throw new IllegalStateException("the token embedding '" + hfName + "' is planned before the vocabulary"
                    + " token count is known");
        }
        long[] shape = src.shape();
        if (shape.length == 0) {
            throw new GgufFormatException("token embedding '" + hfName + "' " + Arrays.toString(shape)
                    + " has no rows");
        }
        return count.getAsInt() < shape[0] ? RowTransforms.truncateRows(src, count.getAsInt()) : src;
    }
}
