package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * Hooks of the T5 converter classes (LC/conversion/t5.py): {@code T5Model} (:20-153, the five MT5/T5/UMT5
 * names) and {@code T5EncoderModel} (:156-288). The two write their own metadata (t5.py:121-138, :259-273 —
 * no {@code super().set_gguf_parameters()} call), the Unigram t5 vocabulary with the charsmap
 * ({@link HfVocabWriters#t5Ugm}, t5.py:30-119, 165-257) and the shared-embedding selection: the first of
 * {@code decoder.embed_tokens.weight}, {@code encoder.embed_tokens.weight} and {@code shared.weight} in
 * index order becomes {@code shared.weight}, the others return (t5.py:140-153, 275-288). The relative
 * attention buckets and the decoder start id are metadata keys, and Tencent's {@code rope_freqs} of the two
 * {@code d_ff x d_model} FFN factors stay to the name map.
 */
public final class HfT5Hooks {

    /** The name the remaining shared embedding keeps (t5.py:145, 282), the one the rest of the view maps. */
    public static final String SHARED = "shared.weight";
    /** The metadata derived default context length '' (t5.py:123). */
    public static final int DEFAULT_CONTEXT = 512;

    private HfT5Hooks() {
    }

    // ------------------------------------------------------------------ vocabulary

    /** {@code set_vocab} (t5.py:27-119, 165-257): the Unigram t5 vocabulary, or SentencePiece for Pile-T5. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.t5Ugm(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code T5Model.set_gguf_parameters} (t5.py:121-138): everything on its own, with the context length
     * default of 512 and the decoder keys that only the T5 (with LM head) rows write.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg = ctx.config();
        OptionalInt nCtx = cfg.findInt("n_positions");
        kv.set(GgufKeys.CONTEXT_LENGTH, nCtx.isPresent() ? nCtx.getAsInt() : DEFAULT_CONTEXT);
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("d_model"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("d_ff"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        if (cfg.findInt("num_decoder_layers").isPresent()) {
            kv.set(GgufKeys.DECODER_BLOCK_COUNT, cfg.requireInt("num_decoder_layers"));      // t5.py:129-130
        }
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("num_heads"));
        int dKv = cfg.requireInt("d_kv");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, dKv);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, dKv);
        double eps = cfg.findFloat(1e-6, "layer_norm_epsilon");
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, eps);                                       // t5.py:134
        kv.set(GgufKeys.ATTENTION_RELATIVE_BUCKETS_COUNT, cfg.requireInt("relative_attention_num_buckets"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps);                                   // t5.py:136
        if (cfg.findInt("decoder_start_token_id").isPresent()) {
            kv.set(GgufKeys.DECODER_START_TOKEN_ID, cfg.requireInt("decoder_start_token_id"));// t5.py:137
        }
    }

    /** {@code T5EncoderModel.set_gguf_parameters} (t5.py:259-273): the same without the decoder keys. */
    public static void encoderMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        meta(ctx, kv, plan);
        kv.remove(GgufKeys.DECODER_BLOCK_COUNT);
        kv.remove(GgufKeys.DECODER_START_TOKEN_ID);
    }

    // ------------------------------------------------------------------ tensors

    /** {@code T5Model.modify_tensors} (t5.py:140-153) over the shared embedding selection, then the map. */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (SHARED_EMBEDDINGS.contains(hfName)) {
            Shared s = ctx.state(STATE_SHARED, Shared.class, Shared::new);
            if (s.first) {                                                                   // t5.py:146-148
                s.first = false;
                HfTensorHook.DEFAULT.apply(ctx, SHARED, src, out);
                return;
            }
            out.drop(hfName, "a later shared token embedding is skipped"
                    + " (t5.py:149-151, 284-286)");                                          // t5.py:150-151
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code T5EncoderModel.modify_tensors} (t5.py:275-288), the same selection. */
    public static void encoderTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        tensors(ctx, hfName, src, out);
    }

    /** The names in the converter's list of t5.py:145, 280, in the order the index visits them. */
    private static final List<String> SHARED_EMBEDDINGS = Collections.unmodifiableList(Arrays.asList(
            "decoder.embed_tokens.weight", "encoder.embed_tokens.weight", SHARED));

    /** The per-load selection state of the shared embedding. */
    private static final class Shared {
        boolean first = true;
    }

    private static final String STATE_SHARED = HfT5Hooks.class.getName() + ".shared";
}
