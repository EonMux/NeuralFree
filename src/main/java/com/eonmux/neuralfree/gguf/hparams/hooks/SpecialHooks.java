package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import java.util.Objects;

/**
 * {@link ArchHparamsHook} implementations for the encoder, classifier, diffusion, drafter, audio,
 * encoder-decoder and looped architectures, referenced from
 * {@link com.eonmux.neuralfree.gguf.arch.ArchTableSpecial}. The hooks live in {@link SpecialHooksEncoder},
 * {@link SpecialHooksHeads}, {@link SpecialHooksGen} and {@link SpecialHooksDraft}; this class holds the
 * helpers they share.
 *
 * <p>Checks that llama.cpp performs in {@code load_arch_tensors} but that depend only on metadata (token type
 * counts, head-size equalities, expert counts) run in the hooks so that a malformed file fails at load.
 */
public final class SpecialHooks {

    private SpecialHooks() {
    }

    /**
     * Hook of an architecture whose C++ class derives from another one and calls the base
     * {@code load_arch_hparams} first (clef.cpp:8, vllm-decision2.cpp:8), or reuses it unchanged
     * (models.h:195-203 llama-embed, models.h:735-737 qwen3tts with {@code own} = {@link ArchHparamsHook#NONE}).
     */
    public static ArchHparamsHook extend(ArchHparamsHook base, ArchHparamsHook own) {
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(own, "own");
        return (b, kv) -> {
            base.apply(b, kv);
            own.apply(b, kv);
        };
    }

    /** Format error prefixed with the architecture name. */
    static GgufFormatException invalid(ModelHparamsBuilder b, String message) {
        return new GgufFormatException(b.arch().ggufName() + ": " + message);
    }

    /** Error for a required key that is absent. */
    static GgufFormatException missingKey(GgufKv kv, String key) {
        return new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
    }

    /** Required boolean key ({@code get_key} without the optional flag). */
    static boolean boolRequired(GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw missingKey(kv, key);
        }
        return kv.bool(key, false);
    }

    /**
     * BERT-style token type count {@code tokenizer.ggml.token_type_count}, read by the vocabulary loader
     * (llama-vocab.cpp:2204) and 0 when absent.
     */
    static int tokenTypeCount(GgufKv kv) {
        return kv.u32(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, 0);
    }

    /**
     * The BERT-graph loaders refuse a vocabulary without token types (bert.cpp:26-28, nomic-bert.cpp:16-18,
     * nomic-bert-moe.cpp:17-19, jina-bert-v3.cpp:16-18).
     */
    static void requireTokenTypes(ModelHparamsBuilder b, GgufKv kv) {
        if (tokenTypeCount(kv) == 0) {
            throw invalid(b, "model needs to define token type count ("
                    + kv.key(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT) + ")");
        }
    }

    /** Key head size of layer 0, {@code n_embd_head_k()} (llama-hparams.cpp:140-146). */
    static int headK0(ModelHparamsBuilder b) {
        return b.isSwa(0) ? b.headKSwa() : b.headKFull();
    }

    /** Value head size of layer 0, {@code n_embd_head_v()} (llama-hparams.cpp:148-154). */
    static int headV0(ModelHparamsBuilder b) {
        return b.isSwa(0) ? b.headVSwa() : b.headVFull();
    }

    /**
     * Requires equal key and value head sizes for full and sliding-window layers
     * (gemma-embedding2.cpp:27-32, gemma4-assistant.cpp:24-29).
     */
    static void requireEqualHeadSizes(ModelHparamsBuilder b) {
        if (headK0(b) != headV0(b)) {
            throw invalid(b, "key and value head sizes must be equal, got " + headK0(b) + " and " + headV0(b));
        }
        if (b.headKSwa() != b.headVSwa()) {
            throw invalid(b, "sliding-window key and value head sizes must be equal, got " + b.headKSwa() + " and "
                    + b.headVSwa());
        }
    }
}
