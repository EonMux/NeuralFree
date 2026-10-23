package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * Which embedding outputs a model offers, shared by the GPU and CPU orchestrators so both accept and reject the same
 * models with the same messages:
 * <ul>
 *   <li>an embedding model ({@link SupportClass#EMBED}) is an encoder;</li>
 *   <li>a decision model ({@link DecisionScorer#present}) has no embedding output: llama.cpp's {@code t_embd} of
 *       vllm-decision2 and clef is the decision score vector (vllm-decision2.cpp:137-146, clef.cpp:312), and the
 *       modern-bert / lfm2 decision blocks consume the hidden rows, so all of them score through classify;</li>
 *   <li>{@code encode} needs an encoder or a pooling type set by the GGUF (llama.cpp embeddings of any model);</li>
 *   <li>{@code encodeTokens} needs NONE pooling and an encoder, or pockettts whose backbone exposes its
 *       {@code t_embd} rows;</li>
 *   <li>{@code rerank} needs RANK pooling, as the llama.cpp server's rerank endpoint does.</li>
 * </ul>
 */
public final class EncodeGate {

    private EncodeGate() {
    }

    /** True for an embedding model: its output is embeddings, not token logits or decision scores. */
    public static boolean isEncoder(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.supportClass() == SupportClass.EMBED;
    }

    /**
     * Checks that {@code encode} applies to {@code spec}.
     *
     * @throws UnsupportedOperationException for a decision model, and for a model that is not an encoder and sets no
     *                                       pooling type
     */
    public static void checkEncode(ModelSpec spec) {
        rejectDecision(spec, "encode");
        if (!isEncoder(spec) && spec.head().pooling() == PoolingKind.NONE) {
            throw new UnsupportedOperationException("encode needs an embedding model or a pooling type; "
                    + spec.arch().ggufName() + " (" + spec.supportClass() + ") has neither");
        }
    }

    /**
     * Checks that {@code encodeTokens} applies to {@code spec}.
     *
     * @throws UnsupportedOperationException for a decision model, a model that pools per sequence, and a model with
     *                                       no per-token embedding output
     */
    public static void checkEncodeTokens(ModelSpec spec) {
        rejectDecision(spec, "encodeTokens");
        PoolingKind pooling = spec.head().pooling();
        if (pooling != PoolingKind.NONE || !(isEncoder(spec) || spec.arch() == Arch.POCKETTTS)) {
            throw new UnsupportedOperationException("per-token embeddings not supported by " + spec.arch().ggufName()
                    + " (" + spec.supportClass() + ", pooling " + pooling + ")");
        }
    }

    /**
     * Checks that {@code rerank} applies to {@code spec}.
     *
     * @throws UnsupportedOperationException when the model's pooling is not RANK
     */
    public static void checkRerank(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        PoolingKind pooling = spec.head().pooling();
        if (pooling != PoolingKind.RANK) {
            throw new UnsupportedOperationException("reranking needs RANK pooling; " + spec.arch().ggufName()
                    + " pools by " + pooling);
        }
    }

    private static void rejectDecision(ModelSpec spec, String what) {
        Objects.requireNonNull(spec, "spec");
        if (DecisionScorer.present(spec)) {
            throw new UnsupportedOperationException(what + ": " + spec.arch().ggufName() + " is a decision model;"
                    + " score it with classify");
        }
    }
}
