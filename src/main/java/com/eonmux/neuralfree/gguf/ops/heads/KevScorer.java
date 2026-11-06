package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Objects;

/**
 * The kev decision read over a plain generative forward's hidden rows: every option of the prompt ends on the
 * {@value #MARKER_TEXT} token, and an option's score is the dot product of the last prompt token's hidden state
 * with the end token's, both rows read as one query and one key half of {@code n_embd_out} (llama.cpp's
 * send_decision, server-context.cpp:2572-2591): {@code score = dot(q, k) / sqrt(n_embd_out / 2)}. Immutable.
 */
public final class KevScorer {

    /** The token that ends every option of a kev prompt (llama.cpp's {@code token_marker}). */
    public static final String MARKER_TEXT = "<|box_end|>";

    private final String arch;
    private final int marker;
    private final int half;

    private KevScorer(String arch, int marker, int half) {
        this.arch = arch;
        this.marker = marker;
        this.half = half;
    }

    /**
     * The kev scorer of a model, its marker found in {@code tokenizer}'s vocabulary as one special token.
     *
     * @throws IllegalStateException when the vocabulary holds no single {@value #MARKER_TEXT} token
     */
    public static KevScorer of(ModelSpec spec, GGUFTokenizer tokenizer) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(tokenizer, "tokenizer");
        int[] toks = tokenizer.encode(MARKER_TEXT, false, true);
        if (toks.length != 1) {
            throw new IllegalStateException("decision model has no " + MARKER_TEXT + " token");
        }
        return new KevScorer(spec.arch().ggufName(), toks[0], spec.hparams().nEmbdOut() / 2);
    }

    /**
     * One score per {@value #MARKER_TEXT} marker of {@code tokens}, in prompt order: the dot of the last token's
     * query half with the marker's key half over the final hidden rows.
     *
     * @param hidden final hidden rows, one per token
     * @param tokens the input token ids
     * @throws IllegalArgumentException when the rows and the tokens disagree in length, or the input holds no
     *                                       marker
     */
    public float[] score(float[][] hidden, int[] tokens) {
        Objects.requireNonNull(hidden, "hidden");
        Objects.requireNonNull(tokens, "tokens");
        if (hidden.length != tokens.length) {
            throw new IllegalArgumentException(hidden.length + " hidden rows for " + tokens.length + " tokens");
        }
        int nMarkers = 0;
        for (int token : tokens) {
            nMarkers += token == marker ? 1 : 0;
        }
        if (nMarkers == 0) {
            throw new IllegalArgumentException(arch + " reads no " + MARKER_TEXT + " marker in " + tokens.length
                    + " tokens");
        }
        float[] q = hidden[tokens.length - 1];
        float scale = (float) Math.sqrt(half);
        float[] out = new float[nMarkers];
        int at = 0;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] != marker) {
                continue;
            }
            float[] k = hidden[i];
            float dot = 0.0f;
            for (int j = 0; j < half; j++) {
                dot += q[j] * k[half + j];
            }
            out[at++] = dot / scale;
        }
        return out;
    }

    /** The id of the {@value #MARKER_TEXT} marker token. */
    public int marker() {
        return marker;
    }

    @Override
    public String toString() {
        return "KevScorer{" + arch + ", marker " + marker + ", half " + half + "}";
    }
}
