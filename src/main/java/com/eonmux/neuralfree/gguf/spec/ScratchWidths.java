package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.hparams.ModelHparams;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Per-token widths (in elements) of the well-known forward-pass scratch slots of one model, the single sizing rule
 * of the CPU and GPU scratch ({@code CpuScratch.forModel}, {@code GpuScratch.forModel}). Immutable.
 *
 * <ul>
 *   <li>{@link #stream()}: the residual slots {@code X}, {@code BRANCH}, {@code NORMED}; the widest residual stream
 *       ({@code n_embd}, {@code hc * n_embd}, the output, posnet and convnext widths) or padded model input
 *       ({@code n_embd_inp}).</li>
 *   <li>{@link #projection()}: the projection slots {@code Q}, {@code K}, {@code V}, {@code ATTN_OUT}; at least the
 *       stream width and the largest input width {@code ne[0]} or output width {@code ne[1]} of any tensor except
 *       the lookup tables (token, position, token-type and per-layer embeddings, output head).</li>
 *   <li>{@link #gateUp()}: {@code FFN_GATE_UP}; twice the projection width, or the gate and up rows of every
 *       selected expert plus the shared expert.</li>
 *   <li>{@link #ffnOut()}: {@code FFN_OUT}; the projection width or the outputs of every selected expert.</li>
 *   <li>{@link #logits()}: {@code LOGITS}; the LM head's vocabulary rows (at least 1). The classifier and pooling
 *       heads work on their own host arrays and do not use this slot.</li>
 *   <li>{@link #input()}: {@code INPUT_EMBD}; the padded input embedding or the per-layer embedding inputs of all
 *       layers.</li>
 * </ul>
 */
public final class ScratchWidths {

    /** Ints per token of the {@code TOKEN_IDS} slot: the token id, then the token type. */
    public static final int TOKEN_ID_INTS = 2;
    /** Ints per token of the {@code POS} slot: one position per rope section. */
    public static final int POS_INTS = 4;

    private final long stream;
    private final long input;
    private final long projection;
    private final long gateUp;
    private final long ffnOut;
    private final long logits;

    private ScratchWidths(long stream, long input, long projection, long gateUp, long ffnOut, long logits) {
        this.stream = stream;
        this.input = input;
        this.projection = projection;
        this.gateUp = gateUp;
        this.ffnOut = ffnOut;
        this.logits = logits;
    }

    /**
     * The widths of {@code spec}, over its trunk, NextN and encoder layers.
     *
     * @throws ArithmeticException when a width overflows a {@code long}
     */
    public static ScratchWidths of(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        List<LayerSpec> layers = new ArrayList<>(spec.layers());
        layers.addAll(spec.nextnLayers());
        layers.addAll(spec.encoderLayers());
        ModelHparams hp = spec.hparams();
        EmbeddingSpec embd = spec.embedding();
        HeadSpec head = spec.head();

        long stream = max(hp.nEmbd(), hp.nEmbdOutImpl(), hp.posnetEmbd(), hp.convnextEmbd());
        for (LayerSpec l : layers) {
            stream = Math.max(stream, l.residual().streamWidth());
        }
        long input = stream;
        if (embd != null) {
            stream = max(stream, embd.nEmbd(), embd.nEmbdInp());
            input = Math.max(stream, Math.multiplyExact((long) embd.nEmbdPerLayer(), layers.size()));
        }
        long proj = Math.max(stream, projectionWidth(spec.tensors(), embd, head));
        long gateUp = Math.multiplyExact(2L, proj);
        long ffnOut = proj;
        for (LayerSpec l : layers) {
            MoeSpec m = l.moe();
            if (m == null) {
                continue;
            }
            long expertFf = max(m.nFfExp(), m.nFfChunk(), m.latentSize());
            long shared = m.sharedExpert() != null ? 2L * m.sharedExpert().nFf() : 0L;
            gateUp = Math.max(gateUp, Math.addExact(Math.multiplyExact(2L * m.nUsed(), expertFf), shared));
            ffnOut = Math.max(ffnOut, Math.multiplyExact((long) m.nUsed(), Math.max(stream, m.latentSize())));
        }
        return new ScratchWidths(stream, input, proj, gateUp, ffnOut, logitsWidth(head));
    }

    /** Elements of a slot of {@code width} per token for {@code ubatch} tokens (a width below 1 counts as 1). */
    public static long rows(int ubatch, long width) {
        return Math.multiplyExact((long) ubatch, Math.max(1L, width));
    }

    /** Residual-stream width. */
    public long stream() {
        return stream;
    }

    /** Embedding-stage input width. */
    public long input() {
        return input;
    }

    /** Widest projection. */
    public long projection() {
        return projection;
    }

    /** FFN gate-and-up width. */
    public long gateUp() {
        return gateUp;
    }

    /** FFN output width. */
    public long ffnOut() {
        return ffnOut;
    }

    /** Logits width. */
    public long logits() {
        return logits;
    }

    /** Width of a logits row: the LM head's vocabulary rows, at least 1 (a model without an LM head writes none). */
    private static long logitsWidth(HeadSpec head) {
        return Math.max(1L, head != null ? head.vocabRows() : 0L);
    }

    /**
     * Largest {@code ne[0]} of any tensor, and largest {@code ne[1]} of any tensor that is not a lookup table (whose
     * rows are vocabulary or position entries rather than per-token outputs).
     */
    private static long projectionWidth(TensorCatalog tensors, EmbeddingSpec embd, HeadSpec head) {
        Set<String> lookups = new HashSet<>();
        long vocab = 0;
        if (embd != null) {
            addName(lookups, embd.tokenEmbd());
            addName(lookups, embd.posEmbd());
            addName(lookups, embd.perLayerTokenEmbd());
            if (embd.tokenEmbd() != null && embd.tokenEmbd().rank() >= 2) {
                vocab = embd.tokenEmbd().ne(1);
            }
        }
        addName(lookups, tensors.get(TensorNames.weight(TensorNames.TOKEN_TYPES)));
        if (head != null) {
            addName(lookups, head.output());
            vocab = Math.max(vocab, head.vocabRows());
        }
        long w = 0;
        for (String name : tensors.names()) {
            TensorRef t = tensors.get(name);
            if (t == null) {
                continue;
            }
            w = Math.max(w, t.ne(0));
            if (t.rank() >= 2 && !lookups.contains(t.name()) && !(vocab > 0 && t.ne(1) == vocab)) {
                w = Math.max(w, t.ne(1));
            }
        }
        return w;
    }

    private static void addName(Set<String> names, TensorRef t) {
        if (t != null) {
            names.add(t.name());
        }
    }

    private static long max(long a, long... rest) {
        long m = a;
        for (long v : rest) {
            m = Math.max(m, v);
        }
        return m;
    }

    @Override
    public String toString() {
        return "ScratchWidths{stream=" + stream + ", input=" + input + ", projection=" + projection + ", gateUp="
                + gateUp + ", ffnOut=" + ffnOut + ", logits=" + logits + '}';
    }
}
