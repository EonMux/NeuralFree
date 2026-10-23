package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Arrays;
import java.util.Objects;

/**
 * Output contracts of the TTS backbones that run on the ordinary decoder trunk.
 *
 * <ul>
 *   <li><b>qwen3tts</b> (llama.cpp src/models/qwen3tts.cpp reuses qwen3vl.cpp): the LM head is the codec head with
 *       {@code n_vocab_out} rows while {@code token_embd} has {@code n_vocab_in} rows; the logits row is
 *       {@code concat(-inf[n_vocab_in - n_vocab_out], head)}, the padding first (qwen3vl.cpp:175-191, the converter's
 *       {@code [text_vocab | codec_vocab]} layout). {@link LogitLayout} holds that layout, built from the model's
 *       {@link HeadSpec} LM head and embedding table, and {@link #qwen3tts} wraps a {@link ForwardBackend} whose
 *       logits rows are head-wide so that every row it returns has the llama.cpp layout.</li>
 *   <li><b>pockettts</b> (llama.cpp src/models/pockettts.cpp): a backbone without an LM head whose product is
 *       {@code t_embd}, the hidden state after the biased output LayerNorm (pockettts.cpp:132-138). {@link
 *       PocketHidden} observes the residual stream entering the output norm through a {@link HiddenTap} during
 *       ordinary KV-cached prefill and decode, and applies the output LayerNorm on the host.</li>
 * </ul>
 */
public final class TtsBackbone {

    /** IEEE-754 bits of {@code -inf}, the fill pattern of the padded logits. */
    public static final int NEG_INF_BITS = Float.floatToRawIntBits(Float.NEGATIVE_INFINITY);

    private TtsBackbone() {
    }

    // ================================================================== qwen3tts

    /**
     * qwen3tts logits layout: {@code vocab} floats per row, rows {@code [0, pad)} at {@code -inf} and rows
     * {@code [pad, vocab)} the LM head output. Immutable.
     */
    public static final class LogitLayout {
        private final int vocab;
        private final int headRows;

        private LogitLayout(int vocab, int headRows) {
            this.vocab = vocab;
            this.headRows = headRows;
        }

        /**
         * Layout of a resolved qwen3tts model: {@code vocab} is the {@code token_embd} row count
         * ({@code n_vocab_in}), {@code headRows} the {@link HeadSpec#output()} row count ({@code n_vocab_out}).
         *
         * @throws IllegalArgumentException when the model is another architecture
         * @throws GgufFormatException      when the head has no output norm (qwen3vl.cpp:179 asserts it), no LM head,
         *                                  or more rows than the embedding table (qwen3vl.cpp:190 aborts)
         */
        public static LogitLayout of(ModelSpec spec) {
            Objects.requireNonNull(spec, "spec");
            if (spec.arch() != Arch.QWEN3TTS) {
                throw new IllegalArgumentException("the padded codec-head layout belongs to "
                        + Arch.QWEN3TTS.ggufName() + ", not " + spec.arch().ggufName());
            }
            HeadSpec head = spec.head();
            TensorRef output = head.output();
            if (output == null) {
                throw new GgufFormatException("qwen3tts: the model has no LM head");
            }
            int vocab = rowCount(spec.embedding().tokenEmbd(), "token embedding");
            int rows = rowCount(output, "LM head");
            if (vocab > rows && head.outputNorm() == null) {
                throw new GgufFormatException("qwen3tts: a padded codec head needs output_norm (qwen3vl.cpp:179)");
            }
            return of(vocab, rows);
        }

        /**
         * Layout of {@code vocab}-wide rows from a {@code headRows}-row head.
         *
         * @throws GgufFormatException when a count is not positive or the head is wider than the vocabulary
         *                             (qwen3vl.cpp:190)
         */
        public static LogitLayout of(int vocab, int headRows) {
            if (vocab <= 0 || headRows <= 0) {
                throw new GgufFormatException("qwen3tts: vocabulary " + vocab + " and head rows " + headRows
                        + " must be positive");
            }
            if (headRows > vocab) {
                throw new GgufFormatException("qwen3tts: the LM head has " + headRows + " rows, more than the "
                        + vocab + "-token embedding table (qwen3vl.cpp:190)");
            }
            return new LogitLayout(vocab, headRows);
        }

        private static int rowCount(TensorRef r, String what) {
            if (r == null) {
                throw new GgufFormatException("qwen3tts: the model has no " + what);
            }
            long rows = r.ne(1);
            if (rows <= 0 || rows > Integer.MAX_VALUE) {
                throw new GgufFormatException("qwen3tts: " + what + " " + r.name() + " has " + rows + " rows");
            }
            return (int) rows;
        }

        /** Logits per row ({@code n_vocab_in}). */
        public int vocab() { return vocab; }
        /** LM head rows ({@code n_vocab_out}). */
        public int headRows() { return headRows; }
        /** Leading {@code -inf} entries per row. */
        public int pad() { return vocab - headRows; }
        /** True when rows are padded. */
        public boolean padded() { return vocab > headRows; }

        /**
         * Writes one padded row: {@code dst[dstOff + i] = -inf} for {@code i < pad}, then the {@code headRows}
         * values of {@code head[headOff ..]}.
         *
         * @throws IllegalArgumentException when a range lies outside its array or the ranges overlap
         */
        public void pad(float[] head, int headOff, float[] dst, int dstOff) {
            Objects.requireNonNull(head, "head");
            Objects.requireNonNull(dst, "dst");
            if (headOff < 0 || dstOff < 0 || (long) headOff + headRows > head.length
                    || (long) dstOff + vocab > dst.length) {
                throw new IllegalArgumentException("padded logits row out of range: head [" + headOff + ", +"
                        + headRows + ") of " + head.length + ", row [" + dstOff + ", +" + vocab + ") of "
                        + dst.length);
            }
            if (head == dst && headOff < dstOff + vocab && dstOff < headOff + headRows) {
                throw new IllegalArgumentException("padded logits row overlaps the head values");
            }
            Arrays.fill(dst, dstOff, dstOff + pad(), Float.NEGATIVE_INFINITY);
            System.arraycopy(head, headOff, dst, dstOff + pad(), headRows);
        }

        /**
         * One row in this layout from a backend row: a head-wide row is padded into a new array; a row already
         * {@code vocab} wide is returned as is when its padding entries are all {@code -inf}.
         *
         * @throws IllegalStateException when the row has another width, or is vocabulary-wide with a finite entry
         *                               where the layout has padding
         */
        public float[] toVocabRow(float[] row) {
            Objects.requireNonNull(row, "row");
            if (!padded() || row.length == vocab && hasPadding(row)) {
                if (row.length != vocab) {
                    throw widthMismatch(row);
                }
                return row;
            }
            if (row.length != headRows) {
                throw widthMismatch(row);
            }
            float[] out = new float[vocab];
            pad(row, 0, out, 0);
            return out;
        }

        private boolean hasPadding(float[] row) {
            for (int i = 0; i < pad(); i++) {
                if (row[i] != Float.NEGATIVE_INFINITY) {
                    return false;
                }
            }
            return true;
        }

        private IllegalStateException widthMismatch(float[] row) {
            return new IllegalStateException("qwen3tts logits row of " + row.length + " floats matches neither the "
                    + headRows + "-row codec head nor the padded " + vocab + "-token layout");
        }

        /**
         * Greedy choice over the padded row of a head-wide row, without building it: the index
         * {@link ForwardBackend#argmax} returns for {@code concat(-inf[pad], head)}. A head with no value above
         * {@code -inf} selects 0, the first padding entry.
         *
         * @throws IllegalArgumentException when the row is not head-wide
         */
        public int argmaxOfHead(float[] head) {
            Objects.requireNonNull(head, "head");
            if (head.length != headRows) {
                throw new IllegalArgumentException("head row of " + head.length + " floats, expected " + headRows);
            }
            if (!padded()) {
                return ForwardBackend.argmax(head);
            }
            int best = 0;
            float bestValue = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < headRows; i++) {
                if (head[i] > bestValue) {
                    bestValue = head[i];
                    best = pad() + i;
                }
            }
            return best;
        }

        /**
         * Device form of {@link #pad}: {@code nRows} head rows at {@code head[headOff + r*headRows]} become padded rows
         * at {@code dst[dstOff + r*vocab]} ({@link GpuOps#fill} of {@code -inf}, then {@link GpuOps#copy2d}). The
         * views must not overlap.
         */
        public void padOnDevice(GpuOps ops, GpuBuffer head, int headOff, GpuBuffer dst, int dstOff, int nRows) {
            Objects.requireNonNull(ops, "ops");
            if (nRows <= 0) {
                throw new IllegalArgumentException("row count must be positive, got " + nRows);
            }
            if (padded()) {
                ops.fill(dst, (long) dstOff * Float.BYTES, (long) nRows * vocab * Float.BYTES, NEG_INF_BITS);
            }
            ops.copy2d(head, headOff, headRows, dst, dstOff + pad(), vocab, nRows, headRows);
        }

        @Override
        public String toString() {
            return "LogitLayout{vocab=" + vocab + ", head=" + headRows + ", pad=" + pad() + '}';
        }
    }

    /**
     * A qwen3tts backend whose logits rows have the llama.cpp layout ({@link LogitLayout}); rows of the wrapped
     * backend may be head-wide or already padded.
     *
     * @throws IllegalArgumentException when the model is not qwen3tts
     * @throws GgufFormatException      as {@link LogitLayout#of(ModelSpec)}
     */
    public static ForwardBackend qwen3tts(ForwardBackend delegate, ModelSpec spec) {
        return new PaddedLogitsBackend(delegate, LogitLayout.of(spec));
    }

    /** {@link ForwardBackend} decorator that pads every logits row; everything else is delegated. */
    static final class PaddedLogitsBackend implements ForwardBackend {
        private final ForwardBackend delegate;
        private final LogitLayout layout;

        PaddedLogitsBackend(ForwardBackend delegate, LogitLayout layout) {
            this.delegate = Objects.requireNonNull(delegate, "delegate");
            this.layout = Objects.requireNonNull(layout, "layout");
        }

        LogitLayout layout() {
            return layout;
        }

        @Override
        public float[] prefill(int[] tokens, int pos0) {
            return layout.toVocabRow(delegate.prefill(tokens, pos0));
        }

        @Override
        public float[] decode(int token, int pos) {
            return layout.toVocabRow(delegate.decode(token, pos));
        }

        /** Selects on the host over the padded row, so ties and degenerate rows resolve as llama.cpp's greedy. */
        @Override
        public int decodeGreedyOnDevice(int token, int pos) {
            float[] row = delegate.decode(token, pos);
            return row.length == layout.headRows() ? layout.argmaxOfHead(row)
                    : ForwardBackend.argmax(layout.toVocabRow(row));
        }

        @Override
        public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            return padAll(delegate.forwardRows(tokens, pos0, outputRows));
        }

        @Override
        public float[][] forwardAll(int[] tokens, int[] outputRows) {
            return padAll(delegate.forwardAll(tokens, outputRows));
        }

        private float[][] padAll(float[][] rows) {
            float[][] out = new float[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                out[i] = layout.toVocabRow(rows[i]);
            }
            return out;
        }

        @Override
        public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            return delegate.hiddenRows(tokens, tokenTypes);
        }

        @Override
        public void reset() {
            delegate.reset();
        }

        @Override
        public boolean canRollback(int nTokens) {
            return delegate.canRollback(nTokens);
        }

        @Override
        public void rollback(int nTokens) {
            delegate.rollback(nTokens);
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            delegate.shiftContext(nKeep, nDiscard);
        }

        @Override
        public ModelMemory memory() {
            return delegate.memory();
        }

        @Override
        public void setHiddenTap(HiddenTap tap) {
            delegate.setHiddenTap(tap);
        }

        @Override
        public void close() {
            delegate.close();
        }

        @Override
        public String toString() {
            return "qwen3tts(" + delegate + ", " + layout + ')';
        }
    }

    // ================================================================== pockettts

    /**
     * pockettts backbone hidden states ({@code t_embd}) over a KV-cached {@link ForwardBackend}: installs a
     * {@link HiddenTap} on the residual stream entering the output norm (tap {@code n_layer}) and applies the output
     * LayerNorm with bias ({@code f_norm_eps}, pockettts.cpp:7, 132-135) on the host. The logits the backend
     * computes from the tied embedding table are not used (pockettts.cpp:23-24). Not thread-safe; closing removes
     * the tap.
     */
    public static final class PocketHidden implements AutoCloseable {
        private final ForwardBackend backend;
        private final int width;
        private final float eps;
        private final float[] normW;
        private final float[] normB;
        private final Capture capture;
        private boolean closed;

        private PocketHidden(ForwardBackend backend, int nLayer, int width, float eps, float[] normW, float[] normB) {
            this.backend = backend;
            this.width = width;
            this.eps = eps;
            this.normW = normW.clone();
            this.normB = normB.clone();
            this.capture = new Capture(nLayer, width);
            backend.setHiddenTap(capture);
        }

        /**
         * Attaches to a pockettts backend with the output norm read from CPU weights.
         *
         * @throws IllegalArgumentException as {@link #attach(ForwardBackend, ModelSpec, float[], float[])}
         */
        public static PocketHidden attach(ForwardBackend backend, ModelSpec spec, CpuWeights weights) {
            Objects.requireNonNull(spec, "spec");
            Objects.requireNonNull(weights, "weights");
            HeadSpec head = spec.head();
            return attach(backend, spec, weights.floats(head.outputNorm()), weights.floats(head.outputNormBias()));
        }

        /**
         * Attaches to a pockettts backend.
         *
         * @param normW {@code output_norm.weight} ({@code n_embd} floats, copied)
         * @param normB {@code output_norm.bias} ({@code n_embd} floats, copied)
         * @throws IllegalArgumentException when the model is not pockettts or a norm vector has the wrong length
         * @throws GgufFormatException      when the head dims break pockettts.cpp:51-52 ({@code n_embd_head_v ==
         *                                  n_embd_head_k == n_rot}) or the model has a hyper-connection stream
         */
        public static PocketHidden attach(ForwardBackend backend, ModelSpec spec, float[] normW, float[] normB) {
            Objects.requireNonNull(backend, "backend");
            Objects.requireNonNull(spec, "spec");
            if (spec.arch() != Arch.POCKETTTS) {
                throw new IllegalArgumentException("backbone hidden states belong to " + Arch.POCKETTTS.ggufName()
                        + ", not " + spec.arch().ggufName());
            }
            ModelHparams hp = spec.hparams();
            if (hp.headVFull() != hp.headKFull() || hp.nRotFull() != hp.headKFull()) {
                throw new GgufFormatException("pockettts: head dims k=" + hp.headKFull() + " v=" + hp.headVFull()
                        + " rot=" + hp.nRotFull() + " must be equal (pockettts.cpp:51-52)");
            }
            int width = hp.nEmbd();
            if (spec.maxStreamWidth() != width) {
                throw new GgufFormatException("pockettts: residual stream width " + spec.maxStreamWidth()
                        + " differs from n_embd " + width);
            }
            checkVector("output_norm.weight", normW, width);
            checkVector("output_norm.bias", normB, width);
            return new PocketHidden(backend, spec.nLayer(), width, hp.normEps(), normW, normB);
        }

        private static void checkVector(String what, float[] v, int width) {
            if (v == null || v.length != width) {
                throw new IllegalArgumentException("pockettts " + what + " must hold " + width + " floats, got "
                        + (v == null ? "none" : v.length));
            }
        }

        /** {@code n_embd}. */
        public int width() {
            return width;
        }

        /**
         * Evaluates {@code tokens} at {@code pos0 ..} into the KV cache and returns {@code t_embd} of the last token.
         *
         * @return a fresh {@code n_embd} row
         */
        public float[] prefill(int[] tokens, int pos0) {
            float[][] rows = prefillRows(tokens, pos0, true);
            return rows[0];
        }

        /**
         * Evaluates {@code tokens} at {@code pos0 ..} into the KV cache and returns {@code t_embd} of every token.
         *
         * @return fresh {@code [tokens.length][n_embd]} rows
         */
        public float[][] prefillAll(int[] tokens, int pos0) {
            return prefillRows(tokens, pos0, false);
        }

        /** Evaluates one token at {@code pos} into the KV cache and returns its {@code t_embd} (a fresh row). */
        public float[] decode(int token, int pos) {
            ensureOpen();
            capture.clear();
            backend.decode(token, pos);
            return normalizedRows(1, true)[0];
        }

        /**
         * {@code t_embd} of a whole sequence from position 0 without touching the KV cache
         * ({@link ForwardBackend#hiddenRows}, which applies the output norm on the device).
         */
        public float[][] hiddenRows(int[] tokens) {
            ensureOpen();
            return backend.hiddenRows(tokens, null);
        }

        private float[][] prefillRows(int[] tokens, int pos0, boolean lastOnly) {
            ensureOpen();
            Objects.requireNonNull(tokens, "tokens");
            if (tokens.length == 0) {
                throw new IllegalArgumentException("prefill needs at least one token");
            }
            capture.clear();
            backend.prefill(tokens, pos0);
            return normalizedRows(tokens.length, lastOnly);
        }

        /**
         * The last {@code nTokens} captured rows, or only the last row, through the output LayerNorm. The last row
         * alone needs one captured row, so a backend that keeps only the output rows at the last layer (llama.cpp
         * {@code inp_out_ids}, pockettts.cpp:98-101) still serves it.
         */
        private float[][] normalizedRows(int nTokens, boolean lastOnly) {
            int expected = lastOnly ? 1 : nTokens;
            if (capture.rows() < expected) {
                throw new IllegalStateException("the backend reported " + capture.rows() + " hidden rows at tap "
                        + capture.layer + " for " + nTokens + " tokens; it must call the hidden tap for every"
                        + " forward call");
            }
            int first = lastOnly ? capture.rows() - 1 : capture.rows() - expected;
            int n = capture.rows() - first;
            float[][] out = new float[n][width];
            for (int i = 0; i < n; i++) {
                NormMath.rowNorm(NormKind.LAYER, eps, capture.data(), (first + i) * width, out[i], 0, width,
                        normW, 0, normB, 0);
            }
            return out;
        }

        /** Removes the tap from the backend; idempotent. The backend stays open. */
        @Override
        public void close() {
            if (!closed) {
                closed = true;
                backend.setHiddenTap(HiddenTap.NONE);
            }
        }

        private void ensureOpen() {
            if (closed) {
                throw new IllegalStateException("pockettts hidden-state reader is closed");
            }
        }

        @Override
        public String toString() {
            return "PocketHidden{width=" + width + ", tap=" + capture.layer + '}';
        }
    }

    /** Collects the rows of one tap point across the forward calls of one prefill or decode. */
    static final class Capture implements HiddenTap {
        private final int layer;
        private final int width;
        private float[] data = new float[0];
        private int rows;

        Capture(int layer, int width) {
            this.layer = layer;
            this.width = width;
        }

        void clear() {
            rows = 0;
        }

        int rows() {
            return rows;
        }

        float[] data() {
            return data;
        }

        @Override
        public boolean wants(int l) {
            return l == layer;
        }

        @Override
        public void onHidden(int l, float[] src, int nRows, int w) {
            if (l != layer) {
                return;
            }
            if (w != width) {
                throw new IllegalStateException("hidden tap " + l + " delivered rows of " + w + " floats, expected "
                        + width);
            }
            long need = (long) (rows + nRows) * width;
            if (need > Integer.MAX_VALUE) {
                throw new IllegalStateException("captured hidden rows exceed one float array");
            }
            if (need > data.length) {
                data = Arrays.copyOf(data, (int) Math.max(need, Math.min(Integer.MAX_VALUE, 2L * data.length)));
            }
            System.arraycopy(src, 0, data, rows * width, nRows * width);
            rows += nRows;
        }
    }
}
