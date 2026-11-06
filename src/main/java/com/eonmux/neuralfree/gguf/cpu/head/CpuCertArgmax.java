package com.eonmux.neuralfree.gguf.cpu.head;

import com.eonmux.neuralfree.gguf.cpu.quant.CertifiedArgmaxScreen;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * One certified exact-argmax readout over an LM head weight: the INT8 screen of
 * {@link CertifiedArgmaxScreen}, the candidate selection under its error bound, the exact rescore of the
 * candidates through the head's own matvec kernel ({@link CpuMatVec} over a compact weight holding the
 * gathered candidate rows), and the full exact head as the fallback for rows whose candidate count
 * exceeds the cap. The result of {@link #argmax} is the argmax of the exact logits: the certified set
 * contains every row whose exact logit can reach the maximum, and the rescore runs the same kernel on
 * the same row bytes the full head would.
 *
 * <p>Not thread-safe; one forward pass at a time, like the head that owns it. Builds once (one full
 * decode of the weight) and allocates nothing per call except the returned ids.</p>
 */
final class CpuCertArgmax {

    private final CpuMatVec matVec;
    private final CpuTensorRef output;
    private final CertifiedArgmaxScreen screen;
    private final int kmax;
    private final int vocab;
    private final int cols;
    private final long rowBytes;
    /** Screen logits of the row being decided, {@link #vocab} floats. */
    private final float[] screened;
    /** Exact logits of one full fallback row, {@link #vocab} floats. */
    private final float[] fallback;
    /** Candidate token ids of the row being decided, ascending. */
    private final int[] candidates;
    /** Exact logits of the compact rescore, {@link #kmax} floats. */
    private final float[] rescored;
    /** Gathered candidate row bytes, {@link #kmax} rows of {@link #rowBytes}. */
    private final ByteBuffer compact;

    private CpuCertArgmax(CpuMatVec matVec, CpuTensorRef output, CertifiedArgmaxScreen screen, int kmax) {
        this.matVec = matVec;
        this.output = output;
        this.screen = screen;
        this.kmax = kmax;
        this.vocab = screen.rows();
        this.cols = screen.cols();
        this.rowBytes = output.rowBytes();
        this.screened = new float[vocab];
        this.fallback = new float[vocab];
        this.candidates = new int[kmax];
        this.rescored = new float[kmax];
        this.compact = ByteBuffer.allocate(Math.multiplyExact(kmax, Math.toIntExact(rowBytes)));
    }

    /**
     * Builds the readout over {@code output}.
     *
     * @throws IllegalArgumentException for a non-positive cap
     */
    static CpuCertArgmax of(CpuMatVec matVec, CpuTensorRef output, int kmax) {
        Objects.requireNonNull(matVec, "matVec");
        Objects.requireNonNull(output, "output");
        if (kmax < 1) {
            throw new IllegalArgumentException("candidate cap must be positive, got " + kmax);
        }
        return new CpuCertArgmax(matVec, output, CertifiedArgmaxScreen.of(output), kmax);
    }

    /** The candidate cap. */
    int kmax() {
        return kmax;
    }

    /**
     * Greedy ids (largest exact logit, lowest token on ties) of {@code nRows} packed hidden rows of
     * {@code cols} floats at {@code hidden[hiddenOff ...]}.
     */
    int[] argmax(float[] hidden, int hiddenOff, int nRows) {
        int[] ids = new int[nRows];
        for (int r = 0; r < nRows; r++) {
            ids[r] = argmaxRow(hidden, hiddenOff + r * cols);
        }
        return ids;
    }

    /** The certified id of one hidden row at {@code hOff}. */
    private int argmaxRow(float[] h, int hOff) {
        float hiddenNormSq = 0.0f;
        for (int i = 0; i < cols; i++) {
            float v = h[hOff + i];
            hiddenNormSq += v * v;
        }
        float hiddenNorm = (float) Math.sqrt(hiddenNormSq);

        screen.screenLogits(h, hOff, screened, 0);
        float lower = Float.NEGATIVE_INFINITY;
        for (int v = 0; v < vocab; v++) {
            float t = screened[v] - screen.bound(v) * hiddenNorm;
            if (t > lower) {
                lower = t;
            }
        }
        float threshold = lower - screen.tie(lower, hiddenNorm);

        int count = 0;
        boolean overflow = false;
        for (int v = 0; v < vocab; v++) {
            if (screened[v] + screen.bound(v) * hiddenNorm >= threshold) {
                if (count == kmax) {
                    overflow = true;
                    break;
                }
                candidates[count++] = v;
            }
        }
        if (overflow || count == 0) {
            return fallbackArgmax(h, hOff);
        }
        return rescoredArgmax(count, h, hOff);
    }

    /** The exact argmax of one row: the full head product, the fallback the cap forces. */
    private int fallbackArgmax(float[] h, int hOff) {
        matVec.matvecRows(output, 0, vocab, h, hOff, fallback, 0);
        int best = 0;
        float bestValue = fallback[0];
        for (int v = 1; v < vocab; v++) {
            if (fallback[v] > bestValue) {
                bestValue = fallback[v];
                best = v;
            }
        }
        return best;
    }

    /**
     * The exact rescore of the {@code count} candidates: their rows gathered into a compact weight and
     * run through the head's own matvec, one token, then the first maximum (candidates ascend, so the
     * lowest token wins ties).
     */
    private int rescoredArgmax(int count, float[] h, int hOff) {
        compact.clear();
        for (int j = 0; j < count; j++) {
            int v = candidates[j];
            ByteBuffer shard = output.rowBuffer(v).duplicate();
            shard.position(output.rowOffset(v));
            shard.limit(output.rowOffset(v) + (int) rowBytes);
            compact.put(shard);
        }
        compact.flip();
        CpuTensorRef gathered = CpuTensorRef.of(output.type(), compact, count, cols);
        matVec.matmul(gathered, h, hOff, rescored, 0, 1);
        int best = 0;
        float bestValue = rescored[0];
        for (int j = 1; j < count; j++) {
            if (rescored[j] > bestValue) {
                bestValue = rescored[j];
                best = j;
            }
        }
        return candidates[best];
    }
}
