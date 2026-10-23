package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

/**
 * kimi-k3 cross-layer attention residuals (src/models/kimi-k3.cpp:188-235, 270-338) on the CPU: the static
 * {@link #scores} and {@link #mix} are the references of {@code LLM_ResStack_Score} and {@code LLM_ResStack_Mix};
 * an instance is the per-forward-step checkpoint stack ({@code stack[(c*n_tokens + t)*n_embd + i]}, the layout of
 * {@link HcKernels}), a mutable runtime object owned by one layer executor and never shared between threads.
 *
 * <p>Per step: {@link #begin} with the step's token count; for each layer {@code il} the attention and FFN inputs
 * are {@link #mixInto mixed}, and on checkpoint layers ({@link HcKernels#isCheckpointLayer}) the raw layer input
 * is {@link #push pushed} after the attention mix (kimi-k3.cpp:284-291). With no checkpoint banked the mix is the
 * identity (kimi-k3.cpp:200-202).
 */
public final class CpuAttnResidualStack {

    private static final int TOKEN_GRAIN = 1;

    private final int nEmbd;
    private final int maxCheckpoints;
    private float[] stack = new float[0];
    private float[] probs = new float[0];
    private int nTokens;
    private int count;

    /**
     * @param nEmbd          row width
     * @param maxCheckpoints checkpoints one step can bank ({@link HcKernels#checkpointCount})
     * @throws IllegalArgumentException when either is out of range
     */
    public CpuAttnResidualStack(int nEmbd, int maxCheckpoints) {
        HcKernels.validateResStack(nEmbd, maxCheckpoints);
        this.nEmbd = nEmbd;
        this.maxCheckpoints = maxCheckpoints;
    }

    /** Empties the stack for a step of {@code nTokens} rows, growing the storage when needed. */
    public void begin(int nTokens) {
        if (nTokens <= 0) {
            throw new IllegalArgumentException("attention-residual step needs at least one token, got " + nTokens);
        }
        long floats = HcKernels.stackFloats(maxCheckpoints, nTokens, nEmbd);
        if (floats > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("attention-residual stack of " + maxCheckpoints + " x " + nTokens + " x "
                    + nEmbd + " floats exceeds a Java array");
        }
        if (stack.length < floats) {
            stack = new float[(int) floats];
        }
        int probFloats = nTokens * (maxCheckpoints + 1);
        if (probs.length < probFloats) {
            probs = new float[probFloats];
        }
        this.nTokens = nTokens;
        this.count = 0;
    }

    /** Checkpoints banked in this step. */
    public int checkpoints() {
        return count;
    }

    /**
     * Banks the raw rows {@code x[xOff + t*nEmbd + i]}, {@code t < nTokens}, as the next checkpoint.
     *
     * @throws IllegalStateException when the step was not begun or the stack is full
     */
    public void push(float[] x, int xOff) {
        if (nTokens == 0) {
            throw new IllegalStateException("attention-residual stack used before begin()");
        }
        if (count >= maxCheckpoints) {
            throw new IllegalStateException("attention-residual stack is full (" + maxCheckpoints + " checkpoints)");
        }
        CpuHyperConnections.requireLength("checkpoint", x, (long) xOff + (long) nTokens * nEmbd);
        System.arraycopy(x, xOff, stack, (int) HcKernels.checkpointOffset(count, nTokens, nEmbd), nTokens * nEmbd);
        count++;
    }

    /**
     * Mixes the banked checkpoints with the current rows {@code cur[curOff + t*nEmbd + i]} using the
     * {@code nEmbd} score weight {@code scoreW}, writing {@code out[outOff + t*nEmbd + i]}; {@code out} may be
     * {@code cur} at the same offset. Without checkpoints the rows are copied unchanged.
     *
     * @throws IllegalStateException when the step was not begun
     */
    public void mixInto(float[] cur, int curOff, float[] scoreW, float eps, float[] out, int outOff) {
        if (nTokens == 0) {
            throw new IllegalStateException("attention-residual stack used before begin()");
        }
        if (count == 0) {
            if (out != cur || outOff != curOff) {
                CpuHyperConnections.requireLength("cur", cur, (long) curOff + (long) nTokens * nEmbd);
                CpuHyperConnections.requireLength("out", out, (long) outOff + (long) nTokens * nEmbd);
                System.arraycopy(cur, curOff, out, outOff, nTokens * nEmbd);
            }
            return;
        }
        final int n = count;
        final float[] st = stack;
        final float[] pr = probs;
        CpuParallel.parallelFor(nTokens, TOKEN_GRAIN, (from, to) -> {
            scores(nTokens, nEmbd, n, eps, st, 0, cur, curOff, scoreW, pr, 0, from, to);
            mix(nTokens, nEmbd, n, st, 0, cur, curOff, pr, 0, out, outOff, from, to);
        });
    }

    /**
     * {@code LLM_ResStack_Score} for tokens {@code [t0, t1)}: for checkpoint {@code c < nCkpt} (row
     * {@code stack[stackOff + (c*nTokens + t)*nEmbd ...]}) and the current row ({@code c == nCkpt},
     * {@code cur[curOff + t*nEmbd ...]}), {@code score_c = sum_i (x_i * r) * w_i} with
     * {@code r = 1/sqrt(sum_i x_i^2 / nEmbd + eps)} (ggml_rms_norm, mul, sum_rows; sums in double as ggml-cpu), then
     * {@code probs[probsOff + t*(nCkpt+1) + c] = softmax_c(score)} (ggml_soft_max).
     *
     * @throws IllegalArgumentException when the shape is invalid or an array is too short
     */
    public static void scores(int nTokens, int nEmbd, int nCkpt, float eps, float[] stack, int stackOff, float[] cur,
                              int curOff, float[] scoreW, float[] probs, int probsOff, int t0, int t1) {
        HcKernels.validateResStack(nEmbd, nCkpt);
        requireShapes(nTokens, nEmbd, nCkpt, stack, stackOff, cur, curOff, probs, probsOff, t0, t1);
        CpuHyperConnections.requireLength("score weight", scoreW, nEmbd);
        for (int t = t0; t < t1; t++) {
            final int p = probsOff + t * (nCkpt + 1);
            for (int c = 0; c <= nCkpt; c++) {
                final float[] src = c < nCkpt ? stack : cur;
                final int x0 = c < nCkpt ? stackOff + (c * nTokens + t) * nEmbd : curOff + t * nEmbd;
                double ss = 0.0;
                for (int i = 0; i < nEmbd; i++) {
                    final float v = src[x0 + i];
                    ss += (double) (v * v);
                }
                final float mean = (float) (ss / nEmbd);
                final float r = 1.0f / (float) Math.sqrt(mean + eps);
                double dw = 0.0;
                for (int i = 0; i < nEmbd; i++) {
                    dw += (double) ((src[x0 + i] * r) * scoreW[i]);
                }
                probs[p + c] = (float) dw;
            }
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c <= nCkpt; c++) {
                max = Math.max(max, probs[p + c]);
            }
            double sum = 0.0;
            for (int c = 0; c <= nCkpt; c++) {
                final float e = (float) Math.exp(probs[p + c] - max);
                probs[p + c] = e;
                sum += e;
            }
            final float inv = (float) (1.0 / sum);
            for (int c = 0; c <= nCkpt; c++) {
                probs[p + c] *= inv;
            }
        }
    }

    /**
     * {@code LLM_ResStack_Mix} for tokens {@code [t0, t1)}:
     * {@code out[outOff + t*nEmbd + i] = sum_c stack[stackOff + (c*nTokens + t)*nEmbd + i] * p_c + cur[curOff + t*nEmbd + i] * p_nCkpt}
     * with {@code p = probs[probsOff + t*(nCkpt+1) ...]}; the checkpoint sum runs in increasing {@code c} before the
     * current-row term is added (ggml_dsv4_hc_pre, then ggml_add). {@code out} may be {@code cur} at the same offset.
     *
     * @throws IllegalArgumentException when the shape is invalid or an array is too short
     */
    public static void mix(int nTokens, int nEmbd, int nCkpt, float[] stack, int stackOff, float[] cur, int curOff,
                           float[] probs, int probsOff, float[] out, int outOff, int t0, int t1) {
        HcKernels.validateResStack(nEmbd, nCkpt);
        requireShapes(nTokens, nEmbd, nCkpt, stack, stackOff, cur, curOff, probs, probsOff, t0, t1);
        CpuHyperConnections.requireLength("out", out, (long) outOff + (long) t1 * nEmbd);
        for (int t = t0; t < t1; t++) {
            final int p = probsOff + t * (nCkpt + 1);
            final float pCur = probs[p + nCkpt];
            for (int i = 0; i < nEmbd; i++) {
                float sum = 0.0f;
                for (int c = 0; c < nCkpt; c++) {
                    sum += stack[stackOff + (c * nTokens + t) * nEmbd + i] * probs[p + c];
                }
                out[outOff + t * nEmbd + i] = sum + cur[curOff + t * nEmbd + i] * pCur;
            }
        }
    }

    private static void requireShapes(int nTokens, int nEmbd, int nCkpt, float[] stack, int stackOff, float[] cur,
                                      int curOff, float[] probs, int probsOff, int t0, int t1) {
        CpuHyperConnections.requireTokenRange(t0, t1);
        if (t1 > nTokens) {
            throw new IllegalArgumentException("token range end " + t1 + " exceeds the step's " + nTokens + " tokens");
        }
        if (nCkpt > 0) {
            CpuHyperConnections.requireLength("stack", stack,
                    (long) stackOff + HcKernels.stackFloats(nCkpt, nTokens, nEmbd));
        }
        CpuHyperConnections.requireLength("cur", cur, (long) curOff + (long) t1 * nEmbd);
        CpuHyperConnections.requireLength("probs", probs, (long) probsOff + (long) t1 * (nCkpt + 1));
    }
}
