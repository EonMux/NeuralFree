package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;

/**
 * The float32 kernels of a finetuning step, each with its backward. Every forward formula is the one the CPU
 * inference engine runs ({@link ActMath}, {@code NormMath}, {@link RopeFreqTable}), so a training forward pass
 * produces the logits the engine produces for the same weights; the backward passes are the analytic gradients of
 * those same formulas.
 *
 * <p>Matrices are row-major in the GGUF storage order: a weight of {@code rows} outputs and {@code cols} inputs is
 * {@code w[row * cols + col]} and a vector input is {@code x[token * cols + col]}, so {@code y = x W^T + b}. Gradient
 * methods accumulate into their destination arrays. Scratch arrays are caller-owned. The row-wise work of the
 * dense products is threaded with {@link CpuParallel}; the rest is sequential.
 */
final class TrainMath {

    private static final int GRAIN = 1 << 12;

    private TrainMath() {
    }

    // ------------------------------------------------------------------ dense products

    /** {@code y[t][r] = dot(w[r], x[t]) + bias[r]} for {@code t < n}. */
    static void linear(float[] w, int rows, int cols, float[] bias, float[] x, int n, float[] y) {
        CpuParallel.parallelFor(rows, Math.max(1, GRAIN / Math.max(1, cols)), (from, to) -> {
            for (int r = from; r < to; r++) {
                int wBase = r * cols;
                float b = bias != null ? bias[r] : 0f;
                for (int t = 0; t < n; t++) {
                    int xBase = t * cols;
                    float sum = 0f;
                    for (int c = 0; c < cols; c++) {
                        sum += w[wBase + c] * x[xBase + c];
                    }
                    y[t * rows + r] = sum + b;
                }
            }
        });
    }

    /** {@code dx[t][c] += sum_r w[r][c] * dy[t][r]}. */
    static void linearInputGrad(float[] w, int rows, int cols, float[] dy, int n, float[] dx) {
        CpuParallel.parallelFor(n, 1, (from, to) -> {
            for (int t = from; t < to; t++) {
                int dxBase = t * cols;
                int dyBase = t * rows;
                for (int r = 0; r < rows; r++) {
                    float g = dy[dyBase + r];
                    if (g == 0f) {
                        continue;
                    }
                    int wBase = r * cols;
                    for (int c = 0; c < cols; c++) {
                        dx[dxBase + c] += w[wBase + c] * g;
                    }
                }
            }
        });
    }

    /** {@code dw[r][c] += sum_t dy[t][r] * x[t][c]}. */
    static void linearWeightGrad(int rows, int cols, float[] x, float[] dy, int n, float[] dw) {
        CpuParallel.parallelFor(rows, Math.max(1, GRAIN / Math.max(1, cols)), (from, to) -> {
            for (int r = from; r < to; r++) {
                int wBase = r * cols;
                for (int c = 0; c < cols; c++) {
                    float sum = 0f;
                    for (int t = 0; t < n; t++) {
                        sum += dy[t * rows + r] * x[t * cols + c];
                    }
                    dw[wBase + c] += sum;
                }
            }
        });
    }

    /** {@code db[r] += sum_t dy[t][r]}. */
    static void linearBiasGrad(int rows, float[] dy, int n, float[] db) {
        for (int t = 0; t < n; t++) {
            int base = t * rows;
            for (int r = 0; r < rows; r++) {
                db[r] += dy[base + r];
            }
        }
    }

    // ------------------------------------------------------------------ normalization

    /**
     * Normalizes {@code nRows} rows of {@code rowLen}: {@code y = norm(x) * w + b}, the row's weight row being
     * {@code r % wRows}. Fills {@code stats[2r]} with the row mean (0 for RMS) and {@code stats[2r + 1]} with the
     * reciprocal scale, which {@link #normBackward} reuses.
     */
    static void normForward(NormKind kind, float eps, float[] x, int nRows, int rowLen, int wRows,
                            float[] w, float[] b, float[] y, float[] stats) {
        for (int r = 0; r < nRows; r++) {
            int base = r * rowLen;
            float mean = kind == NormKind.LAYER ? mean(x, base, rowLen) : 0f;
            float scale = rowScale(kind, eps, x, base, rowLen, mean);
            int wb = (r % wRows) * rowLen;
            stats[2 * r] = mean;
            stats[2 * r + 1] = scale;
            for (int c = 0; c < rowLen; c++) {
                float v = (x[base + c] - mean) * scale;
                if (w != null) {
                    v *= w[wb + c];
                }
                if (b != null) {
                    v += b[wb + c];
                }
                y[base + c] = v;
            }
        }
    }

    /**
     * Backward of {@link #normForward}: accumulates {@code dx}, {@code dw} and {@code db} (null for none) from
     * {@code dy} and the forward {@code stats}. The norm weight's gradient follows the rows it is shared by.
     */
    static void normBackward(NormKind kind, float[] x, float[] dy, int nRows, int rowLen, int wRows,
                             float[] w, float[] stats, float[] dx, float[] dw, float[] db) {
        float[] g = new float[rowLen];
        for (int r = 0; r < nRows; r++) {
            int base = r * rowLen;
            float mean = stats[2 * r];
            float scale = stats[2 * r + 1];
            int wb = (r % wRows) * rowLen;
            float meanG = 0f;
            float meanGx = 0f;
            for (int c = 0; c < rowLen; c++) {
                float gc = w != null ? w[wb + c] * dy[base + c] : dy[base + c];
                g[c] = gc;
                meanG += gc;
                float xh = (x[base + c] - mean) * scale;
                meanGx += gc * xh;
                if (dw != null && w != null) {
                    dw[wb + c] += dy[base + c] * xh;
                }
                if (db != null) {
                    db[wb + c] += dy[base + c];
                }
            }
            if (kind == NormKind.RMS) {
                float sum = 0f;
                for (int c = 0; c < rowLen; c++) {
                    sum += g[c] * x[base + c];
                }
                float k = scale * scale * scale / rowLen;
                for (int c = 0; c < rowLen; c++) {
                    dx[base + c] += scale * g[c] - k * x[base + c] * sum;
                }
            } else {
                meanG /= rowLen;
                meanGx /= rowLen;
                for (int c = 0; c < rowLen; c++) {
                    float xh = (x[base + c] - mean) * scale;
                    dx[base + c] += scale * (g[c] - meanG - xh * meanGx);
                }
            }
        }
    }

    private static float mean(float[] x, int base, int len) {
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            sum += x[base + i];
        }
        return sum / len;
    }

    private static float rowScale(NormKind kind, float eps, float[] x, int base, int len, float mean) {
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            float v = x[base + i] - mean;
            sum += v * v;
        }
        return (float) (1.0 / Math.sqrt(sum / len + eps));
    }

    // ------------------------------------------------------------------ rotary embedding

    /** Rotates {@code nHeads} heads of every token in place ({@link RopeFreqTable#pairFirst} pairs). */
    static void ropeForward(RopeFreqTable table, float[] x, int nTokens, int nHeads, int headStride,
                            int rowStride) {
        float[] cs = new float[2 * table.pairs()];
        int span = table.pairSpan();
        for (int t = 0; t < nTokens; t++) {
            table.cosSin(t, cs);
            int rowBase = t * rowStride;
            for (int h = 0; h < nHeads; h++) {
                int base = rowBase + h * headStride;
                for (int i = 0; i < table.pairs(); i++) {
                    int a = base + table.pairFirst(i);
                    int b = a + span;
                    float x0 = x[a];
                    float x1 = x[b];
                    x[a] = x0 * cs[2 * i] - x1 * cs[2 * i + 1];
                    x[b] = x0 * cs[2 * i + 1] + x1 * cs[2 * i];
                }
            }
        }
    }

    /** The transpose of {@link #ropeForward}: accumulates the rotated gradient back through the same rotation. */
    static void ropeBackward(RopeFreqTable table, float[] dx, int nTokens, int nHeads, int headStride,
                             int rowStride) {
        float[] cs = new float[2 * table.pairs()];
        int span = table.pairSpan();
        for (int t = 0; t < nTokens; t++) {
            table.cosSin(t, cs);
            int rowBase = t * rowStride;
            for (int h = 0; h < nHeads; h++) {
                int base = rowBase + h * headStride;
                for (int i = 0; i < table.pairs(); i++) {
                    int a = base + table.pairFirst(i);
                    int b = a + span;
                    float g0 = dx[a];
                    float g1 = dx[b];
                    dx[a] = g0 * cs[2 * i] + g1 * cs[2 * i + 1];
                    dx[b] = -g0 * cs[2 * i + 1] + g1 * cs[2 * i];
                }
            }
        }
    }

    // ------------------------------------------------------------------ causal attention

    /**
     * Causal grouped-query attention of {@code n} tokens: {@code out[t][h] = softmax_j(q[t][h] . k[j][kvh] * scale) v[j][kvh]}
     * over the keys {@code j <= t} (and {@code j >= t - window + 1} for a positive sliding {@code window}).
     * {@code softcap > 0} caps each scaled score as {@code softcap * tanh(score / softcap)} before the softmax.
     * {@code scores} and {@code probs} are caller scratch of at least {@code nHead * n} values; {@code dp} of at least
     * {@code n}.
     */
    static void attentionForward(int n, int nHead, int nHeadKv, int headDimK, int headDimV, float scale,
                                 int window, float softcap, float[] q, int qStride, float[] k, int kStride,
                                 float[] v, int vStride, float[] out, int outStride, float[] scores, float[] probs) {
        int group = nHead / nHeadKv;
        for (int t = 0; t < n; t++) {
            int qRow = t * qStride;
            int outRow = t * outStride;
            int first = firstKey(t, window);
            for (int h = 0; h < nHead; h++) {
                int kvh = h / group;
                int qBase = qRow + h * headDimK;
                int sBase = h * n;
                for (int j = first; j <= t; j++) {
                    int kBase = j * kStride + kvh * headDimK;
                    float sum = 0f;
                    for (int d = 0; d < headDimK; d++) {
                        sum += q[qBase + d] * k[kBase + d];
                    }
                    scores[sBase + j] = capped(sum * scale, softcap);
                }
                softmaxRange(scores, sBase, probs, sBase, first, t + 1);
                for (int d = 0; d < headDimV; d++) {
                    float sum = 0f;
                    for (int j = first; j <= t; j++) {
                        sum += probs[sBase + j] * v[j * vStride + kvh * headDimV + d];
                    }
                    out[outRow + h * headDimV + d] = sum;
                }
            }
        }
    }

    /**
     * Backward of {@link #attentionForward}: accumulates {@code dq}, {@code dk} and {@code dv} from {@code dout},
     * recomputing each query row's softmax from the current {@code q}, {@code k} and {@code v}.
     */
    static void attentionBackward(int n, int nHead, int nHeadKv, int headDimK, int headDimV, float scale,
                                  int window, float softcap, float[] q, int qStride, float[] k, int kStride,
                                  float[] v, int vStride, float[] dout, int outStride, float[] dq, float[] dk,
                                  float[] dv, float[] scores, float[] probs, float[] dp) {
        int group = nHead / nHeadKv;
        for (int t = 0; t < n; t++) {
            int qRow = t * qStride;
            int outRow = t * outStride;
            int first = firstKey(t, window);
            for (int h = 0; h < nHead; h++) {
                int kvh = h / group;
                int qBase = qRow + h * headDimK;
                int outBase = outRow + h * headDimV;
                int sBase = h * n;
                for (int j = first; j <= t; j++) {
                    int kBase = j * kStride + kvh * headDimK;
                    float sum = 0f;
                    for (int d = 0; d < headDimK; d++) {
                        sum += q[qBase + d] * k[kBase + d];
                    }
                    scores[sBase + j] = capped(sum * scale, softcap);
                }
                softmaxRange(scores, sBase, probs, sBase, first, t + 1);
                float weighted = 0f;
                for (int j = first; j <= t; j++) {
                    int vBase = j * vStride + kvh * headDimV;
                    float sum = 0f;
                    for (int d = 0; d < headDimV; d++) {
                        sum += dout[outBase + d] * v[vBase + d];
                    }
                    dp[j] = sum;
                    weighted += probs[sBase + j] * sum;
                }
                for (int j = first; j <= t; j++) {
                    float ds = probs[sBase + j] * (dp[j] - weighted);
                    if (softcap > 0f) {
                        float tanh = scores[sBase + j] / softcap;
                        ds *= 1f - tanh * tanh;
                    }
                    int kBase = j * kStride + kvh * headDimK;
                    int vBase = j * vStride + kvh * headDimV;
                    for (int d = 0; d < headDimK; d++) {
                        dq[qBase + d] += ds * k[kBase + d] * scale;
                        dk[kBase + d] += ds * q[qBase + d] * scale;
                    }
                    float p = probs[sBase + j];
                    for (int d = 0; d < headDimV; d++) {
                        dv[vBase + d] += p * dout[outBase + d];
                    }
                }
            }
        }
    }

    /** The first key a query at position {@code t} attends: 0 for a full causal window, else its sliding window'. */
    private static int firstKey(int t, int window) {
        return window > 0 ? Math.max(0, t - window + 1) : 0;
    }

    /** A score after the optional attention soft-cap. */
    private static float capped(float score, float softcap) {
        return softcap > 0f ? softcap * (float) Math.tanh(score / softcap) : score;
    }

    /** Softmax of the {@code [from, to)} slice of a row into the same slice of {@code dst}. */
    private static void softmaxRange(float[] src, int srcOff, float[] dst, int dstOff, int from, int to) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = from; i < to; i++) {
            max = Math.max(max, src[srcOff + i]);
        }
        double sum = 0.0;
        for (int i = from; i < to; i++) {
            float e = (float) Math.exp(src[srcOff + i] - max);
            dst[dstOff + i] = e;
            sum += e;
        }
        float inv = (float) (1.0 / sum);
        for (int i = from; i < to; i++) {
            dst[dstOff + i] *= inv;
        }
    }

    /** Softmax of one row of {@code len} values into {@code dst}. */
    static void softmaxRow(float[] src, int srcOff, int len, float[] dst, int dstOff) {
        softmaxRange(src, srcOff, dst, dstOff, 0, len);
    }

    /**
     * The top-{@code nUsed} experts of one token by probability (the lowest index on ties), their weights renormalized
     * to sum to one when {@code normWeights} (the sum floored at {@link MoeSpec#NORM_SUM_MIN}) and scaled by
     * {@code scale}. Fills {@code index[idxOff .. idxOff + nUsed)} and {@code weight[wOff .. wOff + nUsed)}.
     */
    static void selectExperts(float[] probs, int off, int nExpert, int nUsed, boolean normWeights, float scale,
                              int[] index, int idxOff, float[] weight, int wOff) {
        for (int s = 0; s < nUsed; s++) {
            int best = -1;
            for (int e = 0; e < nExpert; e++) {
                boolean taken = false;
                for (int prior = 0; prior < s; prior++) {
                    if (index[idxOff + prior] == e) {
                        taken = true;
                        break;
                    }
                }
                if (!taken && (best < 0 || probs[off + e] > probs[off + best])) {
                    best = e;
                }
            }
            index[idxOff + s] = best;
            weight[wOff + s] = probs[off + best];
        }
        float sum = 0f;
        if (normWeights) {
            for (int s = 0; s < nUsed; s++) {
                sum += weight[wOff + s];
            }
            if (sum < MoeSpec.NORM_SUM_MIN) {
                sum = MoeSpec.NORM_SUM_MIN;
            }
        } else {
            sum = 1f;
        }
        for (int s = 0; s < nUsed; s++) {
            weight[wOff + s] = scale * weight[wOff + s] / sum;
        }
    }

    // ------------------------------------------------------------------ gated linear units

    /** {@code out[i] = glu(gate[i], up[i])} ({@link CpuOps#glu} formula). */
    static void gluForward(GluOp op, float[] gate, float[] up, float[] out, int n) {
        for (int i = 0; i < n; i++) {
            out[i] = gluValue(op, gate[i], up[i]);
        }
    }

    /**
     * Backward of {@link #gluForward}: accumulates {@code dgate} and {@code dup} from {@code dout}, the gate and up
     * values of the forward pass.
     */
    static void gluBackward(GluOp op, float[] gate, float[] up, float[] dout, float[] dgate, float[] dup, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            float u = up[i];
            float d = dout[i];
            dgate[i] += d * gluGateDerivative(op, g) * u;
            dup[i] += d * gluValue(op, g, 1.0f);
        }
    }

    /** The activation half of a GLU applied to {@code g} (the {@code u = 1} special case of {@link #gluValue}). */
    static float gluValue(GluOp op, float g, float u) {
        switch (op) {
            case SWIGLU:
                return ActMath.silu(g) * u;
            case GEGLU_TANH:
                return ActMath.geluTanh(g) * u;
            case GEGLU_ERF:
                return ActMath.geluErf(g) * u;
            case GEGLU_QUICK:
                return ActMath.geluQuick(g) * u;
            case REGLU:
                return ActMath.relu(g) * u;
            default:
                throw new UnsupportedOperationException("finetune does not train the " + op + " gate");
        }
    }

    /** Derivative of the activation half of a GLU. */
    private static float gluGateDerivative(GluOp op, float g) {
        switch (op) {
            case SWIGLU:
                return activationDerivative(ActOp.SILU, g);
            case GEGLU_TANH:
                return activationDerivative(ActOp.GELU_TANH, g);
            case GEGLU_ERF:
                return activationDerivative(ActOp.GELU_ERF, g);
            case GEGLU_QUICK:
                return activationDerivative(ActOp.GELU_QUICK, g);
            case REGLU:
                return g > 0f ? 1f : 0f;
            default:
                throw new UnsupportedOperationException("finetune does not train the " + op + " gate");
        }
    }

    /** {@code act(x)}. */
    static float activation(ActOp op, float x) {
        switch (op) {
            case SILU:
                return ActMath.silu(x);
            case GELU_TANH:
                return ActMath.geluTanh(x);
            case GELU_ERF:
                return ActMath.geluErf(x);
            case GELU_QUICK:
                return ActMath.geluQuick(x);
            case RELU:
                return ActMath.relu(x);
            default:
                throw new UnsupportedOperationException("finetune does not train the " + op + " activation");
        }
    }

    /** {@code d act(x) / dx}. */
    static float activationDerivative(ActOp op, float x) {
        switch (op) {
            case SILU: {
                float s = ActMath.sigmoid(x);
                return s * (1f + x * (1f - s));
            }
            case GELU_TANH: {
                float inner = ActMath.SQRT_2_OVER_PI * x * (1f + ActMath.GELU_COEF_A * x * x);
                float t = (float) Math.tanh(inner);
                float dInner = ActMath.SQRT_2_OVER_PI * (1f + 3f * ActMath.GELU_COEF_A * x * x);
                return 0.5f * (1f + t) + 0.5f * x * (1f - t * t) * dInner;
            }
            case GELU_ERF:
                return 0.5f * (1f + ActMath.erf(x * ActMath.SQRT_2_INV))
                        + (float) (x * Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI));
            case GELU_QUICK: {
                float s = ActMath.sigmoid(ActMath.GELU_QUICK_COEF * x);
                return s * (1f - ActMath.GELU_QUICK_COEF * x * (1f - s));
            }
            case RELU:
                return x > 0f ? 1f : 0f;
            default:
                throw new UnsupportedOperationException("finetune does not train the " + op + " activation");
        }
    }

    // ------------------------------------------------------------------ cross entropy

    /**
     * Fills {@code probs} with the softmax of one logits row and returns {@code -log(probs[label])}.
     *
     * @throws IndexOutOfBoundsException when {@code label} is outside the row
     */
    static float crossEntropy(float[] logits, int n, int label, float[] probs) {
        if (label < 0 || label >= n) {
            throw new IndexOutOfBoundsException("label " + label + " outside the " + n + "-token vocabulary");
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int v = 0; v < n; v++) {
            max = Math.max(max, logits[v]);
        }
        double sum = 0.0;
        for (int v = 0; v < n; v++) {
            float e = (float) Math.exp(logits[v] - max);
            probs[v] = e;
            sum += e;
        }
        float inv = (float) (1.0 / sum);
        for (int v = 0; v < n; v++) {
            probs[v] *= inv;
        }
        return (float) -(Math.log(Math.max(probs[label], Float.MIN_NORMAL)));
    }

    /** The vocabulary id of the highest logit of {@code logits}, the lowest id on ties. */
    static int argmax(float[] logits, int n) {
        int best = 0;
        for (int v = 1; v < n; v++) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        return best;
    }
}
