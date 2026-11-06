package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Row-major float helpers of the clip executors, vision and audio: bias and residual adds, layer scale, row
 * gathers, the {@code ggml_permute(0, 2, 1, 3)} + {@code ggml_cont} reorder the pixel shuffles are written in,
 * and the multi-head softmax attention. A row block {@code [n][width]} is stored flat at {@code t * width + c}.
 * Package-private executor support.
 */
final class VisionRows {

    private VisionRows() {
    }

    static float dot(float[] a, int aOff, float[] b, int bOff, int n) {
        float s = 0f;
        for (int i = 0; i < n; i++) {
            s += a[aOff + i] * b[bOff + i];
        }
        return s;
    }

    /** {@code x[t][i] += bias[i]} over every whole row of {@code bias.length}; null adds nothing. */
    static void addBias(float[] x, float[] bias) {
        if (bias == null) {
            return;
        }
        int n = bias.length;
        for (int base = 0; base + n <= x.length; base += n) {
            for (int i = 0; i < n; i++) {
                x[base + i] += bias[i];
            }
        }
    }

    /** {@code x[t][i] *= scale[i]} over every whole row ({@code ggml_mul} by a row vector); null keeps x. */
    static void mulCols(float[] x, float[] scale) {
        if (scale == null) {
            return;
        }
        int n = scale.length;
        for (int base = 0; base + n <= x.length; base += n) {
            for (int i = 0; i < n; i++) {
                x[base + i] *= scale[i];
            }
        }
    }

    /** {@code dst[i] += src[i]} over the whole of {@code dst}. */
    static void addInto(float[] src, float[] dst) {
        for (int i = 0; i < dst.length; i++) {
            dst[i] += src[i];
        }
    }

    /** {@code dst[i] += src[i]} over the first {@code n} values. */
    static void addPrefix(float[] src, float[] dst, int n) {
        for (int i = 0; i < n; i++) {
            dst[i] += src[i];
        }
    }

    /** The unary op over every value. */
    static void activate(ActOp op, float[] x) {
        CpuOps.unary(op, ActParams.NONE, x, 0, x, 0, x.length);
    }

    /** Layer norm of every {@code width}-wide chunk of {@code x} in place, with weight and optional bias. */
    static void layerNormChunks(float[] x, int width, float[] weight, float[] bias, float eps) {
        CpuOps.layerNorm(x, 0, x, 0, x.length / width, width, weight, bias, eps);
    }

    /**
     * Blocks of {@code width} values concatenated along the channels ({@code ggml_concat(..., 0)}): output row
     * {@code t} is row {@code t} of every block in turn.
     */
    static float[] concatChannels(List<float[]> blocks, int nRows, int width) {
        int k = blocks.size();
        float[] out = new float[nRows * width * k];
        for (int b = 0; b < k; b++) {
            float[] src = blocks.get(b);
            for (int t = 0; t < nRows; t++) {
                System.arraycopy(src, t * width, out, (t * k + b) * width, width);
            }
        }
        return out;
    }

    /** The rows of {@code flat} framed by a leading {@code first} and a trailing {@code last} row (BOI / EOI). */
    static float[][] framed(float[] first, float[] flat, float[] last, int width) {
        int n = flat.length / width;
        float[][] out = new float[n + 2][];
        out[0] = first.clone();
        for (int t = 0; t < n; t++) {
            out[t + 1] = Arrays.copyOfRange(flat, t * width, (t + 1) * width);
        }
        out[n + 1] = last.clone();
        return out;
    }

    /** Splits a flat block into fresh rows of {@code width}. */
    static float[][] tokenRows(float[] flat, int width) {
        float[][] rows = new float[flat.length / width][];
        for (int t = 0; t < rows.length; t++) {
            rows[t] = new float[width];
            System.arraycopy(flat, t * width, rows[t], 0, width);
        }
        return rows;
    }

    static float[][] gatherTokenRows(float[][] rows, int[] idx) {
        float[][] out = new float[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            out[i] = rows[idx[i]];
        }
        return out;
    }

    /** Output block {@code i} is input block {@code idx[i]}, each {@code width} values. */
    static float[] gatherRows(float[] flat, int[] idx, int width) {
        float[] out = new float[idx.length * width];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(flat, idx[i] * width, out, i * width, width);
        }
        return out;
    }

    /**
     * A contiguous {@code [d0, d1, d2]} ggml tensor with dims 1 and 2 swapped, made contiguous again
     * ({@code ggml_cont(ggml_permute(x, 0, 2, 1, 3))}): the result is {@code [d0, d2, d1]} with
     * {@code out[a + d0 * (c + d2 * b)] = x[a + d0 * (b + d1 * c)]}.
     */
    static float[] swap12(float[] x, int d0, int d1, int d2) {
        float[] out = new float[x.length];
        for (int c = 0; c < d2; c++) {
            for (int b = 0; b < d1; b++) {
                System.arraycopy(x, d0 * (b + d1 * c), out, d0 * (c + d2 * b), d0);
            }
        }
        return out;
    }

    /**
     * The key rows each query row attends to, {@code [begin(row), end(row))}: the block-diagonal mask of a windowed
     * attention as key ranges.
     */
    interface KeyRanges {
        int begin(int row);

        int end(int row);
    }

    /** The causal mask: row {@code t} attends to keys {@code [0, t]} (a CLIP text tower). */
    static final KeyRanges CAUSAL = new KeyRanges() {
        @Override
        public int begin(int row) {
            return 0;
        }

        @Override
        public int end(int row) {
            return row + 1;
        }
    };

    /**
     * The query rows grouped into runs that share one key range, {@code {qBegin, qEnd, kBegin, kEnd}} each: one
     * run of every row over every key when {@code ranges} is null (the block-diagonal windows give one run each).
     */
    static List<int[]> keyRuns(KeyRanges ranges, int n) {
        List<int[]> out = new ArrayList<>();
        if (ranges == null) {
            out.add(new int[]{0, n, 0, n});
            return out;
        }
        int start = 0;
        for (int row = 1; row <= n; row++) {
            if (row == n || ranges.begin(row) != ranges.begin(start) || ranges.end(row) != ranges.end(start)) {
                int kb = ranges.begin(start);
                int ke = ranges.end(start);
                if (kb < 0 || ke > n || kb >= ke) {
                    throw new IllegalArgumentException("query row " + start + " attends to keys [" + kb + ", " + ke
                            + ") outside the " + n + " rows");
                }
                out.add(new int[]{start, row, kb, ke});
                start = row;
            }
        }
        return out;
    }

    /**
     * Multi-head softmax attention ({@code softmax(q k^T * scale) v} per head): {@code nq} query rows of
     * {@code nHead * dh} from row offset {@code qRowOff}, {@code nk} key and value rows of {@code kvWidth} from
     * row offset {@code kvRowOff}, query head {@code h} reading KV head {@code h / kvGroup}. Every query row
     * attends to all keys, or to its key range when {@code ranges} is non-null. Sharded over (head, query) pairs
     * on {@link CpuParallel}; every output row is computed by one task, so the result does not depend on the
     * thread count.
     *
     * @return {@code [nq][nHead * dh]} rows
     */
    static float[] attend(float[] q, int qRowOff, int nq, float[] k, float[] v, int kvRowOff, int nk, int kvWidth,
                          int nHead, int kvGroup, int dh, float scale, KeyRanges ranges) {
        return attend(q, qRowOff, nq, k, v, kvRowOff, nk, kvWidth, nHead, kvGroup, dh, scale, ranges, null);
    }

    /**
     * {@link #attend(float[], int, int, float[], float[], int, int, int, int, int, int, float, KeyRanges)} with
     * per-head attention sinks ({@code ggml_soft_max_add_sinks}): head {@code h}'s unscaled logit {@code sinks[h]}
     * joins the softmax maximum and denominator as a key whose value row is zero; null for none.
     */
    static float[] attend(float[] q, int qRowOff, int nq, float[] k, float[] v, int kvRowOff, int nk, int kvWidth,
                          int nHead, int kvGroup, int dh, float scale, KeyRanges ranges, float[] sinks) {
        int qWidth = nHead * dh;
        final float[] qRows = qRowOff == 0 ? q : rows(q, qRowOff, qWidth, nq);
        final float[] kRows = kvRowOff == 0 ? k : rows(k, kvRowOff, kvWidth, nk);
        final float[] vRows = kvRowOff == 0 ? v : rows(v, kvRowOff, kvWidth, nk);
        float[] out = new float[nq * qWidth];
        CpuParallel.parallelFor(nHead * nq, Math.max(1, 4096 / Math.max(1, nk)), (from, to) -> {
            float[] scores = new float[nk];
            for (int idx = from; idx < to; idx++) {
                int h = idx / nq;
                int tq = idx - h * nq;
                int lo = ranges == null ? 0 : ranges.begin(tq);
                int hi = ranges == null ? nk : ranges.end(tq);
                attendRow(qRows, tq * qWidth + h * dh, kRows, vRows, kvWidth, (h / kvGroup) * dh, dh, scale, lo,
                        hi, sinks == null ? Float.NEGATIVE_INFINITY : sinks[h], scores, out);
            }
        });
        return out;
    }

    /** The {@code n} rows of {@code width} values starting at row {@code rowOff}. */
    private static float[] rows(float[] x, int rowOff, int width, int n) {
        float[] out = new float[n * width];
        System.arraycopy(x, rowOff * width, out, 0, out.length);
        return out;
    }

    /**
     * One (head, query row) pair at {@code q[qBase]} over keys {@code [lo, hi)} into {@code out[qBase]}; a finite
     * {@code sink} logit joins the softmax as a key with a zero value row.
     */
    private static void attendRow(float[] q, int qBase, float[] k, float[] v, int kvWidth, int kvOff, int dh,
                                  float scale, int lo, int hi, float sink, float[] scores, float[] out) {
        float max = sink;
        for (int tk = lo; tk < hi; tk++) {
            float s = dot(q, qBase, k, tk * kvWidth + kvOff, dh) * scale;
            scores[tk] = s;
            if (s > max) {
                max = s;
            }
        }
        double sum = sink == Float.NEGATIVE_INFINITY ? 0 : Math.exp(sink - max);
        for (int tk = lo; tk < hi; tk++) {
            double e = Math.exp(scores[tk] - max);
            scores[tk] = (float) e;
            sum += e;
        }
        float inv = (float) (1.0 / sum);
        for (int tk = lo; tk < hi; tk++) {
            float wgt = scores[tk] * inv;
            if (wgt == 0f) {
                continue;
            }
            int vBase = tk * kvWidth + kvOff;
            for (int d = 0; d < dh; d++) {
                out[qBase + d] += wgt * v[vBase + d];
            }
        }
    }
}
