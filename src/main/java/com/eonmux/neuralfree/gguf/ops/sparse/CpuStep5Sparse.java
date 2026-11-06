package com.eonmux.neuralfree.gguf.ops.sparse;

import java.util.Arrays;

/**
 * CPU math of the Step5 sparse attention (vLLM {@code vllm/models/step5/common/ops}: {@code indexer_norm_rope.py},
 * {@code summary_update.py}, {@code sparse_indexer.py} and {@code sparse_decode_attention.py}).
 *
 * <p>A KV region is {@code regionBlock} consecutive tokens. The indexer scores each closed region by its summary key:
 * the online-softmax mean of the region's eight indexer keys, weighted per proxy dimension by {@code exp(index_z)}.
 * Region {@code position / regionBlock} is still open, so only regions below it are scored and the open region is
 * appended to every selection. The sparse attention then reads the selected regions' per-token K/V rows and applies
 * the per-head sigmoid output gate.</p>
 *
 * <p>All routines are stateless and work on caller-owned arrays; offsets are element offsets. The indexer query is
 * RMS-normalized over the proxy head with the zero-centered weight {@code 1 + w}, the indexer key is LayerNormed, and
 * both rotate their leading {@link #neoxRope} dimensions with NeoX pairs.</p>
 */
public final class CpuStep5Sparse {

    private CpuStep5Sparse() {
    }

    /**
     * Zero-centered RMS norm of {@code n} values: {@code x * rsqrt(mean(x^2) + eps) * (1 + weight)}.
     *
     * @param x      input, unread after the call
     * @param weight the checkpoint's norm weight, added to one
     */
    public static void rmsNormZeroCentered(float[] x, int off, int n, float[] weight, float eps,
                                           float[] out, int outOff) {
        checkRange(x, off, n, "rms input");
        checkRange(weight, 0, n, "rms weight");
        checkRange(out, outOff, n, "rms output");
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double v = x[off + i];
            sum += v * v;
        }
        float inv = (float) (1.0 / Math.sqrt(sum / n + eps));
        for (int i = 0; i < n; i++) {
            out[outOff + i] = x[off + i] * inv * (1.0f + weight[i]);
        }
    }

    /** LayerNorm of {@code n} values: {@code (x - mean) * rsqrt(var + eps) * weight + bias}. */
    public static void layerNorm(float[] x, int off, int n, float[] weight, float[] bias, float eps,
                                 float[] out, int outOff) {
        checkRange(x, off, n, "layernorm input");
        checkRange(weight, 0, n, "layernorm weight");
        checkRange(bias, 0, n, "layernorm bias");
        checkRange(out, outOff, n, "layernorm output");
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += x[off + i];
        }
        double mean = sum / n;
        double variance = 0.0;
        for (int i = 0; i < n; i++) {
            double d = x[off + i] - mean;
            variance += d * d;
        }
        float inv = (float) (1.0 / Math.sqrt(variance / n + eps));
        for (int i = 0; i < n; i++) {
            out[outOff + i] = (float) ((x[off + i] - mean) * inv) * weight[i] + bias[i];
        }
    }

    /**
     * NeoX rotary over the leading {@code ropeDim} dimensions of a {@code headDim}-wide head: pair {@code i} with
     * {@code i + ropeDim / 2}, angle {@code pos * theta^(-2i/ropeDim)}; the dimensions past {@code ropeDim} copy.
     */
    public static void neoxRope(float[] x, int off, int headDim, int ropeDim, int pos, double theta,
                                float[] out, int outOff) {
        checkRange(x, off, headDim, "rope input");
        checkRange(out, outOff, headDim, "rope output");
        if (ropeDim < 0 || ropeDim > headDim || ropeDim % 2 != 0) {
            throw new IllegalArgumentException("rope dim " + ropeDim + " must be even and within head dim " + headDim);
        }
        int half = ropeDim / 2;
        for (int i = 0; i < half; i++) {
            double angle = pos / Math.pow(theta, 2.0 * i / ropeDim);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float a = x[off + i];
            float b = x[off + i + half];
            out[outOff + i] = a * cos - b * sin;
            out[outOff + i + half] = a * sin + b * cos;
        }
        for (int i = ropeDim; i < headDim; i++) {
            out[outOff + i] = x[off + i];
        }
    }

    /**
     * The weighted-ReLU indexer score of one query token and KV group against every closed region:
     * {@code sum_h relu(q[h] . summary[r]) * weight[h]}, with {@code weight} already scaled by
     * {@code heads^-0.5} as vLLM's indexer does.
     *
     * @param q         query heads, {@code [head][dim]} from {@code qOff}, {@code headStride} apart
     * @param weights   per-head weights from {@code wOff}, {@code headStride} apart
     * @param summaries closed-region summary keys, {@code [region][dim]} from {@code sOff}, {@code regionStride} apart
     * @param scores    output, one per closed region
     */
    public static void indexerScores(int heads, int dim, float[] q, int qOff, int headStride,
                                     float[] weights, int wOff, int headStrideW,
                                     float[] summaries, int sOff, int regionStride, int regions,
                                     float[] scores, int outOff) {
        checkRange(scores, outOff, regions, "indexer scores");
        checkRange(q, qOff, heads == 0 ? 0 : (heads - 1) * headStride + dim, "indexer query");
        checkRange(weights, wOff, heads == 0 ? 0 : (heads - 1) * headStrideW + 1, "indexer weights");
        checkRange(summaries, sOff, regions == 0 ? 0 : (regions - 1) * regionStride + dim, "indexer summaries");
        for (int r = 0; r < regions; r++) {
            double score = 0.0;
            int sBase = sOff + r * regionStride;
            for (int h = 0; h < heads; h++) {
                int qBase = qOff + h * headStride;
                double dot = 0.0;
                for (int d = 0; d < dim; d++) {
                    dot += (double) q[qBase + d] * summaries[sBase + d];
                }
                if (dot > 0.0) {
                    score += dot * weights[wOff + h * headStrideW];
                }
            }
            scores[outOff + r] = (float) score;
        }
    }

    /**
     * Selects the closed regions with the highest scores and appends the open region {@code closedRegions}
     * (vLLM {@code select_regions}): when every closed region fits, all are emitted in order behind left padding;
     * otherwise the top {@code topK} are emitted sorted ascending. Ties at the cutoff prefer the smaller region id.
     *
     * @param out {@code topK + 1} slots: {@code -1} padding, then the selected region ids, then the open region
     * @return the number of valid entries written (padding excluded)
     */
    public static int selectRegions(float[] scores, int off, int closedRegions, int topK, int[] out, int outOff) {
        checkRange(scores, off, closedRegions, "region scores");
        checkRange(out, outOff, topK + 1, "region selection");
        Arrays.fill(out, outOff, outOff + topK + 1, -1);
        for (int r = 0; r < closedRegions; r++) {
            if (Float.isNaN(scores[off + r])) {
                throw new IllegalArgumentException("region score " + r + " is NaN");
            }
        }
        if (closedRegions <= topK) {
            int first = topK - closedRegions;
            for (int r = 0; r < closedRegions; r++) {
                out[outOff + first + r] = r;
            }
            out[outOff + topK] = closedRegions;
            return closedRegions + 1;
        }
        int[] chosen = new int[topK];
        int n = CpuTopK.topKRow(scores, off, closedRegions, topK, chosen, 0);
        Arrays.sort(chosen, 0, n);
        for (int i = 0; i < n; i++) {
            out[outOff + topK - n + i] = chosen[i];
        }
        out[outOff + topK] = closedRegions;
        return n + 1;
    }

    /**
     * Stores one open-region indexer row {@code [key | z]} in the ring and, when {@code position} closes its region,
     * writes that region's summary (vLLM {@code update_decode_values}): the online-softmax weighted mean over the
     * region's rows, one independent softmax per proxy dimension with the weights {@code exp(z[d])}.
     *
     * @param ring     {@code capacity} rows of {@code 2 * proxy} values, row {@code position % capacity}
     * @param mean     {@code [region][proxy]} summaries, zeroed before the first update
     * @param capacity ring rows; must hold a whole region in position order
     */
    public static void summaryUpdate(int position, int regionBlock, float[] key, int keyOff,
                                     float[] z, int zOff, int proxy, int capacity,
                                     float[][] ring, float[][] mean) {
        if (position < 0) {
            throw new IllegalArgumentException("summary position " + position + " is negative");
        }
        if (capacity < regionBlock) {
            throw new IllegalArgumentException("ring capacity " + capacity + " cannot hold a region of "
                    + regionBlock);
        }
        if (ring.length < capacity) {
            throw new IllegalArgumentException("ring holds " + ring.length + " rows, expected " + capacity);
        }
        checkRange(key, keyOff, proxy, "summary key");
        checkRange(z, zOff, proxy, "summary z");
        int row = position % capacity;
        if (ring[row] == null || ring[row].length < 2 * proxy) {
            ring[row] = new float[2 * proxy];
        }
        System.arraycopy(key, keyOff, ring[row], 0, proxy);
        System.arraycopy(z, zOff, ring[row], proxy, proxy);
        if ((position + 1) % regionBlock != 0) {
            return;
        }
        int region = position / regionBlock;
        if (region >= mean.length) {
            throw new IllegalArgumentException("region " + region + " outside the " + mean.length + " summaries");
        }
        if (mean[region] == null || mean[region].length < proxy) {
            mean[region] = new float[proxy];
        }
        float[] numerator = new float[proxy];
        float[] denominator = new float[proxy];
        float[] maximum = new float[proxy];
        Arrays.fill(maximum, Float.NEGATIVE_INFINITY);
        int first = region * regionBlock;
        for (int p = first; p <= position; p++) {
            float[] r = ring[p % capacity];
            for (int d = 0; d < proxy; d++) {
                float updatedMax = Math.max(maximum[d], r[proxy + d]);
                float scale = denominator[d] > 0f ? (float) Math.exp(maximum[d] - updatedMax) : 0f;
                float weight = (float) Math.exp(r[proxy + d] - updatedMax);
                numerator[d] = numerator[d] * scale + weight * r[d];
                denominator[d] = denominator[d] * scale + weight;
                maximum[d] = updatedMax;
            }
        }
        float[] summary = mean[region];
        for (int d = 0; d < proxy; d++) {
            summary[d] = numerator[d] / Math.max(denominator[d], 1e-20f);
        }
    }

    /**
     * Sparse GQA attention over the selected regions, with the per-head sigmoid output gate (vLLM
     * {@code sparse_decode_attention.py}): every query head attends to the tokens of its KV group's selected regions
     * that are at or before its position, one softmax over all of them, and the output is multiplied by
     * {@code sigmoid(gate)} before the output projection.
     *
     * @param q      {@code [token][head][headDim]}
     * @param k      {@code [token][kvHead][headDim]}
     * @param v      {@code [token][kvHead][headDim]}
     * @param sel       selection of every token and KV group, {@code topK + 1} slots (padding {@code -1}, last slot
     *                  the open region)
     * @param gate      per-head gate values, one per {@code [token][head]}, or null when the layer has no gate
     * @param out       {@code [token][head][headDim]}
     * @param selRow    selection stride between token rows
     * @param selGroup  selection stride between KV groups, the slot count per group
     */
    public static void sparseGqa(int nTokens, int heads, int nKvHeads, int headDim, int regionBlock,
                                 float[] q, int qOff, int qRowStride,
                                 float[] k, int kOff, int kRowStride,
                                 float[] v, int vOff, int vRowStride,
                                 float scale,
                                 int[] sel, int selOff, int selRow, int selGroup,
                                 int[] positions, int posOff,
                                 float[] gate, int gateOff, int gateStride,
                                 float[] out, int outOff, int outRowStride) {
        if (nKvHeads <= 0 || heads % nKvHeads != 0) {
            throw new IllegalArgumentException(heads + " query heads do not divide into " + nKvHeads + " KV heads");
        }
        if (regionBlock <= 0) {
            throw new IllegalArgumentException("region block " + regionBlock + " is not positive");
        }
        int headsPerGroup = heads / nKvHeads;
        int slots = selGroup;
        float[] logits = new float[Math.max(1, slots * regionBlock)];
        for (int t = 0; t < nTokens; t++) {
            int position = positions[posOff + t];
            for (int h = 0; h < heads; h++) {
                int group = h / headsPerGroup;
                int selBase = selOff + t * selRow + group * slots;
                int count = 0;
                for (int s = 0; s < slots; s++) {
                    int region = sel[selBase + s];
                    if (region < 0) {
                        continue;
                    }
                    int first = region * regionBlock;
                    int last = Math.min(first + regionBlock - 1, position);
                    for (int token = first; token <= last; token++) {
                        if (count >= logits.length) {
                            logits = Arrays.copyOf(logits, logits.length * 2);
                        }
                        double dot = 0.0;
                        int qBase = qOff + t * qRowStride + h * headDim;
                        int kBase = kOff + token * kRowStride + group * headDim;
                        for (int d = 0; d < headDim; d++) {
                            dot += (double) q[qBase + d] * k[kBase + d];
                        }
                        logits[count++] = (float) (dot * scale);
                    }
                }
                int outBase = outOff + t * outRowStride + h * headDim;
                if (count == 0) {
                    Arrays.fill(out, outBase, outBase + headDim, 0f);
                    continue;
                }
                float max = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < count; i++) {
                    max = Math.max(max, logits[i]);
                }
                double sum = 0.0;
                for (int i = 0; i < count; i++) {
                    float e = (float) Math.exp(logits[i] - max);
                    logits[i] = e;
                    sum += e;
                }
                float gateValue = gate == null ? 1.0f : sigmoid(gate[gateOff + t * gateStride + h]);
                for (int d = 0; d < headDim; d++) {
                    out[outBase + d] = 0f;
                }
                int cursor = 0;
                for (int s = 0; s < slots; s++) {
                    int region = sel[selBase + s];
                    if (region < 0) {
                        continue;
                    }
                    int first = region * regionBlock;
                    int last = Math.min(first + regionBlock - 1, position);
                    for (int token = first; token <= last; token++) {
                        float p = (float) (logits[cursor++] / sum) * gateValue;
                        int vBase = vOff + token * vRowStride + group * headDim;
                        for (int d = 0; d < headDim; d++) {
                            out[outBase + d] += p * v[vBase + d];
                        }
                    }
                }
            }
        }
    }

    /** The per-head sigmoid attention gate of one token. */
    public static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private static void checkRange(float[] a, int off, int n, String what) {
        if (a == null || off < 0 || n < 0 || off + n > a.length) {
            throw new IllegalArgumentException(what + " range [" + off + ", " + (off + n) + ") is outside "
                    + (a == null ? "a null array" : "length " + a.length));
        }
    }

    private static void checkRange(int[] a, int off, int n, String what) {
        if (a == null || off < 0 || n < 0 || off + n > a.length) {
            throw new IllegalArgumentException(what + " range [" + off + ", " + (off + n) + ") is outside "
                    + (a == null ? "a null array" : "length " + a.length));
        }
    }
}
