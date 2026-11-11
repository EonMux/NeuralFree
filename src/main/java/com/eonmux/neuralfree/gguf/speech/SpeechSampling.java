package com.eonmux.neuralfree.gguf.speech;

import java.util.Arrays;

/**
 * The truncated draw of llama.cpp's {@code clip_graph_qwen3tts_gen::code_gen::do_sampling}
 * (tools/mtmd/models/qwen3tts-gen.cpp:5-64): top-k keeps the {@code k} largest logits, top-p keeps the shortest
 * prefix of the descending distribution whose cumulative probability reaches {@code p} (the crossing element is
 * kept), and the draw walks the cumulative distribution of the kept candidates until it passes {@code rand}.
 *
 * <p>The host form of the same math over a copied logits row; the caller owns the row and the draw. Ties order by
 * the lower vocabulary index, as a stable descending sort of the ggml top-k does.
 */
final class SpeechSampling {

    private SpeechSampling() {
    }

    /**
     * One draw over {@code logits}.
     *
     * @param logits  the logits row; the array is copied, not modified
     * @param topK    top-k cut; 0 or at least the row length disables it
     * @param topP    top-p cut in {@code (0, 1]}; 1 disables it
     * @param rand    the uniform draw in {@code [0, 1)}
     * @return the drawn vocabulary index
     * @throws IllegalArgumentException when the row is empty, {@code topK} is negative, {@code topP} is outside
     *                                  {@code (0, 1]} or {@code rand} is outside {@code [0, 1)}
     */
    static int draw(float[] logits, int topK, float topP, float rand) {
        if (logits == null || logits.length == 0) {
            throw new IllegalArgumentException("sampling needs a non-empty logits row");
        }
        if (topK < 0) {
            throw new IllegalArgumentException("top-k must not be negative, got " + topK);
        }
        if (!(topP > 0f) || topP > 1f || Float.isNaN(topP)) {
            throw new IllegalArgumentException("top-p must be in (0, 1], got " + topP);
        }
        if (!(rand >= 0f) || rand >= 1f) {
            throw new IllegalArgumentException("the sampling draw must be in [0, 1), got " + rand);
        }
        int n = logits.length;
        int kept = topK > 0 && topK < n ? topK : n;
        int[] order = topOrder(logits, kept);
        float[] sorted = new float[kept];
        for (int i = 0; i < kept; i++) {
            sorted[i] = logits[order[i]];
        }
        if (topP < 1f) {
            softmax(sorted);
            float cumulative = 0f;
            int keep = kept;
            for (int i = 0; i < kept; i++) {
                cumulative += sorted[i];
                if (cumulative >= topP) {
                    keep = i + 1;
                    break;
                }
            }
            for (int i = 0; i < keep; i++) {
                sorted[i] = logits[order[i]];
            }
            kept = keep;
        }
        float[] probabilities = Arrays.copyOf(sorted, kept);
        softmax(probabilities);
        float cumulative = 0f;
        for (int i = 0; i < kept; i++) {
            cumulative += probabilities[i];
            if (cumulative > rand) {
                return order[i];
            }
        }
        return order[kept - 1];
    }

    /**
     * The indices of the {@code kept} largest logits in descending order, ties by the lower index: the order of a
     * stable descending sort under {@link Float#compare}. Each candidate is one packed primitive key — the
     * order-reversed sortable bits of its logit above its index — so ascending key order is the wanted order; a
     * bounded max-heap keeps the {@code kept} smallest keys when the cut is below the row length.
     */
    static int[] topOrder(float[] logits, int kept) {
        int n = logits.length;
        long[] keys;
        if (kept >= n) {
            keys = new long[n];
            for (int i = 0; i < n; i++) {
                keys[i] = key(logits[i], i);
            }
        } else {
            keys = new long[kept];
            for (int i = 0; i < kept; i++) {
                keys[i] = key(logits[i], i);
                siftUp(keys, i);
            }
            for (int i = kept; i < n; i++) {
                long k = key(logits[i], i);
                if (k < keys[0]) {
                    keys[0] = k;
                    siftDown(keys, kept);
                }
            }
        }
        Arrays.sort(keys);
        int[] order = new int[keys.length];
        for (int i = 0; i < keys.length; i++) {
            order[i] = (int) keys[i];
        }
        return order;
    }

    /**
     * The sort key of logit {@code v} at {@code index}: the bits of {@code v} mapped to a signed int that orders as
     * {@link Float#compare} does (negative zero below zero, every NaN above infinity), complemented so larger logits
     * come first, above the index so ties order by the lower index.
     */
    private static long key(float v, int index) {
        int bits = Float.floatToIntBits(v);
        int sortable = bits ^ ((bits >> 31) & 0x7fffffff);
        return ((long) ~sortable << 32) | (index & 0xffffffffL);
    }

    /** Restores the max-heap order of {@code heap[0..i]} after {@code heap[i]} was set. */
    private static void siftUp(long[] heap, int i) {
        long v = heap[i];
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (heap[parent] >= v) {
                break;
            }
            heap[i] = heap[parent];
            i = parent;
        }
        heap[i] = v;
    }

    /** Restores the max-heap order of {@code heap[0..size)} after its root was replaced. */
    private static void siftDown(long[] heap, int size) {
        long v = heap[0];
        int i = 0;
        while (true) {
            int child = 2 * i + 1;
            if (child >= size) {
                break;
            }
            if (child + 1 < size && heap[child + 1] > heap[child]) {
                child++;
            }
            if (heap[child] <= v) {
                break;
            }
            heap[i] = heap[child];
            i = child;
        }
        heap[i] = v;
    }

    /** In-place softmax over {@code x[0, n)} with the max subtracted. */
    private static void softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            if (v > max) {
                max = v;
            }
        }
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double e = Math.exp((double) x[i] - max);
            x[i] = (float) e;
            sum += e;
        }
        if (!(sum > 0.0) || Double.isInfinite(sum)) {
            float uniform = 1.0f / x.length;
            Arrays.fill(x, uniform);
            return;
        }
        float scale = (float) (1.0 / sum);
        for (int i = 0; i < x.length; i++) {
            x[i] *= scale;
        }
    }

    /** The highest-logit index, the lowest id on ties. */
    static int argmax(float[] logits) {
        int best = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] > bestValue) {
                bestValue = logits[i];
                best = i;
            }
        }
        return best;
    }
}
