package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;

import java.util.Arrays;

/**
 * The chain {@link com.eonmux.neuralfree.gguf.engine.Sampler} ran before the stage split, as a reference the
 * stage chain must reproduce per candidate: bias, penalties, top-k, typical-p, top-p, min-p, temperature, then the
 * dist draw or the argmax. Test-only support for the order tests; no thread safety.
 */
final class PreSplitChain {

    private final int[] banned;
    private final History history;
    private final float repeatPenalty;
    private final float presencePenalty;
    private final float frequencyPenalty;
    private final boolean penalties;
    private final int topK;
    private final float typicalP;
    private final float topP;
    private final float minP;
    private final float temperature;
    private final boolean greedy;
    private final java.util.SplittableRandom rng;

    private int[] ids = new int[0];
    private float[] logits = new float[0];
    private float[] probs = new float[0];
    private int[] srcIds = new int[0];
    private float[] srcLogits = new float[0];
    private float[] srcProbs = new float[0];
    private long[] keys = new long[0];
    private int size;
    private boolean sorted;
    private int selected = -1;

    PreSplitChain(SamplerConfig c, int[] bannedTokens, long seed) {
        this.banned = bannedTokens.clone();
        this.repeatPenalty = c.repeatPenalty();
        this.presencePenalty = c.presencePenalty();
        this.frequencyPenalty = c.frequencyPenalty();
        this.penalties = c.penaltiesEnabled();
        this.topK = c.topK();
        this.typicalP = c.typicalP();
        this.topP = c.topP();
        this.minP = c.minP();
        this.temperature = c.temperature();
        this.greedy = c.isGreedy();
        this.history = new History(penalties ? c.penaltyLastN() : 0);
        this.rng = new java.util.SplittableRandom(seed);
    }

    void accept(int token) {
        history.accept(token);
    }

    void apply(float[] row) {
        size = row.length;
        ensureScratch(size);
        for (int i = 0; i < size; i++) {
            float v = row[i];
            ids[i] = i;
            logits[i] = Float.isNaN(v) ? Float.NEGATIVE_INFINITY : v;
            probs[i] = 0f;
        }
        for (int id : banned) {
            if (id < size) {
                logits[id] = Float.NEGATIVE_INFINITY;
            }
        }
        applyPenalties();
        if (greedy) {
            if (typicalP < 1f) {
                applyTopK();
                applyTypical();
                applyTopP();
                applyMinP();
            }
            int best = 0;
            for (int i = 1; i < size; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            if (logits[best] == Float.NEGATIVE_INFINITY) {
                throw new IllegalStateException("no candidate has a finite logit");
            }
            selected = best;
        } else {
            applyTopK();
            applyTypical();
            applyTopP();
            applyMinP();
            applyTemperature();
            drawDist();
        }
    }

    int size() {
        return size;
    }

    int id(int i) {
        return ids[i];
    }

    float logit(int i) {
        return logits[i];
    }

    float prob(int i) {
        return probs[i];
    }

    boolean sorted() {
        return sorted;
    }

    /** Candidates are still in vocabulary order here, so a token's candidate index is its id. */
    private void applyPenalties() {
        if (!penalties) {
            return;
        }
        for (int a = 0; a < history.nActive(); a++) {
            int id = history.active(a);
            if (id >= size) {
                continue;
            }
            int count = history.count(id);
            float l = logits[id];
            l = l <= 0f ? l * repeatPenalty : l / repeatPenalty;
            l -= (float) count * frequencyPenalty + presencePenalty;
            logits[id] = l;
        }
        sorted = false;
    }

    private void applyTopK() {
        if (topK <= 0) {
            return;
        }
        int k = Math.min(topK, size);
        if (!sorted) {
            partialSort(k);
        }
        size = k;
    }

    private void applyTypical() {
        if (typicalP >= 1f) {
            return;
        }
        softmax(true);
        int n = size;
        float entropy = 0f;
        for (int i = 0; i < n; i++) {
            float p = probs[i];
            if (p > 0f) {
                entropy += -p * (float) Math.log(p);
            }
        }
        long[] k = keys;
        for (int i = 0; i < n; i++) {
            float p = probs[i];
            float shifted = p > 0f ? Math.abs(-(float) Math.log(p) - entropy) : Float.POSITIVE_INFINITY;
            k[i] = Sampler.Candidates.ascendingKey(shifted, i);
        }
        Arrays.sort(k, 0, n);
        float cum = 0f;
        int last = n;
        for (int i = 0; i < n; i++) {
            cum += probs[(int) k[i]];
            if (cum > typicalP) {
                last = i + 1;
                break;
            }
        }
        permute(k, last);
        size = last;
        sorted = false;
    }

    private void applyTopP() {
        if (topP >= 1f) {
            return;
        }
        softmax(false);
        if (!sorted) {
            sortAll();
        }
        float cum = 0f;
        int last = size;
        for (int i = 0; i < size; i++) {
            cum += probs[i];
            if (cum >= topP) {
                last = i + 1;
                break;
            }
        }
        size = last;
    }

    private void applyMinP() {
        if (minP <= 0f || size == 0) {
            return;
        }
        if (!sorted) {
            float max = maxLogit();
            float minLogit = max + (float) Math.log(minP);
            int kept = 0;
            for (int i = 0; i < size; i++) {
                if (logits[i] >= minLogit) {
                    move(i, kept++);
                }
            }
            size = kept;
            return;
        }
        float minLogit = logits[0] + (float) Math.log(minP);
        int i = 1;
        while (i < size && logits[i] >= minLogit) {
            i++;
        }
        size = i;
    }

    private void applyTemperature() {
        if (temperature <= 0f) {
            return;
        }
        for (int i = 0; i < size; i++) {
            logits[i] /= temperature;
        }
    }

    /** llama_sampler_softmax_impl: probabilities of the current candidates, optionally sorting them first. */
    private void softmax(boolean sort) {
        if (sort && !sorted) {
            sortAll();
        }
        int n = size;
        float max = sorted ? logits[0] : maxLogit();
        if (max == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit");
        }
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            float p = expShift(logits[i], max);
            probs[i] = p;
            sum += p;
        }
        for (int i = 0; i < n; i++) {
            probs[i] /= sum;
        }
    }

    private void drawDist() {
        double u = rng.nextDouble();
        int n = size;
        if (n == 1) {
            if (logits[0] == Float.NEGATIVE_INFINITY) {
                throw new IllegalStateException("no candidate has a finite logit");
            }
            probs[0] = 1f;
            selected = 0;
            return;
        }
        float max = sorted ? logits[0] : maxLogit();
        if (max == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit");
        }
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            float p = expShift(logits[i], max);
            probs[i] = p;
            sum += p;
        }
        double target = sum * u;
        double run = 0.0;
        int draw = -1;
        int lastPositive = -1;
        for (int i = 0; i < n; i++) {
            float p = probs[i];
            if (p > 0f) {
                lastPositive = i;
                if (draw < 0) {
                    run += p;
                    if (run >= target) {
                        draw = i;
                    }
                }
            }
            probs[i] = (float) (p / sum);
        }
        selected = draw >= 0 ? draw : lastPositive;
    }

    private static float expShift(float logit, float max) {
        if (max == Float.POSITIVE_INFINITY) {
            return logit == Float.POSITIVE_INFINITY ? 1f : 0f;
        }
        return (float) Math.exp(logit - max);
    }

    private void sortAll() {
        long[] k = keys;
        for (int i = 0; i < size; i++) {
            k[i] = Sampler.Candidates.descendingKey(logits[i], i);
        }
        Arrays.sort(k, 0, size);
        permute(k, size);
        sorted = true;
    }

    /** Moves the {@code k} highest logits, in descending order, to the front (partial sort in place). */
    private void partialSort(int k) {
        int n = size;
        if ((long) k * 4 >= n) {
            sortAll();
            return;
        }
        long[] heap = new long[k];
        int heapSize = 0;
        for (int i = 0; i < n; i++) {
            long key = Sampler.Candidates.descendingKey(logits[i], i);
            if (heapSize < k) {
                heap[heapSize] = key;
                siftUp(heap, heapSize++);
            } else if (key < heap[0]) {
                heap[0] = key;
                siftDown(heap, heapSize);
            }
        }
        Arrays.sort(heap, 0, k);
        permute(heap, k);
        sorted = true;
    }

    private void permute(long[] order, int n) {
        for (int j = 0; j < n; j++) {
            int src = (int) order[j];
            srcIds[j] = ids[src];
            srcLogits[j] = logits[src];
            srcProbs[j] = probs[src];
        }
        System.arraycopy(srcIds, 0, ids, 0, n);
        System.arraycopy(srcLogits, 0, logits, 0, n);
        System.arraycopy(srcProbs, 0, probs, 0, n);
    }

    private void move(int from, int to) {
        if (from != to) {
            ids[to] = ids[from];
            logits[to] = logits[from];
            probs[to] = probs[from];
        }
    }

    private float maxLogit() {
        float max = logits[0];
        for (int i = 1; i < size; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }
        return max;
    }

    private void ensureScratch(int n) {
        if (ids.length < n) {
            ids = new int[n];
            logits = new float[n];
            probs = new float[n];
            srcIds = new int[n];
            srcLogits = new float[n];
            srcProbs = new float[n];
            keys = new long[n];
        }
    }

    private static void siftUp(long[] h, int i) {
        long v = h[i];
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (h[parent] >= v) {
                break;
            }
            h[i] = h[parent];
            i = parent;
        }
        h[i] = v;
    }

    private static void siftDown(long[] h, int size) {
        long v = h[0];
        int i = 0;
        while (true) {
            int child = 2 * i + 1;
            if (child >= size) {
                break;
            }
            if (child + 1 < size && h[child + 1] > h[child]) {
                child++;
            }
            if (h[child] <= v) {
                break;
            }
            h[i] = h[child];
            i = child;
        }
    }

    /**
     * Accepted tokens of the penalty window, as Sampler kept them before the split: a ring of the last
     * {@code window} tokens, or every token for the whole context ({@code -1}), nothing for a window of 0.
     */
    private static final class History {

        private final int window;
        private int[] ring;
        private int head;
        private int stored;
        private int[] counts = new int[0];
        private int[] activePos = new int[0];
        private int[] active = new int[16];
        private int nActive;

        History(int window) {
            this.window = window;
            this.ring = window > 0 ? new int[window] : null;
        }

        void accept(int token) {
            if (window == 0) {
                return;
            }
            increment(token);
            if (window > 0) {
                if (stored == window) {
                    decrement(ring[head]);
                    ring[head] = token;
                    head = (head + 1) % window;
                } else {
                    ring[stored++] = token;
                }
            }
        }

        int nActive() {
            return nActive;
        }

        int active(int a) {
            return active[a];
        }

        int count(int token) {
            return token < counts.length ? counts[token] : 0;
        }

        private void increment(int token) {
            if (token >= counts.length) {
                int cap = Math.max(token + 1, counts.length * 2);
                counts = Arrays.copyOf(counts, cap);
                activePos = Arrays.copyOf(activePos, cap);
            }
            if (counts[token]++ == 0) {
                if (nActive == active.length) {
                    active = Arrays.copyOf(active, nActive * 2);
                }
                activePos[token] = nActive;
                active[nActive++] = token;
            }
        }

        private void decrement(int token) {
            if (--counts[token] == 0) {
                int pos = activePos[token];
                int last = active[--nActive];
                active[pos] = last;
                activePos[last] = pos;
            }
        }
    }
}