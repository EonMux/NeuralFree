package com.eonmux.neuralfree.gguf.engine;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Host token sampler running llama.cpp's sampler chain over one logits row (llama.cpp src/llama-sampler.cpp, chain
 * order of llama.cpp common/common.h {@code common_params_sampling::samplers} as built by llama.cpp
 * common/sampling.cpp:329-406):
 * <ol>
 *   <li>logit bias: banned tokens ({@code tokenizer.ggml.suppress_tokens}, end-of-generation tokens under
 *       ignore-eos) get {@code -inf} (llama_sampler_logit_bias_apply);</li>
 *   <li>penalties over the last {@code penaltyLastN} accepted tokens (llama_sampler_penalties_apply);</li>
 *   <li>top-k (llama_sampler_top_k_impl), typical-p (llama_sampler_typical_apply), nucleus top-p
 *       (llama_sampler_top_p_apply), min-p (llama_sampler_min_p_apply);</li>
 *   <li>temperature (llama_sampler_temp_impl);</li>
 *   <li>a seeded categorical draw (llama_sampler_dist_apply), or argmax when greedy
 *       (llama_sampler_greedy_apply).</li>
 * </ol>
 * Disabled stages are skipped exactly as llama.cpp skips them. {@code min_keep} is llama.cpp's default 0.
 *
 * <p>Candidates keep llama.cpp's {@code llama_token_data_array} semantics: the stages operate on the current
 * candidate set in its current order, and a {@code sorted} flag records whether it is in descending logit order.
 * Sorting is deterministic: equal logits keep their current relative order, so a vocabulary-ordered set breaks
 * ties by the lower token id.
 *
 * <p>Greedy decoding (an explicit greedy request or a temperature {@code <= 0}) selects the highest logit after
 * bias, penalties and, when typical-p is active, the truncation stages; top-k, top-p and min-p always keep the
 * highest logit, so they are skipped when typical-p is off.
 *
 * <p>A sampler is a mutable runtime object owned by one generation: it holds the penalty history, the RNG and
 * reusable candidate buffers. Not thread-safe.
 */
public final class Sampler {

    private static final int[] NO_TOKENS = new int[0];

    private final int[] banned;
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
    private final long seed;
    private final History history;
    private final Candidates cur = new Candidates();
    private SplittableRandom rng;
    private long[] keys = new long[0];

    private Sampler(int[] banned, float repeatPenalty, float presencePenalty, float frequencyPenalty, int penaltyLastN,
                    boolean penalties, int topK, float typicalP, float topP, float minP, float temperature,
                    boolean greedy, long seed) {
        this.banned = banned;
        this.repeatPenalty = repeatPenalty;
        this.presencePenalty = presencePenalty;
        this.frequencyPenalty = frequencyPenalty;
        this.penalties = penalties;
        this.topK = topK;
        this.typicalP = typicalP;
        this.topP = topP;
        this.minP = minP;
        this.temperature = temperature;
        this.greedy = greedy;
        this.seed = seed;
        this.history = new History(penalties ? penaltyLastN : 0);
        this.rng = new SplittableRandom(seed);
    }

    /** The generation chain for {@code config} without banned tokens. */
    public static Sampler create(SamplerConfig config) {
        return create(config, NO_TOKENS);
    }

    /**
     * The generation chain for {@code config}.
     *
     * @param bannedTokens token ids whose logits are forced to {@code -inf} before every other stage (copied;
     *                     duplicates are allowed)
     * @throws IllegalArgumentException when a banned id is negative
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(bannedTokens, "bannedTokens");
        int[] banned = bannedTokens.clone();
        for (int id : banned) {
            if (id < 0) {
                throw new IllegalArgumentException("banned token id " + id + " is negative");
            }
        }
        long seed = config.seeded() ? config.seed() : randomSeed();
        return new Sampler(banned, config.repeatPenalty(), config.presencePenalty(), config.frequencyPenalty(),
                config.penaltyLastN(), config.penaltiesEnabled(), config.topK(), config.typicalP(), config.topP(),
                config.minP(), config.temperature(), config.isGreedy(), seed);
    }

    /**
     * The masked-diffusion chain of llama.cpp examples/diffusion/diffusion.cpp:134-145: top-k when {@code topK > 0},
     * top-p when {@code topP < 1}, temperature when {@code temperature > 0}, then the seeded draw. A temperature
     * {@code <= 0} only removes the temperature stage; the draw still runs.
     *
     * @param seed RNG seed, or {@link SamplerConfig#RANDOM_SEED}
     * @throws IllegalArgumentException for {@code topP} outside [0, 1], a non-finite temperature or a seed below -1
     */
    public static Sampler diffusionChain(int topK, float topP, float temperature, long seed) {
        if (!(topP >= 0f && topP <= 1f)) {
            throw new IllegalArgumentException("topP must be in [0, 1], got " + topP);
        }
        if (Float.isNaN(temperature) || Float.isInfinite(temperature)) {
            throw new IllegalArgumentException("temperature must be finite, got " + temperature);
        }
        if (seed < SamplerConfig.RANDOM_SEED) {
            throw new IllegalArgumentException("seed must be >= 0, or -1 for a random seed, got " + seed);
        }
        long s = seed == SamplerConfig.RANDOM_SEED ? randomSeed() : seed;
        return new Sampler(NO_TOKENS, 1f, 0f, 0f, 0, false, topK, 1f, topP, 0f, temperature, false, s);
    }

    /** A fresh non-negative seed for {@link SamplerConfig#RANDOM_SEED}. */
    static long randomSeed() {
        return ThreadLocalRandom.current().nextLong() & Long.MAX_VALUE;
    }

    /** The effective RNG seed (drawn once when the configuration asked for a random seed). */
    public long seed() {
        return seed;
    }

    /** Whether the chain ends in argmax rather than a random draw. */
    public boolean greedy() {
        return greedy;
    }

    /**
     * True when the chain reduces to the argmax of the raw logits: no banned tokens, no penalties, and either
     * greedy without typical-p or a top-k of 1 (llama.cpp common/sampling.cpp also replaces the draw by greedy for
     * {@code top_k == 1}). A forward backend's on-device argmax then selects the same token.
     */
    public boolean argmaxOnly() {
        return banned.length == 0 && !penalties && ((greedy && typicalP >= 1f) || topK == 1);
    }

    /** Records an accepted token for the penalties (prompt tokens included, as llama.cpp completion does). */
    public void accept(int token) {
        history.accept(token);
    }

    /** {@link #accept(int)} for {@code tokens[off .. off + n)}. */
    public void accept(int[] tokens, int off, int n) {
        Objects.requireNonNull(tokens, "tokens");
        if (off < 0 || n < 0 || off + n > tokens.length) {
            throw new IndexOutOfBoundsException("token range [" + off + ", " + (off + n) + ") outside " + tokens.length);
        }
        for (int i = off; i < off + n; i++) {
            history.accept(tokens[i]);
        }
    }

    /** Forgets the penalty history and restarts the RNG from {@link #seed()}. */
    public void reset() {
        history.clear();
        rng = new SplittableRandom(seed);
    }

    /**
     * Runs the chain over {@code logits} and returns the selected token id. {@code logits} is not modified.
     *
     * @throws IllegalArgumentException when the row is empty
     * @throws IllegalStateException    when no candidate has a finite logit
     */
    public int sample(float[] logits) {
        return apply(logits).selectedId();
    }

    /**
     * Runs the chain over {@code logits} and returns the final candidate set with its probabilities and
     * selection (llama.cpp {@code cur_p} after {@code llama_sampler_apply}). The view stays valid until the next
     * call on this sampler. {@code logits} is not modified.
     *
     * @throws IllegalArgumentException when the row is empty
     * @throws IllegalStateException    when no candidate has a finite logit
     */
    public Candidates apply(float[] logits) {
        Objects.requireNonNull(logits, "logits");
        if (logits.length == 0) {
            throw new IllegalArgumentException("cannot sample from an empty logits row");
        }
        cur.load(logits);
        applyBias();
        applyPenalties();
        if (greedy) {
            if (typicalP < 1f) {
                applyTopK();
                applyTypical();
                applyTopP();
                applyMinP();
            }
            selectArgmax();
        } else {
            applyTopK();
            applyTypical();
            applyTopP();
            applyMinP();
            applyTemperature();
            drawDist();
        }
        return cur;
    }

    // ------------------------------------------------------------------ stages

    private void applyBias() {
        for (int id : banned) {
            if (id < cur.size) {
                cur.logits[id] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    /** Candidates are still in vocabulary order here, so a token's candidate index is its id. */
    private void applyPenalties() {
        if (!penalties) {
            return;
        }
        for (int a = 0; a < history.nActive; a++) {
            int id = history.active[a];
            if (id >= cur.size) {
                continue;
            }
            int count = history.counts[id];
            float l = cur.logits[id];
            l = l <= 0f ? l * repeatPenalty : l / repeatPenalty;
            l -= (float) count * frequencyPenalty + presencePenalty;
            cur.logits[id] = l;
        }
        cur.sorted = false;
    }

    private void applyTopK() {
        if (topK <= 0) {
            return;
        }
        int k = Math.min(topK, cur.size);
        if (!cur.sorted) {
            partialSort(k);
        }
        cur.size = k;
    }

    /** Locally typical sampling; a zero probability contributes nothing to the entropy and ranks last. */
    private void applyTypical() {
        if (typicalP >= 1f) {
            return;
        }
        softmax(true);
        int n = cur.size;
        float entropy = 0f;
        for (int i = 0; i < n; i++) {
            float p = cur.probs[i];
            if (p > 0f) {
                entropy += -p * (float) Math.log(p);
            }
        }
        long[] k = keys(n);
        for (int i = 0; i < n; i++) {
            float p = cur.probs[i];
            float shifted = p > 0f ? Math.abs(-(float) Math.log(p) - entropy) : Float.POSITIVE_INFINITY;
            k[i] = ascendingKey(shifted, i);
        }
        Arrays.sort(k, 0, n);
        float cum = 0f;
        int last = n;
        for (int i = 0; i < n; i++) {
            cum += cur.probs[(int) k[i]];
            if (cum > typicalP) {
                last = i + 1;
                break;
            }
        }
        cur.permute(k, last);
        cur.size = last;
        cur.sorted = false;
    }

    /** Nucleus sampling: the smallest prefix of the descending order whose probability reaches {@code topP}. */
    private void applyTopP() {
        if (topP >= 1f) {
            return;
        }
        softmax(false);
        if (!cur.sorted) {
            sortAll();
        }
        float cum = 0f;
        int last = cur.size;
        for (int i = 0; i < cur.size; i++) {
            cum += cur.probs[i];
            if (cum >= topP) {
                last = i + 1;
                break;
            }
        }
        cur.size = last;
    }

    private void applyMinP() {
        if (minP <= 0f || cur.size == 0) {
            return;
        }
        if (!cur.sorted) {
            float max = cur.maxLogit();
            float minLogit = max + (float) Math.log(minP);
            int kept = 0;
            for (int i = 0; i < cur.size; i++) {
                if (cur.logits[i] >= minLogit) {
                    cur.move(i, kept++);
                }
            }
            cur.size = kept;
            return;
        }
        float minLogit = cur.logits[0] + (float) Math.log(minP);
        int i = 1;
        while (i < cur.size && cur.logits[i] >= minLogit) {
            i++;
        }
        cur.size = i;
    }

    private void applyTemperature() {
        if (temperature <= 0f) {
            return;
        }
        for (int i = 0; i < cur.size; i++) {
            cur.logits[i] /= temperature;
        }
    }

    /** Argmax in the current order (the first maximum wins). */
    private void selectArgmax() {
        int best = 0;
        for (int i = 1; i < cur.size; i++) {
            if (cur.logits[i] > cur.logits[best]) {
                best = i;
            }
        }
        if (cur.logits[best] == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
        }
        cur.selected = best;
    }

    /**
     * Softmax and a single categorical draw in the current order (llama_sampler_dist_apply): the first candidate
     * whose running sum reaches {@code sum * u}, {@code u} uniform in [0, 1); candidates of probability 0 are never
     * selected. One uniform is drawn per call, also for a single candidate.
     */
    private void drawDist() {
        double u = rng.nextDouble();
        int n = cur.size;
        if (n == 1) {
            if (cur.logits[0] == Float.NEGATIVE_INFINITY) {
                throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
            }
            cur.probs[0] = 1f;
            cur.selected = 0;
            return;
        }
        float max = cur.sorted ? cur.logits[0] : cur.maxLogit();
        if (max == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
        }
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            float p = expShift(cur.logits[i], max);
            cur.probs[i] = p;
            sum += p;
        }
        double target = sum * u;
        double run = 0.0;
        int selected = -1;
        int lastPositive = -1;
        for (int i = 0; i < n; i++) {
            float p = cur.probs[i];
            if (p > 0f) {
                lastPositive = i;
                if (selected < 0) {
                    run += p;
                    if (run >= target) {
                        selected = i;
                    }
                }
            }
            cur.probs[i] = (float) (p / sum);
        }
        cur.selected = selected >= 0 ? selected : lastPositive;
    }

    /** llama_sampler_softmax_impl: probabilities of the current candidates, optionally sorting them first. */
    private void softmax(boolean sort) {
        if (sort && !cur.sorted) {
            sortAll();
        }
        int n = cur.size;
        float max = cur.sorted ? cur.logits[0] : cur.maxLogit();
        if (max == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
        }
        float sum = 0f;
        for (int i = 0; i < n; i++) {
            float p = expShift(cur.logits[i], max);
            cur.probs[i] = p;
            sum += p;
        }
        for (int i = 0; i < n; i++) {
            cur.probs[i] /= sum;
        }
    }

    /**
     * {@code exp(logit - max)}; when {@code max} is {@code +inf} (an overflowed logit, for example after diffusion
     * Gumbel noise) this is its limit: the infinite logits share all the mass.
     */
    private static float expShift(float logit, float max) {
        if (max == Float.POSITIVE_INFINITY) {
            return logit == Float.POSITIVE_INFINITY ? 1f : 0f;
        }
        return (float) Math.exp(logit - max);
    }

    // ------------------------------------------------------------------ ordering

    private void sortAll() {
        int n = cur.size;
        long[] k = keys(n);
        for (int i = 0; i < n; i++) {
            k[i] = descendingKey(cur.logits[i], i);
        }
        Arrays.sort(k, 0, n);
        cur.permute(k, n);
        cur.sorted = true;
    }

    /**
     * Moves the {@code k} highest logits, in descending order, to the front (llama.cpp
     * {@code llama_token_data_array_partial_sort_inplace}); the order of the rest is unspecified.
     */
    private void partialSort(int k) {
        int n = cur.size;
        if ((long) k * 4 >= n) {
            sortAll();
            return;
        }
        long[] heap = keys(k);
        int size = 0;
        for (int i = 0; i < n; i++) {
            long key = descendingKey(cur.logits[i], i);
            if (size < k) {
                heap[size] = key;
                siftUp(heap, size++);
            } else if (key < heap[0]) {
                heap[0] = key;
                siftDown(heap, size);
            }
        }
        Arrays.sort(heap, 0, k);
        cur.permute(heap, k);
        cur.sorted = true;
    }

    /** Max-heap on the packed keys: the root is the worst of the kept candidates. */
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
        h[i] = v;
    }

    private long[] keys(int n) {
        if (keys.length < n) {
            keys = new long[n];
        }
        return keys;
    }

    /** Monotone int image of a float: larger floats map to larger ints; {@code -0} and {@code +0} are equal. */
    private static int sortableBits(float f) {
        int bits = Float.floatToIntBits(f + 0.0f);
        return bits ^ ((bits >> 31) & 0x7fffffff);
    }

    /** Key whose ascending order is descending {@code value}, then ascending {@code index}. */
    static long descendingKey(float value, int index) {
        return ((long) ~sortableBits(value) << 32) | (index & 0xffffffffL);
    }

    /** Key whose ascending order is ascending {@code value}, then ascending {@code index}. */
    static long ascendingKey(float value, int index) {
        return ((long) sortableBits(value) << 32) | (index & 0xffffffffL);
    }

    // ------------------------------------------------------------------ candidate set

    /**
     * The candidate set of the last {@link #apply} call: {@link #size()} candidates with their token id, logit
     * after the chain's transformations and probability from the last softmax or draw, plus the selected index.
     * Owned by the sampler and overwritten by its next call.
     */
    public static final class Candidates {

        private int[] ids = new int[0];
        private float[] logits = new float[0];
        private float[] probs = new float[0];
        private int[] tmpIds = new int[0];
        private float[] tmpLogits = new float[0];
        private float[] tmpProbs = new float[0];
        private int size;
        private boolean sorted;
        private int selected = -1;

        Candidates() {
        }

        /** Vocabulary-ordered candidates of a row; NaN logits become {@code -inf}. */
        void load(float[] row) {
            int n = row.length;
            if (ids.length < n) {
                ids = new int[n];
                logits = new float[n];
                probs = new float[n];
                tmpIds = new int[n];
                tmpLogits = new float[n];
                tmpProbs = new float[n];
            }
            for (int i = 0; i < n; i++) {
                float v = row[i];
                ids[i] = i;
                logits[i] = Float.isNaN(v) ? Float.NEGATIVE_INFINITY : v;
                probs[i] = 0f;
            }
            size = n;
            sorted = false;
            selected = -1;
        }

        float maxLogit() {
            float max = logits[0];
            for (int i = 1; i < size; i++) {
                if (logits[i] > max) {
                    max = logits[i];
                }
            }
            return max;
        }

        /** Reorders the first {@code n} candidates to {@code order[j]}'s low 32 bits (source index). */
        void permute(long[] order, int n) {
            for (int j = 0; j < n; j++) {
                int src = (int) order[j];
                tmpIds[j] = ids[src];
                tmpLogits[j] = logits[src];
                tmpProbs[j] = probs[src];
            }
            System.arraycopy(tmpIds, 0, ids, 0, n);
            System.arraycopy(tmpLogits, 0, logits, 0, n);
            System.arraycopy(tmpProbs, 0, probs, 0, n);
        }

        void move(int from, int to) {
            if (from != to) {
                ids[to] = ids[from];
                logits[to] = logits[from];
                probs[to] = probs[from];
            }
        }

        /** Number of candidates left by the chain. */
        public int size() {
            return size;
        }

        /** Token id of candidate {@code i}. */
        public int id(int i) {
            checkIndex(i);
            return ids[i];
        }

        /** Logit of candidate {@code i} after the chain (penalized, scaled by temperature). */
        public float logit(int i) {
            checkIndex(i);
            return logits[i];
        }

        /** Probability of candidate {@code i} from the chain's last softmax (the draw's, for a sampled chain). */
        public float prob(int i) {
            checkIndex(i);
            return probs[i];
        }

        /** Whether the candidates are in descending logit order. */
        public boolean sorted() {
            return sorted;
        }

        /** Index of the selected candidate. */
        public int selected() {
            return selected;
        }

        /** Token id of the selected candidate. */
        public int selectedId() {
            return ids[selected];
        }

        private void checkIndex(int i) {
            if (i < 0 || i >= size) {
                throw new IndexOutOfBoundsException("candidate " + i + " outside [0, " + size + ")");
            }
        }
    }

    // ------------------------------------------------------------------ penalty history

    /**
     * Accepted tokens of the penalty window (llama_sampler_penalties ring buffer and {@code token_count}): a ring of
     * the last {@code window} tokens, or every token when the window is {@link SamplerConfig#PENALTY_WHOLE_CONTEXT};
     * nothing is kept for a window of 0. Distinct tokens with a positive count are listed in {@code active}.
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
            if (token < 0) {
                throw new IllegalArgumentException("accepted token id " + token + " is negative");
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

        void clear() {
            for (int a = 0; a < nActive; a++) {
                counts[active[a]] = 0;
            }
            nActive = 0;
            head = 0;
            stored = 0;
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
