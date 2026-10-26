package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.BiasStage;
import com.eonmux.neuralfree.gguf.engine.sampling.MinPStage;
import com.eonmux.neuralfree.gguf.engine.sampling.PenaltiesStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.engine.sampling.TopKStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TopPStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TemperatureStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TypicalStage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Host token sampler running llama.cpp's sampler chain over one logits row (llama.cpp src/llama-sampler.cpp), as a
 * stage chain with llama.cpp's configurable order (the {@code common_params_sampling::samplers} list,
 * {@link StageKind}, currently carried by {@link SamplerConfig#order()}):
 * <ol>
 *   <li>the fixed bias stage: banned tokens ({@code tokenizer.ggml.suppress_tokens}, end-of-generation tokens under
 *       ignore-eos) get {@code -inf} (llama_sampler_logit_bias_apply);</li>
 *   <li>the configured stages, one per configured kind that has a stage class, applied in the order's sequence;
 *       a disabled stage (top-k 0, top-p 1, min-p 0, typical-p 1, penalties off) applies nothing exactly as
 *       llama.cpp skips it, and a {@code min_keep} of 0 (the sampler default) leaves the stage unbounded;</li>
 *   <li>the fixed dist stage: a seeded categorical draw (llama_sampler_dist_apply), or the argmax when greedy
 *       (llama_sampler_greedy_apply).</li>
 * </ol>
 * With the stages SA-2 adds (dry, top-n-sigma, xtc) building nothing, the default chain is
 * {@code bias → penalties → top-k → typical-p → top-p → min-p → temperature → dist}.
 *
 * <p>Greedy decoding (an explicit greedy request or a temperature {@code <= 0}) runs the chain without the
 * temperature stage and replaces the draw by the argmax; because top-k, top-p and min-p keep the highest logit
 * on top, they are additionally skipped when locally typical sampling is inactive, as llama.cpp's greedy sampler
 * only needs the argmax of such a set.
 *
 * <p>Candidates keep llama.cpp's {@code llama_token_data_array} semantics: the stages operate on the current
 * candidate set in its current order, and a {@code sorted} flag records whether it is in descending logit order.
 * Sorting is deterministic: equal logits keep their current relative order, so a vocabulary-ordered set breaks
 * ties by the lower token id.
 *
 * <p>A sampler is a mutable runtime object owned by one generation: it holds the chain stages (with the penalty
 * history), the RNG and reusable candidate buffers. Not thread-safe.
 */
public final class Sampler {

    private static final int[] NO_TOKENS = new int[0];

    private final int[] banned;
    private final boolean penalties;
    private final int topK;
    private final float typicalP;
    private final boolean greedy;
    private final long seed;
    private final SamplerStage biasStage;
    private final List<SamplerStage> chain;
    private final Candidates cur = new Candidates();
    private SplittableRandom rng;

    private Sampler(int[] banned, float repeatPenalty, float presencePenalty, float frequencyPenalty, int penaltyLastN,
                    boolean penalties, int topK, float typicalP, float topP, float minP, float temperature,
                    boolean greedy, long seed, List<StageKind> order, int minKeep) {
        this.banned = banned;
        this.penalties = penalties;
        this.topK = topK;
        this.typicalP = typicalP;
        this.greedy = greedy;
        this.seed = seed;
        this.biasStage = new BiasStage(banned);
        this.chain = buildChain(order, repeatPenalty, presencePenalty, frequencyPenalty, penaltyLastN, penalties,
                topK, typicalP, topP, minP, temperature, minKeep);
        this.rng = new SplittableRandom(seed);
    }

    /** One built-in stage per configured kind that has a stage class. */
    private static List<SamplerStage> buildChain(List<StageKind> order, float repeatPenalty, float presencePenalty,
                                                 float frequencyPenalty, int penaltyLastN, boolean penalties, int topK,
                                                 float typicalP, float topP, float minP, float temperature,
                                                 int minKeep) {
        List<SamplerStage> chain = new ArrayList<>();
        for (StageKind kind : order) {
            if (kind == StageKind.PENALTIES) {
                chain.add(new PenaltiesStage(repeatPenalty, presencePenalty, frequencyPenalty, penaltyLastN, penalties));
            } else if (kind == StageKind.TOP_K) {
                chain.add(new TopKStage(topK, minKeep));
            } else if (kind == StageKind.TYPICAL_P) {
                chain.add(new TypicalStage(typicalP, minKeep));
            } else if (kind == StageKind.TOP_P) {
                chain.add(new TopPStage(topP, minKeep));
            } else if (kind == StageKind.MIN_P) {
                chain.add(new MinPStage(minP, minKeep));
            } else if (kind == StageKind.TEMPERATURE) {
                chain.add(new TemperatureStage(temperature));
            }
            // dry, top_n_sigma and xtc build nothing until SA-2 gives their stage classes.
        }
        return chain;
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
                config.minP(), config.temperature(), config.isGreedy(), seed, config.order(), config.minKeep());
    }

    /**
     * The generation chain for {@code config} with the Java-API custom {@code extra} stages applied, in their list
     * order, immediately before the temperature stage (or directly before the dist stage when the order has no
     * temperature kind), so a custom stage always runs, including under greedy decoding.
     *
     * @throws IllegalArgumentException when {@code extra} or one of its entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, List<SamplerStage> extra) {
        Objects.requireNonNull(extra, "extra stages");
        List<SamplerStage> stages = new ArrayList<>(extra);
        if (stages.contains(null)) {
            throw new IllegalArgumentException("extra stage list contains a null stage");
        }
        Sampler sampler = create(config, bannedTokens);
        int insertAt = 0;
        for (int i = 0; i < sampler.chain.size(); i++) {
            if (sampler.chain.get(i).kind() == StageKind.TEMPERATURE) {
                insertAt = i;
                break;
            }
            insertAt = i + 1;
        }
        sampler.chain.addAll(insertAt, stages);
        return sampler;
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
        return new Sampler(NO_TOKENS, 1f, 0f, 0f, 0, false, topK, 1f, topP, 0f, temperature, false, s,
                StageKind.DEFAULT_ORDER, 0);
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

    /** Records an accepted token with the chain's history stages (prompt tokens included, as llama.cpp does). */
    public void accept(int token) {
        for (SamplerStage stage : chain) {
            stage.accept(token);
        }
    }

    /** {@link #accept(int)} for {@code tokens[off .. off + n)}. */
    public void accept(int[] tokens, int off, int n) {
        Objects.requireNonNull(tokens, "tokens");
        if (off < 0 || n < 0 || off + n > tokens.length) {
            throw new IndexOutOfBoundsException("token range [" + off + ", " + (off + n) + ") outside " + tokens.length);
        }
        for (int i = off; i < off + n; i++) {
            accept(tokens[i]);
        }
    }

    /** Forgets the chain's histories and restarts the RNG from {@link #seed()}. */
    public void reset() {
        for (SamplerStage stage : chain) {
            stage.reset();
        }
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
        biasStage.apply(cur);
        for (SamplerStage stage : chain) {
            StageKind kind = stage.kind();
            if (greedy) {
                if (kind == StageKind.TEMPERATURE) {
                    continue;
                }
                if (typicalP >= 1f
                        && (kind == StageKind.TOP_K || kind == StageKind.TOP_P || kind == StageKind.MIN_P)) {
                    continue;
                }
            }
            stage.apply(cur);
        }
        if (greedy) {
            selectArgmax();
        } else {
            drawDist();
        }
        return cur;
    }

    // ------------------------------------------------------------------ fixed chain end

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

    /**
     * Key whose ascending order is descending {@code logit}, then ascending {@code index}; the same image the
     * chain's stages sort by ({@link Candidates#descendingKey}).
     */
    static long descendingKey(float logit, int index) {
        return Candidates.descendingKey(logit, index);
    }

    // ------------------------------------------------------------------ candidate set

    /**
     * The candidate set of the last {@link #apply} call: {@link #size()} candidates with their token id, logit
     * after the chain's transformations and probability from the last softmax or draw, plus the selected index.
     * Owned by the sampler and overwritten by its next call.
     *
     * <p>Alongside the reads below, the set also carries the mutation surface its chain's
     * {@link com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage stages} apply through (logit edits, sorts,
     * permutes and truncations with llama.cpp's {@code sorted} bookkeeping). Those methods are for the sampler and
     * its stages: a set belongs to its sampler, and a caller outside the chain must not mutate it, or the
     * sampler's view of its own last apply breaks.
     */
    public static final class Candidates {

        private int[] ids = new int[0];
        private float[] logits = new float[0];
        private float[] probs = new float[0];
        private int[] tmpIds = new int[0];
        private float[] tmpLogits = new float[0];
        private float[] tmpProbs = new float[0];
        private long[] sortKeys = new long[0];
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
                sortKeys = new long[n];
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

        /** Largest logit of the current candidates (for stages that need the max over an unsorted set). */
        public float maxLogit() {
            float max = logits[0];
            for (int i = 1; i < size; i++) {
                if (logits[i] > max) {
                    max = logits[i];
                }
            }
            return max;
        }

        /** Reorders the first {@code n} candidates to {@code order[j]}'s low 32 bits (source index). */
        public void permute(long[] order, int n) {
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

        /** Moves candidate {@code from} to slot {@code to} (llama.cpp's compaction while filtering). */
        public void move(int from, int to) {
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

        // ---------------------------------------------------------------- stage surface

        /**
         * Key whose ascending order is descending {@code logit}, then ascending {@code index} (llama.cpp sorts on
         * the logit; a vocabulary-first tie-break keeps sorting deterministic). {@code -0} and {@code +0} are equal.
         */
        public static long descendingKey(float logit, int index) {
            return ((long) ~sortableBits(logit) << 32) | (index & 0xffffffffL);
        }

        /** Key whose ascending order is ascending {@code value}, then ascending {@code index}. */
        public static long ascendingKey(float value, int index) {
            return ((long) sortableBits(value) << 32) | (index & 0xffffffffL);
        }

        /** Monotone int image of a float: larger floats map to larger ints; {@code -0} and {@code +0} are equal. */
        private static int sortableBits(float f) {
            int bits = Float.floatToIntBits(f + 0.0f);
            return bits ^ ((bits >> 31) & 0x7fffffff);
        }

        /** Overwrites the logit of candidate {@code i}; the {@code sorted} flag is not touched (a stage decides). */
        public void setLogit(int i, float logit) {
            checkIndex(i);
            logits[i] = logit;
        }

        /** Marks the set as no longer being in descending logit order. */
        public void setSorted(boolean sorted) {
            this.sorted = sorted;
        }

        /**
         * Sorts all candidates in descending logit order, ties by index
         * (llama.cpp {@code llama_token_data_array_partial_sort_inplace(cur_p, cur_p->size)}).
         */
        public void sortDescending() {
            long[] k = sortKeys(size);
            for (int i = 0; i < size; i++) {
                k[i] = descendingKey(logits[i], i);
            }
            Arrays.sort(k, 0, size);
            permute(k, size);
            sorted = true;
        }

        /**
         * Moves the {@code k} highest logits, in descending order, to the front, as a full sort of the primary part
         * (llama.cpp {@code llama_token_data_array_partial_sort_inplace}); the order of the rest is unspecified. The
         * full sort when {@code k} is close to the set size, as a partial sort costs no less then.
         */
        public void partialSortDescending(int k) {
            int n = size;
            if ((long) k * 4 >= n) {
                sortDescending();
                return;
            }
            long[] heap = new long[k];
            int heapSize = 0;
            for (int i = 0; i < n; i++) {
                long key = descendingKey(logits[i], i);
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

        /** Keeps the first {@code newSize} candidates; candidates keep their order, so a sorted set stays sorted. */
        public void truncate(int newSize) {
            if (newSize < 0 || newSize > size) {
                throw new IndexOutOfBoundsException("truncate to " + newSize + " outside [0, " + size + "]");
            }
            size = newSize;
        }

        /**
         * llama_sampler_softmax_impl: probabilities of the current candidates from their logits, with
         * probability 0 never rewritten. Callers sort first (llama.cpp passes a sort flag); the max must be finite.
         *
         * @throws IllegalStateException when no candidate has a finite logit
         */
        public void softmax() {
            int n = size;
            float max = sorted ? logits[0] : maxLogit();
            if (max == Float.NEGATIVE_INFINITY) {
                throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
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

        private long[] sortKeys(int n) {
            if (sortKeys.length < n) {
                sortKeys = new long[n];
            }
            return sortKeys;
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
        }
    }
}
