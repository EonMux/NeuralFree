package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.sampling.AdaptivePStage;
import com.eonmux.neuralfree.gguf.engine.sampling.BiasStage;
import com.eonmux.neuralfree.gguf.engine.sampling.DryStage;
import com.eonmux.neuralfree.gguf.engine.sampling.Mt19937;
import com.eonmux.neuralfree.gguf.engine.sampling.InfillStage;
import com.eonmux.neuralfree.gguf.engine.sampling.MinPStage;
import com.eonmux.neuralfree.gguf.engine.sampling.MirostatStage;
import com.eonmux.neuralfree.gguf.engine.sampling.PenaltiesStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.engine.sampling.TopKStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TopNSigmaStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TopPStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TemperatureStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TypicalStage;
import com.eonmux.neuralfree.gguf.engine.sampling.XtcStage;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
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
 *   <li>the configured stages, one per configured kind, applied in the order's sequence;
 *       a disabled stage (top-k 0, top-p 1, min-p 0, typical-p 1, penalties off, DRY with a 0 multiplier, top-n-sigma
 *       below 0, XTC with a probability of 0 or a threshold above 0.5) applies nothing exactly as llama.cpp skips it,
 *       and a {@code min_keep} of 0 (the sampler default) leaves the stage unbounded; llama.cpp's adaptive-p entry is
 *       built last (common/sampling.cpp:387-407), in place of the dist stage;</li>
 *   <li>the fixed dist stage: a seeded categorical draw (llama_sampler_dist_apply), or the argmax when greedy;
 *       a mirostat chain ({@code mirostat} 1 or 2) is instead {@code temperature + mirostat} with neither the rest of
 *       the order nor the dist stage, and an {@code adaptive_p} order entry replaces the dist stage, both as llama.cpp
 *       builds them.</li>
 * </ol>
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
    private Mt19937 rng;

    private Sampler(int[] banned, int[] biasedIds, float[] biases, SamplerConfig config, long seed, boolean greedy,
                    SamplerVocab vocab) {
        this.banned = banned;
        this.penalties = config.penaltiesEnabled();
        this.topK = config.topK();
        this.typicalP = config.typicalP();
        this.greedy = greedy;
        this.seed = seed;
        this.biasStage = new BiasStage(banned, biasedIds, biases);
        this.chain = buildChain(config, seed, vocab);
        this.rng = new Mt19937(seed);
    }

    /** llama.cpp's chain construction (common/sampling.cpp:344-416). */
    private static List<SamplerStage> buildChain(SamplerConfig config, long seed, SamplerVocab vocab) {
        List<SamplerStage> chain = new ArrayList<>();
        if (config.mirostat() == 1 || config.mirostat() == 2) {
            // llama.cpp's mirostat chains are temperature + the mirostat sampler, nothing else
            chain.add(new TemperatureStage(config.temperature()));
            chain.add(new MirostatStage(config.mirostat(), config.mirostatTau(), config.mirostatEta(), seed));
            return chain;
        }
        boolean adaptiveP = false;
        for (StageKind kind : config.order()) {
            if (kind == StageKind.PENALTIES) {
                chain.add(new PenaltiesStage(config.repeatPenalty(), config.presencePenalty(),
                        config.frequencyPenalty(), config.penaltyLastN(), config.penaltiesEnabled()));
            } else if (kind == StageKind.DRY) {
                chain.add(new DryStage(config.dryMultiplier(), config.dryBase(), config.dryAllowedLength(),
                        config.dryPenaltyLastN(), config.drySequenceBreakers(), vocab));
            } else if (kind == StageKind.TOP_N_SIGMA) {
                chain.add(new TopNSigmaStage(config.topNSigma()));
            } else if (kind == StageKind.TOP_K) {
                chain.add(new TopKStage(config.topK(), config.minKeep()));
            } else if (kind == StageKind.TYPICAL_P) {
                chain.add(new TypicalStage(config.typicalP(), config.minKeep()));
            } else if (kind == StageKind.TOP_P) {
                chain.add(new TopPStage(config.topP(), config.minKeep()));
            } else if (kind == StageKind.MIN_P) {
                chain.add(new MinPStage(config.minP(), config.minKeep()));
            } else if (kind == StageKind.XTC) {
                chain.add(new XtcStage(config.xtcProbability(), config.xtcThreshold(), config.minKeep(),
                        stageSeed(config, seed)));
            } else if (kind == StageKind.TEMPERATURE) {
                chain.add(new TemperatureStage(config.temperature(), config.dynatempRange(), config.dynatempExponent()));
            } else if (kind == StageKind.INFILL) {
                chain.add(new InfillStage(vocab));
            } else if (kind == StageKind.ADAPTIVE_P) {
                // llama.cpp keeps the adaptive-p position in the order but builds it last, in place of the dist stage
                adaptiveP = true;
            }
        }
        if (adaptiveP) {
            chain.add(new AdaptivePStage(config.adaptiveTarget(), config.adaptiveDecay(), stageSeed(config, seed)));
        }
        return chain;
    }

    /** llama.cpp's per-sampler {@code get_rng_seed(seed)}: every stage draws its own seed unless the seed is fixed. */
    private static long stageSeed(SamplerConfig config, long seed) {
        return config.seeded() ? seed : randomSeed();
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
        return create(config, bannedTokens, (SamplerVocab) null);
    }

    /**
     * {@link #create(SamplerConfig, int[])} with the model's vocabulary, which the DRY stage tokenizes its sequence
     * breakers over at construction and the infill stage reads its pieces and end-of-generation ids from. A null
     * vocabulary is accepted as long as neither stage is enabled.
     *
     * @throws IllegalArgumentException when a banned id is negative, or a DRY or infill stage is enabled without a
     *                                  vocabulary
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, SamplerVocab vocab) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(bannedTokens, "bannedTokens");
        int[] banned = bannedTokens.clone();
        for (int id : banned) {
            if (id < 0) {
                throw new IllegalArgumentException("banned token id " + id + " is negative");
            }
        }
        long seed = config.seeded() ? config.seed() : randomSeed();
        return new Sampler(banned, new int[0], new float[0], config, seed, config.isGreedy(), vocab);
    }

    /**
     * {@link #create(SamplerConfig, int[])} with the request's {@code logit_bias} map: every entry's token takes
     * its bias added to the logit at the chain's first stage, before penalties, top-k and the rest see the set —
     * llama.cpp's {@code llama_sampler_init_logit_bias} position. A token in both the banned ids and the map takes
     * the {@code -inf} ban; the additive bias never reaches it.
     *
     * @throws IllegalArgumentException for a negative id or a non-finite bias
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, Map<Integer, Float> logitBias) {
        return create(config, bannedTokens, logitBias, null);
    }

    /** {@link #create(SamplerConfig, int[], Map)} with the vocabulary the DRY and infill stages read. */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, Map<Integer, Float> logitBias,
                                 SamplerVocab vocab) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(bannedTokens, "bannedTokens");
        int[] ids = new int[logitBias == null ? 0 : logitBias.size()];
        float[] biases = new float[ids.length];
        int n = 0;
        if (logitBias != null) {
            for (Map.Entry<Integer, Float> e : logitBias.entrySet()) {
                Objects.requireNonNull(e.getKey(), "a logit_bias key is null");
                Objects.requireNonNull(e.getValue(), "the bias of token " + e.getKey() + " is null");
                ids[n] = e.getKey();
                biases[n] = e.getValue();
                n++;
            }
        }
        int[] banned = bannedTokens.clone();
        for (int id : banned) {
            if (id < 0) {
                throw new IllegalArgumentException("banned token id " + id + " is negative");
            }
        }
        long seed = config.seeded() ? config.seed() : randomSeed();
        return new Sampler(banned, ids, biases, config, seed, config.isGreedy(), vocab);
    }

    /**
     * {@link #create(SamplerConfig, int[], SamplerVocab)} with the Java-API custom {@code extra} stages applied, in
     * their list order, immediately before the temperature stage (or directly before the dist stage when the order
     * has no temperature kind), so a custom stage always runs, including under greedy decoding.
     *
     * @throws IllegalArgumentException when {@code extra} or one of its entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, List<SamplerStage> extra) {
        return create(config, bannedTokens, extra, null);
    }

    /**
     * {@link #create(SamplerConfig, int[], Map)} with the Java-API custom {@code extra} stages applied, in their
     * list order, immediately before the temperature stage (or directly before the dist stage when the order has
     * no temperature kind), so a custom stage always runs, including under greedy decoding.
     *
     * @throws IllegalArgumentException when {@code extra} or one of its entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, Map<Integer, Float> logitBias,
                                 List<SamplerStage> extra, SamplerVocab vocab) {
        Objects.requireNonNull(extra, "extra stages");
        Sampler sampler = create(config, bannedTokens, logitBias, vocab);
        insertExtras(sampler, extra);
        return sampler;
    }

    /** Inserts {@code extra} into {@code sampler}'s chain before its temperature stage. */
    private static void insertExtras(Sampler sampler, List<SamplerStage> extra) {
        for (SamplerStage stage : extra) {
            if (stage == null) {
                throw new IllegalArgumentException("extra stage list contains a null stage");
            }
        }
        int insertAt = 0;
        List<SamplerStage> chain = sampler.chain;
        for (int i = 0; i < chain.size(); i++) {
            if (chain.get(i).kind() == StageKind.TEMPERATURE) {
                insertAt = i;
                break;
            }
            insertAt = i + 1;
        }
        chain.addAll(insertAt, extra);
    }

    /**
     * {@link #create(SamplerConfig, int[], SamplerVocab)} with the Java-API custom {@code extra} stages applied, in
     * their list order, immediately before the temperature stage (or directly before the dist stage when the order
     * has no temperature kind), so a custom stage always runs, including under greedy decoding.
     *
     * @throws IllegalArgumentException when {@code extra} or one of its entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, List<SamplerStage> extra,
                                 SamplerVocab vocab) {
        Objects.requireNonNull(extra, "extra stages");
        Sampler sampler = create(config, bannedTokens, vocab);
        insertExtras(sampler, extra);
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
        SamplerConfig config = SamplerConfig.builder().topK(topK).topP(topP).minP(0f).typicalP(1f)
                .temperature(temperature).penaltyLastN(0).seed(s).build();
        return new Sampler(NO_TOKENS, new int[0], new float[0], config, s, false, null);
    }

    /**
     * The vocabulary handle built on a GGUF model's tokenizer: the model handle the DRY and infill stages read through
     * (llama.cpp's llama_vocab of the chain construction).
     */
    public static SamplerVocab vocab(GGUFTokenizer tokenizer) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        return new GgufVocab(tokenizer);
    }

    /** A SamplerVocab on a GGUF model's tokenizer, following llama.cpp's piece semantics per id. */
    private static final class GgufVocab implements SamplerVocab {

        private static final int SPECIAL_ATTRS = TokenizerVocab.ATTR_UNKNOWN | TokenizerVocab.ATTR_CONTROL;

        private final GGUFTokenizer tokenizer;
        private final TokenizerVocab vocab;
        private final SpecialTokens special;
        private final byte[][] pieces;
        private final byte[][] specialPieces;

        GgufVocab(GGUFTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            this.vocab = tokenizer.vocab();
            this.special = tokenizer.specialTokens();
            this.pieces = new byte[vocab.size()][];
            this.specialPieces = new byte[vocab.size()][];
        }

        @Override
        public int vocabSize() {
            return vocab.size();
        }

        @Override
        public boolean isEog(int id) {
            return special.isEog(id);
        }

        @Override
        public int eotToken() {
            return special.eot();
        }

        @Override
        public int eosToken() {
            return special.eos();
        }

        /** token_to_piece(id, lstrip = 0, special = false): UNKNOWN and CONTROL give no bytes, a byte token its byte. */
        @Override
        public byte[] piece(int id) {
            if ((vocab.attr(id) & SPECIAL_ATTRS) != 0) {
                return new byte[0];
            }
            if (vocab.isByte(id)) {
                return new byte[]{(byte) vocab.byteValue(id)};
            }
            byte[] cached = pieces[id];
            if (cached == null) {
                // the visible piece bytes (GGUFTokenizer.decode of one token, plain decoding)
                cached = tokenizer.decode(id).getBytes(StandardCharsets.UTF_8);
                pieces[id] = cached;
            }
            return cached;
        }

        /** The cached special = true piece: CONTROL and UNKNOWN tokens render their raw text bytes. */
        @Override
        public byte[] grammarPiece(int id) {
            if ((vocab.attr(id) & SPECIAL_ATTRS) != 0) {
                byte[] cached = specialPieces[id];
                if (cached == null) {
                    cached = vocab.text(id).getBytes(StandardCharsets.UTF_8);
                    specialPieces[id] = cached;
                }
                return cached;
            }
            return piece(id);
        }

        /** detokenize({id}, special = true): special tokens render their raw text, visible tokens their piece. */
        @Override
        public String pieceText(int id) {
            if ((vocab.attr(id) & SPECIAL_ATTRS) != 0) {
                return vocab.text(id);
            }
            if (vocab.isByte(id)) {
                return new String(piece(id), StandardCharsets.ISO_8859_1);
            }
            return tokenizer.decode(id);
        }

        @Override
        public int[] tokenize(String text) {
            return tokenizer.encode(text, false, false);
        }
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
     * True when the chain reduces to the argmax of the raw logits: no banned tokens, no penalties, no enabled llama.cpp
     * stage that edits or trims the set (dry, top-n-sigma, xtc, infill, a dynamic temperature range), no mirostat or
     * adaptive-p terminal selector, and either greedy without typical-p or a top-k of 1 (llama.cpp common/sampling.cpp
     * also replaces the draw by greedy for {@code top_k == 1}). A forward backend's on-device argmax then selects the
     * same token.
     */
    public boolean argmaxOnly() {
        if (banned.length != 0 || penalties || !(greedy && typicalP >= 1f) && topK != 1) {
            return false;
        }
        for (SamplerStage stage : chain) {
            StageKind kind = stage.kind();
            if (kind == null || kind == StageKind.PENALTIES || kind == StageKind.TOP_K || kind == StageKind.TYPICAL_P
                    || kind == StageKind.TOP_P || kind == StageKind.MIN_P || kind == StageKind.TEMPERATURE) {
                // the argmax cone (they preserve or are ignored by the argmax; penalties were checked)
                continue;
            }
            // dry, top_n_sigma, xtc, infill and the terminal selectors edit the set - but a dry or top-n-sigma or
            // xtc stage whose llama.cpp disable gate is hit never does, keeping the argmin chain argmax.
            if (stage.modifies()) {
                return false;
            }
        }
        return true;
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
        rng = new Mt19937(seed);
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
     * <p>A chain built with mirostat or with {@code adaptive_p} in the order ends in its terminal selector, which
     * selects the token itself, as llama.cpp's chain construction replaces the dist stage (common/sampling.cpp:344-416);
     * the fixed draw only runs when no terminal selector applied.
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
        boolean terminal = false;
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
            if (kind == StageKind.MIROSTAT || kind == StageKind.ADAPTIVE_P) {
                terminal = true;
                break;
            }
        }
        if (terminal) {
            // the terminal selector set the selected candidate already
        } else if (greedy) {
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
     * Softmax and a single categorical draw in the current order ({@code llama_sampler_dist_apply} through
     * libstdc++'s {@code std::discrete_distribution}): the probabilities renormalized in double, the first
     * candidate whose cumulative reaches the canonical double draw. The chain's own {@link Mt19937} draws once
     * per call, also for a single candidate (llama.cpp keeps the RNG aligned with backend sampling).
     */
    private void drawDist() {
        int n = cur.size;
        if (n == 1) {
            if (cur.logits[0] == Float.NEGATIVE_INFINITY) {
                throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
            }
            cur.probs[0] = 1f;
            cur.selected = rng.distDraw(new float[]{1f});
            return;
        }
        float max = cur.sorted ? cur.logits[0] : cur.maxLogit();
        if (max == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
        }
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double e = expShift(cur.logits[i], max);
            cur.probs[i] = (float) e;
            sum += e;
        }
        for (int i = 0; i < n; i++) {
            cur.probs[i] = (float) (cur.probs[i] / sum);
        }
        float[] probs = new float[n];
        for (int i = 0; i < n; i++) {
            probs[i] = cur.prob(i);
        }
        cur.selected = rng.distDraw(probs);
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
         * llama_sampler_softmax_impl(cur_p, true): sorts into descending logit order when needed, then softmaxes over
         * the current candidates with probability 0 never rewritten.
         *
         * @throws IllegalStateException when no candidate has a finite logit
         */
        public void softmaxSorted() {
            if (!sorted) {
                sortDescending();
            }
            softmax();
        }

        /** llama_sampler_softmax_impl with no sort flag: the softmax over the current order. */
        public void softmaxUnsorted() {
            softmax();
        }

        /** Marks the candidate the sampler chains picked (llama.cpp's {@code cur_p->selected}). */
        public void select(int index) {
            checkIndex(index);
            selected = index;
        }

        /** Overwrites the probability of candidate {@code i}. */
        public void setProb(int i, float prob) {
            checkIndex(i);
            probs[i] = prob;
        }

        /**
         * llama.cpp's {@code cur_p->data[pos_last] += ptr} advance: removes the first {@code k} candidates; the rest
         * keep their order, so a sorted set stays sorted.
         */
        public void removeFirst(int k) {
            if (k < 0 || k > size) {
                throw new IndexOutOfBoundsException("remove first " + k + " outside [0, " + size + "]");
            }
            for (int i = k; i < size; i++) {
                move(i, i - k);
            }
            size -= k;
        }

        /** Replaces the candidate at {@code i} by token {@code id} at {@code logit} (infill's EOT fallback). */
        public void replaceCandidate(int i, int id, float logit) {
            checkIndex(i);
            ids[i] = id;
            logits[i] = logit;
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
            h[i] = v;
        }
    }
}
