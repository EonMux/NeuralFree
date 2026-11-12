package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.sampling.BiasStage;
import com.eonmux.neuralfree.gguf.engine.sampling.Mt19937;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.engine.sampling.TokenGate;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkState;
import com.eonmux.neuralfree.gguf.engine.watermark.Watermarker;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Host token sampler running llama.cpp's sampler chain over one logits row (llama.cpp src/llama-sampler.cpp), as a
 * stage chain with llama.cpp's configurable order (the {@code common_params_sampling::samplers} list,
 * {@link StageKind}, currently carried by {@link SamplerConfig#order()}):
 * <ol>
 *   <li>the fixed bias stage: banned tokens ({@code tokenizer.ggml.suppress_tokens}, end-of-generation tokens under
 *       ignore-eos) get {@code -inf} (llama_sampler_logit_bias_apply);</li>
 *   <li>the constraint head: the chain's constraining stages (the request's grammars) and masking stages
 *       ({@code allowed_token_ids}, bad words, the min-tokens ban) mask the set before any configured stage,
 *       in chain order — the grammars first, the yielding min-tokens ban behind the other masks — so the
 *       truncating stages operate on the admitted set as llama.cpp's grammar-first resample (common/sampling.cpp
 *       applies the grammar before the chain) and vLLM's pre-sampling logits processors (allowed_token_ids and
 *       bad words before any sampling stage) keep them; an empty admitted set refuses with {@link
 *       ConstraintDeadEndException};</li>
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
 * <p>A chain built with a watermark ({@link SamplerConfig#watermark}) replaces the categorical draw by the keyed
 * Gumbel-max selection of {@link Watermarker} over the chain's final candidates, except under greedy decoding,
 * which has no distribution to shift; the reported probabilities stay the ordinary softmax. The chain accepts a
 * prompt through {@link #acceptPrompt}, which marks where the generated part starts for the repeated-context
 * search.
 *
 * <p>Candidates keep llama.cpp's {@code llama_token_data_array} semantics: the stages operate on the current
 * candidate set in its current order, and a {@code sorted} flag records whether it is in descending logit order.
 * Sorting is deterministic: equal logits keep their current relative order, so a vocabulary-ordered set breaks
 * ties by the lower token id.
 *
 * <p>A sampler is a mutable runtime object owned by one generation: it holds the chain stages (with the penalty
 * history), the RNG and reusable candidate buffers. Its stages' state can be saved and restored ({@link #saveState})
 * and a token sequence checked against its grammar stages ({@link #admittedPrefix}), which speculative verification
 * uses to score draft rows ahead of their confirmation. Not thread-safe.
 */
public final class Sampler {

    private static final int[] NO_TOKENS = new int[0];

    /** The {@link #vocab} handle of each tokenizer while an engine holds it; the values do not keep their keys. */
    private static final Map<GGUFTokenizer, WeakReference<SamplerVocab>> VOCABS = new WeakHashMap<>();

    /** Most distinct penalized tokens one device prefilter row carries; a wider window keeps the host path. */
    public static final int MAX_PREFILTER_PENALTY_TOKENS = 16384;
    /** Longest DRY window one device prefilter row carries. */
    public static final int MAX_PREFILTER_DRY_WINDOW = 256;
    /** Most restart sequences one device prefilter row carries. */
    public static final int MAX_PREFILTER_DRY_BREAKERS = 8192;
    /** Most restart-sequence tail tokens one device prefilter row carries. */
    public static final int MAX_PREFILTER_DRY_TAIL_IDS = 16384;

    private final int[] banned;
    private final boolean penalties;
    private final int minKeep;
    private final float dryMultiplier;
    private final int topK;
    private final float topP;
    private final float minP;
    private final float typicalP;
    private final float topNSigma;
    private final float xtcProbability;
    private final float xtcThreshold;
    private final float dynatempRange;
    private final boolean greedy;
    private final long seed;
    /** Candidates the fixed bias stage masks: the banned and the logit-biased tokens. */
    private final int coneExtra;
    private final SamplerStage biasStage;
    private final List<SamplerStage> chain;
    /**
     * The chain's constraint stages ({@link SamplerStage#constrains}, {@link SamplerStage#masks}), the mask the
     * chain applies first.
     */
    private List<SamplerStage> constraints;
    /**
     * Whether the caller handed the chain stage instances (Java-API extra stages or logits processors): a caller may
     * share one instance between requests, so such a chain never samples beside another
     * ({@link #samplesIndependently}), whatever the stages' classes.
     */
    private boolean callerStages;
    /**
     * The chain's device prefilter plan: the penalties and DRY stages a device readout reproduces before its
     * top-k, the cone width it needs and the row it uploads ({@link LogitPrefilter}).
     */
    private final SamplerDeviceFilter deviceFilter;
    private final Candidates cur = new Candidates();
    private Mt19937 rng;
    /** The watermark that replaces the categorical draw, or null for an ordinary chain. */
    private final Watermarker watermarker;
    /** The watermark's PRF context and repeated-context history, or null without a watermark. */
    private final WatermarkState watermarkState;
    /** The request's watermark configuration, or null. */
    private final WatermarkConfig watermarkConfig;
    /** The chain's fixed end: the argmax, the categorical draw or the watermarked selection. */
    private final SamplerChainEnd chainEnd;

    private Sampler(int[] banned, int[] biasedIds, float[] biases, SamplerConfig config, long seed, boolean greedy,
                    SamplerVocab vocab, boolean speculativeTarget) {
        this.banned = banned;
        this.penalties = config.penaltiesEnabled();
        this.minKeep = config.minKeep();
        this.dryMultiplier = config.dryMultiplier();
        this.topK = config.topK();
        this.topP = config.topP();
        this.minP = config.minP();
        this.typicalP = config.typicalP();
        this.topNSigma = config.topNSigma();
        this.xtcProbability = config.xtcProbability();
        this.xtcThreshold = config.xtcThreshold();
        this.dynatempRange = config.dynatempRange();
        this.greedy = greedy;
        this.seed = seed;
        this.coneExtra = banned.length + biasedIds.length;
        boolean positiveBias = false;
        for (float bias : biases) {
            positiveBias |= bias > 0f;
        }
        this.biasStage = new BiasStage(banned, biasedIds, biases);
        this.chain = SamplerChains.build(config, seed, vocab);
        this.constraints = SamplerChains.constraintsOf(chain);
        this.deviceFilter = new SamplerDeviceFilter(this, chain, biasStage, banned, coneExtra, positiveBias, config);
        this.deviceFilter.refresh();
        this.rng = new Mt19937(seed);
        WatermarkConfig watermark = config.watermark();
        if (watermark != null && (config.mirostat() != SamplerConfig.MIROSTAT_OFF
                || config.order().contains(StageKind.ADAPTIVE_P))) {
            // these stages select the token themselves, so the keyed draw would never run
            throw new IllegalArgumentException("watermarking: a mirostat or adaptive_p chain selects tokens itself,"
                    + " so the watermark's keyed draw would not apply; disable the watermark or the terminal"
                    + " selector");
        }
        this.watermarker = watermark == null ? null
                : speculativeTarget ? watermark.targetWatermarker() : Watermarker.create(watermark);
        this.watermarkState = watermark == null ? null : new WatermarkState(watermark);
        this.watermarkConfig = watermark;
        this.chainEnd = new SamplerChainEnd(this.watermarker, this.watermarkState);
    }

    /** Whether the chain carries constraint stages (a grammar, the request's masks), which {@link #coneWidth}
     * refuses: the head masks over the whole row, so a device cone carries no chain that has one. */
    public boolean hasConstraints() {
        return !constraints.isEmpty();
    }

    /**
     * Whether this chain may sample on a worker thread while other sequences' chains sample beside it: the sampler
     * built every stage itself, and each is an instance of one of llama.cpp's built-in stage classes or the fixed
     * logit bias ({@link SamplerChains#INDEPENDENT_STAGES}), which hold only this sampler's own state. A constraint stage (a
     * grammar, the request's masks), a reasoning budget or an infill stage reaches shared vocabulary tables, and a
     * stage the caller handed in (an extra stage or a logits processor, even an instance of a built-in class) may be
     * user code or an instance shared with another request's chain, so such a chain samples on the engine's own
     * thread. The check goes by the stages' final classes, not by {@link SamplerStage#kind()}, which a custom stage
     * may override to report a built-in kind.
     */
    public boolean samplesIndependently() {
        if (callerStages || !constraints.isEmpty()) {
            return false;
        }
        for (SamplerStage stage : chain) {
            if (!SamplerChains.INDEPENDENT_STAGES.contains(stage.getClass())) {
                return false;
            }
        }
        return true;
    }

    /** Index of the chain's first top-k stage, or -1. */
    private int topKIndex() {
        for (int i = 0; i < chain.size(); i++) {
            if (chain.get(i).kind() == StageKind.TOP_K) {
                return i;
            }
        }
        return -1;
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
        return new Sampler(banned, new int[0], new float[0], config, seed, config.isGreedy(), vocab, false);
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
        return create(config, bannedTokens, logitBias, vocab, false);
    }

    /**
     * The chain of a speculative target ({@link SpeculativeDecoder}, the batch engine's speculating sequences): as
     * {@link #create(SamplerConfig, int[], Map, SamplerVocab)}, except that a configured watermark uses its target
     * role ({@link WatermarkConfig#targetWatermarker()}) — key B of a dual-key configuration — because the target
     * never routes: the draft role samples the proposals.
     */
    public static Sampler createForSpeculativeTarget(SamplerConfig config, int[] bannedTokens,
                                                     Map<Integer, Float> logitBias, SamplerVocab vocab) {
        return create(config, bannedTokens, logitBias, vocab, true);
    }

    private static Sampler create(SamplerConfig config, int[] bannedTokens, Map<Integer, Float> logitBias,
                                  SamplerVocab vocab, boolean speculativeTarget) {
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
        return new Sampler(banned, ids, biases, config, seed, config.isGreedy(), vocab, speculativeTarget);
    }

    /**
     * {@link #create(SamplerConfig, int[], SamplerVocab)} with the Java-API custom {@code extra} stages applied,
     * in their list order, immediately before the temperature stage (or directly before the dist stage when the
     * order has no temperature kind), so a custom stage always runs, including under greedy decoding. A stage
     * that constrains ({@link SamplerStage#constrains}, a grammar) or masks ({@link SamplerStage#masks},
     * {@code allowed_token_ids}, bad words, the min-tokens ban) applies from the chain's constraint head
     * instead, before every configured stage ({@link #apply}).
     *
     * @throws NullPointerException     when {@code extra} is null
     * @throws IllegalArgumentException when one of the {@code extra} entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, List<SamplerStage> extra) {
        Objects.requireNonNull(extra, "extra stages");
        Sampler sampler = create(config, bannedTokens, (SamplerVocab) null);
        insertExtras(sampler, extra);
        return sampler;
    }

    /**
     * {@link #create(SamplerConfig, int[], Map)} with the Java-API custom {@code extra} stages applied, in their
     * list order, immediately before the temperature stage (or directly before the dist stage when the order has
     * no temperature kind), so a custom stage always runs, including under greedy decoding; a constraining one
     * ({@link SamplerStage#constrains}) or a masking one ({@link SamplerStage#masks}) applies from the chain's
     * constraint head instead ({@link #apply}).
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
        sampler.callerStages |= !extra.isEmpty();
        sampler.constraints = SamplerChains.constraintsOf(chain);
        sampler.deviceFilter.refresh();
    }

    /**
     * Inserts {@code processors} at the head of {@code sampler}'s configured chain, after the constraint head's
     * grammar and request masks and before penalties, top-k and the rest: vLLM applies custom logits processors to
     * the logits after {@code allowed_token_ids} and bad words and before its sampling stages, so a processor sees
     * (and may change) the whole candidate set. A processor that constrains or masks joins the constraint head after
     * the request's own stages, so the grammar stays the first mask.
     */
    private static void insertLogitsProcessors(Sampler sampler, List<SamplerStage> processors) {
        for (SamplerStage stage : processors) {
            if (stage == null) {
                throw new IllegalArgumentException("logits processor list contains a null stage");
            }
        }
        if (processors.isEmpty()) {
            return;
        }
        List<SamplerStage> chain = sampler.chain;
        chain.addAll(0, processors);
        sampler.callerStages = true;
        List<SamplerStage> constraints = new ArrayList<>(sampler.constraints);
        for (SamplerStage stage : processors) {
            if (stage.constrains() || stage.masks()) {
                constraints.add(stage);
            }
        }
        sampler.constraints = constraints.isEmpty() ? List.of() : Collections.unmodifiableList(constraints);
        sampler.deviceFilter.refresh();
    }

    /**
     * {@link #create(SamplerConfig, int[], SamplerVocab)} with the Java-API custom {@code extra} stages applied, in
     * their list order, immediately before the temperature stage (or directly before the dist stage when the order
     * has no temperature kind), so a custom stage always runs, including under greedy decoding; a constraining one
     * ({@link SamplerStage#constrains}) or a masking one ({@link SamplerStage#masks}) applies from the chain's
     * constraint head instead ({@link #apply}). The vocabulary is required here: request stages read it;
     * {@link #create(SamplerConfig, int[], List)} builds without one.
     *
     * @throws NullPointerException     when {@code extra} or {@code vocab} is null
     * @throws IllegalArgumentException when one of the {@code extra} entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, List<SamplerStage> extra,
                                 SamplerVocab vocab) {
        Objects.requireNonNull(extra, "extra stages");
        Objects.requireNonNull(vocab, "vocab");
        Sampler sampler = create(config, bannedTokens, vocab);
        insertExtras(sampler, extra);
        return sampler;
    }

    /**
     * {@link #create(SamplerConfig, int[], List)} with the custom logits processors {@code processors} applied at
     * the head of the configured chain, so they run after the constraint head's grammar and request masks and
     * before penalties, top-k, top-p, min-p and the rest; a processor that constrains ({@link
     * SamplerStage#constrains}) or masks ({@link SamplerStage#masks}) joins the constraint head after the request's
     * stages. They run under greedy decoding too, and any stage whose {@link SamplerStage#modifies} is true keeps
     * the device argmax and cone paths from skipping them.
     *
     * @throws NullPointerException     when {@code extra} or {@code processors} is null
     * @throws IllegalArgumentException when one of the entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, List<SamplerStage> extra,
                                 List<SamplerStage> processors) {
        Objects.requireNonNull(extra, "extra stages");
        Objects.requireNonNull(processors, "logits processors");
        Sampler sampler = create(config, bannedTokens, extra);
        insertLogitsProcessors(sampler, processors);
        return sampler;
    }

    /**
     * {@link #create(SamplerConfig, int[], Map, List, SamplerVocab)} with the custom logits processors
     * {@code processors} at the chain head ({@link #create(SamplerConfig, int[], List, List)}).
     *
     * @throws NullPointerException     when {@code extra}, {@code processors} or {@code vocab} is null
     * @throws IllegalArgumentException when one of the entries is null
     */
    public static Sampler create(SamplerConfig config, int[] bannedTokens, Map<Integer, Float> logitBias,
                                 List<SamplerStage> extra, List<SamplerStage> processors, SamplerVocab vocab) {
        Objects.requireNonNull(extra, "extra stages");
        Objects.requireNonNull(processors, "logits processors");
        Sampler sampler = create(config, bannedTokens, logitBias, extra, vocab);
        insertLogitsProcessors(sampler, processors);
        return sampler;
    }

    /** {@link #createForSpeculativeTarget(SamplerConfig, int[], Map, SamplerVocab)} with the {@code extra} stages. */
    public static Sampler createForSpeculativeTarget(SamplerConfig config, int[] bannedTokens,
                                                     Map<Integer, Float> logitBias, List<SamplerStage> extra,
                                                     SamplerVocab vocab) {
        Objects.requireNonNull(extra, "extra stages");
        Objects.requireNonNull(vocab, "vocab");
        Sampler sampler = create(config, bannedTokens, logitBias, vocab, true);
        insertExtras(sampler, extra);
        return sampler;
    }

    /** {@link #createForSpeculativeTarget(SamplerConfig, int[], Map, List, SamplerVocab)} with the processors. */
    public static Sampler createForSpeculativeTarget(SamplerConfig config, int[] bannedTokens,
                                                     Map<Integer, Float> logitBias, List<SamplerStage> extra,
                                                     List<SamplerStage> processors, SamplerVocab vocab) {
        Objects.requireNonNull(extra, "extra stages");
        Objects.requireNonNull(processors, "logits processors");
        Sampler sampler = createForSpeculativeTarget(config, bannedTokens, logitBias, extra, vocab);
        insertLogitsProcessors(sampler, processors);
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
        return new Sampler(NO_TOKENS, new int[0], new float[0], config, s, false, null, false);
    }

    /**
     * The vocabulary handle built on a GGUF model's tokenizer: the model handle the DRY and infill stages read through
     * (llama.cpp's llama_vocab of the chain construction). Every engine of one tokenizer shares one handle, so its
     * piece caches and its grammar piece table ({@code GrammarPieces}) are built once per model; the handle lives as
     * long as an engine holds it.
     */
    public static SamplerVocab vocab(GGUFTokenizer tokenizer) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        synchronized (VOCABS) {
            WeakReference<SamplerVocab> ref = VOCABS.get(tokenizer);
            SamplerVocab vocab = ref != null ? ref.get() : null;
            if (vocab == null) {
                vocab = new GgufVocab(tokenizer);
                VOCABS.put(tokenizer, new WeakReference<>(vocab));
            }
            return vocab;
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

    /** The request's watermark configuration, or null for an ordinary chain. */
    public WatermarkConfig watermarkConfig() {
        return watermarkConfig;
    }

    /** Whether the chain selects its token with the watermark's keyed draw. */
    public boolean watermarked() {
        return watermarker != null;
    }

    /** The PRF context of the chain's next draw, oldest first ({@code -1} where the history is shorter). */
    public int[] watermarkContext() {
        if (watermarkState == null) {
            throw new IllegalStateException("the chain carries no watermark");
        }
        return watermarkState.context();
    }

    /** Whether the context of the chain's next draw already occurred in the searched history. */
    public boolean watermarkRepeatedContext() {
        return watermarkState != null && watermarkState.repeatedContext();
    }

    /**
     * The watermark's keyed selection over an explicit candidate set — token ids with their scores — under the
     * chain's current context. Speculative rejection sampling draws a rejected position's replacement from the
     * residual distribution through this method, so the replacement carries the target key like the chain's own
     * draws (vLLM's resample kernel).
     *
     * @throws IllegalStateException when the chain carries no watermark
     * @throws IllegalArgumentException when the arrays disagree with {@code count}
     */
    public int watermarkedDraw(int[] tokenIds, float[] scores, int count) {
        if (watermarker == null) {
            throw new IllegalStateException("the chain carries no watermark");
        }
        return watermarker.sample(tokenIds, scores, count, watermarkState.context(),
                rng::nextCanonicalDouble);
    }

    /**
     * True when the chain reduces to the argmax of the raw logits: no banned tokens, no {@code logit_bias} entry,
     * no penalties, no enabled llama.cpp stage that edits or trims the set (dry, top-n-sigma, xtc, infill, a dynamic
     * temperature range), no mirostat or adaptive-p terminal selector, no Java-API custom stage whose
     * {@link SamplerStage#modifies} is true, no watermark replacing the draw, and either greedy without typical-p or
     * a top-k of 1 (llama.cpp common/sampling.cpp also replaces the draw by greedy for {@code top_k == 1}). A
     * forward backend's on-device argmax then selects the same token.
     */
    public boolean argmaxOnly() {
        return coneExtra == 0 && argmaxChain();
    }

    /**
     * The stage conditions of {@link #argmaxOnly} without the sparse-edit one: a chain this holds for selects the
     * argmax of its candidates whatever the fixed bias stage did to them, because every stage it carries either
     * preserves the argmax or runs after the one candidate is left. With no positive bias the edits only lower a
     * logit, so the winner lies among the largest {@code coneExtra + 1} raw logits ({@link #coneWidth}).
     */
    boolean argmaxChain() {
        if (watermarker != null && !greedy) {
            // the watermark's selection is not the argmax; the device's greedy id would skip it
            return false;
        }
        if (penalties || !(greedy && typicalP >= 1f) && topK != 1) {
            return false;
        }
        for (SamplerStage stage : chain) {
            StageKind kind = stage.kind();
            if (kind == StageKind.PENALTIES || kind == StageKind.TOP_K || kind == StageKind.TYPICAL_P
                    || kind == StageKind.TOP_P || kind == StageKind.MIN_P || kind == StageKind.TEMPERATURE) {
                // the argmax cone (they preserve or are ignored by the argmax; penalties were checked)
                continue;
            }
            // every other stage - a null-kind request stage (grammar, min-tokens, bad-words, reasoning-budget)
            // included - must prove it leaves the argmax alone; the on-device argmax sees no mask
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
        if (watermarkState != null) {
            watermarkState.accept(token);
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

    /**
     * Accepts a prompt into the chain's history stages, marking where the generated part starts: the watermark's
     * {@code single_turn} repeated-context search looks only at the tokens accepted after it. A chain without a
     * watermark accepts the tokens exactly as {@link #accept(int[], int, int)} does.
     */
    public void acceptPrompt(int[] tokens, int off, int n) {
        if (watermarkState == null) {
            accept(tokens, off, n);
            return;
        }
        watermarkState.startPrompt();
        try {
            accept(tokens, off, n);
        } finally {
            watermarkState.endPrompt();
        }
    }

    /** Forgets the chain's histories and restarts the RNG from {@link #seed()}. */
    public void reset() {
        for (SamplerStage stage : chain) {
            stage.reset();
        }
        if (watermarkState != null) {
            watermarkState.reset();
        }
        rng = new Mt19937(seed);
    }

    /**
     * A copy of the chain's state, every stage's {@link SamplerStage#saveState}, which
     * {@link #restoreState} puts back; the random streams are not part of it. Speculative verification feeds draft
     * tokens ahead of their confirmation and restores this state when a rejected one has to leave the history
     * (vLLM's grammar rollback).
     */
    public Object saveState() {
        Object[] states = new Object[chain.size()];
        for (int i = 0; i < states.length; i++) {
            states[i] = chain.get(i).saveState();
        }
        return new ChainState(this, states, watermarkState == null ? null : watermarkState.save());
    }

    /**
     * Puts back a state {@link #saveState} returned on this sampler.
     *
     * @throws IllegalArgumentException for a state of another sampler or of another kind
     */
    public void restoreState(Object state) {
        if (!(state instanceof ChainState) || ((ChainState) state).owner != this) {
            throw new IllegalArgumentException("not a state of this sampler: " + state);
        }
        Object[] states = ((ChainState) state).states;
        for (int i = 0; i < states.length; i++) {
            chain.get(i).restoreState(states[i]);
        }
        WatermarkState.Snapshot watermark = ((ChainState) state).watermark;
        if (watermark != null) {
            watermarkState.restore(watermark);
        }
    }

    /**
     * The length of the longest prefix of {@code tokens} the chain's constraint stages ({@link
     * SamplerStage#constrains}, the grammars) admit in order, each token judged after the ones before it (vLLM's
     * {@code GrammarMatcher.validate_tokens}). {@code tokens.length} when the chain has no constraint; the chain's
     * state is unchanged either way.
     *
     * @throws IllegalArgumentException for a negative token id
     */
    public int admittedPrefix(int[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        List<SamplerStage> constraints = new ArrayList<>();
        for (SamplerStage stage : chain) {
            if (stage.constrains()) {
                constraints.add(stage);
            }
        }
        if (constraints.isEmpty()) {
            return tokens.length;
        }
        Object[] saved = new Object[constraints.size()];
        for (int c = 0; c < saved.length; c++) {
            saved[c] = constraints.get(c).saveState();
        }
        try {
            for (int i = 0; i < tokens.length; i++) {
                if (tokens[i] < 0) {
                    throw new IllegalArgumentException("token " + i + " has negative id " + tokens[i]);
                }
                for (SamplerStage stage : constraints) {
                    if (!stage.admits(tokens[i])) {
                        return i;
                    }
                }
                for (SamplerStage stage : constraints) {
                    stage.accept(tokens[i]);
                }
            }
            return tokens.length;
        } finally {
            for (int c = 0; c < saved.length; c++) {
                constraints.get(c).restoreState(saved[c]);
            }
        }
    }

    /**
     * {@code tokens} cut before its first token the chain's constraint stages refuse after the tokens before it
     * ({@link #admittedPrefix}): the same array when every token is admitted, otherwise a copy of the admitted
     * prefix. Speculative decoding verifies only this prefix of a proposal (vLLM's {@code validate_tokens}): no
     * lossless verifier can confirm a refused token, and the rows after it have no grammar state to be scored under.
     *
     * @throws IllegalArgumentException for a negative token id
     */
    public int[] admitted(int[] tokens) {
        int n = admittedPrefix(tokens);
        return n == tokens.length ? tokens : Arrays.copyOf(tokens, n);
    }

    /** The per-stage states of one {@link #saveState} call, bound to the sampler that made it. */
    private static final class ChainState {
        final Sampler owner;
        final Object[] states;
        final WatermarkState.Snapshot watermark;

        ChainState(Sampler owner, Object[] states, WatermarkState.Snapshot watermark) {
            this.owner = owner;
            this.states = states;
            this.watermark = watermark;
        }
    }

    /**
     * Runs the chain over {@code logits} and returns the selected token id: the token {@link #apply} selects, found
     * sample-first by a greedy chain with a per-token constraint head ({@link #sampleFirst}), which masks the whole
     * row only when the head refuses the unmasked argmax. {@code logits} is not modified.
     *
     * @throws IllegalArgumentException      when the row is empty
     * @throws ConstraintDeadEndException    when the chain's constraints admit no candidate
     * @throws IllegalStateException         when no candidate has a finite logit
     */
    public int sample(float[] logits) {
        int token = sampleFirst(logits);
        return token >= 0 ? token : apply(logits).selectedId();
    }

    /**
     * The greedy token of a constrained chain found sample-first, or -1 when the chain does not qualify
     * ({@link #sampleFirstChain}) or its constraint head refuses the unmasked argmax. The row is loaded, the fixed
     * bias stage and the configured stages run over it unmasked, and every constraint-head stage is asked whether its
     * mask keeps the argmax ({@link TokenGate#keeps}). The head's masks set logits to {@code -inf} by token id and the
     * configured stages of a qualifying chain edit each logit by its own token alone, so masking before or after
     * them leaves the same logits on the kept tokens; an argmax every head stage keeps is therefore the argmax of
     * the masked row, ties included (the first maximum in vocabulary order on both). A refused argmax leaves the
     * caller to run the whole-row mask ({@link #apply}), which the head stages' state allows: {@code keeps} makes
     * only the state changes their apply makes again.
     */
    int sampleFirst(float[] logits) {
        if (logits == null || logits.length == 0 || !sampleFirstChain()) {
            return -1;
        }
        cur.load(logits);
        biasStage.apply(cur);
        applyStages(false);
        int best = 0;
        for (int i = 1; i < cur.size(); i++) {
            if (cur.logit(i) > cur.logit(best)) {
                best = i;
            }
        }
        if (cur.logit(best) == Float.NEGATIVE_INFINITY) {
            return -1;
        }
        int token = cur.id(best);
        for (SamplerStage stage : constraints) {
            if (!((TokenGate) stage).keeps(token)) {
                return -1;
            }
        }
        cur.select(best);
        return token;
    }

    /** Whether {@link #sample} may select sample-first ({@link SampleFirstRules#qualifies}). */
    boolean sampleFirstChain() {
        return greedy && SampleFirstRules.qualifies(constraints, chain, typicalP);
    }

    /**
     * Runs the chain over {@code logits} and returns the final candidate set with its probabilities and
     * selection (llama.cpp {@code cur_p} after {@code llama_sampler_apply}). The view stays valid until the next
     * call on this sampler. {@code logits} is not modified.
     *
     * <p>The chain's constraint stages (the grammars, the request's hard masks — {@code allowed_token_ids},
     * bad words, the min-tokens ban) mask the set first, before every configured stage (llama.cpp applies the
     * grammar before the chain on its grammar-first resample, vLLM applies {@code allowed_token_ids} and bad
     * words to the logits before any sampling stage), so the truncating stages (top-k, typical-p, top-p, min-p,
     * XTC, top-n-sigma) operate on the admitted set and each keeps at least one of its candidates. The
     * reasoning budget's forcing mask keeps its final say, applied behind them.
     *
     * <p>A chain built with mirostat or with {@code adaptive_p} in the order ends in its terminal selector, which
     * selects the token itself, as llama.cpp's chain construction replaces the dist stage (common/sampling.cpp:344-416);
     * the fixed draw only runs when no terminal selector applied.
     *
     * @throws IllegalArgumentException      when the row is empty
     * @throws ConstraintDeadEndException    when the chain's constraints admit no candidate
     * @throws IllegalStateException         when no candidate has a finite logit
     */
    public Candidates apply(float[] logits) {
        Objects.requireNonNull(logits, "logits");
        if (logits.length == 0) {
            throw new IllegalArgumentException("cannot sample from an empty logits row");
        }
        cur.load(logits);
        biasStage.apply(cur);
        applyConstraints();
        return runChain(false);
    }

    /**
     * Runs the chain over a row's top candidates in rank order (larger logit first, the lower id on ties) and
     * returns the selected token the chain picks over them — the same token it picks over the whole row when
     * {@link #coneWidth()} covers the candidates ({@code RowTopK}, llama.cpp backend sampling {@code -bs} over a
     * device top-k readback). The view stays valid until the next call on this sampler.
     *
     * @throws IllegalArgumentException when the arrays differ in length or hold fewer than one candidate
     * @throws IllegalStateException    when the chain carries constraint stages — the head needs the whole row,
     *                                  which {@link #coneWidth()} refuses — or no candidate has a finite logit
     */
    public Candidates applyTopK(int[] ids, float[] logits) {
        return applyTopK(ids, logits, false);
    }

    /**
     * The fixed bias stage over the ranked cone: its apply indexes the candidates by id, which only the
     * vocabulary-ordered row is, so the bans and biases look their candidates up by token id here. A mask or a
     * bias breaks the rank order, so the set re-sorts into the descending order the chain over the whole row
     * carries from its first sort — the same candidates in the same order, ties by the lower token id.
     */
    private void applyBiasToRanked() {
        if (coneExtra == 0) {
            return;
        }
        for (int id : banned) {
            int at = cur.indexOf(id);
            if (at >= 0) {
                cur.setLogit(at, Float.NEGATIVE_INFINITY);
            }
        }
        if (biasStage instanceof BiasStage) {
            BiasStage bias = (BiasStage) biasStage;
            int[] ids = bias.biasedIds();
            float[] values = bias.biases();
            for (int i = 0; i < ids.length; i++) {
                int at = cur.indexOf(ids[i]);
                if (at >= 0) {
                    if (values[i] <= -100f) {
                        cur.setLogit(at, Float.NEGATIVE_INFINITY);
                    } else {
                        cur.setLogit(at, cur.logit(at) + values[i]);
                    }
                }
            }
        }
        cur.sortDescendingById();
    }

    /**
     * The chain over the loaded candidates, from the fixed bias stage to the draw. {@code prefiltered} marks a
     * device readout that already applied the fixed bias stage and the chain's pre-top-k penalties and DRY.
     */
    private Candidates runChain(boolean prefiltered) {
        boolean terminal = applyStages(prefiltered);
        if (terminal) {
            // the terminal selector set the selected candidate already
        } else if (greedy) {
            chainEnd.selectArgmax(cur);
        } else if (watermarker != null) {
            chainEnd.selectWatermarked(cur, rng);
        } else {
            chainEnd.drawDist(cur, rng);
        }
        return cur;
    }

    /**
     * The configured stages over the loaded candidates, the constraint head and the stages greedy decoding skips left
     * out; true when a terminal selector (mirostat, adaptive-p) ran and selected the token.
     */
    private boolean applyStages(boolean prefiltered) {
        SamplerStage skipPenalties = deviceFilter.prefilteredPenalties();
        SamplerStage skipDry = deviceFilter.prefilteredDry();
        for (SamplerStage stage : chain) {
            if (prefiltered && (stage == skipPenalties || stage == skipDry)) {
                continue;
            }
            StageKind kind = stage.kind();
            if (stage.constrains() || stage.masks()) {
                continue; // the constraint head masked the set already
            }
            if (greedy && SampleFirstRules.skippedWhenGreedy(kind, typicalP)) {
                continue;
            }
            stage.apply(cur);
            if (kind == StageKind.MIROSTAT || kind == StageKind.ADAPTIVE_P) {
                return true;
            }
        }
        return false;
    }

    /**
     * The constraint head: the chain's grammars and hard masks ({@code allowed_token_ids}, bad words, the
     * min-tokens ban, in chain order) mask the set before every configured stage, so the truncating stages
     * operate on the admitted set and cannot cut it to nothing (see {@link #apply}).
     *
     * @throws ConstraintDeadEndException when the mask leaves no candidate a finite logit: the admitted set is
     *                                    empty (a grammar dead end, or the request's masks banning every id)
     *                                    and the step has nothing to sample
     */
    private void applyConstraints() {
        if (constraints.isEmpty()) {
            return;
        }
        for (SamplerStage stage : constraints) {
            stage.apply(cur);
        }
        if (cur.maxLogit() == Float.NEGATIVE_INFINITY) {
            throw new ConstraintDeadEndException("the constraint head admits no candidate: every token's logit"
                    + " is -inf under the request's constraints, so the step has nothing to sample");
        }
    }

    /**
     * Width of the chain's cone over a row's largest logits, or -1: when {@code k >= 1}, running the chain over
     * exactly the {@code k} largest raw logits of a row — with the banned and biased tokens among them — selects
     * the token the chain selects over the whole row, and for a truncated chain keeps and draws exactly the
     * candidates it keeps and draws, so a device top-k readback of that width replaces the row (llama.cpp backend
     * sampling {@code -bs}). Every stage that can keep a candidate outside the cone
     * refuses it: penalties and DRY (they lower tokens behind unpenalized ones), a positive logit bias (it raises
     * one past them), a typical, top-p, min-p, top-n-sigma or XTC stage before the chain's top-k (their statistics
     * and keeps run over the whole row), a terminal selector, an infill stage, and a request stage that masks,
     * constrains or forces (a grammar, {@code allowed_token_ids}, bad words, the min-tokens ban, a reasoning
     * budget) — the constraint head masks over the whole row, so a cone carries no chain that has one.
     *
     * <p>Without a top-k stage the cone is the largest {@code coneExtra + 1} raw logits, when the chain selects
     * the argmax ({@link #argmaxChain()}): every ban and non-positive bias only lowers a logit, so the raw argmax
     * or one of the edited tokens wins, and at most {@code coneExtra} edited tokens can outrank the winner. A
     * greedy chain with banned or biased tokens (llama.cpp's logit-bias sampler, Avalon's {@code greedy_argmax})
     * therefore reads back only that prefix. A positive bias raises a token from anywhere in the row, so it keeps
     * the whole-row path.</p>
     */
    public int coneWidth() {
        return deviceFilter.coneWidth();
    }

    /**
     * Width of the chain's cone over a device readout that first applies {@link #prefilterRow()} to the whole row
     * (the logit bias, the penalties and DRY before the top-k), or -1 when such a readout serves the chain no
     * better than the whole row. The bias and the pre-top-k penalties are baked into every row, so the readout
     * carries exactly {@code max(top_k, min_keep)} candidates — no banned-or-biased extra — and a chain these
     * edits cannot reproduce keeps its whole-row path.
     *
     * <p>The prefilter reproduces the same stages {@link #coneWidth()} refuses (penalties, DRY, a positive bias);
     * the whole-row stages before the top-k that need row statistics (top-p, typical, min-p, top-n-sigma, XTC, a
     * dynamic temperature) still refuse it, as do constraint stages, terminal selectors, infill, and a penalty or
     * DRY window wider than the device row carries.</p>
     */
    public int deviceConeWidth() {
        return deviceFilter.deviceConeWidth();
    }

    /** Whether this chain's cone needs a device prefilter (a bias, penalties or DRY) rather than the plain cone. */
    public boolean prefilterRequired() {
        return deviceFilter.required();
    }

    /**
     * The pre-top-k edits this chain's device readout applies to one row: the fixed logit-bias table, the
     * penalties and DRY that precede the top-k stage over the row's current history, and their chain order. A
     * chain with no such edit yields an empty row.
     */
    public LogitPrefilter.Row prefilterRow() {
        return deviceFilter.row();
    }

    /**
     * {@link #prefilterRow()} with {@code pendingTokens} accepted first: the history a completing prefill will
     * hold after the forward (the batch engine accepts a prompt's tokens once its last chunk has sampled). The
     * chain's own state is unchanged.
     */
    public LogitPrefilter.Row prefilterRow(int[] pendingTokens) {
        return deviceFilter.row(pendingTokens);
    }

    /**
     * Runs the chain over a row's top candidates whose whole-row edits a device prefilter already applied
     * ({@link #deviceConeWidth()}, {@link #prefilterRow()}): the fixed bias stage and the chain's pre-top-k
     * penalties and DRY are skipped, and every other stage runs over the candidates exactly as it runs over the
     * whole row.
     *
     * @throws IllegalArgumentException when the arrays differ in length or hold fewer than one candidate
     * @throws IllegalStateException    when the chain carries constraint stages or no candidate has a finite logit
     */
    public Candidates applyTopK(int[] ids, float[] logits, boolean prefiltered) {
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(logits, "logits");
        if (ids.length == 0 || ids.length != logits.length) {
            throw new IllegalArgumentException("top-k candidates of " + ids.length + " ids and " + logits.length
                    + " logits");
        }
        if (!constraints.isEmpty()) {
            throw new IllegalStateException("top-k candidates cannot run the chain's constraint head (a grammar,"
                    + " allowed_token_ids, bad words, the min-tokens ban masks by id over the whole row):"
                    + " coneWidth() is -1 and the device path must read the row back");
        }
        cur.loadRanked(ids, logits);
        if (!prefiltered) {
            applyBiasToRanked();
        }
        return runChain(prefiltered);
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
        /** Whether candidate {@code i}'s token id is {@code i} (a vocabulary-ordered row, not a ranked cone). */
        private boolean indexedById;
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
            indexedById = true;
            selected = -1;
        }

        /** Candidates of a row's top tokens in rank order: larger logit first, the lower id on ties. */
        void loadRanked(int[] tokens, float[] values) {
            int n = tokens.length;
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
                ids[i] = tokens[i];
                logits[i] = Float.isNaN(values[i]) ? Float.NEGATIVE_INFINITY : values[i];
                probs[i] = 0f;
            }
            size = n;
            sorted = true;
            indexedById = false;
            selected = -1;
        }

        /**
         * Whether candidate {@code i}'s token id is {@code i}, so a stage may index the set by token id: true for
         * a vocabulary-ordered row ({@link Sampler#apply}), false for a ranked cone ({@link
         * Sampler#applyTopK}).
         */
        public boolean indexedByTokenId() {
            return indexedById;
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

        /** Slot of candidate {@code token}, or -1 when the set does not carry it. */
        int indexOf(int token) {
            for (int i = 0; i < size; i++) {
                if (ids[i] == token) {
                    return i;
                }
            }
            return -1;
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
            indexedById = false;
        }

        /** Moves candidate {@code from} to slot {@code to} (llama.cpp's compaction while filtering). */
        public void move(int from, int to) {
            if (from != to) {
                ids[to] = ids[from];
                logits[to] = logits[from];
                probs[to] = probs[from];
                indexedById = false;
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
         * Sorts all candidates in descending logit order, ties by ascending token id: the order the chain over the
         * whole row carries after its first sort of a vocabulary-ordered set. A ranked readback loads candidates in
         * raw rank order, whose position is not the token id, so the fixed bias stage re-sorts by id here to keep
         * the ties of edited logits the whole-row chain would take.
         */
        public void sortDescendingById() {
            long[] k = sortKeys(size);
            for (int i = 0; i < size; i++) {
                k[i] = descendingKey(logits[i], ids[i]);
            }
            Arrays.sort(k, 0, size);
            for (int j = 0; j < size; j++) {
                int at = indexOf((int) k[j]);
                tmpIds[j] = ids[at];
                tmpLogits[j] = logits[at];
                tmpProbs[j] = probs[at];
            }
            System.arraycopy(tmpIds, 0, ids, 0, size);
            System.arraycopy(tmpLogits, 0, logits, 0, size);
            System.arraycopy(tmpProbs, 0, probs, 0, size);
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
            indexedById = false;
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
                float p = SamplerChainEnd.expShift(logits[i], max);
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
