package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.draft.BlockVerifier;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSession;
import com.eonmux.neuralfree.gguf.engine.draft.DraftProbs;
import com.eonmux.neuralfree.gguf.engine.draft.FlyVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.RejectionVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.SyntheticVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.Verifier;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.sampling.RequestStages;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;

/**
 * Speculative generation over a target {@link ForwardBackend} and a {@link DraftModel}: the drafter proposes tokens,
 * the target scores them all in one forward call and keeps the longest prefix it would have sampled itself (llama.cpp
 * tools/server speculative loop with {@code common_sampler_sample_and_accept_n}, common/sampling.cpp). Immutable;
 * one generation runs at a time, on the backend's thread.
 *
 * <p>Per {@link #generate} call:
 * <ol>
 *   <li><b>Model defaults.</b> The request takes the decoder's {@link SamplingDefaults} for every sampler setting
 *       and the token budget it leaves unset, as in {@link GenerationLoop}.</li>
 *   <li><b>Memory.</b> With prefix reuse (default on, {@code neuralfree.prefix.reuse}), the target keeps the
 *       longest common prefix of its stored tokens and the prompt behind a rollback check, as
 *       {@code GenerationLoop} does, and prefills only the tail; otherwise it resets. The drafter starts the
 *       generation through {@link DraftModel#beginGeneration(int)}: it keeps its own caches aligned with the
 *       reused rows (re-evaluating the last stored position so its rows pair with the right hidden state) or, when
 *       it cannot keep them, the target replays the whole prefix so its rows are rebuilt. A prompt longer than the
 *       context throws {@link ContextFullException}.</li>
 *   <li><b>Prefill.</b> The prompt is prefilled and enters the penalty history, as in {@link GenerationLoop}. The
 *       first token is sampled from its logits and offered to the {@link StopCriteria} session.</li>
 *   <li><b>Rounds.</b> With the last sampled token {@code t} at the next position {@code P = nPast}, the drafter
 *       proposes up to {@code k} tokens {@code d}. Without a usable proposal (empty, or shorter than
 *       {@code draftMin}) the round decodes {@code t} and samples one token. Otherwise
 *       {@link ForwardBackend#forwardRows} evaluates {@code [t, d_0 .. d_{k-1}]} at {@code P ..} and returns all
 *       {@code k + 1} logits rows, whose processed distributions the verify mode decides over (on the device, the
 *       rows' greedy ids instead; see <i>Device selection</i>):</li>
 * </ol>
 * <p><b>Verification modes</b> ({@link VerifyMode}, the builder's {@code verifyMode}, default {@link VerifyMode#MATCH}):
 * <ul>
 *   <li>{@link VerifyMode#MATCH}: the longest matching prefix of the chain's own samples (llama.cpp
 *       common/sampling.cpp:736 {@code common_sampler_sample_and_accept_n}), as described next.</li>
 *   <li>{@link VerifyMode#REJECTION}: vLLM {@code standard} ({@link RejectionVerifier});</li>
 *   <li>{@link VerifyMode#BLOCK}: Sun et al. block verification ({@link BlockVerifier});</li>
 *   <li>{@link VerifyMode#FLY}: entropy-gated deferred verification ({@link FlyVerifier}, lossy);</li>
 *   <li>{@link VerifyMode#SYNTHETIC}: benchmarking acceptance ({@link SyntheticVerifier}), refused unless the
 *       system property {@code neuralfree.spec.synthetic} is {@code true} at build time.</li>
 * </ul>
 * <p>A proposal is first cut before its first token the request's grammar constraint refuses after the tokens
 * before it ({@link Sampler#admitted}, vLLM's {@code validate_tokens}); a proposal cut below {@code draftMin}
 * decodes plainly. In {@code MATCH} row {@code i} is sampled, accepted into the sampler and offered to the session,
 * continuing while the sample equals {@code d_i}. In every other mode a {@link Verifier} runs the round against
 * the row distributions; each verifier documents its own RNG stream, and a drafter implementing
 * {@link DraftProbs} supplies its proposal distributions (one-hot otherwise). A greedy target chain degenerates
 * the lossless modes into {@code MATCH}'s argmax rule. A request that captures logprobs ({@code logprobs(n)},
 * {@code logprob_token_ids}) scores every emitted token in every mode — a confirmed draft token, a replacement and
 * a bonus alike — with the {@link LogprobCapture} the plain loop uses, over the verify row the token was emitted
 * at: the raw modes read the row itself, the processed modes the chain's distribution of that row under the
 * accepted tokens' chain state, as vLLM scores accepted and bonus tokens from the target logits.
 *
 * <p><b>Device selection</b> (llama.cpp backend sampling {@code -bs}, the builder's {@code verifyOnDevice}, default
 * on). When the mode is {@code MATCH}, no verifier is configured and the request reads no logprobs, the verify call
 * reads only what the chain's selection consumes: a plain argmax chain takes
 * {@link ForwardBackend#forwardRowsGreedy} and a round without a usable proposal decodes with
 * {@link ForwardBackend#decodeGreedyOnDevice} — a GPU target selects every row's greedy id on the device and reads
 * back {@code k + 1} ids. A chain whose selection stays inside a top-k cone ({@code Sampler#coneWidth}: a truncated
 * chain, or a greedy chain whose bans and non-positive biases only lower a logit; no penalties or DRY, no positive
 * logit bias, no grammar or other masking request stage, no whole-row statistics before the chain's top-k) takes
 * {@link ForwardBackend#forwardRowsTopK} and selects each row through the chain over the device's top candidates
 * ({@code Sampler#applyTopK}) — the same token the host chain selects over the whole row, with the request's seed.
 * Every other chain reads the {@code k + 1} logits rows back, and the fallback log names the reason. The tokens are
 * the same either way.
 *
 * <p><b>Request features.</b> The chain is the plain loop's: banned tokens, {@code logit_bias}, penalties, DRY, the
 * grammar constraint, {@code min_tokens}, {@code bad_words}, {@code allowed_token_ids} and the reasoning budget. A
 * verifier's row {@code i} is scored with draft tokens {@code 0 .. i-1} accepted into the chain; draft tokens a
 * verifier scores past the confirmed prefix ({@code BLOCK}, {@code FLY}) are accepted ahead of their confirmation
 * behind a {@link Sampler#saveState} per position, and the chain is restored to the confirmed prefix before the
 * replacement or bonus is accepted, so every emitted token's chain state equals the plain loop's (vLLM's grammar
 * advance over the speculative tokens with its {@code rollback}, structured_output/__init__.py). Mirostat runs in
 * every mode: it adapts {@code mu} on the accepted token's surprise in its row's candidate set, so a verifier's
 * emissions adapt it as the plain loop's draws would. Every non-{@code MATCH} mode refuses {@code adaptive_p}: it
 * folds only its own draw into its moving average, which a verifier's emissions replace.
 *
 * <p><b>Watermarking.</b> A request whose sampler carries a {@link WatermarkConfig} runs under rejection
 * verification, which is vLLM's standard rejection sampler under its {@code draft_sample_method='probabilistic'}
 * and {@code rejection_sample_method='standard'}. The drafter samples each proposal under the configuration's draft
 * role ({@link WatermarkConfig#draftWatermarker()}: key A of a dual-key configuration, or the ordinary key when the
 * configuration allows target-only watermarking) and reports the proposal distribution to the verifier, while the
 * target sampler draws under the target role ({@link WatermarkConfig#targetWatermarker()}: key B of a dual-key
 * configuration, else the key), so a confirmed draft token carries the draft key and a replacement or bonus the
 * target key, as the dual-key detector fuses them. A context the watermark deduplicates falls back to the ordinary
 * draw in both roles. A configuration without a draft key, a drafter that only replays stored tokens, or another
 * verify mode refuses by name.
 *
 * <p><b>Two-level verification</b> ({@code neuralfree.draft.tlv}, SpecPilot's TLV, arXiv 2611.02501). With a
 * quantized view of the target set ({@link QuantizedView}, a {@code Q8_0} or {@code Q6_K} copy of the target's
 * projection matrices over the target's own memory), each tree round verifies the proposal on the view first with
 * plain argmax and commits nothing: the accepted chain only steers a second, narrower draft anchored at the view's
 * bonus token, and the exact target verifies the composed proposal in one pass and commits. The view never decides
 * a token — its rows leave the sequence before the exact round — so every emitted token stays the exact target's
 * own greedy choice whatever the copy proposes; sampled requests are refused by name, and a drafter that drafts no
 * single-pass tree cannot carry the round.
 *
 * <p>With {@code a} drafted tokens confirmed, the round emits
 * {@code d_0 .. d_{a-1}} and the target's next token, and the last {@code k - a} positions leave memory:
 * by {@link ForwardBackend#rollback}, or for recurrent memory by restoring the snapshot taken before the
 * call and prefilling {@code [t, d_0 .. d_{a-1}]} again. A stop inside the verify (end of generation, stop string,
 * budget, cancel) leaves the same state as the ordinary rejection path: the extra verified positions are dropped
 * or replayed before the round returns, never left in the memory as an overshot state. The
 * drafter then hears {@code accept(a)}.
 * </ol>
 * Every emitted token passes through the same {@link Sampler} and {@link StopCriteria.Session} as in
 * {@link GenerationLoop}: the request's chain, stop strings, the token budget and reasoning routing are
 * unchanged. {@code MATCH} with an argmax sampler equals plain greedy decoding; with a stochastic chain the
 * lossless modes ({@code MATCH}, {@code REJECTION}, {@code BLOCK}) keep every emitted token a draw from the
 * target's distribution at its position.
 *
 * <p>{@code k = min(draftMax, nCtx - P - 1, ubatch - 1)}, lowered to the largest {@code k} for which
 * every sliding-window ring still holds the window of the new end after storing {@code k + 1} positions and dropping
 * {@code k} ({@link ModelMemory#ringsKeepWindow}). With DSV4 compressed caches {@code k} is at most the memory's
 * compressor rollback snapshots ({@link MemoryPlan#rollbackSnapshots()}, llama.cpp {@code n_rs_seq}, which the server
 * likewise bounds the draft by, tools/server/server-context.cpp:3805-3830), so 0 without snapshots; recurrent state
 * that cannot be snapshot ({@link ModelMemory#canSnapshot}) gives 0.
 * A full context shifts as in {@link GenerationLoop} when context shift is on ({@code neuralfree.context.shift},
 * the builder's {@code contextShift}). The shift keeps the first positions, so a drafter that reads the target's
 * hidden rows ({@link DraftModel#readsTargetHidden()}) keeps its rows over them through
 * {@link DraftModel#beginGeneration(int)} and the target re-evaluates the moved window through it; a drafter that
 * reads the target's tokens or runs its own model resets and re-aligns at its next draft. Otherwise, or when the
 * memory cannot shift, generation ends with {@link StopReason#CONTEXT_FULL}.
 *
 * <p>When a forward call, the drafter or the sampler fails, the target memory and the drafter are reset before the
 * exception propagates.
 */
public final class SpeculativeDecoder {

    private static final Logger LOGGER = LogManager.getLogger(SpeculativeDecoder.class);
    /** Most candidates one device top-k readback carries ({@code LLM_Row_TopK} supports up to 64). */
    private static final int MAX_DEVICE_TOPK = RowTopK.MAX_DEVICE_CANDIDATES;

    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    /** Counts of one generation's speculation. Immutable. */
    public static final class Stats {
        private final int rounds;
        private final int drafted;
        private final int accepted;

        Stats(int rounds, int drafted, int accepted) {
            this.rounds = rounds;
            this.drafted = drafted;
            this.accepted = accepted;
        }

        /** Verify calls made (rounds with a usable proposal). */
        public int rounds() { return rounds; }

        /** Proposed tokens the target verified. */
        public int drafted() { return drafted; }

        /** Proposed tokens the target confirmed. */
        public int accepted() { return accepted; }

        /** {@code accepted / drafted}, 0 when nothing was drafted. */
        public double acceptanceRate() {
            return drafted == 0 ? 0.0 : (double) accepted / drafted;
        }

        @Override
        public String toString() {
            return "Stats{rounds=" + rounds + ", drafted=" + drafted + ", accepted=" + accepted + '}';
        }
    }

    private final ForwardBackend backend;
    private final DraftModel drafter;
    private final GGUFTokenizer tokenizer;
    private final ReasoningFilter reasoning;
    private final LoraRegistry adapters;
    private final int draftMax;
    private final int draftMin;
    private final Consumer<Stats> statsSink;
    private final SamplingDefaults samplingDefaults;
    /** The custom logits processors of the engine configuration, empty when it configures none. */
    private final LogitsProcessors logitsProcessors;
    private final boolean prefixReuse;
    private final boolean contextShift;
    private final VerifyMode verifyMode;
    private final int flyWindow;
    private final double flyEntropyThreshold;
    private final int flyEntropyTopK;
    private final double[] syntheticRates;
    private final Double syntheticLength;
    private final boolean verifyOnDevice;
    /**
     * The requested tree verify width in nodes ({@code neuralfree.draft.tree.nodes}); a generation clamps it once
     * to the ubatch budget ({@code min(treeNodes, ubatch - 1)}, the verify forward holds the width plus the pending
     * token), so the step keeps one stable shape throughout the run. A non-tree drafter ignores it.
     */
    private final int treeNodes;
    /** Whether the tree verify covers targets with recurrent layers ({@code neuralfree.draft.tree.recurrent}). */
    private final boolean treeRecurrent;
    /** The target's rope type, which decides whether a verify batch carries its depth positions explicitly. */
    private final com.eonmux.neuralfree.gguf.arch.RopeType ropeType;
    /**
     * The quantized target view of two-level verification ({@code neuralfree.draft.tlv}), or null: it verifies
     * the draft tree greedily over the target's memory and commits nothing, and the exact {@link #backend}
     * verifies the composed proposal.
     */
    private final ForwardBackend quantizedBackend;
    /** The view itself, whose Q round opens the Q-view scope on a nested W12 target; null without the view. */
    private final QuantizedView quantizedView;
    /** The target's text-token section pattern for a multi-section rope's verify rows, unused otherwise. */
    private final com.eonmux.neuralfree.gguf.hparams.MropeTextPositions textPositions;
    private boolean hostReasonLogged;

    private SpeculativeDecoder(Builder b) {
        this.backend = b.backend;
        this.drafter = b.drafter;
        this.tokenizer = b.tokenizer;
        this.reasoning = b.reasoning;
        this.logitsProcessors = b.logitsProcessors;
        this.adapters = b.adapters;
        this.draftMax = b.draftMax;
        this.draftMin = b.draftMin;
        this.statsSink = b.statsSink;
        this.samplingDefaults = b.samplingDefaults;
        this.prefixReuse = b.prefixReuse;
        this.contextShift = b.contextShift;
        this.verifyMode = b.verifyMode;
        this.flyWindow = b.flyWindow;
        this.flyEntropyThreshold = b.flyEntropyThreshold;
        this.flyEntropyTopK = b.flyEntropyTopK;
        this.syntheticRates = b.syntheticRates;
        this.syntheticLength = b.syntheticLength;
        this.verifyOnDevice = b.verifyOnDevice;
        this.treeNodes = b.treeNodes;
        this.treeRecurrent = b.treeRecurrent;
        this.ropeType = b.ropeType;
        this.quantizedBackend = b.quantizedView == null ? null : b.quantizedView.backend();
        this.quantizedView = b.quantizedView;
        this.textPositions = b.textPositions;
        if (draftMin > draftMax) {
            throw new IllegalArgumentException("draft min " + draftMin + " exceeds draft max " + draftMax);
        }
    }

    public VerifyMode verifyMode() {
        return verifyMode;
    }

    /**
     * @param backend the target's forward backend
     * @param drafter a drafter bound to that target ({@link DraftModel#bindTarget})
     */
    public static Builder builder(ForwardBackend backend, DraftModel drafter, GGUFTokenizer tokenizer) {
        return new Builder(backend, drafter, tokenizer);
    }

    /**
     * The decoder of an engine model and its bound drafter: the configured reasoning parser
     * ({@link EngineConfig#reasoningParser()}, the chat template's by default, with its tag tokens), the draft lengths of {@code config} ({@code neuralfree.draft.max},
     * {@code neuralfree.draft.min}) and the model's sampling defaults unless {@code config} turns them off
     * ({@link SamplingDefaults#forModel}). The model's per-request adapters ({@link EngineModel#loraRegistry()}) run
     * on the target's passes, as vLLM serves LoRA with speculative decoding.
     *
     * @throws UnsupportedOperationException for models that do not generate autoregressively with a decoder-only
     *                                       memory
     */
    public static SpeculativeDecoder forModel(EngineModel model, DraftModel drafter, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(drafter, "drafter");
        Objects.requireNonNull(config, "config");
        SupportClass support = model.supportClass();
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
                throw new UnsupportedOperationException(model.architecture() + " (" + support
                        + ") has no decoder-only token loop to speculate on");
            default:
                break;
        }
        GGUFTokenizer tok = model.tokenizer();
        ReasoningFilter rf = ReasoningParsers.forModel(config.reasoningParser(), config.toolParser(), tok);
        return builder(model.forwardBackend(), drafter, tok)
                .reasoning(rf)
                .adapters(model.loraRegistry())
                .draftMax(config.draftMax())
                .draftMin(config.draftMin())
                .prefixReuse(config.memory().prefixReuse())
                .contextShift(config.memory().contextShift())
                .samplingDefaults(SamplingDefaults.forModel(model, config))
                .logitsProcessors(LogitsProcessors.forConfig(config))
                .treeNodes(config.speculativeOptions().treeNodes())
                .treeRecurrent(config.speculativeOptions().treeRecurrent())
                .ropeType(model.spec().hparams().ropeType())
                .textPositions(model.spec().hparams().mropeTextPositions())
                .build();
    }

    public ReasoningFilter reasoning() { return reasoning; }
    public int draftMax() { return draftMax; }
    public int draftMin() { return draftMin; }
    public SamplingDefaults samplingDefaults() { return samplingDefaults; }
    public boolean prefixReuse() { return prefixReuse; }
    public boolean contextShift() { return contextShift; }
    /** Whether {@code MATCH} rounds of an argmax chain select their tokens on the target's device. */
    public boolean verifyOnDevice() { return verifyOnDevice; }

    /**
     * Generates after {@code prompt}, streaming the content channel to {@code callback}.
     *
     * @see GenerationLoop#generate(int[], GenerationRequest, TokenCallback)
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        return generate(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null);
    }

    /**
     * Generates after {@code prompt}, routing every piece of output to {@code sink} with its channel.
     *
     * @param renderedPrompt the prompt text used to detect a generation prefix that opens reasoning; null renders it
     *                       from the prompt tokens
     * @throws ContextFullException          when the prompt is longer than the context
     * @throws UnsupportedOperationException when the backend has no memory, or the request scores its prompt rows
     *                                       ({@code prompt_logprobs}, {@code prompt_logprob_token_ids})
     * @throws IllegalArgumentException      for an empty prompt, a negative token id, a model sampling default the
     *                                       sampler rejects, or {@code adaptive_p} under a verify mode other than
     *                                       {@code MATCH}
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                     String renderedPrompt) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        request.requireNoPromptRows("speculative decoding (SpeculativeDecoder)");
        Objects.requireNonNull(sink, "sink");
        requirePrompt(prompt);
        LoraPlan plan = plan(request.lora(), prompt);
        ModelMemory memory = backend.memory();
        if (memory == null) {
            throw new UnsupportedOperationException("speculative generation needs model memory; this backend runs"
                    + " without a cache");
        }
        SequenceState seq = memory.sequence();
        int nCtx = seq.nCtx();
        if (prompt.length > nCtx) {
            throw new ContextFullException(prompt.length, nCtx);
        }
        GenerationRequest effective = request.withModelDefaults(samplingDefaults);
        WatermarkConfig watermark = effective.sampler().watermark();
        requireWatermarkSupport(watermark);
        boolean tree = treeMode();
        if (tree) {
            refuseTree(request, memory, plan, effective);
        }
        ReasoningFilter filter = GenerationLoop.forRequest(reasoning, effective);
        StopCriteria stop = StopCriteria.of(effective, tokenizer, filter);
        String rendered = renderedPrompt != null || filter.mode() == ReasoningFilter.Mode.NONE
                ? renderedPrompt : stop.renderPrompt(prompt);
        SamplerVocab vocab = Sampler.vocab(tokenizer);
        // the constraint head masks first: the grammar runs before the request's masks so a completed
        // constraint's end-of-generation tokens survive the min-tokens ban (the ban yields when nothing
        // else remains) and the budget force keeps the final say over a masked set
        List<SamplerStage> extra = new ArrayList<>();
        SamplerStage grammar = RequestStages.constraintStage(effective.constraint(),
                text -> tokenizer.encode(text, false, true), vocab);
        if (grammar != null) {
            extra.add(grammar);
        }
        extra.addAll(RequestStages.of(effective, vocab, filter, rendered));
        Sampler sampler = Sampler.createForSpeculativeTarget(samplerConfig(effective.sampler(), nCtx),
                stop.bannedTokens(), effective.logitBias(), extra, logitsProcessors.stages(effective), vocab);
        drafter.watermark(watermark, prompt.length, sampler.seed());
        Verifier verifier = verifierFor(effective.sampler(), sampler);
        StopCriteria.Session session = stop.start(sink, rendered);
        try (LoraSession adapters = plan.isDefault() ? null : LoraSession.open(this.adapters, plan)) {
            if (tree) {
                return runTree(prompt, memory, nCtx, sampler, session, effective, rendered);
            }
            return run(prompt, memory, seq, nCtx, sampler, verifier, session, effective, sink, plan, adapters);
        } catch (RuntimeException e) {
            try {
                backend.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            try {
                drafter.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    /**
     * Refuses a watermarked request this decoder cannot carry, naming the setting that would make it possible: vLLM
     * runs a watermark under speculative decoding only through standard rejection sampling
     * ({@code rejection_sample_method='standard'}, {@link VerifyMode#REJECTION} here), and only when the drafter can
     * sample its proposals under the draft role ({@link DraftModel#supportsWatermarking()}).
     *
     * @throws IllegalArgumentException when the watermark cannot be carried
     */
    private void requireWatermarkSupport(WatermarkConfig watermark) {
        if (watermark == null) {
            return;
        }
        watermark.draftWatermarker(); // refuses a single-key configuration without allow_target_only_watermarking
        if (verifyMode != VerifyMode.REJECTION) {
            throw new IllegalArgumentException("watermarking: speculative decoding carries the watermark only"
                    + " through rejection verification (vLLM's standard rejection sampler), not " + verifyMode
                    + "; set neuralfree.draft.verify=rejection or disable the watermark");
        }
        if (!drafter.supportsWatermarking()) {
            throw new IllegalArgumentException("watermarking: the drafter " + drafter + " proposes stored tokens and"
                    + " cannot sample them under the draft watermark key; a watermarked speculative request needs a"
                    + " model drafter (a draft model, the MTP head, EAGLE or DFlash)");
        }
    }

    /**
     * The adapter plan of the request's selection over the target's registry ({@link LoraPlan#resolve}); for a
     * request that selects none, the default plan, or a startup activated LoRA's plan from its invocation start.
     *
     * @throws UnsupportedOperationException when the request selects adapters and the target runs none per request
     */
    private LoraPlan plan(LoraSelection selection, int[] prompt) {
        if (selection == null) {
            return adapters == null ? LoraPlan.DEFAULT : LoraPlan.resolve(adapters, null, prompt);
        }
        if (adapters == null) {
            throw new UnsupportedOperationException("lora: the request selects adapters (" + selection
                    + ") on a target that runs none per request; speculative decoding runs the target's passes under"
                    + " them only when the model carries a per-request registry (SpeculativeDecoder.forModel)");
        }
        return LoraPlan.resolve(adapters, selection, prompt);
    }

    /**
     * Whether this decoder verifies a multi-branch proposal tree: a tree drafter. The verify runs on the target's
     * paged sequences and on a default single-user contiguous memory alike — one fixed-width forward of the one
     * target sequence under the tree mask.
     */
    private boolean treeMode() {
        return drafter instanceof TreeDraftModel && ((TreeDraftModel) drafter).branches() > 1;
    }

    /**
     * Refuses the request and configuration shapes the tree round cannot serve, naming each: a recurrent target
     * the round does not cover (see {@link #refuseRecurrentTree}), compressed memory, sliding-window trims, a
     * ubatch too small for one verify forward or a backend that refuses the batch the masked verify step runs
     * through, the context shift, per-request adapters, a verify mode other than {@code MATCH} and a logprobs
     * capture ({@code logprobs}, {@code logprob_token_ids}). A tree round moves the accepted rows onto their
     * committed positions and rolls the rejected verify rows out of the target sequence; those settings either
     * rewind state the round cannot restore (compressed memory, the shift) or read rows the round does not expose
     * (logprobs). A multi-section rope target ropes every row from its depth positions in all sections, so it
     * verifies like a single-section one.
     */
    private void refuseTree(GenerationRequest request, ModelMemory memory, LoraPlan plan,
                            GenerationRequest effective) {
        if (memory.hasRecurrentState()) {
            refuseRecurrentTree(memory, (TreeDraftModel) drafter);
        }
        if (memory.plan().hasCompressed()) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding rewinds the target"
                    + " rows of a round, which a memory with compressed caches cannot restore");
        }
        com.eonmux.neuralfree.gguf.memory.paged.PagedSequences pages = memory.pagedSequences();
        if (pages != null) {
            for (int g = 0; g < pages.groups(); g++) {
                if (pages.trims(g)) {
                    throw new UnsupportedOperationException("multi-branch (tree) speculative decoding rewinds the"
                            + " target rows of a round, which a sliding-window memory trims away");
                }
            }
        }
        if (memory.plan().ubatch() < 2) {
            throw new IllegalArgumentException("multi-branch (tree) speculative decoding verifies the pending token"
                    + " and its tree nodes in one forward of at least two rows, more than the ubatch "
                    + memory.plan().ubatch() + "; raise neuralfree.ubatch");
        }
        boolean singlePass = ((TreeDraftModel) drafter).singlePassVerify();
        if (pages == null && !singlePass && backend.batchRefusal() != null) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding verifies one target"
                    + " forward per depth over page-shared sequences, which needs paged memory (neuralfree.parallel"
                    + " > 1 or neuralfree.kv.paged=true); the drafter " + drafter + " drafts no single-pass tree");
        }
        if (pages == null && singlePass && backend.batchRefusal() != null) {
            throw new UnsupportedOperationException("the single-pass tree verify of drafter " + drafter
                    + " runs its masked verify step through " + ForwardBackend.class.getSimpleName()
                    + ".forwardBatch, which this backend refuses: " + backend.batchRefusal());
        }
        if (contextShift) {
            throw new IllegalArgumentException("multi-branch (tree) speculative decoding runs over paged sequences"
                    + " and cannot shift the context: clear neuralfree.context.shift");
        }
        if (!plan.isDefault()) {
            throw new UnsupportedOperationException("lora: multi-branch (tree) speculative decoding runs the target"
                    + " and the draft model without per-request adapters (" + plan + ")");
        }
        if (verifyMode != VerifyMode.MATCH) {
            throw new IllegalArgumentException("neuralfree.draft.verify=" + verifyMode + " does not verify a"
                    + " multi-branch proposal tree; use match (the target samples and the round keeps the drafted"
                    + " tokens that sample, llama.cpp's token-match acceptance over the tree)");
        }
        if (request.wantsLogprobs()) {
            throw new UnsupportedOperationException("logprobs: multi-branch (tree) speculative decoding verifies"
                    + " per-depth target rows and does not score them; clear logprobs / logprob_token_ids");
        }
        if (effective.sampler().temperature() < 0f) {
            throw new IllegalArgumentException("sampling temperature must not be negative, got "
                    + effective.sampler().temperature());
        }
        if (quantizedBackend != null && !effective.sampler().isGreedy()) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_TLV + " serves greedy requests only:"
                    + " the quantized target view verifies a proposal with plain argmax and cannot reproduce a"
                    + " sampler's draw, so a sampled request's tokens would diverge from the target-only loop;"
                    + " request greedy decoding (or temperature=0), or clear " + SpeculativeOptions.PROP_TLV);
        }
    }

    /**
     * Lifts the tree round's recurrent refusal for the covered families and refuses the rest by name: the
     * property must not opt out ({@code neuralfree.draft.tree.recurrent=false}), the two-level quantized view
     * must be off (its chain re-decodes target rows through a second backend, which rewinds the recurrent state
     * the exact round's replay builds on), the drafter must draft a single-pass tree (the per-depth rounds
     * re-decode the target row by row, which rewinds recurrent state), and the backend must cover every recurrent
     * layer's mixer, which its probe names (an uncovered mixer kind, a non-paged GPU memory, a non-linear layer
     * schedule).
     */
    private void refuseRecurrentTree(ModelMemory memory, TreeDraftModel tree) {
        if (!treeRecurrent) {
            throw new UnsupportedOperationException("the tree verify through this target's recurrent layers is"
                    + " disabled: set " + SpeculativeOptions.PROP_TREE_RECURRENT + "=true, or clear"
                    + " neuralfree.draft.tree to speculate without a tree");
        }
        if (quantizedBackend != null) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " does not combine with a target"
                    + " with recurrent layers: the quantized view's chain re-decodes target rows, which rewinds the"
                    + " recurrent state the exact round's replay builds on; clear " + SpeculativeOptions.PROP_TLV);
        }
        if (!tree.singlePassVerify()) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding verifies a target"
                    + " with recurrent layers in one masked step, which the drafter " + tree + " does not draft:"
                    + " its per-depth rounds re-decode the target row by row, which rewinds recurrent state; use a"
                    + " drafter whose single-pass tree covers the model (neuralfree.draft.tree=true)");
        }
        String refusal = backend.recurrentTreeRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("multi-branch (tree) speculative decoding verifies a target"
                    + " with recurrent layers through per-node recurrent state, which this backend does not cover:"
                    + " " + refusal);
        }
    }

    /**
     * Generates with a multi-branch proposal: the prompt prefills target sequence 0, and every round drafts a
     * {@link DraftTree}, verifies it (one fixed-width forward of the one sequence under the tree mask for a
     * single-pass drafter, depth by depth over the paged sequences otherwise) and commits the accepted path.
     * Each generation resets the memory (the scheme keeps no reusable prompt prefix).
     */
    private GenerationResult runTree(int[] prompt, ModelMemory memory, int nCtx, Sampler sampler,
                                     StopCriteria.Session session, GenerationRequest effective, String rendered) {
        SequenceMemory seqs = memory.sequences();
        TreeDraftModel tree = (TreeDraftModel) drafter;
        // a recurrent target verifies through the recurrent tree round: refuseTree lifted the refusal only when
        // the backend covers every recurrent layer's mixer and the drafter drafts a single-pass tree
        boolean recurrent = memory.hasRecurrentState() && backend.recurrentTreeRefusal() == null;
        // the single-pass round's acceptance walks on the device when the request asked for it, the sampler
        // reduces to the raw logits' argmax (the kernel sees no sampler mask or draw) and the backend walks there
        boolean deviceAccept = verifyOnDevice && sampler.argmaxOnly() && backend.treeAcceptOnDevice();
        // the verify runs one forward of width + 1 rows: the fixed width of neuralfree.draft.tree.nodes, clamped
        // once per generation to the ubatch budget, so the step keeps one stable shape throughout the run
        int width = Math.min(treeNodes, memory.plan().ubatch() - 1);
        long t0 = System.nanoTime();
        // the generation-start protocol: the tree round prefills the whole prompt (no prefix reuse), so the full
        // prefill covers whatever replay window the drafter asks over its stored rows
        int replay = tree.beginGeneration(prompt.length);
        if (replay < 0 || replay > prompt.length) {
            throw new IllegalStateException("the drafter asked to replay " + replay + " of the " + prompt.length
                    + " prompt positions");
        }
        backend.reset();
        float[] logits;
        try {
            logits = prefillTree(prompt);
        } catch (MemoryFullException e) {
            sampler.acceptPrompt(prompt, 0, prompt.length);
            long t1 = System.nanoTime();
            return session.finish(StopReason.CONTEXT_FULL, GenerationResult.Timings.of(prompt.length, t1 - t0, 0,
                    System.nanoTime() - t1));
        }
        long t1 = System.nanoTime();
        sampler.acceptPrompt(prompt, 0, prompt.length);
        SplittableRandom rng = new SplittableRandom(sampler.seed());
        int rounds = 0;
        int drafted = 0;
        int accepted = 0;
        int sampled = 0;
        int token = -1;
        StopReason reason;
        if (session.cancelled()) {
            reason = StopReason.CANCELLED;
        } else {
            Sampler.Candidates cur = sampler.apply(logits);
            token = cur.id(cur.selected());
            sampler.accept(token);
            sampled++;
            reason = session.offer(token);
        }
        while (reason == null) {
            if (session.cancelled()) {
                reason = StopReason.CANCELLED;
                break;
            }
            int pos = seqs.sequence(0).nPast();
            if (pos >= nCtx) {
                reason = StopReason.CONTEXT_FULL;
                break;
            }
            int k = draftLength(memory, pos, nCtx, !recurrent);
            float temperature = sampler.greedy() ? 0f : effective.sampler().temperature();
            DraftTree proposal;
            try {
                // the single-pass verify holds the fixed width; the linear and per-depth rounds keep the ubatch
                // budget they verify under
                int maxRows = tree.singlePassVerify() ? width : memory.plan().ubatch();
                proposal = k <= 0 ? new DraftTree(java.util.Collections.emptyList())
                        : tree.draftTree(token, k, sampler.hasConstraints() ? 1 : tree.branches(),
                                maxRows, temperature, rng);
            } catch (MemoryFullException e) {
                reason = StopReason.CONTEXT_FULL;
                break;
            }
            if (proposal.size() == 0 || proposal.depth() < draftMin) {
                tree.acceptTree(0);
                try {
                    token = decodeTree(token, pos, sampler);
                } catch (MemoryFullException e) {
                    reason = StopReason.CONTEXT_FULL;
                    break;
                }
                sampled++;
                reason = session.offer(token);
                continue;
            }
            Verified v;
            // a two-level round drafts twice: composedOut carries the tokens the second draft added to the
            // proposal the exact round verifies, on top of the first draft's own
            int[] composedOut = {0};
            try {
                v = quantizedBackend != null
                        ? twoLevelRound(seqs, tree, proposal, token, pos, sampler, session, width, temperature, rng,
                                composedOut, recurrent, deviceAccept)
                        : recurrent
                            ? TreeVerify.round(backend, seqs, proposal, token, pos, sampler, session, width,
                                    ropeType, textPositions, tree, true, deviceAccept)
                            : proposal.linear()
                                ? verifyLinearTree(seqs, proposal, token, pos, sampler, session)
                                : tree.singlePassVerify()
                                    ? TreeVerify.round(backend, seqs, proposal, token, pos, sampler, session, width,
                                            ropeType, textPositions, tree, false, deviceAccept)
                                    : verifyBranchingTree(seqs, proposal, token, pos, sampler, session,
                                            tree.branches());
            } catch (MemoryFullException e) {
                reason = StopReason.CONTEXT_FULL;
                break;
            }
            rounds++;
            drafted += proposal.size() + composedOut[0];
            accepted += v.accepted;
            sampled += v.emitted;
            token = v.next;
            reason = v.reason;
        }
        long t2 = System.nanoTime();
        statsSink.accept(new Stats(rounds, drafted, accepted));
        return session.finish(reason, GenerationResult.Timings.of(prompt.length, t1 - t0, sampled, t2 - t1));
    }

    /**
     * One round of two-level verification: the quantized view verifies the proposal greedily (its chain and
     * bonus only steer a second draft, nothing commits), the committed chain plays back once through the view so
     * the drafter's cache receives the chain rows and the second draft anchors below the chain, and the exact
     * target verifies the composed proposal in one pass and commits — the emitted tokens all come from the exact
     * weights. The view's passes run inside the view's Q scope: on a nested W12 target the scope routes the
     * target's own projections through their 8-bit Q view, on a copied target it is a no-op, and either way it
     * closes before the exact round. The view's rows leave the sequence before the exact round: the exact verify
     * re-stores whatever it accepts, and a wrong chain costs only the width it occupied.
     */
    private Verified twoLevelRound(SequenceMemory seqs, TreeDraftModel tree, DraftTree proposal, int token, int pos,
                                   Sampler sampler, StopCriteria.Session session, int width, float temperature,
                                   SplittableRandom rng, int[] composedOut, boolean recurrent, boolean deviceAccept)
            throws MemoryFullException {
        if (!sampler.greedy()) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_TLV + " serves greedy requests only: the"
                    + " quantized view verifies with plain argmax and cannot reproduce a sampler's draw; disable"
                    + " temperature for this request");
        }
        TwoLevelVerify.Chain chain;
        int chainLen;
        try (QuantizedView.QScope qScope = quantizedView.openQScope()) {
            chain = TwoLevelVerify.round(quantizedBackend, proposal, token, pos, width, ropeType, tree);
            // keep the bonus's subtree inside the fixed width: the chain may run at most width - 1 nodes deep
            chainLen = Math.min(chain.tokens.length, Math.max(1, width - 1));
            TwoLevelVerify.commitChain(quantizedBackend, seqs, token, pos, chain.tokens, chainLen);
        }
        int budget = width - chainLen;
        DraftTree sub;
        try {
            sub = budget <= 0 ? new DraftTree(java.util.Collections.emptyList())
                    : tree.draftTree(chain.bonus, budget, sampler.hasConstraints() ? 1 : tree.branches(), budget,
                            temperature, rng);
        } catch (MemoryFullException e) {
            seqs.remove(0, pos, -1);
            throw e;
        }
        DraftTree composed = TwoLevelVerify.compose(chain, chainLen, sub);
        composedOut[0] = composed.size();
        // the view's rows leave: the exact verify re-stores whatever it accepts with exact weights
        seqs.remove(0, pos, -1);
        return TreeVerify.round(backend, seqs, composed, token, pos, sampler, session, width, ropeType, textPositions,
                tree, recurrent, deviceAccept);
    }

    /** Prefills {@code prompt} into target sequence 0 through one batched forward and returns the last row's logits. */
    private float[] prefillTree(int[] prompt) {
        ForwardBatch.Builder b = ForwardBatch.builder(prompt.length);
        for (int i = 0; i < prompt.length; i++) {
            b.add(prompt[i], i, 0, i == prompt.length - 1);
        }
        float[][] rows = backend.forwardBatch(b.build());
        if (rows.length != 1) {
            throw new IllegalStateException("the prompt prefill returned " + rows.length + " logits rows for one"
                    + " requested row");
        }
        return rows[0];
    }

    /** Decodes {@code token} at {@code pos} on target sequence 0 and samples the next token. */
    private int decodeTree(int token, int pos, Sampler sampler) {
        ForwardBatch batch = ForwardBatch.builder(1).add(token, pos, 0, true).build();
        float[][] rows = backend.forwardBatch(batch);
        Sampler.Candidates cur = sampler.apply(rows[0]);
        int next = cur.id(cur.selected());
        sampler.accept(next);
        return next;
    }

    /**
     * Verifies a linear proposal over one target sequence: the target samples a token after the pending token and
     * after each drafted token, and the round keeps the longest prefix whose samples equal the drafted tokens
     * (llama.cpp's token-match acceptance, its {@code common_sampler_sample_and_accept_n}; a greedy chain samples
     * argmax, a sampling chain draws from the target). The unconfirmed tail leaves the sequence.
     */
    private Verified verifyLinearTree(SequenceMemory seqs, DraftTree proposal, int token, int pos, Sampler sampler,
                                      StopCriteria.Session session) {
        int[] draft = proposal.linearTokens();
        if (sampler.hasConstraints()) {
            draft = sampler.admitted(draft);
        }
        int k = draft.length;
        if (k < Math.max(1, draftMin)) {
            // the constraints left no chain worth verifying: decode the pending token's target row on its own
            int next = decodeTree(token, pos, sampler);
            return new Verified(0, 1, next, session.offer(next));
        }
        int[] batchTokens = new int[k + 1];
        batchTokens[0] = token;
        System.arraycopy(draft, 0, batchTokens, 1, k);
        ForwardBatch.Builder b = ForwardBatch.builder(k + 1);
        for (int i = 0; i <= k; i++) {
            b.add(batchTokens[i], pos + i, 0, true);
        }
        float[][] rows = backend.forwardBatch(b.build());
        Verified v = matchVerify(sampler, session, draft, i -> {
            Sampler.Candidates cur = sampler.apply(rows[i]);
            return cur.id(cur.selected());
        });
        int drop = k - v.accepted;
        if (drop > 0) {
            seqs.remove(0, pos + v.accepted + 1, -1);
        }
        ((TreeDraftModel) drafter).acceptTree(v.accepted);
        return v;
    }

    /**
     * Verifies a branching proposal one target forward per depth: each depth's nodes run on their own page-shared
     * target sequences, the target samples after the accepted node of the previous depth, and the round keeps every
     * node at the depth whose token the sample equals (llama.cpp's {@code common_sampler_sample_and_accept_n} over
     * the tree). The first depth with no match emits the target's own token and ends the round; the accepted branch
     * becomes target sequence 0.
     */
    private Verified verifyBranchingTree(SequenceMemory seqs, DraftTree proposal, int token, int pos,
                                         Sampler sampler, StopCriteria.Session session, int branches) {
        List<DraftTree.Node> nodes = proposal.nodes();
        float[] predecessor = decodeRow(token, pos, 0);
        List<Integer> frontier = java.util.Collections.emptyList();
        int acceptedDepth = 0;
        int emitted = 0;
        int next = -1;
        StopReason reason = null;
        int[] lastDepth = new int[branches];
        java.util.Arrays.fill(lastDepth, -1);
        // whether the last depth processed ended in a match: only then does the target sample the bonus token past
        // the accepted path; a mismatch already sampled the token that ends the round
        boolean advanced = false;
        for (int depth = 1; depth <= proposal.depth(); depth++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < nodes.size(); i++) {
                DraftTree.Node n = nodes.get(i);
                if (n.depth() == depth && (frontier.isEmpty() ? n.parent() == -1 : frontier.contains(n.parent()))) {
                    candidates.add(i);
                }
            }
            if (candidates.isEmpty()) {
                break;
            }
            for (int index : candidates) {
                DraftTree.Node n = nodes.get(index);
                int branch = n.branch();
                int parentBranch = n.parent() < 0 ? 0 : nodes.get(n.parent()).branch();
                if (lastDepth[branch] != depth - 1) {
                    if (branch != parentBranch) {
                        seqs.remove(branch, 0, -1);
                        seqs.copy(parentBranch, branch, 0, -1);
                    }
                    lastDepth[branch] = depth - 1;
                }
            }
            ForwardBatch.Builder b = ForwardBatch.builder(candidates.size());
            for (int index : candidates) {
                DraftTree.Node n = nodes.get(index);
                b.add(n.token(), pos + depth, n.branch(), true);
            }
            float[][] rows = backend.forwardBatch(b.build());
            for (int index : candidates) {
                lastDepth[nodes.get(index).branch()] = depth;
            }
            Sampler.Candidates cur = sampler.apply(predecessor);
            next = cur.id(cur.selected());
            sampler.accept(next);
            emitted++;
            reason = session.offer(next);
            if (reason != null) {
                break;
            }
            List<Integer> matched = new ArrayList<>();
            float[] nextPredecessor = null;
            for (int j = 0; j < candidates.size(); j++) {
                if (nodes.get(candidates.get(j)).token() == next) {
                    matched.add(candidates.get(j));
                    if (nextPredecessor == null) {
                        nextPredecessor = rows[j];
                    }
                }
            }
            if (matched.isEmpty()) {
                advanced = false;
                break;
            }
            frontier = matched;
            acceptedDepth = depth;
            predecessor = nextPredecessor;
            advanced = true;
        }
        if (reason == null && advanced) {
            Sampler.Candidates cur = sampler.apply(predecessor);
            next = cur.id(cur.selected());
            sampler.accept(next);
            emitted++;
            reason = session.offer(next);
        }
        commitTree(seqs, nodes, frontier, pos, acceptedDepth, branches);
        ((TreeDraftModel) drafter).acceptTree(acceptedDepth);
        return new Verified(acceptedDepth, emitted, next, reason);
    }

    /**
     * Keeps the accepted path as sequence 0 and empties the other target sequences: rows past the accepted depth
     * leave every sequence, the accepted branch's pages copy over sequence 0 and every other branch sequence
     * empties.
     */
    private static void commitTree(SequenceMemory seqs, List<DraftTree.Node> nodes, List<Integer> frontier, int pos,
                                   int acceptedDepth, int branches) {
        int keep = pos + acceptedDepth;
        int best = frontier.isEmpty() ? -1 : nodes.get(frontier.get(0)).branch();
        for (int s = 0; s < branches; s++) {
            if (seqs.posMax(s) > keep) {
                seqs.remove(s, keep + 1, -1);
            }
        }
        if (best > 0) {
            seqs.remove(0, 0, -1);
            seqs.copy(best, 0, 0, -1);
        }
        for (int s = 1; s < branches; s++) {
            if (s != best) {
                seqs.remove(s, 0, -1);
            }
        }
    }

    /** One token's logits at {@code pos} on target sequence {@code seq}. */
    private float[] decodeRow(int token, int pos, int seq) {
        ForwardBatch batch = ForwardBatch.builder(1).add(token, pos, seq, true).build();
        float[][] rows = backend.forwardBatch(batch);
        return rows[0];
    }

    private GenerationResult run(int[] prompt, ModelMemory memory, SequenceState seq, int nCtx, Sampler sampler,
                                 Verifier verifier, StopCriteria.Session session, GenerationRequest effective,
                                 ReasoningFilter.Sink sink, LoraPlan plan, LoraSession adapters) {
        long t0 = System.nanoTime();
        int reused = prepareMemory(prompt, seq, plan.identity());
        seq.adapters(plan.identity());
        // a drafter with caches of its own keeps the reused prefix and asks for the last stored position back, so
        // its first row pairs with the hidden row before the new tail; one that cannot keep it replays it in full
        int replay = drafter.beginGeneration(reused);
        if (replay < 0 || replay > reused) {
            throw new IllegalStateException("the drafter asked to replay " + replay + " of the " + reused
                    + " reused prompt positions");
        }
        int start = reused - replay;
        int[] tail = start == 0 ? prompt : Arrays.copyOfRange(prompt, start, prompt.length);
        List<TokenScore[]> tokenLogprobs = new ArrayList<>();
        Scores scores = new Scores(LogprobCapture.of(effective), tokenLogprobs, sink);
        // the rows before an activated LoRA's invocation prefill in the slot before it, the rest in the plan's
        // slot; the verify rows and decodes run in the plan's slot, the drafter's proposals in slot 0 (vLLM: the
        // drafter does not use the target's LoRA)
        float[] logits = prefill(tail, start, plan, adapters);
        sampler.acceptPrompt(prompt, 0, prompt.length);
        long t1 = System.nanoTime();
        // Non-MATCH verification runs per generation with its own uniform stream, seeded like the sampler's chain.
        SplittableRandom roundsRng = verifier == null ? null : new SplittableRandom(sampler.seed());
        DeviceMode mode = deviceMode(verifier, sampler, scores);
        if (mode == DeviceMode.HOST && !hostReasonLogged) {
            String reason = hostReason(verifier, sampler, scores).trim();
            if (!reason.isEmpty()) {
                LOGGER.info("speculative verification of {} reads the rows back: {}", drafter, reason);
                hostReasonLogged = true;
            }
        }

        int rounds = 0;
        int drafted = 0;
        int accepted = 0;
        int sampled = 0;
        int token = -1;
        StopReason reason;
        if (session.cancelled()) {
            reason = StopReason.CANCELLED;
        } else {
            Sampler.Candidates cur = sampler.apply(logits);
            token = cur.id(cur.selected());
            scores.record(logits, cur, token);
            sampler.accept(token);
            sampled++;
            reason = session.offer(token);
        }
        while (reason == null) {
            if (session.cancelled()) {
                reason = StopReason.CANCELLED;
                break;
            }
            int pos = seq.nPast();
            if (pos >= nCtx) {
                GenerationLoop.Shift shift = contextShift
                        ? GenerationLoop.shiftContext(backend, tokenizer, prompt[0], seq) : null;
                if (shift == null) {
                    reason = StopReason.CONTEXT_FULL;
                    break;
                }
                if (drafter.readsTargetHidden()) {
                    realignAfterShift(shift.kept(), seq, plan, adapters);
                } else {
                    // the drafter reads the target's stored tokens or runs its own model: its next draft aligns
                    // itself, and the shifted target rows stay as the shift moved them
                    drafter.reset();
                }
                pos = seq.nPast();
            }
            int k = draftLength(memory, pos, nCtx, true);
            int[] draft;
            if (k > 0) {
                if (adapters != null) {
                    this.adapters.useSlot(0);
                }
                try {
                    draft = sampler.admitted(checkDraft(drafter.draft(token, k,
                            sampler.greedy() ? 0f : effective.sampler().temperature()), k));
                } finally {
                    if (adapters != null) {
                        adapters.runKey();
                    }
                }
            } else {
                draft = new int[0];
            }
            if (draft.length == 0 || draft.length < draftMin) {
                token = decodeOne(token, pos, mode, sampler, scores);
                sampled++;
                reason = session.offer(token);
                continue;
            }
            Verified v = verify(memory, verifier, roundsRng, sampler, session, token, draft, pos, scores, mode);
            rounds++;
            drafted += draft.length;
            accepted += v.accepted;
            sampled += v.emitted;
            token = v.next;
            reason = v.reason;
        }
        long t2 = System.nanoTime();
        statsSink.accept(new Stats(rounds, drafted, accepted));
        GenerationResult result = session.finish(reason,
                GenerationResult.Timings.of(tail.length, t1 - t0, sampled, t2 - t1));
        return tokenLogprobs.isEmpty() ? result : result.withLogprobs(tokenLogprobs);
    }

    /**
     * How this generation's rounds select their tokens: {@link #HOST} reads the rows back and samples on the host,
     * {@link #GREEDY} selects greedy ids on the device, {@link #SAMPLED} samples truncated chains over the device's
     * per-row top candidates (llama.cpp backend sampling {@code -bs}).
     */
    private enum DeviceMode {
        HOST, GREEDY, SAMPLED
    }

    /**
     * Whether this generation's rounds select on the target's device, and how: {@code MATCH} without a logprobs
     * capture and without a verifier over a plain argmax chain selects greedy ids; over a chain whose keeps stay
     * inside a top-k cone ({@code Sampler#coneWidth}) it samples the device's top candidates; anything else reads
     * the rows back, the reason named in the fallback log.
     */
    private DeviceMode deviceMode(Verifier verifier, Sampler sampler, Scores scores) {
        if (!verifyOnDevice || verifier != null || scores.wanted()) {
            return DeviceMode.HOST;
        }
        if (sampler.argmaxOnly()) {
            return DeviceMode.GREEDY;
        }
        int cone = sampler.coneWidth();
        return cone > 0 && cone <= MAX_DEVICE_TOPK && backend.topKOnDevice() ? DeviceMode.SAMPLED : DeviceMode.HOST;
    }

    /** Why a generation selects on the host, for the fallback log. */
    private String hostReason(Verifier verifier, Sampler sampler, Scores scores) {
        if (!verifyOnDevice) {
            return "device verification is off";
        }
        if (verifier != null) {
            return "the " + verifyMode + " verifier scores the rows";
        }
        if (scores.wanted()) {
            return "logprobs read the rows back";
        }
        int cone = sampler.coneWidth();
        if (cone > MAX_DEVICE_TOPK) {
            return "the chain's device cone of " + cone + " candidates exceeds the " + MAX_DEVICE_TOPK
                    + "-candidate readback";
        }
        if (cone > 0 && !backend.topKOnDevice()) {
            return "the backend reads the sampled rows back";
        }
        if (sampler.argmaxOnly()) {
            return "";
        }
        if (sampler.hasConstraints()) {
            return "the request's constraints (a grammar, allowed_token_ids, bad words, the min-tokens ban) mask"
                    + " over the whole row";
        }
        return "the sampler chain keeps candidates outside a device top-k readback";
    }

    /**
     * A round without a usable proposal: decodes {@code token} at {@code pos} and accepts the target's next token into
     * the chain, selected on the device (the greedy id, or the draw over the device's top candidates) or sampled from
     * the read-back logits.
     */
    private int decodeOne(int token, int pos, DeviceMode mode, Sampler sampler, Scores scores) {
        int next;
        if (mode == DeviceMode.GREEDY) {
            next = backend.decodeGreedyOnDevice(token, pos);
            drafter.accept(0);
        } else if (mode == DeviceMode.SAMPLED) {
            RowTopK top = backend.decodeTopKOnDevice(token, pos, coneWidth(sampler));
            drafter.accept(0);
            Sampler.Candidates cur = sampler.applyTopK(top.ids(0), top.logits(0));
            next = cur.id(cur.selected());
        } else {
            float[] logits = backend.decode(token, pos);
            drafter.accept(0);
            Sampler.Candidates cur = sampler.apply(logits);
            next = cur.id(cur.selected());
            scores.record(logits, cur, next);
        }
        sampler.accept(next);
        return next;
    }

    /**
     * Prefills {@code tail} at {@code reused} and leaves the passes in the plan's adapter slot: the rows before an
     * activated LoRA's invocation start run first in the slot before it, as {@link GenerationLoop} splits them.
     * Null {@code adapters} runs one pass in slot 0.
     */
    private float[] prefill(int[] tail, int start, LoraPlan plan, LoraSession adapters) {
        if (adapters == null) {
            return backend.prefill(tail, start);
        }
        int split = Math.max(start, plan.from());
        if (split > start) {
            adapters.runPre();
            backend.prefill(Arrays.copyOfRange(tail, 0, split - start), start);
        }
        adapters.runKey();
        return backend.prefill(Arrays.copyOfRange(tail, split - start, tail.length), split);
    }

    /**
     * Rebuilds the drafter over the target's post-shift context. The positions the shift kept still hold the tokens
     * (and target rows) the drafter cached, so {@link DraftModel#beginGeneration(int)} keeps their rows and asks for
     * the boundary position back; the target then re-evaluates the window from {@code kept - replay} to its end,
     * feeding the drafter the hidden rows of the moved positions. A drafter that cannot keep the kept prefix asks
     * for the whole post-shift context, and a target memory that cannot drop its tail for the window resets and
     * replays the whole context, as llama.cpp's speculative path re-runs the draft context after a shift.
     */
    private void realignAfterShift(int kept, SequenceState seq, LoraPlan plan, LoraSession adapters) {
        int[] stored = seq.tokens();
        int replay = drafter.beginGeneration(kept);
        if (replay < 0 || replay > kept) {
            throw new IllegalStateException("the drafter asked to replay " + replay + " of the " + kept
                    + " positions the context shift kept");
        }
        int start = kept - replay;
        if (!backend.canRollback(stored.length - start)) {
            backend.reset();
            seq.adapters(plan.identity());
            drafter.reset();
            start = 0;
        }
        prefill(Arrays.copyOfRange(stored, start, stored.length), start, plan, adapters);
    }

    /**
     * Brings the target memory to the prompt's start and reports the prompt tokens it already holds: with prefix
     * reuse, the longest common prefix of the stored tokens and the prompt (at most {@code prompt.length - 1}, so
     * the last prompt token is evaluated for logits) stays behind a {@link ForwardBackend#canRollback} check, as
     * in {@code GenerationLoop} — under the adapters of {@code adapterIdentity}, so rows computed under other
     * adapters reuse nothing; the drafter then starts from the reused prefix through
     * {@link DraftModel#beginGeneration(int)}, keeping its own caches over it (a separate draft model prefills the
     * shared prefix into its small cache, the hidden-fed drafters keep their rows and re-evaluate the boundary
     * position). A lost proposal can only lower the acceptance rate, never change a lossless verify's result.
     *
     * @return the number of prompt tokens already in memory, positions {@code [0, reused)}
     */
    private int prepareMemory(int[] prompt, SequenceState seq, String adapterIdentity) {
        int reused = prefixReuse ? Math.min(seq.commonPrefix(prompt, adapterIdentity), prompt.length - 1) : 0;
        int drop = seq.nPast() - reused;
        if (reused > 0 && (drop == 0 || backend.canRollback(drop))) {
            if (drop > 0) {
                backend.rollback(drop);
            }
            return reused;
        }
        backend.reset();
        return 0;
    }

    /**
     * The request's logprobs capture ({@code logprobs(n)}, {@code logprob_token_ids}): every emitted token's
     * {@link LogprobCapture} at its position — the token first, then its best alternatives or the requested ids,
     * in the request's {@link LogprobsMode} — the same capture {@code SequenceRun} makes for the plain loop. Each
     * capture is routed to the sink as an empty content piece carrying it, so a stream carries one entry per
     * token. Inert when the request captures nothing.
     */
    private static final class Scores {
        private final LogprobCapture capture;
        private final List<TokenScore[]> out;
        private final ReasoningFilter.Sink sink;

        Scores(LogprobCapture capture, List<TokenScore[]> out, ReasoningFilter.Sink sink) {
            this.capture = capture;
            this.out = out;
            this.sink = sink;
        }

        /** Whether the request captures logprobs: a round that does must read its rows back. */
        boolean wanted() {
            return capture.active();
        }

        /** Records {@code token} sampled by the chain run over the raw row {@code raw} that left {@code cur}. */
        void record(float[] raw, Sampler.Candidates cur, int token) {
            if (capture.active()) {
                emit(capture.capture(raw, cur, token), token);
            }
        }

        /**
         * Records {@code token} emitted at the verify row {@code raw}, whose chain distribution under the accepted
         * tokens' state is {@code processed}.
         */
        void record(float[] raw, Verifier.Distribution processed, int token) {
            if (capture.active()) {
                emit(capture.capture(raw, processed, token), token);
            }
        }

        private void emit(TokenScore[] scores, int token) {
            out.add(scores);
            sink.accept(ReasoningFilter.Channel.CONTENT, token, "", scores);
        }
    }

    /** Outcome of one verify call. */
    static final class Verified {
        final int accepted;
        final int emitted;
        final int next;
        final StopReason reason;

        Verified(int accepted, int emitted, int next, StopReason reason) {
            this.accepted = accepted;
            this.emitted = emitted;
            this.next = next;
            this.reason = reason;
        }
    }

    /**
     * Scores {@code [token, draft..]} at {@code pos} and emits the confirmation the verify mode decides over, then
     * drops the rejected positions from memory. A stop inside the verify (a non-null reason) leaves the same state
     * as the ordinary rejection path: the extra verified positions are replayed away, never left as an overshot
     * state.
     */
    private Verified verify(ModelMemory memory, Verifier verifier, SplittableRandom roundsRng, Sampler sampler,
                            StopCriteria.Session session, int token, int[] draft, int pos, Scores scores,
                            DeviceMode mode) {
        int k = draft.length;
        int[] batch = new int[k + 1];
        batch[0] = token;
        System.arraycopy(draft, 0, batch, 1, k);
        boolean recurrent = memory.hasRecurrentState();
        if (recurrent) {
            memory.snapshot();
        }
        Verified outcome;
        if (mode == DeviceMode.GREEDY) {
            outcome = deviceVerify(sampler, session, draft, batch, pos);
        } else if (mode == DeviceMode.SAMPLED) {
            outcome = sampledVerify(sampler, session, draft, batch, pos);
        } else {
            outcome = hostVerify(verifier, roundsRng, sampler, session, draft, batch, pos, scores);
        }
        int drop = k - outcome.accepted;
        if (drop > 0) {
            if (recurrent) {
                memory.restore();
                backend.prefill(Arrays.copyOf(batch, outcome.accepted + 1), pos);
            } else {
                if (!backend.canRollback(drop)) {
                    throw new IllegalStateException("the target memory cannot drop the " + drop
                            + " rejected draft positions");
                }
                backend.rollback(drop);
            }
        }
        drafter.accept(outcome.accepted);
        return outcome;
    }

    /** {@link VerifyMode#MATCH} over the greedy ids the target selected on its device, one per batch row. */
    private Verified deviceVerify(Sampler sampler, StopCriteria.Session session, int[] draft, int[] batch, int pos) {
        int[] ids = backend.forwardRowsGreedy(batch, pos, null);
        if (ids.length != batch.length) {
            throw new IllegalStateException("verification of " + batch.length + " tokens returned " + ids.length
                    + " greedy ids");
        }
        return matchVerify(sampler, session, draft, i -> ids[i]);
    }

    /**
     * {@link VerifyMode#MATCH} over the device's per-row top candidates: every row is sampled through the chain
     * over exactly the candidates the device read back ({@code Sampler#applyTopK}), the same draw the host chain
     * makes over the whole row.
     */
    private Verified sampledVerify(Sampler sampler, StopCriteria.Session session, int[] draft, int[] batch, int pos) {
        int cone = coneWidth(sampler);
        RowTopK rows = backend.forwardRowsTopK(batch, pos, cone, null);
        if (rows.rows() != batch.length) {
            throw new IllegalStateException("verification of " + batch.length + " tokens returned " + rows.rows()
                    + " candidate rows");
        }
        return matchVerify(sampler, session, draft, i -> {
            Sampler.Candidates cur = sampler.applyTopK(rows.ids(i), rows.logits(i));
            return cur.id(cur.selected());
        });
    }

    /** The chain's device cone, at most the rows' width (a vocabulary below the cone keeps all of its tokens). */
    private int coneWidth(Sampler sampler) {
        return Math.max(1, sampler.coneWidth());
    }

    /** Verification over the read-back logits rows: {@code MATCH} through the chain, or the mode's verifier. */
    private Verified hostVerify(Verifier verifier, SplittableRandom roundsRng, Sampler sampler,
                                StopCriteria.Session session, int[] draft, int[] batch, int pos, Scores scores) {
        float[][] rows = backend.forwardRows(batch, pos, null);
        if (rows.length != batch.length) {
            throw new IllegalStateException("verification of " + batch.length + " tokens returned " + rows.length
                    + " logits rows");
        }
        if (verifier != null) {
            return verifierVerify(verifier, roundsRng, sampler, session, draft, rows, scores);
        }
        return matchVerify(sampler, session, draft, i -> {
            Sampler.Candidates cur = sampler.apply(rows[i]);
            int sampledId = cur.id(cur.selected());
            scores.record(rows[i], cur, sampledId);
            return sampledId;
        });
    }

    /**
     * {@link VerifyMode#MATCH}: the longest prefix whose chain samples equal the draft (llama.cpp
     * common/sampling.cpp:736); the first mismatching row's sample, or the last row's, is the target's next token.
     *
     * @param rowToken the target's token at verify row {@code i}, taken in ascending row order
     */
    private static Verified matchVerify(Sampler sampler, StopCriteria.Session session, int[] draft,
                                        IntUnaryOperator rowToken) {
        int k = draft.length;
        int nAccepted = 0;
        int emitted = 0;
        int next = -1;
        StopReason reason = null;
        for (int i = 0; i <= k; i++) {
            if (i > 0 && session.cancelled()) {
                reason = StopReason.CANCELLED;
                break;
            }
            next = rowToken.applyAsInt(i);
            sampler.accept(next);
            emitted++;
            reason = session.offer(next);
            if (reason != null || i == k || next != draft[i]) {
                break;
            }
            nAccepted++;
        }
        return new Verified(nAccepted, emitted, next, reason);
    }

    /** A non-MATCH verifier decides the round over the rows' processed distributions ({@link VerifyRound}). */
    private Verified verifierVerify(Verifier verifier, SplittableRandom roundsRng, Sampler sampler,
                                    StopCriteria.Session session, int[] draft, float[][] rows, Scores scores) {
        VerifyRound.Emitter emitter = new VerifyRound.Emitter() {
            @Override
            public boolean cancelled() {
                return session.cancelled();
            }

            @Override
            public StopReason offer(int token, float[] row, Verifier.Distribution processed) {
                scores.record(row, processed, token);
                return session.offer(token);
            }
        };
        VerifyRound round = new VerifyRound(sampler, emitter, draft, rows, 0, roundsRng, draftProbs(sampler));
        verifier.verify(round);
        round.finish(verifier);
        return new Verified(round.accepted(), round.emitted(), round.next(), round.reason());
    }

    /**
     * The drafter's proposal rows when it implements {@link DraftProbs}, or null for one-hot proposals. A null row
     * is a deterministic proposal (a greedy draw); a watermarked chain draws every proposal and must report it.
     *
     * @throws IllegalStateException when a watermarked draft's proposal is missing
     */
    private IntFunction<float[]> draftProbs(Sampler sampler) {
        if (!(drafter instanceof DraftProbs)) {
            return null;
        }
        DraftProbs probs = (DraftProbs) drafter;
        boolean required = sampler.watermarked();
        return position -> {
            float[] row = probs.draftProbs(position);
            if (row == null && required) {
                throw new IllegalStateException("the watermarked drafter returned no proposal probabilities for"
                        + " position " + position);
            }
            return row;
        };
    }

    /**
     * The verify mode's verifier for one generation (stateless, shared across the generation's rounds), or null for
     * {@link VerifyMode#MATCH}, whose rounds sample through the chain itself ({@link VerifyRound#verifierFor}).
     *
     * @throws IllegalArgumentException      when the chain ends in {@code adaptive_p}: it folds only its own draw
     *                                       into its moving average, which a verifier's emissions replace, so only
     *                                       {@code MATCH} reproduces it (a mirostat chain, which adapts on the
     *                                       accepted token and builds no {@code adaptive_p}, runs in every mode)
     */
    private Verifier verifierFor(SamplerConfig config, Sampler sampler) {
        double[] rates = verifyMode != VerifyMode.SYNTHETIC ? null : syntheticRates != null
                ? syntheticRates : SyntheticVerifier.ratesForLength(syntheticLength, draftMax);
        return VerifyRound.verifierFor(verifyMode, config, sampler, flyWindow, flyEntropyThreshold, flyEntropyTopK,
                rates);
    }

    /**
     * The longest proposal the round can verify and then drop: bounded by {@code draftMax}, the free context after
     * {@code token}, one ubatch, then the largest {@code k} for which the memory keeps every
     * sliding window after storing {@code k + 1} positions and dropping {@code k}
     * ({@link ModelMemory#ringsKeepWindow}, monotone in {@code k}). DSV4 compressed caches bound {@code k} by their
     * compressor rollback snapshots ({@link MemoryPlan#rollbackSnapshots()}): the verify stores {@code k + 1 <= ubatch}
     * positions in one step, whose snapshot planes cover dropping any {@code k} of them. A round that rewinds
     * recurrent state per rejected draft ({@code stateRollback}) also needs the memory to snapshot that state —
     * 0 without snapshot shadows; the recurrent tree round replays the accepted path from the unchanged committed
     * state instead, so it drafts over a memory that cannot snapshot. The token budget does not bound {@code k}: a
     * round that emits past the budget stops inside the verify, as in llama.cpp and vLLM.
     */
    private int draftLength(ModelMemory memory, int pos, int nCtx, boolean stateRollback) {
        MemoryPlan plan = memory.plan();
        int k = Math.min(draftMax, nCtx - pos - 1);
        k = Math.min(k, plan.ubatch() - 1);
        if (plan.hasCompressed()) {
            k = Math.min(k, plan.rollbackSnapshots());
        }
        if (k <= 0 || stateRollback && memory.hasRecurrentState() && !memory.canSnapshot()) {
            return 0;
        }
        if (memory.ringsKeepWindow(k + 1, k)) {
            return k;
        }
        int lo = 0;
        int hi = k;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (memory.ringsKeepWindow(mid + 1, mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private static int[] checkDraft(int[] draft, int k) {
        if (draft == null) {
            throw new IllegalStateException("the drafter returned no proposal array");
        }
        if (draft.length > k) {
            throw new IllegalStateException("the drafter proposed " + draft.length + " tokens, at most " + k
                    + " were asked");
        }
        for (int i = 0; i < draft.length; i++) {
            if (draft[i] < 0) {
                throw new IllegalStateException("the drafter proposed negative token id " + draft[i]);
            }
        }
        return draft;
    }

    /** {@code config} with a whole-context penalty window bounded to the context length. */
    private static SamplerConfig samplerConfig(SamplerConfig config, int nCtx) {
        return config.penaltyLastN() == SamplerConfig.PENALTY_WHOLE_CONTEXT
                ? config.toBuilder().penaltyLastN(nCtx).build() : config;
    }

    private static void requirePrompt(int[] prompt) {
        if (prompt.length == 0) {
            throw new IllegalArgumentException("prompt must not be empty");
        }
        for (int i = 0; i < prompt.length; i++) {
            if (prompt[i] < 0) {
                throw new IllegalArgumentException("prompt token " + i + " has negative id " + prompt[i]);
            }
        }
    }

    @Override
    public String toString() {
        return "SpeculativeDecoder{" + drafter + ", draftMax=" + draftMax + ", draftMin=" + draftMin + ", "
                + "verify=" + verifyMode + (verifyOnDevice ? "" : ", host verify") + (contextShift ? ", contextShift" : "") + ", " + reasoning
                + (samplingDefaults.isEmpty() ? "" : ", " + samplingDefaults) + '}';
    }

    public static final class Builder {
        private final ForwardBackend backend;
        private final DraftModel drafter;
        private final GGUFTokenizer tokenizer;
        private ReasoningFilter reasoning = ReasoningFilter.none();
        private LoraRegistry adapters;
        private int draftMax = 3;
        private int draftMin;
        private Consumer<Stats> statsSink = s -> { };
        private SamplingDefaults samplingDefaults = SamplingDefaults.NONE;
        private LogitsProcessors logitsProcessors = LogitsProcessors.empty();
        private boolean prefixReuse = true;
        private boolean contextShift;
        private VerifyMode verifyMode = VerifyMode.MATCH;
        private int flyWindow;
        private double flyEntropyThreshold = 0.3;
        private int flyEntropyTopK = 3;
        private double[] syntheticRates;
        private Double syntheticLength;
        private boolean verifyOnDevice = true;
        private int treeNodes = SpeculativeOptions.DEFAULT_TREE_NODES;
        private boolean treeRecurrent = SpeculativeOptions.DEFAULT_TREE_RECURRENT;
        private com.eonmux.neuralfree.gguf.arch.RopeType ropeType = com.eonmux.neuralfree.gguf.arch.RopeType.NONE;
        private QuantizedView quantizedView;
        private com.eonmux.neuralfree.gguf.hparams.MropeTextPositions textPositions =
                com.eonmux.neuralfree.gguf.hparams.MropeTextPositions.PPP0;

        private Builder(ForwardBackend backend, DraftModel drafter, GGUFTokenizer tokenizer) {
            this.backend = Objects.requireNonNull(backend, "backend");
            this.drafter = Objects.requireNonNull(drafter, "drafter");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        }

        /** Reasoning parser of the output (default {@link ReasoningFilter#none()}). */
        public Builder reasoning(ReasoningFilter v) { this.reasoning = Objects.requireNonNull(v, "reasoning"); return this; }

        /**
         * The target's per-request adapters ({@link EngineModel#loraRegistry()}), which the target's passes run
         * under (vLLM serves LoRA with speculative decoding; the drafter runs none). Null runs every pass in slot 0.
         */
        public Builder adapters(LoraRegistry v) { this.adapters = v; return this; }

        /** Most tokens proposed per round (default 3; {@code neuralfree.draft.max}). */
        public Builder draftMax(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("draft max must be at least 1, got " + v);
            }
            this.draftMax = v;
            return this;
        }

        /** Shorter proposals are not verified (default 0; {@code neuralfree.draft.min}). */
        public Builder draftMin(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("draft min must not be negative, got " + v);
            }
            this.draftMin = v;
            return this;
        }

        /** The fixed tree verify width in nodes (default {@link SpeculativeOptions#DEFAULT_TREE_NODES}). */
        public Builder treeNodes(int v) {
            if (v < 1 || v > SpeculativeOptions.MAX_TREE_NODES) {
                throw new IllegalArgumentException("the tree verify width must lie between 1 and "
                        + SpeculativeOptions.MAX_TREE_NODES + " (one long names a verify batch's rows), got " + v);
            }
            this.treeNodes = v;
            return this;
        }

        /**
         * Whether the tree verify covers targets with recurrent layers ({@code neuralfree.draft.tree.recurrent};
         * default {@link SpeculativeOptions#DEFAULT_TREE_RECURRENT}).
         */
        public Builder treeRecurrent(boolean v) {
            this.treeRecurrent = v;
            return this;
        }

        /** The target's rope type, which decides how a verify batch carries its depth positions. */
        public Builder ropeType(com.eonmux.neuralfree.gguf.arch.RopeType v) {
            this.ropeType = Objects.requireNonNull(v, "ropeType");
            return this;
        }

        /**
         * The quantized target view of two-level verification ({@code neuralfree.draft.tlv}); null (the default)
         * runs every round on the target alone. The view runs over the same memory as {@code backend} — as a
         * quantized copy on its own backend, or as the nested W12 view over {@code backend} itself, whose Q
         * round opens the view's Q scope. The caller owns and closes the view.
         */
        public Builder quantizedView(QuantizedView v) {
            this.quantizedView = v;
            return this;
        }

        /**
         * The target's text-token section pattern for a multi-section rope's verify rows (default
         * {@link com.eonmux.neuralfree.gguf.hparams.MropeTextPositions#PPP0}); unused for a single-section rope.
         */
        public Builder textPositions(com.eonmux.neuralfree.gguf.hparams.MropeTextPositions v) {
            this.textPositions = Objects.requireNonNull(v, "textPositions");
            return this;
        }

        /** Receives each generation's {@link Stats} before its result is built (default: discarded). */
        public Builder statsSink(Consumer<Stats> v) { this.statsSink = Objects.requireNonNull(v, "statsSink"); return this; }

        /**
         * The target model's sampling defaults applied to every request (default {@link SamplingDefaults#NONE};
         * {@link SamplingDefaults#forModel} gives the target's under an engine configuration).
         */
        public Builder samplingDefaults(SamplingDefaults v) {
            this.samplingDefaults = Objects.requireNonNull(v, "samplingDefaults");
            return this;
        }

        /** Keep the common prompt prefix between the target's generations (default true; {@code neuralfree.prefix.reuse}). */
        public Builder prefixReuse(boolean v) { this.prefixReuse = v; return this; }

        /** The custom logits processors every verification consults (default none). */
        public Builder logitsProcessors(LogitsProcessors v) {
            this.logitsProcessors = Objects.requireNonNull(v, "logitsProcessors");
            return this;
        }

        /**
         * Shift the target's context instead of stopping when it is full, as {@link GenerationLoop} does (default
         * false; {@code neuralfree.context.shift}).
         */
        public Builder contextShift(boolean v) { this.contextShift = v; return this; }

        /** The verification mode (default {@link VerifyMode#MATCH}). */
        public Builder verifyMode(VerifyMode v) {
            if (v == null) {
                throw new IllegalArgumentException("verify mode must not be null");
            }
            this.verifyMode = v;
            return this;
        }

        /**
         * Whether {@code MATCH} rounds of a plain argmax chain without a logprobs capture select their tokens on the
         * target's device ({@link ForwardBackend#forwardRowsGreedy}, {@link ForwardBackend#decodeGreedyOnDevice};
         * default true; {@code neuralfree.draft.verify.device}).
         */
        public Builder verifyOnDevice(boolean v) { this.verifyOnDevice = v; return this; }

        /**
         * {@code fly_window_size}: FLy's future native acceptance decisions to check, or 0 for vLLM's default of
         * {@code min(6, k - 1)} of the round's proposal (default 0; only {@link VerifyMode#FLY} reads it).
         */
        public Builder flyWindow(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("fly window must not be negative, got " + v);
            }
            this.flyWindow = v;
            return this;
        }

        /** {@code fly_entropy_threshold}: the entropy a deferral needs (default 0.3; only {@link VerifyMode#FLY}). */
        public Builder flyEntropyThreshold(double v) {
            if (!(v >= 0.0)) {
                throw new IllegalArgumentException("fly entropy threshold must not be negative, got " + v);
            }
            this.flyEntropyThreshold = v;
            return this;
        }

        /** {@code fly_entropy_top_k}: the processed probabilities the entropy covers (default 3; only {@link VerifyMode#FLY}). */
        public Builder flyEntropyTopK(int v) {
            if (v <= 0) {
                throw new IllegalArgumentException("fly entropy top-k must be positive, got " + v);
            }
            this.flyEntropyTopK = v;
            return this;
        }

        /**
         * {@code synthetic_acceptance_rates}: the unconditional per-position acceptance rates of
         * {@link VerifyMode#SYNTHETIC}; mutually exclusive with {@link #syntheticAcceptanceLength(double)}, which
         * vLLM's minimum-variance schedule resolves to these rates.
         */
        public Builder syntheticAcceptanceRates(double... rates) {
            new SyntheticVerifier(rates); // validation
            this.syntheticRates = rates.clone();
            return this;
        }

        /** @{@code synthetic_acceptance_length}: the target mean acceptance length (a shorthand for the rates). */
        public Builder syntheticAcceptanceLength(double length) {
            SyntheticVerifier.ratesForLength(length, draftMax); // validation against the proposal budget
            this.syntheticLength = length;
            return this;
        }

        /**
         * The configured unconditional rates: the explicit {@code synthetic_acceptance_rates}, or the
         * {@code synthetic_acceptance_length} resolved over {@code draftMax} proposal positions (vLLM's
         * minimum-variance schedule).
         */
        private double[] resolvedSyntheticRates() {
            return syntheticRates != null
                    ? syntheticRates
                    : SyntheticVerifier.ratesForLength(syntheticLength, draftMax);
        }

        /**
         * Returns the decoder.
         *
         * @throws IllegalStateException when {@link VerifyMode#SYNTHETIC} is selected without the opt-in property
         *                               {@code neuralfree.spec.synthetic}
         * @throws IllegalArgumentException when both synthetic settings are given, or none outside synthetic mode,
         *                                  when the synthetic settings are themselves invalid, or when the draft
         *                                  lengths disagree
         */
        public SpeculativeDecoder build() {
            boolean synthetic = verifyMode == VerifyMode.SYNTHETIC;
            if (syntheticRates != null && syntheticLength != null) {
                throw new IllegalArgumentException("synthetic_acceptance_rates and synthetic_acceptance_length"
                        + " are mutually exclusive; configure one");
            }
            if (synthetic && syntheticRates == null && syntheticLength == null) {
                throw new IllegalArgumentException("synthetic verification needs synthetic_acceptance_rates or"
                        + " synthetic_acceptance_length");
            }
            if (!synthetic && (syntheticRates != null || syntheticLength != null)) {
                throw new IllegalArgumentException("synthetic_acceptance_rates and synthetic_acceptance_length"
                        + " apply only to synthetic verification");
            }
            if (synthetic) {
                if (!Boolean.getBoolean(SyntheticVerifier.PROP_ENABLED)) {
                    throw new IllegalStateException(SyntheticVerifier.PROP_ENABLED + "=true is required for"
                            + " synthetic verification");
                }
                new SyntheticVerifier(resolvedSyntheticRates()); // validates the resolved rates
            }
            return new SpeculativeDecoder(this);
        }
    }
}
