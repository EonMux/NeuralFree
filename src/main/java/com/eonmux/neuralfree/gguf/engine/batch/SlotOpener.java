package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LookaheadRun;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.engine.SequenceRun;
import com.eonmux.neuralfree.gguf.engine.StopCriteria;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.sampling.ReasoningBudgetStage;
import com.eonmux.neuralfree.gguf.engine.sampling.RequestStages;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Opens the slot of an admitted {@link BatchEngine} submission with the sequence's own token pipeline, as
 * {@code GenerationLoop} builds it: the model's sampling defaults fill every unset request setting, the stop
 * criteria derive the end-of-generation set, and the sampler runs the request's grammar stage, its request stages
 * and its reasoning budget. The reasoning filter is the submission's, else the engine's, selected for the request's
 * chat template variables ({@link ReasoningFilter#forRequest}).
 */
final class SlotOpener {

    /** The request a pooling slot's run opens over: it never samples, so only its shape matters. */
    private static final GenerationRequest POOLING_REQUEST = GenerationRequest.builder()
            .maxNewTokens(1)
            .sampler(SamplerConfig.greedyDecoding())
            .build();

    private final GGUFTokenizer tokenizer;
    /** The sampler's view of {@link #tokenizer}, shared by every request so its piece caches build once. */
    private final SamplerVocab samplerVocab;
    private final ReasoningFilter reasoning;
    private final SamplingDefaults samplingDefaults;
    private final BatchState state;
    /** The engine's speculative settings, or null when it decodes plainly. */
    private final BatchSpec spec;
    /** The engine's custom logits processors, empty when it configures none. */
    private final LogitsProcessors logitsProcessors;

    SlotOpener(GGUFTokenizer tokenizer, ReasoningFilter reasoning, SamplingDefaults samplingDefaults,
               BatchState state, BatchSpec spec, LogitsProcessors logitsProcessors) {
        this.tokenizer = tokenizer;
        this.samplerVocab = Sampler.vocab(tokenizer);
        this.reasoning = reasoning;
        this.samplingDefaults = samplingDefaults;
        this.state = state;
        this.spec = spec;
        this.logitsProcessors = logitsProcessors;
    }

    /**
     * Opens a slot for an admitted submission on its claimed sequence id: an encoder-decoder request first encodes
     * its source into the sequence's cross caches (the encoding time opens the prompt phase), then the sequence's own
     * stop criteria, sampler and run, its pending fork, and the admission reported on the handle. A pooling request
     * opens without the sampling pipeline (see {@link #openPooling}).
     */
    Slot open(PrefixClaims.Claim claim, Submission sub) {
        if (sub.pooling != null) {
            return openPooling(claim, sub);
        }
        long encodeNanos = encode(claim.seq, sub);
        GenerationRequest effective = sub.request.withModelDefaults(samplingDefaults);
        ReasoningFilter filter = (sub.filter != null ? sub.filter : reasoning)
                .forRequest(effective.templateVariables());
        StopCriteria stop = StopCriteria.of(effective, tokenizer, filter);
        String rendered = sub.renderedPrompt != null || filter.mode() == ReasoningFilter.Mode.NONE
                ? sub.renderedPrompt : stop.renderPrompt(sub.renderTokens());
        SamplerVocab vocab = samplerVocab;
        List<SamplerStage> stages = grammarAndRequestStages(sub, effective, vocab);
        ReasoningBudgetStage budget = ReasoningBudgetStage.of(filter, effective, vocab, rendered);
        if (budget != null) {
            stages.add(budget);
        }
        Sampler sampler = sampler(effective, stop, stages, vocab);
        SequenceRun run = SequenceRun.over(effective, tokenizer)                .stopCriteria(stop)
                .sampler(sampler)
                .sink(sub.sink)
                .renderedPrompt(rendered)
                .build();
        Slot slot = new Slot(claim.seq, sub, run, state.nCtx);
        armSpeculation(slot, effective, sampler);
        if (state.lookahead != null) {
            slot.lookahead = new LookaheadRun(state.sequences, claim.seq, state.lookahead, run,
                    tokenizer.vocabSize());
        }
        slot.prefillNanos = encodeNanos;
        slot.forkFrom = claim.forkFrom;
        slot.forkLen = claim.forkLen;
        if (claim.chain != null) {
            slot.chain = claim.chain;
            slot.lastCheckpoint = sub.reused;
        }
        state.attach(slot);
        sub.handle.admit(claim.seq);
        if (budget != null) {
            sub.handle.armed(budget);
        }
        return slot;
    }

    /**
     * Arms a speculating sequence's verify state: the verifier of the engine's verify mode (null for {@code MATCH}),
     * the verify and draft streams seeded by the sampler's seed (as {@code SpeculativeDecoder} seeds its rounds),
     * and the temperature a probabilistic drafter drafts with (0 under a greedy chain).
     */
    private void armSpeculation(Slot slot, GenerationRequest effective, Sampler sampler) {
        if (spec == null) {
            return;
        }
        slot.verifier = spec.verifierFor(effective.sampler(), sampler);
        slot.verifyRng = new SplittableRandom(sampler.seed());
        slot.draftRng = slot.verifyRng.split();
        slot.draftTemperature = sampler.greedy() ? 0f : effective.sampler().temperature();
    }

    /**
     * Opens a pooling request's slot: its run never samples, so it carries a greedy sampler and the request's
     * cancellation hook alone — the pieces a cancel path reads while the pooling prompt prefills.
     */
    private Slot openPooling(PrefixClaims.Claim claim, Submission sub) {
        SamplerVocab vocab = samplerVocab;
        SequenceRun run = SequenceRun.over(POOLING_REQUEST, tokenizer)
                .sampler(Sampler.create(SamplerConfig.greedyDecoding(), new int[0], vocab))
                .build();
        Slot slot = new Slot(claim.seq, sub, run, state.nCtx);
        slot.forkFrom = claim.forkFrom;
        slot.forkLen = claim.forkLen;
        if (claim.chain != null) {
            slot.chain = claim.chain;
        }
        state.attach(slot);
        sub.handle.admit(claim.seq);
        return slot;
    }

    /**
     * Encodes an encoder-decoder submission's source into sequence {@code seq}'s cross caches and returns the time it
     * took, which counts as prompt time; 0 for a decoder-only submission.
     */
    long encode(int seq, Submission sub) {
        if (sub.source == null) {
            return 0L;
        }
        long t0 = System.nanoTime();
        state.encoder.encodeSequence(seq, sub.source);
        long nanos = System.nanoTime() - t0;
        state.stats.promptNanos += nanos;
        return nanos;
    }

    /**
     * The submission's grammar stage, then the request's stages: the constraint head applies them first, the
     * grammar before the request's masks so a completed constraint's end-of-generation tokens survive the
     * min-tokens ban (the ban yields when nothing else remains), and the reasoning budget, appended last, keeps
     * the final say over a masked set.
     */
    private static List<SamplerStage> grammarAndRequestStages(Submission sub, GenerationRequest effective,
                                                              SamplerVocab vocab) {
        List<SamplerStage> stages = new ArrayList<>();
        if (sub.grammar != null) {
            stages.add(sub.grammar.gated()
                    ? new TriggeredGrammarStage(sub.grammar.grammar(), sub.grammar.triggers(), vocab)
                    : new GrammarStage(sub.grammar.grammar(), vocab));
        }
        stages.addAll(RequestStages.of(effective, vocab));
        return stages;
    }

    /** The sequence's sampler over {@code stages}, its custom logits processors and the plain chain when there are
     *  none of either. */
    private Sampler sampler(GenerationRequest effective, StopCriteria stop, List<SamplerStage> stages,
                            SamplerVocab vocab) {
        SamplerConfig config = samplerConfig(effective.sampler(), state.nCtx);
        List<SamplerStage> processors = logitsProcessors.stages(effective);
        if (spec != null) {
            // the target role never routes: the draft role samples the proposals
            if (stages.isEmpty() && processors.isEmpty()) {
                return Sampler.createForSpeculativeTarget(config, stop.bannedTokens(), effective.logitBias(), vocab);
            }
            return Sampler.createForSpeculativeTarget(config, stop.bannedTokens(), effective.logitBias(), stages,
                    processors, vocab);
        }
        if (stages.isEmpty() && processors.isEmpty()) {
            return Sampler.create(config, stop.bannedTokens(), effective.logitBias(), vocab);
        }
        return Sampler.create(config, stop.bannedTokens(), effective.logitBias(), stages, processors, vocab);
    }

    /** {@code config} with a whole-context penalty window bounded to the context length (as GenerationLoop). */
    private static SamplerConfig samplerConfig(SamplerConfig config, int nCtx) {
        return config.penaltyLastN() == SamplerConfig.PENALTY_WHOLE_CONTEXT
                ? config.toBuilder().penaltyLastN(nCtx).build() : config;
    }
}
