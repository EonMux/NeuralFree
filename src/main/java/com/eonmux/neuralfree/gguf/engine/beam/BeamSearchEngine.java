package com.eonmux.neuralfree.gguf.engine.beam;

import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.StopCriteria;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Beam search over the batch engine (vLLM's {@code LLM.beam_search}): every step submits the live beams as their
 * full token sequences ({@code max_tokens=1}, {@code logprobs=2 * beam_width} raw), drains once, and feeds the
 * captured rows to {@link BeamSearch}. The beams of a step share their prompt pages through the engine's prefix
 * cache, whose copy-on-write pages fork each candidate's KV from the parent — the KV sharing beam search needs,
 * with no new fork API. One search at a time; the engine is single-purpose and closes its {@link BatchEngine}.
 *
 * <p>The search serves plain sampling: vLLM's beam search runs without stop strings, bad words, grammar, logit
 * biases, prompt logprobs, explicit logprob token ids, min tokens, repetition detection, reasoning budget or
 * self-extend, and the request features are refused by name. The step requests keep the engine's end-of-generation
 * handling — a sampled end token still lands in its capture row — so every end decision is made host-side over the
 * rows, exactly as vLLM scores its candidates.
 */
public final class BeamSearchEngine implements AutoCloseable {

    private final EngineModel model;
    private final BatchEngine engine;

    private BeamSearchEngine(EngineModel model, BatchEngine engine) {
        this.model = model;
        this.engine = engine;
    }

    /**
     * A beam-search engine over {@code model} with the batch engine {@code config} builds.
     *
     * @throws UnsupportedOperationException when the model is an encoder-decoder (every step re-submits a full
     *                                       decoder sequence, so only decoder-only models serve) or cannot run
     *                                       batched generation
     */
    public static BeamSearchEngine forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        if (model.supportClass() == SupportClass.ENC_DEC) {
            throw new UnsupportedOperationException(model.architecture() + " is an encoder-decoder model; beam"
                    + " search re-submits every beam's full sequence each step and serves decoder-only models");
        }
        return new BeamSearchEngine(model, BatchEngine.forModel(model, config));
    }

    /** The positions one full sequence may hold (the batch engine's context). */
    public int contextLength() {
        return engine.contextLength();
    }

    /**
     * Runs the beam search over the tokenized {@code prompt} with {@code request}'s beam width, length penalty,
     * sampler and token budget, and returns the best beams in ranking order ({@code request.beamWidth()} at most).
     * The request's own {@code logprobs} settings are ignored — the search captures its own rows at the width it
     * expands with.
     *
     * @throws IllegalArgumentException         for an empty prompt or one of the refused request features, named
     * @throws ContextFullException             when the prompt does not fit the context
     * @throws UnsupportedOperationException    when the request selects per-request adapters on a single-slot path
     * @throws RuntimeException                 when a step's sequence failed (the engine's error)
     */
    public BeamSearchOutput search(int[] prompt, GenerationRequest request) {
        Objects.requireNonNull(request, "request");
        requireSupported(request);
        StopCriteria criteria = StopCriteria.of(request, model.tokenizer(), ReasoningFilter.none());
        BeamSearch.Settings settings = BeamSearch.settings()
                .beamWidth(request.beamWidth())
                .maxSteps(request.unlimited() ? Integer.MAX_VALUE : request.maxNewTokens())
                .lengthPenalty(request.lengthPenalty())
                .ignoreEos(request.ignoreEos())
                .endOfGenerationIds(model.tokenizer().specialTokens().eogIds())
                .stopTokenIds(request.stopTokenIds())
                .allowedTokenIds(request.allowedTokenIds())
                .maxSequenceLength(engine.contextLength())
                .cancelled(request.cancellation())
                .renderer(output -> criteria.renderPrompt(output))
                .build();
        GenerationRequest step = stepRequest(request);
        return BeamSearch.run(settings, prompt, sequences -> step(step, sequences));
    }

    @Override
    public void close() {
        engine.close();
    }

    /**
     * The per-step request (vLLM's {@code base_sampling_params}): one token, the top {@code 2 * beam_width} raw
     * log probabilities, the request's sampler, allowed token ids, priority, cache salt and adapters. The step
     * carries no stop tokens, strings or ignore-eos — end tokens sample freely into the capture row, and the
     * host decides over the rows.
     */
    public static GenerationRequest stepRequest(GenerationRequest request) {
        return GenerationRequest.builder()
                .maxNewTokens(1)
                .sampler(request.sampler())
                .logprobs(2 * request.beamWidth())
                .logprobsMode(LogprobsMode.RAW_LOGPROBS)
                .allowedTokenIds(request.allowedTokenIds())
                .priority(request.priority())
                .cacheSalt(request.cacheSalt())
                .lora(request.lora())
                .build();
    }

    /** One scored step: every sequence submitted as its own batch sequence, one drain, one capture row each. */
    private List<TokenScore[]> step(GenerationRequest step, List<int[]> sequences) {
        List<BatchSequence> handles = new ArrayList<>(sequences.size());
        try {
            for (int[] sequence : sequences) {
                handles.add(engine.submit(sequence, step, (com.eonmux.neuralfree.gguf.TokenCallback) null));
            }
            engine.drain();
        } catch (RuntimeException e) {
            engine.cancelAll();
            throw e;
        }
        List<TokenScore[]> rows = new ArrayList<>(handles.size());
        for (BatchSequence handle : handles) {
            rows.add(captureRow(handle));
        }
        return rows;
    }

    /** The step's capture row: the sampled token first, then its best alternatives by raw log probability. */
    private TokenScore[] captureRow(BatchSequence handle) {
        if (handle.state() == BatchSequence.State.FAILED) {
            Throwable error = handle.error();
            if (error instanceof RuntimeException) {
                throw (RuntimeException) error;
            }
            throw new IllegalStateException("the beam search's step sequence failed", error);
        }
        GenerationResult result = handle.result();
        List<TokenScore[]> rows = result == null ? List.of() : result.logprobs();
        if (rows.isEmpty()) {
            throw new IllegalStateException("the beam search's step sequence of state " + handle.state()
                    + " captured no logprobs" + (handle.error() == null ? "" : ": " + handle.error()));
        }
        return rows.get(0);
    }

    /**
     * Refuses, by field name, the request features beam search cannot honor (vLLM's beam search serves plain
     * sampling; its own parameter set draws the line).
     *
     * @throws IllegalArgumentException naming the first unsupported request field
     */
    public static void requireSupported(GenerationRequest request) {
        if (!request.stopStrings().isEmpty()) {
            throw new IllegalArgumentException("stop: beam search does not support stop strings; drop them or"
                    + " sample plainly");
        }
        if (!request.badWords().isEmpty()) {
            throw new IllegalArgumentException("bad_words: beam search does not support bad words");
        }
        if (request.constraint() != null) {
            throw new IllegalArgumentException("constraint: beam search does not support a grammar constraint"
                    + " (a response_format spec)");
        }
        if (!request.logitBias().isEmpty()) {
            throw new IllegalArgumentException("logit_bias: beam search does not support logit biases");
        }
        if (request.namesLogprobTokenIds()) {
            throw new IllegalArgumentException("logprob_token_ids: beam search scores the top rows its candidates"
                    + " expand over");
        }
        if (!request.logitsProcessorArgs().isEmpty()) {
            throw new IllegalArgumentException("vllm_xargs: beam search expands the raw logits rows, so custom"
                    + " logits processors cannot shape its candidates; drop the arguments or sample plainly");
        }
        String promptRows = request.promptRowsField();
        if (promptRows != null) {
            throw new IllegalArgumentException(promptRows + ": beam search scores generated tokens only");
        }
        if (request.minTokens() > 0) {
            throw new IllegalArgumentException("min_tokens: beam search does not support a minimum before"
                    + " end-of-generation tokens");
        }
        if (request.nKeep() != 0) {
            throw new IllegalArgumentException("n_keep: beam search ends a beam at the context instead of shifting"
                    + " it");
        }
        if (request.nDiscard() != 0) {
            throw new IllegalArgumentException("n_discard: beam search ends a beam at the context instead of"
                    + " shifting it");
        }
        requireNoControls(request);
    }

    /** Refuses the sampling-time controls that shape tokens as they are emitted, which the search does not run. */
    private static void requireNoControls(GenerationRequest request) {
        if (request.repetitionDetection().enabled()) {
            throw new IllegalArgumentException("repetition_detection: beam search does not support the repetition"
                    + " stop");
        }
        if (request.reasoningBudget() != GenerationRequest.REASONING_BUDGET_DISABLED || request.reasoningControl()
                || !request.reasoningBudgetStartTags().isEmpty() || !request.reasoningBudgetEndTags().isEmpty()) {
            throw new IllegalArgumentException("reasoning_budget: beam search does not support a reasoning budget"
                    + " or reasoning control");
        }
        if (request.selfExtend()) {
            throw new IllegalArgumentException("grp_attn_n: beam search does not support self-extend position"
                    + " division");
        }
    }
}
