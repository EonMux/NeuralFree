package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.ConstraintDeadEndException;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.StopCriteria;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.beam.BeamSearch;
import com.eonmux.neuralfree.gguf.engine.beam.BeamSearchEngine;
import com.eonmux.neuralfree.gguf.engine.beam.BeamSearchOutput;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;

import io.vertx.core.http.HttpServerRequest;
import jakarta.ws.rs.WebApplicationException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The serving beam search (vLLM's online {@code beam_search} over the API): the request's beams run as
 * {@link EngineService#submit} generations over the replica pool. Every step submits the live beams' full token
 * sequences ({@link BeamSearchEngine#stepRequest}: one token, the top {@code 2 * beam_width} raw log
 * probabilities) and waits for their rows within the request's {@link ServerOptions#requestTimeout} budget. Each
 * submission routes on its own prefix-cache rows, and a replica's cache forks a step's beams from their shared
 * prompt pages, so a step pays one prompt prefill and {@code beam_width} cheap extends. The client going away
 * cancels the search at the next step boundary ({@link ClientLink}).
 *
 * <p>Errors map to HTTP like a generation batch: a refused submission by name is a 400, a refusal on the engine's
 * serving state a 503, full replica queues at the first step a 429 (the search is one request under the queue limit,
 * so a later step is never refused by it), a request that outlived its budget a 503 naming it, anything else a 500.
 */
final class BeamSearchRunner {

    /** The search's outcome: the ranked beams and the prefix-cache rows the steps' submissions reused. */
    record Run(BeamSearchOutput output, long cachedRows) { }

    private final EngineService service;
    private final ServerOptions options;
    private final long deadline;
    private final GenerationRequest step;
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicLong cachedRows = new AtomicLong();
    private final RequestTracing.Trace trace;
    /** The search's standing under the queue limit: its first step admits it, the later steps are never refused. */
    private final QueueAdmission admission = new QueueAdmission();
    /** The submissions of the most recent step, the trace's engine handles. */
    private volatile List<EngineService.Submission> lastStep = List.of();

    private BeamSearchRunner(EngineService service, RequestTracing.Trace trace, GenerationRequest step) {
        this.service = service;
        this.options = service.options();
        this.deadline = System.nanoTime() + options.requestTimeout().toNanos();
        this.trace = trace;
        this.step = step;
    }

    /**
     * Runs the beam search of {@code request} over the tokenized {@code prompt} (vLLM's {@code beam_width} is
     * {@code request.beamWidth()}, its {@code max_tokens} the request's {@code max_new_tokens} budget) and returns
     * the ranked beams. The search decodes every beam's text through the request's rendering settings.
     *
     * @throws EngineUnavailableException        when the service stops serving between the caller's check and a
     *                                           submission
     * @throws jakarta.ws.rs.WebApplicationException on a refused submission (400), a serving-state refusal (503),
     *                                           full replica queues (429), a spent request budget (503) or a
     *                                           failed step (500)
     */
    static Run run(EngineService service, GenerationRequest request, int[] prompt, HttpServerRequest client,
                   String requestId) {
        GenerationRequest step = BeamSearchEngine.stepRequest(request);
        RequestTracing.Trace trace = RequestTracing.begin(service, requestId, client, request.beamWidth(), step);
        BeamSearchRunner runner = new BeamSearchRunner(service, trace, step);
        ClientLink.onClose(client, () -> runner.cancelled.set(true));
        BeamSearch.Settings settings = BeamSearch.settings()
                .beamWidth(request.beamWidth())
                .maxSteps(request.unlimited() ? Integer.MAX_VALUE : request.maxNewTokens())
                .lengthPenalty(request.lengthPenalty())
                .ignoreEos(request.ignoreEos())
                .endOfGenerationIds(service.tokenizer().specialTokens().eogIds())
                .stopTokenIds(request.stopTokenIds())
                .allowedTokenIds(request.allowedTokenIds())
                .maxSequenceLength(service.maxModelLength())
                .cancelled(runner.cancelled::get)
                .renderer(output -> StopCriteria.of(request, service.tokenizer(), ReasoningFilter.none())
                        .renderPrompt(output))
                .build();
        BeamSearchOutput output;
        try {
            output = BeamSearch.run(settings, prompt, runner::step);
        } catch (RuntimeException e) {
            runner.finish(e);
            throw e;
        }
        runner.finish(null);
        return new Run(output, runner.cachedRows.get());
    }

    /** One scored step: every live sequence submitted to the pool, one await each, one capture row each. */
    private List<TokenScore[]> step(List<int[]> sequences) {
        checkBudget();
        List<EngineService.Submission> submissions = new ArrayList<>(sequences.size());
        try {
            for (int[] sequence : sequences) {
                RequestTracing.Trace t = trace;
                if (t != null) {
                    t.registered(submissions.size(), sequence.length);
                }
                submissions.add(service.submit(sequence, step, ReasoningFilter.contentTo((tokenId, text) -> { }),
                        null, null, null, null, admission));
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        } catch (EngineUnavailableException | EngineSaturatedException e) {
            throw unavailable(e);
        }
        lastStep = submissions;
        List<TokenScore[]> rows = new ArrayList<>(submissions.size());
        for (EngineService.Submission submission : submissions) {
            rows.add(rowOf(submission));
        }
        RequestTracing.Trace t = trace;
        if (t != null) {
            for (int i = 0; i < submissions.size(); i++) {
                t.stepped(i);
            }
        }
        return rows;
    }

    /** The step's capture row: the sampled token first, then its best alternatives by raw log probability. */
    private TokenScore[] rowOf(EngineService.Submission submission) {
        boolean done;
        try {
            long left = deadline - System.nanoTime();
            done = left > 0 && submission.await(left, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        }
        if (!done) {
            cancelled.set(true);
            throw budget();
        }
        if (submission.refused() instanceof EngineUnavailableException e) {
            throw unavailable(e);
        }
        if (submission.refused() != null) {
            throw ServerJson.badRequest(String.valueOf(submission.refused().getMessage()));
        }
        BatchSequence sequence = submission.sequence();
        if (sequence == null) {
            throw ServerJson.error(500, "the request never registered with the engine", "server_error");
        }
        if (sequence.state() == BatchSequence.State.FAILED) {
            if (sequence.error() instanceof ConstraintDeadEndException
                    || sequence.error() instanceof UnsupportedOperationException refused) {
                throw ServerJson.badRequest(String.valueOf(sequence.error().getMessage()));
            }
            throw ServerJson.error(500, String.valueOf(sequence.error()), "server_error");
        }
        GenerationResult result = sequence.result();
        List<TokenScore[]> rows = result == null ? List.of() : result.logprobs();
        if (rows.isEmpty() || rows.get(0).length == 0) {
            throw ServerJson.error(500, "the beam search's step captured no logprobs", "server_error");
        }
        cachedRows.addAndGet(Math.max(0, sequence.reusedPrefix()));
        return rows.get(0);
    }

    /** Refuses a step past the request budget: the search cancels at once. */
    private void checkBudget() {
        if (System.nanoTime() - deadline > 0) {
            cancelled.set(true);
            throw budget();
        }
    }

    private WebApplicationException budget() {
        return ServerJson.error(503, "the request outlived its " + options.budgetText() + " budget",
                "unavailable_error");
    }

    /** Ends the request's trace with {@code forced}, or normally over the last step's submissions. */
    private void finish(Throwable forced) {
        RequestTracing.Trace t = trace;
        if (t == null) {
            return;
        }
        List<EngineService.Submission> last = lastStep;
        BatchSequence[] sequences = new BatchSequence[last.size()];
        Throwable[] refusals = new Throwable[last.size()];
        for (int i = 0; i < last.size(); i++) {
            sequences[i] = last.get(i).sequence();
            refusals[i] = last.get(i).refused();
        }
        t.finish(forced, sequences, refusals);
    }

    /** The answer to a refusal on the serving state (503) or on full replica queues (429 with Retry-After). */
    private static WebApplicationException unavailable(IllegalStateException e) {
        return ServerJson.refusal(e);
    }
}
