package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ConstraintDeadEndException;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;

import io.vertx.core.http.HttpServerRequest;
import jakarta.ws.rs.WebApplicationException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The generations of one HTTP request — its choices ({@code n}, prompt arrays) — submitted together so the batch
 * engine decodes them side by side. Their output arrives as {@link Event}s in the order the pump thread routed it:
 * text pieces, harmony tool recipients and, last for each choice, its {@link Done} event. The queue is unbounded
 * so a slow client never stalls the shared pump thread.
 *
 * <p>{@link #cancel()} (a client that went away) cancels every running choice through the requests' cancellation
 * hook. Errors map to HTTP: a refused submission (an oversize prompt, an unusable grammar, embedding rows the
 * backend does not take) is a 400, and so is a forward the backend refuses by name mid-run (an
 * {@link UnsupportedOperationException}: a request kind its batched forward does not serve) and a constraint
 * whose admitted set emptied mid-run ({@link ConstraintDeadEndException}, a grammar dead end or the request's
 * masks — {@code allowed_token_ids}, bad words — banning every id); a refusal on the
 * engine's serving state (a sleep, a pause, a lost model, a shutdown) is a 503; any other engine failure is a 500.
 */
final class GenerationBatch {

    /**
     * One generation to submit: a token prompt, or a media prompt ({@code ofMedia}: encoded media or precomputed
     * embedding rows between text chunks) — exactly one of the two. {@code reasoning} is the request's reasoning
     * filter (a {@code reasoning_format}), or null for the engine's.
     */
    record Job(int[] prompt, MediaPrompt media, GenerationRequest request, String renderedPrompt,
               ConstraintSpec constraint, ReasoningFilter reasoning) {
        Job {
            if ((prompt == null) == (media == null)) {
                throw new IllegalArgumentException("a job takes a token prompt or a media prompt, exactly"
                        + " one of the two");
            }
        }

        Job(int[] prompt, GenerationRequest request, String renderedPrompt, ConstraintSpec constraint,
            ReasoningFilter reasoning) {
            this(prompt, null, request, renderedPrompt, constraint, reasoning);
        }

        /** A job over a media prompt (chat media and embeds parts, completions {@code prompt_embeds}). */
        static Job ofMedia(MediaPrompt media, GenerationRequest request, String renderedPrompt,
                           ConstraintSpec constraint, ReasoningFilter reasoning) {
            return new Job(null, media, request, renderedPrompt, constraint, reasoning);
        }
    }

    /** Something one choice produced. */
    sealed interface Event permits Text, ToolRecipient, ContentFallback, Done {
        int choice();
    }

    /**
     * Routed text; {@code tokenId} released it (text released when generation ends counts as the last generated
     * token's, -1 when there was none), {@code logprobs} is that token's capture, or null, and {@code predicted}
     * counts the tokens generated through the releasing one.
     */
    record Text(int choice, ReasoningFilter.Channel channel, String text, TokenScore[] logprobs, int tokenId,
                int predicted) implements Event { }

    /** A harmony header addressed the tool {@code recipient}; its body follows on the tool-call channel. */
    record ToolRecipient(int choice, String recipient) implements Event { }

    /**
     * The choice ended without content and its reasoning follows again as content
     * ({@link ReasoningFilter.Sink#contentFallback}); an assembled message carries it as the content only.
     */
    record ContentFallback(int choice) implements Event { }

    /** The choice reached its terminal state; every event of it came before. */
    record Done(int choice) implements Event { }

    private final List<EngineService.Submission> submissions = new ArrayList<>();
    /** The prompt of each choice, as submitted, for {@code return_token_ids}' {@code prompt_token_ids}. */
    private final List<int[]> prompts = new ArrayList<>();
    private final BlockingQueue<Event> events = new LinkedBlockingQueue<>();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    /** Choices not yet terminal; the batch is done when it reaches 0. */
    private final AtomicInteger pending;
    private volatile Runnable onAllDone;
    private final ServerOptions options;
    /** The request's standing under the queue limit, shared by every choice. */
    private final QueueAdmission admission;
    private final long deadline;
    /** The batch's arrival, the origin of its choices' time to first token and end-to-end latency. */
    private final long arrivalNanos;
    /** The batch's request trace, null with tracing off; set once in {@link #start} before the submissions. */
    private RequestTracing.Trace trace;

    private GenerationBatch(ServerOptions options, int choices, QueueAdmission admission) {
        this.options = options;
        this.admission = admission;
        this.arrivalNanos = System.nanoTime();
        this.deadline = arrivalNanos + options.requestTimeout().toNanos();
        this.pending = new AtomicInteger(choices);
    }

    /**
     * Submits every job (job {@code i} is choice {@code i}) and links the batch to the client: its going away
     * cancels the batch ({@link ClientLink}). A service that stops serving between the caller's
     * {@link ServerJson#requireLoaded} check and a submission (a sleep, a rejecting pause or an unload landing in
     * between) refuses with the same 503 that check answers, and the choices submitted before it are cancelled.
     *
     * @param client the HTTP request, or null outside the HTTP layer
     * @throws jakarta.ws.rs.WebApplicationException 503 when the service refuses the submission on its serving state
     *                                               ({@link EngineUnavailableException}, which can flip between the
     *                                               route's entry check and this submit), is shutting down, or the
     *                                               caller is interrupted; 429 with {@code Retry-After} when every
     *                                               replica's queue is full ({@link EngineSaturatedException})
     */
    static GenerationBatch start(EngineService service, List<Job> jobs, HttpServerRequest client) {
        return start(service, jobs, client, null);
    }

    /**
     * {@link #start(EngineService, List, HttpServerRequest)} naming the request id the trace's
     * {@code gen_ai.request.id} carries (the id the response names); null when the surface has none.
     */
    static GenerationBatch start(EngineService service, List<Job> jobs, HttpServerRequest client, String requestId) {
        return start(service, jobs, client, requestId, true);
    }

    /**
     * {@link #start(EngineService, List, HttpServerRequest, String)} with {@code cancelOnClose} false leaving the
     * batch unlinked from its client: a resumable stream runs on when its client goes away and records into its
     * session instead ({@link StreamSession}).
     */
    static GenerationBatch start(EngineService service, List<Job> jobs, HttpServerRequest client, String requestId,
                                 boolean cancelOnClose) {
        return start(service, jobs, client, requestId, cancelOnClose, new QueueAdmission());
    }

    /**
     * {@link #start(EngineService, List, HttpServerRequest, String, boolean)} as part of the request
     * {@code admission} stands for under the queue limit: a server tool loop passes one admission to every round,
     * so a request admitted for its first round is never refused for a later one. The batch's choices are admitted
     * together: the limit checks the first and never refuses the others.
     */
    static GenerationBatch start(EngineService service, List<Job> jobs, HttpServerRequest client, String requestId,
                                 boolean cancelOnClose, QueueAdmission admission) {
        GenerationBatch batch = new GenerationBatch(service.options(), jobs.size(), admission);
        batch.trace = RequestTracing.begin(service, requestId, client, jobs.size(),
                jobs.isEmpty() ? null : jobs.get(0).request());
        if (cancelOnClose) {
            ClientLink.onClose(client, batch::cancel);
        }
        try {
            for (int i = 0; i < jobs.size(); i++) {
                batch.submit(service, jobs.get(i), i);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            batch.cancel();
            batch.finishTrace(e);
            throw ServerJson.error(503, "the server is shutting down", "unavailable_error");
        } catch (IllegalStateException e) {
            batch.cancel();
            batch.finishTrace(e);
            throw unavailable(e);
        }
        return batch;
    }

    /** Ends the batch's trace with {@code forced} when the start failed before the choices could report. */
    private void finishTrace(Throwable forced) {
        if (trace != null) {
            trace.finish(forced, sequences(), refusals());
        }
    }

    /** The choices' engine handles as they stand now, index-aligned with {@code refusals()}. */
    private BatchSequence[] sequences() {
        BatchSequence[] out = new BatchSequence[submissions.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = submissions.get(i).sequence();
        }
        return out;
    }

    /** The choices' submit refusals, null where the engine accepted the submission. */
    private Throwable[] refusals() {
        Throwable[] out = new Throwable[submissions.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = submissions.get(i).refused();
        }
        return out;
    }

    private void submit(EngineService service, Job job, int choice) throws InterruptedException {
        prompts.add(job.prompt());
        RequestTracing.Trace trace = this.trace;
        if (trace != null) {
            trace.registered(choice, job.prompt() != null ? job.prompt().length : job.media().nTokens());
        }
        GenerationRequest request = job.request().toBuilder().cancellation(cancelled::get).build();
        RequestLatencies.Choice latency = service.latencies().choice(arrivalNanos);
        ReasoningFilter.Sink sink = new ReasoningFilter.Sink() {
            /** Tokens generated so far and the latest one; the engine thread alone calls the sink. */
            private int predicted;
            private int last = -1;

            @Override
            public void stepped(int tokenId) {
                predicted++;
                last = tokenId;
                latency.token(System.nanoTime());
                if (trace != null) {
                    trace.stepped(choice);
                }
            }

            @Override
            public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                events.add(new Text(choice, channel, text, null, tokenId < 0 ? last : tokenId, predicted));
            }

            @Override
            public void accept(ReasoningFilter.Channel channel, int tokenId, String text, TokenScore[] logprobs) {
                events.add(new Text(choice, channel, text, logprobs != null && logprobs.length > 0 ? logprobs : null,
                        tokenId < 0 ? last : tokenId, predicted));
            }

            @Override
            public void toolCallStarted(String recipient) {
                events.add(new ToolRecipient(choice, recipient));
            }

            @Override
            public void contentFallback() {
                events.add(new ContentFallback(choice));
            }
        };
        Runnable onDone = () -> {
            latency.done(System.nanoTime());
            if (trace != null) {
                trace.done(choice);
            }
            events.add(new Done(choice));
            if (pending.decrementAndGet() == 0) {
                if (trace != null) {
                    trace.finish(null, sequences(), refusals());
                }
                Runnable callback = onAllDone;
                if (callback != null) {
                    callback.run();
                }
            }
        };
        EngineService.Submission submission = job.media() != null
                ? service.submitMedia(job.media(), request, sink, job.renderedPrompt(), job.constraint(),
                        job.reasoning(), onDone, admission)
                : service.submit(job.prompt(), request, sink, job.renderedPrompt(), job.constraint(),
                        job.reasoning(), onDone, admission);
        submissions.add(submission);
        latency.submitted(submission::sequence);
    }

    /**
     * Runs {@code callback} once every choice reached its terminal state, on the last finishing engine thread (a
     * registered completion cleans its tracking entry up there). Registered after {@link #start}, so a batch that is
     * already finished fires at once.
     */
    void onAllDone(Runnable callback) {
        this.onAllDone = callback;
        if (pending.get() <= 0) {
            callback.run();
        }
    }

    /** The engine handles of the choices that are still queued or running (not yet terminal). */
    List<BatchSequence> liveSequences() {
        List<BatchSequence> out = new ArrayList<>();
        for (EngineService.Submission s : submissions) {
            BatchSequence sequence = s.sequence();
            if (sequence != null && !sequence.finished()) {
                out.add(sequence);
            }
        }
        return out;
    }

    /** The engine handle of choice {@code choice}, or null when the engine refused it. */
    BatchSequence sequence(int choice) {
        return submissions.get(choice).sequence();
    }

    /** The number of choices. */
    int size() {
        return submissions.size();
    }

    /** A copy of choice {@code choice}'s prompt tokens, as submitted. */
    int[] promptTokenIds(int choice) {
        return prompts.get(choice).clone();
    }

    /** The prefix-cache tokens the engine reused for choice {@code choice}, or 0. */
    long cachedTokens(int choice) {
        BatchSequence sequence = submissions.get(choice).sequence();
        return sequence == null ? 0 : sequence.reusedPrefix();
    }

    /**
     * Choice {@code choice}'s {@code kv_transfer_params} answer (the prefill instance's transfer address, or the
     * decode instance's completion), or null when the request ran no transfer. Read once the choice is terminal.
     */
    Map<String, Object> kvTransferParams(int choice) {
        return submissions.get(choice).kvTransferParams();
    }

    /**
     * Choice {@code choice}'s per-prompt-token log probabilities once its prefill read the rows back
     * ({@code prompt_logprobs}), or null before that. A stream reads it when the choice's first event arrives,
     * after the prefill but before the result exists.
     */
    List<TokenScore[]> promptLogprobs(int choice) {
        BatchSequence sequence = submissions.get(choice).sequence();
        return sequence == null ? null : sequence.promptLogprobs();
    }

    /**
     * The next event, waiting at most {@code millis}; null on timeout. Events already queued are still delivered
     * after the budget ran out: the generation that produced them finished in time.
     *
     * @throws IllegalStateException once the request outlived its budget ({@link ServerOptions#requestTimeout})
     *                               and no queued event is left
     */
    Event next(long millis) throws InterruptedException {
        Event queued = events.poll();
        if (queued != null) {
            return queued;
        }
        if (System.nanoTime() - deadline > 0) {
            cancel();
            throw new IllegalStateException("the request outlived its " + options.budgetText() + " budget");
        }
        return events.poll(millis, TimeUnit.MILLISECONDS);
    }

    /**
     * The next queued event without waiting, or null when none is queued. Unlike {@link #next} it never checks the
     * request budget: a writer drains what is already queued after a {@link #next} returned an event.
     */
    Event poll() {
        return events.poll();
    }

    /** Waits for every choice's terminal state; the events stay queued for {@link #next}. */
    void awaitAll() {
        try {
            for (EngineService.Submission s : submissions) {
                long left = deadline - System.nanoTime();
                if (left <= 0 || !s.await(left, TimeUnit.NANOSECONDS)) {
                    cancel();
                    throw ServerJson.error(503, "the request outlived its " + options.budgetText() + " budget",
                            "unavailable_error");
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            cancel();
            throw ServerJson.error(503, "request interrupted", "unavailable_error");
        }
    }

    /** Cancels every choice still running. */
    void cancel() {
        cancelled.set(true);
    }

    /**
     * The result of a finished choice.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 when the engine refused the submission, its backend
     *                                               refused the forward by name, or the request's constraint
     *                                               admitted no candidate ({@link ConstraintDeadEndException},
     *                                               a grammar dead end or a mask dead end), 503 when the engine
     *                                               refused on its serving state, 500 when it failed otherwise
     */
    GenerationResult result(int choice) {
        EngineService.Submission s = submissions.get(choice);
        if (s.refused() instanceof EngineUnavailableException e) {
            throw unavailable(e);
        }
        if (s.refused() != null) {
            throw ServerJson.badRequest(String.valueOf(s.refused().getMessage()));
        }
        BatchSequence sequence = s.sequence();
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
        return sequence.result();
    }

    /**
     * The answer to a refusal on the engine's serving state or a replica shutting down (503), or on full replica
     * queues (429 with {@code Retry-After}).
     */
    private static WebApplicationException unavailable(IllegalStateException e) {
        return ServerJson.refusal(e);
    }

    /** The failure of a finished choice as a message, or null when it produced a result. */
    String failure(int choice) {
        EngineService.Submission s = submissions.get(choice);
        if (s.refused() != null) {
            return String.valueOf(s.refused().getMessage());
        }
        BatchSequence sequence = s.sequence();
        if (sequence == null) {
            return "the request never registered with the engine";
        }
        if (sequence.state() != BatchSequence.State.FAILED) {
            return null;
        }
        return sequence.error() instanceof UnsupportedOperationException refused
                ? String.valueOf(refused.getMessage()) : String.valueOf(sequence.error());
    }
}
