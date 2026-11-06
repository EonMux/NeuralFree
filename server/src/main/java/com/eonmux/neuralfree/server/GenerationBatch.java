package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;

import io.vertx.core.http.HttpServerRequest;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The generations of one HTTP request — its choices ({@code n}, prompt arrays) — submitted together so the batch
 * engine decodes them side by side. Their output arrives as {@link Event}s in the order the pump thread routed it:
 * text pieces, harmony tool recipients and, last for each choice, its {@link Done} event. The queue is unbounded
 * so a slow client never stalls the shared pump thread.
 *
 * <p>{@link #cancel()} (a client that went away) cancels every running choice through the requests' cancellation
 * hook. Errors map to HTTP: a refused submission (an oversize prompt, an unusable grammar) is a 400, an engine
 * failure a 500.
 */
final class GenerationBatch {

    /** One generation to submit. */
    record Job(int[] prompt, GenerationRequest request, String renderedPrompt, ConstraintSpec constraint) { }

    /** Something one choice produced. */
    sealed interface Event permits Text, ToolRecipient, Done {
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

    /** The choice reached its terminal state; every event of it came before. */
    record Done(int choice) implements Event { }

    private final List<EngineService.Submission> submissions = new ArrayList<>();
    private final BlockingQueue<Event> events = new LinkedBlockingQueue<>();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final ServerOptions options;
    private final long deadline;

    private GenerationBatch(ServerOptions options) {
        this.options = options;
        this.deadline = System.nanoTime() + options.requestTimeout().toNanos();
    }

    /**
     * Submits every job (job {@code i} is choice {@code i}) and links the batch to the client: its going away
     * cancels the batch ({@link ClientLink}).
     *
     * @param client the HTTP request, or null outside the HTTP layer
     */
    static GenerationBatch start(EngineService service, List<Job> jobs, HttpServerRequest client) {
        GenerationBatch batch = new GenerationBatch(service.options());
        ClientLink.onClose(client, batch::cancel);
        try {
            for (int i = 0; i < jobs.size(); i++) {
                batch.submit(service, jobs.get(i), i);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            batch.cancel();
            throw ServerJson.error(503, "the server is shutting down", "unavailable_error");
        }
        return batch;
    }

    private void submit(EngineService service, Job job, int choice) throws InterruptedException {
        GenerationRequest request = job.request().toBuilder().cancellation(cancelled::get).build();
        ReasoningFilter.Sink sink = new ReasoningFilter.Sink() {
            /** Tokens generated so far and the latest one; the engine thread alone calls the sink. */
            private int predicted;
            private int last = -1;

            @Override
            public void stepped(int tokenId) {
                predicted++;
                last = tokenId;
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
        };
        submissions.add(service.submit(job.prompt(), request, sink, job.renderedPrompt(), job.constraint(),
                () -> events.add(new Done(choice))));
    }

    /** The engine handle of choice {@code choice}, or null when the engine refused it. */
    BatchSequence sequence(int choice) {
        return submissions.get(choice).sequence();
    }

    /** The number of choices. */
    int size() {
        return submissions.size();
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
     * @throws jakarta.ws.rs.WebApplicationException 400 when the engine refused the submission, 500 when it failed
     */
    GenerationResult result(int choice) {
        EngineService.Submission s = submissions.get(choice);
        if (s.refused() != null) {
            throw ServerJson.badRequest(String.valueOf(s.refused().getMessage()));
        }
        BatchSequence sequence = s.sequence();
        if (sequence == null) {
            throw ServerJson.error(500, "the request never registered with the engine", "server_error");
        }
        if (sequence.state() == BatchSequence.State.FAILED) {
            throw ServerJson.error(500, String.valueOf(sequence.error()), "server_error");
        }
        return sequence.result();
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
        return sequence.state() == BatchSequence.State.FAILED ? String.valueOf(sequence.error()) : null;
    }
}
