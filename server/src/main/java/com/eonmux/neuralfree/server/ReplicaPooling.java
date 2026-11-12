package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.engine.PoolingOutput;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.PoolingSequence;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSession;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntSupplier;

/**
 * The pooling and idle-engine forwards of one {@link EngineReplica}: the one-shot model calls run on the pump
 * thread, the pooling requests the engine batches beside generation, and the adapter-slot wrapper of a pooling
 * forward. The replica owns the pump thread, its work queue and its engine; the calls queue into that queue and
 * wait for the pump.
 */
final class ReplicaPooling {

    /** A model forward run on the pump thread. */
    @FunctionalInterface
    interface ModelCall<T> {
        T run() throws IOException;
    }

    /** A queued idle-engine call and the future its caller waits on. */
    private record PoolingCall<T>(ModelCall<T> call, CompletableFuture<T> result, boolean resetsMemory) {
        void run() {
            try {
                result.complete(call.run());
            } catch (IOException | RuntimeException e) {
                result.completeExceptionally(e);
            }
        }
    }

    private final EngineReplica owner;
    /** Pooling forwards waiting for an idle engine; touched by the pump thread only (and by close after it). */
    private final Deque<PoolingCall<?>> pooling = new ArrayDeque<>();
    /** Pooling forwards a caller waits on, queued or running. */
    private final AtomicInteger poolingInFlight = new AtomicInteger();

    ReplicaPooling(EngineReplica owner) {
        this.owner = owner;
    }

    /** Pooling forwards a caller waits on, queued or running. */
    int inFlight() {
        return poolingInFlight.get();
    }

    /** Whether no pooling forward waits for the idle engine. */
    boolean isEmpty() {
        return pooling.isEmpty();
    }

    /**
     * {@link #onPump(ModelCall)} with the forward's passes in the adapter slot {@code selection} resolves to.
     *
     * @throws IllegalArgumentException      naming an adapter the registry does not hold, or a key it refuses
     * @throws UnsupportedOperationException when the request selects adapters on a model that runs none per
     *                                       request, or the rows would run an activated LoRA
     */
    <T> T onPump(LoraSelection selection, ModelCall<T> call) throws IOException {
        return onPump(selection, call, null);
    }

    /**
     * {@link #onPump(LoraSelection, ModelCall)} whose caller keeps waiting past the request budget while
     * {@code progress} (null for none) moves: a call that runs several requests in one handoff reports each one it
     * finished, so the budget bounds each request's wait as it does for one handoff per request.
     */
    private <T> T onPump(LoraSelection selection, ModelCall<T> call, IntSupplier progress) throws IOException {
        return onIdle(() -> {
            EngineModel open = owner.model();
            LoraRegistry registry = open.loraRegistry();
            if (registry == null) {
                if (selection != null) {
                    throw new UnsupportedOperationException("lora: " + open.architecture() + " runs no"
                            + " per-request LoRA adapters (" + selection + ")");
                }
                return call.run();
            }
            LoraPlan plan = LoraPlan.uniform(registry, selection, "a pooling forward runs every row in one"
                    + " adapter slot");
            if (plan.isDefault()) {
                return call.run();
            }
            try (LoraSession session = LoraSession.open(registry, plan)) {
                session.runKey();
                return call.run();
            }
        }, true, progress);
    }

    /**
     * Runs a pooling forward on the pump thread — the model's only thread — once the batch engine is idle: the
     * forward resets the shared memory, which would corrupt running sequences, so it waits for them to finish.
     *
     * @throws IOException                   when the forward fails, the wait outlives the request budget, or the
     *                                       caller is interrupted
     * @throws UnsupportedOperationException when the model does not pool or classify
     */
    <T> T onPump(ModelCall<T> call) throws IOException {
        return onIdle(call, true, null);
    }

    /**
     * Runs a call on the batch engine's state on the pump thread once the engine is idle, without dropping the
     * cached prefixes afterwards: the slot actions read and fill the prefix cache.
     *
     * @throws IOException when the call fails, the wait outlives the request budget, or the caller is interrupted
     */
    <T> T onIdleEngine(ModelCall<T> call) throws IOException {
        return onIdle(call, false, null);
    }

    /**
     * Queues {@code call} for the idle engine and waits for its result for up to the request budget, and on while
     * {@code progress} (null for none) moved within the last budget.
     */
    private <T> T onIdle(ModelCall<T> call, boolean resetsMemory, IntSupplier progress) throws IOException {
        PoolingCall<T> task = new PoolingCall<>(call, new CompletableFuture<>(), resetsMemory);
        poolingInFlight.incrementAndGet();
        try {
            owner.work().put(() -> pooling.add(task));
            long budget = owner.options().requestTimeout().toNanos();
            int seen = progress == null ? 0 : progress.getAsInt();
            while (true) {
                try {
                    return task.result().get(budget, TimeUnit.NANOSECONDS);
                } catch (TimeoutException e) {
                    int now = progress == null ? seen : progress.getAsInt();
                    if (now == seen) {
                        throw new IOException("the model stayed busy past the " + owner.options().budgetText()
                                + " request budget", e);
                    }
                    seen = now;
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while waiting for the model", e);
        } catch (ExecutionException e) {
            throw causeOf(e.getCause());
        } finally {
            poolingInFlight.decrementAndGet();
        }
    }

    /**
     * Runs the model's one-shot pooling forward for {@code output} over {@code tokens} on the pump thread: the
     * memory-resetting form a model without a paged memory pools through. {@code activation} applies or skips the
     * classifier's final probability activation.
     */
    private Object oneShot(PoolingOutput output, int[] tokens, boolean activation) throws IOException {
        EngineModel open = owner.model();
        switch (output) {
            case POOLED:
                return open.encode(tokens);
            case TOKEN_EMBEDDINGS:
                return open.encodeTokens(tokens);
            case TOKEN_SCORES:
                return open.tokenClassify(tokens);
            case SCORES:
                return open.classify(tokens);
            case CLASSIFIER_OUTPUT:
                return open.classifyOutput(tokens, activation);
            case RERANK_SCORE:
                return open.classifyOutput(tokens, activation).scores()[0];
            default:
                throw new IllegalArgumentException("unknown pooling output " + output);
        }
    }

    /**
     * The one-shot pooling forwards of every request of one call, in order, in as few pump handoffs (and adapter
     * sessions) as the pump allows: a handoff runs the requests one after another until all ran or other work waits
     * for the pump (a generation's registration, another caller's pooling call), which then runs before the rest.
     * The caller's wait is bounded per request, as with one handoff per request: it fails when no request of the
     * call finished within the request budget, and the call then runs no further request.
     */
    private Object[] oneShots(PoolingOutput output, int[][] requests, LoraSelection selection, boolean activation)
            throws IOException {
        Object[] out = new Object[requests.length];
        AtomicInteger finished = new AtomicInteger();
        AtomicBoolean abandoned = new AtomicBoolean();
        try {
            while (finished.get() < requests.length) {
                onPump(selection, () -> oneShotsFrom(output, requests, activation, out, finished, abandoned),
                        finished::get);
            }
        } catch (IOException | RuntimeException e) {
            abandoned.set(true);
            throw e;
        }
        return out;
    }

    /**
     * One handoff of {@link #oneShots}: the requests from {@code finished} on, into {@code out}, until every request
     * ran, other work waits for the pump, or the caller {@code abandoned} the call; an encoder's pooled rows run a
     * packed memoryless batch per step ({@link #packedPooling}). Returns how many have finished.
     */
    private Integer oneShotsFrom(PoolingOutput output, int[][] requests, boolean activation, Object[] out,
                                 AtomicInteger finished, AtomicBoolean abandoned) throws IOException {
        int next = finished.get();
        int pack = packedPooling(output);
        while (next < requests.length && !abandoned.get()) {
            if (pack > 1) {
                int end = Math.min(requests.length, next + pack);
                float[][] rows = owner.model().encodeBatch(Arrays.copyOfRange(requests, next, end));
                System.arraycopy(rows, 0, out, next, rows.length);
                next = end;
            } else {
                out[next] = oneShot(output, requests[next], activation);
                next++;
            }
            finished.set(next);
            if (!owner.work().isEmpty()) {
                break;
            }
        }
        return next;
    }

    /**
     * Requests one pump step pools together: the sequences one memoryless batch packs when the model is an encoder
     * planned without memory and {@code output} is its pooled row ({@code EngineModel.encodeBatch} packs them, each
     * sequence attending only itself), else 1.
     */
    private int packedPooling(PoolingOutput output) {
        EngineModel open = owner.model();
        if (output != PoolingOutput.POOLED || open.poolingType() == LLMModel.PoolingType.NONE) {
            return 1;
        }
        ForwardBackend backend = open.forwardBackend();
        return PagedEvaluations.packsMemoryless(backend) ? backend.memorylessSequences() : 1;
    }

    /**
     * Runs one pooling request for {@code output} over {@code tokens}: over the paged memory it queues into the
     * batch engine; over a contiguous or absent memory it runs the model's one-shot forward on the pump thread once
     * the engine is idle.
     */
    Object pooling(int[] tokens, PoolingOutput output, LoraSelection selection) throws IOException {
        return poolingAll(new int[][]{tokens}, output, selection)[0];
    }

    /** {@link #pooling} with the classifier's final probability activation applied or skipped. */
    Object pooling(int[] tokens, PoolingOutput output, LoraSelection selection, boolean activation)
            throws IOException {
        return poolingAll(new int[][]{tokens}, output, selection, activation)[0];
    }

    /** Runs {@code requests} pooling requests over one replica, all queued before any is awaited. */
    Object[] poolingAll(int[][] requests, PoolingOutput output, LoraSelection selection) throws IOException {
        return poolingAll(requests, output, selection, true);
    }

    /** {@link #poolingAll} with the classifier's final probability activation applied or skipped per request. */
    Object[] poolingAll(int[][] requests, PoolingOutput output, LoraSelection selection, boolean activation)
            throws IOException {
        BatchEngine eng = poolingEngine();
        if (eng == null) {
            return oneShots(output, requests, selection, activation);
        }
        List<CompletableFuture<Object>> futures = new ArrayList<>(requests.length);
        for (int i = 0; i < requests.length; i++) {
            futures.add(new CompletableFuture<>());
        }
        try {
            queuePooling(eng, requests, output, selection, activation, futures);
        } catch (InterruptedException e) {
            for (CompletableFuture<Object> f : futures) {
                f.cancel(false);
            }
            throw new IOException("interrupted while queueing the pooling requests", e);
        }
        Object[] out = new Object[futures.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = awaitPoolingAnswer(futures.get(i));
        }
        return out;
    }

    /**
     * Queues the pooling requests of one call into {@code eng} as one pump task, which submits them in order on
     * the pump thread: request {@code i} completes {@code done.get(i)}, and a refused submission fails only its own
     * future.
     */
    private void queuePooling(BatchEngine eng, int[][] requests, PoolingOutput output, LoraSelection selection,
                              boolean activation, List<CompletableFuture<Object>> done) throws InterruptedException {
        poolingInFlight.addAndGet(requests.length);
        try {
            owner.work().put(() -> {
                for (int i = 0; i < requests.length; i++) {
                    CompletableFuture<Object> answer = done.get(i);
                    try {
                        PoolingSequence pooling = eng.submitPooling(requests[i], output, selection, activation);
                        pooling.handle().onFinish(() -> {
                            poolingInFlight.decrementAndGet();
                            completePooling(pooling, answer);
                        });
                    } catch (RuntimeException e) {
                        poolingInFlight.decrementAndGet();
                        answer.completeExceptionally(e);
                    }
                }
            });
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            poolingInFlight.addAndGet(-requests.length); // the registration never ran: its in-flight count returns
            throw e;
        }
    }

    /** Waits for one queued pooling request within the request budget, its failure as an {@code IOException}. */
    private Object awaitPoolingAnswer(CompletableFuture<Object> done) throws IOException {
        try {
            return done.get(owner.options().requestTimeout().toNanos(), TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while waiting for the model", e);
        } catch (TimeoutException e) {
            throw new IOException("the model stayed busy past the " + owner.options().budgetText()
                    + " request budget", e);
        } catch (ExecutionException e) {
            throw causeOf(e.getCause());
        }
    }

    /** Completes a pooling wait with the request's output, or with its failure or cancellation. */
    private static void completePooling(PoolingSequence pooling, CompletableFuture<Object> done) {
        BatchSequence handle = pooling.handle();
        if (handle.state() == BatchSequence.State.DONE) {
            done.complete(pooling.pooled());
            return;
        }
        Throwable error = handle.error();
        done.completeExceptionally(error != null ? error
                : new IOException("the pooling request was cancelled (a sleep or pause aborted it)"));
    }

    /** The cause of a failed model call as its own exception: an {@code IOException} rethrown, any other wrapped. */
    private static IOException causeOf(Throwable cause) {
        if (cause instanceof IOException io) {
            return io;
        }
        if (cause instanceof java.io.UncheckedIOException unchecked) {
            return unchecked.getCause();
        }
        if (cause instanceof RuntimeException re) {
            throw re;
        }
        return new IOException(cause);
    }

    /**
     * The batch engine over the model once its memory is paged (built on first use), or null when its memory is
     * contiguous or absent: pooling then runs the model's one-shot forward on the pump thread.
     */
    private synchronized BatchEngine poolingEngine() {
        BatchEngine current = owner.builtEngine();
        if (current != null) {
            return current;
        }
        EngineModel open = owner.model();
        if (open == null) {
            return null;
        }
        com.eonmux.neuralfree.gguf.memory.ModelMemory memory = open.forwardBackend().memory();
        if (memory == null || memory.pagedSequences() == null) {
            return null;
        }
        return owner.engine();
    }

    /**
     * Runs the waiting idle-engine calls when no sequence uses the memory, then, when a pooling forward among them
     * reset the memory, drops the stale cached prefixes. Returns whether a call ran.
     */
    boolean runWaiting(boolean busy) {
        if (pooling.isEmpty() || busy) {
            return false;
        }
        boolean reset = false;
        while (!pooling.isEmpty()) {
            PoolingCall<?> call = pooling.poll();
            reset |= call.resetsMemory();
            call.run();
        }
        BatchEngine current = owner.builtEngine();
        if (reset && current != null) {
            current.clearPrefixCache();
        }
        return true;
    }

    /** Fails every waiting call: the server is shutting down. */
    void close() {
        PoolingCall<?> call;
        while ((call = pooling.poll()) != null) {
            call.result().completeExceptionally(new IOException("the server is shutting down"));
        }
    }
}
