package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;

import java.util.Objects;
import java.util.function.Supplier;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * The per-request latencies of one {@link EngineService}, vLLM's request-level histograms: the time to the first
 * token and the end-to-end latency from the request's arrival at the HTTP layer, the time between consecutive
 * sampled tokens, and the time the engine kept the request queued before its first admission. Each choice of a
 * request reports through its own {@link Choice}; the {@link Listener}s (the Prometheus timers of
 * {@link EngineMetrics}) receive every observation on the engine thread that made it, so a listener must be cheap and
 * thread-safe.
 */
final class RequestLatencies {

    /** Receives the observations, in nanoseconds. */
    interface Listener {
        /** A choice's time from arrival to its first sampled token. */
        void firstToken(long nanos);

        /** The time between two consecutive sampled tokens of one choice. */
        void interToken(long nanos);

        /** A choice's time from arrival to its terminal state, for a choice that sampled at least one token. */
        void finished(long nanos);

        /** A choice's time in the engine queue, from its submit to its first admission. */
        void queued(long nanos);
    }

    private final CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<>();

    /**
     * Adds {@code listener} unless an equal one is already registered; it receives the observations made from now
     * on.
     */
    void add(Listener listener) {
        listeners.addIfAbsent(Objects.requireNonNull(listener, "listener"));
    }

    /** A tracker for one choice of a request that arrived at {@code arrivalNanos} ({@link System#nanoTime()}). */
    Choice choice(long arrivalNanos) {
        return new Choice(arrivalNanos);
    }

    /**
     * One choice's timings. The engine thread reports its tokens ({@link #token}) and its end ({@link #done}); the
     * request thread hands it the submission once the submit returned ({@link #submitted}), which can be after the
     * end, so the queue time is observed by whichever of the two comes last, reading the engine handle then: a
     * replica creates the handle on its pump thread, before the choice's first output.
     */
    final class Choice {
        private final long arrivalNanos;
        /** The previous token's instant, or -1 before the first; the engine thread alone touches it. */
        private long lastTokenNanos = -1;
        private Supplier<BatchSequence> sequence;
        private boolean done;

        private Choice(long arrivalNanos) {
            this.arrivalNanos = arrivalNanos;
        }

        /** A token sampled at {@code now}. */
        void token(long now) {
            if (lastTokenNanos < 0) {
                for (Listener l : listeners) {
                    l.firstToken(now - arrivalNanos);
                }
            } else {
                for (Listener l : listeners) {
                    l.interToken(now - lastTokenNanos);
                }
            }
            lastTokenNanos = now;
        }

        /** The choice reached its terminal state at {@code now}. */
        void done(long now) {
            if (lastTokenNanos >= 0) {
                for (Listener l : listeners) {
                    l.finished(now - arrivalNanos);
                }
            }
            Supplier<BatchSequence> handle;
            synchronized (this) {
                done = true;
                handle = sequence;
            }
            if (handle != null) {
                queued(handle.get());
            }
        }

        /**
         * The submission's engine handle, read when the queue time is observed: null there when the engine never
         * took the choice.
         */
        void submitted(Supplier<BatchSequence> handle) {
            Objects.requireNonNull(handle, "handle");
            boolean finished;
            synchronized (this) {
                sequence = handle;
                finished = done;
            }
            if (finished) {
                queued(handle.get());
            }
        }

        private void queued(BatchSequence handle) {
            if (handle == null) {
                return;
            }
            long admitted = handle.admittedAtNanos();
            if (admitted == BatchSequence.NEVER_ADMITTED) {
                return;
            }
            for (Listener l : listeners) {
                l.queued(admitted - handle.submittedAtNanos());
            }
        }
    }
}
