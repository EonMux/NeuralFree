package com.eonmux.neuralfree.server;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * The bounded drain of a generation stream loop: the event the loop waited for and the events already queued behind
 * it go to the handler, and the stream flushes once when the queue is empty, after {@link #MAX_EVENTS} events or
 * after {@link #MAX_NANOS} of draining, whichever comes first. The chunks of several tokens (a speculative step's
 * accepted run, or a writer that fell behind) then leave in one flush, while a writer that never catches up still
 * flushes regularly. The bytes are those of one flush per event; only when they leave changes.
 */
final class EventDrain {

    /** The most queued events one drain hands on before it flushes. */
    static final int MAX_EVENTS = 64;
    /** The longest one drain runs before it flushes. */
    static final long MAX_NANOS = TimeUnit.MILLISECONDS.toNanos(10);

    /** Handles one drained event; returns true when the loop ends with it (no further event is drained). */
    interface Handler {
        boolean on(GenerationBatch.Event event) throws IOException;
    }

    /**
     * An output stream whose flushes a drain holds back and issues once when it ends; outside a drain every flush
     * passes through, so the writes of a stream's opening and closing events flush at once.
     */
    static final class HeldFlushes extends FilterOutputStream {
        private boolean holding;

        HeldFlushes(OutputStream out) {
            super(out);
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            out.write(b, off, len);
        }

        @Override
        public void flush() throws IOException {
            if (!holding) {
                out.flush();
            }
        }

        /** Holds the flushes until {@link #release}. */
        void hold() {
            holding = true;
        }

        /** Ends the hold and flushes what it held back. */
        void release() throws IOException {
            holding = false;
            out.flush();
        }
    }

    private EventDrain() {
    }

    /**
     * Hands {@code first} and the events queued behind it to {@code handler} within the drain's bounds, holding the
     * flushes of {@code out} (when not null) until the drain ends.
     *
     * @return true when the handler ended the loop
     */
    static boolean drain(GenerationBatch batch, GenerationBatch.Event first, HeldFlushes out, Handler handler)
            throws IOException {
        Objects.requireNonNull(first, "first");
        if (out != null) {
            out.hold();
        }
        boolean ended = false;
        try {
            long started = System.nanoTime();
            int drained = 0;
            GenerationBatch.Event event = first;
            while (event != null) {
                if (handler.on(event)) {
                    ended = true;
                    break;
                }
                drained++;
                if (drained >= MAX_EVENTS || System.nanoTime() - started >= MAX_NANOS) {
                    break;
                }
                event = batch.poll();
            }
        } catch (IOException | RuntimeException | Error e) {
            if (out != null) {
                out.holding = false;
            }
            throw e;
        }
        if (out != null) {
            out.release();
        }
        return ended;
    }
}
