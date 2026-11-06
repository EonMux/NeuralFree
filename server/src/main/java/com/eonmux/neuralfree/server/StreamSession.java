package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The streaming buffer of one generation (llama.cpp's {@code stream_session}): a bounded byte buffer that
 * survives an HTTP disconnect. The producer appends the server-sent-event bytes it writes; a reader replays
 * them from any offset that the buffer still holds and then blocks for live bytes until the producer
 * {@link #finish}es. Keyed by conversation id, one conversation has at most one live session
 * ({@link StreamSessions}). When the bytes outgrow the cap the front of the buffer drops, and an offset below
 * the dropped prefix is lost.
 */
final class StreamSession {

    /** How {@link #readFrom} ended. */
    enum ReadStatus {
        /** The reader stopped at its own request, or the session finished after the bytes it asked for. */
        OK,
        /** The offset fell below the dropped prefix: those bytes left the buffer. */
        OFFSET_LOST
    }

    /** Receives one drained chunk; returning false stops the read. */
    interface Sink {
        boolean accept(byte[] chunk) throws IOException;
    }

    private final String conversationId;
    private final long startedAt;
    private final int capBytes;

    private final Object lock = new Object();
    private byte[] buffer = new byte[64 * 1024];
    private int length;
    /** Bytes evicted from the front of the buffer; the logical offset of {@code buffer[0]}. */
    private long prefixDropped;
    private boolean done;
    /** 0 while the session is live, the unix second of {@link #finish} after it. */
    private long completedAt;
    /** Polled without the lock, by the producer between chunks and by the explicit stop. */
    private final AtomicBoolean cancelled = new AtomicBoolean();

    /** A session for {@code conversationId} keeping at most {@code capBytes} bytes. */
    StreamSession(String conversationId, int capBytes) {
        if (conversationId == null || conversationId.isEmpty()) {
            throw new IllegalArgumentException("a stream session takes a non-empty conversation id");
        }
        if (capBytes <= 0) {
            throw new IllegalArgumentException("a stream session takes a positive byte cap, got " + capBytes);
        }
        this.conversationId = conversationId;
        this.startedAt = System.currentTimeMillis() / 1000;
        this.capBytes = capBytes;
    }

    /** The conversation id the session is keyed by. */
    String conversationId() {
        return conversationId;
    }

    /** The unix second the session was created at. */
    long startedAt() {
        return startedAt;
    }

    /** Whether {@link #finish} ran. */
    boolean isDone() {
        synchronized (lock) {
            return done;
        }
    }

    /** Whether the explicit stop ({@link StreamSessions#evictAndCancel} or a replacement) ran. */
    boolean isCancelled() {
        return cancelled.get();
    }

    /** The bytes that ever entered the session, dropped prefix included. */
    long totalBytes() {
        synchronized (lock) {
            return prefixDropped + length;
        }
    }

    /** The bytes evicted from the front of the buffer; the lowest offset a reader can still ask for. */
    long droppedPrefix() {
        synchronized (lock) {
            return prefixDropped;
        }
    }

    /** 0 while the session is live, the unix second of {@link #finish} after it. */
    long completedAt() {
        synchronized (lock) {
            return completedAt;
        }
    }

    /** Appends the whole array; see {@link #append(byte[], int, int)}. */
    boolean append(byte[] bytes) {
        return append(bytes, 0, bytes.length);
    }

    /**
     * Appends bytes; false once the session finished, telling the producer to stop recording (its live client
     * keeps streaming). Over the cap the front of the buffer drops; one chunk larger than the cap keeps only
     * its tail. Wakes the readers.
     */
    boolean append(byte[] bytes, int off, int len) {
        if (len <= 0) {
            return true;
        }
        synchronized (lock) {
            if (done) {
                return false;
            }
            if (len >= capBytes) {
                prefixDropped += length + (len - capBytes);
                length = 0;
                appendTail(bytes, off + (len - capBytes), capBytes);
            } else {
                int over = length + len - capBytes;
                if (over > 0) {
                    System.arraycopy(buffer, over, buffer, 0, length - over);
                    length -= over;
                    prefixDropped += over;
                }
                appendTail(bytes, off, len);
            }
            lock.notifyAll();
        }
        return true;
    }

    /** Appends {@code len} bytes at the write position; the caller holds the lock and checked the room. */
    private void appendTail(byte[] bytes, int off, int len) {
        if (length + len > buffer.length) {
            buffer = Arrays.copyOf(buffer, Math.max(buffer.length * 2, length + len));
        }
        System.arraycopy(bytes, off, buffer, length, len);
        length += len;
    }

    /** Ends the session: readers stop waiting and see it done. Idempotent. */
    void finish() {
        synchronized (lock) {
            if (done) {
                return;
            }
            done = true;
            completedAt = System.currentTimeMillis() / 1000;
            lock.notifyAll();
        }
    }

    /** Marks the session cancelled; the producer stops pulling output at its next check. */
    void cancel() {
        cancelled.set(true);
    }

    /**
     * Drains the bytes from {@code offset} into {@code sink}: the buffered suffix first, then blocking for
     * live bytes until the session finishes (or the sink stops it). Runs the sink outside the session lock.
     * Returns {@link ReadStatus#OFFSET_LOST} when {@code offset} fell below the dropped prefix.
     */
    ReadStatus readFrom(long offset, Sink sink) throws IOException, InterruptedException {
        long next = offset;
        while (true) {
            byte[] chunk;
            synchronized (lock) {
                while (true) {
                    if (next < prefixDropped) {
                        return ReadStatus.OFFSET_LOST;
                    }
                    long end = prefixDropped + length;
                    if (next < end) {
                        int from = (int) (next - prefixDropped);
                        chunk = Arrays.copyOfRange(buffer, from, length);
                        next = prefixDropped + length;
                        break;
                    }
                    if (done) {
                        return ReadStatus.OK;
                    }
                    lock.wait(200);
                }
            }
            if (!sink.accept(chunk)) {
                return ReadStatus.OK;
            }
        }
    }

    /**
     * The producer end (llama.cpp's tee-style pipe): an output stream that mirrors every byte into this
     * session while writing it to the client. A client write that fails — the client went away — stops the
     * client side alone and the session keeps recording; a finished session stops recording and the live
     * client keeps its stream. Used by one producer only.
     */
    OutputStream mirror(OutputStream client) {
        return new Mirror(client);
    }

    private final class Mirror extends OutputStream {
        private final OutputStream client;
        private boolean clientGone;
        private boolean recording = true;

        Mirror(OutputStream client) {
            this.client = client;
        }

        @Override
        public void write(int b) throws IOException {
            write(new byte[] {(byte) b}, 0, 1);
        }

        @Override
        public void write(byte[] bytes) throws IOException {
            write(bytes, 0, bytes.length);
        }

        @Override
        public void write(byte[] bytes, int off, int len) throws IOException {
            if (len <= 0) {
                return;
            }
            if (!clientGone) {
                try {
                    client.write(bytes, off, len);
                } catch (IOException e) {
                    clientGone = true;
                }
            }
            if (recording && !append(bytes, off, len)) {
                recording = false;
            }
        }

        @Override
        public void flush() throws IOException {
            if (clientGone) {
                return;
            }
            try {
                client.flush();
            } catch (IOException e) {
                clientGone = true;
            }
        }

        @Override
        public void close() throws IOException {
            if (!clientGone) {
                client.close();
            }
        }
    }
}
