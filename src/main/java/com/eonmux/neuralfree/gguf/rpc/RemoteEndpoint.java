package com.eonmux.neuralfree.gguf.rpc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The client-side state one connection's remote backends share: the authenticated connection, the devices its
 * handshake advertised, and the pipelined command batch ({@code neuralfree.rpc.pipeline}, on by default). Commands
 * that need no answer — writes, fills, copies, dispatches, frees, kernel release and the queue finish — are
 * appended to the batch and flushed to the server as one {@link RpcWire#OP_BATCH} frame, which is answered once
 * after it executed the commands in order; a decode step is then one round trip instead of one per command
 * (llama.cpp's async RPC dispatcher). Commands that need an answer flush the batch first, so every command of
 * every queue of the endpoint reaches the server in issue order. With the pipeline off, every command flushes at
 * once, and the device-observable ones carry a queue-finish member, as the synchronous protocol did.
 *
 * <p>Every appended command gets a sequence number; {@link #awaitThrough} makes the device finish through one, so
 * a deferred {@link RemoteGpuEvent} completes lazily exactly where its caller observes. The endpoint is
 * reference-counted by its backends: the last {@link #release()} closes the connection. All state changes happen
 * under the endpoint lock, which serializes the queues of the endpoint the way the connection serializes them on
 * the wire.</p>
 */
final class RemoteEndpoint {

    /** The property that turns the pipelined batch off, restoring the synchronous round trip per command. */
    static final String PROP_PIPELINE = "neuralfree.rpc.pipeline";
    /** Buffered command bytes one flush assembles into frames at most this large. */
    private static final int BATCH_FLUSH_BYTES = 32 << 20;
    /** The frame overhead of a batch: opcode byte and the member count. */
    private static final int BATCH_OVERHEAD = 5;

    private final RpcConnection connection;
    private final List<RemoteDeviceInfo> devices;
    private final boolean cacheEnabled;
    private final boolean pipeline;
    /** The bodies of the commands appended since the last flush, each starting with its opcode; guarded by this. */
    private final List<RpcWire.Writer> pending = new ArrayList<>();
    private int pendingBytes;
    /** The sequence number the last appended command carries. */
    private long sequence;
    /** The sequence of the last command a batch response covered. */
    private long flushedUpTo;
    /** The sequence of the last command a queue-finish covered. */
    private long finishedUpTo;
    private int refs;
    private boolean closed;

    private RemoteEndpoint(RpcConnection connection, boolean pipeline) {
        this.connection = connection;
        this.devices = connection.devices();
        this.cacheEnabled = connection.cacheEnabled();
        this.pipeline = pipeline;
    }

    /**
     * Connects to {@code spec}, authenticates, and reads the pipelining flag.
     *
     * @throws IllegalArgumentException when {@link #PROP_PIPELINE} is set to anything but true or false
     */
    static RemoteEndpoint open(RpcServerSpec spec) throws IOException {
        boolean pipeline = pipeline();
        return new RemoteEndpoint(RpcConnection.open(spec), pipeline);
    }

    /** The {@link #PROP_PIPELINE} setting; unset is on. */
    static boolean pipeline() {
        String v = System.getProperty(PROP_PIPELINE);
        if (v == null || v.trim().isEmpty()) {
            return true;
        }
        String s = v.trim();
        if (s.equalsIgnoreCase("true")) {
            return true;
        }
        if (s.equalsIgnoreCase("false")) {
            return false;
        }
        throw new IllegalArgumentException(PROP_PIPELINE + "='" + v + "' is invalid; use true or false");
    }

    /** The devices of the endpoint, in the server's {@code -d} order. */
    List<RemoteDeviceInfo> devices() {
        return devices;
    }

    /** Whether the server keeps a weight cache and answers {@link RpcWire#OP_WRITE_HASH} probes. */
    boolean cacheEnabled() {
        return cacheEnabled;
    }

    /** Whether commands batch ({@link #PROP_PIPELINE}). */
    boolean pipelined() {
        return pipeline;
    }

    String describe() {
        return connection.describe();
    }

    /**
     * Appends one command to the batch and returns its sequence; the member starts with its opcode and carries a
     * self-describing body. In pipelined mode the batch flushes to the wire once it holds {@link
     * #BATCH_FLUSH_BYTES} bytes; in synchronous mode the command flushes at once, with a queue-finish member after
     * it when {@code syncAfter} is set, so the server executes and finishes it before the answer, as the
     * synchronous protocol did.
     */
    synchronized long submit(RpcWire.Writer member, boolean syncAfter) {
        Objects.requireNonNull(member, "member");
        long seq = append(member);
        if (!pipeline) {
            if (syncAfter) {
                append(new RpcWire.Writer(RpcWire.OP_FINISH));
            }
            flush();
            if (syncAfter) {
                finishedUpTo = flushedUpTo;
            }
            return seq;
        }
        if (pendingBytes >= BATCH_FLUSH_BYTES) {
            flush();
        }
        return seq;
    }

    /** The sequence number of the last appended command; 0 when nothing was appended yet. */
    synchronized long lastSequence() {
        return sequence;
    }

    /**
     * Appends a queue-finish member and flushes the batch; the response arrives once the device finished every
     * command of the endpoint up to now.
     */
    synchronized void finish() {
        append(new RpcWire.Writer(RpcWire.OP_FINISH));
        flush();
        finishedUpTo = flushedUpTo;
    }

    /** Sends the buffered commands as one or more {@link RpcWire#OP_BATCH} frames; nothing buffered is a no-op. */
    synchronized void flush() {
        if (pending.isEmpty()) {
            return;
        }
        List<Batch> batches = takeBatches();
        try {
            for (Batch batch : batches) {
                batch.checkExecuted(connection.call(batch.frame));
                flushedUpTo = sequence;
            }
        } catch (RuntimeException e) {
            // the commands of a failed batch are gone; a later command must not resend them
            pending.clear();
            pendingBytes = 0;
            throw e;
        }
    }

    /** One {@link RpcWire#OP_BATCH} frame of buffered commands and its member count. */
    private final class Batch {
        final RpcWire.Writer frame;
        final int count;

        Batch(RpcWire.Writer frame, int count) {
            this.frame = frame;
            this.count = count;
        }

        /** Checks the server's answer to the frame: it executed every member. */
        void checkExecuted(RpcWire.Reader answer) {
            int executed = answer.i32();
            if (executed != count) {
                throw new IllegalStateException("remote device " + connection.describe() + " executed "
                        + executed + " of " + count + " batch commands");
            }
        }
    }

    /**
     * Takes the buffered commands as batch frames of at most {@link RpcWire#MAX_FRAME} bytes each, in order; the
     * member count goes before the bodies, so the server can bound its member loop.
     */
    private List<Batch> takeBatches() {
        List<RpcWire.Writer> bodies = new ArrayList<>(pending);
        pending.clear();
        pendingBytes = 0;
        List<Batch> batches = new ArrayList<>(1);
        int i = 0;
        while (i < bodies.size()) {
            int start = i;
            int count = 0;
            int size = BATCH_OVERHEAD;
            while (i < bodies.size()) {
                int needed = 1 + bodies.get(i).size();
                if (count > 0 && size + needed > RpcWire.MAX_FRAME) {
                    break;
                }
                count++;
                size += needed;
                i++;
            }
            RpcWire.Writer w = new RpcWire.Writer(RpcWire.OP_BATCH, size).i32(count);
            for (int j = start; j < i; j++) {
                w.body(bodies.get(j));
            }
            batches.add(new Batch(w, count));
        }
        return batches;
    }

    /**
     * A request that needs an answer: the buffered commands flush first, so the answer covers them, then the
     * request goes out and its response body comes back, positioned after the status byte. The prefix stays pending
     * unless the request itself is an observation point ({@link #request(RpcWire.Writer, boolean)}).
     */
    RpcWire.Reader request(RpcWire.Writer w) {
        return request(w, false);
    }

    /**
     * A request that needs an answer. When {@code finishesPrefix} is set, the request is an observation point — a
     * read, whose data the server makes durable before answering — and the commands before it count as finished,
     * so their deferred events complete without another round trip. A read rides in the same round trip as the
     * last batch of buffered commands ({@link RpcConnection#callPipelined}): it has no side effects, so when the
     * batch fails, the read the server still answered is dropped and the batch's error is raised.
     */
    synchronized RpcWire.Reader request(RpcWire.Writer w, boolean finishesPrefix) {
        Objects.requireNonNull(w, "writer");
        long at = sequence;
        RpcWire.Reader r = w.opcode() == RpcWire.OP_READ ? flushWithRead(w) : flushThenCall(w);
        if (finishesPrefix && finishedUpTo < at) {
            finishedUpTo = at;
        }
        return r;
    }

    private RpcWire.Reader flushThenCall(RpcWire.Writer w) {
        flush();
        return connection.call(w);
    }

    /** Flushes the buffered commands with {@code read} pipelined behind the last batch frame. */
    private RpcWire.Reader flushWithRead(RpcWire.Writer read) {
        if (pending.isEmpty()) {
            return connection.call(read);
        }
        List<Batch> batches = takeBatches();
        try {
            for (int i = 0; i < batches.size() - 1; i++) {
                Batch batch = batches.get(i);
                batch.checkExecuted(connection.call(batch.frame));
                flushedUpTo = sequence;
            }
            Batch last = batches.get(batches.size() - 1);
            RpcConnection.Answers answers = connection.callPipelined(last.frame, read);
            last.checkExecuted(answers.first());
            flushedUpTo = sequence;
            return answers.second();
        } catch (RuntimeException e) {
            pending.clear();
            pendingBytes = 0;
            throw e;
        }
    }

    /**
     * The weight-cache probe of a write: whether the server's cache holds the payload. On a hit the server wrote
     * the cached bytes to the buffer and finished its device queue before answering, so the prefix counts as
     * finished; on a miss nothing ran and nothing is marked.
     */
    synchronized boolean probeWriteHash(RpcWire.Writer w) {
        Objects.requireNonNull(w, "writer");
        flush();
        long at = sequence;
        boolean hit = connection.call(w).bool();
        if (hit && finishedUpTo < at) {
            finishedUpTo = at;
        }
        return hit;
    }

    /**
     * True when the device finished every command up to {@code seq}: a queue-finish covered them.
     */
    synchronized boolean finishedThrough(long seq) {
        return seq <= finishedUpTo;
    }

    /**
     * Makes the device finish through {@code seq}: the buffered commands flush, and when no queue-finish covers
     * {@code seq} yet, a finish goes out and the server answers it after its device drained.
     *
     * @throws IllegalStateException when the connection is closed or failed
     */
    synchronized void awaitThrough(long seq) {
        if (closed) {
            throw new IllegalStateException("remote device " + connection.describe() + " is closed");
        }
        if (finishedThrough(seq)) {
            return;
        }
        flush();
        if (!finishedThrough(seq)) {
            finish();
        }
    }

    /** Registers one more backend sharing this endpoint. */
    synchronized void retain() {
        refs++;
    }

    /**
     * Drops one backend's reference; the last one flushes the buffered commands and closes the connection. The
     * buffered commands of a dying endpoint are flushed so their device work was issued before the connection goes
     * away; a failure keeps the close (the server releases the connection's resources either way).
     */
    synchronized void release() {
        refs--;
        if (refs > 0) {
            return;
        }
        try {
            flush();
        } catch (RuntimeException ignored) {
            // the connection is already unusable; the server releases what the connection owned
        }
        closed = true;
        connection.close();
    }

    /** Closes the connection regardless of the reference count, discarding any buffered commands. */
    synchronized void abandon() {
        if (closed) {
            return;
        }
        closed = true;
        pending.clear();
        pendingBytes = 0;
        connection.close();
    }

    /** Whether {@link #abandon} or the last {@link #release} ran. */
    synchronized boolean isClosed() {
        return closed;
    }

    /** The devices of the endpoint as an unmodifiable view, for messages. */
    List<RemoteDeviceInfo> deviceInfos() {
        return Collections.unmodifiableList(devices);
    }

    /** Appends one command body and returns its sequence; each member carries its opcode prefix and body. */
    private long append(RpcWire.Writer member) {
        pending.add(member);
        pendingBytes += member.size();
        sequence++;
        return sequence;
    }

    @Override
    public String toString() {
        return "RemoteEndpoint{" + connection.describe() + ", " + devices.size() + " device(s), "
                + (pipeline ? "pipelined" : "synchronous") + (cacheEnabled ? ", weight cache" : "") + "}";
    }
}
