package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

/**
 * The completion point of a command a remote backend issued. In the pipelined protocol the command is one member
 * of a batch the server has not necessarily received yet, so the event carries the command's sequence on its
 * endpoint and completes lazily: {@link #awaitCompletion()} flushes the buffered commands and, when no queue
 * finish covers the sequence yet, makes the server finish its device through it — the same deferred contract the
 * local queues run under ({@code finish()} or {@code awaitCompletion()} completes a point, in issue order). The
 * events of commands whose answer already arrived (a read, or every command in the synchronous pipeline mode) are
 * the stateless {@link #COMPLETE}. {@link #getElapsedNanos()} is unknown (-1), as the protocol carries no
 * profiling timestamps. Closing without awaiting leaves the completion to the next observing command, like any
 * in-order queue; an awaited-then-closed or closed event stays complete.
 */
final class RemoteGpuEvent implements GpuEvent {

    static final RemoteGpuEvent COMPLETE = new RemoteGpuEvent();

    private final RemoteEndpoint endpoint;
    private final long sequence;
    private volatile boolean closed;

    private RemoteGpuEvent() {
        this.endpoint = null;
        this.sequence = 0;
    }

    /** The event of the command appended at {@code sequence} on {@code endpoint}. */
    RemoteGpuEvent(RemoteEndpoint endpoint, long sequence) {
        this.endpoint = endpoint;
        this.sequence = sequence;
    }

    @Override
    public void awaitCompletion() throws InterruptedException {
        if (endpoint == null || closed) {
            return;
        }
        endpoint.awaitThrough(sequence);
    }

    @Override
    public boolean isComplete() {
        return closed || endpoint == null || endpoint.finishedThrough(sequence);
    }

    @Override
    public long getElapsedNanos() {
        return -1L;
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return endpoint == null ? "RemoteGpuEvent{complete}" : "RemoteGpuEvent{" + sequence + ", "
                + (closed ? "closed" : isComplete() ? "complete" : "pending") + "}";
    }
}
