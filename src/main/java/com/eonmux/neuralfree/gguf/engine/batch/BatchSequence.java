package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;

import java.util.Objects;

/**
 * One request's handle in a {@link BatchEngine}: its paged sequence id once admitted, its lifecycle state and,
 * after {@link State#DONE} or {@link State#CANCELLED}, the finished {@link GenerationResult}. The engine owns the
 * underlying memory slot; the handle only reports. A finished or failed handle stays valid until the caller drops
 * it, but its sequence id is freed for reuse the moment the engine retires it.
 *
 * <p>States: {@link State#QUEUED} waits for a sequence slot and the pages of its prompt; {@link State#RUNNING}
 * prefills or decodes; {@link State#DONE} and {@link State#CANCELLED} carry a {@link GenerationResult} (a
 * cancellation's result reports {@code StopReason.CANCELLED}); {@link State#FAILED} carries the engine
 * {@link Throwable} that ended it.
 */
public final class BatchSequence {

    /** Lifecycle of one submitted request. */
    public enum State { QUEUED, RUNNING, DONE, CANCELLED, FAILED }

    private final int submitId;
    private volatile State state = State.QUEUED;
    private volatile int seq = -1;
    private volatile GenerationResult result;
    private volatile Throwable error;

    BatchSequence(int submitId) {
        this.submitId = submitId;
    }

    /** The engine-unique submission number, in submit order from 0. */
    public int submitId() {
        return submitId;
    }

    /** The paged sequence id the engine assigned, or -1 while {@link State#QUEUED}. */
    public int seq() {
        return seq;
    }

    /** The current lifecycle state. */
    public State state() {
        return state;
    }

    /** The finished result; non-null only after {@link State#DONE} or {@link State#CANCELLED}. */
    public GenerationResult result() {
        return result;
    }

    /** The engine failure; non-null only after {@link State#FAILED}. */
    public Throwable error() {
        return error;
    }

    /** Whether the request left {@link State#QUEUED} or {@link State#RUNNING} (done, cancelled or failed). */
    public boolean finished() {
        State s = state;
        return s == State.DONE || s == State.CANCELLED || s == State.FAILED;
    }

    void admit(int seq) {
        this.seq = seq;
        this.state = State.RUNNING;
    }

    void complete(GenerationResult result, boolean cancelled) {
        this.result = Objects.requireNonNull(result, "result");
        this.state = cancelled ? State.CANCELLED : State.DONE;
    }

    void fail(Throwable error) {
        this.error = Objects.requireNonNull(error, "error");
        this.state = State.FAILED;
    }

    @Override
    public String toString() {
        State s = state;
        StringBuilder sb = new StringBuilder("BatchSequence#").append(submitId).append(' ').append(s);
        if (s == State.QUEUED) {
            return sb.toString();
        }
        sb.append(" seq ").append(seq);
        if (s == State.FAILED) {
            sb.append(": ").append(error);
        } else if (result != null) {
            sb.append(": ").append(result.stopReason());
        }
        return sb.toString();
    }
}
