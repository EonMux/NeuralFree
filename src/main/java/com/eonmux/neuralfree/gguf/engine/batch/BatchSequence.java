package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.sampling.ReasoningBudgetStage;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One request's handle in a {@link BatchEngine}: its paged sequence id once admitted, its lifecycle state and,
 * after {@link State#DONE} or {@link State#CANCELLED}, the finished {@link GenerationResult}. The engine owns the
 * underlying memory slot; the handle only reports. A finished or failed handle stays valid until the caller drops
 * it, but its sequence id is freed for reuse the moment the engine retires it.
 *
 * <p>States: {@link State#QUEUED} waits for a sequence slot and the pages of its prompt (a prefill that ran out
 * of pages rolls back and queues again); {@link State#RUNNING} prefills or decodes, and stays running while the
 * engine has it preempted (its pages freed, recomputed on re-admission, possibly under another sequence id); {@link State#DONE} and {@link State#CANCELLED} carry a {@link GenerationResult} (a
 * cancellation's result reports {@code StopReason.CANCELLED}); {@link State#FAILED} carries the engine
 * {@link Throwable} that ended it. A pooling request's {@code DONE} carries no result: its output rides the
 * {@link PoolingSequence}.
 */
public final class BatchSequence {

    /** Lifecycle of one submitted request. */
    public enum State { QUEUED, RUNNING, DONE, CANCELLED, FAILED }

    /** The {@link #admittedAtNanos} value of a request never admitted (still queued, refused or failed). */
    public static final long NEVER_ADMITTED = Long.MIN_VALUE;

    private final int submitId;
    private volatile State state = State.QUEUED;
    private volatile long admittedAtNanos = NEVER_ADMITTED;
    private volatile int seq = -1;
    private volatile int reusedPrefix;
    private volatile int preemptions;
    private volatile boolean truncated;
    private volatile GenerationResult result;
    private volatile List<TokenScore[]> promptLogprobs;
    private volatile Throwable error;
    private volatile Runnable onFinish;
    private volatile ReasoningBudgetStage budgetStage;

    BatchSequence(int submitId) {
        this.submitId = submitId;
    }

    /**
     * The tokens the prefix cache reused for this request (copied pages, not prefilled), or 0 without a cache
     * hit. The count sits inside the prompt: at least the last prompt token always prefills.
     */
    public int reusedPrefix() {
        return reusedPrefix;
    }

    void reused(int tokens) {
        this.reusedPrefix = tokens;
    }

    /**
     * How many times the engine preempted this request to free memory (vLLM's {@code num_preemptions}): each
     * preemption freed its pages and recomputed them on re-admission.
     */
    public int preemptions() {
        return preemptions;
    }

    /** Counts one preemption (the engine thread, as it frees the sequence's pages). */
    void preempted() {
        this.preemptions = preemptions + 1;
    }

    /**
     * Whether a context shift moved this request's stored context while it generated (llama.cpp's
     * {@code slot.truncated}): true once the engine shifted the sequence.
     */
    public boolean truncated() {
        return truncated;
    }

    /** Publishes the shifted flag (the engine thread, when a shift runs). */
    void truncated(boolean v) {
        this.truncated = v;
    }

    /**
     * Runs {@code callback} once when the request reaches a terminal state (done, cancelled or failed), on the
     * engine thread. Set it right after {@code submit}, before the first pump can retire the sequence.
     */
    public BatchSequence onFinish(Runnable callback) {
        this.onFinish = Objects.requireNonNull(callback, "callback");
        return this;
    }

    /**
     * The generation's reasoning-budget stage once the engine admits it, or null while it waits or when the request
     * armed none. The engine thread publishes it at admission; a control action reads it to end reasoning
     * mid-stream.
     */
    public ReasoningBudgetStage budgetStage() {
        return budgetStage;
    }

    /** Publishes the admitted generation's budget stage (the engine thread, at slot open). */
    void armed(ReasoningBudgetStage stage) {
        this.budgetStage = stage;
    }

    /** The engine-unique submission number, in submit order from 0. */
    public int submitId() {
        return submitId;
    }

    /**
     * The {@link System#nanoTime()} instant the engine first moved the request out of the queue into a slot, or
     * {@link #NEVER_ADMITTED} while it never had a slot (still queued, refused or failed). A preemption or a
     * rollback that queues the request again does not move it: the first admission counts (vLLM's
     * {@code scheduled_ts}, recorded once and kept over preemptions).
     */
    public long admittedAtNanos() {
        return admittedAtNanos;
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

    /**
     * The per-prompt-token log probabilities once the prefill read every prompt row back
     * ({@code prompt_logprobs}), or null before that — a response builder reads it when the sequence's first
     * output arrives, before the result exists.
     */
    public List<TokenScore[]> promptLogprobs() {
        return promptLogprobs;
    }

    /** Publishes the completed prompt logprobs capture (the engine thread, during prefill). */
    void promptLogprobs(List<TokenScore[]> capture) {
        this.promptLogprobs = Objects.requireNonNull(capture, "capture");
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
        if (admittedAtNanos == NEVER_ADMITTED) {
            admittedAtNanos = System.nanoTime();
        }
        this.seq = seq;
        this.state = State.RUNNING;
    }

    /** Back to the queue before any output: its prefill ran out of pages and rolled back. */
    void requeued() {
        this.seq = -1;
        this.reusedPrefix = 0;
        this.state = State.QUEUED;
    }

    void complete(GenerationResult result, boolean cancelled) {
        this.result = Objects.requireNonNull(result, "result");
        this.state = cancelled ? State.CANCELLED : State.DONE;
        Runnable callback = onFinish;
        if (callback != null) {
            callback.run();
        }
    }

    /** Completes a pooling request ({@code DONE}, no {@link GenerationResult}: the output rides the
     * {@link PoolingSequence}). */
    void pooledDone() {
        this.state = State.DONE;
        Runnable callback = onFinish;
        if (callback != null) {
            callback.run();
        }
    }

    void fail(Throwable error) {
        this.error = Objects.requireNonNull(error, "error");
        this.state = State.FAILED;
        Runnable callback = onFinish;
        if (callback != null) {
            callback.run();
        }
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
