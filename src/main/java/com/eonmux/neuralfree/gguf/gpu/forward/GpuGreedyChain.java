package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.engine.GreedyStep;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayDeque;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * The greedy decode steps a {@link GpuPrefillDriver} enqueued without waiting ({@code ForwardBackend.startGreedy}
 * and {@code chainGreedy}), oldest first, until their ids reach the host. A step completes when its id is asked
 * for or when the driver settles the chain before any other call: completing a step first completes the step whose
 * id it read on the device, then waits for the step's own id readback (never for the whole queue, so a step chained
 * behind it keeps the device busy), and finally writes the id into the sequence entry of the step chained on it,
 * which was stored before its id reached the host ({@link SequenceState#settleToken}).
 *
 * <p>A step is chained only on the latest step the driver enqueued, while nothing else ran since: the device slot
 * that step's selection sits in is overwritten by the next head. A step chained on one whose id already reached the
 * host stores that id at once. A step whose input step failed fails with the same exception, so every caller sees
 * the device's own failure. A completed step keeps only its id (or failure), so a long chain holds no more than its
 * outstanding steps. Not thread-safe; the driver's thread uses it.</p>
 */
final class GpuGreedyChain {

    /** The id a chained step's sequence entry holds until its input id reaches the host. */
    static final int UNSETTLED_TOKEN = -1;

    private final Supplier<SequenceState> sequence;
    private final DecodeStepBreakdown breakdown;
    private final ArrayDeque<Step> outstanding = new ArrayDeque<>();
    /** The latest step enqueued while nothing else ran on the driver since, the only one a step may chain on. */
    private Step latest;
    /** Steps registered as chained on the device since the chain was made, for diagnostics and tests. */
    private long deviceChained;

    /**
     * A chain recording into the sequence {@code sequence} supplies when a step completes, and reporting the time
     * spent waiting for ids to {@code breakdown}.
     */
    GpuGreedyChain(Supplier<SequenceState> sequence, DecodeStepBreakdown breakdown) {
        this.sequence = Objects.requireNonNull(sequence, "sequence");
        this.breakdown = Objects.requireNonNull(breakdown, "breakdown");
    }

    /**
     * Whether a step chained on {@code previous} reads its id on the device: {@code previous} is the latest step of
     * this chain and nothing else ran on the driver since.
     */
    boolean chainsOn(GreedyStep previous) {
        return previous != null && previous == latest;
    }

    /**
     * Registers the step whose reads {@code calls} enqueued; {@code previous} is the step whose device id it read
     * (null for a host token), and {@code entry} its sequence entry. A step chained on a {@code previous} still
     * outstanding stores {@link #UNSETTLED_TOKEN} there until {@code previous} completes; one chained on a completed
     * {@code previous} stored its id already ({@link #inputToken}).
     */
    Step register(GpuCallReads calls, Step previous, int entry) {
        Step step = new Step(this, calls, previous != null && previous.state == State.PENDING ? previous : null,
                entry);
        if (step.previous != null) {
            step.previous.successor = step;
        }
        if (previous != null) {
            deviceChained++;
        }
        outstanding.addLast(step);
        latest = step;
        return step;
    }

    /**
     * The id a step chained on {@code previous} stores in its sequence entry when it is enqueued: {@code previous}'s
     * id once it reached the host, else {@link #UNSETTLED_TOKEN}.
     */
    static int inputToken(Step previous) {
        return previous.state == State.DONE ? previous.id : UNSETTLED_TOKEN;
    }

    /** The steps that read their input id on the device since this chain was made. */
    long deviceChained() {
        return deviceChained;
    }

    /** Something else runs on the driver: no later step may chain on the latest one. */
    void interrupt() {
        latest = null;
    }

    /** Whether some step's id has not reached the host yet. */
    boolean hasOutstanding() {
        return !outstanding.isEmpty();
    }

    /** Completes every outstanding step, oldest first; the first failure propagates after all were attempted. */
    void settle() {
        RuntimeException failure = null;
        while (!outstanding.isEmpty()) {
            try {
                complete(outstanding.peekFirst());
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** Completes the steps up to, but not including, the {@code keep} newest outstanding ones. */
    void settleOlderThan(int keep) {
        while (outstanding.size() > keep) {
            complete(outstanding.peekFirst());
        }
    }

    private void complete(Step step) {
        if (step.state != State.PENDING) {
            if (step.failure != null) {
                throw step.failure;
            }
            return;
        }
        try {
            if (step.previous != null) {
                complete(step.previous);
            }
            long t0 = System.nanoTime();
            step.calls.await();
            breakdown.waited(System.nanoTime() - t0);
            step.id = step.calls.result().ids()[0];
            step.state = State.DONE;
            Step next = step.successor;
            SequenceState seq = sequence.get();
            if (next != null && seq != null && next.entry < seq.nPast()
                    && seq.token(next.entry) == UNSETTLED_TOKEN) {
                seq.settleToken(next.entry, step.id);
            }
        } catch (RuntimeException e) {
            step.state = State.FAILED;
            step.failure = e;
            throw e;
        } finally {
            if (step.state != State.PENDING) {
                outstanding.remove(step);
                if (latest == step && step.state == State.FAILED) {
                    latest = null;
                }
                GpuCallReads calls = step.calls;
                step.calls = null;
                step.previous = null;
                step.successor = null;
                calls.closeEvents();
            }
        }
    }

    private enum State {
        PENDING, DONE, FAILED
    }

    /**
     * One enqueued greedy step: its reads, the outstanding step it chained on and the step chained on it, until it
     * completes; then its id or its failure.
     */
    static final class Step implements GreedyStep {
        private final GpuGreedyChain chain;
        private final ScratchSlot idSlot;
        private final int entry;
        private GpuCallReads calls;
        private Step previous;
        private Step successor;
        private State state = State.PENDING;
        private int id;
        private RuntimeException failure;

        private Step(GpuGreedyChain chain, GpuCallReads calls, Step previous, int entry) {
            this.chain = chain;
            this.calls = calls;
            this.idSlot = calls.idSlot();
            this.previous = previous;
            this.entry = entry;
        }

        /** The device slot holding this step's greedy id until the next head overwrites it. */
        ScratchSlot idSlot() {
            return idSlot;
        }

        /** Whether the step still references an outstanding input step or its reads, for tests. */
        boolean holdsReferences() {
            return previous != null || successor != null || calls != null;
        }

        @Override
        public int id() {
            chain.complete(this);
            return id;
        }
    }
}
