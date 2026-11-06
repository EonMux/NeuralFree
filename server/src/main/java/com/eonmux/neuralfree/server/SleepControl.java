package com.eonmux.neuralfree.server;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The sleep, pause and checkpoint state of the serving engine, with vLLM's semantics: {@code /sleep} pauses
 * scheduling and releases memory at its level (0 scheduling only, 1 the KV cache contents, 2 the weights too),
 * {@code /wake_up} restores it, {@code /pause} and {@code /resume} gate scheduling on their own axis, and the
 * checkpoint states track a capture window ({@code /checkpoint/prepare} drains and releases the transient state,
 * {@code /checkpoint/restore} serves again).
 *
 * <p>The pump thread owns every transition; the HTTP routes read the volatile views and, for a wait-mode
 * operation, block on the latch of the drain the pump thread completes once the engine quiesced. The refusals
 * name the route that clears the state. The idle sleep shares the sleeping state with the source {@link
 * Source#IDLE}: it wakes on the next request instead of refusing it.
 */
final class SleepControl {

    /** How an operation treats the requests already in flight (vLLM's PauseMode). */
    enum Mode {
        /** Cancel every queued and running request at once. */
        ABORT,
        /** Let the queued and running requests finish before the operation completes. */
        WAIT,
        /**
         * Freeze: nothing is scheduled and new requests queue until the state clears. A sleep of level 1 or more
         * preempts the running ones under it, which recompute when the engine wakes.
         */
        KEEP
    }

    /** What sent the engine to sleep: an explicit {@code /sleep}, or the idle timeout. */
    enum Source { REQUEST, IDLE }

    /** The checkpoint capture lifecycle (vLLM's CheckpointState). */
    enum Checkpoint { RUNNING, PREPARING, PREPARED, RESTORING, FAILED }

    private volatile boolean sleeping;
    private volatile int level;
    private volatile Source source = Source.REQUEST;
    private volatile boolean pauseNew;
    private volatile boolean pauseAll;
    private final AtomicBoolean abort = new AtomicBoolean();
    private volatile Checkpoint checkpoint = Checkpoint.RUNNING;
    private volatile String checkpointError;
    /** The wait-mode operation the pump thread owes once the engine quiesced. */
    private volatile PendingDrain pending;

    /** A wait-mode operation: the pump thread runs {@code action} once the engine quiesced, then opens the latch. */
    private record PendingDrain(Runnable action, CountDownLatch done) { }

    /** Whether the engine holds released memory (sleep level 1 or more). */
    boolean sleeping() {
        return sleeping;
    }

    /** The sleep level: 0 scheduling only, 1 the KV cache contents, 2 the weights too. */
    int level() {
        return level;
    }

    /** What sent the engine to sleep. */
    Source source() {
        return source;
    }

    /** Whether new requests must refuse ({@code /pause} in the abort and wait modes). */
    boolean pausedNew() {
        return pauseNew;
    }

    /** Whether a pause is active in any mode. */
    boolean paused() {
        return pauseNew || pauseAll;
    }

    /** Whether nothing is scheduled: a sleep, or a keep-mode pause. */
    boolean frozen() {
        return sleeping || pauseAll;
    }

    Checkpoint checkpoint() {
        return checkpoint;
    }

    String checkpointError() {
        return checkpointError;
    }

    boolean abortRequested() {
        return abort.get();
    }

    /**
     * Why generation must refuse right now, or null when requests are served: an explicit sleep, or a pause that
     * rejects new work. The message names the route that clears the state.
     */
    String schedulingRefusal() {
        if (sleeping && source == Source.REQUEST) {
            return "the engine is sleeping (level " + level + "); POST /wake_up to restore it";
        }
        if (pauseNew) {
            return "the engine is paused; POST /resume to serve again";
        }
        return null;
    }

    // ------------------------------------------------------------- transitions (pump thread)

    /** Enters the sleep at {@code newLevel}; it replaces the pause flags, which a wake clears anyway. */
    void slept(int newLevel, Source sleptBy) {
        this.level = newLevel;
        this.source = sleptBy;
        this.sleeping = true;
        this.pauseNew = false;
        this.pauseAll = false;
    }

    /** Clears the sleep and its scheduling pause. */
    void woke() {
        this.sleeping = false;
        this.pauseNew = false;
        this.pauseAll = false;
    }

    void pauseNew() {
        this.pauseNew = true;
    }

    /** The keep-mode pause: nothing is scheduled, requests keep queueing. */
    void pauseAll() {
        this.pauseAll = true;
        this.pauseNew = false;
    }

    void unpause() {
        this.pauseNew = false;
        this.pauseAll = false;
    }

    void requestAbort() {
        abort.set(true);
    }

    void clearAbort() {
        abort.set(false);
    }

    void checkpointPreparing() {
        checkpoint = Checkpoint.PREPARING;
        checkpointError = null;
    }

    void checkpointPrepared() {
        checkpoint = Checkpoint.PREPARED;
    }

    void checkpointRestoring() {
        checkpoint = Checkpoint.RESTORING;
    }

    void checkpointRunning() {
        checkpoint = Checkpoint.RUNNING;
        checkpointError = null;
    }

    void checkpointFailed(String error) {
        checkpoint = Checkpoint.FAILED;
        checkpointError = error;
    }

    /** Registers the action the pump thread runs once the engine quiesced; the returned latch opens then. */
    CountDownLatch onQuiesce(Runnable action) {
        CountDownLatch done = new CountDownLatch(1);
        pending = new PendingDrain(action, done);
        return done;
    }

    /** Whether a wait-mode operation waits for the engine to quiesce. */
    boolean draining() {
        return pending != null;
    }

    /** Runs the outstanding quiesce action when {@code quiesced} holds, then opens its caller's latch. */
    void completeDrainIfQuiesced(boolean quiesced) {
        PendingDrain drain = pending;
        if (drain == null || !quiesced) {
            return;
        }
        pending = null;
        try {
            drain.action().run();
        } finally {
            drain.done().countDown();
        }
    }
}
