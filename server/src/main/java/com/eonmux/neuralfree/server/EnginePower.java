package com.eonmux.neuralfree.server;

import java.util.List;
import java.util.Locale;

/**
 * The pool power operations of one {@link EngineService}: sleep and wake (vLLM's sleep mode, llama.cpp's
 * {@code --sleep-idle-seconds}), pause and resume, the KV cache release, and the checkpoint capture and restore that
 * run over every replica. Each operation applies to the whole pool and reports the pool's least advanced state; the
 * service owns the replicas and the media projector these operations release and re-open.
 */
final class EnginePower {

    private final EngineService owner;

    EnginePower(EngineService owner) {
        this.owner = owner;
    }

    /**
     * Sleeps every replica at {@code level} — 0 pauses the scheduling only, 1 also drops the KV cache contents,
     * 2 also closes the model and releases its weights — treating the running requests as {@code mode} says; blocks
     * until every replica sleeps.
     *
     * @throws IllegalArgumentException      when the level is outside {@code [0, 2]}, or the keep mode would lose
     *                                       the frozen requests
     * @throws IllegalStateException         when no model is loaded, or a replica stayed busy past the request
     *                                       budget
     * @throws UnsupportedOperationException when a checkpoint capture is under way
     */
    void sleep(int level, SleepControl.Mode mode) throws InterruptedException {
        if (level < 0 || level > 2) {
            throw new IllegalArgumentException("level must be in [0, 2], got " + level);
        }
        if (mode == SleepControl.Mode.KEEP && level >= 2) {
            throw new IllegalArgumentException("the keep mode keeps its requests resumable; a level-2 sleep closes"
                    + " the engine they wait on — use the abort or wait mode");
        }
        owner.requireLoaded();
        SleepControl.Checkpoint checkpoint = lowestCheckpoint();
        if (checkpoint != SleepControl.Checkpoint.RUNNING) {
            throw new IllegalStateException("cannot sleep while the checkpoint state is " + checkpoint);
        }
        for (EngineReplica replica : owner.replicas()) {
            replica.sleepOp(level, mode);
        }
        if (level >= 1) {
            owner.media().clearCache();
        }
        if (level >= 2) {
            owner.releaseProjectorIfAsleep(); // the replicas' hooks skip it while a wake was in flight
        }
    }

    /**
     * Wakes every sleeping replica: the memory a sleep of level 1 or more released restores, then the media
     * projector, and the scheduling resumes.
     *
     * @param tags what to restore: none, or a {@code weights} or {@code kv} tag restores the memory;
     *             {@code scheduling} alone ends only a level-0 sleep, so a deeper sleep stays sleeping
     * @return {@code awake} when every replica serves again, {@code sleeping} when a scheduling-only wake left
     *         memory released
     * @throws IllegalArgumentException when a tag is unknown
     * @throws IllegalStateException    when no model is configured, or a wake outlived the request budget or
     *                                  failed, the projector's reopen included
     */
    String wakeUp(List<String> tags) throws InterruptedException {
        for (String tag : tags) {
            if (!tag.equals("scheduling") && !tag.equals("weights") && !tag.equals("kv")) {
                throw new IllegalArgumentException("unknown wake tag '" + tag
                        + "': the tags are scheduling, weights and kv");
            }
        }
        if (!owner.loaded() && !anySleeping()) {
            owner.requireLoaded();
        }
        if (!anySleeping()) {
            return "awake";
        }
        String state = "awake";
        owner.waking().incrementAndGet();
        try {
            for (EngineReplica replica : owner.replicas()) {
                if (replica.wakeUp(tags).equals("sleeping")) {
                    state = "sleeping";
                }
            }
            owner.reopenProjector();
        } finally {
            owner.waking().decrementAndGet();
        }
        return state;
    }

    /**
     * Pauses the generation of every replica.
     *
     * @throws IllegalStateException when no model is loaded, or a replica stayed busy past the request budget
     */
    void pause(SleepControl.Mode mode, boolean clearCache) throws InterruptedException {
        owner.requireLoaded();
        for (EngineReplica replica : owner.replicas()) {
            replica.pauseOp(mode, clearCache);
        }
    }

    /** Resumes the generation after a pause. */
    void resume() throws InterruptedException {
        for (EngineReplica replica : owner.replicas()) {
            replica.resumeOp();
        }
    }

    /**
     * Drops the KV cache contents of every replica, so its memory serves new requests. The weights stay resident.
     *
     * @throws UnsupportedOperationException when a request still runs, holds or waits for pages
     * @throws IllegalStateException         when no model is loaded
     */
    void releaseKvCacheMemory() throws InterruptedException {
        owner.requireLoaded();
        for (EngineReplica replica : owner.replicas()) {
            replica.releaseKvOp();
        }
        owner.media().clearCache();
    }

    /**
     * Starts a checkpoint capture on every replica: drain as {@code mode} says, then release the transient state a
     * capture must not contain and keep new work out until {@link #restoreCheckpoint()}. Repeating a preparation is
     * a no-op.
     *
     * @return the pool's checkpoint state name at the return
     * @throws IllegalStateException when no model is loaded, a replica sleeps, a preparation cannot start from the
     *                               current state, or an abort drain stayed busy past the request budget
     */
    String prepareCheckpoint(SleepControl.Mode mode) throws InterruptedException {
        if (!owner.loaded() && !anySleeping()) {
            throw new IllegalStateException("no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
        if (anySleeping()) {
            throw new IllegalStateException("the engine is sleeping; wake it before preparing a checkpoint");
        }
        SleepControl.Checkpoint current = lowestCheckpoint();
        if (current == SleepControl.Checkpoint.RESTORING || current == SleepControl.Checkpoint.FAILED) {
            throw new IllegalStateException("cannot prepare a checkpoint from " + current);
        }
        if (current == SleepControl.Checkpoint.RUNNING) {
            for (EngineReplica replica : owner.replicas()) {
                replica.prepareCheckpointOp(mode);
            }
            owner.media().clearCache();
        }
        return lowestCheckpoint().name().toLowerCase(Locale.ROOT);
    }

    /**
     * Restores a prepared checkpoint on every replica: the released state comes back and the pool serves again. A
     * restore in progress or a running pool is a no-op.
     *
     * @return the pool's checkpoint state name at the return
     * @throws IllegalStateException when a preparation is still running or the last one failed
     */
    String restoreCheckpoint() throws InterruptedException {
        if (!owner.loaded() && !anySleeping()) {
            throw new IllegalStateException("no model loaded: set -Dneuralfree.server.model=<gguf path>");
        }
        SleepControl.Checkpoint current = lowestCheckpoint();
        if (current == SleepControl.Checkpoint.PREPARING || current == SleepControl.Checkpoint.FAILED) {
            throw new IllegalStateException("cannot restore a checkpoint from " + current
                    + (firstCheckpointError() == null ? "" : ": " + firstCheckpointError()));
        }
        if (current == SleepControl.Checkpoint.PREPARED) {
            for (EngineReplica replica : owner.replicas()) {
                replica.restoreCheckpointOp();
            }
        }
        return lowestCheckpoint().name().toLowerCase(Locale.ROOT);
    }

    /** The pool's least advanced checkpoint state: the capture window is only as far as its slowest replica. */
    SleepControl.Checkpoint lowestCheckpoint() {
        SleepControl.Checkpoint lowest = SleepControl.Checkpoint.PREPARED;
        for (EngineReplica replica : owner.replicas()) {
            SleepControl.Checkpoint current = replica.checkpoint();
            if (rank(current) < rank(lowest)) {
                lowest = current;
            }
        }
        return owner.replicas().isEmpty() ? SleepControl.Checkpoint.RUNNING : lowest;
    }

    /** The refusal order of the checkpoint states: a failure first, a prepared pool last. */
    private static int rank(SleepControl.Checkpoint state) {
        return switch (state) {
            case FAILED -> 0;
            case PREPARING -> 1;
            case RESTORING -> 2;
            case RUNNING -> 3;
            case PREPARED -> 4;
        };
    }

    /** The first replica's checkpoint failure, or null. */
    String firstCheckpointError() {
        for (EngineReplica replica : owner.replicas()) {
            if (replica.checkpointError() != null) {
                return replica.checkpointError();
            }
        }
        return null;
    }

    /** Whether any replica holds released memory or frozen scheduling. */
    boolean anySleeping() {
        for (EngineReplica replica : owner.replicas()) {
            if (replica.sleeping()) {
                return true;
            }
        }
        return false;
    }

    /** Whether any replica holds released memory (a sleep of level 1 or more). */
    boolean isSleeping() {
        return anySleeping();
    }

    /** The deepest sleep level of the pool: 0 scheduling only, 1 the KV cache contents, 2 the weights too. */
    int sleepLevel() {
        int level = 0;
        for (EngineReplica replica : owner.replicas()) {
            level = Math.max(level, replica.sleepLevel());
        }
        return level;
    }

    /** Whether a pause is active on any replica. */
    boolean isPaused() {
        for (EngineReplica replica : owner.replicas()) {
            if (replica.paused()) {
                return true;
            }
        }
        return false;
    }

    /** The pool's checkpoint capture state as of now. */
    EngineService.CheckpointStatus checkpointStatus() {
        return new EngineService.CheckpointStatus(lowestCheckpoint().name().toLowerCase(Locale.ROOT),
                firstCheckpointError());
    }
}
