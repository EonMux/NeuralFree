package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * The replica's live weight transfer operations (the RL trainer-to-inference weight sync, vLLM's
 * {@code /init_weight_transfer_engine} family): the per-replica {@link WeightUpdater} is created, driven and
 * released on the replica's pump thread, because the writes and the forward stack's rebuild must land between
 * engine steps. The replica's release paths ({@code releaseWeightsAndEngine}, close) forget the updater — the
 * released weights are the ones it wrote into.
 */
final class ReplicaWeightTransfer {

    private final EngineReplica owner;
    /** The live weight transfer updater ({@code EngineWeightTransfer}); null until initialized. */
    private volatile WeightUpdater weightTransfer;

    ReplicaWeightTransfer(EngineReplica owner) {
        this.owner = owner;
    }

    /** Forgets the updater without shutting it down: its weights are gone (a sleep of level 2 or the close). */
    void release() {
        weightTransfer = null;
    }

    /**
     * Creates this replica's live weight updater over its open model on the pump thread ({@link
     * EngineWeightTransfer}); a model without a forward pass refuses here by name. Repeating an initialization
     * releases the previous updater first. {@code version} is the pool's committed weight label, which the fresh
     * updater reports from the start so every replica answers {@code get_weight_version} with the served round.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws UnsupportedOperationException when the model takes no live weight update
     * @throws IllegalStateException         when no model is open, or the pump did not answer in time
     */
    void initWeightTransferOp(String version, long waitMillis) throws InterruptedException {
        owner.requireOpen();
        transferOp("the weight transfer initialization", () -> {
            WeightUpdater previous = weightTransfer;
            if (previous != null) {
                previous.shutdown();
            }
            WeightUpdater fresh = owner.model().weightUpdater();
            if (version != null) {
                fresh.setVersion(version);
            }
            weightTransfer = fresh;
            return null;
        }, waitMillis);
    }

    /**
     * Releases this replica's live weight updater on the pump thread; none is no-op, so a shutdown without an
     * initialization stays idempotent.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     */
    void shutdownWeightTransferOp(long waitMillis) throws InterruptedException {
        transferOp("the weight transfer shutdown", () -> {
            WeightUpdater current = weightTransfer;
            weightTransfer = null;
            if (current != null) {
                current.shutdown();
            }
            return null;
        }, waitMillis);
    }

    /**
     * Opens an update round on the pump thread.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, or the pump did not answer in time
     */
    void startWeightUpdateOp(long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight update start", () -> {
            updater.start();
            return null;
        }, waitMillis);
    }

    /**
     * Writes one tensor on the pump thread, so the write and the forward stack's rebuild land between engine
     * steps: the idle batch engine drops first (its cached prefixes go with it, as vLLM's
     * {@code reset_prefix_cache} does for an RL round), then the write lands.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, the engine is running requests, or the
     *                               pump did not answer in time
     */
    void applyWeightOp(WeightUpdateTensor tensor, long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight update", () -> {
            owner.requireIdleEngine();
            owner.releaseIdleEngine();
            updater.apply(tensor);
            return null;
        }, waitMillis);
    }

    /**
     * Commits the update round on the pump thread, dropping the idle batch engine for the same reasons as
     * {@link #applyWeightOp}.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, the engine is running requests, or the
     *                               pump did not answer in time
     */
    void finishWeightUpdateOp(String version, long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight update finish", () -> {
            owner.requireIdleEngine();
            owner.releaseIdleEngine();
            updater.finish(version);
            return null;
        }, waitMillis);
    }

    /**
     * Stamps the weight version on the pump thread; it writes no weights, so a busy engine may stay running.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the transfer is not initialized, or the pump did not answer in time
     */
    void setWeightVersionOp(String version, long waitMillis) throws InterruptedException {
        WeightUpdater updater = requireTransfer();
        transferOp("the weight version change", () -> {
            updater.setVersion(version);
            return null;
        }, waitMillis);
    }

    /**
     * The version this replica's updater reports ({@code get_weight_version}), or null while the transfer is not
     * initialized (the pool then reports its committed label).
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     * @throws IllegalStateException when the pump did not answer in time
     */
    String weightVersionOp(long waitMillis) throws InterruptedException {
        WeightUpdater updater = weightTransfer;
        if (updater == null) {
            return null;
        }
        return transferOp("the weight version read", updater::version, waitMillis);
    }

    /** The replica's live weight transfer updater, or null while none is initialized. */
    WeightUpdater weightTransfer() {
        return weightTransfer;
    }

    /** The replica's updater, refusing by name when the transfer was never initialized or was shut down. */
    private WeightUpdater requireTransfer() {
        WeightUpdater updater = weightTransfer;
        if (updater == null) {
            throw new IllegalStateException("the weight transfer engine is not initialized;"
                    + " POST /init_weight_transfer_engine first");
        }
        return updater;
    }

    /**
     * Runs one live weight update call on the pump thread and waits for it; past the wait bound it refuses by
     * name, and a refusal of the call itself reaches the caller as its own exception.
     *
     * @param waitMillis how long the pump may take to answer; 0 or less takes the request budget
     */
    private <T> T transferOp(String what, Supplier<T> call, long waitMillis) throws InterruptedException {
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<RuntimeException> failure = new AtomicReference<>();
        AtomicReference<T> result = new AtomicReference<>();
        owner.work().put(() -> {
            try {
                result.set(call.get());
            } catch (RuntimeException e) {
                failure.set(e);
            } finally {
                done.countDown();
            }
        });
        owner.awaitOperation(done, what, waitMillis);
        if (failure.get() != null) {
            throw failure.get();
        }
        return result.get();
    }
}
