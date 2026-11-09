package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;

import java.util.ArrayList;
import java.util.List;

/**
 * The pool's live weight transfer control plane (vLLM's in-process weight transfer engine): {@link #init} creates
 * an updater over every replica's open model, {@link #start} opens an update round, {@link #apply} writes the
 * round's tensors, {@link #finish} commits it and stamps its version, and {@link #shutdown} releases the updaters
 * without touching the models. Each operation applies to the whole pool and runs on every replica's pump thread,
 * so the writes land between that engine's steps; a write also drops the idle replica's batch engine, whose cached
 * prefixes were computed under the old weights (vLLM leaves {@code reset_prefix_cache} to the RL loop, and the
 * drop is this loop's reset). The committed version is a caller-managed label the {@code /weight_info} route
 * reports and every fresh updater starts with, so the replicas of a re-initialized pool agree with it; it survives
 * a shutdown, and a sleep that releases the weights also releases the updaters, so a wake serves again without one
 * until the trainer initializes the transfer anew.
 *
 * <p>Every operation takes a wait bound in milliseconds for the per-replica pump calls; 0 or less takes the
 * replica's request budget. The collective RPC route passes its own timeout, the dedicated routes take the
 * budget.</p>
 */
final class EngineWeightTransfer {

    /** The wait bound of an operation that names none: the replica's request budget. */
    static final long DEFAULT_WAIT_MILLIS = -1;

    private final EngineService owner;
    /** The committed weight version the pool reports; stamped by {@link #finish} and {@link #setVersion}. */
    private String version = WeightUpdater.DEFAULT_VERSION;

    EngineWeightTransfer(EngineService owner) {
        this.owner = owner;
    }

    /**
     * Initializes the in-process transfer on every replica, releasing any updaters a previous initialization
     * left. The fresh updaters start at the pool's committed version, so a re-initialization keeps the label.
     *
     * @param backend the transfer backend the trainer names; null or blank takes
     *                {@link WeightUpdater#INPROCESS_BACKEND}
     * @return the committed weight version at the return
     * @throws IllegalArgumentException      when the backend is not {@link WeightUpdater#INPROCESS_BACKEND}
     * @throws UnsupportedOperationException when the served model has no engine replica to update (a CLIP dual
     *                                       encoder), or a replica's model takes no live weight update
     * @throws IllegalStateException         when no model is loaded, or a pump did not answer in time
     */
    synchronized String init(String backend) throws InterruptedException {
        return init(backend, DEFAULT_WAIT_MILLIS);
    }

    /**
     * Initializes the transfer on every replica with the given wait bound; see {@link #init(String)}.
     *
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     */
    synchronized String init(String backend, long waitMillis) throws InterruptedException {
        String resolved = backend == null || backend.isBlank() ? WeightUpdater.INPROCESS_BACKEND : backend.trim();
        if (!resolved.equalsIgnoreCase(WeightUpdater.INPROCESS_BACKEND)) {
            throw new IllegalArgumentException("the weight transfer backend '" + resolved + "' is not registered:"
                    + " '" + WeightUpdater.INPROCESS_BACKEND + "' (the in-process updater) is the only backend;"
                    + " the NCCL, IPC and M2N transports are cross-process");
        }
        if (owner.replicas().isEmpty()) {
            throw new UnsupportedOperationException("the served model has no engine replica to update:"
                    + " a CLIP dual encoder takes no live weight updates");
        }
        for (EngineReplica replica : owner.replicas()) {
            replica.weightTransferOps().initWeightTransferOp(version, waitMillis);
        }
        return version;
    }

    /**
     * Releases every replica's updater (vLLM's {@code shutdown_weight_transfer_engine}); without one it is a
     * no-op.
     *
     * @throws IllegalStateException when a pump did not answer in time
     */
    synchronized void shutdown() throws InterruptedException {
        shutdown(DEFAULT_WAIT_MILLIS);
    }

    /**
     * Releases every replica's updater with the given wait bound; see {@link #shutdown()}.
     *
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     */
    synchronized void shutdown(long waitMillis) throws InterruptedException {
        for (EngineReplica replica : owner.replicas()) {
            replica.weightTransferOps().shutdownWeightTransferOp(waitMillis);
        }
    }

    /**
     * Opens an update round on every replica.
     *
     * @throws IllegalStateException when the transfer is not initialized, or a pump did not answer in time
     */
    synchronized void start() throws InterruptedException {
        start(DEFAULT_WAIT_MILLIS);
    }

    /**
     * Opens an update round on every replica with the given wait bound; see {@link #start()}.
     *
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     */
    synchronized void start(long waitMillis) throws InterruptedException {
        requireInitialized();
        for (EngineReplica replica : owner.replicas()) {
            replica.weightTransferOps().startWeightUpdateOp(waitMillis);
        }
    }

    /**
     * Writes {@code tensors} into every replica, in order; a refusal stops the rest, and the replicas earlier
     * tensors already reached keep what they hold.
     *
     * @throws IllegalStateException when the transfer is not initialized, an engine is running requests, or a
     *                               pump did not answer in time
     */
    synchronized void apply(List<WeightUpdateTensor> tensors) throws InterruptedException {
        apply(tensors, DEFAULT_WAIT_MILLIS);
    }

    /**
     * Writes {@code tensors} into every replica with the given wait bound; see {@link #apply(List)}.
     *
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     */
    synchronized void apply(List<WeightUpdateTensor> tensors, long waitMillis) throws InterruptedException {
        requireInitialized();
        for (WeightUpdateTensor tensor : tensors) {
            for (EngineReplica replica : owner.replicas()) {
                replica.weightTransferOps().applyWeightOp(tensor, waitMillis);
            }
        }
    }

    /**
     * Commits the update round on every replica; {@code version}, when given, becomes the committed one
     * ({@code finish_weight_update}'s optional {@code weight_version}).
     *
     * @return the committed weight version at the return
     * @throws IllegalStateException when the transfer is not initialized, an engine is running requests, or a
     *                               pump did not answer in time
     */
    synchronized String finish(String version) throws InterruptedException {
        return finish(version, DEFAULT_WAIT_MILLIS);
    }

    /**
     * Commits the update round on every replica with the given wait bound; see {@link #finish(String)}.
     *
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     */
    synchronized String finish(String version, long waitMillis) throws InterruptedException {
        requireInitialized();
        for (EngineReplica replica : owner.replicas()) {
            replica.weightTransferOps().finishWeightUpdateOp(version, waitMillis);
        }
        if (version != null) {
            this.version = version;
        }
        return this.version;
    }

    /**
     * Stamps the weight version without writing weights ({@code update_weight_version}).
     *
     * @return the stamped version
     * @throws IllegalStateException when the transfer is not initialized, or a pump did not answer in time
     */
    synchronized String setVersion(String version) throws InterruptedException {
        return setVersion(version, DEFAULT_WAIT_MILLIS);
    }

    /**
     * Stamps the weight version on every replica with the given wait bound; see {@link #setVersion(String)}.
     *
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     */
    synchronized String setVersion(String version, long waitMillis) throws InterruptedException {
        requireInitialized();
        for (EngineReplica replica : owner.replicas()) {
            replica.weightTransferOps().setWeightVersionOp(version, waitMillis);
        }
        this.version = version;
        return version;
    }

    /**
     * The weight version every replica reports ({@code get_weight_version}), in pool order; a replica without an
     * updater (the transfer never initialized, or shut down) reports the pool's committed label.
     *
     * @throws IllegalStateException when a pump did not answer in time
     */
    synchronized List<String> versions() throws InterruptedException {
        return versions(DEFAULT_WAIT_MILLIS);
    }

    /**
     * The weight versions of the replicas with the given wait bound; see {@link #versions()}.
     *
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     */
    synchronized List<String> versions(long waitMillis) throws InterruptedException {
        List<String> out = new ArrayList<>(owner.replicas().size());
        for (EngineReplica replica : owner.replicas()) {
            String replicaVersion = replica.weightTransferOps().weightVersionOp(waitMillis);
            out.add(replicaVersion == null ? version : replicaVersion);
        }
        return out;
    }

    /** The committed weight version ({@code /weight_info}); {@code default} before the first stamp. */
    synchronized String version() {
        return version;
    }

    /** Whether any replica holds a live updater. */
    synchronized boolean initialized() {
        for (EngineReplica replica : owner.replicas()) {
            if (replica.weightTransfer() != null) {
                return true;
            }
        }
        return false;
    }

    /** Refuses the update operations of a pool the trainer never initialized. */
    private void requireInitialized() {
        if (!initialized()) {
            throw new IllegalStateException("the weight transfer engine is not initialized;"
                    + " POST /init_weight_transfer_engine first");
        }
    }
}
