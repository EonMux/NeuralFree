package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * The pool's elastic data-parallel scaling and fault tolerance (vLLM's {@code POST /scale_elastic_ep} and its
 * {@code /fault_tolerance} routes): a scale drains and closes the removed replicas or opens the grown ones and
 * installs the new pool, {@code retry} recovers every unhealthy or dead replica, and {@code scale_down} drops the
 * ranks a caller names off the HTTP thread (one instruction at a time, over this class's single-thread control
 * executor).
 * The procedures' bookkeeping — the elastic epoch, phase and error, and the per-rank fault states — lives here; the
 * pool's shape itself is the service's ({@code installPool}), whose replacement fires the pool-change listeners.
 */
final class EngineScaling {

    private final EngineService owner;

    /** The elastic-scale lifecycle the status route reports ({@link ElasticPhase}). */
    private volatile ElasticPhase elasticPhase = ElasticPhase.IDLE;
    /** The epoch of the latest elastic scale, null before the first. */
    private volatile String elasticEpoch;
    /** The requested data-parallel size of the latest elastic scale, null before the first. */
    private volatile Integer elasticRequestedSize;
    /** The failure that ended the latest elastic scale, null when it succeeded. */
    private volatile String elasticError;
    /** How many elastic-scale epochs ran; the next epoch is its successor. */
    private long elasticEpochs;
    /** Whether a fault-tolerance scale_down removed ranks, which ends elastic scaling for the pool. */
    private volatile boolean ranksRemoved;
    /** The fault-tolerance state per data-parallel rank; removed ranks leave their entry behind as the record. */
    private final Map<Integer, FaultState> faultStates = new java.util.concurrent.ConcurrentHashMap<>();
    /** Runs a fault-tolerance instruction off the HTTP thread, one at a time. */
    private final java.util.concurrent.ExecutorService controlExecutor =
            java.util.concurrent.Executors.newSingleThreadExecutor(task -> {
                Thread thread = new Thread(task, "neuralfree-server-control");
                thread.setDaemon(true);
                return thread;
            });

    /** The elastic-scale lifecycle of the pool, as vLLM's external elastic EP phases name it. */
    enum ElasticPhase {
        IDLE, PREPARING, COMMITTING, FAILED, COMPLETED;

        /** The phase's lowercase API name. */
        String label() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /** The elastic-scale state {@code POST /is_scaling_elastic_ep} reports. */
    record ElasticStatus(boolean scaling, String phase, String epoch, String error,
                         Integer requestedDataParallelSize) { }

    /**
     * One replica's fault-tolerance view: the engine's data-parallel rank, its health ({@code healthy},
     * {@code unhealthy} or {@code dead}), the failure that made it unhealthy, and the state of the fault-tolerance
     * instruction that last touched it ({@code recovering}, {@code failed} or null).
     */
    record ReplicaHealth(int id, String status, String faultInfo, String ftState, String lastFtRequestId,
                         String ftError) { }

    /** One rank's fault-tolerance bookkeeping; the instruction's state survives while the rank is served. */
    private static final class FaultState {
        private volatile String state;
        private volatile String lastRequestId;
        private volatile String error;
    }

    EngineScaling(EngineService owner) {
        this.owner = owner;
    }

    /** The elastic-scale state the route reports. */
    ElasticStatus elasticStatus() {
        ElasticPhase phase = elasticPhase;
        boolean scaling = phase == ElasticPhase.PREPARING || phase == ElasticPhase.COMMITTING;
        return new ElasticStatus(scaling, phase.label(), elasticEpoch, elasticError, elasticRequestedSize);
    }

    /** Whether an elastic scale may run: every rank of the deployment is local, and no scale_down removed any. */
    boolean elasticScalable() {
        List<EngineReplica> current = owner.replicas();
        return owner.dual() == null && !current.isEmpty() && !ranksRemoved
                && owner.dataParallelLayout().startRank() == 0 && current.size() == owner.dataParallelLayout().size();
    }

    /**
     * Resizes the pool to {@code newSize} replicas behind the one service (vLLM's {@code POST /scale_elastic_ep}):
     * a shrink first stops routing to the removed replicas and drains their requests for at most
     * {@code drainMillis}, then closes them; a grow opens the new replicas before routing to them. The global
     * data-parallel size becomes {@code newSize} because the server holds every rank of the deployment. A failure
     * leaves the pool as it was (a shrink that timed out restores the routing) and records the phase as failed.
     *
     * @throws IllegalArgumentException when the pool cannot scale elastically (unloaded, a dual encoder, ranks held
     *                                  by other servers, or a rank removed by fault tolerance), or the new size is
     *                                  unusable
     * @throws java.util.concurrent.TimeoutException when the removed replicas did not drain within the timeout
     * @throws IllegalStateException when opening the new replicas fails
     */
    void scaleElasticEp(int newSize, long drainMillis) throws InterruptedException,
            java.util.concurrent.TimeoutException {
        if (newSize < 1) {
            throw new IllegalArgumentException("new_data_parallel_size must be a positive integer, got " + newSize);
        }
        if (drainMillis < 1) {
            throw new IllegalArgumentException("drain_timeout must be a positive integer, got " + drainMillis);
        }
        synchronized (owner.poolLock) {
            if (owner.dual() != null) {
                throw new IllegalArgumentException("elastic scaling needs a generation model; this server serves a"
                        + " CLIP dual encoder");
            }
            if (owner.replicas().isEmpty()) {
                throw new IllegalArgumentException("elastic scaling needs a loaded model: set "
                        + "-Dneuralfree.server.model=<gguf path>");
            }
            if (!elasticScalable()) {
                throw new IllegalArgumentException("elastic scaling resizes a server that holds every rank of the"
                        + " deployment: this one holds " + owner.replicas().size() + " of "
                        + owner.dataParallelLayout().size() + " rank(s)"
                        + (ranksRemoved ? " after a fault-tolerance scale_down" : "")
                        + ", so its external deployment owns the topology");
            }
            elasticEpoch = Long.toString(++elasticEpochs);
            elasticRequestedSize = newSize;
            elasticError = null;
            elasticPhase = ElasticPhase.PREPARING;
            try {
                if (newSize < owner.replicas().size()) {
                    shrinkPool(newSize, drainMillis);
                } else if (newSize > owner.replicas().size()) {
                    growPool(newSize);
                }
                elasticPhase = ElasticPhase.COMMITTING;
                elasticPhase = ElasticPhase.COMPLETED;
            } catch (InterruptedException e) {
                elasticError = "the scale was interrupted";
                elasticPhase = ElasticPhase.FAILED;
                throw e;
            } catch (java.util.concurrent.TimeoutException | RuntimeException e) {
                elasticError = e.getMessage();
                elasticPhase = ElasticPhase.FAILED;
                throw e;
            }
        }
    }

    /** Stops routing to the trailing replicas, drains them, closes them, and installs the survivors. */
    private void shrinkPool(int newSize, long drainMillis)
            throws InterruptedException, java.util.concurrent.TimeoutException {
        List<EngineReplica> current = owner.replicas();
        List<EngineReplica> survivors = new ArrayList<>(current.subList(0, newSize));
        List<EngineReplica> removed = new ArrayList<>(current.subList(newSize, current.size()));
        owner.scheduler().replace(survivors.toArray(new EngineReplica[0]));
        try {
            waitForDrain(removed, drainMillis);
        } catch (InterruptedException | java.util.concurrent.TimeoutException | RuntimeException e) {
            owner.scheduler().replace(current.toArray(new EngineReplica[0]));
            throw e;
        }
        EngineService.closeAll(removed);
        owner.installPool(survivors, new DpLayout(newSize, owner.dataParallelLayout().startRank(),
                owner.dataParallelLayout().balance()));
    }

    /**
     * Waits until none of {@code replicas} holds a request. A request that picked a removed replica just before the
     * scheduler swap needs a settling window to land, so the check runs again after a short pause.
     */
    private static void waitForDrain(List<EngineReplica> replicas, long drainMillis)
            throws InterruptedException, java.util.concurrent.TimeoutException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(drainMillis);
        while (draining(replicas)) {
            if (System.nanoTime() >= deadline) {
                throw new java.util.concurrent.TimeoutException("the requests on the replicas being removed did"
                        + " not drain within " + drainMillis + " ms");
            }
            Thread.sleep(10);
        }
        Thread.sleep(50);
        while (draining(replicas)) {
            if (System.nanoTime() >= deadline) {
                throw new java.util.concurrent.TimeoutException("the requests on the replicas being removed did"
                        + " not drain within " + drainMillis + " ms");
            }
            Thread.sleep(10);
        }
    }

    /** Whether any of {@code replicas} still holds a request. */
    private static boolean draining(List<EngineReplica> replicas) {
        for (EngineReplica replica : replicas) {
            if (replica.load() > 0) {
                return true;
            }
        }
        return false;
    }

    /** Opens the replicas past the pool's end, then installs the grown pool. */
    private void growPool(int newSize) {
        List<EngineReplica> current = owner.replicas();
        List<EngineReplica> next = new ArrayList<>(current);
        try {
            for (int i = current.size(); i < newSize; i++) {
                next.add(owner.openReplica(i, owner.dataParallelLayout().startRank() + i));
            }
        } catch (IllegalArgumentException e) {
            closeFrom(next, current.size(), e);
            throw e;
        } catch (IOException | RuntimeException e) {
            closeFrom(next, current.size(), e);
            throw new IllegalStateException("the scale to " + newSize + " replicas failed: " + e.getMessage(), e);
        }
        owner.installPool(next, new DpLayout(newSize, owner.dataParallelLayout().startRank(),
                owner.dataParallelLayout().balance()));
    }

    /** Closes the replicas {@code next} gained past {@code from}, suppressing their failures on {@code cause}. */
    private static void closeFrom(List<EngineReplica> next, int from, Exception cause) {
        for (int i = from; i < next.size(); i++) {
            try {
                next.get(i).close();
            } catch (RuntimeException closeFailure) {
                cause.addSuppressed(closeFailure);
            }
        }
    }

    /** The fault-tolerance status of every replica still served, in pool order (vLLM's engine status list). */
    List<ReplicaHealth> faultToleranceEngines() {
        List<EngineReplica> current = owner.replicas();
        List<ReplicaHealth> out = new ArrayList<>(current.size());
        for (EngineReplica replica : current) {
            FaultState state = faultStates.get(replica.rank());
            String fault = replica.alive() ? replica.faultInfo() : "the replica's pump thread stopped";
            String status = !replica.alive() ? "dead" : fault == null ? "healthy" : "unhealthy";
            out.add(new ReplicaHealth(replica.rank(), status, fault,
                    state == null ? null : state.state,
                    state == null ? null : state.lastRequestId,
                    state == null ? null : state.error));
        }
        return out;
    }

    /**
     * Runs one fault-tolerance instruction off the HTTP thread (vLLM answers 202 and the caller polls the status).
     * {@code retry} recovers every unhealthy or dead replica; {@code scale_down} closes the ranks
     * {@code params.removed_dp_ranks} names, which must be unhealthy or dead, and drops them from the pool while
     * the deployment's global size keeps their slots.
     *
     * @throws IllegalArgumentException for an unknown instruction or unusable parameters, before anything changes
     */
    void submitFaultTolerance(String instruction, Map<?, ?> params, String requestId) {
        if ("retry".equals(instruction)) {
            submitRetry(requestId);
            return;
        }
        if ("scale_down".equals(instruction)) {
            submitScaleDown(params, requestId);
            return;
        }
        throw new IllegalArgumentException("Invalid instruction: '" + instruction + "'.");
    }

    /** Marks the unhealthy replicas recovering and schedules their recovery. */
    private void submitRetry(String requestId) {
        synchronized (owner.poolLock) {
            List<EngineReplica> targets = new ArrayList<>();
            for (EngineReplica replica : owner.replicas()) {
                if (!replica.alive() || replica.faultInfo() != null) {
                    targets.add(replica);
                }
            }
            markFaultState(requestId, targets, null);
            controlExecutor.execute(() -> runRetry(targets, requestId));
        }
    }

    /** Recovers {@code targets} one by one; a failure records the error on the replicas it could not recover. */
    private void runRetry(List<EngineReplica> targets, String requestId) {
        synchronized (owner.poolLock) {
            List<EngineReplica> live = new ArrayList<>(targets.size());
            for (EngineReplica replica : targets) {
                if (owner.replicas().contains(replica)) {
                    live.add(replica);
                }
            }
            if (live.isEmpty()) {
                markFaultFailure(requestId, owner.replicas(),
                        "no unhealthy engine to recover: every replica is healthy");
                return;
            }
            List<EngineReplica> failed = new ArrayList<>();
            String error = null;
            for (EngineReplica replica : live) {
                try {
                    if (!replica.alive()) {
                        replaceReplica(replica);
                    } else {
                        replica.recoverOp();
                    }
                } catch (IOException | InterruptedException | RuntimeException e) {
                    if (e instanceof InterruptedException) {
                        Thread.currentThread().interrupt();
                    }
                    failed.add(replica);
                    error = error == null ? e.getMessage() : error;
                }
            }
            if (failed.isEmpty()) {
                clearFaultState(live);
            } else {
                markFaultFailure(requestId, failed, error);
            }
        }
    }

    /** Opens a replacement for a dead replica at its pool position and rank, then closes the dead one. */
    private void replaceReplica(EngineReplica dead) throws IOException {
        List<EngineReplica> current = owner.replicas();
        int position = current.indexOf(dead);
        if (position < 0) {
            throw new IllegalStateException("the replica of rank " + dead.rank() + " is no longer in the pool");
        }
        EngineReplica replacement = owner.openReplica(position, dead.rank());
        List<EngineReplica> next = new ArrayList<>(current);
        next.set(position, replacement);
        owner.installPool(next, owner.dataParallelLayout());
        dead.close();
    }

    /** Marks the removed ranks recovering and schedules their removal. */
    private void submitScaleDown(Map<?, ?> params, String requestId) {
        synchronized (owner.poolLock) {
            List<Integer> removed = removedRanks(params);
            List<EngineReplica> targets = new ArrayList<>();
            for (EngineReplica replica : owner.replicas()) {
                if (removed.contains(replica.rank())) {
                    targets.add(replica);
                }
            }
            markFaultState(requestId, targets, null);
            controlExecutor.execute(() -> runScaleDown(targets, requestId));
        }
    }

    /** The {@code removed_dp_ranks} of a scale_down, validated against the ranks still served and their health. */
    private List<Integer> removedRanks(Map<?, ?> params) {
        Object raw = params.get("removed_dp_ranks");
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw new IllegalArgumentException("scale_down needs 'removed_dp_ranks', a non-empty array of"
                    + " data-parallel ranks");
        }
        List<EngineReplica> current = owner.replicas();
        List<Integer> removed = new ArrayList<>(list.size());
        for (Object entry : list) {
            if (!(entry instanceof Number number) || number.intValue() != number.doubleValue()) {
                throw new IllegalArgumentException("removed_dp_ranks must hold whole numbers, got " + entry);
            }
            int rank = number.intValue();
            EngineReplica replica = null;
            for (EngineReplica candidate : current) {
                if (candidate.rank() == rank) {
                    replica = candidate;
                    break;
                }
            }
            if (replica == null) {
                throw new IllegalArgumentException("removed_dp_ranks names rank " + rank + ", which this server does"
                        + " not serve (it serves " + ranks(current) + ")");
            }
            if (replica.alive() && replica.faultInfo() == null) {
                throw new IllegalArgumentException("removed_dp_ranks names rank " + rank + ", which is healthy;"
                        + " scale_down removes failed ranks only");
            }
            if (!removed.contains(rank)) {
                removed.add(rank);
            }
        }
        if (current.size() <= removed.size()) {
            throw new IllegalArgumentException("scale_down cannot remove every rank: the pool needs at least one"
                    + " replica to keep serving");
        }
        return removed;
    }

    /** Recovers {@code removed} as far as closing goes: the ranks leave the pool and the layout keeps theirs. */
    private void runScaleDown(List<EngineReplica> removed, String requestId) {
        synchronized (owner.poolLock) {
            List<EngineReplica> current = owner.replicas();
            List<EngineReplica> next = new ArrayList<>(current.size());
            for (EngineReplica replica : current) {
                if (!removed.contains(replica)) {
                    next.add(replica);
                }
            }
            if (next.isEmpty()) {
                markFaultFailure(requestId, removed, "scale_down would remove every replica");
                return;
            }
            for (EngineReplica replica : removed) {
                if (owner.replicas().contains(replica) && replica.alive() && replica.faultInfo() == null) {
                    markFaultFailure(requestId, removed, "rank " + replica.rank() + " became healthy before the"
                            + " scale_down ran; failed ranks only");
                    return;
                }
            }
            EngineService.closeAll(removed);
            for (EngineReplica replica : removed) {
                faultStates.remove(replica.rank());
            }
            ranksRemoved = true;
            owner.installPool(next, owner.dataParallelLayout());
        }
    }

    /** The ranks {@code replicas} serve, comma-separated for a refusal. */
    private static String ranks(List<EngineReplica> replicas) {
        StringBuilder out = new StringBuilder();
        for (EngineReplica replica : replicas) {
            if (out.length() > 0) {
                out.append(", ");
            }
            out.append(replica.rank());
        }
        return out.toString();
    }

    /** Records {@code state} ({@code recovering}) on {@code targets}; {@code error} null on the recovering mark. */
    private void markFaultState(String requestId, List<EngineReplica> targets, String error) {
        for (EngineReplica replica : targets) {
            FaultState state = faultStates.computeIfAbsent(replica.rank(), rank -> new FaultState());
            state.state = "recovering";
            state.lastRequestId = requestId;
            state.error = error;
        }
    }

    /** Records a failed instruction on {@code targets}. */
    private void markFaultFailure(String requestId, List<EngineReplica> targets, String error) {
        for (EngineReplica replica : targets) {
            FaultState state = faultStates.computeIfAbsent(replica.rank(), rank -> new FaultState());
            state.state = "failed";
            state.lastRequestId = requestId;
            state.error = error;
        }
    }

    /** Clears the instruction state of recovered {@code targets}, as vLLM's healthy push replaces it. */
    private void clearFaultState(List<EngineReplica> targets) {
        for (EngineReplica replica : targets) {
            faultStates.remove(replica.rank());
        }
    }

    /** Stops the control executor; the service's close runs it once the pool winds down. */
    void close() {
        controlExecutor.shutdownNow();
    }
}
