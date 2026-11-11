package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The per-replica limit on generation requests waiting to run ({@link ServerOptions#maxQueuedRequests}, SGLang's
 * {@code --max-queued-requests}), counted in requests ({@link QueueAdmission}), not in generations. A replica holds
 * a request waiting when the request has a generation there without a sequence slot that the replica's free slots
 * do not take: the requests' waiting generations claim the free slots in admission order, and each request left
 * with a generation no free slot takes counts once. A replica with free slots therefore holds nothing waiting, and
 * an idle replica admits a request of any size.
 *
 * <p>A request that is not yet admitted goes to the replica the scheduler routed it to while that replica holds
 * fewer than the limit of waiting requests, else to the routed replica holding the fewest when that one has room
 * (the prompt's prefix affinity then moves there); when every replica is full it is refused with
 * {@link EngineSaturatedException} before any of its generations is submitted. An admitted request's later
 * generations follow the routing and are never refused: one that the routing sends to a full replica goes to the
 * replica that admitted the request.
 *
 * <p>Placement is serialized, and a request counts from the moment it is placed, so two concurrent requests never
 * both take the last place of a replica. A limit of 0 places every generation where the routing sends it and tracks
 * nothing.
 */
final class QueueLimit {

    /** Submits one generation to the replica the limit placed it on. */
    @FunctionalInterface
    interface Placement {
        EngineService.Submission submit(EngineReplica replica) throws InterruptedException;
    }

    /** A request with generations waiting on the replica being counted. */
    private record Holder(long ticket, int waiting) { }

    private final DpScheduler scheduler;
    private final int limit;
    /** The requests that may hold a generation in flight; a settled one leaves on the next count. */
    private final Set<QueueAdmission> live = ConcurrentHashMap.newKeySet();
    /** The admission order; guarded by this. */
    private long tickets;

    QueueLimit(DpScheduler scheduler, int limit) {
        if (limit < 0) {
            throw new IllegalArgumentException(ServerOptions.MAX_QUEUED_PROPERTY + " must be 0 or more, got "
                    + limit);
        }
        this.scheduler = scheduler;
        this.limit = limit;
    }

    /** The most waiting requests per replica; 0 for no limit. */
    int limit() {
        return limit;
    }

    /**
     * Submits one generation of {@code admission}'s request (null for a request of its own) through
     * {@code placement}: on {@code picked}, the replica the scheduler routed {@code routing} to, or where the limit
     * places it.
     *
     * @throws EngineSaturatedException when the request is not yet admitted and every replica is full
     */
    EngineService.Submission submit(QueueAdmission admission, EngineReplica picked, int[] routing,
                                    Placement placement) throws InterruptedException {
        if (limit == 0) {
            return placement.submit(picked);
        }
        QueueAdmission request = admission != null ? admission : new QueueAdmission();
        EngineReplica replica = place(request, picked, routing);
        try {
            EngineService.Submission submission = placement.submit(replica);
            request.placed(replica, submission);
            return submission;
        } finally {
            request.released();
        }
    }

    private synchronized EngineReplica place(QueueAdmission request, EngineReplica picked, int[] routing) {
        EngineReplica replica;
        if (request.admitted()) {
            replica = picked == request.home() || waiting(picked) < limit ? picked : request.home();
        } else {
            replica = waiting(picked) < limit ? picked : roomiest();
            if (replica == null) {
                throw new EngineSaturatedException("the request queue is full: every replica holds " + limit
                        + " request(s) waiting to run (" + ServerOptions.MAX_QUEUED_PROPERTY + " = " + limit
                        + "); retry after " + EngineSaturatedException.RETRY_AFTER_SECONDS + " second(s)");
            }
            if (replica != picked && routing != null) {
                scheduler.rerouted(picked, replica, routing);
            }
        }
        request.placing(replica, request.admitted() ? request.ticket() : ++tickets);
        live.add(request);
        return replica;
    }

    /**
     * The routed replica holding the fewest waiting requests, the first in pool order on a tie; null when every
     * replica is full.
     */
    private EngineReplica roomiest() {
        EngineReplica best = null;
        int least = limit;
        for (EngineReplica replica : scheduler.routed()) {
            int waiting = waiting(replica);
            if (waiting < least) {
                least = waiting;
                best = replica;
            }
        }
        return best;
    }

    /**
     * The requests waiting on {@code replica}: the waiting generations claim the replica's free sequence slots in
     * admission order, and each request left with a generation no free slot takes counts once.
     */
    synchronized int waiting(EngineReplica replica) {
        List<Holder> holders = new ArrayList<>();
        int running = 0;
        for (Iterator<QueueAdmission> it = live.iterator(); it.hasNext(); ) {
            QueueAdmission request = it.next();
            QueueAdmission.Load load = request.loadOn(replica);
            running += load.running();
            if (load.waiting() > 0) {
                holders.add(new Holder(request.ticket(), load.waiting()));
            } else if (load.running() == 0 && request.settled()) {
                it.remove();
            }
        }
        holders.sort(Comparator.comparingLong(Holder::ticket));
        int free = Math.max(0, replica.sequenceCapacity() - running);
        int waiting = 0;
        for (Holder holder : holders) {
            if (holder.waiting() <= free) {
                free -= holder.waiting();
            } else {
                free = 0;
                waiting++;
            }
        }
        return waiting;
    }
}
