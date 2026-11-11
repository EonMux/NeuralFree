package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;

import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * One request's standing under the per-replica queue limit ({@link QueueLimit}, {@link
 * ServerOptions#maxQueuedRequests}): the generations a single request submits — the choices of {@code n} or of a
 * prompt array, every step of a beam search, every round of a server tool loop — share one admission. The limit
 * checks the request when its first generation is submitted; once admitted, its later generations are never refused
 * by the limit, so a request is admitted or refused as a whole.
 *
 * <p>A request submits its generations from one thread at a time, as every surface here does.
 */
public final class QueueAdmission {

    /** A request's generations on one replica: those without a sequence slot and those holding one. */
    record Load(int waiting, int running) {
        static final Load NONE = new Load(0, 0);
    }

    /** One submitted generation and the replica it went to. */
    private record Generation(EngineReplica replica, EngineService.Submission submission) {

        /** Whether the generation reached its terminal state or was refused. */
        boolean terminal() {
            if (submission.refused() != null) {
                return true;
            }
            BatchSequence sequence = submission.sequence();
            return sequence != null && sequence.finished();
        }

        /** Whether the generation waits for a sequence slot: not yet registered, or queued in the engine. */
        boolean waiting() {
            BatchSequence sequence = submission.sequence();
            return sequence == null || sequence.state() == BatchSequence.State.QUEUED;
        }
    }

    /** The generations not yet seen terminal; terminal ones leave on the next count. */
    private final Set<Generation> generations = ConcurrentHashMap.newKeySet();
    /** The replica that admitted the request, null before its first generation was placed. */
    private volatile EngineReplica home;
    /** The order the request was admitted in, among the requests of one {@link QueueLimit}. */
    private volatile long ticket;
    /** The replica a generation is being submitted to right now, counted as waiting there, or null. */
    private volatile EngineReplica placing;

    /** A request not yet admitted. */
    public QueueAdmission() {
    }

    /** Whether the limit admitted the request: its first generation was placed. */
    boolean admitted() {
        return home != null;
    }

    /** The replica that admitted the request, or null before it was admitted. */
    EngineReplica home() {
        return home;
    }

    /** The order the request was admitted in; 0 before it was admitted. */
    long ticket() {
        return ticket;
    }

    /**
     * Marks a generation as being submitted to {@code replica}; the first one admits the request there with
     * {@code ticket}, its place in the admission order.
     */
    void placing(EngineReplica replica, long ticket) {
        if (home == null) {
            this.ticket = ticket;
            home = replica;
        }
        placing = replica;
    }

    /** Records the generation {@code submission} that {@code replica} accepted. */
    void placed(EngineReplica replica, EngineService.Submission submission) {
        generations.add(new Generation(replica, submission));
    }

    /** Ends the submission {@link #placing} began, accepted or not. */
    void released() {
        placing = null;
    }

    /**
     * The request's generations on {@code replica}, a submission under way there counted as waiting. Drops the
     * generations that reached their terminal state.
     */
    Load loadOn(EngineReplica replica) {
        int waiting = placing == replica ? 1 : 0;
        int running = 0;
        for (Iterator<Generation> it = generations.iterator(); it.hasNext(); ) {
            Generation generation = it.next();
            if (generation.terminal()) {
                it.remove();
            } else if (generation.replica() == replica) {
                if (generation.waiting()) {
                    waiting++;
                } else {
                    running++;
                }
            }
        }
        return waiting == 0 && running == 0 ? Load.NONE : new Load(waiting, running);
    }

    /** Whether nothing of the request is in flight: no submission under way and every generation terminal. */
    boolean settled() {
        if (placing != null) {
            return false;
        }
        generations.removeIf(Generation::terminal);
        return generations.isEmpty();
    }
}
