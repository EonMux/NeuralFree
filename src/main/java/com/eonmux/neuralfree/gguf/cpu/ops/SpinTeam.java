package com.eonmux.neuralfree.gguf.cpu.ops;

import com.eonmux.neuralfree.gguf.cpu.numa.NumaPlacement;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.LockSupport;

/**
 * A persistent team of worker threads that runs {@link CpuParallel#parallelFor} loops: a caller publishes its loop,
 * the members claim chunks of it from a shared counter, and the caller returns once every chunk has run. Between
 * loops the workers spin for a bounded time ({@code spinNanos}) before they park, so the many short loops of one
 * decode step start without a wake-up each, while an idle engine costs no CPU once the spin runs out.
 *
 * <p>Without placement a caller works on its own loop, so a team of {@code n} members starts {@code n - 1}
 * workers. Under an active {@link NumaPlacement} (NUMA nodes or a pinned CPU set) the team starts {@code n}
 * workers, worker {@code i} pinned as fork/join worker {@code i} would be, and the caller only waits, so every share
 * of a loop runs on the placement's CPUs. A
 * working caller that finds no chunk left spins for the same bounded time, then parks until the last chunk finishes;
 * a caller that only waits parks at once, so the unpinned caller takes no time slices from the pinned workers. A
 * caller whose interrupt status is set still parks, and keeps that status.
 *
 * <p>Callers on several threads share the team: up to {@value #SLOTS} loops are published at once and the workers
 * take one chunk of each in turn, so two callers' loops overlap on the team's threads instead of one of them
 * starting another pool's threads; a further caller waits for a slot. The chunks are exactly the ranges the
 * fork/join form runs ({@code [c * chunk, min(n, (c + 1) * chunk))}), so a loop body sees the same ranges whichever
 * form runs it; only the thread that runs a range differs. A loop started from inside a running loop, by its caller
 * or by a worker ({@link #isMember}), is not taken. The first exception a chunk throws is rethrown to the caller
 * after every claimed chunk has finished; chunks claimed after it are skipped. The team keeps no reference to a loop
 * once it returns. Thread-safe.
 */
final class SpinTeam {

    /** Loops the team runs at once; a caller beyond them waits for a slot. */
    static final int SLOTS = 2;

    private final Worker[] workers;
    private final boolean callerWorks;
    private final long spinNanos;
    private final long callerSpinNanos;
    private final Semaphore slotPermits = new Semaphore(SLOTS, true);
    private final AtomicReferenceArray<Job> slots = new AtomicReferenceArray<>(SLOTS);
    private final AtomicLong generation = new AtomicLong();
    private final AtomicInteger sleepers = new AtomicInteger();

    /**
     * A team of {@code members} threads running each loop.
     *
     * @param members   the threads that run a loop's chunks, at least 2
     * @param spinNanos how long an idle worker, or a working caller waiting for its loop's last chunks, spins before
     *                  it parks; positive
     * @param placement the NUMA placement the workers pin themselves with; when it is active every member is a
     *                  pinned worker and the caller only waits
     * @throws IllegalArgumentException for fewer than two members or a non-positive spin
     */
    SpinTeam(int members, long spinNanos, NumaPlacement placement) {
        if (members < 2) {
            throw new IllegalArgumentException("a spin team needs at least two members, got " + members);
        }
        if (spinNanos <= 0) {
            throw new IllegalArgumentException("spin time must be positive, got " + spinNanos + " ns");
        }
        this.spinNanos = spinNanos;
        this.callerWorks = !placement.active();
        this.callerSpinNanos = callerWorks ? spinNanos : 0L;
        this.workers = new Worker[callerWorks ? members - 1 : members];
        for (int i = 0; i < workers.length; i++) {
            workers[i] = new Worker(this, placement, callerWorks ? i + 1 : i);
            workers[i].start();
        }
    }

    /** The threads that run a loop's chunks, the caller included when it works. */
    int members() {
        return callerWorks ? workers.length + 1 : workers.length;
    }

    /** Whether the caller runs chunks of its own loops: false under an active NUMA placement. */
    boolean callerWorks() {
        return callerWorks;
    }

    /** True when the calling thread is running inside a loop of this team, as its caller or as a worker. */
    boolean isMember() {
        Thread t = Thread.currentThread();
        return isWorker(t) || ownJob(t) != null;
    }

    /**
     * Whether the loop the calling member runs in has a chunk for every member, so a loop nested in it would only
     * add threads to cores the outer loop already keeps busy.
     */
    boolean saturated() {
        Thread t = Thread.currentThread();
        Job j = isWorker(t) ? ((Worker) t).current : ownJob(t);
        return j != null && j.chunks >= members();
    }

    private boolean isWorker(Thread t) {
        return t instanceof Worker && ((Worker) t).team == this;
    }

    /** The published loop whose caller is {@code t}, or null. */
    private Job ownJob(Thread t) {
        for (int i = 0; i < SLOTS; i++) {
            Job j = slots.get(i);
            if (j != null && j.caller == t) {
                return j;
            }
        }
        return null;
    }

    /**
     * Runs the ranges of {@code chunk} indices over {@code [0, n)} on the team, first waiting for a slot while
     * {@value #SLOTS} other callers' loops run.
     *
     * @return false, having run nothing, when the calling thread is a {@link #isMember member} of a running loop
     */
    boolean run(int n, int chunk, CpuParallel.IntRangeConsumer body) {
        if (isMember()) {
            return false;
        }
        slotPermits.acquireUninterruptibly();
        Job j = new Job(n, chunk, body, Thread.currentThread());
        int slot = claimSlot(j);
        try {
            generation.incrementAndGet();
            if (sleepers.get() > 0) {
                for (Worker w : workers) {
                    if (w.sleeping) {
                        LockSupport.unpark(w);
                    }
                }
            }
            boolean more = callerWorks;
            while (more) {
                more = j.runOne();
            }
            j.await(callerSpinNanos);
            j.rethrow();
        } finally {
            slots.set(slot, null);
            slotPermits.release();
        }
        return true;
    }

    /** Publishes {@code j} in a free slot, which the caller's permit guarantees. */
    private int claimSlot(Job j) {
        while (true) {
            for (int i = 0; i < SLOTS; i++) {
                if (slots.get(i) == null && slots.compareAndSet(i, null, j)) {
                    return i;
                }
            }
            Thread.onSpinWait();
        }
    }

    /** Runs chunks of the published loops, one chunk of each in turn, until none has a chunk left. */
    private void workAll(Worker w) {
        boolean ran;
        do {
            ran = false;
            for (int i = 0; i < SLOTS; i++) {
                Job j = slots.get(i);
                if (j != null) {
                    w.current = j;
                    ran |= j.runOne();
                    w.current = null;
                }
            }
        } while (ran);
    }

    /** One published loop and its claim and completion counters. */
    private static final class Job {
        private final int n;
        private final int chunk;
        private final int chunks;
        private final CpuParallel.IntRangeConsumer body;
        private final Thread caller;
        private final AtomicInteger next = new AtomicInteger();
        private final AtomicInteger done = new AtomicInteger();
        private final AtomicReference<Throwable> error = new AtomicReference<>();
        private volatile boolean callerParked;

        Job(int n, int chunk, CpuParallel.IntRangeConsumer body, Thread caller) {
            this.n = n;
            this.chunk = chunk;
            this.chunks = (int) (((long) n + chunk - 1) / chunk);
            this.body = body;
            this.caller = caller;
        }

        /**
         * Claims and runs one chunk, false when none is left; the member finishing the last chunk wakes a parked
         * caller.
         */
        boolean runOne() {
            int c = next.getAndIncrement();
            if (c >= chunks) {
                return false;
            }
            try {
                if (error.get() == null) {
                    int from = c * chunk;
                    body.accept(from, (int) Math.min(n, (long) from + chunk));
                }
            } catch (Throwable t) {
                error.compareAndSet(null, t);
            } finally {
                if (done.incrementAndGet() == chunks && callerParked) {
                    LockSupport.unpark(caller);
                }
            }
            return true;
        }

        /**
         * Waits until every chunk has finished: spinning up to {@code spinNanos} (none for 0), then parked. The
         * interrupt status is cleared while the caller parks, so a pending interrupt cannot turn the park into a
         * spin, and is set again before this returns.
         */
        void await(long spinNanos) {
            if (done.get() >= chunks) {
                return;
            }
            if (spinNanos > 0) {
                long deadline = System.nanoTime() + spinNanos;
                while (done.get() < chunks && System.nanoTime() - deadline < 0) {
                    Thread.onSpinWait();
                }
                if (done.get() >= chunks) {
                    return;
                }
            }
            callerParked = true;
            boolean interrupted = Thread.interrupted();
            try {
                while (done.get() < chunks) {
                    LockSupport.park(this);
                    interrupted |= Thread.interrupted();
                }
            } finally {
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        void rethrow() {
            Throwable t = error.get();
            if (t instanceof RuntimeException) {
                throw (RuntimeException) t;
            }
            if (t instanceof Error) {
                throw (Error) t;
            }
            if (t != null) {
                throw new IllegalStateException("a parallel loop range failed", t);
            }
        }
    }

    /**
     * A team worker: spins for the next loop, parks when none comes within the spin time. It counts loops from the
     * team's first generation, 0, not from the generation it sees when it starts, so a loop published before the
     * thread first runs is not missed.
     */
    private static final class Worker extends Thread {
        private final SpinTeam team;
        private final NumaPlacement placement;
        private final int index;
        private volatile boolean sleeping;
        private volatile Job current;

        Worker(SpinTeam team, NumaPlacement placement, int index) {
            super("nf-cpu-team-" + index);
            this.team = team;
            this.placement = placement;
            this.index = index;
            setDaemon(true);
            setContextClassLoader(null);
        }

        @Override
        public void run() {
            placement.pinCurrentThread(index);
            long seen = 0L;
            while (true) {
                long g = team.generation.get();
                if (g != seen) {
                    seen = g;
                    team.workAll(this);
                    continue;
                }
                long deadline = System.nanoTime() + team.spinNanos;
                while (team.generation.get() == seen && System.nanoTime() - deadline < 0) {
                    Thread.onSpinWait();
                }
                if (team.generation.get() == seen) {
                    sleeping = true;
                    team.sleepers.incrementAndGet();
                    try {
                        while (team.generation.get() == seen) {
                            LockSupport.park(team);
                        }
                    } finally {
                        team.sleepers.decrementAndGet();
                        sleeping = false;
                    }
                }
            }
        }
    }
}
