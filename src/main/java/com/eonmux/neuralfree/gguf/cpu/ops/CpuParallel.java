package com.eonmux.neuralfree.gguf.cpu.ops;

import com.eonmux.neuralfree.gguf.cpu.numa.CpuThreadPolicy;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaPlacement;
import org.apache.logging.log4j.LogManager;

import java.util.Locale;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Data-parallel loops for CPU kernels on one shared {@link ForkJoinPool}.
 *
 * <p>The pool size comes from the system property {@value #THREADS_PROPERTY} (a positive integer) and
 * defaults to the count {@link CpuThreadPolicy} derives from the host's cores ({@link #threadCount()}). Worker
 * threads are daemons named {@code nf-cpu-N}.
 * Loops called from a pool worker run on the same pool (work stealing), so nesting does not deadlock.
 *
 * <p>A loop called from outside the pool runs on a {@link SpinTeam} of the same size, whose workers spin up to
 * {@value #SPIN_PROPERTY} microseconds for the next loop before they park (default {@value #DEFAULT_SPIN_US}; 0,
 * which leaves every loop to the fork/join pool, for a pool larger than the available processors): a decode step is
 * hundreds of short loops, and each fork/join entry would otherwise wake parked workers. The caller works as one of
 * its members, except under NUMA placement, where every member is a pinned worker. Callers on several threads share
 * the team: two loops run on its threads at once, and a further caller waits for one. A loop nested in a team loop
 * runs inline when the outer loop already has a chunk for every member, and on the fork/join pool otherwise. The team
 * runs the same index ranges as the fork/join form; {@code neuralfree.cpu.spin.us=0} keeps every loop on the
 * fork/join pool.
 *
 * <p>When {@value com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy#PROPERTY} selects a NUMA strategy and
 * the host has more than one node, worker {@code N} pins itself to its node's CPUs when it starts
 * ({@link NumaPlacement}); the pool is created on first use, so the strategy must be set before that, as
 * the demos and the server do before they open a model.
 */
public final class CpuParallel {

    /** System property holding the worker count. */
    public static final String THREADS_PROPERTY = "neuralfree.cpu.threads";

    /**
     * System property holding how long, in microseconds, an idle worker of the spin team spins for the next loop
     * before it parks; {@code 0} runs every loop on the fork/join pool instead of the team.
     */
    public static final String SPIN_PROPERTY = "neuralfree.cpu.spin.us";

    /** The spin time of an unset {@value #SPIN_PROPERTY}. */
    public static final long DEFAULT_SPIN_US = 100;

    /** Longest accepted spin time, one second. */
    static final long MAX_SPIN_US = 1_000_000;

    /** Receives one contiguous index range {@code [from, to)}. */
    @FunctionalInterface
    public interface IntRangeConsumer {
        void accept(int from, int to);
    }

    private static final class Holder {
        static final NumaPlacement NUMA = NumaPlacement.fromSystemProperties();
        static final ForkJoinPool POOL = createPool(logged(threadCount()), NUMA);
        static final SpinTeam TEAM = createTeam(POOL.getParallelism(),
                spinMicros(POOL.getParallelism(), Runtime.getRuntime().availableProcessors()), NUMA);
    }

    private static SpinTeam createTeam(int threads, long spinUs, NumaPlacement placement) {
        return threads > 1 && spinUs > 0 ? new SpinTeam(threads, spinUs * 1000L, placement) : null;
    }

    /**
     * The spin time of the {@value #SPIN_PROPERTY} property for a pool of {@code threads} on {@code processors}
     * available processors. Unset, it is {@value #DEFAULT_SPIN_US}, or 0 when the pool has more threads than
     * processors: spinning workers then take time slices from the ones holding chunks.
     *
     * @throws IllegalArgumentException when the value is not an integer in {@code [0, 1000000]}
     */
    static long spinMicros(int threads, int processors) {
        String v = System.getProperty(SPIN_PROPERTY);
        if (v == null || v.trim().isEmpty()) {
            if (threads > processors) {
                LogManager.getLogger(CpuParallel.class).info("CPU pool of {} thread(s) on {} available processor(s):"
                        + " loops run on the fork/join pool without a spinning team", threads, processors);
                return 0L;
            }
            return DEFAULT_SPIN_US;
        }
        long us;
        try {
            us = Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(SPIN_PROPERTY + " must be an integer number of microseconds, got '"
                    + v + "'", e);
        }
        if (us < 0 || us > MAX_SPIN_US) {
            throw new IllegalArgumentException(SPIN_PROPERTY + " must be in [0, " + MAX_SPIN_US + "], got " + us);
        }
        return us;
    }

    private static int logged(int threads) {
        String v = System.getProperty(THREADS_PROPERTY);
        String source = v == null || v.trim().isEmpty()
                ? "the " + CpuThreadPolicy.configuredPolicy().name().toLowerCase(Locale.ROOT)
                        + " thread policy over " + CpuThreadPolicy.host().describe()
                : THREADS_PROPERTY;
        LogManager.getLogger(CpuParallel.class).info("CPU pool: {} thread(s) from {}", threads, source);
        return threads;
    }

    private CpuParallel() {
    }

    /**
     * Runs {@code body} over {@code [0, n)} split into contiguous ranges of at least {@code grain} indices
     * (the last range may be shorter). Returns after every range has run; an exception thrown by a range is
     * rethrown to the caller. Small loops ({@code n <= grain}), a single-worker pool and a loop nested in a team
     * loop that has a chunk for every member run inline.
     *
     * @throws IllegalArgumentException when {@code n < 0} or {@code grain < 1}
     */
    public static void parallelFor(int n, int grain, IntRangeConsumer body) {
        if (n < 0) {
            throw new IllegalArgumentException("loop size must be non-negative, got " + n);
        }
        if (grain < 1) {
            throw new IllegalArgumentException("grain must be positive, got " + grain);
        }
        if (n == 0) {
            return;
        }
        ForkJoinPool pool = Holder.POOL;
        if (n <= grain || pool.getParallelism() == 1) {
            body.accept(0, n);
            return;
        }
        int maxChunks = pool.getParallelism() * 4;
        int chunk = (int) Math.max(grain, ((long) n + maxChunks - 1) / maxChunks);
        if (isWorkerOf(pool)) {
            new RangeTask(0, n, chunk, body).invoke();
            return;
        }
        SpinTeam team = Holder.TEAM;
        if (team != null) {
            if (!team.isMember()) {
                team.run(n, chunk, body);
                return;
            }
            if (team.saturated()) {
                body.accept(0, n);
                return;
            }
        }
        pool.invoke(new RangeTask(0, n, chunk, body));
    }

    /** Worker count of the shared pool. */
    public static int parallelism() {
        return Holder.POOL.getParallelism();
    }

    private static boolean isWorkerOf(ForkJoinPool pool) {
        Thread t = Thread.currentThread();
        return t instanceof ForkJoinWorkerThread && ((ForkJoinWorkerThread) t).getPool() == pool;
    }

    /**
     * The pool size: the {@value #THREADS_PROPERTY} property when it is set, else the count of the
     * {@link CpuThreadPolicy} the {@value CpuThreadPolicy#PROPERTY} property selects over the host's cores.
     *
     * @throws IllegalArgumentException when a property holds an invalid value
     */
    public static int threadCount() {
        String v = System.getProperty(THREADS_PROPERTY);
        if (v == null || v.trim().isEmpty()) {
            return CpuThreadPolicy.host().defaultThreads();
        }
        int n;
        try {
            n = Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(THREADS_PROPERTY + " must be a positive integer, got '" + v + "'", e);
        }
        if (n < 1) {
            throw new IllegalArgumentException(THREADS_PROPERTY + " must be a positive integer, got " + n);
        }
        return n;
    }

    private static ForkJoinPool createPool(int threads, NumaPlacement placement) {
        AtomicInteger counter = new AtomicInteger();
        ForkJoinPool.ForkJoinWorkerThreadFactory factory = pool -> {
            NumaWorkerThread t = new NumaWorkerThread(pool, placement, counter.getAndIncrement());
            t.setName("nf-cpu-" + t.workerIndex());
            t.setDaemon(true);
            t.setContextClassLoader(null);
            return t;
        };
        return new ForkJoinPool(threads, factory, null, false);
    }

    /** A pool worker that pins itself to its NUMA node's CPUs once the pool starts it. */
    private static final class NumaWorkerThread extends ForkJoinWorkerThread {
        private final NumaPlacement placement;
        private final int workerIndex;

        NumaWorkerThread(ForkJoinPool pool, NumaPlacement placement, int workerIndex) {
            super(pool);
            this.placement = placement;
            this.workerIndex = workerIndex;
        }

        int workerIndex() {
            return workerIndex;
        }

        @Override
        protected void onStart() {
            placement.pinCurrentThread(workerIndex);
        }
    }

    /** Splits {@code [from, to)} in halves until a range fits in one chunk. */
    private static final class RangeTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;

        private final int from;
        private final int to;
        private final int chunk;
        private final transient IntRangeConsumer body;

        RangeTask(int from, int to, int chunk, IntRangeConsumer body) {
            this.from = from;
            this.to = to;
            this.chunk = chunk;
            this.body = body;
        }

        @Override
        protected void compute() {
            int len = to - from;
            if (len <= chunk) {
                body.accept(from, to);
                return;
            }
            long chunks = ((long) len + chunk - 1) / chunk;
            int mid = (int) (from + (chunks / 2) * chunk);
            invokeAll(new RangeTask(from, mid, chunk, body), new RangeTask(mid, to, chunk, body));
        }
    }
}
