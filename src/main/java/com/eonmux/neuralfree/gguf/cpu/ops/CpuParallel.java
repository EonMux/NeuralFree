package com.eonmux.neuralfree.gguf.cpu.ops;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Data-parallel loops for CPU kernels on one shared {@link ForkJoinPool}.
 *
 * <p>The pool size comes from the system property {@value #THREADS_PROPERTY} (a positive integer) and
 * defaults to {@link Runtime#availableProcessors()}. Worker threads are daemons named {@code nf-cpu-N}.
 * Loops called from a pool worker run on the same pool (work stealing), so nesting does not deadlock.
 */
public final class CpuParallel {

    /** System property holding the worker count. */
    public static final String THREADS_PROPERTY = "neuralfree.cpu.threads";

    /** Receives one contiguous index range {@code [from, to)}. */
    @FunctionalInterface
    public interface IntRangeConsumer {
        void accept(int from, int to);
    }

    private static final class Holder {
        static final ForkJoinPool POOL = createPool(threadCount());
    }

    private CpuParallel() {
    }

    /**
     * Runs {@code body} over {@code [0, n)} split into contiguous ranges of at least {@code grain} indices
     * (the last range may be shorter). Returns after every range has run; an exception thrown by a range is
     * rethrown to the caller. Small loops ({@code n <= grain}) or a single-worker pool run inline.
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
        RangeTask task = new RangeTask(0, n, chunk, body);
        if (isWorkerOf(pool)) {
            task.invoke();
        } else {
            pool.invoke(task);
        }
    }

    /** Worker count of the shared pool. */
    public static int parallelism() {
        return Holder.POOL.getParallelism();
    }

    private static boolean isWorkerOf(ForkJoinPool pool) {
        Thread t = Thread.currentThread();
        return t instanceof ForkJoinWorkerThread && ((ForkJoinWorkerThread) t).getPool() == pool;
    }

    static int threadCount() {
        String v = System.getProperty(THREADS_PROPERTY);
        if (v == null || v.trim().isEmpty()) {
            return Runtime.getRuntime().availableProcessors();
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

    private static ForkJoinPool createPool(int threads) {
        AtomicInteger counter = new AtomicInteger();
        ForkJoinPool.ForkJoinWorkerThreadFactory factory = pool -> {
            ForkJoinWorkerThread t = ForkJoinPool.defaultForkJoinWorkerThreadFactory.newThread(pool);
            t.setName("nf-cpu-" + counter.getAndIncrement());
            t.setDaemon(true);
            return t;
        };
        return new ForkJoinPool(threads, factory, null, false);
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
