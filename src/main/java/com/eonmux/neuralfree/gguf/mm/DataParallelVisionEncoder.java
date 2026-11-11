package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A data-parallel vision encoder (vLLM's {@code mm_encoder_tp_mode=data}): one whole-weight executor per rank,
 * every rank hosting the full encoder weights, and the images of an encode batch spread over the ranks. vLLM
 * disables the encoder's tensor parallelism for this mode ({@code ColumnParallelLinear(..., disable_tp=...)}) and
 * shards the input ({@code get_load_balance_assignment} / {@code run_dp_sharded_mrope_vision_model}); this class
 * does the same over the executors it was bound with.
 *
 * <ul>
 *   <li>A batch is assigned by pixel count: the largest images go first to the least loaded rank
 *       ({@link #assignment}), as vLLM balances a batch by the patches of its images rather than by image count, so
 *       the ranks finish together.</li>
 *   <li>Each rank encodes its share on its own thread ({@link #encodeBatch}); the results are reassembled in input
 *       order, so a data-parallel encode returns exactly the rows the sequential encoder does.</li>
 *   <li>{@link #encode} sends its one image to the next rank in turn, and {@link #encodeAnyres} runs on rank 0,
 *       whose graph assembles the overview and the tile grid as one image.</li>
 *   <li>Every encode runs on its rank's thread, so callers on several threads may encode at once
 *       ({@link #concurrent()}); a LoRA session is global to the ranks and needs the encoder to itself.</li>
 * </ul>
 *
 * The ranks are independent executors bound over the same model, so a rank's rows never depend on the other ranks'.
 * One task per rank is in flight at a time (each rank runs on its own single-thread executor); a failed rank's
 * exception is rethrown and the other ranks are waited for, so no rank is left computing into its scratch. An
 * interrupted caller returns only once none of its calls runs on a rank any more, and {@link #close} waits for a
 * running call before it closes the ranks.
 * Thread-safe; closing the encoder closes every rank and the owner of the ranks' backends.
 */
public final class DataParallelVisionEncoder implements VisionEncoder {

    private final List<VisionEncoder> ranks;
    private final List<ExecutorService> threads;
    private final String device;
    private final AutoCloseable owner;
    private final AtomicInteger next = new AtomicInteger();
    private volatile boolean closed;

    private DataParallelVisionEncoder(List<VisionEncoder> ranks, AutoCloseable owner) {
        if (ranks.isEmpty()) {
            throw new IllegalArgumentException("a data-parallel vision encoder needs at least one rank");
        }
        ClipVisionModel first = ranks.get(0).model();
        for (int i = 1; i < ranks.size(); i++) {
            ClipVisionModel model = ranks.get(i).model();
            if (model.outputDim() != first.outputDim()
                    || model.hparams().projectorType() != first.hparams().projectorType()) {
                throw new IllegalArgumentException("rank " + i + " encodes " + model.hparams().projectorType()
                        + " to " + model.outputDim() + " values, rank 0 encodes "
                        + first.hparams().projectorType() + " to " + first.outputDim()
                        + "; a data-parallel encoder needs one model on every rank");
            }
        }
        this.ranks = List.copyOf(ranks);
        this.owner = owner;
        List<ExecutorService> pool = new ArrayList<>(ranks.size());
        for (int i = 0; i < ranks.size(); i++) {
            pool.add(Executors.newSingleThreadExecutor(new RankThreads(i)));
        }
        this.threads = Collections.unmodifiableList(pool);
        StringBuilder names = new StringBuilder();
        for (VisionEncoder rank : ranks) {
            if (names.length() > 0) {
                names.append('+');
            }
            names.append(rank.device());
        }
        this.device = names.toString();
    }

    /**
     * A data-parallel encoder over the already-bound {@code ranks} (rank 0 is the encoder whose device
     * {@link #device()} lists first); the ranks, and nothing else, are closed with it.
     *
     * @throws IllegalArgumentException when no rank is given, or a rank encodes another model
     */
    public static DataParallelVisionEncoder of(List<? extends VisionEncoder> ranks) {
        Objects.requireNonNull(ranks, "ranks");
        return new DataParallelVisionEncoder(new ArrayList<VisionEncoder>(ranks), null);
    }

    /**
     * {@link #of(List)} with {@code owner}, released after every rank when the encoder closes (the backend group
     * the ranks were opened from).
     */
    static DataParallelVisionEncoder of(List<? extends VisionEncoder> ranks, AutoCloseable owner) {
        Objects.requireNonNull(ranks, "ranks");
        return new DataParallelVisionEncoder(new ArrayList<VisionEncoder>(ranks), owner);
    }

    /** The number of ranks the batch is spread over. */
    public int ranks() {
        return ranks.size();
    }

    /** The whole encoder of one rank, in rank order (read-only). */
    public List<VisionEncoder> executors() {
        return ranks;
    }

    @Override
    public float[][] encode(ClipImage image) {
        Objects.requireNonNull(image, "image");
        ensureOpen();
        int rank = Math.floorMod(next.getAndIncrement(), ranks.size());
        return run(rank, Collections.singletonList(image)).get(0);
    }

    /**
     * Encodes {@code images} with the batch spread over the ranks by pixel count, every rank's share on its own
     * thread; the results come back in input order.
     */
    @Override
    public List<float[][]> encodeBatch(List<ClipImage> images) {
        Objects.requireNonNull(images, "images");
        ensureOpen();
        if (images.isEmpty()) {
            return new ArrayList<>();
        }
        if (ranks.size() == 1) {
            return run(0, images);
        }
        int[] assignedTo = assignment(sizes(images), ranks.size());
        List<List<Integer>> shares = new ArrayList<>(ranks.size());
        for (int r = 0; r < ranks.size(); r++) {
            shares.add(new ArrayList<Integer>());
        }
        for (int i = 0; i < assignedTo.length; i++) {
            shares.get(assignedTo[i]).add(i);
        }
        List<RankTask<List<float[][]>>> tasks = new ArrayList<>(ranks.size());
        List<Future<List<float[][]>>> futures = new ArrayList<>(ranks.size());
        for (int r = 0; r < ranks.size(); r++) {
            final int rank = r;
            final List<Integer> share = shares.get(r);
            RankTask<List<float[][]>> task = new RankTask<>(() -> runShare(rank, images, share));
            tasks.add(task);
            futures.add(threads.get(r).submit(task));
        }
        List<float[][]> out = new ArrayList<>(Collections.nCopies(images.size(), (float[][]) null));
        RuntimeException failure = null;
        List<RuntimeException> suppressed = new ArrayList<>();
        for (int r = 0; r < ranks.size(); r++) {
            try {
                List<float[][]> rows = futures.get(r).get();
                List<Integer> share = shares.get(r);
                for (int i = 0; i < share.size(); i++) {
                    out.set(share.get(i), rows.get(i));
                }
            } catch (InterruptedException e) {
                tasks.get(r).withdrawOrAwait();
                Thread.currentThread().interrupt();
                if (failure == null) {
                    failure = new IllegalStateException("the data-parallel vision encoder was interrupted", e);
                }
            } catch (CancellationException e) {
                if (failure == null) {
                    failure = closedDuringTheCall(e);
                }
            } catch (ExecutionException e) {
                RuntimeException cause = rethrowable(e.getCause());
                if (failure == null) {
                    failure = cause;
                } else {
                    suppressed.add(cause);
                }
            }
        }
        if (failure != null) {
            for (RuntimeException e : suppressed) {
                failure.addSuppressed(e);
            }
            throw failure;
        }
        return out;
    }

    /**
     * Encodes a batch in input order on rank {@code r} while holding that rank's thread, so a share never runs
     * beside another share of the same rank's executor.
     */
    private List<float[][]> runShare(int r, List<ClipImage> images, List<Integer> share) {
        if (share.isEmpty()) {
            return Collections.emptyList();
        }
        List<ClipImage> local = new ArrayList<>(share.size());
        for (int i : share) {
            local.add(images.get(i));
        }
        return ranks.get(r).encodeBatch(local);
    }

    /** Encodes {@code images} on rank {@code rank} through its thread, awaiting the result. */
    private List<float[][]> run(int rank, List<ClipImage> images) {
        return onRank(rank, () -> ranks.get(rank).encodeBatch(images));
    }

    /**
     * The result of {@code call} run on rank {@code rank}'s thread. An interrupted caller withdraws the call when it
     * has not started, else waits until it ended ({@link RankTask}), and only then throws.
     */
    private <T> T onRank(int rank, Callable<T> call) {
        RankTask<T> task = new RankTask<>(call);
        Future<T> future = threads.get(rank).submit(task);
        try {
            return future.get();
        } catch (InterruptedException e) {
            task.withdrawOrAwait();
            Thread.currentThread().interrupt();
            throw new IllegalStateException("the data-parallel vision encoder was interrupted", e);
        } catch (CancellationException e) {
            throw closedDuringTheCall(e);
        } catch (ExecutionException e) {
            throw rethrowable(e.getCause());
        }
    }

    /** The failure of a call that a {@link #close} dropped from its rank's queue before it started. */
    private static IllegalStateException closedDuringTheCall(CancellationException e) {
        return new IllegalStateException("the vision encoder was closed before the call ran", e);
    }

    /**
     * The rank of every image, by vLLM's {@code get_load_balance_assignment}: the images are taken largest first and
     * each goes to the least loaded rank, so the ranks' patch totals stay close.
     */
    static int[] assignment(int[] sizes, int rankCount) {
        Objects.requireNonNull(sizes, "sizes");
        if (rankCount < 1) {
            throw new IllegalArgumentException("a data-parallel assignment needs at least one rank, got " + rankCount);
        }
        Integer[] order = new Integer[sizes.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> sizes[b] != sizes[a] ? Integer.compare(sizes[b], sizes[a])
                : Integer.compare(a, b));
        long[] loads = new long[rankCount];
        int[] out = new int[sizes.length];
        for (int index : order) {
            int least = 0;
            for (int r = 1; r < rankCount; r++) {
                if (loads[r] < loads[least]) {
                    least = r;
                }
            }
            out[index] = least;
            loads[least] += sizes[index];
        }
        return out;
    }

    /** The pixel count of every image, vLLM's patches-per-image weight for the assignment. */
    private static int[] sizes(List<ClipImage> images) {
        int[] out = new int[images.size()];
        for (int i = 0; i < out.length; i++) {
            ClipImage image = Objects.requireNonNull(images.get(i), "image " + i);
            long pixels = (long) image.width() * image.height();
            out[i] = (int) Math.min(Integer.MAX_VALUE, pixels);
        }
        return out;
    }

    /** A failed rank's cause as a runtime exception (a fatal error is rethrown as it is). */
    private static RuntimeException rethrowable(Throwable cause) {
        if (cause instanceof RuntimeException) {
            return (RuntimeException) cause;
        }
        if (cause instanceof Error) {
            throw (Error) cause;
        }
        return new IllegalStateException("a data-parallel vision encoder rank failed: " + cause, cause);
    }

    @Override
    public float[][] encodeAnyres(ClipImage overview, List<ClipImage> tiles, int gridX, int gridY, int origWidth,
                                  int origHeight) {
        ensureOpen();
        return onRank(0, () -> ranks.get(0).encodeAnyres(overview, tiles, gridX, gridY, origWidth, origHeight));
    }

    /** Every encode runs on its rank's single thread, whichever thread calls it. */
    @Override
    public boolean concurrent() {
        return true;
    }

    /**
     * Attaches the vision pairs to every rank; the returned attachment is rank 0's, the others belong to their
     * rank and close with it. A failed attach leaves the ranks attached before it attached to their encoders, which
     * close with them.
     */
    @Override
    public LoraAttachment attachLoras(List<LoraAttachment.Binding> bindings) {
        Objects.requireNonNull(bindings, "bindings");
        ensureOpen();
        LoraAttachment first = null;
        for (VisionEncoder rank : ranks) {
            LoraAttachment attached = rank.attachLoras(bindings);
            if (first == null) {
                first = attached;
            }
        }
        return first;
    }

    /**
     * Opens a LoRA session on every rank, so the session's pairs apply whichever rank a batch's images encode on; a
     * rank whose session fails to open leaves the sessions opened before it closed again.
     */
    @Override
    public int openLoraSession(List<LoraAttachment.Binding> bindings) {
        Objects.requireNonNull(bindings, "bindings");
        ensureOpen();
        int opened = 0;
        try {
            for (VisionEncoder rank : ranks) {
                rank.openLoraSession(bindings);
                opened++;
            }
        } catch (RuntimeException e) {
            RuntimeException failure = e;
            for (int r = opened - 1; r >= 0; r--) {
                try {
                    ranks.get(r).closeLoraSession(LORA_SESSION_SLOT);
                } catch (RuntimeException closeFailure) {
                    failure.addSuppressed(closeFailure);
                }
            }
            throw failure;
        }
        return LORA_SESSION_SLOT;
    }

    /** Closes the LoRA session on every rank; a rank that refuses keeps the others closing. */
    @Override
    public void closeLoraSession(int slot) {
        RuntimeException failure = null;
        for (int r = ranks.size() - 1; r >= 0; r--) {
            try {
                ranks.get(r).closeLoraSession(slot);
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public ClipVisionModel model() {
        return ranks.get(0).model();
    }

    @Override
    public String device() {
        return device;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        stopRankThreads();
        RuntimeException failure = null;
        for (int i = ranks.size() - 1; i >= 0; i--) {
            try {
                ranks.get(i).close();
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (owner != null) {
            try {
                owner.close();
            } catch (Exception e) {
                RuntimeException wrapped = new IllegalStateException("releasing the encoder's backends failed: "
                        + e.getMessage(), e);
                if (failure == null) {
                    failure = wrapped;
                } else {
                    failure.addSuppressed(wrapped);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "DataParallelVisionEncoder{%s, ranks=%d, on %s}",
                model().hparams().projectorType(), ranks.size(), device);
    }

    /**
     * Stops the rank threads before the ranks close: the calls still queued are dropped (their callers learn the
     * encoder closed), and a call already running is waited for, so no rank is closed under its own encode.
     */
    private void stopRankThreads() {
        for (ExecutorService thread : threads) {
            for (Runnable queued : thread.shutdownNow()) {
                if (queued instanceof Future) {
                    ((Future<?>) queued).cancel(false);
                }
            }
        }
        boolean interrupted = false;
        for (ExecutorService thread : threads) {
            while (true) {
                try {
                    if (thread.awaitTermination(1, TimeUnit.MINUTES)) {
                        break;
                    }
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the vision encoder is closed");
        }
    }

    /** Names one rank's thread after the rank and its device. */
    private static final class RankThreads implements ThreadFactory {
        private final int rank;

        RankThreads(int rank) {
            this.rank = rank;
        }

        @Override
        public Thread newThread(Runnable runnable) {
            Thread thread = new Thread(runnable, "nf-mm-encoder-rank-" + rank);
            thread.setDaemon(true);
            return thread;
        }
    }
}
