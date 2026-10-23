package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Batches of a {@link Dataset} in the order of a {@link Sampler}, loaded on the calling thread or, with
 * {@code numWorkers > 0}, ahead on background threads.
 *
 * <p><b>Tensor lifetime.</b> With {@code pinMemory}, a {@link TensorDataset.Sample} whose tensors live on another
 * backend is copied to the loader's backend, and the batch's sample owns the copies: the consumer closes each sample
 * of a batch once it no longer needs it ({@link TensorDataset.Sample#close()}, a no-op for a sample that owns
 * nothing). Batches loaded ahead but never handed out are released when their pass is closed ({@link Pass#close()})
 * or by {@link #close()}.</p>
 *
 * <p><b>Passes.</b> Each {@link #pass()} (and {@link #iterator()}) is an independent pass over a fresh sample of the
 * dataset, and several passes over one loader may be in progress at once. A consumer that leaves a pass before its
 * last batch closes that pass alone ({@link Pass#close()}); {@link #close()} is the owner's way to end every pass in
 * progress.</p>
 */
public final class DataLoader<T> implements Iterable<List<T>>, AutoCloseable {
    /** Idle time after which a worker thread of an abandoned pass exits. */
    private static final long WORKER_KEEP_ALIVE_SECONDS = 30;

    private final Dataset<T> dataset;
    private final int batchSize;
    private final Sampler sampler;
    private final int numWorkers;
    private final boolean pinMemory;
    private final ComputeBackend backend;
    private final boolean dropLast;
    private final Set<AsyncDataIterator> openIterators = ConcurrentHashMap.newKeySet();

    /**
     * One pass over the loader's batches. {@link #close()} ends this pass only: it stops the loading ahead and
     * releases the batches loaded but not handed out, after which {@link #next()} throws
     * {@link IllegalStateException}; batches already handed out stay with their consumer. Closing is idempotent, and
     * closing a pass that handed out every batch does nothing.
     */
    public interface Pass<T> extends Iterator<List<T>>, AutoCloseable {
        @Override
        void close();
    }

    public DataLoader(Dataset<T> dataset, int batchSize, boolean shuffle) {
        this(dataset, batchSize, shuffle ? new RandomSampler() : new SequentialSampler(), 
             0, false, null, false);
    }

    public DataLoader(Dataset<T> dataset, int batchSize, boolean shuffle, int numWorkers) {
        this(dataset, batchSize, shuffle ? new RandomSampler() : new SequentialSampler(), 
             numWorkers, false, null, false);
    }

    public DataLoader(Dataset<T> dataset, int batchSize, Sampler sampler, 
                      int numWorkers, boolean pinMemory, ComputeBackend backend, boolean dropLast) {
        Objects.requireNonNull(dataset, "Dataset cannot be null");
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive, got: " + batchSize);
        }
        if (numWorkers < 0) {
            throw new IllegalArgumentException("numWorkers must be non-negative, got: " + numWorkers);
        }
        Objects.requireNonNull(sampler, "Sampler cannot be null");
        if (pinMemory && backend == null) {
            throw new IllegalArgumentException("Backend must be provided when pinMemory is true");
        }

        this.dataset = dataset;
        this.batchSize = batchSize;
        this.sampler = sampler;
        this.numWorkers = numWorkers;
        this.pinMemory = pinMemory;
        this.backend = backend;
        this.dropLast = dropLast;
    }

    public int batchSize() {
        return batchSize;
    }

    public int numWorkers() {
        return numWorkers;
    }

    public boolean pinMemory() {
        return pinMemory;
    }

    public int size() {
        return dataset.size();
    }

    /**
     * The number of batches a pass yields: the sampler's index count ({@link Sampler#length(int)}) in batches of
     * {@code batchSize}, the last one partial unless {@code dropLast} ({@code len(DataLoader)} in PyTorch).
     */
    public int numBatches() {
        return batchCount(sampler.length(dataset.size()));
    }

    /** A new {@link #pass()}. */
    @Override
    public Iterator<List<T>> iterator() {
        return pass();
    }

    /**
     * The batches of one pass over the dataset, in sampler order. With {@code numWorkers > 0} up to
     * {@code 2 * numWorkers} batches are loaded ahead on background threads; they are still returned in order, and a
     * batch that fails to load throws its failure from {@link Iterator#next()}. A consumer that leaves the pass before
     * its last batch closes it ({@link Pass#close()}).
     *
     * <p>Each pass takes the sampler's next order: a seeded random sampler yields a new order every pass, and the same
     * sequence of orders again after {@link Sampler#reset()}, as a seeded generator does across PyTorch epochs. A
     * pass with no batch loads nothing ahead, so it starts no worker and leaves nothing in progress.</p>
     */
    public Pass<T> pass() {
        List<Integer> indices = sampler.sample(dataset.size());
        if (numWorkers > 0 && batchCount(indices.size()) > 0) {
            return new AsyncDataIterator(indices);
        }
        return new SyncDataIterator(indices);
    }

    /** The number of passes whose background loading is in progress, which {@link #close()} would end. */
    int passesInProgress() {
        return openIterators.size();
    }

    /** The number of batches a pass over {@code count} sampled indices yields. */
    private int batchCount(int count) {
        return dropLast ? count / batchSize : (count + batchSize - 1) / batchSize;
    }

    /**
     * Batch {@code b} of {@code indices}, each item pinned when the loader pins; on failure the samples pinned so far
     * are released.
     */
    private List<T> loadBatch(List<Integer> indices, int b) {
        int start = b * batchSize;
        int end = Math.min(start + batchSize, indices.size());
        List<T> batch = new ArrayList<>(end - start);
        try {
            for (int i = start; i < end; i++) {
                T item = dataset.get(indices.get(i));
                if (pinMemory && item instanceof TensorDataset.Sample) {
                    item = pinSample((TensorDataset.Sample) item);
                }
                batch.add(item);
            }
        } catch (RuntimeException e) {
            try {
                releaseBatch(batch);
            } catch (RuntimeException releaseFailure) {
                e.addSuppressed(releaseFailure);
            }
            throw e;
        }
        return batch;
    }

    /**
     * Closes the samples of a batch nobody will consume; only the copies the loader made are released
     * ({@link TensorDataset.Sample#close()}). Every sample is tried; the first failure is thrown with the later ones
     * suppressed.
     */
    private static void releaseBatch(List<?> batch) {
        RuntimeException failure = null;
        for (Object item : batch) {
            if (!(item instanceof TensorDataset.Sample)) {
                continue;
            }
            try {
                ((TensorDataset.Sample) item).close();
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

    private final class SyncDataIterator implements Pass<T> {
        private final List<Integer> indices;
        private final int total;
        private int next;
        private volatile boolean closed;

        SyncDataIterator(List<Integer> indices) {
            this.indices = indices;
            this.total = batchCount(indices.size());
        }

        @Override
        public boolean hasNext() {
            return next < total;
        }

        @Override
        public List<T> next() {
            if (!hasNext()) {
                throw new NoSuchElementException("No more batches available");
            }
            if (closed) {
                throw new IllegalStateException("The DataLoader pass was closed");
            }
            return loadBatch(indices, next++);
        }

        /** Nothing is loaded ahead, so closing only ends the pass. */
        @Override
        public void close() {
            closed = true;
        }
    }

    /**
     * Loads batches ahead on {@code numWorkers} daemon threads, which exit once idle. Loaded batches wait in
     * {@code ready} by index until {@link #next()} hands them out in order. {@link #close()} (by the consumer, from
     * {@link DataLoader#close()}, or after a failure) stops the loading and releases the batches loaded but not handed
     * out, including those that finish loading afterwards. Batches are submitted under {@code lock} and only while
     * the pass is open, so a close from another thread never meets a submission to the stopped executor.
     */
    private final class AsyncDataIterator implements Pass<T> {
        private final List<Integer> indices;
        private final int total;
        private final int lookahead;
        private final ExecutorService executor;
        private final Object lock = new Object();
        private final Map<Integer, List<T>> ready = new HashMap<>();
        private final Map<Integer, Throwable> failures = new HashMap<>();
        private boolean closed;
        private int submitted;
        private int next;

        AsyncDataIterator(List<Integer> indices) {
            this.indices = indices;
            this.total = batchCount(indices.size());
            this.lookahead = 2 * numWorkers;
            ThreadPoolExecutor pool = new ThreadPoolExecutor(numWorkers, numWorkers, WORKER_KEEP_ALIVE_SECONDS,
                TimeUnit.SECONDS, new LinkedBlockingQueue<>(), DataLoader::daemonThread);
            pool.allowCoreThreadTimeOut(true);
            this.executor = pool;
            openIterators.add(this);
            submitAhead();
        }

        @Override
        public boolean hasNext() {
            return next < total;
        }

        @Override
        public List<T> next() {
            if (!hasNext()) {
                throw new NoSuchElementException("No more batches available");
            }
            int index = next;
            List<T> batch;
            Throwable failure;
            synchronized (lock) {
                try {
                    while (!closed && !ready.containsKey(index) && !failures.containsKey(index)) {
                        lock.wait();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    IllegalStateException interrupted =
                        new IllegalStateException("Interrupted while waiting for batch " + index, e);
                    closeAfter(interrupted);
                    throw interrupted;
                }
                if (closed && !ready.containsKey(index)) {
                    throw new IllegalStateException("The DataLoader pass was closed");
                }
                batch = ready.remove(index);
                failure = failures.remove(index);
            }
            if (failure != null) {
                RuntimeException thrown = failure instanceof RuntimeException
                    ? (RuntimeException) failure
                    : new IllegalStateException("Loading batch " + index + " failed", failure);
                closeAfter(thrown);
                throw thrown;
            }
            next++;
            submitAhead();
            if (next == total) {
                openIterators.remove(this);
            }
            return batch;
        }

        /**
         * Submits batches until {@code lookahead} are loading or waiting ahead of the next one to hand out; a closed
         * pass submits nothing. {@link #close()} marks the pass closed under {@code lock} before it stops the executor,
         * so a submission made here while the pass is open is never rejected.
         */
        private void submitAhead() {
            synchronized (lock) {
                if (closed) {
                    return;
                }
                while (submitted < total && submitted - next < lookahead) {
                    int index = submitted++;
                    executor.execute(() -> load(index));
                }
                if (submitted == total) {
                    executor.shutdown();
                }
            }
        }

        /** Loads batch {@code index} on a worker thread and hands it to {@link #next()}, or releases it when closed. */
        private void load(int index) {
            List<T> batch;
            try {
                batch = loadBatch(indices, index);
            } catch (RuntimeException | Error e) {
                synchronized (lock) {
                    failures.put(index, e);
                    lock.notifyAll();
                }
                return;
            }
            boolean handedOver;
            synchronized (lock) {
                handedOver = !closed;
                if (handedOver) {
                    ready.put(index, batch);
                    lock.notifyAll();
                }
            }
            if (!handedOver) {
                releaseBatch(batch);
            }
        }

        /** {@link #close()}, with its failures added to {@code primary}. */
        private void closeAfter(RuntimeException primary) {
            try {
                close();
            } catch (RuntimeException e) {
                primary.addSuppressed(e);
            }
        }

        /** Stops loading and releases the batches loaded but not handed out; idempotent. */
        @Override
        public void close() {
            List<List<T>> unconsumed;
            synchronized (lock) {
                if (closed) {
                    return;
                }
                closed = true;
                unconsumed = new ArrayList<>(ready.values());
                ready.clear();
                failures.clear();
                lock.notifyAll();
            }
            openIterators.remove(this);
            executor.shutdownNow();
            RuntimeException failure = null;
            for (List<T> batch : unconsumed) {
                try {
                    releaseBatch(batch);
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
    }

    private static Thread daemonThread(Runnable task) {
        Thread thread = new Thread(task, "DataLoader-worker");
        thread.setDaemon(true);
        return thread;
    }

    /**
     * {@code sample} with its tensors on the loader's backend: a tensor already there is kept, any other is copied,
     * and the returned sample owns the copies ({@link TensorDataset.Sample#close()}).
     */
    @SuppressWarnings("unchecked")
    private T pinSample(TensorDataset.Sample sample) {
        if (backend == null) {
            return (T) sample;
        }

        Tensor pinnedInput = pinTensor(sample.input());
        Tensor pinnedTarget;
        try {
            pinnedTarget = pinTensor(sample.target());
        } catch (RuntimeException e) {
            if (pinnedInput != sample.input()) {
                pinnedInput.close();
            }
            throw e;
        }
        return (T) new TensorDataset.Sample(pinnedInput, pinnedTarget, pinnedInput != sample.input(),
            pinnedTarget != sample.target());
    }

    /** {@code tensor} when it lives on the loader's backend, else a contiguous copy there with the same dtype. */
    private Tensor pinTensor(Tensor tensor) {
        if (tensor.backend() == backend) {
            return tensor;
        }
        TensorStorage storage = TensorStorage.fromHost(backend, tensor.toByteBuffer(), tensor.dtype());
        return Tensor.fromStorage(backend, tensor.shape(), tensor.dtype(), storage);
    }

    /**
     * Stops the background loading of every pass still in progress and releases the batches it loaded but did not
     * hand out. Batches already handed out stay with their consumer, which closes their samples
     * ({@link TensorDataset.Sample#close()}). The loader can be iterated again afterwards. Every pass is tried; the
     * first failure is thrown with the later ones suppressed.
     */
    @Override
    public void close() {
        RuntimeException failure = null;
        for (AsyncDataIterator iterator : new ArrayList<>(openIterators)) {
            try {
                iterator.close();
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
    public String toString() {
        return "DataLoader{batchSize=" + batchSize + 
               ", numWorkers=" + numWorkers + 
               ", pinMemory=" + pinMemory + 
               ", dropLast=" + dropLast + "}";
    }

    public static <T> Builder<T> builder(Dataset<T> dataset) {
        return new Builder<>(dataset);
    }

    public static final class Builder<T> {
        private final Dataset<T> dataset;
        private int batchSize = 32;
        private Sampler sampler;
        private boolean shuffle = false;
        private int numWorkers = 0;
        private boolean pinMemory = false;
        private ComputeBackend backend;
        private boolean dropLast = false;

        public Builder(Dataset<T> dataset) {
            this.dataset = Objects.requireNonNull(dataset, "Dataset cannot be null");
        }

        public Builder<T> batchSize(int batchSize) {
            if (batchSize <= 0) {
                throw new IllegalArgumentException("Batch size must be positive");
            }
            this.batchSize = batchSize;
            return this;
        }

        public Builder<T> shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            this.sampler = null;
            return this;
        }

        public Builder<T> sampler(Sampler sampler) {
            this.sampler = Objects.requireNonNull(sampler, "Sampler cannot be null");
            this.shuffle = false;
            return this;
        }

        public Builder<T> numWorkers(int numWorkers) {
            if (numWorkers < 0) {
                throw new IllegalArgumentException("numWorkers must be non-negative");
            }
            this.numWorkers = numWorkers;
            return this;
        }

        public Builder<T> pinMemory(boolean pinMemory) {
            this.pinMemory = pinMemory;
            return this;
        }

        public Builder<T> backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }

        public Builder<T> dropLast(boolean dropLast) {
            this.dropLast = dropLast;
            return this;
        }

        public DataLoader<T> build() {
            if (sampler == null) {
                sampler = shuffle ? new RandomSampler() : new SequentialSampler();
            }
            return new DataLoader<>(dataset, batchSize, sampler, numWorkers, pinMemory, backend, dropLast);
        }
    }
}
