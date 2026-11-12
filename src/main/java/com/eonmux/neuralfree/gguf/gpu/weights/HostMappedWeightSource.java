package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet.Entry;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * {@link LayerWeightSource} over host-mapped weights: the tensors of a {@link HostMappedPlan} live in
 * page-locked host memory the device reads in place, and every other tensor comes from the delegate
 * (the streamed source, or the resident weights). Nothing is prefetched or awaited for the mapped
 * tensors: they are readable for the life of the source, so {@link #prefetch(int)} and
 * {@link #await(int)} only advance the delegate.
 *
 * <p>Buffers are filled once at open from the host tier, in the upload type of the layer planning
 * (F32 when the host dequantizes), with the input transforms, permutations and output scales the
 * resident path attaches. The compute queue is finished before the buffers are read, and again
 * before {@link #close()} frees them. Owns its buffers and the host tier; the delegate is owned by
 * the caller. Not thread-safe.</p>
 */
public final class HostMappedWeightSource implements LayerWeightSource {

    private static final Logger LOGGER = LogManager.getLogger(HostMappedWeightSource.class);

    private final ComputeQueue computeQueue;
    private final WeightDataProvider data;
    private final LayerWeightSource delegate;
    private final Map<Integer, Map<String, GpuWeight>> weightsByLayer;
    private final List<GpuBuffer> buffers;
    private final String arch;
    private final long hostBytes;
    private boolean closed;

    private HostMappedWeightSource(ComputeQueue computeQueue, WeightDataProvider data, LayerWeightSource delegate,
                                   Map<Integer, Map<String, GpuWeight>> weightsByLayer, List<GpuBuffer> buffers,
                                   String arch, long hostBytes) {
        this.computeQueue = computeQueue;
        this.data = data;
        this.delegate = delegate;
        this.weightsByLayer = Collections.unmodifiableMap(weightsByLayer);
        this.buffers = Collections.unmodifiableList(buffers);
        this.arch = arch;
        this.hostBytes = hostBytes;
    }

    /**
     * Allocates the host-mapped buffers of {@code plan}, fills them and finishes the compute queue so
     * the first dispatch reads them.
     *
     * @param backend  backend allocating the host-mapped buffers
     * @param queue    compute queue every dispatch reading the weights runs on
     * @param data     host tier covering the offloaded tensors; owned by the source once it opens
     * @param arch     architecture name for missing-tensor messages
     * @param plan     the offloaded tensors; non-empty
     * @param delegate source of every tensor the plan does not hold
     * @throws UnsupportedOperationException when the backend has no host-mapped buffers
     * @throws IllegalArgumentException      when the plan is empty
     * @throws UncheckedIOException          when an offloaded tensor cannot be read
     * @throws RuntimeException              from the backend when a buffer cannot be allocated
     */
    public static HostMappedWeightSource open(ComputeBackend backend, ComputeQueue queue, WeightDataProvider data,
                                              String arch, HostMappedPlan plan, LayerWeightSource delegate) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(delegate, "delegate");
        if (plan.isEmpty()) {
            throw new IllegalArgumentException("host-mapped weight source needs at least one offloaded tensor");
        }
        if (!backend.supportsHostMappedBuffers()) {
            throw new UnsupportedOperationException(backend.backendType() + " backend cannot allocate host-mapped"
                    + " weight memory");
        }
        List<GpuBuffer> owned = new ArrayList<>();
        try {
            Map<Integer, Map<String, GpuWeight>> byLayer = new HashMap<>();
            Loader loader = new Loader(backend, data, owned);
            for (LayerWeightSet set : plan.sets()) {
                Map<String, GpuWeight> weights = new LinkedHashMap<>();
                for (Entry e : set.entries()) {
                    weights.put(e.name(), loader.upload(e));
                }
                byLayer.put(set.weightLayer(), Collections.unmodifiableMap(weights));
            }
            queue.finish();
            HostMappedWeightSource source = new HostMappedWeightSource(queue, data, delegate,
                    Collections.unmodifiableMap(byLayer), owned, arch, plan.bytes());
            LOGGER.info("{}: {} host-mapped, the device reads them over the interconnect",
                    arch, plan.describe());
            return source;
        } catch (RuntimeException e) {
            Closeables.closeAll(owned, e);
            data.close();
            throw e;
        }
    }

    @Override
    public void prefetch(int il) {
        ensureOpen();
        delegate.prefetch(il);
    }

    @Override
    public void await(int il) {
        ensureOpen();
        delegate.await(il);
    }

    @Override
    public GpuWeights weightsFor(int il) {
        ensureOpen();
        if (il < 0) {
            throw new IllegalArgumentException("weight layer index " + il + " is negative");
        }
        Map<String, GpuWeight> own = weightsByLayer.get(il);
        GpuWeights rest = delegate.weightsFor(il);
        return own == null ? rest : new MergedWeights(arch, own, rest);
    }

    /** Host bytes of the offloaded tensors. */
    public long hostBytes() {
        return hostBytes;
    }

    /** Number of offloaded tensors. */
    public int tensorCount() {
        int n = 0;
        for (Map<String, GpuWeight> weights : weightsByLayer.values()) {
            n += weights.size();
        }
        return n;
    }

    /** Number of weight layers with host-mapped tensors. */
    public int layerCount() {
        return weightsByLayer.size();
    }

    /** Waits for the compute queue and frees the buffers and the host tier. Idempotent; the delegate is not closed. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        try {
            computeQueue.finish();
        } catch (RuntimeException e) {
            failure = e;
        }
        RuntimeException closeFailure = Closeables.closeAll(buffers, null);
        data.close();
        if (failure == null) {
            failure = closeFailure;
        }
        if (failure != null) {
            throw failure;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("host-mapped weight source is closed");
        }
    }

    @Override
    public String toString() {
        return "HostMappedWeightSource{" + layerCount() + " layers, " + tensorCount() + " tensors, " + hostBytes
                + " B over " + delegate + '}';
    }

    /** Allocates one entry's buffers and copies its rows into them. */
    private static final class Loader {
        private final ComputeBackend backend;
        private final WeightDataProvider data;
        private final List<GpuBuffer> owned;
        private final ShardFiller filler;

        Loader(ComputeBackend backend, WeightDataProvider data, List<GpuBuffer> owned) {
            this.backend = backend;
            this.data = data;
            this.owned = owned;
            this.filler = new ShardFiller(data);
        }

        GpuWeight upload(Entry e) {
            List<GpuBuffer> shards = new ArrayList<>(e.shardCount());
            for (int k = 0; k < e.shardCount(); k++) {
                HostMappedBuffer buffer = backend.allocateHostMappedBuffer(e.shardBytes(k));
                owned.add(buffer);
                shards.add(buffer);
                filler.fill(e, k, buffer, 0L, "host-mapped tensor");
                buffer.flush();
            }
            GpuWeight w = GpuWeight.sharded(e.name(), e.uploadType(), e.ne(), shards, e.shardRowCounts());
            if (e.inputTransform() != null) {
                w = w.withInputTransform(e.inputTransform(), perm(e), preScale(e));
            } else if (e.hadamard() != null) {
                w = w.withHadamard(e.hadamard());
            }
            if (e.scale() != null) {
                w = w.withScale(LayerSlots.readScale(data, e));
            }
            return w;
        }

        /** The host-mapped int buffer of the entry's {@code <base>.input_perm} tensor, or null. */
        private GpuBuffer perm(Entry e) {
            if (!e.inputTransform().hasPerm()) {
                return null;
            }
            if (e.inputPerm() == null) {
                throw new IllegalStateException("tensor '" + e.name() + "': its input transform permutes the input"
                        + " but the model has no " + GpuWeight.permNameOf(e.name()) + " tensor");
            }
            long ints = e.cols();
            if (ints <= 0 || ints > Integer.MAX_VALUE / Integer.BYTES) {
                throw new IllegalStateException("tensor '" + e.name() + "': its input permutation covers " + ints
                        + " columns, outside the int range");
            }
            int permBytes = (int) (ints * Integer.BYTES);
            HostMappedBuffer buffer = backend.allocateHostMappedBuffer(permBytes);
            owned.add(buffer);
            try {
                data.readRange(e.inputPerm().name(), 0, permBytes, buffer.hostView(0, permBytes));
            } catch (IOException ex) {
                throw new UncheckedIOException("host-mapped input permutation '" + e.inputPerm().name()
                        + "' failed: " + ex.getMessage(), ex);
            }
            buffer.flush();
            return buffer;
        }

        /** The host-mapped float buffer of the entry's {@code <base>.input_pre_scale} tensor, or null. */
        private GpuBuffer preScale(Entry e) {
            if (e.inputTransform() == null || !e.inputTransform().hasPreScale()) {
                return null;
            }
            if (e.inputPreScale() == null) {
                throw new IllegalStateException("tensor '" + e.name() + "': its input transform scales the input"
                        + " but the model has no " + GpuWeight.preScaleNameOf(e.name()) + " tensor");
            }
            long floats = e.inputPreScale().numElements();
            if (floats <= 0 || floats > Integer.MAX_VALUE / Float.BYTES) {
                throw new IllegalStateException("tensor '" + e.name() + "': its activation pre-scale holds "
                        + floats + " values, outside the float array range");
            }
            int bytes = (int) (floats * Float.BYTES);
            HostMappedBuffer buffer = backend.allocateHostMappedBuffer(bytes);
            owned.add(buffer);
            try {
                data.readRange(e.inputPreScale().name(), 0, bytes, buffer.hostView(0, bytes));
            } catch (IOException ex) {
                throw new UncheckedIOException("host-mapped activation pre-scale '" + e.inputPreScale().name()
                        + "' failed: " + ex.getMessage(), ex);
            }
            buffer.flush();
            return buffer;
        }
    }

    /** One layer's host-mapped tensors over the delegate's own. */
    private static final class MergedWeights implements GpuWeights {
        private final String arch;
        private final Map<String, GpuWeight> own;
        private final GpuWeights rest;

        MergedWeights(String arch, Map<String, GpuWeight> own, GpuWeights rest) {
            this.arch = arch;
            this.own = own;
            this.rest = rest;
        }

        @Override
        public GpuWeight find(String name) {
            GpuWeight w = own.get(name);
            return w != null ? w : rest.find(name);
        }

        @Override
        public String archName() {
            return arch;
        }

        @Override
        public String toString() {
            return "MergedWeights{" + own.size() + " host-mapped tensors over " + rest + '}';
        }
    }
}
