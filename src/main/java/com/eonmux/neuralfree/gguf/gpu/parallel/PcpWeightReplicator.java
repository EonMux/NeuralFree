package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * Replicates the whole trunk tensors of a prefill context-parallel split to every peer rank. A context split
 * uploads only the attention tensors per rank (every rank keeps every head, so each is whole), but under the
 * DualChunkSwap prefill every rank runs every layer's norms, projections and FFN over its own packed rows, so
 * each peer needs those weights too. Every trunk tensor ({@code blk.*}) the attention plan leaves whole and that
 * the model device holds resident is copied device to device into the peer's shard map; tensors already in the
 * shard map (the attention tensors) stay as uploaded.
 *
 * <p>A tensor whose upload attached a full input transform chain refuses: the chain's permutation and pre-scale
 * buffers live on the model device, and a direct PRISM Hadamard or an output scale carries over as configuration
 * ({@link GpuWeight#withHadamard} / {@link GpuWeight#withScale}). Every copied buffer belongs to this replicator;
 * {@link #close()} releases them.</p>
 */
final class PcpWeightReplicator implements AutoCloseable {

    private final TpAttentionPlan plan;
    private final GpuWeightUploader weights;
    private final ComputeQueue mainQueue;
    private final List<GpuBuffer> owned = new ArrayList<>();

    PcpWeightReplicator(TpAttentionPlan plan, GpuWeightUploader weights, ComputeQueue mainQueue) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.mainQueue = Objects.requireNonNull(mainQueue, "mainQueue");
    }

    /** The trunk tensors a prefill context-parallel split replicates: {@code blk.*} whole and resident. */
    static Predicate<GGUFTensorInfo> replicatedTensors(TpAttentionPlan plan) {
        Objects.requireNonNull(plan, "plan");
        return t -> t.name().startsWith("blk.") && !plan.splits(t.name());
    }

    /** Device bytes one peer's replicas take, planned from the uploader's tensor plans. */
    long replicatedBytes() {
        return weights.plannedBytes(replicatedTensors(plan));
    }

    /**
     * Copies every trunk tensor missing from {@code rank}'s shard map into it.
     *
     * @throws UnsupportedOperationException when a replicated tensor carries an input transform chain or an
     *                                       inverse lookup transform the copy cannot carry
     */
    void replicate(int rank, ComputeBackend backend, ComputeQueue queue, Map<String, GpuWeight> shard) {
        for (GpuWeightUploader.TensorPlan p : weights.plans(replicatedTensors(plan))) {
            if (shard.containsKey(p.name())) {
                continue;
            }
            GpuWeight source = weights.find(p.name());
            if (source == null) {
                throw new IllegalStateException("tensor " + p.name() + " is not resident on the model device;"
                        + " a prefill context-parallel split replicates the whole trunk tensors to every rank");
            }
            shard.put(p.name(), copy(source, rank, backend, queue));
        }
    }

    private GpuWeight copy(GpuWeight source, int rank, ComputeBackend backend, ComputeQueue queue) {
        if (source.hasInputTransform()) {
            throw new UnsupportedOperationException("tensor " + source.name() + " runs an input transform chain"
                    + " whose buffers stay on the model device; the prefill context-parallel split cannot"
                    + " replicate it");
        }
        if (source.inverseHadamard() != null) {
            throw new UnsupportedOperationException("tensor " + source.name() + " carries an inverse lookup"
                    + " transform; the prefill context-parallel split cannot replicate it");
        }
        List<GpuBuffer> buffers = new ArrayList<>();
        long[] counts = new long[source.shardCount()];
        for (int i = 0; i < source.shardCount(); i++) {
            long bytes = source.shardRows(i) * source.rowBytes();
            GpuBuffer dst = backend.allocateBuffer(bytes, true);
            owned.add(dst);
            RankExchange.replayBytes(mainQueue, source.shard(i), 0L, queue, dst, 0L, bytes);
            buffers.add(dst);
            counts[i] = source.shardRows(i);
        }
        GpuWeight out = GpuWeight.sharded(source.name(), source.type(), source.ne(), buffers, counts)
                .withScale(source.scale());
        if (source.hadamard() != null) {
            out = out.withHadamard(source.hadamard());
        }
        return out;
    }

    @Override
    public void close() {
        List<GpuBuffer> buffers = new ArrayList<>(owned);
        owned.clear();
        RuntimeException failure = null;
        for (GpuBuffer b : buffers) {
            try {
                b.close();
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
