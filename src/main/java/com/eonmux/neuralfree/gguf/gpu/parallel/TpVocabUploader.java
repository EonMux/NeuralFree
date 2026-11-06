package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Uploads the per-rank vocabulary shards of a tensor-parallel split ({@link TpVocabPlan}): rank {@code rank}'s row
 * range of the token embedding table and of the LM-head weight, and the matching range of the head bias when the head
 * has one. A quantized table is sliced on whole stored rows; the head bias is F32 and sliced as floats. Every
 * {@link #uploadRank} allocates on the rank's backend, enqueues on its queue and finishes it before returning; the
 * uploader owns every buffer it allocates and {@link #close()} releases them.
 */
public final class TpVocabUploader implements AutoCloseable {

    private final TpVocabPlan plan;
    private final GpuWeightUploader weights;
    private final WeightStore store;
    private final List<GpuBuffer> owned = new ArrayList<>();

    public TpVocabUploader(TpVocabPlan plan, GpuWeightUploader weights, WeightStore store) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.store = Objects.requireNonNull(store, "store");
    }

    /**
     * Uploads rank {@code rank}'s vocabulary shards.
     *
     * @return the rank's shard weights, keyed by their full tensor names
     * @throws IllegalArgumentException when the model has no data for a shard
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when a shard does not fit the rank device
     */
    public Map<String, GpuWeight> uploadRank(int rank, ComputeBackend backend, ComputeQueue queue) {
        Map<String, GpuWeight> out = new LinkedHashMap<>();
        out.put(plan.tokenTensor(), rows(plan.tokenTensor(), plan.tokenStart(rank), plan.tokenCount(rank),
                backend, queue));
        if (!plan.tied()) {
            out.put(plan.headTensor(), rows(plan.headTensor(), plan.headStart(rank), plan.headCount(rank),
                    backend, queue));
        }
        if (plan.headBias() != null) {
            out.put(plan.headBias(), elements(plan.headBias(), plan.headStart(rank), plan.headCount(rank),
                    backend, queue));
        }
        queue.finish();
        weights.syncCopier(queue);
        return out;
    }

    /** A contiguous row range of a table, as one buffer or a row-sharded tensor. */
    private GpuWeight rows(String name, long firstRow, long rowCount, ComputeBackend backend, ComputeQueue queue) {
        GpuWeightUploader.TensorPlan p = weights.plan(name);
        if (p == null) {
            throw new IllegalArgumentException("the model has no tensor " + name);
        }
        long rowBytes = p.deviceRowBytes();
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, rowBytes));
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        for (long done = 0; done < rowCount; done += perBuffer) {
            long n = Math.min(perBuffer, rowCount - done);
            GpuBuffer dst = allocate(n * rowBytes, backend);
            weights.copyRows(name, firstRow + done, n, dst, 0L, queue);
            buffers.add(dst);
            counts.add(n);
        }
        long[] rows = new long[counts.size()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = counts.get(i);
        }
        GpuWeight shard = buffers.size() == 1
                ? GpuWeight.of(name, p.deviceType(), new long[]{p.ne()[0], rowCount}, buffers.get(0))
                : GpuWeight.sharded(name, p.deviceType(), new long[]{p.ne()[0], rowCount}, buffers, rows);
        return withCompanions(shard);
    }

    /** A contiguous element range of an F32 vector. */
    private GpuWeight elements(String name, long first, long count, ComputeBackend backend, ComputeQueue queue) {
        GpuWeightUploader.TensorPlan p = weights.plan(name);
        if (p == null) {
            throw new IllegalArgumentException("the model has no tensor " + name);
        }
        if (p.needsDecode() || p.deviceType() != GGMLType.F32) {
            throw new UnsupportedOperationException("tensor " + name + " is " + p.deviceType()
                    + "; a split head bias needs F32");
        }
        GpuBuffer dst = allocate(count * Float.BYTES, backend);
        ByteBuffer staged = ByteBuffer.allocateDirect((int) (count * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        try {
            store.readTensorDirect(p.info().offset() + first * Float.BYTES, (int) (count * Float.BYTES), staged);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + name, e);
        }
        queue.enqueueWriteBuffer(dst, 0L, count * Float.BYTES, staged);
        return GpuWeight.of(name, GGMLType.F32, new long[]{count}, dst);
    }

    /** Folds the table's single-element {@code <base>.scale} into the shard, and its Hadamard if any. */
    private GpuWeight withCompanions(GpuWeight w) {
        GpuWeight out = w;
        String name = w.name();
        if (name.endsWith(".weight")) {
            String scale = name.substring(0, name.length() - ".weight".length()) + ".scale";
            GpuWeightUploader.TensorPlan p = weights.plan(scale);
            if (p != null && p.info().numElements() == 1) {
                out = out.withScale(weights.readScalar(scale));
            }
        }
        if (weights.transforms() != null && !weights.transforms().isEmpty()) {
            HadamardSpec hadamard = weights.transforms().hadamard(name);
            if (hadamard != null) {
                out = out.withHadamard(hadamard);
            }
        }
        return out;
    }

    private GpuBuffer allocate(long bytes, ComputeBackend backend) {
        GpuBuffer buffer = backend.allocateBuffer(bytes, true);
        owned.add(buffer);
        return buffer;
    }

    @Override
    public void close() {
        List<GpuBuffer> buffers = new ArrayList<>(owned);
        owned.clear();
        for (GpuBuffer b : buffers) {
            b.close();
        }
    }
}
