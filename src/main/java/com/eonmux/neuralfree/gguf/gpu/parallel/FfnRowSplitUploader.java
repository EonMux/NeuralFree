package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;

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
 * Uploads the per-rank shards of a feed-forward row split ({@link FfnRowSplitPlan}) from the model's tensor data.
 * Per rank and per split layer it uploads the rank's row range of {@code ffn_gate} / {@code ffn_up} (the two halves
 * of a fused up projection as two buffers of one sharded weight), the matching column slice of {@code ffn_down}, the
 * matching slice of every hidden-width vector (the up and gate biases, the mpt activation scales) and the whole
 * output-width vectors (the down bias, whose scalar scale attaches to the shard weight). A column slice keeps whole
 * device blocks of the stored type, or dequantizes the stored rows and slices the floats when the device type is
 * F32. Hadamard rotations act on the stream input, so the up and gate shards keep theirs.
 *
 * <p>Every {@link #uploadRank} allocates on the rank's backend, enqueues on its queue and finishes the queue before
 * returning. The uploader owns every buffer it allocates; {@link #close()} releases them. Not thread-safe.</p>
 */
public final class FfnRowSplitUploader implements AutoCloseable {

    private final FfnRowSplitPlan plan;
    private final GpuWeightUploader weights;
    private final WeightStore store;
    private final List<GpuBuffer> owned = new ArrayList<>();

    /**
     * Creates the uploader of {@code plan}'s shards, reading plans and transforms from {@code weights} and the
     * tensor data from {@code store}.
     */
    public FfnRowSplitUploader(FfnRowSplitPlan plan, GpuWeightUploader weights, WeightStore store) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.store = Objects.requireNonNull(store, "store");
    }

    /**
     * Uploads rank {@code rank}'s shard of every split tensor, allocating on {@code backend} and enqueuing on
     * {@code queue}; finishes the queue before returning.
     *
     * @return the rank's shard weights, keyed by their full tensor names ({@code blk.<wl>.<suffix>})
     * @throws IllegalArgumentException when the model has no data for a split tensor
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when a shard does not fit the rank device
     * @throws IllegalStateException    when the device queue rejects an upload
     */
    public Map<String, GpuWeight> uploadRank(int rank, ComputeBackend backend, ComputeQueue queue) {
        Map<String, GpuWeight> out = new LinkedHashMap<>();
        for (FfnRowSplitPlan.Layer layer : plan.layers()) {
            uploadLayer(rank, layer, backend, queue, out);
        }
        queue.finish();
        weights.syncCopier(queue);
        return out;
    }

    private void uploadLayer(int rank, FfnRowSplitPlan.Layer layer, ComputeBackend backend, ComputeQueue queue,
                             Map<String, GpuWeight> out) {
        int h0 = layer.rowStart(rank);
        int shard = layer.shard(rank);
        GpuWeightUploader.TensorPlan up = plan(layer, layer.up());
        out.put(up.name(), upProjection(layer, up, h0, shard, backend, queue));
        if (layer.gate() != null) {
            GpuWeightUploader.TensorPlan gate = plan(layer, layer.gate());
            out.put(gate.name(), rows(gate, gate.name(), h0, shard, backend, queue));
        }
        GpuWeightUploader.TensorPlan down = plan(layer, layer.down());
        out.put(down.name(), downProjection(layer, down, h0, shard, backend, queue));
        if (layer.upBias() != null) {
            GpuWeightUploader.TensorPlan p = plan(layer, layer.upBias());
            out.put(p.name(), hiddenVector(layer, p, h0, shard, backend, queue));
        }
        if (layer.gateBias() != null) {
            GpuWeightUploader.TensorPlan p = plan(layer, layer.gateBias());
            out.put(p.name(), hiddenVector(layer, p, h0, shard, backend, queue));
        }
        if (layer.actScales() != null) {
            GpuWeightUploader.TensorPlan p = plan(layer, layer.actScales());
            out.put(p.name(), hiddenVector(layer, p, h0, shard, backend, queue));
        }
        if (layer.downBias() != null && rank == 0) {
            // the all-reduce sums the partials, so the output-width bias is added once, as vLLM's
            // RowParallelLinear adds its bias on rank 0 only
            GpuWeightUploader.TensorPlan p = plan(layer, layer.downBias());
            out.put(p.name(), outputVector(p, backend, queue));
        }
    }

    /** The rank's rows of the up projection; a fused projection as two halves of one sharded weight. */
    private GpuWeight upProjection(FfnRowSplitPlan.Layer layer, GpuWeightUploader.TensorPlan p, int h0, int shard,
                                   ComputeBackend backend, ComputeQueue queue) {
        if (!layer.fusedUp()) {
            return rows(p, p.name(), h0, shard, backend, queue);
        }
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        appendRowShards(p, h0, shard, buffers, counts, backend, queue);
        appendRowShards(p, (long) layer.nFf() + h0, shard, buffers, counts, backend, queue);
        GpuWeight w = sharded(p.name(), p.deviceType(), new long[]{p.ne()[0], 2L * shard}, buffers, counts);
        return withScaleOf(withHadamard(w, p.name()));
    }

    /** The rank's column slice of the down projection ({@code [shard, nOut]}), split into whole buffer chunks. */
    private GpuWeight downProjection(FfnRowSplitPlan.Layer layer, GpuWeightUploader.TensorPlan p, int h0,
                                     int shard, ComputeBackend backend, ComputeQueue queue) {
        long rowBytes = p.deviceType().getBytesForElements(shard);
        long perBuffer = rowsPerBuffer(rowBytes);
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        for (long done = 0; done < layer.nOut(); done += perBuffer) {
            long rows = Math.min(perBuffer, layer.nOut() - done);
            GpuBuffer dst = allocate(rows * rowBytes, backend);
            copyColumnRows(p, h0, shard, done, rows, dst, 0L, queue);
            buffers.add(dst);
            counts.add(rows);
        }
        return withScaleOf(sharded(p.name(), p.deviceType(), new long[]{shard, layer.nOut()}, buffers,
                counts));
    }

    /** The rank's slice of a vector over the hidden width (a bias or the activation scales). */
    private GpuWeight hiddenVector(FfnRowSplitPlan.Layer layer, GpuWeightUploader.TensorPlan p, int h0,
                                   int shard, ComputeBackend backend, ComputeQueue queue) {
        if (layer.fusedUp() && p.name().equals(layer.tensor(layer.upBias()))) {
            // one contiguous [gate shard | up shard] vector, the layout the fused projection's bias add reads
            long shardBytes = (long) shard * Float.BYTES;
            GpuBuffer dst = allocate(2 * shardBytes, backend);
            copyColumns(p, h0, shard, dst, 0L, queue);
            copyColumns(p, layer.nFf() + h0, shard, dst, shardBytes, queue);
            return GpuWeight.of(p.name(), GGMLType.F32, new long[]{2L * shard}, dst);
        }
        GpuBuffer dst = allocate((long) shard * Float.BYTES, backend);
        copyColumns(p, h0, shard, dst, queue);
        return GpuWeight.of(p.name(), GGMLType.F32, new long[]{shard}, dst);
    }

    /** The whole output-width vector (the down bias), uploaded to rank 0 only. */
    private GpuWeight outputVector(GpuWeightUploader.TensorPlan p, ComputeBackend backend, ComputeQueue queue) {
        GpuBuffer dst = allocate(Math.max(p.deviceBytes(), Float.BYTES), backend);
        copyRows(p, 0, p.rows(), dst, queue);
        return GpuWeight.of(p.name(), GGMLType.F32, new long[]{p.cols()}, dst);
    }

    /** The rank's row range of a projection: a contiguous slice of the stored rows, split into buffer chunks. */
    private GpuWeight rows(GpuWeightUploader.TensorPlan p, String name, long firstRow, long nRows,
                           ComputeBackend backend, ComputeQueue queue) {
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        appendRowShards(p, firstRow, nRows, buffers, counts, backend, queue);
        GpuWeight w = sharded(name, p.deviceType(), new long[]{p.ne()[0], nRows}, buffers, counts);
        return withScaleOf(withHadamard(w, name));
    }

    /** Appends buffers holding {@code nRows} rows from {@code firstRow} on, at most one buffer's worth each. */
    private void appendRowShards(GpuWeightUploader.TensorPlan p, long firstRow, long nRows,
                                 List<GpuBuffer> buffers, List<Long> counts, ComputeBackend backend,
                                 ComputeQueue queue) {
        long perBuffer = rowsPerBuffer(p.deviceRowBytes());
        for (long done = 0; done < nRows; done += perBuffer) {
            long rows = Math.min(perBuffer, nRows - done);
            GpuBuffer dst = allocate(rows * p.deviceRowBytes(), backend);
            copyRows(p, firstRow + done, rows, dst, queue);
            buffers.add(dst);
            counts.add(rows);
        }
    }

    /** Rows of {@code rowBytes} each that fit one device allocation. */
    private long rowsPerBuffer(long rowBytes) {
        return Math.max(1L, weights.maxBufferBytes() / Math.max(1L, rowBytes));
    }

    /** One rank's tensor from the buffers that hold its rows, one buffer or a row-sharded tensor. */
    private static GpuWeight sharded(String name, GGMLType type, long[] ne, List<GpuBuffer> buffers,
                                     List<Long> counts) {
        long[] rows = new long[counts.size()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = counts.get(i);
        }
        return buffers.size() == 1
                ? GpuWeight.of(name, type, ne, buffers.get(0))
                : GpuWeight.sharded(name, type, ne, buffers, rows);
    }

    private void copyRows(GpuWeightUploader.TensorPlan p, long firstRow, long nRows, GpuBuffer dst,
                          ComputeQueue queue) {
        weights.copyRows(p.name(), firstRow, nRows, dst, 0, queue);
    }

    /**
     * Column {@code [firstCol, firstCol + nCols)} of the stored rows {@code [firstRow, firstRow + nRows)} of
     * {@code p} into {@code dst} at byte {@code dstOffset}, one contiguous slice per row ({@code dst} holds
     * {@code nRows} rows of {@code nCols} elements from {@code dstOffset} on).
     */
    private void copyColumns(GpuWeightUploader.TensorPlan p, int firstCol, int nCols, GpuBuffer dst,
                             ComputeQueue queue) {
        copyColumnRows(p, firstCol, nCols, 0L, p.rows(), dst, 0L, queue);
    }

    private void copyColumns(GpuWeightUploader.TensorPlan p, int firstCol, int nCols, GpuBuffer dst, long dstOffset,
                             ComputeQueue queue) {
        copyColumnRows(p, firstCol, nCols, 0L, p.rows(), dst, dstOffset, queue);
    }

    private void copyColumnRows(GpuWeightUploader.TensorPlan p, int firstCol, int nCols, long firstRow, long nRows,
                                GpuBuffer dst, long dstOffset, ComputeQueue queue) {
        if (p.needsDecode()) {
            copyColumnsDecoded(p, firstCol, nCols, firstRow, nRows, dst, dstOffset, queue);
        } else {
            copyColumnsRaw(p, firstCol, nCols, firstRow, nRows, dst, dstOffset, queue);
        }
    }

    /** The stored columns of one row, read at their whole-block byte offsets. */
    private void copyColumnsRaw(GpuWeightUploader.TensorPlan p, int firstCol, int nCols, long firstRow, long nRows,
                                GpuBuffer dst, long dstOffset, ComputeQueue queue) {
        long srcRow = p.sourceRowBytes();
        long offset = p.deviceType().getBytesForElements(firstCol);
        long len = p.deviceType().getBytesForElements(nCols);
        ByteBuffer staged = ByteBuffer.allocateDirect((int) len).order(ByteOrder.LITTLE_ENDIAN);
        for (long r = 0; r < nRows; r++) {
            staged.clear();
            read(p, p.info().offset() + (firstRow + r) * srcRow + offset, (int) len, staged);
            try (GpuEvent e = queue.enqueueWriteBuffer(dst, dstOffset + r * len, len, staged)) {
                // writes copy at enqueue; the caller finishes the queue
            }
        }
    }

    /** The floats of one decoded row, sliced at {@code [firstCol, firstCol + nCols)}. */
    private void copyColumnsDecoded(GpuWeightUploader.TensorPlan p, int firstCol, int nCols, long firstRow,
                                    long nRows, GpuBuffer dst, long dstOffset, ComputeQueue queue) {
        long srcRow = p.sourceRowBytes();
        int cols = (int) p.cols();
        ByteBuffer staged = ByteBuffer.allocateDirect((int) srcRow).order(ByteOrder.LITTLE_ENDIAN);
        float[] decoded = new float[cols];
        float[] slice = new float[nCols];
        for (long r = 0; r < nRows; r++) {
            staged.clear();
            read(p, p.info().offset() + (firstRow + r) * srcRow, (int) srcRow, staged);
            staged.flip();
            CpuBlockDecoders.decodeElements(p.sourceType(), staged, 0L, decoded, 0, cols);
            System.arraycopy(decoded, firstCol, slice, 0, nCols);
            try (GpuEvent e = queue.enqueueWriteBuffer(dst,
                    dstOffset + r * (long) nCols * Float.BYTES, (long) nCols * Float.BYTES, slice)) {
                // writes copy at enqueue; the caller finishes the queue
            }
        }
    }

    /** Folds the projection's single-element {@code <base>.scale} output scale into the shard, as the whole
     * uploader folds it into a resident tensor. */
    private GpuWeight withScaleOf(GpuWeight w) {
        String name = w.name();
        if (!name.endsWith(".weight")) {
            return w;
        }
        String scale = name.substring(0, name.length() - ".weight".length()) + ".scale";
        GpuWeightUploader.TensorPlan p = weights.plan(scale);
        if (p == null || p.info().numElements() != 1) {
            return w;
        }
        return w.withScale(weights.readScalar(scale));
    }

    private GpuWeight withHadamard(GpuWeight w, String name) {
        if (weights.transforms() == null || weights.transforms().isEmpty()) {
            return w;
        }
        com.eonmux.neuralfree.gguf.quant.HadamardSpec hadamard = weights.transforms().hadamard(name);
        return hadamard == null ? w : w.withHadamard(hadamard);
    }

    /** The upload plan of one of the layer's split tensors. */
    private GpuWeightUploader.TensorPlan plan(FfnRowSplitPlan.Layer layer, String suffix) {
        return weights.plan(layer.tensor(suffix));
    }

    private GpuBuffer allocate(long bytes, ComputeBackend backend) {
        GpuBuffer buffer = backend.allocateBuffer(bytes, true);
        owned.add(buffer);
        return buffer;
    }

    private void read(GpuWeightUploader.TensorPlan p, long byteOffset, int n, ByteBuffer dst) {
        try {
            store.readTensorDirect(byteOffset, n, dst);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + n + " bytes of tensor " + p.name() + " at byte "
                    + byteOffset, e);
        }
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
