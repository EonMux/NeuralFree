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
 * Uploads the per-rank shards of a tensor-parallel attention split ({@link TpAttentionPlan}) from the model's tensor
 * data. Per rank and per split layer it uploads the rank's slice of every projection: the head rows of a separate or
 * fused Q/K/V projection (a fused product concatenates the rank's Q, K and V segments into the shard row layout),
 * the matching columns of the output projection, the matching head rows of the per-head Q/K norm weights and biases,
 * and the rank's per-head vectors (sinks, gate bias). The output bias is uploaded to rank 0 only because the
 * all-reduce sums the branch outputs. Stored quant blocks are sliced whole; decoded-to-F32 tensors are sliced as
 * floats.
 *
 * <p>Every {@link #uploadRank} allocates on the rank's backend, enqueues on its queue and finishes it before
 * returning. The uploader owns every buffer it allocates; {@link #close()} releases them.</p>
 */
public final class TpAttentionUploader implements AutoCloseable {

    private final TpAttentionPlan plan;
    private final GpuWeightUploader weights;
    private final WeightStore store;
    private final List<GpuBuffer> owned = new ArrayList<>();

    public TpAttentionUploader(TpAttentionPlan plan, GpuWeightUploader weights, WeightStore store) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.store = Objects.requireNonNull(store, "store");
    }

    /**
     * Uploads rank {@code rank}'s shard of every split attention tensor.
     *
     * @return the rank's shard weights, keyed by their full tensor names ({@code blk.<wl>.<suffix>})
     * @throws IllegalArgumentException when the model has no data for a split tensor
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when a shard does not fit the rank device
     */
    public Map<String, GpuWeight> uploadRank(int rank, ComputeBackend backend, ComputeQueue queue) {
        Map<String, GpuWeight> out = new LinkedHashMap<>();
        for (TpAttentionPlan.Layer layer : plan.layers()) {
            for (Map.Entry<String, TpAttentionPlan.Shard> entry : layer.tensors().entrySet()) {
                String name = layer.tensor(entry.getKey());
                GpuWeightUploader.TensorPlan p = weights.plan(name);
                if (p == null) {
                    throw new IllegalArgumentException("the model has no tensor " + name);
                }
                upload(layer, entry.getValue(), p, rank, backend, queue, out);
            }
        }
        queue.finish();
        weights.syncCopier(queue);
        return out;
    }

    private void upload(TpAttentionPlan.Layer layer, TpAttentionPlan.Shard shard, GpuWeightUploader.TensorPlan p,
                        int rank, ComputeBackend backend, ComputeQueue queue, Map<String, GpuWeight> out) {
        switch (shard) {
            case ROWS:
                out.put(p.name(), rows(layer, p, rank, backend, queue));
                return;
            case FUSED_ROWS:
                out.put(p.name(), fused(layer, p, rank, backend, queue));
                return;
            case NORM_HEADS:
                out.put(p.name(), normHeads(p, rank, backend, queue));
                return;
            case PER_HEAD:
                out.put(p.name(), perHead(p, rank, backend, queue));
                return;
            case OUT_COLUMNS:
                out.put(p.name(), outColumns(layer, p, rank, backend, queue));
                return;
            case OUT_BIAS:
                if (rank == 0) {
                    out.put(p.name(), outputBias(p, backend, queue));
                }
                return;
            case WHOLE:
                out.put(p.name(), whole(p, backend, queue));
                return;
            default:
                throw new IllegalStateException("unknown attention shard " + shard + " of " + p.name());
        }
    }

    /** The rank's uniform row slice of a projection, or its element slice of a bias. */
    private GpuWeight rows(TpAttentionPlan.Layer layer, GpuWeightUploader.TensorPlan p, int rank,
                           ComputeBackend backend, ComputeQueue queue) {
        if (p.ne().length < 2) {
            long per = perRank(p);
            return elements(p, rank * per, per, new long[]{per}, backend, queue);
        }
        long rowCount = perRank(p);
        return rowShard(p, rank * rowCount, rowCount, new long[]{p.ne()[0], rowCount}, backend, queue);
    }

    /** Rows (or elements, for a vector) each rank holds of a tensor that divides evenly. */
    private long perRank(GpuWeightUploader.TensorPlan p) {
        long rows = p.ne().length < 2 ? p.info().numElements() : p.ne()[1];
        long ranks = plan.ranks();
        if (rows % ranks != 0) {
            throw new IllegalArgumentException("tensor " + p.name() + " has " + rows + " row(s), which does not divide"
                    + " into " + ranks + " rank(s)");
        }
        return rows / ranks;
    }

    /** A contiguous row range of a 2-D tensor, as one buffer or a row-sharded tensor. */
    private GpuWeight rowShard(GpuWeightUploader.TensorPlan p, long firstRow, long rowCount, long[] ne,
                               ComputeBackend backend, ComputeQueue queue) {
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, p.deviceRowBytes()));
        for (long done = 0; done < rowCount; done += perBuffer) {
            long n = Math.min(perBuffer, rowCount - done);
            GpuBuffer dst = allocate(n * p.deviceRowBytes(), backend);
            weights.copyRows(p.name(), firstRow + done, n, dst, 0L, queue);
            buffers.add(dst);
            counts.add(n);
        }
        return withCompanions(sharded(p.name(), p.deviceType(), ne, buffers, counts));
    }

    /** The rank's Q, K and V segments of a fused projection, concatenated in shard row order. */
    private GpuWeight fused(TpAttentionPlan.Layer layer, GpuWeightUploader.TensorPlan p, int rank,
                            ComputeBackend backend, ComputeQueue queue) {
        long ranks = plan.ranks();
        long qRows = layer.qProjWidth() / ranks;
        long kRows = layer.kDim() / ranks;
        long vRows = layer.vDim() / ranks;
        long total = qRows + kRows + vRows;
        long[] segStart = {0, layer.qProjWidth(), (long) layer.qProjWidth() + layer.kDim()};
        long[] segRows = {layer.qProjWidth(), layer.kDim(), layer.vDim()};
        long[] shardStart = {(long) rank * qRows, (long) rank * kRows, (long) rank * vRows};
        if (p.ne().length < 2) {
            long[] starts = new long[3];
            for (int i = 0; i < starts.length; i++) {
                starts[i] = segStart[i] + shardStart[i];
            }
            return fusedVector(p, starts, segRows, total, backend, queue);
        }
        long rowBytes = p.deviceRowBytes();
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, rowBytes));
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        for (long done = 0; done < total; done += perBuffer) {
            long n = Math.min(perBuffer, total - done);
            GpuBuffer dst = allocate(n * rowBytes, backend);
            long qEnd = qRows;
            long kEnd = qRows + kRows;
            copyFusedSegment(p, dst, done, n, 0, qEnd, segStart[0] + shardStart[0], queue);
            copyFusedSegment(p, dst, done, n, qEnd, kEnd, segStart[1] + shardStart[1], queue);
            copyFusedSegment(p, dst, done, n, kEnd, total, segStart[2] + shardStart[2], queue);
            buffers.add(dst);
            counts.add(n);
        }
        return withCompanions(sharded(p.name(), p.deviceType(), new long[]{p.ne()[0], total}, buffers, counts));
    }

    /** Copies the part of one fused segment that falls in the output row range {@code [outBase, outBase + n)}. */
    private void copyFusedSegment(GpuWeightUploader.TensorPlan p, GpuBuffer dst, long outBase, long n,
                                  long segLo, long segHi, long sourceFirst, ComputeQueue queue) {
        long lo = Math.max(segLo, outBase);
        long hi = Math.min(segHi, outBase + n);
        if (lo >= hi) {
            return;
        }
        weights.copyRows(p.name(), sourceFirst + (lo - segLo), hi - lo, dst,
                (lo - outBase) * p.deviceRowBytes(), queue);
    }

    /** The rank's segments of a fused bias vector. */
    private GpuWeight fusedVector(GpuWeightUploader.TensorPlan p, long[] starts, long[] segRows, long total,
                                  ComputeBackend backend, ComputeQueue queue) {
        requireF32(p);
        GpuBuffer dst = allocate(total * Float.BYTES, backend);
        long offset = 0;
        for (int i = 0; i < segRows.length; i++) {
            long per = segRows[i] / plan.ranks();
            copyElements(p, starts[i], per, dst, offset * Float.BYTES, queue);
            offset += per;
        }
        return GpuWeight.of(p.name(), GGMLType.F32, new long[]{total}, dst);
    }

    /** The rank's head rows of a per-head Q/K norm tensor, preserving a per-head-scalar shape. */
    private GpuWeight normHeads(GpuWeightUploader.TensorPlan p, int rank, ComputeBackend backend,
                                ComputeQueue queue) {
        long total = p.info().numElements();
        long per = total / plan.ranks();
        long[] ne = p.ne().length >= 2 ? new long[]{p.ne()[0], per / Math.max(1, p.ne()[0])} : new long[]{per};
        return elements(p, rank * per, per, ne, backend, queue);
    }

    /** The rank's slice of a per-head vector (one value per query or KV head). */
    private GpuWeight perHead(GpuWeightUploader.TensorPlan p, int rank, ComputeBackend backend,
                              ComputeQueue queue) {
        long total = p.info().numElements();
        if (total % plan.ranks() != 0) {
            throw new IllegalArgumentException("tensor " + p.name() + " holds " + total + " values, which does not"
                    + " divide into " + plan.ranks() + " rank(s)");
        }
        long per = total / plan.ranks();
        return elements(p, rank * per, per, new long[]{per}, backend, queue);
    }

    /** The rank's column block of the output projection. */
    private GpuWeight outColumns(TpAttentionPlan.Layer layer, GpuWeightUploader.TensorPlan p, int rank,
                                 ComputeBackend backend, ComputeQueue queue) {
        int firstCol = layer.outColumnStart(rank);
        int nCols = layer.outColumns();
        long rowBytes = p.deviceType().getBytesForElements(nCols);
        long perBuffer = Math.max(1L, weights.maxBufferBytes() / Math.max(1L, rowBytes));
        long rowsTotal = p.ne()[1];
        List<GpuBuffer> buffers = new ArrayList<>();
        List<Long> counts = new ArrayList<>();
        for (long done = 0; done < rowsTotal; done += perBuffer) {
            long rows = Math.min(perBuffer, rowsTotal - done);
            GpuBuffer dst = allocate(rows * rowBytes, backend);
            copyColumnRows(p, firstCol, nCols, done, rows, dst, 0L, queue);
            buffers.add(dst);
            counts.add(rows);
        }
        return withCompanions(sharded(p.name(), p.deviceType(), new long[]{nCols, rowsTotal}, buffers, counts));
    }

    /** The whole output-width bias, on rank 0. */
    private GpuWeight outputBias(GpuWeightUploader.TensorPlan p, ComputeBackend backend, ComputeQueue queue) {
        GpuBuffer dst = allocate(Math.max(p.deviceBytes(), Float.BYTES), backend);
        weights.copyRows(p.name(), 0, 1, dst, 0L, queue);
        return GpuWeight.of(p.name(), GGMLType.F32, new long[]{p.info().numElements()}, dst);
    }

    /** A tensor every rank needs whole (a Q/K norm row shared by the heads, or any tensor of a context split). */
    private GpuWeight whole(GpuWeightUploader.TensorPlan p, ComputeBackend backend, ComputeQueue queue) {
        long rows = Math.max(1L, p.ne().length < 2 ? 1L : p.ne()[1]);
        long[] ne = p.ne().length < 2 ? new long[]{p.info().numElements()} : new long[]{p.ne()[0], rows};
        if (p.ne().length >= 2 && p.deviceBytes() > weights.maxBufferBytes()) {
            /* A projection larger than one allocation uploads as row shards, as a tensor-split slice does. */
            return rowShard(p, 0L, rows, ne, backend, queue);
        }
        GpuBuffer dst = allocate(Math.max(p.deviceBytes(), Float.BYTES), backend);
        weights.copyRows(p.name(), 0, rows, dst, 0L, queue);
        return withCompanions(GpuWeight.of(p.name(), p.deviceType(), ne, dst));
    }

    /** A contiguous element slice of an F32 vector. */
    private GpuWeight elements(GpuWeightUploader.TensorPlan p, long first, long count, long[] ne,
                               ComputeBackend backend, ComputeQueue queue) {
        requireF32(p);
        GpuBuffer dst = allocate(count * Float.BYTES, backend);
        copyElements(p, first, count, dst, 0L, queue);
        return GpuWeight.of(p.name(), GGMLType.F32, ne, dst);
    }

    private void requireF32(GpuWeightUploader.TensorPlan p) {
        if (p.deviceType() != GGMLType.F32) {
            throw new UnsupportedOperationException("tensor " + p.name() + " is " + p.deviceType()
                    + "; a split per-head vector needs F32");
        }
    }

    private void copyElements(GpuWeightUploader.TensorPlan p, long first, long count, GpuBuffer dst,
                              long dstOffset, ComputeQueue queue) {
        if (p.needsDecode()) {
            throw new UnsupportedOperationException("tensor " + p.name() + " is decoded on upload and cannot be"
                    + " sliced as a per-head vector");
        }
        ByteBuffer staged = ByteBuffer.allocateDirect((int) Math.min(Integer.MAX_VALUE, count * Float.BYTES))
                .order(ByteOrder.LITTLE_ENDIAN);
        read(p, p.info().offset() + first * Float.BYTES, (int) (count * Float.BYTES), staged);
        try (GpuEvent e = queue.enqueueWriteBuffer(dst, dstOffset, count * Float.BYTES, staged)) {
            // the write copies at enqueue
        }
    }

    /** {@code nRows} rows of {@code nCols} elements from column {@code firstCol} of every stored row. */
    private void copyColumnRows(GpuWeightUploader.TensorPlan p, int firstCol, int nCols, long firstRow, long nRows,
                                GpuBuffer dst, long dstOffset, ComputeQueue queue) {
        if (p.needsDecode()) {
            copyColumnsDecoded(p, firstCol, nCols, firstRow, nRows, dst, dstOffset, queue);
        } else {
            copyColumnsRaw(p, firstCol, nCols, firstRow, nRows, dst, dstOffset, queue);
        }
    }

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
                // writes copy at enqueue
            }
        }
    }

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
                // writes copy at enqueue
            }
        }
    }

    private static GpuWeight sharded(String name, GGMLType type, long[] ne, List<GpuBuffer> buffers,
                                     List<Long> counts) {
        long[] rows = new long[counts.size()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = counts.get(i);
        }
        return buffers.size() == 1 ? GpuWeight.of(name, type, ne, buffers.get(0))
                : GpuWeight.sharded(name, type, ne, buffers, rows);
    }

    /** Folds the projection's single-element {@code <base>.scale} into the shard, and its Hadamard if any. */
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
            com.eonmux.neuralfree.gguf.quant.HadamardSpec hadamard = weights.transforms().hadamard(name);
            if (hadamard != null) {
                out = out.withHadamard(hadamard);
            }
        }
        return out;
    }

    private void read(GpuWeightUploader.TensorPlan p, long byteOffset, int n, ByteBuffer dst) {
        try {
            store.readTensorDirect(byteOffset, n, dst);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + n + " bytes of tensor " + p.name() + " at byte "
                    + byteOffset, e);
        }
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
