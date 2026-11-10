package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Embedding-table lookups of one {@link GpuOps}: the device-id lookup of a single-shard table
 * ({@code {T}_BatchEmbed} over token ids already on the device), the host-id lookup of any shard layout (the ids
 * uploaded for a single-shard table, one {@code {T}_Embed} row launch per token otherwise), and the table's inverse
 * Hadamard transform after either. The public entries are the {@link GpuOps#embed} overloads. Not thread-safe; owned
 * by its GpuOps.
 */
final class GpuEmbeddings {

    private final GpuOps ops;
    private final ComputeQueue queue;
    private final GpuScratch scratch;

    /** Binds the lookups to the queue and scratch of the owning GpuOps. */
    GpuEmbeddings(GpuOps ops, ComputeQueue queue, GpuScratch scratch) {
        this.ops = ops;
        this.queue = queue;
        this.scratch = scratch;
    }

    void embed(GpuWeight table, GpuBuffer tokenIds, int nTokens, GpuBuffer out, int outOff) {
        ops.ensureOpen();
        int dim = GpuOps.cols(table);
        GpuOps.rows(table);
        GpuOps.checkTokens(nTokens);
        Objects.requireNonNull(tokenIds, "tokenIds");
        if ((long) nTokens * Integer.BYTES > tokenIds.sizeInBytes()) {
            throw new IllegalArgumentException("token id buffer of " + tokenIds.sizeInBytes() + " bytes is too small"
                    + " for " + nTokens + " ids");
        }
        if (!table.isSingleShard()) {
            throw new UnsupportedOperationException("embedding table " + table.name() + " is split into "
                    + table.shardCount() + " row shards; device token ids cannot select the shard, use the host-id"
                    + " lookup");
        }
        long n = (long) nTokens * dim;
        GpuOps.checkView("embedding output of " + table.name(), out, outOff, n);
        GpuOps.checkInt(n, "embedding output size of " + table.name());
        WeightKernels k = ops.family(table);
        HadamardSpec inv = table.inverseHadamard();
        boolean direct = outOff == 0 && (inv == null || inv.inPlaceAllowed());
        GpuOps.View dst = direct ? new GpuOps.View(out, 0) : stagingView(GpuScratch.Staging.OUTPUT, n);
        ops.launchMatvec(k.batchEmbed, n, dim, nTokens, tokenIds, table.shard(0), dst.buf);
        finishLookup(inv, dst, out, outOff, nTokens, dim);
    }

    void embed(GpuWeight table, int[] tokens, int count, GpuBuffer out, int outOff) {
        ops.ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (count < 0 || count > tokens.length) {
            throw new IllegalArgumentException("embedding lookup count " + count + " is outside [0, "
                    + tokens.length + "]");
        }
        int nTokens = count;
        int dim = GpuOps.cols(table);
        GpuOps.checkTokens(nTokens);
        long rows = table.rows();
        for (int t = 0; t < nTokens; t++) {
            if (tokens[t] < 0 || tokens[t] >= rows) {
                throw new IllegalArgumentException("token id " + tokens[t] + " at " + t + " is outside the "
                        + rows + " rows of " + table.name());
            }
        }
        long n = (long) nTokens * dim;
        GpuOps.checkView("embedding output of " + table.name(), out, outOff, n);
        GpuOps.checkInt(outOff + n, "embedding output end of " + table.name());
        if (table.isSingleShard()) {
            ScratchSlot ids = scratch.staging(GpuScratch.Staging.INPUT, nTokens);
            writeInts(ids, tokens, nTokens);
            embed(table, ids.buffer(), nTokens, out, outOff);
            return;
        }
        WeightKernels k = ops.family(table);
        HadamardSpec inv = table.inverseHadamard();
        GpuOps.View rowsDst = inv != null && !inv.inPlaceAllowed()
                ? stagingView(GpuScratch.Staging.RESULT, n) : new GpuOps.View(out, outOff);
        for (int t = 0; t < nTokens; t++) {
            int s = table.shardOfRow(tokens[t]);
            int localId = (int) (tokens[t] - table.shardRowStart(s));
            int at = rowsDst.off + t * dim;
            if (at == 0) {
                ops.launchMatvec(k.embed, dim, dim, localId, table.shard(s), rowsDst.buf);
            } else {
                ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, dim);
                ops.launchMatvec(k.embed, dim, dim, localId, table.shard(s), stage.buffer());
                ops.copy2dUnchecked(1, dim, stage.buffer(), stage.offset(), dim, rowsDst.buf, at, dim);
            }
        }
        if (inv != null) {
            ops.hadamard(inv, rowsDst.buf, rowsDst.off, out, outOff, nTokens, dim);
        }
    }

    /** Moves looked-up rows from {@code dst} to the output, through the inverse transform when there is one. */
    private void finishLookup(HadamardSpec inv, GpuOps.View dst, GpuBuffer out, int outOff, int nTokens, int dim) {
        boolean inPlace = dst.buf == out && dst.off == outOff;
        if (inv != null) {
            ops.hadamard(inv, dst.buf, dst.off, out, outOff, nTokens, dim);
        } else if (!inPlace) {
            ops.copy2dUnchecked(nTokens, dim, dst.buf, dst.off, dim, out, outOff, dim);
        }
    }

    /** Uploads the first {@code count} token ids to {@code slot}, dropping the norm-written Q8_1 record first. */
    private void writeInts(ScratchSlot slot, int[] values, int count) {
        ByteBuffer bb = ByteBuffer.allocate(count * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            bb.putInt(values[i]);
        }
        ops.q8.invalidate();
        GpuEvent e = queue.enqueueWriteBuffer(slot.buffer(), (long) slot.offset() * Float.BYTES,
                (long) count * Integer.BYTES, bb.array());
        if (e != null) {
            e.close();
        }
    }

    private GpuOps.View stagingView(GpuScratch.Staging which, long floats) {
        ScratchSlot s = scratch.staging(which, floats);
        return new GpuOps.View(s.buffer(), s.offset());
    }
}
