package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuAllReduce;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The run-time half of the vocabulary split ({@link TpVocabPlan}): the token lookup summed over the ranks and the
 * logits collected from them.
 *
 * <p>The lookup runs the rank's own slice of the token table ({@link GpuOps#embed}) with the step's tokens remapped to
 * local row indices, zeroes the rows whose token falls outside the slice, and sums every rank's rows into the model
 * device's embedding rows ({@link GpuAllReduce#reduceToRoot}). The LM head's product runs on every rank over the
 * same hidden rows; this class replays the hidden rows to the peers, runs their output product and bias, and copies
 * each rank's logits slice into the model device's full row at the rank's column range. Rank 0's own slice is
 * computed by the head before {@link #gather}.</p>
 *
 * <p>The peer products use each rank's {@link ScratchSlots#LOGITS} slot and the replayed input slot the group keeps
 * for the attention exchange. Not thread-safe; the forward path drives it from the model's step.</p>
 */
final class TpVocab implements GpuEmbeddingStage.VocabParallel, GpuLogitsHead.VocabSplit {

    private final TpVocabPlan plan;
    private final TensorParallelGroup group;
    private final GpuWeight[] tables;
    private final GpuWeight[] heads;
    private final GpuWeight[] biases;
    private final int ubatch;
    private final ScratchSlot shardLogits;
    private final GpuBuffer shardBuffer;
    /** Host staging of a peer's logits slice and of one of its rows, reused by {@link #copyStrided}. */
    private float[] stridedStage;
    private float[] stridedRow;

    /**
     * @param tables  each rank's token-table shard (rank 0's included)
     * @param heads   each rank's LM-head shard, or null when the head is tied to the token table
     * @param biases  each rank's head-bias shard, or null without a bias
     */
    TpVocab(TpVocabPlan plan, TensorParallelGroup group, GpuWeight[] tables, GpuWeight[] heads, GpuWeight[] biases,
            int ubatch) {
        this.plan = Objects.requireNonNull(plan, "plan");
        this.group = Objects.requireNonNull(group, "group");
        this.tables = tables.clone();
        this.heads = heads.clone();
        this.biases = biases.clone();
        this.ubatch = ubatch;
        int shard = (int) plan.headCount(0);
        ComputeBackend backend = group.rankBackend(0);
        this.shardBuffer = backend.allocateBuffer(Math.max(1L, (long) ubatch * shard) * Float.BYTES, true);
        this.shardLogits = ScratchSlot.of("tp.vocab.shard", shardBuffer, 0, Math.max(1L, (long) ubatch * shard));
    }

    // ------------------------------------------------------------------ embedding

    @Override
    public int embeddingStart() {
        return 0;
    }

    @Override
    public int embeddingCount() {
        return (int) plan.tokenCount(0);
    }

    @Override
    public int embeddingVocab() {
        return (int) plan.tokenRows();
    }

    @Override
    public void lookup(int[] tokens, int off, int n, ScratchSlot out) {
        Objects.requireNonNull(tokens, "tokens");
        if (off < 0 || n < 0 || off + n > tokens.length) {
            throw new IllegalArgumentException("lookup of " + n + " rows from " + off + " exceeds "
                    + tokens.length + " token(s)");
        }
        int width = (int) tables[0].cols();
        out.require((long) n * width);
        lookupRank(0, tokens, off, n, width, out.buffer(), out.offset());
        for (int rank = 1; rank < tables.length; rank++) {
            ScratchSlot slot = group.inputSlot(rank).require((long) n * width);
            lookupRank(rank, tokens, off, n, width, slot.buffer(), slot.offset());
        }
        List<GpuAllReduce.Participant> participants = new ArrayList<>(tables.length);
        participants.add(new GpuAllReduce.Participant(group.rankQueue(0), out.buffer(), out.offset()));
        for (int rank = 1; rank < tables.length; rank++) {
            ScratchSlot slot = group.inputSlot(rank);
            participants.add(new GpuAllReduce.Participant(group.rankQueue(rank), slot.buffer(), slot.offset()));
        }
        GpuAllReduce.reduceToRoot(group.rankOps(0), group.allReduceTemporary(), participants,
                Math.multiplyExact(n, width));
    }

    /** Looks up the rows that fall in rank {@code rank}'s slice and zeroes the rest. */
    private void lookupRank(int rank, int[] tokens, int off, int n, int width, GpuBuffer dst, int dstOff) {
        long start = plan.tokenStart(rank);
        long end = start + plan.tokenCount(rank);
        int[] ids = new int[n];
        GpuOps ops = group.rankOps(rank);
        for (int i = 0; i < n; i++) {
            int token = tokens[off + i];
            boolean inside = token >= start && token < end;
            ids[i] = inside ? (int) (token - start) : 0;
        }
        ops.embed(tables[rank], ids, n, dst, dstOff);
        // a token outside the slice looked up row 0; its row must contribute nothing to the sum
        for (int i = 0; i < n; i++) {
            int token = tokens[off + i];
            if (token < start || token >= end) {
                ops.fill(dst, (dstOff + (long) i * width) * Float.BYTES, (long) width * Float.BYTES, 0);
            }
        }
    }

    // ------------------------------------------------------------------ LM head

    @Override
    public int headStart() {
        return 0;
    }

    @Override
    public int headCount() {
        return (int) plan.headCount(0);
    }

    @Override
    public int headVocab() {
        return (int) plan.headRows();
    }

    @Override
    public ScratchSlot shardLogits() {
        return shardLogits;
    }

    @Override
    public void gather(int nRows, ScratchSlot hidden, ScratchSlot shardLogits, ScratchSlot fullLogits) {
        int full = (int) plan.headRows();
        int shard = (int) plan.headCount(0);
        if (hidden.floats() % nRows != 0) {
            throw new IllegalArgumentException("the head selection holds " + hidden.floats() + " floats for " + nRows
                    + " rows");
        }
        int width = (int) (hidden.floats() / nRows);
        GpuOps root = group.rankOps(0);
        fullLogits.require((long) nRows * full);
        root.copy2d(shardLogits.buffer(), shardLogits.offset(), shard, fullLogits.buffer(), fullLogits.offset(),
                full, nRows, shard);
        for (int rank = 1; rank < tables.length; rank++) {
            ScratchSlot input = group.inputSlot(rank).require((long) nRows * width);
            RankExchange.replay(root.queue(), hidden.buffer(), hidden.offset(), group.rankQueue(rank),
                    input.buffer(), (long) nRows * width);
            ScratchSlot peerLogits = group.rankScratch(rank).slot(ScratchSlots.LOGITS);
            if (peerLogits.offset() != 0 || peerLogits.floats() < (long) nRows * plan.headCount(rank)) {
                throw new IllegalStateException("rank" + rank + " logits slot " + peerLogits + " cannot hold "
                        + nRows + " rows of " + plan.headCount(rank) + " values at element 0");
            }
            GpuOps ops = group.rankOps(rank);
            GpuWeight head = heads[rank] != null ? heads[rank] : tables[rank];
            ops.matmul(head, input.buffer(), input.offset(), peerLogits.buffer(), 0, nRows);
            if (biases[rank] != null) {
                int widthShard = (int) plan.headCount(rank);
                ops.enqueue(GpuOps.BATCH_BIAS_ADD,
                        new long[]{roundUp(widthShard, ops.local())}, new long[]{ops.local()}, widthShard, nRows,
                        biases[rank].shard(0), peerLogits.buffer());
            }
            copyStrided(rank, peerLogits.buffer(), nRows, (int) plan.headCount(rank), (int) plan.headStart(rank),
                    fullLogits);
        }
        root.queue().finish();
    }

    /**
     * Copies rank {@code rank}'s logits slice into the model device's full logits row: the peer writes every row at
     * its column range, staging the slice through host memory because the copy is strided. The host arrays are
     * reused across calls (a write copies its source at enqueue); the first row is written straight from the slice.
     */
    private void copyStrided(int rank, GpuBuffer src, int nRows, int cols, int dstStart, ScratchSlot fullLogits) {
        int full = (int) plan.headRows();
        int floats = Math.multiplyExact(nRows, cols);
        if (stridedStage == null || stridedStage.length < floats) {
            stridedStage = new float[floats];
        }
        float[] staged = stridedStage;
        try (com.eonmux.neuralfree.acceleration.backend.GpuEvent event = group.rankQueue(rank)
                .enqueueReadBuffer(src, 0L, (long) floats * Float.BYTES, staged)) {
            group.rankQueue(rank).finish();
        }
        ComputeQueue root = group.rankQueue(0);
        for (int row = 0; row < nRows; row++) {
            float[] values = staged;
            if (row > 0) {
                if (stridedRow == null || stridedRow.length < cols) {
                    stridedRow = new float[cols];
                }
                System.arraycopy(staged, row * cols, stridedRow, 0, cols);
                values = stridedRow;
            }
            root.enqueueWriteBuffer(fullLogits.buffer(),
                    (fullLogits.offset() + (long) row * full + dstStart) * Float.BYTES,
                    (long) cols * Float.BYTES, values);
        }
        root.finish();
    }

    private static long roundUp(long n, int local) {
        return (n + local - 1) / local * local;
    }

    void close() {
        shardBuffer.close();
    }
}
