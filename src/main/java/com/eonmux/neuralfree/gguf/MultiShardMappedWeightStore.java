package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.utils.LoggingUtil;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * {@link WeightStore} implementation that spans multiple GGUF shard files.
 *
 * <p>Tensor offsets (as stored in {@link GGUFTensorInfo#offset()}) are in a unified
 * logical address space where shard N's tensors are offset by the cumulative data
 * length of all preceding shards.  Reads are dispatched to the correct per-shard
 * {@link MappedWeightStore} by binary-searching {@code shardBaseOffsets}.</p>
 */
public final class MultiShardMappedWeightStore implements WeightStore {

    private final MappedWeightStore[] shards;
    /**
     * shardBaseOffsets[i] = sum of shards[0..i-1] data lengths.
     * shardBaseOffsets[0] is always 0.
     */
    private final long[] shardBaseOffsets;

    /**
     * @param shardPaths       path to each shard file (length = splitCount)
     * @param fileDataOffsets  byte offset within each file where the data section starts
     * @param virtualBaseOffsets cumulative virtual offset for each shard (index 0 = 0)
     * @param shardDataLengths  byte length of each shard's data section
     */
    public MultiShardMappedWeightStore(
            String[] shardPaths,
            long[] fileDataOffsets,
            long[] virtualBaseOffsets,
            long[] shardDataLengths) throws IOException {

        int n = shardPaths.length;
        this.shards = new MappedWeightStore[n];
        this.shardBaseOffsets = virtualBaseOffsets.clone();

        int emptyShards = 0;
        for (int i = 0; i < n; i++) {
            if (shardDataLengths[i] > 0) {
                shards[i] = new MappedWeightStore(shardPaths[i], fileDataOffsets[i], shardDataLengths[i]);
            }
            // else: shards[i] stays null — header-only shard, never accessed during reads
            else {
                emptyShards++;
            }
        }

        String extra = emptyShards > 0 ? " (" + emptyShards + " header-only shard(s) skipped)" : "";
        LoggingUtil.getLogger(this).info(
                "MultiShardMappedWeightStore: {} shards, total virtual address space {} MB{}",
                n, (virtualBaseOffsets[n - 1] + shardDataLengths[n - 1]) / (1024 * 1024), extra);
    }

    @Override
    public void readTensor(long globalOffset, int size, byte[] dst) throws IOException {
        readTensor(globalOffset, size, dst, 0);
    }

    @Override
    public void readTensor(long globalOffset, int size, byte[] dst, int dstOffset) throws IOException {
        int shard = findShard(globalOffset);
        long localOffset = globalOffset - shardBaseOffsets[shard];
        shards[shard].readTensor(localOffset, size, dst, dstOffset);
    }

    @Override
    public void readTensorDirect(long globalOffset, int size, ByteBuffer dst) throws IOException {
        int shard = findShard(globalOffset);
        long localOffset = globalOffset - shardBaseOffsets[shard];
        shards[shard].readTensorDirect(localOffset, size, dst);
    }

    /** Returns a defensive copy of the shard base offset array (for diagnostics). */
    public long[] shardBaseOffsets() { return shardBaseOffsets.clone(); }

    /**
     * Finds which shard contains a given global offset via binary search on
     * {@code shardBaseOffsets}.  Returns the index of the last shard whose
     * base offset is ≤ {@code globalOffset}.
     */
    private int findShard(long globalOffset) {
        int lo = 0;
        int hi = shardBaseOffsets.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (shardBaseOffsets[mid] <= globalOffset) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /** Closes every shard; the first failure is thrown with the later ones attached as suppressed exceptions. */
    @Override
    public void close() throws IOException {
        IOException first = null;
        for (MappedWeightStore shard : shards) {
            try {
                if (shard != null) shard.close();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null) throw first;
    }
}
