package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.utils.LoggingUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Objects;

/**
 * {@link WeightStore} over the {@link DataSegment}s of several files, such as the splits of a GGUF model.
 *
 * <p>Tensor offsets (as stored in {@link GGUFTensorInfo#offset()}) are in one virtual address space in which segment
 * {@code i} holds {@code [virtualBase, virtualEnd)}. Reads are dispatched to the {@link MappedWeightStore} of the
 * segment holding the offset by binary search over the segment bases. Segments without data (header-only splits)
 * open no store.</p>
 */
public final class MultiShardMappedWeightStore implements WeightStore {

    private final MappedWeightStore[] shards;
    /** Virtual base of each segment, in increasing order. */
    private final long[] shardBaseOffsets;
    private final DataSegment[] segments;

    /**
     * Maps every segment that holds data.
     *
     * @param segments the segments in increasing, non-overlapping virtual order
     * @throws IllegalArgumentException when there are no segments or they are out of order or overlap
     * @throws IOException              when a file cannot be opened
     */
    public MultiShardMappedWeightStore(List<DataSegment> segments) throws IOException {
        Objects.requireNonNull(segments, "segments");
        DataSegment[] segs = segments.toArray(new DataSegment[0]);
        if (segs.length == 0) {
            throw new IllegalArgumentException("a multi-segment weight store needs at least one data segment");
        }
        long[] bases = new long[segs.length];
        for (int i = 0; i < segs.length; i++) {
            Objects.requireNonNull(segs[i], "data segment " + i);
            if (i > 0 && segs[i].virtualBase() < segs[i - 1].virtualEnd()) {
                throw new IllegalArgumentException("data segment " + i + " (" + segs[i] + ") starts before the end"
                        + " of segment " + (i - 1) + " (" + segs[i - 1] + ")");
            }
            bases[i] = segs[i].virtualBase();
        }
        this.segments = segs;
        this.shardBaseOffsets = bases;
        this.shards = new MappedWeightStore[segs.length];

        int emptyShards = 0;
        try {
            for (int i = 0; i < segs.length; i++) {
                if (segs[i].dataLength() > 0) {
                    shards[i] = new MappedWeightStore(segs[i].path(), segs[i].fileDataOffset(), segs[i].dataLength());
                } else {
                    emptyShards++;
                }
            }
        } catch (IOException | RuntimeException e) {
            for (MappedWeightStore opened : shards) {
                if (opened != null) {
                    try {
                        opened.close();
                    } catch (IOException closeFailure) {
                        e.addSuppressed(closeFailure);
                    }
                }
            }
            throw e;
        }

        String extra = emptyShards > 0 ? " (" + emptyShards + " header-only shard(s) skipped)" : "";
        LoggingUtil.getLogger(this).info(
                "MultiShardMappedWeightStore: {} shards, total virtual address space {} MB{}",
                segs.length, segs[segs.length - 1].virtualEnd() / (1024 * 1024), extra);
    }

    @Override
    public void readTensor(long globalOffset, int size, byte[] dst) throws IOException {
        readTensor(globalOffset, size, dst, 0);
    }

    @Override
    public void readTensor(long globalOffset, int size, byte[] dst, int dstOffset) throws IOException {
        int shard = findShard(globalOffset);
        long localOffset = globalOffset - shardBaseOffsets[shard];
        storeOf(shard, globalOffset, size).readTensor(localOffset, size, dst, dstOffset);
    }

    @Override
    public void readTensorDirect(long globalOffset, int size, ByteBuffer dst) throws IOException {
        int shard = findShard(globalOffset);
        long localOffset = globalOffset - shardBaseOffsets[shard];
        storeOf(shard, globalOffset, size).readTensorDirect(localOffset, size, dst);
    }

    /** The store of {@code shard}, which must hold {@code [globalOffset, globalOffset + size)}. */
    private MappedWeightStore storeOf(int shard, long globalOffset, int size) throws IOException {
        if (!segments[shard].contains(globalOffset, size) || shards[shard] == null) {
            throw new IOException("bytes [" + globalOffset + ", +" + size + ") do not lie inside one data segment;"
                    + " the nearest is " + segments[shard]);
        }
        return shards[shard];
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
