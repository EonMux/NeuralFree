package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet.Entry;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Copies the rows of a streamed tensor's shard from the host tier into host-mapped memory, in the shard's upload type:
 * stored bytes are read straight into the mapping, and an entry the host dequantizes is read in chunks of at most
 * {@link LayerSlots#CHUNK_BYTES} bytes into a staging buffer, decoded to F32 and written. Not thread-safe.
 */
final class ShardFiller {

    private final WeightDataProvider data;
    private ByteBuffer staging;
    private float[] decoded;

    ShardFiller(WeightDataProvider data) {
        this.data = Objects.requireNonNull(data, "data");
    }

    /**
     * Copies shard {@code k} of {@code e} into {@code buffer} from byte {@code dstOffset}.
     *
     * @param what names the copy in a read failure
     * @throws UncheckedIOException when the host tier cannot read the tensor
     */
    void fill(Entry e, int k, HostMappedBuffer buffer, long dstOffset, String what) {
        long storedRow = e.storedRowBytes();
        long deviceRow = e.uploadRowBytes();
        long rowStart = rowStart(e, k);
        long rows = e.shardRows(k);
        int chunk = chunkRows(e);
        for (long done = 0; done < rows; ) {
            int n = (int) Math.min(chunk, rows - done);
            long srcOff = (rowStart + done) * storedRow;
            int srcBytes = (int) (n * storedRow);
            long dstOff = dstOffset + done * deviceRow;
            try {
                if (e.dequantize()) {
                    ByteBuffer in = staging();
                    in.clear();
                    data.readRange(e.name(), srcOff, srcBytes, in);
                    int elems = (int) (n * e.cols());
                    float[] out = floats(elems);
                    CpuBlockDecoders.decodeElements(e.source().type(), in, 0, out, 0, elems);
                    buffer.hostView(dstOff, elems * Float.BYTES).asFloatBuffer().put(out, 0, elems);
                } else {
                    data.readRange(e.name(), srcOff, srcBytes, buffer.hostView(dstOff, srcBytes));
                }
            } catch (IOException ex) {
                throw new UncheckedIOException(what + " '" + e.name() + "' failed: " + ex.getMessage(), ex);
            }
            done += n;
        }
    }

    /** Row {@code 0} of shard {@code k}: shards are contiguous from the tensor's first row. */
    static long rowStart(Entry e, int k) {
        long start = 0;
        for (int i = 0; i < k; i++) {
            start += e.shardRows(i);
        }
        return start;
    }

    /** Rows per host chunk of one tensor: at most {@link LayerSlots#CHUNK_BYTES}, at least one row. */
    static int chunkRows(Entry e) {
        long rowBytes = Math.max(e.storedRowBytes(), e.uploadRowBytes());
        long rows = Math.max(1, LayerSlots.CHUNK_BYTES / rowBytes);
        long maxShard = 0;
        for (int k = 0; k < e.shardCount(); k++) {
            maxShard = Math.max(maxShard, e.shardRows(k));
        }
        return (int) Math.min(rows, maxShard);
    }

    private ByteBuffer staging() {
        if (staging == null) {
            staging = ByteBuffer.allocateDirect(LayerSlots.CHUNK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        }
        return staging;
    }

    /** The reusable decode buffer, grown to at least {@code elems} floats. */
    private float[] floats(int elems) {
        if (decoded == null || decoded.length < elems) {
            decoded = new float[elems];
        }
        return decoded;
    }
}
