package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The resident upload over a store that maps its data: raw tensors go to the device straight from the mapped views
 * ({@link WeightStore#mappedSlice}) with no staging read, chunks end at the mapping's boundaries, decoded tensors decode
 * from the views, and the device bytes are those of a store without a mapping.
 */
public class WeightRowCopierMappedTest {

    private static final int COLS = 256;
    private static final int STAGING = 1 << 20;
    /** A mapping boundary that no chunk, row or block size divides. */
    private static final int BOUNDARY = 300_007;

    private FakeComputeBackend backend;
    private GpuWeightUploader uploader;

    @After
    public void tearDown() {
        if (uploader != null) {
            uploader.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    private static byte[] blob(int n) {
        byte[] b = new byte[n];
        new Random(7).nextBytes(b);
        return b;
    }

    private GpuWeightUploader upload(List<GGUFTensorInfo> tensors, WeightStore store) {
        backend = FakeComputeBackend.create();
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), tensors, store, "llama",
                GpuWeightUploader.config((type, use) -> type == GGMLType.F32 || type == GGMLType.Q8_0)
                        .stagingBytes(STAGING).build());
        uploader.uploadResident(t -> false);
        return uploader;
    }

    private static byte[] deviceBytes(GpuWeightUploader up, String name) {
        return ((FakeGpuBuffer) up.find(name).shard(0)).toByteArray();
    }

    @Test
    public void rawTensorsUploadFromTheMappedViewsWithoutStagingReads() {
        int rows = 10_000;
        long bytes = GGMLType.Q8_0.getBytesForElements((long) rows * COLS);
        byte[] data = blob((int) bytes + 100);
        BlobStore mapped = new BlobStore(data, BOUNDARY, true);
        List<GGUFTensorInfo> tensors = Collections.singletonList(
                new GGUFTensorInfo("blk.0.ffn_down.weight", new long[]{COLS, rows}, GGMLType.Q8_0, 50L));
        GpuWeightUploader up = upload(tensors, mapped);
        assertArrayEquals(Arrays.copyOfRange(data, 50, 50 + (int) bytes), deviceBytes(up, "blk.0.ffn_down.weight"));
        assertEquals(0, mapped.directReads);
        int chunks = 0;
        for (long at = 50, end = 50 + bytes; at < end; chunks++) {
            at = Math.min(Math.min(end, at + STAGING), (at / BOUNDARY + 1) * BOUNDARY);
        }
        assertEquals("chunks of at most the staging size, cut at every boundary", chunks, mapped.slices);
    }

    @Test
    public void aStoreWithoutAMappingStagesEveryChunkToTheSameBytes() {
        int rows = 10_000;
        long bytes = GGMLType.Q8_0.getBytesForElements((long) rows * COLS);
        byte[] data = blob((int) bytes + 100);
        BlobStore unmapped = new BlobStore(data, BOUNDARY, false);
        List<GGUFTensorInfo> tensors = Collections.singletonList(
                new GGUFTensorInfo("blk.0.ffn_down.weight", new long[]{COLS, rows}, GGMLType.Q8_0, 50L));
        GpuWeightUploader up = upload(tensors, unmapped);
        assertArrayEquals(Arrays.copyOfRange(data, 50, 50 + (int) bytes), deviceBytes(up, "blk.0.ffn_down.weight"));
        assertEquals((bytes + STAGING - 1) / STAGING, unmapped.directReads);
    }

    @Test
    public void decodedTensorsDecodeFromTheMappedViewsAcrossBoundaries() {
        int rows = 3_000;
        int rowBytes = (int) GGMLType.Q4_0.getBytesForElements(COLS);
        byte[] data = blob(rows * rowBytes + 13);
        for (int block = 0; block < rows * COLS / 32; block++) {
            // scale 1.0 (f16 0x3C00) keeps every decoded value finite
            data[13 + block * 18] = 0x00;
            data[13 + block * 18 + 1] = 0x3C;
        }
        List<GGUFTensorInfo> tensors = Collections.singletonList(
                new GGUFTensorInfo("blk.0.ffn_gate.weight", new long[]{COLS, rows}, GGMLType.Q4_0, 13L));
        float[] expected = new float[rows * COLS];
        ByteBuffer src = ByteBuffer.wrap(data, 13, rows * rowBytes).slice().order(ByteOrder.LITTLE_ENDIAN);
        CpuBlockDecoders.decodeElements(GGMLType.Q4_0, src, 0L, expected, 0, expected.length);

        BlobStore mapped = new BlobStore(data, BOUNDARY, true);
        GpuWeightUploader up = upload(tensors, mapped);
        FakeGpuBuffer device = (FakeGpuBuffer) up.find("blk.0.ffn_gate.weight").shard(0);
        assertArrayEquals(expected, device.getFloats(0, expected.length), 0f);
        assertTrue(mapped.slices > 0);
    }

    /** A blob served by reads, and by mapped views cut at every multiple of a boundary when {@code mapped}. */
    private static final class BlobStore implements WeightStore {
        private final byte[] data;
        private final ByteBuffer direct;
        private final int boundary;
        private final boolean mapped;
        int directReads;
        int slices;

        BlobStore(byte[] data, int boundary, boolean mapped) {
            this.data = data;
            this.direct = ByteBuffer.allocateDirect(data.length);
            this.direct.put(data).clear();
            this.boundary = boundary;
            this.mapped = mapped;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            System.arraycopy(data, (int) tensorOffset, dst, dstOffset, size);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            directReads++;
            dst.put(data, (int) tensorOffset, size);
        }

        @Override
        public ByteBuffer mappedSlice(long tensorOffset, long maxSize) {
            if (!mapped) {
                return null;
            }
            slices++;
            int at = (int) tensorOffset;
            int length = (int) Math.min(maxSize, Math.min(boundary - at % boundary, data.length - at));
            ByteBuffer view = direct.asReadOnlyBuffer();
            view.position(at).limit(at + length);
            return view.slice().order(ByteOrder.LITTLE_ENDIAN);
        }

        @Override
        public void close() {
        }
    }
}
