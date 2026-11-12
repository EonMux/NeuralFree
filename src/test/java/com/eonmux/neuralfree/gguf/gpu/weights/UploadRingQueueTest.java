package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import org.junit.After;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Which uploads take the page-locked upload ring ({@link PinnedUploadRing}) on a discrete fake device with host-mapped
 * memory: only chunks enqueued on the uploader's own queue, never another backend's (a tensor-parallel rank's queue
 * cannot read the ring's slots), only in an upload call that moves at least {@link WeightRowCopier#RING_MIN_BYTES}
 * stored bytes, so a one-tensor {@link GpuWeightUploader#upload} of a small tensor does not open and close it, and only
 * on a backend that prefers that staging over its own writes. Every path lands the stored bytes.
 */
public class UploadRingQueueTest {

    private static final int COLS = 1024;
    private static final int ROW_BYTES = COLS * Float.BYTES;
    /** The store's mapping boundary and the period of its byte pattern. */
    private static final int PERIOD = 4 << 20;
    private static final int STAGING = 8 << 20;

    private final List<AutoCloseable> closeables = new ArrayList<>();

    @After
    public void tearDown() throws Exception {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
        }
    }

    private FakeComputeBackend backend() {
        FakeComputeBackend b = FakeComputeBackend.create();
        closeables.add(b);
        return b;
    }

    private GpuWeightUploader uploader(FakeComputeBackend b, List<GGUFTensorInfo> tensors) {
        GpuWeightUploader up = new GpuWeightUploader(b, b.computeQueue(), tensors, new PatternStore(), "llama",
                GpuWeightUploader.config((type, use) -> type == GGMLType.F32).stagingBytes(STAGING).build());
        closeables.add(up);
        return up;
    }

    private static GGUFTensorInfo tensor(String name, long rows, long offset) {
        return new GGUFTensorInfo(name, new long[]{COLS, rows}, GGMLType.F32, offset);
    }

    private static long rowsOf(long bytes) {
        return bytes / ROW_BYTES;
    }

    private static List<FakeOp> ops(FakeComputeBackend b, FakeOpKind kind) {
        List<FakeOp> out = new ArrayList<>();
        for (FakeOp op : b.ops()) {
            if (op.kind() == kind) {
                out.add(op);
            }
        }
        return out;
    }

    /** Asserts that {@code device} holds the store's bytes from {@code storeOffset}. */
    private static void assertStored(FakeGpuBuffer device, long storeOffset, long bytes) {
        byte[] got = device.toByteArray();
        for (long i = 0; i < bytes; i++) {
            if (got[(int) i] != PatternStore.byteAt(storeOffset + i)) {
                fail("device byte " + i + " is " + got[(int) i] + ", the store holds "
                        + PatternStore.byteAt(storeOffset + i));
            }
        }
    }

    @Test
    public void rowCopiesToAnotherBackendsBufferWriteThroughThatBackendsQueue() {
        FakeComputeBackend main = backend();
        FakeComputeBackend peer = backend();
        long rows = rowsOf(3L * PERIOD);
        GpuWeightUploader up = uploader(main, List.of(tensor("blk.0.ffn_down.weight", rows, 512L)));
        long half = rows / 2;
        GpuBuffer dst = peer.allocateBuffer((rows - half) * ROW_BYTES);
        up.copyRows("blk.0.ffn_down.weight", half, rows - half, dst, 0L, peer.computeQueue());
        peer.computeQueue().finish();
        up.syncCopier(peer.computeQueue());

        assertStored((FakeGpuBuffer) dst, 512L + half * ROW_BYTES, (rows - half) * ROW_BYTES);
        assertTrue("the peer's queue writes the rows", ops(peer, FakeOpKind.WRITE).size() > 0);
        assertEquals("no device copy on the peer", 0, ops(peer, FakeOpKind.COPY).size());
        assertEquals("nothing runs on the uploader's queue", 0, ops(main, FakeOpKind.COPY).size()
                + ops(main, FakeOpKind.WRITE).size());
    }

    @Test
    public void aPlannedRingServesOnlyTheQueueOfItsOwnBackend() {
        FakeComputeBackend main = backend();
        FakeComputeBackend peer = backend();
        long rows = rowsOf(2L * PERIOD);
        GpuWeightUploader up = uploader(main, List.of(tensor("blk.0.ffn_up.weight", rows, 0L)));
        GpuWeightUploader.TensorPlan plan = up.plan("blk.0.ffn_up.weight");
        WeightRowCopier copier = new WeightRowCopier(new PatternStore(), STAGING, W12F16Mode.REFUSE, main,
                main.computeQueue());
        try {
            copier.planRing(WeightRowCopier.RING_MIN_BYTES);
            GpuBuffer onPeer = peer.allocateBuffer(rows * ROW_BYTES);
            copier.copyRows(plan, 0L, rows, onPeer, 0L, peer.computeQueue(), null);
            peer.computeQueue().finish();
            assertEquals("another backend's queue writes its chunks", 0, ops(peer, FakeOpKind.COPY).size());
            assertStored((FakeGpuBuffer) onPeer, 0L, rows * ROW_BYTES);

            GpuBuffer onMain = main.allocateBuffer(rows * ROW_BYTES);
            copier.copyRows(plan, 0L, rows, onMain, 0L, main.computeQueue(), null);
            main.computeQueue().finish();
            assertEquals("the ring's own queue copies each chunk from a slot", 2, ops(main, FakeOpKind.COPY).size());
            assertEquals(0, ops(main, FakeOpKind.WRITE).size());
            assertStored((FakeGpuBuffer) onMain, 0L, rows * ROW_BYTES);
        } finally {
            copier.release();
        }
    }

    @Test
    public void aBackendThatPrefersItsOwnWritesKeepsThemForAPlannedRing() {
        FakeComputeBackend b = FakeComputeBackend.builder().parallelUploadStaging(false).build();
        closeables.add(b);
        long rows = rowsOf(2L * PERIOD);
        GpuWeightUploader up = uploader(b, List.of(tensor("blk.0.ffn_up.weight", rows, 0L)));
        WeightRowCopier copier = new WeightRowCopier(new PatternStore(), STAGING, W12F16Mode.REFUSE, b,
                b.computeQueue());
        try {
            copier.planRing(WeightRowCopier.RING_MIN_BYTES);
            GpuBuffer dst = b.allocateBuffer(rows * ROW_BYTES);
            copier.copyRows(up.plan("blk.0.ffn_up.weight"), 0L, rows, dst, 0L, b.computeQueue(), null);
            b.computeQueue().finish();
            assertEquals("no copies from a ring", 0, ops(b, FakeOpKind.COPY).size());
            assertTrue(ops(b, FakeOpKind.WRITE).size() >= 2);
            assertStored((FakeGpuBuffer) dst, 0L, rows * ROW_BYTES);
        } finally {
            copier.release();
        }
    }

    @Test
    public void anUploadBelowTheRingThresholdWritesThroughTheQueue() {
        FakeComputeBackend b = backend();
        long rows = rowsOf(3L * PERIOD);
        GpuWeightUploader up = uploader(b, List.of(tensor("blk.0.attn_q.weight", rows, 0L),
                tensor("blk.0.attn_k.weight", rows, 64L)));
        up.upload("blk.0.attn_q.weight");
        up.upload("blk.0.attn_k.weight");

        assertEquals("no ring for a small tensor", 0, ops(b, FakeOpKind.COPY).size());
        assertTrue(ops(b, FakeOpKind.WRITE).size() >= 6);
        assertStored((FakeGpuBuffer) up.find("blk.0.attn_q.weight").shard(0), 0L, rows * ROW_BYTES);
        assertStored((FakeGpuBuffer) up.find("blk.0.attn_k.weight").shard(0), 64L, rows * ROW_BYTES);
    }

    @Test
    public void aResidentUploadAtTheThresholdTakesTheRingAndClosesIt() {
        FakeComputeBackend b = backend();
        long third = WeightRowCopier.RING_MIN_BYTES / 3 + PERIOD;
        long rows = rowsOf(third);
        List<GGUFTensorInfo> tensors = List.of(tensor("blk.0.ffn_gate.weight", rows, 0L),
                tensor("blk.0.ffn_up.weight", rows, 256L), tensor("blk.0.ffn_down.weight", rows, 1024L));
        assertTrue("together, not alone, at the threshold",
                3 * rows * ROW_BYTES >= WeightRowCopier.RING_MIN_BYTES
                        && rows * ROW_BYTES < WeightRowCopier.RING_MIN_BYTES);
        GpuWeightUploader up = uploader(b, tensors);
        up.uploadResident(t -> false);

        List<FakeOp> copies = ops(b, FakeOpKind.COPY);
        assertTrue("the mapped chunks reach the device as copies from the ring's slots", copies.size() >= 3);
        assertEquals("no queue writes", 0, ops(b, FakeOpKind.WRITE).size());
        for (FakeOp copy : copies) {
            assertTrue("the ring is closed once the upload returns", copy.source().isClosed());
        }
        long[] offsets = {0L, 256L, 1024L};
        for (int i = 0; i < tensors.size(); i++) {
            assertStored((FakeGpuBuffer) up.find(tensors.get(i).name()).shard(0), offsets[i], rows * ROW_BYTES);
        }
    }

    /**
     * A store whose byte {@code i} is a function of {@code i}, mapped in views cut at every multiple of
     * {@link #PERIOD}; it never reads into staging.
     */
    private static final class PatternStore implements WeightStore {
        private static final ByteBuffer PATTERN = pattern();

        private static ByteBuffer pattern() {
            ByteBuffer b = ByteBuffer.allocateDirect(PERIOD);
            for (int i = 0; i < PERIOD; i++) {
                b.put(i, byteAt(i));
            }
            return b;
        }

        static byte byteAt(long i) {
            long k = i % PERIOD;
            return (byte) (k * 31 + (k >>> 9));
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            for (int i = 0; i < size; i++) {
                dst[dstOffset + i] = byteAt(tensorOffset + i);
            }
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            throw new AssertionError("a mapped store is never read into staging");
        }

        @Override
        public ByteBuffer mappedSlice(long tensorOffset, long maxSize) {
            int at = (int) (tensorOffset % PERIOD);
            int length = (int) Math.min(maxSize, PERIOD - at);
            ByteBuffer view = PATTERN.asReadOnlyBuffer();
            view.position(at).limit(at + length);
            return view.slice().order(ByteOrder.LITTLE_ENDIAN);
        }

        @Override
        public void close() {
        }
    }
}
