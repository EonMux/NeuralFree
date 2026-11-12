package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * A tensor-parallel rank's row copy on CUDA: two backends on one device stand in for two GPUs, an uploader on the first
 * backend reads a memory-mapped raw tensor, and its rows are copied into a buffer of the second backend on that
 * backend's queue. The rows must land through the second backend's own writes: the first backend's page-locked upload
 * ring is not readable there. Skipped without a CUDA device.
 */
public class UploadRingPeerQueueCudaTest {

    private static final int COLS = 256;
    private static final int ROWS = 4096;

    private HWAcceleration hw;
    private final List<AutoCloseable> closeables = new ArrayList<>();

    @Before
    public void setUp() {
        HWAcceleration.enableCuda = true;
        hw = new HWAcceleration();
        Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
    }

    @After
    public void tearDown() throws Exception {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    private <T extends AutoCloseable> T own(T c) {
        closeables.add(c);
        return c;
    }

    @Test
    public void rowsOfAMappedTensorReachAnotherBackendsBufferOnItsQueue() throws InterruptedException {
        int bytes = (int) GGMLType.Q8_0.getBytesForElements((long) ROWS * COLS);
        byte[] data = new byte[bytes];
        new Random(3).nextBytes(data);
        ComputeBackend main = own(new CudaComputeBackend(hw.getCUDevices().get(0)));
        ComputeBackend peer = own(new CudaComputeBackend(hw.getCUDevices().get(0)));
        List<GGUFTensorInfo> tensors = Collections.singletonList(
                new GGUFTensorInfo("blk.0.ffn_down.weight", new long[]{COLS, ROWS}, GGMLType.Q8_0, 0L));
        GpuWeightUploader up = own(new GpuWeightUploader(main, main.computeQueue(), tensors, new MappedBytes(data),
                "llama", GpuWeightUploader.config((type, use) -> type == GGMLType.F32 || type == GGMLType.Q8_0)
                .stagingBytes(1 << 20).build()));

        int half = ROWS / 2;
        long halfBytes = GGMLType.Q8_0.getBytesForElements((long) half * COLS);
        GpuBuffer dst = own(peer.allocateBuffer(bytes - halfBytes));
        up.copyRows("blk.0.ffn_down.weight", half, ROWS - half, dst, 0L, peer.computeQueue());
        peer.computeQueue().finish();
        up.syncCopier(peer.computeQueue());

        byte[] back = new byte[(int) (bytes - halfBytes)];
        try (GpuEvent read = peer.computeQueue().enqueueReadBuffer(dst, 0, back.length, back)) {
            read.awaitCompletion();
        }
        byte[] expected = new byte[back.length];
        System.arraycopy(data, (int) halfBytes, expected, 0, expected.length);
        assertArrayEquals(expected, back);
    }

    /** The bytes of one in-memory tensor blob, served as one mapping. */
    private static final class MappedBytes implements WeightStore {
        private final byte[] data;
        private final ByteBuffer direct;

        MappedBytes(byte[] data) {
            this.data = data;
            this.direct = ByteBuffer.allocateDirect(data.length);
            this.direct.put(data).clear();
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
            dst.put(data, (int) tensorOffset, size);
        }

        @Override
        public ByteBuffer mappedSlice(long tensorOffset, long maxSize) {
            ByteBuffer view = direct.asReadOnlyBuffer();
            view.position((int) tensorOffset).limit((int) Math.min(data.length, tensorOffset + maxSize));
            return view.slice().order(ByteOrder.LITTLE_ENDIAN);
        }

        @Override
        public void close() {
        }
    }
}
