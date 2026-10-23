package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.serialization.LinearMatricesF;
import com.eonmux.neuralfree.datatype.structures.MatrixF;
import com.eonmux.neuralfree.datatype.structures.ObjectSet;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.*;

public class MatrixFAdapterTest {

    private DataStoringMockBackend mockBackend;
    private MatrixFAdapter adapter;

    @Before
    public void setUp() {
        mockBackend = new DataStoringMockBackend();
        adapter = new MatrixFAdapter(mockBackend);
    }

    @Test
    public void testToTensorBasic() {
        MatrixF matrix = new MatrixF(3, 4);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                matrix.setValue(r, c, r * 4 + c + 1);
            }
        }

        Tensor tensor = adapter.toTensor(matrix);

        assertTrue(tensor.isMatrix());
        assertEquals(3, tensor.shape().dim(0));
        assertEquals(4, tensor.shape().dim(1));
        assertEquals(DType.FLOAT32, tensor.dtype());
    }

    @Test
    public void testToTensorRoundTrip() {
        MatrixF original = new MatrixF(2, 3);
        original.setValue(0, 0, 1.0f);
        original.setValue(0, 1, 2.0f);
        original.setValue(0, 2, 3.0f);
        original.setValue(1, 0, 4.0f);
        original.setValue(1, 1, 5.0f);
        original.setValue(1, 2, 6.0f);

        Tensor tensor = adapter.toTensor(original);
        MatrixF restored = adapter.fromTensor(tensor);

        assertEquals(original.getNumRow(), restored.getNumRow());
        assertEquals(original.getNumCol(), restored.getNumCol());

        for (int r = 0; r < original.getNumRow(); r++) {
            for (int c = 0; c < original.getNumCol(); c++) {
                assertEquals(original.getValue(r, c), restored.getValue(r, c), 0.0001f);
            }
        }
    }

    @Test
    public void testFromTensorBasic() {
        float[][] data = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
        Tensor tensor = Tensor.fromArray(mockBackend, data);

        MatrixF matrix = adapter.fromTensor(tensor);

        assertEquals(2, matrix.getNumRow());
        assertEquals(3, matrix.getNumCol());
        assertEquals(MatrixF.Layout.ROW_MAJOR, matrix.getLayout());
    }

    @Test
    public void testFromTensorNon2DThrows() {
        Tensor vector = Tensor.zeros(mockBackend, Shape.vector(5));

        try {
            adapter.fromTensor(vector);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("2D"));
        }
    }

    @Test
    public void testToTensorColMajorConversion() {
        float[] colMajorData = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
        MatrixF colMajorMatrix = new MatrixF(2, 3, colMajorData, MatrixF.Layout.COL_MAJOR);

        Tensor tensor = adapter.toTensor(colMajorMatrix);
        MatrixF restored = adapter.fromTensor(tensor);

        assertEquals(2, restored.getNumRow());
        assertEquals(3, restored.getNumCol());

        assertEquals(1.0f, restored.getValue(0, 0), 0.0001f);
        assertEquals(2.0f, restored.getValue(0, 1), 0.0001f);
        assertEquals(3.0f, restored.getValue(0, 2), 0.0001f);
        assertEquals(4.0f, restored.getValue(1, 0), 0.0001f);
        assertEquals(5.0f, restored.getValue(1, 1), 0.0001f);
        assertEquals(6.0f, restored.getValue(1, 2), 0.0001f);
    }

    @Test
    public void testToTensorListBasic() {
        ObjectSet<MatrixF> matrices = new ObjectSet<>();
        matrices.add(createMatrix(2, 2, new float[]{1, 2, 3, 4}));
        matrices.add(createMatrix(3, 3, new float[]{5, 6, 7, 8, 9, 10, 11, 12, 13}));

        LinearMatricesF linearMatrices = new LinearMatricesF(matrices);

        List<Tensor> tensors = adapter.toTensorList(linearMatrices);

        assertEquals(2, tensors.size());
        assertTrue(tensors.get(0).isMatrix());
        assertTrue(tensors.get(1).isMatrix());
    }

    @Test
    public void testToTensorListRoundTrip() {
        ObjectSet<MatrixF> originalMatrices = new ObjectSet<>();
        originalMatrices.add(createMatrix(2, 3, new float[]{1, 2, 3, 4, 5, 6}));
        originalMatrices.add(createMatrix(3, 2, new float[]{7, 8, 9, 10, 11, 12}));
        originalMatrices.add(createMatrix(2, 2, new float[]{13, 14, 15, 16}));

        LinearMatricesF original = new LinearMatricesF(originalMatrices);

        List<Tensor> tensors = adapter.toTensorList(original);
        LinearMatricesF restored = adapter.fromTensors(tensors);

        ObjectSet<MatrixF> restoredMatrices = restored.getPlanar();
        assertEquals(3, restoredMatrices.size());

        for (int i = 0; i < 3; i++) {
            MatrixF orig = originalMatrices.get(i);
            MatrixF rest = restoredMatrices.get(i);
            assertEquals(orig.getNumRow(), rest.getNumRow());
            assertEquals(orig.getNumCol(), rest.getNumCol());
            assertArrayEquals(orig.getData(), rest.getData(), 0.0001f);
        }
    }

    @Test
    public void testFromTensorsEmptyThrows() {
        List<Tensor> emptyList = new ArrayList<>();

        try {
            adapter.fromTensors(emptyList);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("empty"));
        }
    }

    @Test
    public void testBackendAccessor() {
        assertSame(mockBackend, adapter.backend());
    }

    @Test(expected = NullPointerException.class)
    public void testNullBackendRejected() {
        new MatrixFAdapter(null);
    }

    @Test(expected = NullPointerException.class)
    public void testToTensorNullMatrix() {
        adapter.toTensor((MatrixF) null);
    }

    @Test(expected = NullPointerException.class)
    public void testFromTensorNullTensor() {
        adapter.fromTensor(null);
    }

    @Test(expected = NullPointerException.class)
    public void testToTensorListNullLinearMatrices() {
        adapter.toTensorList(null);
    }

    @Test(expected = NullPointerException.class)
    public void testFromTensorsNullList() {
        adapter.fromTensors(null);
    }

    @Test
    public void testRoundTripPreservesVariousSizes() {
        int[][] sizes = {{1, 1}, {1, 10}, {10, 1}, {5, 5}, {100, 50}};

        for (int[] size : sizes) {
            int rows = size[0];
            int cols = size[1];
            float[] data = new float[rows * cols];
            for (int i = 0; i < data.length; i++) {
                data[i] = i + 1.5f;
            }
            MatrixF original = new MatrixF(rows, cols, data, MatrixF.Layout.ROW_MAJOR);

            Tensor tensor = adapter.toTensor(original);
            MatrixF restored = adapter.fromTensor(tensor);

            assertEquals("Rows mismatch for size " + rows + "x" + cols, rows, restored.getNumRow());
            assertEquals("Cols mismatch for size " + rows + "x" + cols, cols, restored.getNumCol());

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    float expected = r * cols + c + 1.5f;
                    float actual = restored.getValue(r, c);
                    assertEquals("Value mismatch at [" + r + "," + c + "]", expected, actual, 0.0001f);
                }
            }
        }
    }

    private MatrixF createMatrix(int rows, int cols, float[] data) {
        return new MatrixF(rows, cols, data.clone(), MatrixF.Layout.ROW_MAJOR);
    }

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class DataStoringMockBackend implements ComputeBackend {
        private final AtomicLong bufferIdCounter = new AtomicLong(1);
        private final Map<Long, byte[]> bufferData = new HashMap<>();
        private final DataStoringMockQueue computeQueue = new DataStoringMockQueue(bufferData);
        private final DataStoringMockQueue transferQueue = new DataStoringMockQueue(bufferData);
        private volatile boolean closed = false;

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("DataStoringMockDevice")
                    .totalVramBytes(1024 * 1024 * 1024)
                    .computeUnits(8)
                    .maxClockFrequencyMhz(1000)
                    .maxWorkGroupSize(256)
                    .supportsAsyncTransfer(true)
                    .driverVersion("1.0")
                    .build();
        }

        @Override
        public ComputeQueue computeQueue() {
            return computeQueue;
        }

        @Override
        public ComputeQueue transferQueue() {
            return transferQueue;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            long id = bufferIdCounter.getAndIncrement();
            bufferData.put(id, new byte[(int) sizeInBytes]);
            return new MockGpuBuffer(id, sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return new MockKernelHandle(kernelName);
        }

        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return false;
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            if (artifact == null) {
                throw new IllegalArgumentException("kernel artifact must not be null");
            }
            throw new UnsupportedKernelException("mock backend declines " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }

        @Override
        public void close() {
            closed = true;
            bufferData.clear();
        }
    }

    private static class DataStoringMockQueue implements ComputeQueue {
        private final Map<Long, byte[]> bufferData;
        private volatile boolean closed = false;

        DataStoringMockQueue(Map<Long, byte[]> bufferData) {
            this.bufferData = bufferData;
        }

        @Override
        public QueueType getQueueType() {
            return QueueType.TRANSFER;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(dstBuffer.bufferId());
            if (data != null && dstOffset + sizeInBytes <= data.length) {
                System.arraycopy(srcData, 0, data, (int) dstOffset, (int) sizeInBytes);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(dstBuffer.bufferId());
            if (data != null && dstOffset + sizeInBytes <= data.length) {
                ByteBuffer buffer = ByteBuffer.wrap(data, (int) dstOffset, (int) sizeInBytes).order(ByteOrder.nativeOrder());
                for (float f : srcData) {
                    buffer.putFloat(f);
                }
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(srcBuffer.bufferId());
            if (data != null && srcOffset + sizeInBytes <= data.length) {
                System.arraycopy(data, (int) srcOffset, dstData, 0, (int) sizeInBytes);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(srcBuffer.bufferId());
            if (data != null && srcOffset + sizeInBytes <= data.length) {
                ByteBuffer buffer = ByteBuffer.wrap(data, (int) srcOffset, (int) sizeInBytes).order(ByteOrder.nativeOrder());
                for (int i = 0; i < dstData.length; i++) {
                    dstData[i] = buffer.getFloat();
                }
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public void flush() {
        }

        @Override
        public void finish() {
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final long bufferId;
        private final long sizeInBytes;
        private volatile boolean closed = false;

        MockGpuBuffer(long bufferId, long sizeInBytes) {
            this.bufferId = bufferId;
            this.sizeInBytes = sizeInBytes;
        }

        @Override
        public long sizeInBytes() {
            return sizeInBytes;
        }

        @Override
        public long bufferId() {
            return bufferId;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        private volatile boolean closed = false;

        @Override
        public void awaitCompletion() throws InterruptedException {
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String kernelName;
        private volatile boolean closed = false;

        MockKernelHandle(String kernelName) {
            this.kernelName = kernelName;
        }

        @Override
        public String kernelName() {
            return kernelName;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
