package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class AutogradTest {

    private MockComputeBackend backend;

    @Before
    public void setUp() {
        backend = new MockComputeBackend();
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        ComputationGraph.current().clear();
    }

    @Test
    public void testAddGradSimple() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f});
        b = b.withRequiresGrad(true);

        Tensor c = TensorOps.add(a, b);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());
        
        float[] gradA = a.grad().toFloatArray();
        float[] gradB = b.grad().toFloatArray();
        
        assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f}, gradA, 1e-6f);
        assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f}, gradB, 1e-6f);
    }

    @Test
    public void testMulGradSimple() {
        Tensor a = Tensor.fromArray(backend, new float[]{2.0f, 3.0f, 4.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{5.0f, 6.0f, 7.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.mul(a, b);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());
        
        float[] gradA = a.grad().toFloatArray();
        float[] gradB = b.grad().toFloatArray();
        
        assertArrayEquals(new float[]{5.0f, 6.0f, 7.0f}, gradA, 1e-6f);
        assertArrayEquals(new float[]{2.0f, 3.0f, 4.0f}, gradB, 1e-6f);
    }

    @Test
    public void testSubGradSimple() {
        Tensor a = Tensor.fromArray(backend, new float[]{10.0f, 8.0f, 6.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.sub(a, b);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());
        
        float[] gradA = a.grad().toFloatArray();
        float[] gradB = b.grad().toFloatArray();
        
        assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f}, gradA, 1e-6f);
        assertArrayEquals(new float[]{-1.0f, -1.0f, -1.0f}, gradB, 1e-6f);
    }

    @Test
    public void testDivGradSimple() {
        Tensor a = Tensor.fromArray(backend, new float[]{10.0f, 20.0f, 30.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{2.0f, 5.0f, 6.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.div(a, b);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());
        
        float[] gradA = a.grad().toFloatArray();
        float[] gradB = b.grad().toFloatArray();
        
        assertArrayEquals(new float[]{0.5f, 0.2f, 0.16666667f}, gradA, 1e-5f);
    }

    @Test
    public void testNegGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.neg(a);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{-1.0f, -1.0f, -1.0f}, gradA, 1e-6f);
    }

    @Test
    public void testExpGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{0.0f, 1.0f, 2.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.exp(a);
        assertTrue(c.requiresGrad());

        float[] expectedExp = new float[]{1.0f, (float) Math.E, (float) Math.E * (float) Math.E};
        float[] actualExp = c.toFloatArray();
        assertArrayEquals(expectedExp, actualExp, 1e-4f);

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(expectedExp, gradA, 1e-4f);
    }

    @Test
    public void testLogGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, (float) Math.E, (float) Math.E * (float) Math.E}).withRequiresGrad(true);

        Tensor c = TensorOps.log(a);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        
        float[] expected = new float[]{1.0f / 1.0f, 1.0f / (float) Math.E, 1.0f / ((float) Math.E * (float) Math.E)};
        assertArrayEquals(expected, gradA, 1e-5f);
    }

    @Test
    public void testSqrtGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{4.0f, 9.0f, 16.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.sqrt(a);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        
        float[] expected = new float[]{1.0f / (2.0f * 2.0f), 1.0f / (2.0f * 3.0f), 1.0f / (2.0f * 4.0f)};
        assertArrayEquals(expected, gradA, 1e-6f);
    }

    @Test
    public void testPowGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{2.0f, 3.0f, 4.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.pow(a, 2.0f);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        
        float[] expected = new float[]{4.0f, 6.0f, 8.0f};
        assertArrayEquals(expected, gradA, 1e-5f);
    }

    @Test
    public void testMatmulGrad() {
        Tensor a = Tensor.fromArray(backend, new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[][]{{5.0f, 6.0f}, {7.0f, 8.0f}}).withRequiresGrad(true);

        Tensor c = TensorOps.matmul(a, b);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());
        
        float[] gradA = a.grad().toFloatArray();
        float[] gradB = b.grad().toFloatArray();
        
        assertEquals(4, gradA.length);
        assertEquals(4, gradB.length);
    }

    @Test
    public void testSumGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f, 4.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.sum(a);
        assertTrue(c.requiresGrad());
        assertEquals(Shape.scalar(), c.shape());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, gradA, 1e-6f);
    }

    @Test
    public void testMeanGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{2.0f, 4.0f, 6.0f, 8.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.mean(a);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{0.25f, 0.25f, 0.25f, 0.25f}, gradA, 1e-6f);
    }

    @Test
    public void testReshapeGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).withRequiresGrad(true);

        Tensor c = a.reshape(Shape.matrix(2, 3));
        assertTrue(c.requiresGrad());
        assertEquals(Shape.matrix(2, 3), c.shape());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, gradA, 1e-6f);
    }

    @Test
    public void testTransposeGrad() {
        Tensor a = Tensor.fromArray(backend, new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}}).withRequiresGrad(true);

        Tensor c = a.transpose();
        assertTrue(c.requiresGrad());
        assertEquals(Shape.matrix(2, 2), c.shape());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, gradA, 1e-6f);
    }

    @Test
    public void testComposedGrad() {
        Tensor x = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
        Tensor y = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f}).withRequiresGrad(true);

        Tensor z = TensorOps.mul(x, y);
        Tensor w = TensorOps.add(z, x);

        w.backward();

        assertNotNull(x.grad());
        assertNotNull(y.grad());
        
        float[] gradX = x.grad().toFloatArray();
        float[] gradY = y.grad().toFloatArray();
        
        float[] expectedGradX = new float[]{5.0f, 6.0f, 7.0f};
        float[] expectedGradY = new float[]{1.0f, 2.0f, 3.0f};
        
        assertArrayEquals(expectedGradX, gradX, 1e-5f);
        assertArrayEquals(expectedGradY, gradY, 1e-5f);
    }

    @Test
    public void testNoGradScope() {
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            assertFalse(ComputationGraph.isGradEnabled());
            
            Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
            Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f}).withRequiresGrad(true);
            Tensor c = TensorOps.add(a, b);
            
            assertTrue(ComputationGraph.current().isEmpty());
            assertFalse("a result computed under noGrad does not require grad", c.requiresGrad());
            assertNull(c.gradFn());
        }
        
        assertTrue(ComputationGraph.isGradEnabled());
    }

    @Test
    public void testScalarMulGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.mul(a, 2.0f);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{2.0f, 2.0f, 2.0f}, gradA, 1e-6f);
    }

    @Test
    public void testScalarAddGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.add(a, 5.0f);
        assertTrue(c.requiresGrad());

        c.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{1.0f, 1.0f, 1.0f}, gradA, 1e-6f);
    }

    @Test
    public void testCatGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.cat(Arrays.asList(a, b), 0);
        assertTrue(c.requiresGrad());
        assertEquals(4, c.numElements());

        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());
        
        float[] gradA = a.grad().toFloatArray();
        float[] gradB = b.grad().toFloatArray();
        
        assertArrayEquals(new float[]{1.0f, 1.0f}, gradA, 1e-6f);
        assertArrayEquals(new float[]{1.0f, 1.0f}, gradB, 1e-6f);
    }

    @Test
    public void testStackGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.stack(Arrays.asList(a, b), 0);
        assertTrue(c.requiresGrad());
        assertEquals(Shape.of(2, 2), c.shape());

        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());
        
        float[] gradA = a.grad().toFloatArray();
        float[] gradB = b.grad().toFloatArray();
        
        assertArrayEquals(new float[]{1.0f, 1.0f}, gradA, 1e-6f);
        assertArrayEquals(new float[]{1.0f, 1.0f}, gradB, 1e-6f);
    }

    @Test
    public void testGradientAccumulation() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{1.0f, 1.0f, 1.0f});

        Tensor c = TensorOps.add(a, b);
        Tensor d = TensorOps.add(c, a);

        d.backward();

        assertNotNull(a.grad());
        float[] gradA = a.grad().toFloatArray();
        assertArrayEquals(new float[]{2.0f, 2.0f, 2.0f}, gradA, 1e-6f);
    }

    @Test
    public void testGraphRecording() {
        ComputationGraph.current().clear();
        
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f}).withRequiresGrad(true);
        
        assertEquals(0, ComputationGraph.current().size());
        
        Tensor c = TensorOps.add(a, b);
        assertEquals(1, ComputationGraph.current().size());
        
        Tensor d = TensorOps.mul(c, a);
        assertEquals(2, ComputationGraph.current().size());
    }

    @Test
    public void testZeroGrad() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.add(a, b);
        c.backward();

        assertNotNull(a.grad());
        assertNotNull(b.grad());

        ComputationGraph.current().zeroGrad();

        assertNull(a.grad());
        assertNull(b.grad());
    }

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class MockComputeBackend implements ComputeBackend {
        private final Map<Long, MockGpuBuffer> buffers = new HashMap<>();
        private final MockComputeQueue computeQueue = new MockComputeQueue(this);
        private final MockComputeQueue transferQueue = new MockComputeQueue(this);
        private volatile boolean closed = false;
        private long nextBufferId = 1;

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                .deviceName("MockDevice")
                .totalVramBytes(1024 * 1024 * 1024)
                .computeUnits(8)
                .maxClockFrequencyMhz(1000)
                .maxWorkGroupSize(256)
                .supportsAsyncTransfer(true)
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
            MockGpuBuffer buffer = new MockGpuBuffer(sizeInBytes, nextBufferId++);
            buffers.put(buffer.bufferId(), buffer);
            return buffer;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String source, String buildOptions) {
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
            if (!closed) {
                closed = true;
                for (MockGpuBuffer buffer : buffers.values()) {
                    buffer.close();
                }
                buffers.clear();
            }
        }

        MockGpuBuffer getBuffer(long id) {
            return buffers.get(id);
        }
    }

    private static class MockComputeQueue implements ComputeQueue {
        private final MockComputeBackend backend;
        private volatile boolean closed = false;

        MockComputeQueue(MockComputeBackend backend) {
            this.backend = backend;
        }

        @Override
        public QueueType getQueueType() {
            return QueueType.COMPUTE;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer buffer, long dstOffset, long sizeInBytes, byte[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.writeBytes(data, (int) dstOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer buffer, long dstOffset, long sizeInBytes, float[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.writeFloats(data, (int) dstOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer buffer, long srcOffset, long sizeInBytes, byte[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.readBytes(data, (int) srcOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer buffer, long srcOffset, long sizeInBytes, float[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.readFloats(data, (int) srcOffset);
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
        private final byte[] data;
        private final long sizeInBytes;
        private final long bufferId;
        private volatile boolean closed = false;

        MockGpuBuffer(long sizeInBytes, long bufferId) {
            this.sizeInBytes = sizeInBytes;
            this.bufferId = bufferId;
            this.data = new byte[(int) sizeInBytes];
        }

        void writeBytes(byte[] src, int offset) {
            System.arraycopy(src, 0, data, offset, src.length);
        }

        void writeFloats(float[] src, int offset) {
            ByteBuffer buffer = ByteBuffer.wrap(data, offset, src.length * 4).order(ByteOrder.nativeOrder());
            for (float f : src) {
                buffer.putFloat(f);
            }
        }

        void readBytes(byte[] dst, int offset) {
            System.arraycopy(data, offset, dst, 0, dst.length);
        }

        void readFloats(float[] dst, int offset) {
            ByteBuffer buffer = ByteBuffer.wrap(data, offset, dst.length * 4).order(ByteOrder.nativeOrder());
            for (int i = 0; i < dst.length; i++) {
                dst[i] = buffer.getFloat();
            }
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
        private volatile boolean complete = true;
        private volatile boolean closed = false;

        @Override
        public void awaitCompletion() {
            complete = true;
        }

        @Override
        public boolean isComplete() {
            return complete;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String kernelName;

        MockKernelHandle(String kernelName) {
            this.kernelName = kernelName;
        }

        @Override
        public String kernelName() {
            return kernelName;
        }

        @Override
        public void close() {
        }
    }
}
