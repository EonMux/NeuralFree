package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import org.junit.Test;
import org.junit.Before;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.*;

public class TensorTest {

    private MockComputeBackend mockBackend;

    @Before
    public void setUp() {
        mockBackend = new MockComputeBackend();
    }

    @Test
    public void testZeros() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));

        assertEquals(Shape.vector(5), tensor.shape());
        assertEquals(DType.FLOAT32, tensor.dtype());
        assertEquals(5, tensor.numElements());
        assertFalse(tensor.requiresGrad());
    }

    @Test
    public void testZerosInt32() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.matrix(3, 4), DType.INT32);

        assertEquals(Shape.matrix(3, 4), tensor.shape());
        assertEquals(DType.INT32, tensor.dtype());
        assertEquals(12, tensor.numElements());
    }

    @Test
    public void testZerosScalar() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.scalar());

        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(1, tensor.numElements());
        assertTrue(tensor.isScalar());
    }

    @Test
    public void testOnes() {
        Tensor tensor = Tensor.ones(mockBackend, Shape.vector(5));

        assertEquals(Shape.vector(5), tensor.shape());
        assertEquals(DType.FLOAT32, tensor.dtype());
    }

    @Test
    public void testOnesInt32Throws() {
        try {
            Tensor.ones(mockBackend, Shape.vector(5), DType.INT32);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("floating-point"));
        }
    }

    @Test
    public void testRand() {
        Tensor tensor = Tensor.rand(mockBackend, Shape.vector(100), DType.FLOAT32, new Random(42));

        assertEquals(Shape.vector(100), tensor.shape());
        assertEquals(DType.FLOAT32, tensor.dtype());
    }

    @Test
    public void testRandDeterministic() {
        Tensor t1 = Tensor.rand(mockBackend, Shape.vector(10), DType.FLOAT32, new Random(123));
        Tensor t2 = Tensor.rand(mockBackend, Shape.vector(10), DType.FLOAT32, new Random(123));

        assertArrayEquals(t1.toFloatArray(), t2.toFloatArray(), 0.0f);
    }

    @Test
    public void testRandn() {
        Tensor tensor = Tensor.randn(mockBackend, Shape.vector(100), DType.FLOAT32, new Random(42));

        assertEquals(Shape.vector(100), tensor.shape());
        assertEquals(DType.FLOAT32, tensor.dtype());
    }

    @Test
    public void testRandnDeterministic() {
        Tensor t1 = Tensor.randn(mockBackend, Shape.vector(10), DType.FLOAT32, new Random(456));
        Tensor t2 = Tensor.randn(mockBackend, Shape.vector(10), DType.FLOAT32, new Random(456));

        assertArrayEquals(t1.toFloatArray(), t2.toFloatArray(), 0.0f);
    }

    @Test
    public void testFromArrayFloat1D() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Tensor tensor = Tensor.fromArray(mockBackend, data);

        assertEquals(Shape.vector(5), tensor.shape());
        assertEquals(DType.FLOAT32, tensor.dtype());
    }

    @Test
    public void testFromArrayFloat2D() {
        float[][] data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        Tensor tensor = Tensor.fromArray(mockBackend, data);

        assertEquals(Shape.matrix(2, 2), tensor.shape());
        assertEquals(DType.FLOAT32, tensor.dtype());
    }

    @Test
    public void testFromArrayInt() {
        int[] data = {1, 2, 3, 4, 5};
        Tensor tensor = Tensor.fromArray(mockBackend, data);

        assertEquals(Shape.vector(5), tensor.shape());
        assertEquals(DType.INT32, tensor.dtype());
    }

    @Test
    public void testFromArrayByte() {
        byte[] data = {1, 2, 3, 4, 5};
        Tensor tensor = Tensor.fromArray(mockBackend, data);

        assertEquals(Shape.vector(5), tensor.shape());
        assertEquals(DType.INT8, tensor.dtype());
    }

    @Test
    public void testEye() {
        Tensor tensor = Tensor.eye(mockBackend, 3);

        assertEquals(Shape.matrix(3, 3), tensor.shape());
        assertEquals(DType.FLOAT32, tensor.dtype());
    }

    @Test
    public void testEyeRectangular() {
        Tensor tensor = Tensor.eye(mockBackend, 2, 4);

        assertEquals(Shape.matrix(2, 4), tensor.shape());
    }

    @Test
    public void testScalar() {
        Tensor tensor = Tensor.scalar(mockBackend, 3.14f);

        assertEquals(Shape.scalar(), tensor.shape());
        assertTrue(tensor.isScalar());
    }

    @Test
    public void testReshape() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(12));
        Tensor reshaped = tensor.reshape(3, 4);

        assertEquals(Shape.matrix(3, 4), reshaped.shape());
        assertEquals(12, reshaped.numElements());
    }

    @Test
    public void testReshapeInferDimension() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(12));
        Tensor reshaped = tensor.reshape(3, -1);

        assertEquals(Shape.matrix(3, 4), reshaped.shape());
    }

    @Test
    public void testReshapeScalar() {
        Tensor tensor = Tensor.fromArray(mockBackend, new float[]{42.0f});
        Tensor scalar = tensor.reshape(Shape.scalar());

        assertEquals(Shape.scalar(), scalar.shape());
    }

    @Test
    public void testTranspose2D() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.matrix(3, 4));
        Tensor transposed = tensor.transpose();

        assertEquals(Shape.matrix(4, 3), transposed.shape());
    }

    @Test
    public void testTransposeWithAxes() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.of(2, 3, 4));
        Tensor transposed = tensor.transpose(2, 0, 1);

        assertEquals(Shape.of(4, 2, 3), transposed.shape());
    }

    @Test
    public void testPermute() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.of(2, 3, 4));
        Tensor permuted = tensor.permute(2, 0, 1);

        assertEquals(Shape.of(4, 2, 3), permuted.shape());
    }

    @Test
    public void testSqueeze() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.of(1, 3, 1, 4, 1));
        Tensor squeezed = tensor.squeeze();

        assertEquals(Shape.of(3, 4), squeezed.shape());
        assertArrayEquals(new long[]{4, 1}, squeezed.strides().strides());
        assertEquals(0L, squeezed.strides().offset());
    }

    @Test
    public void testSqueezeKeepsStridesOfEveryRemainingAxis() {
        Tensor leading = Tensor.zeros(mockBackend, Shape.of(1, 1, 3));
        assertArrayEquals(new long[]{1}, leading.squeeze().strides().strides());

        Tensor interleaved = Tensor.zeros(mockBackend, Shape.of(2, 1, 1, 5));
        assertArrayEquals(new long[]{5, 1}, interleaved.squeeze().strides().strides());

        Tensor allUnit = Tensor.zeros(mockBackend, Shape.of(1, 1, 1));
        Tensor scalar = allUnit.squeeze();
        assertEquals(0, scalar.shape().rank());
        assertEquals(0, scalar.strides().rank());
    }

    @Test
    public void testSqueezeOfOffsetStridedViewKeepsStridesAndOffset() {
        Tensor source = Tensor.zeros(mockBackend, Shape.of(2, 1, 3, 4));
        // a [1, 1, 3, 1] view at offset 13 with strides [12, 12, 4, 1]
        Tensor view = source.slice(0, 1, 2).slice(3, 1, 2);
        Tensor squeezed = view.squeeze();

        assertEquals(Shape.vector(3), squeezed.shape());
        assertArrayEquals(new long[]{4}, squeezed.strides().strides());
        assertEquals(13L, squeezed.strides().offset());
    }

    @Test
    public void testSqueezeAxis() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.of(1, 3, 4));
        Tensor squeezed = tensor.squeeze(0);

        assertEquals(Shape.of(3, 4), squeezed.shape());
    }

    @Test
    public void testUnsqueeze() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        Tensor unsqueezed = tensor.unsqueeze(0);

        assertEquals(Shape.of(1, 5), unsqueezed.shape());
    }

    @Test
    public void testUnsqueezeNegativeAxis() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        Tensor unsqueezed = tensor.unsqueeze(-1);

        assertEquals(Shape.of(5, 1), unsqueezed.shape());
    }

    @Test
    public void testExpand() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(1));
        Tensor expanded = tensor.expand(Shape.vector(5));

        assertEquals(Shape.vector(5), expanded.shape());
    }

    @Test
    public void testContiguousOnContiguousTensor() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.matrix(3, 4));
        Tensor contiguous = tensor.contiguous();

        assertSame(tensor, contiguous);
    }

    @Test
    public void testContiguousOnTransposedTensor() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.matrix(3, 4));
        Tensor transposed = tensor.transpose();

        assertFalse(transposed.isContiguous());

        Tensor contiguous = transposed.contiguous();
        assertTrue(contiguous.isContiguous());
        assertEquals(Shape.matrix(4, 3), contiguous.shape());
    }

    @Test
    public void testToByteBuffer() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor tensor = Tensor.fromArray(mockBackend, data);

        ByteBuffer buffer = tensor.toByteBuffer();

        assertEquals(16, buffer.remaining());
    }

    @Test
    public void testToFloat32() {
        int[] data = {1, 2, 3, 4, 5};
        Tensor tensor = Tensor.fromArray(mockBackend, data);

        assertEquals(DType.INT32, tensor.dtype());

        Tensor converted = tensor.to(DType.FLOAT32);

        assertEquals(DType.FLOAT32, converted.dtype());
        assertEquals(tensor.shape(), converted.shape());
    }

    @Test
    public void testToSameDType() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5), DType.FLOAT32);
        Tensor converted = tensor.to(DType.FLOAT32);

        assertSame(tensor, converted);
    }

    @Test
    public void testWithRequiresGrad() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        assertFalse(tensor.requiresGrad());

        Tensor withGrad = tensor.withRequiresGrad(true);
        assertTrue(withGrad.requiresGrad());
        assertNull("withRequiresGrad(true) creates a leaf", withGrad.gradFn());
        assertNotSame(tensor, withGrad);
        assertFalse("the source handle keeps its flag", tensor.requiresGrad());
        assertSame(withGrad, withGrad.withRequiresGrad(true));
    }

    @Test
    public void testDetach() {
        ComputationGraph.current().clear();
        boolean gradMode = ComputationGraph.isGradEnabled();
        ComputationGraph.setGradEnabled(true);
        try {
            Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5)).withRequiresGrad(true);
            assertTrue(tensor.requiresGrad());

            Tensor detached = tensor.detach();
            assertFalse(detached.requiresGrad());
            assertNull(detached.gradFn());

            Tensor recorded = tensor.reshape(5, 1);
            assertTrue("a view of a leaf in grad mode is a recorded node output", recorded.requiresGrad());
            assertNotNull(recorded.gradFn());
            Tensor detachedView = recorded.detach();
            assertFalse(detachedView.requiresGrad());
            assertNull("detach() drops the gradient function", detachedView.gradFn());
        } finally {
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(gradMode);
        }
    }

    @Test
    public void testSetGrad() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5)).withRequiresGrad(true);
        Tensor grad = Tensor.zeros(mockBackend, Shape.vector(5));

        tensor.setGrad(grad);

        assertSame(grad, tensor.grad());
    }

    @Test(expected = IllegalStateException.class)
    public void testSetGradOnNonRequiresGradTensor() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        Tensor grad = Tensor.zeros(mockBackend, Shape.vector(5));

        tensor.setGrad(grad);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetGradWithWrongShape() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5)).withRequiresGrad(true);
        Tensor grad = Tensor.zeros(mockBackend, Shape.vector(3));

        tensor.setGrad(grad);
    }

    @Test
    public void testBackwardRequiresGrad() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5)).withRequiresGrad(true);
        tensor.backward();
    }

    @Test
    public void testIsScalarIsVectorIsMatrix() {
        Tensor scalar = Tensor.scalar(mockBackend, 1.0f);
        Tensor vector = Tensor.zeros(mockBackend, Shape.vector(5));
        Tensor matrix = Tensor.zeros(mockBackend, Shape.matrix(3, 4));
        Tensor tensor3d = Tensor.zeros(mockBackend, Shape.of(2, 3, 4));

        assertTrue(scalar.isScalar());
        assertFalse(scalar.isVector());
        assertFalse(scalar.isMatrix());

        assertFalse(vector.isScalar());
        assertTrue(vector.isVector());
        assertFalse(vector.isMatrix());

        assertFalse(matrix.isScalar());
        assertFalse(matrix.isVector());
        assertTrue(matrix.isMatrix());

        assertFalse(tensor3d.isScalar());
        assertFalse(tensor3d.isVector());
        assertFalse(tensor3d.isMatrix());
    }

    @Test
    public void testByteSize() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.matrix(10, 20), DType.FLOAT32);

        assertEquals(800, tensor.byteSize());
    }

    @Test
    public void testClose() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));

        assertFalse(tensor.toString().contains("closed=true"));

        tensor.close();

        assertTrue(tensor.toString().contains("closed=true"));
    }

    @Test(expected = IllegalStateException.class)
    public void testOperationsAfterClose() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        tensor.close();

        tensor.toFloatArray();
    }

    @Test(expected = NullPointerException.class)
    public void testZerosNullBackend() {
        Tensor.zeros(null, Shape.vector(5));
    }

    @Test(expected = NullPointerException.class)
    public void testZerosNullShape() {
        Tensor.zeros(mockBackend, null);
    }

    @Test(expected = NullPointerException.class)
    public void testFromArrayNullBackend() {
        Tensor.fromArray(null, new float[]{1.0f});
    }

    @Test(expected = NullPointerException.class)
    public void testFromArrayNullData() {
        Tensor.fromArray(mockBackend, (float[]) null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromArrayEmpty2D() {
        Tensor.fromArray(mockBackend, new float[0][]);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromArrayJagged2D() {
        Tensor.fromArray(mockBackend, new float[][]{{1.0f, 2.0f}, {3.0f}});
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testTranspose1D() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        tensor.transpose();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSqueezeInvalidAxis() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        tensor.squeeze(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExpandIncompatibleShape() {
        Tensor tensor = Tensor.zeros(mockBackend, Shape.vector(5));
        tensor.expand(Shape.vector(3));
    }

    @Test
    public void testViewKeepsStorageAlive() {
        Tensor original = Tensor.zeros(mockBackend, Shape.vector(100));
        Tensor reshaped = original.reshape(10, 10);
        
        original.close();
        
        assertEquals("View should still work after original closed", Shape.matrix(10, 10), reshaped.shape());
        reshaped.close();
    }

    @Test
    public void testTensorViewStorageRefCount() {
        Tensor original = Tensor.zeros(mockBackend, Shape.vector(100));
        assertEquals(1, original.storage().refCount());
        
        Tensor reshaped = original.reshape(10, 10);
        assertEquals(2, original.storage().refCount());
        assertEquals(2, reshaped.storage().refCount());
        
        original.close();
        assertEquals(1, reshaped.storage().refCount());
        
        reshaped.close();
    }

    @Test
    public void testMultipleTensorViewsKeepStorageAlive() {
        Tensor original = Tensor.zeros(mockBackend, Shape.matrix(4, 5));
        Tensor reshaped = original.reshape(2, 10);
        Tensor transposed = original.transpose();
        Tensor squeezed = original.unsqueeze(0);
        
        assertEquals(4, original.storage().refCount());
        
        original.close();
        assertEquals(3, reshaped.storage().refCount());
        
        reshaped.close();
        assertEquals(2, transposed.storage().refCount());
        
        transposed.close();
        assertEquals(1, squeezed.storage().refCount());
        
        squeezed.close();
        assertEquals(0, mockBackend.aliveBufferCount());
    }

    @Test
    public void testTensorStressCreateClose1000Tensors() {
        long before = mockBackend.aliveBufferCount();
        
        for (int i = 0; i < 1000; i++) {
            Tensor t = Tensor.zeros(mockBackend, Shape.vector(100));
            t.close();
        }
        
        long after = mockBackend.aliveBufferCount();
        assertEquals("All tensor buffers should be freed", before, after);
    }

    @Test
    public void testTensorStressWithViews() {
        long before = mockBackend.aliveBufferCount();
        
        for (int i = 0; i < 100; i++) {
            Tensor t = Tensor.zeros(mockBackend, Shape.matrix(10, 10));
            Tensor r1 = t.reshape(5, 20);
            Tensor r2 = t.transpose();
            Tensor r3 = t.unsqueeze(0);
            
            t.close();
            r1.close();
            r2.close();
            r3.close();
        }
        
        long after = mockBackend.aliveBufferCount();
        assertEquals("All tensor buffers with views should be freed", before, after);
    }

    @Test
    public void testDetachedTensorSharesStorage() {
        Tensor original = Tensor.zeros(mockBackend, Shape.vector(10)).withRequiresGrad(true);
        Tensor detached = original.detach();
        
        assertEquals(3, original.storage().refCount());
        
        original.close();
        assertEquals(2, detached.storage().refCount());
        
        detached.close();
    }

    @Test
    public void testContiguousCreatesNewStorage() {
        Tensor original = Tensor.zeros(mockBackend, Shape.matrix(3, 4));
        Tensor transposed = original.transpose();
        
        assertEquals(2, original.storage().refCount());
        
        Tensor contiguous = transposed.contiguous();
        
        assertNotEquals("Contiguous should create new storage", 
            original.storage().gpuBuffer().bufferId(),
            contiguous.storage().gpuBuffer().bufferId());
        
        original.close();
        transposed.close();
        contiguous.close();
        
        assertEquals(0, mockBackend.aliveBufferCount());
    }

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class MockComputeBackend implements ComputeBackend {
        private final MockComputeQueue computeQueue = new MockComputeQueue();
        private final MockComputeQueue transferQueue = new MockComputeQueue();
        private volatile boolean closed = false;
        private final java.util.concurrent.atomic.AtomicLong aliveBufferCount = new java.util.concurrent.atomic.AtomicLong(0);

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
            aliveBufferCount.incrementAndGet();
            return new MockGpuBuffer(sizeInBytes, aliveBufferCount);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            aliveBufferCount.incrementAndGet();
            return new MockGpuBuffer(sizeInBytes, aliveBufferCount);
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
        }

        long aliveBufferCount() {
            return aliveBufferCount.get();
        }
    }

    private static class MockComputeQueue implements ComputeQueue {
        @Override
        public QueueType getQueueType() {
            return QueueType.TRANSFER;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
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
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final long sizeInBytes;
        private volatile boolean closed = false;
        private final java.util.concurrent.atomic.AtomicLong aliveCounter;

        MockGpuBuffer(long sizeInBytes, java.util.concurrent.atomic.AtomicLong aliveCounter) {
            this.sizeInBytes = sizeInBytes;
            this.aliveCounter = aliveCounter;
        }

        @Override
        public long sizeInBytes() {
            return sizeInBytes;
        }

        @Override
        public long bufferId() {
            return System.identityHashCode(this);
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                aliveCounter.decrementAndGet();
            }
        }

        boolean isClosed() {
            return closed;
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        private volatile boolean complete = true;
        private volatile boolean closed = false;

        @Override
        public void awaitCompletion() throws InterruptedException {
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
