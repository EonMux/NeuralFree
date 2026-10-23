package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.*;

public class ActivationsTest {

    private static final float EPSILON = 1e-5f;
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
    }

    @Test
    public void testReLUCreation() {
        ReLU relu = new ReLU();
        assertNotNull(relu);
    }

    @Test
    public void testReLUForwardPositiveValues() {
        ReLU relu = new ReLU();
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = relu.forward(input);
        
        float[] result = output.toFloatArray();
        assertArrayEquals(data, result, EPSILON);
    }

    @Test
    public void testReLUForwardNegativeValues() {
        ReLU relu = new ReLU();
        float[] data = {-1.0f, -2.0f, -3.0f, -4.0f};
        float[] expected = {0.0f, 0.0f, 0.0f, 0.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = relu.forward(input);
        
        assertArrayEquals(expected, output.toFloatArray(), EPSILON);
    }

    @Test
    public void testReLUForwardMixedValues() {
        ReLU relu = new ReLU();
        float[] data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float[] expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = relu.forward(input);
        
        assertArrayEquals(expected, output.toFloatArray(), EPSILON);
    }

    @Test
    public void testReLUToString() {
        ReLU relu = new ReLU();
        assertTrue(relu.toString().contains("ReLU"));
    }

    @Test
    public void testGELUCreation() {
        GELU gelu = new GELU();
        assertNotNull(gelu);
    }

    @Test
    public void testGELUForwardZero() {
        GELU gelu = new GELU();
        float[] data = {0.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = gelu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertEquals(0.0f, result, EPSILON);
    }

    @Test
    public void testGELUForwardPositive() {
        GELU gelu = new GELU();
        float[] data = {1.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = gelu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertTrue(result > 0.0f);
        assertTrue(result < 1.0f);
    }

    @Test
    public void testGELUForwardNegative() {
        GELU gelu = new GELU();
        float[] data = {-1.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = gelu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertTrue(result < 0.0f);
    }

    @Test
    public void testGELUToString() {
        GELU gelu = new GELU();
        assertTrue(gelu.toString().contains("GELU"));
    }

    @Test
    public void testSiLUCreation() {
        SiLU silu = new SiLU();
        assertNotNull(silu);
    }

    @Test
    public void testSiLUForwardZero() {
        SiLU silu = new SiLU();
        float[] data = {0.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = silu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertEquals(0.0f, result, EPSILON);
    }

    @Test
    public void testSiLUForwardPositive() {
        SiLU silu = new SiLU();
        float[] data = {2.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = silu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertTrue(result > 0.0f);
        assertTrue(result < 2.0f);
    }

    @Test
    public void testSiLUToString() {
        SiLU silu = new SiLU();
        assertTrue(silu.toString().contains("SiLU"));
    }

    @Test
    public void testSigmoidCreation() {
        Sigmoid sigmoid = new Sigmoid();
        assertNotNull(sigmoid);
    }

    @Test
    public void testSigmoidForwardZero() {
        Sigmoid sigmoid = new Sigmoid();
        float[] data = {0.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = sigmoid.forward(input);
        
        float result = output.toFloatArray()[0];
        assertEquals(0.5f, result, EPSILON);
    }

    @Test
    public void testSigmoidForwardRange() {
        Sigmoid sigmoid = new Sigmoid();
        float[] data = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = sigmoid.forward(input);
        
        float[] result = output.toFloatArray();
        for (float v : result) {
            assertTrue(v > 0.0f);
            assertTrue(v < 1.0f);
        }
    }

    @Test
    public void testSigmoidToString() {
        Sigmoid sigmoid = new Sigmoid();
        assertTrue(sigmoid.toString().contains("Sigmoid"));
    }

    @Test
    public void testTanhCreation() {
        Tanh tanh = new Tanh();
        assertNotNull(tanh);
    }

    @Test
    public void testTanhForwardZero() {
        Tanh tanh = new Tanh();
        float[] data = {0.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = tanh.forward(input);
        
        float result = output.toFloatArray()[0];
        assertEquals(0.0f, result, EPSILON);
    }

    @Test
    public void testTanhForwardRange() {
        Tanh tanh = new Tanh();
        float[] data = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = tanh.forward(input);
        
        float[] result = output.toFloatArray();
        for (float v : result) {
            assertTrue(v >= -1.0f);
            assertTrue(v <= 1.0f);
        }
    }

    @Test
    public void testTanhToString() {
        Tanh tanh = new Tanh();
        assertTrue(tanh.toString().contains("Tanh"));
    }

    @Test
    public void testSoftmaxCreation() {
        Softmax softmax = new Softmax();
        assertNotNull(softmax);
        assertEquals(-1, softmax.dim());
    }

    @Test
    public void testSoftmaxCreationWithDim() {
        Softmax softmax = new Softmax(1);
        assertEquals(1, softmax.dim());
    }

    @Test
    public void testSoftmaxForwardSumToOne() {
        Softmax softmax = new Softmax(-1);
        float[] data = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = softmax.forward(input);
        
        float[] result = output.toFloatArray();
        float sum = 0.0f;
        for (float v : result) {
            sum += v;
        }
        assertEquals(1.0f, sum, EPSILON);
    }

    @Test
    public void testSoftmaxForwardPositiveValues() {
        Softmax softmax = new Softmax(-1);
        float[] data = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = softmax.forward(input);
        
        float[] result = output.toFloatArray();
        for (float v : result) {
            assertTrue(v > 0.0f);
        }
    }

    @Test
    public void testSoftmax2D() {
        Softmax softmax = new Softmax(1);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(2, 3));
        Tensor output = softmax.forward(input);
        
        assertEquals(Shape.of(2, 3), output.shape());
        
        float[] result = output.toFloatArray();
        float sum1 = result[0] + result[1] + result[2];
        float sum2 = result[3] + result[4] + result[5];
        assertEquals(1.0f, sum1, EPSILON);
        assertEquals(1.0f, sum2, EPSILON);
    }

    @Test
    public void testSoftmaxToString() {
        Softmax softmax = new Softmax(1);
        assertTrue(softmax.toString().contains("Softmax"));
        assertTrue(softmax.toString().contains("dim=1"));
    }

    @Test
    public void testLogSoftmaxCreation() {
        LogSoftmax logSoftmax = new LogSoftmax();
        assertNotNull(logSoftmax);
        assertEquals(-1, logSoftmax.dim());
    }

    @Test
    public void testLogSoftmaxForwardExpSum() {
        LogSoftmax logSoftmax = new LogSoftmax(-1);
        float[] data = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = logSoftmax.forward(input);
        
        float[] result = output.toFloatArray();
        float sum = 0.0f;
        for (float v : result) {
            sum += (float) Math.exp(v);
        }
        assertEquals(1.0f, sum, EPSILON);
    }

    @Test
    public void testLogSoftmax2D() {
        LogSoftmax logSoftmax = new LogSoftmax(1);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(2, 3));
        Tensor output = logSoftmax.forward(input);
        
        assertEquals(Shape.of(2, 3), output.shape());
    }

    @Test
    public void testLogSoftmaxToString() {
        LogSoftmax logSoftmax = new LogSoftmax(1);
        assertTrue(logSoftmax.toString().contains("LogSoftmax"));
    }

    @Test
    public void testLeakyReLUCreation() {
        LeakyReLU leakyRelu = new LeakyReLU();
        assertNotNull(leakyRelu);
        assertEquals(0.01f, leakyRelu.negativeSlope(), EPSILON);
    }

    @Test
    public void testLeakyReLUCreationWithSlope() {
        LeakyReLU leakyRelu = new LeakyReLU(0.2f);
        assertEquals(0.2f, leakyRelu.negativeSlope(), EPSILON);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLeakyReLUInvalidSlope() {
        new LeakyReLU(-0.1f);
    }

    @Test
    public void testLeakyReLUForwardPositive() {
        LeakyReLU leakyRelu = new LeakyReLU(0.1f);
        float[] data = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = leakyRelu.forward(input);
        
        assertArrayEquals(data, output.toFloatArray(), EPSILON);
    }

    @Test
    public void testLeakyReLUForwardNegative() {
        LeakyReLU leakyRelu = new LeakyReLU(0.1f);
        float[] data = {-1.0f, -2.0f, -3.0f};
        float[] expected = {-0.1f, -0.2f, -0.3f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = leakyRelu.forward(input);
        
        assertArrayEquals(expected, output.toFloatArray(), EPSILON);
    }

    @Test
    public void testLeakyReLUToString() {
        LeakyReLU leakyRelu = new LeakyReLU(0.2f);
        assertTrue(leakyRelu.toString().contains("LeakyReLU"));
        assertTrue(leakyRelu.toString().contains("0.2"));
    }

    @Test
    public void testELUCreation() {
        ELU elu = new ELU();
        assertNotNull(elu);
        assertEquals(1.0f, elu.alpha(), EPSILON);
    }

    @Test
    public void testELUCreationWithAlpha() {
        ELU elu = new ELU(0.5f);
        assertEquals(0.5f, elu.alpha(), EPSILON);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testELUInvalidAlpha() {
        new ELU(0.0f);
    }

    @Test
    public void testELUForwardPositive() {
        ELU elu = new ELU(1.0f);
        float[] data = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = elu.forward(input);
        
        assertArrayEquals(data, output.toFloatArray(), EPSILON);
    }

    @Test
    public void testELUForwardNegative() {
        ELU elu = new ELU(1.0f);
        float[] data = {-1.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = elu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertTrue(result < 0.0f);
        assertTrue(result > -1.0f);
    }

    @Test
    public void testELUToString() {
        ELU elu = new ELU(0.5f);
        assertTrue(elu.toString().contains("ELU"));
        assertTrue(elu.toString().contains("0.5"));
    }

    @Test
    public void testMishCreation() {
        Mish mish = new Mish();
        assertNotNull(mish);
    }

    @Test
    public void testMishForwardZero() {
        Mish mish = new Mish();
        float[] data = {0.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = mish.forward(input);
        
        float result = output.toFloatArray()[0];
        assertEquals(0.0f, result, EPSILON);
    }

    @Test
    public void testMishForwardPositive() {
        Mish mish = new Mish();
        float[] data = {1.0f};
        Tensor input = Tensor.fromArray(backend, data);
        Tensor output = mish.forward(input);
        
        float result = output.toFloatArray()[0];
        assertTrue(result > 0.0f);
        assertTrue(result < 1.0f);
    }

    @Test
    public void testMishToString() {
        Mish mish = new Mish();
        assertTrue(mish.toString().contains("Mish"));
    }

    @Test
    public void testGLUCreation() {
        GLU glu = new GLU();
        assertNotNull(glu);
        assertEquals(-1, glu.dim());
    }

    @Test
    public void testGLUCreationWithDim() {
        GLU glu = new GLU(1);
        assertEquals(1, glu.dim());
    }

    @Test
    public void testGLUForwardHalvesDimension() {
        GLU glu = new GLU(-1);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(4));
        Tensor output = glu.forward(input);
        
        assertEquals(Shape.of(2), output.shape());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGLUOddDimension() {
        GLU glu = new GLU(-1);
        float[] data = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(3));
        glu.forward(input);
    }

    @Test
    public void testGLUForwardValues() {
        GLU glu = new GLU(-1);
        float[] data = {2.0f, 0.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(2));
        Tensor output = glu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertEquals(1.0f, result, EPSILON);
    }

    @Test
    public void testGLUToString() {
        GLU glu = new GLU(1);
        assertTrue(glu.toString().contains("GLU"));
    }

    @Test
    public void testGeGLUCreation() {
        GeGLU geglu = new GeGLU();
        assertNotNull(geglu);
        assertEquals(-1, geglu.dim());
    }

    @Test
    public void testGeGLUCreationWithDim() {
        GeGLU geglu = new GeGLU(1);
        assertEquals(1, geglu.dim());
    }

    @Test
    public void testGeGLUForwardHalvesDimension() {
        GeGLU geglu = new GeGLU(-1);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(4));
        Tensor output = geglu.forward(input);
        
        assertEquals(Shape.of(2), output.shape());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGeGLUOddDimension() {
        GeGLU geglu = new GeGLU(-1);
        float[] data = {1.0f, 2.0f, 3.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(3));
        geglu.forward(input);
    }

    @Test
    public void testGeGLUForwardValues() {
        GeGLU geglu = new GeGLU(-1);
        float[] data = {2.0f, 1.0f};
        Tensor input = Tensor.fromArray(backend, data).reshape(Shape.of(2));
        Tensor output = geglu.forward(input);
        
        float result = output.toFloatArray()[0];
        assertTrue(result > 0.0f);
    }

    @Test
    public void testGeGLUToString() {
        GeGLU geglu = new GeGLU(1);
        assertTrue(geglu.toString().contains("GeGLU"));
    }

    @Test
    public void testActivationPreservesShape() {
        ReLU relu = new ReLU();
        Shape shape = Shape.of(2, 3, 4);
        float[] data = new float[(int) shape.numElements()];
        Tensor input = Tensor.fromArray(backend, data).reshape(shape);
        Tensor output = relu.forward(input);
        
        assertEquals(shape, output.shape());
    }

    @Test(expected = NullPointerException.class)
    public void testReLUNullInput() {
        ReLU relu = new ReLU();
        relu.forward(null);
    }

    @Test(expected = NullPointerException.class)
    public void testSigmoidNullInput() {
        Sigmoid sigmoid = new Sigmoid();
        sigmoid.forward(null);
    }

    @Test(expected = NullPointerException.class)
    public void testSoftmaxNullInput() {
        Softmax softmax = new Softmax();
        softmax.forward(null);
    }

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class MockComputeBackend implements ComputeBackend {
        private final MockComputeQueue computeQueue = new MockComputeQueue();
        private final MockComputeQueue transferQueue = new MockComputeQueue();
        private volatile boolean closed = false;

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
            return new MockGpuBuffer(sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return new MockGpuBuffer(sizeInBytes);
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
    }

    private static class MockComputeQueue implements ComputeQueue {
        @Override
        public QueueType getQueueType() {
            return QueueType.TRANSFER;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            if (dstBuffer instanceof MockGpuBuffer) {
                ((MockGpuBuffer) dstBuffer).writeBytes(srcData, (int) dstOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            if (dstBuffer instanceof MockGpuBuffer) {
                ((MockGpuBuffer) dstBuffer).writeFloats(srcData, (int) dstOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            if (srcBuffer instanceof MockGpuBuffer) {
                ((MockGpuBuffer) srcBuffer).readBytes(dstData, (int) srcOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
            if (srcBuffer instanceof MockGpuBuffer) {
                ((MockGpuBuffer) srcBuffer).readFloats(dstData, (int) srcOffset);
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
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final byte[] data;
        private final long sizeInBytes;
        private volatile boolean closed = false;

        MockGpuBuffer(long sizeInBytes) {
            this.sizeInBytes = sizeInBytes;
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
            return System.identityHashCode(this);
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
        private final String name;
        private volatile boolean closed = false;

        MockKernelHandle(String name) {
            this.name = name;
        }

        @Override
        public String kernelName() {
            return name;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
