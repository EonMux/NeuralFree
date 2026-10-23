package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class TensorOpsTest {

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
    public void testAddSameShape() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f});

        Tensor result = TensorOps.add(a, b);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{5.0f, 7.0f, 9.0f}, data, 1e-6f);
    }

    @Test
    public void testAddBroadcast() {
        Tensor a = Tensor.fromArray(backend, new float[][]{{1.0f, 2.0f}, {3.0f, 4.0f}});
        Tensor b = Tensor.fromArray(backend, new float[]{10.0f, 20.0f});

        Tensor result = TensorOps.add(a, b);

        assertEquals(Shape.matrix(2, 2), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{11.0f, 22.0f, 13.0f, 24.0f}, data, 1e-6f);
    }

    @Test
    public void testSubSameShape() {
        Tensor a = Tensor.fromArray(backend, new float[]{10.0f, 8.0f, 6.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});

        Tensor result = TensorOps.sub(a, b);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{9.0f, 6.0f, 3.0f}, data, 1e-6f);
    }

    @Test
    public void testMulSameShape() {
        Tensor a = Tensor.fromArray(backend, new float[]{2.0f, 3.0f, 4.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{5.0f, 6.0f, 7.0f});

        Tensor result = TensorOps.mul(a, b);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{10.0f, 18.0f, 28.0f}, data, 1e-6f);
    }

    @Test
    public void testDivSameShape() {
        Tensor a = Tensor.fromArray(backend, new float[]{10.0f, 20.0f, 30.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{2.0f, 5.0f, 6.0f});

        Tensor result = TensorOps.div(a, b);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{5.0f, 4.0f, 5.0f}, data, 1e-6f);
    }

    @Test
    public void testPow() {
        Tensor a = Tensor.fromArray(backend, new float[]{2.0f, 3.0f, 4.0f});

        Tensor result = TensorOps.pow(a, 2.0f);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{4.0f, 9.0f, 16.0f}, data, 1e-6f);
    }

    @Test
    public void testNeg() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, -2.0f, 3.0f});

        Tensor result = TensorOps.neg(a);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{-1.0f, 2.0f, -3.0f}, data, 1e-6f);
    }

    @Test
    public void testAbs() {
        Tensor a = Tensor.fromArray(backend, new float[]{-1.0f, 2.0f, -3.0f});

        Tensor result = TensorOps.abs(a);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, data, 1e-6f);
    }

    @Test
    public void testSqrt() {
        Tensor a = Tensor.fromArray(backend, new float[]{4.0f, 9.0f, 16.0f});

        Tensor result = TensorOps.sqrt(a);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{2.0f, 3.0f, 4.0f}, data, 1e-6f);
    }

    @Test
    public void testExp() {
        Tensor a = Tensor.fromArray(backend, new float[]{0.0f, 1.0f, 2.0f});

        Tensor result = TensorOps.exp(a);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertEquals(1.0f, data[0], 1e-6f);
        assertEquals((float) Math.E, data[1], 1e-6f);
        assertEquals((float) Math.E * (float) Math.E, data[2], 1e-6f);
    }

    @Test
    public void testLog() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, (float) Math.E, (float) Math.E * (float) Math.E});

        Tensor result = TensorOps.log(a);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertEquals(0.0f, data[0], 1e-6f);
        assertEquals(1.0f, data[1], 1e-6f);
        assertEquals(2.0f, data[2], 1e-6f);
    }

    @Test
    public void testClamp() {
        Tensor a = Tensor.fromArray(backend, new float[]{-1.0f, 2.0f, 5.0f, 10.0f});

        Tensor result = TensorOps.clamp(a, 0.0f, 5.0f);

        assertEquals(Shape.vector(4), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{0.0f, 2.0f, 5.0f, 5.0f}, data, 1e-6f);
    }

    @Test
    public void testAddScalar() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});

        Tensor result = TensorOps.add(a, 10.0f);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{11.0f, 12.0f, 13.0f}, data, 1e-6f);
    }

    @Test
    public void testMulScalar() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});

        Tensor result = TensorOps.mul(a, 5.0f);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{5.0f, 10.0f, 15.0f}, data, 1e-6f);
    }

    @Test
    public void testMatmul() {
        float[][] aData = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[][] bData = {{5.0f, 6.0f}, {7.0f, 8.0f}};

        Tensor a = Tensor.fromArray(backend, aData);
        Tensor b = Tensor.fromArray(backend, bData);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.matrix(2, 2), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{19.0f, 22.0f, 43.0f, 50.0f}, data, 1e-5f);
    }

    @Test
    public void testBmm() {
        float[] aData = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] bData = {1, 2, 3, 4, 5, 6, 7, 8};

        Tensor a = Tensor.fromArray(backend, aData).reshape(2, 2, 2);
        Tensor b = Tensor.fromArray(backend, bData).reshape(2, 2, 2);

        Tensor result = TensorOps.bmm(a, b);

        assertEquals(Shape.of(2, 2, 2), result.shape());
    }

    @Test
    public void testMatmul3DAnd2D() {
        float[] aData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        float[][] bData = {{1, 2}, {3, 4}};

        Tensor a = Tensor.fromArray(backend, aData).reshape(2, 3, 2);
        Tensor b = Tensor.fromArray(backend, bData);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.of(2, 3, 2), result.shape());
        
        float[] expected = {
            7, 10, 15, 22, 23, 34,
            31, 46, 39, 58, 47, 70
        };
        float[] actual = result.toFloatArray();
        assertArrayEquals(expected, actual, 1e-5f);
    }

    @Test
    public void testMatmul2DAnd3D() {
        float[][] aData = {{1, 2}, {3, 4}};
        float[] bData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        Tensor a = Tensor.fromArray(backend, aData);
        Tensor b = Tensor.fromArray(backend, bData).reshape(2, 2, 3);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.of(2, 2, 3), result.shape());
        
        float[] expected = {
            9, 12, 15, 19, 26, 33,
            27, 30, 33, 61, 68, 75
        };
        float[] actual = result.toFloatArray();
        assertArrayEquals(expected, actual, 1e-5f);
    }

    @Test
    public void testMatmul4DAnd2D() {
        float[] aData = new float[24];
        for (int i = 0; i < 24; i++) aData[i] = i + 1;
        float[][] bData = {{1, 2}, {3, 4}};

        Tensor a = Tensor.fromArray(backend, aData).reshape(2, 2, 3, 2);
        Tensor b = Tensor.fromArray(backend, bData);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.of(2, 2, 3, 2), result.shape());
        
        float[] actual = result.toFloatArray();
        assertEquals(24, actual.length);
        assertEquals(7.0f, actual[0], 1e-5f);
        assertEquals(10.0f, actual[1], 1e-5f);
    }

    @Test
    public void testMatmul3DAnd3D() {
        float[] aData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        float[] bData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        Tensor a = Tensor.fromArray(backend, aData).reshape(2, 3, 2);
        Tensor b = Tensor.fromArray(backend, bData).reshape(2, 2, 3);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.of(2, 3, 3), result.shape());
    }

    @Test
    public void testMatmulBatchBroadcast() {
        float[] aData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        float[] bData = {1, 2, 3, 4};

        Tensor a = Tensor.fromArray(backend, aData).reshape(2, 3, 2);
        Tensor b = Tensor.fromArray(backend, bData).reshape(1, 2, 2);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.of(2, 3, 2), result.shape());
    }

    @Test
    public void testMatmulVectorAnd2D() {
        float[] aData = {1, 2, 3};
        float[][] bData = {{1, 2}, {3, 4}, {5, 6}};

        Tensor a = Tensor.fromArray(backend, aData);
        Tensor b = Tensor.fromArray(backend, bData);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.vector(2), result.shape());
        assertArrayEquals(new float[]{22, 28}, result.toFloatArray(), 1e-5f);
    }

    @Test
    public void testMatmul2DAndVector() {
        float[][] aData = {{1, 2, 3}, {4, 5, 6}};
        float[] bData = {1, 2, 3};

        Tensor a = Tensor.fromArray(backend, aData);
        Tensor b = Tensor.fromArray(backend, bData);

        Tensor result = TensorOps.matmul(a, b);

        assertEquals(Shape.vector(2), result.shape());
        assertArrayEquals(new float[]{14, 32}, result.toFloatArray(), 1e-5f);
    }

    @Test
    public void testMatmulDispatcherDeclinesWithoutTrainKernels() {
        Tensor a = Tensor.fromArray(backend, new float[]{1, 2, 3, 4, 5, 6}).reshape(2, 3);
        Tensor b = Tensor.fromArray(backend, new float[]{1, 2, 3, 4, 5, 6}).reshape(3, 2);
        Tensor c = Tensor.zeros(backend, Shape.matrix(2, 2));

        assertFalse(GpuMatmul.shouldUseGpu(a, b));
        assertNull(GpuMatmul.matmul(a, b));
        assertNull(GpuMatmul.bmm(a.reshape(1, 2, 3), b.reshape(1, 3, 2)));
        assertFalse(GpuMatmul.gemm(a, b, 1.0f, 0.0f, c));
        assertArrayEquals(new float[]{22, 28, 49, 64}, TensorOps.matmul(a, b).toFloatArray(), 1e-5f);
    }

    @Test
    public void testMatmulDispatcherRejectsInvalidShapesWithTheCpuMessage() {
        Tensor a = Tensor.fromArray(backend, new float[6]).reshape(2, 3);
        Tensor b = Tensor.fromArray(backend, new float[8]).reshape(4, 2);

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> GpuMatmul.matmul(a, b));
        assertEquals("Matrix dimensions incompatible for multiplication: A has 3 columns, B has 4 rows",
                e.getMessage());
    }

    @Test
    public void testSumAll() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f, 4.0f});

        Tensor result = TensorOps.sum(a);

        assertTrue(result.shape().isScalar());
    }

    @Test
    public void testSumAxis() {
        float[][] data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        Tensor result = TensorOps.sum(a, 0);

        assertEquals(Shape.vector(2), result.shape());
        float[] resultData = result.toFloatArray();
        assertArrayEquals(new float[]{4.0f, 6.0f}, resultData, 1e-6f);
    }

    @Test
    public void testMeanAll() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f, 4.0f});

        Tensor result = TensorOps.mean(a);

        assertTrue(result.shape().isScalar());
    }

    @Test
    public void testMeanAxis() {
        float[][] data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        Tensor result = TensorOps.mean(a, 1);

        assertEquals(Shape.vector(2), result.shape());
        float[] resultData = result.toFloatArray();
        assertArrayEquals(new float[]{1.5f, 3.5f}, resultData, 1e-6f);
    }

    @Test
    public void testMax() {
        float[][] data = {{1.0f, 5.0f}, {3.0f, 2.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        Tensor result = TensorOps.max(a, 0);

        assertEquals(Shape.vector(2), result.shape());
        float[] resultData = result.toFloatArray();
        assertArrayEquals(new float[]{3.0f, 5.0f}, resultData, 1e-6f);
    }

    @Test
    public void testMin() {
        float[][] data = {{1.0f, 5.0f}, {3.0f, 2.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        Tensor result = TensorOps.min(a, 1);

        assertEquals(Shape.vector(2), result.shape());
        float[] resultData = result.toFloatArray();
        assertArrayEquals(new float[]{1.0f, 2.0f}, resultData, 1e-6f);
    }

    @Test
    public void testArgmax() {
        float[][] data = {{1.0f, 5.0f, 3.0f}, {7.0f, 2.0f, 4.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        Tensor result = TensorOps.argmax(a, 1);

        assertEquals(Shape.vector(2), result.shape());
        assertEquals(DType.INT32, result.dtype());
    }

    @Test
    public void testArgmin() {
        float[][] data = {{1.0f, 5.0f, 3.0f}, {7.0f, 2.0f, 4.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        Tensor result = TensorOps.argmin(a, 1);

        assertEquals(Shape.vector(2), result.shape());
        assertEquals(DType.INT32, result.dtype());
    }

    @Test
    public void testVar() {
        float[][] data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        Tensor result = TensorOps.var(a, 0);

        assertEquals(Shape.vector(2), result.shape());
    }

    @Test
    public void testEq() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{1.0f, 5.0f, 3.0f});

        Tensor result = TensorOps.eq(a, b);

        assertEquals(Shape.vector(3), result.shape());
        assertEquals(DType.BOOL, result.dtype());
    }

    @Test
    public void testGt() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 5.0f, 3.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 4.0f});

        Tensor result = TensorOps.gt(a, b);

        assertEquals(Shape.vector(3), result.shape());
        assertEquals(DType.BOOL, result.dtype());
    }

    @Test
    public void testLt() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 5.0f, 3.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 4.0f});

        Tensor result = TensorOps.lt(a, b);

        assertEquals(Shape.vector(3), result.shape());
        assertEquals(DType.BOOL, result.dtype());
    }

    @Test
    public void testCat() {
        Tensor a = Tensor.fromArray(backend, new float[][]{{1.0f, 2.0f}});
        Tensor b = Tensor.fromArray(backend, new float[][]{{3.0f, 4.0f}});
        List<Tensor> tensors = Arrays.asList(a, b);

        Tensor result = TensorOps.cat(tensors, 0);

        assertEquals(Shape.matrix(2, 2), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, data, 1e-6f);
    }

    @Test
    public void testSplit() {
        float[][] data = {{1.0f, 2.0f, 3.0f, 4.0f}};
        Tensor a = Tensor.fromArray(backend, data);

        List<Tensor> result = TensorOps.split(a, 2, 1);

        assertEquals(2, result.size());
        assertEquals(Shape.matrix(1, 2), result.get(0).shape());
    }

    @Test
    public void testStack() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f});
        Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f});
        List<Tensor> tensors = Arrays.asList(a, b);

        Tensor result = TensorOps.stack(tensors, 0);

        assertEquals(Shape.of(2, 2), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, data, 1e-6f);
    }

    @Test
    public void testGather() {
        float[] inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        Tensor input = Tensor.fromArray(backend, inputData).reshape(2, 3);
        int[] indexData = {0, 2, 1, 0};
        Tensor index = Tensor.fromArray(backend, indexData).reshape(2, 2);

        Tensor result = TensorOps.gather(input, 1, index);

        assertEquals(Shape.matrix(2, 2), result.shape());
    }

    @Test
    public void testScatter() {
        float[] inputData = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        Tensor input = Tensor.fromArray(backend, inputData).reshape(2, 3);
        int[] indexData = {0, 2, 1, 0};
        Tensor index = Tensor.fromArray(backend, indexData).reshape(2, 2);
        float[] srcData = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor src = Tensor.fromArray(backend, srcData).reshape(2, 2);

        Tensor result = TensorOps.scatter(input, 1, index, src);

        assertEquals(Shape.matrix(2, 3), result.shape());
    }

    @Test
    public void testWhere() {
        float[] condData = {1.0f, 0.0f, 1.0f};
        float[] aData = {10.0f, 20.0f, 30.0f};
        float[] bData = {100.0f, 200.0f, 300.0f};

        Tensor condition = Tensor.fromArray(backend, condData);
        Tensor a = Tensor.fromArray(backend, aData);
        Tensor b = Tensor.fromArray(backend, bData);

        Tensor result = TensorOps.where(condition, a, b);

        assertEquals(Shape.vector(3), result.shape());
        float[] data = result.toFloatArray();
        assertArrayEquals(new float[]{10.0f, 200.0f, 30.0f}, data, 1e-6f);
    }

    @Test
    public void testAddInPlace() {
        Tensor dst = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});
        Tensor src = Tensor.fromArray(backend, new float[]{10.0f, 20.0f, 30.0f});

        TensorOps.addInPlace(dst, src, 1.0f);

        float[] data = dst.toFloatArray();
        assertArrayEquals(new float[]{11.0f, 22.0f, 33.0f}, data, 1e-6f);
    }

    @Test
    public void testMulInPlace() {
        Tensor dst = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});

        TensorOps.mulInPlace(dst, 2.0f);

        float[] data = dst.toFloatArray();
        assertArrayEquals(new float[]{2.0f, 4.0f, 6.0f}, data, 1e-6f);
    }

    @Test
    public void testZeroInPlace() {
        Tensor dst = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f});

        TensorOps.zeroInPlace(dst);

        float[] data = dst.toFloatArray();
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f}, data, 1e-6f);
    }

    @Test(expected = NullPointerException.class)
    public void testNullTensorThrows() {
        TensorOps.add(null, Tensor.fromArray(backend, new float[]{1}));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDifferentBackendsThrows() {
        MockComputeBackend otherBackend = new MockComputeBackend();
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f});
        Tensor b = Tensor.fromArray(otherBackend, new float[]{2.0f});

        try {
            TensorOps.add(a, b);
        } finally {
            otherBackend.close();
        }
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
