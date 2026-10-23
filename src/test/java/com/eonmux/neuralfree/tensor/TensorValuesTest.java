package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class TensorValuesTest {

    private static final float[] VALUES = {1.5f, -2f, 0f, 3f};

    @Test
    public void floatingPointTensorsAreReadInEveryDType() {
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor floats = Tensor.fromArray(host, VALUES)) {
            assertArrayEquals(VALUES, TensorValues.toFloats(floats), 0f);
            for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16, DType.FLOAT64}) {
                try (Tensor converted = floats.to(dtype)) {
                    assertEquals(dtype, converted.dtype());
                    assertArrayEquals(dtype.toString(), VALUES, TensorValues.toFloats(converted), 0f);
                }
            }
        } finally {
            TrainGpu.release(host);
        }
    }

    /** A half-precision tensor that requires grad is read without recording a cast. */
    @Test
    public void readingRecordsNothing() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor floats = Tensor.fromArray(host, VALUES);
             Tensor converted = floats.to(DType.FLOAT16);
             Tensor half = converted.withRequiresGrad(true)) {
            int graphBefore = ComputationGraph.current().size();
            assertArrayEquals(VALUES, TensorValues.toFloats(half), 0f);
            assertEquals("no node recorded", graphBefore, ComputationGraph.current().size());
        } finally {
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    @Test
    public void integerAndBooleanTensorsAreReadAsFloats() {
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor int32 = Tensor.fromArray(host, new int[]{3, -7});
             Tensor int8 = Tensor.fromArray(host, new byte[]{-5, 4});
             Tensor int64 = int64Vector(host, 5L, 1L << 40);
             Tensor floats = Tensor.fromArray(host, new float[]{0f, 2f, 0f});
             Tensor bool = floats.to(DType.BOOL)) {
            assertArrayEquals(new float[]{3f, -7f}, TensorValues.toFloats(int32), 0f);
            assertArrayEquals(new float[]{-5f, 4f}, TensorValues.toFloats(int8), 0f);
            assertArrayEquals(new float[]{5f, (float) (1L << 40)}, TensorValues.toFloats(int64), 0f);
            assertEquals(DType.BOOL, bool.dtype());
            assertArrayEquals(new float[]{0f, 1f, 0f}, TensorValues.toFloats(bool), 0f);
        } finally {
            TrainGpu.release(host);
        }
    }

    @Test(expected = NullPointerException.class)
    public void nullTensorIsRejected() {
        TensorValues.toFloats(null);
    }

    @Test
    public void gradModeIsRestored() {
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor floats = Tensor.fromArray(host, VALUES);
             Tensor half = floats.to(DType.FLOAT16);
             ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            TensorValues.toFloats(half);
            assertFalse("the caller's no-grad mode is kept", ComputationGraph.isGradEnabled());
        } finally {
            TrainGpu.release(host);
        }
    }

    private static Tensor int64Vector(ComputeBackend backend, long... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (int i = 0; i < values.length; i++) {
            bytes.putLong(i * Long.BYTES, values[i]);
        }
        return Tensor.fromStorage(backend, Shape.vector(values.length), DType.INT64,
                TensorStorage.fromHost(backend, bytes, DType.INT64));
    }
}
