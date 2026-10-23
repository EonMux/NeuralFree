package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.autograd.ComputationGraph;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * Host reads of a tensor's values as {@code float[]} whatever its dtype: the one reader shared by the state dict and
 * by every caller that exchanges or averages tensor values on the host.
 */
public final class TensorValues {

    private TensorValues() {
    }

    /**
     * The row-major values of {@code tensor} as floats: a FLOAT32 tensor's own values; FLOAT16, BFLOAT16 and FLOAT64
     * values converted by {@link Tensor#to(DType)} (FLOAT64 rounded to the nearest float); integer values rounded to
     * the nearest float; booleans as {@code 0} or {@code 1}. Values cross through the host and nothing is recorded.
     */
    public static float[] toFloats(Tensor tensor) {
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        DType dtype = tensor.dtype();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (dtype == DType.FLOAT32) {
                return tensor.toFloatArray();
            }
            if (dtype.isFloatingPoint()) {
                try (Tensor floats = tensor.to(DType.FLOAT32)) {
                    return floats.toFloatArray();
                }
            }
        }
        ByteBuffer bytes = tensor.toByteBuffer();
        float[] values = new float[Math.toIntExact(tensor.numElements())];
        for (int i = 0; i < values.length; i++) {
            values[i] = integerValue(bytes, dtype, i);
        }
        return values;
    }

    /** Element {@code i} of the native-order, row-major {@code bytes} of an integer or BOOL tensor. */
    private static float integerValue(ByteBuffer bytes, DType dtype, int i) {
        switch (dtype) {
            case BOOL:
                return bytes.get(i) != 0 ? 1f : 0f;
            case INT8:
                return bytes.get(i);
            case INT16:
                return bytes.getShort(i * Short.BYTES);
            case INT32:
                return bytes.getInt(i * Integer.BYTES);
            case INT64:
                return bytes.getLong(i * Long.BYTES);
            default:
                throw new IllegalStateException("Not an integer or boolean dtype: " + dtype);
        }
    }
}
