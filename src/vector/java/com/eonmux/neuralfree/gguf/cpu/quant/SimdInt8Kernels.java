package com.eonmux.neuralfree.gguf.cpu.quant;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The Vector API int8 primitives of the fused CPU paths ({@link CpuInt8Kernels}): the exact int8 dot product of the
 * quantized attention scores, the fused dequantizing accumulation of the attention values and the fused weighted
 * accumulation of the expert outputs.
 *
 * <p>The dot widens 16-byte vectors into eight int lanes ({@link SimdI8#mulAdd}) and sums the lanes horizontally, so
 * every lane product and partial sum is the exact integer of the scalar loop in any lane order. The accumulations
 * widen eight signed bytes to float lanes and apply the same {@code (code * scale) * weight} and
 * {@code weight * y} products lane by lane as {@link ScalarInt8Kernels}, so the two implementations agree bit for
 * bit. Instances are immutable and thread-safe.</p>
 */
public final class SimdInt8Kernels implements CpuInt8Kernels {

    /** The engine's single instance. */
    public static final SimdInt8Kernels INSTANCE = new SimdInt8Kernels();

    /** 8 byte lanes, the source of one widening conversion. */
    private static final VectorSpecies<Byte> B8 = ByteVector.SPECIES_64;
    /** 8 float lanes, the destination of one widening conversion and of one accumulation step. */
    private static final VectorSpecies<Float> F8 = SimdFp.F8;

    private SimdInt8Kernels() {
    }

    @Override
    public int dot(byte[] a, int aOff, byte[] b, int bOff, int n) {
        IntVector acc = IntVector.zero(SimdI8.I8);
        int i = 0;
        for (; i + 32 <= n; i += 32) {
            acc = SimdI8.mulAdd(ByteVector.fromArray(SimdI8.B16, a, aOff + i),
                    ByteVector.fromArray(SimdI8.B16, b, bOff + i), acc, SimdI8.I8);
            acc = SimdI8.mulAdd(ByteVector.fromArray(SimdI8.B16, a, aOff + i + 16),
                    ByteVector.fromArray(SimdI8.B16, b, bOff + i + 16), acc, SimdI8.I8);
        }
        int sum = SimdI8.sum(acc);
        for (; i < n; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Override
    public void accumulateScaled(byte[] codes, int cOff, float scale, float weight, float[] acc, int accOff, int n) {
        FloatVector scaleV = FloatVector.broadcast(F8, scale);
        FloatVector weightV = FloatVector.broadcast(F8, weight);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            FloatVector code = asFloat(ByteVector.fromArray(B8, codes, cOff + i));
            FloatVector a = FloatVector.fromArray(F8, acc, accOff + i);
            code.mul(scaleV).mul(weightV).add(a).intoArray(acc, accOff + i);
        }
        for (; i < n; i++) {
            acc[accOff + i] += weight * (codes[cOff + i] * scale);
        }
    }

    @Override
    public void accumulateWeighted(float[] y, int yOff, float weight, float[] acc, int accOff, int n) {
        FloatVector weightV = FloatVector.broadcast(F8, weight);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            FloatVector a = FloatVector.fromArray(F8, acc, accOff + i);
            FloatVector.fromArray(F8, y, yOff + i).mul(weightV).add(a).intoArray(acc, accOff + i);
        }
        for (; i < n; i++) {
            acc[accOff + i] += weight * y[yOff + i];
        }
    }

    /** The eight signed bytes of {@code v} as float lanes. */
    private static FloatVector asFloat(ByteVector v) {
        IntVector widened = (IntVector) v.convertShape(VectorOperators.B2I, SimdI8.I8, 0);
        return (FloatVector) widened.convertShape(VectorOperators.I2F, F8, 0);
    }

    @Override
    public String describe() {
        return "vector/" + SimdI8.I8.vectorBitSize() + "b (jdk.incubator.vector)";
    }

    @Override
    public String toString() {
        return "SimdInt8Kernels{" + describe() + '}';
    }
}
