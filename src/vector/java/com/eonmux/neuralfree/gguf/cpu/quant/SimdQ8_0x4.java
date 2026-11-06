package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.repack.Q8_0x4;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q8_0 weights in the interleaved {@link Q8_0x4} layout with Q8_0 activations,
 * {@link Q8_0x4VecDot} with each 16-byte interleave slice widened to four int vectors (one per block) and
 * dotted against the matching four activation quants. The float combination
 * {@code sumi * (d_x * d_y)} per block follows {@link Q8VecDot#Q8_0}, so the kernel returns exactly what the
 * scalar repacked kernel returns. Immutable, stateless and thread-safe.
 */
public final class SimdQ8_0x4 implements CpuVecDot {

    /** The Q8_0x4 vec-dot. */
    public static final SimdQ8_0x4 INSTANCE = new SimdQ8_0x4();

    private SimdQ8_0x4() {
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(CpuVecDotRows.copyRow(w, row), 0, x, SimdI8.activations(x));
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return dot(row, off, x, SimdI8.activations(x));
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q8_0);
        x.requireCompatible(w, GGMLType.Q8_0);
        if (w.cols() % Q8_0x4.ELEMENTS != 0) {
            throw new IllegalArgumentException("the Q8_0x4 kernel reads rows of whole 128-element groups; "
                    + w + " has " + w.cols() + " columns");
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final int groups = x.n / Q8_0x4.ELEMENTS;
        float sumf = 0;
        for (int g = 0; g < groups; g++, off += Q8_0x4.BYTES) {
            final int qs = off + Q8_0x4.QS;
            final int xo = g * Q8_0x4.ELEMENTS;
            // one accumulator per block; the four widenings of a slice are unrolled because the widening of
            // part j is intrinsic only while the part is a compile-time constant
            IntVector a0 = IntVector.zero(SimdI8.I4);
            IntVector a1 = IntVector.zero(SimdI8.I4);
            IntVector a2 = IntVector.zero(SimdI8.I4);
            IntVector a3 = IntVector.zero(SimdI8.I4);
            for (int c = 0; c < 8; c++) {
                final int slice = qs + c * 16;
                a0 = a0.add(widen(b, slice, 0).mul(IntVector.fromArray(SimdI8.I4, yw, xo + c * 4)));
                a1 = a1.add(widen(b, slice, 1).mul(IntVector.fromArray(SimdI8.I4, yw, xo + 32 + c * 4)));
                a2 = a2.add(widen(b, slice, 2).mul(IntVector.fromArray(SimdI8.I4, yw, xo + 64 + c * 4)));
                a3 = a3.add(widen(b, slice, 3).mul(IntVector.fromArray(SimdI8.I4, yw, xo + 96 + c * 4)));
            }
            final int ib = g * Q8_0x4.BLOCKS;
            final int d = off + Q8_0x4.D;
            sumf += SimdI8.sum(a0) * (fp16At(b, d) * yd[ib]);
            sumf += SimdI8.sum(a1) * (fp16At(b, d + 2) * yd[ib + 1]);
            sumf += SimdI8.sum(a2) * (fp16At(b, d + 4) * yd[ib + 2]);
            sumf += SimdI8.sum(a3) * (fp16At(b, d + 6) * yd[ib + 3]);
        }
        return sumf;
    }

    /**
     * Block {@code j} of the 16-byte slice at {@code off} widened to four int lanes. The slice holds four bytes
     * of each block, so one 64-bit half of it is exactly one block's four quants: the load and the widening take
     * the species pair the other kernels use (a 64-bit byte vector widened to four int lanes), which the JIT
     * intrinsifies; a 128-bit byte vector converted to four int lanes falls back to the allocating template.
     */
    private static IntVector widen(byte[] b, int off, int block) {
        return SimdI8.widen(ByteVector.fromArray(SimdI8.B8, b, off + 4 * block), 0, SimdI8.I4);
    }

    @Override
    public String toString() {
        return "SimdQ8_0x4{Q8_0x4 x Q8_0}";
    }
}
