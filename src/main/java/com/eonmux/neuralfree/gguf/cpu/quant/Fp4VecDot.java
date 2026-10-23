package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.QuantTables;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.dotI8;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_E;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_MXFP4;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.e8m0ToFp32Half;

/**
 * CPU vec-dot of MXFP4 weights with Q8_0 activations, {@code ggml_vec_dot_mxfp4_q8_0_generic}
 * (llama.cpp ggml/src/ggml-cpu/quants.c:298-327).
 * <p>
 * A block holds one E8M0 exponent and 32 E2M1 codes in split nibble order (byte {@code j}: element
 * {@code j} low, element {@code j + 16} high). The codes are mapped through the doubled codebook
 * {@code kvalues_mxfp4} (llama.cpp ggml-common.h, {@link QuantTables#kvalueFp4(int)}) into the per-thread block
 * buffer and reduced against the activation quants; the halved exponent scale
 * ({@code GGML_E8M0_TO_FP32_HALF}) undoes the doubling.
 * <p>
 * Immutable, stateless and thread-safe.
 */
public final class Fp4VecDot implements CpuVecDot {

    /** The MXFP4 vec-dot. */
    public static final Fp4VecDot MXFP4 = new Fp4VecDot();

    private static final Set<GGMLType> SUPPORTED = Collections.unmodifiableSet(EnumSet.of(GGMLType.MXFP4));

    /** {@code kvalues_mxfp4}: doubled E2M1 values of the 16 codes. */
    private static final byte[] KVALUES = buildCodebook();

    private Fp4VecDot() {
    }

    private static byte[] buildCodebook() {
        byte[] kv = new byte[16];
        for (int i = 0; i < kv.length; i++) {
            kv[i] = (byte) QuantTables.kvalueFp4(i);
        }
        return kv;
    }

    /** True for the single type this class handles, MXFP4. */
    public static boolean supports(GGMLType type) {
        return type == GGMLType.MXFP4;
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return SUPPORTED;
    }

    /**
     * @throws IllegalArgumentException when {@code type} is not MXFP4
     */
    public static Fp4VecDot forType(GGMLType type) {
        if (!supports(type)) {
            throw new IllegalArgumentException("Fp4VecDot handles MXFP4 weights only, not " + type);
        }
        return MXFP4;
    }

    /** The weight type this vec-dot reads, MXFP4. */
    public GGMLType weightType() {
        return GGMLType.MXFP4;
    }

    /** The activation type this vec-dot consumes, Q8_0. */
    public GGMLType activationType() {
        return GGMLType.Q8_0;
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(w.rowBuffer(row), w.rowOffset(row), x, Q8VecDot.scratch().quants);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        final byte[] aux = Q8VecDot.scratch().quants;
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, y, yOff, (b, off) -> dot(b, off, x, aux));
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.MXFP4);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    /**
     * quants.c:314-325: {@code d = d_y * E8M0_HALF(e)},
     * {@code sumi1 = sum_j y.qs[j] * kv[qs[j] & 0xf]}, {@code sumi2 = sum_j y.qs[j + 16] * kv[qs[j] >> 4]},
     * {@code sumf += d * (sumi1 + sumi2)}.
     */
    private static float dot(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] kv = KVALUES;
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int half = QK_MXFP4 / 2;
        final int nb = x.n / QK_MXFP4;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += MXFP4_BYTES) {
            final int qs = off + MXFP4_QS;
            final long lo = b.getLong(qs);
            final long hi = b.getLong(qs + 8);
            for (int j = 0; j < 8; j++) {
                final int q0 = (int) (lo >>> (8 * j)) & 0xFF;
                final int q1 = (int) (hi >>> (8 * j)) & 0xFF;
                aux[j] = kv[q0 & 0x0F];
                aux[j + half] = kv[q0 >>> 4];
                aux[j + 8] = kv[q1 & 0x0F];
                aux[j + 8 + half] = kv[q1 >>> 4];
            }
            final int sumi = dotI8(aux, 0, yq, ib * QK_MXFP4, QK_MXFP4);
            final float d = yd[ib] * e8m0ToFp32Half(b.get(off + MXFP4_E));
            sumf += d * sumi;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "Fp4VecDot{MXFP4 x Q8_0}";
    }
}
