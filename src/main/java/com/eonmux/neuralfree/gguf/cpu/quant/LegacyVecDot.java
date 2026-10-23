package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.LOW_NIBBLES;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.dotI8;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.putLong;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.subtractBias;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of the legacy 32-element 4- and 5-bit weight types with 8-bit activations, ports of
 * llama.cpp ggml/src/ggml-cpu/quants.c:
 * <ul>
 *   <li>Q4_0 x Q8_0, {@code ggml_vec_dot_q4_0_q8_0_generic} (quants.c:225-259)</li>
 *   <li>Q4_1 x Q8_1, {@code ggml_vec_dot_q4_1_q8_1_generic} (quants.c:262-296)</li>
 *   <li>Q5_0 x Q8_0, {@code ggml_vec_dot_q5_0_q8_0_generic} (quants.c:365-406)</li>
 *   <li>Q5_1 x Q8_1, {@code ggml_vec_dot_q5_1_q8_1_generic} (quants.c:408-449)</li>
 * </ul>
 * Each block's 16 {@code qs} bytes are read as two little-endian longs; the split nibble layout (byte
 * {@code j} holds element {@code j} low and element {@code j + 16} high) is unpacked with 64-bit SWAR into
 * element order, the fifth bits of Q5_x are merged from a byte-spread table, and the 32 signed or unsigned
 * quants are reduced against the activation quants with {@link Q8VecDot#dotI8}. The integer block sums are
 * exactly those of ggml, and the float combination per block uses ggml's expression and evaluation order.
 * <p>
 * Instances are immutable, stateless and thread-safe.
 */
public final class LegacyVecDot implements CpuVecDot {

    public static final LegacyVecDot Q4_0 = new LegacyVecDot(GGMLType.Q4_0);
    public static final LegacyVecDot Q4_1 = new LegacyVecDot(GGMLType.Q4_1);
    public static final LegacyVecDot Q5_0 = new LegacyVecDot(GGMLType.Q5_0);
    public static final LegacyVecDot Q5_1 = new LegacyVecDot(GGMLType.Q5_1);

    private static final Set<GGMLType> SUPPORTED = Collections.unmodifiableSet(
            EnumSet.of(GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1));

    /** {@code 0x80 - 8} in every byte: maps nibbles {@code 0..15} to {@code -8..7}. */
    private static final long BIAS_8 = 0x7878787878787878L;
    /** {@code 0x80 - 16} in every byte: maps 5-bit values {@code 0..31} to {@code -16..15}. */
    private static final long BIAS_16 = 0x7070707070707070L;

    /**
     * {@code QH_SPREAD[g]} holds {@code 0x10} in byte {@code k} when bit {@code k} of {@code g} is set: eight
     * consecutive {@code qh} bits as the fifth bit of eight consecutive element bytes.
     */
    private static final long[] QH_SPREAD = buildQhSpread();

    private final GGMLType weightType;
    private final GGMLType activationType;

    private LegacyVecDot(GGMLType weightType) {
        this.weightType = weightType;
        this.activationType = CpuActivationQuantizer.activationType(weightType);
    }

    private static long[] buildQhSpread() {
        long[] t = new long[256];
        for (int g = 0; g < 256; g++) {
            long v = 0;
            for (int k = 0; k < 8; k++) {
                if (((g >>> k) & 1) != 0) {
                    v |= 0x10L << (8 * k);
                }
            }
            t[g] = v;
        }
        return t;
    }

    /** True for Q4_0, Q4_1, Q5_0 and Q5_1. */
    public static boolean supports(GGMLType type) {
        return type != null && SUPPORTED.contains(type);
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return SUPPORTED;
    }

    /**
     * @throws IllegalArgumentException when {@code type} is not one of the types {@link #supports} accepts
     */
    public static LegacyVecDot forType(GGMLType type) {
        if (type == GGMLType.Q4_0) {
            return Q4_0;
        }
        if (type == GGMLType.Q4_1) {
            return Q4_1;
        }
        if (type == GGMLType.Q5_0) {
            return Q5_0;
        }
        if (type == GGMLType.Q5_1) {
            return Q5_1;
        }
        throw new IllegalArgumentException("LegacyVecDot handles Q4_0, Q4_1, Q5_0 and Q5_1 weights, not " + type);
    }

    /** The weight type this instance reads. */
    public GGMLType weightType() {
        return weightType;
    }

    /** The activation type this instance consumes: Q8_0 for Q4_0 / Q5_0, Q8_1 for Q4_1 / Q5_1. */
    public GGMLType activationType() {
        return activationType;
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

    private void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, weightType);
        x.requireCompatible(w, activationType);
    }

    private float dot(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        switch (weightType) {
            case Q4_0:
                return dotQ4_0(b, off, x, aux);
            case Q4_1:
                return dotQ4_1(b, off, x, aux);
            case Q5_0:
                return dotQ5_0(b, off, x, aux);
            default:
                return dotQ5_1(b, off, x, aux);
        }
    }

    /**
     * Writes the 32 nibbles of a split-nibble {@code qs} field, given as its two little-endian halves, to
     * {@code aux[0 .. 32)} in element order. {@code h0}, {@code h8}, {@code h16} and {@code h24} hold the
     * fifth bits ({@code 0x10} per byte, or zero) of elements {@code 0-7}, {@code 8-15}, {@code 16-23} and
     * {@code 24-31}. When {@code signed}, every byte is then shifted down by the bias encoded in
     * {@code biasComplement}.
     */
    private static void unpackNibbles(long qsLo, long qsHi, long h0, long h8, long h16, long h24,
                                      long biasComplement, boolean signed, byte[] aux) {
        long e0 = (qsLo & LOW_NIBBLES) | h0;
        long e8 = (qsHi & LOW_NIBBLES) | h8;
        long e16 = ((qsLo >>> 4) & LOW_NIBBLES) | h16;
        long e24 = ((qsHi >>> 4) & LOW_NIBBLES) | h24;
        if (signed) {
            e0 = subtractBias(e0, biasComplement);
            e8 = subtractBias(e8, biasComplement);
            e16 = subtractBias(e16, biasComplement);
            e24 = subtractBias(e24, biasComplement);
        }
        putLong(aux, 0, e0);
        putLong(aux, 8, e8);
        putLong(aux, 16, e16);
        putLong(aux, 24, e24);
    }

    /**
     * quants.c:242-258: {@code v0 = (qs[j] & 0x0F) - 8}, {@code v1 = (qs[j] >> 4) - 8} paired with
     * {@code y.qs[j]} and {@code y.qs[j + 16]}; {@code sumf += sumi * d_x * d_y}.
     */
    private static float dotQ4_0(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int nb = x.n / QK4_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q4_0_BYTES) {
            final int qs = off + Q4_0_QS;
            unpackNibbles(b.getLong(qs), b.getLong(qs + 8), 0, 0, 0, 0, BIAS_8, true, aux);
            final int sumi = dotI8(aux, 0, yq, ib * QK4_0, QK4_0);
            sumf += sumi * fp16At(b, off + Q4_0_D) * yd[ib];
        }
        return sumf;
    }

    /**
     * quants.c:279-295: unsigned nibbles; {@code sumf += (d_x * d_y) * sumi + m_x * s_y}, where {@code s_y}
     * is the Q8_1 block sum term.
     */
    private static float dotQ4_1(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final float[] ys = x.s;
        final int nb = x.n / QK4_1;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q4_1_BYTES) {
            final int qs = off + Q4_1_QS;
            unpackNibbles(b.getLong(qs), b.getLong(qs + 8), 0, 0, 0, 0, 0, false, aux);
            final int sumi = dotI8(aux, 0, yq, ib * QK4_1, QK4_1);
            sumf += (fp16At(b, off + Q4_1_D) * yd[ib]) * sumi + fp16At(b, off + Q4_1_M) * ys[ib];
        }
        return sumf;
    }

    /**
     * quants.c:381-405: {@code xh_0 = bit j of qh << 4}, {@code xh_1 = bit (j + 16) of qh << 4},
     * {@code x0 = ((qs[j] & 0x0F) | xh_0) - 16}, {@code x1 = ((qs[j] >> 4) | xh_1) - 16};
     * {@code sumf += (d_x * d_y) * sumi}.
     */
    private static float dotQ5_0(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int nb = x.n / QK5_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q5_0_BYTES) {
            final int qh = b.getInt(off + Q5_0_QH);
            final int qs = off + Q5_0_QS;
            unpackNibbles(b.getLong(qs), b.getLong(qs + 8),
                    QH_SPREAD[qh & 0xFF], QH_SPREAD[(qh >>> 8) & 0xFF],
                    QH_SPREAD[(qh >>> 16) & 0xFF], QH_SPREAD[qh >>> 24], BIAS_16, true, aux);
            final int sumi = dotI8(aux, 0, yq, ib * QK5_0, QK5_0);
            sumf += (fp16At(b, off + Q5_0_D) * yd[ib]) * sumi;
        }
        return sumf;
    }

    /**
     * quants.c:424-448: unsigned 5-bit values {@code (qs[j] & 0xF) | xh_0} and {@code (qs[j] >> 4) | xh_1};
     * {@code sumf += (d_x * d_y) * sumi + m_x * s_y}.
     */
    private static float dotQ5_1(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final float[] ys = x.s;
        final int nb = x.n / QK5_1;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q5_1_BYTES) {
            final int qh = b.getInt(off + Q5_1_QH);
            final int qs = off + Q5_1_QS;
            unpackNibbles(b.getLong(qs), b.getLong(qs + 8),
                    QH_SPREAD[qh & 0xFF], QH_SPREAD[(qh >>> 8) & 0xFF],
                    QH_SPREAD[(qh >>> 16) & 0xFF], QH_SPREAD[qh >>> 24], 0, false, aux);
            final int sumi = dotI8(aux, 0, yq, ib * QK5_1, QK5_1);
            sumf += (fp16At(b, off + Q5_1_D) * yd[ib]) * sumi + fp16At(b, off + Q5_1_M) * ys[ib];
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "LegacyVecDot{" + weightType + " x " + activationType + "}";
    }
}
