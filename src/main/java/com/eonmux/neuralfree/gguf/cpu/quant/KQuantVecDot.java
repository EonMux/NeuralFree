package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.KQuantBlockDecoders;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.BYTE_ONES;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.LOW_NIBBLES;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.dotI8;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.putLong;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.subtractBias;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of the K-quant super-block types with Q8_K activations, ports of llama.cpp ggml/src/ggml-cpu/quants.c:
 * <ul>
 *   <li>Q2_K, {@code ggml_vec_dot_q2_K_q8_K_generic} (quants.c:565-615)</li>
 *   <li>Q3_K, {@code ggml_vec_dot_q3_K_q8_K_generic} (quants.c:617-694)</li>
 *   <li>Q4_K, {@code ggml_vec_dot_q4_K_q8_K_generic} (quants.c:696-769)</li>
 *   <li>Q5_K, {@code ggml_vec_dot_q5_K_q8_K_generic} (quants.c:771-849)</li>
 *   <li>Q6_K, {@code ggml_vec_dot_q6_K_q8_K_generic} (quants.c:851-904)</li>
 * </ul>
 * Like ggml's generic kernels, each 256-element super-block is first unpacked into a 256-byte buffer
 * ({@code aux8}) holding the integer weight quant of every element in element order, here with 64-bit SWAR
 * over little-endian {@code getLong} reads. Each 16- or 32-element sub-block is then reduced against the
 * Q8_K quants with {@link Q8VecDot#dotI8} and weighted by its integer scale. The per-super-block integer
 * sums ({@code isum}, and {@code summs} / {@code sumi} for the mins via {@code bsums}) are exactly ggml's.
 * ggml spreads {@code isum} over eight int lanes and eight float partial sums; here it is one integer per
 * super-block, multiplied by the same float scale, so results agree with ggml up to float summation order.
 * <p>
 * Instances are immutable, stateless and thread-safe.
 */
public final class KQuantVecDot implements CpuVecDot {

    public static final KQuantVecDot Q2_K = new KQuantVecDot(GGMLType.Q2_K);
    public static final KQuantVecDot Q3_K = new KQuantVecDot(GGMLType.Q3_K);
    public static final KQuantVecDot Q4_K = new KQuantVecDot(GGMLType.Q4_K);
    public static final KQuantVecDot Q5_K = new KQuantVecDot(GGMLType.Q5_K);
    public static final KQuantVecDot Q6_K = new KQuantVecDot(GGMLType.Q6_K);

    private static final Set<GGMLType> SUPPORTED = Collections.unmodifiableSet(
            EnumSet.of(GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K));

    /** Number of {@code bsums} entries per Q8_K block (one per 16 quants). */
    private static final int BSUMS_PER_BLOCK = QK_K / 16;

    /** Bits 0-1 of every byte. */
    private static final long LOW_PAIRS = 0x0303030303030303L;
    /** {@code 0x80 - 4} in every byte: maps 3-bit values {@code 0..7} to {@code -4..3}. */
    private static final long BIAS_4 = 0x7C7C7C7C7C7C7C7CL;
    /** {@code 0x80 - 32} in every byte: maps 6-bit values {@code 0..63} to {@code -32..31}. */
    private static final long BIAS_32 = 0x6060606060606060L;

    /** {@code kmask1}, {@code kmask2}, {@code kmask3} of the Q4_K / Q5_K scale unpacking (quants.c:709-711). */
    private static final int KMASK1 = 0x3f3f3f3f;
    private static final int KMASK2 = 0x0f0f0f0f;
    private static final int KMASK3 = 0x03030303;

    private final GGMLType weightType;

    private KQuantVecDot(GGMLType weightType) {
        this.weightType = weightType;
    }

    /** True for Q2_K, Q3_K, Q4_K, Q5_K and Q6_K. */
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
    public static KQuantVecDot forType(GGMLType type) {
        if (type == GGMLType.Q2_K) {
            return Q2_K;
        }
        if (type == GGMLType.Q3_K) {
            return Q3_K;
        }
        if (type == GGMLType.Q4_K) {
            return Q4_K;
        }
        if (type == GGMLType.Q5_K) {
            return Q5_K;
        }
        if (type == GGMLType.Q6_K) {
            return Q6_K;
        }
        throw new IllegalArgumentException("KQuantVecDot handles Q2_K, Q3_K, Q4_K, Q5_K and Q6_K weights, not "
                + type);
    }

    /** The weight type this instance reads. */
    public GGMLType weightType() {
        return weightType;
    }

    /** The activation type every K-quant vec-dot consumes, Q8_K. */
    public GGMLType activationType() {
        return GGMLType.Q8_K;
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(w.rowBuffer(row), w.rowOffset(row), x, Q8VecDot.scratch());
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        final Q8VecDot.Scratch scratch = Q8VecDot.scratch();
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, y, yOff, (b, off) -> dot(b, off, x, scratch));
    }

    private void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, weightType);
        x.requireCompatible(w, GGMLType.Q8_K);
    }

    private float dot(ByteBuffer b, int off, QuantizedActivations x, Q8VecDot.Scratch scratch) {
        switch (weightType) {
            case Q2_K:
                return dotQ2_K(b, off, x, scratch.quants);
            case Q3_K:
                return dotQ3_K(b, off, x, scratch.quants, scratch.scales);
            case Q4_K:
                return dotQ4_K(b, off, x, scratch.quants, scratch.scales);
            case Q5_K:
                return dotQ5_K(b, off, x, scratch.quants, scratch.scales);
            default:
                return dotQ6_K(b, off, x, scratch.quants);
        }
    }

    /**
     * quants.c:578-614. Element {@code e = 128k + 32s + l} of the super-block is
     * {@code (qs[32k + l] >> 2s) & 3}, and 16-element group {@code g = e / 16} has the 4-bit scale
     * {@code scales[g] & 0xF} and the 4-bit min {@code scales[g] >> 4}. Per super-block:
     * {@code summs = sum_g bsums[g] * (scales[g] >> 4)}, {@code isum = sum_g (scales[g] & 0xF) * sum q8 * q2},
     * {@code sumf += (d_y * d) * isum - (d_y * dmin) * summs}.
     */
    private static float dotQ2_K(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q2_K_BYTES) {
            for (int k = 0; k < QK_K / 128; k++) {
                final int q2 = off + Q2_K_QS + 32 * k;
                for (int t = 0; t < 4; t++) {
                    final long v = b.getLong(q2 + 8 * t);
                    for (int s = 0; s < 4; s++) {
                        putLong(aux, 128 * k + 32 * s + 8 * t, (v >>> (2 * s)) & LOW_PAIRS);
                    }
                }
            }
            final long sc0 = b.getLong(off + Q2_K_SCALES);
            final long sc1 = b.getLong(off + Q2_K_SCALES + 8);
            final int xo = i * QK_K;
            final int bo = i * BSUMS_PER_BLOCK;
            int summs = 0;
            int isum = 0;
            for (int g = 0; g < 16; g++) {
                final int sc = (int) ((g < 8 ? sc0 : sc1) >>> (8 * (g & 7))) & 0xFF;
                summs += bsums[bo + g] * (sc >>> 4);
                isum += (sc & 0xF) * dotI8(aux, 16 * g, yq, xo + 16 * g, 16);
            }
            final float dall = yd[i] * fp16At(b, off + Q2_K_D);
            final float dmin = yd[i] * fp16At(b, off + Q2_K_DMIN);
            sumf += dall * isum - dmin * summs;
        }
        return sumf;
    }

    /**
     * quants.c:647-692. Element {@code e = 128c + 32s + l} is
     * {@code ((qs[32c + l] >> 2s) & 3) - ((hmask[l] & (1 << (4c + s))) ? 0 : 4)}, and 16-element group
     * {@code g} has the signed 6-bit scale {@code scales[g] - 32} unpacked with {@code kmask1/kmask2}
     * ({@link KQuantBlockDecoders#q3kScales}). Per super-block:
     * {@code sumf += (d * d_y) * sum_g (scales[g] - 32) * sum q8 * a}.
     */
    private static float dotQ3_K(ByteBuffer b, int off, QuantizedActivations x, byte[] aux, byte[] scales) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q3_K_BYTES) {
            final int hm = off + Q3_K_HMASK;
            for (int c = 0; c < QK_K / 128; c++) {
                final int q3 = off + Q3_K_QS + 32 * c;
                for (int t = 0; t < 4; t++) {
                    final long v = b.getLong(q3 + 8 * t);
                    final long h = b.getLong(hm + 8 * t);
                    for (int s = 0; s < 4; s++) {
                        final long low = (v >>> (2 * s)) & LOW_PAIRS;
                        final long high = ((h >>> (4 * c + s)) & BYTE_ONES) << 2;
                        putLong(aux, 128 * c + 32 * s + 8 * t, subtractBias(low | high, BIAS_4));
                    }
                }
            }
            KQuantBlockDecoders.q3kScales(b, off + Q3_K_SCALES, scales);
            final int xo = i * QK_K;
            int isum = 0;
            for (int g = 0; g < 16; g++) {
                isum += (scales[g] - 32) * dotI8(aux, 16 * g, yq, xo + 16 * g, 16);
            }
            final float d = fp16At(b, off + Q3_K_D) * yd[i];
            sumf += d * isum;
        }
        return sumf;
    }

    /**
     * Q4_K / Q5_K scale and min unpacking of quants.c:736-741 (identical at 816-821): from the three
     * little-endian words {@code utmp[0..2]} of {@code scales},
     * {@code utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)},
     * {@code uaux = utmp[1] & kmask1}, {@code utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4)},
     * {@code utmp[2] = uaux}, {@code utmp[0] &= kmask1}. Bytes {@code 0..7} of {@code utmp} are the eight
     * 6-bit scales and bytes {@code 8..15} the eight 6-bit mins, the values {@code get_scale_min_k4}
     * (llama.cpp ggml-quants.c:880-887, {@link KQuantBlockDecoders#scaleMinK4}) yields for {@code j = 0..7}.
     *
     * @param out receives the scales in {@code out[0 .. 8)} and the mins in {@code out[8 .. 16)}
     */
    private static void unpackScalesMins(ByteBuffer b, int scalesOff, byte[] out) {
        final int u0 = b.getInt(scalesOff);
        final int u1 = b.getInt(scalesOff + 4);
        final int u2 = b.getInt(scalesOff + 8);
        final int mins4to7 = ((u2 >>> 4) & KMASK2) | (((u1 >>> 6) & KMASK3) << 4);
        final int mins0to3 = u1 & KMASK1;
        final int scales4to7 = (u2 & KMASK2) | (((u0 >>> 6) & KMASK3) << 4);
        final int scales0to3 = u0 & KMASK1;
        putLong(out, 0, (scales0to3 & 0xFFFFFFFFL) | ((long) scales4to7 << 32));
        putLong(out, 8, (mins0to3 & 0xFFFFFFFFL) | ((long) mins4to7 << 32));
    }

    /**
     * {@code sumi = sum_{j<16} bsums[j] * mins[j / 2]} of quants.c:744 (Q4_K) and 824 (Q5_K): each 6-bit min
     * covers two 16-quant {@code bsums} of super-block {@code i}.
     *
     * @param sm the output of {@link #unpackScalesMins}
     */
    private static int minsSum(short[] bsums, int i, byte[] sm) {
        final int bo = i * BSUMS_PER_BLOCK;
        int sumi = 0;
        for (int j = 0; j < BSUMS_PER_BLOCK; j++) {
            sumi += bsums[bo + j] * sm[8 + (j >>> 1)];
        }
        return sumi;
    }

    /**
     * {@code isum = sum_{j<8} scales[j] * sum_{l<32} q8[32j + l] * aux8[32j + l]} of quants.c:745-761 (Q4_K)
     * and 825-841 (Q5_K) for super-block {@code i}.
     *
     * @param sm the output of {@link #unpackScalesMins}
     */
    private static int scaledQuantSum(byte[] aux, byte[] sm, byte[] yq, int i) {
        final int xo = i * QK_K;
        int isum = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            isum += sm[j] * dotI8(aux, 32 * j, yq, xo + 32 * j, 32);
        }
        return isum;
    }

    /**
     * quants.c:722-768, the {@code q4_K x q8_K} trace:
     * <ol>
     *   <li>{@code aux8[64j + l] = qs[32j + l] & 0xF} and {@code aux8[64j + 32 + l] = qs[32j + l] >> 4}
     *       for {@code j < 4}, {@code l < 32} (two 32-element sub-blocks per 32 {@code qs} bytes);</li>
     *   <li>scales and mins from {@link #unpackScalesMins} (6-bit, unsigned);</li>
     *   <li>{@code sumi = sum_{j<16} bsums[j] * mins[j / 2]}: each min covers two 16-quant {@code bsums};</li>
     *   <li>{@code isum = sum_{j<8} scales[j] * sum_{l<32} q8[32j + l] * aux8[32j + l]};</li>
     *   <li>{@code sums += (d_x * d_y) * isum}, {@code sumf -= (dmin_x * d_y) * sumi};
     *       the result is {@code sumf + sums}.</li>
     * </ol>
     * This equals {@code sum_e (d * sc * q_e - dmin * m) * (d_y * q8_e)} of {@code dequantize_row_q4_K}
     * (llama.cpp ggml-quants.c:1530-1552) because {@code bsums} are the exact 16-quant sums of the activations.
     */
    private static float dotQ4_K(ByteBuffer b, int off, QuantizedActivations x, byte[] aux, byte[] sm) {
        final int nb = x.n / QK_K;
        float sumf = 0;
        float sums = 0;
        for (int i = 0; i < nb; i++, off += Q4_K_BYTES) {
            for (int j = 0; j < QK_K / 64; j++) {
                final int q4 = off + Q4_K_QS + 32 * j;
                for (int t = 0; t < 4; t++) {
                    final long v = b.getLong(q4 + 8 * t);
                    putLong(aux, 64 * j + 8 * t, v & LOW_NIBBLES);
                    putLong(aux, 64 * j + 32 + 8 * t, (v >>> 4) & LOW_NIBBLES);
                }
            }
            unpackScalesMins(b, off + Q4_K_SCALES, sm);
            final int sumi = minsSum(x.bsums, i, sm);
            final int isum = scaledQuantSum(aux, sm, x.qs, i);
            final float d = fp16At(b, off + Q4_K_D) * x.d[i];
            sums += d * isum;
            final float dmin = fp16At(b, off + Q4_K_DMIN) * x.d[i];
            sumf -= dmin * sumi;
        }
        return sumf + sums;
    }

    /**
     * quants.c:797-848: as Q4_K, with the fifth bit of element {@code 64j + l} taken from
     * {@code qh[l] & (1 << 2j)} and of element {@code 64j + 32 + l} from {@code qh[l] & (1 << (2j + 1))},
     * each adding 16.
     */
    private static float dotQ5_K(ByteBuffer b, int off, QuantizedActivations x, byte[] aux, byte[] sm) {
        final int nb = x.n / QK_K;
        float sumf = 0;
        float sums = 0;
        for (int i = 0; i < nb; i++, off += Q5_K_BYTES) {
            final int qh = off + Q5_K_QH;
            for (int j = 0; j < QK_K / 64; j++) {
                final int q4 = off + Q5_K_QS + 32 * j;
                for (int t = 0; t < 4; t++) {
                    final long v = b.getLong(q4 + 8 * t);
                    final long h = b.getLong(qh + 8 * t);
                    final long hLow = ((h >>> (2 * j)) & BYTE_ONES) << 4;
                    final long hHigh = ((h >>> (2 * j + 1)) & BYTE_ONES) << 4;
                    putLong(aux, 64 * j + 8 * t, (v & LOW_NIBBLES) | hLow);
                    putLong(aux, 64 * j + 32 + 8 * t, ((v >>> 4) & LOW_NIBBLES) | hHigh);
                }
            }
            unpackScalesMins(b, off + Q5_K_SCALES, sm);
            final int sumi = minsSum(x.bsums, i, sm);
            final int isum = scaledQuantSum(aux, sm, x.qs, i);
            final float d = fp16At(b, off + Q5_K_D) * x.d[i];
            sums += d * isum;
            final float dmin = fp16At(b, off + Q5_K_DMIN) * x.d[i];
            sumf -= dmin * sumi;
        }
        return sumf + sums;
    }

    /**
     * quants.c:871-902. Per 128-element half {@code c} ({@code ql += 64c}, {@code qh += 32c}), for {@code l < 32}:
     * {@code a[l] = (ql[l] & 0xF | ((qh[l] >> 0) & 3) << 4) - 32},
     * {@code a[l + 32] = (ql[l + 32] & 0xF | ((qh[l] >> 2) & 3) << 4) - 32},
     * {@code a[l + 64] = (ql[l] >> 4 | ((qh[l] >> 4) & 3) << 4) - 32},
     * {@code a[l + 96] = (ql[l + 32] >> 4 | ((qh[l] >> 6) & 3) << 4) - 32}. 16-element group {@code g} has the
     * signed 8-bit scale {@code scales[g]}; {@code sums += (d * d_y) * sum_g scales[g] * sum q8 * a}.
     */
    private static float dotQ6_K(ByteBuffer b, int off, QuantizedActivations x, byte[] aux) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q6_K_BYTES) {
            for (int c = 0; c < QK_K / 128; c++) {
                final int ql = off + Q6_K_QL + 64 * c;
                final int qh = off + Q6_K_QH + 32 * c;
                final int a = 128 * c;
                for (int t = 0; t < 4; t++) {
                    final long lo = b.getLong(ql + 8 * t);
                    final long hi = b.getLong(ql + 32 + 8 * t);
                    final long h = b.getLong(qh + 8 * t);
                    putLong(aux, a + 8 * t,
                            subtractBias((lo & LOW_NIBBLES) | ((h & LOW_PAIRS) << 4), BIAS_32));
                    putLong(aux, a + 32 + 8 * t,
                            subtractBias((hi & LOW_NIBBLES) | (((h >>> 2) & LOW_PAIRS) << 4), BIAS_32));
                    putLong(aux, a + 64 + 8 * t,
                            subtractBias(((lo >>> 4) & LOW_NIBBLES) | (((h >>> 4) & LOW_PAIRS) << 4), BIAS_32));
                    putLong(aux, a + 96 + 8 * t,
                            subtractBias(((hi >>> 4) & LOW_NIBBLES) | (((h >>> 6) & LOW_PAIRS) << 4), BIAS_32));
                }
            }
            final int sc = off + Q6_K_SCALES;
            final int xo = i * QK_K;
            int isum = 0;
            for (int g = 0; g < 16; g++) {
                isum += b.get(sc + g) * dotI8(aux, 16 * g, yq, xo + 16 * g, 16);
            }
            final float d = fp16At(b, off + Q6_K_D) * yd[i];
            sumf += d * isum;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "KQuantVecDot{" + weightType + " x Q8_K}";
    }
}
