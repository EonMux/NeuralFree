package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.IQuantBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of the importance-matrix types against quantized activations: IQ2_XXS, IQ2_XS, IQ2_S,
 * IQ3_XXS, IQ3_S, IQ1_S, IQ1_M and IQ4_XS against Q8_K, and IQ4_NL against Q8_0. Each kernel is a port of
 * the matching {@code ggml_vec_dot_<type>_q8_*_generic} function of llama.cpp ggml/src/ggml-cpu/quants.c with the
 * same integer sums and the same float accumulation order, so results equal ggml's reference path.
 * <p>
 * Weights are read in place from the row buffers of a {@link CpuTensorRef} with absolute little-endian
 * reads; activations come from {@link CpuActivationQuantizer}. Instances are immutable, thread-safe and
 * allocate nothing per call.
 */
public final class IQuantVecDot implements CpuVecDot {

    /** IQ1_M grid offset ({@code IQ1M_DELTA}, llama.cpp ggml-common.h:1133). */
    private static final float IQ1M_DELTA = 0.125f;

    public static final IQuantVecDot IQ2_XXS = new IQuantVecDot(GGMLType.IQ2_XXS, IQuantVecDot::dotIq2Xxs);
    public static final IQuantVecDot IQ2_XS = new IQuantVecDot(GGMLType.IQ2_XS, IQuantVecDot::dotIq2Xs);
    public static final IQuantVecDot IQ2_S = new IQuantVecDot(GGMLType.IQ2_S, IQuantVecDot::dotIq2S);
    public static final IQuantVecDot IQ3_XXS = new IQuantVecDot(GGMLType.IQ3_XXS, IQuantVecDot::dotIq3Xxs);
    public static final IQuantVecDot IQ3_S = new IQuantVecDot(GGMLType.IQ3_S, IQuantVecDot::dotIq3S);
    public static final IQuantVecDot IQ1_S = new IQuantVecDot(GGMLType.IQ1_S, IQuantVecDot::dotIq1S);
    public static final IQuantVecDot IQ1_M = new IQuantVecDot(GGMLType.IQ1_M, IQuantVecDot::dotIq1M);
    public static final IQuantVecDot IQ4_NL = new IQuantVecDot(GGMLType.IQ4_NL, IQuantVecDot::dotIq4Nl);
    public static final IQuantVecDot IQ4_XS = new IQuantVecDot(GGMLType.IQ4_XS, IQuantVecDot::dotIq4Xs);

    private static final Map<GGMLType, IQuantVecDot> BY_TYPE = index(
            IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS);

    /** One row kernel: {@code nb} weight blocks starting at absolute index {@code off} of {@code b}. */
    @FunctionalInterface
    private interface RowKernel {
        float dot(ByteBuffer b, int off, QuantizedActivations x, int nb);
    }

    private final GGMLType type;
    private final GGMLType activationType;
    private final int blockElements;
    private final RowKernel kernel;

    private IQuantVecDot(GGMLType type, RowKernel kernel) {
        this.type = type;
        this.activationType = CpuActivationQuantizer.activationType(type);
        this.blockElements = BlockLayout.of(type).blockElements();
        this.kernel = kernel;
    }

    private static Map<GGMLType, IQuantVecDot> index(IQuantVecDot... dots) {
        Map<GGMLType, IQuantVecDot> m = new EnumMap<>(GGMLType.class);
        for (IQuantVecDot d : dots) {
            m.put(d.type, d);
        }
        return Collections.unmodifiableMap(m);
    }

    /** True when this class has a vec-dot for {@code type}. */
    public static boolean supports(GGMLType type) {
        return type != null && BY_TYPE.containsKey(type);
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return BY_TYPE.keySet();
    }

    /**
     * Returns the vec-dot of {@code type}.
     *
     * @throws IllegalArgumentException when {@code type} is not an importance-matrix type
     */
    public static IQuantVecDot forType(GGMLType type) {
        IQuantVecDot d = type == null ? null : BY_TYPE.get(type);
        if (d == null) {
            throw new IllegalArgumentException("IQuantVecDot has no kernel for ggml type " + type
                    + "; it handles " + BY_TYPE.keySet());
        }
        return d;
    }

    /** The weight type of this vec-dot. */
    public GGMLType weightType() {
        return type;
    }

    /** The activation type this vec-dot consumes (Q8_K, or Q8_0 for IQ4_NL). */
    public GGMLType activationType() {
        return activationType;
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return kernel.dot(w.rowBuffer(row), w.rowOffset(row), x, (int) (w.cols() / blockElements));
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        final int nb = (int) (w.cols() / blockElements);
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, y, yOff, (b, off) -> kernel.dot(b, off, x, nb));
    }

    private void check(CpuTensorRef w, QuantizedActivations x) {
        if (w == null || x == null) {
            throw new IllegalArgumentException("weight tensor and activations must not be null");
        }
        if (w.type() != type) {
            throw new IllegalArgumentException("the " + type + " vec-dot cannot read a " + w.type() + " tensor");
        }
        x.requireCompatible(w, activationType);
    }

    // ------------------------------------------------------------------ shared lane helpers

    /**
     * {@code sum_j grid[j] * q8[o + j] * (signs & kmask_iq2xs[j] ? -1 : 1)} over the 8 unsigned lanes of a
     * packed uint64 grid entry.
     */
    private static int signedGridDot8(long grid, int signs, byte[] q8, int o) {
        int s = 0;
        for (int j = 0; j < 8; j++) {
            final int neg = -((signs >>> j) & 1);
            s += ((((int) (grid >>> (j << 3)) & 0xFF) ^ neg) - neg) * q8[o + j];
        }
        return s;
    }

    /**
     * {@code sum_j grid[j] * q8[o + j] * (signs & kmask_iq2xs[j + bit0] ? -1 : 1)} over the 4 unsigned lanes
     * of a packed uint32 grid entry.
     */
    private static int signedGridDot4(int grid, int signs, int bit0, byte[] q8, int o) {
        int s = 0;
        for (int j = 0; j < 4; j++) {
            final int neg = -((signs >>> (j + bit0)) & 1);
            s += ((((grid >>> (j << 3)) & 0xFF) ^ neg) - neg) * q8[o + j];
        }
        return s;
    }

    /** {@code sum_j (int8) grid[j] * q8[o + j]} over the 8 signed lanes of a packed {@code iq1s_grid} entry. */
    private static int signedLaneDot8(long grid, byte[] q8, int o) {
        int s = 0;
        for (int j = 0; j < 8; j++) {
            s += ((byte) (grid >>> (j << 3))) * q8[o + j];
        }
        return s;
    }

    // ------------------------------------------------------------------ kernels

    /** {@code ggml_vec_dot_iq2_xxs_q8_K_generic} (llama.cpp ggml-cpu/quants.c:906-946). */
    private static float dotIq2Xxs(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        int o = 0;
        float sumf = 0.f;
        for (int i = 0; i < nb; i++, off += IQ2_XXS_BYTES) {
            final float d = fp16At(b, off + IQ2_XXS_D) * yd[i];
            int q2 = off + IQ2_XXS_QS;
            int bsum = 0;
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, q2 += 8) {
                final long aux = b.getLong(q2);
                final int aux0 = (int) aux;
                final int aux1 = (int) (aux >>> 32);
                final int ls = 2 * (aux1 >>> 28) + 1;
                int sumi = 0;
                for (int l = 0; l < 4; ++l, o += 8) {
                    sumi += signedGridDot8(QuantTables.iq2xxsGrid((aux0 >>> (8 * l)) & 0xFF),
                            QuantTables.ksignsIq2xs((aux1 >>> (7 * l)) & 127), q8, o);
                }
                bsum += sumi * ls;
            }
            sumf += d * bsum;
        }
        return 0.125f * sumf;
    }

    /** {@code ggml_vec_dot_iq2_xs_q8_K_generic} (llama.cpp ggml-cpu/quants.c:948-996). */
    private static float dotIq2Xs(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        int o = 0;
        float sumf = 0.f;
        for (int i = 0; i < nb; i++, off += IQ2_XS_BYTES) {
            final float d = fp16At(b, off + IQ2_XS_D) * yd[i];
            int q2 = off + IQ2_XS_QS;
            int bsum = 0;
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, q2 += 8) {
                final int sc = b.get(off + IQ2_XS_SCALES + ib32) & 0xFF;
                final int ls1 = 2 * (sc & 0xf) + 1;
                final int ls2 = 2 * (sc >>> 4) + 1;
                final long q4 = b.getLong(q2);
                int sumi = 0;
                for (int l = 0; l < 2; ++l, o += 8) {
                    final int q = (int) (q4 >>> (16 * l)) & 0xFFFF;
                    sumi += signedGridDot8(QuantTables.iq2xsGrid(q & 511), QuantTables.ksignsIq2xs(q >>> 9), q8, o);
                }
                bsum += sumi * ls1;
                sumi = 0;
                for (int l = 2; l < 4; ++l, o += 8) {
                    final int q = (int) (q4 >>> (16 * l)) & 0xFFFF;
                    sumi += signedGridDot8(QuantTables.iq2xsGrid(q & 511), QuantTables.ksignsIq2xs(q >>> 9), q8, o);
                }
                bsum += sumi * ls2;
            }
            sumf += d * bsum;
        }
        return 0.125f * sumf;
    }

    /** {@code ggml_vec_dot_iq2_s_q8_K_generic} (llama.cpp ggml-cpu/quants.c:998-1048). */
    private static float dotIq2S(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        int o = 0;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += IQ2_S_BYTES) {
            final float d = fp16At(b, off + IQ2_S_D) * yd[i];
            int qs = off + IQ2_S_QS;
            int signs = off + IQ2_S_SIGNS;
            int bsum = 0;
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, qs += 4, signs += 4) {
                final int sc = b.get(off + IQ2_S_SCALES + ib32) & 0xFF;
                final int ls1 = 1 + 2 * (sc & 0xf);
                final int ls2 = 1 + 2 * (sc >>> 4);
                final int qh = b.get(off + IQ2_S_QH + ib32) & 0xFF;
                final int q4 = b.getInt(qs);
                final int s4 = b.getInt(signs);
                int sumi1 = 0;
                int sumi2 = 0;
                for (int l = 0; l < 2; ++l, o += 8) {
                    final int idx = ((q4 >>> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
                    sumi1 += signedGridDot8(QuantTables.iq2sGrid(idx), (s4 >>> (8 * l)) & 0xFF, q8, o);
                }
                for (int l = 2; l < 4; ++l, o += 8) {
                    final int idx = ((q4 >>> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
                    sumi2 += signedGridDot8(QuantTables.iq2sGrid(idx), (s4 >>> (8 * l)) & 0xFF, q8, o);
                }
                bsum += ls1 * sumi1 + ls2 * sumi2;
            }
            sumf += d * bsum;
        }
        return 0.125f * sumf;
    }

    /** {@code ggml_vec_dot_iq3_xxs_q8_K_generic} (llama.cpp ggml-cpu/quants.c:1050-1092). */
    private static float dotIq3Xxs(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        int o = 0;
        float sumf = 0.f;
        for (int i = 0; i < nb; i++, off += IQ3_XXS_BYTES) {
            final float d = fp16At(b, off + IQ3_XXS_D) * yd[i];
            int q3 = off + IQ3_XXS_QS;
            int gas = off + IQ3_XXS_SAS;
            int bsum = 0;
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, q3 += 8, gas += 4) {
                final int aux32 = b.getInt(gas);
                final int ls = 2 * (aux32 >>> 28) + 1;
                final long idx8 = b.getLong(q3);
                int sumi = 0;
                for (int l = 0; l < 4; ++l, o += 8) {
                    final int grid1 = QuantTables.iq3xxsGrid((int) (idx8 >>> (16 * l)) & 0xFF);
                    final int grid2 = QuantTables.iq3xxsGrid((int) (idx8 >>> (16 * l + 8)) & 0xFF);
                    final int signs = QuantTables.ksignsIq2xs((aux32 >>> (7 * l)) & 127);
                    sumi += signedGridDot4(grid1, signs, 0, q8, o) + signedGridDot4(grid2, signs, 4, q8, o + 4);
                }
                bsum += sumi * ls;
            }
            sumf += d * bsum;
        }
        return 0.25f * sumf;
    }

    /** {@code ggml_vec_dot_iq3_s_q8_K_generic} (llama.cpp ggml-cpu/quants.c:1094-1148). */
    private static float dotIq3S(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        int o = 0;
        float sumf = 0.f;
        for (int i = 0; i < nb; i++, off += IQ3_S_BYTES) {
            final float d = fp16At(b, off + IQ3_S_D) * yd[i];
            int qs = off + IQ3_S_QS;
            int signs = off + IQ3_S_SIGNS;
            final int qh = off + IQ3_S_QH;
            int bsum = 0;
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                final int sc = b.get(off + IQ3_S_SCALES + ib32 / 2) & 0xFF;
                final int ls1 = 2 * (sc & 0xf) + 1;
                final int ls2 = 2 * (sc >>> 4) + 1;
                bsum += iq3sHalf(b.getLong(qs), b.get(qh + ib32) & 0xFF, b.getInt(signs), q8, o) * ls1;
                qs += 8;
                signs += 4;
                o += 32;
                bsum += iq3sHalf(b.getLong(qs), b.get(qh + ib32 + 1) & 0xFF, b.getInt(signs), q8, o) * ls2;
                qs += 8;
                signs += 4;
                o += 32;
            }
            sumf += d * bsum;
        }
        return sumf;
    }

    /**
     * Integer sum of one 32-element IQ3_S half: four groups of two 4-lane grid entries whose 9th index bit
     * comes from the {@code qh} byte {@code h}.
     *
     * @param idx8   the 8 grid index bytes {@code qs[0..8)} of this half, little-endian
     * @param signs4 the 4 sign bytes {@code signs[0..4)} of this half, little-endian
     */
    private static int iq3sHalf(long idx8, int h, int signs4, byte[] q8, int o) {
        int sumi = 0;
        for (int l = 0; l < 4; ++l, o += 8) {
            final int i1 = ((int) (idx8 >>> (16 * l)) & 0xFF) | ((h << (8 - 2 * l)) & 256);
            final int i2 = ((int) (idx8 >>> (16 * l + 8)) & 0xFF) | ((h << (7 - 2 * l)) & 256);
            final int s = (signs4 >>> (8 * l)) & 0xFF;
            sumi += signedGridDot4(QuantTables.iq3sGrid(i1), s, 0, q8, o)
                    + signedGridDot4(QuantTables.iq3sGrid(i2), s, 4, q8, o + 4);
        }
        return sumi;
    }

    /**
     * {@code ggml_vec_dot_iq1_s_q8_K_generic} (llama.cpp ggml-cpu/quants.c:1150-1191): the {@code ±IQ1S_DELTA}
     * offset term uses the Q8_K {@code bsums} of each 32-element group.
     */
    private static float dotIq1S(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        int o = 0;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += IQ1_S_BYTES) {
            int qs = off + IQ1_S_QS;
            final int bs = i * (QK_K / 16);
            int sumi = 0;
            int sumi1 = 0;
            for (int ib = 0; ib < QK_K / 32; ++ib, qs += 4) {
                final int qh = b.getShort(off + IQ1_S_QH + 2 * ib) & 0xFFFF;
                final int ls = 2 * ((qh >>> 12) & 7) + 1;
                final int delta = (qh & 0x8000) != 0 ? -1 : 1;
                final int q4 = b.getInt(qs);
                int lsum = 0;
                for (int l = 0; l < 4; ++l, o += 8) {
                    final int idx = ((q4 >>> (8 * l)) & 0xFF) | (((qh >>> (3 * l)) & 7) << 8);
                    lsum += signedLaneDot8(QuantTables.iq1sGrid(idx), q8, o);
                }
                sumi += ls * lsum;
                sumi1 += ls * delta * (bsums[bs + 2 * ib] + bsums[bs + 2 * ib + 1]);
            }
            sumf += fp16At(b, off + IQ1_S_D) * yd[i] * (sumi + QuantTables.IQ1S_DELTA * sumi1);
        }
        return sumf;
    }

    /**
     * {@code ggml_vec_dot_iq1_m_q8_K_generic} (llama.cpp ggml-cpu/quants.c:1193-1252): per-group sign of the
     * {@code IQ1M_DELTA} offset from {@code qh}, two 3-bit sub-scales per 32 elements, and the super-block
     * half scale assembled from the top nibbles of the four {@code scales} words.
     */
    private static float dotIq1M(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        int o = 0;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += IQ1_M_BYTES) {
            final int sc = off + IQ1_M_SCALES;
            final float scale = IQuantBlockDecoders.iq1mSuperScale(b, sc);
            int qs = off + IQ1_M_QS;
            int qh = off + IQ1_M_QH;
            int sumi1 = 0;
            int sumi2 = 0;
            for (int ib = 0; ib < QK_K / 32; ++ib, qs += 4, qh += 2) {
                final int q4 = b.getInt(qs);
                int sum1Lo = 0;
                int sum1Hi = 0;
                int sum2Lo = 0;
                int sum2Hi = 0;
                for (int l = 0; l < 4; ++l, o += 8) {
                    final int h = b.get(qh + l / 2) & 0xFF;
                    final int idx = ((q4 >>> (8 * l)) & 0xFF) | ((h << (8 - 4 * (l % 2))) & 0x700);
                    final int lsum1 = signedLaneDot8(QuantTables.iq1sGrid(idx), q8, o);
                    final int lsum2 = laneSum8(q8, o);
                    final int delta = (h & (l % 2 == 0 ? 0x08 : 0x80)) != 0 ? -1 : 1;
                    if (l < 2) {
                        sum1Lo += lsum1;
                        sum2Lo += lsum2 * delta;
                    } else {
                        sum1Hi += lsum1;
                        sum2Hi += lsum2 * delta;
                    }
                }
                final int s16 = b.getShort(sc + 2 * (ib / 2)) & 0xFFFF;
                final int ls1 = 2 * ((s16 >>> (6 * (ib % 2))) & 0x7) + 1;
                final int ls2 = 2 * ((s16 >>> (6 * (ib % 2) + 3)) & 0x7) + 1;
                sumi1 += sum1Lo * ls1 + sum1Hi * ls2;
                sumi2 += sum2Lo * ls1 + sum2Hi * ls2;
            }
            sumf += scale * yd[i] * (sumi1 + IQ1M_DELTA * sumi2);
        }
        return sumf;
    }

    /** {@code sum_j q8[o + j]} over 8 activations. */
    private static int laneSum8(byte[] q8, int o) {
        int s = 0;
        for (int j = 0; j < 8; j++) {
            s += q8[o + j];
        }
        return s;
    }

    /** {@code ggml_vec_dot_iq4_nl_q8_0_generic} (llama.cpp ggml-cpu/quants.c:1254-1281). */
    private static float dotIq4Nl(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int half = QK4_NL / 2;
        float sumf = 0;
        for (int ib = 0, o = 0; ib < nb; ++ib, off += IQ4_NL_BYTES, o += QK4_NL) {
            final float d = yd[ib] * fp16At(b, off + IQ4_NL_D);
            final int qs = off + IQ4_NL_QS;
            int sumi1 = 0;
            int sumi2 = 0;
            for (int j = 0; j < half; ++j) {
                final int q = b.get(qs + j) & 0xFF;
                sumi1 += q8[o + j] * QuantTables.kvalueIq4nl(q & 0xf);
                sumi2 += q8[o + j + half] * QuantTables.kvalueIq4nl(q >>> 4);
            }
            sumf += d * (sumi1 + sumi2);
        }
        return sumf;
    }

    /** {@code ggml_vec_dot_iq4_xs_q8_K_generic} (llama.cpp ggml-cpu/quants.c:1283-1327). */
    private static float dotIq4Xs(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        int o = 0;
        float sumf = 0;
        for (int ibl = 0; ibl < nb; ++ibl, off += IQ4_XS_BYTES) {
            final float d4d8 = fp16At(b, off + IQ4_XS_D) * yd[ibl];
            int h = b.getShort(off + IQ4_XS_SCALES_H) & 0xFFFF;
            int qs = off + IQ4_XS_QS;
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                final int sl = b.get(off + IQ4_XS_SCALES_L + ib / 2) & 0xFF;
                final int ls1 = (sl & 0xf) | ((h << 4) & 0x30);
                final int ls2 = (sl >>> 4) | ((h << 2) & 0x30);
                h >>>= 4;
                final float d1 = d4d8 * (ls1 - 32);
                final float d2 = d4d8 * (ls2 - 32);
                sumf += d1 * iq4Sub(b, qs, q8, o);
                qs += 16;
                o += 32;
                sumf += d2 * iq4Sub(b, qs, q8, o);
                qs += 16;
                o += 32;
            }
        }
        return sumf;
    }

    /** {@code sumi1 + sumi2} of one 32-element IQ4_XS sub-block: 16 bytes of split nibbles. */
    private static int iq4Sub(ByteBuffer b, int qs, byte[] q8, int o) {
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < 16; ++j) {
            final int q = b.get(qs + j) & 0xFF;
            sumi1 += q8[o + j] * QuantTables.kvalueIq4nl(q & 0xf);
            sumi2 += q8[o + j + 16] * QuantTables.kvalueIq4nl(q >>> 4);
        }
        return sumi1 + sumi2;
    }

    @Override
    public String toString() {
        return "IQuantVecDot{" + type + " x " + activationType + "}";
    }
}
