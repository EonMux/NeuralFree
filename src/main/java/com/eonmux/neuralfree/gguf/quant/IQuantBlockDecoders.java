package com.eonmux.neuralfree.gguf.quant;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16ToFp32;

/**
 * CPU decoders for the importance-matrix types IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M,
 * IQ4_NL and IQ4_XS, using the grids and codebooks of {@link QuantTables}.
 * <p>
 * Kernels are package-private: they expect a little-endian buffer and a range already validated by
 * {@link CpuBlockDecoders}.
 */
public final class IQuantBlockDecoders {

    private IQuantBlockDecoders() {
    }

    /** {@code +1} or {@code -1} for sign bit {@code j} of an 8-bit sign pattern, as {@code kmask_iq2xs} tests it. */
    private static float sign(int signs, int j) {
        return (signs & (1 << j)) != 0 ? -1.f : 1.f;
    }

    /**
     * IQ1_M super-block scale: the high nibbles of the four 16-bit {@code scales} words form a half
     * ({@code iq1m_scale_t}, llama.cpp ggml-quants.c:2687-2689).
     *
     * @param scalesOff absolute index of the 8-byte {@code scales} field
     */
    public static float iq1mSuperScale(ByteBuffer le, int scalesOff) {
        final int sc0 = le.getShort(scalesOff) & 0xFFFF;
        final int sc1 = le.getShort(scalesOff + 2) & 0xFFFF;
        final int sc2 = le.getShort(scalesOff + 4) & 0xFFFF;
        final int sc3 = le.getShort(scalesOff + 6) & 0xFFFF;
        final int u16 = (sc0 >>> 12) | ((sc1 >>> 8) & 0x00f0) | ((sc2 >>> 4) & 0x0f00) | (sc3 & 0xf000);
        return fp16ToFp32(u16);
    }

    /** {@code dequantize_row_iq2_xxs} (llama.cpp ggml-quants.c:2489-2513). */
    static void decodeIq2Xxs(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ2_XXS_BYTES) {
            final float d = fp16At(b, off + IQ2_XXS_D);
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                final int a = off + IQ2_XXS_QS + 8 * ib32;
                final int aux0 = b.getInt(a);
                final int aux1 = b.getInt(a + 4);
                final float db = d * (0.5f + (aux1 >>> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    final long grid = QuantTables.iq2xxsGrid((aux0 >>> (8 * l)) & 0xFF);
                    final int signs = QuantTables.ksignsIq2xs((aux1 >>> (7 * l)) & 127);
                    for (int j = 0; j < 8; ++j) {
                        y[yOff + j] = db * QuantTables.laneU8(grid, j) * sign(signs, j);
                    }
                    yOff += 8;
                }
            }
        }
    }

    /** {@code dequantize_row_iq2_xs} (llama.cpp ggml-quants.c:2517-2540). */
    static void decodeIq2Xs(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ2_XS_BYTES) {
            final float d = fp16At(b, off + IQ2_XS_D);
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                final int sc = b.get(off + IQ2_XS_SCALES + ib32) & 0xFF;
                final float db0 = d * (0.5f + (sc & 0xf)) * 0.25f;
                final float db1 = d * (0.5f + (sc >>> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    final int q = b.getShort(off + IQ2_XS_QS + 2 * (4 * ib32 + l)) & 0xFFFF;
                    final long grid = QuantTables.iq2xsGrid(q & 511);
                    final int signs = QuantTables.ksignsIq2xs(q >>> 9);
                    final float dl = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; ++j) {
                        y[yOff + j] = dl * QuantTables.laneU8(grid, j) * sign(signs, j);
                    }
                    yOff += 8;
                }
            }
        }
    }

    /** {@code dequantize_row_iq2_s} (llama.cpp ggml-quants.c:2544-2572). */
    static void decodeIq2S(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ2_S_BYTES) {
            final float d = fp16At(b, off + IQ2_S_D);
            int qs = off + IQ2_S_QS;
            int signs = off + IQ2_S_SIGNS;
            final int qh = off + IQ2_S_QH;
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                final int sc = b.get(off + IQ2_S_SCALES + ib32) & 0xFF;
                final float db0 = d * (0.5f + (sc & 0xf)) * 0.25f;
                final float db1 = d * (0.5f + (sc >>> 4)) * 0.25f;
                final int h = b.get(qh + ib32) & 0xFF;
                for (int l = 0; l < 4; ++l) {
                    final float dl = l < 2 ? db0 : db1;
                    final long grid = QuantTables.iq2sGrid((b.get(qs + l) & 0xFF) | ((h << (8 - 2 * l)) & 0x300));
                    final int s = b.get(signs + l) & 0xFF;
                    for (int j = 0; j < 8; ++j) {
                        y[yOff + j] = dl * QuantTables.laneU8(grid, j) * sign(s, j);
                    }
                    yOff += 8;
                }
                qs += 4;
                signs += 4;
            }
        }
    }

    /** {@code dequantize_row_iq3_xxs} (llama.cpp ggml-quants.c:2576-2604). */
    static void decodeIq3Xxs(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ3_XXS_BYTES) {
            final float d = fp16At(b, off + IQ3_XXS_D);
            int qs = off + IQ3_XXS_QS;
            final int sas = off + IQ3_XXS_SAS;
            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                final int aux32 = b.getInt(sas + 4 * ib32);
                final float db = d * (0.5f + (aux32 >>> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    final int signs = QuantTables.ksignsIq2xs((aux32 >>> (7 * l)) & 127);
                    final int grid1 = QuantTables.iq3xxsGrid(b.get(qs + 2 * l) & 0xFF);
                    final int grid2 = QuantTables.iq3xxsGrid(b.get(qs + 2 * l + 1) & 0xFF);
                    for (int j = 0; j < 4; ++j) {
                        y[yOff + j] = db * QuantTables.laneU8(grid1, j) * sign(signs, j);
                        y[yOff + j + 4] = db * QuantTables.laneU8(grid2, j) * sign(signs, j + 4);
                    }
                    yOff += 8;
                }
                qs += 8;
            }
        }
    }

    /** {@code dequantize_row_iq3_s} (llama.cpp ggml-quants.c:2608-2647). */
    static void decodeIq3S(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ3_S_BYTES) {
            final float d = fp16At(b, off + IQ3_S_D);
            int qs = off + IQ3_S_QS;
            int qh = off + IQ3_S_QH;
            int signs = off + IQ3_S_SIGNS;
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                final int sc = b.get(off + IQ3_S_SCALES + ib32 / 2) & 0xFF;
                final float db1 = d * (1 + 2 * (sc & 0xf));
                final float db2 = d * (1 + 2 * (sc >>> 4));
                yOff = iq3sHalf(b, qs, b.get(qh) & 0xFF, signs, db1, y, yOff);
                qs += 8;
                signs += 4;
                yOff = iq3sHalf(b, qs, b.get(qh + 1) & 0xFF, signs, db2, y, yOff);
                qh += 2;
                qs += 8;
                signs += 4;
            }
        }
    }

    /** One 32-element half of an IQ3_S pair: four 8-element groups sharing {@code qh} byte {@code h}. */
    private static int iq3sHalf(ByteBuffer b, int qs, int h, int signs, float db, float[] y, int yOff) {
        for (int l = 0; l < 4; ++l) {
            final int grid1 = QuantTables.iq3sGrid((b.get(qs + 2 * l) & 0xFF) | ((h << (8 - 2 * l)) & 256));
            final int grid2 = QuantTables.iq3sGrid((b.get(qs + 2 * l + 1) & 0xFF) | ((h << (7 - 2 * l)) & 256));
            final int s = b.get(signs + l) & 0xFF;
            for (int j = 0; j < 4; ++j) {
                y[yOff + j] = db * QuantTables.laneU8(grid1, j) * sign(s, j);
                y[yOff + j + 4] = db * QuantTables.laneU8(grid2, j) * sign(s, j + 4);
            }
            yOff += 8;
        }
        return yOff;
    }

    /** {@code dequantize_row_iq1_s} (llama.cpp ggml-quants.c:2651-2674): {@code y = dl * (grid + delta)}. */
    static void decodeIq1S(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ1_S_BYTES) {
            final float d = fp16At(b, off + IQ1_S_D);
            int qs = off + IQ1_S_QS;
            for (int ib = 0; ib < QK_K / 32; ++ib) {
                final int qh = b.getShort(off + IQ1_S_QH + 2 * ib) & 0xFFFF;
                final float dl = d * (2 * ((qh >>> 12) & 7) + 1);
                final float delta = (qh & 0x8000) != 0 ? -QuantTables.IQ1S_DELTA : QuantTables.IQ1S_DELTA;
                for (int l = 0; l < 4; ++l) {
                    final long grid = QuantTables.iq1sGrid((b.get(qs + l) & 0xFF) | (((qh >>> (3 * l)) & 7) << 8));
                    for (int j = 0; j < 8; ++j) {
                        y[yOff + j] = dl * (QuantTables.laneS8(grid, j) + delta);
                    }
                    yOff += 8;
                }
                qs += 4;
            }
        }
    }

    /** {@code dequantize_row_iq1_m} (llama.cpp ggml-quants.c:2676-2724). */
    static void decodeIq1M(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ1_M_BYTES) {
            final int sc = off + IQ1_M_SCALES;
            final float d = iq1mSuperScale(b, sc);
            int qs = off + IQ1_M_QS;
            int qh = off + IQ1_M_QH;
            for (int ib = 0; ib < QK_K / 32; ++ib) {
                final int s = b.getShort(sc + 2 * (ib / 2)) & 0xFFFF;
                final float dl1 = d * (2 * ((s >>> (6 * (ib % 2))) & 0x7) + 1);
                final float dl2 = d * (2 * ((s >>> (6 * (ib % 2) + 3)) & 0x7) + 1);
                final int h0 = b.get(qh) & 0xFF;
                final int h1 = b.get(qh + 1) & 0xFF;
                yOff = iq1mGroup((b.get(qs) & 0xFF) | ((h0 << 8) & 0x700), (h0 & 0x08) != 0, dl1, y, yOff);
                yOff = iq1mGroup((b.get(qs + 1) & 0xFF) | ((h0 << 4) & 0x700), (h0 & 0x80) != 0, dl1, y, yOff);
                yOff = iq1mGroup((b.get(qs + 2) & 0xFF) | ((h1 << 8) & 0x700), (h1 & 0x08) != 0, dl2, y, yOff);
                yOff = iq1mGroup((b.get(qs + 3) & 0xFF) | ((h1 << 4) & 0x700), (h1 & 0x80) != 0, dl2, y, yOff);
                qs += 4;
                qh += 2;
            }
        }
    }

    /** One 8-element IQ1_M group: {@code y = dl * (grid[idx] + delta)}. */
    private static int iq1mGroup(int idx, boolean negDelta, float dl, float[] y, int yOff) {
        final float delta = negDelta ? -QuantTables.IQ1S_DELTA : QuantTables.IQ1S_DELTA;
        final long grid = QuantTables.iq1sGrid(idx);
        for (int j = 0; j < 8; ++j) {
            y[yOff + j] = dl * (QuantTables.laneS8(grid, j) + delta);
        }
        return yOff + 8;
    }

    /** {@code dequantize_row_iq4_nl} (llama.cpp ggml-quants.c:2726-2742): split nibbles into the IQ4_NL codebook. */
    static void decodeIq4Nl(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int half = QK4_NL / 2;
        for (int i = 0; i < nb; i++, off += IQ4_NL_BYTES, yOff += QK4_NL) {
            final float d = fp16At(b, off + IQ4_NL_D);
            final int qs = off + IQ4_NL_QS;
            for (int j = 0; j < half; ++j) {
                final int q = b.get(qs + j) & 0xFF;
                y[yOff + j] = d * QuantTables.kvalueIq4nl(q & 0xf);
                y[yOff + j + half] = d * QuantTables.kvalueIq4nl(q >>> 4);
            }
        }
    }

    /** {@code dequantize_row_iq4_xs} (llama.cpp ggml-quants.c:2744-2765): 6-bit sub-block scales, IQ4_NL codebook. */
    static void decodeIq4Xs(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += IQ4_XS_BYTES) {
            final float d = fp16At(b, off + IQ4_XS_D);
            final int scalesH = b.getShort(off + IQ4_XS_SCALES_H) & 0xFFFF;
            int qs = off + IQ4_XS_QS;
            for (int ib = 0; ib < QK_K / 32; ++ib) {
                final int ls = (((b.get(off + IQ4_XS_SCALES_L + ib / 2) & 0xFF) >>> (4 * (ib % 2))) & 0xf)
                        | (((scalesH >>> (2 * ib)) & 3) << 4);
                final float dl = d * (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    final int q = b.get(qs + j) & 0xFF;
                    y[yOff + j] = dl * QuantTables.kvalueIq4nl(q & 0xf);
                    y[yOff + j + 16] = dl * QuantTables.kvalueIq4nl(q >>> 4);
                }
                yOff += 32;
                qs += 16;
            }
        }
    }
}
