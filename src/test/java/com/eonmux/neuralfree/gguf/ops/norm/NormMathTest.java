package com.eonmux.neuralfree.gguf.ops.norm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

/**
 * Reference values of the CPU norm math (the twins of {@code LLM_RowNorm}, {@code LLM_GroupRms},
 * {@code LLM_GroupNorm}, {@code LLM_ResidualMix} and {@code LLM_MulRowScale}) and the kernel argument lists.
 */
public class NormMathTest {

    private static final float EPS = 1e-6f;

    @Test
    public void rmsNormMatchesGgml() {
        float[] x = {1f, 2f, 3f, 4f};
        float[] y = new float[4];
        NormMath.rowNorm(NormKind.RMS, 0f, x, 0, y, 0, 4, null, 0, null, 0);
        float s = (float) (1.0 / Math.sqrt(7.5));
        assertArrayEquals(new float[]{s, 2 * s, 3 * s, 4 * s}, y, EPS);

        float[] w = {1f, 2f, 0.5f, -1f};
        NormMath.rowNorm(NormKind.RMS, 0f, x, 0, y, 0, 4, w, 0, null, 0);
        assertArrayEquals(new float[]{s, 4 * s, 1.5f * s, -4 * s}, y, EPS);
    }

    @Test
    public void layerNormUsesBiasedVarianceWeightAndBias() {
        float[] x = {1f, 2f, 3f, 4f};
        float[] w = {1f, 2f, 1f, 2f};
        float[] b = {0f, 1f, 0f, 1f};
        float[] y = new float[4];
        NormMath.rowNorm(NormKind.LAYER, 0f, x, 0, y, 0, 4, w, 0, b, 0);
        float s = (float) (1.0 / Math.sqrt(1.25));
        assertArrayEquals(new float[]{-1.5f * s, -0.5f * s * 2 + 1, 0.5f * s, 1.5f * s * 2 + 1}, y, EPS);
    }

    @Test
    public void l2NormsFollowTheirEpsConventions() {
        float[] y = new float[2];
        NormMath.rowNorm(NormKind.L2_CLAMP, 1e-12f, new float[]{3f, 4f}, 0, y, 0, 2, null, 0, null, 0);
        assertArrayEquals(new float[]{0.6f, 0.8f}, y, EPS);

        NormMath.rowNorm(NormKind.L2_CLAMP, 1e-12f, new float[]{0f, 0f}, 0, y, 0, 2, null, 0, null, 0);
        assertArrayEquals(new float[]{0f, 0f}, y, 0f);

        NormMath.rowNorm(NormKind.L2_CLAMP, 10f, new float[]{3f, 4f}, 0, y, 0, 2, null, 0, null, 0);
        assertArrayEquals(new float[]{0.3f, 0.4f}, y, EPS);

        NormMath.rowNorm(NormKind.L2_GDN, 11f, new float[]{3f, 4f}, 0, y, 0, 2, null, 0, null, 0);
        assertArrayEquals(new float[]{0.5f, 4f / 6f}, y, EPS);
    }

    @Test
    public void rowNormKernelTwinUsesWeightRowModulo() {
        float[] x = {3f, 4f, 1f, 1f, 6f, 8f, 2f, 2f};
        float[] w = {1f, 1f, 2f, 3f};
        float[] y = new float[8];
        NormMath.rowNorm(NormKind.L2_GDN, 0f, 4, 2, 2, 2, x, 0, w, null, y, 0);
        float r = (float) Math.sqrt(0.5);
        assertArrayEquals(new float[]{0.6f, 0.8f, 2 * r, 3 * r, 0.6f, 0.8f, 2 * r, 3 * r}, y, EPS);
    }

    @Test
    public void rowNormStridesAndOffsets() {
        float[] x = {9f, 3f, 4f, 9f, 9f, 6f, 8f, 9f};
        float[] y = new float[10];
        NormMath.rowNorm(NormKind.L2_GDN, 0f, 2, 2, 4, 1, x, 1, null, null, y, 2);
        assertArrayEquals(new float[]{0f, 0f, 0.6f, 0.8f, 0f, 0f, 0.6f, 0.8f, 0f, 0f}, y, EPS);
    }

    @Test
    public void rowNormInPlaceEqualsOutOfPlace() {
        float[] x = {0.5f, -1.25f, 3f, 2f, -0.75f, 1f};
        float[] w = {1.5f, 0.5f, -1f, 2f, 1f, 0.25f};
        float[] b = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        for (NormKind k : new NormKind[]{NormKind.RMS, NormKind.LAYER, NormKind.L2_CLAMP, NormKind.L2_GDN}) {
            float[] out = new float[6];
            NormMath.rowNorm(k, 1e-5f, x, 0, out, 0, 6, w, 0, b, 0);
            float[] in = x.clone();
            NormMath.rowNorm(k, 1e-5f, in, 0, in, 0, 6, w, 0, b, 0);
            assertArrayEquals(k.name(), out, in, 0f);
        }
    }

    @Test
    public void groupRmsNormalizesEachGroupWithItsWeights() {
        float[] x = {7f, 3f, 4f, 1f, 1f, 7f};
        float[] w = {1f, 1f, 2f, 2f};
        float[] y = new float[4];
        NormMath.groupRms(1, 4, 2, 0f, x, 1, 4, w, y, 0);
        float r = (float) (1.0 / Math.sqrt(12.5));
        assertArrayEquals(new float[]{3 * r, 4 * r, 2f, 2f}, y, EPS);
    }

    @Test
    public void groupRmsInPlaceOverDenseRows() {
        float[] x = {3f, 4f, 2f, 2f, 1f, 1f, 6f, 8f};
        float[] w = {1f, 1f, 1f, 1f};
        float[] expected = new float[8];
        NormMath.groupRms(2, 4, 2, 0f, x, 0, 4, w, expected, 0);
        NormMath.groupRms(2, 4, 2, 0f, x, 0, 4, w, x, 0);
        assertArrayEquals(expected, x, 0f);
    }

    @Test
    public void groupNormMatchesGgmlGroups() {
        float[] x = {1f, 2f, 3f, 4f, 5f, 7f};
        float[] w = {1f, 2f, 3f};
        float[] b = {0f, 0f, 1f};
        float[] y = new float[6];
        NormMath.groupNorm(3, 2, 2, 0f, x, 0, w, b, y, 0);
        float s0 = (float) (1.0 / Math.sqrt(1.25));
        assertArrayEquals(new float[]{-1.5f * s0, -0.5f * s0, 0.5f * s0 * 2, 1.5f * s0 * 2, -3f + 1f, 3f + 1f}, y,
                EPS);
    }

    @Test
    public void groupNormSkipsEmptyGroups() {
        float[] x = {1f, 3f, 2f, 6f};
        float[] w = {1f, 1f};
        float[] b = {0f, 0f};
        float[] y = {9f, 9f, 9f, 9f};
        NormMath.groupNorm(2, 2, 4, 0f, x, 0, w, b, y, 0);
        assertArrayEquals(new float[]{-1f, 1f, -1f, 1f}, y, EPS);
    }

    @Test
    public void residualMixModes() {
        float[] inp = {1f, 2f, 3f, 4f};
        float[] cur = {10f, 20f, 30f, 40f};
        float[] out = new float[4];
        NormMath.residualMix(MixMode.ADD, 4, 0f, inp, 0, cur, 0, null, null, 0, out, 0);
        assertArrayEquals(new float[]{11f, 22f, 33f, 44f}, out, 0f);
        NormMath.residualMix(MixMode.SCALED, 4, 0.5f, inp, 0, cur, 0, null, null, 0, out, 0);
        assertArrayEquals(new float[]{6f, 12f, 18f, 24f}, out, 0f);
        NormMath.residualMix(MixMode.NORMED_INPUT_SCALED, 4, 2f, inp, 0, cur, 0, null, null, 0, out, 0);
        assertArrayEquals(new float[]{12f, 24f, 36f, 48f}, out, 0f);
        float[] alpha = {1f, 0.5f};
        float[] beta = {0.1f, 2f};
        NormMath.residualMix(MixMode.ALPHA_BETA, 4, 0f, inp, 0, cur, 0, alpha, beta, 2, out, 0);
        assertArrayEquals(new float[]{2f, 41f, 6f, 82f}, out, EPS);

        float[] x = inp.clone();
        NormMath.residualMix(MixMode.ADD, 2, 0f, x, 2, cur, 0, null, null, 0, x, 2);
        assertArrayEquals(new float[]{1f, 2f, 13f, 24f}, x, 0f);
    }

    @Test
    public void mulRowScaleBroadcastsScalarOrColumns() {
        float[] x = {1f, 2f, 3f, 4f};
        NormMath.mulRowScale(2, 2, new float[]{0f, 3f}, 1, 1, x, 0);
        assertArrayEquals(new float[]{3f, 6f, 9f, 12f}, x, 0f);
        NormMath.mulRowScale(2, 2, new float[]{1f, -1f}, 0, 2, x, 0);
        assertArrayEquals(new float[]{3f, -6f, 9f, -12f}, x, 0f);
    }

    @Test
    public void invalidArgumentsAreRejected() {
        float[] x = new float[8];
        expectIae(() -> NormMath.rowNorm(NormKind.GROUP, 0f, x, 0, x, 0, 4, null, 0, null, 0));
        expectIae(() -> NormMath.rowNorm(NormKind.RMS, -1f, x, 0, x, 0, 4, null, 0, null, 0));
        expectIae(() -> NormMath.rowNorm(NormKind.RMS, Float.NaN, x, 0, x, 0, 4, null, 0, null, 0));
        expectIae(() -> NormMath.rowNorm(NormKind.RMS, 0f, x, 0, x, 2, 4, null, 0, null, 0));
        expectIae(() -> NormMath.rowNorm(NormKind.RMS, 0f, x, 6, x, 0, 4, null, 0, null, 0));
        expectIae(() -> NormMath.rowNorm(NormKind.RMS, 0f, 2, 4, 3, 1, x, 0, null, null, new float[8], 0));
        expectIae(() -> NormMath.rowNorm(NormKind.RMS, 0f, 2, 2, 2, 2, x, 0, new float[3], null, new float[4], 0));
        expectIae(() -> NormMath.groupRms(1, 5, 2, 0f, x, 0, 5, new float[5], new float[5], 0));
        expectIae(() -> NormMath.groupRms(2, 2, 1, 0f, x, 0, 3, new float[2], x, 1));
        expectIae(() -> NormMath.groupNorm(2, 2, 1, 0f, x, 0, new float[2], new float[2], x, 2));
        expectIae(() -> NormMath.residualMix(MixMode.ADD, 4, 0f, x, 0, x, 4, null, null, 0, x, 1));
        expectIae(() -> NormMath.residualMix(MixMode.ALPHA_BETA, 4, 0f, x, 0, x, 4, null, null, 0, new float[4], 0));
        expectIae(() -> NormMath.mulRowScale(2, 4, new float[4], 0, 3, x, 0));
    }

    @Test
    public void kernelCodesMatchTheEnums() {
        assertEquals(0, NormKind.RMS.rowNormMode());
        assertEquals(1, NormKind.LAYER.rowNormMode());
        assertEquals(2, NormKind.L2_CLAMP.rowNormMode());
        assertEquals(3, NormKind.L2_GDN.rowNormMode());
        for (MixMode m : MixMode.values()) {
            assertSame(m, MixMode.fromCode(m.code()));
        }
        assertSame(MixMode.ADD, MixMode.forResidual(ResidualKind.STANDARD));
        assertSame(MixMode.SCALED, MixMode.forResidual(ResidualKind.SCALED));
        assertSame(MixMode.NORMED_INPUT_SCALED, MixMode.forResidual(ResidualKind.NORMED_INPUT_SCALED));
        expectIae(() -> MixMode.forResidual(ResidualKind.PARALLEL));
        expectIae(() -> MixMode.fromCode(4));
    }

    @Test
    public void kernelArgumentListsFollowTheSignatures() {
        GpuBuffer x = buffer(64);
        GpuBuffer w = buffer(8);
        GpuBuffer ph = buffer(1);
        Object[] a = NormKernels.rowNormArgs(NormKind.LAYER, 1e-5f, 4, 8, 8, x, 0, x, 0, w, null, 1, ph);
        assertEquals(NormKernels.ROW_NORM_ARG_COUNT, a.length);
        assertEquals(4, a[NormKernels.ARG_ROW_NORM_N_ROWS]);
        assertEquals(1, a[NormKernels.ARG_ROW_NORM_MODE]);
        assertEquals(NormKernels.FLAG_HAS_W, a[NormKernels.ARG_ROW_NORM_FLAGS]);
        assertSame(w, a[NormKernels.ARG_ROW_NORM_W]);
        assertSame(ph, a[NormKernels.ARG_ROW_NORM_B]);
        assertSame(x, a[NormKernels.ARG_ROW_NORM_Y]);
        expectIae(() -> NormKernels.rowNormArgs(NormKind.RMS, 0f, 4, 8, 8, x, 0, x, 4, null, null, 1, ph));
        expectIae(() -> NormKernels.rowNormArgs(NormKind.RMS, 0f, 4, 8, 8, x, 0, x, 0, w, null, 2, ph));

        Object[] m = NormKernels.residualMixArgs(MixMode.ALPHA_BETA, 16, 0f, x, 0, x, 16, w, w, 8, x, 0, ph);
        assertEquals(NormKernels.RESIDUAL_MIX_ARG_COUNT, m.length);
        assertEquals(2, m[NormKernels.ARG_MIX_MODE]);
        assertEquals(8, m[NormKernels.ARG_MIX_AB_LEN]);
        Object[] m0 = NormKernels.residualMixArgs(MixMode.ADD, 16, 0f, x, 0, x, 16, null, null, 0, x, 0, ph);
        assertEquals(1, m0[NormKernels.ARG_MIX_AB_LEN]);
        assertSame(ph, m0[NormKernels.ARG_MIX_ALPHA]);
        expectIae(() -> NormKernels.residualMixArgs(MixMode.ALPHA_BETA, 12, 0f, x, 0, x, 16, w, w, 8, x, 0, ph));

        assertEquals(NormKernels.GROUP_RMS_ARG_COUNT,
                NormKernels.groupRmsArgs(2, 8, 2, 1e-5f, x, 0, 8, w, x).length);
        expectIae(() -> NormKernels.groupRmsArgs(2, 8, 2, 1e-5f, x, 4, 8, w, x));
        assertEquals(NormKernels.GROUP_NORM_ARG_COUNT,
                NormKernels.groupNormArgs(8, 4, 2, 1e-5f, x, w, w, buffer(32)).length);
        expectIae(() -> NormKernels.groupNormArgs(8, 4, 2, 1e-5f, x, w, w, x));
        assertEquals(NormKernels.MUL_ROW_SCALE_ARG_COUNT, NormKernels.mulRowScaleArgs(8, 8, w, 8, x).length);

        assertArrayEquals(new long[]{4 * 64}, NormKernels.reduceGlobal(4, 64));
        assertArrayEquals(new long[]{64}, NormKernels.reduceGlobal(0, 64));
        expectIae(() -> NormKernels.reduceGlobal(4, 48));
        expectIae(() -> NormKernels.reduceGlobal(4, 512));
    }

    private static GpuBuffer buffer(long floats) {
        return new GpuBuffer() {
            @Override
            public long sizeInBytes() {
                return floats * Float.BYTES;
            }

            @Override
            public long bufferId() {
                return 0;
            }

            @Override
            public void close() {
            }
        };
    }

    private static void expectIae(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // the rejection under test
        }
    }
}
