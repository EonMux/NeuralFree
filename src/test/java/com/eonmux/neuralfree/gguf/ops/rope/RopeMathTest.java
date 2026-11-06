package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link RopeMath}: frequency and YaRN formulas against llama.cpp, the context-level attention factor,
 * and the pair geometry, section selection and exponents of ggml {@code ggml_mrope_cache_init}.
 */
public class RopeMathTest {

    private static final float EPS = 1e-6f;

    @Test
    public void thetaScaleMatchesPowf() {
        assertEquals((float) Math.pow(10000f, -2.0f / 128), RopeMath.thetaScale(10000f, 128), 0f);
        RopeSpec spec = RopeSpec.builder(RopeType.NEOX).nDims(128).freqBase(1000000f).build();
        assertEquals(spec.thetaScale(), RopeMath.thetaScale(1000000f, 128), 0f);
    }

    @Test
    public void corrDimsMatchGgml() {
        float[] dims = RopeMath.corrDims(128, 4096, 10000f, 32f, 1f);
        // ggml_rope_yarn_corr_dim(128, 4096, 32, 10000) = 128 * ln(4096 / (64 pi)) / (2 ln 10000) = 20.94 -> floor 20
        assertEquals(20f, dims[0], 0f);
        // beta_slow 1: 128 * ln(4096 / (2 pi)) / (2 ln 10000) = 45.03 -> ceil 46
        assertEquals(46f, dims[1], 0f);
    }

    @Test
    public void yarnRampEdges() {
        assertEquals(1f, RopeMath.yarnRamp(20f, 46f, 0), 0f);
        assertEquals(1f, RopeMath.yarnRamp(20f, 46f, 20), 0f);
        assertEquals(0f, RopeMath.yarnRamp(20f, 46f, 46), 0f);
        assertEquals(0f, RopeMath.yarnRamp(20f, 46f, 63), 0f);
        assertEquals(0.5f, RopeMath.yarnRamp(20f, 46f, 33), EPS);
        // degenerate span uses 0.001 as the denominator
        assertEquals(0f, RopeMath.yarnRamp(5f, 5f, 6), 0f);
    }

    @Test
    public void angleAndMscale() {
        assertEquals(0.25f * 3f, RopeMath.angle(3f, 0.25f, 0f, 20f, 46f, 10), 0f);
        // ramp 1 below corr_lo: pure extrapolation
        assertEquals(3f, RopeMath.angle(3f, 0.25f, 1f, 20f, 46f, 10), EPS);
        // ramp 0 above corr_hi: pure interpolation
        assertEquals(0.75f, RopeMath.angle(3f, 0.25f, 1f, 20f, 46f, 50), EPS);
        assertEquals(2f, RopeMath.rotationMscale(2f, 0.25f, 0f), 0f);
        assertEquals(2f * (1f + 0.1f * (float) Math.log(4f)), RopeMath.rotationMscale(2f, 0.25f, 1f), EPS);

        float[] cs = new float[2];
        RopeMath.cosSin(1.5f, 1f, 0f, 1f, 0f, 0f, 0, cs, 0);
        assertEquals((float) Math.cos(1.5), cs[0], EPS);
        assertEquals((float) Math.sin(1.5), cs[1], EPS);
    }

    @Test
    public void yarnAttnFactorCancelsKernelMscale() {
        assertEquals(1f, RopeMath.defaultExtFactor(RopeScalingType.YARN), 0f);
        assertEquals(0f, RopeMath.defaultExtFactor(RopeScalingType.LINEAR), 0f);
        // no log multiplier: get_mscale(4, 1) / (1 + 0.1 ln 4) == 1
        assertEquals(1f, RopeMath.yarnAttnFactor(1f, 1f, 0.25f, 0f, false, 1f), EPS);
        // without extrapolation the configured factor is kept, then scaled by rope.attn_factor
        assertEquals(1.5f, RopeMath.yarnAttnFactor(0.75f, 0f, 0.25f, 0.1f, false, 2f), EPS);
        // DeepSeek-V2 with mscale_all_dims != 1: mscale == mscale_all_dims, ratio 1
        float expected = 1f / (1f + 0.1f * (float) Math.log(40f));
        assertEquals(expected, RopeMath.yarnAttnFactor(1f, 1f, 1f / 40f, 0.707f, true, 1f), EPS);
        // other archs keep mscale 1
        float ratio = RopeMath.yarnGetMscale(40f, 1f) / RopeMath.yarnGetMscale(40f, 0.707f);
        assertEquals(ratio * expected, RopeMath.yarnAttnFactor(1f, 1f, 1f / 40f, 0.707f, false, 1f), EPS);
    }

    @Test
    public void deepseekKqMscale() {
        float fs = 1f / 40f;
        float org = 0.8f * (1f + 0.1f * (float) Math.log(40f));
        assertEquals(org * (1f + 0.1f * 0.707f * (float) Math.log(40f)), RopeMath.deepseekKqMscale(0.8f, fs, 0.707f),
                EPS);
    }

    @Test
    public void pairGeometry() {
        assertEquals(64, RopeMath.pairCount(RopeType.NEOX, 128));
        assertEquals(64, RopeMath.pairCount(RopeType.VISION, 64));
        assertEquals(10, RopeMath.pairFirst(RopeType.NORM, 5));
        assertEquals(5, RopeMath.pairFirst(RopeType.IMROPE, 5));
        assertEquals(1, RopeMath.pairSpan(RopeType.NORM, 128));
        assertEquals(64, RopeMath.pairSpan(RopeType.MROPE, 128));
        assertEquals(64, RopeMath.pairSpan(RopeType.VISION, 64));
        assertEquals(128, RopeMath.rotatedWidth(RopeType.VISION, 64));
        assertEquals(4, RopeMath.positionsPerToken(RopeType.IMROPE));
        assertEquals(4, RopeMath.positionsPerToken(RopeType.VISION));
        assertEquals(1, RopeMath.positionsPerToken(RopeType.NEOX));
        assertEquals(1, RopeMath.positionsPerToken(RopeType.NONE));
    }

    /** qwen3vl IMROPE sections [24, 20, 20, 0]; ops.cpp:6338-6347. */
    @Test
    public void imropeSectorsOfQwen3vl() {
        int[] s = {24, 20, 20, 0};
        assertEquals(0, RopeMath.section(RopeType.IMROPE, 0, s));
        assertEquals(1, RopeMath.section(RopeType.IMROPE, 1, s));
        assertEquals(2, RopeMath.section(RopeType.IMROPE, 2, s));
        assertEquals(1, RopeMath.section(RopeType.IMROPE, 58, s));
        assertEquals(2, RopeMath.section(RopeType.IMROPE, 59, s));
        assertEquals(0, RopeMath.section(RopeType.IMROPE, 60, s));
        // sector 61: 61 % 3 == 1 but 61 >= 3 * 20, and 61 % 3 is neither 2 nor 0 -> e section
        assertEquals(3, RopeMath.section(RopeType.IMROPE, 61, s));
        assertEquals(3, RopeMath.section(RopeType.IMROPE, 62, s));
        assertEquals(0, RopeMath.section(RopeType.IMROPE, 63, s));
        assertEquals(61, RopeMath.thetaExponent(RopeType.IMROPE, 61, 3, s));
    }

    /** qwen2vl MROPE sections [16, 24, 24, 0]: contiguous sections, exponent = pair index. */
    @Test
    public void mropeContiguousSections() {
        int[] s = {16, 24, 24, 0};
        assertEquals(0, RopeMath.section(RopeType.MROPE, 15, s));
        assertEquals(1, RopeMath.section(RopeType.MROPE, 16, s));
        assertEquals(1, RopeMath.section(RopeType.MROPE, 39, s));
        assertEquals(2, RopeMath.section(RopeType.MROPE, 40, s));
        assertEquals(2, RopeMath.section(RopeType.MROPE, 63, s));
        assertEquals(40, RopeMath.thetaExponent(RopeType.MROPE, 40, 2, s));
        assertEquals(0, RopeMath.section(RopeType.NEOX, 40, s));
    }

    /** VISION restarts the exponent at each section start, every cycle (indep_sects). */
    @Test
    public void visionExponentsRestartPerSection() {
        int[] s = {16, 16, 0, 0};
        assertEquals(0, RopeMath.thetaExponent(RopeType.VISION, 0, RopeMath.section(RopeType.VISION, 0, s), s));
        assertEquals(1, RopeMath.section(RopeType.VISION, 17, s));
        assertEquals(1, RopeMath.thetaExponent(RopeType.VISION, 17, 1, s));
        assertEquals(0, RopeMath.section(RopeType.VISION, 33, s));
        assertEquals(1, RopeMath.thetaExponent(RopeType.VISION, 33, 0, s));
        assertEquals(1, RopeMath.section(RopeType.VISION, 50, s));
        assertEquals(2, RopeMath.thetaExponent(RopeType.VISION, 50, 1, s));
        // s0 == 0: h never restarts (ggml else-if chain), so its exponent keeps growing across cycles
        int[] z = {0, 8, 0, 0};
        assertEquals(1, RopeMath.section(RopeType.VISION, 9, z));
        assertEquals(9, RopeMath.thetaExponent(RopeType.VISION, 9, 1, z));
    }

    @Test
    public void shiftTypes() {
        assertEquals(RopeType.NORM, RopeMath.shiftType(RopeType.NORM));
        assertEquals(RopeType.NEOX, RopeMath.shiftType(RopeType.NEOX));
        assertEquals(RopeType.NEOX, RopeMath.shiftType(RopeType.MROPE));
        assertEquals(RopeType.NEOX, RopeMath.shiftType(RopeType.IMROPE));
        try {
            RopeMath.shiftType(RopeType.VISION);
            fail("VISION caches cannot be shifted");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    // ------------------------------------------------------------------ dynamic NTK scaling

    /** vLLM dynamic_ntk_scaling_rope.py: base * ((factor * max_pos / max_trained) - (factor - 1))^(d / (d - 2)). */
    @Test
    public void dynamicNtkFactorBaseMatchesVllm() {
        float base = 10000f;
        float factor = 4f;
        int extended = 16384;
        int trained = 8192;
        int dims = 128;
        double ratio = (double) factor * extended / trained - (factor - 1.0);
        float expected = (float) (base * Math.pow(ratio, dims / (double) (dims - 2)));
        assertEquals(expected, RopeMath.dynamicNtkBase(base, factor, extended, trained, dims), expected * 1e-6f);

        // the extended length is four times the trained length in the reference formula above
        assertEquals(5.0, ratio, 0.0);
        // at the trained length the factor cancels, so the base is the trained one
        assertEquals(base, RopeMath.dynamicNtkBase(base, factor, trained, trained, dims), 1e-3f);
        // a factor of 1 scales by the extension ratio alone
        float byRatio = (float) (base * Math.pow(4.0, dims / (double) (dims - 2)));
        assertEquals(byRatio, RopeMath.dynamicNtkBase(base, 1f, 4 * trained, trained, dims), byRatio * 1e-6f);
    }

    /** vLLM dynamic_ntk_alpha_rope.py: base * alpha^(d / (d - 2)). */
    @Test
    public void dynamicAlphaBaseMatchesVllm() {
        float base = 10000f;
        float alpha = 1000f;
        int dims = 128;
        float expected = (float) (base * Math.pow(alpha, dims / (double) (dims - 2)));
        assertEquals(expected, RopeMath.dynamicAlphaBase(base, alpha, dims), expected * 1e-6f);
        // a head dim of 64 grows by a smaller power of the same alpha
        float d64 = (float) (base * Math.pow(alpha, 64.0 / 62.0));
        assertEquals(d64, RopeMath.dynamicAlphaBase(base, alpha, 64), d64 * 1e-6f);
    }

    @Test
    public void dynamicNtkRejectsUnusableParameters() {
        assertThrows(IllegalArgumentException.class, () -> RopeMath.dynamicNtkBase(10000f, 0f, 8192, 4096, 128));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.dynamicNtkBase(10000f, 4f, 0, 4096, 128));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.dynamicNtkBase(10000f, 4f, 8192, 0, 128));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.dynamicNtkBase(10000f, 4f, 8192, 4096, 2));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.dynamicAlphaBase(10000f, 0f, 128));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.dynamicAlphaBase(10000f, 1000f, 2));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.dynamicAlphaBase(0f, 1000f, 128));
    }

    // ------------------------------------------------------------------ fixed / mixed NTK scaling

    /**
     * vLLM ntk_scaling_rope.py fixed form: {@code base *= factor}, then every frequency is divided by
     * {@code factor^(2/rotary_dim)}. Relative to the trained {@code theta_scale^i}, the divisor of pair {@code i} is
     * {@code factor^(2(i+1)/rotary_dim)}.
     */
    @Test
    public void fixedNtkFactorsMatchVllm() {
        float factor = 8f;
        int dims = 64;
        float[] f = RopeMath.ntkFactors(factor, Float.NaN, dims);
        assertEquals(dims / 2, f.length);
        for (int i = 0; i < f.length; i++) {
            float expected = (float) Math.pow(factor, 2.0 * (i + 1) / dims);
            assertEquals("pair " + i, expected, f[i], expected * 1e-6f);
        }
        // the reference chain: inv_freq from the scaled base, then divided by factor^(2/d)
        float base = 10000f;
        float theta = RopeMath.thetaScale(base, dims);
        float scaled = RopeMath.thetaScale(base * factor, dims);
        for (int i = 0; i < f.length; i++) {
            double reference = Math.pow(scaled, i) / Math.pow(factor, 2.0 / dims);
            assertEquals("pair " + i, reference, Math.pow(theta, i) / f[i], reference * 1e-6);
        }
    }

    /** vLLM ntk_scaling_rope.py mixed form: {@code a = ln(factor)/(d/2)^b}, divisor {@code exp(a * (i+1)^b)}. */
    @Test
    public void mixedNtkFactorsMatchVllm() {
        float factor = 4f;
        float mixedB = 0.25f;
        int dims = 128;
        float[] f = RopeMath.ntkFactors(factor, mixedB, dims);
        assertEquals(dims / 2, f.length);
        double a = Math.log(factor) / Math.pow(dims / 2.0, mixedB);
        for (int i = 0; i < f.length; i++) {
            double expected = Math.exp(a * Math.pow(i + 1, mixedB));
            assertEquals("pair " + i, expected, f[i], expected * 1e-6);
        }
        // mixed_b 1 divides pair i+1 by factor^((i+1)/(d/2))
        float[] linear = RopeMath.ntkFactors(factor, 1f, dims);
        for (int i = 0; i < linear.length; i++) {
            double expected = Math.pow(factor, (i + 1) / (dims / 2.0));
            assertEquals("pair " + i, expected, linear[i], expected * 1e-6);
        }
    }

    @Test
    public void ntkFactorsRejectUnusableParameters() {
        assertThrows(IllegalArgumentException.class, () -> RopeMath.ntkFactors(0f, Float.NaN, 128));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.ntkFactors(Float.POSITIVE_INFINITY, Float.NaN,
                128));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.ntkFactors(2f, Float.NaN, 2));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.ntkFactors(2f, Float.NaN, 7));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.ntkFactors(2f, Float.POSITIVE_INFINITY, 8));
    }

    // ------------------------------------------------------------------ Fourier position embedding

    /** vLLM fope.py {@code _compute_inv_freq}: the first {@code num_inv_freq} frequencies of the full list. */
    @Test
    public void fopeInvFreqKeepsTheLeadingFrequencies() {
        float base = 10000f;
        int rotary = 128;
        float[] f = RopeMath.fopeInvFreq(base, rotary, 8, 4096);
        assertEquals(8, f.length);
        for (int i = 0; i < 8; i++) {
            assertEquals((float) Math.pow(base, -2.0 * i / rotary), f[i], 1e-9f);
        }
    }

    /** Without {@code num_inv_freq} the frequencies above {@code 2 pi / max_position} are kept. */
    @Test
    public void fopeInvFreqSelectsByPeriod() {
        float base = 10000f;
        int rotary = 128;
        int context = 4096;
        float[] f = RopeMath.fopeInvFreq(base, rotary, 0, context);
        float threshold = (float) (2.0 * Math.PI / context);
        int expected = 0;
        for (int i = 0; i < rotary / 2; i++) {
            if (Math.pow(base, -2.0 * i / rotary) > threshold) {
                expected++;
            }
        }
        assertEquals(expected, f.length);
        for (int i = 0; i < f.length; i++) {
            assertTrue("frequency " + i + " is above the threshold", f[i] > threshold);
        }
        assertThrows(IllegalArgumentException.class, () -> RopeMath.fopeInvFreq(base, rotary, rotary, context));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.fopeInvFreq(base, 7, 1, context));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.fopeInvFreq(0f, rotary, 1, context));
        assertThrows(IllegalArgumentException.class, () -> RopeMath.fopeInvFreq(base, rotary, 1, 0));
    }
}
