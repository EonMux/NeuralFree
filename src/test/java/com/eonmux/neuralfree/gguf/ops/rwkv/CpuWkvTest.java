package com.eonmux.neuralfree.gguf.ops.rwkv;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the RWKV CPU references {@link CpuWkv} and {@link CpuRwkvMix}: hand-traced recurrences, the
 * llama.cpp state layouts ({@code [h][key][value]} for wkv6/gla, {@code [h][value][key]} for wkv7), equivalence
 * of one prefill call with token-by-token decoding, the token-shift modes, the decays, the RWKV7 key
 * preparation and bonus, and the launch geometry and kernel names of {@link RwkvKernels}.
 */
public class CpuWkvTest {

    private static final float EPS = 1e-5f;

    private static float[] random(Random rnd, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (rnd.nextFloat() * 2.0f - 1.0f) * scale;
        }
        return a;
    }

    private static float[] uniform(Random rnd, int n, float lo, float hi) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = lo + rnd.nextFloat() * (hi - lo);
        }
        return a;
    }

    private static float[] row(float[] a, int t, int width) {
        return Arrays.copyOfRange(a, t * width, (t + 1) * width);
    }

    // ------------------------------------------------------------------ wkv6

    @Test
    public void wkv6MatchesHandTracedSingleHead() {
        // H = 1, S = 2, two tokens, zero initial state.
        float[] k = {1f, 2f, 3f, 4f};
        float[] v = {5f, 6f, 7f, 8f};
        float[] r = {0.5f, 1f, 2f, 0.25f};
        float[] tf = {0.1f, 0.2f};
        float[] td = {0.9f, 0.8f, 0.7f, 0.6f};
        float[] state = new float[4];
        float[] y = new float[4];
        CpuWkv.wkv6(2, 2, 1, k, 0, v, 0, r, 0, tf, td, 0, state, 0, y, 0);

        // token 0: s = 0, so y_j = sum_i r_i * tf_i * k_i * v_j; s[i][j] = k_i * v_j
        float y00 = 0.5f * (0.1f * 1f * 5f) + 1f * (0.2f * 2f * 5f);
        float y01 = 0.5f * (0.1f * 1f * 6f) + 1f * (0.2f * 2f * 6f);
        float[] s1 = {1f * 5f, 1f * 6f, 2f * 5f, 2f * 6f};
        // token 1: y_j = sum_i r_i * (tf_i * k_i * v_j + s[i][j]); s = s * td_i + k_i * v_j
        float y10 = 2f * (0.1f * 3f * 7f + s1[0]) + 0.25f * (0.2f * 4f * 7f + s1[2]);
        float y11 = 2f * (0.1f * 3f * 8f + s1[1]) + 0.25f * (0.2f * 4f * 8f + s1[3]);
        float[] s2 = {s1[0] * 0.7f + 21f, s1[1] * 0.7f + 24f, s1[2] * 0.6f + 28f, s1[3] * 0.6f + 32f};
        assertArrayEquals(new float[]{y00, y01, y10, y11}, y, EPS);
        assertArrayEquals("state is [key][value]", s2, state, EPS);
    }

    @Test
    public void wkv6PrefillEqualsTokenByToken() {
        Random rnd = new Random(11);
        int nHead = 3;
        int s = 8;
        int c = nHead * s;
        int nTokens = 5;
        float[] k = random(rnd, nTokens * c, 1f);
        float[] v = random(rnd, nTokens * c, 1f);
        float[] r = random(rnd, nTokens * c, 1f);
        float[] tf = random(rnd, c, 1f);
        float[] td = uniform(rnd, nTokens * c, 0.5f, 1f);
        float[] init = random(rnd, c * s, 1f);

        float[] state = init.clone();
        float[] y = new float[nTokens * c];
        CpuWkv.wkv6(nTokens, c, nHead, k, 0, v, 0, r, 0, tf, td, 0, state, 0, y, 0);

        float[] state2 = init.clone();
        for (int t = 0; t < nTokens; t++) {
            float[] yt = new float[c];
            CpuWkv.wkv6(1, c, nHead, k, t * c, v, t * c, r, t * c, tf, td, t * c, state2, 0, yt, 0);
            assertArrayEquals("token " + t, row(y, t, c), yt, EPS);
        }
        assertArrayEquals(state, state2, EPS);
    }

    @Test
    public void wkv6UsesOnlyTheGivenStateRow() {
        Random rnd = new Random(12);
        int c = 4;
        int s = 4;
        float[] k = random(rnd, c, 1f);
        float[] v = random(rnd, c, 1f);
        float[] r = random(rnd, c, 1f);
        float[] tf = random(rnd, c, 1f);
        float[] td = uniform(rnd, c, 0.5f, 1f);
        float[] rows = random(rnd, 2 * c * s, 1f);
        float[] before = rows.clone();
        float[] y = new float[c];
        CpuWkv.wkv6(1, c, 1, k, 0, v, 0, r, 0, tf, td, 0, rows, c * s, y, 0);
        assertArrayEquals("row 0 untouched", Arrays.copyOf(before, c * s), Arrays.copyOf(rows, c * s), 0f);
    }

    // ------------------------------------------------------------------ gla

    @Test
    public void glaUpdatesStateBeforeReadout() {
        // H = 1, S = 2, one token, zero state: s[i][j] = k_i * v_j, y_j = sum_i (q_i * scale) * s[i][j].
        float[] k = {1f, 2f};
        float[] v = {3f, 4f};
        float[] q = {0.5f, -1f};
        float[] g = {0.9f, 0.8f};
        float[] state = new float[4];
        float[] y = new float[2];
        float scale = RwkvKernels.glaScale(2);
        CpuWkv.gla(1, 2, 1, RwkvKernels.GLA_G_PER_CHANNEL, scale, k, 0, v, 0, q, 0, g, 0, state, 0, y, 0);
        assertArrayEquals(new float[]{3f, 4f, 6f, 8f}, state, EPS);
        assertArrayEquals(new float[]{(0.5f * 3f - 6f) * scale, (0.5f * 4f - 8f) * scale}, y, EPS);
        assertEquals((float) (1.0 / Math.sqrt(2.0)), scale, 1e-7f);
    }

    @Test
    public void glaPerHeadGateEqualsBroadcastPerChannelGate() {
        Random rnd = new Random(13);
        int nHead = 2;
        int s = 4;
        int c = nHead * s;
        int nTokens = 3;
        float[] k = random(rnd, nTokens * c, 1f);
        float[] v = random(rnd, nTokens * c, 1f);
        float[] q = random(rnd, nTokens * c, 1f);
        float[] gHead = uniform(rnd, nTokens * nHead, 0.5f, 1f);
        float[] gChan = new float[nTokens * c];
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < nHead; h++) {
                Arrays.fill(gChan, t * c + h * s, t * c + (h + 1) * s, gHead[t * nHead + h]);
            }
        }
        float[] init = random(rnd, c * s, 1f);
        float[] s1 = init.clone();
        float[] s2 = init.clone();
        float[] y1 = new float[nTokens * c];
        float[] y2 = new float[nTokens * c];
        CpuWkv.gla(nTokens, c, nHead, RwkvKernels.GLA_G_PER_CHANNEL, 0.5f, k, 0, v, 0, q, 0, gChan, 0, s1, 0, y1, 0);
        CpuWkv.gla(nTokens, c, nHead, RwkvKernels.GLA_G_PER_HEAD, 0.5f, k, 0, v, 0, q, 0, gHead, 0, s2, 0, y2, 0);
        assertArrayEquals(y1, y2, 0f);
        assertArrayEquals(s1, s2, 0f);
    }

    @Test
    public void glaPrefillEqualsTokenByToken() {
        Random rnd = new Random(14);
        int nHead = 2;
        int s = 6;
        int c = nHead * s;
        int nTokens = 4;
        float[] k = random(rnd, nTokens * c, 1f);
        float[] v = random(rnd, nTokens * c, 1f);
        float[] q = random(rnd, nTokens * c, 1f);
        float[] g = uniform(rnd, nTokens * c, 0.5f, 1f);
        float[] init = random(rnd, c * s, 1f);
        float[] state = init.clone();
        float[] y = new float[nTokens * c];
        CpuWkv.gla(nTokens, c, nHead, RwkvKernels.GLA_G_PER_CHANNEL, 0.25f, k, 0, v, 0, q, 0, g, 0, state, 0, y, 0);
        float[] state2 = init.clone();
        for (int t = 0; t < nTokens; t++) {
            float[] yt = new float[c];
            CpuWkv.gla(1, c, nHead, RwkvKernels.GLA_G_PER_CHANNEL, 0.25f,
                    k, t * c, v, t * c, q, t * c, g, t * c, state2, 0, yt, 0);
            assertArrayEquals("token " + t, row(y, t, c), yt, EPS);
        }
        assertArrayEquals(state, state2, EPS);
    }

    // ------------------------------------------------------------------ wkv7

    @Test
    public void wkv7StateIsValueMajor() {
        // a = b = 0, w = 1, zero state: s[i][j] = v_i * k_j and y_i = sum_j s[i][j] * r_j.
        float[] r = {1f, 2f};
        float[] w = {1f, 1f};
        float[] k = {3f, 5f};
        float[] v = {7f, 11f};
        float[] zero = new float[2];
        float[] state = new float[4];
        float[] y = new float[2];
        CpuWkv.wkv7(1, 2, 1, r, 0, w, 0, k, 0, v, 0, zero, 0, zero, 0, state, 0, y, 0);
        assertArrayEquals(new float[]{21f, 35f, 33f, 55f}, state, 0f);
        assertArrayEquals(new float[]{21f + 70f, 33f + 110f}, y, 0f);
    }

    @Test
    public void wkv7MatchesHandTracedDeltaTerm() {
        // One token, S = 2, initial state rows: s = [[1, 2], [3, 4]] (row = value).
        float[] r = {1f, 1f};
        float[] w = {0.5f, 0.25f};
        float[] k = {1f, 0f};
        float[] v = {2f, 3f};
        float[] a = {1f, -1f};
        float[] b = {0.5f, 2f};
        float[] state = {1f, 2f, 3f, 4f};
        float[] y = new float[2];
        CpuWkv.wkv7(1, 2, 1, r, 0, w, 0, k, 0, v, 0, a, 0, b, 0, state, 0, y, 0);
        // row 0: sa = 1 - 2 = -1; s = [1*0.5 + 2*1 - 0.5, 2*0.25 + 0 - 2] = [2, -1.5]
        // row 1: sa = 3 - 4 = -1; s = [3*0.5 + 3*1 - 0.5, 4*0.25 + 0 - 2] = [4, -1]
        assertArrayEquals(new float[]{2f, -1.5f, 4f, -1f}, state, EPS);
        assertArrayEquals(new float[]{0.5f, 3f}, y, EPS);
    }

    @Test
    public void wkv7PrefillEqualsTokenByToken() {
        Random rnd = new Random(15);
        int nHead = 2;
        int s = 8;
        int c = nHead * s;
        int nTokens = 4;
        float[] r = random(rnd, nTokens * c, 1f);
        float[] w = uniform(rnd, nTokens * c, 0.5f, 1f);
        float[] k = random(rnd, nTokens * c, 1f);
        float[] v = random(rnd, nTokens * c, 1f);
        float[] a = random(rnd, nTokens * c, 0.2f);
        float[] b = random(rnd, nTokens * c, 0.2f);
        float[] init = random(rnd, c * s, 1f);
        float[] state = init.clone();
        float[] y = new float[nTokens * c];
        CpuWkv.wkv7(nTokens, c, nHead, r, 0, w, 0, k, 0, v, 0, a, 0, b, 0, state, 0, y, 0);
        float[] state2 = init.clone();
        for (int t = 0; t < nTokens; t++) {
            float[] yt = new float[c];
            int o = t * c;
            CpuWkv.wkv7(1, c, nHead, r, o, w, o, k, o, v, o, a, o, b, o, state2, 0, yt, 0);
            assertArrayEquals("token " + t, row(y, t, c), yt, EPS);
        }
        assertArrayEquals(state, state2, EPS);
    }

    @Test
    public void wkvRejectsInvalidShapes() {
        float[] x = new float[16];
        try {
            CpuWkv.wkv6(1, 6, 4, x, 0, x, 0, x, 0, x, x, 0, x, 0, new float[6], 0);
            fail("C not divisible by H must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        try {
            CpuWkv.wkv7(2, 4, 1, x, 0, x, 0, x, 0, x, 0, x, 0, x, 10, x, 0, new float[8], 0);
            fail("b rows past the array end must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        try {
            CpuWkv.gla(1, 4, 1, 7, 1f, x, 0, x, 0, x, 0, x, 0, x, 0, new float[4], 0);
            fail("unknown g_mode must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    // ------------------------------------------------------------------ token shift and mixing

    @Test
    public void tokenShiftLerpModes() {
        int nEmbd = 2;
        int nTokens = 2;
        float[] cur = {1f, 2f, 3f, 5f};
        float[] lerpX = {0.5f, 0.25f};
        float[] lerpFused = {0.1f, 0.2f, 0.3f, 0.4f};

        // LERP, two mixes; prev(0) = shift state, prev(1) = cur[0].
        float[] shift = {10f, 20f};
        float[] out = new float[2 * nTokens * nEmbd];
        CpuRwkvMix.tokenShiftLerp(nTokens, nEmbd, 2, RwkvKernels.SHIFT_MODE_LERP, cur, 0, shift, 0, null, lerpFused, out, 0);
        float[] sx = {10f - 1f, 20f - 2f, 1f - 3f, 2f - 5f};
        float[] expect = new float[8];
        for (int m = 0; m < 2; m++) {
            for (int i = 0; i < 4; i++) {
                expect[m * 4 + i] = sx[i] * lerpFused[m * 2 + i % 2] + cur[i];
            }
        }
        assertArrayEquals(expect, out, EPS);
        assertArrayEquals("shift state holds the last token", new float[]{3f, 5f}, shift, 0f);

        // LERP_X leaves the shift state; LORA on the same state then stores it.
        float[] shift2 = {10f, 20f};
        float[] xOut = new float[nTokens * nEmbd];
        CpuRwkvMix.tokenShiftLerp(nTokens, nEmbd, 0, RwkvKernels.SHIFT_MODE_LERP_X, cur, 0, shift2, 0, lerpX, null, xOut, 0);
        for (int i = 0; i < 4; i++) {
            assertEquals(sx[i] * lerpX[i % 2] + cur[i], xOut[i], EPS);
        }
        assertArrayEquals(new float[]{10f, 20f}, shift2, 0f);
        float[] lora = {1f, -1f, 2f, -2f, 0.5f, 0.5f, 0.25f, 0.75f};
        float[] loraOut = lora.clone();
        CpuRwkvMix.tokenShiftLerp(nTokens, nEmbd, 2, RwkvKernels.SHIFT_MODE_LORA, cur, 0, shift2, 0, null, lerpFused, loraOut, 0);
        for (int m = 0; m < 2; m++) {
            for (int i = 0; i < 4; i++) {
                assertEquals((lora[m * 4 + i] + lerpFused[m * 2 + i % 2]) * sx[i] + cur[i], loraOut[m * 4 + i], EPS);
            }
        }
        assertArrayEquals(new float[]{3f, 5f}, shift2, 0f);

        // PAIR: slot 0 from lerp_x, slot 1 from lerp_fused[0]; shift state at an offset.
        float[] shift3 = {0f, 0f, 10f, 20f};
        float[] pair = new float[2 * nTokens * nEmbd];
        CpuRwkvMix.tokenShiftLerp(nTokens, nEmbd, 1, RwkvKernels.SHIFT_MODE_PAIR, cur, 0, shift3, 2, lerpX, lerpFused, pair, 0);
        for (int i = 0; i < 4; i++) {
            assertEquals(sx[i] * lerpX[i % 2] + cur[i], pair[i], EPS);
            assertEquals(sx[i] * lerpFused[i % 2] + cur[i], pair[4 + i], EPS);
        }
        assertArrayEquals(new float[]{0f, 0f, 3f, 5f}, shift3, 0f);

        // Input rows at an element offset (cur_off) give the same result as the unshifted rows.
        float[] padded = new float[3 + cur.length];
        System.arraycopy(cur, 0, padded, 3, cur.length);
        float[] shift4 = {10f, 20f};
        float[] offOut = new float[2 * nTokens * nEmbd];
        CpuRwkvMix.tokenShiftLerp(nTokens, nEmbd, 2, RwkvKernels.SHIFT_MODE_LERP, padded, 3, shift4, 0, null, lerpFused, offOut, 0);
        assertArrayEquals(expect, offOut, 0f);
        assertArrayEquals(new float[]{3f, 5f}, shift4, 0f);
    }

    @Test
    public void decaysFollowTheGraphFormulas() {
        float[] in = {-1f, 0f, 0.5f, 2f};
        float[] base = {0.25f, -0.5f};
        float[] d6 = new float[4];
        float[] d7 = new float[4];
        CpuRwkvMix.decay6(4, 2, in, 0, base, d6, 0);
        CpuRwkvMix.decay7(4, 2, in, 0, base, d7, 0);
        for (int i = 0; i < 4; i++) {
            double x = in[i] + base[i % 2];
            assertEquals(Math.exp(-Math.exp(x)), d6[i], 1e-6);
            assertEquals(Math.exp(-0.606531 / (1.0 + Math.exp(-x))), d7[i], 1e-6);
        }
    }

    @Test
    public void kkPrepNormalizesPerHead() {
        int nHead = 2;
        int s = 2;
        float[] k = {3f, 4f, 1f, 0f};
        float[] kK = {1f, 1f, 2f, 2f};
        float[] kA = {0.5f, 0.5f, 1f, 1f};
        float[] a = {0.2f, 0.4f, 0.6f, 0.8f};
        float[] kk = new float[4];
        float[] kOut = new float[4];
        float[] aa = new float[4];
        float[] bb = new float[4];
        CpuRwkvMix.kkPrep(1, nHead, s, k, 0, kK, kA, a, 0, kk, 0, kOut, 0, aa, 0, bb, 0);
        assertArrayEquals(new float[]{0.6f, 0.8f, 1f, 0f}, kk, EPS);
        for (int i = 0; i < 4; i++) {
            float ka = k[i] * kA[i];
            assertEquals(k[i] + (a[i] * ka - ka), kOut[i], EPS);
            assertEquals(-kk[i], aa[i], 0f);
            assertEquals(kk[i] * a[i], bb[i], 0f);
        }
    }

    @Test
    public void kkPrepKeepsZeroHeadsFinite() {
        float[] zero = new float[2];
        float[] ones = {1f, 1f};
        float[] kk = new float[2];
        float[] kOut = new float[2];
        float[] aa = new float[2];
        float[] bb = new float[2];
        CpuRwkvMix.kkPrep(1, 1, 2, zero, 0, ones, ones, ones, 0, kk, 0, kOut, 0, aa, 0, bb, 0);
        assertArrayEquals(new float[]{0f, 0f}, kk, 0f);
    }

    @Test
    public void bonusAddsValueTimesHeadSum() {
        int nHead = 2;
        int s = 2;
        float[] k = {1f, 2f, 3f, 4f};
        float[] r = {1f, 1f, 0.5f, 0.5f};
        float[] rK = {2f, 1f, 1f, -1f};
        float[] v = {1f, 2f, 3f, 4f};
        float[] x = {10f, 10f, 10f, 10f};
        CpuRwkvMix.bonus(1, nHead, s, k, 0, r, 0, rK, v, 0, x, 0);
        float rk0 = 1f * 1f * 2f + 2f * 1f * 1f;
        float rk1 = 3f * 0.5f * 1f - 4f * 0.5f * 1f;
        assertArrayEquals(new float[]{10f + rk0, 10f + 2f * rk0, 10f + 3f * rk1, 10f + 4f * rk1}, x, EPS);
    }

    // ------------------------------------------------------------------ launch geometry

    @Test
    public void wkvLaunchGeometryCoversEveryValue() {
        int[][] cases = {
                // S, parts, local, groups per head
                {16, 1, 32, 1},
                {32, 1, 32, 1},
                {64, 2, 128, 1},
                {96, 4, 256, 2},
                {128, 4, 256, 2},
                {256, 8, 256, 8},
        };
        for (int[] cs : cases) {
            int s = cs[0];
            assertEquals("parts S=" + s, cs[1], RwkvKernels.wkvParts(s));
            assertEquals("local S=" + s, cs[2], RwkvKernels.wkvLocalSize(s));
            assertEquals("global S=" + s, 3L * cs[3] * cs[2], RwkvKernels.wkvGlobalSize(3, s));
            int valuesPerGroup = cs[2] / cs[1];
            assertEquals("every value index covered S=" + s, cs[3], (s + valuesPerGroup - 1) / valuesPerGroup);
            assertTrue("keys covered S=" + s, cs[1] * RwkvKernels.KEYS_PER_ITEM >= s);
        }
        assertEquals(64, RwkvKernels.headRowLocalSize(64));
        assertEquals(32, RwkvKernels.headRowLocalSize(8));
        assertEquals(256, RwkvKernels.headRowLocalSize(300));
        assertEquals(5L * 4 * 64, RwkvKernels.headRowGlobalSize(5, 4, 64));
        assertEquals(512L, RwkvKernels.elementwiseGlobalSize(300));
        assertEquals(1024L, RwkvKernels.tokenShiftGlobalSize(3, 300));
        try {
            RwkvKernels.tokenShiftGlobalSize(1 << 16, 1 << 16);
            fail("a token-shift plane above the int range must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        try {
            RwkvKernels.wkvLocalSize(RwkvKernels.MAX_HEAD_SIZE + 1);
            fail("head size above the maximum must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void everyKernelNameIsDefinedInTheSource() throws IOException {
        String source;
        try (InputStream in = CpuWkvTest.class.getClassLoader().getResourceAsStream("kernels/llm/recurrent/rwkv.cl")) {
            assertNotNull("kernels/llm/recurrent/rwkv.cl is on the classpath", in);
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            for (int n = in.read(buf); n >= 0; n = in.read(buf)) {
                bytes.write(buf, 0, n);
            }
            source = new String(bytes.toByteArray(), StandardCharsets.UTF_8);
        }
        for (String name : RwkvKernels.ALL) {
            assertTrue(name + " is defined in rwkv.cl", source.contains("NF_KERNEL void " + name + "("));
        }
    }
}
