package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.function.IntUnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Model-free parts of the dflash drafter checked against hand-computed values: the DFlash2 convolution, the
 * lattice walk, the target-feature gathering, the draft KV cache bookkeeping, the metadata flags and the
 * sampler probability.
 */
public class DflashDrafterTest {

    private static final float EPS = 1e-5f;

    // ------------------------------------------------------------------ DflashConv (dflash.cpp:483-546)

    /** hidden 4, kernel 2, group 2: two taps, weights = dynamic[group] + base[channel]. */
    @Test
    public void convInputSideMixesTapsWithGroupCoefficients() {
        DflashConv conv = new DflashConv(4, 2, 2);
        assertEquals(8, conv.dynamicWidth());
        float[] x = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] dyn = {0.1f, 0.2f, 0.3f, 0.4f, 1, 1, 1, 1, 0.5f, 0.6f, 0.7f, 0.8f, 2, 2, 2, 2};
        float[] base = new float[conv.baseLength()];
        for (int c = 0; c < 4; c++) {
            base[c] = 1.0f;
        }
        float[] out = new float[8];
        conv.apply(x, 0, dyn, 0, base, 0, 2, out, 0);
        assertArrayEquals(new float[]{1.1f, 2.2f, 3.6f, 4.8f, 8.2f, 10.4f, 13.6f, 16.0f}, out, EPS);
    }

    @Test
    public void convOutputSideReadsSecondHalfOfCoefficients() {
        DflashConv conv = new DflashConv(4, 2, 2);
        float[] x = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] dyn = {0.1f, 0.2f, 0.3f, 0.4f, 1, 1, 1, 1, 0.5f, 0.6f, 0.7f, 0.8f, 2, 2, 2, 2};
        float[] out = new float[8];
        conv.apply(x, 0, dyn, 0, new float[conv.baseLength()], 1, 2, out, 0);
        assertArrayEquals(new float[]{1, 2, 3, 4, 12, 16, 20, 24}, out, EPS);
    }

    @Test
    public void convSkipsTapsBeyondTheBlock() {
        DflashConv conv = new DflashConv(2, 3, 1);
        float[] x = {2, 3};
        float[] dyn = new float[conv.dynamicWidth()];
        dyn[0] = 0.5f;
        dyn[1] = 0.25f;
        float[] out = new float[2];
        conv.apply(x, 0, dyn, 0, new float[conv.baseLength()], 0, 1, out, 0);
        assertArrayEquals(new float[]{1.0f, 0.75f}, out, EPS);
    }

    @Test
    public void convRejectsIndivisibleGroups() {
        assertThrows(IllegalArgumentException.class, () -> new DflashConv(6, 2, 4));
    }

    // ------------------------------------------------------------------ lattice walk (speculative.cpp:1291-1320)

    private static DflashSelector.Lattice lattice() {
        DflashSelector.Lattice l = new DflashSelector.Lattice(2, 3);
        l.setCandidate(1, 0, 10);
        l.setCandidate(1, 1, 11);
        l.setCandidate(2, 0, 20);
        l.setCandidate(2, 1, 21);
        l.setScore(1, 0, 0, 1.0f);
        l.setScore(1, 1, 0, 3.0f);
        l.setScore(2, 0, 0, 0.0f);
        l.setScore(2, 1, 0, 9.0f);
        l.setScore(2, 0, 1, 5.0f);
        l.setScore(2, 1, 1, 2.0f);
        return l;
    }

    @Test
    public void walkFollowsThePredecessorRow() {
        assertArrayEquals(new int[]{11, 20}, DflashSelector.walk(lattice(), 2, 0.0f));
        assertArrayEquals(new int[]{11}, DflashSelector.walk(lattice(), 1, 0.0f));
    }

    @Test
    public void walkStopsBelowPMin() {
        // row 1: 1 / (1 + e^-2) = 0.8808; row 2 after candidate 1: 1 / (1 + e^-3) = 0.9526
        assertArrayEquals(new int[0], DflashSelector.walk(lattice(), 2, 0.9f));
        assertArrayEquals(new int[]{11, 20}, DflashSelector.walk(lattice(), 2, 0.85f));
    }

    @Test
    public void walkRejectsTooManySteps() {
        assertThrows(IllegalArgumentException.class, () -> DflashSelector.walk(lattice(), 3, 0.0f));
    }

    @Test
    public void firstMaxPrefersTheLowerIndexOnTies() {
        assertEquals(1, DflashSelector.firstMax(new float[]{0, 4, 4, 1}, 0, 4));
        assertEquals(0, DflashSelector.firstMax(new float[]{9, 4, 4, 1}, 1, 3));
    }

    // ------------------------------------------------------------------ target features

    @Test
    public void featuresConcatenateTargetLayersAndCollapseStreams() {
        final float[][] got = new float[1][];
        final int[] rows = new int[1];
        DflashTargetFeatures f = new DflashTargetFeatures(new int[]{2, 0}, 2, (feats, n) -> {
            got[0] = java.util.Arrays.copyOf(feats, n * 4);
            rows[0] = n;
        });
        assertTrue(f.wants(0));
        assertTrue(f.wants(2));
        assertFalse(f.wants(1));
        f.onHidden(0, new float[]{1, 2, 3, 4}, 2, 2);
        assertEquals(null, got[0]);
        f.onHidden(2, new float[]{1, 3, 5, 7, 2, 2, 4, 4}, 2, 4);
        assertEquals(2, rows[0]);
        assertArrayEquals(new float[]{3, 5, 1, 2, 3, 3, 3, 4}, got[0], EPS);
    }

    @Test
    public void featuresRestartWhenACallIsIncomplete() {
        final int[] calls = new int[1];
        DflashTargetFeatures f = new DflashTargetFeatures(new int[]{1, 3}, 1, (feats, n) -> calls[0]++);
        f.onHidden(1, new float[]{1, 2}, 2, 1);
        f.onHidden(1, new float[]{1}, 1, 1);
        f.onHidden(3, new float[]{5}, 1, 1);
        assertEquals(1, calls[0]);
    }

    @Test
    public void featuresRejectRowsOfAnotherWidth() {
        DflashTargetFeatures f = new DflashTargetFeatures(new int[]{0}, 3, (feats, n) -> { });
        assertThrows(IllegalStateException.class, () -> f.onHidden(0, new float[4], 1, 4));
    }

    // ------------------------------------------------------------------ draft KV cache

    @Test
    public void cacheTrimForgetsPositionsFromTheBound() {
        DflashKvCache c = new DflashKvCache(8, 2, 0, new boolean[]{false}, false, false, new int[]{1}, 2,
                KvDType.F32, true);
        float[] row = {1, 2};
        for (int p = 0; p < 4; p++) {
            c.write(0, p, row, 0, row, 0);
        }
        assertEquals(4, c.rowEnd(0));
        c.trim(2);
        IntUnaryOperator pos = c.rowPositions(0);
        assertEquals(1, pos.applyAsInt(1));
        assertEquals(-1, pos.applyAsInt(2));
        assertEquals(-1, pos.applyAsInt(3));
        assertEquals(2, c.rowEnd(0));
        assertTrue(c.fits(7, 1));
        assertFalse(c.fits(7, 2));
        assertThrows(IllegalStateException.class, () -> c.write(0, 8, row, 0, row, 0));
    }

    @Test
    public void slidingWindowLayersUseAPaddedRing() {
        assertEquals(256, DflashKvCache.ringRows(1000, 4, 2));
        assertEquals(100, DflashKvCache.ringRows(100, 4, 2));
        DflashKvCache c = new DflashKvCache(1000, 2, 4, new boolean[]{true}, false, true, new int[]{1}, 2,
                KvDType.F32, true);
        assertEquals(256, c.rows(0));
        float[] row = {1, 2};
        c.write(0, 300, row, 0, null, 0);
        assertEquals(300, c.rowPositions(0).applyAsInt(300 % 256));
    }

    /** Full layers hold the capacity and windowed layers their ring: K and V rows plus one int position per row. */
    @Test
    public void cacheBytesCountEveryLayersRowsAndPositions() {
        boolean[] swa = {false, true};
        int[] heads = {2, 1};
        long full = 1000L * (2 * 4 * Float.BYTES) * 2 + 1000L * Integer.BYTES;
        long ring = 256L * (4 * Float.BYTES) * 2 + 256L * Integer.BYTES;
        assertEquals(full + ring, DflashKvCache.bytes(1000, 2, 4, swa, false, heads, 4, KvDType.F32));
        DflashKvCache c = new DflashKvCache(1000, 2, 4, swa, false, false, heads, 4, KvDType.F32, false);
        assertEquals(1000, c.rows(0));
        assertEquals(256, c.rows(1));
        long keyOnly = 256L * (2 * Float.BYTES) + 256L * Integer.BYTES;
        assertEquals(keyOnly, DflashKvCache.bytes(1000, 2, 4, new boolean[]{true}, true, new int[]{1}, 2,
                KvDType.F32));
        assertThrows(IllegalArgumentException.class,
                () -> DflashKvCache.bytes(0, 2, 4, swa, false, heads, 4, KvDType.F32));
    }

    /** llama-kv-cache.cpp:324-360: quantized rows only, K block doubling over the head, V blocks of 64. */
    @Test
    public void quantizedCachesCarryTheHadamardRotation() {
        DflashKvCache f16 = new DflashKvCache(8, 2, 0, new boolean[]{false}, false, false, new int[]{1}, 128,
                KvDType.F16, true);
        assertNull(f16.rotationK());
        assertNull(f16.rotationV());
        DflashKvCache q8 = new DflashKvCache(8, 2, 0, new boolean[]{false}, false, false, new int[]{2}, 128,
                KvDType.Q8_0, true);
        assertEquals(128, q8.rotationK().block());
        assertEquals(64, q8.rotationV().block());
        DflashKvCache keyOnly = new DflashKvCache(8, 2, 4, new boolean[]{true}, false, true, new int[]{1}, 512,
                KvDType.Q8_0, true);
        assertEquals(512, keyOnly.rotationK().block());
        assertNull(keyOnly.rotationV());
        DflashKvCache disabled = new DflashKvCache(8, 2, 0, new boolean[]{false}, false, false, new int[]{1}, 128,
                KvDType.Q8_0, false);
        assertNull(disabled.rotationK());
        assertNull(disabled.rotationV());
    }

    // ------------------------------------------------------------------ metadata and sampling

    @Test
    public void metaFlagsReadBoolsAndStrings() {
        Map<String, Object> m = new HashMap<>();
        m.put(DflashSpec.KEY_SAMPLE_FROM_ANCHOR, "false");
        m.put(DflashSpec.KEY_ATTENTION_CAUSAL, Boolean.TRUE);
        m.put(DflashSpec.KEY_HAS_CONFIDENCE_HEAD, "true");
        m.put("dflash.numeric_flag", 1);
        GgufKv kv = GgufKv.ofMap(m);
        assertFalse(DflashSpec.metaFlag(kv, DflashSpec.KEY_SAMPLE_FROM_ANCHOR, true));
        assertTrue(DflashSpec.metaFlag(kv, DflashSpec.KEY_ATTENTION_CAUSAL, false));
        assertTrue(DflashSpec.metaFlag(kv, DflashSpec.KEY_HAS_CONFIDENCE_HEAD, false));
        assertFalse(DflashSpec.metaFlag(kv, "dflash.numeric_flag", true));
        assertTrue(DflashSpec.metaFlag(GgufKv.ofMap(new HashMap<>()), DflashSpec.KEY_SAMPLE_FROM_ANCHOR, true));
    }

    @Test
    public void topProbabilityIsTheSoftmaxOfTheBestCandidates() {
        // 1 / (1 + 2 e^-2)
        assertEquals(0.786986f, DflashDrafter.topProbability(new float[]{2, 0, 0}, 0, 3), EPS);
        float[] wide = new float[12];
        wide[5] = 1.0f;
        // eleven zeros, only nine of them among the ten best: 1 / (1 + 9 e^-1)
        assertEquals(1.0f / (1.0f + 9.0f * (float) Math.exp(-1.0)), DflashDrafter.topProbability(wide, 0, 12), EPS);
    }
}
