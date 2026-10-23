package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for {@link CpuRope}, {@link RopeFreqTable} and {@link RopePositions}: rotation layouts, full IMROPE sector
 * selection for text tokens, KV stores (F32, F16, BF16, the block types with C {@code roundf}), and context shifts that
 * compose with the original rotation, including Hadamard-rotated block-type caches.
 */
public class CpuRopeTest {

    private static float[] random(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2f - 1f;
        }
        return v;
    }

    private static RopeSpec spec(RopeType type, int nDims) {
        return RopeSpec.builder(type).nDims(nDims).freqBase(10000f).build();
    }

    @Test
    public void positionZeroIsIdentity() {
        RopeFreqTable table = RopeFreqTable.of(spec(RopeType.NEOX, 64), null);
        float[] x = random(2 * 64, 1);
        float[] y = x.clone();
        CpuRope.apply(table, y, 0, 1, 2, 64, 128, 0, new int[]{0}, 0, 1);
        assertArrayEquals(x, y, 0f);
    }

    @Test
    public void normRotatesAdjacentPairs() {
        int nDims = 8;
        RopeSpec spec = spec(RopeType.NORM, nDims);
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        float[] x = random(nDims, 2);
        float[] y = x.clone();
        int p = 7;
        CpuRope.apply(table, y, 0, 1, 1, nDims, nDims, 0, new int[]{p}, 0, 1);
        for (int i = 0; i < nDims / 2; i++) {
            float theta = p * (float) Math.pow(spec.thetaScale(), i);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            assertEquals(x[2 * i] * c - x[2 * i + 1] * s, y[2 * i], 1e-6f);
            assertEquals(x[2 * i] * s + x[2 * i + 1] * c, y[2 * i + 1], 1e-6f);
        }
    }

    @Test
    public void partialRotationLeavesOtherDimsUntouched() {
        RopeFreqTable table = RopeFreqTable.of(spec(RopeType.NEOX, 16), null);
        float[] x = random(64, 3);
        float[] y = x.clone();
        CpuRope.apply(table, y, 0, 1, 2, 32, 64, 8, new int[]{5}, 0, 1);
        for (int h = 0; h < 2; h++) {
            for (int d = 0; d < 32; d++) {
                if (d < 8 || d >= 24) {
                    assertEquals(x[h * 32 + d], y[h * 32 + d], 0f);
                }
            }
        }
    }

    /** IMROPE text tokens keep the e section at position 0: pairs 61-62 of qwen3vl stay unrotated (no NEOX shortcut). */
    @Test
    public void imropeTextKeepsESectionAtZero() {
        int[] sections = {24, 20, 20, 0};
        RopeSpec spec = RopeSpec.builder(RopeType.IMROPE).nDims(128).freqBase(5000000f).sections(sections).build();
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        int p = 100000;
        int[] pos = RopePositions.text(spec, p, 1);
        assertArrayEquals(new int[]{p, p, p, 0}, pos);

        float[] x = random(128, 4);
        x[61] = 0.5f;
        x[61 + 64] = 0.75f;
        float[] y = x.clone();
        CpuRope.apply(table, y, 0, 1, 1, 128, 128, 0, pos, 0, 1);
        for (int i : new int[]{61, 62}) {
            assertEquals(x[i], y[i], 0f);
            assertEquals(x[i + 64], y[i + 64], 0f);
        }
        float[] neox = x.clone();
        CpuRope.apply(RopeFreqTable.of(spec(RopeType.NEOX, 128), null), neox, 0, 1, 1, 128, 128, 0, new int[]{p}, 0, 1);
        assertTrue("IMROPE text must differ from a NEOX rotation", Math.abs(neox[61] - y[61]) > 1e-4f);
    }

    @Test
    public void mropeSectionsReadTheirOwnPositions() {
        int[] sections = {16, 24, 24, 0};
        RopeSpec spec = RopeSpec.builder(RopeType.MROPE).nDims(128).freqBase(1000000f).sections(sections).build();
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        int[] pos = {3, 5, 7, 0};
        float[] cs = new float[128];
        table.cosSin(pos, 0, 0, 1, cs);
        float[] expect = new float[2];
        RopeMath.cosSin(7 * (float) Math.pow(spec.thetaScale(), 40), 1f, 0f, 1f, 0f, 0f, 40, expect, 0);
        assertEquals(expect[0], cs[80], 0f);
        assertEquals(expect[1], cs[81], 0f);
    }

    @Test
    public void storeF32F16() {
        float[] src = random(2 * 8, 5);
        float[] dst = new float[4 * 8];
        CpuRope.storeF32(src, 0, 8, 2, 8, new int[]{3, 1}, dst, 8);
        for (int e = 0; e < 8; e++) {
            assertEquals(src[e], dst[3 * 8 + e], 0f);
            assertEquals(src[8 + e], dst[8 + e], 0f);
        }
        short[] half = new short[4 * 8];
        CpuRope.storeF16(src, 0, 8, 2, 8, new int[]{-1, 2}, half, 8);
        for (int e = 0; e < 8; e++) {
            assertEquals(0, half[e]);
            assertEquals(FpBlockDecoders.roundToFp16(src[8 + e]), FpBlockDecoders.fp16ToFp32(half[2 * 8 + e]), 0f);
        }
    }

    /** d = 127 / 127 = 1, so q = roundf(x): halfway cases round away from zero. */
    @Test
    public void storeQ8_0RoundsHalfAwayFromZero() {
        float[] src = new float[32];
        src[0] = 127f;
        src[1] = 2.5f;
        src[2] = -2.5f;
        src[3] = 0.5f;
        src[4] = -0.49999997f;
        ByteBuffer dst = ByteBuffer.allocate(2 * 34).order(ByteOrder.LITTLE_ENDIAN);
        CpuRope.storeBlocks(KvDType.Q8_0, src, 0, 32, 1, 32, new int[]{1}, dst, 32);
        int base = 34;
        assertEquals(1f, FpBlockDecoders.fp16ToFp32(dst.getShort(base) & 0xFFFF), 0f);
        assertEquals(127, dst.get(base + 2));
        assertEquals(3, dst.get(base + 3));
        assertEquals(-3, dst.get(base + 4));
        assertEquals(1, dst.get(base + 5));
        assertEquals(0, dst.get(base + 6));
        float[] back = new float[32];
        KvBlockCodec.dequantize(KvDType.Q8_0, dst, base, back, 0);
        assertEquals(3f, back[1], 0f);
    }

    /** rope(x, a) shifted by d equals rope(x, a + d) (rotations compose). */
    @Test
    public void shiftComposesWithRotationF32() {
        RopeSpec spec = spec(RopeType.NEOX, 32);
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        RopeFreqTable shift = RopeFreqTable.forShift(spec, null);
        int nHeads = 2;
        int width = nHeads * 32;
        float[] x = random(2 * width, 6);
        float[] cache = x.clone();
        CpuRope.apply(table, cache, 0, 2, nHeads, 32, width, 0, new int[]{10, 11}, 0, 2);
        CpuRope.shiftF32(shift, cache, 2, nHeads, 32, width, 0, new int[]{-4, 0});

        float[] expect = x.clone();
        CpuRope.apply(table, expect, 0, 2, nHeads, 32, width, 0, new int[]{6, 11}, 0, 2);
        for (int i = 0; i < expect.length; i++) {
            assertEquals(expect[i], cache[i], 1e-5f);
        }

        short[] half = new short[cache.length];
        float[] rotated = x.clone();
        CpuRope.apply(table, rotated, 0, 2, nHeads, 32, width, 0, new int[]{10, 11}, 0, 2);
        CpuRope.storeF16(rotated, 0, width, 2, width, new int[]{0, 1}, half, width);
        CpuRope.shiftF16(shift, half, 2, nHeads, 32, width, 0, new int[]{-4, 0});
        for (int i = 0; i < expect.length; i++) {
            assertEquals(expect[i], FpBlockDecoders.fp16ToFp32(half[i]), 2e-3f);
        }
    }

    /** A Hadamard-rotated Q8_0 cache shifts to the rotation of rope(x, a + d), within quantization error. */
    @Test
    public void shiftQ8_0WithHadamardRotation() {
        RopeSpec spec = spec(RopeType.NEOX, 128);
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        RopeFreqTable shift = RopeFreqTable.forShift(spec, null);
        int nHeads = 2;
        int width = nHeads * 128;
        float[] x = random(width, 7);

        float[] k = x.clone();
        CpuRope.apply(table, k, 0, 1, nHeads, 128, width, 0, new int[]{20}, 0, 1);
        for (int c = 0; c < width; c += 128) {
            CpuHadamard.fwht(k, c, 128);
        }
        ByteBuffer cache = ByteBuffer.allocate(width / 32 * 34).order(ByteOrder.LITTLE_ENDIAN);
        CpuRope.storeBlocks(KvDType.Q8_0, k, 0, width, 1, width, new int[]{0}, cache, width);
        CpuRope.shiftBlocks(KvDType.Q8_0, shift, cache, 1, nHeads, 128, width, 0, 128, new int[]{-7});

        float[] got = new float[width];
        for (int b = 0; b < width / 32; b++) {
            KvBlockCodec.dequantize(KvDType.Q8_0, cache, b * 34, got, b * 32);
        }
        for (int c = 0; c < width; c += 128) {
            CpuHadamard.fwht(got, c, 128);
        }
        float[] expect = x.clone();
        CpuRope.apply(table, expect, 0, 1, nHeads, 128, width, 0, new int[]{13}, 0, 1);
        for (int i = 0; i < width; i++) {
            assertEquals(expect[i], got[i], 0.05f);
        }
    }

    /**
     * Every block type: a Hadamard-rotated cache shifts to the rotation of rope(x, a + d) within the RMS error of two
     * quantizations (store and requantize), about {@code step / sqrt(6)} for the type's step.
     */
    @Test
    public void shiftBlocksWithHadamardRotationEveryType() {
        RopeSpec spec = spec(RopeType.NEOX, 128);
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        RopeFreqTable shift = RopeFreqTable.forShift(spec, null);
        int nHeads = 2;
        int width = nHeads * 128;
        float[] x = random(width, 17);
        float[] expect = x.clone();
        CpuRope.apply(table, expect, 0, 1, nHeads, 128, width, 0, new int[]{13}, 0, 1);
        for (KvDType type : KvDType.all()) {
            if (!type.quantized()) {
                continue;
            }
            float[] k = x.clone();
            CpuRope.apply(table, k, 0, 1, nHeads, 128, width, 0, new int[]{20}, 0, 1);
            for (int c = 0; c < width; c += 128) {
                CpuHadamard.fwht(k, c, 128);
            }
            int bb = type.blockBytes();
            ByteBuffer cache = ByteBuffer.allocate(width / 32 * bb).order(ByteOrder.LITTLE_ENDIAN);
            CpuRope.storeBlocks(type, k, 0, width, 1, width, new int[]{0}, cache, width);
            CpuRope.shiftBlocks(type, shift, cache, 1, nHeads, 128, width, 0, 128, new int[]{-7});

            float[] got = new float[width];
            for (int b = 0; b < width / 32; b++) {
                KvBlockCodec.dequantize(type, cache, b * bb, got, b * 32);
            }
            for (int c = 0; c < width; c += 128) {
                CpuHadamard.fwht(got, c, 128);
            }
            double se = 0;
            for (int i = 0; i < width; i++) {
                double e = expect[i] - got[i];
                se += e * e;
            }
            double rms = Math.sqrt(se / width);
            double bound = type == KvDType.Q8_0 ? 0.01 : (type == KvDType.Q5_0 || type == KvDType.Q5_1 ? 0.05 : 0.1);
            assertTrue(type + " shift RMS error " + rms + " exceeds " + bound, rms < bound);
        }
    }

    /** A BF16 cache shifts like F32 within two bfloat16 roundings. */
    @Test
    public void shiftComposesWithRotationBf16() {
        RopeSpec spec = spec(RopeType.NEOX, 32);
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        RopeFreqTable shift = RopeFreqTable.forShift(spec, null);
        int nHeads = 2;
        int width = nHeads * 32;
        float[] x = random(2 * width, 6);
        float[] rotated = x.clone();
        CpuRope.apply(table, rotated, 0, 2, nHeads, 32, width, 0, new int[]{10, 11}, 0, 2);
        short[] bf = new short[rotated.length];
        CpuRope.storeBf16(rotated, 0, width, 2, width, new int[]{0, 1}, bf, width);
        for (int i = 0; i < rotated.length; i++) {
            assertEquals(KvBlockCodec.fp32ToBf16(rotated[i]), bf[i] & 0xFFFF);
        }
        CpuRope.shiftBf16(shift, bf, 2, nHeads, 32, width, 0, new int[]{-4, 0});

        float[] expect = x.clone();
        CpuRope.apply(table, expect, 0, 2, nHeads, 32, width, 0, new int[]{6, 11}, 0, 2);
        for (int i = 0; i < expect.length; i++) {
            assertEquals(expect[i], FpBlockDecoders.bf16ToFp32(bf[i] & 0xFFFF), 2e-2f);
        }
    }

    @Test
    public void storeBlocksRejectsNonBlockTypes() {
        ByteBuffer dst = ByteBuffer.allocate(4 * 32);
        for (KvDType type : new KvDType[]{KvDType.F32, KvDType.F16, KvDType.BF16}) {
            try {
                CpuRope.storeBlocks(type, new float[32], 0, 32, 1, 32, new int[]{0}, dst, 32);
                fail(type + " is not a block type");
            } catch (IllegalArgumentException expected) {
                // expected
            }
        }
    }

    /** Partial rotary dims inside one Hadamard chunk: the whole chunk is un-rotated, shifted and re-rotated. */
    @Test
    public void partialShiftQ8_0WithHadamardRotation() {
        int head = 128;
        RopeSpec spec = spec(RopeType.NEOX, 32);
        RopeFreqTable table = RopeFreqTable.of(spec, null);
        RopeFreqTable shift = RopeFreqTable.forShift(spec, null);
        float[] x = random(head, 11);

        float[] k = x.clone();
        CpuRope.apply(table, k, 0, 1, 1, head, head, 0, new int[]{30}, 0, 1);
        CpuHadamard.fwht(k, 0, head);
        ByteBuffer cache = ByteBuffer.allocate(head / 32 * 34).order(ByteOrder.LITTLE_ENDIAN);
        CpuRope.storeBlocks(KvDType.Q8_0, k, 0, head, 1, head, new int[]{0}, cache, head);
        CpuRope.shiftBlocks(KvDType.Q8_0, shift, cache, 1, 1, head, head, 0, head, new int[]{-12});

        float[] got = new float[head];
        for (int b = 0; b < head / 32; b++) {
            KvBlockCodec.dequantize(KvDType.Q8_0, cache, b * 34, got, b * 32);
        }
        CpuHadamard.fwht(got, 0, head);
        float[] expect = x.clone();
        CpuRope.apply(table, expect, 0, 1, 1, head, head, 0, new int[]{18}, 0, 1);
        for (int i = 0; i < head; i++) {
            assertEquals(expect[i], got[i], 0.05f);
        }
    }

    /** Without rotation the window is the rotated dims widened to whole blocks; the rest of the block round-trips. */
    @Test
    public void shiftQ8_0WidensToWholeBlocks() {
        int head = 64;
        RopeFreqTable shift = RopeFreqTable.forShift(spec(RopeType.NEOX, 16), null);
        ByteBuffer cache = ByteBuffer.allocate(head / 32 * 34).order(ByteOrder.LITTLE_ENDIAN);
        CpuRope.storeBlocks(KvDType.Q8_0, random(head, 12), 0, head, 1, head, new int[]{0}, cache, head);
        float[] before = new float[head];
        for (int b = 0; b < head / 32; b++) {
            KvBlockCodec.dequantize(KvDType.Q8_0, cache, b * 34, before, b * 32);
        }
        assertEquals(32, RopeKernels.shiftWindowStart(40, 0));
        assertEquals(64, RopeKernels.shiftWindowEnd(16, 40, 0));
        CpuRope.shiftBlocks(KvDType.Q8_0, shift, cache, 1, 1, head, head, 40, 0, new int[]{5});

        float[] after = new float[head];
        for (int b = 0; b < head / 32; b++) {
            KvBlockCodec.dequantize(KvDType.Q8_0, cache, b * 34, after, b * 32);
        }
        for (int i = 0; i < head; i++) {
            if (i < 40 || i >= 56) {
                assertEquals("dim " + i + " is outside the rotated dims", before[i], after[i], 0f);
            }
        }
    }

    @Test
    public void shiftRejectsInvalidQ8_0Geometry() {
        RopeFreqTable shift = RopeFreqTable.forShift(spec(RopeType.NEOX, 16), null);
        ByteBuffer cache = ByteBuffer.allocate(8 * 34);
        try {
            CpuRope.shiftBlocks(KvDType.Q8_0, shift, cache, 1, 2, 48, 128, 0, 0, new int[]{1});
            fail("a head stride of 48 does not start every head on a Q8_0 block");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        try {
            CpuRope.shiftBlocks(KvDType.Q8_0, shift, cache, 1, 2, 64, 128, 0, 16, new int[]{1});
            fail("a 16-value Hadamard chunk is smaller than a Q8_0 block");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        try {
            CpuRope.shiftBlocks(KvDType.Q8_0, shift, cache, 1, 1, 96, 128, 64, 128, new int[]{1});
            fail("the 128-value Hadamard chunk around dims [64, 80) exceeds the head stride 96");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void contextShiftDeltas() {
        int[] rowPos = {0, 1, 2, 3, 4, 5, -1, 7};
        int[] newPos = new int[rowPos.length];
        int[] delta = new int[rowPos.length];
        RopePositions.contextShift(rowPos, 2, 3, newPos, delta);
        assertArrayEquals(new int[]{0, 1, -1, -1, -1, 2, -1, 4}, newPos);
        assertArrayEquals(new int[]{0, 0, 0, 0, 0, -3, 0, -3}, delta);
    }

    @Test
    public void stepPositionsUseTheArchPattern() {
        int[] step = new int[8];
        RopePositions.fillStep(MropeTextPositions.P000, 4, 2, step);
        assertArrayEquals(new int[]{4, 5, 0, 0, 0, 0, 0, 0}, step);
        int[] single = new int[3];
        RopePositions.fill(RopeType.NEOX, MropeTextPositions.PPP0, new int[]{9, 3, 5}, 0, 3, single);
        assertArrayEquals(new int[]{9, 3, 5}, single);
    }
}
