package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import org.junit.Test;

import java.math.BigInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The DeepSeek-V4.1 Engram hash ({@link CpuEngramExtra#hashRows}) and gate against a direct evaluation of
 * engram.py {@code NgramHashState.forward} and models/deepseek41.cpp:456-500.
 */
public class CpuEngramExtraTest {

    private static final long[] MULT = {7L, 11L, 13L};
    private static final long[] OFFSETS = {0L, 100L, 200L, 300L};
    private static final long[] VOCAB = {5L, 7L, 11L, 13L};
    private static final int PAD = 3;
    private static final int NGRAM = 3;
    private static final int PER_GRAM = 2;
    private static final int COLS = 4;

    /** Unsigned {@code rolling mod prime + offset}, independently of the production long arithmetic. */
    private static int row(long rolling, long prime, long offset) {
        BigInteger r = new BigInteger(Long.toUnsignedString(rolling));
        return r.mod(BigInteger.valueOf(prime)).intValue() + Math.toIntExact(offset);
    }

    /** engram.py: {@code toks[0]} is the token, {@code toks[s]} the token {@code s} positions back. */
    private static void expected(int[] toks, int[] out, int outOff) {
        long rolling = toks[0] * MULT[0];
        for (int o = 1; o < NGRAM; o++) {
            rolling ^= toks[o] * MULT[o];
            for (int h = 0; h < PER_GRAM; h++) {
                int col = (o - 1) * PER_GRAM + h;
                out[outOff + col] = row(rolling, VOCAB[col], OFFSETS[col]);
            }
        }
    }

    private static int[] identityMap(int n) {
        int[] map = new int[n];
        for (int i = 0; i < n; i++) {
            map[i] = i;
        }
        return map;
    }

    @Test
    public void hashMatchesTheHandComputedNgrams() {
        int[] window = {5, 9, 2, 7};
        int[] ids = new int[2 * COLS];
        CpuEngramExtra.hashRows(window, NGRAM - 1, 2, NGRAM, PER_GRAM, COLS, MULT, OFFSETS, VOCAB, PAD,
                identityMap(16), new long[NGRAM], ids);
        int[] want = new int[2 * COLS];
        // token 2 with predecessors 9 (one back) and 5 (two back); token 7 with 2 and 9
        expected(new int[]{2, 9, 5}, want, 0);
        expected(new int[]{7, 2, 9}, want, COLS);
        for (int i = 0; i < want.length; i++) {
            assertEquals("id " + i, want[i], ids[i]);
        }
        // the concrete values of the first token's n-gram-1 columns
        assertEquals(109 % 5, ids[0]);
        assertEquals(109 % 7 + 100, ids[1]);
    }

    @Test
    public void aMissingPredecessorBlocksThatShiftAndEveryLongerOne() {
        int[] window = {5, -1, 2};   // back = 2: valid two back, missing one back (an embedding row)
        int[] ids = new int[COLS];
        CpuEngramExtra.hashRows(window, NGRAM - 1, 1, NGRAM, PER_GRAM, COLS, MULT, OFFSETS, VOCAB, PAD,
                identityMap(16), new long[NGRAM], ids);
        // once the one-back shift is dead, the valid two-back token is not read: both predecessors hash as the pad
        int[] want = new int[COLS];
        expected(new int[]{2, PAD, PAD}, want, 0);
        for (int i = 0; i < COLS; i++) {
            assertEquals("id " + i, want[i], ids[i]);
        }
    }

    @Test
    public void unmappedAndEmbeddingTokensAreDead() {
        int[] map = identityMap(4);
        map[2] = -1;
        int[] ids = new int[2 * COLS];
        // token 1 with predecessors 2 (mapped to -1) and 9 (outside the map); then an embedding row (-1)
        int[] window = {9, 2, 1, -1};
        CpuEngramExtra.hashRows(window, NGRAM - 1, 2, NGRAM, PER_GRAM, COLS, MULT, OFFSETS, VOCAB, PAD, map,
                new long[NGRAM], ids);
        int[] want = new int[2 * COLS];
        expected(new int[]{1, PAD, PAD}, want, 0);
        expected(new int[]{PAD, PAD, PAD}, want, COLS);
        for (int i = 0; i < want.length; i++) {
            assertEquals("id " + i, want[i], ids[i]);
        }
    }

    @Test
    public void theRollingHashIsUnsigned() {
        long[] mult = {Long.MIN_VALUE, 3L, 5L};
        long[] vocab = {97L, 89L, 83L, 79L};
        int[] ids = new int[COLS];
        CpuEngramExtra.hashRows(new int[]{4, 6, 9}, 2, 1, NGRAM, PER_GRAM, COLS, mult, OFFSETS, vocab, 0,
                identityMap(16), new long[NGRAM], ids);
        int[] want = new int[COLS];
        expectedUnsigned(new int[]{9, 6, 4}, mult, vocab, want);
        for (int i = 0; i < COLS; i++) {
            assertEquals("id " + i, want[i], ids[i]);
        }
    }

    private static void expectedUnsigned(int[] toks, long[] mult, long[] vocab, int[] out) {
        long rolling = toks[0] * mult[0];
        for (int o = 1; o < NGRAM; o++) {
            rolling ^= toks[o] * mult[o];
            for (int h = 0; h < PER_GRAM; h++) {
                int col = (o - 1) * PER_GRAM + h;
                out[col] = row(rolling, vocab[col], OFFSETS[col]);
            }
        }
    }

    @Test
    public void gateIsSigmoidOfCopysignSqrtMagnitude() {
        assertEquals(ActMath.sigmoid((float) Math.sqrt(1e-6)), CpuEngramExtra.gate(0f), 1e-7f);
        assertEquals(ActMath.sigmoid(-(float) Math.sqrt(1e-6)), CpuEngramExtra.gate(-1e-12f), 1e-7f);
        assertEquals(ActMath.sigmoid(2f), CpuEngramExtra.gate(4f), 1e-6f);
        assertEquals(ActMath.sigmoid(-2f), CpuEngramExtra.gate(-4f), 1e-6f);
        // |s| below the clamp: sqrt(1e-6)
        assertEquals(CpuEngramExtra.gate(0f), CpuEngramExtra.gate(1e-9f), 0f);
    }

    @Test
    public void inconsistentInputsAreRefused() {
        int[] ids = new int[COLS];
        long[] toks = new long[NGRAM];
        int[] map = identityMap(16);
        assertThrows(IllegalArgumentException.class, () -> CpuEngramExtra.hashRows(new int[]{0, 1}, 2, 1,
                NGRAM, PER_GRAM, COLS, MULT, OFFSETS, VOCAB, PAD, map, toks, ids));
        assertThrows(IllegalArgumentException.class, () -> CpuEngramExtra.hashRows(new int[]{0, 1, 2}, 2, 1,
                NGRAM, 3, COLS, MULT, OFFSETS, VOCAB, PAD, map, toks, ids));
        assertThrows(IllegalArgumentException.class, () -> CpuEngramExtra.hashRows(new int[]{0, 1, 2}, 2, 1,
                NGRAM, PER_GRAM, COLS, MULT, OFFSETS, VOCAB, PAD, map, new long[2], ids));
        assertThrows(IllegalArgumentException.class, () -> CpuEngramExtra.hashRows(new int[]{0, 1, 2}, 2, 1,
                NGRAM, PER_GRAM, COLS, MULT, OFFSETS, VOCAB, PAD, map, toks, new int[3]));
    }
}
