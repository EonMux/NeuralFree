package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.cpu.KvInt8Rows;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link CpuInt8Attention} against the dequantizing {@link CpuAttentionCore} over the same int8 cache: the fused
 * kernel must return the generic outputs within the query quantization error, and exactly when the query values are
 * already exact int8 multiples of their scale. GQA, several rows, position biases (including the Inkling table row of
 * each query), windows, paged slot maps, empty rows and the refused static sink tokens.
 */
public class CpuInt8AttentionTest {

    /**
     * Largest output difference the int8 query quantization may cause on the fixture's random values: the query
     * codes carry a rounding of half a code step ({@code absmax / 254}) per element, and the outputs are convex
     * combinations of value rows of magnitude up to 2. The exactly representable queries below agree to
     * {@code 2e-5}, so this bound is the quantization error alone.
     */
    private static final float PARITY_TOL = 1e-2f;

    /** Rounding difference of the split-K combine against the single pass. */
    private static final float SPLIT_TOL = 1e-5f;

    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int D_K = 32;
    private static final int D_V = 24;
    private static final int ROWS = 12;

    private static float[] random(int n, long seed, float span) {
        Random rnd = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (rnd.nextFloat() * 2f - 1f) * span;
        }
        return x;
    }

    /** An int8 cache of {@link #ROWS} rows whose row {@code r} holds the position {@code r}. */
    private static CpuKvLayer cache(long seed) {
        return cache(KvWindow.FULL_CAUSAL, seed);
    }

    private static CpuKvLayer cache(KvWindow window, long seed) {
        return cache(window, seed, ROWS);
    }

    private static CpuKvLayer cache(KvWindow window, long seed, int rows) {
        KvLayout layout = KvLayout.of(rows, false, N_HEAD_KV * D_K, N_HEAD_KV * D_V, KvDType.INT8_PER_TOKEN_HEAD,
                KvDType.INT8_PER_TOKEN_HEAD, window).withHeads(N_HEAD_KV);
        CpuKvLayer cache = CpuKvLayer.allocate(0, layout, 0, N_HEAD_KV, D_K, D_V);
        for (int r = 0; r < rows; r++) {
            cache.writeK(r, random(N_HEAD_KV * D_K, seed + r, 1.5f), 0);
            cache.writeV(r, random(N_HEAD_KV * D_V, 10_000 + seed + r, 2f), 0);
        }
        return cache;
    }

    private static int[] positions() {
        return positions(ROWS);
    }

    private static int[] positions(int rows) {
        int[] pos = new int[rows];
        for (int r = 0; r < rows; r++) {
            pos[r] = r;
        }
        return pos;
    }

    private static float[] generic(CpuAttentionCore.Params p, int n, float[] q, int[] qPos, CpuKvLayer cache,
                                   int[] pos, int rowStart, int rowEnd) {
        float[] out = new float[n * N_HEAD * D_V];
        CpuAttentionCore.attendRows(p, n, q, 0, N_HEAD * D_K, qPos, 0, cache, r -> pos[r], rowStart, rowEnd, out, 0,
                N_HEAD * D_V);
        return out;
    }

    private static float[] fused(CpuAttentionCore.Params p, int n, float[] q, int[] qPos, CpuKvLayer cache,
                                 int[] pos, int rowStart, int rowEnd) {
        KvInt8Rows rows = cache.int8Rows();
        assertNotNull("an INT8_PER_TOKEN_HEAD cache exposes its raw rows", rows);
        float[] out = new float[n * N_HEAD * D_V];
        CpuInt8Attention.attendRows(p, n, q, 0, N_HEAD * D_K, qPos, 0, rows, r -> pos[r], rowStart, rowEnd, out, 0,
                N_HEAD * D_V);
        return out;
    }

    private static void assertParity(String what, float[] expected, float[] actual, float tol) {
        assertEquals(expected.length, actual.length);
        float max = 0f;
        for (int i = 0; i < expected.length; i++) {
            max = Math.max(max, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(what + ": largest difference " + max, max <= tol);
    }

    @Test
    public void decodeAndPrefillMatchTheDequantizingCore() {
        CpuKvLayer cache = cache(1L);
        int[] pos = positions();
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.35f).build();
        float[] q1 = random(N_HEAD * D_K, 7L, 1f);
        assertParity("one row", generic(p, 1, q1, new int[]{ROWS - 1}, cache, pos, 0, ROWS),
                fused(p, 1, q1, new int[]{ROWS - 1}, cache, pos, 0, ROWS), PARITY_TOL);
        float[] q3 = random(3 * N_HEAD * D_K, 8L, 1f);
        int[] qPos = {5, 8, ROWS - 1};
        assertParity("three rows", generic(p, 3, q3, qPos, cache, pos, 0, ROWS),
                fused(p, 3, q3, qPos, cache, pos, 0, ROWS), PARITY_TOL);
    }

    /**
     * Key splits: a decode row over four key blocks runs one partial online-softmax state per split, whose combine
     * must return the single-pass result whatever the split count.
     */
    @Test
    public void keySplitsMatchTheSinglePass() {
        int rows = 4 * CpuAttentionCore.KEY_BLOCK;
        CpuKvLayer cache = cache(KvWindow.FULL_CAUSAL, 2L, rows);
        int[] pos = positions(rows);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.5f).build();
        float[] q = random(N_HEAD * D_K, 9L, 1f);
        int[] qPos = {rows - 1};
        KvInt8Rows view = cache.int8Rows();
        float[] single = new float[N_HEAD * D_V];
        CpuInt8Attention.attendRowsSplit(p, 1, q, 0, N_HEAD * D_K, qPos, 0, view, r -> pos[r], 0, rows, single, 0,
                N_HEAD * D_V, 1);
        assertParity("one split equals the fused call", fused(p, 1, q, qPos, cache, pos, 0, rows), single, PARITY_TOL);
        for (int splits : new int[]{2, 3, 5, 100}) {
            float[] split = new float[N_HEAD * D_V];
            CpuInt8Attention.attendRowsSplit(p, 1, q, 0, N_HEAD * D_K, qPos, 0, view, r -> pos[r], 0, rows, split, 0,
                    N_HEAD * D_V, splits);
            assertParity("splits " + splits, single, split, SPLIT_TOL);
        }
    }

    /** A query whose values are exact multiples of its scale is quantized without error, so the scores agree. */
    @Test
    public void exactQueriesMatchTheDequantizingCore() {
        CpuKvLayer cache = cache(3L);
        int[] pos = positions();
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.25f).build();
        float[] q = new float[N_HEAD * D_K];
        Random rnd = new Random(11L);
        for (int h = 0; h < N_HEAD; h++) {
            int base = h * D_K;
            q[base] = 127 * 0.25f;
            for (int d = 1; d < D_K; d++) {
                q[base + d] = (rnd.nextInt(255) - 127) * 0.25f;
            }
        }
        assertParity("exact query codes", generic(p, 1, q, new int[]{ROWS - 1}, cache, pos, 0, ROWS),
                fused(p, 1, q, new int[]{ROWS - 1}, cache, pos, 0, ROWS), 2e-5f);
    }

    @Test
    public void windowsMaskTheSameKeys() {
        KvWindow window = KvWindow.of(SwaType.STANDARD, 4, true);
        CpuKvLayer cache = cache(window, 4L);
        int[] pos = positions();
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.4f).window(window).build();
        float[] q = random(N_HEAD * D_K, 12L, 1f);
        assertParity("sliding window", generic(p, 1, q, new int[]{ROWS - 1}, cache, pos, 0, ROWS),
                fused(p, 1, q, new int[]{ROWS - 1}, cache, pos, 0, ROWS), PARITY_TOL);
    }

    @Test
    public void positionBiasAndSinkLogitsMatch() {
        CpuKvLayer cache = cache(5L);
        int[] pos = positions();
        float[] sinks = {0.5f, -1.0f, 2.0f, 0.1f};
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.3f).alibi(8.0f)
                .sinks(sinks).build();
        float[] q = random(N_HEAD * D_K, 13L, 1f);
        assertParity("ALiBi and sinks", generic(p, 1, q, new int[]{ROWS - 1}, cache, pos, 0, ROWS),
                fused(p, 1, q, new int[]{ROWS - 1}, cache, pos, 0, ROWS), PARITY_TOL);
    }

    /**
     * The Inkling relative table holds one row per query of the step, so both paths must look it up by the query's
     * row in that table rather than by the head alone.
     */
    @Test
    public void inklingRelativeBiasUsesTheQueryRow() {
        CpuKvLayer cache = cache(16L);
        int[] pos = positions();
        int relExtent = 16;
        int rows = 3;
        float[] table = random(rows * N_HEAD * relExtent, 17L, 4f);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.35f).inkling(relExtent, table)
                .build();
        float[] q = random(rows * N_HEAD * D_K, 18L, 1f);
        int[] qPos = {5, 8, ROWS - 1};
        assertParity("Inkling relative bias", generic(p, rows, q, qPos, cache, pos, 0, ROWS),
                fused(p, rows, q, qPos, cache, pos, 0, ROWS), PARITY_TOL);
    }

    @Test
    public void emptyRowsAreNeverVisible() {
        CpuKvLayer cache = cache(6L);
        int[] pos = positions();
        Arrays.fill(pos, 8, ROWS, -1);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.4f).build();
        float[] q = random(N_HEAD * D_K, 14L, 1f);
        assertParity("empty tail rows", generic(p, 1, q, new int[]{7}, cache, pos, 0, ROWS),
                fused(p, 1, q, new int[]{7}, cache, pos, 0, ROWS), PARITY_TOL);
    }

    /** A paged slot map reaches the same physical rows as the contiguous span the slots name. */
    @Test
    public void aPagedRowMapReadsTheRowsTheSlotsName() {
        CpuKvLayer cache = cache(7L);
        int[] pos = positions();
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.45f).build();
        float[] q = random(N_HEAD * D_K, 15L, 1f);
        float[] expected = generic(p, 1, q, new int[]{ROWS - 1}, cache, pos, 0, ROWS);
        float[] actual = new float[N_HEAD * D_V];
        KvInt8Rows rows = cache.int8Rows();
        CpuInt8Attention.attendRows(p, 1, q, 0, N_HEAD * D_K, new int[]{ROWS - 1}, 0, rows, r -> pos[r],
                slot -> slot, 0, ROWS, actual, 0, N_HEAD * D_V);
        assertParity("identity slot map", expected, actual, PARITY_TOL);

        // one visible key: every slot maps to the row holding position 5, the span the generic call names
        Arrays.fill(pos, -1);
        pos[9] = 5;
        float[] one = generic(p, 1, q, new int[]{5}, cache, pos, 9, 10);
        float[] mapped = new float[N_HEAD * D_V];
        CpuInt8Attention.attendRows(p, 1, q, 0, N_HEAD * D_K, new int[]{5}, 0, rows, r -> pos[r],
                slot -> 9, 0, 3, mapped, 0, N_HEAD * D_V);
        assertParity("three slots on one row", one, mapped, PARITY_TOL);
    }

    @Test
    public void learnedStaticSinkTokensAreRefusedByName() {
        CpuKvLayer cache = cache(8L);
        KvInt8Rows rows = cache.int8Rows();
        float[] sinkK = new float[N_HEAD_KV * D_K];
        float[] sinkV = new float[N_HEAD_KV * D_V];
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(1f)
                .sinkKv(sinkK, sinkV, 1).build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> CpuInt8Attention.attendRows(p, 1, new float[N_HEAD * D_K], 0, N_HEAD * D_K, new int[]{0}, 0,
                        rows, r -> 0, 0, ROWS, new float[N_HEAD * D_V], 0, N_HEAD * D_V));
        assertTrue(e.getMessage(), e.getMessage().contains("sink tokens"));
    }

    @Test
    public void mismatchedHeadCountsAreRejected() {
        CpuKvLayer cache = cache(9L);
        KvInt8Rows rows = cache.int8Rows();
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD + 1).scale(1f).build();
        assertThrows(IllegalArgumentException.class,
                () -> CpuInt8Attention.attendRows(p, 1, new float[(N_HEAD + 1) * D_K], 0, (N_HEAD + 1) * D_K,
                        new int[]{0}, 0, rows, r -> 0, 0, ROWS, new float[(N_HEAD + 1) * D_V], 0,
                        (N_HEAD + 1) * D_V));
    }

    /** The raw view returns the row's stored encoding, not a value decoded or re-quantized on the way out. */
    @Test
    public void theRawViewReturnsTheStoredCodesAndRowScales() {
        CpuKvLayer cache = cache(KvWindow.FULL_CAUSAL, 10L);
        float[] k = random(N_HEAD_KV * D_K, 13L, 1.5f);
        float[] v = random(N_HEAD_KV * D_V, 10_013L, 2f);
        KvInt8Rows view = cache.int8Rows();
        byte[] codes = new byte[Math.max(D_K, D_V)];
        for (int h = 0; h < N_HEAD_KV; h++) {
            float ks = KvBlockCodec.perTokenHeadScale(KvDType.INT8_PER_TOKEN_HEAD, k, h * D_K, D_K);
            assertEquals("K scale of head " + h, ks, view.kScale(3, h), 0f);
            view.readKCodes(3, h, codes, 0);
            for (int d = 0; d < D_K; d++) {
                assertEquals("K code " + d, KvBlockCodec.perTokenHeadCode(KvDType.INT8_PER_TOKEN_HEAD,
                        k[h * D_K + d], ks), codes[d] & 0xFF);
            }
            float vs = KvBlockCodec.perTokenHeadScale(KvDType.INT8_PER_TOKEN_HEAD, v, h * D_V, D_V);
            assertEquals("V scale of head " + h, vs, view.vScale(3, h), 0f);
            view.readVCodes(3, h, codes, 0);
            for (int d = 0; d < D_V; d++) {
                assertEquals("V code " + d, KvBlockCodec.perTokenHeadCode(KvDType.INT8_PER_TOKEN_HEAD,
                        v[h * D_V + d], vs), codes[d] & 0xFF);
            }
        }
    }

    @Test
    public void aCacheInAnotherDtypeHasNoRawView() {
        KvLayout layout = KvLayout.of(ROWS, false, N_HEAD_KV * D_K, N_HEAD_KV * D_V, KvDType.F16, KvDType.F16,
                KvWindow.FULL_CAUSAL).withHeads(N_HEAD_KV);
        assertNull(CpuKvLayer.allocate(0, layout, 0, N_HEAD_KV, D_K, D_V).int8Rows());
        KvLayout mixed = KvLayout.of(ROWS, false, N_HEAD_KV * D_K, N_HEAD_KV * D_V,
                KvDType.INT8_PER_TOKEN_HEAD, KvDType.F16, KvWindow.FULL_CAUSAL).withHeads(N_HEAD_KV);
        assertNull("an int8 K with an F16 V has no int8 value rows",
                CpuKvLayer.allocate(0, mixed, 0, N_HEAD_KV, D_K, D_V).int8Rows());
    }
}
