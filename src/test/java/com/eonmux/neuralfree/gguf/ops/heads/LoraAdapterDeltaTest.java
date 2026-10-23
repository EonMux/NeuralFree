package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Host LoRA math of {@link LoraAdapter.Delta} and {@link CpuLoraDeltas}: the effective scale (llama-adapter.h:53-57),
 * the projection delta of {@code build_lora_mm} (llama-graph.cpp:1648-1664), the per-expert delta of
 * {@code build_lora_mm_id} (:1734-1746) and the embedding-row delta (:2694-2709), each against a scalar reference.
 */
public class LoraAdapterDeltaTest {

    private static final float EPS = 1e-5f;
    private static final String Q = "blk.0.attn_q.weight";
    private static final String UP_EXPS = "blk.0.ffn_up_exps.weight";
    private static final String EMBD = "token_embd.weight";

    @Test
    public void scaleIsUserTimesAlphaOverRankOrTheUserScaleWithoutAlpha() {
        LoraAdapter.Pair p = LoraFixtures.projection(Q, 5, 3, 4, 1);
        assertEquals(1.5f, LoraFixtures.adapter("a.gguf", 8.0f, p).scale(Q, 0.75f), 0f);
        assertEquals(0.75f, LoraFixtures.adapter("a.gguf", 0.0f, p).scale(Q, 0.75f), 0f);
    }

    @Test
    public void projectionDeltaAddsScaledBTimesAxOnStridedRows() {
        int in = 5;
        int out = 3;
        LoraAdapter.Pair p = LoraFixtures.projection(Q, in, out, 2, 11);
        LoraAdapter a = LoraFixtures.adapter("a.gguf", 4.0f, p);
        float s = a.scale(Q, 0.5f);
        int n = 3;
        int xOff = 2;
        int xStride = in + 1;
        int yOff = 1;
        int yStride = out + 2;
        float[] x = LoraFixtures.random(21, xOff + n * xStride);
        float[] y0 = LoraFixtures.random(22, yOff + n * yStride);
        float[] expected = y0.clone();
        for (int t = 0; t < n; t++) {
            LoraFixtures.project(p, s, 0, x, xOff + t * xStride, expected, yOff + t * yStride);
        }

        float[] direct = y0.clone();
        a.delta(Q, 0.5f).add(n, x, xOff, xStride, new float[n * p.rank()], 0, direct, yOff, yStride);
        assertArrayEquals(expected, direct, EPS);

        CpuProjectionDeltas.Delta host = (CpuProjectionDeltas.Delta) CpuLoraDeltas.of(a.delta(Q, 0.5f));
        float[] viaHost = y0.clone();
        host.add(n, x, xOff, xStride, viaHost, yOff, yStride);
        assertArrayEquals(expected, viaHost, EPS);
    }

    @Test
    public void expertDeltaUsesTheSelectedExpertOfEveryPair() {
        int in = 4;
        int out = 3;
        int k = 2;
        int nTokens = 2;
        LoraAdapter.Pair p = LoraFixtures.experts(UP_EXPS, in, out, 2, 3, 31);
        LoraAdapter a = LoraFixtures.adapter("a.gguf", 2.0f, p);
        float s = a.scale(UP_EXPS, 1.0f);
        int[] ids = {2, 0, 1, 2};
        int pairs = nTokens * k;
        CpuProjectionDeltas.ExpertDelta host = (CpuProjectionDeltas.ExpertDelta) CpuLoraDeltas.of(
                a.delta(UP_EXPS, 1.0f));

        float[] tokenRows = LoraFixtures.random(41, 1 + nTokens * in);
        float[] y0 = LoraFixtures.random(42, pairs * out);
        float[] expected = y0.clone();
        for (int q = 0; q < pairs; q++) {
            LoraFixtures.project(p, s, ids[q], tokenRows, 1 + (q / k) * in, expected, q * out);
        }
        float[] y = y0.clone();
        host.add(pairs, k, ids, tokenRows, 1, true, y, 0);
        assertArrayEquals(expected, y, EPS);

        float[] pairRows = LoraFixtures.random(43, pairs * in);
        float[] expectedPairs = y0.clone();
        for (int q = 0; q < pairs; q++) {
            LoraFixtures.project(p, s, ids[q], pairRows, q * in, expectedPairs, q * out);
        }
        float[] y2 = y0.clone();
        host.add(pairs, k, ids, pairRows, 0, false, y2, 0);
        assertArrayEquals(expectedPairs, y2, EPS);
    }

    @Test
    public void expertDeltaRejectsAnExpertOutsideThePair() {
        LoraAdapter.Pair p = LoraFixtures.experts(UP_EXPS, 4, 3, 2, 3, 51);
        LoraAdapter.Delta d = LoraFixtures.adapter("a.gguf", 0.0f, p).delta(UP_EXPS, 1.0f);
        try {
            d.addExperts(1, 1, new int[]{3}, new float[4], 0, false, new float[5], new float[3], 0);
            fail("expert 3 of a three-expert pair must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(UP_EXPS));
        }
    }

    @Test
    public void embeddingDeltaAddsScaledBTimesTheLookedUpARow() {
        int nVocab = 6;
        int width = 4;
        LoraAdapter.Pair p = LoraFixtures.embedding(EMBD, nVocab, width, 2, 61);
        LoraAdapter a = LoraFixtures.adapter("a.gguf", 1.0f, p);
        float s = a.scale(EMBD, 3.0f);
        int[] ids = {9, 5, 0, 3};
        int yOff = 1;
        int yStride = width + 1;
        float[] y0 = LoraFixtures.random(62, yOff + 3 * yStride);
        float[] expected = y0.clone();
        for (int t = 0; t < 3; t++) {
            LoraFixtures.embed(p, s, ids[1 + t], expected, yOff + t * yStride);
        }
        CpuProjectionDeltas.EmbeddingDelta host = (CpuProjectionDeltas.EmbeddingDelta) CpuLoraDeltas.of(
                a.delta(EMBD, 3.0f));
        float[] y = y0.clone();
        host.add(3, ids, 1, y, yOff, yStride);
        assertArrayEquals(expected, y, EPS);
    }

    @Test
    public void aDeltaRefusesTheApplicationOfAnotherKind() {
        LoraAdapter.Pair p = LoraFixtures.experts(UP_EXPS, 4, 3, 2, 3, 71);
        LoraAdapter.Delta d = LoraFixtures.adapter("a.gguf", 0.0f, p).delta(UP_EXPS, 1.0f);
        try {
            d.add(1, new float[4], 0, 4, new float[2], 0, new float[3], 0, 3);
            fail("an expert pair must refuse the dense projection delta");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("EXPERTS"));
        }
    }
}
