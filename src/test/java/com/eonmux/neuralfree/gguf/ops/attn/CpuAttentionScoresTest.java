package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuAttnKernels;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;
import com.eonmux.neuralfree.gguf.cpu.quant.ScalarAttnKernels;
import org.junit.Assume;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The float primitives of the CPU attention core and the scores built on them: the Vector API kernels against their
 * scalar reference bit for bit, and the core's single-row and query-block calls against a plain float loop in the
 * one-pair order ({@code dot += q[d] * k[d]}, then the softmax and {@code o += p v} in key order), bit for bit.
 */
public class CpuAttentionScoresTest {

    private static float[] floats(Random rnd, int n) {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = rnd.nextFloat() * 4f - 2f;
        }
        return f;
    }

    private static void assertBits(String what, float[] expected, float[] actual, int from, int to) {
        for (int i = from; i < to; i++) {
            assertEquals(what + " [" + i + "]", Float.floatToRawIntBits(expected[i]),
                    Float.floatToRawIntBits(actual[i]));
        }
    }

    @Test
    public void theScalarDotsRunEachUsedLaneInTheOnePairOrderAndSkipThePadding() {
        Random rnd = new Random(3);
        final int lanes = 16;
        final int dK = 37;
        final int keys = 7;
        float[] qT = floats(rnd, dK * lanes);
        float[] k = floats(rnd, 2 + keys * (dK + 5));
        for (int used : new int[]{1, 3, 8, 9, 16}) {
            float[] out = new float[1 + keys * lanes];
            Arrays.fill(out, Float.NaN);
            ScalarAttnKernels.INSTANCE.dots(qT, lanes, used, k, 2, dK + 5, keys, dK, out, 1);
            for (int j = 0; j < keys; j++) {
                for (int l = 0; l < lanes; l++) {
                    final int at = 1 + j * lanes + l;
                    if (l >= used) {
                        assertTrue("used " + used + ": padding lane " + l + " of key " + j + " is not computed",
                                Float.isNaN(out[at]));
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < dK; d++) {
                        dot += qT[d * lanes + l] * k[2 + j * (dK + 5) + d];
                    }
                    assertEquals("used " + used + ": key " + j + " lane " + l, Float.floatToRawIntBits(dot),
                            Float.floatToRawIntBits(out[at]));
                }
            }
            assertTrue("out[0] is left alone", Float.isNaN(out[0]));
        }
    }

    @Test
    public void theVectorDotsEqualTheScalarDotsExactly() {
        Assume.assumeTrue("the Vector API engine is unavailable on this JVM", CpuSimd.available());
        CpuAttnKernels vector = CpuSimd.vectorAttention();
        assertNotSame("the engine serves its Vector API kernels", ScalarAttnKernels.INSTANCE, vector);
        Random rnd = new Random(11);
        for (int lanes : new int[]{8, 16, 24, 64}) {
            for (int used : new int[]{1, lanes / 2 + 1, lanes - 7, lanes}) {
                for (int dK : new int[]{1, 7, 64, 128, 130}) {
                    for (int keys = 0; keys <= 9; keys++) {
                        final int stride = dK + 3;
                        float[] qT = floats(rnd, dK * lanes);
                        float[] k = floats(rnd, 4 + keys * stride);
                        float[] want = new float[2 + keys * lanes];
                        float[] got = new float[2 + keys * lanes];
                        ScalarAttnKernels.INSTANCE.dots(qT, lanes, used, k, 4, stride, keys, dK, want, 2);
                        vector.dots(qT, lanes, used, k, 4, stride, keys, dK, got, 2);
                        for (int j = 0; j < keys; j++) {
                            assertBits("lanes " + lanes + " used " + used + " dK " + dK + " key " + j, want, got,
                                    2 + j * lanes, 2 + j * lanes + used);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void theVectorAccumulationEqualsTheScalarAccumulationExactly() {
        Assume.assumeTrue("the Vector API engine is unavailable on this JVM", CpuSimd.available());
        CpuAttnKernels vector = CpuSimd.vectorAttention();
        Random rnd = new Random(12);
        for (int n = 0; n <= 70; n++) {
            float[] v = floats(rnd, n + 5);
            float[] want = floats(rnd, n + 3);
            float[] got = want.clone();
            float w = rnd.nextFloat();
            ScalarAttnKernels.INSTANCE.accumulate(v, 4, w, want, 2, n);
            vector.accumulate(v, 4, w, got, 2, n);
            assertBits("n=" + n, want, got, 0, want.length);
        }
    }

    @Test
    public void bothKernelsRefuseALaneCountThatIsNotWholeVectorsOrUsedLanesOutsideIt() {
        float[] buf = new float[64];
        CpuAttnKernels vector = CpuSimd.vectorAttention();
        for (CpuAttnKernels kernels : new CpuAttnKernels[]{ScalarAttnKernels.INSTANCE,
                vector != null ? vector : ScalarAttnKernels.INSTANCE}) {
            assertThrows(kernels + " lanes", IllegalArgumentException.class,
                    () -> kernels.dots(buf, 12, 12, buf, 0, 1, 1, 1, buf, 0));
            assertThrows(kernels + " no lanes", IllegalArgumentException.class,
                    () -> kernels.dots(buf, 0, 0, buf, 0, 1, 1, 1, buf, 0));
            assertThrows(kernels + " no used lane", IllegalArgumentException.class,
                    () -> kernels.dots(buf, 8, 0, buf, 0, 1, 1, 1, buf, 0));
            assertThrows(kernels + " more used lanes than the layout", IllegalArgumentException.class,
                    () -> kernels.dots(buf, 8, 9, buf, 0, 1, 1, 1, buf, 0));
        }
    }

    /** A {@link CpuAttnKernels} that records the lane counts of every {@link #dots} call and runs the scalar kernels. */
    private static final class RecordingKernels implements CpuAttnKernels {
        final java.util.List<int[]> dots = new java.util.ArrayList<>();

        @Override
        public void dots(float[] qT, int lanes, int used, float[] k, int kOff, int kStride, int nKeys, int dK,
                         float[] out, int outOff) {
            dots.add(new int[]{lanes, used});
            ScalarAttnKernels.INSTANCE.dots(qT, lanes, used, k, kOff, kStride, nKeys, dK, out, outOff);
        }

        @Override
        public void accumulate(float[] v, int vOff, float weight, float[] acc, int accOff, int n) {
            ScalarAttnKernels.INSTANCE.accumulate(v, vOff, weight, acc, accOff, n);
        }

        @Override
        public String describe() {
            return "recording";
        }
    }

    /** Key rows {@code r} at position {@code r}, K and V head {@code h} of row {@code r} at {@code (r * heads + h) * d}. */
    private static CpuAttentionCore.Rows rows(float[] k, float[] v, int heads, int dK, int dV) {
        return new CpuAttentionCore.Rows() {
            @Override public int nHeadKv() { return heads; }
            @Override public int dK() { return dK; }
            @Override public int dV() { return dV; }
            @Override public int pos(int row) { return row; }

            @Override
            public void readK(int row, int head, float[] dst, int off) {
                System.arraycopy(k, (row * heads + head) * dK, dst, off, dK);
            }

            @Override
            public void readV(int row, int head, float[] dst) {
                System.arraycopy(v, (row * heads + head) * dV, dst, 0, dV);
            }
        };
    }

    /**
     * The scoring of a unit whose {@code nt * gc} (row, head) pairs do not fill whole vectors hands the kernels that
     * pair count as the used lanes, so the padding lanes are never scored: a query block of 3 rows x 3 heads (9 of 16
     * lanes) and a single row of 3 heads (3 of 8 lanes), each over two key groups.
     */
    @Test
    public void theScoringPassesTheKernelsTheUsedLanesOfAPaddedUnit() {
        final int nHead = 6;
        final int gc = 3;
        final int dK = 16;
        final int dV = 8;
        final int keys = 2 * CpuAttentionScores.KEY_GROUP + 3;
        Random rnd = new Random(9);
        float[] q = floats(rnd, 3 * nHead * dK);
        CpuAttentionCore.Rows rows = rows(floats(rnd, keys * 2 * dK), floats(rnd, keys * 2 * dV), 2, dK, dV);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(nHead).scale(0.25f).build();
        int[] qPos = {keys - 3, keys - 2, keys - 1};
        for (int nt : new int[]{3, 1}) {
            CpuAttentionCore.Scratch s = new CpuAttentionCore.Scratch();
            s.ensure(dK, dV, nt * gc);
            if (nt == 1) {
                CpuAttentionScores.prepareRow(s, q, nHead * dK, qPos[1], 1, gc, gc, dK);
            } else {
                CpuAttentionScores.prepare(s, q, 0, nHead * dK, nt, qPos, 0, gc, gc, dK);
            }
            RecordingKernels recording = new RecordingKernels();
            s.kernels = recording;
            CpuAttentionScores.score(p, rows, 0, keys, s, gc, 1, null, Integer.MAX_VALUE);
            final int lanes = nt == 1 ? 8 : 16;
            assertEquals(nt + " rows: one dots call per key group", 3, recording.dots.size());
            for (int[] call : recording.dots) {
                assertEquals(nt + " rows: the layout's lanes", lanes, call[0]);
                assertEquals(nt + " rows: the used lanes are the unit's (row, head) pairs", nt * gc, call[1]);
            }
        }
    }

    /**
     * An invalid kernel property is reported as the {@link IllegalArgumentException} that names it at every
     * attention call, not as a class initialization error on the first call and a missing class on the next.
     */
    @Test
    public void anInvalidKernelPropertyIsRejectedByNameAtEveryAttentionCall() throws Exception {
        String out = AttentionKernelProbe.run("-D" + CpuSimd.PROP_SIMD + "=maybe");
        assertTrue(out, out.contains("call1=IllegalArgumentException"));
        assertTrue(out, out.contains("call2=IllegalArgumentException"));
        assertTrue(out, out.contains(CpuSimd.PROP_SIMD + "=maybe"));
        String width = AttentionKernelProbe.run("-D" + CpuSimd.PROP_WIDTH + "=512");
        assertTrue(width, width.contains("call1=IllegalArgumentException"));
        assertTrue(width, width.contains("call2=IllegalArgumentException"));
        String attention = AttentionKernelProbe.run("-D" + CpuSimd.PROP_ATTENTION + "=yes");
        assertTrue(attention, attention.contains("call1=IllegalArgumentException"));
        assertTrue(attention, attention.contains("call2=IllegalArgumentException"));
        assertTrue(attention, attention.contains(CpuSimd.PROP_ATTENTION + "=yes"));
        String valid = AttentionKernelProbe.run("-D" + CpuSimd.PROP_SIMD + "=false");
        assertTrue(valid, valid.contains("call1=ok"));
        assertTrue(valid, valid.contains("call2=ok"));
    }

    /**
     * The core runs the Vector API kernels wherever the engine is available unless {@value CpuSimd#PROP_ATTENTION}
     * is {@code false}, and the scalar kernels without the engine; either way its query-block and single-row outputs
     * are those of the one-pair loop, bit for bit.
     */
    @Test
    public void theVectorKernelsServeTheCoreByDefaultWithTheSameBits() throws Exception {
        AttentionKernelProbe.GqaCase c = new AttentionKernelProbe.GqaCase();
        final int want = AttentionKernelProbe.bitsHash(onePairReference(c.q, c.k, c.v, c.nHead, c.nHeadKv, c.dK,
                c.dV, c.keys, c.nt, c.first, c.scale));
        final String[][] cases = {
                {},
                {"-D" + CpuSimd.PROP_ATTENTION + "=false"},
                {"-D" + CpuSimd.PROP_ATTENTION + "= TRUE "},
                {"-D" + CpuSimd.PROP_SIMD + "=false"},
        };
        final boolean[] vector = {true, false, true, false};
        for (int i = 0; i < cases.length; i++) {
            String out = AttentionKernelProbe.run(cases[i]);
            assertEquals(out, vector[i] && CpuSimd.available(), out.contains("kernels=vector/"));
            assertEquals(out, !(vector[i] && CpuSimd.available()),
                    out.contains("kernels=" + ScalarAttnKernels.INSTANCE.describe() + "\n"));
            assertTrue(out, out.contains("blocks=" + want + "\n"));
            assertTrue(out, out.contains("single=" + want + "\n"));
        }
    }

    /**
     * Causal attention of {@code nt} query rows at positions {@code first + t} over {@code keys} cache rows with the
     * scalar loops of one (row, head) pair: within one key block, {@code m = max s}, {@code p = e^(s - m)},
     * {@code l = sum p}, {@code o = sum p v} (both in key order, a zero weight skipped) and {@code out = o * 1 / l}.
     */
    private static float[] onePairReference(float[] q, float[] k, float[] v, int nHead, int nHeadKv, int dK, int dV,
                                            int keys, int nt, int first, float scale) {
        final int g0 = nHead / nHeadKv;
        float[] out = new float[nt * nHead * dV];
        float[] s = new float[keys];
        for (int t = 0; t < nt; t++) {
            for (int h = 0; h < nHead; h++) {
                final int kvh = h / g0;
                float m = Float.NEGATIVE_INFINITY;
                for (int r = 0; r < keys; r++) {
                    if (r > first + t) {
                        s[r] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < dK; d++) {
                        dot += q[(t * nHead + h) * dK + d] * k[(r * nHeadKv + kvh) * dK + d];
                    }
                    s[r] = dot * scale;
                    m = Math.max(m, s[r]);
                }
                float l = 0.0f;
                float[] o = new float[dV];
                for (int r = 0; r < keys; r++) {
                    final float e = s[r] == Float.NEGATIVE_INFINITY ? 0.0f : (float) Math.exp(s[r] - m);
                    l += e;
                    if (e != 0.0f) {
                        for (int d = 0; d < dV; d++) {
                            o[d] += e * v[(r * nHeadKv + kvh) * dV + d];
                        }
                    }
                }
                for (int d = 0; d < dV; d++) {
                    out[(t * nHead + h) * dV + d] = o[d] * 1.0f / l;
                }
            }
        }
        return out;
    }

    @Test
    public void singleRowsAndQueryBlocksEqualTheOnePairLoopBitForBit() {
        AttentionKernelProbe.GqaCase c = new AttentionKernelProbe.GqaCase();
        float[] want = onePairReference(c.q, c.k, c.v, c.nHead, c.nHeadKv, c.dK, c.dV, c.keys, c.nt, c.first,
                c.scale);
        float[] rows = c.queryBlocks();
        assertBits("query blocks", want, rows, 0, want.length);
        float[] single = c.singleRows();
        assertBits("single rows", want, single, 0, want.length);
    }
}
