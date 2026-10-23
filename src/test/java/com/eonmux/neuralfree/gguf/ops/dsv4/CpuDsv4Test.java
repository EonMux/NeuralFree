package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuCompressedKv;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class CpuDsv4Test {

    private static final float EPS = 1e-6f;

    /** F32 single-head key-only cache. */
    private static final class F32Cache implements KvAccessor {
        private final KvLayout layout;
        private final float[] data;

        F32Cache(int rows, int width) {
            this.layout = KvLayout.latent(rows, false, width, width, KvDType.F32, KvWindow.FULL_CAUSAL);
            this.data = new float[rows * width];
        }

        @Override public int layer() { return 0; }
        @Override public KvLayout layout() { return layout; }
        @Override public int group() { return -1; }
        @Override public int nHeadKv() { return 1; }
        @Override public int headDimK() { return layout.widthK(); }
        @Override public int headDimV() { return layout.widthV(); }

        @Override
        public void readK(int row, int head, float[] dst, int off) {
            if (row < 0 || row >= layout.rows() || head != 0) {
                throw new IndexOutOfBoundsException("row " + row + " head " + head);
            }
            System.arraycopy(data, row * headDimK(), dst, off, headDimK());
        }

        @Override
        public void readV(int row, int head, float[] dst, int off) {
            readK(row, head, dst, off);
        }

        @Override
        public void writeK(int row, float[] src, int off) {
            if (row < 0 || row >= layout.rows()) {
                throw new IndexOutOfBoundsException("row " + row);
            }
            System.arraycopy(src, off, data, row * headDimK(), headDimK());
        }

        @Override
        public void writeKSlice(int row, int from, int n, float[] src, int off) {
            if (row < 0 || row >= layout.rows()) {
                throw new IndexOutOfBoundsException("row " + row);
            }
            System.arraycopy(src, off, data, row * headDimK() + from, n);
        }

        @Override
        public void writeV(int row, float[] src, int off) {
            throw new UnsupportedOperationException("key-only cache");
        }

        float[] row(int r) {
            return Arrays.copyOfRange(data, r * headDimK(), (r + 1) * headDimK());
        }
    }

    private static float[] random(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) rnd.nextGaussian();
        }
        return v;
    }

    private static float[] ones(int n) {
        float[] v = new float[n];
        Arrays.fill(v, 1f);
        return v;
    }

    @Test
    public void hadamardMatchesNormalizedSylvesterMatrix() {
        final int n = 32;
        final int block = 16;
        float[] x = random(new Random(1), n);
        float[] y = x.clone();
        CpuDsv4Compressor.hadamard(y, 0, n, block);
        for (int b = 0; b < n / block; b++) {
            for (int i = 0; i < block; i++) {
                double acc = 0;
                for (int j = 0; j < block; j++) {
                    int sign = (Integer.bitCount(i & j) & 1) == 0 ? 1 : -1;
                    acc += sign * x[b * block + j];
                }
                assertEquals(acc / Math.sqrt(block), y[b * block + i], 1e-5);
            }
        }
        CpuDsv4Compressor.hadamard(y, 0, n, block);
        assertArrayEquals(x, y, 1e-5f);
    }

    @Test
    public void ropeInverseRestoresTheRow() {
        for (RopeType type : new RopeType[]{RopeType.NORM, RopeType.NEOX}) {
            Dsv4Kernels.RopeParams rope = Dsv4Kernels.RopeParams.of(8, type, 160000f, 0.25f, 1f,
                    Dsv4Kernels.attnFactor(0.25f, 1f), 1.5f, 3.0f);
            float[] x = random(new Random(2), 24);
            float[] y = x.clone();
            CpuDsv4Compressor.rope(y, 4, 16, rope, 1234, false);
            assertArrayEquals(Arrays.copyOfRange(x, 0, 12), Arrays.copyOfRange(y, 0, 12), 0f);
            CpuDsv4Compressor.rope(y, 4, 16, rope, 1234, true);
            assertArrayEquals(x, y, 1e-4f);
        }
    }

    @Test
    public void ropeRotatesTailPairs() {
        Dsv4Kernels.RopeParams rope = Dsv4Kernels.RopeParams.of(4, RopeType.NORM, 10000f, 1f, 0f, 1f, 0f, 0f);
        float[] x = {1f, 2f, 3f, 4f, 1f, 0f, 0f, 1f};
        CpuDsv4Compressor.rope(x, 0, 8, rope, 3, false);
        double t1 = 3 * Math.pow(10000, -0.5);
        assertEquals(Math.cos(3), x[4], EPS);
        assertEquals(Math.sin(3), x[5], EPS);
        assertEquals(-Math.sin(t1), x[6], EPS);
        assertEquals(Math.cos(t1), x[7], EPS);
        assertEquals(4f, x[3], 0f);
    }

    @Test
    public void attnFactorFollowsDeepSeek4() {
        assertEquals(1f, Dsv4Kernels.attnFactor(0.25f, 0f), 0f);
        assertEquals(1.0 / (1.0 + 0.1 * Math.log(4.0)), Dsv4Kernels.attnFactor(0.25f, 1f), EPS);
    }

    @Test
    public void firstOverlappedBlockPoolsOnlyTheCurrentHalf() {
        final int width = 4;
        final int ratio = Dsv4Spec.RATIO_CSA;
        CpuDsv4Compressor.Config cfg = CpuDsv4Compressor.Config.of(ratio, width, 1e-6f,
                Dsv4Kernels.RopeParams.none(), 0);
        Random rnd = new Random(3);
        int sw = cfg.stateWidth();
        float[] curKv = random(rnd, ratio * sw);
        float[] curScore = random(rnd, ratio * sw);
        float[] ape = random(rnd, ratio * sw);
        float[] normW = random(rnd, width);
        float[] stateKv = random(rnd, cfg.stateRows() * sw);
        float[] stateScore = random(rnd, cfg.stateRows() * sw);
        F32Cache cache = new F32Cache(4, width);
        CpuDsv4Compressor.compress(cfg, new CpuDsv4Compressor.Step(0, ratio, curKv, 0, curScore, 0, ape),
                stateKv, stateScore, normW, cache);

        float[] expect = new float[width];
        for (int c = 0; c < width; c++) {
            double m = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < ratio; t++) {
                m = Math.max(m, curScore[t * sw + width + c] + ape[t * sw + width + c]);
            }
            double l = 0;
            double acc = 0;
            for (int t = 0; t < ratio; t++) {
                double e = Math.exp(curScore[t * sw + width + c] + ape[t * sw + width + c] - m);
                l += e;
                acc += e * curKv[t * sw + width + c];
            }
            expect[c] = (float) (acc / l);
        }
        double ss = 0;
        for (float v : expect) {
            ss += v * v;
        }
        double scale = 1.0 / Math.sqrt(ss / width + 1e-6);
        for (int c = 0; c < width; c++) {
            expect[c] = (float) (expect[c] * scale * normW[c]);
        }
        assertArrayEquals(expect, cache.row(0), 1e-5f);
    }

    /**
     * The indexer (LID) cache has the context's K type (llama-kv-cache-dsv4.cpp:1308-1312): the rows the LID update
     * writes into an F16 or Q8_0 cache are the F32 rows rounded by that type (ggml_set_rows: fp32 to fp16, or
     * quantize_row_q8_0_ref per 32-value block).
     */
    @Test
    public void lidUpdateStoresRowsInTheCacheType() {
        final int width = 64;
        final int ratio = Dsv4Spec.RATIO_CSA;
        final int nTokens = 8;
        CpuDsv4Compressor.Config cfg = CpuDsv4Compressor.Config.of(ratio, width, 1e-6f,
                Dsv4Kernels.RopeParams.none(), 64);
        Random rnd = new Random(11);
        int sw = cfg.stateWidth();
        float[] curKv = random(rnd, nTokens * sw);
        float[] curScore = random(rnd, nTokens * sw);
        float[] ape = random(rnd, ratio * sw);
        float[] normW = random(rnd, width);
        float[] stateKv = new float[cfg.stateRows() * sw];
        float[] stateScore = new float[cfg.stateRows() * sw];
        CpuDsv4Compressor.Step step = new CpuDsv4Compressor.Step(0, nTokens, curKv, 0, curScore, 0, ape);
        F32Cache ref = new F32Cache(4, width);
        CpuDsv4Compressor.lidUpdate(cfg, step, stateKv.clone(), stateScore.clone(), normW, ref);

        for (KvDType type : new KvDType[]{KvDType.F32, KvDType.F16, KvDType.Q8_0}) {
            CpuCompressedKv cache = CpuCompressedKv.allocate(0,
                    CompressedKvPlan.of(CompressedKvPlan.Kind.INDEXER, 16, 512, width, type));
            CpuDsv4Compressor.lidUpdate(cfg, step, stateKv.clone(), stateScore.clone(), normW, cache);
            float[] got = new float[width];
            for (int r = 0; r < nTokens / ratio; r++) {
                cache.readK(r, 0, got, 0);
                assertArrayEquals(type + " row " + r, stored(type, ref.row(r)), got, 0f);
            }
        }
    }

    /** {@code x} after a store into a cache of {@code type} and a read back. */
    private static float[] stored(KvDType type, float[] x) {
        float[] y = new float[x.length];
        switch (type) {
            case F32:
                return x.clone();
            case F16:
                for (int i = 0; i < x.length; i++) {
                    y[i] = FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(x[i]));
                }
                return y;
            default:
                for (int b = 0; b < x.length; b += 32) {
                    float amax = 0f;
                    for (int j = 0; j < 32; j++) {
                        amax = Math.max(amax, Math.abs(x[b + j]));
                    }
                    float d = amax / 127f;
                    float id = d != 0f ? 1f / d : 0f;
                    float d16 = FpBlockDecoders.fp16ToFp32(FpBlockDecoders.fp32ToFp16(d));
                    for (int j = 0; j < 32; j++) {
                        float v = x[b + j] * id;
                        y[b + j] = d16 * (float) (v < 0 ? -Math.floor(-v + 0.5) : Math.floor(v + 0.5));
                    }
                }
                return y;
        }
    }

    @Test
    public void csaSplitUbatchesMatchOneUbatch() {
        checkSplitMatchesWhole(Dsv4Spec.RATIO_CSA, 8, 37, new int[]{5, 1, 3, 9, 2, 17}, 8);
    }

    @Test
    public void hcaSplitUbatchesMatchOneUbatch() {
        checkSplitMatchesWhole(Dsv4Spec.RATIO_HCA, 8, 300, new int[]{100, 31, 1, 128, 40}, 0);
    }

    private static void checkSplitMatchesWhole(int ratio, int width, int nTotal, int[] chunks, int rotBlock) {
        Dsv4Kernels.RopeParams rope = Dsv4Kernels.RopeParams.of(4, RopeType.NEOX, 160000f, 1f, 0f, 1f, 0f, 0f);
        CpuDsv4Compressor.Config cfg = CpuDsv4Compressor.Config.of(ratio, width, 1e-6f, rope, rotBlock);
        Random rnd = new Random(4 + ratio);
        int sw = cfg.stateWidth();
        float[] curKv = random(rnd, nTotal * sw);
        float[] curScore = random(rnd, nTotal * sw);
        float[] ape = random(rnd, ratio * sw);
        float[] normW = random(rnd, width);
        int rows = nTotal / ratio + 1;

        F32Cache whole = new F32Cache(rows, width);
        float[] kvA = new float[cfg.stateRows() * sw];
        float[] scA = new float[cfg.stateRows() * sw];
        CpuDsv4Compressor.Step all = new CpuDsv4Compressor.Step(0, nTotal, curKv, 0, curScore, 0, ape);
        CpuDsv4Compressor.compress(cfg, all, kvA, scA, normW, whole);
        CpuDsv4Compressor.stateUpdate(cfg, all, kvA, scA);

        F32Cache split = new F32Cache(rows, width);
        float[] kvB = new float[cfg.stateRows() * sw];
        float[] scB = new float[cfg.stateRows() * sw];
        int pos = 0;
        for (int n : chunks) {
            CpuDsv4Compressor.Step step = new CpuDsv4Compressor.Step(pos, n, curKv, pos * sw, curScore, pos * sw, ape);
            CpuDsv4Compressor.compress(cfg, step, kvB, scB, normW, split);
            CpuDsv4Compressor.stateUpdate(cfg, step, kvB, scB);
            pos += n;
        }
        assertEquals(nTotal, pos);
        for (int r = 0; r < nTotal / ratio; r++) {
            assertArrayEquals("row " + r, whole.row(r), split.row(r), 1e-6f);
        }
        assertArrayEquals(kvA, kvB, 0f);
        assertArrayEquals(scA, scB, 0f);
    }

    @Test
    public void stateUpdateKeepsTheLatestTokenOfEachRow() {
        final int width = 4;
        CpuDsv4Compressor.Config cfg = CpuDsv4Compressor.Config.of(Dsv4Spec.RATIO_CSA, width, 1e-6f,
                Dsv4Kernels.RopeParams.none(), 0);
        int sw = cfg.stateWidth();
        int pos0 = 5;
        int n = 11;
        float[] curKv = new float[n * sw];
        float[] curScore = new float[n * sw];
        for (int t = 0; t < n; t++) {
            Arrays.fill(curKv, t * sw, (t + 1) * sw, pos0 + t);
        }
        float[] ape = new float[Dsv4Spec.RATIO_CSA * sw];
        for (int r = 0; r < Dsv4Spec.RATIO_CSA; r++) {
            Arrays.fill(ape, r * sw, (r + 1) * sw, 100f * r);
        }
        float[] kv = new float[cfg.stateRows() * sw];
        float[] sc = new float[cfg.stateRows() * sw];
        CpuDsv4Compressor.stateUpdate(cfg, new CpuDsv4Compressor.Step(pos0, n, curKv, 0, curScore, 0, ape), kv, sc);
        for (int r = 0; r < cfg.stateRows(); r++) {
            int latest = -1;
            for (int p = pos0; p < pos0 + n; p++) {
                if (p % cfg.stateRows() == r) {
                    latest = p;
                }
            }
            assertEquals(latest, kv[r * sw], 0f);
            assertEquals(100f * (latest % Dsv4Spec.RATIO_CSA), sc[r * sw + sw - 1], 0f);
        }
    }

    @Test
    public void cacheStoreRotatesAndSkipsNegativeCells() {
        F32Cache cache = new F32Cache(4, 8);
        float[] src = random(new Random(5), 3 * 10);
        CpuDsv4Compressor.cacheStore(3, 8, 8, src, 0, 10, new int[]{2, -1, 0}, cache);
        float[] expect = Arrays.copyOfRange(src, 0, 8);
        CpuDsv4Compressor.hadamard(expect, 0, 8, 8);
        assertArrayEquals(expect, cache.row(2), 0f);
        float[] expect2 = Arrays.copyOfRange(src, 20, 28);
        CpuDsv4Compressor.hadamard(expect2, 0, 8, 8);
        assertArrayEquals(expect2, cache.row(0), 0f);
        assertArrayEquals(new float[8], cache.row(1), 0f);
    }

    @Test
    public void denseAttentionMatchesBruteForce() {
        checkAttention(Dsv4Spec.RATIO_HCA, false, 0);
    }

    @Test
    public void selectedAttentionWithSinksAndRotationMatchesBruteForce() {
        checkAttention(Dsv4Spec.RATIO_CSA, true, 16);
    }

    @Test
    public void windowOnlyAttentionMatchesBruteForce() {
        checkAttention(0, true, 0);
    }

    private static void checkAttention(int ratio, boolean sinks, int rotBlock) {
        final int nHead = 3;
        final int hd = 16;
        final int nSwa = 6;
        final int nTokens = 4;
        final int nRaw = 8;
        final int nCompRows = 10;
        Random rnd = new Random(6 + ratio);
        Dsv4Kernels.RopeParams derope = Dsv4Kernels.RopeParams.of(8, RopeType.NEOX, 10000f, 1f, 0f, 1f, 0f, 0f);
        float[] sinkV = sinks ? random(rnd, nHead) : null;
        CpuDsv4Attention.Layer layer = CpuDsv4Attention.Layer.of(nHead, hd, 0.25f, nSwa, ratio, rotBlock, derope,
                sinkV);

        F32Cache raw = new F32Cache(nRaw, hd);
        final int base = ratio == Dsv4Spec.RATIO_HCA ? 1000 : 40;
        int[] rawPos = {base, base + 1, base + 2, base + 3, base - 4, base - 3, base - 2, -1};
        for (int r = 0; r < nRaw; r++) {
            raw.writeK(r, random(rnd, hd), 0);
        }
        F32Cache compCache = new F32Cache(nCompRows, hd);
        for (int r = 0; r < nCompRows; r++) {
            compCache.writeK(r, random(rnd, hd), 0);
        }
        int[] qPos = {base, base + 1, base + 2, base + 3};
        int nSel = 5;
        int[] sel = new int[nTokens * nSel];
        for (int t = 0; t < nTokens; t++) {
            int[] row = {9, 3, -1, 8 - t, 12};
            System.arraycopy(row, 0, sel, t * nSel, nSel);
        }
        CpuDsv4Attention.Compressed comp;
        if (ratio == 0) {
            comp = CpuDsv4Attention.Compressed.none();
        } else if (ratio == Dsv4Spec.RATIO_CSA) {
            comp = CpuDsv4Attention.Compressed.selected(compCache, nSel, sel, 0, nSel);
        } else {
            comp = CpuDsv4Attention.Compressed.dense(compCache, nCompRows);
        }
        float[] q = random(rnd, nTokens * nHead * hd);
        float[] out = new float[nTokens * nHead * hd];
        CpuDsv4Attention.attend(layer, nTokens, q, 0, qPos, raw, nRaw, rawPos, comp, out, 0);

        for (int t = 0; t < nTokens; t++) {
            int p = qPos[t];
            List<float[]> keys = new ArrayList<>();
            for (int r = 0; r < nRaw; r++) {
                if (rawPos[r] >= 0 && rawPos[r] <= p && p - rawPos[r] < nSwa) {
                    keys.add(raw.row(r));
                }
            }
            if (ratio != 0) {
                int nVis = (p + 1) / ratio;
                for (int j = 0; j < (ratio == Dsv4Spec.RATIO_CSA ? nSel : nCompRows); j++) {
                    int c = ratio == Dsv4Spec.RATIO_CSA ? sel[t * nSel + j] : j;
                    if (c >= 0 && c < nVis) {
                        keys.add(compCache.row(c));
                    }
                }
            }
            for (int h = 0; h < nHead; h++) {
                float[] qh = Arrays.copyOfRange(q, (t * nHead + h) * hd, (t * nHead + h + 1) * hd);
                if (rotBlock > 0) {
                    CpuDsv4Compressor.hadamard(qh, 0, hd, rotBlock);
                }
                double[] s = new double[keys.size()];
                double m = sinks ? sinkV[h] : Double.NEGATIVE_INFINITY;
                for (int k = 0; k < s.length; k++) {
                    double dot = 0;
                    for (int d = 0; d < hd; d++) {
                        dot += qh[d] * keys.get(k)[d];
                    }
                    s[k] = dot * 0.25;
                    m = Math.max(m, s[k]);
                }
                double l = sinks ? Math.exp(sinkV[h] - m) : 0;
                double[] o = new double[hd];
                for (int k = 0; k < s.length; k++) {
                    double e = Math.exp(s[k] - m);
                    l += e;
                    for (int d = 0; d < hd; d++) {
                        o[d] += e * keys.get(k)[d];
                    }
                }
                float[] expect = new float[hd];
                for (int d = 0; d < hd; d++) {
                    expect[d] = l > 0 ? (float) (o[d] / l) : 0f;
                }
                if (rotBlock > 0) {
                    CpuDsv4Compressor.hadamard(expect, 0, hd, rotBlock);
                }
                CpuDsv4Compressor.rope(expect, 0, hd, derope, p, true);
                float[] got = Arrays.copyOfRange(out, (t * nHead + h) * hd, (t * nHead + h + 1) * hd);
                assertArrayEquals("token " + t + " head " + h, expect, got, 1e-4f);
            }
        }
    }

    @Test
    public void kernelGeometry() {
        assertEquals(0, Dsv4Kernels.blockCount(0, 3, 4));
        assertEquals(1, Dsv4Kernels.blockCount(3, 1, 4));
        assertEquals(2, Dsv4Kernels.blockCount(3, 5, 4));
        assertEquals(1, Dsv4Kernels.firstBlock(7, 4));
        assertEquals(2, Dsv4Kernels.visibleRows(11, 4));
        assertEquals(0, Dsv4Kernels.visibleRows(126, 128));
        assertEquals(8, Dsv4Kernels.stateRows(4));
        assertEquals(128, Dsv4Kernels.stateRows(128));
        assertEquals(256, Dsv4Kernels.stateWidth(4, 128));
        assertEquals(256, Dsv4Kernels.compressGlobal(3, 5, 4, 128)[0]);
        assertEquals(8 * 256, Dsv4Kernels.stateUpdateGlobal(4, 128, 128)[0]);
        assertEquals(2 * 2 * 128, Dsv4Kernels.attnGlobal(2, 20, 16, 128)[0]);
        assertEquals(9, Dsv4Kernels.attnHeadTile(512, 64, 128, 11264));
        assertEquals(2, Dsv4Kernels.attnHeadTile(512, 64, 128, 3072));
        assertEquals(3, Dsv4Kernels.attnHeadTile(64, 3, 32, 11264));
        assertEquals(Dsv4Kernels.CompMode.SELECTED, Dsv4Kernels.CompMode.forRatio(4));
        assertEquals(Dsv4Kernels.CompMode.DENSE, Dsv4Kernels.CompMode.forRatio(128));
        assertEquals(Dsv4Kernels.CompMode.NONE, Dsv4Kernels.CompMode.forRatio(0));
        assertEquals(KvDType.Q8_0.code(), Dsv4Kernels.kvTypeCode(KvDType.Q8_0));
        assertEquals(Dsv4Kernels.ROPE_NEOX, Dsv4Kernels.ropeModeCode(RopeType.NEOX));
        assertEquals(5 + KvDType.all().size(), Dsv4Kernels.ALL.size());
        assertTrue(Dsv4Kernels.ALL.contains(Dsv4Kernels.STATE_SNAPSHOT));
        assertEquals(12, Dsv4Kernels.SNAPSHOT_ARG_COUNT);
        assertEquals(24, Dsv4Kernels.COMPRESS_ARG_COUNT);
        assertEquals(22, Dsv4Kernels.LID_ARG_COUNT);
        assertEquals(11, Dsv4Kernels.STATE_ARG_COUNT);
        assertEquals(9, Dsv4Kernels.STORE_ARG_COUNT);
        assertEquals(33, Dsv4Kernels.ATTN_ARG_COUNT);
        assertEquals(13, Dsv4Kernels.ARG_ATTN_SEL_STRIDE);
    }

    @Test
    public void kernelValidationRejectsUnsupportedShapes() {
        expectIae(() -> Dsv4Kernels.checkLocal(48));
        expectIae(() -> Dsv4Kernels.checkLocal(512));
        expectIae(() -> Dsv4Kernels.checkRatio(8));
        expectIae(() -> Dsv4Kernels.checkWidth(1028, KvDType.F16));
        expectIae(() -> Dsv4Kernels.checkWidth(48, KvDType.Q8_0));
        expectIae(() -> Dsv4Kernels.checkRotBlock(48, 96));
        expectIae(() -> Dsv4Kernels.checkRotBlock(64, 96));
        expectIae(() -> Dsv4Kernels.checkLidCache(KvDType.Q8_0, 16, 48));
        expectIae(() -> Dsv4Kernels.checkLidCache(KvDType.F16, 0, 128));
        expectIae(() -> Dsv4Kernels.checkLidCache(KvDType.Q8_0, 0, 128));
        expectIae(() -> Dsv4Kernels.attnHeadTile(1024, 8, 256, 2048));
        expectIae(() -> Dsv4Kernels.checkAttn(8, 512, 4, 128, 3072, KvDType.F16, KvDType.F16, 0,
                Dsv4Kernels.RopeParams.none()));
        expectIae(() -> Dsv4Kernels.ropeModeCode(RopeType.MROPE));
        expectIae(() -> Dsv4Kernels.checkRowWidth(6));
        expectIae(() -> Dsv4Kernels.checkCompress(8, 512, KvDType.F16, 0, Dsv4Kernels.RopeParams.none(), 128));
        expectIae(() -> Dsv4Kernels.checkCompress(4, 512, KvDType.F16, 0, Dsv4Kernels.RopeParams.none(), 96));
        expectIae(() -> Dsv4Kernels.checkCacheStore(48, KvDType.Q8_0, 0, 128));
        expectIae(() -> Dsv4Kernels.RopeParams.of(7, RopeType.NORM, 10000f, 1f, 0f, 1f, 0f, 0f));
        Dsv4Kernels.checkAttn(8, 512, 2, 128, 3072, KvDType.F16, KvDType.Q8_0, 512, Dsv4Kernels.RopeParams.none());
        Dsv4Kernels.checkLidCache(KvDType.F16, 128, 128);
        Dsv4Kernels.checkLidCache(KvDType.F32, 64, 128);
        Dsv4Kernels.checkLidCache(KvDType.Q8_0, 128, 128);
        assertEquals("LLM_DSV4_LidUpdate_F32", Dsv4Kernels.lidUpdate(KvDType.F32));
        assertEquals("LLM_DSV4_LidUpdate_F16", Dsv4Kernels.lidUpdate(KvDType.F16));
        assertEquals("LLM_DSV4_LidUpdate_Q8_0", Dsv4Kernels.lidUpdate(KvDType.Q8_0));
        for (KvDType t : KvDType.all()) {
            assertEquals(t.code(), Dsv4Kernels.kvTypeCode(t));
            assertTrue(Dsv4Kernels.ALL.contains(Dsv4Kernels.lidUpdate(t)));
        }
        Dsv4Kernels.checkLidCache(KvDType.Q4_0, 128, 128);
        Dsv4Kernels.checkLidCache(KvDType.BF16, 4, 100);
        expectIae(() -> Dsv4Kernels.checkWidth(48, KvDType.IQ4_NL));
        Dsv4Kernels.checkCompress(128, 512, KvDType.Q8_0, 512,
                Dsv4Kernels.RopeParams.of(64, RopeType.NORM, 160000f, 1f, 0f, 1f, 0f, 0f), 128);
        Dsv4Kernels.checkCacheStore(512, KvDType.F16, 0, 128);
    }

    private static void expectIae(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage() != null && !expected.getMessage().isEmpty());
        }
    }
}
