package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The grouped-query paged decode ({@code LLM_FlashDecodePagedGqa_<KV>}, kernels/llm/flash_decode_gqa.cl) against a
 * double-precision host attention over the same paged F16, BF16 or Q8_0 cache, and against the paged decode it
 * stands in for ({@code LLM_FlashDecodePaged_<KV>}) on the same arguments: per-sequence key counts (including an empty
 * sequence), shuffled physical pages, unmapped pages, empty cells, causal positions below the cached ones, softcap,
 * and both output forms — the normalized rows of one direct split (with and without sinks) and the (o, m, l)
 * partials of several splits, which the host merges into the normalized rows and which match the paged decode's
 * split by split. Covers query groups of 1, 2 and 4 heads at head dims 64 and 128 and launches of 1 to 16
 * sequences. CUDA only; skipped elsewhere.
 */
public class FlashDecodeGqaGpuTest {

    private static final int PAGE_SHIFT = 4;
    private static final int PAGE = 1 << PAGE_SHIFT;
    private static final int MAX_KEYS = 200;
    private static final int MAX_PAGES = (MAX_KEYS + PAGE - 1) / PAGE + 1;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private final List<GpuBuffer> buffers = new ArrayList<>();

    @Before
    public void openDevice() {
        try {
            HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            backend = new CudaComputeBackend(accel.getCUDevices().get(0));
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            kernels = LlmKernels.open(backend);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable CUDA GPU: " + e.getMessage(), false);
        }
        Assume.assumeTrue("no grouped-query paged decode on this device",
                kernels.has(AttnKernels.flashDecodePagedGqa(KvDType.F16, KvDType.F16)));
    }

    @After
    public void closeDevice() {
        for (GpuBuffer b : buffers) {
            b.close();
        }
        buffers.clear();
        if (backend != null) {
            try {
                backend.close();
            } catch (RuntimeException ignored) {
            }
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
    }

    @Test
    public void everyGroupAndHeadDimMatchesThePagedDecode() throws InterruptedException {
        int seed = 1;
        for (KvDType type : new KvDType[]{KvDType.F16, KvDType.BF16, KvDType.Q8_0}) {
            for (int d : new int[]{64, 128}) {
                for (int g : new int[]{1, 2, 4}) {
                    for (int nSeq : new int[]{1, 3, 16}) {
                        Case c = new Case(type, d, 8 / Math.min(g, 2), g, nSeq, seed++);
                        c.compare(1, 128, 0f, false);
                        c.compare(1, 128, 30f, true);
                        c.compare(4, 64, 0f, false);
                        c.compare(7, 32, 20f, false);
                    }
                }
            }
        }
    }

    /** One random paged cache, query rows and per-sequence geometry. */
    private final class Case {
        final KvDType type;
        final int d;
        final int nHeadKv;
        final int nHead;
        final int nSeq;
        final int[] kvLen;
        final GpuBuffer q;
        final GpuBuffer k;
        final GpuBuffer v;
        final GpuBuffer kvPos;
        final GpuBuffer sinks;
        final GpuBuffer pageTable;
        final GpuBuffer seqKvLen;
        final GpuBuffer qPosSeq;
        final GpuBuffer dummy;
        final GpuBuffer ones;
        final String what;
        final float[] hq;
        final float[] hk;
        final float[] hv;
        final float[] hSinks;
        final int[] hPos;
        final int[] hPt;
        final int[] hQPos;

        Case(KvDType type, int d, int nHeadKv, int g, int nSeq, long seed) throws InterruptedException {
            this.type = type;
            this.d = d;
            this.nHeadKv = nHeadKv;
            this.nHead = nHeadKv * g;
            this.nSeq = nSeq;
            this.what = String.format(Locale.ROOT, "%s d=%d heads=%d/%d seqs=%d", type, d, nHead, nHeadKv, nSeq);
            Random rnd = new Random(seed);
            int physPages = nSeq * MAX_PAGES;
            List<Integer> pool = new ArrayList<>();
            for (int p = 0; p < physPages; p++) {
                pool.add(p);
            }
            Collections.shuffle(pool, rnd);
            int rows = physPages * PAGE;
            int rowElems = nHeadKv * d;
            int[] pos = new int[rows];
            java.util.Arrays.fill(pos, -1);
            int[] pt = new int[nSeq * MAX_PAGES];
            java.util.Arrays.fill(pt, -1);
            kvLen = new int[nSeq];
            int[] qPos = new int[nSeq];
            int next = 0;
            for (int s = 0; s < nSeq; s++) {
                kvLen[s] = s == 1 ? 0 : 1 + rnd.nextInt(MAX_KEYS);
                qPos[s] = s % 3 == 2 ? Math.max(0, kvLen[s] - 1 - rnd.nextInt(20)) : kvLen[s] - 1;
                for (int pg = 0; pg * PAGE < kvLen[s]; pg++) {
                    if (pg > 0 && rnd.nextInt(9) == 0) {
                        continue;   /* an unmapped page: its keys are masked */
                    }
                    int phys = pool.get(next++);
                    pt[s * MAX_PAGES + pg] = phys;
                    for (int i = 0; i < PAGE; i++) {
                        int t = pg * PAGE + i;
                        if (t < kvLen[s] && rnd.nextInt(17) != 0) {
                            pos[phys * PAGE + i] = t;   /* else an empty cell */
                        }
                    }
                }
            }
            hq = randoms(rnd, nSeq * nHead * d, 1f);
            hk = stored(randoms(rnd, rows * rowElems, 1f));
            hv = stored(randoms(rnd, rows * rowElems, 1f));
            hSinks = randoms(rnd, nHead, 2f);
            hPos = pos;
            hPt = pt;
            hQPos = qPos;
            q = floats(hq);
            k = cache(hk);
            v = cache(hv);
            kvPos = ints(pos);
            sinks = floats(hSinks);
            pageTable = ints(pt);
            seqKvLen = ints(kvLen);
            qPosSeq = ints(qPos);
            dummy = floats(new float[16]);
            float[] unit = new float[2 * nHeadKv];
            java.util.Arrays.fill(unit, 1f);
            ones = floats(unit);
        }

        /** Runs both kernels with {@code nSplit} splits of {@code splitLen} keys and compares their outputs. */
        void compare(int nSplit, int splitLen, float softcap, boolean sinksOn) throws InterruptedException {
            int maxLen = 0;
            for (int len : kvLen) {
                maxLen = Math.max(maxLen, len);
            }
            int splits = Math.max(nSplit, (maxLen + splitLen - 1) / splitLen);
            boolean direct = splits == 1;
            int flags = AttnKernels.FLAG_CAUSAL | (direct ? AttnKernels.FLAG_DIRECT : 0)
                    | (sinksOn && direct ? AttnKernels.FLAG_SINKS : 0);
            int outFloats = direct ? nSeq * nHead * d : nSeq * nHead * splits * (d + 2);
            float[] ref = run(AttnKernels.flashDecodePaged(type, type),
                    new long[]{(long) splits * FlashAttnTiling.WG, (long) nSeq * nHead}, splits, splitLen, flags,
                    softcap, direct, outFloats);
            float[] got = run(AttnKernels.flashDecodePagedGqa(type, type),
                    FlashAttnTiling.decodeGqaGlobal(nSeq, nHeadKv, splits), splits, splitLen, flags, softcap,
                    direct, outFloats);
            String where = String.format(Locale.ROOT, "%s splits=%d x %d softcap=%.0f sinks=%b", what, splits,
                    splitLen, softcap, sinksOn && direct);
            double[] host = host(softcap, sinksOn && direct);
            float[] gotRows = direct ? got : merge(got, splits);
            float[] refRows = direct ? ref : merge(ref, splits);
            for (int i = 0; i < host.length; i++) {
                double tol = 1e-4 * Math.max(1.0, Math.abs(host[i]));
                String at = where + " output " + i + " (seq " + i / (nHead * d) + "): host " + host[i];
                assertTrue(at + ", grouped " + gotRows[i], Math.abs(host[i] - gotRows[i]) <= tol);
                assertTrue(at + ", paged " + refRows[i], Math.abs(host[i] - refRows[i]) <= tol);
            }
            if (!direct) {
                for (int i = 0; i < outFloats; i++) {
                    boolean same = (Float.isInfinite(ref[i]) && ref[i] == got[i])
                            || Math.abs(ref[i] - got[i]) <= 2e-5f * Math.max(1f, Math.abs(ref[i]));
                    assertTrue(where + " partial " + i + ": paged " + ref[i] + ", grouped " + got[i], same);
                }
            }
        }

        /** The normalized rows of {@code splits} (o, m, l) partials per (sequence, head), merged on the host. */
        private float[] merge(float[] partial, int splits) {
            float[] rows = new float[nSeq * nHead * d];
            for (int r = 0; r < nSeq * nHead; r++) {
                double m = Double.NEGATIVE_INFINITY;
                for (int sp = 0; sp < splits; sp++) {
                    m = Math.max(m, partial[(r * splits + sp) * (d + 2) + d]);
                }
                double l = 0;
                double[] o = new double[d];
                for (int sp = 0; sp < splits; sp++) {
                    int base = (r * splits + sp) * (d + 2);
                    double ms = partial[base + d];
                    double a = ms == Double.NEGATIVE_INFINITY ? 0 : Math.exp(ms - m);
                    l += partial[base + d + 1] * a;
                    for (int i = 0; i < d; i++) {
                        o[i] += partial[base + i] * a;
                    }
                }
                for (int i = 0; i < d; i++) {
                    rows[r * d + i] = l > 0 ? (float) (o[i] / l) : 0f;
                }
            }
            return rows;
        }

        /** The normalized attention rows of every (sequence, head) in double precision. */
        private double[] host(float softcap, boolean withSinks) {
            double[] out = new double[nSeq * nHead * d];
            double scale = 1.0 / Math.sqrt(d);
            int g = nHead / nHeadKv;
            for (int s = 0; s < nSeq; s++) {
                for (int h = 0; h < nHead; h++) {
                    int kvh = h / g;
                    List<double[]> keys = new ArrayList<>();
                    for (int t = 0; t < kvLen[s]; t++) {
                        int phys = hPt[s * MAX_PAGES + (t >> PAGE_SHIFT)];
                        if (phys < 0) {
                            continue;
                        }
                        int row = phys * PAGE + (t & (PAGE - 1));
                        int pk = hPos[row];
                        if (pk < 0 || pk > hQPos[s]) {
                            continue;
                        }
                        double dot = 0;
                        for (int i = 0; i < d; i++) {
                            dot += hq[(s * nHead + h) * d + i] * hk[(row * nHeadKv + kvh) * d + i];
                        }
                        double sc = dot * scale;
                        if (softcap > 0) {
                            sc = softcap * Math.tanh(sc / softcap);
                        }
                        keys.add(new double[]{sc, row});
                    }
                    double m = withSinks ? hSinks[h] : Double.NEGATIVE_INFINITY;
                    for (double[] e : keys) {
                        m = Math.max(m, e[0]);
                    }
                    double l = withSinks ? Math.exp(hSinks[h] - m) : 0;
                    double[] o = new double[d];
                    for (double[] e : keys) {
                        double p = Math.exp(e[0] - m);
                        l += p;
                        int row = (int) e[1];
                        for (int i = 0; i < d; i++) {
                            o[i] += p * hv[(row * nHeadKv + kvh) * d + i];
                        }
                    }
                    for (int i = 0; i < d; i++) {
                        out[(s * nHead + h) * d + i] = l > 0 ? o[i] / l : 0;
                    }
                }
            }
            return out;
        }

        /** The values the cache stores for {@code x}: rounded or quantized to the cache type. */
        private float[] stored(float[] x) {
            return KvCacheTestValues.stored(type, x);
        }

        private float[] run(String kernel, long[] global, int nSplit, int splitLen, int flags, float softcap,
                            boolean direct, int outFloats) throws InterruptedException {
            float[] poison = new float[outFloats];
            java.util.Arrays.fill(poison, Float.NaN);
            GpuBuffer out = floats(poison);
            Object[] args = new Object[AttnKernels.FDP_ARG_COUNT];
            args[AttnKernels.ARG_FD_N_HEAD] = nHead;
            args[AttnKernels.ARG_FD_N_HEAD_KV] = nHeadKv;
            args[AttnKernels.ARG_FD_D_K] = d;
            args[AttnKernels.ARG_FD_D_V] = d;
            args[AttnKernels.ARG_FD_G_TILE] = 1;
            args[AttnKernels.ARG_FD_N_SPLIT] = nSplit;
            args[AttnKernels.ARG_FD_SPLIT_LEN] = splitLen;
            args[AttnKernels.ARG_FD_KV_START] = 0;
            args[AttnKernels.ARG_FD_KV_END] = 0;
            args[AttnKernels.ARG_FD_Q_POS] = 0;
            args[AttnKernels.ARG_FD_K_ROW_STRIDE] = nHeadKv * d;
            args[AttnKernels.ARG_FD_K_HEAD_STRIDE] = d;
            args[AttnKernels.ARG_FD_V_ROW_STRIDE] = nHeadKv * d;
            args[AttnKernels.ARG_FD_V_HEAD_STRIDE] = d;
            args[AttnKernels.ARG_FD_SCALE] = (float) (1.0 / Math.sqrt(d));
            args[AttnKernels.ARG_FD_SOFTCAP] = softcap;
            args[AttnKernels.ARG_FD_MAX_BIAS] = 0f;
            args[AttnKernels.ARG_FD_FLAGS] = flags;
            args[AttnKernels.ARG_FD_SWA_TYPE] = 0;
            args[AttnKernels.ARG_FD_N_SWA] = 0;
            args[AttnKernels.ARG_FD_BIAS_MODE] = 0;
            args[AttnKernels.ARG_FD_N_BUCKETS] = 0;
            args[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
            args[AttnKernels.ARG_FD_Q_OFF] = 0;
            args[AttnKernels.ARG_FD_OUT_OFF] = 0;
            args[AttnKernels.ARG_FDP_PAGE_SHIFT] = PAGE_SHIFT;
            args[AttnKernels.ARG_FDP_MAX_PAGES] = MAX_PAGES;
            args[AttnKernels.ARG_FDP_N_SEQ] = nSeq;
            args[AttnKernels.ARG_FDP_Q_SEQ_STRIDE] = nHead * d;
            args[AttnKernels.ARG_FDP_OUT_SEQ_STRIDE] = nHead * d;
            args[AttnKernels.ARG_FDP_BUF_Q] = q;
            args[AttnKernels.ARG_FDP_BUF_K] = k;
            args[AttnKernels.ARG_FDP_BUF_V] = v;
            args[AttnKernels.ARG_FDP_BUF_KV_POS] = kvPos;
            args[AttnKernels.ARG_FDP_BUF_SINKS] = sinks;
            args[AttnKernels.ARG_FDP_BUF_REL_BIAS] = dummy;
            args[AttnKernels.ARG_FDP_BUF_OUT] = direct ? out : dummy;
            args[AttnKernels.ARG_FDP_BUF_PARTIAL] = direct ? dummy : out;
            args[AttnKernels.ARG_FDP_BUF_PAGE_TABLE] = pageTable;
            args[AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN] = seqKvLen;
            args[AttnKernels.ARG_FDP_BUF_Q_POS_SEQ] = qPosSeq;
            args[AttnKernels.ARG_FDP_BUF_KV_SCALES] = ones;   /* an unscaled cache's per-head K and V scales */
            args[AttnKernels.ARG_FDP_BUF_TREE_MASK] = dummy;
            backend.computeQueue().enqueueKernel(kernels.get(kernel), global, FlashAttnTiling.decodeLocal(), args);
            float[] y = new float[outFloats];
            backend.computeQueue().enqueueReadBuffer(out, 0, (long) outFloats * 4, y).awaitCompletion();
            return y;
        }

        private GpuBuffer cache(float[] values) throws InterruptedException {
            return bytes(KvCacheTestValues.bytes(type, values));
        }
    }

    private static float[] randoms(Random rnd, int n, float scale) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (rnd.nextFloat() * 2f - 1f) * scale;
        }
        return x;
    }

    private GpuBuffer floats(float[] x) throws InterruptedException {
        GpuBuffer b = backend.allocateBuffer((long) x.length * 4);
        buffers.add(b);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) x.length * 4, x).awaitCompletion();
        return b;
    }

    private GpuBuffer ints(int[] x) throws InterruptedException {
        ByteBuffer bb = ByteBuffer.allocate(x.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : x) {
            bb.putInt(i);
        }
        return bytes(bb.array());
    }

    private GpuBuffer bytes(byte[] x) throws InterruptedException {
        GpuBuffer b = backend.allocateBuffer(x.length);
        buffers.add(b);
        backend.computeQueue().enqueueWriteBuffer(b, 0, x.length, x).awaitCompletion();
        return b;
    }
}
