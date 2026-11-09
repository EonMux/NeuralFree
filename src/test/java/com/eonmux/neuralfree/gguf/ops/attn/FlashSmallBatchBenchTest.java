package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Random;

/**
 * Opt-in micro-benchmark of the small-batch decode attention: at the multi-row verify shape of a
 * Qwen3-like model (32 query heads over 8 KV heads of 128 dims, F16 caches, causal, a few hundred
 * live keys) it times the three kernels a 2..16-row step can run today &mdash; one
 * {@code LLM_FlashDecode_*} launch per row, one tiled {@code LLM_FlashAttn_*} launch, one tensor-core
 * {@code LLM_FlashAttnMma_*} launch &mdash; each with the split-K partials the engine's
 * {@link FlashAttnTiling} rules pick and their {@code LLM_FlashAttn_Combine} merges. Two timings per
 * variant: back-to-back submission of every round with one final sync (the engine's regime, launch
 * overhead hidden) and a sync after every launch (the launch-latency regime). Opt-in through
 * {@code -Dneuralfree.test.flashbench=true}; the shared suite skips in seconds.
 */
public class FlashSmallBatchBenchTest {

    /** System property that turns the benchmark on. */
    public static final String BENCH_PROPERTY = "neuralfree.test.flashbench";

    private static final int N_HEAD = 32;
    private static final int N_HEAD_KV = 8;
    private static final int D = 128;
    private static final int CONTEXT = 256;
    private static final int[] ROWS = {1, 2, 4, 8, 9, 16};
    private static final int ROUNDS = 200;
    private static final int SYNC_ROUNDS = 40;

    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;

    @Before
    public void open() {
        Assume.assumeTrue("the flash bench is opt-in (" + BENCH_PROPERTY + "=true)",
                Boolean.getBoolean(BENCH_PROPERTY));
        GpuTestUtil.assumeCudaAvailable();
        hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
        backend = hw.createBackend(hw.getCUDevices().get(0));
        Assume.assumeTrue("LLM kernels unavailable on CUDA", LlmKernels.availability(backend).available());
        kernels = LlmKernels.open(backend);
    }

    @After
    public void close() {
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    /** The step's query rows over a filled cache, every variant's input. */
    private static int kvEnd(int n) {
        return CONTEXT + n;
    }

    @Test
    public void benchSmallBatchKernels() {
        String mma = AttnKernels.flashAttnMma(KvDType.F16);
        boolean hasMma = mma != null && kernels.has(mma)
                && backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX);
        System.out.printf(Locale.ROOT,
                "FLASHBENCH device=%s cu=%d smem=%d mma=%s n_head=%d/%d d=%d context=%d%n",
                backend.capabilities().getDeviceName(), backend.capabilities().getComputeUnits(),
                kernels.faSmemFloats(), hasMma, N_HEAD, N_HEAD_KV, D, CONTEXT);
        for (int n : ROWS) {
            benchRow(n, hasMma);
        }
        /* A longer cache: the same row counts against 1024 live keys. */
        System.out.println("FLASHBENCH long cache");
        for (int n : ROWS) {
            benchRowLongCache(n, hasMma);
        }
    }

    private void benchRow(int n, boolean hasMma) {
        benchVariant(n, CONTEXT, "decode-per-row", hasMma, Mode.PER_ROW_DECODE);
        benchVariant(n, CONTEXT, "tiled", hasMma, Mode.TILED);
        if (hasMma) {
            benchVariant(n, CONTEXT, "mma", hasMma, Mode.MMA);
        }
    }

    private void benchRowLongCache(int n, boolean hasMma) {
        benchVariant(n, 1024, "decode-per-row", hasMma, Mode.PER_ROW_DECODE);
        benchVariant(n, 1024, "tiled", hasMma, Mode.TILED);
        if (hasMma) {
            benchVariant(n, 1024, "mma", hasMma, Mode.MMA);
        }
    }

    /** Which kernel the virtual layer launches. */
    private enum Mode {
        PER_ROW_DECODE, TILED, MMA
    }

    /**
     * Times one variant at one row count: {@code rounds} virtual layers are prepared, enqueued
     * back-to-back, and closed with one sync (the pipelined figure), then a second pass syncs after
     * every virtual layer (the launch-latency figure, median of {@code SYNC_ROUNDS}).
     */
    private void benchVariant(int n, int context, String name, boolean hasMma, Mode mode) {
        Layer caseData = prepare(n, context);
        try {
            FlashAttnTiling plan = mode == Mode.PER_ROW_DECODE ? caseData.decodePlan
                    : mode == Mode.TILED ? caseData.tiledPlan : caseData.mmaPlan;
            Runnable[] layers = new Runnable[ROUNDS];
            for (int r = 0; r < ROUNDS; r++) {
                layers[r] = virtualLayer(caseData, mode);
            }
            // warm-up
            for (int r = 0; r < 20; r++) {
                layers[r].run();
            }
            backend.computeQueue().finish();

            long t0 = System.nanoTime();
            for (int r = 0; r < ROUNDS; r++) {
                layers[r].run();
            }
            backend.computeQueue().finish();
            double pipelinedUs = (System.nanoTime() - t0) / 1e3 / ROUNDS;

            long[] samples = new long[SYNC_ROUNDS];
            for (int r = 0; r < SYNC_ROUNDS; r++) {
                long s0 = System.nanoTime();
                layers[r % ROUNDS].run();
                backend.computeQueue().finish();
                samples[r] = System.nanoTime() - s0;
            }
            Arrays.sort(samples);
            double syncUs = samples[SYNC_ROUNDS / 2] / 1e3;

            System.out.printf(Locale.ROOT,
                    "FLASHBENCH-ROW n=%d context=%d variant=%s plan=%s pipelined-us=%.2f sync-us=%.2f%n",
                    n, context, name, plan, pipelinedUs, syncUs);
        } finally {
            caseData.close();
        }
    }

    /** The buffers and plans one row-count case shares across its variants. */
    private static final class Layer {
        final int n;
        final int nKv;
        final FlashAttnTiling decodePlan;
        final FlashAttnTiling tiledPlan;
        final FlashAttnTiling mmaPlan;
        final GpuBuffer q;
        final GpuBuffer k;
        final GpuBuffer v;
        final GpuBuffer qPos;
        final GpuBuffer kvPos;
        final GpuBuffer sinks;
        final GpuBuffer relBias;
        final GpuBuffer out;
        final GpuBuffer partial;
        final GpuBuffer kvScales;
        final GpuBuffer placeholder;
        final long partialFloats;

        Layer(int n, int nKv, FlashAttnTiling decodePlan, FlashAttnTiling tiledPlan, FlashAttnTiling mmaPlan,
              GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer qPos, GpuBuffer kvPos, GpuBuffer sinks,
              GpuBuffer relBias, GpuBuffer out, GpuBuffer partial, GpuBuffer kvScales, GpuBuffer placeholder,
              long partialFloats) {
            this.n = n;
            this.nKv = nKv;
            this.decodePlan = decodePlan;
            this.tiledPlan = tiledPlan;
            this.mmaPlan = mmaPlan;
            this.q = q;
            this.k = k;
            this.v = v;
            this.qPos = qPos;
            this.kvPos = kvPos;
            this.sinks = sinks;
            this.relBias = relBias;
            this.out = out;
            this.partial = partial;
            this.kvScales = kvScales;
            this.placeholder = placeholder;
            this.partialFloats = partialFloats;
        }

        void close() {
            GpuBuffer[] bufs = {q, k, v, qPos, kvPos, sinks, relBias, out, partial, kvScales, placeholder};
            for (GpuBuffer b : bufs) {
                b.close();
            }
        }
    }

    /** Allocates and fills the case of one row count over a cache of {@code context} live keys. */
    private Layer prepare(int n, int context) {
        int nKv = context + n;
        int smem = kernels.faSmemFloats();
        int cu = Math.max(1, backend.capabilities().getComputeUnits());
        FlashAttnTiling decodePlan = FlashAttnTiling.decode(N_HEAD, N_HEAD_KV, D, D, nKv, smem, cu,
                backend.capabilities().hasFeature(DeviceFeature.SUBGROUP_SHUFFLE));
        FlashAttnTiling tiledPlan = FlashAttnTiling.prefill(N_HEAD, N_HEAD_KV, D, D, nKv, smem, cu);
        FlashAttnTiling mmaPlan = FlashAttnTiling.mma(N_HEAD, N_HEAD_KV, D, D, nKv, smem, cu);

        Random rnd = new Random(7);
        float[] qh = new float[n * N_HEAD * D];
        for (int i = 0; i < qh.length; i++) {
            qh[i] = rnd.nextFloat() - 0.5f;
        }
        short[] kh = new short[nKv * N_HEAD_KV * D];
        short[] vh = new short[nKv * N_HEAD_KV * D];
        for (int i = 0; i < kh.length; i++) {
            kh[i] = f16(rnd.nextFloat() - 0.5f);
            vh[i] = f16(rnd.nextFloat() - 0.5f);
        }
        int[] qPositions = new int[n];
        for (int t = 0; t < n; t++) {
            qPositions[t] = context + t;
        }
        int[] kvPositions = new int[nKv];
        for (int j = 0; j < nKv; j++) {
            kvPositions[j] = j;
        }
        // the largest partial every variant may write: the engine's prefill bound capped at
        // MAX_PREFILL_SPLITS, and the decode plan's own bound
        long prefillFloats = FlashAttnTiling.partialFloats((long) n * N_HEAD,
                FlashAttnTiling.MAX_PREFILL_SPLITS, D);
        long partialFloats = Math.max(prefillFloats,
                FlashAttnTiling.partialFloats((long) N_HEAD, decodePlan.maxSplits(), D));

        GpuBuffer q = backend.allocateBuffer((long) qh.length * 4, true);
        GpuBuffer k = backend.allocateBuffer((long) kh.length * 2, true);
        GpuBuffer v = backend.allocateBuffer((long) vh.length * 2, true);
        GpuBuffer qPos = backend.allocateBuffer((long) n * 4, true);
        GpuBuffer kvPos = backend.allocateBuffer((long) nKv * 4, true);
        GpuBuffer sinkBuf = backend.allocateBuffer((long) N_HEAD * 4, true);
        GpuBuffer relBias = backend.allocateBuffer(4, true);
        GpuBuffer out = backend.allocateBuffer((long) n * N_HEAD * D * 4, true);
        GpuBuffer partial = backend.allocateBuffer(partialFloats * 4, true);
        GpuBuffer kvScales = backend.allocateBuffer((long) 2 * N_HEAD_KV * 4, true);
        GpuBuffer placeholder = backend.allocateBuffer(16 * 4, true);

        float[] ones = new float[2 * N_HEAD_KV];
        Arrays.fill(ones, 1f);
        enqueue(kvScales, ones);
        enqueue(q, qh);
        enqueueHalfs(k, kh);
        enqueueHalfs(v, vh);
        enqueue(qPos, qPositions);
        enqueue(kvPos, kvPositions);
        backend.computeQueue().finish();
        return new Layer(n, nKv, decodePlan, tiledPlan, mmaPlan, q, k, v, qPos, kvPos, sinkBuf, relBias, out,
                partial, kvScales, placeholder, partialFloats);
    }

    /** The launches of one virtual layer in {@code mode}: what the engine enqueues per layer. */
    private Runnable virtualLayer(Layer c, Mode mode) {
        switch (mode) {
            case PER_ROW_DECODE:
                return () -> {
                    for (int t = 0; t < c.n; t++) {
                        decodeRow(c, t);
                    }
                };
            case TILED:
                return () -> tiledLayer(c);
            case MMA:
                return () -> mmaLayer(c);
            default:
                throw new IllegalStateException(mode.toString());
        }
    }

    /** One decode launch of one query row, the engine's {@code decodeRow} rules. */
    private void decodeRow(Layer c, int row) {
        int nKeys = c.nKv;
        FlashAttnTiling.Split s = c.decodePlan.decodeSplit(nKeys);
        int count = s.count();
        boolean direct = s.direct();
        int qPos = CONTEXT + row;
        Object[] args = new Object[AttnKernels.FD_ARG_COUNT];
        args[AttnKernels.ARG_FD_N_HEAD] = N_HEAD;
        args[AttnKernels.ARG_FD_N_HEAD_KV] = N_HEAD_KV;
        args[AttnKernels.ARG_FD_D_K] = D;
        args[AttnKernels.ARG_FD_D_V] = D;
        args[AttnKernels.ARG_FD_G_TILE] = c.decodePlan.gTile();
        args[AttnKernels.ARG_FD_N_SPLIT] = count;
        args[AttnKernels.ARG_FD_SPLIT_LEN] = s.length();
        args[AttnKernels.ARG_FD_KV_START] = 0;
        args[AttnKernels.ARG_FD_KV_END] = nKeys;
        args[AttnKernels.ARG_FD_Q_POS] = qPos;
        args[AttnKernels.ARG_FD_K_ROW_STRIDE] = N_HEAD_KV * D;
        args[AttnKernels.ARG_FD_K_HEAD_STRIDE] = D;
        args[AttnKernels.ARG_FD_V_ROW_STRIDE] = N_HEAD_KV * D;
        args[AttnKernels.ARG_FD_V_HEAD_STRIDE] = D;
        args[AttnKernels.ARG_FD_SCALE] = 0.125f;
        args[AttnKernels.ARG_FD_SOFTCAP] = 0f;
        args[AttnKernels.ARG_FD_MAX_BIAS] = 0f;
        args[AttnKernels.ARG_FD_FLAGS] = AttnKernels.attnFlags(true, false, direct);
        args[AttnKernels.ARG_FD_SWA_TYPE] = SwaType.NONE.code();
        args[AttnKernels.ARG_FD_N_SWA] = 0;
        args[AttnKernels.ARG_FD_BIAS_MODE] = 0;
        args[AttnKernels.ARG_FD_N_BUCKETS] = 0;
        args[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
        args[AttnKernels.ARG_FD_Q_OFF] = row * N_HEAD * D;
        args[AttnKernels.ARG_FD_OUT_OFF] = row * N_HEAD * D;
        args[AttnKernels.ARG_FD_BUF_Q] = c.q;
        args[AttnKernels.ARG_FD_BUF_K] = c.k;
        args[AttnKernels.ARG_FD_BUF_V] = c.v;
        args[AttnKernels.ARG_FD_BUF_KV_POS] = c.kvPos;
        args[AttnKernels.ARG_FD_BUF_SINKS] = c.sinks;
        args[AttnKernels.ARG_FD_BUF_REL_BIAS] = c.relBias;
        args[AttnKernels.ARG_FD_BUF_OUT] = c.out;
        args[AttnKernels.ARG_FD_BUF_PARTIAL] = c.partial;
        args[AttnKernels.ARG_FD_BUF_KV_SCALES] = c.kvScales;
        args[AttnKernels.ARG_FD_BUF_TREE_MASK] = c.placeholder;
        dispatch(AttnKernels.flashDecode(KvDType.F16), c.decodePlan.decodeGlobal(count),
                new long[]{FlashAttnTiling.WG, 1}, args);
        if (!direct) {
            combine(c, N_HEAD, count, row * N_HEAD * D);
        }
    }

    /** One tiled prefill launch and its combine, the engine's {@code prefill} rules. */
    private void tiledLayer(Layer c) {
        FlashAttnTiling plan = c.tiledPlan;
        FlashAttnTiling.Split s = plan.prefillSplit(c.n, c.nKv, c.partialFloats);
        int splits = s.count();
        boolean direct = s.direct();
        Object[] args = tiledArgs(c, plan, splits, s.length(), direct);
        dispatch(AttnKernels.flashAttn(KvDType.F16), plan.prefillGlobal(c.n, splits),
                new long[]{FlashAttnTiling.WG, 1, 1}, args);
        if (!direct) {
            combine(c, c.n * N_HEAD, splits, 0);
        }
    }

    /** One tensor-core prefill launch and its combine, the engine's {@code prefill} rules. */
    private void mmaLayer(Layer c) {
        FlashAttnTiling plan = c.mmaPlan;
        FlashAttnTiling.Split s = plan.prefillSplit(c.n, c.nKv, c.partialFloats);
        int splits = s.count();
        boolean direct = s.direct();
        Object[] args = mmaArgs(c, plan, splits, s.length(), direct);
        dispatch("LLM_FlashAttnMma_F16", plan.prefillGlobal(c.n, splits),
                new long[]{FlashAttnTiling.WG, 1, 1}, args);
        if (!direct) {
            combine(c, c.n * N_HEAD, splits, 0);
        }
    }

    /** The {@code LLM_FlashAttn_F16} argument list of one call. */
    private Object[] tiledArgs(Layer c, FlashAttnTiling plan, int nSplit, int splitLen, boolean direct) {
        return new Object[]{
                c.n, N_HEAD, N_HEAD_KV, D, D, plan.bq(), plan.bk(), plan.gTile(), nSplit, splitLen,
                0, c.nKv, N_HEAD * D, N_HEAD * D, N_HEAD_KV * D, D, N_HEAD_KV * D, D,
                0.125f, 0f, 0f,
                AttnKernels.attnFlags(true, false, direct),
                SwaType.NONE.code(), 0, 0, 0, 0,
                0, 0,
                c.q, c.k, c.v, c.qPos, c.kvPos, c.sinks, c.relBias, c.out, c.partial, c.kvScales,
                c.placeholder};
    }

    /** The {@code LLM_FlashAttnMma_F16} argument list of one call (no kv_scales buffer). */
    private Object[] mmaArgs(Layer c, FlashAttnTiling plan, int nSplit, int splitLen, boolean direct) {
        return new Object[]{
                c.n, N_HEAD, N_HEAD_KV, D, D, plan.bq(), plan.bk(), plan.gTile(), nSplit, splitLen,
                0, c.nKv, N_HEAD * D, N_HEAD * D, N_HEAD_KV * D, D, N_HEAD_KV * D, D,
                0.125f, 0f, 0f,
                AttnKernels.attnFlags(true, false, direct),
                SwaType.NONE.code(), 0, 0, 0, 0,
                0, 0,
                c.q, c.k, c.v, c.qPos, c.kvPos, c.sinks, c.relBias, c.out, c.partial, c.placeholder};
    }

    /** The merge launch of {@code rows} partial rows at output offset {@code outOff}. */
    private void combine(Layer c, int rows, int splits, int outOff) {
        Object[] args = new Object[]{rows, D, splits, N_HEAD, 0, outOff, c.partial, c.sinks, c.out};
        dispatch(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(rows),
                new long[]{FlashAttnTiling.WG}, args);
    }

    private void dispatch(String kernel, long[] global, long[] local, Object[] args) {
        GpuEvent e = backend.computeQueue().enqueueKernel(kernels.get(kernel), global, local, args);
        if (e != null) {
            e.close();
        }
    }

    private void enqueue(GpuBuffer buf, float[] data) {
        GpuEvent e = backend.computeQueue().enqueueWriteBuffer(buf, 0, (long) data.length * 4, data);
        if (e != null) {
            e.close();
        }
    }

    private void enqueue(GpuBuffer buf, int[] data) {
        byte[] bytes = new byte[data.length * 4];
        for (int i = 0; i < data.length; i++) {
            bytes[4 * i] = (byte) data[i];
            bytes[4 * i + 1] = (byte) (data[i] >> 8);
            bytes[4 * i + 2] = (byte) (data[i] >> 16);
            bytes[4 * i + 3] = (byte) (data[i] >> 24);
        }
        GpuEvent e = backend.computeQueue().enqueueWriteBuffer(buf, 0, bytes.length, bytes);
        if (e != null) {
            e.close();
        }
    }

    private void enqueueHalfs(GpuBuffer buf, short[] data) {
        byte[] bytes = new byte[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            bytes[2 * i] = (byte) (data[i] & 0xff);
            bytes[2 * i + 1] = (byte) ((data[i] >> 8) & 0xff);
        }
        GpuEvent e = backend.computeQueue().enqueueWriteBuffer(buf, 0, bytes.length, bytes);
        if (e != null) {
            e.close();
        }
    }

    /** IEEE 754 binary16 rounding to nearest even, as the kernels' half stores. */
    private static short f16(float f) {
        int bits = Float.floatToIntBits(f);
        int sign = (bits >>> 16) & 0x8000;
        int mag = bits & 0x7fffffff;
        if (mag >= 0x7f800000) {
            return (short) (sign | 0x7c00 | (mag == 0x7f800000 ? 0 : 1));
        }
        int exp = (mag >>> 23) - 127 + 15;
        int frac = mag & 0x7fffff;
        if (exp >= 31) {
            return (short) (sign | 0x7c00);
        }
        if (exp <= 0) {
            return (short) sign;
        }
        int mant = frac >>> 13;
        int rest = frac & 0x1fff;
        if (rest > 0x1000 || (rest == 0x1000 && (mant & 1) == 1)) {
            mant++;
            if (mant == 0x400) {
                mant = 0;
                exp++;
                if (exp >= 31) {
                    return (short) (sign | 0x7c00);
                }
            }
        }
        return (short) (sign | (exp << 10) | mant);
    }
}
