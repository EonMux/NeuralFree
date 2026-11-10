package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.vulkan.VkInstanceManager;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Parity of the tensor-core mma prefill kernels ({@code LLM_FlashAttnMma_F16} of kernels/llm/flash_mma.cl on CUDA,
 * of shaders/llm/flash_mma_cm2.comp on Vulkan) against the CPU reference {@link CpuAttentionCore} on a device that
 * runs them ({@link LlmKernels#flashMmaPath()}): causal prefill and multi-row decode (split-K partials merged by
 * {@code LLM_FlashAttn_Combine}), GQA packing, softcap, ALiBi, sinks and the SWA windows, with F16 K and V caches.
 * The Vulkan case opens the first device reporting {@link DeviceFeature#COOPERATIVE_MATRIX_2_ATTENTION}. Skipped
 * where no device offers the kernel (the OpenCL and HIP builds define no mma attention).
 */
@RunWith(Parameterized.class)
public class FlashMmaKernelGpuTest {

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{ComputeBackend.BackendType.CUDA},
                new Object[]{ComputeBackend.BackendType.VULKAN});
    }

    private final ComputeBackend.BackendType api;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;

    public FlashMmaKernelGpuTest(ComputeBackend.BackendType api) {
        this.api = api;
    }

    @Before
    public void open() {
        if (api == ComputeBackend.BackendType.CUDA) {
            GpuTestUtil.assumeCudaAvailable();
            hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            backend = hw.createBackend(hw.getCUDevices().get(0));
        } else {
            GpuTestUtil.assumeVulkanAvailable();
            backend = openVulkanAttentionDevice();
            Assume.assumeTrue("no Vulkan device with the cooperative-matrix-2 attention tile", backend != null);
        }
        Assume.assumeTrue("LLM kernels unavailable on " + api, LlmKernels.availability(backend).available());
        kernels = LlmKernels.open(backend);
        Assume.assumeTrue("device runs no mma attention", kernels.flashMmaPath());
        Assume.assumeTrue("no mma kernel in the " + api + " index", kernels.has("LLM_FlashAttnMma_F16"));
    }

    /** The first Vulkan device reporting the cooperative-matrix-2 attention tile, or null. */
    private static ComputeBackend openVulkanAttentionDevice() {
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            for (VkPhysicalDeviceInfo info : manager.devices()) {
                if (info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX_2_ATTENTION)) {
                    return new VkComputeBackend(manager, info);
                }
            }
            return null;
        } finally {
            manager.release();
        }
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

    /** Cache layout of a case: dense rows, or 64-key pages mapped in order or in reverse. */
    private enum Paging { DENSE, IDENTITY, REVERSED }

    /**
     * Runs one shape twice against the reference: split-K prefill over all {@code nKv} rows with the combine,
     * then the direct-out form of the same call, and asserts both within {@code tol}. A paged case launches
     * {@code LLM_FlashAttnMmaPaged_F16} over 64-key pages: the K, V and position rows of logical page p sit at
     * physical page {@code pages[p]}, the identity or the reversed map.
     */
    private void runCase(int nHead, int nHeadKv, int d, int nKv, int nTokens, CpuAttentionCore.Params params,
                         float tol, Paging paging) {
        final boolean paged = paging != Paging.DENSE;
        final int G = nHead / nHeadKv;
        final int gTile = Math.min(Math.min(G, FlashAttnTiling.MMA_MAX_G_TILE), FlashAttnTiling.MMA_ROWS);
        final int bq = FlashAttnTiling.MMA_ROWS / gTile;
        final Random rnd = new Random(42);
        float[] q = new float[nTokens * nHead * d];
        for (int i = 0; i < q.length; i++) {
            q[i] = rnd.nextFloat() - 0.5f;
        }
        short[] k = new short[nKv * nHeadKv * d];
        short[] v = new short[nKv * nHeadKv * d];
        for (int i = 0; i < k.length; i++) {
            k[i] = f16(rnd.nextFloat() - 0.5f);
            v[i] = f16(rnd.nextFloat() - 0.5f);
        }
        int[] qPos = new int[nTokens];
        int[] kvPos = new int[nKv];
        for (int t = 0; t < nTokens; t++) {
            qPos[t] = t;
        }
        for (int j = 0; j < nKv; j++) {
            kvPos[j] = j;
        }
        float[] sinks = new float[nHead];
        for (int h = 0; h < nHead; h++) {
            sinks[h] = rnd.nextFloat() * 2f - 1f;
        }
        float[] relBias = new float[32 * nHead];
        for (int i = 0; i < relBias.length; i++) {
            relBias[i] = rnd.nextFloat() - 0.5f;
        }

        // CPU reference over the F16-rounded cache values, with the device buffers' sink and bias tables
        CpuAttentionCore.Params reference = reference(params, sinks, relBias);
        float[] kF32 = new float[k.length];
        float[] vF32 = new float[v.length];
        for (int i = 0; i < k.length; i++) {
            kF32[i] = f32(k[i]);
        }
        for (int i = 0; i < v.length; i++) {
            vF32[i] = f32(v[i]);
        }
        CpuAttentionCore.DenseKv dense = CpuAttentionCore.DenseKv.of(kF32, 0, nHeadKv * d, d,
                vF32, 0, nHeadKv * d, d, nHeadKv, d, d, kvPos, 0);
        float[] want = new float[nTokens * nHead * d];
        CpuAttentionCore.attendRows(reference, nTokens, q, 0, nHead * d, qPos, 0, dense, 0, nKv, want, 0,
                nHead * d);

        int[] pages = new int[(nKv + 63) / 64];
        for (int p = 0; p < pages.length; p++) {
            pages[p] = paging == Paging.REVERSED ? pages.length - 1 - p : p;
        }
        int rows = paged ? pages.length * 64 : nKv;
        int rowLen = nHeadKv * d;
        short[] kRows = new short[rows * rowLen];
        short[] vRows = new short[rows * rowLen];
        int[] posRows = new int[rows];
        Arrays.fill(posRows, -1);
        for (int j = 0; j < nKv; j++) {
            int row = paged ? pages[j / 64] * 64 + j % 64 : j;
            System.arraycopy(k, j * rowLen, kRows, row * rowLen, rowLen);
            System.arraycopy(v, j * rowLen, vRows, row * rowLen, rowLen);
            posRows[row] = kvPos[j];
        }
        GpuBuffer dq = backend.allocateBuffer((long) q.length * 4, true);
        GpuBuffer dk = backend.allocateBuffer((long) kRows.length * 2, true);
        GpuBuffer dv = backend.allocateBuffer((long) vRows.length * 2, true);
        GpuBuffer dqp = backend.allocateBuffer((long) nTokens * 4, true);
        GpuBuffer dkv = backend.allocateBuffer((long) rows * 4, true);
        GpuBuffer dsinks = backend.allocateBuffer((long) nHead * 4, true);
        GpuBuffer drel = backend.allocateBuffer((long) relBias.length * 4, true);
        GpuBuffer dout = backend.allocateBuffer((long) nTokens * nHead * d * 4, true);
        GpuBuffer dph = backend.allocateBuffer(4, true);
        // the mask slot stays bound as in the engine's launches; the tree flag alone admits it
        GpuBuffer dmask = backend.allocateBuffer((long) nTokens * 8, true);
        GpuBuffer dpt = backend.allocateBuffer((long) pages.length * 4, true);
        enqueue(dpt, pages);
        try {
            enqueue(dq, q);
            enqueueHalfs(dk, kRows);
            enqueueHalfs(dv, vRows);
            enqueue(dqp, qPos);
            enqueue(dkv, posRows);
            enqueue(dsinks, sinks);
            enqueue(drel, relBias);

            FlashAttnTiling plan = FlashAttnTiling.mma(nHead, nHeadKv, d, d, nKv, kernels.faSmemFloats(),
                    Math.max(1, backend.capabilities().getComputeUnits()));
            Assume.assumeTrue("no mma plan for this shape", plan != null);
            int splitLen = 96;
            int nSplit = (nKv + splitLen - 1) / splitLen;
            GpuBuffer dpartial = backend.allocateBuffer(
                    (long) nTokens * nHead * nSplit * (d + 2) * 4, true);
            try {
                String kernel = paged ? "LLM_FlashAttnMmaPaged_F16" : "LLM_FlashAttnMma_F16";
                Object[] args = paged
                        ? mmaPagedArgs(nTokens, nHead, nHeadKv, d, plan, nSplit, splitLen, nKv, params, false,
                                dq, dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dmask, dpt)
                        : mmaArgs(nTokens, nHead, nHeadKv, d, plan, nSplit, splitLen, nKv, params, false,
                                dq, dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dmask);
                dispatch(kernel, plan.prefillGlobal(nTokens, nSplit), args);
                Object[] combine = new Object[]{
                        nTokens * nHead, d, nSplit, nHead, reference.hasSinks() ? 1 : 0, 0,
                        dpartial, dsinks, dout};
                dispatch("LLM_FlashAttn_Combine", FlashAttnTiling.combineGlobal(nTokens * nHead), combine);
                float[] got = new float[nTokens * nHead * d];
                read(dout, got);
                double err = 0;
                for (int i = 0; i < got.length; i++) {
                    err = Math.max(err, Math.abs(got[i] - want[i]));
                }
                assertTrue("split-K outputs differ from the CPU reference by " + err, err <= tol);

                // direct out: one split over the whole range, the kernel normalizes itself
                Object[] direct = paged
                        ? mmaPagedArgs(nTokens, nHead, nHeadKv, d, plan, 1, nKv, nKv, params, true,
                                dq, dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dmask, dpt)
                        : mmaArgs(nTokens, nHead, nHeadKv, d, plan, 1, nKv, nKv, params, true,
                                dq, dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dmask);
                dispatch(kernel, plan.prefillGlobal(nTokens, 1), direct);
                read(dout, got);
                err = 0;
                for (int i = 0; i < got.length; i++) {
                    err = Math.max(err, Math.abs(got[i] - want[i]));
                }
                assertTrue("direct outputs differ from the CPU reference by " + err, err <= tol);
            } finally {
                dpartial.close();
            }
        } finally {
            close(dq, dk, dv, dqp, dkv, dsinks, drel, dout, dph, dmask, dpt);
        }
    }

    /** The reference params carrying the device buffers' sink and relative-bias tables. */
    private static CpuAttentionCore.Params reference(CpuAttentionCore.Params base, float[] sinks,
                                                     float[] relBias) {
        CpuAttentionCore.Builder b = CpuAttentionCore.Params.builder(base.nHead())
                .scale(base.scale())
                .softcap(base.softcap())
                .window(base.window());
        if (base.biasMode() == BiasMode.ALIBI) {
            b.alibi(base.maxBias());
        } else if (base.biasMode() == BiasMode.T5_CAUSAL
                || base.biasMode() == BiasMode.T5_BIDIR) {
            b.t5(base.biasMode() == BiasMode.T5_BIDIR, base.nBuckets(), base.maxDistance(),
                    relBias);
        }
        if (base.hasSinks()) {
            b.sinks(sinks);
        }
        return b.build();
    }

    /** The {@code LLM_FlashAttnMma_F16} argument list for one call (29 scalars, then the 10 buffers). */
    private static Object[] mmaArgs(int nTokens, int nHead, int nHeadKv, int d, FlashAttnTiling plan,
                                    int nSplit, int splitLen, int nKv, CpuAttentionCore.Params params,
                                    boolean direct, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer qPos,
                                    GpuBuffer kvPos, GpuBuffer sinks, GpuBuffer relBias, GpuBuffer out,
                                    GpuBuffer partial, GpuBuffer treeMask) {
        KvWindow w = params.window();
        return new Object[]{
                nTokens, nHead, nHeadKv, d, d, plan.bq(), plan.bk(), plan.gTile(), nSplit, splitLen,
                0, nKv, nHead * d, nHead * d, nHeadKv * d, d, nHeadKv * d, d,
                params.scale(), params.softcap(), params.maxBias(),
                AttnKernels.attnFlags(w.causal(), params.hasSinks(), direct),
                w.type().code(), w.windowed() ? w.nSwa() : 0,
                params.biasMode().code(), params.nBuckets(), params.maxDistance(),
                0, 0,
                q, k, v, qPos, kvPos, sinks, relBias, out, partial, treeMask};
    }

    /**
     * The {@code LLM_FlashAttnMmaPaged_F16} argument list: page_shift and pt_off follow the base scalars
     * (positions 29, 30), the base buffers shift behind them, and the page-table buffer closes the list (64-key
     * pages).
     */
    private static Object[] mmaPagedArgs(int nTokens, int nHead, int nHeadKv, int d, FlashAttnTiling plan,
                                         int nSplit, int splitLen, int nKv, CpuAttentionCore.Params params,
                                         boolean direct, GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer qPos,
                                         GpuBuffer kvPos, GpuBuffer sinks, GpuBuffer relBias, GpuBuffer out,
                                         GpuBuffer partial, GpuBuffer treeMask, GpuBuffer pageTable) {
        Object[] base = mmaArgs(nTokens, nHead, nHeadKv, d, plan, nSplit, splitLen, nKv, params, direct,
                q, k, v, qPos, kvPos, sinks, relBias, out, partial, treeMask);
        Object[] args = new Object[base.length + 3];
        System.arraycopy(base, 0, args, 0, 29);
        args[29] = 6;                                            // page_shift: 64-key pages
        args[30] = 0;                                            // pt_off
        System.arraycopy(base, 29, args, 31, 9);                 // q .. partial
        args[40] = pageTable;
        args[41] = base[38];                                     // the tree mask follows the page table
        return args;
    }

    private void dispatch(String kernel, long[] global, Object[] args) {
        long[] local = global.length == 3 ? FlashAttnTiling.prefillLocal() : FlashAttnTiling.combineLocal();
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

    private void read(GpuBuffer buf, float[] dst) {
        GpuEvent e = backend.computeQueue().enqueueReadBuffer(buf, 0, (long) dst.length * 4, dst);
        if (e != null) {
            e.close();
        }
        backend.computeQueue().finish();
    }

    private static void close(GpuBuffer... bufs) {
        for (GpuBuffer b : bufs) {
            b.close();
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

    private static float f32(short h) {
        int sign = (h & 0x8000) << 16;
        int exp = (h >>> 10) & 0x1f;
        int mant = h & 0x3ff;
        int bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                int e = -1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    e++;
                }
                mant &= 0x3ff;
                bits = sign | ((127 - 15 - e) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7f800000 | (mant << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(bits);
    }

    @Test
    public void causalGqaPrefillMatchesTheReference() {
        runCase(8, 2, 64, 96, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .window(KvWindow.FULL_CAUSAL).build(), 3e-3f, Paging.DENSE);
    }

    @Test
    public void causalGqaHeadDim128MatchesTheReference() {
        runCase(8, 2, 128, 160, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .window(KvWindow.FULL_CAUSAL).build(), 4e-3f, Paging.DENSE);
    }

    @Test
    public void batchedVerifyWithSinksAndSoftcapMatchesTheReference() {
        runCase(16, 2, 64, 400, 32, CpuAttentionCore.Params.builder(16).scale(0.125f).softcap(8f)
                .window(KvWindow.FULL_CAUSAL).sinks(new float[16]).build(), 3e-3f, Paging.DENSE);
    }

    @Test
    public void alibiMatchesTheReference() {
        runCase(8, 8, 64, 128, 8, CpuAttentionCore.Params.builder(8).scale(0.125f).alibi(8f)
                .window(KvWindow.FULL_CAUSAL).build(), 3e-3f, Paging.DENSE);
    }

    @Test
    public void swaStandardMatchesTheReference() {
        runCase(8, 2, 64, 300, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .window(KvWindow.of(SwaType.STANDARD, 64, true)).build(), 3e-3f, Paging.DENSE);
    }

    @Test
    public void bidirectionalMatchesTheReference() {
        runCase(8, 2, 64, 96, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .window(KvWindow.FULL_BIDIRECTIONAL).build(), 3e-3f, Paging.DENSE);
    }

    @Test
    public void pagedIdentityPagesMatchTheReference() {
        runCase(8, 2, 64, 160, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .window(KvWindow.FULL_CAUSAL).build(), 3e-3f, Paging.IDENTITY);
    }

    @Test
    public void pagedReversedPagesMatchTheReference() {
        runCase(8, 2, 128, 300, 40, CpuAttentionCore.Params.builder(8).scale(0.0884f)
                .window(KvWindow.FULL_CAUSAL).build(), 4e-3f, Paging.REVERSED);
    }

    @Test
    public void t5BidirectionalBiasMatchesTheReference() {
        runCase(8, 2, 64, 96, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .t5(true, 32, 128, new float[32 * 8]).window(KvWindow.FULL_BIDIRECTIONAL).build(), 3e-3f,
                Paging.DENSE);
    }
}
