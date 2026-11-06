package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Parity of the paged tiled prefill kernels ({@code LLM_FlashAttnPaged_<KV>} and {@code LLM_FlashAttnPaged_Mixed} of
 * kernels/llm/flash_prefill.cl, the GLSL twin shaders/llm/flash_prefill.comp) against {@link CpuAttentionCore} on
 * every GPU API with a device: a shuffled page table with a masked page, the packed {@code flags} argument (causal,
 * sinks, direct output) in its split-K form merged by {@code LLM_FlashAttn_Combine} and its direct form, GQA, softcap
 * and the sliding window. The Vulkan run binds the whole scalar list through the 128-byte push block.
 */
@RunWith(Parameterized.class)
public class FlashAttnPagedKernelGpuTest {

    /** log2 of the page size: 16-slot pages. */
    private static final int PAGE_SHIFT = 4;
    private static final int PAGE = 1 << PAGE_SHIFT;
    /** Leading table entries of other segments; the tested segment's block starts here. */
    private static final int PT_OFF = 3;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> apis() {
        List<Object[]> rows = new ArrayList<>();
        for (ComputeBackend.BackendType t : EnumSet.of(ComputeBackend.BackendType.OPENCL,
                ComputeBackend.BackendType.CUDA, ComputeBackend.BackendType.VULKAN)) {
            rows.add(new Object[]{t});
        }
        return rows;
    }

    private final ComputeBackend.BackendType api;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;

    public FlashAttnPagedKernelGpuTest(ComputeBackend.BackendType api) {
        this.api = api;
    }

    @Before
    public void open() {
        switch (api) {
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            default:
                GpuTestUtil.assumeVulkanAvailable();
                break;
        }
        hw = new HWAcceleration(EnumSet.of(api));
        HardwareDevice device = firstDevice();
        Assume.assumeTrue("no " + api + " device enumerated", device != null);
        backend = hw.createBackend(device);
        Availability a = LlmKernels.availability(backend);
        Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + a.reason(), a.available());
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

    private HardwareDevice firstDevice() {
        switch (api) {
            case OPENCL:
                return hw.getCLDevices().isEmpty() ? null : hw.getCLDevices().get(0);
            case CUDA:
                return hw.getCUDevices().isEmpty() ? null : hw.getCUDevices().get(0);
            default:
                return hw.getVkDevices().isEmpty() ? null : hw.getVkDevices().get(0);
        }
    }

    @Test
    public void causalGqaPrefillWithSinksAndSoftcapOverShuffledPages() {
        runCase(KvDType.F32, KvDType.F32, 8, 2, 64, 120, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .softcap(8f).window(KvWindow.FULL_CAUSAL).sinks(new float[8]).build(), 1e-4f);
    }

    @Test
    public void f16CacheSlidingWindowOverShuffledPages() {
        runCase(KvDType.F16, KvDType.F16, 8, 4, 64, 150, 12, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .window(KvWindow.of(SwaType.STANDARD, 40, true)).build(), 1e-4f);
    }

    @Test
    public void bidirectionalPrefillOverShuffledPages() {
        runCase(KvDType.F32, KvDType.F32, 4, 4, 64, 100, 6, CpuAttentionCore.Params.builder(4).scale(0.125f)
                .window(KvWindow.FULL_BIDIRECTIONAL).build(), 1e-4f);
    }

    /** {@code LLM_FlashAttnPaged_Mixed}: the widest scalar list (32 scalars, the full push block). */
    @Test
    public void mixedCacheTypesWithSinksOverShuffledPages() {
        runCase(KvDType.F32, KvDType.F16, 8, 2, 64, 130, 8, CpuAttentionCore.Params.builder(8).scale(0.125f)
                .window(KvWindow.FULL_CAUSAL).sinks(new float[8]).build(), 1e-4f);
    }

    /**
     * A verify step's tree mask over the same shuffled pages: the segment's last five slots are the fresh rows —
     * the pending root, two depth-1 siblings and two deeper nodes — each query row ropes at its depth position
     * (siblings share one) and sees the committed prefix plus exactly its ancestors and itself, so the split-K
     * launch and the direct launch both equal the CPU tree reference row by row. The rows' stored positions sit in
     * every row's causal past, so without the mask the same launch would read the siblings' slots: the parity bites
     * the kernel's tree walk itself, not merely the ordinary masking.
     */
    @Test
    public void treeMaskedPrefillOverShuffledPages() {
        Random rnd = new Random(17);
        int nHead = 8;
        int nHeadKv = 2;
        int d = 64;
        int nFresh = 5;
        int nKv = 120;
        int nTokens = nFresh;
        int rowWidth = nHeadKv * d;
        int firstFresh = nKv - nFresh;
        int nPages = (nKv + PAGE - 1) / PAGE;
        int physPages = nPages + 2;
        int[] pageTable = new int[PT_OFF + nPages];
        for (int i = 0; i < PT_OFF; i++) {
            pageTable[i] = -7;
        }
        int[] phys = shuffledPages(physPages, rnd);
        for (int p = 0; p < nPages; p++) {
            pageTable[PT_OFF + p] = p == 1 ? -1 : phys[p];
        }
        float[] kPhys = new float[physPages * PAGE * rowWidth];
        float[] vPhys = new float[physPages * PAGE * rowWidth];
        int[] kvPosPhys = new int[physPages * PAGE];
        for (int i = 0; i < kPhys.length; i++) {
            kPhys[i] = rnd.nextFloat() - 0.5f;
            vPhys[i] = rnd.nextFloat() - 0.5f;
        }
        Arrays.fill(kvPosPhys, -1);
        float[] kLog = new float[nKv * rowWidth];
        float[] vLog = new float[nKv * rowWidth];
        int[] kvPosLog = new int[nKv];
        for (int j = 0; j < nKv; j++) {
            int page = pageTable[PT_OFF + (j >> PAGE_SHIFT)];
            if (page < 0) {
                kvPosLog[j] = -1;
                continue;
            }
            int row = (page << PAGE_SHIFT) | (j & (PAGE - 1));
            kvPosPhys[row] = j;
            kvPosLog[j] = j;
            System.arraycopy(kPhys, row * rowWidth, kLog, j * rowWidth, rowWidth);
            System.arraycopy(vPhys, row * rowWidth, vLog, j * rowWidth, rowWidth);
        }
        // fresh row 0 is the pending root, rows 1 and 2 are its depth-1 children (siblings), row 3 is row 1's
        // depth-2 child and row 4 that node's depth-3 child; a row's mask names its ancestors and itself
        int[] parent = {-1, 0, 0, 1, 3};
        long[] mask = new long[nFresh];
        int[] depth = new int[nFresh];
        int[] qPos = new int[nTokens];
        for (int t = 0; t < nFresh; t++) {
            mask[t] = 1L << t;
            depth[t] = parent[t] < 0 ? 0 : depth[parent[t]] + 1;
            qPos[t] = firstFresh + depth[t];
            for (int a = parent[t]; a >= 0; a = parent[a]) {
                mask[t] |= 1L << a;
            }
        }
        float[] q = new float[nTokens * nHead * d];
        for (int i = 0; i < q.length; i++) {
            q[i] = rnd.nextFloat() - 0.5f;
        }
        CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(nHead).scale(0.125f)
                .window(KvWindow.FULL_CAUSAL).build();
        float[] want = new float[nTokens * nHead * d];
        CpuAttentionCore.attendRows(params, nTokens, q, 0, nHead * d, qPos, 0,
                CpuAttentionCore.DenseKv.of(kLog, 0, rowWidth, d, vLog, 0, rowWidth, d, nHeadKv, d, d, kvPosLog, 0),
                0, nKv, want, 0, nHead * d, new CpuAttentionCore.TreeMask(mask, nFresh));

        FlashAttnTiling plan = FlashAttnTiling.prefill(nHead, nHeadKv, d, d, nKv, kernels.faSmemFloats(),
                Math.max(1, backend.capabilities().getComputeUnits()));
        Assume.assumeTrue("no tiled plan for this shape: " + plan, plan.plan() == FlashAttnTiling.Plan.TILED);
        int splitLen = plan.bk();
        int nSplit = (nKv + splitLen - 1) / splitLen;
        assertTrue("the split-K launch needs several splits", nSplit > 1);

        List<GpuBuffer> bufs = new ArrayList<>();
        try {
            GpuBuffer dq = upload(bufs, q);
            GpuBuffer dk = upload(bufs, kPhys);
            GpuBuffer dv = upload(bufs, vPhys);
            GpuBuffer dqp = upload(bufs, qPos);
            GpuBuffer dkv = upload(bufs, kvPosPhys);
            GpuBuffer dsinks = upload(bufs, new float[nHead]);
            GpuBuffer drel = upload(bufs, new float[1]);
            GpuBuffer dpt = upload(bufs, pageTable);
            GpuBuffer dscales = upload(bufs, ones(2 * nHeadKv));
            GpuBuffer dmask = upload(bufs, mask);
            GpuBuffer dout = alloc(bufs, (long) want.length * 4);
            GpuBuffer dpartial = alloc(bufs, (long) nTokens * nHead * nSplit * (d + 2) * 4);
            String kernel = AttnKernels.flashAttnPaged(KvDType.F32, KvDType.F32);

            Object[] split = treeArgs(plan, nTokens, nHead, nHeadKv, d, nKv, nSplit, splitLen, params, false, dq,
                    dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dpt, dscales, dmask, nFresh);
            dispatch(kernel, plan.prefillGlobal(nTokens, nSplit), FlashAttnTiling.prefillLocal(), split);
            dispatch(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(nTokens * nHead),
                    FlashAttnTiling.combineLocal(), new Object[]{nTokens * nHead, d, nSplit, nHead, 0, 0,
                            dpartial, dsinks, dout});
            assertClose("tree split-K", want, read(dout, want.length), 1e-4f);

            write(dout, new float[want.length]);
            Object[] direct = treeArgs(plan, nTokens, nHead, nHeadKv, d, nKv, 1, nSplit * splitLen, params, true,
                    dq, dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dpt, dscales, dmask, nFresh);
            dispatch(kernel, plan.prefillGlobal(nTokens, 1), FlashAttnTiling.prefillLocal(), direct);
            assertClose("tree direct", want, read(dout, want.length), 1e-4f);
        } finally {
            for (GpuBuffer b : bufs) {
                b.close();
            }
        }
    }

    @Test
    public void pagedScalarListsFitTheVulkanPushBlock() {
        // every paged prefill scalar list fits the Vulkan spec's 128-byte push-constant minimum: the fresh-row
        // count rides in the flags word (bits 5-31), so the widest list is the mixed kernel's 32 scalars exactly
        assertEquals(128, AttnKernels.FAP_MIXED_SCALAR_COUNT * 4);
        assertTrue(AttnKernels.FAP_SCALAR_COUNT * 4 <= 128);
    }

    /**
     * One segment of {@code nKv} slots on {@code nPages} shuffled physical pages (page {@code 1} masked with id -1)
     * and {@code nTokens} query rows at the segment's last positions: the split-K launch merged by the combine and
     * the direct launch, each asserted against the reference within {@code tol}.
     */
    private void runCase(KvDType typeK, KvDType typeV, int nHead, int nHeadKv, int d, int nKv, int nTokens,
                         CpuAttentionCore.Params base, float tol) {
        Random rnd = new Random(7);
        int nPages = (nKv + PAGE - 1) / PAGE;
        int physPages = nPages + 2;
        int[] pageTable = new int[PT_OFF + nPages];
        for (int i = 0; i < PT_OFF; i++) {
            pageTable[i] = -7;
        }
        int[] phys = shuffledPages(physPages, rnd);
        for (int p = 0; p < nPages; p++) {
            pageTable[PT_OFF + p] = p == 1 ? -1 : phys[p];
        }
        int rowWidth = nHeadKv * d;
        float[] kPhys = new float[physPages * PAGE * rowWidth];
        float[] vPhys = new float[physPages * PAGE * rowWidth];
        int[] kvPosPhys = new int[physPages * PAGE];
        for (int i = 0; i < kPhys.length; i++) {
            kPhys[i] = round(typeK, rnd.nextFloat() - 0.5f);
            vPhys[i] = round(typeV, rnd.nextFloat() - 0.5f);
        }
        Arrays.fill(kvPosPhys, -1);
        // logical (dense) view of the segment for the reference: slot j at position j, masked page -> -1
        float[] kLog = new float[nKv * rowWidth];
        float[] vLog = new float[nKv * rowWidth];
        int[] kvPosLog = new int[nKv];
        for (int j = 0; j < nKv; j++) {
            int page = pageTable[PT_OFF + (j >> PAGE_SHIFT)];
            if (page < 0) {
                kvPosLog[j] = -1;
                continue;
            }
            int row = (page << PAGE_SHIFT) | (j & (PAGE - 1));
            kvPosPhys[row] = j;
            kvPosLog[j] = j;
            System.arraycopy(kPhys, row * rowWidth, kLog, j * rowWidth, rowWidth);
            System.arraycopy(vPhys, row * rowWidth, vLog, j * rowWidth, rowWidth);
        }
        float[] q = new float[nTokens * nHead * d];
        for (int i = 0; i < q.length; i++) {
            q[i] = rnd.nextFloat() - 0.5f;
        }
        int[] qPos = new int[nTokens];
        for (int t = 0; t < nTokens; t++) {
            qPos[t] = nKv - nTokens + t;
        }
        float[] sinks = new float[nHead];
        for (int h = 0; h < nHead; h++) {
            sinks[h] = rnd.nextFloat() * 2f - 1f;
        }
        CpuAttentionCore.Params params = withSinks(base, sinks);
        float[] want = new float[nTokens * nHead * d];
        CpuAttentionCore.attendRows(params, nTokens, q, 0, nHead * d, qPos, 0,
                CpuAttentionCore.DenseKv.of(kLog, 0, rowWidth, d, vLog, 0, rowWidth, d, nHeadKv, d, d, kvPosLog, 0),
                0, nKv, want, 0, nHead * d);

        FlashAttnTiling plan = FlashAttnTiling.prefill(nHead, nHeadKv, d, d, nKv, kernels.faSmemFloats(),
                Math.max(1, backend.capabilities().getComputeUnits()));
        Assume.assumeTrue("no tiled plan for this shape: " + plan, plan.plan() == FlashAttnTiling.Plan.TILED);
        int splitLen = plan.bk();
        int nSplit = (nKv + splitLen - 1) / splitLen;
        assertTrue("the split-K launch needs several splits", nSplit > 1);

        List<GpuBuffer> bufs = new ArrayList<>();
        try {
            GpuBuffer dq = upload(bufs, q);
            GpuBuffer dk = upload(bufs, typeK, kPhys);
            GpuBuffer dv = upload(bufs, typeV, vPhys);
            GpuBuffer dqp = upload(bufs, qPos);
            GpuBuffer dkv = upload(bufs, kvPosPhys);
            GpuBuffer dsinks = upload(bufs, sinks);
            GpuBuffer drel = upload(bufs, new float[1]);
            GpuBuffer dpt = upload(bufs, pageTable);
            GpuBuffer dscales = upload(bufs, ones(2 * nHeadKv));
            // the mask slot stays bound as in the engine's launches; the tree flag alone admits it
            GpuBuffer dmask = upload(bufs, new long[nTokens]);
            GpuBuffer dout = alloc(bufs, (long) want.length * 4);
            GpuBuffer dpartial = alloc(bufs, (long) nTokens * nHead * nSplit * (d + 2) * 4);
            String kernel = AttnKernels.flashAttnPaged(typeK, typeV);
            Object[] split = args(typeK, typeV, plan, nTokens, nHead, nHeadKv, d, nKv, nSplit, splitLen, params,
                    false, dq, dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dpt, dscales, dmask);
            dispatch(kernel, plan.prefillGlobal(nTokens, nSplit), FlashAttnTiling.prefillLocal(), split);
            dispatch(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(nTokens * nHead),
                    FlashAttnTiling.combineLocal(), new Object[]{nTokens * nHead, d, nSplit, nHead,
                            params.hasSinks() ? 1 : 0, 0, dpartial, dsinks, dout});
            assertClose("split-K", want, read(dout, want.length), tol);

            write(dout, new float[want.length]);
            Object[] direct = args(typeK, typeV, plan, nTokens, nHead, nHeadKv, d, nKv, 1, nSplit * splitLen,
                    params, true, dq, dk, dv, dqp, dkv, dsinks, drel, dout, dpartial, dpt, dscales, dmask);
            dispatch(kernel, plan.prefillGlobal(nTokens, 1), FlashAttnTiling.prefillLocal(), direct);
            assertClose("direct", want, read(dout, want.length), tol);
        } finally {
            for (GpuBuffer b : bufs) {
                b.close();
            }
        }
    }

    /** The paged argument list: base scalars, page geometry, base buffers, page table, kv_scales, the tree mask, then kv_types if mixed. */
    private static Object[] args(KvDType typeK, KvDType typeV, FlashAttnTiling plan, int nTokens, int nHead,
                                 int nHeadKv, int d, int nKv, int nSplit, int splitLen,
                                 CpuAttentionCore.Params p, boolean direct, GpuBuffer q, GpuBuffer k, GpuBuffer v,
                                 GpuBuffer qPos, GpuBuffer kvPos, GpuBuffer sinks, GpuBuffer relBias, GpuBuffer out,
                                 GpuBuffer partial, GpuBuffer pageTable, GpuBuffer kvScales, GpuBuffer treeMask) {
        KvWindow w = p.window();
        Object[] a = new Object[AttnKernels.FAP_ARG_COUNT];
        a[AttnKernels.ARG_FA_N_TOKENS] = nTokens;
        a[AttnKernels.ARG_FA_N_HEAD] = nHead;
        a[AttnKernels.ARG_FA_N_HEAD_KV] = nHeadKv;
        a[AttnKernels.ARG_FA_D_K] = d;
        a[AttnKernels.ARG_FA_D_V] = d;
        a[AttnKernels.ARG_FA_BQ] = plan.bq();
        a[AttnKernels.ARG_FA_BK] = plan.bk();
        a[AttnKernels.ARG_FA_G_TILE] = plan.gTile();
        a[AttnKernels.ARG_FA_N_SPLIT] = nSplit;
        a[AttnKernels.ARG_FA_SPLIT_LEN] = splitLen;
        a[AttnKernels.ARG_FA_KV_START] = 0;
        a[AttnKernels.ARG_FA_KV_END] = nKv;
        a[AttnKernels.ARG_FA_Q_ROW_STRIDE] = nHead * d;
        a[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = nHead * d;
        a[AttnKernels.ARG_FA_K_ROW_STRIDE] = nHeadKv * d;
        a[AttnKernels.ARG_FA_K_HEAD_STRIDE] = d;
        a[AttnKernels.ARG_FA_V_ROW_STRIDE] = nHeadKv * d;
        a[AttnKernels.ARG_FA_V_HEAD_STRIDE] = d;
        a[AttnKernels.ARG_FA_SCALE] = p.scale();
        a[AttnKernels.ARG_FA_SOFTCAP] = p.softcap();
        a[AttnKernels.ARG_FA_MAX_BIAS] = p.maxBias();
        a[AttnKernels.ARG_FA_FLAGS] = AttnKernels.attnFlags(w.causal(), p.hasSinks(), direct);
        a[AttnKernels.ARG_FA_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FA_N_SWA] = w.windowed() ? w.nSwa() : 0;
        a[AttnKernels.ARG_FA_BIAS_MODE] = p.biasMode().code();
        a[AttnKernels.ARG_FA_N_BUCKETS] = p.nBuckets();
        a[AttnKernels.ARG_FA_MAX_DISTANCE] = p.maxDistance();
        a[AttnKernels.ARG_FA_Q_OFF] = 0;
        a[AttnKernels.ARG_FA_OUT_OFF] = 0;
        a[AttnKernels.ARG_FAP_PAGE_SHIFT] = PAGE_SHIFT;
        a[AttnKernels.ARG_FAP_PT_OFF] = PT_OFF;
        a[AttnKernels.ARG_FAP_BUF_Q] = q;
        a[AttnKernels.ARG_FAP_BUF_K] = k;
        a[AttnKernels.ARG_FAP_BUF_V] = v;
        a[AttnKernels.ARG_FAP_BUF_Q_POS] = qPos;
        a[AttnKernels.ARG_FAP_BUF_KV_POS] = kvPos;
        a[AttnKernels.ARG_FAP_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FAP_BUF_REL_BIAS] = relBias;
        a[AttnKernels.ARG_FAP_BUF_OUT] = out;
        a[AttnKernels.ARG_FAP_BUF_PARTIAL] = partial;
        a[AttnKernels.ARG_FAP_BUF_PAGE_TABLE] = pageTable;
        a[AttnKernels.ARG_FAP_BUF_KV_SCALES] = kvScales;
        a[AttnKernels.ARG_FAP_BUF_TREE_MASK] = treeMask;
        return AttnKernels.pagedWithKvTypes(a, typeK, typeV);
    }

    private static float[] ones(int n) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, 1f);
        return a;
    }

    /**
     * {@link #args} for a tree verify step: the flags gain the tree bit with the fresh-row count
     * {@code nFresh} packed in bits 5-31 ({@link AttnKernels#treeFlags}) over the bound ancestor bitmask.
     */
    private static Object[] treeArgs(FlashAttnTiling plan, int nTokens, int nHead, int nHeadKv, int d, int nKv,
                                     int nSplit, int splitLen, CpuAttentionCore.Params p, boolean direct,
                                     GpuBuffer q, GpuBuffer k, GpuBuffer v, GpuBuffer qPos, GpuBuffer kvPos,
                                     GpuBuffer sinks, GpuBuffer relBias, GpuBuffer out, GpuBuffer partial,
                                     GpuBuffer pageTable, GpuBuffer kvScales, GpuBuffer treeMask, int nFresh) {
        Object[] a = args(KvDType.F32, KvDType.F32, plan, nTokens, nHead, nHeadKv, d, nKv, nSplit, splitLen, p,
                direct, q, k, v, qPos, kvPos, sinks, relBias, out, partial, pageTable, kvScales, treeMask);
        a[AttnKernels.ARG_FA_FLAGS] = AttnKernels.treeFlags((int) a[AttnKernels.ARG_FA_FLAGS], nFresh);
        return a;
    }

    /** {@code base} with the device buffer's sink logits when it uses sinks. */
    private static CpuAttentionCore.Params withSinks(CpuAttentionCore.Params base, float[] sinks) {
        CpuAttentionCore.Builder b = CpuAttentionCore.Params.builder(base.nHead())
                .scale(base.scale()).softcap(base.softcap()).window(base.window());
        if (base.hasSinks()) {
            b.sinks(sinks);
        }
        return b.build();
    }

    private static int[] shuffledPages(int n, Random rnd) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private static float round(KvDType t, float f) {
        return t == KvDType.F16 ? f32(f16(f)) : f;
    }

    private static void assertClose(String what, float[] want, float[] got, float tol) {
        double err = 0;
        for (int i = 0; i < got.length; i++) {
            assertTrue(what + " output " + i + " is not finite: " + got[i], Float.isFinite(got[i]));
            err = Math.max(err, Math.abs(got[i] - want[i]));
        }
        assertTrue(what + " outputs differ from the CPU reference by " + err, err <= tol);
    }

    private void dispatch(String kernel, long[] global, long[] local, Object[] args) {
        GpuEvent e = backend.computeQueue().enqueueKernel(kernels.get(kernel), global, local, args);
        if (e != null) {
            e.close();
        }
    }

    private GpuBuffer alloc(List<GpuBuffer> bufs, long bytes) {
        GpuBuffer b = backend.allocateBuffer(bytes, true);
        bufs.add(b);
        return b;
    }

    private GpuBuffer upload(List<GpuBuffer> bufs, float[] data) {
        GpuBuffer b = alloc(bufs, (long) data.length * 4);
        write(b, data);
        return b;
    }

    private GpuBuffer upload(List<GpuBuffer> bufs, KvDType type, float[] data) {
        if (type == KvDType.F32) {
            return upload(bufs, data);
        }
        byte[] bytes = new byte[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            short h = f16(data[i]);
            bytes[2 * i] = (byte) h;
            bytes[2 * i + 1] = (byte) (h >> 8);
        }
        GpuBuffer b = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
        return b;
    }

    private GpuBuffer upload(List<GpuBuffer> bufs, int[] data) {
        byte[] bytes = new byte[data.length * 4];
        for (int i = 0; i < data.length; i++) {
            bytes[4 * i] = (byte) data[i];
            bytes[4 * i + 1] = (byte) (data[i] >> 8);
            bytes[4 * i + 2] = (byte) (data[i] >> 16);
            bytes[4 * i + 3] = (byte) (data[i] >> 24);
        }
        GpuBuffer b = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
        return b;
    }

    /** The kernels' {@code ulong} words are little-endian, as the Java uploads are. */
    private GpuBuffer upload(List<GpuBuffer> bufs, long[] data) {
        byte[] bytes = new byte[data.length * 8];
        for (int i = 0; i < data.length; i++) {
            for (int b = 0; b < 8; b++) {
                bytes[8 * i + b] = (byte) (data[i] >>> (8 * b));
            }
        }
        GpuBuffer buf = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(buf, 0, bytes.length, bytes));
        return buf;
    }

    private void write(GpuBuffer b, float[] data) {
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data));
    }

    private float[] read(GpuBuffer b, int n) {
        float[] dst = new float[n];
        close(backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * 4, dst));
        backend.computeQueue().finish();
        return dst;
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    /** IEEE 754 binary16 rounding to nearest even (normal range; the test values stay in it). */
    private static short f16(float f) {
        int bits = Float.floatToIntBits(f);
        int sign = (bits >>> 16) & 0x8000;
        int mag = bits & 0x7fffffff;
        int exp = (mag >>> 23) - 127 + 15;
        if (exp <= 0) {
            return (short) sign;
        }
        int mant = (mag & 0x7fffff) >>> 13;
        int rest = mag & 0x1fff;
        if (rest > 0x1000 || (rest == 0x1000 && (mant & 1) == 1)) {
            mant++;
            if (mant == 0x400) {
                mant = 0;
                exp++;
            }
        }
        return (short) (sign | (exp << 10) | mant);
    }

    private static float f32(short h) {
        int sign = (h & 0x8000) << 16;
        int exp = (h >>> 10) & 0x1f;
        int mant = h & 0x3ff;
        if (exp == 0) {
            return Float.intBitsToFloat(sign);
        }
        return Float.intBitsToFloat(sign | ((exp - 15 + 127) << 23) | (mant << 13));
    }
}
