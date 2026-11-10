package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.forward.SubgroupFormBackends;
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
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The subgroup flash-attention kernels ({@link SubgroupFlashAttention}: {@code LLM_FlashDecodeSg_<KV>},
 * {@code LLM_FlashDecodePagedSg_<KV>}, {@code LLM_FlashAttnSg_<KV>} and {@code LLM_FlashAttnPagedSg_<KV>}) against
 * {@link CpuAttentionCore} on every device whose kernels include them ({@link SubgroupFormBackends}: the OpenCL kernels
 * of kernels/llm/flash_*_sg.cl and their Vulkan ports shaders/llm/flash_*_sg.comp, the Intel GPUs; other devices
 * skip): F32,
 * F16 and BF16 caches at head dims 64 (half the subgroup's lanes idle), 80 and 128 (every lane), GQA groups of 1 to 4
 * (3 leaves a partial head tile), empty cache cells, sinks, softcap, the sliding window and bidirectional attention.
 * Each decode case runs the plan's split of the step (split-K merged by {@code LLM_FlashAttn_Combine}, or direct) and
 * a direct launch whose one split scans every key; each prefill case a split-K and a direct launch; the paged cases
 * run over shuffled pages with a masked page (three decode sequences of different lengths in one launch, and a
 * prefill segment with and without a verify step's tree mask).
 */
@RunWith(Parameterized.class)
public class SubgroupFlashAttentionGpuTest {

    private static final int PAGE_SHIFT = 4;
    private static final int PAGE = 1 << PAGE_SHIFT;
    private static final float TOL = 1e-4f;
    /**
     * The fewest compute units the decode plans of this test assume: more than any tested device reports (the Arrow
     * Lake iGPU reports 64 through OpenCL, none through Mesa ANV) or {@code FlashAttnTiling.planningUnits} assumes, so
     * the decode runs split and merged on every device.
     */
    private static final int PLAN_COMPUTE_UNITS = 128;

    @Parameterized.Parameters(name = "{0} {1}")
    public static List<Object[]> types() {
        List<Object[]> rows = new ArrayList<>();
        for (BackendType api : new BackendType[]{BackendType.OPENCL, BackendType.VULKAN}) {
            for (KvDType t : new KvDType[]{KvDType.F32, KvDType.F16, KvDType.BF16}) {
                rows.add(new Object[]{api, t});
            }
        }
        return rows;
    }

    private final BackendType api;
    private final KvDType type;
    private SubgroupFormBackends devices;
    private List<ComputeBackend> backends;

    public SubgroupFlashAttentionGpuTest(BackendType api, KvDType type) {
        this.api = api;
        this.type = type;
    }

    @Before
    public void open() {
        devices = SubgroupFormBackends.open(api);
        backends = devices.backends();
    }

    @After
    public void close() {
        if (devices != null) {
            devices.close();
        }
    }

    @Test
    public void denseDecodeMatchesTheCpuReference() {
        int served = 0;
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null) {
                continue;
            }
            served++;
            String dev = backend.capabilities().getDeviceName();
            CpuAttentionCore.Params causal = CpuAttentionCore.Params.builder(8).scale(0.125f)
                    .window(KvWindow.FULL_CAUSAL).build();
            for (int d : new int[]{64, 80, 128}) {
                runDense(backend, kernels, dev + " causal d=" + d, 8, 2, d, causal, false);
            }
            runDense(backend, kernels, dev + " sinks+softcap", 8, 4, 128, CpuAttentionCore.Params.builder(8)
                    .scale(0.09f).softcap(20f).window(KvWindow.FULL_CAUSAL).sinks(new float[8]).build(), false);
            runDense(backend, kernels, dev + " swa", 8, 2, 64, CpuAttentionCore.Params.builder(8).scale(0.125f)
                    .window(KvWindow.of(SwaType.STANDARD, 70, true)).build(), false);
            runDense(backend, kernels, dev + " empty cells", 8, 8, 128, causal, true);
        }
        Assume.assumeTrue("no " + api + " device builds the subgroup decode kernels", served > 0);
    }

    @Test
    public void pagedDecodeMatchesTheCpuReference() {
        int served = 0;
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null) {
                continue;
            }
            served++;
            runPaged(backend, kernels, backend.capabilities().getDeviceName(), 8, 2, 128, new int[]{37, 300, 900});
        }
        Assume.assumeTrue("no " + api + " device builds the subgroup decode kernels", served > 0);
    }

    @Test
    public void densePrefillMatchesTheCpuReference() {
        int served = 0;
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null) {
                continue;
            }
            served++;
            String dev = backend.capabilities().getDeviceName();
            runPrefill(backend, kernels, dev + " causal gqa2", false, 8, 4, 128, 150, 37, causal(8), null);
            runPrefill(backend, kernels, dev + " sinks+softcap gqa4", false, 8, 2, 128, 130, 20,
                    CpuAttentionCore.Params.builder(8).scale(0.09f).softcap(20f).window(KvWindow.FULL_CAUSAL)
                            .sinks(new float[8]).build(), null);
            runPrefill(backend, kernels, dev + " swa d=64", false, 8, 2, 64, 150, 40,
                    CpuAttentionCore.Params.builder(8).scale(0.125f)
                            .window(KvWindow.of(SwaType.STANDARD, 33, true)).build(), null);
            runPrefill(backend, kernels, dev + " partial head tile gqa3 d=80", false, 6, 2, 80, 100, 9, causal(6),
                    null);
            runPrefill(backend, kernels, dev + " bidirectional mha", false, 4, 4, 128, 100, 100,
                    CpuAttentionCore.Params.builder(4).scale(0.125f).window(KvWindow.FULL_BIDIRECTIONAL).build(),
                    null);
        }
        Assume.assumeTrue("no " + api + " device builds the subgroup attention kernels", served > 0);
    }

    @Test
    public void pagedPrefillMatchesTheCpuReference() {
        int served = 0;
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null) {
                continue;
            }
            served++;
            String dev = backend.capabilities().getDeviceName();
            runPrefill(backend, kernels, dev + " paged causal", true, 8, 2, 128, 150, 12, causal(8), null);
            // fresh row 0 is the pending root, rows 1 and 2 its depth-1 children, row 3 row 1's child and row 4
            // that node's child; a row's mask names its ancestors and itself
            runPrefill(backend, kernels, dev + " paged tree", true, 8, 2, 64, 120, 5, causal(8),
                    new int[]{-1, 0, 0, 1, 3});
        }
        Assume.assumeTrue("no " + api + " device builds the subgroup attention kernels", served > 0);
    }

    /** The device's kernels when its program builds this type's subgroup kernels, else null. */
    private LlmKernels kernelsServing(ComputeBackend backend) {
        if (!LlmKernels.availability(backend).available()) {
            return null;
        }
        LlmKernels kernels = LlmKernels.open(backend);
        return SubgroupFlashAttention.serves(kernels, type, type, 128, 128) ? kernels : null;
    }

    /**
     * One query row at the last position of {@code n} keys, for every n of a direct step, a split-K step and a long
     * step: the plan's launch and a one-split direct launch, each against the reference. {@code holes} empties every
     * seventh cache cell ({@code kv_pos} -1).
     */
    private void runDense(ComputeBackend backend, LlmKernels kernels, String what, int nHead, int nHeadKv, int d,
                          CpuAttentionCore.Params base, boolean holes) {
        int nKvMax = 1500;
        Random rnd = new Random(31L * d + nHeadKv);
        int rowWidth = nHeadKv * d;
        float[] kF = randomCache(rnd, nKvMax * rowWidth);
        float[] vF = randomCache(rnd, nKvMax * rowWidth);
        int[] kvPos = new int[nKvMax];
        for (int i = 0; i < nKvMax; i++) {
            kvPos[i] = holes && i % 7 == 3 ? -1 : i;
        }
        float[] q = randomFloats(rnd, nHead * d);
        float[] sinks = randomFloats(rnd, nHead);
        CpuAttentionCore.Params params = withSinks(base, sinks);
        FlashAttnTiling plan = FlashAttnTiling.decodeSubgroup(nHead, nHeadKv, d, d, nKvMax, kernels.faSmemFloats(),
                planUnits(backend));
        assertTrue(plan.toString(), plan.subgroup() && plan.gTile() == 1);
        try (Bufs bufs = new Bufs(backend)) {
            GpuBuffer dq = bufs.upload(q);
            GpuBuffer dk = bufs.uploadCache(type, kF);
            GpuBuffer dv = bufs.uploadCache(type, vF);
            GpuBuffer dpos = bufs.upload(kvPos);
            GpuBuffer dsinks = bufs.upload(sinks);
            GpuBuffer drel = bufs.upload(new float[1]);
            GpuBuffer dscales = bufs.upload(ones(2 * nHeadKv));
            GpuBuffer dmask = bufs.upload(new long[1]);
            GpuBuffer dout = bufs.alloc((long) nHead * d * 4);
            GpuBuffer dpartial = bufs.alloc(plan.decodePartialFloats() * 4);
            for (int n : new int[]{5, 200, nKvMax}) {
                float[] want = new float[nHead * d];
                CpuAttentionCore.attendRows(params, 1, q, 0, nHead * d, new int[]{n - 1}, 0,
                        CpuAttentionCore.DenseKv.of(kF, 0, rowWidth, d, vF, 0, rowWidth, d, nHeadKv, d, d, kvPos, 0),
                        0, n, want, 0, nHead * d);
                FlashAttnTiling.Split split = plan.decodeSplit(n);
                Object[] args = denseArgs(nHead, nHeadKv, d, split.count(), split.length(), n, params,
                        split.direct(), dq, dk, dv, dpos, dsinks, drel, dout, dpartial, dscales, dmask);
                bufs.write(dout, new float[want.length]);
                dispatch(backend, kernels, SubgroupFlashAttention.kernel(type), plan.decodeGlobal(split.count()),
                        plan.decodeLocalSize(), args);
                if (!split.direct()) {
                    combine(backend, kernels, nHead, d, split.count(), params, dpartial, dsinks, dout);
                }
                assertClose(what + " " + type + " n=" + n + " " + split, want, bufs.read(dout, want.length));

                int whole = (n + 31) / 32 * 32;
                Object[] direct = denseArgs(nHead, nHeadKv, d, 1, whole, n, params, true, dq, dk, dv, dpos,
                        dsinks, drel, dout, dpartial, dscales, dmask);
                bufs.write(dout, new float[want.length]);
                dispatch(backend, kernels, SubgroupFlashAttention.kernel(type), plan.decodeGlobal(1),
                        plan.decodeLocalSize(), direct);
                assertClose(what + " " + type + " n=" + n + " one split", want, bufs.read(dout, want.length));
            }
        }
    }

    /**
     * {@code kvLens.length} sequences in one paged launch, each over its own shuffled pages (page 1 of every
     * sequence masked) and querying at its last slot; split by the plan over the longest sequence and merged by
     * the combine over {@code n_seq * n_head} rows.
     */
    private void runPaged(ComputeBackend backend, LlmKernels kernels, String what, int nHead, int nHeadKv, int d,
                          int[] kvLens) {
        Random rnd = new Random(11);
        int nSeq = kvLens.length;
        int maxLen = Arrays.stream(kvLens).max().getAsInt();
        int maxPages = (maxLen + PAGE - 1) / PAGE;
        int physPages = nSeq * maxPages + 3;
        int rowWidth = nHeadKv * d;
        float[] kPhys = randomCache(rnd, physPages * PAGE * rowWidth);
        float[] vPhys = randomCache(rnd, physPages * PAGE * rowWidth);
        int[] kvPosPhys = new int[physPages * PAGE];
        Arrays.fill(kvPosPhys, -1);
        int[] phys = shuffled(physPages, rnd);
        int[] pageTable = new int[nSeq * maxPages];
        Arrays.fill(pageTable, -1);
        float[] q = randomFloats(rnd, nSeq * nHead * d);
        float[] sinks = randomFloats(rnd, nHead);
        int[] qPos = new int[nSeq];
        CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(nHead).scale(0.125f)
                .window(KvWindow.FULL_CAUSAL).build();
        float[] want = new float[nSeq * nHead * d];
        int next = 0;
        for (int s = 0; s < nSeq; s++) {
            int len = kvLens[s];
            int pages = (len + PAGE - 1) / PAGE;
            float[] kLog = new float[len * rowWidth];
            float[] vLog = new float[len * rowWidth];
            int[] kvPosLog = new int[len];
            for (int p = 0; p < pages; p++) {
                pageTable[s * maxPages + p] = p == 1 ? -1 : phys[next++];
            }
            for (int j = 0; j < len; j++) {
                int page = pageTable[s * maxPages + (j >> PAGE_SHIFT)];
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
            qPos[s] = len - 1;
            CpuAttentionCore.attendRows(params, 1, q, s * nHead * d, nHead * d, new int[]{qPos[s]}, 0,
                    CpuAttentionCore.DenseKv.of(kLog, 0, rowWidth, d, vLog, 0, rowWidth, d, nHeadKv, d, d,
                            kvPosLog, 0),
                    0, len, want, s * nHead * d, nHead * d);
        }
        FlashAttnTiling plan = FlashAttnTiling.decodePagedSubgroup(nSeq, nHead, nHeadKv, d, d,
                maxPages << PAGE_SHIFT, kernels.faSmemFloats(), planUnits(backend));
        FlashAttnTiling.Split split = plan.decodeSplit(maxLen);
        assertTrue("the paged launch needs several splits: " + split, !split.direct());
        try (Bufs bufs = new Bufs(backend)) {
            GpuBuffer dout = bufs.alloc((long) want.length * 4);
            Object[] a = new Object[AttnKernels.FDP_ARG_COUNT];
            Object[] dense = denseArgs(nHead, nHeadKv, d, split.count(), split.length(), 0, params, false,
                    bufs.upload(q), bufs.uploadCache(type, kPhys), bufs.uploadCache(type, vPhys),
                    bufs.upload(kvPosPhys), bufs.upload(sinks), bufs.upload(new float[1]), dout,
                    bufs.alloc(plan.decodePagedPartialFloats(nSeq) * 4), bufs.upload(ones(2 * nHeadKv)),
                    bufs.upload(new long[nSeq]));
            System.arraycopy(dense, 0, a, 0, AttnKernels.ARG_FDP_PAGE_SHIFT);
            a[AttnKernels.ARG_FDP_PAGE_SHIFT] = PAGE_SHIFT;
            a[AttnKernels.ARG_FDP_MAX_PAGES] = maxPages;
            a[AttnKernels.ARG_FDP_N_SEQ] = nSeq;
            a[AttnKernels.ARG_FDP_Q_SEQ_STRIDE] = nHead * d;
            a[AttnKernels.ARG_FDP_OUT_SEQ_STRIDE] = nHead * d;
            a[AttnKernels.ARG_FDP_BUF_Q] = dense[AttnKernels.ARG_FD_BUF_Q];
            a[AttnKernels.ARG_FDP_BUF_K] = dense[AttnKernels.ARG_FD_BUF_K];
            a[AttnKernels.ARG_FDP_BUF_V] = dense[AttnKernels.ARG_FD_BUF_V];
            a[AttnKernels.ARG_FDP_BUF_KV_POS] = dense[AttnKernels.ARG_FD_BUF_KV_POS];
            a[AttnKernels.ARG_FDP_BUF_SINKS] = dense[AttnKernels.ARG_FD_BUF_SINKS];
            a[AttnKernels.ARG_FDP_BUF_REL_BIAS] = dense[AttnKernels.ARG_FD_BUF_REL_BIAS];
            a[AttnKernels.ARG_FDP_BUF_OUT] = dout;
            GpuBuffer partial = (GpuBuffer) dense[AttnKernels.ARG_FD_BUF_PARTIAL];
            a[AttnKernels.ARG_FDP_BUF_PARTIAL] = partial;
            a[AttnKernels.ARG_FDP_BUF_PAGE_TABLE] = bufs.upload(pageTable);
            a[AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN] = bufs.upload(kvLens);
            a[AttnKernels.ARG_FDP_BUF_Q_POS_SEQ] = bufs.upload(qPos);
            a[AttnKernels.ARG_FDP_BUF_KV_SCALES] = dense[AttnKernels.ARG_FD_BUF_KV_SCALES];
            a[AttnKernels.ARG_FDP_BUF_TREE_MASK] = dense[AttnKernels.ARG_FD_BUF_TREE_MASK];
            dispatch(backend, kernels, SubgroupFlashAttention.pagedKernel(type), plan.decodePagedGlobal(nSeq,
                    split.count()), plan.decodeLocalSize(), a);
            combine(backend, kernels, nSeq * nHead, d, split.count(), params, partial,
                    (GpuBuffer) dense[AttnKernels.ARG_FD_BUF_SINKS], dout);
            assertClose(what + " " + type + " paged " + Arrays.toString(kvLens) + " " + split, want,
                    bufs.read(dout, want.length));
        }
    }

    private static CpuAttentionCore.Params causal(int nHead) {
        return CpuAttentionCore.Params.builder(nHead).scale(0.125f).window(KvWindow.FULL_CAUSAL).build();
    }

    /**
     * {@code nTokens} query rows at the last positions of {@code nKv} keys ({@code parent} null), or a verify step's
     * {@code parent.length} fresh rows at the end of the range under their tree mask, through the dense or the paged
     * prefill kernel (16-slot pages, shuffled, page 1 masked, the segment's table at offset 3): split-K over 48-key
     * splits merged by the combine, and direct over one split, each against the reference.
     */
    private void runPrefill(ComputeBackend backend, LlmKernels kernels, String what, boolean paged, int nHead,
                            int nHeadKv, int d, int nKv, int nTokens, CpuAttentionCore.Params base, int[] parent) {
        Random rnd = new Random(31L * nKv + nTokens);
        int rowWidth = nHeadKv * d;
        int ptOff = paged ? 3 : 0;
        int nPages = (nKv + PAGE - 1) / PAGE;
        int physRows = paged ? (nPages + 2) * PAGE : nKv;
        int[] pageTable = new int[ptOff + nPages];
        Arrays.fill(pageTable, -7);
        int[] phys = shuffled(nPages + 2, rnd);
        for (int p = 0; p < nPages; p++) {
            pageTable[ptOff + p] = p == 1 ? -1 : phys[p];
        }
        float[] kPhys = randomCache(rnd, physRows * rowWidth);
        float[] vPhys = randomCache(rnd, physRows * rowWidth);
        int[] kvPosPhys = new int[physRows];
        Arrays.fill(kvPosPhys, -1);
        float[] kLog = new float[nKv * rowWidth];
        float[] vLog = new float[nKv * rowWidth];
        int[] kvPosLog = new int[nKv];
        for (int j = 0; j < nKv; j++) {
            int row = j;
            if (paged) {
                int page = pageTable[ptOff + (j >> PAGE_SHIFT)];
                if (page < 0) {
                    kvPosLog[j] = -1;
                    continue;
                }
                row = (page << PAGE_SHIFT) | (j & (PAGE - 1));
            }
            kvPosPhys[row] = j;
            kvPosLog[j] = j;
            System.arraycopy(kPhys, row * rowWidth, kLog, j * rowWidth, rowWidth);
            System.arraycopy(vPhys, row * rowWidth, vLog, j * rowWidth, rowWidth);
        }
        int[] qPos = new int[nTokens];
        long[] mask = new long[nTokens];
        int nFresh = parent == null ? 0 : parent.length;
        for (int t = 0; t < nTokens; t++) {
            qPos[t] = nKv - nTokens + t;
        }
        if (parent != null) {
            int[] depth = new int[nFresh];
            for (int t = 0; t < nFresh; t++) {
                mask[t] = 1L << t;
                depth[t] = parent[t] < 0 ? 0 : depth[parent[t]] + 1;
                qPos[t] = nKv - nFresh + depth[t];
                for (int a = parent[t]; a >= 0; a = parent[a]) {
                    mask[t] |= 1L << a;
                }
            }
        }
        float[] q = randomFloats(rnd, nTokens * nHead * d);
        float[] sinks = randomFloats(rnd, nHead);
        CpuAttentionCore.Params params = withSinks(base, sinks);
        float[] want = new float[nTokens * nHead * d];
        CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(kLog, 0, rowWidth, d, vLog, 0, rowWidth, d,
                nHeadKv, d, d, kvPosLog, 0);
        if (parent == null) {
            CpuAttentionCore.attendRows(params, nTokens, q, 0, nHead * d, qPos, 0, kv, 0, nKv, want, 0, nHead * d);
        } else {
            CpuAttentionCore.attendRows(params, nTokens, q, 0, nHead * d, qPos, 0, kv, 0, nKv, want, 0, nHead * d,
                    new CpuAttentionCore.TreeMask(mask, nFresh));
        }

        FlashAttnTiling plan = FlashAttnTiling.prefillSubgroup(nHead, nHeadKv, d, d, nKv, kernels.faSmemFloats(),
                planUnits(backend));
        assertTrue(plan.toString(), plan.subgroup() && plan.rows() == SubgroupFlashAttention.PREFILL_ROWS);
        int splitLen = 48;
        int nSplit = (nKv + splitLen - 1) / splitLen;
        String kernel = paged ? SubgroupFlashAttention.pagedPrefillKernel(type)
                : SubgroupFlashAttention.prefillKernel(type);
        try (Bufs bufs = new Bufs(backend)) {
            GpuBuffer[] b = {bufs.upload(q), bufs.uploadCache(type, kPhys), bufs.uploadCache(type, vPhys),
                    bufs.upload(qPos), bufs.upload(kvPosPhys), bufs.upload(sinks), bufs.upload(new float[1]),
                    bufs.alloc((long) want.length * 4), bufs.alloc((long) want.length / d * nSplit * (d + 2) * 4),
                    bufs.upload(pageTable), bufs.upload(ones(2 * nHeadKv)), bufs.upload(mask)};
            dispatch(backend, kernels, kernel, plan.prefillGlobal(nTokens, nSplit), plan.prefillLocalSize(),
                    prefillArgs(paged, plan, nTokens, nHead, nHeadKv, d, nKv, nSplit, splitLen, params, false,
                            nFresh, ptOff, b));
            combine(backend, kernels, nTokens * nHead, d, nSplit, params, b[8], b[5], b[7]);
            assertClose(what + " " + type + " split-K", want, bufs.read(b[7], want.length));

            bufs.write(b[7], new float[want.length]);
            dispatch(backend, kernels, kernel, plan.prefillGlobal(nTokens, 1), plan.prefillLocalSize(),
                    prefillArgs(paged, plan, nTokens, nHead, nHeadKv, d, nKv, 1, nSplit * splitLen, params, true,
                            nFresh, ptOff, b));
            assertClose(what + " " + type + " direct", want, bufs.read(b[7], want.length));
        }
    }

    /**
     * The tiled prefill argument list ({@code LLM_FlashAttn_<KV>}, or {@code LLM_FlashAttnPaged_<KV>} with the page
     * geometry): buffers {@code b} are q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, page table, kv_scales
     * and the tree mask; {@code nFresh > 0} sets the tree flag over the step's last {@code nFresh} slots.
     */
    private static Object[] prefillArgs(boolean paged, FlashAttnTiling plan, int nTokens, int nHead, int nHeadKv,
                                        int d, int nKv, int nSplit, int splitLen, CpuAttentionCore.Params p,
                                        boolean direct, int nFresh, int ptOff, GpuBuffer[] b) {
        KvWindow w = p.window();
        Object[] a = new Object[paged ? AttnKernels.FAP_ARG_COUNT : AttnKernels.FA_ARG_COUNT];
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
        int flags = AttnKernels.attnFlags(w.causal(), p.hasSinks(), direct);
        a[AttnKernels.ARG_FA_FLAGS] = nFresh > 0 ? AttnKernels.treeFlags(flags, nFresh) : flags;
        a[AttnKernels.ARG_FA_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FA_N_SWA] = w.windowed() ? w.nSwa() : 0;
        a[AttnKernels.ARG_FA_BIAS_MODE] = p.biasMode().code();
        a[AttnKernels.ARG_FA_N_BUCKETS] = p.nBuckets();
        a[AttnKernels.ARG_FA_MAX_DISTANCE] = p.maxDistance();
        a[AttnKernels.ARG_FA_Q_OFF] = 0;
        a[AttnKernels.ARG_FA_OUT_OFF] = 0;
        if (paged) {
            a[AttnKernels.ARG_FAP_PAGE_SHIFT] = PAGE_SHIFT;
            a[AttnKernels.ARG_FAP_PT_OFF] = ptOff;
            a[AttnKernels.ARG_FAP_BUF_Q] = b[0];
            a[AttnKernels.ARG_FAP_BUF_K] = b[1];
            a[AttnKernels.ARG_FAP_BUF_V] = b[2];
            a[AttnKernels.ARG_FAP_BUF_Q_POS] = b[3];
            a[AttnKernels.ARG_FAP_BUF_KV_POS] = b[4];
            a[AttnKernels.ARG_FAP_BUF_SINKS] = b[5];
            a[AttnKernels.ARG_FAP_BUF_REL_BIAS] = b[6];
            a[AttnKernels.ARG_FAP_BUF_OUT] = b[7];
            a[AttnKernels.ARG_FAP_BUF_PARTIAL] = b[8];
            a[AttnKernels.ARG_FAP_BUF_PAGE_TABLE] = b[9];
            a[AttnKernels.ARG_FAP_BUF_KV_SCALES] = b[10];
            a[AttnKernels.ARG_FAP_BUF_TREE_MASK] = b[11];
        } else {
            a[AttnKernels.ARG_FA_BUF_Q] = b[0];
            a[AttnKernels.ARG_FA_BUF_K] = b[1];
            a[AttnKernels.ARG_FA_BUF_V] = b[2];
            a[AttnKernels.ARG_FA_BUF_Q_POS] = b[3];
            a[AttnKernels.ARG_FA_BUF_KV_POS] = b[4];
            a[AttnKernels.ARG_FA_BUF_SINKS] = b[5];
            a[AttnKernels.ARG_FA_BUF_REL_BIAS] = b[6];
            a[AttnKernels.ARG_FA_BUF_OUT] = b[7];
            a[AttnKernels.ARG_FA_BUF_PARTIAL] = b[8];
            a[AttnKernels.ARG_FA_BUF_KV_SCALES] = b[10];
            a[AttnKernels.ARG_FA_BUF_TREE_MASK] = b[11];
        }
        return a;
    }

    private static Object[] denseArgs(int nHead, int nHeadKv, int d, int nSplit, int splitLen, int n,
                                      CpuAttentionCore.Params p, boolean direct, GpuBuffer q, GpuBuffer k,
                                      GpuBuffer v, GpuBuffer kvPos, GpuBuffer sinks, GpuBuffer rel, GpuBuffer out,
                                      GpuBuffer partial, GpuBuffer scales, GpuBuffer mask) {
        KvWindow w = p.window();
        Object[] a = new Object[AttnKernels.FD_ARG_COUNT];
        a[AttnKernels.ARG_FD_N_HEAD] = nHead;
        a[AttnKernels.ARG_FD_N_HEAD_KV] = nHeadKv;
        a[AttnKernels.ARG_FD_D_K] = d;
        a[AttnKernels.ARG_FD_D_V] = d;
        a[AttnKernels.ARG_FD_G_TILE] = 1;
        a[AttnKernels.ARG_FD_N_SPLIT] = nSplit;
        a[AttnKernels.ARG_FD_SPLIT_LEN] = splitLen;
        a[AttnKernels.ARG_FD_KV_START] = 0;
        a[AttnKernels.ARG_FD_KV_END] = n;
        a[AttnKernels.ARG_FD_Q_POS] = n - 1;
        a[AttnKernels.ARG_FD_K_ROW_STRIDE] = nHeadKv * d;
        a[AttnKernels.ARG_FD_K_HEAD_STRIDE] = d;
        a[AttnKernels.ARG_FD_V_ROW_STRIDE] = nHeadKv * d;
        a[AttnKernels.ARG_FD_V_HEAD_STRIDE] = d;
        a[AttnKernels.ARG_FD_SCALE] = p.scale();
        a[AttnKernels.ARG_FD_SOFTCAP] = p.softcap();
        a[AttnKernels.ARG_FD_MAX_BIAS] = p.maxBias();
        a[AttnKernels.ARG_FD_FLAGS] = AttnKernels.attnFlags(w.causal(), p.hasSinks(), direct);
        a[AttnKernels.ARG_FD_SWA_TYPE] = w.type().code();
        a[AttnKernels.ARG_FD_N_SWA] = w.windowed() ? w.nSwa() : 0;
        a[AttnKernels.ARG_FD_BIAS_MODE] = p.biasMode().code();
        a[AttnKernels.ARG_FD_N_BUCKETS] = p.nBuckets();
        a[AttnKernels.ARG_FD_MAX_DISTANCE] = p.maxDistance();
        a[AttnKernels.ARG_FD_Q_OFF] = 0;
        a[AttnKernels.ARG_FD_OUT_OFF] = 0;
        a[AttnKernels.ARG_FD_BUF_Q] = q;
        a[AttnKernels.ARG_FD_BUF_K] = k;
        a[AttnKernels.ARG_FD_BUF_V] = v;
        a[AttnKernels.ARG_FD_BUF_KV_POS] = kvPos;
        a[AttnKernels.ARG_FD_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FD_BUF_REL_BIAS] = rel;
        a[AttnKernels.ARG_FD_BUF_OUT] = out;
        a[AttnKernels.ARG_FD_BUF_PARTIAL] = partial;
        a[AttnKernels.ARG_FD_BUF_KV_SCALES] = scales;
        a[AttnKernels.ARG_FD_BUF_TREE_MASK] = mask;
        return a;
    }

    private static void combine(ComputeBackend backend, LlmKernels kernels, int rows, int d, int nSplit,
                                CpuAttentionCore.Params p, GpuBuffer partial, GpuBuffer sinks, GpuBuffer out) {
        int nHead = p.nHead();
        dispatch(backend, kernels, AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(rows),
                FlashAttnTiling.combineLocal(), new Object[]{rows, d, nSplit, nHead, p.hasSinks() ? 1 : 0, 0,
                        partial, sinks, out});
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

    /** Cache values rounded to {@link #type} as the kernels decode them (F16 nearest-even, BF16 truncated). */
    private float[] randomCache(Random rnd, int n) {
        float[] v = randomFloats(rnd, n);
        for (int i = 0; i < n; i++) {
            v[i] = round(type, v[i]);
        }
        return v;
    }

    private static float round(KvDType t, float f) {
        if (t == KvDType.F16) {
            return f32(f16(f));
        }
        if (t == KvDType.BF16) {
            return Float.intBitsToFloat(Float.floatToIntBits(f) & 0xffff0000);
        }
        return f;
    }

    private static float[] randomFloats(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() - 0.5f;
        }
        return v;
    }

    private static float[] ones(int n) {
        float[] a = new float[n];
        Arrays.fill(a, 1f);
        return a;
    }

    private static int[] shuffled(int n, Random rnd) {
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

    private static void assertClose(String what, float[] want, float[] got) {
        double err = 0;
        for (int i = 0; i < got.length; i++) {
            assertTrue(what + " output " + i + " is not finite: " + got[i], Float.isFinite(got[i]));
            err = Math.max(err, Math.abs(got[i] - want[i]));
        }
        assertTrue(what + " outputs differ from the CPU reference by " + err, err <= TOL);
    }

    private static void dispatch(ComputeBackend backend, LlmKernels kernels, String kernel, long[] global,
                                 long[] local, Object[] args) {
        close(backend.computeQueue().enqueueKernel(kernels.get(kernel), global, local, args));
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    /** The buffers of one case, released together. */
    /** The compute-unit count the plans use: the device's, but at least {@link #PLAN_COMPUTE_UNITS}. */
    private static int planUnits(ComputeBackend backend) {
        return Math.max(PLAN_COMPUTE_UNITS, backend.capabilities().getComputeUnits());
    }

    private static final class Bufs implements AutoCloseable {
        private final ComputeBackend backend;
        private final List<GpuBuffer> all = new ArrayList<>();

        Bufs(ComputeBackend backend) {
            this.backend = backend;
        }

        GpuBuffer alloc(long bytes) {
            GpuBuffer b = backend.allocateBuffer(bytes, true);
            all.add(b);
            return b;
        }

        GpuBuffer upload(float[] data) {
            GpuBuffer b = alloc((long) data.length * 4);
            write(b, data);
            return b;
        }

        GpuBuffer upload(int[] data) {
            byte[] bytes = new byte[data.length * 4];
            for (int i = 0; i < data.length; i++) {
                for (int s = 0; s < 4; s++) {
                    bytes[4 * i + s] = (byte) (data[i] >> (8 * s));
                }
            }
            return upload(bytes);
        }

        GpuBuffer upload(long[] data) {
            byte[] bytes = new byte[data.length * 8];
            for (int i = 0; i < data.length; i++) {
                for (int s = 0; s < 8; s++) {
                    bytes[8 * i + s] = (byte) (data[i] >>> (8 * s));
                }
            }
            return upload(bytes);
        }

        /** A cache of {@code type}: F32 words, or little-endian F16 / BF16 halves of the already rounded values. */
        GpuBuffer uploadCache(KvDType type, float[] data) {
            if (type == KvDType.F32) {
                return upload(data);
            }
            byte[] bytes = new byte[data.length * 2];
            for (int i = 0; i < data.length; i++) {
                short h = type == KvDType.F16 ? f16(data[i]) : (short) (Float.floatToIntBits(data[i]) >>> 16);
                bytes[2 * i] = (byte) h;
                bytes[2 * i + 1] = (byte) (h >> 8);
            }
            return upload(bytes);
        }

        private GpuBuffer upload(byte[] bytes) {
            GpuBuffer b = alloc(bytes.length);
            SubgroupFlashAttentionGpuTest.close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
            return b;
        }

        void write(GpuBuffer b, float[] data) {
            SubgroupFlashAttentionGpuTest.close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data));
        }

        float[] read(GpuBuffer b, int n) {
            float[] dst = new float[n];
            SubgroupFlashAttentionGpuTest.close(backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * 4, dst));
            backend.computeQueue().finish();
            return dst;
        }

        @Override
        public void close() {
            backend.computeQueue().finish();
            for (GpuBuffer b : all) {
                b.close();
            }
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
