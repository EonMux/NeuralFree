package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link GpuGroupedRows} end to end on an OpenCL device: one instance sized past {@link MoeKernels#BUCKET_CHUNK}
 * pairs sorts a large step on the multi-work-group path and a small step on {@code LLM_MoE_Bucketize}, and both
 * write exactly the counts, offsets, sorted rows and aligned layout of {@link MoeCombine}. Skips without an OpenCL
 * GPU.
 */
public class GpuGroupedRowsOpenClTest {

    private static final int UBATCH = 1100;
    private static final int K = 2;
    private static final int N_EXPERT = 9;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private GroupedGemmHarness h;

    @Before
    public void setUp() {
        try {
            hwAccel = new HWAcceleration();
            if (!hwAccel.getCLDevices().isEmpty()) {
                backend = new OclComputeBackend(hwAccel.getCLDevices().get(0));
                LlmKernels kernels = LlmKernels.open(backend);
                scratch = GpuScratch.builder(backend, UBATCH).build();
                placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
                ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
                h = new GroupedGemmHarness(backend, kernels);
            }
        } catch (Throwable e) {
            backend = null;
        }
        Assume.assumeTrue("no OpenCL GPU", backend != null);
    }

    @After
    public void tearDown() {
        if (h != null) {
            h.close();
        }
        if (ops != null) {
            ops.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (scratch != null) {
            scratch.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            try {
                hwAccel.shutdown();
            } catch (Throwable ignored) {
                // nothing left to release
            }
        }
    }

    @Test
    public void largeAndSmallStepsMatchTheCpuSortAndAlignment() {
        int[] idValues = GroupedGemmHarness.ids(UBATCH * K, N_EXPERT, 7, true);
        GpuGroupedRows rows = new GpuGroupedRows(ops, scratch, h.kernels(), "test.rows", UBATCH, K, N_EXPERT,
                h.ints(idValues), true);
        try {
            assertTrue("multi-group sort prepared", rows.multiGroupSort());
            check(rows, 1050, idValues);
            check(rows, 5, idValues);
            check(rows, UBATCH, idValues);
        } finally {
            rows.close();
        }
    }

    private void check(GpuGroupedRows rows, int nTokens, int[] allIds) {
        rows.update(nTokens);
        rows.dispatch();
        int nRows = nTokens * K;
        int[] ids = Arrays.copyOf(allIds, nRows);
        MoeCombine.Buckets b = MoeCombine.bucketize(nTokens, K, N_EXPERT, ids);
        MoeCombine.Aligned a = MoeCombine.alignBlocks(nTokens, K, N_EXPERT, b.counts(), b.offsets(), b.rowToken(),
                b.rowSlot());
        String step = nTokens + " tokens: ";
        assertArrayEquals(step + "counts", b.counts(), h.readInts(rows.counts(), N_EXPERT));
        assertArrayEquals(step + "offsets", b.offsets(), h.readInts(rows.offsets(), N_EXPERT));
        assertArrayEquals(step + "row_token", b.rowToken(), h.readInts(rows.rowToken(), nRows));
        assertArrayEquals(step + "row_slot", b.rowSlot(), h.readInts(rows.rowSlot(), nRows));
        int positions = rows.nTiles() * MoeKernels.ALIGN_TILE;
        assertArrayEquals(step + "gemm_token", padded(a.gemmToken(), positions),
                h.readInts(rows.gemmToken(), positions));
        assertArrayEquals(step + "gemm_slot", padded(a.gemmSlot(), positions),
                h.readInts(rows.gemmSlot(), positions));
    }

    private static int[] padded(int[] used, int positions) {
        int[] out = Arrays.copyOf(used, positions);
        Arrays.fill(out, used.length, positions, -1);
        return out;
    }
}
