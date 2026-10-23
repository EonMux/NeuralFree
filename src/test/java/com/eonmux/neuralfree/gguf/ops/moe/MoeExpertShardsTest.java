package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Expert-shard plumbing of the MoE blocks: the {@code expert_base, expert_count} arguments of the three id-matvec
 * kernels (after {@code has_scale}), the partition comparison that decides whether separate gate and up shards can
 * share one {@code {T}_MatVecIdGateUp} dispatch, and the CPU expert product over a host tensor whose row shards split
 * an expert.
 */
public class MoeExpertShardsTest {

    private FakeComputeBackend backend;

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    private GpuBuffer buffer() {
        if (backend == null) {
            backend = FakeComputeBackend.builder().build();
        }
        return backend.allocateBuffer(64, true);
    }

    // ------------------------------------------------------------------ kernel arguments

    @Test
    public void matVecIdArgsCarryTheExpertRangeAfterHasScale() {
        GpuBuffer slots = buffer();
        GpuBuffer weight = buffer();
        GpuBuffer input = buffer();
        GpuBuffer scales = buffer();
        GpuBuffer output = buffer();
        Object[] a = MoeKernels.matVecIdArgs(32, 64, 2, 0, 0, 0, true, 192, 192, slots, weight, input, scales,
                output);
        assertEquals(MoeKernels.ID_ARG_COUNT, a.length);
        assertEquals(MoeKernels.SCALE_SHARED, a[MoeKernels.ARG_ID_HAS_SCALE]);
        assertEquals(192, a[MoeKernels.ARG_ID_EXPERT_BASE]);
        assertEquals(192, a[MoeKernels.ARG_ID_EXPERT_COUNT]);
        assertSame(slots, a[MoeKernels.ARG_ID_SLOTS]);
        assertSame(weight, a[MoeKernels.ARG_ID_WEIGHT]);
        assertSame(input, a[MoeKernels.ARG_ID_INPUT]);
        assertSame(scales, a[MoeKernels.ARG_ID_SCALES]);
        assertSame(output, a[MoeKernels.ARG_ID_OUTPUT]);
        assertEquals(MoeKernels.ARG_ID_HAS_SCALE + 1, MoeKernels.ARG_ID_EXPERT_BASE);
    }

    @Test
    public void gateUpAndBatchedArgsCarryTheExpertRangeAfterHasScale() {
        GpuBuffer b = buffer();
        Object[] gu = MoeKernels.gateUpArgs(32, 64, 2, 0, 0, GluOp.SWIGLU, ActParams.NONE, true,
                MoeKernels.SCALE_SHARED, 5, 3, b, b, b, b, b, b);
        assertEquals(MoeKernels.GU_ARG_COUNT, gu.length);
        assertEquals(MoeKernels.ARG_GU_HAS_SCALE + 1, MoeKernels.ARG_GU_EXPERT_BASE);
        assertEquals(5, gu[MoeKernels.ARG_GU_EXPERT_BASE]);
        assertEquals(3, gu[MoeKernels.ARG_GU_EXPERT_COUNT]);
        assertEquals(1, gu[MoeKernels.ARG_GU_FUSED]);
        assertEquals(0, gu[MoeKernels.ARG_GU_IN_OFF]);

        Object[] bmm = MoeKernels.batchMatMulIdArgs(32, 64, 10, 2, 0, false, 0, 8, b, b, b, b, b, b);
        assertEquals(MoeKernels.BMM_ARG_COUNT, bmm.length);
        assertEquals(MoeKernels.ARG_BMM_HAS_SCALE + 1, MoeKernels.ARG_BMM_EXPERT_BASE);
        assertEquals(0, bmm[MoeKernels.ARG_BMM_EXPERT_BASE]);
        assertEquals(8, bmm[MoeKernels.ARG_BMM_EXPERT_COUNT]);
        assertEquals(10, bmm[MoeKernels.ARG_BMM_N_ROWS]);
    }

    @Test
    public void emptyOrNegativeExpertRangesAreRejected() {
        GpuBuffer b = buffer();
        assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.matVecIdArgs(32, 64, 2, 0, 0, 0, false, 0, 0, b, b, b, b, b));
        assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.matVecIdArgs(32, 64, 2, 0, 0, 0, false, -1, 4, b, b, b, b, b));
        assertThrows(IllegalArgumentException.class, () -> MoeKernels.gateUpArgs(32, 64, 2, 0, 0, GluOp.SWIGLU,
                ActParams.NONE, false, MoeKernels.SCALE_NONE, 0, 0, b, b, b, b, b, b));
        assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.batchMatMulIdArgs(32, 64, 10, 2, 0, false, -3, 8, b, b, b, b, b, b));
    }

    // ------------------------------------------------------------------ partitions

    @Test
    public void partitionsCompareExpertRangesNotBuffers() {
        List<MoeExpertWeights.Shard> gate = Arrays.asList(new MoeExpertWeights.Shard(buffer(), 0, 192),
                new MoeExpertWeights.Shard(buffer(), 192, 192));
        List<MoeExpertWeights.Shard> up = Arrays.asList(new MoeExpertWeights.Shard(buffer(), 0, 192),
                new MoeExpertWeights.Shard(buffer(), 192, 192));
        List<MoeExpertWeights.Shard> whole = Collections.singletonList(new MoeExpertWeights.Shard(buffer(), 0, 384));
        List<MoeExpertWeights.Shard> skewed = Arrays.asList(new MoeExpertWeights.Shard(buffer(), 0, 200),
                new MoeExpertWeights.Shard(buffer(), 200, 184));
        assertTrue(MoeExpertWeights.samePartition(gate, up));
        assertFalse(MoeExpertWeights.samePartition(gate, whole));
        assertFalse(MoeExpertWeights.samePartition(gate, skewed));
        assertThrows(IllegalArgumentException.class, () -> new MoeExpertWeights.Shard(buffer(), 0, 0));
    }

    // ------------------------------------------------------------------ CPU expert products over row shards

    /**
     * Four experts of three rows each, cut into host shards at rows 4 and 9: expert 1 spans shards 0 and 1, expert 3
     * starts shard 2. Every expert product equals the one over the unsharded tensor and the direct dot products.
     */
    @Test
    public void cpuExpertProductsReadExpertsAcrossRowShards() {
        int cols = 4;
        int rowsPerExpert = 3;
        int nExpert = 4;
        int rows = rowsPerExpert * nExpert;
        float[] w = new float[rows * cols];
        for (int i = 0; i < w.length; i++) {
            w[i] = (i % 7) - 3 + 0.25f * (i % 3);
        }
        float[] x = {0.5f, -1.0f, 2.0f, 0.75f};
        long rowBytes = (long) cols * Float.BYTES;
        CpuTensorRef whole = CpuTensorRef.of(GGMLType.F32, f32(w, 0, rows * cols), rows, cols);
        int[] cuts = {0, 4, 9, rows};
        ByteBuffer[] shards = new ByteBuffer[cuts.length - 1];
        long[] offsets = new long[cuts.length - 1];
        for (int s = 0; s < shards.length; s++) {
            shards[s] = f32(w, cuts[s] * cols, cuts[s + 1] * cols);
            offsets[s] = cuts[s] * rowBytes;
        }
        CpuTensorRef sharded = new CpuTensorRef(GGMLType.F32, shards, offsets, rows, cols, rowBytes);
        assertEquals(3, sharded.shardCount());

        for (int e = 0; e < nExpert; e++) {
            float[] expected = new float[rowsPerExpert];
            for (int r = 0; r < rowsPerExpert; r++) {
                double acc = 0;
                for (int c = 0; c < cols; c++) {
                    acc += (double) w[((e * rowsPerExpert) + r) * cols + c] * x[c];
                }
                expected[r] = (float) acc;
            }
            float[] a = new float[rowsPerExpert];
            float[] b = new float[rowsPerExpert];
            CpuMatVec.PLAIN.matvecExpert(whole, e, nExpert, x, 0, a, 0);
            CpuMatVec.PLAIN.matvecExpert(sharded, e, nExpert, x, 0, b, 0);
            assertArrayEquals("expert " + e, a, b, 0f);
            assertArrayEquals("expert " + e, expected, b, 1e-5f);
        }
    }

    private static ByteBuffer f32(float[] v, int from, int to) {
        ByteBuffer b = ByteBuffer.allocate((to - from) * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = from; i < to; i++) {
            b.putFloat(v[i]);
        }
        b.flip();
        return b;
    }
}
