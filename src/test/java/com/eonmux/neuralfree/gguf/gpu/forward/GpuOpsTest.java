package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of {@link GpuOps} against the host-only fake backend with strict LLM argument rules: kernel choice
 * per token count, row shards scattered with {@code LLM_Copy2D_F32} at their row offsets, input staging, the PRISM
 * rotation, the tiled GEMM, accumulation, output scales, element-wise aliasing and prepared dispatches.
 */
public class GpuOpsTest {

    private static final int COLS = 256;
    private static final int ROWS = 96;
    private static final int SHARD_ROWS = 32;
    private static final long Q4K_ROW_BYTES = 144;
    private static final int UBATCH = 16;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int local;

    private void open(FakeComputeBackend fake) {
        open(fake, fake);
    }

    /** Opens the ops on {@code fake}, loading and preparing the kernels through {@code kernelBackend}. */
    private void open(FakeComputeBackend fake, ComputeBackend kernelBackend) {
        backend = fake;
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(kernelBackend), scratch, placeholders);
        local = ops.local();
    }

    /** Opens the ops on a fake whose {@code kernels} cannot run work-groups above {@code limit} work-items. */
    private void openWithKernelLimit(long limit, String... kernels) {
        FakeComputeBackend fake = FakeComputeBackend.builder().strictLlmArgs(true).build();
        open(fake, new KernelLimitBackend(fake, limit, kernels));
    }

    private void open() {
        open(FakeComputeBackend.builder().strictLlmArgs(true).build());
    }

    @After
    public void tearDown() {
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
    }

    // ------------------------------------------------------------------ fixtures

    private GpuWeight q4kSharded() {
        List<GpuBuffer> shards = new ArrayList<>();
        for (int i = 0; i < ROWS / SHARD_ROWS; i++) {
            shards.add(backend.allocateBuffer(SHARD_ROWS * Q4K_ROW_BYTES, true));
        }
        return GpuWeight.sharded("blk.0.ffn_up.weight", GGMLType.Q4_K, new long[]{COLS, ROWS}, shards,
                new long[]{SHARD_ROWS, SHARD_ROWS, SHARD_ROWS});
    }

    private GpuWeight q4kSingle() {
        return GpuWeight.of("blk.0.attn_q.weight", GGMLType.Q4_K, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * Q4K_ROW_BYTES, true));
    }

    private GpuBuffer floats(long n) {
        return backend.allocateBuffer(n * Float.BYTES);
    }

    private List<Dispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    private static List<String> names(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static int[] ints(Dispatch d, int from, int to) {
        int[] v = new int[to - from];
        for (int i = from; i < to; i++) {
            v[i - from] = d.intArg(i);
        }
        return v;
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    // ------------------------------------------------------------------ sharded projections

    @Test
    public void prewarmWeightKernelsPreparesTheFamiliesOfEachUse() {
        open();
        ops.prewarmWeightKernels(GGMLType.Q4_K, true, false);
        List<String> projected = backend.preparedKernels();
        assertTrue(projected.toString(), projected.contains("Q4_K_MatVec"));
        assertTrue(projected.toString(), projected.contains("Q4_K_MatVecAdd"));
        assertTrue(projected.toString(), projected.contains("Q4_K_BatchMatMul"));
        assertTrue(projected.toString(), projected.stream().noneMatch(n -> n.contains("EmbeddingLookup")));

        ops.prewarmWeightKernels(GGMLType.Q8_0, false, true);
        List<String> all = backend.preparedKernels();
        assertTrue(all.toString(), all.contains("LLM_Batch_Q8_0_EmbeddingLookup"));
        assertTrue(all.toString(), all.stream().noneMatch(n -> n.startsWith("Q8_0_MatVec")));
        assertThrows(UnsupportedOperationException.class, () -> ops.prewarmWeightKernels(GGMLType.I32, true, true));
    }

    @Test
    public void constructionPreparesEveryHelperKernel() {
        open();
        List<String> prepared = backend.preparedKernels();
        for (String kernel : GpuOps.HELPER_KERNELS) {
            assertTrue(kernel + " not in " + prepared, prepared.contains(kernel));
        }
    }

    @Test
    public void constructionLeavesAMissingHelperKernelToItsFirstLaunch() {
        open(FakeComputeBackend.builder().strictLlmArgs(true).unavailableKernels(HadamardSpec.KERNEL).build());
        List<String> prepared = backend.preparedKernels();
        assertTrue(prepared.toString(), prepared.contains(GpuOps.ROW_NORM));
        assertFalse(prepared.toString(), prepared.contains(HadamardSpec.KERNEL));
    }

    @Test
    public void prewarmPreparesAnAvailableKernelAtTheGivenSize() {
        open();
        assertTrue(ops.prewarm(NormKernels.RESIDUAL_MIX, new long[]{local}));
        assertTrue(backend.preparedKernels().contains(NormKernels.RESIDUAL_MIX));
    }

    @Test
    public void prewarmSkipsAKernelTheBackendDoesNotProvide() {
        open(FakeComputeBackend.builder().strictLlmArgs(true).unavailableKernels(NormKernels.RESIDUAL_MIX).build());
        assertFalse(ops.prewarm(NormKernels.RESIDUAL_MIX, new long[]{local}));
        assertFalse(backend.preparedKernels().contains(NormKernels.RESIDUAL_MIX));
    }

    @Test
    public void prewarmRejectsMalformedSizesAndAClosedOps() {
        open();
        assertThrows(IllegalArgumentException.class, () -> ops.prewarm("", new long[]{local}));
        assertThrows(IllegalArgumentException.class, () -> ops.prewarm(GpuOps.AXPBY, new long[0]));
        assertThrows(IllegalArgumentException.class, () -> ops.prewarm(GpuOps.AXPBY, new long[]{1, 1, 1, 1}));
        assertThrows(IllegalArgumentException.class, () -> ops.prewarm(GpuOps.AXPBY, new long[]{0}));
        assertThrows(NullPointerException.class, () -> ops.prewarm(GpuOps.AXPBY, null));
        ops.close();
        assertThrows(IllegalStateException.class, () -> ops.prewarm(GpuOps.AXPBY, new long[]{local}));
    }

    @Test
    public void prepareWarmsTheKernelAtItsLaunchSize() {
        open();
        GpuBuffer x = floats(COLS);
        try (PreparedDispatch d = ops.prepare("Q4_K_MatVec", new long[]{(long) ROWS * local}, new long[]{local},
                ROWS, COLS, x, x, x)) {
            assertTrue(backend.preparedKernels().contains("Q4_K_MatVec"));
        }
    }

    @Test
    public void matvecOneTokenScattersEachShardAtItsRowOffset() {
        open();
        GpuWeight w = q4kSharded();
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Arrays.asList("Q4_K_MatVec", GpuOps.COPY2D, "Q4_K_MatVec", GpuOps.COPY2D,
                "Q4_K_MatVec", GpuOps.COPY2D), names(ds));
        for (int s = 0; s < 3; s++) {
            Dispatch mv = ds.get(2 * s);
            Dispatch cp = ds.get(2 * s + 1);
            assertArrayEquals(new long[]{(long) SHARD_ROWS * local}, mv.global());
            assertArrayEquals(new long[]{local}, mv.local());
            assertArrayEquals(new int[]{SHARD_ROWS, COLS}, ints(mv, 0, 2));
            assertSame(w.shard(s), mv.bufferArg(2));
            assertSame(in, mv.bufferArg(3));
            assertNotSame(out, mv.bufferArg(4));
            assertArrayEquals(new int[]{1, SHARD_ROWS, SHARD_ROWS, 0, ROWS, s * SHARD_ROWS}, ints(cp, 0, 6));
            assertSame(mv.bufferArg(4), cp.bufferArg(6));
            assertSame(out, cp.bufferArg(7));
            assertArrayEquals(new long[]{roundUp(SHARD_ROWS)}, cp.global());
        }
    }

    @Test
    public void matvecFourTokensUsesTwoDimensionalBatchAndPerTokenRowOffsets() {
        open();
        GpuWeight w = q4kSharded();
        GpuBuffer in = floats(4L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        int outOff = 2 * ROWS;
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, outOff, 4));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D", GpuOps.COPY2D, "Q4_K_BatchMatMul_2D", GpuOps.COPY2D,
                "Q4_K_BatchMatMul_2D", GpuOps.COPY2D), names(ds));
        for (int s = 0; s < 3; s++) {
            Dispatch mm = ds.get(2 * s);
            Dispatch cp = ds.get(2 * s + 1);
            assertArrayEquals(new long[]{(long) SHARD_ROWS * local, 4}, mm.global());
            assertArrayEquals(new long[]{local, 1}, mm.local());
            assertArrayEquals(new int[]{SHARD_ROWS, COLS, 4}, ints(mm, 0, 3));
            assertArrayEquals(new int[]{4, SHARD_ROWS, SHARD_ROWS, 0, ROWS, outOff + s * SHARD_ROWS},
                    ints(cp, 0, 6));
            assertSame(mm.bufferArg(5), cp.bufferArg(6));
            assertArrayEquals(new long[]{roundUp(4L * SHARD_ROWS)}, cp.global());
        }
    }

    @Test
    public void batchFallsBackToOneDimensionalKernelBeyondGroupCountLimit() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("SmallGridDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .maxWorkGroupCount(new long[]{65535, 2, 1})
                .build();
        open(FakeComputeBackend.builder().strictLlmArgs(true).capabilities(caps).build());
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(4L * COLS);
        GpuBuffer out = floats(4L * ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 4));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul"), names(ds));
        assertArrayEquals(new long[]{(long) ROWS * local}, ds.get(0).global());
        assertArrayEquals(new int[]{ROWS, COLS, 4}, ints(ds.get(0), 0, 3));
        assertSame(out, ds.get(0).bufferArg(5));
    }

    @Test
    public void singleShardAtOffsetZeroWritesDirectly() {
        open();
        GpuWeight w = GpuWeight.of("output.weight", GGMLType.F32, new long[]{COLS, ROWS},
                backend.allocateBuffer((long) ROWS * COLS * Float.BYTES, true));
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Arrays.asList("Float_MatVec"), names(ds));
        assertSame(out, ds.get(0).bufferArg(4));
    }

    @Test
    public void inputAtOffsetIsStagedFirst() {
        open();
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(5L * COLS);
        GpuBuffer out = floats(2L * ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, COLS, out, 0, 2));

        assertEquals(Arrays.asList(GpuOps.COPY2D, "Q4_K_BatchMatMul_2D"), names(ds));
        assertArrayEquals(new int[]{2, COLS, COLS, COLS, COLS, 0}, ints(ds.get(0), 0, 6));
        assertSame(in, ds.get(0).bufferArg(6));
        assertSame(ds.get(0).bufferArg(7), ds.get(1).bufferArg(4));
        assertSame(out, ds.get(1).bufferArg(5));
    }

    @Test
    public void prismWeightRotatesItsInputIntoStaging() {
        open();
        float[] signs = new float[COLS];
        Arrays.fill(signs, 1.0f);
        signs[3] = -1.0f;
        HadamardSpec h = HadamardSpec.of(64, HadamardSpec.SignOrder.BEFORE_ROTATION, signs, 0, 0, 0);
        GpuWeight w = q4kSingle().withHadamard(h);
        GpuBuffer in = floats(COLS * 2L);
        GpuBuffer out = floats(ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, COLS, out, 0, 1));

        assertEquals(Arrays.asList(HadamardSpec.KERNEL, "Q4_K_MatVec"), names(ds));
        Dispatch rot = ds.get(0);
        assertArrayEquals(new int[]{1, COLS, 64, 1, 0, 0, 0, COLS, 0}, ints(rot, 0, 9));
        assertArrayEquals(new long[]{(COLS / 64L) * local}, rot.global());
        assertSame(in, rot.bufferArg(HadamardSpec.ARG_X));
        assertNotSame(in, rot.bufferArg(HadamardSpec.ARG_Y));
        assertSame(rot.bufferArg(HadamardSpec.ARG_Y), ds.get(1).bufferArg(3));

        List<Dispatch> again = run(() -> ops.matvec(w, in, 0, out, 0, 1));
        assertSame("one signs buffer per sign vector", rot.bufferArg(HadamardSpec.ARG_SIGNS),
                again.get(0).bufferArg(HadamardSpec.ARG_SIGNS));
    }

    @Test
    public void matmulUsesTiledGemmWithOffsetsAndStrides() {
        open();
        GpuWeight w = q4kSharded();
        GpuBuffer in = floats(9L * COLS);
        GpuBuffer out = floats(9L * ROWS);
        List<Dispatch> ds = run(() -> ops.matmul(w, in, COLS, out, ROWS, 8));

        assertEquals(Arrays.asList("Q4_K_MatMul_Tiled", "Q4_K_MatMul_Tiled", "Q4_K_MatMul_Tiled"), names(ds));
        for (int s = 0; s < 3; s++) {
            Dispatch d = ds.get(s);
            assertArrayEquals(new long[]{GpuOps.TILED_WG, 1}, d.global());
            assertArrayEquals(new long[]{GpuOps.TILED_WG, 1}, d.local());
            assertArrayEquals(new int[]{SHARD_ROWS, COLS, 8, COLS, COLS, ROWS, ROWS + s * SHARD_ROWS},
                    ints(d, 0, 7));
            assertSame(w.shard(s), d.bufferArg(7));
            assertSame(in, d.bufferArg(8));
            assertSame(out, d.bufferArg(9));
        }
    }

    @Test
    public void matmulFallsBackToBatchedMatvecWithoutTiledKernel() {
        open(FakeComputeBackend.builder().strictLlmArgs(true).unavailableKernels("Q4_K_MatMul_Tiled").build());
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        List<Dispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, 8));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(ds));
    }

    @Test
    public void tiledGemmAboveItsKernelLimitIsSkippedAtPrewarmAndMatmulFallsBack() {
        openWithKernelLimit(GpuOps.TILED_WG / 2, "Q4_K_MatMul_Tiled");
        ops.prewarmWeightKernels(GGMLType.Q4_K, true, false);
        List<String> prepared = backend.preparedKernels();
        assertFalse(prepared.toString(), prepared.contains("Q4_K_MatMul_Tiled"));
        assertTrue(prepared.toString(), prepared.contains("Q4_K_MatVec"));

        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        List<Dispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, 8));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(ds));
    }

    @Test
    public void tiledGemmOfAWeightTypeNeverPrewarmedIsCheckedOnFirstUse() {
        openWithKernelLimit(GpuOps.TILED_WG / 2, "Q4_K_MatMul_Tiled");
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(run(() -> ops.matmul(w, in, 0, out, 0, 8))));
        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(run(() -> ops.matmul(w, in, 0, out, 0, 8))));
    }

    @Test
    public void batch2dAboveItsKernelLimitFallsBackToTheOneDimensionalBatch() {
        openWithKernelLimit(1, "Q4_K_BatchMatMul_2D");
        ops.prewarmWeightKernels(GGMLType.Q4_K, true, false);
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(2L * COLS);
        GpuBuffer out = floats(2L * ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 2));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul"), names(ds));
    }

    @Test
    public void matvecAddAboveItsKernelLimitProjectsThenAccumulates() {
        openWithKernelLimit(1, "Q4_K_MatVecAdd");
        ops.prewarmWeightKernels(GGMLType.Q4_K, true, false);
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(COLS);
        GpuBuffer acc = floats(ROWS);
        List<String> ds = names(run(() -> ops.matvecAdd(w, in, 0, acc, 0, 1)));

        assertFalse(ds.toString(), ds.contains("Q4_K_MatVecAdd"));
        assertEquals("Q4_K_MatVec", ds.get(0));
        assertEquals(GpuOps.AXPBY, ds.get(ds.size() - 1));
    }

    @Test
    public void requiredMatvecAboveItsKernelLimitFailsThePrewarm() {
        openWithKernelLimit(1, "Q4_K_MatVec");
        assertThrows(UnsupportedKernelException.class,
                () -> ops.prewarmWeightKernels(GGMLType.Q4_K, true, false));
    }

    @Test
    public void matvecBelowTiledThresholdNeverUsesGemm() {
        open();
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(7L * COLS);
        GpuBuffer out = floats(7L * ROWS);
        List<Dispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, 7));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(ds));
    }

    // ------------------------------------------------------------------ accumulate and scale

    @Test
    public void matvecAddFastPathAccumulatesInKernel() {
        open();
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(COLS);
        GpuBuffer acc = floats(ROWS);
        List<Dispatch> ds = run(() -> ops.matvecAdd(w, in, 0, acc, 0, 1));

        assertEquals(Arrays.asList("Q4_K_MatVecAdd"), names(ds));
        assertSame(acc, ds.get(0).bufferArg(4));
    }

    @Test
    public void shardedMatvecAddStagesThenAxpbys() {
        open();
        GpuWeight w = q4kSharded();
        GpuBuffer in = floats(COLS);
        GpuBuffer acc = floats(2L * ROWS);
        List<Dispatch> ds = run(() -> ops.matvecAdd(w, in, 0, acc, ROWS, 1));

        assertEquals(Arrays.asList("Q4_K_MatVec", GpuOps.COPY2D, "Q4_K_MatVec", GpuOps.COPY2D,
                "Q4_K_MatVec", GpuOps.COPY2D, GpuOps.AXPBY), names(ds));
        GpuBuffer result = ds.get(1).bufferArg(7);
        assertNotSame(acc, result);
        Dispatch ax = ds.get(6);
        assertEquals(ROWS, ax.intArg(0));
        assertEquals(1.0f, ax.floatArg(1), 0.0f);
        assertEquals(1.0f, ax.floatArg(2), 0.0f);
        assertArrayEquals(new int[]{0, ROWS, ROWS}, ints(ax, 3, 6));
        assertSame(result, ax.bufferArg(6));
        assertSame(acc, ax.bufferArg(7));
        assertSame(acc, ax.bufferArg(8));
    }

    @Test
    public void outputScaleMultipliesInPlace() {
        open();
        GpuWeight w = q4kSingle().withScale(0.5f);
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Arrays.asList("Q4_K_MatVec", GpuOps.AXPBY), names(ds));
        Dispatch ax = ds.get(1);
        assertEquals(0.0f, ax.floatArg(1), 0.0f);
        assertEquals(0.5f, ax.floatArg(2), 0.0f);
        assertSame(out, ax.bufferArg(7));
        assertSame(out, ax.bufferArg(8));
    }

    // ------------------------------------------------------------------ element-wise

    @Test
    public void addMovesTheOperandThatAliasesTheOutputToTheAliasableSlot() {
        open();
        GpuBuffer x = floats(64);
        GpuBuffer y = floats(64);
        List<Dispatch> ds = run(() -> ops.add(x, 0, y, 0, x, 0, 64));

        Dispatch ax = ds.get(0);
        assertEquals(GpuOps.AXPBY, ax.kernelName());
        assertSame(y, ax.bufferArg(6));
        assertSame(x, ax.bufferArg(7));
        assertSame(x, ax.bufferArg(8));
        assertArrayEquals(new long[]{roundUp(64)}, ax.global());
    }

    @Test
    public void partiallyOverlappingViewsAreRejected() {
        open();
        GpuBuffer x = floats(128);
        GpuBuffer y = floats(64);
        assertThrows(IllegalArgumentException.class, () -> ops.add(x, 0, y, 0, x, 8, 64));
        assertThrows(IllegalArgumentException.class, () -> ops.copy(x, 0, x, 32, 64));
        GpuWeight w = q4kSingle();
        GpuBuffer io = floats(COLS + ROWS);
        assertThrows(IllegalArgumentException.class, () -> ops.matvec(w, io, 0, io, 64, 1));
        assertThrows(IllegalArgumentException.class, () -> ops.matvec(w, io, 0, y, 0, 1));
    }

    @Test
    public void copyAtOffsetsUsesCopy2d() {
        open();
        GpuBuffer x = floats(128);
        List<Dispatch> ds = run(() -> ops.copy(x, 0, x, 64, 64));

        assertEquals(Arrays.asList(GpuOps.COPY2D), names(ds));
        assertArrayEquals(new int[]{1, 64, 64, 0, 64, 64}, ints(ds.get(0), 0, 6));
    }

    @Test
    public void interleavedColumnWindowsOfOneBufferAreDisjoint() {
        open();
        GpuBuffer x = floats(4L * 128);
        List<Dispatch> ds = run(() -> ops.copy2d(x, 0, 128, x, 64, 128, 4, 64));

        assertEquals(Arrays.asList(GpuOps.COPY2D), names(ds));
        assertArrayEquals(new int[]{4, 64, 128, 0, 128, 64}, ints(ds.get(0), 0, 6));
        assertThrows(IllegalArgumentException.class, () -> ops.copy2d(x, 0, 128, x, 32, 128, 4, 64));
        assertThrows(IllegalArgumentException.class, () -> ops.copy2d(x, 0, 128, x, 64, 96, 4, 64));
    }

    // ------------------------------------------------------------------ prepared dispatches

    @Test
    public void preparedDispatchBindsOnceOnAClonedHandle() {
        open();
        GpuBuffer a = floats(64);
        GpuBuffer b = floats(64);
        PreparedDispatch d = ops.prepare(GpuOps.AXPBY, new long[]{roundUp(64)}, new long[]{local},
                64, 1.0f, 1.0f, 0, 0, 0, a, b, b);
        backend.clearLog();
        d.setInt(0, 32);
        d.dispatch();
        Dispatch first = backend.dispatches().get(0);
        assertTrue(first.isPreBound());
        assertEquals(32, first.intArg(0));
        assertThrows(IllegalArgumentException.class, () -> d.setFloat(0, 1.0f));
        assertThrows(IllegalArgumentException.class, () -> d.setBuffer(1, a));
        assertThrows(IllegalArgumentException.class, () -> d.setInt(9, 1));

        backend.clearLog();
        ops.add(a, 0, b, 0, b, 0, 64);
        d.dispatch();
        assertEquals("the library handle's arguments do not disturb the clone", 32,
                backend.dispatches().get(1).intArg(0));
        d.close();
        assertThrows(IllegalStateException.class, d::dispatch);
    }

    // ------------------------------------------------------------------ weights and scratch

    @Test
    public void weightValidatesShardsAndTransforms() {
        open();
        GpuBuffer small = backend.allocateBuffer(SHARD_ROWS * Q4K_ROW_BYTES - 1, true);
        GpuBuffer ok = backend.allocateBuffer(SHARD_ROWS * Q4K_ROW_BYTES, true);
        assertThrows(IllegalArgumentException.class, () -> GpuWeight.sharded("t", GGMLType.Q4_K,
                new long[]{COLS, 2 * SHARD_ROWS}, Arrays.asList(ok, small), new long[]{SHARD_ROWS, SHARD_ROWS}));
        assertThrows(IllegalArgumentException.class, () -> GpuWeight.sharded("t", GGMLType.Q4_K,
                new long[]{COLS, 3 * SHARD_ROWS}, Arrays.asList(ok, ok), new long[]{SHARD_ROWS, SHARD_ROWS}));
        assertThrows(IllegalArgumentException.class, () -> GpuWeight.of("t", GGMLType.Q4_K, new long[]{100, 4}, ok));

        GpuWeight wide = GpuWeight.of("wide", GGMLType.F32, new long[]{2048, 1},
                backend.allocateBuffer(2048L * Float.BYTES, true));
        assertThrows(IllegalArgumentException.class, () -> wide.withHadamard(HadamardSpec.rotation(2048)));
        assertEquals(1, q4kSharded().shardOfRow(SHARD_ROWS));
    }

    @Test
    public void temporariesGrowWithoutReleasingEarlierBuffers() {
        open();
        ScratchSlot a = scratch.temp("attn.scores", 100);
        assertSame(a.buffer(), scratch.temp("attn.scores", 50).buffer());
        ScratchSlot b = scratch.temp("attn.scores", 1000);
        assertNotSame(a.buffer(), b.buffer());
        assertTrue(backend.liveBuffers().contains(a.buffer()));
        assertThrows(IllegalArgumentException.class, () -> scratch.temp(ScratchSlots.X, 4));
        assertThrows(IllegalArgumentException.class, () -> scratch.slot(ScratchSlots.LOGITS));
    }
}
