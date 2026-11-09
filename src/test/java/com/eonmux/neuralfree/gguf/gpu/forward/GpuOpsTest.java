package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqMode;
import com.eonmux.neuralfree.gguf.ops.gemm.SplitKGemmPlan;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

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

    private List<FakeDispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    private static List<String> names(List<FakeDispatch> ds) {
        List<String> out = new ArrayList<>();
        for (FakeDispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static int[] ints(FakeDispatch d, int from, int to) {
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
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Arrays.asList("Q4_K_MatVec", GpuOps.COPY2D, "Q4_K_MatVec", GpuOps.COPY2D,
                "Q4_K_MatVec", GpuOps.COPY2D), names(ds));
        for (int s = 0; s < 3; s++) {
            FakeDispatch mv = ds.get(2 * s);
            FakeDispatch cp = ds.get(2 * s + 1);
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

    /** A single-shard weight whose paired launch has at least {@link GpuOps#TWO_ROW_MIN_THREADS} threads. */
    private GpuWeight q4kSingleOfRows(int rows) {
        return q4kSingleOfRows(rows, "blk.0.big.weight");
    }

    private GpuWeight q4kSingleOfRows(int rows, String name) {
        return GpuWeight.of(name, GGMLType.Q4_K, new long[]{COLS, rows},
                backend.allocateBuffer(rows * Q4K_ROW_BYTES, true));
    }

    @Test
    public void matvecOneTokenPairsRowsInTheTwoRowKernelAboveTheThreadFloor() {
        open();
        GpuWeight w = q4kSingleOfRows(1024);
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(1024);
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Collections.singletonList("Q4_K_MatVec2"), names(ds));
        FakeDispatch mv = ds.get(0);
        assertArrayEquals(new long[]{512L * local}, mv.global());
        assertArrayEquals(new long[]{local}, mv.local());
        assertArrayEquals(new int[]{1024, COLS}, ints(mv, 0, 2));
        assertSame(w.shard(0), mv.bufferArg(2));
        assertSame(in, mv.bufferArg(3));
        assertSame(out, mv.bufferArg(4));
    }

    @Test
    public void matvecOneTokenBelowTheTwoRowThreadFloorKeepsTheSingleRowKernel() {
        open();
        GpuWeight w = q4kSingleOfRows(128);
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(128);
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Collections.singletonList("Q4_K_MatVec"), names(ds));
        assertArrayEquals(new long[]{128L * local}, ds.get(0).global());
    }

    @Test
    public void fusedGateUpOneTokenPairsRowsInTheTwoRowKernelAboveTheThreadFloor() {
        open();
        GpuWeight gate = q4kSingleOfRows(1024, "blk.0.ffn_gate.weight");
        GpuWeight up = q4kSingleOfRows(1024, "blk.0.ffn_up.weight");
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(1024);
        List<FakeDispatch> ds = run(() -> ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, out));

        assertEquals(Collections.singletonList("Q4_K_MatVec2FusedGateUp"), names(ds));
        FakeDispatch fu = ds.get(0);
        assertArrayEquals(new long[]{512L * local}, fu.global());
        assertArrayEquals(new long[]{local}, fu.local());
        assertArrayEquals(new int[]{1024, COLS}, ints(fu, 0, 2));
        assertSame(gate.shard(0), fu.bufferArg(2));
        assertSame(up.shard(0), fu.bufferArg(3));
        assertSame(in, fu.bufferArg(4));
        assertSame(out, fu.bufferArg(5));
    }

    @Test
    public void fusedGateUpOneTokenBelowTheTwoRowThreadFloorKeepsTheSingleRowKernel() {
        open();
        GpuWeight gate = q4kSingleOfRows(128, "blk.0.ffn_gate.weight");
        GpuWeight up = q4kSingleOfRows(128, "blk.0.ffn_up.weight");
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(128);
        List<FakeDispatch> ds = run(() -> ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, out));

        assertEquals(Collections.singletonList("Q4_K_MatVecFusedGateUp"), names(ds));
        assertArrayEquals(new long[]{128L * local}, ds.get(0).global());
    }

    @Test
    public void matvecFourTokensUsesTwoDimensionalBatchAndPerTokenRowOffsets() {
        open();
        GpuWeight w = q4kSharded();
        GpuBuffer in = floats(4L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        int outOff = 2 * ROWS;
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, outOff, 4));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D", GpuOps.COPY2D, "Q4_K_BatchMatMul_2D", GpuOps.COPY2D,
                "Q4_K_BatchMatMul_2D", GpuOps.COPY2D), names(ds));
        for (int s = 0; s < 3; s++) {
            FakeDispatch mm = ds.get(2 * s);
            FakeDispatch cp = ds.get(2 * s + 1);
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
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 4));

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
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Arrays.asList("Float_MatVec"), names(ds));
        assertSame(out, ds.get(0).bufferArg(4));
    }

    @Test
    public void inputAtOffsetIsStagedFirst() {
        open();
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(5L * COLS);
        GpuBuffer out = floats(2L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, COLS, out, 0, 2));

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
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, COLS, out, 0, 1));

        assertEquals(Arrays.asList(HadamardSpec.KERNEL, "Q4_K_MatVec"), names(ds));
        FakeDispatch rot = ds.get(0);
        assertArrayEquals(new int[]{1, COLS, 64, 1, 0, 0, 0, COLS, 0}, ints(rot, 0, 9));
        assertArrayEquals(new long[]{(COLS / 64L) * local}, rot.global());
        assertSame(in, rot.bufferArg(HadamardSpec.ARG_X));
        assertNotSame(in, rot.bufferArg(HadamardSpec.ARG_Y));
        assertSame(rot.bufferArg(HadamardSpec.ARG_Y), ds.get(1).bufferArg(3));

        List<FakeDispatch> again = run(() -> ops.matvec(w, in, 0, out, 0, 1));
        assertSame("one signs buffer per sign vector", rot.bufferArg(HadamardSpec.ARG_SIGNS),
                again.get(0).bufferArg(HadamardSpec.ARG_SIGNS));
    }

    @Test
    public void matmulUsesMmqGemmWithOffsetsAndStrides() {
        open();
        GpuWeight w = q4kSharded();
        GpuBuffer in = floats(9L * COLS);
        GpuBuffer out = floats(9L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(w, in, COLS, out, ROWS, 8));

        assertEquals("the input is staged once, then every shard's row tile reads the staged records",
                Arrays.asList(GemmKernels.QUANT_MMQ_Q8_1, "Q4_K_MatMul_MMQ", "Q4_K_MatMul_MMQ", "Q4_K_MatMul_MMQ"),
                names(ds));
        FakeDispatch quant = ds.get(0);
        assertArrayEquals("the staging pass reads the caller's input view",
                new int[]{8, COLS, MmqConfig.PACKED_DOT.stageDepth(), COLS, COLS}, ints(quant, 0, 5));
        assertSame(in, quant.bufferArg(5));
        GpuBuffer staged = quant.bufferArg(6);
        for (int s = 0; s < 3; s++) {
            FakeDispatch d = ds.get(1 + s);
            assertArrayEquals(new long[]{GpuProjections.TILED_WG, 1}, d.global());
            assertArrayEquals(new long[]{GpuProjections.TILED_WG, 1}, d.local());
            assertArrayEquals(new int[]{SHARD_ROWS, COLS, 8, MmqConfig.PACKED_DOT.rowRecords(COLS), 0,
                    ROWS, ROWS + s * SHARD_ROWS}, ints(d, 0, 7));
            assertSame(w.shard(s), d.bufferArg(7));
            assertSame("every row tile reads the same staged records", staged, d.bufferArg(8));
            assertSame(out, d.bufferArg(9));
        }
    }

    @Test
    public void matmulOfAnMmqTypeFallsBackToTiledWithoutTheMmqKernel() {
        open(FakeComputeBackend.builder().strictLlmArgs(true).unavailableKernels("Q4_K_MatMul_MMQ").build());
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, 8));

        assertEquals(Arrays.asList("Q4_K_MatMul_Tiled"), names(ds));
    }

    /**
     * {@code neuralfree.gemm.mmq=off} serves the MMQ types with the exact tiled GEMM and skips the
     * fast lane at prewarm; {@code on} and {@code auto} keep the MMQ kernel. The property binds once
     * per GpuOps, and an unknown value fails the bind.
     */
    @Test
    public void theMmqSwitchChoosesBetweenTheFastQuantLaneAndTheTiledGemm() {
        System.setProperty(MmqMode.PROPERTY, "off");
        try {
            open();
            GpuWeight w = q4kSingle();
            GpuBuffer in = floats(8L * COLS);
            GpuBuffer out = floats(8L * ROWS);
            ops.prewarmWeightKernels(GGMLType.Q4_K, true, false);
            assertFalse(backend.preparedKernels().toString(), backend.preparedKernels().contains("Q4_K_MatMul_MMQ"));
            assertFalse(backend.preparedKernels().toString(),
                    backend.preparedKernels().contains(GemmKernels.QUANT_MMQ_Q8_1));
            assertTrue(backend.preparedKernels().toString(), backend.preparedKernels().contains("Q4_K_MatMul_Tiled"));
            assertEquals(Arrays.asList("Q4_K_MatMul_Tiled"), names(run(() -> ops.matmul(w, in, 0, out, 0, 8))));
        } finally {
            System.clearProperty(MmqMode.PROPERTY);
        }

        System.setProperty(MmqMode.PROPERTY, "on");
        try {
            open();
            GpuWeight w = q4kSingle();
            GpuBuffer in = floats(8L * COLS);
            GpuBuffer out = floats(8L * ROWS);
            assertEquals(Arrays.asList(GemmKernels.QUANT_MMQ_Q8_1, "Q4_K_MatMul_MMQ"),
                    names(run(() -> ops.matmul(w, in, 0, out, 0, 8))));
        } finally {
            System.clearProperty(MmqMode.PROPERTY);
        }

        System.setProperty(MmqMode.PROPERTY, "maybe");
        try {
            open();
            fail("expected the unknown value to fail the bind");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(MmqMode.PROPERTY));
            assertTrue(e.getMessage(), e.getMessage().contains("maybe"));
        } finally {
            System.clearProperty(MmqMode.PROPERTY);
        }
    }

    /**
     * On a Vulkan device with cooperative matrices the fast quantized-weight GEMM runs the CM form,
     * the switch turns the whole lane off, and F16 weights keep their measured crossover
     * ({@link SmallBatchCrossover}) the switch never touches.
     */
    @Test
    public void vulkanCooperativeMatricesServeQuantWeightsThroughTheCmForm() {
        open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN)
                .strictLlmArgs(true)
                .capabilities(tensorCores(ComputeBackend.BackendType.VULKAN, 32)).build());
        GpuWeight q4k = q4kSingle();
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        assertEquals(Arrays.asList("Q4_K_MatMul_CM"), names(run(() -> ops.matmul(q4k, in, 0, out, 0, 8))));

        GpuWeight f16 = GpuWeight.of("blk.0.attn_q.weight", GGMLType.F16, new long[]{COLS, ROWS},
                backend.allocateBuffer((long) ROWS * COLS * 2, true));
        // The measured Vulkan crossover (SmallBatchCrossover) retires the MMA lane: it lost at every
        // step count on the machine's Vulkan device, so F16 weights fall through to the tiled GEMM
        // at this row count like the quantized types do with their lane off.
        assertEquals(Arrays.asList("F16_MatMul_Tiled"), names(run(() -> ops.matmul(f16, in, 0, out, 0, 8))));

        System.setProperty(MmqMode.PROPERTY, "off");
        try {
            open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN)
                    .strictLlmArgs(true)
                    .capabilities(tensorCores(ComputeBackend.BackendType.VULKAN, 32)).build());
            GpuWeight q4kOff = q4kSingle();
            GpuWeight f16Off = GpuWeight.of("blk.0.attn_q.weight", GGMLType.F16, new long[]{COLS, ROWS},
                    backend.allocateBuffer((long) ROWS * COLS * 2, true));
            GpuBuffer inOff = floats(8L * COLS);
            GpuBuffer outOff = floats(8L * ROWS);
            assertEquals(Arrays.asList("Q4_K_MatMul_Tiled"),
                    names(run(() -> ops.matmul(q4kOff, inOff, 0, outOff, 0, 8))));
            assertEquals("the MMA lane is not part of the switch",
                    Arrays.asList("F16_MatMul_Tiled"), names(run(() -> ops.matmul(f16Off, inOff, 0, outOff, 0, 8))));
        } finally {
            System.clearProperty(MmqMode.PROPERTY);
        }

        System.setProperty(MmqMode.PROPERTY, "on");
        try {
            open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN)
                    .strictLlmArgs(true)
                    .capabilities(tensorCores(ComputeBackend.BackendType.VULKAN, 32)).build());
            GpuWeight q4kOn = q4kSingle();
            GpuBuffer inOn = floats(8L * COLS);
            GpuBuffer outOn = floats(8L * ROWS);
            assertEquals("on forces the int8 MMQ GEMM over the CM form",
                    Arrays.asList(GemmKernels.QUANT_MMQ_Q8_1, "Q4_K_MatMul_MMQ"),
                    names(run(() -> ops.matmul(q4kOn, inOn, 0, outOn, 0, 8))));
        } finally {
            System.clearProperty(MmqMode.PROPERTY);
        }
    }

    /** BF16 weights keep the exact tiled GEMM on Vulkan, whose MMA twin stages fp16 tiles. */
    @Test
    public void bf16WeightsKeepTheTiledGemmOnVulkan() {
        open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN)
                .strictLlmArgs(true)
                .capabilities(tensorCores(ComputeBackend.BackendType.VULKAN, 32)).build());
        GpuWeight bf16 = GpuWeight.of("blk.0.attn_v.weight", GGMLType.BF16, new long[]{COLS, ROWS},
                backend.allocateBuffer((long) ROWS * COLS * 2, true));
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        assertEquals(Arrays.asList("BF16_MatMul_Tiled"), names(run(() -> ops.matmul(bf16, in, 0, out, 0, 8))));
    }

    /**
     * The MMQ launch grid follows the device's tile config (llama.cpp's mmq-config tables): 64-token tiles on
     * CUDA sm_80+ (the int8 tensor path) and 16-token tiles below it, so 40 tokens need one or three tile
     * columns. The config and the grid it plans are pure geometry, so they are checked without loading the
     * kernels for a faked device.
     */
    @Test
    public void mmqGridsUseTheTileConfigOfTheDevice() {
        assertEquals(MmqConfig.INT8_TENSOR, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 80));
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 75));
        assertEquals("sm_120 uses the 128-token int8-tensor tile",
                128, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 120).tileTokens());
        assertEquals("sm_75 uses the 16-token packed-dot tile",
                16, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 75).tileTokens());
        int[][] cases = {{120, 1}, {80, 1}, {75, 3}};       /* compute capability, tile columns of 40 tokens */
        for (int[] c : cases) {
            MmqConfig config = MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, c[0]);
            assertEquals("sm_" + c[0] + " tile columns", c[1], GpuOps.ceilDiv(40L, config.tileTokens()));
        }
    }

    /** BF16 weights run the BF16 row of the MMA family on tensor-core devices. */
    @Test
    public void bf16WeightsRunTheMmaKernelOnTensorCoreDevices() {
        open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.CUDA)
                .strictLlmArgs(true)
                .capabilities(tensorCores(ComputeBackend.BackendType.CUDA, 32)).build());
        GpuWeight bf16 = GpuWeight.of("blk.0.attn_v.weight", GGMLType.BF16, new long[]{COLS, ROWS},
                backend.allocateBuffer((long) ROWS * COLS * 2, true));
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(bf16, in, 0, out, 0, 8));

        assertEquals(Arrays.asList("BF16_MatMul_MMA"), names(ds));
        for (FakeDispatch d : ds) {
            assertEquals(ROWS, d.intArg(GemmKernels.ARG_Y_STRIDE));
            assertEquals((long) ROWS * COLS * 2, d.bufferArg(GemmKernels.ARG_WEIGHT).sizeInBytes());
        }
    }

    /**
     * A tall-skinny MMA launch over a large K splits into {@code <T>_MatMul_MMA_KSplit} planes plus
     * one {@code LLM_SplitK_Reduce}; a launch the heuristic does not split stays on the plain kernel.
     */
    @Test
    public void tallSkinnyMmaLaunchesSplitKIntoPlanesAndAReduce() {
        open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.CUDA)
                .strictLlmArgs(true)
                .capabilities(tensorCores(ComputeBackend.BackendType.CUDA, 4)).build());
        final int wideCols = 2048;
        GpuWeight f16 = GpuWeight.of("blk.0.ffn_down.weight", GGMLType.F16, new long[]{wideCols, ROWS},
                backend.allocateBuffer((long) ROWS * wideCols * 2, true));
        GpuBuffer in = floats(8L * wideCols);
        GpuBuffer out = floats(8L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(f16, in, 0, out, 0, 8));

        assertEquals("two 64-row tiles x one token tile over 4 units split by the fill ratio", 2,
                SplitKGemmPlan.splitsFor(4, 2, 1, wideCols));
        assertEquals(Arrays.asList("F16_MatMul_MMA_KSplit", GemmKernels.SPLIT_K_REDUCE), names(ds));
        FakeDispatch partial = ds.get(0);
        assertArrayEquals(new int[]{ROWS, wideCols, 8, wideCols, 0, SplitKGemmPlan.sliceLength(wideCols, 2)},
                ints(partial, 0, 6));
        assertSame(f16.shard(0), partial.bufferArg(GemmKernels.KSPLIT_ARG_WEIGHT));
        assertSame(in, partial.bufferArg(GemmKernels.KSPLIT_ARG_X));
        FakeDispatch reduce = ds.get(1);
        assertArrayEquals(new int[]{ROWS * 8, 2, ROWS, ROWS, 0}, ints(reduce, 0, 5));
        assertSame(reduce.bufferArg(GemmKernels.REDUCE_ARG_PARTIAL), partial.bufferArg(GemmKernels.KSPLIT_ARG_PARTIAL));
        assertSame(out, reduce.bufferArg(GemmKernels.REDUCE_ARG_Y));

        GpuBuffer acc = floats(8L * ROWS);
        List<FakeDispatch> accumulate = run(() -> ops.matvecAdd(f16, in, 0, acc, 0, 8));
        assertEquals(Arrays.asList("F16_MatMul_MMA_KSplit", GemmKernels.SPLIT_K_REDUCE),
                names(accumulate).subList(0, 2));
        assertNotSame("the planes never alias the staged projection result",
                accumulate.get(0).bufferArg(GemmKernels.KSPLIT_ARG_PARTIAL),
                accumulate.get(1).bufferArg(GemmKernels.REDUCE_ARG_Y));

        GpuWeight denseGrid = GpuWeight.of("blk.1.ffn_down.weight", GGMLType.F16, new long[]{wideCols, 512},
                backend.allocateBuffer(512L * wideCols * 2, true));
        GpuBuffer wideOut = floats(8L * 512);
        assertEquals("a tile grid that fills the device stays on the plain kernel",
                Arrays.asList("F16_MatMul_MMA"), names(run(() -> ops.matmul(denseGrid, in, 0, wideOut, 0, 8))));
    }

    /**
     * A tall-skinny MMQ launch splits into {@code <T>_MatMul_MMQ_KSplit} planes plus one
     * {@code LLM_SplitK_Reduce}, the slices covering whole weight blocks of the staged records.
     */
    @Test
    public void tallSkinnyMmqLaunchesSplitKIntoPlanesAndAReduce() {
        open(FakeComputeBackend.builder().strictLlmArgs(true).capabilities(capsWithUnits(4)).build());
        final int wideCols = 2048;
        GpuWeight q4k = GpuWeight.of("blk.0.ffn_down.weight", GGMLType.Q4_K, new long[]{wideCols, ROWS},
                backend.allocateBuffer((long) ROWS / 16 * 144 * wideCols, true));
        GpuBuffer in = floats(8L * wideCols);
        GpuBuffer out = floats(8L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(q4k, in, 0, out, 0, 8));

        assertEquals(2, SplitKGemmPlan.splitsFor(4, 2, 1, wideCols));
        assertEquals(Arrays.asList(GemmKernels.QUANT_MMQ_Q8_1, "Q4_K_MatMul_MMQ_KSplit",
                GemmKernels.SPLIT_K_REDUCE), names(ds));
        FakeDispatch quant = ds.get(0);
        assertSame(in, quant.bufferArg(5));
        GpuBuffer staged = quant.bufferArg(6);
        FakeDispatch partial = ds.get(1);
        assertArrayEquals(new int[]{ROWS, wideCols, 8, MmqConfig.PACKED_DOT.rowRecords(wideCols), 0,
                        SplitKGemmPlan.sliceLength(wideCols, 2, GGMLType.Q4_K.getBlockSize())},
                ints(partial, 0, 6));
        assertSame(q4k.shard(0), partial.bufferArg(GemmKernels.KSPLIT_ARG_WEIGHT));
        assertSame(staged, partial.bufferArg(GemmKernels.KSPLIT_ARG_X));
        FakeDispatch reduce = ds.get(2);
        assertArrayEquals(new int[]{ROWS * 8, 2, ROWS, ROWS, 0}, ints(reduce, 0, 5));
        assertSame(reduce.bufferArg(GemmKernels.REDUCE_ARG_PARTIAL), partial.bufferArg(GemmKernels.KSPLIT_ARG_PARTIAL));
        assertSame(out, reduce.bufferArg(GemmKernels.REDUCE_ARG_Y));
    }

    /**
     * A tall-skinny CM launch on a cooperative-matrix Vulkan device splits into
     * {@code <T>_MatMul_CM_KSplit} planes plus one {@code LLM_SplitK_Reduce} over the unstaged
     * activation view: the view's offset and stride travel into the partial kernel's arguments.
     */
    @Test
    public void tallSkinnyCmLaunchesSplitKIntoPlanesAndAReduce() {
        open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN)
                .strictLlmArgs(true)
                .capabilities(tensorCores(ComputeBackend.BackendType.VULKAN, 4)).build());
        final int wideCols = 2048;
        GpuWeight q4k = GpuWeight.of("blk.0.ffn_down.weight", GGMLType.Q4_K, new long[]{wideCols, ROWS},
                backend.allocateBuffer((long) ROWS / 16 * 144 * wideCols, true));
        GpuBuffer in = floats(9L * wideCols);
        GpuBuffer out = floats(8L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(q4k, in, wideCols, out, 0, 8));

        assertEquals(2, SplitKGemmPlan.splitsFor(4, 2, 1, wideCols));
        assertEquals(Arrays.asList("Q4_K_MatMul_CM_KSplit", GemmKernels.SPLIT_K_REDUCE), names(ds));
        FakeDispatch partial = ds.get(0);
        assertArrayEquals(new int[]{ROWS, wideCols, 8, wideCols, wideCols,
                        SplitKGemmPlan.sliceLength(wideCols, 2)},
                ints(partial, 0, 6));
        assertSame(q4k.shard(0), partial.bufferArg(GemmKernels.KSPLIT_ARG_WEIGHT));
        assertSame("the CM lane reads the unstaged activation view", in,
                partial.bufferArg(GemmKernels.KSPLIT_ARG_X));
        FakeDispatch reduce = ds.get(1);
        assertArrayEquals(new int[]{ROWS * 8, 2, ROWS, ROWS, 0}, ints(reduce, 0, 5));
        assertSame(reduce.bufferArg(GemmKernels.REDUCE_ARG_PARTIAL), partial.bufferArg(GemmKernels.KSPLIT_ARG_PARTIAL));
        assertSame(out, reduce.bufferArg(GemmKernels.REDUCE_ARG_Y));
    }

    /**
     * A device that does not report its compute units (Vulkan without the shader-core extensions) never splits,
     * whatever the lane or the grid: the plain kernel serves the launch instead of the pipeline failing in the
     * heuristic. Only a reported count lets the fill ratio choose a split.
     */
    @Test
    public void unknownComputeUnitsKeepTallSkinnyMmqOnThePlainKernel() {
        open(FakeComputeBackend.builder().strictLlmArgs(true)
                .capabilities(capsWithUnits(DeviceCapabilities.UNKNOWN_COMPUTE_UNITS)).build());
        final int wideCols = 2048;
        GpuWeight q4k = GpuWeight.of("blk.0.ffn_down.weight", GGMLType.Q4_K, new long[]{wideCols, ROWS},
                backend.allocateBuffer((long) ROWS / 16 * 144 * wideCols, true));
        GpuBuffer in = floats(8L * wideCols);
        GpuBuffer out = floats(8L * ROWS);

        assertEquals(Arrays.asList(GemmKernels.QUANT_MMQ_Q8_1, "Q4_K_MatMul_MMQ"),
                names(run(() -> ops.matmul(q4k, in, 0, out, 0, 8))));
    }

    @Test
    public void unknownComputeUnitsKeepTallSkinnyCmOnThePlainKernel() {
        open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN)
                .strictLlmArgs(true)
                .capabilities(tensorCores(ComputeBackend.BackendType.VULKAN,
                        DeviceCapabilities.UNKNOWN_COMPUTE_UNITS)).build());
        final int wideCols = 2048;
        GpuWeight q4k = GpuWeight.of("blk.0.ffn_down.weight", GGMLType.Q4_K, new long[]{wideCols, ROWS},
                backend.allocateBuffer((long) ROWS / 16 * 144 * wideCols, true));
        GpuBuffer in = floats(8L * wideCols);
        GpuBuffer out = floats(8L * ROWS);

        assertEquals(Arrays.asList("Q4_K_MatMul_CM"), names(run(() -> ops.matmul(q4k, in, 0, out, 0, 8))));
    }

    @Test
    public void unknownComputeUnitsKeepTallSkinnyMmaOnThePlainKernel() {
        open(FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.CUDA)
                .strictLlmArgs(true)
                .capabilities(tensorCores(ComputeBackend.BackendType.CUDA,
                        DeviceCapabilities.UNKNOWN_COMPUTE_UNITS)).build());
        final int wideCols = 2048;
        GpuWeight f16 = GpuWeight.of("blk.0.ffn_down.weight", GGMLType.F16, new long[]{wideCols, ROWS},
                backend.allocateBuffer((long) ROWS * wideCols * 2, true));
        GpuBuffer in = floats(8L * wideCols);
        GpuBuffer out = floats(8L * ROWS);

        assertEquals(Arrays.asList("F16_MatMul_MMA"), names(run(() -> ops.matmul(f16, in, 0, out, 0, 8))));
    }

    /** Device capabilities of the default fake device with the given compute-unit count. */
    private static DeviceCapabilities capsWithUnits(int computeUnits) {
        return new DeviceCapabilities.Builder()
                .deviceName("StageDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .maxLocalMemoryBytes(48L * 1024)
                .build();
    }

    /**
     * Fake tensor-core device capabilities of a backend type: CUDA with sm_120, Vulkan with the LLM
     * library's storage and timeline features, both with {@link DeviceFeature#COOPERATIVE_MATRIX}.
     * The vendor and class claim the measured device of the backend (NVIDIA discrete over CUDA,
     * Intel integrated over Vulkan), so the dispatch reads that device kind's measured entries.
     */
    private static DeviceCapabilities tensorCores(ComputeBackend.BackendType type, int computeUnits) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("TensorDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .feature(DeviceFeature.COOPERATIVE_MATRIX);
        if (type == ComputeBackend.BackendType.CUDA) {
            // sm_80+ always reports both matrix features (CudaCapabilitiesBuilder), so the fake
            // device mirrors what the CUDA backend builds for the compute capability below.
            b.computeCapability(120).feature(DeviceFeature.BF16_COOPERATIVE_MATRIX)
                    .deviceClass(DeviceClass.DISCRETE)
                    .identity(new DeviceIdentity("TensorDevice", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null));
        } else {
            b.feature(DeviceFeature.STORAGE_8BIT).feature(DeviceFeature.STORAGE_16BIT)
                    .feature(DeviceFeature.TIMELINE_SEMAPHORE)
                    .deviceClass(DeviceClass.INTEGRATED)
                    .identity(new DeviceIdentity("TensorDevice", FakeComputeBackend.INTEL_VENDOR_ID, null, null));
        }
        return b.build();
    }

    @Test
    public void matmulFallsBackToBatchedMatvecWithoutAnyGemmKernel() {
        open(FakeComputeBackend.builder().strictLlmArgs(true)
                .unavailableKernels("Q4_K_MatMul_MMQ", "Q4_K_MatMul_Tiled").build());
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, 8));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(ds));
    }

    @Test
    public void gemmsAboveTheirKernelLimitAreSkippedAtPrewarmAndMatmulFallsBack() {
        openWithKernelLimit(GpuProjections.TILED_WG / 2, "Q4_K_MatMul_Tiled", "Q4_K_MatMul_MMQ");
        ops.prewarmWeightKernels(GGMLType.Q4_K, true, false);
        List<String> prepared = backend.preparedKernels();
        assertFalse(prepared.toString(), prepared.contains("Q4_K_MatMul_Tiled"));
        assertFalse(prepared.toString(), prepared.contains("Q4_K_MatMul_MMQ"));
        assertTrue(prepared.toString(), prepared.contains("Q4_K_MatVec"));

        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(8L * COLS);
        GpuBuffer out = floats(8L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, 8));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(ds));
    }

    @Test
    public void gemmOfAWeightTypeNeverPrewarmedIsCheckedOnFirstUse() {
        openWithKernelLimit(GpuProjections.TILED_WG / 2, "Q4_K_MatMul_Tiled", "Q4_K_MatMul_MMQ");
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
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 2));

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
    public void belowTheFastGemmThresholdTheMatvecServes() {
        open();
        GpuWeight w = q4kSingle();
        int n = SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS - 1;
        GpuBuffer in = floats((long) n * COLS);
        GpuBuffer out = floats((long) n * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, n));

        assertEquals(Arrays.asList("Q4_K_BatchMatMul_2D"), names(ds));
    }

    @Test
    public void fromTheFastGemmThresholdAWeightWithAFastLaneTakesTheGemm() {
        open();
        GpuWeight w = q4kSingle();
        int n = LlmKernelConstants.TILED_MIN_TOKENS - 1;
        GpuBuffer in = floats((long) n * COLS);
        GpuBuffer out = floats((long) n * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matmul(w, in, 0, out, 0, n));

        assertEquals(Arrays.asList(GemmKernels.QUANT_MMQ_Q8_1, "Q4_K_MatMul_MMQ"), names(ds));
    }

    @Test
    public void aWeightWithoutAFastLaneKeepsTheMatvecBelowTheTiledThreshold() {
        open();
        GpuWeight w = GpuWeight.of("blk.0.attn_q.weight", GGMLType.F32, new long[]{COLS, ROWS},
                backend.allocateBuffer((long) ROWS * COLS * Float.BYTES, true));
        int n = LlmKernelConstants.TILED_MIN_TOKENS - 1;
        GpuBuffer in = floats((long) n * COLS);
        GpuBuffer out = floats((long) n * ROWS);
        List<String> names = names(run(() -> ops.matmul(w, in, 0, out, 0, n)));

        assertTrue(names.toString(), names.stream().noneMatch(k -> k.contains("MatMul_MMQ") || k.contains("MatMul_MMA")
                || k.contains("Tiled")));
    }

    // ------------------------------------------------------------------ accumulate and scale

    @Test
    public void matvecAddFastPathAccumulatesInKernel() {
        open();
        GpuWeight w = q4kSingle();
        GpuBuffer in = floats(COLS);
        GpuBuffer acc = floats(ROWS);
        List<FakeDispatch> ds = run(() -> ops.matvecAdd(w, in, 0, acc, 0, 1));

        assertEquals(Arrays.asList("Q4_K_MatVecAdd"), names(ds));
        assertSame(acc, ds.get(0).bufferArg(4));
    }

    @Test
    public void shardedMatvecAddStagesThenAxpbys() {
        open();
        GpuWeight w = q4kSharded();
        GpuBuffer in = floats(COLS);
        GpuBuffer acc = floats(2L * ROWS);
        List<FakeDispatch> ds = run(() -> ops.matvecAdd(w, in, 0, acc, ROWS, 1));

        assertEquals(Arrays.asList("Q4_K_MatVec", GpuOps.COPY2D, "Q4_K_MatVec", GpuOps.COPY2D,
                "Q4_K_MatVec", GpuOps.COPY2D, GpuOps.AXPBY), names(ds));
        GpuBuffer result = ds.get(1).bufferArg(7);
        assertNotSame(acc, result);
        FakeDispatch ax = ds.get(6);
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
        List<FakeDispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Arrays.asList("Q4_K_MatVec", GpuOps.AXPBY), names(ds));
        FakeDispatch ax = ds.get(1);
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
        List<FakeDispatch> ds = run(() -> ops.add(x, 0, y, 0, x, 0, 64));

        FakeDispatch ax = ds.get(0);
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
        List<FakeDispatch> ds = run(() -> ops.copy(x, 0, x, 64, 64));

        assertEquals(Arrays.asList(GpuOps.COPY2D), names(ds));
        assertArrayEquals(new int[]{1, 64, 64, 0, 64, 64}, ints(ds.get(0), 0, 6));
    }

    @Test
    public void interleavedColumnWindowsOfOneBufferAreDisjoint() {
        open();
        GpuBuffer x = floats(4L * 128);
        List<FakeDispatch> ds = run(() -> ops.copy2d(x, 0, 128, x, 64, 128, 4, 64));

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
        FakeDispatch first = backend.dispatches().get(0);
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
        // a transform the GPU Hadamard kernel limit refuses is a model the device cannot serve, not a bad argument
        assertThrows(GpuModelUnsupportedException.class, () -> wide.withHadamard(HadamardSpec.rotation(2048)));
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
