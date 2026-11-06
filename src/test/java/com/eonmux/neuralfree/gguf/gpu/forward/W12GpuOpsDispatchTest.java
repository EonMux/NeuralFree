package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.W12Codec;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
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

/**
 * Dispatch traces of the W12 projections of {@link GpuOps} against the host-only fake backend with strict LLM
 * argument rules: the row kernels decode the per-row records in registers for up to
 * {@link GpuOps#W12_MATVEC_MAX_TOKENS} tokens, a wider prefill decodes the weight into its BF16 mirror first
 * ({@code W12_DecodeBF16}), an open {@link GpuOps.W12QView} scope routes the projections through the
 * {@code W12Q_*} row set, the fused gate+up form streams both weights through {@code W12_MatVecFusedGateUp},
 * and the row sets are prepared at bind time.
 */
public class W12GpuOpsDispatchTest {

    private static final int COLS = 256;
    private static final int ROWS = 96;
    private static final int SHARD_ROWS = 32;
    private static final int TOKENS = 96;
    private static final long W12_ROW_BYTES = W12Codec.rowBytes(COLS);
    private static final int UBATCH = 16;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int local;

    private void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        local = ops.local();
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

    private GpuWeight w12Single() {
        return GpuWeight.of("blk.0.attn_q.weight", GGMLType.W12, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * W12_ROW_BYTES, true));
    }

    private GpuWeight w12Sharded() {
        List<GpuBuffer> shards = new ArrayList<>();
        for (int i = 0; i < ROWS / SHARD_ROWS; i++) {
            shards.add(backend.allocateBuffer(SHARD_ROWS * W12_ROW_BYTES, true));
        }
        return GpuWeight.sharded("blk.0.ffn_up.weight", GGMLType.W12, new long[]{COLS, ROWS}, shards,
                new long[]{SHARD_ROWS, SHARD_ROWS, SHARD_ROWS});
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

    // ------------------------------------------------------------------ the row kernels

    @Test
    public void theTypeRegistersItsRowFamily() {
        assertTrue(GpuOps.supportsWeightType(GGMLType.W12));
        assertEquals("W12_MatVec", GpuOps.matVecKernel(GGMLType.W12));
    }

    @Test
    public void matvecOneTokenDecodesTheRowsInRegisters() {
        open();
        GpuWeight w = w12Single();
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Collections.singletonList("W12_MatVec"), names(ds));
        Dispatch mv = ds.get(0);
        assertArrayEquals(new long[]{(long) ROWS * local}, mv.global());
        assertArrayEquals(new long[]{local}, mv.local());
        assertArrayEquals(new int[]{ROWS, COLS}, ints(mv, 0, 2));
        assertSame(w.shard(0), mv.bufferArg(2));
        assertSame(in, mv.bufferArg(3));
        assertSame(out, mv.bufferArg(4));
    }

    @Test
    public void matvecOneTokenScattersEachShardAtItsRowOffset() {
        open();
        GpuWeight w = w12Sharded();
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 1));

        assertEquals(Arrays.asList("W12_MatVec", GpuOps.COPY2D, "W12_MatVec", GpuOps.COPY2D,
                "W12_MatVec", GpuOps.COPY2D), names(ds));
        for (int s = 0; s < 3; s++) {
            Dispatch cp = ds.get(2 * s + 1);
            assertArrayEquals(new int[]{1, SHARD_ROWS, SHARD_ROWS, 0, ROWS, s * SHARD_ROWS}, ints(cp, 0, 6));
        }
    }

    @Test
    public void batchTokensUseTheTwoDimensionalBatch() {
        open();
        GpuWeight w = w12Single();
        GpuBuffer in = floats(4L * COLS);
        GpuBuffer out = floats(4L * ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, 4));

        assertEquals(Collections.singletonList("W12_BatchMatMul_2D"), names(ds));
        Dispatch mm = ds.get(0);
        assertArrayEquals(new long[]{(long) ROWS * local, 4}, mm.global());
        assertArrayEquals(new long[]{local, 1}, mm.local());
        assertArrayEquals(new int[]{ROWS, COLS, 4}, ints(mm, 0, 3));
        assertSame(w.shard(0), mm.bufferArg(3));
        assertSame(in, mm.bufferArg(4));
        assertSame(out, mm.bufferArg(5));
    }

    @Test
    public void aWidePrefillDecodesTheWeightIntoItsBf16Mirror() {
        open();
        GpuWeight w = w12Single();
        GpuBuffer in = floats((long) TOKENS * COLS);
        GpuBuffer out = floats((long) TOKENS * ROWS);
        List<Dispatch> ds = run(() -> ops.matvec(w, in, 0, out, 0, TOKENS));

        Dispatch decode = ds.get(0);
        assertEquals("W12_DecodeBF16", decode.kernelName());
        assertArrayEquals(new long[]{(long) ROWS * local}, decode.global());
        assertArrayEquals(new long[]{local}, decode.local());
        assertArrayEquals(new int[]{ROWS, COLS, 0}, ints(decode, 0, 3));
        assertSame(w.shard(0), decode.bufferArg(3));
        assertNotSame(in, decode.bufferArg(4));
        assertNotSame(out, decode.bufferArg(4));
        for (Dispatch d : ds) {
            assertFalse(d.kernelName(), d.kernelName().startsWith("W12_MatVec"));
            assertFalse(d.kernelName(), d.kernelName().startsWith("W12_Batch"));
            assertFalse(d.kernelName(), d.kernelName().startsWith("W12Q_"));
        }
        assertTrue("the mirror's BF16 projection follows the decode", ds.size() > 1);
    }

    // ------------------------------------------------------------------ the Q view

    @Test
    public void anOpenQViewScopeRoutesTheProjectionsThroughTheQRowSet() {
        open();
        GpuWeight w = w12Single();
        GpuBuffer in = floats(4L * COLS);
        GpuBuffer out = floats(4L * ROWS);
        GpuBuffer accum = floats(4L * ROWS);
        try (GpuOps.W12QView scope = ops.openW12QView()) {
            assertEquals(Collections.singletonList("W12Q_MatVec"),
                    names(run(() -> ops.matvec(w, in, 0, out, 0, 1))));
            assertEquals(Collections.singletonList("W12Q_MatVecAdd"),
                    names(run(() -> ops.matvecAdd(w, in, 0, accum, 0, 1))));
            assertEquals(Collections.singletonList("W12Q_BatchMatMul_2D"),
                    names(run(() -> ops.matvec(w, in, 0, out, 0, 4))));
            IllegalStateException e = assertThrows(IllegalStateException.class, ops::openW12QView);
            assertTrue(e.getMessage(), e.getMessage().contains("already open"));
        }
        // closed scope restores the plain row set
        assertEquals(Collections.singletonList("W12_MatVec"), names(run(() -> ops.matvec(w, in, 0, out, 0, 1))));
    }

    @Test
    public void closingTheQViewScopeTwiceIsIdempotent() {
        open();
        GpuOps.W12QView scope = ops.openW12QView();
        scope.close();
        scope.close();
        GpuWeight w = w12Single();
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        assertEquals(Collections.singletonList("W12_MatVec"), names(run(() -> ops.matvec(w, in, 0, out, 0, 1))));
    }

    @Test
    public void anOpenQViewScopeRefusesTheFusedGateUp() {
        open();
        GpuWeight gate = GpuWeight.of("blk.0.ffn_gate.weight", GGMLType.W12, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * W12_ROW_BYTES, true));
        GpuWeight up = GpuWeight.of("blk.0.ffn_up.weight", GGMLType.W12, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * W12_ROW_BYTES, true));
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        try (GpuOps.W12QView scope = ops.openW12QView()) {
            assertFalse(ops.gateUpGluUsable(gate, up, GluOp.SWIGLU, ActParams.NONE));
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, out));
            assertTrue(e.getMessage(), e.getMessage().contains("fused gate+up"));
        }
    }

    @Test
    public void theFusedGateUpServesW12OutsideTheQViewScope() {
        open();
        GpuWeight gate = GpuWeight.of("blk.0.ffn_gate.weight", GGMLType.W12, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * W12_ROW_BYTES, true));
        GpuWeight up = GpuWeight.of("blk.0.ffn_up.weight", GGMLType.W12, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * W12_ROW_BYTES, true));
        GpuBuffer in = floats(COLS);
        GpuBuffer out = floats(ROWS);
        assertTrue(ops.gateUpGluUsable(gate, up, GluOp.SWIGLU, ActParams.NONE));
        assertEquals(Collections.singletonList("W12_MatVecFusedGateUp"),
                names(run(() -> ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, out))));
    }

    // ------------------------------------------------------------------ preparation

    @Test
    public void prewarmWeightKernelsPreparesBothW12RowSetsAndTheDecode() {
        open();
        ops.prewarmWeightKernels(GGMLType.W12, true, false);
        List<String> prepared = backend.preparedKernels();
        for (String kernel : new String[]{"W12_MatVec", "W12_MatVecAdd", "W12_BatchMatMul", "W12_BatchMatMul_2D",
                "W12_DecodeBF16", "W12Q_MatVec", "W12Q_MatVecAdd", "W12Q_BatchMatMul", "W12Q_BatchMatMul_2D"}) {
            assertTrue(kernel + " not in " + prepared, prepared.contains(kernel));
        }
        assertTrue(prepared.toString(), prepared.stream().noneMatch(n -> n.startsWith("W12")
                && n.contains("Embedding")));
    }
}
