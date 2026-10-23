package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Heap KV storage and the heap memory: rows of every KV type (Q8_0 bit-exact with ggml {@code quantize_row_q8_0_ref},
 * round half away from zero; BF16 to nearest even; the other block types through {@link KvBlockCodec}), per-head
 * reads across block boundaries, slice writes, the MLA latent alias, head-shape validation, and the heap memory's
 * step marking, ring rollback (including rollbacks after earlier drops), recurrent snapshot with reserved shadows
 * and context shift.
 */
public class CpuKvLayerTest {

    private static final int N_SWA = 8;
    private static final int UBATCH = 8;
    private static final KvWindow SWA = KvWindow.of(SwaType.STANDARD, N_SWA, true);

    private static float[] random(int n, long seed) {
        Random r = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) (r.nextGaussian() * 3.0);
        }
        return x;
    }

    private static CpuKvLayer layer(KvDType t, int nHeadKv, int headDim) {
        int width = nHeadKv * headDim;
        KvLayout l = KvLayout.of(4, false, width, width, t, t, KvWindow.FULL_CAUSAL);
        return CpuKvLayer.allocate(0, l, 0, nHeadKv, headDim, headDim);
    }

    private static float[] readRow(CpuKvLayer c, int row, boolean v) {
        int hd = v ? c.headDimV() : c.headDimK();
        float[] out = new float[c.nHeadKv() * hd];
        for (int h = 0; h < c.nHeadKv(); h++) {
            if (v) {
                c.readV(row, h, out, h * hd);
            } else {
                c.readK(row, h, out, h * hd);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ storage

    @Test
    public void f32RowsRoundTripExactly() {
        CpuKvLayer c = layer(KvDType.F32, 2, 32);
        float[] x = random(64, 1);
        c.writeK(3, x, 0);
        c.writeV(1, x, 0);
        assertArrayEquals(x, readRow(c, 3, false), 0f);
        assertArrayEquals(x, readRow(c, 1, true), 0f);
        assertArrayEquals(new float[64], readRow(c, 0, false), 0f);
    }

    @Test
    public void f16RowsRoundToNearestEvenHalves() {
        CpuKvLayer c = layer(KvDType.F16, 2, 32);
        float[] x = random(64, 2);
        x[5] = 65520f;
        c.writeK(2, x, 0);
        float[] back = readRow(c, 2, false);
        for (int i = 0; i < x.length; i++) {
            assertEquals(FpBlockDecoders.roundToFp16(x[i]), back[i], 0f);
        }
        assertTrue(Float.isInfinite(back[5]));
    }

    /** {@code d = amax / 127 = 1}: x * id lands exactly on .5 values, which round away from zero. */
    @Test
    public void q8RoundsHalfAwayFromZero() {
        CpuKvLayer c = layer(KvDType.Q8_0, 1, 32);
        float[] x = new float[32];
        x[0] = 127f;
        x[1] = 2.5f;
        x[2] = -2.5f;
        x[3] = 0.5f;
        x[4] = -0.5f;
        x[5] = -126.5f;
        c.writeK(0, x, 0);
        float[] back = readRow(c, 0, false);
        assertEquals(127f, back[0], 0f);
        assertEquals(3f, back[1], 0f);
        assertEquals(-3f, back[2], 0f);
        assertEquals(1f, back[3], 0f);
        assertEquals(-1f, back[4], 0f);
        assertEquals(-127f, back[5], 0f);
        assertEquals(0f, back[6], 0f);
    }

    @Test
    public void q8MatchesTheGgmlReferenceQuantizer() {
        CpuKvLayer c = layer(KvDType.Q8_0, 4, 32);
        float[] x = random(128, 3);
        c.writeK(1, x, 0);
        float[] back = readRow(c, 1, false);
        CpuVecDot.QuantizedActivations ref = CpuActivationQuantizer.quantizeQ8_0(x, 0, 128);
        for (int i = 0; i < 128; i++) {
            assertEquals(ref.quant(i) * ref.scale(i / 32), back[i], 0f);
        }
    }

    @Test
    public void headReadsCrossQ8BlockBoundaries() {
        CpuKvLayer c = layer(KvDType.Q8_0, 4, 24);
        float[] x = random(96, 4);
        c.writeK(0, x, 0);
        float[] full = readRow(c, 0, false);
        float[] head1 = new float[24];
        c.readK(0, 1, head1, 0);
        for (int i = 0; i < 24; i++) {
            assertEquals(full[24 + i], head1[i], 0f);
        }
        assertThrows(IndexOutOfBoundsException.class, () -> c.readK(0, 4, head1, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> c.readK(4, 0, head1, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> c.readK(0, 0, head1, 1));
    }

    @Test
    public void bf16RowsRoundToNearestEven() {
        CpuKvLayer c = layer(KvDType.BF16, 2, 32);
        float[] x = random(64, 6);
        c.writeV(3, x, 0);
        float[] back = readRow(c, 3, true);
        for (int i = 0; i < x.length; i++) {
            assertEquals(FpBlockDecoders.bf16ToFp32(KvBlockCodec.fp32ToBf16(x[i])), back[i], 0f);
        }
    }

    /** Every block type stores each 32-value block with its reference quantizer; heads of 24 cross blocks. */
    @Test
    public void blockTypesStoreThroughTheCodecAndReadAcrossBlocks() {
        for (KvDType t : KvDType.all()) {
            if (!t.quantized()) {
                continue;
            }
            CpuKvLayer c = layer(t, 4, 24);
            float[] x = random(96, 7 + t.code());
            c.writeK(2, x, 0);
            float[] expect = new float[96];
            byte[] b = new byte[t.blockBytes()];
            for (int blk = 0; blk < 3; blk++) {
                KvBlockCodec.quantize(t, x, blk * 32, b, 0);
                KvBlockCodec.dequantize(t, b, 0, 0, 32, expect, blk * 32);
            }
            assertArrayEquals(t.toString(), expect, readRow(c, 2, false), 0f);
            float[] head2 = new float[24];
            c.readK(2, 2, head2, 0);
            for (int i = 0; i < 24; i++) {
                assertEquals(t + " head 2 value " + i, expect[48 + i], head2[i], 0f);
            }
        }
    }

    /** A slice write replaces only its values: whole blocks for block types, any range for F32 / F16 / BF16. */
    @Test
    public void sliceWritesKeepTheRestOfTheRow() {
        for (KvDType t : KvDType.all()) {
            CpuKvLayer c = layer(t, 1, 96);
            float[] x = random(96, 20 + t.code());
            c.writeK(1, x, 0);
            float[] before = readRow(c, 1, false);
            float[] y = random(32, 40 + t.code());
            c.writeKSlice(1, 32, 32, y, 0);
            float[] after = readRow(c, 1, false);
            CpuKvLayer ref = layer(t, 1, 96);
            float[] merged = x.clone();
            System.arraycopy(y, 0, merged, 32, 32);
            ref.writeK(1, merged, 0);
            float[] expect = readRow(ref, 1, false);
            for (int i = 0; i < 96; i++) {
                if (i < 32 || i >= 64) {
                    assertEquals(t + " value " + i + " is outside the slice", before[i], after[i], 0f);
                } else {
                    assertEquals(t + " value " + i, expect[i], after[i], 0f);
                }
            }
            if (t.quantized()) {
                assertThrows(IllegalArgumentException.class, () -> c.writeKSlice(1, 16, 32, y, 0));
                assertThrows(IllegalArgumentException.class, () -> c.writeKSlice(1, 32, 16, y, 0));
            } else {
                c.writeKSlice(1, 5, 3, y, 0);
            }
            assertThrows(IndexOutOfBoundsException.class, () -> c.writeKSlice(1, 64, 64, new float[64], 0));
        }
    }

    @Test
    public void latentVReadsTheKRowPrefixAndRejectsWrites() {
        KvLayout l = KvLayout.latent(4, false, 64 + 32, 64, KvDType.F16, KvWindow.FULL_CAUSAL);
        CpuKvLayer c = CpuKvLayer.allocate(0, l, 0, 1, 96, 64);
        float[] x = random(96, 5);
        c.writeK(2, x, 0);
        float[] v = new float[64];
        c.readV(2, 0, v, 0);
        for (int i = 0; i < 64; i++) {
            assertEquals(FpBlockDecoders.roundToFp16(x[i]), v[i], 0f);
        }
        assertThrows(UnsupportedOperationException.class, () -> c.writeV(2, x, 0));
        assertThrows(IllegalArgumentException.class, () -> CpuKvLayer.allocate(0, l, 0, 2, 48, 32));
    }

    @Test
    public void headShapeMustFillTheRow() {
        KvLayout l = KvLayout.of(4, false, 128, 64, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        CpuKvLayer ok = CpuKvLayer.allocate(0, l, 0, 2, 64, 32);
        assertEquals(64, ok.headDimK());
        assertEquals(32, ok.headDimV());
        assertThrows(IllegalArgumentException.class, () -> CpuKvLayer.allocate(0, l, 0, 2, 64, 64));
        assertThrows(IllegalArgumentException.class, () -> CpuKvLayer.allocate(0, l, 0, 0, 64, 32));
    }

    // ------------------------------------------------------------------ heap memory

    private static HeapModelMemory.KvHeads twoHeads() {
        return (il, layout, cross) -> new int[]{2, layout.widthK() / 2, layout.widthV() / 2};
    }

    private static MemoryPlan ringPlan(int rows) {
        KvLayout ring = KvLayout.of(rows, true, 8, 8, KvDType.F32, KvDType.F32, SWA);
        return MemoryPlan.builder(MemoryKind.KV, 64, UBATCH)
                .group(MemoryPlan.KvGroup.of(0, rows, true, SWA))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(ring, 0).build())
                .build();
    }

    private static void step(HeapModelMemory mem, int pos0, int n) {
        mem.beginStep(pos0, n);
        int[] tokens = new int[n];
        for (int t = 0; t < n; t++) {
            tokens[t] = 100 + pos0 + t;
            float[] row = new float[8];
            Arrays.fill(row, pos0 + t);
            int r = mem.plan().group(0).rowOf(pos0 + t);
            mem.kv(0).writeK(r, row, 0);
            mem.kv(0).writeV(r, row, 0);
        }
        mem.sequence().append(tokens, 0, n);
    }

    /** Same trace as the device test: a 5-token draft wrapping a ring of nSwa + ubatch rows, then rollback(3). */
    @Test
    public void heapRollbackOnARingEmptiesTheDraftRows() {
        try (HeapModelMemory mem = HeapModelMemory.allocate(ringPlan(N_SWA + UBATCH), twoHeads(), null)) {
            step(mem, 0, 8);
            step(mem, 8, 6);
            step(mem, 14, 5);
            assertTrue(mem.sequence().wrapped());
            assertTrue(mem.canRollback(3));
            assertFalse(mem.canRollback(10));
            mem.rollback(3);
            for (int r = 0; r < 3; r++) {
                assertEquals(-1, mem.positionOfRow(0, r));
            }
            for (int r = 3; r < 16; r++) {
                assertEquals(r, mem.positionOfRow(0, r));
            }
            assertEquals(16, mem.sequence().nPast());
        }
    }

    @Test
    public void heapContextShiftRotatesRingRows() {
        try (HeapModelMemory mem = HeapModelMemory.allocate(ringPlan(16), twoHeads(), null)) {
            step(mem, 0, 8);
            step(mem, 8, 8);
            step(mem, 16, 4);
            int[][] delta = mem.shiftPositions(2, 5);
            float[] k = new float[4];
            for (int r = 0; r < 16; r++) {
                if (r >= 2 && r <= 14) {
                    assertEquals(r, mem.positionOfRow(0, r));
                    mem.kv(0).readK(r, 1, k, 0);
                    assertEquals(r + 5f, k[0], 0f);
                    assertEquals(-5, delta[0][r]);
                } else {
                    assertEquals(-1, mem.positionOfRow(0, r));
                    assertEquals(0, delta[0][r]);
                }
            }
            assertEquals(15, mem.sequence().nPast());
        }
    }

    /** The next query (at 10) would need kept position 3, which the ring overwrote with 19. */
    @Test
    public void heapContextShiftRefusesToLoseKeptPositionsTheNextWindowNeeds() {
        try (HeapModelMemory mem = HeapModelMemory.allocate(ringPlan(16), twoHeads(), null)) {
            step(mem, 0, 8);
            step(mem, 8, 8);
            step(mem, 16, 4);
            assertThrows(UnsupportedOperationException.class, () -> mem.shiftPositions(4, 10));
            assertEquals(20, mem.sequence().nPast());
            assertEquals(19, mem.positionOfRow(0, 3));
            assertThrows(IllegalStateException.class, mem::shiftDelta);
        }
    }

    @Test
    public void heapSnapshotRestoresRecurrentStateAndPositions() {
        KvLayout full = KvLayout.of(32, false, 8, 8, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.HYBRID, 32, UBATCH)
                .config(MemoryConfig.builder().stateSnapshots(true).build())
                .group(MemoryPlan.KvGroup.of(0, 32, false, KvWindow.FULL_CAUSAL))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(full, 0).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.RECURRENT)
                        .state(LayerMemoryPlan.StateTensor.of(LayerMemoryPlan.STATE_SCAN,
                                LayerMemoryPlan.StateLayout.DELTA, 16))
                        .build())
                .build();
        try (HeapModelMemory mem = HeapModelMemory.allocate(plan, twoHeads(), null)) {
            float[] s = mem.recurrent(1).tensor(LayerMemoryPlan.STATE_SCAN);
            step(mem, 0, 5);
            s[3] = 2f;
            mem.snapshot();
            step(mem, 5, 3);
            s[3] = 9f;
            assertFalse(mem.canRollback(3));
            mem.restore();
            assertEquals(2f, s[3], 0f);
            assertEquals(5, mem.sequence().nPast());
            assertEquals(4, mem.positionOfRow(0, 4));
            assertEquals(-1, mem.positionOfRow(0, 5));
            assertThrows(UnsupportedOperationException.class, () -> mem.shiftPositions(1, 1));
            mem.reset();
            assertEquals(0f, s[3], 0f);
            assertEquals(-1, mem.positionOfRow(0, 0));
            assertThrows(IllegalStateException.class, mem::restore);
        }
    }

    /** Same trace as the device test: the second of two 9-position rollbacks would expose emptied rows 1..7. */
    @Test
    public void heapRollbackAfterAnEarlierDropIsRefusedWhenItsWindowWasLost() {
        try (HeapModelMemory mem = HeapModelMemory.allocate(ringPlan(16), twoHeads(), null)) {
            for (int pos = 0; pos < 40; pos += UBATCH) {
                step(mem, pos, UBATCH);
            }
            assertTrue(mem.ringsKeepWindow(10, 9));
            assertFalse(mem.ringsKeepWindow(12, 11));
            mem.rollback(9);
            step(mem, 31, 2);
            assertTrue(mem.canRollback(1));
            assertFalse(mem.canRollback(9));
            assertThrows(IllegalStateException.class, () -> mem.rollback(9));
            assertEquals(33, mem.sequence().nPast());
        }
    }

    @Test
    public void heapSnapshotNeedsReservedShadows() {
        KvLayout full = KvLayout.of(32, false, 8, 8, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.HYBRID, 32, UBATCH)
                .config(MemoryConfig.defaults())
                .group(MemoryPlan.KvGroup.of(0, 32, false, KvWindow.FULL_CAUSAL))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(full, 0).build())
                .layer(LayerMemoryPlan.builder(1, LayerMemoryPlan.Kind.RECURRENT)
                        .state(LayerMemoryPlan.StateTensor.of(LayerMemoryPlan.STATE_SCAN,
                                LayerMemoryPlan.StateLayout.DELTA, 16))
                        .build())
                .build();
        assertEquals(0L, plan.snapshotBytes());
        try (HeapModelMemory mem = HeapModelMemory.allocate(plan, twoHeads(), null)) {
            assertFalse(mem.canSnapshot());
            assertThrows(UnsupportedOperationException.class, mem::snapshot);
        }
    }

    @Test
    public void beginStepValidatesBeforeMarking() {
        try (HeapModelMemory mem = HeapModelMemory.allocate(ringPlan(16), twoHeads(), null)) {
            assertThrows(IllegalArgumentException.class, () -> mem.beginStep(0, UBATCH + 1));
            assertThrows(IllegalArgumentException.class, () -> mem.beginStep(-1, 1));
            assertEquals(-1, mem.positionOfRow(0, 0));
        }
    }
}
