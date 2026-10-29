package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ImatrixCollector} on direct observer calls: the dense accumulation (one matrix, per-token rows summed
 * column-wise, one count per activation row), the per-expert matrices of a 3-D expert tensor over its dispatches,
 * the namer refusals and the reset. An engine-level run over a synthetic llama model covers the collection the
 * running matmuls produce.
 */
public class ImatrixCollectorTest {

    /** A F32 weight view of {@code rows} rows of {@code cols} values over {@code data}. */
    private static CpuTensorRef weight(float[] data, int rows, int cols) {
        ByteBuffer buf = ByteBuffer.allocate(data.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().put(data);
        return CpuTensorRef.of(GGMLType.F32, buf, rows, cols);
    }

    @Test
    public void denseRowsSumColumnWiseWithOneCountPerRow() {
        ImatrixCollector c = new ImatrixCollector(w -> "blk.0.attn_q.weight");
        CpuTensorRef w = weight(new float[8 * 4], 8, 4);
        // One dispatch of two activation rows, then one of a single row: sum2 over all three, count 3.
        c.observe(w, 0L, 8, 0, new float[]{1, 2, 3, 4, -1, -2, -3, -4}, 0, 4, 4, 2);
        c.observe(w, 0L, 8, 0, new float[]{0.5f, 0.5f, 0.5f, 0.5f}, 0, 4, 4, 1);
        List<ImatrixFile.Entry> entries = c.entries();
        assertEquals(1, entries.size());
        ImatrixFile.Entry e = entries.get(0);
        assertEquals("blk.0.attn_q.weight", e.tensor());
        assertEquals(1, e.nMat());
        assertEquals(4, e.cols());
        assertSumSq(e, 0, new float[]{1 + 1 + 0.25f, 4 + 4 + 0.25f, 9 + 9 + 0.25f, 16 + 16 + 0.25f});
        assertEquals(3L, e.count(0));
        assertEquals(3L, c.count("blk.0.attn_q.weight", 0));
        assertEquals(0L, c.count("blk.0.attn_q.weight", 1));
    }

    @Test
    public void expertDispatchesFillOneMatrixPerExpert() {
        ImatrixCollector c = new ImatrixCollector(w -> "blk.0.ffn_gate_exps.weight");
        CpuTensorRef w = weight(new float[16 * 4], 16, 4); // 4 experts of 4 rows
        // matvecExpert passes one expert per call: rowStart = expert * perExpert, rows = perExpert.
        c.observe(w, 4L, 4, 4, new float[]{2, 0, 0, 0}, 0, 4, 4, 1);
        c.observe(w, 12L, 4, 4, new float[]{0, 3, 0, 0}, 0, 4, 4, 1);
        List<ImatrixFile.Entry> entries = c.entries();
        assertEquals(1, entries.size());
        ImatrixFile.Entry e = entries.get(0);
        assertEquals(4, e.nMat());
        assertSumSq(e, 1, new float[]{4, 0, 0, 0});
        assertSumSq(e, 3, new float[]{0, 9, 0, 0});
        assertEquals(1L, e.count(1));
        assertEquals(1L, e.count(3));
        assertEquals(0L, e.count(0));
    }

    @Test
    public void anUnnamedWeightFailsTheCollectionAndResetClears() {
        ImatrixCollector c = new ImatrixCollector(ref -> null);
        CpuTensorRef w = weight(new float[4], 1, 4);
        assertThrows(IllegalStateException.class,
                () -> c.observe(w, 0L, 1, 0, new float[]{1, 2, 3, 4}, 0, 4, 4, 1));
        assertTrue(c.entries().isEmpty());

        ImatrixCollector named = new ImatrixCollector(ref -> "blk.1.ffn_up.weight");
        CpuTensorRef w2 = weight(new float[8], 2, 4);
        named.observe(w2, 0L, 2, 0, new float[]{1, 1, 1, 1}, 0, 4, 4, 1);
        assertEquals(1, named.tensorCount());
        named.reset();
        assertEquals(0, named.tensorCount());
        assertTrue(named.entries().isEmpty());

        // Two weights naming one entry are refused at the second: a shared statistic cannot be quantized against.
        AtomicInteger n = new AtomicInteger();
        ImatrixCollector clashing = new ImatrixCollector(ref -> n.incrementAndGet() == 1 ? "t" : "t");
        CpuTensorRef a = weight(new float[4], 1, 4);
        CpuTensorRef b = weight(new float[8], 2, 4);
        clashing.observe(a, 0L, 1, 0, new float[]{1, 1, 1, 1}, 0, 4, 4, 1);
        assertThrows(IllegalStateException.class,
                () -> clashing.observe(b, 0L, 2, 0, new float[]{1, 1, 1, 1}, 0, 4, 4, 1));

        // An expert index outside the reported matrices is refused: the dispatch contradicts the tensor.
        ImatrixCollector bounded = new ImatrixCollector(ref -> "blk.0.ffn_down_exps.weight");
        CpuTensorRef w3 = weight(new float[16 * 4], 16, 4);
        assertThrows(IllegalStateException.class,
                () -> bounded.observe(w3, 16L, 4, 4, new float[]{1, 1, 1, 1}, 0, 4, 4, 1));
    }

    /** {@code entry}'s matrix {@code m} against {@code expected}, entry-major order. */
    private static void assertSumSq(ImatrixFile.Entry e, int m, float[] expected) {
        float[] all = e.sumSq();
        int cols = e.cols();
        for (int j = 0; j < cols; j++) {
            assertEquals("matrix " + m + " column " + j, expected[j], all[m * cols + j], 1e-5f);
        }
    }
}
