package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;

import java.util.Map;
import java.util.TreeMap;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Statistic-by-statistic comparison of two collected imatrix files. */
final class ImatrixParity {

    private ImatrixParity() {
    }

    /**
     * Asserts that {@code a} and {@code b} hold the same tensors with equal shapes and counts, and squared and plain
     * activation sums within {@code relTol} of the larger magnitude of the tensor's sums.
     */
    static void assertSameStatistics(String what, ImatrixFile.Read a, ImatrixFile.Read b, float relTol) {
        Map<String, ImatrixFile.Entry> left = byName(a);
        Map<String, ImatrixFile.Entry> right = byName(b);
        assertEquals(what + ": the collected tensors", left.keySet(), right.keySet());
        for (ImatrixFile.Entry x : left.values()) {
            ImatrixFile.Entry y = right.get(x.tensor());
            String t = what + ": " + x.tensor();
            assertEquals(t + " matrices", x.nMat(), y.nMat());
            assertEquals(t + " columns", x.cols(), y.cols());
            assertArrayEquals(t + " counts", x.counts(), y.counts());
            assertClose(t + " in_sum2", x.sumSq(), y.sumSq(), relTol);
            assertTrue(t + " carries in_sum", x.hasSums() && y.hasSums());
            assertClose(t + " in_sum", x.sums(), y.sums(), relTol);
        }
    }

    private static void assertClose(String what, float[] x, float[] y, float relTol) {
        assertNotNull(what, x);
        assertEquals(what + " length", x.length, y.length);
        float scale = 0f;
        for (int i = 0; i < x.length; i++) {
            scale = Math.max(scale, Math.max(Math.abs(x[i]), Math.abs(y[i])));
        }
        float tol = Math.max(scale * relTol, 1e-6f);
        for (int i = 0; i < x.length; i++) {
            assertEquals(what + "[" + i + "]", x[i], y[i], tol);
        }
    }

    private static Map<String, ImatrixFile.Entry> byName(ImatrixFile.Read read) {
        Map<String, ImatrixFile.Entry> out = new TreeMap<>();
        for (ImatrixFile.Entry e : read.entries()) {
            out.put(e.tensor(), e);
        }
        return out;
    }
}
