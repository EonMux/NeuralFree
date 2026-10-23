package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class WorkSizesTest {

    @Test
    public void groupCountsRoundUp() {
        assertArrayEquals(new long[] {4}, WorkSizes.groupCounts(new long[] {1000}, new long[] {256}));
        assertArrayEquals(new long[] {4}, WorkSizes.groupCounts(new long[] {1024}, new long[] {256}));
        assertArrayEquals(new long[] {1}, WorkSizes.groupCounts(new long[] {1}, new long[] {256}));
        assertArrayEquals(new long[] {3, 2, 1},
                WorkSizes.groupCounts(new long[] {48, 17, 4}, new long[] {16, 16, 4}));
    }

    @Test
    public void groupCountsRejectInvalidSizes() {
        expectIllegal(() -> WorkSizes.groupCounts(new long[] {256}, new long[] {0}));
        expectIllegal(() -> WorkSizes.groupCounts(new long[] {0}, new long[] {64}));
        expectIllegal(() -> WorkSizes.groupCounts(new long[] {256, 2}, new long[] {64}));
        expectIllegal(() -> WorkSizes.groupCounts(new long[] {256}, null));
        expectIllegal(() -> WorkSizes.groupCounts(new long[] {1, 1, 1, 1}, new long[] {1, 1, 1, 1}));
    }

    @Test
    public void defaultLocalDividesGlobal() {
        long[] globals = {1, 255, 256, 1000, 4096};
        long[] expected = {1, 1, 256, 8, 256};
        for (int i = 0; i < globals.length; i++) {
            long[] local = WorkSizes.defaultLocal(new long[] {globals[i]}, 1024);
            assertEquals("global " + globals[i], expected[i], local[0]);
            assertEquals("global " + globals[i] + " must be divisible", 0, globals[i] % local[0]);
            assertTrue(Long.bitCount(local[0]) == 1);
        }
    }

    @Test
    public void defaultLocalRespectsDeviceLimit() {
        assertArrayEquals(new long[] {64}, WorkSizes.defaultLocal(new long[] {4096}, 64));
        assertArrayEquals(new long[] {64}, WorkSizes.defaultLocal(new long[] {4096}, 100));
        assertArrayEquals(new long[] {1}, WorkSizes.defaultLocal(new long[] {4096}, 1));
        assertArrayEquals(new long[] {128, 1, 1}, WorkSizes.defaultLocal(new long[] {384, 7, 3}, 1024));
        expectIllegal(() -> WorkSizes.defaultLocal(new long[] {256}, 0));
    }

    @Test
    public void pad3FillsTrailingDimensions() {
        assertArrayEquals(new long[] {5, 1, 1}, WorkSizes.pad3(new long[] {5}, 1));
        assertArrayEquals(new long[] {5, 6, 0}, WorkSizes.pad3(new long[] {5, 6}, 0));
        assertArrayEquals(new long[] {5, 6, 7}, WorkSizes.pad3(new long[] {5, 6, 7}, 1));
        expectIllegal(() -> WorkSizes.pad3(new long[0], 1));
        expectIllegal(() -> WorkSizes.pad3(null, 1));
    }

    @Test
    public void validateLocalChecksProductAndPositivity() {
        WorkSizes.validateLocal(new long[] {16, 16}, 256, "k");
        expectIllegal(() -> WorkSizes.validateLocal(new long[] {32, 16}, 256, "k"));
        expectIllegal(() -> WorkSizes.validateLocal(new long[] {0}, 256, "k"));
        try {
            WorkSizes.validateLocal(new long[] {512}, 256, "LLM_Test");
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("LLM_Test"));
        }
    }

    @Test
    public void productMultipliesAndDetectsOverflow() {
        assertEquals(24, WorkSizes.product(new long[] {2, 3, 4}));
        assertEquals(1, WorkSizes.product(new long[0]));
        try {
            WorkSizes.product(new long[] {Long.MAX_VALUE, 2});
            fail("expected ArithmeticException");
        } catch (ArithmeticException expected) {
            // overflow is reported, not wrapped
        }
    }

    private static void expectIllegal(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // contract violation is rejected
        }
    }
}
