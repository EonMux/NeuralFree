package com.eonmux.neuralfree.demo.tools;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link Mt19937} against golden values generated with the reference algorithms themselves (g++ libstdc++):
 * the raw outputs of {@code std::mt19937(1)}, the shrinking-pool draws of perplexity.cpp's HellaSwag
 * selection ({@code uniform_int_distribution} downscaling) and the {@code (int)(scale * rng() * size)} picks
 * of the WinoGrande and multiple-choice selection. Task selection is only llama.cpp's when every draw
 * matches.
 */
public class Mt19937Test {

    @Test
    public void theRawStreamIsTheReferenceStream() {
        Mt19937 rng = new Mt19937(1);
        long[] expected = {1791095845L, 4282876139L, 3093770124L, 4005303368L, 491263L, 550290313L};
        for (long value : expected) {
            assertEquals(value, rng.nextUnsigned());
        }
    }

    @Test
    public void theBoundedDrawsAreTheReferenceDraws() {
        // perplexity.cpp's HellaSwag loop: one draw over the shrinking pool per selected task
        for (int total = 2; total <= 60; total++) {
            long[] expected;
            if (total == 10) {
                expected = new long[] {4, 8, 5, 6, 0};
            } else if (total == 100) {
                expected = new long[] {41, 98, 70, 90, 0};
            } else if (total == 10042) {
                expected = new long[] {4187, 10012, 7232, 9361, 1};
            } else {
                expected = null;
            }
            if (expected == null) {
                continue;
            }
            Mt19937 rng = new Mt19937(1);
            long pool = total;
            for (long value : expected) {
                assertEquals("pool " + pool, value, rng.nextBounded(pool));
                pool--;
            }
        }
    }

    @Test
    public void aPoolOfOneConsumesDrawsUntilTheRejectionBoundPasses() {
        Mt19937 rng = new Mt19937(1);
        assertEquals(0, rng.nextBounded(1));
        // the raw stream advanced past every rejected draw
        assertEquals(4282876139L, rng.nextUnsigned());
    }

    @Test
    public void theFloatScaledDrawsAreTheReferenceDraws() {
        // the draw index j of perplexity.cpp's (int)(scale * rng() * aux.size()), the reference debugged
        // against libstdc++: j, not the picked aux[j], is what the generator produces
        Mt19937 rng = new Mt19937(1);
        int[] sizes = {10, 9, 8, 7, 6, 5, 4};
        int[] expected = {4, 8, 5, 6, 0, 0, 1};
        for (int i = 0; i < sizes.length; i++) {
            assertEquals("size " + sizes[i], expected[i], rng.nextFloatScaled(sizes[i]));
        }
        Mt19937 big = new Mt19937(1);
        int[] bigSizes = {1767, 1766, 1765, 1764, 1763, 1762, 1761};
        int[] bigExpected = {736, 1761, 1271, 1645, 0, 225, 532};
        for (int i = 0; i < bigSizes.length; i++) {
            assertEquals("size " + bigSizes[i], bigExpected[i], big.nextFloatScaled(bigSizes[i]));
        }
    }

    @Test
    public void everyDrawStaysInsideItsBound() {
        Mt19937 rng = new Mt19937(42);
        for (int pool = 1; pool <= 50; pool++) {
            for (int i = 0; i < 100; i++) {
                long draw = rng.nextBounded(pool);
                if (draw < 0 || draw >= pool) {
                    throw new AssertionError("draw " + draw + " outside [0, " + pool + ")");
                }
            }
        }
    }

    @Test
    public void invalidBoundsAreRefused() {
        Mt19937 rng = new Mt19937(1);
        assertThrows(IllegalArgumentException.class, () -> rng.nextBounded(0));
        assertThrows(IllegalArgumentException.class, () -> rng.nextFloatScaled(0));
        assertThrows(UnsupportedOperationException.class, () -> rng.nextBounded(0x100000000L));
    }
}
