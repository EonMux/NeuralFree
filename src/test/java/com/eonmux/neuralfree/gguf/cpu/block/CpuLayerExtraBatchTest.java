package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.memory.StepLayout;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batched-hook contract of {@link CpuLayerExtra}: an extra that declares its hooks layout-independent
 * ({@code batchRefusal()} null) runs them over the whole step's rows on every layout — contiguous, paged
 * single-segment and multi-segment alike — while a refusing extra throws its refusal for a non-contiguous step and
 * still runs the contiguous delegation.
 */
public class CpuLayerExtraBatchTest {

    /** Records the (pos0, n) pair of every hook run. */
    private static final class Recording implements CpuLayerExtra {
        int calls;
        int lastPos0;
        int lastN;
        final boolean batchable;

        private Recording(boolean batchable) {
            this.batchable = batchable;
        }

        @Override
        public void bind(com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext ctx,
                         com.eonmux.neuralfree.gguf.spec.LayerSpec layer) {
        }

        @Override
        public String batchRefusal() {
            return batchable ? null : Recording.class.getSimpleName() + " has no multi-sequence step";
        }

        @Override
        public void beforeBlock(int pos0, int n, float[] x) {
            calls++;
            lastPos0 = pos0;
            lastN = n;
        }

        @Override
        public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
            calls++;
            lastPos0 = pos0;
            lastN = n;
            return batchable;
        }
    }

    private static final int[] SEQ_ONE = {0};
    private static final int[] START_ZERO = {0};
    private static final int[] POS_ZERO = {0};
    private static final int[] NO_ROWS = {};

    @Test
    public void aBatchableExtraRunsTheWholeStepOnEveryLayout() {
        Recording extra = new Recording(true);
        StepLayout paged = StepLayout.of(8, SEQ_ONE, START_ZERO, new int[]{8}, POS_ZERO, NO_ROWS, true);
        extra.beforeBlock(paged, new float[8]);
        assertEquals("the batched hook runs the contiguous hook over the whole step", 8, extra.lastN);
        assertEquals(1, extra.calls);
        assertTrue(extra.afterMixer(paged, new float[8], new float[8]));
        assertEquals(2, extra.calls);

        StepLayout mixed = StepLayout.of(8, new int[]{0, 3}, new int[]{0, 5}, new int[]{5, 3},
                new int[]{0, 2}, NO_ROWS, true);
        extra.beforeBlock(mixed, new float[8]);
        assertEquals("a multi-segment step runs whole too (row-wise hooks)", 8, extra.lastN);
        extra.afterBlock(mixed, new float[8]);
        assertEquals(8, extra.lastN);
    }

    @Test
    public void aRefusingExtraThrowsOnNonContiguousStepsAndRunsTheContiguousOnes() {
        Recording extra = new Recording(false);
        StepLayout contiguous = StepLayout.of(8, SEQ_ONE, START_ZERO, new int[]{8}, POS_ZERO, NO_ROWS, false);
        float[] x = new float[8];
        extra.beforeBlock(contiguous, x);
        assertEquals("the contiguous path keeps the plain delegation", 8, extra.lastN);
        assertFalse(extra.afterMixer(contiguous, x, x));

        StepLayout paged = StepLayout.of(8, SEQ_ONE, START_ZERO, new int[]{8}, POS_ZERO, NO_ROWS, true);
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> extra.beforeBlock(paged, x));
        assertTrue(e.getMessage(), e.getMessage().contains("has no multi-sequence step"));
        assertThrows(UnsupportedOperationException.class, () -> extra.afterMixer(paged, x, x));
        assertThrows(UnsupportedOperationException.class, () -> extra.afterFfn(paged, x, x));
        assertThrows(UnsupportedOperationException.class, () -> extra.afterBlock(paged, x));
    }

    @Test
    public void theContiguousHookMathIsUntouchedByTheDelegation() {
        Recording extra = new Recording(true);
        float[] x = new float[8];
        extra.beforeBlock(StepLayout.of(8, SEQ_ONE, START_ZERO, new int[]{8}, POS_ZERO, NO_ROWS, false), x);
        assertArrayEquals(x, new float[8], 0f);
        assertEquals(0, extra.lastPos0);
    }
}
