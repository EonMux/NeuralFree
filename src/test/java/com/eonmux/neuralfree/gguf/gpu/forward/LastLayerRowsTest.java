package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.engine.lora.AdapterRows;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The row sets {@link LastLayerRows#compactable} accepts: strictly ascending step rows, fewer than the step's, whose
 * move to the front takes at most {@link LastLayerRows#MAX_RUNS} copies, none of which overlaps its own destination.
 */
public class LastLayerRowsTest {

    @Test
    public void acceptsRowSetsThatMoveForwardInDisjointRuns() {
        assertTrue("nothing read", LastLayerRows.compactable(4, new int[0]));
        assertTrue("the last row", LastLayerRows.compactable(16, new int[]{15}));
        assertTrue("a prefix stays in place", LastLayerRows.compactable(8, new int[]{0, 1, 2}));
        assertTrue("a prefix and a far run", LastLayerRows.compactable(8, new int[]{0, 1, 5, 6}));
        assertTrue("a run whose source starts at its destination's end", LastLayerRows.compactable(8,
                new int[]{2, 3}));
        assertTrue(LastLayerRows.MAX_RUNS + " single-row runs",
                LastLayerRows.compactable(8, new int[]{1, 3, 5, 7}));
    }

    @Test
    public void refusesARunThatOverlapsItsDestination() {
        assertFalse("rows 1..2 onto 0..1", LastLayerRows.compactable(4, new int[]{1, 2}));
        assertFalse("rows 2..3 onto 1..2 after the kept row 0", LastLayerRows.compactable(8, new int[]{0, 2, 3}));
        assertFalse("rows 3..5 onto 1..3", LastLayerRows.compactable(8, new int[]{0, 3, 4, 5}));
    }

    @Test
    public void refusesMoreThanMaxRunsCopies() {
        assertFalse(LastLayerRows.MAX_RUNS + 1 + " single-row runs",
                LastLayerRows.compactable(12, new int[]{1, 3, 5, 7, 9}));
        assertTrue("a kept prefix takes no copy", LastLayerRows.compactable(12, new int[]{0, 3, 5, 7, 9}));
    }

    @Test
    public void refusesSetsThatAreNotAscendingStepRowsOrCoverTheStep() {
        assertFalse("no row set", LastLayerRows.compactable(4, null));
        assertFalse("every row", LastLayerRows.compactable(3, new int[]{0, 1, 2}));
        assertFalse("descending", LastLayerRows.compactable(8, new int[]{5, 3}));
        assertFalse("a repeated row", LastLayerRows.compactable(8, new int[]{3, 3}));
        assertFalse("a negative row", LastLayerRows.compactable(8, new int[]{-1}));
        assertFalse("a row past the step", LastLayerRows.compactable(8, new int[]{8}));
    }

    @Test
    public void aMixedAdapterTableFollowsTheCompactedRows() {
        AdapterRows table = new AdapterRows();
        table.set(new int[]{0, 0, 2, 2, 1, 1});
        LastLayerRows.selectAdapterRows(table, new int[]{1, 3, 5});
        assertEquals(3, table.rows());
        assertEquals(0, table.slot(0));
        assertEquals(2, table.slot(1));
        assertEquals(1, table.slot(2));

        AdapterRows uniform = new AdapterRows();
        uniform.set(new int[]{3, 3, 3});
        LastLayerRows.selectAdapterRows(uniform, new int[]{2});
        assertEquals("a uniform table serves any rows", 3, uniform.uniform());
    }
}
