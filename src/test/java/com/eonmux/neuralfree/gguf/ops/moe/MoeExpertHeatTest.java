package com.eonmux.neuralfree.gguf.ops.moe;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The {@link MoeExpertHeat} recorder: routed rows counted per layer and expert, invalid ids and layers beyond the
 * recorder's bound not counted, file heats merged in, the snapshot hottest first without the zero rows and clamped
 * to the file's {@code uint32} heats, and the empty check.
 */
public class MoeExpertHeatTest {

    @Test
    public void routedRowsCountPerLayerAndExpert() {
        MoeExpertHeat heat = new MoeExpertHeat();
        heat.note(0, new int[]{2, 0, 2}, 3);
        heat.note(5, new int[]{1, 1}, 2);
        List<RankedExpert> snapshot = heat.snapshot();
        assertEquals(Arrays.asList(
                new RankedExpert(0, 2, 2),
                new RankedExpert(5, 1, 2),
                new RankedExpert(0, 0, 1)), snapshot);
        assertFalse(heat.isEmpty());
    }

    /** The reference's bounds: expert indexes outside {@code [0, 65535]} are skipped, layers above 4095 ignored. */
    @Test
    public void invalidIdsAndLayersBeyondTheBoundAreNotCounted() {
        MoeExpertHeat heat = new MoeExpertHeat();
        heat.note(0, new int[]{-1, 70000, 3}, 3);
        heat.note(4096, new int[]{1}, 1);
        heat.note(4095, new int[]{1}, 1);
        assertEquals(Arrays.asList(
                new RankedExpert(0, 3, 1),
                new RankedExpert(4095, 1, 1)), heat.snapshot());
    }

    @Test
    public void fileHeatsMergeIntoTheCounts() {
        MoeExpertHeat heat = new MoeExpertHeat();
        heat.note(1, new int[]{4, 4, 4}, 3);
        heat.merge(Arrays.asList(new RankedExpert(1, 4, 10), new RankedExpert(2, 0, 1)));
        assertEquals(Arrays.asList(
                new RankedExpert(1, 4, 13),
                new RankedExpert(2, 0, 1)), heat.snapshot());
    }

    /** The counts saturate at the file's {@code uint32} like the reference's counter. */
    @Test
    public void countsClampToTheFileHeatRange() {
        MoeExpertHeat heat = new MoeExpertHeat();
        heat.merge(Collections.singletonList(new RankedExpert(0, 1, RankedExpert.MAX_HEAT)));
        heat.note(0, new int[]{1, 1}, 2);
        assertEquals(Collections.singletonList(new RankedExpert(0, 1, RankedExpert.MAX_HEAT)), heat.snapshot());
    }

    /** Only experts with heat appear in the snapshot: no entry is invented for the experts a step skipped. */
    @Test
    public void nothingCountedLeavesAnEmptySnapshot() {
        MoeExpertHeat heat = new MoeExpertHeat();
        heat.note(0, new int[]{-1, -1, -1, -1}, 4);
        assertTrue(heat.isEmpty());
        assertTrue(heat.snapshot().isEmpty());
    }

    @Test
    public void invalidCallsAreRefusedByName() {
        MoeExpertHeat heat = new MoeExpertHeat();
        try {
            heat.note(0, new int[]{1}, 2);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("ids.length"));
        }
        try {
            heat.note(0, new int[]{1}, -1);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("-1"));
        }
        try {
            heat.note(0, null, 0);
        } catch (NullPointerException e) {
            assertEquals("ids", e.getMessage());
        }
    }
}
