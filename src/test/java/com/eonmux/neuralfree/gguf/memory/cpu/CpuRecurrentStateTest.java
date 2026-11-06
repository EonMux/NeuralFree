package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * Multi-slot recurrent state ({@code RecurrentStateAccessor}): a paged plan's tensors hold one row per state slot
 * ({@code parallel x (1 + rollbackSnapshots)}, {@code MemoryPlanner#stateSlots}), {@link CpuRecurrentState}
 * allocates and zeroes all of them, {@link CpuRecurrentState#clearSlot(int)} zeroes one sequence's row beside the
 * live rows of the other slots, and {@code snapshot} / {@code restore} copy the whole tensors.
 */
public class CpuRecurrentStateTest {

    private static List<LayerMemoryPlan.StateTensor> tensors(int slots) {
        return List.of(
                LayerMemoryPlan.StateTensor.of(LayerMemoryPlan.STATE_CONV, LayerMemoryPlan.StateLayout.CONV, 10,
                        slots),
                LayerMemoryPlan.StateTensor.of(LayerMemoryPlan.STATE_SCAN, LayerMemoryPlan.StateLayout.MAMBA_SCAN,
                        24, slots));
    }

    @Test
    public void tensorsHoldOneRowPerSlot() {
        CpuRecurrentState s = CpuRecurrentState.allocate(3, tensors(4));
        assertEquals(4, s.slots());
        assertEquals("the conv row width per slot", 10, s.slotFloats(LayerMemoryPlan.STATE_CONV));
        assertEquals(40, s.tensor(LayerMemoryPlan.STATE_CONV).length);
        assertEquals(96, s.tensor(LayerMemoryPlan.STATE_SCAN).length);
    }

    @Test
    public void singleSlotStateKeepsTheContiguousLayout() {
        CpuRecurrentState s = CpuRecurrentState.allocate(0, tensors(1));
        assertEquals(1, s.slots());
        assertEquals(10, s.tensor(LayerMemoryPlan.STATE_CONV).length);
    }

    @Test
    public void clearSlotZeroesOnlyThatSequenceRow() {
        CpuRecurrentState s = CpuRecurrentState.allocate(1, tensors(3));
        float[] conv = s.tensor(LayerMemoryPlan.STATE_CONV);
        float[] scan = s.tensor(LayerMemoryPlan.STATE_SCAN);
        for (int slot = 0; slot < 3; slot++) {
            for (int i = 0; i < 10; i++) {
                conv[slot * 10 + i] = 1 + slot;
            }
            for (int i = 0; i < 24; i++) {
                scan[slot * 24 + i] = 2 + slot;
            }
        }
        s.clearSlot(1);
        for (int i = 0; i < 10; i++) {
            assertEquals("conv slot 1 row zeroed", 0f, conv[10 + i], 0f);
            assertEquals("conv slot 0 kept", 1f, conv[i], 0f);
            assertEquals("conv slot 2 kept", 3f, conv[20 + i], 0f);
        }
        for (int i = 0; i < 24; i++) {
            assertEquals("scan slot 1 row zeroed", 0f, scan[24 + i], 0f);
            assertEquals("scan slot 0 kept", 2f, scan[i], 0f);
            assertEquals("scan slot 2 kept", 4f, scan[48 + i], 0f);
        }
        assertThrows(IndexOutOfBoundsException.class, () -> s.clearSlot(3));
    }

    @Test
    public void snapshotAndRestoreCoverEverySlot() {
        CpuRecurrentState s = CpuRecurrentState.allocate(2, tensors(2), true);
        float[] conv = s.tensor(LayerMemoryPlan.STATE_CONV);
        for (int i = 0; i < conv.length; i++) {
            conv[i] = 0.5f * (i + 1);
        }
        s.snapshot();
        java.util.Arrays.fill(conv, 7f);
        s.restore();
        float[] expected = new float[conv.length];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = 0.5f * (i + 1);
        }
        assertArrayEquals(expected, conv, 0f);
    }

    @Test
    public void mismatchedSlotCountsRefuse() {
        List<LayerMemoryPlan.StateTensor> mixed = List.of(
                LayerMemoryPlan.StateTensor.of(LayerMemoryPlan.STATE_CONV, LayerMemoryPlan.StateLayout.CONV, 10, 2),
                LayerMemoryPlan.StateTensor.of(LayerMemoryPlan.STATE_SCAN, LayerMemoryPlan.StateLayout.MAMBA_SCAN,
                        24, 4));
        assertThrows(IllegalArgumentException.class, () -> CpuRecurrentState.allocate(0, mixed));
    }
}
