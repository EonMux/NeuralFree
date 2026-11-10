package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.hybrid.HybridFit;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest.deepseek32;
import static com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest.gemma3;
import static com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest.qwen35;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The plan charges every byte the device memory allocates ({@link MemoryPlan#totalBytes()}): the caches plus the
 * addressing buffers of the step inputs (page tables, per-sequence key lengths and query positions, the per-step
 * position and index buffers, the shift deltas a context shift reserves at allocation, and the per-sequence
 * {@code kv_pos} copies of the groups a sparse mixer reads per sequence). A fit that budgets the plan's total can
 * therefore never be overrun by the memory it planned.
 */
public class MemoryPlanAddressingTest {

    private FakeComputeBackend backend;

    @Before
    public void setUp() {
        backend = FakeComputeBackend.create();
    }

    @After
    public void tearDown() {
        backend.close();
    }

    private static MemoryConfig config(int pageSize, int parallel, boolean shift, boolean unified) {
        MemoryConfig.Builder b = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .contextShift(shift).kvUnified(unified);
        if (pageSize > 0) {
            b.kvPageSize(pageSize).parallel(parallel);
        }
        return b.build();
    }

    /** Allocates the plan's memory, reserves what its sparse mixers reserve, and sums the live device bytes. */
    private long allocatedBytes(MemoryPlan plan) {
        long before = liveBytes();
        try (DeviceModelMemory mem = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan,
                MropeTextPositions.PPP0, null)) {
            for (int g = 0; g < plan.groups().size(); g++) {
                if (plan.sequenceAddressed(g)) {
                    mem.stepInputs().reserveSequenceAddressing(g);
                }
            }
            return liveBytes() - before;
        }
    }

    private long liveBytes() {
        long bytes = 0;
        for (FakeGpuBuffer b : backend.liveBuffers()) {
            bytes += b.sizeInBytes();
        }
        return bytes;
    }

    private void assertChargesEveryAllocation(ModelSpec spec, MemoryConfig cfg, int nCtx) {
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, nCtx);
        assertEquals(plan.cacheBytes() + plan.addressingBytes(), plan.totalBytes());
        assertEquals(plan.toString(), plan.totalBytes(), allocatedBytes(plan));
    }

    @Test
    public void denseAndSlidingWindowPlansChargeTheirAddressing() {
        for (boolean shift : new boolean[] {false, true}) {
            assertChargesEveryAllocation(gemma3(), config(0, 1, shift, true), 256);
            for (int parallel : new int[] {1, 2, 4}) {
                assertChargesEveryAllocation(gemma3(), config(16, parallel, shift, true), 256);
                assertChargesEveryAllocation(gemma3(), config(16, parallel, shift, false), 256);
            }
        }
    }

    @Test
    public void hybridPlansChargeTheirAddressing() {
        for (int parallel : new int[] {1, 3}) {
            assertChargesEveryAllocation(qwen35(), config(16, parallel, true, true), 128);
        }
    }

    @Test
    public void sparsePlansChargeThePerSequencePositions() {
        assertChargesEveryAllocation(deepseek32(), config(0, 1, false, true), 128);
        for (int parallel : new int[] {1, 2, 4}) {
            assertChargesEveryAllocation(deepseek32(), config(16, parallel, false, true), 128);
            assertChargesEveryAllocation(deepseek32(), config(16, parallel, true, false), 128);
        }
    }

    @Test
    public void onlyTheSparseMixersAddressTheirGroupsPerSequence() {
        MemoryPlan sparse = MemoryPlanner.planForContext(deepseek32(), config(16, 2, false, true), 128);
        assertTrue(sparse.sequenceAddressed(0));
        MemoryPlan dense = MemoryPlanner.planForContext(gemma3(), config(16, 2, false, true), 128);
        for (int g = 0; g < dense.groups().size(); g++) {
            assertFalse(dense.sequenceAddressed(g));
        }
    }

    @Test
    public void thePerSequencePositionsScaleWithTheSlotsAndTheRows() {
        int rows = 4096;
        MemoryPlan two = MemoryPlanner.planForContext(deepseek32(), config(16, 2, false, true), rows);
        MemoryPlan four = MemoryPlanner.planForContext(deepseek32(), config(16, 4, false, true), rows);
        // two more slots: two more kv_pos copies of the group, two more page-table rows of nCtx / 16 pages, two
        // more key lengths and two more query positions
        long perSlot = (long) rows * Integer.BYTES + (rows / 16) * Integer.BYTES + 2 * Integer.BYTES;
        assertEquals(2 * perSlot, four.addressingBytes() - two.addressingBytes());
        MemoryPlan split = MemoryPlanner.planForContext(deepseek32(), config(16, 4, false, false), rows);
        // without the unified context a page table row maps one slot's share of the context
        assertEquals(4L * (rows / 16 - rows / 4 / 16) * Integer.BYTES,
                four.addressingBytes() - split.addressingBytes());
    }

    @Test
    public void theContextShiftReservesOneDeltaPerRow() {
        MemoryPlan plain = MemoryPlanner.planForContext(gemma3(), config(16, 2, false, true), 256);
        MemoryPlan shifted = MemoryPlanner.planForContext(gemma3(), config(16, 2, true, true), 256);
        assertEquals(plain.positionBytes(), shifted.addressingBytes() - plain.addressingBytes());
    }

    /**
     * A hybrid split whose every layer is on the device charges exactly what the device memory allocates: the fit
     * takes the position and addressing bytes from the plan, with the context shift and several sequence slots.
     */
    @Test
    public void theHybridFitOfEveryDeviceLayerChargesEveryAllocation() {
        for (boolean shift : new boolean[] {false, true}) {
            assertHybridFitChargesEveryAllocation(gemma3(), config(0, 1, shift, true), 256);
            for (int parallel : new int[] {1, 4}) {
                assertHybridFitChargesEveryAllocation(gemma3(), config(16, parallel, shift, true), 256);
                assertHybridFitChargesEveryAllocation(deepseek32(), config(16, parallel, shift, true), 128);
            }
        }
    }

    private void assertHybridFitChargesEveryAllocation(ModelSpec spec, MemoryConfig cfg, int nCtx) {
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, nCtx);
        assertEquals(plan.toString(), allocatedBytes(plan), HybridFit.contextBytes(plan, il -> true));
    }
}
