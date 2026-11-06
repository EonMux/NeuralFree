package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.gpu.pipeline.SplitSpecs;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Filtered memory plans and the per-device context fit of a layer split: a device's plan keeps the slot of every
 * layer but only plans its own spans' caches, renumbers its KV groups, refuses a shared-KV reader separated from
 * its owner, and fits a larger context than the whole-model fit over the same budget.
 */
public class LayeredMemoryPlanningTest {

    private static MemoryConfig cfg() {
        return MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).nCtx(64).ubatch(8).build();
    }

    @Test
    public void theFilteredPlanKeepsEverySlotAndPlansOnlyItsOwnCaches() {
        ModelSpec spec = SplitSpecs.trunk(4, MemoryKind.KV);
        MemoryConfig cfg = cfg();
        MemoryPlan full = MemoryPlanner.planForContext(spec, cfg, 64);
        MemoryPlan late = MemoryPlanner.planForContext(spec, cfg, 64, il -> il >= 2 && il < 4);

        assertEquals(4, late.layers().size());
        assertEquals(LayerMemoryPlan.Kind.NONE, late.layer(0).kind());
        assertEquals(LayerMemoryPlan.Kind.NONE, late.layer(1).kind());
        assertEquals(LayerMemoryPlan.Kind.KV, late.layer(2).kind());
        assertEquals(LayerMemoryPlan.Kind.KV, late.layer(3).kind());
        assertEquals(full.layer(2).kv().toString(), late.layer(2).kv().toString());
        assertEquals(1, late.groups().size());
        assertEquals(full.kvBytes() / 2, late.kvBytes());
        assertEquals(64, late.nCtx());
        assertEquals(8, late.ubatch());
    }

    @Test
    public void theFirstHalfOfASplitPlansItsOwnHalf() {
        ModelSpec spec = SplitSpecs.trunk(4, MemoryKind.KV);
        MemoryConfig cfg = cfg();
        MemoryPlan early = MemoryPlanner.planForContext(spec, cfg, 64, il -> il >= 0 && il < 2);
        assertEquals(LayerMemoryPlan.Kind.KV, early.layer(0).kind());
        assertEquals(LayerMemoryPlan.Kind.NONE, early.layer(3).kind());
        assertEquals(1, early.groups().size());
    }

    @Test
    public void theTwoHalvesCarryIndependentPositionBuffers() {
        ModelSpec spec = SplitSpecs.trunk(4, MemoryKind.KV);
        MemoryConfig cfg = cfg();
        MemoryPlan early = MemoryPlanner.planForContext(spec, cfg, 64, il -> il >= 0 && il < 2);
        MemoryPlan late = MemoryPlanner.planForContext(spec, cfg, 64, il -> il >= 2 && il < 4);
        assertEquals(early.positionBytes(), late.positionBytes());
        assertTrue(early.positionBytes() > 0);
    }

    @Test
    public void aSharedKvReaderSeparatedFromItsOwnerIsRefused() {
        ModelSpec spec = SplitSpecs.sharedKv(4, 2, 1);
        MemoryConfig cfg = cfg();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MemoryPlanner.planForContext(spec, cfg, 64, il -> il >= 2 && il < 4));
        assertTrue(e.getMessage(), e.getMessage().contains("layer filter"));
    }

    @Test
    public void aSharedKvReaderPlannedWithItsOwnerReadsItsCache() {
        ModelSpec spec = SplitSpecs.sharedKv(4, 2, 1);
        MemoryConfig cfg = cfg();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, 64, il -> il >= 0 && il < 3);
        assertEquals(LayerMemoryPlan.Kind.NONE, plan.layer(3).kind());
        assertEquals(LayerMemoryPlan.Kind.SHARED_KV, plan.layer(2).kind());
        assertEquals(plan.layer(1).kv(), plan.layer(2).kv());
        assertEquals(plan.layer(1).kvGroup(), plan.layer(2).kvGroup());
    }

    @Test
    public void thePerDeviceFitExceedsTheWholeModelFitOverTheSameBudget() {
        ModelSpec spec = SplitSpecs.trunk(8, MemoryKind.KV);
        MemoryConfig cfg = cfg().toBuilder().nCtx(0).build();
        long budget = ContextFitter.requiredBytes(spec, cfg, 512);
        int whole = ContextFitter.fit(spec, cfg, budget);
        assertEquals(512, whole);
        int late = ContextFitter.fitLayers(spec, cfg, budget, il -> il >= 4 && il < 8);
        int early = ContextFitter.fitLayers(spec, cfg, budget, il -> il >= 0 && il < 4);
        assertTrue("half-model fits " + late + " over " + whole, late > whole);
        assertTrue("half-model fits " + early + " over " + whole, early > whole);
    }

    @Test
    public void anExplicitContextIsCheckedAgainstTheDeviceBudget() {
        ModelSpec spec = SplitSpecs.trunk(4, MemoryKind.KV);
        MemoryConfig cfg = cfg();
        java.util.function.IntPredicate earlyLayers = il -> il >= 0 && il < 2;
        long budget = MemoryPlanner.planForContext(spec, cfg, 64, earlyLayers).totalBytes() - 1024;
        assertThrows(ContextFitter.ContextBudgetException.class,
                () -> ContextFitter.fitLayers(spec, cfg, budget, earlyLayers));
        long exact = MemoryPlanner.planForContext(spec, cfg, 64, earlyLayers).totalBytes();
        assertEquals(64, ContextFitter.fitLayers(spec, cfg, exact, earlyLayers));
    }
}
