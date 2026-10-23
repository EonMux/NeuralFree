package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Context selection of {@link ContextFitter} on the gemma3 fixture of {@link MemoryPlannerTest} (n_ctx_train 8192).
 * For {@code n >= 1024} its plan takes {@code 5 * 1024 * 1024} ring bytes, {@code 1024 * 4} ring positions and
 * {@code n * (1024 + 4)} bytes of full-attention KV and positions: {@code 5_246_976 + 1028 * n}. Also covers the
 * recurrent-state snapshot shadows, the NextN draft memory and the companion (drafter) memory that the fit reserves.
 */
public class ContextFitterTest {

    private static final long FULL_TRAIN_BYTES = 13_668_352L;

    private static long gemma3Bytes(int nCtx) {
        return 5_246_976L + 1028L * nCtx;
    }

    private static MemoryConfig auto() {
        return MemoryConfig.builder().build();
    }

    private static MemoryConfig user(int nCtx) {
        return MemoryConfig.builder().nCtx(nCtx).build();
    }

    @Test
    public void requiredBytesMatchesTheTrace() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        assertEquals(FULL_TRAIN_BYTES, ContextFitter.requiredBytes(spec, auto(), 8192));
        assertEquals(gemma3Bytes(8192), FULL_TRAIN_BYTES);
        assertEquals(gemma3Bytes(3584), ContextFitter.requiredBytes(spec, auto(), 3584));
    }

    @Test
    public void autoUsesTrainingContextWhenItFits() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        assertEquals(8192, ContextFitter.fit(spec, auto(), ContextFitter.UNLIMITED));
        assertEquals(8192, ContextFitter.fit(spec, auto(), FULL_TRAIN_BYTES));
    }

    @Test
    public void autoPicksLargestFittingMultipleOf256() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        long budget = 9_000_000L;
        assertTrue(gemma3Bytes(3584) <= budget);
        assertTrue(gemma3Bytes(3840) > budget);
        assertEquals(3584, ContextFitter.fit(spec, auto(), budget));
        assertEquals(8192 - 256, ContextFitter.fit(spec, auto(), FULL_TRAIN_BYTES - 1));
    }

    @Test
    public void planHonoursTheBudget() {
        MemoryPlan plan = MemoryPlanner.plan(MemoryPlannerTest.gemma3(), auto(), 9_000_000L);
        assertEquals(3584, plan.nCtx());
        assertEquals(gemma3Bytes(3584), plan.totalBytes());
        MemoryPlannerTest.assertTotalsMatchLayers(plan);
    }

    @Test
    public void autoThrowsWhenNotEvenTheSmallestContextFits() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        long smallest = ContextFitter.requiredBytes(spec, auto(), ContextFitter.CONTEXT_STEP);
        assertEquals(6L * 256 * 1024 + 2L * 256 * 4, smallest);
        ContextFitter.ContextBudgetException e = assertThrows(ContextFitter.ContextBudgetException.class,
                () -> ContextFitter.fit(spec, auto(), smallest - 1));
        assertEquals(ContextFitter.CONTEXT_STEP, e.nCtx());
        assertEquals(smallest, e.requiredBytes());
        assertEquals(smallest - 1, e.budgetBytes());
    }

    @Test
    public void userContextIsUsedExactlyWhenItFits() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        assertEquals(3000, ContextFitter.fit(spec, user(3000), gemma3Bytes(3000)));
        assertEquals(16_384, ContextFitter.fit(spec, user(16_384), ContextFitter.UNLIMITED));
    }

    @Test
    public void userContextThatExceedsTheBudgetIsRejected() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        ContextFitter.ContextBudgetException e = assertThrows(ContextFitter.ContextBudgetException.class,
                () -> ContextFitter.fit(spec, user(4096), gemma3Bytes(4096) - 1));
        assertEquals(4096, e.nCtx());
    }

    @Test
    public void recurrentModelsKeepTheirLargeTrainingContext() {
        assertEquals(1_048_576, ContextFitter.fit(MemoryPlannerTest.mamba2(), auto(), 2_000_000L));
    }

    @Test
    public void stateSnapshotsAreCountedInTheRequiredBytes() {
        MemoryConfig snapshots = MemoryConfig.builder().stateSnapshots(true).build();
        assertEquals(1_615_872L, ContextFitter.requiredBytes(MemoryPlannerTest.mamba2(), auto(), 4096));
        assertEquals(2 * 1_615_872L, ContextFitter.requiredBytes(MemoryPlannerTest.mamba2(), snapshots, 4096));
    }

    /**
     * NextN fixture of {@link MemoryPlannerTest#withNextn()}: the main context takes {@code 1028 * n} bytes (two KV
     * layers of 512 bytes per row plus positions), the NextN draft memory {@code 516 * n}.
     */
    @Test
    public void nextnDraftMemoryIsReservedBesideThePlan() {
        ModelSpec spec = MemoryPlannerTest.withNextn();
        MemoryConfig mtp = MemoryConfig.builder().nextnMemory(true).build();
        assertEquals(1028L * 4096, ContextFitter.requiredBytes(spec, auto(), 4096));
        assertEquals(1544L * 4096, ContextFitter.requiredBytes(spec, mtp, 4096));
        long budget = 1544L * 4096;
        assertEquals(4096, ContextFitter.fit(spec, mtp, budget));
        assertEquals(6144, ContextFitter.fit(spec, auto(), budget));
        ContextFitter.ContextBudgetException e = assertThrows(ContextFitter.ContextBudgetException.class,
                () -> ContextFitter.fit(spec, MemoryConfig.builder().nCtx(4096).nextnMemory(true).build(), budget - 1));
        assertEquals(1544L * 4096, e.requiredBytes());
        assertTrue(e.getMessage().contains("NextN draft memory"));
    }

    @Test
    public void companionIsReservedAtEveryCandidateContext() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        ContextFitter.Companion weights = ContextFitter.Companion.of("draft model weights", 1_000_000L, 0L);
        assertEquals(gemma3Bytes(3584) + 1_000_000L, ContextFitter.requiredBytes(spec, auto(), 3584, weights));
        assertEquals(3584, ContextFitter.fit(spec, auto(), 10_000_000L, weights));
        assertEquals(4608, ContextFitter.fit(spec, auto(), 10_000_000L));
        assertEquals(4608, ContextFitter.fit(spec, auto(), 10_000_000L, ContextFitter.Companion.NONE));
    }

    @Test
    public void companionStepBytesFollowThePlannedUbatch() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        ContextFitter.Companion scratch = ContextFitter.Companion.of("draft scratch", 0L, 100L);
        assertEquals(gemma3Bytes(4096) + 100L * MemoryConfig.DEFAULT_UBATCH,
                ContextFitter.requiredBytes(spec, auto(), 4096, scratch));
    }

    @Test
    public void companionCachesArePlannedAtTheTargetContextWithTheirOwnSettings() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        ContextFitter.Companion drafter = ContextFitter.Companion.withCaches("eagle3 drafter", 0L, 0L,
                MemoryPlannerTest.gemma3(), MemoryConfig.defaults());
        assertTrue(drafter.hasCaches());
        assertEquals(2 * gemma3Bytes(4096), ContextFitter.requiredBytes(spec, auto(), 4096, drafter));
        MemoryConfig q8 = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).build();
        assertEquals(gemma3Bytes(4096), ContextFitter.requiredBytes(spec, q8, 4096, drafter)
                - ContextFitter.requiredBytes(spec, q8, 4096));
    }

    @Test
    public void companionCacheBytesAreSizedAtTheTargetContext() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        ContextFitter.Companion drafter = ContextFitter.Companion.withCacheBytes("dflash drafter", 0L, 0L,
                "the dflash KV cache", (nCtx, ubatch) -> 10L * nCtx);
        assertTrue(drafter.hasCaches());
        assertEquals(gemma3Bytes(4096) + 10L * 4096, ContextFitter.requiredBytes(spec, auto(), 4096, drafter));
        assertThrows(IllegalArgumentException.class, () -> ContextFitter.Companion.withCacheBytes("dflash drafter",
                0L, 0L, "the dflash KV cache", (nCtx, ubatch) -> -1L));
    }

    @Test
    public void companionAppearsInTheBudgetMessage() {
        ModelSpec spec = MemoryPlannerTest.gemma3();
        ContextFitter.Companion weights = ContextFitter.Companion.of("draft model weights", 1_000_000L, 0L);
        ContextFitter.ContextBudgetException e = assertThrows(ContextFitter.ContextBudgetException.class,
                () -> ContextFitter.fit(spec, user(4096), gemma3Bytes(4096) + 999_999L, weights));
        assertEquals(gemma3Bytes(4096) + 1_000_000L, e.requiredBytes());
        assertTrue(e.getMessage().contains("draft model weights"));
    }

    @Test
    public void reservedBytesAreTheNextnAndCompanionMemoryBesideThePlan() {
        ModelSpec spec = MemoryPlannerTest.withNextn();
        MemoryConfig mtp = MemoryConfig.builder().nextnMemory(true).build();
        ContextFitter.Companion weights = ContextFitter.Companion.of("draft model weights", 1000L, 0L);
        MemoryPlan withNextn = MemoryPlanner.planForContext(spec, mtp, 4096);
        assertEquals(516L * 4096, ContextFitter.reservedBytes(spec, mtp, withNextn, ContextFitter.Companion.NONE));
        assertEquals(516L * 4096 + 1000L, ContextFitter.reservedBytes(spec, mtp, withNextn, weights));
        MemoryPlan plain = MemoryPlanner.planForContext(spec, auto(), 4096);
        assertEquals(1000L, ContextFitter.reservedBytes(spec, auto(), plain, weights));
    }

    @Test
    public void companionRejectsInvalidSizes() {
        assertThrows(IllegalArgumentException.class, () -> ContextFitter.Companion.of(" ", 0L, 0L));
        assertThrows(IllegalArgumentException.class, () -> ContextFitter.Companion.of("drafter", -1L, 0L));
        assertThrows(IllegalArgumentException.class, () -> ContextFitter.Companion.of("drafter", 0L, -1L));
    }

    @Test
    public void rejectsNegativeBudget() {
        assertThrows(IllegalArgumentException.class,
                () -> ContextFitter.fit(MemoryPlannerTest.gemma3(), auto(), -1));
    }
}
