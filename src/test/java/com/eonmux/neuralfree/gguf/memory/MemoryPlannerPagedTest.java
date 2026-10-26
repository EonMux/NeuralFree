package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.junit.Test;

import static com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest.assertTotalsMatchLayers;
import static com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest.gemma3;
import static com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest.mamba2;
import static com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest.qwen35;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Paged rows of {@link MemoryPlanner} (PARITY MS-2): a paged setting ({@code neuralfree.parallel > 1} or
 * {@code neuralfree.kv.paged}) replaces the rings with page pools of {@code neuralfree.kv.page.size} rows — full
 * attention keeps {@code roundUp(nCtx, P)}, SWA keeps
 * {@code min(roundUp(nCtx, P), parallel x (roundUp(window + nUbatch, P) + P))}, recurrent state keeps
 * {@code parallel x (1 + rollbackSnapshots)} slots per tensor and a paged group never wraps (llama.cpp
 * {@code -c} over {@code --kv-unified}); {@link ContextFitter} fits with the same formulas, and
 * {@code parallel = 1} plans stay byte-identical to the unpaged planner output ({@link MemoryPlan#toString()}).
 * The synthetic specs (gemma3 attention / SWA, qwen35 hybrid recurrecurrent, mamba2 recurrent) are the planner
 * test's own fixtures, planned here in the paged form.
 */
public class MemoryPlannerPagedTest {

    private static MemoryConfig f16() {
        return MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).build();
    }

    private static MemoryConfig paged(int pageSize, int parallel) {
        return f16().toBuilder().kvPageSize(pageSize).parallel(parallel).build();
    }

    @Test
    public void pagedAttentionPlansRoundUpPagesAndDropsTheRings() {
        MemoryPlan plan = MemoryPlanner.planForContext(gemma3(), paged(16, 4), 8192);

        assertEquals(2, plan.groups().size());
        MemoryPlan.KvGroup swa = plan.group(0);
        MemoryPlan.KvGroup full = plan.group(1);
        // full attention: roundUp(nCtx, P) cells shared by all sequences (8192 = 512 pages of 16)
        assertEquals(8192, full.rows());
        assertEquals(16, full.pageSize());
        assertEquals(512, full.pages());
        assertTrue(full.paged());
        assertFalse(full.ring());
        // SWA: min(roundUp(nCtx, P) = 8192,
        //          parallel x (roundUp(window + nUbatch, P) + P) = 4 x (roundUp(512 + 512, 16) + 16) = 4160)
        assertEquals(4160, swa.rows());
        assertEquals(260, swa.pages());
        assertFalse(swa.ring());

        // the layer caches read exactly the paged groups and wrap nowhere
        for (int il = 0; il < 6; il++) {
            assertEquals(plan.layer(il).kv().rows(), il < 5 ? swa.rows() : full.rows());
            assertFalse(plan.layer(il).kv().ring());
        }
        // positions of paged groups map through PagedSequences, never through rowOf
        assertThrows(UnsupportedOperationException.class, () -> full.rowOf(0));
        assertThrows(UnsupportedOperationException.class, () -> swa.rowOf(0));

        assertTrue(plan.toString().contains("pages=260x16"));
        assertFalse(plan.toString().contains(" ring"));
        assertTotalsMatchLayers(plan);

        // the paged rule makes swaFull irrelevant: the SWA pool stays the per-sequence window pool
        MemoryPlan swaFull = MemoryPlanner.planForContext(gemma3(),
                paged(16, 4).toBuilder().swaFull(true).build(), 8192);
        assertEquals(4160, swaFull.group(0).rows());
        assertFalse(swaFull.group(0).ring());
    }

    @Test
    public void pagedRecurrentStateKeepsASlotPerSequencePlusSnapshots() {
        MemoryPlan plain = MemoryPlanner.planForContext(mamba2(), f16(), 4096);
        assertEquals(1_615_872L, plain.stateBytes());      // 2 layers x 807 936 bytes, one slot per tensor

        MemoryPlan three = MemoryPlanner.planForContext(mamba2(), paged(16, 3), 4096);
        for (int il = 0; il < 2; il++) {
            assertFalse(three.layer(il).states().isEmpty());
            for (LayerMemoryPlan.StateTensor s : three.layer(il).states()) {
                assertEquals(3, s.slots());
            }
            assertEquals(3 * plain.layer(il).stateBytes(), three.layer(il).stateBytes());
        }
        assertEquals(3 * plain.stateBytes(), three.stateBytes());
        assertEquals(0, three.kvBytes());
        assertEquals(0, three.positionBytes());
        assertTotalsMatchLayers(three);

        // the rollback snapshots come per sequence, as llama.cpp puts n_rs_seq rows into the recurrent cache
        MemoryPlan rolled = MemoryPlanner.planForContext(mamba2(),
                f16().forRollback(3).toBuilder().parallel(3).build(), 4096);
        for (int il = 0; il < 2; il++) {
            for (LayerMemoryPlan.StateTensor s : rolled.layer(il).states()) {
                assertEquals(3 * (1 + 3), s.slots());      // 3 live slots + 3 sequences x 3 snapshots
                assertEquals(12 * plain.layer(il).stateBytes(), rolled.layer(il).stateBytes());
            }
        }
        assertEquals(12 * plain.stateBytes(), rolled.stateBytes());
        assertTotalsMatchLayers(rolled);
        // the PC-2 checkpoint slots stay 0 until PC-2: the plan counts only parallel x (1 + snapshots) slots
    }

    @Test
    public void pagedHybridPlansRecurrentSlotsAndPagedKvTogether() {
        MemoryPlan plan = MemoryPlanner.planForContext(qwen35(), paged(16, 2), 4096);
        assertEquals(MemoryKind.HYBRID, plan.kind());
        // the attention layer's full-attention group: roundUp(4096, 16) cells = 256 pages of 16
        assertEquals(1, plan.groups().size());
        MemoryPlan.KvGroup g = plan.group(0);
        assertEquals(4096, g.rows());
        assertEquals(256, g.pages());
        assertTrue(g.paged());
        assertFalse(g.ring());
        // the GDN layers' state tensors carry one slot per sequence
        for (int il = 0; il < 3; il++) {
            for (LayerMemoryPlan.StateTensor s : plan.layer(il).states()) {
                assertEquals(2, s.slots());
            }
            assertEquals(2 * 274_432L, plan.layer(il).stateBytes());
        }
        assertEquals(8_388_608L, plan.kvBytes());
        assertEquals(4096L * 4, plan.positionBytes());
        assertTotalsMatchLayers(plan);
    }

    @Test
    public void parallelOnePlansStayByteIdenticalToTheUnpagedPlanner() {
        MemoryConfig one = f16().toBuilder().parallel(1).build();
        ModelSpec[] specs = {gemma3(), qwen35(), mamba2()};
        int[] contexts = {8192, 4096, 4096};
        for (int i = 0; i < specs.length; i++) {
            MemoryPlan plain = MemoryPlanner.planForContext(specs[i], f16(), contexts[i]);
            MemoryPlan parallelOne = MemoryPlanner.planForContext(specs[i], one, contexts[i]);
            assertEquals("parallel = 1 replans the same " + archName(specs[i]),
                    plain.toString(), parallelOne.toString());
            // and the same spec replanned paged differs: the pools replace the rings
            MemoryPlan paged = MemoryPlanner.planForContext(specs[i], paged(16, 4), contexts[i]);
            assertNotEquals(plain.toString(), paged.toString());
            if (plain.groups().size() > 0 && plain.group(0).rows() < contexts[i]) {
                assertTrue(plain.toString().contains(" ring"));
                assertFalse(paged.toString().contains(" ring"));
            }
        }
    }

    private static String archName(ModelSpec spec) {
        return spec.arch().ggufName();
    }

    @Test
    public void contextFitterSizesThePagedPlanWithTheSameFormulas() {
        ModelSpec spec = gemma3();
        MemoryConfig cfg = paged(16, 4);
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, 8192);
        assertEquals(plan.totalBytes(), ContextFitter.requiredBytes(spec, cfg, 8192));
        // the training context fits its own budget exactly, so the fitter hands it out unchanged
        assertEquals(8192, ContextFitter.fit(spec, cfg, plan.totalBytes()));
        // a budget one byte under the training context steps down to a CONTEXT_STEP multiple whose plan fits
        int fitted = ContextFitter.fit(spec, cfg, plan.totalBytes() - 1);
        assertTrue(fitted < 8192);
        assertEquals(0, fitted % ContextFitter.CONTEXT_STEP);
        assertTrue(ContextFitter.requiredBytes(spec, cfg, fitted) <= plan.totalBytes() - 1);
        assertTrue(ContextFitter.requiredBytes(spec, cfg, fitted + ContextFitter.CONTEXT_STEP)
                > plan.totalBytes() - 1);
    }

    @Test
    public void handBuiltPagedGroupsCarryTheSameRules() {
        // the paged form is a MemoryPlan.KvGroup form: page size a power of two in [1, 256], rows = pages x size
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, 64, false,
                KvWindow.of(SwaType.NONE, 0, true), 16, 4);
        assertTrue(g.paged());
        assertEquals(16, g.pageSize());
        assertEquals(4, g.pages());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryPlan.KvGroup.of(0, 60, false, KvWindow.FULL_CAUSAL, 16, 4));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryPlan.KvGroup.of(0, 64, true, KvWindow.FULL_CAUSAL, 16, 4));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryPlan.KvGroup.of(0, 64, false, KvWindow.FULL_CAUSAL, 15, 4));
        assertThrows(IllegalArgumentException.class, () -> MemoryPlan.KvGroup.of(0, 64, false, KvWindow.FULL_CAUSAL,
                0, 4));
        assertThrows(IllegalArgumentException.class, () -> MemoryPlan.KvGroup.of(0, 64, false, KvWindow.FULL_CAUSAL,
                16, 0));
        // the contiguous form stays the old one
        MemoryPlan.KvGroup contiguous = MemoryPlan.KvGroup.of(0, 64, true, KvWindow.FULL_CAUSAL);
        assertFalse(contiguous.paged());
        assertEquals(0, contiguous.pageSize());
        assertEquals(0, contiguous.pages());
        assertEquals(0, contiguous.rowOf(64));             // the ring of 64 rows
        assertTrue(contiguous.toString().contains(" ring"));
    }
}
