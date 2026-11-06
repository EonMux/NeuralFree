package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Where {@link ContextCheckpoints} takes a text prefill's checkpoints (llama.cpp's server: a checkpoint shortly
 * before the prompt's end, and every {@code --checkpoint-min-step} tokens on the way, 0 being no minimum), what
 * makes a memory need them, and the recurrent half of capture and restore.
 */
public class ContextCheckpointsTest {

    private static final int PAGE = 4;

    private static GGUFModel open(GgufTestWriter w) throws IOException {
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        return new GGUFModel(w.writeTempFile().toString(), cfg);
    }

    private static ContextCheckpoints checkpoints(GGUFModel m, int perSlot, int minStep) {
        ModelMemory memory = m.forwardBackend().memory();
        return new ContextCheckpoints(memory.pagedSequences(), memory, perSlot, minStep);
    }

    @Test
    public void aMemoryNeedsCheckpointsForRecurrentStateOrTrimmingGroups() throws IOException {
        String[] expected = {"recurrent state", "sliding-window groups", "recurrent state and sliding-window groups",
                null, "sliding-window groups and DSV4 compressed caches"};
        GgufTestWriter[] models = {SyntheticModels.mamba2(), SyntheticModels.gemma2(), SyntheticModels.lfm3(),
                SyntheticModels.llama(), SyntheticRoutedModels.deepseek4()};
        for (int i = 0; i < models.length; i++) {
            try (GGUFModel m = open(models[i])) {
                ModelMemory memory = m.forwardBackend().memory();
                assertEquals(m.architecture(), expected[i],
                        ContextCheckpoints.needs(memory, memory.pagedSequences()));
            }
        }
    }

    @Test
    public void theLimitIsPerSequenceSlotAndBadSettingsRefuse() throws IOException {
        try (GGUFModel m = open(SyntheticModels.mamba2())) {
            assertEquals(6, checkpoints(m, 3, 0).limit());
            assertFalse(checkpoints(m, 3, 0).windowed());
            assertThrows(IllegalArgumentException.class, () -> checkpoints(m, 0, 0));
            assertThrows(IllegalArgumentException.class, () -> checkpoints(m, 1, -1));
        }
        try (GGUFModel m = open(SyntheticModels.gemma2())) {
            assertTrue(checkpoints(m, 1, 0).windowed());
        }
    }

    @Test
    public void spacedTargetsPrecedeTheTailAndRoundUpToAPage() throws IOException {
        try (GGUFModel m = open(SyntheticModels.mamba2())) {
            ContextCheckpoints cp = checkpoints(m, 4, 6);         // 6 rounds up to 8 rows
            assertEquals(8, cp.nextTarget(0, 40, 0));
            assertEquals(16, cp.nextTarget(8, 40, 8));
            assertEquals("a target past the committed rows from an older checkpoint", 8, cp.nextTarget(5, 40, 0));
            assertEquals("the tail (last page boundary before the final row) ends the spacing", 36,
                    cp.nextTarget(32, 40, 32));
            assertEquals(-1, cp.nextTarget(36, 40, 36));
            assertEquals("a prompt within its first page has no checkpoint row", -1, cp.nextTarget(0, 3, 0));
            assertTrue(cp.due(8, 40, 0));
            assertFalse("between spaced rows", cp.due(12, 40, 8));
            assertTrue("the tail", cp.due(36, 40, 32));
            assertFalse("off a page boundary", cp.due(10, 40, 0));
            assertFalse("the final row is no checkpoint row on the way", cp.due(40, 40, 36));
            assertFalse("at the last checkpoint", cp.due(8, 40, 8));
        }
    }

    @Test
    public void noMinimumSpacingMakesEveryChunkEndACheckpointRow() throws IOException {
        try (GGUFModel m = open(SyntheticModels.mamba2())) {
            ContextCheckpoints cp = checkpoints(m, 4, 0);
            assertEquals("only the tail bounds a chunk", 36, cp.nextTarget(0, 40, 0));
            assertTrue(cp.due(4, 40, 0));
            assertTrue(cp.due(12, 40, 8));
            ContextCheckpoints sparse = checkpoints(m, 4, Integer.MAX_VALUE);
            assertEquals("no overflow past the largest spacing", 36, sparse.nextTarget(0, 40, 0));
            assertFalse(sparse.due(8, 40, 0));
            assertTrue(sparse.due(36, 40, 0));
        }
    }

    @Test
    public void aCaptureCopiesTheRecurrentStateAndARestoreWritesItBack() throws IOException {
        try (GGUFModel m = open(SyntheticModels.mamba2())) {
            ModelMemory memory = m.forwardBackend().memory();
            ContextCheckpoints cp = checkpoints(m, 4, 0);
            RecurrentStateCopy zero = memory.saveSequenceState(0);
            float[][] rows = new float[zero.rows()][];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = new float[zero.floats(i)];
                Arrays.fill(rows[i], 0.25f * (i + 1));
            }
            memory.restoreSequenceState(0, RecurrentStateCopy.of(rows));
            ContextCheckpoint c = cp.capture(0, 0);
            assertEquals(0, c.rows);
            for (int g = 0; g < c.windows().length; g++) {
                assertNull("no trimming group, no window", c.windows()[g]);
            }
            memory.clearSequenceState(0);
            cp.restore(1, c);
            RecurrentStateCopy back = memory.saveSequenceState(1);
            for (int i = 0; i < rows.length; i++) {
                float[] row = new float[back.floats(i)];
                back.copyInto(i, row, 0);
                assertArrayEquals(rows[i], row, 0f);
            }
            cp.release(c);
            PagedSequences paged = memory.pagedSequences();
            assertEquals(paged.totalCells(), paged.freeCells());
        }
    }
}
