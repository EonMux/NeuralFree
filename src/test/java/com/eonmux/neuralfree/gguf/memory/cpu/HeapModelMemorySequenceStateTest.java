package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.CompressedStateCopy;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-sequence recurrent state rows of the paged CPU memory ({@link ModelMemory#saveSequenceState},
 * {@link ModelMemory#restoreSequenceState}, {@link ModelMemory#clearSequenceState}): a copy holds one row per
 * state tensor of every recurrent layer, a restore writes one sequence's rows and leaves the others, a clear
 * zeroes them; the contiguous memory, which keeps one state for its single sequence, refuses by name, and a
 * memory without recurrent state copies nothing.
 */
public class HeapModelMemorySequenceStateTest {

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(32).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.kvPaged(true).parallel(2).kvPageSize(4);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GGUFModel open(GgufTestWriter w, boolean paged) throws IOException {
        EngineConfig cfg = config(paged);
        return new GGUFModel(w.writeTempFile().toString(), cfg);
    }

    /** {@code like}'s shape with every float {@code value}. */
    private static RecurrentStateCopy filled(RecurrentStateCopy like, float value) {
        float[][] rows = new float[like.rows()][];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = new float[like.floats(i)];
            Arrays.fill(rows[i], value);
        }
        return RecurrentStateCopy.of(rows);
    }

    private static float[] row(RecurrentStateCopy copy, int i) {
        float[] out = new float[copy.floats(i)];
        copy.copyInto(i, out, 0);
        return out;
    }

    @Test
    public void aRestoreWritesOneSequencesRowsAndAClearZeroesThem() throws IOException {
        try (GGUFModel m = open(SyntheticModels.mamba2(), true)) {
            ModelMemory memory = m.forwardBackend().memory();
            assertTrue(memory.hasRecurrentState());
            assertNull(memory.sequenceStateRefusal());
            RecurrentStateCopy fresh = memory.saveSequenceState(0);
            assertTrue("a row per state tensor of each layer", fresh.rows() >= 2);
            for (int i = 0; i < fresh.rows(); i++) {
                assertArrayEquals("a fresh slot is zero", new float[fresh.floats(i)], row(fresh, i), 0f);
            }
            RecurrentStateCopy ones = filled(fresh, 1.5f);
            memory.restoreSequenceState(1, ones);
            RecurrentStateCopy back = memory.saveSequenceState(1);
            assertEquals(ones.bytes(), back.bytes());
            for (int i = 0; i < back.rows(); i++) {
                assertArrayEquals(row(ones, i), row(back, i), 0f);
                assertArrayEquals("the other sequence keeps its rows", row(fresh, i),
                        row(memory.saveSequenceState(0), i), 0f);
            }
            memory.clearSequenceState(1);
            RecurrentStateCopy cleared = memory.saveSequenceState(1);
            for (int i = 0; i < cleared.rows(); i++) {
                assertArrayEquals(row(fresh, i), row(cleared, i), 0f);
            }
        }
    }

    @Test
    public void aCopyOfAnotherShapeOrASequenceOutsideRefuses() throws IOException {
        try (GGUFModel m = open(SyntheticModels.mamba2(), true)) {
            ModelMemory memory = m.forwardBackend().memory();
            RecurrentStateCopy fresh = memory.saveSequenceState(0);
            assertThrows(IllegalArgumentException.class,
                    () -> memory.restoreSequenceState(0, RecurrentStateCopy.EMPTY));
            float[][] narrow = new float[fresh.rows()][];
            for (int i = 0; i < narrow.length; i++) {
                narrow[i] = new float[fresh.floats(i) - (i == 0 ? 1 : 0)];
            }
            assertThrows(IllegalArgumentException.class,
                    () -> memory.restoreSequenceState(0, RecurrentStateCopy.of(narrow)));
            assertThrows(IndexOutOfBoundsException.class, () -> memory.saveSequenceState(2));
            assertThrows(IndexOutOfBoundsException.class, () -> memory.clearSequenceState(-1));
        }
    }

    @Test
    public void theContiguousMemoryRefusesPerSequenceStateByName() throws IOException {
        try (GGUFModel m = open(SyntheticModels.mamba2(), false)) {
            ModelMemory memory = m.forwardBackend().memory();
            String refusal = memory.sequenceStateRefusal();
            assertNotNull(refusal);
            assertTrue(refusal, refusal.contains("neuralfree.parallel"));
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> memory.saveSequenceState(0));
            assertEquals(refusal, e.getMessage());
            assertThrows(UnsupportedOperationException.class, () -> memory.clearSequenceState(0));
        }
    }

    @Test
    public void aMemoryWithoutRecurrentStateCopiesNothing() throws IOException {
        try (GGUFModel m = open(SyntheticModels.llama(), true)) {
            ModelMemory memory = m.forwardBackend().memory();
            assertNull(memory.sequenceStateRefusal());
            assertEquals(0, memory.saveSequenceState(0).rows());
            memory.restoreSequenceState(0, RecurrentStateCopy.EMPTY);
            memory.clearSequenceState(0);
            float[][] one = {{1f}};
            assertThrows(IllegalArgumentException.class,
                    () -> memory.restoreSequenceState(0, RecurrentStateCopy.of(one)));
        }
    }

    /**
     * The per-sequence DSV4 compressed streams of the paged CPU memory ({@link ModelMemory#saveCompressedState},
     * {@link ModelMemory#restoreCompressedState}): a copy holds every compressed cache's rows and compressor
     * planes, a restore writes one sequence's streams and leaves the others, and a shape mismatch refuses. The
     * memory needs the deepseek4 fixture (its compressed caches), so the state rows live beside streams whose
     * rows the forward re-encodes.
     */
    @Test
    public void aCompressedStateCopyCarriesTheStreamsAndARestoreWritesOneSequenceBack() throws IOException {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(48).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .kvPaged(true).parallel(2).kvPageSize(4);
        EngineConfig cfg = EngineConfig.builder().memory(mc.build()).build();
        try (GGUFModel m = new GGUFModel(SyntheticRoutedModels.deepseek4().writeTempFile().toString(), cfg)) {
            ModelMemory memory = m.forwardBackend().memory();
            assertTrue(memory.plan().hasCompressed());
            assertNull(memory.compressedStateRefusal());
            CompressedStateCopy fresh = memory.saveCompressedState(0);
            assertTrue("a stream per compressed cache", fresh.streams() >= 1);
            assertEquals("a fresh slot's rollback record", 0, fresh.lastStepTokens());
            memory.restoreCompressedState(1, fresh);
            memory.restoreCompressedState(0, fresh);
            CompressedStateCopy other = memory.saveCompressedState(1);
            assertEquals(fresh.streams(), other.streams());
            for (int i = 0; i < other.streams(); i++) {
                assertEquals(fresh.stream(i).rows(), other.stream(i).rows());
            }
            assertThrows(IndexOutOfBoundsException.class, () -> memory.saveCompressedState(2));
            assertThrows(IllegalArgumentException.class,
                    () -> memory.restoreCompressedState(0, CompressedStateCopy.EMPTY));
            CompressedStateCopy narrow = CompressedStateCopy.of(java.util.List.of(
                    CompressedStateCopy.Stream.of(new float[0][], new float[0], new float[0])), 0, false);
            assertThrows(IllegalArgumentException.class, () -> memory.restoreCompressedState(0, narrow));
        }
    }

    @Test
    public void aStateCopyIsADeepImmutableCopy() {
        float[][] rows = {{1f, 2f}, {3f}};
        RecurrentStateCopy copy = RecurrentStateCopy.of(rows);
        rows[0][0] = 9f;
        assertEquals(2, copy.rows());
        assertEquals(2, copy.floats(0));
        assertEquals(12L, copy.bytes());
        float[] dst = new float[3];
        copy.copyInto(0, dst, 1);
        assertArrayEquals(new float[] {0f, 1f, 2f}, dst, 0f);
        assertThrows(IndexOutOfBoundsException.class, () -> copy.copyInto(1, new float[1], 1));
        assertThrows(IndexOutOfBoundsException.class, () -> copy.floats(2));
        assertEquals(0, RecurrentStateCopy.EMPTY.rows());
    }
}
