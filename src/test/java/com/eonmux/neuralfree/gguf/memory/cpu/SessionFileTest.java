package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The session file ({@code llama_state_save_file}/{@code llama_state_load_file}) over a real engine: a saved
 * sequence restores so that the next decode produces exactly the logits the uninterrupted run produced — tokens,
 * reuse frontier and wrap flag, every KV group's position mirror (a shifted context included), the K/V rows and
 * the live recurrent state. A paged memory refuses by name, and a file saved for another plan is refused with
 * the mismatching field.
 */
public class SessionFileTest {

    private static EngineConfig config() {
        return config(KvDType.F32);
    }

    private static EngineConfig config(KvDType type) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                        .typeK(type).typeV(type).build())
                .build();
    }

    /** Ten prompt tokens within the vocab, then the token decoded after them. */
    private static final int[] PROMPT = {4, 9, 17, 33, 65, 129, 7, 11, 23, 47};
    private static final int NEXT = 101;

    @Test
    public void aRestoredSessionContinuesTheSequenceWithTheSameLogits() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path session = Files.createTempDirectory("nf-session-").resolve("session.nfst");
        float[] expected;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            HeapModelMemory memory = memoryOf(m);
            m.forwardBackend().prefill(PROMPT, 0);
            memory.saveSession(session);
            assertEquals("the sequence is stored", PROMPT.length, memory.sequence().nPast());
            assertEquals("the frontier follows the contiguous prefill", PROMPT.length, memory.sequence().reusable());
            assertFalse(memory.sequence().wrapped());
            expected = m.forwardBackend().decode(NEXT, PROMPT.length);
        }

        // a different prompt tramples the memory, then the session brings the saved one back
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            HeapModelMemory memory = memoryOf(m);
            int[] other = {8, 16, 24, 32, 40, 48, 56, 64};
            m.forwardBackend().prefill(other, 0);
            m.forwardBackend().decode(5, other.length);
            memory.loadSession(session);
            assertEquals(PROMPT.length, memory.sequence().nPast());
            assertArrayEquals(PROMPT, memory.sequence().tokens());
            assertEquals("the reuse frontier is restored", PROMPT.length, memory.sequence().reusable());
            float[] logits = m.forwardBackend().decode(NEXT, PROMPT.length);
            assertArrayEquals("the restored memory decodes exactly like the saved run", expected, logits, 1e-4f);
        } finally {
            Files.deleteIfExists(session);
        }
    }

    @Test
    public void aShiftedContextRestoresWithItsMovedRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path session = Files.createTempDirectory("nf-session-").resolve("shifted.nfst");
        float[] expected;
        int lastPos;
        int lastToken = 102;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            HeapModelMemory memory = memoryOf(m);
            m.forwardBackend().prefill(PROMPT, 0);
            m.forwardBackend().shiftContext(2, 4);
            m.forwardBackend().decode(101, memory.sequence().nPast());
            lastPos = memory.sequence().nPast() - 1;
            expected = m.forwardBackend().decode(lastToken, lastPos);
            assertTrue("the shift lowered the frontier", memory.sequence().reusable() <= 2);
            memory.saveSession(session);
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            HeapModelMemory memory = memoryOf(m);
            memory.loadSession(session);
            float[] logits = m.forwardBackend().decode(lastToken, lastPos);
            assertArrayEquals("the shifted rows land back where the shift put them", expected, logits, 1e-4f);
        } finally {
            Files.deleteIfExists(session);
        }
    }

    /** A per-token-head session carries the per-(token, KV head) scales, also across a context shift. */
    @Test
    public void perTokenHeadSessionsRestoreTheRowScales() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path session = Files.createTempDirectory("nf-session-").resolve("per-token-head.nfst");
        try {
            for (KvDType type : new KvDType[]{KvDType.FP8_PER_TOKEN_HEAD, KvDType.INT8_PER_TOKEN_HEAD,
                    KvDType.INT4_PER_TOKEN_HEAD}) {
                EngineConfig cfg = config(type);
                float[] expected;
                int lastPos;
                int lastToken = 102;
                try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
                    HeapModelMemory memory = memoryOf(m);
                    m.forwardBackend().prefill(PROMPT, 0);
                    m.forwardBackend().shiftContext(2, 4);
                    m.forwardBackend().decode(101, memory.sequence().nPast());
                    lastPos = memory.sequence().nPast() - 1;
                    expected = m.forwardBackend().decode(lastToken, lastPos);
                    memory.saveSession(session);
                }
                try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
                    HeapModelMemory memory = memoryOf(m);
                    memory.loadSession(session);
                    float[] logits = m.forwardBackend().decode(lastToken, lastPos);
                    assertArrayEquals(type + " scales and codes restore together", expected, logits, 1e-4f);
                }
            }
        } finally {
            Files.deleteIfExists(session);
        }
    }

    @Test
    public void aPagedMemoryRefusesAndAMismatchedPlanIsNamed() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path session = Files.createTempDirectory("nf-session-").resolve("plan.nfst");
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            HeapModelMemory memory = memoryOf(m);
            m.forwardBackend().prefill(PROMPT, 0);
            memory.saveSession(session);
        }
        // a paged plan refuses by name, before anything is written or reset
        MemoryConfig pagedMemory = MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        try (GGUFModel paged = new GGUFModel(model.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(pagedMemory).build())) {
            UnsupportedOperationException refused = assertThrows(UnsupportedOperationException.class,
                    () -> memoryOf(paged).saveSession(session));
            assertTrue(refused.getMessage(), refused.getMessage().contains("paged"));
            assertThrows(UnsupportedOperationException.class, () -> memoryOf(paged).loadSession(session));
        }
        // a file saved for another plan refuses with the mismatching field
        MemoryConfig other = MemoryConfig.builder().nCtx(32).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        try (GGUFModel small = new GGUFModel(model.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(other).build())) {
            HeapModelMemory memory = memoryOf(small);
            IOException mismatch = assertThrows(IOException.class, () -> memory.loadSession(session));
            assertTrue(mismatch.getMessage(), mismatch.getMessage().contains("n_ctx"));
        } finally {
            Files.deleteIfExists(session);
        }
    }

    /** The heap memory of a CPU model ({@code GGUFModel.memoryOf} is the engine's own accessor). */
    private static HeapModelMemory memoryOf(GGUFModel m) {
        return (HeapModelMemory) m.forwardBackend().memory();
    }
}
