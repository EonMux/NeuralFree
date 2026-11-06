package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The model's session API ({@code GGUFModel.saveSession} / {@code loadSession}, llama.cpp's
 * {@code llama_state_save_file} / {@code _load_file}): a session saved after one generation loads into a fresh
 * model whose next request with the same prompt re-evaluates only its last token over the restored rows and
 * decodes exactly what the original run did, and a paged memory refuses by name.
 */
public class GGUFModelSessionTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 6;

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt() {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i * 7 + 3) % 250;
        }
        return t;
    }

    @Test
    public void aSavedSessionLoadsIntoAFreshModelAndReusesItsTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path session = Files.createTempDirectory("nf-session-").resolve("cli.session");
        int[] prompt = prompt();
        int[] expected;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            expected = m.generate(prompt, request(), null).tokens();
            m.saveSession(session);
            assertTrue("the session file was written", Files.size(session) > 0);
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            m.loadSession(session);
            GenerationResult second = m.generate(prompt, request(), null);
            assertEquals("only the last prompt token evaluates over the restored rows", 1,
                    second.timings().promptTokens());
            assertArrayEquals("the restored session decodes exactly like the original run", expected,
                    second.tokens());
            // the loaded session keeps saving: the state after the second turn overwrites the file
            m.saveSession(session);
            try (GGUFModel again = new GGUFModel(model.toString(), config())) {
                again.loadSession(session);
                GenerationResult third = again.generate(prompt, request(), null);
                assertEquals("the saved-over file restores too", 1, third.timings().promptTokens());
                assertArrayEquals(expected, third.tokens());
            }
        }
    }

    @Test
    public void aPagedMemoryRefusesTheSessionByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path session = Files.createTempDirectory("nf-session-").resolve("paged.session");
        MemoryConfig paged = MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2).kvPageSize(4)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build();
        EngineConfig config = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(paged).build();
        try (GGUFModel m = new GGUFModel(model.toString(), config)) {
            UnsupportedOperationException save = assertThrows(UnsupportedOperationException.class,
                    () -> m.saveSession(session));
            assertTrue(save.getMessage(), save.getMessage().contains("paged"));
            UnsupportedOperationException load = assertThrows(UnsupportedOperationException.class,
                    () -> m.loadSession(session));
            assertTrue(load.getMessage(), load.getMessage().contains("paged"));
        }
    }
}
