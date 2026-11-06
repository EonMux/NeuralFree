package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * the two engine halves of disaggregated serving: a prefill engine exports the chain its one-token generation
 * leaves ({@code exportKv}), and a fresh decode engine imports it ({@code importKv}), adopts the prefilled rows
 * (all but the last prompt token, which the decode computes) and generates exactly what the same engine generates
 * without the transfer. A chain of another page layout refuses by name, and so does exporting media rows.
 */
public class BatchKvTransferTest {

    private static final int PAGE = 4;
    private static final int N_NEW = 5;

    private static EngineConfig config(int pageSize) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2).kvPageSize(pageSize)
                        .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build())
                .build();
    }

    private static GenerationRequest request(int newTokens) {
        return GenerationRequest.builder().maxNewTokens(newTokens).sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** Fifteen prompt tokens: three whole blocks of four, then a partial block. */
    private static int[] prompt() {
        int[] tokens = new int[15];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = 4 + (i * 37) % 250;
        }
        return tokens;
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt, int newTokens)
            throws InterruptedException {
        BatchSequence sequence = engine.submit(prompt, request(newTokens), (TokenCallback) null);
        engine.drain();
        assertEquals(BatchSequence.State.DONE, sequence.state());
        return sequence;
    }

    @Test
    public void aPrefillExportFeedsAFreshDecodeEngine() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = prompt();
        int[] expected;
        byte[] chain;
        EngineConfig config = config(PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine prefill = BatchEngine.forModel(m, config)) {
            run(prefill, prompt, 1);
            chain = prefill.exportKv(0);
            assertTrue(chain.length > 0);
            // the same engine prefills the prompt again from its own cache: the reference the decode must match
            expected = run(prefill, prompt, N_NEW).result().tokens();
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine decode = BatchEngine.forModel(m, config)) {
            assertEquals("the chain holds every prompt row", prompt.length, decode.importKv(chain));
            BatchSequence decoded = run(decode, prompt, N_NEW);
            assertEquals("the decode adopts the prefilled rows and computes the last prompt token",
                    prompt.length - 1, decoded.reusedPrefix());
            assertArrayEquals("the transferred rows decode exactly like an untransferred run",
                    expected, decoded.result().tokens());
            assertTrue("the decode engine exports its own history", decode.exportKv(decoded.seq()).length > 0);
        }
    }

    @Test
    public void aChainOfAnotherLayoutRefusesByName() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        byte[] chain;
        EngineConfig four = config(PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), four);
             BatchEngine engine = BatchEngine.forModel(m, four)) {
            run(engine, prompt(), 1);
            chain = engine.exportKv(0);
        }
        EngineConfig eight = config(8);
        try (GGUFModel m = new GGUFModel(model.toString(), eight);
             BatchEngine engine = BatchEngine.forModel(m, eight)) {
            IOException mismatch = assertThrows(IOException.class, () -> engine.importKv(chain));
            assertTrue(mismatch.getMessage(), mismatch.getMessage().contains("page size"));
        }
    }

    @Test
    public void mediaRowsCannotLeaveTheMemory() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = config(PAGE);
        float[][] rows = new float[2][64];
        for (int r = 0; r < rows.length; r++) {
            for (int c = 0; c < rows[r].length; c++) {
                rows[r][c] = 0.01f * (r + 1) * (c + 1);
            }
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            engine.submit(com.eonmux.neuralfree.gguf.engine.MediaPrompt.builder()
                    .addText(new int[] {4, 11, 18})
                    .addEmbeds(3, rows)
                    .build(), request(1), (TokenCallback) null);
            engine.drain();
            UnsupportedOperationException refused = assertThrows(UnsupportedOperationException.class,
                    () -> engine.exportKv(0));
            assertTrue(refused.getMessage(), refused.getMessage().contains("media"));
        }
    }

    @Test
    public void importIntoARunningSlotRefuses() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = config(PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            run(engine, prompt(), 1);
            byte[] chain = engine.exportKv(0);
            engine.submit(prompt(), request(2), (TokenCallback) null);
            engine.submit(prompt(), request(2), (TokenCallback) null);
            engine.pump();
            assertNotEquals(0, engine.running());
            assertThrows(IllegalStateException.class, () -> engine.importKv(0, chain));
            assertThrows(IllegalStateException.class, () -> engine.importKv(chain));
            engine.drain();
            assertEquals(prompt().length, engine.importKv(chain));
        }
    }

    @Test
    public void importRefusesAnEmptyChain() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = config(PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            assertThrows(IOException.class, () -> engine.importKv(new byte[] {1, 2, 3}));
        }
    }
}
