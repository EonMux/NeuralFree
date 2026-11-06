package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Generation over precomputed embedding rows ({@code GenerationLoop.generate(EmbedsPrompt, ...)}): a prompt of
 * the token table's own rows generates the same tokens and text as the tokens themselves, a spliced prompt
 * (token runs with row runs between them) matches the all-token run of the same sequence, and a prompt whose
 * rows exceed the context refuses with its row count.
 */
public class GenerationLoopEmbedsTest {

    private static final int N_TOKENS = 6;
    private static final int N_NEW = 8;
    private static final float TOL = 1e-6f;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The token table's rows of {@code tokens}, read from the GGUF file. */
    private static float[][] tableRows(Path model, int[] tokens, int nEmbd) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
            GGUFTensorInfo table = parser.getTensor("token_embd.weight");
            float[] flat = parser.readTensorDataAsFloat(table);
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = Arrays.copyOfRange(flat, tokens[i] * nEmbd, (tokens[i] + 1) * nEmbd);
            }
            return rows;
        }
    }

    private static GenerationResult generate(Path model, EngineConfig cfg, GenerationRequest request,
                                             int[] tokens) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(m, cfg).generate(tokens, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    @Test
    public void aRowsOnlyPromptFollowsItsTokenTwin() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(false);
        int[] tokens = tokens(1);
        GenerationRequest request = request();
        GenerationResult twin = generate(model, cfg, request, tokens);

        GenerationResult result;
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            float[][] rows = tableRows(model, tokens, m.embeddingLength());
            result = GenerationLoop.forModel(m, cfg).generate(EmbedsPrompt.ofRows(rows), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
        assertArrayEquals("the rows of the table decode like the tokens they came from",
                twin.tokens(), result.tokens());
        assertEquals(twin.text(), result.text());
        assertEquals(twin.stopReason(), result.stopReason());
    }

    @Test
    public void aSplicedPromptFollowsTheAllTokenRun() throws IOException {
        // the single-sequence loop runs on the contiguous memory (the batch engine covers the paged one)
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(false);
        int[] tokens = tokens(2);
        GenerationRequest request = request();
        GenerationResult twin = generate(model, cfg, request, tokens);

        GenerationResult result;
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            float[][] rows = tableRows(model, Arrays.copyOfRange(tokens, 2, 5), m.embeddingLength());
            EmbedsPrompt prompt = EmbedsPrompt.of(
                    new int[][] {Arrays.copyOfRange(tokens, 0, 2), Arrays.copyOfRange(tokens, 5, 6)},
                    new float[][][] {rows});
            result = GenerationLoop.forModel(m, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
        assertArrayEquals("rows spliced between token runs fill the positions they occupy",
                twin.tokens(), result.tokens());
        assertEquals(twin.text(), result.text());
    }

    @Test
    public void promptLogprobsOverRowsRefuseByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(false);
        int[] tokens = tokens(1);
        GenerationRequest request = request().toBuilder().promptLogprobs(1).build();
        UnsupportedOperationException refused = assertThrows(UnsupportedOperationException.class, () -> {
            try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
                float[][] rows = tableRows(model, tokens, m.embeddingLength());
                GenerationLoop.forModel(m, cfg).generate(EmbedsPrompt.ofRows(rows), request,
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
            }
        });
        assertTrue(refused.getMessage().contains("prompt_logprobs"));
    }

    @Test
    public void oversizeRowsRefuseWithTheRowCount() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(false);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            float[][] rows = new float[65][m.embeddingLength()];
            for (float[] row : rows) {
                Arrays.fill(row, 0.5f);
            }
            ContextFullException e = assertThrows(ContextFullException.class, () ->
                    GenerationLoop.forModel(m, cfg).generate(EmbedsPrompt.ofRows(rows), request(),
                            (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertEquals(65, e.requiredPositions());
            assertEquals(64, e.contextLength());
        }
    }

    @Test
    public void aRowsPromptPrefillsEveryRowOfItsRuns() throws IOException {
        // the spliced prompt's rows occupy positions 2..4, so the decode starts at position 6 like the twin
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(false);
        int[] tokens = tokens(4);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            float[][] rows = tableRows(model, Arrays.copyOfRange(tokens, 1, 3), m.embeddingLength());
            EmbedsPrompt prompt = EmbedsPrompt.of(
                    new int[][] {Arrays.copyOfRange(tokens, 0, 1), Arrays.copyOfRange(tokens, 3, 6)},
                    new float[][][] {rows});
            GenerationResult result = GenerationLoop.forModel(m, cfg).generate(prompt, request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertTrue("the generation ran", result.tokenCount() > 0);
            assertEquals(N_TOKENS + result.tokenCount() - 1,
                    m.forwardBackend().memory().sequence().nPast());
        }
    }
}
