package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Lookahead (Jacobi) decoding served by the batch engine: a request submitted to a lookahead service produces the
 * tokens the same request produces without lookahead, the engine reports its steps and confirmations, and the
 * sequence groups {@code W + G + 1} sequences per running request.
 */
class LookaheadServingTest {

    private static final int WINDOW = 2;
    private static final int NGRAM = 3;
    private static final int MAX_NGRAMS = 2;
    private static final int SEQUENCES = WINDOW + MAX_NGRAMS + 1;

    private static final String REQUEST = "{\"prompt\": \"hello there\", \"max_tokens\": 12,"
            + " \"temperature\": 0, \"return_token_ids\": true}";

    @Test
    void servedLookaheadProducesThePlainTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            EngineConfig plain = ServerTestSupport.config()
                    .memory(MemoryConfig.builder().nCtx(2048).ubatch(8).parallel(2).typeK(KvDType.F32)
                            .typeV(KvDType.F32).build())
                    .build();
            EngineConfig lookahead = ServerTestSupport.config()
                    .memory(MemoryConfig.builder().nCtx(2048).ubatch(8).parallel(SEQUENCES).kvUnified(true)
                            .typeK(KvDType.F32).typeV(KvDType.F32).build())
                    .lookaheadOptions(LookaheadOptions.builder().enabled(true).window(WINDOW).ngramSize(NGRAM)
                            .maxNgrams(MAX_NGRAMS).build())
                    .build();
            List<?> expected;
            List<?> actual;
            try (EngineService plainService = new EngineService(model.toString(), plain)) {
                expected = tokenIds(new OpenAiResource(plainService).completions(REQUEST));
            }
            try (EngineService lookaheadService = new EngineService(model.toString(), lookahead)) {
                actual = tokenIds(new OpenAiResource(lookaheadService).completions(REQUEST));
                assertTrue(lookaheadService.stats().lookaheadRounds() > 0,
                        "the served request ran lookahead steps: " + lookaheadService.stats());
                assertTrue(lookaheadService.stats().lookaheadEmitted() >= actual.size() - 1,
                        "the lookahead steps emitted the served tokens after the prefill's: "
                                + lookaheadService.stats());
            }
            assertEquals(12, actual.size(), "the token budget is spent");
            assertEquals(expected, actual, "lookahead serving keeps the plain greedy tokens");
        } finally {
            Files.deleteIfExists(model);
        }
    }

    /** The generated token ids of a completion response's single choice. */
    private static List<?> tokenIds(jakarta.ws.rs.core.Response response) {
        return (List<?>) ServerTestSupport.at(ServerTestSupport.view(response.getEntity()), "choices", 0)
                .raw("token_ids");
    }
}
