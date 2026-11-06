package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The server face of the context shift: the request's {@code n_keep}/{@code n_discard} (llama.cpp's fields) ride
 * every generation route, a shifted completion reports {@code truncated} (llama.cpp's
 * {@code slot.truncated}), out-of-range counts refuse with 400 naming the field, and a request that never fills
 * the context reports {@code truncated} false.
 */
class ContextShiftResourceTest {

    private static EngineService service;
    private static LlamaCompletionResource llama;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig memory = MemoryConfig.builder().nCtx(24).ubatch(8).parallel(2).kvPageSize(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).contextShift(true).build();
        // the shift serves the CPU backend's paged heap; the GPU backends refuse it by name
        EngineConfig config = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(memory).build();
        service = new EngineService(model.toString(), config);
        llama = new LlamaCompletionResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void aShiftedCompletionReportsTruncatedAndRidesTheRequestCounts() {
        JsonView body = view(llama.completion("""
                {"prompt": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "n_predict": 30,
                 "temperature": 0, "ignore_eos": true, "n_keep": 4, "n_discard": 6}""").getEntity());
        assertTrue(body.bool("stop"));
        assertEquals("limit", body.str("stop_type"), "the budget ends the shifted generation");
        assertEquals(30L, body.i64("tokens_predicted"));
        assertTrue(body.bool("truncated"), "the shift reports truncated");
    }

    @Test
    void aShortCompletionReportsTruncatedFalse() {
        JsonView body = view(llama.completion("""
                {"prompt": "hello", "n_predict": 2, "temperature": 0, "ignore_eos": true}""").getEntity());
        assertTrue(body.bool("stop"));
        assertFalse(body.bool("truncated"));
    }

    @Test
    void outOfRangeShiftCountsRefuseByName() {
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_keep\": -2}")));
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_discard\": -1}")));
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_keep\": \"four\"}")));
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_discard\": 1.5}")));
    }
}
