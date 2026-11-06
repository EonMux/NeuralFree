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
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The server face of self-extend: a request's {@code grp_attn_n}/{@code grp_attn_w} (llama.cpp's
 * {@code --grp-attn-n}/{-w}) group the generated positions, a grouped completion runs its whole budget, and a
 * malformed pair or one beside a context shift refuses with 400 naming the fields.
 */
class SelfExtendResourceTest {

    private static EngineService service;
    private static LlamaCompletionResource llama;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig memory = MemoryConfig.builder().nCtx(24).ubatch(8).parallel(2).kvPageSize(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        // group attention compresses positions through the CPU backend's paged heap
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
    void aGroupedCompletionRunsItsBudget() {
        JsonView body = view(llama.completion("""
                {"prompt": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "n_predict": 8,
                 "temperature": 0, "ignore_eos": true, "grp_attn_n": 2, "grp_attn_w": 8}""").getEntity());
        assertTrue(body.bool("stop"));
        assertEquals("limit", body.str("stop_type"));
        assertEquals(8L, body.i64("tokens_predicted"));
    }

    @Test
    void anUngroupedCompletionStillRuns() {
        JsonView body = view(llama.completion("""
                {"prompt": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "n_predict": 4,
                 "temperature": 0, "ignore_eos": true}""").getEntity());
        assertTrue(body.bool("stop"));
        assertEquals(4L, body.i64("tokens_predicted"));
    }

    @Test
    void aMalformedGroupPairRefusesByName() {
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"grp_attn_n\": 0}")));
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"grp_attn_w\": 0}")));
        // llama.cpp's ga_w % ga_n == 0; the default width is 512
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"grp_attn_n\": 3}")));
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"grp_attn_n\": 2, \"grp_attn_w\": \"eight\"}")));
    }

    @Test
    void aGroupPairBesideAContextShiftRefusesByName() {
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"grp_attn_n\": 2, \"grp_attn_w\": 8, \"n_keep\": 4}")));
    }

    @Test
    void chunkReuseBeyondTheCommonPrefixParsesByName() {
        // 0 (the default) and a positive count parse; the engine refuses what it cannot serve by name
        assertEquals(200, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_predict\": 1, \"n_cache_reuse\": 0}")));
        assertEquals(200, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_predict\": 1, \"n_cache_reuse\": 256}")));
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_cache_reuse\": \"many\"}")));
        assertEquals(400, statusOf(() -> llama.completion(
                "{\"prompt\": \"hi\", \"n_cache_reuse\": -4}")));
    }
}
