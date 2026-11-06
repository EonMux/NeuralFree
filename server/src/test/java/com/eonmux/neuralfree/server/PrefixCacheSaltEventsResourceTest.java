package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The prefix cache's {@code cache_salt} partition and its KV cache events through the real service, over pages of
 * four rows: a prompt reuses only the cached blocks of its own salt on the completions and chat endpoints (the
 * unsalted pool is one more partition), a malformed salt refuses by name, and the publisher
 * {@code neuralfree.kv.events} names receives the stores — the salt in the first block's extra keys — and the
 * clear of a prefix-cache reset, stamped with the global data-parallel rank the replica serves.
 */
public class PrefixCacheSaltEventsResourceTest {

    private static EngineService service;
    private static OpenAiResource openAi;
    private static OpsResource ops;

    private static Path model;
    private static EngineConfig config;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        config = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(256).ubatch(8)
                .kvPaged(true).kvPageSize(4).parallel(2).prefixReuse(true).typeK(KvDType.F32).typeV(KvDType.F32)
                .kvEvents(RecordingKvEventPublisher.class.getName()).build()).build();
        service = new EngineService(model.toString(), config);
        openAi = new OpenAiResource(service);
        ops = new OpsResource(service, new EngineProfiler(), DevMode.ON, ProfilerConfig.OFF);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @BeforeEach
    void freshCache() throws InterruptedException {
        assertTrue(service.resetPrefixCache());
        RecordingKvEventPublisher.BATCHES.clear();
    }

    private static JsonView usage(Response response) {
        assertEquals(200, response.getStatus(), String.valueOf(response.getEntity()));
        return view(response.getEntity()).obj("usage");
    }

    private static long cached(JsonView usage) {
        return usage.obj("prompt_tokens_details").i64("cached_tokens");
    }

    private static JsonView complete(String salt) {
        String field = salt == null ? "" : ", \"cache_salt\": \"" + salt + "\"";
        return usage(openAi.completions("{\"prompt\": \"the salted prefix of a shared prompt\", \"max_tokens\": 2,"
                + " \"temperature\": 0" + field + "}"));
    }

    @Test
    void aCompletionReusesOnlyTheBlocksOfItsOwnSalt() {
        JsonView first = complete("tenant-a");
        assertEquals(0, cached(first));
        assertEquals(0, cached(complete("tenant-b")), "another salt shares no block");
        assertEquals(0, cached(complete(null)), "the unsalted pool shares no salted block");
        JsonView same = complete("tenant-a");
        assertEquals(same.i64("prompt_tokens") - 1, cached(same), "the same salt reuses all but the last token");
        assertEquals(same.i64("prompt_tokens") - 1, cached(complete(null)), "and so does the unsalted repeat");
    }

    @Test
    void aChatReusesOnlyTheBlocksOfItsOwnSalt() {
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"a chat turn that repeats\"}],"
                + " \"max_tokens\": 2, \"temperature\": 0, \"cache_salt\": \"%s\"}";
        assertEquals(0, cached(usage(openAi.chatCompletions(body.formatted("chat-a")))));
        assertEquals(0, cached(usage(openAi.chatCompletions(body.formatted("chat-b")))));
        JsonView same = usage(openAi.chatCompletions(body.formatted("chat-a")));
        assertEquals(same.i64("prompt_tokens") - 1, cached(same));
    }

    @Test
    void aMalformedSaltRefusesByName() {
        String body = "{\"prompt\": \"x\", \"max_tokens\": 1, \"cache_salt\": \"a/b\"}";
        assertEquals(400, statusOf(() -> openAi.completions(body)));
        assertTrue(errorOf(() -> openAi.completions(body)).contains("cache_salt"));
        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"max_tokens\": 1,"
                + " \"cache_salt\": \"" + "s".repeat(OpenAiRequests.MAX_CACHE_SALT + 1) + "\"}";
        assertEquals(400, statusOf(() -> openAi.chatCompletions(chat)));
    }

    @Test
    void theNamedPublisherReceivesTheStoresAndTheResetsClear() throws InterruptedException {
        complete("tenant-events");
        List<KvCacheEvent.BlockStored> stored = new ArrayList<>();
        for (KvEventBatch batch : RecordingKvEventPublisher.BATCHES) {
            assertEquals(0, batch.dataParallelRank(), "the replica stamps its rank");
            for (KvCacheEvent e : batch.events()) {
                assertTrue(e instanceof KvCacheEvent.BlockStored, "a first run only stores, got " + e);
                stored.add((KvCacheEvent.BlockStored) e);
            }
        }
        assertFalse(stored.isEmpty());
        KvCacheEvent.BlockStored head = stored.get(0);
        assertNull(head.parentBlockHashInt(), "the chain's first block has no parent");
        assertEquals(List.of(BlockExtraKey.salt("tenant-events")), head.extraKeys().get(0),
                "the salt keys the first block");
        assertEquals(4, head.blockSize());
        assertTrue(Set.of(KvCacheEvent.MEDIUM_CPU, KvCacheEvent.MEDIUM_GPU).contains(head.medium()),
                "the medium names the memory the replica opened on, got " + head.medium());
        assertTrue(stored.stream().allMatch(e -> head.medium().equals(e.medium())));
        for (int i = 1; i < stored.size(); i++) {
            long[] before = stored.get(i - 1).blockHashInts();
            assertEquals(before[before.length - 1], (long) stored.get(i).parentBlockHashInt(),
                    "each store extends the previous one");
        }

        assertEquals(200, ops.resetPrefixCache(false, false).getStatus());
        List<KvCacheEvent> all = new ArrayList<>();
        for (KvEventBatch batch : RecordingKvEventPublisher.BATCHES) {
            all.addAll(batch.events());
        }
        assertEquals(KvCacheEvent.AllBlocksCleared.INSTANCE, all.get(all.size() - 1), "the reset clears last");
        assertEquals(1, all.stream().filter(e -> e instanceof KvCacheEvent.AllBlocksCleared).count());
        assertTrue(all.stream().noneMatch(e -> e instanceof KvCacheEvent.BlockRemoved),
                "a clear drops the blocks without one removal each");
    }

    /**
     * vLLM's {@code kv_cache_report_mode}: a request whose {@code vllm_xargs} sets it to {@code full} publishes a
     * {@code BlockStored} for every whole cached block it reuses, while the default incremental mode publishes none.
     */
    @Test
    void aFullModeRequestReportsTheBlocksItReuses() {
        String plain = "{\"prompt\": \"the salted prefix of a shared prompt\", \"max_tokens\": 2,"
                + " \"temperature\": 0}";
        assertEquals(0, cached(usage(openAi.completions(plain))));
        List<Long> stored = new ArrayList<>();
        for (KvEventBatch batch : RecordingKvEventPublisher.BATCHES) {
            for (KvCacheEvent e : batch.events()) {
                if (e instanceof KvCacheEvent.BlockStored) {
                    for (long hash : ((KvCacheEvent.BlockStored) e).blockHashInts()) {
                        stored.add(hash);
                    }
                }
            }
        }
        assertFalse(stored.isEmpty(), "the first run stores the prompt's blocks");
        RecordingKvEventPublisher.BATCHES.clear();

        String full = "{\"prompt\": \"the salted prefix of a shared prompt\", \"max_tokens\": 2,"
                + " \"temperature\": 0, \"vllm_xargs\": {\"kv_cache_report_mode\": \"full\"}}";
        JsonView second = usage(openAi.completions(full));
        assertTrue(cached(second) > 0, "the repeat reuses the cached prefix");
        KvCacheEvent.BlockStored report = null;
        for (KvEventBatch batch : RecordingKvEventPublisher.BATCHES) {
            for (KvCacheEvent e : batch.events()) {
                if (e instanceof KvCacheEvent.BlockStored && ((KvCacheEvent.BlockStored) e).parentBlockHashInt()
                        == null) {
                    report = (KvCacheEvent.BlockStored) e;
                }
            }
        }
        assertNotNull(report, "the full mode reports the reused blocks");
        assertEquals(4, report.blockSize());
        for (int k = 0; k < report.blockHashInts().length; k++) {
            assertEquals(stored.get(k), (Long) report.blockHashInts()[k],
                    "the reused block " + k + " is the stored one");
        }

        RecordingKvEventPublisher.BATCHES.clear();
        assertEquals(cached(second), cached(usage(openAi.completions(plain))),
                "the incremental repeat reuses the same prefix");
        for (KvEventBatch batch : RecordingKvEventPublisher.BATCHES) {
            for (KvCacheEvent e : batch.events()) {
                assertFalse(e instanceof KvCacheEvent.BlockStored
                                && ((KvCacheEvent.BlockStored) e).parentBlockHashInt() == null,
                        "the default mode reports no reused block");
            }
        }
    }

    @Test
    void anExternalLbReplicaStampsItsGlobalRank() throws IOException {
        ServerOptions rankTwo = new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 1, null)
                .withDataParallel(new DpLayout(4, 2, LoadBalance.SHORTEST_QUEUE));
        try (EngineService ranked = new EngineService(model.toString(), config, rankTwo)) {
            assertEquals(200, new OpenAiResource(ranked).completions("{\"prompt\": \"a prompt served at rank two\","
                    + " \"max_tokens\": 1, \"temperature\": 0}").getStatus());
        }
        assertFalse(RecordingKvEventPublisher.BATCHES.isEmpty());
        for (KvEventBatch batch : RecordingKvEventPublisher.BATCHES) {
            assertEquals(2, batch.dataParallelRank(), "the replica stamps the global rank it serves");
        }
    }
}
