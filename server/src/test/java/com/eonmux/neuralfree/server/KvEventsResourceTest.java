package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code GET /kv_event_sources} over the real service: the map is empty when the pool publishes nothing, and every
 * served rank appears with its resolved publisher configuration, a ZMQ publisher's endpoints offset by the rank as
 * vLLM's per-rank publishers are.
 */
class KvEventsResourceTest {

    private static Path model;

    private EngineService service;

    @BeforeAll
    static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @AfterEach
    void close() {
        if (service != null) {
            service.close();
        }
    }

    private static EngineConfig config(String kvEvents) {
        return ServerTestSupport.config().memory(MemoryConfig.builder()
                .nCtx(64).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32)
                .kvEvents(kvEvents).build()).build();
    }

    @Test
    void aPoolThatPublishesNothingAnswersAnEmptyMap() throws IOException {
        service = new EngineService(model.toString(), config(null),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        JsonView sources = view(new KvEventsResource(service).sources().getEntity());
        assertTrue(sources.isEmpty(), "no rank is listed");
    }

    @Test
    void everyRankReportsItsPublisher() throws IOException {
        service = new EngineService(model.toString(), config("log"),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        JsonView sources = view(new KvEventsResource(service).sources().getEntity());
        assertEquals(2, sources.keys().size());
        assertEquals("log", sources.obj("0").str("publisher"));
        assertTrue(sources.obj("0").bool("enable_kv_cache_events"));
        assertEquals("log", sources.obj("1").str("publisher"));
        assertNull(sources.obj("1").raw("endpoint"), "a non-ZMQ publisher reports no transport");
    }

    @Test
    void aZmqPublisherReportsItsRankOffsetEndpoints() throws IOException {
        service = new EngineService(model.toString(), config("{\"enable_kv_cache_events\": true,"
                + " \"endpoint\": \"tcp://*:5560\", \"replay_endpoint\": \"tcp://*:5570\", \"topic\": \"kv\"}"),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        JsonView sources = view(new KvEventsResource(service).sources().getEntity());
        JsonView first = sources.obj("0");
        assertEquals("zmq", first.str("publisher"));
        assertEquals("tcp://*:5560", first.str("endpoint"));
        assertEquals("tcp://*:5570", first.str("replay_endpoint"));
        assertEquals("kv", first.str("topic"));
        assertEquals(10_000, first.i64("buffer_steps"));
        assertEquals(100_000, first.i64("hwm"));
        assertEquals(100_000, first.i64("max_queue_size"));
        JsonView second = sources.obj("1");
        assertEquals("tcp://*:5561", second.str("endpoint"), "the rank offsets the PUB port");
        assertEquals("tcp://*:5571", second.str("replay_endpoint"));
    }

    @Test
    void aDisabledConfigAnswersAnEmptyMap() throws IOException {
        service = new EngineService(model.toString(),
                config("{\"enable_kv_cache_events\": false, \"publisher\": \"zmq\"}"),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        assertTrue(view(new KvEventsResource(service).sources().getEntity()).isEmpty());
    }
}
