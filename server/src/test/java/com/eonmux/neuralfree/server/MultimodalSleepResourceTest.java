package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.function.Supplier;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The media cache and the projector lifecycle over the real service with a llava projector beside the text model:
 * vLLM's media part {@code uuid} lets a repeated image come as the uuid alone on the chat and Responses routes with
 * the same prompt and answer, a uuid the cache does not hold and a uuid on precomputed embeddings are refused by
 * name, {@code POST /reset_mm_cache} and a sleep of level 1 drop the cached encodings, a sleep of level 2 releases
 * the projector with the model weights and {@code /wake_up} reopens it with the same answers, and the idle sleep
 * releases it until the next request wakes the pool.
 *
 * <p>The tests run in a declared order: the cache tests run before the sleeps that drop the cache.</p>
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class MultimodalSleepResourceTest {

    private static EngineService service;
    private static EngineService idle;
    private static OpenAiResource openai;
    private static SleepResource sleep;
    private static String dataUrl;
    private static int width;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        width = SyntheticModels.Config.llama().nEmbd();
        Path mmproj = MmprojTestFiles.llava(width);
        dataUrl = "data:image/png;base64,"
                + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(12, 8));
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        idle = new EngineService(model.toString(), config,
                new ServerOptions(null, Duration.ofSeconds(60), false, 0, 1), MediaOptions.of(mmproj.toString()));
        openai = new OpenAiResource(service);
        sleep = new SleepResource(service, DevMode.ON, SleepModeSwitch.ON);
    }

    @AfterAll
    static void close() {
        service.close();
        idle.close();
    }

    /** A greedy one-message chat of an {@code image_url} part: {@code url} (null for none) and {@code uuid}. */
    private static String chat(String url, String uuid) {
        String media = url == null ? "null" : "{\"url\": \"" + url + "\"}";
        String key = uuid == null ? "" : ", \"uuid\": \"" + uuid + "\"";
        return "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"describe\"},"
                + " {\"type\": \"image_url\", \"image_url\": " + media + key + "}]}], \"max_tokens\": 3,"
                + " \"temperature\": 0}";
    }

    /** The prompt rows and the answer of a chat completion that must succeed. */
    private static String answer(OpenAiResource over, String body) {
        JsonView response = view(ok(() -> over.chatCompletions(body)).getEntity());
        long prompt = JsonView.of(response.raw("usage"), "usage").i64("prompt_tokens");
        JsonView choice = JsonView.of(response.list("choices").get(0), "choice");
        String content = choice.obj("message").optStr("content", "");
        return prompt + " rows: " + content;
    }

    private static Response ok(Supplier<Response> call) {
        try {
            Response response = call.get();
            assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
            return response;
        } catch (WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
    }

    // ------------------------------------------------------------------ uuid

    @Test
    @Timeout(120)
    @Order(1)
    void aRepeatedUuidComesAloneWithTheSamePromptAndAnswer() {
        String first = answer(openai, chat(dataUrl, "cat"));
        assertTrue(service.media().cachedBytes() > 0, "the encoding is cached under its uuid");
        assertEquals(first, answer(openai, chat(null, "cat")));
        assertEquals(first, answer(openai, chat(dataUrl, null)), "the uuid serves what the image encodes to");

        ResponsesResource responses = new ResponsesResource(service);
        String body = "{\"input\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_image\", \"uuid\":"
                + " \"cat\"}]}], \"max_output_tokens\": 3, \"temperature\": 0}";
        ok(() -> responses.create(body));
        assertTrue(errorOf(() -> responses.create(body.replace("\"cat\"", "\"dog\"")))
                .contains("uuid 'dog'"));
        assertTrue(errorOf(() -> responses.create("{\"input\": [{\"role\": \"user\", \"content\": [{\"type\":"
                + " \"input_image\"}]}]}")).contains("requires 'image_url'"));
    }

    @Test
    @Timeout(120)
    @Order(2)
    void aUuidTheCacheDoesNotHoldOrOnEmbeddingsIsRefusedByName() {
        assertEquals(400, statusOf(() -> openai.chatCompletions(chat(null, "never-sent"))));
        String error = errorOf(() -> openai.chatCompletions(chat(null, "never-sent")));
        assertTrue(error.contains("messages[0].content[1].image_url"), error);
        assertTrue(error.contains("uuid 'never-sent' is not in the media cache"), error);

        StringBuilder row = new StringBuilder("[");
        for (int i = 0; i < width; i++) {
            row.append(i == 0 ? "" : ", ").append("0.0");
        }
        String embeds = "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_embeds\","
                + " \"image_embeds\": [" + row + "]], \"uuid\": \"rows\"}]}], \"max_tokens\": 2}";
        assertEquals(400, statusOf(() -> openai.chatCompletions(embeds)));
        assertTrue(errorOf(() -> openai.chatCompletions(embeds)).contains("'uuid' is not supported on precomputed"
                + " embeddings"));
    }

    @Test
    @Timeout(120)
    @Order(3)
    void resetMmCacheDropsTheCachedEncodings() {
        answer(openai, chat(dataUrl, "reset-me"));
        OpsResource ops = new OpsResource(service, new EngineProfiler(), DevMode.ON, ProfilerConfig.OFF);
        assertEquals(200, ops.resetMmCache(null, null).getStatus());
        assertEquals(0, service.media().cachedBytes());
        assertTrue(errorOf(() -> openai.chatCompletions(chat(null, "reset-me"))).contains("uuid 'reset-me'"));
    }

    // ------------------------------------------------------------------ sleep

    @Test
    @Timeout(120)
    @Order(4)
    void aLevelOneSleepDropsTheCachedEncodingsAndKeepsTheProjector() {
        answer(openai, chat(dataUrl, "level-1"));
        assertEquals("sleeping", view(sleep.sleep("1", "abort").getEntity()).str("status"));
        assertEquals(0, service.media().cachedBytes(), "vLLM's sleep of level 1 or more clears the media cache");
        assertFalse(service.media().released(), "level 1 keeps the weights, the projector's too");
        assertEquals("awake", view(sleep.wakeUp(null).getEntity()).str("status"));
    }

    @Test
    @Timeout(180)
    @Order(5)
    void aLevelTwoSleepReleasesTheProjectorAndTheWakeReopensIt() {
        String before = answer(openai, chat(dataUrl, null));
        assertEquals("sleeping", view(sleep.sleep("2", "abort").getEntity()).str("status"));
        assertFalse(service.loaded());
        assertTrue(service.media().released(), "the projector releases with the model weights");
        assertEquals(0, service.media().cachedBytes());
        JsonView modalities = view(new LlamaCppResource(service).props(null, null).getEntity()).obj("modalities");
        assertEquals(Boolean.TRUE, modalities.raw("vision"), "/props still reports the sleeping projector");
        assertEquals(503, statusOf(() -> openai.chatCompletions(chat(dataUrl, null))));

        assertEquals("awake", view(sleep.wakeUp(null).getEntity()).str("status"));
        assertFalse(service.media().released(), "the wake reopens the projector with the model");
        assertEquals(before, answer(openai, chat(dataUrl, null)));
    }

    @Test
    @Timeout(180)
    @Order(6)
    void theIdleSleepReleasesTheProjectorUntilARequestWakesThePool() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 30_000;
        while (!idle.media().released() && System.currentTimeMillis() < deadline) {
            Thread.sleep(50);
        }
        assertTrue(idle.isSleeping(), "the pool sleeps after its idle second");
        assertTrue(idle.media().released(), "the idle sleep releases the projector with the model");

        String answer = answer(new OpenAiResource(idle), chat(dataUrl, null));
        assertFalse(idle.media().released(), "the request that woke the pool reopened the projector");
        assertTrue(answer.length() > 0);
    }
}
