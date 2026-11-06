package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.parsed;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The surfaces merged after the router was written, over the real engine: speech ({@code /tts},
 * {@code /v1/audio/speech}), slots, LoRA adapters, the Responses API, the lifecycle routes and operations follow
 * the model they name in router mode, the router's own reports answer where no model is named, and an unknown
 * name is a 404 as everywhere else in the router.
 */
class RouterSurfacesResourceTest {

    @TempDir
    static Path dir;

    private static ModelRouter router;
    private static ModelRouter speechRouter;
    private static EngineService unloaded;
    private static EngineService speechUnloaded;
    private static TtsResource tts;
    private static TtsResource speechTts;
    private static SlotsResource slots;
    private static LoraResource loras;
    private static ResponsesResource responses;
    private static SleepResource sleep;
    private static OpsResource ops;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config llama = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Files.copy(SyntheticModels.llama(llama).writeTempFile(), dir.resolve("a.gguf"));
        Files.copy(SyntheticModels.llama(llama).writeTempFile(), dir.resolve("b.gguf"));
        Path speechDir = Files.createDirectories(dir.resolve("speech-models"));
        Files.copy(SyntheticModels.wavtokenizer(8000, 8, 2).writeTempFile(),
                speechDir.resolve("speech.gguf"));

        ServerOptions options = new ServerOptions(null, Duration.ofMinutes(2), false, 8,
                Files.createDirectories(dir.resolve("slots")),
                ServerOptions.DEFAULT_SLEEP_IDLE_SECONDS, 1, null, false, null, null, null,
                List.of("get_info"), null, null, null, null);
        EngineConfig config = pagedConfig(1024, 2);
        unloaded = new EngineService(null, config, options);
        router = new ModelRouter(dir, null, 0, 0, true, config, options, ModelRouter.systemDeviceMemory());

        EngineConfig speechConfig = EngineConfig.builder().backend(ServerTestSupport.backend())
                .sourceOptions(ModelSourceOptions.builder().warmup(false).build()).build();
        ServerOptions speechOptions = new ServerOptions(null, Duration.ofMinutes(2));
        speechUnloaded = new EngineService(null, speechConfig, speechOptions);
        speechRouter = new ModelRouter(speechDir, null, 0, 0, true, speechConfig, speechOptions,
                ModelRouter.systemDeviceMemory());

        tts = new TtsResource(unloaded, router);
        speechTts = new TtsResource(speechUnloaded, speechRouter);
        slots = new SlotsResource(unloaded, router);
        loras = new LoraResource(unloaded, router);
        responses = new ResponsesResource(unloaded, router);
        sleep = new SleepResource(unloaded, router, DevMode.ON, SleepModeSwitch.ON);
        Path profileDir = Files.createDirectories(dir.resolve("profile"));
        ops = new OpsResource(unloaded, new EngineProfiler(profileDir), router, DevMode.ON,
                ProfilerConfig.of(ProfilerConfig.TORCH, profileDir));
        openAi = new OpenAiResource(unloaded, router);
    }

    @AfterAll
    static void close() {
        router.close();
        speechRouter.close();
        unloaded.close();
        speechUnloaded.close();
    }

    @Test
    @Timeout(120)
    void speechServesOnTheModelItNames() {
        Response wav = speechTts.speech("{\"model\": \"speech\", \"codes\": [1, 2, 3]}");
        assertEquals(200, wav.getStatus());
        assertTrue(((byte[]) wav.getEntity()).length > 44, "a WAV header and samples");
        assertEquals(200, speechTts.tts("{\"model\": \"speech\", \"codes\": [3, 4]}").getStatus(),
                "the llama.cpp-native alias routes the same");
        assertEquals(400, statusOf(() -> speechTts.speech("{\"codes\": [1]}")), "the model is required");
        assertTrue(errorOf(() -> speechTts.speech("{\"codes\": [1]}")).contains("model"));
        assertEquals(404, statusOf(() -> speechTts.speech("{\"model\": \"zzz\", \"codes\": [1]}")));
        assertEquals(501, statusOf(() -> tts.speech("{\"model\": \"a\", \"codes\": [1]}")),
                "a routed model without a speech decoder refuses by name");
    }

    @Test
    @Timeout(120)
    @SuppressWarnings("unchecked")
    void slotsFollowTheModelQuery() {
        // the pool's engine builds on the first request, as it does in the single-model server
        openAi.completions("{\"model\": \"a\", \"prompt\": \"hi\", \"max_tokens\": 1}");
        List<Object> list = (List<Object>) parsed(slots.slots(null, "a", null).getEntity());
        assertEquals(2, list.size(), "the routed model's own engine answers");
        assertEquals(200, slots.action(0, "erase", "a", null, "{}").getStatus(),
                "a slot action follows the model query");
        assertEquals(400, statusOf(() -> slots.action(0, "erase", null, null, "{}")),
                "the model is required");
        assertEquals(400, statusOf(() -> slots.slots(null, null, null)), "the model is required");
        assertEquals(404, statusOf(() -> slots.slots(null, "zzz", null)));
    }

    @Test
    @Timeout(120)
    void theAdapterRoutesFollowTheModelQuery() {
        assertEquals(List.of(), parsed(loras.list("a", null).getEntity()));
        assertEquals(400, statusOf(() -> loras.list(null, null)), "the model is required");
        assertEquals(404, statusOf(() -> loras.list("zzz", null)));
    }

    @Test
    @Timeout(180)
    void responsesRouteByModelAndFindTheirStoreOnRetrieval() {
        String body = "{\"model\": \"a\", \"input\": \"Hello\", \"store\": true,"
                + " \"max_output_tokens\": 4, \"temperature\": 0}";
        Response created;
        try {
            created = responses.create(body);
        } catch (jakarta.ws.rs.WebApplicationException e) {
            throw new AssertionError("create refused: " + errorOf(() -> responses.create(body)), e);
        }
        assertEquals(200, created.getStatus());
        JsonView response = view(created.getEntity());
        String id = response.str("id");
        assertTrue(id.startsWith("resp_"));
        assertEquals(200, responses.retrieve(id, false, null, "a", null).getStatus());
        assertEquals(200, responses.retrieve(id, false, null, null, null).getStatus(),
                "an unnamed id finds the store that holds it");
        assertEquals(400, statusOf(() -> responses.create("{\"input\": \"Hello\", \"store\": false}")),
                "the model is required");
        assertEquals(404, statusOf(() -> responses.retrieve(id, false, null, "b", null)),
                "another model's store does not hold the response");
    }

    @Test
    @Timeout(120)
    void theLifecycleRoutesFollowTheModelQuery() {
        assertEquals(Boolean.FALSE, view(sleep.isSleeping("a", null).getEntity()).raw("is_sleeping"));
        assertEquals("sleeping", view(sleep.sleep("0", "abort", "a", null).getEntity()).str("status"));
        assertEquals(Boolean.TRUE, view(sleep.isSleeping("a", null).getEntity()).raw("is_sleeping"));
        assertEquals("awake", view(sleep.wakeUp(null, "a", null).getEntity()).str("status"));
        assertEquals("paused", view(sleep.pause("keep", null, "a", null).getEntity()).str("status"));
        assertEquals("resumed", view(sleep.resume("a", null).getEntity()).str("status"));
        assertEquals(400, statusOf(() -> sleep.sleep("0", "abort", null, null)), "the model is required");
        assertEquals(404, statusOf(() -> sleep.isSleeping("zzz", null)));
    }

    @Test
    @Timeout(180)
    void theOperationalRoutesFollowTheModelQuery() {
        JsonView tokenizer = view(ops.tokenizerInfo("a", null).getEntity());
        assertTrue(tokenizer.i64("vocab_size") > 0);
        assertEquals(Boolean.TRUE, view(ops.serverInfo(null, "a", null).getEntity()).raw("loaded"));
        JsonView report = view(ops.serverInfo(null, null, null).getEntity());
        assertEquals("router", report.str("role"));
        assertEquals(200, ops.ready().getStatus());
        assertEquals(200, ops.resetMmCache(null, null, "a", null).getStatus());
        assertTrue(view(ops.resetPrefixCache(false, false, "a", null).getEntity()).bool("success"));
        assertEquals(200, ops.startProfile(null, "a", null).getStatus());
        assertNotNull(view(ops.stopProfile("a", null).getEntity()));
        assertEquals(400, statusOf(() -> ops.tokenizerInfo(null, null)), "the model is required");
        assertEquals(404, statusOf(() -> ops.tokenizerInfo("zzz", null)));
    }

    @Test
    @Timeout(180)
    void theRoutedRequestTouchesTheListedLoadState() throws IOException {
        JsonView before = view(openAi.models(null).getEntity());
        JsonView entry = ServerTestSupport.at(before, "data", 0);
        assertNotNull(entry.raw("status"));
        JsonView first = view(openAi.chatCompletions("{\"model\": \"a\", \"messages\":"
                + " [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 2}").getEntity());
        assertFalse(first.str("model").isEmpty());
        JsonView after = view(openAi.models(null).getEntity());
        assertEquals("loaded", JsonView.of(ServerTestSupport.at(after, "data", 0).raw("status"), "status")
                .str("value"));
    }

    @Test
    @Timeout(120)
    void theControlRouteAnswersInRouterModeWithoutALoadedLifecycleService() {
        assertFalse(unloaded.loaded());
        assertEquals(400, statusOf(() -> openAi.control("{\"id\": \"chatcmpl-none\","
                + " \"action\": \"reasoning_end\"}")), "an unknown id is a named 400, not the unloaded 503");
    }

    @Test
    @Timeout(180)
    void theRouterLoadAndUnloadRoutesServeTheirModels() {
        ModelsResource models = new ModelsResource(router);
        boolean bRunning = router.list().stream().anyMatch(e -> e.name().equals("b") && e.running());
        if (bRunning) {
            models.unload("{\"model\": \"b\"}");
        }
        assertEquals(true, view(models.load("{\"model\": \"b\"}").getEntity()).raw("success"));
        assertEquals(400, statusOf(() -> models.load("{\"model\": \"b\"}")), "already running");
        assertEquals(200, models.unload("{\"model\": \"b\"}").getStatus());
        assertTrue(router.list().stream().anyMatch(e -> e.name().equals("b")
                && e.status() == ModelRouter.Status.UNLOADED));
    }

    @Test
    @Timeout(120)
    void theRouterReportsItsOwnLoadAndAborts() {
        assertEquals(0L, view(ops.load().getEntity()).i64("server_load"));
        assertEquals(0L, view(ops.abortRequests("{\"abort_all\": false}").getEntity()).i64("aborted"));
        assertTrue(view(ops.abortRequests(null).getEntity()).i64("aborted") >= 0);
    }
}
