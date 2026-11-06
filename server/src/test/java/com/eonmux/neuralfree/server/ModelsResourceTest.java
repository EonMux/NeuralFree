package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The router endpoints over the real engine ({@link ModelsResource} and the routed serving endpoints): the list,
 * the load and unload flows with their named refusals, the registration and removal, the SSE frames of the state
 * changes, and a request that names its model serves on that model's own engine and carries its name back.
 */
class ModelsResourceTest {

    private static EngineConfig config;
    private static EngineService unloaded;
    private static Path registeredPath;

    @TempDir
    Path dir;

    private static ModelRouter router;
    private static ModelsResource resource;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        config = pagedConfig(512, 2);
        unloaded = new EngineService(null, config);
    }

    @AfterAll
    static void close() {
        unloaded.close();
    }

    @AfterEach
    void closeRouter() {
        if (router != null) {
            router.close();
            router = null;
        }
    }

    private void openRouter() throws IOException {
        Files.copy(SyntheticModels.llama().writeTempFile(), dir.resolve("a.gguf"));
        Files.copy(SyntheticModels.llama().writeTempFile(), dir.resolve("b.gguf"));
        Files.createDirectories(dir.resolve("extra"));
        registeredPath = dir.resolve("extra").resolve("registered.gguf");
        Files.copy(SyntheticModels.llama().writeTempFile(), registeredPath);
        router = new ModelRouter(dir, null, 0, 0, true, config, new ServerOptions(null, Duration.ofSeconds(30)),
                ModelRouter.systemDeviceMemory());
        resource = new ModelsResource(router);
        openAi = new OpenAiResource(unloaded, router);
    }

    // ------------------------------------------------------------------ the list

    @Test
    @Timeout(180)
    void theListCarriesEveryEntryWithItsStatus() throws IOException {
        openRouter();
        Response response = openAi.models(null);
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals("list", body.str("object"));
        JsonView a = ServerTestSupport.at(body, "data", 0);
        assertEquals("a", a.str("id"));
        assertEquals("model", a.str("object"));
        assertEquals("neuralfree", a.str("owned_by"));
        assertEquals("unloaded", JsonView.of(a.raw("status"), "status").str("value"));
        assertEquals("models_dir", a.str("source"));
        assertEquals(false, a.raw("can_remove"));
    }

    // ------------------------------------------------------------------ load and unload

    @Test
    @Timeout(180)
    void loadAndUnloadRoundTripWithTheirRefusals() throws IOException {
        openRouter();
        assertEquals(true, view(resource.load("{\"model\":\"a\"}").getEntity()).raw("success"));
        assertEquals(400, statusOf(() -> resource.load("{\"model\":\"a\"}")), "already running");
        assertTrue(errorOf(() -> resource.load("{\"model\":\"a\"}")).contains("already"));
        assertEquals(404, statusOf(() -> resource.load("{\"model\":\"zzz\"}")));
        assertEquals(400, statusOf(() -> resource.load("{\"model\":\"\"}")));
        assertEquals(400, statusOf(() -> resource.load("{\"nomodel\":true}")));

        assertEquals(200, resource.unload("{\"model\":\"a\"}").getStatus());
        assertEquals(400, statusOf(() -> resource.unload("{\"model\":\"a\"}")), "not running");
        assertTrue(errorOf(() -> resource.unload("{\"model\":\"a\"}")).contains("not running"));
        assertEquals(404, statusOf(() -> resource.unload("{\"model\":\"zzz\"}")));
        assertEquals(400, statusOf(() -> resource.unload("[]")));
    }

    @Test
    @Timeout(180)
    void registerAndRemoveRoundTrip() throws IOException {
        openRouter();
        Response registered = resource.register("{\"model\":\"" + registeredPath + "\",\"alias\":[\"reg\"],"
                + "\"tags\":[\"mine\"]}");
        assertEquals(200, registered.getStatus());
        JsonView body = view(registered.getEntity());
        assertEquals("registered", body.str("model"));
        assertEquals(3, sizeOf(router));

        assertEquals(400, statusOf(() -> resource.register("{\"model\":\"" + registeredPath + "\"}")),
                "the name is taken");
        assertEquals(400, statusOf(() -> resource.register("{\"model\":\"" + dir.resolve("no.gguf") + "\"}")),
                "the file must exist");
        assertEquals(400, statusOf(() -> resource.register("{\"model\":\"" + dir.resolve("a.gguf")
                .getFileName() + "\"}")), "a directory name is no path to a file");

        assertEquals(200, resource.unregister("reg").getStatus());
        assertEquals(404, statusOf(() -> resource.unregister("reg")));
        assertEquals(400, statusOf(() -> resource.unregister("a")));
        assertTrue(errorOf(() -> resource.unregister("a")).contains(ModelRouter.DIR_PROPERTY));
        assertEquals(400, statusOf(() -> resource.unregister(null)));
    }

    private static int sizeOf(ModelRouter router) {
        return router.list().size();
    }

    // ------------------------------------------------------------------ the events

    @Test
    @Timeout(180)
    void theSseStreamCarriesTheStateChanges() throws Exception {
        openRouter();
        Response response = resource.sse();
        assertEquals("text/event-stream", response.getMediaType().toString());
        StreamingOutput stream = (StreamingOutput) response.getEntity();
        router.load("a");
        router.unload("a");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        // the stream runs until its client goes away; the test reads the frames and then disconnects it
        Thread writer = Thread.ofPlatform().daemon(true).start(() -> {
            try {
                stream.write(out);
            } catch (IOException interrupted) {
                // the disconnect
            }
        });
        try {
            long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
            while (!out.toString(StandardCharsets.UTF_8).contains("\"status\":\"unloaded\"")
                    && System.nanoTime() < until) {
                TimeUnit.MILLISECONDS.sleep(20);
            }
        } finally {
            writer.interrupt();
            writer.join(TimeUnit.SECONDS.toMillis(5));
        }
        String frames = out.toString(StandardCharsets.UTF_8);
        assertTrue(frames.contains("\"model\":\"a\""), frames);
        assertTrue(frames.contains("\"event\":\"model_status\""), frames);
        assertTrue(frames.contains("\"status\":\"loading\""), frames);
        assertTrue(frames.contains("\"status\":\"loaded\""), frames);
        assertTrue(frames.contains("\"status\":\"unloaded\""), frames);
    }

    // ------------------------------------------------------------------ the routed serving

    @Test
    @Timeout(180)
    void aChatCompletionServesOnTheModelItNames() throws IOException {
        openRouter();
        Response response = openAi.chatCompletions("""
                {"model": "b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0}""");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals("b", body.str("model"), "the answer names the routed model");
        assertEquals(ModelRouter.Status.LOADED, router.list().get(1).status());
        assertFalse(router.list().get(0).ready(), "only the named model opened");

        assertEquals("a", view(openAi.completions("""
                {"model": "a", "prompt": "hello", "max_tokens": 2, "temperature": 0}""").getEntity()).str("model"));
        assertEquals(404, statusOf(() -> openAi.chatCompletions("""
                {"model": "zzz", "messages": [{"role": "user", "content": "hi"}]}""")));
        assertEquals(400, statusOf(() -> openAi.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}]}""")));
    }

    @Test
    @Timeout(180)
    void theTokenizeAndCountRoutesServeTheNamedModel() throws IOException {
        openRouter();
        LlamaCppResource llama = new LlamaCppResource(unloaded, router);
        JsonView tokens = view(llama.tokenize("{\"model\": \"a\", \"content\": \"hello\"}").getEntity());
        assertNotNull(tokens.raw("tokens"), "the routed tokenizer answers");
        assertEquals(200, llama.detokenize("{\"model\": \"a\", \"tokens\": [1, 2, 3]}").getStatus());
        assertEquals(200, llama.applyTemplate("{\"model\": \"a\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}]}").getStatus());

        AnthropicResource anthropic = new AnthropicResource(unloaded, router);
        JsonView count = view(anthropic.countTokens("{\"model\": \"b\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}]}").getEntity());
        assertTrue(((Number) count.raw("input_tokens")).intValue() >= 1, "the routed model counts the prompt");
    }

    // ------------------------------------------------------------------ outside the router mode

    @Test
    @Timeout(60)
    void outsideTheRouterModeTheRoutesRefuseAndTheListStaysTheSingleModels() throws IOException {
        ModelRouter inactive = new ModelRouter(null, null, 0, 0, true, config,
                new ServerOptions(null, Duration.ofSeconds(2)), ModelRouter.systemDeviceMemory());
        ModelsResource plain = new ModelsResource(inactive);
        OpenAiResource plainAi = new OpenAiResource(unloaded, inactive);
        try {
            assertEquals(400, statusOf(() -> plain.load("{\"model\":\"a\"}")));
            assertTrue(errorOf(() -> plain.load("{\"model\":\"a\"}")).contains(ModelRouter.DIR_PROPERTY));
            assertEquals(400, statusOf(() -> plain.unload("{\"model\":\"a\"}")));
            assertEquals(400, statusOf(() -> plain.register("{\"model\":\"a.gguf\"}")));
            assertEquals(400, statusOf(() -> plain.unregister("a")));
            assertEquals(400, statusOf(plain::sse));
            assertEquals(503, statusOf(plainAi::modelsAlias), "the single-model list refuses while unloaded");
            assertEquals(unloaded, ModelRouter.target(inactive, unloaded,
                    ServerJson.parse("{\"model\":\"a\"}"), null), "the fallback serves");
        } finally {
            inactive.close();
        }
    }
}
