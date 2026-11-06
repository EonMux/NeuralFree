package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.parsed;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Per-request LoRA adapters over HTTP: a startup module ({@code --lora-modules}) is listed by {@code /v1/models}
 * and {@code /lora-adapters}; a request selects it by its {@code model} (vLLM) or its {@code lora} list
 * (llama.cpp) and produces the text the same adapter attached to every row ({@code --lora}) produces; the global
 * scales of {@code POST /lora-adapters} apply to requests that select nothing; {@code /v1/load_lora_adapter} and
 * {@code /v1/unload_lora_adapter} change the adapters at runtime with vLLM's answers, and stay off unless enabled;
 * a load of an unusable file is refused by name, an unload of an adapter a running request uses answers 409 until
 * the request finishes, and requests of two adapters in flight run their own adapters; a full sleep's wake loads
 * the adapters again. The class runs with batch invariance on ({@link BatchInvariance}), so a request's text is
 * bit-identical whether it runs alone or beside another request and the adapter isolation claim is exact.
 */
class LoraResourceTest {

    private static final String BASE = "base";
    private static final String PROMPT = "The quick brown fox";

    private static Path adapterA;
    private static Path adapterB;
    private static EngineService service;
    private static EngineService referenceA;
    private static EngineService referenceB;

    @BeforeAll
    static void open() throws IOException {
        /* The isolation test compares a solo run with a run decoded beside another request; batch invariance makes
         * that comparison exact instead of tolerant. */
        BatchInvariance.setEnabled(true);
        Path model = SyntheticModels.llama().writeTempFile();
        adapterA = LoraAdapterFiles.llama(1000, 4, 8.0f);
        adapterB = LoraAdapterFiles.llama(5000, 4, 8.0f);
        ServerOptions options = new ServerOptions(BASE, ServerOptions.DEFAULT_TIMEOUT);
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2), options,
                new ServerLoras(List.of(new ServerLoras.Module("a", adapterA.toString())), true));
        referenceA = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2).toBuilder()
                .addLora(adapterA.toString(), 1.0f).build(), options);
        referenceB = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2).toBuilder()
                .addLora(adapterB.toString(), 1.0f).build(), options);
    }

    @AfterAll
    static void close() {
        service.close();
        referenceA.close();
        referenceB.close();
        BatchInvariance.clearOverride();
    }

    /** The greedy completion text of the base model's name with {@code extra} body fields. */
    private static String completion(EngineService on, String extra) {
        return completionAs(on, BASE, extra);
    }

    /** The greedy completion text of {@code model} with {@code extra} body fields; the response echoes it. */
    private static String completionAs(EngineService on, String model, String extra) {
        String body = "{\"model\": \"" + model + "\", \"prompt\": \"" + PROMPT + "\", \"max_tokens\": 6,"
                + " \"temperature\": 0" + extra + "}";
        JsonView response = view(new OpenAiResource(on).completions(body).getEntity());
        assertEquals(model, response.str("model"));
        return (String) ((Map<?, ?>) ((List<?>) response.raw("choices")).get(0)).get("text");
    }

    @Test
    void aStartupAdapterIsListedAsAModelAndAsALlamaCppAdapter() {
        JsonView models = view(new OpenAiResource(service).models(null).getEntity());
        assertEquals(2, size(models, "data"));
        assertEquals(BASE, at(models, "data", 0).str("id"));
        assertNull(at(models, "data", 0).raw("parent"));
        assertEquals("a", at(models, "data", 1).str("id"));
        assertEquals(adapterA.toString(), at(models, "data", 1).str("root"));
        assertEquals(BASE, at(models, "data", 1).str("parent"));

        List<?> adapters = (List<?>) parsed(new LoraResource(service).list().getEntity());
        assertEquals(1, adapters.size());
        Map<?, ?> first = (Map<?, ?>) adapters.get(0);
        assertEquals(0L, ((Number) first.get("id")).longValue());
        assertEquals(adapterA.toString(), first.get("path"));
        assertEquals(0.0, ((Number) first.get("scale")).doubleValue());
        assertEquals("a", first.get("name"));
    }

    @Test
    void aRequestNamingTheAdapterOrListingItRunsItAsAStaticAdapterWould() {
        String base = completion(service, "");
        String adapted = completionAs(service, "a", "");
        assertEquals(completion(referenceA, ""), adapted);
        assertNotEquals(base, adapted, "the adapter must change the text");
        assertEquals(adapted, completion(service, ", \"lora\": [{\"id\": 0, \"scale\": 1.0}]"));
        assertEquals(base, completion(service, ", \"lora\": [{\"id\": 0, \"scale\": 0}]"));
        assertEquals(base, completion(service, ", \"lora\": []"));

        String llama = "{\"prompt\": \"" + PROMPT + "\", \"n_predict\": 6, \"temperature\": 0,"
                + " \"lora\": [{\"id\": 0, \"scale\": 1.0}]}";
        assertEquals(adapted, view(new LlamaCompletionResource(service).completion(llama).getEntity())
                .str("content"));
    }

    @Test
    void anInvalidSelectionIsRefusedByName() {
        OpenAiResource openAi = new OpenAiResource(service);
        String unknown = errorOf(() -> openAi.completions("{\"prompt\": \"hi\", \"max_tokens\": 1,"
                + " \"lora\": [{\"id\": 7, \"scale\": 1}]}"));
        assertTrue(unknown.contains("lora") && unknown.contains("7"), unknown);
        String both = errorOf(() -> openAi.completions("{\"model\": \"a\", \"prompt\": \"hi\", \"max_tokens\": 1,"
                + " \"lora\": [{\"id\": 0, \"scale\": 1}]}"));
        assertTrue(both.startsWith("lora:"), both);
        assertEquals(400, statusOf(() -> openAi.completions("{\"prompt\": \"hi\", \"max_tokens\": 1,"
                + " \"lora\": {\"id\": 0}}")));
        assertEquals(404, statusOf(() -> openAi.completions("{\"model\": \"z\", \"prompt\": \"hi\","
                + " \"max_tokens\": 1}")));
    }

    @Test
    void theGlobalScalesApplyToRequestsThatSelectNothing() {
        LoraResource loras = new LoraResource(service);
        String base = completion(service, "");
        String adapted = completionAs(service, "a", "");
        try {
            assertEquals(Map.of("success", true), parsed(loras.setScales("[{\"id\": 0, \"scale\": 1.0}]")
                    .getEntity()));
            assertEquals(1.0, ((Number) ((Map<?, ?>) ((List<?>) parsed(loras.list().getEntity())).get(0))
                    .get("scale")).doubleValue());
            assertEquals(adapted, completion(service, ""));
            assertEquals(base, completion(service, ", \"lora\": [{\"id\": 0, \"scale\": 0}]"));
        } finally {
            loras.setScales("[]");
        }
        assertEquals(base, completion(service, ""));
        assertEquals(400, statusOf(() -> loras.setScales("{\"id\": 0}")));
        assertEquals(400, statusOf(() -> loras.setScales("[{\"id\": 3, \"scale\": 1}]")));
    }

    @Test
    void theRuntimeEndpointsLoadAndUnloadAdaptersWithVllmsAnswers() {
        LoraResource loras = new LoraResource(service);
        String load = "{\"lora_name\": \"b\", \"lora_path\": \"" + adapterB + "\"}";
        assertEquals("Success: LoRA adapter 'b' added successfully.", loras.load(load).getEntity());
        try {
            assertEquals(completion(referenceB, ""), completionAs(service, "b", ""));
            assertEquals(2, ((List<?>) parsed(loras.list().getEntity())).size());
            assertEquals(400, statusOf(() -> loras.load(load)));
            assertEquals(200, statusOf(() -> loras.load("{\"lora_name\": \"b\", \"lora_path\": \"" + adapterB
                    + "\", \"load_inplace\": true}")));
        } finally {
            assertEquals("Success: LoRA adapter 'b' removed successfully.",
                    loras.unload("{\"lora_name\": \"b\"}").getEntity());
        }
        assertEquals(404, statusOf(() -> loras.unload("{\"lora_name\": \"b\"}")));
        assertEquals(404, statusOf(() -> completionAs(service, "b", "")));
        assertEquals(404, statusOf(() -> loras.load("{\"lora_name\": \"c\", \"lora_path\": \"/no/such.gguf\"}")));
        assertEquals(400, statusOf(() -> loras.load("{\"lora_name\": \"c\"}")));
        String reserved = errorOf(() -> loras.load("{\"lora_name\": \"" + BASE + "\", \"lora_path\": \""
                + adapterB + "\"}"));
        assertTrue(reserved.contains("conflicts"), reserved);
        assertEquals(1, ((List<?>) parsed(loras.list().getEntity())).size());
    }

    @Test
    void aLoadOfAnUnusableAdapterFileIsRefusedNamingTheProblem() throws IOException {
        Path garbage = Files.createTempFile("nf-lora", ".gguf");
        Files.writeString(garbage, "this is not a GGUF file");
        try {
            WebApplicationException refused = assertThrows(WebApplicationException.class,
                    () -> new LoraResource(service).load("{\"lora_name\": \"g\", \"lora_path\": \""
                            + garbage + "\"}"));
            assertEquals(400, refused.getResponse().getStatus());
            String message = view(refused.getResponse().getEntity()).obj("error").str("message");
            assertTrue(message.contains("magic"), message);
        } finally {
            Files.deleteIfExists(garbage);
        }
        assertEquals(1, ((List<?>) parsed(new LoraResource(service).list().getEntity())).size());
    }

    @Test
    void anUnloadOfAnAdapterARunningRequestUsesIsRefusedUntilItFinishes() throws Exception {
        LoraResource loras = new LoraResource(service);
        assertEquals("Success: LoRA adapter 'b' added successfully.",
                loras.load("{\"lora_name\": \"b\", \"lora_path\": \"" + adapterB + "\"}").getEntity());
        Response stream = new OpenAiResource(service).chatCompletions(chatBody("b", 64, true));
        SignalledBuffer out = new SignalledBuffer();
        Thread generating = generate(stream, out);
        try {
            assertTrue(out.firstContent.await(30, TimeUnit.SECONDS), "the stream generated its first content");
            try {
                loras.unload("{\"lora_name\": \"b\"}");
                fail("unloading an adapter a running request uses must be refused");
            } catch (WebApplicationException refused) {
                assertEquals(409, refused.getResponse().getStatus());
                assertTrue(view(refused.getResponse().getEntity()).obj("error").str("message")
                        .contains("running request"), "the refusal names the running request");
            }
        } finally {
            generating.join(TimeUnit.SECONDS.toMillis(30));
        }
        assertFalse(generating.isAlive());
        assertEquals("Success: LoRA adapter 'b' removed successfully.",
                loras.unload("{\"lora_name\": \"b\"}").getEntity());
        assertEquals(404, statusOf(() -> completionAs(service, "b", "")));
    }

    @Test
    void requestsOfDifferentAdaptersInFlightTogetherRunTheirOwnAdapter() throws Exception {
        LoraResource loras = new LoraResource(service);
        assertEquals("Success: LoRA adapter 'b' added successfully.",
                loras.load("{\"lora_name\": \"b\", \"lora_path\": \"" + adapterB + "\"}").getEntity());
        try {
            String expectedA = chatText(referenceA, BASE, 24);
            String expectedB = chatText(referenceB, BASE, 24);
            assertNotEquals(expectedA, expectedB, "the adapters must change the text");
            Response stream = new OpenAiResource(service).chatCompletions(chatBody("a", 24, true));
            SignalledBuffer out = new SignalledBuffer();
            Thread generating = generate(stream, out);
            try {
                assertTrue(out.firstContent.await(30, TimeUnit.SECONDS), "the stream generated its first content");
                assertEquals(expectedB, chatText(service, "b", 24));
            } finally {
                generating.join(TimeUnit.SECONDS.toMillis(30));
            }
            assertFalse(generating.isAlive());
            assertEquals(expectedA, streamContent(out.toString(StandardCharsets.UTF_8)));
        } finally {
            loras.unload("{\"lora_name\": \"b\"}");
        }
        assertEquals(1, ((List<?>) parsed(loras.list().getEntity())).size());
    }

    @Test
    void theRuntimeEndpointsStayOffUnlessEnabled() throws IOException {
        try (EngineService off = new EngineService(null, ServerTestSupport.pagedConfig(64, 2))) {
            LoraResource loras = new LoraResource(off);
            String message = errorOf(() -> loras.load("{\"lora_name\": \"b\", \"lora_path\": \"x\"}"));
            assertTrue(message.contains(ServerLoras.RUNTIME_PROPERTY), message);
            assertEquals(404, statusOf(() -> loras.unload("{\"lora_name\": \"b\"}")));
        }
    }

    @Test
    void aFullSleepsWakeLoadsTheAdaptersAgain() throws InterruptedException {
        String adapted = completionAs(service, "a", "");
        service.sleep(2, SleepControl.Mode.ABORT);
        assertTrue(service.isSleeping());
        assertEquals("awake", service.wakeUp(List.of()));
        assertEquals(adapted, completionAs(service, "a", ""));
    }

    @Test
    void theLaunchFlagsMapToTheLoraProperties() {
        Map<String, String> p = ServerLaunchArgs.parse("--lora-modules", "a=" + adapterA,
                "{\"name\": \"b\", \"path\": \"" + adapterB + "\"}", "--max-loras", "3", "--max-lora-rank", "16",
                "--max-cpu-loras", "2", "--enable-lora").properties();
        assertEquals("a=" + adapterA + ",b=" + adapterB, p.get(ServerLoras.MODULES_PROPERTY));
        assertEquals("3", p.get("neuralfree.lora.max"));
        assertEquals("16", p.get("neuralfree.lora.max.rank"));
        assertEquals("2", p.get("neuralfree.lora.cache"));
        assertEquals("true", p.get(LoraConfig.PROP_ENABLE));
        assertNull(ServerLaunchArgs.parse("--max-loras", "3").properties().get(LoraConfig.PROP_ENABLE));
        assertEquals(List.of(new ServerLoras.Module("a", adapterA.toString()),
                new ServerLoras.Module("b", adapterB.toString())), ServerLoras.modules(
                p.get(ServerLoras.MODULES_PROPERTY)));
        assertTrue(errorMessage(() -> ServerLaunchArgs.parse("--lora-modules")).contains("--lora-modules"));
        assertTrue(errorMessage(() -> ServerLaunchArgs.parse("--lora-modules", "a=/no/such"))
                .contains("does not exist"));
        assertTrue(errorMessage(() -> ServerLaunchArgs.parse("--lora-modules",
                "{\"name\": \"a\", \"path\": \"" + adapterA + "\", \"base_model_name\": \"m\"}"))
                .contains("base_model_name"));
    }

    private static String errorMessage(Runnable call) {
        try {
            call.run();
        } catch (IllegalArgumentException e) {
            return e.getMessage();
        }
        throw new AssertionError("the call was not refused");
    }

    /** The chat-completions body of {@code model} generating {@code maxTokens} greedy tokens, streamed on
     * {@code stream}. */
    private static String chatBody(String model, int maxTokens, boolean stream) {
        return "{\"model\": \"" + model + "\", \"messages\": [{\"role\": \"user\", \"content\": \"" + PROMPT
                + "\"}], \"max_tokens\": " + maxTokens + ", \"temperature\": 0"
                + (stream ? ", \"stream\": true" : "") + "}";
    }

    /** The message content of a blocking chat completion of {@code on} for {@code model}; the response echoes it. */
    private static String chatText(EngineService on, String model, int maxTokens) {
        JsonView response = view(new OpenAiResource(on).chatCompletions(chatBody(model, maxTokens, false))
                .getEntity());
        assertEquals(model, response.str("model"));
        return at(response, "choices", 0).obj("message").str("content");
    }

    /** The concatenated {@code delta.content} texts of an OpenAI chat stream. */
    private static String streamContent(String stream) {
        StringBuilder text = new StringBuilder();
        for (JsonView frame : dataFrames(stream)) {
            if (!(frame.raw("choices") instanceof List<?> choices) || choices.isEmpty()) {
                continue;
            }
            JsonView delta = at(frame, "choices", 0).obj("delta");
            if (delta.has("content")) {
                text.append(delta.str("content"));
            }
        }
        return text.toString();
    }

    /** Writes {@code stream} on its own thread through {@code out}; join the thread to drain the stream. */
    private static Thread generate(Response stream, SignalledBuffer out) {
        Thread generating = new Thread(() -> {
            try {
                ((StreamingOutput) stream.getEntity()).write(out);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
        generating.start();
        return generating;
    }

    /** A stream buffer that signals {@code firstContent} once a written chunk carries content. */
    private static final class SignalledBuffer extends ByteArrayOutputStream {

        final CountDownLatch firstContent = new CountDownLatch(1);

        @Override
        public synchronized void write(byte[] b, int off, int len) {
            super.write(b, off, len);
            if (new String(b, off, len, StandardCharsets.UTF_8).contains("\"content\"")) {
                firstContent.countDown();
            }
        }
    }
}
