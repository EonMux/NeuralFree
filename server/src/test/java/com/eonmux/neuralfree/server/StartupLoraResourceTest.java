package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.parsed;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A startup adapter ({@code --lora}) beside per-request modules, as llama.cpp server serves it: {@code /lora-adapters}
 * lists it first at its startup scale with every adapter's PEFT metadata ({@code task_name}, {@code prompt_prefix},
 * and an activated LoRA's {@code alora_invocation_string} / {@code alora_invocation_tokens}); a request that selects
 * nothing runs it, a request's {@code lora} list rescales or disables it (unlisted at 0), the global scales apply to
 * it and survive a full sleep's wake, a pooling request selects the same way, and it is neither unloaded nor
 * replaced at runtime.
 */
class StartupLoraResourceTest {

    private static final String BASE = "base";
    private static final String PROMPT = "The quick brown fox";
    private static final int[] INVOCATION = {77, 91};

    private static Path startup;
    private static EngineService service;
    private static EngineService plain;
    private static EngineService referenceA;
    private static EngineService referenceB;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        startup = LoraAdapterFiles.llama(1000, 4, 8.0f);
        Path tasked = LoraAdapterFiles.tasked(5000, 4, 8.0f, "task-b", "prefix-b");
        Path alora = LoraAdapterFiles.activated(3000, 4, 8.0f, INVOCATION);
        ServerOptions options = new ServerOptions(BASE, ServerOptions.DEFAULT_TIMEOUT);
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2).toBuilder()
                .addLora(startup.toString(), 1.0f).build(), options,
                new ServerLoras(List.of(new ServerLoras.Module("b", tasked.toString()),
                        new ServerLoras.Module("al", alora.toString())), true));
        plain = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2), options);
        referenceA = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2).toBuilder()
                .addLora(startup.toString(), 1.0f).build(), options);
        referenceB = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2).toBuilder()
                .addLora(tasked.toString(), 1.0f).build(), options);
    }

    @AfterAll
    static void close() {
        service.close();
        plain.close();
        referenceA.close();
        referenceB.close();
    }

    private static String completion(EngineService on, String model, String extra) {
        String body = "{\"model\": \"" + model + "\", \"prompt\": \"" + PROMPT + "\", \"max_tokens\": 6,"
                + " \"temperature\": 0" + extra + "}";
        JsonView response = view(new OpenAiResource(on).completions(body).getEntity());
        return (String) ((Map<?, ?>) ((List<?>) response.raw("choices")).get(0)).get("text");
    }

    private static String completion(EngineService on, String extra) {
        return completion(on, BASE, extra);
    }

    private static List<?> adapters() {
        return (List<?>) parsed(new LoraResource(service).list().getEntity());
    }

    private static String startupName() {
        String file = startup.getFileName().toString();
        return file.substring(0, file.length() - ".gguf".length());
    }

    @Test
    void theStartupAdapterIsListedFirstWithEveryAdaptersMetadata() {
        List<?> list = adapters();
        assertEquals(3, list.size());
        Map<?, ?> first = (Map<?, ?>) list.get(0);
        assertEquals(0L, ((Number) first.get("id")).longValue());
        assertEquals(startup.toString(), first.get("path"));
        assertEquals(1.0, ((Number) first.get("scale")).doubleValue());
        assertEquals(startupName(), first.get("name"));
        assertEquals("", first.get("task_name"));
        assertEquals("", first.get("prompt_prefix"));
        assertFalse(first.containsKey("alora_invocation_tokens"), "a plain adapter lists no invocation");

        Map<?, ?> tasked = (Map<?, ?>) list.get(1);
        assertEquals("b", tasked.get("name"));
        assertEquals(0.0, ((Number) tasked.get("scale")).doubleValue());
        assertEquals("task-b", tasked.get("task_name"));
        assertEquals("prefix-b", tasked.get("prompt_prefix"));

        Map<?, ?> alora = (Map<?, ?>) list.get(2);
        assertEquals("al", alora.get("name"));
        assertEquals(List.of(77L, 91L), ((List<?>) alora.get("alora_invocation_tokens")).stream()
                .map(n -> ((Number) n).longValue()).toList());
        assertEquals(service.tokenText(77) + service.tokenText(91), alora.get("alora_invocation_string"));
    }

    @Test
    void requestsRunRescaleOrDisableTheStartupAdapter() {
        String withA = completion(referenceA, "");
        String bare = completion(plain, "");
        String onlyB = completion(referenceB, "");
        assertNotEquals(withA, bare, "the startup adapter must change the text");
        assertEquals(withA, completion(service, ""));
        assertEquals(bare, completion(service, ", \"lora\": [{\"id\": 0, \"scale\": 0}]"));
        assertEquals(withA, completion(service, ", \"lora\": [{\"id\": 0, \"scale\": 1}]"));
        assertEquals(onlyB, completion(service, ", \"lora\": [{\"id\": 1, \"scale\": 1}]"),
                "an unlisted startup adapter runs at 0");
        assertEquals(onlyB, completion(service, "b", ""));
        assertEquals(withA, completion(service, startupName(), ""));
    }

    @Test
    void theGlobalScalesRescaleTheStartupAdapter() {
        LoraResource loras = new LoraResource(service);
        String bare = completion(plain, "");
        try {
            loras.setScales("[{\"id\": 0, \"scale\": 0}]");
            assertEquals(0.0, ((Number) ((Map<?, ?>) adapters().get(0)).get("scale")).doubleValue());
            assertEquals(bare, completion(service, ""));
        } finally {
            loras.setScales("[{\"id\": 0, \"scale\": 1}]");
        }
        assertEquals(completion(referenceA, ""), completion(service, ""));
    }

    @Test
    void aFullSleepsWakeKeepsTheStartupAdapterFirstWithItsGlobalScale() throws InterruptedException {
        LoraResource loras = new LoraResource(service);
        String bare = completion(plain, "");
        try {
            loras.setScales("[{\"id\": 0, \"scale\": 0}, {\"id\": 1, \"scale\": 0}]");
            service.sleep(2, SleepControl.Mode.ABORT);
            assertEquals("awake", service.wakeUp(List.of()));
            List<?> list = adapters();
            assertEquals(3, list.size());
            assertEquals(startupName(), ((Map<?, ?>) list.get(0)).get("name"));
            assertEquals(0.0, ((Number) ((Map<?, ?>) list.get(0)).get("scale")).doubleValue());
            assertEquals("task-b", ((Map<?, ?>) list.get(1)).get("task_name"));
            assertEquals(bare, completion(service, ""));
        } finally {
            loras.setScales("[{\"id\": 0, \"scale\": 1}]");
        }
        assertEquals(completion(referenceA, ""), completion(service, ""));
    }

    @Test
    void theStartupAdapterIsNeitherUnloadedNorReplaced() {
        LoraResource loras = new LoraResource(service);
        String unload = errorOf(() -> loras.unload("{\"lora_name\": \"" + startupName() + "\"}"));
        assertTrue(unload.contains("startup adapter"), unload);
        assertEquals(400, statusOf(() -> loras.unload("{\"lora_name\": \"" + startupName() + "\"}")));
        assertEquals(400, statusOf(() -> loras.load("{\"lora_name\": \"" + startupName() + "\", \"lora_path\": \""
                + startup + "\", \"load_inplace\": true}")));
        assertEquals(3, adapters().size());
        assertEquals(completion(referenceA, ""), completion(service, ""));
    }

    @Test
    void generativeScoringSelectsAdaptersLikeGeneration() {
        String body = "{\"query\": \"" + PROMPT + "\", \"items\": [\" jumps\"], \"label_token_ids\": [5, 9]";
        double withA = score(referenceA, body + "}");
        double bare = score(plain, body + "}");
        assertNotEquals(withA, bare, "the startup adapter must change the score");
        assertEquals(withA, score(service, body + "}"), 1e-6);
        assertEquals(bare, score(service, body + ", \"lora\": [{\"id\": 0, \"scale\": 0}]}"), 1e-6);
        assertEquals(score(referenceB, body + "}"), score(service, body + ", \"model\": \"b\"}"), 1e-6);
        String alora = errorOf(() -> new PoolingResource(service).generativeScoring(body
                + ", \"lora\": [{\"id\": 2, \"scale\": 1}]}"));
        assertTrue(alora.contains("activated LoRA"), alora);
    }

    private static double score(EngineService on, String body) {
        return at(view(new PoolingResource(on).generativeScoring(body).getEntity()), "data", 0).f64("score");
    }
}
