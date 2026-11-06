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
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Per-request LoRA adapters over a data-parallel pool of two replicas routed round robin: a startup module and an
 * adapter loaded at runtime ({@code /v1/load_lora_adapter}) run on every replica with the text the adapter attached
 * to every row produces, by {@code model} (vLLM) and by the {@code lora} list (llama.cpp); an unloaded adapter is
 * refused on every replica.
 */
class LoraDpResourceTest {

    private static final String BASE = "base";
    private static final String PROMPT = "The quick brown fox";

    private static Path adapterB;
    private static EngineService pool;
    private static EngineService referenceA;
    private static EngineService referenceB;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path adapterA = LoraAdapterFiles.llama(1000, 4, 8.0f);
        adapterB = LoraAdapterFiles.llama(5000, 4, 8.0f);
        ServerOptions single = new ServerOptions(BASE, ServerOptions.DEFAULT_TIMEOUT);
        ServerOptions replicated = new ServerOptions(BASE, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null)
                .withDataParallel(new DpLayout(2, 0, LoadBalance.ROUND_ROBIN));
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2), replicated,
                new ServerLoras(List.of(new ServerLoras.Module("a", adapterA.toString())), true));
        referenceA = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2).toBuilder()
                .addLora(adapterA.toString(), 1.0f).build(), single);
        referenceB = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2).toBuilder()
                .addLora(adapterB.toString(), 1.0f).build(), single);
    }

    @AfterAll
    static void close() {
        pool.close();
        referenceA.close();
        referenceB.close();
    }

    /** The greedy completion text of {@code model} with {@code extra} body fields; the response echoes it. */
    private static String completionAs(EngineService on, String model, String extra) {
        String body = "{\"model\": \"" + model + "\", \"prompt\": \"" + PROMPT + "\", \"max_tokens\": 6,"
                + " \"temperature\": 0" + extra + "}";
        JsonView response = view(new OpenAiResource(on).completions(body).getEntity());
        assertEquals(model, response.str("model"));
        return (String) ((Map<?, ?>) ((List<?>) response.raw("choices")).get(0)).get("text");
    }

    /** Prompt tokens replica {@code r} has processed, prefilled or reused from its prefix cache. */
    private static long served(int r) {
        return pool.replicaStats(r).promptTokens() + pool.replicaStats(r).promptTokensCached();
    }

    /**
     * Asserts that two consecutive requests of {@code model} with {@code extra}, one per replica under round robin,
     * both produce {@code expected}, and that each replica served one of them.
     */
    private static void assertBothReplicasRun(String expected, String model, String extra) {
        long before0 = served(0);
        long before1 = served(1);
        assertEquals(expected, completionAs(pool, model, extra));
        assertEquals(expected, completionAs(pool, model, extra));
        awaitServed(0, before0);
        awaitServed(1, before1);
    }

    /** Waits until replica {@code r}'s published stats show prompt tokens beyond {@code before}. */
    private static void awaitServed(int r, long before) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (served(r) <= before) {
            assertTrue(System.nanoTime() < deadline, "replica " + r + " served one of the requests");
            Thread.onSpinWait();
        }
    }

    @Test
    void everyReplicaRunsTheStartupAndTheRuntimeAdapters() {
        assertEquals(2, pool.replicaCount());
        String textA = completionAs(referenceA, BASE, "");
        assertBothReplicasRun(textA, "a", "");
        assertBothReplicasRun(textA, BASE, ", \"lora\": [{\"id\": 0, \"scale\": 1.0}]");

        LoraResource loras = new LoraResource(pool);
        String load = "{\"lora_name\": \"b\", \"lora_path\": \"" + adapterB + "\"}";
        assertEquals("Success: LoRA adapter 'b' added successfully.", loras.load(load).getEntity());
        try {
            assertBothReplicasRun(completionAs(referenceB, BASE, ""), "b", "");
        } finally {
            assertEquals("Success: LoRA adapter 'b' removed successfully.",
                    loras.unload("{\"lora_name\": \"b\"}").getEntity());
        }
        assertEquals(404, statusOf(() -> completionAs(pool, "b", "")));
        assertEquals(404, statusOf(() -> completionAs(pool, "b", "")));
        assertBothReplicasRun(textA, "a", "");
    }
}
