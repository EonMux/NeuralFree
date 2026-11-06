package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Base64;
import java.util.regex.Pattern;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The encoded-media cache over the real service with a llava projector: vLLM's {@code vllm:mm_cache_queries} and
 * {@code vllm:mm_cache_hits} counters on the Prometheus exposition (with vLLM's names, help texts and
 * {@code model_name} / {@code engine} labels) count one query per media item and one hit per repeated item, and
 * per-request {@code media_io_kwargs} are refused by name while a null or empty value is accepted, as in vLLM.
 */
class MediaCacheResourceTest {

    private static EngineService service;
    private static OpenAiResource openai;
    private static String dataUrl;
    /** The bound meters; micrometer reads them through a weak reference, so the test keeps them reachable. */
    private static volatile EngineMetrics bound;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path mmproj = MmprojTestFiles.llava(SyntheticModels.Config.llama().nEmbd());
        dataUrl = "data:image/png;base64,"
                + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(10, 6));
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        openai = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
        bound = null;
    }

    /** A Prometheus registry with the engine meters bound over the open service. */
    private static PrometheusMeterRegistry prometheus() {
        EngineLifecycle lifecycle = new EngineLifecycle() {
            @Override
            EngineService current() {
                return service;
            }
        };
        PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        bound = new EngineMetrics(lifecycle);
        bound.bindTo(registry);
        return registry;
    }

    /** A one-message chat of {@code text} and the image at {@code url}, with {@code extra} body fields. */
    private static String chat(String text, String url, String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": ["
                + "{\"type\": \"text\", \"text\": \"" + text + "\"},"
                + "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + url + "\"}}]}],"
                + " \"max_tokens\": 2, \"temperature\": 0" + extra + "}";
    }

    private static double counter(PrometheusMeterRegistry registry, String name) {
        return registry.get(name).tag("model_name", service.served().primary()).tag("engine", "0")
                .functionCounter().count();
    }

    /** The exposition line of {@code name} with vLLM's labels and {@code value}, a label set's comma optional. */
    private static Pattern sample(String name, double value) {
        return Pattern.compile("(?m)^" + Pattern.quote(name) + "\\{engine=\"0\",model_name=\""
                + Pattern.quote(service.served().primary()) + "\",?} " + Pattern.quote(Double.toString(value)) + "$");
    }

    @Test
    void theMediaCacheCountersExportUnderVllmsNamesAndLabels() {
        PrometheusMeterRegistry registry = prometheus();
        double queries = counter(registry, "vllm:mm_cache_queries");
        double hits = counter(registry, "vllm:mm_cache_hits");
        String counted = "data:image/png;base64,"
                + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(7, 11));
        String body = chat("look", counted, "");

        assertEquals(200, openai.chatCompletions(body).getStatus());
        assertEquals(queries + 1, counter(registry, "vllm:mm_cache_queries"), "one query per media item");
        assertEquals(hits, counter(registry, "vllm:mm_cache_hits"), "a new item misses");
        assertEquals(200, openai.chatCompletions(body).getStatus());
        assertEquals(queries + 2, counter(registry, "vllm:mm_cache_queries"));
        assertEquals(hits + 1, counter(registry, "vllm:mm_cache_hits"), "the repeated item hits");

        String scrape = registry.scrape();
        assertTrue(sample("vllm:mm_cache_queries_total", queries + 2).matcher(scrape).find(), scrape);
        assertTrue(sample("vllm:mm_cache_hits_total", hits + 1).matcher(scrape).find(), scrape);
        assertTrue(scrape.contains("# HELP vllm:mm_cache_queries_total "
                + "Multi-modal cache queries, in terms of number of queried items."), scrape);
        assertTrue(scrape.contains("# HELP vllm:mm_cache_hits_total "
                + "Multi-modal cache hits, in terms of number of cached items."), scrape);
    }

    @Test
    void perRequestMediaLoaderOptionsAreRefusedByName() {
        String gray = ", \"media_io_kwargs\": {\"image\": {\"image_mode\": \"L\"}}";
        String message = errorOf(() -> openai.chatCompletions(chat("look", dataUrl, gray)));
        assertTrue(message.contains("media_io_kwargs") && message.contains("--media-io-kwargs"), message);
        assertEquals(400, statusOf(() -> openai.chatCompletions(chat("look", dataUrl, gray))));
        assertEquals(400, statusOf(() -> openai.chatCompletions(chat("look", dataUrl,
                ", \"media_io_kwargs\": {\"image\": {}}"))), "a modality with no options is still an option set");

        for (String unset : new String[] {", \"media_io_kwargs\": {}", ", \"media_io_kwargs\": null"}) {
            JsonView usage = JsonView.of(view(openai.chatCompletions(chat("look", dataUrl, unset)).getEntity())
                    .raw("usage"), "usage");
            assertTrue(usage.i64("prompt_tokens") > 0, unset);
        }
    }
}
