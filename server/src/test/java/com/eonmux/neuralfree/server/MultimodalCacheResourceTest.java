package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.ref.Reference;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.function.BooleanSupplier;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Multimodal caching over the real service with a llava projector beside the text model: a repeated image chat
 * reuses its prompt from the prefix cache and its encoding from the projector's media cache, a part's {@code uuid}
 * (vLLM's {@code multi_modal_uuids}) lets a later part omit its data, {@code /reset_encoder_cache} drops every
 * encoder output (the projector's encodings and the engine's shared encoder cache) while the processed media keep
 * serving the uuid, {@code /reset_mm_cache} drops the whole media cache (a uuid-only part is then refused by name),
 * both refuse arguments by name, the media cache's counters reach {@code /metrics}, and media prompts route by their
 * rows' prefix-cache keys.
 */
class MultimodalCacheResourceTest {

    private static EngineService service;
    private static OpenAiResource openai;
    private static OpsResource ops;
    private static String png64;
    private static String otherPng64;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int width = SyntheticModels.Config.llama().nEmbd();
        Path mmproj = MmprojTestFiles.llava(width);
        png64 = Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(12, 8));
        otherPng64 = Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(10, 10));
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder().nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32)
                        .typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        openai = new OpenAiResource(service);
        ops = new OpsResource(service, new EngineProfiler(), DevMode.ON, ProfilerConfig.OFF);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static String chat(String text, String imagePart) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"" + text
                + "\"}, " + imagePart + "]}], \"max_tokens\": 2, \"temperature\": 0}";
    }

    private static String image(String b64, String uuid) {
        return "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64," + b64 + "\"}"
                + (uuid == null ? "" : ", \"uuid\": \"" + uuid + "\"") + "}";
    }

    private static String uuidOnly(String uuid) {
        return "{\"type\": \"image_url\", \"image_url\": null, \"uuid\": \"" + uuid + "\"}";
    }

    /** The usage of a successful chat completion of {@code body}. */
    private static JsonView usage(String body) {
        Response response;
        try {
            response = openai.chatCompletions(body);
        } catch (WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return JsonView.of(view(response.getEntity()).raw("usage"), "usage");
    }

    private static long cached(JsonView usage) {
        return JsonView.of(usage.raw("prompt_tokens_details"), "details").i64("cached_tokens");
    }

    /** Waits up to five seconds for {@code condition}, which a replica's next published snapshot makes true. */
    private static void awaitTrue(String what, BooleanSupplier condition) throws InterruptedException {
        long deadline = System.nanoTime() + 5_000_000_000L;
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
        assertTrue(condition.getAsBoolean(), what);
    }

    private static int engineEncoderRows() {
        int rows = 0;
        for (var s : service.replicaStats()) {
            rows += s.encoderCacheRows();
        }
        return rows;
    }

    @Test
    void aRepeatedImageChatReusesItsPromptAndItsEncoding() {
        String body = chat("what is in the first picture", image(png64, null));
        long queries = service.mediaCacheQueries();
        long hits = service.mediaCacheHits();
        JsonView first = usage(body);
        JsonView second = usage(body);
        assertEquals(first.i64("prompt_tokens"), second.i64("prompt_tokens"));
        assertEquals(second.i64("prompt_tokens") - 1, cached(second),
                "the whole prompt but its last row is served from the prefix cache, the image rows included");
        assertEquals(queries + 2, service.mediaCacheQueries());
        assertTrue(service.mediaCacheHits() >= hits + 1, "the repeat's image is not decoded or encoded again");
    }

    @Test
    void aUuidStandsForTheImageAndLetsALaterPartOmitItsData() {
        String text = "describe the uuid picture";
        JsonView named = usage(chat(text, image(otherPng64, "picture-7")));
        JsonView byUuid = usage(chat(text, uuidOnly("picture-7")));
        assertEquals(named.i64("prompt_tokens"), byUuid.i64("prompt_tokens"));
        assertEquals(byUuid.i64("prompt_tokens") - 1, cached(byUuid), "the uuid's image rows are cached");
        JsonView noUrl = usage(chat(text, "{\"type\": \"image_url\", \"image_url\": {}, \"uuid\": \"picture-7\"}"));
        assertEquals(named.i64("prompt_tokens"), noUrl.i64("prompt_tokens"));

        String unknown = errorOf(() -> openai.chatCompletions(chat(text, uuidOnly("never-sent"))));
        assertTrue(unknown.contains("uuid 'never-sent'") && unknown.contains("carries no data"), unknown);
        assertTrue(errorOf(() -> openai.chatCompletions(chat(text, "{\"type\": \"image_url\", \"image_url\": null}")))
                .contains("requires a 'url'"));
    }

    @Test
    void resetEncoderCacheDropsEveryEncoderOutputAndKeepsTheProcessedMedia() throws InterruptedException {
        String text = "the reset picture";
        usage(chat(text, image(png64, "reset-1")));
        ChatMedia media = service.media();
        assertTrue(media.cachedBytes() > 0, "the encoding is cached");
        long processed = media.cacheUsageBytes() - media.cachedBytes();
        assertTrue(processed > 0, "the prepared tiles are cached");
        awaitTrue("the engine shares the image's rows", () -> engineEncoderRows() > 0);

        assertEquals(200, ops.resetEncoderCache(null, null).getStatus());
        assertEquals(0, media.cachedBytes());
        assertEquals(processed, media.cacheUsageBytes(), "the processed media stay");
        awaitTrue("the engine's encoder cache is empty", () -> engineEncoderRows() == 0);

        long hits = service.mediaCacheHits();
        usage(chat(text, uuidOnly("reset-1")));
        assertTrue(service.mediaCacheHits() > hits, "the processed tiles served the uuid alone");
        assertTrue(media.cachedBytes() > 0, "the tiles were encoded again");
    }

    @Test
    void resetMmCacheDropsTheWholeMediaCache() {
        String text = "the mm reset picture";
        usage(chat(text, image(png64, "reset-2")));
        assertTrue(service.mediaCacheUsageBytes() > 0);
        assertEquals(200, ops.resetMmCache("", null).getStatus());
        assertEquals(0, service.mediaCacheUsageBytes());
        String refused = errorOf(() -> openai.chatCompletions(chat(text, uuidOnly("reset-2"))));
        assertTrue(refused.contains("uuid 'reset-2'"), refused);
    }

    @Test
    void theResetEndpointsRefuseArgumentsByName() {
        assertEquals(400, statusOf(() -> ops.resetMmCache("{\"all\": true}", null)));
        assertTrue(errorOf(() -> ops.resetMmCache("{\"all\": true}", null)).contains("reset_mm_cache takes no all"));
        assertTrue(errorOf(() -> ops.resetEncoderCache("{\"modality\": \"image\"}", null))
                .contains("reset_encoder_cache takes no modality"));
        assertEquals(200, ops.resetEncoderCache("{}", null).getStatus());
    }

    @Test
    void theResetEndpointsAnswerWithoutAProjectorOrAModel() throws IOException {
        try (EngineService plain = new EngineService(null, ServerTestSupport.pagedConfig(64, 2))) {
            OpsResource none = new OpsResource(plain, new EngineProfiler(), DevMode.ON, ProfilerConfig.OFF);
            assertEquals(200, none.resetMmCache(null, null).getStatus());
            assertEquals(200, none.resetEncoderCache(null, null).getStatus());
            assertEquals(0, plain.mediaCacheQueries());
            assertEquals(0, plain.mediaCacheUsageBytes());
        }
    }

    @Test
    void theMediaCacheCountersReachTheMetrics() {
        EngineLifecycle lifecycle = new EngineLifecycle() {
            @Override
            EngineService current() {
                return service;
            }
        };
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        EngineMetrics metrics = new EngineMetrics(lifecycle);
        metrics.bindTo(registry);
        usage(chat("the metrics picture", image(png64, null)));
        double queries = registry.get("neuralfree.mm.cache.queries").functionCounter().count();
        assertTrue(queries > 0);
        assertEquals((double) service.mediaCacheQueries(), queries);
        assertEquals((double) service.mediaCacheHits(),
                registry.get("neuralfree.mm.cache.hits").functionCounter().count());
        assertEquals((double) service.mediaCacheUsageBytes(),
                registry.get("neuralfree.mm.cache.usage.bytes").gauge().value());
        Reference.reachabilityFence(metrics); // the registry holds the meters' state weakly
    }

    @Test
    void mediaPromptsRouteByTheirRowsPrefixCacheKeys() {
        String rendered = "ab" + ChatMessages.MEDIA_MARKER;
        MediaPrompt a = service.media().prompt(rendered, service.media().items(List.of(png64), "multimodal_data"),
                List.of(), -1, t -> new int[] {7});
        MediaPrompt again = service.media().prompt(rendered,
                service.media().items(List.of(png64), "multimodal_data"), List.of(), -1, t -> new int[] {7});
        MediaPrompt b = service.media().prompt(rendered,
                service.media().items(List.of(otherPng64), "multimodal_data"), List.of(), -1, t -> new int[] {7});
        int[] ra = DpScheduler.routingRows(a);
        assertEquals(7, ra[0], "a text row routes on its token id");
        assertTrue(ra.length > 1 && ra[1] < 0, "a media row routes on a key that never equals a token");
        assertArrayEquals(ra, DpScheduler.routingRows(again), "the same image routes alike");
        assertNotEquals(ra[1], DpScheduler.routingRows(b)[1], "different images route on different keys");
    }

    @Test
    void aUuidNamedItemRoutesByItsUuidAsThePrefixCacheKeysIt() {
        MediaRopePositions at = MediaRopePositions.normal(1, 2);
        MediaKey cat = MediaKey.ofUuid(MediaLimits.Modality.IMAGE, "cat", null);
        int[] one = DpScheduler.routingRows(MediaPrompt.builder().addText(new int[] {7})
                .addMedia(cat, 0, at, new float[][] {{1f, 2f}, {3f, 4f}}, false).build());
        int[] two = DpScheduler.routingRows(MediaPrompt.builder().addText(new int[] {7})
                .addMedia(cat, 0, at, new float[][] {{5f, 6f}, {7f, 8f}}, false).build());
        int[] dog = DpScheduler.routingRows(MediaPrompt.builder().addText(new int[] {7})
                .addMedia(MediaKey.ofUuid(MediaLimits.Modality.IMAGE, "dog", null), 0, at,
                        new float[][] {{1f, 2f}, {3f, 4f}}, false).build());
        assertArrayEquals(one, two, "one uuid is one item, whatever rows carry it");
        assertNotEquals(one[1], dog[1], "another uuid routes on another key");
        assertNotEquals(one[1], one[2], "an item's rows route on their own keys");
    }
}
