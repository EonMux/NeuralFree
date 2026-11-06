package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The batch runner's Prometheus server, without Quarkus: the engine's meters and the JVM's in the text format on
 * any path, zero before the engine opens and the engine's counts once it runs, the opened engine's replica and
 * multimodal cache series once the runner reports it open, GET and HEAD only, the API key required when configured,
 * and an occupied port refused naming the address.
 */
class BatchMetricsTest {

    private static EngineService service;
    private static final HttpClient CLIENT = HttpClient.newHttpClient();

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2));
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static HttpResponse<String> get(BatchMetrics metrics, String path, String authorization)
            throws IOException, InterruptedException {
        HttpRequest.Builder request = HttpRequest.newBuilder(
                URI.create("http://127.0.0.1:" + metrics.port() + path)).GET();
        if (authorization != null) {
            request.header("Authorization", authorization);
        }
        return CLIENT.send(request.build(), HttpResponse.BodyHandlers.ofString());
    }

    /** The value of one unlabelled series of a scrape. */
    private static double value(String scrape, String series) {
        Matcher m = Pattern.compile("(?m)^" + Pattern.quote(series) + " (\\S+)$").matcher(scrape);
        assertTrue(m.find(), series + " is in the scrape:\n" + scrape);
        return Double.parseDouble(m.group(1));
    }

    @Test
    void theEngineMetersReadZeroUntilTheEngineOpensThenItsCounts() throws Exception {
        AtomicReference<EngineService> current = new AtomicReference<>();
        try (BatchMetrics metrics = BatchMetrics.start("127.0.0.1", 0, current::get)) {
            HttpResponse<String> before = get(metrics, "/metrics", null);
            assertEquals(200, before.statusCode());
            assertEquals(BatchMetrics.CONTENT_TYPE, before.headers().firstValue("Content-Type").orElse(""));
            assertEquals(0.0, value(before.body(), "neuralfree_tokens_predicted_total"));
            assertTrue(before.body().contains("jvm_memory_used_bytes"), "the JVM meters ride along");

            current.set(service);
            new OpenAiResource(service).completions(
                    "{\"prompt\": \"hello there\", \"max_tokens\": 5, \"ignore_eos\": true}");
            System.gc(); // the meters' state is weakly held by the registry: it must survive a collection
            long deadline = System.nanoTime() + 5_000_000_000L;
            double predicted = 0;
            while (predicted < 5 && System.nanoTime() < deadline) {
                predicted = value(get(metrics, "/", null).body(), "neuralfree_tokens_predicted_total");
                Thread.sleep(10);
            }
            assertTrue(predicted >= 5, "every path serves the meters, and the decode shows: " + predicted);
        }
    }

    @Test
    void theOpenedEnginesSeriesRegisterWhenTheRunnerReportsItOpen() throws Exception {
        AtomicReference<EngineService> current = new AtomicReference<>();
        try (BatchMetrics metrics = BatchMetrics.start("127.0.0.1", 0, current::get)) {
            String before = get(metrics, "/metrics", null).body();
            assertFalse(before.contains("vllm:mm_cache_queries_total"), "no model name to tag before the engine opens");

            current.set(service);
            metrics.opened(service);
            String after = get(metrics, "/metrics", null).body();
            String labels = "\\{engine=\"0\",model_name=\"" + Pattern.quote(service.served().primary()) + "\",?}";
            assertTrue(Pattern.compile("(?m)^vllm:mm_cache_queries_total" + labels + " 0\\.0$").matcher(after).find(),
                    after);
            assertTrue(Pattern.compile("(?m)^vllm:mm_cache_hits_total" + labels + " 0\\.0$").matcher(after).find(),
                    "a text-only engine has a media cache of none: " + after);
            assertTrue(after.contains("neuralfree_replica_requests_processing{"), "the replica series ride along");
        }
    }

    @Test
    void onlyGetAndHeadAreAnswered() throws Exception {
        try (BatchMetrics metrics = BatchMetrics.start("127.0.0.1", 0, () -> null)) {
            URI uri = URI.create("http://127.0.0.1:" + metrics.port() + "/metrics");
            HttpResponse<String> post = CLIENT.send(HttpRequest.newBuilder(uri)
                    .POST(HttpRequest.BodyPublishers.ofString("x")).build(), HttpResponse.BodyHandlers.ofString());
            assertEquals(405, post.statusCode());
            HttpResponse<String> head = CLIENT.send(HttpRequest.newBuilder(uri)
                    .method("HEAD", HttpRequest.BodyPublishers.noBody()).build(), HttpResponse.BodyHandlers.ofString());
            assertEquals(200, head.statusCode());
            assertEquals("", head.body());
        }
    }

    @Test
    void aConfiguredApiKeyGuardsTheScrape() throws Exception {
        String saved = System.getProperty(ApiKeys.PROPERTY);
        System.setProperty(ApiKeys.PROPERTY, "batch-key");
        try (BatchMetrics metrics = BatchMetrics.start("127.0.0.1", 0, () -> null)) {
            HttpResponse<String> refused = get(metrics, "/metrics", null);
            assertEquals(401, refused.statusCode());
            assertTrue(refused.body().contains("authentication_error"), refused.body());
            assertEquals(401, get(metrics, "/metrics", "Bearer wrong").statusCode());
            assertEquals(200, get(metrics, "/metrics", "Bearer batch-key").statusCode());
        } finally {
            if (saved == null) {
                System.clearProperty(ApiKeys.PROPERTY);
            } else {
                System.setProperty(ApiKeys.PROPERTY, saved);
            }
        }
    }

    @Test
    void anOccupiedPortIsRefusedNamingTheAddress() throws Exception {
        try (BatchMetrics first = BatchMetrics.start("127.0.0.1", 0, () -> null)) {
            IOException e = assertThrows(IOException.class,
                    () -> BatchMetrics.start("127.0.0.1", first.port(), () -> null));
            assertTrue(e.getMessage().contains("127.0.0.1:" + first.port()), e.getMessage());
        }
    }
}
