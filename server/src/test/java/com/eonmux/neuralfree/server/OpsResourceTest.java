package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The operational endpoints over the real service (vLLM's instrumentator and dev routers, sglang's control
 * endpoints): version, server info, load, the ready probes and ping, tokenizer info, the prefix-cache reset (its
 * effect on the prefix reuse), aborting live generations to {@code finish_reason: "abort"} and the profile window.
 * An unloaded service keeps the reporting endpoints and refuses the probes with 503.
 */
public class OpsResourceTest {

    private static EngineService service;
    private static EngineService unloaded;
    private static EngineProfiler profiler;
    private static OpsResource resource;
    private static OpsResource unloadedResource;
    private static OpenAiResource openAi;
    private static ExecutorService http;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), pagedConfig(512, 2));
        unloaded = new EngineService(null, pagedConfig(64, 2));
        Path profileDir = Files.createTempDirectory("neuralfree-profile");
        profiler = new EngineProfiler(profileDir);
        ProfilerConfig profilerConfig = ProfilerConfig.of(ProfilerConfig.TORCH, profileDir);
        resource = new OpsResource(service, profiler, DevMode.ON, profilerConfig);
        unloadedResource = new OpsResource(unloaded, profiler, DevMode.ON, profilerConfig);
        openAi = new OpenAiResource(service);
        http = Executors.newFixedThreadPool(4);
    }

    @AfterAll
    static void close() {
        http.shutdownNow();
        service.close();
        unloaded.close();
    }

    @Test
    void versionIsTheBuildVersion() throws IOException {
        Matcher declared = Pattern.compile("(?m)^version = '([^']+)'")
                .matcher(Files.readString(Path.of("..", "build.gradle")));
        assertTrue(declared.find(), "the root build.gradle declares the version");
        assertEquals(declared.group(1), view(resource.version().getEntity()).str("version"));
        assertEquals(declared.group(1), view(unloadedResource.version().getEntity()).str("version"));
    }

    @Test
    void serverInfoReportsConfigurationAndState() {
        JsonView body = view(resource.serverInfo(null).getEntity());
        assertEquals(service.modelPath(), view(resource.serverInfo("json").getEntity()).str("model_path"));
        assertTrue(errorOf(() -> resource.serverInfo("text")).contains("config_format"));
        assertEquals(ServerVersion.VALUE, body.str("version"));
        assertTrue(body.bool("loaded"));
        assertEquals(service.modelPath(), body.str("model_path"));
        assertNotEquals("", body.str("model_alias"));
        assertEquals(service.architecture(), body.str("model_arch"));
        assertEquals(service.contextLength(), body.i64("context_length"));
        assertTrue(body.i64("slots") >= 1);
        assertTrue(body.has("backend"));
        assertTrue(body.has("neuralfree_env"));
        JsonView unloadedBody = view(unloadedResource.serverInfo(null).getEntity());
        assertFalse(unloadedBody.bool("loaded"));
        assertNull(unloadedBody.raw("model_path"));
        assertNull(unloadedBody.raw("model_alias"));
    }

    @Test
    void readyAndPingFollowTheProbeSemantics() {
        assertEquals(200, resource.ready().getStatus());
        assertNull(resource.ready().getEntity(), "vLLM's probes answer empty bodies");
        assertEquals(200, resource.readyz().getStatus());
        assertEquals(503, statusOf(unloadedResource::ready));
        assertEquals(503, statusOf(unloadedResource::readyz));
        assertEquals(200, resource.ping().getStatus());
        assertNull(resource.ping().getEntity());
        assertEquals(200, resource.pingPost().getStatus());
        assertEquals(503, statusOf(unloadedResource::ping), "ping answers as /health does");
        assertEquals(503, statusOf(unloadedResource::pingPost));
    }

    @Test
    void serverInfoOmitsTheWatermarkConfigurationAndItsKey() {
        String saved = System.getProperty(WatermarkConfig.PROP_CONFIG);
        try {
            System.setProperty(WatermarkConfig.PROP_CONFIG,
                    "{\"key\": 987654321987, \"algorithm\": \"gumbel\"}");
            String body = (String) resource.serverInfo("json").getEntity();
            assertFalse(body.contains("987654321987"), body);
            assertFalse(body.contains(WatermarkConfig.PROP_CONFIG), body);
        } finally {
            if (saved == null) {
                System.clearProperty(WatermarkConfig.PROP_CONFIG);
            } else {
                System.setProperty(WatermarkConfig.PROP_CONFIG, saved);
            }
        }
    }

    @Test
    void serverInfoOmitsTheRpcEntriesWithTheirInlineKeys() {
        String saved = System.getProperty(EngineConfig.PROP_RPC);
        try {
            System.setProperty(EngineConfig.PROP_RPC, "gpu-box:50052@s3cret-rpc-key");
            String body = (String) resource.serverInfo("json").getEntity();
            assertFalse(body.contains("s3cret-rpc-key"), "an inline RPC key never reaches a client: " + body);
            assertFalse(body.contains(EngineConfig.PROP_RPC), body);
        } finally {
            if (saved == null) {
                System.clearProperty(EngineConfig.PROP_RPC);
            } else {
                System.setProperty(EngineConfig.PROP_RPC, saved);
            }
        }
    }

    @Test
    void serverInfoOmitsTheKvTransferConfigurationAndItsToken() {
        String saved = System.getProperty(KvTransferConfig.SERVER_PROPERTY);
        try {
            System.setProperty(KvTransferConfig.SERVER_PROPERTY,
                    "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_both\","
                            + " \"kv_connector_extra_config\": {\"auth_token\": \"kv-transfer-secret\"}}");
            String body = (String) resource.serverInfo("json").getEntity();
            assertFalse(body.contains("kv-transfer-secret"),
                    "the KV transfer token never reaches a client: " + body);
            assertFalse(body.contains(KvTransferConfig.SERVER_PROPERTY), body);
        } finally {
            if (saved == null) {
                System.clearProperty(KvTransferConfig.SERVER_PROPERTY);
            } else {
                System.setProperty(KvTransferConfig.SERVER_PROPERTY, saved);
            }
        }
    }

    @Test
    void serverInfoOmitsEverySettingSecretPropertiesMarksSecret() {
        String[] names = {"neuralfree.server.api.key", "neuralfree.remote.gcs.token",
                "neuralfree.remote.azure.sas", "neuralfree.server.mcp.json"};
        String[] saved = new String[names.length];
        String publicName = "neuralfree.server.replicas";
        String savedPublic = System.getProperty(publicName);
        try {
            for (int i = 0; i < names.length; i++) {
                saved[i] = System.getProperty(names[i]);
                System.setProperty(names[i], "secret-value-" + i);
            }
            System.setProperty(publicName, "1");
            String body = (String) resource.serverInfo("json").getEntity();
            for (int i = 0; i < names.length; i++) {
                assertFalse(body.contains(names[i]), names[i] + " reached the report: " + body);
                assertFalse(body.contains("secret-value-" + i), body);
            }
            assertTrue(body.contains(publicName), body);
        } finally {
            for (int i = 0; i < names.length; i++) {
                if (saved[i] == null) {
                    System.clearProperty(names[i]);
                } else {
                    System.setProperty(names[i], saved[i]);
                }
            }
            if (savedPublic == null) {
                System.clearProperty(publicName);
            } else {
                System.setProperty(publicName, savedPublic);
            }
        }
    }

    @Test
    void loadCountsTheRequestsInFlight() throws Exception {
        assertEquals(0L, view(resource.load().getEntity()).i64("server_load"));
        assertEquals(0L, view(unloadedResource.load().getEntity()).i64("server_load"));
        List<Future<Response>> calls = runFourLongCompletions();
        awaitAllLive();
        assertEquals(4L, view(resource.load().getEntity()).i64("server_load"),
                "running and waiting requests are in flight alike");
        abortAndDrain(calls);
        assertEquals(0L, view(resource.load().getEntity()).i64("server_load"), "finished requests leave the load");
    }

    @Test
    void tokenizerInfoNamesTheModelTokenizer() {
        JsonView body = view(resource.tokenizerInfo().getEntity());
        assertEquals(service.vocabSize(), body.i64("vocab_size"));
        assertEquals(service.tokenizer().tokenizerModel(), body.str("tokenizer_class"));
        assertEquals(service.tokenizer().bosId(), body.i64("bos_token_id"));
        assertEquals(service.tokenizer().eosId(), body.i64("eos_token_id"));
        assertEquals(service.tokenizer().getToken(service.tokenizer().eosId()), body.str("eos_token"));
        assertEquals(service.maxModelLength(), body.i64("model_max_length"));
        List<?> stops = (List<?>) body.raw("stop_token_ids");
        assertEquals(service.tokenizer().stopTokenIds().size(), stops.size());
        for (Object id : stops) {
            assertTrue(service.tokenizer().isStopToken(((Number) id).intValue()),
                    "stop_token_ids lists the eog ids");
        }
        assertEquals(503, statusOf(unloadedResource::tokenizerInfo));
    }

    @Test
    void resetPrefixCacheRefusesByNamesAndDropsTheCachedPrefixes() throws Exception {
        assertEquals(400, statusOf(() -> resource.resetPrefixCache(true, false)));
        assertTrue(errorOf(() -> resource.resetPrefixCache(true, false)).contains("reset_running_requests"));
        assertEquals(400, statusOf(() -> resource.resetPrefixCache(false, true)));
        assertTrue(errorOf(() -> resource.resetPrefixCache(false, true)).contains("reset_external"));
        assertEquals(503, statusOf(() -> unloadedResource.resetPrefixCache(false, false)));

        // a small budget keeps the sequence's pages far below the pool, so the cache holds it for the rerun
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(6)
                .sampler(SamplerConfig.greedyDecoding()).build();
        // the prefix cache shares whole pages only: the probe prompt must fill more than one page
        int[] prompt = service.encode("the quick brown fox jumps over the lazy dog ".repeat(4), true, true);
        submitAndAwait(prompt, request);
        assertTrue(rerunReusedPrefix(prompt, request) > 0, "the rerun reuses the cached prefix");
        assertTrue(view(resource.resetPrefixCache(false, false).getEntity()).bool("success"));
        assertEquals(0, rerunReusedPrefix(prompt, request), "the reset dropped the cached prefix");
    }

    @Test
    void abortRequestsStopsEveryLiveGeneration() throws Exception {
        assertEquals(0L, view(resource.abortRequests(null).getEntity()).i64("aborted"));
        assertEquals(0L, view(resource.abortRequests("{\"abort_all\": false}").getEntity()).i64("aborted"));
        assertTrue(errorOf(() -> resource.abortRequests("{\"rid\": \"x\"}")).contains("rid"));
        assertEquals(400, statusOf(() -> resource.abortRequests("{\"nonsense\": true}")));
        assertTrue(errorOf(() -> resource.abortRequests("{\"abort_all\": \"yes\"}")).contains("abort_all"));

        List<Future<Response>> calls = runFourLongCompletions();
        awaitAllLive();
        assertFalse(view(resource.resetPrefixCache(false, false).getEntity()).bool("success"),
                "live requests hold pages of the memory, so the reset reports failure");
        long aborted = view(resource.abortRequests(null).getEntity()).i64("aborted");
        assertEquals(4L, aborted, "every live request takes the abort switch");
        for (Future<Response> call : calls) {
            JsonView body = view(call.get(60, TimeUnit.SECONDS).getEntity());
            assertEquals("abort", firstChoiceFinishReason(body), "the aborted client learns the abort: " + body);
        }
        assertEquals(0L, view(resource.abortRequests(null).getEntity()).i64("aborted"), "none left live");
    }

    @Test
    void profileWindowReportsTheEngineDeltas() throws Exception {
        assertEquals(400, statusOf(resource::stopProfile));
        profiler.start(service.stats());
        assertEquals(400, statusOf(this::startSecondProfile));
        assertTrue(errorOf(() -> resource.startProfile("{\"profile_prefix\": \"x\"}"))
                .contains("profile_prefix"));

        GenerationRequest request = GenerationRequest.builder().maxNewTokens(6)
                .sampler(SamplerConfig.greedyDecoding()).build();
        submitAndAwait(service.encode("profile me", true, true), request);
        JsonView report = view(resource.stopProfile().getEntity());
        assertTrue(report.f64("duration_seconds") > 0);
        assertTrue(report.i64("generated_tokens") >= 1);
        assertTrue(report.i64("decode_steps") >= 1);
        assertTrue(report.i64("prompt_tokens") >= 1);
        assertTrue(report.f64("generated_tokens_per_second") > 0);
        assertTrue(report.str("recording").endsWith(".jfr"));
        assertTrue(Files.isRegularFile(Path.of(report.str("recording"))),
                "the profile writes its JFR recording");
        assertEquals(400, statusOf(resource::stopProfile), "the window closed");
    }

    // ------------------------------------------------------------------ helpers

    /** Opens a second profile window while the first is open: the resource refuses it with 400. */
    private void startSecondProfile() {
        resource.startProfile(null);
    }

    /** Four non-streaming completions with token budgets no slot drains quickly. */
    private List<Future<Response>> runFourLongCompletions() {
        List<Future<Response>> calls = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            calls.add(http.submit(() -> openAi.completions(
                    "{\"prompt\": \"hello\", \"max_tokens\": 511, \"temperature\": 0}")));
        }
        return calls;
    }

    /** Waits until every submitted request is live: running or waiting for a slot. */
    private void awaitAllLive() throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (service.stats().running() + service.stats().waiting() < 4) {
            assertTrue(System.nanoTime() < deadline, "the requests never went live");
            Thread.sleep(5);
        }
    }

    /** Aborts every live request, joins the HTTP calls and waits until the engine is idle. */
    private void abortAndDrain(List<Future<Response>> calls) throws Exception {
        assertTrue(view(resource.abortRequests(null).getEntity()).i64("aborted") >= 1);
        for (Future<Response> call : calls) {
            call.get(60, TimeUnit.SECONDS);
        }
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (service.stats().running() + service.stats().waiting() > 0) {
            assertTrue(System.nanoTime() < deadline, "the aborted requests never drained");
            Thread.sleep(5);
        }
    }

    /** The finish reason of a completion response's first choice. */
    private static String firstChoiceFinishReason(JsonView body) {
        return JsonView.of(((List<?>) body.raw("choices")).get(0), "choices[0]").str("finish_reason");
    }

    /** Submits {@code prompt} with a small greedy budget and waits until it leaves the engine. */
    private static void submitAndAwait(int[] prompt, GenerationRequest request) throws Exception {
        EngineService.Submission submission = service.submit(prompt, request, (tokenId, text) -> { });
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
        while (submission.sequence() == null || !submission.sequence().finished()) {
            assertTrue(System.nanoTime() < deadline, "the generation ran past the wait; refused="
                    + submission.refused());
            Thread.sleep(10);
        }
    }

    /** Submits {@code prompt} again and returns the prefix tokens its run reused. */
    private static int rerunReusedPrefix(int[] prompt, GenerationRequest request) throws Exception {
        EngineService.Submission submission = service.submit(prompt, request, (tokenId, text) -> { });
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
        while (submission.sequence() == null || !submission.sequence().finished()) {
            assertTrue(System.nanoTime() < deadline, "the generation ran past the wait; refused="
                    + submission.refused());
            Thread.sleep(10);
        }
        return submission.sequence().reusedPrefix();
    }
}
