package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
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
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The development-endpoint gate over the real engine: vLLM's development and RL control routes answer 404 without
 * {@link DevMode} and serve with it, in single-model and router mode alike, while the routes vLLM (or llama.cpp)
 * serves outside the dev mode — the probes, {@code /version}, {@code /load}, {@code /tokenizer_info}, the profile
 * window (its {@link ProfilerConfig} satisfied) and the checkpoint window (its {@link SleepModeSwitch} satisfied) —
 * stay reachable.
 */
class DevModeGateTest {

    @TempDir
    static Path dir;

    private static EngineService service;
    private static EngineService unloaded;
    private static ModelRouter router;
    private static OpsResource opsOff;
    private static OpsResource opsOn;
    private static SleepResource sleepOff;
    private static SleepResource sleepOn;
    private static WeightTransferResource weightsOff;
    private static WeightTransferResource weightsOn;
    private static CollectiveRpcResource rpcOff;
    private static CollectiveRpcResource rpcOn;
    private static CollectiveRpcResource routerRpcOff;
    private static CollectiveRpcResource routerRpcOn;
    private static OpsResource routerOpsOff;
    private static OpsResource routerOpsOn;
    private static SleepResource routerSleepOff;
    private static WeightTransferResource routerWeightsOff;
    private static WeightTransferResource routerWeightsOn;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), pagedConfig(256, 2));
        Files.copy(model, dir.resolve("a.gguf"));

        EngineProfiler profiler = new EngineProfiler(Files.createDirectories(dir.resolve("profile")));
        EngineProfiler routerProfiler = new EngineProfiler(Files.createDirectories(dir.resolve("router-profile")));
        ProfilerConfig profilerConfig = ProfilerConfig.of(ProfilerConfig.TORCH,
                Files.createDirectories(dir.resolve("profile")));
        opsOff = new OpsResource(service, profiler, DevMode.OFF, profilerConfig);
        opsOn = new OpsResource(service, profiler, DevMode.ON, profilerConfig);
        sleepOff = new SleepResource(service, DevMode.OFF, SleepModeSwitch.ON);
        sleepOn = new SleepResource(service, DevMode.ON, SleepModeSwitch.ON);
        weightsOff = new WeightTransferResource(service, DevMode.OFF);
        weightsOn = new WeightTransferResource(service, DevMode.ON);
        rpcOff = new CollectiveRpcResource(service, DevMode.OFF);
        rpcOn = new CollectiveRpcResource(service, DevMode.ON);

        ServerOptions options = new ServerOptions(null, Duration.ofMinutes(2));
        EngineConfig config = pagedConfig(256, 2);
        unloaded = new EngineService(null, config, options);
        router = new ModelRouter(dir, null, 0, 0, true, config, options, ModelRouter.systemDeviceMemory());
        routerOpsOff = new OpsResource(unloaded, routerProfiler, router, DevMode.OFF, profilerConfig);
        routerOpsOn = new OpsResource(unloaded, routerProfiler, router, DevMode.ON, profilerConfig);
        routerSleepOff = new SleepResource(unloaded, router, DevMode.OFF, SleepModeSwitch.ON);
        routerWeightsOff = new WeightTransferResource(unloaded, router, DevMode.OFF);
        routerWeightsOn = new WeightTransferResource(unloaded, router, DevMode.ON);
        routerRpcOff = new CollectiveRpcResource(unloaded, router, DevMode.OFF);
        routerRpcOn = new CollectiveRpcResource(unloaded, router, DevMode.ON);
    }

    @AfterAll
    static void close() throws InterruptedException {
        if (service.isSleeping()) {
            service.wakeUp(List.of());
        }
        if (service.isPaused()) {
            service.resume();
        }
        router.close();
        unloaded.close();
        service.close();
    }

    // ------------------------------------------------------------------ the ops routes

    @Test
    void opsRoutesAnswer404WithoutDevModeAndServeWithIt() {
        assertGated("server_info", () -> opsOff.serverInfo(null, null, null));
        assertGated("server_info config_format", () -> opsOff.serverInfo("json", null, null));
        assertGated("reset_prefix_cache", () -> opsOff.resetPrefixCache(false, false, null, null));
        assertGated("reset_mm_cache", () -> opsOff.resetMmCache(null, null, null, null));
        assertGated("reset_encoder_cache", () -> opsOff.resetEncoderCache(null, null, null, null));
        assertGated("abort_requests", () -> opsOff.abortRequests(null));

        assertEquals(200, opsOn.serverInfo(null, null, null).getStatus());
        JsonView info = view(opsOn.serverInfo(null, null, null).getEntity());
        assertEquals(service.modelPath(), info.str("model_path"));
        assertEquals(200, opsOn.resetPrefixCache(false, false, null, null).getStatus());
        assertEquals(200, opsOn.resetMmCache(null, null, null, null).getStatus());
        assertEquals(200, opsOn.resetEncoderCache(null, null, null, null).getStatus());
        JsonView aborted = view(opsOn.abortRequests(null).getEntity());
        assertEquals(0L, aborted.i64("aborted"));
    }

    @Test
    void sleepAndPauseRoutesAnswer404WithoutDevModeAndServeWithIt() {
        assertGated("sleep", () -> sleepOff.sleep("0", "abort", null, null));
        assertGated("wake_up", () -> sleepOff.wakeUp(null, null, null));
        assertGated("is_sleeping", () -> sleepOff.isSleeping(null, null));
        assertGated("release_kv_cache_memory", () -> sleepOff.releaseKvCacheMemory(null, null));
        assertGated("pause", () -> sleepOff.pause("abort", null, null, null));
        assertGated("resume", () -> sleepOff.resume(null, null));
        assertGated("is_paused", () -> sleepOff.isPaused(null, null));

        assertEquals("sleeping", view(sleepOn.sleep("0", "abort", null, null).getEntity()).str("status"));
        assertEquals(Boolean.TRUE, view(sleepOn.isSleeping(null, null).getEntity()).raw("is_sleeping"));
        assertEquals("awake", view(sleepOn.wakeUp(null, null, null).getEntity()).str("status"));
        assertEquals(Boolean.FALSE, view(sleepOn.isSleeping(null, null).getEntity()).raw("is_sleeping"));
        assertEquals(200, sleepOn.releaseKvCacheMemory(null, null).getStatus());
        assertEquals("paused", view(sleepOn.pause("abort", null, null, null).getEntity()).str("status"));
        assertEquals(Boolean.TRUE, view(sleepOn.isPaused(null, null).getEntity()).raw("is_paused"));
        assertEquals("resumed", view(sleepOn.resume(null, null).getEntity()).str("status"));
        assertEquals(Boolean.FALSE, view(sleepOn.isPaused(null, null).getEntity()).raw("is_paused"));
    }

    @Test
    @Timeout(120)
    void weightTransferRoutesAnswer404WithoutDevModeAndServeWithIt() {
        assertGated("init_weight_transfer_engine", () -> weightsOff.initWeightTransferEngine(null, null, null));
        assertGated("start_weight_update", () -> weightsOff.startWeightUpdate(null, null, null));
        assertGated("update_weights", () -> weightsOff.updateWeights("[]", null, null));
        assertGated("finish_weight_update", () -> weightsOff.finishWeightUpdate(null, null, null));
        assertGated("update_weight_version", () -> weightsOff.updateWeightVersion("{\"new_version\": \"v1\"}",
                null, null));
        assertGated("weight_info", () -> weightsOff.weightInfo(null, null));
        assertGated("shutdown_weight_transfer_engine", () -> weightsOff.shutdownWeightTransferEngine(null, null,
                null));

        assertEquals(200, weightsOn.initWeightTransferEngine(null, null, null).getStatus());
        assertEquals(200, weightsOn.startWeightUpdate(null, null, null).getStatus());
        assertEquals(400, statusOf(() -> weightsOn.updateWeights("", null, null)),
                "the route serves; the empty body is refused by name");
        assertEquals(200, weightsOn.finishWeightUpdate(null, null, null).getStatus());
        JsonView stamped = view(weightsOn.updateWeightVersion("{\"new_version\": \"v2\"}", null, null).getEntity());
        assertEquals("v2", stamped.str("new_version"));
        JsonView info = view(weightsOn.weightInfo(null, null).getEntity());
        assertEquals("v2", info.str("weight_version"));
        assertEquals(200, weightsOn.shutdownWeightTransferEngine(null, null, null).getStatus());
    }

    @Test
    @Timeout(120)
    void collectiveRpcAnswers404WithoutDevModeAndServesWithIt() {
        assertGated("collective_rpc", () -> rpcOff.collectiveRpc("{\"method\": \"get_weight_version\"}"));

        JsonView versions = view(rpcOn.collectiveRpc("{\"method\": \"get_weight_version\"}").getEntity());
        List<?> results = (List<?>) versions.raw("results");
        assertEquals(1, results.size(), "one result per replica");
        assertEquals("default", results.get(0));
        assertEquals(400, statusOf(() -> rpcOn.collectiveRpc("{\"method\": \"frobnicate\"}")),
                "the route serves; an unknown method is refused by name");
        assertEquals(200, rpcOn.collectiveRpc("{\"method\": \"shutdown_weight_transfer_engine\"}").getStatus());
    }

    // ------------------------------------------------------------------ router mode

    @Test
    void routerModeGatesBeforeTheModelQuery() {
        // the gate precedes the model query: without a model name the answer stays 404, not the router's 400
        assertGated("router sleep", () -> routerSleepOff.sleep("0", "abort", null, null));
        assertEquals(404, statusOf(() -> routerOpsOff.serverInfo(null, null, null)));
        assertEquals(404, statusOf(() -> routerWeightsOff.weightInfo(null, null)));
        assertEquals(404, statusOf(() -> routerRpcOff.collectiveRpc(null, null, null)));
        assertEquals(404, statusOf(() -> routerOpsOff.serverInfo(null, "zzz", null)));

        JsonView info = view(routerOpsOn.serverInfo(null, "a", null).getEntity());
        assertEquals("a", info.str("model_alias"), "the routed copy serves");
        JsonView routedVersions = view(routerRpcOn.collectiveRpc(
                "{\"method\": \"get_weight_version\"}", "a", null).getEntity());
        assertEquals("default", ((List<?>) routedVersions.raw("results")).get(0), "the routed call serves");
        assertEquals(Boolean.FALSE, view(new SleepResource(unloaded, router, DevMode.ON, SleepModeSwitch.ON)
                .isSleeping("a", null).getEntity()).raw("is_sleeping"));
        JsonView weight = view(routerWeightsOn.weightInfo("a", null).getEntity());
        assertEquals("default", weight.str("weight_version"));
    }

    // ------------------------------------------------------------------ the routes outside the dev mode

    @Test
    void routesOutsideTheDevModeStayReachable() {
        assertEquals(200, opsOff.version().getStatus());
        assertEquals(200, opsOff.load().getStatus());
        assertEquals(200, opsOff.ping().getStatus());
        assertEquals(200, opsOff.tokenizerInfo().getStatus());
        assertEquals(200, opsOff.startProfile(null).getStatus());
        assertEquals(200, opsOff.stopProfile().getStatus());
        assertNotEquals(404, statusOf(() -> sleepOff.checkpointStatus()),
                "the checkpoint window is vLLM's sleep-mode route, not a development one");
    }

    /** Asserts that {@code call} answers the 404 an unregistered route does, naming the dev-mode switch. */
    private static void assertGated(String route, Runnable call) {
        assertEquals(404, statusOf(call), route);
        String message = errorOf(call);
        assertTrue(message.contains(DevMode.FLAG) && message.contains(DevMode.PROPERTY), route + ": " + message);
    }
}
