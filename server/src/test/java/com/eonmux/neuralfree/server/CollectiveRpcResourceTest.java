package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.mm.ClipDualFixtures;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The development collective RPC route (vLLM's {@code POST /collective_rpc}) over a two-replica pool: every method
 * answers one result per replica in pool order, the weight-transfer lifecycle it drives changes the very next
 * completion and stamps the pool's version, the reads report each replica's state, and an unknown method or
 * argument refuses by name. The timeout bounds a replica's wait and refuses a value that is not a positive number.
 *
 * <p>The tests run in a declared order: the refusals and the uninitialized reads come first, the update round and
 * the version stamps follow, and the shutdown ends the lifecycle before the busy-engine refusal.</p>
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class CollectiveRpcResourceTest {

    private static final int N_EMBD = 64;
    private static final int N_VOCAB = 320;
    private static final String NORM = "blk.0.attn_norm.weight";
    private static final String OUTPUT = "output.weight";

    private static EngineService service;
    private static CollectiveRpcResource rpc;
    private static LlamaCompletionResource completions;
    private static Path model;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        // round robin routes each warm-up completion to the next replica, so both map the tensors an update targets
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, null,
                        ServerOptions.DEFAULT_SLEEP_IDLE_SECONDS, 2, null, false,
                        new DpLayout(2, 0, LoadBalance.ROUND_ROBIN), null));
        rpc = new CollectiveRpcResource(service, DevMode.ON);
        completions = new LlamaCompletionResource(service);
    }

    @AfterAll
    static void close() throws IOException {
        service.close();
        Files.deleteIfExists(model);
    }

    // ------------------------------------------------------------------ helpers

    /** A short greedy completion body over {@code prompt} the engine finishes deterministically. */
    private static String completion(String prompt, int nPredict) {
        return "{\"prompt\": \"" + prompt + "\", \"n_predict\": " + nPredict + ", \"ignore_eos\": true,"
                + " \"temperature\": 0}";
    }

    /**
     * The chosen token's logprob of a one-token greedy completion ({@code n_probs}): the weights' fingerprint on
     * the next-token distribution — the synthetic model's tokens decode to empty text, so the distribution is
     * the observable an update must move.
     */
    private static double firstLogprob() {
        String body = "{\"prompt\": \"hello\", \"n_predict\": 1, \"ignore_eos\": true, \"temperature\": 0,"
                + " \"n_probs\": 1}";
        JsonView completion = view(completions.completion(body).getEntity());
        List<?> steps = (List<?>) completion.raw("completion_probabilities");
        assertEquals(1, steps.size());
        return JsonView.of(steps.get(0), "completion_probabilities[0]").f64("logprob");
    }

    /** A call on the wired service, its JSON answer as a view. */
    private static JsonView call(String body) {
        return view(rpc.collectiveRpc(body).getEntity());
    }

    /** The per-replica results of a call. */
    private static List<?> results(String body) {
        return (List<?>) call(body).raw("results");
    }

    /** The update payload of one F32 tensor: its name, dtype, ggml shape and base64 little-endian data. */
    private static String tensor(String name, long[] ne, float[] values) {
        StringBuilder dims = new StringBuilder("[");
        for (int i = 0; i < ne.length; i++) {
            dims.append(i > 0 ? ", " : "").append(ne[i]);
        }
        dims.append(']');
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        return "{\"name\": \"" + name + "\", \"dtype\": \"F32\", \"ne\": " + dims + ", \"data_b64\": \""
                + Base64.getEncoder().encodeToString(bytes.array()) + "\"}";
    }

    private static float[] fill(int n, float v) {
        float[] fs = new float[n];
        Arrays.fill(fs, v);
        return fs;
    }

    /** A JSON string literal for a payload carried inside the RPC body. */
    private static String quoted(String text) {
        return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    /** Serves one completion on each replica, so both have mapped the tensors an update targets. */
    private static void warmBothReplicas() {
        completions.completion(completion("hello", 4));
        completions.completion(completion("world", 4));
        List<BatchStats> stats = service.replicaStats();
        assertEquals(2, stats.size());
        assertTrue(stats.get(0).promptTokens() > 0, "replica 0 never served");
        assertTrue(stats.get(1).promptTokens() > 0, "replica 1 never served");
    }

    // ------------------------------------------------------------------ refusals and uninitialized reads

    @Test
    @Timeout(120)
    @Order(1)
    void theRouteRefusesBadBodiesAndUnknownMethodsByName() {
        String unknown = errorOf(() -> rpc.collectiveRpc("{\"method\": \"frobnicate\"}"));
        assertTrue(unknown.contains("frobnicate") && unknown.contains("init_weight_transfer_engine"), unknown);

        assertTrue(errorOf(() -> rpc.collectiveRpc("{\"args\": []}")).contains("method"));
        assertTrue(errorOf(() -> rpc.collectiveRpc("[]")).contains("object"));
        assertTrue(errorOf(() -> rpc.collectiveRpc("{\"method\": \"get_weight_version\", \"n\": 1}")).contains("'n'"));
        assertTrue(errorOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"args\": 3}")).contains("args"));
        assertTrue(errorOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"args\": [3]}")).contains("args[0]"));
        assertTrue(errorOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"kwargs\": []}")).contains("kwargs"));
        assertTrue(errorOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"kwargs\": {\"a\": 3}}")).contains("kwargs.a"));
        assertTrue(errorOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"timeout\": 0}")).contains("timeout"));
        assertTrue(errorOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"timeout\": \"soon\"}")).contains("timeout"));

        assertEquals(400, statusOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"args\": [\"x\"]}")), "the read takes no argument");
        assertEquals(400, statusOf(() -> rpc.collectiveRpc(
                "{\"method\": \"get_weight_version\", \"kwargs\": {\"k\": \"v\"}}")));
        assertEquals(400, statusOf(() -> rpc.collectiveRpc(
                "{\"method\": \"update_weights\", \"args\": [\"{}\"]}")), "an empty update carries no tensors");
        assertEquals(400, statusOf(() -> rpc.collectiveRpc(
                "{\"method\": \"update_weight_version\"}")), "the version is required");
        assertTrue(errorOf(() -> rpc.collectiveRpc("{\"method\": \"init_weight_transfer_engine\", \"args\":"
                + " [" + quoted("{\"backend\": 3}") + "]}")).contains("backend"),
                "a wrongly typed backend is refused by name");
    }

    @Test
    @Timeout(120)
    @Order(2)
    void theUninitializedPoolReportsItsReplicasAndRefusesTheLifecycle() {
        List<?> versions = results("{\"method\": \"get_weight_version\"}");
        assertEquals(2, versions.size(), "one result per replica");
        assertEquals(List.of("default", "default"), versions);

        assertEquals(List.of(true, true), results("{\"method\": \"reset_prefix_cache\"}"),
                "nothing cached on either replica");
        assertEquals(2, results("{\"method\": \"reset_prefix_cache\", \"timeout\": 30.0}").size(),
                "a timeout is a positive number of seconds");

        assertEquals(503, statusOf(() -> rpc.collectiveRpc("{\"method\": \"start_weight_update\"}")));
        assertTrue(errorOf(() -> rpc.collectiveRpc("{\"method\": \"start_weight_update\"}"))
                .contains("init_weight_transfer_engine"));
        assertEquals(503, statusOf(() -> rpc.collectiveRpc(
                "{\"method\": \"update_weights\", \"args\": ["
                        + quoted(tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 1f))) + "]}")));
        assertEquals(503, statusOf(() -> rpc.collectiveRpc("{\"method\": \"finish_weight_update\"}")));
        assertEquals(503, statusOf(() -> rpc.collectiveRpc(
                "{\"method\": \"update_weight_version\", \"args\": [\"v1\"]}")));

        List<?> shut = results("{\"method\": \"shutdown_weight_transfer_engine\"}");
        assertEquals(2, shut.size());
        assertNull(shut.get(0), "a shutdown without an initialization is a no-op");
        assertNull(shut.get(1));
    }

    @Test
    @Timeout(180)
    void aModelWithoutEngineReplicasRefusesTheInitialization() throws IOException {
        Path clipModel = ClipDualFixtures.twoTower();
        ModelSourceOptions source = ModelSourceOptions.builder()
                .tokenizerDir(ClipDualFixtures.tokenizerDir()).build();
        try (EngineService clip = new EngineService(clipModel.toString(),
                ServerTestSupport.config().sourceOptions(source).build())) {
            CollectiveRpcResource clipRpc = new CollectiveRpcResource(clip, DevMode.ON);
            String e = errorOf(() -> clipRpc.collectiveRpc("{\"method\": \"init_weight_transfer_engine\"}"));
            assertTrue(e.contains("dual encoder"), e);
            assertEquals(409, statusOf(
                    () -> clipRpc.collectiveRpc("{\"method\": \"init_weight_transfer_engine\"}")));
        }
    }

    // ------------------------------------------------------------------ the update round

    @Test
    @Timeout(180)
    @Order(3)
    void anUpdateRoundThroughTheRpcChangesTheGenerationAndStampsItsVersion() {
        warmBothReplicas();
        double before = firstLogprob();

        assertEquals(Arrays.asList(null, null), results("{\"method\": \"init_weight_transfer_engine\","
                + " \"args\": [\"inprocess\"]}"));
        assertEquals(Arrays.asList(null, null), results("{\"method\": \"start_weight_update\"}"));

        String update = "[" + tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 3f)) + ", "
                + tensor(OUTPUT, new long[]{N_EMBD, N_VOCAB}, fill(N_EMBD * N_VOCAB, 0.5f)) + "]";
        assertEquals(Arrays.asList(null, null), results("{\"method\": \"update_weights\", \"kwargs\": {"
                + "\"update_info\": " + quoted(update) + "}}"), "vLLM's keyword carries the serialized payload");

        assertEquals(Arrays.asList(null, null), results("{\"method\": \"finish_weight_update\", \"args\":"
                + " [" + quoted("{\"weight_version\": \"rl-step-3\"}") + "]}"));

        assertEquals(List.of("rl-step-3", "rl-step-3"), results("{\"method\": \"get_weight_version\"}"),
                "every replica reports the committed label");
        assertEquals("rl-step-3", service.weightTransfer().version(), "the pool reports it on /weight_info");
        assertNotEquals((Double) before, (Double) firstLogprob(), "the update reaches the very next completion");
    }

    @Test
    @Timeout(180)
    @Order(4)
    void theVersionStampAndTheBareApplyWorkWithoutARound() {
        assertEquals(Arrays.asList(null, null), results("{\"method\": \"update_weight_version\", \"args\":"
                + " [" + quoted("{\"new_version\": \"rl-step-4\"}") + "]}"));
        assertEquals(List.of("rl-step-4", "rl-step-4"), results("{\"method\": \"get_weight_version\"}"));

        double before = firstLogprob();
        // a head filled with one constant makes every logit equal, so token 0's row is raised to change the
        // distribution's shape, not just its scale
        float[] head = fill(N_EMBD * N_VOCAB, 0.9f);
        Arrays.fill(head, 0, N_EMBD, 2f);
        assertEquals(Arrays.asList(null, null), results("{\"method\": \"update_weights\", \"args\": ["
                + quoted(tensor(OUTPUT, new long[]{N_EMBD, N_VOCAB}, head)) + "]}"));
        assertNotEquals((Double) before, (Double) firstLogprob(), "an apply without a round still reaches the"
                + " completions");
    }

    @Test
    @Timeout(180)
    @Order(5)
    void theShutdownStopsFurtherUpdatesAndTheLabelSurvives() {
        List<?> shut = results("{\"method\": \"shutdown_weight_transfer_engine\"}");
        assertEquals(2, shut.size());
        assertNull(shut.get(0));
        assertNull(shut.get(1));

        assertEquals(503, statusOf(() -> rpc.collectiveRpc(
                "{\"method\": \"update_weights\", \"args\": ["
                        + quoted(tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 1f))) + "]}")));
        assertTrue(errorOf(() -> rpc.collectiveRpc("{\"method\": \"start_weight_update\"}"))
                .contains("not initialized"));
        assertEquals(List.of("rl-step-4", "rl-step-4"), results("{\"method\": \"get_weight_version\"}"),
                "the committed label survives the shutdown");

        assertEquals(Arrays.asList(null, null), results("{\"method\": \"init_weight_transfer_engine\"}"));
        assertEquals(List.of("rl-step-4", "rl-step-4"), results("{\"method\": \"get_weight_version\"}"),
                "a re-initialization keeps the committed label");
    }

    @Test
    @Timeout(180)
    @Order(6)
    void aBusyEngineRefusesAnUpdateByName() throws Exception {
        assertEquals(Arrays.asList(null, null), results("{\"method\": \"start_weight_update\"}"));
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newSingleThreadExecutor();
        try {
            java.util.concurrent.Future<?> generating = executor.submit(
                    () -> completions.completion(completion("hello", 64)));
            long deadline = System.currentTimeMillis() + 30_000;
            while (service.stats().running() == 0 && System.currentTimeMillis() < deadline) {
                Thread.sleep(5);
            }
            assertTrue(service.stats().running() > 0, "the generation never reached the engine");

            String e = errorOf(() -> rpc.collectiveRpc("{\"method\": \"update_weights\", \"args\": ["
                    + quoted(tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 5f))) + "]}"));
            assertTrue(e.contains("running requests") && e.contains("/pause"), e);

            generating.get(60, java.util.concurrent.TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
        assertEquals(Arrays.asList(null, null), results("{\"method\": \"finish_weight_update\"}"));
        results("{\"method\": \"shutdown_weight_transfer_engine\"}");
    }
}
