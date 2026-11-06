package com.eonmux.neuralfree.server;

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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The live weight transfer endpoints (vLLM's online RL routes): an update round over the running engine changes
 * the very next completion, a bare apply works without a round, the committed version reports on
 * {@code /weight_info} and survives a shutdown, and every refusal — an unknown backend, a wrong tensor, shape or
 * dtype, an update beside a running request, an update before the initialization — names itself.
 *
 * <p>The tests run in a declared order: the refusals before an initialization come first, the update round and
 * the bare apply follow on the initialized engine, and the shutdown ends the lifecycle.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class WeightTransferResourceTest {

    private static final int N_EMBD = 64;
    private static final int N_VOCAB = 320;
    private static final String NORM = "blk.0.attn_norm.weight";
    private static final String OUTPUT = "output.weight";

    private static EngineService service;
    private static WeightTransferResource transfer;
    private static LlamaCompletionResource completions;
    private static Path model;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2));
        transfer = new WeightTransferResource(service, DevMode.ON);
        completions = new LlamaCompletionResource(service);
    }

    @AfterAll
    static void close() throws IOException {
        service.close();
        Files.deleteIfExists(model);
    }

    // ------------------------------------------------------------------ helpers

    /** A short greedy completion body the engine finishes deterministically. */
    private static String completion(int nPredict) {
        return "{\"prompt\": \"hello\", \"n_predict\": " + nPredict + ", \"ignore_eos\": true, \"temperature\": 0}";
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

    /** The update body over one F32 tensor: its name, dtype, ggml shape and base64 little-endian data. */
    private static String tensor(String name, long[] ne, float[] values) {
        return tensorBody(name, "F32", ne, f32Bytes(values));
    }

    /** The update body over one tensor of {@code dtype} with raw little-endian bytes. */
    private static String tensorBody(String name, String dtype, long[] ne, byte[] data) {
        StringBuilder dims = new StringBuilder("[");
        for (int i = 0; i < ne.length; i++) {
            dims.append(i > 0 ? ", " : "").append(ne[i]);
        }
        dims.append(']');
        return "{\"name\": \"" + name + "\", \"dtype\": \"" + dtype + "\", \"ne\": " + dims
                + ", \"data_b64\": \"" + Base64.getEncoder().encodeToString(data) + "\"}";
    }

    private static byte[] f32Bytes(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        return bytes.array();
    }

    private static float[] fill(int n, float v) {
        float[] fs = new float[n];
        Arrays.fill(fs, v);
        return fs;
    }

    // ------------------------------------------------------------------ the lifecycle

    @Test
    @Timeout(120)
    @Order(1)
    void theRoutesRefuseBeforeAnInitAndWeightInfoReportsTheDefault() {
        JsonView info = view(transfer.weightInfo().getEntity());
        assertEquals("default", info.str("weight_version"));
        assertEquals("inprocess", info.str("backend"));
        assertEquals(Boolean.FALSE, info.raw("initialized"));

        assertEquals(503, statusOf(() -> transfer.startWeightUpdate(null)));
        assertTrue(errorOf(() -> transfer.startWeightUpdate(null)).contains("init_weight_transfer_engine"));
        assertEquals(503, statusOf(() -> transfer.updateWeights(tensor(OUTPUT, new long[]{N_EMBD, N_VOCAB},
                fill(N_EMBD * N_VOCAB, 0.5f)))));
        assertEquals(503, statusOf(() -> transfer.finishWeightUpdate(null)));
        assertEquals(503, statusOf(() -> transfer.updateWeightVersion("{\"new_version\": \"v1\"}")));

        assertEquals(200, statusOf(() -> transfer.shutdownWeightTransferEngine(null)),
                "a shutdown without an initialization is a no-op");
    }

    @Test
    @Timeout(120)
    @Order(2)
    void anUpdateRoundChangesTheGenerationAndStampsItsVersion() {
        double before = firstLogprob();

        JsonView init = view(transfer.initWeightTransferEngine("{\"backend\": \"inprocess\"}").getEntity());
        assertEquals("Weight transfer initialized", init.str("message"));
        assertEquals("inprocess", init.str("backend"));
        assertEquals("default", init.str("weight_version"));
        assertEquals(Boolean.TRUE, view(transfer.weightInfo().getEntity()).raw("initialized"));

        assertEquals("Weight update started", view(transfer.startWeightUpdate(null).getEntity()).str("message"));

        String update = "[" + tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 3f)) + ", "
                + tensor(OUTPUT, new long[]{N_EMBD, N_VOCAB}, fill(N_EMBD * N_VOCAB, 0.5f)) + "]";
        JsonView updated = view(transfer.updateWeights(update).getEntity());
        assertEquals("Weights updated", updated.str("message"));
        assertEquals(2L, updated.i64("tensors"));

        JsonView finished = view(transfer.finishWeightUpdate("{\"weight_version\": \"rl-step-9\"}").getEntity());
        assertEquals("Weight update finished", finished.str("message"));
        assertEquals("rl-step-9", finished.str("weight_version"));

        assertNotEquals((Double) before, (Double) firstLogprob(), "the update reaches the very next completion");
        assertEquals("rl-step-9", view(transfer.weightInfo().getEntity()).str("weight_version"));
    }

    @Test
    @Timeout(120)
    @Order(3)
    void aBareApplyChangesTheGenerationWithoutARound() {
        double before = firstLogprob();
        // a head filled with one constant makes every logit equal, so token 0's row is raised to change the
        // distribution's shape, not just its scale
        float[] head = fill(N_EMBD * N_VOCAB, 0.9f);
        Arrays.fill(head, 0, N_EMBD, 2f);
        assertEquals(200, statusOf(() -> transfer.updateWeights(tensor(OUTPUT, new long[]{N_EMBD, N_VOCAB},
                head))));
        assertNotEquals((Double) before, (Double) firstLogprob(), "an apply without a round still reaches the"
                + " completions");
    }

    @Test
    @Timeout(120)
    @Order(4)
    void anUpdateInfoWrapperCarriesTheSamePayload() {
        double before = firstLogprob();
        // vLLM's HTTP route embeds the payload: the object under update_info, not the bare tensor
        float[] head = fill(N_EMBD * N_VOCAB, 0.7f);
        Arrays.fill(head, 0, N_EMBD, 4f);
        String body = "{\"update_info\": " + tensor(OUTPUT, new long[]{N_EMBD, N_VOCAB}, head) + "}";
        JsonView updated = view(transfer.updateWeights(body).getEntity());
        assertEquals("Weights updated", updated.str("message"));
        assertEquals(1L, updated.i64("tensors"));
        assertNotEquals((Double) before, (Double) firstLogprob(), "the wrapped payload reaches the engine");

        assertTrue(errorOf(() -> transfer.updateWeights(
                "{\"update_info\": " + tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 1f)) + ", \"x\": 1}"))
                .contains("'x'"), "the wrapper takes no other field");
    }

    @Test
    @Timeout(120)
    @Order(5)
    void theRoutesRefuseBadInputByName() {
        String e = errorOf(() -> transfer.initWeightTransferEngine("{\"backend\": \"nccl\"}"));
        assertTrue(e.contains("nccl") && e.contains("inprocess"));

        assertTrue(errorOf(() -> transfer.initWeightTransferEngine("{\"rank\": 3}")).contains("rank"));

        assertTrue(errorOf(() -> transfer.updateWeights(tensor("blk.9.attn_norm.weight", new long[]{N_EMBD},
                fill(N_EMBD, 1f)))).contains("blk.9.attn_norm.weight"));

        String wrongShape = errorOf(() -> transfer.updateWeights(tensor(NORM, new long[]{N_EMBD / 2},
                fill(N_EMBD / 2, 1f))));
        assertTrue(wrongShape.contains("[64]") && wrongShape.contains("[32]"), wrongShape);

        assertTrue(errorOf(() -> transfer.updateWeights(
                "{\"name\": \"" + OUTPUT + "\", \"dtype\": \"F32\", \"ne\": [" + N_EMBD + ", " + N_VOCAB + "]"
                        + ", \"data_b64\": \"\", \"src\": 1}")).contains("src"));

        assertTrue(errorOf(() -> transfer.updateWeights(tensorBody(OUTPUT, "F32", new long[]{N_EMBD, N_VOCAB},
                new byte[4]))).contains("needs"));

        assertTrue(errorOf(() -> transfer.updateWeights(
                tensorBody(OUTPUT, "FLOAT32", new long[]{N_EMBD, N_VOCAB},
                        new byte[N_EMBD * N_VOCAB * Float.BYTES]))).contains("not a ggml type name"));

        String bf16 = errorOf(() -> transfer.updateWeights(tensorBody(OUTPUT, "BF16",
                new long[]{N_EMBD, N_VOCAB}, new byte[N_EMBD * N_VOCAB * 2])));
        assertTrue(bf16.contains("BF16"), bf16);

        assertTrue(errorOf(() -> transfer.updateWeights("")).contains("empty"));
        assertTrue(errorOf(() -> transfer.updateWeights("{\"weight_version\": 9}")).contains("weight_version"));
    }

    @Test
    @Timeout(120)
    @Order(6)
    void theShutdownStopsFurtherUpdatesAndTheLabelSurvives() {
        assertEquals(200, statusOf(() -> transfer.shutdownWeightTransferEngine(null)));
        assertEquals(Boolean.FALSE, view(transfer.weightInfo().getEntity()).raw("initialized"));

        String e = errorOf(() -> transfer.updateWeights(tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 1f))));
        assertTrue(e.contains("not initialized") && e.contains("init_weight_transfer_engine"), e);

        assertEquals("rl-step-9", view(transfer.weightInfo().getEntity()).str("weight_version"),
                "the committed label survives the shutdown");

        JsonView reinit = view(transfer.initWeightTransferEngine(null).getEntity());
        assertEquals("Weight transfer initialized", reinit.str("message"));
        assertEquals("rl-step-9", reinit.str("weight_version"), "a re-initialization keeps the committed label");
        assertEquals(200, statusOf(() -> transfer.shutdownWeightTransferEngine(null)));
    }

    @Test
    @Timeout(180)
    @Order(7)
    void aBusyEngineRefusesAnUpdateByName() throws Exception {
        transfer.initWeightTransferEngine(null);
        transfer.startWeightUpdate(null);
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newSingleThreadExecutor();
        try {
            java.util.concurrent.Future<?> generating = executor.submit(() -> completions.completion(completion(64)));
            long deadline = System.currentTimeMillis() + 30_000;
            while (service.stats().running() == 0 && System.currentTimeMillis() < deadline) {
                Thread.sleep(5);
            }
            assertTrue(service.stats().running() > 0, "the generation never reached the engine");

            String e = errorOf(() -> transfer.updateWeights(tensor(NORM, new long[]{N_EMBD}, fill(N_EMBD, 5f))));
            assertTrue(e.contains("running requests") && e.contains("/pause"), e);

            generating.get(60, java.util.concurrent.TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
        assertEquals(200, statusOf(() -> transfer.finishWeightUpdate(null)));
    }
}
