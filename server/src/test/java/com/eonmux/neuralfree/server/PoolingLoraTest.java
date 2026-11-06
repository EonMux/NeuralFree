package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Per-request LoRA adapters on the pooling endpoints (vLLM serves LoRA on pooling models): {@code /v1/embeddings},
 * llama.cpp's {@code /embeddings}, {@code /v1/rerank}, {@code /classify} and {@code /score} select adapters by
 * {@code model} or the {@code lora} list exactly as generation does, and each answers what a model loaded with that
 * adapter ({@code --lora}) answers; selecting nothing runs the base model, and an unknown selection refuses by name.
 */
class PoolingLoraTest {

    private static final String BASE = "base";
    private static final double EPS = 1e-5;

    private static EngineService bert;
    private static EngineService bertRef;
    private static EngineService bertPlain;
    private static EngineService rank;
    private static EngineService rankRef;
    private static EngineService rankPlain;

    @BeforeAll
    static void open() throws IOException {
        Path adapter = LoraAdapterFiles.bert(2000, 4, 8.0f);
        Path bertModel = SyntheticModels.bert().writeTempFile();
        Path rankModel = SyntheticModels.rank("neg", "pos").writeTempFile();
        ServerOptions options = new ServerOptions(BASE, ServerOptions.DEFAULT_TIMEOUT);
        ServerLoras modules = new ServerLoras(List.of(new ServerLoras.Module("ad", adapter.toString())), false);
        bert = new EngineService(bertModel.toString(), ServerTestSupport.config().build(), options, modules);
        bertRef = new EngineService(bertModel.toString(), ServerTestSupport.config()
                .addLora(adapter.toString(), 1.0f).build(), options);
        bertPlain = new EngineService(bertModel.toString(), ServerTestSupport.config().build(), options);
        rank = new EngineService(rankModel.toString(), ServerTestSupport.config().build(), options, modules);
        rankRef = new EngineService(rankModel.toString(), ServerTestSupport.config()
                .addLora(adapter.toString(), 1.0f).build(), options);
        rankPlain = new EngineService(rankModel.toString(), ServerTestSupport.config().build(), options);
    }

    @AfterAll
    static void close() {
        for (EngineService s : new EngineService[]{bert, bertRef, bertPlain, rank, rankRef, rankPlain}) {
            s.close();
        }
    }

    private static double[] vector(Object raw) {
        List<?> values = (List<?>) raw;
        double[] out = new double[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) values.get(i)).doubleValue();
        }
        return out;
    }

    private static double[] embedding(EngineService on, String extra) {
        JsonView body = view(new PoolingResource(on).embeddings("{\"input\": \"hello world\"" + extra + "}")
                .getEntity());
        return vector(at(body, "data", 0).raw("embedding"));
    }

    private static double[] nativeEmbedding(EngineService on, String extra) {
        List<?> out = (List<?>) ServerTestSupport.parsed(new PoolingResource(on)
                .nativeEmbeddings("{\"content\": \"hello world\"" + extra + "}").getEntity());
        List<?> vectors = (List<?>) ((Map<?, ?>) out.get(0)).get("embedding");
        return vector(vectors.get(0));
    }

    @Test
    void embeddingsSelectTheAdapterByModelOrList() {
        double[] adapted = embedding(bertRef, "");
        double[] base = embedding(bertPlain, "");
        assertFalse(Arrays.equals(adapted, base), "the adapter must change the embedding");
        assertArrayEquals(base, embedding(bert, ""), EPS);
        assertArrayEquals(adapted, embedding(bert, ", \"model\": \"ad\""), EPS);
        assertArrayEquals(adapted, embedding(bert, ", \"lora\": [{\"id\": 0, \"scale\": 1.0}]"), EPS);
        assertArrayEquals(base, embedding(bert, ", \"lora\": [{\"id\": 0, \"scale\": 0}]"), EPS);
        assertEquals("ad", view(new PoolingResource(bert).embeddings("{\"input\": \"hi\", \"model\": \"ad\"}")
                .getEntity()).str("model"));
        assertArrayEquals(nativeEmbedding(bertRef, ""),
                nativeEmbedding(bert, ", \"lora\": [{\"id\": 0, \"scale\": 1.0}]"), EPS);
        String unknown = errorOf(() -> new PoolingResource(bert).embeddings("{\"input\": \"hi\","
                + " \"lora\": [{\"id\": 4, \"scale\": 1}]}"));
        assertTrue(unknown.contains("lora") && unknown.contains("4"), unknown);
    }

    @Test
    void scoreSelectsTheAdapterOnABiEncoder() {
        String body = "{\"queries\": \"a cat\", \"documents\": [\"a dog\", \"the sea\"]";
        assertScores(bertRef, bert, body, "/score");
    }

    @Test
    void rerankClassifyAndScoreSelectTheAdapterOnACrossEncoder() {
        String rerank = "{\"query\": \"a cat\", \"documents\": [\"a dog\", \"the sea\"]";
        double[] adapted = relevance(rankRef, rerank + "}");
        assertFalse(Arrays.equals(adapted, relevance(rankPlain, rerank + "}")),
                "the adapter must change the scores");
        assertArrayEquals(adapted, relevance(rank, rerank + ", \"model\": \"ad\"}"), EPS);
        assertArrayEquals(relevance(rankPlain, rerank + "}"), relevance(rank, rerank + "}"), EPS);

        String classify = "{\"input\": [\"a cat\", \"the sea\"]";
        assertArrayEquals(probs(rankRef, classify + "}"),
                probs(rank, classify + ", \"lora\": [{\"id\": 0, \"scale\": 1}]}"), EPS);
        assertScores(rankRef, rank, "{\"queries\": \"a cat\", \"documents\": [\"a dog\", \"the sea\"]", "/score");
    }

    /** The {@code /score} answers of {@code on} with the adapter selected equal those of {@code reference}. */
    private static void assertScores(EngineService reference, EngineService on, String body, String what) {
        double[] expected = scores(reference, body + "}");
        assertArrayEquals(expected, scores(on, body + ", \"model\": \"ad\"}"), EPS, what);
    }

    private static double[] scores(EngineService on, String body) {
        JsonView v = view(new PoolingResource(on).score(body).getEntity());
        int n = ServerTestSupport.size(v, "data");
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = at(v, "data", i).f64("score");
        }
        return out;
    }

    /** Relevance scores in document order. */
    private static double[] relevance(EngineService on, String body) {
        JsonView v = view(new PoolingResource(on).rerank(body).getEntity());
        int n = ServerTestSupport.size(v, "results");
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            JsonView r = at(v, "results", i);
            out[(int) r.i64("index")] = r.f64("relevance_score");
        }
        return out;
    }

    private static double[] probs(EngineService on, String body) {
        JsonView v = view(new PoolingResource(on).classify(body).getEntity());
        int n = ServerTestSupport.size(v, "data");
        double[] out = new double[0];
        for (int i = 0; i < n; i++) {
            double[] p = vector(at(v, "data", i).raw("probs"));
            double[] next = Arrays.copyOf(out, out.length + p.length);
            System.arraycopy(p, 0, next, out.length, p.length);
            out = next;
        }
        return out;
    }
}
