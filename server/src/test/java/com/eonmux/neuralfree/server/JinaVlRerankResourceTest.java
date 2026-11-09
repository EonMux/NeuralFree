package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A jina_vl-shaped multimodal reranker (vLLM's {@code JinaVLForSequenceClassification}): a Qwen2-VL RANK model with
 * the ReLU two-layer score head, the document-first score template and the appended score token, beside a Qwen2-VL
 * projector. Text pairs score through the template with the score token appended; pairs whose sides hold images
 * score as one media prompt through the classifier on {@code /rerank} and {@code /score}, counting the image rows,
 * and the per-side truncation of {@code /score} refuses media sides by name.
 */
class JinaVlRerankResourceTest {

    private static final String TEMPLATE = "**Document**:\n{document}\n**Query**:\n{query}";
    private static final int SCORE_TOKEN = 100;
    private static final double TOLERANCE = 1e-5;

    private static EngineService service;
    private static PoolingResource pooling;
    private static String image;
    private static String otherImage;

    @BeforeAll
    static void open() throws IOException {
        int width = SyntheticModels.Config.llama().nEmbd();
        GgufTestWriter qwen = SyntheticModels.qwen2vl();
        qwen.u32(qwen.archKey("pooling_type"), SyntheticModels.POOLING_RANK);
        qwen.str(qwen.archKey("classifier.activation"), "relu");
        qwen.str("tokenizer.chat_template.rerank", TEMPLATE);
        qwen.i32Array(GgufKeys.NF_RERANK_SUFFIX_IDS, SCORE_TOKEN);
        qwen.tensorRandom("cls.weight", GGMLType.F32, 91, -0.2f, 0.2f, width, width);
        qwen.tensorRandom("cls.bias", GGMLType.F32, 92, -0.1f, 0.1f, width);
        qwen.tensorRandom("cls.output.weight", GGMLType.F32, 93, -0.2f, 0.2f, width, 1);
        qwen.tensorZeros("cls.output.bias", GGMLType.F32, 1);
        Path model = qwen.writeTempFile();
        Path mmproj = MmprojTestFiles.qwen2vl(width);
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(512).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        pooling = new PoolingResource(service);
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(16, 16));
        otherImage = "data:image/png;base64,"
                + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(9, 20));
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }

    private static String imagePart(String url) {
        return "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + url + "\"}}";
    }

    private static double rerankScore(String query, String document) {
        return at(ok(pooling.rerank("{\"query\": " + query + ", \"documents\": [" + document + "]}")), "results", 0)
                .f64("relevance_score");
    }

    @Test
    void aTextPairScoresTheFilledTemplateWithTheScoreTokenAppended() throws IOException {
        int[] prompt = service.encode(TEMPLATE.replace("{query}", "a cat").replace("{document}", "a dog"), false,
                true);
        int[] withToken = Arrays.copyOf(prompt, prompt.length + 1);
        withToken[prompt.length] = SCORE_TOKEN;
        JsonView body = ok(pooling.rerank("{\"query\": \"a cat\", \"documents\": [\"a dog\"]}"));
        assertEquals(service.pooling().rerankScore(null, withToken), at(body, "results", 0).f64("relevance_score"), TOLERANCE);
        assertEquals(withToken.length, body.obj("usage").i64("total_tokens"));
    }

    @Test
    void anImageDocumentScoresAsOneMediaPromptThroughTheClassifier() {
        String document = "{\"content\": [" + imagePart(image) + ", {\"type\": \"text\", \"text\": \"a cat\"}]}";
        JsonView ranked = ok(pooling.rerank("{\"query\": \"a cat\", \"documents\": [" + document + ", \"a cat\"]}"));
        long mediaTokens = ranked.obj("usage").i64("total_tokens");
        long textTokens = ok(pooling.rerank("{\"query\": \"a cat\", \"documents\": [\"a cat\"]}")).obj("usage")
                .i64("total_tokens");
        assertTrue(mediaTokens > textTokens + 1, mediaTokens + " media rows vs " + textTokens + " text tokens");
        double media = rerankScore("\"a cat\"", document);
        assertNotEquals(rerankScore("\"a cat\"", "\"a cat\""), media, 1e-7, "the image rows reach the classifier");
        assertEquals(media, rerankScore("\"a cat\"", document), TOLERANCE, "the same pair scores the same");
        JsonView scored = ok(pooling.score("{\"queries\": \"a cat\", \"documents\": [" + document + "]}"));
        assertEquals(media, at(scored, "data", 0).f64("score"), TOLERANCE, "/score answers the rerank score");
    }

    @Test
    void eachSideKeepsItsOwnImages() {
        String a = "{\"content\": [" + imagePart(image) + "]}";
        String b = "{\"content\": [" + imagePart(otherImage) + "]}";
        double ab = rerankScore(a, b);
        double ba = rerankScore(b, a);
        assertNotEquals(ab, ba, 1e-7, "the query image and the document image sit at their own markers");
        assertEquals(ab, rerankScore(a, b), TOLERANCE);
    }

    @Test
    void perSideTruncationRefusesAMediaSideByName() {
        String document = "{\"content\": [" + imagePart(image) + "]}";
        assertTrue(errorOf(() -> pooling.score("{\"queries\": \"a cat\", \"documents\": [" + document + "],"
                + " \"max_tokens_per_doc\": 4}")).contains("max_tokens_per_doc"));
        assertTrue(errorOf(() -> pooling.rerank("{\"query\": \"a cat\", \"documents\": [" + document + "],"
                + " \"truncate_prompt_tokens\": 8}")).contains("truncate_prompt_tokens"),
                "media rows keep their length, so the pair truncation refuses by name");
    }
}
