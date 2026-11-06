package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The vLLM score and rerank request fields over the real service, per field class: the task {@code instruction}
 * applied through the model's {@code {instruction}} score template (and refused where no score prompt can place
 * it), {@code chat_template_kwargs} folded beside it, the per-side caps {@code max_tokens_per_query} and {@code
 * max_tokens_per_doc}, the pair truncation ({@code truncate_prompt_tokens}, {@code truncation_side}), the echoed
 * {@code request_id}, the shared refusals ({@code priority}, {@code cache_salt}, {@code padding}, {@code
 * mm_processor_kwargs}, {@code top_n}) and the removed {@code normalize} and pooling tasks, refused where vLLM
 * refuses them.
 */
public class ScoringRequestFieldsTest {

    /** The score template of the template fixture: the instruction sits between the sides. */
    private static final String TEMPLATE = "i:{instruction}|q:{query}|d:{document}";
    private static final String INSTRUCTION = "rank by cat-ness";
    private static final double TOLERANCE = 1e-9;

    private static EngineService templateService;
    private static EngineService bareService;
    private static EngineService fixedTemplateService;
    private static EngineService bertService;
    private static PoolingResource templatePooling;
    private static PoolingResource barePooling;
    private static PoolingResource fixedPooling;
    private static PoolingResource biPooling;

    @BeforeAll
    static void open() throws IOException {
        templateService = service(SyntheticModels.rank("neg", "pos")
                .str("tokenizer.chat_template.rerank", TEMPLATE));
        bareService = service(SyntheticModels.rank("neg", "pos"));
        fixedTemplateService = service(SyntheticModels.rank("neg", "pos")
                .str("tokenizer.chat_template.rerank", "q:{query}|d:{document}"));
        bertService = service(SyntheticModels.bert());
        templatePooling = new PoolingResource(templateService);
        barePooling = new PoolingResource(bareService);
        fixedPooling = new PoolingResource(fixedTemplateService);
        biPooling = new PoolingResource(bertService);
    }

    private static EngineService service(com.eonmux.neuralfree.gguf.testutil.GgufTestWriter model) throws IOException {
        Path path = model.writeTempFile();
        return new EngineService(path.toString(), ServerTestSupport.config().build());
    }

    @AfterAll
    static void close() {
        templateService.close();
        bareService.close();
        fixedTemplateService.close();
        bertService.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }

    private static double firstScore(PoolingResource pooling, String body) {
        return at(ok(pooling.score(body)), "data", 0).f64("score");
    }

    private static double firstRelevance(PoolingResource pooling, String body) {
        return at(ok(pooling.rerank(body)), "results", 0).f64("relevance_score");
    }

    /** The classifier score of one manually filled template, the reference the endpoint must reproduce. */
    private static double filledTemplateScore(String query, String document, String instruction) throws IOException {
        String prompt = TEMPLATE.replace("{instruction}", instruction)
                .replace("{query}", query).replace("{document}", document);
        return templateService.rerankScore(null, templateService.encode(prompt, false, true));
    }

    @Test
    void instructionScoresThroughTheTemplatePlaceholder() throws IOException {
        double want = filledTemplateScore("a cat", "a dog", INSTRUCTION);
        assertEquals(want, firstScore(templatePooling,
                "{\"queries\": \"a cat\", \"documents\": \"a dog\", \"instruction\": \"" + INSTRUCTION + "\"}"),
                TOLERANCE, "/score applies the instruction through the template");
        assertEquals(want, firstRelevance(templatePooling,
                "{\"query\": \"a cat\", \"documents\": [\"a dog\"], \"instruction\": \"" + INSTRUCTION + "\"}"),
                TOLERANCE, "/rerank applies the same instruction");
        assertNotEquals(want, firstScore(templatePooling, "{\"queries\": \"a cat\", \"documents\": \"a dog\"}"),
                1e-7, "the instruction reaches the classifier");
    }

    @Test
    void instructionRefusesWhereNoScorePromptCanPlaceIt() {
        assertTrue(errorOf(() -> barePooling.score(
                        "{\"queries\": \"a\", \"documents\": \"b\", \"instruction\": \"x\"}"))
                .contains("instruction"), "no rerank prompt at all");
        assertTrue(errorOf(() -> fixedPooling.score(
                        "{\"queries\": \"a\", \"documents\": \"b\", \"instruction\": \"x\"}"))
                .contains("instruction"), "a rerank prompt without the placeholder");
        assertTrue(errorOf(() -> biPooling.score(
                        "{\"queries\": \"a\", \"documents\": \"b\", \"instruction\": \"x\"}"))
                .contains("instruction"), "a bi-encoder reads no instruction");
    }

    @Test
    void chatTemplateKwargsFoldWithTheTopLevelInstruction() throws IOException {
        double byKwarg = firstScore(templatePooling, "{\"queries\": \"a cat\", \"documents\": \"a dog\","
                + " \"chat_template_kwargs\": {\"instruction\": \"" + INSTRUCTION + "\"}}");
        assertEquals(filledTemplateScore("a cat", "a dog", INSTRUCTION), byKwarg, TOLERANCE);
        double overridden = firstScore(templatePooling, "{\"queries\": \"a cat\", \"documents\": \"a dog\","
                + " \"instruction\": \"first\", \"chat_template_kwargs\": {\"instruction\": \"" + INSTRUCTION + "\"}}");
        assertEquals(byKwarg, overridden, TOLERANCE, "the explicit kwarg wins over the top-level field");
        assertEquals(400, statusOf(() -> templatePooling.score("{\"queries\": \"a\", \"documents\": \"b\","
                + " \"chat_template_kwargs\": {\"chat_template\": \"x\"}}")));
        assertTrue(errorOf(() -> templatePooling.score("{\"queries\": \"a\", \"documents\": \"b\","
                        + " \"chat_template_kwargs\": {\"tools\": []}}")).contains("reserved keys"));
        assertTrue(errorOf(() -> templatePooling.score("{\"queries\": \"a\", \"documents\": \"b\","
                        + " \"chat_template_kwargs\": {\"enable_thinking\": false}}")).contains("enable_thinking"));
        assertTrue(errorOf(() -> templatePooling.score(
                "{\"queries\": \"a\", \"documents\": \"b\", \"chat_template_kwargs\": \"x\"}"))
                .contains("chat_template_kwargs"));
        assertEquals(400, statusOf(() -> templatePooling.score(
                "{\"queries\": \"a\", \"documents\": \"b\", \"instruction\": 5}")));
    }

    @Test
    void perSideCapsTruncateTheCrossEncoderPair() {
        String longDocument = "\"" + "word ".repeat(10) + "\"";
        long bare = ok(templatePooling.rerank("{\"query\": \"a cat\", \"documents\": [" + longDocument + "]}"))
                .obj("usage").i64("total_tokens");
        JsonView capped = ok(templatePooling.rerank("{\"query\": \"a cat\", \"documents\": [" + longDocument + "],"
                + " \"max_tokens_per_query\": 1, \"max_tokens_per_doc\": 2}"));
        assertTrue(capped.obj("usage").i64("total_tokens") < bare,
                capped.obj("usage").i64("total_tokens") + " capped vs " + bare);
        assertEquals(capped.obj("usage").i64("total_tokens"),
                ok(templatePooling.score("{\"queries\": \"a cat\", \"documents\": [" + longDocument + "],"
                        + " \"max_tokens_per_query\": 1, \"max_tokens_per_doc\": 2}")).obj("usage")
                        .i64("prompt_tokens"), "/score and /rerank cut alike");
        assertEquals(firstRelevance(templatePooling, "{\"query\": \"a cat\", \"documents\": [" + longDocument + "],"
                        + " \"max_tokens_per_query\": 1, \"max_tokens_per_doc\": 2}"),
                firstScore(templatePooling, "{\"queries\": \"a cat\", \"documents\": [" + longDocument + "],"
                        + " \"max_tokens_per_query\": 1, \"max_tokens_per_doc\": 2}"), TOLERANCE);
        assertTrue(errorOf(() -> barePooling.score("{\"queries\": \"a\", \"documents\": \"b\","
                + " \"max_tokens_per_doc\": -1}")).contains("max_tokens_per_doc"));
        assertTrue(errorOf(() -> barePooling.score("{\"queries\": \"a\", \"documents\": \"b\","
                + " \"max_tokens_per_query\": " + bareService.maxModelLength() + "}")).contains("max_model_len"));
    }

    @Test
    void truncatePromptTokensCutsTheComposedPair() throws IOException {
        int[] pair = bareService.rerankInput("aaaa", "bbbb");
        assertTrue(pair.length > 3, "the fixture pair packs more than the cut");
        String body = "{\"queries\": \"aaaa\", \"documents\": \"bbbb\", \"truncate_prompt_tokens\": 3}";
        JsonView scored = ok(barePooling.score(body));
        assertEquals(3L, scored.obj("usage").i64("prompt_tokens"), "the composed pair counts the cut");
        int[] last = Arrays.copyOfRange(pair, pair.length - 3, pair.length);
        assertEquals(bareService.rerankScore(null, last), at(scored, "data", 0).f64("score"), TOLERANCE,
                "truncation_side left keeps the last tokens");
        int[] first = Arrays.copyOf(pair, 3);
        assertEquals(bareService.rerankScore(null, first), firstScore(barePooling, body
                        .replace("}", ", \"truncation_side\": \"right\"}")), TOLERANCE,
                "truncation_side right keeps the first tokens");
    }

    @Test
    void batchedTextPairsComposePerPairWithTheOptions() throws IOException {
        int cut = 5;
        int[] first = templateService.rerankInput("aaaa", "bbbb", INSTRUCTION);
        int[] second = templateService.rerankInput("cccc", "dddd", INSTRUCTION);
        assertTrue(first.length > cut && second.length > cut, "the fixture pairs pack more than the cut");
        double want1 = templateService.rerankScore(null,
                Arrays.copyOfRange(first, first.length - cut, first.length));
        double want2 = templateService.rerankScore(null,
                Arrays.copyOfRange(second, second.length - cut, second.length));
        JsonView scored = ok(templatePooling.score("{\"queries\": [\"aaaa\", \"cccc\"],"
                + " \"documents\": [\"bbbb\", \"dddd\"], \"instruction\": \"" + INSTRUCTION + "\","
                + " \"truncate_prompt_tokens\": " + cut + "}"));
        assertEquals(want1, at(scored, "data", 0).f64("score"), TOLERANCE, "pair 0 composes with the instruction");
        assertEquals(want2, at(scored, "data", 1).f64("score"), TOLERANCE, "pair 1 composes with the instruction");
        assertEquals(2 * (long) cut, scored.obj("usage").i64("prompt_tokens"),
                "the usage counts every composed pair after the cut");
        JsonView ranked = ok(templatePooling.rerank("{\"query\": \"aaaa\", \"documents\": [\"bbbb\", \"dddd\"],"
                + " \"instruction\": \"" + INSTRUCTION + "\", \"truncate_prompt_tokens\": " + cut + "}"));
        assertEquals(2, ranked.list("results").size(), "every document answers");
        for (int r = 0; r < 2; r++) {
            int index = (int) at(ranked, "results", r).i64("index");
            assertEquals(index == 0 ? want1 : want2, at(ranked, "results", r).f64("relevance_score"), TOLERANCE,
                    "the pair of document " + index + " composes alike");
        }
    }

    @Test
    void biEncoderCapsCutTheVectorsNotOnlyTheUsage() throws IOException {
        String query = "some query text";
        String document = "other text";
        int[] content = bertService.encode(query, false, false);
        int[] cappedQuery = bertService.encode(
                bertService.tokenizer().decode(Arrays.copyOf(content, 1)), true, true);
        double want = bertService.biEncoderScore(cosine(bertService.embed(null, cappedQuery),
                bertService.embed(null, bertService.encode(document, true, true))));
        String body = "{\"queries\": \"" + query + "\", \"documents\": \"" + document + "\"}";
        assertNotEquals(firstScore(biPooling, body), want, 1e-7, "the cap changes the vector");
        assertEquals(want, firstScore(biPooling, body.replace("documents", "max_tokens_per_query\": 1,"
                + " \"documents")), TOLERANCE, "the capped query pools the truncated text");
        assertEquals(firstScore(biPooling, body.replace("documents", "max_tokens_per_query\": 1,"
                        + " \"documents")),
                firstRelevance(biPooling, "{\"query\": \"" + query + "\", \"documents\": [\"" + document + "\"],"
                        + " \"max_tokens_per_query\": 1}"), TOLERANCE, "/rerank applies the query cap too");
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return Math.sqrt(na) * Math.sqrt(nb) == 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    @Test
    void requestIdNamesTheResponse() {
        assertEquals("score-req-7", ok(barePooling.score(
                "{\"queries\": \"a\", \"documents\": \"b\", \"request_id\": \"req-7\"}")).str("id"));
        assertEquals("rerank-req-7", ok(barePooling.rerank(
                "{\"query\": \"a\", \"documents\": [\"b\"], \"request_id\": \"req-7\"}")).str("id"));
        assertEquals("embd-req-7", ok(biPooling.embeddings(
                "{\"input\": \"a\", \"request_id\": \"req-7\"}")).str("id"));
        assertEquals("classify-req-7", ok(barePooling.classify(
                "{\"input\": \"a\", \"request_id\": \"req-7\"}")).str("id"));
        assertEquals("pooling-req-7", ok(biPooling.pooling(
                "{\"input\": \"a\", \"request_id\": \"req-7\"}")).str("id"));
        assertEquals(400, statusOf(() -> barePooling.score(
                "{\"queries\": \"a\", \"documents\": \"b\", \"request_id\": 7}")));
    }

    @Test
    void sharedPoolingFieldsStayExplicit() {
        String pair = "{\"queries\": \"a\", \"documents\": \"b\", \"request_id\": \"r\", \"%s\": %s}";
        assertEquals(400, statusOf(() -> biPooling.score(String.format(pair, "priority", "5"))),
                "fcfs serves priority 0 only");
        assertEquals(200, statusOf(() -> biPooling.score(String.format(pair, "priority", "0"))));
        assertEquals(400, statusOf(() -> biPooling.score(String.format(pair, "cache_salt", "\"bad@salt\""))));
        assertEquals(200, statusOf(() -> biPooling.score(String.format(pair, "cache_salt", "\"ok\""))));
        assertEquals(400, statusOf(() -> biPooling.score(String.format(pair, "padding", "\"max_length\""))));
        assertEquals(200, statusOf(() -> biPooling.score(String.format(pair, "padding", "\"do_not_pad\""))));
        assertTrue(errorOf(() -> biPooling.score(String.format(pair, "padding", "\"middle\"")))
                .contains("padding"));
        assertEquals(400, statusOf(() -> biPooling.score(
                String.format(pair, "mm_processor_kwargs", "{\"x\": 1}"))));
        assertEquals(200, statusOf(() -> biPooling.score(
                String.format(pair, "mm_processor_kwargs", "{}"))));
        assertEquals(400, statusOf(() -> biPooling.rerank(
                "{\"query\": \"a\", \"documents\": [\"b\"], \"top_n\": -1}")));
    }

    @Test
    void normalizeIsRemovedAndRemovedTasksRefuse() {
        assertEquals(400, statusOf(() -> biPooling.embeddings("{\"input\": \"a\", \"normalize\": false}")));
        assertEquals(400, statusOf(() -> biPooling.pooling("{\"input\": \"a\", \"task\": \"embed\","
                + " \"normalize\": false}")));
        assertEquals(400, statusOf(() -> barePooling.classify("{\"input\": \"a\", \"normalize\": false}")));
        assertEquals("Parameter `normalize` was removed; use `use_activation` instead.",
                errorOf(() -> biPooling.embeddings("{\"input\": \"a\", \"normalize\": false}")));
        assertEquals(200, statusOf(() -> barePooling.score(
                "{\"queries\": \"a\", \"documents\": \"b\", \"normalize\": false}")),
                "the score protocol has no removed-parameter check");
        assertEquals(400, statusOf(() -> biPooling.pooling("{\"input\": \"a\", \"task\": \"score\"}")));
        assertEquals("`score` task was removed; use `classify` instead.",
                errorOf(() -> biPooling.pooling("{\"input\": \"a\", \"task\": \"score\"}")));
        assertEquals("`encode` task was removed; use `token_embed` or `token_classify` instead.",
                errorOf(() -> biPooling.pooling("{\"input\": \"a\", \"task\": \"encode\"}")));
        assertEquals(400, statusOf(() -> biPooling.embeddings("{\"input\": \"a\", \"task\": \"score\"}")));
        assertEquals(400, statusOf(() -> barePooling.classify("{\"input\": \"a\", \"task\": \"score\"}")));
    }
}
