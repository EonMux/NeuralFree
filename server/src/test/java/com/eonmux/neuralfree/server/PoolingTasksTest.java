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
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The vLLM pooling-task endpoints over the real service: {@code /score} through the synthetic rank classifier
 * (cross-encoder, agreeing with rerank) and the synthetic bert (bi-encoder cosine), {@code /classify}'s
 * probabilities and labels, {@code /pooling}'s tasks and encodings, and the refusals a generative model names.
 */
public class PoolingTasksTest {

    private static EngineService bert;
    private static EngineService rank;
    private static EngineService llama;
    private static PoolingResource pooled;
    private static PoolingResource classifier;
    private static PoolingResource generative;

    @BeforeAll
    static void open() throws IOException {
        Path bertModel = SyntheticModels.bert().writeTempFile();
        bert = new EngineService(bertModel.toString(), ServerTestSupport.config().build());
        Path rankModel = SyntheticModels.rank("neg", "pos").writeTempFile();
        rank = new EngineService(rankModel.toString(), ServerTestSupport.config().build());
        Path llamaModel = SyntheticModels.llama().writeTempFile();
        llama = new EngineService(llamaModel.toString(), ServerTestSupport.pagedConfig(64, 2));
        pooled = new PoolingResource(bert);
        classifier = new PoolingResource(rank);
        generative = new PoolingResource(llama);
    }

    @AfterAll
    static void close() {
        bert.close();
        rank.close();
        llama.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus());
        return view(response.getEntity());
    }

    private static double score(JsonView body, int index) {
        return at(body, "data", index).f64("score");
    }

    private static int listSize(Object raw) {
        return ((List<?>) raw).size();
    }

    @Test
    void scoreAnswersOnePairPerCrossEncoderForward() {
        JsonView body = ok(classifier.score("""
                {"queries": "q", "documents": ["d one", "d two"]}"""));
        assertEquals("list", body.str("object"));
        assertTrue(body.str("id").startsWith("score-"), body.str("id"));
        assertEquals(2, size(body, "data"));
        for (int i = 0; i < 2; i++) {
            assertEquals("score", at(body, "data", i).str("object"));
            assertEquals(i, at(body, "data", i).i64("index"));
            double s = score(body, i);
            assertTrue(!Double.isNaN(s) && Math.abs(s) <= 1.0, "a tanh score of the synthetic head: " + s);
        }
        assertTrue(body.obj("usage").i64("total_tokens") > 0, "usage counts the pairs");
    }

    @Test
    void crossEncoderScoreAgreesWithRerank() {
        double reranked = at(ok(classifier.rerank(
                "{\"query\": \"question\", \"documents\": [\"answer\"], \"return_documents\": true}")),
                "results", 0).f64("relevance_score");
        double scored = score(ok(classifier.score(
                "{\"queries\": \"question\", \"documents\": [\"answer\"]}")), 0);
        assertEquals(reranked, scored, 1e-9, "the same pair forward answers both endpoints");
    }

    @Test
    void scoreServesEverySideNaming() {
        for (String body : new String[]{
                "{\"queries\": [\"a\"], \"items\": [\"b\"]}",
                "{\"data_1\": \"a\", \"data_2\": [\"b\", \"c\"]}",
                "{\"text_1\": \"a\", \"text_2\": \"b\"}"}) {
            assertEquals(200, statusOf(() -> classifier.score(body)), body);
        }
        assertEquals(2, size(ok(classifier.score(
                "{\"queries\": \"q\", \"documents\": [\"a\", \"b\"]}")), "data"),
                "one query is scored against every document");
        assertEquals(2, size(ok(classifier.score(
                "{\"queries\": [\"q1\", \"q2\"], \"documents\": [\"a\", \"b\"]}")), "data"),
                "two queries pair with two documents");
        for (String bad : new String[]{
                "{\"queries\": \"q\"}",
                "{\"documents\": [\"d\"]}",
                "{\"queries\": [\"a\", \"b\"], \"documents\": [\"c\"]}",
                "{\"queries\": \"q\", \"documents\": \"d\", \"max_tokens_per_query\": -1}",
                "{\"queries\": \"q\", \"documents\": \"d\", \"use_activation\": \"yes\"}",
                "{\"queries\": \"q\", \"documents\": [4]}"}) {
            assertEquals(400, statusOf(() -> classifier.score(bad)), bad);
        }
    }

    @Test
    void useActivationFalseAnswersTheRawCrossEncoderScore() throws IOException {
        double applied = score(ok(classifier.score("{\"queries\": \"q\", \"documents\": \"d\"}")), 0);
        assertEquals(applied, score(ok(classifier.score(
                "{\"queries\": \"q\", \"documents\": \"d\", \"use_activation\": true}")), 0), 0.0,
                "true answers the applied score");
        int[] input = rank.pooling().rerankInput(rank.encode("q", false, false), rank.encode("d", false, false));
        float raw = rank.pooling().classifyOutput(null, input, false).scores()[0];
        assertEquals(raw, score(ok(classifier.score(
                "{\"queries\": \"q\", \"documents\": \"d\", \"use_activation\": false}")), 0), 1e-9,
                "false answers the raw head score");
        double relevance = at(ok(classifier.rerank(
                "{\"query\": \"q\", \"documents\": [\"d\"], \"use_activation\": false}")),
                "results", 0).f64("relevance_score");
        assertEquals(raw, relevance, 1e-9, "/rerank skips the activation the same way");
    }

    @Test
    void useActivationFalseSkipsTheActivationOfEveryPairInOneBatch() throws IOException {
        int[] first = rank.pooling().rerankInput(rank.encode("q", false, false), rank.encode("d one", false, false));
        int[] second = rank.pooling().rerankInput(rank.encode("q", false, false), rank.encode("d two", false, false));
        double[] raw = {rank.pooling().classifyOutput(null, first, false).scores()[0],
                rank.pooling().classifyOutput(null, second, false).scores()[0]};
        JsonView scored = ok(classifier.score(
                "{\"queries\": \"q\", \"documents\": [\"d one\", \"d two\"], \"use_activation\": false}"));
        assertEquals(raw[0], score(scored, 0), 1e-9, "pair 0 answers its raw head score");
        assertEquals(raw[1], score(scored, 1), 1e-9, "pair 1 answers its raw head score");
        JsonView reranked = ok(classifier.rerank(
                "{\"query\": \"q\", \"documents\": [\"d one\", \"d two\"], \"use_activation\": false}"));
        double[] answers = {at(reranked, "results", 0).f64("relevance_score"),
                at(reranked, "results", 1).f64("relevance_score")};
        Arrays.sort(raw);
        Arrays.sort(answers);
        assertArrayEquals(raw, answers, 1e-9, "the batched rerank answers every pair's raw score");
    }

    @Test
    void scoreTruncatesEachSide() {
        JsonView full = ok(classifier.score("{\"queries\": \"long query here\", \"documents\": [\"doc\"]}"));
        JsonView cut = ok(classifier.score(
                "{\"queries\": \"long query here\", \"documents\": [\"doc\"], \"max_tokens_per_query\": 1}"));
        assertTrue(cut.obj("usage").i64("total_tokens") < full.obj("usage").i64("total_tokens"),
                "the query truncates to one token");
    }

    @Test
    void biEncoderScoreIsTheCosineOfTheEmbeddings() {
        JsonView embeddings = ok(pooled.embeddings("{\"input\": [\"same text\", \"other text\"]}"));
        float[] a = vector(at(embeddings, "data", 0).raw("embedding"));
        float[] b = vector(at(embeddings, "data", 1).raw("embedding"));
        double expected = cosine(a, b);
        double identical = score(ok(pooled.score(
                "{\"queries\": \"same text\", \"documents\": [\"same text\"]}")), 0);
        double different = score(ok(pooled.score(
                "{\"text_1\": \"same text\", \"text_2\": \"other text\"}")), 0);
        assertEquals(1.0, identical, 1e-4, "an identical pair scores one");
        assertEquals(expected, different, 1e-6, "the score is the cosine of the pooled vectors");
    }

    @Test
    void scoreRefusesByModelAndByBody() {
        assertTrue(errorOf(() -> generative.score("{\"queries\": \"q\", \"documents\": [\"d\"]}"))
                .contains("scoring"), "the generative refusal names what scoring needs");
        assertTrue(errorOf(() -> pooled.classify("{\"input\": \"hello\"}")).contains("RANK"),
                "the bert refusal names the classifier pooling");
        assertEquals(400, statusOf(() -> generative.classify("{\"input\": \"hello\"}")));
        assertEquals(400, statusOf(() -> generative.pooling("{\"input\": \"hello\"}")));
    }

    @Test
    void rerankAnswersItsV2Alias() {
        JsonView body = ok(classifier.rerankV2Alias("{\"query\": \"q\", \"documents\": [\"a\", \"b\"]}"));
        assertEquals(2, size(body, "results"));
        assertEquals(400, statusOf(() -> classifier.rerankV2Alias("{\"query\": \"q\"}")));
    }

    @Test
    void classifyAnswersProbabilitiesWithLabels() {
        JsonView body = ok(classifier.classify("{\"input\": [\"good text\", \"bad text\"]}"));
        assertEquals("list", body.str("object"));
        assertTrue(body.str("id").startsWith("classify-"), body.str("id"));
        assertEquals(2, size(body, "data"));
        JsonView first = at(body, "data", 0);
        assertEquals(0L, first.i64("index"));
        List<?> probs = (List<?>) first.raw("probs");
        assertEquals(2, probs.size(), "one probability per class");
        double sum = 0;
        for (Object p : probs) {
            sum += ((Number) p).doubleValue();
        }
        assertEquals(1.0, sum, 1e-5, "the probabilities sum to one");
        assertEquals(2L, first.i64("num_classes"));
        String label = first.str("label");
        assertTrue("neg".equals(label) || "pos".equals(label), label);
        assertTrue(body.obj("usage").i64("total_tokens") > 0);
    }

    @Test
    void classifyTakesTokenIdsAndChatMessages() {
        JsonView ids = ok(classifier.classify("{\"input\": [7, 8, 9]}"));
        assertEquals(2, listSize(at(ids, "data", 0).raw("probs")));
        JsonView chat = ok(classifier.classify(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}"));
        assertEquals(2, listSize(at(chat, "data", 0).raw("probs")));
        JsonView batched = ok(classifier.classify(
                "{\"messages\": [[{\"role\": \"user\", \"content\": \"a\"}],"
                        + "[{\"role\": \"user\", \"content\": \"b\"}]]}"));
        assertEquals(2, size(batched, "data"));
        JsonView inputMessages = ok(classifier.classify(
                "{\"input\": [{\"role\": \"user\", \"content\": \"hello\"}]}"));
        assertEquals(at(chat, "data", 0).raw("probs"), at(inputMessages, "data", 0).raw("probs"),
                "the messages form and the message input share the forward");
        assertEquals(400, statusOf(() -> classifier.classify("{\"messages\": [{\"content\": \"no role\"}]}")));
    }

    @Test
    void useActivationFalseAnswersTheRawClassScores() throws IOException {
        int[] ids = rank.encode("x", true, true);
        float[] raw = rank.pooling().classifyOutput(null, ids, false).scores();
        List<?> probs = (List<?>) at(ok(classifier.classify(
                "{\"input\": \"x\", \"use_activation\": false}")), "data", 0).raw("probs");
        assertEquals(raw.length, probs.size(), "one raw score per class");
        double rawSum = 0;
        for (int j = 0; j < raw.length; j++) {
            assertEquals((double) raw[j], ((Number) probs.get(j)).doubleValue(), 1e-6, "score " + j);
            rawSum += raw[j];
        }
        assertTrue(Math.abs(rawSum - 1.0) > 1e-5, "the raw scores are not probabilities: " + rawSum);
        JsonView applied = ok(classifier.classify("{\"input\": \"x\"}"));
        assertEquals(at(applied, "data", 0).raw("probs"), at(ok(classifier.classify(
                "{\"input\": \"x\", \"use_activation\": true}")), "data", 0).raw("probs"),
                "true answers the applied probabilities");
        double sum = 0;
        for (Object p : (List<?>) at(applied, "data", 0).raw("probs")) {
            sum += ((Number) p).doubleValue();
        }
        assertEquals(1.0, sum, 1e-5, "the default applies the activation: the probabilities sum to one");
    }

    @Test
    void poolingAnswersTheModelTaskByDefault() {
        JsonView pooledBody = ok(pooled.pooling("{\"input\": \"hello world\"}"));
        assertTrue(pooledBody.str("id").startsWith("pooling-"), pooledBody.str("id"));
        assertEquals("pooling", at(pooledBody, "data", 0).str("object"));
        JsonView embeddings = ok(pooled.embeddings("{\"input\": \"hello world\"}"));
        assertEquals(at(embeddings, "data", 0).raw("embedding"), at(pooledBody, "data", 0).raw("data"),
                "the embed task answers the embeddings vector");
        JsonView classifyBody = ok(classifier.pooling("{\"input\": \"hello world\"}"));
        JsonView classified = ok(classifier.classify("{\"input\": \"hello world\"}"));
        assertEquals(at(classified, "data", 0).raw("probs"), at(classifyBody, "data", 0).raw("data"),
                "the rank model's task is classify");
    }

    @Test
    void poolingTasksAndEncodingsValidate() {
        assertEquals(at(ok(pooled.pooling("{\"input\": \"x\"}")), "data", 0).raw("data"),
                at(ok(pooled.pooling("{\"input\": \"x\", \"task\": \"embed\"}")), "data", 0).raw("data"),
                "embed is the embedding model's own task, also when named");
        for (String bad : new String[]{
                "{\"input\": \"x\", \"task\": \"token_embed\"}",
                "{\"input\": \"x\", \"task\": \"plugin\"}",
                "{\"input\": \"x\", \"dimensions\": 4}"}) {
            assertEquals(400, statusOf(() -> pooled.pooling(bad)), bad);
        }
        assertEquals(400, statusOf(() -> classifier.pooling("{\"input\": \"x\", \"task\": \"embed\"}")));
        assertEquals(400, statusOf(() -> classifier.pooling("{\"input\": \"x\", \"task\": \"token_embed\"}")));
        JsonView base64 = ok(pooled.pooling("{\"input\": \"hello\", \"encoding_format\": \"base64\"}"));
        assertTrue(at(base64, "data", 0).raw("data") instanceof String, "base64 carries a string");
        JsonView floats = ok(pooled.pooling("{\"input\": \"hello\"}"));
        assertEquals(at(floats, "data", 0).raw("data"), decodeBase64(at(base64, "data", 0).str("data")),
                "both encodings carry the same vector");
        JsonView classifyBase64 = ok(classifier.pooling(
                "{\"input\": \"hello\", \"task\": \"classify\", \"encoding_format\": \"base64\"}"));
        assertTrue(at(classifyBase64, "data", 0).raw("data") instanceof String);
    }

    @Test
    void poolingTokenEmbedNeedsAPerTokenModel() {
        String message = errorOf(() -> pooled.pooling("{\"input\": \"x\", \"task\": \"token_embed\"}"));
        assertEquals(400, statusOf(() -> pooled.pooling("{\"input\": \"x\", \"task\": \"token_embed\"}")));
        assertTrue(message.contains("per-token"), message);
    }

    @Test
    void generativeScoringAnswersLabelProbabilities() {
        int first = llama.vocabSize() / 2;
        int second = llama.vocabSize() / 2 + 1;
        Response response = generative.generativeScoring("{\"query\": \"rate this\","
                + " \"items\": [\"item a\", \"item b\"], \"label_token_ids\": [" + first + ", " + second + "]}");
        JsonView body = ok(response);
        assertTrue(body.str("id").startsWith("generative-scoring-"), body.str("id"));
        assertEquals(2, size(body, "data"));
        for (int i = 0; i < 2; i++) {
            double s = score(body, i);
            assertTrue(s >= 0.0 && s <= 1.0, "a probability: " + s);
            assertEquals("score", at(body, "data", i).str("object"));
        }
        long prompt = body.obj("usage").i64("prompt_tokens");
        assertTrue(prompt > 0);
        assertEquals(2L, body.obj("usage").i64("completion_tokens"), "two items complete one token each");
        assertEquals(prompt + 2, body.obj("usage").i64("total_tokens"));
    }

    @Test
    void generativeScoringScoresMatchTheNextTokenDistribution() throws Exception {
        int first = 5;
        int second = 300;
        String request = "{\"query\": \"score\", \"items\": [\"this one\"],"
                + " \"label_token_ids\": [" + first + ", " + second + "]}";
        double scored = score(ok(generative.generativeScoring(request)), 0);
        int[] query = llama.encode("score", true, true);
        int[] item = llama.encode("this one", false, true);
        int[] prompt = new int[query.length + item.length];
        System.arraycopy(query, 0, prompt, 0, query.length);
        System.arraycopy(item, 0, prompt, query.length, item.length);
        float[] logits = llama.pooling().nextLogits(null, prompt);
        double max = Double.NEGATIVE_INFINITY;
        for (int id : new int[]{first, second}) {
            max = Math.max(max, logits[id]);
        }
        double sum = Math.exp(logits[first] - max) + Math.exp(logits[second] - max);
        assertEquals(Math.exp(logits[first] - max) / sum, scored, 1e-9,
                "the score is the softmax over the label logits");
        String raw = "{\"query\": \"score\", \"items\": [\"this one\"], \"apply_softmax\": false,"
                + " \"label_token_ids\": [" + first + ", " + second + "]}";
        double rawScore = score(ok(generative.generativeScoring(raw)), 0);
        assertTrue(rawScore < scored, "the full-vocabulary probability is the smaller one here");
    }

    @Test
    void generativeScoringTakesIdSidesAndItemFirst() {
        int label = 9;
        JsonView ids = ok(generative.generativeScoring("{\"query\": [5, 6], \"items\": [[7, 8]],"
                + " \"label_token_ids\": [" + label + "]}"));
        assertEquals(1, size(ids, "data"));
        assertEquals(1.0, score(ids, 0), 1e-9, "one label softmaxes to one");
        JsonView flipped = ok(generative.generativeScoring("{\"query\": \"q\", \"items\": [\"i\"],"
                + " \"item_first\": true, \"label_token_ids\": [" + label + "]}"));
        assertEquals(1.0, score(flipped, 0), 1e-9);
        for (String bad : new String[]{
                "{\"query\": \"q\", \"items\": [], \"label_token_ids\": [1]}",
                "{\"query\": \"q\", \"items\": [\"i\"], \"label_token_ids\": []}",
                "{\"query\": \"q\", \"items\": [\"i\"], \"label_token_ids\": [-1]}",
                "{\"query\": \"q\", \"items\": [\"i\"], \"label_token_ids\": [1000000]}",
                "{\"query\": \"q\", \"label_token_ids\": [1]}",
                "{\"query\": {}, \"items\": [\"i\"], \"label_token_ids\": [1]}",
                "{\"items\": [\"i\"], \"label_token_ids\": [1]}"}) {
            assertEquals(400, statusOf(() -> generative.generativeScoring(bad)), bad);
        }
        String refusal = errorOf(() -> pooled.generativeScoring(
                "{\"query\": \"q\", \"items\": [\"i\"], \"label_token_ids\": [1]}"));
        assertTrue(refusal.contains("generative"), refusal);
    }

    private static float[] vector(Object embedding) {
        List<?> values = (List<?>) embedding;
        float[] out = new float[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) values.get(i)).floatValue();
        }
        return out;
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0;
        double normA = 0;
        double normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            normA += (double) a[i] * a[i];
            normB += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static List<?> decodeBase64(String encoded) {
        java.nio.ByteBuffer bytes = java.nio.ByteBuffer
                .wrap(java.util.Base64.getDecoder().decode(encoded))
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        java.util.List<Object> out = new java.util.ArrayList<>();
        while (bytes.hasRemaining()) {
            out.add((double) bytes.getFloat());
        }
        return out;
    }
}
