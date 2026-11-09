package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The decision endpoints over the real service: the synthetic vllm-decision2 model answers
 * {@code /v1/systemone} (choice, score with the model's offsets, noul, per-question prompts, marker
 * sanitization and the reference's validation messages) and {@code /v1/decisions} (predicate, choice and score
 * answers echoing the request's values, vLLM's usage block, the joint refusal and the question caps); a bert
 * answers the not-a-decision-model 501, a broken template the unexpected-layout 500.
 */
public class DecisionResourceTest {

    private static EngineService decision;
    private static EngineService notADecision;
    private static EngineService brokenLayout;
    private static EngineService emptyPiece;
    private static DecisionResource resource;
    private static DecisionResource bertResource;
    private static DecisionResource broken;
    private static DecisionResource hollow;
    /** The served name responses carry: no served names are configured, so it is the model path. */
    private static String decisionName;
    private static String bertName;

    @BeforeAll
    static void open() throws IOException {
        EngineConfig config = ServerTestSupport.config().build();
        Path decisionPath = decisionModel(SyntheticModels.VLLM_DECISION2_TEMPLATE);
        decision = new EngineService(decisionPath.toString(), config);
        decisionName = decisionPath.toString();
        Path bertPath = SyntheticModels.bert().writeTempFile();
        notADecision = new EngineService(bertPath.toString(), config);
        bertName = bertPath.toString();
        brokenLayout = new EngineService(decisionModel(
                "{% for q in questions %}{{ sep }}{{ mark_question }}\n{{ q.instructions | tojson }}"
                        + "{% for o in q.options %}{{ sep }}\nplain{% endfor %}{% endfor %}").toString(), config);
        emptyPiece = new EngineService(decisionModel(
                "{% for q in questions %}{{ sep }}{{ mark_question }}"
                        + "{% for o in q.options %}{{ sep }}{{ mark_option }}{% endfor %}{% endfor %}").toString(),
                config);
        resource = new DecisionResource(decision);
        bertResource = new DecisionResource(notADecision);
        broken = new DecisionResource(brokenLayout);
        hollow = new DecisionResource(emptyPiece);
    }

    /** The decision fixture at a context the prompts fit into. */
    private static Path decisionModel(String template) throws IOException {
        return SyntheticModels.vllmDecision2(
                SyntheticModels.Config.qwen2().toBuilder().nCtxTrain(1024).build(), template).writeTempFile();
    }

    @AfterAll
    static void close() {
        decision.close();
        notADecision.close();
        brokenLayout.close();
        emptyPiece.close();
    }

    // ------------------------------------------------------------------ /v1/systemone

    @Test
    void systemoneAnswersAChoiceQuestionWithKeyedProbabilitiesAndEntropyConfidence() {
        Response_ response = systemone("""
                {"state": "the sky is blue",
                 "questions": {"color": {"type": "choice", "instructions": "What color is the sky?",
                     "criteria": {"red": "warm", "green": "plant", "blue": "cool"}}}}""");
        assertEquals(decisionName, response.body.str("model"));
        JsonView answer = response.body.obj("answers").obj("color");
        assertEquals("choice", answer.str("type"));
        assertTrue(List.of("red", "green", "blue").contains(answer.str("choice")));
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(3, probabilities.keys().size());
        double sum = 0;
        for (String key : List.of("red", "green", "blue")) {
            sum += probabilities.f64(key);
        }
        assertEquals(1.0, sum, 1e-6);
        double entropy = 0;
        for (String key : List.of("red", "green", "blue")) {
            double p = probabilities.f64(key);
            if (p > 0) {
                entropy -= p * Math.log(p);
            }
        }
        assertEquals(Math.max(0.0, Math.min(1.0, 1.0 - entropy / Math.log(3))), answer.f64("confidence"), 1e-9);
        assertTrue(response.body.obj("usage").i64("input_tokens") > 0);
        assertEquals(0L, response.body.obj("usage").i64("output_tokens"));
    }

    @Test
    void systemoneScoresApplyTheModelOffsetsAndTemperature() {
        String body = """
                {"state": {"text": "rate it"},
                 "questions": {"rate": {"type": "score", "instructions": "Rate the answer.",
                     "criteria": ["bad", "meh", "ok", "good"]}}}""";
        JsonView answer = systemone(body).body.obj("answers").obj("rate");
        assertEquals("score", answer.str("type"));
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(4, probabilities.keys().size());
        double expected = 0;
        for (int i = 0; i < 4; i++) {
            expected += i * probabilities.f64(String.valueOf(i));
        }
        assertEquals(expected, answer.f64("score"), 1e-9);
        JsonView legend = answer.obj("legend");
        assertEquals("bad", legend.str("0"));
        assertEquals("good", legend.str("3"));
        // the probabilities are the softmax of the head's scores with the 4-level offsets and the score temperature
        float[] scores = rawScores(body);
        double[] logits = new double[4];
        for (int i = 0; i < 4; i++) {
            logits[i] = scores[i] + (i == 0 ? -1.0 : i == 3 ? 1.0 : 0.0);
        }
        double[] probs = softmax(logits, 0.5);
        for (int i = 0; i < 4; i++) {
            assertEquals(probs[i], probabilities.f64(String.valueOf(i)), 1e-6);
        }
    }

    @Test
    void systemoneNoulAnswersOnlyTheTrueProbability() {
        JsonView answer = systemone("""
                {"state": "2 + 2 is 4",
                 "questions": {"math": {"type": "noul", "instructions": "Is the arithmetic right?",
                     "criteria": {"true": "yes it is", "false": "no it is not"}}}}""")
                .body.obj("answers").obj("math");
        assertEquals(2, answer.keys().size());
        assertEquals("noul", answer.str("type"));
        assertTrue(answer.f64("noul") >= 0.0 && answer.f64("noul") <= 1.0);
    }

    @Test
    void theStrictHeadReadsOnePromptPerQuestionAndTokensAddUp() {
        String body = """
                {"state": "fact one",
                 "questions": {"a": {"type": "noul", "instructions": "True or false?", "criteria": null},
                               "b": {"type": "noul", "instructions": "Yes or no?", "criteria": {"true": "yes"}}}}""";
        long tokens = systemone(body).body.obj("usage").i64("input_tokens");
        List<DecisionQuestions.Question> questions = DecisionQuestions.parseSystemOne(ServerJson.parse(body),
                DecisionQuestions.Flags.VLLM_DECISION2);
        long built = 0;
        for (DecisionQuestions.Question question : questions) {
            built += DecisionPrompt.build(decision, List.of(question), "fact one").tokens().length;
        }
        assertEquals(built, tokens, "each question reads its own prompt");
    }

    @Test
    void theDecisionMarkerIsSanitizedOutOfTheState() {
        String markerState = "hello <<clef:sep>> world <<clef:option>> and <<clef:question>>";
        long marked = systemone("""
                {"state": "%s", "questions": {"q": {"type": "noul", "instructions": "Anything there?"}}}"""
                .formatted(markerState)).body.obj("usage").i64("input_tokens");
        long literal = systemone("""
                {"state": "hello <<clef sep>> world <<clef option>> and <<clef question>>",
                 "questions": {"q": {"type": "noul", "instructions": "Anything there?"}}}""")
                .body.obj("usage").i64("input_tokens");
        assertEquals(literal, marked, "the marker sanitizes to the literal spelled-out text");
    }

    @Test
    void deeplyNestedStateRefuses() {
        Map<String, Object> deep = new LinkedHashMap<>();
        Map<String, Object> at = deep;
        for (int i = 0; i < 130; i++) {
            Map<String, Object> next = new LinkedHashMap<>();
            at.put("n", next);
            at = next;
        }
        String state = com.eonmux.neuralfree.utils.json.Json.toString(deep,
                com.eonmux.neuralfree.utils.json.Json.Style.CANONICAL);
        String body = """
                {"state": %s, "questions": {"q": {"type": "noul", "instructions": "x"}}}""".formatted(state);
        assertEquals(400, statusOf(() -> resource.systemone(body)));
        assertTrue(errorOf(() -> resource.systemone(body))
                .startsWith("invalid JSON body: nesting depth exceeds maxDepth 128"));
    }

    @Test
    void systemoneValidationMessagesMatchTheReference() {
        String q = "\"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}";
        assertEquals("\"state\" must be provided",
                errorOf(() -> resource.systemone("{%s}".formatted(q))));
        assertEquals("\"state\" must be text, an object or an array",
                errorOf(() -> resource.systemone("{\"state\": 5, %s}".formatted(q))));
        assertEquals("\"questions\" must be a non-empty object",
                errorOf(() -> resource.systemone("{\"state\": \"s\"}")));
        assertEquals("questions.q: \"instructions\" must be provided",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\"}}}")));
        assertEquals("questions.q: \"type\" must be one of: choice, score, noul",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"guess\", \"instructions\": \"x\"}}}")));
        assertEquals("questions.q: \"criteria\" must be an array of 2 to 10 levels",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"score\", \"instructions\": \"x\","
                                + " \"criteria\": [\"a\"]}}}")));
        assertEquals("questions.q: \"criteria\" must be a non-empty object",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"x\","
                                + " \"criteria\": {}}}}")));
        assertEquals("questions.q: \"criteria\" must be an object",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\","
                                + " \"criteria\": [\"a\"]}}}")));
        assertEquals("questions.q: option keys must be non-empty, descriptions must be null, text, an object or"
                        + " an array",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"x\","
                                + " \"criteria\": {\"\": 5}}}}")));
        assertEquals("questions.q: \"criteria\" must have at least 2 options",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"x\","
                                + " \"criteria\": {\"only\": \"one\"}}}}")));
    }

    @Test
    void tooManyOptionsRefuseNamingTheCap() {
        StringBuilder criteria = new StringBuilder("{");
        for (int i = 0; i < 256; i++) {
            criteria.append(i > 0 ? ", " : "").append("\"k").append(i).append("\": \"d\"").append("");
        }
        criteria.append("}");
        assertEquals("questions.q: too many options (256), this model supports at most 255",
                errorOf(() -> resource.systemone(
                        "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"x\","
                                + " \"criteria\": " + criteria + "}}}")));
    }

    @Test
    void aModelWithoutADecisionHeadRefusesWithTheReferenceMessage() {
        String body = "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}";
        assertEquals(501, statusOf(() -> bertResource.systemone(body)));
        assertEquals("This model is not a decision model", errorOf(() -> bertResource.systemone(body)));
        assertEquals(501, statusOf(() -> bertResource.decisions(body)));
    }

    @Test
    void systemoneMediaRefusalsFollowTheReference() {
        String q = "\"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}";
        assertEquals("\"videos\" is not supported",
                errorOf(() -> resource.systemone("{\"state\": \"s\", \"videos\": [\"data:image/png;base64,AA\"],"
                        + q + "}")));
        assertEquals("\"files\" must be an array",
                errorOf(() -> resource.systemone("{\"state\": \"s\", \"files\": \"data:image/png;base64,AA\","
                        + q + "}")));
        assertEquals("media must be data URLs (data:image/..., data:audio/... or data:video/...;base64,...)",
                errorOf(() -> resource.systemone("{\"state\": \"s\", \"files\": [\"http://x/y.png\"],"
                        + q + "}")));
        String refusal = "This server does not support image input for decisions. For a model that supports it,"
                + " start it with `--mmproj`";
        assertEquals(501, statusOf(() -> resource.systemone(
                "{\"state\": \"s\", \"files\": [\"data:image/png;base64,AA\"]," + q + "}")));
        assertEquals(refusal, errorOf(() -> resource.systemone(
                "{\"state\": \"s\", \"images\": [\"data:image/png;base64,AA\"]," + q + "}")));
        assertEquals(refusal, errorOf(() -> resource.systemone(
                "{\"state\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\":"
                        + " {\"url\": \"data:image/png;base64,AA\"}}]}]," + q + "}")));
        // an empty files array serves
        assertEquals(200, statusOf(() -> resource.systemone(
                "{\"state\": \"s\", \"files\": [], " + q + "}")));
    }

    @Test
    void aMismatchedTemplateRefusesAsAServerError() {
        String body = "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}";
        assertEquals(500, statusOf(() -> broken.systemone(body)));
        assertEquals("unexpected layout of the decision prompt", errorOf(() -> broken.systemone(body)));
    }

    @Test
    void anEmptyMarkedPieceRefusesAsABadRequest() {
        String body = "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}";
        assertEquals(400, statusOf(() -> hollow.systemone(body)));
        assertEquals("the instructions and the options of a question must not be empty",
                errorOf(() -> hollow.systemone(body)));
    }

    // ------------------------------------------------------------------ /v1/decisions

    @Test
    void decisionsAnswerPredicateChoiceAndScore() {
        String body = """
                {"input": "the sky is blue",
                 "questions": [
                     {"type": "predicate", "instructions": "Is the sky blue?"},
                     {"type": "choice", "name": "color", "instructions": "What color is the sky?",
                      "choices": [{"value": true, "description": "yes"}, {"value": "red"}, {"value": "blue"}]},
                     {"type": "score", "name": "rate", "instructions": "Rate the sky.",
                      "levels": [{"label": "bad"}, {"label": "ok"}, {"label": "great"}]}]}""";
        JsonView out = view(resource.decisions(body).getEntity());
        assertEquals(decisionName, out.str("model"));
        assertEquals(3, size(out, "answers"));

        JsonView predicate = at(out, "answers", 0);
        assertEquals("predicate", predicate.str("type"));
        assertNull(predicate.raw("name"));
        assertTrue(predicate.f64("probability") >= 0.0 && predicate.f64("probability") <= 1.0);
        assertNull(predicate.raw("confidence"));

        JsonView choice = at(out, "answers", 1);
        assertEquals("choice", choice.str("type"));
        assertEquals("color", choice.str("name"));
        assertTrue(List.of("true", "\"red\"", "\"blue\"").contains(
                com.eonmux.neuralfree.utils.json.Json.toString(choice.raw("choice"),
                        com.eonmux.neuralfree.utils.json.Json.Style.CANONICAL)));
        assertEquals(3, size(choice, "probabilities"));
        List<Object> values = ((List<?>) choice.raw("probabilities")).stream()
                .map(row -> JsonView.of(row, "probability").raw("value")).toList();
        assertEquals(List.of(Boolean.TRUE, "red", "blue"), values, "the choice values echo in request order");
        JsonView firstProbability = at(choice, "probabilities", 0);
        assertEquals(Boolean.TRUE, firstProbability.raw("value"));
        double top = 0;
        double sum = 0;
        for (int i = 0; i < 3; i++) {
            double p = at(choice, "probabilities", i).f64("probability");
            top = Math.max(top, p);
            sum += p;
        }
        assertEquals(1.0, sum, 1e-6);
        assertEquals(top, choice.f64("confidence"), 1e-12);

        JsonView score = at(out, "answers", 2);
        assertEquals("score", score.str("type"));
        assertEquals("rate", score.str("name"));
        double expected = 0;
        double probabilitySum = 0;
        for (int i = 0; i < 3; i++) {
            JsonView row = at(score, "probabilities", i);
            assertEquals((long) i, row.i64("value"));
            assertEquals(List.of("bad", "ok", "great").get(i), row.str("label"));
            expected += i * row.f64("probability");
            probabilitySum += row.f64("probability");
        }
        assertEquals(expected, score.f64("score"), 1e-9);
        assertEquals(1.0, probabilitySum, 1e-6);

        JsonView usage = out.obj("usage");
        assertTrue(usage.i64("input_tokens") > 0);
        assertEquals(usage.i64("input_tokens"), usage.i64("total_tokens"));
        assertEquals(0L, usage.i64("output_tokens"));
        assertEquals(0L, usage.obj("input_tokens_details").i64("cached_tokens"));
        assertEquals(0L, usage.obj("input_tokens_details").i64("cache_write_tokens"));
        assertEquals(0L, usage.obj("output_tokens_details").i64("reasoning_tokens"));
    }

    @Test
    void decisionsMessagesInputReadsLikeTheJoinedString() {
        String plain = """
                {"input": "one state\\nsecond part",
                 "questions": [{"type": "predicate", "instructions": "True?"}]}""";
        String messages = """
                {"input": [
                     {"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": "one state"},
                                  {"type": "input_text", "text": "second part"}]}],
                 "questions": [{"type": "predicate", "instructions": "True?"}]}""";
        long asString = view(resource.decisions(plain).getEntity()).obj("usage").i64("input_tokens");
        long asMessages = view(resource.decisions(messages).getEntity()).obj("usage").i64("input_tokens");
        assertEquals(asString, asMessages, "the parts join with a newline, the messages with a blank line");
    }

    @Test
    void decisionsRefuseJointByNamingIt() {
        String body = """
                {"input": "s", "joint": true,
                 "questions": [{"type": "predicate", "instructions": "True?"},
                               {"type": "predicate", "instructions": "False?"}]}""";
        assertEquals(400, statusOf(() -> resource.decisions(body)));
        assertEquals("joint: this model reads each question on its own; joint reads need DiffusionGemma",
                errorOf(() -> resource.decisions(body)));
    }

    @Test
    void decisionsValidationMessagesMatchTheReference() {
        String q = "\"questions\": [{\"type\": \"predicate\", \"instructions\": \"x\"}]";
        assertEquals("questions: at most 64 for this model",
                errorOf(() -> resource.decisions("{\"input\": \"s\", \"questions\": " + manyQuestions(65) + "}")));
        assertEquals("\"questions\" must be a non-empty array",
                errorOf(() -> resource.decisions("{\"input\": \"s\", \"questions\": []}")));
        assertEquals("questions.0: \"type\" must be one of: predicate, choice, score",
                errorOf(() -> resource.decisions(
                        "{\"input\": \"s\", \"questions\": [{\"type\": \"guess\", \"instructions\": \"x\"}]}")));
        assertEquals("questions.0: unknown field \"criteria\"",
                errorOf(() -> resource.decisions(
                        "{\"input\": \"s\", \"questions\": [{\"type\": \"predicate\", \"instructions\": \"x\","
                                + " \"criteria\": {}}]}")));
        assertEquals("questions.0: \"instructions\" must be provided as text",
                errorOf(() -> resource.decisions(
                        "{\"input\": \"s\", \"questions\": [{\"type\": \"predicate\"}]}")));
        assertEquals("questions.0: \"name\" must be a string when provided",
                errorOf(() -> resource.decisions(
                        "{\"input\": \"s\", \"questions\": [{\"type\": \"predicate\", \"instructions\": \"x\","
                                + " \"name\": 5}]}")));
        assertEquals("questions.0: \"choices\" must be an array of 2 to 255 choices",
                errorOf(() -> resource.decisions(
                        "{\"input\": \"s\", \"questions\": [{\"type\": \"choice\", \"instructions\": \"x\","
                                + " \"choices\": [{\"value\": \"a\"}]}]}")));
        assertEquals("questions.0: \"choices\" must be an array of 2 to 255 choices",
                errorOf(() -> resource.decisions("{\"input\": \"s\", \"questions\": [{\"type\": \"choice\","
                        + " \"instructions\": \"x\", \"choices\": " + manyChoices(256) + "}]}")));
        assertEquals("This model supports at most 26 choices per question",
                errorOf(() -> resource.decisions("{\"input\": \"s\", \"questions\": [{\"type\": \"choice\","
                        + " \"instructions\": \"x\", \"choices\": " + manyChoices(27) + "}]}")));
        assertEquals("questions.0: choice values must be distinct",
                errorOf(() -> resource.decisions(
                        "{\"input\": \"s\", \"questions\": [{\"type\": \"choice\", \"instructions\": \"x\","
                                + " \"choices\": [{\"value\": \"a\"}, {\"value\": \"a\"}]}]}")));
        assertEquals("questions.0: \"levels\" must be an array of 2 to 10 levels",
                errorOf(() -> resource.decisions(
                        "{\"input\": \"s\", \"questions\": [{\"type\": \"score\", \"instructions\": \"x\","
                                + " \"levels\": [{\"label\": \"a\"}]}]}")));
        assertEquals("\"input\" must be text or an array of messages",
                errorOf(() -> resource.decisions("{" + q + "}")));
        assertEquals("input: message \"role\" must be \"user\"",
                errorOf(() -> resource.decisions(
                        "{\"input\": [{\"role\": \"system\", \"content\": \"s\"}]," + q + "}")));
        assertEquals("Decisions supports text input only; images are not supported",
                errorOf(() -> resource.decisions(
                        "{\"input\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_image\","
                                + " \"image_url\": \"x\"}]}]," + q + "}")));
        assertEquals("input: every content part must be an object with \"type\": \"input_text\" and text",
                errorOf(() -> resource.decisions(
                        "{\"input\": [{\"role\": \"user\", \"content\": [{\"type\": \"audio\", \"data\": \"x\"}]}],"
                                + q + "}")));
        assertEquals(400, statusOf(() -> resource.decisions(
                "{\"input\": \"s\", \"safety_identifier\": " + "\"" + "x".repeat(129) + "\", " + q + "}")));
    }

    // ------------------------------------------------------------------ option order

    @Test
    void theJointHeadReordersOptionsTheWayItReadsThem() {
        List<DecisionQuestions.Option> request = List.of(
                new DecisionQuestions.Option("\"zebra\"", "z"),
                new DecisionQuestions.Option("\"apple\"", "a"),
                new DecisionQuestions.Option("\"mango\"", "m"));
        List<DecisionQuestions.Option> clef = DecisionQuestions.reorder(request, DecisionQuestions.CHOICE,
                DecisionQuestions.Flags.CLEF);
        assertEquals(List.of("\"apple\"", "\"mango\"", "\"zebra\""), keys(clef));
        // the head's output rows follow the prompt order; the answer probabilities return to the request order
        int[] orders = DecisionQuestions.orders(clef, request);
        double[] promptProbs = {0.2, 0.3, 0.5};
        DecisionQuestions.Question question = DecisionQuestions.Question.vllm("0", "pick", 1, "pick one", clef,
                List.of("zebra", "apple", "mango"), null, orders);
        double[] requestProbs = question.requestOrder(promptProbs);
        assertEquals(List.of(0.5, 0.2, 0.3), List.of(requestProbs[0], requestProbs[1], requestProbs[2]));

        List<DecisionQuestions.Option> flags = List.of(new DecisionQuestions.Option("false", null),
                new DecisionQuestions.Option("true", null));
        assertEquals(List.of("true", "false"), keys(DecisionQuestions.reorder(flags, DecisionQuestions.NOUL,
                DecisionQuestions.Flags.CLEF)));
        assertEquals(List.of("false", "true"), keys(DecisionQuestions.reorder(flags, DecisionQuestions.NOUL,
                DecisionQuestions.Flags.VLLM_DECISION2)), "the strict head keeps the request order");
        assertEquals(List.of("false", "true"), keys(DecisionQuestions.reorder(flags, DecisionQuestions.CHOICE,
                DecisionQuestions.Flags.CLEF)), "true-first is a noul rule, a two-option choice keeps its order");
    }

    // ------------------------------------------------------------------ helpers

    private record Response_(JsonView body) {
    }

    private Response_ systemone(String body) {
        return new Response_(view(resource.systemone(body).getEntity()));
    }

    /** The head's raw scores of the one-question {@code body}'s prompt, through the service. */
    private float[] rawScores(String body) {
        List<DecisionQuestions.Question> questions = DecisionQuestions.parseSystemOne(ServerJson.parse(body),
                DecisionQuestions.Flags.VLLM_DECISION2);
        DecisionPrompt.Built prompt = DecisionPrompt.build(decision, questions, Map.of("text", "rate it"));
        try {
            return decision.pooling().decisionScores(null, prompt.tokens(), prompt.order());
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    private static double[] softmax(double[] logits, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }
        double[] out = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.exp((logits[i] - max) / temperature);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    private static String manyQuestions(int n) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < n; i++) {
            out.append(i > 0 ? ", " : "").append("{\"type\": \"predicate\", \"instructions\": \"x\"}");
        }
        return out.append("]").toString();
    }

    private static String manyChoices(int n) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < n; i++) {
            out.append(i > 0 ? ", " : "").append("{\"value\": \"v").append(i).append("\"}");
        }
        return out.append("]").toString();
    }

    private static List<String> keys(List<DecisionQuestions.Option> options) {
        List<String> out = new ArrayList<>();
        for (DecisionQuestions.Option option : options) {
            out.add(option.key());
        }
        return out;
    }
}
