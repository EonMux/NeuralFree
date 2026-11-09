package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The decision families over the real service: openjev's letter readout, lev's two option orders and nine-point
 * rating, nimble's all-questions prompt, vllm-d3's code readout with a null state and canonical legends, laya's
 * marker columns over a modern-bert decision checkpoint and lfm2-d1's answer-code logits. Every read is compared
 * against the same computation over the model's own output logits (or decision scores) on the prompt the family
 * renders, so the endpoint cannot pass without the reference arithmetic.
 */
public class DecisionFamiliesTest {

    private static EngineService openjev;
    private static EngineService lev;
    private static EngineService nimble;
    private static EngineService d3;
    private static EngineService laya;
    private static EngineService lfm2;
    private static DecisionResource openjevResource;
    private static DecisionResource levResource;
    private static DecisionResource nimbleResource;
    private static DecisionResource d3Resource;
    private static DecisionResource layaResource;
    private static DecisionResource lfm2Resource;

    @BeforeAll
    static void open() throws IOException {
        EngineConfig config = ServerTestSupport.config().build();
        openjev = service(SyntheticModels.decisionLabels("openjev", context()), config);
        lev = service(SyntheticModels.decisionLabels("lev", context()), config);
        nimble = service(SyntheticModels.decisionLabels("nimble", context(), nimbleTemplate()), config);
        d3 = service(SyntheticModels.decisionLabels("vllm-d3", context()), config);
        laya = service(SyntheticModels.modernBertDecision(), config);
        lfm2 = service(SyntheticModels.lfm2Decision(), config);
        openjevResource = new DecisionResource(openjev);
        levResource = new DecisionResource(lev);
        nimbleResource = new DecisionResource(nimble);
        d3Resource = new DecisionResource(d3);
        layaResource = new DecisionResource(laya);
        lfm2Resource = new DecisionResource(lfm2);
    }

    @AfterAll
    static void close() {
        for (EngineService service : List.of(openjev, lev, nimble, d3, laya, lfm2)) {
            if (service != null) {
                service.close();
            }
        }
    }

    /** The label fixtures' context: the test prompts fit comfortably. */
    private static SyntheticModels.Config context() {
        return SyntheticModels.Config.qwen2().toBuilder().nCtxTrain(1024).build();
    }

    private static EngineService service(GgufTestWriter writer, EngineConfig config) throws IOException {
        Path path = writer.writeTempFile();
        return new EngineService(path.toString(), config);
    }

    /** The nimble template, which also lists every question of the request. */
    private static String nimbleTemplate() {
        return "{% for q in questions %}{{ q.type }}|{{ q.instructions }}|"
                + "{% for o in q.options %}{{ o.label }}{{ o.key }}:{{ o.description }};{% endfor %}"
                + "|{{ state }}|{{ questions | length }}|Answer:{% endfor %}";
    }

    // ------------------------------------------------------------------ the label families

    @Test
    void openjevReadsTheLetterLogitsOfTheLastPromptToken() throws IOException {
        String body = """
                {"state": "the sky is blue",
                 "questions": {"color": {"type": "choice", "instructions": "What color is the sky?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = systemone(openjevResource, body).obj("answers").obj("color");
        assertNotNull(answer);
        double[] expected = labelProbabilities(openjev, renderPrompt(SyntheticModels.LABEL_TEMPLATE, "choice",
                "What color is the sky?", "the sky is blue",
                List.of(option("A", "red", "warm"), option("B", "blue", "cool"))),
                new String[]{"A", "B"}, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
        assertEquals(expected[0] > expected[1] ? "red" : "blue", answer.str("choice"));
        assertEquals(Math.max(0.0, (Math.max(expected[0], expected[1]) - 0.5) / 0.5),
                answer.f64("confidence"), 1e-9);
    }

    @Test
    void levAveragesTheTwoOptionOrdersUnderItsSmallTemperature() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice", "instructions": "Which?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = systemone(levResource, body).obj("answers").obj("q");
        double[] forward = labelProbabilities(lev, renderPrompt(SyntheticModels.LABEL_TEMPLATE, "choice", "Which?",
                "x", List.of(option("A", "red", "warm"), option("B", "blue", "cool"))),
                new String[]{"A", "B"}, 0.5f);
        double[] reversed = labelProbabilities(lev, renderPrompt(SyntheticModels.LABEL_TEMPLATE, "choice", "Which?",
                "x", List.of(option("A", "blue", "cool"), option("B", "red", "warm"))),
                new String[]{"A", "B"}, 0.5f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals((forward[0] + reversed[1]) / 2.0, probabilities.f64("red"), 1e-5);
        assertEquals((forward[1] + reversed[0]) / 2.0, probabilities.f64("blue"), 1e-5);
    }

    @Test
    void levNoulReadsTheNinePointRatingScale() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "noul", "instructions": "Is it true?",
                     "criteria": {"true": "yes", "false": "no"}}}}""";
        JsonView answer = systemone(levResource, body).obj("answers").obj("q");
        double[] ratings = labelProbabilities(lev, renderPrompt(SyntheticModels.LABEL_TEMPLATE, "noul",
                "Is it true?", "x", List.of(option("A", "false", "no"), option("B", "true", "yes"))),
                new String[]{"A", "B", "C", "D", "E", "F", "G", "H", "I"}, 0.5f);
        double expected = 0.0;
        for (int i = 0; i < ratings.length; i++) {
            expected += ratings[i] * i / (ratings.length - 1);
        }
        assertEquals(expected, answer.f64("noul"), 1e-9);
        assertEquals(2, answer.keys().size(), "a noul answer carries only its type and noul");
    }

    @Test
    void nimbleListsEveryQuestionOfTheRequestInEachPrompt() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"a": {"type": "choice", "instructions": "A?",
                                     "criteria": {"red": "warm", "blue": "cool"}},
                               "b": {"type": "choice", "instructions": "B?",
                                     "criteria": {"up": "high", "down": "low"}}}}""";
        JsonView answer = systemone(nimbleResource, body).obj("answers").obj("a");
        List<Map<String, Object>> questions = List.of(
                question("a", "choice", "A?", List.of(option("A", "red", "warm"), option("B", "blue", "cool"))),
                question("b", "choice", "B?", List.of(option("A", "up", "high"), option("B", "down", "low"))));
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", "choice");
        inp.put("instructions", "A?");
        inp.put("state", "x");
        inp.put("options", questions.get(0).get("options"));
        inp.put("questions", questions);
        inp.put("images", List.of());
        double[] expected = labelProbabilities(nimble, MiniJinja.render(nimbleTemplate(), inp),
                new String[]{"A", "B"}, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void vllmD3ReadsCodesAndDumpsNonTextLegendsCanonically() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": [{"level": 1}, "b"]}}}""";
        JsonView answer = systemone(d3Resource, body).obj("answers").obj("q");
        Map<String, Object> level = new LinkedHashMap<>();
        level.put("level", 1L);
        double[] expected = labelProbabilities(d3, renderPrompt(SyntheticModels.LABEL_TEMPLATE, "score", "Rate.",
                "x", List.of(option("A", "0", level), option("B", "1", "b"))),
                new String[]{"A", "B"}, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("0"), 1e-5);
        assertEquals(expected[1], probabilities.f64("1"), 1e-5);
        JsonView legend = answer.obj("legend");
        assertTrue(legend.str("0").contains("level"), "a non-text description enters as its JSON dump");
        double entropy = 0.0;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] > 0) {
                entropy -= expected[i] * Math.log(expected[i]);
            }
        }
        assertEquals(1.0 - entropy / Math.log(2), answer.f64("confidence"), 1e-9);
    }

    @Test
    void vllmD3AcceptsANullStateAndADefaultQuestion() {
        String body = """
                {"state": null,
                 "questions": {"q": {"type": "choice",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = systemone(d3Resource, body).obj("answers").obj("q");
        assertEquals("choice", answer.str("type"));
        assertEquals(1.0, answer.obj("probabilities").f64("red")
                + answer.obj("probabilities").f64("blue"), 1e-9);
    }

    // ------------------------------------------------------------------ the marker head

    @Test
    void layaReadsTheQuestionTypeColumnAtEachMarker() throws IOException {
        String body = """
                {"state": "the sky",
                 "questions": {"q": {"type": "choice", "instructions": "What color?",
                     "criteria": {"red": null, "blue": null}}}}""";
        JsonView answer = systemone(layaResource, body).obj("answers").obj("q");
        double[] expected = markerProbabilities(laya, renderPrompt(SyntheticModels.LAYA_TEMPLATE, "choice",
                "What color?", "the sky", List.of(option("", "red", null), option("", "blue", null))),
                DecisionQuestions.CHOICE);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void layaReadsTheScoreColumnOfAScoreQuestion() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": ["bad", "ok", "good"]}}}""";
        JsonView answer = systemone(layaResource, body).obj("answers").obj("q");
        double[] expected = markerProbabilities(laya, renderPrompt(SyntheticModels.LAYA_TEMPLATE, "score",
                "Rate.", "x", List.of(option("0", "0", "bad"), option("1", "1", "ok"), option("2", "2", "good"))),
                DecisionQuestions.SCORE);
        JsonView probabilities = answer.obj("probabilities");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], probabilities.f64(String.valueOf(i)), 1e-5);
        }
        double score = 0.0;
        for (int i = 0; i < expected.length; i++) {
            score += i * expected[i];
        }
        assertEquals(score, answer.f64("score"), 1e-9);
    }

    // ------------------------------------------------------------------ lfm2-d1

    @Test
    void lfm2D1ReadsTheAnswerCodeLogits() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice", "instructions": "Pick one",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = systemone(lfm2Resource, body).obj("answers").obj("q");
        double[] expected = labelProbabilities(lfm2, renderPrompt(SyntheticModels.LFM2_D1_TEMPLATE, "choice",
                "Pick one", "x", List.of(option("A", "red", "warm"), option("B", "blue", "cool"))),
                new String[]{"A", "B"}, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void lfm2D1ReadsTheLevelDigitsOfAScoreQuestion() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": ["bad", "ok", "good"]}}}""";
        JsonView answer = systemone(lfm2Resource, body).obj("answers").obj("q");
        double[] expected = labelProbabilities(lfm2, renderPrompt(SyntheticModels.LFM2_D1_TEMPLATE, "score",
                "Rate.", "x", List.of(option("0", "0", "bad"), option("1", "1", "ok"), option("2", "2", "good"))),
                new String[]{"0", "1", "2"}, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], probabilities.f64(String.valueOf(i)), 1e-5);
        }
    }

    // ------------------------------------------------------------------ /v1/decisions

    @Test
    void decisionsReadTheLabelFamiliesWithVllmShapes() throws IOException {
        String body = """
                {"input": "the sky is blue",
                 "questions": [
                     {"type": "choice", "name": "color", "instructions": "What color is the sky?",
                      "choices": [{"value": "red", "description": "warm"},
                                  {"value": "blue", "description": "cool"}]}]}""";
        JsonView out = decisions(openjevResource, body);
        JsonView answer = at(out, "answers", 0);
        assertEquals("color", answer.str("name"));
        // vLLM's choices enter the prompt under their canonical JSON, the answer echoes the raw value
        double[] expected = labelProbabilities(openjev, renderPrompt(SyntheticModels.LABEL_TEMPLATE, "choice",
                "What color is the sky?", "the sky is blue",
                List.of(option("A", "\"red\"", "warm"), option("B", "\"blue\"", "cool"))),
                new String[]{"A", "B"}, 1.0f);
        JsonView probabilities = at(answer, "probabilities", 0);
        assertEquals("red", probabilities.str("value"));
        assertEquals(expected[0], probabilities.f64("probability"), 1e-5);
        JsonView second = at(answer, "probabilities", 1);
        assertEquals(expected[1], second.f64("probability"), 1e-5);
        assertEquals(expected[0] > expected[1] ? "red" : "blue", answer.str("choice"));
        assertTrue(out.obj("usage").i64("input_tokens") > 0);
    }

    @Test
    void decisionsCapTheOptionsAtTheFamilysLabelCount() {
        StringBuilder choices = new StringBuilder("[");
        for (int i = 0; i < 27; i++) {
            choices.append(i > 0 ? ", " : "").append("{\"value\": \"v").append(i).append("\"}");
        }
        choices.append("]");
        String body = "{\"input\": \"x\", \"questions\": [{\"type\": \"choice\", \"instructions\": \"Pick\","
                + " \"choices\": " + choices + "}]}";
        assertEquals(400, statusOf(() -> openjevResource.decisions(body)));
        assertEquals("This model supports at most 26 choices per question",
                errorOf(() -> openjevResource.decisions(body)));
    }

    @Test
    void theJointOptionIsAdmittedOnlyByTheJointHead() {
        assertTrue(DecisionFamily.of("clef").jointReads);
        assertFalse(DecisionFamily.of("vllm-decision2").jointReads);
        assertFalse(DecisionFamily.of("openjev").jointReads);
        assertTrue(DecisionFamily.of("vllm-d3").canUseImages);
        assertTrue(DecisionFamily.of("lfm2-d1").canUseImages);
        assertFalse(DecisionFamily.of("lev").canUseImages);
        assertFalse(DecisionFamily.of("laya").canUseImages);
        assertEquals("small", DecisionFamily.of("lev").bucket(8));
        assertEquals("mid", DecisionFamily.of("lev").bucket(9));
        assertEquals("large", DecisionFamily.of("lev").bucket(27));
        assertEquals("3_5", DecisionFamily.of("openjev").bucket(3));
        assertTrue(DecisionFamily.of("granite-switch").noulTrueFirst);
        assertFalse(DecisionFamily.of("granite-switch").canUseImages);
        assertFalse(DecisionFamily.of("granite-switch").jointReads);
        assertFalse(DecisionFamily.of("kev").canUseImages);
        assertFalse(DecisionFamily.of("kev").jointReads);
        assertFalse(DecisionFamily.of("kev").noulTrueFirst);
    }

    @Test
    void aModelTypeTheEngineCannotReadRefusesByName() throws IOException {
        Path path = SyntheticModels.decisionLabels("unknown-ev", context()).writeTempFile();
        EngineService model = new EngineService(path.toString(), ServerTestSupport.config().build());
        try {
            DecisionResource resource = new DecisionResource(model);
            String body = "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\","
                    + " \"instructions\": \"x\"}}}";
            assertEquals(501, statusOf(() -> resource.systemone(body)));
            String error = errorOf(() -> resource.systemone(body));
            assertTrue(error.contains("unknown-ev"));
            assertTrue(error.contains("lfm3-d1"));
        } finally {
            model.close();
        }
    }

    /** A systemone call, its error body shown when it refuses. */
    private static JsonView systemone(DecisionResource resource, String body) {
        return call(() -> resource.systemone(body));
    }

    /** A decisions call, its error body shown when it refuses. */
    private static JsonView decisions(DecisionResource resource, String body) {
        return call(() -> resource.decisions(body));
    }

    private static JsonView call(java.util.function.Supplier<jakarta.ws.rs.core.Response> call) {
        try {
            return view(call.get().getEntity());
        } catch (WebApplicationException e) {
            throw new AssertionError("HTTP " + e.getResponse().getStatus() + ": "
                    + e.getResponse().getEntity(), e);
        }
    }

    // ------------------------------------------------------------------ reference computations

    /**
     * The label read over the model's own output logits: the prompt tokenized as the server tokenizes it, the last
     * row's logits at each answer code, and the temperature-scaled softmax.
     */
    private static double[] labelProbabilities(EngineService model, String prompt, String[] labels, float temperature)
            throws IOException {
        int[] tokens = model.encode(prompt, false, true);
        float[] logits = model.pooling().nextLogits(null, tokens);
        float[] scores = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            int[] ids = model.encode(labels[i], false, false);
            assertEquals(1, ids.length, "label " + labels[i] + " must be a single token");
            scores[i] = logits[ids[0]];
        }
        return softmax(scores, temperature);
    }

    /**
     * The laya read over the model's own decision scores: the marker positions of the [MASK] tokens and the
     * question type's column at each.
     */
    private static double[] markerProbabilities(EngineService model, String prompt, int type) throws IOException {
        int[] tokens = model.encode(prompt, false, true);
        int mask = model.tokenizer().specialTokens().mask();
        int n = 0;
        for (int token : tokens) {
            if (token == mask) {
                n++;
            }
        }
        assertTrue(n > 0, "the prompt carries a mask token per option");
        float[] scores = model.pooling().decisionScores(null, tokens, null);
        float[] columns = new float[n];
        int at = 0;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] == mask) {
                columns[at++] = scores[i * 3 + type];
            }
        }
        return softmax(columns, 1.0f);
    }

    /** A label-family prompt: the template over one question, as llama.cpp's render fills it. */
    private static String renderPrompt(String template, String type, String instructions, Object state,
                                       List<Map<String, Object>> options) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", type);
        inp.put("instructions", instructions);
        inp.put("state", state);
        inp.put("options", options);
        inp.put("images", List.of());
        return MiniJinja.render(template, inp);
    }

    /** One rendered option: its label, key and description. */
    private static Map<String, Object> option(String label, String key, Object description) {
        Map<String, Object> option = new LinkedHashMap<>();
        option.put("label", label);
        option.put("key", key);
        option.put("description", description);
        return option;
    }

    /** One rendered question of a nimble prompt. */
    private static Map<String, Object> question(String id, String type, String instructions,
                                                List<Map<String, Object>> options) {
        Map<String, Object> question = new LinkedHashMap<>();
        question.put("id", id);
        question.put("type", type);
        question.put("instructions", instructions);
        question.put("options", options);
        return question;
    }

    /** The softmax the answers apply: the max-shifted exponential over the temperature. */
    private static double[] softmax(float[] scores, float temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (float score : scores) {
            max = Math.max(max, score);
        }
        double[] out = new double[scores.length];
        double sum = 0.0;
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.exp((scores[i] - max) / temperature);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }
}
