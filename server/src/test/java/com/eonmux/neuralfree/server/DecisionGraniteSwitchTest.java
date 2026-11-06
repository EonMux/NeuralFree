package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The granite-switch family over the real service: the label readout of the codes — the logits of one answer code
 * per option at the last prompt token, the true option first for noul (llama.cpp's {@code noul_true_first}),
 * TypeSafe's temperature buckets and confidence formulas, and no image input. The prompt itself is plain text; the
 * in-graph switched adapter of the granite-switch checkpoint rides the generative forward and takes no part in the
 * decision read. Every read is compared against the same computation over the model's own output logits on the
 * prompt the family renders.
 */
class DecisionGraniteSwitchTest {

    private static EngineService granite;
    private static DecisionResource resource;

    @BeforeAll
    static void open() throws IOException {
        granite = new EngineService(SyntheticModels.decisionLabels("granite-switch", context())
                .writeTempFile().toString(), config().build());
        resource = new DecisionResource(granite);
    }

    @AfterAll
    static void close() {
        if (granite != null) {
            granite.close();
        }
    }

    /** The label fixture's context: the test prompts fit comfortably. */
    private static SyntheticModels.Config context() {
        return SyntheticModels.Config.qwen2().toBuilder().nCtxTrain(1024).build();
    }

    @Test
    void choiceAnswersTheAnswerCodesAtTheLastPromptToken() throws IOException {
        String body = """
                {"state": "the sky is blue",
                 "questions": {"q": {"type": "choice", "instructions": "What color is the sky?",
                     "criteria": {"red": "warm", "blue": "cool", "green": "leafy"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = granitePrompt("choice", "the sky is blue", "What color is the sky?",
                option("A", "red", "warm"), option("B", "blue", "cool"), option("C", "green", "leafy"));
        double[] expected = probabilities(prompt, new String[] {"A", "B", "C"}, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
        assertEquals(expected[2], probabilities.f64("green"), 1e-5);
        assertEquals(expected[0] > expected[1] && expected[0] > expected[2] ? "red"
                : expected[1] > expected[2] ? "blue" : "green", answer.str("choice"));
        // TypeSafe's choice confidence: the top probability's edge over uniform, rescaled
        double top = Math.max(expected[0], Math.max(expected[1], expected[2]));
        assertEquals(Math.max(0.0, (top - 1.0 / 3) / (1.0 - 1.0 / 3)), answer.f64("confidence"), 1e-9);
    }

    @Test
    void noulReadsTheTrueOptionFirst() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "noul", "instructions": "Does it hold?",
                     "criteria": {"true": "holds", "false": "not"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        // the parse puts the true option first, so its code reads the first label of the prompt
        String prompt = granitePrompt("noul", "x", "Does it hold?", option("A", "true", "holds"),
                option("B", "false", "not"));
        double[] expected = probabilities(prompt, new String[] {"A", "B"}, 1.0f);
        assertEquals(expected[0], answer.f64("noul"), 1e-9);
        assertEquals(2, answer.keys().size(), "a noul answer carries only its type and noul");
    }

    @Test
    void scoreAnswersTheExpectedLevelWithTheScoreConfidence() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": ["poor", "fair", "fine"]}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = granitePrompt("score", "x", "Rate.", option("A", "0", "poor"),
                option("B", "1", "fair"), option("C", "2", "fine"));
        double[] expected = probabilities(prompt, new String[] {"A", "B", "C"}, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("0"), 1e-5);
        assertEquals(expected[1], probabilities.f64("1"), 1e-5);
        assertEquals(expected[2], probabilities.f64("2"), 1e-5);
        // TypeSafe's score confidence: one minus the mean distance to the mode, relative to uniform's
        int mode = 0;
        for (int i = 1; i < expected.length; i++) {
            if (expected[i] > expected[mode]) {
                mode = i;
            }
        }
        double dist = 0.0;
        for (int i = 0; i < expected.length; i++) {
            dist += expected[i] * Math.abs((double) i - mode);
        }
        double distUniform = 0.0;
        for (int i = 0; i < expected.length; i++) {
            distUniform += Math.abs(i - (expected.length - 1) / 2.0) / expected.length;
        }
        assertEquals(Math.max(0.0, 1.0 - dist / distUniform), answer.f64("confidence"), 1e-9);
    }

    @Test
    void decisionsServeTheVllmShapesAndCapAtTheCodesLimit() throws IOException {
        String body = """
                {"input": "x",
                 "questions": [
                     {"type": "choice", "name": "color", "instructions": "Pick one",
                      "choices": [{"value": "red", "description": "warm"},
                                  {"value": "blue", "description": "cool"}]}]}""";
        JsonView out = view(resource.decisions(body).getEntity());
        JsonView answer = at(out, "answers", 0);
        assertEquals("color", answer.str("name"));
        // vLLM's choices enter the prompt under their canonical JSON
        String prompt = granitePrompt("choice", "x", "Pick one", option("A", "\"red\"", "warm"),
                option("B", "\"blue\"", "cool"));
        double[] expected = probabilities(prompt, new String[] {"A", "B"}, 1.0f);
        JsonView probabilities = at(answer, "probabilities", 0);
        assertEquals("red", probabilities.str("value"));
        assertEquals(expected[0], probabilities.f64("probability"), 1e-5);
        assertTrue(out.obj("usage").i64("input_tokens") > 0);

        StringBuilder choices = new StringBuilder("[");
        for (int i = 0; i < 27; i++) {
            choices.append(i > 0 ? ", " : "").append("{\"value\": \"v").append(i).append("\"}");
        }
        choices.append("]");
        String tooMany = "{\"input\": \"x\", \"questions\": [{\"type\": \"choice\","
                + " \"instructions\": \"Pick\", \"choices\": " + choices + "}]}";
        assertEquals(400, statusOf(() -> resource.decisions(tooMany)));
        assertEquals("This model supports at most 26 choices per question",
                errorOf(() -> resource.decisions(tooMany)));
    }

    @Test
    void anImageInTheStateRefuses() {
        String body = """
                {"state": {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "what is shown"}]}]},
                 "questions": {"q": {"type": "choice", "instructions": "Pick one",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        // granite-switch reads text prompts only (llama.cpp's can_use_images)
        assertEquals(501, statusOf(() -> resource.systemone(body)));
        assertEquals("This server does not support image input for decisions. For a model that supports it,"
                        + " start it with `--mmproj`",
                errorOf(() -> resource.systemone(body)));
    }

    // ------------------------------------------------------------------ reference computations

    /** The granite-switch prompt: the label template over one question, as llama.cpp's render fills it. */
    private static String granitePrompt(String type, Object state, Object instructions,
                                        Map<String, Object>... options) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", type);
        inp.put("instructions", instructions);
        inp.put("state", state);
        inp.put("options", List.of(options));
        inp.put("images", List.of());
        return MiniJinja.render(SyntheticModels.LABEL_TEMPLATE, inp);
    }

    /** One rendered option: its label, key and description. */
    private static Map<String, Object> option(String label, String key, Object description) {
        Map<String, Object> option = new LinkedHashMap<>();
        option.put("label", label);
        option.put("key", key);
        option.put("description", description);
        return option;
    }

    /** The label read over the model's own output logits at the answer codes. */
    private static double[] probabilities(String prompt, String[] labels, float temperature) throws IOException {
        int[] tokens = granite.encode(prompt, false, true);
        float[] logits = granite.nextLogits(null, tokens);
        float[] scores = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            int[] ids = granite.encode(labels[i], false, false);
            assertEquals(1, ids.length, labels[i] + " must be a single token");
            scores[i] = logits[ids[0]];
        }
        double max = Double.NEGATIVE_INFINITY;
        for (float score : scores) {
            max = Math.max(max, score);
        }
        double[] probs = new double[scores.length];
        double sum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            probs[i] = Math.exp((scores[i] - max) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }
}
