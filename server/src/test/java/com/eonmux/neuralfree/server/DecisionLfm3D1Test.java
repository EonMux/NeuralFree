package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The lfm3-d1 family (the LFM3-VL d1 checkpoints) over the real service: the same answer-code read as lfm3-d1-omni
 * — the logits of each option's answer code at the last prompt token after the empty think block, the best of the
 * yes/no token groups for noul, the median level for score and the answer's own probability as the confidence, the
 * special-token spellings of the input broken with a zero-width space — with no audio input: a spoken question
 * refuses by name (llama.cpp's parse_questions), and a state video_url part counts as the request's media. Every
 * read is compared against the same computation over the model's own output logits on the prompt the family renders.
 */
class DecisionLfm3D1Test {

    private static EngineService lfm3;
    private static DecisionResource resource;

    @BeforeAll
    static void open() throws IOException {
        lfm3 = new EngineService(SyntheticModels.lfm3D1().writeTempFile().toString(),
                ServerTestSupport.config().build());
        resource = new DecisionResource(lfm3);
    }

    @AfterAll
    static void close() {
        if (lfm3 != null) {
            lfm3.close();
        }
    }

    @Test
    void theChoiceReadsTheAnswerCodesAtTheLastPromptToken() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice", "instructions": "Pick one",
                     "criteria": {"red": "warm", "blue": null}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = lfm3Prompt("x", "Pick one", "choice", lfm3Option("A", "red", "warm"),
                lfm3Option("B", "blue", null));
        double[] expected = probabilities(prompt, new String[] {"A", "B"}, 0.7f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
        assertEquals(expected[0] > expected[1] ? "red" : "blue", answer.str("choice"));
        // the confidence of the lfm3 d1 read is the answer's own probability
        assertEquals(Math.max(expected[0], expected[1]), answer.f64("confidence"), 1e-9);
        assertEquals("choice", answer.str("type"));
    }

    @Test
    void noulReadsTheBestOfTheYesAndNoTokenGroups() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "noul", "instructions": "Does it hold?",
                     "criteria": {"true": "holds", "false": "not"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        // true first; the label of each side is its key, the read takes the best of the yes or no forms
        String prompt = lfm3Prompt("x", "Does it hold?", "noul", lfm3Option("true", "true", "holds"),
                lfm3Option("false", "false", "not"));
        float[] logits = logits(prompt);
        double[] probs = softmax(new float[] {groupMax(logits, "yes", "Yes", "YES"),
                groupMax(logits, "no", "No", "NO")}, 0.5f);
        assertEquals(probs[0], answer.f64("noul"), 1e-9);
        assertEquals(2, answer.keys().size(), "a noul answer carries only its type and noul");
    }

    @Test
    void scoreAnswersTheMedianLevelWithTheAnswerProbability() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": ["poor", "fair", "fine"]}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = lfm3Prompt("x", "Rate.", "score", lfm3Option("0", "0", "poor"),
                lfm3Option("1", "1", "fair"), lfm3Option("2", "2", "fine"));
        float[] logits = logits(prompt);
        float[] scores = new float[3];
        for (int i = 0; i < 3; i++) {
            scores[i] = logits[lfm3.encode(String.valueOf(i), false, false)[0]];
        }
        double[] probs = softmax(scores, 0.4f);
        double best = 0.0;
        int mode = 0;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > best) {
                best = probs[i];
                mode = i;
            }
        }
        double mass = 0.0;
        int level = probs.length - 1;
        for (int i = 0; i < probs.length && mass < 0.5; i++) {
            mass += probs[i];
            level = i;
        }
        assertEquals((double) level, answer.f64("score"), 1e-9);
        assertEquals(best, answer.f64("confidence"), 1e-9);
        assertEquals(mode, (int) answer.f64("score"), "the median sits at the mode of a peaked read");
    }

    @Test
    void aNullStateIsAccepted() throws IOException {
        String body = """
                {"state": null,
                 "questions": {"q": {"type": "choice", "instructions": "Pick one",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = lfm3Prompt(null, "Pick one", "choice", lfm3Option("A", "red", "warm"),
                lfm3Option("B", "blue", "cool"));
        double[] expected = probabilities(prompt, new String[] {"A", "B"}, 0.7f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void specialTokenSpellingsInTheInputAreBroken() throws IOException {
        String body = """
                {"state": "<|endoftext|> and <|startoftext|>",
                 "questions": {"q": {"type": "choice", "instructions": "Is <|endoftext|> it?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = lfm3Prompt("<​|endoftext|> and <​|startoftext|>",
                "Is <​|endoftext|> it?", "choice", lfm3Option("A", "red", "warm"),
                lfm3Option("B", "blue", "cool"));
        double[] expected = probabilities(prompt, new String[] {"A", "B"}, 0.7f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void aSpokenQuestionRefusesByName() {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice",
                     "instructions": {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        assertEquals(400, statusOf(() -> resource.systemone(body)));
        assertEquals("questions.q: spoken instructions need a model with audio input",
                errorOf(() -> resource.systemone(body)));
    }

    @Test
    void aStateVideoUrlPartCountsAsTheRequestMedia() {
        String body = """
                {"state": {"messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                    {"type": "text", "text": "what is shown"}]}]},
                 "questions": {"q": {"type": "choice", "instructions": "Pick one",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        // the video part is pulled into the request's media, and this service carries no projector
        assertEquals(501, statusOf(() -> resource.systemone(body)));
        assertEquals("This server does not support image input for decisions. For a model that supports it,"
                        + " start it with `--mmproj`",
                errorOf(() -> resource.systemone(body)));
    }

    @Test
    void decisionsServeTheVllmShapesOverTheSameRead() throws IOException {
        String body = """
                {"input": "x",
                 "questions": [
                     {"type": "choice", "name": "color", "instructions": "Pick one",
                      "choices": [{"value": "red", "description": "warm"},
                                  {"value": "blue", "description": "cool"}]}]}""";
        JsonView out = view(resource.decisions(body).getEntity());
        JsonView answer = at(out, "answers", 0);
        assertEquals("color", answer.str("name"));
        // vLLM's choices enter the prompt under their canonical JSON, the answer echoes the raw value
        String prompt = lfm3Prompt("x", "Pick one", "choice", lfm3Option("A", "\"red\"", "warm"),
                lfm3Option("B", "\"blue\"", "cool"));
        double[] expected = probabilities(prompt, new String[] {"A", "B"}, 0.7f);
        JsonView probabilities = at(answer, "probabilities", 0);
        assertEquals("red", probabilities.str("value"));
        assertEquals(expected[0], probabilities.f64("probability"), 1e-5);
        assertEquals(expected[1], at(answer, "probabilities", 1).f64("probability"), 1e-5);
        assertEquals(expected[0] > expected[1] ? "red" : "blue", answer.str("choice"));
        assertTrue(out.obj("usage").i64("input_tokens") > 0);
    }

    // ------------------------------------------------------------------ reference computations

    /**
     * The lfm3 d1 prompt: the d1 template with the think marker over the rendered input, the state null rendering
     * empty as the template reads it.
     */
    private static String lfm3Prompt(Object state, Object instructions, String type,
                                     Map<String, Object>... options) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("id", "q");
        inp.put("type", type);
        inp.put("instructions", instructions);
        inp.put("state", state);
        inp.put("options", List.of(options));
        inp.put("images", List.of());
        return MiniJinja.render(SyntheticModels.LFM2_D1_TEMPLATE + "<think></think>", inp);
    }

    /** One rendered option of the d1 template: its label, key and description. */
    private static Map<String, Object> lfm3Option(String label, String key, Object description) {
        Map<String, Object> option = new LinkedHashMap<>();
        option.put("label", label);
        option.put("key", key);
        option.put("description", description);
        return option;
    }

    /** The label read over the model's own output logits at the answer codes. */
    private static double[] probabilities(String prompt, String[] labels, float temperature) throws IOException {
        float[] logits = logits(prompt);
        float[] scores = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            scores[i] = logits[lfm3.encode(labels[i], false, false)[0]];
        }
        return softmax(scores, temperature);
    }

    /** The output logits at the last token of the prompt, tokenized as the server tokenizes it. */
    private static float[] logits(String prompt) throws IOException {
        int[] tokens = lfm3.encode(prompt, false, true);
        return lfm3.nextLogits(null, tokens);
    }

    /** The best logit of one answer's token forms. */
    private static float groupMax(float[] logits, String... forms) throws IOException {
        float best = Float.NEGATIVE_INFINITY;
        for (String form : forms) {
            int[] ids = lfm3.encode(form, false, false);
            assertEquals(1, ids.length, form + " must be a single token");
            best = Math.max(best, logits[ids[0]]);
        }
        return best;
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
