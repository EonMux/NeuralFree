package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
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
 * The kev family over the real service: the prompt tokenized with its {@code <|box_end|>} option markers, and every
 * option's score the dot product of the last prompt token's hidden state with its marker's (llama.cpp's
 * {@code fill_task} and the pointer read of {@code send_decision}). The state, the instructions and the option texts
 * flatten before the template, their special-token spellings broken. Every read is compared against the dot
 * computation over the model's own hidden rows on the prompt the family renders.
 */
class DecisionKevTest {

    private static final String MARKER = "<|box_end|>";
    /** The request's spelling of a marker the kev prompt flattens away. */
    private static final String RAW_ENDO = "<|endoftext|>";
    /** kev breaks the spellings of the request's special tokens with U+00A6 in place of the pipes. */
    private static final String BROKEN_MARKER = "<¦endoftext¦>";

    private static EngineService kev;
    private static EngineService wrongLayout;
    private static DecisionResource resource;
    private static DecisionResource wrongLayoutResource;

    @BeforeAll
    static void open() throws IOException {
        EngineConfig config = config().build();
        kev = service(SyntheticModels.decisionKev(kevContext()), config);
        wrongLayout = service(SyntheticModels.decisionKev(kevContext(),
                "{{ type }}|{{ instructions }}|<|box_end|>|{{ state }}|Answer:"), config);
        resource = new DecisionResource(kev);
        wrongLayoutResource = new DecisionResource(wrongLayout);
    }

    @AfterAll
    static void close() {
        for (EngineService service : List.of(kev, wrongLayout)) {
            if (service != null) {
                service.close();
            }
        }
    }

    /** The kev fixture's context: the test prompts fit comfortably. */
    private static SyntheticModels.Config kevContext() {
        return SyntheticModels.Config.qwen2().toBuilder().nCtxTrain(1024).build();
    }

    private static EngineService service(GgufTestWriter writer, EngineConfig config) throws IOException {
        Path path = writer.writeTempFile();
        return new EngineService(path.toString(), config);
    }

    @Test
    void choiceAnswersTheDotsAgainstEveryOptionEndMarker() throws IOException {
        String body = """
                {"state": "the sky is blue",
                 "questions": {"q": {"type": "choice", "instructions": "What color is the sky?",
                     "criteria": {"red": "warm", "blue": "cool", "green": "leafy"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = kevPrompt("choice", "the sky is blue", "What color is the sky?",
                kevOption("red", "warm"), kevOption("blue", "cool"), kevOption("green", "leafy"));
        double[] expected = kevProbabilities(prompt, 3);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
        assertEquals(expected[2], probabilities.f64("green"), 1e-5);
        assertEquals(expected[0] > expected[1] && expected[0] > expected[2] ? "red"
                : expected[1] > expected[2] ? "blue" : "green", answer.str("choice"));
        // TypeSafe's choice confidence: the top probability's edge over uniform, rescaled
        double top = Math.max(expected[0], Math.max(expected[1], expected[2]));
        assertEquals(Math.max(0.0, (top - 1.0 / 3) / (1.0 - 1.0 / 3)), answer.f64("confidence"), 1e-9);
        assertEquals("choice", answer.str("type"));
    }

    @Test
    void noulAnswersTheTrueOptionMarkerDot() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "noul", "instructions": "Does it hold?",
                     "criteria": {"true": "holds", "false": "not"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        // kev keeps llama.cpp's default parse order: the false option first, then the true one
        String prompt = kevPrompt("noul", "x", "Does it hold?", kevOption("false", "not"),
                kevOption("true", "holds"));
        double[] expected = kevProbabilities(prompt, 2);
        assertEquals(expected[1], answer.f64("noul"), 1e-9);
        assertEquals(2, answer.keys().size(), "a noul answer carries only its type and noul");
    }

    @Test
    void scoreAnswersTheDotsWithTheScoreConfidence() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": ["poor", "fair", "fine"]}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        String prompt = kevPrompt("score", "x", "Rate.", kevOption("0", "poor"), kevOption("1", "fair"),
                kevOption("2", "fine"));
        double[] expected = kevProbabilities(prompt, 3);
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
    void specialTokenSpellingsInTheInputAreBroken() throws IOException {
        String body = """
                {"state": "sky <|endoftext|> blue",
                 "questions": {"q": {"type": "choice", "instructions": "Is <|endoftext|> it?",
                     "criteria": {"<|endoftext|>": "warm", "blue": "cool"}}}}""";
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        // the request's spellings reach the prompt broken, and the keys of the answer stay as sent
        String prompt = kevPrompt("choice", "sky " + BROKEN_MARKER + " blue", "Is " + BROKEN_MARKER + " it?",
                kevOption(BROKEN_MARKER, "warm"), kevOption("blue", "cool"));
        double[] expected = kevProbabilities(prompt, 2);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64(RAW_ENDO), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void aPromptWithoutOneMarkerPerOptionRefuses() {
        // the fixture's template writes its single marker outside the options, so the layout cannot match
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice", "instructions": "Pick one",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        assertEquals(500, statusOf(() -> wrongLayoutResource.systemone(body)));
        assertEquals("unexpected layout of the decision prompt",
                errorOf(() -> wrongLayoutResource.systemone(body)));
    }

    @Test
    void aVocabularyWithoutTheEndTokenRefusesByName() throws IOException {
        // the label fixture carries the kev type but no <|box_end|> token in its vocabulary
        EngineService model = service(SyntheticModels.decisionLabels("kev", kevContext(),
                SyntheticModels.KEV_TEMPLATE), config().build());
        try {
            DecisionResource noMarker = new DecisionResource(model);
            IllegalStateException e = org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                    () -> noMarker.systemone("{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"choice\","
                            + " \"instructions\": \"i\"}}}"));
            assertEquals("decision model has no " + MARKER + " token", e.getMessage());
        } finally {
            model.close();
        }
    }

    @Test
    void anImageInTheStateRefuses() {
        String body = """
                {"state": {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "what is shown"}]}]},
                 "questions": {"q": {"type": "choice", "instructions": "Pick one",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        // kev reads text prompts only (llama.cpp's can_use_images)
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
        // vLLM's choices enter the prompt under their canonical JSON
        String prompt = kevPrompt("choice", "x", "Pick one", kevOption("\"red\"", "warm"),
                kevOption("\"blue\"", "cool"));
        double[] expected = kevProbabilities(prompt, 2);
        JsonView probabilities = at(answer, "probabilities", 0);
        assertEquals("red", probabilities.str("value"));
        assertEquals(expected[0], probabilities.f64("probability"), 1e-5);
        assertEquals(expected[1], at(answer, "probabilities", 1).f64("probability"), 1e-5);
        assertTrue(out.obj("usage").i64("input_tokens") > 0);
    }

    // ------------------------------------------------------------------ reference computations

    /** The kev prompt: the template over the flattened input, as the family renders it. */
    private static String kevPrompt(String type, Object state, Object instructions,
                                    Map<String, Object>... options) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("id", "q");
        inp.put("type", type);
        inp.put("instructions", instructions);
        inp.put("state", state);
        inp.put("options", List.of(options));
        inp.put("images", List.of());
        return MiniJinja.render(SyntheticModels.KEV_TEMPLATE, inp);
    }

    /** One rendered option: its key and description. */
    private static Map<String, Object> kevOption(String key, Object description) {
        Map<String, Object> option = new LinkedHashMap<>();
        option.put("key", key);
        option.put("description", description);
        return option;
    }

    /**
     * The kev read over the model's own hidden rows: the marker positions of the prompt, the last token's row dotted
     * with each marker's second half, and the softmax at temperature one.
     */
    private static double[] kevProbabilities(String prompt, int nOptions) throws IOException {
        int marker = kev.encode(MARKER, false, true)[0];
        int[] tokens = kev.encode(prompt, false, true);
        int nMarkers = 0;
        for (int token : tokens) {
            nMarkers += token == marker ? 1 : 0;
        }
        assertEquals(nOptions, nMarkers, "the prompt carries one end marker per option");
        // the same dispatch the model itself uses: paged memory maps its positions through PagedSequences
        ForwardBackend backend = kev.primary().model().forwardBackend();
        float[][] hidden = PagedEvaluations.paged(backend)
                ? PagedEvaluations.hiddenRows(backend, PagedEvaluations.untyped(tokens, null))
                : backend.hiddenRows(tokens, null);
        int half = hidden[0].length / 2;
        float[] q = hidden[tokens.length - 1];
        float[] scores = new float[nMarkers];
        int at = 0;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] != marker) {
                continue;
            }
            float dot = 0.0f;
            for (int j = 0; j < half; j++) {
                dot += q[j] * hidden[i][half + j];
            }
            scores[at++] = dot / (float) Math.sqrt(half);
        }
        double max = Double.NEGATIVE_INFINITY;
        for (float score : scores) {
            max = Math.max(max, score);
        }
        double[] probs = new double[scores.length];
        double sum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            probs[i] = Math.exp(scores[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }
}
