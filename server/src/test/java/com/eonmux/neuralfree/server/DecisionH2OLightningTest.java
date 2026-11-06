package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The h2o-lightning family over the real service: the option labels come from the checkpoint's {@code
 * decision.labels}, the prompt follows h2o_lightning.py's systemone template with the Python-compatible JSON of
 * object instructions and score legends, the special-token spellings of the input stay text through {@code
 * h2o_quote}, and a state beyond 32000 tokens keeps its first and last halves. Every read is compared against the
 * same computation over the model's own output logits on the prompt the family renders.
 */
class DecisionH2OLightningTest {

    private static EngineService service;
    private static EngineService cut;
    private static EngineService mediaService;
    private static DecisionResource resource;
    private static DecisionResource mediaResource;
    private static String image;

    @BeforeAll
    static void open() throws IOException {
        EngineConfig config = ServerTestSupport.config().build();
        Path model = SyntheticModels.h2oLightning(context()).writeTempFile();
        service = new EngineService(model.toString(), config);
        resource = new DecisionResource(service);
        Path cutModel = SyntheticModels.h2oLightning(SyntheticModels.Config.qwen2().toBuilder()
                .nCtxTrain(40000).build()).writeTempFile();
        cut = new EngineService(cutModel.toString(), ServerTestSupport.config()
                .memory(MemoryConfig.builder().nCtx(40000).ubatch(8192)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build()).build());
        int width = SyntheticModels.Config.qwen2().nEmbd();
        mediaService = new EngineService(model.toString(), ServerTestSupport.config()
                        .memory(MemoryConfig.builder().nCtx(1024).ubatch(64).parallel(2)
                                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build(),
                com.eonmux.neuralfree.server.ServerOptions.defaults(),
                MediaOptions.of(MmprojTestFiles.llava(width).toString()));
        mediaResource = new DecisionResource(mediaService);
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(png());
    }

    @AfterAll
    static void close() {
        for (EngineService s : List.of(service, cut, mediaService)) {
            if (s != null) {
                s.close();
            }
        }
    }

    private static SyntheticModels.Config context() {
        return SyntheticModels.Config.qwen2().toBuilder().nCtxTrain(1024).build();
    }

    /** A 12x8 RGB PNG the llava projector takes. */
    private static byte[] png() throws IOException {
        java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(12, 8,
                java.awt.image.BufferedImage.TYPE_INT_RGB);
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(image, "png", out);
        return out.toByteArray();
    }

    // ------------------------------------------------------------------ the reads

    @Test
    void theChoiceReadAnswersWithTheMetadataLabels() throws IOException {
        String body = """
                {"state": "the sky is blue",
                 "questions": {"q": {"type": "choice", "instructions": "What color is the sky?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = systemone(body).obj("answers").obj("q");
        String prompt = h2oPrompt("the sky is blue", "What color is the sky?",
                option(" yes", "red", "warm"), option(" no", "blue", "cool"));
        double[] expected = labelProbabilities(service, prompt, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
        assertEquals(expected[0] > expected[1] ? "red" : "blue", answer.str("choice"));
        assertEquals(Math.max(0.0, (Math.max(expected[0], expected[1]) - 0.5) / 0.5), answer.f64("confidence"),
                1e-9);
    }

    @Test
    void objectInstructionsAreDumpedLikePythonAndSpellingsStayText() throws IOException {
        // json.dumps(instructions, indent=1) lays the object out over one-space indents; the special-token
        // spellings of the text enter quoted, so the tokenizer keeps them as text (h2o_quote)
        String body = """
                {"state": "keep <|special|> as text",
                 "questions": {"q": {"type": "choice", "instructions": {"ask": ["what?", "why?"]},
                     "criteria": {"red": "warm", "blue": "cool"}}}}""";
        JsonView answer = systemone(body).obj("answers").obj("q");
        DecisionQuotes quotes = DecisionQuotes.of(service.tokenizer().vocab(),
                com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab.ATTR_CONTROL
                        | com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab.ATTR_USER_DEFINED, 1);
        String prompt = h2oPrompt(quotes.quote("keep <|special|> as text"),
                quotes.quote("{\n \"ask\": [\n  \"what?\",\n  \"why?\"\n ]\n}"),
                option(" yes", "red", "warm"), option(" no", "blue", "cool"));
        double[] expected = labelProbabilities(service, prompt, 1.0f);
        assertEquals(expected[0], answer.obj("probabilities").f64("red"), 1e-5);
        assertEquals(expected[1], answer.obj("probabilities").f64("blue"), 1e-5);
    }

    @Test
    void theNoulQuestionUsesItsAliasDescriptionsAndTemperature() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "noul", "instructions": "Does it hold?",
                     "criteria": {"true": "holds", "false": "not"}}}}""";
        JsonView answer = systemone(body).obj("answers").obj("q");
        // true first, the noul temperature of the checkpoint (0.6); the answer reports P(true)
        String prompt = h2oPrompt("x", "Does it hold?", option(" yes", "true", "holds"),
                option(" no", "false", "not"));
        double[] expected = labelProbabilities(service, prompt, 0.6f);
        assertEquals(expected[0], answer.f64("noul"), 1e-9);
        assertEquals(2, answer.keys().size(), "a noul answer carries only its type and noul");
    }

    @Test
    void aScoreQuestionShowsJsonNullLegendsAndStripsTheLastText() throws IOException {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "score", "instructions": "Rate.",
                     "criteria": [{"level": 1}, "b "]}}}""";
        JsonView answer = systemone(body).obj("answers").obj("q");
        // a level without a text keeps the JSON null, the last option's trailing whitespace is stripped here
        // because the template's trim would take it silently
        String prompt = h2oPrompt("x", "Rate.", option(" yes", "0", "{\"level\": 1}"),
                option(" no", "1", "b"));
        double[] expected = labelProbabilities(service, prompt, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], probabilities.f64(String.valueOf(i)), 1e-5);
        }
        double score = 0.0;
        for (int i = 0; i < expected.length; i++) {
            score += i * expected[i];
        }
        assertEquals(score, answer.f64("score"), 1e-9);
        // the answer echoes the criteria as given: the object stays an object, the trailing space stays
        // (h2o-lightning is not a canonical-legend family, only the prompt render dumps and strips)
        assertEquals(1.0, answer.obj("legend").obj("0").f64("level"), 0.0);
        assertEquals("b ", answer.obj("legend").str("1"));
    }

    @Test
    void aNonBreakingSpaceOnTheLastOptionIsStrippedLikePython() throws IOException {
        // the template only trims ASCII whitespace, so a trailing NBSP only disappears because the server
        // strips it with Python's str.strip() first
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice", "instructions": "Which?",
                     "criteria": {"red": "warm", "blue": "cool\\u00a0"}}}}""";
        JsonView answer = systemone(body).obj("answers").obj("q");
        String prompt = h2oPrompt("x", "Which?", option(" yes", "red", "warm"), option(" no", "blue", "cool"));
        double[] expected = labelProbabilities(service, prompt, 1.0f);
        assertEquals(expected[1], answer.obj("probabilities").f64("blue"), 1e-5);
    }

    @Test
    void aStateBeyondItsTokenLimitKeepsItsFirstAndLastHalves() throws IOException {
        StringBuilder state = new StringBuilder();
        while (state.length() < 40000) {
            state.append("alpha beta ");
        }
        String body = "{\"state\": \"" + state + "\", \"questions\": {\"q\": {\"type\": \"choice\","
                + " \"instructions\": \"Which?\", \"criteria\": {\"red\": \"warm\", \"blue\": \"cool\"}}}}";
        DecisionResource cutResource = new DecisionResource(cut);
        JsonView answer = view(cutResource.systemone(body).getEntity()).obj("answers").obj("q");
        int[] tokens = cut.encode(state.toString(), false, true);
        assertTrue(tokens.length > 32000, "the state must exceed the 32000-token state budget, got "
                + tokens.length);
        StringBuilder prepared = new StringBuilder();
        for (int i = 0; i < 16000; i++) {
            prepared.append(cut.tokenizer().getToken(tokens[i]));
        }
        prepared.append("\n...\n");
        for (int i = tokens.length - 16000; i < tokens.length; i++) {
            prepared.append(cut.tokenizer().getToken(tokens[i]));
        }
        String prompt = h2oPromptOn(cut, prepared.toString(), "Which?", option(" yes", "red", "warm"),
                option(" no", "blue", "cool"));
        double[] expected = labelProbabilities(cut, prompt, 1.0f);
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    @Test
    void anImageReadTakesTheImageTemperature() throws IOException {
        String body = """
                {"state": "the picture",
                 "files": ["%s"],
                 "questions": {"q": {"type": "choice", "instructions": "What is shown?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""".formatted(image);
        JsonView answer = view(mediaResource.systemone(body).getEntity()).obj("answers").obj("q");
        double[] expected = mediaProbabilities("the picture", "What is shown?",
                option(" yes", "red", "warm"), option(" no", "blue", "cool"));
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
    }

    // ------------------------------------------------------------------ the refusals

    @Test
    void theMetadataLabelsBoundTheOptionCount() {
        String body = """
                {"state": "x",
                 "questions": {"q": {"type": "choice", "instructions": "Pick",
                     "criteria": {"red": "a", "blue": "b", "green": "c"}}}}""";
        assertEquals(400, statusOf(() -> resource.systemone(body)));
        assertEquals("questions.q: too many options (3), this model supports at most 2",
                errorOf(() -> resource.systemone(body)));
    }

    @Test
    void labelsThatAreNotSingleDistinctTokensRefuse() throws IOException {
        SyntheticModels.Config c = context();
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = SyntheticModels.h2oLightning(c);
        w.strArray(w.archKey("decision.labels"), " yes", " maybe");
        Path path = w.writeTempFile();
        EngineService model = new EngineService(path.toString(), ServerTestSupport.config().build());
        try {
            DecisionResource bad = new DecisionResource(model);
            IllegalStateException e = org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                    () -> bad.systemone("{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"noul\","
                            + " \"instructions\": \"i\"}}}"));
            assertTrue(e.getMessage().contains("\" maybe\""), "the refusal names the label: " + e.getMessage());
        } finally {
            model.close();
        }
    }

    // ------------------------------------------------------------------ reference computations

    /** The h2o prompt of one question: the checkpoint's template over the rendered input. */
    private static String h2oPrompt(Object state, Object instructions,
                                    Map<String, Object>... options) {
        return h2oPromptOn(service, state, instructions, options);
    }

    private static String h2oPromptOn(EngineService model, Object state, Object instructions,
                                      Map<String, Object>... options) {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", "choice");
        inp.put("state", state);
        inp.put("instructions", instructions);
        inp.put("options", List.of(options));
        inp.put("images", List.of());
        return MiniJinja.render(SyntheticModels.H2O_LIGHTNING_TEMPLATE, inp);
    }

    /** One rendered option: its label, key and description. */
    private static Map<String, Object> option(String label, String key, Object description) {
        Map<String, Object> option = new LinkedHashMap<>();
        option.put("label", label);
        option.put("key", key);
        option.put("description", description);
        return option;
    }

    /** The label read over the model's own output logits at the two {@code decision.labels} tokens. */
    private static double[] labelProbabilities(EngineService model, String prompt, float temperature)
            throws IOException {
        int[] tokens = model.encode(prompt, false, true);
        float[] logits = model.nextLogits(null, tokens);
        float[] scores = new float[2];
        String[] labels = {" yes", " no"};
        for (int i = 0; i < 2; i++) {
            int[] ids = model.encode(labels[i], false, false);
            assertEquals(1, ids.length, "label " + labels[i] + " must be a single token");
            scores[i] = logits[ids[0]];
        }
        return softmax(scores, temperature);
    }

    /** The media read: the rendered prompt, the projector's rows, and the last row's logits. */
    private static double[] mediaProbabilities(Object state, Object instructions,
                                               Map<String, Object>... options) throws IOException {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", "choice");
        inp.put("state", state);
        inp.put("instructions", instructions);
        inp.put("options", List.of(options));
        inp.put("images", List.of(com.eonmux.neuralfree.gguf.chat.ChatMessages.MEDIA_MARKER));
        String rendered = MiniJinja.render(SyntheticModels.H2O_LIGHTNING_TEMPLATE, inp);
        List<ChatMedia.Item> items = mediaService.media()
                .decisionItems(List.of(image), "files");
        com.eonmux.neuralfree.gguf.engine.MediaPrompt prompt = mediaService.media().prompt(rendered, items,
                List.of(), -1, text -> mediaService.encode(text, false, true));
        float[] logits = mediaService.nextLogits(null, prompt);
        float[] scores = new float[2];
        String[] labels = {" yes", " no"};
        for (int i = 0; i < 2; i++) {
            scores[i] = logits[mediaService.encode(labels[i], false, false)[0]];
        }
        return softmax(scores, 0.9f);
    }

    private static JsonView systemone(String body) {
        return view(resource.systemone(body).getEntity());
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
