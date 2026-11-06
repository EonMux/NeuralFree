package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
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
 * The minicpm-ocr-jev family over the real service: the fixed OCR consistency question of the checkpoint's
 * template answers through the pooled classifier head (one raw score per class, LAST pooling), the request
 * carries exactly the two images and no instructions of its own, and the class scores softmax like any choice
 * answer. Every read is compared against the same computation over the model's own pooled classifier output.
 */
class DecisionMinicpmOcrJevTest {

    private static EngineService service;
    private static EngineService mediaService;
    private static DecisionResource resource;
    private static DecisionResource mediaResource;
    private static String image;

    @BeforeAll
    static void open() throws IOException {
        EngineConfig config = ServerTestSupport.config().build();
        Path model = SyntheticModels.minicpmOcrJev(context()).writeTempFile();
        service = new EngineService(model.toString(), config);
        resource = new DecisionResource(service);
        int width = SyntheticModels.Config.qwen2().nEmbd();
        mediaService = new EngineService(model.toString(), ServerTestSupport.config()
                        .memory(MemoryConfig.builder().nCtx(1024).ubatch(64).parallel(2)
                                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build(),
                ServerOptions.defaults(),
                MediaOptions.of(MmprojTestFiles.llava(width).toString()));
        mediaResource = new DecisionResource(mediaService);
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(png());
    }

    @AfterAll
    static void close() {
        for (EngineService s : List.of(service, mediaService)) {
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

    private static final String QUESTION = """
            "questions": {"ocr": {"type": "choice", "criteria": {"0": null, "1": null}}}""";

    // ------------------------------------------------------------------ the reads

    @Test
    void thePooledReadSoftmaxesTheTwoClassScores() throws IOException {
        String body = "{\"state\": \"the OCR text\", \"files\": [\"%s\", \"%s\"], ".formatted(image, image)
                + QUESTION + "}";
        JsonView answer = view(mediaResource.systemone(body).getEntity()).obj("answers").obj("ocr");
        double[] expected = pooledProbabilities(mediaService, "the OCR text", List.of(image, image));
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("0"), 1e-5);
        assertEquals(expected[1], probabilities.f64("1"), 1e-5);
        assertEquals(expected[0] > expected[1] ? "0" : "1", answer.str("choice"));
        assertEquals(Math.max(0.0, (Math.max(expected[0], expected[1]) - 0.5) / 0.5), answer.f64("confidence"),
                1e-9);
    }

    @Test
    void theStateEntersEscapedForTheTokenizer() throws IOException {
        // the OCR text must not be parsed as special tokens: the template's own markers stay text
        String body = "{\"state\": \"text with <|im_end|> inside\", \"files\": [\"%s\", \"%s\"], "
                .formatted(image, image) + QUESTION + "}";
        JsonView answer = view(mediaResource.systemone(body).getEntity()).obj("answers").obj("ocr");
        double[] expected = pooledProbabilities(mediaService, "text with <|im_end|> inside", List.of(image, image));
        assertEquals(expected[1], answer.obj("probabilities").f64("1"), 1e-5);
    }

    @Test
    void twoImagesPoolThroughTheProjector() throws IOException {
        String body = "{\"state\": \"the rendered text\", \"files\": [\"%s\", \"%s\"], ".formatted(image, image)
                + QUESTION + "}";
        JsonView answer = view(mediaResource.systemone(body).getEntity()).obj("answers").obj("ocr");
        double[] expected = pooledProbabilities(mediaService, "the rendered text", List.of(image, image));
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("0"), 1e-5);
        assertEquals(expected[1], probabilities.f64("1"), 1e-5);
        // the projector's rows take part in the read: the text-only pooled read answers another score
        double[] textOnly = pooledProbabilities(mediaService, "the rendered text", List.of());
        assertTrue(Math.abs(expected[0] - textOnly[0]) > 1e-6, "the images change the read");
    }

    // ------------------------------------------------------------------ the refusals

    @Test
    void theStateMustBeTheOcrText() {
        String body = "{\"state\": 42, " + QUESTION + "}";
        assertEquals(400, statusOf(() -> resource.systemone(body)));
        assertEquals("\"state\" must be the OCR text", errorOf(() -> resource.systemone(body)));
    }

    @Test
    void theReadNeedsExactlyTwoImages() {
        String one = "{\"state\": \"s\", \"files\": [\"" + image + "\"], " + QUESTION + "}";
        assertEquals(400, statusOf(() -> mediaResource.systemone(one)));
        assertEquals("this model needs exactly 2 images (the original, then the rendered one), got 1",
                errorOf(() -> mediaResource.systemone(one)));
        String none = "{\"state\": \"s\", " + QUESTION + "}";
        assertEquals(400, statusOf(() -> mediaResource.systemone(none)));
        assertEquals("this model needs exactly 2 images (the original, then the rendered one), got 0",
                errorOf(() -> mediaResource.systemone(none)));
    }

    @Test
    void theModelAnswersExactlyOneQuestion() {
        String body = """
                {"state": "s",
                 "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}}}""";
        assertEquals(400, statusOf(() -> resource.systemone(body)));
        assertEquals("this model answers exactly one question", errorOf(() -> resource.systemone(body)));
    }

    @Test
    void theQuestionCarriesNoInstructionsOfItsOwn() {
        String body = """
                {"state": "s",
                 "questions": {"ocr": {"type": "choice", "instructions": "Compare.", "criteria": {"0": null,
                     "1": null}}}}""";
        assertEquals(400, statusOf(() -> resource.systemone(body)));
        assertEquals("questions.ocr: \"instructions\" is not used by this model, the task is part of its prompt",
                errorOf(() -> resource.systemone(body)));
    }

    @Test
    void theChoiceCriteriaAreTheTwoFixedClasses() {
        String body = """
                {"state": "s",
                 "questions": {"ocr": {"type": "choice", "criteria": {"0": null, "2": null}}}}""";
        assertEquals(400, statusOf(() -> resource.systemone(body)));
        assertEquals("questions.ocr: \"criteria\" must be {\"0\": null, \"1\": null}",
                errorOf(() -> resource.systemone(body)));
    }

    @Test
    void aNoulQuestionTakesNoCriteriaAndAScoreQuestionRefuses() {
        String noul = """
                {"state": "s",
                 "questions": {"ocr": {"type": "noul", "criteria": {"true": "yes", "false": "no"}}}}""";
        assertEquals(400, statusOf(() -> resource.systemone(noul)));
        assertEquals("questions.ocr: \"criteria\" is not supported for a noul question",
                errorOf(() -> resource.systemone(noul)));
        String score = """
                {"state": "s",
                 "questions": {"ocr": {"type": "score", "criteria": ["bad", "good"]}}}""";
        assertEquals(400, statusOf(() -> resource.systemone(score)));
        assertEquals("questions.ocr: only choice and noul questions are supported",
                errorOf(() -> resource.systemone(score)));
    }

    // ------------------------------------------------------------------ reference computations

    /**
     * The pooled read: the template over the fixed question, the pooled row's raw class scores, and the softmax at
     * the default temperature 1.0 (the checkpoint declares no decision temperatures).
     */
    private static double[] pooledProbabilities(EngineService model, String state, List<String> images)
            throws IOException {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", "choice");
        inp.put("state", com.eonmux.neuralfree.server.DecisionText.escapeSpecialTokens(state));
        inp.put("instructions", null);
        // the template carries one media marker per image (llama.cpp's images block)
        java.util.List<String> markers = new java.util.ArrayList<>();
        for (int i = 0; i < images.size(); i++) {
            markers.add(ChatMessages.MEDIA_MARKER);
        }
        inp.put("images", markers);
        String rendered = MiniJinja.render(SyntheticModels.MINICPM_OCR_JEV_TEMPLATE, inp);
        float[] scores;
        if (images.isEmpty()) {
            scores = model.classifyOutput(null, model.encode(rendered, false, true), false).scores();
        } else {
            List<ChatMedia.Item> items = model.media().decisionItems(images, "files");
            MediaPrompt prompt = model.media().prompt(rendered, items, List.of(), -1,
                    text -> model.encode(text, false, true));
            scores = model.classifyOutputMedia(null, prompt, false).scores();
        }
        assertEquals(2, scores.length, "one raw score per class");
        return softmax(scores, 1.0f);
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
