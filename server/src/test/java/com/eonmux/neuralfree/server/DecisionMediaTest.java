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
 * Decision media over the real service with a llava projector beside the text model (llama.cpp {@code --mmproj}):
 * the {@code files} data URLs of an openjev prompt splice the projector's rows where the template's media markers
 * stand, and the answer is the same computation over the model's own media logits; a family without an image slot
 * refuses media with llama.cpp's 501.
 */
class DecisionMediaTest {

    /** The openjev template with the media markers at the start, as the checkpoints' templates have them. */
    private static final String MEDIA_TEMPLATE =
            "{% for image in images %}{{ image }}{% endfor %}{{ type }}|{{ instructions }}|"
                    + "{% for o in options %}{{ o.label }}{{ o.key }}:{{ o.description }};{% endfor %}"
                    + "|{{ state }}|Answer:";

    private static EngineService service;
    private static EngineService levService;
    private static DecisionResource resource;
    private static DecisionResource levResource;
    private static String image;

    @BeforeAll
    static void open() throws IOException {
        int width = SyntheticModels.Config.qwen2().nEmbd();
        Path mmproj = MmprojTestFiles.llava(width);
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        Path model = SyntheticModels.decisionLabels("openjev", context(), MEDIA_TEMPLATE).writeTempFile();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        resource = new DecisionResource(service);
        Path levModel = SyntheticModels.decisionLabels("lev", context(), MEDIA_TEMPLATE).writeTempFile();
        levService = new EngineService(levModel.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        levResource = new DecisionResource(levService);
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(png());
    }

    @AfterAll
    static void close() {
        if (service != null) {
            service.close();
        }
        if (levService != null) {
            levService.close();
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

    @Test
    void filesSpliceTheProjectorsRowsIntoTheDecisionPrompt() throws IOException {
        String body = """
                {"state": "the picture",
                 "files": ["%s"],
                 "questions": {"q": {"type": "choice", "instructions": "What is shown?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""".formatted(image);
        JsonView answer = view(resource.systemone(body).getEntity()).obj("answers").obj("q");
        double[] expected = mediaProbabilities("the picture", "What is shown?",
                List.of(option("A", "red", "warm"), option("B", "blue", "cool")));
        JsonView probabilities = answer.obj("probabilities");
        assertEquals(expected[0], probabilities.f64("red"), 1e-5);
        assertEquals(expected[1], probabilities.f64("blue"), 1e-5);
        // the image changes the read: the text-only prompt answers differently
        double[] textOnly = textProbabilities("the picture", "What is shown?",
                List.of(option("A", "red", "warm"), option("B", "blue", "cool")));
        assertTrue(Math.abs(expected[0] - textOnly[0]) > 1e-6, "the projector's rows take part in the read");
    }

    @Test
    void aMessageImagePartBecomesThePromptsMedia() {
        String state = """
                [{"role": "user", "content": [{"type": "text", "text": "the picture"}]}]""";
        String question = """
                "questions": {"q": {"type": "choice", "instructions": "What is shown?",
                     "criteria": {"red": "warm", "blue": "cool"}}}""";
        String fromFiles = "{\"state\": " + state + ", \"files\": [\"" + image + "\"], " + question + "}";
        String fromPart = """
                {"state": [{"role": "user", "content": [
                     {"type": "text", "text": "the picture"},
                     {"type": "image_url", "image_url": {"url": "%s"}}]}],
                 """.formatted(image) + question + "}";
        JsonView files = view(resource.systemone(fromFiles).getEntity()).obj("answers").obj("q");
        JsonView part = view(resource.systemone(fromPart).getEntity()).obj("answers").obj("q");
        assertEquals(files.obj("probabilities").f64("red"), part.obj("probabilities").f64("red"), 1e-9);
        assertEquals(files.obj("probabilities").f64("blue"), part.obj("probabilities").f64("blue"), 1e-9);
    }

    @Test
    void aFamilyWithoutAnImageSlotRefusesMedia() {
        String body = """
                {"state": "x", "files": ["%s"],
                 "questions": {"q": {"type": "choice", "instructions": "Which?",
                     "criteria": {"red": "warm", "blue": "cool"}}}}""".formatted(image);
        assertEquals(501, statusOf(() -> levResource.systemone(body)));
        assertEquals("This server does not support image input for decisions. For a model that supports it, start"
                + " it with `--mmproj`", errorOf(() -> levResource.systemone(body)));
    }

    // ------------------------------------------------------------------ reference computations

    /** The media read: the rendered prompt with one marker, the projector's rows, and the answer-code logits. */
    private static double[] mediaProbabilities(Object state, String instructions,
                                               List<Map<String, Object>> options) throws IOException {
        return logitsProbabilities(service, state, instructions, options, 1);
    }

    /** The text-only read: the same prompt without media. */
    private static double[] textProbabilities(Object state, String instructions,
                                              List<Map<String, Object>> options) throws IOException {
        return logitsProbabilities(service, state, instructions, options, 0);
    }

    /**
     * The prompt rendered with {@code nImages} media markers, spliced through the projector, and the last row's
     * logits at the two answer codes.
     */
    private static double[] logitsProbabilities(EngineService model, Object state, String instructions,
                                                List<Map<String, Object>> options, int nImages) throws IOException {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", "choice");
        inp.put("instructions", instructions);
        inp.put("state", state);
        inp.put("options", options);
        List<String> images = new java.util.ArrayList<>();
        for (int i = 0; i < nImages; i++) {
            images.add(ChatMessages.MEDIA_MARKER);
        }
        inp.put("images", images);
        String rendered = MiniJinja.render(MEDIA_TEMPLATE, inp);
        float[] logits;
        if (nImages > 0) {
            List<ChatMedia.Item> items = model.media().decisionItems(List.of(image), "files");
            MediaPrompt prompt = model.media().prompt(rendered, items, List.of(), -1,
                    text -> model.encode(text, false, true));
            logits = model.pooling().nextLogits(null, prompt);
        } else {
            logits = model.pooling().nextLogits(null, model.encode(rendered, false, true));
        }
        float[] scores = new float[2];
        String[] labels = {"A", "B"};
        for (int i = 0; i < scores.length; i++) {
            int[] ids = model.encode(labels[i], false, false);
            scores[i] = logits[ids[0]];
        }
        return softmax(scores, 1.0f);
    }

    /** One rendered option. */
    private static Map<String, Object> option(String label, String key, Object description) {
        Map<String, Object> option = new LinkedHashMap<>();
        option.put("label", label);
        option.put("key", key);
        option.put("description", description);
        return option;
    }

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
