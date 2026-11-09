package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.mm.ClipDualEncoder;
import com.eonmux.neuralfree.gguf.mm.ClipDualFixtures;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A CLIP dual encoder (a two-tower clip GGUF, vLLM's {@code CLIPEmbeddingModel}) served as the model: text and image
 * embeddings equal to the encoder's own towers, image and text scoring and reranking by cosine similarity with the
 * multimodal document echo, the named refusals of mixed and multi-image inputs, of chat-template rendering, of the
 * generative routes and of LoRA adapters (at load and per request), the metadata routes, and the text-only model's
 * refusal of image inputs.
 */
class ClipDualEncoderResourceTest {

    private static final String TEXT = "A cat  DOG";
    private static final double TOLERANCE = 1e-5;

    private static EngineService clip;
    private static EngineService bert;
    private static PoolingResource pooling;
    private static PoolingResource textOnly;
    private static String image;
    private static String otherImage;
    private static float[] textVector;
    private static float[] imageVector;
    private static float[] otherImageVector;
    private static float[] dogVector;

    @BeforeAll
    static void open() throws IOException {
        Path model = ClipDualFixtures.twoTower();
        ModelSourceOptions source = ModelSourceOptions.builder().tokenizerDir(ClipDualFixtures.tokenizerDir()).build();
        EngineConfig config = ServerTestSupport.config().sourceOptions(source).build();
        clip = new EngineService(model.toString(), config);
        bert = new EngineService(SyntheticModels.bert().writeTempFile().toString(),
                ServerTestSupport.config().build());
        pooling = new PoolingResource(clip);
        textOnly = new PoolingResource(bert);
        byte[] png = MultimodalChatResourceTest.png(12, 8);
        byte[] otherPng = MultimodalChatResourceTest.png(5, 17);
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(png);
        otherImage = "data:image/png;base64," + Base64.getEncoder().encodeToString(otherPng);
        try (ClipDualEncoder reference = ServerTestSupport.dualReference(model.toString(), config)) {
            textVector = reference.embedText(reference.tokenizer().encode(TEXT, true, true));
            dogVector = reference.embedText(reference.tokenizer().encode("a dog", true, true));
            imageVector = reference.embedImage(png);
            otherImageVector = reference.embedImage(otherPng);
        }
    }

    @AfterAll
    static void close() {
        clip.close();
        bert.close();
    }

    @Test
    void embeddingsServeTextThroughTheTextTowerAndImagesThroughTheVisionTower() {
        JsonView body = ok(() -> pooling.embeddings("{\"input\": \"" + TEXT + "\", \"embd_normalize\": -1}"));
        assertVector(textVector, at(body, "data", 0).raw("embedding"));
        assertEquals(5L, usage(body).i64("prompt_tokens"), "start, a, cat, dog, end");

        JsonView images = ok(() -> pooling.embeddings("{\"messages\": [" + imageConversation(image) + ", "
                + imageConversation(otherImage) + "], \"embd_normalize\": -1}"));
        assertEquals(2, size(images, "data"));
        assertVector(imageVector, at(images, "data", 0).raw("embedding"));
        assertVector(otherImageVector, at(images, "data", 1).raw("embedding"));
        assertEquals(0L, usage(images).i64("prompt_tokens"), "images carry no tokens");

        JsonView input = ok(() -> pooling.embeddings("{\"input\": " + imageConversation(image)
                + ", \"embd_normalize\": -1}"));
        assertVector(imageVector, at(input, "data", 0).raw("embedding"));
    }

    @Test
    void poolingEmbedsImagesAndTextsBesideEachOther() {
        JsonView body = ok(() -> pooling.pooling("{\"messages\": [" + imageConversation(image)
                + "], \"embd_normalize\": -1}"));
        assertEquals("pooling", at(body, "data", 0).str("object"));
        assertVector(imageVector, at(body, "data", 0).raw("data"));
        assertTrue(errorOf(() -> pooling.pooling("{\"task\": \"classify\", \"messages\": ["
                + imageConversation(image) + "]}")).contains("classify"));
    }

    @Test
    void scoreAnswersTheCosineOfAnImageAgainstTexts() {
        JsonView body = ok(() -> pooling.score("{\"queries\": " + imageEntry(image) + ", \"documents\": [\""
                + TEXT + "\", \"a dog\"]}"));
        assertEquals(2, size(body, "data"));
        assertEquals(cosine(imageVector, textVector), at(body, "data", 0).f64("score"), TOLERANCE);
        assertEquals(cosine(imageVector, dogVector), at(body, "data", 1).f64("score"), TOLERANCE);

        JsonView pairs = ok(() -> pooling.score("{\"data_1\": [" + imageEntry(image) + ", \"a dog\"], \"data_2\": ["
                + imageEntry(otherImage) + ", {\"content\": [{\"type\": \"text\", \"text\": \"" + TEXT + "\"}]}]}"));
        assertEquals(cosine(imageVector, otherImageVector), at(pairs, "data", 0).f64("score"), TOLERANCE);
        assertEquals(cosine(dogVector, textVector), at(pairs, "data", 1).f64("score"), TOLERANCE);
    }

    @Test
    void rerankOrdersImageDocumentsAndEchoesTheirContent() {
        JsonView body = ok(() -> pooling.rerank("{\"query\": \"" + TEXT + "\", \"documents\": [" + imageEntry(image)
                + ", \"a dog\", " + imageEntry(otherImage) + "], \"return_documents\": true}"));
        double[] expected = {cosine(textVector, imageVector), cosine(textVector, dogVector),
                cosine(textVector, otherImageVector)};
        assertEquals(3, size(body, "results"));
        double previous = Double.POSITIVE_INFINITY;
        for (int r = 0; r < 3; r++) {
            JsonView result = at(body, "results", r);
            int index = (int) result.i64("index");
            double score = result.f64("relevance_score");
            assertEquals(expected[index], score, TOLERANCE);
            assertTrue(score <= previous, "results order by relevance");
            previous = score;
            Map<?, ?> document = (Map<?, ?>) result.raw("document");
            if (index == 1) {
                assertEquals("a dog", document.get("text"));
            } else {
                List<?> parts = (List<?>) document.get("multi_modal");
                assertEquals("image_url", ((Map<?, ?>) parts.get(0)).get("type"));
            }
        }

        JsonView imageQuery = ok(() -> pooling.rerank("{\"query\": " + imageEntry(image) + ", \"documents\": [\""
                + TEXT + "\"]}"));
        assertEquals(cosine(imageVector, textVector), at(imageQuery, "results", 0).f64("relevance_score"),
                TOLERANCE);
    }

    @Test
    void mixedAndMultiImageInputsRefuseByName() {
        String mixed = "{\"content\": [{\"type\": \"text\", \"text\": \"a cat\"}, " + imagePart(image) + "]}";
        assertTrue(errorOf(() -> pooling.score("{\"queries\": " + mixed + ", \"documents\": [\"a\"]}"))
                .contains("text-only or image-only inputs, not both"));
        String twoImages = "{\"content\": [" + imagePart(image) + ", " + imagePart(otherImage) + "]}";
        assertTrue(errorOf(() -> pooling.rerank("{\"query\": \"a\", \"documents\": [" + twoImages + "]}"))
                .contains("at most 1 image"));
        String mixedChat = "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"a cat\"}, "
                + imagePart(image) + "]}]";
        assertTrue(errorOf(() -> pooling.embeddings("{\"messages\": " + mixedChat + "}")).contains("not both"));
        assertTrue(errorOf(() -> pooling.score("{\"queries\": {\"content\": [{\"type\": \"video_url\"}]},"
                + " \"documents\": [\"a\"]}")).contains("video_url"));
    }

    @Test
    void inputsTheTowersCannotTakeRefuseByName() {
        assertTrue(errorOf(() -> pooling.embeddings("{\"messages\": [{\"role\": \"user\", \"content\": \"a cat\"}]}"))
                .contains("no chat template"));
        assertTrue(errorOf(() -> pooling.embeddings("{\"input\": \"a a a a a a a a a a\"}"))
                .contains("clip.text.context_length"));
        assertEquals(400, statusOf(() -> pooling.embeddings("{\"messages\": ["
                + imageConversation("data:image/png;base64,AAAA") + "]}")));
        assertTrue(errorOf(() -> pooling.classify("{\"input\": \"a cat\"}")).contains("RANK"));
        assertTrue(errorOf(() -> pooling.pooling("{\"task\": \"token_embed\", \"input\": \"a cat\"}"))
                .contains("token_embed"));
    }

    @Test
    void generativeRoutesRefuseTheDualEncoderByName() {
        OpenAiResource openai = new OpenAiResource(clip);
        String refusal = errorOf(() -> openai.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}"));
        assertTrue(refusal.contains("CLIP dual encoder"), refusal);
        assertTrue(errorOf(() -> openai.completions("{\"prompt\": \"a cat\", \"max_tokens\": 1}"))
                .contains("CLIP dual encoder"));
    }

    @Test
    void adapterSelectionsRefuseByName() {
        int[] tokens = clip.encode(TEXT, true, true);
        UnsupportedOperationException text = assertThrows(UnsupportedOperationException.class,
                () -> clip.pooling().embed(LoraSelection.of("sql"), tokens));
        assertTrue(text.getMessage().contains("runs no LoRA adapters"), text.getMessage());
        UnsupportedOperationException img = assertThrows(UnsupportedOperationException.class,
                () -> clip.pooling().embedImage(LoraSelection.of("sql"), image, "input"));
        assertTrue(img.getMessage().contains("runs no LoRA adapters"), img.getMessage());
        assertEquals(400, statusOf(() -> pooling.embeddings("{\"input\": \"" + TEXT
                + "\", \"lora\": [{\"id\": 0, \"scale\": 1.0}]}")));
        JsonView none = ok(() -> pooling.embeddings("{\"input\": \"" + TEXT + "\", \"lora\": [],"
                + " \"embd_normalize\": -1}"));
        assertVector(textVector, at(none, "data", 0).raw("embedding"));
    }

    @Test
    void metadataRoutesDescribeTheDualEncoder() {
        OpenAiResource openai = new OpenAiResource(clip);
        LlamaCppResource llama = new LlamaCppResource(clip);
        assertEquals(200, llama.health().getStatus());
        JsonView props = ok(() -> llama.props(null, null));
        assertEquals("clip", props.str("model_arch"));
        JsonView models = ok(() -> openai.models(null));
        assertEquals((long) ClipDualFixtures.CONTEXT, at(models, "data", 0).i64("max_model_len"));
        JsonView tokens = ok(() -> llama.tokenize("{\"content\": \"" + TEXT + "\"}"));
        assertTrue(size(tokens, "tokens") > 0, "the text tower's tokenizer answers /tokenize");
    }

    @Test
    void aTextOnlyModelRefusesImageInputsByName() {
        assertTrue(errorOf(() -> textOnly.embeddings("{\"messages\": [" + imageConversation(image) + "]}"))
                .contains("embeds text only"));
        assertTrue(errorOf(() -> textOnly.score("{\"queries\": " + imageEntry(image) + ", \"documents\": [\"a\"]}"))
                .contains("embeds text only"));
    }

    @Test
    void adaptersBesideTheDualEncoderRefuseByName() throws IOException {
        Path model = ClipDualFixtures.twoTower();
        Path adapter = LoraAdapterFiles.llama(1000, 4, 8.0f);
        ModelSourceOptions source = ModelSourceOptions.builder().tokenizerDir(ClipDualFixtures.tokenizerDir()).build();
        IllegalArgumentException startup = assertThrows(IllegalArgumentException.class, () -> new EngineService(
                model.toString(), ServerTestSupport.config().sourceOptions(source).addLora(adapter.toString(), 1.0f)
                .build()));
        assertTrue(startup.getMessage().contains("--lora"), startup.getMessage());
        IllegalArgumentException modules = assertThrows(IllegalArgumentException.class, () -> new EngineService(
                model.toString(), ServerTestSupport.config().sourceOptions(source).build(), ServerOptions.defaults(),
                MediaOptions.none(), new ServerLoras(List.of(new ServerLoras.Module("a", adapter.toString())),
                false)));
        assertTrue(modules.getMessage().contains("--lora-modules"), modules.getMessage());
    }

    @Test
    void aProjectorBesideTheDualEncoderRefusesByName() throws IOException {
        Path model = ClipDualFixtures.twoTower();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> new EngineService(
                model.toString(), ServerTestSupport.config().build(), ServerOptions.defaults(),
                MediaOptions.of(model.toString())));
        assertTrue(e.getMessage().contains(MediaOptions.MMPROJ_PROPERTY), e.getMessage());
    }

    private static JsonView ok(Supplier<Response> call) {
        Response response;
        try {
            response = call.get();
        } catch (WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
        assertEquals(200, response.getStatus());
        return view(response.getEntity());
    }

    private static JsonView usage(JsonView body) {
        return JsonView.of(body.raw("usage"), "usage");
    }

    private static String imagePart(String url) {
        return "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + url + "\"}}";
    }

    private static String imageEntry(String url) {
        return "{\"content\": [" + imagePart(url) + "]}";
    }

    private static String imageConversation(String url) {
        return "[{\"role\": \"user\", \"content\": [" + imagePart(url) + "]}]";
    }

    private static void assertVector(float[] expected, Object actual) {
        List<?> values = (List<?>) actual;
        assertEquals(expected.length, values.size(), "the vector width");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], ((Number) values.get(i)).doubleValue(), TOLERANCE, "value " + i);
        }
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
