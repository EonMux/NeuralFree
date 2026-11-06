package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.media.MediaConnector;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import com.sun.net.httpserver.HttpServer;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Image input over the real service with a llava projector beside the text model (llama.cpp {@code --mmproj}): chat
 * {@code image_url} parts as data, http and file URLs, the Responses {@code input_image} part, Anthropic image
 * blocks (a repeat reading its image rows from the prefix cache), llama.cpp's {@code /completion}
 * {@code multimodal_data}, media-aware token counts that equal the usage of the generation, the {@code /props}
 * modalities, the video modality of a vision projector, and the named refusals of file URLs without a local
 * media path, disallowed domains and audio input on a vision-only projector.
 */
class MultimodalChatResourceTest {

    /** The llava fixture's rows per image: a 2x2 patch grid, the class token dropped. */
    private static final int IMAGE_ROWS = 4;

    private static EngineService service;
    private static OpenAiResource openai;
    private static Path mediaDir;
    private static HttpServer server;
    private static String imageUrl;
    private static String png64;
    /** The bound meters; micrometer reads them through a weak reference, so the test keeps them reachable. */
    private static volatile EngineMetrics metrics;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int width = SyntheticModels.Config.llama().nEmbd();
        Path mmproj = MmprojTestFiles.llava(width);
        byte[] png = png(12, 8);
        png64 = Base64.getEncoder().encodeToString(png);
        mediaDir = Files.createTempDirectory("nf-media");
        Files.write(mediaDir.resolve("cat.png"), png);
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/cat.png", ex -> {
            ex.sendResponseHeaders(200, png.length);
            try (OutputStream out = ex.getResponseBody()) {
                out.write(png);
            }
        });
        server.start();
        imageUrl = "http://127.0.0.1:" + server.getAddress().getPort() + "/cat.png";
        MediaConnector.Options connector = MediaConnector.Options.builder()
                .allowedDomains(List.of("127.0.0.1")).localRoot(mediaDir).build();
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()).withConnector(connector));
        openai = new OpenAiResource(service);
    }

    @AfterAll
    static void close() throws IOException {
        service.close();
        server.stop(0);
        try (var files = Files.walk(mediaDir)) {
            files.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    static byte[] png(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img.setRGB(x, y, ((x * 29) & 0xFF) << 16 | ((y * 41) & 0xFF) << 8 | 0x40);
            }
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            ImageIO.write(img, "png", out);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return out.toByteArray();
    }

    /** A one-message chat body whose user content is {@code parts}. */
    private static String chat(String parts) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": [" + parts + "]}], \"max_tokens\": 3,"
                + " \"temperature\": 0}";
    }

    private static String text(String t) {
        return "{\"type\": \"text\", \"text\": \"" + t + "\"}";
    }

    private static String image(String url) {
        return "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + url + "\"}}";
    }

    /** The prompt rows the usage of a chat completion of {@code body} reports. */
    private static long promptTokens(String body) {
        Response response = ok(() -> openai.chatCompletions(body));
        return JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens");
    }

    private static long inputTokens(String body) {
        Response response = ok(() -> openai.chatInputTokens(body));
        return view(response.getEntity()).i64("input_tokens");
    }

    // ------------------------------------------------------------------ chat completions

    /** The response of a call that must succeed; a refusal fails the test with its message. */
    private static Response ok(java.util.function.Supplier<Response> call) {
        try {
            Response response = call.get();
            assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
            return response;
        } catch (jakarta.ws.rs.WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
    }

    @Test
    void aDataUrlImageAddsItsRowsToThePromptAndTheCountMatchesTheUsage() {
        String withImage = chat(text("describe ") + "," + image("data:image/png;base64," + png64));
        long prompt = promptTokens(withImage);
        assertEquals(prompt, inputTokens(withImage), "the token count encodes the image like the generation");
        long textOnly = promptTokens(chat(text("describe ")));
        assertTrue(prompt >= textOnly + IMAGE_ROWS - 1, prompt + " rows for " + textOnly + " text tokens");
    }

    @Test
    void httpFileAndDataUrlsOfOneImageGiveOnePrompt() {
        long data = promptTokens(chat(image("data:image/png;base64," + png64)));
        assertEquals(data, promptTokens(chat(image(imageUrl))));
        assertEquals(data, promptTokens(chat(image("file://cat.png"))));
        assertEquals(data, promptTokens(chat(
                "{\"type\": \"image_url\", \"image_url\": \"" + imageUrl + "\"}")));
    }

    @Test
    void twoImagesTakeTwiceTheRows() {
        long one = promptTokens(chat(image(imageUrl)));
        long two = promptTokens(chat(image(imageUrl) + "," + image(imageUrl)));
        assertEquals(one + IMAGE_ROWS, two);
    }

    @Test
    void repeatedImagesShareOneEncoderCacheEntry() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        metrics = new EngineMetrics(new EngineLifecycle() {
            @Override
            EngineService current() {
                return service;
            }
        });
        metrics.bindTo(registry);
        String body = chat(image("data:image/png;base64," + png64));
        assertEquals(200, ok(() -> openai.chatCompletions(body)).getStatus());
        long rows = service.stats().encoderCacheRows();
        assertTrue(rows >= IMAGE_ROWS, "the image's rows hold a cache entry: " + rows);
        long hits = service.stats().encoderCacheHits();
        assertEquals(200, ok(() -> openai.chatCompletions(body)).getStatus());
        assertEquals(hits + 1, service.stats().encoderCacheHits(),
                "the second request hit the entry instead of storing its own");
        assertEquals(rows, service.stats().encoderCacheRows(), "no second copy of the rows was stored");
        assertEquals(1.0, registry.get("neuralfree.encoder.cache.hits").functionCounter().count() - hits);
        assertEquals((double) rows, registry.get("neuralfree.encoder.cache.rows").gauge().value());
    }

    @Test
    void refusedMediaNameTheirPart() throws IOException {
        Path outside = Files.createTempFile("nf-outside", ".png");
        try {
            Files.write(outside, Base64.getDecoder().decode(png64));
            assertTrue(errorOf(() -> openai.chatCompletions(chat(image(outside.toUri().toString()))))
                    .contains("must be inside the allowed local media path"));
        } finally {
            Files.delete(outside);
        }
        assertTrue(errorOf(() -> openai.chatCompletions(chat(image("http://example.com/cat.png"))))
                .contains("messages[0].content[0].image_url.url: the URL must be from one of the allowed domains"));
        assertTrue(errorOf(() -> openai.chatCompletions(chat(image("data:image/png,raw"))))
                .contains("without ';base64'"));
        assertTrue(errorOf(() -> openai.chatCompletions(chat(image("data:image/png;base64,"
                + Base64.getEncoder().encodeToString("not an image".getBytes())))))
                .contains("the image cannot be decoded"));
        assertTrue(errorOf(() -> openai.chatCompletions(chat(image("http://example.com/clip.mp4"))
                .replace("image_url", "video_url")))
                .contains("messages[0].content[0].video_url.url: the URL must be from one of the allowed"
                        + " domains"));
        assertTrue(errorOf(() -> openai.chatCompletions(chat(
                "{\"type\": \"input_audio\", \"input_audio\": {\"data\": \"AAAA\", \"format\": \"wav\"}}")))
                .contains("audio input is not supported - hint: if this is unexpected, you may need to provide"
                        + " the mmproj"));
    }

    // ------------------------------------------------------------------ responses and anthropic

    @Test
    void theResponsesInputImagePartIsAChatImage() {
        ResponsesResource responses = new ResponsesResource(service);
        String body = "{\"input\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_image\", \"image_url\": \""
                + imageUrl + "\"}]}], \"max_output_tokens\": 3, \"temperature\": 0}";
        Response response = ok(() -> responses.create(body));
        JsonView usage = JsonView.of(view(response.getEntity()).raw("usage"), "usage");
        assertEquals(view(responses.inputTokens(body).getEntity()).i64("input_tokens"), usage.i64("input_tokens"));
        assertTrue(errorOf(() -> responses.create("{\"input\": [{\"role\": \"user\", \"content\": [{\"type\":"
                + " \"input_image\", \"file_id\": \"file-1\"}]}]}")).contains("file_id is refused"));
    }

    @Test
    void anthropicImageBlocksCountLikeTheyGenerate() {
        AnthropicResource anthropic = new AnthropicResource(service);
        String content = "[{\"type\": \"image\", \"source\": {\"type\": \"base64\", \"media_type\": \"image/png\","
                + " \"data\": \"" + png64 + "\"}}, {\"type\": \"text\", \"text\": \"what is it\"},"
                + " {\"type\": \"image\", \"source\": {\"type\": \"url\", \"url\": \"" + imageUrl + "\"}}]";
        String body = "{\"model\": \"m\", \"max_tokens\": 3, \"messages\": [{\"role\": \"user\", \"content\": "
                + content + "}]}";
        Response response = ok(() -> anthropic.messages(body));
        JsonView usage = JsonView.of(view(response.getEntity()).raw("usage"), "usage");
        long counted = view(anthropic.countTokens(body).getEntity()).i64("input_tokens");
        assertEquals(counted, usage.i64("input_tokens") + usage.i64("cache_read_input_tokens"));
        JsonView again = JsonView.of(view(ok(() -> anthropic.messages(body)).getEntity()).raw("usage"), "usage");
        assertEquals(counted - 1, again.i64("cache_read_input_tokens"),
                "a repeat reads every row but the last from the prefix cache, both images' rows included");
        assertEquals(1, again.i64("input_tokens"));
        assertTrue(errorOf(() -> anthropic.messages("{\"model\": \"m\", \"max_tokens\": 3, \"messages\": [{\"role\":"
                + " \"user\", \"content\": [{\"type\": \"image\", \"source\": {\"type\": \"file\", \"file_id\":"
                + " \"f\"}}]}]}")).contains("file"));
    }

    // ------------------------------------------------------------------ llama.cpp routes

    @Test
    void completionMultimodalDataFillsTheMtmdMarkers() {
        LlamaCompletionResource completion = new LlamaCompletionResource(service);
        String withImage = "{\"prompt\": {\"prompt_string\": \"look <__media__> now\", \"multimodal_data\": [\""
                + png64 + "\"]}, \"n_predict\": 2, \"temperature\": 0}";
        Response response = ok(() -> completion.completion(withImage));
        long evaluated = view(response.getEntity()).i64("tokens_evaluated");
        Response plain = completion.completion("{\"prompt\": \"look  now\", \"n_predict\": 2, \"temperature\": 0}");
        long textOnly = view(plain.getEntity()).i64("tokens_evaluated");
        assertTrue(evaluated >= textOnly + IMAGE_ROWS - 1, evaluated + " rows for " + textOnly + " text tokens");
        assertTrue(errorOf(() -> completion.completion("{\"prompt\": {\"prompt_string\": \"no marker\","
                + " \"multimodal_data\": [\"" + png64 + "\"]}, \"n_predict\": 1}")).contains("media marker"));
    }

    @Test
    void openAiCompletionsTakePromptStringObjectsWithMultimodalData() {
        String withImage = "{\"prompt\": [{\"prompt_string\": \"look <__media__> now\", \"multimodal_data\": [\""
                + png64 + "\"]}, \"look  now\"], \"max_tokens\": 2, \"temperature\": 0}";
        Response response = ok(() -> openai.completions(withImage));
        long rows = JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens");
        long text = view(ok(() -> new LlamaCppResource(service).tokenize("{\"content\": \"look  now\","
                + " \"add_special\": true}")).getEntity()).i64("count");
        // the image prompt and the text prompt, each counted once
        assertTrue(rows >= 2 * text + IMAGE_ROWS - 1, rows + " rows for two prompts of " + text + " text tokens");
        assertEquals(2, ((List<?>) view(response.getEntity()).raw("choices")).size());
        assertTrue(errorOf(() -> openai.completions("{\"prompt\": {\"prompt_string\": \"x <__media__>\","
                + " \"multimodal_data\": [\"" + png64 + "\"]}, \"truncate_prompt_tokens\": 2}"))
                .contains("truncate_prompt_tokens cannot be combined with multimodal_data"));
        assertTrue(errorOf(() -> openai.completions("{\"prompt\": {\"multimodal_data\": []}}"))
                .contains("prompt is an object without a string 'prompt_string'"));
    }

    @Test
    void tokenizeCountsMediaRowsAsNullTokensAndApplyTemplateRendersTheMarker() {
        LlamaCppResource llama = new LlamaCppResource(service);
        String messages = "\"messages\": [{\"role\": \"user\", \"content\": [" + text("see ") + ","
                + image("data:image/png;base64," + png64) + "]}]";
        JsonView body = view(ok(() -> llama.tokenize("{" + messages + ", \"with_pieces\": true}")).getEntity());
        List<?> tokens = (List<?>) body.raw("tokens");
        long nulls = tokens.stream().filter(t -> ((Map<?, ?>) t).get("id") instanceof Number n && n.intValue()
                == LlamaCppResource.MEDIA_ROW && ((Map<?, ?>) t).get("piece") == null).count();
        assertEquals(IMAGE_ROWS, nulls);
        assertEquals(tokens.size(), body.i64("count"));
        String chatBody = "{" + messages + ", \"max_tokens\": 1, \"temperature\": 0}";
        assertEquals(promptTokens(chatBody), body.i64("count"), "the count is the generation's prompt rows");

        String prompt = view(ok(() -> llama.applyTemplate("{" + messages + "}")).getEntity()).str("prompt");
        assertTrue(prompt.contains("see " + view(llama.props(null, null).getEntity()).str("media_marker")), prompt);
    }

    @Test
    void propsReportTheProjectorsModalities() {
        JsonView props = view(new LlamaCppResource(service).props(null, null).getEntity());
        JsonView modalities = props.obj("modalities");
        assertTrue(modalities.bool("vision"));
        assertFalse(modalities.bool("audio"));
        assertTrue(modalities.bool("video"), "a vision projector takes video frames too");
        assertFalse(props.str("media_marker").isEmpty());
    }
}
