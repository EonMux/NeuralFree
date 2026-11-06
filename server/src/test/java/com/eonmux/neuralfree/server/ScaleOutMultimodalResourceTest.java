package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The scale-out multimodal path: an image chat rendered on one server instance and generated on another, the
 * features (hashes, placeholder spans, serialized source) travelling between them, {@code content_parts} expansion,
 * per-token character offsets, and the generate endpoint's {@code output_mode="derender"}.
 */
public class ScaleOutMultimodalResourceTest {

    /** The llava fixture's rows per image: a 2x2 patch grid, the class token dropped. */
    private static final int IMAGE_ROWS = 4;

    /** The render-side server: its own media cache. */
    private static EngineService renderService;
    private static ScaleOutResource renderer;
    /** The generate-side server: a second instance with its own, empty media cache. */
    private static EngineService generateService;
    private static ScaleOutResource generator;
    private static String dataUrl;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int width = SyntheticModels.Config.llama().nEmbd();
        Path mmproj = MmprojTestFiles.llava(width);
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        renderService = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        generateService = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        renderer = new ScaleOutResource(renderService);
        generator = new ScaleOutResource(generateService);
        dataUrl = "data:image/png;base64," + Base64.getEncoder().encodeToString(png(12, 8));
    }

    @AfterAll
    static void close() {
        renderService.close();
        generateService.close();
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

    /** An image chat body over {@code dataUrl} with a text question. */
    private static String imageChat(String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\":"
                + " {\"url\": \"" + dataUrl + "\"}}, {\"type\": \"text\", \"text\": \"what is it\"}]}],"
                + " \"max_tokens\": 4, \"temperature\": 0, \"ignore_eos\": true" + extra + "}";
    }

    private static Response ok(java.util.function.Supplier<Response> call) {
        try {
            Response response = call.get();
            assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
            return response;
        } catch (jakarta.ws.rs.WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
    }

    /** The render answer as the generate body: its tokens, features, offsets and sampling parameters unchanged. */
    private static String generateBodyFrom(JsonView rendered, String requestId) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("request_id", requestId);
        body.put("token_ids", rendered.raw("token_ids"));
        body.put("features", rendered.raw("features"));
        body.put("sampling_params", rendered.raw("sampling_params"));
        body.put("model", rendered.raw("model"));
        body.put("stream", false);
        body.put("return_token_ids", true);
        return ServerJson.write(body);
    }

    @Test
    void anImageRenderedOnOneServerGeneratesOnAnother() {
        JsonView rendered = view(ok(() -> renderer.renderChat(imageChat(""))).getEntity());
        JsonView features = rendered.obj("features");
        assertEquals(1, size(features.obj("mm_hashes"), "image"));
        assertEquals(1, size(features.obj("mm_placeholders"), "image"));
        assertTrue(features.raw("kwargs_data") instanceof Map<?, ?>, "the source travels with the features");
        long rows = ((Number) ((Map<?, ?>) ((List<?>) features.obj("mm_placeholders").raw("image")).get(0))
                .get("length")).longValue();
        assertEquals(IMAGE_ROWS, rows, "the fixture encodes one image to a 2x2 grid");

        generateService.resetMediaCache();
        long hitsBefore = generateService.mediaCacheHits();
        long bytesBefore = generateService.mediaCacheUsageBytes();
        JsonView generated = view(ok(() -> generator.generate(generateBodyFrom(rendered, "cross"))).getEntity());
        assertEquals("tokens", generated.str("output_mode"));
        assertTrue(size(at(generated, "choices", 0), "token_ids") > 0);
        assertEquals(size(rendered, "token_ids"),
                ((List<?>) generated.raw("prompt_token_ids")).size(),
                "the generate server reproduces the rendered prompt exactly");
        assertEquals(rendered.raw("token_ids"), generated.raw("prompt_token_ids"));
        assertEquals(features.raw("mm_placeholders"), generated.raw("mm_placeholders"));
        assertEquals((long) size(rendered, "token_ids"), generated.obj("usage").i64("prompt_tokens"),
                "the expanded prompt, media rows included, is the prompt token count");
        assertEquals(hitsBefore, generateService.mediaCacheHits(),
                "the second instance's cache is empty, so it encodes the serialized source");
        assertTrue(generateService.mediaCacheUsageBytes() > bytesBefore,
                "the encoded image is now in the second instance's cache");
        ok(() -> generator.generate(generateBodyFrom(rendered, "cross-again")));
        assertEquals(hitsBefore + 1, generateService.mediaCacheHits(),
                "a repeat reads the encoding from the cache keyed by the hash");
    }

    /**
     * The Anthropic messages render encodes an image block through the projector exactly as the chat render does,
     * so its tokens and features drive a generate server on another instance.
     */
    @Test
    void anAnthropicMessageRendersItsImageAndGeneratesOnAnotherServer() {
        String body = "{\"model\": \"test\", \"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\": ["
                + "{\"type\": \"image\", \"source\": {\"type\": \"base64\", \"media_type\": \"image/png\","
                + " \"data\": \"" + Base64.getEncoder().encodeToString(png(12, 8)) + "\"}},"
                + "{\"type\": \"text\", \"text\": \"what is it\"}]}]}";
        JsonView rendered = view(ok(() -> renderer.renderMessages(body)).getEntity());
        JsonView features = rendered.obj("features");
        assertNotNull(features, "an image block renders through the projector into features");
        assertEquals(1, size(features.obj("mm_hashes"), "image"));
        assertEquals(1, size(features.obj("mm_placeholders"), "image"));
        long rows = ((Number) ((Map<?, ?>) ((List<?>) features.obj("mm_placeholders").raw("image")).get(0))
                .get("length")).longValue();
        assertEquals(IMAGE_ROWS, rows);
        assertTrue(size(rendered, "token_ids") > 0);

        JsonView generated = view(ok(() -> generator.generate(generateBodyFrom(rendered, "anthropic"))).getEntity());
        assertEquals(rendered.raw("token_ids"), generated.raw("prompt_token_ids"),
                "the generate server reproduces the rendered messages prompt exactly");
        assertEquals(features.raw("mm_placeholders"), generated.raw("mm_placeholders"));
    }

    @Test
    void theSameInstanceServesTheItemFromItsEncodedMediaCache() {
        JsonView rendered = view(ok(() -> renderer.renderChat(imageChat(""))).getEntity());
        long hits = renderService.mediaCacheHits();
        ok(() -> renderer.generate(generateBodyFrom(rendered, "same-instance")));
        assertEquals(hits + 1, renderService.mediaCacheHits(),
                "the hash names the encoded image the render step already cached");
    }

    @Test
    void contentPartsExpandACollapsedPromptLikeTheRenderStep() {
        JsonView rendered = view(ok(() -> renderer.renderChat(imageChat(""))).getEntity());
        List<Object> collapsed = collapsed(rendered);

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("token_ids", collapsed);
        body.put("content_parts", List.of(Map.of("type", "image_url", "url", dataUrl)));
        body.put("sampling_params", rendered.raw("sampling_params"));
        body.put("return_token_ids", true);
        JsonView generated = view(ok(() -> generator.generate(ServerJson.write(body))).getEntity());
        assertEquals(rendered.raw("token_ids"), generated.raw("prompt_token_ids"),
                "the parts expand into the prompt the render step produced");
        assertEquals(rendered.obj("features").raw("mm_placeholders"), generated.raw("mm_placeholders"));
        assertEquals((long) size(rendered, "token_ids"), generated.obj("usage").i64("prompt_tokens"));
    }

    @Test
    void aContentPartMayNameACachedUuidWithoutItsData() {
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"uuid\":"
                + " \"cat-1\", \"image_url\": {\"url\": \"" + dataUrl + "\"}}, {\"type\": \"text\", \"text\":"
                + " \"what is it\"}]}], \"max_tokens\": 2, \"temperature\": 0}";
        JsonView rendered = view(ok(() -> renderer.renderChat(body)).getEntity());
        Map<String, Object> generate = new LinkedHashMap<>();
        generate.put("token_ids", collapsed(rendered));
        generate.put("content_parts", List.of(Map.of("type", "image_url", "uuid", "cat-1")));
        generate.put("sampling_params", rendered.raw("sampling_params"));
        generate.put("return_token_ids", true);
        JsonView generated = view(ok(() -> renderer.generate(ServerJson.write(generate))).getEntity());
        assertEquals(rendered.raw("token_ids"), generated.raw("prompt_token_ids"));
        Map<String, Object> unknown = new LinkedHashMap<>(generate);
        unknown.put("content_parts", List.of(Map.of("type", "image_url", "uuid", "no-such-cat")));
        assertTrue(errorOf(() -> renderer.generate(ServerJson.write(unknown))).contains("not in the media cache"));
    }

    @Test
    void generateRefusesMalformedOrIncompleteFeaturesByName() {
        JsonView rendered = view(ok(() -> renderer.renderChat(imageChat(""))).getEntity());
        Map<String, Object> body = bodyMap(rendered);
        body.put("content_parts", List.of(Map.of("type", "image_url", "url", dataUrl)));
        assertTrue(errorOf(() -> generator.generate(ServerJson.write(body))).contains("mutually exclusive"));

        Map<String, Object> noPayload = bodyMap(rendered);
        ((Map<String, Object>) noPayload.get("features")).remove("kwargs_data");
        generateService.resetMediaCache();
        assertTrue(errorOf(() -> generator.generate(ServerJson.write(noPayload))).contains("not in the media cache"));

        Map<String, Object> badKey = bodyMap(rendered);
        Map<String, Object> badFeatures = (Map<String, Object>) badKey.get("features");
        badFeatures.put("mm_hashes", Map.of("image", List.of("not a key")));
        assertTrue(errorOf(() -> generator.generate(ServerJson.write(badKey))).contains("media key handle"));

        Map<String, Object> unknown = bodyMap(rendered);
        ((Map<String, Object>) unknown.get("features")).put("mm_metadata", Map.of());
        assertTrue(errorOf(() -> generator.generate(ServerJson.write(unknown))).contains("mm_metadata"));

        Map<String, Object> overlap = bodyMap(rendered);
        Map<String, Object> overlapFeatures = (Map<String, Object>) overlap.get("features");
        overlapFeatures.put("mm_placeholders", Map.of("image", List.of(Map.of("offset", 0, "length", 1),
                Map.of("offset", 0, "length", 1))));
        overlapFeatures.put("mm_hashes", Map.of("image", List.of(
                ((List<?>) ((Map<?, ?>) ((Map<?, ?>) rendered.raw("features")).get("mm_hashes")).get("image"))
                        .get(0),
                ((List<?>) ((Map<?, ?>) ((Map<?, ?>) rendered.raw("features")).get("mm_hashes")).get("image"))
                        .get(0))));
        ((Map<String, Object>) ((Map<String, Object>) overlap.get("features"))).remove("kwargs_data");
        assertTrue(errorOf(() -> generator.generate(ServerJson.write(overlap))).contains("overlap"));
    }

    @Test
    void embeddingPartsTravelAsSerializedRows() {
        int width = renderService.embeddingLength();
        StringBuilder row = new StringBuilder("[");
        for (int i = 0; i < width; i++) {
            row.append(i == 0 ? "0" : ", 0");
        }
        row.append(']');
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"look\"},"
                + " {\"type\": \"image_embeds\", \"image_embeds\": [" + row + ", " + row + "]}]}],"
                + " \"max_tokens\": 2, \"temperature\": 0}";
        JsonView rendered = view(ok(() -> renderer.renderChat(body)).getEntity());
        JsonView features = rendered.obj("features");
        assertEquals(1, size(features.obj("mm_hashes"), "embeddings"));
        assertEquals(2L, ((Number) ((Map<?, ?>) ((List<?>) features.obj("mm_placeholders").raw("embeddings"))
                .get(0)).get("length")).longValue());
        JsonView generated = view(ok(() -> generator.generate(generateBodyFrom(rendered, "embeds"))).getEntity());
        assertEquals(rendered.raw("token_ids"), generated.raw("prompt_token_ids"),
                "the serialized rows splice back where the part sat");
    }

    @Test
    void completionsMultimodalDataRendersFeatures() {
        String png64 = dataUrl.substring(dataUrl.indexOf(',') + 1);
        Object parsed = Json.parse(String.valueOf(ok(() -> renderer.renderCompletions("{\"prompt\":"
                + " {\"prompt_string\": \"look <__media__> now\", \"multimodal_data\": [\"" + png64 + "\"]},"
                + " \"max_tokens\": 2, \"temperature\": 0}")).getEntity()), Json.Limits.UNTRUSTED);
        JsonView request = JsonView.of(((List<?>) parsed).get(0), "requests[0]");
        JsonView features = request.obj("features");
        assertEquals(1, size(features.obj("mm_hashes"), "image"));
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("token_ids", request.raw("token_ids"));
        body.put("features", request.raw("features"));
        body.put("sampling_params", request.raw("sampling_params"));
        body.put("return_token_ids", true);
        JsonView generated = view(ok(() -> generator.generate(ServerJson.write(body))).getEntity());
        assertEquals(request.raw("token_ids"), generated.raw("prompt_token_ids"));
    }

    @Test
    void renderReportsPerTokenCharacterOffsets() {
        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello there\"}],"
                + " \"return_token_offsets\": true, \"max_tokens\": 1}";
        JsonView rendered = view(ok(() -> renderer.renderChat(chat)).getEntity());
        List<?> offsets = (List<?>) rendered.raw("token_offsets");
        assertNotNull(offsets);
        assertEquals(size(rendered, "token_ids"), offsets.size());
        String text = renderService.formatChat(
                OpenAiRequests.chat(JsonView.of(Json.parse(chat, Json.Limits.UNTRUSTED), "body"),
                        renderService.options()), false);
        int previous = 0;
        StringBuilder covered = new StringBuilder();
        for (Object raw : offsets) {
            List<?> span = (List<?>) raw;
            int start = ((Number) span.get(0)).intValue();
            int end = ((Number) span.get(1)).intValue();
            assertTrue(0 <= start && start <= end && end <= text.length(), start + ".." + end + " of " + text.length());
            assertTrue(start >= previous || (start == 0 && end == 0), "offsets advance");
            if (start != 0 || end != 0) {
                previous = end;
                covered.append(text, start, end);
            }
        }
        assertTrue(covered.toString().contains("hello"), "the token spans cover the prompt text: " + covered);
        JsonView plain = view(ok(() -> renderer.renderChat(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")).getEntity());
        assertNull(plain.raw("token_offsets"), "offsets are null unless return_token_offsets asks for them");
    }

    @Test
    void completionRenderOffsetsFollowTruncation() {
        String prompt = "The quick brown fox jumps over the lazy dog.";
        Object parsed = Json.parse(String.valueOf(ok(() -> renderer.renderCompletions("{\"prompt\": \""
                + prompt + "\", \"return_token_offsets\": true, \"truncate_prompt_tokens\": 4,"
                + " \"truncation_side\": \"left\"}")).getEntity()), Json.Limits.UNTRUSTED);
        List<?> requests = (List<?>) parsed;
        assertEquals(1, requests.size());
        JsonView request = JsonView.of(requests.get(0), "requests[0]");
        List<?> offsets = (List<?>) request.raw("token_offsets");
        assertEquals(size(request, "token_ids"), offsets.size());
        assertEquals(4, offsets.size());
        assertEquals(prompt.length(), ((Number) ((List<?>) offsets.get(3)).get(1)).intValue(),
                "left truncation keeps the offsets of the prompt's end");
        assertTrue(((Number) ((List<?>) offsets.get(0)).get(0)).intValue() > 0);
    }

    @Test
    void generateDerenderAnswersTheParsedMessage() {
        int[] prompt = renderService.encode("hello", true, true);
        String sampling = "\"max_tokens\": 4, \"temperature\": 0, \"ignore_eos\": true";
        String text = "{\"token_ids\": " + json(prompt) + ", \"sampling_params\": {" + sampling + "},"
                + " \"output_mode\": \"text\"}";
        JsonView textAnswer = view(ok(() -> generator.generate(text)).getEntity());
        String derender = text.replace("\"output_mode\": \"text\"", "\"output_mode\": \"derender\"");
        JsonView answer = view(ok(() -> generator.generate(derender)).getEntity());
        assertEquals("derender", answer.str("output_mode"));
        JsonView choice = at(answer, "choices", 0);
        assertEquals(at(textAnswer, "choices", 0).str("text"), choice.obj("message").str("content"));
        assertEquals("assistant", choice.obj("message").str("role"));
        assertNotNull(choice.raw("token_ids"), "without a parse context the ids stay visible");
        assertEquals(400, statusOf(() -> generator.generate(derender.replace("\"output_mode\": \"derender\"",
                "\"output_mode\": \"derender\", \"stream\": true"))));

        Map<String, Object> hidden = new LinkedHashMap<>();
        hidden.put("token_ids", ScaleOutRequests.ints(prompt));
        hidden.put("sampling_params", Json.parse("{" + sampling + "}", Json.Limits.UNTRUSTED));
        hidden.put("output_mode", "derender");
        hidden.put("parse_context", Map.of("include_reasoning", false));
        JsonView hiddenAnswer = view(ok(() -> generator.generate(ServerJson.write(hidden))).getEntity());
        assertNull(at(hiddenAnswer, "choices", 0).raw("token_ids"),
                "a hidden reasoning keeps the token ids out of the answer");
    }

    @Test
    void generateAcceptsTheRenderAnswersTokenOffsets() {
        JsonView rendered = view(ok(() -> renderer.renderChat("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"hello there\"}], \"return_token_offsets\": true, \"max_tokens\": 2, \"temperature\": 0}"))
                .getEntity());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("token_ids", rendered.raw("token_ids"));
        body.put("token_offsets", rendered.raw("token_offsets"));
        body.put("sampling_params", rendered.raw("sampling_params"));
        assertEquals(200, ok(() -> generator.generate(ServerJson.write(body))).getStatus());
        body.put("token_offsets", List.of(List.of(0, 0)));
        assertTrue(errorOf(() -> generator.generate(ServerJson.write(body))).contains("token_offsets"));
    }

    // ------------------------------------------------------------------ helpers

    /** The generate body of a render answer as a mutable map. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> bodyMap(JsonView rendered) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("token_ids", rendered.raw("token_ids"));
        body.put("features", new LinkedHashMap<>((Map<String, Object>) rendered.raw("features")));
        body.put("sampling_params", rendered.raw("sampling_params"));
        body.put("return_token_ids", true);
        return body;
    }

    /** The render answer's tokens with every media span replaced by one {@code -1} placeholder. */
    private static List<Object> collapsed(JsonView rendered) {
        List<?> expanded = (List<?>) rendered.raw("token_ids");
        List<Object> collapsed = new ArrayList<>();
        int cursor = 0;
        for (Object raw : (List<?>) rendered.obj("features").obj("mm_placeholders").raw("image")) {
            Map<?, ?> range = (Map<?, ?>) raw;
            int offset = ((Number) range.get("offset")).intValue();
            int length = ((Number) range.get("length")).intValue();
            for (int i = cursor; i < offset; i++) {
                collapsed.add(expanded.get(i));
            }
            collapsed.add(-1);
            cursor = offset + length;
        }
        for (int i = cursor; i < expanded.size(); i++) {
            collapsed.add(expanded.get(i));
        }
        return collapsed;
    }

    private static String json(int[] values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            out.append(i == 0 ? "" : ", ").append(values[i]);
        }
        return out.append(']').toString();
    }
}
