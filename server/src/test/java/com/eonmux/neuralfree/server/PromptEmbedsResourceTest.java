package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Precomputed embedding inputs over the real service (vLLM prompt_embeds parity): completions run from
 * {@code prompt_embeds} rows — the token table's own rows answer exactly like the text prompt they encode —
 * chat runs from {@code image_embeds}/{@code audio_embeds} content parts spliced where the parts sit (and counts
 * their rows in {@code input_tokens}), a repeated embedding prompt reads its rows back from the prefix cache
 * ({@code prompt_tokens_details.cached_tokens}) while an embedding part keys as one item the cache never splits, and
 * every malformed input refuses with its field named: no prompt at all, a row off the model's embedding width,
 * ragged or non-finite rows, echo with embeds, or a base64 part payload.
 */
class PromptEmbedsResourceTest {

    private static EngineService service;
    private static OpenAiResource resource;
    private static Path modelPath;
    private static int width;

    @BeforeAll
    static void open() throws IOException {
        modelPath = SyntheticModels.llama().writeTempFile();
        EngineConfig config = ServerTestSupport.pagedConfig(64, 2);
        service = new EngineService(modelPath.toString(), config);
        resource = new OpenAiResource(service);
        width = service.embeddingLength();
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** The token table's rows of the first {@code count} tokens {@code text} encodes as a completion prompt. */
    private static float[][] rowsOf(String text, int count) {
        int[] tokens = service.encode(text, true, true);
        return tableRows(Arrays.copyOf(tokens, Math.min(count, tokens.length)));
    }

    /** The token table's rows of {@code tokens}, read from the served GGUF file. */
    private static float[][] tableRows(int[] tokens) {
        try (GGUFFileParser parser = new GGUFFileParser(modelPath.toString())) {
            GGUFTensorInfo table = parser.getTensor("token_embd.weight");
            float[] flat = parser.readTensorDataAsFloat(table);
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = Arrays.copyOfRange(flat, tokens[i] * width, (tokens[i] + 1) * width);
            }
            return rows;
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /** The rows as a JSON array of arrays. */
    private static String json(float[][] rows) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < rows.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append('[');
            for (int j = 0; j < rows[i].length; j++) {
                if (j > 0) {
                    sb.append(',');
                }
                sb.append(rows[i][j]);
            }
            sb.append(']');
        }
        return sb.append(']').toString();
    }

    private static JsonView firstChoice(JsonView body) {
        return JsonView.of(((List<?>) body.raw("choices")).get(0), "choices[0]");
    }

    @Test
    void aCompletionFromEmbeddingRowsAnswersLikeItsTextPrompt() {
        Response response = resource.completions(
                "{\"prompt_embeds\": " + json(rowsOf("hello", Integer.MAX_VALUE)) + ", \"max_tokens\": 4,"
                        + " \"temperature\": 0}");
        assertEquals(200, response.getStatus());
        JsonView embedsBody = view(response.getEntity());
        String fromEmbeds = firstChoice(embedsBody).str("text");
        long rows = JsonView.of(embedsBody.raw("usage"), "usage").i64("prompt_tokens");

        Response twin = resource.completions("{\"prompt\": \"hello\", \"max_tokens\": 4, \"temperature\": 0}");
        assertEquals(firstChoice(view(twin.getEntity())).str("text"), fromEmbeds,
                "the rows of the prompt's tokens decode like the text prompt itself");
        assertEquals(rows, JsonView.of(view(twin.getEntity()).raw("usage"), "usage").i64("prompt_tokens"),
                "the usage counts the rows as the prompt tokens");
    }

    @Test
    void aRepeatedEmbeddingPromptReadsItsRowsFromThePrefixCache() {
        String completion = "{\"prompt_embeds\": " + json(rowsOf("rows read back", Integer.MAX_VALUE))
                + ", \"max_tokens\": 3, \"temperature\": 0}";
        JsonView first = view(resource.completions(completion).getEntity());
        JsonView again = view(resource.completions(completion).getEntity());
        JsonView usage = JsonView.of(again.raw("usage"), "usage");
        assertEquals(usage.i64("prompt_tokens") - 1, usage.obj("prompt_tokens_details").i64("cached_tokens"),
                "the repeat reuses every row but the last");
        assertEquals(firstChoice(first).str("text"), firstChoice(again).str("text"),
                "the cached rows answer as the computed ones");

        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": ["
                + "{\"type\": \"text\", \"text\": \"see \"},"
                + "{\"type\": \"image_embeds\", \"image_embeds\": " + json(rowsOf("pic", 3)) + "},"
                + "{\"type\": \"text\", \"text\": \" it\"}]}], \"max_tokens\": 3, \"temperature\": 0}";
        JsonView chatFirst = view(resource.chatCompletions(chat).getEntity());
        JsonView chatAgain = view(resource.chatCompletions(chat).getEntity());
        JsonView chatUsage = JsonView.of(chatAgain.raw("usage"), "usage");
        assertEquals(chatUsage.i64("prompt_tokens") - 1,
                chatUsage.obj("prompt_tokens_details").i64("cached_tokens"),
                "a repeated chat reuses its text and its embedding part");
        assertEquals(JsonView.of(firstChoice(chatFirst).raw("message"), "message").str("content"),
                JsonView.of(firstChoice(chatAgain).raw("message"), "message").str("content"));
    }

    /** A chat of {@code "see "}, an {@code image_embeds} part of {@code rows} and {@code " it"}. */
    private static String embedsChat(float[][] rows) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": ["
                + "{\"type\": \"text\", \"text\": \"see \"},"
                + "{\"type\": \"image_embeds\", \"image_embeds\": " + json(rows) + "},"
                + "{\"type\": \"text\", \"text\": \" it\"}]}], \"max_tokens\": 2, \"temperature\": 0}";
    }

    private static long cachedTokens(String chat) {
        JsonView usage = JsonView.of(view(resource.chatCompletions(chat).getEntity()).raw("usage"), "usage");
        return usage.obj("prompt_tokens_details").i64("cached_tokens");
    }

    /** {@code rows} with row {@code r} shifted, the other rows equal. */
    private static float[][] shifted(float[][] rows, int r) {
        float[][] out = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            out[i] = rows[i].clone();
        }
        out[r][0] += 0.5f;
        return out;
    }

    @Test
    void anEmbeddingPartIsOneItemThePrefixCacheNeverSplits() {
        float[][] rows = rowsOf("one item of rows", 3);
        assertEquals(3, rows.length);
        resource.chatCompletions(embedsChat(rows));
        long lastRowDiffers = cachedTokens(embedsChat(shifted(rows, 2)));
        long firstRowDiffers = cachedTokens(embedsChat(shifted(rows, 0)));
        assertTrue(firstRowDiffers > 0, "the text before the part is reused");
        assertEquals(firstRowDiffers, lastRowDiffers,
                "rows equal up to the last one reuse nothing of the part: it keys as one item, as vLLM's");
    }

    @Test
    void aFlatRowIsAPromptOfOneRow() {
        float[] row = rowsOf("hi", 1)[0];
        StringBuilder sb = new StringBuilder("[");
        for (int j = 0; j < row.length; j++) {
            if (j > 0) {
                sb.append(',');
            }
            sb.append(row[j]);
        }
        Response response = resource.completions("{\"prompt_embeds\": " + sb.append(']').toString()
                + ", \"max_tokens\": 2, \"temperature\": 0}");
        assertEquals(200, response.getStatus());
        assertEquals(1L, JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens"));
    }

    @Test
    void aThreeDeepPromptEmbedsIsOnePromptPerEntry() {
        float[][] rows = rowsOf("hi", Integer.MAX_VALUE);
        // one prompt per entry: each entry is a list of rows, like vLLM's prompt_embeds list[bytes]
        Response response = resource.completions("{\"prompt_embeds\": ["
                + json(new float[][] {rows[0]}) + ", " + json(new float[][] {rows[1]})
                + "], \"max_tokens\": 1, \"temperature\": 0}");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals(2, ((List<?>) body.raw("choices")).size(),
                "two prompts, one choice each over n = 1");
        assertEquals(2L, JsonView.of(body.raw("usage"), "usage").i64("prompt_tokens"));
    }

    @Test
    void bothPromptAndPromptEmbedsRunAsSeparatePrompts() {
        Response response = resource.completions("{\"prompt\": \"hi\", \"n\": 1, \"prompt_embeds\": "
                + json(rowsOf("hi", 1)) + ", \"max_tokens\": 1, \"temperature\": 0}");
        assertEquals(200, response.getStatus());
        assertEquals(2, ((List<?>) view(response.getEntity()).raw("choices")).size(),
                "vLLM runs the embeds prompt and the text prompt side by side");
    }

    @Test
    void chatRunsFromEmbeddingPartsSplicedWhereTheySit() {
        float[][] rows = rowsOf("hi", 3);
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": ["
                + "{\"type\": \"text\", \"text\": \"look \"},"
                + "{\"type\": \"image_embeds\", \"image_embeds\": " + json(rows) + "},"
                + "{\"type\": \"text\", \"text\": \" now\"}]}],"
                + " \"max_tokens\": 3, \"temperature\": 0}";
        Response response = resource.chatCompletions(body);
        assertEquals(200, response.getStatus());
        JsonView usage = JsonView.of(view(response.getEntity()).raw("usage"), "usage");
        String rendered = service.formatChat(OpenAiRequests.chat(view(body), false));
        int encoded = 0;
        for (String segment : rendered.split(java.util.regex.Pattern.quote(
                com.eonmux.neuralfree.gguf.chat.ChatMessages.PROMPT_EMBEDS_MARKER), -1)) {
            encoded += service.encode(segment, false, true).length;
        }
        assertEquals((long) encoded + rows.length, usage.i64("prompt_tokens"),
                "the prompt rows are its tokens plus the part's rows");
    }

    @Test
    void anAudioEmbedsPartRunsLikeAnImageEmbedsPart() {
        float[][] rows = rowsOf("hi", 2);
        Response response = resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\": ["
                + "{\"type\": \"audio_embeds\", \"audio_embeds\": " + json(rows) + "}]}],"
                + " \"max_tokens\": 2, \"temperature\": 0}");
        assertEquals(200, response.getStatus());
    }

    @Test
    void refusalsNameTheirField() {
        assertEquals("Either prompt or prompt_embeds must be provided and non-empty.",
                errorOf(() -> resource.completions("{\"prompt_embeds\": [], \"max_tokens\": 2}")));
        assertEquals(400, statusOf(() -> resource.completions("{\"max_tokens\": 2}")));

        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt_embeds\": " + json(rowsOf("hi", 1)) + ", \"echo\": true, \"max_tokens\": 2}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt_embeds\": [[1, 2]], \"max_tokens\": 2}")));
        assertTrue(errorOf(() -> resource.completions("{\"prompt_embeds\": [[1, 2]], \"max_tokens\": 2}"))
                .contains("embedding width"));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt_embeds\": [[1, 2], [3]], \"max_tokens\": 2}")));
        assertTrue(errorOf(() -> resource.completions("{\"prompt_embeds\": [[1, 2], [3]], \"max_tokens\": 2}"))
                .contains("prompt_embeds[0]"));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt_embeds\": [[1e40]], \"max_tokens\": 2}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt_embeds\": \"not rows\", \"max_tokens\": 2}")));

        assertEquals(400, statusOf(() -> resource.chatCompletions("{\"messages\": [{\"role\": \"user\","
                + " \"content\": [{\"type\": \"image_embeds\", \"image_embeds\": \"AAEC\"}]}],"
                + " \"max_tokens\": 2}")));
        assertTrue(errorOf(() -> resource.chatCompletions("{\"messages\": [{\"role\": \"user\","
                        + " \"content\": [{\"type\": \"image_embeds\", \"image_embeds\": \"AAEC\"}]}],"
                        + " \"max_tokens\": 2}"))
                .contains("image_embeds"));
        assertEquals(400, statusOf(() -> resource.chatCompletions("{\"messages\": [{\"role\": \"user\","
                        + " \"content\": [{\"type\": \"image_embeds\", \"image_embeds\": [[1, 2]]}]}],"
                        + " \"max_tokens\": 2}")),
                "a part row off the model's width is a 400 naming the part");
    }

    @Test
    void inputTokensCountEmbeddingRowsLikeTheGeneration() {
        float[][] rows = rowsOf("hi", 2);
        String messages = "\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\":"
                + " \"see\"}, {\"type\": \"image_embeds\", \"image_embeds\": " + json(rows) + "}]}]";
        long counted = view(resource.chatInputTokens("{" + messages + "}").getEntity()).i64("input_tokens");
        Response response = resource.chatCompletions("{" + messages + ", \"max_tokens\": 2, \"temperature\": 0}");
        assertEquals(200, response.getStatus());
        assertEquals(JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens"), counted,
                "the count holds the embedding rows the generation's prompt takes");
        assertTrue(counted >= rows.length, counted + " rows");
    }
}
