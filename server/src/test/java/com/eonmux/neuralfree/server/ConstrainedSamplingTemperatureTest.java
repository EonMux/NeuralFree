package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Grammar-constrained completions at the temperatures a client sends: the default (the engine's 0.8, no
 * {@code temperature} in the body), 0.8 and 1.0 over {@code response_format json_object}, a GBNF grammar, a
 * regex and a structural tag. The constraint head's masks (the grammar, {@code allowed_token_ids}, bad words)
 * compose with the truncating stages of the default and of a tight chain the way llama.cpp's grammar-first
 * resample and vLLM's pre-sampling logits processors keep them, so the request completes and a constraint whose
 * admitted set is empty answers a 400 naming it, not a 500.
 */
class ConstrainedSamplingTemperatureTest {

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = pagedConfig(256, 2);
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** The first choice's text of a completions request, which must answer 200. */
    private static String completionText(String body) {
        Response response = resource.completions(body);
        assertEquals(200, response.getStatus());
        JsonView choice = JsonView.of(((List<?>) view(response.getEntity()).raw("choices")).get(0), "choices[0]");
        return choice.str("text");
    }

    @Test
    void aJsonObjectConstrainsTheDefaultTemperatureCompletion() {
        String text = completionText("{\"prompt\": \"value:\", \"max_tokens\": 120, \"seed\": 7, "
                + "\"response_format\": {\"type\": \"json_object\"}}").trim();
        Json.parse(text, Json.Limits.UNTRUSTED);
    }

    @Test
    void theConstraintKindsConstrainTemperaturesAboveZero() {
        String format = "{\"type\": \"tag\", \"begin\": \"<answer>\", \"content\": {\"type\": \"json_schema\","
                + " \"json_schema\": {\"const\": \"yes\"}}, \"end\": \"</answer>\"}";
        List<String> bodies = List.of(
                "{\"prompt\": \"value:\", \"max_tokens\": 64, \"temperature\": 0.8, \"seed\": 7, "
                        + "\"grammar\": \"root ::= \\\"yes\\\" (\\\"es\\\")*\"}",
                "{\"prompt\": \"value:\", \"max_tokens\": 64, \"temperature\": 0.8, \"seed\": 7, "
                        + "\"guided_regex\": \"yes|no\"}",
                "{\"prompt\": \"value:\", \"max_tokens\": 64, \"temperature\": 1.0, \"seed\": 7, "
                        + "\"grammar\": \"root ::= \\\"yes\\\" (\\\"es\\\")*\"}",
                "{\"prompt\": \"value:\", \"max_tokens\": 64, \"temperature\": 1.0, \"seed\": 7, "
                        + "\"guided_regex\": \"yes|no\"}",
                "{\"prompt\": \"value:\", \"max_tokens\": 64, \"temperature\": 0.8, \"seed\": 7, "
                        + "\"guided_structural_tag\": {\"type\": \"structural_tag\", \"format\": " + format + "}}",
                "{\"prompt\": \"value:\", \"max_tokens\": 64, \"temperature\": 1.0, \"seed\": 7, "
                        + "\"response_format\": {\"type\": \"structural_tag\", \"format\": " + format + "}}");
        List<String> shapes = List.of("yes(es)*", "yes|no", "yes(es)*", "yes|no",
                "<answer>\"yes\"</answer>", "<answer>\"yes\"</answer>");
        for (int i = 0; i < bodies.size(); i++) {
            String text = completionText(bodies.get(i));
            assertTrue(text.matches(shapes.get(i)),
                    "constrained text was '" + text + "' for " + bodies.get(i));
        }
    }

    @Test
    void aDeadEndConstraintAnswersARefusalNamingIt() {
        // the grammar asks for the vocabulary token 999, which no candidate carries: its admitted set is empty
        WebApplicationException refused = assertThrows(WebApplicationException.class,
                () -> resource.completions("{\"prompt\": \"value:\", \"max_tokens\": 8, \"seed\": 7, "
                        + "\"grammar\": \"root ::= <[999]>\"}"));
        assertEquals(400, refused.getResponse().getStatus());
        String message = view(refused.getResponse().getEntity()).obj("error").str("message");
        assertTrue(message.contains("no candidate"), message);
    }

    /** The first {@code n} distinct vocabulary ids whose decoded text re-encodes to the id alone. */
    private static int[] roundTrippingTokens(int n) {
        GGUFTokenizer tokenizer = service.tokenizer();
        int[] out = new int[n];
        int found = 0;
        for (int id = 0; id < tokenizer.vocabSize() && found < n; id++) {
            String text = tokenizer.decode(new int[] {id});
            if (text.isEmpty()) {
                continue;
            }
            int[] encoded = tokenizer.encode(text, false, false);
            if (encoded.length == 1 && encoded[0] == id) {
                out[found++] = id;
            }
        }
        assertEquals(n, found, "found only " + found + " round-tripping tokens");
        return out;
    }

    @Test
    void theRequestMasksConfineTemperaturesAboveZero() {
        int banned = roundTrippingTokens(1)[0];
        int kept = roundTrippingTokens(2)[1];
        String bannedWord = service.tokenizer().decode(new int[] {banned});
        int[] repeated = new int[8];
        java.util.Arrays.fill(repeated, kept);
        // the sequence decode drops the first piece's space marker space; the streamed response may keep it
        String expected = service.tokenizer().decode(repeated);
        String withLeadingSpace = " " + expected;
        // allowed_token_ids confines the step to the two tokens and bad_words bans one of them; the tight
        // top-k/top-p/min-p operate on what the constraint head leaves, so only the kept token is sampled
        // (vLLM applies allowed_token_ids and bad words to the logits before any sampling stage)
        for (String temperature : new String[] {"0.8", "1.0"}) {
            String text = completionText("{\"prompt\": \"value:\", \"max_tokens\": 8, \"temperature\": "
                    + temperature + ", \"seed\": 7, \"top_k\": 2, \"top_p\": 0.1, \"min_p\": 0.9, "
                    + "\"allowed_token_ids\": [" + banned + ", " + kept + "], "
                    + "\"bad_words\": [\"" + bannedWord + "\"]}");
            assertTrue(text.equals(expected) || text.equals(withLeadingSpace),
                    "temperature " + temperature + " masked text was '" + text + "'");
        }
    }

    @Test
    void aMaskDeadEndAnswersARefusalNamingIt() {
        // the allowed ids leave only the one token and the bad word bans that token: the constraint head
        // leaves no candidate, the same named refusal as a grammar dead end
        int only = roundTrippingTokens(1)[0];
        String bannedWord = service.tokenizer().decode(new int[] {only});
        WebApplicationException refused = assertThrows(WebApplicationException.class,
                () -> resource.completions("{\"prompt\": \"value:\", \"max_tokens\": 8, \"seed\": 7, "
                        + "\"allowed_token_ids\": [" + only + "], \"bad_words\": [\"" + bannedWord + "\"]}"));
        assertEquals(400, refused.getResponse().getStatus());
        String message = view(refused.getResponse().getEntity()).obj("error").str("message");
        assertTrue(message.contains("no candidate"), message);
    }
}
