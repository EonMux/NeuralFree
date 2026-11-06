package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The modern xgrammar structural-tag {@code format} object over HTTP: a {@code guided_structural_tag} payload or a
 * vLLM {@code response_format} of type {@code structural_tag} holds the completion to its format — a plain tag,
 * the forced single-tag {@code triggered_tags} form, and a {@code triggered_tags} nested in a sequence — and an
 * unsupported member is refused by name with a 400.
 */
public class OpenAiStructuralTagTest {

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = pagedConfig(64, 2);
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void aFormatObjectConstrainsTheCompletion() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 24, "temperature": 0,
                 "guided_structural_tag": {"type": "structural_tag", "format": {"type": "tag",
                   "begin": "<answer>", "content": {"type": "json_schema",
                     "json_schema": {"const": "yes"}},
                   "end": "</answer>"}}}""");
        assertEquals(200, response.getStatus());
        assertEquals("<answer>\"yes\"</answer>", firstText(view(response.getEntity())));
    }

    @Test
    void aForcedTriggeredTagsPayloadConstrainsTheCompletion() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 48, "temperature": 0,
                 "guided_structural_tag": {"format": {"type": "triggered_tags", "triggers": ["<call"],
                   "at_least_one": true, "stop_after_first": true,
                   "tags": [{"begin": "<call>", "content": {"type": "json_schema",
                     "json_schema": {"const": 7}}, "end": "</call>"}]}}}""");
        assertEquals(200, response.getStatus());
        assertEquals("<call>7</call>", firstText(view(response.getEntity())));
    }

    @Test
    void aResponseFormatStructuralTagNestsTriggeredTagsInASequence() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 48, "temperature": 0,
                 "response_format": {"type": "structural_tag", "format": {"type": "sequence", "elements": [
                   {"type": "const_string", "value": "go:"},
                   {"type": "triggered_tags", "triggers": ["<call"], "at_least_one": true,
                    "stop_after_first": true, "tags": [{"begin": "<call>", "content": {"type": "regex",
                      "pattern": "[0-9]"}, "end": "</call>"}]},
                   {"type": "const_string", "value": "."}]}}}""");
        assertEquals(200, response.getStatus());
        String text = firstText(view(response.getEntity()));
        assertTrue(text.matches("go:<call>[0-9]</call>\\."), "the nested format holds the output: " + text);
    }

    @Test
    void anUnsupportedMemberIsRefusedByName() {
        String body = """
                {"prompt": "x", "guided_structural_tag": {"format": {"type": "sequence", "elements": [
                  {"type": "json_schema", "json_schema": {"type": "object"}, "style": "minimax_m3_xml"}]}}}""";
        assertEquals(400, statusOf(() -> resource.completions(body)));
        String error = errorOf(() -> resource.completions(body));
        assertTrue(error.contains("style 'minimax_m3_xml'"), "the refusal names the field: " + error);
    }

    @Test
    void anUnknownResponseFormatTypeNamesTheStructuralTagChoice() {
        String error = errorOf(() -> resource.completions("""
                {"prompt": "x", "response_format": {"type": "yaml"}}"""));
        assertTrue(error.contains("structural_tag"), "the refusal lists the accepted types: " + error);
    }

    private static String firstText(JsonView body) {
        JsonView choice = JsonView.of(((java.util.List<?>) body.raw("choices")).get(0), "choices[0]");
        return choice.str("text");
    }
}
