package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The llama.cpp tokenizer endpoints with their vLLM spellings: {@code /tokenize} over content, prompt and chat
 * messages, {@code with_pieces}, {@code /detokenize} round trips, {@code /apply-template} and the health alias.
 */
public class LlamaCppResourceTest {

    private static EngineService service;
    private static EngineService unloaded;
    private static LlamaCppResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(64, 2),
                ServerOptions.defaults().withTrustRequestChatTemplate(true));
        unloaded = new EngineService(null, ServerTestSupport.pagedConfig(64, 2));
        resource = new LlamaCppResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
        unloaded.close();
    }

    @Test
    void propsNeverReportsTheWatermarkKey() {
        String saved = System.getProperty(WatermarkConfig.PROP_CONFIG);
        try {
            System.setProperty(WatermarkConfig.PROP_CONFIG,
                    "{\"key\": 987654321987, \"algorithm\": \"gumbel\"}");
            String body = (String) resource.props(null, null).getEntity();
            assertFalse(body.contains("987654321987"), body);
            assertFalse(body.contains(WatermarkConfig.PROP_CONFIG), body);
        } finally {
            if (saved == null) {
                System.clearProperty(WatermarkConfig.PROP_CONFIG);
            } else {
                System.setProperty(WatermarkConfig.PROP_CONFIG, saved);
            }
        }
    }

    @Test
    void tokenizeFollowsBothSpellingsOfTheSpecialTokenDefault() {
        JsonView llama = view(resource.tokenize("{\"content\": \"hello world\"}").getEntity());
        JsonView vllm = view(resource.tokenize("{\"prompt\": \"hello world\"}").getEntity());
        int plain = size(llama, "tokens");
        assertEquals(plain, llama.i64("count"));
        assertEquals(plain + 1, size(vllm, "tokens"), "vLLM adds the BOS by default, llama.cpp does not");
        assertEquals(1L, ((Number) ((List<?>) vllm.raw("tokens")).get(0)).longValue(), "the BOS id leads");
        assertEquals(service.maxModelLength(), vllm.i64("max_model_len"));
        JsonView withBos = view(resource.tokenize("{\"content\": \"hello world\", \"add_special\": true}")
                .getEntity());
        assertEquals(plain + 1, size(withBos, "tokens"));
    }

    @Test
    void withPiecesPairsEachIdWithItsText() {
        JsonView body = view(resource.tokenize("{\"content\": \"hello\", \"with_pieces\": true}").getEntity());
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < size(body, "tokens"); i++) {
            JsonView entry = at(body, "tokens", i);
            assertTrue(entry.i64("id") >= 0);
            text.append(entry.str("piece"));
        }
        assertTrue(text.toString().contains("hello"), "the pieces spell the content: " + text);
    }

    @Test
    void detokenizeRoundTripsTheTokens() {
        List<?> tokens = (List<?>) view(resource.tokenize("{\"content\": \"hello world\"}").getEntity())
                .raw("tokens");
        StringBuilder ids = new StringBuilder();
        for (Object id : tokens) {
            ids.append(ids.length() == 0 ? "" : ", ").append(id);
        }
        JsonView body = view(resource.detokenize("{\"tokens\": [" + ids + "]}").getEntity());
        assertEquals("hello world", body.str("content").strip());
        assertEquals(body.str("content"), body.str("prompt"));
        assertEquals("", view(resource.detokenize("{\"tokens\": []}").getEntity()).str("content"));
        assertEquals(400, statusOf(() -> resource.detokenize("{\"tokens\": [100000]}")));
        assertEquals(400, statusOf(() -> resource.detokenize("{}")));
    }

    @Test
    void chatMessagesTokenizeAndRenderThroughTheTemplate() {
        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}";
        String prompt = view(resource.applyTemplate(chat).getEntity()).str("prompt");
        assertTrue(prompt.contains("hi"), "the template rendered the message: " + prompt);
        JsonView tokens = view(resource.tokenize(chat).getEntity());
        assertEquals(service.encode(prompt, false, true).length, tokens.i64("count"));
        assertEquals(400, statusOf(() -> resource.tokenize("{}")));
        assertEquals(400, statusOf(() -> resource.applyTemplate("{\"messages\": []}")));
    }

    @Test
    void aChatTemplateOverrideRendersInPlaceOfTheModelTemplate() {
        String override = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}";
        JsonView rendered = view(resource.applyTemplate(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"chat_template\": \"" + override.replace("\"", "\\\"") + "\"}").getEntity());
        assertEquals("<user>hi", rendered.str("prompt"), "the override rendered, the model template did not");
        assertEquals(400, statusOf(() -> resource.applyTemplate(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"chat_template\": \"{{ messages[ }}\"}")), "an override that does not parse");
    }

    @Test
    void aChatTemplateOverrideMayNameALlamaCppBuiltIn() {
        JsonView rendered = view(resource.applyTemplate("{\"messages\": [{\"role\": \"system\", \"content\": \"S\"},"
                + " {\"role\": \"user\", \"content\": \" hi \"}], \"chat_template\": \"gemma\"}").getEntity());
        assertEquals("<start_of_turn>user\nS\n\nhi<end_of_turn>\n<start_of_turn>model\n", rendered.str("prompt"),
                "llama.cpp's gemma style rendered, the system text joining the user turn");
    }

    @Test
    void documentsRenderIntoTheTemplateContext() {
        JsonView rendered = view(resource.applyTemplate(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                        + " \"chat_template\": \"{{ documents[0]['title'] }}{% for m in messages %}{{ m['content']"
                        + " }}{% endfor %}\","
                        + " \"documents\": [{\"title\": \"Doc\", \"text\": \"Body\"}]}").getEntity());
        assertEquals("Dochi", rendered.str("prompt"));
        assertEquals(400, statusOf(() -> resource.applyTemplate(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"documents\": [3]}")));
    }

    @Test
    void healthAnswersOnBothPaths() {
        assertEquals("ok", view(resource.healthAlias().getEntity()).str("status"));
        assertEquals("ok", view(resource.health().getEntity()).str("status"));
    }

    @Test
    void unloadedHealthRefusesWithLlamaCppSemantics() {
        LlamaCppResource off = new LlamaCppResource(unloaded);
        assertEquals(503, statusOf(off::health));
        assertEquals(503, statusOf(off::healthAlias));
        assertTrue(errorOf(off::health).contains("no model loaded"));
    }

    @Test
    void propsReportsTheWebUiFieldsLlamaCppServes() {
        String savedConfig = System.getProperty(UiResource.CONFIG_PROPERTY);
        String savedWebui = System.getProperty(UiResource.WEBUI_PROPERTY);
        try {
            System.clearProperty(UiResource.CONFIG_PROPERTY);
            System.clearProperty(UiResource.WEBUI_PROPERTY);
            JsonView props = view(resource.props(null, null).getEntity());
            assertTrue(props.bool("ui"), "the bundled UI serves by default");
            assertEquals(Map.of(), props.raw("ui_settings"), "without --ui-config the defaults are empty");

            System.setProperty(UiResource.CONFIG_PROPERTY, "{\"theme\": \"light\"}");
            System.setProperty(UiResource.WEBUI_PROPERTY, "false");
            JsonView configured = view(resource.props(null, null).getEntity());
            assertFalse(configured.bool("ui"), "--no-ui reports the off UI");
            org.junit.jupiter.api.Assertions.assertEquals(Map.of("theme", "light"),
                    configured.raw("ui_settings"), "the --ui-config defaults travel to the UI");
        } finally {
            if (savedConfig == null) {
                System.clearProperty(UiResource.CONFIG_PROPERTY);
            } else {
                System.setProperty(UiResource.CONFIG_PROPERTY, savedConfig);
            }
            if (savedWebui == null) {
                System.clearProperty(UiResource.WEBUI_PROPERTY);
            } else {
                System.setProperty(UiResource.WEBUI_PROPERTY, savedWebui);
            }
        }
    }
}
