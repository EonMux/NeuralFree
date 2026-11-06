package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The server on the non-Jinja route ({@code --no-jinja}): the model's template is replaced by the
 * built-in its text detects when the model loads, and {@code /props} reports that built-in's source
 * as {@code chat_template} — what llama-server reports after its own template resolution.
 */
class ChatTemplateNoJinjaResourceTest {

    /** A model template the detector answers chatml for, though it is not the built-in ChatML source. */
    private static final String TEMPLATE =
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                    + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    private static String savedJinja;
    private static EngineService service;
    private static LlamaCppResource llama;

    @BeforeAll
    static void open() throws IOException {
        savedJinja = System.getProperty(ChatTemplateSource.JINJA_PROPERTY);
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "false");
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama()).str("tokenizer.chat_template", TEMPLATE)
                .writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 1), ServerOptions.defaults());
        llama = new LlamaCppResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
        if (savedJinja == null) {
            System.clearProperty(ChatTemplateSource.JINJA_PROPERTY);
        } else {
            System.setProperty(ChatTemplateSource.JINJA_PROPERTY, savedJinja);
        }
    }

    @Test
    void propsReportTheDetectedBuiltIn() {
        JsonView props = view(llama.props(null, null).getEntity());
        assertEquals(ChatTemplateSource.CHATML, props.str("chat_template"));
    }

    @Test
    void applyTemplateRendersThroughTheDetectedBuiltIn() {
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}],"
                + " \"add_generation_prompt\": true}";
        assertEquals("<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n",
                view(llama.applyTemplate(body).getEntity()).str("prompt"));
    }
}
