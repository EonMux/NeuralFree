package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * DeepSeek V4.1's message rendering ({@code render_message_to_json_deepseek_v4_1}, llama.cpp common/chat.cpp:568)
 * against llama.cpp's own minimal V4.1 template and its {@code test_deepseek_v4_1_message_fields} assertions
 * (tests/test-chat.cpp:8054): the message's own keys reach the template, a tool call's keys beside
 * {@code type}/{@code function}/{@code id} reach the rendered call, and media markers stay separate text parts
 * the template joins itself.
 */
public class DeepSeekV41RenderTest {

    /** llama.cpp's DEEPSEEK_V4_1_MIN_TEMPLATE (tests/test-chat.cpp:156). */
    private static final String TEMPLATE = "{%- for m in messages -%}\n"
            + "{%- if m.role == 'system' -%}<\uFF5CSystem\uFF5C>{{ m.content }}\n"
            + "{%- elif m.role == 'user' -%}<\uFF5CUser\uFF5C>\n"
            + "{%- if m.content is string -%}{{ m.content }}{%- else -%}{%- for b in m.content -%}"
            + "{%- if not loop.first -%}{{ '\\n\\n' }}{%- endif -%}{{ b.text }}{%- endfor -%}{%- endif -%}\n"
            + "{%- if m.task -%}<\uFF5C{{ m.task }}\uFF5C>{%- endif -%}\n"
            + "{%- elif m.role == 'assistant' -%}<\uFF5CAssistant\uFF5C></think>{{ m.content }}\n"
            + "{%- for tc in m.tool_calls or [] -%}<\uFF5CDSML\uFF5C calls><\uFF5CDSML\uFF5C invoke name="
            + "\"{% if tc.namespace %}{{ tc.namespace }}::{% endif %}{{ tc.function.name }}\">"
            + "</\uFF5CDSML\uFF5C invoke></\uFF5CDSML\uFF5C calls>{%- endfor -%}\n"
            + "<\uFF5Cend▁of▁sentence\uFF5C>\n"
            + "{%- endif -%}\n"
            + "{%- endfor -%}\n"
            + "{%- if reasoning_effort is defined and reasoning_effort is not none -%}"
            + "<\uFF5Ceffort={{ reasoning_effort }}\uFF5C>{%- endif -%}\n"
            + "{%- if add_generation_prompt -%}<\uFF5CAssistant\uFF5C>{%- if enable_thinking -%}<think>"
            + "{%- else -%}</think>{%- endif -%}{%- endif -%}";

    private static ChatFormatter formatter() {
        return new ChatFormatter(ChatTemplateSource.resolve(TEMPLATE), "", "", false, false, "llama");
    }

    private static Map<String, Object> message(String role, Object content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    private static ChatRequest chat(boolean addGenerationPrompt, Map<String, Object>... messages) {
        return ChatRequest.builder().messages(Arrays.asList(messages)).enableThinking(false)
                .addGenerationPrompt(addGenerationPrompt).build();
    }

    @Test
    public void theExtraMessageKeysReachTheTemplate() {
        Map<String, Object> user = message("user", "q");
        user.put("task", "action");
        assertEquals("m.task renders, as llama.cpp's test asserts", "<\uFF5CUser\uFF5C>q<\uFF5Caction\uFF5C>",
                formatter().format(chat(false, user)));
    }

    @Test
    public void theExtraToolCallKeysReachTheRenderedCall() {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", "f");
        function.put("arguments", Collections.singletonMap("x", 1));
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("type", "function");
        call.put("namespace", "ns");
        call.put("function", function);
        Map<String, Object> assistant = message("assistant", "");
        assistant.put("tool_calls", Collections.singletonList(call));
        String rendered = formatter().format(chat(false, message("user", "q"), assistant));
        assertTrue(rendered, rendered.contains("<\uFF5CDSML\uFF5C invoke name=\"ns::f\">"));
        assertTrue("the template names the DSML calls block, so the V4.1 rendering applies",
                ChatMessages.isDeepSeekV41Template(ChatTemplateSource.resolve(TEMPLATE)));
    }

    @Test
    public void theMediaMarkersStaySeparateTextPartsTheTemplateJoinsItself() {
        Map<String, Object> image = new LinkedHashMap<>();
        image.put("type", ChatMessages.PART_IMAGE);
        image.put("image_url", Collections.singletonMap("url", "data:x"));
        Map<String, Object> user = message("user", Arrays.asList(textPart("a"), image, textPart("b")));
        assertEquals("the template's own '\\n\\n' join separates the blocks",
                "<\uFF5CUser\uFF5C>a\n\n" + ChatMessages.MEDIA_MARKER + "\n\nb<\uFF5CAssistant\uFF5C></think>",
                formatter().format(chat(true, user), true));
    }

    @Test
    public void aStringOnlyDeepSeekTemplateJoinsThePartsAroundTheMarker() {
        String stringOnly = "{# \uFF5CDSML\uFF5C calls> #}{% for m in messages %}[{{ m.content }}]{% endfor %}";
        Map<String, Object> image = new LinkedHashMap<>();
        image.put("type", ChatMessages.PART_IMAGE);
        image.put("image_url", Collections.singletonMap("url", "data:x"));
        Map<String, Object> user = message("user", Arrays.asList(textPart("a"), image, textPart("b")));
        assertEquals("the string form carries the marker inline, the parts' newline join around it",
                "[a\n" + ChatMessages.MEDIA_MARKER + "\nb]",
                new ChatFormatter(stringOnly, "", "", false, false, "llama").format(chat(false, user), true));
    }

    @Test
    public void aToolReferenceInAMediaMessageBecomesAnEmptyTextBlock() {
        Map<String, Object> image = new LinkedHashMap<>();
        image.put("type", ChatMessages.PART_IMAGE);
        image.put("image_url", Collections.singletonMap("url", "data:x"));
        Map<String, Object> reference = new LinkedHashMap<>();
        reference.put("type", ChatMessages.PART_TOOL_REFERENCE);
        reference.put("name", "buy");
        Map<String, Object> user = message("user", Arrays.asList(textPart("a"), reference, image));
        assertEquals("llama.cpp retypes every part of a media message as text, the reference's text empty",
                "<\uFF5CUser\uFF5C>a\n\n\n\n" + ChatMessages.MEDIA_MARKER,
                formatter().format(chat(false, user), true));
    }

    @Test
    public void aRequestOverrideNamingTheDsmlCallsBlockRendersTheV41Way() {
        Map<String, Object> image = new LinkedHashMap<>();
        image.put("type", ChatMessages.PART_IMAGE);
        image.put("image_url", Collections.singletonMap("url", "data:x"));
        Map<String, Object> user = message("user", Arrays.asList(textPart("a"), image));
        ChatRequest request = ChatRequest.builder().messages(Collections.singletonList(user))
                .addGenerationPrompt(false).chatTemplate(TEMPLATE).build();
        ChatFormatter model = new ChatFormatter("{{ messages[0].content }}", "", "", false, false, "llama");
        assertEquals("the override's own input conventions apply, not the model template's",
                "<\uFF5CUser\uFF5C>a\n\n" + ChatMessages.MEDIA_MARKER, model.format(request, true));
    }

    private static Map<String, Object> textPart(String text) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "text");
        part.put("text", text);
        return part;
    }
}
