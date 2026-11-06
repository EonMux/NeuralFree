package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Messages reach a template in the forms its capabilities read, as llama.cpp's {@code messages_inp_normalizer}
 * and {@code func_args_not_string} give them (common/chat.cpp:489-552, :1532): typed text parts joined with
 * newlines for a string-only template, a string wrapped as a text part for a typed-only one, both kept for a
 * template reading both, and tool-call arguments as an object only for a template that reads them as one; and
 * {@code preserve_reasoning} binds llama.cpp's reasoning-history aliases (common/chat.cpp:1077-1080).
 */
public class TemplateNormalizationTest {

    private static final String STRING_ONLY = "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}";
    private static final String TYPED_ONLY = "{% for m in messages %}[{{ m.role }}]"
            + "{% for p in m.content %}{{ p.text }}{% endfor %}{% endfor %}";
    private static final String BOTH = "{% for m in messages %}[{{ m.role }}]{% if m.content is string %}"
            + "S:{{ m.content }}{% else %}T:{% for p in m.content %}({{ p.text }}){% endfor %}{% endif %}{% endfor %}";

    private static ChatFormatter formatter(String source) {
        return new ChatFormatter(source, "<bos>", "<eos>", true, false, "llama");
    }

    private static Map<String, Object> message(String role, Object content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    private static Map<String, Object> text(String t) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("type", "text");
        p.put("text", t);
        return p;
    }

    private static ChatRequest chat(Map<String, Object>... messages) {
        return ChatRequest.builder().messages(Arrays.asList(messages)).build();
    }

    // ------------------------------------------------------------------ content forms

    @Test
    public void aTypedOnlyTemplateGetsStringContentAsOneTextPart() {
        ChatFormatter f = formatter(TYPED_ONLY);
        assertFalse(f.caps().supportsStringContent());
        assertTrue(f.caps().supportsTypedContent());
        assertEquals("[user]Hi", f.format(chat(message("user", "Hi"))));
    }

    @Test
    public void aStringOnlyTemplateGetsTypedTextPartsJoinedWithNewlines() {
        ChatFormatter f = formatter(STRING_ONLY);
        assertEquals("[user]a\nb", f.format(chat(message("user", Arrays.asList(text("a"), text("b"))))));
    }

    @Test
    public void aTemplateReadingBothGetsEachContentAsGiven() {
        ChatFormatter f = formatter(BOTH);
        assertEquals("[user]S:x[user]T:(a)(b)", f.format(chat(message("user", "x"),
                message("user", Arrays.asList(text("a"), text("b"))))));
    }

    @Test
    public void mediaPartsStayAStringWithTheMarkerAndWrapForATypedOnlyTemplate() {
        Map<String, Object> image = new LinkedHashMap<>();
        image.put("type", ChatMessages.PART_IMAGE);
        image.put("image_url", Collections.singletonMap("url", "data:x"));
        ChatRequest request = chat(message("user", Arrays.asList(text("see"), image, text("then"), text("this"))));
        String m = ChatMessages.MEDIA_MARKER;
        assertEquals("[user]see" + m + "then\nthis", formatter(TYPED_ONLY).format(request, true));
        assertEquals("[user]see" + m + "then\nthis", formatter(STRING_ONLY).format(request, true));
        assertThrows(UnsupportedOperationException.class, () -> formatter(TYPED_ONLY).format(request));
    }

    @Test
    public void theSystemPromptMergesIntoTypedContent() {
        String noSystem = "{% if messages[0].role == 'system' %}{{ raise_exception('no system') }}{% endif %}"
                + TYPED_ONLY;
        ChatFormatter f = formatter(noSystem);
        assertFalse(f.supportsSystemRole());
        assertEquals("[user]S\nU", f.format(chat(message("system", "S"), message("user", "U"))));
        assertEquals("[user]S\nU1\nU2", formatter(noSystem.replace(TYPED_ONLY, STRING_ONLY)).format(
                chat(message("system", "S"), message("user", Arrays.asList(text("U1"), text("U2"))))));
    }

    @Test
    public void theHarmonyFallbackGetsTypedTextPartsAsText() {
        ChatFormatter f = new ChatFormatter("{{ raise_exception('unsupported') }}", "", "", false, false, "gpt-oss");
        String out = f.format(chat(message("user", Arrays.asList(text("a"), text("b")))));
        assertTrue(out, out.endsWith("<|start|>user<|message|>a\nb<|end|><|start|>assistant"));
    }

    @Test
    public void aForcedContentFormatOverridesTheDetectedOne() {
        ChatRequest request = chat(message("user", "x"), message("user", Arrays.asList(text("a"), text("b"))));
        ChatFormatter openai = new ChatFormatter(BOTH, "", "", false, false, "llama", null,
                TemplateCaps.ContentFormat.OPENAI);
        assertEquals("[user]T:(x)[user]T:(a)(b)", openai.format(request));
        assertFalse(openai.caps().supportsStringContent());
        ChatFormatter string = new ChatFormatter(BOTH, "", "", false, false, "llama", null,
                TemplateCaps.ContentFormat.STRING);
        assertEquals("[user]S:x[user]S:a\nb", string.format(request));
        ChatRequest override = ChatRequest.builder().messages(Collections.singletonList(message("user", "y")))
                .chatTemplate(BOTH).build();
        assertEquals("the forced format holds for an override too", "[user]T:(y)", openai.format(override));
    }

    @Test
    public void theContentFormatPropertyParsesOrIsRefused() {
        String key = ChatFormatter.CONTENT_FORMAT_PROPERTY;
        String before = System.getProperty(key);
        try {
            System.clearProperty(key);
            assertEquals(TemplateCaps.ContentFormat.AUTO, ChatFormatter.contentFormatFromProperties());
            System.setProperty(key, "OpenAI");
            assertEquals(TemplateCaps.ContentFormat.OPENAI, ChatFormatter.contentFormatFromProperties());
            System.setProperty(key, "typed");
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    ChatFormatter::contentFormatFromProperties);
            assertTrue(e.getMessage(), e.getMessage().contains(key));
        } finally {
            if (before == null) System.clearProperty(key);
            else System.setProperty(key, before);
        }
    }

    @Test
    public void theLegacyRenderNormalisesToo() {
        List<Map<String, ?>> conv = new ArrayList<>();
        conv.add(message("user", "Hi"));
        assertEquals("[user]Hi", formatter(TYPED_ONLY).format(conv, false));
    }

    @Test
    public void anOverrideIsNormalisedForItsOwnCapabilities() {
        ChatRequest request = ChatRequest.builder().messages(Collections.singletonList(message("user", "Hi")))
                .chatTemplate(TYPED_ONLY).build();
        assertEquals("[user]Hi", formatter(STRING_ONLY).format(request));
    }

    @Test
    public void aStringOnlyTemplateRefusesNonTextPartsByIndex() {
        Map<String, Object> odd = new LinkedHashMap<>();
        odd.put("type", "file");
        List<Map<String, ?>> conv = new ArrayList<>();
        conv.add(message("user", Arrays.asList(text("a"), odd)));
        // the legacy render takes parts as given, and a string-only template cannot carry the file part
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> formatter(STRING_ONLY).format(conv, false));
        assertTrue(e.getMessage(), e.getMessage().contains("message 0 content part 1 is not a text part"));
    }

    @Test
    public void aRealStringOnlyTemplateRendersTypedPartsAsTheJoinedString() throws IOException {
        ChatFormatter f = formatter(resource("Qwen-Qwen2.5-7B-Instruct"));
        assertEquals(f.format(chat(message("user", "a\nb"))),
                f.format(chat(message("user", Arrays.asList(text("a"), text("b"))))));
    }

    @Test
    public void aRealTypedTemplateRendersStringAndTypedContentAlike() throws IOException {
        ChatFormatter f = formatter(resource("Mistral-Small-3.2-24B-Instruct-2506"));
        assertTrue(f.caps().supportsTypedContent());
        assertEquals(f.format(chat(message("user", "Hello"))),
                f.format(chat(message("user", Collections.singletonList(text("Hello"))))));
    }

    // ------------------------------------------------------------------ StepFun per-part trimming

    /** A StepFun prompt line, then a template that prints each message's content between brackets. */
    private static final String STEPFUN = "{# You have access to the following functions in JSONSchema format #}"
            + "{% for m in messages %}[{% if m.content is string %}{{ m.content }}{% else %}"
            + "{% for p in m.content %}{{ p.text }}{% endfor %}{% endif %}]{% endfor %}";

    private static Map<String, Object> image() {
        Map<String, Object> image = new LinkedHashMap<>();
        image.put("type", ChatMessages.PART_IMAGE);
        image.put("image_url", Collections.singletonMap("url", "data:x"));
        return image;
    }

    @Test
    public void aStepFunTemplateTrimsEachPartBeforeThePartsJoin() {
        Map<String, Object> message = message("user", Arrays.asList(text("a"), image(), text(" b ")));
        assertEquals("the trailing part's space is gone, not only the string's outer whitespace",
                "[a" + ChatMessages.MEDIA_MARKER + "b]",
                formatter(STEPFUN).format(chat(message), true));
        assertEquals("another template keeps the caller's text as given", "[a" + ChatMessages.MEDIA_MARKER
                        + " b ]", formatter(STEPFUN.substring(STEPFUN.indexOf("{%"))).format(chat(message), true));
    }

    @Test
    public void aStepFunJoinedContentIsNotTrimmedAgain() {
        Map<String, Object> message = message("user", Arrays.asList(image(), text(" a "), text("  ")));
        assertEquals("llama.cpp joins the trimmed parts, so a blank last part still leaves its newline",
                "[" + ChatMessages.MEDIA_MARKER + "a\n]", formatter(STEPFUN).format(chat(message), true));
    }

    @Test
    public void aStepFunRequestTrimsStringContentAndReasoning() {
        String source = STEPFUN + "{% for m in messages %}<{{ m.reasoning_content }}>{% endfor %}";
        Map<String, Object> assistant = message("assistant", "\n answer \t");
        assistant.put("reasoning_content", "  why\n");
        assertEquals("[q][answer]<><why>", formatter(source).format(chat(message("user", " q "), assistant)));
    }

    // ------------------------------------------------------------------ tool_reference parts

    private static final String TOOL_REF =
            "{% for m in messages %}[{{ m.role }}]{% for p in m.content %}"
                    + "{% if p.type == 'tool_reference' %}R:{{ p.name }}{% else %}{{ p.text }}{% endif %}"
                    + "{% endfor %}{% endfor %}";

    private static Map<String, Object> toolReference(Object name) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "tool_reference");
        part.put("name", name);
        return part;
    }

    @Test
    public void aToolReferencePartReachesATypedTemplateWithItsName() {
        ChatRequest request = chat(message("user", Arrays.asList(text("a"), toolReference("get_time"))));
        assertEquals("[user]aR:get_time", formatter(TOOL_REF).format(request));
    }

    @Test
    public void aStringJoinSkipsAToolReferencePart() {
        ChatRequest request = chat(message("user", Arrays.asList(text("a"), toolReference("get_time"))));
        assertEquals("llama.cpp's join ignores the types it cannot print", "[user]a",
                formatter(STRING_ONLY).format(request));
        assertEquals("and so does the join of a message that carries media", "[user]a"
                        + ChatMessages.MEDIA_MARKER,
                formatter(STRING_ONLY).format(chat(message("user",
                        Arrays.asList(text("a"), toolReference("get_time"), image()))), true));
    }

    @Test
    public void aToolReferencePartWithoutAStringNameIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> formatter(TOOL_REF).format(chat(message("user",
                        Collections.singletonList(toolReference(true))))));
        assertTrue(e.getMessage(), e.getMessage().contains("tool_reference part has no string 'name'"));
    }

    private static String resource(String name) throws IOException {
        try (InputStream in = TemplateNormalizationTest.class.getResourceAsStream("/chat/templates/" + name
                + ".jinja")) {
            assertNotNull(name, in);
            return ChatTemplateSource.resolve(new String(in.readAllBytes(), StandardCharsets.UTF_8), null);
        }
    }

    // ------------------------------------------------------------------ tool-call arguments

    private static ChatRequest toolCall(Object arguments) {
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", "get_weather");
        function.put("arguments", arguments);
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("id", "c1");
        call.put("type", "function");
        call.put("function", function);
        Map<String, Object> assistant = message("assistant", "");
        assistant.put("tool_calls", Collections.singletonList(call));
        return chat(message("user", "Weather?"), assistant);
    }

    private static final String ARGS_TEXT = "{% for m in messages %}{% for tc in m.tool_calls %}"
            + "{{ tc.function.name }}={{ tc.function.arguments }}{% endfor %}{% endfor %}";
    private static final String ARGS_OBJECT = "{% for m in messages %}{% for tc in m.tool_calls %}"
            + "{{ tc.function.name }}={{ tc.function.arguments | tojson }};{{ tc.arguments.city }}{% endfor %}{% endfor %}";

    @Test
    public void aTemplatePrintingTheArgumentsGetsJsonText() {
        ChatFormatter f = formatter(ARGS_TEXT);
        assertFalse(f.caps().supportsObjectArguments());
        assertEquals("get_weather={\"city\":\"Paris\"}",
                f.format(toolCall(Collections.singletonMap("city", "Paris"))));
        assertEquals("text arguments stay as the caller wrote them", "get_weather={ \"city\" : \"Paris\" }",
                f.format(toolCall("{ \"city\" : \"Paris\" }")));
    }

    @Test
    public void aTemplateReadingTheArgumentsGetsAnObject() {
        ChatFormatter f = formatter(ARGS_OBJECT);
        assertTrue(f.caps().supportsObjectArguments());
        assertEquals("get_weather={\"city\": \"Paris\"};Paris", f.format(toolCall("{\"city\":\"Paris\"}")));
        assertEquals("get_weather={\"city\": \"Paris\"};Paris",
                f.format(toolCall(Collections.singletonMap("city", "Paris"))));
    }

    @Test
    public void argumentsThatAreNotAJsonObjectAreRefusedByName() {
        for (String template : Arrays.asList(ARGS_TEXT, ARGS_OBJECT)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> formatter(template).format(toolCall("{not json")));
            assertTrue(e.getMessage(), e.getMessage().contains("tool call 0 of message 1"));
            assertThrows(IllegalArgumentException.class, () -> formatter(template).format(toolCall("[1]")));
        }
    }

    // ------------------------------------------------------------------ preserve_reasoning and effort aliases

    private static final String ALIASES = "{% if preserve_thinking %}P{% endif %}{% if clear_thinking %}C{% endif %}"
            + "{% if truncate_history_thinking %}T{% endif %}{% if drop_thinking %}D{% endif %}"
            + "{% if preserve_reasoning is defined %}={{ preserve_reasoning }}{% endif %}";

    private static ChatRequest withKwarg(String key, Object value) {
        return ChatRequest.builder().messages(Collections.singletonList(message("user", "U")))
                .templateKwarg(key, value).build();
    }

    @Test
    public void preserveReasoningBindsLlamaCppsAliases() {
        ChatFormatter f = formatter(ALIASES);
        assertEquals("P=True", f.format(withKwarg("preserve_reasoning", true)));
        assertEquals("CTD=False", f.format(withKwarg("preserve_reasoning", false)));
        assertEquals("without any kwarg, llama.cpp's default of true holds", "P=True",
                f.format(chat(message("user", "U"))));
    }

    @Test
    public void theKwargsStillOverrideOneAlias() {
        ChatRequest request = ChatRequest.builder().messages(Collections.singletonList(message("user", "U")))
                .templateKwarg("preserve_reasoning", false).templateKwarg("drop_thinking", false).build();
        assertEquals("CT=False", formatter(ALIASES).format(request));
    }

    @Test
    public void theFormatterDefaultAppliesUntilARequestSetsIt() {
        ChatFormatter f = new ChatFormatter(ALIASES, "", "", false, false, "llama", Boolean.TRUE,
                TemplateCaps.ContentFormat.AUTO);
        assertEquals("P=True", f.format(chat(message("user", "U"))));
        assertEquals("CTD=False", f.format(withKwarg("preserve_reasoning", false)));
        ChatFormatter off = new ChatFormatter(ALIASES, "", "", false, false, "llama", Boolean.FALSE,
                TemplateCaps.ContentFormat.AUTO);
        assertEquals("a formatter default of false keeps the aliases off", "CTD=False",
                off.format(chat(message("user", "U"))));
        assertEquals("P=True", off.format(withKwarg("preserve_reasoning", true)));
    }

    @Test
    public void aNonBooleanPreserveReasoningIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> formatter(ALIASES).format(withKwarg("preserve_reasoning", "yes")));
        assertTrue(e.getMessage(), e.getMessage().contains("chat_template_kwargs.preserve_reasoning"));
    }

    @Test
    public void thePreserveDefaultPropertyParsesOrIsRefused() {
        String key = ChatFormatter.PRESERVE_REASONING_PROPERTY;
        String before = System.getProperty(key);
        try {
            System.clearProperty(key);
            assertEquals(null, ChatFormatter.preserveReasoningFromProperties());
            System.setProperty(key, "TRUE");
            assertEquals(Boolean.TRUE, ChatFormatter.preserveReasoningFromProperties());
            System.setProperty(key, "false");
            assertEquals(Boolean.FALSE, ChatFormatter.preserveReasoningFromProperties());
            System.setProperty(key, "maybe");
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    ChatFormatter::preserveReasoningFromProperties);
            assertTrue(e.getMessage(), e.getMessage().contains(key));
        } finally {
            if (before == null) System.clearProperty(key);
            else System.setProperty(key, before);
        }
    }

    @Test
    public void theReasoningEffortIsAlsoBoundAsReasoningStrength() {
        ChatRequest request = ChatRequest.builder().messages(Collections.singletonList(message("user", "U")))
                .reasoningEffort("high").build();
        assertEquals("high", formatter("{{ reasoning_strength }}").format(request));
    }
}
