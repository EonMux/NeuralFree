package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Configured chat template variables: {@link ChatTemplateKwargs} parses the JSON object of llama.cpp's
 * {@code --chat-template-kwargs} and vLLM's {@code --default-chat-template-kwargs}, and
 * {@link ChatRequest#withDefaultTemplateKwargs} merges them under a request's own variables as vLLM's
 * {@code ChatParams.with_defaults} does.
 */
public class ChatTemplateKwargsTest {

    /** Renders {@code enable_thinking} and {@code reasoning_effort} as the template sees them. */
    private static final String PROBE = "{% if enable_thinking is defined %}"
            + "{{ 'true' if enable_thinking else 'false' }}{% else %}unset{% endif %}"
            + "/{{ reasoning_effort | default('unset') }}/{{ style | default('unset') }}";

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    private static ChatRequest.Builder request() {
        return ChatRequest.builder().messages(Collections.singletonList(message("user", "U")));
    }

    private static String render(ChatRequest request) {
        return new ChatFormatter(PROBE, "", "", false, false, "llama").format(request);
    }

    @Test
    public void parseReadsAJsonObjectInMemberOrder() {
        Map<String, Object> kwargs = ChatTemplateKwargs.parse(
                "{\"enable_thinking\": false, \"style\": {\"n\": 2, \"tags\": [\"a\", null]}, \"x\": 1.5}", "--flag");
        assertEquals(Arrays.asList("enable_thinking", "style", "x"), new ArrayList<>(kwargs.keySet()));
        assertEquals(Boolean.FALSE, kwargs.get("enable_thinking"));
        assertEquals(Arrays.asList("a", null), ((Map<?, ?>) kwargs.get("style")).get("tags"));
        assertThrows(UnsupportedOperationException.class, () -> kwargs.put("y", 1));
        assertTrue(ChatTemplateKwargs.parse("{}", "--flag").isEmpty());
    }

    @Test
    public void parseRefusesAnythingButAnObjectNamingTheSource() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChatTemplateKwargs.parse("[1, 2]", "--chat-template-kwargs"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--chat-template-kwargs must be a JSON object"));
        e = assertThrows(IllegalArgumentException.class, () -> ChatTemplateKwargs.parse("{bad", "--x"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--x is not valid JSON"));
        e = assertThrows(IllegalArgumentException.class, () -> ChatTemplateKwargs.parse("{\"\": 1}", "--x"));
        assertTrue(e.getMessage(), e.getMessage().contains("empty variable name"));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateKwargs.parse(null, "--x"));
    }

    @Test
    public void thePropertyIsReadWhenSetAndRoundTripsThroughToJson() {
        Properties p = new Properties();
        assertTrue(ChatTemplateKwargs.fromProperties(p).isEmpty());
        p.setProperty(ChatTemplateKwargs.PROPERTY, "  ");
        assertTrue(ChatTemplateKwargs.fromProperties(p).isEmpty());
        Map<String, Object> kwargs = ChatTemplateKwargs.parse("{\"a\": \"é\", \"b\": [true, null]}", "x");
        p.setProperty(ChatTemplateKwargs.PROPERTY, ChatTemplateKwargs.toJson(kwargs));
        assertEquals(kwargs, ChatTemplateKwargs.fromProperties(p));
        p.setProperty(ChatTemplateKwargs.PROPERTY, "true");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChatTemplateKwargs.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().startsWith(ChatTemplateKwargs.PROPERTY));
    }

    @Test
    public void defaultsApplyToARequestThatLeavesTheVariablesUnset() {
        Map<String, Object> defaults = ChatTemplateKwargs.parse(
                "{\"enable_thinking\": false, \"reasoning_effort\": \"low\", \"style\": \"terse\"}", "x");
        ChatRequest merged = request().build().withDefaultTemplateKwargs(defaults);
        assertEquals("false/low/terse", render(merged));
        assertEquals(defaults, merged.templateKwargs());
        assertEquals("the reasoning parsers see the defaults", Boolean.FALSE,
                merged.templateVariables().get("enable_thinking"));
    }

    @Test
    public void theRequestsOwnVariablesWinOverTheDefaults() {
        Map<String, Object> defaults = ChatTemplateKwargs.parse(
                "{\"enable_thinking\": false, \"reasoning_effort\": \"low\", \"style\": \"terse\"}", "x");
        ChatRequest own = request().enableThinking(true).reasoningEffort("high").templateKwarg("style", "long")
                .build().withDefaultTemplateKwargs(defaults);
        assertEquals("true/high/long", render(own));
        assertEquals(Boolean.TRUE, own.templateVariables().get("enable_thinking"));
        assertEquals("high", own.templateVariables().get("reasoning_effort"));

        ChatRequest kwargsOnly = request().templateKwarg("enable_thinking", true).build()
                .withDefaultTemplateKwargs(defaults);
        assertEquals("true/low/terse", render(kwargsOnly));
        assertEquals("applied defaults lead, the request's own kwargs follow",
                Arrays.asList("reasoning_effort", "style", "enable_thinking"),
                new ArrayList<>(kwargsOnly.templateKwargs().keySet()));
    }

    @Test
    public void aNullOrAutoRequestKwargCountsAsUnset() {
        Map<String, Object> defaults = ChatTemplateKwargs.parse("{\"style\": \"terse\", \"reasoning_effort\": \"low\"}",
                "x");
        ChatRequest request = request().templateKwarg("style", null).templateKwarg("reasoning_effort", "auto")
                .build().withDefaultTemplateKwargs(defaults);
        assertEquals("terse", request.templateKwargs().get("style"));
        assertEquals("low", request.templateKwargs().get("reasoning_effort"));
        ChatRequest withoutDefault = request().templateKwarg("other", null).build()
                .withDefaultTemplateKwargs(defaults);
        assertTrue("an unset kwarg without a default stays", withoutDefault.templateKwargs().containsKey("other"));
        assertNull(withoutDefault.templateKwargs().get("other"));
    }

    @Test
    public void noApplicableDefaultLeavesTheRequestAsItIs() {
        ChatRequest request = request().templateKwarg("style", "long").build();
        assertSame(request, request.withDefaultTemplateKwargs(null));
        assertSame(request, request.withDefaultTemplateKwargs(Collections.emptyMap()));
        assertSame(request, request.withDefaultTemplateKwargs(Collections.singletonMap("style", "terse")));
    }

    @Test
    public void theMergedRequestKeepsEveryOtherField() {
        List<Map<String, Object>> documents = Collections.singletonList(Collections.singletonMap("title", "T"));
        ChatRequest request = request().chatTemplate("llama3").documents(documents).addGenerationPrompt(false)
                .build().withDefaultTemplateKwargs(Collections.singletonMap("style", "terse"));
        assertEquals("llama3", request.chatTemplate());
        assertEquals(documents, request.documents());
        assertEquals(false, request.addGenerationPrompt());
        assertEquals(1, request.messages().size());
    }
}
