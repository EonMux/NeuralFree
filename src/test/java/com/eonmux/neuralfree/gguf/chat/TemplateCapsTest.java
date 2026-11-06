package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.MiniJinja;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The capability probe matches llama.cpp's {@code caps_get} (common/jinja/caps.cpp): the templates of
 * llama.cpp's models/templates report the capabilities llama.cpp's test-chat-analysis prints for them, the
 * cases of llama.cpp's tests/test-jinja.cpp {@code test_caps} hold, and each probe reacts to the construct it
 * looks for.
 */
public class TemplateCapsTest {

    private static TemplateCaps caps(String template) {
        return TemplateCaps.probe(MiniJinja.compile(template));
    }

    private static String resource(String name) throws IOException {
        try (InputStream in = TemplateCapsTest.class.getResourceAsStream("/chat/templates/" + name + ".jinja")) {
            assertNotNull(name, in);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    /** tools, tool_calls, system_role, parallel_tool_calls, typed_content, string_content as llama.cpp reports them. */
    private static void assertLlamaCppCaps(String name, boolean... expected) throws IOException {
        TemplateCaps c = caps(resource(name));
        boolean[] actual = {c.supportsTools(), c.supportsToolCalls(), c.supportsSystemRole(),
                c.supportsParallelToolCalls(), c.supportsTypedContent(), c.supportsStringContent()};
        assertEquals(name + " " + c, Arrays.toString(expected), Arrays.toString(actual));
    }

    // ------------------------------------------------------------------ parity with llama.cpp's templates

    @Test
    public void llamaCppTemplatesReportLlamaCppsCapabilities() throws IOException {
        assertLlamaCppCaps("Qwen-Qwen2.5-7B-Instruct", true, true, true, true, false, true);
        assertLlamaCppCaps("Qwen-Qwen3-0.6B", true, true, true, true, false, true);
        assertLlamaCppCaps("meta-llama-Llama-3.1-8B-Instruct", true, true, true, false, false, true);
        assertLlamaCppCaps("MiniMax-M1", true, true, true, true, true, true);
        assertLlamaCppCaps("google-gemma-2-2b-it", false, false, false, false, false, true);
        assertLlamaCppCaps("google-gemma-4-31B-it", true, true, true, true, true, true);
        assertLlamaCppCaps("Mistral-Small-3.2-24B-Instruct-2506", true, true, true, true, true, true);
        assertLlamaCppCaps("poolside-Laguna-S-2.1", true, true, true, true, false, true);
        assertLlamaCppCaps("deepseek-ai-DeepSeek-V4", true, true, true, true, false, true);
        assertLlamaCppCaps("GLM-4.7-Flash", true, true, true, true, true, true);
        assertLlamaCppCaps("microsoft-Phi-3.5-mini-instruct", false, false, true, false, false, true);
        assertLlamaCppCaps("Kimi-K2-Instruct", true, true, true, true, true, true);
    }

    @Test
    public void preserveReasoningMatchesLlamaCppsAssertions() throws IOException {
        // tests/test-chat-auto-parser.cpp test_laguna_s_preserve_reasoning, tests/test-chat.cpp (DeepSeek V4)
        assertTrue(caps(resource("poolside-Laguna-S-2.1")).supportsPreserveReasoning());
        assertTrue(caps(resource("deepseek-ai-DeepSeek-V4")).supportsPreserveReasoning());
        assertTrue(caps(resource("GLM-4.7-Flash")).supportsPreserveReasoning());
        assertFalse(caps(resource("Qwen-Qwen2.5-7B-Instruct")).supportsPreserveReasoning());
    }

    @Test
    public void templatesThatSerialiseOrIterateTheArgumentsReadThemAsAnObject() throws IOException {
        for (String name : Arrays.asList("Qwen-Qwen2.5-7B-Instruct", "meta-llama-Llama-3.1-8B-Instruct",
                "MiniMax-M1", "Mistral-Small-3.2-24B-Instruct-2506", "GLM-4.7-Flash", "Kimi-K2-Instruct",
                "google-gemma-4-31B-it", "deepseek-ai-DeepSeek-V4", "poolside-Laguna-S-2.1")) {
            assertTrue(name, caps(resource(name)).supportsObjectArguments());
        }
        assertFalse(caps(resource("google-gemma-2-2b-it")).supportsObjectArguments());
        assertFalse(caps(resource("microsoft-Phi-3.5-mini-instruct")).supportsObjectArguments());
    }

    // ------------------------------------------------------------------ llama.cpp tests/test-jinja.cpp test_caps

    @Test
    public void stringContent() {
        TemplateCaps c = caps("{% for message in messages %}{{ message['role'] + ': ' + message['content'] }}"
                + "{% endfor %}");
        assertTrue(c.supportsStringContent());
        assertFalse(c.supportsTypedContent());
    }

    @Test
    public void typedContentThatRaisesOrDropsAString() {
        TemplateCaps selectattr = caps("{% for message in messages %}"
                + "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
                + "{{ content['text'] }}{% endfor %}{% endfor %}");
        assertFalse(selectattr.supportsStringContent());
        assertTrue(selectattr.supportsTypedContent());
        TemplateCaps subscript = caps("{% for message in messages %}{{ message['content'][0]['text'] }}{% endfor %}");
        assertFalse(subscript.supportsStringContent());
        assertTrue(subscript.supportsTypedContent());
    }

    @Test
    public void stringAndTypedContentBehindATruthinessCheck() {
        TemplateCaps c = caps("{% for message in messages %}{% if message['content'] %}"
                + "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
                + "{% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endif %}{% endif %}"
                + "{% endfor %}");
        assertTrue(c.supportsStringContent());
        assertTrue(c.supportsTypedContent());
    }

    // ------------------------------------------------------------------ single probes

    @Test
    public void theSystemRoleNeedsTheSystemContentRead() {
        assertTrue(caps("{% for m in messages %}{{ m.content }}{% endfor %}").supportsSystemRole());
        assertFalse(caps("{% if messages[0].role == 'system' %}{{ raise_exception('no') }}{% endif %}"
                + "{% for m in messages %}{{ m.content }}{% endfor %}").supportsSystemRole());
    }

    @Test
    public void withSystemRoleSetsOnlyTheSystemRole() {
        TemplateCaps dropping = caps("{% if messages[0].role == 'system' %}{{ raise_exception('no') }}{% endif %}"
                + "{% for m in messages %}{{ m.content }}{% endfor %}");
        TemplateCaps forced = dropping.withSystemRole();
        assertTrue(forced.supportsSystemRole());
        assertFalse("the probed caps stay as they are", dropping.supportsSystemRole());
        Map<String, Object> expected = dropping.toMap();
        expected.put("supports_system_role", true);
        assertEquals(expected, forced.toMap());
        TemplateCaps rendering = caps("{% for m in messages %}{{ m.content }}{% endfor %}");
        assertSame(rendering, rendering.withSystemRole());
    }

    @Test
    public void aBuiltInTemplateKeepsItsProbedCapsWithTheSystemRole() {
        for (String name : ChatTemplateSource.names()) {
            String source = ChatTemplateSource.resolve(name);
            Map<String, Object> expected = caps(source).toMap();
            expected.put("supports_system_role", true);
            assertEquals(name, expected, new ChatFormatter(source, "", "", false, false, "llama").caps().toMap());
        }
    }

    @Test
    public void toolsAndToolCallsAreSeparateCapabilities() {
        TemplateCaps toolsOnly = caps("{% for t in tools %}{{ t.function.name }}{% endfor %}"
                + "{% for m in messages %}{{ m.content }}{% endfor %}");
        assertTrue(toolsOnly.supportsTools());
        assertFalse(toolsOnly.supportsToolCalls());
        TemplateCaps callsOnly = caps("{% for m in messages %}{{ m.content }}{% for tc in m.tool_calls %}"
                + "{{ tc.function.name }}({{ tc.function.arguments }}){% endfor %}{% endfor %}");
        assertFalse(callsOnly.supportsTools());
        assertTrue(callsOnly.supportsToolCalls());
        assertFalse("the arguments print as given", callsOnly.supportsObjectArguments());
        assertTrue(callsOnly.supportsParallelToolCalls());
    }

    @Test
    public void aTemplateReadingOnlyTheFirstCallDoesNotSupportParallelCalls() {
        TemplateCaps c = caps("{% for m in messages %}{% if m.tool_calls %}"
                + "{{ m.tool_calls[0].function.name }}{% endif %}{% endfor %}");
        assertTrue(c.supportsToolCalls());
        assertFalse(c.supportsParallelToolCalls());
    }

    @Test
    public void theFlatToolCallKeysCountLikeTheNestedOnes() {
        TemplateCaps c = caps("{% for m in messages %}{% for tc in m.tool_calls %}{{ tc.name }}"
                + "{{ tc.arguments | tojson }}{% endfor %}{% endfor %}");
        assertTrue(c.supportsToolCalls());
        assertTrue(c.supportsObjectArguments());
        assertTrue(c.supportsParallelToolCalls());
    }

    @Test
    public void aTemplateRequiringReasoningOnAssistantTurnsIsRetriedWithEmptyReasoning() {
        TemplateCaps c = caps("{% for m in messages %}{% if m.role == 'assistant' and m.reasoning_content is not"
                + " defined %}{{ raise_exception('reasoning required') }}{% endif %}{% for tc in m.tool_calls %}"
                + "{{ tc.function.arguments | tojson }}{% endfor %}{% endfor %}");
        assertTrue(c.supportsToolCalls());
        assertTrue(c.supportsObjectArguments());
        assertTrue(c.supportsParallelToolCalls());
    }

    @Test
    public void preserveReasoningNeedsEarlierReasoningInTheOutput() {
        String history = "{% for m in messages %}{% if m.reasoning_content and preserve_thinking %}"
                + "<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}{% endfor %}";
        assertTrue(caps(history).supportsPreserveReasoning());
        String lastOnly = "{% for m in messages %}{% if m.reasoning_content and loop.last %}"
                + "<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}{% endfor %}";
        assertFalse(caps(lastOnly).supportsPreserveReasoning());
    }

    @Test
    public void reasoningEffortLevelsKeepTheHighestOfLevelsThatRenderAlike() {
        TemplateCaps c = caps("{% if reasoning_effort == 'high' or reasoning_effort == 'max' %}H"
                + "{% elif reasoning_effort in ['low', 'minimal'] %}L{% else %}M{% endif %}");
        assertTrue(c.supportsReasoningEffort());
        assertEquals(Arrays.asList("low", "max"), c.reasoningEffortLevels());
        assertTrue("reasoning_strength is the same value",
                caps("{{ reasoning_strength }}").supportsReasoningEffort());
        TemplateCaps none = caps("{% for m in messages %}{{ m.content }}{% endfor %}");
        assertFalse(none.supportsReasoningEffort());
        assertEquals(Collections.emptyList(), none.reasoningEffortLevels());
    }

    @Test
    public void theMapCarriesLlamaCppsKeys() {
        Map<String, Object> map = caps(ChatTemplateSource.CHATML).toMap();
        assertEquals(Arrays.asList("supports_object_arguments", "supports_parallel_tool_calls",
                "supports_preserve_reasoning", "supports_reasoning_effort", "supports_string_content",
                "supports_system_role", "supports_tool_calls", "supports_tools", "supports_typed_content",
                "reasoning_effort_levels"), Arrays.asList(map.keySet().toArray()));
        assertEquals(Boolean.TRUE, map.get("supports_string_content"));
        assertEquals(Boolean.FALSE, map.get("supports_tools"));
        assertTrue(map.get("reasoning_effort_levels") instanceof List);
    }

    @Test
    public void aNullTemplateIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> TemplateCaps.probe(null));
    }
}
