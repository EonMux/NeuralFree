package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Chat template resolution, the system-role workaround and the fallback formats follow llama.cpp
 * (common/chat.cpp common_chat_templates_init and common_chat_templates_apply_jinja).
 */
public class ChatFormatterTest {

    /** llama.cpp models/templates/google-gemma-2-2b-it.jinja. */
    private static final String GEMMA2 = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
            + "{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}"
            + "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            + "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
            + "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}"
            + "{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
            + "{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}";

    private static final String ROLES = "{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}{% endfor %}";

    private static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    private static ChatFormatter formatter(String source, String arch) {
        return new ChatFormatter(source, "<bos>", "<eos>", true, false, arch);
    }

    // ------------------------------------------------------------------ template resolution

    @Test
    public void missingEmptyOrChatmlTemplateResolvesToToolUseOrChatMl() {
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.resolve(null, null));
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.resolve("", null));
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.resolve("chatml", ""));
        assertEquals("TOOLS", ChatTemplateSource.resolve(null, "TOOLS"));
        assertEquals("TOOLS", ChatTemplateSource.resolve("chatml", "TOOLS"));
        assertEquals("OWN", ChatTemplateSource.resolve("OWN", "TOOLS"));
    }

    @Test
    public void chatMlRendersEveryMessageAndTheGenerationPrompt() {
        ChatFormatter f = formatter(ChatTemplateSource.resolve(null, null), "llama");
        List<Map<String, String>> conv = Arrays.asList(msg("system", "S"), msg("user", "U"),
                msg("assistant", "A"), msg("tool", "T"), msg("user", "V"));
        assertEquals("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
                + "<|im_start|>assistant\nA<|im_end|>\n<|im_start|>tool\nT<|im_end|>\n"
                + "<|im_start|>user\nV<|im_end|>\n<|im_start|>assistant\n", f.format(conv, false));
    }

    @Test
    public void llamaCppSourcePatchesDisableTheRejectingChecks() {
        String harmony = "<|channel|>{%- if \"<|channel|>analysis<|message|>\" in message.content or "
                + "\"<|channel|>final<|message|>\" in message.content %}X";
        assertEquals("<|channel|>{%- if false %}X", ChatTemplateSource.resolve(harmony, null));
        String mistral = "[TOOL_CALLS]{%- if (message['content'] is none or message['content'] == '' or "
                + "message['content']|length == 0) and (message['tool_calls'] is not defined or "
                + "message['tool_calls'] is none or message['tool_calls']|length == 0) %}Y";
        assertEquals("[TOOL_CALLS]{%- if false %}Y", ChatTemplateSource.resolve(mistral, null));
    }

    // ------------------------------------------------------------------ system role

    @Test
    public void gemmaMergesTheSystemPromptIntoTheFirstUserMessage() {
        ChatFormatter f = formatter(GEMMA2, "gemma2");
        assertFalse(f.supportsSystemRole());
        List<Map<String, String>> conv = Arrays.asList(msg("system", "Be brief."), msg("user", "Hi"));
        assertEquals("<start_of_turn>user\nBe brief.\nHi<end_of_turn>\n<start_of_turn>model\n", f.format(conv, false));
    }

    @Test
    public void gemmaMultiTurnKeepsTurnMarkers() {
        ChatFormatter f = formatter(GEMMA2, "gemma2");
        List<Map<String, String>> conv = Arrays.asList(msg("system", "S"), msg("user", "U1"),
                msg("assistant", "A1"), msg("user", "U2"));
        assertEquals("<start_of_turn>user\nS\nU1<end_of_turn>\n<start_of_turn>model\nA1<end_of_turn>\n"
                + "<start_of_turn>user\nU2<end_of_turn>\n<start_of_turn>model\n", f.format(conv, false));
    }

    @Test
    public void aLoneSystemMessageIsDroppedForTemplatesWithoutSystemRole() {
        ChatFormatter f = formatter(ROLES.replace("{% for", "{% if messages and messages[0]['role'] == 'system' %}"
                + "{{ raise_exception('no system') }}{% endif %}{% for"), "llama");
        assertFalse(f.supportsSystemRole());
        assertEquals("", f.format(Collections.singletonList(msg("system", "S")), false));
    }

    @Test
    public void templatesThatRenderTheSystemTextSupportTheSystemRole() {
        ChatFormatter f = formatter(ROLES, "llama");
        assertTrue(f.supportsSystemRole());
        assertEquals("[system]S[user]U", f.format(Arrays.asList(msg("system", "S"), msg("user", "U")), false));
    }

    @Test
    public void developerMapsToSystemExceptForHarmonyTemplates() {
        List<Map<String, String>> conv = Arrays.asList(msg("developer", "D"), msg("user", "U"));
        assertEquals("[system]D[user]U", formatter(ROLES, "llama").format(conv, false));
        String harmony = ROLES + "{# <|channel|> #}";
        assertEquals("[developer]D[user]U", formatter(harmony, "llama").format(conv, false));
    }

    @Test
    public void callerMessagesAreNotModified() {
        List<Map<String, String>> conv = new ArrayList<>(Arrays.asList(msg("system", "S"), msg("user", "U")));
        formatter(GEMMA2, "gemma2").format(conv, false);
        assertEquals(2, conv.size());
        assertEquals("S", conv.get(0).get("content"));
        assertEquals("U", conv.get(1).get("content"));
    }

    @Test
    public void stepFunTemplatesTrimMessageContent() {
        String stepFun = ROLES + "{# You have access to the following functions in JSONSchema format #}";
        assertEquals("[user]U", formatter(stepFun, "llama").format(Collections.singletonList(msg("user", "  U\n")), false));
    }

    @Test
    public void stepFunTemplatesTrimTypedTextParts() {
        String stepFun = "{% for m in messages %}{% for p in m['content'] %}({{ p['text'] }}){% endfor %}{% endfor %}"
                + "{# You have access to the following functions in JSONSchema format #}";
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "text");
        part.put("text", " T\t");
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", "user");
        m.put("content", Collections.singletonList(part));
        assertEquals("(T)", formatter(stepFun, "llama").format(Collections.singletonList(m), false));
        assertEquals(" T\t", part.get("text"));
    }

    @Test
    public void templatesSeeLlamaCppDateVariables() {
        String out = formatter("{{ date_string }}|{{ datetime }}", "llama")
                .format(Collections.singletonList(msg("user", "U")), false);
        assertTrue(out, out.matches("\\d{2} [A-Z][a-z]{2} \\d{4}\\|[A-Z][a-z]{2} \\d{2} \\d{4}"));
    }

    // ------------------------------------------------------------------ fallbacks

    @Test
    public void aFailingTemplateFallsBackToChatMl() {
        ChatFormatter f = formatter("{{ raise_exception('unsupported') }}", "llama");
        assertEquals("<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n",
                f.format(Collections.singletonList(msg("user", "U")), false));
    }

    @Test
    public void aTemplateThatDoesNotParseFallsBackToChatMl() {
        ChatFormatter f = formatter("{{ messages[ }}", "llama");
        assertEquals("<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n",
                f.format(Collections.singletonList(msg("user", "U")), false));
    }

    @Test
    public void gptOssFallsBackToHarmonyForSingleAndMultiTurn() {
        ChatFormatter f = formatter("{{ raise_exception('unsupported') }}", "gpt-oss");
        String single = f.format(Arrays.asList(msg("system", "S"), msg("user", "U")), false);
        assertTrue(single, single.contains("<|start|>developer<|message|># Instructions\n\nS<|end|>"));
        assertTrue(single, single.endsWith("<|start|>user<|message|>U<|end|><|start|>assistant"));
        String multi = f.format(Arrays.asList(msg("system", "S"), msg("user", "U"), msg("assistant", "A"),
                msg("user", "V")), false);
        assertTrue(multi, multi.endsWith("<|start|>user<|message|>U<|end|>"
                + "<|start|>assistant<|channel|>final<|message|>A<|end|>"
                + "<|start|>user<|message|>V<|end|><|start|>assistant"));
    }

    @Test
    public void harmonyFallbackUsesDefaultInstructionsWithoutSystemMessage() {
        List<Map<String, Object>> conv = ChatMessages.copyOf(Collections.singletonList(msg("user", "U")));
        String out = HarmonyFallback.format(conv, LocalDate.of(2026, 1, 2));
        assertTrue(out, out.contains("Current date: 2026-01-02\n"));
        assertTrue(out, out.contains("# Instructions\n\nYou are a helpful assistant.<|end|>"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void messagesWithoutRoleAreRejected() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("content", "x");
        formatter(ROLES, "llama").format(Collections.singletonList(m), false);
    }
}
