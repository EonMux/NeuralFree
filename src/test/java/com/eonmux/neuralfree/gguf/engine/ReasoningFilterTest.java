package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.EnumMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

/**
 * Template dispatch and streaming splits mirror llama.cpp's chat parsers (tests/test-chat.cpp): Muse Spark and
 * Muse Glimmer recipients, Apriel 1.6 forced-open reasoning, and OpenAI harmony channels.
 */
public class ReasoningFilterTest {

    private static final String MUSE_SPARK_TEMPLATE =
            "{{- '<|start|>assistant to=commentary<|message|>' + m.content + '<|eom|>' -}}"
            + "{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}";
    private static final String MUSE_GLIMMER_TEMPLATE =
            "{{- '<atem:function_calls>' -}}{{- '<|eom|>' -}}"
            + "{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}";
    private static final String MUSE_PROMPT = "<|start|>user<|message|>Hi<|eot|><|start|>assistant";

    private static final String APRIEL_16_TEMPLATE =
            "{%- set reasoning_asst_turn_start = 'Here are my reasoning steps:\\n' -%}"
            + "{{ message['reasoning_content'] + '\\n[BEGIN FINAL RESPONSE]\\n' }}"
            + "{%- if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text -%}{%- endif -%}"
            + "{{ '\\n<|begin_assistant|>\\n' + reasoning_asst_turn_start }}";
    private static final String APRIEL_15_TEMPLATE =
            "provide the final solution in the following format: [BEGIN FINAL RESPONSE] ... [END FINAL RESPONSE].";
    private static final String APRIEL_PROMPT = "<|begin_user|>\nHi\n<|begin_assistant|>\nHere are my reasoning steps:\n";

    /** Text routed per channel. */
    private static Map<ReasoningFilter.Channel, String> run(ReasoningFilter f, String prompt, String output, int chunk) {
        Map<ReasoningFilter.Channel, StringBuilder> acc = new EnumMap<>(ReasoningFilter.Channel.class);
        for (ReasoningFilter.Channel c : ReasoningFilter.Channel.values()) acc.put(c, new StringBuilder());
        ReasoningFilter.State state = f.newState((channel, id, text) -> acc.get(channel).append(text), prompt);
        for (int i = 0; i < output.length(); i += chunk) {
            state.onToken(i, output.substring(i, Math.min(output.length(), i + chunk)));
        }
        state.finish();
        Map<ReasoningFilter.Channel, String> out = new EnumMap<>(ReasoningFilter.Channel.class);
        for (Map.Entry<ReasoningFilter.Channel, StringBuilder> e : acc.entrySet()) out.put(e.getKey(), e.getValue().toString());
        return out;
    }

    private static void assertSplit(Map<ReasoningFilter.Channel, String> got, String reasoning, String content, String tool) {
        assertEquals("reasoning", reasoning, got.get(ReasoningFilter.Channel.REASONING));
        assertEquals("content", content, got.get(ReasoningFilter.Channel.CONTENT));
        assertEquals("tool call", tool, got.get(ReasoningFilter.Channel.TOOL_CALL));
    }

    // ------------------------------------------------------------------ Muse

    @Test
    public void museTemplatesSelectTheHeaderBodyMode() {
        assertEquals(ReasoningFilter.Mode.HARMONY_CHANNELS, ReasoningFilter.forChatTemplate(MUSE_SPARK_TEMPLATE).mode());
        assertEquals(ReasoningFilter.Mode.HARMONY_CHANNELS, ReasoningFilter.forChatTemplate(MUSE_GLIMMER_TEMPLATE).mode());
    }

    @Test
    public void museSparkRoutesSelfToReasoningAndCommentaryAndUserToContent() {
        String output = " to=self<|message|>Thinking.<|eom|>"
                + "<|start|>assistant to=commentary<|message|>Working on it.<|eom|>"
                + "<|start|>assistant to=user<|message|>Hello!<|eot|>";
        ReasoningFilter f = ReasoningFilter.forChatTemplate(MUSE_SPARK_TEMPLATE);
        for (int chunk : new int[] {1, 3, 7, output.length()}) {
            assertSplit(run(f, MUSE_PROMPT, output, chunk), "Thinking.", "Working on it.Hello!", "");
        }
    }

    @Test
    public void musePlainAnswerIsContent() {
        ReasoningFilter f = ReasoningFilter.forChatTemplate(MUSE_GLIMMER_TEMPLATE);
        assertSplit(run(f, MUSE_PROMPT, " to=user<|message|>Hello, world!\nWhat's up?<|eot|>", 2),
                "", "Hello, world!\nWhat's up?", "");
        assertSplit(run(f, MUSE_PROMPT, "<|message|>Hi", 1), "", "Hi", "");
    }

    @Test
    public void museOtherRecipientsAreToolCalls() {
        ReasoningFilter f = ReasoningFilter.forChatTemplate(MUSE_SPARK_TEMPLATE);
        String output = " to=commentary<|message|>Let me look that up.<|eom|>"
                + "<|start|>assistant to=special_function<|message|><atem:function_calls></atem:function_calls>";
        assertSplit(run(f, MUSE_PROMPT, output, 4), "", "Let me look that up.",
                "<atem:function_calls></atem:function_calls>");
    }

    @Test
    public void museTerminatorsAreRecognizedByTokenId() {
        ReasoningFilter f = ReasoningFilter.forChatTemplate(MUSE_SPARK_TEMPLATE).withTagTokens(t -> {
            switch (t) {
                case "<|eom|>": return 900;
                case "<|message|>": return 901;
                default: return -1;
            }
        });
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State state = f.newState((channel, id, text) -> {
            if (channel == ReasoningFilter.Channel.REASONING) reasoning.append(text);
            if (channel == ReasoningFilter.Channel.CONTENT) content.append(text);
        }, MUSE_PROMPT);
        state.onToken(1, " to=self");
        state.onToken(901, "");
        state.onToken(2, "T");
        state.onToken(900, "");
        state.onToken(3, "<|start|>assistant to=user");
        state.onToken(901, "");
        state.onToken(4, "A");
        state.finish();
        assertEquals("T", reasoning.toString());
        assertEquals("A", content.toString());
    }

    // ------------------------------------------------------------------ Apriel

    @Test
    public void apriel16ReasoningIsForcedOpenUntilTheFinalResponseMarker() {
        ReasoningFilter f = ReasoningFilter.forChatTemplate(APRIEL_16_TEMPLATE);
        assertEquals(ReasoningFilter.Mode.THINK_TAGS, f.mode());
        String output = "I'm\nthinking\n[BEGIN FINAL RESPONSE]\nHello, world!\nWhat's up?";
        for (int chunk : new int[] {1, 5, output.length()}) {
            assertSplit(run(f, APRIEL_PROMPT, output, chunk), "I'm\nthinking", "Hello, world!\nWhat's up?", "");
        }
    }

    @Test
    public void apriel15HasNoReasoning() {
        assertSame(ReasoningFilter.none(), ReasoningFilter.forChatTemplate(APRIEL_15_TEMPLATE));
    }

    // ------------------------------------------------------------------ harmony

    @Test
    public void gptOssAnalysisIsReasoningAndFinalIsContent() {
        ReasoningFilter f = ReasoningFilter.forChatTemplate("{{ '<|start|>assistant<|channel|>final<|message|>' }}");
        String output = "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                + "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?";
        assertSplit(run(f, "<|start|>user<|message|>Hi<|end|><|start|>assistant", output, 3),
                "I'm\nthinking", "Hello, world!\nWhat's up?", "");
    }
}
