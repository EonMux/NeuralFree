package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.EnumMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Template dispatch and streaming splits mirror llama.cpp's chat parsers (tests/test-chat.cpp): Muse Spark and
 * Muse Glimmer recipients, Apriel 1.6 forced-open reasoning, and OpenAI harmony channels, with the raw harmony
 * variant of {@code reasoning_format none} (reasoning inline in the content, tool calls still routed).
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
    private static final String GPT_OSS_TEMPLATE = "{{ '<|start|>assistant<|channel|>final<|message|>' }}";
    private static final String GPT_OSS_PROMPT = "<|start|>user<|message|>Hi<|end|><|start|>assistant";

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

    // ------------------------------------------------------------------ raw harmony (reasoning_format none)

    @Test
    public void rawGptOssKeepsTheAnalysisMessageInlineInTheContent() {
        // llama.cpp tests/test-chat.cpp gpt-oss: reasoning_format NONE expects exactly this content
        ReasoningFilter f = ReasoningFilter.forChatTemplate(GPT_OSS_TEMPLATE).asRaw();
        String output = "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
                + "<|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?";
        for (int chunk : new int[] {1, 3, 7, output.length()}) {
            assertSplit(run(f, GPT_OSS_PROMPT, output, chunk), "",
                    "<|channel|>analysis<|message|>I'm\nthinking<|end|>Hello, world!\nWhat's up?", "");
        }
    }

    @Test
    public void rawGptOssStillRoutesToolCalls() {
        ReasoningFilter f = ReasoningFilter.forChatTemplate(GPT_OSS_TEMPLATE).asRaw();
        String output = "<|channel|>analysis<|message|>Need the weather.<|end|>"
                + "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
                + "<|message|>{\"city\": \"Paris\"}<|call|>";
        StringBuilder recipients = new StringBuilder();
        Map<ReasoningFilter.Channel, StringBuilder> acc = new EnumMap<>(ReasoningFilter.Channel.class);
        for (ReasoningFilter.Channel c : ReasoningFilter.Channel.values()) acc.put(c, new StringBuilder());
        ReasoningFilter.State state = f.newState(new ReasoningFilter.Sink() {
            @Override public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                acc.get(channel).append(text);
            }

            @Override public void toolCallStarted(String recipient) {
                recipients.append(recipient).append(';');
            }
        }, GPT_OSS_PROMPT);
        for (int i = 0; i < output.length(); i += 2) {
            state.onToken(i, output.substring(i, Math.min(output.length(), i + 2)));
        }
        state.finish();
        assertEquals("functions.get_weather;", recipients.toString());
        assertEquals("<|channel|>analysis<|message|>Need the weather.<|end|>",
                acc.get(ReasoningFilter.Channel.CONTENT).toString());
        assertEquals("{\"city\": \"Paris\"}", acc.get(ReasoningFilter.Channel.TOOL_CALL).toString());
        assertEquals("", acc.get(ReasoningFilter.Channel.REASONING).toString());
    }

    @Test
    public void rawGptOssDropsThePreambleAndFinalHeadersButKeepsAStrayCommentaryPrefix() {
        // llama.cpp: preamble and final messages are content without headers; the analysis rule wraps the
        // optional "<|channel|>commentary to=assistant" prefix into the content with the analysis header
        ReasoningFilter f = ReasoningFilter.forChatTemplate(GPT_OSS_TEMPLATE).asRaw();
        String output = "<|channel|>commentary<|message|>Working.<|end|>"
                + "<|start|>assistant<|channel|>commentary to=assistant<|channel|>analysis<|message|>T<|end|>"
                + "<|start|>assistant<|channel|>final<|message|>Done.";
        assertSplit(run(f, GPT_OSS_PROMPT, output, 5), "",
                "Working.<|channel|>commentary to=assistant<|channel|>analysis<|message|>T<|end|>Done.", "");
    }

    @Test
    public void rawGptOssMatchesTagTokensById() {
        ReasoningFilter f = ReasoningFilter.forChatTemplate(GPT_OSS_TEMPLATE).withTagTokens(t -> {
            switch (t) {
                case "<|channel|>": return 800;
                case "<|message|>": return 801;
                case "<|end|>": return 802;
                case "<|start|>": return 803;
                default: return -1;
            }
        }).asRaw();
        assertTrue(f.reasoningInline());
        StringBuilder content = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        ReasoningFilter.State state = f.newState((channel, id, text) -> {
            if (channel == ReasoningFilter.Channel.CONTENT) content.append(text);
            if (channel == ReasoningFilter.Channel.REASONING) reasoning.append(text);
        }, GPT_OSS_PROMPT);
        state.onToken(800, "");
        state.onToken(1, "analysis");
        state.onToken(801, "");
        state.onToken(2, "T");
        assertTrue("an inline analysis body is still reasoning for budgets and gates", state.inReasoning());
        state.onToken(802, "");
        assertFalse(state.inReasoning());
        state.onToken(803, "");
        state.onToken(3, "assistant");
        state.onToken(800, "");
        state.onToken(4, "final");
        state.onToken(801, "");
        state.onToken(5, "A");
        state.finish();
        assertEquals("<|channel|>analysis<|message|>T<|end|>A", content.toString());
        assertEquals("", reasoning.toString());
    }

    @Test
    public void rawMuseRoutesTheSelfBodyToContent() {
        // llama.cpp muse-spark.cpp with extract_reasoning false: the to=self body becomes content, its header and
        // terminator are consumed
        ReasoningFilter f = ReasoningFilter.forChatTemplate(MUSE_SPARK_TEMPLATE).asRaw();
        String output = " to=self<|message|>Thinking.<|eom|>"
                + "<|start|>assistant to=special_function<|message|><atem:function_calls></atem:function_calls>";
        assertSplit(run(f, MUSE_PROMPT, output, 3), "", "Thinking.", "<atem:function_calls></atem:function_calls>");
    }

    @Test
    public void asRawKeepsHarmonyParsingAndFlattensTagFilters() {
        ReasoningFilter harmony = ReasoningFilter.forChatTemplate(GPT_OSS_TEMPLATE);
        assertFalse(harmony.reasoningInline());
        ReasoningFilter raw = harmony.withTagTokens(t -> "<|end|>".equals(t) ? 7 : -1).asRaw();
        assertEquals(ReasoningFilter.Mode.HARMONY_CHANNELS, raw.mode());
        assertTrue(raw.reasoningInline());
        assertEquals("<|end|>", raw.tagTokens().get(7));
        ReasoningFilter textOnly = raw.withoutTagTokens();
        assertTrue("dropping the token ids keeps the raw variant", textOnly.reasoningInline());
        assertTrue(textOnly.tagTokens().isEmpty());
        ReasoningFilter think = ReasoningFilter.thinkTags().asRaw();
        assertEquals(ReasoningFilter.Mode.NONE, think.mode());
        assertFalse(think.reasoningInline());
    }
}
