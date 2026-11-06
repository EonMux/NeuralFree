package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.chat.tools.ToolTestSupport.accepts;
import static com.eonmux.neuralfree.gguf.chat.tools.ToolTestSupport.chat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Tool routing per {@code tool_choice}: {@code auto} reads the template's format, {@code required} and a named
 * function force a grammar the output is read back from (gated after the reasoning end tag when the prompt leaves
 * reasoning open), harmony recipients become calls, and a single-call {@code parallel_tool_calls} — explicit or
 * the template's default — keeps one call.
 */
public class ToolRoutingTest {

    private static final String HERMES_TEMPLATE = "{% for t in tools %}<tool_call>{{ t }}</tool_call>{% endfor %}";
    private static final ReasoningFilter NO_REASONING = ReasoningFilter.none();

    @Test
    public void autoReadsTheTemplatesFormatAndNoneReadsNothing() {
        ToolRouting auto = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "auto", HERMES_TEMPLATE,
                NO_REASONING, null, ParallelToolCalls.PARALLEL);
        assertTrue(auto.active());
        assertNull(auto.constraint());
        ToolParse p = run(auto, "Sure.<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>");
        assertEquals("Sure.", p.content());
        assertEquals("noop", p.calls().get(0).name());
        assertTrue(p.calls().get(0).id().startsWith("call_"));

        assertFalse(ToolRouting.plan(chat(ChatRequest.ToolChoice.none()), "auto", HERMES_TEMPLATE, NO_REASONING,
                null, ParallelToolCalls.PARALLEL).active());
        assertFalse("the none parser leaves calls in the content", ToolRouting.plan(
                chat(ChatRequest.ToolChoice.auto()), "none", HERMES_TEMPLATE, NO_REASONING, null,
                ParallelToolCalls.PARALLEL).active());
        assertFalse("an undetected template reads nothing", ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()),
                "auto", "{{ messages }}", NO_REASONING, null, ParallelToolCalls.PARALLEL).active());
        assertThrows(IllegalArgumentException.class, () -> ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()),
                "no_such_parser", HERMES_TEMPLATE, NO_REASONING, null, ParallelToolCalls.PARALLEL));
    }

    @Test
    public void requiredForcesTheFormatsNativeCalls() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "auto", HERMES_TEMPLATE,
                NO_REASONING, null, ParallelToolCalls.PARALLEL);
        assertNotNull(r.constraint());
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertFalse(c.gated());
        String good = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>"
                + "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>";
        assertTrue(accepts(c, good));
        assertFalse("an undeclared function is refused", accepts(c,
                "<tool_call>{\"name\": \"other\", \"arguments\": {}}</tool_call>"));
        assertFalse("free text is refused", accepts(c, "I will not call a tool."));
        ToolParse p = run(r, good);
        assertEquals(2, p.calls().size());
        assertEquals("{\"city\": \"Paris\"}", p.calls().get(0).arguments());

        ToolRouting single = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "auto", HERMES_TEMPLATE,
                NO_REASONING, null, ParallelToolCalls.SINGLE);
        assertFalse("parallel_tool_calls false allows one call",
                accepts(GbnfBuilder.compile(single.constraint()), good));
    }

    @Test
    public void requiredWithoutAKnownFormatForcesAJsonArrayOfToolCalls() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "auto", "{{ messages }}",
                NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertFalse(c.gated());
        String good = "[{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}, "
                + "{\"name\": \"noop\", \"parameters\": {}}]";
        assertTrue(accepts(c, good));
        assertFalse("an undeclared function is refused", accepts(c, "[{\"name\": \"other\", \"parameters\": {}}]"));
        assertFalse("an empty array is refused", accepts(c, "[]"));
        assertFalse("free text is refused", accepts(c, "I will not call a tool."));
        ToolParse p = run(r, good);
        assertEquals(2, p.calls().size());
        assertEquals("{\"city\": \"Paris\"}", p.calls().get(0).arguments());

        ToolRouting single = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "auto", "{{ messages }}",
                NO_REASONING, null, ParallelToolCalls.SINGLE);
        assertFalse("parallel_tool_calls false allows one item",
                accepts(GbnfBuilder.compile(single.constraint()), good));
    }

    @Test
    public void aNamedFunctionForcesItsNativeCall() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.function("get_weather")), "auto",
                HERMES_TEMPLATE, NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertTrue(accepts(c, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}"
                + "</tool_call>"));
        assertFalse(accepts(c, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"town\": \"Oslo\"}}"
                + "</tool_call>"));
        ToolParse p = run(r, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}"
                + "</tool_call>");
        assertEquals("", p.content());
        assertEquals("get_weather", p.calls().get(0).name());
        assertEquals("{\"city\": \"Oslo\"}", p.calls().get(0).arguments());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> ToolRouting.plan(
                chat(ChatRequest.ToolChoice.function("missing")), "auto", HERMES_TEMPLATE, NO_REASONING, null,
                ParallelToolCalls.PARALLEL));
        assertTrue(e.getMessage(), e.getMessage().contains("missing"));

        ToolRouting generic = ToolRouting.plan(chat(ChatRequest.ToolChoice.function("get_weather")), "auto",
                "{{ messages }}", NO_REASONING, null, ParallelToolCalls.PARALLEL);
        assertTrue("an unknown format keeps the parameter object", accepts(
                GbnfBuilder.compile(generic.constraint()), "{\"city\": \"Oslo\"}"));
    }

    @Test
    public void anOpenReasoningPromptGatesTheForcingGrammarOnTheEndTag() {
        ReasoningFilter think = ReasoningFilter.thinkTags();
        ToolRouting open = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "auto", HERMES_TEMPLATE,
                think, "<|im_start|>assistant\n<think>\n", ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(open.constraint());
        assertTrue(c.gated());
        assertTrue("reasoning is free, then the calls follow the end tag",
                accepts(c, "Let me think.</think>\n\n<tool_call>{\"name\": \"noop\", \"arguments\": {}}"
                        + "</tool_call>"));
        assertFalse(accepts(c, "Let me think.</think>\n\nNo tools today."));

        ToolRouting closed = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "auto", HERMES_TEMPLATE,
                think, "<|im_start|>assistant\n<think>\n\n</think>\n\n", ParallelToolCalls.PARALLEL);
        assertFalse("a closed reasoning block constrains from the start",
                GbnfBuilder.compile(closed.constraint()).gated());
    }

    @Test
    public void harmonyRecipientsBecomeCallsAndBuiltInsAreDropped() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "auto", null,
                ReasoningFilter.harmony(), null, ParallelToolCalls.PARALLEL);
        assertTrue(r.active());
        assertNull("harmony takes no forcing grammar", r.constraint());
        ToolOutput out = r.newOutput(() -> "call_x");
        ToolTestSupport.Recorder rec = new ToolTestSupport.Recorder();
        out.content("Checking. ", rec);
        out.toolCallStarted("browser.search", rec);
        out.toolCallText("{\"query\": \"weather\"}", rec);
        out.toolCallStarted("functions.get_weather", rec);
        out.toolCallText("{\"city\":", rec);
        out.toolCallText(" \"Paris\"}", rec);
        ToolParse p = out.finish(rec);
        assertEquals("Checking. ", p.content());
        assertEquals(1, p.calls().size());
        assertEquals("get_weather", p.calls().get(0).name());
        assertEquals("{\"city\": \"Paris\"}", p.calls().get(0).arguments());
        assertEquals("call_x", p.calls().get(0).id());
        assertEquals("{\"city\": \"Paris\"}", rec.args.get(0).toString());
    }

    @Test
    public void parallelToolCallsFalseKeepsTheFirstCall() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "hermes", null, NO_REASONING, null,
                ParallelToolCalls.SINGLE);
        ToolTestSupport.Recorder rec = new ToolTestSupport.Recorder();
        ToolOutput out = r.newOutput();
        out.content("<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>"
                + "<tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>", rec);
        ToolParse p = out.finish(rec);
        assertEquals(1, p.calls().size());
        assertEquals("a", p.calls().get(0).name());
        assertEquals(Collections.singletonList("a"), rec.names);

        // the template's one-call default (llama.cpp server-common.cpp:1447) keeps the first call without asking
        ToolRouting defaulted = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "hermes", null, NO_REASONING,
                null, ParallelToolCalls.of(null, false));
        ToolTestSupport.Recorder dropped = new ToolTestSupport.Recorder();
        ToolOutput defaultedOut = defaulted.newOutput();
        defaultedOut.content("<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>"
                + "<tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>", dropped);
        ToolParse single = defaultedOut.finish(dropped);
        assertEquals(1, single.calls().size());
        assertEquals("a", single.calls().get(0).name());
    }

    @Test
    public void theHarmonyFilterAnnouncesToolRecipients() {
        List<String> events = new ArrayList<>();
        ReasoningFilter.State st = ReasoningFilter.harmony().newState(new ReasoningFilter.Sink() {
            @Override
            public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                events.add(channel + ":" + text);
            }

            @Override
            public void toolCallStarted(String recipient) {
                events.add("call:" + recipient);
            }
        });
        String out = "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>commentary"
                + " to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Rome\"}<|call|>";
        for (int i = 0; i < out.length(); i += 5) {
            st.onToken(i, out.substring(i, Math.min(out.length(), i + 5)));
        }
        st.finish();
        int call = events.indexOf("call:functions.get_weather");
        assertTrue(events.toString(), call >= 0);
        StringBuilder body = new StringBuilder();
        for (String e : events.subList(call + 1, events.size())) {
            assertTrue(e, e.startsWith("TOOL_CALL:"));
            body.append(e.substring("TOOL_CALL:".length()));
        }
        assertEquals("{\"city\":\"Rome\"}", body.toString());
    }

    // ------------------------------------------------------------------ helpers

    private static ToolParse run(ToolRouting r, String text) {
        ToolOutput out = r.newOutput();
        ToolTestSupport.Recorder rec = new ToolTestSupport.Recorder();
        for (int i = 0; i < text.length(); i += 4) {
            out.content(text.substring(i, Math.min(text.length(), i + 4)), rec);
        }
        ToolParse p = out.finish(rec);
        assertEquals("streamed content is the final content", p.content(), rec.content.toString());
        for (int i = 0; i < p.calls().size(); i++) {
            assertEquals("streamed arguments are the final arguments", p.calls().get(i).arguments(),
                    rec.args.get(i).toString());
        }
        return p;
    }
}
