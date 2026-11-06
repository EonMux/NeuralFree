package com.eonmux.neuralfree.gguf.chat.reasoning;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The request-dependent and family-specific reasoning parsers, against the outputs of vLLM's tests
 * (tests/reasoning/test_hy_v3_reasoning_parser.py, tests/parser/engine/test_minicpmv_reasoning.py,
 * tests/parser/engine/test_plamo3.py, tests/parser/engine/test_inkling.py): {@code hy_v3} / {@code hy_v4} with
 * their token suffix and {@code reasoning_effort}, {@code granite_thinking_parser} / {@code nemotron_v3},
 * {@code minicpmv}, {@code plamo3}, {@code inkling} and {@code minimax_m2_append_think}. Every case streams one
 * character per token and as one piece, and both must agree.
 */
public class ReasoningParserFamiliesTest {

    private static final Map<String, Object> NO_VARS = Collections.emptyMap();

    // ------------------------------------------------------------------ hy_v3 / hy_v4

    @Test
    public void hyV3IsContentUnlessTheEffortThinks() {
        assertParse("hy_v3", NO_VARS, "This is the rest", "", "This is the rest");
        assertParse("hy_v3", vars("reasoning_effort", "no_think"), "<think>a</think>b", "", "<think>a</think>b");
        Map<String, Object> high = vars("reasoning_effort", "high");
        assertParse("hy_v3", high, "This is a reasoning section</think>This is the rest",
                "This is a reasoning section", "This is the rest");
        assertParse("hy_v3", high, "This is a reasoning\nsection</think>This is the rest\nThat",
                "This is a reasoning\nsection", "This is the rest\nThat");
        assertParse("hy_v3", high, "This is a reasoning section</think>", "This is a reasoning section", "");
        assertParse("hy_v3", high, "This is a reasoning section", "This is a reasoning section", "");
        assertParse("hy_v3", vars("reasoning_effort", "low"), "<think>r</think>c", "r", "c");
    }

    @Test
    public void hyV4ThinksUnlessTheEffortIsNoThink() {
        assertParse("hy_v4", NO_VARS, "plan</think>answer", "plan", "answer");
        assertParse("hy_v4", vars("reasoning_effort", ""), "plan</think>answer", "plan", "answer");
        assertParse("hy_v4", vars("reasoning_effort", "no_think"), "plan</think>answer", "", "plan</think>answer");
    }

    @Test
    public void hyRowsTakeTheVocabularySuffix() {
        GGUFTokenizer tok = tokenizer(Arrays.asList("<tool_calls:6124c78e>", "<think:6124c78e>", "</think:6124c78e>"),
                "{{ messages }}");
        assertEquals(":6124c78e", ReasoningParsers.hunyuanSuffix(tok));
        ReasoningFilter f = ReasoningParsers.forModel("hy_v4", tok);
        int end = tok.getTokenId("</think:6124c78e>");
        assertTrue("the suffixed end tag renders by id", f.tagTokens().containsKey(end));
        assertEquals(Collections.singletonList("<think:6124c78e>"), f.startTags());
        assertEquals(Arrays.asList("plan", "answer"), run(f, NO_VARS, null, "plan</think:6124c78e>answer"));
        ReasoningFilter v3 = ReasoningParsers.forModel("hy_v3", tok).forRequest(vars("reasoning_effort", "high"));
        assertEquals(Collections.singletonList("</think:6124c78e>"), v3.endTags());
        assertTrue("forRequest replays the tag-token derivation", v3.tagTokens().containsKey(end));
        assertEquals("a vocabulary without HY markers has the empty suffix", "",
                ReasoningParsers.hunyuanSuffix(tokenizer(Collections.singletonList("<|im_end|>"), "x")));
    }

    @Test
    public void namedHyToolParsersTakeTheVocabularySuffixOnALoadedModel() {
        GGUFTokenizer tok = tokenizer(Arrays.asList("<tool_calls:6124c78e>", "<tool_call:6124c78e>",
                "<tool_sep:6124c78e>", "<arg_key:6124c78e>", "<arg_value:6124c78e>"), "{{ messages }}");
        String call = "<tool_calls:6124c78e><tool_call:6124c78e>get_weather<tool_sep:6124c78e>"
                + "<arg_key:6124c78e>city</arg_key:6124c78e><arg_value:6124c78e>SF</arg_value:6124c78e>"
                + "</tool_call:6124c78e></tool_calls:6124c78e>";
        ToolParse unsuffixed = ToolParserRegistry.forName("hy_v3").create(null).parse(call, false);
        assertTrue("without a model the name keeps the plain markers", unsuffixed.calls().isEmpty());
        ToolParse p = ToolParserRegistry.forName("hy_v3").forModel(tok).create(null).parse(call, false);
        assertEquals(1, p.calls().size());
        assertEquals("get_weather", p.calls().get(0).name());
        assertTrue(p.calls().get(0).arguments(), p.calls().get(0).arguments().contains("\"SF\""));

        ChatRequest chat = ChatRequest.builder().addMessage(message())
                .tools(Collections.singletonList(com.eonmux.neuralfree.gguf.chat.ChatTool.builder("get_weather")
                        .build()))
                .build();
        com.eonmux.neuralfree.gguf.chat.tools.ToolRouting routing = com.eonmux.neuralfree.gguf.chat.tools.ToolRouting
                .plan(chat, "hy-v4", tok, ReasoningFilter.none(), null,
                        ParallelToolCalls.PARALLEL, com.eonmux.neuralfree.gguf.chat.tools.ToolStrict.AUTO);
        ToolParse v4 = routing.parser().parse("<tool_calls:6124c78e><tool_call:6124c78e>f<arg_key:6124c78e>k"
                + "</arg_key:6124c78e><arg_value:6124c78e>v</arg_value:6124c78e></tool_call:6124c78e>"
                + "</tool_calls:6124c78e>", false);
        assertEquals(1, v4.calls().size());
        assertEquals("f", v4.calls().get(0).name());
    }

    @Test
    public void hunyuanMarkerSuffixesFollowVllmsDetection() {
        assertEquals(":opensource", ReasoningParsers.hunyuanMarkerSuffix("<think:opensource>"));
        assertEquals(":x", ReasoningParsers.hunyuanMarkerSuffix("</arg_value:x>"));
        assertEquals(":x", ReasoningParsers.hunyuanMarkerSuffix("<tool_call:x>"));
        assertEquals(":x", ReasoningParsers.hunyuanMarkerSuffix("<tool_calls:x>"));
        assertEquals("", ReasoningParsers.hunyuanMarkerSuffix("<think>"));
        assertEquals(null, ReasoningParsers.hunyuanMarkerSuffix("<thinking>"));
        assertEquals(null, ReasoningParsers.hunyuanMarkerSuffix("<think:>"));
        assertEquals(null, ReasoningParsers.hunyuanMarkerSuffix("<think: x>"));
        assertEquals(null, ReasoningParsers.hunyuanMarkerSuffix("think"));
    }

    @Test
    public void hyTemplatesSelectSuffixedTagsUnderAuto() {
        String hy3 = "{%- set HYTK = ':opensource' %}\n{%- set think_begin_token = '<think{}>'.format(HYTK) %}"
                + "{%- set think_end_token = '</think{}>'.format(HYTK) %}";
        assertEquals(":opensource", ReasoningFilter.hunyuanTemplateSuffix(hy3));
        ReasoningFilter auto = ReasoningFilter.forChatTemplate(hy3);
        assertEquals(Collections.singletonList("<think:opensource>"), auto.startTags());
        assertEquals(Collections.singletonList("</think:opensource>"), auto.endTags());
        assertEquals(null, ReasoningFilter.hunyuanTemplateSuffix("{{ '<think>' }}"));
    }

    @Test
    public void wrongVariableTypesAreRefusedByName() {
        IllegalArgumentException effort = assertThrows(IllegalArgumentException.class,
                () -> filter("hy_v3").forRequest(vars("reasoning_effort", 3)));
        assertTrue(effort.getMessage(), effort.getMessage().contains("reasoning_effort"));
        IllegalArgumentException thinking = assertThrows(IllegalArgumentException.class,
                () -> filter("minicpmv").forRequest(vars("enable_thinking", "no")));
        assertTrue(thinking.getMessage(), thinking.getMessage().contains("enable_thinking"));
        IllegalArgumentException forced = assertThrows(IllegalArgumentException.class,
                () -> filter("granite_thinking_parser").forRequest(vars("force_nonempty_content", 1)));
        assertTrue(forced.getMessage(), forced.getMessage().contains("force_nonempty_content"));
    }

    // ------------------------------------------------------------------ granite_thinking_parser / nemotron_v3

    @Test
    public void graniteThinkingSplitsQwenTagsAndStripsTheContentsLeadingNewlines() {
        assertParse("granite_thinking_parser", NO_VARS, "reason\n</think>\n\nanswer", "reason", "answer");
        assertParse("granite_thinking_parser", NO_VARS, "<think>reason</think>answer</think> more", "reason",
                "answer more");
        assertParse("granite_thinking_parser", NO_VARS, "reason<tool_call>{}", "reason", "<tool_call>{}");
        assertParse("granite_thinking_parser", NO_VARS, "truncated reasoning", "truncated reasoning", "");
    }

    @Test
    public void thinkingOffParsesNoReasoningAtAll() {
        Map<String, Object> off = vars("enable_thinking", false);
        for (String name : Arrays.asList("granite_thinking_parser", "nemotron_v3")) {
            assertParse(name, off, "plain answer", "", "plain answer");
            // vLLM's engine starts in its content state, which has no transition into reasoning; a stray end tag
            // is dropped there
            assertParse(name, off, "<think>only reasoning</think>", "", "<think>only reasoning");
            assertParse(name, off, "a<think>b</think>c", "", "a<think>bc");
            assertFalse(name, filter(name).forRequest(off).startsInReasoning("<|im_start|>assistant\n<think>\n"));
        }
    }

    @Test
    public void forcedContentMovesAnAnswerlessReasoningToTheContent() {
        assertParse("nemotron_v3", vars("force_nonempty_content", true), "cut short", "cut short", "cut short");
        assertParse("nemotron_v3", vars("force_nonempty_content", true), "r</think>\n\n", "r", "r");
        assertParse("nemotron_v3", vars("force_nonempty_content", true), "r</think>a", "r", "a");
        assertParse("nemotron_v3", NO_VARS, "cut short", "cut short", "");
    }

    @Test
    public void graniteWithThinkingOffStripsTheNewlinesTheContentStartsWith() {
        Map<String, Object> off = vars("enable_thinking", false);
        assertParse("granite_thinking_parser", off, "\n\nplain answer\n", "", "plain answer\n");
        assertParse("granite_thinking_parser", off, "\n \nanswer", "", " \nanswer");
        assertParse("granite_thinking_parser", off, "\n\n", "", "");
        assertParse("nemotron_v3", off, "\n\nplain answer", "", "\n\nplain answer");
    }

    @Test
    public void graniteSwapsOnlyAnAnswerWithNoContentAtAll() {
        Map<String, Object> forced = vars("force_nonempty_content", true);
        assertParse("granite_thinking_parser", forced, "cut short", "cut short", "cut short");
        assertParse("granite_thinking_parser", forced, "r</think>\n\n", "r", "");
        assertParse("nemotron_v3", forced, "r</think>\n\n", "r", "r");
    }

    @Test
    public void theContentFallbackIsAnnouncedRightBeforeItsContent() {
        List<String> events = new ArrayList<>();
        ReasoningFilter.Sink sink = new ReasoningFilter.Sink() {
            @Override
            public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
                events.add(channel + ":" + text);
            }

            @Override
            public void contentFallback() {
                events.add("fallback");
            }
        };
        ReasoningFilter.State st = filter("nemotron_v3").forRequest(vars("force_nonempty_content", true))
                .newState(sink, null);
        st.onToken(0, "cut");
        st.finish();
        assertEquals(Arrays.asList("REASONING:cut", "fallback", "CONTENT:cut"), events);

        events.clear();
        st = filter("nemotron_v3").forRequest(vars("force_nonempty_content", true)).newState(sink, null);
        st.onToken(0, "r</think>a");
        st.finish();
        assertFalse("an answer with content keeps its split", events.contains("fallback"));
        events.clear();
        st = filter("nemotron_v3").newState(sink, null);
        st.onToken(0, "cut");
        st.finish();
        assertEquals(Collections.singletonList("REASONING:cut"), events);
    }

    @Test
    public void implicitReasoningYieldsToAPromptThatClosedTheBlock() {
        ReasoningFilter granite = filter("granite_thinking_parser");
        assertEquals("callers without request variables still follow a template's empty block",
                Arrays.asList("", "answer"), run(granite, NO_VARS, "<|im_start|>assistant\n<think></think>", "answer"));
        assertEquals(Arrays.asList("r", "a"), run(granite, NO_VARS, "<|im_start|>assistant\n<think>\n", "r</think>a"));
        assertEquals(Arrays.asList("r", "a"), run(granite, NO_VARS, "<|im_start|>assistant\n", "r</think>a"));
        assertEquals(Arrays.asList("", "c"), run(filter("hy_v4"), NO_VARS, "x<think></think>\n\n", "c"));
        assertFalse(granite.newState((channel, tokenId, t) -> { }, "a<think>\n\n</think>\n\n").inReasoning());
        assertTrue(granite.newState((channel, tokenId, t) -> { }, "a").inReasoning());
    }

    // ------------------------------------------------------------------ minicpmv

    @Test
    public void minicpmvWithoutThinkingDropsReasoningAndMarkers() {
        Map<String, Object> off = vars("enable_thinking", false);
        assertParse("minicpmv", off, "The answer is 42.", "", "The answer is 42.");
        assertParse("minicpmv", off,
                "<reserved_12>reasoning<reserved_13><|reserved_14|>tool call<|reserved_15|>", "", "reasoningtool call");
        assertParse("minicpmv", off, "<reserved_11>content<|reserved_16|>", "", "<reserved_11>content<|reserved_16|>");
        assertParse("minicpmv", off, "first part </think>\n\nfinal part", "", "first part \n\nfinal part");
        assertParse("minicpmv", off, "prefix<think>private reasoning", "", "prefix");
        assertParse("minicpmv", off, "<think>private reasoning</think>final answer", "", "final answer");
        assertParse("minicpmv", off, "<tool_call><function=f><parameter=city>SF</parameter></function></tool_call>",
                "", "<tool_call><function=f><parameter=city>SF</parameter></function></tool_call>");
    }

    @Test
    public void minicpmvThinksByDefault() {
        assertParse("minicpmv", NO_VARS, "<think>private reasoning</think>final answer", "private reasoning",
                "final answer");
        assertParse("minicpmv", vars("enable_thinking", true),
                "<think><reserved_12>private reasoning</think><|reserved_13|>final answer", "private reasoning",
                "final answer");
        assertParse("minicpmv", NO_VARS, "Answer: 111<|im_end|>", "Answer: 111", "");
    }

    @Test
    public void minicpmvRecoversEscapedNewlinesButNotInCodeOrToolCalls() {
        assertParse("minicpmv", NO_VARS, "reason\\nnext</think>answer\\nend `a\\nb`", "reason\nnext",
                "answer\nend `a\\nb`");
        String call = "<tool_call><function=f><parameter=s>x\\ny</parameter></function></tool_call>";
        assertParse("minicpmv", vars("enable_thinking", false), "a\\nb" + call + "c\\nd", "", "a\nb" + call + "c\nd");
    }

    // ------------------------------------------------------------------ plamo3

    @Test
    public void plamo3ReasonsFromTheStartAndReopens() {
        String begin = ReasoningFilter.PLAMO_BEGIN_THINK;
        String end = ReasoningFilter.PLAMO_END_THINK;
        assertParse("plamo3", NO_VARS, begin + "Reasoning." + end + "Answer.", "Reasoning.", "Answer.");
        assertParse("plamo3", vars("enable_thinking", true), "Answer.", "Answer.", "");
        assertParse("plamo3", NO_VARS, end + "A" + end + "B", "", "AB");
        assertParse("plamo3", NO_VARS, end + "A" + begin + "R" + end, "R", "A");
        assertParse("plamo3", NO_VARS, "R" + ReasoningFilter.PLAMO_TOOL_REQUESTS + "x", "R",
                ReasoningFilter.PLAMO_TOOL_REQUESTS + "x");
        assertParse("plamo3", NO_VARS, "R" + ReasoningFilter.PLAMO_EOT + "after", "R", "after");
    }

    @Test
    public void plamo3DropsAMarkerTheEndOfGenerationCutOff() {
        String end = ReasoningFilter.PLAMO_END_THINK;
        assertParse("plamo3", NO_VARS, "Reasoning.<|plamo:end_", "Reasoning.", "");
        assertParse("plamo3", NO_VARS, "Reasoning.<|plamo:end_think", "Reasoning.", "");
        assertParse("plamo3", NO_VARS, "Reasoning.<|pla", "Reasoning.<|pla", "");
        assertParse("plamo3", NO_VARS, "R" + end + "Answer.<|plamo:ta", "R", "Answer.");
        assertParse("plamo3", NO_VARS, "Reasoning.<|plamo:end_think:plamo|>", "Reasoning.", "");
    }

    @Test
    public void plamo3WithoutThinkingIsContent() {
        String out = ReasoningFilter.PLAMO_BEGIN_THINK + "Reasoning." + ReasoningFilter.PLAMO_END_THINK + "Answer.";
        assertParse("plamo3", vars("enable_thinking", false), out, "", out);
        assertParse("plamo3", vars("enable_thinking", false), "Answer.", "", "Answer.");
    }

    // ------------------------------------------------------------------ inkling

    @Test
    public void inklingRoutesThinkingBlocksAndDropsTextMarkers() {
        String out = ReasoningFilter.INKLING_MESSAGE + ReasoningFilter.INKLING_THINKING + "think"
                + ReasoningFilter.INKLING_END + ReasoningFilter.INKLING_MESSAGE + ReasoningFilter.INKLING_TEXT
                + "hello" + ReasoningFilter.INKLING_END;
        assertParse("inkling", NO_VARS, out, "think", "hello");
        assertParse("inkling", NO_VARS, ReasoningFilter.INKLING_THINKING + "a" + ReasoningFilter.INKLING_END_SAMPLING
                + ReasoningFilter.INKLING_TOOL_ERROR + "oops" + ReasoningFilter.INKLING_END, "a", "oops");
        assertParse("inkling", NO_VARS, "plain answer", "", "plain answer");
        assertParse("inkling", NO_VARS, ReasoningFilter.INKLING_THINKING + "t" + ReasoningFilter.INKLING_THINKING + "u"
                + ReasoningFilter.INKLING_END + "x", "tu", "x");
    }

    @Test
    public void inklingPassesToolBlocksWholeToItsToolParser() {
        String body = "{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}";
        String block = ReasoningFilter.INKLING_TOOL_JSON + body + ReasoningFilter.INKLING_END;
        String out = ReasoningFilter.INKLING_THINKING + "check" + ReasoningFilter.INKLING_END
                + ReasoningFilter.INKLING_MESSAGE + "get_weather" + block;
        List<String> parsed = run(filter("inkling"), NO_VARS, null, out);
        assertEquals("check", parsed.get(0));
        assertEquals("the header and its function name are dropped, the block passes whole", block, parsed.get(1));
        ToolParse calls = ToolParserRegistry.forName("inkling").create(null).parse(parsed.get(1), false);
        assertEquals(1, calls.calls().size());
        assertEquals("get_weather", calls.calls().get(0).name());
        assertEquals("", calls.content());
        assertParse("inkling", NO_VARS, ReasoningFilter.INKLING_THINKING + "unclosed" + block, "unclosed", block);
    }

    @Test
    public void inklingSeedsItsStateFromThePromptsLastMarker() {
        ReasoningFilter f = filter("inkling");
        String thinking = "<|message_user|>q" + ReasoningFilter.INKLING_MESSAGE + ReasoningFilter.INKLING_THINKING;
        assertEquals(Arrays.asList("r", "c"), run(f, NO_VARS, thinking, "r" + ReasoningFilter.INKLING_END + "c"));
        String text = ReasoningFilter.INKLING_MESSAGE + ReasoningFilter.INKLING_TEXT;
        assertEquals(Arrays.asList("", "c"), run(f, NO_VARS, text, "c" + ReasoningFilter.INKLING_END));
        String header = "q" + ReasoningFilter.INKLING_MESSAGE;
        assertEquals("text before a kind marker is the header", Arrays.asList("", "c"),
                run(f, NO_VARS, header, "fn" + ReasoningFilter.INKLING_TEXT + "c"));
        assertTrue(f.newState((channel, tokenId, t) -> { }, thinking).inReasoning());
        assertFalse(f.newState((channel, tokenId, t) -> { }, text).inReasoning());
    }

    @Test
    public void inklingTemplatesSelectTheBlockLayoutUnderAuto() {
        ReasoningFilter auto = ReasoningFilter.forChatTemplate("{{ '<|message_model|><|content_thinking|>' }}");
        assertEquals(Collections.singletonList(ReasoningFilter.INKLING_THINKING), auto.startTags());
        assertEquals(Arrays.asList("t", "x"), run(auto, NO_VARS, null,
                ReasoningFilter.INKLING_THINKING + "t" + ReasoningFilter.INKLING_END + ReasoningFilter.INKLING_TEXT + "x"));
    }

    // ------------------------------------------------------------------ minimax_m2_append_think and the rest

    @Test
    public void minimaxAppendThinkKeepsEverythingAsContentAfterTheOpenedTag() {
        assertParse("minimax_m2_append_think", NO_VARS, "abc</think>answer", "", "<think>abc</think>answer");
        assertEquals(Arrays.asList("", "<think>"), run(filter("minimax_m2_append_think"), NO_VARS, null, ""));
        ReasoningFilter f = ReasoningParsers.forModel("minimax_m2_append_think",
                tokenizer(Arrays.asList("<think>", "</think>"), "x"));
        assertTrue("the tags render as text even when hidden",
                f.tagTokens().values().containsAll(Arrays.asList("<think>", "</think>")));
        assertNotSame(ReasoningParsers.forName("minimax_m2"), ReasoningParsers.forName("minimax_m2_append_think"));
    }

    @Test
    public void newNamesAreRegisteredWithoutAliasesOnHunyuanA13b() {
        for (String name : Arrays.asList("granite_thinking_parser", "minicpmv", "plamo3", "inkling", "hy_v3", "hy_v4",
                "minimax_m2_append_think")) {
            assertTrue(name, ReasoningParsers.names().contains(name));
        }
        assertTrue(ReasoningParsers.forName("hunyuan_a13b").aliases().isEmpty());
    }

    @Test
    public void fixedFiltersIgnoreRequestVariables() {
        ReasoningFilter qwen = filter("qwen3");
        assertSame(qwen, qwen.forRequest(vars("enable_thinking", false)));
        assertFalse(qwen.selectsPerRequest());
        assertTrue(filter("hy_v3").selectsPerRequest());
        assertFalse(filter("hy_v3").forRequest(NO_VARS).selectsPerRequest());
    }

    @Test
    public void rawVariantsOfPerRequestFiltersStayPerRequest() {
        ReasoningFilter raw = filter("hy_v4").asRaw();
        assertTrue(raw.selectsPerRequest());
        assertEquals(ReasoningFilter.Mode.NONE, raw.forRequest(NO_VARS).mode());
        assertEquals(ReasoningFilter.Mode.NONE, raw.forRequest(vars("reasoning_effort", "no_think")).mode());
    }

    @Test
    public void chatRequestTemplateVariablesLayerKwargsOverTopLevelSwitches() {
        ChatRequest chat = ChatRequest.builder()
                .addMessage(message())
                .enableThinking(true)
                .reasoningEffort("low")
                .templateKwarg("enable_thinking", false)
                .templateKwarg("force_nonempty_content", true)
                .build();
        Map<String, Object> v = chat.templateVariables();
        assertEquals(false, v.get("enable_thinking"));
        assertEquals("low", v.get("reasoning_effort"));
        assertEquals(true, v.get("force_nonempty_content"));
        assertTrue(ChatRequest.builder().addMessage(message()).build().templateVariables().isEmpty());
        assertParse("minicpmv", v, "<think>hidden</think>shown", "", "shown");
    }

    // ------------------------------------------------------------------ helpers

    /** Parses {@code output} with no prompt, char by char and whole, and checks both splits. */
    private static void assertParse(String parser, Map<String, Object> vars, String output, String reasoning,
                                    String content) {
        ReasoningFilter f = filter(parser);
        assertEquals(parser + " " + vars + " streamed", Arrays.asList(reasoning, content), run(f, vars, null, output));
        assertEquals(parser + " " + vars + " whole", Arrays.asList(reasoning, content), runWhole(f, vars, output));
    }

    private static ReasoningFilter filter(String name) {
        return ReasoningParsers.forName(name).create();
    }

    /** [reasoning, content] of {@code output} fed one character per token. */
    private static List<String> run(ReasoningFilter f, Map<String, Object> vars, String prompt, String output) {
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State st = f.forRequest(vars).newState((channel, tokenId, text) ->
                (channel == ReasoningFilter.Channel.REASONING ? reasoning : content).append(text), prompt);
        for (int i = 0; i < output.length(); i++) {
            st.onToken(i, output.substring(i, i + 1));
        }
        st.finish();
        return Arrays.asList(reasoning.toString(), content.toString());
    }

    private static List<String> runWhole(ReasoningFilter f, Map<String, Object> vars, String output) {
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State st = f.forRequest(vars).newState((channel, tokenId, text) ->
                (channel == ReasoningFilter.Channel.REASONING ? reasoning : content).append(text), null);
        st.onToken(0, output);
        st.finish();
        return Arrays.asList(reasoning.toString(), content.toString());
    }

    private static Map<String, Object> vars(String name, Object value) {
        Map<String, Object> m = new HashMap<>();
        m.put(name, value);
        return m;
    }

    private static Map<String, Object> message() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", "user");
        m.put("content", "hi");
        return m;
    }

    /** A SentencePiece vocabulary with {@code extra} as user-defined tokens after the byte tokens. */
    private static GGUFTokenizer tokenizer(List<String> extra, String template) {
        List<Object> tokens = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, scores, types, "<unk>", 2);
        add(tokens, scores, types, "<s>", 3);
        add(tokens, scores, types, "</s>", 3);
        for (int b = 0; b < 256; b++) {
            add(tokens, scores, types, String.format("<0x%02X>", b), 6);
        }
        for (String t : extra) {
            add(tokens, scores, types, t, 4);
        }
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "llama");
        md.put("tokenizer.ggml.tokens", tokens);
        md.put("tokenizer.ggml.scores", scores);
        md.put("tokenizer.ggml.token_type", types);
        md.put("tokenizer.ggml.bos_token_id", 1);
        md.put("tokenizer.ggml.eos_token_id", 2);
        md.put("tokenizer.ggml.unknown_token_id", 0);
        md.put("tokenizer.chat_template", template);
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static void add(List<Object> tokens, List<Object> scores, List<Object> types, String t, int type) {
        tokens.add(t);
        scores.add(-1.0f);
        types.add(type);
    }
}
