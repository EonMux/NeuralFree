package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.chat.tools.ToolTestSupport.accepts;
import static com.eonmux.neuralfree.gguf.chat.tools.ToolTestSupport.chat;
import static com.eonmux.neuralfree.gguf.chat.tools.ToolTestSupport.tools;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The native-format tool grammars: a forcing choice constrains the output to the format's own call syntax and
 * reads back through the format's parser, {@code tool_choice: auto} applies the same syntax as a lazy grammar
 * behind the format's call openers, and the strict level and flag decide whether the argument schemas hold.
 */
public class NativeToolGrammarTest {

    private static final ReasoningFilter NO_REASONING = ReasoningFilter.none();

    // ------------------------------------------------------------------ hermes: forcing, read-back, strictness

    @Test
    public void hermesRequiredForcesNativeCallsAndReadsThemBack() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "hermes", null, NO_REASONING,
                null, ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertFalse(c.gated());
        assertTrue("the format's own call", accepts(c,
                "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>"));
        assertTrue("the template's newline variants", accepts(c,
                "<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>"));
        assertTrue("parallel calls", accepts(c,
                "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>"
                        + "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}"
                        + "</tool_call>"));
        assertFalse("an undeclared function is refused", accepts(c,
                "<tool_call>{\"name\": \"other\", \"arguments\": {}}</tool_call>"));
        assertFalse("free text is refused", accepts(c, "I will not call a tool."));
        assertFalse("a broken envelope is refused", accepts(c,
                "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call> junk"));
        ToolParse p = run(r, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"
                + "</tool_call>");
        assertEquals("", p.content());
        assertEquals(1, p.calls().size());
        assertEquals("get_weather", p.calls().get(0).name());
        assertEquals("{\"city\": \"Paris\"}", p.calls().get(0).arguments());

        ToolRouting single = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "hermes", null,
                NO_REASONING, null, ParallelToolCalls.SINGLE);
        assertFalse("parallel_tool_calls false allows one call", accepts(
                GbnfBuilder.compile(single.constraint()),
                "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>"
                        + "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>"));
    }

    @Test
    public void aNamedFunctionForcesThatFunctionsNativeCall() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.function("get_weather")), "hermes", null,
                NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertTrue(accepts(c, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}"
                + "</tool_call>"));
        assertFalse("only the named function", accepts(c,
                "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>"));
        ToolParse p = run(r, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}"
                + "</tool_call>");
        assertEquals("get_weather", p.calls().get(0).name());
        assertEquals("{\"city\": \"Oslo\"}", p.calls().get(0).arguments());
    }

    @Test
    public void aStrictFlagHoldsTheArgumentsAndFalseLeavesThemFree() {
        CompiledConstraint strict = GbnfBuilder.compile(ToolRouting.plan(chat(ChatRequest.ToolChoice.required()),
                "hermes", null, NO_REASONING, null, ParallelToolCalls.PARALLEL).constraint());
        assertTrue("the schema holds when the tool does not opt out", accepts(strict,
                "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>"));
        assertFalse("the declared property is pinned", accepts(strict,
                "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"town\": \"Paris\"}}</tool_call>"));

        List<ChatTool> flagged = Arrays.asList(
                ChatTool.builder("get_weather").strict(Boolean.FALSE).build(),
                ChatTool.builder("noop").strict(Boolean.TRUE).build());
        ToolRouting routing = ToolRouting.plan(request(flagged, ChatRequest.ToolChoice.required()), "hermes",
                null, NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(routing.constraint());
        assertTrue("a tool that declines strict keeps free arguments", accepts(c,
                "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"any\": \"thing\"}}</tool_call>"));
        assertTrue("a strict tool keeps its schema", accepts(c,
                "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>"));
        assertFalse(accepts(c, "<tool_call>{\"name\": \"noop\", \"arguments\": \"no object\"}</tool_call>"));
    }

    @Test
    public void autoAppliesTheLazyGrammarBehindTheOpeners() {
        Map<String, Object> strictParams = strictTool();
        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictParams).strict(Boolean.TRUE).build());
        ChatRequest request = request(flagged, ChatRequest.ToolChoice.auto());
        ToolRouting r = ToolRouting.plan(request, "auto", HERMES_TEMPLATE, NO_REASONING, null,
                ParallelToolCalls.PARALLEL);
        assertNotNull("a strict tool asks for the lazy grammar", r.constraint());
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertTrue(c.gated());
        assertTrue("free text, then the call from the opener on", accepts(c,
                "Let me check. <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"
                        + "</tool_call>"));
        assertTrue(accepts(c, "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"
                + "</tool_call>"));
        assertFalse("a malformed call is refused once the opener fired", accepts(c,
                "<tool_call>{\"name\": \"other\"}"));
        ToolParse p = run(r, "Sure.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"
                + "</tool_call>");
        assertEquals("Sure.", p.content());
        assertEquals(1, p.calls().size());

        ToolRouting plain = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "auto", HERMES_TEMPLATE,
                NO_REASONING, null, ParallelToolCalls.PARALLEL);
        assertNull("no strict tool, level auto: no grammar", plain.constraint());
        assertTrue(plain.active());
        ToolRouting always = ToolRouting.plan(chat(ChatRequest.ToolChoice.auto()), "auto", HERMES_TEMPLATE,
                NO_REASONING, null, ParallelToolCalls.PARALLEL, ToolStrict.FUNCTION);
        assertNotNull("level function constrains every request with tools", always.constraint());
        assertTrue(accepts(GbnfBuilder.compile(always.constraint()),
                "Checking. <tool_call>{\"name\": \"noop\", \"arguments\": {\"any\": \"text\"}}</tool_call>"));
    }

    // ------------------------------------------------------------------ the other format families

    @Test
    public void qwen3CoderForcesItsXmlParametersAndReadsThemBack() {
        ToolRouting r = routing("qwen3_coder", ChatRequest.ToolChoice.required(), true);
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertTrue(accepts(c, "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                + "</function>\n</tool_call>"));
        assertTrue("an optional parameter may follow", accepts(c, "<tool_call>\n<function=noop>\n</function>\n"
                + "</tool_call>"));
        assertFalse(accepts(c, "<tool_call>\n<function=other>\n</function>\n</tool_call>"));
        ToolParse p = run(r, "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                + "</function>\n</tool_call>");
        assertEquals("get_weather", p.calls().get(0).name());
        assertEquals("{\"city\":\"Paris\"}", p.calls().get(0).arguments());
    }

    @Test
    public void seedOssAndLongcatForceTheirOwnTags() {
        CompiledConstraint seed = GbnfBuilder.compile(routing("seed_oss", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(seed, "<seed:tool_call><function=get_weather><parameter=city>Paris</parameter>"
                + "</function></seed:tool_call>"));
        assertFalse(accepts(seed, "<seed:tool_call><function=get_weather><parameter=town>Paris</parameter>"
                + "</function></seed:tool_call>"));
        CompiledConstraint longcat = GbnfBuilder.compile(routing("longcat", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(longcat, "<longcat_tool_call>{\"name\": \"get_weather\", "
                + "\"arguments\": {\"city\": \"Paris\"}}</longcat_tool_call>"));
        assertFalse(accepts(longcat, "<tool_call>{\"name\": \"get_weather\", "
                + "\"arguments\": {\"city\": \"Paris\"}}</tool_call>"));
    }

    @Test
    public void glmAndMinimaxForceTheirTaggedArguments() {
        CompiledConstraint glm = GbnfBuilder.compile(routing("glm45", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(glm, "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
                + "</tool_call>"));
        assertTrue(accepts(glm, "<tool_call>noop</tool_call>"));
        assertFalse(accepts(glm, "<tool_call>other\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
                + "</tool_call>"));
        ToolParse glmRead = run(routing("glm45", ChatRequest.ToolChoice.required(), true),
                "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>");
        assertEquals("{\"city\":\"Paris\"}", glmRead.calls().get(0).arguments());

        CompiledConstraint minimax = GbnfBuilder.compile(routing("minimax_m2", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(minimax, "<minimax:tool_call>\n<invoke name=\"get_weather\">\n"
                + "<parameter name=\"city\">Paris</parameter>\n</invoke>\n</minimax:tool_call>"));
        assertFalse(accepts(minimax, "<minimax:tool_call>\n<invoke name=\"get_weather\">\n"
                + "<parameter name=\"city\">Paris</parameter>\n</invoke>\n</minimax:tool_call>"
                + "<minimax:tool_call>\n<invoke name=\"noop\">\n</invoke>\n</minimax:tool_call>"));
    }

    @Test
    public void delimitedFormatsForceTheirSections() {
        CompiledConstraint ds3 = GbnfBuilder.compile(routing("deepseek_v3", ChatRequest.ToolChoice.required(),
                false).constraint());
        String ds3Call = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
                + "```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        assertTrue(acceptsText(ds3, ds3Call));
        assertFalse(acceptsText(ds3, "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>other\n"
                + "```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"));
        ToolParse ds3Read = run(routing("deepseek_v3", ChatRequest.ToolChoice.required(), false), ds3Call);
        assertEquals("get_weather", ds3Read.calls().get(0).name());
        assertEquals("{\"city\": \"Paris\"}", ds3Read.calls().get(0).arguments());

        CompiledConstraint ds31 = GbnfBuilder.compile(routing("deepseek_v31", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(acceptsText(ds31, "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
                + "{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"));

        CompiledConstraint kimi = GbnfBuilder.compile(routing("kimi_k2", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(kimi, "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
                + "<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>"
                + "<|tool_calls_section_end|>"));
        ToolParse kimiRead = run(routing("kimi_k2", ChatRequest.ToolChoice.required(), true),
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
                        + "<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>"
                        + "<|tool_calls_section_end|>");
        assertEquals("get_weather", kimiRead.calls().get(0).name());
        assertEquals("{\"city\": \"Paris\"}", kimiRead.calls().get(0).arguments());
    }

    /** DSML calls as xgrammar's builtin {@code deepseek_v3_2}, {@code deepseek_v4} and {@code deepseek_v4_1} tags write them. */
    @Test
    public void dsmlFormatsForceXgrammarsDeepseekTags() {
        String[][] families = {{"deepseek_v32", "function_calls", "invoke", ""}, {"deepseek_v4", "tool_calls",
                "invoke", ""}, {"deepseek_v41", " calls", " invoke", " "}};
        for (String[] f : families) {
            String head = "\n\n<｜DSML｜" + f[1] + ">\n";
            String tail = "</｜DSML｜" + f[1] + ">";
            String weather = "<｜DSML｜" + f[2] + " name=\"get_weather\">\n<｜DSML｜" + f[3] + "parameter name=\"city\""
                    + " string=\"true\">Paris</｜DSML｜" + f[3] + "parameter>\n</｜DSML｜" + f[2] + ">\n";
            String noop = "<｜DSML｜" + f[2] + " name=\"noop\">\n</｜DSML｜" + f[2] + ">\n";
            CompiledConstraint one = GbnfBuilder.compile(routing(f[0], ChatRequest.ToolChoice.required(), false)
                    .constraint());
            assertTrue(f[0], acceptsText(one, head + weather + tail));
            assertTrue(f[0], acceptsText(one, head + noop + tail));
            assertFalse(f[0] + " opens with the blank line", acceptsText(one, (head + weather + tail).substring(2)));
            assertFalse(f[0] + " writes one call", acceptsText(one, head + weather + noop + tail));
            assertFalse(f[0] + " types the string", acceptsText(one,
                    head + weather.replace("string=\"true\"", "string=\"false\"") + tail));

            CompiledConstraint many = GbnfBuilder.compile(routing(f[0], ChatRequest.ToolChoice.required(), true)
                    .constraint());
            assertTrue(f[0], acceptsText(many, head + weather + noop + tail));
            assertFalse(f[0], acceptsText(many, head + tail));
            ToolParse read = run(routing(f[0], ChatRequest.ToolChoice.required(), true), head + weather + noop + tail);
            assertEquals(2, read.calls().size());
            assertEquals("get_weather", read.calls().get(0).name());
            assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

            List<ChatTool> loose = Collections.singletonList(ChatTool.builder("get_weather")
                    .parameters(strictTool()).strict(Boolean.FALSE).build());
            CompiledConstraint free = GbnfBuilder.compile(ToolRouting.plan(request(loose,
                    ChatRequest.ToolChoice.required()), f[0], null, NO_REASONING, null, ParallelToolCalls.PARALLEL)
                    .constraint());
            assertTrue(f[0] + " leaves a loose tool's names free", acceptsText(free,
                    head + weather.replace("city", "town") + tail));
            assertTrue(f[0] + " leaves a loose tool's values free", acceptsText(free,
                    head + weather.replace("string=\"true\">Paris", "string=\"false\">3") + tail));
        }
    }

    /** MiMo calls as xgrammar's builtin {@code mimo} tag writes them: {@code qwen_xml} parameters, no newlines. */
    @Test
    public void mimoForcesXgrammarsQwenXmlCalls() {
        String weather = "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>";
        String noop = "<tool_call><function=noop></function></tool_call>";
        CompiledConstraint one = GbnfBuilder.compile(routing("mimo", ChatRequest.ToolChoice.required(), false)
                .constraint());
        assertTrue(accepts(one, weather));
        assertTrue(accepts(one, noop));
        assertTrue(accepts(one, "<tool_call><function=get_weather>\n<parameter=city>Paris</parameter>\n</function>"
                + "</tool_call>"));
        assertTrue(accepts(one, "<tool_call><function=get_weather><parameter=city>\nParis\n</parameter></function>"
                + "</tool_call>"));
        assertFalse(accepts(one, weather + noop));
        assertFalse(accepts(one, weather.replace("city", "town")));
        assertTrue(accepts(GbnfBuilder.compile(routing("mimo", ChatRequest.ToolChoice.required(), true)
                .constraint()), weather + noop));
        ToolParse read = run(routing("mimo", ChatRequest.ToolChoice.required(), true), weather + noop);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.TRUE).build());
        CompiledConstraint lazy = GbnfBuilder.compile(ToolRouting.plan(request(flagged,
                ChatRequest.ToolChoice.auto()), "mimo", null, NO_REASONING, null, ParallelToolCalls.PARALLEL)
                .constraint());
        assertTrue(lazy.gated());
        assertTrue(accepts(lazy, "Sure. " + weather));
        assertTrue("a bare function opens a call too", accepts(lazy,
                "Sure. <function=get_weather><parameter=city>Oslo</parameter></function>"));
        assertFalse(accepts(lazy, "Sure. <tool_call><function=other></function></tool_call>"));
    }

    @Test
    public void dsmlAutoRunsTheSectionGrammarFromAnyOpener() {
        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.TRUE).build());
        ToolRouting r = ToolRouting.plan(request(flagged, ChatRequest.ToolChoice.auto()), "deepseek_v4", null,
                NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertTrue(c.gated());
        String call = "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Oslo"
                + "</｜DSML｜parameter>\n</｜DSML｜invoke>\n";
        assertTrue(acceptsText(c, "Checking.\n\n<｜DSML｜tool_calls>\n" + call + "</｜DSML｜tool_calls>"));
        assertTrue(acceptsText(c, "Checking.<｜DSML｜toolcalls>\n" + call + "</｜DSML｜tool_calls>"));
        assertTrue("a bare invoke is the parser's other trigger", acceptsText(c, "Checking." + call));
        assertFalse(acceptsText(c, "Checking.<｜DSML｜tool_calls>\n" + call.replace("Oslo</", "Oslo")));
    }

    @Test
    public void arrayAndPythonicFormatsForceTheirOwnShapes() {
        CompiledConstraint mistral = GbnfBuilder.compile(routing("mistral", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(mistral, "[TOOL_CALLS][{\"name\": \"get_weather\", "
                + "\"arguments\": {\"city\": \"Paris\"}}]"));

        CompiledConstraint jamba = GbnfBuilder.compile(routing("jamba", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(jamba, "<tool_calls>[{\"name\": \"get_weather\", "
                + "\"arguments\": {\"city\": \"Paris\"}}, {\"name\": \"noop\", \"arguments\": {}}]</tool_calls>"));

        CompiledConstraint granite = GbnfBuilder.compile(routing("granite", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(granite, "<|tool_call|>[{\"name\": \"noop\", \"arguments\": {}}]"));

        CompiledConstraint llama3 = GbnfBuilder.compile(routing("llama3_json", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(llama3, "<|python_tag|>{\"name\": \"get_weather\", "
                + "\"parameters\": {\"city\": \"Paris\"}}"));
        ToolParse llama3Read = run(routing("llama3_json", ChatRequest.ToolChoice.required(), true),
                "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}");
        assertEquals("{\"city\": \"Paris\"}", llama3Read.calls().get(0).arguments());

        CompiledConstraint pythonic = GbnfBuilder.compile(routing("pythonic", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(pythonic, "[get_weather(city=\"Paris\")]"));
        assertTrue(accepts(pythonic, "[noop()]"));
        assertFalse(accepts(pythonic, "[other(city=\"Paris\")]"));
        ToolParse pythonicRead = run(routing("pythonic", ChatRequest.ToolChoice.required(), true),
                "[get_weather(city=\"Paris\")]");
        assertEquals("get_weather", pythonicRead.calls().get(0).name());
        assertEquals("{\"city\":\"Paris\"}", pythonicRead.calls().get(0).arguments());

        CompiledConstraint llama4 = GbnfBuilder.compile(routing("llama4_pythonic",
                ChatRequest.ToolChoice.required(), true).constraint());
        assertTrue(accepts(llama4, "<|python_start|>[get_weather(city=\"Paris\")]<|python_end|>"));

        CompiledConstraint olmo = GbnfBuilder.compile(routing("olmo3", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(olmo, "<function_calls>get_weather(city=\"Paris\")\n</function_calls>"));

        CompiledConstraint internlm = GbnfBuilder.compile(routing("internlm", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(internlm, "<|action_start|><|plugin|>{\"name\": \"get_weather\", "
                + "\"arguments\": {\"city\": \"Paris\"}}<|action_end|>"));

        CompiledConstraint phi4 = GbnfBuilder.compile(routing("phi4_mini_json",
                ChatRequest.ToolChoice.required(), true).constraint());
        assertTrue(accepts(phi4, "functools[{\"name\": \"noop\", \"arguments\": {}}]"));

        CompiledConstraint ernie = GbnfBuilder.compile(routing("ernie45", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(ernie, "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>"));

        CompiledConstraint xlam = GbnfBuilder.compile(routing("xlam", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(xlam, "[{\"name\": \"noop\", \"arguments\": {}}]"));
    }

    @Test
    public void aFormatWithoutANativeGrammarKeepsTheGenericBehavior() {
        ToolRouting r = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "auto", "{{ messages }}",
                NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(r.constraint());
        assertTrue(accepts(c, "[{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}]"));
        assertFalse(accepts(c, "<tool_call>{\"name\": \"noop\", \"arguments\": {}}</tool_call>"));

        ToolRouting named = ToolRouting.plan(chat(ChatRequest.ToolChoice.function("get_weather")), "auto",
                "{{ messages }}", NO_REASONING, null, ParallelToolCalls.PARALLEL);
        assertTrue(accepts(GbnfBuilder.compile(named.constraint()), "{\"city\": \"Oslo\"}"));
    }

    @Test
    public void anOpenReasoningPromptGatesTheNativeGrammarOnTheEndTag() {
        ReasoningFilter think = ReasoningFilter.thinkTags();
        ToolRouting open = ToolRouting.plan(chat(ChatRequest.ToolChoice.required()), "hermes", null, think,
                "<|im_start|>assistant\n<think>\n", ParallelToolCalls.PARALLEL);
        CompiledConstraint c = GbnfBuilder.compile(open.constraint());
        assertTrue(c.gated());
        assertTrue(accepts(c, "Let me think.</think>\n\n<tool_call>{\"name\": \"noop\", \"arguments\": {}}"
                + "</tool_call>"));
        assertFalse(accepts(c, "Let me think.</think>\n\nNo tools today."));
    }

    // ------------------------------------------------------------------ vLLM's builtin structural-tag builders

    /** PLaMo 3, vLLM's builtin {@code plamo3} tag: a requests block, its closer alone or with the end-of-turn tag. */
    @Test
    public void plamo3ForcesItsRequestBlocksAndReadsThemBack() {
        String open = "<|plamo:begin_tool_request:plamo|><|plamo:begin_tool_name:plamo|>";
        String nameEnd = "<|plamo:end_tool_name:plamo|>";
        String argsOpen = "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>";
        String close = "<|plamo:end_tool_arguments:plamo|><|plamo:end_tool_request:plamo|>";
        String weather = open + "get_weather" + nameEnd + argsOpen + "{\"city\": \"Paris\"}" + close;
        String noop = open + "noop" + nameEnd + argsOpen + "{}" + close;
        String sectionOpen = "<|plamo:begin_tool_requests:plamo|>";
        String sectionClose = "<|plamo:end_tool_requests:plamo|>";

        CompiledConstraint one = GbnfBuilder.compile(routing("plamo3", ChatRequest.ToolChoice.required(), false)
                .constraint());
        assertTrue(accepts(one, sectionOpen + weather + sectionClose));
        assertTrue("the closer may carry the end-of-turn tag", accepts(one,
                sectionOpen + weather + sectionClose + "<|plamo:tag|>"));
        assertFalse(accepts(one, weather));
        assertFalse("one call under parallel_tool_calls false", accepts(one,
                sectionOpen + weather + noop + sectionClose));

        CompiledConstraint many = GbnfBuilder.compile(routing("plamo3", ChatRequest.ToolChoice.required(), true)
                .constraint());
        assertTrue(accepts(many, sectionOpen + weather + noop + sectionClose));
        assertFalse(accepts(many, sectionOpen + sectionClose));
        ToolParse read = run(routing("plamo3", ChatRequest.ToolChoice.required(), true),
                sectionOpen + weather + noop + sectionClose);
        assertEquals(2, read.calls().size());
        assertEquals("get_weather", read.calls().get(0).name());
        assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());

        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.TRUE).build());
        CompiledConstraint lazy = GbnfBuilder.compile(ToolRouting.plan(request(flagged,
                ChatRequest.ToolChoice.auto()), "plamo3", null, NO_REASONING, null, ParallelToolCalls.PARALLEL)
                .constraint());
        assertTrue(lazy.gated());
        assertTrue(accepts(lazy, "Sure. " + sectionOpen + weather + sectionClose));
        assertTrue("a bare request opens a call too", accepts(lazy, "Sure. " + weather));
        assertTrue(accepts(lazy, "Sure. " + weather + "<|plamo:tag|>"));
        assertFalse(accepts(lazy, "Sure. " + open + "other" + nameEnd + argsOpen + "{}" + close));
    }

    /** Kimi K3, vLLM's builtin {@code kimi_k3} tag: XTML response and tools channels. */
    @Test
    public void kimiK3ForcesItsXtmlChannelsAndReadsThemBack() {
        String call = "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
                + "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|>"
                + "<|close|>call<|sep|>";
        String channel = "<|open|>tools<|sep|>" + call + "<|close|>tools<|sep|>";
        String prefix = "<|open|>response<|sep|>Rain?" + "<|close|>response<|sep|>";

        CompiledConstraint one = GbnfBuilder.compile(routing("kimi_k3", ChatRequest.ToolChoice.required(), false)
                .constraint());
        assertTrue(accepts(one, prefix + channel));
        assertTrue("the response opener is optional", accepts(one, "Rain?" + "<|close|>response<|sep|>" + channel));
        assertTrue("the message closer is optional", accepts(one, prefix + channel + "<|close|>message<|sep|>"));
        assertFalse("the tools channel needs the response close", accepts(one, channel));
        assertFalse("the required argument is missing", accepts(one, prefix
                + "<|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|><|close|>call<|sep|>"
                + "<|close|>tools<|sep|>"));
        assertFalse("the key is pinned", accepts(one, prefix + channel.replace("key=\"city\"", "key=\"town\"")));
        assertFalse("the type is pinned", accepts(one, prefix + channel.replace("type=\"string\"", "type=\"number\"")));
        assertFalse("a reserved marker cannot start inside the response text", accepts(one,
                prefix.replace("Rain?", "Ra<|open|>in") + channel));

        CompiledConstraint many = GbnfBuilder.compile(routing("kimi_k3", ChatRequest.ToolChoice.required(), true)
                .constraint());
        assertTrue("a parameterless strict call stays permissive", accepts(many, prefix
                + "<|open|>tools<|sep|><|open|>call tool=\"noop\" index=\"2\"<|sep|><|close|>call<|sep|>"
                + "<|close|>tools<|sep|>"));
        ToolParse read = run(routing("kimi_k3", ChatRequest.ToolChoice.required(), true),
                prefix + channel + "<|close|>message<|sep|>");
        assertEquals("Rain?", read.content());
        assertEquals(1, read.calls().size());
        assertEquals("get_weather", read.calls().get(0).name());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.TRUE).build());
        CompiledConstraint lazy = GbnfBuilder.compile(ToolRouting.plan(request(flagged,
                ChatRequest.ToolChoice.auto()), "kimi_k3", null, NO_REASONING, null, ParallelToolCalls.PARALLEL)
                .constraint());
        assertTrue(lazy.gated());
        assertTrue(accepts(lazy, "Sure. " + channel + "<|close|>message<|sep|>"));
        assertFalse(accepts(lazy, "Sure. " + channel.replace("Paris<|close|>", "Paris")));
    }

    /** HY v4, vLLM's builtin {@code hy_v4} tag: {@code arg_key}/{@code arg_value} pairs, values free. */
    @Test
    public void hyV4ForcesItsArgPairsAndReadsThemBack() {
        String weather = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        String noop = "<tool_call>noop</tool_call>";
        String open = "<tool_calls>";
        String close = "</tool_calls>";

        CompiledConstraint one = GbnfBuilder.compile(routing("hy_v4", ChatRequest.ToolChoice.required(), false)
                .constraint());
        assertTrue(accepts(one, open + weather + close));
        assertTrue("a call with no arguments keeps free content", accepts(one, open + noop + close));
        assertFalse("a call opens inside the section", accepts(one, weather));
        assertFalse("the key is pinned", accepts(one, open
                + weather.replace("city", "town") + close));
        assertFalse("the value may not hold the closer", accepts(one, open
                + weather.replace("Paris</arg_value>", "Par</arg_value>is</arg_value>") + close));

        CompiledConstraint many = GbnfBuilder.compile(routing("hy_v4", ChatRequest.ToolChoice.required(), true)
                .constraint());
        assertTrue("adjacent calls carry no separator", accepts(many, open + weather + noop + close));
        ToolParse read = run(routing("hy_v4", ChatRequest.ToolChoice.required(), true), open + weather + close);
        assertEquals(1, read.calls().size());
        assertEquals("get_weather", read.calls().get(0).name());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.TRUE).build());
        CompiledConstraint lazy = GbnfBuilder.compile(ToolRouting.plan(request(flagged,
                ChatRequest.ToolChoice.auto()), "hy_v4", null, NO_REASONING, null, ParallelToolCalls.PARALLEL)
                .constraint());
        assertTrue(lazy.gated());
        assertTrue(accepts(lazy, "Sure. " + open + weather + close));
        assertTrue("a bare call opens a call too", accepts(lazy, "Sure. " + weather));
        assertFalse(accepts(lazy, "Sure. " + open + weather));
    }

    /** GLM 4.7, vLLM's builtin {@code glm_4_7} tag: strict schemas in the {@code glm_xml} style, loose keys pinned. */
    @Test
    public void glm47PinsStrictSchemasAndLooseKeys() {
        String strict = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        CompiledConstraint one = GbnfBuilder.compile(routing("glm_4_7", ChatRequest.ToolChoice.required(), false)
                .constraint());
        assertTrue("the strict tool's schema holds in the glm_xml style", accepts(one, strict));
        assertTrue("a string value is raw text, quotes included", accepts(one,
                strict.replace("Paris", "\"Paris\"")));
        assertFalse("a required parameter cannot be dropped", accepts(one, "<tool_call>get_weather</tool_call>"));

        Map<String, Object> mode = new LinkedHashMap<>();
        mode.put("type", "string");
        mode.put("enum", Arrays.asList("celsius", "fahrenheit"));
        Map<String, Object> days = new LinkedHashMap<>();
        days.put("type", "integer");
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("mode", mode);
        props.put("days", days);
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("type", "object");
        params.put("properties", props);
        List<ChatTool> loose = Collections.singletonList(ChatTool.builder("forecast")
                .parameters(params).strict(Boolean.FALSE).build());
        CompiledConstraint shallow = GbnfBuilder.compile(ToolRouting.plan(request(loose,
                ChatRequest.ToolChoice.required()), "glm_4_7", null, NO_REASONING, null,
                ParallelToolCalls.PARALLEL).constraint());
        assertTrue("the enum pins the raw value", accepts(shallow,
                "<tool_call>forecast<arg_key>mode</arg_key><arg_value>celsius</arg_value>"
                        + "<arg_key>days</arg_key><arg_value>5</arg_value></tool_call>"));
        assertFalse(accepts(shallow, "<tool_call>forecast<arg_key>mode</arg_key><arg_value>kelvin</arg_value>"
                + "</tool_call>"));
        assertTrue("pairs repeat in any order", accepts(shallow,
                "<tool_call>forecast<arg_key>days</arg_key><arg_value>3</arg_value>"
                        + "<arg_key>mode</arg_key><arg_value>fahrenheit</arg_value>"
                        + "<arg_key>days</arg_key><arg_value>4</arg_value></tool_call>"));

        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.TRUE).build());
        CompiledConstraint lazy = GbnfBuilder.compile(ToolRouting.plan(request(flagged,
                ChatRequest.ToolChoice.auto()), "glm_4_7", null, NO_REASONING, null, ParallelToolCalls.PARALLEL)
                .constraint());
        assertTrue(lazy.gated());
        assertTrue(accepts(lazy, "Sure. " + strict));
        assertFalse(accepts(lazy, "Sure. " + strict.replace("</arg_value>", "")));
    }

    // ------------------------------------------------------------------ the remaining registered formats

    private static final String MM = "]<]minimax[>[";
    private static final String MINIMAX_WEATHER = MM + "<tool_call>" + MM + "<invoke name=\"get_weather\">"
            + MM + "<city>Paris" + MM + "</city>" + MM + "</invoke>" + MM + "</tool_call>";
    private static final String MINIMAX_BOTH = MM + "<tool_call>" + MM + "<invoke name=\"get_weather\">"
            + MM + "<city>Paris" + MM + "</city>" + MM + "</invoke>\n"
            + MM + "<invoke name=\"noop\">" + MM + "</invoke>" + MM + "</tool_call>";

    @Test
    public void minimaxM3ForcesItsNamespaceElementsAndReadsThemBack() {
        CompiledConstraint one = GbnfBuilder.compile(routing("minimax_m3", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, MINIMAX_WEATHER));
        assertTrue("a call with no arguments", accepts(one, MM + "<tool_call>"
                + MM + "<invoke name=\"noop\">" + MM + "</invoke>" + MM + "</tool_call>"));
        assertFalse("parallel_tool_calls false writes one invoke", accepts(one, MINIMAX_BOTH));
        assertFalse("the key is pinned", accepts(one, MINIMAX_WEATHER.replace("<city>", "<town>")
                .replace("</city>", "</town>")));
        assertFalse("the function is pinned", accepts(one, MINIMAX_WEATHER.replace("get_weather", "other")));

        CompiledConstraint many = GbnfBuilder.compile(routing("minimax_m3", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, MINIMAX_BOTH));
        ToolParse read = run(routing("minimax_m3", ChatRequest.ToolChoice.required(), true), MINIMAX_BOTH);
        assertEquals(2, read.calls().size());
        assertEquals("get_weather", read.calls().get(0).name());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("minimax_m3");
        assertTrue(lazy.gated());
        assertTrue(accepts(lazy, "Sure. " + MINIMAX_WEATHER));
        assertTrue("a bare invoke opens a call too", accepts(lazy, "Sure. " + MM
                + "<invoke name=\"get_weather\">" + MM + "<city>Oslo" + MM + "</city>" + MM + "</invoke>"));
        assertFalse(accepts(lazy, "Sure. " + MM + "<tool_call>"
                + MM + "<invoke name=\"get_weather\">" + MM + "</invoke>" + MM + "</tool_call>"));
    }

    @Test
    public void minimaxM3WritesNestedObjectsAndArraysAsRecursiveElements() {
        Map<String, Object> city = Collections.singletonMap("type", "string");
        Map<String, Object> tags = new LinkedHashMap<>();
        tags.put("type", "array");
        tags.put("items", Collections.singletonMap("type", "string"));
        Map<String, Object> locationProps = new LinkedHashMap<>();
        locationProps.put("lat", Collections.singletonMap("type", "integer"));
        locationProps.put("lon", Collections.singletonMap("type", "integer"));
        Map<String, Object> location = new LinkedHashMap<>();
        location.put("type", "object");
        location.put("properties", locationProps);
        location.put("required", Arrays.asList("lat", "lon"));
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("city", city);
        props.put("tags", tags);
        props.put("location", location);
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", props);
        parameters.put("required", Arrays.asList("city", "tags", "location"));
        ChatTool tool = ChatTool.builder("get_weather").parameters(parameters).strict(Boolean.TRUE).build();
        List<ChatTool> nested = Collections.singletonList(tool);

        String call = MM + "<tool_call>" + MM + "<invoke name=\"get_weather\">"
                + MM + "<city>Paris" + MM + "</city>"
                + MM + "<tags>" + MM + "<item>a" + MM + "</item>" + MM + "<item>b" + MM + "</item>"
                + MM + "</tags>"
                + MM + "<location>" + MM + "<lat>48" + MM + "</lat>" + MM + "<lon>2" + MM + "</lon>"
                + MM + "</location>"
                + MM + "</invoke>" + MM + "</tool_call>";
        CompiledConstraint c = GbnfBuilder.compile(
                NativeToolGrammar.forcing(ToolParserRegistry.forName("minimax_m3"), nested, false, true));
        assertTrue("the template expands objects and arrays into nested elements", accepts(c, call));
        assertFalse("a nested key is pinned", accepts(c, call.replace("<lat>", "<latitude>")
                .replace("</lat>", "</latitude>")));
        assertFalse("an array element is an <item> element", accepts(c, call.replace(
                MM + "<item>a" + MM + "</item>", MM + "<value>a" + MM + "</value>")));
        ToolParse read = ToolParserRegistry.forName("minimax_m3").create(nested).parse(call, false);
        assertEquals(1, read.calls().size());
        assertEquals("{\"city\":\"Paris\",\"tags\":[\"a\",\"b\"],\"location\":{\"lat\":48,\"lon\":2}}",
                read.calls().get(0).arguments());

        String jsonText = MM + "<tool_call>" + MM + "<invoke name=\"get_weather\">"
                + MM + "<city>Paris" + MM + "</city>"
                + MM + "<tags>[\"a\",\"b\"]" + MM + "</tags>"
                + MM + "<location>{\"lat\":48,\"lon\":2}" + MM + "</location>"
                + MM + "</invoke>" + MM + "</tool_call>";
        ToolParse legacy = ToolParserRegistry.forName("minimax_m3").create(nested).parse(jsonText, false);
        assertEquals("the parser still reads JSON text inside an element",
                read.calls().get(0).arguments(), legacy.calls().get(0).arguments());

        String empties = MM + "<tool_call>" + MM + "<invoke name=\"get_weather\">"
                + MM + "<city>Paris" + MM + "</city>" + MM + "<tags>" + MM + "</tags>"
                + MM + "<location>" + MM + "</location>"
                + MM + "</invoke>" + MM + "</tool_call>";
        ToolParse emptyRead = ToolParserRegistry.forName("minimax_m3").create(nested).parse(empties, false);
        assertEquals("{\"city\":\"Paris\",\"tags\":[],\"location\":{}}",
                emptyRead.calls().get(0).arguments());
    }

    @Test
    public void k2HorizonWritesTheJsonCallBodyTheTemplateSelects() {
        String jsonCall = "<ifm|tool_calls><ifm|tool_call>{\"name\": \"get_weather\", "
                + "\"arguments\": {\"city\": \"Paris\"}}</ifm|tool_call></ifm|tool_calls>";
        String xmlCall = "<ifm|tool_calls><ifm|tool_call>get_weather<ifm|arg_key>city</ifm|arg_key>"
                + "<ifm|arg_value>Paris</ifm|arg_value></ifm|tool_call></ifm|tool_calls>";
        ToolRouting json = ToolRouting.plan(templateKwarg("json"),
                "k2_horizon", null, NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint forced = GbnfBuilder.compile(json.constraint());
        assertTrue(accepts(forced, jsonCall));
        assertFalse("the json layout writes the call as an object", accepts(forced, xmlCall));
        ToolParse read = run(json, jsonCall);
        assertEquals(1, read.calls().size());
        assertEquals("get_weather", read.calls().get(0).name());
        assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());

        ToolRouting xml = ToolRouting.plan(templateKwarg("xml"),
                "k2_horizon", null, NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint xmlForced = GbnfBuilder.compile(xml.constraint());
        assertTrue(accepts(xmlForced, xmlCall));
        assertFalse(accepts(xmlForced, jsonCall));

        ToolRouting typed = ToolRouting.plan(templateKwarg("xml_typed"),
                "k2_horizon", null, NO_REASONING, null, ParallelToolCalls.PARALLEL);
        CompiledConstraint typedForced = GbnfBuilder.compile(typed.constraint());
        assertTrue("xml_typed adds the optional type tag", accepts(typedForced,
                "<ifm|tool_calls><ifm|tool_call>get_weather<ifm|arg_key>city</ifm|arg_key>"
                        + "<ifm|arg_type>string</ifm|arg_type><ifm|arg_value>Paris</ifm|arg_value>"
                        + "</ifm|tool_call></ifm|tool_calls>"));

        IllegalArgumentException refused = null;
        try {
            ToolRouting.plan(templateKwarg("yaml"), "k2_horizon", null,
                    NO_REASONING, null, ParallelToolCalls.PARALLEL);
        } catch (IllegalArgumentException e) {
            refused = e;
        }
        assertNotNull("an unknown call layout is refused by name", refused);
        assertTrue(refused.getMessage(), refused.getMessage().contains("tool_call_format"));
    }

    @Test
    public void k2HorizonForcesItsArgTagsAndReadsThemBack() {
        String weather = "<ifm|tool_call>get_weather<ifm|arg_key>city</ifm|arg_key>"
                + "<ifm|arg_value>Paris</ifm|arg_value></ifm|tool_call>";
        String noop = "<ifm|tool_call>noop</ifm|tool_call>";
        String section = "<ifm|tool_calls>" + weather + "</ifm|tool_calls>";
        CompiledConstraint one = GbnfBuilder.compile(routing("k2_horizon", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, section));
        assertTrue(accepts(one, "<ifm|tool_calls>" + noop + "</ifm|tool_calls>"));
        assertFalse(accepts(one, "<ifm|tool_calls>" + weather + noop + "</ifm|tool_calls>"));
        assertFalse("the key is pinned", accepts(one, "<ifm|tool_calls>"
                + weather.replace("arg_key>city<", "arg_key>town<") + "</ifm|tool_calls>"));

        CompiledConstraint many = GbnfBuilder.compile(routing("k2_horizon", ChatRequest.ToolChoice.required(),
                true).constraint());
        String both = "<ifm|tool_calls>" + weather + "\n" + noop + "</ifm|tool_calls>";
        assertTrue(accepts(many, both));
        ToolParse read = run(routing("k2_horizon", ChatRequest.ToolChoice.required(), true), both);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("k2_horizon");
        assertTrue(lazy.gated());
        assertTrue(accepts(lazy, "Sure. <ifm|tool_calls>" + weather + "</ifm|tool_calls>"));
        assertTrue("a bare call is the parser's other opener", accepts(lazy, "Sure. " + weather));
    }

    @Test
    public void granite20bFcForcesItsFunctionCallObjects() {
        String weather = "<function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}";
        String noop = "<function_call>{\"name\": \"noop\", \"arguments\": {}}";
        CompiledConstraint one = GbnfBuilder.compile(routing("granite-20b-fc", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, weather));
        assertTrue(accepts(one, noop));
        assertFalse(accepts(one, weather + noop));
        assertFalse("the argument schema holds", accepts(one,
                "<function_call>{\"name\": \"get_weather\", \"arguments\": {\"town\": \"Paris\"}}"));

        CompiledConstraint many = GbnfBuilder.compile(routing("granite-20b-fc", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, weather + "\n" + noop));
        ToolParse read = run(routing("granite-20b-fc", ChatRequest.ToolChoice.required(), true),
                weather + "\n" + noop);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("granite-20b-fc");
        assertTrue(accepts(lazy, "Sure. " + weather));
    }

    @Test
    public void gigachat3ForcesItsJsonHeadersAndReadsThemBack() {
        String weather = "function call<|role_sep|>\n{\"name\": \"get_weather\", \"arguments\": "
                + "{\"city\": \"Paris\"}}";
        String flat = "<|function_call|>{\"name\": \"noop\", \"arguments\": {}}";
        CompiledConstraint one = GbnfBuilder.compile(routing("gigachat3", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, weather));
        assertTrue("the template's message separator may precede the header", accepts(one,
                "<|message_sep|>\n\n" + weather));
        assertTrue(accepts(one, flat));
        assertFalse(accepts(one, weather + flat));

        CompiledConstraint many = GbnfBuilder.compile(routing("gigachat3", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, weather + "\n" + flat));
        ToolParse read = run(routing("gigachat3", ChatRequest.ToolChoice.required(), true), weather);
        assertEquals(1, read.calls().size());
        assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("gigachat3");
        assertTrue(accepts(lazy, "The weather:<|message_sep|>\n\n" + weather));
        assertTrue(accepts(lazy, "Sure. <|function_call|>{\"name\": \"get_weather\", "
                + "\"arguments\": {\"city\": \"Paris\"}}"));
    }

    @Test
    public void minicpm5ForcesItsParamTagsAndReadsThemBack() {
        String weather = "<function name=\"get_weather\"><param name=\"city\">Paris</param></function>";
        String noop = "<function name=\"noop\"></function>";
        CompiledConstraint one = GbnfBuilder.compile(routing("minicpm5", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, weather));
        assertTrue(accepts(one, noop));
        assertFalse(accepts(one, weather + noop));
        assertFalse("the key is pinned", accepts(one,
                "<function name=\"get_weather\"><param name=\"town\">Paris</param></function>"));

        CompiledConstraint many = GbnfBuilder.compile(routing("minicpm5", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, weather + "\n" + noop));
        ToolParse read = run(routing("minicpm5", ChatRequest.ToolChoice.required(), true), weather + "\n" + noop);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("minicpm5");
        assertTrue(accepts(lazy, "Checking.\n" + weather));
    }

    @Test
    public void step3ForcesItsSteptmlInvokesAndReadsThemBack() {
        String weather = "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
                + "<steptml:invoke name=\"get_weather\"><steptml:parameter name=\"city\">Paris"
                + "</steptml:parameter></steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>";
        String noop = "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
                + "<steptml:invoke name=\"noop\"></steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>";
        String both = "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
                + "<steptml:invoke name=\"get_weather\"><steptml:parameter name=\"city\">Paris"
                + "</steptml:parameter></steptml:invoke><｜tool_call_end｜><｜tool_sep｜>"
                + "<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name=\"noop\">"
                + "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>";
        CompiledConstraint one = GbnfBuilder.compile(routing("step3", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(acceptsText(one, weather));
        assertTrue(acceptsText(one, noop));
        assertFalse("parallel_tool_calls false writes one call", acceptsText(one, both));
        assertFalse("the parameter is pinned", acceptsText(one,
                weather.replace("name=\"city\"", "name=\"town\"")));

        CompiledConstraint many = GbnfBuilder.compile(routing("step3", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(acceptsText(many, both));
        ToolParse read = run(routing("step3", ChatRequest.ToolChoice.required(), true), both);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("step3");
        assertTrue(acceptsText(lazy, "Sure. " + weather));
    }

    @Test
    public void apertusForcesItsSingleKeyArrayAndReadsThemBack() {
        String weather = "<|tools_prefix|>[{\"get_weather\": {\"city\": \"Paris\"}}]<|tools_suffix|>";
        String both = "<|tools_prefix|>[{\"get_weather\": {\"city\": \"Paris\"}}, {\"noop\": {}}]"
                + "<|tools_suffix|>";
        CompiledConstraint one = GbnfBuilder.compile(routing("apertus", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, weather));
        assertFalse(accepts(one, both));
        assertFalse("the argument schema holds", accepts(one,
                "<|tools_prefix|>[{\"get_weather\": {\"town\": \"Paris\"}}]<|tools_suffix|>"));

        CompiledConstraint many = GbnfBuilder.compile(routing("apertus", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, both));
        ToolParse read = run(routing("apertus", ChatRequest.ToolChoice.required(), true), both);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());
        assertEquals("{}", read.calls().get(1).arguments());

        CompiledConstraint lazy = lazyStrict("apertus");
        assertTrue(accepts(lazy, "I will check. " + weather));

        String unsuffixed = "<|tools_prefix|>[{\"get_weather\": {\"city\": \"Paris\"}}]";
        assertTrue("the parser ends the block after the array, so the grammar does too", accepts(one, unsuffixed));
        assertTrue(accepts(many, "<|tools_prefix|>[{\"get_weather\": {\"city\": \"Paris\"}}, {\"noop\": {}}]"));
        assertTrue("a suffix still ends the block", accepts(one, weather));
        ToolParse openEnded = run(routing("apertus", ChatRequest.ToolChoice.required(), false), unsuffixed);
        assertEquals(1, openEnded.calls().size());
        assertEquals("{\"city\": \"Paris\"}", openEnded.calls().get(0).arguments());
    }

    @Test
    public void functiongemmaForcesItsEscapeValuesAndReadsThemBack() {
        String weather = "<start_function_call>call:get_weather{city:<escape>Paris<escape>}"
                + "<end_function_call>";
        String noop = "<start_function_call>call:noop{}<end_function_call>";
        CompiledConstraint one = GbnfBuilder.compile(routing("functiongemma", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, weather));
        assertTrue(accepts(one, noop));
        assertFalse(accepts(one, weather + noop));
        assertFalse("the key is pinned", accepts(one,
                "<start_function_call>call:get_weather{town:<escape>Paris<escape>}<end_function_call>"));

        CompiledConstraint many = GbnfBuilder.compile(routing("functiongemma", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, weather + "\n" + noop));
        ToolParse read = run(routing("functiongemma", ChatRequest.ToolChoice.required(), true),
                weather + "\n" + noop);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("functiongemma");
        assertTrue(accepts(lazy, "Sure. " + weather));

        Map<String, Object> days = Collections.singletonMap("type", "integer");
        Map<String, Object> props = Collections.singletonMap("days", days);
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", props);
        parameters.put("required", Collections.singletonList("days"));
        ChatTool typedTool = ChatTool.builder("get_time").parameters(parameters).strict(Boolean.TRUE).build();
        List<ChatTool> typedTools = Collections.singletonList(typedTool);
        CompiledConstraint typed = GbnfBuilder.compile(NativeToolGrammar.forcing(
                ToolParserRegistry.forName("functiongemma"), typedTools, false, true));
        String daysCall = "<start_function_call>call:get_time{days:<escape>3<escape>}"
                + "<end_function_call>";
        assertTrue(accepts(typed, daysCall));
        assertFalse("a non-string value is its JSON schema", accepts(typed,
                "<start_function_call>call:get_time{days:<escape>three<escape>}<end_function_call>"));
        ToolParse typedRead = ToolParserRegistry.forName("functiongemma").create(typedTools).parse(daysCall, false);
        assertEquals("{\"days\":3}", typedRead.calls().get(0).arguments());
    }

    @Test
    public void gemma4ForcesItsDictSyntaxAndReadsItBack() {
        String weather = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";
        String noop = "<|tool_call>call:noop{}<tool_call|>";
        CompiledConstraint one = GbnfBuilder.compile(routing("gemma4", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, weather));
        assertTrue(accepts(one, noop));
        assertFalse(accepts(one, weather + noop));
        assertFalse("the key is pinned", accepts(one,
                "<|tool_call>call:get_weather{town:<|\"|>Paris<|\"|>}<tool_call|>"));

        CompiledConstraint many = GbnfBuilder.compile(routing("gemma4", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, weather + "\n" + noop));
        ToolParse read = run(routing("gemma4", ChatRequest.ToolChoice.required(), true), weather + "\n" + noop);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        List<ChatTool> loose = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.FALSE).build());
        CompiledConstraint free = GbnfBuilder.compile(ToolRouting.plan(request(loose,
                ChatRequest.ToolChoice.required()), "gemma4", null, NO_REASONING, null,
                ParallelToolCalls.PARALLEL).constraint());
        assertTrue("a loose tool keeps a free dict", accepts(free,
                "<|tool_call>call:get_weather{any:<|\"|>thing<|\"|>,n:3}<tool_call|>"));

        CompiledConstraint lazy = lazyStrict("gemma4");
        assertTrue(accepts(lazy, "Sure. " + weather));

        Map<String, Object> city = new LinkedHashMap<>();
        city.put("type", "string");
        Map<String, Object> tags = new LinkedHashMap<>();
        tags.put("type", "array");
        tags.put("items", Collections.singletonMap("type", "string"));
        Map<String, Object> prefsProps = new LinkedHashMap<>();
        prefsProps.put("city", city);
        prefsProps.put("tags", tags);
        Map<String, Object> prefs = new LinkedHashMap<>();
        prefs.put("type", "object");
        prefs.put("properties", prefsProps);
        prefs.put("required", Collections.singletonList("city"));
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("prefs", prefs);
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", props);
        parameters.put("required", Collections.singletonList("prefs"));
        ChatTool nestedTool = ChatTool.builder("set_prefs").parameters(parameters).strict(Boolean.TRUE).build();
        List<ChatTool> nestedTools = Collections.singletonList(nestedTool);
        CompiledConstraint nested = GbnfBuilder.compile(NativeToolGrammar.forcing(
                ToolParserRegistry.forName("gemma4"), nestedTools, false, true));
        String nestedCall = "<|tool_call>call:set_prefs{prefs:{city:<|\"|>Paris<|\"|>,"
                + "tags:[<|\"|>a<|\"|>]}}<tool_call|>";
        assertTrue("nested dicts and arrays keep the Gemma syntax", accepts(nested, nestedCall));
        ToolParse nestedRead = ToolParserRegistry.forName("gemma4").create(nestedTools).parse(nestedCall, false);
        assertEquals("{\"prefs\":{\"city\":\"Paris\",\"tags\":[\"a\"]}}",
                nestedRead.calls().get(0).arguments());
    }

    @Test
    public void gemma4HoldsStringEnumsPatternsAndLengthsInsideTheDelimiters() {
        Map<String, Object> mode = new LinkedHashMap<>();
        mode.put("type", "string");
        mode.put("enum", Arrays.asList("fast", "slow"));
        Map<String, Object> code = new LinkedHashMap<>();
        code.put("type", "string");
        code.put("minLength", 2);
        code.put("maxLength", 4);
        Map<String, Object> slug = new LinkedHashMap<>();
        slug.put("type", "string");
        slug.put("pattern", "[a-z]+-[0-9]+");
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("mode", mode);
        props.put("code", code);
        props.put("slug", slug);
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", props);
        parameters.put("required", Arrays.asList("mode", "code", "slug"));
        ChatTool tool = ChatTool.builder("pick").parameters(parameters).strict(Boolean.TRUE).build();
        List<ChatTool> constrained = Collections.singletonList(tool);
        CompiledConstraint c = GbnfBuilder.compile(
                NativeToolGrammar.forcing(ToolParserRegistry.forName("gemma4"), constrained, false, true));

        String call = "<|tool_call>call:pick{mode:<|\"|>fast<|\"|>,code:<|\"|>abcd<|\"|>,"
                + "slug:<|\"|>abc-123<|\"|>}<tool_call|>";
        assertTrue(accepts(c, call));
        assertFalse("the enum holds inside the delimiters", accepts(c,
                call.replace("<|\"|>fast<|\"|>", "<|\"|>medium<|\"|>")));
        assertFalse("maxLength holds inside the delimiters", accepts(c,
                call.replace("<|\"|>abcd<|\"|>", "<|\"|>abcde<|\"|>")));
        assertFalse("minLength holds inside the delimiters", accepts(c,
                call.replace("<|\"|>abcd<|\"|>", "<|\"|>a<|\"|>")));
        assertFalse("the pattern holds inside the delimiters", accepts(c,
                call.replace("<|\"|>abc-123<|\"|>", "<|\"|>ABC<|\"|>")));
        ToolParse read = ToolParserRegistry.forName("gemma4").create(constrained).parse(call, false);
        assertEquals("{\"mode\":\"fast\",\"code\":\"abcd\",\"slug\":\"abc-123\"}",
                read.calls().get(0).arguments());

        Map<String, Object> open = new LinkedHashMap<>();
        open.put("type", "string");
        open.put("pattern", ".*");
        Map<String, Object> openProps = new LinkedHashMap<>();
        openProps.put("any", open);
        Map<String, Object> openParameters = new LinkedHashMap<>();
        openParameters.put("type", "object");
        openParameters.put("properties", openProps);
        ChatTool openTool = ChatTool.builder("free").parameters(openParameters).strict(Boolean.TRUE).build();
        CompiledConstraint fallback = GbnfBuilder.compile(NativeToolGrammar.forcing(
                ToolParserRegistry.forName("gemma4"), Collections.singletonList(openTool), false, true));
        assertTrue("a pattern that may write the delimiter falls back to free text", accepts(fallback,
                "<|tool_call>call:free{any:<|\"|>a<b<|\"|>}<tool_call|>"));
    }

    @Test
    public void inklingForcesItsToolJsonBlocksAndReadsThemBack() {
        String weather = "<|message_model|><|content_invoke_tool_json|>{\"name\": \"get_weather\", "
                + "\"args\": {\"city\": \"Paris\"}}<|end_message|>";
        String noop = "<|message_model|><|content_invoke_tool_json|>{\"name\": \"noop\", "
                + "\"args\": {}}<|end_message|>";
        String text = "<|message_model|><|content_text|>Sunny.<|end_message|>";
        CompiledConstraint one = GbnfBuilder.compile(routing("inkling", ChatRequest.ToolChoice.required(),
                false).constraint());
        assertTrue(accepts(one, weather));
        assertFalse("a forcing choice writes calls only", accepts(one, text));
        assertFalse(accepts(one, weather + noop));
        assertFalse("the argument schema holds", accepts(one,
                "<|message_model|><|content_invoke_tool_json|>{\"name\": \"get_weather\", "
                        + "\"args\": {\"town\": \"Paris\"}}<|end_message|>"));

        CompiledConstraint many = GbnfBuilder.compile(routing("inkling", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue(accepts(many, weather + "\n" + noop));
        ToolParse read = run(routing("inkling", ChatRequest.ToolChoice.required(), true), weather + "\n" + noop);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("inkling");
        assertTrue("a text block may precede the call", accepts(lazy, text + weather));
        assertTrue(accepts(lazy, weather));
    }

    @Test
    public void cohereCommand3And4ForceTheirActionArrays() {
        String weather = "<|START_ACTION|>[{\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", "
                + "\"parameters\": {\"city\": \"Paris\"}}]<|END_ACTION|>";
        String both = "<|START_ACTION|>[{\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", "
                + "\"parameters\": {\"city\": \"Paris\"}}, {\"tool_call_id\": \"1\", \"tool_name\": \"noop\", "
                + "\"parameters\": {}}]<|END_ACTION|>";
        for (String format : Arrays.asList("cohere_command3", "cohere_command4")) {
            CompiledConstraint one = GbnfBuilder.compile(routing(format, ChatRequest.ToolChoice.required(),
                    false).constraint());
            assertTrue(format, accepts(one, weather));
            assertFalse(format + " writes one call", accepts(one, both));
            assertFalse(format + " pins the argument schema", accepts(one,
                    weather.replace("\"city\": \"Paris\"", "\"town\": \"Paris\"")));

            CompiledConstraint many = GbnfBuilder.compile(routing(format, ChatRequest.ToolChoice.required(),
                    true).constraint());
            assertTrue(format, accepts(many, both));
            ToolParse read = run(routing(format, ChatRequest.ToolChoice.required(), true), both);
            assertEquals(2, read.calls().size());
            assertEquals("0", read.calls().get(0).id());
            assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());

            CompiledConstraint lazy = lazyStrict(format);
            assertTrue(format, accepts(lazy, "Sure. " + weather));
        }
    }

    @Test
    public void functionaryV32ForcesItsRecipientsAndReadsThemBack() {
        String weather = ">>>get_weather\n{\"city\": \"Paris\"}";
        String noop = ">>>noop\n{}";
        CompiledConstraint one = GbnfBuilder.compile(routing("functionary_v3_2",
                ChatRequest.ToolChoice.required(), false).constraint());
        assertTrue(accepts(one, weather));
        assertFalse(accepts(one, weather + noop));
        assertFalse("the argument schema holds", accepts(one, ">>>get_weather\n{\"town\": \"Paris\"}"));

        CompiledConstraint many = GbnfBuilder.compile(routing("functionary_v3_2",
                ChatRequest.ToolChoice.required(), true).constraint());
        assertTrue(accepts(many, weather + "\n" + noop));
        ToolParse read = run(routing("functionary_v3_2", ChatRequest.ToolChoice.required(), true),
                weather + "\n" + noop);
        assertEquals(2, read.calls().size());
        assertEquals("{\"city\": \"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint lazy = lazyStrict("functionary_v3_2");
        assertTrue("a reply may open the turn", accepts(lazy, ">>>all\nLet me check."));
        assertTrue("the reply may precede the calls", accepts(lazy, ">>>all\nLet me check.\n" + weather));
        assertTrue(accepts(lazy, weather));
        assertFalse("a cut-off call is refused", accepts(lazy, ">>>all\nLet me check.\n"
                + ">>>get_weather\n{\"city\": "));
    }

    @Test
    public void hunyuanFormatsBakeTheTemplateSuffixIntoTheirCalls() {
        String template = "{%- set HYTK = ':opensource' %}\n"
                + "{%- set toolcalls_begin_token = '<tool_calls{}>'.format(HYTK) %}"
                + "{%- set think_begin_token = '<think{}>'.format(HYTK) %}"
                + "{%- set tool_call_begin_token = '<tool_call{}>'.format(HYTK) %}"
                + "{%- set tool_sep_token = '<tool_sep{}>'.format(HYTK) %}{{ '<arg_key' }}";
        ToolParser.Factory v3 = ToolParserRegistry.detect(template);
        assertEquals("hy_v3", v3.name());
        String weather = "<tool_call:opensource>get_weather<tool_sep:opensource>"
                + "<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris"
                + "</arg_value:opensource></tool_call:opensource>";
        String one = "<tool_calls:opensource>" + weather + "</tool_calls:opensource>";
        String both = "<tool_calls:opensource>" + weather
                + "<tool_call:opensource>noop<tool_sep:opensource></tool_call:opensource>"
                + "</tool_calls:opensource>";
        CompiledConstraint single = GbnfBuilder.compile(NativeToolGrammar.forcing(v3, tools(), false, true));
        assertTrue(accepts(single, one));
        assertFalse("the separator is part of the format", accepts(single,
                one.replace("<tool_sep:opensource>", "")));
        assertFalse(accepts(single, both));
        ToolParse read = v3.create(tools()).parse(one, false);
        assertEquals(1, read.calls().size());
        assertEquals("{\"city\":\"Paris\"}", read.calls().get(0).arguments());

        CompiledConstraint parallel = GbnfBuilder.compile(NativeToolGrammar.forcing(v3, tools(), true, false));
        assertTrue(accepts(parallel, both));
        ToolParse readBoth = v3.create(tools()).parse(both, false);
        assertEquals(2, readBoth.calls().size());
        assertEquals("get_weather", readBoth.calls().get(0).name());
        assertEquals("noop", readBoth.calls().get(1).name());
        assertTrue("a bare call is an opener too", accepts(GbnfBuilder.compile(
                NativeToolGrammar.lazy(v3, tools(), v3.create(tools()).triggers())), weather));

        CompiledConstraint unsuffixed = GbnfBuilder.compile(routing("hy_v3", ChatRequest.ToolChoice.required(),
                true).constraint());
        assertTrue("the registry's plain HY v3 writes the bare tokens", accepts(unsuffixed,
                "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key>"
                        + "<arg_value>Paris</arg_value></tool_call></tool_calls>"));

        String v4Template = "{%- set HYTK = ':opensource' %}\n"
                + "{%- set toolcalls_begin_token = '<tool_calls{}>'.format(HYTK) %}"
                + "{%- set think_begin_token = '<think{}>'.format(HYTK) %}"
                + "{%- set tool_call_begin_token = '<tool_call{}>'.format(HYTK) %}{{ '<arg_key' }}";
        ToolParser.Factory v4 = ToolParserRegistry.detect(v4Template);
        assertEquals("hy_v4", v4.name());
        String v4Call = "<tool_calls:opensource><tool_call:opensource>get_weather"
                + "<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris"
                + "</arg_value:opensource></tool_call:opensource></tool_calls:opensource>";
        CompiledConstraint v4Forced = GbnfBuilder.compile(NativeToolGrammar.forcing(v4, tools(), false, true));
        assertTrue(accepts(v4Forced, v4Call));
        assertFalse("HY v4 writes no separator", accepts(v4Forced, v4Call.replace("<arg_key:opensource>",
                "<tool_sep:opensource><arg_key:opensource>")));
        ToolParse readV4 = v4.create(tools()).parse(v4Call, false);
        assertEquals("{\"city\":\"Paris\"}", readV4.calls().get(0).arguments());
    }

    /** The lazy grammar of one format over a tool that declares strict. */
    private static CompiledConstraint lazyStrict(String format) {
        List<ChatTool> flagged = Collections.singletonList(ChatTool.builder("get_weather")
                .parameters(strictTool()).strict(Boolean.TRUE).build());
        return GbnfBuilder.compile(ToolRouting.plan(request(flagged, ChatRequest.ToolChoice.auto()), format,
                null, NO_REASONING, null, ParallelToolCalls.PARALLEL).constraint());
    }

    // ------------------------------------------------------------------ helpers

    private static final String HERMES_TEMPLATE =
            "{% for t in tools %}<tool_call>{{ t }}</tool_call>{% endfor %}";

    /** A plan over the shared tools for one format and choice. */
    private static ToolRouting routing(String format, ChatRequest.ToolChoice choice, boolean parallel) {
        return ToolRouting.plan(chat(choice), format, null, NO_REASONING, null,
                parallel ? ParallelToolCalls.PARALLEL : ParallelToolCalls.SINGLE);
    }

    /** The shared tools offered required, with K2 Horizon's {@code tool_call_format} template variable. */
    private static ChatRequest templateKwarg(String callFormat) {
        Map<String, Object> user = new LinkedHashMap<>();
        user.put("role", "user");
        user.put("content", "weather?");
        return ChatRequest.builder().addMessage(user).tools(tools())
                .toolChoice(ChatRequest.ToolChoice.required())
                .templateKwarg("tool_call_format", callFormat).build();
    }

    /** A request whose tools carry no schemas: the envelope is all the format pins. */
    private static ChatRequest request(List<ChatTool> tools, ChatRequest.ToolChoice choice) {
        Map<String, Object> user = new LinkedHashMap<>();
        user.put("role", "user");
        user.put("content", "weather?");
        return ChatRequest.builder().addMessage(user).tools(tools).toolChoice(choice).build();
    }

    /** Whether the constraint lets greedy decoding write {@code text} (UTF-8, byte by byte) and then end. */
    private static boolean acceptsText(CompiledConstraint c, String text) {
        SamplerVocab vocab = new Utf8Vocab();
        SamplerStage stage = c.gated() ? new TriggeredGrammarStage(c.grammar(), c.triggers(), vocab)
                : new GrammarStage(c.grammar(), vocab);
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.singletonList(stage), vocab);
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (int i = 0; i <= bytes.length; i++) {
            int want = i < bytes.length ? (bytes[i] & 0xFF) + 1 : 0;
            float[] logits = new float[257];
            Arrays.fill(logits, -100f);
            logits[want] = 10f;
            int token;
            try {
                token = sampler.sample(logits);
            } catch (RuntimeException e) {
                return false;
            }
            if (token != want) {
                return false;
            }
            if (want != 0) {
                sampler.accept(token);
            }
        }
        return true;
    }

    /** Single-byte tokens: id {@code b + 1} carries byte {@code b}, id 0 ends the generation. */
    private static final class Utf8Vocab implements SamplerVocab {
        @Override public int vocabSize() { return 257; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id <= 0 ? new byte[0] : new byte[]{(byte) (id - 1)}; }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            byte[] b = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[b.length];
            for (int i = 0; i < b.length; i++) {
                ids[i] = (b[i] & 0xFF) + 1;
            }
            return ids;
        }
    }

    /** The get_weather schema, for the strict-flagged tool of the auto test. */
    private static Map<String, Object> strictTool() {
        return tools().get(0).parameters();
    }

    /** Streams {@code text} through the routing's output, as ToolRoutingTest does. */
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
