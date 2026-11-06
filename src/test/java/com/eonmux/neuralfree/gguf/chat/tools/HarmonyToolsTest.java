package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.HarmonyDialect;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.chat.tools.ToolTestSupport.accepts;
import static com.eonmux.neuralfree.gguf.chat.tools.ToolTestSupport.chat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Harmony-family tool calls: the forcing grammars of gpt-oss and Muse (reasoning and preamble messages first, then
 * a call over the request's tools), and Muse's ATEM tool messages read back as calls with schema-typed values.
 */
public class HarmonyToolsTest {

    private static final String GPT_OSS_TEMPLATE = "<|start|>assistant<|channel|>final<|message|>{{ content }}";
    private static final String MUSE_TEMPLATE = "<|start|>assistant to=user<|message|>{{ content }}<|eot|>"
            + "{% if x %}<|eom|>{% endif %}";
    private static final String HEADER = "<|start|>user<|message|>weather?<|end|><|start|>assistant";

    @Test
    public void gptOssRequiredLetsTheModelReasonThenForcesACall() {
        CompiledConstraint c = forcing(ChatRequest.ToolChoice.required(), GPT_OSS_TEMPLATE, HEADER);
        assertFalse("the grammar holds from the first token", c.gated());
        assertTrue(accepts(c, "<|channel|>analysis<|message|>The user wants weather.<|end|><|start|>assistant"
                + "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}"));
        assertTrue("the recipient may sit in the role header", accepts(c,
                " to=functions.noop<|channel|>commentary<|message|>{}"));
        assertTrue("a preamble may precede the call", accepts(c, "<|channel|>commentary<|message|>Checking."
                + "<|end|><|start|>assistant<|channel|>commentary to=functions.noop<|message|>{}"));
        assertFalse("a final answer is no call", accepts(c, "<|channel|>final<|message|>It is sunny."));
        assertFalse("an undeclared function is refused", accepts(c,
                "<|channel|>commentary to=functions.other<|message|>{}"));
        assertFalse("the arguments follow the schema", accepts(c,
                "<|channel|>commentary to=functions.get_weather<|message|>{\"town\": \"Paris\"}"));
    }

    @Test
    public void gptOssNamedFunctionAdmitsOnlyThatFunction() {
        CompiledConstraint c = forcing(ChatRequest.ToolChoice.function("noop"), GPT_OSS_TEMPLATE, HEADER);
        assertTrue(accepts(c, "<|channel|>commentary to=functions.noop<|message|>{}"));
        assertFalse(accepts(c, "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Rome\"}"));
    }

    @Test
    public void aPromptLeftInsideReasoningStartsTheGrammarInTheBody() {
        CompiledConstraint c = forcing(ChatRequest.ToolChoice.required(), GPT_OSS_TEMPLATE,
                HEADER + "<|channel|>analysis<|message|>");
        assertTrue(accepts(c, "Need a tool.<|end|><|start|>assistant<|channel|>commentary to=functions.noop"
                + "<|message|>{}"));
        assertFalse("the reasoning body cannot run into the call header", accepts(c,
                "<|channel|>commentary to=functions.noop<|message|>{}"));
    }

    @Test
    public void museRequiredForcesAnAtemToolMessage() {
        CompiledConstraint c = forcing(ChatRequest.ToolChoice.required(), MUSE_TEMPLATE, "<|start|>assistant");
        String call = " to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\">"
                + "<atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke></atem:function_calls>";
        assertTrue(accepts(c, " to=self<|message|>Look it up.<|eom|><|start|>assistant" + call));
        assertTrue("parallel calls follow one another", accepts(c, call + "<|eom|><|start|>assistant" + call));
        assertFalse("a user answer is no call", accepts(c, " to=user<|message|>It is sunny."));
        assertFalse("an undeclared function is refused", accepts(c, " to=x<|message|><atem:function_calls>"
                + "<atem:invoke name=\"other\"></atem:invoke></atem:function_calls>"));
    }

    @Test
    public void museToolMessagesBecomeCallsWithTypedValues() {
        ToolRouting r = ToolRouting.plan(museChat(), "auto", MUSE_TEMPLATE, ReasoningFilter.harmony(), null, true);
        assertNull("auto adds no grammar", r.constraint());
        ToolOutput out = r.newOutput(() -> "call_m");
        ToolTestSupport.Recorder rec = new ToolTestSupport.Recorder();
        out.content("On it.", rec);
        out.toolCallStarted("forecast", rec);
        String body = "<atem:function_calls>\n<atem:invoke name=\"forecast\">\n<atem:parameter name=\"city\">"
                + "Oslo</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n</atem:invoke>\n"
                + "</atem:function_calls>";
        for (int i = 0; i < body.length(); i += 7) {
            out.toolCallText(body.substring(i, Math.min(body.length(), i + 7)), rec);
        }
        out.toolCallStarted("python", rec);
        out.toolCallText("print(1)", rec);
        ToolParse p = out.finish(rec);
        assertEquals("On it.", p.content());
        assertEquals(1, p.calls().size());
        assertEquals("forecast", p.calls().get(0).name());
        assertEquals("{\"city\":\"Oslo\",\"days\":3}", p.calls().get(0).arguments());
        assertEquals("call_m", p.calls().get(0).id());
        assertEquals("the streamed arguments are the final ones", p.calls().get(0).arguments(),
                rec.args.get(0).toString());
    }

    @Test
    public void theDialectFollowsTheTemplate() {
        assertEquals(HarmonyDialect.MUSE, HarmonyDialect.of(MUSE_TEMPLATE));
        assertEquals(HarmonyDialect.GPT_OSS, HarmonyDialect.of(GPT_OSS_TEMPLATE));
        assertEquals(HarmonyDialect.GPT_OSS, HarmonyDialect.of(null));
    }

    private static CompiledConstraint forcing(ChatRequest.ToolChoice choice, String template, String prompt) {
        ToolRouting r = ToolRouting.plan(chat(choice), "auto", template, ReasoningFilter.harmony(), prompt, true);
        assertTrue(r.active());
        return GbnfBuilder.compile(r.constraint());
    }

    private static ChatRequest museChat() {
        Map<String, Object> city = new LinkedHashMap<>();
        city.put("type", "string");
        Map<String, Object> days = new LinkedHashMap<>();
        days.put("type", "integer");
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("city", city);
        props.put("days", days);
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("type", "object");
        params.put("properties", props);
        List<ChatTool> tools = Arrays.asList(ChatTool.builder("forecast").parameters(params).build());
        Map<String, Object> user = new LinkedHashMap<>();
        user.put("role", "user");
        user.put("content", "forecast?");
        return ChatRequest.builder().addMessage(user).tools(tools).toolChoice(ChatRequest.ToolChoice.auto())
                .build();
    }
}
