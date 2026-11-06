package com.eonmux.neuralfree.gguf.chat.response;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;
import com.eonmux.neuralfree.gguf.grammar.GrammarMatcher;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.chat.response.TemplateGrammarTest.tool;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The grammar of the {@code beam} parser for one request ({@link BeamGrammar}) over the Beam checkpoints' response
 * template, vLLM's {@code tests/parser/test_beam_parser.py} request cases: forced, automatic and forbidden tool calls,
 * the reasoning the message opens with, a prompt that left a region open, structured outputs framed as the answer, and
 * the guarantees refused by name where the template cannot express them.
 */
public class BeamGrammarTest {

    /** The {@code response_template} of the Beam checkpoints. */
    static final String BEAM_TEMPLATE = "{'defaults': {'role': 'assistant'},"
            + " 'start_anchor_pattern': '<\\\\|start\\\\|>assistant', 'fields': {"
            + "'thinking': {'open': '<|channel|>analysis<|message|>', 'close': '<|end|><|start|>assistant',"
            + " 'content': 'text'},"
            + "'content': {'open': '<|channel|>final<|message|>', 'close': ['<|return|>', '<|end|>', '<|call|>'],"
            + " 'content': 'text'},"
            + "'tool_calls': {'open_pattern': ' to=(?P<name>[^\\\\s<]+)<\\\\|channel\\\\|>analysis"
            + " <\\\\|constrain\\\\|>xml<\\\\|message\\\\|>', 'close': ['<|end|><|start|>assistant', '<|call|>'],"
            + " 'repeats': true, 'content': 'xml-inline', 'content_args': {"
            + "'tag_pattern': '<rfl:parameter name=\\'(?P<key>[^\\']+)\\'>\\\\n(?P<value>.*?)\\\\n"
            + "(?:</rfl:parameter>|(?=<rfl:parameter name=\\'))',"
            + " 'value_parser': {'name': 'text', 'args': {'strip': false}}},"
            + " 'transform': {'type': 'function', 'function': {'name': '{name}', 'arguments': '{content}'}}}}}";
    private static final TemplateParsing BEAM = TemplateParsing.of("beam",
            ResponseTemplate.load(TemplateGrammarTest.json(BEAM_TEMPLATE)));
    private static final String PROMPT = "<|start|>assistant";
    private static final String THOUGHT = "<|channel|>analysis<|message|>plan<|end|><|start|>assistant";
    private static final String FINAL = "<|channel|>final<|message|>";
    private static final String NEXT_MESSAGE = "<|end|><|start|>assistant";
    private static final List<ChatTool> STRICT_TOOLS = Arrays.asList(
            beamTool("get_weather", "{'city': {'type': 'string'}}", null),
            beamTool("set_count", "{'count': {'type': 'integer'}}", true));
    private static final List<ChatTool> TOOLS = Arrays.asList(
            beamTool("get_weather", "{'city': {'type': 'string'}}", null),
            beamTool("set_count", "{'count': {'type': 'integer'}}", null));
    private static final String WEATHER_CALL = call("get_weather", parameter("city", "Paris"));
    private static final String STRUCTURAL_OUTPUT = "{\"format\": {\"type\": \"const_string\", \"value\": \"x\"}}";

    private static ChatTool beamTool(String name, String properties, Boolean strict) {
        return tool("{'type': 'object', 'properties': " + properties + "}", name, strict);
    }

    private static String parameter(String name, String value) {
        return "<rfl:parameter name=\"" + name + "\">\n" + value + "\n</rfl:parameter>";
    }

    private static String call(String name, String body) {
        return call(name, body, "<|call|>");
    }

    private static String call(String name, String body, String terminator) {
        return " to=" + name + "<|channel|>analysis <|constrain|>xml<|message|>" + body + terminator;
    }

    private static ChatRequest chat(List<ChatTool> tools, ChatRequest.ToolChoice choice) {
        ChatRequest.Builder b = ChatRequest.builder().addMessage(Collections.singletonMap("role", "user"));
        if (tools != null) b.tools(tools);
        if (choice != null) b.toolChoice(choice);
        return b.build();
    }

    /** vLLM's {@code adjust}: a beam reasoning and tool parser, reasoning shown, after {@code prompt}. */
    private static BeamGrammar.Request request(ChatRequest chat, String prompt) {
        return BeamGrammar.Request.of(chat).prompt(prompt).toolParsing(true).reasoningGated(true);
    }

    private static BeamGrammar adjust(ChatRequest chat) {
        return BeamGrammar.derive(BEAM, request(chat, PROMPT));
    }

    private static boolean accepts(BeamGrammar grammar, String text) {
        assertNotNull("a grammar is derived", grammar);
        return GrammarMatcher.matches(GbnfBuilder.grammarOf(grammar.constraint()), text);
    }

    private static void refuses(BeamGrammar.Request request, String guarantee, String parameter) {
        try {
            BeamGrammar.derive(BEAM, request);
            fail("expected a refusal naming " + guarantee);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(guarantee));
            assertTrue(e.getMessage(), e.getMessage().contains("'" + parameter + "'"));
        }
    }

    @Test
    public void forcedToolCallsFollowTheTemplateGrammarFromTheMessageStart() {
        BeamGrammar required = adjust(chat(STRICT_TOOLS, ChatRequest.ToolChoice.required()));
        assertTrue(required.fromMessageStart());
        assertTrue(accepts(required, THOUGHT + WEATHER_CALL));
        assertFalse("the message opens with reasoning", accepts(required, WEATHER_CALL));
        assertTrue(accepts(required, THOUGHT + call("get_weather", parameter("city", "Paris"), NEXT_MESSAGE)
                + call("set_count", parameter("count", "7"))));
        assertFalse(accepts(required, THOUGHT + call("set_count", parameter("count", "x"))));
        assertFalse(accepts(required, THOUGHT + call("get_weather", parameter("city", "a<|call|>b"))));
        assertFalse(accepts(required, THOUGHT + FINAL + "Paris<|return|>"));
        BeamGrammar setCount = adjust(chat(STRICT_TOOLS, ChatRequest.ToolChoice.function("set_count")));
        assertFalse(accepts(setCount, THOUGHT + WEATHER_CALL));
        BeamGrammar weather = adjust(chat(STRICT_TOOLS, ChatRequest.ToolChoice.function("get_weather")));
        assertTrue(accepts(weather, THOUGHT + WEATHER_CALL));
        assertFalse("a named choice makes one call", accepts(weather, THOUGHT
                + call("get_weather", parameter("city", "Paris"), NEXT_MESSAGE) + WEATHER_CALL));
    }

    @Test
    public void theReasoningMayBeSkippedWhenThinkingIsDisabled() {
        ChatRequest chat = chat(STRICT_TOOLS, ChatRequest.ToolChoice.required()).toBuilder().enableThinking(false)
                .build();
        BeamGrammar g = adjust(chat);
        assertTrue(accepts(g, WEATHER_CALL));
        assertTrue(accepts(g, THOUGHT + WEATHER_CALL));
    }

    @Test
    public void parallelToolCallsFalseAllowsASingleCall() {
        BeamGrammar g = BeamGrammar.derive(BEAM,
                request(chat(STRICT_TOOLS, ChatRequest.ToolChoice.required()), PROMPT)
                        .parallelToolCalls(ParallelToolCalls.SINGLE));
        assertTrue(accepts(g, THOUGHT + WEATHER_CALL));
        assertFalse(accepts(g, THOUGHT + call("get_weather", parameter("city", "Paris"), NEXT_MESSAGE)
                + WEATHER_CALL));
    }

    @Test
    public void aDefaultedSingleCallLimitStillHoldsAndAsksForNothing() {
        // the template's one-call default (llama.cpp server-common.cpp:1447) enforces like an explicit false,
        // and vLLM refuses only request.parallel_tool_calls is False
        BeamGrammar g = BeamGrammar.derive(BEAM,
                request(chat(STRICT_TOOLS, ChatRequest.ToolChoice.required()), PROMPT)
                        .parallelToolCalls(ParallelToolCalls.of(null, false)));
        assertTrue(accepts(g, THOUGHT + WEATHER_CALL));
        assertFalse(accepts(g, THOUGHT + call("get_weather", parameter("city", "Paris"), NEXT_MESSAGE)
                + WEATHER_CALL));
    }

    @Test
    public void anExplicitFalseLosesTheGuaranteeWhereTheDefaultDoesNot() {
        // a prompt that left the answer open derives no call grammar: the explicit false is refused by name,
        // the template's default asks for nothing
        BeamGrammar.Request lost = request(chat(TOOLS, ChatRequest.ToolChoice.auto()), PROMPT + FINAL)
                .output(ConstraintSpec.jsonSchema(
                        "{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"}}}", null));
        refuses(lost.parallelToolCalls(ParallelToolCalls.SINGLE), "parallel_tool_calls=False",
                "parallel_tool_calls");
        assertNull(BeamGrammar.derive(BEAM, lost.parallelToolCalls(ParallelToolCalls.of(null, false))));
        assertNull("the output stays the request's own constraint either way",
                BeamGrammar.derive(BEAM, lost.parallelToolCalls(ParallelToolCalls.PARALLEL)));
    }

    @Test
    public void parametersAreSeparatedByAtMostANewline() {
        String[][] cases = {{"", "true"}, {"\n", "true"}, {"\n\n", "false"}, {" ", "false"}};
        for (boolean strict : new boolean[]{false, true}) {
            ChatTool write = beamTool("write", "{'path': {'type': 'string'}, 'content': {'type': 'string'}}", strict);
            BeamGrammar g = adjust(chat(Collections.singletonList(write), ChatRequest.ToolChoice.required()));
            for (String[] c : cases) {
                String body = parameter("path", "a.py") + c[0] + parameter("content", "x");
                boolean accepted = Boolean.parseBoolean(c[1]);
                assertEquals(strict + " '" + c[0] + "'", accepted, accepts(g, THOUGHT + call("write", body)));
                assertEquals(strict + " leading '" + c[0] + "'", accepted,
                        accepts(g, THOUGHT + call("write", c[0] + body)));
            }
        }
    }

    @Test
    public void forcedCallsInPrefilledReasoningStartAfterIt() {
        BeamGrammar g = BeamGrammar.derive(BEAM, request(chat(STRICT_TOOLS, ChatRequest.ToolChoice.required()),
                PROMPT + "<|channel|>analysis<|message|>plan"));
        assertFalse("the reasoning gate defers the grammar", g.fromMessageStart());
        assertTrue(accepts(g, WEATHER_CALL));
    }

    @Test
    public void forcedCallsCannotFollowAPrefilledAnswer() {
        refuses(request(chat(STRICT_TOOLS, ChatRequest.ToolChoice.required()), PROMPT + FINAL),
                "required tool choice", "tool_choice");
    }

    @Test
    public void aPrefilledAnswerCanOnlyBeFinished() {
        BeamGrammar g = BeamGrammar.derive(BEAM, request(chat(TOOLS, null), PROMPT + FINAL));
        assertTrue(accepts(g, "Sunny.<|return|>"));
        assertFalse(accepts(g, "Sunny." + NEXT_MESSAGE + WEATHER_CALL));
    }

    @Test
    public void aForcedCallWithAnUnencodableToolNameIsRefusedOnTools() {
        ChatTool spaced = beamTool("get weather", "{'city': {'type': 'string'}}", null);
        refuses(request(chat(Collections.singletonList(spaced), ChatRequest.ToolChoice.required()), PROMPT),
                "tool name", "tools");
    }

    @Test
    public void stringFormatsAndPatternsAreHeldInRawText() {
        ChatTool contact = tool("{'type': 'object', 'properties': {'email': {'type': 'string', 'format': 'email'},"
                + " 'code': {'type': 'string', 'pattern': '^[A-Z]{3}$'}, 'n': {'type': 'integer'}},"
                + " 'required': ['email', 'code', 'n']}", "set_contact", true);
        BeamGrammar g = adjust(chat(Collections.singletonList(contact), ChatRequest.ToolChoice.required()));
        assertTrue(accepts(g, contactCall("a.b@x.co", "ABC", parameter("n", "1"))));
        assertFalse(accepts(g, contactCall("ab", "ABC", parameter("n", "1"))));
        assertFalse(accepts(g, contactCall("a.b@x.co", "ABCD", parameter("n", "1"))));
        assertFalse(accepts(g, contactCall("a.b@x.co", "ABC", "")));
        // line breaks could end the value
        ChatTool dotted = beamTool("set_code", "{'code': {'type': 'string', 'pattern': '^a.b$'}}", true);
        refuses(request(chat(Collections.singletonList(dotted), ChatRequest.ToolChoice.required()), PROMPT),
                "required tool choice", "tool_choice");
    }

    private static String contactCall(String email, String code, String rest) {
        return THOUGHT + call("set_contact", parameter("email", email) + parameter("code", code) + rest);
    }

    @Test
    public void strictToolsConstrainTheAutomaticToolChoice() {
        BeamGrammar g = adjust(chat(STRICT_TOOLS, null));
        assertTrue(accepts(g, THOUGHT + WEATHER_CALL));
        assertTrue(accepts(g, THOUGHT + FINAL + "Sunny.<|return|>"));
        assertFalse(accepts(g, THOUGHT + FINAL + "Sunny." + NEXT_MESSAGE + WEATHER_CALL));
        assertFalse(accepts(g, THOUGHT + call("set_count", parameter("count", "x"))));
    }

    @Test
    public void toolChoiceNoneAllowsOnlyAnAnswer() {
        BeamGrammar g = adjust(chat(TOOLS, ChatRequest.ToolChoice.none()));
        assertTrue(g.fromMessageStart());
        assertTrue(accepts(g, THOUGHT + FINAL + "Sunny.<|return|>"));
        assertFalse(accepts(g, FINAL + "Sunny.<|return|>"));
        assertFalse(accepts(g, WEATHER_CALL));
        assertFalse(accepts(g, THOUGHT + WEATHER_CALL));
        assertFalse(accepts(g, FINAL + "Sunny." + NEXT_MESSAGE + WEATHER_CALL));
    }

    @Test
    public void theAlwaysConstrainedOutputFollowsTheTemplateFraming() {
        String countCall = THOUGHT + call("set_count", parameter("count", "3"));
        for (boolean withTools : new boolean[]{true, false}) {
            BeamGrammar g = adjust(chat(withTools ? TOOLS : null, null));
            assertTrue(accepts(g, THOUGHT + FINAL + "Sunny.<|return|>"));
            assertFalse(accepts(g, FINAL + "Sunny.<|return|>"));
            assertFalse(accepts(g, THOUGHT.replace("plan", "") + FINAL + "Sunny.<|return|>"));
            assertFalse(accepts(g, THOUGHT + "Sunny."));
            assertEquals(withTools, accepts(g, countCall));
        }
    }

    @Test
    public void anAutomaticChoiceLetsCallsNameUndeclaredTools() {
        String undeclared = THOUGHT + call("read_file", parameter("path", "a.py"));
        for (Boolean strict : new Boolean[]{null, true}) {
            List<ChatTool> bash = Collections.singletonList(beamTool("bash", "{'command': {'type': 'string'}}",
                    strict));
            for (ChatRequest.ToolChoice choice : Arrays.asList(ChatRequest.ToolChoice.auto(),
                    ChatRequest.ToolChoice.required())) {
                boolean auto = choice.kind() == ChatRequest.ToolChoice.Kind.AUTO;
                List<ConstraintSpec> outputs = auto ? Arrays.asList(null, ConstraintSpec.jsonObject(null))
                        : Collections.<ConstraintSpec>singletonList(null);
                for (ConstraintSpec output : outputs) {
                    BeamGrammar g = BeamGrammar.derive(BEAM, request(chat(bash, choice), PROMPT).output(output));
                    assertTrue(accepts(g, THOUGHT + call("bash", parameter("command", "ls"))));
                    assertEquals(strict + " " + choice + " " + output, auto, accepts(g, undeclared));
                }
            }
        }
    }

    @Test
    public void automaticCallsEnforceDeclaredArgumentsBesideAnyOtherName() {
        ChatTool find = tool("{'type': 'object', 'properties': {'q': {'type': 'string'}}, 'minProperties': 1}",
                "find", null);
        BeamGrammar g = adjust(chat(Arrays.asList(STRICT_TOOLS.get(0), STRICT_TOOLS.get(1), find), null));
        String twice = parameter("x", "1") + parameter("x", "y");
        assertTrue(accepts(g, THOUGHT + call("set_count", parameter("count", "1"))));
        assertFalse(accepts(g, THOUGHT + call("set_count", parameter("count", "x"))));
        assertFalse(accepts(g, THOUGHT + call("get_weather", parameter("town", "Paris"))));
        // the grammar cannot enforce minProperties
        assertTrue(accepts(g, THOUGHT + call("find", twice)));
        assertTrue(accepts(g, THOUGHT + call("search_files", twice)));
        assertTrue(accepts(g, THOUGHT + call("set_coun", twice)));
        assertTrue(accepts(g, THOUGHT + call("set_counts", twice)));
    }

    @Test
    public void aValueEndsBeforeAnOpenerOnTheNextLine() {
        ChatTool write = beamTool("write", "{'path': {'type': 'string'}, 'content': {'type': 'string'}}", null);
        BeamGrammar g = adjust(chat(Collections.singletonList(write), ChatRequest.ToolChoice.required()));
        String body = "<rfl:parameter name=\"content\">\nprint(1)\n\n" + parameter("path", "a.py");
        assertTrue(accepts(g, THOUGHT + call("write", body)));
    }

    @Test
    public void theAlwaysConstrainedOutputCanBeTurnedOff() {
        assertNull(BeamGrammar.derive(BEAM, request(chat(TOOLS, null), PROMPT).alwaysConstrain(false)));
        assertNotNull("a guarantee still derives the grammar", BeamGrammar.derive(BEAM,
                request(chat(TOOLS, ChatRequest.ToolChoice.required()), PROMPT).alwaysConstrain(false)));
        assertNotNull("a strict tool still derives the grammar", BeamGrammar.derive(BEAM,
                request(chat(STRICT_TOOLS, null), PROMPT).alwaysConstrain(false)));
        assertNotNull("so does the function strict level", BeamGrammar.derive(BEAM,
                request(chat(TOOLS, null), PROMPT).alwaysConstrain(false).functionStrict(true)));
    }

    @Test
    public void strictToolsTheTemplateCannotHoldAreRefusedByName() {
        ChatTool dotted = beamTool("set_code", "{'code': {'type': 'string', 'pattern': '^a.b$'}}", true);
        refuses(request(chat(Collections.singletonList(dotted), null), PROMPT), "strict tools", "tools");
    }

    @Test
    public void aStructuredOutputIsFramedAsTheAnswer() {
        ConstraintSpec schema = ConstraintSpec.jsonSchema(
                "{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"integer\"}}}", null);
        for (boolean withTools : new boolean[]{false, true}) {
            BeamGrammar g = BeamGrammar.derive(BEAM, request(chat(withTools ? TOOLS : null, null), PROMPT)
                    .output(schema));
            assertTrue(accepts(g, THOUGHT + FINAL + "{\"a\": 1}<|return|>"));
            assertFalse(accepts(g, THOUGHT + FINAL + "Sunny.<|return|>"));
            assertEquals(withTools, accepts(g, THOUGHT + WEATHER_CALL));
        }
        BeamGrammar choice = BeamGrammar.derive(BEAM, request(chat(null, null), PROMPT)
                .output(ConstraintSpec.choice(Arrays.asList("yes", "no"))));
        assertTrue(accepts(choice, THOUGHT + FINAL + "yes<|return|>"));
        assertFalse(accepts(choice, THOUGHT + FINAL + "maybe<|return|>"));
        BeamGrammar regex = BeamGrammar.derive(BEAM, request(chat(null, null), PROMPT)
                .output(ConstraintSpec.regex("a.c", false)));
        assertTrue(accepts(regex, THOUGHT + FINAL + "abc"));
        assertFalse("the request's regex keeps its line semantics", accepts(regex, THOUGHT + FINAL + "a\nc"));
    }

    @Test
    public void aJsonObjectAnswerIsAnObject() {
        BeamGrammar g = BeamGrammar.derive(BEAM, request(chat(null, null), PROMPT)
                .output(ConstraintSpec.jsonObject(null)));
        assertTrue(accepts(g, THOUGHT + FINAL + "{\"a\": 1}<|return|>"));
        assertFalse("vLLM's {\"type\": \"object\"}", accepts(g, THOUGHT + FINAL + "[1]<|return|>"));
        assertFalse(accepts(g, THOUGHT + FINAL + "1<|return|>"));
        BeamGrammar compact = BeamGrammar.derive(BEAM, request(chat(null, null), PROMPT)
                .output(ConstraintSpec.jsonObject(JsonSchemaToGbnf.WhitespaceMode.COMPACT)));
        assertTrue(accepts(compact, THOUGHT + FINAL + "{\"a\":1}<|return|>"));
        assertFalse("the request's whitespace mode holds", accepts(compact, THOUGHT + FINAL + "{\"a\": 1}<|return|>"));
        assertFalse(accepts(compact, THOUGHT + FINAL + "[1]<|return|>"));
    }

    @Test
    public void anOutputWithoutAnAnswerFormatKeepsTheRequestsOwnConstraint() {
        // vLLM leaves a structural tag, or a constraint xgrammar cannot hold, as the request's own
        assertNull(BeamGrammar.derive(BEAM, request(chat(null, null), PROMPT)
                .output(ConstraintSpec.structuralTag(STRUCTURAL_OUTPUT))));
        assertNull(BeamGrammar.derive(BEAM, request(chat(TOOLS, null), PROMPT)
                .output(ConstraintSpec.structuralTag(STRUCTURAL_OUTPUT))));
        assertNull("so does an output inside an open reasoning without the gate", BeamGrammar.derive(BEAM,
                request(chat(null, null), PROMPT + "<|channel|>analysis<|message|>").reasoningGated(false)
                        .output(ConstraintSpec.jsonObject(null))));
    }

    @Test
    public void anAnswerWithoutAnOpenerKeepsTheRequestsOwnConstraint() {
        TemplateParsing implicit = TemplateParsing.of("beam", ResponseTemplate.parse("{\"start_anchor\": \"<s>\","
                + " \"fields\": {\"thinking\": {\"open\": \"<t>\", \"close\": \"</t>\"}, \"content\": {}}}"));
        assertNull(BeamGrammar.derive(implicit, request(chat(null, null), "<s>")
                .output(ConstraintSpec.jsonObject(null))));
        assertNull("nor is the framing alone derived without an answer opener",
                BeamGrammar.derive(implicit, request(chat(null, null), "<s>")));
    }

    @Test
    public void theRemainingRefusalsNameTheirParameter() {
        refuses(request(chat(TOOLS, ChatRequest.ToolChoice.required()), PROMPT).output(ConstraintSpec.jsonObject(null)),
                "cannot be combined", "response_format");
        refuses(request(chat(STRICT_TOOLS, null), PROMPT).output(ConstraintSpec.structuralTag(STRUCTURAL_OUTPUT)),
                "strict tools beside this structured output", "tools");
        // a prompt inside the reasoning needs the reasoning gate for the grammar
        refuses(BeamGrammar.Request.of(chat(TOOLS, ChatRequest.ToolChoice.none())).toolParsing(true)
                .prompt(PROMPT + "<|channel|>analysis<|message|>"), "tool_choice none", "tool_choice");
        TemplateParsing hf = TemplateParsing.of("hf", BEAM.template());
        try {
            BeamGrammar.derive(hf, request(chat(null, null), PROMPT));
            fail("hf derives no grammar");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("derives no grammar"));
        }
    }
}
