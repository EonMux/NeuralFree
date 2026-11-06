package com.eonmux.neuralfree.gguf.chat.autoparser;

import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The per-template patches llama.cpp applies after the differential probes: their triggers are the reference's
 * source fragments, and each one rewrites the structure, preserved markers and stops the reference writes.
 */
public class AutoParserWorkaroundTest {

    private static final String BODY =
            "{% for m in messages %}{{ m['content'] }}{% endfor %}";

    private static AutoParserAnalysis analyze(String marker) {
        return AutoParser.analyze("{# " + marker + " #}" + BODY);
    }

    private static String resource(String name) throws IOException {
        try (InputStream in = AutoParserWorkaroundTest.class.getResourceAsStream("/chat/templates/" + name
                + ".jinja")) {
            assertNotNull(name, in);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    public void oldQwenTemplatesGetThinkMarkers() {
        AutoParserAnalysis a = analyze("content.split('</think>')");
        assertEquals(AutoParserAnalysis.ReasoningMode.TAG_BASED, a.reasoning().mode());
        assertEquals("<think>", a.reasoning().start());
        assertEquals("</think>", a.reasoning().end());
        assertTrue(a.preservedTokens().contains("<think>"));
        assertTrue(a.preservedTokens().contains("</think>"));
    }

    @Test
    public void granite33GetsReasoningAndResponseMarkers() {
        AutoParserAnalysis a = analyze("Write your thoughts between <think></think> and write your response between "
                + "<response></response>");
        assertEquals(AutoParserAnalysis.ReasoningMode.TAG_BASED, a.reasoning().mode());
        assertEquals("<think>", a.reasoning().start());
        assertEquals("</think>", a.reasoning().end());
        assertEquals(AutoParserAnalysis.ContentMode.WRAPPED_WITH_REASONING, a.content().mode());
        assertEquals("<response>", a.content().start());
        assertEquals("</response>", a.content().end());
        assertTrue(a.preservedTokens().contains("<response>"));
    }

    @Test
    public void cohereCommandRWrapsItsContent() {
        AutoParserAnalysis a = analyze("<|CHATBOT_TOKEN|> and <|END_OF_TURN_TOKEN|>");
        assertEquals(AutoParserAnalysis.ContentMode.ALWAYS_WRAPPED, a.content().mode());
        assertEquals("<|CHATBOT_TOKEN|>", a.content().start());
        assertEquals("<|END_OF_TURN_TOKEN|>", a.content().end());
        assertEquals("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", a.userStart());
        assertTrue(a.preservedTokens().contains("<|CHATBOT_TOKEN|>"));
        assertTrue(a.preservedTokens().contains("<|END_OF_TURN_TOKEN|>"));
    }

    @Test
    public void functionaryGetsTheBareFunctionCallSyntax() {
        AutoParserAnalysis a = analyze("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", "
                + "\"code_interpreter\") | list | length > 0");
        assertEquals(AutoParserAnalysis.ContentMode.PLAIN, a.content().mode());
        assertEquals("", a.content().end());
        assertEquals("", a.tools().sectionStart());
        assertEquals("", a.tools().sectionEnd());
        assertEquals("<function=", a.tools().perCallStart());
        assertEquals("</function>", a.tools().perCallEnd());
        assertEquals("", a.tools().functionNamePrefix());
        assertEquals("", a.tools().functionClose());
        assertEquals(java.util.Arrays.asList("<|eot_id|>", "<|eom_id|>", "<function=", ">", "</function>"),
                a.preservedTokens());
    }

    @Test
    public void deepseekR1DistillGetsItsCallMarkers() {
        AutoParserAnalysis a = analyze("{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + "
                + "'<｜tool▁sep｜>'");
        assertEquals("<｜tool▁calls▁begin｜>", a.tools().sectionStart());
        assertEquals("<｜tool▁calls▁end｜>", a.tools().sectionEnd());
        assertEquals("<｜tool▁call▁begin｜>function", a.tools().perCallStart());
        assertEquals("<｜tool▁sep｜>", a.tools().functionNamePrefix());
        assertEquals("<｜tool▁call▁end｜>", a.tools().perCallEnd());
        assertEquals("```", a.tools().functionClose());
    }

    @Test
    public void nemotronGetsItsJsonNativeCallFormat() {
        AutoParserAnalysis a = analyze("<SPECIAL_10><SPECIAL_11><SPECIAL_12><TOOL_RESPONSE>");
        assertEquals(AutoParserAnalysis.ToolFormat.JSON_NATIVE, a.tools().mode());
        assertEquals("<TOOLCALL>", a.tools().perCallStart());
        assertEquals("</TOOLCALL>", a.tools().perCallEnd());
        assertTrue(a.tools().toolsArrayWrapped());
        assertEquals(AutoParserAnalysis.ContentMode.PLAIN, a.content().mode());
        assertEquals("<think>\n", a.reasoning().start());
        assertEquals("</think>", a.reasoning().end());
        assertEquals("<SPECIAL_11>Assistant", a.assistantStart());
        assertEquals("<SPECIAL_11>User", a.userStart());
        assertEquals(java.util.Arrays.asList("<SPECIAL_11>", "</think>", "<TOOLCALL>", "</TOOLCALL>"),
                a.preservedTokens());
    }

    @Test
    public void fireworksGetsItsHeaderMarkers() {
        AutoParserAnalysis a = analyze("{%- set system_prompt = '<|start_header_id|>' + 'system' + "
                + "'<|end_header_id|>\\n\\n' + message['content'] | trim + '\\n' + system_prompt_suffix + "
                + "'<|eot_id|>' -%}");
        assertEquals("<|start_header_id|>assistant<|end_header_id|>", a.assistantStart());
        assertEquals("<|start_header_id|>user<|end_header_id|>", a.userStart());
    }

    @Test
    public void solarOpenGetsItsAssistantHeader() {
        AutoParserAnalysis a = analyze("<|begin|>assistant<|think|><|end|>");
        assertEquals("<|begin|>assistant", a.assistantStart());
    }

    @Test
    public void aprielGetsItsRoleMarkers() {
        AutoParserAnalysis a = analyze("if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text");
        assertEquals("<|begin_user|>", a.userStart());
        assertEquals("<|begin_assistant|>", a.assistantStart());
    }

    @Test
    public void aJsonNameParametersInstructionTriggersTheOpenaiWrapper() {
        AutoParserAnalysis a = analyze("Respond in the format {\"name\": function name, \"parameters\": ...}. "
                + "Do not use variables.");
        assertTrue(a.tools().openaiWrapperTrigger());
    }

    @Test
    public void lagunaTrimsItsMarkersToleratesWhitespaceAndStopsOnAssistant() throws IOException {
        AutoParserAnalysis a = AutoParser.analyze(resource("poolside-Laguna-S-2.1"));
        assertEquals("<think>", a.reasoning().start());
        assertEquals("</think>", a.reasoning().end());
        assertTrue(a.tools().tolerateIntertagWhitespace());
        assertEquals("</arg_value>", a.tools().argValueSuffix());
        assertTrue(a.additionalStops().contains("</assistant>"));
    }

    @Test
    public void bailingTrimsItsValueSuffixAndToleratesWhitespace() {
        AutoParserAnalysis a = analyze("Bailing V3 chat template");
        assertTrue(a.tools().tolerateIntertagWhitespace());
        assertEquals("", a.tools().argValueSuffix());
        assertFalse(a.tools().openaiWrapperTrigger());
    }
}
