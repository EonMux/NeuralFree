package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The differential analysis reads a template's reasoning, content and tool-call structure, matching llama.cpp's
 * {@code chat-diff-analyzer.cpp} on synthetic unknown templates and on llama.cpp's own template files.
 */
public class AutoParserAnalysisTest {

    private static AutoParserAnalysis analyze(String template) {
        return AutoParserAnalyzer.analyze(template);
    }

    private static String resource(String name) throws IOException {
        try (InputStream in = AutoParserAnalysisTest.class.getResourceAsStream("/chat/templates/" + name + ".jinja")) {
            assertNotNull(name, in);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    public void tagWithJsonIsDetected() {
        AutoParserAnalysis a = analyze(AutoParserTestSupport.TAG_JSON_TEMPLATE);
        assertEquals(AutoParserAnalysis.ReasoningMode.TAG_BASED, a.reasoning().mode());
        assertEquals("<|think|>", a.reasoning().start());
        assertEquals("<|/think|>", a.reasoning().end());
        assertEquals(AutoParserAnalysis.ContentMode.PLAIN, a.content().mode());
        assertEquals(AutoParserAnalysis.ToolFormat.TAG_WITH_JSON, a.tools().mode());
        assertEquals("", a.tools().sectionStart());
        assertEquals("<|call|>", a.tools().perCallStart());
        assertEquals("<|/call|>", a.tools().perCallEnd());
        assertEquals("<function=", a.tools().functionNamePrefix());
        assertEquals(">", a.tools().functionNameSuffix());
        assertEquals("</function>", a.tools().functionClose());
        assertEquals("<|user|>", a.userStart());
        assertEquals("<|assistant|>", a.assistantStart());
    }

    @Test
    public void taggedArgumentsAreDetected() {
        AutoParserAnalysis a = analyze(AutoParserTestSupport.TAGGED_TEMPLATE);
        assertEquals(AutoParserAnalysis.ToolFormat.TAG_WITH_TAGGED, a.tools().mode());
        assertEquals("<|tool|>", a.tools().sectionStart());
        assertEquals("<|/tool|>", a.tools().sectionEnd());
        assertEquals("<|call|>", a.tools().perCallStart());
        assertEquals("<|/call|>", a.tools().perCallEnd());
        assertEquals("<function=", a.tools().functionNamePrefix());
        assertEquals(">", a.tools().functionNameSuffix());
        assertEquals("</function>", a.tools().functionClose());
        assertEquals("<parameter=", a.tools().argNamePrefix());
        assertEquals(">", a.tools().argNameSuffix());
        assertEquals("</parameter>", a.tools().argValueSuffix());
    }

    @Test
    public void jsonNativeIsDetected() {
        AutoParserAnalysis a = analyze(AutoParserTestSupport.JSON_NATIVE_TEMPLATE);
        assertEquals(AutoParserAnalysis.ToolFormat.JSON_NATIVE, a.tools().mode());
        assertEquals("<|tool_calls|>", a.tools().sectionStart());
        assertEquals("<|/tool_calls|>", a.tools().sectionEnd());
        assertTrue(a.tools().toolsArrayWrapped());
        assertEquals("tool_name", a.tools().nameField());
        assertEquals("parameters", a.tools().argsField());
        assertEquals("tool_name", a.tools().effectiveNameField());
        assertEquals("parameters", a.tools().effectiveArgsField());
        assertFalse(a.tools().funNameIsKey());
    }

    @Test
    public void markersAreCollectedAsPreservedTokens() {
        AutoParserAnalysis a = analyze(AutoParserTestSupport.TAGGED_TEMPLATE);
        List<String> expected = Arrays.asList("<|think|>", "<|/think|>", "<|tool|>", "<|/tool|>", "<|call|>",
                "<|/call|>", "<function=", ">", "</function>", "<parameter=", "</parameter>");
        assertEquals(expected, a.preservedTokens());
    }

    @Test
    public void lagunaTemplateMatchesLlamaCppAnalysis() throws IOException {
        // llama.cpp tests/test-chat-auto-parser.cpp test_laguna_s_analysis: TAG_BASED <think>...</think>
        AutoParserAnalysis a = analyze(resource("poolside-Laguna-S-2.1"));
        assertEquals(AutoParserAnalysis.ReasoningMode.TAG_BASED, a.reasoning().mode());
        assertEquals("<think>", a.reasoning().start().trim());
        assertEquals("</think>", a.reasoning().end().trim());
        assertEquals(AutoParserAnalysis.ToolFormat.TAG_WITH_TAGGED, a.tools().mode());
        assertEquals("</arg_value>", a.tools().argValueSuffix().trim());
        assertEquals("<user>", a.userStart());
        assertEquals("<assistant>", a.assistantStart());
    }

    @Test
    public void glmTemplateMatchesItsReasoningAndRoleMarkers() throws IOException {
        AutoParserAnalysis a = analyze(resource("GLM-4.7-Flash"));
        assertEquals(AutoParserAnalysis.ReasoningMode.TAG_BASED, a.reasoning().mode());
        assertEquals("<think>", a.reasoning().start().trim());
        assertEquals("</think>", a.reasoning().end().trim());
        assertEquals(AutoParserAnalysis.ToolFormat.TAG_WITH_TAGGED, a.tools().mode());
        assertEquals("<|user|>", a.userStart());
        assertEquals("<|assistant|>", a.assistantStart());
    }

    @Test
    public void qwenTemplateIsJsonNative() throws IOException {
        AutoParserAnalysis a = analyze(resource("Qwen-Qwen2.5-7B-Instruct"));
        assertEquals(AutoParserAnalysis.ToolFormat.JSON_NATIVE, a.tools().mode());
        assertEquals("name", a.tools().effectiveNameField());
        assertEquals("arguments", a.tools().effectiveArgsField());
        assertEquals("<|im_start|>assistant", a.assistantStart());
    }

    @Test
    public void templateWithoutMarkersHasNoStructure() {
        AutoParserAnalysis a = analyze("{% for m in messages %}{{ m['content'] }}{% endfor %}");
        assertEquals(AutoParserAnalysis.ReasoningMode.NONE, a.reasoning().mode());
        assertEquals(AutoParserAnalysis.ContentMode.PLAIN, a.content().mode());
        assertEquals(AutoParserAnalysis.ToolFormat.NONE, a.tools().mode());
        assertFalse(a.hasTools());
        assertTrue(a.preservedTokens().isEmpty());
    }

    @Test
    public void unparsableTemplateIsRefusedByName() {
        assertNull(AutoParser.analyze("{% if %}"));
        assertNull(AutoParserFormat.forTemplate("{% if %}"));
    }

    @Test
    public void analysisIsCachedPerSource() {
        assertTrue(AutoParser.analyze(AutoParserTestSupport.TAGGED_TEMPLATE)
                == AutoParser.analyze(AutoParserTestSupport.TAGGED_TEMPLATE));
    }

    @Test
    public void theAnalysisYieldsTheReasoningFilter() {
        ReasoningFilter filter = AutoParserFormat.reasoningFilter(
                analyze(AutoParserTestSupport.TAGGED_TEMPLATE));
        assertNotNull(filter);
        assertEquals(ReasoningFilter.Mode.THINK_TAGS, filter.mode());
        assertEquals(Arrays.asList("<|think|>"), filter.startTags());
        assertEquals(Arrays.asList("<|/think|>"), filter.endTags());
        assertEquals(Arrays.asList("<|tool|>"), filter.interruptTags());
        assertNull(AutoParserFormat.reasoningFilter(
                analyze("{% for m in messages %}{{ m['content'] }}{% endfor %}")));
    }

    @Test
    public void detectUsesTheAutoparserForAnUnknownSyntax() {
        assertEquals(AutoParserFormat.NAME,
                com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry.detect(
                        AutoParserTestSupport.TAGGED_TEMPLATE).name());
        assertEquals("hermes", com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry.detect(
                "<tool_call>{{\"name\": x}}</tool_call>").name());
    }
}
