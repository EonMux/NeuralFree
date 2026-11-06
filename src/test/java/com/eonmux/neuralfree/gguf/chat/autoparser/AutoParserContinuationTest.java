package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The continued final message on an autoparsed template: llama.cpp's autoparser writes the generation prompt up
 * to the derived reasoning start, then the reasoning content and, unless only the reasoning continues, the
 * reasoning end and the content.
 */
public class AutoParserContinuationTest {

    private static final String TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('reasoning_content') -%}<|think|>{{ message['reasoning_content'] }}"
            + "<|/think|>{%- endif -%}<|end|>\n"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|><|think|>{%- endif -%}";

    private static ChatFormatter formatter() {
        return new ChatFormatter(TEMPLATE, "<bos>", "<eos>", false, false, "llama");
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    private static Map<String, Object> assistant(String content, String reasoning) {
        Map<String, Object> m = message("assistant", content);
        m.put("reasoning_content", reasoning);
        return m;
    }

    private static ChatRequest request(Map<String, Object> assistant) {
        return ChatRequest.builder()
                .addMessage(message("user", "hi"))
                .addMessage(assistant)
                .continueFinalMessage(true)
                .build();
    }

    @Test
    public void aReasoningOnlyContinuationLeavesTheBlockOpen() {
        AutoParserAnalysis analysis = AutoParser.analyze(TEMPLATE);
        String prompt = formatter().format(request(assistant("", "abc")), false, analysis);
        assertEquals("<|user|>hi<|end|><|assistant|><|think|>abc", prompt);
    }

    @Test
    public void aContentContinuationClosesTheReasoningAndAddsTheContent() {
        AutoParserAnalysis analysis = AutoParser.analyze(TEMPLATE);
        String prompt = formatter().format(request(assistant("OK", "abc")), false, analysis);
        assertEquals("<|user|>hi<|end|><|assistant|><|think|>abc<|/think|>OK", prompt);
    }

    @Test
    public void anEmptyContinuedMessageRendersThroughTheTemplate() {
        AutoParserAnalysis analysis = AutoParser.analyze(TEMPLATE);
        String prompt = formatter().format(request(assistant("", "")), false, analysis);
        assertEquals("<|user|>hi<|end|><|assistant|><|end|>", prompt);
    }

    @Test
    public void withoutTheAnalysisTheTemplateRendersTheContinuedMessage() {
        String prompt = formatter().format(request(assistant("", "abc")), false, null);
        assertTrue(prompt, prompt.contains("<|think|>abc<|/think|>"));
        assertFalse("the generic path renders the template's end of message",
                prompt.endsWith("abc"));
    }
}
