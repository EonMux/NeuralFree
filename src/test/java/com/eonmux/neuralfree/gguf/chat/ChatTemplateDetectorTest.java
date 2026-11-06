package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * {@link ChatTemplateDetector} answers llama.cpp's {@code llm_chat_detect_template}: every template of
 * llama.cpp's own test corpus (tests/test-chat-template.cpp, plus edge cases) detects what the C++
 * detector answers, a built-in name detects itself, and a text no style fits detects nothing — the
 * non-Jinja route refuses it.
 */
public class ChatTemplateDetectorTest {

    /** One corpus entry: a model's template text and the name llama.cpp's detector answers for it. */
    private static final class Case {
        final String name;
        final String template;
        final String detected;

        Case(String name, String template, String detected) {
            this.name = name;
            this.template = template;
            this.detected = detected;
        }
    }

    private static List<Case> cases() throws IOException {
        try (InputStream in = ChatTemplateDetectorTest.class.getResourceAsStream(
                "/chat/llamacpp-detect-templates.json")) {
            assertNotNull("corpus resource /chat/llamacpp-detect-templates.json", in);
            List<?> raw = (List<?>) Json.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8));
            List<Case> cases = new ArrayList<>();
            for (Object entry : raw) {
                Map<?, ?> map = (Map<?, ?>) entry;
                cases.add(new Case(String.valueOf(map.get("name")), String.valueOf(map.get("template")),
                        map.get("detected") == null ? null : String.valueOf(map.get("detected"))));
            }
            return cases;
        }
    }

    @Test
    public void everyCorpusTemplateDetectsWhatLlamaCppDetects() throws IOException {
        for (Case c : cases()) {
            assertEquals(c.name, c.detected, ChatTemplateDetector.detect(c.template));
        }
    }

    @Test
    public void aBuiltInNameDetectsItself() {
        for (String name : ChatTemplateSource.names()) {
            assertEquals(name, name, ChatTemplateDetector.detect(name));
        }
    }

    @Test
    public void theDetectionOnlyNameIsNeverAName() {
        assertNull("a text that only names dots1 detects nothing, as llama.cpp's does",
                ChatTemplateDetector.detect("dots1"));
    }

    @Test
    public void aTextNoStyleFitsDetectsNothing() {
        assertNull(ChatTemplateDetector.detect(null));
        assertNull(ChatTemplateDetector.detect(""));
        assertNull(ChatTemplateDetector.detect("[{{ m['role'] }}]{{ m['content'] }}"));
        assertNull(ChatTemplateDetector.detect("You are a helpful assistant."));
    }

    @Test
    public void llamaCppsBranchOrderHolds() {
        assertEquals("the ChatML check answers before the Mistral one",
                "chatml", ChatTemplateDetector.detect("<|im_start|>[INST]<|im_end|>"));
        assertEquals("the phi3 check answers before the <|assistant|>+<|user|> one",
                "phi3", ChatTemplateDetector.detect("<|assistant|><|end|><|user|>"));
        assertEquals("the falcon3 </s> marks falcon3 over glmedge",
                "falcon3", ChatTemplateDetector.detect("<|assistant|><|user|></s>"));
        assertEquals("strip marks llama2-sys-strip before the <<SYS>> check",
                "llama2-sys-strip", ChatTemplateDetector.detect("[INST] content.strip() [INST]"));
    }

    @Test
    public void theRefusalNamesTheTemplateAndBoundsItsSnippet() {
        String template = "[{{ m['role'] }}]{{ m['content'] }}" + "x".repeat(120);
        String refusal = ChatTemplateDetector.refusal(template);
        assertTrue(refusal, refusal.startsWith("no built-in chat template was detected in the chat template"));
        assertTrue(refusal, refusal.contains("80 of " + template.length() + " chars shown"));
        assertTrue(refusal, refusal.contains("the non-Jinja route renders only llama.cpp's known templates"));

        String multiline = ChatTemplateDetector.refusal("{% for m in messages %}\n{{ m.content }}\n{% endfor %}");
        String escaped = "{% for m in messages %}\\n{{ m.content }}\\n{% endfor %}";
        assertTrue("newlines escape, so the refusal is one line, and the count is the template's own: " + multiline,
                multiline.contains("(52 of 52 chars shown: '" + escaped));
        assertEquals("a null template shows nothing", "no built-in chat template was detected in the chat"
                        + " template (0 of 0 chars shown: ''); the non-Jinja route renders only llama.cpp's"
                        + " known templates",
                ChatTemplateDetector.refusal(null));
    }
}
