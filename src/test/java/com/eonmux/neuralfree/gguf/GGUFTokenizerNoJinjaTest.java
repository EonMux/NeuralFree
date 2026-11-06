package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.chat.ChatTemplateDetector;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * llama.cpp's non-Jinja route on a tokenizer ({@code --no-jinja}): the model's template text is
 * replaced by the built-in template {@link ChatTemplateDetector} guesses from it, the detection-only
 * {@code dots1} included, and a template no style fits is refused at load the way llama.cpp's
 * {@code common_chat_verify_template} refuses it without {@code --jinja} (common/arg.cpp:940-946).
 * With the switch at its default the model's template is kept, and a template override goes through
 * detection as its raw text.
 */
public class GGUFTokenizerNoJinjaTest {

    /** A text the detector answers chatml for, though it is not the built-in ChatML source. */
    private static final String CHATML_LIKE =
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                    + "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    private static final String UNDETECTABLE = "[{{ m['role'] }}]{{ m['content'] }}";
    private static final String DOTS1_LIKE = "{% for m in messages %}{{ m['content'] }}<|endofuserprompt|>{% endfor %}";

    private String savedJinja;
    private String savedInline;

    @Before
    public void saveProperties() {
        savedJinja = System.getProperty(ChatTemplateSource.JINJA_PROPERTY);
        savedInline = System.getProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        System.clearProperty(ChatTemplateSource.JINJA_PROPERTY);
        System.clearProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
    }

    @After
    public void restoreProperties() {
        if (savedJinja == null) {
            System.clearProperty(ChatTemplateSource.JINJA_PROPERTY);
        } else {
            System.setProperty(ChatTemplateSource.JINJA_PROPERTY, savedJinja);
        }
        if (savedInline == null) {
            System.clearProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        } else {
            System.setProperty(ChatTemplateSource.OVERRIDE_PROPERTY, savedInline);
        }
    }

    /** A SentencePiece vocabulary whose chat template is {@code template}. */
    private static GGUFTokenizer tokenizer(String template) {
        List<Object> tokens = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, scores, types, "<unk>", 2);
        add(tokens, scores, types, "<s>", 3);
        add(tokens, scores, types, "</s>", 3);
        for (int b = 0; b < 256; b++) {
            add(tokens, scores, types, String.format("<0x%02X>", b), 6);
        }
        for (String p : new String[]{"a", "b", "u", "s", "e", "r"}) {
            add(tokens, scores, types, p, 1);
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

    @Test
    public void theDefaultKeepsTheModelTemplate() {
        assertEquals(ChatTemplateSource.resolve(CHATML_LIKE), tokenizer(CHATML_LIKE).chatTemplate());
    }

    @Test
    public void anExplicitJinjaTrueKeepsTheModelTemplateToo() {
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "true");
        assertEquals(ChatTemplateSource.resolve(CHATML_LIKE), tokenizer(CHATML_LIKE).chatTemplate());
    }

    @Test
    public void theNonJinjaRouteReplacesTheTemplateWithTheDetectedOne() {
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "false");
        GGUFTokenizer tok = tokenizer(CHATML_LIKE);
        assertEquals("the text detected chatml, so the built-in renders the prompts",
                ChatTemplateSource.CHATML, tok.chatTemplate());
        assertEquals("<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nab<|im_end|>\n"
                + "<|im_start|>assistant\n", tok.formatChat("sys", "ab", false));
    }

    @Test
    public void theDetectionOnlyTemplateIsReachable() {
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "false");
        GGUFTokenizer tok = tokenizer(DOTS1_LIKE);
        assertEquals(ChatTemplateSource.detectedSource("dots1"), tok.chatTemplate());
        assertEquals("<|userprompt|>ab<|endofuserprompt|><|response|>", tok.formatChat(null, "ab", false));
    }

    @Test
    public void anUndetectableModelTemplateIsRefused() {
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "false");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> tokenizer(UNDETECTABLE));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--no-jinja ("));
        assertTrue(e.getMessage(), e.getMessage().contains(ChatTemplateSource.JINJA_PROPERTY));
        assertTrue(e.getMessage(), e.getMessage()
                .contains(ChatTemplateDetector.refusal(UNDETECTABLE)));
    }

    @Test
    public void theOverrideGoesThroughDetectionAsItsRawText() {
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "false");
        System.setProperty(ChatTemplateSource.OVERRIDE_PROPERTY, "llama3");
        GGUFTokenizer tok = tokenizer(UNDETECTABLE);
        assertEquals("the named override replaces the model's template before detection",
                ChatTemplateSource.resolve("llama3"), tok.chatTemplate());
    }
}
