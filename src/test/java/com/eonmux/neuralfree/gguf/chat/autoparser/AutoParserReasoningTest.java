package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The reasoning a template renders is derived by the same differential analysis that derives its tool calls: an
 * unknown template whose markers {@link ReasoningFilter#forChatTemplate(String)} does not recognize still gets a
 * reasoning filter when the {@code auto} parser runs ({@link AutoParserFormat#reasoningFilter}).
 */
public class AutoParserReasoningTest {

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
        for (String p : new String[]{"a", "b", "c"}) {
            add(tokens, scores, types, p, 1);
        }
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("tokenizer.ggml.model", "llama");
        metadata.put("tokenizer.ggml.tokens", tokens);
        metadata.put("tokenizer.ggml.scores", scores);
        metadata.put("tokenizer.ggml.token_type", types);
        metadata.put("tokenizer.ggml.bos_token_id", 1);
        metadata.put("tokenizer.ggml.eos_token_id", 2);
        metadata.put("tokenizer.ggml.unknown_token_id", 0);
        metadata.put("tokenizer.chat_template", template);
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(metadata));
    }

    private static void add(List<Object> tokens, List<Object> scores, List<Object> types, String token, int type) {
        tokens.add(token);
        scores.add(-1.0f);
        types.add(type);
    }

    @Test
    public void autoReasoningComesFromTheAnalysis() {
        ReasoningFilter filter = ReasoningParsers.forModel("auto",
                tokenizer(AutoParserTestSupport.TAGGED_TEMPLATE));
        assertEquals(ReasoningFilter.Mode.THINK_TAGS, filter.mode());
        assertEquals(Collections.singletonList("<|think|>"), filter.startTags());
        assertEquals(Collections.singletonList("<|/think|>"), filter.endTags());
        assertEquals(Collections.singletonList("<|tool|>"), filter.interruptTags());
    }

    @Test
    public void theDerivedFilterSplitsReasoningFromTheAnswer() {
        ReasoningFilter filter = ReasoningParsers.forModel("auto",
                tokenizer(AutoParserTestSupport.TAGGED_TEMPLATE));
        StringBuilder reasoning = new StringBuilder();
        StringBuilder content = new StringBuilder();
        ReasoningFilter.State state = filter.newState((channel, tokenId, text) ->
                (channel == ReasoningFilter.Channel.REASONING ? reasoning : content).append(text));
        String output = "<|think|>weigh the options<|/think|>The answer is 4.";
        for (int i = 0; i < output.length(); i++) {
            state.onToken(i, output.substring(i, i + 1));
        }
        state.finish();
        assertEquals("weigh the options", reasoning.toString().trim());
        assertTrue(content.toString(), content.toString().contains("The answer is 4."));
    }

    @Test
    public void aKnownTemplateKeepsItsHandWrittenFilter() {
        String template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n<think>{{ m['content'] }}"
                + "</think><|im_end|>{% endfor %}";
        ReasoningFilter filter = ReasoningParsers.forModel("auto", tokenizer(template));
        assertEquals(ReasoningFilter.Mode.THINK_TAGS, filter.mode());
        assertEquals(Collections.singletonList("<think>"), filter.startTags());
    }

    @Test
    public void aTemplateWithoutReasoningGetsNoFilter() {
        String template = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}";
        ReasoningFilter filter = ReasoningParsers.forModel("auto", tokenizer(template));
        assertEquals(ReasoningFilter.Mode.NONE, filter.mode());
        assertNotNull(filter);
    }
}
