package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The per-choice reasoning presentation: by default reasoning pieces stream as both vLLM's
 * {@code reasoning} and llama.cpp's {@code reasoning_content}, and a request with
 * {@code include_reasoning} false drops them while the token's logprobs capture still reaches its delta.
 */
public class ChatChoiceReasoningTest {

    private static GenerationBatch.Text reasoning(String text) {
        return new GenerationBatch.Text(0, ReasoningFilter.Channel.REASONING, text, null, 1, 1);
    }

    @Test
    void reasoningStreamsAsReasoningContentAndReasoningByDefault() {
        ChatChoice choice = new ChatChoice(ToolRouting.none().newOutput(), true);
        choice.accept(reasoning("thinking"));
        List<ChatChoice.Delta> deltas = choice.drain();
        assertEquals(1, deltas.size());
        assertEquals("thinking", deltas.get(0).delta().get("reasoning_content"));
        assertEquals("thinking", deltas.get(0).delta().get("reasoning"));
        assertEquals("thinking", choice.reasoning());
    }

    @Test
    void includeReasoningFalseDropsTheReasoning() {
        ChatChoice choice = new ChatChoice(ToolRouting.none().newOutput(), false);
        choice.accept(reasoning("thinking"));
        assertTrue(choice.drain().isEmpty(), "no reasoning delta");
        assertEquals("", choice.reasoning(), "no reasoning content");
        choice.finish();
        assertEquals("", choice.reasoning());
    }

    @Test
    void includeReasoningFalseKeepsTheLogprobsCapture() {
        ChatChoice choice = new ChatChoice(ToolRouting.none().newOutput(), false);
        choice.accept(new GenerationBatch.Text(0, ReasoningFilter.Channel.REASONING, "thinking",
                new TokenScore[] {new TokenScore(9, -0.5f)}, 1, 1));
        List<ChatChoice.Delta> deltas = choice.drain();
        assertEquals(1, deltas.size(), "the capture rides an empty delta");
        assertEquals(1, choice.logprobs().size());
        assertTrue(choice.reasoning().isEmpty());
    }
}
