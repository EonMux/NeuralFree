package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The reasoning-budget request fields across the APIs: llama.cpp's {@code reasoning_budget_tokens} family and vLLM's
 * {@code thinking_token_budget} on the OpenAI routes, Anthropic's {@code thinking.budget_tokens} on the Messages
 * route — onto the engine request, with the malformed values refusing by name.
 */
class ReasoningBudgetParamsTest {

    private static final int VOCAB = 16;

    private static GenerationRequest sampling(String body) {
        return OpenAiRequests.sampling(view(body), -1, VOCAB, false);
    }

    @Test
    void llamaCppBudgetFieldsReachTheRequest() {
        GenerationRequest r = sampling("""
                {"reasoning_budget_tokens": 128, "reasoning_budget_message": "time is up",
                 "reasoning_budget_start_tag": "<think>", "reasoning_budget_end_tags": ["</think>", "<tool_call>"]}""");
        assertEquals(128, r.reasoningBudget());
        assertEquals("time is up", r.reasoningBudgetMessage());
        assertEquals(List.of("<think>"), r.reasoningBudgetStartTags());
        assertEquals(List.of("</think>", "<tool_call>"), r.reasoningBudgetEndTags());
        assertFalse(r.reasoningControl());
    }

    @Test
    void theSingularEndTagAliasAndTheControlSwitchMap() {
        GenerationRequest r = sampling("""
                {"reasoning_control": true, "reasoning_budget_start_tag": "[THINK]",
                 "reasoning_budget_end_tag": "[/THINK]"}""");
        assertTrue(r.reasoningControl());
        assertEquals(GenerationRequest.REASONING_BUDGET_DISABLED, r.reasoningBudget());
        assertEquals(List.of("[THINK]"), r.reasoningBudgetStartTags());
        assertEquals(List.of("[/THINK]"), r.reasoningBudgetEndTags());
    }

    @Test
    void vllmThinkingTokenBudgetMapsOntoTheBudget() {
        assertEquals(256, sampling("{\"thinking_token_budget\": 256}").reasoningBudget());
        assertEquals(GenerationRequest.REASONING_BUDGET_DISABLED,
                sampling("{\"thinking_token_budget\": -1}").reasoningBudget());
    }

    @Test
    void malformedBudgetsRefuseNamingTheField() {
        for (String body : new String[]{
                "{\"thinking_token_budget\": 1.5}",
                "{\"thinking_token_budget\": true}",
                "{\"thinking_token_budget\": -5}",
                "{\"reasoning_budget_tokens\": 0.25}",
                "{\"reasoning_budget_tokens\": \"many\"}",
                "{\"reasoning_budget_message\": 3}",
                "{\"reasoning_budget_start_tag\": \"\"}",
                "{\"reasoning_budget_end_tags\": []}",
                "{\"reasoning_budget_end_tags\": [\"\"]}"}) {
            assertEquals(400, statusOf(() -> sampling(body)), body);
        }
        assertTrue(errorOf(() -> sampling("{\"thinking_token_budget\": -5}")).contains("thinking_token_budget"));
        assertTrue(errorOf(() -> sampling("{\"reasoning_budget_tokens\": \"many\"}"))
                .contains("reasoning_budget_tokens"));
    }

    @Test
    void anthropicBudgetTokensBecomeTheEngineBudget() {
        assertEquals(2048, AnthropicRequests.request(view("""
                {"max_tokens": 64, "thinking": {"type": "enabled", "budget_tokens": 2048}}""")).reasoningBudget());
        assertEquals(GenerationRequest.REASONING_BUDGET_DISABLED, AnthropicRequests.request(view("""
                {"max_tokens": 64, "thinking": {"type": "enabled"}}""")).reasoningBudget());
        assertEquals(GenerationRequest.REASONING_BUDGET_DISABLED, AnthropicRequests.request(view("""
                {"max_tokens": 64, "thinking": {"type": "disabled", "budget_tokens": 2048}}""")).reasoningBudget());
        assertEquals(GenerationRequest.REASONING_BUDGET_DISABLED,
                AnthropicRequests.request(view("{\"max_tokens\": 64}")).reasoningBudget());
    }

    @Test
    void malformedAnthropicBudgetsRefuseNamingTheField() {
        assertEquals(400, statusOf(() -> AnthropicRequests.request(view(
                "{\"max_tokens\": 64, \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": -3}}"))));
        assertEquals(400, statusOf(() -> AnthropicRequests.request(view(
                "{\"max_tokens\": 64, \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": \"a lot\"}}"))));
        assertTrue(errorOf(() -> AnthropicRequests.request(view(
                "{\"max_tokens\": 64, \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": -3}}")))
                .contains("thinking.budget_tokens"));
    }
}
