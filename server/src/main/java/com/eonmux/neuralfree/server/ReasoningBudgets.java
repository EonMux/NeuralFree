package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The reasoning-budget fields of a request body (llama.cpp's {@code reasoning_budget_tokens} family and vLLM's
 * {@code thinking_token_budget}) onto the engine request: the budget, the injected message, the mid-stream control
 * switch and the explicit tag overrides. Malformed values refuse with 400 naming the field.
 */
final class ReasoningBudgets {

    private ReasoningBudgets() {
    }

    /** Applies the body's reasoning-budget fields to {@code builder}; absent fields keep the defaults. */
    static void apply(JsonView body, GenerationRequest.Builder builder) {
        if (body.raw("reasoning_budget_tokens") != null) {
            builder.reasoningBudget(budget(body.raw("reasoning_budget_tokens"), "reasoning_budget_tokens"));
        }
        if (body.raw("thinking_token_budget") != null) {
            builder.reasoningBudget(budget(body.raw("thinking_token_budget"), "thinking_token_budget"));
        }
        if (body.raw("reasoning_budget_message") != null) {
            Object message = body.raw("reasoning_budget_message");
            if (!(message instanceof String)) {
                throw ServerJson.badRequest("reasoning_budget_message must be a string");
            }
            builder.reasoningBudgetMessage((String) message);
        }
        if (body.optBool("reasoning_control", false)) {
            builder.reasoningControl(true);
        }
        if (body.raw("reasoning_budget_start_tag") != null) {
            builder.reasoningBudgetStartTags(List.of(string(body.raw("reasoning_budget_start_tag"),
                    "reasoning_budget_start_tag")));
        }
        if (body.raw("reasoning_budget_end_tags") != null) {
            builder.reasoningBudgetEndTags(strings(body.raw("reasoning_budget_end_tags"), "reasoning_budget_end_tags"));
        } else if (body.raw("reasoning_budget_end_tag") != null) {
            builder.reasoningBudgetEndTags(List.of(string(body.raw("reasoning_budget_end_tag"),
                    "reasoning_budget_end_tag")));
        }
    }

    /** Anthropic's {@code thinking.budget_tokens} of an enabled-thinking body, or the request default. */
    static int anthropicBudget(JsonView body) {
        if (body.raw("thinking") instanceof Map<?, ?> thinking && "enabled".equals(thinking.get("type"))
                && thinking.get("budget_tokens") != null) {
            return budget(thinking.get("budget_tokens"), "thinking.budget_tokens");
        }
        return GenerationRequest.REASONING_BUDGET_DISABLED;
    }

    /** A whole-number budget: -1 disables, a negative value otherwise refuses (vLLM's validator). */
    private static int budget(Object raw, String key) {
        if (!(raw instanceof Integer || raw instanceof Long)) {
            throw ServerJson.badRequest(key + " must be a non-negative integer or -1 for unlimited, got " + raw);
        }
        long value = ((Number) raw).longValue();
        if (value < GenerationRequest.REASONING_BUDGET_DISABLED || value > Integer.MAX_VALUE) {
            throw ServerJson.badRequest(key + " must be a non-negative integer or -1 for unlimited, got " + value);
        }
        return (int) value;
    }

    private static String string(Object raw, String key) {
        if (!(raw instanceof String) || ((String) raw).isEmpty()) {
            throw ServerJson.badRequest(key + " must be a non-empty string");
        }
        return (String) raw;
    }

    private static List<String> strings(Object raw, String key) {
        if (!(raw instanceof List<?> list) || list.isEmpty()) {
            throw ServerJson.badRequest(key + " must be a non-empty array of strings");
        }
        List<String> out = new ArrayList<>(list.size());
        for (Object o : list) {
            out.add(string(o, key));
        }
        return out;
    }
}
