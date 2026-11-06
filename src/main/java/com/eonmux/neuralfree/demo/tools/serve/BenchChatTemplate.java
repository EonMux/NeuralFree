package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The model's chat template as the benchmark applies it on the client (vLLM's
 * {@code tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=False)}): a
 * conversation is rendered through NeuralFree's Jinja engine to the prompt text the completions
 * endpoint receives. The local tokenizer and the server's reported template both provide one.
 */
public interface BenchChatTemplate {

    /**
     * Renders {@code messages} with the generation prompt appended, binding {@code kwargs} as the
     * template's variables.
     *
     * @throws IllegalArgumentException when the template does not render the conversation
     */
    String render(List<Map<String, Object>> messages, Map<String, Object> kwargs);

    /** Renders one user message carrying {@code prompt}. */
    default String render(String prompt, Map<String, Object> kwargs) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "user");
        message.put("content", prompt);
        List<Map<String, Object>> messages = new ArrayList<>(1);
        messages.add(message);
        return render(messages, kwargs == null ? Collections.emptyMap() : kwargs);
    }

    /** A system message followed by a user message. */
    default String render(String system, String prompt, Map<String, Object> kwargs) {
        List<Map<String, Object>> messages = new ArrayList<>(2);
        Map<String, Object> systemMessage = new LinkedHashMap<>();
        systemMessage.put("role", "system");
        systemMessage.put("content", system);
        messages.add(systemMessage);
        Map<String, Object> userMessage = new LinkedHashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", prompt);
        messages.add(userMessage);
        return render(messages, kwargs == null ? Collections.emptyMap() : kwargs);
    }
}
