package com.eonmux.neuralfree.gguf.chat;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

/**
 * The gpt-oss "harmony" prompt for a conversation, written without the model's template: the system
 * header, the developer instructions, every message as {@code <|start|>role<|message|>content<|end|>}
 * (assistant messages on the {@code final} channel), then the open assistant turn. It mirrors the
 * structure of the gpt-oss chat template (llama.cpp models/templates/openai-gpt-oss-120b.jinja:233-330) and
 * reproduces its fixed system-header strings; the template is OpenAI's, under the Apache License 2.0 (see
 * THIRD-PARTY-NOTICES.md).
 */
final class HarmonyFallback {

    private static final String DEFAULT_INSTRUCTIONS = "You are a helpful assistant.";

    private HarmonyFallback() {}

    /**
     * @param messages messages from {@link ChatMessages#copyOf}; a leading system or developer message
     *                 becomes the developer instructions
     * @param today    the date written into the system header
     */
    static String format(List<Map<String, Object>> messages, LocalDate today) {
        int first = 0;
        String instructions = DEFAULT_INSTRUCTIONS;
        if (!messages.isEmpty()) {
            Object role = messages.get(0).get(ChatMessages.ROLE);
            String content = String.valueOf(messages.get(0).get(ChatMessages.CONTENT));
            if ("system".equals(role) || "developer".equals(role)) {
                first = 1;
                if (!content.isEmpty()) instructions = content;
            }
        }
        StringBuilder sb = new StringBuilder(512);
        sb.append("<|start|>system<|message|>")
          .append("You are ChatGPT, a large language model trained by OpenAI.\n")
          .append("Knowledge cutoff: 2024-06\n")
          .append("Current date: ").append(today).append("\n\n")
          .append("Reasoning: medium\n\n")
          .append("# Valid channels: analysis, commentary, final. Channel must be included for every message.")
          .append("<|end|>");
        sb.append("<|start|>developer<|message|>")
          .append("# Instructions\n\n")
          .append(instructions)
          .append("<|end|>");
        for (int i = first; i < messages.size(); i++) {
            Map<String, Object> m = messages.get(i);
            String role = (String) m.get(ChatMessages.ROLE);
            sb.append("<|start|>").append(role);
            if ("assistant".equals(role)) sb.append("<|channel|>final");
            sb.append("<|message|>").append(String.valueOf(m.get(ChatMessages.CONTENT))).append("<|end|>");
        }
        sb.append("<|start|>assistant");
        return sb.toString();
    }
}
