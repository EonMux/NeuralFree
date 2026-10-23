package com.eonmux.neuralfree.gguf.chat;

/**
 * Resolves the Jinja source a model's prompts are rendered with, as llama.cpp does when it initializes
 * a model's chat templates (common/chat.cpp common_chat_templates_init, :729-780).
 *
 * <p>The default template is {@code tokenizer.chat_template}. When it is absent, empty or the literal
 * {@code chatml}, the {@code tokenizer.chat_template.tool_use} template replaces it, or the built-in
 * {@link #CHATML} template when there is none (common/chat.cpp:752-758). The resolved source then
 * receives llama.cpp's two source patches, which disable checks that would otherwise abort rendering
 * of valid conversations (common/chat.cpp:760-780).</p>
 */
public final class ChatTemplateSource {

    /** GGUF key of the default chat template. */
    public static final String KEY_DEFAULT = "tokenizer.chat_template";
    /** GGUF key of the tool-use chat template ({@code llama_model_chat_template(model, "tool_use")}). */
    public static final String KEY_TOOL_USE = "tokenizer.chat_template.tool_use";

    /** llama.cpp's built-in ChatML template (common/chat.cpp:687-693, CHATML_TEMPLATE_SRC). */
    public static final String CHATML =
            "{%- for message in messages -%}\n"
            + "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n"
            + "{%- endfor -%}\n"
            + "{%- if add_generation_prompt -%}\n"
            + "  {{- '<|im_start|>assistant\n' -}}\n"
            + "{%- endif -%}";

    /** Name that selects the built-in ChatML template in place of a template source. */
    private static final String CHATML_NAME = "chatml";

    private static final String HARMONY_CHANNEL = "<|channel|>";
    private static final String HARMONY_CHECK_HINT = "in message.content or";
    private static final String HARMONY_CHECK =
            "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
            + "\"<|channel|>final<|message|>\" in message.content %}";

    private static final String MISTRAL_TOOL_CALLS = "[TOOL_CALLS]";
    private static final String MISTRAL_CHECK_HINT = "if (message['content'] is none or";
    private static final String MISTRAL_CHECK =
            "{%- if (message['content'] is none or message['content'] == '' or "
            + "message['content']|length == 0) and (message['tool_calls'] is not defined or "
            + "message['tool_calls'] is none or message['tool_calls']|length == 0) %}";

    private static final String DISABLED_CHECK = "{%- if false %}";

    private ChatTemplateSource() {}

    /**
     * The template source prompts are rendered with.
     *
     * @param defaultTemplate the {@code tokenizer.chat_template} value, or null when absent
     * @param toolUseTemplate the {@code tokenizer.chat_template.tool_use} value, or null when absent
     * @return the resolved and patched Jinja source; never null or empty
     */
    public static String resolve(String defaultTemplate, String toolUseTemplate) {
        String src = defaultTemplate == null ? "" : defaultTemplate;
        String toolUse = toolUseTemplate == null ? "" : toolUseTemplate;
        if (src.isEmpty() || CHATML_NAME.equals(src)) {
            src = toolUse.isEmpty() ? CHATML : toolUse;
        }
        return patch(src);
    }

    /**
     * Applies llama.cpp's source patches: the gpt-oss check that rejects assistant history containing
     * channel markers, and the Mistral check that rejects assistant turns without content or tool calls,
     * both become {@code {%- if false %}} (common/chat.cpp:760-780).
     */
    static String patch(String src) {
        String out = src;
        if (out.contains(HARMONY_CHANNEL) && out.contains(HARMONY_CHECK_HINT)) {
            out = out.replace(HARMONY_CHECK, DISABLED_CHECK);
        }
        if (out.contains(MISTRAL_TOOL_CALLS) && out.contains(MISTRAL_CHECK_HINT)) {
            out = out.replace(MISTRAL_CHECK, DISABLED_CHECK);
        }
        return out;
    }
}
