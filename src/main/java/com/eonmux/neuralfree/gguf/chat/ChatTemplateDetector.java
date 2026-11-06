package com.eonmux.neuralfree.gguf.chat;

/**
 * Guesses one of llama.cpp's built-in chat templates from a template's text, as llama.cpp's
 * {@code llm_chat_detect_template} does (src/llama-chat.cpp:89-240): the text is first read as a
 * built-in name ({@link ChatTemplateSource#isName}), then the substring checks of llama.cpp run in
 * its order. It is not a Jinja parser; a text that matches no style detects nothing, which the
 * non-Jinja route refuses (llama.cpp's {@code common_chat_verify_template} with
 * {@code use_jinja = false}, common/arg.cpp:940-946).
 *
 * <p>The guess may be a name the name table does not carry: llama.cpp's {@code dots1} exists only
 * as a detection result ({@link ChatTemplateSource} keeps it out of {@link ChatTemplateSource#names()}).
 * The needles are llama.cpp's, so a model template llama.cpp detects, this detects, and one llama.cpp
 * misses (for example a Zephyr text that also names {@code <|assistant|>}, which llama.cpp's Falcon
 * and GLM-Edge checks answer first) stays undetected here too.</p>
 */
public final class ChatTemplateDetector {

    private ChatTemplateDetector() {}

    /**
     * The built-in template {@code template}'s text detects, or null when nothing is detected
     * (llama.cpp's {@code LLM_CHAT_TEMPLATE_UNKNOWN}).
     */
    public static String detect(String template) {
        if (template == null) return null;
        if (ChatTemplateSource.isName(template)) return template;

        if (contains(template, "<|im_start|>")) {
            if (contains(template, "<|im_sep|>")) return "phi4";
            if (contains(template, "<end_of_utterance>")) return "smolvlm";
            return "chatml";
        }
        if (template.startsWith("mistral") || contains(template, "[INST]")) {
            if (contains(template, "[SYSTEM_PROMPT]")) return "mistral-v7";
            if (contains(template, "' [INST] ' + system_message") || contains(template, "[AVAILABLE_TOOLS]")) {
                // Official mistral 'v1', 'v3' and 'v3-tekken' templates
                if (contains(template, " [INST]")) return "mistral-v1";
                if (contains(template, "\"[INST]\"")) return "mistral-v3-tekken";
                return "mistral-v3";
            }
            // llama2 template and its variants
            boolean stripMessage = contains(template, "content.strip()");
            boolean addBosInsideHistory = contains(template, "bos_token + '[INST]");
            if (stripMessage) return "llama2-sys-strip";
            if (addBosInsideHistory) return "llama2-sys-bos";
            if (contains(template, "<<SYS>>")) return "llama2-sys";
            return "llama2";
        }
        if (contains(template, "<|assistant|>") && contains(template, "<|end|>")) return "phi3";
        if (contains(template, "[gMASK]<sop>")) return "chatglm4";
        if (contains(template, "<|assistant|>") && contains(template, "<|user|>")) {
            if (contains(template, "<|tool_declare|>")) return "exaone-moe";
            return contains(template, "</s>") ? "falcon3" : "glmedge";
        }
        if (contains(template, "<|{{ item['role'] }}|>") && contains(template, "<|begin_of_image|>")) return "glmedge";
        if (contains(template, "<|user|>") && contains(template, "<|endoftext|>")) return "zephyr";
        if (contains(template, "bos_token + message['role']")) return "monarch";
        if (contains(template, "<start_of_turn>")) return "gemma";
        if (contains(template, "'\\n\\nAssistant: ' + eos_token")) return "orion";
        if (contains(template, "GPT4 Correct ")) return "openchat";
        if (contains(template, "USER: ") && contains(template, "ASSISTANT: ")) {
            return contains(template, "SYSTEM: ") ? "vicuna-orca" : "vicuna";
        }
        if (contains(template, "### Instruction:") && contains(template, "<|EOT|>")) return "deepseek";
        if (contains(template, "<|START_OF_TURN_TOKEN|>") && contains(template, "<|USER_TOKEN|>")) return "command-r";
        if (contains(template, "<|start_header_id|>") && contains(template, "<|end_header_id|>")) return "llama3";
        if (contains(template, "[gMASK]sop")) return "chatglm3";
        if (contains(template, "<用户>")) return "minicpm";
        if (contains(template, "'Assistant: ' + message['content'] + eos_token")) return "deepseek2";
        if (contains(template, "<｜Assistant｜>") && contains(template, "<｜User｜>")
                && contains(template, "<｜end▁of▁sentence｜>")) return "deepseek3";
        if (contains(template, "[|system|]") && contains(template, "[|assistant|]")
                && contains(template, "[|endofturn|]")) {
            return contains(template, "[|tool|]") ? "exaone4" : "exaone3";
        }
        if (contains(template, "rwkv-world")
                || contains(template, "{{- 'User: ' + message['content']|trim + '\\n\\n' -}}")) return "rwkv-world";
        if (contains(template, "<|start_of_role|>")) {
            if (contains(template, "<tool_call>") || contains(template, "<tools>")) {
                return contains(template, "g4_default_system_message") ? "granite-4.0" : "granite-4.1";
            }
            return "granite";
        }
        if (contains(template, "message['role'] + additional_special_tokens[0] + message['content']"
                + " + additional_special_tokens[1]")) return "gigachat";
        if (contains(template, "<|role_start|>")) return "megrez";
        if (contains(template, " Ассистент:")) return "yandex";
        if (contains(template, "<role>ASSISTANT</role>") && contains(template, "'HUMAN'")) return "bailing";
        if (contains(template, "<role>ASSISTANT</role>") && contains(template, "\"HUMAN\"")
                && contains(template, "<think>")) return "bailing-think";
        if (contains(template, "<role>ASSISTANT</role>") && contains(template, "<role>HUMAN</role>")
                && contains(template, "<|role_end|>")) return "bailing2";
        if (contains(template, "<|header_start|>") && contains(template, "<|header_end|>")) return "llama4";
        if (contains(template, "<|endofuserprompt|>")) return ChatTemplateSource.DOTS1_NAME;
        if (contains(template, "<|extra_0|>") && contains(template, "<|extra_4|>")) return "hunyuan-moe";
        if (contains(template, "<|start|>") && contains(template, "<|channel|>")) return "gpt-oss";
        if (contains(template, "<｜hy_Assistant｜>") && contains(template, "<｜hy_begin▁of▁sentence｜>"))
            return "hunyuan-vl";
        if (contains(template, "<｜hy_Assistant｜>") && contains(template, "<｜hy_place▁holder▁no▁3｜>"))
            return "hunyuan-dense";
        if (contains(template, "<|im_assistant|>assistant<|im_middle|>")) return "kimi-k2";
        if (contains(template, "<seed:bos>")) return "seed_oss";
        if (contains(template, "'Assistant: '  + message['content'] + '<|separator|>")) return "grok-2";
        if (contains(template, "[unused9]系统：[unused10]")) return "pangu-embedded";
        if (contains(template, "<|begin|>") && contains(template, "<|end|>") && contains(template, "<|content|>"))
            return "solar-open";
        return null;
    }

    /**
     * The refusal for a template text that {@link #detect} finds nothing in, naming the template the
     * way llama.cpp's verify does ("the supplied chat template is not supported", common/arg.cpp:941):
     * at most 80 characters of it, newlines escaped, ahead of the count of chars shown of the whole.
     */
    public static String refusal(String template) {
        int length = template == null ? 0 : template.length();
        String shown = template == null ? "" : template.substring(0, Math.min(80, length));
        return "no built-in chat template was detected in the chat template (" + shown.length() + " of " + length
                + " chars shown: '" + shown.replace("\n", "\\n") + "'); the non-Jinja route renders only"
                + " llama.cpp's known templates";
    }

    private static boolean contains(String template, String needle) {
        return template.contains(needle);
    }
}
