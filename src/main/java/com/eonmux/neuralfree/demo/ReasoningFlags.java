package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.chat.TemplateCaps;

import java.util.Map;

/**
 * llama.cpp's reasoning chat flags ({@code common/arg.cpp}) as template variables: {@code -rea/--reasoning},
 * {@code --reasoning-effort} and {@code --reasoning-preserve}/{@code --no-reasoning-preserve} each write one entry
 * of the CLI's template variables, as llama.cpp writes its {@code default_template_kwargs}, so a later flag
 * replaces an earlier one and client mode forwards them as {@code chat_template_kwargs}.
 */
final class ReasoningFlags {

    private ReasoningFlags() {
    }

    /**
     * {@code --reasoning <value>}: llama.cpp's truthy spellings ({@code on}, {@code enabled}, {@code true},
     * {@code 1}) bind {@code enable_thinking = true} and return true, the falsey ones ({@code off},
     * {@code disabled}, {@code false}, {@code 0}) bind false and return false, and {@code auto} or {@code -1}
     * returns null (the template's own default) and leaves the variables as they are.
     *
     * @throws UsageException for any other value, naming {@code flag}
     */
    static Boolean reasoning(String flag, String value, Map<String, Object> kwargs) throws UsageException {
        switch (value) {
            case "on": case "enabled": case "true": case "1":
                kwargs.put("enable_thinking", true);
                return Boolean.TRUE;
            case "off": case "disabled": case "false": case "0":
                kwargs.put("enable_thinking", false);
                return Boolean.FALSE;
            case "auto": case "-1":
                return null;
            default:
                throw new UsageException(flag + " must be on, off or auto, got '" + value + "'");
        }
    }

    /**
     * {@code --reasoning-effort <level>}: binds {@code reasoning_effort} to {@code level}; {@code default} removes
     * it, leaving the template's own default.
     *
     * @throws UsageException for an empty level, naming {@code flag}
     */
    static void effort(String flag, String level, Map<String, Object> kwargs) throws UsageException {
        if (level.trim().isEmpty()) {
            throw new UsageException(flag + " needs a level such as low, medium or high, or default");
        }
        if ("default".equals(level)) {
            kwargs.remove(LocalChat.REASONING_EFFORT);
        } else {
            kwargs.put(LocalChat.REASONING_EFFORT, level);
        }
    }

    /** {@code --reasoning-preserve} (true) or {@code --no-reasoning-preserve} (false): binds {@code preserve_reasoning}. */
    static void preserve(boolean preserve, Map<String, Object> kwargs) {
        kwargs.put(TemplateCaps.PRESERVE_REASONING, preserve);
    }
}
