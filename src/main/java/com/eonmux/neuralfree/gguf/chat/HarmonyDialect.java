package com.eonmux.neuralfree.gguf.chat;

/**
 * The harmony-family dialect of a chat template: how a header/body model (the {@code HARMONY_CHANNELS} reasoning
 * mode) addresses its messages.
 */
public enum HarmonyDialect {
    /** OpenAI harmony (gpt-oss): {@code <|channel|>} headers, {@code <|end|>} between messages. */
    GPT_OSS,
    /** Muse Spark / Glimmer: {@code to=} recipients, {@code <|eom|>} between messages, ATEM tool bodies. */
    MUSE;

    /** The dialect of {@code template}: Muse when it writes {@code <|eom|>} without channels, else gpt-oss. */
    public static HarmonyDialect of(String template) {
        if (template != null && !template.contains("<|channel|>") && template.contains("<|eom|>")) {
            return MUSE;
        }
        return GPT_OSS;
    }
}
