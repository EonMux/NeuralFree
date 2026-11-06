package com.eonmux.neuralfree.gguf.chat.tools;

/**
 * The server-side floor for tool-call grammars (vLLM's {@code --tool-strict-level},
 * {@code tool_strict_level.py}): how far a request's tools are constrained in the model's own call syntax.
 *
 * <ul>
 *   <li>{@link #AUTO}: a grammar is built for a forcing {@code tool_choice} ({@code required} or a named
 *       function), and under {@code auto} only when a tool declares {@code strict: true}.</li>
 *   <li>{@link #FUNCTION}: every request with tools gets the call-envelope grammar (the format's tags and the
 *       declared function and parameter names), {@code auto} included.</li>
 *   <li>{@link #PARAMETER}: as {@link #FUNCTION}, plus every tool's argument schema is enforced as if it were
 *       {@code strict: true}.</li>
 * </ul>
 *
 * A forcing choice constrains every call, and a tool's argument schema holds unless the tool declares
 * {@code strict: false}; under {@code auto} the schema holds only for a tool that declares
 * {@code strict: true}, unless the level is {@link #PARAMETER}. The flag is vLLM's and OpenAI's per-tool
 * {@code strict} field.
 */
public enum ToolStrict {

    AUTO, FUNCTION, PARAMETER;

    /** The level a name selects ({@code neuralfree.tool.strict}); null or blank means {@link #AUTO}. */
    public static ToolStrict fromName(String name) {
        if (name == null || name.trim().isEmpty()) {
            return AUTO;
        }
        String trimmed = name.trim().toLowerCase(java.util.Locale.ROOT).replace('-', '_');
        try {
            return valueOf(trimmed.toUpperCase(java.util.Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("tool strict level '" + name + "' is not one of auto, function,"
                    + " parameter");
        }
    }

    /** Whether this level enforces every tool's argument schema. */
    public boolean atLeast(ToolStrict other) {
        return ordinal() >= other.ordinal();
    }
}
