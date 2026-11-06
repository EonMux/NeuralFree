package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.MiniJinja;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * What a chat template renders, llama.cpp's {@code jinja::caps} (common/jinja/caps.h): whether it describes
 * offered tools, renders assistant tool calls (and several in one message), renders a system message, keeps the
 * reasoning of earlier assistant turns, reads {@code reasoning_effort} (and which levels it renders distinctly),
 * reads message content as a string or as a list of typed parts, and reads tool-call arguments as an object.
 * {@link #probe} detects them by rendering probe conversations ({@link TemplateCapsProbe}).
 *
 * <p>The content and argument capabilities drive the template input ({@link ChatMessages#normalize}): string
 * content is wrapped as one text part for a template that reads only typed parts, typed text parts are joined to
 * a string for a template that reads only strings, and tool-call arguments are an object only for a template
 * that reads them as one.</p>
 *
 * <p>Immutable.</p>
 */
public final class TemplateCaps {

    /** The template variable that switches the reasoning-history aliases on or off ({@link #applyPreserveReasoning}). */
    public static final String PRESERVE_REASONING = "preserve_reasoning";

    /**
     * vLLM's {@code --chat-template-content-format}: {@code auto} keeps the detected content capabilities,
     * {@code string} renders every content as a string and {@code openai} every content as typed parts
     * ({@link #withContentFormat}).
     */
    public enum ContentFormat {
        AUTO, STRING, OPENAI;

        /** The lower-case option value, as the launch flag spells it. */
        public String value() { return name().toLowerCase(Locale.ROOT); }

        /**
         * The format named {@code value} ({@code auto}, {@code string} or {@code openai}).
         *
         * @throws IllegalArgumentException for any other value
         */
        public static ContentFormat parse(String value) {
            for (ContentFormat f : values()) {
                if (f.value().equals(value)) return f;
            }
            throw new IllegalArgumentException("chat template content format must be auto, string or openai, got '"
                    + value + "'");
        }
    }

    private final boolean supportsTools;
    private final boolean supportsToolCalls;
    private final boolean supportsSystemRole;
    private final boolean supportsParallelToolCalls;
    private final boolean supportsPreserveReasoning;
    private final boolean supportsReasoningEffort;
    private final List<String> reasoningEffortLevels;
    private final boolean supportsStringContent;
    private final boolean supportsTypedContent;
    private final boolean supportsObjectArguments;

    private TemplateCaps(Builder b) {
        this.supportsTools = b.supportsTools;
        this.supportsToolCalls = b.supportsToolCalls;
        this.supportsSystemRole = b.supportsSystemRole;
        this.supportsParallelToolCalls = b.supportsParallelToolCalls;
        this.supportsPreserveReasoning = b.supportsPreserveReasoning;
        this.supportsReasoningEffort = b.supportsReasoningEffort;
        this.reasoningEffortLevels = Collections.unmodifiableList(new ArrayList<>(b.reasoningEffortLevels));
        this.supportsStringContent = b.supportsStringContent;
        this.supportsTypedContent = b.supportsTypedContent;
        this.supportsObjectArguments = b.supportsObjectArguments;
    }

    /**
     * The capabilities of {@code template}, detected as llama.cpp's {@code caps_get} detects them.
     *
     * @throws IllegalArgumentException if {@code template} is null
     */
    public static TemplateCaps probe(MiniJinja.Template template) {
        if (template == null) throw new IllegalArgumentException("chat template is null");
        return TemplateCapsProbe.run(template);
    }

    /** Whether the template describes the offered {@code tools} (it reads a tool's function name). */
    public boolean supportsTools() { return supportsTools; }

    /** Whether the template renders an assistant message's {@code tool_calls}. */
    public boolean supportsToolCalls() { return supportsToolCalls; }

    /** Whether the template renders a leading system message; when not, it is merged into the next message. */
    public boolean supportsSystemRole() { return supportsSystemRole; }

    /** Whether the template renders every call of a message carrying several {@code tool_calls}. */
    public boolean supportsParallelToolCalls() { return supportsParallelToolCalls; }

    /** Whether the template renders the reasoning of earlier assistant turns when asked to preserve it. */
    public boolean supportsPreserveReasoning() { return supportsPreserveReasoning; }

    /** Whether the template reads {@code reasoning_effort}. */
    public boolean supportsReasoningEffort() { return supportsReasoningEffort; }

    /** The reasoning-effort levels the template renders distinctly, lowest first; unmodifiable. */
    public List<String> reasoningEffortLevels() { return reasoningEffortLevels; }

    /** Whether the template renders a message whose {@code content} is a string. */
    public boolean supportsStringContent() { return supportsStringContent; }

    /** Whether the template reads a message {@code content} given as a list of typed parts. */
    public boolean supportsTypedContent() { return supportsTypedContent; }

    /** Whether the template reads tool-call {@code arguments} as an object rather than as JSON text. */
    public boolean supportsObjectArguments() { return supportsObjectArguments; }

    /**
     * These capabilities with the content capabilities {@code format} forces: {@link ContentFormat#STRING} reads
     * only strings, {@link ContentFormat#OPENAI} only typed parts; {@link ContentFormat#AUTO} returns this.
     */
    public TemplateCaps withContentFormat(ContentFormat format) {
        if (format == null || format == ContentFormat.AUTO) return this;
        Builder b = toBuilder();
        b.supportsStringContent = format == ContentFormat.STRING;
        b.supportsTypedContent = format == ContentFormat.OPENAI;
        return b.build();
    }

    /**
     * These capabilities with the system role supported, for a template that places (or leaves out) system
     * messages itself, so they reach it unmerged; this when the system role is already supported.
     */
    public TemplateCaps withSystemRole() {
        if (supportsSystemRole) return this;
        Builder b = toBuilder();
        b.supportsSystemRole = true;
        return b.build();
    }

    /** The content format detected: {@code openai} when typed parts are read and strings are not, else {@code string}. */
    public ContentFormat detectedContentFormat() {
        return supportsTypedContent && !supportsStringContent ? ContentFormat.OPENAI : ContentFormat.STRING;
    }

    private Builder toBuilder() {
        Builder b = new Builder();
        b.supportsTools = supportsTools;
        b.supportsToolCalls = supportsToolCalls;
        b.supportsSystemRole = supportsSystemRole;
        b.supportsParallelToolCalls = supportsParallelToolCalls;
        b.supportsPreserveReasoning = supportsPreserveReasoning;
        b.supportsReasoningEffort = supportsReasoningEffort;
        b.reasoningEffortLevels.addAll(reasoningEffortLevels);
        b.supportsStringContent = supportsStringContent;
        b.supportsTypedContent = supportsTypedContent;
        b.supportsObjectArguments = supportsObjectArguments;
        return b;
    }

    /**
     * The capabilities as llama.cpp's {@code /props} {@code chat_template_caps} reports them
     * ({@code caps::to_map} plus {@code reasoning_effort_levels}); a new mutable map.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("supports_object_arguments", supportsObjectArguments);
        out.put("supports_parallel_tool_calls", supportsParallelToolCalls);
        out.put("supports_preserve_reasoning", supportsPreserveReasoning);
        out.put("supports_reasoning_effort", supportsReasoningEffort);
        out.put("supports_string_content", supportsStringContent);
        out.put("supports_system_role", supportsSystemRole);
        out.put("supports_tool_calls", supportsToolCalls);
        out.put("supports_tools", supportsTools);
        out.put("supports_typed_content", supportsTypedContent);
        out.put("reasoning_effort_levels", new ArrayList<>(reasoningEffortLevels));
        return out;
    }

    /**
     * Binds llama.cpp's reasoning-history aliases into a template context ({@code caps_apply_preserve_reasoning}):
     * {@code preserve_thinking} to {@code enabled} and {@code clear_thinking}, {@code truncate_history_thinking}
     * and {@code drop_thinking} to its negation, the variables the templates that can keep earlier reasoning read.
     */
    public static void applyPreserveReasoning(Map<String, Object> context, boolean enabled) {
        context.put("preserve_thinking", enabled);
        context.put("clear_thinking", !enabled);
        context.put("truncate_history_thinking", !enabled);
        context.put("drop_thinking", !enabled);
    }

    /**
     * Binds a reasoning effort into a template context as llama.cpp does ({@code caps_apply_reasoning_effort}):
     * one value as both {@code reasoning_effort} and {@code reasoning_strength}.
     */
    public static void applyReasoningEffort(Map<String, Object> context, String effort) {
        context.put("reasoning_effort", effort);
        context.put("reasoning_strength", effort);
    }

    @Override
    public String toString() {
        return "TemplateCaps" + toMap();
    }

    /** Mutable accumulator of a probe; starts at llama.cpp's defaults. */
    static final class Builder {
        boolean supportsTools = true;
        boolean supportsToolCalls = true;
        boolean supportsSystemRole = true;
        boolean supportsParallelToolCalls = true;
        boolean supportsPreserveReasoning;
        boolean supportsReasoningEffort;
        final List<String> reasoningEffortLevels = new ArrayList<>();
        boolean supportsStringContent = true;
        boolean supportsTypedContent;
        boolean supportsObjectArguments;

        TemplateCaps build() {
            return new TemplateCaps(this);
        }
    }
}
