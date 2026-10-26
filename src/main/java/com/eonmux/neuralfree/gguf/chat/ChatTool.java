package com.eonmux.neuralfree.gguf.chat;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A function tool definition a chat request offers the model, as the OpenAI ({@code tools})
 * and llama.cpp ({@code common_params.tools}, JSON-schema function definitions) shapes do:
 * a function {@code name}, an optional {@code description} and an optional {@code parameters}
 * JSON-schema object the arguments follow.
 *
 * <p>{@link #templateObject()} renders the shape llama.cpp binds as an entry of the template's
 * {@code tools} list (common/chat.cpp apply_jinja), from which templates such as Qwen's write
 * the tool preamble with {@code tojson}. Immutable; copies its parameter tree in and out, so
 * the caller's tree can never be modified through an instance.</p>
 */
public final class ChatTool {

    private final String name;
    private final String description;
    private final Map<String, Object> parameters;

    private ChatTool(Builder b) {
        this.name = b.name;
        this.description = b.description;
        this.parameters = b.parameters;
    }

    /** Starts a definition for a function whose name is set later on the builder. */
    public static Builder builder() {
        return new Builder();
    }

    /** Starts a definition for the function {@code name}. */
    public static Builder builder(String name) {
        return new Builder().name(name);
    }

    /** The function name the model calls it by; never null or empty. */
    public String name() { return name; }

    /** The description the template writes into the tool preamble, or null when absent. */
    public String description() { return description; }

    /**
     * The JSON-schema object the function's arguments follow; a new deep copy on every call,
     * or null when the function declares none.
     */
    public Map<String, Object> parameters() {
        return parameters == null ? null : Collections.unmodifiableMap(chatCopyObject(parameters, name));
    }

    /** The map bound as one entry of the template's {@code tools} list ({@code tojson} input). */
    public Map<String, Object> templateObject() {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", name);
        if (description != null) fn.put("description", description);
        if (parameters != null) fn.put("parameters", parameters);
        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", "function");
        tool.put("function", fn);
        return tool;
    }

    /** A new builder with this instance's values. */
    public Builder toBuilder() {
        return new Builder().name(name).description(description).parameters(parameters);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ChatTool)) return false;
        ChatTool that = (ChatTool) o;
        return name.equals(that.name)
                && Objects.equals(description, that.description)
                && Objects.equals(parameters, that.parameters);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, description, parameters);
    }

    @Override
    public String toString() {
        return "ChatTool(" + name + ", description=" + description + ", parameters=" + parameters + ")";
    }

    /** Builder of an immutable {@link ChatTool}; starts with no description and no parameters. */
    public static final class Builder {

        private String name = null;
        private String description = null;
        private Map<String, Object> parameters = null;

        private Builder() {}

        /**
         * @throws IllegalArgumentException when {@code name} is null, empty or blank
         */
        public Builder name(String name) {
            if (name == null) throw new IllegalArgumentException("tool name is null");
            String trimmed = name.trim();
            if (trimmed.isEmpty()) throw new IllegalArgumentException("tool name is empty");
            this.name = trimmed;
            return this;
        }

        /** Sets the description; null clears it. */
        public Builder description(String description) {
            this.description = description == null || description.isEmpty() ? null : description;
            return this;
        }

        /**
         * Sets the JSON-schema object of the function's arguments, copied.
         *
         * @throws IllegalArgumentException when the schema is not a JSON object or its keys are not strings
         */
        public Builder parameters(Map<String, Object> parameters) {
            this.parameters = parameters == null ? null : chatCopyObject(parameters, name);
            return this;
        }

        public ChatTool build() {
            if (name == null) throw new IllegalArgumentException("tool name is null");
            return new ChatTool(this);
        }
    }

    /**
     * Deep copy of a JSON object tree; the keys must be strings and the values JSON values.
     * {@code toolName} names the tool in validation messages (a static context, separate from the
     * instance's {@link #name}).
     */
    private static Map<String, Object> chatCopyObject(Map<?, ?> parameters, String toolName) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : parameters.entrySet()) {
            if (!(e.getKey() instanceof String)) {
                throw new IllegalArgumentException("the parameter schema of tool '" + toolName
                        + "' has a non-string key");
            }
            copy.put((String) e.getKey(), ChatMessages.copyValue(e.getValue(), "tool '" + toolName + "'"));
        }
        return copy;
    }
}
