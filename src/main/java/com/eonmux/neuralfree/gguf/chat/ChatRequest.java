package com.eonmux.neuralfree.gguf.chat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One chat-formatting request, the transport-neutral form the serving layer maps OpenAI,
 * llama.cpp and vLLM chat requests into and {@code GGUFTokenizer.formatChat(ChatRequest)} takes
 * ({@code ChatFormatter#format}): the {@code messages}, the {@code tools} (JSON-schema function
 * definitions, {@link ChatTool}), the {@code tool_choice}, the template variables
 * ({@code chat_template_kwargs}, {@code enable_thinking}, {@code reasoning_effort}), the
 * generation-prompt controls ({@code add_generation_prompt}, {@code continue_final_message}) and
 * the per-request template override ({@code chat_template}) with its {@code documents} —
 * the inputs common_chat_template rendering takes in llama.cpp (common/chat.cpp apply_jinja,
 * server-chat.cpp:593).
 *
 * <p>Immutable. The builder copies the message list and every JSON-valued field in, and the
 * getters hand out unmodifiable views, so neither a caller nor a renderer can mutate a request:
 * request descriptors are immutable and copy arrays in and out. Rendering happens on the
 * request thread; one request formats one prompt.</p>
 */
public final class ChatRequest {

    private final List<Map<String, Object>> messages;
    private final List<ChatTool> tools;
    private final ToolChoice toolChoice;
    private final Map<String, Object> templateKwargs;
    private final boolean addGenerationPrompt;
    private final boolean continueFinalMessage;
    private final String reasoningEffort;
    private final Boolean enableThinking;
    private final String chatTemplate;
    private final List<Map<String, Object>> documents;

    private ChatRequest(Builder b) {
        this.messages = b.messages;
        this.tools = b.tools;
        this.toolChoice = b.toolChoice;
        this.templateKwargs = b.templateKwargs;
        this.addGenerationPrompt = b.addGenerationPrompt;
        this.continueFinalMessage = b.continueFinalMessage;
        this.reasoningEffort = b.reasoningEffort;
        this.enableThinking = b.enableThinking;
        this.chatTemplate = b.chatTemplate;
        this.documents = b.documents;
    }

    /** A new builder: an empty conversation, no tools and no kwargs, generation prompt on. */
    public static Builder builder() {
        return new Builder();
    }

    /** The validated messages; unmodifiable, each with at least a string {@code role}. */
    public List<Map<String, Object>> messages() {
        return Collections.unmodifiableList(messages);
    }

    /** The offered function definitions; unmodifiable, never null. */
    public List<ChatTool> tools() {
        return Collections.unmodifiableList(tools);
    }

    /** The {@code tool_choice}; null (the default) lets the model decide, as {@code auto} does. */
    public ToolChoice toolChoice() { return toolChoice; }

    /**
     * The {@code chat_template_kwargs}: each entry is bound into the template context last, so
     * it can carry the template's own variables, as llama.cpp merges them (server-chat.cpp:593).
     * Unmodifiable.
     */
    public Map<String, Object> templateKwargs() {
        return Collections.unmodifiableMap(templateKwargs);
    }

    /**
     * The template variables this request sets, as the template sees them: {@code reasoning_effort} and
     * {@code enable_thinking} when set, then every {@code chat_template_kwargs} entry over them (the order
     * {@code ChatFormatter} binds them in). The request-dependent reasoning parsers read their switches here
     * ({@code ReasoningFilter#forRequest}). Unmodifiable.
     */
    public Map<String, Object> templateVariables() {
        Map<String, Object> out = new LinkedHashMap<>();
        if (reasoningEffort != null) out.put("reasoning_effort", reasoningEffort);
        if (enableThinking != null) out.put("enable_thinking", enableThinking);
        out.putAll(templateKwargs);
        return Collections.unmodifiableMap(out);
    }

    /** Whether the template adds the assistant generation prompt after the messages. */
    public boolean addGenerationPrompt() { return addGenerationPrompt; }

    /**
     * Whether the last assistant message is continued instead of a new turn generated (vLLM
     * {@code continue_final_message}): rendered without the generation prompt and with the
     * template's end-of-turn after that message stripped ({@code ChatFormatter#format}).
     */
    public boolean continueFinalMessage() { return continueFinalMessage; }

    /**
     * The {@code reasoning_effort} a reasoning model's template receives ({@code low | medium |
     * high}); null when unset, leaving the template's own default.
     */
    public String reasoningEffort() { return reasoningEffort; }

    /**
     * The {@code enable_thinking} of thinking templates; null when unset, which leaves the
     * variable undefined and the template's own default in force.
     */
    public Boolean enableThinking() { return enableThinking; }

    /**
     * The per-request Jinja template that replaces the model's resolved template for this
     * rendering only (vLLM {@code chat_template}); null renders with the model's own.
     */
    public String chatTemplate() { return chatTemplate; }

    /**
     * The documents a retrieval template renders into its context (vLLM {@code documents}); each
     * is a JSON object, deep-copied. Empty when none were given, leaving the variable undefined.
     */
    public List<Map<String, Object>> documents() {
        return Collections.unmodifiableList(documents);
    }

    /**
     * This request over configured default template variables (vLLM's {@code --default-chat-template-kwargs},
     * llama.cpp's {@code --chat-template-kwargs}), as vLLM's {@code ChatParams.with_defaults} merges them: a
     * default applies unless the request sets that variable itself, through {@code chat_template_kwargs} or the
     * top-level {@code enable_thinking} and {@code reasoning_effort}. A request kwarg that is null or
     * {@code "auto"} counts as unset (vLLM's {@code merge_kwargs}), so the default replaces it. The applied
     * defaults lead the kwargs, the request's own follow.
     *
     * @param defaults the default variables; null or empty leaves the request as it is
     * @return this request when no default applies, else a new request
     * @throws IllegalArgumentException when a default value is not a JSON value tree
     */
    public ChatRequest withDefaultTemplateKwargs(Map<String, ?> defaults) {
        if (defaults == null || defaults.isEmpty()) return this;
        Map<String, Object> merged = new LinkedHashMap<>();
        for (Map.Entry<String, ?> e : defaults.entrySet()) {
            if (!setsVariable(e.getKey())) merged.put(e.getKey(), e.getValue());
        }
        if (merged.isEmpty()) return this;
        for (Map.Entry<String, Object> e : templateKwargs.entrySet()) {
            if (!merged.containsKey(e.getKey())) merged.put(e.getKey(), e.getValue());
        }
        return toBuilder().templateKwargs(merged).build();
    }

    /** Whether this request sets template variable {@code name} to a value a default must not replace. */
    private boolean setsVariable(String name) {
        if ("enable_thinking".equals(name) && enableThinking != null) return true;
        if ("reasoning_effort".equals(name) && reasoningEffort != null) return true;
        if (!templateKwargs.containsKey(name)) return false;
        Object value = templateKwargs.get(name);
        return value != null && !"auto".equals(value);
    }

    /** A new builder with this request's values. */
    public Builder toBuilder() {
        Builder b = new Builder();
        b.messages = messages;
        b.tools = tools;
        b.toolChoice = toolChoice;
        b.templateKwargs = templateKwargs;
        b.addGenerationPrompt = addGenerationPrompt;
        b.continueFinalMessage = continueFinalMessage;
        b.reasoningEffort = reasoningEffort;
        b.enableThinking = enableThinking;
        b.chatTemplate = chatTemplate;
        b.documents = documents;
        return b;
    }

    @Override
    public String toString() {
        return "ChatRequest(" + messages.size() + " messages, " + tools.size() + " tools, toolChoice="
                + toolChoice + ", kwargs=" + templateKwargs.keySet() + ", addGenerationPrompt="
                + addGenerationPrompt + ", continueFinalMessage=" + continueFinalMessage
                + ", reasoningEffort=" + reasoningEffort + ", enableThinking=" + enableThinking
                + ", chatTemplate=" + (chatTemplate == null ? "null" : chatTemplate.length() + " chars")
                + ", documents=" + documents.size() + ")";
    }

    /** Builder of an immutable {@link ChatRequest}. */
    public static final class Builder {

        private List<Map<String, Object>> messages = new ArrayList<>();
        private List<ChatTool> tools = new ArrayList<>();
        private ToolChoice toolChoice = null;
        private Map<String, Object> templateKwargs = new LinkedHashMap<>();
        private boolean addGenerationPrompt = true;
        private boolean continueFinalMessage = false;
        private String reasoningEffort = null;
        private Boolean enableThinking = null;
        private String chatTemplate = null;
        private List<Map<String, Object>> documents = new ArrayList<>();

        private Builder() {}

        /**
         * Replaces the messages; each is validated (a map with a string {@code role}) and copied,
         * as {@link ChatMessages#copyOf} does.
         *
         * @throws IllegalArgumentException for a null list, a null message, or a roleless message
         */
        public Builder messages(List<? extends Map<String, ?>> messages) {
            this.messages = ChatMessages.copyOf(messages);
            return this;
        }

        /** Appends one message, validated as {@link #messages} validates it. */
        public Builder addMessage(Map<String, ?> message) {
            this.messages = append(this.messages, Collections.singletonList(message));
            return this;
        }

        private static List<Map<String, Object>> append(List<Map<String, Object>> list,
                                                        List<? extends Map<String, ?>> more) {
            List<Map<String, Object>> out = new ArrayList<>(list.size() + more.size());
            out.addAll(list);
            out.addAll(ChatMessages.copyOf(more));
            return out;
        }

        /** Replaces the offered tools. */
        public Builder tools(List<ChatTool> tools) {
            this.tools = tools == null ? new ArrayList<>() : new ArrayList<>(tools);
            return this;
        }

        public Builder toolChoice(ToolChoice toolChoice) {
            this.toolChoice = toolChoice;
            return this;
        }

        /**
         * Sets the {@code chat_template_kwargs}, deep-copied.
         *
         * @throws IllegalArgumentException when a value is not a JSON value tree
         */
        public Builder templateKwargs(Map<String, ?> templateKwargs) {
            if (templateKwargs == null) {
                this.templateKwargs = new LinkedHashMap<>();
                return this;
            }
            Map<String, Object> copy = new LinkedHashMap<>();
            for (Map.Entry<String, ?> e : templateKwargs.entrySet()) {
                copy.put(e.getKey(), copyKwargValue(e.getKey(), e.getValue()));
            }
            this.templateKwargs = copy;
            return this;
        }

        /** Adds one template variable, as {@link #templateKwargs} validates it. */
        public Builder templateKwarg(String name, Object value) {
            if (name == null) throw new IllegalArgumentException("template variable name is null");
            Map<String, Object> copy = new LinkedHashMap<>(this.templateKwargs);
            copy.put(name, copyKwargValue(name, value));
            this.templateKwargs = copy;
            return this;
        }

        private static Object copyKwargValue(String name, Object value) {
            return ChatMessages.copyValue(value, "chat_template_kwargs key '" + name + "'");
        }

        public Builder addGenerationPrompt(boolean addGenerationPrompt) {
            this.addGenerationPrompt = addGenerationPrompt;
            return this;
        }

        public Builder continueFinalMessage(boolean continueFinalMessage) {
            this.continueFinalMessage = continueFinalMessage;
            return this;
        }

        /**
         * Sets the {@code reasoning_effort} pass-through ({@code low | medium | high}).
         *
         * @throws IllegalArgumentException when the value is empty
         */
        public Builder reasoningEffort(String reasoningEffort) {
            if (reasoningEffort != null && reasoningEffort.isEmpty()) {
                throw new IllegalArgumentException("reasoning_effort is empty");
            }
            this.reasoningEffort = reasoningEffort;
            return this;
        }

        /** Sets {@code enable_thinking} of thinking templates; null leaves the variable unset. */
        public Builder enableThinking(Boolean enableThinking) {
            this.enableThinking = enableThinking;
            return this;
        }

        /**
         * Sets the per-request template override (vLLM {@code chat_template}).
         *
         * @throws IllegalArgumentException when the template is empty
         */
        public Builder chatTemplate(String chatTemplate) {
            if (chatTemplate != null && chatTemplate.isEmpty()) {
                throw new IllegalArgumentException("chat_template override is empty");
            }
            this.chatTemplate = chatTemplate;
            return this;
        }

        /**
         * Sets the documents a retrieval template renders into its context; each is validated (a
         * JSON object) and deep-copied, as {@link #templateKwargs} copies its values.
         *
         * @throws IllegalArgumentException for a null list, a null document or a non-object document
         */
        public Builder documents(List<? extends Map<String, ?>> documents) {
            if (documents == null) {
                this.documents = new ArrayList<>();
                return this;
            }
            List<Map<String, Object>> copy = new ArrayList<>(documents.size());
            for (int i = 0; i < documents.size(); i++) {
                Map<String, ?> document = documents.get(i);
                if (document == null) {
                    throw new IllegalArgumentException("document " + i + " is null");
                }
                Map<String, Object> entry = new LinkedHashMap<>();
                for (Map.Entry<String, ?> e : document.entrySet()) {
                    entry.put(String.valueOf(e.getKey()),
                            ChatMessages.copyValue(e.getValue(), "document " + i + " key '" + e.getKey() + "'"));
                }
                copy.add(entry);
            }
            this.documents = copy;
            return this;
        }

        public ChatRequest build() {
            if (continueFinalMessage) {
                if (messages.isEmpty()) {
                    throw new IllegalArgumentException("continue_final_message requires a last"
                            + " assistant message; the conversation is empty");
                }
                if (!"assistant".equals(messages.get(messages.size() - 1).get(ChatMessages.ROLE))) {
                    throw new IllegalArgumentException("continue_final_message requires an"
                            + " assistant message last, got role '"
                            + messages.get(messages.size() - 1).get(ChatMessages.ROLE) + "'");
                }
            }
            return new ChatRequest(this);
        }
    }

    /**
     * The {@code tool_choice}: {@code none}, {@code auto}, {@code required} or one named
     * {@code function}; {@link com.eonmux.neuralfree.gguf.chat.tools.ToolRouting} turns the
     * forcing choices into tool grammars, as llama.cpp's common/chat.cpp does. Immutable;
     * {@link Kind#FUNCTION} always carries its function name.
     */
    public static final class ToolChoice {

        /** What the model is constrained to. */
        public enum Kind { NONE, AUTO, REQUIRED, FUNCTION }

        private final Kind kind;
        private final String functionName;

        private ToolChoice(Kind kind, String functionName) {
            this.kind = kind;
            this.functionName = functionName;
        }

        public static ToolChoice none() { return new ToolChoice(Kind.NONE, null); }

        public static ToolChoice auto() { return new ToolChoice(Kind.AUTO, null); }

        public static ToolChoice required() { return new ToolChoice(Kind.REQUIRED, null); }

        /** The named function, as OpenAI's {@code {"type": "function", "function": {"name": ...}}} does. */
        public static ToolChoice function(String name) {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("tool_choice function name is null or empty");
            }
            return new ToolChoice(Kind.FUNCTION, name);
        }

        /** The OpenAI {@code tool_choice} string ({@code none | auto | required}), or null when unset. */
        public static ToolChoice fromOpenAi(String value) {
            if (value == null) return null;
            if (value.isEmpty() || "auto".equals(value)) return value.isEmpty() ? null : auto();
            if ("none".equals(value)) return none();
            if ("required".equals(value)) return required();
            throw new IllegalArgumentException("tool_choice '" + value + "' is not one of none, auto, required");
        }

        public Kind kind() { return kind; }

        /** The named function; null for the other kinds. */
        public String functionName() { return functionName; }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof ToolChoice)) return false;
            ToolChoice that = (ToolChoice) o;
            return kind == that.kind && Objects.equals(functionName, that.functionName);
        }

        @Override
        public int hashCode() {
            return Objects.hash(kind, functionName);
        }

        @Override
        public String toString() {
            return kind == Kind.FUNCTION ? "function(" + functionName + ")" : kind.name().toLowerCase();
        }
    }
}
