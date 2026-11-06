package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.jinja.JinjaUsage;
import com.eonmux.neuralfree.gguf.jinja.JinjaValue;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The capability probe of {@link TemplateCaps}, llama.cpp's {@code caps_get} (common/jinja/caps.cpp): each
 * capability renders one probe conversation with value-usage statistics on ({@link JinjaUsage}) and reads back
 * which inputs the template used. The probe context is llama.cpp's: the messages, the tools (an empty list when
 * the probe offers none), empty BOS and EOS texts and the generation prompt.
 *
 * <p>A render that raises is retried once with an empty {@code reasoning_content} on every assistant message that
 * lacks one, for templates that require it on each assistant turn. The probe messages pass through
 * {@link ChatMessages#forTemplate}, so the template sees the tool-call shape it sees at render time.</p>
 */
final class TemplateCapsProbe {

    private static final Logger LOGGER = LogManager.getLogger(TemplateCapsProbe.class);

    private static final String CONTENT_MARKER = "STRING_MARKER";
    private static final String REASONING_PLACEHOLDER = "<REASONING_CONTENT_PLACEHOLDER>";
    private static final String UNKNOWN_EFFORT = "REASONING_EFFORT_UNKNOWN";
    /** llama.cpp's candidate reasoning-effort levels, lowest first. */
    private static final List<String> EFFORT_CANDIDATES = Collections.unmodifiableList(
            Arrays.asList("minimal", "low", "medium", "high", "xhigh", "max"));

    private final MiniJinja.Template template;
    private final TemplateCaps.Builder caps = new TemplateCaps.Builder();

    private TemplateCapsProbe(MiniJinja.Template template) {
        this.template = template;
    }

    static TemplateCaps run(MiniJinja.Template template) {
        TemplateCapsProbe probe = new TemplateCapsProbe(template);
        probe.content();
        probe.systemRole();
        probe.tools();
        probe.parallelToolCalls();
        probe.preserveReasoning();
        probe.reasoningEffort();
        TemplateCaps result = probe.caps.build();
        LOGGER.debug("Chat template capabilities: {}", result);
        return result;
    }

    // ------------------------------------------------------------------ probes

    /** String versus typed content: a string marker first, a typed marker part when the template tests for strings. */
    private void content() {
        Result plain = execute(list(message("user", CONTENT_MARKER)), null, null);
        Object content = plain.at(0, ChatMessages.CONTENT);
        boolean checksForString = plain.usage.hasOp(content, JinjaUsage.TEST_PREFIX + "string");
        boolean usedAsArray = usedAsArray(plain.usage, content);
        if (usedAsArray) caps.supportsTypedContent = true;
        if (!plain.success) {
            caps.supportsStringContent = false;
        } else if (usedAsArray && !plain.output.contains(CONTENT_MARKER)) {
            caps.supportsStringContent = false;
        }
        if (!checksForString) return;
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", s("text"));
        part.put("text", s(CONTENT_MARKER));
        Result typed = execute(list(message("user", list(part))), null, null);
        if (typed.success && usedAsArray(typed.usage, typed.at(0, ChatMessages.CONTENT))) {
            caps.supportsTypedContent = true;
        }
    }

    private static boolean usedAsArray(JinjaUsage usage, Object content) {
        return usage.hasOp(content, "selectattr") || usage.hasOp(content, JinjaUsage.ARRAY_ACCESS);
    }

    /** The system role is supported when the template reads the system message's content. */
    private void systemRole() {
        Result r = execute(list(message("system", "System message"), message("user", "User message")), null, null);
        if (!r.usage.used(r.at(0, ChatMessages.CONTENT))) caps.supportsSystemRole = false;
    }

    /** Tool descriptions and tool calls, with object arguments first and JSON-text arguments when those are unused. */
    private void tools() {
        Result object = execute(toolConversation(1, true), null, tool());
        if (object.success) {
            if (!object.usage.used(toolName(object))) caps.supportsTools = false;
            Object calls = object.at(1, "tool_calls");
            if (!object.usage.used(calls)) {
                caps.supportsToolCalls = false;
            } else if (object.usage.used(at(at(at(at(calls, 0), "function"), "arguments"), "arg"))) {
                caps.supportsObjectArguments = true;
            }
        }
        if (caps.supportsObjectArguments) return;
        Result text = execute(toolConversation(1, false), null, tool());
        if (!text.success) {
            caps.supportsToolCalls = false;
            caps.supportsTools = false;
            return;
        }
        if (!text.usage.used(toolName(text))) caps.supportsTools = false;
        if (!text.usage.used(text.at(1, "tool_calls"))) caps.supportsToolCalls = false;
    }

    private static Object toolName(Result r) {
        return at(at(at(r.usage.variable("tools"), 0), "function"), "name");
    }

    /**
     * Two calls in one message: supported when the template reads the second one's function, through the nested
     * {@code function} or the flat {@code name}/{@code arguments} keys the render binds beside it.
     */
    private void parallelToolCalls() {
        Result r = execute(toolConversation(2, caps.supportsObjectArguments), null, tool());
        if (!r.success) {
            caps.supportsParallelToolCalls = false;
            return;
        }
        Object second = at(r.at(1, "tool_calls"), 1);
        if (!r.usage.used(at(second, "function")) && !r.usage.used(at(second, "name"))
                && !r.usage.used(at(second, "arguments"))) {
            caps.supportsParallelToolCalls = false;
        }
    }

    /** Preserved reasoning: the reasoning of an earlier assistant turn reaches the output with the aliases on. */
    private void preserveReasoning() {
        Map<String, Object> earlier = message("assistant", "Assistant message");
        putReasoning(earlier, REASONING_PLACEHOLDER);
        Map<String, Object> last = message("assistant", "Assistant message");
        putReasoning(last, "Reasoning content");
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("enable_thinking", true);
        TemplateCaps.applyPreserveReasoning(extra, true);
        Result r = execute(list(message("user", "User message"), earlier, message("user", "User message"), last,
                message("user", "User message")), extra, null);
        if (r.output.contains(REASONING_PLACEHOLDER)) caps.supportsPreserveReasoning = true;
    }

    /**
     * Reasoning effort: supported when the template reads it; then every candidate level that renders differently
     * from an unknown level is kept, and of the levels that render alike the highest stands for them.
     */
    private void reasoningEffort() {
        Result low = effort("low");
        caps.supportsReasoningEffort = low.usage.used(low.usage.variable("reasoning_effort"));
        if (!caps.supportsReasoningEffort) return;
        Result unknownResult = effort(UNKNOWN_EFFORT);
        String unknown = unknownResult.success ? unknownResult.output : null;
        List<String> renders = new ArrayList<>();
        for (String level : EFFORT_CANDIDATES) {
            Result r = effort(level);
            if (!r.success || r.output.equals(unknown)) continue;
            int alias = renders.indexOf(r.output);
            if (alias >= 0) {
                caps.reasoningEffortLevels.set(alias, level);
            } else {
                renders.add(r.output);
                caps.reasoningEffortLevels.add(level);
            }
        }
    }

    private Result effort(String level) {
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("enable_thinking", true);
        TemplateCaps.applyReasoningEffort(extra, s(level));
        return execute(list(message("user", "User message")), extra, null);
    }

    // ------------------------------------------------------------------ probe conversations

    /**
     * llama.cpp's tool conversation: a user turn, an assistant turn with {@code calls} calls of {@code tool1}
     * ({@code {"arg": "value"}} as an object or as JSON text), the tool response, an assistant reply and a user turn.
     */
    private static List<Map<String, Object>> toolConversation(int calls, boolean objectArguments) {
        List<Object> toolCalls = new ArrayList<>();
        for (int c = 1; c <= calls; c++) {
            Object arguments;
            if (objectArguments) {
                Map<String, Object> args = new LinkedHashMap<>();
                args.put("arg", s("value"));
                arguments = args;
            } else {
                arguments = s("{\"arg\": \"value\"}");
            }
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", s("tool1"));
            function.put("arguments", arguments);
            Map<String, Object> call = new LinkedHashMap<>();
            call.put("id", s("call0000" + c));
            call.put("type", s("function"));
            call.put("function", function);
            toolCalls.add(call);
        }
        Map<String, Object> assistant = message("assistant", "");
        assistant.put("tool_calls", toolCalls);
        Map<String, Object> response = message("tool", "Tool response");
        response.put("tool_call_id", s("call00001"));
        return ChatMessages.forTemplate(list(message("user", "User message"), assistant, response,
                message("assistant", "The tool response was 'tool response'"), message("user", "User message")));
    }

    /** llama.cpp's probe tool {@code tool1} with one required string argument, in {@link ChatTool}'s template shape. */
    private static List<Map<String, Object>> tool() {
        Map<String, Object> arg = new LinkedHashMap<>();
        arg.put("type", s("string"));
        arg.put("description", s("Arg description"));
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("arg", arg);
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", s("object"));
        parameters.put("properties", properties);
        parameters.put("required", list(s("arg")));
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", s("tool1"));
        function.put("description", s("Tool description"));
        function.put("parameters", parameters);
        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", s("function"));
        tool.put("function", function);
        return list(tool);
    }

    private static Map<String, Object> message(String role, Object content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put(ChatMessages.ROLE, s(role));
        m.put(ChatMessages.CONTENT, content instanceof String ? s((String) content) : content);
        return m;
    }

    /**
     * A distinct instance of {@code text}: usage is tracked by identity, so every probe string gets its own
     * instance and a query about one input position never sees another position's use.
     */
    @SuppressWarnings("StringOperationCanBeSimplified")
    private static String s(String text) {
        return new String(text);
    }

    @SafeVarargs
    private static <T> List<T> list(T... items) {
        return new ArrayList<>(Arrays.asList(items));
    }

    // ------------------------------------------------------------------ execution

    /**
     * Renders one probe ({@code caps_try_execute}): llama.cpp's probe context with {@code extra} bound over it.
     * A raising render is retried once with an empty {@code reasoning_content} added to the assistant messages
     * that lack one, when there are such messages.
     */
    private Result execute(List<Map<String, Object>> messages, Map<String, Object> extra,
                           List<Map<String, Object>> tools) {
        List<Map<String, Object>> msgs = messages;
        for (int attempt = 0; ; attempt++) {
            Map<String, Object> context = new LinkedHashMap<>();
            context.put("messages", msgs);
            context.put("tools", tools != null ? tools : new ArrayList<Map<String, Object>>());
            context.put("bos_token", "");
            context.put("eos_token", "");
            context.put("add_generation_prompt", true);
            if (extra != null) context.putAll(extra);
            JinjaUsage usage = new JinjaUsage();
            try {
                return new Result(true, template.render(context, usage), usage);
            } catch (RuntimeException e) {
                LOGGER.trace("Chat template capability probe raised: {}", e.getMessage());
                List<Map<String, Object>> retried = attempt == 0 ? withEmptyReasoning(msgs) : null;
                if (retried == null) return new Result(false, "", usage);
                msgs = retried;
            }
        }
    }

    /**
     * {@code messages} with an empty reasoning on every assistant message that lacks one, under both
     * {@code reasoning_content} and {@code reasoning}, or null when every assistant message already carries it.
     */
    private static List<Map<String, Object>> withEmptyReasoning(List<Map<String, Object>> messages) {
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        boolean added = false;
        for (Map<String, Object> m : messages) {
            if (!"assistant".equals(m.get(ChatMessages.ROLE))) {
                out.add(m);
                continue;
            }
            Object reasoning = m.get(ChatMessages.REASONING);
            Object alias = m.get(ChatMessages.REASONING_CONTENT);
            Object text = reasoning instanceof String ? reasoning : alias instanceof String ? alias : s("");
            if (!(reasoning instanceof String) || !(alias instanceof String)) {
                Map<String, Object> copy = new LinkedHashMap<>(m);
                copy.put(ChatMessages.REASONING_CONTENT, text);
                copy.put(ChatMessages.REASONING, text);
                out.add(copy);
                added = true;
            } else {
                out.add(m);
            }
        }
        return added ? out : null;
    }

    /** Sets the message's reasoning text under both vLLM's {@code reasoning} and its deprecated alias. */
    private static void putReasoning(Map<String, Object> message, String text) {
        message.put(ChatMessages.REASONING_CONTENT, s(text));
        message.put(ChatMessages.REASONING, s(text));
    }

    /** {@code value[key]} of a converted list or dict, or Undefined. */
    private static Object at(Object value, Object key) {
        if (value instanceof List && key instanceof Integer) {
            List<?> list = (List<?>) value;
            int i = (Integer) key;
            return i < list.size() ? list.get(i) : JinjaValue.UNDEFINED;
        }
        if (value instanceof Map && ((Map<?, ?>) value).containsKey(key)) return ((Map<?, ?>) value).get(key);
        return JinjaValue.UNDEFINED;
    }

    /** One probe render: whether it succeeded, its output (empty on failure) and its usage statistics. */
    private static final class Result {
        final boolean success;
        final String output;
        final JinjaUsage usage;

        Result(boolean success, String output, JinjaUsage usage) {
            this.success = success;
            this.output = output;
            this.usage = usage;
        }

        /** The converted {@code messages[index][key]} the template saw. */
        Object at(int index, String key) {
            return TemplateCapsProbe.at(TemplateCapsProbe.at(usage.variable("messages"), index), key);
        }
    }
}
