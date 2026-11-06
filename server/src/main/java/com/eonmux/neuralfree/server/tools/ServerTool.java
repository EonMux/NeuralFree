package com.eonmux.neuralfree.server.tools;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One server-executed tool (llama.cpp's {@code server_tool}): a fixed name and display name, a write permission,
 * whether it resolves paths against a working directory, the OpenAI function {@link #definition} the model calls
 * it by, and the {@link #invoke} the {@code POST /tools} endpoint and the chat tool loop run with the call's
 * arguments. A result is a JSON object map: {@code {"plain_text_response": ...}} for text the model reads,
 * {@code {"error": ...}} for a failure the model may recover from. A malformed argument list (a missing or
 * mistyped required field) throws {@link IllegalArgumentException}; the caller maps that to a 400 or feeds the
 * error text back to the model.
 */
public abstract class ServerTool {

    private final String name;
    private final String displayName;
    private final boolean write;
    private final boolean usesCwd;

    /** The tool's fixed settings; subclasses carry no mutable state. */
    protected ServerTool(String name, String displayName, boolean write, boolean usesCwd) {
        this.name = name;
        this.displayName = displayName;
        this.write = write;
        this.usesCwd = usesCwd;
    }

    /** The name the model calls the tool by and the registry lists it under. */
    public final String name() {
        return name;
    }

    /** The human-facing name the {@code GET /tools} entry carries. */
    public final String displayName() {
        return displayName;
    }

    /** Whether the tool writes (files, commands); the {@code permissions.write} of its listing. */
    public final boolean writePermission() {
        return write;
    }

    /** Whether the tool resolves its paths against the working directory. */
    public final boolean usesCwd() {
        return usesCwd;
    }

    /** The kind of tool: {@code server} for a built-in, {@code mcp} for one from an MCP server. */
    public String type() {
        return "server";
    }

    /** Whether {@code POST /tools} may run this tool with {@code stream: true}. */
    public boolean supportsStream() {
        return false;
    }

    /**
     * The streaming variant of {@link #invoke}: each output piece reaches {@code push} as it is produced, the
     * caller writes the SSE frames. Only a tool with {@link #supportsStream()} is asked; the default refuses
     * naming the tool.
     */
    public void invokeStream(Map<String, Object> params, ToolIo io, java.util.function.Consumer<String> push) {
        throw new IllegalArgumentException("tool \"" + name() + "\" does not support stream = true");
    }

    /** The OpenAI function definition the model sees: {@code {type: function, function: {name, ...}}}. */
    public abstract Map<String, Object> definition();

    /** The tool's answer for {@code params}, the call's arguments, run over {@code io}. */
    public abstract Map<String, Object> invoke(Map<String, Object> params, ToolIo io);

    /** The {@code GET /tools} entry of the tool (llama.cpp's {@code server_tool::to_json}). */
    public final Map<String, Object> toMap() {
        Map<String, Object> permissions = new LinkedHashMap<>();
        permissions.put("write", write);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("display_name", displayName);
        out.put("tool", name);
        out.put("type", type());
        out.put("permissions", permissions);
        out.put("uses_cwd", usesCwd);
        out.put("definition", definition());
        return out;
    }

    // ------------------------------------------------------------------ argument helpers

    /** The OpenAI function shape the definitions build on: type, function name, description, parameters. */
    protected static Map<String, Object> functionDefinition(String name, String description,
                                                            Map<String, Object> parameters) {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", name);
        fn.put("description", description);
        fn.put("parameters", parameters);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("type", "function");
        out.put("function", fn);
        return out;
    }

    /** A JSON-schema object of {@code type: object} over the named {@code properties} and the required fields. */
    protected static Map<String, Object> parameters(Map<String, Object> properties, String... required) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("type", "object");
        out.put("properties", properties);
        out.put("required", List.of(required));
        return out;
    }

    /** The string value of {@code key}, refused when it is missing or not a string. */
    protected static String string(Map<String, Object> params, String key) {
        Object value = params.get(key);
        if (!(value instanceof String s)) {
            throw new IllegalArgumentException(key + " is required and must be a string");
        }
        return s;
    }

    /** The optional string value of {@code key}, or {@code fallback} when it is absent or null. */
    protected static String optString(Map<String, Object> params, String key, String fallback) {
        Object value = params.get(key);
        return value instanceof String s && !s.isEmpty() ? s : fallback;
    }

    /** The optional integer value of {@code key}, or {@code fallback} when absent or null. */
    protected static int optInt(Map<String, Object> params, String key, int fallback) {
        return params.get(key) instanceof Number n ? n.intValue() : fallback;
    }

    /** The optional boolean value of {@code key}, or {@code fallback} when absent or null. */
    protected static boolean optBool(Map<String, Object> params, String key, boolean fallback) {
        return params.get(key) instanceof Boolean b ? b : fallback;
    }

    /** The optional list value of {@code key}, or null when absent; refused when it is not a list. */
    @SuppressWarnings("unchecked")
    protected static List<Object> optList(Map<String, Object> params, String key) {
        Object value = params.get(key);
        if (value == null) {
            return null;
        }
        if (!(value instanceof List<?> list)) {
            throw new IllegalArgumentException(key + " must be an array");
        }
        return (List<Object>) list;
    }

    /** An answer holding only {@code plain_text_response}. */
    protected static Map<String, Object> text(String plain) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("plain_text_response", plain);
        return out;
    }

    /** An answer holding only {@code error}. */
    protected static Map<String, Object> error(String message) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("error", message);
        return out;
    }
}
