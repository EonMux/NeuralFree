package com.eonmux.neuralfree.server.tools;

import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * One MCP server of the launch configuration (llama.cpp's {@code server_mcp_server_config}, the Cursor-compatible
 * {@code mcpServers} JSON): the config key names the server, and the entry carries the child {@code command} with
 * its {@code args}, the {@code env} merged over the parent environment, the working directory and the per-call
 * {@code timeout_ms}. Only the stdio transport runs in this build: the server is a child process reading one
 * JSON-RPC message per line on stdin and writing its replies on stdout.
 *
 * @param name      the config key the tool names are prefixed with
 * @param command   the executable, never empty
 * @param args      the command arguments
 * @param env       the environment entries merged over the parent environment
 * @param cwd       the child's working directory, or null for the server process one
 * @param timeoutMs the per-call budget, at least 1
 */
public record McpServerConfig(String name, String command, List<String> args, Map<String, String> env,
                              String cwd, int timeoutMs) {

    private static final Logger LOGGER = Logger.getLogger(McpServerConfig.class.getName());

    /** The per-call budget when the config leaves it out (llama.cpp's default). */
    public static final int DEFAULT_TIMEOUT_MS = 30_000;

    /**
     * The servers of a {@code mcpServers} JSON document (a file's text or an inline one), skipping entries
     * without a command and duplicate names with a logged warning.
     *
     * @throws JsonException            when the text does not parse
     * @throws IllegalArgumentException when the document is not an object, an entry is not an object, or an
     *                                  args, env or timeout value has the wrong type
     */
    public static List<McpServerConfig> parse(String json, String source) {
        Object parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        if (!(parsed instanceof Map<?, ?> document)) {
            throw new IllegalArgumentException(source + " must hold a JSON object with an mcpServers entry");
        }
        Object servers = document.get("mcpServers");
        if (!(servers instanceof Map<?, ?> entries)) {
            return List.of();
        }
        List<McpServerConfig> out = new ArrayList<>();
        for (Map.Entry<?, ?> entry : entries.entrySet()) {
            String name = String.valueOf(entry.getKey());
            if (!(entry.getValue() instanceof Map<?, ?> raw)) {
                throw new IllegalArgumentException(source + ": server '" + name + "' must be an object");
            }
            McpServerConfig config = entry(name, raw);
            if (config == null) {
                LOGGER.warning("MCP config: server '" + name + "' has no command, skipping");
                continue;
            }
            out.add(config);
        }
        return List.copyOf(out);
    }

    /** The entry as a config, or null when it carries no command. */
    private static McpServerConfig entry(String name, Map<?, ?> raw) {
        Object command = raw.get("command");
        if (!(command instanceof String executable) || executable.isBlank()) {
            return null;
        }
        List<String> args = new ArrayList<>();
        if (raw.get("args") instanceof List<?> list) {
            for (int i = 0; i < list.size(); i++) {
                if (!(list.get(i) instanceof String arg)) {
                    throw new IllegalArgumentException("MCP config: server '" + name + "' args[" + i
                            + "] must be a string");
                }
                args.add(arg);
            }
        } else if (raw.get("args") != null) {
            throw new IllegalArgumentException("MCP config: server '" + name + "' args must be an array");
        }
        Map<String, String> env = new LinkedHashMap<>();
        if (raw.get("env") instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (!(e.getValue() instanceof String value)) {
                    throw new IllegalArgumentException("MCP config: server '" + name + "' env." + e.getKey()
                            + " must be a string");
                }
                env.put(String.valueOf(e.getKey()), value);
            }
        } else if (raw.get("env") != null) {
            throw new IllegalArgumentException("MCP config: server '" + name + "' env must be an object");
        }
        String cwd = raw.get("cwd") instanceof String directory && !directory.isBlank() ? directory : null;
        int timeout = raw.get("timeout_ms") instanceof Number seconds ? seconds.intValue() : DEFAULT_TIMEOUT_MS;
        if (timeout < 1) {
            throw new IllegalArgumentException("MCP config: server '" + name + "' timeout_ms must be 1 or more,"
                    + " got " + timeout);
        }
        return new McpServerConfig(name, executable, List.copyOf(args), Map.copyOf(env), cwd, timeout);
    }

    /** The compact JSON document of {@code servers} ({@code {"mcpServers": {...}}}), for the tests. */
    public static String toJson(List<McpServerConfig> servers) {
        Map<String, Object> entries = new LinkedHashMap<>();
        for (McpServerConfig server : servers) {
            Map<String, Object> raw = new LinkedHashMap<>();
            raw.put("command", server.command());
            raw.put("args", server.args());
            raw.put("env", server.env());
            if (server.cwd() != null) {
                raw.put("cwd", server.cwd());
            }
            raw.put("timeout_ms", server.timeoutMs());
            entries.put(server.name(), raw);
        }
        Map<String, Object> document = new LinkedHashMap<>();
        document.put("mcpServers", entries);
        StringBuilder out = new StringBuilder();
        try {
            Json.write(document, Json.Style.API, out);
        } catch (java.io.IOException e) {
            throw new IllegalStateException("an MCP config does not write as JSON", e);
        }
        return out.toString();
    }
}
