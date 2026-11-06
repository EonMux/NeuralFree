package com.eonmux.neuralfree.server.tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * The server-executed tools of one server process (llama.cpp's {@code server_tools}): the enabled built-ins
 * first, then the MCP servers' tools under their joined {@code <server>_<tool>} names, an MCP name colliding
 * with an existing tool skipped with a logged warning. The registry answers the {@code GET /tools} listing,
 * the lookups by name and the invocations the {@code POST /tools} endpoint and the chat tool loop run; a
 * registry with no tool at all marks the feature off ({@code /tools} then answers 403).
 */
public final class ToolsRegistry {

    private static final Logger LOGGER = Logger.getLogger(ToolsRegistry.class.getName());

    /** The registry with no tool: {@code /tools} is refused, no chat request may loop. */
    public static final ToolsRegistry EMPTY = new ToolsRegistry(Map.of(), null);

    /** The built-in tools, by name, enabled with {@code --tools all} or a comma-separated list. */
    private static final Map<String, ServerTool> BUILTINS = builtins();

    private final Map<String, ServerTool> tools;
    private final McpManager mcp;

    private ToolsRegistry(Map<String, ServerTool> tools, McpManager mcp) {
        this.tools = tools;
        this.mcp = mcp;
    }

    /**
     * The registry of the enabled built-in {@code tools} ({@code all} or the names, an unknown one refused
     * naming the tool) and the {@code mcp} server configs, whose tools are discovered at construction (a
     * spawn failure skips that server, logged).
     *
     * @throws IllegalArgumentException when {@code tools} names an unknown built-in
     */
    public static ToolsRegistry of(List<String> tools, List<McpServerConfig> mcp) {
        boolean all = tools != null && tools.stream().anyMatch(t -> t.equals("all"));
        Map<String, ServerTool> enabled = new LinkedHashMap<>();
        if (tools != null && !tools.isEmpty()) {
            if (all) {
                enabled.putAll(BUILTINS);
            }
            for (String name : tools) {
                if (name.equals("all")) {
                    continue;
                }
                ServerTool tool = BUILTINS.get(name);
                if (tool == null) {
                    throw new IllegalArgumentException("unknown tool '" + name + "'. available tools: "
                            + String.join(", ", BUILTINS.keySet()));
                }
                enabled.put(name, tool);
            }
        }
        McpManager manager = mcp == null || mcp.isEmpty() ? null : McpManager.of(mcp);
        if (manager != null) {
            for (McpTool.Def def : manager.registry()) {
                McpTool tool = new McpTool(def.server(), def, manager);
                if (enabled.containsKey(tool.name())) {
                    LOGGER.warning("MCP tool '" + tool.name() + "' from server '" + def.server()
                            + "' collides with an existing tool, skipping");
                    continue;
                }
                enabled.put(tool.name(), tool);
            }
        }
        return new ToolsRegistry(Collections.unmodifiableMap(new LinkedHashMap<>(enabled)), manager);
    }

    /** Whether any tool is enabled ({@code /tools} serves, chat may loop). */
    public boolean enabled() {
        return !tools.isEmpty();
    }

    /** The tool {@code name}, or null. */
    public ServerTool find(String name) {
        return tools.get(name);
    }

    /** The {@code GET /tools} listing: every tool's entry, built-ins before MCP tools. */
    public List<Map<String, Object>> list() {
        List<Map<String, Object>> out = new ArrayList<>();
        for (ServerTool tool : tools.values()) {
            out.add(tool.toMap());
        }
        return out;
    }

    /**
     * Runs the tool {@code name} over the call's {@code params}: a {@code cwd} entry (a string) resolves the
     * call's paths and hosts its command, the rest reaches the tool. Unknown names answer null (the endpoint
     * turns that into a 404); the tool's answer map is the result.
     */
    public Map<String, Object> invoke(String name, Map<String, Object> params) {
        ServerTool tool = tools.get(name);
        if (tool == null) {
            return null;
        }
        String cwd = params.get("cwd") instanceof String directory ? directory : null;
        return tool.invoke(params, new ToolIo(cwd));
    }

    /** Stops the MCP manager's sessions. */
    public void close() {
        if (mcp != null) {
            mcp.close();
        }
    }

    /** The built-in catalog, in llama.cpp's order. */
    private static Map<String, ServerTool> builtins() {
        List<ServerTool> tools = List.of(new ReadFileTool(), new FileGlobSearchTool(), new GrepSearchTool(),
                new ExecShellCommandTool(), new WriteFileTool(), new EditFileTool(), new GetInfoTool());
        Map<String, ServerTool> out = new LinkedHashMap<>();
        for (ServerTool tool : tools) {
            out.put(tool.name(), tool);
        }
        return Collections.unmodifiableMap(out);
    }

    /** Whether {@code name} names a built-in tool. */
    public static boolean isBuiltin(String name) {
        return BUILTINS.containsKey(name);
    }

    /** The built-in tool names. */
    public static List<String> builtinNames() {
        return List.copyOf(BUILTINS.keySet());
    }
}
