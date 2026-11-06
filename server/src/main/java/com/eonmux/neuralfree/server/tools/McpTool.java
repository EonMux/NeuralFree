package com.eonmux.neuralfree.server.tools;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One tool of an MCP server, exposed under the joined name {@code <server>_<tool>} (llama.cpp's
 * {@code server_mcp_tool}): its listing carries the {@code mcp} type, and a call routes to the manager, which
 * (re)spawns the server's session. The arguments pass through as they arrive, the working directory never
 * applies — the child process runs where its config puts it.
 */
public final class McpTool extends ServerTool {

    /** A tool definition a server advertised ({@code tools/list}): its server's config name first. */
    public record Def(String server, String name, String description, Map<String, Object> inputSchema) { }

    private final String server;
    private final Def def;
    private final McpManager manager;

    /** The tool of server {@code server} over {@code def}, called through {@code manager}. */
    public McpTool(String server, Def def, McpManager manager) {
        super(server + "_" + def.name(), server + "_" + def.name(), false, false);
        this.server = server;
        this.def = def;
        this.manager = manager;
    }

    @Override
    public String type() {
        return "mcp";
    }

    @Override
    public Map<String, Object> definition() {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", name());
        fn.put("description", def.description());
        fn.put("parameters", def.inputSchema() == null ? Map.of() : def.inputSchema());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("type", "function");
        out.put("function", fn);
        return out;
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        return manager.callTool(server, def.name(), params);
    }
}
