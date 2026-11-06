package com.eonmux.neuralfree.server.tools;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * The MCP servers of the launch configuration (llama.cpp's {@code server_mcp}): each server is spawned once to
 * discover its tools (bounded by the warmup budget, failures logged and the server skipped), then closed again;
 * a call respawns the session on demand, and a server whose spawn just failed sits out the cooldown before
 * another attempt. The registry answers the definitions {@link ToolsRegistry} appends after the built-ins.
 */
public final class McpManager {

    private static final Logger LOGGER = Logger.getLogger(McpManager.class.getName());

    /** The seconds a server whose session just failed to start waits before the next attempt. */
    static final long COOLDOWN_SECONDS = 5;

    /** The budget one server's tool discovery may spend, so an unresponsive child cannot stall the launch. */
    static final long WARMUP_MS = 10_000;

    private final List<McpServerConfig> configs;
    private final Map<String, McpSession> sessions = new LinkedHashMap<>();
    private final Map<String, Long> deadUntil = new LinkedHashMap<>();
    private final List<McpTool.Def> registry = new ArrayList<>();
    private final AtomicBoolean stopping = new AtomicBoolean();

    /** The manager of {@code configs}, warmed up at once: every server listed, the registry filled. */
    public static McpManager of(List<McpServerConfig> configs) {
        McpManager manager = new McpManager(configs);
        manager.warmup();
        return manager;
    }

    private McpManager(List<McpServerConfig> configs) {
        this.configs = List.copyOf(configs);
    }

    /** Spawns every server once to list its tools, then closes the session again (llama.cpp's warmup). */
    private void warmup() {
        for (McpServerConfig config : configs) {
            McpSession session = new McpSession(config);
            if (!session.start() || !session.alive()) {
                LOGGER.warning("MCP warmup: failed to spawn '" + config.name() + "': " + session.diagnostics());
                session.close();
                continue;
            }
            List<McpTool.Def> tools = session.listTools(WARMUP_MS);
            LOGGER.info("MCP warmup: '" + config.name() + "' discovered " + tools.size() + " tool(s)");
            registry.addAll(tools);
            session.close();
        }
    }

    /** The discovered tools, in config order. */
    public List<McpTool.Def> registry() {
        return List.copyOf(registry);
    }

    /** The config of server {@code name}, or null. */
    private McpServerConfig config(String name) {
        for (McpServerConfig config : configs) {
            if (config.name().equals(name)) {
                return config;
            }
        }
        return null;
    }

    /**
     * Calls the tool {@code tool} of server {@code server} with {@code arguments}: the session's answer, or an
     * {@code {"error": ...}} map when the server is unknown, cooling down or unreachable.
     */
    public Map<String, Object> callTool(String server, String tool, Map<String, Object> arguments) {
        McpSession session = session(server);
        if (session == null) {
            return Map.of("error", "MCP server unavailable: " + server);
        }
        return session.callTool(tool, arguments);
    }

    /**
     * The live session of server {@code name}, (re)spawned on demand: a session that died is closed and
     * respawned from the config; a server whose spawn just failed answers null until its cooldown is over.
     */
    private synchronized McpSession session(String name) {
        if (stopping.get()) {
            return null;
        }
        long now = System.nanoTime();
        Long until = deadUntil.get(name);
        if (until != null) {
            if (now < until) {
                return null;
            }
            deadUntil.remove(name);
        }
        McpSession session = sessions.get(name);
        if (session != null) {
            if (session.alive()) {
                return session;
            }
            LOGGER.warning("MCP '" + name + "' is no longer alive: " + session.diagnostics());
            session.close();
            sessions.remove(name);
        }
        McpServerConfig config = config(name);
        if (config == null) {
            return null;
        }
        McpSession fresh = new McpSession(config);
        if (fresh.start() && fresh.alive()) {
            sessions.put(name, fresh);
            return fresh;
        }
        LOGGER.warning("MCP '" + name + "': failed to start: " + fresh.diagnostics());
        fresh.close();
        deadUntil.put(name, now + TimeUnit.SECONDS.toNanos(COOLDOWN_SECONDS));
        return null;
    }

    /** Closes every live session; further calls answer unavailable. Idempotent. */
    public void close() {
        stopping.set(true);
        List<McpSession> open;
        synchronized (this) {
            open = new ArrayList<>(sessions.values());
            sessions.clear();
        }
        for (McpSession session : open) {
            session.close();
        }
    }
}
