package com.eonmux.neuralfree.server.tools;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The MCP manager: the warmup lists every server's tools into the registry, a call respawns the session the
 * warmup closed, and a server that cannot spawn answers unavailable.
 */
public class McpManagerTest {

    private static final String SERVER = """
            #!/bin/sh
            while IFS= read -r line; do
              case "$line" in
                *'"method":"initialize"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{}}\\n' "$id"
                  ;;
                *'"method":"tools/list"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"ping","description":"Pings back","inputSchema":{"type":"object"}}]}}\\n' "$id"
                  ;;
                *'"method":"tools/call"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"PONG"}]}}\\n' "$id"
                  ;;
              esac
            done
            """;

    @TempDir
    Path dir;

    /** The config of the fake server script under {@code name}, the script's path back. */
    private McpServerConfig config(String name) throws Exception {
        Path file = dir.resolve(name + ".sh");
        Files.writeString(file, SERVER);
        return new McpServerConfig(name, "sh", List.of(file.toString()), Map.of(), null, 5000);
    }

    @Test
    void theWarmupFillsTheRegistryAndCallsRespawnTheSession() throws Exception {
        McpManager manager = McpManager.of(List.of(config("alive")));
        List<McpTool.Def> registry = manager.registry();
        assertEquals(1, registry.size());
        assertEquals("alive", registry.get(0).server());
        assertEquals("ping", registry.get(0).name());
        assertEquals(Map.of("plain_text_response", "PONG"),
                manager.callTool("alive", "ping", Map.of()));
        manager.close();
    }

    @Test
    void aServerThatCannotSpawnAnswersUnavailable() throws Exception {
        McpManager manager = McpManager.of(List.of(new McpServerConfig("dead", "no-such-mcp-command",
                List.of(), Map.of(), null, 5000)));
        assertEquals(0, manager.registry().size(), "the spawn failure is logged, the launch carries on");
        assertEquals(Map.of("error", "MCP server unavailable: dead"),
                manager.callTool("dead", "ping", Map.of()));
        assertEquals(Map.of("error", "MCP server unavailable: never-configured"),
                manager.callTool("never-configured", "ping", Map.of()));
        manager.close();
        manager.close();
    }
}
