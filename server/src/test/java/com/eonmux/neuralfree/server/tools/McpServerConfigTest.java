package com.eonmux.neuralfree.server.tools;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The {@code mcpServers} document parsing (the Cursor-compatible launch config). */
public class McpServerConfigTest {

    private static final String DOC = """
            {"mcpServers": {
               "filesystem": {"command": "mcp-server-fs", "args": ["--root", "/tmp"],
                              "env": {"DEBUG": "1"}, "cwd": "/work", "timeout_ms": 5000},
               "broken": {},
               "script": {"command": "sh", "args": ["server.sh"]}
             }}""";

    @Test
    void theDocumentParsesIntoTheServersInOrder() {
        List<McpServerConfig> servers = McpServerConfig.parse(DOC, "mcp.json");
        assertEquals(2, servers.size(), "the command-less entry is skipped");
        McpServerConfig first = servers.get(0);
        assertEquals("filesystem", first.name());
        assertEquals("mcp-server-fs", first.command());
        assertEquals(List.of("--root", "/tmp"), first.args());
        assertEquals("1", first.env().get("DEBUG"));
        assertEquals("/work", first.cwd());
        assertEquals(5000, first.timeoutMs());
        McpServerConfig second = servers.get(1);
        assertEquals("script", second.name());
        assertEquals(30000, second.timeoutMs(), "the default budget when the entry leaves it out");
        assertNull(second.cwd());
    }

    @Test
    void aDocumentWithoutServersAnswersEmpty() {
        assertTrue(McpServerConfig.parse("{\"other\": 1}", "mcp.json").isEmpty());
        assertTrue(McpServerConfig.parse("{}", "mcp.json").isEmpty());
    }

    @Test
    void malformedEntriesRefuseNamingTheServer() {
        IllegalArgumentException args = assertThrows(IllegalArgumentException.class,
                () -> McpServerConfig.parse(
                        "{\"mcpServers\": {\"a\": {\"command\": \"c\", \"args\": [1]}}}", "mcp.json"));
        assertTrue(args.getMessage().contains("'a'") && args.getMessage().contains("args"));
        IllegalArgumentException env = assertThrows(IllegalArgumentException.class,
                () -> McpServerConfig.parse(
                        "{\"mcpServers\": {\"a\": {\"command\": \"c\", \"env\": {\"K\": 2}}}}", "mcp.json"));
        assertTrue(env.getMessage().contains("'a'") && env.getMessage().contains("env"));
        IllegalArgumentException timeout = assertThrows(IllegalArgumentException.class,
                () -> McpServerConfig.parse(
                        "{\"mcpServers\": {\"a\": {\"command\": \"c\", \"timeout_ms\": 0}}}", "mcp.json"));
        assertTrue(timeout.getMessage().contains("timeout_ms"));
        assertThrows(RuntimeException.class,
                () -> McpServerConfig.parse("{not json", "mcp.json"));
        assertThrows(IllegalArgumentException.class,
                () -> McpServerConfig.parse("\"a string\"", "mcp.json"), "the document must be an object");
        assertThrows(IllegalArgumentException.class,
                () -> McpServerConfig.parse("{\"mcpServers\": {\"a\": \"not an object\"}}", "mcp.json"));
    }

    @Test
    void theDocumentRoundTripsThroughToJson() {
        List<McpServerConfig> servers = McpServerConfig.parse(DOC.replace("  \"broken\": {},", ""), "mcp.json");
        List<McpServerConfig> reparsed = McpServerConfig.parse(McpServerConfig.toJson(servers), "inline");
        assertEquals(servers, reparsed);
    }
}
