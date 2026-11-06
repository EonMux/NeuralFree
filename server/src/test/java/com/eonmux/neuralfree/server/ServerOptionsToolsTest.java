package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.server.tools.McpServerConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The server tools options ({@code neuralfree.server.tools}, {@code neuralfree.server.mcp.config} and
 * {@code neuralfree.server.mcp.json}): parsing, the file plus inline merge, and refusals naming the property.
 */
class ServerOptionsToolsTest {

    private static final String[] PROPERTIES = {ServerOptions.TOOLS_PROPERTY, ServerOptions.MCP_CONFIG_PROPERTY,
            ServerOptions.MCP_JSON_PROPERTY};

    @TempDir
    Path dir;

    private final Map<String, String> saved = new HashMap<>();

    @BeforeEach
    void save() {
        for (String property : PROPERTIES) {
            saved.put(property, System.getProperty(property));
            System.clearProperty(property);
        }
    }

    @AfterEach
    void restore() {
        saved.forEach((property, value) -> {
            if (value == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, value);
            }
        });
    }

    @Test
    void withoutThePropertiesNoToolServes() {
        ServerOptions options = ServerOptions.fromSystemProperties();
        assertNull(options.tools());
        assertNull(options.mcpServers());
    }

    @Test
    void theToolsListSplitsOnCommas() {
        System.setProperty(ServerOptions.TOOLS_PROPERTY, "read_file, get_info");
        assertEquals(List.of("read_file", "get_info"),
                ServerOptions.fromSystemProperties().tools());
        System.setProperty(ServerOptions.TOOLS_PROPERTY, "all");
        assertEquals(List.of("all"), ServerOptions.fromSystemProperties().tools());
    }

    @Test
    void theMcpConfigFileAndTheInlineJsonMergeInThePropertyOrder() throws IOException {
        Path file = dir.resolve("mcp.json");
        Files.writeString(file, "{\"mcpServers\": {\"one\": {\"command\": \"c1\", \"timeout_ms\": 100}}}");
        System.setProperty(ServerOptions.MCP_CONFIG_PROPERTY, file.toString());
        System.setProperty(ServerOptions.MCP_JSON_PROPERTY,
                "{\"mcpServers\": {\"two\": {\"command\": \"c2\"}}}");
        List<McpServerConfig> servers = ServerOptions.fromSystemProperties().mcpServers();
        assertEquals(2, servers.size());
        assertEquals("one", servers.get(0).name());
        assertEquals(100, servers.get(0).timeoutMs());
        assertEquals("two", servers.get(1).name());
    }

    @Test
    void theFileServerKeepsItsPlaceBesideTheInlineOneOfADifferentName() throws IOException {
        Path file = dir.resolve("mcp.json");
        Files.writeString(file, "{\"mcpServers\": {\"one\": {\"command\": \"c1\"}}}");
        System.setProperty(ServerOptions.MCP_CONFIG_PROPERTY, file.toString());
        System.setProperty(ServerOptions.MCP_JSON_PROPERTY,
                "{\"mcpServers\": {\"one\": {\"command\": \"inline\"}}}");
        List<McpServerConfig> servers = ServerOptions.fromSystemProperties().mcpServers();
        assertEquals(1, servers.size());
        assertEquals("c1", servers.get(0).command(),
                "a name named twice keeps its first entry, as llama.cpp's merge does");
    }

    @Test
    void theRefusalsNameTheProperty() throws IOException {
        System.setProperty(ServerOptions.TOOLS_PROPERTY, " , ");
        IllegalArgumentException tools = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertTrue(tools.getMessage().contains(ServerOptions.TOOLS_PROPERTY));
        System.clearProperty(ServerOptions.TOOLS_PROPERTY);

        System.setProperty(ServerOptions.MCP_CONFIG_PROPERTY, dir.resolve("absent.json").toString());
        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertTrue(missing.getMessage().contains(ServerOptions.MCP_CONFIG_PROPERTY));
        System.clearProperty(ServerOptions.MCP_CONFIG_PROPERTY);

        Path bad = Files.writeString(dir.resolve("bad.json"), "{\"mcpServers\": {\"a\": {\"args\": [4]}}}");
        System.setProperty(ServerOptions.MCP_JSON_PROPERTY, bad.toString());
        IllegalArgumentException malformed = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertTrue(malformed.getMessage().contains(ServerOptions.MCP_JSON_PROPERTY));
    }
}
