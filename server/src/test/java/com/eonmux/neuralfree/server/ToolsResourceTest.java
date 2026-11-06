package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.server.tools.McpServerConfig;
import com.eonmux.neuralfree.server.tools.ToolsRegistry;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code GET /tools} and {@code POST /tools} endpoints over a registry wired directly (llama.cpp's server
 * tools REST surface): the listing, the invocation round trips, the refusals (403 off, 404 unknown and
 * non-streaming, 400 malformed) and the streamed shell command.
 */
public class ToolsResourceTest {

    @TempDir
    static Path dir;

    private static ToolsRegistry registry;
    private static ToolsResource resource;

    @BeforeAll
    static void open() throws Exception {
        Path script = Files.writeString(dir.resolve("mcp.sh"), MCP_PING_SERVER);
        registry = ToolsRegistry.of(
                List.of("read_file", "write_file", "exec_shell_command"),
                List.of(new McpServerConfig("fs", "sh", List.of(script.toString()), Map.of(), null, 5000)));
        resource = new ToolsResource(registry);
    }

    @AfterAll
    static void close() {
        registry.close();
    }

    @Test
    void theListingNamesTheBuiltinsAndTheMcpTools() {
        List<?> listing = (List<?>) ServerTestSupport.parsed(resource.list().getEntity());
        assertEquals(4, listing.size());
        Map<?, ?> builtin = (Map<?, ?>) listing.get(0);
        assertEquals("read_file", builtin.get("tool"));
        assertEquals("server", builtin.get("type"));
        assertTrue(builtin.containsKey("definition") && builtin.containsKey("permissions")
                && builtin.containsKey("uses_cwd"));
        Map<?, ?> mcp = (Map<?, ?>) listing.get(3);
        assertEquals("fs_ping", mcp.get("tool"));
        assertEquals("mcp", mcp.get("type"));
    }

    @Test
    void aCallAnswersTheToolResult() throws Exception {
        Path note = Files.writeString(dir.resolve("note.txt"), "payload\n");
        Response response = resource.invoke("{\"tool\": \"read_file\", \"params\": {\"path\": \""
                + note + "\"}}");
        assertEquals("payload\n", view(response.getEntity()).str("plain_text_response"));
        Response mcp = resource.invoke("{\"tool\": \"fs_ping\", \"params\": {}}");
        assertEquals("PONG", view(mcp.getEntity()).str("plain_text_response"));
    }

    @Test
    void aStreamedShellCommandAnswersChunkFramesAndADoneFrame() throws Exception {
        Response response = resource.invoke("{\"tool\": \"exec_shell_command\", \"stream\": true,"
                + " \"params\": {\"command\": \"echo streamed\"}}");
        String stream = streamText(response);
        assertTrue(stream.contains("{\"chunk\":\"streamed\\n\"}"));
        assertTrue(stream.contains("{\"done\":true}"));
    }

    @Test
    void theRefusalsNameTheProblem() {
        assertEquals(403, statusOf(() -> new ToolsResource(ToolsRegistry.EMPTY).list()));
        assertEquals(403, statusOf(() -> new ToolsResource(ToolsRegistry.EMPTY).invoke("{}")));
        assertEquals(404, statusOf(() -> resource.invoke("{\"tool\": \"no_such\"}")));
        assertTrue(errorOf(() -> resource.invoke("{\"tool\": \"no_such\"}")).contains("no_such"));
        assertTrue(errorOf(() -> resource.invoke("{\"tool\": \"read_file\", \"stream\": true}"))
                .contains("does not support stream"));
        assertEquals(400, statusOf(() -> resource.invoke("{\"params\": {}}")));
        assertEquals(400, statusOf(() -> resource.invoke("{\"tool\": \"read_file\", \"params\": [1]}")));
        assertEquals(400, statusOf(() -> resource.invoke("not json")));
    }

    /** A 403 answer carries the enable message in llama.cpp's error envelope. */
    @Test
    void theDisabledServerNamesTheFlags() {
        String message = errorOf(() -> new ToolsResource(ToolsRegistry.EMPTY).list());
        assertTrue(message.contains("--tools") && message.contains("--mcp-servers-config"));
    }

    private static final String MCP_PING_SERVER = """
            #!/bin/sh
            while IFS= read -r line; do
              case "$line" in
                *'"method":"initialize"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{}}\\n' "$id"
                  ;;
                *'"method":"tools/list"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"ping","description":"Pongs","inputSchema":{"type":"object"}}]}}\\n' "$id"
                  ;;
                *'"method":"tools/call"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"PONG"}]}}\\n' "$id"
                  ;;
              esac
            done
            """;
}
