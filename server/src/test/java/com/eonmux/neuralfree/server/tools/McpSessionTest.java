package com.eonmux.neuralfree.server.tools;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * One MCP session over its stdio NDJSON transport, against a fake server script that answers the JSON-RPC
 * handshake, {@code tools/list} (with an unsolicited notification before the reply, which the session must
 * skip) and {@code tools/call}.
 */
public class McpSessionTest {

    private static final String SERVER = """
            #!/bin/sh
            # A minimal MCP server: one JSON-RPC reply per request line.
            while IFS= read -r line; do
              case "$line" in
                *'"method":"initialize"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"1"}}}\\n' "$id"
                  ;;
                *'"method":"tools/list"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","method":"notifications/progress","params":{}}\\n'
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echo a fixed text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}}\\n' "$id"
                  ;;
                *'"method":"tools/call"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  case "$line" in
                    *'"fail"'*)
                      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"boom"}],"isError":true}}\\n' "$id"
                      ;;
                    *)
                      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"ECHO_OK"}]}}\\n' "$id"
                      ;;
                  esac
                  ;;
              esac
            done
            """;

    private static final String SILENT_CALL = """
            #!/bin/sh
            # Answers the handshake, then never answers a call.
            while IFS= read -r line; do
              case "$line" in
                *'"method":"initialize"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{}}\\n' "$id"
                  ;;
                *'"method":"tools/list"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[]}}\\n' "$id"
                  ;;
              esac
            done
            """;

    @TempDir
    Path dir;

    private McpServerConfig config(String script) throws Exception {
        Path file = dir.resolve("mcp-" + script.hashCode() + ".sh");
        Files.writeString(file, script);
        return new McpServerConfig("fake", "sh", List.of(file.toString()), Map.of(), null, 5000);
    }

    @Test
    void theSessionListsTheToolsOfTheHandshakedServer() throws Exception {
        McpSession session = new McpSession(config(SERVER));
        assertTrue(session.start());
        assertTrue(session.alive());
        List<McpTool.Def> tools = session.listTools(5000);
        assertEquals(1, tools.size());
        assertEquals("fake", tools.get(0).server());
        assertEquals("echo", tools.get(0).name());
        assertEquals("Echo a fixed text", tools.get(0).description());
        assertEquals("object", tools.get(0).inputSchema().get("type"));
        session.close();
        assertFalse(session.alive());
    }

    @Test
    void aCallCarriesTheTextContentBackAndSurfacesIsError() throws Exception {
        McpSession session = new McpSession(config(SERVER));
        session.start();
        assertEquals(Map.of("plain_text_response", "ECHO_OK"),
                session.callTool("echo", Map.of("text", "hi")));
        assertEquals(Map.of("error", "boom"), session.callTool("echo", Map.of("text", "fail")));
        session.close();
    }

    @Test
    void aCallPastTheBudgetAnswersTheTimeoutError() throws Exception {
        McpSession session = new McpSession(new McpServerConfig("fake", "sh",
                List.of(script(SILENT_CALL, "silent.sh")), Map.of(), null, 300));
        session.start();
        Map<String, Object> answer = session.callTool("echo", Map.of("text", "hi"));
        assertEquals("request timed out", answer.get("error"));
        session.close();
    }

    @Test
    void aCommandThatCannotSpawnAnswersTheFailureInDiagnostics() throws Exception {
        McpSession session = new McpSession(new McpServerConfig("fake", "no-such-command-xyz",
                List.of(), Map.of(), null, 5000));
        assertFalse(session.start());
        assertTrue(session.diagnostics().contains("no-such-command-xyz"));
        session.close();
    }

    /** The script of {@code text} written under {@code name}, its path back. */
    private String script(String text, String name) throws Exception {
        Path file = dir.resolve(name);
        Files.writeString(file, text);
        return file.toString();
    }
}
