package com.eonmux.neuralfree.server.tools;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The built-in server tools end to end over real files and processes (llama.cpp's server tools, the shapes of
 * their answers included), plus the registry rules: the enabled set, the unknown-name refusal, and the MCP
 * tools appended under their joined names.
 */
public class ToolsRegistryTest {

    @TempDir
    Path dir;

    // ------------------------------------------------------------------ registry

    @Test
    void allEnablesEveryBuiltInInTheLlamaCppOrder() {
        ToolsRegistry registry = ToolsRegistry.of(List.of("all"), null);
        List<String> names = registry.list().stream().map(entry -> String.valueOf(entry.get("tool"))).toList();
        assertEquals(List.of("read_file", "file_glob_search", "grep_search", "exec_shell_command",
                "write_file", "edit_file", "get_info"), names);
        for (Map<String, Object> entry : registry.list()) {
            assertEquals("server", entry.get("type"));
            assertNotNull(entry.get("display_name"));
            assertNotNull(entry.get("permissions"));
            assertNotNull(entry.get("uses_cwd"));
            @SuppressWarnings("unchecked")
            Map<String, Object> definition = (Map<String, Object>) entry.get("definition");
            assertEquals("function", definition.get("type"));
            @SuppressWarnings("unchecked")
            Map<String, Object> fn = (Map<String, Object>) definition.get("function");
            assertEquals(entry.get("tool"), fn.get("name"));
        }
        assertTrue(registry.enabled());
    }

    @Test
    void aNamedSetEnablesOnlyThoseToolsAndAnUnknownNameRefuses() {
        ToolsRegistry registry = ToolsRegistry.of(List.of("read_file", "get_info"), null);
        assertEquals(2, registry.list().size());
        assertNotNull(registry.find("read_file"));
        assertNull(registry.find("grep_search"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ToolsRegistry.of(List.of("no_such_tool"), null));
        assertTrue(e.getMessage().contains("no_such_tool"));
        assertTrue(e.getMessage().contains("read_file"), "the message names the available tools");
    }

    @Test
    void anEmptyRegistryIsDisabled() {
        assertFalse(ToolsRegistry.EMPTY.enabled());
        assertEquals(List.of(), ToolsRegistry.EMPTY.list());
        assertNull(ToolsRegistry.EMPTY.invoke("read_file", Map.of()));
    }

    // ------------------------------------------------------------------ read_file

    @Test
    void readFileAnswersTheLinesWithAnOptionalRangeAndNumbers() throws Exception {
        Path file = Files.writeString(dir.resolve("note.txt"), "l1\nl2\nl3\n");
        ToolsRegistry registry = ToolsRegistry.of(List.of("read_file"), null);
        Map<String, Object> whole = registry.invoke("read_file", Map.of("path", file.toString()));
        assertEquals("l1\nl2\nl3\n", whole.get("plain_text_response"));
        Map<String, Object> ranged = registry.invoke("read_file",
                Map.of("path", file.toString(), "start_line", 2, "end_line", 3));
        assertEquals("l2\nl3\n", ranged.get("plain_text_response"));
        Map<String, Object> numbered = registry.invoke("read_file",
                Map.of("path", file.toString(), "append_loc", true));
        assertEquals("1→l1\n2→l2\n3→l3\n", numbered.get("plain_text_response"));
    }

    @Test
    void readFileRefusesAMissingFileAndAnOverrunWholeRead() throws Exception {
        Files.writeString(dir.resolve("big.txt"), "x".repeat(ToolsRegistryTest.kb(20)));
        ToolsRegistry registry = ToolsRegistry.of(List.of("read_file"), null);
        assertEquals(Map.of("error", "cannot stat file: " + dir.resolve("absent.txt")),
                registry.invoke("read_file", Map.of("path", dir.resolve("absent.txt").toString())));
        Map<String, Object> big = registry.invoke("read_file", Map.of("path", dir.resolve("big.txt").toString()));
        assertTrue(String.valueOf(big.get("error")).contains("file too large"));
        Map<String, Object> ranged = registry.invoke("read_file",
                Map.of("path", dir.resolve("big.txt").toString(), "end_line", 1));
        assertFalse(String.valueOf(ranged.get("plain_text_response")).isEmpty());
    }

    // ------------------------------------------------------------------ write_file, edit_file

    @Test
    void writeFileCreatesTheFileWithItsParents() {
        ToolsRegistry registry = ToolsRegistry.of(List.of("write_file"), null);
        Path file = dir.resolve("made/inner/out.txt");
        Map<String, Object> result = registry.invoke("write_file",
                Map.of("path", file.toString(), "content", "hello"));
        assertEquals("file written successfully", result.get("result"));
        assertEquals(file.toString(), result.get("path"));
        assertEquals(5, result.get("bytes"));
        assertEquals("hello", read(file));
    }

    @Test
    void editFileAppliesExactReplacementsAndRefusesTheBrokenOnes() throws Exception {
        Path file = Files.writeString(dir.resolve("code.txt"), "alpha\nbeta\ngamma\n");
        ToolsRegistry registry = ToolsRegistry.of(List.of("edit_file"), null);
        Map<String, Object> edits = registry.invoke("edit_file", Map.of("path", file.toString(),
                "edits", List.of(Map.of("old_text", "beta", "new_text", "BETA"),
                        Map.of("old_text", "gamma", "new_text", "GAMMA"))));
        assertEquals("file edited successfully", edits.get("result"));
        assertEquals(2, edits.get("edits_applied"));
        assertEquals("alpha\nBETA\nGAMMA\n", read(file));
        assertEquals(Map.of("error", "\"edits\" must be a non-empty array"),
                registry.invoke("edit_file", Map.of("path", file.toString(), "edits", List.of())));
        assertTrue(String.valueOf(registry.invoke("edit_file", Map.of("path", file.toString(),
                "edits", List.of(Map.of("old_text", "absent", "new_text", "x")))).get("error"))
                .contains("must match the file's current content"));
        assertTrue(String.valueOf(registry.invoke("edit_file", Map.of("path", file.toString(),
                "edits", List.of(Map.of("old_text", "alpha", "new_text", "alpha")))).get("error"))
                .contains("no changes made"));
    }

    @Test
    void editFileRefusesADuplicateAndOverlappingOldText() throws Exception {
        Path file = Files.writeString(dir.resolve("twice.txt"), "same\nsame\n");
        ToolsRegistry registry = ToolsRegistry.of(List.of("edit_file"), null);
        assertTrue(String.valueOf(registry.invoke("edit_file", Map.of("path", file.toString(),
                "edits", List.of(Map.of("old_text", "same", "new_text", "x")))).get("error"))
                .contains("must be unique"));
        Path code = Files.writeString(dir.resolve("code.txt"), "alpha beta\n");
        assertTrue(String.valueOf(registry.invoke("edit_file", Map.of("path", code.toString(),
                "edits", List.of(Map.of("old_text", "alpha bet", "new_text", "X"),
                        Map.of("old_text", "alpha", "new_text", "Y")))).get("error"))
                .contains("overlap"));
    }

    // ------------------------------------------------------------------ file_glob_search

    @Test
    void globSearchListsTheMatchingFilesSkippingTheJunkDirectories() throws Exception {
        Files.createDirectories(dir.resolve("src"));
        Files.createDirectories(dir.resolve("node_modules"));
        Files.writeString(dir.resolve("a.cpp"), "x");
        Files.writeString(dir.resolve("src/b.cpp"), "x");
        Files.writeString(dir.resolve("node_modules/c.cpp"), "x");
        ToolsRegistry registry = ToolsRegistry.of(List.of("file_glob_search"), null);
        Map<String, Object> search = registry.invoke("file_glob_search",
                Map.of("path", dir.toString(), "include", "*.cpp"));
        assertTrue(String.valueOf(search.get("plain_text_response"))
                .contains("---\nTotal matches: 2\n"));
        assertEquals(dir.toString(), search.get("base"));
        assertEquals(Set.of("a.cpp", "src/b.cpp"), pathsOf(search));
        Map<String, Object> excluded = registry.invoke("file_glob_search",
                Map.of("path", dir.toString(), "include", "*.cpp", "exclude", "b.cpp"));
        assertEquals(Set.of("a.cpp"), pathsOf(excluded));
        Map<String, Object> deep = registry.invoke("file_glob_search",
                Map.of("path", dir.toString(), "include", "*.cpp", "max_depth", 1));
        assertEquals(Set.of("a.cpp"), pathsOf(deep));
    }

    /** The entry paths of a search answer. */
    private static Set<String> pathsOf(Map<String, Object> search) {
        Set<String> out = new java.util.HashSet<>();
        for (Object entry : (List<?>) search.get("entries")) {
            out.add(String.valueOf(((Map<?, ?>) entry).get("path")));
        }
        return out;
    }

    @Test
    void globSearchHonorsTheEntryKindAndLimitAndRefusesTheInvalidOnes() throws Exception {
        Files.createDirectories(dir.resolve("sub"));
        Files.writeString(dir.resolve("sub/one.txt"), "x");
        Files.writeString(dir.resolve("two.txt"), "x");
        ToolsRegistry registry = ToolsRegistry.of(List.of("file_glob_search"), null);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> dirs = (List<Map<String, Object>>) registry.invoke("file_glob_search",
                Map.of("path", dir.toString(), "type", "dir")).get("entries");
        assertEquals("sub/", dirs.get(0).get("path"));
        Map<String, Object> limited = registry.invoke("file_glob_search",
                Map.of("path", dir.toString(), "limit", 1));
        assertTrue(String.valueOf(limited.get("plain_text_response")).contains("results limit reached"));
        assertTrue(String.valueOf(registry.invoke("file_glob_search",
                Map.of("path", dir.toString(), "limit", 0)).get("error")).contains("invalid limit"));
        assertTrue(String.valueOf(registry.invoke("file_glob_search",
                Map.of("path", dir.toString(), "type", "symlink")).get("error")).contains("invalid type"));
    }

    // ------------------------------------------------------------------ grep_search

    @Test
    void grepSearchAnswersTheMatchingLinesWithTheRequestedShape() throws Exception {
        Path file = Files.writeString(dir.resolve("log.txt"), "Alpha one\nbeta two\nAlpha three\na.c\nabc\n");
        ToolsRegistry registry = ToolsRegistry.of(List.of("grep_search"), null);
        Map<String, Object> plain = registry.invoke("grep_search",
                Map.of("path", file.toString(), "pattern", "Alpha"));
        assertEquals(file.toString() + ":Alpha one\n" + file.toString() + ":Alpha three\n"
                + "\n---\nTotal matches: 2\n", plain.get("plain_text_response"));
        Map<String, Object> numbered = registry.invoke("grep_search",
                Map.of("path", file.toString(), "pattern", "beta", "return_line_numbers", true));
        assertEquals(file.toString() + ":2:beta two\n\n---\nTotal matches: 1\n",
                numbered.get("plain_text_response"));
        Map<String, Object> folded = registry.invoke("grep_search",
                Map.of("path", file.toString(), "pattern", "alpha", "ignore_case", true));
        assertEquals(2, countMatches(String.valueOf(folded.get("plain_text_response"))),
                "the case-folded search finds the two capitalized lines");
        Map<String, Object> cased = registry.invoke("grep_search",
                Map.of("path", file.toString(), "pattern", "alpha"));
        assertEquals(0, countMatches(String.valueOf(cased.get("plain_text_response"))),
                "the case-sensitive search finds none");
        Map<String, Object> literal = registry.invoke("grep_search",
                Map.of("path", file.toString(), "pattern", "a.c", "literal", true));
        assertEquals(1, countMatches(String.valueOf(literal.get("plain_text_response"))),
                "the dot stays literal");
        Map<String, Object> regex = registry.invoke("grep_search",
                Map.of("path", file.toString(), "pattern", "a.c"));
        assertEquals(2, countMatches(String.valueOf(regex.get("plain_text_response"))),
                "the dot matches any character");
    }

    @Test
    void grepSearchAddsContextAndCapsTheMatches() throws Exception {
        StringBuilder lines = new StringBuilder();
        for (int i = 0; i < 120; i++) {
            lines.append("hit ").append(i).append('\n');
        }
        Path file = Files.writeString(dir.resolve("many.txt"), lines.toString());
        ToolsRegistry registry = ToolsRegistry.of(List.of("grep_search"), null);
        Map<String, Object> capped = registry.invoke("grep_search",
                Map.of("path", file.toString(), "pattern", "hit"));
        String text = String.valueOf(capped.get("plain_text_response"));
        assertEquals(100, countMatches(text));
        assertTrue(text.contains("matches limit reached"));
        Path small = Files.writeString(dir.resolve("small.txt"), "one\ntwo\nthree\n");
        Map<String, Object> context = registry.invoke("grep_search",
                Map.of("path", small.toString(), "pattern", "two", "context_lines", 1,
                        "return_line_numbers", true));
        assertEquals(small + "-1-one\n" + small + ":2:two\n" + small + "-3-three\n--\n"
                + "\n---\nTotal matches: 1\n", context.get("plain_text_response"));
        assertTrue(String.valueOf(registry.invoke("grep_search",
                Map.of("path", small.toString(), "pattern", "(")).get("error")).contains("invalid regex"));
        assertTrue(String.valueOf(registry.invoke("grep_search",
                Map.of("path", dir.resolve("absent.txt").toString(), "pattern", "x")).get("error"))
                .contains("path does not exist"));
    }

    // ------------------------------------------------------------------ exec_shell_command

    @Test
    void shellCommandAnswersItsOutputAndExitCode() {
        ToolsRegistry registry = ToolsRegistry.of(List.of("exec_shell_command"), null);
        Map<String, Object> hello = registry.invoke("exec_shell_command", Map.of("command", "echo hello"));
        assertEquals("hello\n\n[exit code: 0]", hello.get("plain_text_response"),
                "the raw output (its newline kept) before the exit-code line, as llama.cpp writes it");
        Map<String, Object> failed = registry.invoke("exec_shell_command",
                Map.of("command", "echo boom >&2; exit 3"));
        assertEquals("boom\n\n[exit code: 3]", failed.get("plain_text_response"),
                "stdout and stderr combine");
    }

    @Test
    void shellCommandTruncatesTheOutputAndKillsATimeout() {
        ToolsRegistry registry = ToolsRegistry.of(List.of("exec_shell_command"), null);
        Map<String, Object> truncated = registry.invoke("exec_shell_command",
                Map.of("command", "printf 'aaaaaaaaaaaaaaaaaaaa'", "max_output_size", 10));
        assertTrue(String.valueOf(truncated.get("plain_text_response")).contains("[output truncated]"));
        Map<String, Object> timedOut = registry.invoke("exec_shell_command",
                Map.of("command", "sleep 10", "timeout", 1));
        String text = String.valueOf(timedOut.get("plain_text_response"));
        assertTrue(text.contains("[exit due to timed out]"));
    }

    @Test
    void theShellCommandStreamsItsChunks() {
        ExecShellCommandTool tool = new ExecShellCommandTool();
        assertTrue(tool.supportsStream());
        StringBuilder streamed = new StringBuilder();
        tool.invokeStream(Map.of("command", "echo streamed"), new ToolIo(dir.toString()), streamed::append);
        assertTrue(streamed.toString().startsWith("streamed"));
        assertTrue(streamed.toString().endsWith("[exit code: 0]"));
        assertFalse(new ReadFileTool().supportsStream());
    }

    // ------------------------------------------------------------------ get_info

    @Test
    void getInfoNamesTheSystemAndTheWorkingDirectoryOfTheCall() {
        ToolsRegistry registry = ToolsRegistry.of(List.of("get_info"), null);
        Map<String, Object> info = registry.invoke("get_info", Map.of("cwd", dir.toString()));
        assertTrue(String.valueOf(info.get("os")).contains("Linux"));
        assertEquals(dir.toString(), info.get("cwd"));
    }

    // ------------------------------------------------------------------ MCP tools

    @Test
    void mcpToolsListUnderTheirJoinedNamesAndCallThrough() throws Exception {
        Path script = Files.writeString(dir.resolve("mcp.sh"), MCP_ECHO_SERVER);
        ToolsRegistry registry = ToolsRegistry.of(null, List.of(new McpServerConfig("fs", "sh",
                List.of(script.toString()), Map.of(), null, 5000)));
        Map<String, Object> entry = registry.list().stream()
                .filter(tool -> "mcp".equals(tool.get("type"))).findFirst().orElseThrow();
        assertEquals("fs_echo", entry.get("tool"));
        assertEquals("mcp", entry.get("type"));
        assertFalse(Boolean.TRUE.equals(((Map<?, ?>) entry.get("permissions")).get("write")));
        Map<String, Object> answer = registry.invoke("fs_echo", Map.of("text", "hi"));
        assertEquals("ECHO_OK", answer.get("plain_text_response"));
        registry.close();
    }

    // ------------------------------------------------------------------ helpers

    private static final String MCP_ECHO_SERVER = """
            #!/bin/sh
            while IFS= read -r line; do
              case "$line" in
                *'"method":"initialize"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{}}\\n' "$id"
                  ;;
                *'"method":"tools/list"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echoes","inputSchema":{"type":"object"}}]}}\\n' "$id"
                  ;;
                *'"method":"tools/call"'*)
                  id=${line%%,'"method"'*}; id=${id##*'"id"':}
                  printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"ECHO_OK"}]}}\\n' "$id"
                  ;;
              esac
            done
            """;

    private static int kb(int bytes) {
        return bytes * 1024;
    }

    /** The count of non-empty result lines before the total line. */
    private static int countMatches(String text) {
        int count = 0;
        for (String line : text.split("\n")) {
            if (!line.isEmpty() && !line.startsWith("---") && !line.contains("Total matches")
                    && !line.startsWith("[")) {
                count++;
            }
        }
        return count;
    }

    private static String read(Path file) {
        try {
            return Files.readString(file);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
