package com.eonmux.neuralfree.server.tools;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code read_file}: the contents of a file, optionally a 1-based line range, each line optionally prefixed with
 * its {@code line→} number (llama.cpp's {@code server_tool_read_file}). A file larger than the 16 KB answer cap
 * refuses with an error asking for a range; a range that still overruns the cap truncates with the marker
 * {@code [output truncated]}.
 */
public final class ReadFileTool extends ServerTool {

    /** The most text one read answers with. */
    static final int MAX_SIZE = 16 * 1024;

    public ReadFileTool() {
        super("read_file", "Read file", false, true);
    }

    @Override
    public Map<String, Object> definition() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("path", Map.of("type", "string", "description", "Path to the file"));
        properties.put("start_line", Map.of("type", "integer", "description",
                "First line to read, 1-based (default: 1)"));
        properties.put("end_line", Map.of("type", "integer", "description",
                "Last line to read, 1-based inclusive (default: end of file)"));
        properties.put("append_loc", Map.of("type", "boolean", "description",
                "Prefix each line with its line number (e.g. \"1→...\")"));
        return functionDefinition(name(),
                "Read the contents of a file. Optionally specify a 1-based line range. If append_loc is true,"
                        + " each line is prefixed with its line number (e.g. \"1→...\").",
                parameters(properties, "path"));
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        String path = string(params, "path");
        int startLine = optInt(params, "start_line", 1);
        int endLine = optInt(params, "end_line", -1);
        boolean appendLoc = optBool(params, "append_loc", false);
        long size = io.fileSize(path);
        if (size < 0) {
            return error("cannot stat file: " + path);
        }
        if (size > MAX_SIZE && endLine == -1) {
            return error("file too large (" + size + " bytes, max " + MAX_SIZE
                    + "). Use start_line/end_line to read a portion.");
        }
        String content = io.read(path);
        if (content == null) {
            return error("failed to open file: " + path);
        }
        return text(ranged(content, startLine, endLine, appendLoc));
    }

    /** The lines of {@code content} inside the inclusive 1-based range, joined with newlines. */
    private static String ranged(String content, int startLine, int endLine, boolean appendLoc) {
        String[] lines = content.split("\n", -1);
        if (content.endsWith("\n") && lines.length > 0 && lines[lines.length - 1].isEmpty()) {
            lines = java.util.Arrays.copyOf(lines, lines.length - 1);
        }
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < lines.length; i++) {
            int number = i + 1;
            if (number < Math.max(1, startLine)) {
                continue;
            }
            if (endLine != -1 && number > endLine) {
                break;
            }
            String line = (appendLoc ? number + "→" : "") + lines[i] + "\n";
            if (out.length() + line.length() > MAX_SIZE) {
                out.append("[output truncated]");
                break;
            }
            out.append(line);
        }
        return out.toString();
    }
}
