package com.eonmux.neuralfree.server.tools;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code write_file}: content written to a file, created together with its parent directories when missing
 * (llama.cpp's {@code server_tool_write_file}).
 */
public final class WriteFileTool extends ServerTool {

    public WriteFileTool() {
        super("write_file", "Write file", true, true);
    }

    @Override
    public Map<String, Object> definition() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("path", Map.of("type", "string", "description", "Path of the file to write"));
        properties.put("content", Map.of("type", "string", "description", "Content to write"));
        return functionDefinition(name(),
                "Write content to a file, creating it (including parent directories) if it does not exist."
                        + " May use with edit_file for more complex edits.",
                parameters(properties, "path", "content"));
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        String path = string(params, "path");
        String content = string(params, "content");
        if (!io.write(path, content)) {
            return error("failed to write file: " + path);
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("result", "file written successfully");
        out.put("path", path);
        out.put("bytes", content.length());
        return out;
    }
}
