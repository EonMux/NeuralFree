package com.eonmux.neuralfree.server.tools;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code get_info}: the runtime the tools work on — the OS name ({@code uname -a}) and the working directory the
 * other tools resolve against (llama.cpp's {@code server_tool_get_info}).
 */
public final class GetInfoTool extends ServerTool {

    public GetInfoTool() {
        super("get_info", "Get Runtime Info", false, true);
    }

    @Override
    public Map<String, Object> definition() {
        return functionDefinition(name(),
                "Returns runtime info: the OS name/version and the current working directory",
                parameters(Map.of()));
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        String os = System.getProperty("os.name", "unknown") + " " + System.getProperty("os.version", "");
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("os", os.trim());
        out.put("cwd", io.resolve("").toString());
        return out;
    }
}
