package com.eonmux.neuralfree.server.tools;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code exec_shell_command}: a command run under {@code sh -c} in the working directory, its combined stdout
 * and stderr followed by the exit code line, killed at its timeout (default 10 s, at most 60 s) and held to the
 * output cap (default and maximum 16 KB) (llama.cpp's {@code server_tool_exec_shell_command}).
 */
public final class ExecShellCommandTool extends ServerTool {

    /** The default and most bytes one command answers with. */
    static final int MAX_OUTPUT = 16 * 1024;

    /** The default and most seconds one command may run. */
    static final int MAX_TIMEOUT = 60;

    public ExecShellCommandTool() {
        super("exec_shell_command", "Execute shell command", true, true);
    }

    @Override
    public Map<String, Object> definition() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("command", Map.of("type", "string", "description", "Shell command to execute"));
        properties.put("timeout", Map.of("type", "integer", "description",
                "Timeout in seconds (default 10, max " + MAX_TIMEOUT + ")"));
        properties.put("max_output_size", Map.of("type", "integer", "description",
                "Maximum output size in bytes (default " + MAX_OUTPUT + ")"));
        return functionDefinition(name(),
                "Execute a shell command and return its output (stdout and stderr combined).",
                parameters(properties, "command"));
    }

    @Override
    public boolean supportsStream() {
        return true;
    }

    @Override
    public void invokeStream(Map<String, Object> params, ToolIo io, java.util.function.Consumer<String> push) {
        String command = string(params, "command");
        int timeout = Math.min(optInt(params, "timeout", 10), MAX_TIMEOUT);
        int maxOutput = Math.min(optInt(params, "max_output_size", MAX_OUTPUT), MAX_OUTPUT);
        ToolIo.Run run = io.run(command, timeout, maxOutput, push);
        push.accept("\n[exit code: " + run.exitCode() + "]" + (run.timedOut() ? " [exit due to timed out]" : ""));
    }

    @Override
    public Map<String, Object> invoke(Map<String, Object> params, ToolIo io) {
        String command = string(params, "command");
        int timeout = Math.min(optInt(params, "timeout", 10), MAX_TIMEOUT);
        int maxOutput = Math.min(optInt(params, "max_output_size", MAX_OUTPUT), MAX_OUTPUT);
        ToolIo.Run run = io.run(command, timeout, maxOutput);
        String text = run.output() + "\n[exit code: " + run.exitCode() + "]";
        if (run.timedOut()) {
            text += " [exit due to timed out]";
        }
        return text(text);
    }
}
