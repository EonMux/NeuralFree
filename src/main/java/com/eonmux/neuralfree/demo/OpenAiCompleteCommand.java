package com.eonmux.neuralfree.demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * {@code neuralfree complete}: vLLM's {@code CompleteCommand} (vllm/entrypoints/cli/openai.py). It streams text
 * completions from a running OpenAI-compatible server ({@code --url}, default
 * {@code http://localhost:8000/v1}): every line read from standard input is one prompt, completed with the
 * server's default sampling unless {@code --max-tokens} caps it. {@code -q/--quick} completes one prompt and
 * exits. The model is {@code --model-name}, else the first model the server lists, and {@code --stats} prints the
 * TTFT and the TPS after each answer.
 */
public final class OpenAiCompleteCommand {

    private OpenAiCompleteCommand() {
    }

    /** Runs the command on the process's streams and exits with its code. */
    public static void run(String[] args) {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        int code;
        try {
            code = execute(args, in, System.out, System.err);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            code = 1;
        }
        System.exit(code);
    }

    /**
     * Runs the command over the given streams and returns its exit code: 0 after the prompts, 1 when the server
     * cannot be reached or a request fails, 2 for an invalid command line.
     */
    static int execute(String[] args, BufferedReader in, PrintStream out, PrintStream err) throws IOException {
        OpenAiClientOptions options;
        try {
            options = OpenAiClientOptions.parse(args, false, out);
        } catch (OpenAiClientOptions.UsageException e) {
            err.println("Error: " + e.getMessage());
            OpenAiClientOptions.usage(false, err);
            return 2;
        }
        if (options == null) {
            return 0;
        }
        OpenAiClientCommand.Session session = OpenAiClientCommand.connect(options, out, err);
        if (session == null) {
            return 1;
        }
        if (options.quick() != null) {
            return complete(session, options, options.quick(), out, err);
        }
        out.println("Please enter prompt to complete:");
        while (true) {
            String line = OpenAiClientCommand.read(in, out);
            if (line == null) {
                return 0;
            }
            int code = complete(session, options, line, out, err);
            if (code != 0) {
                return code;
            }
        }
    }

    /** Sends one prompt as one streamed completion and prints the output as it arrives. */
    private static int complete(OpenAiClientCommand.Session session, OpenAiClientOptions options, String prompt,
                                PrintStream out, PrintStream err) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", session.model());
        body.put("prompt", prompt);
        body.put("stream", true);
        if (options.maxTokens() != null) {
            body.put("max_tokens", options.maxTokens());
        }
        if (options.stats()) {
            body.put("stream_options", Collections.singletonMap("include_usage", true));
        }
        return OpenAiClientCommand.send(session, body, false, options.stats(), out, err) == null ? 1 : 0;
    }
}
