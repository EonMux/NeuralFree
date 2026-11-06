package com.eonmux.neuralfree.demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code neuralfree chat}: vLLM's {@code ChatCommand} (vllm/entrypoints/cli/openai.py). It streams chat completions
 * from a running OpenAI-compatible server ({@code --url}, default {@code http://localhost:8000/v1}): the
 * conversation opens with {@code --system-prompt} when it is given, then every line read from standard input is a
 * user message and every answer grows the conversation with the assistant's content. {@code -q/--quick} sends one
 * message and exits. The model is {@code --model-name}, else the first model the server lists. Reasoning streams
 * between {@code <think>} and {@code </think>}, and {@code --stats} prints the TTFT and the TPS after each answer.
 */
public final class OpenAiChatCommand {

    private OpenAiChatCommand() {
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
     * Runs the command over the given streams and returns its exit code: 0 after the conversation, 1 when the
     * server cannot be reached or a request fails, 2 for an invalid command line.
     */
    static int execute(String[] args, BufferedReader in, PrintStream out, PrintStream err) throws IOException {
        OpenAiClientOptions options;
        try {
            options = OpenAiClientOptions.parse(args, true, out);
        } catch (OpenAiClientOptions.UsageException e) {
            err.println("Error: " + e.getMessage());
            OpenAiClientOptions.usage(true, err);
            return 2;
        }
        if (options == null) {
            return 0;
        }
        OpenAiClientCommand.Session session = OpenAiClientCommand.connect(options, out, err);
        if (session == null) {
            return 1;
        }
        List<Map<String, Object>> conversation = new ArrayList<>();
        if (options.systemPrompt() != null) {
            conversation.add(message("system", options.systemPrompt()));
        }
        if (options.quick() != null) {
            conversation.add(message("user", options.quick()));
            return turn(session, conversation, options.stats(), out, err);
        }
        out.println("Please enter a message for the chat model:");
        while (true) {
            String line = OpenAiClientCommand.read(in, out);
            if (line == null) {
                return 0;
            }
            conversation.add(message("user", line));
            int code = turn(session, conversation, options.stats(), out, err);
            if (code != 0) {
                return code;
            }
        }
    }

    /**
     * Sends the conversation as one streamed request, prints the answer and appends it to the conversation as the
     * assistant's message.
     */
    private static int turn(OpenAiClientCommand.Session session, List<Map<String, Object>> conversation,
                            boolean stats, PrintStream out, PrintStream err) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", session.model());
        body.put("messages", new ArrayList<>(conversation));
        body.put("stream", true);
        if (stats) {
            body.put("stream_options", Collections.singletonMap("include_usage", true));
        }
        OpenAiClientCommand.Answer answer = OpenAiClientCommand.send(session, body, true, stats, out, err);
        if (answer == null) {
            return 1;
        }
        conversation.add(message("assistant", answer.content()));
        return 0;
    }

    /** One OpenAI chat message. */
    static Map<String, Object> message(String role, String content) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }
}
