package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.tools.ToolCall;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.server.tools.ToolsRegistry;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The chat tool loop: when a chat request lists server-executed tools — entries of {@code type: "server"} or
 * {@code type: "mcp"} naming a tool of the launch's {@link ToolsRegistry} — a round whose calls all name such
 * tools runs them on the server (llama.cpp exposes the same tools over {@code POST /tools}; the loop is the
 * server-side form of the Web UI's agent flow), feeds every answer back as its tool message, and generates the
 * next round. A round holding a client function call ends the loop: the whole round goes back to the client
 * unchanged, as tool calls always have. The loop answers one choice, does not stream, and stops after
 * {@code neuralfree.server.tools.max.rounds} rounds (a request whose {@code tool_choice} forces a call every
 * round would otherwise run forever); the response then carries the pending calls.
 *
 * <p>A {@code cwd}, {@code runtime} or {@code resp_type} entry in the model's call arguments is dropped, as
 * llama.cpp drops them: the model cannot choose where the tools run.</p>
 */
final class ToolsLoop {

    /** The system property bounding the rounds of one request. */
    static final String MAX_ROUNDS_PROPERTY = "neuralfree.server.tools.max.rounds";

    /** The default round bound (vLLM's built-in tool loop is unbounded; the bound is this server's guard). */
    static final int DEFAULT_MAX_ROUNDS = 10;

    /** One executed call: the OpenAI id (the format's, or a synthesized one), the call and its answer. */
    record Executed(String id, String name, String arguments, Map<String, Object> result) { }

    /** The loop's answer: the final round and everything it sums up for the response. */
    record Outcome(ChatPlan plan, ChatChoice choice, GenerationResult result, List<GenerationResult> rounds,
                   List<Executed> executed, long promptRows, long cached) { }

    /** The request as the loop runs it: the substituted body and the server tool names it listed. */
    record Prepared(Map<String, Object> body, Set<String> names) { }

    private ToolsLoop() {
    }

    /**
     * The request prepared for the loop: the server-typed tool entries replaced by their function definitions
     * (the entry's own, or the registry's) so the plan renders them like any tool, and the names they listed.
     * Every entry's {@code type} is checked here: a type the server does not run refuses naming it. The
     * original body stands when no entry is server-typed ({@code names} empty, {@code body} null).
     *
     * @throws jakarta.ws.rs.WebApplicationException a 400 for a malformed entry, an unsupported tool type, or a
     *                                               name the server does not run
     */
    static Prepared prepare(EngineService service, JsonView request) {
        Object raw = request.raw("tools");
        if (!(raw instanceof List<?> list)) {
            return new Prepared(null, Set.of());
        }
        boolean serverTyped = false;
        for (Object entry : list) {
            if (!(entry instanceof Map<?, ?> tool)) {
                continue; // a non-object entry is the tools parser's refusal, naming the position
            }
            String type = String.valueOf(tool.get("type"));
            if (type.equals("server") || type.equals("mcp")) {
                serverTyped = true;
            } else if (!type.equals("function")) {
                throw ServerJson.badRequest("tools type '" + type + "' is not supported; use \"function\" or a"
                        + " server tool entry (\"server\" or \"mcp\")");
            }
        }
        if (!serverTyped) {
            return new Prepared(null, Set.of());
        }
        Map<String, Object> body = new LinkedHashMap<>();
        for (String key : request.keys()) {
            body.put(key, request.raw(key));
        }
        ToolsRegistry registry = service.tools();
        List<Object> tools = new ArrayList<>();
        Set<String> names = new LinkedHashSet<>();
        for (int i = 0; i < list.size(); i++) {
            Map<?, ?> entry = list.get(i) instanceof Map<?, ?> e ? e : Map.of();
            String type = String.valueOf(entry.get("type"));
            if (type.equals("function")) {
                tools.add(entry);
                continue;
            }
            String name = entry.get("tool") instanceof String named && !named.isEmpty() ? named : null;
            if (name == null) {
                throw ServerJson.badRequest("tools[" + i + "].tool must name a server tool");
            }
            if (registry == null || registry.find(name) == null) {
                throw ServerJson.badRequest("tools[" + i + "] names the server tool '" + name + "', which this"
                        + " server does not run; start it with --tools or --mcp-servers-config");
            }
            Map<String, Object> definition = entry.get("definition") instanceof Map<?, ?> own
                    ? OpenAiRequests.stringKeys(own) : registry.find(name).definition();
            tools.add(definition);
            names.add(name);
        }
        body.put("tools", tools);
        return new Prepared(body, names);
    }

    /**
     * The rounds: generate, execute the round's server tool calls when it made only those, feed the answers
     * back, generate again — until a round answers without such calls or the round bound is reached, whichever
     * comes first.
     */
    static Outcome run(EngineService service, Map<String, Object> body, Set<String> serverTools,
                       HttpServerRequest client) {
        int maxRounds = maxRounds();
        List<Executed> executed = new ArrayList<>();
        List<GenerationResult> rounds = new ArrayList<>();
        long promptRows = 0;
        long cached = 0;
        ChatPlan plan = null;
        ChatChoice choice = null;
        GenerationResult result = null;
        for (int round = 0; round < maxRounds; round++) {
            plan = ChatPlan.of(service, JsonView.of(body, "body"));
            if (round == 0 && plan.bestOf() > 1) {
                throw ServerJson.badRequest("server tool execution answers the best candidate of one: set"
                        + " best_of to 1 or drop it");
            }
            GenerationBatch batch = GenerationBatch.start(service, plan.jobs(), client);
            batch.awaitAll();
            List<ChatChoice> choices = new ArrayList<>();
            for (int i = 0; i < batch.size(); i++) {
                choices.add(plan.newChoice());
            }
            OpenAiResource.drainInto(batch, choices);
            result = batch.result(0);
            choice = choices.get(0);
            promptRows += plan.promptRows();
            cached += batch.cachedTokens(0);
            rounds.add(result);
            ToolParse parse = choice.finish();
            List<ToolCall> calls = parse.calls();
            if (calls.isEmpty() || !allServer(calls, serverTools)) {
                break;
            }
            body = extend(body, parse, calls, executed, service.tools());
        }
        return new Outcome(plan, choice, result, List.copyOf(rounds), List.copyOf(executed), promptRows, cached);
    }

    /** The round bound of the {@code neuralfree.server.tools.max.rounds} property, at least 1. */
    private static int maxRounds() {
        String raw = System.getProperty(MAX_ROUNDS_PROPERTY);
        if (raw == null || raw.isBlank()) {
            return DEFAULT_MAX_ROUNDS;
        }
        int rounds;
        try {
            rounds = Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(MAX_ROUNDS_PROPERTY + " must be a whole number, got '" + raw + "'", e);
        }
        if (rounds < 1) {
            throw new IllegalArgumentException(MAX_ROUNDS_PROPERTY + " must be 1 or more, got " + rounds);
        }
        return rounds;
    }

    /** Whether every call names a tool of {@code serverTools}. */
    private static boolean allServer(List<ToolCall> calls, Set<String> serverTools) {
        for (ToolCall call : calls) {
            if (!serverTools.contains(call.name())) {
                return false;
            }
        }
        return true;
    }

    /**
     * The conversation of the next round: the assistant message with its calls (ids the format carried, or
     * synthesized {@code call_N} ones), then each call's tool message with the executed answer as JSON text.
     */
    private static Map<String, Object> extend(Map<String, Object> body, ToolParse parse, List<ToolCall> calls,
                                              List<Executed> executed, ToolsRegistry registry) {
        List<Object> messages = new ArrayList<>(body.get("messages") instanceof List<?> list
                ? list : List.of());
        List<Object> toolCalls = new ArrayList<>();
        for (int i = 0; i < calls.size(); i++) {
            toolCalls.add(callBody(calls.get(i), synthesizedId(executed.size(), calls.get(i))));
        }
        Map<String, Object> assistant = new LinkedHashMap<>();
        assistant.put("role", "assistant");
        String content = parse.content();
        assistant.put("content", content.isEmpty() && !calls.isEmpty() ? null : content);
        assistant.put("tool_calls", toolCalls);
        messages.add(assistant);
        for (int i = 0; i < calls.size(); i++) {
            String id = (String) ((Map<?, ?>) toolCalls.get(i)).get("id");
            Map<String, Object> answer = execute(registry, calls.get(i));
            executed.add(new Executed(id, calls.get(i).name(), calls.get(i).arguments(), answer));
            Map<String, Object> tool = new LinkedHashMap<>();
            tool.put("role", "tool");
            tool.put("tool_call_id", id);
            tool.put("content", ServerJson.write(answer));
            messages.add(tool);
        }
        Map<String, Object> extended = new LinkedHashMap<>(body);
        extended.put("messages", messages);
        return extended;
    }

    /** The call id the format carried, or a synthesized {@code call_N} (the response convention). */
    private static String synthesizedId(int executedSoFar, ToolCall call) {
        return call.id() != null ? call.id() : "call_" + executedSoFar;
    }

    /** The OpenAI {@code tool_calls} entry of one call. */
    private static Map<String, Object> callBody(ToolCall call, String id) {
        Map<String, Object> fn = new LinkedHashMap<>();
        fn.put("name", call.name());
        fn.put("arguments", call.arguments());
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", id);
        out.put("type", "function");
        out.put("function", fn);
        return out;
    }

    /**
     * The answer of one call: its arguments parsed as a JSON object (a parse failure answers the error the
     * model reads), the working-directory and isolate selectors dropped, the registry's result. An unknown name
     * answers an error rather than a refusal — the loop already checked the round's names.
     */
    private static Map<String, Object> execute(ToolsRegistry registry, ToolCall call) {
        Object args;
        try {
            args = Json.parse(call.arguments(), Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            return Map.of("error", "the call arguments do not parse as JSON: " + e.getMessage());
        }
        if (!(args instanceof Map<?, ?> raw)) {
            return Map.of("error", "the call arguments must be a JSON object");
        }
        Map<String, Object> params = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            String key = String.valueOf(entry.getKey());
            if (key.equals("cwd") || key.equals("runtime") || key.equals("resp_type")) {
                continue;
            }
            params.put(key, entry.getValue());
        }
        Map<String, Object> result = registry.invoke(call.name(), params);
        return result == null ? Map.of("error", "unknown tool \"" + call.name() + "\"") : result;
    }
}
