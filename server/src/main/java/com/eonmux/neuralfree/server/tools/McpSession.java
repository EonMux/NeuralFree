package com.eonmux.neuralfree.server.tools;

import com.eonmux.neuralfree.utils.json.Json;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * One MCP server session (llama.cpp's {@code server_mcp_transport} + {@code server_mcp_stdio}): the child
 * process of a {@link McpServerConfig}, one JSON-RPC 2.0 message per line over its stdin and stdout, stderr
 * drained into a bounded tail for diagnostics. The JSON-RPC session owns the handshake ({@code initialize},
 * then the {@code notifications/initialized} notification), {@code tools/list} and {@code tools/call}, matching
 * replies by their monotonically increasing id and skipping notifications and stale frames; every call runs
 * under the config's per-call budget and answers {@code {"error": ...}} on a timeout, a dead transport or an
 * RPC error, the shape the callers pass on.
 */
public final class McpSession {

    /** The protocol version the handshake offers (llama.cpp's {@code MCP_PROTOCOL_VERSION}). */
    static final String PROTOCOL_VERSION = "2024-11-05";

    /** The largest NDJSON line a child may write before the reader drops until the next newline. */
    static final int MAX_LINE_CHARS = 8 * 1024 * 1024;

    private static final Logger LOGGER = Logger.getLogger(McpSession.class.getName());

    private final McpServerConfig config;
    private final AtomicBoolean running = new AtomicBoolean();
    private final BlockingQueue<String> frames = new LinkedBlockingQueue<>();
    /** Guards the stdin writer; the reader and errlog threads never touch it. */
    private final Object writeLock = new Object();
    private Process process;
    private Writer stdin;
    private volatile long nextId = 1;
    private volatile boolean initialized;
    private volatile String lastError = "";
    private volatile String errTail = "";

    /** Spawns the child of {@code config}; {@link #alive()} says whether it started. */
    public McpSession(McpServerConfig config) {
        this.config = config;
    }

    /** Starts the child and its pumps; false when the spawn failed, with the reason in {@link #diagnostics()}. */
    public boolean start() {
        if (running.get()) {
            return true;
        }
        List<String> command = new java.util.ArrayList<>(config.args());
        command.add(0, config.command());
        try {
            ProcessBuilder builder = new ProcessBuilder(command);
            builder.environment().putAll(config.env());
            if (config.cwd() != null) {
                builder.directory(new java.io.File(config.cwd()));
            }
            process = builder.start();
            stdin = new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8);
            running.set(true);
            Thread reader = Thread.ofPlatform().name("mcp-" + config.name() + "-reader").daemon()
                    .start(this::readFrames);
            errlog(config.name(), process, this);
            return process.isAlive();
        } catch (IOException e) {
            lastError = "failed to spawn '" + config.command() + "': " + e.getMessage();
            running.set(false);
            return false;
        }
    }

    /** Whether the child is running and has not been closed. */
    public boolean alive() {
        return running.get() && process != null && process.isAlive();
    }

    /** The last failure the session saw: a spawn, an RPC or a transport error. */
    public String diagnostics() {
        String tail = errTail.isBlank() ? "" : "; stderr: " + errTail.replace('\n', ' ').strip();
        return lastError + tail;
    }

    /**
     * The tools the server advertises: the handshake first, then {@code tools/list}, the definitions of
     * {@link #callTool}-ready tools. Empty on a failure, with the reason in {@link #diagnostics()}.
     */
    public List<McpTool.Def> listTools(long timeoutMs) {
        if (!ensureInit(timeoutMs)) {
            return List.of();
        }
        Map<String, Object> reply = sendRpc(request("tools/list"), timeoutMs);
        if (reply.containsKey("error")) {
            lastError = "tools/list failed: " + errorText(reply);
            return List.of();
        }
        return tools(reply, config.name());
    }

    /** The tools of a {@code tools/list} reply: name, description and input schema of each entry. */
    private static List<McpTool.Def> tools(Map<String, Object> reply, String server) {
        Object result = reply.get("result");
        if (!(result instanceof Map<?, ?> body) || !(body.get("tools") instanceof List<?> list)) {
            return List.of();
        }
        List<McpTool.Def> out = new java.util.ArrayList<>();
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> tool) || !(tool.get("name") instanceof String name)) {
                continue;
            }
            String description = tool.get("description") instanceof String text ? text : "";
            Map<String, Object> schema = tool.get("inputSchema") instanceof Map<?, ?> input
                    ? objectOf(input) : null;
            out.add(new McpTool.Def(server, name, description, schema));
        }
        return out;
    }

    /** A parsed JSON object tree as its string-keyed map (the parser guarantees string keys). */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> objectOf(Map<?, ?> map) {
        return (Map<String, Object>) map;
    }

    /**
     * Calls the tool {@code name} with {@code arguments}: the answer text of its content parts, an MCP
     * {@code isError} result surfaced as {@code {"error": ...}}.
     */
    public Map<String, Object> callTool(String name, Map<String, Object> arguments) {
        if (!ensureInit(config.timeoutMs())) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("error", lastError);
            return out;
        }
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("name", name);
        params.put("arguments", arguments);
        Map<String, Object> request = request("tools/call");
        request.put("params", params);
        Map<String, Object> reply = sendRpc(request, config.timeoutMs());
        if (reply.containsKey("error")) {
            return Map.of("error", errorText(reply));
        }
        return result(reply);
    }

    /**
     * An MCP {@code tools/call} result as the {@code POST /tools} contract (llama.cpp's
     * {@code mcp_result_to_response}): the {@code text} parts of {@code content} joined with newlines under
     * {@code plain_text_response}, an {@code isError} result as {@code error}.
     */
    private static Map<String, Object> result(Map<String, Object> reply) {
        StringBuilder text = new StringBuilder();
        if (reply.get("result") instanceof Map<?, ?> body && body.get("content") instanceof List<?> parts) {
            for (Object part : parts) {
                if (part instanceof Map<?, ?> piece && "text".equals(piece.get("type"))
                        && piece.get("text") instanceof String chunk) {
                    if (!text.isEmpty()) {
                        text.append('\n');
                    }
                    text.append(chunk);
                }
            }
        }
        boolean isError = reply.get("result") instanceof Map<?, ?> body && Boolean.TRUE.equals(body.get("isError"));
        if (isError) {
            return Map.of("error", text.isEmpty() ? "MCP tool returned an error" : text.toString());
        }
        return Map.of("plain_text_response", text.toString());
    }

    /** Closes the session: the pumps stop, the child is destroyed and reaped. Idempotent. */
    public void close() {
        running.set(false);
        if (process == null) {
            return;
        }
        synchronized (writeLock) {
            try {
                stdin.close();
            } catch (IOException | UncheckedIOException e) {
                // the child may already be gone; the destroy below reaps it
            }
        }
        process.destroy();
        try {
            if (!process.waitFor(5, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                process.waitFor(2, TimeUnit.SECONDS);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            process.destroyForcibly();
        }
        process = null;
    }

    // ------------------------------------------------------------------ JSON-RPC plumbing

    /** The handshake once per session: {@code initialize}, then the initialized notification. */
    private boolean ensureInit(long timeoutMs) {
        if (initialized) {
            return true;
        }
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("protocolVersion", PROTOCOL_VERSION);
        params.put("capabilities", Map.of());
        params.put("clientInfo", Map.of("name", "neuralfree", "version", version()));
        Map<String, Object> request = request("initialize");
        request.put("params", params);
        Map<String, Object> reply = sendRpc(request, timeoutMs);
        if (!reply.containsKey("result")) {
            lastError = "initialize failed: " + errorText(reply);
            return false;
        }
        initialized = true;
        Map<String, Object> notification = new LinkedHashMap<>();
        notification.put("jsonrpc", "2.0");
        notification.put("method", "notifications/initialized");
        write(notification);
        return true;
    }

    /** A request body with the next id and the {@code method}. */
    private Map<String, Object> request(String method) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("jsonrpc", "2.0");
        out.put("id", nextId++);
        out.put("method", method);
        return out;
    }

    /**
     * Writes {@code request} and reads until its reply (the frame with its id), a notification-free timeout or
     * the transport's death; the reply map, or {@code {"error": {...}}} with the failure. A request without a
     * result or error (a malformed reply) answers the same error shape.
     */
    private Map<String, Object> sendRpc(Map<String, Object> request, long timeoutMs) {
        if (!write(request)) {
            return transportError("transport closed");
        }
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        Object id = request.get("id");
        while (System.nanoTime() - deadline < 0) {
            String frame;
            try {
                frame = frames.poll(50, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return transportError("interrupted");
            }
            if (frame == null) {
                if (!alive()) {
                    return transportError("transport closed" + (lastError.isEmpty() ? "" : ": " + lastError));
                }
                continue;
            }
            Map<String, Object> reply;
            try {
                reply = Json.parse(frame, Json.Limits.UNTRUSTED) instanceof Map<?, ?> parsed
                        ? objectOf(parsed) : null;
            } catch (RuntimeException e) {
                continue; // a malformed frame is dropped, the wait continues
            }
            if (reply == null || !reply.containsKey("id") || !reply.get("id").equals(id)) {
                continue; // a notification or a stale reply of an already-given-up request
            }
            if (!reply.containsKey("result") && !reply.containsKey("error")) {
                return transportError("invalid response from MCP server");
            }
            return reply;
        }
        return transportError("request timed out");
    }

    /** The {@code error} of a reply as its message text. */
    private static String errorText(Map<String, Object> reply) {
        Object error = reply.get("error");
        if (error instanceof Map<?, ?> body && body.get("message") instanceof String message) {
            return message;
        }
        return error instanceof String s ? s : "unknown error";
    }

    /** The error map a transport failure answers. */
    private static Map<String, Object> transportError(String message) {
        return Map.of("error", Map.of("code", -32603, "message", message));
    }

    /** Writes one message as an NDJSON line; false when the pipe broke. */
    private boolean write(Map<String, Object> message) {
        synchronized (writeLock) {
            if (stdin == null || !running.get()) {
                return false;
            }
            try {
                stdin.write(Json.toString(message, Json.Style.API));
                stdin.write("\n");
                stdin.flush();
                return true;
            } catch (IOException | UncheckedIOException e) {
                lastError = "write failed: " + e.getMessage();
                return false;
            }
        }
    }

    /** The reader loop: stdout lines into the frame queue, an over-long line dropped until its newline. */
    private void readFrames() {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            StringBuilder line = new StringBuilder();
            char[] buffer = new char[4096];
            boolean skipping = false;
            int read;
            while (running.get() && (read = reader.read(buffer)) >= 0) {
                for (int i = 0; i < read; i++) {
                    char c = buffer[i];
                    if (c == '\n') {
                        if (!skipping && !line.isEmpty()) {
                            frames.offer(line.toString());
                        }
                        line.setLength(0);
                        skipping = false;
                        continue;
                    }
                    if (skipping) {
                        continue;
                    }
                    if (line.length() > MAX_LINE_CHARS) {
                        LOGGER.warning("MCP '" + config.name() + "': dropping oversized line from child stdout");
                        line.setLength(0);
                        skipping = true;
                        continue;
                    }
                    line.append(c);
                }
            }
        } catch (IOException e) {
            if (running.get()) {
                lastError = "read failed: " + e.getMessage();
            }
        }
    }

    /** The stderr pump: a bounded tail kept for {@link #diagnostics()} when the child dies. */
    private static void errlog(String name, Process process, McpSession session) {
        Thread.ofPlatform().name("mcp-" + name + "-stderr").daemon().start(() -> {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
                StringBuilder tail = new StringBuilder();
                char[] buffer = new char[2048];
                int read;
                while ((read = reader.read(buffer)) >= 0) {
                    tail.append(buffer, 0, read);
                    if (tail.length() > 4096) {
                        tail.delete(0, tail.length() - 4096);
                    }
                }
                session.errTail = tail.toString().strip();
                if (!session.errTail.isEmpty()) {
                    LOGGER.fine("MCP '" + name + "' stderr: " + session.errTail);
                }
            } catch (IOException e) {
                LOGGER.log(Level.FINE, "MCP '" + name + "' stderr pump stopped", e);
            }
        });
    }

    /** The client version the handshake names (llama.cpp names its own build as {@code 1.0}). */
    private static String version() {
        return "1.0";
    }
}
