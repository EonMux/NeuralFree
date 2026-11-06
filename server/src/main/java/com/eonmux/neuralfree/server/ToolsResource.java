package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.server.tools.ServerTool;
import com.eonmux.neuralfree.server.tools.ToolIo;
import com.eonmux.neuralfree.server.tools.ToolsRegistry;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The server tools endpoint (llama.cpp's {@code GET /tools} and {@code POST /tools}): the listing of every
 * server-executed tool — the built-ins of {@code --tools} and the {@code --mcp-servers-*} MCP servers — and the
 * direct invocation of one by name with its arguments, the same execution path the chat tool loop runs. Off
 * (403) until the launch enables a tool; a call of an unknown tool answers 404, one that does not stream with
 * {@code stream: true} likewise, and a malformed body 400. The {@code X-Tool-Cwd} header sets the working
 * directory of the call, overriding the server process one; a {@code cwd} in the body never does — it is
 * dropped, as llama.cpp drops it, because the model cannot choose where the tools run.
 *
 * <p>These tools give the model file and command access to the host. Start the server with them only in
 * trusted environments, as llama.cpp warns for the same feature.</p>
 *
 * <p>The routes are server-scoped, not per model: llama.cpp's router serves {@code /tools} from the router
 * process itself, and every routed engine carries the same launch's registry, so router mode answers here from
 * the lifecycle service without routing.</p>
 */
@Path("/")
public class ToolsResource {

    @Inject
    EngineService service;

    /** The current HTTP request (a request-scoped proxy), for the {@code X-Tool-Cwd} header; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The registry the resource answers from, when wired over one directly (tests). */
    private final ToolsRegistry direct;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public ToolsResource(EngineService service) {
        this.service = service;
        this.direct = null;
    }

    /** Wiring over an explicit registry, engine-independent. */
    public ToolsResource(ToolsRegistry registry) {
        this.service = null;
        this.direct = registry;
    }

    public ToolsResource() {
        this.direct = null;
    }

    /** The registry of the wired service, or the directly wired one; null when neither carries one. */
    private ToolsRegistry registry() {
        if (direct != null) {
            return direct;
        }
        return service == null ? null : service.tools();
    }

    /** Every tool's entry: name, display name, kind, write permission, and the OpenAI function definition. */
    @GET
    @Path("/tools")
    @Produces(MediaType.APPLICATION_JSON)
    public Response list() {
        ToolsRegistry registry = registry();
        if (registry == null || !registry.enabled()) {
            throw disabled();
        }
        return Response.ok(ServerJson.write(registry.list())).build();
    }

    /**
     * Runs one tool: {@code {"tool": name, "params": {...}, "stream": false}} answers the tool's result object,
     * or, with {@code stream: true} on a streaming tool, the server-sent-event frames of its output
     * ({@code {"chunk": ...}}, then {@code {"done": true}}).
     */
    @POST
    @Path("/tools")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response invoke(String bodyText) {
        ToolsRegistry registry = registry();
        if (registry == null || !registry.enabled()) {
            throw disabled();
        }
        return OpenAiResource.mapped(() -> invokeOf(registry, ServerJson.parse(bodyText)));
    }

    private Response invokeOf(ToolsRegistry registry, JsonView body) {
        String name = body.optStr("tool", "");
        if (name.isEmpty()) {
            throw ServerJson.badRequest("tool is required: the name of a server tool to run");
        }
        Map<String, Object> params = params(body);
        boolean stream = body.optBool("stream", false);
        ServerTool tool = registry.find(name);
        if (tool == null) {
            throw ServerJson.error(404, "unknown tool \"" + name + "\"", "invalid_request_error");
        }
        if (stream && !tool.supportsStream()) {
            throw ServerJson.error(404, "tool \"" + name + "\" does not support stream = true",
                    "invalid_request_error");
        }
        if (stream) {
            StreamingOutput frames = out -> stream(out, tool, params);
            return Response.ok(frames).type("text/event-stream").build();
        }
        Map<String, Object> result = registry.invoke(name, params);
        return Response.ok(ServerJson.write(result)).build();
    }

    /**
     * The call's arguments as an object: a body {@code cwd} dropped, the {@code X-Tool-Cwd} header put in its
     * place, the isolate and response-type selectors of llama.cpp dropped (no isolate runs in this build).
     */
    private Map<String, Object> params(JsonView body) {
        Map<String, Object> params = new LinkedHashMap<>();
        if (body.raw("params") == null) {
            return params;
        }
        if (!(body.raw("params") instanceof Map<?, ?> raw)) {
            throw ServerJson.badRequest("params must be an object");
        }
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            String key = String.valueOf(entry.getKey());
            if (key.equals("cwd") || key.equals("runtime") || key.equals("resp_type")) {
                continue;
            }
            params.put(key, entry.getValue());
        }
        if (http != null) {
            String cwd = http.headers().get("X-Tool-Cwd");
            if (cwd != null && !cwd.isBlank()) {
                params.put("cwd", cwd.trim());
            }
        }
        return params;
    }

    /** The event frames of a streamed call: every output chunk, then the done frame, an error carried in it. */
    private static void stream(OutputStream out, ServerTool tool, Map<String, Object> params) throws IOException {
        try {
            tool.invokeStream(params, new ToolIo(null), chunk -> {
                try {
                    ServerJson.event(out, null, Map.of("chunk", chunk));
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        } catch (UncheckedIOException e) {
            throw e.getCause(); // the client went away mid-stream
        } catch (RuntimeException e) {
            ServerJson.event(out, null, doneFrame(e.getMessage() == null
                    ? e.getClass().getSimpleName() : e.getMessage()));
            return;
        }
        ServerJson.event(out, null, doneFrame(null));
    }

    /** The final frame of a streamed call: {@code done}, the failure message when one was caught. */
    private static Map<String, Object> doneFrame(String error) {
        Map<String, Object> done = new LinkedHashMap<>();
        done.put("done", true);
        if (error != null) {
            done.put("error", error);
        }
        return done;
    }

    /** The 403 of a server launched without tools (llama.cpp's {@code res_403}). */
    private static WebApplicationException disabled() {
        return ServerJson.error(403, "server tools are disabled: start the server with --tools or"
                + " --mcp-servers-config to enable them", "invalid_request_error");
    }
}
