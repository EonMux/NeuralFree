package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import io.quarkus.runtime.StartupEvent;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Singleton;
import jakarta.ws.rs.HttpMethod;
import jakarta.ws.rs.Priorities;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.ConfigProvider;
import org.jboss.resteasy.reactive.server.ServerRequestFilter;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Startup security warnings and a guard on requests naming server-executed tools: their {@code Host} must be trusted
 * (DNS rebinding) and so must a browser's {@code Origin}, since the server answers CORS for any origin and a page on
 * another site could otherwise drive the tools of a local server. The startup warnings name the enabled tool
 * features and, as llama.cpp does, an open CORS policy without an API key.
 */
@Singleton
public class ToolsSecurityFilter {

    private static final Logger LOGGER = Logger.getLogger(ToolsSecurityFilter.class.getName());
    private static final Set<String> CHAT_PATHS = Set.of("v1/chat/completions", "chat/completions",
            "v1/chat/completions/batch", "v1/chat/completions/input_tokens", "chat/completions/input_tokens",
            "v1/responses", "responses", "v1/responses/input_tokens", "responses/input_tokens",
            "v1/messages", "v1/messages/count_tokens");

    private static final String ORIGIN = "Origin";

    private final AllowedToolHosts hosts;
    private final String features;
    private final ApiKeys keys;
    private final CorsPolicy cors;

    public ToolsSecurityFilter() {
        this(ServerOptions.fromSystemProperties(),
                new AllowedToolHosts(ConfigProvider.getConfig().getOptionalValue(AllowedToolHosts.BIND_PROPERTY,
                        String.class).orElse("127.0.0.1"), System.getProperty(AllowedToolHosts.PROPERTY)),
                ApiKeys.from(System.getenv(), System.getProperties()),
                CorsPolicy.from(ConfigProvider.getConfig()));
    }

    ToolsSecurityFilter(ServerOptions options, AllowedToolHosts hosts, ApiKeys keys) {
        this(options, hosts, keys, CorsPolicy.disabled());
    }

    ToolsSecurityFilter(ServerOptions options, AllowedToolHosts hosts, ApiKeys keys, CorsPolicy cors) {
        this.hosts = hosts;
        this.keys = keys;
        this.cors = cors;
        boolean tools = options.tools() != null && !options.tools().isEmpty();
        boolean mcp = options.mcpServers() != null && !options.mcpServers().isEmpty();
        features = (tools ? "server tools (experimental)" : "")
                + (mcp ? (tools ? ", " : "") + "MCP servers (experimental)" : "");
    }

    /** Emits each warning once at application startup, without loading the model or spawning MCP processes. */
    void onStartup(@Observes StartupEvent event) {
        if (cors.admitsEveryOrigin() && !keys.enabled()) {
            LOGGER.warning("security: no API key is set and CORS allows all origins (see "
                    + "https://github.com/ggml-org/llama.cpp/pull/25655)");
        }
        if (!features.isEmpty()) {
            LOGGER.warning("security: " + features + " enabled - do not expose to untrusted environments");
            if (!keys.enabled()) {
                LOGGER.warning("security: " + features + " enabled with no API key configured - set "
                        + ApiKeys.PROPERTY + " (--api-key), " + ApiKeys.FILE_PROPERTY + " or " + ApiKeys.ENV);
            }
        }
    }

    /**
     * Runs after resource matching, beside the API-key filter, on the endpoint's worker thread: the guarded paths'
     * Host authority is checked without touching the body, and a foreign-host JSON request to a chat surface is read
     * and restored byte for byte so ordinary clients keep their resource's normal parsing and error handling.
     */
    @ServerRequestFilter(priority = Priorities.USER + 100)
    public Response filter(ContainerRequestContext request) {
        if (features.isEmpty() || hosts.disabled() || HttpMethod.OPTIONS.equals(request.getMethod())) {
            return null;
        }
        String path = path(request.getUriInfo().getPath());
        boolean chat = HttpMethod.POST.equals(request.getMethod()) && CHAT_PATHS.contains(path);
        if (!path.equals("tools") && !chat) {
            return null;
        }
        if (trusted(request)) {
            return null;
        }
        if (!chat) {
            return refusal();
        }
        MediaType type = request.getMediaType();
        if (type == null || !MediaType.APPLICATION_JSON_TYPE.isCompatible(type)) {
            return null; // the resource's @Consumes rejects unsupported content types
        }
        return guardChatBody(request, path);
    }

    /**
     * Inspects a foreign-host chat body for server-executed tools: a body that cannot be read leaves the consumed
     * stream to the resource's empty-body refusal, and every other branch answers over the restored exact bytes.
     */
    /** A single trusted Host header, and at most one Origin header, which must be trusted when present. */
    private boolean trusted(ContainerRequestContext request) {
        List<String> host = request.getHeaders().get(HttpHeaders.HOST);
        List<String> origin = request.getHeaders().get(ORIGIN);
        boolean hostTrusted = host != null && host.size() == 1 && hosts.accepts(host.get(0));
        boolean originTrusted = origin == null || origin.isEmpty()
                || (origin.size() == 1 && hosts.acceptsOrigin(origin.get(0)));
        return hostTrusted && originTrusted;
    }

    private Response guardChatBody(ContainerRequestContext request, String path) {
        byte[] bytes;
        try {
            bytes = request.getEntityStream().readAllBytes();
        } catch (IOException e) {
            return null;
        }
        request.setEntityStream(new ByteArrayInputStream(bytes));
        JsonView body;
        try {
            body = ServerJson.parse(new String(bytes, StandardCharsets.UTF_8));
        } catch (WebApplicationException e) {
            return e.getResponse();
        }
        boolean toolRequest = serverTools(body.raw("tools"));
        if ((path.equals("responses") || path.startsWith("responses/") || path.startsWith("v1/responses"))
                && body.raw("extra_params") instanceof Map<?, ?> extra) {
            toolRequest |= serverTools(extra.get("tools"));
        }
        return toolRequest ? refusal() : null;
    }

    private static boolean serverTools(Object raw) {
        if (raw instanceof List<?> tools) {
            for (Object entry : tools) {
                if (entry instanceof Map<?, ?> tool
                        && ("server".equals(tool.get("type")) || "mcp".equals(tool.get("type")))) {
                    return true;
                }
            }
        }
        return false;
    }

    /** The relative resource path, without matrix parameters, empty segments or dot segments. */
    private static String path(String raw) {
        var parts = new ArrayDeque<String>();
        for (String segment : raw.split("/")) {
            int matrix = segment.indexOf(';');
            String part = matrix < 0 ? segment : segment.substring(0, matrix);
            if (part.equals("..")) {
                parts.pollLast();
            } else if (!part.isEmpty() && !part.equals(".")) {
                parts.addLast(part);
            }
        }
        return String.join("/", parts);
    }

    private static Response refusal() {
        return Response.status(403).type(MediaType.APPLICATION_JSON)
                .entity(ServerJson.write(ServerJson.errorBody("Host or Origin is not allowed for server tools or MCP servers;"
                        + " configure " + AllowedToolHosts.PROPERTY + " (" + AllowedToolHosts.FLAG
                        + ") with trusted host names", "invalid_request_error")))
                .build();
    }
}
