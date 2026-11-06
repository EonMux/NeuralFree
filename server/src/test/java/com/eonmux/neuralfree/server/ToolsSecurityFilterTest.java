package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.server.tools.McpServerConfig;
import io.quarkus.runtime.StartupEvent;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.MultivaluedHashMap;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.UriInfo;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The startup warnings and the narrow, body-preserving DNS-rebinding guard on server tool requests. */
class ToolsSecurityFilterTest {

    private static final String SERVER = "{\"tools\":[{\"type\":\"server\",\"tool\":\"get_info\"}]}";
    private static final String MCP = "{\"tools\":[{\"type\":\"mcp\",\"tool\":\"test_info\"}]}";
    private static final String FOREIGN = "attacker.example:8080";
    private static final String CHAT_BODY = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";

    private static ServerOptions options(boolean tools, boolean mcp) {
        return new ServerOptions(null, Duration.ofMinutes(1), false, 0, null, -1, 1, null,
                false, null, null, null, tools ? List.of("get_info") : null,
                mcp ? McpServerConfig.parse("{\"mcpServers\":{\"test\":{\"command\":\"unused\"}}}", "test") : null,
                null, null, null);
    }

    private static ApiKeys keys(boolean enabled) {
        Properties properties = new Properties();
        if (enabled) {
            properties.setProperty(ApiKeys.PROPERTY, "test-only-key");
        }
        return ApiKeys.from(Map.of(), properties);
    }

    private static ToolsSecurityFilter filter(boolean tools, boolean mcp, String allowed) {
        return new ToolsSecurityFilter(options(tools, mcp), new AllowedToolHosts("127.0.0.1", allowed), keys(false));
    }

    /** A policy of one origin: {@code *} when {@code everyOrigin}, a single site otherwise. */
    private static CorsPolicy cors(boolean everyOrigin) {
        return CorsPolicy.of(true, List.of(everyOrigin ? "*" : "http://ui.example"), true);
    }

    @Test
    void foreignHostsAreRefusedForToolsAndEveryToolBearingApiAlias() {
        ToolsSecurityFilter filter = filter(true, false, null);
        for (String path : List.of("tools", "/tools/", "tools;ignored=yes", "v1/../tools")) {
            assertForbidden(new Request("POST", path, SERVER, FOREIGN).run(filter));
            assertForbidden(new Request("GET", path, "", FOREIGN).run(filter));
        }
        for (String path : List.of("v1/chat/completions", "chat/completions", "v1/responses", "responses",
                "v1/messages", "v1/chat/completions/batch", "v1/chat/completions/input_tokens",
                "chat/completions/input_tokens", "v1/responses/input_tokens", "responses/input_tokens",
                "v1/messages/count_tokens", "v1;ignored=yes/chat/completions")) {
            assertForbidden(new Request("POST", path, SERVER, FOREIGN).run(filter));
            assertForbidden(new Request("POST", path, MCP, FOREIGN).run(filter));
        }
    }

    @Test
    void allowedHostsPassAndOtherSurfacesStayUntouched() {
        ToolsSecurityFilter filter = filter(true, false, "proxy.example");
        for (String host : List.of("localhost:8080", "127.2.3.4:1234", "[::1]:8080", "proxy.example")) {
            Request request = new Request("POST", "v1/chat/completions", SERVER, host);
            assertNull(request.run(filter));
            assertEquals(0, request.streamReads.get());
        }
        for (String path : List.of("v1/models", "models", "health", "v1/completions", "ui/app.js")) {
            Request request = new Request("GET", path, SERVER, FOREIGN);
            assertNull(request.run(filter));
            assertEquals(0, request.streamReads.get());
        }
        assertNull(new Request("OPTIONS", "tools", "", FOREIGN).run(filter));
    }

    @Test
    void inspectingAClientToolRequestPreservesItsExactBody() throws IOException {
        String body = " {\"messages\":[],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"server\"}}],"
                + "\"extra\":\"λ\"}\n";
        Request request = new Request("POST", "v1/chat/completions", body, FOREIGN);
        assertNull(request.run(filter(true, false, null)));
        assertEquals(1, request.streamReads.get());
        assertEquals(body, new String(request.entity.get().readAllBytes(), StandardCharsets.UTF_8));
        assertNull(new Request("POST", "v1/responses", "{\"input\":\"hello\"}", FOREIGN)
                .run(filter(true, false, null)));
    }

    @Test
    void toolDetectionUsesParsedJsonAndMalformedJsonIsRefused() {
        String escaped = "{\"tools\":[{\"type\":\"ser\\u0076er\",\"tool\":\"get_info\"}]}";
        assertForbidden(new Request("POST", "v1/chat/completions", escaped, FOREIGN)
                .run(filter(true, false, null)));
        assertEquals(400, new Request("POST", "v1/chat/completions", "{", FOREIGN)
                .run(filter(true, false, null)).getStatus());
    }

    @Test
    void aForeignHostNonJsonBodyIsLeftToTheConsumesCheck() {
        List<MediaType> types = new ArrayList<>();
        types.add(null);
        types.add(MediaType.TEXT_PLAIN_TYPE);
        types.add(new MediaType("application", "x-www-form-urlencoded"));
        for (MediaType type : types) {
            Request request = new Request("POST", "v1/chat/completions", SERVER, type, FOREIGN);
            assertNull(request.run(filter(true, false, null)));
            assertEquals(0, request.streamReads.get());
        }
    }

    @Test
    void mcpAloneEnablesTheGuardAndKeysDoNotDisableIt() {
        assertForbidden(new Request("POST", "tools", "{}", FOREIGN).run(filter(false, true, null)));
        ToolsSecurityFilter keyed = new ToolsSecurityFilter(options(true, false),
                new AllowedToolHosts("127.0.0.1", null), keys(true));
        assertForbidden(new Request("POST", "v1/chat/completions", SERVER, FOREIGN).run(keyed));
    }

    @Test
    void aBrowserOriginOnAnotherSiteCannotDriveTheToolsOfALoopbackHost() {
        ToolsSecurityFilter filter = filter(true, false, null);
        for (String origin : new String[]{"http://attacker.example", "https://127.0.0.1.attacker.example",
                "null", "http://localhost/path", "http://user@localhost", "file://", "ftp://localhost"}) {
            assertForbidden(new Request("POST", "tools", "{}", "localhost").withOrigin(origin).run(filter));
            assertForbidden(new Request("POST", "v1/chat/completions", SERVER, "127.0.0.1").withOrigin(origin)
                    .run(filter));
        }
        assertForbidden(new Request("GET", "tools", "", "localhost")
                .withOrigin("http://localhost", "http://attacker.example").run(filter));
    }

    @Test
    void aSameSiteOrALoopbackOriginKeepsTheTools() {
        ToolsSecurityFilter filter = filter(true, false, "ui.example");
        for (String origin : new String[]{"http://localhost:8080", "https://127.0.0.1", "http://[::1]:5173",
                "https://ui.example"}) {
            assertNull(new Request("POST", "tools", "{}", "localhost").withOrigin(origin).run(filter));
        }
        Request ordinary = new Request("POST", "v1/chat/completions", CHAT_BODY, "localhost")
                .withOrigin("http://attacker.example");
        assertNull(ordinary.run(filter));
    }

    @Test
    void missingOrMultipleHostHeadersFailClosed() {
        ToolsSecurityFilter filter = filter(true, false, null);
        assertForbidden(new Request("GET", "tools", "").run(filter));
        assertForbidden(new Request("GET", "tools", "", "localhost", "attacker.example").run(filter));
        assertForbidden(new Request("GET", "tools", "", "localhost,attacker.example").run(filter));
    }

    @Test
    void disabledToolsAndAStarLeaveTheRequestAlone() {
        for (ToolsSecurityFilter filter : List.of(filter(false, false, null), filter(true, false, "*"))) {
            Request request = new Request("POST", "v1/chat/completions", SERVER, FOREIGN);
            assertNull(request.run(filter));
            assertEquals(0, request.streamReads.get());
            assertNull(new Request("GET", "tools", "", FOREIGN).run(filter));
        }
    }

    @Test
    void startupWarningsNameOnlyTheEnabledFeaturesAndMentionMissingKeys() {
        Logger logger = Logger.getLogger(ToolsSecurityFilter.class.getName());
        List<LogRecord> records = new ArrayList<>();
        Handler handler = new Handler() {
            @Override public void publish(LogRecord record) { records.add(record); }
            @Override public void flush() { }
            @Override public void close() { }
        };
        logger.addHandler(handler);
        try {
            for (boolean tools : List.of(false, true)) {
                for (boolean mcp : List.of(false, true)) {
                    for (boolean keyed : List.of(false, true)) {
                        records.clear();
                        ToolsSecurityFilter filter = new ToolsSecurityFilter(options(tools, mcp),
                                new AllowedToolHosts("127.0.0.1", null), keys(keyed));
                        filter.onStartup(new StartupEvent());
                        int expected = !tools && !mcp ? 0 : keyed ? 1 : 2;
                        assertEquals(expected, records.size());
                        if (expected == 0) {
                            continue;
                        }
                        String features = tools ? "server tools (experimental)" : "";
                        features += mcp ? (tools ? ", " : "") + "MCP servers (experimental)" : "";
                        assertEquals("security: " + features + " enabled - do not expose to untrusted environments",
                                records.get(0).getMessage());
                        assertTrue(records.stream().allMatch(r -> r.getLevel() == Level.WARNING));
                        if (!keyed) {
                            assertTrue(records.get(1).getMessage().contains("no API key"));
                            assertTrue(records.get(1).getMessage().contains(ApiKeys.PROPERTY));
                        }
                        new Request("GET", "tools", "", "localhost").run(filter);
                        assertEquals(expected, records.size(), "requests must not repeat startup warnings");
                    }
                }
            }
        } finally {
            logger.removeHandler(handler);
        }
    }

    @Test
    void startupWarnsAboutOpenCorsWithoutAKeyAndOnlyOnce() {
        Logger logger = Logger.getLogger(ToolsSecurityFilter.class.getName());
        List<LogRecord> records = new ArrayList<>();
        Handler handler = new Handler() {
            @Override public void publish(LogRecord record) { records.add(record); }
            @Override public void flush() { }
            @Override public void close() { }
        };
        logger.addHandler(handler);
        try {
            ToolsSecurityFilter open = new ToolsSecurityFilter(options(false, false),
                    new AllowedToolHosts("127.0.0.1", null), keys(false), cors(true));
            open.onStartup(new StartupEvent());
            assertEquals(1, records.size(), "an open CORS policy without a key is a warning of its own");
            assertEquals(Level.WARNING, records.get(0).getLevel());
            assertTrue(records.get(0).getMessage().contains("no API key"));
            assertTrue(records.get(0).getMessage().contains("CORS allows all origins"));
            new Request("GET", "v1/models", "", "localhost").run(open);
            assertEquals(1, records.size(), "requests must not repeat startup warnings");

            records.clear();
            new ToolsSecurityFilter(options(false, false), new AllowedToolHosts("127.0.0.1", null),
                    keys(true), cors(true)).onStartup(new StartupEvent());
            assertEquals(0, records.size(), "a configured key silences the open-CORS warning");

            records.clear();
            new ToolsSecurityFilter(options(false, false), new AllowedToolHosts("127.0.0.1", null),
                    keys(false), cors(false)).onStartup(new StartupEvent());
            assertEquals(0, records.size(), "a restricted origin list is not an open CORS policy");
        } finally {
            logger.removeHandler(handler);
        }
    }

    private static void assertForbidden(Response response) {
        assertEquals(403, response.getStatus());
        assertEquals(MediaType.APPLICATION_JSON_TYPE, response.getMediaType());
        var error = view(response.getEntity()).obj("error");
        assertEquals("invalid_request_error", error.str("type"));
        assertTrue(error.str("message").contains("neuralfree.server.allowed.hosts"));
    }

    /** A request with a real byte stream, recording stream reads and stream replacement. */
    private static final class Request {
        private final AtomicReference<InputStream> entity;
        private final AtomicInteger streamReads = new AtomicInteger();
        private final MediaType type;
        private final ContainerRequestContext context;
        private final MultivaluedHashMap<String, String> headers = new MultivaluedHashMap<>();

        Request(String method, String path, String body, String... hosts) {
            this(method, path, body, MediaType.APPLICATION_JSON_TYPE, hosts);
        }

        Request withOrigin(String... origins) {
            headers.put("Origin", List.of(origins));
            return this;
        }

        Request(String method, String path, String body, MediaType type, String... hosts) {
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            entity = new AtomicReference<>(new ByteArrayInputStream(bytes));
            this.type = type;
            headers.put(HttpHeaders.HOST, List.of(hosts));
            UriInfo uri = (UriInfo) Proxy.newProxyInstance(UriInfo.class.getClassLoader(),
                    new Class<?>[]{UriInfo.class}, (p, m, a) -> {
                        if (m.getName().equals("getPath")) {
                            return path;
                        }
                        throw new UnsupportedOperationException(m.getName());
                    });
            context = (ContainerRequestContext) Proxy.newProxyInstance(ContainerRequestContext.class.getClassLoader(),
                    new Class<?>[]{ContainerRequestContext.class}, (p, m, a) -> switch (m.getName()) {
                        case "getMethod" -> method;
                        case "getUriInfo" -> uri;
                        case "getHeaders" -> headers;
                        case "getMediaType" -> type;
                        case "getEntityStream" -> {
                            streamReads.incrementAndGet();
                            yield entity.get();
                        }
                        case "setEntityStream" -> { entity.set((InputStream) a[0]); yield null; }
                        default -> throw new UnsupportedOperationException(m.getName());
                    });
        }

        Response run(ToolsSecurityFilter filter) {
            return filter.filter(context);
        }
    }
}
