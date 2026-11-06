package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import io.quarkus.test.common.QuarkusTestResource;
import io.quarkus.test.common.QuarkusTestResourceLifecycleManager;
import io.quarkus.test.common.http.TestHTTPResource;
import io.quarkus.test.junit.QuarkusTest;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpClient;
import io.vertx.core.http.HttpMethod;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Real Quarkus HTTP requests with forged Host headers over the real engine and a synthetic model. */
@QuarkusTest
@QuarkusTestResource(value = ToolsSecurityHttpTest.Launch.class, restrictToAnnotatedClass = true)
class ToolsSecurityHttpTest {

    @TestHTTPResource
    URI base;

    private static Vertx vertx;
    private static HttpClient client;

    private static final String CHAT = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
            + "\"max_tokens\":2,\"temperature\":0}";
    private static final String SERVER = "{\"messages\":[],\"tools\":[{\"type\":\"server\",\"tool\":\"get_info\"}]}";

    @BeforeAll
    static void open() {
        vertx = Vertx.vertx();
        client = vertx.createHttpClient();
    }

    @AfterAll
    static void close() throws Exception {
        vertx.close().toCompletionStage().toCompletableFuture().get(10, TimeUnit.SECONDS);
    }

    @Test
    void allowedHostCanInvokeAToolAndForeignHostsCannot() throws Exception {
        Reply allowed = send(HttpMethod.POST, "/tools", "localhost:8080", "{\"tool\":\"get_info\"}");
        assertEquals(200, allowed.status(), allowed.body());
        assertTrue(view(allowed.body()).has("os"));
        assertForbidden(send(HttpMethod.POST, "/tools", "attacker.example", "{\"tool\":\"get_info\"}"));
        assertForbidden(send(HttpMethod.GET, "/tools", "attacker.example", ""));
        assertEquals(200, send(HttpMethod.GET, "/tools", "trusted-proxy.example", "").status());
    }

    @Test
    void foreignToolRequestsAreBlockedAcrossTheApiFamiliesAndAliases() throws Exception {
        for (String path : new String[]{"/v1/chat/completions", "/chat/completions", "/v1/responses",
                "/responses", "/v1/messages", "/v1;ignored=yes/chat/completions"}) {
            assertForbidden(send(HttpMethod.POST, path, "attacker.example", SERVER));
            assertForbidden(send(HttpMethod.POST, path, "attacker.example", SERVER.replace("server", "mcp")));
        }
        assertForbidden(send(HttpMethod.GET, "/tools;ignored=yes", "attacker.example", ""));
    }

    @Test
    void aForeignHostStillReachesModelsAndOrdinaryChatWithTheOriginalBody() throws Exception {
        Reply models = send(HttpMethod.GET, "/v1/models", "attacker.example", "");
        assertEquals(200, models.status(), models.body());
        assertEquals("list", view(models.body()).str("object"));
        Reply foreign = send(HttpMethod.POST, "/v1/chat/completions", "attacker.example", CHAT);
        Reply local = send(HttpMethod.POST, "/v1/chat/completions", "localhost", CHAT);
        assertEquals(200, foreign.status(), foreign.body());
        assertEquals(200, local.status(), local.body());
        assertEquals(view(local.body()).raw("choices"), view(foreign.body()).raw("choices"));
    }

    @Test
    void aForeignBrowserOriginCannotDriveTheToolsThroughTheOpenCors() throws Exception {
        Reply foreign = sendFrom("http://attacker.example", "/tools", "{\"tool\":\"get_info\"}");
        assertForbidden(foreign);
        Reply foreignChat = sendFrom("http://attacker.example", "/v1/chat/completions", SERVER);
        assertForbidden(foreignChat);
        Reply local = sendFrom("http://localhost:" + base.getPort(), "/tools", "{\"tool\":\"get_info\"}");
        assertEquals(200, local.status(), local.body());
        Reply ordinary = sendFrom("http://attacker.example", "/v1/chat/completions", CHAT);
        assertEquals(200, ordinary.status(), ordinary.body());
    }

    private Reply sendFrom(String origin, String path, String body) throws Exception {
        return client.request(HttpMethod.POST, base.getPort(), "127.0.0.1", path)
                .compose(request -> request.putHeader("Host", "127.0.0.1:" + base.getPort())
                        .putHeader("Origin", origin).putHeader("Content-Type", "application/json").send(body))
                .compose(response -> response.body().map(bytes -> new Reply(response.statusCode(), bytes.toString())))
                .toCompletionStage().toCompletableFuture().get(15, TimeUnit.SECONDS);
    }

    @Test
    void forwardedHostCannotOverrideTheActualHostHeader() throws Exception {
        var reply = client.request(HttpMethod.POST, base.getPort(), "127.0.0.1", "/tools")
                .compose(request -> request.putHeader("Host", "attacker.example")
                        .putHeader("X-Forwarded-Host", "localhost")
                        .putHeader("Forwarded", "host=localhost")
                        .putHeader("Content-Type", "application/json").send("{\"tool\":\"get_info\"}"))
                .compose(response -> response.body().map(body -> new Reply(response.statusCode(), body.toString())))
                .toCompletionStage().toCompletableFuture().get(15, TimeUnit.SECONDS);
        assertForbidden(reply);
    }

    private Reply send(HttpMethod method, String path, String host, String body) throws Exception {
        return client.request(method, base.getPort(), "127.0.0.1", path)
                .compose(request -> request.putHeader("Host", host).putHeader("Content-Type", "application/json")
                        .send(body))
                .compose(response -> response.body().map(bytes -> new Reply(response.statusCode(), bytes.toString())))
                .toCompletionStage().toCompletableFuture().get(15, TimeUnit.SECONDS);
    }

    private static void assertForbidden(Reply reply) {
        assertEquals(403, reply.status(), reply.body());
        var error = view(reply.body()).obj("error");
        assertEquals("invalid_request_error", error.str("type"));
        assertTrue(error.str("message").contains("neuralfree.server.allowed.hosts"));
    }

    private record Reply(int status, String body) { }

    /** Installs launch properties before Quarkus startup and restores them after the test server stops. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        private final Map<String, String> backups = new LinkedHashMap<>();
        private Path model;

        @Override
        public Map<String, String> start() {
            try {
                model = SyntheticModels.llama().writeTempFile();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            Map<String, String> properties = Map.of(
                    "neuralfree.server.model", model.toString(),
                    "neuralfree.backend", "cpu",
                    "neuralfree.ctx", "128",
                    "neuralfree.kv.paged", "true",
                    "neuralfree.parallel", "1",
                    ServerOptions.TOOLS_PROPERTY, "get_info",
                    "neuralfree.server.allowed.hosts", "trusted-proxy.example");
            properties.forEach((name, value) -> backups.put(name, System.setProperty(name, value)));
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }

        @Override
        public void stop() {
            backups.forEach((name, value) -> {
                if (value == null) {
                    System.clearProperty(name);
                } else {
                    System.setProperty(name, value);
                }
            });
            if (model != null) {
                try {
                    Files.deleteIfExists(model);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        }
    }
}
