package com.eonmux.neuralfree.server;

import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.UriInfo;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * API-key authentication: the keys of the environment, the property and the key file, the Bearer and
 * {@code x-api-key} headers, the open liveness and bundled-UI paths and the 401 envelope.
 */
public class ApiKeyFilterTest {

    private static ApiKeys keys(String env, String property) {
        Map<String, String> environment = new HashMap<>();
        if (env != null) {
            environment.put(ApiKeys.ENV, env);
        }
        Properties properties = new Properties();
        if (property != null) {
            properties.setProperty(ApiKeys.PROPERTY, property);
        }
        return ApiKeys.from(environment, properties);
    }

    @Test
    void keysComeFromTheEnvironmentThePropertyAndTheFile(@TempDir Path dir) throws IOException {
        ApiKeys none = keys(null, null);
        assertFalse(none.enabled());
        ApiKeys both = keys("alpha, beta", "gamma");
        assertTrue(both.enabled());
        assertTrue(both.accepts("alpha") && both.accepts("beta") && both.accepts("gamma"));
        assertFalse(both.accepts("delta"));
        assertFalse(both.accepts(null));
        assertFalse(both.accepts("alph"), "a prefix is not the key");

        Path file = dir.resolve("keys.txt");
        Files.writeString(file, "# serving keys\n\nfile-key\n");
        Properties properties = new Properties();
        properties.setProperty(ApiKeys.FILE_PROPERTY, file.toString());
        ApiKeys fromFile = ApiKeys.from(Map.of(), properties);
        assertTrue(fromFile.accepts("file-key"));
        assertFalse(fromFile.accepts("# serving keys"));
    }

    @Test
    void misconfiguredKeysFailLoudly(@TempDir Path dir) {
        assertThrows(IllegalStateException.class, () -> keys("a,,b", null));
        Properties properties = new Properties();
        properties.setProperty(ApiKeys.FILE_PROPERTY, dir.resolve("missing.txt").toString());
        assertThrows(IllegalStateException.class, () -> ApiKeys.from(Map.of(), properties));
    }

    @Test
    void theFilterAdmitsTheKeyAndRefusesTheRest() {
        ApiKeyFilter filter = new ApiKeyFilter(keys("secret-1", null));
        assertNull(run(filter, "POST", "v1/chat/completions", "Authorization", "Bearer secret-1"));
        assertNull(run(filter, "POST", "v1/messages", "x-api-key", "secret-1"));
        assertNull(run(filter, "POST", "v1/chat/completions", "Authorization", "bearer secret-1"),
                "the scheme is case-insensitive");
        Response refused = run(filter, "POST", "v1/chat/completions", "Authorization", "Bearer wrong");
        assertEquals(401, refused.getStatus());
        assertTrue(String.valueOf(refused.getEntity()).contains("authentication_error"));
        assertEquals(401, run(filter, "GET", "v1/models", null, null).getStatus());
    }

    @Test
    void livenessPathsAndPreflightStayOpen() {
        ApiKeyFilter filter = new ApiKeyFilter(keys("secret-1", null));
        assertNull(run(filter, "GET", "health", null, null));
        assertNull(run(filter, "GET", "/v1/health", null, null));
        assertNull(run(filter, "GET", "q/health/live", null, null));
        assertNull(run(filter, "GET", "ready", null, null));
        assertNull(run(filter, "GET", "/readyz", null, null));
        assertNull(run(filter, "POST", "ping", null, null));
        assertEquals(401, run(filter, "GET", "server_info", null, null).getStatus());
        assertEquals(401, run(filter, "POST", "reset_prefix_cache", null, null).getStatus());
        assertEquals(401, run(filter, "POST", "abort_requests", null, null).getStatus());
        assertEquals(401, run(filter, "POST", "start_profile", null, null).getStatus(),
                "the profile window is an admin route like the cache resets");
        assertEquals(401, run(filter, "POST", "stop_profile", null, null).getStatus());
        assertNull(run(filter, "OPTIONS", "v1/chat/completions", null, null));
        assertEquals(401, run(filter, "GET", "props", null, null).getStatus());
    }

    @Test
    void theBundledUiStaysOpenWhileTheApiStaysKeyed() {
        ApiKeyFilter filter = new ApiKeyFilter(keys("secret-1", null));
        assertNull(run(filter, "GET", "", null, null), "the document at the root");
        assertNull(run(filter, "GET", "index.html", null, null));
        assertNull(run(filter, "GET", "ui", null, null));
        assertNull(run(filter, "GET", "ui/app.js", null, null));
        assertNull(run(filter, "GET", "/ui/app.css", null, null));
        assertEquals(401, run(filter, "GET", "uix/app.js", null, null).getStatus(),
                "the prefix matches the asset directory, not a name that starts with it");
        assertEquals(401, run(filter, "GET", "props", null, null).getStatus(),
                "the API endpoints the UI calls stay keyed");
    }

    @Test
    void disablingTheUiClosesAllItsPublicPathsIncludingUnderAPrefix() {
        for (String prefix : new String[]{null, "/llm"}) {
            ApiKeyFilter filter = new ApiKeyFilter(keys("secret-1", null), false, prefix);
            String base = prefix == null ? "" : prefix;
            for (String path : new String[]{"/", "/index.html", "/ui", "/ui/app.js"}) {
                assertEquals(401, run(filter, "GET", base + path, null, null).getStatus(), base + path);
                assertNull(run(filter, "GET", base + path, "Authorization", "Bearer secret-1"));
            }
            assertNull(run(filter, "GET", base + "/health", null, null));
        }
    }

    @Test
    void prefixedUiPathsStayOpenButTraversalNeverOpensApiPaths() {
        for (boolean enabled : new boolean[]{true, false}) {
            ApiKeyFilter filter = new ApiKeyFilter(keys("secret-1", null), enabled, "/llm");
            if (enabled) {
                assertNull(run(filter, "GET", "/llm/", null, null));
                assertNull(run(filter, "GET", "/llm/ui/app.js", null, null));
            }
            for (String path : new String[]{"/llm/props", "/llm/v1/models", "/llm/ui/../v1/models",
                    "/llm/index.html/../v1/models", "/llm/ui/a/../../v1/models"}) {
                assertEquals(401, run(filter, "GET", path, null, null).getStatus(), path);
            }
        }
    }

    @Test
    void withoutKeysEverythingPasses() {
        assertNull(run(new ApiKeyFilter(keys(null, null)), "POST", "v1/completions", null, null));
    }

    @Test
    void aRefusalCarriesTheCorsHeadersOfTheRequestOrigin() {
        CorsPolicy cors = CorsPolicy.of(true, java.util.List.of("http://ui.example"), true);
        ApiKeyFilter filter = new ApiKeyFilter(keys("secret-1", null), cors, true, null);

        Response allowed = run(filter, "GET", "server_info", headers("Origin", "http://ui.example"));
        assertEquals(401, allowed.getStatus());
        assertEquals("http://ui.example", allowed.getHeaderString("Access-Control-Allow-Origin"));
        assertEquals("true", allowed.getHeaderString("Access-Control-Allow-Credentials"));
        assertEquals("Origin", allowed.getHeaderString("Vary"));

        Response foreign = run(filter, "GET", "server_info", headers("Origin", "http://attacker.example"));
        assertEquals(401, foreign.getStatus());
        assertNull(foreign.getHeaderString("Access-Control-Allow-Origin"),
                "an origin the policy does not admit gets no header");

        Response noOrigin = run(filter, "GET", "server_info", headers("Authorization", "Bearer wrong"));
        assertEquals(401, noOrigin.getStatus());
        assertNull(noOrigin.getHeaderString("Access-Control-Allow-Origin"),
                "a request without Origin is not cross-origin");
        assertNull(noOrigin.getHeaderString("Access-Control-Allow-Credentials"));
    }

    private static Map<String, String> headers(String name, String value) {
        Map<String, String> headers = new HashMap<>();
        headers.put(name, value);
        return headers;
    }

    /** Runs the filter over a request; the response it aborted with, or null when it passed. */
    private static Response run(ApiKeyFilter filter, String method, String path, String header, String value) {
        Map<String, String> headers = new HashMap<>();
        if (header != null) {
            headers.put(header, value);
        }
        return run(filter, method, path, headers);
    }

    private static Response run(ApiKeyFilter filter, String method, String path, Map<String, String> headers) {
        AtomicReference<Response> aborted = new AtomicReference<>();
        UriInfo uri = (UriInfo) Proxy.newProxyInstance(UriInfo.class.getClassLoader(),
                new Class<?>[]{UriInfo.class}, (proxy, m, args) -> {
                    if (m.getName().equals("getPath")) {
                        return path;
                    }
                    throw new UnsupportedOperationException(m.getName());
                });
        ContainerRequestContext request = (ContainerRequestContext) Proxy.newProxyInstance(
                ContainerRequestContext.class.getClassLoader(), new Class<?>[]{ContainerRequestContext.class},
                (proxy, m, args) -> switch (m.getName()) {
                    case "getMethod" -> method;
                    case "getUriInfo" -> uri;
                    case "getHeaderString" -> headers.entrySet().stream()
                            .filter(entry -> entry.getKey().equalsIgnoreCase(args[0].toString()))
                            .map(Map.Entry::getValue).findFirst().orElse(null);
                    case "abortWith" -> {
                        aborted.set((Response) args[0]);
                        yield null;
                    }
                    default -> throw new UnsupportedOperationException(m.getName());
                });
        filter.filter(request);
        return aborted.get();
    }
}
