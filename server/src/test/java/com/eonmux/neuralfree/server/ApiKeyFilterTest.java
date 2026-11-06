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
 * {@code x-api-key} headers, the open liveness paths and the 401 envelope.
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
        assertNull(run(filter, "OPTIONS", "v1/chat/completions", null, null));
        assertEquals(401, run(filter, "GET", "props", null, null).getStatus());
    }

    @Test
    void withoutKeysEverythingPasses() {
        assertNull(run(new ApiKeyFilter(keys(null, null)), "POST", "v1/completions", null, null));
    }

    /** Runs the filter over a request; the response it aborted with, or null when it passed. */
    private static Response run(ApiKeyFilter filter, String method, String path, String header, String value) {
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
                    case "getHeaderString" -> args[0].toString().equalsIgnoreCase(String.valueOf(header))
                            ? value : null;
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
