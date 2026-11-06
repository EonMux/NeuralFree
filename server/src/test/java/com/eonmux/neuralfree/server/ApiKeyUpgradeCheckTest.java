package com.eonmux.neuralfree.server;

import io.quarkus.websockets.next.HttpUpgradeCheck.CheckResult;
import io.quarkus.websockets.next.HttpUpgradeCheck.HttpUpgradeContext;
import io.vertx.core.MultiMap;
import io.vertx.core.http.HttpServerRequest;
import jakarta.ws.rs.core.HttpHeaders;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Proxy;
import java.util.Map;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The API key of WebSocket upgrades: the keyed HTTP routes' rule applied to an upgrade request, whose key forms and
 * refusal are the filter's. The check runs over proxy requests so the test needs no server.
 */
public class ApiKeyUpgradeCheckTest {

    @Test
    void upgradesAreOpenWhenNoKeyIsConfigured() {
        ApiKeyUpgradeCheck check = new ApiKeyUpgradeCheck(ApiKeys.from(Map.of(), new Properties()));
        CheckResult permitted = result(check, null, null);
        assertTrue(permitted.isUpgradePermitted());
        assertNull(permitted.getResponseHeaders().get(HttpHeaders.WWW_AUTHENTICATE));
    }

    @Test
    void bothKeyHeaderFormsAdmitTheUpgrade() {
        ApiKeyUpgradeCheck check = check("secret-1");
        assertTrue(result(check, "Bearer secret-1", null).isUpgradePermitted());
        assertTrue(result(check, "bearer secret-1", null).isUpgradePermitted(), "the scheme is case-insensitive");
        assertTrue(result(check, null, "secret-1").isUpgradePermitted());
    }

    @Test
    void aMissingOrWrongKeyRefusesTheUpgradeWithTheFilterEnvelope() {
        ApiKeyUpgradeCheck check = check("secret-1");
        for (String authorization : new String[] {null, "Bearer wrong", "secret-1"}) {
            CheckResult refused = result(check, authorization, null);
            assertFalse(refused.isUpgradePermitted(), "authorization " + authorization);
            assertEquals(401, refused.getHttpResponseCode());
            assertEquals(java.util.List.of("Bearer"), refused.getResponseHeaders().get(HttpHeaders.WWW_AUTHENTICATE));
        }
        assertFalse(result(check, null, "wrong").isUpgradePermitted());
    }

    private static ApiKeyUpgradeCheck check(String key) {
        return new ApiKeyUpgradeCheck(ApiKeys.from(Map.of(ApiKeys.ENV, key), new Properties()));
    }

    private static CheckResult result(ApiKeyUpgradeCheck check, String authorization, String xApiKey) {
        return check.perform(context(authorization, xApiKey)).await().indefinitely();
    }

    private static HttpUpgradeContext context(String authorization, String xApiKey) {
        MultiMap headers = MultiMap.caseInsensitiveMultiMap();
        if (authorization != null) {
            headers.add(HttpHeaders.AUTHORIZATION, authorization);
        }
        if (xApiKey != null) {
            headers.add("x-api-key", xApiKey);
        }
        HttpServerRequest request = proxy(HttpServerRequest.class,
                (method) -> method.equals("headers") ? headers : null);
        return proxy(HttpUpgradeContext.class,
                (method) -> method.equals("httpRequest") ? request : null);
    }

    /**
     * A proxy of {@code type} whose methods answer {@code answer}: the method name and its fixed answer, or null
     * for the remaining methods and the primitives' defaults.
     */
    private static <T> T proxy(Class<T> type, JavaLangAnswer answer) {
        return type.cast(Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[] {type}, (self, method, args) -> {
            Object fixed = answer.of(method.getName());
            if (fixed != null) {
                return fixed;
            }
            return switch (method.getReturnType().getName()) {
                case "boolean" -> false;
                case "int", "short", "byte" -> 0;
                case "long" -> 0L;
                case "float" -> 0f;
                case "double" -> 0d;
                case "char" -> (char) 0;
                default -> null;
            };
        }));
    }

    @FunctionalInterface
    private interface JavaLangAnswer {
        Object of(String method);
    }
}
