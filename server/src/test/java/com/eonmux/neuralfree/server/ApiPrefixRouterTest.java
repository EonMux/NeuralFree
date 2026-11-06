package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@code --api-prefix} value: canonical forms of {@code api}, {@code /api} and {@code /api/} are one, unusable
 * prefixes refuse naming the property, and the request paths strip their prefix so the guards see the bare routes.
 */
public class ApiPrefixRouterTest {

    @Test
    void prefixSpellingsCanonicalize() {
        assertEquals("/api", ApiPrefixRouter.canonical(ApiPrefixRouter.PROPERTY, "api"));
        assertEquals("/api", ApiPrefixRouter.canonical(ApiPrefixRouter.PROPERTY, "/api"));
        assertEquals("/api", ApiPrefixRouter.canonical(ApiPrefixRouter.PROPERTY, "/api/"));
        assertEquals("/api/v1", ApiPrefixRouter.canonical(ApiPrefixRouter.PROPERTY, "api/v1"));
        assertEquals("/llm", ApiPrefixRouter.fromProperties(Map.of(ApiPrefixRouter.PROPERTY, "llm")));
        assertNull(ApiPrefixRouter.fromProperties(Map.of()), "without the property there is no prefix");
        assertNull(ApiPrefixRouter.fromProperties(Map.of(ApiPrefixRouter.PROPERTY, "  ")));
    }

    @Test
    void unusablePrefixesRefuseNamingTheProperty() {
        for (String value : new String[]{"", "/api b", "a?b", "a#b", "..", "/..", "/api/../v1",
                "a".repeat(129)}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> ApiPrefixRouter.canonical(ApiPrefixRouter.PROPERTY, value), value);
            assertTrue(e.getMessage().startsWith(ApiPrefixRouter.PROPERTY), value);
        }
    }

    @Test
    void requestPathsStripTheirPrefix() {
        assertEquals("/", ApiPrefixRouter.stripped("/api", "/api"));
        assertEquals("/", ApiPrefixRouter.stripped("/api/", "/api"));
        assertEquals("/health", ApiPrefixRouter.stripped("/api/health", "/api"));
        assertEquals("/v1/models", ApiPrefixRouter.stripped("/api/v1/models", "/api"));
        assertEquals("/other", ApiPrefixRouter.stripped("/other", "/api"), "a path under no prefix stays itself");
        assertEquals("/apix", ApiPrefixRouter.stripped("/apix", "/api"), "the prefix matches whole segments");
        assertEquals("/api", ApiPrefixRouter.stripped("/api", null));
        assertEquals("/api", ApiPrefixRouter.stripped("/api", ""));
    }

    @Test
    void thePropertyReadsFromSystemPropertiesLikeADLaunch() {
        Map<String, String> launch = new HashMap<>();
        launch.put(ApiPrefixRouter.PROPERTY, "/api/v1");
        assertEquals("/api/v1", ApiPrefixRouter.fromProperties(launch));
    }
}
