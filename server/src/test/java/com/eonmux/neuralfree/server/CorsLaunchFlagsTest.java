package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The CORS launch flags resolve to the Quarkus CORS properties with llama.cpp's defaults: origins {@code *},
 * methods {@code GET, POST, DELETE, OPTIONS}, headers {@code *}, credentials on, and with credentials the wildcard
 * origin echoes the request's Origin ({@code quarkus.http.cors.return-exact-origins}), while without credentials
 * the wildcard answers the literal {@code *}. Unspecified dimensions never overwrite an explicit system property.
 */
public class CorsLaunchFlagsTest {

    private static String refusalOf(Runnable call) {
        IllegalArgumentException e = org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                call::run);
        return e.getMessage();
    }

    @Test
    void everyFlagMapsToItsQuarkusProperty() {
        CorsLaunchFlags cors = new CorsLaunchFlags();
        cors = cors.with("--cors-origins", "http://a.example,http://b.example")
                .with("--cors-methods", "GET,PUT")
                .with("--cors-headers", "X-Custom,Content-Type")
                .with("--no-cors-credentials", "");
        Map<String, String> properties = new HashMap<>();
        cors.resolveInto(properties);
        assertEquals("http://a.example,http://b.example",
                properties.get(CorsLaunchFlags.ORIGINS));
        assertEquals("GET,PUT", properties.get(CorsLaunchFlags.METHODS));
        assertEquals("X-Custom,Content-Type", properties.get(CorsLaunchFlags.HEADERS));
        assertEquals("false", properties.get(CorsLaunchFlags.CREDENTIALS));
        assertEquals("true", properties.get(CorsLaunchFlags.ENABLED));
    }

    @Test
    void missingDimensionsTakeLlamaCppsDefaults() {
        Map<String, String> properties = new HashMap<>();
        new CorsLaunchFlags().with("--cors-origins", "*").resolveInto(properties);
        assertEquals("*", properties.get(CorsLaunchFlags.ORIGINS));
        assertEquals("GET, POST, DELETE, OPTIONS", properties.get(CorsLaunchFlags.METHODS),
                "llama.cpp's default method list, spaces included");
        assertEquals("*", properties.get(CorsLaunchFlags.HEADERS));
        assertEquals("true", properties.get(CorsLaunchFlags.CREDENTIALS));
    }

    @Test
    void theWildcardOriginEchoesWithCredentialsAndAnswersLiterallyWithout() {
        Map<String, String> withCredentials = new HashMap<>();
        new CorsLaunchFlags().with("--cors-origins", "*").with("--cors-credentials", "").resolveInto(withCredentials);
        assertEquals("true", withCredentials.get(CorsLaunchFlags.EXACT_ORIGINS),
                "the wildcard echoes the Origin so credentials stay usable, as llama.cpp does");
        Map<String, String> withoutCredentials = new HashMap<>();
        new CorsLaunchFlags().with("--no-cors-credentials", "").resolveInto(withoutCredentials);
        assertEquals("false", withoutCredentials.get(CorsLaunchFlags.EXACT_ORIGINS),
                "the wildcard answers the literal * when credentials are off, as llama.cpp does");
        assertEquals("false", withoutCredentials.get(CorsLaunchFlags.CREDENTIALS));
    }

    @Test
    void theLocalhostOriginBecomesALocalhostOnlyPattern() {
        Map<String, String> properties = new HashMap<>();
        new CorsLaunchFlags().with("--cors-origins", "localhost").resolveInto(properties);
        assertEquals(CorsLaunchFlags.LOCALHOST_PATTERN, properties.get(CorsLaunchFlags.ORIGINS));
        assertTrue(properties.get(CorsLaunchFlags.ORIGINS).contains("localhost"));
    }

    @Test
    void explicitSystemPropertiesSurviveTheDefaults() {
        String saved = System.getProperty(CorsLaunchFlags.METHODS);
        try {
            System.setProperty(CorsLaunchFlags.METHODS, "PUT");
            Map<String, String> properties = new HashMap<>();
            new CorsLaunchFlags().with("--cors-origins", "http://a.example").resolveInto(properties);
            assertEquals("PUT", properties.get(CorsLaunchFlags.METHODS),
                    "a -D the launch already carries is not overwritten by a default");
        } finally {
            if (saved == null) {
                System.clearProperty(CorsLaunchFlags.METHODS);
            } else {
                System.setProperty(CorsLaunchFlags.METHODS, saved);
            }
        }
    }

    @Test
    void unusableValuesAreRefusedNamingTheFlag() {
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--cors-origins", "a,,b")).contains("--cors-origins"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--cors-origins", "a\nb")).contains("--cors-origins"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--cors-methods", "GET PUT")).contains("--cors-methods"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--cors-headers", "a;b")).contains("--cors-headers"));
    }

    @Test
    void vllmsSpellingsTakeJsonListsAndMapToTheSameProperties() {
        CorsLaunchFlags cors = new CorsLaunchFlags()
                .with("--allowed-origins", "[\"http://a.example\", \"http://b.example\"]")
                .with("--allowed-methods", "[\"GET\", \"PUT\"]")
                .with("--allowed-headers", "[\"X-Custom\", \"Content-Type\"]")
                .with("--allow-credentials", "");
        Map<String, String> properties = new HashMap<>();
        cors.resolveInto(properties);
        assertEquals("http://a.example,http://b.example", properties.get(CorsLaunchFlags.ORIGINS));
        assertEquals("GET,PUT", properties.get(CorsLaunchFlags.METHODS));
        assertEquals("X-Custom,Content-Type", properties.get(CorsLaunchFlags.HEADERS));
        assertEquals("true", properties.get(CorsLaunchFlags.CREDENTIALS));
        assertEquals("true", properties.get(CorsLaunchFlags.EXACT_ORIGINS));
        assertEquals("true", properties.get(CorsLaunchFlags.ENABLED));
    }

    @Test
    void vllmAliasesPassTheSameValidationAndAreRefusedByName() {
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-origins", "http://a.example"))
                .contains("--allowed-origins"), "vLLM's flag takes a JSON list, not a bare origin");
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-origins", "[1]"))
                .contains("--allowed-origins"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-origins", "[]"))
                .contains("--allowed-origins"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-origins", "[\"a,,b\"]"))
                .contains("--allowed-origins"), "the origin list's empty entry refuses like --cors-origins");
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-origins", "[\"a\\nb\"]"))
                .contains("--allowed-origins"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-methods", "[\"GET PUT\"]"))
                .contains("--allowed-methods"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-headers", "[\"a;b\"]"))
                .contains("--allowed-headers"));
        assertTrue(refusalOf(() -> new CorsLaunchFlags().with("--allowed-headers", "{}"))
                .contains("--allowed-headers"));
    }
}
