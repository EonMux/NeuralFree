package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * llama-server's CORS flags and vLLM's spellings, mapped onto the same Quarkus properties accepted by a {@code -D}
 * launch. llama.cpp's flags take comma-separated values; vLLM's {@code --allowed-origins}, {@code --allowed-methods}
 * and {@code --allowed-headers} take a JSON list of strings and {@code --allow-credentials} is a switch. Every value
 * passes the same validation whichever spelling names it.
 */
record CorsLaunchFlags(Map<String, String> values) {

    static final String ENABLED = "quarkus.http.cors.enabled";
    static final String ORIGINS = "quarkus.http.cors.origins";
    static final String METHODS = "quarkus.http.cors.methods";
    static final String HEADERS = "quarkus.http.cors.headers";
    static final String CREDENTIALS = "quarkus.http.cors.access-control-allow-credentials";
    static final String EXACT_ORIGINS = "quarkus.http.cors.return-exact-origins";
    static final String DEFAULT_METHODS = "GET, POST, DELETE, OPTIONS";
    static final String LOCALHOST_PATTERN = "/https?:\\/\\/(localhost|127\\.0\\.0\\.1|\\[::1\\])(:[0-9]+)?/";
    private static final Set<String> FLAGS = Set.of("--cors-origins", "--cors-methods", "--cors-headers",
            "--cors-credentials", "--no-cors-credentials",
            "--allowed-origins", "--allowed-methods", "--allowed-headers", "--allow-credentials");
    private static final Pattern TOKEN = Pattern.compile("[!#$%&'*+.^_`|~0-9A-Za-z-]+");

    CorsLaunchFlags {
        values = Map.copyOf(values);
    }

    CorsLaunchFlags() {
        this(Map.of());
    }

    static boolean handles(String flag) {
        return FLAGS.contains(flag);
    }

    CorsLaunchFlags with(String flag, String value) {
        String property = switch (flag) {
            case "--cors-origins", "--allowed-origins" -> ORIGINS;
            case "--cors-methods", "--allowed-methods" -> METHODS;
            case "--cors-headers", "--allowed-headers" -> HEADERS;
            case "--cors-credentials", "--no-cors-credentials", "--allow-credentials" -> CREDENTIALS;
            default -> throw new IllegalArgumentException("Unknown CORS flag '" + flag + "'");
        };
        String validated = switch (flag) {
            case "--cors-origins" -> origins(flag, value);
            case "--allowed-origins" -> origins(flag, jsonList(flag, value));
            case "--cors-methods", "--cors-headers" -> tokens(flag, value);
            case "--allowed-methods", "--allowed-headers" -> tokens(flag, jsonList(flag, value));
            default -> Boolean.toString(flag.equals("--cors-credentials") || flag.equals("--allow-credentials"));
        };
        Map<String, String> next = new LinkedHashMap<>(values);
        next.put(property, validated);
        return new CorsLaunchFlags(next);
    }

    /** Missing dimensions take llama.cpp's defaults; explicit system properties are not overwritten by defaults. */
    void resolveInto(Map<String, String> properties) {
        if (values.isEmpty()) {
            return;
        }
        properties.put(ENABLED, "true");
        properties.put(ORIGINS, values.getOrDefault(ORIGINS, System.getProperty(ORIGINS, "*")));
        properties.put(METHODS, values.getOrDefault(METHODS, System.getProperty(METHODS, DEFAULT_METHODS)));
        properties.put(HEADERS, values.getOrDefault(HEADERS, System.getProperty(HEADERS, "*")));
        properties.put(CREDENTIALS, values.getOrDefault(CREDENTIALS, System.getProperty(CREDENTIALS, "true")));
        boolean wildcard = properties.get(ORIGINS).equals("*");
        boolean credentials = Boolean.parseBoolean(properties.get(CREDENTIALS));
        properties.put(EXACT_ORIGINS, System.getProperty(EXACT_ORIGINS, Boolean.toString(!wildcard || credentials)));
    }

    private static String origins(String flag, String value) {
        if (value.equals("localhost")) {
            return LOCALHOST_PATTERN;
        }
        for (String origin : value.split(",", -1)) {
            if (origin.isBlank() || origin.chars().anyMatch(Character::isISOControl)) {
                throw new IllegalArgumentException(flag + " needs non-empty origins without control characters");
            }
        }
        return value.trim();
    }

    private static String tokens(String flag, String value) {
        for (String token : value.split(",", -1)) {
            if (!TOKEN.matcher(token.trim()).matches()) {
                throw new IllegalArgumentException(flag + " contains an invalid HTTP token '" + token + "'");
            }
        }
        return value.trim();
    }

    /**
     * The comma-separated value of vLLM's JSON-list flag ({@code --allowed-origins}, {@code --allowed-methods},
     * {@code --allowed-headers}), refused naming the flag when the value is not a non-empty JSON array of strings.
     */
    private static String jsonList(String flag, String json) {
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(flag + " is not a JSON list of strings: " + e.getMessage(), e);
        }
        if (!(parsed instanceof List<?> list)) {
            throw new IllegalArgumentException(flag + " is a JSON list of strings, got "
                    + (parsed == null ? "null" : parsed.getClass().getSimpleName()));
        }
        if (list.isEmpty()) {
            throw new IllegalArgumentException(flag + " needs at least one entry");
        }
        List<String> entries = new ArrayList<>(list.size());
        for (Object entry : list) {
            if (!(entry instanceof String text)) {
                throw new IllegalArgumentException(flag + " holds a non-string entry '" + entry + "'");
            }
            entries.add(text);
        }
        return String.join(",", entries);
    }
}
