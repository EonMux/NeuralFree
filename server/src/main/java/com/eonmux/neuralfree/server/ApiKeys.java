package com.eonmux.neuralfree.server;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * The API keys the server accepts (llama.cpp {@code --api-key} / {@code --api-key-file}, vLLM {@code --api-key}):
 * comma-separated keys from the {@code NEURALFREE_API_KEY} environment variable and the
 * {@code neuralfree.server.api.key} property, plus one key per line of the file named by
 * {@code neuralfree.server.api.key.file} (blank lines and {@code #} comments skipped). No key configured means no
 * authentication. Keys compare in constant time. Immutable.
 */
final class ApiKeys {

    static final String ENV = "NEURALFREE_API_KEY";
    static final String PROPERTY = "neuralfree.server.api.key";
    static final String FILE_PROPERTY = "neuralfree.server.api.key.file";

    /** The message of a refusal for a missing or wrong key, in the HTTP envelope and the gRPC status. */
    static final String REFUSAL_MESSAGE = "Invalid API Key";

    private static final String BEARER = "Bearer ";

    private final List<byte[]> keys;

    private ApiKeys(List<byte[]> keys) {
        this.keys = Collections.unmodifiableList(keys);
    }

    /**
     * The keys of {@code env} and {@code properties}.
     *
     * @throws IllegalStateException when the key file cannot be read or a configured key is blank
     */
    static ApiKeys from(Map<String, String> env, Properties properties) {
        List<byte[]> keys = new ArrayList<>();
        addList(keys, env.get(ENV), ENV);
        addList(keys, properties.getProperty(PROPERTY), PROPERTY);
        String file = properties.getProperty(FILE_PROPERTY);
        if (file != null && !file.isBlank()) {
            try {
                for (String line : Files.readAllLines(Path.of(file.trim()), StandardCharsets.UTF_8)) {
                    String key = line.trim();
                    if (!key.isEmpty() && !key.startsWith("#")) {
                        keys.add(key.getBytes(StandardCharsets.UTF_8));
                    }
                }
            } catch (IOException e) {
                throw new IllegalStateException(FILE_PROPERTY + " names a file that cannot be read: " + file, e);
            }
        }
        return new ApiKeys(keys);
    }

    private static void addList(List<byte[]> keys, String value, String source) {
        if (value == null) {
            return;
        }
        for (String part : value.split(",")) {
            String key = part.trim();
            if (key.isEmpty()) {
                throw new IllegalStateException(source + " holds a blank API key");
            }
            keys.add(key.getBytes(StandardCharsets.UTF_8));
        }
    }

    /** Whether any key is configured (authentication is on). */
    boolean enabled() {
        return !keys.isEmpty();
    }

    /**
     * The key a request presents: the {@code Authorization: Bearer} token, else the {@code x-api-key} header
     * (Anthropic), or null with neither.
     */
    static String presented(String authorization, String xApiKey) {
        if (authorization != null && authorization.regionMatches(true, 0, BEARER, 0, BEARER.length())) {
            return authorization.substring(BEARER.length()).trim();
        }
        return xApiKey == null ? null : xApiKey.trim();
    }

    /** The refusal body of a request without an accepted key. */
    static String refusalBody() {
        return ServerJson.write(ServerJson.errorBody(REFUSAL_MESSAGE, "authentication_error"));
    }

    /** Whether {@code presented} is one of the keys; every key is compared, in constant time each. */
    boolean accepts(String presented) {
        if (presented == null) {
            return false;
        }
        byte[] given = presented.getBytes(StandardCharsets.UTF_8);
        boolean match = false;
        for (byte[] key : keys) {
            match |= MessageDigest.isEqual(key, given);
        }
        return match;
    }
}
