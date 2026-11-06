package com.eonmux.neuralfree.demo.tools.eval;

import java.net.URI;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * One OpenAI-compatible endpoint of an evaluation run: the base URL, the number of requests it runs
 * concurrently, its display name, and the bearer token every request carries. The driver assigns
 * every task of the shared queue to whichever worker of whichever server takes it, so a run spreads
 * over all of its servers.
 */
public final class EvalServer {

    private final String url;
    private final int threads;
    private final String name;
    private final String apiKey;

    /**
     * @param url     the server's base URL, without the {@code /v1} suffix
     * @param threads the requests this server runs concurrently, at least 1
     * @param name    the display name, empty to use the URL
     * @param apiKey  the bearer token, null for an open server
     * @throws IllegalArgumentException naming a blank URL or a non-positive thread count
     */
    public EvalServer(String url, int threads, String name, String apiKey) {
        if (url == null || url.trim().isEmpty()) {
            throw new IllegalArgumentException("--server must name a server URL");
        }
        if (threads < 1) {
            throw new IllegalArgumentException("--threads must be at least 1, got " + threads);
        }
        String base = url.trim();
        while (base.endsWith("/")) {
            base = base.substring(0, base.length() - 1);
        }
        if (base.isEmpty()) {
            throw new IllegalArgumentException("--server '" + url + "' is not a server URL");
        }
        host(base);
        this.url = base;
        this.threads = threads;
        this.name = name == null || name.isEmpty() ? base : name;
        this.apiKey = apiKey == null || apiKey.isEmpty() ? null : apiKey;
    }

    /** The base URL, trailing slashes removed. */
    public String url() {
        return url;
    }

    /** The requests this server runs concurrently. */
    public int threads() {
        return threads;
    }

    /** The display name of the server, the URL when none was given. */
    public String name() {
        return name;
    }

    /** The bearer token, null for an open server. */
    public String apiKey() {
        return apiKey;
    }

    /**
     * The headers of one request: JSON content, the bearer token when configured, and — for an
     * opencode gateway only — the session id the task's turns share.
     */
    public Map<String, String> headers(String sessionId) {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("Content-Type", "application/json");
        if (apiKey != null) {
            headers.put("Authorization", "Bearer " + apiKey);
        }
        if (sessionId != null && isOpencode()) {
            headers.put("x-opencode-session", sessionId);
        }
        return headers;
    }

    /** Whether the host is an opencode gateway rather than a llama-server. */
    boolean isOpencode() {
        return host(url).toLowerCase(Locale.ROOT).contains("opencode");
    }

    /** The URL's host; a URL without one is refused by name. */
    private static String host(String url) {
        try {
            String host = URI.create(url).getHost();
            if (host == null || host.isEmpty()) {
                throw new IllegalArgumentException("--server '" + url + "' carries no host");
            }
            return host;
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("--server '" + url + "' is not a URL: "
                    + e.getMessage(), e);
        }
    }

    /** The {@code /v1/models} address of this server. */
    String modelsUrl() {
        return url + "/v1/models";
    }

    /** The chat completions address of this server. */
    String chatUrl() {
        return url + "/v1/chat/completions";
    }
}
