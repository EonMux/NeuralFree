package com.eonmux.neuralfree.server;

import java.time.Duration;
import java.util.Objects;

/**
 * The server's own settings, beside the engine's {@code EngineConfig}: the served model names ({@link ServedModels}),
 * the request budget, the longest one request may wait on the engine before it is cancelled and refused with 503
 * (llama.cpp's {@code --timeout}, there a socket timeout), the suffix-prefix-middle infill order (llama.cpp's
 * {@code --spm-infill}) and the Responses API store size (vLLM's {@code VLLM_ENABLE_RESPONSES_API_STORE}, here
 * bounded). Immutable.
 *
 * @param servedNames    the comma-separated served names, or null to serve the model path
 * @param requestTimeout the request budget, positive
 * @param spmInfill      whether {@code /infill} puts the suffix before the prefix
 * @param responseStore  how many Responses API responses the server keeps, 0 for none
 */
record ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore) {

    /** The request budget in seconds. */
    static final String TIMEOUT_PROPERTY = "neuralfree.server.request.timeout";

    /** The default request budget. */
    static final Duration DEFAULT_TIMEOUT = Duration.ofMinutes(30);

    /** The longest request budget: a year, far inside the nanosecond clock's range. */
    static final Duration MAX_TIMEOUT = Duration.ofDays(365);

    /** Whether {@code /infill} uses the suffix-prefix-middle order. */
    static final String SPM_INFILL_PROPERTY = "neuralfree.server.spm.infill";

    /** How many Responses API responses the server keeps for retrieval and {@code previous_response_id}. */
    static final String RESPONSE_STORE_PROPERTY = "neuralfree.server.responses.store";

    ServerOptions {
        Objects.requireNonNull(requestTimeout, "requestTimeout");
        if (requestTimeout.isNegative() || requestTimeout.isZero()) {
            throw new IllegalArgumentException(TIMEOUT_PROPERTY + " must be positive, got " + requestTimeout);
        }
        if (requestTimeout.compareTo(MAX_TIMEOUT) > 0) {
            throw new IllegalArgumentException(TIMEOUT_PROPERTY + " must be at most " + MAX_TIMEOUT.toSeconds()
                    + " seconds, got " + requestTimeout.toSeconds());
        }
        if (responseStore < 0) {
            throw new IllegalArgumentException(RESPONSE_STORE_PROPERTY + " must be 0 or more, got " + responseStore);
        }
    }

    /** {@code servedNames} and {@code requestTimeout}, the prefix-suffix-middle infill order and no response store. */
    ServerOptions(String servedNames, Duration requestTimeout) {
        this(servedNames, requestTimeout, false, 0);
    }

    /** The served model path and the default budget. */
    static ServerOptions defaults() {
        return new ServerOptions(null, DEFAULT_TIMEOUT);
    }

    /** A copy keeping at most {@code responses} Responses API responses. */
    ServerOptions withResponseStore(int responses) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responses);
    }

    /**
     * The options of the {@code neuralfree.server.*} system properties.
     *
     * @throws IllegalArgumentException when the timeout or the store size is not a whole number
     */
    static ServerOptions fromSystemProperties() {
        long timeout = wholeNumber(TIMEOUT_PROPERTY, DEFAULT_TIMEOUT.toSeconds());
        long store = wholeNumber(RESPONSE_STORE_PROPERTY, 0);
        if (store > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(RESPONSE_STORE_PROPERTY + " is too large: " + store);
        }
        return new ServerOptions(System.getProperty(ServedModels.PROPERTY), Duration.ofSeconds(timeout),
                Boolean.parseBoolean(System.getProperty(SPM_INFILL_PROPERTY, "false").trim()), (int) store);
    }

    private static long wholeNumber(String property, long fallback) {
        String raw = System.getProperty(property);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Long.parseLong(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + " must be a whole number, got '" + raw + "'", e);
        }
    }

    /** The budget as the refusal message names it. */
    String budgetText() {
        return requestTimeout.toSeconds() + "-second";
    }
}
