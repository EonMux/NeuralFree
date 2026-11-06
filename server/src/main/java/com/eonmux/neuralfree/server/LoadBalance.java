package com.eonmux.neuralfree.server;

import java.util.Locale;

/**
 * How the {@link DpScheduler} picks the replica of a request: SGLang's {@code --load-balance-method} choices over
 * the local replica pool. Each method has its SGLang name, and the two load-aware ones also answer to the names
 * SGLang gives them today ({@code total_requests}, {@code total_tokens}).
 */
enum LoadBalance {

    /** The replica with the fewest queued plus running requests; idle ties go to a replica holding the request's
     *  prefix, else round robin (SGLang's {@code shortest_queue}, now {@code total_requests}). The default. */
    SHORTEST_QUEUE("shortest_queue"),

    /** Each request to the next replica in turn, whatever its load (SGLang's {@code round_robin}). */
    ROUND_ROBIN("round_robin"),

    /** The replica with the fewest prompt tokens in flight, ties to the fewest requests, then round robin
     *  (SGLang's {@code minimum_tokens}, now {@code total_tokens}). */
    MINIMUM_TOKENS("minimum_tokens");

    private final String label;

    LoadBalance(String label) {
        this.label = label;
    }

    /** The method's name as the property and the launch flag take it. */
    String label() {
        return label;
    }

    /**
     * The method {@code name} names, case-insensitively: a method's own name, SGLang's current aliases
     * {@code total_requests} and {@code total_tokens}, or {@code auto} for the default.
     *
     * @param field the property or flag the name came from, for the refusal
     * @throws IllegalArgumentException when the name is none of these
     */
    static LoadBalance parse(String name, String field) {
        String key = name.trim().toLowerCase(Locale.ROOT);
        switch (key) {
            case "auto", "total_requests" -> {
                return SHORTEST_QUEUE;
            }
            case "total_tokens" -> {
                return MINIMUM_TOKENS;
            }
            default -> {
                for (LoadBalance method : values()) {
                    if (method.label.equals(key)) {
                        return method;
                    }
                }
                throw new IllegalArgumentException(field + " must be shortest_queue, round_robin or minimum_tokens"
                        + " (also auto, total_requests, total_tokens), got '" + name + "'");
            }
        }
    }
}
