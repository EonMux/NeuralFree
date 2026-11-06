package com.eonmux.neuralfree.gguf.engine.kvevents;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * vLLM's {@code KVEventsConfig} ({@code --kv-events-config} and the JSON form of {@code neuralfree.kv.events}): the
 * {@code enable_kv_cache_events} switch, the {@code publisher} it selects, and the ZMQ transport fields of
 * {@link ZmqEventPublisherConfig} ({@code endpoint}, {@code replay_endpoint}, {@code buffer_steps}, {@code hwm},
 * {@code max_queue_size}, {@code topic}), vLLM's defaults where a field is absent. An enabled config with no
 * publisher resolves to {@code zmq} as vLLM's does; a disabled one, or the {@code "null"} (or {@code "none"})
 * publisher, publishes nothing whatever else it names; {@code "log"} and a {@code KvEventPublisher} class name are this engine's extensions, and a transport
 * field beside one of them is refused by name, since nothing would read it. Immutable.
 */
public final class KvEventsConfig {

    /** The publisher name of vLLM's ZMQ transport. */
    public static final String ZMQ = "zmq";

    private static final List<String> TRANSPORT_FIELDS = Arrays.asList("endpoint", "replay_endpoint",
            "buffer_steps", "hwm", "max_queue_size", "topic");

    private static final Set<String> FIELDS = fields();

    private final String publisher;
    private final ZmqEventPublisherConfig zmq;
    private final boolean transportFields;

    private KvEventsConfig(String publisher, ZmqEventPublisherConfig zmq, boolean transportFields) {
        this.publisher = publisher;
        this.zmq = zmq;
        this.transportFields = transportFields;
    }

    private static Set<String> fields() {
        Set<String> all = new LinkedHashSet<>(Arrays.asList("enable_kv_cache_events", "publisher"));
        all.addAll(TRANSPORT_FIELDS);
        return all;
    }

    /**
     * The config {@code json} holds, {@code refusedBy} (the flag or property) named in every refusal.
     *
     * @throws IllegalArgumentException when the JSON is not an object, a field is unknown or of the wrong type, a
     *                                  transport field is out of range, or a transport field sits beside a
     *                                  publisher other than {@code zmq}
     */
    public static KvEventsConfig parse(String json, String refusedBy) {
        Map<?, ?> config = object(json, refusedBy);
        boolean enabled = enable(config, refusedBy);
        String publisher = publisher(config, refusedBy);
        String resolved = publisher == null ? (enabled ? ZMQ : null) : publisher;
        if (!enabled || "null".equalsIgnoreCase(resolved) || "none".equalsIgnoreCase(resolved)) {
            return new KvEventsConfig(null, null, false);
        }
        String transportField = firstTransportField(config);
        if (!ZMQ.equalsIgnoreCase(resolved)) {
            if (transportField != null) {
                throw new IllegalArgumentException(refusedBy + " field " + transportField + " configures the ZMQ"
                        + " transport, which the publisher " + resolved + " does not read");
            }
            return new KvEventsConfig(resolved, null, false);
        }
        return new KvEventsConfig(ZMQ, zmq(config, refusedBy), transportField != null);
    }

    /** The JSON object {@code json} holds, refused unless every field is one vLLM's config declares. */
    private static Map<?, ?> object(String json, String refusedBy) {
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException(refusedBy + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map<?, ?>)) {
            throw new IllegalArgumentException(refusedBy + " must be a JSON object, got " + json);
        }
        Map<?, ?> config = (Map<?, ?>) parsed;
        for (Object key : config.keySet()) {
            if (!FIELDS.contains(String.valueOf(key))) {
                throw new IllegalArgumentException(refusedBy + " has no field " + key + " (it reads "
                        + String.join(", ", FIELDS) + ")");
            }
        }
        return config;
    }

    /** The {@code enable_kv_cache_events} switch, default false as vLLM's. */
    private static boolean enable(Map<?, ?> config, String refusedBy) {
        Object enable = config.get("enable_kv_cache_events");
        if (enable == null) {
            return false;
        }
        if (!(enable instanceof Boolean)) {
            throw new IllegalArgumentException(refusedBy + " enable_kv_cache_events must be a boolean, got "
                    + enable);
        }
        return (Boolean) enable;
    }

    /** The {@code publisher} the config names, trimmed; null when absent or JSON null (vLLM's unset default). */
    private static String publisher(Map<?, ?> config, String refusedBy) {
        Object publisher = config.get("publisher");
        if (publisher == null) {
            return null;
        }
        if (!(publisher instanceof String) || ((String) publisher).isBlank()) {
            throw new IllegalArgumentException(refusedBy + " publisher must be a non-empty string, got " + publisher);
        }
        return ((String) publisher).trim();
    }

    private static String firstTransportField(Map<?, ?> config) {
        for (String field : TRANSPORT_FIELDS) {
            if (config.containsKey(field)) {
                return field;
            }
        }
        return null;
    }

    /** The ZMQ transport the config's fields describe, vLLM's defaults where a field is absent. */
    private static ZmqEventPublisherConfig zmq(Map<?, ?> config, String refusedBy) {
        String endpoint = text(config, "endpoint", refusedBy, ZmqEventPublisherConfig.DEFAULT_ENDPOINT);
        String replayEndpoint = text(config, "replay_endpoint", refusedBy, null);
        int bufferSteps = count(config, "buffer_steps", refusedBy, ZmqEventPublisherConfig.DEFAULT_BUFFER_STEPS);
        int hwm = count(config, "hwm", refusedBy, ZmqEventPublisherConfig.DEFAULT_HWM);
        int maxQueueSize = count(config, "max_queue_size", refusedBy,
                ZmqEventPublisherConfig.DEFAULT_MAX_QUEUE_SIZE);
        String topic = text(config, "topic", refusedBy, "");
        try {
            return new ZmqEventPublisherConfig(endpoint, replayEndpoint, bufferSteps, hwm, maxQueueSize, topic, 0);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(refusedBy + " " + e.getMessage(), e);
        }
    }

    private static String text(Map<?, ?> config, String field, String refusedBy, String fallback) {
        Object value = config.get(field);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof String)) {
            throw new IllegalArgumentException(refusedBy + " " + field + " must be a string, got " + value);
        }
        return (String) value;
    }

    /** The whole number {@code field} holds, inside {@code [0, Integer.MAX_VALUE]}. */
    private static int count(Map<?, ?> config, String field, String refusedBy, int fallback) {
        Object value = config.get(field);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof Long || value instanceof Integer)) {
            throw new IllegalArgumentException(refusedBy + " " + field + " must be a whole number, got " + value);
        }
        long number = ((Number) value).longValue();
        if (number < 0 || number > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(refusedBy + " " + field + " must be between 0 and "
                    + Integer.MAX_VALUE + ", got " + number);
        }
        return (int) number;
    }

    /** Whether the config publishes at all. */
    public boolean enabled() { return publisher != null; }

    /** The publisher the config selects: {@code zmq}, {@code log}, a class name, or null for none. */
    public String publisher() { return publisher; }

    /** The ZMQ transport when {@link #publisher()} is {@code zmq}, null otherwise. */
    public ZmqEventPublisherConfig zmq() { return zmq; }

    /** Whether the config names a ZMQ transport field, so vLLM's defaults alone do not describe it. */
    public boolean hasTransportFields() { return transportFields; }

    @Override
    public String toString() {
        return "KvEventsConfig{publisher=" + publisher + (zmq == null ? "" : ", " + zmq) + "}";
    }
}
