package com.eonmux.neuralfree.server;

import jakarta.inject.Singleton;

import java.util.Map;

/**
 * The distributed-serving control switches: vLLM's {@code --enable-elastic-ep} and
 * {@code --enable-fault-tolerance}. Both are off by default; with one off its routes answer 404 as an unregistered
 * vLLM route does, so a topology-changing endpoint is never served by accident. {@code neuralfree.server.elastic.ep}
 * and {@code neuralfree.server.fault.tolerance} (the launch flags) decide them. Immutable.
 */
@Singleton
public final class ClusterSwitches {

    /** The property that turns the elastic-scale routes on. */
    static final String ELASTIC_PROPERTY = "neuralfree.server.elastic.ep";

    /** The launch flag that sets {@link #ELASTIC_PROPERTY}. */
    static final String ELASTIC_FLAG = "--enable-elastic-ep";

    /** The property that turns the fault-tolerance routes on. */
    static final String FAULT_PROPERTY = "neuralfree.server.fault.tolerance";

    /** The launch flag that sets {@link #FAULT_PROPERTY}. */
    static final String FAULT_FLAG = "--enable-fault-tolerance";

    private final boolean elasticEp;
    private final boolean faultTolerance;

    /** The switches of the launch's system properties. */
    public ClusterSwitches() {
        this(flag(System.getProperties(), ELASTIC_PROPERTY), flag(System.getProperties(), FAULT_PROPERTY));
    }

    /** Both switches in the given state. */
    ClusterSwitches(boolean elasticEp, boolean faultTolerance) {
        this.elasticEp = elasticEp;
        this.faultTolerance = faultTolerance;
    }

    /**
     * Whether {@code property} is on: absent or blank is off, {@code true} and {@code false} are read literally.
     *
     * @throws IllegalArgumentException when the value is neither true nor false
     */
    static boolean flag(Map<?, ?> properties, String property) {
        Object value = properties.get(property);
        if (value == null || String.valueOf(value).isBlank()) {
            return false;
        }
        String text = String.valueOf(value).trim();
        if (text.equalsIgnoreCase("true")) {
            return true;
        }
        if (text.equalsIgnoreCase("false")) {
            return false;
        }
        throw new IllegalArgumentException(property + " must be true or false, got '" + text + "'");
    }

    /** Whether the elastic-scale routes serve. */
    boolean elasticEp() {
        return elasticEp;
    }

    /** Whether the fault-tolerance routes serve. */
    boolean faultTolerance() {
        return faultTolerance;
    }

    /** The switch of {@code properties} for the tests' manual wiring. */
    static ClusterSwitches of(Map<?, ?> properties) {
        return new ClusterSwitches(flag(properties, ELASTIC_PROPERTY), flag(properties, FAULT_PROPERTY));
    }

    /** Refuses a gated route with the 404 an unregistered vLLM route answers while the switch is off. */
    static void require(boolean enabled, String flag) {
        if (!enabled) {
            throw ServerJson.error(404, "this endpoint is off: start the server with " + flag, "NotFoundError");
        }
    }
}
