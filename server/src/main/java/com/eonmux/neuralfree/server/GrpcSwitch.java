package com.eonmux.neuralfree.server;

import io.quarkus.runtime.Startup;
import jakarta.inject.Singleton;

import java.util.Map;

/**
 * The gRPC serving switch (vLLM's {@code vllm serve --grpc}): the server serves the {@code neuralfree.Inference}
 * and {@code neuralfree.Control} services, server reflection and the standard gRPC health service only while this
 * is on. {@code neuralfree.server.grpc} (the {@code --grpc} launch flag) decides it; off by default, so a plain
 * HTTP launch exposes no gRPC surface. When off, {@link GrpcAccessInterceptor} answers every gRPC call
 * {@code UNIMPLEMENTED}, whatever port it arrives on. Immutable.
 */
@Startup
@Singleton
public final class GrpcSwitch {

    /** The property that turns the gRPC surface on. */
    static final String PROPERTY = "neuralfree.server.grpc";

    /** The launch flag that sets {@link #PROPERTY}. */
    static final String FLAG = "--grpc";

    /** The gRPC surface off: the default. */
    static final GrpcSwitch OFF = new GrpcSwitch(false);

    /** The gRPC surface on. */
    static final GrpcSwitch ON = new GrpcSwitch(true);

    private final boolean enabled;

    /**
     * The switch of the launch's properties.
     *
     * @throws IllegalArgumentException when the property is neither true nor false
     */
    public GrpcSwitch() {
        this(from(System.getProperties()).enabled);
    }

    private GrpcSwitch(boolean enabled) {
        this.enabled = enabled;
    }

    /** The switch in state {@code enabled}. */
    static GrpcSwitch of(boolean enabled) {
        return enabled ? ON : OFF;
    }

    /** Whether the gRPC services serve. */
    boolean enabled() {
        return enabled;
    }

    /**
     * The switch of {@code properties} ({@link #PROPERTY}).
     *
     * @throws IllegalArgumentException when the property is neither true nor false
     */
    static GrpcSwitch from(Map<?, ?> properties) {
        Object value = properties.get(PROPERTY);
        if (value == null || String.valueOf(value).isBlank()) {
            return OFF;
        }
        String text = String.valueOf(value).trim();
        if (text.equalsIgnoreCase("true")) {
            return ON;
        }
        if (text.equalsIgnoreCase("false")) {
            return OFF;
        }
        throw new IllegalArgumentException(PROPERTY + " must be true or false, got '" + text + "'");
    }
}
