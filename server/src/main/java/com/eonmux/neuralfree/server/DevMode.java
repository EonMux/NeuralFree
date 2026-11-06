package com.eonmux.neuralfree.server;

import io.quarkus.runtime.Startup;
import jakarta.inject.Singleton;

import java.util.Map;
import java.util.logging.Logger;

/**
 * The development-endpoint switch (vLLM's {@code VLLM_SERVER_DEV_MODE}): the routes vLLM registers only in its
 * development mode — the cache resets, {@code /server_info}, the sleep and release routes, the RL pause, resume and
 * abort routes and the weight-transfer routes — serve only while this is on, and answer 404 as an unregistered vLLM
 * route does while it is off. {@code neuralfree.server.dev.mode} (the {@code --dev-mode} launch flag) decides the
 * switch when it is set; vLLM's {@code VLLM_SERVER_DEV_MODE} integer value ({@code 0} off, any other value on) is
 * honoured otherwise. Off by default; an enabled switch logs vLLM's security warning once, when the server starts.
 * Immutable.
 */
@Startup
@Singleton
public final class DevMode {

    /** The property that turns the development endpoints on. */
    static final String PROPERTY = "neuralfree.server.dev.mode";

    /** The launch flag that sets {@link #PROPERTY}. */
    static final String FLAG = "--dev-mode";

    /** vLLM's environment switch, read when {@link #PROPERTY} is unset. */
    static final String ENV = "VLLM_SERVER_DEV_MODE";

    /** vLLM's log line when it registers the development routes. */
    static final String WARNING = "SECURITY WARNING: Development endpoints are enabled! This should NOT be used"
            + " in production!";

    private static final Logger LOGGER = Logger.getLogger(DevMode.class.getName());

    /** The development endpoints off: the default. */
    static final DevMode OFF = new DevMode(false);

    /** The development endpoints on. */
    static final DevMode ON = new DevMode(true);

    private final boolean enabled;

    /**
     * The switch of the launch's properties and environment.
     *
     * @throws IllegalArgumentException when the property is neither true nor false, or the environment value is not
     *                                  an integer
     */
    public DevMode() {
        this(from(System.getProperties(), System.getenv()).enabled);
        if (enabled) {
            LOGGER.warning(WARNING);
        }
    }

    private DevMode(boolean enabled) {
        this.enabled = enabled;
    }

    /** The switch in state {@code enabled}. */
    static DevMode of(boolean enabled) {
        return enabled ? ON : OFF;
    }

    /** Whether the development endpoints serve. */
    boolean enabled() {
        return enabled;
    }

    /** Refuses a development route with the 404 an unregistered vLLM route answers while the switch is off. */
    void require() {
        if (!enabled) {
            throw ServerJson.error(404, "development endpoints are off: start the server with " + FLAG + " or -D"
                    + PROPERTY + "=true (vLLM's " + ENV + ")", "NotFoundError");
        }
    }

    /**
     * The switch of {@code properties} ({@link #PROPERTY}) and, when it holds no value, of {@code env}
     * ({@link #ENV}), which carries vLLM's integer form.
     *
     * @throws IllegalArgumentException when the property is neither true nor false, or the environment value is not
     *                                  an integer
     */
    static DevMode from(Map<?, ?> properties, Map<String, String> env) {
        Object property = properties.get(PROPERTY);
        if (property != null && !String.valueOf(property).isBlank()) {
            String value = String.valueOf(property).trim();
            if (value.equalsIgnoreCase("true")) {
                return ON;
            }
            if (value.equalsIgnoreCase("false")) {
                return OFF;
            }
            throw new IllegalArgumentException(PROPERTY + " must be true or false, got '" + value + "'");
        }
        String value = env.get(ENV);
        if (value == null) {
            return OFF;
        }
        try {
            return of(Integer.parseInt(value.trim()) != 0);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(ENV + " must be an integer, 0 turning the development endpoints"
                    + " off, got '" + value + "'", e);
        }
    }
}
