package com.eonmux.neuralfree.server;

import io.quarkus.runtime.Startup;
import jakarta.inject.Singleton;

import java.util.Map;

/**
 * The sleep-mode switch (vLLM's {@code --enable-sleep-mode}): vLLM registers its checkpoint routes only when sleep
 * mode is on, and its engine sleeps only then. This switch decides whether the checkpoint window
 * ({@code /checkpoint/prepare}, {@code /checkpoint/restore}, {@code /checkpoint/status}) is registered and whether
 * {@code /sleep}, {@code /wake_up} and {@code /release_kv_cache_memory} may release memory.
 * {@code neuralfree.server.sleep.mode} (the {@code --enable-sleep-mode} launch flag) decides it; off by default.
 * llama.cpp's idle sleep ({@code --sleep-idle-seconds}) is a serving option of its own and does not need it.
 * Immutable.
 */
@Startup
@Singleton
public final class SleepModeSwitch {

    /** The property that turns sleep mode on. */
    static final String PROPERTY = "neuralfree.server.sleep.mode";

    /** The launch flag that sets {@link #PROPERTY}. */
    static final String FLAG = "--enable-sleep-mode";

    /** Sleep mode off: the default. */
    static final SleepModeSwitch OFF = new SleepModeSwitch(false);

    /** Sleep mode on. */
    static final SleepModeSwitch ON = new SleepModeSwitch(true);

    private final boolean enabled;

    /**
     * The switch of the launch's properties.
     *
     * @throws IllegalArgumentException when the property is neither true nor false
     */
    public SleepModeSwitch() {
        this(from(System.getProperties()));
    }

    private SleepModeSwitch(boolean enabled) {
        this.enabled = enabled;
    }

    private SleepModeSwitch(SleepModeSwitch other) {
        this.enabled = other.enabled;
    }

    /** The switch in state {@code enabled}. */
    static SleepModeSwitch of(boolean enabled) {
        return enabled ? ON : OFF;
    }

    /** Whether sleep mode serves. */
    boolean enabled() {
        return enabled;
    }

    /**
     * Refuses a sleep operation while the switch is off, naming the flag: vLLM's engine sleeps only with
     * {@code --enable-sleep-mode}.
     */
    void requireSleep() {
        if (!enabled) {
            throw ServerJson.badRequest("sleep mode is not enabled: start the server with " + FLAG + " or -D"
                    + PROPERTY + "=true (vLLM's engine refuses to sleep without it)");
        }
    }

    /**
     * Refuses the checkpoint window with the 404 an unregistered vLLM route answers while the switch is off: vLLM
     * attaches the checkpoint router only with {@code --enable-sleep-mode}.
     */
    void requireCheckpoint() {
        if (!enabled) {
            throw ServerJson.error(404, "the checkpoint window is off: start the server with " + FLAG + " or -D"
                    + PROPERTY + "=true (vLLM registers /checkpoint/prepare, /checkpoint/restore and"
                    + " /checkpoint/status only with --enable-sleep-mode)", "NotFoundError");
        }
    }

    /**
     * The switch of {@code properties} ({@link #PROPERTY}).
     *
     * @throws IllegalArgumentException when the property is neither true nor false
     */
    static SleepModeSwitch from(Map<?, ?> properties) {
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
