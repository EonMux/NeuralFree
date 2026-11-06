package com.eonmux.neuralfree.gguf.cpu.numa;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * How CPU work is placed over the host's NUMA nodes (llama.cpp {@code --numa},
 * {@code ggml_numa_strategy} in {@code ggml/include/ggml-cpu.h}).
 *
 * <p>{@link #DISTRIBUTE} spreads the CPU pool's threads evenly over every node, {@link #ISOLATE} keeps them on the
 * node the process started on, and {@link #NUMACTL} keeps the CPU set {@code numactl} gave the process. The
 * property is {@value #PROPERTY}; absent or blank means {@link #DISABLED}.
 */
public enum NumaStrategy {

    /** No placement: threads run wherever the OS schedules them (the default). */
    DISABLED("disabled"),
    /** Threads are spread over all nodes, one node per thread in creation order. */
    DISTRIBUTE("distribute"),
    /** Every thread runs on the node the process started on. */
    ISOLATE("isolate"),
    /** Every thread keeps the CPU set inherited from {@code numactl}. */
    NUMACTL("numactl");

    /** The system property that selects the strategy ({@code neuralfree.numa}). */
    public static final String PROPERTY = "neuralfree.numa";

    private final String label;

    NumaStrategy(String label) {
        this.label = label;
    }

    /** The name the flag and the property spell this strategy with. */
    public String label() {
        return label;
    }

    /** Whether the strategy places threads at all. */
    public boolean enabled() {
        return this != DISABLED;
    }

    /**
     * Parses one strategy name: {@code distribute}, {@code isolate}, {@code numactl}, or {@code disabled} (also
     * spelled {@code none} and {@code off}). Names are case-insensitive.
     *
     * @throws IllegalArgumentException naming {@code value} when it is not one of those names
     */
    public static NumaStrategy parse(String value) {
        String s = Objects.requireNonNull(value, "numa strategy").trim().toLowerCase(Locale.ROOT);
        if (s.equals("disabled") || s.equals("none") || s.equals("off")) {
            return DISABLED;
        }
        for (NumaStrategy strategy : values()) {
            if (strategy.label.equals(s)) {
                return strategy;
            }
        }
        throw new IllegalArgumentException(PROPERTY + "=" + value + " is invalid; use distribute, isolate,"
                + " numactl or disabled");
    }

    /**
     * The strategy of a property set: {@link #DISABLED} when {@value #PROPERTY} is absent or blank.
     *
     * @throws IllegalArgumentException naming the property when the value is not a strategy name
     */
    public static NumaStrategy fromProperties(Properties properties) {
        Objects.requireNonNull(properties, "properties");
        String value = properties.getProperty(PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return DISABLED;
        }
        return parse(value);
    }
}
