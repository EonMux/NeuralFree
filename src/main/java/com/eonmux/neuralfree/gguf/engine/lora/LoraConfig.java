package com.eonmux.neuralfree.gguf.engine.lora;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * Settings of the runtime adapter registry (vLLM's {@code enable_lora}, {@code max_loras}, {@code max_lora_rank}
 * and {@code max_cpu_loras}):
 * <ul>
 *   <li>{@code neuralfree.lora.enable}: per-request adapters on a GPU load, default false (vLLM
 *       {@code --enable-lora}): the load reserves the device memory of {@code max} adapter slots of rank
 *       {@code max.rank} over the model's adapted products in its fit, so opening a slot never allocates past the
 *       budget; without it a GPU model refuses per-request adapter loads by name. The CPU serves them either
 *       way;</li>
 *   <li>{@code neuralfree.lora.max}: resident adapters ({@link LoraRegistry#load}), default 4
 *       (vLLM {@code max_loras});</li>
 *   <li>{@code neuralfree.lora.max.rank}: the largest rank a loaded adapter's pairs may carry, default 64
 *       (vLLM {@code max_lora_rank}); a larger pair is refused at load, keeping the per-token slot stack
 *       bounded;</li>
 *   <li>{@code neuralfree.lora.cache}: entries of the host cache of unloaded adapters kept with their identity
 *       ({@link LoraRegistry#unload}), default 8 (vLLM {@code max_cpu_loras}).</li>
 * </ul>
 * One instance stays an input of {@link LoraRegistry}; it is not read again by the registry. Immutable; build one
 * with {@link Builder}, the defaults with {@link #defaults()} and one from system properties with
 * {@link #fromProperties(Properties)}.
 */
public final class LoraConfig {

    /** Per-request adapters on a GPU load, with their slots reserved ({@code neuralfree.lora.enable}). */
    public static final String PROP_ENABLE = "neuralfree.lora.enable";
    /** Resident adapters ({@code neuralfree.lora.max}) — the count {@link LoraRegistry#load} keeps. */
    public static final String PROP_MAX = "neuralfree.lora.max";
    /** Maximum rank of an adapter pair ({@code neuralfree.lora.max.rank}). */
    public static final String PROP_MAX_RANK = "neuralfree.lora.max.rank";
    /** Cache entries of unloaded adapters ({@code neuralfree.lora.cache}). */
    public static final String PROP_CACHE = "neuralfree.lora.cache";

    private static final int DEFAULT_MAX = 4;
    private static final int DEFAULT_MAX_RANK = 64;
    private static final int DEFAULT_CACHE = 8;

    private final boolean enabled;
    private final int maxAdapters;
    private final int maxRank;
    private final int cacheEntries;

    private LoraConfig(Builder b) {
        this.enabled = b.enabled;
        this.maxAdapters = b.maxAdapters;
        this.maxRank = b.maxRank;
        this.cacheEntries = b.cacheEntries;
    }

    /** The {@code neuralfree.lora.*} defaults: not enabled on the GPU, 4 resident, rank 64, 8 cached. */
    public static LoraConfig defaults() {
        return builder().build();
    }

    /** An empty builder of the defaults. */
    public static Builder builder() {
        return new Builder();
    }

    /** The settings of {@code properties}, with the defaults for the keys it leaves unset. */
    public static LoraConfig fromProperties(Properties properties) {
        Objects.requireNonNull(properties, "properties");
        Builder b = builder();
        String enable = properties.getProperty(PROP_ENABLE);
        if (enable != null) {
            b.enabled(parseBoolean(PROP_ENABLE, enable));
        }
        String max = properties.getProperty(PROP_MAX);
        if (max != null) {
            b.maxAdapters(parseInt(PROP_MAX, max));
        }
        String rank = properties.getProperty(PROP_MAX_RANK);
        if (rank != null) {
            b.maxRank(parseInt(PROP_MAX_RANK, rank));
        }
        String cache = properties.getProperty(PROP_CACHE);
        if (cache != null) {
            b.cacheEntries(parseInt(PROP_CACHE, cache));
        }
        return b.build();
    }

    /** The int value of {@code raw}, naming the property in the error. */
    private static int parseInt(String key, String raw) {
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + " needs an integer, got '" + raw + "'");
        }
    }

    /** The boolean value of {@code raw} ({@code true} / {@code false}, {@code 1} / {@code 0}), naming the property. */
    private static boolean parseBoolean(String key, String raw) {
        String v = raw.trim().toLowerCase(Locale.ROOT);
        if (v.equals("true") || v.equals("1")) {
            return true;
        }
        if (v.equals("false") || v.equals("0")) {
            return false;
        }
        throw new IllegalArgumentException(key + " needs true or false, got '" + raw + "'");
    }

    /** Whether a GPU load serves per-request adapters and reserves their slots. */
    public boolean enabled() {
        return enabled;
    }

    /** Resident adapter count. */
    public int maxAdapters() {
        return maxAdapters;
    }

    /** Maximum rank of an adapter pair. */
    public int maxRank() {
        return maxRank;
    }

    /** Cache entries of unloaded adapters. */
    public int cacheEntries() {
        return cacheEntries;
    }

    /** A builder of an immutable {@link LoraConfig}. */
    public static final class Builder {
        private boolean enabled;
        private int maxAdapters = DEFAULT_MAX;
        private int maxRank = DEFAULT_MAX_RANK;
        private int cacheEntries = DEFAULT_CACHE;

        /** Whether a GPU load serves per-request adapters and reserves their slots. */
        public Builder enabled(boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        /** Resident adapter count.
         *
         * @throws IllegalArgumentException when {@code maxAdapters} is below 1
         */
        public Builder maxAdapters(int maxAdapters) {
            if (maxAdapters < 1) {
                throw new IllegalArgumentException(PROP_MAX + " needs at least 1, got " + maxAdapters);
            }
            this.maxAdapters = maxAdapters;
            return this;
        }

        /** Maximum rank of an adapter pair.
         *
         * @throws IllegalArgumentException when {@code maxRank} is below 1
         */
        public Builder maxRank(int maxRank) {
            if (maxRank < 1) {
                throw new IllegalArgumentException(PROP_MAX_RANK + " needs at least 1, got " + maxRank);
            }
            this.maxRank = maxRank;
            return this;
        }

        /** Cache entries of unloaded adapters.
         *
         * @throws IllegalArgumentException when {@code cacheEntries} is negative
         */
        public Builder cacheEntries(int cacheEntries) {
            if (cacheEntries < 0) {
                throw new IllegalArgumentException(PROP_CACHE + " needs at least 0, got " + cacheEntries);
            }
            this.cacheEntries = cacheEntries;
            return this;
        }

        /** Builds the configuration. */
        public LoraConfig build() {
            return new LoraConfig(this);
        }
    }

    /** An {@link LoraConfig} copy with mutated fields, as every configuration object builds its copies. */
    public Builder toBuilder() {
        return new Builder().enabled(enabled).maxAdapters(maxAdapters).maxRank(maxRank).cacheEntries(cacheEntries);
    }

    /** Value equality: the same four settings. */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof LoraConfig)) {
            return false;
        }
        LoraConfig c = (LoraConfig) o;
        return enabled == c.enabled && maxAdapters == c.maxAdapters && maxRank == c.maxRank
                && cacheEntries == c.cacheEntries;
    }

    @Override
    public int hashCode() {
        return Objects.hash(enabled, maxAdapters, maxRank, cacheEntries);
    }

    @Override
    public String toString() {
        return "LoraConfig{" + PROP_ENABLE + "=" + enabled + ", " + PROP_MAX + "=" + maxAdapters + ", "
                + PROP_MAX_RANK + "=" + maxRank + ", " + PROP_CACHE + "=" + cacheEntries + '}';
    }
}
