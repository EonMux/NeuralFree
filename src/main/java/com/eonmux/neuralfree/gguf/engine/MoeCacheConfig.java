package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.ops.moe.MoeTailSkip;

import java.util.Locale;
import java.util.Objects;

/**
 * The MoE expert cache configuration (llama.cpp {@code --moe-cache-mib}, Avalon {@code --moe-cache}): hot routed
 * experts of layers whose expert stacks stay in host RAM are cached in spare VRAM, the GPU computes the cached
 * experts and the host the misses. {@code neuralfree.moe.cache}:
 * <ul>
 *   <li>{@code off} (default): every weight is placed as without the cache.</li>
 *   <li>{@code on}: the routed expert stacks stay in host RAM (as {@code neuralfree.cpu.moe} places them) and the
 *       cache serves them from the VRAM left after loading (Avalon {@code --moe-cache on}).</li>
 *   <li>{@code auto}: currently the same as {@code on} (Avalon {@code --moe-cache auto}).</li>
 *   <li>a number: {@code on} with that budget in MiB ({@code --moe-cache-mib N}).</li>
 * </ul>
 * {@code neuralfree.moe.cache.mb} sets the same budget in MiB and wins when both give one.
 * The ranking keys ({@link MoeCacheRanking.Options}) ride along: {@code neuralfree.moe.cache.ranking} names the
 * Avalon {@code moe-cache-ranking.bin} ranking the cache preloads from (unset follows Avalon and reads the one next
 * to the model when it exists, {@code off} preloads nothing) and {@code neuralfree.moe.cache.ranking.save} the file
 * the routed-row profile is written to when the model closes.
 * {@code neuralfree.moe.cache.prefetch}: the prefill expert prefetch (Avalon {@code GGML_SCHED_PREFETCH_EXPERTS}),
 * the number of layer stacks a prefill-sized step borrows from the cache to upload the experts it does not hold;
 * {@value #PREFETCH_ZONES_DEFAULT} by default (one layer computing while the next is copied), {@code 0} or
 * {@code off} computes those misses on the host instead. {@code neuralfree.moe.cache.hosted.rows} (Avalon
 * {@code GGML_MOE_HOSTED_MAX_ROWS}): steps of at most that many routed rows keep the host misses even when they are
 * prefill-sized; {@code 0} (default) prefetches every prefill-sized step. {@code neuralfree.moe.cache.tail.skip}
 * (Avalon {@code GGML_CUDA_MOE_TAIL_SKIP}, Strata's {@code STRATA_ROUTE_TAIL_SKIP}): the hosted tail-skip rank, a
 * 0-based top-k position from 1 to {@value #TAIL_SKIP_MAX_RANK}; a missed expert that every token of a step routes
 * only at that position or lower is not computed, and its output rows stay zero. {@code off} or {@code 0} (the
 * default) keeps every miss, where Avalon enables the rule with rank 7; a rank needs the cache enabled. Immutable.
 */
public final class MoeCacheConfig {

    /** Property holding the mode or budget. */
    public static final String PROP_MOE_CACHE = "neuralfree.moe.cache";
    /** Property holding the prefill expert prefetch zone count. */
    public static final String PROP_MOE_CACHE_PREFETCH = "neuralfree.moe.cache.prefetch";
    /** Property holding the routed-row count up to which a step keeps the host misses. */
    public static final String PROP_MOE_CACHE_HOSTED_ROWS = "neuralfree.moe.cache.hosted.rows";
    /** Property holding the hosted tail-skip rank ({@link MoeTailSkip}). */
    public static final String PROP_MOE_CACHE_TAIL_SKIP = "neuralfree.moe.cache.tail.skip";
    /** Default {@link #prefetchZones()}: the layer being computed and the next one being copied. */
    public static final int PREFETCH_ZONES_DEFAULT = 2;
    /** Highest accepted hosted tail-skip rank (0, the default, turns the rule off). */
    public static final int TAIL_SKIP_MAX_RANK = MoeTailSkip.MAX_RANK;

    /** When the cache serves the routed experts. */
    public enum Mode {
        /** Every weight is placed as without the cache. */
        OFF,
        /** The routed expert stacks stay in host RAM; the cache serves them from spare VRAM. */
        ON,
        /** Currently the same as {@link #ON}. */
        AUTO;

        /**
         * The mode of a setting value: {@code off}, {@code on} or {@code auto}, ignoring case.
         *
         * @throws IllegalArgumentException for any other value, naming the valid ones
         */
        public static Mode parse(String value) {
            String s = Objects.requireNonNull(value, "mode").trim().toLowerCase(Locale.ROOT);
            switch (s) {
                case "off":
                case "false":
                    return OFF;
                case "on":
                case "true":
                    return ON;
                case "auto":
                    return AUTO;
                default:
                    throw new IllegalArgumentException(PROP_MOE_CACHE + "=" + value + " is invalid; use off, on,"
                            + " auto or a budget in MiB");
            }
        }
    }

    /** {@link #budgetMiB()} value meaning "the VRAM left after loading". */
    public static final long BUDGET_AUTO = -1L;

    private static final MoeCacheConfig OFF = new MoeCacheConfig(Mode.OFF, BUDGET_AUTO,
            MoeCacheRanking.Options.auto(), PREFETCH_ZONES_DEFAULT, 0, MoeTailSkip.OFF);

    private final Mode mode;
    private final long budgetMiB;
    private final MoeCacheRanking.Options ranking;
    private final int prefetchZones;
    private final int hostedMaxRows;
    private final int tailSkipRank;

    private MoeCacheConfig(Mode mode, long budgetMiB, MoeCacheRanking.Options ranking, int prefetchZones,
                           int hostedMaxRows, int tailSkipRank) {
        this.mode = mode;
        this.budgetMiB = budgetMiB;
        this.ranking = ranking;
        this.prefetchZones = prefetchZones;
        this.hostedMaxRows = hostedMaxRows;
        this.tailSkipRank = tailSkipRank;
    }

    /**
     * The default: off, automatic budget, the default ranking options, {@value #PREFETCH_ZONES_DEFAULT} prefetch
     * zones.
     */
    public static MoeCacheConfig off() {
        return OFF;
    }

    /**
     * The configuration from a property set; absent keys keep their defaults.
     *
     * @throws IllegalArgumentException naming the property when a value is invalid
     */
    public static MoeCacheConfig fromProperties(java.util.Properties p) {
        String v = p.getProperty(PROP_MOE_CACHE);
        MoeCacheConfig out = v == null || v.trim().isEmpty() ? OFF : parse(v);
        out = out.withRanking(MoeCacheRanking.Options.fromProperties(p));
        String prefetch = p.getProperty(PROP_MOE_CACHE_PREFETCH);
        if (prefetch != null && !prefetch.trim().isEmpty()) {
            out = out.withPrefetchZones(parsePrefetchZones(prefetch));
        }
        String hosted = p.getProperty(PROP_MOE_CACHE_HOSTED_ROWS);
        if (hosted != null && !hosted.trim().isEmpty()) {
            out = out.withHostedMaxRows(parseHostedMaxRows(hosted));
        }
        String tailSkip = p.getProperty(PROP_MOE_CACHE_TAIL_SKIP);
        if (tailSkip != null && !tailSkip.trim().isEmpty()) {
            out = out.withTailSkipRank(parseTailSkipRank(tailSkip));
        }
        return out;
    }

    /**
     * The hosted tail-skip rank of a setting value: {@code off}, {@code false} or {@code 0} turns the rule off, and
     * a rank from 1 to {@value #TAIL_SKIP_MAX_RANK} drops the experts every token routes only at that 0-based
     * top-k position or lower.
     *
     * @throws IllegalArgumentException naming the property for any other value
     */
    public static int parseTailSkipRank(String value) {
        String v = Objects.requireNonNull(value, "value").trim().toLowerCase(Locale.ROOT);
        if (v.equals("off") || v.equals("false")) {
            return MoeTailSkip.OFF;
        }
        try {
            int rank = Integer.parseInt(v);
            if (rank < MoeTailSkip.OFF || rank > TAIL_SKIP_MAX_RANK) {
                throw new IllegalArgumentException(PROP_MOE_CACHE_TAIL_SKIP + "=" + value + " must be "
                        + MoeTailSkip.OFF + " (off) or 1-" + TAIL_SKIP_MAX_RANK);
            }
            return rank;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_TAIL_SKIP + "=" + value + " is invalid; use off, 0 or"
                    + " a rank from 1 to " + TAIL_SKIP_MAX_RANK, e);
        }
    }

    /**
     * The hosted routed-row bound of a setting value: a count {@code >= 0}.
     *
     * @throws IllegalArgumentException naming the property for any other value
     */
    public static int parseHostedMaxRows(String value) {
        String v = Objects.requireNonNull(value, "value").trim();
        try {
            int rows = Integer.parseInt(v);
            if (rows < 0) {
                throw new IllegalArgumentException(PROP_MOE_CACHE_HOSTED_ROWS + "=" + value + " must be >= 0");
            }
            return rows;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_HOSTED_ROWS + "=" + value + " is invalid; use a routed"
                    + " row count >= 0", e);
        }
    }

    /**
     * The prefetch zone count of a setting value: a count {@code >= 0}, or {@code off} / {@code false} for 0.
     *
     * @throws IllegalArgumentException naming the property for any other value
     */
    public static int parsePrefetchZones(String value) {
        String v = Objects.requireNonNull(value, "value").trim().toLowerCase(Locale.ROOT);
        if (v.equals("off") || v.equals("false")) {
            return 0;
        }
        try {
            int zones = Integer.parseInt(v);
            if (zones < 0) {
                throw new IllegalArgumentException(PROP_MOE_CACHE_PREFETCH + "=" + value + " must be >= 0");
            }
            return zones;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_PREFETCH + "=" + value + " is invalid; use a zone"
                    + " count >= 0 or off", e);
        }
    }

    /**
     * The configuration of a setting value: {@code off}, {@code on}, {@code auto} or a budget in MiB
     * ({@code >= 0}).
     *
     * @throws IllegalArgumentException naming the property when the value is neither
     */
    public static MoeCacheConfig parse(String value) {
        String v = Objects.requireNonNull(value, "value").trim();
        if (v.isEmpty()) {
            return OFF;
        }
        try {
            long mib = Long.parseLong(v);
            if (mib < 0) {
                throw new IllegalArgumentException(PROP_MOE_CACHE + "=" + value + " must be >= 0");
            }
            return new MoeCacheConfig(Mode.ON, mib, MoeCacheRanking.Options.auto(), PREFETCH_ZONES_DEFAULT, 0,
                    MoeTailSkip.OFF);
        } catch (NumberFormatException e) {
            return new MoeCacheConfig(Mode.parse(v), BUDGET_AUTO, MoeCacheRanking.Options.auto(),
                    PREFETCH_ZONES_DEFAULT, 0, MoeTailSkip.OFF);
        }
    }

    public Mode mode() {
        return mode;
    }

    /** The budget in MiB, or {@link #BUDGET_AUTO} for the VRAM left after loading. */
    public long budgetMiB() {
        return budgetMiB;
    }

    /** Where the model's ranking session preloads from and saves its profile to. */
    public MoeCacheRanking.Options ranking() {
        return ranking;
    }

    /** Layer stacks a prefill-sized step borrows from the cache for its expert uploads; 0 disables the prefetch. */
    public int prefetchZones() {
        return prefetchZones;
    }

    /** Routed rows up to which a step keeps the host misses instead of prefetching (0: every prefill step). */
    public int hostedMaxRows() {
        return hostedMaxRows;
    }

    /** The hosted tail-skip rank ({@link MoeTailSkip}); 0 keeps every miss. */
    public int tailSkipRank() {
        return tailSkipRank;
    }

    /** True when the mode can serve the routed experts from a cache. */
    public boolean enabled() {
        return mode == Mode.ON || mode == Mode.AUTO;
    }

    /** This configuration with other ranking options. */
    public MoeCacheConfig withRanking(MoeCacheRanking.Options options) {
        Objects.requireNonNull(options, "ranking");
        return options.equals(ranking) ? this
                : new MoeCacheConfig(mode, budgetMiB, options, prefetchZones, hostedMaxRows, tailSkipRank);
    }

    /**
     * This configuration with another prefetch zone count.
     *
     * @throws IllegalArgumentException naming the property for a negative count
     */
    public MoeCacheConfig withPrefetchZones(int zones) {
        if (zones < 0) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_PREFETCH + " must be >= 0, got " + zones);
        }
        return new MoeCacheConfig(mode, budgetMiB, ranking, zones, hostedMaxRows, tailSkipRank);
    }

    /**
     * This configuration with another hosted routed-row bound.
     *
     * @throws IllegalArgumentException naming the property for a negative count
     */
    public MoeCacheConfig withHostedMaxRows(int rows) {
        if (rows < 0) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_HOSTED_ROWS + " must be >= 0, got " + rows);
        }
        return new MoeCacheConfig(mode, budgetMiB, ranking, prefetchZones, rows, tailSkipRank);
    }

    /**
     * This configuration with another hosted tail-skip rank ({@link MoeTailSkip}).
     *
     * @throws IllegalArgumentException naming the property for a rank outside {@code 0..}{@value #TAIL_SKIP_MAX_RANK}
     */
    public MoeCacheConfig withTailSkipRank(int rank) {
        if (rank < MoeTailSkip.OFF || rank > TAIL_SKIP_MAX_RANK) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_TAIL_SKIP + " must be " + MoeTailSkip.OFF
                    + " (off) or 1-" + TAIL_SKIP_MAX_RANK + ", got " + rank);
        }
        return rank == tailSkipRank ? this
                : new MoeCacheConfig(mode, budgetMiB, ranking, prefetchZones, hostedMaxRows, rank);
    }

    /**
     * This configuration with another budget in MiB: {@code neuralfree.moe.cache.mb} overrides the budget the
     * {@code neuralfree.moe.cache} value gave.
     *
     * @throws IllegalArgumentException for a negative budget other than {@link #BUDGET_AUTO}
     */
    public MoeCacheConfig withBudgetMiB(long budgetMiB) {
        if (budgetMiB < BUDGET_AUTO) {
            throw new IllegalArgumentException(PROP_MOE_CACHE + " budget must be >= 0 (or " + BUDGET_AUTO
                    + " for the VRAM left after loading), got " + budgetMiB);
        }
        return budgetMiB == this.budgetMiB ? this
                : new MoeCacheConfig(mode, budgetMiB, ranking, prefetchZones, hostedMaxRows, tailSkipRank);
    }

    /**
     * This configuration turned {@code on} with a budget in MiB: {@code neuralfree.moe.cache.mb} enables the cache
     * on its own; the ranking options and the prefetch settings are kept.
     *
     * @throws IllegalArgumentException for a negative budget
     */
    public MoeCacheConfig onWithBudgetMiB(long budgetMiB) {
        if (budgetMiB < 0) {
            throw new IllegalArgumentException(PROP_MOE_CACHE + " budget must be >= 0, got " + budgetMiB);
        }
        return new MoeCacheConfig(Mode.ON, budgetMiB, ranking, prefetchZones, hostedMaxRows, tailSkipRank);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof MoeCacheConfig)) {
            return false;
        }
        MoeCacheConfig other = (MoeCacheConfig) o;
        return mode == other.mode && budgetMiB == other.budgetMiB
                && ranking.equals(other.ranking) && prefetchZones == other.prefetchZones
                && hostedMaxRows == other.hostedMaxRows && tailSkipRank == other.tailSkipRank;
    }

    @Override
    public int hashCode() {
        return Objects.hash(mode, budgetMiB, ranking, prefetchZones, hostedMaxRows, tailSkipRank);
    }

    @Override
    public String toString() {
        String m = mode == Mode.ON && budgetMiB != BUDGET_AUTO
                ? budgetMiB + " MiB" : mode.name().toLowerCase(Locale.ROOT);
        return "moeCache=" + m + (ranking.isDefault() ? "" : ", " + ranking)
                + (prefetchZones != PREFETCH_ZONES_DEFAULT ? ", moeCachePrefetch=" + prefetchZones : "")
                + (hostedMaxRows != 0 ? ", moeCacheHostedRows=" + hostedMaxRows : "")
                + (tailSkipRank != MoeTailSkip.OFF ? ", moeCacheTailSkip=" + tailSkipRank : "");
    }
}
