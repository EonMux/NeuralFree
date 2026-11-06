package com.eonmux.neuralfree.gguf.engine;

import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * The expert-parallel load balancer configuration (vLLM's {@code --enable-eplb} and {@code EplbConfig}): the expert
 * placement of a tensor split is packed by recorded routed-row load instead of the contiguous expert blocks.
 *
 * <ul>
 *   <li>{@code neuralfree.eplb}: {@code off} (default) keeps the contiguous blocks, {@code on} places the experts by
 *       load ({@link #PROP_EPLB}).</li>
 *   <li>{@code neuralfree.eplb.profile}: the expert-load profile read at load ({@link #PROP_EPLB_PROFILE}), a
 *       {@code moe-cache-ranking.bin} file of routed rows per (layer, expert). The layers it counts are balanced;
 *       the others keep their contiguous blocks.</li>
 *   <li>{@code neuralfree.eplb.save}: the file the routed rows counted while this model runs are written to when it
 *       closes ({@link #PROP_EPLB_SAVE}), for the next load's placement.</li>
 *   <li>{@code neuralfree.eplb.policy}: the placement policy ({@link #PROP_EPLB_POLICY}); {@code default} is vLLM's
 *       weighted packing, {@code hierarchical} packs expert groups to nodes first.</li>
 *   <li>{@code neuralfree.eplb.num-redundant-experts}: extra physical replicas of the heaviest experts
 *       ({@link #PROP_EPLB_NUM_REDUNDANT}); the physical count must divide over the ranks.</li>
 *   <li>{@code neuralfree.eplb.num-groups}: the logical expert group count of the hierarchical policy
 *       ({@link #PROP_EPLB_NUM_GROUPS}); 0 takes each layer's router groups.</li>
 *   <li>{@code neuralfree.eplb.num-nodes}: the server node count of the hierarchical policy
 *       ({@link #PROP_EPLB_NUM_NODES}).</li>
 *   <li>{@code neuralfree.eplb.rebalance}: rearrange the experts between steps from the live load
 *       ({@link #PROP_EPLB_REBALANCE}); the load of {@code window-size} steps is collected and a rearrangement runs
 *       every {@code step-interval} steps ({@link #PROP_EPLB_STEP_INTERVAL}, {@link #PROP_EPLB_WINDOW_SIZE}).</li>
 * </ul>
 *
 * <p>An enabled configuration needs a profile or a save path, and the load needs expert parallelism
 * ({@link EngineConfig#expertParallel()}); {@link EngineConfig.Builder#build()} checks both. Runtime rebalancing
 * needs only {@code neuralfree.eplb=on} and an initial placement, which the profile, the save file or the contiguous
 * blocks provide. Immutable.</p>
 */
public final class EplbConfig {

    /** Property holding {@code on} or {@code off}. */
    public static final String PROP_EPLB = "neuralfree.eplb";
    /** Property naming the expert-load profile read at load. */
    public static final String PROP_EPLB_PROFILE = "neuralfree.eplb.profile";
    /** Property naming the file the counted expert load is written to at close. */
    public static final String PROP_EPLB_SAVE = "neuralfree.eplb.save";
    /** Property naming the placement policy. */
    public static final String PROP_EPLB_POLICY = "neuralfree.eplb.policy";
    /** Property holding the redundant physical expert count. */
    public static final String PROP_EPLB_NUM_REDUNDANT = "neuralfree.eplb.num-redundant-experts";
    /** Property holding the hierarchical policy's logical expert group count. */
    public static final String PROP_EPLB_NUM_GROUPS = "neuralfree.eplb.num-groups";
    /** Property holding the hierarchical policy's server node count. */
    public static final String PROP_EPLB_NUM_NODES = "neuralfree.eplb.num-nodes";
    /** Property holding {@code on} or {@code off} for runtime rearrangement. */
    public static final String PROP_EPLB_REBALANCE = "neuralfree.eplb.rebalance";
    /** Property holding the steps between runtime rearrangements. */
    public static final String PROP_EPLB_STEP_INTERVAL = "neuralfree.eplb.step-interval";
    /** Property holding the steps of live load a rearrangement reads. */
    public static final String PROP_EPLB_WINDOW_SIZE = "neuralfree.eplb.window-size";

    /** The expert placement policy. */
    public enum Policy {
        /** vLLM's {@code default}: heaviest experts first into the lightest rank with a free slot. */
        DEFAULT,
        /** vLLM's hierarchical rearrangement: groups to nodes, replicas within a node, physical slots to its ranks. */
        HIERARCHICAL;

        /** The policy name the configuration and the reports use. */
        public String label() {
            return name().toLowerCase(Locale.ROOT);
        }

        /**
         * A policy name, ignoring case.
         *
         * @throws IllegalArgumentException naming the valid policies for any other value
         */
        public static Policy parse(String value) {
            String s = Objects.requireNonNull(value, "policy").trim().toLowerCase(Locale.ROOT);
            if (s.equals("default")) {
                return DEFAULT;
            }
            if (s.equals("hierarchical")) {
                return HIERARCHICAL;
            }
            throw new IllegalArgumentException("expert placement policy '" + value + "' is not supported; use"
                    + " default (vLLM's weighted packing) or hierarchical (groups to nodes first)");
        }
    }

    private static final EplbConfig OFF = new EplbConfig(false, null, null, Policy.DEFAULT, 0, 0, 1, false, 1000, 8);

    private final boolean enabled;
    private final Path profile;
    private final Path save;
    private final Policy policy;
    private final int numRedundant;
    private final int numGroups;
    private final int numNodes;
    private final boolean rebalance;
    private final int stepInterval;
    private final int windowSize;

    private EplbConfig(boolean enabled, Path profile, Path save, Policy policy, int numRedundant, int numGroups,
                       int numNodes, boolean rebalance, int stepInterval, int windowSize) {
        this.enabled = enabled;
        this.profile = profile;
        this.save = save;
        this.policy = Objects.requireNonNull(policy, "policy");
        this.numRedundant = numRedundant;
        this.numGroups = numGroups;
        this.numNodes = numNodes;
        this.rebalance = rebalance;
        this.stepInterval = stepInterval;
        this.windowSize = windowSize;
    }

    /** The default: off, no profile and no save. */
    public static EplbConfig off() {
        return OFF;
    }

    /**
     * A configuration with the default policy settings (no replicas, one node, no runtime rearrangement).
     *
     * @param enabled whether the load balances the experts by recorded load
     * @param profile the profile read at load, or null for none
     * @param save    the file the counted load is written to at close, or null for none
     * @throws IllegalArgumentException when a path is given with the balancer off
     */
    public static EplbConfig of(boolean enabled, Path profile, Path save, Policy policy) {
        return of(enabled, profile, save, policy, 0, 0, 1, false, 1000, 8);
    }

    /**
     * A configuration.
     *
     * @param enabled       whether the load balances the experts by recorded load
     * @param profile       the profile read at load, or null for none
     * @param save          the file the counted load is written to at close, or null for none
     * @param policy        the placement policy
     * @param numRedundant  redundant physical experts per layer
     * @param numGroups     the hierarchical policy's group count, 0 to take each layer's router groups
     * @param numNodes      the hierarchical policy's node count
     * @param rebalance     whether the placement is rearranged from the live load between steps
     * @param stepInterval  steps between runtime rearrangements
     * @param windowSize    steps of live load a rearrangement reads
     * @throws IllegalArgumentException when a path is given with the balancer off, or a count is out of range
     */
    public static EplbConfig of(boolean enabled, Path profile, Path save, Policy policy, int numRedundant,
                                int numGroups, int numNodes, boolean rebalance, int stepInterval, int windowSize) {
        if (!enabled && (profile != null || save != null)) {
            throw new IllegalArgumentException((profile != null ? PROP_EPLB_PROFILE : PROP_EPLB_SAVE)
                    + " needs " + PROP_EPLB + "=on; the expert placement is only balanced when the balancer runs");
        }
        if (!enabled && (numRedundant != 0 || rebalance)) {
            throw new IllegalArgumentException(PROP_EPLB_NUM_REDUNDANT + " and " + PROP_EPLB_REBALANCE + " need "
                    + PROP_EPLB + "=on; the expert placement is only rearranged when the balancer runs");
        }
        if (numRedundant < 0) {
            throw new IllegalArgumentException(PROP_EPLB_NUM_REDUNDANT + " must be >= 0, got " + numRedundant);
        }
        if (numGroups < 0) {
            throw new IllegalArgumentException(PROP_EPLB_NUM_GROUPS + " must be >= 0 (0 takes the router's groups),"
                    + " got " + numGroups);
        }
        if (numNodes < 1) {
            throw new IllegalArgumentException(PROP_EPLB_NUM_NODES + " must be >= 1, got " + numNodes);
        }
        if (stepInterval < 1) {
            throw new IllegalArgumentException(PROP_EPLB_STEP_INTERVAL + " must be >= 1, got " + stepInterval);
        }
        if (windowSize < 1) {
            throw new IllegalArgumentException(PROP_EPLB_WINDOW_SIZE + " must be >= 1, got " + windowSize);
        }
        return enabled ? new EplbConfig(true, profile, save, policy, numRedundant, numGroups, numNodes, rebalance,
                stepInterval, windowSize) : OFF;
    }

    /** The configuration of a property set; an unset key keeps its default. */
    public static EplbConfig fromProperties(Properties props) {
        Objects.requireNonNull(props, "props");
        boolean enabled = mode(EngineProperties.stringProperty(props, PROP_EPLB));
        Path profile = path(props, PROP_EPLB_PROFILE);
        Path save = path(props, PROP_EPLB_SAVE);
        String policyName = EngineProperties.stringProperty(props, PROP_EPLB_POLICY);
        Policy policy;
        try {
            policy = policyName == null ? Policy.DEFAULT : Policy.parse(policyName);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_EPLB_POLICY + ": " + e.getMessage(), e);
        }
        int numRedundant = integer(props, PROP_EPLB_NUM_REDUNDANT, 0);
        int numGroups = integer(props, PROP_EPLB_NUM_GROUPS, 0);
        int numNodes = integer(props, PROP_EPLB_NUM_NODES, 1);
        boolean rebalance = bool(EngineProperties.stringProperty(props, PROP_EPLB_REBALANCE),
                PROP_EPLB_REBALANCE);
        int stepInterval = integer(props, PROP_EPLB_STEP_INTERVAL, 1000);
        int windowSize = integer(props, PROP_EPLB_WINDOW_SIZE, 8);
        return of(enabled, profile, save, policy, numRedundant, numGroups, numNodes, rebalance, stepInterval,
                windowSize);
    }

    /** A whole-number property, or {@code fallback} when unset. */
    private static int integer(Properties props, String property, int fallback) {
        String value = EngineProperties.stringProperty(props, property);
        if (value == null) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + "='" + value + "' is not a whole number", e);
        }
    }

    /** {@code on}/{@code true}/{@code 1} or {@code off}/{@code false}/{@code 0}, ignoring case. */
    private static boolean mode(String value) {
        return bool(value, PROP_EPLB);
    }

    /** A switch property, or false when unset. */
    private static boolean bool(String value, String property) {
        if (value == null) {
            return false;
        }
        String s = value.trim().toLowerCase(Locale.ROOT);
        switch (s) {
            case "on":
            case "true":
            case "1":
                return true;
            case "off":
            case "false":
            case "0":
                return false;
            default:
                throw new IllegalArgumentException(property + "=" + value + " is not a mode; use on or off");
        }
    }

    private static Path path(Properties props, String property) {
        String value = EngineProperties.stringProperty(props, property);
        if (value == null) {
            return null;
        }
        try {
            return Paths.get(value);
        } catch (InvalidPathException e) {
            throw new IllegalArgumentException(property + "='" + value + "' is not a path: " + e.getMessage(), e);
        }
    }

    /** True when the load balances the experts by recorded load. */
    public boolean enabled() {
        return enabled;
    }

    /** The expert-load profile read at load, or null for none. */
    public Path profile() {
        return profile;
    }

    /** The file the counted load is written to at close, or null for none. */
    public Path save() {
        return save;
    }

    /** The placement policy. */
    public Policy policy() {
        return policy;
    }

    /** Redundant physical experts added to every layer's logical experts. */
    public int numRedundant() {
        return numRedundant;
    }

    /** The hierarchical policy's logical expert group count, or 0 to take each layer's router groups. */
    public int numGroups() {
        return numGroups;
    }

    /** The hierarchical policy's server node count. */
    public int numNodes() {
        return numNodes;
    }

    /** True when the placement is rearranged from the live load between steps. */
    public boolean rebalance() {
        return rebalance;
    }

    /** Steps between runtime rearrangements. */
    public int stepInterval() {
        return stepInterval;
    }

    /** Steps of live load a runtime rearrangement reads. */
    public int windowSize() {
        return windowSize;
    }

    /** True for the defaults: off, no profile and no save. */
    public boolean isDefault() {
        return !enabled && profile == null && save == null && numRedundant == 0 && !rebalance;
    }

    /** This configuration on or off; disabling drops the paths. */
    public EplbConfig withEnabled(boolean v) {
        return v ? new EplbConfig(true, profile, save, policy, numRedundant, numGroups, numNodes, rebalance,
                stepInterval, windowSize) : OFF;
    }

    /** This configuration reading {@code v} at load, or none for null. */
    public EplbConfig withProfile(Path v) {
        return new EplbConfig(enabled, v, save, policy, numRedundant, numGroups, numNodes, rebalance, stepInterval,
                windowSize);
    }

    /** This configuration writing its counted load to {@code v} at close, or nothing for null. */
    public EplbConfig withSave(Path v) {
        return new EplbConfig(enabled, profile, v, policy, numRedundant, numGroups, numNodes, rebalance, stepInterval,
                windowSize);
    }

    /** This configuration with another placement policy. */
    public EplbConfig withPolicy(Policy v) {
        return new EplbConfig(enabled, profile, save, v, numRedundant, numGroups, numNodes, rebalance, stepInterval,
                windowSize);
    }

    /** This configuration with another redundant expert count. */
    public EplbConfig withNumRedundant(int v) {
        return new EplbConfig(enabled, profile, save, policy, v, numGroups, numNodes, rebalance, stepInterval,
                windowSize);
    }

    /** This configuration with another hierarchical group count (0 takes each layer's router groups). */
    public EplbConfig withNumGroups(int v) {
        return new EplbConfig(enabled, profile, save, policy, numRedundant, v, numNodes, rebalance, stepInterval,
                windowSize);
    }

    /** This configuration with another hierarchical node count. */
    public EplbConfig withNumNodes(int v) {
        return new EplbConfig(enabled, profile, save, policy, numRedundant, numGroups, v, rebalance, stepInterval,
                windowSize);
    }

    /** This configuration with runtime rearrangement on or off. */
    public EplbConfig withRebalance(boolean v) {
        return new EplbConfig(enabled, profile, save, policy, numRedundant, numGroups, numNodes, v, stepInterval,
                windowSize);
    }

    /** This configuration with another rearrangement interval and load window. */
    public EplbConfig withSchedule(int stepIntervalV, int windowSizeV) {
        return new EplbConfig(enabled, profile, save, policy, numRedundant, numGroups, numNodes, rebalance,
                stepIntervalV, windowSizeV);
    }

    @Override
    public String toString() {
        if (!enabled) {
            return "off";
        }
        return "on (policy " + policy.label() + (profile != null ? ", profile " + profile : "")
                + (save != null ? ", save " + save : "")
                + (numRedundant != 0 ? ", " + numRedundant + " redundant" : "")
                + (numGroups != 0 ? ", " + numGroups + " groups" : "")
                + (numNodes != 1 ? ", " + numNodes + " nodes" : "")
                + (rebalance ? ", rebalance every " + stepInterval + " step(s) over " + windowSize : "") + ")";
    }
}
