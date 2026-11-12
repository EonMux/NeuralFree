package com.eonmux.neuralfree.gguf.cpu.numa;

import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Objects;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

/**
 * Places the CPU pool's threads on NUMA nodes ({@code set_numa_thread_affinity} in
 * {@code ggml/src/ggml-cpu/ggml-cpu.c}): {@code distribute} gives thread {@code n} the CPUs of node
 * {@code n % n_nodes}, {@code isolate} gives every thread the node the process started on, and {@code numactl}
 * gives every thread the CPU set the process inherited from {@code numactl}.
 *
 * <p>NUMA placement applies only when the host has more than one NUMA node; a single-node host or one without the
 * Linux topology files leaves threads where the OS schedules them. The strategy comes from the
 * {@value NumaStrategy#PROPERTY} property. Without an active NUMA placement, {@value #PIN_PROPERTY} restricts every
 * pool thread to one CPU set instead, llama.cpp's {@code --cpu-mask} / {@code --cpu-range}: {@code policy} names the
 * CPUs of the {@link CpuThreadPolicy} that sizes the pool (the performance cores by default), so the scheduler cannot
 * move a worker onto an efficiency core, and a cpulist such as {@code 16-23} names the CPUs directly. The set holds
 * every pool thread and the scheduler balances them within it; under it the calling thread waits while the pinned
 * workers run each loop, as under NUMA placement. Immutable.
 */
public final class NumaPlacement {

    private static final Logger LOGGER = LoggingUtil.getLogger(NumaPlacement.class);
    private static final AtomicBoolean BALANCING_WARNED = new AtomicBoolean();
    private static final AtomicBoolean SINGLE_NODE_LOGGED = new AtomicBoolean();

    /**
     * System property restricting the pool threads to one CPU set when no NUMA placement is active: absent, blank or
     * {@code off} for none, {@code policy} for the CPUs of the configured {@link CpuThreadPolicy}, or a cpulist.
     */
    public static final String PIN_PROPERTY = "neuralfree.cpu.pin";

    private final NumaStrategy strategy;
    private final NumaTopology topology;
    private final int[] pinned;
    private final boolean active;

    private NumaPlacement(NumaStrategy strategy, NumaTopology topology, int[] pinned) {
        this.strategy = strategy;
        this.topology = topology;
        this.pinned = pinned;
        this.active = pinned != null || strategy.enabled() && topology != null && topology.isNuma();
    }

    /** No placement. */
    public static NumaPlacement disabled() {
        return new NumaPlacement(NumaStrategy.DISABLED, null, null);
    }

    /**
     * Every pool thread on the CPU set {@code cpus}, the {@value #PIN_PROPERTY} placement.
     *
     * @throws IllegalArgumentException when {@code cpus} is empty or holds a negative CPU number
     */
    public static NumaPlacement pinned(int[] cpus) {
        if (cpus.length == 0) {
            throw new IllegalArgumentException(PIN_PROPERTY + " needs at least one CPU");
        }
        final int[] sorted = cpus.clone();
        Arrays.sort(sorted);
        if (sorted[0] < 0) {
            throw new IllegalArgumentException(PIN_PROPERTY + " CPU numbers are not negative, got " + sorted[0]);
        }
        LOGGER.info("{}: the CPU pool runs on CPUs {}", PIN_PROPERTY, CpuLists.format(sorted));
        return new NumaPlacement(NumaStrategy.DISABLED, null, sorted);
    }

    /**
     * The placement of {@code properties}: the active NUMA placement of {@value NumaStrategy#PROPERTY}, else the
     * CPU set of {@value #PIN_PROPERTY}, else {@link #disabled()}. A pin set that an active NUMA placement overrides
     * is logged as having no effect.
     *
     * @throws IllegalArgumentException naming the property when a value is not a strategy name or CPU set
     */
    public static NumaPlacement fromProperties(Properties properties) {
        return fromProperties(properties, NumaTopology::host);
    }

    /** {@link #fromProperties(Properties)} over the topology {@code host} reads, asked only for a NUMA strategy. */
    static NumaPlacement fromProperties(Properties properties, Supplier<NumaTopology> host) {
        NumaStrategy strategy = NumaStrategy.fromProperties(properties);
        int[] pin = pinCpus(properties);
        if (strategy.enabled()) {
            NumaPlacement numa = of(strategy, host.get());
            if (numa.active() && pin != null) {
                LOGGER.warn("{}={} has no effect: {}={} places the CPU pool on its NUMA nodes", PIN_PROPERTY,
                        properties.getProperty(PIN_PROPERTY).trim(), NumaStrategy.PROPERTY, strategy.label());
            }
            if (numa.active() || pin == null) {
                return numa;
            }
        }
        return pin == null ? disabled() : pinned(pin);
    }

    /**
     * The CPU set {@value #PIN_PROPERTY} names in {@code properties}: null for none, the configured
     * {@link CpuThreadPolicy}'s CPUs for {@code policy} (null on a host without the Linux CPU topology), or the
     * cpulist's CPUs.
     *
     * @throws IllegalArgumentException naming the property when the value is neither a keyword nor a cpulist
     */
    static int[] pinCpus(Properties properties) {
        final String v = properties.getProperty(PIN_PROPERTY);
        if (v == null || v.trim().isEmpty() || v.trim().equalsIgnoreCase("off")) {
            return null;
        }
        if (v.trim().equalsIgnoreCase("policy")) {
            final int[] cpus = CpuThreadPolicy.host().cpus(CpuThreadPolicy.configuredPolicy());
            if (cpus.length == 0) {
                LOGGER.info("{}=policy has no effect: the host reports no CPU topology", PIN_PROPERTY);
                return null;
            }
            return cpus;
        }
        final int[] cpus;
        try {
            cpus = CpuLists.parse(v);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PIN_PROPERTY + " must be off, policy or a cpulist such as 16-23, got '"
                    + v + "': " + e.getMessage(), e);
        }
        if (cpus.length == 0) {
            throw new IllegalArgumentException(PIN_PROPERTY + " must name at least one CPU, got '" + v + "'");
        }
        return cpus;
    }

    /** The placement of {@link System#getProperties()}. */
    public static NumaPlacement fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    /**
     * Whether a property set's strategy places threads here: a strategy is set and the host has more than one NUMA
     * node. Reads no layout and logs nothing; {@link #fromProperties} is the reporting form.
     *
     * @throws IllegalArgumentException naming the property when its value is not a strategy name
     */
    public static boolean activeIn(Properties properties) {
        return NumaStrategy.fromProperties(properties).enabled() && NumaTopology.host().isNuma();
    }

    /**
     * The placement of a strategy over a topology, with the host's layout logged once.
     *
     * @throws NullPointerException when either argument is null
     */
    public static NumaPlacement of(NumaStrategy strategy, NumaTopology topology) {
        Objects.requireNonNull(strategy, "strategy");
        Objects.requireNonNull(topology, "topology");
        NumaPlacement placement = new NumaPlacement(strategy, topology, null);
        if (!strategy.enabled()) {
            return placement;
        }
        if (!topology.available()) {
            if (SINGLE_NODE_LOGGED.compareAndSet(false, true)) {
                LOGGER.info("{}={} has no effect: the host reports no NUMA topology", NumaStrategy.PROPERTY,
                        strategy.label());
            }
        } else if (!topology.isNuma()) {
            if (SINGLE_NODE_LOGGED.compareAndSet(false, true)) {
                LOGGER.info("{}={} has no effect: {}", NumaStrategy.PROPERTY, strategy.label(),
                        topology.describe());
            }
        } else {
            LOGGER.info("{}={}: {}", NumaStrategy.PROPERTY, strategy.label(), topology.describe());
            warnOnNumaBalancing();
        }
        return placement;
    }

    /** The configured strategy. */
    public NumaStrategy strategy() {
        return strategy;
    }

    /** The host layout the placement was built over, or null when disabled. */
    public NumaTopology topology() {
        return topology;
    }

    /**
     * Whether placement happens: a strategy is set and the host has more than one NUMA node, or the pool runs on a
     * {@value #PIN_PROPERTY} CPU set.
     */
    public boolean active() {
        return active;
    }

    /** The {@value #PIN_PROPERTY} CPU set (a copy), or null when the placement is not a pinned one. */
    public int[] pinnedCpus() {
        return pinned == null ? null : pinned.clone();
    }

    /**
     * The CPUs worker {@code workerIndex} runs on: its node's CPUs under {@code distribute}, the starting node's
     * under {@code isolate}, the inherited CPU set under {@code numactl}, the CPU set of a pinned placement.
     *
     * @return the CPUs, or null when placement is not active
     */
    public int[] cpusForWorker(int workerIndex) {
        if (!active) {
            return null;
        }
        if (pinned != null) {
            return pinned.clone();
        }
        switch (strategy) {
            case DISTRIBUTE:
                return topology.cpusOf(Math.floorMod(workerIndex, topology.nodeCount()));
            case ISOLATE:
                return topology.cpusOf(topology.currentNode());
            case NUMACTL:
                int[] affinity = topology.affinityCpus();
                return affinity.length == 0 ? null : affinity;
            default:
                return null;
        }
    }

    /**
     * Restricts the calling thread to {@link #cpusForWorker(int) its node's CPUs}. Called from the thread itself,
     * since Linux affinity is per thread; a no-op when placement is inactive. A failed call is logged and leaves
     * the thread where the OS schedules it.
     */
    public void pinCurrentThread(int workerIndex) {
        int[] cpus = cpusForWorker(workerIndex);
        if (cpus == null || cpus.length == 0) {
            return;
        }
        if (CpuAffinity.setCurrentThreadCpus(cpus)) {
            if (pinned != null) {
                LOGGER.debug("pinned CPU worker {} to CPUs {}", workerIndex, CpuLists.format(cpus));
            } else if (strategy == NumaStrategy.NUMACTL) {
                LOGGER.debug("pinned CPU worker {} to the inherited CPU set ({} CPU(s))", workerIndex,
                        cpus.length);
            } else {
                LOGGER.debug("pinned CPU worker {} to {} CPU(s) of node {}", workerIndex, cpus.length,
                        workerNode(workerIndex));
            }
        }
    }

    /** The node {@link #cpusForWorker(int)} places a worker on. */
    private int workerNode(int workerIndex) {
        return strategy == NumaStrategy.ISOLATE ? topology.currentNode()
                : Math.floorMod(workerIndex, topology.nodeCount());
    }

    /** ggml's warning: automatic NUMA balancing has been observed to hurt CPU inference. */
    private static void warnOnNumaBalancing() {
        String balancing = CpuAffinity.read("/proc/sys/kernel/numa_balancing");
        if (balancing != null && !balancing.trim().equals("0") && BALANCING_WARNED.compareAndSet(false, true)) {
            LOGGER.warn("/proc/sys/kernel/numa_balancing is enabled; automatic NUMA balancing has been observed to"
                    + " impair CPU inference, disable it with 'sysctl kernel.numa_balancing=0'");
        }
    }
}
