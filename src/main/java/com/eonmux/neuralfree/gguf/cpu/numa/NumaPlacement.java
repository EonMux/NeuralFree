package com.eonmux.neuralfree.gguf.cpu.numa;

import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.Objects;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Places the CPU pool's threads on NUMA nodes ({@code set_numa_thread_affinity} in
 * {@code ggml/src/ggml-cpu/ggml-cpu.c}): {@code distribute} gives thread {@code n} the CPUs of node
 * {@code n % n_nodes}, {@code isolate} gives every thread the node the process started on, and {@code numactl}
 * gives every thread the CPU set the process inherited from {@code numactl}.
 *
 * <p>Placement applies only when the host has more than one NUMA node; a single-node host or one without the Linux
 * topology files leaves threads where the OS schedules them. The strategy comes from the
 * {@value NumaStrategy#PROPERTY} property. Immutable.
 */
public final class NumaPlacement {

    private static final Logger LOGGER = LoggingUtil.getLogger(NumaPlacement.class);
    private static final AtomicBoolean BALANCING_WARNED = new AtomicBoolean();
    private static final AtomicBoolean SINGLE_NODE_LOGGED = new AtomicBoolean();

    private final NumaStrategy strategy;
    private final NumaTopology topology;
    private final boolean active;

    private NumaPlacement(NumaStrategy strategy, NumaTopology topology) {
        this.strategy = strategy;
        this.topology = topology;
        this.active = strategy.enabled() && topology != null && topology.isNuma();
    }

    /** No placement. */
    public static NumaPlacement disabled() {
        return new NumaPlacement(NumaStrategy.DISABLED, null);
    }

    /**
     * The placement of {@code properties}' {@value NumaStrategy#PROPERTY}; {@link #disabled()} when it is absent,
     * blank or {@code disabled}.
     *
     * @throws IllegalArgumentException naming the property when its value is not a strategy name
     */
    public static NumaPlacement fromProperties(Properties properties) {
        NumaStrategy strategy = NumaStrategy.fromProperties(properties);
        if (!strategy.enabled()) {
            return disabled();
        }
        return of(strategy, NumaTopology.host());
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
        NumaPlacement placement = new NumaPlacement(strategy, topology);
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

    /** Whether placement happens: a strategy is set and the host has more than one NUMA node. */
    public boolean active() {
        return active;
    }

    /**
     * The CPUs worker {@code workerIndex} runs on: its node's CPUs under {@code distribute}, the starting node's
     * under {@code isolate}, the inherited CPU set under {@code numactl}.
     *
     * @return the CPUs, or null when placement is not active
     */
    public int[] cpusForWorker(int workerIndex) {
        if (!active) {
            return null;
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
            if (strategy == NumaStrategy.NUMACTL) {
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
