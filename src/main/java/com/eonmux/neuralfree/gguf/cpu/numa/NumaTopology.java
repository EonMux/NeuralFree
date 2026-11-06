package com.eonmux.neuralfree.gguf.cpu.numa;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The host's NUMA layout: the CPUs of each node, the CPUs the calling thread may run on and the CPU and node it
 * started on ({@code ggml_numa_nodes} in {@code ggml/src/ggml-cpu/ggml-cpu.c}).
 *
 * <p>Read from the Linux sysfs and procfs trees ({@code /sys/devices/system/node/nodeN/cpulist}, the CPU
 * directories of a node, {@code /proc/self/status}) through {@link CpuAffinity}; a host without them gets
 * {@link #none()}, and a machine with one node reports {@link #isNuma()} false, in which case placement has nothing
 * to do. Immutable.
 */
public final class NumaTopology {

    private static final String NODE_ROOT = "/sys/devices/system/node";
    private static final String CPU_ROOT = "/sys/devices/system/cpu";

    private final int[][] nodeCpus;
    private final int[] affinityCpus;
    private final int currentCpu;
    private final int currentNode;

    private NumaTopology(int[][] nodes, int[] affinity, int currentCpu) {
        this.nodeCpus = nodes;
        this.affinityCpus = affinity;
        this.currentCpu = currentCpu;
        this.currentNode = nodeOf(nodes, affinity, currentCpu);
    }

    /** A topology with no node, the reading a host without sysfs NUMA information gets. */
    public static NumaTopology none() {
        return new NumaTopology(new int[0][], new int[0], -1);
    }

    /**
     * The host's topology ({@link #read()}), cached: the layout does not change while the process runs.
     */
    public static NumaTopology host() {
        return HostHolder.TOPOLOGY;
    }

    private static final class HostHolder {
        static final NumaTopology TOPOLOGY = read();
    }

    /**
     * Reads the host's topology: the nodes of {@code /sys/devices/system/node/possible}, the CPUs of each node
     * ({@code cpulist}, else its {@code cpuN} directories), the calling thread's affinity and the CPU it runs on.
     */
    public static NumaTopology read() {
        if (!CpuAffinity.available() && !CpuAffinity.exists(Paths.get(NODE_ROOT))) {
            return none();
        }
        int[][] nodes = readNodes();
        int[] affinity = CpuAffinity.currentThreadCpus();
        if (affinity.length == 0) {
            affinity = affinityFromStatus();
        }
        return of(nodes, affinity, CpuAffinity.currentCpu());
    }

    /**
     * A topology of explicit values.
     *
     * @param nodes     element {@code n}: the CPUs of node {@code n}; every CPU must be non-negative and a CPU may
     *                  appear in at most one node
     * @param affinity  the CPUs the process may run on, or empty when unknown
     * @param currentCpu the CPU the process started on, or -1 when unknown
     * @throws IllegalArgumentException when a CPU is negative or two nodes claim the same CPU
     */
    public static NumaTopology of(int[][] nodes, int[] affinity, int currentCpu) {
        Objects.requireNonNull(nodes, "nodes");
        Objects.requireNonNull(affinity, "affinity");
        int[][] copy = new int[nodes.length][];
        Set<Integer> seen = new HashSet<>();
        for (int n = 0; n < nodes.length; n++) {
            Objects.requireNonNull(nodes[n], "CPUs of node " + n);
            copy[n] = nodes[n].clone();
            Arrays.sort(copy[n]);
            for (int cpu : copy[n]) {
                if (cpu < 0) {
                    throw new IllegalArgumentException("CPU " + cpu + " of node " + n + " is negative");
                }
                if (!seen.add(Integer.valueOf(cpu))) {
                    throw new IllegalArgumentException("CPU " + cpu + " is listed by more than one NUMA node");
                }
            }
        }
        int[] affinityCopy = affinity.clone();
        Arrays.sort(affinityCopy);
        return new NumaTopology(copy, affinityCopy, currentCpu);
    }

    /** Whether the host reported any NUMA node. */
    public boolean available() {
        return nodeCpus.length > 0;
    }

    /** Whether the host has more than one NUMA node, the condition that makes placement worthwhile. */
    public boolean isNuma() {
        return nodeCpus.length > 1;
    }

    /** Number of NUMA nodes. */
    public int nodeCount() {
        return nodeCpus.length;
    }

    /**
     * The CPUs of node {@code node}, in increasing order (a copy).
     *
     * @throws IllegalArgumentException when there is no such node
     */
    public int[] cpusOf(int node) {
        if (node < 0 || node >= nodeCpus.length) {
            throw new IllegalArgumentException("NUMA node " + node + " does not exist; the host has "
                    + nodeCpus.length);
        }
        return nodeCpus[node].clone();
    }

    /** The CPUs the process may run on, in increasing order (a copy); empty when unknown. */
    public int[] affinityCpus() {
        return affinityCpus.clone();
    }

    /** The CPU the process started on, or -1 when unknown. */
    public int currentCpu() {
        return currentCpu;
    }

    /** The node of {@link #currentCpu()} (or of the first affinity CPU when the CPU is unknown), or -1 with no node. */
    public int currentNode() {
        return currentNode;
    }

    /** Every CPU number of every node, in increasing order. */
    public int[] cpuNumbers() {
        List<Integer> all = new ArrayList<>();
        for (int[] cpus : nodeCpus) {
            for (int cpu : cpus) {
                all.add(Integer.valueOf(cpu));
            }
        }
        all.sort(null);
        int[] out = new int[all.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = all.get(i).intValue();
        }
        return out;
    }

    /** A one-line description of the layout for logs. */
    public String describe() {
        if (!available()) {
            return "no NUMA nodes";
        }
        StringBuilder sb = new StringBuilder();
        sb.append(nodeCpus.length).append(" NUMA node(s)");
        for (int n = 0; n < nodeCpus.length; n++) {
            sb.append(n == 0 ? ": node" : ", node").append(n).append(" CPUs ");
            appendCpuList(sb, nodeCpus[n]);
        }
        if (currentNode >= 0) {
            sb.append("; the process started on node ").append(currentNode);
            if (currentCpu >= 0) {
                sb.append(" (CPU ").append(currentCpu).append(')');
            }
        }
        return sb.toString();
    }

    private static void appendCpuList(StringBuilder sb, int[] cpus) {
        if (cpus.length == 0) {
            sb.append("none");
            return;
        }
        int first = cpus[0];
        int last = first;
        boolean wrote = false;
        for (int i = 1; i < cpus.length; i++) {
            if (cpus[i] == last + 1) {
                last = cpus[i];
                continue;
            }
            if (wrote) {
                sb.append(',');
            }
            sb.append(first);
            if (last != first) {
                sb.append('-').append(last);
            }
            wrote = true;
            first = last = cpus[i];
        }
        if (wrote) {
            sb.append(',');
        }
        sb.append(first);
        if (last != first) {
            sb.append('-').append(last);
        }
    }

    // ------------------------------------------------------------------ reading

    /** The nodes of {@code /sys/devices/system/node/possible}, else node0, node1, ... until one is missing. */
    private static int[][] readNodes() {
        Path root = Paths.get(NODE_ROOT);
        if (!CpuAffinity.exists(root)) {
            return new int[0][];
        }
        int[] possible = CpuLists.parse(CpuAffinity.read(root.resolve("possible").toString()));
        List<int[]> nodes = new ArrayList<>();
        if (possible.length > 0) {
            for (int node : possible) {
                Path dir = root.resolve("node" + node);
                if (CpuAffinity.exists(dir)) {
                    nodes.add(nodeCpus(dir));
                }
            }
        } else {
            for (int node = 0; ; node++) {
                Path dir = root.resolve("node" + node);
                if (!CpuAffinity.exists(dir)) {
                    break;
                }
                nodes.add(nodeCpus(dir));
            }
        }
        return nodes.toArray(new int[0][]);
    }

    /** The CPUs of a node: its {@code cpulist}, else the CPU directories it holds. */
    private static int[] nodeCpus(Path dir) {
        int[] cpus = CpuLists.parse(CpuAffinity.read(dir.resolve("cpulist").toString()));
        if (cpus.length > 0) {
            return cpus;
        }
        int maxCpu = highestCpu(CPU_ROOT + "/possible");
        if (maxCpu < 0) {
            return new int[0];
        }
        List<Integer> found = new ArrayList<>();
        for (int cpu = 0; cpu <= maxCpu; cpu++) {
            if (CpuAffinity.exists(dir.resolve("cpu" + cpu))) {
                found.add(Integer.valueOf(cpu));
            }
        }
        int[] out = new int[found.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = found.get(i).intValue();
        }
        return out;
    }

    /** The highest CPU number of a cpulist file, or -1 when it does not read. */
    private static int highestCpu(String path) {
        int max = -1;
        for (int cpu : CpuLists.parse(CpuAffinity.read(path))) {
            max = Math.max(max, cpu);
        }
        return max;
    }

    /** {@code Cpus_allowed_list} of {@code /proc/self/status}, the affinity fallback without libc. */
    private static int[] affinityFromStatus() {
        String status = CpuAffinity.read("/proc/self/status");
        if (status == null) {
            return new int[0];
        }
        for (String line : status.split("\n")) {
            if (line.startsWith("Cpus_allowed_list:")) {
                try {
                    return CpuLists.parse(line.substring("Cpus_allowed_list:".length()));
                } catch (IllegalArgumentException e) {
                    return new int[0];
                }
            }
        }
        return new int[0];
    }

    /** The node holding {@code cpu}, else the node of the first affinity CPU, else node 0. */
    private static int nodeOf(int[][] nodes, int[] affinity, int cpu) {
        if (nodes.length == 0) {
            return -1;
        }
        int node = nodeHolding(nodes, cpu);
        if (node >= 0) {
            return node;
        }
        for (int candidate : affinity) {
            node = nodeHolding(nodes, candidate);
            if (node >= 0) {
                return node;
            }
        }
        return 0;
    }

    private static int nodeHolding(int[][] nodes, int cpu) {
        if (cpu < 0) {
            return -1;
        }
        for (int n = 0; n < nodes.length; n++) {
            for (int candidate : nodes[n]) {
                if (candidate == cpu) {
                    return n;
                }
            }
        }
        return -1;
    }
}
