package com.eonmux.neuralfree.gguf.cpu.numa;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * The CPU pool size the CPU engine uses when no explicit thread count is set: the logical CPUs, one CPU per physical
 * core, or one CPU per performance core of the CPUs the process may run on, each capped at
 * {@link Runtime#availableProcessors()} (which follows a cgroup CPU quota and {@code -XX:ActiveProcessorCount}).
 *
 * <p>The cores are read from the Linux sysfs tree: {@code cpuN/topology/core_cpus_list} (else
 * {@code thread_siblings_list}) groups the SMT siblings of one core. Core classes come from one of two sources. The
 * hybrid PMU list {@code /sys/devices/cpu_core/cpus} names the performance cores of an Intel hybrid part. Otherwise
 * {@code cpuN/cpu_capacity} separates them when some cores fall below {@value #CLASS_RATIO_PERCENT}% of the highest
 * capacity, as the LITTLE cores of an ARM big.LITTLE part do; the smaller spread between the preferred and the other
 * cores of one class (ITMT, boost binning) stays one class. When every core's {@code regs/identification/midr_el1}
 * reads (ARM), a slower core's type must also differ from the faster cores' types, so cores of one design never
 * split; when one does not read, the capacity ratio alone decides. A host without that tree (Windows, macOS) gets the
 * processor count under every policy.
 *
 * <p>The default is {@link Policy#PERFORMANCE}: decode is bound by memory bandwidth, which the performance cores
 * already saturate, and a fork/join step waits for its slowest worker, so an efficiency core that runs a share of
 * every loop slows the whole step. That holds only while the process may run on every performance core; when its
 * affinity leaves some out, the efficiency cores it may use are the remaining parallelism, so every allowed physical
 * core counts. Immutable.
 */
public final class CpuThreadPolicy {

    /** System property selecting the policy: {@code logical}, {@code physical} or {@code performance}. */
    public static final String PROPERTY = "neuralfree.cpu.threads.policy";

    /** A core whose capacity is below this percentage of the highest one belongs to a slower core class. */
    static final int CLASS_RATIO_PERCENT = 75;

    private static final String CPU_ROOT = "/sys/devices/system/cpu";
    private static final String HYBRID_CORE = "/sys/devices/cpu_core/cpus";

    /** How the default thread count is derived from the host's cores. */
    public enum Policy {
        /** Every CPU the process may run on, SMT siblings and efficiency cores included. */
        LOGICAL,
        /** One CPU per physical core. */
        PHYSICAL,
        /**
         * One CPU per physical performance core while the process may run on all of them; every allowed physical
         * core otherwise, and on a host with one core class.
         */
        PERFORMANCE;

        /**
         * The policy named {@code text}, case-insensitively.
         *
         * @throws IllegalArgumentException naming the accepted values when the name is unknown
         */
        public static Policy parse(String text) {
            String s = Objects.requireNonNull(text, "policy").trim().toLowerCase(Locale.ROOT);
            for (Policy p : values()) {
                if (p.name().toLowerCase(Locale.ROOT).equals(s)) {
                    return p;
                }
            }
            throw new IllegalArgumentException(PROPERTY + " must be logical, physical or performance, got '"
                    + text + "'");
        }
    }

    /** The policy of an unset {@value #PROPERTY}. */
    public static final Policy DEFAULT_POLICY = Policy.PERFORMANCE;

    private final int[] logical;
    private final int[] physical;
    private final int[] performance;
    private final int limit;
    private final boolean topology;
    private final String classes;

    private CpuThreadPolicy(int[] logical, int[] physical, int[] performance, int limit, boolean topology,
                            String classes) {
        this.logical = logical;
        this.physical = physical;
        this.performance = performance;
        this.limit = limit;
        this.topology = topology;
        this.classes = classes;
    }

    /** The host's cores, cached: they do not change while the process runs. */
    public static CpuThreadPolicy host() {
        return HostHolder.POLICY;
    }

    private static final class HostHolder {
        static final CpuThreadPolicy POLICY = read(Paths.get(CPU_ROOT), Paths.get(HYBRID_CORE),
                CpuAffinity.currentThreadCpus(), Runtime.getRuntime().availableProcessors());
    }

    /**
     * Reads the cores under {@code cpuRoot} (the layout of {@code /sys/devices/system/cpu}) and the Intel hybrid
     * performance-core list at {@code hybridCoreList}.
     *
     * @param affinity       the CPUs the process may run on, or empty when unknown (every online CPU then counts)
     * @param processorLimit the processors the JVM may use ({@link Runtime#availableProcessors()}): every policy's
     *                       count is capped at it, and it is every policy's count when the tree lists no CPU
     * @throws IllegalArgumentException when {@code processorLimit} is not positive
     */
    public static CpuThreadPolicy read(Path cpuRoot, Path hybridCoreList, int[] affinity, int processorLimit) {
        Objects.requireNonNull(cpuRoot, "cpuRoot");
        Objects.requireNonNull(hybridCoreList, "hybridCoreList");
        Objects.requireNonNull(affinity, "affinity");
        if (processorLimit < 1) {
            throw new IllegalArgumentException("processor limit must be positive, got " + processorLimit);
        }
        int[] online = parseOrEmpty(read(cpuRoot.resolve("online")));
        int[] allowed = affinity.length == 0 ? online : intersect(online.length == 0 ? affinity : online, affinity);
        if (allowed.length == 0) {
            int[] none = new int[0];
            return new CpuThreadPolicy(none, none, none, processorLimit, false, "no sysfs CPU topology");
        }
        int[] all = online.length == 0 ? allowed : online;
        Map<Integer, Integer> coreOf = new TreeMap<>();
        for (int cpu : all) {
            coreOf.put(cpu, firstSibling(cpuRoot, cpu));
        }
        int[] physical = onePerCore(allowed, coreOf);
        Classes found = hybridClasses(hybridCoreList, all, coreOf);
        if (found == null) {
            found = capacityClasses(cpuRoot, onePerCore(all, coreOf), coreOf);
        }
        int[] perf = physical;
        String classes = "one core class";
        if (found != null) {
            int[] allowedPerf = filter(physical, found.performanceCores, coreOf);
            if (allowedPerf.length == found.performanceCores.size()) {
                perf = allowedPerf;
                classes = found.description;
            } else {
                classes = found.description + "; the affinity holds " + allowedPerf.length + " of the "
                        + found.performanceCores.size() + " performance cores, so every allowed physical core counts";
            }
        }
        return new CpuThreadPolicy(allowed, physical, perf, processorLimit, true, classes);
    }

    /**
     * The default thread count under the policy the {@value #PROPERTY} property names, {@link #DEFAULT_POLICY}
     * when it is unset.
     *
     * @throws IllegalArgumentException when the property names no policy
     */
    public int defaultThreads() {
        return threads(configuredPolicy());
    }

    /** The policy the {@value #PROPERTY} property names, {@link #DEFAULT_POLICY} when it is unset or blank. */
    public static Policy configuredPolicy() {
        String v = System.getProperty(PROPERTY);
        return v == null || v.trim().isEmpty() ? DEFAULT_POLICY : Policy.parse(v);
    }

    /**
     * The thread count of {@code policy}: the number of {@link #cpus(Policy) its CPUs}, at least 1 and at most the
     * processor limit; the processor limit itself on a host without a sysfs topology.
     */
    public int threads(Policy policy) {
        Objects.requireNonNull(policy, "policy");
        return topology ? Math.min(limit, Math.max(1, cpus(policy).length)) : limit;
    }

    /**
     * The CPUs {@code policy} counts, in increasing order (a copy), before the processor limit; empty on a host
     * without a sysfs topology.
     */
    public int[] cpus(Policy policy) {
        switch (Objects.requireNonNull(policy, "policy")) {
            case LOGICAL:
                return logical.clone();
            case PHYSICAL:
                return physical.clone();
            default:
                return performance.clone();
        }
    }

    /** One line naming the counts of each policy and how the core classes were found, for logs. */
    public String describe() {
        if (!topology) {
            return classes + ": " + limit + " available processor(s)";
        }
        String s = logical.length + " logical CPU(s), " + physical.length + " physical core(s), " + performance.length
                + " performance core(s) (" + classes + ")";
        return limit < logical.length ? s + ", at most " + limit + " available processor(s)" : s;
    }

    @Override
    public String toString() {
        return "CpuThreadPolicy{" + describe() + "}";
    }

    /** The performance cores (by first sibling) of a host with more than one core class, and how they were found. */
    private static final class Classes {
        final Set<Integer> performanceCores;
        final String description;

        Classes(Set<Integer> performanceCores, String description) {
            this.performanceCores = performanceCores;
            this.description = description;
        }
    }

    // ------------------------------------------------------------------ core classes

    /** The Intel hybrid classes: the cores of the listed CPUs, when the list holds some but not all CPUs. */
    private static Classes hybridClasses(Path hybridCoreList, int[] all, Map<Integer, Integer> coreOf) {
        int[] listed = intersect(all, parseOrEmpty(read(hybridCoreList)));
        if (listed.length == 0 || listed.length == all.length) {
            return null;
        }
        Set<Integer> cores = new TreeSet<>();
        for (int cpu : listed) {
            cores.add(coreOf.get(cpu));
        }
        return new Classes(cores, "hybrid performance cores " + CpuLists.format(listed));
    }

    /**
     * The capacity classes of the cores {@code firsts} (one CPU each): the cores below {@value #CLASS_RATIO_PERCENT}%
     * of the highest {@code cpu_capacity} are a slower class, unless their ARM core type matches a faster core's.
     * Null when every capacity reads and none is that low, or when a capacity does not read.
     */
    private static Classes capacityClasses(Path cpuRoot, int[] firsts, Map<Integer, Integer> coreOf) {
        long[] cap = new long[firsts.length];
        long max = 0;
        for (int i = 0; i < firsts.length; i++) {
            cap[i] = parseLong(read(cpuRoot.resolve("cpu" + firsts[i]).resolve("cpu_capacity")), -1L);
            if (cap[i] < 0) {
                return null;
            }
            max = Math.max(max, cap[i]);
        }
        long[] type = coreTypes(cpuRoot, firsts);
        Set<Long> fastTypes = new HashSet<>();
        boolean[] slow = new boolean[firsts.length];
        for (int i = 0; i < firsts.length; i++) {
            slow[i] = cap[i] * 100 < max * CLASS_RATIO_PERCENT;
            if (!slow[i] && type != null) {
                fastTypes.add(type[i]);
            }
        }
        Set<Integer> perf = new TreeSet<>();
        for (int i = 0; i < firsts.length; i++) {
            if (!slow[i] || (type != null && fastTypes.contains(type[i]))) {
                perf.add(coreOf.get(firsts[i]));
            }
        }
        if (perf.size() == firsts.length) {
            return null;
        }
        return new Classes(perf, "cores of at least " + CLASS_RATIO_PERCENT + "% of the highest cpu_capacity " + max);
    }

    /**
     * The ARM core type (implementer and part number of {@code midr_el1}) of every CPU of {@code cpus}, or null when
     * one does not read (every non-ARM host).
     */
    private static long[] coreTypes(Path cpuRoot, int[] cpus) {
        long[] type = new long[cpus.length];
        for (int i = 0; i < cpus.length; i++) {
            String v = read(cpuRoot.resolve("cpu" + cpus[i]).resolve("regs").resolve("identification")
                    .resolve("midr_el1"));
            if (v == null || !v.startsWith("0x")) {
                return null;
            }
            long midr = parseHex(v.substring(2));
            if (midr < 0) {
                return null;
            }
            type[i] = midr & 0xFF00FFF0L;
        }
        return type;
    }

    // ------------------------------------------------------------------ sysfs

    /** The lowest CPU of {@code cpu}'s SMT sibling list, or {@code cpu} itself when the list does not read. */
    private static int firstSibling(Path cpuRoot, int cpu) {
        Path topo = cpuRoot.resolve("cpu" + cpu).resolve("topology");
        String list = read(topo.resolve("core_cpus_list"));
        if (list == null) {
            list = read(topo.resolve("thread_siblings_list"));
        }
        int[] siblings = parseOrEmpty(list);
        return siblings.length == 0 ? cpu : Math.min(siblings[0], cpu);
    }

    /** The first CPU of {@code cpus} of every core, in increasing order. */
    private static int[] onePerCore(int[] cpus, Map<Integer, Integer> coreOf) {
        Set<Integer> seen = new HashSet<>();
        int[] out = new int[cpus.length];
        int n = 0;
        for (int cpu : cpus) {
            if (seen.add(coreOf.getOrDefault(cpu, cpu))) {
                out[n++] = cpu;
            }
        }
        return Arrays.copyOf(out, n);
    }

    /** The CPUs of {@code cpus} whose core is in {@code cores}. */
    private static int[] filter(int[] cpus, Set<Integer> cores, Map<Integer, Integer> coreOf) {
        int[] out = new int[cpus.length];
        int n = 0;
        for (int cpu : cpus) {
            if (cores.contains(coreOf.getOrDefault(cpu, cpu))) {
                out[n++] = cpu;
            }
        }
        return Arrays.copyOf(out, n);
    }

    /** The CPUs in both lists, in increasing order. */
    private static int[] intersect(int[] a, int[] b) {
        int[] sb = b.clone();
        Arrays.sort(sb);
        int[] out = new int[a.length];
        int n = 0;
        for (int cpu : a) {
            if (Arrays.binarySearch(sb, cpu) >= 0) {
                out[n++] = cpu;
            }
        }
        int[] r = Arrays.copyOf(out, n);
        Arrays.sort(r);
        return r;
    }

    /** The cpulist {@code text}, or empty when it is absent or malformed. */
    private static int[] parseOrEmpty(String text) {
        try {
            return CpuLists.parse(text);
        } catch (IllegalArgumentException e) {
            return new int[0];
        }
    }

    /** {@code text} as a decimal long, or {@code otherwise} when it is absent or malformed. */
    private static long parseLong(String text, long otherwise) {
        if (text == null) {
            return otherwise;
        }
        try {
            return Long.parseLong(text);
        } catch (NumberFormatException e) {
            return otherwise;
        }
    }

    /** {@code text} as an unsigned hexadecimal long of at most 16 digits, or -1 when it is malformed. */
    private static long parseHex(String text) {
        if (text.isEmpty() || text.length() > 16) {
            return -1L;
        }
        try {
            return Long.parseUnsignedLong(text, 16) & Long.MAX_VALUE;
        } catch (NumberFormatException e) {
            return -1L;
        }
    }

    /** A sysfs file's trimmed text, or null when it does not read. */
    private static String read(Path path) {
        try {
            return new String(Files.readAllBytes(path), StandardCharsets.US_ASCII).trim();
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }
}
