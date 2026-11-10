package com.eonmux.neuralfree.gguf.cpu.numa;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link CpuThreadPolicy} over fake sysfs trees: SMT siblings collapse to one CPU per core, an Intel hybrid part
 * keeps its performance cores, capacity tiers split core classes only for a large ratio and distinct ARM core types,
 * the process affinity bounds every policy and drops the efficiency cores only while it holds every performance core,
 * the processor limit caps every policy, and a host without the tree falls back to the processor count.
 */
public class CpuThreadPolicyTest {

    private Path root;
    private Path cpuRoot;
    private Path hybrid;
    private String saved;

    @Before
    public void setUp() throws IOException {
        root = Files.createTempDirectory("nf-sysfs");
        cpuRoot = root.resolve("cpu");
        hybrid = root.resolve("cpu_core").resolve("cpus");
        saved = System.getProperty(CpuThreadPolicy.PROPERTY);
    }

    @After
    public void tearDown() throws IOException {
        if (saved == null) {
            System.clearProperty(CpuThreadPolicy.PROPERTY);
        } else {
            System.setProperty(CpuThreadPolicy.PROPERTY, saved);
        }
        try (Stream<Path> walk = Files.walk(root)) {
            walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    private void write(Path p, String text) throws IOException {
        Files.createDirectories(p.getParent());
        Files.write(p, text.getBytes(StandardCharsets.US_ASCII));
    }

    private void cpu(int cpu, String siblings, Integer capacity) throws IOException {
        Path dir = cpuRoot.resolve("cpu" + cpu);
        write(dir.resolve("topology").resolve("core_cpus_list"), siblings + "\n");
        if (capacity != null) {
            write(dir.resolve("cpu_capacity"), capacity + "\n");
        }
    }

    private void midr(int cpu, String value) throws IOException {
        write(cpuRoot.resolve("cpu" + cpu).resolve("regs").resolve("identification").resolve("midr_el1"),
                value + "\n");
    }

    /**
     * A Core Ultra 9 275HX: 24 single-thread cores, P-cores 0-7 at ITMT capacities 991 and 1024, E-cores 8-23 at
     * 747, the hybrid PMU list naming 0-7.
     */
    private void hostLikeCoreUltra9() throws IOException {
        write(cpuRoot.resolve("online"), "0-23\n");
        for (int c = 0; c < 24; c++) {
            cpu(c, String.valueOf(c), c >= 8 ? 747 : c % 2 == 1 ? 1024 : 991);
        }
        write(hybrid, "0-7\n");
    }

    private CpuThreadPolicy read(int... affinity) {
        return CpuThreadPolicy.read(cpuRoot, hybrid, affinity, 99);
    }

    @Test
    public void smtSiblingsCollapseToOneCpuPerCore() throws IOException {
        write(cpuRoot.resolve("online"), "0-7\n");
        for (int c = 0; c < 4; c++) {
            String pair = c + "," + (c + 4);
            cpu(c, pair, null);
            cpu(c + 4, pair, null);
        }
        CpuThreadPolicy policy = read();
        assertEquals(8, policy.threads(CpuThreadPolicy.Policy.LOGICAL));
        assertArrayEquals(new int[] {0, 1, 2, 3}, policy.cpus(CpuThreadPolicy.Policy.PHYSICAL));
        assertEquals("one core class: every physical core is a performance core", 4,
                policy.threads(CpuThreadPolicy.Policy.PERFORMANCE));
    }

    @Test
    public void theOlderSiblingListServesWhenCoreCpusListIsAbsent() throws IOException {
        write(cpuRoot.resolve("online"), "0-3\n");
        for (int c = 0; c < 4; c++) {
            write(cpuRoot.resolve("cpu" + c).resolve("topology").resolve("thread_siblings_list"),
                    (c < 2 ? "0-1" : "2-3") + "\n");
        }
        assertArrayEquals(new int[] {0, 2}, read().cpus(CpuThreadPolicy.Policy.PHYSICAL));
    }

    @Test
    public void anIntelHybridPartKeepsItsPerformanceCores() throws IOException {
        write(cpuRoot.resolve("online"), "0-11\n");
        for (int c = 0; c < 4; c++) {
            String pair = (2 * c) + "-" + (2 * c + 1);
            cpu(2 * c, pair, null);
            cpu(2 * c + 1, pair, null);
        }
        for (int c = 8; c < 12; c++) {
            cpu(c, String.valueOf(c), null);
        }
        write(hybrid, "0-7\n");
        CpuThreadPolicy policy = read();
        assertEquals(12, policy.threads(CpuThreadPolicy.Policy.LOGICAL));
        assertArrayEquals(new int[] {0, 2, 4, 6, 8, 9, 10, 11}, policy.cpus(CpuThreadPolicy.Policy.PHYSICAL));
        assertArrayEquals(new int[] {0, 2, 4, 6}, policy.cpus(CpuThreadPolicy.Policy.PERFORMANCE));
        assertTrue(policy.describe(), policy.describe().contains("hybrid performance cores 0-7"));
    }

    @Test
    public void armCapacitiesRankBigOverLittleCores() throws IOException {
        write(cpuRoot.resolve("online"), "0-7\n");
        int[] caps = {1024, 1024, 870, 870, 446, 446, 446, 446};
        for (int c = 0; c < 8; c++) {
            cpu(c, String.valueOf(c), caps[c]);
        }
        CpuThreadPolicy policy = read();
        assertEquals(8, policy.threads(CpuThreadPolicy.Policy.PHYSICAL));
        assertArrayEquals("without midr_el1, every core at 75% or more of the highest capacity", new int[] {0, 1, 2, 3},
                policy.cpus(CpuThreadPolicy.Policy.PERFORMANCE));
    }

    @Test
    public void theProcessAffinityBoundsEveryPolicy() throws IOException {
        write(cpuRoot.resolve("online"), "0-11\n");
        for (int c = 0; c < 12; c++) {
            cpu(c, String.valueOf(c), null);
        }
        write(hybrid, "0-3\n");
        CpuThreadPolicy policy = read(0, 1, 2, 3, 4, 5, 6);
        assertArrayEquals(new int[] {0, 1, 2, 3, 4, 5, 6}, policy.cpus(CpuThreadPolicy.Policy.LOGICAL));
        assertArrayEquals("every performance core allowed: the efficiency cores drop", new int[] {0, 1, 2, 3},
                policy.cpus(CpuThreadPolicy.Policy.PERFORMANCE));

        CpuThreadPolicy onlyEfficiency = read(8, 9);
        assertArrayEquals("no allowed performance core: the allowed cores serve", new int[] {8, 9},
                onlyEfficiency.cpus(CpuThreadPolicy.Policy.PERFORMANCE));
    }

    @Test
    public void anAffinityMissingSomePerformanceCoresCountsEveryAllowedPhysicalCore() throws IOException {
        hostLikeCoreUltra9();
        int[] mask = new int[17];
        for (int i = 0; i < mask.length; i++) {
            mask[i] = 7 + i;
        }
        CpuThreadPolicy policy = read(mask);
        assertEquals("taskset -c 7-23: one P-core and 16 E-cores all count", 17,
                policy.threads(CpuThreadPolicy.Policy.PERFORMANCE));
        assertTrue(policy.describe(), policy.describe().contains("holds 1 of the 8 performance cores"));

        CpuThreadPolicy smtHalf = read(0, 2, 4, 6, 8, 9);
        assertEquals("four of eight P-cores plus two E-cores", 6, smtHalf.threads(CpuThreadPolicy.Policy.PERFORMANCE));
    }

    @Test
    public void aHybridHostWithItmtCapacitiesKeepsItsEightPerformanceCores() throws IOException {
        hostLikeCoreUltra9();
        CpuThreadPolicy policy = read();
        assertEquals(24, policy.threads(CpuThreadPolicy.Policy.PHYSICAL));
        assertArrayEquals(new int[] {0, 1, 2, 3, 4, 5, 6, 7}, policy.cpus(CpuThreadPolicy.Policy.PERFORMANCE));
    }

    @Test
    public void capacitiesWithoutTheHybridListSeparateEfficiencyCores() throws IOException {
        hostLikeCoreUltra9();
        Files.delete(hybrid);
        CpuThreadPolicy policy = read();
        assertArrayEquals("747 is below 3/4 of 1024; 991 is not", new int[] {0, 1, 2, 3, 4, 5, 6, 7},
                policy.cpus(CpuThreadPolicy.Policy.PERFORMANCE));
        assertTrue(policy.describe(), policy.describe().contains("cpu_capacity 1024"));
    }

    @Test
    public void preferredCoreCapacitiesAreOneCoreClass() throws IOException {
        write(cpuRoot.resolve("online"), "0-7\n");
        for (int c = 0; c < 8; c++) {
            cpu(c, String.valueOf(c), c % 4 == 1 ? 1024 : 991);
        }
        CpuThreadPolicy policy = read();
        assertEquals("ITMT preferred cores are not a core class", 8,
                policy.threads(CpuThreadPolicy.Policy.PERFORMANCE));
        assertTrue(policy.describe(), policy.describe().contains("one core class"));
    }

    @Test
    public void armCoresOfOneTypeNeverSplitByCapacity() throws IOException {
        write(cpuRoot.resolve("online"), "0-7\n");
        for (int c = 0; c < 8; c++) {
            cpu(c, String.valueOf(c), c < 4 ? 1024 : 600);
            midr(c, "0x00000000410fd050");
        }
        assertEquals("one Cortex-A55 design clocked in two clusters", 8,
                read().threads(CpuThreadPolicy.Policy.PERFORMANCE));

        for (int c = 4; c < 8; c++) {
            midr(c, "0x00000000411fd030");
        }
        assertArrayEquals("a distinct LITTLE core type below 3/4 of the capacity drops", new int[] {0, 1, 2, 3},
                read().cpus(CpuThreadPolicy.Policy.PERFORMANCE));
    }

    @Test
    public void theProcessorLimitCapsEveryPolicy() throws IOException {
        write(cpuRoot.resolve("online"), "0-15\n");
        for (int c = 0; c < 16; c++) {
            cpu(c, String.valueOf(c), null);
        }
        CpuThreadPolicy policy = CpuThreadPolicy.read(cpuRoot, hybrid, new int[0], 2);
        for (CpuThreadPolicy.Policy p : CpuThreadPolicy.Policy.values()) {
            assertEquals("a 2-CPU cgroup quota or -XX:ActiveProcessorCount=2: " + p, 2, policy.threads(p));
        }
        assertEquals(16, policy.cpus(CpuThreadPolicy.Policy.LOGICAL).length);
        assertTrue(policy.describe(), policy.describe().contains("at most 2 available processor(s)"));
    }

    @Test
    public void aHostWithoutTheTreeFallsBackToTheProcessorCount() {
        CpuThreadPolicy policy = read();
        for (CpuThreadPolicy.Policy p : CpuThreadPolicy.Policy.values()) {
            assertEquals(p.name(), 99, policy.threads(p));
        }
        assertTrue(policy.describe(), policy.describe().contains("no sysfs CPU topology"));
    }

    @Test
    public void thePropertySelectsThePolicyAndRejectsUnknownNames() throws IOException {
        write(cpuRoot.resolve("online"), "0-3\n");
        for (int c = 0; c < 4; c++) {
            cpu(c, c < 2 ? "0-1" : "2-3", null);
        }
        CpuThreadPolicy policy = read();
        System.setProperty(CpuThreadPolicy.PROPERTY, " Logical ");
        assertEquals(4, policy.defaultThreads());
        System.setProperty(CpuThreadPolicy.PROPERTY, "physical");
        assertEquals(2, policy.defaultThreads());
        System.clearProperty(CpuThreadPolicy.PROPERTY);
        assertEquals(CpuThreadPolicy.DEFAULT_POLICY, CpuThreadPolicy.configuredPolicy());
        System.setProperty(CpuThreadPolicy.PROPERTY, "fastest");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, policy::defaultThreads);
        assertTrue(e.getMessage(), e.getMessage().contains("logical, physical or performance"));
    }

    @Test
    public void theHostReadingIsPlausible() {
        CpuThreadPolicy host = CpuThreadPolicy.host();
        int logical = host.threads(CpuThreadPolicy.Policy.LOGICAL);
        int physical = host.threads(CpuThreadPolicy.Policy.PHYSICAL);
        int perf = host.threads(CpuThreadPolicy.Policy.PERFORMANCE);
        assertTrue(host.describe(), perf >= 1 && perf <= physical && physical <= logical);
        int available = Runtime.getRuntime().availableProcessors();
        for (CpuThreadPolicy.Policy p : CpuThreadPolicy.Policy.values()) {
            int n = host.threads(p);
            assertTrue(p + ": " + n + " of " + available + " available, " + host.describe(), n >= 1 && n <= available);
        }
    }

    @Test
    public void aCpulistFormatsWithRanges() {
        assertEquals("0-3,8,10-11", CpuLists.format(new int[] {0, 1, 2, 3, 8, 10, 11}));
        assertEquals("none", CpuLists.format(new int[0]));
        assertEquals("5", CpuLists.format(new int[] {5}));
    }
}
