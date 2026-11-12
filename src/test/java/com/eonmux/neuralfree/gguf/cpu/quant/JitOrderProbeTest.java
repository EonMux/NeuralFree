package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Assume;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The weight kernels compile to the same code whatever Vector API code the JVM ran first ({@code SimdI8}'s rules):
 * {@link JitOrderProbe} times each kernel group in fresh JVMs, cold and after every other vector kernel and the
 * vector attention have run. A kernel that loses its intrinsics in the polluted order runs the Vector API's Java
 * fallback, which allocates a vector object per operation and runs at a half to a fifth of its cold rate for the
 * life of the JVM. Two checks guard it:
 * <ul>
 *   <li>no product of any JVM allocates more than {@link #MAX_PRODUCT_BYTES} on its thread: a compiled kernel
 *       allocates nothing per product and a fallback megabytes, so this check does not depend on the machine's
 *       load;</li>
 *   <li>no product falls below {@link #MIN_RATIO} of its group's typical polluted-to-cold rate ratio (the median
 *       over the group's products). The JVMs of both orders run at the same time and each order keeps the faster
 *       of its JVMs, but a cold JVM can still land on a performance core while the polluted ones share efficiency
 *       cores, which moves every product of the group alike; a lost intrinsic slows its own products against the
 *       others. A group below the bound runs again, up to {@link #ATTEMPTS} times, since a lost intrinsic slows
 *       every attempt.</li>
 * </ul>
 * A third check is exact: in a polluted JVM of each group, no C2 compilation of a Vector API kernel
 * ({@code com.eonmux.neuralfree.gguf.cpu.quant.Simd*}) reaches the inlining node cutoff. C2 parses each Vector API
 * call into a few hundred nodes before it intrinsifies it and stops inlining once a compilation passes its node
 * cutoff, so a kernel near the cutoff loses the intrinsics past it in some JVMs and not others, depending on what
 * the profile let it inline first; one that stays clear of the cutoff compiles alike in every JVM. The check reads
 * the JVM's compilation log ({@code -XX:+LogCompilation}).
 */
public class JitOrderProbeTest {

    /**
     * The least polluted-to-cold rate ratio, relative to the group's median ratio, every timed product must keep in
     * one of the attempts.
     */
    static final double MIN_RATIO = 0.7;

    /** The most heap one timed product may allocate on its thread. */
    static final long MAX_PRODUCT_BYTES = 64 * 1024;

    /** The runs of a group before its rate ratio counts as lost. */
    static final int ATTEMPTS = 3;

    private static final int JVMS_PER_ORDER = 2;

    private static final String[] PRODUCTS = {".tiles", ".decode"};

    /** The longest one probe JVM may run before the test destroys it and fails. */
    static final long PROBE_TIMEOUT_SECONDS = 120;

    /** The longest the test waits for a destroyed or exited child JVM and the end of its output. */
    static final long DESTROY_SECONDS = 10;

    /** The class-name prefix of the Vector API kernels, as the compilation log names a method's holder. */
    static final String KERNEL_PREFIX = "com.eonmux.neuralfree.gguf.cpu.quant.Simd";

    @Test
    public void everyKernelGroupKeepsItsColdRateAfterTheOtherVectorKernelsRan() throws Exception {
        Assume.assumeTrue(CpuSimd.available());
        final StringBuilder report = new StringBuilder();
        for (Map.Entry<String, GGMLType[]> group : JitOrderProbe.GROUPS.entrySet()) {
            double worst = 0;
            for (int attempt = 1; attempt <= ATTEMPTS && worst < MIN_RATIO; attempt++) {
                final Map<String, Map<String, Long>> best = run(group.getKey(), group.getValue());
                final Map<String, Double> ratios = new LinkedHashMap<>();
                for (GGMLType t : group.getValue()) {
                    for (String product : PRODUCTS) {
                        final String key = t + product;
                        ratios.put(key, (double) best.get("cold").get(key) / best.get("polluted").get(key));
                    }
                }
                final double[] sorted = ratios.values().stream().mapToDouble(Double::doubleValue).sorted().toArray();
                final double median = (sorted[(sorted.length - 1) / 2] + sorted[sorted.length / 2]) / 2;
                worst = Double.MAX_VALUE;
                for (Map.Entry<String, Double> r : ratios.entrySet()) {
                    worst = Math.min(worst, r.getValue() / median);
                    report.append(String.format("%s attempt %d: %s cold %d ns, polluted %d ns, ratio %.2f, %.2f of"
                                    + " the median %.2f%n", group.getKey(), attempt, r.getKey(),
                            best.get("cold").get(r.getKey()), best.get("polluted").get(r.getKey()), r.getValue(),
                            r.getValue() / median, median));
                }
            }
            assertTrue("the " + group.getKey() + " kernels lost their cold rate after the other vector kernels ran"
                    + " in every attempt:\n" + report, worst >= MIN_RATIO);
        }
    }

    /**
     * The best time of every product of {@code group} per order, over the group's JVMs started together; fails when
     * a product of any JVM allocated more than {@link #MAX_PRODUCT_BYTES}.
     */
    private static Map<String, Map<String, Long>> run(String group, GGMLType[] types) throws Exception {
        final List<Process> procs = new ArrayList<>();
        final List<String> orders = new ArrayList<>();
        for (int i = 0; i < JVMS_PER_ORDER; i++) {
            for (String order : new String[]{"cold", "polluted"}) {
                procs.add(start(order, group));
                orders.add(order);
            }
        }
        final Map<String, Map<String, Long>> best = new HashMap<>();
        try {
            for (int i = 0; i < procs.size(); i++) {
                final Map<String, Long> out = read(procs.get(i), types.length);
                for (GGMLType t : types) {
                    for (String product : PRODUCTS) {
                        final String key = t + product;
                        final long bytes = out.get(key + ".bytes");
                        assertTrue(String.format("%s in a %s JVM allocated %d bytes per product, more than %d: its"
                                        + " Vector API calls run the library's Java fallback", key, orders.get(i),
                                bytes, MAX_PRODUCT_BYTES), bytes <= MAX_PRODUCT_BYTES);
                        best.computeIfAbsent(orders.get(i), k -> new HashMap<>()).merge(key, out.get(key),
                                Math::min);
                    }
                }
            }
        } finally {
            for (Process p : procs) {
                p.destroyForcibly();
            }
        }
        return best;
    }

    @Test
    public void noKernelCompilationReachesTheInliningNodeCutoff() throws Exception {
        Assume.assumeTrue(CpuSimd.available());
        final List<String> reached = new ArrayList<>();
        for (Map.Entry<String, GGMLType[]> group : JitOrderProbe.GROUPS.entrySet()) {
            final Path log = Files.createTempFile("jit-order-" + group.getKey(), ".log");
            Process p = null;
            try {
                p = start("polluted", group.getKey(), "-XX:+UnlockDiagnosticVMOptions", "-XX:+LogCompilation",
                        "-XX:LogFile=" + log);
                read(p, group.getValue().length);
                for (String method : cutoffMethods(log)) {
                    reached.add(group.getKey() + ": " + method);
                }
            } finally {
                if (p != null) {
                    p.destroyForcibly();
                }
                Files.deleteIfExists(log);
            }
        }
        assertTrue("these kernel compilations reached C2's inlining node cutoff, so the Vector API calls past it"
                + " can run the library's allocating Java fallback: " + reached, reached.isEmpty());
    }

    /**
     * The root methods of the compilations in {@code log} that stopped inlining at the node cutoff and belong to a
     * Vector API kernel class ({@value #KERNEL_PREFIX}).
     */
    static Set<String> cutoffMethods(Path log) throws IOException {
        final Set<String> methods = new TreeSet<>();
        String task = null;
        try (BufferedReader r = Files.newBufferedReader(log, StandardCharsets.ISO_8859_1)) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.startsWith("<task ")) {
                    final int m = line.indexOf(" method='");
                    final int end = m < 0 ? -1 : line.indexOf(" (", m);
                    task = end < 0 ? null : line.substring(m + 9, end);
                } else if (line.startsWith("</task>")) {
                    task = null;
                } else if (task != null && task.startsWith(KERNEL_PREFIX)
                        && line.contains("reason='NodeCountInliningCutoff'")) {
                    methods.add(task);
                }
            }
        }
        return methods;
    }

    /**
     * A child JVM that hangs fails the read at its deadline and is destroyed, rather than holding the test task until
     * it exits on its own.
     */
    @Test(timeout = 60_000)
    public void aHangingChildJvmFailsAtTheDeadlineAndIsDestroyed() throws Exception {
        final Process p = startJava(new String[0], Hang.class.getName());
        try {
            final long t0 = System.nanoTime();
            final AssertionError e = assertThrows(AssertionError.class, () -> output(p, 3, TimeUnit.SECONDS));
            final double seconds = (System.nanoTime() - t0) / 1e9;
            assertTrue(e.getMessage(), e.getMessage().contains("did not exit within 3 SECONDS"));
            assertTrue("the output so far is reported: " + e.getMessage(), e.getMessage().contains("hanging"));
            assertTrue("the read returns at the deadline, after " + seconds + " s", seconds < 3 + 2 * DESTROY_SECONDS);
            assertFalse("the child is destroyed", p.isAlive());
        } finally {
            p.destroyForcibly();
        }
    }

    private static Process start(String order, String group, String... jvmArgs) throws IOException {
        return startJava(jvmArgs, JitOrderProbe.class.getName(), order, group);
    }

    /**
     * A child JVM of this test's runtime and class path with the vector module and {@code jvmArgs}, running
     * {@code mainClass} with {@code args}; its standard error joins its output.
     */
    private static Process startJava(String[] jvmArgs, String mainClass, String... args) throws IOException {
        final List<String> cmd = new ArrayList<>();
        cmd.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        cmd.add("--add-modules=jdk.incubator.vector");
        cmd.add("-Xmx1g");
        cmd.addAll(Arrays.asList(jvmArgs));
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(mainClass);
        cmd.addAll(Arrays.asList(args));
        final ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        pb.environment().remove("JAVA_TOOL_OPTIONS");
        pb.environment().remove("_JAVA_OPTIONS");
        pb.environment().remove("JDK_JAVA_OPTIONS");
        return pb.start();
    }

    /** The probe's {@code <type>.<product>[.bytes]=<value>} lines of one JVM. */
    private static Map<String, Long> read(Process p, int types) throws Exception {
        final String out = output(p, PROBE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        final Map<String, Long> values = new HashMap<>();
        for (String line : out.split("\n")) {
            final int eq = line.indexOf('=');
            if (eq > 0 && (line.contains(".tiles") || line.contains(".decode"))) {
                values.put(line.substring(0, eq), Long.parseLong(line.substring(eq + 1).trim()));
            }
        }
        assertEquals("probe exit: " + out, 0, p.exitValue());
        assertEquals("probe output: " + out, 2 * PRODUCTS.length * types, values.size());
        return values;
    }

    /**
     * The whole output of child JVM {@code p} once it exits within {@code timeout}. A separate thread drains the
     * output while this one waits on the deadline, so a child that hangs, with or without writing, fails the test at
     * the deadline: it is destroyed forcibly and the output so far is reported.
     *
     * @throws AssertionError when the child outlives the deadline, or its output cannot be read
     */
    static String output(Process p, long timeout, TimeUnit unit) throws InterruptedException {
        final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        final AtomicReference<IOException> failure = new AtomicReference<>();
        final Thread drain = new Thread(() -> {
            try (InputStream in = p.getInputStream()) {
                final byte[] buf = new byte[8192];
                for (int n; (n = in.read(buf)) >= 0; ) {
                    bytes.write(buf, 0, n);
                }
            } catch (IOException e) {
                failure.set(e);
            }
        }, "jit-order-probe-output");
        drain.setDaemon(true);
        drain.start();
        if (!p.waitFor(timeout, unit)) {
            p.destroyForcibly();
            final boolean gone = p.waitFor(DESTROY_SECONDS, TimeUnit.SECONDS);
            drain.join(TimeUnit.SECONDS.toMillis(DESTROY_SECONDS));
            throw new AssertionError("the child JVM did not exit within " + timeout + " " + unit
                    + (gone ? " and was destroyed" : " and outlived its forcible destruction") + "; its output: "
                    + new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        }
        drain.join(TimeUnit.SECONDS.toMillis(DESTROY_SECONDS));
        if (drain.isAlive()) {
            throw new AssertionError("the output of the exited child JVM did not close within " + DESTROY_SECONDS
                    + " s; read so far: " + new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        }
        if (failure.get() != null) {
            throw new AssertionError("reading the child JVM's output failed", failure.get());
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    /** A child JVM that never exits, for the deadline test. */
    public static final class Hang {

        private Hang() {
        }

        public static void main(String[] args) throws InterruptedException {
            System.out.println("hanging");
            System.out.flush();
            Thread.sleep(Long.MAX_VALUE);
        }
    }
}
