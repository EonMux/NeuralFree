package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Test;

import java.io.File;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.ref.WeakReference;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** The CPU heap budget, including model loading after an allocation burst in a bounded, separate JVM. */
public class HeapMemoryTest {

    private static final long MIB = 1024L * 1024L;

    @Test
    public void anUnboundedHeapHasAnUnlimitedBudget() {
        assertEquals(ContextFitter.UNLIMITED, HeapMemory.budgetBytes(Long.MAX_VALUE, 100 * MIB, 40 * MIB));
    }

    @Test
    public void theBudgetIsTheFreeHeapLessTheHeadroom() {
        assertEquals("free 508 MiB: the 64 MiB floor is the headroom", 444 * MIB,
                HeapMemory.budgetBytes(512 * MIB, 6 * MIB, 2 * MIB));
        assertEquals("free 600 MiB: an eighth of the free heap is the headroom", 525 * MIB,
                HeapMemory.budgetBytes(1024 * MIB, 424 * MIB, 0));
    }

    @Test
    public void aFreeHeapWithinTheHeadroomReadsZero() {
        assertEquals(0, HeapMemory.budgetBytes(512 * MIB, 512 * MIB, 0));
        assertEquals(0, HeapMemory.budgetBytes(512 * MIB, 512 * MIB, 40 * MIB));
        assertEquals(0, HeapMemory.budgetBytes(512 * MIB, 512 * MIB, 64 * MIB));
        assertEquals(1, HeapMemory.budgetBytes(512 * MIB, 512 * MIB, 64 * MIB + 1));
    }

    @Test
    public void aHealthyHeapNeedsNoCollection() throws Exception {
        runProbe("healthy");
    }

    @Test
    public void unreachableAllocationsDoNotPreventLoadingAContext() throws Exception {
        runProbe("garbage");
    }

    @Test
    public void liveAllocationsStillPreventLoadingAContext() throws Exception {
        runProbe("live");
    }

    private static void runProbe(String scenario) throws Exception {
        String java = new File(new File(System.getProperty("java.home"), "bin"), "java").toString();
        Process process = new ProcessBuilder(java, "-Xms128m", "-Xmx128m", "-XX:+UseSerialGC",
                "-cp", runtimeClasspath(), PressureProbe.class.getName(), scenario).redirectErrorStream(true).start();
        try {
            assertTrue("heap probe timed out: " + scenario, process.waitFor(60, TimeUnit.SECONDS));
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertEquals(output, 0, process.exitValue());
            assertTrue(output, output.contains("passed " + scenario));
        } finally {
            process.destroyForcibly();
        }
    }

    /** Includes Gradle's isolated test class loader as well as the launcher class path. */
    private static String runtimeClasspath() throws Exception {
        Set<String> paths = new LinkedHashSet<>();
        paths.add(System.getProperty("java.class.path"));
        for (ClassLoader loader = HeapMemoryTest.class.getClassLoader(); loader != null; loader = loader.getParent()) {
            if (loader instanceof URLClassLoader) {
                for (URL url : ((URLClassLoader) loader).getURLs()) {
                    paths.add(new File(url.toURI()).toString());
                }
            }
        }
        return String.join(File.pathSeparator, paths);
    }

    /** A small Serial-GC heap keeps the pressure scenarios independent of the test worker's heap and collector. */
    public static final class PressureProbe {
        private static volatile byte[] retained;

        public static void main(String[] args) throws Exception {
            Path path = SyntheticRoutedModels.deepseek4().writeTempFile();
            try {
                EngineConfig config = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(16)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
                // Initialize model-loading classes before measuring the allocation burst.
                try (GGUFModel model = new GGUFModel(path.toString(), config)) {
                    check(model.memoryPlan().nCtx() == 64, "the warmup context must have 64 tokens");
                }
                System.gc();
                if ("healthy".equals(args[0])) {
                    healthy();
                } else {
                    pressure(path, config, "live".equals(args[0]));
                }
                System.out.println("passed " + args[0]);
            } finally {
                retained = null;
                Files.deleteIfExists(path);
            }
        }

        private static void healthy() {
            long collections = collectionCount();
            check(HeapMemory.budgetBytes() > 0, "a healthy heap must have a budget");
            check(collectionCount() == collections, "a healthy reading must not request collection");
        }

        private static void pressure(Path path, EngineConfig config, boolean live) throws Exception {
            retained = new byte[64 * (int) MIB];
            WeakReference<byte[]> allocation = new WeakReference<>(retained);
            check(rawBudget() == 0, "the allocation must exhaust the pre-collection budget");
            if (!live) {
                retained = null;
            }
            if (live) {
                try (GGUFModel ignored = new GGUFModel(path.toString(), config)) {
                    throw new AssertionError("a live allocation must not be ignored by the context fit");
                } catch (ContextFitter.ContextBudgetException expected) {
                    check(expected.budgetBytes() == 0, "a live allocation must leave a zero budget");
                }
                check(allocation.get() != null, "the live allocation must stay reachable");
            } else {
                try (GGUFModel model = new GGUFModel(path.toString(), config)) {
                    check(model.memoryPlan().nCtx() == 64, "the requested context must load after reclamation");
                }
                check(allocation.get() == null, "the unreachable allocation must have been collected");
            }
        }

        private static long rawBudget() {
            Runtime runtime = Runtime.getRuntime();
            return HeapMemory.budgetBytes(runtime.maxMemory(), runtime.totalMemory(), runtime.freeMemory());
        }

        private static long collectionCount() {
            long count = 0;
            for (GarbageCollectorMXBean collector : ManagementFactory.getGarbageCollectorMXBeans()) {
                count += collector.getCollectionCount();
            }
            return count;
        }

        private static void check(boolean condition, String message) {
            if (!condition) {
                throw new AssertionError(message);
            }
        }
    }
}
