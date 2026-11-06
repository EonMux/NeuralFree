package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;

import org.junit.Assume;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The shared Vector API module detection: the option exists on Java 17 and later only, a relaunch is
 * needed exactly on a Java 17+ JVM without the module, has not already relaunched and did not switch
 * the kernels off, the relaunch command carries this JVM's own options, class path and arguments, and a
 * launcher started without the module really runs its command with the module resolved.
 */
public class JvmVectorModuleTest {

    private static final String JAVA = Paths.get(System.getProperty("java.home"), "bin", "java").toString();

    @Test
    public void theOptionExistsOnJava17AndLaterOnly() {
        assertTrue(JvmVectorModule.options(11).isEmpty());
        assertTrue(JvmVectorModule.options(16).isEmpty());
        assertEquals(Collections.singletonList("--add-modules=jdk.incubator.vector"),
                JvmVectorModule.options(17));
        assertEquals(JvmVectorModule.options(17), JvmVectorModule.options(21));
    }

    @Test
    public void theRelaunchIsNeededOnlyOnAFreshJava17JvmWithoutTheModule() {
        assertTrue(JvmVectorModule.needsRelaunch(17, false, false, true));
        assertTrue(JvmVectorModule.needsRelaunch(21, false, false, true));
        assertFalse("Java 11 has no module", JvmVectorModule.needsRelaunch(11, false, false, true));
        assertFalse("the module is resolved", JvmVectorModule.needsRelaunch(17, true, false, true));
        assertFalse("the child of a relaunch", JvmVectorModule.needsRelaunch(17, false, true, true));
        assertFalse("the scalar kernels are requested", JvmVectorModule.needsRelaunch(17, false, false, false));
    }

    @Test
    public void theRelaunchCommandCarriesTheOwnOptionsClassPathAndArguments() {
        List<String> input = Arrays.asList("-Xmx4g", "--add-opens=java.base/java.util=ALL-UNNAMED",
                "-Dneuralfree.cpu.vector.relaunched=true", "-classpath", "/stale/class/path");
        List<String> cmd = JvmVectorModule.relaunchCommand(17, input, "/app/lib/core.jar",
                Launcher.class.getName(), new String[]{"cli", "--model", "m.gguf"});
        assertEquals(JAVA, cmd.get(0));
        assertEquals("-Xmx4g", cmd.get(1));
        assertEquals("--add-opens=java.base/java.util=ALL-UNNAMED", cmd.get(2));
        assertEquals("--add-modules=jdk.incubator.vector", cmd.get(3));
        assertEquals("-Dneuralfree.cpu.vector.relaunched=true", cmd.get(4));
        assertEquals("-cp", cmd.get(5));
        assertEquals("/app/lib/core.jar", cmd.get(6));
        assertEquals(Launcher.class.getName(), cmd.get(7));
        assertEquals(Arrays.asList("cli", "--model", "m.gguf"), cmd.subList(8, cmd.size()));
        // a marker or class path option that came in does not ride along twice
        assertEquals(1, Collections.frequency(cmd, "-Dneuralfree.cpu.vector.relaunched=true"));
        assertEquals(0, Collections.frequency(cmd, "-classpath"));
        assertFalse("the stale class path value must not become an argument", cmd.contains("/stale/class/path"));
    }

    @Test
    public void theRelaunchNeverTurnsAnEnvironmentKeyIntoACommandLineOne() {
        // a key supplied through NEURALFREE_WATERMARK_KEY is inherited by the child process; the relaunch must
        // not synthesize a -D property or an argument that would expose it in /proc/<pid>/cmdline
        String key = "3141592653589793238";
        List<String> cmd = JvmVectorModule.relaunchCommand(17, Collections.singletonList("-Xmx4g"),
                "/app/lib/core.jar", Launcher.class.getName(),
                new String[]{"chat", "--model", "m.gguf"});
        for (String argument : cmd) {
            assertFalse(argument, argument.contains(key));
            assertFalse(argument, argument.contains("NEURALFREE_WATERMARK_KEY"));
        }
        assertEquals(Arrays.asList("chat", "--model", "m.gguf"),
                cmd.subList(cmd.size() - 3, cmd.size()));
    }

    @Test
    public void aJava11RelaunchCommandCarriesNoOption() {
        List<String> cmd = JvmVectorModule.relaunchCommand(11, Collections.emptyList(), "lib/core.jar",
                Launcher.class.getName(), new String[]{"version"});
        assertFalse(cmd.contains("--add-modules=jdk.incubator.vector"));
    }

    @Test
    public void theSimdPropertySwitchesTheRelaunchOff() {
        Properties p = new Properties();
        assertTrue(JvmVectorModule.simdEnabled(p));
        p.setProperty(CpuSimd.PROP_SIMD, "false");
        assertFalse(JvmVectorModule.simdEnabled(p));
        p.setProperty(CpuSimd.PROP_SIMD, "nonsense");
        assertTrue("an invalid value is left to the engine to reject by name",
                JvmVectorModule.simdEnabled(p));
    }

    @Test
    public void aDebugAgentStopsTheRelaunch() {
        assertTrue(JvmVectorModule.underDebugAgent(Arrays.asList(
                "-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005")));
        assertTrue(JvmVectorModule.underDebugAgent(Arrays.asList("-Xrunjdwp:transport=dt_socket")));
        assertFalse(JvmVectorModule.underDebugAgent(Arrays.asList("-Xmx4g",
                "--add-opens=java.base/java.util=ALL-UNNAMED", "-Dneuralfree.cpu.simd=true")));
    }

    @Test
    public void aJvmThatAlreadyResolvedTheModuleDoesNotRelaunch() {
        Assume.assumeTrue(JvmVectorModule.resolved());
        assertNull(JvmVectorModule.relaunchIfNeeded(true, Launcher.class.getName(),
                new String[]{"version"}));
    }

    @Test
    public void aLauncherStartedWithoutTheModuleRelaunchesAndRunsWithIt() throws Exception {
        Assume.assumeTrue(Runtime.version().feature() >= JvmVectorModule.FIRST_FEATURE);
        String out = spawnWithoutTheModule(Probe.class.getName(), "relaunch");
        assertTrue(out, out.contains("parent-resolved=false"));
        assertTrue(out, out.contains("child-resolved=true"));
        assertTrue(out, out.contains("child-simd=true"));
        assertTrue(out, out.contains("exit=0"));
    }

    /** A JVM started without the module: the first run relaunches itself, the child reports its state. */
    public static final class Probe {

        public static void main(String[] args) {
            if (System.getProperty(JvmVectorModule.RELAUNCHED_PROPERTY) == null) {
                System.out.println("parent-resolved=" + JvmVectorModule.resolved());
                Integer code = JvmVectorModule.relaunchIfNeeded(true, Probe.class.getName(),
                        new String[]{"child"});
                System.out.println("exit=" + code);
                System.exit(code == null ? 1 : code);
            }
            System.out.println("child-resolved=" + JvmVectorModule.resolved());
            System.out.println("child-simd=" + CpuSimd.available());
            System.exit(0);
        }
    }

    /** Runs {@code mainClass} in a fresh JVM without the module, with the host-level option variables cleared. */
    private static String spawnWithoutTheModule(String mainClass, String argument) throws Exception {
        List<String> cmd = new ArrayList<>();
        cmd.add(JAVA);
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(mainClass);
        cmd.add(argument);
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        pb.environment().remove("JAVA_TOOL_OPTIONS");
        pb.environment().remove("_JAVA_OPTIONS");
        pb.environment().remove("JDK_JAVA_OPTIONS");
        Process p = pb.start();
        StringBuilder out = new StringBuilder();
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                out.append(line).append('\n');
            }
        }
        assertTrue("probe timed out: " + out, p.waitFor(60, TimeUnit.SECONDS));
        assertEquals("probe exit: " + out, 0, p.exitValue());
        return out.toString();
    }
}
