package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

/**
 * The {@code jdk.incubator.vector} module the CPU SIMD kernels ({@link CpuSimd}) run on: the JVM option
 * that resolves it, and the relaunch a launcher performs when its JVM was started without it. The module
 * exists on Java 17 and later only and is resolved at start-up, so a launcher started on Java 17 or later
 * without the option re-executes itself — the selected {@code java.home}, the JVM options that started it
 * (heap, {@code --add-opens}, properties and agents) and its arguments — with the option added. Java 11
 * and earlier never receive the option, where it would be an error, and the scalar kernels run. The
 * relaunch marks the child with {@value #RELAUNCHED_PROPERTY}, so a runtime image without the module ends
 * the loop instead of relaunching forever. One class so every packaged launcher and every in-process
 * command shares the detection.
 */
public final class JvmVectorModule {

    /** The incubator module of the CPU SIMD kernels. */
    public static final String MODULE = "jdk.incubator.vector";

    /** The JVM option that resolves {@link #MODULE}. */
    public static final String OPTION = "--add-modules=" + MODULE;

    /** The first Java feature release that offers the module. */
    public static final int FIRST_FEATURE = 17;

    /** Marks the JVM a relaunch started; its child does not relaunch again. */
    static final String RELAUNCHED_PROPERTY = "neuralfree.cpu.vector.relaunched";

    private JvmVectorModule() {
    }

    /**
     * The JVM options that resolve the module on {@code feature}: the one option from Java 17, none
     * before where the module does not exist and the JVM would reject it.
     */
    public static List<String> options(int feature) {
        return feature >= FIRST_FEATURE ? Collections.singletonList(OPTION) : Collections.emptyList();
    }

    /** Whether this JVM's boot layer resolved the module. */
    public static boolean resolved() {
        return ModuleLayer.boot().findModule(MODULE).isPresent();
    }

    /**
     * Whether a JVM must relaunch itself to run the SIMD kernels: it is on Java 17 or later, the module
     * is not resolved, the kernels are enabled, and this is not already the child of a relaunch.
     */
    static boolean needsRelaunch(int feature, boolean resolved, boolean relaunched, boolean simdEnabled) {
        return simdEnabled && !relaunched && !resolved && feature >= FIRST_FEATURE;
    }

    /**
     * Whether the kernels are enabled, the gate {@link CpuSimd#enabled} reads. An invalid value keeps the
     * relaunch on and is rejected by name when the engine is asked for a kernel.
     */
    static boolean simdEnabled(Properties properties) {
        try {
            return CpuSimd.enabled(properties);
        } catch (IllegalArgumentException invalid) {
            return true;
        }
    }

    /**
     * The command that runs {@code mainClass} again with the module resolved: this {@code java.home}'s
     * executable, {@code inputArguments} (the options this JVM started with, without a class path option
     * or an earlier relaunch marker), {@link #OPTION}, the relaunch marker, the class path, the main class
     * and {@code args}. The JVM options precede the class path, as they must.
     */
    static List<String> relaunchCommand(int feature, List<String> inputArguments, String classPath,
                                        String mainClass, String[] args) {
        List<String> cmd = new ArrayList<>();
        cmd.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        for (int i = 0; i < inputArguments.size(); i++) {
            String argument = inputArguments.get(i);
            if (argument == null || argument.trim().isEmpty()) {
                continue;
            }
            if (isClassPathOption(argument)) {
                if (!argument.contains("=")) {
                    i++; // its value is a path, not a JVM option that may ride along
                }
                continue;
            }
            if (!argument.startsWith("-D" + RELAUNCHED_PROPERTY + "=")) {
                cmd.add(argument);
            }
        }
        cmd.addAll(options(feature));
        cmd.add("-D" + RELAUNCHED_PROPERTY + "=true");
        cmd.add("-cp");
        cmd.add(classPath);
        cmd.add(mainClass);
        Collections.addAll(cmd, args);
        return cmd;
    }

    /**
     * Whether this JVM runs under a debug agent, which a child JVM cannot share on the same address: such a
     * launcher keeps the scalar kernels rather than breaking the debugger's attach.
     */
    static boolean underDebugAgent(List<String> inputArguments) {
        for (String argument : inputArguments) {
            if (argument == null) {
                continue;
            }
            if (argument.startsWith("-agentlib:jdwp") || argument.startsWith("-Xrunjdwp")
                    || argument.equals("-Xdebug")
                    || (argument.startsWith("-agentpath:") && argument.contains("jdwp"))) {
                return true;
            }
        }
        return false;
    }

    /**
     * Relaunches this program with the module resolved when {@link #needsRelaunch} asks for it, waits for
     * the child and returns its exit code; null when no relaunch ran. A child that cannot start leaves the
     * launcher running with its scalar kernels and says why on standard error.
     */
    static Integer relaunchIfNeeded(boolean simdEnabled, String mainClass, String[] args) {
        int feature = Runtime.version().feature();
        boolean relaunched = Boolean.parseBoolean(System.getProperty(RELAUNCHED_PROPERTY));
        if (!needsRelaunch(feature, resolved(), relaunched, simdEnabled)) {
            return null;
        }
        List<String> inputArguments = ManagementFactory.getRuntimeMXBean().getInputArguments();
        if (underDebugAgent(inputArguments)) {
            System.err.println("Warning: the JVM runs under a debug agent; " + OPTION
                    + " is not added and the CPU kernels stay scalar");
            return null;
        }
        String classPath = System.getProperty("java.class.path", "");
        if (classPath.isEmpty()) {
            return null; // no class path to relaunch with: keep the scalar kernels rather than fail
        }
        List<String> cmd = relaunchCommand(feature, inputArguments, classPath, mainClass, args);
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        try {
            Process child = pb.start();
            Runtime.getRuntime().addShutdownHook(new Thread(child::destroy));
            return child.waitFor();
        } catch (IOException e) {
            System.err.println("Warning: cannot relaunch the JVM with " + OPTION + " (" + e.getMessage()
                    + "); the CPU kernels stay scalar");
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Warning: the relaunch with " + OPTION + " was interrupted; the CPU kernels"
                    + " stay scalar");
            return null;
        }
    }

    /** Whether an argument names the class path, which the relaunch supplies itself. */
    private static boolean isClassPathOption(String argument) {
        return argument.equals("-cp") || argument.equals("-classpath")
                || argument.equals("--class-path") || argument.startsWith("--class-path=");
    }
}
