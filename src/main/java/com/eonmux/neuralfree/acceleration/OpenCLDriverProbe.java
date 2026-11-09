package com.eonmux.neuralfree.acceleration;

import org.apache.logging.log4j.Logger;
import org.jocl.CLException;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.clGetPlatformIDs;

/**
 * The subprocess probe that guards {@link HWAcceleration}'s OpenCL initialization against broken
 * drivers: it runs {@link OpenCLProbe} in a child JVM, so an ICD whose dispatch tables are corrupt
 * kills only the probe, and classifies platforms as unsafe from the probe's exit code and output.
 */
final class OpenCLDriverProbe {

    // Matches Mesa/libdrm "pci id for fd N: VVVV:DDDD, driver (null)" messages.
    // These are emitted by libdrm when it enumerates PCI devices but can't identify
    // the kernel driver for a GPU. They are produced during Mesa-based ICD
    // initialization (rusticl, clover) — the message indicates the Mesa ICD is in a
    // bad state, *not* that the vendor whose PCI ID appears is broken.
    private static final Pattern DRIVER_NULL_PATTERN =
            Pattern.compile("pci id for fd \\d+: [0-9a-fA-F]{4}:[0-9a-fA-F]{4},\\s*driver \\(null\\)");

    private OpenCLDriverProbe() {
    }

    /**
     * Probes OpenCL platforms using a subprocess to detect broken drivers.
     * <p>
     * Detection strategies:
     * <ol>
     *   <li><b>Success whitelist:</b> Any platform for which the probe successfully
     *       enumerated at least one device (printed {@code DEVICE:i:j:OK:...}) is
     *       trusted as safe and never marked unsafe.</li>
     *   <li><b>Crash detection:</b> If the probe subprocess was killed by a signal
     *       (SIGSEGV=139), per-platform probes identify which platform(s) crash.</li>
     *   <li><b>Mesa/libdrm warning heuristic:</b> If the probe output contains
     *       {@code "driver (null)"} lines, Mesa-based platforms (rusticl, clover)
     *       that did NOT successfully enumerate devices are marked unsafe — they
     *       may return stub {@code cl_device_id}s with corrupt dispatch tables.</li>
     *   <li><b>Device query failure:</b> Platforms whose probe devices returned
     *       {@code QUERY_FAILED} with no OK siblings are marked unsafe.</li>
     * </ol>
     *
     * @param log the logger the probe traces its subprocess output through ({@link
     *            HWAcceleration}'s, so the trace reads as before)
     * @return set of platform indices that should be skipped (empty if all safe)
     */
    static Set<Integer> probeForBrokenPlatforms(Logger log) {
        Set<Integer> unsafe = new HashSet<>();

        if (Boolean.getBoolean("neuralfree.opencl.skip.probe")) {
            return unsafe;
        }

        try {
            // Probe ALL platforms in one subprocess, capturing output
            List<String> probeOutput = new ArrayList<>();
            int exitCode = runOpenCLProbe(-1, probeOutput, log);

            if (exitCode < 0) {
                // Probe mechanism couldn't run (classpath, JOCL missing, etc.)
                System.out.println("OpenCL probe could not run. " +
                        "Proceeding with direct initialization.");
                return unsafe;
            }

            // Parse probe output: platforms, device statuses, and libdrm warnings.
            boolean driverNullWarning = false;
            Map<Integer, String> platformNames = new HashMap<>();
            Set<Integer> platformsWithOkDevice = new HashSet<>();
            Set<Integer> platformsWithFailedDevice = new HashSet<>();

            for (String line : probeOutput) {
                if (DRIVER_NULL_PATTERN.matcher(line).find()) {
                    driverNullWarning = true;
                }
                if (line.startsWith("PLATFORM:")) {
                    String[] parts = line.split(":", 3);
                    if (parts.length == 3) {
                        try {
                            platformNames.put(Integer.parseInt(parts[1]), parts[2]);
                        } catch (NumberFormatException ignored) {}
                    }
                } else if (line.startsWith("DEVICE:")) {
                    // Format: DEVICE:platformIdx:deviceIdx:OK:name
                    //     or: DEVICE:platformIdx:deviceIdx:QUERY_FAILED:errcode
                    String[] parts = line.split(":", 5);
                    if (parts.length >= 4) {
                        try {
                            int platIdx = Integer.parseInt(parts[1]);
                            if ("OK".equals(parts[3])) {
                                platformsWithOkDevice.add(platIdx);
                            } else if ("QUERY_FAILED".equals(parts[3])) {
                                platformsWithFailedDevice.add(platIdx);
                            }
                        } catch (NumberFormatException ignored) {}
                    }
                }
            }

            // Strategy 1: per-platform probing if the initial probe crashed.
            if (exitCode >= 128) {
                System.out.println("OpenCL probe crashed (signal " + (exitCode - 128) +
                        "). Probing individual platforms...");

                int[] numPlatforms = new int[1];
                if (platformCount(numPlatforms) != CL_SUCCESS || numPlatforms[0] <= 0) {
                    return unsafe;
                }

                for (int i = 0; i < numPlatforms[0]; i++) {
                    List<String> platOutput = new ArrayList<>();
                    int result = runOpenCLProbe(i, platOutput, log);
                    if (result >= 128) {
                        unsafe.add(i);
                        System.out.println("  Platform " + i + ": UNSAFE (signal " +
                                (result - 128) + ")");
                    } else {
                        System.out.println("  Platform " + i + ": OK");
                        // Absorb successful device info from per-platform output too.
                        for (String line : platOutput) {
                            if (line.startsWith("DEVICE:")) {
                                String[] parts = line.split(":", 5);
                                if (parts.length >= 4 && "OK".equals(parts[3])) {
                                    try {
                                        platformsWithOkDevice.add(Integer.parseInt(parts[1]));
                                    } catch (NumberFormatException ignored) {}
                                }
                            }
                        }
                    }
                }
            }

            // Strategy 2: Mesa/libdrm "driver (null)" warnings point at a broken
            // Mesa ICD, not at the vendor whose PCI ID appears in the message.
            // Mark rusticl/clover/mesa platforms unsafe only when they did not
            // successfully enumerate devices (so we don't punish working ones).
            if (driverNullWarning) {
                for (Map.Entry<Integer, String> platEntry : platformNames.entrySet()) {
                    int platIdx = platEntry.getKey();
                    if (platformsWithOkDevice.contains(platIdx)) continue;

                    String lowerName = platEntry.getValue().toLowerCase();
                    boolean isMesaIcd = lowerName.contains("rusticl")
                            || lowerName.contains("clover")
                            || lowerName.contains("mesa");
                    if (isMesaIcd && unsafe.add(platIdx)) {
                        System.out.println("Skipping OpenCL platform " + platIdx +
                                " ('" + platEntry.getValue() + "'): Mesa-based ICD" +
                                " reported libdrm 'driver (null)' and enumerated no" +
                                " usable devices — stub cl_device_id would crash.");
                    }
                }
            }

            // Strategy 3: platforms whose probe devices failed and which had no OK
            // siblings are untrustworthy.
            for (Integer platIdx : platformsWithFailedDevice) {
                if (platformsWithOkDevice.contains(platIdx)) continue;
                if (unsafe.add(platIdx)) {
                    String name = platformNames.getOrDefault(platIdx, "?");
                    System.out.println("Skipping OpenCL platform " + platIdx +
                            " ('" + name + "'): probe reported device query failure.");
                }
            }
        } catch (Throwable t) {
            System.out.println("OpenCL probe mechanism failed: " + t.getMessage());
        }

        return unsafe;
    }

    /**
     * Queries the OpenCL platform count into {@code numPlatforms[0]} and returns the call's
     * status. JOCL's exception mode is process-wide and has no getter, so it is left as the
     * caller found it: a status returned (exceptions disabled, JOCL's default and the mode the
     * rest of this class relies on) and a {@link CLException} thrown (exceptions enabled) both
     * come back as the status.
     */
    private static int platformCount(int[] numPlatforms) {
        try {
            return clGetPlatformIDs(0, null, numPlatforms);
        } catch (CLException e) {
            return e.getStatus();
        }
    }

    /**
     * Runs the OpenCL probe in a subprocess.
     *
     * @param platformIndex platform to probe, or -1 for all platforms
     * @param capturedOutput list to receive all lines of stdout+stderr from the probe
     * @param log            the logger the probe traces each captured line through
     * @return process exit code: 0=success, 1=catchable error, 128+=signal kill,
     *         -1=probe mechanism failed (couldn't launch subprocess)
     */
    private static int runOpenCLProbe(int platformIndex, List<String> capturedOutput,
                                      Logger log) {
        try {
            String javaHome = System.getProperty("java.home");
            String classpath = System.getProperty("java.class.path");
            String libraryPath = System.getProperty("java.library.path");

            List<String> cmd = new ArrayList<>();
            cmd.add(javaHome + "/bin/java");
            cmd.add("-cp");
            cmd.add(classpath);
            cmd.add("-Xmx64m");
            String errorFile = System.getProperty("os.name", "").toLowerCase().contains("win")
                    ? "NUL" : "/dev/null";
            cmd.add("-XX:ErrorFile=" + errorFile);
            if (libraryPath != null && !libraryPath.isEmpty()) {
                cmd.add("-Djava.library.path=" + libraryPath);
            }
            cmd.add(OpenCLProbe.class.getName());
            if (platformIndex >= 0) {
                cmd.add(String.valueOf(platformIndex));
            }

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // Capture output lines (needed for broken-driver heuristic)
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    capturedOutput.add(line);
                    if (log.isDebugEnabled()) {
                        log.debug("OpenCL probe: {}", line);
                    }
                }
            }

            boolean completed = process.waitFor(15, TimeUnit.SECONDS);
            if (!completed) {
                process.destroyForcibly().waitFor(5, TimeUnit.SECONDS);
                System.out.println("OpenCL probe timed out" +
                        (platformIndex >= 0 ? " for platform " + platformIndex : ""));
                return 139; // Treat timeout as crash
            }

            return process.exitValue();
        } catch (Exception e) {
            System.out.println("Failed to run OpenCL probe: " + e.getMessage());
            return -1; // Probe mechanism failed
        }
    }
}
