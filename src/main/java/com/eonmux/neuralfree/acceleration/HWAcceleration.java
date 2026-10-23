package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaDeviceRanking;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDeviceRanking;
import com.eonmux.neuralfree.acceleration.vulkan.VkDevice;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import com.eonmux.neuralfree.acceleration.vulkan.VkInstanceManager;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.datatype.structures.ObjectSet;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.CudaException;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import org.jocl.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static jcuda.driver.JCudaDriver.*;
import static org.jocl.CL.*;


public class HWAcceleration {

    public static boolean enableCuda = false;
    public static boolean enableVulkan = false;

    ObjectSet<CLDevice> CLDevices = new ObjectSet<>();
    ObjectSet<CUDevice> CUDevices = new ObjectSet<>();
    ObjectSet<VkDevice> VkDevices = new ObjectSet<>();

    /** The APIs this instance enumerates. */
    private final Set<BackendType> apis;
    /** True when {@link #apis} came from the static enable flags rather than an explicit set. */
    private final boolean apisFromStaticFlags;
    /** Why an enumerated API offers no device, when its initialization failed. */
    private final Map<BackendType, String> enumerationFailures = new EnumMap<>(BackendType.class);
    /** Why the last explicit-mode backend creation of an API failed; cleared when one succeeds. */
    private final Map<BackendType, String> creationFailures =
            Collections.synchronizedMap(new EnumMap<>(BackendType.class));
    /** Backends handed out by the createBackend methods, closed by {@link #shutdown()}. */
    private final List<ComputeBackend> createdBackends = Collections.synchronizedList(new ArrayList<>());
    /** Memoized OpenCL ranking: the winner for the device-filter properties it was computed under. */
    private CLDevice rankedOpenClDevice;
    /** Which device-filter property values {@link #rankedOpenClDevice} was ranked under. */
    private String rankedOpenClFilter;
    private boolean openClRanked;

    /**
     * Enumerates OpenCL always, CUDA when {@link #enableCuda} is set and Vulkan when
     * {@link #enableVulkan} is set.
     */
    public HWAcceleration() {
        this(defaultApis(), true);
    }

    /**
     * Enumerates exactly the given APIs; the static {@link #enableCuda} and
     * {@link #enableVulkan} flags are not consulted.
     *
     * @param apis the APIs to enumerate; not null, not empty
     * @throws IllegalArgumentException when {@code apis} is null, empty or holds null
     */
    public HWAcceleration(Set<BackendType> apis) {
        this(apis, false);
    }

    private HWAcceleration(Set<BackendType> apis, boolean apisFromStaticFlags) {
        this.apis = validatedApis(apis);
        this.apisFromStaticFlags = apisFromStaticFlags;
        /*
            It is possible we didn't properly shutdown the program previously, in-which case the GPU will be in a bad-state
            So we call shutdown first to ensure that it is ready to be used again.
        */
        shutdown();

        if (this.apis.contains(BackendType.OPENCL)) {
            try {
                initCL();
            } catch (Throwable e) {
                System.out.println("OpenCL initialization failed: " + e.getMessage());
                enumerationFailures.put(BackendType.OPENCL, String.valueOf(e.getMessage()));
            }
        }
        if (this.apis.contains(BackendType.CUDA)) {
            try {
                initCUDA();
            } catch (Throwable e) {
                System.out.println("CUDA initialization failed (native libraries may be unavailable): " + e.getMessage());
                enumerationFailures.put(BackendType.CUDA, String.valueOf(e.getMessage()));
                enableCuda = false;
            }
        }
        if (this.apis.contains(BackendType.VULKAN)) {
            try {
                initVulkan();
            } catch (Throwable e) {
                System.out.println("Vulkan initialization failed: " + e.getMessage());
                enumerationFailures.put(BackendType.VULKAN, String.valueOf(e.getMessage()));
                enableVulkan = false;
            }
        }
    }

    /**
     * An instance that enumerates what {@code mode} can use: every API for AUTO, otherwise
     * only the API the mode names. The static {@link #enableCuda} and {@link #enableVulkan}
     * flags are not consulted, so {@code forMode(CUDA)} enumerates CUDA without setting them.
     *
     * @param mode the acceleration mode; null means AUTO
     */
    public static HWAcceleration forMode(ACCELERATION_MODE mode) {
        if (mode == null) {
            mode = ACCELERATION_MODE.AUTO;
        }
        switch (mode) {
            case AUTO:
                return new HWAcceleration(EnumSet.allOf(BackendType.class));
            case OPENCL:
                return new HWAcceleration(EnumSet.of(BackendType.OPENCL));
            case CUDA:
                return new HWAcceleration(EnumSet.of(BackendType.CUDA));
            case VULKAN:
                return new HWAcceleration(EnumSet.of(BackendType.VULKAN));
            default:
                throw new IllegalArgumentException("Unknown acceleration mode: " + mode);
        }
    }

    /** The APIs this instance enumerated (unmodifiable). */
    public Set<BackendType> getEnumeratedApis() {
        return apis;
    }

    private static Set<BackendType> defaultApis() {
        Set<BackendType> defaults = EnumSet.of(BackendType.OPENCL);
        if (enableCuda) {
            defaults.add(BackendType.CUDA);
        }
        if (enableVulkan) {
            defaults.add(BackendType.VULKAN);
        }
        return defaults;
    }

    private static Set<BackendType> validatedApis(Set<BackendType> apis) {
        if (apis == null || apis.isEmpty()) {
            throw new IllegalArgumentException("HWAcceleration needs at least one API to enumerate");
        }
        Set<BackendType> copy = EnumSet.noneOf(BackendType.class);
        for (BackendType api : apis) {
            if (api == null) {
                throw new IllegalArgumentException("API set must not contain null: " + apis);
            }
            copy.add(api);
        }
        return Collections.unmodifiableSet(copy);
    }

    /**
     * This method is called to free up GPU resources
     *
     * <p>Backends the createBackend methods handed out are closed first, then the
     * devices are released. CUDA devices drop only their own primary-context
     * retains, so a backend created elsewhere on the same device keeps working.</p>
     */
    public void shutdown() {
        closeCreatedBackends();

        if (!CUDevices.isEmpty()) {
            //CUDA
            try {
                for (CUDevice device : CUDevices) {
                    //Release the primary-context retains the devices hold
                    device.release();
                }
                CUDevices.clear();
            } catch (Throwable e) {
                System.out.println("CUDA shutdown failed (native libraries may be unavailable): " + e.getMessage());
            }
        }

        if (!VkDevices.isEmpty()) {
            //VULKAN
            try {
                for (VkDevice device : VkDevices) {
                    device.release();
                }
                VkDevices.clear();
            } catch (Throwable e) {
                System.out.println("Vulkan shutdown failed: " + e.getMessage());
            }
        }

        //OPENCL
        try {
            for (CLDevice clDevice : CLDevices) {
                clDevice.release();
            }
            CLDevices.clear();
        } catch (Throwable e) {
            System.out.println("OpenCL shutdown failed: " + e.getMessage());
        }
        synchronized (this) {
            rankedOpenClDevice = null;
            rankedOpenClFilter = null;
            openClRanked = false;
        }
    }

    /**
     * Closes every backend the createBackend methods handed out, before the devices
     * underneath them are released. Closing is idempotent, so a backend its caller
     * already closed is unaffected.
     */
    private void closeCreatedBackends() {
        List<ComputeBackend> toClose;
        synchronized (createdBackends) {
            toClose = new ArrayList<>(createdBackends);
            createdBackends.clear();
        }
        for (ComputeBackend backend : toClose) {
            try {
                backend.close();
            } catch (Throwable e) {
                LoggingUtil.getLogger(this).warn("Closing the {} backend failed: {}", backend.backendType(),
                        e.getMessage());
            }
        }
    }

    /** Records a backend handed out by a createBackend method so {@link #shutdown()} closes it. */
    private ComputeBackend track(ComputeBackend backend) {
        if (backend != null) {
            synchronized (createdBackends) {
                if (!createdBackends.contains(backend)) {
                    createdBackends.add(backend);
                }
            }
        }
        return backend;
    }

    private void initCUDA() {
        // Enable CUDA Exceptions
        try {
            JCudaDriver.setExceptionsEnabled(true);
        } catch (UnsatisfiedLinkError | NoClassDefFoundError e) {
            throw new IllegalStateException("JCuda driver library unavailable: " + e.getMessage(), e);
        }

        // Initialize CUDA
        try {
            cuInit(0);
        } catch (CudaException ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            enumerationFailures.put(BackendType.CUDA, "cuInit failed: " + ex.getMessage());
            return;
        }

        // Get our list of CUDA Devices
        int[] cudaCount = {0};
        cuDeviceGetCount(cudaCount);

        // Enumeration creates no context: each backend retains the device's primary context (CudaContext).
        for (int i = 0; i < cudaCount[0]; i++) {
            CUdevice device = new CUdevice();
            cuDeviceGet(device, i);
            CUDevices.add(new CUDevice(device));
        }

        queryCudaDeviceInfo();

        // Let the user know if we've found any CUDA capable devices
        LoggingUtil.getLogger(this).info("Found " + cudaCount[0] + " CUDA capable device(s)");
    }

    private void initCL() {
        if (Boolean.getBoolean("neuralfree.opencl.disable")) {
            System.out.println("OpenCL disabled via system property neuralfree.opencl.disable");
            return;
        }

        // Probe OpenCL platforms in a subprocess to detect broken drivers.
        // Some OpenCL ICDs (e.g., NVIDIA on certain GPUs) have corrupted dispatch
        // tables that cause SIGSEGV in clGetDeviceInfo — a native crash that Java
        // try/catch cannot intercept. Running the probe in a subprocess isolates
        // the crash so only the probe process dies, not the main JVM.
        Set<Integer> unsafePlatforms = probeForBrokenPlatforms();

        int[] numPlatforms = new int[1];
        int err = clGetPlatformIDs(0, null, numPlatforms);
        if (err != CL_SUCCESS || numPlatforms[0] <= 0) {
            LoggingUtil.getLogger(this).info("No OpenCL platforms found");
            return;
        }

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
        err = clGetPlatformIDs(platforms.length, platforms, null);
        if (err != CL_SUCCESS) {
            LoggingUtil.getLogger(this).warn("Failed to enumerate OpenCL platforms");
            return;
        }

        for (int i = 0; i < platforms.length; i++) {
            if (unsafePlatforms.contains(i)) {
                System.out.println("Skipping OpenCL platform " + i +
                        " (driver probe detected crash — likely broken ICD)");
                continue;
            }

            try {
                int[] numDevices = new int[1];
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, null, numDevices);

                if (numDevices[0] <= 0) continue;

                cl_platform_id platform = platforms[i];
                cl_context_properties contextProperties = new cl_context_properties();
                contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

                cl_device_id[] devicesArray = new cl_device_id[numDevices[0]];
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices[0], devicesArray, null);

                for (cl_device_id device : devicesArray) {
                    if (device == null) continue;

                    long[] typeSize = new long[1];
                    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, typeSize);
                    if (typeSize[0] <= 0 || typeSize[0] > 1024) {
                        continue;
                    }

                    cl_context context = clCreateContext(
                            contextProperties, 1, new cl_device_id[]{device},
                            null, null, null);
                    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, null, null);

                    CLDevice clDevice = new CLDevice(device, context, queue);

                    String deviceName = (String) clDevice.getDeviceInfo().get("CL_DEVICE_NAME");
                    if (deviceName == null || deviceName.trim().isEmpty()) {
                        clReleaseCommandQueue(queue);
                        clReleaseContext(context);
                        continue;
                    }

                    CLDevices.add(clDevice);
                }
            } catch (Throwable t) {
                System.out.println("OpenCL platform " + i + " failed: " + t.getMessage());
            }
        }

        queryCLDeviceInfo();

        LoggingUtil.getLogger(this).info("Found " + CLDevices.size() + " OpenCL capable device(s)");
    }

    // Matches Mesa/libdrm "pci id for fd N: VVVV:DDDD, driver (null)" messages.
    // These are emitted by libdrm when it enumerates PCI devices but can't identify
    // the kernel driver for a GPU. They are produced during Mesa-based ICD
    // initialization (rusticl, clover) — the message indicates the Mesa ICD is in a
    // bad state, *not* that the vendor whose PCI ID appears is broken.
    private static final Pattern DRIVER_NULL_PATTERN =
            Pattern.compile("pci id for fd \\d+: [0-9a-fA-F]{4}:[0-9a-fA-F]{4},\\s*driver \\(null\\)");

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
     * @return set of platform indices that should be skipped (empty if all safe)
     */
    private Set<Integer> probeForBrokenPlatforms() {
        Set<Integer> unsafe = new HashSet<>();

        if (Boolean.getBoolean("neuralfree.opencl.skip.probe")) {
            return unsafe;
        }

        try {
            // Probe ALL platforms in one subprocess, capturing output
            List<String> probeOutput = new ArrayList<>();
            int exitCode = runOpenCLProbe(-1, probeOutput);

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
                    int result = runOpenCLProbe(i, platOutput);
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
     * @return process exit code: 0=success, 1=catchable error, 128+=signal kill,
     *         -1=probe mechanism failed (couldn't launch subprocess)
     */
    private int runOpenCLProbe(int platformIndex, List<String> capturedOutput) {
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
                    if (LoggingUtil.getLogger(this).isDebugEnabled()) {
                        LoggingUtil.getLogger(this).debug("OpenCL probe: {}", line);
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

    private void queryCudaDeviceInfo() {
        if (!LoggingUtil.getLogger(this).isDebugEnabled()) {
            return;
        }

        for (CUDevice cuDevices : CUDevices) {
            cuDevices.queryCapabilities();
        }
    }

    private void queryCLDeviceInfo() {
        if (!LoggingUtil.getLogger(this).isDebugEnabled()) {
            return;
        }

        for (CLDevice clDevice : CLDevices) {
            clDevice.queryCapabilities();
        }
    }

    public ObjectSet<CLDevice> getCLDevices() {
        return CLDevices;
    }

    public ObjectSet<CUDevice> getCUDevices() {
        return CUDevices;
    }

    public ObjectSet<VkDevice> getVkDevices() {
        return VkDevices;
    }

    /**
     * Enumerates every Vulkan device that can run the compute backend
     * ({@link VkDeviceRanking#eligible}). Enumeration creates no logical device; each
     * {@link VkDevice} holds its own reference on the shared instance.
     *
     * @throws NoBackendAvailableException when the Vulkan loader or a 1.2 instance is unavailable
     */
    private void initVulkan() {
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            List<VkPhysicalDeviceInfo> physical = manager.devices();
            for (VkPhysicalDeviceInfo info : VkDeviceRanking.eligible(physical)) {
                VkDevices.add(new VkDevice(manager, info));
            }
            if (VkDevices.isEmpty() && !physical.isEmpty()) {
                enumerationFailures.put(BackendType.VULKAN, ineligibleVulkanDevices(physical));
            }
        } finally {
            manager.release();
        }
        queryVulkanDeviceInfo();
        LoggingUtil.getLogger(this).info("Found " + VkDevices.size() + " Vulkan capable device(s)");
    }

    private static String ineligibleVulkanDevices(List<VkPhysicalDeviceInfo> physical) {
        StringBuilder sb = new StringBuilder("no Vulkan device can run the compute backend:");
        for (VkPhysicalDeviceInfo info : physical) {
            String reason = VkDeviceRanking.unsupportedReason(info);
            sb.append(" '").append(info.name()).append("': ").append(reason == null ? "eligible" : reason)
                    .append(';');
        }
        return sb.toString();
    }

    private void queryVulkanDeviceInfo() {
        if (!LoggingUtil.getLogger(this).isDebugEnabled()) {
            return;
        }

        for (VkDevice vkDevice : VkDevices) {
            vkDevice.queryCapabilities();
        }
    }

    public ComputeBackend createBackend(HardwareDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("Device cannot be null");
        }
        
        if (device instanceof CLDevice) {
            return track(new OclComputeBackend((CLDevice) device));
        } else if (device instanceof CUDevice) {
            return track(new CudaComputeBackend((CUDevice) device));
        } else if (device instanceof VkDevice) {
            return track(((VkDevice) device).getBackend());
        }

        throw new IllegalArgumentException("Unknown device type: " + device.getClass().getName());
    }

    public ComputeBackend createBackend(CLDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("CLDevice cannot be null");
        }
        return track(new OclComputeBackend(device));
    }

    public ComputeBackend createBackend(CUDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("CUDevice cannot be null");
        }
        return track(new CudaComputeBackend(device));
    }

    public ComputeBackend createBackend(VkDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("VkDevice cannot be null");
        }
        return track(device.getBackend());
    }

    /**
     * Creates a ComputeBackend on the fastest device this host offers.
     *
     * <p><b>The mode picks the API, not the device.</b> Whichever mode is asked
     * for, the backend is built on the best device that API can reach — never on
     * whichever one the driver happened to enumerate first. Each backend applies
     * the same policy: a discrete GPU beats an integrated one, and between
     * equals the faster device wins, decided by measurement where the published
     * attributes are not comparable. See
     * {@link com.eonmux.neuralfree.acceleration.ocl.CLDeviceRanking} for why
     * reading the attributes alone gets this wrong.</p>
     *
     * <p>AUTO additionally chooses the API, but only among the APIs this instance
     * enumerated and that found a device. Which APIs those are is fixed at
     * construction: {@link #HWAcceleration()} enumerates OpenCL, plus CUDA when
     * {@link #enableCuda} and Vulkan when {@link #enableVulkan} was set before
     * constructing (both default to {@code false}, so such an instance offers AUTO
     * OpenCL alone unless a flag was set); {@link #HWAcceleration(Set)} enumerates
     * exactly the given APIs; {@link #forMode}{@code (AUTO)} enumerates all three.</p>
     *
     * <p>AUTO is device-first. It settles on one physical device, then drives it
     * with the first API in the order that can: the device named by
     * {@code -Dneuralfree.device=<substring>} when set, otherwise the OpenCL
     * ranking's measured winner (else Vulkan's, else CUDA's); the API order is
     * {@code -Dneuralfree.backend.order} (default {@code vulkan,cuda,opencl}).
     * The views one GPU offers through several APIs are recognised as one device
     * by {@link DeviceIdentity#samePhysicalDevice}. When no API runs that device,
     * each API's own winner is tried, discrete first.</p>
     *
     * @param mode The acceleration mode; null means AUTO. AUTO selects among the
     *             enumerated APIs as described above. Specific modes (VULKAN, CUDA,
     *             OPENCL) only try that API.
     * @return A ComputeBackend for the best available GPU
     * @throws NoBackendAvailableException if no GPU backend can be initialized
     */
    public ComputeBackend createBackend(ACCELERATION_MODE mode) {
        return createBackend(mode, BackendRequirement.ANY);
    }

    /**
     * {@link #createBackend(ACCELERATION_MODE)} with a condition the backend must
     * meet, for example {@code KernelLibraryRequirement.of(KernelLibraries.LLM)}.
     *
     * <p>AUTO skips every backend the requirement rejects, closes it, and logs the
     * reason. An explicit mode builds its API's fastest device and, when the
     * requirement rejects it, closes it and throws
     * {@code "<API> backend unavailable: <reason>"}.</p>
     *
     * @param mode        the acceleration mode; null means AUTO
     * @param requirement condition the backend must meet; {@link BackendRequirement#ANY} accepts all
     * @return an accepted backend; {@link #shutdown()} closes it
     * @throws NoBackendAvailableException when no backend is available or accepted
     * @throws IllegalArgumentException    when {@code requirement} is null or
     *                                     {@code -Dneuralfree.backend.order} is malformed
     */
    public ComputeBackend createBackend(ACCELERATION_MODE mode, BackendRequirement requirement) {
        if (requirement == null) {
            throw new IllegalArgumentException("Backend requirement cannot be null; use BackendRequirement.ANY");
        }
        if (mode == null) {
            mode = ACCELERATION_MODE.AUTO;
        }

        switch (mode) {
            case AUTO:
                return track(createBackendWithFallback(requirement));
            case VULKAN:
                return track(BackendSelector.requireAccepted(BackendType.VULKAN, createVulkanBackendOrThrow(),
                        requirement));
            case CUDA:
                return track(BackendSelector.requireAccepted(BackendType.CUDA, createCudaBackendOrThrow(),
                        requirement));
            case OPENCL:
                return track(BackendSelector.requireAccepted(BackendType.OPENCL, createOpenCLBackendOrThrow(),
                        requirement));
            default:
                throw new IllegalArgumentException("Unknown acceleration mode: " + mode);
        }
    }

    /**
     * Creates a ComputeBackend using automatic detection (AUTO mode).
     * This is equivalent to createBackend(ACCELERATION_MODE.AUTO).
     * 
     * @return A ComputeBackend for the best available GPU
     * @throws NoBackendAvailableException if no GPU backend can be initialized
     */
    public ComputeBackend createBackend() {
        return createBackend(ACCELERATION_MODE.AUTO);
    }

    /**
     * AUTO: device first, then API ({@link BackendSelector}). Each API contributes
     * every device it enumerated plus its own ranking's winner: {@code CLDeviceRanking},
     * {@code CudaDeviceRanking} and {@code VkDeviceRanking} (hinted with the OpenCL
     * winner's identity). Every backend the requirement rejects is closed and its
     * reason logged; the exception lists every reason collected.
     */
    private ComputeBackend createBackendWithFallback(BackendRequirement requirement) {
        List<BackendType> order = BackendSelector.apiOrder();
        BackendSelector.Selection selection = BackendSelector.selectAuto(autoViews(), order,
                BackendSelector.deviceFilter(), requirement);
        if (selection.backend != null) {
            return selection.backend;
        }

        StringBuilder failureReasons = new StringBuilder();
        for (String reason : selection.reasons) {
            failureReasons.append(reason).append("; ");
        }

        throw new NoBackendAvailableException(
            "No GPU backend available. " +
            "Please ensure you have a supported GPU with up-to-date drivers. " +
            "Troubleshooting: " + failureReasons.toString() + ". " +
            "For OpenCL, install OpenCL runtime from your GPU vendor (NVIDIA/AMD/Intel). " +
            "For CUDA, install CUDA Toolkit from NVIDIA. " +
            "For Vulkan, install Vulkan drivers from your GPU vendor."
        );
    }

    /**
     * Why a backend did not answer, told apart from a backend that was never
     * asked.
     *
     * <p>"no devices or initialization failed" reads as a hardware or driver
     * problem, and printing it for a backend whose enumeration is switched off
     * sends the reader to reinstall drivers they do not need. {@link #enableCuda}
     * and {@link #enableVulkan} are both {@code false} unless a caller sets them
     * before constructing, so on a stock setup that is the common case, not the
     * rare one.</p>
     */
    private static String whyUnavailable(String backend, String enableField,
                                         boolean enumerated, int deviceCount) {
        if (!enumerated) {
            return backend + ": enumeration disabled (set HWAcceleration." + enableField
                    + " before constructing to probe for it); ";
        }
        if (deviceCount == 0) {
            return backend + ": no devices found; ";
        }
        return backend + ": " + deviceCount + " device(s) found but initialization failed; ";
    }

    /**
     * Why an API offers no usable device: its initialization failure when recorded; for an API an
     * explicit API set left out, how to enumerate it; otherwise {@link #whyUnavailable}.
     */
    private String absentReason(BackendType api) {
        String label = BackendSelector.label(api);
        String failure = enumerationFailures.get(api);
        if (failure != null) {
            return label + ": initialization failed: " + failure;
        }
        if (!apis.contains(api) && (!apisFromStaticFlags || api == BackendType.OPENCL)) {
            return label + ": not enumerated by this instance (use HWAcceleration.forMode or include it in the API set)";
        }
        String reason;
        switch (api) {
            case VULKAN:
                reason = whyUnavailable(label, "enableVulkan", apis.contains(api), VkDevices.size());
                break;
            case CUDA:
                reason = whyUnavailable(label, "enableCuda", apis.contains(api), CUDevices.size());
                break;
            default:
                reason = whyUnavailable(label, "n/a", apis.contains(api), CLDevices.size());
                break;
        }
        return reason.endsWith("; ") ? reason.substring(0, reason.length() - 2) : reason;
    }

    /** What each API offers AUTO: its devices as candidates and its own ranking's winner. */
    private List<BackendSelector.ApiView> autoViews() {
        List<BackendSelector.ApiView> views = new ArrayList<>(3);
        views.add(openClView());
        views.add(cudaView());
        views.add(vulkanView());
        return views;
    }

    private BackendSelector.ApiView openClView() {
        if (CLDevices.isEmpty()) {
            return BackendSelector.ApiView.absent(BackendType.OPENCL, absentReason(BackendType.OPENCL));
        }
        CLDevice best = openClWinner();
        List<BackendSelector.Candidate> devices = new ArrayList<>(CLDevices.size());
        BackendSelector.Candidate winner = null;
        for (CLDevice device : CLDevices) {
            BackendSelector.Candidate c = new BackendSelector.Candidate(BackendType.OPENCL, openClIdentity(device),
                    openClDeviceClass(device), () -> new OclComputeBackend(device));
            devices.add(c);
            if (device == best) {
                winner = c;
            }
        }
        return new BackendSelector.ApiView(BackendType.OPENCL, devices, winner == null ? devices.get(0) : winner,
                null);
    }

    private BackendSelector.ApiView cudaView() {
        if (CUDevices.isEmpty()) {
            return BackendSelector.ApiView.absent(BackendType.CUDA, absentReason(BackendType.CUDA));
        }
        CUDevice best = fastestCudaDevice();
        List<BackendSelector.Candidate> devices = new ArrayList<>(CUDevices.size());
        BackendSelector.Candidate winner = null;
        for (CUDevice device : CUDevices) {
            BackendSelector.Candidate c = new BackendSelector.Candidate(BackendType.CUDA, cudaIdentity(device),
                    device.getDeviceClass(), () -> new CudaComputeBackend(device));
            devices.add(c);
            if (device == best) {
                winner = c;
            }
        }
        return new BackendSelector.ApiView(BackendType.CUDA, devices, winner == null ? devices.get(0) : winner,
                null);
    }

    private BackendSelector.ApiView vulkanView() {
        if (VkDevices.isEmpty()) {
            return BackendSelector.ApiView.absent(BackendType.VULKAN, absentReason(BackendType.VULKAN));
        }
        VkDevice best = VkDeviceRanking.fastestDevice(VkDevices, openClHint());
        List<BackendSelector.Candidate> devices = new ArrayList<>(VkDevices.size());
        BackendSelector.Candidate winner = null;
        for (VkDevice device : VkDevices) {
            BackendSelector.Candidate c = new BackendSelector.Candidate(BackendType.VULKAN, device.getIdentity(),
                    device.getInfo().deviceClass(), device::getBackend);
            devices.add(c);
            if (device == best) {
                winner = c;
            }
        }
        return new BackendSelector.ApiView(BackendType.VULKAN, devices, winner == null ? devices.get(0) : winner,
                null);
    }

    /**
     * The OpenCL device to run on: {@link CLDeviceRanking#fastest} over the devices,
     * narrowed first to the names matching {@code -Dneuralfree.device} unless
     * {@code -Dneuralfree.opencl.device} is set (the ranking honours that one itself).
     * Memoized per value of both properties, because the ranking may time a probe kernel.
     */
    private synchronized CLDevice openClWinner() {
        if (CLDevices.isEmpty()) {
            return null;
        }
        String openClFilter = System.getProperty(CLDeviceRanking.DEVICE_PROPERTY);
        boolean ownFilter = openClFilter != null && !openClFilter.trim().isEmpty();
        String filter = ownFilter ? null : BackendSelector.deviceFilter();
        String rankingKey = ownFilter ? "opencl:" + openClFilter.trim() : "device:" + filter;
        if (openClRanked && Objects.equals(rankingKey, rankedOpenClFilter)) {
            return rankedOpenClDevice;
        }
        List<CLDevice> candidates = BackendSelector.filterByName(CLDevices,
                d -> openClIdentity(d).name(), filter, "OpenCL");
        rankedOpenClDevice = CLDeviceRanking.fastest(candidates);
        rankedOpenClFilter = rankingKey;
        openClRanked = true;
        return rankedOpenClDevice;
    }

    /** Identity of the OpenCL winner, the measured cross-vendor hint for Vulkan; null without OpenCL. */
    private DeviceIdentity openClHint() {
        CLDevice best = openClWinner();
        return best == null ? null : openClIdentity(best);
    }

    private static DeviceIdentity openClIdentity(CLDevice device) {
        try {
            return device.getIdentity();
        } catch (RuntimeException e) {
            Object name = device.getDeviceInfo() == null ? null : device.getDeviceInfo().get("CL_DEVICE_NAME");
            return DeviceIdentity.ofName(name instanceof String ? (String) name : "");
        }
    }

    /** The device's cross-API identity, or a name-only identity when the driver query fails. */
    private static DeviceIdentity cudaIdentity(CUDevice device) {
        try {
            return device.getIdentity();
        } catch (RuntimeException e) {
            String name = device.getDeviceName();
            return DeviceIdentity.ofName(name == null ? "" : name);
        }
    }

    /**
     * Only GPU devices are enumerated: integrated when they share host memory
     * ({@code CL_DEVICE_HOST_UNIFIED_MEMORY}), discrete otherwise, unknown when the
     * driver declined the query.
     */
    private static DeviceClass openClDeviceClass(CLDevice device) {
        Object unified = device.getDeviceInfo() == null ? null
                : device.getDeviceInfo().get("CL_DEVICE_HOST_UNIFIED_MEMORY");
        if (!(unified instanceof Boolean)) {
            return DeviceClass.UNKNOWN;
        }
        return ((Boolean) unified) ? DeviceClass.INTEGRATED : DeviceClass.DISCRETE;
    }

    private ComputeBackend createVulkanBackendOrThrow() {
        ComputeBackend backend = tryCreateVulkanBackend();
        if (backend == null) {
            throw explicitModeUnavailable(BackendType.VULKAN, VkDevices.size(),
                "Vulkan backend requested but no Vulkan devices available. " +
                "Please install Vulkan drivers from your GPU vendor (NVIDIA/AMD/Intel)."
            );
        }
        return backend;
    }

    private ComputeBackend createCudaBackendOrThrow() {
        ComputeBackend backend = tryCreateCudaBackend();
        if (backend == null) {
            throw explicitModeUnavailable(BackendType.CUDA, CUDevices.size(),
                "CUDA backend requested but no CUDA devices available. " +
                "Please install CUDA Toolkit from NVIDIA and ensure you have an NVIDIA GPU."
            );
        }
        return backend;
    }

    private ComputeBackend createOpenCLBackendOrThrow() {
        ComputeBackend backend = tryCreateOpenCLBackend();
        if (backend == null) {
            throw explicitModeUnavailable(BackendType.OPENCL, CLDevices.size(),
                "OpenCL backend requested but no OpenCL devices available. " +
                "Please install OpenCL runtime from your GPU vendor (NVIDIA/AMD/Intel)."
            );
        }
        return backend;
    }

    /**
     * Why an explicit mode got no backend: with no device, {@code noDeviceMessage} followed by
     * the enumeration cause; with devices, {@code "<API> backend unavailable: backend creation
     * failed: <cause>"}.
     */
    private NoBackendAvailableException explicitModeUnavailable(BackendType api, int deviceCount,
                                                                String noDeviceMessage) {
        if (deviceCount == 0) {
            return new NoBackendAvailableException(noDeviceMessage + " Cause: " + absentReason(api));
        }
        String failure = creationFailures.get(api);
        return new NoBackendAvailableException(BackendSelector.label(api) + " backend unavailable: "
                + "backend creation failed: " + (failure == null ? "no device was selected" : failure));
    }

    private ComputeBackend tryCreateVulkanBackend() {
        if (VkDevices.isEmpty()) {
            return null;
        }
        try {
            // The ranking breaks ties with the OpenCL-measured winner when OpenCL was enumerated.
            VkDevice vkDevice = VkDeviceRanking.fastestDevice(VkDevices, openClHint());
            if (vkDevice != null) {
                ComputeBackend backend = vkDevice.getBackend();
                creationFailures.remove(BackendType.VULKAN);
                return backend;
            }
        } catch (Exception e) {
            LoggingUtil.getLogger(this).warn("Failed to create Vulkan backend: {}", e.getMessage());
            creationFailures.put(BackendType.VULKAN, String.valueOf(e.getMessage()));
        }
        return null;
    }

    private ComputeBackend tryCreateCudaBackend() {
        if (CUDevices.isEmpty()) {
            return null;
        }
        try {
            CUDevice cuDevice = fastestCudaDevice();
            if (cuDevice != null) {
                ComputeBackend backend = new CudaComputeBackend(cuDevice);
                creationFailures.remove(BackendType.CUDA);
                return backend;
            }
        } catch (Exception e) {
            LoggingUtil.getLogger(this).warn("Failed to create CUDA backend: {}", e.getMessage());
            creationFailures.put(BackendType.CUDA, String.valueOf(e.getMessage()));
        }
        return null;
    }

    /**
     * The fastest CUDA device, by multiprocessor count times clock.
     *
     * <p>Unlike the OpenCL list this one is single-vendor, so those two numbers
     * really are comparable across its entries and no timed probe is needed.
     * Integrated parts (Tegra, and the iGPU half of a laptop) lose to any
     * discrete device present, matching what
     * {@link com.eonmux.neuralfree.acceleration.ocl.CLDeviceRanking} does for
     * OpenCL and what {@code VkComputeBackend} has always done for Vulkan.</p>
     *
     * <p>The policy lives in {@link CudaDeviceRanking}, which also honours
     * {@code -Dneuralfree.cuda.device} and {@code -Dneuralfree.device}.</p>
     */
    private CUDevice fastestCudaDevice() {
        return CudaDeviceRanking.fastest(CUDevices);
    }

    private ComputeBackend tryCreateOpenCLBackend() {
        if (CLDevices.isEmpty()) {
            return null;
        }
        try {
            // NOT CLDevices.get(0): that is ICD enumeration order, which on a
            // laptop lists the integrated GPU before the discrete card and so
            // runs every kernel on the slowest device in the machine.
            // CLDeviceRanking.fastest, narrowed by -Dneuralfree.device when set.
            CLDevice clDevice = openClWinner();
            if (clDevice != null) {
                ComputeBackend backend = new OclComputeBackend(clDevice);
                creationFailures.remove(BackendType.OPENCL);
                return backend;
            }
        } catch (Exception e) {
            LoggingUtil.getLogger(this).warn("Failed to create OpenCL backend: {}", e.getMessage());
            creationFailures.put(BackendType.OPENCL, String.valueOf(e.getMessage()));
        }
        return null;
    }
}
