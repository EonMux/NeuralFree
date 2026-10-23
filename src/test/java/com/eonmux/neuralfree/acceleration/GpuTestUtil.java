package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.vulkan.VkDevice;
import org.junit.Assume;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public final class GpuTestUtil {

    private static final AtomicBoolean openCLChecked = new AtomicBoolean(false);
    private static final AtomicBoolean openCLAvailable = new AtomicBoolean(false);
    private static final AtomicReference<String> openCLDeviceName = new AtomicReference<>("");

    private static final AtomicBoolean cudaChecked = new AtomicBoolean(false);
    private static final AtomicBoolean cudaAvailable = new AtomicBoolean(false);
    private static final AtomicReference<String> cudaDeviceName = new AtomicReference<>("");

    private static final AtomicBoolean vulkanChecked = new AtomicBoolean(false);
    private static final AtomicBoolean vulkanAvailable = new AtomicBoolean(false);
    private static final AtomicReference<String> vulkanDeviceName = new AtomicReference<>("");

    private GpuTestUtil() {
    }

    public static void assumeOpenCLAvailable() {
        if (!openCLChecked.getAndSet(true)) {
            checkOpenCL();
        }
        String deviceName = openCLDeviceName.get();
        Assume.assumeTrue("OpenCL GPU not available" + (deviceName.isEmpty() ? "" : " (expected: " + deviceName + ")"), 
                          openCLAvailable.get());
    }

    public static void assumeCudaAvailable() {
        if (!cudaChecked.getAndSet(true)) {
            checkCuda();
        }
        String deviceName = cudaDeviceName.get();
        Assume.assumeTrue("CUDA GPU not available" + (deviceName.isEmpty() ? "" : " (expected: " + deviceName + ")"), 
                          cudaAvailable.get());
    }

    public static void assumeVulkanAvailable() {
        if (!vulkanChecked.getAndSet(true)) {
            checkVulkan();
        }
        String deviceName = vulkanDeviceName.get();
        Assume.assumeTrue("Vulkan GPU not available" + (deviceName.isEmpty() ? "" : " (expected: " + deviceName + ")"), 
                          vulkanAvailable.get());
    }

    public static boolean isOpenCLAvailable() {
        if (!openCLChecked.getAndSet(true)) {
            checkOpenCL();
        }
        return openCLAvailable.get();
    }

    public static boolean isCudaAvailable() {
        if (!cudaChecked.getAndSet(true)) {
            checkCuda();
        }
        return cudaAvailable.get();
    }

    public static boolean isVulkanAvailable() {
        if (!vulkanChecked.getAndSet(true)) {
            checkVulkan();
        }
        return vulkanAvailable.get();
    }

    public static String getOpenCLDeviceName() {
        if (!openCLChecked.getAndSet(true)) {
            checkOpenCL();
        }
        return openCLDeviceName.get();
    }

    public static String getCudaDeviceName() {
        if (!cudaChecked.getAndSet(true)) {
            checkCuda();
        }
        return cudaDeviceName.get();
    }

    public static String getVulkanDeviceName() {
        if (!vulkanChecked.getAndSet(true)) {
            checkVulkan();
        }
        return vulkanDeviceName.get();
    }

    private static void checkOpenCL() {
        HWAcceleration hwAccel = null;
        try {
            hwAccel = new HWAcceleration();
            if (!hwAccel.getCLDevices().isEmpty()) {
                openCLAvailable.set(true);
                CLDevice device = hwAccel.getCLDevices().get(0);
                Object name = device.getDeviceInfo().get("CL_DEVICE_NAME");
                openCLDeviceName.set(name != null ? name.toString() : "Unknown");
                System.out.println("OpenCL GPU available: " + openCLDeviceName.get());
            } else {
                System.out.println("No OpenCL GPU devices available");
            }
        } catch (Throwable e) {
            System.out.println("OpenCL check failed: " + e.getMessage());
            openCLAvailable.set(false);
        } finally {
            if (hwAccel != null) {
                try {
                    hwAccel.shutdown();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    private static void checkCuda() {
        HWAcceleration hwAccel = null;
        try {
            HWAcceleration.enableCuda = true;
            hwAccel = new HWAcceleration();
            if (!hwAccel.getCUDevices().isEmpty()) {
                cudaAvailable.set(true);
                CUDevice device = hwAccel.getCUDevices().get(0);
                cudaDeviceName.set(device.getDeviceName());
                System.out.println("CUDA GPU available: " + cudaDeviceName.get());
            } else {
                System.out.println("No CUDA GPU devices available");
            }
        } catch (Throwable e) {
            System.out.println("CUDA check failed: " + e.getMessage());
            cudaAvailable.set(false);
        } finally {
            if (hwAccel != null) {
                try {
                    hwAccel.shutdown();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    private static void checkVulkan() {
        HWAcceleration hwAccel = null;
        try {
            HWAcceleration.enableVulkan = true;
            hwAccel = new HWAcceleration();
            if (!hwAccel.getVkDevices().isEmpty()) {
                vulkanAvailable.set(true);
                VkDevice device = hwAccel.getVkDevices().get(0);
                vulkanDeviceName.set(device.getBackend().capabilities().getDeviceName());
                System.out.println("Vulkan GPU available: " + vulkanDeviceName.get());
            } else {
                System.out.println("No Vulkan GPU devices available");
            }
        } catch (Throwable e) {
            System.out.println("Vulkan check failed: " + e.getMessage());
            vulkanAvailable.set(false);
        } finally {
            if (hwAccel != null) {
                try {
                    hwAccel.shutdown();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    public static void resetCache() {
        openCLChecked.set(false);
        openCLAvailable.set(false);
        openCLDeviceName.set("");
        cudaChecked.set(false);
        cudaAvailable.set(false);
        cudaDeviceName.set("");
        vulkanChecked.set(false);
        vulkanAvailable.set(false);
        vulkanDeviceName.set("");
    }
}
