package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorDevice;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorProviders;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import com.eonmux.neuralfree.acceleration.hip.HipDriver;
import com.eonmux.neuralfree.acceleration.metal.MetalDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.vulkan.VkDevice;

import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import static jcuda.driver.JCudaDriver.cuDeviceTotalMem;

/**
 * The device listing behind {@code --list-devices}, the mirror of llama.cpp's
 * {@code common_print_available_devices} ({@code common/arg.cpp:1128}): every device the engine can
 * enumerate, per API with its total memory, and the {@code (none)} line when no API has one. The
 * engine's enumeration decides which devices appear ({@link HWAcceleration}, OpenCL always and CUDA,
 * Vulkan, HIP and Metal as its initialization allows, plus every accelerator API whose backend
 * provider is installed); the listing adds no policy of its own. Memory is the
 * figure the API reports per device — OpenCL's {@code CL_DEVICE_GLOBAL_MEM_SIZE}, CUDA's device
 * total, HIP's {@code hipMemGetInfo} total, Vulkan's device-local heaps — and the unknown case prints
 * no size instead of a guess. Each line prints the index {@code -dev}/{@code -devd} selects: the
 * device's 0-based position inside its API's list, the number the engine's device selection matches
 * ({@code GpuDevices}), so a listed device is named by the same number on the command line.
 */
public final class DeviceListing {

    /** One enumerable device: its API, its selector index inside that API's list and its total memory. */
    public static final class Entry {

        private final String api;
        private final int index;
        private final String name;
        private final long totalBytes;

        Entry(String api, int index, String name, long totalBytes) {
            this.api = api;
            this.index = index;
            this.name = name;
            this.totalBytes = totalBytes;
        }

        /**
         * The API that enumerates the device: {@code OpenCL}, {@code CUDA}, {@code Vulkan}, {@code HIP},
         * {@code Metal} or an accelerator API's label ({@code SYCL}, {@code zDNN}, ...).
         */
        public String api() {
            return api;
        }

        /** The device's 0-based position inside its API's list: the index {@code -dev} selects it by. */
        public int index() {
            return index;
        }

        /** The device name as the driver reports it. */
        public String name() {
            return name;
        }

        /** The device's total memory in bytes, or -1 when the API does not report it. */
        public long totalBytes() {
            return totalBytes;
        }
    }

    private static final long MIB = 1024L * 1024L;

    private DeviceListing() {
    }

    /** Every device of every API the engine enumerates; an API that initializes to nothing is skipped. */
    public static List<Entry> available() {
        return available(AcceleratorProviders.installed());
    }

    /** {@link #available()} over an explicit provider table, for an embedder or a test. */
    static List<Entry> available(AcceleratorProviders providers) {
        List<Entry> out = new ArrayList<>();
        HWAcceleration hw = new HWAcceleration(EnumSet.allOf(BackendType.class), providers);
        try {
            int openCl = 0;
            for (CLDevice d : hw.getCLDevices()) {
                out.add(new Entry("OpenCL", openCl++, openClName(d), openClTotalBytes(d)));
            }
            int cuda = 0;
            for (CUDevice d : hw.getCUDevices()) {
                out.add(new Entry("CUDA", cuda++, name(d.getDeviceName()), cudaTotalBytes(d)));
            }
            int vulkan = 0;
            for (VkDevice d : hw.getVkDevices()) {
                out.add(new Entry("Vulkan", vulkan++, name(d.getDeviceName()), vulkanTotalBytes(d)));
            }
            int hip = 0;
            for (HipDevice d : hw.getHipDevices()) {
                out.add(new Entry("HIP", hip++, name(d.getDeviceName()), hipTotalBytes(d)));
            }
            int metal = 0;
            for (MetalDevice d : hw.getMetalDevices()) {
                out.add(new Entry("Metal", metal++, name(d.getDeviceName()), metalTotalBytes(d)));
            }
            for (BackendType api : BackendType.values()) {
                if (!api.isAccelerator()) {
                    continue;
                }
                int index = 0;
                for (AcceleratorDevice d : hw.getAcceleratorDevices(api)) {
                    out.add(new Entry(api.label(), index++, name(d.name()), d.totalMemoryBytes()));
                }
            }
        } finally {
            hw.shutdown();
        }
        return out;
    }

    /**
     * The OpenCL device's name from its cross-API identity: the engine's enumeration fills
     * {@code CL_DEVICE_NAME} when it accepts the device, and {@code getDeviceName} of
     * {@link CLDevice} does not report it.
     */
    private static String openClName(CLDevice device) {
        try {
            return name(device.getIdentity().name());
        } catch (Throwable t) {
            return "(unnamed)";
        }
    }

    /** A device name that always prints: an empty or missing name is the {@code (unnamed)} placeholder. */
    private static String name(String name) {
        return name == null || name.trim().isEmpty() ? "(unnamed)" : name.trim();
    }

    /** The OpenCL device's global memory, or -1 when the query fails. */
    private static long openClTotalBytes(CLDevice device) {
        try {
            device.queryCapabilities();
            Object size = device.getDeviceInfo().get("CL_DEVICE_GLOBAL_MEM_SIZE");
            if (size instanceof Number) {
                return ((Number) size).longValue();
            }
        } catch (Throwable t) {
            // an attribute the driver refuses must not hide the device
        }
        return -1;
    }

    /** The CUDA device's total memory, or -1 when the query fails. */
    private static long cudaTotalBytes(CUDevice device) {
        try {
            long[] total = new long[1];
            cuDeviceTotalMem(total, device.getDeviceId());
            return total[0];
        } catch (Throwable t) {
            return -1;
        }
    }

    /** The Vulkan device's device-local memory, or -1 when its properties are missing. */
    private static long vulkanTotalBytes(VkDevice device) {
        try {
            return device.getInfo().totalDeviceLocalBytes();
        } catch (Throwable t) {
            return -1;
        }
    }

    /**
     * The HIP device's total memory from {@code hipMemGetInfo}: the runtime reports free and total of
     * the calling thread's current device, so the device is selected first. {@code hipDeviceTotalMem},
     * the same figure per ordinal, is the fallback when the info query fails; -1 when both fail.
     */
    /** The Metal device's recommended working set, the framework's memory estimate, or -1 when the query fails. */
    private static long metalTotalBytes(MetalDevice device) {
        try {
            return Math.max(0L, device.getRecommendedMaxWorkingSet());
        } catch (Throwable t) {
            return -1;
        }
    }

    private static long hipTotalBytes(HipDevice device) {
        HipDriver driver = device.getDriver();
        try {
            driver.setDevice(device.getIndex());
            long[] info = driver.memGetInfo();
            if (info != null && info.length >= 2) {
                return info[1];
            }
        } catch (Throwable t) {
            // an info query the driver refuses must not hide the device
        }
        try {
            return driver.totalMem(device.getIndex());
        } catch (Throwable t) {
            return -1;
        }
    }

    /** The listing llama.cpp prints: the header, one line per device, or the {@code (none)} line. */
    public static String lines(List<Entry> devices) {
        StringBuilder sb = new StringBuilder("Available devices:\n");
        if (devices.isEmpty()) {
            sb.append("  (none)\n");
            return sb.toString();
        }
        for (Entry d : devices) {
            sb.append("  ").append(d.api()).append(' ').append(d.index()).append(": ")
                    .append(d.name());
            if (d.totalBytes() >= 0) {
                sb.append(" (").append(d.totalBytes() / MIB).append(" MiB)");
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /** Prints the available devices of this machine to {@code out}. */
    public static void print(Appendable out) {
        try {
            out.append(lines(available()));
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the device listing", e);
        }
    }
}
