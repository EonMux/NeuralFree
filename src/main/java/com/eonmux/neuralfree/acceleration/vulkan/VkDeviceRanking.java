package com.eonmux.neuralfree.acceleration.vulkan;

import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import static org.lwjgl.vulkan.VK12.VK_API_VERSION_1_2;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_AMD_OPEN_SOURCE;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_AMD_PROPRIETARY;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_MESA_DOZEN;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_MESA_NVK;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_MESA_RADV;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_MESA_TURNIP;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_MOLTENVK;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_NVIDIA_PROPRIETARY;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_QUALCOMM_PROPRIETARY;

/**
 * Picks the Vulkan device to run on: a forced name match first; then devices
 * below Vulkan 1.2, without a compute queue, or of CPU type (unless the only one)
 * are excluded; then discrete beats integrated beats virtual; ties go to the
 * OpenCL-measured winner when known, else to the largest device-local heap.
 *
 * <ol>
 *   <li><b>Eligibility.</b> A device needs Vulkan 1.2, a compute queue family and timeline
 *       semaphores (what {@code VkDeviceFeatures} requires). When two entries are the same
 *       physical device seen through two drivers (same PCI address or device UUID), the
 *       preferred driver is kept, following llama.cpp
 *       ({@code ggml-vulkan.cpp ggml_vk_instance_init}); two MoltenVK entries are never merged.
 *       CPU devices (llvmpipe, SwiftShader) are dropped when any other device remains.</li>
 *   <li><b>An explicit choice wins.</b> {@code -Dneuralfree.vulkan.device=<i>substring</i>}
 *       (or, when unset, {@code -Dneuralfree.device}) selects the first eligible device whose
 *       name contains that text, case-insensitively. A filter that matches nothing is
 *       reported and the ranking proceeds normally.</li>
 *   <li><b>Class.</b> Discrete, then integrated, then virtual, then unknown, then CPU.</li>
 *   <li><b>Measured hint.</b> Within the best class, the device that is the same physical
 *       device as the OpenCL-measured winner ({@code CLDeviceRanking}) wins: published
 *       attributes cannot be compared across vendors, the measurement can.</li>
 *   <li><b>Device memory</b>, then <b>shader cores</b> (when the driver reports them),
 *       then enumeration order.</li>
 * </ol>
 */
public final class VkDeviceRanking {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkDeviceRanking.class);

    /** Forces a Vulkan device by name substring, case-insensitive. */
    public static final String DEVICE_PROPERTY = "neuralfree.vulkan.device";
    /** API-independent device filter, consulted when {@link #DEVICE_PROPERTY} is unset. */
    public static final String GLOBAL_DEVICE_PROPERTY = "neuralfree.device";

    private static final int VENDOR_AMD = 0x1002;
    private static final int VENDOR_INTEL = 0x8086;
    private static final int VENDOR_NVIDIA = 0x10DE;
    private static final int VENDOR_QUALCOMM = 0x5143;

    private VkDeviceRanking() {
    }

    /** The devices that pass the exclusion step, in input order. */
    public static List<VkPhysicalDeviceInfo> eligible(List<VkPhysicalDeviceInfo> devices) {
        if (devices == null || devices.isEmpty()) {
            return Collections.emptyList();
        }
        List<VkPhysicalDeviceInfo> usable = new ArrayList<>(devices.size());
        for (VkPhysicalDeviceInfo info : devices) {
            if (info == null) {
                continue;
            }
            String reason = unsupportedReason(info);
            if (reason != null) {
                LOGGER.info("Vulkan device '{}' excluded: {}", info.name(), reason);
            } else {
                addDeduplicated(usable, info);
            }
        }
        List<VkPhysicalDeviceInfo> nonCpu = new ArrayList<>(usable.size());
        for (VkPhysicalDeviceInfo info : usable) {
            if (info.deviceClass() != DeviceClass.CPU) {
                nonCpu.add(info);
            }
        }
        if (!nonCpu.isEmpty() && nonCpu.size() < usable.size()) {
            for (VkPhysicalDeviceInfo info : usable) {
                if (info.deviceClass() == DeviceClass.CPU) {
                    LOGGER.debug("Vulkan CPU device '{}' excluded: a GPU is available", info.name());
                }
            }
            return Collections.unmodifiableList(nonCpu);
        }
        return Collections.unmodifiableList(usable);
    }

    /**
     * Why {@code info} cannot run the compute backend, or null when it can.
     */
    public static String unsupportedReason(VkPhysicalDeviceInfo info) {
        if (!info.apiAtLeast(VK_API_VERSION_1_2)) {
            return "Vulkan 1.2 is required, the device supports " + info.apiVersionString();
        }
        if (info.computeQueueFamily() < 0) {
            return "no queue family supports compute";
        }
        if (!info.supports(VkPhysicalDeviceInfo.Feature.TIMELINE_SEMAPHORE)) {
            return "timeline semaphores are not supported";
        }
        return null;
    }

    /**
     * The device to run on, or null when none is eligible.
     *
     * @param measuredHint identity of the OpenCL-measured fastest device, or null
     */
    public static VkPhysicalDeviceInfo fastest(List<VkPhysicalDeviceInfo> devices, DeviceIdentity measuredHint) {
        List<VkPhysicalDeviceInfo> candidates = eligible(devices);
        if (candidates.isEmpty()) {
            return null;
        }
        List<Profile> profiles = new ArrayList<>(candidates.size());
        for (VkPhysicalDeviceInfo info : candidates) {
            profiles.add(new Profile(info.name(), info.deviceClass(), info.identity(), info.totalDeviceLocalBytes(),
                    info.shaderCoreCount()));
        }
        VkPhysicalDeviceInfo best = candidates.get(rank(profiles, measuredHint));
        if (candidates.size() > 1) {
            LOGGER.info("Selected Vulkan device '{}' of {} eligible", best.name(), candidates.size());
        }
        return best;
    }

    /** {@link #fastest(List, DeviceIdentity)} without a hint. */
    public static VkPhysicalDeviceInfo fastest(List<VkPhysicalDeviceInfo> devices) {
        return fastest(devices, null);
    }

    /** The enumerated {@link VkDevice} whose physical device wins {@link #fastest(List, DeviceIdentity)}. */
    public static VkDevice fastestDevice(List<VkDevice> devices, DeviceIdentity measuredHint) {
        if (devices == null || devices.isEmpty()) {
            return null;
        }
        List<VkPhysicalDeviceInfo> infos = new ArrayList<>(devices.size());
        for (VkDevice device : devices) {
            if (device != null) {
                infos.add(device.getInfo());
            }
        }
        VkPhysicalDeviceInfo best = fastest(infos, measuredHint);
        if (best == null) {
            return null;
        }
        for (VkDevice device : devices) {
            if (device != null && device.getInfo() == best) {
                return device;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------- policy

    /** Index of the winning profile, or -1 when the list is empty. */
    static int rank(List<Profile> profiles, DeviceIdentity measuredHint) {
        if (profiles == null || profiles.isEmpty()) {
            return -1;
        }
        int forced = forcedChoice(profiles);
        if (forced >= 0) {
            return forced;
        }
        int best = 0;
        for (int i = 1; i < profiles.size(); i++) {
            if (better(profiles.get(i), profiles.get(best), measuredHint)) {
                best = i;
            }
        }
        return best;
    }

    /** True when {@code a} ranks strictly above {@code b}. */
    private static boolean better(Profile a, Profile b, DeviceIdentity hint) {
        int classA = classRank(a.deviceClass);
        int classB = classRank(b.deviceClass);
        if (classA != classB) {
            return classA < classB;
        }
        boolean hintA = matches(a.identity, hint);
        boolean hintB = matches(b.identity, hint);
        if (hintA != hintB) {
            return hintA;
        }
        if (a.deviceLocalBytes != b.deviceLocalBytes) {
            return a.deviceLocalBytes > b.deviceLocalBytes;
        }
        return a.shaderCores > b.shaderCores;
    }

    static int classRank(DeviceClass deviceClass) {
        switch (deviceClass) {
            case DISCRETE:
                return 0;
            case INTEGRATED:
                return 1;
            case VIRTUAL:
                return 2;
            case UNKNOWN:
                return 3;
            case CPU:
            default:
                return 4;
        }
    }

    private static boolean matches(DeviceIdentity identity, DeviceIdentity hint) {
        return hint != null && identity != null && identity.samePhysicalDevice(hint);
    }

    private static int forcedChoice(List<Profile> profiles) {
        String property = DEVICE_PROPERTY;
        String wanted = System.getProperty(DEVICE_PROPERTY);
        if (wanted == null || wanted.trim().isEmpty()) {
            property = GLOBAL_DEVICE_PROPERTY;
            wanted = System.getProperty(GLOBAL_DEVICE_PROPERTY);
        }
        if (wanted == null || wanted.trim().isEmpty()) {
            return -1;
        }
        String needle = wanted.trim().toLowerCase(Locale.ROOT);
        for (int i = 0; i < profiles.size(); i++) {
            if (profiles.get(i).name.toLowerCase(Locale.ROOT).contains(needle)) {
                LOGGER.info("Selected Vulkan device '{}': matched -D{}={}", profiles.get(i).name, property, wanted);
                return i;
            }
        }
        LOGGER.warn("-D{}={} matched none of {} eligible Vulkan device(s); ranking normally",
                property, wanted, profiles.size());
        return -1;
    }

    // ---------------------------------------------------------------- de-duplication

    /**
     * Adds {@code info} to {@code kept} unless an entry already there is the same physical
     * device; in that case the entry with the preferred driver stays.
     */
    private static void addDeduplicated(List<VkPhysicalDeviceInfo> kept, VkPhysicalDeviceInfo info) {
        for (int i = 0; i < kept.size(); i++) {
            VkPhysicalDeviceInfo old = kept.get(i);
            if (!sameDevice(old, info)) {
                continue;
            }
            if (driverPriority(info) < driverPriority(old)) {
                LOGGER.info("Vulkan device '{}' is listed twice; using driver '{}' over '{}'",
                        info.name(), info.driverName(), old.driverName());
                kept.set(i, info);
            } else {
                LOGGER.info("Vulkan device '{}' is listed twice; using driver '{}' over '{}'",
                        old.name(), old.driverName(), info.driverName());
            }
            return;
        }
        kept.add(info);
    }

    /**
     * Same physical device by PCI address or device UUID. Name-only identities never match,
     * so two identical cards are kept apart; two MoltenVK entries never match because
     * MoltenVK may report one UUID for distinct GPUs of a multi-GPU card.
     */
    private static boolean sameDevice(VkPhysicalDeviceInfo a, VkPhysicalDeviceInfo b) {
        if (a.driverId() == VK_DRIVER_ID_MOLTENVK && b.driverId() == VK_DRIVER_ID_MOLTENVK) {
            return false;
        }
        DeviceIdentity ia = a.identity();
        DeviceIdentity ib = b.identity();
        boolean comparable = (ia.pciBusId() != null && ib.pciBusId() != null)
                || (ia.uuid() != null && ib.uuid() != null);
        return comparable && ia.samePhysicalDevice(ib);
    }

    /**
     * Smaller is preferred; drivers outside the table rank last. The table is llama.cpp's driver_priorities
     * (ggml-vulkan.cpp; MIT License, see THIRD-PARTY-NOTICES.md).
     */
    static int driverPriority(VkPhysicalDeviceInfo info) {
        int driver = info.driverId();
        if (driver == VK_DRIVER_ID_MESA_DOZEN) {
            return 100;
        }
        switch (info.vendorId()) {
            case VENDOR_AMD:
                if (driver == VK_DRIVER_ID_MESA_RADV) {
                    return 1;
                }
                if (driver == VK_DRIVER_ID_AMD_OPEN_SOURCE) {
                    return 2;
                }
                if (driver == VK_DRIVER_ID_AMD_PROPRIETARY) {
                    return 3;
                }
                break;
            case VENDOR_INTEL:
                if (driver == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) {
                    return 1;
                }
                if (driver == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) {
                    return 2;
                }
                break;
            case VENDOR_NVIDIA:
                if (driver == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
                    return 1;
                }
                if (driver == VK_DRIVER_ID_MESA_NVK) {
                    return 2;
                }
                break;
            case VENDOR_QUALCOMM:
                if (driver == VK_DRIVER_ID_QUALCOMM_PROPRIETARY) {
                    return 1;
                }
                if (driver == VK_DRIVER_ID_MESA_TURNIP) {
                    return 2;
                }
                break;
            default:
                break;
        }
        return Integer.MAX_VALUE;
    }

    /** What the ranking knows about one candidate, separate from Vulkan handles so the policy is testable. */
    static final class Profile {
        final String name;
        final DeviceClass deviceClass;
        final DeviceIdentity identity;
        final long deviceLocalBytes;
        final int shaderCores;

        Profile(String name, DeviceClass deviceClass, DeviceIdentity identity, long deviceLocalBytes,
                int shaderCores) {
            this.name = name == null ? "<unnamed>" : name;
            this.deviceClass = deviceClass == null ? DeviceClass.UNKNOWN : deviceClass;
            this.identity = identity;
            this.deviceLocalBytes = deviceLocalBytes;
            this.shaderCores = shaderCores;
        }
    }
}
