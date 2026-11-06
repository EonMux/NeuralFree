package com.eonmux.neuralfree.acceleration.metal;

import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.UUID;

/**
 * One Metal device found during enumeration.
 *
 * <p>Enumeration creates no device state; the Metal framework keeps the device object and this
 * class addresses it by its ordinal. Every query is a native call, cached after the first
 * successful answer.</p>
 *
 * <p>The device's cross-API identity is its name. Metal exposes a registry id but no PCI address
 * or UUID through its public API, and Vulkan on the same machine (MoltenVK) does report a UUID, so
 * synthesising one here would make {@link DeviceIdentity#samePhysicalDevice} treat the two views of
 * one Apple GPU as different devices. Leaving them unknown lets the names decide.</p>
 */
public final class MetalDevice implements HardwareDevice {

    private static final Logger LOGGER = LoggingUtil.getLogger(MetalDevice.class);

    private final UUID uuid = UUID.randomUUID();
    private final int index;
    private final String deviceName;
    private volatile DeviceIdentity identity;
    private volatile Long maxThreadsPerThreadgroup;
    private volatile Long maxThreadgroupMemory;
    private volatile Long recommendedMaxWorkingSet;
    private volatile Long maxBufferLength;
    private volatile Boolean unifiedMemory;
    private volatile String deviceFamily;

    /**
     * @param index the Metal device ordinal
     * @throws IllegalArgumentException when the device has no name
     * @throws IllegalStateException    when the shim cannot report the device
     */
    public MetalDevice(int index) {
        if (index < 0) {
            throw new IllegalArgumentException("Metal device index must not be negative: " + index);
        }
        String name = MetalNative.getDeviceName(index);
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Metal device " + index + " reports no name");
        }
        this.index = index;
        this.deviceName = name.trim();
    }

    @Override
    public void initialize() {
        // The Metal framework needs no per-device initialization before use.
    }

    @Override
    public void release() {
        // A device object is owned by the framework; nothing is retained per enumeration.
    }

    @Override
    public UUID getUUID() {
        return uuid;
    }

    @Override
    public String getDeviceName() {
        return deviceName;
    }

    /** The Metal device ordinal. */
    public int getIndex() {
        return index;
    }

    /** Identity by device name; see the class comment for why no stronger key is reported. */
    public DeviceIdentity getIdentity() {
        DeviceIdentity id = identity;
        if (id == null) {
            id = DeviceIdentity.ofName(deviceName);
            identity = id;
        }
        return id;
    }

    /**
     * {@link DeviceClass#INTEGRATED} when the GPU shares memory with the CPU (Apple silicon and
     * Intel integrated GPUs), {@link DeviceClass#DISCRETE} otherwise, unknown when the query fails.
     */
    public DeviceClass getDeviceClass() {
        Boolean unified = hasUnifiedMemory();
        if (unified == null) {
            return DeviceClass.UNKNOWN;
        }
        return unified ? DeviceClass.INTEGRATED : DeviceClass.DISCRETE;
    }

    /** Threads per threadgroup, the product of the three limits. */
    public long getMaxThreadsPerThreadgroup() {
        Long cached = maxThreadsPerThreadgroup;
        if (cached == null) {
            cached = attribute(MetalNative.ATTR_MAX_THREADS_PER_THREADGROUP);
            maxThreadsPerThreadgroup = cached;
        }
        return cached;
    }

    /** Threadgroup memory in bytes. */
    public long getMaxThreadgroupMemory() {
        Long cached = maxThreadgroupMemory;
        if (cached == null) {
            cached = attribute(MetalNative.ATTR_MAX_THREADGROUP_MEMORY);
            maxThreadgroupMemory = cached;
        }
        return cached;
    }

    /** The recommended maximum working-set size in bytes, the framework's VRAM estimate. */
    public long getRecommendedMaxWorkingSet() {
        Long cached = recommendedMaxWorkingSet;
        if (cached == null) {
            cached = attribute(MetalNative.ATTR_RECOMMENDED_MAX_WORKING_SET);
            recommendedMaxWorkingSet = cached;
        }
        return cached;
    }

    /** The largest buffer allocation in bytes. */
    public long getMaxBufferLength() {
        Long cached = maxBufferLength;
        if (cached == null) {
            cached = attribute(MetalNative.ATTR_MAX_BUFFER_LENGTH);
            maxBufferLength = cached;
        }
        return cached;
    }

    /** True when the GPU and CPU share memory; null when the query failed. */
    public Boolean hasUnifiedMemory() {
        Boolean cached = unifiedMemory;
        if (cached == null) {
            try {
                cached = attribute(MetalNative.ATTR_HAS_UNIFIED_MEMORY) != 0;
            } catch (RuntimeException e) {
                LOGGER.debug("Metal unified-memory query failed for '{}': {}", deviceName, e.toString());
                return null;
            }
            unifiedMemory = cached;
        }
        return cached;
    }

    /** The GPU families the device supports, for diagnostics, for example {@code apple7,common3}. */
    public String getDeviceFamily() {
        String cached = deviceFamily;
        if (cached == null) {
            cached = family();
            deviceFamily = cached;
        }
        return cached;
    }

    /** True when the device is a low-power GPU (the integrated one of a MacBook). */
    public boolean isLowPower() {
        return attribute(MetalNative.ATTR_IS_LOW_POWER) != 0;
    }

    /** One integer device attribute of the shim's selector table. */
    public long getAttribute(int selector) {
        return attribute(selector);
    }

    @Override
    public void queryCapabilities() {
        LOGGER.debug("Metal Device Detected");
        LOGGER.debug("====================");
        LOGGER.debug("Device {}: {} ({}, {})", uuid, deviceName, getDeviceFamily(), getDeviceClass());
        LOGGER.debug("Max threads per threadgroup: {}", getMaxThreadsPerThreadgroup());
        LOGGER.debug("Threadgroup memory: {} bytes", getMaxThreadgroupMemory());
        LOGGER.debug("Recommended working set: {} MB", getRecommendedMaxWorkingSet() / (1024 * 1024));
        LOGGER.debug("Unified memory: {}", hasUnifiedMemory());
    }

    @Override
    public String toString() {
        return "MetalDevice{" + index + ", '" + deviceName + "'}";
    }

    private long attribute(int selector) {
        long[] out = new long[1];
        int status = MetalNative.deviceAttribute(index, selector, out);
        if (status != 0) {
            throw new IllegalStateException("Metal device '" + deviceName + "' attribute " + selector
                    + " query failed (status " + status + ")");
        }
        return out[0];
    }

    private String family() {
        String value = MetalNative.getDeviceFamily(index);
        return value == null || value.trim().isEmpty() ? "unknown" : value.trim();
    }
}
