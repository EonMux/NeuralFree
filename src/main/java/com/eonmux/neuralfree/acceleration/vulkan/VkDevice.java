package com.eonmux.neuralfree.acceleration.vulkan;

import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * One enumerated Vulkan physical device.
 *
 * <p>Enumeration creates no logical device. {@link #getBackend()} creates the
 * {@link VkComputeBackend} on first use, caches it, and creates a new one when the
 * cached backend has been closed. Each {@code VkDevice} holds one reference to the
 * {@link VkInstanceManager}, dropped by {@link #release()}.</p>
 */
public final class VkDevice implements HardwareDevice {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkDevice.class);

    private final VkInstanceManager instanceManager;
    private final VkPhysicalDeviceInfo info;
    private final UUID uuid;
    private VkComputeBackend backend;
    private boolean released;

    /**
     * The fastest eligible device of the shared instance ({@link VkDeviceRanking#fastest(List)}).
     *
     * @throws NoBackendAvailableException when Vulkan is unavailable or no device is eligible
     */
    public VkDevice() {
        this(VkDeviceSelection.fastest());
    }

    /**
     * A device of {@code instanceManager}; takes its own reference on the manager.
     *
     * @param info one of {@code instanceManager.devices()}
     * @throws IllegalArgumentException when an argument is null or {@code info} belongs to another instance
     * @throws IllegalStateException    when the manager has already been destroyed
     */
    public VkDevice(VkInstanceManager instanceManager, VkPhysicalDeviceInfo info) {
        this(VkDeviceSelection.adopt(instanceManager, info));
    }

    /** Takes the selection's instance reference over, or releases it when the device cannot be described. */
    private VkDevice(VkDeviceSelection selection) {
        UUID id;
        try {
            id = uuidOf(selection.info());
        } catch (RuntimeException e) {
            selection.release();
            throw e;
        }
        this.instanceManager = selection.manager();
        this.info = selection.info();
        this.uuid = id;
    }

    /**
     * Every eligible device ({@link VkDeviceRanking#eligible(List)}) of the shared instance, in
     * enumeration order; empty when Vulkan is unavailable. Each device holds its own instance reference.
     */
    public static List<VkDevice> enumerate() {
        VkInstanceManager manager;
        try {
            manager = VkInstanceManager.acquire();
        } catch (NoBackendAvailableException e) {
            LOGGER.info("Vulkan unavailable: {}", e.getMessage());
            return Collections.emptyList();
        }
        try {
            List<VkDevice> devices = new ArrayList<>();
            for (VkPhysicalDeviceInfo device : VkDeviceRanking.eligible(manager.devices())) {
                devices.add(new VkDevice(manager, device));
            }
            return Collections.unmodifiableList(devices);
        } finally {
            manager.release();
        }
    }

    @Override
    public void initialize() {
    }

    /** Closes the cached backend and drops this device's instance reference. Idempotent. */
    @Override
    public synchronized void release() {
        if (released) {
            return;
        }
        released = true;
        try {
            if (backend != null && !backend.isClosed()) {
                backend.close();
            }
        } finally {
            backend = null;
            instanceManager.release();
        }
    }

    /** Stable per physical device: derived from the Vulkan device UUID when the driver reports one. */
    @Override
    public UUID getUUID() {
        return uuid;
    }

    @Override
    public String getDeviceName() {
        return info.name();
    }

    /** Cross-API identity of this device. */
    public DeviceIdentity getIdentity() {
        return info.identity();
    }

    /** The physical device snapshot. */
    public VkPhysicalDeviceInfo getInfo() {
        return info;
    }

    /**
     * The backend on this device, created on first call and re-created after it was closed.
     *
     * @throws IllegalStateException when this device has been released
     */
    public synchronized VkComputeBackend getBackend() {
        if (released) {
            throw new IllegalStateException("Vulkan device '" + info.name() + "' has been released");
        }
        if (backend == null || backend.isClosed()) {
            backend = new VkComputeBackend(instanceManager, info);
        }
        return backend;
    }

    @Override
    public void queryCapabilities() {
        if (!LOGGER.isDebugEnabled()) {
            return;
        }
        LOGGER.debug("");
        LOGGER.debug("Vulkan Device Detected");
        LOGGER.debug("=====================");
        LOGGER.debug("Device UUID: {}", uuid);
        LOGGER.debug("Device Name: {}", info.name());
        LOGGER.debug("Device Class: {}", info.deviceClass());
        LOGGER.debug("API Version: {}", info.apiVersionString());
        LOGGER.debug("Driver: {} {}", info.driverName(), info.driverInfo());
        LOGGER.debug("Device Memory: {} MB", info.totalDeviceLocalBytes() / (1024 * 1024));
        LOGGER.debug("Shader Cores: {}", info.shaderCoreCount());
        LOGGER.debug("Max Work Group Invocations: {}", info.maxComputeWorkGroupInvocations());
        LOGGER.debug("Subgroup Size: {} ({}..{})", info.subgroupSize(), info.minSubgroupSize(), info.maxSubgroupSize());
        LOGGER.debug("Compute Queue Family: {}, Transfer Queue Family: {}", info.computeQueueFamily(),
                info.transferQueueFamily());
        LOGGER.debug("Identity: {}", info.identity());
        LOGGER.debug("");
    }

    private static UUID uuidOf(VkPhysicalDeviceInfo info) {
        String hex = info.identity().uuid();
        if (hex == null) {
            return UUID.randomUUID();
        }
        return new UUID(Long.parseUnsignedLong(hex.substring(0, 16), 16), Long.parseUnsignedLong(hex.substring(16), 16));
    }
}
