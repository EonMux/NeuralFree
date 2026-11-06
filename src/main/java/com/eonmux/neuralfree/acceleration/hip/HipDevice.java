package com.eonmux.neuralfree.acceleration.hip;

import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.utils.LoggingUtil;

import java.util.UUID;

/**
 * One HIP device found during enumeration, the AMD counterpart of a {@code CUDevice}.
 *
 * <p>Enumeration creates no state on the device: the HIP runtime gives every device an implicit
 * context that a backend selects per thread with {@code hipSetDevice} ({@code HipContext}).
 * {@link #getComputeCapability()} reports the target on llama.cpp's AMD compute-capability line
 * ({@code 0x1000000 | gfx}), which is what {@code DeviceCapabilities} carries for this backend.
 * Every attribute query goes through the {@link HipDriver}, so tests run against their own.</p>
 */
public class HipDevice implements HardwareDevice {

    private final UUID uuid = UUID.randomUUID();
    private final int index;
    private final HipDriver driver;
    private final String deviceName;
    private final HipGfxTarget gfxTarget;
    private volatile DeviceIdentity identity;

    /**
     * @param index  the HIP device ordinal
     * @param driver the runtime operations this device is queried through
     * @throws IllegalArgumentException when the driver is null or the device has no name
     */
    public HipDevice(int index, HipDriver driver) {
        if (driver == null) {
            throw new IllegalArgumentException("HIP driver must not be null");
        }
        String name = driver.deviceName(index);
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("HIP device " + index + " reports no name");
        }
        this.index = index;
        this.driver = driver;
        this.deviceName = name.trim();
        String arch = driver.gcnArchName(index);
        if (arch == null || arch.trim().isEmpty()) {
            throw new IllegalArgumentException("HIP device '" + this.deviceName + "' reports no gcnArchName;"
                    + " the HIP backend needs the processor name to pick its kernels");
        }
        this.gfxTarget = HipGfxTarget.parse(arch);
    }

    @Override
    public void initialize() {
        // The HIP runtime needs no per-device initialization before use.
    }

    @Override
    public synchronized void release() {
        // The runtime holds no retain on a device that this object must drop; the implicit
        // context outlives it and is torn down with the runtime, not with the enumeration.
    }

    @Override
    public UUID getUUID() {
        return uuid;
    }

    @Override
    public String getDeviceName() {
        return deviceName;
    }

    /** The HIP device ordinal. */
    public int getIndex() {
        return index;
    }

    /** The runtime operations this device is queried and driven through. */
    public HipDriver getDriver() {
        return driver;
    }

    /** The processor target this device runs ({@code gcnArchName}). */
    public HipGfxTarget getGfxTarget() {
        return gfxTarget;
    }

    /**
     * Cross-API identity: name, the AMD PCI vendor id, PCI bus id ({@code hipDeviceGetPCIBusId})
     * and device UUID ({@code hipDeviceGetUuid}). A field the driver will not report is left
     * unknown. Computed once; needs no device state.
     */
    public DeviceIdentity getIdentity() {
        DeviceIdentity id = identity;
        if (id == null) {
            byte[] uuidBytes = driver.deviceUuid(index);
            id = new DeviceIdentity(deviceName, HipGfxTarget.AMD_VENDOR_ID, pciBusId(),
                    uuidBytes == null ? null : DeviceIdentity.hexUuid(uuidBytes));
            identity = id;
        }
        return id;
    }

    /** {@link DeviceClass#INTEGRATED} for APUs, {@link DeviceClass#DISCRETE} otherwise; unknown when the query fails. */
    public DeviceClass getDeviceClass() {
        try {
            return driver.deviceAttribute(index, HipNative.ATTR_INTEGRATED) != 0
                    ? DeviceClass.INTEGRATED
                    : DeviceClass.DISCRETE;
        } catch (RuntimeException e) {
            LoggingUtil.getLogger(this).debug("hipDeviceAttribute(INTEGRATED) failed for '{}': {}",
                    deviceName, e.toString());
            return DeviceClass.UNKNOWN;
        }
    }

    /**
     * The target on llama.cpp's AMD compute-capability line: {@code 0x1000000 | gfx}, for example
     * 0x1001100 for gfx1100.
     */
    public int getComputeCapability() {
        return gfxTarget.amdComputeCapability();
    }

    /** One integer device attribute of the shim's selector table; needs no device state. */
    public int getAttribute(int selector) {
        return driver.deviceAttribute(index, selector);
    }

    @Override
    public void queryCapabilities() {
        LoggingUtil.getLogger(this).debug("");
        LoggingUtil.getLogger(this).debug("HIP Device Detected");
        LoggingUtil.getLogger(this).debug("===================");
        LoggingUtil.getLogger(this).debug("Device " + uuid + ": " + deviceName + " (" + gfxTarget
                + ", CC 0x" + Integer.toHexString(getComputeCapability()) + ")");
        LoggingUtil.getLogger(this).debug("Compute units: " + getAttribute(HipNative.ATTR_MULTIPROCESSOR_COUNT));
        LoggingUtil.getLogger(this).debug("Wavefront size: " + getAttribute(HipNative.ATTR_WARP_SIZE));
        LoggingUtil.getLogger(this).debug("Max work-group size: " + getAttribute(HipNative.ATTR_MAX_THREADS_PER_BLOCK));
        LoggingUtil.getLogger(this).debug("Shared memory per block: " + getAttribute(HipNative.ATTR_SHARED_PER_BLOCK));
        LoggingUtil.getLogger(this).debug("");
    }

    private String pciBusId() {
        try {
            return driver.pciBusId(index);
        } catch (RuntimeException e) {
            LoggingUtil.getLogger(this).debug("hipDeviceGetPCIBusId failed for '{}': {}", deviceName, e.toString());
            return null;
        }
    }

    @Override
    public String toString() {
        return "hip" + index + " '" + deviceName + "' (" + gfxTarget + ")";
    }
}
