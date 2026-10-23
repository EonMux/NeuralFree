package com.eonmux.neuralfree.acceleration.cuda;

import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.datatype.structures.DataMap;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdevice_attribute;
import jcuda.driver.CUuuid;

import java.util.List;
import java.util.UUID;

import static jcuda.driver.CUdevice_attribute.*;
import static jcuda.driver.JCudaDriver.*;

/**
 * One CUDA device found during enumeration.
 *
 * <p>Enumeration creates no context. Backends run on the device's primary context
 * ({@code CudaContext}); {@link #getDeviceContext()} retains that same primary
 * context on first use for callers that need a raw handle.</p>
 */
public class CUDevice implements HardwareDevice {

    /** PCI vendor id of NVIDIA, the only vendor CUDA devices come from. */
    public static final int NVIDIA_VENDOR_ID = 0x10DE;

    private static final int PCI_BUS_ID_CHARS = 64;

    UUID uuid;
    String deviceName;
    CUdevice deviceId;
    DataMap<String, Object> deviceInfo;

    /** Context handed over by a caller that created it, destroyed by {@link #release()}; null otherwise. */
    private CUcontext adoptedContext;
    /** Primary context retained by {@link #getDeviceContext()}, released by {@link #release()}. */
    private CUcontext retainedPrimary;
    private volatile DeviceIdentity identity;

    /**
     * @param device the driver's device handle; the driver must be initialised
     * @throws IllegalArgumentException when {@code device} is null
     */
    public CUDevice(CUdevice device) {
        if (device == null) {
            throw new IllegalArgumentException("CUdevice must not be null");
        }
        this.uuid = UUID.randomUUID();
        this.deviceId = device;

        byte[] name = new byte[1024];
        cuDeviceGetName(name, name.length, this.deviceId);
        this.deviceName = CUDA_Utils.createString(name);

        this.deviceInfo = new DataMap<>();
    }

    /**
     * Adopts a context the caller created for this device: {@link #getDeviceContext()}
     * returns it and {@link #release()} destroys it. Backends do not use it; they run on
     * the primary context.
     *
     * @param device        the driver's device handle
     * @param createdContext a context created by the caller with {@code cuCtxCreate}; may be null
     */
    public CUDevice(CUdevice device, CUcontext createdContext) {
        this(device);
        this.adoptedContext = createdContext;
    }

    @Override
    public void initialize() {

    }

    /**
     * Releases what this object holds: the primary-context retain taken by
     * {@link #getDeviceContext()}, and an adopted context. Idempotent.
     */
    @Override
    public synchronized void release() {
        if (retainedPrimary != null) {
            retainedPrimary = null;
            cuDevicePrimaryCtxRelease(this.deviceId);
        }
        if (adoptedContext != null) {
            CUcontext ctx = adoptedContext;
            adoptedContext = null;
            cuCtxDestroy(ctx);
        }
    }

    @Override
    public UUID getUUID() {
        return this.uuid;
    }

    @Override
    public String getDeviceName() {
        return this.deviceName;
    }

    public CUdevice getDeviceId() {
        return deviceId;
    }

    /**
     * The adopted context when one was handed over, otherwise the device's primary
     * context, retained on first call and released by {@link #release()}.
     */
    public synchronized CUcontext getDeviceContext() {
        if (adoptedContext != null) {
            return adoptedContext;
        }
        if (retainedPrimary == null) {
            CUcontext ctx = new CUcontext();
            cuDevicePrimaryCtxRetain(ctx, this.deviceId);
            retainedPrimary = ctx;
        }
        return retainedPrimary;
    }

    /**
     * Cross-API identity: name, NVIDIA vendor id, PCI bus id ({@code cuDeviceGetPCIBusId})
     * and device UUID ({@code cuDeviceGetUuid}). A field the driver will not report is
     * left unknown. Computed once; needs no context.
     */
    public DeviceIdentity getIdentity() {
        DeviceIdentity id = identity;
        if (id == null) {
            id = new DeviceIdentity(deviceName == null ? "" : deviceName, NVIDIA_VENDOR_ID, queryPciBusId(),
                    queryUuid());
            identity = id;
        }
        return id;
    }

    /** {@link DeviceClass#INTEGRATED} or {@link DeviceClass#DISCRETE}; {@link DeviceClass#UNKNOWN} when the query fails. */
    public DeviceClass getDeviceClass() {
        try {
            return getAttribute(CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0 ? DeviceClass.INTEGRATED : DeviceClass.DISCRETE;
        } catch (RuntimeException e) {
            LoggingUtil.getLogger(this).debug("CU_DEVICE_ATTRIBUTE_INTEGRATED query failed for '{}': {}",
                    deviceName, e.toString());
            return DeviceClass.UNKNOWN;
        }
    }

    /** Compute capability as {@code major * 10 + minor}, for example 120 for 12.0. */
    public int getComputeCapability() {
        return getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10
                + getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    }

    /** One integer device attribute ({@code cuDeviceGetAttribute}); needs no context. */
    public int getAttribute(int attribute) {
        int[] value = new int[1];
        int result = cuDeviceGetAttribute(value, attribute, this.deviceId);
        if (result != 0) {
            throw new IllegalStateException("cuDeviceGetAttribute(" + CUdevice_attribute.stringFor(attribute)
                    + ") failed for '" + deviceName + "' with CUDA error " + result);
        }
        return value[0];
    }

    @Override
    public void queryCapabilities() {
        // Obtain the compute capability
        int[] majorArray = {0};
        int[] minorArray = {0};
        cuDeviceGetAttribute(majorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, this.deviceId);
        cuDeviceGetAttribute(minorArray, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, this.deviceId);
        int major = majorArray[0];
        int minor = minorArray[0];

        LoggingUtil.getLogger(this).debug("");
        LoggingUtil.getLogger(this).debug("CUDA Device Detected");
        LoggingUtil.getLogger(this).debug("====================");

        LoggingUtil.getLogger(this)
                .debug("Device " + this.uuid.toString() + ": " + this.deviceName + " with Compute Capability " + major + "." + minor);

        // Obtain the device attributes
        int[] array = {0};
        List<Integer> attributes = CUDA_Utils.getAttributes();
        for (Integer attribute : attributes) {
            String description = CUDA_Utils.getAttributeDescription(attribute);
            cuDeviceGetAttribute(array, attribute, this.deviceId);
            int value = array[0];
            try {
                this.deviceInfo.put(CUdevice_attribute.stringFor(attribute), value);
                LoggingUtil.getLogger(this).debug(String.format("%-52s : %d", description, value));
            } catch (NullPointerException e) {
                LoggingUtil.getLogger(this).debug("NullPointerException occured with CUDA device: " + this.uuid.toString() + " with attribute: " + attribute);
            }
        }
        LoggingUtil.getLogger(this).debug("");
    }

    private String queryPciBusId() {
        try {
            String[] busId = new String[1];
            int result = cuDeviceGetPCIBusId(busId, PCI_BUS_ID_CHARS, this.deviceId);
            if (result != 0 || busId[0] == null || busId[0].trim().isEmpty()) {
                return null;
            }
            // Validate through DeviceIdentity's parser so a malformed answer counts as unknown.
            new DeviceIdentity(deviceName == null ? "" : deviceName, NVIDIA_VENDOR_ID, busId[0], null);
            return busId[0].trim();
        } catch (RuntimeException e) {
            LoggingUtil.getLogger(this).debug("cuDeviceGetPCIBusId failed for '{}': {}", deviceName, e.toString());
            return null;
        }
    }

    private String queryUuid() {
        try {
            CUuuid id = new CUuuid();
            int result = cuDeviceGetUuid(id, this.deviceId);
            if (result != 0 || id.bytes == null || id.bytes.length != 16) {
                return null;
            }
            return DeviceIdentity.hexUuid(id.bytes);
        } catch (RuntimeException e) {
            LoggingUtil.getLogger(this).debug("cuDeviceGetUuid failed for '{}': {}", deviceName, e.toString());
            return null;
        }
    }
}
