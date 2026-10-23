package com.eonmux.neuralfree.acceleration.ocl;

import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.datatype.structures.DataMap;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.jocl.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.UUID;

import static org.jocl.CL.*;

/**
 * One OpenCL device found during enumeration, with its device and platform info queries.
 *
 * <p>The device-info helpers ({@code getInt}, {@code getInts}, {@code getLong}, {@code getLongs}, {@code getString},
 * {@code getSize}, {@code getSizes}) are derived from the JOCLDeviceQuery sample of JOCL (MIT License,
 * Copyright (c) 2009-2019 Marco Hutter); see THIRD-PARTY-NOTICES.md.</p>
 */
public class CLDevice implements HardwareDevice {
    enum DEVICE_TYPE {
        CL_DEVICE_TYPE_CPU,
        CL_DEVICE_TYPE_GPU,
        CL_DEVICE_TYPE_ACCELERATOR,
        CL_DEVICE_TYPE_DEFAULT
    }

    enum MEMORY_TYPE {
        LOCAL,
        GLOBAL
    }

    enum PROPERTIES {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        CL_QUEUE_PROFILING_ENABLE
    }

    // clGetDeviceInfo parameters of OpenCL extensions (Khronos cl_ext.h)
    static final int CL_DEVICE_UUID_KHR = 0x106A;
    static final int CL_UUID_SIZE_KHR = 16;
    static final int CL_DEVICE_PCI_BUS_INFO_KHR = 0x410F;
    static final int CL_DEVICE_PCI_BUS_ID_NV = 0x4008;
    static final int CL_DEVICE_PCI_SLOT_ID_NV = 0x4009;
    static final int CL_DEVICE_PCI_DOMAIN_ID_NV = 0x400A;
    static final int CL_DEVICE_TOPOLOGY_AMD = 0x4037;
    static final int CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD = 1;

    cl_device_id deviceID;
    cl_context deviceContext;
    cl_command_queue deviceCommandQueue;
    DataMap<String, Object> deviceInfo;

    public CLDevice(cl_device_id deviceID, cl_context context, cl_command_queue deviceCommandQueue) {
        this.deviceInfo = new DataMap<>();
        this.deviceID = deviceID;
        this.deviceContext = context;
        this.deviceCommandQueue = deviceCommandQueue;
        queryCapabilities();
    }

    @Override
    public void initialize() {

    }

    @Override
    public void release() {
        //Release the command queue
        clReleaseCommandQueue(this.deviceCommandQueue);
        //Then the device context
        clReleaseContext(this.deviceContext);
    }

    @Override
    public UUID getUUID() {
        return null;
    }

    @Override
    public String getDeviceName() {
        return null;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param paramName The parameter name
     * @return The value
     */
    int getInt(int paramName) {
        return getInts(paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    int[] getInts(int paramName, int numValues) {
        int[] values = new int[numValues];
        clGetDeviceInfo(this.deviceID, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param paramName The parameter name
     * @return The value
     */
    long getLong(int paramName) {
        return getLongs(paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    long[] getLongs(int paramName, int numValues) {
        long[] values = new long[numValues];
        clGetDeviceInfo(this.deviceID, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param paramName The parameter name
     * @return The value
     */
    String getString(int paramName) {
        long[] size = new long[1];
        clGetDeviceInfo(this.deviceID, paramName, 0, null, size);

        if (size[0] <= 0 || size[0] > 65536) {
            return "";
        }

        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(this.deviceID, paramName, buffer.length, Pointer.to(buffer), null);

        return new String(buffer, 0, buffer.length - 1);
    }

    /**
     * Returns the value of the platform info parameter with the given name
     *
     * @param platform  The platform
     * @param paramName The parameter name
     * @return The value
     */
    String getString(cl_platform_id platform, int paramName) {
        // Obtain the length of the string that will be queried
        long[] size = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);

        // Handle empty or missing string parameters
        if (size[0] <= 0) {
            return "";
        }

        // Create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int) size[0]];
        clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length - 1);
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param paramName The parameter name
     * @return The value
     */
    long getSize(int paramName) {
        return getSizes(paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    long[] getSizes(int paramName, int numValues) {
        // The size of the returned data has to depend on
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
                numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(this.deviceID, paramName, Sizeof.size_t * numValues,
                Pointer.to(buffer), null);
        long[] values = new long[numValues];
        if (Sizeof.size_t == 4) {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        } else {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }

    @Override
    public void queryCapabilities() {

        // CL_DEVICE_NAME
        String deviceName = this.getString(CL_DEVICE_NAME);
        this.deviceInfo.put("CL_DEVICE_NAME", deviceName);
        LoggingUtil.getLogger(this).debug("");
        LoggingUtil.getLogger(this).debug("OpenCL Device Detected");
        LoggingUtil.getLogger(this).debug("======================");
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_NAME: \t\t\t\t\t: %s", deviceName));

        // CL_DEVICE_VENDOR
        String deviceVendor = this.getString(CL_DEVICE_VENDOR);
        this.deviceInfo.put("CL_DEVICE_VENDOR",deviceVendor);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_VENDOR: \t\t\t\t\t: %s", deviceVendor));

        // CL_DRIVER_VERSION
        String deviceDriverVersion = this.getString(CL_DRIVER_VERSION);
        this.deviceInfo.put("CL_DRIVE_VERSION", deviceDriverVersion);
        LoggingUtil.getLogger(this).debug(String.format("CL_DRIVER_VERSION: \t\t\t\t\t: %s", deviceDriverVersion));

        // CL_DEVICE_TYPE
        long deviceType = this.getLong(CL_DEVICE_TYPE);
        if ((deviceType & CL_DEVICE_TYPE_CPU) != 0) {
            this.deviceInfo.put("CL_DEVICE_TYPE", DEVICE_TYPE.CL_DEVICE_TYPE_CPU);
            LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_TYPE:\t\t\t\t\t: %s", "CL_DEVICE_TYPE_CPU"));
        }
        if ((deviceType & CL_DEVICE_TYPE_GPU) != 0) {
            this.deviceInfo.put("CL_DEVICE_TYPE", DEVICE_TYPE.CL_DEVICE_TYPE_GPU);
            LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_TYPE:\t\t\t\t\t: %s", "CL_DEVICE_TYPE_GPU"));
        }
        if ((deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0) {
            this.deviceInfo.put("CL_DEVICE_TYPE", DEVICE_TYPE.CL_DEVICE_TYPE_ACCELERATOR);
            LoggingUtil.getLogger(this)
                    .debug(String.format("CL_DEVICE_TYPE:\t\t\t\t\tt%s", "CL_DEVICE_TYPE_ACCELERATOR"));
        }
        if ((deviceType & CL_DEVICE_TYPE_DEFAULT) != 0) {
            this.deviceInfo.put("CL_DEVICE_TYPE", DEVICE_TYPE.CL_DEVICE_TYPE_DEFAULT);
            LoggingUtil.getLogger(this)
                    .debug(String.format("CL_DEVICE_TYPE:\t\t\t\t\t\t: %s", "CL_DEVICE_TYPE_DEFAULT"));
        }

        // CL_DEVICE_HOST_UNIFIED_MEMORY
        //
        // Whether the device shares the host's address space, i.e. whether it is
        // an integrated GPU. This is the OpenCL counterpart of the
        // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU test VkComputeBackend already
        // makes, and it is the only attribute here that reliably separates an
        // iGPU from a discrete card: an integrated part can report MORE compute
        // units, a HIGHER clock and far more "global memory" (it counts system
        // RAM) than a discrete GPU an order of magnitude faster than it.
        //
        // Deprecated in OpenCL 2.0, so a conforming 2.x-only ICD may refuse the
        // query. That is not a reason to lose the device: leave the key absent
        // and let callers treat it as unknown.
        try {
            this.deviceInfo.put("CL_DEVICE_HOST_UNIFIED_MEMORY",
                    this.getInt(CL_DEVICE_HOST_UNIFIED_MEMORY) != 0);
        } catch (Throwable t) {
            LoggingUtil.getLogger(this)
                    .debug("CL_DEVICE_HOST_UNIFIED_MEMORY unavailable: {}", t.toString());
        }

        // CL_DEVICE_VENDOR_ID
        //
        // The PCI vendor id (0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel), or a
        // Khronos-assigned id above 0xffff for non-PCI devices. Part of the
        // cross-API identity returned by getIdentity().
        try {
            this.deviceInfo.put("CL_DEVICE_VENDOR_ID", this.queryUInt(CL_DEVICE_VENDOR_ID));
        } catch (Throwable t) {
            LoggingUtil.getLogger(this)
                    .debug("CL_DEVICE_VENDOR_ID unavailable: {}", t.toString());
        }

        // CL_DEVICE_EXTENSIONS
        //
        // Needed below to decide which vendor queries the driver understands.
        String deviceExtensions = "";
        try {
            deviceExtensions = this.getString(CL_DEVICE_EXTENSIONS);
            this.deviceInfo.put("CL_DEVICE_EXTENSIONS", deviceExtensions);
        } catch (Throwable t) {
            LoggingUtil.getLogger(this)
                    .debug("CL_DEVICE_EXTENSIONS unavailable: {}", t.toString());
        }

        // CL_DEVICE_PCI_BUS_ID
        //
        // The device's PCI address as "dddd:bb:dd.f". The same GPU is usually
        // visible through OpenCL, CUDA and Vulkan at once, and the bus id is
        // what lets backend selection recognise those views as one device.
        // OpenCL has no core query for it, so the sources are tried in order:
        // cl_khr_pci_bus_info, the NVIDIA attribute query, the AMD topology
        // query. A device none of them describes leaves the key absent.
        try {
            String pciBusId = this.queryPciBusId(deviceExtensions);
            if (pciBusId != null) {
                this.deviceInfo.put("CL_DEVICE_PCI_BUS_ID", pciBusId);
                LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_PCI_BUS_ID:\t\t\t\t\t: %s", pciBusId));
            }
        } catch (Throwable t) {
            LoggingUtil.getLogger(this)
                    .debug("CL_DEVICE_PCI_BUS_ID unavailable: {}", t.toString());
        }

        // CL_DEVICE_UUID_KHR
        //
        // The 16-byte device UUID of cl_khr_device_uuid, as 32 hex characters.
        // It matches the UUID Vulkan and CUDA report for the same device, so it
        // identifies the device when no PCI bus id is available.
        try {
            if (deviceExtensions.contains("cl_khr_device_uuid")) {
                byte[] uuid = new byte[CL_UUID_SIZE_KHR];
                this.queryDeviceInfo(CL_DEVICE_UUID_KHR, uuid.length, Pointer.to(uuid));
                this.deviceInfo.put("CL_DEVICE_UUID_KHR", DeviceIdentity.hexUuid(uuid));
            }
        } catch (Throwable t) {
            LoggingUtil.getLogger(this)
                    .debug("CL_DEVICE_UUID_KHR unavailable: {}", t.toString());
        }

        // CL_DEVICE_MAX_COMPUTE_UNITS
        int deviceMaxComputeUnits = this.getInt(CL_DEVICE_MAX_COMPUTE_UNITS);
        this.deviceInfo.put("CL_DEVICE_MAX_COMPUTE_UNITS", deviceMaxComputeUnits);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_MAX_COMPUTE_UNITS:\t\t\t\t: %d", deviceMaxComputeUnits));

        // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
        long deviceMaxWorkItemDimensions = this.getLong(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        this.deviceInfo.put("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", deviceMaxWorkItemDimensions);
        LoggingUtil.getLogger(this)
                .debug(String.format("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t\t\t: %d", deviceMaxWorkItemDimensions));

        // CL_DEVICE_MAX_WORK_ITEM_SIZES
        long[] deviceMaxWorkItemSizes = this.getSizes(CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
        this.deviceInfo.put("CL_DEVICE_MAX_WORK_ITEM_SIZES", deviceMaxWorkItemSizes);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_MAX_WORK_ITEM_SIZES:\t\t\t: %d / %d / %d ",
                deviceMaxWorkItemSizes[0], deviceMaxWorkItemSizes[1], deviceMaxWorkItemSizes[2]));

        // CL_DEVICE_MAX_WORK_GROUP_SIZE
        long deviceMaxWorkGroupSize = this.getSize(CL_DEVICE_MAX_WORK_GROUP_SIZE);
        this.deviceInfo.put("CL_DEVICE_MAX_WORK_GROUP_SIZE", deviceMaxWorkGroupSize);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t\t: %d", deviceMaxWorkGroupSize));

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        long deviceMaxClockFrequency = this.getLong(CL_DEVICE_MAX_CLOCK_FREQUENCY);
        this.deviceInfo.put("CL_DEVICE_MAX_CLOCK_FREQUENCY", deviceMaxClockFrequency);
        LoggingUtil.getLogger(this)
                .debug(String.format("CL_DEVICE_MAX_CLOCK_FREQUENCY:\t\t\t: %d MHz", deviceMaxClockFrequency));

        // CL_DEVICE_ADDRESS_BITS
        int deviceAddressBits = this.getInt(CL_DEVICE_ADDRESS_BITS);
        this.deviceInfo.put("CL_DEVICE_ADDRESS_BITS", deviceAddressBits);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_ADDRESS_BITS:\t\t\t\t: %d", deviceAddressBits));

        // CL_DEVICE_MAX_MEM_ALLOC_SIZE
        long deviceMaxMemAllocSize = this.getLong(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        this.deviceInfo.put("CL_DEVICE_MAX_MEM_ALLOC_SIZE", deviceMaxMemAllocSize);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t\t: %d MByte",
                (int) (deviceMaxMemAllocSize / (1024 * 1024))));

        // CL_DEVICE_GLOBAL_MEM_SIZE
        long deviceGlobalMemSize = this.getLong(CL_DEVICE_GLOBAL_MEM_SIZE);
        this.deviceInfo.put("CL_DEVICE_GLOBAL_MEM_SIZE", deviceGlobalMemSize);
        LoggingUtil.getLogger(this).debug(
                String.format("CL_DEVICE_GLOBAL_MEM_SIZE:\t\t\t\t: %d MByte", (int) (deviceGlobalMemSize / (1024 * 1024))));

        // CL_DEVICE_ERROR_CORRECTION_SUPPORT
        int errorCorrectionSupport = this.getInt(CL_DEVICE_ERROR_CORRECTION_SUPPORT);
        boolean deviceErrorCorrectionSupport = errorCorrectionSupport != 0;
        this.deviceInfo.put("CL_DEVICE_ERROR_CORRECTION_SUPPORT", deviceErrorCorrectionSupport);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t\t\t: %s",
                errorCorrectionSupport != 0 ? "yes" : "no"));

        // CL_DEVICE_LOCAL_MEM_TYPE
        long localMemType = this.getInt(CL_DEVICE_LOCAL_MEM_TYPE);
        MEMORY_TYPE deviceMemoryType = localMemType == 1 ? MEMORY_TYPE.LOCAL : MEMORY_TYPE.GLOBAL;
        this.deviceInfo.put("CL_DEVICE_LOCAL_MEM_TYPE", deviceMemoryType);
        LoggingUtil.getLogger(this).debug(
                String.format("CL_DEVICE_LOCAL_MEM_TYPE:\t\t\t\t: %s", localMemType == 1 ? "local" : "global"));

        // CL_DEVICE_LOCAL_MEM_SIZE
        long deviceLocalMemSize = this.getLong(CL_DEVICE_LOCAL_MEM_SIZE);
        this.deviceInfo.put("CL_DEVICE_LOCAL_MEM_SIZE", deviceLocalMemSize);
        LoggingUtil.getLogger(this)
                .debug(String.format("CL_DEVICE_LOCAL_MEM_SIZE:\t\t\t\t: %d KByte", (int) (deviceLocalMemSize / 1024)));

        // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
        long deviceMaxConstantBufferSize = this.getLong(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
        this.deviceInfo.put("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE", deviceMaxConstantBufferSize);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t\t\t: %d KByte",
                (int) (deviceMaxConstantBufferSize / 1024)));

        // cl_command_queue_properties - bitfield
        // CL_DEVICE_QUEUE_ON_HOST_PROPERTIES
        long queueProperties = this.getLong(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES);
        if ((queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0) {
            PROPERTIES deviceQueueOnHostProperties = PROPERTIES.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            this.deviceInfo.put("CL_DEVICE_QUEUE_ON_HOST_PROPERTIES", deviceQueueOnHostProperties);
            LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_QUEUE_ON_HOST_PROPERTIES:\t\t\t\t: %s",
                    "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE"));
        }
        if ((queueProperties & CL_QUEUE_PROFILING_ENABLE) != 0) {
            PROPERTIES deviceQueueOnHostProperties = PROPERTIES.CL_QUEUE_PROFILING_ENABLE;
            this.deviceInfo.put("CL_DEVICE_QUEUE_ON_HOST_PROPERTIES", deviceQueueOnHostProperties);
            LoggingUtil.getLogger(this)
                    .debug(String.format("CL_DEVICE_QUEUE_ON_HOST_PROPERTIES:\t\t\t\t: %s", "CL_QUEUE_PROFILING_ENABLE"));
        }

        // CL_DEVICE_IMAGE_SUPPORT
        boolean deviceImageSupport = this.getInt(CL_DEVICE_IMAGE_SUPPORT) == 1;
        this.deviceInfo.put("CL_DEVICE_IMAGE_SUPPORT", deviceImageSupport);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_IMAGE_SUPPORT:\t\t\t\t: %d", deviceImageSupport ? 1 : 0));

        // CL_DEVICE_MAX_READ_IMAGE_ARGS
        int deviceMaxReadImageArgs = this.getInt(CL_DEVICE_MAX_READ_IMAGE_ARGS);
        this.deviceInfo.put("CL_DEVICE_MAX_READ_IMAGE_ARGS", deviceMaxReadImageArgs);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_MAX_READ_IMAGE_ARGS:\t\t\t: %d", deviceMaxReadImageArgs));

        // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
        int deviceMaxWriteImageArgs = this.getInt(CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
        this.deviceInfo.put("CL_DEVICE_MAX_WRITE_IMAGE_ARGS", deviceMaxWriteImageArgs);
        LoggingUtil.getLogger(this)
                .debug(String.format("CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t\t\t: %d", deviceMaxWriteImageArgs));

        // CL_DEVICE_SINGLE_FP_CONFIG
        long deviceSingleFpConfig = this.getLong(CL_DEVICE_SINGLE_FP_CONFIG);
        this.deviceInfo.put("CL_DEVICE_SINGLE_FP_CONFIG", deviceSingleFpConfig);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_SINGLE_FP_CONFIG:\t\t\t\t: %s",
                stringFor_cl_device_fp_config(deviceSingleFpConfig)));

        // CL_DEVICE_IMAGE2D_MAX_WIDTH
        long deviceImage2dMaxWidth = this.getSize(CL_DEVICE_IMAGE2D_MAX_WIDTH);
        this.deviceInfo.put("CL_DEVICE_IMAGE2D_MAX_WIDTH", deviceImage2dMaxWidth);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_2D_MAX_WIDTH\t\t\t\t: %d", deviceImage2dMaxWidth));

        // CL_DEVICE_IMAGE2D_MAX_HEIGHT
        long deviceImage2dMaxHeight = this.getSize(CL_DEVICE_IMAGE2D_MAX_HEIGHT);
        this.deviceInfo.put("CL_DEVICE_IMAGE2D_MAX_HEIGHT", deviceImage2dMaxHeight);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_2D_MAX_HEIGHT\t\t\t\t: %d", deviceImage2dMaxHeight));

        // CL_DEVICE_IMAGE3D_MAX_WIDTH
        long deviceImage3dMaxWidth = this.getSize(CL_DEVICE_IMAGE3D_MAX_WIDTH);
        this.deviceInfo.put("CL_DEVICE_IMAGE3D_MAX_WIDTH", deviceImage3dMaxWidth);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_3D_MAX_WIDTH\t\t\t\t: %d", deviceImage3dMaxWidth));

        // CL_DEVICE_IMAGE3D_MAX_HEIGHT
        long deviceImage3dMaxHeight = this.getSize(CL_DEVICE_IMAGE3D_MAX_HEIGHT);
        this.deviceInfo.put("CL_DEVICE_IMAGE3D_MAX_HEIGHT", deviceImage3dMaxHeight);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_3D_MAX_HEIGHT\t\t\t\t: %d", deviceImage3dMaxHeight));

        // CL_DEVICE_IMAGE3D_MAX_DEPTH
        long deviceImage3dMaxDepth = this.getSize(CL_DEVICE_IMAGE3D_MAX_DEPTH);
        this.deviceInfo.put("CL_DEVICE_IMAGE3D_MAX_DEPTH", deviceImage3dMaxDepth);
        LoggingUtil.getLogger(this).debug(String.format("CL_DEVICE_3D_MAX_DEPTH\t\t\t\t: %d", deviceImage3dMaxDepth));

        // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
        int devicePreferredVectorWidthChar = this.getInt(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
        this.deviceInfo.put("CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", devicePreferredVectorWidthChar);
        int devicePreferredVectorWidthShort = this.getInt(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
        this.deviceInfo.put("CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", devicePreferredVectorWidthShort);
        int devicePreferredVectorWidthInt = this.getInt(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
        this.deviceInfo.put("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", devicePreferredVectorWidthInt);
        int devicePreferredVectorWidthLong = this.getInt(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
        this.deviceInfo.put("CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG", devicePreferredVectorWidthLong);
        int devicePreferredVectorWidthFloat = this.getInt(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
        this.deviceInfo.put("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", devicePreferredVectorWidthFloat);
        int devicePreferredVectorWidthDouble = this.getInt(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
        this.deviceInfo.put("CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE", devicePreferredVectorWidthDouble);
        LoggingUtil.getLogger(this).debug(String.format(
                "CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t\t\t: CHAR %d, SHORT %d, INT %d, LONG %d, FLOAT %d, DOUBLE %d\n",
                devicePreferredVectorWidthChar, devicePreferredVectorWidthShort, devicePreferredVectorWidthInt,
                devicePreferredVectorWidthLong, devicePreferredVectorWidthFloat, devicePreferredVectorWidthDouble));
    }

    public cl_device_id getDeviceID() {
        return deviceID;
    }

    public cl_context getDeviceContext() {
        return deviceContext;
    }

    public cl_command_queue getDeviceCommandQueue() {
        return deviceCommandQueue;
    }

    public DataMap<String, Object> getDeviceInfo() {
        return deviceInfo;
    }

    /**
     * Cross-API identity of this device: name, vendor id, PCI bus id and UUID,
     * each as far as the driver reports it.
     *
     * @return the identity; never null
     */
    public DeviceIdentity getIdentity() {
        Object name = this.deviceInfo.get("CL_DEVICE_NAME");
        Object vendorId = this.deviceInfo.get("CL_DEVICE_VENDOR_ID");
        Object pciBusId = this.deviceInfo.get("CL_DEVICE_PCI_BUS_ID");
        Object uuid = this.deviceInfo.get("CL_DEVICE_UUID_KHR");
        return new DeviceIdentity(
                name instanceof String ? (String) name : "",
                vendorId instanceof Integer && (Integer) vendorId >= 0 ? (Integer) vendorId : DeviceIdentity.UNKNOWN_VENDOR,
                pciBusId instanceof String ? (String) pciBusId : null,
                uuid instanceof String ? (String) uuid : null);
    }

    /**
     * Returns the PCI address of this device as "dddd:bb:dd.f", or null when no
     * supported extension reports it. A source whose query fails is skipped.
     *
     * @param extensions the device's CL_DEVICE_EXTENSIONS string
     * @return the PCI bus id, or null
     */
    String queryPciBusId(String extensions) {
        // cl_khr_pci_bus_info: struct { cl_uint domain, bus, device, function; }
        if (extensions.contains("cl_khr_pci_bus_info")) {
            try {
                int[] info = new int[4];
                this.queryDeviceInfo(CL_DEVICE_PCI_BUS_INFO_KHR, Sizeof.cl_uint * 4, Pointer.to(info));
                return DeviceIdentity.formatPciBusId(info[0], info[1], info[2], info[3]);
            } catch (Throwable t) {
                LoggingUtil.getLogger(this).debug("CL_DEVICE_PCI_BUS_INFO_KHR unavailable: {}", t.toString());
            }
        }
        // cl_nv_device_attribute_query: bus, slot = device << 3 | function, and the
        // domain on drivers that report it (older ones do not; it is then 0).
        if (extensions.contains("cl_nv_device_attribute_query")) {
            try {
                int bus = this.queryUInt(CL_DEVICE_PCI_BUS_ID_NV);
                int slot = this.queryUInt(CL_DEVICE_PCI_SLOT_ID_NV);
                int domain = 0;
                try {
                    domain = this.queryUInt(CL_DEVICE_PCI_DOMAIN_ID_NV);
                } catch (Throwable t) {
                    LoggingUtil.getLogger(this).debug("CL_DEVICE_PCI_DOMAIN_ID_NV unavailable: {}", t.toString());
                }
                return DeviceIdentity.formatPciBusId(domain, bus & 0xFF, (slot >>> 3) & 0x1F, slot & 0x7);
            } catch (Throwable t) {
                LoggingUtil.getLogger(this).debug("CL_DEVICE_PCI_BUS_ID_NV unavailable: {}", t.toString());
            }
        }
        // cl_amd_device_attribute_query: cl_device_topology_amd, a 24-byte union whose
        // pcie view is { cl_uint type; cl_char unused[17]; cl_char bus, device, function; }.
        // The domain is not reported and taken as 0.
        if (extensions.contains("cl_amd_device_attribute_query")) {
            try {
                byte[] topology = new byte[24];
                this.queryDeviceInfo(CL_DEVICE_TOPOLOGY_AMD, topology.length, Pointer.to(topology));
                int type = ByteBuffer.wrap(topology).order(ByteOrder.nativeOrder()).getInt(0);
                if (type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
                    return DeviceIdentity.formatPciBusId(0,
                            topology[21] & 0xFF, topology[22] & 0xFF, topology[23] & 0xFF);
                }
            } catch (Throwable t) {
                LoggingUtil.getLogger(this).debug("CL_DEVICE_TOPOLOGY_AMD unavailable: {}", t.toString());
            }
        }
        return null;
    }

    /**
     * Reads a cl_uint device info parameter.
     *
     * @throws IllegalStateException when the driver rejects the query
     */
    int queryUInt(int paramName) {
        int[] value = new int[1];
        this.queryDeviceInfo(paramName, Sizeof.cl_uint, Pointer.to(value));
        return value[0];
    }

    /**
     * Reads a device info parameter into {@code target}. Fails whether or not
     * JOCL exceptions are enabled, so a rejected query never yields zeros.
     *
     * @throws IllegalStateException when the driver rejects the query
     */
    void queryDeviceInfo(int paramName, long size, Pointer target) {
        int status = clGetDeviceInfo(this.deviceID, paramName, size, target, null);
        if (status != CL_SUCCESS) {
            throw new IllegalStateException("clGetDeviceInfo(0x" + Integer.toHexString(paramName)
                    + ") failed: " + CL.stringFor_errorCode(status));
        }
    }

    /**
     * Create a direct FloatBuffer with the given size and the
     * native byte order
     *
     * @param n The size
     * @return The FloatBuffer
     */
    public static FloatBuffer createFloatBuffer(int n) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(n * Sizeof.cl_float);
        return byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer();
    }

    /**
     * Returns a string containing the first entries of the
     * given buffer.
     *
     * @param b   The buffer
     * @param max The maximum number of elements to show
     * @return The string
     */
    public static String stringFor(FloatBuffer b, int max) {
        StringBuilder sb = new StringBuilder();
        max = Math.min(max, b.capacity());
        for (int i = 0; i < max; i++) {
            sb.append(b.get(i));
            if (i < max - 1) {
                sb.append(", ");
            } else if (b.capacity() > max) {
                sb.append(" ...");
            }
        }
        return sb.toString();
    }
}
