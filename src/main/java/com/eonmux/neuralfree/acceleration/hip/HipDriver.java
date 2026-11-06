package com.eonmux.neuralfree.acceleration.hip;

import java.nio.ByteBuffer;

/**
 * The HIP runtime operations the backend, its queues, pools and events run through, so every
 * class of the HIP backend is testable without AMD hardware: production code runs
 * {@link JniHipDriver} over {@link HipNative}; tests supply their own driver.
 *
 * <p>Every method translates a failing {@code hipError_t} status into the backend's exception
 * types ({@link com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException}, {@code
 * UnsupportedKernelException} or {@code GpuBackendException}); the interface never returns raw
 * status codes. Device-scoped methods run with the calling thread's device already set to the
 * backend's device.</p>
 */
public interface HipDriver {

    /**
     * Initialises the runtime and counts the devices ({@code hipGetDeviceCount}).
     *
     * @throws RuntimeException the translated failure
     */
    int deviceCount();

    /** Makes {@code device} the calling thread's device ({@code hipSetDevice}). */
    void setDevice(int device);

    /** The marketing name of the device, or null when the driver declines. */
    String deviceName(int device);

    /** The processor name of the device, for example {@code gfx1100}, or null when the driver declines. */
    String gcnArchName(int device);

    /** Bytes of global memory of the device; -1 when the query fails. */
    long totalMem(int device);

    /** One integer attribute of the shim's selector table; the driver must answer. */
    int deviceAttribute(int device, int selector);

    /** The runtime version as {@code 1000 * major + 10 * minor} ({@code hipDriverGetVersion}). */
    int driverVersion();

    /** The 16 UUID bytes of the device, or null when the driver has none. */
    byte[] deviceUuid(int device);

    /** The PCI bus id of the device, or null when the driver declines. */
    String pciBusId(int device);

    /**
     * Allocates {@code bytes} of device memory ({@code hipMalloc}).
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when the device is out of memory
     * @throws RuntimeException                                           any other translated failure
     */
    long memAlloc(long bytes);

    /** Frees a device allocation ({@code hipFree}). */
    void memFree(long pointer);

    /** Free and total global memory of the current device ({@code hipMemGetInfo}). */
    long[] memGetInfo();

    /**
     * Allocates {@code bytes} of page-locked portable host memory ({@code hipHostMalloc}).
     *
     * @throws RuntimeException the translated failure
     */
    long hostAlloc(long bytes);

    /** Frees page-locked host memory ({@code hipHostFree}). */
    void hostFree(long pointer);

    /** A direct ByteBuffer view of {@code [pointer, pointer + bytes)}. */
    ByteBuffer pinnedView(long pointer, int bytes);

    /** Page-locked host to device, ordered on {@code stream} ({@code hipMemcpyAsync}). */
    void memcpyHtoDAsync(long dst, long srcHost, long bytes, long stream);

    /** Device to page-locked host, ordered on {@code stream} ({@code hipMemcpyAsync}). */
    void memcpyDtoHAsync(long dstHost, long src, long bytes, long stream);

    /** Device to device, ordered on {@code stream} ({@code hipMemcpyAsync}). */
    void memcpyDtoDAsync(long dst, long src, long bytes, long stream);

    /** Fills {@code count} 32-bit words with {@code value}, ordered on {@code stream}. */
    void memsetD32Async(long dst, int value, long count, long stream);

    /** Creates a non-blocking stream ({@code hipStreamCreateWithFlags}). */
    long streamCreate();

    /** Destroys a stream. */
    void streamDestroy(long stream);

    /** Blocks until the stream drains. */
    void streamSynchronize(long stream);

    /** Makes {@code stream} wait for {@code event} ({@code hipStreamWaitEvent}). */
    void streamWaitEvent(long stream, long event);

    /** Creates a timing-disabled event ({@code hipEventCreateWithFlags}). */
    long eventCreate();

    /** Destroys an event. */
    void eventDestroy(long event);

    /** Records {@code event} as a point on {@code stream}. */
    void eventRecord(long event, long stream);

    /** True when the point is reached ({@code hipEventQuery}); false while it is not. */
    boolean eventQuery(long event);

    /** Blocks until the point is reached ({@code hipEventSynchronize}). */
    void eventSynchronize(long event);

    /**
     * Loads a code-object image ({@code hipModuleLoadData}).
     *
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the image does not support this device
     * @throws RuntimeException                                              any other translated failure
     */
    long moduleLoadData(byte[] image);

    /** Unloads a module ({@code hipModuleUnload}). */
    void moduleUnload(long module);

    /**
     * The function of {@code name} in the module ({@code hipModuleGetFunction}).
     *
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the module holds no such function
     * @throws RuntimeException                                              any other translated failure
     */
    long moduleGetFunction(long module, String name);

    /** One kernel-function attribute ({@code hipFuncGetAttribute}); the driver must answer. */
    int funcGetAttribute(long function, int attribute);

    /**
     * Launches {@code function} over the parameter snapshot ({@code hipModuleLaunchKernel}).
     *
     * @throws RuntimeException the translated failure
     */
    void launchKernel(long function, int gridDimX, int gridDimY, int gridDimZ,
                      int blockDimX, int blockDimY, int blockDimZ,
                      int sharedMemBytes, long stream, Object[] params);

    /** Blocks until every stream of the current device drains ({@code hipDeviceSynchronize}). */
    void synchronize();

    // -----------------------------------------------------------------------------------------
    // Stream capture and graphs
    // -----------------------------------------------------------------------------------------

    /** Begins recording the commands enqueued on {@code stream} into a graph. */
    void streamBeginCapture(long stream);

    /** Ends the capture and returns the recorded graph; nothing recorded executes yet. */
    long streamEndCapture(long stream);

    /** Instantiates {@code graph} as an executable graph. */
    long graphInstantiate(long graph);

    /** Launches an executable graph on {@code stream}. */
    void graphLaunch(long exec, long stream);

    /** Destroys a captured graph. */
    void graphDestroy(long graph);

    /** Destroys an executable graph. */
    void graphExecDestroy(long exec);

    /** The number of nodes of {@code graph}. */
    int graphNodeCount(long graph);

    /** The nodes of {@code graph} in dependency (capture) order. */
    long[] graphNodesInOrder(long graph);

    /**
     * The node kind: {@link HipNative#GRAPH_NODE_KERNEL}, {@link HipNative#GRAPH_NODE_MEMCPY},
     * {@link HipNative#GRAPH_NODE_MEMSET}, or -1 for another kind.
     */
    int graphNodeType(long node);

    /** Replaces the grid, block and parameters of a captured kernel node. */
    void graphExecKernelNodeSetParams(long exec, long node, long function, int gridDimX, int gridDimY,
                                      int gridDimZ, int blockDimX, int blockDimY, int blockDimZ,
                                      Object[] params);

    /** Replaces the buffers and range of a captured device-to-device copy. */
    void graphExecMemcpyNodeSetParams(long exec, long node, long dst, long src, long bytes);

    /** Replaces the destination, 32-bit pattern and word count of a captured fill. */
    void graphExecMemsetNodeSetParams(long exec, long node, long dst, int value, long count);

    /** The symbolic name the runtime reports for a {@code hipError_t} code. */
    String errorString(int code);
}
