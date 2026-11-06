package com.eonmux.neuralfree.acceleration.hip;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The JNI surface of the HIP runtime, one native method per shim entry point.
 *
 * <p>The shim is a small C library ({@code native/hip/hipshim.c}) built by hipcc and loaded
 * here through {@link System#loadLibrary(String)} or the absolute path of the system property
 * {@value #SHIM_PATH_PROPERTY}. Without it — no ROCm runtime installed, or the shim not built —
 * {@link #isAvailable()} is false and the backend refuses to start with that reason.</p>
 *
 * <p>Contract, mirrored by the C file:</p>
 * <ul>
 *   <li>Every method that can fail returns the {@code hipError_t} status as an {@code int};
 *       results travel through out-arrays. {@code 0} is {@code hipSuccess}.</li>
 *   <li>Streams, events, modules, functions and device pointers are opaque {@code long}
 *       handles; pinned host memory is a {@code long} host pointer.</li>
 *   <li>{@link #launchKernel} receives the arguments as an {@code Object[]} snapshot, one
 *       element per parameter: {@code Long} a device pointer, {@code Integer}/{@code Boolean}
 *       a 4-byte int, {@code Float} a float, {@code Double} a double, primitive arrays by
 *       reference for the launch.</li>
 *   <li>{@link #deviceAttribute} takes the shim's own selector table ({@link #ATTR_WARP_SIZE}
 *       and friends), not a {@code hipDeviceAttribute_t} value.</li>
 * </ul>
 *
 * <p>The methods may only be called while the shim is loaded ({@link #ensureLoaded()}).</p>
 */
public final class HipNative {

    /** Absolute path from which the shim is loaded instead of {@code System.loadLibrary}. */
    public static final String SHIM_PATH_PROPERTY = "neuralfree.hip.shim.path";
    /** Short name of the shim library ({@code libhipshim.so}). */
    public static final String SHIM_LIBRARY = "hipshim";

    /** Device attribute selector of the shim's table: threads per wavefront ({@code warpSize}). */
    public static final int ATTR_WARP_SIZE = 0;
    /** Device attribute selector: largest work-group size ({@code maxThreadsPerBlock}). */
    public static final int ATTR_MAX_THREADS_PER_BLOCK = 1;
    /** Device attribute selector: {@code maxGridSize[0]}. */
    public static final int ATTR_MAX_GRID_DIM_X = 2;
    /** Device attribute selector: {@code maxGridSize[1]}. */
    public static final int ATTR_MAX_GRID_DIM_Y = 3;
    /** Device attribute selector: {@code maxGridSize[2]}. */
    public static final int ATTR_MAX_GRID_DIM_Z = 4;
    /** Device attribute selector: compute units ({@code multiProcessorCount}). */
    public static final int ATTR_MULTIPROCESSOR_COUNT = 5;
    /** Device attribute selector: peak clock in kHz ({@code clockRate}). */
    public static final int ATTR_CLOCK_RATE = 6;
    /** Device attribute selector: shared memory per block in bytes. */
    public static final int ATTR_SHARED_PER_BLOCK = 7;
    /** Device attribute selector: opt-in shared memory per block in bytes. */
    public static final int ATTR_SHARED_PER_BLOCK_OPTIN = 8;
    /** Device attribute selector: non-zero for an APU ({@code integrated}). */
    public static final int ATTR_INTEGRATED = 9;
    /** Device attribute selector: copy engines ({@code asyncEngineCount}). */
    public static final int ATTR_ASYNC_ENGINE_COUNT = 10;

    /** Kernel-function attribute: largest block the function launches with. */
    public static final int FUNC_MAX_THREADS_PER_BLOCK = 0;
    /** Kernel-function attribute: statically allocated shared memory per block, in bytes. */
    public static final int FUNC_SHARED_SIZE_BYTES = 1;

    /** {@code hipGraphNodeType} of a captured kernel dispatch. */
    public static final int GRAPH_NODE_KERNEL = 0;
    /** {@code hipGraphNodeType} of a captured device copy. */
    public static final int GRAPH_NODE_MEMCPY = 1;
    /** {@code hipGraphNodeType} of a captured fill. */
    public static final int GRAPH_NODE_MEMSET = 2;

    private static final AtomicBoolean LOADED = new AtomicBoolean(false);
    private static volatile String loadFailure;

    private HipNative() {
    }

    /**
     * Loads the shim once per process; later calls do nothing.
     *
     * @throws UnsatisfiedLinkError when the shim library is missing or refuses to load
     */
    public static void ensureLoaded() {
        if (LOADED.get()) {
            return;
        }
        synchronized (HipNative.class) {
            if (LOADED.get()) {
                return;
            }
            String path = System.getProperty(SHIM_PATH_PROPERTY);
            try {
                if (path == null || path.trim().isEmpty()) {
                    System.loadLibrary(SHIM_LIBRARY);
                } else {
                    System.load(path.trim());
                }
            } catch (UnsatisfiedLinkError | SecurityException e) {
                loadFailure = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
                throw e;
            }
            LOADED.set(true);
        }
    }

    /** True when the shim loaded and the native methods are callable. */
    public static boolean isAvailable() {
        try {
            ensureLoaded();
            return true;
        } catch (UnsatisfiedLinkError | SecurityException e) {
            return false;
        }
    }

    /** Why the shim is not loaded, or null when it is. */
    public static String loadFailure() {
        return LOADED.get() ? null : loadFailure;
    }

    // -----------------------------------------------------------------------------------------
    // Device enumeration and queries
    // -----------------------------------------------------------------------------------------

    /** Number of HIP devices, or the negated {@code hipError_t} status of the enumeration. */
    public static native int getDeviceCount();

    /** Makes {@code device} the calling thread's device ({@code hipSetDevice}). */
    public static native int setDevice(int device);

    /** The marketing name of the device, or null when the properties query fails. */
    public static native String getDeviceName(int device);

    /** The processor name of the device, for example {@code gfx1100} or {@code gfx90a:xnack-}, or null. */
    public static native String getGcnArchName(int device);

    /** Bytes of global memory, or -1 when the query fails. */
    public static native long deviceTotalMem(int device);

    /** Reads the attribute the shim's selector names into {@code out[0]}; returns the status. */
    public static native int deviceAttribute(int device, int selector, int[] out);

    /** The HIP runtime version as {@code 1000 * major + 10 * minor} into {@code out[0]}; returns the status. */
    public static native int getDriverVersion(int[] out);

    /** Writes the 16 device-UUID bytes into {@code out}; returns the status. */
    public static native int getDeviceUuid(int device, byte[] out);

    /** Writes the PCI bus id, ASCII and NUL-terminated, into {@code out}; returns the status. */
    public static native int getPciBusId(int device, byte[] out);

    // -----------------------------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------------------------

    /** Allocates device memory into {@code out[0]}; returns the status. */
    public static native int memAlloc(long bytes, long[] out);

    /** Frees a device allocation. */
    public static native int memFree(long pointer);

    /** Free and total global memory of the current device into the out arrays; returns the status. */
    public static native int memGetInfo(long[] freeOut, long[] totalOut);

    /** Allocates page-locked portable host memory into {@code out[0]}; returns the status. */
    public static native int hostAlloc(long bytes, long[] out);

    /** Frees page-locked host memory. */
    public static native int hostFree(long pointer);

    /** A direct ByteBuffer view of {@code [pointer, pointer + bytes)}, or null. */
    public static native Object pinnedView(long pointer, int bytes);

    /** Page-locked host to device, ordered on {@code stream}; returns the status. */
    public static native int memcpyHtoDAsync(long dst, long src, long bytes, long stream);

    /** Device to page-locked host, ordered on {@code stream}; returns the status. */
    public static native int memcpyDtoHAsync(long dst, long src, long bytes, long stream);

    /** Device to device, ordered on {@code stream}; returns the status. */
    public static native int memcpyDtoDAsync(long dst, long src, long bytes, long stream);

    /** Fills {@code count} 32-bit words with {@code value}, ordered on {@code stream}; returns the status. */
    public static native int memsetD32Async(long dst, int value, long count, long stream);

    // -----------------------------------------------------------------------------------------
    // Streams and events
    // -----------------------------------------------------------------------------------------

    /** Creates a non-blocking stream into {@code out[0]}; returns the status. */
    public static native int streamCreate(long[] out);

    /** Destroys a stream. */
    public static native int streamDestroy(long stream);

    /** Blocks until the stream drains. */
    public static native int streamSynchronize(long stream);

    /** Makes {@code stream} wait for {@code event}. */
    public static native int streamWaitEvent(long stream, long event);

    /** Creates a timing-disabled event into {@code out[0]}; returns the status. */
    public static native int eventCreate(long[] out);

    /** Destroys an event. */
    public static native int eventDestroy(long event);

    /** Records {@code event} as a point on {@code stream}. */
    public static native int eventRecord(long event, long stream);

    /** {@code hipSuccess} when the point is reached, {@code hipErrorNotReady} while it is not. */
    public static native int eventQuery(long event);

    /** Blocks until the point is reached. */
    public static native int eventSynchronize(long event);

    // -----------------------------------------------------------------------------------------
    // Modules and kernels
    // -----------------------------------------------------------------------------------------

    /** Loads a code-object image into {@code out[0]}; returns the status. */
    public static native int moduleLoadData(long[] out, byte[] image);

    /** Unloads a module. */
    public static native int moduleUnload(long module);

    /** The function of {@code name} into {@code out[0]}; returns the status. */
    public static native int moduleGetFunction(long[] out, long module, String name);

    /** Reads the kernel-function attribute into {@code out[0]}; returns the status. */
    public static native int funcGetAttribute(int[] out, int attribute, long function);

    /** Launches {@code function} over the parameter snapshot; returns the status. */
    public static native int launchKernel(long function, int gridDimX, int gridDimY, int gridDimZ,
                                          int blockDimX, int blockDimY, int blockDimZ,
                                          int sharedMemBytes, long stream, Object[] params);

    /** Blocks until every stream of the current device drains ({@code hipDeviceSynchronize}). */
    public static native int synchronize();

    // -----------------------------------------------------------------------------------------
    // Stream capture and graphs
    // -----------------------------------------------------------------------------------------

    /** Begins recording {@code stream}'s commands into a graph ({@code hipStreamBeginCapture}). */
    public static native int streamBeginCapture(long stream);

    /** Ends the capture, writing the recorded graph into {@code out[0]}; returns the status. */
    public static native int streamEndCapture(long stream, long[] out);

    /** Instantiates the executable graph into {@code out[0]}; returns the status. */
    public static native int graphInstantiate(long[] out, long graph);

    /** Launches an executable graph on {@code stream} ({@code hipGraphLaunch}). */
    public static native int graphLaunch(long exec, long stream);

    /** Destroys a captured graph. */
    public static native int graphDestroy(long graph);

    /** Destroys an executable graph. */
    public static native int graphExecDestroy(long exec);

    /** Writes the number of nodes of {@code graph} into {@code out[0]}; returns the status. */
    public static native int graphNodeCount(long graph, int[] out);

    /** Writes the nodes of {@code graph} into {@code out} in dependency (capture) order. */
    public static native int graphNodesInOrder(long graph, long[] out);

    /** Writes the {@code hipGraphNodeType} of {@code node} into {@code out[0]}. */
    public static native int graphNodeType(long node, int[] out);

    /** Replaces the grid, block and parameters of a captured kernel node; returns the status. */
    public static native int graphExecKernelNodeSetParams(long exec, long node, long function,
                                                          int gridDimX, int gridDimY, int gridDimZ,
                                                          int blockDimX, int blockDimY, int blockDimZ,
                                                          Object[] params);

    /** Replaces the buffers and range of a captured device copy; returns the status. */
    public static native int graphExecMemcpyNodeSetParams(long exec, long node, long dst, long src, long bytes);

    /** Replaces the destination, pattern and width of a captured fill; returns the status. */
    public static native int graphExecMemsetNodeSetParams(long exec, long node, long dst, int value, long count);

    /** The symbolic name the runtime reports for a {@code hipError_t} code. */
    public static native String errorString(int code);
}
