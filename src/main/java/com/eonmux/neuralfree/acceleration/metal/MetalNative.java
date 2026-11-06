package com.eonmux.neuralfree.acceleration.metal;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The JNI surface of Apple's Metal framework, one native method per shim entry point.
 *
 * <p>The shim is an Objective-C library ({@code native/metal/metalshim.m}) built with clang on
 * macOS and loaded here through {@link System#loadLibrary(String)} or the absolute path of the
 * system property {@value #SHIM_PATH_PROPERTY}. Without it — any other operating system, or the
 * shim not built — {@link #isAvailable()} is false and the backend refuses to start with that
 * reason.</p>
 *
 * <p>Contract, mirrored by the Objective-C file:</p>
 * <ul>
 *   <li>Metal objects (devices are never handed out, buffers, command queues, libraries,
 *       functions, pipelines, shared events, command buffers and encoders) are opaque
 *       {@code long} handles. Every handle this class returns is retained by the shim and
 *       released by {@link #releaseObject(long)}; an encoder is released by
 *       {@link #endEncoding(long)}.</li>
 *   <li>Device queries return an error status or a value; failures of an operation that creates an
 *       object throw {@link IllegalStateException} with the framework's message instead of
 *       returning 0.</li>
 *   <li>{@link #deviceAttribute} takes the shim's own selector table ({@link #ATTR_MAX_THREADS_PER_THREADGROUP}
 *       and friends), not a Metal property selector.</li>
 *   <li>Host data crosses the boundary as arrays or direct buffers; the shim never keeps a
 *       pointer into JVM memory beyond the call.</li>
 * </ul>
 *
 * <p>The methods may only be called while the shim is loaded ({@link #ensureLoaded()}).</p>
 */
public final class MetalNative {

    /** Absolute path from which the shim is loaded instead of {@code System.loadLibrary}. */
    public static final String SHIM_PATH_PROPERTY = "neuralfree.metal.shim.path";
    /** Short name of the shim library ({@code libmetalshim.dylib}). */
    public static final String SHIM_LIBRARY = "metalshim";

    /** Device attribute selector: threads per threadgroup, the product of the three dimensions. */
    public static final int ATTR_MAX_THREADS_PER_THREADGROUP = 0;
    /** Device attribute selector: threadgroup memory in bytes. */
    public static final int ATTR_MAX_THREADGROUP_MEMORY = 1;
    /** Device attribute selector: recommended maximum working-set size in bytes. */
    public static final int ATTR_RECOMMENDED_MAX_WORKING_SET = 2;
    /** Device attribute selector: non-zero when the GPU and CPU share memory. */
    public static final int ATTR_HAS_UNIFIED_MEMORY = 3;
    /** Device attribute selector: non-zero for a low-power GPU. */
    public static final int ATTR_IS_LOW_POWER = 4;
    /** Device attribute selector: non-zero for a removable GPU. */
    public static final int ATTR_IS_REMOVABLE = 5;
    /** Device attribute selector: largest buffer allocation in bytes. */
    public static final int ATTR_MAX_BUFFER_LENGTH = 6;
    /** Device attribute selector: registry id, unique among the GPUs of the machine. */
    public static final int ATTR_REGISTRY_ID = 7;
    /** Device attribute selector: argument-buffer tier (0 disabled, 1 tier 1, 2 tier 2). */
    public static final int ATTR_ARGUMENT_BUFFERS_TIER = 8;

    private static final AtomicBoolean LOADED = new AtomicBoolean(false);
    private static volatile String loadFailure;

    private MetalNative() {
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
        synchronized (MetalNative.class) {
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

    /** Number of Metal devices, or a negative value when the framework cannot enumerate them. */
    public static native int getDeviceCount();

    /** The marketing name of the device, or null when the query fails. */
    public static native String getDeviceName(int device);

    /** A comma-separated list of the GPU families the device supports, for example {@code apple7,common3}. */
    public static native String getDeviceFamily(int device);

    /** Reads the attribute the shim's selector names into {@code out[0]}; returns 0 on success. */
    public static native int deviceAttribute(int device, int selector, long[] out);

    /** The operating-system release that provides the Metal driver, for example {@code Version 14.4.1}. */
    public static native String driverVersion();

    // -----------------------------------------------------------------------------------------
    // Memory and other Metal objects
    // -----------------------------------------------------------------------------------------

    /** Creates a shared-storage buffer of {@code length} bytes. */
    public static native long newBuffer(int device, long length);

    /**
     * A direct byte buffer over the host mapping of a shared buffer, so the CPU can fill a weight
     * buffer in place. The view is valid until the buffer is released.
     */
    public static native java.nio.ByteBuffer bufferHostView(long buffer, long length);

    /** Retains an object so it outlives the command that references it. */
    public static native void retainObject(long object);

    /** Releases a handle returned by this class. Idempotent for a 0 handle. */
    public static native void releaseObject(long object);

    // -----------------------------------------------------------------------------------------
    // Command queues, libraries and pipelines
    // -----------------------------------------------------------------------------------------

    /** Creates a command queue on the device. */
    public static native long newCommandQueue(int device);

    /**
     * Compiles Metal Shading Language {@code source} into a library. {@code options} is a
     * whitespace-separated list of {@code -DNAME} or {@code -DNAME=VALUE} preprocessor macros.
     */
    public static native long newLibrary(int device, String source, String options);

    /** Looks a kernel function up by name. */
    public static native long newFunction(long library, String name);

    /** Creates the compute pipeline of a function. */
    public static native long newComputePipeline(int device, long function);

    /** {@code specKinds} entry of a {@code uint} function constant. */
    public static final int SPEC_CONSTANT_UINT = 0;
    /** {@code specKinds} entry of an {@code int} function constant. */
    public static final int SPEC_CONSTANT_INT = 1;
    /** {@code specKinds} entry of a {@code float} function constant. */
    public static final int SPEC_CONSTANT_FLOAT = 2;

    /**
     * Looks a kernel function up by name with the values of its {@code [[function_constant(id)]]}
     * specialisation constants, and builds its compute pipeline. The three arrays are parallel:
     * {@code specIds} the constant ids, {@code specKinds} one of the {@code SPEC_CONSTANT_*}
     * type codes and {@code specValues} the 32-bit value. Empty arrays mean the function has no
     * function constants.
     *
     * @throws IllegalStateException when the library has no such function or the pipeline does not build
     */
    public static native long newComputePipelineWithConstants(int device, long library, String name,
                                                              int[] specIds, int[] specKinds, long[] specValues);

    /** Largest threadgroup size a pipeline can launch with. */
    public static native int pipelineMaxThreads(long pipeline);

    // -----------------------------------------------------------------------------------------
    // Synchronisation
    // -----------------------------------------------------------------------------------------

    /** Creates the shared event one backend tracks every batch on. */
    public static native long newSharedEvent(int device);

    /** The highest value {@code event} has been signaled at so far. */
    public static native long eventSignaledValue(long event);

    /** Blocks until {@code event} reaches {@code value}. */
    public static native void eventWait(long event, long value);

    // -----------------------------------------------------------------------------------------
    // Command encoding
    // -----------------------------------------------------------------------------------------

    /** Creates a command buffer on the queue, retained for the caller. */
    public static native long newCommandBuffer(long queue);

    /** Creates (and retains) the compute encoder of a command buffer. */
    public static native long computeEncoder(long commandBuffer);

    /** Ends and releases a compute encoder. */
    public static native void endEncoding(long encoder);

    /** Binds the compute pipeline the following dispatches of this encoder run. */
    public static native void setPipeline(long encoder, long pipeline);

    /** Binds a buffer at an argument index for the next dispatch. */
    public static native void setBuffer(long encoder, long buffer, long offset, int index);

    /** Binds {@code length} bytes of {@code data} at an argument index for the next dispatch. */
    public static native void setBytes(long encoder, byte[] data, int length, int index);

    /** Dispatches a grid of threadgroups of the given size. */
    public static native void dispatchThreadgroups(long encoder, long groupsX, long groupsY, long groupsZ,
                                                   long threadsX, long threadsY, long threadsZ);

    /** Makes the command buffer wait for {@code event} to reach {@code value}. */
    public static native void encodeWait(long commandBuffer, long event, long value);

    /** Makes the command buffer signal {@code event} at {@code value} when it completes. */
    public static native void encodeSignal(long commandBuffer, long event, long value);

    /** Encodes a device-to-device buffer copy as a blit pass. */
    public static native void copyBuffer(long commandBuffer, long src, long srcOffset, long dst, long dstOffset,
                                          long size);

    /** Encodes a device-side fill of {@code size} bytes with the 32-bit {@code value}. */
    public static native void fillBuffer(long commandBuffer, long buffer, long offset, long size, int value);

    /** Commits the command buffer for execution. */
    public static native void commit(long commandBuffer);

    /** The failure message of a failed command buffer, or null when it has not failed. */
    public static native String commandBufferError(long commandBuffer);
}
