package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.cache.CacheKey;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.jocl.CLException;
import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

import static org.jocl.CL.*;

/**
 * OpenCL implementation of {@link ComputeBackend} on one {@link CLDevice}.
 *
 * <p><b>Programs.</b> Kernels built from the same program share one compiled {@code cl_program}.
 * {@link #loadKernel(KernelArtifact)} keys programs by {@link KernelArtifact#programKey()} and
 * evaluates the artifact's source only on a program-cache miss; {@link #compileKernel} keys them by
 * a SHA-256 of the source plus the options. Below the in-memory cache, an
 * {@link OclProgramBinaryCache} stores built binaries on disk and reloads them on the next run,
 * falling back to a source build when the driver rejects a binary.</p>
 *
 * <p><b>Queues.</b> {@link #computeQueue()} and {@link #transferQueue()} are owned by the backend:
 * closing them only finishes them, and {@link #close()} releases them. Queues are created with
 * {@code CL_QUEUE_PROFILING_ENABLE} only when {@code -Dneuralfree.gpu.profile=true}.</p>
 *
 * <p><b>Errors.</b> Driver failures are translated by {@link OclErrors} in either JOCL exception
 * mode. Each kernel handle carries its {@code CL_KERNEL_NUM_ARGS}, {@code CL_KERNEL_WORK_GROUP_SIZE}
 * and {@code CL_KERNEL_LOCAL_MEM_SIZE}, which the queues and {@link #prepareKernel} enforce.</p>
 *
 * <p><b>Subgroups.</b> The subgroup capability is reported only when OpenCL C 1.2 programs can
 * enable subgroup reductions ({@code cl_intel_subgroups}); see {@link #programsCanUseSubgroups}.</p>
 */
@NotThreadSafe("Delegates to non-thread-safe OclComputeQueue instances for compute and transfer. "
        + "KernelCache is thread-safe, but buffer allocation/queue operations require external synchronization. "
        + "Recommended: one backend instance per thread, or external mutex for shared access.")
public final class OclComputeBackend implements ComputeBackend {
    private static final Logger LOGGER = LoggingUtil.getLogger(OclComputeBackend.class);

    /** System property that enables per-command profiling on every queue of the backend. */
    public static final String PROFILE_PROPERTY = "neuralfree.gpu.profile";

    private static final String SOURCE_PROGRAM_PREFIX = "source:";
    private static final String ARTIFACT_PROGRAM_PREFIX = "program:";

    private final CLDevice clDevice;
    private final DeviceCapabilities capabilities;
    private final KernelCache kernelCache;
    // Compiled cl_program cache. Keys are "source:<sha256>|<options>" for compileKernel and
    // "program:<programKey>" for loadKernel. All kernels of one program share ONE compiled program;
    // programs are owned by the backend and released in close().
    private final java.util.Map<String, cl_program> programCache = new java.util.HashMap<>();
    // Build options seen per artifact program key; one program key must always carry the same options.
    private final java.util.Map<String, String> artifactOptions = new java.util.HashMap<>();
    private final OclProgramBinaryCache binaryCache;
    private final boolean profiling;
    private final OclComputeQueue computeQueue;
    private final OclComputeQueue transferQueue;
    private final AtomicLong allocatedBytes = new AtomicLong(0);
    private boolean closed = false;

    public OclComputeBackend(CLDevice clDevice) {
        if (clDevice == null) {
            throw new IllegalArgumentException("CLDevice cannot be null");
        }
        this.clDevice = clDevice;
        this.capabilities = buildCapabilities(clDevice);
        this.kernelCache = new KernelCache();
        this.profiling = Boolean.getBoolean(PROFILE_PROPERTY);

        cl_device_id deviceId = clDevice.getDeviceID();
        this.binaryCache = OclProgramBinaryCache.forDevice(
                getStringDeviceInfo(deviceId, CL_DEVICE_VENDOR),
                capabilities.getDeviceName(),
                capabilities.getDriverVersion());

        cl_context context = clDevice.getDeviceContext();
        cl_command_queue computeCmdQueue = createCommandQueue(context, deviceId);
        cl_command_queue transferCmdQueue = null;
        try {
            transferCmdQueue = createCommandQueue(context, deviceId);
            this.computeQueue = new OclComputeQueue(computeCmdQueue, ComputeQueue.QueueType.COMPUTE, true, profiling);
            this.transferQueue = new OclComputeQueue(transferCmdQueue, ComputeQueue.QueueType.TRANSFER, true,
                    profiling);
        } catch (RuntimeException | Error e) {
            // close() cannot run on a backend that was never constructed, so the queues are released here.
            releaseQueue(computeCmdQueue, e);
            if (transferCmdQueue != null) {
                releaseQueue(transferCmdQueue, e);
            }
            throw e;
        }

        LOGGER.info("OpenCL backend initialized - Device: {}, VRAM: {} MB, Compute Units: {}, Max Work Group Size: {}, "
                        + "class: {}, subgroup: {}, profiling: {}, binary cache: {}",
                capabilities.getDeviceName(),
                capabilities.getTotalVramBytes() / (1024 * 1024),
                capabilities.getComputeUnits(),
                capabilities.getMaxWorkGroupSize(),
                capabilities.getDeviceClass(),
                capabilities.getSubgroupSize(),
                profiling,
                binaryCache.isEnabled() ? binaryCache.directory() : "disabled");
    }

    @Override
    public BackendType backendType() {
        return BackendType.OPENCL;
    }

    @Override
    public DeviceCapabilities capabilities() {
        return capabilities;
    }

    @Override
    public ComputeQueue computeQueue() {
        ensureOpen();
        return computeQueue;
    }

    @Override
    public ComputeQueue transferQueue() {
        ensureOpen();
        return transferQueue;
    }

    /** Alignment for GPU buffer allocations (256 bytes = GPU cache line / memory bank boundary).
     *  Ensures buffers start at aligned boundaries for optimal memory access patterns. */
    private static final long GPU_BUFFER_ALIGNMENT = 256L;

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        ensureOpen();
        if (sizeInBytes <= 0) {
            throw new IllegalArgumentException("sizeInBytes must be positive");
        }

        // Round up to 256-byte alignment for optimal GPU memory bank alignment
        long alignedSize = ((sizeInBytes + GPU_BUFFER_ALIGNMENT - 1)
                / GPU_BUFFER_ALIGNMENT) * GPU_BUFFER_ALIGNMENT;

        long flags = readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
        int[] error = new int[1];
        cl_mem mem;
        try {
            mem = clCreateBuffer(clDevice.getDeviceContext(), flags, alignedSize, null, error);
        } catch (CLException e) {
            throw OclErrors.translate(e.getStatus(), "clCreateBuffer", alignedSize + " bytes", sizeInBytes, e);
        }
        if (error[0] != CL_SUCCESS) {
            throw OclErrors.translate(error[0], "clCreateBuffer", alignedSize + " bytes", sizeInBytes, null);
        }
        if (mem == null) {
            throw new GpuBackendException(OclErrors.BACKEND, CL_SUCCESS, "clCreateBuffer (" + alignedSize
                    + " bytes) reported success but returned no buffer");
        }

        allocatedBytes.addAndGet(alignedSize);
        LOGGER.trace("Allocated OpenCL buffer: {} bytes (aligned to {}), readOnly={}", alignedSize, GPU_BUFFER_ALIGNMENT, readOnly);
        return new OclGpuBuffer(mem, alignedSize, allocatedBytes);
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        ensureOpen();
        Objects.requireNonNull(kernelName, "kernelName cannot be null");
        Objects.requireNonNull(kernelSource, "kernelSource cannot be null");

        String sourceHash = computeSourceHash(kernelSource);
        String options = buildOptions != null ? buildOptions : "";
        CacheKey cacheKey = new CacheKey(kernelName, sourceHash, options);

        boolean wasCached = kernelCache.contains(cacheKey);
        long startTime = System.nanoTime();
        KernelHandle handle = kernelCache.getOrCompile(cacheKey, key -> doCompileKernel(kernelName, kernelSource, options, sourceHash));
        long compileTimeMs = (System.nanoTime() - startTime) / 1_000_000;

        if (wasCached) {
            LOGGER.debug("Kernel cache hit: {} ({} ms)", kernelName, compileTimeMs);
        } else {
            LOGGER.info("Kernel compiled: {} ({} ms)", kernelName, compileTimeMs);
        }

        return handle;
    }

    /**
     * Loads a kernel from an {@link KernelArtifact.Kind#OPENCL_SOURCE} artifact. The program is
     * cached under the artifact's program key, so its source is assembled and built (or loaded
     * from the binary cache) once for all kernels of the program.
     *
     * @throws UnsupportedKernelException when the artifact is not OpenCL source, or the program has
     *                                    no kernel of that name or no executable for this device
     * @throws IllegalArgumentException   when one program key is used with different build options
     * @throws GpuBackendException        when the program fails to build (the message holds the build log)
     */
    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        ensureOpen();
        if (artifact == null) {
            throw new IllegalArgumentException("kernel artifact must not be null");
        }
        if (artifact.kind() != KernelArtifact.Kind.OPENCL_SOURCE) {
            throw new UnsupportedKernelException("OpenCL backend cannot load " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName() + " on " + capabilities.getDeviceName());
        }
        String programKey = artifact.programKey();
        String options = artifact.buildOptions();
        synchronized (this) {
            String previous = artifactOptions.putIfAbsent(programKey, options);
            if (previous != null && !previous.equals(options)) {
                throw new IllegalArgumentException("OpenCL program key '" + programKey
                        + "' is used with different build options: '" + previous + "' and '" + options + "'");
            }
        }
        String cacheMapKey = ARTIFACT_PROGRAM_PREFIX + programKey;
        CacheKey cacheKey = new CacheKey(artifact.kernelName(), cacheMapKey, options);
        return kernelCache.getOrCompile(cacheKey, key -> {
            cl_program program = getOrBuildProgram(cacheMapKey, artifact::sourceCode, options, "program " + programKey);
            return createKernel(program, artifact.kernelName(), "program " + programKey);
        });
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        return kind == KernelArtifact.Kind.OPENCL_SOURCE;
    }

    /**
     * Checks that the kernel accepts work-groups of this size, so a kernel whose compiled
     * {@code CL_KERNEL_WORK_GROUP_SIZE} is below the requested size is reported when it is prepared
     * rather than at its first dispatch.
     *
     * @throws UnsupportedKernelException when the work-group exceeds the kernel's limit
     * @throws IllegalArgumentException   when the handle is not an OpenCL kernel of this backend or the
     *                                    size is malformed
     */
    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        ensureOpen();
        if (!(kernel instanceof OclKernelHandle)) {
            throw new IllegalArgumentException("kernel must be an OclKernelHandle");
        }
        if (localWorkSize == null || localWorkSize.length < 1 || localWorkSize.length > 3) {
            throw new IllegalArgumentException("local work size of " + kernel.kernelName()
                    + " must have 1 to 3 dimensions");
        }
        for (long l : localWorkSize) {
            if (l <= 0) {
                throw new IllegalArgumentException("local work size of " + kernel.kernelName()
                        + " must be positive: " + java.util.Arrays.toString(localWorkSize));
            }
        }
        ((OclKernelHandle) kernel).checkLocalSize(localWorkSize);
    }

    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        ensureOpen();
        if (!(template instanceof OclKernelHandle)) {
            throw new IllegalArgumentException("template must be an OclKernelHandle");
        }
        OclKernelHandle ocl = (OclKernelHandle) template;
        if (ocl.getCLProgram() == null) {
            throw new IllegalStateException("cannot clone closed OpenCL kernel " + ocl.kernelName());
        }
        return createKernel(ocl.getCLProgram(), ocl.kernelName(), "program of " + ocl.kernelName());
    }

    /** Creates a caller-owned compute queue on the same context; its {@code close()} releases it. */
    @Override
    public ComputeQueue createComputeQueue() {
        ensureOpen();
        cl_context context = clDevice.getDeviceContext();
        cl_device_id deviceId = clDevice.getDeviceID();
        cl_command_queue cmdQueue = createCommandQueue(context, deviceId);
        return new OclComputeQueue(cmdQueue, ComputeQueue.QueueType.COMPUTE, false, profiling);
    }

    @Override
    public long getAllocatedBytes() {
        return allocatedBytes.get();
    }

    @Override
    public long getFreeVramBytes() {
        return capabilities.getTotalVramBytes() - allocatedBytes.get();
    }

    @Override
    public void close() {
        RuntimeException failure = null;
        try {
            BackendResources.closeAll(this);
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            releaseResources();
        } catch (RuntimeException e) {
            if (failure == null) {
                failure = e;
            } else {
                failure.addSuppressed(e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** The device-side part of {@link #close()}, after the tensor runtime's per-backend resources are closed. */
    private void releaseResources() {
        if (closed) {
            return;
        }
        closed = true;
        // Every resource is released even when an earlier step reports a device error.
        RuntimeException failure = null;
        for (OclComputeQueue queue : new OclComputeQueue[]{computeQueue, transferQueue}) {
            try {
                queue.closeInternal();
            } catch (RuntimeException e) {
                LOGGER.error("Closing an OpenCL {} queue failed: {}", queue.getQueueType(), e.toString());
                if (failure == null) {
                    failure = e;
                }
            }
        }
        kernelCache.clear();
        synchronized (this) {
            for (cl_program p : programCache.values()) {
                releaseProgram(p);
            }
            programCache.clear();
            artifactOptions.clear();
        }
        if (failure != null) {
            throw failure;
        }
    }

    public CLDevice getCLDevice() {
        return clDevice;
    }

    public KernelCache getKernelCache() {
        return kernelCache;
    }

    /** The on-disk program binary cache of this device. */
    public OclProgramBinaryCache getBinaryCache() {
        return binaryCache;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("OclComputeBackend already closed");
        }
    }

    /**
     * Creates an in-order queue, with {@code CL_QUEUE_PROFILING_ENABLE} only when profiling is on.
     * Uses {@code clCreateCommandQueueWithProperties} (OpenCL 2.0) and falls back to
     * {@code clCreateCommandQueue} on platforms that reject it.
     */
    private cl_command_queue createCommandQueue(cl_context context, cl_device_id deviceId) {
        long properties = profiling ? CL_QUEUE_PROFILING_ENABLE : 0L;
        int[] error = new int[1];
        int status;
        try {
            // The properties bitfield goes under the CL_QUEUE_PROPERTIES key; passing the bitfield
            // value itself as a key is rejected with CL_INVALID_VALUE by strict drivers.
            cl_queue_properties props = new cl_queue_properties();
            if (properties != 0L) {
                props.addProperty(CL_QUEUE_PROPERTIES, properties);
            }
            cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceId, props, error);
            if (error[0] == CL_SUCCESS && queue != null) {
                return queue;
            }
            status = error[0];
        } catch (CLException e) {
            status = e.getStatus();
        }
        LOGGER.debug("clCreateCommandQueueWithProperties failed ({}); using clCreateCommandQueue",
                stringFor_errorCode(status));
        try {
            error[0] = CL_SUCCESS;
            @SuppressWarnings("deprecation")
            cl_command_queue queue = clCreateCommandQueue(context, deviceId, properties, error);
            if (error[0] != CL_SUCCESS || queue == null) {
                throw new GpuBackendException("OpenCL", error[0], "clCreateCommandQueue failed: "
                        + stringFor_errorCode(error[0]));
            }
            return queue;
        } catch (CLException e) {
            throw new GpuBackendException("OpenCL", e.getStatus(), "clCreateCommandQueue failed: "
                    + stringFor_errorCode(e.getStatus()), e);
        }
    }

    private KernelHandle doCompileKernel(String kernelName, String kernelSource, String buildOptions, String sourceHash) {
        String options = buildOptions != null && !buildOptions.isEmpty() ? buildOptions : "-cl-std=CL1.1";
        String label = "source " + sourceHash.substring(0, Math.min(12, sourceHash.length()));
        // Reuse the compiled program for this (source, options) — all kernels built from the same
        // source share it, so the full source is compiled once instead of once per kernel.
        cl_program program = getOrBuildProgram(SOURCE_PROGRAM_PREFIX + sourceHash + "|" + options,
                () -> kernelSource, options, label);
        return createKernel(program, kernelName, label);
    }

    /**
     * Creates a kernel object of {@code program}. The handle does not own the program, which is
     * released by the program cache in {@link #close()}.
     */
    private KernelHandle createKernel(cl_program program, String kernelName, String programLabel) {
        int[] error = new int[1];
        cl_kernel kernel;
        int status;
        try {
            kernel = clCreateKernel(program, kernelName, error);
            status = error[0];
        } catch (CLException e) {
            kernel = null;
            status = e.getStatus();
        }
        if (status == CL_SUCCESS && kernel != null) {
            return describeKernel(program, kernel, kernelName, programLabel);
        }
        if (status == CL_INVALID_KERNEL_NAME) {
            throw new UnsupportedKernelException("OpenCL " + programLabel + " on " + capabilities.getDeviceName()
                    + " has no kernel named " + kernelName + " (CL_INVALID_KERNEL_NAME)");
        }
        if (status == CL_INVALID_PROGRAM_EXECUTABLE) {
            throw new UnsupportedKernelException("OpenCL " + programLabel + " has no executable for "
                    + capabilities.getDeviceName() + ", so kernel " + kernelName
                    + " is unavailable (CL_INVALID_PROGRAM_EXECUTABLE)");
        }
        throw new GpuBackendException("OpenCL", status, "clCreateKernel(" + kernelName + ") of " + programLabel
                + " failed: " + stringFor_errorCode(status));
    }

    /**
     * Wraps a created kernel with the limits the driver reports for it. A kernel whose static local
     * memory exceeds the device's local memory cannot be launched, so it is released and reported
     * as unavailable.
     */
    private OclKernelHandle describeKernel(cl_program program, cl_kernel kernel, String kernelName,
                                           String programLabel) {
        cl_device_id deviceId = clDevice.getDeviceID();
        long numArgs = queryKernelUInt(kernel, CL_KERNEL_NUM_ARGS, kernelName);
        long maxWorkGroupSize = queryKernelSize(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, kernelName);
        long localMemBytes = queryKernelULong(kernel, deviceId, CL_KERNEL_LOCAL_MEM_SIZE, kernelName);
        long deviceLocalMem = capabilities.getMaxLocalMemoryBytes();
        if (localMemBytes > 0 && deviceLocalMem > 0 && localMemBytes > deviceLocalMem) {
            releaseKernel(kernel);
            throw new UnsupportedKernelException("OpenCL kernel " + kernelName + " of " + programLabel + " needs "
                    + localMemBytes + " bytes of local memory (CL_KERNEL_LOCAL_MEM_SIZE), but "
                    + capabilities.getDeviceName() + " has " + deviceLocalMem);
        }
        LOGGER.debug("OpenCL kernel {}: {} args, max work-group {}, local memory {} bytes", kernelName, numArgs,
                maxWorkGroupSize, localMemBytes);
        return new OclKernelHandle(program, kernel, kernelName, false, (int) numArgs, maxWorkGroupSize,
                localMemBytes, capabilities.getDeviceName());
    }

    /** A {@code cl_uint} kernel attribute, or {@link OclKernelHandle#UNKNOWN} when the driver rejects the query. */
    private static long queryKernelUInt(cl_kernel kernel, int paramName, String kernelName) {
        int[] value = new int[1];
        int status;
        try {
            status = clGetKernelInfo(kernel, paramName, org.jocl.Sizeof.cl_uint, Pointer.to(value), null);
        } catch (CLException e) {
            status = e.getStatus();
        }
        if (status != CL_SUCCESS) {
            LOGGER.debug("clGetKernelInfo(0x{}) of {} unavailable: {}", Integer.toHexString(paramName), kernelName,
                    stringFor_errorCode(status));
            return OclKernelHandle.UNKNOWN;
        }
        return value[0] & 0xFFFFFFFFL;
    }

    /** A {@code size_t} work-group attribute, or {@link OclKernelHandle#UNKNOWN} when the driver rejects the query. */
    private static long queryKernelSize(cl_kernel kernel, cl_device_id deviceId, int paramName, String kernelName) {
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(org.jocl.Sizeof.size_t)
                .order(java.nio.ByteOrder.nativeOrder());
        int status;
        try {
            status = clGetKernelWorkGroupInfo(kernel, deviceId, paramName, org.jocl.Sizeof.size_t,
                    Pointer.to(buffer), null);
        } catch (CLException e) {
            status = e.getStatus();
        }
        if (status != CL_SUCCESS) {
            LOGGER.debug("clGetKernelWorkGroupInfo(0x{}) of {} unavailable: {}", Integer.toHexString(paramName),
                    kernelName, stringFor_errorCode(status));
            return OclKernelHandle.UNKNOWN;
        }
        return org.jocl.Sizeof.size_t == 4 ? buffer.getInt(0) & 0xFFFFFFFFL : buffer.getLong(0);
    }

    /** A {@code cl_ulong} work-group attribute, or {@link OclKernelHandle#UNKNOWN} when the driver rejects the query. */
    private static long queryKernelULong(cl_kernel kernel, cl_device_id deviceId, int paramName, String kernelName) {
        long[] value = new long[1];
        int status;
        try {
            status = clGetKernelWorkGroupInfo(kernel, deviceId, paramName, org.jocl.Sizeof.cl_ulong,
                    Pointer.to(value), null);
        } catch (CLException e) {
            status = e.getStatus();
        }
        if (status != CL_SUCCESS) {
            LOGGER.debug("clGetKernelWorkGroupInfo(0x{}) of {} unavailable: {}", Integer.toHexString(paramName),
                    kernelName, stringFor_errorCode(status));
            return OclKernelHandle.UNKNOWN;
        }
        return value[0];
    }

    private static void releaseKernel(cl_kernel kernel) {
        try {
            clReleaseKernel(kernel);
        } catch (CLException e) {
            LOGGER.debug("clReleaseKernel failed: {}", stringFor_errorCode(e.getStatus()));
        }
    }

    /** Releases a queue of a backend whose construction failed, attaching any release failure to it. */
    private static void releaseQueue(cl_command_queue queue, Throwable constructionFailure) {
        try {
            int status = clReleaseCommandQueue(queue);
            if (status != CL_SUCCESS) {
                constructionFailure.addSuppressed(new GpuBackendException(OclErrors.BACKEND, status,
                        "clReleaseCommandQueue failed: " + stringFor_errorCode(status)));
            }
        } catch (CLException e) {
            constructionFailure.addSuppressed(e);
        }
    }

    /**
     * Returns the cached program for {@code cacheMapKey}, or builds it: first from a verified
     * binary in the {@link OclProgramBinaryCache}, else from source, storing the fresh binary.
     * The source supplier is evaluated only here, on a miss.
     */
    private synchronized cl_program getOrBuildProgram(String cacheMapKey, Supplier<String> source, String options,
                                                      String label) {
        cl_program cached = programCache.get(cacheMapKey);
        if (cached != null) {
            return cached;
        }
        String kernelSource = source.get();
        String binaryKey = binaryCache.isEnabled() ? binaryCache.keyFor(kernelSource, options) : null;
        cl_program program = binaryKey == null ? null : buildFromCachedBinary(binaryKey, options, label);
        if (program == null) {
            program = buildFromSource(kernelSource, options, label);
            if (binaryKey != null) {
                byte[] binary = OclProgramBinaryCache.extractBinary(program, clDevice.getDeviceID());
                if (binary != null) {
                    binaryCache.store(binaryKey, binary);
                } else {
                    LOGGER.debug("OpenCL driver returned no program binary for {}; nothing cached", label);
                }
            }
        }
        programCache.put(cacheMapKey, program);
        return program;
    }

    /**
     * Creates and builds {@code label} from the cached binary under {@code binaryKey}. Returns null,
     * after invalidating the entry, when the binary is rejected at creation
     * ({@code CL_INVALID_BINARY}) or fails to build; the caller then builds from source.
     */
    private cl_program buildFromCachedBinary(String binaryKey, String options, String label) {
        byte[] binary = binaryCache.load(binaryKey);
        if (binary == null) {
            return null;
        }
        long start = System.nanoTime();
        cl_program program = OclProgramBinaryCache.createProgram(clDevice.getDeviceContext(),
                clDevice.getDeviceID(), binary);
        if (program == null) {
            binaryCache.invalidate(binaryKey);
            return null;
        }
        int status = buildProgram(program, options);
        if (status != CL_SUCCESS) {
            LOGGER.info("Cached OpenCL binary of {} failed to build ({}); rebuilding from source. Log:\n{}",
                    label, stringFor_errorCode(status), buildLog(program));
            releaseProgram(program);
            binaryCache.invalidate(binaryKey);
            return null;
        }
        LOGGER.info("OpenCL {} loaded from the binary cache ({} bytes, {} ms)", label, binary.length,
                (System.nanoTime() - start) / 1_000_000);
        return program;
    }

    private cl_program buildFromSource(String kernelSource, String options, String label) {
        int[] error = new int[1];
        cl_program program;
        try {
            // Null lengths: the source string is passed NUL-terminated.
            program = clCreateProgramWithSource(clDevice.getDeviceContext(), 1, new String[]{kernelSource},
                    null, error);
        } catch (CLException e) {
            throw new GpuBackendException("OpenCL", e.getStatus(), "clCreateProgramWithSource failed for "
                    + label + ": " + stringFor_errorCode(e.getStatus()), e);
        }
        if (error[0] != CL_SUCCESS || program == null) {
            throw new GpuBackendException("OpenCL", error[0], "clCreateProgramWithSource failed for "
                    + label + ": " + stringFor_errorCode(error[0]));
        }

        int sourceBytes = kernelSource.getBytes(StandardCharsets.UTF_8).length;
        LOGGER.info("Building OpenCL {} ({} bytes, options: {})", label, sourceBytes, options);
        long start = System.nanoTime();
        int status = buildProgram(program, options);
        if (status != CL_SUCCESS) {
            String msg = "clBuildProgram failed for " + label + " (" + stringFor_errorCode(status) + "):\n"
                    + buildLog(program);
            releaseProgram(program);
            LOGGER.error(msg);
            throw new GpuBackendException("OpenCL", status, msg);
        }
        LOGGER.info("Built OpenCL {} in {} ms", label, (System.nanoTime() - start) / 1_000_000);
        return program;
    }

    /** Builds for this device and returns the status, whether or not JOCL exceptions are enabled. */
    private int buildProgram(cl_program program, String options) {
        try {
            return clBuildProgram(program, 1, new cl_device_id[]{clDevice.getDeviceID()}, options, null, null);
        } catch (CLException e) {
            return e.getStatus();
        }
    }

    private String buildLog(cl_program program) {
        try {
            long[] logSize = new long[1];
            clGetProgramBuildInfo(program, clDevice.getDeviceID(), CL_PROGRAM_BUILD_LOG, 0, null, logSize);
            if (logSize[0] <= 1 || logSize[0] > Integer.MAX_VALUE) {
                return "(empty build log)";
            }
            byte[] logData = new byte[(int) logSize[0]];
            clGetProgramBuildInfo(program, clDevice.getDeviceID(), CL_PROGRAM_BUILD_LOG, logData.length,
                    Pointer.to(logData), null);
            return new String(logData, 0, logData.length - 1, StandardCharsets.UTF_8);
        } catch (CLException e) {
            return "(build log unavailable: " + stringFor_errorCode(e.getStatus()) + ")";
        }
    }

    private static void releaseProgram(cl_program program) {
        try {
            clReleaseProgram(program);
        } catch (CLException e) {
            LOGGER.debug("clReleaseProgram failed: {}", stringFor_errorCode(e.getStatus()));
        }
    }

    private String computeSourceHash(String source) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(source.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required by every Java platform", e);
        }
    }

    /** Vendor-specific {@code clGetDeviceInfo} parameters (Khronos cl_ext.h). */
    private static final int CL_DEVICE_WARP_SIZE_NV = 0x4003;
    private static final int CL_DEVICE_WAVEFRONT_WIDTH_AMD = 0x4043;
    private static final int CL_DEVICE_SUB_GROUP_SIZES_INTEL = 0x4108;
    private static final int PCI_VENDOR_NVIDIA = 0x10DE;
    private static final int PCI_VENDOR_AMD = 0x1002;
    /** Subgroup sizes assumed when the vendor attribute query is unavailable. */
    private static final int NVIDIA_WARP_SIZE = 32;
    private static final int AMD_WAVEFRONT_SIZE = 64;
    /** OpenCL has no per-dimension work-group count limit beyond the int range kernels index with. */
    private static final long[] OPENCL_MAX_WORK_GROUP_COUNT = {Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE};

    private DeviceCapabilities buildCapabilities(CLDevice device) {
        cl_device_id deviceId = device.getDeviceID();

        String deviceName = getStringDeviceInfo(deviceId, CL_DEVICE_NAME);
        long globalMemSize = getLongDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE);
        long maxAllocSize = getLongDeviceInfo(deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        int computeUnits = getIntDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS);
        long maxClockFreq = getIntDeviceInfo(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY) & 0xFFFFFFFFL;
        long maxWorkGroupSize = getSizeDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        String driverVersion = getStringDeviceInfo(deviceId, CL_DRIVER_VERSION);
        long localMemSize = getLongDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_SIZE);
        long deviceType = getLongDeviceInfo(deviceId, CL_DEVICE_TYPE);
        String profile = getStringDeviceInfo(deviceId, CL_DEVICE_PROFILE);

        String extensions = getStringDeviceInfo(deviceId, CL_DEVICE_EXTENSIONS);
        boolean supportsFloat64 = hasExtension(extensions, "cl_khr_fp64");
        boolean supportsFloat16 = hasExtension(extensions, "cl_khr_fp16");
        boolean supportsSubgroups = programsCanUseSubgroups(extensions);
        if (!supportsSubgroups && hasExtension(extensions, "cl_khr_subgroups")) {
            LOGGER.info("{} exposes cl_khr_subgroups, which OpenCL C 1.2 programs cannot enable "
                    + "(it requires OpenCL C 2.0); kernels use local-memory reductions", deviceName);
        }

        boolean supportsAsyncTransfer = true;

        java.util.Set<String> activationFunctions = new java.util.HashSet<>(java.util.Arrays.asList(
                "SIGMOID", "RELU", "TANH", "LEAKY_RELU", "ELU", "SOFTPLUS", "SOFTSIGN",
                "ISRU", "GAUSSIAN", "SINC", "SWISH", "SELU", "HARD_SIGMOID",
                "BENT_IDENTITY", "PRELU", "SRELU", "SOFTEXPONENTIAL", "IDENTITY"
        ));

        java.util.Set<String> costFunctions = new java.util.HashSet<>(java.util.Arrays.asList(
                "MSE", "MAE", "HUBER", "LOG_COSH", "KL_DIVERGENCE", "CROSS_ENTROPY"
        ));

        java.util.Set<String> blasLevels = new java.util.HashSet<>(java.util.Arrays.asList("1", "2", "3"));

        DeviceIdentity identity = device.getIdentity();
        int[] subgroupSizes = subgroupSizes(deviceId, extensions, identity.vendorId());

        java.util.EnumSet<DeviceFeature> features = java.util.EnumSet.of(
                DeviceFeature.CONCURRENT_QUEUES_PROFITABLE,
                // char/short global stores and char arithmetic are core since OpenCL 1.1.
                DeviceFeature.STORAGE_8BIT, DeviceFeature.STORAGE_16BIT, DeviceFeature.SHADER_INT8);
        if (supportsSubgroups) {
            features.add(DeviceFeature.SUBGROUP_ARITHMETIC);
        }
        if (supportsFloat16) {
            features.add(DeviceFeature.SHADER_FLOAT16);
        }
        if ("FULL_PROFILE".equals(profile.trim())) {
            features.add(DeviceFeature.SHADER_INT64);
        }
        if (hasExtension(extensions, "cl_khr_integer_dot_product")) {
            features.add(DeviceFeature.INTEGER_DOT_PRODUCT);
        }

        return new DeviceCapabilities.Builder()
                .deviceName(deviceName)
                .totalVramBytes(globalMemSize)
                .maxAllocSizeBytes(maxAllocSize)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz((int) maxClockFreq)
                .maxWorkGroupSize(maxWorkGroupSize)
                .supportsAsyncTransfer(supportsAsyncTransfer)
                .driverVersion(driverVersion)
                .supportsFloat64(supportsFloat64)
                .supportsFloat16(supportsFloat16)
                .supportsSubgroups(supportsSubgroups)
                .maxLocalMemoryBytes(localMemSize)
                .supportedActivationFunctions(activationFunctions)
                .supportedCostFunctions(costFunctions)
                .supportedBLASLevels(blasLevels)
                .deviceClass(deviceClass(deviceType, device))
                .identity(identity)
                .subgroupSize(subgroupSizes[0])
                .minSubgroupSize(subgroupSizes[1])
                .maxSubgroupSize(subgroupSizes[2])
                .maxWorkGroupCount(OPENCL_MAX_WORK_GROUP_COUNT)
                .maxSharedMemoryBytes(localMemSize)
                .features(features)
                .build();
    }

    /**
     * Whether the backend's programs can use subgroup reductions. Every program is OpenCL C 1.2 or
     * older (kernel libraries build with {@code -cl-std=CL1.2}, {@link #compileKernel} defaults to
     * CL1.1), and there only {@code cl_intel_subgroups} is enabled: {@code cl_khr_subgroups} is an
     * OpenCL C 2.0 extension whose macro the compiler does not define under CL1.2, so the kernels'
     * {@code USE_SUBGROUPS} gate in {@code common/dialect.h} would fall back to local memory. The
     * subgroup flag, and with it the {@code __sg} program variant, is therefore reported only when the
     * compiled kernels really use subgroups.
     */
    static boolean programsCanUseSubgroups(String extensions) {
        return hasExtension(extensions, "cl_intel_subgroups");
    }

    /** Whether the space-separated {@code CL_DEVICE_EXTENSIONS} list names exactly this extension. */
    static boolean hasExtension(String extensions, String name) {
        if (extensions == null || name == null || name.isEmpty()) {
            return false;
        }
        for (String token : extensions.trim().split("\\s+")) {
            if (token.equals(name)) {
                return true;
            }
        }
        return false;
    }

    /**
     * CPU devices are {@link DeviceClass#CPU}. A GPU is {@link DeviceClass#INTEGRATED} when it shares
     * the host's memory ({@code CL_DEVICE_HOST_UNIFIED_MEMORY}) and {@link DeviceClass#DISCRETE}
     * otherwise; when the driver refused that query the class is {@link DeviceClass#UNKNOWN}.
     */
    private static DeviceClass deviceClass(long deviceType, CLDevice device) {
        if ((deviceType & CL_DEVICE_TYPE_CPU) != 0) {
            return DeviceClass.CPU;
        }
        if ((deviceType & CL_DEVICE_TYPE_GPU) == 0) {
            return DeviceClass.UNKNOWN;
        }
        Object unified = device.getDeviceInfo() == null ? null : device.getDeviceInfo().get("CL_DEVICE_HOST_UNIFIED_MEMORY");
        if (!(unified instanceof Boolean)) {
            return DeviceClass.UNKNOWN;
        }
        return ((Boolean) unified) ? DeviceClass.INTEGRATED : DeviceClass.DISCRETE;
    }

    /**
     * Returns {preferred, min, max} subgroup sizes, or zeros when unknown. Sources in order:
     * {@code CL_DEVICE_SUB_GROUP_SIZES_INTEL} (preferred = largest), {@code CL_DEVICE_WARP_SIZE_NV}
     * or 32 on NVIDIA, {@code CL_DEVICE_WAVEFRONT_WIDTH_AMD} or 64 on AMD.
     */
    private static int[] subgroupSizes(cl_device_id deviceId, String extensions, int vendorId) {
        if (extensions.contains("cl_intel_required_subgroup_size")) {
            long[] sizes = querySizeArray(deviceId, CL_DEVICE_SUB_GROUP_SIZES_INTEL);
            if (sizes != null && sizes.length > 0) {
                long min = Long.MAX_VALUE;
                long max = 0;
                for (long v : sizes) {
                    if (v > 0) {
                        min = Math.min(min, v);
                        max = Math.max(max, v);
                    }
                }
                if (max > 0 && max <= Integer.MAX_VALUE) {
                    return new int[]{(int) max, (int) min, (int) max};
                }
            }
        }
        if (vendorId == PCI_VENDOR_NVIDIA || extensions.contains("cl_nv_device_attribute_query")) {
            int warp = extensions.contains("cl_nv_device_attribute_query")
                    ? queryOptionalUInt(deviceId, CL_DEVICE_WARP_SIZE_NV) : 0;
            int size = warp > 0 ? warp : NVIDIA_WARP_SIZE;
            return new int[]{size, size, size};
        }
        if (vendorId == PCI_VENDOR_AMD || extensions.contains("cl_amd_device_attribute_query")) {
            int wave = extensions.contains("cl_amd_device_attribute_query")
                    ? queryOptionalUInt(deviceId, CL_DEVICE_WAVEFRONT_WIDTH_AMD) : 0;
            int size = wave > 0 ? wave : AMD_WAVEFRONT_SIZE;
            return new int[]{size, size, size};
        }
        return new int[]{0, 0, 0};
    }

    /** A {@code cl_uint} vendor attribute, or 0 when the driver rejects the query. */
    private static int queryOptionalUInt(cl_device_id deviceId, int paramName) {
        int[] value = new int[1];
        try {
            int status = clGetDeviceInfo(deviceId, paramName, org.jocl.Sizeof.cl_uint, Pointer.to(value), null);
            return status == CL_SUCCESS ? value[0] : 0;
        } catch (CLException e) {
            LOGGER.debug("clGetDeviceInfo(0x{}) unavailable: {}", Integer.toHexString(paramName),
                    stringFor_errorCode(e.getStatus()));
            return 0;
        }
    }

    /** A {@code size_t[]} attribute of driver-defined length, or null when the driver rejects the query. */
    private static long[] querySizeArray(cl_device_id deviceId, int paramName) {
        try {
            long[] bytes = new long[1];
            if (clGetDeviceInfo(deviceId, paramName, 0, null, bytes) != CL_SUCCESS
                    || bytes[0] <= 0 || bytes[0] % org.jocl.Sizeof.size_t != 0) {
                return null;
            }
            int n = (int) (bytes[0] / org.jocl.Sizeof.size_t);
            java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate((int) bytes[0])
                    .order(java.nio.ByteOrder.nativeOrder());
            if (clGetDeviceInfo(deviceId, paramName, bytes[0], Pointer.to(buffer), null) != CL_SUCCESS) {
                return null;
            }
            long[] values = new long[n];
            for (int i = 0; i < n; i++) {
                values[i] = org.jocl.Sizeof.size_t == 4
                        ? buffer.getInt(i * 4) & 0xFFFFFFFFL
                        : buffer.getLong(i * 8);
            }
            return values;
        } catch (CLException e) {
            LOGGER.debug("clGetDeviceInfo(0x{}) unavailable: {}", Integer.toHexString(paramName),
                    stringFor_errorCode(e.getStatus()));
            return null;
        }
    }

    private static String getStringDeviceInfo(cl_device_id deviceId, int paramName) {
        long[] size = new long[1];
        clGetDeviceInfo(deviceId, paramName, 0, null, size);
        if (size[0] <= 0) {
            return "";
        }
        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(deviceId, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length - 1);
    }

    private static int getIntDeviceInfo(cl_device_id deviceId, int paramName) {
        int[] values = new int[1];
        clGetDeviceInfo(deviceId, paramName, org.jocl.Sizeof.cl_int, Pointer.to(values), null);
        return values[0];
    }

    private static long getLongDeviceInfo(cl_device_id deviceId, int paramName) {
        long[] values = new long[1];
        clGetDeviceInfo(deviceId, paramName, org.jocl.Sizeof.cl_long, Pointer.to(values), null);
        return values[0];
    }

    private static long getSizeDeviceInfo(cl_device_id deviceId, int paramName) {
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(org.jocl.Sizeof.size_t)
                .order(java.nio.ByteOrder.nativeOrder());
        clGetDeviceInfo(deviceId, paramName, org.jocl.Sizeof.size_t, Pointer.to(buffer), null);
        if (org.jocl.Sizeof.size_t == 4) {
            return buffer.getInt(0);
        } else {
            return buffer.getLong(0);
        }
    }
}
