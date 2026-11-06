package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;

import java.util.function.Supplier;

/**
 * One Metal kernel (a function of a compiled MSL library) plus its sticky launch arguments.
 *
 * <p>The pipelines are owned by the backend's pipeline cache, so no handle owns a framework
 * resource:</p>
 * <ul>
 *   <li>a <b>backend-owned</b> handle (from {@code loadKernel} or {@code compileKernel}) is cached
 *       and shared; its {@link #close()} does nothing, and it becomes unusable when the backend
 *       closes;</li>
 *   <li>a <b>clone</b> (from {@code cloneKernel}) shares the pipelines but has its own arguments;
 *       its {@link #close()} retires it.</li>
 * </ul>
 * A handle loaded from a kernel library carries the kernel's build-time {@link KernelSignature},
 * which every launch's arguments are checked against ({@link MetalArgState}); clones share it.
 *
 * <p><b>One pipeline per work-group size.</b> The generated MSL takes its local size from
 * specialisation constants 0-2, as MSL function constants or, when they appear in constant
 * expressions, as the {@code SPIRV_CROSS_CONSTANT_ID_*} compile macros. A pipeline is therefore
 * built for the threadgroup a dispatch runs with: the {@link MetalComputeBackend#prepareKernel
 * prepared} size at model bind, any other size on its first dispatch; the macro form additionally
 * recompiles the library for each size. Kernels without specialisation constants have one pipeline
 * for every size. {@code NF_FA_SMEM_FLOATS} (id 3) is a device property of the backend, so it does
 * not add pipeline variants.</p>
 */
public final class MetalKernelHandle implements KernelHandle {

    private final MetalComputeBackend backend;
    private final String programKey;
    private final String kernelName;
    private final String options;
    private final Supplier<String> source;
    private final SpecConstants constants;
    private final boolean subgroupOps;
    private final boolean backendOwned;
    private final KernelSignature signature;
    private final MetalArgState args;
    private final int maxThreadsPerThreadgroup;
    private volatile boolean closed;

    /**
     * @param programKey               cache key of the compiled MSL library
     * @param source                   supplier of the MSL text
     * @param options                  Metal compile options
     * @param constants                the specialisation constants the source declares
     * @param subgroupOps              whether the MSL uses SIMD-group builtins or reductions
     * @param signature                build-time argument signature, or null when none is known
     * @param maxThreadsPerThreadgroup the device's largest threadgroup
     */
    MetalKernelHandle(MetalComputeBackend backend, String programKey, String kernelName,
                      Supplier<String> source, String options, SpecConstants constants, boolean subgroupOps,
                      KernelSignature signature, int maxThreadsPerThreadgroup) {
        if (backend == null) {
            throw new IllegalArgumentException("Metal backend must not be null");
        }
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernelName must not be null or empty");
        }
        if (source == null || constants == null) {
            throw new IllegalArgumentException("kernel " + kernelName + " needs its MSL source and constants");
        }
        this.backend = backend;
        this.programKey = programKey;
        this.kernelName = kernelName;
        this.source = source;
        this.options = options == null ? "" : options;
        this.constants = constants;
        this.subgroupOps = subgroupOps;
        this.backendOwned = true;
        this.maxThreadsPerThreadgroup = maxThreadsPerThreadgroup;
        this.signature = signature;
        this.args = new MetalArgState(kernelName, signature);
    }

    /** A caller-owned clone of {@code template}: same pipelines, fresh arguments. */
    private MetalKernelHandle(MetalKernelHandle template) {
        this.backend = template.backend;
        this.programKey = template.programKey;
        this.kernelName = template.kernelName;
        this.source = template.source;
        this.options = template.options;
        this.constants = template.constants;
        this.subgroupOps = template.subgroupOps;
        this.backendOwned = false;
        this.maxThreadsPerThreadgroup = template.maxThreadsPerThreadgroup;
        this.signature = template.signature;
        this.args = new MetalArgState(template.kernelName, template.signature);
    }

    /** A caller-owned handle sharing this handle's pipelines, with no arguments set. */
    MetalKernelHandle cloneHandle() {
        ensureUsable();
        return new MetalKernelHandle(this);
    }

    @Override
    public String kernelName() {
        return kernelName;
    }

    /** The backend that compiled the library. */
    public MetalComputeBackend backend() {
        return backend;
    }

    /**
     * The pipeline for a dispatch with this local work size, built on first use and cached by the
     * backend.
     *
     * @param localWorkSize threadgroup of the dispatch, one to three dimensions
     * @throws IllegalStateException when this clone or the backend was closed
     */
    public long pipeline(long[] localWorkSize) {
        ensureUsable();
        return backend.pipeline(programKey, kernelName, source, options, constants, localWorkSize);
    }

    /** Builds and returns the pipeline for this work-group size now, so the first dispatch does not. */
    long prepare(long[] localWorkSize) {
        return pipeline(localWorkSize);
    }

    /**
     * Rejects a threadgroup a SIMD-group kernel cannot run: Metal forms {@value
     * MetalCapabilitiesBuilder#SIMD_GROUP_SIZE}-lane SIMD groups from consecutive threads, so the x
     * dimension of a kernel that reduces across a subgroup must be a multiple of that width.
     *
     * @throws IllegalArgumentException naming the kernel and the threadgroup when it is not
     */
    void requireValidThreadgroup(long[] localWorkSize) {
        if (!subgroupOps) {
            return;
        }
        long x = localWorkSize[0];
        if (x % MetalCapabilitiesBuilder.SIMD_GROUP_SIZE != 0) {
            throw new IllegalArgumentException("kernel " + kernelName + " uses SIMD-group reductions; the"
                    + " threadgroup x dimension must be a multiple of "
                    + MetalCapabilitiesBuilder.SIMD_GROUP_SIZE + ", got " + x);
        }
    }

    /**
     * Whether an MSL translation unit uses SIMD-group operations: SPIRV-Cross emits the SIMD-group
     * builtins and {@code simd_*} calls for the Vulkan subgroup builtins and group operations.
     *
     * @param msl MSL source of one kernel
     */
    static boolean usesSubgroups(String msl) {
        return msl.contains("thread_index_in_simdgroup") || msl.contains("threads_per_simdgroup")
                || msl.contains("simdgroup_index_in_threadgroup") || msl.contains("simdgroups_per_threadgroup")
                || msl.contains("simd_");
    }

    /** Build-time argument signature, or null when the kernel's arguments are not checked. */
    public KernelSignature signature() {
        return signature;
    }

    /** The sticky launch arguments of this handle. */
    public MetalArgState args() {
        return args;
    }

    /**
     * Largest threadgroup the device can launch. The pipelines this handle builds do not lower it:
     * their own bound is checked against each dispatch after they are built.
     */
    public int maxThreadsPerThreadgroup() {
        return maxThreadsPerThreadgroup;
    }

    /** True for cached handles released by the backend, false for caller-owned clones. */
    public boolean isBackendOwned() {
        return backendOwned;
    }

    /** True once this clone was closed, or once the backend was closed. */
    public boolean isClosed() {
        return closed || backend.isClosed();
    }

    /**
     * Retires a clone. A backend-owned handle stays valid until the backend closes, because other
     * callers share it through the kernel cache. Idempotent.
     */
    @Override
    public void close() {
        if (!backendOwned) {
            closed = true;
        }
    }

    @Override
    public String toString() {
        return "MetalKernelHandle{" + kernelName + (backendOwned ? "" : ", clone")
                + (constants.isEmpty() ? "" : ", functionConstants")
                + (isClosed() ? ", closed" : "") + "}";
    }

    private void ensureUsable() {
        if (closed) {
            throw new IllegalStateException("Metal kernel " + kernelName
                    + " was used after its clone handle was closed");
        }
        if (backend.isClosed()) {
            throw new IllegalStateException("Metal kernel " + kernelName + " was used after its backend was closed");
        }
    }
}
