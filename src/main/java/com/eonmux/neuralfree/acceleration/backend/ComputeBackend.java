package com.eonmux.neuralfree.acceleration.backend;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;

/**
 * One device opened through one GPU API.
 *
 * <p>Kernel handles returned by {@link #compileKernel} and {@link #loadKernel} are
 * owned and cached by the backend and released by {@link #close()}; clones from
 * {@link #cloneKernel} are owned by the caller and never release the program or
 * module they share. The dispatch, argument, event and transfer contract is
 * documented on {@link ComputeQueue}.</p>
 */
public interface ComputeBackend extends AutoCloseable {
    enum BackendType {
        OPENCL,
        CUDA,
        VULKAN
    }

    BackendType backendType();

    DeviceCapabilities capabilities();

    ComputeQueue computeQueue();

    ComputeQueue transferQueue();

    GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly);

    GpuBuffer allocateBuffer(long sizeInBytes);

    /**
     * Compiles or loads a kernel from backend-specific text: OpenCL C source on OpenCL,
     * PTX text on CUDA, and a classpath {@code .spv} path on Vulkan.
     */
    KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions);

    /**
     * Loads a kernel described by a resolved artifact. The handle is backend-owned and cached.
     *
     * <p>The default serves OpenCL source artifacts through {@link #compileKernel}; backends
     * override it to load their binary kinds and to key their program cache by
     * {@link KernelArtifact#programKey()}.</p>
     *
     * @throws UnsupportedKernelException when this backend cannot load the artifact's kind
     */
    default KernelHandle loadKernel(KernelArtifact artifact) {
        if (artifact == null) {
            throw new IllegalArgumentException("kernel artifact must not be null");
        }
        if (artifact.kind() == KernelArtifact.Kind.OPENCL_SOURCE && backendType() == BackendType.OPENCL) {
            return compileKernel(artifact.kernelName(), artifact.sourceCode(), artifact.buildOptions());
        }
        throw new UnsupportedKernelException(backendType() + " backend cannot load " + artifact.kind()
                + " artifact for kernel " + artifact.kernelName());
    }

    /** True when {@link #loadKernel(KernelArtifact)} accepts artifacts of this kind. */
    default boolean supportsArtifact(KernelArtifact.Kind kind) {
        return kind == KernelArtifact.Kind.OPENCL_SOURCE && backendType() == BackendType.OPENCL;
    }

    /**
     * Optional warm-up for a kernel that will be dispatched with this local size: Vulkan
     * creates the pipeline now instead of at the first dispatch. The default does nothing.
     */
    default void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
    }

    /**
     * Creates a new kernel object from the same compiled program as the template.
     * The clone has independent argument state, allowing pre-binding of per-layer
     * arguments. The clone does NOT own the program — only the template does.
     */
    default KernelHandle cloneKernel(KernelHandle template) {
        throw new UnsupportedOperationException("cloneKernel not implemented");
    }

    /**
     * Creates an additional compute queue on the same device and context.
     * Used for multi-queue concurrent dispatch (e.g., QKV on separate queues).
     * The caller is responsible for closing the returned queue.
     */
    default ComputeQueue createComputeQueue() {
        throw new UnsupportedOperationException("createComputeQueue not implemented");
    }

    /**
     * Returns the total number of bytes currently allocated on the device via
     * {@link #allocateBuffer}. Implementations that track allocations should
     * increment on allocation and decrement on buffer release.
     */
    default long getAllocatedBytes() { return 0; }

    /**
     * Returns the actual free VRAM on the device in bytes, accounting for
     * memory used by other processes (display compositor, other apps, etc.).
     * Falls back to {@code totalVram - allocatedBytes} if unsupported.
     */
    default long getFreeVramBytes() {
        return capabilities().getTotalVramBytes() - getAllocatedBytes();
    }

    void close();
}
