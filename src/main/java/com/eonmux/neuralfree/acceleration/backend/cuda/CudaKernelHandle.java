package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import jcuda.driver.CUfunction;
import jcuda.driver.CUfunction_attribute;

import static jcuda.driver.JCudaDriver.cuFuncGetAttribute;

/**
 * One CUDA kernel function plus its sticky launch arguments.
 *
 * <p>The function belongs to a module owned by the backend's {@link CudaModuleCache},
 * so no handle owns driver resources:</p>
 * <ul>
 *   <li>a <b>backend-owned</b> handle (from {@code compileKernel} or {@code loadKernel}) is
 *       cached and shared; its {@link #close()} does nothing, and it becomes unusable when
 *       the backend closes;</li>
 *   <li>a <b>clone</b> (from {@code cloneKernel}) shares the function but has its own
 *       arguments; its {@link #close()} retires it.</li>
 * </ul>
 * The function attributes the queue validates launches against are read once. A handle loaded
 * from a kernel library carries the kernel's build-time {@link KernelSignature}, which every
 * launch's arguments are checked against ({@link CudaArgState}); clones share it.
 */
public final class CudaKernelHandle implements KernelHandle {

    private final CudaComputeBackend backend;
    private final CUfunction function;
    private final String kernelName;
    private final boolean backendOwned;
    private final int maxThreadsPerBlock;
    private final int staticSharedMemoryBytes;
    private final KernelSignature signature;
    private final CudaArgState args;
    private volatile boolean closed;

    /**
     * Creates a backend-owned handle and reads the function attributes. The backend's
     * context must be current.
     *
     * @param signature build-time argument signature, or null when none is known (launches are then unchecked)
     * @throws IllegalArgumentException when an argument other than the signature is null or the name is empty
     * @throws RuntimeException         the translated driver failure of an attribute query
     */
    CudaKernelHandle(CudaComputeBackend backend, CUfunction function, String kernelName, KernelSignature signature) {
        if (backend == null) {
            throw new IllegalArgumentException("CUDA backend must not be null");
        }
        if (function == null) {
            throw new IllegalArgumentException("CUfunction must not be null");
        }
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernelName must not be null or empty");
        }
        this.backend = backend;
        this.function = function;
        this.kernelName = kernelName;
        this.backendOwned = true;
        this.maxThreadsPerBlock = attribute(function, CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                kernelName);
        this.staticSharedMemoryBytes = attribute(function, CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                kernelName);
        this.signature = signature;
        this.args = new CudaArgState(backend, kernelName, signature);
    }

    /** A caller-owned clone of {@code template}: same function and attributes, fresh arguments. */
    private CudaKernelHandle(CudaKernelHandle template) {
        this.backend = template.backend;
        this.function = template.function;
        this.kernelName = template.kernelName;
        this.backendOwned = false;
        this.maxThreadsPerBlock = template.maxThreadsPerBlock;
        this.staticSharedMemoryBytes = template.staticSharedMemoryBytes;
        this.signature = template.signature;
        this.args = new CudaArgState(template.backend, template.kernelName, template.signature);
    }

    /** A caller-owned handle sharing this handle's function, with no arguments set. */
    CudaKernelHandle cloneHandle() {
        ensureUsable();
        return new CudaKernelHandle(this);
    }

    @Override
    public String kernelName() {
        return kernelName;
    }

    /** The backend that loaded the function. */
    public CudaComputeBackend backend() {
        return backend;
    }

    /**
     * The function to launch.
     *
     * @throws IllegalStateException when this clone was closed or the backend was closed
     */
    public CUfunction function() {
        ensureUsable();
        return function;
    }

    /** Build-time argument signature, or null when the kernel's arguments are not checked. */
    public KernelSignature signature() {
        return signature;
    }

    /** The sticky launch arguments of this handle. */
    public CudaArgState args() {
        return args;
    }

    /** {@code CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK}: the largest block this function can launch with. */
    public int maxThreadsPerBlock() {
        return maxThreadsPerBlock;
    }

    /** {@code CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES}: statically allocated shared memory per block. */
    public int staticSharedMemoryBytes() {
        return staticSharedMemoryBytes;
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
     * Retires a clone. A backend-owned handle stays valid until the backend closes, because
     * other callers share it through the kernel cache. Idempotent.
     */
    @Override
    public void close() {
        if (!backendOwned) {
            closed = true;
        }
    }

    @Override
    public String toString() {
        return "CudaKernelHandle{" + kernelName + (backendOwned ? "" : ", clone") + (isClosed() ? ", closed" : "") + "}";
    }

    private void ensureUsable() {
        if (closed) {
            throw new IllegalStateException("CUDA kernel " + kernelName + " was used after its clone handle was closed");
        }
        if (backend.isClosed()) {
            throw new IllegalStateException("CUDA kernel " + kernelName + " was used after its backend was closed");
        }
    }

    private static int attribute(CUfunction function, int attribute, String kernelName) {
        int[] value = new int[1];
        CudaErrors.check("cuFuncGetAttribute(" + CUfunction_attribute.stringFor(attribute) + ", " + kernelName + ")",
                () -> cuFuncGetAttribute(value, attribute, function));
        return value[0];
    }
}
