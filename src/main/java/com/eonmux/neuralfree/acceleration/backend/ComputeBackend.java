package com.eonmux.neuralfree.acceleration.backend;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;

import java.util.Locale;

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

    /**
     * The API a backend is driven through. {@link #OPENCL}, {@link #CUDA}, {@link #VULKAN}, {@link #HIP}
     * and {@link #METAL} are implemented by the engine; the rest are accelerator APIs whose backends come
     * from an {@link com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorBackendProvider}
     * ({@link #isAccelerator()}).
     */
    enum BackendType {
        OPENCL("OpenCL", false),
        CUDA("CUDA", false),
        VULKAN("Vulkan", false),
        HIP("HIP", false),
        /** Apple GPUs through Metal (MSL kernels compiled by the Metal framework). */
        METAL("Metal", false),
        /** SYCL, driven through oneAPI and its Level Zero runtime. */
        SYCL("SYCL", true),
        /** Intel XPU (vLLM's {@code xpu} platform). */
        XPU("XPU", true),
        /** Huawei Ascend (llama.cpp's {@code ggml-cann}). */
        CANN("CANN", true),
        /** Qualcomm Hexagon (llama.cpp's {@code ggml-hexagon}). */
        HEXAGON("Hexagon", true),
        /** Intel OpenVINO (llama.cpp's {@code ggml-openvino}). */
        OPENVINO("OpenVINO", true),
        /** WebGPU (llama.cpp's {@code ggml-webgpu}). */
        WEBGPU("WebGPU", true),
        /** Moore Threads MUSA (llama.cpp's {@code ggml-musa}). */
        MUSA("MUSA", true),
        /** IBM Z zDNN (llama.cpp's {@code ggml-zdnn}). */
        ZDNN("zDNN", true),
        /** AMD ZenDNN (llama.cpp's {@code ggml-zendnn}). */
        ZENDNN("ZenDNN", true);

        private final String label;
        private final boolean accelerator;

        BackendType(String label, boolean accelerator) {
            this.label = label;
            this.accelerator = accelerator;
        }

        /** The name users read: {@code OpenCL}, {@code CUDA}, {@code SYCL}, {@code zDNN}, ... */
        public String label() {
            return label;
        }

        /** True for an API served by an installed {@code AcceleratorBackendProvider}, not by the engine. */
        public boolean isAccelerator() {
            return accelerator;
        }

        /**
         * The API a user names, case-insensitively: the canonical name ({@code opencl}, {@code metal},
         * {@code sycl}, {@code webgpu}, {@code zdnn}, {@code zendnn}) or a runtime spelling
         * ({@code level-zero}, {@code level_zero}, {@code web-gpu}, {@code z-dnn}, {@code zen-dnn}).
         *
         * @throws IllegalArgumentException naming the accepted APIs when {@code name} matches none
         */
        public static BackendType fromApiName(String name) {
            if (name == null) {
                throw new IllegalArgumentException("no backend API was named; use " + acceptedApiNames());
            }
            String key = name.trim().toLowerCase(Locale.ROOT).replace('_', '-');
            switch (key) {
                case "opencl":
                    return OPENCL;
                case "cuda":
                    return CUDA;
                case "vulkan":
                    return VULKAN;
                case "hip":
                    return HIP;
                case "metal":
                    return METAL;
                case "sycl":
                case "level-zero":
                case "levelzero":
                    return SYCL;
                case "xpu":
                    return XPU;
                case "cann":
                    return CANN;
                case "hexagon":
                    return HEXAGON;
                case "openvino":
                    return OPENVINO;
                case "webgpu":
                case "web-gpu":
                    return WEBGPU;
                case "musa":
                    return MUSA;
                case "zdnn":
                case "z-dnn":
                    return ZDNN;
                case "zendnn":
                case "zen-dnn":
                    return ZENDNN;
                default:
                    throw new IllegalArgumentException("'" + name + "' names no backend API; use "
                            + acceptedApiNames());
            }
        }

        /** The API names users may type, for error messages. */
        public static String acceptedApiNames() {
            return "opencl, cuda, vulkan, hip, metal, sycl (level-zero), xpu, cann, hexagon, openvino,"
                    + " webgpu, musa, zdnn or zendnn";
        }
    }

    BackendType backendType();

    DeviceCapabilities capabilities();

    ComputeQueue computeQueue();

    ComputeQueue transferQueue();

    GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly);

    GpuBuffer allocateBuffer(long sizeInBytes);

    /**
     * True when {@link #allocateHostMappedBuffer(long)} works on this backend: CUDA mapped host
     * memory, Vulkan host-visible memory, or a Metal shared-storage buffer. The default is false;
     * OpenCL and HIP have no zero-copy weight path in this engine.
     */
    default boolean supportsHostMappedBuffers() {
        return false;
    }

    /**
     * Allocates page-locked host memory the device reads in place ({@link HostMappedBuffer}). The
     * allocation is not device memory: it counts against neither {@link #getAllocatedBytes()} nor the
     * device's VRAM.
     *
     * @throws UnsupportedOperationException when {@link #supportsHostMappedBuffers()} is false
     * @throws IllegalArgumentException      when the size is not positive
     */
    default HostMappedBuffer allocateHostMappedBuffer(long sizeInBytes) {
        throw new UnsupportedOperationException(backendType() + " backend cannot allocate host-mapped weight"
                + " memory; use the streamed tier instead of the UVA offload");
    }

    /**
     * Compiles or loads a kernel from backend-specific text: OpenCL C source on OpenCL,
     * PTX text on CUDA, and a classpath {@code .spv} path on Vulkan. Metal compiles the
     * given Metal Shading Language source. HIP takes no source text: its kernels are
     * hipcc-built code objects, loaded through {@link #loadKernel(KernelArtifact)}.
     */
    KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions);

    /**
     * Loads a kernel described by a resolved artifact. The handle is backend-owned and cached.
     *
     * <p>The default serves OpenCL source artifacts through {@link #compileKernel}; backends
     * override it to load their binary and Metal source kinds and to key their program cache by
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
        if (kind == KernelArtifact.Kind.OPENCL_SOURCE) {
            return backendType() == BackendType.OPENCL;
        }
        return kind == KernelArtifact.Kind.METAL_SOURCE && backendType() == BackendType.METAL;
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
