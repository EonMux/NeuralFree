package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;

import java.util.EnumSet;
import java.util.Objects;

/**
 * Ids and default registrations of the built-in kernel libraries.
 *
 * <p>The LLM library ({@link #LLM}) keeps its dual-dialect sources and {@code manifest.txt} under
 * {@code kernels/llm}, the CUDA fatbins under {@code kernels/llm/cuda} and the SPIR-V modules under
 * {@code shaders/llm}. CUDA and Vulkan require the complete kernel inventory, so backend selection never
 * accepts a backend that would miss a kernel at the first layer.</p>
 *
 * <p>The training library ({@link #TRAIN}) uses the same layout under {@code kernels/train},
 * {@code kernels/train/cuda} and {@code shaders/train}, also requires its complete inventory, and has its own
 * build profile: static OpenCL option {@code -cl-std=CL1.2} only (no contraction or signed-zero relaxation), no
 * subgroup variants (its reductions use a fixed-order tree on every backend), and only
 * {@link DeviceFeature#TIMELINE_SEMAPHORE} on Vulkan (sub-word data is accessed as 32-bit words, so no 8- or
 * 16-bit storage is needed).</p>
 */
public final class KernelLibraries {

    /** Id of the LLM inference kernel library. */
    public static final String LLM = "llm";

    /** Classpath layout of the LLM library. */
    public static final KernelLibraryConfig LLM_CONFIG =
            new KernelLibraryConfig(LLM, "kernels/llm", "kernels/llm/cuda", "shaders/llm", true);

    /** Id of the training and classical-ML kernel library. */
    public static final String TRAIN = "train";

    /** Classpath layout and build profile of the training library. */
    public static final KernelLibraryConfig TRAIN_CONFIG =
            new KernelLibraryConfig(TRAIN, "kernels/train", "kernels/train/cuda", "shaders/train", true,
                    "-cl-std=CL1.2", false, EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE));

    private static volatile boolean defaultsRegistered;

    private KernelLibraries() {
    }

    /**
     * Registers the OpenCL, CUDA and Vulkan libraries of {@link #LLM_CONFIG} and {@link #TRAIN_CONFIG} in
     * the global registry, keeping any library already registered under the same id and backend type.
     * Idempotent.
     */
    public static void ensureDefaults() {
        if (defaultsRegistered) {
            return;
        }
        synchronized (KernelLibraries.class) {
            if (!defaultsRegistered) {
                KernelLibraryRegistry r = KernelLibraryRegistry.global();
                for (KernelLibraryConfig config : new KernelLibraryConfig[]{LLM_CONFIG, TRAIN_CONFIG}) {
                    r.registerIfAbsent(new OpenClKernelLibrary(config));
                    r.registerIfAbsent(new CudaKernelLibrary(config));
                    r.registerIfAbsent(new VulkanKernelLibrary(config));
                }
                defaultsRegistered = true;
            }
        }
    }

    /**
     * Registers the OpenCL, CUDA and Vulkan libraries of a configuration in the global registry,
     * replacing the libraries registered under the same id. Kernel sets opened afterwards use them.
     *
     * @throws IllegalArgumentException when the id is already registered with another OpenCL source root
     *                                  (see {@link KernelLibraryRegistry#register}); nothing is replaced then
     */
    public static void register(KernelLibraryConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        synchronized (KernelLibraries.class) {
            ensureDefaults();
            KernelLibraryRegistry r = KernelLibraryRegistry.global();
            r.register(new OpenClKernelLibrary(config));
            r.register(new CudaKernelLibrary(config));
            r.register(new VulkanKernelLibrary(config));
        }
    }
}
