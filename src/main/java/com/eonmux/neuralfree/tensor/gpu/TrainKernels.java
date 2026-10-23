package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryRequirement;

/**
 * The {@code train} kernel library: training, tensor and classical-ML kernels with identical names and semantics on
 * OpenCL, CUDA and Vulkan. Kernels are reached only through {@link TrainGpu}, which opens the library once per
 * backend.
 */
public final class TrainKernels {

    /** Id of the train kernel library. */
    public static final String LIBRARY = KernelLibraries.TRAIN;

    private TrainKernels() {
    }

    /** Backend-selection requirement that the train library is registered and available on the backend. */
    public static KernelLibraryRequirement requirement() {
        return KernelLibraryRequirement.of(LIBRARY);
    }

    /**
     * Whether tensor ops can run on the backend's train kernels, with the reason when they cannot; the cached result
     * of {@link TrainGpu#forBackend} (opening and prewarming the library on the first call).
     */
    public static Availability availability(ComputeBackend b) {
        return TrainGpu.availability(b);
    }
}
