package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;

/**
 * Builds a {@link GpuOps} with an explicit fused-kernel selection for tests outside this package, which cannot reach
 * the package-private constructor the tests of this package pin the modes with. Production builds read
 * {@link FusedMode#configured()}.
 */
public final class GpuOpsFixture {

    private GpuOpsFixture() {
    }

    /** A {@link GpuOps} on {@code queue} whose fused kernels are selected by {@code mode}. */
    public static GpuOps withFusedMode(ComputeQueue queue, LlmKernels kernels, GpuScratch scratch,
                                       GpuPlaceholders placeholders, FusedMode mode) {
        return new GpuOps(queue, kernels, scratch, placeholders, Int8MatvecMode.configured(), mode);
    }
}
