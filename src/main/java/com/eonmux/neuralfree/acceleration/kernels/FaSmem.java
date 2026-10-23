package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;

import java.util.Objects;

/**
 * Size of the static shared-memory array of the flash-attention kernels ({@code NF_FA_SMEM_FLOATS}),
 * in floats, per backend and device.
 *
 * <ul>
 *   <li>CUDA: {@value #MAX_FLOATS}, the value the fatbins are compiled with (48 KiB static shared
 *       memory minus 4 KiB for the kernels' other shared arrays).</li>
 *   <li>OpenCL and Vulkan: {@code min(MAX_FLOATS, sharedBytes/4 - RESERVED_FLOATS)} with a floor of
 *       {@value #MIN_FLOATS}; OpenCL passes it as {@code -DNF_FA_SMEM_FLOATS}, Vulkan as specialisation
 *       constant {@value #VULKAN_SPEC_ID}.</li>
 * </ul>
 */
public final class FaSmem {

    /** Upper bound and CUDA value, in floats (44 KiB). */
    public static final int MAX_FLOATS = 11264;
    /** Lower bound, in floats (12 KiB); fits the 16 KiB Vulkan minimum with the reserve. */
    public static final int MIN_FLOATS = 3072;
    /** Shared memory kept free for the flash kernels' other shared arrays, in floats (4 KiB). */
    public static final int RESERVED_FLOATS = 1024;
    /** Vulkan specialisation constant id that carries the value. */
    public static final int VULKAN_SPEC_ID = 3;

    private FaSmem() {
    }

    /**
     * {@code NF_FA_SMEM_FLOATS} for a backend type and device.
     *
     * @throws IllegalArgumentException when an argument is null
     */
    public static int faSmemFloats(ComputeBackend.BackendType type, DeviceCapabilities caps) {
        if (type == null || caps == null) {
            throw new IllegalArgumentException("backend type and capabilities must not be null");
        }
        if (type == ComputeBackend.BackendType.CUDA) {
            return MAX_FLOATS;
        }
        long sharedFloats = Math.max(0L, caps.getMaxSharedMemoryBytes()) / 4L - RESERVED_FLOATS;
        long n = Math.min((long) MAX_FLOATS, sharedFloats);
        return (int) Math.max((long) MIN_FLOATS, n);
    }

    /** Same as {@link #faSmemFloats(ComputeBackend.BackendType, DeviceCapabilities)} for a backend. */
    public static int faSmemFloats(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        return faSmemFloats(backend.backendType(), backend.capabilities());
    }
}
