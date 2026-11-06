package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.metal.MetalDevice;
import com.eonmux.neuralfree.acceleration.metal.MetalNative;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Set;

/**
 * Builds {@link DeviceCapabilities} for a Metal device: working set, buffer limits, threadgroup
 * limits, subgroup width, and the features MSL kernels can rely on.
 *
 * <ul>
 *   <li><b>Memory.</b> {@code recommendedMaxWorkingSetSize} is the framework's VRAM estimate and
 *       becomes the total; {@code maxBufferLength} bounds one allocation. Metal has no free-memory
 *       query, so the backend reports {@code total - allocated}.</li>
 *   <li><b>Subgroups.</b> MSL SIMD groups are {@value #SIMD_GROUP_SIZE} lanes wide on every Apple
 *       GPU, and the kernels may use {@code simd_*} reductions.</li>
 *   <li><b>Storage.</b> Metal buffers are untyped byte memory, so 8- and 16-bit loads and stores,
 *       64-bit integers and half precision are always available; double precision never is.</li>
 * </ul>
 */
public final class MetalCapabilitiesBuilder {

    /** SIMD-group (subgroup) width of an Apple GPU. */
    public static final int SIMD_GROUP_SIZE = 32;

    /** Metal threadgroups have no separately reported count limit; the framework sizes them by NSUInteger. */
    private static final long MAX_THREADGROUPS = Integer.MAX_VALUE;

    private static final Set<String> ACTIVATION_FUNCTIONS = new HashSet<>(Arrays.asList(
            "SIGMOID", "RELU", "TANH", "LEAKY_RELU", "ELU", "SOFTPLUS", "SOFTSIGN",
            "ISRU", "GAUSSIAN", "SINC", "SWISH", "SELU", "HARD_SIGMOID",
            "BENT_IDENTITY", "PRELU", "SRELU", "SOFTEXPONENTIAL", "IDENTITY"));
    private static final Set<String> COST_FUNCTIONS = new HashSet<>(Arrays.asList(
            "MSE", "MAE", "HUBER", "LOG_COSH", "KL_DIVERGENCE", "CROSS_ENTROPY"));
    private static final Set<String> BLAS_LEVELS = new HashSet<>(Arrays.asList("1", "2", "3"));

    private MetalCapabilitiesBuilder() {
    }

    /**
     * Queries the device; needs no device state.
     *
     * @throws IllegalArgumentException when {@code device} is null
     * @throws RuntimeException         the translated framework failure of a required query
     */
    public static DeviceCapabilities build(MetalDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("MetalDevice must not be null");
        }
        long maxThreads = device.getMaxThreadsPerThreadgroup();
        long threadgroupMemory = device.getMaxThreadgroupMemory();
        long workingSet = device.getRecommendedMaxWorkingSet();
        long maxBuffer = device.getMaxBufferLength();
        EnumSet<DeviceFeature> features = EnumSet.of(
                DeviceFeature.SUBGROUP_ARITHMETIC,
                DeviceFeature.STORAGE_8BIT,
                DeviceFeature.STORAGE_16BIT,
                DeviceFeature.SHADER_INT8,
                DeviceFeature.SHADER_INT64,
                DeviceFeature.SHADER_FLOAT16,
                DeviceFeature.CONCURRENT_QUEUES_PROFITABLE);
        return new DeviceCapabilities.Builder()
                .deviceName(device.getDeviceName())
                .totalVramBytes(Math.max(0L, workingSet))
                .maxAllocSizeBytes(maxBuffer > 0 ? maxBuffer : Math.max(0L, workingSet))
                .computeUnits(DeviceCapabilities.UNKNOWN_COMPUTE_UNITS)
                .maxClockFrequencyMhz(0)
                .maxWorkGroupSize(maxThreads)
                .supportsAsyncTransfer(true)
                .driverVersion(driverVersion(device))
                .supportsFloat64(false)
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(threadgroupMemory)
                .maxSharedMemoryBytes(threadgroupMemory)
                .maxStorageBufferRange(maxBuffer > 0 ? maxBuffer : Math.max(0L, workingSet))
                .supportedActivationFunctions(ACTIVATION_FUNCTIONS)
                .supportedCostFunctions(COST_FUNCTIONS)
                .supportedBLASLevels(BLAS_LEVELS)
                .deviceClass(device.getDeviceClass())
                .identity(device.getIdentity())
                .subgroupSize(SIMD_GROUP_SIZE)
                .minSubgroupSize(SIMD_GROUP_SIZE)
                .maxSubgroupSize(SIMD_GROUP_SIZE)
                .maxWorkGroupCount(new long[]{MAX_THREADGROUPS, MAX_THREADGROUPS, MAX_THREADGROUPS})
                .features(features)
                .build();
    }

    /** {@code <driver release>; <GPU family>}, for logs and selection reasons. */
    private static String driverVersion(MetalDevice device) {
        String release = MetalNative.driverVersion();
        if (release == null || release.trim().isEmpty()) {
            release = "unknown";
        }
        return release.trim() + "; " + device.getDeviceFamily();
    }
}
