package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import com.eonmux.neuralfree.acceleration.hip.HipGfxTarget;
import com.eonmux.neuralfree.acceleration.hip.HipNative;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Set;

/**
 * Builds {@link DeviceCapabilities} for a HIP device from the runtime properties and the
 * instruction sets of its {@link HipGfxTarget}: VRAM, limits, the AMD compute capability
 * ({@code 0x1000000 | gfx}), wavefront size, features and identity.
 *
 * <p>The feature mapping follows llama.cpp's per-target feature predicates: {@code dp4a} on
 * Vega 20, RDNA2 and everything newer, the MFMA and WMMA matrix instructions on CDNA and RDNA3+
 * respectively, and bfloat16 matrix arithmetic on CDNA and RDNA3+. The wavefront size (64 on
 * CDNA and GCN, 32 on RDNA) is what the reduction and flash kernels are tuned by.</p>
 */
public final class HipCapabilitiesBuilder {

    /** Static shared memory per block available without an opt-in carve-out. */
    public static final long STATIC_SHARED_MEMORY_BYTES = 48L * 1024;

    /** Kernel parameter space per launch on every supported architecture. */
    public static final int KERNEL_PARAM_BYTES = 4096;

    private static final Set<String> ACTIVATION_FUNCTIONS = new HashSet<>(Arrays.asList(
            "SIGMOID", "RELU", "TANH", "LEAKY_RELU", "ELU", "SOFTPLUS", "SOFTSIGN",
            "ISRU", "GAUSSIAN", "SINC", "SWISH", "SELU", "HARD_SIGMOID",
            "BENT_IDENTITY", "PRELU", "SRELU", "SOFTEXPONENTIAL", "IDENTITY"));
    private static final Set<String> COST_FUNCTIONS = new HashSet<>(Arrays.asList(
            "MSE", "MAE", "HUBER", "LOG_COSH", "KL_DIVERGENCE", "CROSS_ENTROPY"));
    private static final Set<String> BLAS_LEVELS = new HashSet<>(Arrays.asList("1", "2", "3"));

    private HipCapabilitiesBuilder() {
    }

    /**
     * Queries the device; needs no device state.
     *
     * @throws IllegalArgumentException when {@code device} is null
     * @throws RuntimeException         the translated driver failure of a required query
     */
    public static DeviceCapabilities build(HipDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("HipDevice must not be null");
        }
        HipGfxTarget target = device.getGfxTarget();
        int warp = device.getAttribute(HipNative.ATTR_WARP_SIZE);
        long sharedPerBlock = device.getAttribute(HipNative.ATTR_SHARED_PER_BLOCK);
        long sharedOptIn = Math.max(sharedPerBlock, device.getAttribute(HipNative.ATTR_SHARED_PER_BLOCK_OPTIN));

        return new DeviceCapabilities.Builder()
                .deviceName(device.getDeviceName())
                .totalVramBytes(device.getDriver().totalMem(device.getIndex()))
                .computeUnits(device.getAttribute(HipNative.ATTR_MULTIPROCESSOR_COUNT))
                .maxClockFrequencyMhz(device.getAttribute(HipNative.ATTR_CLOCK_RATE) / 1000)
                .maxWorkGroupSize(device.getAttribute(HipNative.ATTR_MAX_THREADS_PER_BLOCK))
                .supportsAsyncTransfer(device.getAttribute(HipNative.ATTR_ASYNC_ENGINE_COUNT) >= 1)
                .driverVersion(describeDriver(device.getDriver().driverVersion()))
                .supportsFloat64(true)
                .supportsFloat16(target.float16Available())
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(sharedOptIn)
                .supportedActivationFunctions(ACTIVATION_FUNCTIONS)
                .supportedCostFunctions(COST_FUNCTIONS)
                .supportedBLASLevels(BLAS_LEVELS)
                .deviceClass(device.getDeviceClass())
                .identity(device.getIdentity())
                .subgroupSize(warp)
                .maxWorkGroupCount(new long[]{
                        device.getAttribute(HipNative.ATTR_MAX_GRID_DIM_X),
                        device.getAttribute(HipNative.ATTR_MAX_GRID_DIM_Y),
                        device.getAttribute(HipNative.ATTR_MAX_GRID_DIM_Z)})
                .maxPushConstantBytes(KERNEL_PARAM_BYTES)
                .maxSharedMemoryBytes(Math.min(STATIC_SHARED_MEMORY_BYTES, sharedPerBlock))
                .computeCapability(target.amdComputeCapability())
                .features(features(target))
                .build();
    }

    /** {@code "HIP runtime 6.2"} for 6020. */
    public static String describeDriver(int version) {
        return "HIP runtime " + (version / 1000) + "." + ((version % 1000) / 10);
    }

    /** The features of one gfx target, the llama.cpp instruction-set predicates. */
    public static Set<DeviceFeature> features(HipGfxTarget target) {
        Set<DeviceFeature> features = EnumSet.of(
                DeviceFeature.SUBGROUP_ARITHMETIC,
                DeviceFeature.SUBGROUP_SHUFFLE,
                DeviceFeature.SHADER_INT8,
                DeviceFeature.SHADER_INT64,
                DeviceFeature.STORAGE_8BIT,
                DeviceFeature.STORAGE_16BIT,
                DeviceFeature.SHADER_FLOAT16);
        if (target.dp4aAvailable()) {
            features.add(DeviceFeature.INTEGER_DOT_PRODUCT);
        }
        if (target.matrixInstructionsAvailable()) {
            features.add(DeviceFeature.COOPERATIVE_MATRIX);
        }
        if (target.bf16MmaAvailable()) {
            features.add(DeviceFeature.BF16_COOPERATIVE_MATRIX);
        }
        return features;
    }
}
