package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import jcuda.driver.CUdevice;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Set;

import static jcuda.driver.CUdevice_attribute.*;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceTotalMem;
import static jcuda.driver.JCudaDriver.cuDriverGetVersion;

/**
 * Builds {@link DeviceCapabilities} for a CUDA device from driver attributes:
 * VRAM, limits, compute capability, warp size, features, identity and driver version.
 */
public final class CudaCapabilitiesBuilder {

    /** Static shared memory per block available without an opt-in carve-out. */
    public static final long STATIC_SHARED_MEMORY_BYTES = 48L * 1024;

    /** Kernel parameter space per launch on every supported architecture. */
    public static final int KERNEL_PARAM_BYTES = 4096;

    /** Half-precision arithmetic needs compute capability 5.3. */
    private static final int MIN_CC_FLOAT16 = 53;
    /** {@code dp4a} packed integer dot products need compute capability 6.1. */
    private static final int MIN_CC_DP4A = 61;

    private static final Set<String> ACTIVATION_FUNCTIONS = new HashSet<>(Arrays.asList(
            "SIGMOID", "RELU", "TANH", "LEAKY_RELU", "ELU", "SOFTPLUS", "SOFTSIGN",
            "ISRU", "GAUSSIAN", "SINC", "SWISH", "SELU", "HARD_SIGMOID",
            "BENT_IDENTITY", "PRELU", "SRELU", "SOFTEXPONENTIAL", "IDENTITY"));
    private static final Set<String> COST_FUNCTIONS = new HashSet<>(Arrays.asList(
            "MSE", "MAE", "HUBER", "LOG_COSH", "KL_DIVERGENCE", "CROSS_ENTROPY"));
    private static final Set<String> BLAS_LEVELS = new HashSet<>(Arrays.asList("1", "2", "3"));

    private CudaCapabilitiesBuilder() {
    }

    /**
     * Queries the device; needs no current context.
     *
     * @throws IllegalArgumentException when {@code device} is null
     * @throws RuntimeException         the translated driver failure of a required query
     */
    public static DeviceCapabilities build(CUDevice device) {
        if (device == null) {
            throw new IllegalArgumentException("CUDevice must not be null");
        }
        CUdevice dev = device.getDeviceId();
        int cc = device.getComputeCapability();
        int warp = attribute(dev, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        long sharedPerBlock = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        long sharedOptIn = Math.max(sharedPerBlock, attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN));

        return new DeviceCapabilities.Builder()
                .deviceName(device.getDeviceName())
                .totalVramBytes(totalMemory(dev))
                .computeUnits(attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
                .maxClockFrequencyMhz(attribute(dev, CU_DEVICE_ATTRIBUTE_CLOCK_RATE) / 1000)
                .maxWorkGroupSize(attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
                .supportsAsyncTransfer(attribute(dev, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT) >= 1)
                .driverVersion(describeDriver(driverVersion()))
                .supportsFloat64(true)
                .supportsFloat16(cc >= MIN_CC_FLOAT16)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(sharedOptIn)
                .supportedActivationFunctions(ACTIVATION_FUNCTIONS)
                .supportedCostFunctions(COST_FUNCTIONS)
                .supportedBLASLevels(BLAS_LEVELS)
                .deviceClass(device.getDeviceClass())
                .identity(device.getIdentity())
                .subgroupSize(warp)
                .maxWorkGroupCount(new long[]{
                        attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
                        attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
                        attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)})
                .maxPushConstantBytes(KERNEL_PARAM_BYTES)
                .maxSharedMemoryBytes(Math.min(STATIC_SHARED_MEMORY_BYTES, sharedPerBlock))
                .computeCapability(cc)
                .features(features(cc))
                .build();
    }

    /**
     * The installed driver's CUDA version as {@code 1000 * major + 10 * minor}
     * ({@code cuDriverGetVersion}), for example 13020 for 13.2.
     */
    public static int driverVersion() {
        int[] version = new int[1];
        CudaErrors.check("cuDriverGetVersion", () -> cuDriverGetVersion(version));
        return version[0];
    }

    /** {@code "CUDA driver 13.2"} for 13020. */
    public static String describeDriver(int version) {
        return "CUDA driver " + (version / 1000) + "." + ((version % 1000) / 10);
    }

    private static Set<DeviceFeature> features(int cc) {
        Set<DeviceFeature> features = EnumSet.of(
                DeviceFeature.SUBGROUP_ARITHMETIC,
                DeviceFeature.SHADER_INT8,
                DeviceFeature.SHADER_INT64,
                DeviceFeature.STORAGE_8BIT,
                DeviceFeature.STORAGE_16BIT);
        if (cc >= MIN_CC_FLOAT16) {
            features.add(DeviceFeature.SHADER_FLOAT16);
        }
        if (cc >= MIN_CC_DP4A) {
            features.add(DeviceFeature.INTEGER_DOT_PRODUCT);
        }
        return features;
    }

    private static long totalMemory(CUdevice dev) {
        long[] bytes = new long[1];
        CudaErrors.check("cuDeviceTotalMem", () -> cuDeviceTotalMem(bytes, dev));
        return bytes[0];
    }

    private static int attribute(CUdevice dev, int attribute) {
        int[] value = new int[1];
        CudaErrors.check("cuDeviceGetAttribute(" + stringFor(attribute) + ")",
                () -> cuDeviceGetAttribute(value, attribute, dev));
        return value[0];
    }
}
