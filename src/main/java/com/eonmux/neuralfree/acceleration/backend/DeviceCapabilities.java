package com.eonmux.neuralfree.acceleration.backend;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Set;

/**
 * Immutable description of what a compute device offers.
 *
 * <p>Instances are created through {@link Builder}. The 15-argument constructor
 * remains for existing callers; it leaves every newer field at its documented
 * Builder default.</p>
 */
public final class DeviceCapabilities {

    /** Default maximum work-group count per dimension (the CUDA and Vulkan minimum guarantees). */
    private static final long[] DEFAULT_MAX_WORK_GROUP_COUNT = {Integer.MAX_VALUE, 65535L, 65535L};
    /** Default push-constant block limit (the Vulkan minimum guarantee). */
    private static final int DEFAULT_MAX_PUSH_CONSTANT_BYTES = 128;

    private final String deviceName;
    private final long totalVramBytes;
    private final long maxAllocSizeBytes;
    private final int computeUnits;
    private final int maxClockFrequencyMhz;
    private final long maxWorkGroupSize;
    private final boolean supportsAsyncTransfer;
    private final String driverVersion;
    private final boolean supportsFloat64;
    private final boolean supportsFloat16;
    private final boolean supportsSubgroups;
    private final long maxLocalMemoryBytes;
    private final Set<String> supportedActivationFunctions;
    private final Set<String> supportedCostFunctions;
    private final Set<String> supportedBLASLevels;
    private final DeviceClass deviceClass;
    private final DeviceIdentity identity;
    private final int subgroupSize;
    private final int minSubgroupSize;
    private final int maxSubgroupSize;
    private final long[] maxWorkGroupCount;
    private final int maxPushConstantBytes;
    private final long maxSharedMemoryBytes;
    private final long maxStorageBufferRange;
    private final int computeCapability;
    private final Set<DeviceFeature> features;

    public DeviceCapabilities(String deviceName, long totalVramBytes, long maxAllocSizeBytes, int computeUnits,
                              int maxClockFrequencyMhz, long maxWorkGroupSize,
                              boolean supportsAsyncTransfer, String driverVersion,
                              boolean supportsFloat64, boolean supportsFloat16,
                              boolean supportsSubgroups,
                              long maxLocalMemoryBytes,
                              Set<String> supportedActivationFunctions,
                              Set<String> supportedCostFunctions,
                              Set<String> supportedBLASLevels) {
        this(new Builder()
                .deviceName(deviceName)
                .totalVramBytes(totalVramBytes)
                .maxAllocSizeBytes(maxAllocSizeBytes)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz(maxClockFrequencyMhz)
                .maxWorkGroupSize(maxWorkGroupSize)
                .supportsAsyncTransfer(supportsAsyncTransfer)
                .driverVersion(driverVersion)
                .supportsFloat64(supportsFloat64)
                .supportsFloat16(supportsFloat16)
                .supportsSubgroups(supportsSubgroups)
                .maxLocalMemoryBytes(maxLocalMemoryBytes)
                .supportedActivationFunctions(supportedActivationFunctions)
                .supportedCostFunctions(supportedCostFunctions)
                .supportedBLASLevels(supportedBLASLevels));
    }

    private DeviceCapabilities(Builder b) {
        this.deviceName = b.deviceName;
        this.totalVramBytes = b.totalVramBytes;
        this.maxAllocSizeBytes = b.maxAllocSizeBytes > 0 ? b.maxAllocSizeBytes : b.totalVramBytes;
        this.computeUnits = b.computeUnits;
        this.maxClockFrequencyMhz = b.maxClockFrequencyMhz;
        this.maxWorkGroupSize = b.maxWorkGroupSize;
        this.supportsAsyncTransfer = b.supportsAsyncTransfer;
        this.driverVersion = b.driverVersion;
        this.supportsFloat64 = b.supportsFloat64;
        this.supportsFloat16 = b.supportsFloat16;
        this.supportsSubgroups = b.supportsSubgroups;
        this.maxLocalMemoryBytes = b.maxLocalMemoryBytes;
        this.supportedActivationFunctions = Collections.unmodifiableSet(new HashSet<>(b.supportedActivationFunctions));
        this.supportedCostFunctions = Collections.unmodifiableSet(new HashSet<>(b.supportedCostFunctions));
        this.supportedBLASLevels = Collections.unmodifiableSet(new HashSet<>(b.supportedBLASLevels));

        this.deviceClass = b.deviceClass;
        this.identity = b.identity != null
                ? b.identity
                : DeviceIdentity.ofName(b.deviceName == null ? "" : b.deviceName);
        if (b.subgroupSize < 0 || b.minSubgroupSize < 0 || b.maxSubgroupSize < 0) {
            throw new IllegalArgumentException("subgroup sizes must not be negative: size=" + b.subgroupSize
                    + " min=" + b.minSubgroupSize + " max=" + b.maxSubgroupSize);
        }
        this.subgroupSize = b.subgroupSize;
        this.minSubgroupSize = b.minSubgroupSize > 0 ? b.minSubgroupSize : b.subgroupSize;
        this.maxSubgroupSize = b.maxSubgroupSize > 0 ? b.maxSubgroupSize : b.subgroupSize;
        if (this.minSubgroupSize > this.maxSubgroupSize) {
            throw new IllegalArgumentException("minSubgroupSize " + this.minSubgroupSize
                    + " exceeds maxSubgroupSize " + this.maxSubgroupSize);
        }
        this.maxWorkGroupCount = b.maxWorkGroupCount.clone();
        if (b.maxPushConstantBytes < 0) {
            throw new IllegalArgumentException("maxPushConstantBytes must not be negative: " + b.maxPushConstantBytes);
        }
        this.maxPushConstantBytes = b.maxPushConstantBytes;
        this.maxSharedMemoryBytes = b.maxSharedMemoryBytes >= 0 ? b.maxSharedMemoryBytes : b.maxLocalMemoryBytes;
        this.maxStorageBufferRange = b.maxStorageBufferRange > 0 ? b.maxStorageBufferRange : this.maxAllocSizeBytes;
        if (b.computeCapability < 0) {
            throw new IllegalArgumentException("computeCapability must not be negative: " + b.computeCapability);
        }
        this.computeCapability = b.computeCapability;
        this.features = b.features.isEmpty()
                ? Collections.<DeviceFeature>emptySet()
                : Collections.unmodifiableSet(EnumSet.copyOf(b.features));
    }

    public String getDeviceName() {
        return deviceName;
    }

    public long getTotalVramBytes() {
        return totalVramBytes;
    }

    public long getMaxAllocSizeBytes() {
        return maxAllocSizeBytes;
    }

    public int getComputeUnits() {
        return computeUnits;
    }

    public int getMaxClockFrequencyMhz() {
        return maxClockFrequencyMhz;
    }

    public long getMaxWorkGroupSize() {
        return maxWorkGroupSize;
    }

    public boolean supportsAsyncTransfer() {
        return supportsAsyncTransfer;
    }

    public String getDriverVersion() {
        return driverVersion;
    }

    public boolean supportsFloat64() {
        return supportsFloat64;
    }

    public boolean supportsFloat16() {
        return supportsFloat16;
    }

    public boolean supportsSubgroups() {
        return supportsSubgroups;
    }

    public long getMaxLocalMemoryBytes() {
        return maxLocalMemoryBytes;
    }

    public Set<String> getSupportedActivationFunctions() {
        return supportedActivationFunctions;
    }

    public Set<String> getSupportedCostFunctions() {
        return supportedCostFunctions;
    }

    public Set<String> getSupportedBLASLevels() {
        return supportedBLASLevels;
    }

    public boolean supportsActivationFunction(String activationFunction) {
        return supportedActivationFunctions.contains(activationFunction);
    }

    public boolean supportsCostFunction(String costFunction) {
        return supportedCostFunctions.contains(costFunction);
    }

    public boolean supportsBLASLevel(String level) {
        return supportedBLASLevels.contains(level);
    }

    /** Physical class of the device; {@link DeviceClass#UNKNOWN} when the backend could not tell. */
    public DeviceClass getDeviceClass() {
        return deviceClass;
    }

    /** Cross-API identity of the device; at least the device name. */
    public DeviceIdentity getIdentity() {
        return identity;
    }

    /** Preferred subgroup (warp, wavefront) size; 0 when unknown. */
    public int getSubgroupSize() {
        return subgroupSize;
    }

    /** Smallest subgroup size the device may use; equals {@link #getSubgroupSize()} unless it varies. */
    public int getMinSubgroupSize() {
        return minSubgroupSize;
    }

    /** Largest subgroup size the device may use; equals {@link #getSubgroupSize()} unless it varies. */
    public int getMaxSubgroupSize() {
        return maxSubgroupSize;
    }

    /** Maximum number of work-groups per dispatch dimension (copy; three entries). */
    public long[] getMaxWorkGroupCount() {
        return maxWorkGroupCount.clone();
    }

    /** Maximum push-constant block size in bytes. */
    public int getMaxPushConstantBytes() {
        return maxPushConstantBytes;
    }

    /**
     * Shared (work-group local) memory a kernel can use without opting in to a
     * larger carve-out, in bytes. Defaults to {@link #getMaxLocalMemoryBytes()}.
     */
    public long getMaxSharedMemoryBytes() {
        return maxSharedMemoryBytes;
    }

    /** Largest byte range one buffer binding can address. Defaults to {@link #getMaxAllocSizeBytes()}. */
    public long getMaxStorageBufferRange() {
        return maxStorageBufferRange;
    }

    /** CUDA compute capability as {@code major * 10 + minor}; 0 on other backends. */
    public int getComputeCapability() {
        return computeCapability;
    }

    /** Unmodifiable set of enabled optional features. */
    public Set<DeviceFeature> getFeatures() {
        return features;
    }

    public boolean hasFeature(DeviceFeature feature) {
        return features.contains(feature);
    }

    public static class Builder {
        private String deviceName;
        private long totalVramBytes;
        private long maxAllocSizeBytes;
        private int computeUnits;
        private int maxClockFrequencyMhz;
        private long maxWorkGroupSize;
        private boolean supportsAsyncTransfer;
        private String driverVersion;
        private boolean supportsFloat64 = true;
        private boolean supportsFloat16 = false;
        private boolean supportsSubgroups = false;
        private long maxLocalMemoryBytes = 32768;
        private Set<String> supportedActivationFunctions = new HashSet<>(Arrays.asList(
                "SIGMOID", "RELU", "TANH", "LEAKY_RELU", "ELU", "SOFTPLUS", "SOFTSIGN",
                "ISRU", "GAUSSIAN", "SINC", "SWISH", "SELU", "HARD_SIGMOID",
                "BENT_IDENTITY", "PRELU", "SRELU", "SOFTEXPONENTIAL", "IDENTITY"
        ));
        private Set<String> supportedCostFunctions = new HashSet<>(Arrays.asList(
                "MSE", "MAE", "HUBER", "LOG_COSH", "KL_DIVERGENCE", "CROSS_ENTROPY"
        ));
        private Set<String> supportedBLASLevels = new HashSet<>(Arrays.asList("1", "2", "3"));
        private DeviceClass deviceClass = DeviceClass.UNKNOWN;
        private DeviceIdentity identity;
        private int subgroupSize;
        private int minSubgroupSize;
        private int maxSubgroupSize;
        private long[] maxWorkGroupCount = DEFAULT_MAX_WORK_GROUP_COUNT.clone();
        private int maxPushConstantBytes = DEFAULT_MAX_PUSH_CONSTANT_BYTES;
        private long maxSharedMemoryBytes = -1;
        private long maxStorageBufferRange = -1;
        private int computeCapability;
        private final Set<DeviceFeature> features = EnumSet.noneOf(DeviceFeature.class);

        public Builder deviceName(String deviceName) {
            this.deviceName = deviceName;
            return this;
        }

        public Builder totalVramBytes(long totalVramBytes) {
            this.totalVramBytes = totalVramBytes;
            return this;
        }

        public Builder maxAllocSizeBytes(long maxAllocSizeBytes) {
            this.maxAllocSizeBytes = maxAllocSizeBytes;
            return this;
        }

        public Builder computeUnits(int computeUnits) {
            this.computeUnits = computeUnits;
            return this;
        }

        public Builder maxClockFrequencyMhz(int maxClockFrequencyMhz) {
            this.maxClockFrequencyMhz = maxClockFrequencyMhz;
            return this;
        }

        public Builder maxWorkGroupSize(long maxWorkGroupSize) {
            this.maxWorkGroupSize = maxWorkGroupSize;
            return this;
        }

        public Builder supportsAsyncTransfer(boolean supportsAsyncTransfer) {
            this.supportsAsyncTransfer = supportsAsyncTransfer;
            return this;
        }

        public Builder driverVersion(String driverVersion) {
            this.driverVersion = driverVersion;
            return this;
        }

        public Builder supportsFloat64(boolean supportsFloat64) {
            this.supportsFloat64 = supportsFloat64;
            return this;
        }

        public Builder supportsFloat16(boolean supportsFloat16) {
            this.supportsFloat16 = supportsFloat16;
            return this;
        }

        public Builder supportsSubgroups(boolean supportsSubgroups) {
            this.supportsSubgroups = supportsSubgroups;
            return this;
        }

        public Builder maxLocalMemoryBytes(long maxLocalMemoryBytes) {
            this.maxLocalMemoryBytes = maxLocalMemoryBytes;
            return this;
        }

        public Builder supportedActivationFunctions(Set<String> functions) {
            this.supportedActivationFunctions = functions != null
                    ? new HashSet<>(functions)
                    : new HashSet<>();
            return this;
        }

        public Builder supportedCostFunctions(Set<String> functions) {
            this.supportedCostFunctions = functions != null
                    ? new HashSet<>(functions)
                    : new HashSet<>();
            return this;
        }

        public Builder supportedBLASLevels(Set<String> levels) {
            this.supportedBLASLevels = levels != null
                    ? new HashSet<>(levels)
                    : new HashSet<>();
            return this;
        }

        /** Default {@link DeviceClass#UNKNOWN}. */
        public Builder deviceClass(DeviceClass deviceClass) {
            this.deviceClass = deviceClass != null ? deviceClass : DeviceClass.UNKNOWN;
            return this;
        }

        /** Default: an identity holding only the device name. */
        public Builder identity(DeviceIdentity identity) {
            this.identity = identity;
            return this;
        }

        /** Default 0 (unknown). */
        public Builder subgroupSize(int subgroupSize) {
            this.subgroupSize = subgroupSize;
            return this;
        }

        /** Default 0, meaning equal to the subgroup size. */
        public Builder minSubgroupSize(int minSubgroupSize) {
            this.minSubgroupSize = minSubgroupSize;
            return this;
        }

        /** Default 0, meaning equal to the subgroup size. */
        public Builder maxSubgroupSize(int maxSubgroupSize) {
            this.maxSubgroupSize = maxSubgroupSize;
            return this;
        }

        /**
         * Default {2^31-1, 65535, 65535}.
         *
         * @throws IllegalArgumentException unless exactly three positive entries are given
         */
        public Builder maxWorkGroupCount(long[] maxWorkGroupCount) {
            if (maxWorkGroupCount == null || maxWorkGroupCount.length != 3) {
                throw new IllegalArgumentException("maxWorkGroupCount must have exactly 3 entries");
            }
            for (int d = 0; d < 3; d++) {
                if (maxWorkGroupCount[d] <= 0) {
                    throw new IllegalArgumentException("maxWorkGroupCount[" + d + "] must be positive, got "
                            + maxWorkGroupCount[d]);
                }
            }
            this.maxWorkGroupCount = maxWorkGroupCount.clone();
            return this;
        }

        /** Default 128. */
        public Builder maxPushConstantBytes(int maxPushConstantBytes) {
            this.maxPushConstantBytes = maxPushConstantBytes;
            return this;
        }

        /** Default: the value of {@link #maxLocalMemoryBytes(long)}. */
        public Builder maxSharedMemoryBytes(long maxSharedMemoryBytes) {
            this.maxSharedMemoryBytes = maxSharedMemoryBytes;
            return this;
        }

        /** Default: the effective maximum allocation size. */
        public Builder maxStorageBufferRange(long maxStorageBufferRange) {
            this.maxStorageBufferRange = maxStorageBufferRange;
            return this;
        }

        /** CUDA {@code major * 10 + minor}; default 0. */
        public Builder computeCapability(int computeCapability) {
            this.computeCapability = computeCapability;
            return this;
        }

        /** Replaces the feature set; default empty. */
        public Builder features(Collection<DeviceFeature> features) {
            this.features.clear();
            if (features != null) {
                for (DeviceFeature f : features) {
                    if (f == null) {
                        throw new IllegalArgumentException("feature set must not contain null");
                    }
                    this.features.add(f);
                }
            }
            return this;
        }

        /** Adds one feature. */
        public Builder feature(DeviceFeature feature) {
            if (feature == null) {
                throw new IllegalArgumentException("feature must not be null");
            }
            this.features.add(feature);
            return this;
        }

        public DeviceCapabilities build() {
            return new DeviceCapabilities(this);
        }
    }

    @Override
    public String toString() {
        return String.format("DeviceCapabilities{name='%s', vram=%d MB, computeUnits=%d, maxClock=%d MHz, " +
                        "maxWorkGroup=%d, asyncTransfer=%s, float64=%s, float16=%s, localMem=%d KB, " +
                        "activations=%d, costs=%d, blas=%s, class=%s, subgroup=%d, sharedMem=%d KB, " +
                        "computeCapability=%d, features=%s}",
                deviceName, totalVramBytes / (1024 * 1024), computeUnits,
                maxClockFrequencyMhz, maxWorkGroupSize, supportsAsyncTransfer,
                supportsFloat64, supportsFloat16, maxLocalMemoryBytes / 1024,
                supportedActivationFunctions.size(), supportedCostFunctions.size(), supportedBLASLevels,
                deviceClass, subgroupSize, maxSharedMemoryBytes / 1024, computeCapability, features);
    }
}
