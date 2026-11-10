package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;

import java.util.Arrays;
import java.util.Objects;

import static org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_BALLOT_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_BASIC_BIT;

/**
 * The device facts that decide whether a compute pipeline may pin its subgroup width
 * ({@code VkPipelineShaderStageRequiredSubgroupSizeCreateInfo} with {@code REQUIRE_FULL_SUBGROUPS}), read once from
 * the physical-device properties and the enabled features ({@link #of}). Immutable.
 *
 * <p>{@link VkProgramCache} refuses a pinned module the device cannot run ({@link #refusal}), and
 * {@link VkKernelHandle} refuses a local size the pinned pipeline cannot hold ({@link #localSizeRefusal}). Keeping the
 * facts in this value lets the rules be checked without a device.</p>
 */
public final class VkPinnedSubgroups {

    /** The subgroup operations a pinned module may use in compute shaders: basic, ballot and arithmetic. */
    static final int OPERATIONS = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT
            | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;

    private final int supportedStages;
    private final int supportedOperations;
    private final boolean sizeControl;
    private final boolean fullSubgroups;
    private final int requiredSizeStages;
    private final int minSize;
    private final int maxSize;
    private final int maxWorkgroupSubgroups;

    /**
     * @param supportedStages       {@code subgroupSupportedStages}
     * @param supportedOperations   {@code subgroupSupportedOperations}
     * @param sizeControl           whether {@code subgroupSizeControl} is enabled
     * @param fullSubgroups         whether {@code computeFullSubgroups} is enabled
     * @param requiredSizeStages    {@code requiredSubgroupSizeStages}
     * @param minSize               {@code minSubgroupSize}
     * @param maxSize               {@code maxSubgroupSize}
     * @param maxWorkgroupSubgroups {@code maxComputeWorkgroupSubgroups}
     */
    VkPinnedSubgroups(int supportedStages, int supportedOperations, boolean sizeControl, boolean fullSubgroups,
                      int requiredSizeStages, int minSize, int maxSize, int maxWorkgroupSubgroups) {
        this.supportedStages = supportedStages;
        this.supportedOperations = supportedOperations;
        this.sizeControl = sizeControl;
        this.fullSubgroups = fullSubgroups;
        this.requiredSizeStages = requiredSizeStages;
        this.minSize = minSize;
        this.maxSize = maxSize;
        this.maxWorkgroupSubgroups = maxWorkgroupSubgroups;
    }

    /** The facts of a device with these properties and enabled features. */
    public static VkPinnedSubgroups of(VkPhysicalDeviceInfo info, VkDeviceFeatures.Enabled enabled) {
        Objects.requireNonNull(info, "physical device info must not be null");
        Objects.requireNonNull(enabled, "enabled features must not be null");
        return new VkPinnedSubgroups(info.subgroupSupportedStages(), info.subgroupSupportedOperations(),
                enabled.isEnabled(VkPhysicalDeviceInfo.Feature.SUBGROUP_SIZE_CONTROL), enabled.computeFullSubgroups(),
                info.requiredSubgroupSizeStages(), info.minSubgroupSize(), info.maxSubgroupSize(),
                info.maxComputeWorkgroupSubgroups());
    }

    /**
     * Why the device cannot run a compute pipeline pinned to {@code size}-lane full subgroups, or null when it can:
     * {@code size} a power of two, the basic, ballot and arithmetic subgroup operations ({@link #OPERATIONS}, all a
     * pinned module uses) in compute shaders, {@code subgroupSizeControl} and {@code computeFullSubgroups} enabled,
     * the compute stage in {@code requiredSubgroupSizeStages}, and {@code size} within
     * {@code [minSubgroupSize, maxSubgroupSize]}.
     */
    public String refusal(int size) {
        if (size <= 0 || Integer.bitCount(size) != 1) {
            return size + " is not a power of two";
        }
        if ((supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0 || (supportedOperations & OPERATIONS) != OPERATIONS) {
            return "compute shaders lack the basic, ballot or arithmetic subgroup operations";
        }
        if (!sizeControl) {
            return "subgroupSizeControl is not enabled";
        }
        if (!fullSubgroups) {
            return "computeFullSubgroups is not enabled";
        }
        if ((requiredSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
            return "requiredSubgroupSizeStages excludes compute shaders";
        }
        if (size < minSize || size > maxSize) {
            return "its subgroups have " + minSize + " to " + maxSize + " lanes";
        }
        return null;
    }

    /**
     * Why a pipeline pinned to {@code size}-lane full subgroups cannot run work-groups of {@code local3}, or null
     * when it can: full subgroups need {@code local_x % size == 0}, and a work-group may hold at most
     * {@code maxComputeWorkgroupSubgroups} subgroups, so at most {@code size * maxComputeWorkgroupSubgroups}
     * invocations.
     *
     * @throws IllegalArgumentException when {@code size} is not positive or {@code local3} is not three positive
     *                                  sizes
     */
    String localSizeRefusal(int size, int[] local3) {
        if (size <= 0) {
            throw new IllegalArgumentException("pinned subgroup size must be positive, got " + size);
        }
        if (local3 == null || local3.length != 3 || local3[0] <= 0 || local3[1] <= 0 || local3[2] <= 0) {
            throw new IllegalArgumentException("local size must be three positive sizes, got "
                    + Arrays.toString(local3));
        }
        long invocations = (long) local3[0] * local3[1] * local3[2];
        long limit = (long) size * maxWorkgroupSubgroups;
        if (local3[0] % size != 0 || invocations > limit) {
            return "local size " + Arrays.toString(local3) + " cannot run full subgroups of " + size + " lanes (local"
                    + " x must be a multiple of " + size + ", at most " + limit + " invocations)";
        }
        return null;
    }
}
