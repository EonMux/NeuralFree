package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo.Feature;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.Struct;
import org.lwjgl.vulkan.VkBaseOutStructure;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures2;
import org.lwjgl.vulkan.VkPhysicalDeviceShaderIntegerDotProductFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceSubgroupSizeControlFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan11Features;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan12Features;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan13Features;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import static org.lwjgl.vulkan.EXTMemoryBudget.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
import static org.lwjgl.vulkan.EXTSubgroupSizeControl.VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRPortabilitySubset.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRPushDescriptor.VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRShaderIntegerDotProduct.VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
import static org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_BASIC_BIT;
import static org.lwjgl.vulkan.VK13.VK_API_VERSION_1_3;

/**
 * Chooses the device features and extensions the compute backend enables:
 * timeline semaphores and a compute queue are required; 8/16-bit storage,
 * int8/16/64, float16, subgroup controls, push descriptors and the memory budget
 * are enabled when supported.
 *
 * <p>Every feature the device reports in {@link VkPhysicalDeviceInfo.Feature} is enabled:
 * core {@code shaderInt64}, {@code shaderInt16}, {@code shaderFloat64}; Vulkan 1.1
 * {@code storageBuffer16BitAccess}, {@code uniformAndStorageBuffer16BitAccess}; Vulkan 1.2
 * {@code storageBuffer8BitAccess}, {@code uniformAndStorageBuffer8BitAccess}, {@code shaderInt8},
 * {@code shaderFloat16}, {@code timelineSemaphore}, {@code subgroupBroadcastDynamicId};
 * {@code subgroupSizeControl}, {@code computeFullSubgroups} and {@code shaderIntegerDotProduct}
 * (Vulkan 1.3 core, or their extensions on a 1.2 device); {@code maintenance4} (1.3).</p>
 *
 * <p>Device extensions enabled when the device lists them: {@code VK_KHR_push_descriptor},
 * {@code VK_EXT_memory_budget}, {@code VK_KHR_portability_subset} (mandatory when listed), and on a
 * Vulkan 1.2 device {@code VK_EXT_subgroup_size_control} and {@code VK_KHR_shader_integer_dot_product}
 * when their features are enabled.</p>
 */
public final class VkDeviceFeatures {

    private VkDeviceFeatures() {
    }

    /**
     * Selects the features to enable on {@code info}.
     *
     * @throws NoBackendAvailableException naming the missing requirement when the device is below
     *         Vulkan 1.2 or lacks timeline semaphores or a compute queue
     */
    public static Enabled select(VkPhysicalDeviceInfo info) {
        if (info == null) {
            throw new IllegalArgumentException("physical device info must not be null");
        }
        String reason = VkDeviceRanking.unsupportedReason(info);
        if (reason != null) {
            throw new NoBackendAvailableException("Vulkan device '" + info.name() + "' cannot run the compute backend: "
                    + reason);
        }
        boolean v13 = info.apiAtLeast(VK_API_VERSION_1_3);

        EnumSet<Feature> on = EnumSet.noneOf(Feature.class);
        for (Feature f : Feature.values()) {
            if (info.supports(f)) {
                on.add(f);
            }
        }

        List<String> extensions = new ArrayList<>();
        boolean pushDescriptors = info.hasExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)
                && info.maxPushDescriptors() > 0;
        if (pushDescriptors) {
            extensions.add(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        }
        boolean memoryBudget = info.hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        if (memoryBudget) {
            extensions.add(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
        if (info.hasExtension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            extensions.add(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
        boolean sizeControlExt = !v13
                && (on.contains(Feature.SUBGROUP_SIZE_CONTROL) || on.contains(Feature.COMPUTE_FULL_SUBGROUPS));
        if (sizeControlExt) {
            extensions.add(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
        boolean dotProductExt = !v13 && on.contains(Feature.SHADER_INTEGER_DOT_PRODUCT);
        if (dotProductExt) {
            extensions.add(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        }

        boolean subgroupArithmetic = (info.subgroupSupportedStages() & VK_SHADER_STAGE_COMPUTE_BIT) != 0
                && (info.subgroupSupportedOperations() & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0
                && (info.subgroupSupportedOperations() & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;

        EnumSet<DeviceFeature> mapped = EnumSet.of(DeviceFeature.TIMELINE_SEMAPHORE);
        if (subgroupArithmetic) {
            mapped.add(DeviceFeature.SUBGROUP_ARITHMETIC);
        }
        mapIf(mapped, on, Feature.STORAGE_BUFFER_8BIT_ACCESS, DeviceFeature.STORAGE_8BIT);
        mapIf(mapped, on, Feature.STORAGE_BUFFER_16BIT_ACCESS, DeviceFeature.STORAGE_16BIT);
        mapIf(mapped, on, Feature.SHADER_INT8, DeviceFeature.SHADER_INT8);
        mapIf(mapped, on, Feature.SHADER_INT64, DeviceFeature.SHADER_INT64);
        mapIf(mapped, on, Feature.SHADER_FLOAT16, DeviceFeature.SHADER_FLOAT16);
        mapIf(mapped, on, Feature.COMPUTE_FULL_SUBGROUPS, DeviceFeature.FULL_SUBGROUPS);
        mapIf(mapped, on, Feature.SHADER_INTEGER_DOT_PRODUCT, DeviceFeature.INTEGER_DOT_PRODUCT);
        if (pushDescriptors) {
            mapped.add(DeviceFeature.PUSH_DESCRIPTORS);
        }
        if (memoryBudget) {
            mapped.add(DeviceFeature.MEMORY_BUDGET);
        }
        return new Enabled(v13, on, extensions, mapped, pushDescriptors, memoryBudget, sizeControlExt, dotProductExt);
    }

    private static void mapIf(Set<DeviceFeature> mapped, Set<Feature> on, Feature feature, DeviceFeature as) {
        if (on.contains(feature)) {
            mapped.add(as);
        }
    }

    /** The immutable selection: enabled feature bits, extensions and their {@link DeviceFeature} mapping. */
    public static final class Enabled {

        private final boolean vulkan13;
        private final Set<Feature> features;
        private final List<String> extensions;
        private final Set<DeviceFeature> deviceFeatures;
        private final boolean pushDescriptors;
        private final boolean memoryBudget;
        private final boolean subgroupSizeControlExtension;
        private final boolean integerDotProductExtension;

        private Enabled(boolean vulkan13, EnumSet<Feature> features, List<String> extensions,
                        EnumSet<DeviceFeature> deviceFeatures, boolean pushDescriptors, boolean memoryBudget,
                        boolean subgroupSizeControlExtension, boolean integerDotProductExtension) {
            this.vulkan13 = vulkan13;
            this.features = Collections.unmodifiableSet(EnumSet.copyOf(features));
            this.extensions = Collections.unmodifiableList(new ArrayList<>(extensions));
            this.deviceFeatures = Collections.unmodifiableSet(EnumSet.copyOf(deviceFeatures));
            this.pushDescriptors = pushDescriptors;
            this.memoryBudget = memoryBudget;
            this.subgroupSizeControlExtension = subgroupSizeControlExtension;
            this.integerDotProductExtension = integerDotProductExtension;
        }

        public boolean isEnabled(Feature feature) {
            return features.contains(feature);
        }

        /** Unmodifiable list of device extensions to enable. */
        public List<String> extensions() {
            return extensions;
        }

        public boolean hasExtension(String name) {
            return extensions.contains(name);
        }

        /** Unmodifiable backend-neutral features the enabled set provides. */
        public Set<DeviceFeature> deviceFeatures() {
            return deviceFeatures;
        }

        /** True when {@code VK_KHR_push_descriptor} is enabled. */
        public boolean pushDescriptors() {
            return pushDescriptors;
        }

        /** True when {@code VK_EXT_memory_budget} is enabled. */
        public boolean memoryBudget() {
            return memoryBudget;
        }

        /** True when pipelines may require full subgroups. */
        public boolean computeFullSubgroups() {
            return features.contains(Feature.COMPUTE_FULL_SUBGROUPS);
        }

        /**
         * Writes the feature-struct {@code pNext} chain and the extension names into
         * {@code createInfo}, allocating the structs on {@code stack}.
         *
         * <p>Chain order: {@code VkPhysicalDeviceFeatures2 → Vulkan11Features → Vulkan12Features →}
         * {@code Vulkan13Features} on a 1.3 device, otherwise {@code → SubgroupSizeControlFeatures}
         * and {@code → ShaderIntegerDotProductFeatures} when their extensions are enabled; any
         * {@code pNext} chain already on {@code createInfo} follows the last feature struct.
         * {@code pEnabledFeatures} is cleared, since the core features travel in
         * {@code VkPhysicalDeviceFeatures2}. The structs live as long as {@code stack}'s frame.</p>
         */
        public void applyTo(VkDeviceCreateInfo createInfo, MemoryStack stack) {
            if (createInfo == null || stack == null) {
                throw new IllegalArgumentException("createInfo and stack must not be null");
            }
            VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack).sType$Default();
            VkPhysicalDeviceFeatures core = features2.features();
            core.shaderInt64(isEnabled(Feature.SHADER_INT64))
                    .shaderInt16(isEnabled(Feature.SHADER_INT16))
                    .shaderFloat64(isEnabled(Feature.SHADER_FLOAT64));

            VkPhysicalDeviceVulkan11Features v11 = VkPhysicalDeviceVulkan11Features.calloc(stack).sType$Default()
                    .storageBuffer16BitAccess(isEnabled(Feature.STORAGE_BUFFER_16BIT_ACCESS))
                    .uniformAndStorageBuffer16BitAccess(isEnabled(Feature.UNIFORM_AND_STORAGE_BUFFER_16BIT_ACCESS));

            VkPhysicalDeviceVulkan12Features v12 = VkPhysicalDeviceVulkan12Features.calloc(stack).sType$Default()
                    .storageBuffer8BitAccess(isEnabled(Feature.STORAGE_BUFFER_8BIT_ACCESS))
                    .uniformAndStorageBuffer8BitAccess(isEnabled(Feature.UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS))
                    .shaderInt8(isEnabled(Feature.SHADER_INT8))
                    .shaderFloat16(isEnabled(Feature.SHADER_FLOAT16))
                    .timelineSemaphore(isEnabled(Feature.TIMELINE_SEMAPHORE))
                    .subgroupBroadcastDynamicId(isEnabled(Feature.SUBGROUP_BROADCAST_DYNAMIC_ID));

            List<Struct<?>> chain = new ArrayList<>();
            chain.add(features2);
            chain.add(v11);
            chain.add(v12);
            if (vulkan13) {
                chain.add(VkPhysicalDeviceVulkan13Features.calloc(stack).sType$Default()
                        .subgroupSizeControl(isEnabled(Feature.SUBGROUP_SIZE_CONTROL))
                        .computeFullSubgroups(isEnabled(Feature.COMPUTE_FULL_SUBGROUPS))
                        .maintenance4(isEnabled(Feature.MAINTENANCE4))
                        .shaderIntegerDotProduct(isEnabled(Feature.SHADER_INTEGER_DOT_PRODUCT)));
            } else {
                if (subgroupSizeControlExtension) {
                    chain.add(VkPhysicalDeviceSubgroupSizeControlFeatures.calloc(stack).sType$Default()
                            .subgroupSizeControl(isEnabled(Feature.SUBGROUP_SIZE_CONTROL))
                            .computeFullSubgroups(isEnabled(Feature.COMPUTE_FULL_SUBGROUPS)));
                }
                if (integerDotProductExtension) {
                    chain.add(VkPhysicalDeviceShaderIntegerDotProductFeatures.calloc(stack).sType$Default()
                            .shaderIntegerDotProduct(true));
                }
            }
            long tail = createInfo.pNext();
            for (int i = 0; i + 1 < chain.size(); i++) {
                setNext(chain.get(i), chain.get(i + 1).address());
            }
            setNext(chain.get(chain.size() - 1), tail);

            createInfo.pNext(features2.address());
            createInfo.pEnabledFeatures(null);
            if (extensions.isEmpty()) {
                createInfo.ppEnabledExtensionNames(null);
            } else {
                PointerBuffer names = stack.mallocPointer(extensions.size());
                for (String name : extensions) {
                    names.put(stack.UTF8(name));
                }
                createInfo.ppEnabledExtensionNames(names.flip());
            }
        }

        private static void setNext(Struct<?> struct, long next) {
            MemoryUtil.memPutAddress(struct.address() + VkBaseOutStructure.PNEXT, next);
        }

        @Override
        public String toString() {
            return "VkDeviceFeatures.Enabled{features=" + features + ", extensions=" + extensions
                    + ", deviceFeatures=" + deviceFeatures + '}';
        }
    }
}
