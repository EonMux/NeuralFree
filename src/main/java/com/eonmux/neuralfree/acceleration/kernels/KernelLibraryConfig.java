package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;

import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Immutable classpath layout and build profile of one kernel library across the three backends.
 *
 * <ul>
 *   <li>{@code sourceRoot}: dual-dialect sources and {@code manifest.txt} (OpenCL builds them at run time),
 *       for example {@code kernels/llm};</li>
 *   <li>{@code cudaRoot}: prebuilt fatbins with {@code index.tsv} and {@code build.properties},
 *       for example {@code kernels/llm/cuda};</li>
 *   <li>{@code spvRoot}: SPIR-V modules with {@code index.properties}, for example {@code shaders/llm};</li>
 *   <li>{@code requireCompleteInventory}: CUDA and Vulkan report the library unavailable unless every
 *       non-OpenCL-only manifest kernel was built;</li>
 *   <li>{@code openClOptions}: the static OpenCL compiler options of the library's program, to which
 *       {@link OpenClKernelLibrary} appends its defines;</li>
 *   <li>{@code subgroupVariants}: whether the library has subgroup-reduction variants (the OpenCL
 *       {@code USE_SUBGROUPS} program and the Vulkan {@code __sg} modules); when false neither is used and
 *       {@link KernelSet#subgroups()} is false on every backend;</li>
 *   <li>{@code vulkanFeatures}: the device features the Vulkan library requires.</li>
 * </ul>
 * Roots are classpath resource paths without leading or trailing slashes.
 */
public final class KernelLibraryConfig {

    /** Vulkan features of the five-argument constructor: the LLM shaders use 8- and 16-bit storage. */
    private static final Set<DeviceFeature> DEFAULT_VULKAN_FEATURES = Collections.unmodifiableSet(EnumSet.of(
            DeviceFeature.STORAGE_8BIT, DeviceFeature.STORAGE_16BIT, DeviceFeature.TIMELINE_SEMAPHORE));

    private final String libraryId;
    private final String sourceRoot;
    private final String cudaRoot;
    private final String spvRoot;
    private final boolean requireCompleteInventory;
    private final String openClOptions;
    private final boolean subgroupVariants;
    private final Set<DeviceFeature> vulkanFeatures;

    /**
     * A library with the LLM build profile: OpenCL options {@link OpenClKernelLibrary#BASE_OPTIONS},
     * subgroup variants, and the Vulkan features {@link DeviceFeature#STORAGE_8BIT},
     * {@link DeviceFeature#STORAGE_16BIT} and {@link DeviceFeature#TIMELINE_SEMAPHORE}.
     *
     * @throws IllegalArgumentException when an id or root is null, blank, or has a leading or
     *                                  trailing slash
     */
    public KernelLibraryConfig(String libraryId, String sourceRoot, String cudaRoot, String spvRoot,
                               boolean requireCompleteInventory) {
        this(libraryId, sourceRoot, cudaRoot, spvRoot, requireCompleteInventory, OpenClKernelLibrary.BASE_OPTIONS,
                true, DEFAULT_VULKAN_FEATURES);
    }

    /**
     * @param openClOptions    static OpenCL compiler options, without surrounding whitespace (may be empty)
     * @param subgroupVariants whether subgroup-reduction variants exist and may be selected
     * @param vulkanFeatures   device features the Vulkan library requires; copied
     * @throws IllegalArgumentException when an id or root is null, blank, or has a leading or trailing
     *                                  slash, when the options are null or have surrounding whitespace or a
     *                                  line break, or when the feature collection is null or holds null
     */
    public KernelLibraryConfig(String libraryId, String sourceRoot, String cudaRoot, String spvRoot,
                               boolean requireCompleteInventory, String openClOptions, boolean subgroupVariants,
                               Collection<DeviceFeature> vulkanFeatures) {
        this.libraryId = requireId(libraryId);
        this.sourceRoot = requireRoot(sourceRoot, "sourceRoot");
        this.cudaRoot = requireRoot(cudaRoot, "cudaRoot");
        this.spvRoot = requireRoot(spvRoot, "spvRoot");
        this.requireCompleteInventory = requireCompleteInventory;
        this.openClOptions = requireOptions(openClOptions);
        this.subgroupVariants = subgroupVariants;
        this.vulkanFeatures = copyFeatures(vulkanFeatures);
    }

    public String libraryId() {
        return libraryId;
    }

    public String sourceRoot() {
        return sourceRoot;
    }

    public String cudaRoot() {
        return cudaRoot;
    }

    public String spvRoot() {
        return spvRoot;
    }

    public boolean requireCompleteInventory() {
        return requireCompleteInventory;
    }

    /** Static OpenCL compiler options of the library's program. */
    public String openClOptions() {
        return openClOptions;
    }

    /** Whether subgroup-reduction variants exist and may be selected. */
    public boolean subgroupVariants() {
        return subgroupVariants;
    }

    /** Unmodifiable set of the device features the Vulkan library requires, in declaration order. */
    public Set<DeviceFeature> vulkanFeatures() {
        return vulkanFeatures;
    }

    /** A copy of this configuration with a different inventory requirement; every other field is kept. */
    public KernelLibraryConfig withRequireCompleteInventory(boolean require) {
        if (require == requireCompleteInventory) {
            return this;
        }
        return new KernelLibraryConfig(libraryId, sourceRoot, cudaRoot, spvRoot, require, openClOptions,
                subgroupVariants, vulkanFeatures);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KernelLibraryConfig)) {
            return false;
        }
        KernelLibraryConfig that = (KernelLibraryConfig) o;
        return requireCompleteInventory == that.requireCompleteInventory
                && subgroupVariants == that.subgroupVariants
                && libraryId.equals(that.libraryId)
                && sourceRoot.equals(that.sourceRoot)
                && cudaRoot.equals(that.cudaRoot)
                && spvRoot.equals(that.spvRoot)
                && openClOptions.equals(that.openClOptions)
                && vulkanFeatures.equals(that.vulkanFeatures);
    }

    @Override
    public int hashCode() {
        int h = libraryId.hashCode();
        h = 31 * h + sourceRoot.hashCode();
        h = 31 * h + cudaRoot.hashCode();
        h = 31 * h + spvRoot.hashCode();
        h = 31 * h + (requireCompleteInventory ? 1 : 0);
        h = 31 * h + openClOptions.hashCode();
        h = 31 * h + (subgroupVariants ? 1 : 0);
        return 31 * h + vulkanFeatures.hashCode();
    }

    @Override
    public String toString() {
        return "KernelLibraryConfig{id=" + libraryId + ", source=" + sourceRoot + ", cuda=" + cudaRoot
                + ", spv=" + spvRoot + ", requireCompleteInventory=" + requireCompleteInventory
                + ", openClOptions='" + openClOptions + "', subgroupVariants=" + subgroupVariants
                + ", vulkanFeatures=" + vulkanFeatures + "}";
    }

    private static String requireId(String id) {
        if (id == null || id.trim().isEmpty()) {
            throw new IllegalArgumentException("libraryId must not be null or blank");
        }
        if (!id.equals(id.trim())) {
            throw new IllegalArgumentException("libraryId '" + id + "' must not have surrounding whitespace");
        }
        return id;
    }

    private static String requireRoot(String root, String what) {
        if (root == null || root.trim().isEmpty()) {
            throw new IllegalArgumentException(what + " must not be null or blank");
        }
        if (root.startsWith("/") || root.endsWith("/")) {
            throw new IllegalArgumentException(what + " '" + root
                    + "' must be a classpath path without leading or trailing '/'");
        }
        return root;
    }

    private static String requireOptions(String options) {
        if (options == null) {
            throw new IllegalArgumentException("openClOptions must not be null");
        }
        if (!options.equals(options.trim())) {
            throw new IllegalArgumentException("openClOptions '" + options + "' must not have surrounding whitespace");
        }
        if (options.indexOf('\n') >= 0 || options.indexOf('\r') >= 0) {
            throw new IllegalArgumentException("openClOptions must be a single line: '" + options + "'");
        }
        return options;
    }

    private static Set<DeviceFeature> copyFeatures(Collection<DeviceFeature> features) {
        if (features == null) {
            throw new IllegalArgumentException("vulkanFeatures must not be null");
        }
        EnumSet<DeviceFeature> copy = EnumSet.noneOf(DeviceFeature.class);
        for (DeviceFeature f : features) {
            if (f == null) {
                throw new IllegalArgumentException("vulkanFeatures must not contain null: " + features);
            }
            copy.add(f);
        }
        return Collections.unmodifiableSet(copy);
    }
}
