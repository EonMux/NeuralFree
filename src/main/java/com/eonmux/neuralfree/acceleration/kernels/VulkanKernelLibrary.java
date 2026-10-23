package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Vulkan side of a kernel library: one SPIR-V module per kernel, {@code <spvRoot>/<name>.spv}, plus
 * {@code <name>__sg.spv} for kernels built with subgroup reductions.
 *
 * <p>{@code <spvRoot>/index.properties} describes every kernel with four keys:
 * {@code <name>.buffers} (buffer count), {@code <name>.ro} (one {@code 0}/{@code 1} read-only flag per
 * buffer), {@code <name>.push} (comma list of scalar kinds {@code i}, {@code u}, {@code f}) and
 * {@code <name>.sg} ({@code 1} when the {@code __sg} module exists; absent means {@code 0}). Other keys
 * are ignored.</p>
 *
 * <p>The library requires the device features of {@link KernelLibraryConfig#vulkanFeatures()} (for the
 * LLM library {@link DeviceFeature#STORAGE_8BIT}, {@link DeviceFeature#STORAGE_16BIT} and
 * {@link DeviceFeature#TIMELINE_SEMAPHORE}), plus the complete inventory when the configuration asks for
 * it. {@link #resolve} picks the {@code __sg} module only when the configuration has subgroup variants,
 * the index lists the module and the device reports {@link DeviceFeature#SUBGROUP_ARITHMETIC}; the backend
 * applies the local-size rule at dispatch and binds the plain module when a dispatch does not meet it.
 * Specialisation constant {@value FaSmem#VULKAN_SPEC_ID} carries {@code NF_FA_SMEM_FLOATS} (a module that
 * does not declare it ignores the entry); constants 0-2 (local size) are set by the backend per
 * dispatch.</p>
 */
public final class VulkanKernelLibrary implements KernelLibrary {

    /** Index file name inside {@code spvRoot}. */
    public static final String INDEX_FILE = "index.properties";
    /** File-name suffix of the subgroup variant of a kernel module. */
    public static final String SUBGROUP_SUFFIX = "__sg";

    private final KernelLibraryConfig config;
    private final ClassLoader classLoader;
    private final Object indexLock = new Object();
    private volatile Index index;

    public VulkanKernelLibrary(KernelLibraryConfig config) {
        this(config, VulkanKernelLibrary.class.getClassLoader());
    }

    public VulkanKernelLibrary(KernelLibraryConfig config, ClassLoader classLoader) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.classLoader = Objects.requireNonNull(classLoader, "class loader must not be null");
    }

    @Override
    public String libraryId() {
        return config.libraryId();
    }

    @Override
    public ComputeBackend.BackendType backendType() {
        return ComputeBackend.BackendType.VULKAN;
    }

    /** The configuration this library was created from. */
    public KernelLibraryConfig config() {
        return config;
    }

    @Override
    public Availability availability(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.VULKAN) {
            return Availability.no("the Vulkan " + libraryId() + " library cannot serve a "
                    + backend.backendType() + " backend");
        }
        if (!backend.supportsArtifact(KernelArtifact.Kind.SPIRV)) {
            return Availability.no("the Vulkan backend cannot load SPIR-V artifacts");
        }
        Index idx = index();
        if (idx.error != null) {
            return Availability.no(idx.error);
        }
        DeviceCapabilities caps = backend.capabilities();
        List<String> lacking = new ArrayList<>();
        for (DeviceFeature f : config.vulkanFeatures()) {
            if (!caps.hasFeature(f)) {
                lacking.add(f.name());
            }
        }
        if (!lacking.isEmpty()) {
            return Availability.no("the Vulkan device does not enable " + String.join(", ", lacking));
        }
        if (config.requireCompleteInventory()) {
            return KernelManifest.inventoryCoverage(classLoader, libraryId(), config.sourceRoot(),
                    idx.kernels.keySet(), "the Vulkan shader index");
        }
        return Availability.ok();
    }

    @Override
    public KernelArtifact resolve(String kernelName, ComputeBackend backend) {
        KernelManifest.requireKernelName(kernelName);
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.VULKAN) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": the Vulkan " + libraryId()
                    + " library cannot serve a " + backend.backendType() + " backend");
        }
        Index idx = index();
        if (idx.error != null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on Vulkan: " + idx.error);
        }
        Kernel k = idx.kernels.get(kernelName);
        if (k == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " is not in the Vulkan shader index classpath:"
                    + config.spvRoot() + "/" + INDEX_FILE);
        }
        DeviceCapabilities caps = backend.capabilities();
        boolean sg = config.subgroupVariants() && k.subgroupVariant
                && caps.hasFeature(DeviceFeature.SUBGROUP_ARITHMETIC);
        String resource = config.spvRoot() + "/" + kernelName + (sg ? SUBGROUP_SUFFIX : "") + ".spv";
        if (classLoader.getResource(resource) == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": SPIR-V module classpath:" + resource
                    + " listed in the index is missing");
        }
        SortedMap<Integer, Integer> specs = new TreeMap<>();
        specs.put(FaSmem.VULKAN_SPEC_ID, FaSmem.faSmemFloats(ComputeBackend.BackendType.VULKAN, caps));
        return KernelArtifact.spirv(kernelName, resource, specs, k.signature);
    }

    /** The kernels of the shader index; empty when the shaders were not built. */
    @Override
    public Set<String> kernelNames() {
        Index idx = index();
        return idx.error != null ? Collections.<String>emptySet() : idx.kernels.keySet();
    }

    /**
     * The build-time signature of a kernel.
     *
     * @throws UnsupportedKernelException when the index is unusable or does not list the kernel
     */
    public KernelSignature signature(String kernelName) {
        KernelManifest.requireKernelName(kernelName);
        Index idx = index();
        if (idx.error != null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on Vulkan: " + idx.error);
        }
        Kernel k = idx.kernels.get(kernelName);
        if (k == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " is not in the Vulkan shader index");
        }
        return k.signature;
    }

    @Override
    public String toString() {
        return "VulkanKernelLibrary{" + libraryId() + " @ " + config.spvRoot() + "}";
    }

    private Index index() {
        Index i = index;
        if (i != null) {
            return i;
        }
        synchronized (indexLock) {
            if (index == null) {
                index = Index.load(classLoader, config.spvRoot());
            }
            return index;
        }
    }

    /** One indexed kernel. */
    private static final class Kernel {
        final KernelSignature signature;
        final boolean subgroupVariant;

        Kernel(KernelSignature signature, boolean subgroupVariant) {
            this.signature = signature;
            this.subgroupVariant = subgroupVariant;
        }
    }

    /** Parsed shader index, or the reason it is unusable. */
    private static final class Index {
        private static final String BUFFERS = ".buffers";

        final Map<String, Kernel> kernels;
        final String error;

        private Index(Map<String, Kernel> kernels, String error) {
            this.kernels = kernels;
            this.error = error;
        }

        static Index load(ClassLoader cl, String spvRoot) {
            String path = spvRoot + "/" + INDEX_FILE;
            try {
                String text = KernelManifest.readText(cl, path);
                if (text == null) {
                    return new Index(Collections.<String, Kernel>emptyMap(),
                            "Vulkan shaders were not built: glslc not found at build time (classpath:" + path
                                    + " is missing)");
                }
                Properties props = new Properties();
                props.load(new StringReader(text));
                return new Index(parse(props, "classpath:" + path), null);
            } catch (IOException | RuntimeException e) {
                return new Index(Collections.<String, Kernel>emptyMap(),
                        "Vulkan shader index classpath:" + path + " is unusable: " + e.getMessage());
            }
        }

        private static Map<String, Kernel> parse(Properties props, String origin) {
            Map<String, Kernel> kernels = new TreeMap<>();
            for (String key : props.stringPropertyNames()) {
                if (!key.endsWith(BUFFERS)) {
                    continue;
                }
                String name = key.substring(0, key.length() - BUFFERS.length());
                if (!KernelManifest.isIdentifier(name)) {
                    throw new IllegalArgumentException(origin + ": '" + name + "' is not a kernel name");
                }
                String ro = props.getProperty(name + ".ro");
                String push = props.getProperty(name + ".push");
                if (ro == null || push == null) {
                    throw new IllegalArgumentException(origin + ": kernel " + name + " lacks "
                            + (ro == null ? name + ".ro" : name + ".push"));
                }
                KernelSignature sig;
                try {
                    sig = KernelSignature.parse(props.getProperty(key), ro, push);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(origin + ": kernel " + name + ": " + e.getMessage(), e);
                }
                String sg = props.getProperty(name + ".sg", "0").trim();
                if (!sg.equals("0") && !sg.equals("1")) {
                    throw new IllegalArgumentException(origin + ": " + name + ".sg must be 0 or 1, not '" + sg + "'");
                }
                kernels.put(name, new Kernel(sig, sg.equals("1")));
            }
            return Collections.unmodifiableMap(kernels);
        }
    }
}
