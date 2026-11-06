package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;

import java.io.IOException;
import java.io.StringReader;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;

/**
 * Metal side of a kernel library: one MSL file per kernel, {@code <metalRoot>/<name>.metal}.
 *
 * <p>The Metal root is {@code <spvRoot>/metal}, so a library whose Vulkan shaders live under
 * {@code shaders/llm} has its Metal twins under {@code shaders/llm/metal}.
 * {@code <metalRoot>/index.properties} describes every kernel with the same four keys the Vulkan
 * index uses: {@code <name>.buffers} (buffer count), {@code <name>.ro} (one {@code 0}/{@code 1}
 * read-only flag per buffer), {@code <name>.push} (comma list of scalar kinds {@code i}, {@code u},
 * {@code f}) and {@code <name>.sg} (the subgroup flag of the Vulkan row, informational here: one
 * MSL file holds the kernel, translated from the {@code __sg} module when the row has one because
 * Metal SIMD groups are always available). Other keys are ignored.</p>
 *
 * <p>The files are generated at build time from the same SPIR-V modules the Vulkan backend uses
 * (SPIRV-Cross), so a kernel's MSL argument layout and signature are the Vulkan index's. Buffers
 * bind at MSL buffer indices {@code 0..n-1} and the push constants are packed into one constant
 * buffer at {@code n}, matching {@code MetalArgState}.</p>
 *
 * <p>{@link #resolve} returns a {@link KernelArtifact.Kind#METAL_SOURCE} artifact whose source is the
 * MSL file and whose program key is the file's classpath path, so the backend compiles one Metal
 * library per file. The indexed {@link KernelSignature} is the argument contract of every launch.</p>
 *
 * <p>With {@link KernelLibraryConfig#requireCompleteInventory()} the library is unavailable unless
 * the index covers every inventory kernel the manifest build provides, exactly as the Vulkan
 * library requires; a partly ported Metal shader set therefore never gets selected for a model.</p>
 */
public final class MetalKernelLibrary implements KernelLibrary {

    /** Subdirectory of {@code spvRoot} that holds the MSL sources. */
    public static final String METAL_DIR = "metal";
    /** Index file name inside the Metal root. */
    public static final String INDEX_FILE = "index.properties";
    /** File-name suffix of a Metal kernel source. */
    public static final String SOURCE_SUFFIX = ".metal";

    private final KernelLibraryConfig config;
    private final ClassLoader classLoader;
    private final Object indexLock = new Object();
    private volatile Index index;

    public MetalKernelLibrary(KernelLibraryConfig config) {
        this(config, MetalKernelLibrary.class.getClassLoader());
    }

    public MetalKernelLibrary(KernelLibraryConfig config, ClassLoader classLoader) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.classLoader = Objects.requireNonNull(classLoader, "class loader must not be null");
    }

    @Override
    public String libraryId() {
        return config.libraryId();
    }

    @Override
    public ComputeBackend.BackendType backendType() {
        return ComputeBackend.BackendType.METAL;
    }

    /** The configuration this library was created from. */
    public KernelLibraryConfig config() {
        return config;
    }

    /** Classpath directory that holds this library's MSL sources and index. */
    public String metalRoot() {
        return config.spvRoot() + "/" + METAL_DIR;
    }

    @Override
    public Availability availability(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.METAL) {
            return Availability.no("the Metal " + libraryId() + " library cannot serve a "
                    + backend.backendType() + " backend");
        }
        if (!backend.supportsArtifact(KernelArtifact.Kind.METAL_SOURCE)) {
            return Availability.no("the Metal backend cannot load Metal source artifacts");
        }
        Index idx = index();
        if (idx.error != null) {
            return Availability.no(idx.error);
        }
        if (config.requireCompleteInventory()) {
            return KernelManifest.portableInventoryCoverage(classLoader, libraryId(), config.sourceRoot(),
                    idx.kernels.keySet(), "the Metal shader index");
        }
        return Availability.ok();
    }

    @Override
    public KernelArtifact resolve(String kernelName, ComputeBackend backend) {
        KernelManifest.requireKernelName(kernelName);
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.METAL) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": the Metal " + libraryId()
                    + " library cannot serve a " + backend.backendType() + " backend");
        }
        Index idx = index();
        if (idx.error != null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on Metal: " + idx.error);
        }
        IndexedKernel kernel = idx.kernels.get(kernelName);
        if (kernel == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " is not in the Metal shader index"
                    + " classpath:" + metalRoot() + "/" + INDEX_FILE);
        }
        String resource = metalRoot() + "/" + kernelName + SOURCE_SUFFIX;
        String text = readSource(kernelName, resource);
        return KernelArtifact.metalSource(kernelName, resource, () -> text, "", kernel.signature);
    }

    /** The kernels of the Metal shader index; empty when the MSL sources were not built. */
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
            throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on Metal: " + idx.error);
        }
        IndexedKernel kernel = idx.kernels.get(kernelName);
        if (kernel == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " is not in the Metal shader index");
        }
        return kernel.signature;
    }

    @Override
    public String toString() {
        return "MetalKernelLibrary{" + libraryId() + " @ " + metalRoot() + "}";
    }

    private String readSource(String kernelName, String resource) {
        String text = KernelManifest.readText(classLoader, resource);
        if (text == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": Metal source classpath:" + resource
                    + " is missing");
        }
        return text;
    }

    private Index index() {
        Index i = index;
        if (i != null) {
            return i;
        }
        synchronized (indexLock) {
            if (index == null) {
                index = Index.load(classLoader, metalRoot());
            }
            return index;
        }
    }

    /** One indexed kernel. Immutable. */
    private static final class IndexedKernel {
        final KernelSignature signature;

        IndexedKernel(KernelSignature signature) {
            this.signature = signature;
        }
    }

    /** Parsed Metal shader index, or the reason it is unusable. */
    private static final class Index {
        private static final String BUFFERS = ".buffers";

        final Map<String, IndexedKernel> kernels;
        final String error;

        private Index(Map<String, IndexedKernel> kernels, String error) {
            this.kernels = kernels;
            this.error = error;
        }

        static Index load(ClassLoader cl, String metalRoot) {
            String path = metalRoot + "/" + INDEX_FILE;
            try {
                String text = KernelManifest.readText(cl, path);
                if (text == null) {
                    return new Index(Collections.<String, IndexedKernel>emptyMap(),
                            "Metal shaders were not built: classpath:" + path + " is missing");
                }
                Properties props = new Properties();
                props.load(new StringReader(text));
                return new Index(parse(props, "classpath:" + path), null);
            } catch (IOException | RuntimeException e) {
                return new Index(Collections.<String, IndexedKernel>emptyMap(),
                        "Metal shader index classpath:" + path + " is unusable: " + e.getMessage());
            }
        }

        private static Map<String, IndexedKernel> parse(Properties props, String origin) {
            Map<String, IndexedKernel> kernels = new TreeMap<>();
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
                kernels.put(name, new IndexedKernel(sig));
            }
            return Collections.unmodifiableMap(kernels);
        }
    }
}
