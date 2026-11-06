package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.hip.HipComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.hip.HipKernelSignatures;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * HIP side of a kernel library: hipcc-built code objects on the classpath, one per gfx target the
 * build compiled, with the same layout and inventory as the CUDA side.
 *
 * <p>Build outputs under {@code hipRoot} ({@code kernels/&lt;id&gt;/hip}, the sibling of the CUDA
 * root):</p>
 * <ul>
 *   <li>{@code index.tsv}: {@code kernel_name<TAB>code_object_resource}, one kernel per line, the
 *       resource the object that covers the device's gfx target. A resource without {@code /} is
 *       a file name inside {@code hipRoot}; otherwise it is a classpath path.</li>
 *   <li>{@code build.properties}: {@code hipcc.version}, {@code hip.archs} (comma list of the
 *       {@code --offload-arch} targets the build compiled, for example {@code gfx1100,gfx942}),
 *       plus the optional {@code fa.smem.floats} (the {@code NF_FA_SMEM_FLOATS} the objects were
 *       compiled with; when present it must equal {@link FaSmem#MAX_FLOATS}, the value
 *       {@link KernelSet#faSmemFloats()} reports on HIP).</li>
 * </ul>
 *
 * <p><b>Argument signatures.</b> The kernel inventory {@code <sourceRoot>/inventory.txt} records
 * each kernel's parameters exactly as the CUDA side reads it; {@link #resolve} registers the
 * signatures in {@link HipKernelSignatures}, so the HIP backend checks every launch of a kernel
 * against its declared parameters.</p>
 *
 * <p>The library is unavailable, with the reason, when the kernels were not built, when the
 * build compiled no target for the device's processor ({@code hip.archs} misses the device's
 * {@code gcnArchName}, capability suffixes ignored), or, with {@code requireCompleteInventory},
 * when a kernel of the inventory whose source is not a {@code [cuda]} manifest entry is missing
 * from the index. The {@code [cuda]} sources (the PTX-only mma.sync kernels) compile on no HIP
 * pass, so the HIP index holds the scalar-twin and non-mma kernels only; the host selects the
 * mma kernels through DeviceFeature.COOPERATIVE_MATRIX on CUDA and Vulkan.</p>
 */
public final class HipKernelLibrary implements KernelLibrary {

    /** Index file name inside the HIP root. */
    public static final String INDEX_FILE = "index.tsv";
    /** Build metadata file name inside the HIP root. */
    public static final String BUILD_FILE = "build.properties";
    /** Build metadata key of the {@code NF_FA_SMEM_FLOATS} value the code objects were compiled with. */
    static final String FA_SMEM_KEY = "fa.smem.floats";

    private final KernelLibraryConfig config;
    private final String hipRoot;
    private final ClassLoader classLoader;
    private final Object buildLock = new Object();
    private volatile Build build;

    public HipKernelLibrary(KernelLibraryConfig config) {
        this(config, HipKernelLibrary.class.getClassLoader());
    }

    public HipKernelLibrary(KernelLibraryConfig config, ClassLoader classLoader) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.classLoader = Objects.requireNonNull(classLoader, "class loader must not be null");
        this.hipRoot = config.sourceRoot() + "/hip";
    }

    @Override
    public String libraryId() {
        return config.libraryId();
    }

    @Override
    public ComputeBackend.BackendType backendType() {
        return ComputeBackend.BackendType.HIP;
    }

    /** The configuration this library was created from. */
    public KernelLibraryConfig config() {
        return config;
    }

    /** Classpath root of the prebuilt code objects ({@code <sourceRoot>/hip}). */
    public String hipRoot() {
        return hipRoot;
    }

    @Override
    public Availability availability(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.HIP) {
            return Availability.no("the HIP " + libraryId() + " library cannot serve a "
                    + backend.backendType() + " backend");
        }
        if (!backend.supportsArtifact(KernelArtifact.Kind.HIP_IMAGE)) {
            return Availability.no("the HIP backend cannot load prebuilt kernel code objects");
        }
        Build b = build();
        if (b.error != null) {
            return Availability.no(b.error);
        }
        if (!(backend instanceof HipComputeBackend)) {
            return Availability.no("the backend exposes no gfx target");
        }
        String gcnArchName = ((HipComputeBackend) backend).getHipDevice().getGfxTarget().gcnArchName();
        if (!archCovered(b.archs, gcnArchName)) {
            return Availability.no("HIP kernels were built for " + archList(b.archs) + " (hipcc " + b.hipccVersion
                    + "); the device runs " + (gcnArchName == null ? "an unknown target" : gcnArchName));
        }
        if (config.requireCompleteInventory()) {
            return KernelManifest.portableInventoryCoverage(classLoader, libraryId(), config.sourceRoot(),
                    b.index.keySet(), "the HIP kernel index");
        }
        return Availability.ok();
    }

    @Override
    public KernelArtifact resolve(String kernelName, ComputeBackend backend) {
        KernelManifest.requireKernelName(kernelName);
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.HIP) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": the HIP " + libraryId()
                    + " library cannot serve a " + backend.backendType() + " backend");
        }
        Build b = build();
        if (b.error != null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on HIP: " + b.error);
        }
        String resource = b.index.get(kernelName);
        if (resource == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " is not in the HIP kernel index classpath:"
                    + hipRoot + "/" + INDEX_FILE);
        }
        if (classLoader.getResource(resource) == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": HIP code object classpath:" + resource
                    + " listed in the index is missing");
        }
        KernelSignature signature = b.signatures.get(kernelName);
        if (signature != null) {
            try {
                HipKernelSignatures.register(resource, kernelName, signature);
            } catch (IllegalStateException e) {
                throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on HIP: "
                        + e.getMessage(), e);
            }
        }
        return KernelArtifact.hipImage(kernelName, resource);
    }

    /** The kernels of the HIP index; empty when the HIP kernels were not built. */
    @Override
    public Set<String> kernelNames() {
        Build b = build();
        return b.error != null ? Collections.<String>emptySet() : b.index.keySet();
    }

    @Override
    public String toString() {
        return "HipKernelLibrary{" + libraryId() + " @ " + hipRoot + "}";
    }

    /**
     * Whether a build that compiled the targets of {@code archs} runs on a device whose processor
     * name is {@code gcnArchName}: the names compare capability-suffix-stripped and
     * case-insensitively ({@code gfx90a:xnack-} matches a {@code gfx90a} target), because
     * {@code --offload-arch=gfxXXXX} produces code that runs on exactly that processor.
     */
    static boolean archCovered(List<String> archs, String gcnArchName) {
        if (gcnArchName == null || gcnArchName.trim().isEmpty()) {
            return false;
        }
        String device = stripSuffix(gcnArchName);
        for (String arch : archs) {
            if (stripSuffix(arch).equals(device)) {
                return true;
            }
        }
        return false;
    }

    /** The name without its {@code :capability} suffix, lowercased and trimmed. */
    private static String stripSuffix(String name) {
        String s = name.trim().toLowerCase(Locale.ROOT);
        int colon = s.indexOf(':');
        return colon >= 0 ? s.substring(0, colon) : s;
    }

    /** Parses an arch list such as {@code gfx1100,gfx942}; empty input gives an empty list. */
    static List<String> parseArchs(String value, String key) {
        List<String> out = new ArrayList<>();
        if (value == null || value.trim().isEmpty()) {
            return out;
        }
        for (String part : value.split(",")) {
            String p = part.trim();
            requireGfxTarget(p, key);
            out.add(stripSuffix(p));
        }
        return out;
    }

    /** Rejects anything that is not a gfx target name ({@code gfx} plus hex digits, suffix allowed). */
    private static void requireGfxTarget(String name, String key) {
        String base = stripSuffix(name);
        if (!base.startsWith("gfx") || base.length() <= 3) {
            throw new IllegalArgumentException(key + " entry '" + name + "' is not a gfx target");
        }
        for (int i = 3; i < base.length(); i++) {
            char c = base.charAt(i);
            if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) {
                throw new IllegalArgumentException(key + " entry '" + name + "' is not a gfx target"
                        + " (expected gfx plus hex digits)");
            }
        }
    }

    private static String archList(List<String> archs) {
        if (archs.isEmpty()) {
            return "(no target)";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < archs.size(); i++) {
            sb.append(i == 0 ? "" : ",").append(archs.get(i));
        }
        return sb.toString();
    }

    private Build build() {
        Build b = build;
        if (b != null) {
            return b;
        }
        synchronized (buildLock) {
            if (build == null) {
                build = Build.load(classLoader, hipRoot, config.sourceRoot());
            }
            return build;
        }
    }

    /** Parsed build outputs, or the reason they are unusable. */
    private static final class Build {
        final Map<String, String> index;
        final Map<String, KernelSignature> signatures;
        final String hipccVersion;
        final List<String> archs;
        final String error;

        private Build(Map<String, String> index, Map<String, KernelSignature> signatures, String hipccVersion,
                      List<String> archs, String error) {
            this.index = index;
            this.signatures = signatures;
            this.hipccVersion = hipccVersion;
            this.archs = archs;
            this.error = error;
        }

        static Build failed(String error) {
            return new Build(Collections.<String, String>emptyMap(), Collections.<String, KernelSignature>emptyMap(),
                    null, Collections.<String>emptyList(), error);
        }

        static Build load(ClassLoader cl, String hipRoot, String sourceRoot) {
            String indexPath = hipRoot + "/" + INDEX_FILE;
            String buildPath = hipRoot + "/" + BUILD_FILE;
            try {
                String indexText = KernelManifest.readText(cl, indexPath);
                if (indexText == null) {
                    return failed("HIP kernels were not built: hipcc not found at build time (classpath:"
                            + indexPath + " is missing)");
                }
                String buildText = KernelManifest.readText(cl, buildPath);
                if (buildText == null) {
                    return failed("HIP kernel build metadata classpath:" + buildPath + " is missing");
                }
                Map<String, String> index = parseIndex(indexText, hipRoot, "classpath:" + indexPath);
                Properties props = new Properties();
                props.load(new StringReader(buildText));
                String hipcc = required(props, "hipcc.version", buildPath);
                List<String> archs = parseArchs(props.getProperty("hip.archs"), "hip.archs");
                if (archs.isEmpty()) {
                    return failed("HIP kernel build metadata classpath:" + buildPath + " lists no hip.archs");
                }
                String smemMismatch = checkFaSmem(props.getProperty(FA_SMEM_KEY), buildPath);
                if (smemMismatch != null) {
                    return failed(smemMismatch);
                }
                String inventoryPath = sourceRoot + "/" + KernelManifest.INVENTORY_FILE;
                String inventoryText = KernelManifest.readText(cl, inventoryPath);
                Map<String, KernelSignature> signatures = inventoryText == null
                        ? Collections.<String, KernelSignature>emptyMap()
                        : CudaKernelLibrary.parseSignatures(inventoryText, "classpath:" + inventoryPath);
                return new Build(index, signatures, hipcc, Collections.unmodifiableList(archs), null);
            } catch (IOException | RuntimeException e) {
                return failed("HIP kernel build outputs under classpath:" + hipRoot + " are unusable: "
                        + e.getMessage());
            }
        }

        /**
         * Null when the recorded {@code NF_FA_SMEM_FLOATS} is absent or equals {@link FaSmem#MAX_FLOATS};
         * otherwise the reason the code objects cannot be used: their flash kernels would hold a shared
         * tile of a different size than the one the engine plans with.
         */
        private static String checkFaSmem(String value, String path) {
            if (value == null || value.trim().isEmpty()) {
                return null;
            }
            int built;
            try {
                built = Integer.parseInt(value.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("classpath:" + path + " has a non-integer " + FA_SMEM_KEY
                        + " '" + value.trim() + "'", e);
            }
            if (built == FaSmem.MAX_FLOATS) {
                return null;
            }
            return "HIP kernels were built with NF_FA_SMEM_FLOATS=" + built + " (classpath:" + path
                    + "), but the HIP runtime plans flash-attention tiles for " + FaSmem.MAX_FLOATS;
        }

        private static String required(Properties props, String key, String path) {
            String v = props.getProperty(key);
            if (v == null || v.trim().isEmpty()) {
                throw new IllegalArgumentException("classpath:" + path + " has no " + key);
            }
            return v.trim();
        }

        private static Map<String, String> parseIndex(String text, String hipRoot, String origin) {
            Map<String, String> index = new LinkedHashMap<>();
            String[] lines = text.split("\r?\n", -1);
            for (int i = 0; i < lines.length; i++) {
                String line = lines[i].trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] cols = line.split("\t");
                if (cols.length != 2 || cols[0].trim().isEmpty() || cols[1].trim().isEmpty()) {
                    throw new IllegalArgumentException(origin + ":" + (i + 1)
                            + ": expected kernel_name<TAB>code_object_resource");
                }
                String name = cols[0].trim();
                if (!KernelManifest.isIdentifier(name)) {
                    throw new IllegalArgumentException(origin + ":" + (i + 1) + ": '" + name + "' is not a kernel name");
                }
                String res = cols[1].trim();
                String resource = res.indexOf('/') < 0 ? hipRoot + "/" + res
                        : (res.startsWith("/") ? res.substring(1) : res);
                if (index.put(name, resource) != null) {
                    throw new IllegalArgumentException(origin + ":" + (i + 1) + ": duplicate kernel '" + name + "'");
                }
            }
            return Collections.unmodifiableMap(index);
        }
    }
}
