package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaKernelSignatures;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * CUDA side of a kernel library: prebuilt fatbins on the classpath, one per non-OpenCL-only source file.
 *
 * <p>Build outputs under {@code cudaRoot}:</p>
 * <ul>
 *   <li>{@code index.tsv}: {@code kernel_name<TAB>fatbin_resource}, one kernel per line. A resource
 *       without {@code /} is a file name inside {@code cudaRoot}; otherwise it is a classpath path.</li>
 *   <li>{@code build.properties}: {@code nvcc.version}, {@code cuda.driver.min} (the toolkit's CUDA
 *       version in the {@code cuDriverGetVersion} encoding {@code 1000*major + 10*minor}, or
 *       {@code major.minor}),
 *       {@code sass.archs} and {@code ptx.archs} (comma lists such as {@code 75,80,120}; {@code sm_} and
 *       {@code compute_} prefixes are accepted), plus the optional {@code fa.smem.floats} (the
 *       {@code NF_FA_SMEM_FLOATS} the modules were compiled with; when present it must equal
 *       {@link FaSmem#MAX_FLOATS}, the value {@link KernelSet#faSmemFloats()} reports on CUDA).</li>
 * </ul>
 *
 * <p><b>Argument signatures.</b> The kernel inventory {@code <sourceRoot>/inventory.txt}
 * ({@code name<TAB>source<TAB>buffers<TAB>scalars<TAB>kinds<TAB>layout}) records each kernel's buffer
 * count, scalar kinds and which parameters, in declaration order, are buffers. {@link #resolve}
 * registers them in {@link CudaKernelSignatures}, so the CUDA backend checks every launch of the kernel
 * against its declared parameters, position by position, as Vulkan does against the shader index. An inventory line without the tab-separated count columns records no signature; a
 * missing inventory leaves the kernels unchecked.</p>
 *
 * <p>The library is unavailable, with the reason, when the kernels were not built, when neither a SASS
 * image of the device's major architecture (with minor version not above the device's) nor a PTX image
 * of an architecture not above the device's exists, when the driver is too old for the images the device
 * will run ({@link #requiredDriver}), or, with {@code requireCompleteInventory}, when a kernel of the
 * kernel inventory is missing from the index.</p>
 */
public final class CudaKernelLibrary implements KernelLibrary {

    /** Index file name inside {@code cudaRoot}. */
    public static final String INDEX_FILE = "index.tsv";
    /** Build metadata file name inside {@code cudaRoot}. */
    public static final String BUILD_FILE = "build.properties";
    /** Build metadata key of the {@code NF_FA_SMEM_FLOATS} value the modules were compiled with. */
    static final String FA_SMEM_KEY = "fa.smem.floats";

    private final KernelLibraryConfig config;
    private final ClassLoader classLoader;
    private final Object buildLock = new Object();
    private volatile Build build;

    public CudaKernelLibrary(KernelLibraryConfig config) {
        this(config, CudaKernelLibrary.class.getClassLoader());
    }

    public CudaKernelLibrary(KernelLibraryConfig config, ClassLoader classLoader) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.classLoader = Objects.requireNonNull(classLoader, "class loader must not be null");
    }

    @Override
    public String libraryId() {
        return config.libraryId();
    }

    @Override
    public ComputeBackend.BackendType backendType() {
        return ComputeBackend.BackendType.CUDA;
    }

    /** The configuration this library was created from. */
    public KernelLibraryConfig config() {
        return config;
    }

    @Override
    public Availability availability(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.CUDA) {
            return Availability.no("the CUDA " + libraryId() + " library cannot serve a "
                    + backend.backendType() + " backend");
        }
        if (!backend.supportsArtifact(KernelArtifact.Kind.CUDA_IMAGE)) {
            return Availability.no("the CUDA backend cannot load prebuilt kernel images");
        }
        Build b = build();
        if (b.error != null) {
            return Availability.no(b.error);
        }
        int cc = backend.capabilities().getComputeCapability();
        if (cc <= 0) {
            return Availability.no("the device's compute capability is unknown");
        }
        if (!archCovered(b.sassArchs, b.ptxArchs, cc)) {
            return Availability.no("CUDA kernels were built for SASS " + archList("sm_", b.sassArchs) + " and PTX "
                    + archList("compute_", b.ptxArchs) + " (nvcc " + b.nvccVersion + "); the device is sm_" + cc);
        }
        int driver;
        try {
            driver = driverVersion();
        } catch (RuntimeException | LinkageError e) {
            return Availability.no("cannot query the CUDA driver version: " + e.getMessage());
        }
        boolean sass = sassCovered(b.sassArchs, cc);
        int required = requiredDriver(b.driverMin, sass);
        if (driver < required) {
            return Availability.no("the CUDA driver supports CUDA " + formatVersion(driver)
                    + ", but the kernels (nvcc " + b.nvccVersion + ", " + (sass ? "SASS" : "PTX JIT")
                    + " on sm_" + cc + ") need CUDA " + formatVersion(required) + " or newer");
        }
        if (config.requireCompleteInventory()) {
            return KernelManifest.inventoryCoverage(classLoader, libraryId(), config.sourceRoot(), b.index.keySet(),
                    "the CUDA kernel index");
        }
        return Availability.ok();
    }

    @Override
    public KernelArtifact resolve(String kernelName, ComputeBackend backend) {
        KernelManifest.requireKernelName(kernelName);
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.CUDA) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": the CUDA " + libraryId()
                    + " library cannot serve a " + backend.backendType() + " backend");
        }
        Build b = build();
        if (b.error != null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on CUDA: " + b.error);
        }
        String resource = b.index.get(kernelName);
        if (resource == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + " is not in the CUDA kernel index classpath:"
                    + config.cudaRoot() + "/" + INDEX_FILE);
        }
        if (classLoader.getResource(resource) == null) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": CUDA image classpath:" + resource
                    + " listed in the index is missing");
        }
        KernelSignature signature = b.signatures.get(kernelName);
        if (signature != null) {
            try {
                CudaKernelSignatures.register(resource, kernelName, signature);
            } catch (IllegalStateException e) {
                throw new UnsupportedKernelException("kernel " + kernelName + " unavailable on CUDA: "
                        + e.getMessage(), e);
            }
        }
        return KernelArtifact.cudaImage(kernelName, resource);
    }

    /** The kernels of the CUDA index; empty when the CUDA kernels were not built. */
    @Override
    public Set<String> kernelNames() {
        Build b = build();
        return b.error != null ? Collections.<String>emptySet() : b.index.keySet();
    }

    @Override
    public String toString() {
        return "CudaKernelLibrary{" + libraryId() + " @ " + config.cudaRoot() + "}";
    }

    /**
     * Whether a device of compute capability {@code cc} ({@code 10*major + minor}) can run the build:
     * a SASS image of the same major with minor not above the device's, or any PTX image of an
     * architecture not above the device's (JIT-compiled by the driver).
     */
    static boolean archCovered(List<Integer> sassArchs, List<Integer> ptxArchs, int cc) {
        if (sassCovered(sassArchs, cc)) {
            return true;
        }
        for (int ptx : ptxArchs) {
            if (ptx <= cc) {
                return true;
            }
        }
        return false;
    }

    /** Whether a SASS image of the device's major architecture with minor not above the device's exists. */
    static boolean sassCovered(List<Integer> sassArchs, int cc) {
        for (int sass : sassArchs) {
            if (sass / 10 == cc / 10 && sass % 10 <= cc % 10) {
                return true;
            }
        }
        return false;
    }

    /**
     * Oldest driver ({@code cuDriverGetVersion} encoding) that runs the build on a device. SASS images
     * load on every driver of the toolkit's major release (CUDA minor-version compatibility), so they need
     * {@code 1000 * major} of {@code cuda.driver.min}. PTX is JIT-compiled by the driver, which must support
     * the toolkit's PTX ISA, so PTX-only devices need {@code cuda.driver.min} itself.
     */
    static int requiredDriver(int driverMin, boolean sass) {
        return sass ? (driverMin / 1000) * 1000 : driverMin;
    }

    /** Parses a {@code cuDriverGetVersion}-encoded version, given either as {@code 13000} or {@code 13.0}. */
    static int parseDriverVersion(String value) {
        String v = value.trim();
        int dot = v.indexOf('.');
        try {
            if (dot < 0) {
                int encoded = Integer.parseInt(v);
                if (encoded <= 0) {
                    throw new IllegalArgumentException("driver version must be positive: " + value);
                }
                return encoded;
            }
            int major = Integer.parseInt(v.substring(0, dot));
            int minor = Integer.parseInt(v.substring(dot + 1));
            if (major <= 0 || minor < 0 || minor > 99) {
                throw new IllegalArgumentException("invalid driver version " + value);
            }
            return major * 1000 + minor * 10;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("driver version '" + value + "' is neither 13000 nor 13.0 form", e);
        }
    }

    /** Parses an arch list such as {@code 75,80,sm_120}; empty input gives an empty list. */
    static List<Integer> parseArchs(String value, String key) {
        List<Integer> out = new ArrayList<>();
        if (value == null || value.trim().isEmpty()) {
            return out;
        }
        for (String part : value.split(",")) {
            String p = part.trim();
            if (p.startsWith("sm_")) {
                p = p.substring(3);
            } else if (p.startsWith("compute_")) {
                p = p.substring(8);
            }
            try {
                int arch = Integer.parseInt(p);
                if (arch < 10) {
                    throw new NumberFormatException("arch below sm_10");
                }
                out.add(arch);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(key + " entry '" + part.trim() + "' is not an architecture", e);
            }
        }
        return out;
    }

    /**
     * The argument signatures of an inventory: one per line that carries the count and kind columns
     * ({@code name<TAB>source<TAB>buffers<TAB>scalars<TAB>kinds}, optionally followed by
     * {@code <TAB>layout}, which adds the positional buffer-or-scalar check of
     * {@link KernelSignature#parseInventory(String, String, String, String)}); a line without
     * tab-separated columns (a name, optionally followed by space-separated text) has none.
     *
     * @throws IllegalArgumentException naming the origin and line of a malformed or duplicate entry
     */
    static Map<String, KernelSignature> parseSignatures(String text, String origin) {
        Map<String, KernelSignature> out = new LinkedHashMap<>();
        String[] lines = text.split("\r?\n", -1);
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i].trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String[] cols = lines[i].split("\t", -1);
            String name = cols.length == 1 ? line.split("[\t ]+", 2)[0] : cols[0].trim();
            if (!KernelManifest.isIdentifier(name)) {
                throw new IllegalArgumentException(origin + ":" + (i + 1) + ": '" + name + "' is not a kernel name");
            }
            if (cols.length == 1) {
                continue;
            }
            if (cols.length != 5 && cols.length != 6) {
                throw new IllegalArgumentException(origin + ":" + (i + 1)
                        + ": expected kernel_name<TAB>source<TAB>buffers<TAB>scalars<TAB>kinds[<TAB>layout], got "
                        + cols.length + " columns");
            }
            KernelSignature signature;
            try {
                signature = cols.length == 6
                        ? KernelSignature.parseInventory(cols[2], cols[3], cols[4], cols[5])
                        : KernelSignature.parseInventory(cols[2], cols[3], cols[4]);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(origin + ":" + (i + 1) + ": kernel " + name + ": "
                        + e.getMessage(), e);
            }
            if (out.put(name, signature) != null) {
                throw new IllegalArgumentException(origin + ":" + (i + 1) + ": duplicate kernel '" + name + "'");
            }
        }
        return Collections.unmodifiableMap(out);
    }

    private static int driverVersion() {
        int[] version = new int[1];
        int rc = JCudaDriver.cuDriverGetVersion(version);
        if (rc != CUresult.CUDA_SUCCESS) {
            throw new IllegalStateException("cuDriverGetVersion failed: " + CUresult.stringFor(rc));
        }
        return version[0];
    }

    private static String formatVersion(int encoded) {
        return (encoded / 1000) + "." + ((encoded % 1000) / 10);
    }

    private static String archList(String prefix, List<Integer> archs) {
        if (archs.isEmpty()) {
            return "(none)";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < archs.size(); i++) {
            sb.append(i == 0 ? "" : ",").append(prefix).append(archs.get(i));
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
                build = Build.load(classLoader, config.cudaRoot(), config.sourceRoot());
            }
            return build;
        }
    }

    /** Parsed build outputs, or the reason they are unusable. */
    private static final class Build {
        final Map<String, String> index;
        final Map<String, KernelSignature> signatures;
        final String nvccVersion;
        final int driverMin;
        final List<Integer> sassArchs;
        final List<Integer> ptxArchs;
        final String error;

        private Build(Map<String, String> index, Map<String, KernelSignature> signatures, String nvccVersion,
                      int driverMin, List<Integer> sassArchs, List<Integer> ptxArchs, String error) {
            this.index = index;
            this.signatures = signatures;
            this.nvccVersion = nvccVersion;
            this.driverMin = driverMin;
            this.sassArchs = sassArchs;
            this.ptxArchs = ptxArchs;
            this.error = error;
        }

        static Build failed(String error) {
            return new Build(Collections.<String, String>emptyMap(), Collections.<String, KernelSignature>emptyMap(),
                    null, 0, Collections.<Integer>emptyList(), Collections.<Integer>emptyList(), error);
        }

        static Build load(ClassLoader cl, String cudaRoot, String sourceRoot) {
            String indexPath = cudaRoot + "/" + INDEX_FILE;
            String buildPath = cudaRoot + "/" + BUILD_FILE;
            try {
                String indexText = KernelManifest.readText(cl, indexPath);
                if (indexText == null) {
                    return failed("CUDA kernels were not built: nvcc not found at build time (classpath:"
                            + indexPath + " is missing)");
                }
                String buildText = KernelManifest.readText(cl, buildPath);
                if (buildText == null) {
                    return failed("CUDA kernel build metadata classpath:" + buildPath + " is missing");
                }
                Map<String, String> index = parseIndex(indexText, cudaRoot, "classpath:" + indexPath);
                Properties props = new Properties();
                props.load(new StringReader(buildText));
                String nvcc = required(props, "nvcc.version", buildPath);
                int driverMin = parseDriverVersion(required(props, "cuda.driver.min", buildPath));
                List<Integer> sass = parseArchs(props.getProperty("sass.archs"), "sass.archs");
                List<Integer> ptx = parseArchs(props.getProperty("ptx.archs"), "ptx.archs");
                if (sass.isEmpty() && ptx.isEmpty()) {
                    return failed("CUDA kernel build metadata classpath:" + buildPath
                            + " lists neither sass.archs nor ptx.archs");
                }
                String smemMismatch = checkFaSmem(props.getProperty(FA_SMEM_KEY), buildPath);
                if (smemMismatch != null) {
                    return failed(smemMismatch);
                }
                String inventoryPath = sourceRoot + "/" + KernelManifest.INVENTORY_FILE;
                String inventoryText = KernelManifest.readText(cl, inventoryPath);
                Map<String, KernelSignature> signatures = inventoryText == null
                        ? Collections.<String, KernelSignature>emptyMap()
                        : parseSignatures(inventoryText, "classpath:" + inventoryPath);
                return new Build(index, signatures, nvcc, driverMin, Collections.unmodifiableList(sass),
                        Collections.unmodifiableList(ptx), null);
            } catch (IOException | RuntimeException e) {
                return failed("CUDA kernel build outputs under classpath:" + cudaRoot + " are unusable: "
                        + e.getMessage());
            }
        }

        /**
         * Null when the recorded {@code NF_FA_SMEM_FLOATS} is absent or equals {@link FaSmem#MAX_FLOATS};
         * otherwise the reason the modules cannot be used: their flash kernels would hold a shared tile of
         * a different size than the one the engine plans with.
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
            return "CUDA kernels were built with NF_FA_SMEM_FLOATS=" + built + " (classpath:" + path
                    + "), but the CUDA runtime plans flash-attention tiles for " + FaSmem.MAX_FLOATS;
        }

        private static String required(Properties props, String key, String path) {
            String v = props.getProperty(key);
            if (v == null || v.trim().isEmpty()) {
                throw new IllegalArgumentException("classpath:" + path + " has no " + key);
            }
            return v.trim();
        }

        private static Map<String, String> parseIndex(String text, String cudaRoot, String origin) {
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
                            + ": expected kernel_name<TAB>fatbin_resource");
                }
                String name = cols[0].trim();
                if (!KernelManifest.isIdentifier(name)) {
                    throw new IllegalArgumentException(origin + ":" + (i + 1) + ": '" + name + "' is not a kernel name");
                }
                String res = cols[1].trim();
                String resource = res.indexOf('/') < 0 ? cudaRoot + "/" + res
                        : (res.startsWith("/") ? res.substring(1) : res);
                if (index.put(name, resource) != null) {
                    throw new IllegalArgumentException(origin + ":" + (i + 1) + ": duplicate kernel '" + name + "'");
                }
            }
            return Collections.unmodifiableMap(index);
        }
    }
}
