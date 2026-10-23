package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * OpenCL side of a kernel library: the program is the manifest concatenation, compiled at run time.
 *
 * <p>One program exists per (library, subgroup flag, {@code NF_FA_SMEM_FLOATS}):
 * {@code programKey = libraryId + (sg ? "__sg" : "") + "#" + smem}. Build options are the library's
 * {@link KernelLibraryConfig#openClOptions()} ({@link #BASE_OPTIONS} for the LLM library) plus
 * {@code -DNF_FA_SMEM_FLOATS=<n>}, plus {@code -DUSE_SUBGROUPS=1} when the library has subgroup variants
 * ({@link KernelLibraryConfig#subgroupVariants()}) and the device has subgroup reductions. The
 * {@code NF_FA_SMEM_FLOATS} define sizes the LLM flash-attention tiles; sources that do not reference it
 * are unaffected. No finite-math or fast-relaxed-math option is used: the kernels rely on IEEE infinities
 * as reduction and masking sentinels.</p>
 *
 * <p>The assembled source is computed once per library and shared by every artifact.</p>
 */
public final class OpenClKernelLibrary implements KernelLibrary {

    /** Static OpenCL options of the LLM library (the five-argument {@link KernelLibraryConfig} profile). */
    public static final String BASE_OPTIONS = "-cl-std=CL1.2 -cl-mad-enable -cl-no-signed-zeros";

    private final KernelLibraryConfig config;
    private final ClassLoader classLoader;
    private final Object sourceLock = new Object();
    private volatile String source;

    public OpenClKernelLibrary(KernelLibraryConfig config) {
        this(config, OpenClKernelLibrary.class.getClassLoader());
    }

    public OpenClKernelLibrary(KernelLibraryConfig config, ClassLoader classLoader) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        this.classLoader = Objects.requireNonNull(classLoader, "class loader must not be null");
    }

    /**
     * Build options of a program variant: the library's static options, then the flash shared-memory size,
     * then the subgroup define.
     *
     * @param libraryOptions the library's {@link KernelLibraryConfig#openClOptions()}
     * @throws IllegalArgumentException when {@code faSmemFloats} is not positive
     */
    public static String buildOptions(String libraryOptions, boolean subgroups, int faSmemFloats) {
        Objects.requireNonNull(libraryOptions, "library options must not be null");
        if (faSmemFloats <= 0) {
            throw new IllegalArgumentException("NF_FA_SMEM_FLOATS must be positive: " + faSmemFloats);
        }
        String defines = "-DNF_FA_SMEM_FLOATS=" + faSmemFloats + (subgroups ? " -DUSE_SUBGROUPS=1" : "");
        return libraryOptions.isEmpty() ? defines : libraryOptions + " " + defines;
    }

    /** Program cache key of a program variant. */
    public static String programKey(String libraryId, boolean subgroups, int faSmemFloats) {
        return libraryId + (subgroups ? "__sg" : "") + "#" + faSmemFloats;
    }

    /**
     * Whether OpenCL programs for this device are built with {@code USE_SUBGROUPS}: the device reports
     * subgroup arithmetic or subgroup support. The OpenCL backend reports either only when the
     * {@code -cl-std=CL1.2} programs can enable a subgroup extension ({@code cl_intel_subgroups};
     * {@code cl_khr_subgroups} needs OpenCL C 2.0), so the {@code __sg} program key and
     * {@link KernelSet#subgroups()} match what the kernels compile. A library without subgroup variants
     * never uses them, whatever this returns.
     */
    public static boolean useSubgroups(DeviceCapabilities caps) {
        Objects.requireNonNull(caps, "capabilities must not be null");
        return caps.hasFeature(DeviceFeature.SUBGROUP_ARITHMETIC) || caps.supportsSubgroups();
    }

    @Override
    public String libraryId() {
        return config.libraryId();
    }

    @Override
    public ComputeBackend.BackendType backendType() {
        return ComputeBackend.BackendType.OPENCL;
    }

    /** The configuration this library was created from. */
    public KernelLibraryConfig config() {
        return config;
    }

    @Override
    public Availability availability(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.OPENCL) {
            return Availability.no("the OpenCL " + libraryId() + " library cannot serve a "
                    + backend.backendType() + " backend");
        }
        if (!backend.supportsArtifact(KernelArtifact.Kind.OPENCL_SOURCE)) {
            return Availability.no("the OpenCL backend cannot load OpenCL source artifacts");
        }
        KernelManifest manifest;
        try {
            manifest = KernelManifest.load(classLoader, config.sourceRoot());
        } catch (RuntimeException e) {
            return Availability.no("kernel sources unusable: " + e.getMessage());
        }
        List<String> missing = KernelSourceAssembler.missingSources(manifest, classLoader, config.sourceRoot());
        if (!missing.isEmpty()) {
            return Availability.no(KernelSourceAssembler.missingMessage(config.sourceRoot(), missing));
        }
        return Availability.ok();
    }

    @Override
    public KernelArtifact resolve(String kernelName, ComputeBackend backend) {
        KernelManifest.requireKernelName(kernelName);
        Objects.requireNonNull(backend, "backend must not be null");
        if (backend.backendType() != ComputeBackend.BackendType.OPENCL) {
            throw new UnsupportedKernelException("kernel " + kernelName + ": the OpenCL " + libraryId()
                    + " library cannot serve a " + backend.backendType() + " backend");
        }
        DeviceCapabilities caps = backend.capabilities();
        boolean sg = config.subgroupVariants() && useSubgroups(caps);
        int smem = FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, caps);
        return KernelArtifact.openClSource(kernelName, programKey(libraryId(), sg, smem), this::source,
                buildOptions(config.openClOptions(), sg, smem));
    }

    /**
     * The kernels of the checked-in inventory (every kernel of the untagged manifest entries); empty
     * when {@code inventory.txt} is not on the classpath. OpenCL-only kernels of {@code [opencl]} entries are
     * resolvable but not listed.
     */
    @Override
    public Set<String> kernelNames() {
        Optional<Set<String>> inventory = KernelManifest.inventory(classLoader, config.sourceRoot());
        return inventory.isPresent() ? inventory.get() : Collections.<String>emptySet();
    }

    /**
     * The assembled program source, computed on first use.
     *
     * @throws UnsupportedKernelException when the manifest or a source file is missing or unreadable
     */
    public String source() {
        String s = source;
        if (s != null) {
            return s;
        }
        synchronized (sourceLock) {
            if (source == null) {
                try {
                    KernelManifest manifest = KernelManifest.load(classLoader, config.sourceRoot());
                    source = KernelSourceAssembler.assemble(manifest, classLoader, config.sourceRoot());
                } catch (RuntimeException e) {
                    throw new UnsupportedKernelException("cannot assemble the OpenCL " + libraryId()
                            + " program from classpath:" + config.sourceRoot() + ": " + e.getMessage(), e);
                }
            }
            return source;
        }
    }

    @Override
    public String toString() {
        return "OpenClKernelLibrary{" + libraryId() + " @ " + config.sourceRoot() + "}";
    }
}
