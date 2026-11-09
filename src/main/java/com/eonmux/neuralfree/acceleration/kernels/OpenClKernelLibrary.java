package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * OpenCL side of a kernel library: the program is the manifest concatenation, compiled at run time.
 *
 * <p>One program exists per (library, subgroup flag, {@code NF_FA_SMEM_FLOATS}, MMQ tile, required
 * subgroup size): {@code programKey = libraryId + (sg ? "__sg" : "") + "#" + smem + tile + (sgSize > 0
 * ? "@" + sgSize : "")}. Build options are the library's {@link KernelLibraryConfig#openClOptions()}
 * ({@link #BASE_OPTIONS} for the LLM library) plus {@code -DNF_FA_SMEM_FLOATS=<n>}, plus
 * {@code -DUSE_SUBGROUPS=1} when the library has subgroup variants
 * ({@link KernelLibraryConfig#subgroupVariants()}) and the device has subgroup reductions, plus
 * {@code -DNF_REQD_SG_SIZE=<n>} when such a program runs on a device with a supported subgroup size
 * range ({@link #requiredSubgroupSize}); the kernels pin their subgroup reductions to it. The
 * {@code NF_FA_SMEM_FLOATS} define sizes the LLM flash-attention tiles; sources that do not reference it
 * are unaffected. A device whose {@link MmqConfig} is not the packed-dot default also gets
 * {@code -DNF_MMQ_TM} / {@code -DNF_MMQ_TN} / {@code -DNF_MMQ_TK} ({@link MmqConfig#kernelDefines}),
 * which sizes the MMQ GEMM tile; the default tile is built into the kernel header, so its options stay
 * unchanged. No finite-math or fast-relaxed-math option is used: the kernels rely on IEEE infinities
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
        return buildOptions(libraryOptions, subgroups, faSmemFloats, "");
    }

    /**
     * Build options of a program variant with further defines (the MMQ tile of
     * {@link com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig#kernelDefines}): the library's static
     * options, the flash shared-memory size, the subgroup define and then {@code extraDefines}.
     *
     * @throws IllegalArgumentException when {@code faSmemFloats} is not positive
     */
    public static String buildOptions(String libraryOptions, boolean subgroups, int faSmemFloats,
                                      String extraDefines) {
        return buildOptions(libraryOptions, subgroups, faSmemFloats, 0, extraDefines);
    }

    /**
     * Build options of a program variant pinned to a subgroup size: as the four-argument form, plus
     * {@code -DNF_REQD_SG_SIZE=<requiredSubgroupSize>} between the subgroup define and
     * {@code extraDefines} when {@code requiredSubgroupSize} is positive. The size must be a power of
     * two (every subgroup size is), or 0 for no pin.
     *
     * @throws IllegalArgumentException when {@code faSmemFloats} is not positive or
     *         {@code requiredSubgroupSize} is negative or not a power of two
     */
    public static String buildOptions(String libraryOptions, boolean subgroups, int faSmemFloats,
                                      int requiredSubgroupSize, String extraDefines) {
        Objects.requireNonNull(libraryOptions, "library options must not be null");
        Objects.requireNonNull(extraDefines, "extra defines must not be null");
        if (faSmemFloats <= 0) {
            throw new IllegalArgumentException("NF_FA_SMEM_FLOATS must be positive: " + faSmemFloats);
        }
        if (requiredSubgroupSize < 0 || (requiredSubgroupSize & (requiredSubgroupSize - 1)) != 0) {
            throw new IllegalArgumentException("required subgroup size must be a power of two: "
                    + requiredSubgroupSize);
        }
        String defines = "-DNF_FA_SMEM_FLOATS=" + faSmemFloats + (subgroups ? " -DUSE_SUBGROUPS=1" : "")
                + (requiredSubgroupSize > 0 ? " -DNF_REQD_SG_SIZE=" + requiredSubgroupSize : "")
                + (extraDefines.isEmpty() ? "" : " " + extraDefines);
        return libraryOptions.isEmpty() ? defines : libraryOptions + " " + defines;
    }

    /** Program cache key of a program variant. */
    public static String programKey(String libraryId, boolean subgroups, int faSmemFloats) {
        return programKey(libraryId, subgroups, faSmemFloats, "");
    }

    /** Program cache key of a program variant with the MMQ tile suffix of {@code tileKey}. */
    public static String programKey(String libraryId, boolean subgroups, int faSmemFloats, String tileKey) {
        return programKey(libraryId, subgroups, faSmemFloats, 0, tileKey);
    }

    /** Program cache key of a program variant pinned to a subgroup size: the tile-key form with the
     * {@code @<size>} suffix appended when {@code requiredSubgroupSize} is positive. */
    public static String programKey(String libraryId, boolean subgroups, int faSmemFloats,
                                    int requiredSubgroupSize, String tileKey) {
        return libraryId + (subgroups ? "__sg" : "") + "#" + faSmemFloats + tileKey
                + (requiredSubgroupSize > 0 ? "@" + requiredSubgroupSize : "");
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

    /**
     * The subgroup size a subgroup program is pinned to, or 0 when the compiler keeps its per-kernel
     * choice. A program is pinned only when it compiles with subgroup reductions and the device
     * reports a supported size range: the OpenCL backend fills the capabilities' range from
     * {@code CL_DEVICE_SUB_GROUP_SIZES_INTEL}, which it queries only when the device exposes
     * {@code cl_intel_required_subgroup_size}, so every size in that range is one the device accepts
     * for every kernel. Pinning holds each single-row/multi-row matvec pair
     * ({@code {T}_MatVec}/{@code {T}_MatVecN}, {@code {T}_MatVec_Q8}/{@code {T}_MatVecN_Q8}) at one
     * subgroup size, which the pair's bit-identical reduction contract requires: the hardware
     * reduction width otherwise follows the compiler's per-kernel SIMD choice. The pin is the
     * largest size in the range: the matvec reductions are bandwidth-bound, and on the Intel
     * Arrow-Lake iGPU a Qwen3 0.6B Q8_0 forward step measured 669 ms at 9 rows pinned to 32 against
     * 748 unpinned, 760 pinned to 16 and 1097 pinned to 8 (147, 140, 184 and 256 ms at one row).
     * Programs without subgroup reductions get 0 whatever size the capabilities carry (NVIDIA and
     * AMD report their warp/wave size without exposing a required-size extension), and so does a
     * subgroup program on a device that reports no size range at all.
     */
    public static int requiredSubgroupSize(DeviceCapabilities caps, boolean subgroups) {
        Objects.requireNonNull(caps, "capabilities must not be null");
        if (!subgroups || caps.getMinSubgroupSize() <= 0) {
            return 0;
        }
        int size = caps.getMaxSubgroupSize();
        return size > 0 ? size : 0;
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
        int reqdSg = requiredSubgroupSize(caps, sg);
        int smem = FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, caps);
        MmqConfig mmq = MmqConfig.forDevice(ComputeBackend.BackendType.OPENCL, caps.getComputeCapability());
        return KernelArtifact.openClSource(kernelName,
                programKey(libraryId(), sg, smem, reqdSg, mmq.kernelKey(false)),
                this::source,
                buildOptions(config.openClOptions(), sg, smem, reqdSg, mmq.kernelDefines(false)));
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
