package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.ThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The kernels of one library on one backend: the single lookup path from a kernel name to a
 * backend-owned {@link KernelHandle}.
 *
 * <p>{@link #get} resolves the name through the backend type's {@link KernelLibrary} into a
 * {@link KernelArtifact} and loads it with {@link ComputeBackend#loadKernel(KernelArtifact)}. Handles
 * are cached per name; a kernel that cannot be provided is remembered with its reason, so repeated
 * lookups fail fast. Handles belong to the backend and are released when it closes, so a kernel set
 * has no {@code close()}.</p>
 *
 * <p>{@link #open} fails fast with the reason when the library is unavailable on the backend, and
 * probe-loads one kernel of the library's kernel list, so an incompatible driver, device architecture
 * or program build surfaces at selection time rather than at the first layer.</p>
 */
@ThreadSafe
public final class KernelSet {

    private static final Logger LOGGER = LogManager.getLogger(KernelSet.class);

    /** Preferred row-reduce work-group size on OpenCL. */
    private static final int OPENCL_MATVEC_LOCAL = 256;
    /** Preferred row-reduce work-group size on CUDA (two warps). */
    private static final int CUDA_MATVEC_LOCAL = 64;
    /** Smallest preferred row-reduce work-group size on Vulkan. */
    private static final int VULKAN_MIN_MATVEC_LOCAL = 64;

    private final ComputeBackend backend;
    private final KernelLibrary library;
    private final int faSmemFloats;
    private final boolean subgroups;
    private final int preferredMatvecLocal;
    private final Map<String, KernelHandle> handles = new ConcurrentHashMap<>();
    private final Map<String, UnsupportedKernelException> failures = new ConcurrentHashMap<>();
    private final Set<String> reportedOptional = ConcurrentHashMap.newKeySet();
    private final Object loadLock = new Object();

    private KernelSet(ComputeBackend backend, KernelLibrary library) {
        this.backend = backend;
        this.library = library;
        DeviceCapabilities caps = backend.capabilities();
        this.faSmemFloats = FaSmem.faSmemFloats(backend.backendType(), caps);
        this.subgroups = subgroupVariants(library) && subgroups(backend.backendType(), caps);
        this.preferredMatvecLocal = preferredMatvecLocal(backend.backendType(), caps);
    }

    /**
     * Opens a library on a backend.
     *
     * @throws UnsupportedKernelException with the reason when the library is not registered for the
     *                                    backend type, is unavailable on the backend, or its probe
     *                                    kernel cannot be loaded
     */
    public static KernelSet open(ComputeBackend backend, String libraryId) {
        Objects.requireNonNull(backend, "backend must not be null");
        requireLibraryId(libraryId);
        Optional<KernelLibrary> library = lookup(backend, libraryId);
        if (!library.isPresent()) {
            throw new UnsupportedKernelException(describe(backend) + ": " + notRegistered(backend, libraryId));
        }
        Availability a = check(library.get(), backend);
        if (!a.available()) {
            throw new UnsupportedKernelException(describe(backend) + ": " + libraryId + " kernels unavailable: "
                    + a.reason());
        }
        KernelSet set = new KernelSet(backend, library.get());
        set.probe();
        return set;
    }

    /** Whether a library can be opened on a backend, with the reason when it cannot. Loads nothing. */
    public static Availability availability(ComputeBackend backend, String libraryId) {
        Objects.requireNonNull(backend, "backend must not be null");
        requireLibraryId(libraryId);
        Optional<KernelLibrary> library = lookup(backend, libraryId);
        if (!library.isPresent()) {
            return Availability.no(notRegistered(backend, libraryId));
        }
        return check(library.get(), backend);
    }

    /**
     * The handle of a kernel, loaded on first use and cached.
     *
     * @throws UnsupportedKernelException naming the kernel, the backend and the reason when the kernel
     *                                    cannot be provided
     * @throws GpuOutOfMemoryException    when the device runs out of memory while loading
     */
    public KernelHandle get(String name) {
        KernelManifest.requireKernelName(name);
        KernelHandle h = handles.get(name);
        if (h != null) {
            return h;
        }
        UnsupportedKernelException failed = failures.get(name);
        if (failed != null) {
            throw new UnsupportedKernelException(failed.getMessage(), failed);
        }
        synchronized (loadLock) {
            h = handles.get(name);
            if (h != null) {
                return h;
            }
            failed = failures.get(name);
            if (failed != null) {
                throw new UnsupportedKernelException(failed.getMessage(), failed);
            }
            try {
                KernelArtifact artifact = library.resolve(name, backend);
                h = backend.loadKernel(artifact);
                if (h == null) {
                    throw new UnsupportedKernelException("the backend returned no handle for " + artifact);
                }
            } catch (GpuOutOfMemoryException | IllegalStateException e) {
                throw e;
            } catch (RuntimeException e) {
                UnsupportedKernelException u = new UnsupportedKernelException("kernel " + name + " unavailable on "
                        + describe(backend) + ": " + reasonOf(e), e);
                failures.put(name, u);
                throw u;
            }
            handles.put(name, h);
            return h;
        }
    }

    /**
     * The handle of an optional kernel, or null when the kernel cannot be provided; the first miss of
     * each name is logged with its reason.
     */
    public KernelHandle tryGet(String name) {
        try {
            return get(name);
        } catch (UnsupportedKernelException e) {
            if (reportedOptional.add(name)) {
                LOGGER.info("Optional kernel {} not used: {}", name, e.getMessage());
            }
            return null;
        }
    }

    /** Whether the kernel can be provided on this backend; loads it when it was not loaded yet. */
    public boolean has(String name) {
        KernelManifest.requireKernelName(name);
        if (handles.containsKey(name)) {
            return true;
        }
        if (failures.containsKey(name)) {
            return false;
        }
        try {
            get(name);
            return true;
        } catch (UnsupportedKernelException e) {
            LOGGER.debug("Kernel {} not available: {}", name, e.getMessage());
            return false;
        }
    }

    /**
     * Loads every named kernel now (program build, module load, pipeline layout), so the first dispatch
     * does not pay for it.
     *
     * @throws UnsupportedKernelException for the first kernel that cannot be provided
     */
    public void prewarm(Collection<String> names) {
        Objects.requireNonNull(names, "names must not be null");
        for (String name : names) {
            get(name);
        }
    }

    /**
     * Loads a kernel and lets the backend prepare it for dispatches with this work-group size (Vulkan
     * creates the pipeline now; OpenCL checks the size against the kernel's compiled
     * {@code CL_KERNEL_WORK_GROUP_SIZE}).
     *
     * @throws UnsupportedKernelException when the kernel cannot be provided, or cannot run work-groups
     *                                    of this size on the device
     */
    public void prewarm(String name, long[] localWorkSize) {
        Objects.requireNonNull(localWorkSize, "localWorkSize must not be null");
        backend.prepareKernel(get(name), localWorkSize.clone());
    }

    /** {@code NF_FA_SMEM_FLOATS} the flash kernels of this backend were built or specialised with. */
    public int faSmemFloats() {
        return faSmemFloats;
    }

    /**
     * Whether the kernels use subgroup reductions. Never for a library without subgroup variants
     * ({@link KernelLibraryConfig#subgroupVariants()}); otherwise always on CUDA, on OpenCL when the program
     * is built with {@code USE_SUBGROUPS}, and on Vulkan when the device reports subgroup arithmetic (kernels
     * with an {@code __sg} module then use it).
     */
    public boolean subgroups() {
        return subgroups;
    }

    /**
     * Preferred work-group size of row-reduce (matvec) kernels: OpenCL 256, CUDA 64, Vulkan
     * {@code max(64, 2 * subgroupSize)}, each capped at the largest power of two not above the device's
     * maximum work-group size.
     */
    public int preferredMatvecLocal() {
        return preferredMatvecLocal;
    }

    public ComputeBackend backend() {
        return backend;
    }

    public DeviceCapabilities caps() {
        return backend.capabilities();
    }

    public String libraryId() {
        return library.libraryId();
    }

    /** The library this set resolves through. */
    public KernelLibrary library() {
        return library;
    }

    /** Unmodifiable set of the kernels the library lists for this backend type. */
    public Set<String> kernelNames() {
        return Collections.unmodifiableSet(library.kernelNames());
    }

    @Override
    public String toString() {
        return "KernelSet{" + library.libraryId() + " on " + describe(backend) + ", loaded=" + handles.size() + "}";
    }

    /**
     * Whether a library has subgroup variants: its configuration's flag for the built-in library classes;
     * true for any other implementation, whose artifacts this set cannot inspect.
     */
    static boolean subgroupVariants(KernelLibrary library) {
        if (library instanceof OpenClKernelLibrary) {
            return ((OpenClKernelLibrary) library).config().subgroupVariants();
        }
        if (library instanceof CudaKernelLibrary) {
            return ((CudaKernelLibrary) library).config().subgroupVariants();
        }
        if (library instanceof VulkanKernelLibrary) {
            return ((VulkanKernelLibrary) library).config().subgroupVariants();
        }
        return true;
    }

    /** Whether a library with subgroup variants uses them on a backend type and device. */
    static boolean subgroups(ComputeBackend.BackendType type, DeviceCapabilities caps) {
        switch (type) {
            case CUDA:
                return true;
            case OPENCL:
                return OpenClKernelLibrary.useSubgroups(caps);
            case VULKAN:
                return caps.hasFeature(DeviceFeature.SUBGROUP_ARITHMETIC);
            default:
                throw new IllegalArgumentException("unknown backend type " + type);
        }
    }

    static int preferredMatvecLocal(ComputeBackend.BackendType type, DeviceCapabilities caps) {
        int preferred;
        switch (type) {
            case OPENCL:
                preferred = OPENCL_MATVEC_LOCAL;
                break;
            case CUDA:
                preferred = CUDA_MATVEC_LOCAL;
                break;
            case VULKAN:
                preferred = Math.max(VULKAN_MIN_MATVEC_LOCAL, 2 * Math.max(0, caps.getSubgroupSize()));
                break;
            default:
                throw new IllegalArgumentException("unknown backend type " + type);
        }
        long maxWg = caps.getMaxWorkGroupSize();
        if (maxWg > 0 && maxWg < preferred) {
            preferred = Integer.highestOneBit((int) maxWg);
        }
        return preferred;
    }

    private void probe() {
        Set<String> names = library.kernelNames();
        if (names.isEmpty()) {
            LOGGER.debug("{} kernels on {}: no kernel list to probe", library.libraryId(), describe(backend));
            return;
        }
        String name = Collections.min(names);
        try {
            get(name);
        } catch (UnsupportedKernelException e) {
            throw new UnsupportedKernelException(describe(backend) + ": " + library.libraryId()
                    + " kernels unusable: probe load of " + name + " failed: " + e.getMessage(), e);
        }
    }

    private static Optional<KernelLibrary> lookup(ComputeBackend backend, String libraryId) {
        KernelLibraries.ensureDefaults();
        ComputeBackend.BackendType type = backend.backendType();
        if (type == null) {
            throw new IllegalArgumentException("backend " + backend + " reports no backend type");
        }
        return KernelLibraryRegistry.global().get(libraryId, type);
    }

    private static Availability check(KernelLibrary library, ComputeBackend backend) {
        try {
            Availability a = library.availability(backend);
            if (a == null) {
                return Availability.no(library + " returned no availability");
            }
            return a;
        } catch (RuntimeException e) {
            return Availability.no("availability check failed: " + reasonOf(e));
        }
    }

    private static String notRegistered(ComputeBackend backend, String libraryId) {
        return "no '" + libraryId + "' kernel library is registered for " + backend.backendType();
    }

    private static String describe(ComputeBackend backend) {
        String device;
        try {
            device = backend.capabilities().getDeviceName();
        } catch (RuntimeException e) {
            device = null;
        }
        return backend.backendType() + (device == null || device.isEmpty() ? "" : " (" + device + ")");
    }

    private static String reasonOf(Throwable e) {
        String m = e.getMessage();
        return m == null || m.isEmpty() ? e.getClass().getSimpleName() : m;
    }

    private static void requireLibraryId(String libraryId) {
        if (libraryId == null || libraryId.isEmpty()) {
            throw new IllegalArgumentException("library id must not be null or empty");
        }
    }
}
