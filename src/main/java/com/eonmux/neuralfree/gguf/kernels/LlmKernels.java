package com.eonmux.neuralfree.gguf.kernels;

import com.eonmux.neuralfree.acceleration.BackendRequirement;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryRequirement;
import com.eonmux.neuralfree.acceleration.kernels.KernelSet;

import java.util.Collection;
import java.util.Objects;
import java.util.Set;

/**
 * The LLM inference kernels on one backend: a thin wrapper over
 * {@code KernelSet.open(backend, KernelLibraries.LLM)} that the GGUF engine uses for every kernel lookup.
 *
 * <p>Kernel names are identical on OpenCL, CUDA and Vulkan. Handles are owned by the backend.</p>
 */
public final class LlmKernels {

    private final KernelSet set;

    private LlmKernels(KernelSet set) {
        this.set = set;
    }

    /**
     * Opens the LLM kernels on a backend.
     *
     * @throws UnsupportedKernelException with the reason when the LLM kernels are unavailable on it
     */
    public static LlmKernels open(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        return new LlmKernels(KernelSet.open(backend, KernelLibraries.LLM));
    }

    /** Whether the LLM kernels can be opened on a backend, with the reason when they cannot. */
    public static Availability availability(ComputeBackend backend) {
        return KernelSet.availability(backend, KernelLibraries.LLM);
    }

    /** Backend-selection requirement that the LLM kernels are available. */
    public static BackendRequirement requirement() {
        return KernelLibraryRequirement.of(KernelLibraries.LLM);
    }

    /**
     * The handle of a required kernel.
     *
     * @throws UnsupportedKernelException naming the kernel, the backend and the reason
     */
    public KernelHandle get(String name) {
        return set.get(name);
    }

    /** The handle of an optional kernel, or null when it is unavailable (logged once per name). */
    public KernelHandle tryGet(String name) {
        return set.tryGet(name);
    }

    /** Whether the kernel can be provided on this backend. */
    public boolean has(String name) {
        return set.has(name);
    }

    /** Loads every named kernel now; throws for the first that cannot be provided. */
    public void prewarm(Collection<String> names) {
        set.prewarm(names);
    }

    /** Loads a kernel and prepares it for dispatches with this work-group size. */
    public void prewarm(String name, long[] localWorkSize) {
        set.prewarm(name, localWorkSize);
    }

    /** {@code NF_FA_SMEM_FLOATS} of the flash kernels on this backend. */
    public int faSmemFloats() {
        return set.faSmemFloats();
    }

    /** Whether the kernels use subgroup reductions on this backend. */
    public boolean subgroups() {
        return set.subgroups();
    }

    /** Preferred work-group size of row-reduce kernels on this backend. */
    public int preferredMatvecLocal() {
        return set.preferredMatvecLocal();
    }

    /** Work-group size for row-reduce kernels: {@code min(LLM_DIMX, preferredMatvecLocal())}. */
    public int matvecLocal() {
        return Math.min(LlmKernelConstants.LLM_DIMX, set.preferredMatvecLocal());
    }

    public ComputeBackend backend() {
        return set.backend();
    }

    public DeviceCapabilities caps() {
        return set.caps();
    }

    /** Kernels the LLM library lists for this backend type. */
    public Set<String> kernelNames() {
        return set.kernelNames();
    }

    @Override
    public String toString() {
        return "LlmKernels{" + set + "}";
    }
}
