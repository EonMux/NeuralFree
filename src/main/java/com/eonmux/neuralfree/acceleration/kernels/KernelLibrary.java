package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;

import java.util.Set;

/**
 * One kernel library (for example {@code "llm"}) as built for one backend type.
 *
 * <p>Kernel names are identical on every backend. A library turns a name into a
 * {@link KernelArtifact} that {@link ComputeBackend#loadKernel(KernelArtifact)}
 * can load on a given backend.</p>
 */
public interface KernelLibrary {

    /** Library id, for example {@code "llm"}. */
    String libraryId();

    /** The backend type this library serves. */
    ComputeBackend.BackendType backendType();

    /**
     * Whether this library can serve kernels on the given backend, with the reason when it cannot
     * (for example: kernels not built, device architecture not covered, inventory incomplete).
     */
    Availability availability(ComputeBackend backend);

    /**
     * Resolves one kernel for the given backend.
     *
     * @throws UnsupportedKernelException naming the kernel, the backend and the reason when the
     *                                    kernel is not available
     */
    KernelArtifact resolve(String kernelName, ComputeBackend backend);

    /** Unmodifiable set of every kernel name this library provides. */
    Set<String> kernelNames();
}
