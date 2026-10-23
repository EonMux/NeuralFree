package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.BackendRequirement;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

/**
 * Backend-selection requirement that a kernel library is available on the backend, as decided by
 * {@link KernelSet#availability(ComputeBackend, String)}. Immutable.
 */
public final class KernelLibraryRequirement implements BackendRequirement {

    private final String libraryId;

    private KernelLibraryRequirement(String libraryId) {
        this.libraryId = libraryId;
    }

    /**
     * Requirement for one library, for example {@code of(KernelLibraries.LLM)}.
     *
     * @throws IllegalArgumentException when the id is null or empty
     */
    public static KernelLibraryRequirement of(String libraryId) {
        if (libraryId == null || libraryId.isEmpty()) {
            throw new IllegalArgumentException("library id must not be null or empty");
        }
        return new KernelLibraryRequirement(libraryId);
    }

    public String libraryId() {
        return libraryId;
    }

    /** Null when the library is available on the backend, otherwise the reason it is not. */
    @Override
    public String check(ComputeBackend backend) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        Availability a = KernelSet.availability(backend, libraryId);
        return a.available() ? null : a.reason();
    }

    @Override
    public boolean equals(Object o) {
        return this == o || (o instanceof KernelLibraryRequirement
                && libraryId.equals(((KernelLibraryRequirement) o).libraryId));
    }

    @Override
    public int hashCode() {
        return libraryId.hashCode();
    }

    @Override
    public String toString() {
        return "KernelLibraryRequirement{" + libraryId + "}";
    }
}
