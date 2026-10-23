package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

/**
 * A condition a freshly created backend must meet before backend selection
 * accepts it, for example that a kernel library is available on it.
 */
@FunctionalInterface
public interface BackendRequirement {

    /** Accepts every backend. */
    BackendRequirement ANY = backend -> null;

    /**
     * Checks a backend.
     *
     * @return null when the backend is acceptable, otherwise a human-readable reason it is not
     */
    String check(ComputeBackend backend);

    /** A requirement that holds when both this one and {@code other} hold; reports the first failure. */
    default BackendRequirement and(BackendRequirement other) {
        if (other == null) {
            throw new IllegalArgumentException("requirement must not be null");
        }
        BackendRequirement self = this;
        return backend -> {
            String reason = self.check(backend);
            return reason != null ? reason : other.check(backend);
        };
    }
}
