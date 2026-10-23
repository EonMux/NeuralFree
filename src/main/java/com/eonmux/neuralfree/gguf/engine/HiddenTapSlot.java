package com.eonmux.neuralfree.gguf.engine;

import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;

/**
 * Ownership of the single {@link HiddenTap} of each {@link ForwardBackend}. A backend delivers its hidden rows to one
 * observer ({@link ForwardBackend#setHiddenTap}); speculative drafters and the pockettts hidden-state reader each
 * install one, so at most one of them may be bound to a backend at a time. A user takes a {@link Lease} before it
 * installs its tap and closes the lease after it removed the tap.
 *
 * <p>Leases are held per backend instance; a closed backend that is no longer referenced is forgotten. Thread-safe.
 */
public final class HiddenTapSlot {

    private static final Map<ForwardBackend, Lease> HOLDERS = new WeakHashMap<>();

    private HiddenTapSlot() {
    }

    /**
     * Takes the tap slot of {@code backend} for {@code holder}.
     *
     * @param holder a description of the user, named in the error of a later conflicting acquire
     * @throws IllegalStateException when another user holds the slot, naming that user
     */
    public static Lease acquire(ForwardBackend backend, String holder) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(holder, "holder");
        synchronized (HOLDERS) {
            Lease current = HOLDERS.get(backend);
            if (current != null) {
                throw new IllegalStateException("the hidden-state tap of this model is already used by "
                        + current.holder + "; close it before binding " + holder);
            }
            Lease lease = new Lease(backend, holder);
            HOLDERS.put(backend, lease);
            return lease;
        }
    }

    /** The description of the user holding the slot of {@code backend}, or null when the slot is free. */
    public static String holder(ForwardBackend backend) {
        Objects.requireNonNull(backend, "backend");
        synchronized (HOLDERS) {
            Lease current = HOLDERS.get(backend);
            return current == null ? null : current.holder;
        }
    }

    /** One user's hold on a backend's tap slot; {@link #close()} frees it. Idempotent. */
    public static final class Lease implements AutoCloseable {
        private final ForwardBackend backend;
        private final String holder;
        private boolean released;

        private Lease(ForwardBackend backend, String holder) {
            this.backend = backend;
            this.holder = holder;
        }

        /** The backend whose slot this lease holds. */
        public ForwardBackend backend() {
            return backend;
        }

        /** The user this lease was taken for. */
        public String holder() {
            return holder;
        }

        @Override
        public void close() {
            synchronized (HOLDERS) {
                if (released) {
                    return;
                }
                released = true;
                if (HOLDERS.get(backend) == this) {
                    HOLDERS.remove(backend);
                }
            }
        }

        @Override
        public String toString() {
            return "HiddenTapSlot.Lease{" + holder + (released ? ", released" : "") + '}';
        }
    }
}
