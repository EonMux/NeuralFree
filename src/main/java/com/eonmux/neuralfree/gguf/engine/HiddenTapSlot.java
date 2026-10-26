package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;

/**
 * Ownership of the hidden-state taps of each {@link ForwardBackend}. A backend delivers its hidden rows to the
 * observers it is told about ({@link ForwardBackend#setHiddenTap}); speculative drafters and the pockettts
 * hidden-state reader each install one, a target may serve several at a time through a chain of drafters,
 * and every user takes a {@link Lease} before it installs its tap and closes the lease after it removed the tap.
 *
 * <p>A lease carries a {@link TapKind}: {@link TapKind#DRAFTER_FEED} marks the tap as a drafter's input, the one
 * reader that feeds a drafter rather than observes the model, so consumers of the target that count input tap
 * readers versus observation channels can tell them apart. {@link TapKind#OBSERVER} is the default of every
 * {@code acquire} that names no kind (plain generation instrumentation, embeddings heads, diagnostics). A
 * {@code DRAFTER_FEED} lease supports one {@link DrafterFeed} tap at most; any other tap installed by any route (a
 * lease or {@code setHiddenTap} directly) is an observation channel: a reader costs forward-pass hits and never
 * correctness. {@code DRAFTER_FEED} leases of a backend coexist (several drafter types chain on one target); an
 * observer lease is exclusive against every other lease of the same backend, because an observation channel must
 * know it is alone.
 *
 * <p>Leases are held per backend instance; a closed backend that is no longer referenced is forgotten. Thread-safe.
 */
public final class HiddenTapSlot {

    /** {@code DRAFTER_FEED}: the lease's tap feeds a speculative drafter's input ({@link DrafterFeed}). */
    public enum TapKind {
        DRAFTER_FEED,
        /** Every tap that is not a drafter's input: an observation channel. */
        OBSERVER
    }

    private static final Map<ForwardBackend, List<Lease>> HOLDERS = new WeakHashMap<>();

    private HiddenTapSlot() {
    }

    /**
     * Takes an observer lease of {@code backend} for {@code holder}: exclusive against any other lease of the
     * backend.
     *
     * @param holder a description of the user, named in the error of a later conflicting acquire
     * @throws IllegalStateException when another user holds the slot, naming that user
     */
    public static Lease acquire(ForwardBackend backend, String holder) {
        return acquire(backend, holder, TapKind.OBSERVER, null);
    }

    /**
     * Takes a lease of {@code backend} for {@code holder}. A {@link TapKind#DRAFTER_FEED} lease coexists with the
     * backend's other drafter-feed leases; an {@link TapKind#OBSERVER} one is exclusive against every lease.
     *
     * @param tapLayers the tap points the lease's tap reads, in tap-point coordinates ({@code [0, nLayer]}), or an
     *                  empty list for every point; DRAFTER_FEED feeds declare their layer set, OBSERVER leases take none
     * @throws IllegalStateException    when an exclusive lease is in conflict, naming the holder and the kind
     * @throws IllegalArgumentException for a null kind
     */
    public static Lease acquire(ForwardBackend backend, String holder, TapKind kind, int[] tapLayers) {
        Objects.requireNonNull(kind, "kind");
        int[] layers = kind == TapKind.DRAFTER_FEED
                ? (tapLayers == null ? new int[0] : tapLayers.clone()) : null;
        StringBuilder conflict = null;
        synchronized (HOLDERS) {
            List<Lease> current = HOLDERS.get(backend);
            boolean hasObserver = current != null && current.stream().anyMatch(l -> l.kind == TapKind.OBSERVER);
            if (current != null && !current.isEmpty()
                    && (kind == TapKind.OBSERVER || hasObserver)) {
                conflict = new StringBuilder();
                for (Lease l : current) {
                    conflict.append(conflict.length() > 0 ? ", " : "").append(l.holder).append(" (")
                            .append(l.kind.name().toLowerCase(java.util.Locale.ROOT)).append(')');
                }
            }
            if (conflict != null) {
                throw new IllegalStateException("the hidden-state taps of this model are already used by "
                        + conflict + "; close " + (current.size() == 1 ? "it" : "them") + " before binding "
                        + holder);
            }
            Lease lease = new Lease(backend, holder, kind, layers);
            HOLDERS.computeIfAbsent(backend, b -> new ArrayList<>()).add(lease);
            return lease;
        }
    }

    /**
     * The descriptions of every user holding a lease of {@code backend}, or null while none does; several
     * DRAFTER_FEED leases are joined with {@code ", "}.
     */
    public static String holder(ForwardBackend backend) {
        Objects.requireNonNull(backend, "backend");
        synchronized (HOLDERS) {
            List<Lease> current = HOLDERS.get(backend);
            if (current == null || current.isEmpty()) {
                return null;
            }
            StringBuilder sb = new StringBuilder();
            for (Lease l : current) {
                sb.append(sb.length() > 0 ? ", " : "").append(l.holder);
            }
            return sb.toString();
        }
    }

    /** Every lease of {@code backend}, or an empty list. */
    public static List<Lease> leases(ForwardBackend backend) {
        synchronized (HOLDERS) {
            List<Lease> current = HOLDERS.get(backend);
            return current == null ? List.of() : List.copyOf(current);
        }
    }

    /** One user's hold on a backend's tap slot; {@link #close()} frees it. Idempotent. */
    public static final class Lease implements AutoCloseable {
        private final ForwardBackend backend;
        private final String holder;
        private final TapKind kind;
        private final int[] tapLayers;
        private boolean released;

        private Lease(ForwardBackend backend, String holder, TapKind kind, int[] tapLayers) {
            this.backend = backend;
            this.holder = holder;
            this.kind = kind;
            this.tapLayers = tapLayers;
        }

        /** The backend whose slot this lease holds. */
        public ForwardBackend backend() {
            return backend;
        }

        /** The user this lease was taken for. */
        public String holder() {
            return holder;
        }

        /** The lease's tap kind. */
        public TapKind kind() {
            return kind;
        }

        /**
         * The tap points the lease's feed reads, in tap-point coordinates; an empty array reads every point, null
         * for an OBSERVER lease.
         */
        public int[] tapLayers() {
            return tapLayers == null ? null : tapLayers.clone();
        }

        /**
         * Builds the {@link DrafterFeed} that routes a {@link HiddenTap} through this lease's declared
         * {@link TapKind#DRAFTER_FEED} layer set: it observes only layers of the set (every point when the set is
         * empty) and leaves {@code r_out} of the backend's requests unchanged.
         *
         * @param delegate the consumer's own tap implementation
         * @throws IllegalStateException for an OBSERVER lease (an observer is not a drafter feed)
         * @throws IllegalStateException when the lease was already released
         */
        public DrafterFeed drafterFeed(HiddenTap delegate) {
            if (kind != TapKind.DRAFTER_FEED) {
                throw new IllegalStateException("an observation lease of " + holder + " is not a drafter feed");
            }
            if (released) {
                throw new IllegalStateException("the drafter-feed lease of " + holder + " was already released");
            }
            Objects.requireNonNull(delegate, "tap");
            return new DrafterFeed(this, delegate);
        }

        @Override
        public void close() {
            synchronized (HOLDERS) {
                if (released) {
                    return;
                }
                released = true;
                List<Lease> current = HOLDERS.get(backend);
                if (current != null) {
                    current.remove(this);
                    if (current.isEmpty()) {
                        HOLDERS.remove(backend);
                    }
                }
            }
        }

        @Override
        public String toString() {
            return "HiddenTapSlot.Lease{" + holder + ", " + kind.name().toLowerCase(java.util.Locale.ROOT)
                    + (released ? ", released" : "") + '}';
        }
    }

    /**
     * The tap class only a {@link TapKind#DRAFTER_FEED} lease creates, which PC's request accounting does not count:
     * it routes the rows of its lease's declared layer set to the drafter's own {@link HiddenTap} and stays an
     * input channel, not an observation of the target.
     */
    public static final class DrafterFeed implements HiddenTap {

        private final Lease lease;
        private final HiddenTap delegate;

        private DrafterFeed(Lease lease, HiddenTap delegate) {
            this.lease = lease;
            this.delegate = delegate;
        }

        /** The DRAFTER_FEED lease this feed runs under. */
        public Lease lease() {
            return lease;
        }

        @Override
        public void onHidden(int layer, float[] rows, int nRows, int width) {
            if (!wants(layer)) {
                return;
            }
            delegate.onHidden(layer, rows, nRows, width);
        }

        @Override
        public boolean wants(int layer) {
            return !lease.released && delegate.wants(layer) && inSet(layer);
        }

        private boolean inSet(int layer) {
            int[] layers = lease.tapLayers;
            if (layers.length == 0) {
                return true;
            }
            for (int l : layers) {
                if (l == layer) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public String toString() {
            return "HiddenTapSlot.DrafterFeed{" + lease + '}';
        }
    }
}
