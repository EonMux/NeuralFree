package com.eonmux.neuralfree.gguf.engine;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

/**
 * The target view of two-level verification ({@code neuralfree.draft.tlv}, SpecPilot's TLV, arXiv 2611.02501):
 * its {@link #backend()} runs the same graph as the target's backend over the target's own memory, reads and
 * writes the target's KV cache, and never decides a token — the decoder uses it to verify a draft tree greedily
 * (no commit), so the exact target's one-pass verify of the composed proposal stays the only thing that samples
 * and commits. The view takes one of two shapes:
 *
 * <ul>
 * <li>a quantized copy — a second forward backend running {@code Q8_0} or {@code Q6_K} rows of the target's
 * projection matrices beside the target's own passes; or</li>
 * <li>a nested W12 view — the target's own backend, whose projection matrices stream as W12 rows, opened for
 * the Q round inside an 8-bit Q-view scope ({@link #openQScope}) that routes them through the nested 8-bit
 * view of their code bytes: no copy, no second backend, no added weight bytes.</li>
 * </ul>
 *
 * <p>A copied view owns only what was built beside the target: its weights (and, on a device target, its upload
 * and weight store). The target's memory, scratch and kernels are shared and stay with the target; closing the
 * view leaves the target untouched. Opened by the GGUF engines ({@code GGUFModel#openQuantizedView},
 * {@code GGUFModelGPU#openQuantizedView}); other models refuse by name.
 */
public final class QuantizedView implements AutoCloseable {

    private final ForwardBackend backend;
    private final List<AutoCloseable> owned;
    private final String description;
    private final long weightBytes;
    private final int bits;
    private final Supplier<QScope> qScope;

    /**
     * A view over {@code backend} owning {@code owned}: the resources built beside the target, closed most recent
     * first on {@link #close()}. Its Q round runs outside any scope.
     *
     * @param description how the view names itself in messages and logs
     * @param weightBytes the copy's weight bytes where the target runs them
     * @param bits        the copy's bits per weight, 6 or 8
     */
    public QuantizedView(ForwardBackend backend, List<AutoCloseable> owned, String description, long weightBytes,
                         int bits) {
        this(backend, owned, description, weightBytes, bits, () -> QScope.NONE);
    }

    /**
     * A view over {@code backend} owning {@code owned}, whose Q round opens {@link #openQScope} a scope supplied
     * by {@code qScope} — on a nested W12 view the target's own backend under its Q-view scope, on a copy none.
     *
     * @param description how the view names itself in messages and logs
     * @param weightBytes the weight bytes the view adds where the target runs them, zero for a nested view
     * @param bits        the view's bits per weight, 6 or 8
     */
    public QuantizedView(ForwardBackend backend, List<AutoCloseable> owned, String description, long weightBytes,
                         int bits, Supplier<QScope> qScope) {
        this.backend = backend;
        this.owned = Collections.unmodifiableList(new ArrayList<>(owned));
        this.description = description;
        this.weightBytes = weightBytes;
        this.bits = bits;
        this.qScope = java.util.Objects.requireNonNull(qScope, "qScope");
    }

    /** The backend running the view's Q round over the target's memory (the copy's, or the target's own). */
    public ForwardBackend backend() {
        return backend;
    }

    /**
     * Opens the scope the Q round runs the view's passes inside: on a target whose projection matrices stream
     * as W12 rows this routes them through the nested 8-bit view of their code bytes, and on a copied target it
     * does nothing. The caller closes the scope before the exact round reads the same weights.
     */
    public QScope openQScope() {
        return qScope.get();
    }

    /** The weight bytes the view adds where the target runs them, zero for a nested W12 view. */
    public long weightBytes() {
        return weightBytes;
    }

    /** The view's bits per weight (6 or 8). */
    public int bits() {
        return bits;
    }

    @Override
    public String toString() {
        return description;
    }

    /** Closes what the view built beside the target, most recent first; the target's own state stays. */
    @Override
    public void close() throws IOException {
        IOException failure = null;
        for (int i = owned.size() - 1; i >= 0; i--) {
            try {
                owned.get(i).close();
            } catch (IOException e) {
                if (failure != null) {
                    failure.addSuppressed(e);
                } else {
                    failure = e;
                }
            } catch (Exception e) {
                if (failure != null) {
                    failure.addSuppressed(e);
                } else {
                    failure = new IOException("closing the quantized target view failed", e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** Collects the resources of a view under construction, closing them if the build fails. */
    public static final class Build {
        private final List<AutoCloseable> owned = new ArrayList<>();

        /** Adds a resource the view owns and closes last. */
        public Build owns(AutoCloseable resource) {
            owned.add(resource);
            return this;
        }

        /** Closes what was added, most recent first, chaining {@code failure} if closing fails. */
        public RuntimeException close(RuntimeException failure) {
            RuntimeException first = failure;
            for (int i = owned.size() - 1; i >= 0; i--) {
                try {
                    owned.get(i).close();
                } catch (Throwable e) {
                    if (first == null) {
                        first = e instanceof RuntimeException ? (RuntimeException) e
                                : new RuntimeException(e);
                    } else {
                        first.addSuppressed(e);
                    }
                }
            }
            owned.clear();
            return first;
        }

        /** What the build collected, in acquisition order. */
        public List<AutoCloseable> owned() {
            return Collections.unmodifiableList(owned);
        }

        @Override
        public String toString() {
            return Arrays.toString(owned.toArray());
        }
    }

    /**
     * A scope the Q round's view passes run inside: on a nested W12 target it routes the target's own
     * projections through the nested 8-bit view of their W12 code bytes, and on a copied target it does
     * nothing. Closing is idempotent and throws nothing, so the round opens it in try-with-resources.
     */
    public interface QScope extends AutoCloseable {

        @Override
        void close();

        /** A scope whose open and close do nothing, for a view whose passes need no scoping. */
        QScope NONE = () -> {
        };
    }
}
