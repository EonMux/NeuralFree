package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The causal masks of a backend: for a {@code querySeq x keySeq} score block, the BOOL tensor that is true at every key
 * position {@code j > i} of query row {@code i}. A mask is built once per shape and backend and kept as a
 * {@link BackendResources} attachment, so the attention compositions reuse it instead of uploading positions and
 * comparing them in every layer of every step.
 *
 * <p>At most {@link #CAPACITY} shapes are kept per backend; the least recently used one is closed when a new shape
 * would exceed it. {@link #future} returns a new handle of the cached mask ({@link Tensor#detach()}, sharing its
 * storage) that the caller closes, so a mask evicted by another thread while the caller still uses it stays valid
 * until the caller's handle is closed. The backend's {@code close()} closes the cache. Thread-safe.</p>
 */
final class CausalMasks implements AutoCloseable {

    /** Shapes kept per backend. */
    static final int CAPACITY = 8;

    private static final Object KEY = new Object() {
        @Override
        public String toString() {
            return "causal-masks";
        }
    };

    private final ComputeBackend backend;
    /** Access-ordered; guarded by {@code this}. */
    private final LinkedHashMap<Long, Tensor> masks = new LinkedHashMap<>(16, 0.75f, true);
    private boolean closed;

    private CausalMasks(ComputeBackend backend) {
        this.backend = backend;
    }

    /** The mask of {@code querySeq x keySeq} on {@code backend}: a new handle of the cached mask, closed by the caller. */
    static Tensor future(ComputeBackend backend, int querySeq, int keySeq) {
        Objects.requireNonNull(backend, "backend must not be null");
        if (querySeq < 1 || keySeq < 1) {
            throw new IllegalArgumentException("causal mask of " + querySeq + " x " + keySeq + " positions");
        }
        return BackendResources.attachment(backend, KEY, () -> new CausalMasks(backend)).get(querySeq, keySeq);
    }

    private synchronized Tensor get(int querySeq, int keySeq) {
        if (closed) {
            throw new IllegalStateException("the causal masks of " + backend.backendType() + " were closed");
        }
        long key = ((long) querySeq << 32) | keySeq;
        Tensor mask = masks.get(key);
        if (mask != null) {
            return mask.detach();
        }
        mask = build(querySeq, keySeq);
        masks.put(key, mask);
        if (masks.size() > CAPACITY) {
            Iterator<Map.Entry<Long, Tensor>> eldest = masks.entrySet().iterator();
            Tensor evicted = eldest.next().getValue();
            eldest.remove();
            evicted.close();
        }
        return mask.detach();
    }

    private Tensor build(int querySeq, int keySeq) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor rows = Tensor.fromArray(backend, positions(querySeq));
             Tensor rowCol = rows.reshape(Shape.of(querySeq, 1));
             Tensor cols = Tensor.fromArray(backend, positions(keySeq));
             Tensor colRow = cols.reshape(Shape.of(1, keySeq))) {
            return TensorOps.gt(colRow, rowCol);
        }
    }

    private static float[] positions(int n) {
        float[] p = new float[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        return p;
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (Tensor t : masks.values()) {
            try {
                t.close();
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        masks.clear();
        if (failure != null) {
            throw failure;
        }
    }
}
