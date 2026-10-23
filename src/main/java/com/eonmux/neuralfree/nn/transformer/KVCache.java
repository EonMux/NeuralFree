package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.Arrays;
import java.util.Objects;

/**
 * Per-layer key and value caches for incremental attention, each a preallocated zero-filled
 * {@code [batchSize, numHeads, maxSeqLen, headDim]} FLOAT32 tensor on the cache's backend.
 *
 * <p>{@link #append} copies new rows into the next sequence positions with a strided device copy
 * ({@link GpuCoreOps#copyInto}) when the train kernels are available on the backend, and through the CPU path
 * ({@link GpuOptimizerOps#storeFloats}) otherwise; {@link #getKeys} and {@link #getValues} return new tensors holding
 * the filled positions as independent copies, so later appends, {@link #clear()} and {@link #resize(int)} never change
 * a tensor returned earlier.</p>
 *
 * <p><b>Autograd.</b> A cache holds constants: every write and read runs under {@link ComputationGraph#noGrad()},
 * {@link #append} copies {@link Tensor#detach()} handles of the new keys and values, and the cache tensors and the
 * copies {@link #getKeys}/{@link #getValues} return never require grad, so no gradient flows through cached positions
 * and no cache write meets the in-place guard. The cache never holds a reference to an appended tensor.</p>
 */
public final class KVCache implements AutoCloseable {
    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int batchSize;
    private int maxSeqLen;

    private final Tensor[] keyCache;
    private final Tensor[] valueCache;
    private final int[] seqLengths;

    private ComputeBackend backend;
    private volatile boolean closed = false;

    public KVCache(int numLayers, int numHeads, int headDim, int maxSeqLen, int batchSize) {
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        if (headDim <= 0) {
            throw new IllegalArgumentException("headDim must be positive: " + headDim);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
        }

        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;

        this.keyCache = new Tensor[numLayers];
        this.valueCache = new Tensor[numLayers];
        this.seqLengths = new int[numLayers];
    }

    /** Allocates zero-filled caches on {@code backend} for every layer, replacing (and closing) any earlier ones. */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        ensureOpen();

        this.backend = backend;
        Shape cacheShape = cacheShape(maxSeqLen);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int layer = 0; layer < numLayers; layer++) {
                resetLayer(layer, cacheShape);
            }
        }
    }

    /**
     * Writes the values of {@code newKeys} and {@code newValues}, each {@code [batchSize, numHeads, n, headDim]} (any
     * views, which may require grad), into sequence positions {@code [len, len + n)} of layer {@code layerIdx}. The
     * copy is detached: it records nothing and keeps no reference to either tensor.
     *
     * @throws IllegalArgumentException when the layer is out of bounds, a tensor is not FLOAT32 on the cache's backend,
     *                                  or the shapes do not match the cache
     * @throws IllegalStateException    when the appended length would exceed the layer's capacity ({@link #maxSeqLen()})
     */
    public void append(int layerIdx, Tensor newKeys, Tensor newValues) {
        ensureOpen();
        ensureInitialized();
        checkLayer(layerIdx);
        Objects.requireNonNull(newKeys, "newKeys cannot be null");
        Objects.requireNonNull(newValues, "newValues cannot be null");
        requireRows("newKeys", newKeys);
        requireRows("newValues", newValues);

        int newSeqLen = newKeys.shape().dim(2);
        if (newValues.shape().dim(2) != newSeqLen) {
            throw new IllegalArgumentException("newKeys and newValues must have the same sequence length, got "
                + newSeqLen + " and " + newValues.shape().dim(2));
        }
        int currentLen = seqLengths[layerIdx];
        int totalLen = currentLen + newSeqLen;
        int capacity = keyCache[layerIdx].shape().dim(2);

        if (totalLen > capacity) {
            throw new IllegalStateException("Cache overflow: current=" + currentLen +
                ", new=" + newSeqLen + ", max=" + capacity);
        }
        if (newSeqLen == 0) {
            return;
        }

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor keys = newKeys.detach();
             Tensor values = newValues.detach();
             Tensor keySlots = keyCache[layerIdx].slice(2, currentLen, totalLen);
             Tensor valueSlots = valueCache[layerIdx].slice(2, currentLen, totalLen)) {
            copyInto(keys, keySlots);
            copyInto(values, valueSlots);
        }
        seqLengths[layerIdx] = totalLen;
    }

    /**
     * A new {@code [batchSize, numHeads, len, headDim]} tensor of layer {@code layerIdx}'s keys that does not require
     * grad; null when empty.
     */
    public Tensor getKeys(int layerIdx) {
        ensureOpen();
        ensureInitialized();
        checkLayer(layerIdx);
        return filled(keyCache[layerIdx], seqLengths[layerIdx]);
    }

    /**
     * A new {@code [batchSize, numHeads, len, headDim]} tensor of layer {@code layerIdx}'s values that does not require
     * grad; null when empty.
     */
    public Tensor getValues(int layerIdx) {
        ensureOpen();
        ensureInitialized();
        checkLayer(layerIdx);
        return filled(valueCache[layerIdx], seqLengths[layerIdx]);
    }

    public int getSeqLength(int layerIdx) {
        ensureOpen();
        checkLayer(layerIdx);
        return seqLengths[layerIdx];
    }

    /** Empties every layer; initialized caches are zero-filled in place. */
    public void clear() {
        ensureOpen();

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int layer = 0; layer < numLayers; layer++) {
                if (keyCache[layer] != null) {
                    TensorOps.zeroInPlace(keyCache[layer]);
                    TensorOps.zeroInPlace(valueCache[layer]);
                }
                seqLengths[layer] = 0;
            }
        }
    }

    /**
     * Changes the capacity to {@code newMaxSeqLen}. Each initialized layer gets new zero-filled caches holding its
     * first {@code min(len, newMaxSeqLen)} positions, and its length is truncated to that. Layers are replaced one at a
     * time; when an allocation or copy fails, the layers not yet replaced keep their caches and lengths, and
     * {@link #append} bounds every layer by the capacity of its own caches.
     */
    public void resize(int newMaxSeqLen) {
        ensureOpen();

        if (newMaxSeqLen <= 0) {
            throw new IllegalArgumentException("newMaxSeqLen must be positive: " + newMaxSeqLen);
        }

        if (backend != null) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Shape newShape = cacheShape(newMaxSeqLen);
                for (int layer = 0; layer < numLayers; layer++) {
                    resizeLayer(layer, newShape, Math.min(seqLengths[layer], newMaxSeqLen));
                }
            }
        }
        maxSeqLen = newMaxSeqLen;
    }

    /** Gives {@code layer} new zero-filled caches of {@code shape} holding its first {@code keep} positions. */
    private void resizeLayer(int layer, Shape shape, int keep) {
        Tensor newKeys = Tensor.zeros(backend, shape);
        Tensor newValues = null;
        try {
            newValues = Tensor.zeros(backend, shape);
            if (keep > 0) {
                copyPrefix(keyCache[layer], newKeys, keep);
                copyPrefix(valueCache[layer], newValues, keep);
            }
        } catch (RuntimeException e) {
            newKeys.close();
            if (newValues != null) {
                newValues.close();
            }
            throw e;
        }
        replace(layer, newKeys, newValues);
        seqLengths[layer] = keep;
    }

    private Shape cacheShape(int seqCapacity) {
        return Shape.of(batchSize, numHeads, seqCapacity, headDim);
    }

    /** Gives {@code layer} new zero-filled caches of {@code shape} and length 0. */
    private void resetLayer(int layer, Shape shape) {
        Tensor keys = Tensor.zeros(backend, shape);
        Tensor values;
        try {
            values = Tensor.zeros(backend, shape);
        } catch (RuntimeException e) {
            keys.close();
            throw e;
        }
        replace(layer, keys, values);
        seqLengths[layer] = 0;
    }

    private void replace(int layer, Tensor keys, Tensor values) {
        Tensor oldKeys = keyCache[layer];
        Tensor oldValues = valueCache[layer];
        keyCache[layer] = keys;
        valueCache[layer] = values;
        if (oldKeys != null) {
            oldKeys.close();
        }
        if (oldValues != null) {
            oldValues.close();
        }
    }

    /** The first {@code len} sequence positions of {@code cache} as a new tensor, or null when {@code len} is 0. */
    private static Tensor filled(Tensor cache, int len) {
        if (len == 0) {
            return null;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor view = cache.slice(2, 0, len)) {
            return FlashAttentionOp.copyOf(view);
        }
    }

    private void copyPrefix(Tensor from, Tensor to, int len) {
        try (Tensor src = from.slice(2, 0, len);
             Tensor dst = to.slice(2, 0, len)) {
            copyInto(src, dst);
        }
    }

    /**
     * Copies {@code src} into the elements of the cache view {@code dst} on the device, or through the CPU path. Called
     * only under {@link ComputationGraph#noGrad()}, with a destination that never requires grad.
     */
    private void copyInto(Tensor src, Tensor dst) {
        if (TrainGpu.forBackend(backend) != null) {
            GpuCoreOps.copyInto(src, dst);
        } else {
            GpuOptimizerOps.storeFloats(dst, src.toFloatArray());
        }
    }

    private void requireRows(String name, Tensor t) {
        Shape s = t.shape();
        if (s.rank() != 4) {
            throw new IllegalArgumentException(name + " must be 4D [batch, heads, seq, head_dim], got rank "
                + s.rank());
        }
        if (s.dim(0) != batchSize || s.dim(1) != numHeads || s.dim(3) != headDim) {
            throw new IllegalArgumentException(name + " shape " + s + " does not match the cache [" + batchSize
                + ", " + numHeads + ", seq, " + headDim + "]");
        }
        if (t.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException(name + " must be FLOAT32, got " + t.dtype());
        }
        if (t.backend() != backend) {
            throw new IllegalArgumentException(name + " lives on a different backend than the cache");
        }
    }

    private void checkLayer(int layerIdx) {
        if (layerIdx < 0 || layerIdx >= numLayers) {
            throw new IllegalArgumentException("layerIdx out of bounds: " + layerIdx);
        }
    }

    private void ensureInitialized() {
        if (backend == null) {
            throw new IllegalStateException("KVCache not initialized. Call initialize(backend) first.");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("KVCache has been closed");
        }
    }

    public int numLayers() {
        return numLayers;
    }

    public int numHeads() {
        return numHeads;
    }

    public int headDim() {
        return headDim;
    }

    public int maxSeqLen() {
        return maxSeqLen;
    }

    public int batchSize() {
        return batchSize;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;

            for (int i = 0; i < numLayers; i++) {
                replace(i, null, null);
                seqLengths[i] = 0;
            }
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "KVCache{closed=true}";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("KVCache{");
        sb.append("numLayers=").append(numLayers);
        sb.append(", numHeads=").append(numHeads);
        sb.append(", headDim=").append(headDim);
        sb.append(", maxSeqLen=").append(maxSeqLen);
        sb.append(", batchSize=").append(batchSize);
        sb.append(", seqLengths=").append(Arrays.toString(seqLengths));
        sb.append(", initialized=").append(backend != null);
        sb.append("}");
        return sb.toString();
    }
}
