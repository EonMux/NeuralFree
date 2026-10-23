package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.tensor.BroadcastEngine;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;

import java.util.Arrays;
import java.util.Objects;

/**
 * A tensor view in the form strided train kernels take: a buffer, an {@code int} element offset from the buffer
 * start ({@code storage.offsetBytes()/dtype.byteSize() + strides.offset()}), and {@code int} dims and strides in
 * elements of the tensor's dtype. Immutable.
 *
 * <p>Kernels take at most {@link #MAX_RANK} dims: {@link #coalesce} merges adjacent dims that are contiguous in
 * every operand, and {@link #splitLeading} turns a view still above {@link #MAX_RANK} into one view per index of
 * the leading dims.</p>
 */
public final class StridedView {

    /** Dim and stride slots of a strided train kernel. */
    public static final int MAX_RANK = 6;

    private final GpuBuffer buffer;
    private final int offset;
    private final int[] dims;
    private final int[] strides;

    private StridedView(GpuBuffer buffer, int offset, int[] dims, int[] strides) {
        this.buffer = buffer;
        this.offset = offset;
        this.dims = dims;
        this.strides = strides;
    }

    /**
     * The view of a tensor.
     *
     * @throws IllegalArgumentException when the tensor has more than {@code 2^31-1} elements, its storage ends
     *                                  beyond element {@code 2^31-1}, its storage offset is not a whole number of
     *                                  elements, or its highest reachable element lies outside its storage
     */
    public static StridedView of(Tensor t) {
        Objects.requireNonNull(t, "tensor must not be null");
        return create(t, t.shape().dims(), t.strides().strides(), t.strides().offset());
    }

    /**
     * The view of a tensor broadcast to {@code target} ({@link BroadcastEngine} rules): stride 0 on prepended dims
     * and on dims of size 1 that broadcast.
     *
     * @throws IllegalArgumentException when the shape does not broadcast to {@code target}, or as {@link #of}
     */
    public static StridedView broadcast(Tensor t, Shape target) {
        Objects.requireNonNull(t, "tensor must not be null");
        Objects.requireNonNull(target, "target shape must not be null");
        long[] s = BroadcastEngine.broadcastStrides(t.shape(), target, t.strides());
        return create(t, target.dims(), s, t.strides().offset());
    }

    /**
     * Views with identical dims, with adjacent dims merged where every view is contiguous across them and dims of
     * size 1 dropped. The result has rank at least 1; an empty view becomes {@code [0]}.
     *
     * @throws IllegalArgumentException when the views do not have identical dims
     */
    public static StridedView[] coalesce(StridedView... views) {
        Objects.requireNonNull(views, "views must not be null");
        if (views.length == 0) {
            return new StridedView[0];
        }
        int[] d = views[0].dims;
        for (StridedView v : views) {
            Objects.requireNonNull(v, "view must not be null");
            if (!Arrays.equals(d, v.dims)) {
                throw new IllegalArgumentException("coalesced views need identical dims: " + Arrays.toString(d)
                        + " vs " + Arrays.toString(v.dims));
            }
        }
        int n = views.length;
        int[] outDims = new int[d.length];
        int[][] outStrides = new int[n][d.length];
        int rank = 0;
        for (int k = 0; k < d.length; k++) {
            if (d[k] == 0) {
                StridedView[] empty = new StridedView[n];
                for (int i = 0; i < n; i++) {
                    empty[i] = new StridedView(views[i].buffer, views[i].offset, new int[]{0}, new int[]{1});
                }
                return empty;
            }
            if (d[k] == 1) {
                continue;
            }
            boolean merge = rank > 0;
            for (int i = 0; i < n && merge; i++) {
                merge = (long) outStrides[i][rank - 1] == (long) views[i].strides[k] * d[k];
            }
            if (merge) {
                outDims[rank - 1] *= d[k];
                for (int i = 0; i < n; i++) {
                    outStrides[i][rank - 1] = views[i].strides[k];
                }
            } else {
                outDims[rank] = d[k];
                for (int i = 0; i < n; i++) {
                    outStrides[i][rank] = views[i].strides[k];
                }
                rank++;
            }
        }
        int finalRank = Math.max(rank, 1);
        StridedView[] out = new StridedView[n];
        for (int i = 0; i < n; i++) {
            int[] od = rank == 0 ? new int[]{1} : Arrays.copyOf(outDims, finalRank);
            int[] os = rank == 0 ? new int[]{0} : Arrays.copyOf(outStrides[i], finalRank);
            out[i] = new StridedView(views[i].buffer, views[i].offset, od, os);
        }
        return out;
    }

    /**
     * This view as views of rank {@link #MAX_RANK}, one per index of the leading {@code rank - MAX_RANK} dims in
     * row-major order, each offset to that index; {@code {this}} when the rank is at most {@link #MAX_RANK}. Views
     * with identical dims split into aligned arrays.
     */
    public StridedView[] splitLeading() {
        if (dims.length <= MAX_RANK) {
            return new StridedView[]{this};
        }
        int lead = dims.length - MAX_RANK;
        long count = 1;
        for (int k = 0; k < lead; k++) {
            count *= dims[k];
        }
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("too many leading indices to split: " + count);
        }
        int[] innerDims = Arrays.copyOfRange(dims, lead, dims.length);
        int[] innerStrides = Arrays.copyOfRange(strides, lead, strides.length);
        StridedView[] out = new StridedView[(int) count];
        int[] idx = new int[lead];
        for (int c = 0; c < count; c++) {
            long off = offset;
            for (int k = 0; k < lead; k++) {
                off += (long) idx[k] * strides[k];
            }
            out[c] = new StridedView(buffer, (int) off, innerDims, innerStrides);
            for (int k = lead - 1; k >= 0; k--) {
                if (++idx[k] < dims[k]) {
                    break;
                }
                idx[k] = 0;
            }
        }
        return out;
    }

    /**
     * This view of elements made of {@code wordsPerElement} 32-bit words (2 for an 8-byte dtype) as a view of those
     * words: the offset and strides scaled by {@code wordsPerElement}, plus a trailing dim of {@code wordsPerElement}
     * words with stride 1. A word copy over the result moves every element bit for bit. {@code this} for 1.
     *
     * @throws IllegalArgumentException when {@code wordsPerElement} is below 1, or the words the view reaches or
     *                                  addresses exceed the 32-bit kernel limit {@code 2^31-1}
     */
    public StridedView asWords(int wordsPerElement) {
        if (wordsPerElement < 1) {
            throw new IllegalArgumentException("words per element must be at least 1, got " + wordsPerElement);
        }
        if (wordsPerElement == 1) {
            return this;
        }
        long highest = offset;
        for (int k = 0; k < dims.length; k++) {
            if (dims[k] > 1) {
                highest += (long) (dims[k] - 1) * strides[k];
            }
        }
        long lastWord = highest * wordsPerElement + wordsPerElement - 1;
        long words = (long) numElements() * wordsPerElement;
        if (lastWord > Integer.MAX_VALUE || words > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("view " + this + " of " + wordsPerElement
                    + "-word elements reaches word " + lastWord + " and addresses " + words
                    + " words, beyond the 32-bit kernel limit " + Integer.MAX_VALUE);
        }
        int[] wordDims = Arrays.copyOf(dims, dims.length + 1);
        int[] wordStrides = new int[strides.length + 1];
        for (int k = 0; k < strides.length; k++) {
            wordStrides[k] = dims[k] > 1 ? strides[k] * wordsPerElement : 0;
        }
        wordDims[dims.length] = wordsPerElement;
        wordStrides[strides.length] = 1;
        return new StridedView(buffer, offset * wordsPerElement, wordDims, wordStrides);
    }

    public GpuBuffer buffer() {
        return buffer;
    }

    /** Element offset of the view's first element from the start of the buffer. */
    public int offset() {
        return offset;
    }

    public int rank() {
        return dims.length;
    }

    /** Dim {@code k}. */
    public int dim(int k) {
        return dims[k];
    }

    /** Element stride of dim {@code k}. */
    public int stride(int k) {
        return strides[k];
    }

    /** Number of elements the view addresses. */
    public int numElements() {
        long n = 1;
        for (int d : dims) {
            n *= d;
        }
        return (int) n;
    }

    /** True when the elements are dense in row-major order starting at {@link #offset()}. */
    public boolean contiguous() {
        long expected = 1;
        for (int k = dims.length - 1; k >= 0; k--) {
            if (dims[k] == 0) {
                return true;
            }
            if (dims[k] != 1 && strides[k] != expected) {
                return false;
            }
            expected *= dims[k];
        }
        return true;
    }

    /**
     * The {@link #MAX_RANK} dim arguments of a strided kernel; slots past the rank carry 1.
     *
     * @throws IllegalStateException when the rank is above {@link #MAX_RANK} ({@link #splitLeading} first)
     */
    public Object[] dimArgs() {
        return slots(dims, 1);
    }

    /**
     * The {@link #MAX_RANK} stride arguments of a strided kernel; slots past the rank carry 0.
     *
     * @throws IllegalStateException when the rank is above {@link #MAX_RANK} ({@link #splitLeading} first)
     */
    public Object[] strideArgs() {
        return slots(strides, 0);
    }

    private Object[] slots(int[] values, int fill) {
        if (values.length > MAX_RANK) {
            throw new IllegalStateException("rank " + values.length + " exceeds " + MAX_RANK
                    + "; split the view with splitLeading() first");
        }
        Object[] out = new Object[MAX_RANK];
        for (int k = 0; k < MAX_RANK; k++) {
            out[k] = k < values.length ? values[k] : fill;
        }
        return out;
    }

    private static StridedView create(Tensor t, int[] dims, long[] strides, long viewOffset) {
        TensorStorage storage = t.storage();
        int elementBytes = t.dtype().byteSize();
        if (storage.offsetBytes() % elementBytes != 0) {
            throw new IllegalArgumentException("storage offset of " + storage.offsetBytes()
                    + " bytes is not a whole number of " + t.dtype() + " elements");
        }
        long base = storage.offsetBytes() / elementBytes + viewOffset;
        long end = (storage.offsetBytes() + storage.lengthBytes()) / elementBytes;
        if (end > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("storage ends at element " + end + ", beyond the 32-bit kernel limit "
                    + Integer.MAX_VALUE);
        }
        long count = 1;
        long highest = base;
        int[] s = new int[strides.length];
        for (int k = 0; k < dims.length; k++) {
            count *= dims[k];
            if (dims[k] > 1) {
                highest += (long) (dims[k] - 1) * strides[k];
                s[k] = (int) Math.min(strides[k], Integer.MAX_VALUE);
            }
        }
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("view of " + count + " elements exceeds the 32-bit kernel limit "
                    + Integer.MAX_VALUE);
        }
        if (count > 0 ? highest >= end : base > end) {
            throw new IllegalArgumentException("view " + Arrays.toString(dims) + " with strides "
                    + Arrays.toString(strides) + " at element " + base + " reaches element " + highest
                    + " outside its storage of " + end + " elements");
        }
        return new StridedView(storage.gpuBuffer(), (int) base, dims.clone(), s);
    }

    @Override
    public String toString() {
        return "StridedView{offset=" + offset + ", dims=" + Arrays.toString(dims) + ", strides="
                + Arrays.toString(strides) + "}";
    }
}
