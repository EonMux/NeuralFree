package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.Objects;

/**
 * Data movement on the train kernels: contiguous copies, strided copies into views, zero-filled tensors and dtype
 * casts. Results stay on the device; nothing is read back.
 *
 * <p>Every method returns {@code null} exactly when the train library is unavailable on the tensor's backend
 * ({@link TrainGpu#forBackend}) or the dtype is not supported as documented; that is the caller's CPU path. Once the
 * library is available, failures propagate. All results are bit-identical to the CPU path (data movement).</p>
 *
 * <p>These are raw operations: they never record a graph node and every result has {@code requiresGrad=false}
 * ({@link Tensor#contiguous()} and {@link Tensor#to(DType)} record on top of them). {@link #copyInto} is an in-place
 * write and refuses a destination that requires grad while grad mode is on.</p>
 */
public final class GpuCoreOps {

    private GpuCoreOps() {
    }

    /**
     * A new row-major contiguous copy of {@code t} (always a new tensor that owns fresh storage and does not require
     * grad), or null when the library is unavailable or the dtype is not FLOAT32, INT32, INT64 or FLOAT64. 8-byte
     * elements are copied as their two 32-bit words, so no 64-bit device support is needed.
     */
    public static Tensor contiguous(Tensor t) {
        Objects.requireNonNull(t, "tensor must not be null");
        int words = wordsPerElement(t.dtype());
        if (words == 0) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(t.backend());
        if (g == null) {
            return null;
        }
        StridedView in = StridedView.of(t).asWords(words);
        Tensor result = Tensor.wrap(t.shape(), t.dtype(), Strides.contiguous(t.shape()),
                g.allocate(t.numElements(), t.dtype()), t.backend(), false);
        try {
            if (t.numElements() > 0) {
                StridedView out = StridedView.of(result).asWords(words);
                g.exclusive(() -> {
                    copy(g, in, out);
                    return null;
                });
            }
        } catch (RuntimeException e) {
            result.close();
            throw e;
        }
        return result;
    }

    /**
     * Copies {@code src}, broadcast to the shape of {@code dstView}, into the elements {@code dstView} addresses
     * (any strides and offset). When both share a buffer, {@code src} is first copied to a temporary.
     *
     * @throws IllegalStateException    when {@code dstView} requires grad while grad mode is on (an in-place write
     *                                  into a tensor that requires grad; run it under
     *                                  {@link ComputationGraph#noGrad()}), or the train library is unavailable on
     *                                  the backend (callers check {@link TrainGpu#forBackend} first)
     * @throws IllegalArgumentException when the backends or dtypes differ, the dtype is not FLOAT32 or INT32, the
     *                                  shape does not broadcast, or {@code dstView} addresses an element twice
     */
    public static void copyInto(Tensor src, Tensor dstView) {
        Objects.requireNonNull(src, "source must not be null");
        Objects.requireNonNull(dstView, "destination must not be null");
        GpuTensorOps.requireWritable(dstView);
        if (src.backend() != dstView.backend()) {
            throw new IllegalArgumentException("source and destination live on different backends");
        }
        if (src.dtype() != dstView.dtype() || !isWordType(src.dtype())) {
            throw new IllegalArgumentException("copyInto needs equal FLOAT32 or INT32 dtypes, got " + src.dtype()
                    + " and " + dstView.dtype());
        }
        if (!BroadcastEngine.isBroadcastableTo(src.shape(), dstView.shape())) {
            throw new IllegalArgumentException("shape " + src.shape() + " does not broadcast to " + dstView.shape());
        }
        long[] dstStrides = dstView.strides().strides();
        for (int k = 0; k < dstStrides.length; k++) {
            if (dstStrides[k] == 0 && dstView.shape().dim(k) > 1) {
                throw new IllegalArgumentException("destination view " + dstView.shape() + " with strides "
                        + dstView.strides() + " addresses an element more than once");
            }
        }
        TrainGpu g = TrainGpu.forBackend(dstView.backend());
        if (g == null) {
            throw new IllegalStateException("train kernels unavailable on " + dstView.backend().backendType() + ": "
                    + TrainGpu.availability(dstView.backend()).reason());
        }
        if (dstView.numElements() == 0) {
            return;
        }
        if (src.storage().gpuBuffer() == dstView.storage().gpuBuffer()) {
            try (Tensor staged = contiguous(src)) {
                copyInto(staged, dstView);
            }
            return;
        }
        StridedView in = StridedView.broadcast(src, dstView.shape());
        StridedView out = StridedView.of(dstView);
        g.exclusive(() -> {
            copy(g, in, out);
            return null;
        });
    }

    /**
     * A zero-filled contiguous tensor (every byte 0: {@code +0.0f}, {@code 0}, {@code false}), filled on the device,
     * or null when the library is unavailable or the dtype is not FLOAT32, INT32 or BOOL.
     */
    public static Tensor zeros(ComputeBackend b, Shape s, DType d) {
        Objects.requireNonNull(b, "backend must not be null");
        Objects.requireNonNull(s, "shape must not be null");
        Objects.requireNonNull(d, "dtype must not be null");
        if (d != DType.FLOAT32 && d != DType.INT32 && d != DType.BOOL) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(b);
        if (g == null) {
            return null;
        }
        TensorStorage storage = g.allocate(s.numElements(), d);
        try {
            g.zero(storage.gpuBuffer());
        } catch (RuntimeException e) {
            storage.close();
            throw e;
        }
        return Tensor.wrap(s, d, Strides.contiguous(s), storage, b, false);
    }

    /**
     * {@code t} converted to {@code target} on the device for the pairs INT32 to FLOAT32, FLOAT32 to INT32 (Java
     * {@code (int)}: NaN to 0, saturating), BOOL to FLOAT32, FLOAT32 to BOOL ({@code x != 0}, NaN to true), FLOAT32
     * to and from FLOAT16 and BFLOAT16 ({@link HalfFloats}), and INT64 to INT32 (low word, Java {@code (int)}).
     * Returns null for any other pair, when the library is unavailable, and for a non-dense sub-word (BOOL, FLOAT16,
     * BFLOAT16) source; a non-dense INT64 source is first copied contiguously on the device. The result is a new
     * contiguous tensor that does not require grad.
     */
    public static Tensor cast(Tensor t, DType target) {
        Objects.requireNonNull(t, "tensor must not be null");
        Objects.requireNonNull(target, "target dtype must not be null");
        String kernel = castKernel(t.dtype(), target);
        if (kernel == null) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(t.backend());
        if (g == null) {
            return null;
        }
        StridedView in = StridedView.of(t);
        Tensor staged = null;
        if (!in.contiguous()) {
            staged = contiguous(t);
            if (staged == null) {
                return null;
            }
            in = StridedView.of(staged);
        }
        try {
            int n = in.numElements();
            Tensor result = Tensor.wrap(t.shape(), target, Strides.contiguous(t.shape()), g.allocate(n, target),
                    t.backend(), false);
            try {
                g.launch1d(kernel, castItems(target, n), n, in.offset(), 0, in.buffer(), result.storage().gpuBuffer());
            } catch (RuntimeException e) {
                result.close();
                throw e;
            }
            return result;
        } finally {
            if (staged != null) {
                staged.close();
            }
        }
    }

    /** The cast kernel of a dtype pair, or null when the pair has no device cast. */
    static String castKernel(DType from, DType to) {
        if (from == DType.INT32 && to == DType.FLOAT32) {
            return Kernels.CAST_I32_F32;
        }
        if (from == DType.FLOAT32) {
            switch (to) {
                case INT32:
                    return Kernels.CAST_F32_I32;
                case BOOL:
                    return Kernels.CAST_F32_BOOL;
                case FLOAT16:
                    return Kernels.CAST_F32_F16;
                case BFLOAT16:
                    return Kernels.CAST_F32_BF16;
                default:
                    return null;
            }
        }
        if (to == DType.FLOAT32) {
            switch (from) {
                case BOOL:
                    return Kernels.CAST_BOOL_F32;
                case FLOAT16:
                    return Kernels.CAST_F16_F32;
                case BFLOAT16:
                    return Kernels.CAST_BF16_F32;
                default:
                    return null;
            }
        }
        return from == DType.INT64 && to == DType.INT32 ? Kernels.CAST_I64_I32 : null;
    }

    /** Work-items of a cast launch: one per output word for sub-word outputs, else one per element. */
    private static long castItems(DType target, int n) {
        switch (target) {
            case BOOL:
                return (n + 3L) / 4;
            case FLOAT16:
            case BFLOAT16:
                return (n + 1L) / 2;
            default:
                return n;
        }
    }

    /** Copies view {@code in} into view {@code out} of the same dims (32-bit words); holds the backend lock. */
    private static void copy(TrainGpu g, StridedView in, StridedView out) {
        StridedView[] merged = StridedView.coalesce(in, out);
        StridedView src = merged[0];
        StridedView dst = merged[1];
        int n = src.numElements();
        if (src.contiguous() && dst.contiguous()) {
            g.launch1d(Kernels.COPY_F32, n, n, src.offset(), dst.offset(), src.buffer(), dst.buffer());
            return;
        }
        StridedView[] srcParts = src.splitLeading();
        StridedView[] dstParts = dst.splitLeading();
        for (int i = 0; i < srcParts.length; i++) {
            StridedView s = srcParts[i];
            StridedView d = dstParts[i];
            Object[] args = new Object[24];
            args[0] = s.numElements();
            args[1] = s.rank();
            System.arraycopy(s.dimArgs(), 0, args, 2, StridedView.MAX_RANK);
            System.arraycopy(s.strideArgs(), 0, args, 8, StridedView.MAX_RANK);
            System.arraycopy(d.strideArgs(), 0, args, 14, StridedView.MAX_RANK);
            args[20] = s.offset();
            args[21] = d.offset();
            args[22] = s.buffer();
            args[23] = d.buffer();
            g.launch1d(Kernels.COPY_STRIDED_F32, s.numElements(), args);
        }
    }

    private static boolean isWordType(DType d) {
        return d == DType.FLOAT32 || d == DType.INT32;
    }

    /** 32-bit words per element of a dtype {@link #contiguous} moves: 1 or 2, or 0 for a sub-word dtype. */
    private static int wordsPerElement(DType d) {
        switch (d) {
            case FLOAT32:
            case INT32:
                return 1;
            case INT64:
            case FLOAT64:
                return 2;
            default:
                return 0;
        }
    }
}
