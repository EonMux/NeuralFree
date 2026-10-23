package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Matrix products on the train library's strided GEMM ({@code TR_Gemm_F32}) and GEMV ({@code TR_Gemv_F32}) kernels.
 * Results stay on the device; nothing is read back.
 *
 * <p><b>Semantics.</b> {@link #matmul} follows the rank rules of {@code TensorOps.matmul}: when both operands have
 * rank at most 2, a rank-1 {@code a} of length K times {@code [K, N]} gives {@code [N]}, {@code [M, K]} times a rank-1
 * {@code b} gives {@code [M]}, two rank-1 operands give {@code [1]}, and two matrices give {@code [M, N]}. When either
 * operand has rank above 2, the leading dims of both broadcast (a rank-1 or rank-2 operand has none) and the result is
 * {@code batch + [M, N]}, with {@code M = 1} for a rank-1 {@code a} and {@code N = 1} for a rank-1 {@code b}.
 * {@link #bmm} follows {@code TensorOps.bmm}. Every operand may be any strided view (transposes, slices, expands): the
 * kernels address it through its element offset and strides.</p>
 *
 * <p><b>Accumulation and tolerance.</b> Each output is the sum over k of its K products: TR_Gemm_F32 adds them in
 * ascending k with {@code fma}, TR_Gemv_F32 adds k-strided partials over a fixed 256-wide tree. The CPU reference
 * adds separately rounded products in ascending k, so results agree within the GEMM tolerance class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}:
 * {@code |gpu - ref64| <= 1e-5 * sqrt(K) * sum_k |a_k * b_k| + K * FLT_MIN} against a double-precision reference. The
 * kernel choice depends only on the shapes and strides.</p>
 *
 * <p><b>Dispatch.</b> Broadcast batch dims are merged into one batch stride where the three operands allow it, and a
 * batch that shares {@code b} and walks {@code a} and the output as consecutive row blocks is folded into the rows of
 * one product; any remaining leading batch dims get one launch per index. A product that is then a single
 * matrix-vector product ({@code N == 1}, or {@code M == 1} without a batch) runs on TR_Gemv_F32, every other product
 * on TR_Gemm_F32. A TR_Gemm_F32 launch covers at most {@value #MAX_GROUPS} batch entries (larger batches are
 * chunked) and at most {@value #MAX_GROUPS} tiles of {@value TrainOps#GEMM_TILE} rows or columns, the group-count
 * limit every API guarantees; a product needing more tiles throws {@link IllegalArgumentException}.</p>
 *
 * <p><b>Contract.</b> A method returns {@code null} (or {@code false}) exactly when the train library is unavailable on
 * the operands' backend or an operand is not FLOAT32; that is the caller's CPU path. Invalid shapes throw
 * {@link IllegalArgumentException} with the CPU path's message. Once the library is available, kernel failures
 * propagate. Results never require grad, whatever the operands' flags; only the recording facades mark a result as
 * requiring grad.</p>
 */
public final class GpuMatmul {

    /** Largest work-group count per dimension that OpenCL, CUDA (grid y and z) and Vulkan all accept. */
    public static final int MAX_GROUPS = 65535;


    private GpuMatmul() {
    }

    /**
     * Whether {@link #matmul} and {@link #bmm} run on the device for these operands: both live on one backend where
     * the train library is available, and both are FLOAT32.
     */
    public static boolean shouldUseGpu(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "Tensor a cannot be null");
        Objects.requireNonNull(b, "Tensor b cannot be null");
        return a.backend() == b.backend() && context(a, b) != null;
    }

    /**
     * {@code a @ b} with the rank and broadcast rules of {@code TensorOps.matmul}, as a new contiguous tensor; null
     * when the train library is unavailable or an operand is not FLOAT32.
     *
     * @throws IllegalArgumentException when the backends differ, the shapes do not multiply, or the product exceeds
     *                                  the 32-bit kernel ABI or the work-group limits
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "Tensor a cannot be null");
        Objects.requireNonNull(b, "Tensor b cannot be null");
        requireSameBackend(a, b);
        Product p = Product.matmul(a.shape(), b.shape());
        TrainGpu g = context(a, b);
        return g == null ? null : compute(g, p, a, b);
    }

    /**
     * {@code a @ b} for two rank-3 operands with equal batch, with the rules of {@code TensorOps.bmm}, as a new
     * contiguous {@code [batch, M, N]} tensor; null when the train library is unavailable or an operand is not
     * FLOAT32.
     *
     * @throws IllegalArgumentException when the backends differ, an operand is not rank 3, the batches or inner
     *                                  dimensions differ, or the product exceeds the kernel limits
     */
    public static Tensor bmm(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "Tensor a cannot be null");
        Objects.requireNonNull(b, "Tensor b cannot be null");
        requireSameBackend(a, b);
        Product p = Product.bmm(a.shape(), b.shape());
        TrainGpu g = context(a, b);
        return g == null ? null : compute(g, p, a, b);
    }

    /**
     * In place {@code c = alpha * (a @ b) + beta * c}, where {@code a @ b} follows {@link #matmul} and {@code c} is
     * any view of the product's shape. Each element is {@code alpha * s} when {@code beta == 0} ({@code c} is not
     * read, so NaN in {@code c} is ignored), otherwise {@code alpha * s + beta * c}, with {@code s} the sum of the K
     * products and each product and the final sum rounded separately. When {@code c} shares a buffer with {@code a}
     * or {@code b}, or its column stride is not 1, the product is computed into a temporary and copied into
     * {@code c}. Returns false, leaving {@code c} unchanged, when the train library is unavailable or a tensor is not
     * FLOAT32.
     *
     * @throws IllegalArgumentException when the backends differ, the shapes do not multiply, {@code c} does not have
     *                                  the product's shape or addresses an element twice, or the product exceeds
     *                                  the kernel limits
     * @throws IllegalStateException    when {@code c} requires grad while grad mode is enabled (on every backend,
     *                                  before anything is written)
     */
    public static boolean gemm(Tensor a, Tensor b, float alpha, float beta, Tensor c) {
        Objects.requireNonNull(a, "Tensor a cannot be null");
        Objects.requireNonNull(b, "Tensor b cannot be null");
        Objects.requireNonNull(c, "Tensor c cannot be null");
        GpuTensorOps.requireWritable(c);
        requireSameBackend(a, b);
        requireSameBackend(a, c);
        Product p = Product.matmul(a.shape(), b.shape());
        if (!c.shape().equals(p.outShape)) {
            throw new IllegalArgumentException("gemm output shape " + c.shape() + " does not match the product shape "
                    + p.outShape);
        }
        requireDistinctElements(c);
        TrainGpu g = c.dtype() == DType.FLOAT32 ? context(a, b) : null;
        if (g == null) {
            return false;
        }
        Operand ca = Operand.of(a, p, Role.A);
        Operand cb = Operand.of(b, p, Role.B);
        Operand cc = Operand.of(c, p, Role.C);
        boolean direct = cc.buffer != ca.buffer && cc.buffer != cb.buffer && (p.n <= 1 || cc.colStride == 1);
        if (direct) {
            run(g, p, ca, cb, cc, alpha, beta);
            return true;
        }
        Tensor staged = beta != 0.0f ? GpuCoreOps.contiguous(c) : newOutput(g, p.outShape);
        if (staged == null) {
            throw new IllegalStateException("train kernels became unavailable on " + c.backend().backendType()
                    + " while staging gemm output");
        }
        try {
            run(g, p, ca, cb, Operand.of(staged, p, Role.C), alpha, beta);
            GpuCoreOps.copyInto(staged, c);
        } finally {
            staged.close();
        }
        return true;
    }

    /** Releases the train context of every backend ({@link TrainGpu#releaseAll}). */
    public static void clearCache() {
        TrainGpu.releaseAll();
    }

    /** Releases the train context of one backend ({@link TrainGpu#release}). */
    public static void clearCache(ComputeBackend backend) {
        TrainGpu.release(backend);
    }

    // ------------------------------------------------------------------ dispatch

    /** The train context of the operands' backend, or null when it is unavailable or an operand is not FLOAT32. */
    private static TrainGpu context(Tensor a, Tensor b) {
        if (a.dtype() != DType.FLOAT32 || b.dtype() != DType.FLOAT32) {
            return null;
        }
        return TrainGpu.forBackend(a.backend());
    }

    private static Tensor compute(TrainGpu g, Product p, Tensor a, Tensor b) {
        Operand ca = Operand.of(a, p, Role.A);
        Operand cb = Operand.of(b, p, Role.B);
        Tensor c = newOutput(g, p.outShape);
        try {
            run(g, p, ca, cb, Operand.of(c, p, Role.C), 1.0f, 0.0f);
        } catch (RuntimeException e) {
            c.close();
            throw e;
        }
        return c;
    }

    /** A new contiguous FLOAT32 tensor with fresh storage that does not require grad. */
    private static Tensor newOutput(TrainGpu g, Shape shape) {
        return Tensor.wrap(shape, DType.FLOAT32, Strides.contiguous(shape), g.allocate(shape.numElements(),
                DType.FLOAT32), g.backend(), false);
    }

    /**
     * Enqueues {@code c = alpha * (a @ b) + beta * c} for every batch entry. The batch dims are merged jointly; the
     * innermost merged dim becomes the kernel batch (or is folded into the rows) and the others are iterated with
     * adjusted offsets. The geometry is planned and checked before anything is enqueued.
     */
    private static void run(TrainGpu g, Product p, Operand a, Operand b, Operand c, float alpha, float beta) {
        if (p.m == 0 || p.n == 0) {
            return;
        }
        for (int d : p.batchDims) {
            if (d == 0) {
                return;
            }
        }
        List<long[]> groups = mergeBatchDims(p.batchDims, a, b, c);
        long[] inner = groups.isEmpty() ? new long[]{1, 0, 0, 0} : groups.remove(groups.size() - 1);
        Step step = Step.plan(p, a, b, c, inner);
        g.exclusive(() -> {
            int[] idx = new int[groups.size()];
            while (true) {
                long aOff = a.offset;
                long bOff = b.offset;
                long cOff = c.offset;
                for (int i = 0; i < idx.length; i++) {
                    long[] grp = groups.get(i);
                    aOff += idx[i] * grp[1];
                    bOff += idx[i] * grp[2];
                    cOff += idx[i] * grp[3];
                }
                step.enqueue(g, a.buffer, (int) aOff, b.buffer, (int) bOff, c.buffer, (int) cOff, alpha, beta);
                int i = idx.length - 1;
                while (i >= 0 && ++idx[i] == groups.get(i)[0]) {
                    idx[i--] = 0;
                }
                if (i < 0) {
                    return null;
                }
            }
        });
    }

    /**
     * The broadcast batch dims as {@code {dim, aStride, bStride, cStride}} groups in order: dims of size 1 dropped,
     * and adjacent dims merged where every operand steps through them as one dim.
     */
    private static List<long[]> mergeBatchDims(int[] dims, Operand a, Operand b, Operand c) {
        List<long[]> groups = new ArrayList<>();
        for (int i = 0; i < dims.length; i++) {
            long d = dims[i];
            if (d == 1) {
                continue;
            }
            long sa = a.batchStrides[i];
            long sb = b.batchStrides[i];
            long sc = c.batchStrides[i];
            int lastIndex = groups.size() - 1;
            long[] last = lastIndex < 0 ? null : groups.get(lastIndex);
            if (last != null && last[1] == sa * d && last[2] == sb * d && last[3] == sc * d) {
                groups.set(lastIndex, new long[]{last[0] * d, sa, sb, sc});
            } else {
                groups.add(new long[]{d, sa, sb, sc});
            }
        }
        return groups;
    }

    private static void requireSameBackend(Tensor a, Tensor b) {
        if (a.backend() != b.backend()) {
            throw new IllegalArgumentException("Tensors must be on the same backend");
        }
    }

    private static void requireDistinctElements(Tensor c) {
        long[] strides = c.strides().strides();
        for (int k = 0; k < strides.length; k++) {
            if (strides[k] == 0 && c.shape().dim(k) > 1) {
                throw new IllegalArgumentException("gemm output view " + c.shape() + " with strides " + c.strides()
                        + " addresses an element more than once");
            }
        }
    }

    private static int tiles(long extent) {
        return (int) ((extent + TrainOps.GEMM_TILE - 1) / TrainOps.GEMM_TILE);
    }

    // ------------------------------------------------------------------ geometry

    /** The operand roles of a product: A is [M, K], B is [K, N], C is [M, N]. */
    private enum Role { A, B, C }

    /** Shape geometry of one product under the CPU rank rules. Immutable. */
    private static final class Product {
        final boolean aVector;
        final int m;
        final int n;
        final int k;
        /** Broadcast batch dims; empty when both operands have rank at most 2. */
        final int[] batchDims;
        final Shape outShape;

        private Product(boolean aVector, int m, int n, int k, int[] batchDims, Shape outShape) {
            this.aVector = aVector;
            this.m = m;
            this.n = n;
            this.k = k;
            this.batchDims = batchDims;
            this.outShape = outShape;
        }

        static Product matmul(Shape a, Shape b) {
            if (a.rank() < 1 || b.rank() < 1) {
                throw new IllegalArgumentException("Both tensors must have at least 1 dimension for matmul");
            }
            int kA = a.dim(a.rank() - 1);
            int kB = b.rank() > 1 ? b.dim(b.rank() - 2) : b.dim(0);
            if (kA != kB) {
                throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication: "
                        + "A has " + kA + " columns, B has " + kB + " rows");
            }
            boolean aVector = a.rank() == 1;
            boolean bVector = b.rank() == 1;
            int m = aVector ? 1 : a.dim(a.rank() - 2);
            int n = bVector ? 1 : b.dim(b.rank() - 1);
            if (a.rank() <= 2 && b.rank() <= 2) {
                Shape out = aVector ? Shape.vector(n) : bVector ? Shape.vector(m) : Shape.matrix(m, n);
                return new Product(aVector, m, n, kA, new int[0], out);
            }
            int[] batchA = leadingDims(a);
            int[] batchB = leadingDims(b);
            int rank = Math.max(batchA.length, batchB.length);
            int[] batch = new int[rank];
            for (int i = 0; i < rank; i++) {
                int dA = i < rank - batchA.length ? 1 : batchA[i - (rank - batchA.length)];
                int dB = i < rank - batchB.length ? 1 : batchB[i - (rank - batchB.length)];
                if (dA != dB && dA != 1 && dB != 1) {
                    throw new IllegalArgumentException("Batch dimensions cannot be broadcast: "
                            + Arrays.toString(batchA) + " vs " + Arrays.toString(batchB));
                }
                batch[i] = Math.max(dA, dB);
            }
            int[] out = Arrays.copyOf(batch, rank + 2);
            out[rank] = m;
            out[rank + 1] = n;
            return new Product(aVector, m, n, kA, batch, Shape.of(out));
        }

        static Product bmm(Shape a, Shape b) {
            if (a.rank() != 3 || b.rank() != 3) {
                throw new IllegalArgumentException("Both tensors must be 3D for bmm: got ranks " + a.rank() + " and "
                        + b.rank());
            }
            int batch = a.dim(0);
            if (batch != b.dim(0)) {
                throw new IllegalArgumentException("Batch dimensions must match: " + batch + " vs " + b.dim(0));
            }
            int k = a.dim(2);
            if (k != b.dim(1)) {
                throw new IllegalArgumentException("Inner dimensions must match: " + k + " vs " + b.dim(1));
            }
            int m = a.dim(1);
            int n = b.dim(2);
            return new Product(false, m, n, k, new int[]{batch}, Shape.of(batch, m, n));
        }

        private static int[] leadingDims(Shape s) {
            return s.rank() <= 2 ? new int[0] : Arrays.copyOf(s.dims(), s.rank() - 2);
        }
    }

    /**
     * One operand as a batched matrix: buffer, element offset, the strides of its matrix rows and columns, and its
     * stride along each broadcast batch dim (0 where it broadcasts). A row or column of extent 1 has stride 0.
     */
    private static final class Operand {
        final GpuBuffer buffer;
        final int offset;
        final int rowStride;
        final int colStride;
        final int[] batchStrides;

        private Operand(GpuBuffer buffer, int offset, int rowStride, int colStride, int[] batchStrides) {
            this.buffer = buffer;
            this.offset = offset;
            this.rowStride = rowStride;
            this.colStride = colStride;
            this.batchStrides = batchStrides;
        }

        static Operand of(Tensor t, Product p, Role role) {
            StridedView v = StridedView.of(t);
            int r = v.rank();
            int rowStride;
            int colStride;
            if (r >= 2) {
                rowStride = v.stride(r - 2);
                colStride = v.stride(r - 1);
            } else if (role == Role.A || (role == Role.C && p.aVector)) {
                rowStride = 0;
                colStride = v.stride(0);
            } else {
                rowStride = v.stride(0);
                colStride = 0;
            }
            int nb = p.batchDims.length;
            int own = r > 2 ? r - 2 : 0;
            int[] batchStrides = new int[nb];
            for (int i = 0; i < nb; i++) {
                int j = i - (nb - own);
                batchStrides[i] = j >= 0 ? v.stride(j) : 0;
            }
            return new Operand(v.buffer(), v.offset(), rowStride, colStride, batchStrides);
        }
    }

    /** The kernel of one launch geometry. */
    private enum Kind {
        /** TR_Gemv_F32 over the rows of A, with x the single column of B. */
        GEMV_A,
        /** TR_Gemv_F32 over the columns of B (the rows of B transposed), with x the single row of A. */
        GEMV_BT,
        /** TR_Gemm_F32, batched. */
        GEMM
    }

    /**
     * The geometry every launch of one product shares; launches differ only in their base offsets. Immutable.
     */
    private static final class Step {
        final Kind kind;
        final int rows;
        final int n;
        final int k;
        final int batch;
        final int aRs;
        final int aCs;
        final int aBs;
        final int bRs;
        final int bCs;
        final int bBs;
        final int cRs;
        final int cBs;

        private Step(Kind kind, int rows, int n, int k, int batch, long aRs, int aCs, long aBs, int bRs, int bCs,
                     long bBs, long cRs, long cBs) {
            this.kind = kind;
            this.rows = rows;
            this.n = n;
            this.k = k;
            this.batch = batch;
            this.aRs = (int) aRs;
            this.aCs = aCs;
            this.aBs = (int) aBs;
            this.bRs = bRs;
            this.bCs = bCs;
            this.bBs = (int) bBs;
            this.cRs = (int) cRs;
            this.cBs = (int) cBs;
        }

        /**
         * Chooses the kernel for a batch of {@code inner[0]} products with batch strides {@code inner[1..3]}. A batch
         * that shares B ({@code bStride == 0}) and walks A and C as consecutive row blocks folds into one product of
         * {@code batch * M} rows. C's column stride is 1 unless {@code N == 1}.
         *
         * @throws IllegalArgumentException when TR_Gemm_F32 would need more than {@value GpuMatmul#MAX_GROUPS} row
         *                                  or column tiles
         */
        static Step plan(Product p, Operand a, Operand b, Operand c, long[] inner) {
            long nb = inner[0];
            long sa = inner[1];
            long sb = inner[2];
            long sc = inner[3];
            int m = p.m;
            long aRs = m == 1 ? sa : a.rowStride;
            long cRs = m == 1 ? sc : c.rowStride;
            boolean foldable = nb > 1 && sb == 0 && sa == m * aRs && sc == m * cRs;
            long foldedRows = nb * m;
            if (p.n == 1) {
                if (nb == 1 && (m == 1 || c.rowStride == 1)) {
                    return new Step(Kind.GEMV_A, m, 1, p.k, 1, a.rowStride, a.colStride, 0, b.rowStride, 0, 0, 1, 0);
                }
                if (foldable && cRs == 1) {
                    return new Step(Kind.GEMV_A, (int) foldedRows, 1, p.k, 1, aRs, a.colStride, 0, b.rowStride, 0, 0,
                            1, 0);
                }
            } else if (m == 1 && nb == 1) {
                return new Step(Kind.GEMV_BT, 1, p.n, p.k, 1, 0, a.colStride, 0, b.rowStride, b.colStride, 0, 0, 0);
            }
            requireTiles("columns", p.n);
            if (foldable && tiles(foldedRows) <= MAX_GROUPS) {
                return new Step(Kind.GEMM, (int) foldedRows, p.n, p.k, 1, aRs, a.colStride, 0, b.rowStride,
                        b.colStride, 0, cRs, 0);
            }
            requireTiles("rows", m);
            return new Step(Kind.GEMM, m, p.n, p.k, (int) nb, a.rowStride, a.colStride, sa, b.rowStride, b.colStride,
                    sb, c.rowStride, sc);
        }

        private static void requireTiles(String what, long extent) {
            if (tiles(extent) > MAX_GROUPS) {
                throw new IllegalArgumentException("matmul with " + extent + " " + what + " needs " + tiles(extent)
                        + " tiles of " + TrainOps.GEMM_TILE + ", above the " + MAX_GROUPS + " work-group limit of "
                        + Kernels.GEMM_F32);
            }
        }

        /** Enqueues this step at the given base offsets; the caller holds the backend lock. */
        void enqueue(TrainGpu g, GpuBuffer aBuf, int aOff, GpuBuffer bBuf, int bOff, GpuBuffer cBuf, int cOff,
                     float alpha, float beta) {
            switch (kind) {
                case GEMV_A:
                    g.launchRows(Kernels.GEMV_F32, rows, rows, k, aOff, aRs, aCs, bOff, bRs, cOff, alpha, beta,
                            aBuf, bBuf, cBuf);
                    return;
                case GEMV_BT:
                    g.launchRows(Kernels.GEMV_F32, n, n, k, bOff, bCs, bRs, aOff, aCs, cOff, alpha, beta,
                            bBuf, aBuf, cBuf);
                    return;
                default:
                    long[] local = TrainOps.localSize(Kernels.GEMM_F32);
                    long gx = (long) tiles(n) * local[0];
                    long gy = (long) tiles(rows) * local[1];
                    for (long start = 0; start < batch; start += MAX_GROUPS) {
                        int count = (int) Math.min(MAX_GROUPS, batch - start);
                        g.launch(Kernels.GEMM_F32, new long[]{gx, gy, count}, local,
                                rows, n, k, count,
                                (int) (aOff + start * aBs), aRs, aCs, aBs,
                                (int) (bOff + start * bBs), bRs, bCs, bBs,
                                (int) (cOff + start * cBs), cRs, cBs,
                                alpha, beta,
                                aBuf, bBuf, cBuf);
                    }
            }
        }
    }
}
