package com.eonmux.neuralfree.ml.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Device support for the classical-ML models ({@code ml.KMeans}, {@code ml.PCA}, {@code ml.SVM},
 * {@code ml.GaussianProcess}): host arrays to contiguous FLOAT32 device tensors and back, and the launches of the
 * train library's ML kernels ({@code kernels/train/ml/kmeans.cl}, {@code kernels/train/ml/kernel_matrix.cl} and their
 * GLSL twins), together with the host twins of those kernels that the models' CPU paths use.
 *
 * <p><b>Contract.</b> Every launch takes the {@link TrainGpu} context of the tensors' backend (callers obtain it from
 * {@link #context} and take their CPU path when it is null). Input matrices are dense row-major {@code [rows, cols]}
 * FLOAT32 tensors, possibly row slices, addressed through their element offset. Every buffer a kernel
 * addresses at element offset 0 (distances, assignments, partial sums, centroid ping-pong buffers, the shift word,
 * kernel-matrix outputs) is a fresh allocation from {@link #allocate}. Kernel failures propagate as exceptions.</p>
 *
 * <p><b>Host twins.</b> {@link #sqDist} and {@link #dot} accumulate in ascending feature order with
 * {@link Math#fma(float, float, float)} from {@code +0}, exactly as the kernels do, and {@link #svmKernel(float[], int,
 * float[], int, int, int, float, int, float)} and {@link #gpKernel(float[], int, float[], int, int, int, float, float)}
 * evaluate the kernels' formulas with Java's left-to-right float association. On OpenCL and CUDA the device
 * accumulations agree with them bit for bit on normal values; on Vulkan an {@code fma} may round twice, which stays
 * within the reduction class. Results involving {@code exp}, {@code pow} or the device's integer power agree within
 * the transcendental class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class MlGpu {

    /**
     * Rows above which the k-means centroid update is split into row chunks; a shape-only rule, so the chunking never
     * depends on the device.
     */
    public static final int KMEANS_CHUNK_ROWS_ABOVE = 4096;
    /** Points per k-means chunk before the chunk count is capped. */
    public static final int KMEANS_CHUNK_ROWS = 1024;
    /** Largest k-means chunk count. */
    public static final int KMEANS_MAX_CHUNKS = 256;

    /** {@code (float) Math.sqrt(3)}, the constant of {@code ML_GP_Kernel}'s MATERN32 branch. */
    private static final float SQRT3 = (float) Math.sqrt(3);
    /** {@code (float) Math.sqrt(5)}, the constant of {@code ML_GP_Kernel}'s MATERN52 branch. */
    private static final float SQRT5 = (float) Math.sqrt(5);

    private MlGpu() {
    }

    // ------------------------------------------------------------------ context and transfers

    /** The train context of {@code backend}, or null when the train kernels are unavailable there (the CPU path). */
    public static TrainGpu context(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        return TrainGpu.forBackend(backend);
    }

    /**
     * The rows of {@code x} as one row-major array.
     *
     * @throws IllegalArgumentException when {@code x} is null or empty, a row is null, the rows differ in length, a row
     *                                  is empty, or the matrix has more than {@code 2^31-1} elements
     */
    public static float[] flatten(float[][] x, String name) {
        if (x == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        if (x.length == 0) {
            throw new IllegalArgumentException(name + " must have at least one row");
        }
        if (x[0] == null) {
            throw new IllegalArgumentException(name + " row 0 must not be null");
        }
        int cols = x[0].length;
        if (cols == 0) {
            throw new IllegalArgumentException(name + " rows must have at least one column");
        }
        float[] flat = new float[elements(x.length, cols, name)];
        for (int i = 0; i < x.length; i++) {
            if (x[i] == null) {
                throw new IllegalArgumentException(name + " row " + i + " must not be null");
            }
            if (x[i].length != cols) {
                throw new IllegalArgumentException(name + " row " + i + " has " + x[i].length + " columns, expected "
                        + cols);
            }
            System.arraycopy(x[i], 0, flat, i * cols, cols);
        }
        return flat;
    }

    /**
     * {@code rows * cols} as an int.
     *
     * @throws IllegalArgumentException when the product exceeds {@code 2^31-1}, the limit of the 32-bit kernel ABI
     */
    public static int elements(long rows, long cols, String what) {
        long n = rows * cols;
        if (rows < 0 || cols < 0 || n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " of " + rows + " x " + cols + " elements exceeds 2^31-1");
        }
        return (int) n;
    }

    /** A new contiguous FLOAT32 tensor of shape {@code dims} on {@code backend} holding {@code data} row-major. */
    public static Tensor upload(ComputeBackend backend, float[] data, int... dims) {
        Objects.requireNonNull(data, "data must not be null");
        Shape shape = Shape.of(dims);
        if (shape.numElements() != data.length) {
            throw new IllegalArgumentException("shape " + shape + " does not hold " + data.length + " elements");
        }
        TensorStorage storage = TensorStorage.fromHostFloat(backend, data);
        return own(backend, shape, DType.FLOAT32, storage);
    }

    /** A new uninitialized contiguous tensor of shape {@code dims} on the context's backend, at element offset 0. */
    public static Tensor allocate(TrainGpu g, DType dtype, int... dims) {
        Objects.requireNonNull(g, "train context must not be null");
        Shape shape = Shape.of(dims);
        TensorStorage storage = g.allocate(shape.numElements(), dtype);
        return own(g.backend(), shape, dtype, storage);
    }

    /** The FLOAT32 elements of {@code t} in row-major order (a host read after the pending train work). */
    public static float[] readFloats(Tensor t) {
        return Objects.requireNonNull(t, "tensor must not be null").toFloatArray();
    }

    /** The INT32 elements of {@code t} in row-major order (a host read after the pending train work). */
    public static int[] readInts(Tensor t) {
        return Objects.requireNonNull(t, "tensor must not be null").toIntArray();
    }

    /**
     * {@code t}, which a dispatcher returned while the train library was available on its backend.
     *
     * @throws IllegalStateException when {@code t} is null, which breaks the dispatcher contract
     */
    public static Tensor require(Tensor t, String op) {
        if (t == null) {
            throw new IllegalStateException(op + " returned no device result although the train library is available");
        }
        return t;
    }

    // ------------------------------------------------------------------ k-means

    /**
     * The chunk count of the k-means centroid update for {@code n} points:
     * {@code n > 4096 ? min(256, ceil(n / 1024)) : 1}. It depends only on the shape, so every backend and the CPU path
     * sum in the same order.
     */
    public static int kmeansChunks(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("k-means needs at least one point, got " + n);
        }
        if (n <= KMEANS_CHUNK_ROWS_ABOVE) {
            return 1;
        }
        int chunks = n / KMEANS_CHUNK_ROWS + (n % KMEANS_CHUNK_ROWS != 0 ? 1 : 0);
        return Math.min(KMEANS_MAX_CHUNKS, chunks);
    }

    /**
     * Assigns every point to its nearest centroid: {@code ML_KMeans_Distances} writes the squared distances
     * {@code dist[i*k + c]}, then {@code ML_KMeans_Assign} takes the first strict minimum from {@code Float.MAX_VALUE}
     * (cluster 0 when none is below it).
     *
     * @param points    {@code [n, d]}
     * @param centroids {@code [k, d]}
     * @param dist      {@code [n, k]} scratch at offset 0
     * @param assign    INT32 {@code [n]} at offset 0
     */
    public static void kmeansAssign(TrainGpu g, Tensor points, Tensor centroids, Tensor dist, Tensor assign) {
        StridedView p = matrix(points, "points");
        StridedView c = matrix(centroids, "centroids");
        int n = points.shape().dim(0);
        int d = points.shape().dim(1);
        int k = centroids.shape().dim(0);
        requireCols(centroids, d, "centroids");
        StridedView dv = fresh(dist, DType.FLOAT32, (long) n * k, "dist");
        StridedView av = fresh(assign, DType.INT32, n, "assign");
        g.launch1d(Kernels.ML_KMEANS_DISTANCES, (long) n * k, n, k, d, p.offset(), c.offset(),
                p.buffer(), c.buffer(), dv.buffer());
        g.launch1d(Kernels.ML_KMEANS_ASSIGN, n, n, k, dv.buffer(), av.buffer());
    }

    /**
     * New centroids from the assignment: {@code ML_KMeans_UpdatePartial} sums each chunk's points per cluster in
     * ascending point order and counts them, then {@code ML_KMeans_UpdateFinalize} adds the chunks in ascending order
     * and writes {@code sum / count}, or the old centroid for an empty cluster.
     *
     * @param psum   {@code chunks * k * d} FLOAT32 scratch at offset 0, with {@code chunks} from {@link #kmeansChunks}
     * @param pcount {@code chunks * k} INT32 scratch at offset 0
     * @param oldC   {@code [k, d]} centroids the assignment used, at offset 0
     * @param newC   {@code [k, d]} output at offset 0, a different buffer from {@code oldC}
     */
    public static void kmeansUpdate(TrainGpu g, Tensor points, Tensor assign, Tensor psum, Tensor pcount, Tensor oldC,
                                    Tensor newC) {
        StridedView p = matrix(points, "points");
        matrix(oldC, "old centroids");
        int n = points.shape().dim(0);
        int d = points.shape().dim(1);
        int k = oldC.shape().dim(0);
        int chunks = kmeansChunks(n);
        StridedView av = fresh(assign, DType.INT32, n, "assign");
        StridedView sv = fresh(psum, DType.FLOAT32, (long) chunks * k * d, "psum");
        StridedView cv = fresh(pcount, DType.INT32, (long) chunks * k, "pcount");
        StridedView ov = fresh(oldC, DType.FLOAT32, (long) k * d, "old centroids");
        StridedView nv = fresh(newC, DType.FLOAT32, (long) k * d, "new centroids");
        requireCols(oldC, d, "old centroids");
        requireCols(newC, d, "new centroids");
        requireDistinct(ov, nv, "old and new centroids");
        g.launch1d(Kernels.ML_KMEANS_UPDATE_PARTIAL, (long) chunks * k * d, n, k, d, chunks, p.offset(),
                p.buffer(), av.buffer(), sv.buffer(), cv.buffer());
        g.launch1d(Kernels.ML_KMEANS_UPDATE_FINALIZE, (long) k * d, k, d, chunks,
                sv.buffer(), cv.buffer(), ov.buffer(), nv.buffer());
    }

    /**
     * The k-means convergence measure: {@code ML_KMeans_Shift} (one work-group) computes the Java {@code Math.max} from
     * 0 over clusters of the float square root of the squared centroid movement, and this method reads that one float
     * back. This is the documented host synchronization point of k-means.
     *
     * @param shift FLOAT32 {@code [1]} scratch at offset 0
     */
    public static float kmeansShift(TrainGpu g, Tensor oldC, Tensor newC, Tensor shift) {
        matrix(oldC, "old centroids");
        int k = oldC.shape().dim(0);
        int d = oldC.shape().dim(1);
        StridedView ov = fresh(oldC, DType.FLOAT32, (long) k * d, "old centroids");
        StridedView nv = fresh(newC, DType.FLOAT32, (long) k * d, "new centroids");
        StridedView sv = fresh(shift, DType.FLOAT32, 1, "shift");
        requireCols(newC, d, "new centroids");
        g.launchRows(Kernels.ML_KMEANS_SHIFT, 1, k, d, ov.buffer(), nv.buffer(), sv.buffer());
        return g.readScalar(sv.buffer(), 0);
    }

    // ------------------------------------------------------------------ kernel matrices

    /**
     * The SVM kernel matrix {@code K[i, j] = kernel(a row i, b row j)} as a new {@code [m, n]} tensor
     * ({@code ML_SVM_Kernel}). Training passes the same tensor as {@code a} and {@code b}.
     *
     * @param type {@code TrainOps.ML_SVM_*}
     */
    public static Tensor svmKernel(TrainGpu g, Tensor a, Tensor b, int type, float gamma, int degree, float coef0) {
        StridedView av = matrix(a, "a");
        StridedView bv = matrix(b, "b");
        int m = a.shape().dim(0);
        int n = b.shape().dim(0);
        int f = a.shape().dim(1);
        requireCols(b, f, "b");
        Tensor k = allocate(g, DType.FLOAT32, m, n);
        try {
            g.launch1d(Kernels.ML_SVM_KERNEL, (long) m * n, m, n, f, type, gamma, degree, coef0,
                    av.offset(), bv.offset(), av.buffer(), bv.buffer(), k.storage().gpuBuffer());
        } catch (RuntimeException e) {
            k.close();
            throw e;
        }
        return k;
    }

    /**
     * The Gaussian-process kernel matrix as a new {@code [m, n]} tensor ({@code ML_GP_Kernel}). With
     * {@code noiseOnDiagonal} and {@code m == n} the diagonal carries {@code + noise} (the training covariance
     * {@code K + noise I}, for which training passes the same tensor as {@code a} and {@code b}); otherwise no element
     * does.
     *
     * @param type {@code TrainOps.ML_GP_*}
     */
    public static Tensor gpKernel(TrainGpu g, Tensor a, Tensor b, int type, float lengthScale, float variance,
                                  float noise, boolean noiseOnDiagonal) {
        StridedView av = matrix(a, "a");
        StridedView bv = matrix(b, "b");
        int m = a.shape().dim(0);
        int n = b.shape().dim(0);
        int f = a.shape().dim(1);
        requireCols(b, f, "b");
        Tensor k = allocate(g, DType.FLOAT32, m, n);
        try {
            g.launch1d(Kernels.ML_GP_KERNEL, (long) m * n, m, n, f, type, lengthScale, variance, noise,
                    av.offset(), bv.offset(), noiseOnDiagonal ? 1 : 0, av.buffer(), bv.buffer(),
                    k.storage().gpuBuffer());
        } catch (RuntimeException e) {
            k.close();
            throw e;
        }
        return k;
    }

    // ------------------------------------------------------------------ host twins

    /** Sum over {@code j < f} of {@code (a[ai + j] - b[bi + j])^2}, ascending, as {@code fma(diff, diff, acc)}. */
    public static float sqDist(float[] a, int ai, float[] b, int bi, int f) {
        float acc = 0.0f;
        for (int j = 0; j < f; j++) {
            float diff = a[ai + j] - b[bi + j];
            acc = Math.fma(diff, diff, acc);
        }
        return acc;
    }

    /** Sum over {@code j < f} of {@code a[ai + j] * b[bi + j]}, ascending, as {@code acc = fma(a, b, acc)}. */
    public static float dot(float[] a, int ai, float[] b, int bi, int f) {
        float acc = 0.0f;
        for (int j = 0; j < f; j++) {
            acc = Math.fma(a[ai + j], b[bi + j], acc);
        }
        return acc;
    }

    /**
     * The host twin of {@code ML_SVM_Kernel} for the rows at {@code ai} and {@code bi}: LINEAR {@code dot}, RBF
     * {@code exp(-gamma * sqdist)}, POLY {@code (float) Math.pow(gamma * dot + coef0, degree)}, any other type
     * {@code dot}.
     */
    public static float svmKernel(float[] a, int ai, float[] b, int bi, int f, int type, float gamma, int degree,
                                  float coef0) {
        switch (type) {
            case TrainOps.ML_SVM_RBF:
                return (float) Math.exp(-gamma * sqDist(a, ai, b, bi, f));
            case TrainOps.ML_SVM_POLY:
                return (float) Math.pow(gamma * dot(a, ai, b, bi, f) + coef0, degree);
            default:
                return dot(a, ai, b, bi, f);
        }
    }

    /**
     * The host twin of {@code ML_GP_Kernel} without noise for the rows at {@code ai} and {@code bi}: LINEAR
     * {@code variance * dot}; MATERN32 {@code t = sqrt3 * dist / ls}, {@code variance * (1 + t) * exp(-t)}; MATERN52
     * {@code t = sqrt5 * dist / ls}, {@code variance * (1 + t + 5 * sqdist / (3 * ls * ls)) * exp(-t)}; RBF and any
     * other type {@code variance * exp(-sqdist / (2 * ls * ls))}, with {@code dist} the float square root of
     * {@code sqdist}.
     */
    public static float gpKernel(float[] a, int ai, float[] b, int bi, int f, int type, float lengthScale,
                                 float variance) {
        if (type == TrainOps.ML_GP_LINEAR) {
            return variance * dot(a, ai, b, bi, f);
        }
        float sq = sqDist(a, ai, b, bi, f);
        if (type == TrainOps.ML_GP_MATERN32) {
            float t = SQRT3 * (float) Math.sqrt(sq) / lengthScale;
            return variance * (1.0f + t) * (float) Math.exp(-t);
        }
        if (type == TrainOps.ML_GP_MATERN52) {
            float t = SQRT5 * (float) Math.sqrt(sq) / lengthScale;
            float q = 5.0f * sq / (3.0f * lengthScale * lengthScale);
            return variance * (1.0f + t + q) * (float) Math.exp(-t);
        }
        return variance * (float) Math.exp(-sq / (2.0f * lengthScale * lengthScale));
    }

    // ------------------------------------------------------------------ resource scope

    /** A new empty {@link Scope}. */
    public static Scope scope() {
        return new Scope();
    }

    /**
     * Tensors closed together, in reverse order of {@link #add}, when the scope closes; a tensor handed out with
     * {@link #keep} survives. Used with try-with-resources, so a failure in the body stays the primary exception.
     */
    public static final class Scope implements AutoCloseable {
        private final List<Tensor> tensors = new ArrayList<>();

        private Scope() {
        }

        /** Registers {@code t} (null is ignored) and returns it. */
        public Tensor add(Tensor t) {
            if (t != null) {
                tensors.add(t);
            }
            return t;
        }

        /** Removes {@code t} from the scope, so it stays open, and returns it. */
        public Tensor keep(Tensor t) {
            for (int i = tensors.size() - 1; i >= 0; i--) {
                if (tensors.get(i) == t) {
                    tensors.remove(i);
                    break;
                }
            }
            return t;
        }

        /** Closes every registered tensor; the first failure is rethrown with the later ones suppressed. */
        @Override
        public void close() {
            RuntimeException failure = null;
            for (int i = tensors.size() - 1; i >= 0; i--) {
                try {
                    tensors.get(i).close();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            tensors.clear();
            if (failure != null) {
                throw failure;
            }
        }
    }

    /** Closes {@code tensors} (nulls are skipped); the first failure is rethrown with the later ones suppressed. */
    public static void close(Tensor... tensors) {
        try (Scope s = scope()) {
            for (Tensor t : tensors) {
                s.add(t);
            }
        }
    }

    // ------------------------------------------------------------------ validation

    private static Tensor own(ComputeBackend backend, Shape shape, DType dtype, TensorStorage storage) {
        try {
            return Tensor.fromStorage(backend, shape, dtype, storage);
        } catch (RuntimeException e) {
            storage.close();
            throw e;
        }
    }

    /** The view of a dense row-major FLOAT32 matrix (any element offset) with at least one row and column. */
    private static StridedView matrix(Tensor t, String name) {
        Objects.requireNonNull(t, name + " must not be null");
        if (t.dtype() != DType.FLOAT32 || t.shape().rank() != 2 || !t.isDense()) {
            throw new IllegalArgumentException(name + " must be a dense row-major FLOAT32 matrix, got " + t.dtype()
                    + " " + t.shape() + " with strides " + t.strides());
        }
        if (t.shape().dim(0) < 1 || t.shape().dim(1) < 1) {
            throw new IllegalArgumentException(name + " must have at least one row and one column, got " + t.shape());
        }
        return StridedView.of(t);
    }

    /** The view of a contiguous tensor of {@code elements} elements of {@code dtype} at element offset 0. */
    private static StridedView fresh(Tensor t, DType dtype, long elements, String name) {
        Objects.requireNonNull(t, name + " must not be null");
        if (t.dtype() != dtype || t.numElements() != elements || !t.isContiguous()) {
            throw new IllegalArgumentException(name + " must be a contiguous " + dtype + " tensor of " + elements
                    + " elements, got " + t.dtype() + " " + t.shape());
        }
        StridedView v = StridedView.of(t);
        if (v.offset() != 0) {
            throw new IllegalArgumentException(name + " must start at element offset 0, got " + v.offset());
        }
        return v;
    }

    private static void requireCols(Tensor t, int cols, String name) {
        if (t.shape().rank() != 2 || t.shape().dim(1) != cols) {
            throw new IllegalArgumentException(name + " must have " + cols + " columns, got " + t.shape());
        }
    }

    private static void requireDistinct(StridedView a, StridedView b, String what) {
        if (a.buffer() == b.buffer()) {
            throw new IllegalArgumentException(what + " must be different buffers");
        }
    }
}
