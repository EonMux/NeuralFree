package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.gpu.MlGpu;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.Random;

/**
 * Lloyd's k-means on row-major FLOAT32 data.
 *
 * <p><b>Algorithm.</b> The initial centroids are {@code k} distinct points drawn with {@code new Random(42)}. Each
 * iteration assigns every point to the nearest centroid by squared Euclidean distance (the first strict minimum from
 * {@code Float.MAX_VALUE}, so ties go to the lowest cluster), recomputes each centroid as the mean of its points (an
 * empty cluster keeps its centroid), and stops once the largest centroid movement (Euclidean) is below the tolerance
 * or after {@code maxIterations}. The returned assignment is the one the last update used.</p>
 *
 * <p><b>Execution.</b> When the train kernels are available on the backend ({@link TrainGpu#forBackend}), the points
 * and both centroid buffers stay on the device for the whole fit and each iteration reads back one float, the
 * movement; the device buffers are released before {@link #fit} returns. Otherwise the CPU
 * reference runs: it evaluates the same operations in the same order (fused squared differences, chunked ascending
 * sums with the chunk count of {@link MlGpu#kmeansChunks}). On OpenCL and CUDA both paths therefore agree bit for bit
 * on normal values except for the division and square root (the division class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}); on Vulkan the fused squared differences may round
 * twice (the reduction class), which can change an assignment only at a near tie.</p>
 */
public final class KMeans implements AutoCloseable {

    private final ComputeBackend backend;
    private final int k;
    private final int maxIterations;
    private final float tolerance;

    private int dimensions;
    private float[] centroids;
    private boolean closed;

    public KMeans(ComputeBackend backend, int k, int maxIterations, float tolerance) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1");
        }
        if (maxIterations < 1) {
            throw new IllegalArgumentException("maxIterations must be at least 1");
        }
        if (!(tolerance > 0)) {
            throw new IllegalArgumentException("tolerance must be positive");
        }
        this.backend = backend;
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
    }

    public KMeans(ComputeBackend backend, int k) {
        this(backend, k, 100, 1e-4f);
    }

    /**
     * Clusters {@code numPoints} points of {@code dimensions} features stored row-major in {@code data}.
     *
     * @throws IllegalArgumentException when the data is null, its length is not {@code numPoints * dimensions}, a size
     *                                  is not positive, or {@code k} exceeds the number of points
     */
    public KMeansResult fit(float[] data, int numPoints, int dimensions) {
        ensureOpen();
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        if (numPoints < 1 || dimensions < 1) {
            throw new IllegalArgumentException("numPoints and dimensions must be positive, got " + numPoints + " and "
                    + dimensions);
        }
        if (data.length != (long) numPoints * dimensions) {
            throw new IllegalArgumentException("data length must equal numPoints * dimensions");
        }
        if (k > numPoints) {
            throw new IllegalArgumentException("k (" + k + ") must not exceed numPoints (" + numPoints + ")");
        }
        MlGpu.elements(numPoints, k, "k-means distance matrix");
        MlGpu.elements((long) MlGpu.kmeansChunks(numPoints) * k, dimensions, "k-means partial sums");

        float[] initial = initialCentroids(data, numPoints, dimensions);
        TrainGpu g = MlGpu.context(backend);
        KMeansResult result = g != null
                ? fitDevice(g, data, numPoints, dimensions, initial)
                : fitHost(data, numPoints, dimensions, initial);
        this.dimensions = dimensions;
        this.centroids = result.centroids();
        return result;
    }

    /** The cluster whose centroid is nearest to {@code point}, with the assignment rule of {@link #fit}. */
    public int predict(float[] point) {
        ensureOpen();
        if (centroids == null) {
            throw new IllegalStateException("Model not fitted yet");
        }
        if (point == null || point.length != dimensions) {
            throw new IllegalArgumentException("point must have " + dimensions + " dimensions");
        }
        int best = 0;
        float bestDistance = Float.MAX_VALUE;
        for (int c = 0; c < k; c++) {
            float distance = MlGpu.sqDist(point, 0, centroids, c * dimensions, dimensions);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = c;
            }
        }
        return best;
    }

    public float[] getCentroids() {
        ensureOpen();
        if (centroids == null) {
            throw new IllegalStateException("Model not fitted yet");
        }
        return centroids.clone();
    }

    /** Draws {@code k} distinct point indices with {@code new Random(42)} and copies those points. */
    private float[] initialCentroids(float[] data, int numPoints, int dims) {
        Random rng = new Random(42);
        int[] indices = new int[k];
        for (int i = 0; i < k; i++) {
            indices[i] = rng.nextInt(numPoints);
            for (int j = 0; j < i; j++) {
                if (indices[j] == indices[i]) {
                    i--;
                    break;
                }
            }
        }
        float[] initial = new float[k * dims];
        for (int i = 0; i < k; i++) {
            System.arraycopy(data, indices[i] * dims, initial, i * dims, dims);
        }
        return initial;
    }

    // ------------------------------------------------------------------ device path

    private KMeansResult fitDevice(TrainGpu g, float[] data, int n, int d, float[] initial) {
        int chunks = MlGpu.kmeansChunks(n);
        try (MlGpu.Scope scope = MlGpu.scope()) {
            Tensor points = scope.add(MlGpu.upload(backend, data, n, d));
            Tensor current = scope.add(MlGpu.upload(backend, initial, k, d));
            Tensor next = scope.add(MlGpu.allocate(g, DType.FLOAT32, k, d));
            Tensor dist = scope.add(MlGpu.allocate(g, DType.FLOAT32, n, k));
            Tensor assign = scope.add(MlGpu.allocate(g, DType.INT32, n));
            Tensor psum = scope.add(MlGpu.allocate(g, DType.FLOAT32, chunks * k * d));
            Tensor pcount = scope.add(MlGpu.allocate(g, DType.INT32, chunks * k));
            Tensor shift = scope.add(MlGpu.allocate(g, DType.FLOAT32, 1));

            int iterations = 0;
            boolean converged = false;
            for (int iter = 0; iter < maxIterations; iter++) {
                iterations = iter + 1;
                MlGpu.kmeansAssign(g, points, current, dist, assign);
                MlGpu.kmeansUpdate(g, points, assign, psum, pcount, current, next);
                float maxShift = MlGpu.kmeansShift(g, current, next, shift);
                Tensor swap = current;
                current = next;
                next = swap;
                if (maxShift < tolerance) {
                    converged = true;
                    break;
                }
            }
            return new KMeansResult(MlGpu.readFloats(current), MlGpu.readInts(assign), iterations, converged);
        }
    }

    // ------------------------------------------------------------------ CPU reference

    private KMeansResult fitHost(float[] data, int n, int d, float[] initial) {
        int chunks = MlGpu.kmeansChunks(n);
        float[] current = initial;
        float[] next = new float[k * d];
        int[] assign = new int[n];
        float[] psum = new float[chunks * k * d];
        int[] pcount = new int[chunks * k];

        int iterations = 0;
        boolean converged = false;
        for (int iter = 0; iter < maxIterations; iter++) {
            iterations = iter + 1;
            hostAssign(data, n, d, current, assign);
            hostUpdate(data, n, d, chunks, assign, psum, pcount, current, next);
            float maxShift = hostShift(d, current, next);
            float[] swap = current;
            current = next;
            next = swap;
            if (maxShift < tolerance) {
                converged = true;
                break;
            }
        }
        return new KMeansResult(current, assign, iterations, converged);
    }

    /** ML_KMeans_Distances and ML_KMeans_Assign, one point at a time. */
    private void hostAssign(float[] points, int n, int d, float[] centroids, int[] assign) {
        for (int i = 0; i < n; i++) {
            float best = Float.MAX_VALUE;
            int bestC = 0;
            for (int c = 0; c < k; c++) {
                float v = MlGpu.sqDist(points, i * d, centroids, c * d, d);
                if (v < best) {
                    best = v;
                    bestC = c;
                }
            }
            assign[i] = bestC;
        }
    }

    /** ML_KMeans_UpdatePartial over chunks of {@code ceil(n / chunks)} points, then ML_KMeans_UpdateFinalize. */
    private void hostUpdate(float[] points, int n, int d, int chunks, int[] assign, float[] psum, int[] pcount,
                            float[] oldC, float[] newC) {
        int chunkSize = n / chunks + (n % chunks != 0 ? 1 : 0);
        for (int ch = 0; ch < chunks; ch++) {
            int begin = ch * chunkSize;
            int end = (n - begin > chunkSize) ? begin + chunkSize : n;
            for (int c = 0; c < k; c++) {
                int count = 0;
                int base = (ch * k + c) * d;
                for (int j = 0; j < d; j++) {
                    psum[base + j] = 0.0f;
                }
                for (int i = begin; i < end; i++) {
                    if (assign[i] == c) {
                        for (int j = 0; j < d; j++) {
                            psum[base + j] = psum[base + j] + points[i * d + j];
                        }
                        count++;
                    }
                }
                pcount[ch * k + c] = count;
            }
        }
        int stride = k * d;
        for (int t = 0; t < stride; t++) {
            int c = t / d;
            float s = psum[t];
            int count = pcount[c];
            for (int ch = 1; ch < chunks; ch++) {
                s = s + psum[ch * stride + t];
                count += pcount[ch * k + c];
            }
            newC[t] = count > 0 ? s / (float) count : oldC[t];
        }
    }

    /** ML_KMeans_Shift: Java {@code Math.max} from 0 over clusters of the float sqrt of the fused squared movement. */
    private float hostShift(int d, float[] oldC, float[] newC) {
        float m = 0.0f;
        for (int c = 0; c < k; c++) {
            float s = MlGpu.sqDist(newC, c * d, oldC, c * d, d);
            m = Math.max(m, (float) Math.sqrt(s));
        }
        return m;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("KMeans has been closed");
        }
    }

    /** Marks the model closed; it holds no device buffers between calls. */
    @Override
    public void close() {
        closed = true;
    }

    public static final class KMeansResult {
        private final float[] centroids;
        private final int[] assignments;
        private final int iterations;
        private final boolean converged;

        KMeansResult(float[] centroids, int[] assignments, int iterations, boolean converged) {
            this.centroids = centroids.clone();
            this.assignments = assignments.clone();
            this.iterations = iterations;
            this.converged = converged;
        }

        public float[] centroids() {
            return centroids.clone();
        }

        public int[] assignments() {
            return assignments.clone();
        }

        public int iterations() {
            return iterations;
        }

        public boolean converged() {
            return converged;
        }
    }
}
