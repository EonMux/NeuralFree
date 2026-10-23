package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.gpu.MlGpu;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuMatmul;
import com.eonmux.neuralfree.tensor.GpuReduceOps;
import com.eonmux.neuralfree.tensor.GpuTensorOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Principal component analysis of row-major FLOAT32 data.
 *
 * <p><b>Algorithm.</b> The mean is each feature's sum over the samples in ascending order divided by the sample count;
 * the covariance is {@code alpha * sum_i xc[i][a] * xc[i][b]} with {@code alpha = 1f / (n - 1)} and {@code xc} the
 * centered data. Its eigen-decomposition uses cyclic Jacobi rotations in double precision until the off-diagonal mass
 * is below {@code 1e-24} of the total. Components are the eigenvectors of the largest eigenvalues (ties keep the lower
 * index), each signed so that its entry of largest magnitude (the first such entry) is positive; the explained
 * variance is {@code max(0, eigenvalue)}. The number of components kept is {@code min(nComponents, nSamples,
 * nFeatures)}.</p>
 *
 * <p><b>Execution.</b> When the train kernels are available on the backend ({@link TrainGpu#forBackend}), the mean
 * ({@link GpuReduceOps#mean}), centering ({@link GpuTensorOps#sub}), covariance ({@link GpuMatmul#gemm} of the
 * transposed centered data with itself) and {@link #transform} ({@link GpuMatmul#matmul} with the device copy of the
 * components) run on the device. The Jacobi iteration is sequential and runs on the host, which
 * reads back the covariance and the mean. Otherwise the CPU reference evaluates the same formulas; the two paths agree
 * within the reduction and GEMM classes of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 */
public final class PCA implements AutoCloseable {

    /** Largest number of Jacobi sweeps; a sweep rotates every off-diagonal pair once. */
    private static final int MAX_SWEEPS = 100;
    /** Convergence: off-diagonal sum of squares at most this fraction of the total sum of squares. */
    private static final double OFF_DIAGONAL_TOLERANCE = 1e-24;

    private final ComputeBackend backend;
    private final int nComponents;

    private float[] mean;
    private float[] components;
    private float[] explainedVariance;
    private int nFeatures;
    private int actualComponents;
    /** Device copies of the fitted mean {@code [f]} and components {@code [c, f]}; null on the CPU path. */
    private Tensor meanDevice;
    private Tensor componentsDevice;
    private boolean closed;

    public PCA(ComputeBackend backend, int nComponents) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (nComponents < 1) {
            throw new IllegalArgumentException("nComponents must be at least 1");
        }
        this.backend = backend;
        this.nComponents = nComponents;
    }

    /**
     * Fits the components of {@code nSamples} samples of {@code nFeatures} features stored row-major in {@code data},
     * replacing any previous fit.
     *
     * @throws IllegalArgumentException when the data is null or its length is not {@code nSamples * nFeatures},
     *                                  there are fewer than 2 samples (the covariance divides by {@code n - 1}), a
     *                                  size is not positive, {@code nComponents} exceeds {@code nFeatures}, or the
     *                                  covariance is not finite
     */
    public PCAResult fit(float[] data, int nSamples, int nFeatures) {
        ensureOpen();
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        if (nSamples < 1 || nFeatures < 1) {
            throw new IllegalArgumentException("nSamples and nFeatures must be positive, got " + nSamples + " and "
                    + nFeatures);
        }
        if (data.length != (long) nSamples * nFeatures) {
            throw new IllegalArgumentException("data length must equal nSamples * nFeatures");
        }
        if (nSamples == 1) {
            throw new IllegalArgumentException("PCA needs at least 2 samples: the covariance divides by nSamples - 1");
        }
        if (nComponents > nFeatures) {
            throw new IllegalArgumentException("nComponents cannot exceed nFeatures");
        }
        MlGpu.elements(nFeatures, nFeatures, "covariance matrix");
        int kept = Math.min(nComponents, Math.min(nSamples, nFeatures));
        float alpha = 1.0f / (nSamples - 1);

        TrainGpu g = MlGpu.context(backend);
        float[][] meanAndCov = g != null
                ? momentsDevice(g, data, nSamples, nFeatures, alpha)
                : momentsHost(data, nSamples, nFeatures, alpha);
        float[] fittedMean = meanAndCov[0];
        Eigen eigen = jacobi(meanAndCov[1], nFeatures);
        float[] fittedComponents = new float[kept * nFeatures];
        float[] fittedVariance = new float[kept];
        for (int c = 0; c < kept; c++) {
            int idx = eigen.order[c];
            System.arraycopy(eigen.vectors[idx], 0, fittedComponents, c * nFeatures, nFeatures);
            fittedVariance[c] = (float) Math.max(0.0, eigen.values[idx]);
        }

        Tensor newMean = null;
        Tensor newComponents = null;
        if (g != null) {
            try (MlGpu.Scope scope = MlGpu.scope()) {
                newMean = scope.add(MlGpu.upload(backend, fittedMean, nFeatures));
                newComponents = scope.add(MlGpu.upload(backend, fittedComponents, kept, nFeatures));
                scope.keep(newMean);
                scope.keep(newComponents);
            }
        }
        Tensor oldMean = meanDevice;
        Tensor oldComponents = componentsDevice;
        this.meanDevice = newMean;
        this.componentsDevice = newComponents;
        this.mean = fittedMean;
        this.components = fittedComponents;
        this.explainedVariance = fittedVariance;
        this.nFeatures = nFeatures;
        this.actualComponents = kept;
        MlGpu.close(oldMean, oldComponents);
        return new PCAResult(fittedMean, fittedComponents, fittedVariance, nSamples, nFeatures, kept);
    }

    /**
     * Projects row-major samples of the fitted feature count onto the fitted components: a row-major
     * {@code [samples, components]} result of {@code sum_j (x[j] - mean[j]) * component[j]}.
     *
     * @throws IllegalArgumentException when the data is null or its length is not a multiple of the feature count
     */
    public float[] transform(float[] data) {
        ensureOpen();
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        if (mean == null || components == null) {
            throw new IllegalStateException("PCA not fitted yet");
        }
        if (data.length % nFeatures != 0) {
            throw new IllegalArgumentException("data length " + data.length + " is not a multiple of nFeatures ("
                    + nFeatures + ")");
        }
        int samples = data.length / nFeatures;
        MlGpu.elements(samples, actualComponents, "transform result");
        if (samples == 0) {
            return new float[0];
        }
        return meanDevice != null ? transformDevice(data, samples) : transformHost(data, samples);
    }

    public float[] getMean() {
        ensureOpen();
        if (mean == null) {
            throw new IllegalStateException("PCA not fitted yet");
        }
        return mean.clone();
    }

    public float[] getComponents() {
        ensureOpen();
        if (components == null) {
            throw new IllegalStateException("PCA not fitted yet");
        }
        return components.clone();
    }

    public float[] getExplainedVariance() {
        ensureOpen();
        if (explainedVariance == null) {
            throw new IllegalStateException("PCA not fitted yet");
        }
        return explainedVariance.clone();
    }

    // ------------------------------------------------------------------ device path

    /** The mean and covariance computed on the device, read back for the host eigen-decomposition. */
    private float[][] momentsDevice(TrainGpu g, float[] data, int n, int f, float alpha) {
        try (MlGpu.Scope scope = MlGpu.scope()) {
            Tensor x = scope.add(MlGpu.upload(backend, data, n, f));
            Tensor m = scope.add(MlGpu.require(GpuReduceOps.mean(x, 0), "GpuReduceOps.mean"));
            Tensor xc = scope.add(MlGpu.require(GpuTensorOps.sub(x, m), "GpuTensorOps.sub"));
            Tensor xcT = scope.add(xc.transpose());
            Tensor cov = scope.add(MlGpu.allocate(g, DType.FLOAT32, f, f));
            if (!GpuMatmul.gemm(xcT, xc, alpha, 0.0f, cov)) {
                throw new IllegalStateException("GpuMatmul.gemm declined the covariance although the train library is"
                        + " available");
            }
            return new float[][]{MlGpu.readFloats(m), MlGpu.readFloats(cov)};
        }
    }

    private float[] transformDevice(float[] data, int samples) {
        try (MlGpu.Scope scope = MlGpu.scope()) {
            Tensor x = scope.add(MlGpu.upload(backend, data, samples, nFeatures));
            Tensor xc = scope.add(MlGpu.require(GpuTensorOps.sub(x, meanDevice), "GpuTensorOps.sub"));
            Tensor ct = scope.add(componentsDevice.transpose());
            Tensor out = scope.add(MlGpu.require(GpuMatmul.matmul(xc, ct), "GpuMatmul.matmul"));
            return MlGpu.readFloats(out);
        }
    }

    // ------------------------------------------------------------------ CPU reference

    /** Mean (ascending sum, one division) and covariance ({@code alpha * s}, s the ascending sum of products). */
    private static float[][] momentsHost(float[] data, int n, int f, float alpha) {
        float[] m = new float[f];
        for (int j = 0; j < f; j++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum = sum + data[i * f + j];
            }
            m[j] = sum / n;
        }
        float[] xc = new float[data.length];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < f; j++) {
                xc[i * f + j] = data[i * f + j] - m[j];
            }
        }
        float[] cov = new float[f * f];
        for (int a = 0; a < f; a++) {
            for (int b = 0; b < f; b++) {
                float s = 0.0f;
                for (int i = 0; i < n; i++) {
                    s = s + xc[i * f + a] * xc[i * f + b];
                }
                cov[a * f + b] = alpha * s;
            }
        }
        return new float[][]{m, cov};
    }

    private float[] transformHost(float[] data, int samples) {
        float[] result = new float[samples * actualComponents];
        float[] centered = new float[nFeatures];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                centered[j] = data[i * nFeatures + j] - mean[j];
            }
            for (int c = 0; c < actualComponents; c++) {
                float dot = 0.0f;
                for (int j = 0; j < nFeatures; j++) {
                    dot = dot + centered[j] * components[c * nFeatures + j];
                }
                result[i * actualComponents + c] = dot;
            }
        }
        return result;
    }

    // ------------------------------------------------------------------ eigen-decomposition (host)

    /** Eigenvalues, sign-normalized eigenvectors ({@code vectors[i]} belongs to {@code values[i]}) and their order. */
    private static final class Eigen {
        final double[] values;
        final float[][] vectors;
        final int[] order;

        Eigen(double[] values, float[][] vectors, int[] order) {
            this.values = values;
            this.vectors = vectors;
            this.order = order;
        }
    }

    /**
     * Cyclic Jacobi eigen-decomposition of the symmetric {@code n x n} matrix {@code cov} in double precision.
     *
     * @throws IllegalArgumentException when the matrix holds a non-finite value
     * @throws IllegalStateException    when the rotations do not converge within {@link #MAX_SWEEPS} sweeps
     */
    private static Eigen jacobi(float[] cov, int n) {
        double[] a = new double[n * n];
        double total = 0.0;
        for (int i = 0; i < a.length; i++) {
            if (!Float.isFinite(cov[i])) {
                throw new IllegalArgumentException("covariance is not finite: the data holds NaN or infinite values,"
                        + " or overflows");
            }
            a[i] = cov[i];
            total += a[i] * a[i];
        }
        double[] v = new double[n * n];
        for (int i = 0; i < n; i++) {
            v[i * n + i] = 1.0;
        }
        boolean converged = false;
        for (int sweep = 0; sweep < MAX_SWEEPS && !converged; sweep++) {
            double off = 0.0;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    off += 2.0 * a[p * n + q] * a[p * n + q];
                }
            }
            if (off <= OFF_DIAGONAL_TOLERANCE * total) {
                converged = true;
                break;
            }
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (a[p * n + q] != 0.0) {
                        rotate(a, v, n, p, q);
                    }
                }
            }
        }
        if (!converged) {
            throw new IllegalStateException("Jacobi eigen-decomposition did not converge in " + MAX_SWEEPS + " sweeps");
        }

        double[] values = new double[n];
        float[][] vectors = new float[n][n];
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            values[i] = a[i * n + i];
            order[i] = i;
            int pivot = 0;
            for (int j = 1; j < n; j++) {
                if (Math.abs(v[j * n + i]) > Math.abs(v[pivot * n + i])) {
                    pivot = j;
                }
            }
            double sign = v[pivot * n + i] < 0.0 ? -1.0 : 1.0;
            for (int j = 0; j < n; j++) {
                vectors[i][j] = (float) (sign * v[j * n + i]);
            }
        }
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> values[i]).reversed());
        int[] sorted = new int[n];
        for (int i = 0; i < n; i++) {
            sorted[i] = order[i];
        }
        return new Eigen(values, vectors, sorted);
    }

    /** One rotation {@code A <- J^T A J}, {@code V <- V J}, that zeroes {@code A[p][q]}. */
    private static void rotate(double[] a, double[] v, int n, int p, int q) {
        double app = a[p * n + p];
        double aqq = a[q * n + q];
        double apq = a[p * n + q];
        double theta = 0.5 * Math.atan2(2.0 * apq, app - aqq);
        double c = Math.cos(theta);
        double s = Math.sin(theta);
        for (int i = 0; i < n; i++) {
            double aip = a[i * n + p];
            double aiq = a[i * n + q];
            a[i * n + p] = c * aip + s * aiq;
            a[i * n + q] = -s * aip + c * aiq;
            double vip = v[i * n + p];
            double viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
        for (int j = 0; j < n; j++) {
            double apj = a[p * n + j];
            double aqj = a[q * n + j];
            a[p * n + j] = c * apj + s * aqj;
            a[q * n + j] = -s * apj + c * aqj;
        }
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
    }

    // ------------------------------------------------------------------ lifecycle

    private void releaseDevice() {
        Tensor m = meanDevice;
        Tensor c = componentsDevice;
        meanDevice = null;
        componentsDevice = null;
        MlGpu.close(m, c);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("PCA has been closed");
        }
    }

    /** Releases the device copies of the fitted model. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        releaseDevice();
    }

    public static final class PCAResult {
        private final float[] mean;
        private final float[] components;
        private final float[] explainedVariance;
        private final int nSamples;
        private final int nFeatures;
        private final int nComponents;

        PCAResult(float[] mean, float[] components, float[] explainedVariance,
                  int nSamples, int nFeatures, int nComponents) {
            this.mean = mean.clone();
            this.components = components.clone();
            this.explainedVariance = explainedVariance.clone();
            this.nSamples = nSamples;
            this.nFeatures = nFeatures;
            this.nComponents = nComponents;
        }

        public float[] mean() {
            return mean.clone();
        }

        public float[] components() {
            return components.clone();
        }

        public float[] explainedVariance() {
            return explainedVariance.clone();
        }

        public float[] explainedVarianceRatio() {
            float total = 0.0f;
            for (float v : explainedVariance) {
                total += v;
            }
            float[] ratio = new float[explainedVariance.length];
            if (total > 0) {
                for (int i = 0; i < ratio.length; i++) {
                    ratio[i] = explainedVariance[i] / total;
                }
            }
            return ratio;
        }

        public int nComponents() {
            return nComponents;
        }
    }
}
