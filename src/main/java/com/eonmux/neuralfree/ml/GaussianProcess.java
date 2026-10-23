package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.gpu.MlGpu;
import com.eonmux.neuralfree.tensor.GpuMatmul;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;

/**
 * Gaussian-process regression with a zero prior mean.
 *
 * <p><b>Algorithm.</b> The kernels are RBF, MATERN32, MATERN52 and LINEAR ({@link MlGpu#gpKernel(float[], int, float[],
 * int, int, int, float, float)}). Fitting factors the training covariance {@code K + noise I} as {@code L L^T}
 * (Cholesky, in double precision) and solves {@code alpha = (K + noise I)^-1 y}. A test point's mean is
 * {@code k*.alpha} and its variance {@code max(0, k(x, x) - |L^-1 k*|^2)}, where {@code k*} holds its kernel values
 * against the training points.</p>
 *
 * <p><b>Execution.</b> When the train kernels are available on the backend ({@link TrainGpu#forBackend}), the training
 * covariance and the test cross-kernel matrices are computed on the device ({@code ML_GP_Kernel}), the training points
 * and {@code alpha} stay on the device, and the predicted means are the cross-kernel matrix times {@code alpha}
 * ({@link GpuMatmul#matmul}). The Cholesky factorization and the triangular solves are sequential and run on the host,
 * which reads back the training covariance once per fit and the cross-kernel matrix once per
 * prediction. Otherwise the CPU reference evaluates the same formulas. LINEAR kernel values agree with the CPU path in
 * the reduction class (bit for bit on OpenCL and CUDA for normal values), the others within the transcendental class,
 * and the means within the GEMM class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class GaussianProcess implements AutoCloseable {

    public enum KernelType {
        RBF,
        MATERN32,
        MATERN52,
        LINEAR
    }

    private final ComputeBackend backend;
    private final KernelType kernelType;
    private final float lengthScale;
    private final float variance;
    private final float noiseVariance;

    private int nTrain;
    private int nFeatures;
    private float[] xTrain;
    private float[] alpha;
    /** Row-major lower Cholesky factor of {@code K + noise I}. */
    private double[] cholesky;
    /** Device copies of the training points {@code [n, f]} and {@code alpha [n]}; null on the CPU path. */
    private Tensor xDevice;
    private Tensor alphaDevice;
    private boolean closed;

    public GaussianProcess(ComputeBackend backend, KernelType kernelType,
                           float lengthScale, float variance, float noiseVariance) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (kernelType == null) {
            throw new IllegalArgumentException("kernelType must not be null");
        }
        if (!(lengthScale > 0)) {
            throw new IllegalArgumentException("lengthScale must be positive");
        }
        if (!(variance > 0)) {
            throw new IllegalArgumentException("variance must be positive");
        }
        if (!(noiseVariance >= 0)) {
            throw new IllegalArgumentException("noiseVariance must be non-negative");
        }
        this.backend = backend;
        this.kernelType = kernelType;
        this.lengthScale = lengthScale;
        this.variance = variance;
        this.noiseVariance = noiseVariance;
    }

    public GaussianProcess(ComputeBackend backend) {
        this(backend, KernelType.RBF, 1.0f, 1.0f, 1e-6f);
    }

    /**
     * Fits the targets {@code y} of the rows of {@code x}, replacing any previous fit.
     *
     * @throws IllegalArgumentException when the inputs are null, empty or ragged, their lengths differ, the covariance
     *                                  would exceed {@code 2^31-1} elements, or {@code K + noise I} is not positive
     *                                  definite (a Cholesky pivot is not positive; a larger noise variance helps)
     */
    public void fit(float[][] x, float[] y) {
        ensureOpen();
        if (x == null || y == null) {
            throw new IllegalArgumentException("x and y must not be null");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must have same number of samples");
        }
        float[] xFlat = MlGpu.flatten(x, "x");
        int n = x.length;
        int f = x[0].length;
        MlGpu.elements(n, n, "kernel matrix");

        TrainGpu g = MlGpu.context(backend);
        Tensor newX = null;
        Tensor newAlpha = null;
        try (MlGpu.Scope scope = MlGpu.scope()) {
            float[] k;
            if (g != null) {
                newX = scope.add(MlGpu.upload(backend, xFlat, n, f));
                try (MlGpu.Scope temp = MlGpu.scope()) {
                    Tensor kt = temp.add(MlGpu.gpKernel(g, newX, newX, kernelId(), lengthScale, variance,
                            noiseVariance, true));
                    k = MlGpu.readFloats(kt);
                }
            } else {
                k = covarianceHost(xFlat, n, f);
            }
            double[] l = choleskyFactor(k, n);
            float[] a = solveAlpha(l, y, n);
            if (g != null) {
                newAlpha = scope.add(MlGpu.upload(backend, a, n));
                scope.keep(newX);
                scope.keep(newAlpha);
            }
            Tensor oldX = xDevice;
            Tensor oldAlpha = alphaDevice;
            this.xDevice = newX;
            this.alphaDevice = newAlpha;
            this.nTrain = n;
            this.nFeatures = f;
            this.xTrain = xFlat;
            this.alpha = a;
            this.cholesky = l;
            MlGpu.close(oldX, oldAlpha);
        }
    }

    /**
     * The predictive mean and variance of the rows of {@code xTest} (empty for no rows).
     *
     * @throws IllegalArgumentException when {@code xTest} is null or ragged, or its rows do not have the fitted feature
     *                                  count, or the cross-kernel matrix would exceed {@code 2^31-1} elements
     */
    public GPPrediction predict(float[][] xTest) {
        ensureOpen();
        if (xTest == null) {
            throw new IllegalArgumentException("xTest must not be null");
        }
        if (xTrain == null) {
            throw new IllegalStateException("GaussianProcess not fitted yet");
        }
        if (xTest.length == 0) {
            return new GPPrediction(new float[0], new float[0]);
        }
        float[] flat = MlGpu.flatten(xTest, "xTest");
        if (xTest[0].length != nFeatures) {
            throw new IllegalArgumentException("xTest rows must have " + nFeatures + " features, got "
                    + xTest[0].length);
        }
        int m = xTest.length;
        MlGpu.elements(m, nTrain, "cross-kernel matrix");

        float[] kStar = new float[m * nTrain];
        float[] mean = xDevice != null ? predictDevice(flat, m, kStar) : predictHost(flat, m, kStar);
        float[] var = new float[m];
        double[] v = new double[nTrain];
        int type = kernelId();
        for (int i = 0; i < m; i++) {
            forwardSubstitute(kStar, i * nTrain, v);
            double vv = 0.0;
            for (int j = 0; j < nTrain; j++) {
                vv += v[j] * v[j];
            }
            float kss = MlGpu.gpKernel(flat, i * nFeatures, flat, i * nFeatures, nFeatures, type, lengthScale,
                    variance);
            var[i] = (float) Math.max(0.0, kss - vv);
        }
        return new GPPrediction(mean, var);
    }

    private int kernelId() {
        switch (kernelType) {
            case RBF:
                return TrainOps.ML_GP_RBF;
            case MATERN32:
                return TrainOps.ML_GP_MATERN32;
            case MATERN52:
                return TrainOps.ML_GP_MATERN52;
            case LINEAR:
                return TrainOps.ML_GP_LINEAR;
            default:
                throw new IllegalStateException("unknown kernel type " + kernelType);
        }
    }

    // ------------------------------------------------------------------ device path

    /**
     * Means through the device cross-kernel matrix and {@code alpha}; fills {@code kStar} with that matrix, read back
     * for the triangular solves.
     */
    private float[] predictDevice(float[] flat, int m, float[] kStar) {
        TrainGpu g = MlGpu.context(backend);
        if (g == null) {
            throw new IllegalStateException("the train kernels are no longer available on " + backend.backendType()
                    + " for a model fitted on the device");
        }
        try (MlGpu.Scope scope = MlGpu.scope()) {
            Tensor x = scope.add(MlGpu.upload(backend, flat, m, nFeatures));
            Tensor k = scope.add(MlGpu.gpKernel(g, x, xDevice, kernelId(), lengthScale, variance, noiseVariance,
                    false));
            Tensor mean = scope.add(MlGpu.require(GpuMatmul.matmul(k, alphaDevice), "GpuMatmul.matmul"));
            float[] means = MlGpu.readFloats(mean);
            float[] kv = MlGpu.readFloats(k);
            System.arraycopy(kv, 0, kStar, 0, kv.length);
            return means;
        }
    }

    // ------------------------------------------------------------------ CPU reference

    /** {@code K + noise I}: the kernel of every training pair, with the noise added to the diagonal. */
    private float[] covarianceHost(float[] xFlat, int n, int f) {
        int type = kernelId();
        float[] k = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float v = MlGpu.gpKernel(xFlat, i * f, xFlat, j * f, f, type, lengthScale, variance);
                k[i * n + j] = i == j ? v + noiseVariance : v;
            }
        }
        return k;
    }

    /** Fills {@code kStar} with the cross-kernel matrix and returns {@code kStar.alpha} per row, ascending. */
    private float[] predictHost(float[] flat, int m, float[] kStar) {
        int type = kernelId();
        float[] mean = new float[m];
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < nTrain; j++) {
                float kv = MlGpu.gpKernel(flat, i * nFeatures, xTrain, j * nFeatures, nFeatures, type, lengthScale,
                        variance);
                kStar[i * nTrain + j] = kv;
                sum = sum + kv * alpha[j];
            }
            mean[i] = sum;
        }
        return mean;
    }

    // ------------------------------------------------------------------ factorization (host)

    /**
     * The lower Cholesky factor of the symmetric matrix {@code k} (lower triangle read), in double precision.
     *
     * @throws IllegalArgumentException when a pivot is not positive (or NaN): the matrix is not positive definite
     */
    private double[] choleskyFactor(float[] k, int n) {
        double[] l = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = k[i * n + j];
                for (int p = 0; p < j; p++) {
                    sum -= l[i * n + p] * l[j * n + p];
                }
                if (i == j) {
                    if (!(sum > 0.0)) {
                        throw new IllegalArgumentException("the training covariance K + noise I is not positive"
                                + " definite: Cholesky pivot " + i + " is " + sum + " (noiseVariance "
                                + noiseVariance + "; duplicate or near-duplicate points need a larger noise variance)");
                    }
                    l[i * n + i] = Math.sqrt(sum);
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        return l;
    }

    /** {@code alpha = L^-T L^-1 y}. */
    private static float[] solveAlpha(double[] l, float[] y, int n) {
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = y[i];
            for (int j = 0; j < i; j++) {
                sum -= l[i * n + j] * z[j];
            }
            z[i] = sum / l[i * n + i];
        }
        double[] a = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = z[i];
            for (int j = i + 1; j < n; j++) {
                sum -= l[j * n + i] * a[j];
            }
            a[i] = sum / l[i * n + i];
        }
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) a[i];
        }
        return out;
    }

    /** {@code v = L^-1 b} for the row of {@code b} starting at {@code offset}. */
    private void forwardSubstitute(float[] b, int offset, double[] v) {
        int n = nTrain;
        double[] l = cholesky;
        for (int i = 0; i < n; i++) {
            double sum = b[offset + i];
            for (int j = 0; j < i; j++) {
                sum -= l[i * n + j] * v[j];
            }
            v[i] = sum / l[i * n + i];
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GaussianProcess has been closed");
        }
    }

    /** Releases the device copies of the training points and {@code alpha}. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        Tensor x = xDevice;
        Tensor a = alphaDevice;
        xDevice = null;
        alphaDevice = null;
        MlGpu.close(x, a);
    }

    public static final class GPPrediction {
        private final float[] mean;
        private final float[] variance;

        GPPrediction(float[] mean, float[] variance) {
            this.mean = mean.clone();
            this.variance = variance.clone();
        }

        public float[] mean() {
            return mean.clone();
        }

        public float[] variance() {
            return variance.clone();
        }

        public float[] stdDev() {
            float[] std = new float[variance.length];
            for (int i = 0; i < variance.length; i++) {
                std[i] = (float) Math.sqrt(Math.max(0, variance[i]));
            }
            return std;
        }
    }
}
