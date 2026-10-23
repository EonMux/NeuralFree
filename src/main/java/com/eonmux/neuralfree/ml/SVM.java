package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.ml.gpu.MlGpu;
import com.eonmux.neuralfree.tensor.GpuMatmul;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;

import java.util.Arrays;

/**
 * Binary support vector classifier trained by sequential minimal optimization (SMO). Labels {@code >= 0} are the
 * positive class, the others the negative class; predictions are {@code 1} or {@code -1}.
 *
 * <p><b>Algorithm.</b> The kernels are LINEAR {@code x.y}, RBF {@code exp(-gamma |x - y|^2)} and POLY
 * {@code (gamma x.y + coef0)^degree} ({@link MlGpu#svmKernel(float[], int, float[], int, int, int, float, int,
 * float)}). SMO keeps the error cache {@code E_k = sum_l a_l y_l K_lk - y_k} (without the bias) and, after a step on
 * the pair {@code (i, j)}, updates it by {@code E_k += da_i y_i K_ik + da_j y_j K_jk}. The decision value of {@code x}
 * is {@code sum over support vectors (a_j y_j) K(x, sv_j) + b}, where the support vectors are the samples with
 * {@code a_j > tolerance}.</p>
 *
 * <p><b>Execution.</b> When the train kernels are available on the backend ({@link TrainGpu#forBackend}), the training
 * kernel matrix is computed on the device ({@code ML_SVM_Kernel}) and read back once for SMO, whose pair selection is
 * sequential and runs on the host. The support vectors and their coefficients stay on the device,
 * and {@link #decisionFunction} computes the cross-kernel matrix there and multiplies it by the coefficients
 * ({@link GpuMatmul#matmul}, a matrix-vector product). Otherwise the CPU reference evaluates the same formulas. The
 * LINEAR kernel values agree with the CPU path in the reduction class (bit for bit on OpenCL and CUDA for normal
 * values), the others within the transcendental class, and the decision values within the GEMM class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 */
public final class SVM implements AutoCloseable {

    public enum KernelType {
        LINEAR,
        RBF,
        POLY
    }

    private final ComputeBackend backend;
    private final KernelType kernelType;
    private final float c;
    private final float gamma;
    private final int degree;
    private final float coef0;
    private final float tolerance;
    private final int maxIterations;

    private int nFeatures;
    private float[] xTrain;
    private float[] yTrain;
    private float[] alpha;
    private float b;
    /** Indices of the support vectors in ascending order. */
    private int[] support;
    /** Coefficients {@code a_j y_j} of the support vectors, in {@link #support} order. */
    private float[] dualCoef;
    /** Device copies of the support vectors {@code [nsv, f]} and coefficients {@code [nsv]}; null on the CPU path. */
    private Tensor supportDevice;
    private Tensor coefDevice;
    private boolean closed;

    public SVM(ComputeBackend backend, KernelType kernelType, float c, float gamma,
               int degree, float coef0, float tolerance, int maxIterations) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (kernelType == null) {
            throw new IllegalArgumentException("kernelType must not be null");
        }
        if (!(c > 0)) {
            throw new IllegalArgumentException("C must be positive");
        }
        if (!(gamma > 0) && kernelType == KernelType.RBF) {
            throw new IllegalArgumentException("gamma must be positive for RBF kernel");
        }
        if (!(tolerance >= 0)) {
            throw new IllegalArgumentException("tolerance must be non-negative");
        }
        if (maxIterations < 1) {
            throw new IllegalArgumentException("maxIterations must be at least 1");
        }
        this.backend = backend;
        this.kernelType = kernelType;
        this.c = c;
        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    public SVM(ComputeBackend backend, KernelType kernelType) {
        this(backend, kernelType, 1.0f, 0.1f, 3, 0.0f, 1e-3f, 1000);
    }

    public SVM(ComputeBackend backend) {
        this(backend, KernelType.RBF);
    }

    /**
     * Trains on the rows of {@code x} with labels {@code y}, replacing any previous fit.
     *
     * @throws IllegalArgumentException when the inputs are null, empty or ragged, their lengths differ, or the kernel
     *                                  matrix would exceed {@code 2^31-1} elements
     */
    public void fit(float[][] x, int[] y) {
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
        float[] labels = new float[n];
        for (int i = 0; i < n; i++) {
            labels[i] = y[i] >= 0 ? 1.0f : -1.0f;
        }

        TrainGpu g = MlGpu.context(backend);
        float[] kernel = g != null ? kernelMatrixDevice(g, xFlat, n, f) : kernelMatrixHost(xFlat, n, f);
        Smo smo = new Smo(kernel, labels, n);
        smo.run();

        int[] sv = supportIndices(smo.alpha);
        float[] coef = new float[sv.length];
        float[] svRows = new float[sv.length * f];
        for (int s = 0; s < sv.length; s++) {
            coef[s] = smo.alpha[sv[s]] * labels[sv[s]];
            System.arraycopy(xFlat, sv[s] * f, svRows, s * f, f);
        }
        Tensor newSupport = null;
        Tensor newCoef = null;
        if (g != null && sv.length > 0) {
            try (MlGpu.Scope scope = MlGpu.scope()) {
                newSupport = scope.add(MlGpu.upload(backend, svRows, sv.length, f));
                newCoef = scope.add(MlGpu.upload(backend, coef, sv.length));
                scope.keep(newSupport);
                scope.keep(newCoef);
            }
        }
        Tensor oldSupport = supportDevice;
        Tensor oldCoef = coefDevice;
        this.supportDevice = newSupport;
        this.coefDevice = newCoef;
        this.nFeatures = f;
        this.xTrain = xFlat;
        this.yTrain = labels;
        this.alpha = smo.alpha;
        this.b = smo.b;
        this.support = sv;
        this.dualCoef = coef;
        MlGpu.close(oldSupport, oldCoef);
    }

    public int[] predict(float[][] x) {
        float[] decisions = decisionFunction(x);
        int[] predictions = new int[decisions.length];
        for (int i = 0; i < decisions.length; i++) {
            predictions[i] = decisions[i] >= 0 ? 1 : -1;
        }
        return predictions;
    }

    /**
     * The decision values of the rows of {@code x} (none for no rows).
     *
     * @throws IllegalArgumentException when {@code x} is null or ragged, or its rows do not have the fitted
     *                                  feature count
     */
    public float[] decisionFunction(float[][] x) {
        ensureOpen();
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        if (xTrain == null) {
            throw new IllegalStateException("SVM not fitted yet");
        }
        if (x.length == 0) {
            return new float[0];
        }
        float[] flat = MlGpu.flatten(x, "x");
        if (x[0].length != nFeatures) {
            throw new IllegalArgumentException("x rows must have " + nFeatures + " features, got " + x[0].length);
        }
        int m = x.length;
        float[] sums;
        if (support.length == 0) {
            sums = new float[m];
        } else if (supportDevice != null) {
            sums = supportSumsDevice(flat, m);
        } else {
            sums = supportSumsHost(flat, m);
        }
        float[] decisions = new float[m];
        for (int i = 0; i < m; i++) {
            decisions[i] = sums[i] + b;
        }
        return decisions;
    }

    public float[] getSupportVectors() {
        ensureOpen();
        if (xTrain == null || alpha == null) {
            throw new IllegalStateException("SVM not fitted yet");
        }
        float[] supportVectors = new float[support.length * nFeatures];
        for (int s = 0; s < support.length; s++) {
            System.arraycopy(xTrain, support[s] * nFeatures, supportVectors, s * nFeatures, nFeatures);
        }
        return supportVectors;
    }

    private int[] supportIndices(float[] a) {
        int[] idx = new int[a.length];
        int count = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > tolerance) {
                idx[count++] = i;
            }
        }
        return Arrays.copyOf(idx, count);
    }

    private int kernelId() {
        switch (kernelType) {
            case RBF:
                return TrainOps.ML_SVM_RBF;
            case POLY:
                return TrainOps.ML_SVM_POLY;
            case LINEAR:
                return TrainOps.ML_SVM_LINEAR;
            default:
                throw new IllegalStateException("unknown kernel type " + kernelType);
        }
    }

    // ------------------------------------------------------------------ device path

    /** The training kernel matrix computed on the device and read back for SMO. */
    private float[] kernelMatrixDevice(TrainGpu g, float[] xFlat, int n, int f) {
        try (MlGpu.Scope scope = MlGpu.scope()) {
            Tensor x = scope.add(MlGpu.upload(backend, xFlat, n, f));
            Tensor k = scope.add(MlGpu.svmKernel(g, x, x, kernelId(), gamma, degree, coef0));
            return MlGpu.readFloats(k);
        }
    }

    /** {@code sum_j coef_j K(x_i, sv_j)} for every row: the cross-kernel matrix times the coefficients. */
    private float[] supportSumsDevice(float[] flat, int m) {
        TrainGpu g = MlGpu.context(backend);
        if (g == null) {
            throw new IllegalStateException("the train kernels are no longer available on " + backend.backendType()
                    + " for a model fitted on the device");
        }
        try (MlGpu.Scope scope = MlGpu.scope()) {
            Tensor x = scope.add(MlGpu.upload(backend, flat, m, nFeatures));
            Tensor k = scope.add(MlGpu.svmKernel(g, x, supportDevice, kernelId(), gamma, degree, coef0));
            Tensor s = scope.add(MlGpu.require(GpuMatmul.matmul(k, coefDevice), "GpuMatmul.matmul"));
            return MlGpu.readFloats(s);
        }
    }

    // ------------------------------------------------------------------ CPU reference

    private float[] kernelMatrixHost(float[] xFlat, int n, int f) {
        int type = kernelId();
        float[] k = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                k[i * n + j] = MlGpu.svmKernel(xFlat, i * f, xFlat, j * f, f, type, gamma, degree, coef0);
            }
        }
        return k;
    }

    /** {@code sum_j coef_j K(x_i, sv_j)} in ascending support-vector order, each product rounded. */
    private float[] supportSumsHost(float[] flat, int m) {
        int type = kernelId();
        float[] sums = new float[m];
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int s = 0; s < support.length; s++) {
                float kv = MlGpu.svmKernel(flat, i * nFeatures, xTrain, support[s] * nFeatures, nFeatures, type, gamma,
                        degree, coef0);
                sum = sum + dualCoef[s] * kv;
            }
            sums[i] = sum;
        }
        return sums;
    }

    // ------------------------------------------------------------------ SMO (host)

    /** One SMO run over a row-major {@code n x n} kernel matrix. */
    private final class Smo {
        private final float[] kernel;
        private final float[] y;
        private final int n;
        final float[] alpha;
        final float[] error;
        float b;

        Smo(float[] kernel, float[] y, int n) {
            this.kernel = kernel;
            this.y = y;
            this.n = n;
            this.alpha = new float[n];
            this.error = new float[n];
            for (int i = 0; i < n; i++) {
                error[i] = -y[i];
            }
        }

        void run() {
            for (int iter = 0; iter < maxIterations; iter++) {
                boolean changed = false;
                for (int i = 0; i < n; i++) {
                    float r = y[i] * error[i];
                    if ((alpha[i] < c && r < -tolerance) || (alpha[i] > 0 && r > tolerance)) {
                        changed |= step(i, secondIndex(i));
                    }
                }
                if (!changed) {
                    break;
                }
            }
        }

        /** The {@code j != i} that maximizes {@code |y_i E_i - y_j E_j|}; the first such j, or {@code (i+1) % n}. */
        private int secondIndex(int i) {
            float maxDiff = 0;
            int best = (i + 1) % n;
            float ri = y[i] * error[i];
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                float diff = Math.abs(ri - y[j] * error[j]);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    best = j;
                }
            }
            return best;
        }

        /** Optimizes the pair {@code (i, j)}; false when the step is skipped. */
        private boolean step(int i, int j) {
            if (i == j) {
                return false;
            }
            float oldAi = alpha[i];
            float oldAj = alpha[j];
            float yi = y[i];
            float yj = y[j];
            float lower;
            float upper;
            if (yi != yj) {
                lower = Math.max(0, oldAj - oldAi);
                upper = Math.min(c, c + oldAj - oldAi);
            } else {
                lower = Math.max(0, oldAi + oldAj - c);
                upper = Math.min(c, oldAi + oldAj);
            }
            if (upper - lower < tolerance) {
                return false;
            }
            float eta = kernel[i * n + i] + kernel[j * n + j] - 2 * kernel[i * n + j];
            if (!(eta > 0)) {
                return false;
            }
            float newAj = oldAj + yj * (error[i] - error[j]) / eta;
            newAj = Math.max(lower, Math.min(upper, newAj));
            if (Math.abs(newAj - oldAj) < tolerance) {
                return false;
            }
            float newAi = oldAi + yi * yj * (oldAj - newAj);
            alpha[i] = newAi;
            alpha[j] = newAj;
            float di = (newAi - oldAi) * yi;
            float dj = (newAj - oldAj) * yj;
            for (int k = 0; k < n; k++) {
                error[k] += di * kernel[i * n + k] + dj * kernel[j * n + k];
            }
            updateBias(i, j);
            return true;
        }

        private void updateBias(int i, int j) {
            float bi = 0.0f;
            float bj = 0.0f;
            for (int k = 0; k < n; k++) {
                bi += alpha[k] * y[k] * kernel[i * n + k];
                bj += alpha[k] * y[k] * kernel[j * n + k];
            }
            bi = y[i] - bi;
            bj = y[j] - bj;
            if (0 < alpha[i] && alpha[i] < c) {
                b = bi;
            } else if (0 < alpha[j] && alpha[j] < c) {
                b = bj;
            } else {
                b = (bi + bj) / 2;
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SVM has been closed");
        }
    }

    /** Releases the device copies of the support vectors and coefficients. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        Tensor s = supportDevice;
        Tensor coef = coefDevice;
        supportDevice = null;
        coefDevice = null;
        MlGpu.close(s, coef);
    }
}
