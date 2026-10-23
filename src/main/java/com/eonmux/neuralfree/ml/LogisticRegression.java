package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.Arrays;

/**
 * Binary logistic regression trained by full-batch gradient descent with optional L2 regularization of the weights.
 *
 * <p><b>Training.</b> {@link #fit} starts from zero weights and bias and runs at most {@code maxIterations}
 * iterations. Each iteration computes p<sub>i</sub> = logistic(b + &Sigma;<sub>j</sub> w<sub>j</sub>
 * x<sub>ij</sub>) for every sample, the mean gradients grad_w[j] = (&Sigma;<sub>i</sub> (p<sub>i</sub> &minus;
 * y<sub>i</sub>) x<sub>ij</sub>) / n and grad_b = (&Sigma;<sub>i</sub> (p<sub>i</sub> &minus; y<sub>i</sub>)) / n,
 * the convergence measure m = max(max<sub>j</sub> |grad_w[j]|, |grad_b|) taken with {@code Math.max} semantics, and
 * the step w &larr; w &minus; lr (grad_w + regularization w), b &larr; b &minus; lr grad_b. Training stops after the
 * step of the first iteration whose m is below {@code tolerance} (a NaN measure never stops it). The logit is the
 * fused ascending sum of {@link Perceptron#affine} and the logistic is {@link Perceptron#logistic}; the gradient sums
 * run in ascending sample order, fused for the weights.</p>
 *
 * <p><b>Execution.</b> When the train kernel library is available on the backend at {@link #fit} time
 * ({@link TrainGpu#forBackend}), the data and parameters stay on the device during the fit: each iteration runs
 * ML_LogReg_Forward, ML_LogReg_Grad (1-D launches in work-groups of {@link TrainGpu#LINEAR_LOCAL}) and
 * ML_LogReg_Update (one work-group), and the only per-iteration readback is the 4-byte convergence measure through
 * {@link TrainGpu#readScalar}, which Java's stopping decision consumes; with
 * {@code tolerance <= 0} the measure can never stop training, so it is not read. The fitted parameters stay on the
 * device for {@link #predictProbability}, which runs ML_LogReg_Forward, until the next {@code fit} or {@link #close};
 * {@link #getWeights()} and {@link #getBias()} read them back. Otherwise the same math runs on the host (the CPU
 * reference path) and the backend is not used. The fused sums agree bit for bit between the paths on normal values;
 * the logistic follows the transcendental tolerance class and the sums the reduction class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 *
 * <p>Thread-safe: every method is serialized on the instance.</p>
 */
public final class LogisticRegression implements AutoCloseable {

    private final ComputeBackend backend;
    private final float learningRate;
    private final float regularization;
    private final int maxIterations;
    private final float tolerance;

    /** Fitted parameters on the device path; null before the first successful fit and on the CPU path. */
    private TensorStorage weightsDev;
    private TensorStorage biasDev;
    /** Fitted parameters on the CPU path; null before the first successful fit. */
    private float[] hostWeights;
    private float hostBias;

    /** Feature count of the fitted model; 0 while unfitted. */
    private int nFeatures;
    private boolean closed;

    /**
     * @param learningRate   step size, positive and finite
     * @param regularization L2 coefficient of the weights (not the bias), non-negative and finite
     * @param maxIterations  iteration limit, non-negative
     * @param tolerance      convergence threshold on the largest absolute gradient component, not NaN
     */
    public LogisticRegression(ComputeBackend backend, float learningRate, float regularization,
                              int maxIterations, float tolerance) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (!(learningRate > 0) || Float.isInfinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be positive and finite");
        }
        if (!(regularization >= 0) || Float.isInfinite(regularization)) {
            throw new IllegalArgumentException("regularization must be non-negative and finite");
        }
        if (maxIterations < 0) {
            throw new IllegalArgumentException("maxIterations must not be negative");
        }
        if (Float.isNaN(tolerance)) {
            throw new IllegalArgumentException("tolerance must not be NaN");
        }
        this.backend = backend;
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
    }

    public LogisticRegression(ComputeBackend backend) {
        this(backend, 0.01f, 0.0f, 1000, 1e-6f);
    }

    /**
     * Fits the model to {@code n >= 1} samples of equal, positive feature count with labels 0 or 1, replacing any
     * earlier fit.
     */
    public synchronized void fit(float[][] x, int[] y) {
        ensureOpen();
        if (x == null || y == null) {
            throw new IllegalArgumentException("x and y must not be null");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must have same length");
        }
        if (x.length == 0 || x[0] == null || x[0].length == 0) {
            throw new IllegalArgumentException("x must hold at least one sample with at least one feature");
        }
        int f = x[0].length;
        int n = Perceptron.requireBatch(x, f);
        float[] labels = new float[n];
        for (int i = 0; i < n; i++) {
            if (y[i] != 0 && y[i] != 1) {
                throw new IllegalArgumentException("labels must be 0 or 1, got " + y[i] + " at sample " + i);
            }
            labels[i] = y[i];
        }
        float[] flat = Perceptron.flatten(x, f);
        RuntimeException failure = releaseParameters(null);
        if (failure != null) {
            throw failure;
        }
        TrainGpu gpu = TrainGpu.forBackend(backend);
        if (gpu == null) {
            hostFit(flat, labels, n, f);
        } else {
            deviceFit(gpu, flat, labels, n, f);
        }
        nFeatures = f;
    }

    /** Probability of class 1 for each sample (computed on the device on the GPU path). */
    public synchronized float[] predictProbability(float[][] x) {
        ensureOpen();
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        ensureFitted();
        int n = Perceptron.requireBatch(x, nFeatures);
        if (n == 0) {
            return new float[0];
        }
        float[] flat = Perceptron.flatten(x, nFeatures);
        if (weightsDev == null) {
            float[] probabilities = new float[n];
            for (int i = 0; i < n; i++) {
                probabilities[i] = Perceptron.logistic(
                        Perceptron.affine(flat, i * nFeatures, hostWeights, hostBias, nFeatures));
            }
            return probabilities;
        }
        return devicePredict(Perceptron.context(backend), flat, n);
    }

    /** Class labels: 1 where {@link #predictProbability} is at least 0.5, else 0. */
    public synchronized int[] predict(float[][] x) {
        ensureOpen();
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        float[] probs = predictProbability(x);
        int[] predictions = new int[probs.length];
        for (int i = 0; i < probs.length; i++) {
            predictions[i] = probs[i] >= 0.5f ? 1 : 0;
        }
        return predictions;
    }

    /** The fitted weights (read back from the device on the GPU path). */
    public synchronized float[] getWeights() {
        ensureOpen();
        ensureFitted();
        return weightsDev == null ? hostWeights.clone() : Perceptron.download(backend, weightsDev, nFeatures);
    }

    /** The fitted bias (read back from the device on the GPU path). */
    public synchronized float getBias() {
        ensureOpen();
        ensureFitted();
        return biasDev == null ? hostBias : Perceptron.download(backend, biasDev, 1)[0];
    }

    /** Releases the fitted device parameters; idempotent. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = releaseParameters(null);
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ CPU reference path

    private void hostFit(float[] flat, float[] labels, int n, int f) {
        float[] w = new float[f];
        float b = 0.0f;
        float[] pred = new float[n];
        float[] sums = new float[f];
        float count = (float) n;
        for (int iter = 0; iter < maxIterations; iter++) {
            for (int i = 0; i < n; i++) {
                pred[i] = Perceptron.logistic(Perceptron.affine(flat, i * f, w, b, f));
            }
            Arrays.fill(sums, 0.0f);
            float biasSum = 0.0f;
            for (int i = 0; i < n; i++) {
                float e = pred[i] - labels[i];
                int row = i * f;
                for (int j = 0; j < f; j++) {
                    sums[j] = Math.fma(e, flat[row + j], sums[j]);
                }
                biasSum = biasSum + e;
            }
            float maxGrad = 0.0f;
            for (int j = 0; j < f; j++) {
                float g = sums[j] / count;
                maxGrad = Math.max(maxGrad, Math.abs(g));
                float wj = w[j];
                w[j] = wj - learningRate * (g + regularization * wj);
            }
            float gb = biasSum / count;
            maxGrad = Math.max(maxGrad, Math.abs(gb));
            b = b - learningRate * gb;
            if (maxGrad < tolerance) {
                break;
            }
        }
        hostWeights = w;
        hostBias = b;
    }

    // ------------------------------------------------------------------ device path

    private void deviceFit(TrainGpu gpu, float[] flat, float[] labels, int n, int f) {
        TensorStorage[] work = Perceptron.allocate(gpu, n * f, n, n, f, 1, 1);
        TensorStorage[] fitted = null;
        RuntimeException failure = null;
        try {
            fitted = Perceptron.allocate(gpu, f, 1);
            train(gpu, work, fitted[0], fitted[1], flat, labels, n, f);
        } catch (RuntimeException e) {
            failure = e;
        }
        failure = Perceptron.release(failure, work);
        if (failure != null) {
            if (fitted != null) {
                failure = Perceptron.release(failure, fitted);
            }
            throw failure;
        }
        weightsDev = fitted[0];
        biasDev = fitted[1];
    }

    /** Runs the iterations on {@code work = {x, y, predictions, gradW, gradB, maxGrad}} and the parameters. */
    private void train(TrainGpu gpu, TensorStorage[] work, TensorStorage w, TensorStorage b, float[] flat,
                       float[] labels, int n, int f) {
        TensorStorage xDev = work[0];
        TensorStorage yDev = work[1];
        TensorStorage predDev = work[2];
        TensorStorage gradWDev = work[3];
        TensorStorage gradBDev = work[4];
        TensorStorage maxGradDev = work[5];
        Perceptron.upload(backend, xDev, flat);
        Perceptron.upload(backend, yDev, labels);
        gpu.fill(w.gpuBuffer(), 0, f, 0.0f);
        gpu.fill(b.gpuBuffer(), 0, 1, 0.0f);
        boolean canConverge = tolerance > 0;
        for (int iter = 0; iter < maxIterations; iter++) {
            gpu.launch1d(Kernels.ML_LOGREG_FORWARD, n, n, f, xDev.gpuBuffer(), w.gpuBuffer(), b.gpuBuffer(),
                    predDev.gpuBuffer());
            gpu.launch1d(Kernels.ML_LOGREG_GRAD, f, n, f, xDev.gpuBuffer(), yDev.gpuBuffer(), predDev.gpuBuffer(),
                    gradWDev.gpuBuffer(), gradBDev.gpuBuffer());
            gpu.launchRows(Kernels.ML_LOGREG_UPDATE, 1, f, learningRate, regularization, w.gpuBuffer(),
                    gradWDev.gpuBuffer(), b.gpuBuffer(), gradBDev.gpuBuffer(), maxGradDev.gpuBuffer());
            if (canConverge && gpu.readScalar(maxGradDev.gpuBuffer(), 0) < tolerance) {
                break;
            }
        }
    }

    private float[] devicePredict(TrainGpu gpu, float[] flat, int n) {
        TensorStorage[] t = Perceptron.allocate(gpu, n * nFeatures, n);
        float[] probabilities = null;
        RuntimeException failure = null;
        try {
            Perceptron.upload(backend, t[0], flat);
            gpu.launch1d(Kernels.ML_LOGREG_FORWARD, n, n, nFeatures, t[0].gpuBuffer(), weightsDev.gpuBuffer(),
                    biasDev.gpuBuffer(), t[1].gpuBuffer());
            probabilities = Perceptron.download(backend, t[1], n);
        } catch (RuntimeException e) {
            failure = e;
        }
        failure = Perceptron.release(failure, t);
        if (failure != null) {
            throw failure;
        }
        return probabilities;
    }

    // ------------------------------------------------------------------ state

    /** Drops the fitted parameters (the model becomes unfitted), adding any close failure to {@code failure}. */
    private RuntimeException releaseParameters(RuntimeException failure) {
        RuntimeException result = Perceptron.release(failure, weightsDev, biasDev);
        weightsDev = null;
        biasDev = null;
        hostWeights = null;
        hostBias = 0.0f;
        nFeatures = 0;
        return result;
    }

    private void ensureFitted() {
        if (nFeatures == 0) {
            throw new IllegalStateException("LogisticRegression not fitted yet");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("LogisticRegression has been closed");
        }
    }
}
