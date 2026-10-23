package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

/**
 * Online forecast unit: one shared feature row driving two linear heads, trained by streaming mini-batch SGD steps.
 *
 * <ul>
 *   <li><b>Direction head</b>: a logistic unit estimating the probability that a forecast's direction materializes.
 *       Trained with the cross-entropy gradient on soft labels in [0, 1]: a label of 0.5 encodes "reality showed no
 *       edge either way" and pulls the estimate toward the prior instead of punishing it.</li>
 *   <li><b>Magnitude head</b>: a linear unit estimating the expected move in the caller's normalized units. Trained
 *       with the Huber gradient (per-sample error clamped to &plusmn;1), so one fat-tailed
 *       outlier move cannot blow up the online estimate; with that bound both heads' errors live in [-1, 1] and one
 *       shared learning rate serves both.</li>
 * </ul>
 *
 * <p>Where {@link Perceptron} is a single logistic unit judging a binary outcome, this class exists for
 * forecast-versus-reality learning: predict what will happen, compare when reality catches up, and descend the
 * forecast error. Call {@link #update} whenever graded forecasts arrive and {@link #predict} at decision time.
 * Deterministic zero initialization (each head is a single linear unit with no symmetry to break), optional L2
 * weight decay, thread-safe (every method is serialized on the instance).</p>
 *
 * <p><b>Execution.</b> When the train kernel library is available on the backend at construction
 * ({@link TrainGpu#forBackend}), the parameters of both heads live in device buffers for the lifetime of the model and
 * each step runs as the ML_Forecast_Forward, ML_Forecast_Grad and ML_Forecast_Update kernels (1-D launches in
 * work-groups of {@link TrainGpu#LINEAR_LOCAL}) on the backend's compute queue, with no host synchronization;
 * parameters leave the device only through {@link #predict} and the explicit readbacks, and {@link #setParameters}
 * uploads them. Batch buffers grow on demand and are reused. Otherwise the same math runs on the host (the CPU
 * reference path) and the backend is not used.</p>
 *
 * <p><b>Math</b> (both paths; {@code kernels/train/ml/forecast.cl}). Each head's pre-activation is the fused
 * ascending sum of {@link Perceptron#affine}; the direction head applies {@link Perceptron#logistic}. Per batch of n
 * samples with importance weights s: e<sub>d,i</sub> = p<sub>i</sub> &minus; y<sub>d,i</sub> and e<sub>m,i</sub> =
 * {@code Math.max(-1, Math.min(1, v_i - y_m,i))}; grad_w[j] = (&Sigma;<sub>i</sub> (s<sub>i</sub> e<sub>i</sub>)
 * x<sub>ij</sub>) / n and grad_b = (&Sigma;<sub>i</sub> s<sub>i</sub> e<sub>i</sub>) / n per head, fused in ascending
 * i; then w &larr; w &minus; lr (grad_w + l2 w) and b &larr; b &minus; lr grad_b. The fused sums agree bit for bit
 * between the paths on normal values; the logistic follows the transcendental tolerance class and the sums the
 * reduction class of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 */
public final class ForecastUnit implements AutoCloseable {

    /** Bound of the magnitude head's per-sample error ({@code FC_HUBER_DELTA} in forecast.cl and forecast.comp). */
    static final float HUBER_DELTA = 1.0f;

    private final ComputeBackend backend;
    /** The parameters live on the device (the train kernels were available at construction); else the CPU path. */
    private final boolean onDevice;
    private final int nFeatures;
    private final float learningRate;
    private final float l2;

    /** Every device parameter and gradient storage below, for release. */
    private final TensorStorage[] params;
    private final TensorStorage dirWDev;
    private final TensorStorage dirBDev;
    private final TensorStorage moveWDev;
    private final TensorStorage moveBDev;
    private final TensorStorage gradDirWDev;
    private final TensorStorage gradDirBDev;
    private final TensorStorage gradMoveWDev;
    private final TensorStorage gradMoveBDev;
    private TensorStorage xDev;
    private TensorStorage yDirDev;
    private TensorStorage yMoveDev;
    private TensorStorage sampleWDev;
    private TensorStorage predDirDev;
    private TensorStorage predMoveDev;
    private int batchCapacity;

    private final float[] hostDirWeights;
    private final float[] hostMoveWeights;
    private float hostDirBias;
    private float hostMoveBias;

    private long samplesSeen;
    private boolean closed;

    public ForecastUnit(ComputeBackend backend, int nFeatures, float learningRate, float l2) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (nFeatures <= 0) {
            throw new IllegalArgumentException("nFeatures must be positive");
        }
        if (!(learningRate > 0) || Float.isInfinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be positive and finite");
        }
        if (!(l2 >= 0) || Float.isInfinite(l2)) {
            throw new IllegalArgumentException("l2 must be non-negative and finite");
        }
        this.backend = backend;
        this.nFeatures = nFeatures;
        this.learningRate = learningRate;
        this.l2 = l2;
        TrainGpu gpu = TrainGpu.forBackend(backend);
        this.onDevice = gpu != null;
        if (gpu == null) {
            this.params = new TensorStorage[0];
            this.dirWDev = null;
            this.dirBDev = null;
            this.moveWDev = null;
            this.moveBDev = null;
            this.gradDirWDev = null;
            this.gradDirBDev = null;
            this.gradMoveWDev = null;
            this.gradMoveBDev = null;
            this.hostDirWeights = new float[nFeatures];
            this.hostMoveWeights = new float[nFeatures];
            return;
        }
        this.hostDirWeights = null;
        this.hostMoveWeights = null;
        this.params = Perceptron.allocate(gpu, nFeatures, 1, nFeatures, 1, nFeatures, 1, nFeatures, 1);
        this.dirWDev = params[0];
        this.dirBDev = params[1];
        this.moveWDev = params[2];
        this.moveBDev = params[3];
        this.gradDirWDev = params[4];
        this.gradDirBDev = params[5];
        this.gradMoveWDev = params[6];
        this.gradMoveBDev = params[7];
        try {
            gpu.fill(dirWDev.gpuBuffer(), 0, nFeatures, 0.0f);
            gpu.fill(dirBDev.gpuBuffer(), 0, 1, 0.0f);
            gpu.fill(moveWDev.gpuBuffer(), 0, nFeatures, 0.0f);
            gpu.fill(moveBDev.gpuBuffer(), 0, 1, 0.0f);
        } catch (RuntimeException e) {
            throw Perceptron.release(e, params);
        }
    }

    public ForecastUnit(ComputeBackend backend, int nFeatures) {
        this(backend, nFeatures, 0.05f, 0.0f);
    }

    /** Forecasts for a batch: one {@code {directionProbability, expectedMove}} pair per sample. */
    public synchronized float[][] predict(float[][] x) {
        ensureOpen();
        int n = Perceptron.requireBatch(x, nFeatures);
        if (n == 0) {
            return new float[0][];
        }
        float[] flat = Perceptron.flatten(x, nFeatures);
        float[] dir;
        float[] move;
        if (!onDevice) {
            dir = new float[n];
            move = new float[n];
            hostForward(flat, n, dir, move);
        } else {
            TrainGpu gpu = Perceptron.context(backend);
            ensureBatchCapacity(gpu, n);
            Perceptron.upload(backend, xDev, flat);
            launchForward(gpu, n);
            dir = Perceptron.download(backend, predDirDev, n);
            move = Perceptron.download(backend, predMoveDev, n);
        }
        float[][] out = new float[n][2];
        for (int i = 0; i < n; i++) {
            out[i][0] = dir[i];
            out[i][1] = move[i];
        }
        return out;
    }

    /** Forecast for a single sample: {@code {directionProbability, expectedMove}}. */
    public float[] predictOne(float[] x) {
        return predict(new float[][]{x})[0];
    }

    /**
     * One streaming SGD step on a batch of graded forecasts: forward, mean per-head gradients (cross-entropy for
     * direction, Huber for magnitude) and in-place parameter updates; nothing is read back to the host. Every sample
     * contributes at importance 1; see {@link #update(float[][], float[], float[], float[])} to weight samples
     * unevenly.
     *
     * @param yDirection soft direction labels in [0, 1]: how much reality agreed with the forecast direction
     * @param yMove      realized moves in the caller's normalized units (finite; the Huber bound caps each sample's
     *                   pull)
     */
    public void update(float[][] x, float[] yDirection, float[] yMove) {
        update(x, yDirection, yMove, null);
    }

    /**
     * {@link #update(float[][], float[], float[])} with per-sample importance weights: each sample's gradient
     * contribution is scaled by its weight, so a weight-4 outcome pulls the parameters four times as hard as a
     * weight-1 one (weight 1 is an ordinary sample; {@code null} means all ones). Weights must be finite and
     * non-negative.
     */
    public synchronized void update(float[][] x, float[] yDirection, float[] yMove, float[] sampleWeights) {
        ensureOpen();
        int n = Perceptron.requireBatch(x, nFeatures);
        if (yDirection == null || yDirection.length != n) {
            throw new IllegalArgumentException("yDirection must match the batch size");
        }
        if (yMove == null || yMove.length != n) {
            throw new IllegalArgumentException("yMove must match the batch size");
        }
        for (float v : yDirection) {
            if (!(v >= 0.0f) || v > 1.0f) {
                throw new IllegalArgumentException("yDirection labels must be within [0, 1]");
            }
        }
        for (float v : yMove) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                throw new IllegalArgumentException("yMove labels must be finite");
            }
        }
        Perceptron.requireSampleWeights(sampleWeights, n);
        if (n == 0) {
            return;
        }
        float[] flat = Perceptron.flatten(x, nFeatures);
        if (!onDevice) {
            hostStep(flat, yDirection, yMove, sampleWeights, n);
        } else {
            deviceStep(flat, yDirection, yMove, sampleWeights, n);
        }
        samplesSeen += n;
    }

    /** Current direction-head weights (read back from the device on the GPU path). */
    public synchronized float[] directionWeights() {
        ensureOpen();
        return onDevice ? Perceptron.download(backend, dirWDev, nFeatures) : hostDirWeights.clone();
    }

    /** Current direction-head bias (read back from the device on the GPU path). */
    public synchronized float directionBias() {
        ensureOpen();
        return onDevice ? Perceptron.download(backend, dirBDev, 1)[0] : hostDirBias;
    }

    /** Current magnitude-head weights (read back from the device on the GPU path). */
    public synchronized float[] moveWeights() {
        ensureOpen();
        return onDevice ? Perceptron.download(backend, moveWDev, nFeatures) : hostMoveWeights.clone();
    }

    /** Current magnitude-head bias (read back from the device on the GPU path). */
    public synchronized float moveBias() {
        ensureOpen();
        return onDevice ? Perceptron.download(backend, moveBDev, 1)[0] : hostMoveBias;
    }

    /** Restores persisted parameters for both heads (uploaded to the device on the GPU path). */
    public synchronized void setParameters(float[] directionWeights, float directionBias,
                                           float[] moveWeights, float moveBias) {
        ensureOpen();
        if (directionWeights == null || directionWeights.length != nFeatures
                || moveWeights == null || moveWeights.length != nFeatures) {
            throw new IllegalArgumentException("both heads' weights must have length " + nFeatures);
        }
        if (!onDevice) {
            System.arraycopy(directionWeights, 0, hostDirWeights, 0, nFeatures);
            System.arraycopy(moveWeights, 0, hostMoveWeights, 0, nFeatures);
            hostDirBias = directionBias;
            hostMoveBias = moveBias;
            return;
        }
        Perceptron.upload(backend, dirWDev, directionWeights);
        Perceptron.upload(backend, dirBDev, new float[]{directionBias});
        Perceptron.upload(backend, moveWDev, moveWeights);
        Perceptron.upload(backend, moveBDev, new float[]{moveBias});
    }

    public int featureCount() {
        return nFeatures;
    }

    /** Total graded samples consumed by {@link #update} so far. */
    public synchronized long samplesSeen() {
        return samplesSeen;
    }

    /** Releases the device buffers; idempotent. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (onDevice) {
            RuntimeException failure = Perceptron.release(null, params);
            failure = releaseBatchBuffers(failure);
            if (failure != null) {
                throw failure;
            }
        }
    }

    // ------------------------------------------------------------------ CPU reference path

    /** {@code Math.max(-HUBER_DELTA, Math.min(HUBER_DELTA, e))}: NaN propagates and +0 ranks above -0. */
    static float huber(float e) {
        return Math.max(-HUBER_DELTA, Math.min(HUBER_DELTA, e));
    }

    private void hostForward(float[] flat, int n, float[] dir, float[] move) {
        for (int i = 0; i < n; i++) {
            int row = i * nFeatures;
            dir[i] = Perceptron.logistic(Perceptron.affine(flat, row, hostDirWeights, hostDirBias, nFeatures));
            move[i] = Perceptron.affine(flat, row, hostMoveWeights, hostMoveBias, nFeatures);
        }
    }

    private void hostStep(float[] flat, float[] yDirection, float[] yMove, float[] sampleWeights, int n) {
        float[] dir = new float[n];
        float[] move = new float[n];
        hostForward(flat, n, dir, move);
        float[] dirSums = new float[nFeatures];
        float[] moveSums = new float[nFeatures];
        float dirBiasSum = 0.0f;
        float moveBiasSum = 0.0f;
        for (int i = 0; i < n; i++) {
            float s = sampleWeights == null ? 1.0f : sampleWeights[i];
            float ed = dir[i] - yDirection[i];
            float em = huber(move[i] - yMove[i]);
            float wd = s * ed;
            float wm = s * em;
            int row = i * nFeatures;
            for (int j = 0; j < nFeatures; j++) {
                float xi = flat[row + j];
                dirSums[j] = Math.fma(wd, xi, dirSums[j]);
                moveSums[j] = Math.fma(wm, xi, moveSums[j]);
            }
            dirBiasSum = Math.fma(s, ed, dirBiasSum);
            moveBiasSum = Math.fma(s, em, moveBiasSum);
        }
        float count = (float) n;
        for (int j = 0; j < nFeatures; j++) {
            float wd = hostDirWeights[j];
            float wm = hostMoveWeights[j];
            hostDirWeights[j] = wd - learningRate * (dirSums[j] / count + l2 * wd);
            hostMoveWeights[j] = wm - learningRate * (moveSums[j] / count + l2 * wm);
        }
        hostDirBias = hostDirBias - learningRate * (dirBiasSum / count);
        hostMoveBias = hostMoveBias - learningRate * (moveBiasSum / count);
    }

    // ------------------------------------------------------------------ device path

    private void deviceStep(float[] flat, float[] yDirection, float[] yMove, float[] sampleWeights, int n) {
        TrainGpu gpu = Perceptron.context(backend);
        ensureBatchCapacity(gpu, n);
        Perceptron.upload(backend, xDev, flat);
        Perceptron.upload(backend, yDirDev, yDirection);
        Perceptron.upload(backend, yMoveDev, yMove);
        if (sampleWeights == null) {
            gpu.fill(sampleWDev.gpuBuffer(), 0, n, 1.0f);
        } else {
            Perceptron.upload(backend, sampleWDev, sampleWeights);
        }
        launchForward(gpu, n);
        gpu.launch1d(Kernels.ML_FORECAST_GRAD, nFeatures, n, nFeatures, xDev.gpuBuffer(), yDirDev.gpuBuffer(),
                yMoveDev.gpuBuffer(), sampleWDev.gpuBuffer(), predDirDev.gpuBuffer(), predMoveDev.gpuBuffer(),
                gradDirWDev.gpuBuffer(), gradDirBDev.gpuBuffer(), gradMoveWDev.gpuBuffer(), gradMoveBDev.gpuBuffer());
        gpu.launch1d(Kernels.ML_FORECAST_UPDATE, nFeatures, nFeatures, learningRate, l2,
                dirWDev.gpuBuffer(), gradDirWDev.gpuBuffer(), dirBDev.gpuBuffer(), gradDirBDev.gpuBuffer(),
                moveWDev.gpuBuffer(), gradMoveWDev.gpuBuffer(), moveBDev.gpuBuffer(), gradMoveBDev.gpuBuffer());
    }

    private void launchForward(TrainGpu gpu, int n) {
        gpu.launch1d(Kernels.ML_FORECAST_FORWARD, n, n, nFeatures, xDev.gpuBuffer(), dirWDev.gpuBuffer(),
                dirBDev.gpuBuffer(), moveWDev.gpuBuffer(), moveBDev.gpuBuffer(), predDirDev.gpuBuffer(),
                predMoveDev.gpuBuffer());
    }

    private void ensureBatchCapacity(TrainGpu gpu, int n) {
        if (n <= batchCapacity) {
            return;
        }
        RuntimeException failure = releaseBatchBuffers(null);
        if (failure != null) {
            throw failure;
        }
        TensorStorage[] b = Perceptron.allocate(gpu, n * nFeatures, n, n, n, n, n);
        xDev = b[0];
        yDirDev = b[1];
        yMoveDev = b[2];
        sampleWDev = b[3];
        predDirDev = b[4];
        predMoveDev = b[5];
        batchCapacity = n;
    }

    private RuntimeException releaseBatchBuffers(RuntimeException failure) {
        RuntimeException result = Perceptron.release(failure, xDev, yDirDev, yMoveDev, sampleWDev, predDirDev,
                predMoveDev);
        xDev = null;
        yDirDev = null;
        yMoveDev = null;
        sampleWDev = null;
        predDirDev = null;
        predMoveDev = null;
        batchCapacity = 0;
        return result;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ForecastUnit is closed");
        }
    }
}
