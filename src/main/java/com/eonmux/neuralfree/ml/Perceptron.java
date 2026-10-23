package com.eonmux.neuralfree.ml;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

/**
 * Online perceptron: a single logistic unit trained by streaming mini-batch SGD steps.
 *
 * <p>Where {@link LogisticRegression} is a one-shot batch estimator (allocate, {@code fit}, discard), this class is
 * built for continuous feedback loops: call {@link #update} whenever new labelled outcomes arrive and
 * {@link #predict} at decision time. Deterministic zero initialization (a single logistic unit has no symmetry to
 * break), optional L2 weight decay, thread-safe (every method is serialized on the instance).</p>
 *
 * <p><b>Execution.</b> When the train kernel library is available on the backend at construction
 * ({@link TrainGpu#forBackend}), the parameters live in device buffers for the lifetime of the model and each step
 * runs as the ML_Perceptron_Forward, ML_Perceptron_Grad and ML_Perceptron_Update kernels (1-D launches in work-groups
 * of {@link TrainGpu#LINEAR_LOCAL}) on the backend's compute queue, with no host synchronization; parameters leave the
 * device only through {@link #predict}, {@link #weights()} and {@link #bias()}, and {@link #setParameters} uploads
 * them. Batch buffers grow on demand and are reused, so steady-state operation allocates nothing per call. Otherwise
 * the same math runs on the host (the CPU reference path) and the backend is not used.</p>
 *
 * <p><b>Math</b> (both paths; {@code kernels/train/ml/linear.cl}). For a sample x: z = b + &Sigma;<sub>j</sub>
 * w<sub>j</sub> x<sub>j</sub>, accumulated in ascending j with fused multiply-adds starting at the bias, and p is the
 * stable logistic {@link #logistic}. For a batch of n samples with labels y and importance weights s:
 * grad_w[j] = (&Sigma;<sub>i</sub> (s<sub>i</sub> (p<sub>i</sub> &minus; y<sub>i</sub>)) x<sub>ij</sub>) / n and
 * grad_b = (&Sigma;<sub>i</sub> s<sub>i</sub> (p<sub>i</sub> &minus; y<sub>i</sub>)) / n, fused in ascending i; then
 * w &larr; w &minus; lr (grad_w + l2 w) and b &larr; b &minus; lr grad_b (no decay on the bias). The fused sums agree
 * bit for bit between the paths on normal values; the logistic follows the transcendental tolerance class and the
 * sums the reduction class of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 */
public final class Perceptron implements AutoCloseable {

    private final ComputeBackend backend;
    /** The parameters live on the device (the train kernels were available at construction); else the CPU path. */
    private final boolean onDevice;
    private final int nFeatures;
    private final float learningRate;
    private final float l2;

    private final TensorStorage weightsDev;
    private final TensorStorage biasDev;
    private final TensorStorage gradWDev;
    private final TensorStorage gradBDev;
    private TensorStorage xDev;
    private TensorStorage yDev;
    private TensorStorage sampleWDev;
    private TensorStorage predDev;
    private int batchCapacity;

    private final float[] hostWeights;
    private float hostBias;

    private long samplesSeen;
    private boolean closed;

    public Perceptron(ComputeBackend backend, int nFeatures, float learningRate, float l2) {
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
            this.weightsDev = null;
            this.biasDev = null;
            this.gradWDev = null;
            this.gradBDev = null;
            this.hostWeights = new float[nFeatures];
            return;
        }
        TensorStorage[] p = allocate(gpu, nFeatures, 1, nFeatures, 1);
        this.weightsDev = p[0];
        this.biasDev = p[1];
        this.gradWDev = p[2];
        this.gradBDev = p[3];
        this.hostWeights = null;
        try {
            gpu.fill(weightsDev.gpuBuffer(), 0, nFeatures, 0.0f);
            gpu.fill(biasDev.gpuBuffer(), 0, 1, 0.0f);
        } catch (RuntimeException e) {
            throw release(e, p);
        }
    }

    public Perceptron(ComputeBackend backend, int nFeatures) {
        this(backend, nFeatures, 0.05f, 0.0f);
    }

    /** Probability estimates for a batch. */
    public synchronized float[] predict(float[][] x) {
        ensureOpen();
        int n = requireBatch(x, nFeatures);
        if (n == 0) {
            return new float[0];
        }
        float[] flat = flatten(x, nFeatures);
        if (!onDevice) {
            float[] predictions = new float[n];
            for (int i = 0; i < n; i++) {
                predictions[i] = logistic(affine(flat, i * nFeatures, hostWeights, hostBias, nFeatures));
            }
            return predictions;
        }
        TrainGpu gpu = context(backend);
        ensureBatchCapacity(gpu, n);
        upload(backend, xDev, flat);
        launchForward(gpu, n);
        return download(backend, predDev, n);
    }

    /** Probability estimate for a single sample. */
    public float predictOne(float[] x) {
        return predict(new float[][]{x})[0];
    }

    /**
     * One streaming SGD step on a labelled mini-batch ({@code y} in {0,1}): forward, mean logistic-loss gradient and
     * in-place parameter update; nothing is read back to the host. Every sample contributes at importance 1; see
     * {@link #update(float[][], float[], float[])} to weight samples unevenly.
     */
    public void update(float[][] x, float[] y) {
        update(x, y, null);
    }

    /**
     * {@link #update(float[][], float[])} with per-sample importance weights: each sample's gradient contribution is
     * scaled by its weight, so a weight-4 outcome pulls the parameters four times as hard as a weight-1 one (weight 1
     * is an ordinary sample; {@code null} means all ones). Weights must be finite and non-negative.
     */
    public synchronized void update(float[][] x, float[] y, float[] sampleWeights) {
        ensureOpen();
        int n = requireBatch(x, nFeatures);
        if (y == null || y.length != n) {
            throw new IllegalArgumentException("y must match the batch size");
        }
        requireSampleWeights(sampleWeights, n);
        if (n == 0) {
            return;
        }
        float[] flat = flatten(x, nFeatures);
        if (!onDevice) {
            hostStep(flat, y, sampleWeights, n);
        } else {
            deviceStep(flat, y, sampleWeights, n);
        }
        samplesSeen += n;
    }

    /** Repeated {@link #update} passes over one dataset (small-data helper). */
    public void fit(float[][] x, float[] y, int epochs) {
        for (int e = 0; e < epochs; e++) {
            update(x, y);
        }
    }

    /** Current weights (read back from the device on the GPU path). */
    public synchronized float[] weights() {
        ensureOpen();
        return onDevice ? download(backend, weightsDev, nFeatures) : hostWeights.clone();
    }

    /** Current bias (read back from the device on the GPU path). */
    public synchronized float bias() {
        ensureOpen();
        return onDevice ? download(backend, biasDev, 1)[0] : hostBias;
    }

    /** Restores persisted parameters (uploaded to the device on the GPU path). */
    public synchronized void setParameters(float[] weights, float bias) {
        ensureOpen();
        if (weights == null || weights.length != nFeatures) {
            throw new IllegalArgumentException("weights must have length " + nFeatures);
        }
        if (!onDevice) {
            System.arraycopy(weights, 0, hostWeights, 0, nFeatures);
            hostBias = bias;
            return;
        }
        upload(backend, weightsDev, weights);
        upload(backend, biasDev, new float[]{bias});
    }

    public int featureCount() {
        return nFeatures;
    }

    /** Total labelled samples consumed by {@link #update} so far. */
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
            RuntimeException failure = release(null, weightsDev, biasDev, gradWDev, gradBDev);
            failure = releaseBatchBuffers(failure);
            if (failure != null) {
                throw failure;
            }
        }
    }

    // ------------------------------------------------------------------ CPU reference path

    private void hostStep(float[] flat, float[] y, float[] sampleWeights, int n) {
        float[] pred = new float[n];
        for (int i = 0; i < n; i++) {
            pred[i] = logistic(affine(flat, i * nFeatures, hostWeights, hostBias, nFeatures));
        }
        float[] sums = new float[nFeatures];
        float biasSum = 0.0f;
        for (int i = 0; i < n; i++) {
            float s = sampleWeights == null ? 1.0f : sampleWeights[i];
            float e = pred[i] - y[i];
            float weighted = s * e;
            int row = i * nFeatures;
            for (int j = 0; j < nFeatures; j++) {
                sums[j] = Math.fma(weighted, flat[row + j], sums[j]);
            }
            biasSum = Math.fma(s, e, biasSum);
        }
        float count = (float) n;
        for (int j = 0; j < nFeatures; j++) {
            float w = hostWeights[j];
            hostWeights[j] = w - learningRate * (sums[j] / count + l2 * w);
        }
        hostBias = hostBias - learningRate * (biasSum / count);
    }

    // ------------------------------------------------------------------ device path

    private void deviceStep(float[] flat, float[] y, float[] sampleWeights, int n) {
        TrainGpu gpu = context(backend);
        ensureBatchCapacity(gpu, n);
        upload(backend, xDev, flat);
        upload(backend, yDev, y);
        if (sampleWeights == null) {
            gpu.fill(sampleWDev.gpuBuffer(), 0, n, 1.0f);
        } else {
            upload(backend, sampleWDev, sampleWeights);
        }
        launchForward(gpu, n);
        gpu.launch1d(Kernels.ML_PERCEPTRON_GRAD, nFeatures, n, nFeatures, xDev.gpuBuffer(), yDev.gpuBuffer(),
                sampleWDev.gpuBuffer(), predDev.gpuBuffer(), gradWDev.gpuBuffer(), gradBDev.gpuBuffer());
        gpu.launch1d(Kernels.ML_PERCEPTRON_UPDATE, nFeatures, nFeatures, learningRate, l2, weightsDev.gpuBuffer(),
                gradWDev.gpuBuffer(), biasDev.gpuBuffer(), gradBDev.gpuBuffer());
    }

    private void launchForward(TrainGpu gpu, int n) {
        gpu.launch1d(Kernels.ML_PERCEPTRON_FORWARD, n, n, nFeatures, xDev.gpuBuffer(), weightsDev.gpuBuffer(),
                biasDev.gpuBuffer(), predDev.gpuBuffer());
    }

    private void ensureBatchCapacity(TrainGpu gpu, int n) {
        if (n <= batchCapacity) {
            return;
        }
        RuntimeException failure = releaseBatchBuffers(null);
        if (failure != null) {
            throw failure;
        }
        TensorStorage[] b = allocate(gpu, n * nFeatures, n, n, n);
        xDev = b[0];
        yDev = b[1];
        sampleWDev = b[2];
        predDev = b[3];
        batchCapacity = n;
    }

    private RuntimeException releaseBatchBuffers(RuntimeException failure) {
        RuntimeException result = release(failure, xDev, yDev, sampleWDev, predDev);
        xDev = null;
        yDev = null;
        sampleWDev = null;
        predDev = null;
        batchCapacity = 0;
        return result;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Perceptron is closed");
        }
    }

    // ------------------------------------------------------------------ unit math and buffers shared with
    // ForecastUnit and LogisticRegression

    /**
     * The stable logistic of {@code kernels/train/ml/linear.cl}: {@code z >= 0 ? 1 / (1 + exp(-z)) : exp(z) / (1 +
     * exp(z))} with exp rounded to float; NaN takes the second branch and stays NaN.
     */
    static float logistic(float z) {
        if (z >= 0.0f) {
            return 1.0f / (1.0f + (float) Math.exp(-z));
        }
        float e = (float) Math.exp(z);
        return e / (1.0f + e);
    }

    /** {@code bias + sum_j w[j] * x[row + j]}, fused multiply-adds in ascending j, as the forward kernels do. */
    static float affine(float[] x, int row, float[] w, float bias, int nFeatures) {
        float z = bias;
        for (int j = 0; j < nFeatures; j++) {
            z = Math.fma(w[j], x[row + j], z);
        }
        return z;
    }

    /**
     * The sample count of a batch after checking that it is non-null, that every row is non-null with
     * {@code nFeatures} values, and that the batch holds at most {@code Integer.MAX_VALUE} values (the 32-bit kernel
     * ABI and a Java array).
     */
    static int requireBatch(float[][] x, int nFeatures) {
        if (x == null) {
            throw new IllegalArgumentException("x must not be null");
        }
        for (float[] row : x) {
            if (row == null || row.length != nFeatures) {
                throw new IllegalArgumentException("every sample must have " + nFeatures + " features");
            }
        }
        if ((long) x.length * nFeatures > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("a batch of " + x.length + " samples x " + nFeatures
                    + " features exceeds " + Integer.MAX_VALUE + " values");
        }
        return x.length;
    }

    /** Checks optional importance weights: {@code null}, or {@code n} finite non-negative values. */
    static void requireSampleWeights(float[] sampleWeights, int n) {
        if (sampleWeights == null) {
            return;
        }
        if (sampleWeights.length != n) {
            throw new IllegalArgumentException("sampleWeights must match the batch size");
        }
        for (float v : sampleWeights) {
            if (!(v >= 0) || Float.isInfinite(v)) {
                throw new IllegalArgumentException("sampleWeights must be finite and non-negative");
            }
        }
    }

    /** The rows of a batch checked by {@link #requireBatch}, row-major. */
    static float[] flatten(float[][] x, int nFeatures) {
        float[] flat = new float[x.length * nFeatures];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, flat, i * nFeatures, nFeatures);
        }
        return flat;
    }

    /**
     * The train context of a backend whose model keeps its parameters on the device. It is looked up on every use,
     * because {@link TrainGpu#release} (and the dispatchers' {@code clearCache}) closes a context while the model's
     * storages, which belong to the backend, stay valid; the next lookup reopens it.
     *
     * @throws IllegalStateException when the train kernels are no longer available on the backend
     */
    static TrainGpu context(ComputeBackend backend) {
        TrainGpu gpu = TrainGpu.forBackend(backend);
        if (gpu == null) {
            throw new IllegalStateException("the train kernels are no longer available on " + backend.backendType()
                    + " (" + TrainGpu.availability(backend).reason()
                    + "), but the model's parameters are on the device");
        }
        return gpu;
    }

    /** FLOAT32 device storages of the given element counts; on failure the ones already made are released. */
    static TensorStorage[] allocate(TrainGpu gpu, int... lengths) {
        TensorStorage[] made = new TensorStorage[lengths.length];
        try {
            for (int i = 0; i < lengths.length; i++) {
                made[i] = gpu.allocate(lengths[i], DType.FLOAT32);
            }
        } catch (RuntimeException e) {
            throw release(e, made);
        }
        return made;
    }

    /** Writes {@code values} to the start of {@code dst} on the transfer queue, after the queued device work. */
    static void upload(ComputeBackend backend, TensorStorage dst, float[] values) {
        GpuEvent done = dst.copyFromHostFloat(values, backend.transferQueue());
        if (done != null) {
            done.close();
        }
    }

    /** The first {@code count} floats of {@code src}, read after the train work already enqueued completes. */
    static float[] download(ComputeBackend backend, TensorStorage src, int count) {
        return src.copyToHostFloat(0, count, backend.transferQueue());
    }

    /**
     * Closes every non-null storage, attaching a close failure to {@code failure} (or returning the first one when
     * {@code failure} is null).
     */
    static RuntimeException release(RuntimeException failure, TensorStorage... storages) {
        RuntimeException result = failure;
        for (TensorStorage s : storages) {
            if (s == null) {
                continue;
            }
            try {
                s.close();
            } catch (RuntimeException e) {
                if (result == null) {
                    result = e;
                } else {
                    result.addSuppressed(e);
                }
            }
        }
        return result;
    }
}
