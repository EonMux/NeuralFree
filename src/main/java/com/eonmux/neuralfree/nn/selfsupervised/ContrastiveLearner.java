package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.autograd.CheckpointFunction;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * SimCLR-style contrastive learning with an NT-Xent loss. Every tensor computation is a {@link TensorOps}
 * composition (device when the train kernels are available, CPU reference otherwise), recorded when grad mode is on
 * and a parameter or input requires grad. The augmentation draws its drop decisions and noise from the module's
 * {@link Random} on the host and applies them on the device.
 *
 * <p>Ownership: intermediate projections are closed here; the representations and normalized projections of a
 * {@link ContrastiveOutput} belong to the caller.</p>
 */
public final class ContrastiveLearner extends Module {

    /**
     * The result of {@link #forwardContrastive}, owned by the caller. The projections and representations are outputs
     * of the recorded graph: they are valid until a loss computed from them (such as {@link #computeNTXentLoss}) is
     * back-propagated, whose backward pass closes them; take {@link Tensor#detach()} handles first of any value read
     * afterwards.
     */
    public static final class ContrastiveOutput {
        private final Tensor projection1;
        private final Tensor projection2;
        private final Tensor representation1;
        private final Tensor representation2;

        public ContrastiveOutput(Tensor projection1, Tensor projection2,
                                 Tensor representation1, Tensor representation2) {
            this.projection1 = Objects.requireNonNull(projection1);
            this.projection2 = Objects.requireNonNull(projection2);
            this.representation1 = Objects.requireNonNull(representation1);
            this.representation2 = Objects.requireNonNull(representation2);
        }

        public Tensor projection1() { return projection1; }
        public Tensor projection2() { return projection2; }
        public Tensor representation1() { return representation1; }
        public Tensor representation2() { return representation2; }
    }

    private final Module encoder;
    private final Sequential projectionHead;
    private final int embeddingDim;
    private final int projectionDim;
    private final float temperature;
    private Random random;
    private volatile boolean closed = false;

    public ContrastiveLearner(Module encoder, int embeddingDim, int projectionDim) {
        this(encoder, embeddingDim, projectionDim, 0.07f);
    }

    public ContrastiveLearner(Module encoder, int embeddingDim, int projectionDim, float temperature) {
        Objects.requireNonNull(encoder, "Encoder cannot be null");
        if (embeddingDim <= 0) {
            throw new IllegalArgumentException("Embedding dim must be positive: " + embeddingDim);
        }
        if (projectionDim <= 0) {
            throw new IllegalArgumentException("Projection dim must be positive: " + projectionDim);
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive: " + temperature);
        }

        this.encoder = registerModule("encoder", encoder);
        this.embeddingDim = embeddingDim;
        this.projectionDim = projectionDim;
        this.temperature = temperature;
        this.random = new Random();

        this.projectionHead = new Sequential();
        registerModule("projection_head", projectionHead);
    }

    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");

        int hiddenDim = (embeddingDim + projectionDim) / 2;

        Linear linear1 = new Linear(embeddingDim, hiddenDim);
        linear1.initialize(backend, random);
        projectionHead.add(linear1);

        Linear linear2 = new Linear(hiddenDim, projectionDim);
        linear2.initialize(backend, random);
        projectionHead.add(linear2);
    }

    public ContrastiveOutput forwardContrastive(Tensor view1, Tensor view2) {
        ensureOpen();
        Objects.requireNonNull(view1, "View1 cannot be null");
        Objects.requireNonNull(view2, "View2 cannot be null");

        Tensor representation1 = null;
        Tensor representation2 = null;
        Tensor projection1 = null;
        try {
            representation1 = encoder.forward(view1);
            representation2 = encoder.forward(view2);
            projection1 = project(representation1, view1, view2, representation1, representation2);
            Tensor projection2 = project(representation2, view1, view2, representation1, representation2);
            return new ContrastiveOutput(projection1, projection2, representation1, representation2);
        } catch (RuntimeException e) {
            release(projection1, view1, view2);
            release(representation1, view1, view2);
            release(representation2, view1, view2);
            throw e;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        Tensor representation = encoder.forward(input);
        Tensor output;
        try {
            output = projectionHead.forward(representation);
        } catch (RuntimeException e) {
            release(representation, input);
            throw e;
        }
        release(representation, input, output);
        return output;
    }

    /**
     * NT-Xent over the {@code 2B} samples {@code z = cat(projection1, projection2)}: with {@code s = z z^T /
     * temperature}, the loss of anchor {@code a} is {@code -log(exp(s[a, p(a)]) / sum_{j != a} exp(s[a, j]))}, where
     * {@code p(a)} is the other view of the same sample, and the result is the mean over all anchors. The diagonal is
     * excluded by adding {@code -inf} before {@code logSoftmax}, which then contributes exactly 0 to each row's sum.
     *
     * @throws IllegalArgumentException when a projection does not hold {@code [batch, projectionDim]} values with
     *                                  the batch of {@code projection1}
     */
    public Tensor computeNTXentLoss(Tensor projection1, Tensor projection2) {
        ensureOpen();
        Objects.requireNonNull(projection1, "Projection1 cannot be null");
        Objects.requireNonNull(projection2, "Projection2 cannot be null");

        int batchSize = projection1.shape().dim(0);
        int totalSamples = 2 * batchSize;
        ComputeBackend backend = projection1.backend();
        try (Tensor diagonalMask = diagonalMask(backend, totalSamples);
             Tensor positives = positiveIndex(backend, batchSize);
             Tensor z1 = rows(projection1, batchSize, "projection1");
             Tensor z2 = rows(projection2, batchSize, "projection2");
             Tensor all = TensorOps.cat(Arrays.asList(z1, z2), 0);
             Tensor allT = all.transpose();
             Tensor dots = TensorOps.matmul(all, allT);
             Tensor temp = constant(backend, temperature);
             Tensor similarities = TensorOps.div(dots, temp);
             Tensor masked = TensorOps.add(similarities, diagonalMask);
             Tensor logProbs = TensorOps.logSoftmax(masked, 1);
             Tensor positiveLogProbs = TensorOps.gather(logProbs, 1, positives);
             Tensor meanLogProb = TensorOps.mean(positiveLogProbs)) {
            return TensorOps.neg(meanLogProb);
        }
    }

    /**
     * Per element in row-major order: with probability {@code dropProb} the element becomes 0, otherwise
     * {@code x + (float) (nextGaussian() * noiseStd)}. The draws come from the module's {@link Random} element by
     * element (a Gaussian is drawn only for kept elements) through {@link CheckpointFunction#replayable}, so a
     * checkpointed recompute sees the same draws; they are uploaded once and applied on the device. The result is
     * recorded ({@code where} and {@code add}) when grad mode is on and the input requires grad: the kept elements
     * pass the gradient through, the dropped ones receive none.
     */
    public Tensor augment(Tensor input, float noiseStd, float dropProb) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");

        int n = (int) input.numElements();
        float[][] draws = CheckpointFunction.replayable(() -> drawAugmentation(n, noiseStd, dropProb));
        Shape shape = input.shape();
        ComputeBackend backend = input.backend();
        try (Tensor keepMask = floats(backend, draws[0], shape);
             Tensor noiseTensor = floats(backend, draws[1], shape);
             Tensor noisy = TensorOps.add(input, noiseTensor)) {
            return TensorOps.where(keepMask, noisy, noiseTensor);
        }
    }

    /** {@code {keep, noise}}: the per-element keep flags (1 or 0) and noise of {@link #augment}, from the Random. */
    private float[][] drawAugmentation(int n, float noiseStd, float dropProb) {
        float[] keep = new float[n];
        float[] noise = new float[n];
        for (int i = 0; i < n; i++) {
            if (!(random.nextFloat() < dropProb)) {
                keep[i] = 1.0f;
                noise[i] = (float) (random.nextGaussian() * noiseStd);
            }
        }
        return new float[][]{keep, noise};
    }

    /**
     * Projection head and L2 normalization of a representation along its last dimension
     * ({@code F.normalize(x, dim=-1)}); the head output is closed here unless it is one of {@code keep} (an empty
     * head returns its input).
     */
    private Tensor project(Tensor representation, Tensor... keep) {
        Tensor projected = projectionHead.forward(representation);
        try {
            return NormOps.l2Normalize(projected, -1);
        } finally {
            release(projected, keep);
        }
    }

    /** {@code -inf} on the diagonal of an {@code [n, n]} additive mask, 0 elsewhere. */
    private static Tensor diagonalMask(ComputeBackend backend, int n) {
        float[] mask = new float[n * n];
        for (int i = 0; i < n; i++) {
            mask[i * n + i] = Float.NEGATIVE_INFINITY;
        }
        return floats(backend, mask, Shape.of(n, n));
    }

    /** INT32 {@code [2B, 1]}: each anchor's other view, {@code a + B} for the first view, else {@code a - B}. */
    private static Tensor positiveIndex(ComputeBackend backend, int batchSize) {
        int[] index = new int[2 * batchSize];
        for (int a = 0; a < index.length; a++) {
            index[a] = a < batchSize ? a + batchSize : a - batchSize;
        }
        try (Tensor flat = Tensor.fromArray(backend, index)) {
            return flat.reshape(index.length, 1);
        }
    }

    /** A {@code [batch, projectionDim]} view of a projection. */
    private Tensor rows(Tensor projection, int batchSize, String what) {
        if (projection.numElements() != (long) batchSize * projectionDim) {
            throw new IllegalArgumentException("The " + what + " of shape " + projection.shape()
                + " does not hold [" + batchSize + ", " + projectionDim + "] values");
        }
        return projection.reshape(batchSize, projectionDim);
    }

    /** A FLOAT32 tensor of {@code shape} holding {@code values} in row-major order. */
    private static Tensor floats(ComputeBackend backend, float[] values, Shape shape) {
        try (Tensor flat = Tensor.fromArray(backend, values)) {
            return flat.reshape(shape);
        }
    }

    /** A rank-0 tensor holding {@code value}, filled on the device when the train kernels are available. */
    private static Tensor constant(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar())) {
            return TensorOps.add(zero, value);
        }
    }

    /** Closes {@code t} unless it is one of {@code keep}. */
    private static void release(Tensor t, Tensor... keep) {
        if (t == null) {
            return;
        }
        for (Tensor k : keep) {
            if (t == k) {
                return;
            }
        }
        t.close();
    }

    public Module encoder() { return encoder; }
    public Sequential projectionHead() { return projectionHead; }
    public int embeddingDim() { return embeddingDim; }
    public int projectionDim() { return projectionDim; }
    public float temperature() { return temperature; }

    public void setRandom(Random random) {
        this.random = Objects.requireNonNull(random);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ContrastiveLearner has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) return "ContrastiveLearner{closed=true}";
        return "ContrastiveLearner{embeddingDim=" + embeddingDim +
               ", projectionDim=" + projectionDim +
               ", temperature=" + temperature + "}";
    }
}
