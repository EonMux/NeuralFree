package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * CLIP-style contrastive image-text training. Every tensor computation is a {@link TensorOps} composition, so it runs
 * on the device when the train kernels are available on the inputs' backend and on the CPU reference otherwise; no
 * intermediate is read back except the similarity scalar that {@link #computeSimilarity} returns.
 *
 * <p>Temperature: with {@code learnTemperature} the temperature is learned as CLIP does, through the registered
 * rank-0 parameter {@code logit_scale = log(1 / temperature)} created by {@link #initialize}; the logits are scaled by
 * {@code exp(logit_scale)}, recorded, so the loss trains it with the encoders. Otherwise the temperature is a
 * constant and the logits are divided by it.</p>
 *
 * <p>Ownership: the encoders' outputs are temporaries of this module and are closed after use unless an encoder
 * returns its input tensor; every tensor in a {@link CLIPOutput} belongs to the caller.</p>
 */
public final class CLIPTrainer extends Module {

    /** {@code (float) Math.log(1e-10f)}: the lower clamp of the CPU loss {@code -log(max(1e-10f, p))} in log space. */
    private static final float LOG_MIN_PROB = (float) Math.log(1e-10f);
    private static final String LOGIT_SCALE = "logit_scale";
    private static final float DEFAULT_TEMPERATURE = 0.07f;

    /**
     * The result of {@link #forwardCLIP}, owned by the caller. {@link #clipLoss()} is the recorded loss to
     * back-propagate. The embeddings and logits are outputs of the same recorded graph: they are valid until the loss
     * is back-propagated, whose backward pass closes them; take {@link Tensor#detach()} handles first of any value
     * read afterwards.
     */
    public static final class CLIPOutput {
        private final Tensor imageEmbedding;
        private final Tensor textEmbedding;
        private final Tensor logitsPerImage;
        private final Tensor logitsPerText;
        private final Tensor clipLoss;

        public CLIPOutput(Tensor imageEmbedding, Tensor textEmbedding,
                          Tensor logitsPerImage, Tensor logitsPerText, Tensor clipLoss) {
            this.imageEmbedding = Objects.requireNonNull(imageEmbedding);
            this.textEmbedding = Objects.requireNonNull(textEmbedding);
            this.logitsPerImage = Objects.requireNonNull(logitsPerImage);
            this.logitsPerText = Objects.requireNonNull(logitsPerText);
            this.clipLoss = Objects.requireNonNull(clipLoss);
        }

        public Tensor imageEmbedding() { return imageEmbedding; }
        public Tensor textEmbedding() { return textEmbedding; }
        public Tensor logitsPerImage() { return logitsPerImage; }
        public Tensor logitsPerText() { return logitsPerText; }
        public Tensor clipLoss() { return clipLoss; }
    }

    private final Module imageEncoder;
    private final Module textEncoder;
    private final Linear imageProjection;
    private final Linear textProjection;
    private final int imageEmbedDim;
    private final int textEmbedDim;
    private final int sharedEmbedDim;
    /** The constant temperature, or the initial temperature of {@link #logitScale} until it exists. */
    private float temperature;
    private final boolean learnTemperature;
    private Parameter logitScale;
    private Random random;
    private volatile boolean closed = false;

    public CLIPTrainer(Module imageEncoder, Module textEncoder,
                        int imageEmbedDim, int textEmbedDim, int sharedEmbedDim) {
        this(imageEncoder, textEncoder, imageEmbedDim, textEmbedDim, sharedEmbedDim, true);
    }

    public CLIPTrainer(Module imageEncoder, Module textEncoder,
                        int imageEmbedDim, int textEmbedDim, int sharedEmbedDim,
                        boolean learnTemperature) {
        Objects.requireNonNull(imageEncoder, "Image encoder cannot be null");
        Objects.requireNonNull(textEncoder, "Text encoder cannot be null");
        if (imageEmbedDim <= 0) throw new IllegalArgumentException("Image embed dim must be positive");
        if (textEmbedDim <= 0) throw new IllegalArgumentException("Text embed dim must be positive");
        if (sharedEmbedDim <= 0) throw new IllegalArgumentException("Shared embed dim must be positive");

        this.imageEncoder = registerModule("image_encoder", imageEncoder);
        this.textEncoder = registerModule("text_encoder", textEncoder);
        this.imageEmbedDim = imageEmbedDim;
        this.textEmbedDim = textEmbedDim;
        this.sharedEmbedDim = sharedEmbedDim;
        this.learnTemperature = learnTemperature;
        this.temperature = DEFAULT_TEMPERATURE;
        this.random = new Random();

        this.imageProjection = new Linear(imageEmbedDim, sharedEmbedDim);
        this.textProjection = new Linear(textEmbedDim, sharedEmbedDim);
        registerModule("image_projection", imageProjection);
        registerModule("text_projection", textProjection);
    }

    /**
     * Initializes both projections on {@code backend} from the module's {@link Random} and, with a learned
     * temperature, registers the parameter {@code logit_scale = log(1 / temperature)} (the current temperature).
     * Records nothing.
     *
     * @throws IllegalArgumentException when the module is already initialized
     */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            imageProjection.initialize(backend, random);
            textProjection.initialize(backend, random);
            if (learnTemperature) {
                logitScale = registerParameter(LOGIT_SCALE,
                    new Parameter(Tensor.scalar(backend, logScale(temperature)), LOGIT_SCALE));
            }
        }
    }

    /**
     * Encodes both batches, L2-normalizes the projections and computes the symmetric contrastive loss.
     * {@code logitsPerImage[i, j] = <image_i, text_j> / temperature} (with a learned temperature
     * {@code <image_i, text_j> * exp(logit_scale)}) and {@code logitsPerText} is its transpose. The
     * loss is the mean over {@code i} of the average of {@code -log(max(1e-10, softmax(logitsPerImage)[i, i]))} and
     * the same term of {@code logitsPerText}, computed in log space by {@code logSoftmax} and a gather of the
     * diagonal.
     *
     * @throws IllegalArgumentException when the projections do not hold {@code [batch, sharedEmbedDim]} values, with
     *                                  the batch taken from the image input, for both modalities
     */
    public CLIPOutput forwardCLIP(Tensor imageInput, Tensor textInput) {
        ensureOpen();
        Objects.requireNonNull(imageInput, "Image input cannot be null");
        Objects.requireNonNull(textInput, "Text input cannot be null");

        int batchSize = imageInput.shape().dim(0);
        Tensor imageEmbedding = null;
        Tensor textEmbedding = null;
        Tensor logitsPerImage = null;
        Tensor logitsPerText = null;
        try {
            imageEmbedding = embed(imageEncoder, imageProjection, imageInput);
            textEmbedding = embed(textEncoder, textProjection, textInput);
            try (Tensor img = rows(imageEmbedding, batchSize, "image");
                 Tensor txt = rows(textEmbedding, batchSize, "text")) {
                logitsPerImage = scaledSimilarities(img, txt);
                logitsPerText = scaledSimilarities(txt, img);
            }
            Tensor loss = symmetricLoss(logitsPerImage, logitsPerText, batchSize);
            return new CLIPOutput(imageEmbedding, textEmbedding, logitsPerImage, logitsPerText, loss);
        } catch (RuntimeException e) {
            closeAll(imageEmbedding, textEmbedding, logitsPerImage, logitsPerText);
            throw e;
        }
    }

    public Tensor encodeImage(Tensor imageInput) {
        ensureOpen();
        Objects.requireNonNull(imageInput, "Image input cannot be null");
        return embed(imageEncoder, imageProjection, imageInput);
    }

    public Tensor encodeText(Tensor textInput) {
        ensureOpen();
        Objects.requireNonNull(textInput, "Text input cannot be null");
        return embed(textEncoder, textProjection, textInput);
    }

    /**
     * {@code <imageEmbedding, textEmbedding> / temperature} over all elements in row-major order, scaled as the
     * logits of {@link #forwardCLIP} are. The dot product is computed on the embeddings' backend under
     * {@link ComputationGraph#noGrad()} (an inspection API, nothing is recorded); the returned value is the one host
     * read (the API returns a Java float).
     *
     * @throws IllegalArgumentException when the element counts differ or the embeddings live on different backends
     */
    public float computeSimilarity(Tensor imageEmbedding, Tensor textEmbedding) {
        ensureOpen();
        Objects.requireNonNull(imageEmbedding, "Image embedding cannot be null");
        Objects.requireNonNull(textEmbedding, "Text embedding cannot be null");
        long n = imageEmbedding.numElements();
        if (n != textEmbedding.numElements()) {
            throw new IllegalArgumentException("Embedding dimensions must match");
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor a = imageEmbedding.reshape((int) n);
             Tensor b = textEmbedding.reshape((int) n);
             Tensor products = TensorOps.mul(a, b);
             Tensor dot = TensorOps.sum(products);
             Tensor similarity = scaleLogits(dot)) {
            return similarity.toFloatArray()[0];
        }
    }

    /**
     * Encoder, projection and L2 normalization of each embedding along its last dimension
     * ({@code F.normalize(x, dim=-1)}); the encoder output and the projection are closed here.
     */
    private static Tensor embed(Module encoder, Linear projection, Tensor input) {
        Tensor features = encoder.forward(input);
        Tensor projected;
        try {
            projected = projection.forward(features);
        } finally {
            release(features, input);
        }
        try {
            return NormOps.l2Normalize(projected, -1);
        } finally {
            release(projected, input);
        }
    }

    /** {@code a @ b^T} scaled by {@link #scaleLogits}, for row-major {@code [batch, dim]} operands. */
    private Tensor scaledSimilarities(Tensor a, Tensor b) {
        try (Tensor bt = b.transpose();
             Tensor dots = TensorOps.matmul(a, bt)) {
            return scaleLogits(dots);
        }
    }

    /**
     * {@code dots * exp(logit_scale)} (recorded, so the logit scale receives a gradient) with a learned temperature,
     * else {@code dots / temperature}.
     */
    private Tensor scaleLogits(Tensor dots) {
        if (logitScale != null) {
            try (Tensor scale = TensorOps.exp(logitScale.tensor())) {
                return TensorOps.mul(dots, scale);
            }
        }
        try (Tensor temp = constant(dots.backend(), temperature)) {
            return TensorOps.div(dots, temp);
        }
    }

    /**
     * {@code mean_i((lossImage_i + lossText_i) / 2)} with {@code loss_i = -max(logSoftmax(logits)[i, i],
     * log(1e-10f))}; halving is the exact multiplication by 0.5.
     */
    private static Tensor symmetricLoss(Tensor logitsPerImage, Tensor logitsPerText, int batchSize) {
        int[] diagonal = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            diagonal[i] = i;
        }
        try (Tensor index = intColumn(logitsPerImage.backend(), diagonal);
             Tensor imageLoss = diagonalLoss(logitsPerImage, index);
             Tensor textLoss = diagonalLoss(logitsPerText, index);
             Tensor pairSum = TensorOps.add(imageLoss, textLoss);
             Tensor pairMean = TensorOps.mul(pairSum, 0.5f)) {
            return TensorOps.mean(pairMean);
        }
    }

    private static Tensor diagonalLoss(Tensor logits, Tensor index) {
        try (Tensor logProbs = TensorOps.logSoftmax(logits, 1);
             Tensor picked = TensorOps.gather(logProbs, 1, index);
             Tensor clamped = TensorOps.clamp(picked, LOG_MIN_PROB, Float.POSITIVE_INFINITY)) {
            return TensorOps.neg(clamped);
        }
    }

    /** A {@code [batch, sharedEmbedDim]} view of an embedding. */
    private Tensor rows(Tensor embedding, int batchSize, String what) {
        if (embedding.numElements() != (long) batchSize * sharedEmbedDim) {
            throw new IllegalArgumentException("The " + what + " embedding of shape " + embedding.shape()
                + " does not hold [" + batchSize + ", " + sharedEmbedDim + "] values");
        }
        return embedding.reshape(batchSize, sharedEmbedDim);
    }

    /** {@code log(1 / temperature)}, the logit scale of {@code temperature}. */
    private static float logScale(float temperature) {
        return (float) Math.log(1.0 / temperature);
    }

    /** A rank-0 tensor holding {@code value}, filled on the device when the train kernels are available. */
    private static Tensor constant(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar())) {
            return TensorOps.add(zero, value);
        }
    }

    /** An INT32 {@code [n, 1]} index tensor. */
    private static Tensor intColumn(ComputeBackend backend, int[] values) {
        try (Tensor flat = Tensor.fromArray(backend, values)) {
            return flat.reshape(values.length, 1);
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

    private static void closeAll(Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t != null) {
                t.close();
            }
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        return encodeImage(input);
    }

    public Module imageEncoder() { return imageEncoder; }
    public Module textEncoder() { return textEncoder; }
    public Linear imageProjection() { return imageProjection; }
    public Linear textProjection() { return textProjection; }
    public int imageEmbedDim() { return imageEmbedDim; }
    public int textEmbedDim() { return textEmbedDim; }
    public int sharedEmbedDim() { return sharedEmbedDim; }

    /**
     * The temperature: {@code exp(-logit_scale)} read back from the learned parameter once it
     * exists, else the constant (or initial) temperature.
     */
    public float temperature() {
        if (logitScale == null) {
            return temperature;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return (float) Math.exp(-logitScale.tensor().toFloatArray()[0]);
        }
    }

    /**
     * Sets the temperature: assigns {@code log(1 / temperature)} to the learned {@code logit_scale} once it exists
     * (an assignment that records nothing), else sets the constant (or initial) temperature.
     *
     * @throws IllegalArgumentException when {@code temperature} is not positive
     */
    public void setTemperature(float temperature) {
        if (temperature <= 0) throw new IllegalArgumentException("Temperature must be positive");
        if (logitScale == null) {
            this.temperature = temperature;
            return;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor value = Tensor.scalar(logitScale.backend(), logScale(temperature));
             Tensor typed = value.to(logitScale.dtype())) {
            logitScale.copyFrom(typed);
        }
    }

    /** Whether the temperature is learned through the {@code logit_scale} parameter. */
    public boolean learnsTemperature() {
        return learnTemperature;
    }

    public void setRandom(Random random) {
        this.random = Objects.requireNonNull(random);
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("CLIPTrainer has been closed");
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
        if (closed) return "CLIPTrainer{closed=true}";
        return "CLIPTrainer{imageEmbedDim=" + imageEmbedDim +
               ", textEmbedDim=" + textEmbedDim +
               ", sharedEmbedDim=" + sharedEmbedDim +
               ", temperature=" + temperature() + "}";
    }
}
