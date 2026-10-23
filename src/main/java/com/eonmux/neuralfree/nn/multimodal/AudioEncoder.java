package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Audio encoder: log-mel spectrogram, two 3x3 convolutions with GELU (the second with stride 2), a linear projection
 * of each time step's channel and frequency features, pre-norm {@link EncoderBlock}s and a final layer norm. Every
 * step is a {@link TensorOps} composition, so the encoder runs on the device when the train kernels are available on
 * the waveform's backend.
 *
 * <p><b>Spectrogram.</b> There are {@code (numSamples - nFft) / hop + 1} frames (integer division); frame {@code f}
 * covers samples {@code [f*hop, f*hop + nFft)}, zero beyond the waveform's end, and frames are gathered by an index
 * built from the shapes. Each frame is multiplied by the Hann window
 * {@code (float) (0.5 * (1 - cos(2*pi*i / (nFft-1))))}, and its power spectrum {@code re^2 + im^2} over bins
 * {@code k <= nFft/2} comes from two products with the DFT matrices {@code (float) cos(2*pi*k*t/nFft)} and
 * {@code (float) sin(2*pi*k*t/nFft)}. The mel energies are the product with the triangular filter bank and the
 * result is {@code log(max(energy, 1e-10f))}, laid out as {@code [batch, 1, nMels, numFrames]}. The window, DFT
 * matrices and filter bank are computed on the host once and held as non-persistent buffers: they follow the module
 * across backends and dtypes but stay out of {@link #stateDict()}, since the configuration determines them.</p>
 *
 * <p><b>Sequence.</b> The second convolution's output {@code [batch, hiddenDim, h, w]} becomes
 * {@code [batch, w, hiddenDim * h]} (feature {@code c * h + row}), with {@code h = (nMels - 1) / 2 + 1}; the
 * projection maps {@code hiddenDim * h} features to {@code hiddenDim}.</p>
 *
 * <p><b>Embedding dropout</b>: in training mode the projected sequence passes through the registered {@link Dropout}
 * child {@value #EMBEDDING_DROPOUT} with rate {@code dropout} before the blocks. Its keep decisions are host draws, one
 * per element in row-major order, through {@code dropout}'s seeded generator; {@link #initialize} seeds it from its
 * {@code Random} after every weight when {@code dropout > 0}, so equal seeds give equal masks on every backend. In
 * evaluation mode, or with {@code dropout == 0}, the sequence passes unchanged. A training-mode dropout records
 * {@code MaskScaleGrad}, so a dropped element passes no gradient.</p>
 */
public class AudioEncoder extends Module {

    /** Registered name of the embedding-dropout child module. */
    public static final String EMBEDDING_DROPOUT = "embeddingDropout";

    private static final String HANN_WINDOW = "hannWindow";
    private static final String DFT_COS = "dftCos";
    private static final String DFT_SIN = "dftSin";
    private static final String MEL_FILTERS = "melFilters";
    private static final float LOG_FLOOR = 1e-10f;

    private final int sampleRate;
    private final int nMels;
    private final int nFft;
    private final int hopLength;
    private final int hiddenDim;
    private final int numLayers;
    private final int numHeads;
    private final String encoderType;
    private final float dropout;
    private final Dropout embeddingDropout;

    private Conv2d conv1;
    private Conv2d conv2;
    private Linear projection;
    private final List<EncoderBlock> encoderLayers;
    private LayerNorm finalNorm;

    private boolean initialized = false;

    public AudioEncoder(int sampleRate, int nMels, int nFft, int hopLength,
                        int hiddenDim, int numLayers, int numHeads, String encoderType, float dropout) {
        if (sampleRate <= 0) throw new IllegalArgumentException("sampleRate must be positive");
        if (nMels <= 0) throw new IllegalArgumentException("nMels must be positive");
        if (nFft <= 0) throw new IllegalArgumentException("nFft must be positive");
        if (hopLength <= 0) throw new IllegalArgumentException("hopLength must be positive");
        if (hiddenDim <= 0) throw new IllegalArgumentException("hiddenDim must be positive");
        if (numLayers < 0) throw new IllegalArgumentException("numLayers must be non-negative");
        if (numHeads <= 0) throw new IllegalArgumentException("numHeads must be positive");
        if (!(dropout >= 0 && dropout < 1)) throw new IllegalArgumentException("dropout must be in [0, 1): " + dropout);
        ScaledDotProductAttention.requireDivisible(hiddenDim, numHeads);

        this.sampleRate = sampleRate;
        this.nMels = nMels;
        this.nFft = nFft;
        this.hopLength = hopLength;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.encoderType = encoderType != null ? encoderType : "transformer";
        this.dropout = dropout;
        this.embeddingDropout = registerModule(EMBEDDING_DROPOUT, new Dropout(dropout));

        this.encoderLayers = new ArrayList<>();
    }

    public AudioEncoder(int sampleRate, int nMels, int hiddenDim, int numLayers, int numHeads) {
        this(sampleRate, nMels, 400, 160, hiddenDim, numLayers, numHeads, "transformer", 0.1f);
    }

    /**
     * Creates the parameters and buffers on {@code backend}, drawing every weight from {@code random} (a new one when
     * null), then, when {@code dropout > 0}, the seed of the embedding-dropout masks.
     */
    public void initialize(ComputeBackend backend, Random random) {
        if (initialized) return;

        Random r = random != null ? random : new Random();

        conv1 = new Conv2d(1, hiddenDim, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
        conv2 = new Conv2d(hiddenDim, hiddenDim, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1});
        conv1.initialize(backend, r);
        conv2.initialize(backend, r);
        registerModule("conv1", conv1);
        registerModule("conv2", conv2);

        projection = new Linear(Math.multiplyExact(hiddenDim, convHeight()), hiddenDim);
        projection.initialize(backend, r);
        registerModule("projection", projection);

        for (int i = 0; i < numLayers; i++) {
            encoderLayers.add(registerModule("layer_" + i, new EncoderBlock(backend, hiddenDim, numHeads, r)));
        }

        finalNorm = new LayerNorm(new int[]{hiddenDim});
        finalNorm.initialize(backend);
        registerModule("finalNorm", finalNorm);

        registerSpectrogramBuffers(backend);
        if (dropout > 0) {
            embeddingDropout.setSeed(r.nextLong());
        }
        initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        return encode(input);
    }

    /**
     * Encodes a {@code [batch, numSamples]} (or {@code [numSamples]}) waveform into
     * {@code [batch, numTimeSteps, hiddenDim]}. In training mode the projected sequence passes through the embedding
     * dropout before the blocks.
     *
     * @throws IllegalArgumentException when the waveform is not rank 1 or 2 or yields no frame
     *                                  ({@code (numSamples - nFft) / hopLength + 1 < 1} in integer arithmetic)
     */
    public Tensor encode(Tensor waveform) {
        if (!initialized) {
            throw new IllegalStateException("AudioEncoder not initialized. Call initialize() first.");
        }
        if (waveform == null) {
            throw new IllegalArgumentException("waveform tensor cannot be null");
        }

        Tensor sequence;
        try (Tensor mel = computeMelSpectrogram(waveform);
             Tensor c1 = conv1.forward(mel);
             Tensor a1 = TensorOps.gelu(c1);
             Tensor c2 = conv2.forward(a1);
             Tensor a2 = TensorOps.gelu(c2)) {
            sequence = toSequence(a2);
        }
        Tensor projected;
        try (Tensor s = sequence) {
            projected = projection.forward(s);
        }
        Tensor hidden = dropEmbeddings(projected);
        for (EncoderBlock layer : encoderLayers) {
            try (Tensor previous = hidden) {
                hidden = layer.forward(previous);
            }
        }
        try (Tensor last = hidden) {
            return finalNorm.forward(last);
        }
    }

    /**
     * {@code embeddings} through the embedding dropout; the result is owned by the caller and {@code embeddings} is
     * closed when the dropout made a new tensor (or failed).
     */
    private Tensor dropEmbeddings(Tensor embeddings) {
        Tensor dropped;
        try {
            dropped = embeddingDropout.forward(embeddings);
        } catch (RuntimeException e) {
            embeddings.close();
            throw e;
        }
        if (dropped != embeddings) {
            embeddings.close();
        }
        return dropped;
    }

    private Tensor computeMelSpectrogram(Tensor waveform) {
        int rank = waveform.shape().rank();
        if (rank != 1 && rank != 2) {
            throw new IllegalArgumentException("waveform must be [batch, numSamples] or [numSamples], got "
                    + waveform.shape());
        }
        int batchSize = rank == 2 ? waveform.shape().dim(0) : 1;
        int numSamples = waveform.shape().dim(rank - 1);
        int numFrames = (numSamples - nFft) / hopLength + 1;
        if (numFrames < 1) {
            throw new IllegalArgumentException("waveform has " + numSamples + " samples, too few for one frame of "
                    + "nFft = " + nFft + " with hopLength = " + hopLength);
        }

        try (Tensor samples = waveform.reshape(Shape.of(batchSize, numSamples));
             Tensor wave = padToFrame(samples, batchSize, numSamples);
             Tensor index = frameIndex(waveform.backend(), batchSize, numFrames);
             Tensor framed = TensorOps.gather(wave, 1, index);
             Tensor frames = framed.reshape(Shape.of(batchSize * numFrames, nFft));
             Tensor windowed = TensorOps.mul(frames, buffer(HANN_WINDOW));
             Tensor re = TensorOps.matmul(windowed, buffer(DFT_COS));
             Tensor im = TensorOps.matmul(windowed, buffer(DFT_SIN));
             Tensor re2 = TensorOps.mul(re, re);
             Tensor im2 = TensorOps.mul(im, im);
             Tensor power = TensorOps.add(re2, im2);
             Tensor energy = TensorOps.matmul(power, buffer(MEL_FILTERS));
             Tensor floored = TensorOps.clamp(energy, LOG_FLOOR, Float.POSITIVE_INFINITY);
             Tensor logMel = TensorOps.log(floored);
             Tensor byFrame = logMel.reshape(Shape.of(batchSize, numFrames, nMels));
             Tensor byMel = byFrame.permute(0, 2, 1)) {
            return byMel.reshape(Shape.of(batchSize, 1, nMels, numFrames));
        }
    }

    /**
     * {@code samples} followed by zeros up to {@code nFft} columns when it is shorter (a single frame, as the integer
     * frame count allows), else a view of it.
     */
    private Tensor padToFrame(Tensor samples, int batchSize, int numSamples) {
        if (numSamples >= nFft) {
            return samples.reshape(samples.shape());
        }
        try (Tensor zeros = Tensor.zeros(samples.backend(), Shape.of(batchSize, nFft - numSamples))) {
            return TensorOps.cat(Arrays.asList(samples, zeros), 1);
        }
    }

    /** INT32 {@code [batch, numFrames * nFft]} sample indices {@code f * hop + i}, one row broadcast over the batch. */
    private Tensor frameIndex(ComputeBackend backend, int batchSize, int numFrames) {
        int length = Math.multiplyExact(numFrames, nFft);
        int[] index = new int[length];
        for (int f = 0; f < numFrames; f++) {
            for (int i = 0; i < nFft; i++) {
                index[f * nFft + i] = f * hopLength + i;
            }
        }
        try (Tensor flat = Tensor.fromArray(backend, index);
             Tensor row = flat.reshape(Shape.of(1, length))) {
            return row.expand(Shape.of(batchSize, length));
        }
    }

    /** {@code [batch, channels, h, w]} to {@code [batch, w, channels * h]}. */
    private static Tensor toSequence(Tensor convolved) {
        int[] dims = convolved.shape().dims();
        try (Tensor byTime = convolved.permute(0, 3, 1, 2)) {
            return byTime.reshape(Shape.of(dims[0], dims[3], dims[1] * dims[2]));
        }
    }

    /** Height of the second convolution's output: 3x3, stride 2, padding 1 over {@code nMels} rows. */
    private int convHeight() {
        return (nMels - 1) / 2 + 1;
    }

    private void registerSpectrogramBuffers(ComputeBackend backend) {
        int bins = nFft / 2 + 1;
        float[] hann = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            hann[i] = (float) (0.5 * (1 - Math.cos(2 * Math.PI * i / (nFft - 1))));
        }
        float[] cos = new float[Math.multiplyExact(nFft, bins)];
        float[] sin = new float[cos.length];
        for (int t = 0; t < nFft; t++) {
            for (int k = 0; k < bins; k++) {
                double angle = 2.0 * Math.PI * k * t / nFft;
                cos[t * bins + k] = (float) Math.cos(angle);
                sin[t * bins + k] = (float) Math.sin(angle);
            }
        }
        float[] filters = createMelFilterBank();
        float[] filtersByBin = new float[filters.length];
        for (int m = 0; m < nMels; m++) {
            for (int k = 0; k < bins; k++) {
                filtersByBin[k * nMels + m] = filters[m * bins + k];
            }
        }
        registerBuffer(HANN_WINDOW, upload(backend, Shape.vector(nFft), hann), false);
        registerBuffer(DFT_COS, upload(backend, Shape.matrix(nFft, bins), cos), false);
        registerBuffer(DFT_SIN, upload(backend, Shape.matrix(nFft, bins), sin), false);
        registerBuffer(MEL_FILTERS, upload(backend, Shape.matrix(bins, nMels), filtersByBin), false);
    }

    private static Tensor upload(ComputeBackend backend, Shape shape, float[] data) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    /** Triangular mel filters, {@code [nMels, nFft/2 + 1]} row-major. */
    private float[] createMelFilterBank() {
        int bins = nFft / 2 + 1;
        float[] melFilterBank = new float[nMels * bins];

        float lowMel = hzToMel(0.0f);
        float highMel = hzToMel(sampleRate / 2.0f);

        float[] melPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) {
            melPoints[i] = lowMel + (highMel - lowMel) * i / (nMels + 1);
        }

        int[] freqIndices = new int[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) {
            float freq = melToHz(melPoints[i]);
            freqIndices[i] = (int) Math.floor((nFft + 1) * freq / sampleRate);
        }

        for (int m = 0; m < nMels; m++) {
            int left = freqIndices[m];
            int center = freqIndices[m + 1];
            int right = freqIndices[m + 2];

            for (int k = left; k < center && k < bins; k++) {
                if (center != left) {
                    melFilterBank[m * bins + k] = (float) (k - left) / (center - left);
                }
            }
            for (int k = center; k < right && k < bins; k++) {
                if (right != center) {
                    melFilterBank[m * bins + k] = (float) (right - k) / (right - center);
                }
            }
        }

        return melFilterBank;
    }

    private static float hzToMel(float hz) {
        return (float) (2595.0 * Math.log10(1.0 + hz / 700.0));
    }

    private static float melToHz(float mel) {
        return (float) (700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0));
    }

    public int sampleRate() { return sampleRate; }
    public int nMels() { return nMels; }
    public int nFft() { return nFft; }
    public int hopLength() { return hopLength; }
    public int hiddenDim() { return hiddenDim; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public String encoderType() { return encoderType; }
    public float dropout() { return dropout; }
    public boolean isInitialized() { return initialized; }

    @Override
    public String toString() {
        return String.format("AudioEncoder(sampleRate=%d, nMels=%d, nFft=%d, hopLength=%d, hiddenDim=%d, numLayers=%d, numHeads=%d, encoderType=%s, dropout=%.2f)",
                sampleRate, nMels, nFft, hopLength, hiddenDim, numLayers, numHeads, encoderType, dropout);
    }
}
