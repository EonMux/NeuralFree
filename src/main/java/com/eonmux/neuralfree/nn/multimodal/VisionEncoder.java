package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Vision transformer encoder: non-overlapping patches, a linear patch embedding, an optional class token, learned
 * positions, pre-norm {@link EncoderBlock}s and layer norms before and after them. Every step is a {@link TensorOps}
 * composition, so the encoder runs on the device when the train kernels are available on the image's backend.
 *
 * <p>Patchify is a reshape of {@code [B, C, H, W]} to {@code [B, C, nH, p, nW, p]}, a permute to
 * {@code [B, nH, nW, C, p, p]} and one contiguous copy to {@code [B, nH*nW, C*p*p]}: patch {@code ph * nW + pw}
 * holds its pixels in channel, row, column order. Rows and columns past the last whole patch are ignored. The patch
 * embedding is {@code patches @ patchEmbed} with {@code patchEmbed} of shape {@code [C*p*p, hiddenDim]} (no
 * bias).</p>
 *
 * <p><b>Embedding dropout</b> (the {@code pos_drop} of timm's ViT and the encoder dropout of torchvision's
 * {@code VisionTransformer}): in training mode the tokens, after the class token and the positions are added and
 * before the pre-norm and the blocks, pass through the registered {@link Dropout} child {@value #EMBEDDING_DROPOUT}
 * with rate {@code dropout}. Its keep decisions are host draws, one per element in row-major order, through
 * {@code CheckpointFunction.replayable}; {@link #initialize} seeds them from its {@code Random} after every weight
 * when {@code dropout > 0}, so equal seeds give equal masks on every backend. In evaluation mode, or with
 * {@code dropout == 0}, the tokens pass unchanged and the encoding is identical to that of an encoder without
 * dropout.</p>
 *
 * <p><b>Gradients.</b> The class token enters as a recorded reshape and {@code expand} of {@code clsToken} over the
 * batch, concatenated before the patches, and the positions as a recorded {@code slice} of {@code posEmbed}, so the
 * batch-summed gradients reach both parameters; the patchify views and the patch embedding record as well. A
 * training-mode dropout records {@code MaskScaleGrad}, so a dropped element passes no gradient.</p>
 */
public class VisionEncoder extends Module {

    /** Registered name of the embedding-dropout child module. */
    public static final String EMBEDDING_DROPOUT = "embeddingDropout";

    private final int imageSize;
    private final int patchSize;
    private final int inChannels;
    private final int hiddenDim;
    private final int numLayers;
    private final int numHeads;
    private final String backboneType;
    private final boolean useCLS;
    private final float dropout;
    private final Dropout embeddingDropout;

    private Parameter patchEmbed;
    private Parameter clsToken;
    private Parameter posEmbed;
    private LayerNorm preNorm;
    private final List<EncoderBlock> transformerBlocks;
    private LayerNorm postNorm;

    private boolean initialized = false;

    public VisionEncoder(int imageSize, int patchSize, int inChannels, int hiddenDim,
                         int numLayers, int numHeads, String backboneType, boolean useCLS, float dropout) {
        if (imageSize <= 0) throw new IllegalArgumentException("imageSize must be positive");
        if (patchSize <= 0) throw new IllegalArgumentException("patchSize must be positive");
        if (imageSize % patchSize != 0) throw new IllegalArgumentException("imageSize must be divisible by patchSize");
        if (inChannels <= 0) throw new IllegalArgumentException("inChannels must be positive");
        if (hiddenDim <= 0) throw new IllegalArgumentException("hiddenDim must be positive");
        if (numLayers < 0) throw new IllegalArgumentException("numLayers must be non-negative");
        if (numHeads <= 0) throw new IllegalArgumentException("numHeads must be positive");
        if (!(dropout >= 0 && dropout < 1)) throw new IllegalArgumentException("dropout must be in [0, 1): " + dropout);
        ScaledDotProductAttention.requireDivisible(hiddenDim, numHeads);

        this.imageSize = imageSize;
        this.patchSize = patchSize;
        this.inChannels = inChannels;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.backboneType = backboneType != null ? backboneType : "transformer";
        this.useCLS = useCLS;
        this.dropout = dropout;
        this.embeddingDropout = registerModule(EMBEDDING_DROPOUT, new Dropout(dropout));

        this.transformerBlocks = new ArrayList<>();
    }

    public VisionEncoder(int imageSize, int patchSize, int inChannels, int hiddenDim, int numLayers, int numHeads) {
        this(imageSize, patchSize, inChannels, hiddenDim, numLayers, numHeads, "transformer", true, 0.1f);
    }

    /**
     * Creates the parameters on {@code backend}, drawing every weight from {@code random} (a new one when null), then,
     * when {@code dropout > 0}, the seed of the embedding-dropout masks.
     */
    public void initialize(ComputeBackend backend, Random random) {
        if (initialized) return;

        Random r = random != null ? random : new Random();
        int numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        int seqLen = useCLS ? numPatches + 1 : numPatches;
        int patchDim = patchSize * patchSize * inChannels;

        patchEmbed = registerParameter("patchEmbed", randomParameter(backend, Shape.of(patchDim, hiddenDim), r));
        if (useCLS) {
            clsToken = registerParameter("clsToken", randomParameter(backend, Shape.of(1, hiddenDim), r));
        }
        posEmbed = registerParameter("posEmbed", randomParameter(backend, Shape.of(seqLen, hiddenDim), r));

        preNorm = new LayerNorm(new int[]{hiddenDim});
        preNorm.initialize(backend);
        registerModule("preNorm", preNorm);

        for (int i = 0; i < numLayers; i++) {
            transformerBlocks.add(registerModule("block_" + i, new EncoderBlock(backend, hiddenDim, numHeads, r)));
        }

        postNorm = new LayerNorm(new int[]{hiddenDim});
        postNorm.initialize(backend);
        registerModule("postNorm", postNorm);

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
     * Encodes a {@code [batch, channels, height, width]} image into {@code [batch, seqLen, hiddenDim]}, where
     * {@code seqLen} is the number of whole patches, plus one with the class token. In training mode the embedded
     * tokens pass through the embedding dropout before the pre-norm.
     *
     * @throws IllegalArgumentException when the image is not rank 4, is smaller than a patch, or has more patches
     *                                  than the learned positions cover
     */
    public Tensor encode(Tensor image) {
        if (!initialized) {
            throw new IllegalStateException("VisionEncoder not initialized. Call initialize() first.");
        }
        if (image == null) {
            throw new IllegalArgumentException("image tensor cannot be null");
        }

        Tensor tokens = projectPatches(image);
        if (useCLS) {
            try (Tensor patchTokens = tokens) {
                tokens = prependClsToken(patchTokens);
            }
        }
        Tensor hidden;
        try (Tensor t = tokens;
             Tensor positioned = addPositionalEmbedding(t)) {
            Tensor embedded = embeddingDropout.forward(positioned);
            try {
                hidden = preNorm.forward(embedded);
            } finally {
                if (embedded != positioned) {
                    embedded.close();
                }
            }
        }
        for (EncoderBlock block : transformerBlocks) {
            try (Tensor previous = hidden) {
                hidden = block.forward(previous);
            }
        }
        try (Tensor last = hidden) {
            return postNorm.forward(last);
        }
    }

    private Tensor projectPatches(Tensor image) {
        try (Tensor patches = extractPatches(image)) {
            return TensorOps.matmul(patches, patchEmbed.tensor());
        }
    }

    private Tensor extractPatches(Tensor image) {
        if (image.shape().rank() != 4) {
            throw new IllegalArgumentException("image must be [batch, channels, height, width], got " + image.shape());
        }
        int[] dims = image.shape().dims();
        int batchSize = dims[0];
        int channels = dims[1];
        int numPatchesH = dims[2] / patchSize;
        int numPatchesW = dims[3] / patchSize;
        if (numPatchesH == 0 || numPatchesW == 0) {
            throw new IllegalArgumentException("image " + image.shape() + " is smaller than one " + patchSize + "x"
                    + patchSize + " patch");
        }

        try (Tensor rows = image.slice(2, 0, (long) numPatchesH * patchSize);
             Tensor cropped = rows.slice(3, 0, (long) numPatchesW * patchSize);
             Tensor grid = cropped.reshape(Shape.of(batchSize, channels, numPatchesH, patchSize, numPatchesW,
                     patchSize));
             Tensor ordered = grid.permute(0, 2, 4, 1, 3, 5)) {
            return ordered.reshape(Shape.of(batchSize, numPatchesH * numPatchesW,
                    channels * patchSize * patchSize));
        }
    }

    private Tensor prependClsToken(Tensor patches) {
        int batchSize = patches.shape().dim(0);
        try (Tensor cls = clsToken.tensor().reshape(Shape.of(1, 1, hiddenDim));
             Tensor clsRows = cls.expand(Shape.of(batchSize, 1, hiddenDim))) {
            return TensorOps.cat(Arrays.asList(clsRows, patches), 1);
        }
    }

    private Tensor addPositionalEmbedding(Tensor x) {
        int seqLen = x.shape().dim(1);
        Tensor positions = posEmbed.tensor();
        if (seqLen > positions.shape().dim(0)) {
            throw new IllegalArgumentException("image yields " + seqLen + " tokens but the positional embedding "
                    + "covers " + positions.shape().dim(0) + " (imageSize " + imageSize + ")");
        }
        try (Tensor rows = positions.slice(0, 0, seqLen)) {
            return TensorOps.add(x, rows);
        }
    }

    /** A trainable parameter that takes over a standard-normal draw seeded from {@code random}. */
    private static Parameter randomParameter(ComputeBackend backend, Shape shape, Random random) {
        return new Parameter(Tensor.randn(backend, shape, random.nextLong()));
    }

    public int imageSize() { return imageSize; }
    public int patchSize() { return patchSize; }
    public int inChannels() { return inChannels; }
    public int hiddenDim() { return hiddenDim; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public String backboneType() { return backboneType; }
    public boolean useCLS() { return useCLS; }
    public float dropout() { return dropout; }
    public boolean isInitialized() { return initialized; }

    @Override
    public String toString() {
        return String.format("VisionEncoder(imageSize=%d, patchSize=%d, inChannels=%d, hiddenDim=%d, numLayers=%d, numHeads=%d, backboneType=%s, useCLS=%s, dropout=%.2f)",
                imageSize, patchSize, inChannels, hiddenDim, numLayers, numHeads, backboneType, useCLS, dropout);
    }
}
