package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.autograd.CheckpointFunction;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Masked autoencoder over image patches. Patchify and unpatchify are view permutations made contiguous, patch
 * selection is a row gather by index, the reconstruction scatters the decoded rows over the patches, and the masked
 * loss is a {@link TensorOps} composition; all of it runs on the device when the train kernels are available and on
 * the CPU reference otherwise. The mask is drawn on the host from the module's {@link Random} and uploaded as row
 * indices; inside a forward pass it is drawn through {@link CheckpointFunction#replayable}, so a
 * checkpointed recompute sees the same mask. Patch selection, encoding, decoding and the masked loss are recorded
 * when grad mode is on and a parameter or the image requires grad.
 *
 * <p>Layouts: images are {@code [batch, channels, imageHeight, imageWidth]} row-major; patches are
 * {@code [batch, numPatches, channels * patchSize * patchSize]} with patch {@code ph * (imageWidth / patchSize) + pw}
 * and element {@code c * patchSize^2 + i * patchSize + j}. Rows and columns beyond the last whole patch are not
 * patchified, and unpatchify leaves them 0. The decoder predicts the masked patches,
 * {@code [batch, numMasked, patchDim]} values.</p>
 */
public final class MaskedAutoencoder extends Module {

    public static final class MAEMask {
        private final int[] maskedIndices;
        private final int[] visibleIndices;
        private final float maskRatio;

        public MAEMask(int[] maskedIndices, int[] visibleIndices, float maskRatio) {
            this.maskedIndices = maskedIndices;
            this.visibleIndices = visibleIndices;
            this.maskRatio = maskRatio;
        }

        public int[] maskedIndices() { return maskedIndices; }
        public int[] visibleIndices() { return visibleIndices; }
        public float maskRatio() { return maskRatio; }
        public int numMasked() { return maskedIndices.length; }
        public int numVisible() { return visibleIndices.length; }
    }

    /**
     * The result of {@link #forwardMAE}, owned by the caller. {@link #maskedLoss()} is the recorded loss to
     * back-propagate. {@link #encoded()} and {@link #decoded()} are outputs of the same recorded graph, valid until the
     * loss is back-propagated (whose backward pass closes them); take {@link Tensor#detach()} handles first of any
     * value read afterwards. {@link #reconstruction()} is computed from the decoded rows without recording (an
     * inspection output that does not require grad) and stays valid; {@link MaskedAutoencoder#forward} returns a
     * recorded reconstruction instead.
     */
    public static final class MAEOutput {
        private final Tensor reconstruction;
        private final Tensor encoded;
        private final Tensor decoded;
        private final MAEMask mask;
        private final Tensor maskedLoss;

        public MAEOutput(Tensor reconstruction, Tensor encoded, Tensor decoded,
                         MAEMask mask, Tensor maskedLoss) {
            this.reconstruction = Objects.requireNonNull(reconstruction);
            this.encoded = Objects.requireNonNull(encoded);
            this.decoded = decoded;
            this.mask = mask;
            this.maskedLoss = maskedLoss;
        }

        public Tensor reconstruction() { return reconstruction; }
        public Tensor encoded() { return encoded; }
        public Tensor decoded() { return decoded; }
        public MAEMask mask() { return mask; }
        public Tensor maskedLoss() { return maskedLoss; }
    }

    private final Module encoder;
    private final Module decoder;
    private final int patchSize;
    private final int imageHeight;
    private final int imageWidth;
    private final int channels;
    private final float maskRatio;
    private Random random;
    private volatile boolean closed = false;

    public MaskedAutoencoder(Module encoder, Module decoder, int patchSize,
                              int imageHeight, int imageWidth, int channels) {
        this(encoder, decoder, patchSize, imageHeight, imageWidth, channels, 0.75f);
    }

    public MaskedAutoencoder(Module encoder, Module decoder, int patchSize,
                              int imageHeight, int imageWidth, int channels, float maskRatio) {
        Objects.requireNonNull(encoder, "Encoder cannot be null");
        Objects.requireNonNull(decoder, "Decoder cannot be null");
        if (patchSize <= 0) throw new IllegalArgumentException("Patch size must be positive");
        if (imageHeight <= 0 || imageWidth <= 0) {
            throw new IllegalArgumentException("Image dimensions must be positive");
        }
        if (channels <= 0) throw new IllegalArgumentException("Channels must be positive");
        if (maskRatio <= 0 || maskRatio >= 1) throw new IllegalArgumentException("Mask ratio must be in (0, 1)");

        this.encoder = registerModule("encoder", encoder);
        this.decoder = registerModule("decoder", decoder);
        this.patchSize = patchSize;
        this.imageHeight = imageHeight;
        this.imageWidth = imageWidth;
        this.channels = channels;
        this.maskRatio = maskRatio;
        this.random = new Random();
    }

    public int numPatches() {
        return (imageHeight / patchSize) * (imageWidth / patchSize);
    }

    /** A random split of the patch indices: {@code (int) (numPatches * maskRatio)} masked, the rest visible. */
    public MAEMask createMask() {
        int totalPatches = numPatches();
        int numMasked = (int) (totalPatches * maskRatio);

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalPatches; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);

        int[] maskedIndices = new int[numMasked];
        int[] visibleIndices = new int[totalPatches - numMasked];

        for (int i = 0; i < numMasked; i++) {
            maskedIndices[i] = indices.get(i);
        }
        for (int i = 0; i < totalPatches - numMasked; i++) {
            visibleIndices[i] = indices.get(numMasked + i);
        }

        return new MAEMask(maskedIndices, visibleIndices, maskRatio);
    }

    /**
     * {@code [batch, numPatches, patchDim]} patches of an image: the image viewed as
     * {@code [B, C, nH, p, nW, p]}, permuted to {@code [B, nH, nW, C, p, p]} and made contiguous.
     *
     * @throws IllegalArgumentException when the image does not hold {@code [batch, channels, imageHeight,
     *                                  imageWidth]} values, or the image is smaller than one patch
     */
    public Tensor patchify(Tensor image) {
        ensureOpen();
        Objects.requireNonNull(image, "Image cannot be null");
        requirePatches();

        int batchSize = image.shape().dim(0);
        int numPatchesH = imageHeight / patchSize;
        int numPatchesW = imageWidth / patchSize;
        if (image.numElements() != (long) batchSize * channels * imageHeight * imageWidth) {
            throw new IllegalArgumentException("Image of shape " + image.shape() + " does not hold [" + batchSize
                + ", " + channels + ", " + imageHeight + ", " + imageWidth + "] values");
        }
        try (Tensor nchw = image.reshape(batchSize, channels, imageHeight, imageWidth);
             Tensor rowsCropped = nchw.slice(2, 0, (long) numPatchesH * patchSize);
             Tensor cropped = rowsCropped.slice(3, 0, (long) numPatchesW * patchSize);
             Tensor blocks = cropped.reshape(batchSize, channels, numPatchesH, patchSize, numPatchesW, patchSize);
             Tensor ordered = blocks.permute(0, 2, 4, 1, 3, 5)) {
            return denseReshape(ordered, Shape.of(batchSize, numPatches(), patchDim()));
        }
    }

    /**
     * The inverse of {@link #patchify}: {@code [batch, channels, imageHeight, imageWidth]} with 0 where no whole
     * patch reaches.
     *
     * @throws IllegalArgumentException when the patches do not hold {@code [batch, numPatches, patchDim]} values
     */
    public Tensor unpatchify(Tensor patches) {
        ensureOpen();
        Objects.requireNonNull(patches, "Patches cannot be null");
        requirePatches();

        int batchSize = patches.shape().dim(0);
        int numPatchesH = imageHeight / patchSize;
        int numPatchesW = imageWidth / patchSize;
        if (patches.numElements() != (long) batchSize * numPatches() * patchDim()) {
            throw new IllegalArgumentException("Patches of shape " + patches.shape() + " do not hold [" + batchSize
                + ", " + numPatches() + ", " + patchDim() + "] values");
        }
        int coveredH = numPatchesH * patchSize;
        int coveredW = numPatchesW * patchSize;
        Tensor image;
        try (Tensor blocks = patches.reshape(batchSize, numPatchesH, numPatchesW, channels, patchSize, patchSize);
             Tensor ordered = blocks.permute(0, 3, 1, 4, 2, 5)) {
            image = denseReshape(ordered, Shape.of(batchSize, channels, coveredH, coveredW));
        }
        image = padWithZeros(image, 3, imageWidth - coveredW);
        return padWithZeros(image, 2, imageHeight - coveredH);
    }

    /**
     * The visible patches {@code [batch, numVisible, patchDim]}: row {@code b * numPatches + visible[i]} of the
     * patches viewed as {@code [batch * numPatches, patchDim]}, with {@code patchDim = numElements / (batch *
     * numPatches)}.
     *
     * @throws IllegalArgumentException  when the image is smaller than one patch, the patches do not hold
     *                                   {@code batch * numPatches} whole rows, or the mask has no visible patch
     * @throws IndexOutOfBoundsException when a visible index is outside {@code [0, numPatches)}
     */
    public Tensor applyMask(Tensor patches, MAEMask mask) {
        ensureOpen();
        Objects.requireNonNull(patches, "Patches cannot be null");
        Objects.requireNonNull(mask, "Mask cannot be null");
        requirePatches();
        if (mask.numVisible() == 0) {
            throw new IllegalArgumentException("The mask leaves no patch visible");
        }

        int batchSize = patches.shape().dim(0);
        int rowCount = batchSize * numPatches();
        int patchDim = (int) (patches.numElements() / rowCount);
        if ((long) rowCount * patchDim != patches.numElements()) {
            throw new IllegalArgumentException("Patches of shape " + patches.shape() + " do not split into "
                + rowCount + " patch rows");
        }
        int numVisible = mask.numVisible();
        try (Tensor rows = patches.reshape(rowCount, patchDim);
             Tensor index = patchRows(patches.backend(), batchSize, mask.visibleIndices());
             Tensor selected = TensorOps.indexSelectRows(rows, index, false)) {
            return selected.reshape(batchSize, numVisible, patchDim);
        }
    }

    /**
     * Patchifies, masks, encodes the visible patches and decodes. The masked loss is
     * {@code mean((patch[b, masked[i]] - decoded[b, i])^2)} over all masked rows and elements, and the
     * reconstruction is the patches with each masked patch replaced by its decoded row (see {@link MAEOutput} for
     * what is recorded).
     *
     * @throws IllegalArgumentException when the mask ratio masks no patch of {@link #numPatches()} (there is nothing
     *                                  to reconstruct), or the decoder output does not hold
     *                                  {@code [batch, numMasked, patchDim]} values
     */
    public MAEOutput forwardMAE(Tensor image) {
        return run(image, true);
    }

    /**
     * The reconstruction of {@link #forwardMAE}, recorded through the scatter of the decoded rows (so a loss on it
     * trains the encoder and decoder), without computing the masked loss; the encoder and decoder outputs are closed.
     */
    @Override
    public Tensor forward(Tensor input) {
        MAEOutput output = run(input, false);
        Tensor reconstruction = output.reconstruction();
        release(output.decoded(), input, reconstruction);
        release(output.encoded(), input, reconstruction, output.decoded());
        return reconstruction;
    }

    /**
     * Patchify, mask, encode and decode; then the recorded masked loss and an unrecorded reconstruction
     * ({@code withLoss}), or only the recorded reconstruction (the loss of the result is then null).
     */
    private MAEOutput run(Tensor image, boolean withLoss) {
        ensureOpen();
        Objects.requireNonNull(image, "Image cannot be null");

        int batchSize = image.shape().dim(0);
        int patchDim = patchDim();
        int totalRows = batchSize * numPatches();
        Tensor patches = patchify(image);
        Tensor visible = null;
        Tensor encoded = null;
        Tensor decoded = null;
        try {
            MAEMask mask = CheckpointFunction.replayable(this::createMask);
            if (mask.numMasked() == 0) {
                throw new IllegalArgumentException("Mask ratio " + maskRatio + " masks none of the " + numPatches()
                    + " patches");
            }
            int maskedRows = batchSize * mask.numMasked();
            visible = applyMask(patches, mask);
            encoded = encoder.forward(visible);
            decoded = decoder.forward(encoded);
            if (decoded.numElements() != (long) maskedRows * patchDim) {
                throw new IllegalArgumentException("Decoder output of shape " + decoded.shape() + " does not hold ["
                    + batchSize + ", " + mask.numMasked() + ", " + patchDim + "] values");
            }
            Tensor maskedLoss = null;
            Tensor reconstruction;
            try (Tensor rows = patches.reshape(totalRows, patchDim);
                 Tensor maskedIndex = patchRows(image.backend(), batchSize, mask.maskedIndices());
                 Tensor predicted = decoded.reshape(maskedRows, patchDim)) {
                if (!withLoss) {
                    reconstruction = reconstruct(rows, maskedIndex, predicted, batchSize);
                } else {
                    try (Tensor target = TensorOps.indexSelectRows(rows, maskedIndex, false);
                         Tensor diff = TensorOps.sub(target, predicted);
                         Tensor squared = TensorOps.mul(diff, diff)) {
                        maskedLoss = TensorOps.mean(squared);
                    }
                    try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                        reconstruction = reconstruct(rows, maskedIndex, predicted, batchSize);
                    } catch (RuntimeException e) {
                        maskedLoss.close();
                        throw e;
                    }
                }
            }
            return new MAEOutput(reconstruction, encoded, decoded, mask, maskedLoss);
        } catch (RuntimeException e) {
            release(decoded);
            release(encoded);
            throw e;
        } finally {
            release(visible, encoded, decoded);
            patches.close();
        }
    }

    /**
     * {@code [batch, numPatches, patchDim]}: the patch {@code rows} with row {@code maskedIndex[i]} replaced by row
     * {@code i} of {@code predicted} (a scatter along dim 0), recorded when grad mode is on.
     */
    private Tensor reconstruct(Tensor rows, Tensor maskedIndex, Tensor predicted, int batchSize) {
        int maskedRows = predicted.shape().dim(0);
        int patchDim = predicted.shape().dim(1);
        try (Tensor indexColumn = maskedIndex.reshape(maskedRows, 1);
             Tensor scatterIndex = indexColumn.expand(maskedRows, patchDim);
             Tensor merged = TensorOps.scatter(rows, 0, scatterIndex, predicted)) {
            return merged.reshape(batchSize, numPatches(), patchDim);
        }
    }

    private int patchDim() {
        return channels * patchSize * patchSize;
    }

    private void requirePatches() {
        if (numPatches() == 0) {
            throw new IllegalArgumentException("Image " + imageHeight + "x" + imageWidth + " is smaller than one "
                + patchSize + "x" + patchSize + " patch");
        }
    }

    /**
     * INT32 rows {@code b * numPatches + patches[i]} of the {@code [batch * numPatches, patchDim]} view, batch-major.
     *
     * @throws IndexOutOfBoundsException when a patch index is outside {@code [0, numPatches)}
     */
    private Tensor patchRows(ComputeBackend backend, int batchSize, int[] patches) {
        int total = numPatches();
        int[] rows = new int[batchSize * patches.length];
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < patches.length; i++) {
                int patch = patches[i];
                if (patch < 0 || patch >= total) {
                    throw new IndexOutOfBoundsException("Patch index " + patch + " out of bounds for " + total
                        + " patches");
                }
                rows[b * patches.length + i] = b * total + patch;
            }
        }
        return Tensor.fromArray(backend, rows);
    }

    /** {@code view} made contiguous and reshaped to {@code shape}, as a new tensor; the caller keeps {@code view}. */
    private static Tensor denseReshape(Tensor view, Shape shape) {
        Tensor dense = view.contiguous();
        try {
            return dense.reshape(shape);
        } finally {
            if (dense != view) {
                dense.close();
            }
        }
    }

    /** {@code image} (consumed) followed by {@code extra} zeros along {@code dim}; {@code image} when none. */
    private static Tensor padWithZeros(Tensor image, int dim, int extra) {
        if (extra == 0) {
            return image;
        }
        int[] padDims = image.shape().dims();
        padDims[dim] = extra;
        try (Tensor zeros = Tensor.zeros(image.backend(), Shape.of(padDims))) {
            return TensorOps.cat(Arrays.asList(image, zeros), dim);
        } finally {
            image.close();
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
    public Module decoder() { return decoder; }
    public int patchSize() { return patchSize; }
    public int imageHeight() { return imageHeight; }
    public int imageWidth() { return imageWidth; }
    public int channels() { return channels; }
    public float maskRatio() { return maskRatio; }

    public void setRandom(Random random) {
        this.random = Objects.requireNonNull(random);
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("MaskedAutoencoder has been closed");
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
        if (closed) return "MaskedAutoencoder{closed=true}";
        return "MaskedAutoencoder{patchSize=" + patchSize +
               ", imageSize=" + imageHeight + "x" + imageWidth +
               ", maskRatio=" + maskRatio + "}";
    }
}
