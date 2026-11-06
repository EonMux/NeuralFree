package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Rows;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Source;

import java.util.Arrays;
import java.util.Objects;

/**
 * The fusion projection every EAGLE drafter applies to its features: {@code g = fc . features}, plus the projection
 * bias of the checkpoints that keep one ({@code EagleCohereForCausalLM}'s {@code fc.bias}). Both the EAGLE-3 encoder
 * ({@link Eagle3DraftModel}) and the concat-fc runtime ({@link EagleFcDraftModel}) fuse through this class, so the
 * shape check and the bias handling live in one place.
 *
 * <p>The bias rows are tiled on the device once per head: {@link #project} adds row {@code 0..n} of the tiled bias
 * after the projection. Immutable apart from the row buffers it allocates on {@link #allocate()}; not thread-safe.
 */
final class EagleFusion {

    private final MtpHead head;
    private final Source source;
    private final String fc;
    private final float[] bias;
    private final int outWidth;

    private Rows biasRows;
    private float[] tiled;

    private EagleFusion(MtpHead head, Source source, String fc, float[] bias, int outWidth) {
        this.head = head;
        this.source = source;
        this.fc = fc;
        this.bias = bias;
        this.outWidth = outWidth;
    }

    /**
     * Binds the fusion projection {@code fc} of {@code source}.
     *
     * @param inWidth  the feature width the projection reads
     * @param outWidth the fused width it writes
     * @param bias     the projection bias tensor, or null when the family has none; an absent optional bias tensor
     *                 leaves the projection unbiased
     * @throws IllegalArgumentException when the projection is absent or does not have the expected shape, or the
     *                                  bias does not hold one value per output
     */
    static EagleFusion bind(MtpHead head, Source source, String fc, String bias, int inWidth, int outWidth) {
        Objects.requireNonNull(head, "head");
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(fc, "fc");
        if (!head.has(source, fc)) {
            throw new IllegalArgumentException("the drafter needs " + fc);
        }
        long[] shape = head.shape(source, fc);
        if (shape.length != 2 || shape[0] != inWidth || shape[1] != outWidth) {
            throw new IllegalArgumentException(fc + " has shape " + Arrays.toString(shape) + ", expected ["
                    + inWidth + ", " + outWidth + "]");
        }
        float[] values = null;
        if (bias != null && head.has(source, bias)) {
            values = head.vector(source, bias);
            if (values.length != outWidth) {
                throw new IllegalArgumentException(bias + " holds " + values.length + " values, expected "
                        + outWidth);
            }
        }
        return new EagleFusion(head, source, fc, values, outWidth);
    }

    /** Allocates the row buffers; called with the other row buffers of the drafter. */
    void allocate() {
        if (bias == null) {
            return;
        }
        biasRows = head.rows("eagle.fc_bias", outWidth);
        tiled = new float[Math.multiplyExact(head.ubatch(), outWidth)];
        for (int r = 0; r < head.ubatch(); r++) {
            System.arraycopy(bias, 0, tiled, r * outWidth, outWidth);
        }
    }

    /** {@code out = fc . in}, plus the bias over the {@code n} rows when the checkpoint keeps one. */
    void project(Rows in, Rows out, int n) {
        head.project(source, fc, in, out, n);
        if (bias == null) {
            return;
        }
        head.write(biasRows, tiled, n);
        head.add(out, biasRows, out, n);
    }

    @Override
    public String toString() {
        return "EagleFusion{" + fc + " -> " + outWidth + (bias != null ? ", bias" : "") + '}';
    }
}
