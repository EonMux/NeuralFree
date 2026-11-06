package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Objects;
import java.util.function.DoubleSupplier;

/**
 * A Gumbel-max watermark sampler: given a step's candidate logits and the context of preceding tokens, it selects
 * the token to emit instead of an ordinary categorical draw. The selection is
 * {@code argmax(logit + Gumbel(PRF(key, context, tokenId)))} over the candidates, which is a categorical draw from
 * {@code softmax(logits)} but with the randomness taken from the keyed PRF rather than the request's RNG; that is
 * what makes the output detectable from the key alone.
 *
 * <p>The candidates are the token ids and logits the sampler chain left after its truncating stages (top-k, top-p and
 * the rest), exactly the set vLLM watermarks after {@code apply_top_k_top_p}; a token masked to {@code -inf} never
 * wins. {@link DualKeyGumbelWatermarker} additionally consumes one routing uniform per call through
 * {@code routing}, so callers must pass a supplier that draws once per sampled row even when they discard the
 * result. The watermarked token is not removed from the ordinary probability distribution: callers that report
 * logprobs still compute the softmax probabilities themselves.
 */
public abstract class Watermarker {

    /** {@code torch.finfo(torch.float32).tiny}, the clamp vLLM applies to a PRF draw before the Gumbel transform. */
    static final float FLOAT32_TINY = Float.MIN_NORMAL;

    /** The number of prior tokens the PRF keys on. */
    public abstract int contextWidth();

    /**
     * The watermarked token among {@code logits[0..count)}.
     *
     * @param tokenIds the candidate token ids, aligned with {@code logits}
     * @param logits   the candidate logits
     * @param count    how many candidates to consider
     * @param context  the preceding {@link #contextWidth()} token ids, oldest first (-1 where the history is shorter)
     * @param routing  one uniform draw in {@code [0, 1)} for dual-key routing (never read by the single-key
     *                 watermarker)
     * @return the selected candidate's token id
     * @throws IllegalArgumentException when the arrays disagree with {@code count} or the context has the wrong width
     */
    public abstract int sample(int[] tokenIds, float[] logits, int count, int[] context, DoubleSupplier routing);

    /**
     * {@link #sample(int[], float[], int, int[], DoubleSupplier)} over a whole logits row, whose candidate token ids
     * are their positions in the row.
     */
    public final int sample(float[] logits, int[] context, DoubleSupplier routing) {
        Objects.requireNonNull(logits, "logits");
        int[] tokenIds = new int[logits.length];
        for (int i = 0; i < tokenIds.length; i++) {
            tokenIds[i] = i;
        }
        return sample(tokenIds, logits, logits.length, context, routing);
    }

    /**
     * The watermarker of a configuration.
     *
     * @throws IllegalArgumentException when the configuration is null or names no algorithm
     */
    public static Watermarker create(WatermarkConfig config) {
        Objects.requireNonNull(config, "config");
        if (config.algorithm() == WatermarkAlgorithm.DUAL_KEY_GUMBEL) {
            return new DualKeyGumbelWatermarker(config.key(), config.contextWidth(), config.alpha(),
                    config.prf());
        }
        if (config.algorithm() == WatermarkAlgorithm.GUMBEL) {
            return new GumbelWatermarker(config.key(), config.contextWidth(), config.prf());
        }
        throw new IllegalArgumentException("unknown watermarking algorithm " + config.algorithm());
    }

    /**
     * The Gumbel-max selection over the candidates, the common step of both algorithms.
     *
     * @throws IllegalArgumentException for a count outside the arrays or a context of the wrong width
     */
    static int gumbelSelect(WatermarkPrf prf, int[] tokenIds, float[] logits, int count, int[] context, int width) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(context, "context");
        if (count < 1 || count > tokenIds.length || count > logits.length) {
            throw new IllegalArgumentException("cannot watermark " + count + " candidates of " + tokenIds.length
                    + " token ids and " + logits.length + " logits");
        }
        if (context.length != width) {
            throw new IllegalArgumentException("the watermark context has " + context.length + " tokens, expected "
                    + width);
        }
        double[] uniforms = new double[count];
        prf.uniforms(context, tokenIds, count, uniforms);
        int best = 0;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            float score = logits[i] + gumbelNoise(uniforms[i]);
            if (i == 0 || score > bestScore) {
                best = i;
                bestScore = score;
            }
        }
        return tokenIds[best];
    }

    /**
     * {@code -log(-log(u))} in {@code float32}, with the draw clamped to the smallest normal float as vLLM does.
     * Computed with the float rounding of each step (the logarithm, its negation, then the logarithm again) so the
     * noise matches PyTorch's {@code float32} tensor arithmetic.
     */
    static float gumbelNoise(double uniform) {
        float u = (float) uniform;
        if (!(u > FLOAT32_TINY)) {
            u = FLOAT32_TINY;
        } else if (u >= 1f) {
            u = Math.nextDown(1f);
        }
        float logU = (float) Math.log(u);
        return -(float) Math.log(-logU);
    }
}
