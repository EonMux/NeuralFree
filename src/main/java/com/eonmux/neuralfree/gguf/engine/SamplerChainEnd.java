package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.Mt19937;
import com.eonmux.neuralfree.gguf.engine.watermark.Watermarker;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkState;

/**
 * The fixed chain end of one sampler: the argmax under greedy decoding, the categorical draw over the final
 * candidates (llama.cpp's {@code llama_sampler_dist_apply}), or the watermark's keyed selection in its place.
 * Holds the chain's watermark; the sampler hands over its candidate set and RNG per step. Not thread-safe.
 */
final class SamplerChainEnd {

    /** The watermark that replaces the categorical draw, or null for an ordinary chain. */
    private final Watermarker watermarker;
    /** The watermark's PRF context and repeated-context history, or null without a watermark. */
    private final WatermarkState watermarkState;

    SamplerChainEnd(Watermarker watermarker, WatermarkState watermarkState) {
        this.watermarker = watermarker;
        this.watermarkState = watermarkState;
    }

    /** Argmax in the current order (the first maximum wins). */
    void selectArgmax(Sampler.Candidates cur) {
        int best = 0;
        for (int i = 1; i < cur.size(); i++) {
            if (cur.logit(i) > cur.logit(best)) {
                best = i;
            }
        }
        if (cur.logit(best) == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
        }
        cur.select(best);
    }

    /**
     * Softmax and a single categorical draw in the current order ({@code llama_sampler_dist_apply} through
     * libstdc++'s {@code std::discrete_distribution}): the probabilities renormalized in double, the first
     * candidate whose cumulative reaches the canonical double draw. The chain's own {@link Mt19937} draws once
     * per call, also for a single candidate (llama.cpp keeps the RNG aligned with backend sampling).
     */
    void drawDist(Sampler.Candidates cur, Mt19937 rng) {
        int n = cur.size();
        if (n == 1) {
            if (cur.logit(0) == Float.NEGATIVE_INFINITY) {
                throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
            }
            cur.setProb(0, 1f);
            cur.select(rng.distDraw(new float[]{1f}));
            return;
        }
        float[] probs = softmaxDistribution(cur);
        cur.select(rng.distDraw(probs));
    }

    /**
     * The softmax over the current candidates, written into the set's probabilities and returned as a probability
     * array of exactly the candidate count (the draw reads its length); a single candidate takes all the mass. Used
     * by both the ordinary draw and a watermarked step, whose reported logprobs are the chain's ordinary
     * distribution even though the token is drawn from the keyed one.
     *
     * @throws IllegalStateException when no candidate has a finite logit
     */
    private float[] softmaxDistribution(Sampler.Candidates cur) {
        int n = cur.size();
        float max = cur.sorted() ? cur.logit(0) : cur.maxLogit();
        if (max == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("no candidate has a finite logit (all banned, -inf or NaN)");
        }
        if (n == 1) {
            cur.setProb(0, 1f);
            return new float[]{1f};
        }
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double e = expShift(cur.logit(i), max);
            cur.setProb(i, (float) e);
            sum += e;
        }
        for (int i = 0; i < n; i++) {
            cur.setProb(i, (float) (cur.prob(i) / sum));
        }
        float[] probs = new float[n];
        for (int i = 0; i < n; i++) {
            probs[i] = cur.prob(i);
        }
        return probs;
    }

    /**
     * The watermarked selection ({@link Watermarker}): the candidates the chain kept are presented to the watermark,
     * which picks the argmax of {@code logit + Gumbel(PRF(key, context, token))}; the set's probabilities are the
     * chain's ordinary softmax either way. A context the repeated-context search already saw ({@link
     * WatermarkState#repeatedContext}) falls back to the ordinary categorical draw, though a dual-key watermarker has
     * already taken its routing draw.
     *
     * @throws IllegalStateException when no candidate has a finite logit
     */
    void selectWatermarked(Sampler.Candidates cur, Mt19937 rng) {
        int n = cur.size();
        float[] probs = softmaxDistribution(cur);
        boolean repeated = watermarkState.repeatedContext();
        int[] tokenIds = new int[n];
        float[] logits = new float[n];
        for (int i = 0; i < n; i++) {
            tokenIds[i] = cur.id(i);
            logits[i] = cur.logit(i);
        }
        int token = watermarker.sample(tokenIds, logits, n, watermarkState.context(),
                rng::nextCanonicalDouble);
        if (repeated) {
            cur.select(rng.distDraw(probs));
        } else {
            int at = cur.indexOf(token);
            if (at < 0) {
                throw new IllegalStateException("the watermark selected token " + token
                        + ", which is not among the chain's candidates");
            }
            cur.select(at);
        }
    }

    /**
     * {@code exp(logit - max)}; when {@code max} is {@code +inf} (an overflowed logit, for example after diffusion
     * Gumbel noise) this is its limit: the infinite logits share all the mass.
     */
    static float expShift(float logit, float max) {
        if (max == Float.POSITIVE_INFINITY) {
            return logit == Float.POSITIVE_INFINITY ? 1f : 0f;
        }
        return (float) Math.exp(logit - max);
    }
}
