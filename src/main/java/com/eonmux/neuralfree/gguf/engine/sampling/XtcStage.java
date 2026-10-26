package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's XTC sampler (llama_sample_xtc_apply, llama.cpp src/llama-sampler.cpp:2350): with probability
 * {@code probability}, every candidate whose probability reaches {@code threshold} is removed from the front of the
 * descending set - the tail behind the last candidate over the threshold stays. The stage softmaxes over the sorted
 * set first (llama_sampler_softmax_impl(cur_p, true)), so the trimmed tail stays sorted.
 *
 * <p>{@code probability <= 0}, a {@code threshold > 0.5} (the disabled marker, llama.cpp's init check) and sets of
 * fewer than two candidates apply nothing, exactly as llama.cpp's empty sampler does; neither does the stage draw when
 * {@code chance > probability} (std::uniform_real_distribution&lt;float&gt;(0, 1) over its own std::mt19937, seeded
 * from the chain's seed as llama.cpp's init does, restarted by {@link #reset()} as llama_sampler_xtc_reset does).
 * {@code min_keep} keeps that many candidates when the tail cut would go below it.
 */
public final class XtcStage extends SamplerStage {

    private final float probability;
    private final float threshold;
    private final int minKeep;
    private final long seed;
    private final Mt19937 rng;

    public XtcStage(float probability, float threshold, int minKeep, long seed) {
        this.probability = probability;
        this.threshold = threshold;
        this.minKeep = minKeep;
        this.seed = seed;
        this.rng = new Mt19937(seed);
    }

    @Override
    public StageKind kind() {
        return StageKind.XTC;
    }

    @Override
    public boolean modifies() {
        return probability > 0.0f && threshold <= 0.5f;
    }

    @Override
    public void apply(Candidates candidates) {
        if (probability <= 0.0f || threshold > 0.5f || candidates.size() < 2) {
            return;
        }
        float chance = rng.nextCanonicalFloat();
        if (chance > probability) {
            return;
        }
        candidates.softmaxSorted();
        int posLast = 0;
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.prob(i) >= threshold) {
                posLast = i;
            } else {
                break;
            }
        }
        int keep = candidates.size() - posLast;
        if (keep >= minKeep && posLast > 0) {
            candidates.removeFirst(posLast);
        }
    }

    @Override
    public void reset() {
        rng.seed(seed);
    }
}
