package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

import java.util.Arrays;

/**
 * llama.cpp's locally typical sampler (llama_sampler_typical_apply): among the candidates whose cumulative
 * probability first exceeds {@code typicalP}, the ones closest to the entropy of the distribution. A
 * {@code typicalP >= 1} applies nothing. {@code minKeep} stops the truncation only once that many candidates are
 * kept (llama.cpp's {@code i >= min_keep - 1} check). Distant candidates keep the candidate index as their tie-break,
 * and the survivors stay in their shifted-distance order, so the set is unsorted afterwards.
 */
public final class TypicalStage extends SamplerStage {

    private final float typicalP;
    private final int minKeep;
    /** Scratch for the shifted distances' sort keys (reused across applies of one sampler). */
    private long[] keys = new long[0];

    public TypicalStage(float typicalP, int minKeep) {
        this.typicalP = typicalP;
        this.minKeep = minKeep;
    }

    @Override
    public StageKind kind() {
        return StageKind.TYPICAL_P;
    }

    @Override
    public void apply(Candidates candidates) {
        if (typicalP >= 1f) {
            return;
        }
        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        candidates.softmax();
        int n = candidates.size();
        float entropy = 0f;
        for (int i = 0; i < n; i++) {
            float p = candidates.prob(i);
            if (p > 0f) {
                entropy += -p * (float) Math.log(p);
            }
        }
        long[] k = keys(n);
        for (int i = 0; i < n; i++) {
            float p = candidates.prob(i);
            float shifted = p > 0f ? Math.abs(-(float) Math.log(p) - entropy) : Float.POSITIVE_INFINITY;
            k[i] = Candidates.ascendingKey(shifted, i);
        }
        Arrays.sort(k, 0, n);
        float cum = 0f;
        int last = n;
        for (int i = 0; i < n; i++) {
            cum += candidates.prob((int) k[i]);
            if (cum > typicalP && (minKeep == 0 || i >= minKeep - 1)) {
                last = i + 1;
                break;
            }
        }
        candidates.permute(k, last);
        candidates.truncate(last);
        candidates.setSorted(false);
    }

    private long[] keys(int n) {
        if (keys.length < n) {
            keys = new long[n];
        }
        return keys;
    }
}
