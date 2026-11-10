package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

import java.util.Arrays;

/**
 * llama.cpp's locally typical sampler (llama_sampler_typical_apply): among the candidates whose cumulative
 * probability first exceeds {@code typicalP}, the ones closest to the entropy of the distribution. A
 * {@code typicalP >= 1} applies nothing. {@code minKeep} stops the truncation only once that many candidates are
 * kept (llama.cpp's {@code i >= min_keep - 1} check). Distant candidates keep the candidate index as their tie-break,
 * and the survivors stay in their shifted-distance order, so the set is unsorted afterwards.
 *
 * <p>The shifted distances are not sorted whole: the stage first selects the {@value SmallestKeys#FIRST} nearest
 * ({@link SmallestKeys}), and a wider survivor set walks their ascending order bucket by bucket
 * ({@link BucketedOrder}), sorting only the buckets the walk reaches. The survivors, their order and the cumulative
 * sums are those of the full sort. The row-sized scratch of the keys and the walk is borrowed for the apply
 * ({@link TruncationScratch}), so the stage holds none between applies.</p>
 */
public final class TypicalStage extends SamplerStage {

    private final float typicalP;
    private final int minKeep;
    /** The cumulative mass of the order walked so far in one apply. */
    private float mass;
    /** The shifted distance (nats) past which every candidate shares the last bucket. */
    private static final float SPAN = 64f;

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
        TruncationScratch scratch = TruncationScratch.borrow();
        try {
            truncate(candidates, n, entropy, scratch);
        } finally {
            TruncationScratch.release(scratch);
        }
    }

    /** Truncates the {@code n} sorted candidates of entropy {@code entropy} to the typical set. */
    private void truncate(Candidates candidates, int n, float entropy, TruncationScratch scratch) {
        long[] k = scratch.keys(n);
        float[] d = scratch.offsets(n);
        boolean nan = false;
        for (int i = 0; i < n; i++) {
            float p = candidates.prob(i);
            float shifted = p > 0f ? Math.abs(-(float) Math.log(p) - entropy) : Float.POSITIVE_INFINITY;
            k[i] = Candidates.ascendingKey(shifted, i);
            d[i] = shifted;
            nan |= Float.isNaN(shifted);
        }
        if (!nan && !SmallestKeys.fullSort(SmallestKeys.FIRST, n)) {
            long[] order = scratch.prefix.select(k, n, SmallestKeys.FIRST);
            mass = 0f;
            int last = survivors(candidates, order, 0, SmallestKeys.FIRST);
            if (last > 0) {
                keep(candidates, order, last);
                return;
            }
            order = scratch.buckets.prepare(k, d, n, SPAN);
            mass = 0f;
            int done = 0;
            for (int b = 0; b < BucketedOrder.BUCKETS; b++) {
                int end = scratch.buckets.sortBucket(b);
                last = survivors(candidates, order, done, end);
                if (last > 0) {
                    keep(candidates, order, last);
                    return;
                }
                done = end;
            }
            keep(candidates, order, n);
            return;
        }
        Arrays.sort(k, 0, n);
        mass = 0f;
        int last = survivors(candidates, k, 0, n);
        keep(candidates, k, last > 0 ? last : n);
    }

    /**
     * The survivors once {@code order[from, to)} joins the walk whose mass so far is {@link #mass}: the prefix whose
     * mass first exceeds {@code typicalP}, or 0 with {@link #mass} carrying the walk's mass on.
     */
    private int survivors(Candidates candidates, long[] order, int from, int to) {
        float cum = mass;
        for (int i = from; i < to; i++) {
            cum += candidates.prob((int) order[i]);
            if (cum > typicalP && (minKeep == 0 || i >= minKeep - 1)) {
                return i + 1;
            }
        }
        mass = cum;
        return 0;
    }

    private static void keep(Candidates candidates, long[] order, int last) {
        candidates.permute(order, last);
        candidates.truncate(last);
        candidates.setSorted(false);
    }
}
