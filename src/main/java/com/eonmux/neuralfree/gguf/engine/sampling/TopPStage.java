package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

/**
 * llama.cpp's nucleus sampler (llama_sampler_top_p_apply): the smallest prefix of the descending logit order whose
 * cumulative probability reaches {@code topP}, probabilities coming from the last softmax over the current logits. A
 * {@code topP >= 1} applies nothing. {@code minKeep} stops the truncation only once that many candidates are kept
 * (llama.cpp's {@code i + 1 >= min_keep} check). The survivors stay in descending order.
 *
 * <p>An unsorted set (a whole row with no top-k before) is not sorted whole. The stage first selects the
 * {@value SmallestKeys#FIRST} highest logits ({@link SmallestKeys}), which hold a peaked distribution's nucleus;
 * a wider nucleus walks the descending order bucket by bucket ({@link BucketedOrder}, buckets by distance below
 * the highest logit), sorting only the buckets the walk reaches. The survivors, their order and the cumulative sums
 * are those of the full sort. The walk's row-sized scratch is borrowed for the apply ({@link TruncationScratch}), so
 * the stage holds none between applies.</p>
 */
public final class TopPStage extends SamplerStage {

    private final float topP;
    private final int minKeep;
    /** The logit distance below the highest one past which every candidate shares the last bucket. */
    private static final float SPAN = 64f;

    public TopPStage(float topP, int minKeep) {
        this.topP = topP;
        this.minKeep = minKeep;
    }

    @Override
    public StageKind kind() {
        return StageKind.TOP_P;
    }

    @Override
    public void apply(Candidates candidates) {
        if (topP >= 1f) {
            return;
        }
        candidates.softmax();
        if (!candidates.sorted() && truncatedFromPrefix(candidates)) {
            return;
        }
        if (!candidates.sorted()) {
            candidates.sortDescending();
        }
        float cum = 0f;
        int last = candidates.size();
        for (int i = 0; i < candidates.size(); i++) {
            cum += candidates.prob(i);
            if (cum >= topP && i + 1 >= minKeep) {
                last = i + 1;
                break;
            }
        }
        candidates.truncate(last);
    }

    /**
     * Truncates an unsorted set to its nucleus from the descending order's prefix, sorting only that prefix; false,
     * the set untouched, for a small set, a NaN logit or a highest logit that is not finite (the full sort orders
     * those).
     */
    private boolean truncatedFromPrefix(Candidates candidates) {
        int n = candidates.size();
        if (SmallestKeys.fullSort(SmallestKeys.FIRST, n)) {
            return false;
        }
        float max = candidates.maxLogit();
        if (!Float.isFinite(max)) {
            return false;
        }
        TruncationScratch scratch = TruncationScratch.borrow();
        try {
            return truncatedFromPrefix(candidates, n, max, scratch);
        } finally {
            TruncationScratch.release(scratch);
        }
    }

    /** {@link #truncatedFromPrefix(Candidates)} over {@code n} candidates whose highest logit is {@code max}. */
    private boolean truncatedFromPrefix(Candidates candidates, int n, float max, TruncationScratch scratch) {
        long[] k = scratch.keys(n);
        float[] below = scratch.offsets(n);
        for (int i = 0; i < n; i++) {
            float logit = candidates.logit(i);
            if (Float.isNaN(logit)) {
                return false;
            }
            k[i] = Candidates.descendingKey(logit, i);
            below[i] = max - logit;
        }
        long[] top = scratch.prefix.select(k, n, SmallestKeys.FIRST);
        if (nucleus(candidates, top, 0, SmallestKeys.FIRST, 0f) < 0) {
            return true;
        }
        long[] order = scratch.buckets.prepare(k, below, n, SPAN);
        float cum = 0f;
        int done = 0;
        for (int b = 0; b < BucketedOrder.BUCKETS; b++) {
            int end = scratch.buckets.sortBucket(b);
            float reached = nucleus(candidates, order, done, end, cum);
            if (reached < 0) {
                return true;
            }
            cum = reached;
            done = end;
        }
        candidates.permute(order, n);
        candidates.setSorted(true);
        return true;
    }

    /**
     * Continues the cumulative sum {@code cum} over {@code order[from, to)}: at the first entry that completes the
     * nucleus the set is truncated to it and -1 returned, else the sum so far.
     */
    private float nucleus(Candidates candidates, long[] order, int from, int to, float cum) {
        for (int i = from; i < to; i++) {
            cum += candidates.prob((int) order[i]);
            if (cum >= topP && i + 1 >= minKeep) {
                candidates.permute(order, i + 1);
                candidates.truncate(i + 1);
                candidates.setSorted(true);
                return -1f;
            }
        }
        return cum;
    }
}
