package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.DraftVocabMapping;
import com.eonmux.neuralfree.gguf.engine.RowTopK;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * The children a draft tree's node takes from its logits row ({@link BatchModelDrafter#draftTreeBatch}): the row's
 * best draft ids in rank order (larger logit first, the lower id on ties; non-finite logits never extend, nor ids
 * outside the target vocabulary, nor a second draft id of a target id already taken), each scored by its
 * log-softmax over the row. A row comes from a whole logits row selected on the host, or from a device top-k
 * readout that carries each row's log-sum-exp ({@code ForwardBackend#forwardBatchTopKNormalized}); both give the
 * same ids and, up to the device's float sum, the same scores. Immutable.
 */
final class TreeCandidates {

    private final int[] ids;
    private final float[] logits;
    private final double logSumExp;

    private TreeCandidates(int[] ids, float[] logits, double logSumExp) {
        this.ids = ids;
        this.logits = logits;
        this.logSumExp = logSumExp;
    }

    /**
     * The device readout width that yields {@code branches} children per row exactly over an identity vocabulary:
     * the branches plus every draft id past the target vocabulary, which may outrank them and never extends; -1
     * when the mapping is not the identity or the width exceeds a device readout ({@link
     * RowTopK#MAX_DEVICE_CANDIDATES}), where the rows are read whole.
     */
    static int deviceWidth(int branches, DraftVocabMapping vocab) {
        if (branches < 1 || !vocab.isIdentity()) {
            return -1;
        }
        long width = (long) branches + Math.max(0, vocab.draftVocabSize() - vocab.targetVocabSize());
        return width <= RowTopK.MAX_DEVICE_CANDIDATES ? (int) width : -1;
    }

    /**
     * The best {@code branches} children of a whole logits row after the vocabulary's mask ({@link
     * DraftVocabMapping#constrain}, applied to {@code logits} in place), scored over the masked row.
     *
     * @throws IllegalArgumentException when {@code branches} is below one
     */
    static TreeCandidates ofRow(float[] logits, int branches, DraftVocabMapping vocab) {
        Objects.requireNonNull(logits, "logits");
        if (branches < 1) {
            throw new IllegalArgumentException("a tree row extends at least one branch, got " + branches);
        }
        vocab.constrain(logits);
        int ranking = 0;
        for (float v : logits) {
            if (ranks(v)) {
                ranking++;
            }
        }
        if (ranking == 0) {
            return new TreeCandidates(new int[0], new float[0], Double.NaN);
        }
        double lse = RowTopK.logSumExp(logits);
        for (int width = Math.min(branches, ranking); ; width = Math.min(2 * width, ranking)) {
            int[] ranked = topRanked(logits, width);
            int[] kept = keep(ranked, ranked.length, branches, vocab);
            if (kept.length == branches || width == ranking) {
                float[] vals = new float[kept.length];
                for (int i = 0; i < kept.length; i++) {
                    vals[i] = logits[kept[i]];
                }
                return new TreeCandidates(kept, vals, lse);
            }
        }
    }

    /**
     * The best {@code branches} children of row {@code row} of a normalized device readout of {@link #deviceWidth}
     * candidates over an identity vocabulary.
     *
     * @throws IllegalArgumentException when {@code branches} is below one
     * @throws IllegalStateException    when the readout carries no log-sum-exp
     */
    static TreeCandidates ofTopK(RowTopK top, int row, int branches, DraftVocabMapping vocab) {
        Objects.requireNonNull(top, "top");
        if (branches < 1) {
            throw new IllegalArgumentException("a tree row extends at least one branch, got " + branches);
        }
        int[] ranked = new int[top.k()];
        int n = 0;
        for (int j = 0; j < top.k(); j++) {
            if (!ranks(top.logit(row, j))) {
                break;
            }
            ranked[n++] = top.id(row, j);
        }
        int[] kept = keep(ranked, n, branches, vocab);
        float[] vals = new float[kept.length];
        for (int i = 0, j = 0; i < kept.length; j++) {
            if (top.id(row, j) == kept[i]) {
                vals[i++] = top.logit(row, j);
            }
        }
        return new TreeCandidates(kept, vals, top.logSumExp(row));
    }

    /** Children the row offers, at most the branches asked. */
    int size() {
        return ids.length;
    }

    /** Draft id of child {@code i}, best first. */
    int id(int i) {
        return ids[i];
    }

    /** Child {@code i}'s log-softmax over its row. */
    double logProb(int i) {
        return logits[i] - logSumExp;
    }

    /**
     * The first {@code branches} of the {@code n} ranked draft ids that extend: a target id of their own, each
     * target id once.
     */
    private static int[] keep(int[] ranked, int n, int branches, DraftVocabMapping vocab) {
        int[] kept = new int[Math.min(branches, n)];
        Set<Integer> seen = new HashSet<>();
        int count = 0;
        for (int i = 0; i < n && count < kept.length; i++) {
            int target = vocab.toTarget(ranked[i]);
            if (target >= 0 && seen.add(target)) {
                kept[count++] = ranked[i];
            }
        }
        return count == kept.length ? kept : Arrays.copyOf(kept, count);
    }

    /**
     * The indices of the {@code width} largest values of {@code row} that rank (neither NaN nor {@code -inf}), larger
     * first and the lower index first among equal values; {@code width} must not exceed the row's ranking values.
     */
    static int[] topRanked(float[] row, int width) {
        int[] idx = new int[width];
        int filled = 0;
        for (int i = 0; i < row.length; i++) {
            float v = row[i];
            if (!ranks(v) || filled == width && !(v > row[idx[width - 1]])) {
                continue;
            }
            int pos = filled < width ? filled++ : width - 1;
            while (pos > 0 && v > row[idx[pos - 1]]) {
                idx[pos] = idx[pos - 1];
                pos--;
            }
            idx[pos] = i;
        }
        return idx;
    }

    /** Whether a logit can rank as a child: neither NaN nor {@code -inf}. */
    private static boolean ranks(float v) {
        return !Float.isNaN(v) && v != Float.NEGATIVE_INFINITY;
    }
}
