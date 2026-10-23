package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * DSpark Markov and confidence heads, {@code build_dspark_markov_head} (dflash.cpp:369-481), for one block.
 * With {@code prev} starting at the anchor token (block position 0) and positions {@code i} from 0 (anchor-first)
 * or 1:
 * <pre>
 *   bias      = scatter_d2t(markov_w2 . markov_w1[prev])          (times markov_w2.scale when present)
 *   logits[i] = logits[i] + bias
 *   conf[i]   = sigmoid(conf_proj . [t_embd[i] ; markov_w1[prev]] + conf_proj.bias)
 *   prev      = argmax(logits[i])                                  (before the last position)
 * </pre>
 * Without the anchor slot, position 0 keeps its logits and gets {@code conf[0] = sigmoid(logits[0][0])}. A block
 * longer than {@code dflash.block_size} is left unchanged with no confidences (dflash.cpp:399-401). Immutable.
 */
final class DflashMarkovHead {

    private final DflashMatrix w1;
    private final DflashMatrix w2;
    private final DflashMatrix confProj;
    private final float[] confBias;
    private final int[] d2t;
    private final int nVocab;
    private final int nEmbd;
    private final int rank;
    private final int blockSize;
    private final boolean sampleFromAnchor;

    /**
     * @param d2t target id of each draft-head row, or null without a reduced vocabulary
     * @throws IllegalArgumentException when a tensor is missing or mis-shaped
     */
    DflashMarkovHead(DflashSpec spec, CpuWeights draft, int[] d2t) {
        this.nVocab = spec.nVocab();
        this.nEmbd = spec.nEmbd();
        this.w1 = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DSPARK_MARKOV_W1));
        this.rank = w1.cols();
        w1.expect(nVocab, rank);
        this.w2 = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DSPARK_MARKOV_W2))
                .expect(spec.nVocabDraft(), rank);
        this.confProj = DflashMatrix.orNull(draft, TensorNames.weight(TensorNames.DSPARK_CONF_PROJ));
        if (confProj != null) {
            confProj.expect(1, (long) nEmbd + rank);
        }
        float[] b = draft.floatsOrNull(TensorNames.bias(TensorNames.DSPARK_CONF_PROJ));
        this.confBias = b == null ? null : DflashBackbone.vector(b, 1, TensorNames.bias(TensorNames.DSPARK_CONF_PROJ));
        this.d2t = d2t;
        this.blockSize = spec.graphBlockSize();
        this.sampleFromAnchor = spec.sampleFromAnchor();
    }

    /** True when the model has a confidence head ({@code conf_proj}). */
    boolean hasConfidence() {
        return confProj != null;
    }

    /**
     * Biases {@code logits[i]} ({@code [nTokens][nVocab]}, target vocabulary) in place.
     *
     * @param tEmbd  {@code t_embd} rows {@code [nTokens][nEmbd]}
     * @param anchor the token at block position 0
     * @return per-position confidences, or null without a confidence head or for a block longer than
     *         {@code dflash.block_size}
     */
    float[] apply(float[] logits, float[] tEmbd, int nTokens, int anchor) {
        if (nTokens > blockSize) {
            return null;
        }
        float[] conf = confProj != null ? new float[nTokens] : null;
        final int iBeg = sampleFromAnchor ? 0 : 1;
        if (!sampleFromAnchor && conf != null && nTokens > 0) {
            conf[0] = ActMath.sigmoid(logits[0]);
        }
        float[] w1Prev = new float[rank];
        float[] bias = new float[w2.rows()];
        float[] feat = conf != null ? new float[nEmbd + rank] : null;
        float[] c = new float[1];
        int prev = anchor;
        for (int i = iBeg; i < nTokens; i++) {
            w1.row(prev, w1Prev, 0);
            w2.matvec(w1Prev, 0, bias, 0);
            final int row = i * nVocab;
            if (d2t == null) {
                for (int v = 0; v < nVocab; v++) {
                    logits[row + v] += bias[v];
                }
            } else {
                float[] full = new float[nVocab];
                for (int r = 0; r < d2t.length; r++) {
                    full[d2t[r]] = bias[r];
                }
                for (int v = 0; v < nVocab; v++) {
                    logits[row + v] += full[v];
                }
            }
            if (conf != null) {
                System.arraycopy(tEmbd, i * nEmbd, feat, 0, nEmbd);
                System.arraycopy(w1Prev, 0, feat, nEmbd, rank);
                confProj.matvec(feat, 0, c, 0);
                conf[i] = ActMath.sigmoid(confBias != null ? c[0] + confBias[0] : c[0]);
            }
            if (i + 1 < nTokens) {
                prev = CpuOps.argmax(logits, row, nVocab);
            }
        }
        return conf;
    }
}
