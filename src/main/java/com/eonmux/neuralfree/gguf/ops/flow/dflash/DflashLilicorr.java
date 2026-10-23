package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * LiLiCorr lattice scorer, {@code build_lilicorr_scores} (dflash.cpp:641-801), for one block of
 * {@code block_size} positions. Slot {@code s} is block position {@code s + 1}; node {@code (s, k)} is its
 * {@code k}-th candidate (descending logit). Per node:
 * <pre>
 *   feature = [log p, p, log p - log p_top, k / (topK - 1), k == 0]       p = softmax(logits)[cand]
 *   x       = token(cand) + pass(t_embd[s + 1]) + mlp3(silu(mlp1(layernorm(feature)))) + slot[s] + rank[k]
 * </pre>
 * where {@code token} is the embedding (through {@code token_proj} when the head is narrower than the model) and
 * {@code pass} is {@code pass_hidden_proj}. The nodes run {@code lilicorr.num_layers} pre-norm transformer layers
 * (full attention between all nodes with the learned bias {@code attention_bias[h][q][k]}, SiLU MLP), then
 * {@code rms * output_norm}. With {@code a = rms(context_proj . anchor) * anchor_norm} for the fused target
 * feature {@code anchor} of the position before the block, {@code f = silu(factor_input_proj . [x, a, x*a])},
 * {@code out = l2(out_head . f)}, {@code in = l2(in_head . f)} and {@code ao = l2(anchor_out_head . a)}:
 * <pre>
 *   score(1, j, .) = logit_scale * in(0, j) . ao
 *   score(s + 1, j, i) = logit_scale * in(s, j) . out(s - 1, i)      for s >= 1
 * </pre>
 * Not thread-safe.
 */
final class DflashLilicorr {

    private static final int FEATURES = 5;
    private static final float FEATURE_NORM_EPS = 1.0e-5f;

    private static final class Layer {
        float[] attnNorm;
        float[] mlpNorm;
        DflashMatrix inProj;
        float[] inProjB;
        DflashMatrix outProj;
        float[] outProjB;
        DflashMatrix mlp0;
        float[] mlp0B;
        DflashMatrix mlp2;
        float[] mlp2B;
    }

    private final DflashMatrix tokEmbd;
    private final DflashMatrix tokenProj;
    private final float[] tokenProjB;
    private final DflashMatrix passProj;
    private final float[] passProjB;
    private final float[] featureNorm;
    private final float[] featureNormB;
    private final DflashMatrix featureMlp1;
    private final float[] featureMlp1B;
    private final DflashMatrix featureMlp3;
    private final float[] featureMlp3B;
    private final float[] slotEmbd;
    private final float[] rankEmbd;
    private final float[] attnBias;
    private final DflashMatrix contextProj;
    private final float[] contextProjB;
    private final Layer[] layers;
    private final float[] outputNorm;
    private final float[] anchorNorm;
    private final DflashMatrix factorIn;
    private final float[] factorInB;
    private final DflashMatrix outHead;
    private final float[] outHeadB;
    private final DflashMatrix inHead;
    private final float[] inHeadB;
    private final DflashMatrix anchorOutHead;
    private final float[] anchorOutHeadB;

    private final int nEmbd;
    private final int nVocab;
    private final int hidden;
    private final int topK;
    private final int heads;
    private final int slots;
    private final int nodes;
    private final int factorDim;
    private final float rmsEps;
    private final float vectorEps;
    private final float logitScale;

    /**
     * @param tokEmbd the token embeddings of the noise block (the draft's or the target's)
     * @throws IllegalArgumentException when a tensor is missing or mis-shaped
     */
    DflashLilicorr(DflashSpec spec, CpuWeights w, DflashMatrix tokEmbd) {
        this.tokEmbd = tokEmbd;
        this.nEmbd = spec.nEmbd();
        this.nVocab = spec.nVocab();
        this.hidden = spec.lilicorrHidden();
        this.topK = spec.lilicorrTopK();
        this.heads = spec.lilicorrHeads();
        this.slots = spec.graphBlockSize() - 1;
        this.nodes = slots * topK;
        this.factorDim = spec.lilicorrFactorDim();
        this.rmsEps = spec.rmsEps();
        this.vectorEps = spec.lilicorrVectorEps();
        this.logitScale = spec.lilicorrLogitScale();
        final int mlp = spec.lilicorrMlpHidden();
        final int featureBid = spec.nLayerAll() + spec.lilicorrLayers();

        if (hidden != nEmbd) {
            tokenProj = DflashMatrix.require(w, TensorNames.weight(TensorNames.LILICORR_TOKEN_PROJ)).expect(hidden, nEmbd);
            tokenProjB = model(w, TensorNames.bias(TensorNames.LILICORR_TOKEN_PROJ), hidden);
        } else {
            tokenProj = null;
            tokenProjB = null;
        }
        passProj = DflashMatrix.require(w, TensorNames.weight(TensorNames.LILICORR_PASS_HIDDEN_PROJ))
                .expect(hidden, nEmbd);
        passProjB = model(w, TensorNames.bias(TensorNames.LILICORR_PASS_HIDDEN_PROJ), hidden);
        featureNorm = layer(w, featureBid, TensorNames.weight(TensorNames.FFN_NORM), FEATURES);
        featureNormB = layer(w, featureBid, TensorNames.bias(TensorNames.FFN_NORM), FEATURES);
        featureMlp1 = DflashMatrix.layer(w, featureBid, TensorNames.weight(TensorNames.FFN_UP)).expect(hidden, FEATURES);
        featureMlp1B = layer(w, featureBid, TensorNames.bias(TensorNames.FFN_UP), hidden);
        featureMlp3 = DflashMatrix.layer(w, featureBid, TensorNames.weight(TensorNames.FFN_DOWN)).expect(hidden, hidden);
        featureMlp3B = layer(w, featureBid, TensorNames.bias(TensorNames.FFN_DOWN), hidden);
        slotEmbd = model(w, TensorNames.weight(TensorNames.LILICORR_SLOT_EMBD), slots * hidden);
        rankEmbd = model(w, TensorNames.weight(TensorNames.LILICORR_RANK_EMBD), topK * hidden);
        attnBias = model(w, TensorNames.bias(TensorNames.LILICORR_ATTN_BIAS), nodes * nodes * heads);
        contextProj = DflashMatrix.require(w, TensorNames.weight(TensorNames.LILICORR_CONTEXT_PROJ))
                .expect(hidden, nEmbd);
        contextProjB = model(w, TensorNames.bias(TensorNames.LILICORR_CONTEXT_PROJ), hidden);
        layers = new Layer[spec.lilicorrLayers()];
        for (int i = 0; i < layers.length; i++) {
            final int bid = spec.nLayerAll() + i;
            Layer l = new Layer();
            l.attnNorm = layer(w, bid, TensorNames.weight(TensorNames.ATTN_NORM), hidden);
            l.mlpNorm = layer(w, bid, TensorNames.weight(TensorNames.FFN_NORM), hidden);
            l.inProj = DflashMatrix.stack(w, TensorNames.STACK_LILICORR, i,
                    TensorNames.weight(TensorNames.LILICORR_ATTN_IN_PROJ)).expect(3L * hidden, hidden);
            l.inProjB = DflashBackbone.vector(requireVector(w.stackFloatsOrNull(TensorNames.STACK_LILICORR, i,
                    TensorNames.bias(TensorNames.LILICORR_ATTN_IN_PROJ)), TensorNames.STACK_LILICORR + "." + i + "."
                    + TensorNames.bias(TensorNames.LILICORR_ATTN_IN_PROJ)), 3 * hidden, "lilicorr in_proj bias");
            l.outProj = DflashMatrix.layer(w, bid, TensorNames.weight(TensorNames.ATTN_OUT)).expect(hidden, hidden);
            l.outProjB = layer(w, bid, TensorNames.bias(TensorNames.ATTN_OUT), hidden);
            l.mlp0 = DflashMatrix.layer(w, bid, TensorNames.weight(TensorNames.FFN_UP)).expect(mlp, hidden);
            l.mlp0B = layer(w, bid, TensorNames.bias(TensorNames.FFN_UP), mlp);
            l.mlp2 = DflashMatrix.layer(w, bid, TensorNames.weight(TensorNames.FFN_DOWN)).expect(hidden, mlp);
            l.mlp2B = layer(w, bid, TensorNames.bias(TensorNames.FFN_DOWN), hidden);
            layers[i] = l;
        }
        outputNorm = model(w, TensorNames.weight(TensorNames.LILICORR_OUTPUT_NORM), hidden);
        anchorNorm = model(w, TensorNames.weight(TensorNames.LILICORR_ANCHOR_NORM), hidden);
        factorIn = DflashMatrix.require(w, TensorNames.weight(TensorNames.LILICORR_FACTOR_INPUT_PROJ))
                .expect(hidden, 3L * hidden);
        factorInB = model(w, TensorNames.bias(TensorNames.LILICORR_FACTOR_INPUT_PROJ), hidden);
        outHead = DflashMatrix.require(w, TensorNames.weight(TensorNames.LILICORR_OUT_HEAD)).expect(factorDim, hidden);
        outHeadB = model(w, TensorNames.bias(TensorNames.LILICORR_OUT_HEAD), factorDim);
        inHead = DflashMatrix.require(w, TensorNames.weight(TensorNames.LILICORR_IN_HEAD)).expect(factorDim, hidden);
        inHeadB = model(w, TensorNames.bias(TensorNames.LILICORR_IN_HEAD), factorDim);
        anchorOutHead = DflashMatrix.require(w, TensorNames.weight(TensorNames.LILICORR_ANCHOR_OUT_HEAD))
                .expect(factorDim, hidden);
        anchorOutHeadB = model(w, TensorNames.bias(TensorNames.LILICORR_ANCHOR_OUT_HEAD), factorDim);
    }

    /**
     * Scores one block.
     *
     * @param logits target-vocabulary logits {@code [blockSize][nVocab]}
     * @param tEmbd  {@code t_embd} rows {@code [blockSize][nEmbd]}
     * @param anchor fused target feature ({@code nEmbd} values) of the position before the block
     * @return the lattice with rows {@code 1 .. blockSize - 1}
     */
    DflashSelector.Lattice score(float[] logits, float[] tEmbd, float[] anchor) {
        DflashSelector.Lattice lattice = new DflashSelector.Lattice(topK, slots + 1);
        float[] x = nodeInputs(logits, tEmbd, lattice);
        for (Layer l : layers) {
            block(l, x);
        }
        CpuOps.rmsNorm(x, 0, x, 0, nodes, hidden, outputNorm, rmsEps);

        float[] a = new float[hidden];
        contextProj.matvec(anchor, 0, a, 0);
        addBias(a, 0, contextProjB);
        NormMath.rowNorm(NormKind.RMS, rmsEps, a, 0, a, 0, hidden, anchorNorm, 0, null, 0);

        float[] fin = new float[nodes * 3 * hidden];
        for (int nd = 0; nd < nodes; nd++) {
            final int o = nd * 3 * hidden;
            System.arraycopy(x, nd * hidden, fin, o, hidden);
            System.arraycopy(a, 0, fin, o + hidden, hidden);
            for (int i = 0; i < hidden; i++) {
                fin[o + 2 * hidden + i] = x[nd * hidden + i] * a[i];
            }
        }
        float[] f = new float[nodes * hidden];
        factorIn.matmul(fin, 0, f, 0, nodes);
        for (int nd = 0; nd < nodes; nd++) {
            addBias(f, nd * hidden, factorInB);
        }
        silu(f);
        float[] out = head(outHead, outHeadB, f, nodes);
        float[] in = head(inHead, inHeadB, f, nodes);
        float[] ao = head(anchorOutHead, anchorOutHeadB, a, 1);

        for (int j = 0; j < topK; j++) {
            final float v = logitScale * dot(in, j * factorDim, ao, 0, factorDim);
            for (int i = 0; i < topK; i++) {
                lattice.setScore(1, j, i, i == 0 ? v : 0.0f);
            }
        }
        for (int s = 1; s < slots; s++) {
            for (int i = 0; i < topK; i++) {
                for (int j = 0; j < topK; j++) {
                    lattice.setScore(s + 1, j, i, logitScale
                            * dot(in, (s * topK + j) * factorDim, out, ((s - 1) * topK + i) * factorDim, factorDim));
                }
            }
        }
        return lattice;
    }

    /** Candidates and node inputs {@code x} ({@code [nodes][hidden]}); fills the lattice candidates. */
    private float[] nodeInputs(float[] logits, float[] tEmbd, DflashSelector.Lattice lattice) {
        int[] cand = new int[topK];
        float[] probs = new float[nVocab];
        float[] feature = new float[nodes * FEATURES];
        float[] tokens = new float[nodes * nEmbd];
        for (int s = 0; s < slots; s++) {
            final int lOff = (s + 1) * nVocab;
            CpuOps.topK(logits, lOff, nVocab, topK, cand, 0);
            System.arraycopy(logits, lOff, probs, 0, nVocab);
            CpuOps.softmax(probs, 0, nVocab);
            final float lpTop = (float) Math.log(probs[cand[0]]);
            for (int k = 0; k < topK; k++) {
                lattice.setCandidate(s + 1, k, cand[k]);
                final int nd = s * topK + k;
                final float p = probs[cand[k]];
                final float lp = (float) Math.log(p);
                final int fo = nd * FEATURES;
                feature[fo] = lp;
                feature[fo + 1] = p;
                feature[fo + 2] = lp - lpTop;
                feature[fo + 3] = topK == 1 ? 0.0f : k * (1.0f / (topK - 1));
                feature[fo + 4] = k == 0 ? 1.0f : 0.0f;
                tokEmbd.row(cand[k], tokens, nd * nEmbd);
            }
        }
        float[] x;
        if (tokenProj != null) {
            x = new float[nodes * hidden];
            tokenProj.matmul(tokens, 0, x, 0, nodes);
            for (int nd = 0; nd < nodes; nd++) {
                addBias(x, nd * hidden, tokenProjB);
            }
        } else {
            x = tokens;
        }
        float[] pass = new float[slots * hidden];
        passProj.matmul(tEmbd, nEmbd, pass, 0, slots);
        float[] fs = featureStates(feature);
        for (int s = 0; s < slots; s++) {
            for (int k = 0; k < topK; k++) {
                final int o = (s * topK + k) * hidden;
                for (int i = 0; i < hidden; i++) {
                    float v = x[o + i] + (pass[s * hidden + i] + passProjB[i]);
                    v += fs[o + i];
                    x[o + i] = v + (slotEmbd[s * hidden + i] + rankEmbd[k * hidden + i]);
                }
            }
        }
        return x;
    }

    /** {@code mlp3(silu(mlp1(layernorm(feature) * w + b) + b1)) + b3} per node. */
    private float[] featureStates(float[] feature) {
        CpuOps.layerNorm(feature, 0, feature, 0, nodes, FEATURES, featureNorm, featureNormB, FEATURE_NORM_EPS);
        float[] h = new float[nodes * hidden];
        featureMlp1.matmul(feature, 0, h, 0, nodes);
        for (int nd = 0; nd < nodes; nd++) {
            addBias(h, nd * hidden, featureMlp1B);
        }
        silu(h);
        float[] out = new float[nodes * hidden];
        featureMlp3.matmul(h, 0, out, 0, nodes);
        for (int nd = 0; nd < nodes; nd++) {
            addBias(out, nd * hidden, featureMlp3B);
        }
        return out;
    }

    /** One transformer layer over the nodes, in place (dflash.cpp:734-756). */
    private void block(Layer l, float[] x) {
        final int headDim = hidden / heads;
        final float scale = 1.0f / (float) Math.sqrt(headDim);
        float[] h = new float[nodes * hidden];
        CpuOps.rmsNorm(x, 0, h, 0, nodes, hidden, l.attnNorm, rmsEps);
        float[] qkv = new float[nodes * 3 * hidden];
        l.inProj.matmul(h, 0, qkv, 0, nodes);
        for (int nd = 0; nd < nodes; nd++) {
            addBias(qkv, nd * 3 * hidden, l.inProjB);
        }
        float[] joined = new float[nodes * hidden];
        float[] p = new float[nodes];
        for (int hd = 0; hd < heads; hd++) {
            final int qo = hd * headDim;
            final int ko = hidden + hd * headDim;
            final int vo = 2 * hidden + hd * headDim;
            for (int q = 0; q < nodes; q++) {
                final int biasRow = (hd * nodes + q) * nodes;
                for (int k = 0; k < nodes; k++) {
                    final float kq = dot(qkv, k * 3 * hidden + ko, qkv, q * 3 * hidden + qo, headDim);
                    p[k] = kq * scale + attnBias[biasRow + k];
                }
                CpuOps.softmax(p, 0, nodes);
                final int out = q * hidden + hd * headDim;
                for (int k = 0; k < nodes; k++) {
                    final float pk = p[k];
                    final int vr = k * 3 * hidden + vo;
                    for (int d = 0; d < headDim; d++) {
                        joined[out + d] += qkv[vr + d] * pk;
                    }
                }
            }
        }
        float[] y = new float[nodes * hidden];
        l.outProj.matmul(joined, 0, y, 0, nodes);
        for (int nd = 0; nd < nodes; nd++) {
            addBias(y, nd * hidden, l.outProjB);
        }
        add(x, y);
        CpuOps.rmsNorm(x, 0, h, 0, nodes, hidden, l.mlpNorm, rmsEps);
        final int mlp = l.mlp0.rows();
        float[] m = new float[nodes * mlp];
        l.mlp0.matmul(h, 0, m, 0, nodes);
        for (int nd = 0; nd < nodes; nd++) {
            addBias(m, nd * mlp, l.mlp0B);
        }
        silu(m);
        l.mlp2.matmul(m, 0, y, 0, nodes);
        for (int nd = 0; nd < nodes; nd++) {
            addBias(y, nd * hidden, l.mlp2B);
        }
        add(x, y);
    }

    /** {@code l2(w . f + b)} of {@code n} rows ({@code ggml_l2_norm}: {@code x / max(||x||, eps)}). */
    private float[] head(DflashMatrix w, float[] b, float[] f, int n) {
        float[] y = new float[n * factorDim];
        w.matmul(f, 0, y, 0, n);
        for (int r = 0; r < n; r++) {
            addBias(y, r * factorDim, b);
            NormMath.rowNorm(NormKind.L2_CLAMP, vectorEps, y, r * factorDim, y, r * factorDim, factorDim, null, 0,
                    null, 0);
        }
        return y;
    }

    private static float dot(float[] a, int aOff, float[] b, int bOff, int n) {
        float s = 0.0f;
        for (int i = 0; i < n; i++) {
            s += a[aOff + i] * b[bOff + i];
        }
        return s;
    }

    private static void addBias(float[] x, int off, float[] b) {
        for (int i = 0; i < b.length; i++) {
            x[off + i] += b[i];
        }
    }

    private static void add(float[] acc, float[] x) {
        for (int i = 0; i < acc.length; i++) {
            acc[i] += x[i];
        }
    }

    private static void silu(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = ActMath.silu(x[i]);
        }
    }

    private static float[] model(CpuWeights w, String name, int len) {
        return DflashBackbone.vector(requireVector(w.floatsOrNull(name), name), len, name);
    }

    private static float[] layer(CpuWeights w, int bid, String suffix, int len) {
        String name = "blk." + bid + "." + suffix;
        return DflashBackbone.vector(requireVector(w.layerFloatsOrNull(bid, suffix), name), len, name);
    }

    private static float[] requireVector(float[] v, String name) {
        if (v == null) {
            throw new IllegalArgumentException("dflash drafter: model has no tensor '" + name + "'");
        }
        return v;
    }
}
