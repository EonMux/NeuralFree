package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.act.GluMath;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionCore;
import com.eonmux.neuralfree.gguf.ops.dsv4.CpuDsv4Compressor;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels;
import com.eonmux.neuralfree.gguf.ops.hc.CpuHyperConnections;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeRouting;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * The DeepSeek-V4 DSpark decoder stack of a dflash draft (dflash.cpp:1108-1280), with the deepseek4 blocks it
 * reuses (deepseek4.cpp:269-468 hyper-connections, :885-1229 attention on the uncompressed path):
 * <ul>
 *   <li>Injection: {@code kv = rms(wkv . g) * attn_kv_a_norm}, the trailing {@code n_rot} dims rotated
 *       ({@code freq_base}, scale 1, no YaRN), stored as a single shared K=V head (dflash.cpp:1140-1158).</li>
 *   <li>Noise block: the token embeddings repeated over {@code hc} streams (no embedding scale); per layer the mHC
 *       pre-mix ({@code pre = sigmoid(mixes*scale + base) + eps}, {@code post = 2 sigmoid(.)}, Sinkhorn comb) over
 *       {@code hc_attn_*}, {@code rms * attn_norm}, attention, mHC post-mix, the same around
 *       {@code rms * ffn_norm} and the MoE (sqrt-softplus router with {@code exp_probs_b}, clamped SwiGLU) plus the
 *       shared expert; then the HC head collapse, {@code t_embd} (pre-norm), {@code rms * output_norm} and the
 *       output head.</li>
 *   <li>Attention: {@code q = wq_b . (rms(wq_a . x) * attn_q_a_norm)}, weightless per-head RMS, trailing-dim RoPE;
 *       {@code kv} as in injection; softmax over the cached rows ({@code 1/sqrt(head)}, sinks, sliding window, the
 *       block's causal flag) with V = K; the output's trailing dims rotated back; grouped {@code wo_a} then
 *       {@code wo_b}.</li>
 *   <li>A quantized cache rotates Q and the stored K rows after RoPE and the attention output back before the
 *       inverse RoPE (dflash.cpp:1154-1156, llama-graph.cpp:3593-3631).</li>
 * </ul>
 * Not thread-safe.
 */
final class DflashDsv4Backbone implements DflashStack {

    private static final float CLAMP_EPS = 1e-6f;

    private static final class Layer {
        float[] attnNorm;
        float[] sinks;
        DflashMatrix wqA;
        float[] qANorm;
        DflashMatrix wqB;
        DflashMatrix wkv;
        float[] kvNorm;
        DflashMatrix woA;
        DflashMatrix woB;
        DflashMatrix hcAttnFn;
        float[] hcAttnBase;
        float[] hcAttnScale;
        DflashMatrix hcFfnFn;
        float[] hcFfnBase;
        float[] hcFfnScale;
        DflashMatrix gateInp;
        float[] expProbsB;
        float[] ffnNorm;
        DflashMatrix gateExps;
        DflashMatrix upExps;
        DflashMatrix downExps;
        DflashMatrix gateShexp;
        DflashMatrix upShexp;
        DflashMatrix downShexp;
        CpuAttentionCore.Params attn;
        MoeRouting.Params router;
    }

    /** Normed sub-block input of one mHC pre-mix and the post / comb coefficients of its post-mix. */
    private static final class HcMix {
        final float[] input;
        final float[] post;
        final float[] comb;

        HcMix(float[] input, float[] post, float[] comb) {
            this.input = input;
            this.post = post;
            this.comb = comb;
        }
    }

    private final DflashSpec spec;
    private final DflashKvCache cache;
    private final DflashMatrix tokEmbd;
    private final DflashMatrix output;
    private final float[] outputNorm;
    private final DflashMatrix hcHeadFn;
    private final float[] hcHeadBase;
    private final float[] hcHeadScale;
    private final Layer[] layers;
    private final Dsv4Kernels.RopeParams rope;
    private final HcKernels.CoefficientParams mhc;
    private final HcKernels.CoefficientParams head;
    private final int nEmbd;
    private final int hc;
    private final int headDim;
    private final int nHead;
    private final HadamardSpec rotK;

    /**
     * @throws IllegalArgumentException when a required tensor is missing or mis-shaped
     */
    DflashDsv4Backbone(DflashSpec spec, RopeType ropeType, CpuWeights draft, DflashMatrix tokEmbd,
                       DflashMatrix output, DflashKvCache cache) {
        this.spec = spec;
        this.cache = cache;
        this.tokEmbd = tokEmbd;
        this.output = output;
        this.nEmbd = spec.nEmbd();
        this.hc = spec.hc();
        this.headDim = spec.headDim();
        this.nHead = spec.nHead(0);
        this.rotK = cache.rotationK();
        HcKernels.validateStreams(hc);
        this.mhc = HcKernels.CoefficientParams.mhc(hc, spec.hcEps(), spec.hcSinkhornIters());
        this.head = HcKernels.CoefficientParams.head(hc, spec.hcEps());
        this.rope = Dsv4Kernels.RopeParams.of(spec.nRot(), ropeType, spec.ropeFreqBase(), 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        this.outputNorm = DflashBackbone.vector(draft.floats(TensorNames.weight(TensorNames.OUTPUT_NORM)), nEmbd,
                TensorNames.OUTPUT_NORM);
        this.hcHeadFn = DflashMatrix.require(draft, TensorNames.weight(TensorNames.HC_HEAD_FN))
                .expect(head.mixDim(), (long) hc * nEmbd);
        this.hcHeadBase = DflashBackbone.vector(draft.floats(TensorNames.weight(TensorNames.HC_HEAD_BASE)),
                head.baseCount(), TensorNames.HC_HEAD_BASE);
        this.hcHeadScale = DflashBackbone.vector(draft.floats(TensorNames.weight(TensorNames.HC_HEAD_SCALE)),
                head.scaleCount(), TensorNames.HC_HEAD_SCALE);
        this.layers = new Layer[spec.nLayer()];
        for (int il = 0; il < layers.length; il++) {
            layers[il] = loadLayer(il, draft);
        }
    }

    private Layer loadLayer(int il, CpuWeights w) {
        Layer l = new Layer();
        final int qWidth = nHead * headDim;
        final int oGroupDim = qWidth / spec.oGroups();
        final int mixDim = mhc.mixDim();
        final int hcDim = hc * nEmbd;
        l.attnNorm = floats(w, il, TensorNames.ATTN_NORM, nEmbd);
        l.sinks = floats(w, il, TensorNames.ATTN_SINKS, nHead);
        l.wqA = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_Q_A)).expect(spec.nLoraQ(), nEmbd);
        l.qANorm = floats(w, il, TensorNames.ATTN_Q_A_NORM, spec.nLoraQ());
        l.wqB = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_Q_B)).expect(qWidth, spec.nLoraQ());
        l.wkv = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_KV)).expect(headDim, nEmbd);
        l.kvNorm = floats(w, il, TensorNames.ATTN_KV_NORM, headDim);
        l.woA = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_OUT_A))
                .expect((long) spec.oLoraRank() * spec.oGroups(), oGroupDim);
        l.woB = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_OUT_B))
                .expect(nEmbd, (long) spec.oGroups() * spec.oLoraRank());
        l.hcAttnFn = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.HC_ATTN_FN)).expect(mixDim, hcDim);
        l.hcAttnBase = floats(w, il, TensorNames.HC_ATTN_BASE, mhc.baseCount());
        l.hcAttnScale = floats(w, il, TensorNames.HC_ATTN_SCALE, mhc.scaleCount());
        l.hcFfnFn = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.HC_FFN_FN)).expect(mixDim, hcDim);
        l.hcFfnBase = floats(w, il, TensorNames.HC_FFN_BASE, mhc.baseCount());
        l.hcFfnScale = floats(w, il, TensorNames.HC_FFN_SCALE, mhc.scaleCount());
        l.gateInp = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_GATE_INP))
                .expect(spec.nExpert(), nEmbd);
        float[] b = w.layerFloatsOrNull(il, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B));
        if (b == null) {
            throw new IllegalArgumentException("dflash drafter: model has no tensor 'blk." + il + "."
                    + TensorNames.bias(TensorNames.FFN_EXP_PROBS_B) + "'");
        }
        l.expProbsB = DflashBackbone.vector(b, spec.nExpert(), "blk." + il + "." + TensorNames.FFN_EXP_PROBS_B);
        l.ffnNorm = floats(w, il, TensorNames.FFN_NORM, nEmbd);
        l.gateExps = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_GATE_EXPS));
        l.upExps = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_UP_EXPS))
                .expect(l.gateExps.rows(), nEmbd);
        l.downExps = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_DOWN_EXPS));
        l.gateShexp = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_GATE_SHEXP));
        l.upShexp = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_UP_SHEXP))
                .expect(l.gateShexp.rows(), nEmbd);
        l.downShexp = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_DOWN_SHEXP))
                .expect(nEmbd, l.gateShexp.rows());
        if (l.gateExps.rows() % spec.nExpert() != 0 || l.downExps.rows() != nEmbd * spec.nExpert()) {
            throw new IllegalArgumentException("dflash drafter: expert tensors of layer " + il + " do not hold "
                    + spec.nExpert() + " experts: " + l.gateExps + ", " + l.downExps);
        }
        l.attn = CpuAttentionCore.Params.builder(nHead)
                .scale(1.0f / (float) Math.sqrt(headDim))
                .window(cache.window(il))
                .sinks(l.sinks)
                .build();
        l.router = new MoeRouting.Params(spec.nExpert(), spec.nExpertUsed(il), GatingFunc.SQRT_SOFTPLUS, 1, 1,
                spec.expertWeightsNorm(), spec.expertWeightsScale(), true, MoeKernels.SEL_PROBS, 0);
        return l;
    }

    @Override
    public void inject(float[] inpG, int nRows, int pos0) {
        float[] kv = new float[nRows * headDim];
        for (int il = 0; il < layers.length; il++) {
            keyRows(layers[il], inpG, nRows, pos0, kv);
            for (int t = 0; t < nRows; t++) {
                cache.write(il, pos0 + t, kv, t * headDim, null, 0);
            }
        }
    }

    @Override
    public Block forward(int[] tokens, int pos0) {
        final int n = tokens.length;
        final int hcDim = hc * nEmbd;
        float[] embd = new float[n * nEmbd];
        for (int t = 0; t < n; t++) {
            tokEmbd.row(tokens[t], embd, t * nEmbd);
        }
        float[] streams = new float[n * hcDim];
        CpuHyperConnections.expandStreams(hc, nEmbd, n, embd, 0, streams, 0);
        int[] qPos = new int[n];
        for (int t = 0; t < n; t++) {
            qPos[t] = pos0 + t;
        }
        for (int il = 0; il < layers.length; il++) {
            final Layer l = layers[il];
            HcMix attnMix = hcPre(l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, streams, n, l.attnNorm);
            hcPost(attention(il, l, attnMix.input, n, pos0, qPos), attnMix, streams, n);
            HcMix ffnMix = hcPre(l.hcFfnFn, l.hcFfnScale, l.hcFfnBase, streams, n, l.ffnNorm);
            hcPost(ffn(il, l, ffnMix.input, n), ffnMix, streams, n);
        }
        float[] tEmbd = collapse(streams, n);
        float[] normed = new float[n * nEmbd];
        CpuOps.rmsNorm(tEmbd, 0, normed, 0, n, nEmbd, outputNorm, spec.rmsEps());
        final int vocab = output.rows();
        float[] logits = new float[n * vocab];
        output.matmul(normed, 0, logits, 0, n);
        return new Block(n, logits, vocab, tEmbd, nEmbd);
    }

    /**
     * mHC pre-mix of one sub-block (deepseek4.cpp:353-409): mixes of the weightless-RMS-normed streams, the
     * pre / post / comb coefficients, the pre-weighted stream sum, then {@code rms * norm}.
     */
    private HcMix hcPre(DflashMatrix fn, float[] scale, float[] base, float[] streams, int n, float[] norm) {
        float[] mixes = mixes(fn, streams, n, mhc.mixDim());
        float[] pre = new float[n * hc];
        float[] post = new float[n * hc];
        float[] comb = new float[n * hc * hc];
        CpuHyperConnections.coefficients(mhc, 0, n, mixes, 0, scale, base, pre, 0, post, 0, comb, 0);
        float[] cur = new float[n * nEmbd];
        CpuHyperConnections.pre(HcKernels.PreParams.weights(), hc, nEmbd, n, streams, 0, pre, 0, cur, 0);
        CpuOps.rmsNorm(cur, 0, cur, 0, n, nEmbd, norm, spec.rmsEps());
        return new HcMix(cur, post, comb);
    }

    /**
     * mHC post-mix in place (deepseek4.cpp:411-446):
     * {@code streams[dst] = y * post[dst] + sum_src streams[src] * comb[dst + hc*src]}.
     */
    private void hcPost(float[] y, HcMix mix, float[] streams, int n) {
        CpuHyperConnections.post(HcKernels.PostParams.mixing(), hc, nEmbd, n, y, 0, streams, 0, mix.post, 0,
                mix.comb, 0, streams, 0);
    }

    /** HC head (deepseek4.cpp:448-468): pre-only coefficients from {@code output_hc_*}, then the weighted sum. */
    private float[] collapse(float[] streams, int n) {
        float[] mixes = mixes(hcHeadFn, streams, n, head.mixDim());
        float[] pre = new float[n * hc];
        CpuHyperConnections.coefficients(head, 0, n, mixes, 0, hcHeadScale, hcHeadBase, pre, 0, null, 0, null, 0);
        float[] out = new float[n * nEmbd];
        CpuHyperConnections.pre(HcKernels.PreParams.weights(), hc, nEmbd, n, streams, 0, pre, 0, out, 0);
        return out;
    }

    /** {@code fn . rms(flat streams)} per token, the RMS weightless over all {@code hc * n_embd} values. */
    private float[] mixes(DflashMatrix fn, float[] streams, int n, int mixDim) {
        final int hcDim = hc * nEmbd;
        float[] flat = new float[n * hcDim];
        CpuOps.rmsNorm(streams, 0, flat, 0, n, hcDim, null, spec.rmsEps());
        float[] mixes = new float[n * mixDim];
        fn.matmul(flat, 0, mixes, 0, n);
        return mixes;
    }

    /**
     * Shared K=V rows {@code rope(rms(wkv . x) * kv_norm)} of {@code n} rows into {@code kv}, rotated for a quantized
     * cache.
     */
    private void keyRows(Layer l, float[] x, int n, int pos0, float[] kv) {
        l.wkv.matmul(x, 0, kv, 0, n);
        CpuOps.rmsNorm(kv, 0, kv, 0, n, headDim, l.kvNorm, spec.rmsEps());
        for (int t = 0; t < n; t++) {
            CpuDsv4Compressor.rope(kv, t * headDim, headDim, rope, pos0 + t, false);
        }
        if (rotK != null) {
            CpuHadamard.applyInPlace(rotK, kv, 0, n, headDim);
        }
    }

    private float[] attention(int il, Layer l, float[] x, int n, int pos0, int[] qPos) {
        final int qWidth = nHead * headDim;
        final float eps = spec.rmsEps();
        float[] qr = new float[n * spec.nLoraQ()];
        l.wqA.matmul(x, 0, qr, 0, n);
        CpuOps.rmsNorm(qr, 0, qr, 0, n, spec.nLoraQ(), l.qANorm, eps);
        float[] q = new float[n * qWidth];
        l.wqB.matmul(qr, 0, q, 0, n);
        NormMath.rowNorm(NormKind.RMS, eps, n * nHead, headDim, headDim, 1, q, 0, null, null, q, 0);
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < nHead; h++) {
                CpuDsv4Compressor.rope(q, t * qWidth + h * headDim, headDim, rope, pos0 + t, false);
            }
        }
        if (rotK != null) {
            CpuHadamard.applyInPlace(rotK, q, 0, n, qWidth);
        }
        float[] kv = new float[n * headDim];
        keyRows(l, x, n, pos0, kv);
        for (int t = 0; t < n; t++) {
            cache.write(il, pos0 + t, kv, t * headDim, null, 0);
        }
        float[] out = new float[n * qWidth];
        CpuAttentionCore.attendRows(l.attn, n, q, 0, qWidth, qPos, 0, cache.layer(il), cache.rowPositions(il), 0,
                cache.rowEnd(il), out, 0, qWidth);
        if (rotK != null) {
            CpuHadamard.applyInPlace(rotK, out, 0, n, qWidth);
        }
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < nHead; h++) {
                CpuDsv4Compressor.rope(out, t * qWidth + h * headDim, headDim, rope, pos0 + t, true);
            }
        }
        final int groups = spec.oGroups();
        final int rank = spec.oLoraRank();
        final int groupDim = qWidth / groups;
        float[] oa = new float[n * groups * rank];
        for (int t = 0; t < n; t++) {
            for (int g = 0; g < groups; g++) {
                l.woA.matvecRows((long) g * rank, rank, out, t * qWidth + g * groupDim, oa, (t * groups + g) * rank);
            }
        }
        float[] y = new float[n * nEmbd];
        l.woB.matmul(oa, 0, y, 0, n);
        return y;
    }

    /** Routed experts plus the shared expert (dflash.cpp:1226-1246). */
    private float[] ffn(int il, Layer l, float[] x, int n) {
        final int nExpert = spec.nExpert();
        float[] logits = new float[n * nExpert];
        l.gateInp.matmul(x, 0, logits, 0, n);
        MoeRouting.Selection sel = MoeRouting.route(l.router, logits, n, l.expProbsB);
        final int nFfExp = l.gateExps.rows() / nExpert;
        final float limitExp = spec.clampExp(il);
        float[] g = new float[nFfExp];
        float[] u = new float[nFfExp];
        float[] d = new float[nEmbd];
        float[] y = new float[n * nEmbd];
        for (int t = 0; t < n; t++) {
            final int xOff = t * nEmbd;
            for (int j = 0; j < sel.k(); j++) {
                final int e = sel.id(t, j);
                final float wgt = sel.weight(t, j);
                l.gateExps.matvecExpert(e, nExpert, x, xOff, g, 0);
                l.upExps.matvecExpert(e, nExpert, x, xOff, u, 0);
                glu(g, u, limitExp);
                l.downExps.matvecExpert(e, nExpert, g, 0, d, 0);
                for (int i = 0; i < nEmbd; i++) {
                    y[xOff + i] += d[i] * wgt;
                }
            }
        }
        final int nFfSh = l.gateShexp.rows();
        float[] gs = new float[n * nFfSh];
        float[] us = new float[n * nFfSh];
        l.gateShexp.matmul(x, 0, gs, 0, n);
        l.upShexp.matmul(x, 0, us, 0, n);
        glu(gs, us, spec.clampShexp(il));
        float[] sh = new float[n * nEmbd];
        l.downShexp.matmul(gs, 0, sh, 0, n);
        for (int i = 0; i < y.length; i++) {
            y[i] += sh[i];
        }
        return y;
    }

    /**
     * {@code g = swiglu_clamp(g, u, limit)} when {@code limit > 1e-6}, else {@code silu(g) * u}
     * (llama-graph.cpp:2003-2027 and :2424-2452 for the DSV4 DSpark draft).
     */
    private static void glu(float[] g, float[] u, float limit) {
        final boolean clamp = limit > CLAMP_EPS;
        for (int i = 0; i < g.length; i++) {
            g[i] = clamp ? GluMath.swigluClamp(g[i], u[i], limit) : ActMath.silu(g[i]) * u[i];
        }
    }

    private static float[] floats(CpuWeights w, int il, String base, int len) {
        float[] v = w.layerFloatsOrNull(il, TensorNames.weight(base));
        if (v == null) {
            throw new IllegalArgumentException("dflash drafter: model has no tensor 'blk." + il + "."
                    + TensorNames.weight(base) + "'");
        }
        return DflashBackbone.vector(v, len, "blk." + il + "." + base);
    }
}
