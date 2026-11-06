package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * The routed experts of one MoE layer computed on the host, for the pairs whose expert is not in the GPU expert
 * cache: the expert part of {@code CpuMoeFfn} restricted to a subset of the step's pairs (the grouped products of
 * {@link CpuExpertMatmul} with their input transforms and per-expert scales, the activation, the per-expert biases
 * and the fused tensor's up scale). The returned rows are complete expert outputs; the weighted sum and the
 * post-expert norm run on the GPU over all pairs.
 *
 * <p>Immutable after construction; every {@link #run} reuses its scratch. Not thread-safe.</p>
 */
final class HostedMoeExperts {

    private final MoeExpertWeights plan;
    private final MoeExpertWeights.Cpu w;
    private final CpuExpertMatmul products;
    private final int k;
    private final float[] gate;
    private final float[] up;
    private final float[] act;
    private final float[] gathered;
    private final int[] missIds;

    /**
     * The host chain of one layer's routed experts.
     *
     * @param plan   expert set of the layer (its {@code resolve(CpuWeights, LayerSpec)} tensors)
     * @param weights host tensors of the model; their {@code matVec} transforms serve the per-expert scales
     * @param k      routed experts per token
     * @param ubatch largest token count of one step; sizes the scratch
     * @throws IllegalArgumentException for a top-k or ubatch below one
     */
    HostedMoeExperts(MoeExpertWeights plan, CpuWeights weights, int k, int ubatch, LayerSpec layer) {
        this.plan = Objects.requireNonNull(plan, "plan");
        if (k < 1) {
            throw new IllegalArgumentException("top-k must be positive, got " + k);
        }
        if (ubatch < 1) {
            throw new IllegalArgumentException("ubatch must be positive, got " + ubatch);
        }
        this.w = plan.resolve(Objects.requireNonNull(weights, "weights"), layer);
        this.products = new CpuExpertMatmul(weights.matVec());
        this.k = k;
        int pairs = Math.multiplyExact(ubatch, k);
        this.gate = plan.gated ? new float[Math.multiplyExact(pairs, plan.gateUpRows())] : null;
        this.up = plan.gated && !plan.fused ? new float[Math.multiplyExact(pairs, plan.nFf)] : null;
        this.act = new float[Math.multiplyExact(pairs, plan.nFf)];
        this.gathered = new float[Math.multiplyExact(pairs, plan.inDim)];
        this.missIds = new int[pairs];
    }

    /**
     * Computes the expert outputs of the miss pairs and returns them: row {@code q} holds the routed FFN of the
     * expert of pair {@code missPairs[q]} applied to source row {@code missPairs[q] / pairsPerRow} of {@code x}.
     *
     * @param ids        routed expert of every pair of the step ({@code t * k + j})
     * @param missPairs  pair indices of the misses
     * @param missCount  number of misses
     * @param x          source rows of {@code in_dim} values: the tokens' rows ({@code pairsPerRow = k}), or the
     *                   pair rows the weight-before-FFN input scaling produced ({@code pairsPerRow = 1})
     * @param pairsPerRow pairs sharing one source row
     * @return the miss rows of {@code outDim} values each, compact ({@code missCount * outDim})
     * @throws IllegalArgumentException when a view does not hold the pairs or an id is outside the experts
     */
    float[] run(int[] ids, int[] missPairs, int missCount, float[] x, int pairsPerRow) {
        Objects.requireNonNull(ids, "ids");
        if (missCount < 0 || missCount > missPairs.length) {
            throw new IllegalArgumentException("miss count " + missCount + " outside [0, " + missPairs.length + "]");
        }
        float[] out = new float[Math.multiplyExact(missCount, plan.outDim)];
        if (missCount == 0) {
            return out;
        }
        if (pairsPerRow < 1) {
            throw new IllegalArgumentException("pairs per source row must be positive, got " + pairsPerRow);
        }
        int inDim = plan.inDim;
        int nFf = plan.nFf;
        int gu = plan.gateUpRows();
        for (int q = 0; q < missCount; q++) {
            int p = missPairs[q];
            if (p < 0 || p >= ids.length) {
                throw new IllegalArgumentException("miss pair " + p + " outside the step's [0, " + ids.length + ")");
            }
            int e = ids[p];
            if (e < 0 || e >= plan.nExpert) {
                throw new IllegalArgumentException("pair " + p + " routes to expert " + e + " outside [0, "
                        + plan.nExpert + ")");
            }
            missIds[q] = e;
            System.arraycopy(x, (p / pairsPerRow) * inDim, gathered, q * inDim, inDim);
        }
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(missIds, missCount, plan.nExpert);
        if (plan.fused) {
            products.run(w.gateUp, plan.nExpert, g, gathered, 0, 1, gate, 0);
            if (w.fusedUpScale != null) {
                for (int q = 0; q < missCount; q++) {
                    NormMath.mulRowScale(1, gu, w.fusedUpScale, missIds[q], 1, gate, q * gu);
                }
            }
            addBias(missCount, gu, missIds, w.gateUpBias, gate);
            CpuOps.gluRows(plan.glu, plan.params, missCount, nFf, gate, 0, gu, gate, nFf, gu, act, 0, nFf);
        } else if (plan.gated) {
            products.run(w.up, plan.nExpert, g, gathered, 0, 1, up, 0);
            products.run(w.gate, plan.nExpert, g, gathered, 0, 1, gate, 0);
            addBias(missCount, nFf, missIds, w.upBias, up);
            addBias(missCount, nFf, missIds, w.gateBias, gate);
            CpuOps.gluRows(plan.glu, plan.params, missCount, nFf, gate, 0, nFf, up, 0, nFf, act, 0, nFf);
        } else {
            products.run(w.up, plan.nExpert, g, gathered, 0, 1, act, 0);
            addBias(missCount, nFf, missIds, w.upBias, act);
            CpuOps.unary(plan.act, plan.params, act, 0, act, 0, missCount * nFf);
        }
        products.run(w.down, plan.nExpert, g, act, 0, 1, out, 0);
        addBias(missCount, plan.outDim, missIds, w.downBias, out);
        return out;
    }

    /**
     * True when the installed CPU matmul tap ({@link CpuMatVec#askedObserver}) collects the routed products of one of
     * the layer's expert tensors: the pairs computed here reach it exactly as on the CPU engine.
     */
    boolean observed() {
        for (CpuTensorRef t : new CpuTensorRef[]{w.gate, w.up, w.gateUp, w.down}) {
            if (t != null && CpuMatVec.askedObserver(t, true, 1) != null) {
                return true;
            }
        }
        return false;
    }

    /** {@code data[q * dim + d] += bias[ids[q]][d]} over the miss rows. */
    private static void addBias(int count, int dim, int[] ids, float[] bias, float[] data) {
        if (bias != null) {
            MoeCombine.addId(count, dim, ids, bias, data);
        }
    }

    /** Rows of one expert of the down product ({@code plan.outDim}). */
    int outDim() {
        return plan.outDim;
    }

    @Override
    public String toString() {
        return "HostedMoeExperts{" + plan + ", k=" + k + "}";
    }
}
