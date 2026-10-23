package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.moe.CpuExpertMatmul;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCombine;
import com.eonmux.neuralfree.gguf.ops.moe.MoeRouting;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;

import java.util.Objects;

/**
 * CPU twin of {@link GpuValueExperts}: the MoVA value projection {@code build_routed_value}
 * (models/k2-horizon.cpp:137-205) with the CPU router ({@link MoeRouting#route}), the expert products grouped by
 * expert ({@link CpuExpertMatmul}, the twin of {@code ggml_mul_mat_id}) and their LoRA deltas, {@code silu}, and the
 * weighted sum of {@link MoeCombine#weightedSum} in selection order. Working arrays are {@code attn.vexp.*}
 * {@link CpuScratch} temporaries reserved at bind for the ubatch. Not thread-safe.
 */
final class CpuValueExperts {

    private static final String T_LOGITS = "attn.vexp.logits";
    private static final String T_IDS = "attn.vexp.ids";
    private static final String T_WEIGHTS = "attn.vexp.weights";
    private static final String T_OUT = "attn.vexp.out";

    private final CpuScratch scratch;
    private final CpuMatVec matVec;
    private final CpuExpertMatmul products;
    private final int layer;
    private final int ubatch;
    private final int nEmbd;
    private final int vDim;
    private final int nExpert;
    private final int k;
    private final MoeRouting.Params params;
    private final CpuTensorRef router;
    private final CpuTensorRef exps;
    private final float[] gateBias;
    private final CpuProjectionDeltas.Site routerDeltas;
    private final CpuProjectionDeltas.ExpertSite expertDeltas;

    /**
     * @param gateBias {@code attn_v_gate.bias} ({@code n_value_expert} values); required when the spec's router has a
     *                 selection bias, else null
     * @throws IllegalStateException    when a tensor's shape does not match the spec
     * @throws IllegalArgumentException when the routing configuration is inconsistent
     */
    CpuValueExperts(CpuForwardContext ctx, LayerSpec layer, QkvProjection proj, MoeSpec spec, float[] gateBias) {
        this.scratch = ctx.scratch();
        this.matVec = ctx.matVec();
        this.products = new CpuExpertMatmul(matVec);
        this.layer = layer.il();
        this.ubatch = scratch.ubatch();
        this.vDim = spec.nFfExp();
        this.nExpert = spec.nExpert();
        this.k = spec.nUsed();
        CpuWeights w = ctx.weights();
        this.router = w.layer(layer, proj.vGateWeight());
        this.exps = w.layer(layer, proj.vExpsWeight());
        this.nEmbd = Math.toIntExact(router.cols());
        if (router.rows() != nExpert) {
            throw new IllegalStateException("layer " + this.layer + ": " + proj.vGateWeight() + " must be [n_embd, "
                    + nExpert + "], got " + router);
        }
        if (exps.cols() != nEmbd || exps.rows() != (long) vDim * nExpert) {
            throw new IllegalStateException("layer " + this.layer + ": " + proj.vExpsWeight() + " must hold "
                    + nExpert + " experts of [" + nEmbd + ", " + vDim + "], got " + exps);
        }
        if (spec.probBias() != (gateBias != null)) {
            throw new IllegalStateException("layer " + this.layer + ": the value router's selection bias "
                    + (gateBias == null ? "is missing" : "is not part of the spec"));
        }
        if (gateBias != null) {
            QkvProjection.checkLength(proj.vGateBias(), gateBias.length, nExpert);
        }
        this.gateBias = gateBias;
        this.params = MoeRouting.Params.forExperts(spec);
        this.routerDeltas = ctx.projectionDeltas().serve(layer, proj.vGateWeight(), router, matVec);
        this.expertDeltas = ctx.projectionDeltas().serveExperts(layer, proj.vExpsWeight(), exps, nExpert, matVec);
        long pairs = (long) ubatch * k;
        if (pairs * Math.max(vDim, 1) > Integer.MAX_VALUE || (long) ubatch * nExpert > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("layer " + this.layer + ": value experts at ubatch " + ubatch
                    + " exceed one array");
        }
        scratch.temp(T_LOGITS, ubatch * nExpert);
        scratch.intTemp(T_IDS, (int) pairs);
        scratch.temp(T_WEIGHTS, (int) pairs);
        scratch.temp(T_OUT, (int) pairs * vDim);
    }

    /** Values per input row. */
    int inputWidth() {
        return nEmbd;
    }

    /**
     * Writes V of {@code n} tokens ({@code v[t * vDim + c]}) from the dense input rows {@code x[t * n_embd + c]}.
     */
    void run(int n, float[] x, float[] v) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(v, "v");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer + ": value experts of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        float[] logits = scratch.temp(T_LOGITS, ubatch * nExpert);
        int[] ids = scratch.intTemp(T_IDS, ubatch * k);
        float[] weights = scratch.temp(T_WEIGHTS, ubatch * k);
        float[] out = scratch.temp(T_OUT, ubatch * k * vDim);
        matVec.matmul(router, x, 0, logits, 0, n);
        routerDeltas.applyRows(n, x, 0, logits, 0);
        MoeRouting.Selection sel = MoeRouting.route(params, logits, n, gateBias);
        int pairs = n * k;
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < k; j++) {
                int p = t * k + j;
                ids[p] = sel.id(t, j);
                weights[p] = sel.weight(t, j);
            }
        }
        products.run(exps, nExpert, CpuExpertMatmul.Groups.of(ids, pairs, nExpert), x, 0, k, out, 0);
        if (!expertDeltas.isEmpty()) {
            expertDeltas.apply(pairs, k, ids, x, 0, true, out, 0);
        }
        CpuOps.unary(ActOp.SILU, ActParams.NONE, out, 0, out, 0, pairs * vDim);
        MoeCombine.weightedSum(n, vDim, k, 0, out, weights, v);
    }

    @Override
    public String toString() {
        return "CpuValueExperts{layer " + layer + ", " + k + " of " + nExpert + " experts of " + vDim + '}';
    }
}
