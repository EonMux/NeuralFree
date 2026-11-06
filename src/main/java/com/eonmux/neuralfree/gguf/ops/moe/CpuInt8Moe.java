package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Fusion;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Kernels;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.act.GluMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import com.eonmux.neuralfree.gguf.ops.moe.CpuExpertMatmul.TileEpilogue;

import java.util.Arrays;
import java.util.Objects;

/**
 * The fused int8 expert pass of a MoE layer, the twin of vLLM's {@code cpu_fused_moe_int8}: one call runs the
 * step's selected experts from the quantized input rows to the weighted output rows, over expert weights stored as
 * int8 ({@code Q8_0}'s 32-value int8 blocks or {@code I8_B128}'s 128-value blocks).
 *
 * <p>The pass is the pipeline of {@code build_moe_ffn} with every stage fused into the expert product it feeds:
 * <ol>
 *   <li>the gate and up projections grouped by expert over the step's pair blocks ({@link CpuExpertMatmul}), whose
 *       tile epilogue adds each pair's expert bias and, for the gate product, computes the gated activation into the
 *       pair's activation row, so a gate or up tile is consumed while it is still in cache and no separate bias or
 *       GLU pass runs;</li>
 *   <li>the activated rows multiplied by the down projection, grouped by expert, whose tile epilogue adds each
 *       pair's expert down bias;</li>
 *   <li>the weighted sum of each token's selected expert outputs, accumulated through
 *       {@link CpuInt8Kernels#accumulateWeighted} in pair order.</li>
 * </ol>
 * Every product is the same int8 vec-dot over the same quantized row as the generic per-pair path, so the fused pass
 * returns the generic results bit for bit: it saves the separate bias, activation and weighted-sum passes and their
 * temporary rows, not arithmetic. A gated set with separate gate and up tensors and an ungated set are handled; a
 * fused {@code ffn_gate_up_exps} tensor, LoRA deltas, a post-expert norm and llama4's weighted input scaling are
 * not, and the caller keeps the generic path for those features (see {@code CpuMoeFfn.ExpertPass}).
 */
final class CpuInt8Moe {

    private static final CpuInt8Kernels KERNELS = CpuInt8Fusion.kernels();

    private final CpuExpertMatmul products;

    CpuInt8Moe(CpuMatVec matVec) {
        this.products = new CpuExpertMatmul(Objects.requireNonNull(matVec, "matVec"));
    }

    /** True when {@code w} holds int8 values: {@code Q8_0} blocks or {@code I8_B128} blocks. */
    static boolean isInt8(CpuTensorRef w) {
        return w != null && (w.type() == GGMLType.Q8_0 || w.type() == GGMLType.I8_B128);
    }

    /**
     * Runs one expert set over the step's pairs: {@code dst[t] = sum_j weights[t*k + j] * expert(ids[t*k + j],
     * x[t])}, with the gate, up and down biases of each pair's expert.
     *
     * @param up         up projection of the set
     * @param gate       gate projection, or null for an ungated set
     * @param down       down projection
     * @param nExpert    experts the projections hold
     * @param groups     the step's pairs grouped by expert
     * @param x          input rows, one per token
     * @param pairsPerRow pairs sharing one input row (the token's selection size)
     * @param ids        expert of every pair, {@code ids[p]}
     * @param weights    routing weight of every pair
     * @param k          pairs per token
     * @param upBias     per-expert up bias, or null
     * @param gateBias   per-expert gate bias, or null
     * @param downBias   per-expert down bias, or null
     * @param glu        gated op of a gated set
     * @param actOp      activation op of an ungated set (null for a gated set)
     * @param params     activation parameters of {@code glu} or {@code actOp}
     * @param nFf        rows of each expert's gate and up blocks
     * @param outDim     rows of each expert's down block
     * @param upBuf      pairs x {@code nFf} gate/up working buffer (the ungated set may pass {@code actBuf})
     * @param gateBuf    pairs x {@code nFf} working buffer of the gate rows, or null for an ungated set
     * @param actBuf     pairs x {@code nFf} activated rows
     * @param expertOut  pairs x {@code outDim} per-pair expert outputs
     * @param dst        output rows, one per token, overwritten
     * @throws IllegalArgumentException when a buffer, count or shape is inconsistent
     */
    void run(CpuTensorRef up, CpuTensorRef gate, CpuTensorRef down, int nExpert, CpuExpertMatmul.Groups groups,
             float[] x, int pairsPerRow, int[] ids, float[] weights, int k, float[] upBias, float[] gateBias,
             float[] downBias, GluOp glu, ActOp actOp, ActParams params, int nFf, int outDim, float[] upBuf,
             float[] gateBuf, float[] actBuf, float[] expertOut, float[] dst) {
        Objects.requireNonNull(up, "up");
        Objects.requireNonNull(down, "down");
        Objects.requireNonNull(groups, "groups");
        final int pairs = groups.pairs();
        check(k, pairs, ids, weights, dst, outDim, nFf, upBuf, gateBuf, actBuf, expertOut);
        final boolean gated = gate != null;
        if (gated) {
            products.run(up, nExpert, groups, x, 0, pairsPerRow, upBuf, 0,
                    bias(upBuf, upBias, ids, nFf));
            products.run(gate, nExpert, groups, x, 0, pairsPerRow, gateBuf, 0, (pair, rowStart, rows) -> {
                addBias(gateBuf, pair, rowStart, rows, gateBias, ids[pair], nFf);
                for (int i = 0; i < rows; i++) {
                    int o = pair * nFf + rowStart + i;
                    actBuf[o] = GluMath.apply(glu, params, gateBuf[o], upBuf[o]);
                }
            });
        } else {
            products.run(up, nExpert, groups, x, 0, pairsPerRow, actBuf, 0, (pair, rowStart, rows) -> {
                addBias(actBuf, pair, rowStart, rows, upBias, ids[pair], nFf);
                for (int i = 0; i < rows; i++) {
                    int o = pair * nFf + rowStart + i;
                    actBuf[o] = ActMath.apply(actOp, params, actBuf[o]);
                }
            });
        }
        products.run(down, nExpert, groups, actBuf, 0, 1, expertOut, 0, bias(expertOut, downBias, ids, outDim));
        weightedSum(pairs / k, outDim, k, expertOut, weights, dst);
    }

    /** The epilogue adding the bias of each pair's expert, or {@link TileEpilogue#NONE} without a bias. */
    private static TileEpilogue bias(float[] data, float[] bias, int[] ids, int dim) {
        if (bias == null) {
            return TileEpilogue.NONE;
        }
        return (pair, rowStart, rows) -> addBias(data, pair, rowStart, rows, bias, ids[pair], dim);
    }

    /** {@code data[pair][rowStart + i] += bias[expert][rowStart + i]}, as {@code ggml_add_id}; none without a bias. */
    private static void addBias(float[] data, int pair, int rowStart, int rows, float[] bias, int expert, int dim) {
        if (bias == null) {
            return;
        }
        int b = expert * dim + rowStart;
        int o = pair * dim + rowStart;
        for (int i = 0; i < rows; i++) {
            data[o + i] += bias[b + i];
        }
    }

    /**
     * {@code dst[t][c] = sum_j weights[t*k + j] * expertOut[t*k + j][c]}, pair index ascending, the accumulation
     * order of {@code MoeCombine.weightedSum} without its stored sum.
     */
    private static void weightedSum(int tokens, int outDim, int k, float[] expertOut, float[] weights,
                                    float[] dst) {
        for (int t = 0; t < tokens; t++) {
            int off = t * outDim;
            Arrays.fill(dst, off, off + outDim, 0.0f);
            for (int j = 0; j < k; j++) {
                int p = t * k + j;
                KERNELS.accumulateWeighted(expertOut, p * outDim, weights[p], dst, off, outDim);
            }
        }
    }

    private static void check(int k, int pairs, int[] ids, float[] weights, float[] dst, int outDim, int nFf,
                              float[] upBuf, float[] gateBuf, float[] actBuf, float[] expertOut) {
        if (k < 1 || pairs % k != 0) {
            throw new IllegalArgumentException(pairs + " pairs do not split into " + k + " per token");
        }
        if (outDim <= 0 || nFf <= 0) {
            throw new IllegalArgumentException("expert rows must be positive, got n_ff " + nFf + ", out " + outDim);
        }
        if (ids.length < pairs || weights.length < pairs) {
            throw new IllegalArgumentException(pairs + " pairs need " + pairs + " ids and weights, got "
                    + ids.length + " and " + weights.length);
        }
        int tokens = pairs / k;
        requireRows("output", dst, (long) tokens * outDim);
        requireRows("activated rows", actBuf, (long) pairs * nFf);
        requireRows("expert outputs", expertOut, (long) pairs * outDim);
        requireRows("up rows", upBuf, (long) pairs * nFf);
        if (gateBuf != null) {
            requireRows("gate rows", gateBuf, (long) pairs * nFf);
        }
    }

    private static void requireRows(String what, float[] a, long n) {
        if (a == null || a.length < n) {
            throw new IllegalArgumentException(what + " holds " + (a == null ? 0 : a.length) + " values, the pass"
                    + " writes " + n);
        }
    }
}
