package com.eonmux.neuralfree.gguf.ops.deltanet;

/**
 * CPU reference of {@code LLM_LightningAttn} (MiniMax-01 lightning attention) and of the slope schedule llama.cpp
 * feeds it. Pure static math over {@code float[]} with explicit offsets and strides, using the kernel's index
 * formulas and the {@code [h][v][k]} state layout ({@code s[h*D*D + v*D + k]}).
 *
 * <p>With {@code s_h = slopes[h] * layerScale}, the per-token recurrence
 * {@code kv = e^{-s_h} kv + k_t (x) v_t, o_t = q_t . kv} equals the block form of src/models/minimax-01.cpp:288-398
 * ({@code o_t = e^{-s(t+1)} q_t.kv_old + sum_{j<=t} e^{-s(t-j)} (q_t.k_j) v_j},
 * {@code kv_new = e^{-sT} kv_old + sum_j e^{-s(T-1-j)} k_j (x) v_j}). No {@code 1/sqrt(D)} scale is applied.
 */
public final class CpuLightning {

    private CpuLightning() {
    }

    /**
     * Per-head base slopes of src/models/minimax-01.cpp:98-103:
     * {@code start = 2^(-2^(-(log2(n_head) - 3)))}, {@code slope_h = start * start^h}.
     *
     * @throws IllegalArgumentException when {@code nHead <= 0}
     */
    public static float[] baseSlopes(int nHead) {
        if (nHead <= 0) {
            throw new IllegalArgumentException("lightning attention needs at least one head, got " + nHead);
        }
        float log2Heads = (float) (Math.log(nHead) / Math.log(2.0));
        float start = (float) Math.pow(2.0, -(float) Math.pow(2.0, -(log2Heads - 3.0f)));
        float[] slopes = new float[nHead];
        for (int h = 0; h < nHead; h++) {
            slopes[h] = start * (float) Math.pow(start, h);
        }
        return slopes;
    }

    /**
     * Per-layer slope scale of src/models/minimax-01.cpp:288: {@code 1 - il/(n_layer-1) + 1e-5}, evaluated in
     * double and rounded to float as llama.cpp does.
     *
     * @throws IllegalArgumentException when {@code nLayer < 2} (the formula divides by {@code n_layer - 1}) or
     *                                  {@code il} is outside {@code [0, nLayer)}
     */
    public static float layerScale(int il, int nLayer) {
        if (nLayer < 2) {
            throw new IllegalArgumentException("lightning slope scale needs at least two layers, got " + nLayer);
        }
        if (il < 0 || il >= nLayer) {
            throw new IllegalArgumentException("layer " + il + " outside [0, " + nLayer + ")");
        }
        return (float) (1.0 - 1.0 * il / (nLayer - 1) + 1e-5);
    }

    /** {@link #forwardHeads} over every head. */
    public static void forward(int nTokens, int nHead, int headDim, float layerScale, float[] slopes,
                               float[] qkv, int qkvOff, int qkvStride, float[] state, int stateOff,
                               float[] out, int outOff, int outStride) {
        forwardHeads(nTokens, nHead, headDim, layerScale, slopes, qkv, qkvOff, qkvStride, state, stateOff,
                out, outOff, outStride, 0, nHead);
    }

    /**
     * {@code LLM_LightningAttn} for heads {@code [h0, h1)} over {@code nTokens} consecutive tokens.
     * <ul>
     *   <li>qkv (after silu): head h of token t is {@code qkv[qkvOff + t*qkvStride + h*3D + {0, D, 2D} + i]}
     *       (q | k | v); the kernel's dense layout has {@code qkvStride = n_head*3D}.</li>
     *   <li>out: {@code out[outOff + t*outStride + h*D + v]}; the kernel's dense layout has {@code outStride = n_head*D}.</li>
     *   <li>state row at {@code stateOff}: {@code s[h*D*D + v*D + k]}.</li>
     * </ul>
     *
     * @throws IllegalArgumentException when the kernel cannot run this shape
     */
    public static void forwardHeads(int nTokens, int nHead, int headDim, float layerScale, float[] slopes,
                                    float[] qkv, int qkvOff, int qkvStride, float[] state, int stateOff,
                                    float[] out, int outOff, int outStride, int h0, int h1) {
        DeltaNetKernels.validateLightning(nHead, headDim);
        if (slopes.length < nHead) {
            throw new IllegalArgumentException("lightning attention needs " + nHead + " slopes, got " + slopes.length);
        }
        if (h0 < 0 || h1 > nHead || h0 > h1) {
            throw new IllegalArgumentException("head range [" + h0 + ", " + h1 + ") outside [0, " + nHead + ")");
        }
        final int d = headDim;
        for (int h = h0; h < h1; h++) {
            final float decay = (float) Math.exp(-(slopes[h] * layerScale));
            final int headBase = stateOff + h * d * d;
            for (int t = 0; t < nTokens; t++) {
                final int base = qkvOff + t * qkvStride + h * 3 * d;
                final int kBase = base + d;
                final int vBase = base + 2 * d;
                final int oBase = outOff + t * outStride + h * d;
                for (int col = 0; col < d; col++) {
                    final int s0 = headBase + col * d;
                    final float vv = qkv[vBase + col];
                    float o = 0.0f;
                    for (int i = 0; i < d; i++) {
                        float s = state[s0 + i] * decay + qkv[kBase + i] * vv;
                        state[s0 + i] = s;
                        o += s * qkv[base + i];
                    }
                    out[oBase + col] = o;
                }
            }
        }
    }
}
