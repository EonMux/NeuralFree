package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

/**
 * CPU reference of the RWKV mixing kernels, the twins of {@link RwkvKernels#TOKEN_SHIFT_LERP},
 * {@link RwkvKernels#DECAY6}, {@link RwkvKernels#DECAY7}, {@link RwkvKernels#KK_PREP} and {@link RwkvKernels#BONUS}.
 * The formulas and their operation order follow models/rwkv6-base.cpp and models/rwkv7-base.cpp as evaluated by
 * ggml-cpu (float element ops; row sums accumulated in double as {@code ggml_float}).
 *
 * <p>Activations are token-major rows starting at their offsets; weight vectors ({@code lerp_*}, decay bases,
 * {@code k_k}, {@code k_a}, {@code r_k}) start at index 0. The ops are stateless; the shift state is updated in
 * place.
 */
public final class CpuRwkvMix {

    private static final int ELEMENT_GRAIN = 4096;

    private CpuRwkvMix() {
    }

    /**
     * Token shift fused with the RWKV lerps. With {@code prev(t) = t == 0 ? shiftState[shiftOff + c] : cur[t-1][c]}
     * and {@code sx = prev - cur[t][c]}, slot m of {@code out} is the plane {@code out[outOff + m*nTokens*nEmbd]}:
     * <ul>
     *   <li>{@link RwkvKernels#SHIFT_MODE_LERP}: {@code out[m] = sx * lerpFused[m][c] + cur}, {@code m < nMix};</li>
     *   <li>{@link RwkvKernels#SHIFT_MODE_LERP_X}: {@code out[0] = sx * lerpX[c] + cur}; the shift state is left
     *       unchanged so a following {@code LORA} pass sees the same previous token;</li>
     *   <li>{@link RwkvKernels#SHIFT_MODE_LORA}: {@code out[m] = (out[m] + lerpFused[m][c]) * sx + cur} in place,
     *       {@code m < nMix};</li>
     *   <li>{@link RwkvKernels#SHIFT_MODE_PAIR}: {@code out[0] = sx * lerpX[c] + cur},
     *       {@code out[1 + m] = sx * lerpFused[m][c] + cur}, {@code m < nMix}.</li>
     * </ul>
     * Every mode except {@code LERP_X} then stores {@code cur[nTokens-1]} as the new shift state.
     *
     * @param lerpX     {@code [nEmbd]}; unused (may be null) in modes {@code LERP} and {@code LORA}
     * @param lerpFused {@code [nMix][nEmbd]}; unused (may be null) in mode {@code LERP_X}
     */
    public static void tokenShiftLerp(int nTokens, int nEmbd, int nMix, int mode,
                                      float[] cur, int curOff, float[] shiftState, int shiftOff,
                                      float[] lerpX, float[] lerpFused, float[] out, int outOff) {
        if (!RwkvKernels.isShiftMode(mode)) {
            throw new IllegalArgumentException("unknown RWKV token-shift mode " + mode);
        }
        if (nEmbd < 1) {
            throw new IllegalArgumentException("n_embd must be positive, got " + nEmbd);
        }
        boolean useX = mode == RwkvKernels.SHIFT_MODE_LERP_X || mode == RwkvKernels.SHIFT_MODE_PAIR;
        int nm = mode == RwkvKernels.SHIFT_MODE_LERP_X ? 0 : nMix;
        if (nm < 0 || (!useX && nm == 0)) {
            throw new IllegalArgumentException("token-shift mode " + mode + " needs a positive n_mix, got " + nMix);
        }
        int m0 = mode == RwkvKernels.SHIFT_MODE_PAIR ? 1 : 0;
        CpuWkv.checkRows("cur", cur, curOff, nTokens, nEmbd);
        CpuWkv.checkRows("shift_state", shiftState, shiftOff, 1, nEmbd);
        if (useX) {
            CpuWkv.checkRows("lerp_x", lerpX, 0, 1, nEmbd);
        }
        if (nm > 0) {
            CpuWkv.checkRows("lerp_fused", lerpFused, 0, nm, nEmbd);
        }
        CpuWkv.checkRows("out", out, outOff, (m0 + nm) * nTokens, nEmbd);
        int plane = nTokens * nEmbd;
        CpuParallel.parallelFor(nEmbd, 256, (c0, c1) -> {
            for (int c = c0; c < c1; c++) {
                float lx = useX ? lerpX[c] : 0.0f;
                float prev = shiftState[shiftOff + c];
                for (int t = 0; t < nTokens; t++) {
                    int idx = t * nEmbd + c;
                    float x = cur[curOff + idx];
                    float sx = prev - x;
                    if (useX) {
                        out[outOff + idx] = sx * lx + x;
                    }
                    for (int m = 0; m < nm; m++) {
                        float lf = lerpFused[m * nEmbd + c];
                        int o = outOff + (m0 + m) * plane + idx;
                        out[o] = mode == RwkvKernels.SHIFT_MODE_LORA ? (out[o] + lf) * sx + x : sx * lf + x;
                    }
                    prev = x;
                }
                if (mode != RwkvKernels.SHIFT_MODE_LERP_X) {
                    shiftState[shiftOff + c] = prev;
                }
            }
        });
    }

    /** RWKV6 decay: {@code out[i] = exp(-exp(in[i] + base[i % nCols]))} for {@code i < n}. */
    public static void decay6(int n, int nCols, float[] in, int inOff, float[] base, float[] out, int outOff) {
        checkDecay(n, nCols, in, inOff, base, out, outOff);
        CpuParallel.parallelFor(n, ELEMENT_GRAIN, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                float x = in[inOff + i] + base[i % nCols];
                float e = (float) Math.exp(x);
                out[outOff + i] = (float) Math.exp(-e);
            }
        });
    }

    /** RWKV7 decay: {@code out[i] = exp(sigmoid(in[i] + base[i % nCols]) * -0.606531)} for {@code i < n}. */
    public static void decay7(int n, int nCols, float[] in, int inOff, float[] base, float[] out, int outOff) {
        checkDecay(n, nCols, in, inOff, base, out, outOff);
        CpuParallel.parallelFor(n, ELEMENT_GRAIN, (i0, i1) -> {
            for (int i = i0; i < i1; i++) {
                float x = in[inOff + i] + base[i % nCols];
                float sig = 1.0f / (1.0f + (float) Math.exp(-x));
                out[outOff + i] = (float) Math.exp(sig * RwkvKernels.RWKV7_DECAY_SCALE);
            }
        });
    }

    /**
     * RWKV7 key preparation, per token t and head h over {@code c = h * S + d}:
     * {@code kk = l2_norm(k * kK)} over the head with eps 1e-12 (ggml-cpu/ops.cpp:4570),
     * {@code ka = k * kA}, {@code kOut = k + (a * ka - ka)}, {@code kkOut = kk}, {@code aaOut = -kk},
     * {@code bbOut = kk * a}. All activation arrays share the row layout of {@code k} at their own offsets.
     */
    public static void kkPrep(int nTokens, int nHead, int headSize,
                              float[] k, int kOff, float[] kK, float[] kA, float[] a, int aOff,
                              float[] kkOut, int kkOff, float[] kOut, int kOutOff,
                              float[] aaOut, int aaOff, float[] bbOut, int bbOff) {
        int c = channels(nHead, headSize);
        CpuWkv.checkRows("k", k, kOff, nTokens, c);
        CpuWkv.checkRows("k_k", kK, 0, 1, c);
        CpuWkv.checkRows("k_a", kA, 0, 1, c);
        CpuWkv.checkRows("a", a, aOff, nTokens, c);
        CpuWkv.checkRows("kk_out", kkOut, kkOff, nTokens, c);
        CpuWkv.checkRows("k_out", kOut, kOutOff, nTokens, c);
        CpuWkv.checkRows("aa_out", aaOut, aaOff, nTokens, c);
        CpuWkv.checkRows("bb_out", bbOut, bbOff, nTokens, c);
        CpuParallel.parallelFor(nTokens * nHead, 16, (r0, r1) -> {
            for (int row = r0; row < r1; row++) {
                int base = row * headSize;
                int w = (row % nHead) * headSize;
                double sum = 0.0;
                for (int d = 0; d < headSize; d++) {
                    float x = k[kOff + base + d] * kK[w + d];
                    sum += x * x;
                }
                float scale = 1.0f / Math.max((float) Math.sqrt((float) sum), RwkvKernels.KK_NORM_EPS);
                for (int d = 0; d < headSize; d++) {
                    float kv = k[kOff + base + d];
                    float av = a[aOff + base + d];
                    float kk = kv * kK[w + d] * scale;
                    float ka = kv * kA[w + d];
                    kkOut[kkOff + base + d] = kk;
                    kOut[kOutOff + base + d] = kv + (av * ka - ka);
                    aaOut[aaOff + base + d] = -kk;
                    bbOut[bbOff + base + d] = kk * av;
                }
            }
        });
    }

    /**
     * RWKV7 r_k bonus, per token t and head h over {@code c = h * S + d}:
     * {@code rk = sum_d (k * r) * rK[c]} (summed in double as ggml_vec_sum_f32), then {@code x[t,c] += v * rk}.
     */
    public static void bonus(int nTokens, int nHead, int headSize,
                             float[] k, int kOff, float[] r, int rOff, float[] rK, float[] v, int vOff,
                             float[] x, int xOff) {
        int c = channels(nHead, headSize);
        CpuWkv.checkRows("k", k, kOff, nTokens, c);
        CpuWkv.checkRows("r", r, rOff, nTokens, c);
        CpuWkv.checkRows("r_k", rK, 0, 1, c);
        CpuWkv.checkRows("v", v, vOff, nTokens, c);
        CpuWkv.checkRows("x", x, xOff, nTokens, c);
        CpuParallel.parallelFor(nTokens * nHead, 16, (r0, r1) -> {
            for (int row = r0; row < r1; row++) {
                int base = row * headSize;
                int w = (row % nHead) * headSize;
                double sum = 0.0;
                for (int d = 0; d < headSize; d++) {
                    float kr = k[kOff + base + d] * r[rOff + base + d];
                    sum += kr * rK[w + d];
                }
                float rk = (float) sum;
                for (int d = 0; d < headSize; d++) {
                    x[xOff + base + d] += v[vOff + base + d] * rk;
                }
            }
        });
    }

    private static void checkDecay(int n, int nCols, float[] in, int inOff, float[] base, float[] out, int outOff) {
        if (n < 0 || nCols < 1) {
            throw new IllegalArgumentException("invalid RWKV decay size: n " + n + ", n_cols " + nCols);
        }
        CpuWkv.checkRows("w_in", in, inOff, 1, n);
        CpuWkv.checkRows("base", base, 0, 1, Math.min(n, nCols));
        CpuWkv.checkRows("out", out, outOff, 1, n);
    }

    private static int channels(int nHead, int headSize) {
        if (nHead < 1 || headSize < 1) {
            throw new IllegalArgumentException("invalid RWKV head layout: n_head " + nHead + ", head size " + headSize);
        }
        return nHead * headSize;
    }
}
