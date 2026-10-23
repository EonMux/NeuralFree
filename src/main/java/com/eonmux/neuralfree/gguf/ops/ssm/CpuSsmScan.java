package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;

/**
 * CPU twins of {@link SsmKernels#SSM_SCAN} (the Mamba1/Mamba2/PLaMo2 selective scan, ggml_ssm_scan,
 * ggml-cpu/ops.cpp:10092-10308, with the graph's dt bias folded in) and {@link SsmKernels#MAMBA_GATE_OUT}
 * (the D skip and z gate, models/mamba-base.cpp:140-141 / 286-288). Pure and stateless; the caller owns every
 * array.
 *
 * <p>The scan state has the llama.cpp layout {@code s[(h*headDim + p)*dState + n]} inside the row starting at
 * {@code stateRowBase}; it is updated in place.
 */
public final class CpuSsmScan {

    /** Channels per parallel task. */
    private static final int GRAIN = 16;

    private CpuSsmScan() {
    }

    /**
     * Selective scan over {@code nTokens} tokens. For channel {@code ch = h*headDim + p}, group
     * {@code g = h / (nHead / nGroup)}, token {@code t} and state {@code n}:
     * <pre>
     *   dt'  = softplus(dt[dtOff + t*dtStride + h] + (dtBias != null ? dtBias[h] : 0))
     *   x_dt = x[xOff + t*xStride + ch] * dt'
     *   dA   = exp(dt' * (aPerState ? a[h*dState + n] : a[h]))
     *   s    = s * dA + bMat[bOff + t*bStride + g*dState + n] * x_dt
     *   y[yOff + t*yStride + ch] = sum_n s * cMat[cOff + t*cStride + g*dState + n]
     * </pre>
     * Mamba1 is {@code nHead = d_inner, headDim = 1, nGroup = 1, aPerState = true}; Mamba2 and PLaMo2 use
     * {@code aPerState = false} (PLaMo2 with {@code nGroup = 1}).
     *
     * @param dtBias       per-head dt bias, or null when the graph adds none
     * @param stateRowBase float index of the state row ({@code row * nHead*headDim*dState})
     * @throws IllegalArgumentException on a bad shape or an array too short for the indices
     */
    public static void scan(int dState, int headDim, int nHead, int nGroup, int nTokens, boolean aPerState,
                            float[] x, int xOff, int xStride,
                            float[] dt, int dtOff, int dtStride, float[] dtBias,
                            float[] a,
                            float[] bMat, int bOff, int bStride,
                            float[] cMat, int cOff, int cStride,
                            float[] state, int stateRowBase,
                            float[] y, int yOff, int yStride) {
        SsmKernels.checkScanShape(dState, headDim, nHead, nGroup);
        if (nTokens < 0) {
            throw new IllegalArgumentException("SSM scan n_tokens must be non-negative, got " + nTokens);
        }
        final int nCh = nHead * headDim;
        if (nTokens > 0) {
            CpuSsmConv.requireRows("x", x, xOff, xStride, nTokens, nCh);
            CpuSsmConv.requireRows("dt", dt, dtOff, dtStride, nTokens, nHead);
            CpuSsmConv.requireRows("B", bMat, bOff, bStride, nTokens, nGroup * dState);
            CpuSsmConv.requireRows("C", cMat, cOff, cStride, nTokens, nGroup * dState);
            CpuSsmConv.requireRows("y", y, yOff, yStride, nTokens, nCh);
        }
        if (dtBias != null) {
            CpuSsmConv.requireSpan("dt bias", dtBias, 0, nHead);
        }
        CpuSsmConv.requireSpan("A", a, 0, aPerState ? (long) nHead * dState : nHead);
        CpuSsmConv.requireSpan("state", state, stateRowBase, (long) nCh * dState);
        final int headsPerGroup = nHead / nGroup;

        CpuParallel.parallelFor(nCh, GRAIN, (from, to) -> {
            for (int ch = from; ch < to; ch++) {
                int h = ch / headDim;
                scanChannel(ch, h, h / headsPerGroup, dState, nTokens, aPerState, x, xOff, xStride, dt, dtOff, dtStride,
                        dtBias != null ? dtBias[h] : 0.0f, a, bMat, bOff, bStride, cMat, cOff, cStride,
                        state, stateRowBase + ch * dState, y, yOff, yStride);
            }
        });
    }

    private static void scanChannel(int ch, int h, int g, int dState, int nTokens, boolean aPerState,
                                    float[] x, int xOff, int xStride, float[] dt, int dtOff, int dtStride, float bias,
                                    float[] a, float[] bMat, int bOff, int bStride, float[] cMat, int cOff, int cStride,
                                    float[] state, int sBase, float[] y, int yOff, int yStride) {
        final int aBase = h * dState;
        for (int t = 0; t < nTokens; t++) {
            float dtSp = ActMath.softplus(dt[dtOff + t * dtStride + h] + bias);
            float xDt = x[xOff + t * xStride + ch] * dtSp;
            float dAHead = aPerState ? 0.0f : (float) Math.exp(dtSp * a[h]);
            int bt = bOff + t * bStride + g * dState;
            int ct = cOff + t * cStride + g * dState;
            float sum = 0.0f;
            for (int n = 0; n < dState; n++) {
                float dA = aPerState ? (float) Math.exp(dtSp * a[aBase + n]) : dAHead;
                float s = state[sBase + n] * dA + bMat[bt + n] * xDt;
                state[sBase + n] = s;
                sum += s * cMat[ct + n];
            }
            y[yOff + t * yStride + ch] = sum;
        }
    }

    /**
     * Mamba output gate over {@code nTokens} rows of {@code width} channels:
     * {@code out[outOff + t*width + c] = (y[yOff + t*width + c] + d[c / headDim] * x[xOff + t*xStride + c])
     * * silu(z[zOff + t*zStride + c])}. {@code d} is per channel for Mamba1 ({@code headDim = 1}) and per head
     * for Mamba2/PLaMo2. {@code out} may be {@code y} with {@code outOff == yOff}.
     *
     * @throws IllegalArgumentException on a bad shape or an array too short for the indices
     */
    public static void gateOut(int nTokens, int width, int headDim,
                               float[] y, int yOff,
                               float[] x, int xOff, int xStride,
                               float[] d,
                               float[] z, int zOff, int zStride,
                               float[] out, int outOff) {
        if (nTokens < 0 || width <= 0 || headDim <= 0 || width % headDim != 0) {
            throw new IllegalArgumentException("Mamba gate needs n_tokens >= 0 and head_dim dividing width: n_tokens="
                    + nTokens + " width=" + width + " head_dim=" + headDim);
        }
        if (nTokens == 0) {
            return;
        }
        CpuSsmConv.requireRows("y", y, yOff, width, nTokens, width);
        CpuSsmConv.requireRows("x", x, xOff, xStride, nTokens, width);
        CpuSsmConv.requireRows("z", z, zOff, zStride, nTokens, width);
        CpuSsmConv.requireRows("out", out, outOff, width, nTokens, width);
        CpuSsmConv.requireSpan("D", d, 0, width / headDim);
        int n = nTokens * width;
        CpuParallel.parallelFor(n, 4096, (from, to) -> {
            for (int i = from; i < to; i++) {
                int t = i / width;
                int c = i - t * width;
                float v = y[yOff + i] + d[c / headDim] * x[xOff + t * xStride + c];
                out[outOff + i] = v * ActMath.silu(z[zOff + t * zStride + c]);
            }
        });
    }
}
