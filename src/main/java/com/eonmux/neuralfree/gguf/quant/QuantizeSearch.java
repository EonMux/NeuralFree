package com.eonmux.neuralfree.gguf.quant;

/**
 * The scale-search helpers of llama.cpp ggml-quants.c that the row encoders share: {@code nearest_int},
 * {@code make_qx_quants} (:637-698), {@code make_q3_quants} (:706-764), {@code make_qkx2_quants}
 * (:808-877), {@code make_qkx3_quants} (:1002-1156) and {@code make_qp_quants} (:1158-1207). Each search
 * fits one scale to a small value group and leaves the chosen levels in {@code L}; the caller packs them
 * into its block format. The loops, clamps and search steps follow the reference line for line, in float
 * arithmetic, so a level vector equals the reference's for the same input.
 * <p>
 * Package-private: the seam is {@link RowQuantizers}, not these helpers.
 */
final class QuantizeSearch {

    /** Smallest max |x| a group quantizes with a non-zero scale ({@code GROUP_MAX_EPS}). */
    static final float GROUP_MAX_EPS = 1e-15f;

    private QuantizeSearch() {
    }

    /**
     * {@code nearest_int} (llama.cpp ggml-quants.c:630-635): round to nearest, ties away from zero,
     * through the float-mantissa trick the reference uses; exact for |fval| &le; 4194303 like the C
     * assertion allows.
     */
    static int nearestInt(float fval) {
        float val = fval + 12582912.0f;
        int i = Float.floatToRawIntBits(val);
        return (i & 0x007fffff) - 0x00400000;
    }

    /**
     * {@code make_qx_quants}: the symmetric max-|x| search with the &plusmn;0.1&middot;nmax grid refinement
     * and the weighted ({@code qw}) or unweighted ({@code rmse_type} 1 = x&sup2;, 2 = 1, 3 = |x|, 4 =
     * &radic;|x|) least-squares scale. {@code rmse_type} &lt; 0 returns early with the average of the
     * initial and the least-squares scale ({@code return_early}). {@code qw} may be null when the unweighted
     * weight applies. Returns the fitted scale; {@code L[lOff + i]} holds level {@code i} + {@code nmax}.
     */
    static float makeQxQuants(int n, int nmax, float[] x, int xOff, byte[] L, int lOff, int rmseType,
                              float[] qw, int qwOff) {
        float max = 0;
        float amax = 0;
        for (int i = 0; i < n; ++i) {
            float ax = Math.abs(x[xOff + i]);
            if (ax > amax) {
                amax = ax;
                max = x[xOff + i];
            }
        }
        if (amax < GROUP_MAX_EPS) { // all zero
            for (int i = 0; i < n; ++i) {
                L[lOff + i] = 0;
            }
            return 0.f;
        }
        float iscale = -nmax / max;
        if (rmseType == 0) {
            for (int i = 0; i < n; ++i) {
                int l = nearestInt(iscale * x[xOff + i]);
                L[lOff + i] = (byte) (nmax + clamp(l, -nmax, nmax - 1));
            }
            return 1 / iscale;
        }
        boolean returnEarly = false;
        if (rmseType < 0) {
            rmseType = -rmseType;
            returnEarly = true;
        }
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearestInt(iscale * x[xOff + i]);
            l = clamp(l, -nmax, nmax - 1);
            L[lOff + i] = (byte) (l + nmax);
            float w = qw != null ? qw[qwOff + i] : rmseType == 1 ? x[xOff + i] * x[xOff + i]
                    : rmseType == 2 ? 1 : rmseType == 3 ? Math.abs(x[xOff + i])
                    : (float) Math.sqrt(Math.abs(x[xOff + i]));
            sumlx += w * x[xOff + i] * l;
            suml2 += w * l * l;
        }
        float scale = suml2 != 0 ? sumlx / suml2 : 0.0f;
        if (returnEarly) {
            return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
        }
        float best = scale * sumlx;
        for (int is = -9; is <= 9; ++is) {
            if (is == 0) {
                continue;
            }
            iscale = -(nmax + 0.1f * is) / max;
            sumlx = suml2 = 0;
            for (int i = 0; i < n; ++i) {
                int l = nearestInt(iscale * x[xOff + i]);
                l = clamp(l, -nmax, nmax - 1);
                float w = qw != null ? qw[qwOff + i] : rmseType == 1 ? x[xOff + i] * x[xOff + i]
                        : rmseType == 2 ? 1 : rmseType == 3 ? Math.abs(x[xOff + i])
                        : (float) Math.sqrt(Math.abs(x[xOff + i]));
                sumlx += w * x[xOff + i] * l;
                suml2 += w * l * l;
            }
            if (suml2 > 0 && sumlx * sumlx > best * suml2) {
                for (int i = 0; i < n; ++i) {
                    int l = nearestInt(iscale * x[xOff + i]);
                    L[lOff + i] = (byte) (nmax + clamp(l, -nmax, nmax - 1));
                }
                scale = sumlx / suml2;
                best = scale * sumlx;
            }
        }
        return scale;
    }

    /**
     * {@code make_q3_quants}: the symmetric search with the 5-pass greedy level refinement when
     * {@code doRmse}, else the plain scale. Returns the fitted scale; {@code L[lOff + i]} holds the
     * unsigned level {@code i} + {@code nmax}.
     */
    static float makeQ3Quants(int n, int nmax, float[] x, int xOff, byte[] L, int lOff, boolean doRmse) {
        float max = 0;
        float amax = 0;
        for (int i = 0; i < n; ++i) {
            float ax = Math.abs(x[xOff + i]);
            if (ax > amax) {
                amax = ax;
                max = x[xOff + i];
            }
        }
        if (amax < GROUP_MAX_EPS) { // all zero
            for (int i = 0; i < n; ++i) {
                L[lOff + i] = 0;
            }
            return 0.f;
        }
        float iscale = -nmax / max;
        if (doRmse) {
            float sumlx = 0;
            float suml2 = 0;
            for (int i = 0; i < n; ++i) {
                int l = nearestInt(iscale * x[xOff + i]);
                l = clamp(l, -nmax, nmax - 1);
                L[lOff + i] = (byte) l;
                float w = x[xOff + i] * x[xOff + i];
                sumlx += w * x[xOff + i] * l;
                suml2 += w * l * l;
            }
            for (int itry = 0; itry < 5; ++itry) {
                int nChanged = 0;
                for (int i = 0; i < n; ++i) {
                    float w = x[xOff + i] * x[xOff + i];
                    float slx = sumlx - w * x[xOff + i] * L[lOff + i];
                    if (slx > 0) {
                        float sl2 = suml2 - w * L[lOff + i] * L[lOff + i];
                        int newL = nearestInt(x[xOff + i] * sl2 / slx);
                        newL = clamp(newL, -nmax, nmax - 1);
                        if (newL != L[lOff + i]) {
                            slx += w * x[xOff + i] * newL;
                            sl2 += w * newL * newL;
                            if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                                L[lOff + i] = (byte) newL;
                                sumlx = slx;
                                suml2 = sl2;
                                ++nChanged;
                            }
                        }
                    }
                }
                if (nChanged == 0) {
                    break;
                }
            }
            for (int i = 0; i < n; ++i) {
                L[lOff + i] += nmax;
            }
            return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
        }
        for (int i = 0; i < n; ++i) {
            int l = nearestInt(iscale * x[xOff + i]);
            l = clamp(l, -nmax, nmax - 1);
            L[lOff + i] = (byte) (l + nmax);
        }
        return 1 / iscale;
    }

    /**
     * {@code make_qkx2_quants}: the affine search over {@code nstep} + 1 candidates of
     * {@code min = -rmin - rdelta&middot;is}, each with the closed-form weighted least-squares
     * (scale, min) and the squared-error ({@code useMad} false) or absolute-error tie-break.
     * {@code weights} has {@code n} values. Returns the fitted scale and writes {@code -min} to
     * {@code theMin[0]}.
     */
    static float makeQkx2Quants(int n, int nmax, float[] x, int xOff, float[] weights, byte[] L, int lOff,
                                float[] theMin, byte[] Laux, int lauxOff, float rmin, float rdelta, int nstep,
                                boolean useMad) {
        float min = x[xOff];
        float max = x[xOff];
        float sumW = weights[0];
        float sumX = sumW * x[xOff];
        for (int i = 1; i < n; ++i) {
            if (x[xOff + i] < min) {
                min = x[xOff + i];
            }
            if (x[xOff + i] > max) {
                max = x[xOff + i];
            }
            float w = weights[i];
            sumW += w;
            sumX += w * x[xOff + i];
        }
        if (min > 0) {
            min = 0;
        }
        if (max == min) {
            for (int i = 0; i < n; ++i) {
                L[lOff + i] = 0;
            }
            theMin[0] = -min;
            return 0.f;
        }
        float iscale = nmax / (max - min);
        float scale = 1 / iscale;
        float bestError = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearestInt(iscale * (x[xOff + i] - min));
            L[lOff + i] = (byte) clamp(l, 0, nmax);
            float diff = scale * (L[lOff + i] & 0xFF) + min - x[xOff + i];
            diff = useMad ? Math.abs(diff) : diff * diff;
            float w = weights[i];
            bestError += w * diff;
        }
        if (nstep < 1) {
            theMin[0] = -min;
            return scale;
        }
        for (int is = 0; is <= nstep; ++is) {
            iscale = (rmin + rdelta * is + nmax) / (max - min);
            float sumL = 0;
            float sumL2 = 0;
            float sumXL = 0;
            for (int i = 0; i < n; ++i) {
                int l = nearestInt(iscale * (x[xOff + i] - min));
                l = clamp(l, 0, nmax);
                Laux[lauxOff + i] = (byte) l;
                float w = weights[i];
                sumL += w * l;
                sumL2 += w * l * l;
                sumXL += w * l * x[xOff + i];
            }
            float D = sumW * sumL2 - sumL * sumL;
            if (D > 0) {
                float thisScale = (sumW * sumXL - sumX * sumL) / D;
                float thisMin = (sumL2 * sumX - sumL * sumXL) / D;
                if (thisMin > 0) {
                    thisMin = 0;
                    thisScale = sumXL / sumL2;
                }
                float curError = 0;
                for (int i = 0; i < n; ++i) {
                    float diff = thisScale * (Laux[lauxOff + i] & 0xFF) + thisMin - x[xOff + i];
                    diff = useMad ? Math.abs(diff) : diff * diff;
                    float w = weights[i];
                    curError += w * diff;
                }
                if (curError < bestError) {
                    for (int i = 0; i < n; ++i) {
                        L[lOff + i] = Laux[lauxOff + i];
                    }
                    bestError = curError;
                    scale = thisScale;
                    min = thisMin;
                }
            }
        }
        theMin[0] = -min;
        return scale;
    }

    /**
     * {@code make_qkx3_quants}: the qkx2 search whose candidates clamp the level computation against
     * overflow ({@code v &gt; 0 ? nearest_int(min(v, nmax)) : 0}) and whose default weight is x&sup2;
     * when {@code weights} is null. Returns the fitted scale and writes {@code -min} to {@code theMin[0]}.
     */
    static float makeQkx3Quants(int n, int nmax, float[] x, int xOff, float[] weights, byte[] L, int lOff,
                                float[] theMin, byte[] Laux, int lauxOff, float rmin, float rdelta, int nstep,
                                boolean useMad) {
        float min = x[xOff];
        float max = x[xOff];
        float sumW = weights != null ? weights[0] : x[xOff] * x[xOff];
        float sumX = sumW * x[xOff];
        for (int i = 1; i < n; ++i) {
            if (x[xOff + i] < min) {
                min = x[xOff + i];
            }
            if (x[xOff + i] > max) {
                max = x[xOff + i];
            }
            float w = weights != null ? weights[i] : x[xOff + i] * x[xOff + i];
            sumW += w;
            sumX += w * x[xOff + i];
        }
        if (min > 0) {
            min = 0;
        }
        if (max <= min) {
            for (int i = 0; i < n; ++i) {
                L[lOff + i] = 0;
            }
            theMin[0] = -min;
            return 0.f;
        }
        float iscale = nmax / (max - min);
        float scale = 1 / iscale;
        float bestMad = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearestInt(iscale * (x[xOff + i] - min));
            L[lOff + i] = (byte) clamp(l, 0, nmax);
            float diff = scale * (L[lOff + i] & 0xFF) + min - x[xOff + i];
            diff = useMad ? Math.abs(diff) : diff * diff;
            float w = weights != null ? weights[i] : x[xOff + i] * x[xOff + i];
            bestMad += w * diff;
        }
        if (nstep < 1) {
            theMin[0] = -min;
            return scale;
        }
        for (int is = 0; is <= nstep; ++is) {
            iscale = (rmin + rdelta * is + nmax) / (max - min);
            float sumL = 0;
            float sumL2 = 0;
            float sumXL = 0;
            for (int i = 0; i < n; ++i) {
                // min is the best fit so far and can be at or near max, so v can be inf, nan or out of
                // range for nearestInt; the reference guards exactly like this
                float v = iscale * (x[xOff + i] - min);
                int l = v > 0 ? nearestInt(Math.min(v, nmax)) : 0;
                Laux[lauxOff + i] = (byte) l;
                float w = weights != null ? weights[i] : x[xOff + i] * x[xOff + i];
                sumL += w * l;
                sumL2 += w * l * l;
                sumXL += w * l * x[xOff + i];
            }
            float D = sumW * sumL2 - sumL * sumL;
            if (D > 0) {
                float thisScale = (sumW * sumXL - sumX * sumL) / D;
                float thisMin = (sumL2 * sumX - sumL * sumXL) / D;
                if (thisMin > 0) {
                    thisMin = 0;
                    thisScale = sumXL / sumL2;
                }
                float mad = 0;
                for (int i = 0; i < n; ++i) {
                    float diff = thisScale * (Laux[lauxOff + i] & 0xFF) + thisMin - x[xOff + i];
                    diff = useMad ? Math.abs(diff) : diff * diff;
                    float w = weights != null ? weights[i] : x[xOff + i] * x[xOff + i];
                    mad += w * diff;
                }
                if (mad < bestMad) {
                    for (int i = 0; i < n; ++i) {
                        L[lOff + i] = Laux[lauxOff + i];
                    }
                    bestMad = mad;
                    scale = thisScale;
                    min = thisMin;
                }
            }
        }
        theMin[0] = -min;
        return scale;
    }

    /**
     * {@code make_qp_quants}: the non-negative group search over the &plusmn;0.1 grid followed by the
     * 5-pass greedy refinement, weighted by {@code quantWeights}. All values must be &ge; 0 (the scale and
     * min groups the K-quant encoders pass). Returns the fitted scale; {@code L[lOff + i]} holds level
     * {@code i} in {@code [0, nmax]}.
     */
    static float makeQpQuants(int n, int nmax, float[] x, int xOff, byte[] L, int lOff, float[] quantWeights,
                              int qwOff) {
        float max = 0;
        for (int i = 0; i < n; ++i) {
            max = Math.max(max, x[xOff + i]);
        }
        if (max < GROUP_MAX_EPS) { // all zero
            for (int i = 0; i < n; ++i) {
                L[lOff + i] = 0;
            }
            return 0.f;
        }
        float iscale = nmax / max;
        for (int i = 0; i < n; ++i) {
            L[lOff + i] = (byte) nearestInt(iscale * x[xOff + i]);
        }
        float scale = 1 / iscale;
        float bestMse = 0;
        for (int i = 0; i < n; ++i) {
            float diff = x[xOff + i] - scale * (L[lOff + i] & 0xFF);
            float w = quantWeights[qwOff + i];
            bestMse += w * diff * diff;
        }
        for (int is = -4; is <= 4; ++is) {
            if (is == 0) {
                continue;
            }
            float iscaleIs = (0.1f * is + nmax) / max;
            float scaleIs = 1 / iscaleIs;
            float mse = 0;
            for (int i = 0; i < n; ++i) {
                int l = nearestInt(iscaleIs * x[xOff + i]);
                l = Math.min(nmax, l);
                float diff = x[xOff + i] - scaleIs * l;
                float w = quantWeights[qwOff + i];
                mse += w * diff * diff;
            }
            if (mse < bestMse) {
                bestMse = mse;
                iscale = iscaleIs;
            }
        }
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearestInt(iscale * x[xOff + i]);
            l = Math.min(nmax, l);
            L[lOff + i] = (byte) l;
            float w = quantWeights[qwOff + i];
            sumlx += w * x[xOff + i] * l;
            suml2 += w * l * l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int nChanged = 0;
            for (int i = 0; i < n; ++i) {
                float w = quantWeights[qwOff + i];
                float slx = sumlx - w * x[xOff + i] * (L[lOff + i] & 0xFF);
                float sl2 = suml2 - w * (L[lOff + i] & 0xFF) * (L[lOff + i] & 0xFF);
                if (slx > 0 && sl2 > 0) {
                    int newL = nearestInt(x[xOff + i] * sl2 / slx);
                    newL = Math.min(nmax, newL);
                    if (newL != (L[lOff + i] & 0xFF)) {
                        slx += w * x[xOff + i] * newL;
                        sl2 += w * newL * newL;
                        if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[lOff + i] = (byte) newL;
                            sumlx = slx;
                            suml2 = sl2;
                            ++nChanged;
                        }
                    }
                }
            }
            if (nChanged == 0) {
                break;
            }
        }
        return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }

    /** C clamp of an int to {@code [lo, hi]}. */
    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
