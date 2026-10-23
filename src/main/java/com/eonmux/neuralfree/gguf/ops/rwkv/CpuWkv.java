package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

import java.util.Arrays;

/**
 * CPU reference of the RWKV WKV recurrences, the twins of {@link RwkvKernels#WKV6}, {@link RwkvKernels#GLA} and
 * {@link RwkvKernels#WKV7}. Each follows the scalar loop of ggml-cpu/ops.cpp in the same operation order.
 *
 * <p>Layouts: {@code k, v, r, ...} and {@code y} are token-major rows of {@code C = H * S} floats starting at their
 * offsets ({@code x[off + t * C + h * S + d]}). {@code state} holds one sequence row of {@code H * S * S} floats at
 * {@code stateOff} and is updated in place to the state after the last token:
 * <ul>
 *   <li>wkv6 / gla: {@code s[stateOff + h*S*S + i*S + j]}, i = key, j = value;</li>
 *   <li>wkv7: {@code s[stateOff + h*S*S + i*S + j]}, i = value, j = key.</li>
 * </ul>
 * Heads are independent and run in parallel; tokens run in order. The ops are stateless.
 */
public final class CpuWkv {

    private CpuWkv() {
    }

    /**
     * RWKV6 WKV (ggml-cpu/ops.cpp:10732). Per token t, head h, key i, value j, with {@code kv = v[j] * k[i]}:
     * {@code y[j] += (kv * tf[h,i] + s[i][j]) * r[i]} and {@code s[i][j] = s[i][j] * td[i] + kv}.
     *
     * @param tf time_mix_first, {@code [H][S]} from index 0
     * @param td per-token decay rows
     */
    public static void wkv6(int nTokens, int c, int nHead,
                            float[] k, int kOff, float[] v, int vOff, float[] r, int rOff,
                            float[] tf, float[] td, int tdOff,
                            float[] state, int stateOff, float[] y, int yOff) {
        int s = headSize(c, nHead);
        checkRows("k", k, kOff, nTokens, c);
        checkRows("v", v, vOff, nTokens, c);
        checkRows("r", r, rOff, nTokens, c);
        checkRows("td", td, tdOff, nTokens, c);
        checkRows("tf", tf, 0, 1, c);
        checkRows("state", state, stateOff, 1, c * s);
        checkRows("y", y, yOff, nTokens, c);
        CpuParallel.parallelFor(nHead, 1, (h0, h1) -> {
            for (int h = h0; h < h1; h++) {
                int sh = stateOff + h * s * s;
                for (int t = 0; t < nTokens; t++) {
                    int th = t * c + h * s;
                    int yRow = yOff + th;
                    Arrays.fill(y, yRow, yRow + s, 0.0f);
                    for (int i = 0; i < s; i++) {
                        float kVal = k[kOff + th + i];
                        float rVal = r[rOff + th + i];
                        float tfVal = tf[h * s + i];
                        float tdVal = td[tdOff + th + i];
                        int si = sh + i * s;
                        for (int j = 0; j < s; j++) {
                            float kv = v[vOff + th + j] * kVal;
                            float prev = state[si + j];
                            float temp = kv * tfVal + prev;
                            y[yRow + j] += temp * rVal;
                            state[si + j] = prev * tdVal + kv;
                        }
                    }
                }
            }
        });
    }

    /**
     * Gated linear attention (ggml-cpu/ops.cpp:10944). Per token t, head h, key i, value j, with
     * {@code kv = v[j] * k[i]}: {@code s[i][j] = s[i][j] * g + kv} and then {@code y[j] += s[i][j] * (q[i] * scale)}.
     *
     * @param gMode {@link RwkvKernels#GLA_G_PER_CHANNEL} ({@code g} rows of C at {@code gOff}) or
     *              {@link RwkvKernels#GLA_G_PER_HEAD} ({@code g[gOff + t * H + h]})
     */
    public static void gla(int nTokens, int c, int nHead, int gMode, float scale,
                           float[] k, int kOff, float[] v, int vOff, float[] q, int qOff, float[] g, int gOff,
                           float[] state, int stateOff, float[] y, int yOff) {
        int s = headSize(c, nHead);
        if (gMode != RwkvKernels.GLA_G_PER_CHANNEL && gMode != RwkvKernels.GLA_G_PER_HEAD) {
            throw new IllegalArgumentException("unknown GLA g_mode " + gMode);
        }
        checkRows("k", k, kOff, nTokens, c);
        checkRows("v", v, vOff, nTokens, c);
        checkRows("q", q, qOff, nTokens, c);
        checkRows("g", g, gOff, nTokens, gMode == RwkvKernels.GLA_G_PER_CHANNEL ? c : nHead);
        checkRows("state", state, stateOff, 1, c * s);
        checkRows("y", y, yOff, nTokens, c);
        CpuParallel.parallelFor(nHead, 1, (h0, h1) -> {
            for (int h = h0; h < h1; h++) {
                int sh = stateOff + h * s * s;
                for (int t = 0; t < nTokens; t++) {
                    int th = t * c + h * s;
                    int yRow = yOff + th;
                    Arrays.fill(y, yRow, yRow + s, 0.0f);
                    float gHead = gMode == RwkvKernels.GLA_G_PER_HEAD ? g[gOff + t * nHead + h] : 0.0f;
                    for (int i = 0; i < s; i++) {
                        float kVal = k[kOff + th + i];
                        float qVal = q[qOff + th + i] * scale;
                        float gVal = gMode == RwkvKernels.GLA_G_PER_CHANNEL ? g[gOff + th + i] : gHead;
                        int si = sh + i * s;
                        for (int j = 0; j < s; j++) {
                            float kv = v[vOff + th + j] * kVal;
                            float temp = state[si + j] * gVal + kv;
                            y[yRow + j] += temp * qVal;
                            state[si + j] = temp;
                        }
                    }
                }
            }
        });
    }

    /**
     * RWKV7 WKV (ggml-cpu/ops.cpp:11827). Per token t, head h, value row i, key column j:
     * {@code sa = sum_j a[j] * s[i][j]}, {@code s[i][j] = s[i][j] * w[j] + v[i] * k[j] + sa * b[j]},
     * {@code y[i] = sum_j s[i][j] * r[j]}.
     */
    public static void wkv7(int nTokens, int c, int nHead,
                            float[] r, int rOff, float[] w, int wOff, float[] k, int kOff, float[] v, int vOff,
                            float[] a, int aOff, float[] b, int bOff,
                            float[] state, int stateOff, float[] y, int yOff) {
        int s = headSize(c, nHead);
        checkRows("r", r, rOff, nTokens, c);
        checkRows("w", w, wOff, nTokens, c);
        checkRows("k", k, kOff, nTokens, c);
        checkRows("v", v, vOff, nTokens, c);
        checkRows("a", a, aOff, nTokens, c);
        checkRows("b", b, bOff, nTokens, c);
        checkRows("state", state, stateOff, 1, c * s);
        checkRows("y", y, yOff, nTokens, c);
        CpuParallel.parallelFor(nHead, 1, (h0, h1) -> {
            for (int h = h0; h < h1; h++) {
                int sh = stateOff + h * s * s;
                for (int t = 0; t < nTokens; t++) {
                    int th = t * c + h * s;
                    for (int i = 0; i < s; i++) {
                        int si = sh + i * s;
                        float vVal = v[vOff + th + i];
                        float sa = 0.0f;
                        for (int j = 0; j < s; j++) {
                            sa += a[aOff + th + j] * state[si + j];
                        }
                        float result = 0.0f;
                        for (int j = 0; j < s; j++) {
                            float kv = vVal * k[kOff + th + j];
                            float next = state[si + j] * w[wOff + th + j] + kv + sa * b[bOff + th + j];
                            state[si + j] = next;
                            result += next * r[rOff + th + j];
                        }
                        y[yOff + th + i] = result;
                    }
                }
            }
        });
    }

    /** Head size {@code C / H}, validated against {@link RwkvKernels#MAX_HEAD_SIZE}. */
    static int headSize(int c, int nHead) {
        if (nHead < 1 || c < 1 || c % nHead != 0) {
            throw new IllegalArgumentException("RWKV channel count " + c + " is not a positive multiple of the head count " + nHead);
        }
        int s = c / nHead;
        RwkvKernels.checkHeadSize(s);
        return s;
    }

    /** Checks that {@code a} holds {@code rows * width} floats from {@code off}. */
    static void checkRows(String name, float[] a, int off, int rows, int width) {
        if (a == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        if (rows < 0) {
            throw new IllegalArgumentException("row count must be non-negative, got " + rows);
        }
        long need = (long) rows * width;
        if (off < 0 || off + need > a.length) {
            throw new IllegalArgumentException(name + ": " + rows + " x " + width + " floats at offset " + off
                    + " exceed its length " + a.length);
        }
    }
}
