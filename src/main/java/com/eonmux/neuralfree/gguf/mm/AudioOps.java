package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;

/**
 * The ggml operators of the audio tower graphs (tools/mtmd/models/*.cpp) over two layouts: token-major row blocks
 * {@code [n][width]} stored flat at {@code t * width + c} (the contiguous {@code [width, n]} ggml tensor), and
 * planar images {@code [W, H, C]} stored at {@code x + W * (y + H * c)} (the contiguous ggml image). Matmuls run
 * through {@link CpuMatVec} in the stored weight type; the depthwise convolutions read float kernels. Static
 * methods only; package-private executor support.
 */
final class AudioOps {

    /** Epsilon of the norms the audio graphs fix at 1e-5 instead of reading {@code clip.audio.attention.layer_norm_epsilon}. */
    static final float GRAPH_EPS = 1e-5f;

    private AudioOps() {
    }

    /** {@code build_mm(w, x)} plus the optional bias: {@code [n][w.rows()]} rows. */
    static float[] linear(CpuMatVec matVec, CpuTensorRef w, float[] bias, float[] x, int n) {
        float[] y = new float[Math.multiplyExact(n, (int) w.rows())];
        matVec.matmul(w, x, 0, y, 0, n);
        VisionRows.addBias(y, bias);
        return y;
    }

    /** {@code build_ffn(up, down)} without a gate: up, bias, the activation, down, bias. */
    static float[] ffn(CpuMatVec matVec, float[] x, int n, CpuTensorRef up, float[] upB, ActOp act,
                       CpuTensorRef down, float[] downB) {
        float[] h = linear(matVec, up, upB, x, n);
        activate(act, h);
        return linear(matVec, down, downB, h, n);
    }

    /** The unary op over every value; {@link ActOp#RELU_SQR} is {@code ggml_relu} then {@code ggml_sqr}. */
    static void activate(ActOp op, float[] x) {
        if (op == ActOp.RELU_SQR) {
            for (int i = 0; i < x.length; i++) {
                float v = Math.max(x[i], 0f);
                x[i] = v * v;
            }
            return;
        }
        VisionRows.activate(op, x);
    }

    /** {@code build_norm} with {@code NORM_TYPE_NORMAL} over every row, weight and bias when non-null. */
    static void layerNorm(float[] x, int n, int width, float[] w, float[] b, float eps) {
        CpuOps.layerNorm(x, 0, x, 0, n, width, w, b, eps);
    }

    /** {@code build_norm} with {@code NORM_TYPE_RMS} over every row, then the optional weight and bias. */
    static void rmsNorm(float[] x, int n, int width, float[] w, float[] b, float eps) {
        CpuOps.rmsNorm(x, 0, x, 0, n, width, w, eps);
        VisionRows.addBias(x, b);
    }

    /** {@code dst[i] += s * src[i]} over the whole of {@code dst}. */
    static void addScaled(float[] dst, float[] src, float s) {
        for (int i = 0; i < dst.length; i++) {
            dst[i] += s * src[i];
        }
    }

    /** The sigmoid GLU of the conformer conv modules: {@code a * sigmoid(b)} over the two halves of each row. */
    static float[] sigmoidGlu(float[] x, int n, int half) {
        float[] out = new float[n * half];
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < half; c++) {
                float g = x[t * 2 * half + half + c];
                out[t * half + c] = x[t * 2 * half + c] * (float) (1.0 / (1.0 + Math.exp(-g)));
            }
        }
        return out;
    }

    /** {@code ggml_cont(ggml_transpose(x))} of a {@code [n][width]} row block: {@code [width][n]}. */
    static float[] transpose(float[] x, int n, int width) {
        float[] out = new float[x.length];
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < width; c++) {
                out[c * n + t] = x[t * width + c];
            }
        }
        return out;
    }

    /** Output size of a convolution along one axis: {@code (in + 2 * pad - kernel) / stride + 1}. */
    static int convOut(int in, int kernel, int stride, int pad) {
        return (in + 2 * pad - kernel) / stride + 1;
    }

    /**
     * {@code ggml_conv_2d(kernel, img, s, s, p, p, 1, 1)} over a planar {@code [w, h, cIn]} image as an im2col
     * matmul: {@code kernel} is the {@code [cOut][kw * kh * cIn]} matrix view of the ggml {@code [kw, kh, cIn,
     * cOut]} kernel. Returns the planar {@code [ow, oh, cOut]} image, bias per output channel when non-null.
     */
    static float[] conv2d(CpuMatVec matVec, float[] img, int w, int h, int cIn, CpuTensorRef kernel, int kw, int kh,
                          int stride, int pad, float[] bias) {
        int ow = convOut(w, kw, stride, pad);
        int oh = convOut(h, kh, stride, pad);
        int span = kw * kh * cIn;
        int nPix = ow * oh;
        float[] cols = new float[Math.multiplyExact(nPix, span)];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int base = (oy * ow + ox) * span;
                for (int ci = 0; ci < cIn; ci++) {
                    for (int ky = 0; ky < kh; ky++) {
                        int sy = oy * stride + ky - pad;
                        for (int kx = 0; kx < kw; kx++) {
                            int sx = ox * stride + kx - pad;
                            boolean in = sy >= 0 && sy < h && sx >= 0 && sx < w;
                            cols[base + kx + kw * (ky + kh * ci)] = in ? img[sx + w * (sy + h * ci)] : 0f;
                        }
                    }
                }
            }
        }
        int cOut = (int) kernel.rows();
        float[] rows = new float[nPix * cOut];
        matVec.matmul(kernel, cols, 0, rows, 0, nPix);
        float[] out = new float[rows.length];
        for (int p = 0; p < nPix; p++) {
            for (int co = 0; co < cOut; co++) {
                out[p + nPix * co] = rows[p * cOut + co] + (bias == null ? 0f : bias[co]);
            }
        }
        return out;
    }

    /**
     * {@code ggml_conv_2d_dw_direct(kernel, img, s, s, p, p, 1, 1)}: every channel of the planar {@code [w, h, c]}
     * image convolved with its own {@code [kw, kh]} plane of the ggml {@code [kw, kh, 1, c]} kernel, plus the
     * channel bias when non-null. Returns the planar {@code [ow, oh, c]} image.
     */
    static float[] depthwiseConv2d(float[] img, int w, int h, int c, float[] kernel, int kw, int kh, int stride,
                                   int pad, float[] bias) {
        int ow = convOut(w, kw, stride, pad);
        int oh = convOut(h, kh, stride, pad);
        float[] out = new float[ow * oh * c];
        for (int ch = 0; ch < c; ch++) {
            float b = bias == null ? 0f : bias[ch];
            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    float acc = 0f;
                    for (int ky = 0; ky < kh; ky++) {
                        int sy = oy * stride + ky - pad;
                        if (sy < 0 || sy >= h) {
                            continue;
                        }
                        for (int kx = 0; kx < kw; kx++) {
                            int sx = ox * stride + kx - pad;
                            if (sx >= 0 && sx < w) {
                                acc += kernel[kx + kw * (ky + kh * ch)] * img[sx + w * (sy + h * ch)];
                            }
                        }
                    }
                    out[ox + ow * (oy + oh * ch)] = acc + b;
                }
            }
        }
        return out;
    }

    /**
     * A depthwise convolution along time over {@code [n][c]} rows with {@code padLeft} zero frames before and
     * {@code padRight} after: {@code out[t][ch] = sum_k x[t + k - padLeft][ch] * kernel[k + kernelLen * ch]}
     * ({@code ggml_ssm_conv} over the padded sequence, or {@code ggml_conv_1d_dw} at stride 1). The output holds
     * {@code n + padLeft + padRight - kernelLen + 1} rows.
     */
    static float[] depthwiseConv1d(float[] x, int n, int c, float[] kernel, int kernelLen, int padLeft,
                                   int padRight) {
        int nOut = n + padLeft + padRight - kernelLen + 1;
        float[] out = new float[nOut * c];
        for (int t = 0; t < nOut; t++) {
            for (int ch = 0; ch < c; ch++) {
                float acc = 0f;
                for (int k = 0; k < kernelLen; k++) {
                    int src = t + k - padLeft;
                    if (src >= 0 && src < n) {
                        acc += x[src * c + ch] * kernel[k + kernelLen * ch];
                    }
                }
                out[t * c + ch] = acc;
            }
        }
        return out;
    }

    /**
     * The flatten of a planar {@code [ow, oh, c]} conv output into {@code [ow][oh * c]} token rows with feature
     * {@code f + oh * ch} ({@code ggml_permute(2, 0, 1, 3)} + reshape, the qwen3a / dots3note layout).
     */
    static float[] flattenFreqMajor(float[] img, int ow, int oh, int c) {
        float[] out = new float[ow * oh * c];
        for (int t = 0; t < ow; t++) {
            for (int ch = 0; ch < c; ch++) {
                for (int f = 0; f < oh; f++) {
                    out[t * oh * c + f + oh * ch] = img[t + ow * (f + oh * ch)];
                }
            }
        }
        return out;
    }

    /**
     * The flatten of a planar {@code [w, h, c]} conv output, time on {@code h}, into {@code [h][w * c]} token rows
     * with feature {@code x + w * ch} ({@code ggml_permute(0, 2, 1, 3)} + reshape, the conformer subsampling
     * layout).
     */
    static float[] flattenTimeOnH(float[] img, int w, int h, int c) {
        float[] out = new float[w * h * c];
        for (int t = 0; t < h; t++) {
            for (int ch = 0; ch < c; ch++) {
                System.arraycopy(img, w * (t + h * ch), out, t * w * c + w * ch, w);
            }
        }
        return out;
    }

    /**
     * NEOX-style rotary embedding ({@code ggml_rope_ext} with {@code GGML_ROPE_TYPE_NEOX}) of the first
     * {@code nRot} dims of every {@code dh}-wide head of {@code [n][heads * dh]} rows, at positions
     * {@code pos[t]}: dim {@code i < nRot / 2} pairs with {@code i + nRot / 2} at angle
     * {@code pos * theta^(-2i / nRot)}.
     */
    static void ropeNeox(float[] x, int n, int heads, int dh, int nRot, int[] pos, float theta) {
        int half = nRot / 2;
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < half; i++) {
                double ang = pos[t] * Math.pow(theta, -2.0 * i / nRot);
                float cos = (float) Math.cos(ang);
                float sin = (float) Math.sin(ang);
                for (int h = 0; h < heads; h++) {
                    int base = t * heads * dh + h * dh;
                    float a = x[base + i];
                    float b = x[base + i + half];
                    x[base + i] = a * cos - b * sin;
                    x[base + i + half] = a * sin + b * cos;
                }
            }
        }
    }

    /**
     * Normal (GPT-J interleaved) rotary embedding ({@code ggml_rope_ext} with mode 0) of every {@code dh}-wide head
     * of {@code [n][heads * dh]} rows at positions {@code pos[t]}: dims {@code 2i, 2i + 1} rotate by
     * {@code pos * theta^(-2i / dh)}.
     */
    static void ropeNormal(float[] x, int n, int heads, int dh, int[] pos, float theta) {
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < dh / 2; i++) {
                double ang = pos[t] * Math.pow(theta, -2.0 * i / dh);
                float cos = (float) Math.cos(ang);
                float sin = (float) Math.sin(ang);
                for (int h = 0; h < heads; h++) {
                    int base = t * heads * dh + h * dh + 2 * i;
                    float a = x[base];
                    float b = x[base + 1];
                    x[base] = a * cos - b * sin;
                    x[base + 1] = a * sin + b * cos;
                }
            }
        }
    }

    /** Positions {@code 0 .. n - 1}. */
    static int[] iota(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        return p;
    }
}
