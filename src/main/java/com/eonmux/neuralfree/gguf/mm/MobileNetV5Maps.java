package com.eonmux.neuralfree.gguf.mm;

/**
 * The spatial operations of the MobileNetV5 graph over host feature maps, each a {@code [h * w][c]} array of
 * pixel rows in raster order (ggml's {@code [W, H, C]} map, channels innermost): the SAME padding of
 * {@code pad_same_2d}, the output side of a conv, the depthwise {@code ggml_conv_2d_dw}, the average pool of
 * {@code ggml_pool_2d}, the nearest upscale of {@code ggml_upscale} and the channel concat of
 * {@code ggml_concat}. Values stay F32 throughout. Static methods only; every result is a new array. The CPU
 * executor runs the graph's depthwise convolutions and pools here; the GPU executor runs them as the device
 * kernels of {@code gguf.ops.vision.VisionKernels}.
 */
final class MobileNetV5Maps {

    private MobileNetV5Maps() {
    }

    /**
     * The SAME padding before a {@code k}-wide stride-{@code s} conv over {@code size} pixels
     * ({@code pad_same_2d}): {@code {before, total}}, the output covering {@code ceil(size / s)} pixels and any odd
     * pixel padded after.
     */
    static int[] samePad(int size, int k, int s) {
        int out = (size + s - 1) / s;
        int total = Math.max(0, (out - 1) * s + k - size);
        return new int[]{total / 2, total};
    }

    /** The output side of a {@code k}-wide stride-{@code s} conv over {@code size} pixels padded by {@code total}. */
    static int convOut(int size, int k, int s, int total) {
        int out = (size + total - k) / s + 1;
        if (out <= 0) {
            throw new IllegalArgumentException("a " + k + "-wide conv does not fit " + size + " pixels padded by "
                    + total);
        }
        return out;
    }

    /**
     * {@code ggml_conv_2d_dw}: per channel, the {@code k x k} stride-{@code s} kernel {@code weight} (value
     * {@code (ch * k + ky) * k + kx}) over the input, zero outside the map.
     */
    static float[] depthwise(float[] x, int w, int h, int c, float[] weight, int k, int s, int padY, int padX,
                             int ow, int oh) {
        float[] out = new float[Math.multiplyExact(ow * oh, c)];
        float[] taps = new float[k * k];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int dst = (oy * ow + ox) * c;
                for (int ch = 0; ch < c; ch++) {
                    for (int ky = 0; ky < k; ky++) {
                        int iy = oy * s + ky - padY;
                        for (int kx = 0; kx < k; kx++) {
                            int ix = ox * s + kx - padX;
                            boolean inside = iy >= 0 && iy < h && ix >= 0 && ix < w;
                            taps[ky * k + kx] = inside ? x[(iy * w + ix) * c + ch] : 0f;
                        }
                    }
                    float sum = 0f;
                    int base = ch * k * k;
                    for (int t = 0; t < taps.length; t++) {
                        sum += weight[base + t] * taps[t];
                    }
                    out[dst + ch] = sum;
                }
            }
        }
        return out;
    }

    /** {@code ggml_pool_2d(AVG, ky, kx, ky, kx, 0, 0)}: the mean of every whole {@code ky x kx} window. */
    static float[] avgPool(float[] x, int w, int h, int c, int ky, int kx, int ow, int oh) {
        float[] out = new float[Math.multiplyExact(ow * oh, c)];
        float window = (float) ky * kx;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int dst = (oy * ow + ox) * c;
                for (int ch = 0; ch < c; ch++) {
                    float sum = 0f;
                    for (int dy = 0; dy < ky; dy++) {
                        for (int dx = 0; dx < kx; dx++) {
                            sum += x[((oy * ky + dy) * w + ox * kx + dx) * c + ch];
                        }
                    }
                    out[dst + ch] = sum / window;
                }
            }
        }
        return out;
    }

    /** {@code ggml_upscale(NEAREST)} by the integer factor {@code f} on both sides. */
    static float[] upscaleNearest(float[] x, int w, int h, int c, int f) {
        int ow = w * f;
        int oh = h * f;
        float[] out = new float[Math.multiplyExact(ow * oh, c)];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                System.arraycopy(x, ((oy / f) * w + ox / f) * c, out, (oy * ow + ox) * c, c);
            }
        }
        return out;
    }

    /** {@code ggml_concat} along the channels: pixel {@code p} is pixel {@code p} of {@code a}, then of {@code b}. */
    static float[] concat(float[] a, int ca, float[] b, int cb, int pixels) {
        float[] out = new float[Math.multiplyExact(pixels, ca + cb)];
        for (int p = 0; p < pixels; p++) {
            System.arraycopy(a, p * ca, out, p * (ca + cb), ca);
            System.arraycopy(b, p * cb, out, p * (ca + cb) + ca, cb);
        }
        return out;
    }
}
