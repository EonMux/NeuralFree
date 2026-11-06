package com.eonmux.neuralfree.gguf.mm;

/**
 * The 2D rotary position encodings of the pixtral, llama4, kimivl and gemma4v vision encoders, applied in place to
 * Q or K rows of {@code nHead * dHead} values per token; the qwen-rope encoders (qwen2vl, qwen3vl, paddleocr, dots)
 * run the m-rope of {@link QwenVisionGrid#rope} through {@code CpuRope} ({@link ClipVit#qwenRope}). Package-private
 * executor support.
 */
final class VisionRope {

    private VisionRope() {
    }

    /**
     * clip.cpp {@code build_rope_2d}: the first half of every head rotates with {@code posA}, the second half
     * with {@code posB}, each half a normal-mode rope over {@code d_head / 2} dims (adjacent pairs
     * {@code (2i, 2i + 1)} at {@code pos * base^(-4i / d_head)}); with {@code interleaveFreq} the second half's
     * frequencies shift by {@code base^(-2 / d_head)}, so the two halves take the even and odd frequencies of one
     * full-width rope.
     */
    static void twoHalves(float[] x, int[] posA, int[] posB, int nTokens, int nHead, int dh, double base,
                          boolean interleaveFreq) {
        int width = nHead * dh;
        int half = dh / 2;
        double scaleOdd = interleaveFreq ? Math.pow(base, -2.0 / dh) : 1.0;
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < nHead; h++) {
                int head = t * width + h * dh;
                for (int i = 0; 2 * i < half; i++) {
                    double freq = Math.pow(base, -4.0 * i / dh);
                    rotate(x, head + 2 * i, head + 2 * i + 1, posA[t] * freq);
                    rotate(x, head + half + 2 * i, head + half + 2 * i + 1, scaleOdd * posB[t] * freq);
                }
            }
        }
    }

    /**
     * The Gemma 4 vision rope (models/gemma4v.cpp {@code add_pos}): the first half of every head rotates with
     * {@code posA}, the second half with {@code posB}, each half a NEOX-mode rope over its {@code d_head / 2} dims
     * (pairs {@code (i, i + d_head / 4)} at {@code pos * base^(-4i / d_head)}).
     */
    static void neoxHalves(float[] x, int[] posA, int[] posB, int nTokens, int nHead, int dh, double base) {
        int width = nHead * dh;
        int half = dh / 2;
        int quarter = half / 2;
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < nHead; h++) {
                int head = t * width + h * dh;
                for (int i = 0; i < quarter; i++) {
                    double freq = Math.pow(base, -4.0 * i / dh);
                    rotate(x, head + i, head + i + quarter, posA[t] * freq);
                    rotate(x, head + half + i, head + half + i + quarter, posB[t] * freq);
                }
            }
        }
    }

    private static void rotate(float[] x, int i0, int i1, double angle) {
        float cos = (float) Math.cos(angle);
        float sin = (float) Math.sin(angle);
        float x0 = x[i0];
        float x1 = x[i1];
        x[i0] = x0 * cos - x1 * sin;
        x[i1] = x0 * sin + x1 * cos;
    }
}
