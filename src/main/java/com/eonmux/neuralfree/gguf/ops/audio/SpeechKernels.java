package com.eonmux.neuralfree.gguf.ops.audio;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Names, argument order and launch geometry of the speech-generator kernels ({@code kernels/llm/speech.cl} and the
 * GLSL twins under {@code shaders/llm}): the causal convolutions, activations, attention and RVQ / row primitives of
 * the qwen3tts code predictor and code2wav decoder. Signals are token/frame major ({@code [rows][channels]},
 * element {@code (t, c)} at {@code v[t*channels + c]}); every scalar is a 32-bit {@code int} unless marked
 * {@code f}. Arguments are passed scalars first, then buffers, in the order the kernels declare them.
 *
 * <ul>
 *   <li>{@link #CONV1D_CAUSAL} {@code (n, ic, oc, k, dil, pad; x, w, b, state, y, nstate)}: dense causal conv,
 *       {@code w[OC][IC][K]}.</li>
 *   <li>{@link #CONV1D_DEPTHWISE_CAUSAL} {@code (n, c, k, pad; x, w, b, state, y, nstate)}: depthwise causal conv,
 *       {@code w[C][K]}.</li>
 *   <li>{@link #CONV_TRANSPOSE1D_CAUSAL} {@code (n, ic, oc, k, stride; x, w, b, state, y, nstate)}: causal
 *       ConvTranspose1d, {@code w[IC][OC][K]}, overlap tail in {@code state}/{@code nstate}.</li>
 *   <li>{@link #CONV_TRANSPOSE1D_DEPTHWISE_CAUSAL} {@code (n, c, k, stride; x, w, b, state, y, nstate)}: depthwise
 *       causal ConvTranspose1d, {@code w[C][K]}, overlap tail in {@code state}/{@code nstate}.</li>
 *   <li>{@link #SNAKE_BETA} {@code (n, channels; x, alpha, beta, y)}: {@code x + sin(alpha*x)^2 * beta}.</li>
 *   <li>{@link #ELU} {@code (n; x, y)}: {@code x > 0 ? x : exp(x) - 1}.</li>
 *   <li>{@link #HEAD_RMS_NORM_ROPE} {@code (n_heads, head_dim, pos, theta_scale(f), eps(f); x, w, y)}: per-head RMS
 *       norm and NEOX RoPE at one step position.</li>
 *   <li>{@link #TTS_CAUSAL_ATTN} {@code (n_head, n_head_kv, d_head, pos; q, keys, values, out)}: one step's causal
 *       attention over cache rows {@code 0..pos}; one work-group per head ({@link #causalAttentionGlobal}).</li>
 *   <li>{@link #TTS_WINDOW_ATTN} {@code (n_q, n_head, n_head_kv, d_head, prefix, window, base; q, keys, values,
 *       out)}: the banded causal window attention over a persisted prefix; one work-group per (query row,
 *       head).</li>
 *   <li>{@link #RVQ_SUM_ROWS} {@code (n_pages, n_frames, dim; x, y)}: per-frame sum of a codebook group's rows.</li>
 *   <li>{@link #MATVEC_ROWS_F32} {@code (cols, rows, row_off; w, x, y)}: one page of a stacked F32 LM head.</li>
 *   <li>{@link #RMS_NORM_ROPE_ROWS} {@code (rows, n_heads, head_dim, pos_base, off, theta_scale(f), eps(f); x, w,
 *       y)}: per-head RMS norm plus NEOX RoPE over a block of rows.</li>
 *   <li>{@link #NORMAL_ROPE_ROWS} {@code (rows, n_heads, head_dim, pos_base, off, theta_scale(f); x, y)}: per-head
 *       NORMAL (adjacent-pair) RoPE over a block of rows.</li>
 *   <li>{@link #TTS_FULL_ATTN} {@code (n_pos, n_head, n_head_kv, d_head; q, keys, values, out)}: bidirectional
 *       attention over every row; one work-group per (row, head).</li>
 *   <li>{@link #HEAD_SIGMOID_GATE} {@code (n_rows, n_head, d_head; x, gate, y)}: per-head sigmoid gate.</li>
 *   <li>{@link #ROW_SCALE_ADD} {@code (rows, cols, scale_off; a, b, scale, y)}: {@code a + b * scale[c]}.</li>
 *   <li>{@link #ROW_AFFINE} {@code (rows, cols; x, scale, bias, y)}: per-channel affine.</li>
 *   <li>{@link #ADA_IN_MODULATE} {@code (rows, cols; x, mod, y)}: AdaLN modulation of a normalized row.</li>
 *   <li>{@link #RMS_NORM_UNBIASED} {@code (n, eps(f); x, w, y)}: the flow time embedder's unbiased-variance
 *       normalization.</li>
 *   <li>{@link #REPEAT_ROWS} {@code (src_rows, repeat, cols; x, y)}: nearest-neighbour row repeat.</li>
 * </ul>
 */
public final class SpeechKernels {

    public static final String CONV1D_CAUSAL = "LLM_Conv1dCausal";
    public static final String CONV1D_DEPTHWISE_CAUSAL = "LLM_Conv1dDepthwiseCausal";
    public static final String CONV_TRANSPOSE1D_CAUSAL = "LLM_ConvTranspose1dCausal";
    public static final String CONV_TRANSPOSE1D_DEPTHWISE_CAUSAL = "LLM_ConvTranspose1dDepthwiseCausal";
    public static final String SNAKE_BETA = "LLM_SnakeBeta";
    public static final String ELU = "LLM_ELU";
    public static final String HEAD_RMS_NORM_ROPE = "LLM_HeadRmsNormRope";
    public static final String TTS_CAUSAL_ATTN = "LLM_TtsCausalAttn";
    public static final String TTS_WINDOW_ATTN = "LLM_TtsWindowAttn";
    public static final String RVQ_SUM_ROWS = "LLM_RvqSumRows";
    public static final String MATVEC_ROWS_F32 = "LLM_MatVecRowsF32";
    public static final String RMS_NORM_ROPE_ROWS = "LLM_RmsNormRopeRows";
    public static final String NORMAL_ROPE_ROWS = "LLM_NormalRopeRows";
    public static final String TTS_FULL_ATTN = "LLM_TtsFullAttn";
    public static final String HEAD_SIGMOID_GATE = "LLM_HeadSigmoidGate";
    public static final String ROW_SCALE_ADD = "LLM_RowScaleAdd";
    public static final String ROW_AFFINE = "LLM_RowAffine";
    public static final String ADA_IN_MODULATE = "LLM_AdaInModulate";
    public static final String RMS_NORM_UNBIASED = "LLM_RmsNormUnbiased";
    public static final String REPEAT_ROWS = "LLM_RepeatRows";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.unmodifiableList(Arrays.asList(
            CONV1D_CAUSAL, CONV1D_DEPTHWISE_CAUSAL, CONV_TRANSPOSE1D_CAUSAL, CONV_TRANSPOSE1D_DEPTHWISE_CAUSAL,
            SNAKE_BETA, ELU, HEAD_RMS_NORM_ROPE,
            TTS_CAUSAL_ATTN, TTS_WINDOW_ATTN, RVQ_SUM_ROWS, MATVEC_ROWS_F32, RMS_NORM_ROPE_ROWS, NORMAL_ROPE_ROWS,
            TTS_FULL_ATTN, HEAD_SIGMOID_GATE, ROW_SCALE_ADD, ROW_AFFINE, ADA_IN_MODULATE, RMS_NORM_UNBIASED,
            REPEAT_ROWS));

    /** Work-group size of the element-wise, convolution and attention launches. */
    public static final int LOCAL = 64;

    private SpeechKernels() {
    }

    /** Global size of {@link #CONV1D_CAUSAL}: the larger of its output and next-state element counts. */
    public static long convGlobal(int n, int ic, int oc, int pad) {
        return roundUp(Math.max((long) n * oc, (long) pad * ic), LOCAL);
    }

    /** Global size of {@link #CONV1D_DEPTHWISE_CAUSAL}: the larger of its output and next-state element counts. */
    public static long depthwiseGlobal(int n, int channels, int pad) {
        return roundUp(Math.max((long) n * channels, (long) pad * channels), LOCAL);
    }

    /** Global size of {@link #CONV_TRANSPOSE1D_DEPTHWISE_CAUSAL}. */
    public static long depthwiseTransposeGlobal(int n, int channels, int k, int stride) {
        int emit = n * stride;
        int trim = k - stride;
        return roundUp(Math.max((long) emit * channels, (long) trim * channels), LOCAL);
    }

    /** Global size of {@link #CONV_TRANSPOSE1D_CAUSAL}: the larger of its emit and next-tail element counts. */
    public static long transposeGlobal(int n, int oc, int k, int stride) {
        int emit = n * stride;
        int trim = k - stride;
        return roundUp(Math.max((long) emit * oc, (long) trim * oc), LOCAL);
    }

    /** Global size of an element-wise kernel over {@code n} elements. */
    public static long elementGlobal(long n) {
        return roundUp(n, LOCAL);
    }

    /** Global size of {@link #TTS_CAUSAL_ATTN}: one work-group of {@link #LOCAL} per head. */
    public static long causalAttentionGlobal(int nHead) {
        return (long) nHead * LOCAL;
    }

    /** Global size of {@link #TTS_WINDOW_ATTN}: one work-group of {@link #LOCAL} per query row and head. */
    public static long windowAttentionGlobal(int nQ, int nHead) {
        return (long) nQ * nHead * LOCAL;
    }

    /** Global size of {@link #HEAD_RMS_NORM_ROPE}: one work-item per head and rotation pair. */
    public static long headNormRopeGlobal(int nHeads, int headDim) {
        return roundUp((long) nHeads * (headDim / 2), LOCAL);
    }

    /** Global size of {@link #RMS_NORM_ROPE_ROWS} and {@link #NORMAL_ROPE_ROWS}. */
    public static long ropeRowsGlobal(int rows, int nHeads, int headDim) {
        return roundUp((long) rows * nHeads * (headDim / 2), LOCAL);
    }

    /** Global size of {@link #TTS_FULL_ATTN}: one work-group of {@link #LOCAL} per (row, head). */
    public static long fullAttentionGlobal(int nPos, int nHead) {
        return (long) nPos * nHead * LOCAL;
    }

    /** Global size of {@link #HEAD_SIGMOID_GATE}: one work-item per (row, head). */
    public static long headGateGlobal(int rows, int nHead) {
        return roundUp((long) rows * nHead, LOCAL);
    }

    /** {@code n} rounded up to a multiple of {@link #LOCAL}. */
    public static long roundUp(long n) {
        return roundUp(n, LOCAL);
    }

    private static long roundUp(long n, int local) {
        return (n + local - 1) / local * local;
    }
}
