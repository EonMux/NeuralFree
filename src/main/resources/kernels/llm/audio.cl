/* audio.cl - LLM_Conv1dSame and LLM_Conv1dDepthwise, the wavtokenizer-dec convolutions. Requires: common/dialect.h, common/config.h */
/*
 * Channel-major F32 layout (the ggml {L, C} tensors of models/wavtokenizer-dec.cpp): element (c, t) of a
 * signal is v[v_off + c*len + t], time innermost. Offsets are in float elements; addresses are formed in
 * nf_i64. Java: gguf.ops.audio.AudioKernels; CPU twin: gguf.ops.audio.CpuConv1d.
 *
 * Both kernels are ggml_conv_1d_ph / ggml_conv_1d_dw_ph with stride 1 (ggml.c:4639-4699): padding
 * p = K/2 (integer division, not scaled by the dilation), zero outside [0, L), and output length
 *     OL = L + 2*(K/2) - dil*(K-1)        (ggml_calc_conv_output_size, ggml.c:4562; OL = L for odd K, dil 1)
 * Input tap of output step t and kernel tap k: t + k*dil - p (im2col, ggml-cpu/ops.cpp:6802-6875).
 * The bias is added after the full dot product, as the separate ggml_add of the model graph.
 *
 * LLM_Conv1dSame(i L, i IC, i OC, i K, i dil, i x_off, i y_off, R x, R w, R b, W y)
 *     x[IC][L], w[OC][IC][K] (ggml {K, IC, OC}), b[OC] (ggml {1, OC}), y[OC][OL]:
 *     y[oc][t] = (sum_{ic asc} sum_{k asc, 0 <= t+k*dil-p < L} w[oc][ic][k] * x[ic][t+k*dil-p]) + b[oc]
 *     (the im2col row index ic*K + k of the mul_mat, ggml.c:4647-4651).
 *
 * LLM_Conv1dDepthwise(i L, i C, i K, i x_off, i y_off, R x, R w, R b, W y)
 *     x[C][L], w[C][K] (ggml {K, 1, C}), b[C], y[C][OL] with dil = 1:
 *     y[c][t] = (sum_{k asc, 0 <= t+k-p < L} w[c][k] * x[c][t+k-p]) + b[c]
 *     (ggml_conv_1d_dw: per-channel mul_mat of the im2col rows with the channel's taps, ggml.c:4672-4688).
 *
 * Dispatch: 1-D, one work-item per output element (index = channel*OL + t, so neighbouring work-items
 * read neighbouring x), global = roundUp(channels*OL, L), local = {L}. No barriers. y must not
 * overlap x.
 */

/* Output length of a stride-1 half-padded convolution. */
NF_INLINE int nf_audio_conv_out_len(const int len, const int k, const int dil) {
    return len + 2 * (k / 2) - dil * (k - 1);
}

NF_KERNEL void LLM_Conv1dSame(
    const int L,
    const int IC,
    const int OC,
    const int K,
    const int dil,
    const int x_off,
    const int y_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int OL = nf_audio_conv_out_len(L, K, dil);
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (OL <= 0 || idx >= (nf_i64)OC * OL) return;
    const int oc = (int)(idx / OL);
    const int t = (int)(idx - (nf_i64)oc * OL);
    const int pad = K / 2;
    const nf_i64 w_oc = (nf_i64)oc * IC * K;
    float acc = 0.0f;
    for (int ic = 0; ic < IC; ic++) {
        const nf_i64 x_row = (nf_i64)x_off + (nf_i64)ic * L;
        const nf_i64 w_row = w_oc + (nf_i64)ic * K;
        for (int k = 0; k < K; k++) {
            const int src = t + k * dil - pad;
            if (src >= 0 && src < L) {
                acc += w[w_row + k] * x[x_row + src];
            }
        }
    }
    y[(nf_i64)y_off + idx] = acc + b[oc];
}

NF_KERNEL void LLM_Conv1dDepthwise(
    const int L,
    const int C,
    const int K,
    const int x_off,
    const int y_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int OL = nf_audio_conv_out_len(L, K, 1);
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (OL <= 0 || idx >= (nf_i64)C * OL) return;
    const int c = (int)(idx / OL);
    const int t = (int)(idx - (nf_i64)c * OL);
    const int pad = K / 2;
    const nf_i64 x_row = (nf_i64)x_off + (nf_i64)c * L;
    const nf_i64 w_row = (nf_i64)c * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const int src = t + k - pad;
        if (src >= 0 && src < L) {
            acc += w[w_row + k] * x[x_row + src];
        }
    }
    y[(nf_i64)y_off + idx] = acc + b[c];
}
