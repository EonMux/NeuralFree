/* recurrent/ssm_conv.cl - depthwise conv1d kernels: LLM_SSM_Conv (causal, in-place conv-state update) and LLM_ShortConvCentered (centered, stateless). Requires: common/dialect.h, common/config.h, common/math.h */
/*
 * Depthwise 1-D convolutions over token-major activations x[t][c] (element offset x_off, row stride
 * x_stride), with weights w[c][k] (ggml ssm_conv1d {d_conv, n_ch}, F32) and no barriers.
 *
 * LLM_SSM_Conv (ggml_ssm_conv, ggml-cpu/ops.cpp:10022-10066, and the graph around it in
 * models/mamba-base.cpp:57-82, 207-237 and models/lfm-base.cpp:45-75):
 *   xs[c] = concat(state[c][0 .. d_conv-2], x[0 .. n_tokens-1][c])            (j = 0 oldest)
 *   out[t][c] = silu?( sum_{k < d_conv} xs[c][t + k] * w[c][k]  (+ b[c]) )      flags: 1 bias, 2 silu
 *   state[c][j] = xs[c][n_tokens + j] for j < d_conv - 1                         (the last d_conv-1 inputs)
 * The conv state has the llama.cpp layout conv[c][j] (time innermost). Row r of the state buffer starts at
 * r * state_stride floats (state_stride = floats per sequence); the dispatch covers the channel slice that
 * starts state_off channels into the row, so channel c of this dispatch is
 *   conv_state[state_rows[0] * state_stride + (state_off + c) * (d_conv - 1) + j].
 * A slice dispatch serves the KDA [q | k | v] conv state with one dispatch per slice.
 * One work-item per channel loops over the tokens; it alone reads and writes its state row, so the in-place
 * update is race-free, and it reads the old state only before overwriting it (new[j] reads old[n_tokens + j],
 * an index above every index written before it). Decode and prefill use the same kernel.
 * Aliasing: x and out are distinct (out[t] is written while x[t + 1 ..] is still read through the window).
 * Dispatch: 1-D, global = roundUp(n_ch, L), any L.
 *
 * LLM_ShortConvCentered (LFM2 non-causal shortconv, models/lfm2.cpp:349-380):
 *   p = (d_conv - 1) / 2, out[t][c] = x[t][c] * w[c][p] + sum_{k != p, 0 <= t+k-p < n_tokens} x[t+k-p][c] * w[c][k]
 * i.e. a centered window zero-padded on both sides of the ubatch, without state. The center tap is summed
 * first and the others in ascending k, the order of llama.cpp (y = bx*tap1 + prv*tap0 + nxt*tap2).
 * d_conv is odd. Dispatch: 1-D, one work-item per (t, c), global = roundUp(n_tokens * n_ch, L).
 */

NF_KERNEL void LLM_SSM_Conv(
        const int n_ch,
        const int d_conv,
        const int n_tokens,
        const int flags,
        const int x_stride,
        const int x_off,
        const int out_stride,
        const int out_off,
        const int state_off,
        const int state_stride,
        NF_GLOBAL const float* NF_RESTRICT x,
        NF_GLOBAL const float* NF_RESTRICT w,
        NF_GLOBAL const float* NF_RESTRICT b,
        NF_GLOBAL float* conv_state,
        NF_GLOBAL const int* NF_RESTRICT state_rows,
        NF_GLOBAL float* NF_RESTRICT out) {
    const int c = (int)get_global_id(0);
    if (c >= n_ch) {
        return;
    }
    const int steps = d_conv - 1;
    const nf_i64 s_base = (nf_i64)state_rows[0] * state_stride + (nf_i64)(state_off + c) * steps;
    const nf_i64 w_base = (nf_i64)c * d_conv;
    const float bias = (flags & 1) ? b[c] : 0.0f;

    for (int t = 0; t < n_tokens; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < d_conv; ++k) {
            const int src = t + k - steps;
            const float v = src >= 0 ? x[(nf_i64)x_off + (nf_i64)src * x_stride + c] : conv_state[s_base + t + k];
            sum += v * w[w_base + k];
        }
        if (flags & 1) {
            sum += bias;
        }
        if (flags & 2) {
            sum = nf_silu(sum);
        }
        out[(nf_i64)out_off + (nf_i64)t * out_stride + c] = sum;
    }

    for (int j = 0; j < steps; ++j) {
        const int src = n_tokens + j - steps;
        conv_state[s_base + j] = src >= 0 ? x[(nf_i64)x_off + (nf_i64)src * x_stride + c]
                                          : conv_state[s_base + n_tokens + j];
    }
}

NF_KERNEL void LLM_ShortConvCentered(
        const int n_ch,
        const int d_conv,
        const int n_tokens,
        const int x_stride,
        const int x_off,
        const int out_stride,
        NF_GLOBAL const float* NF_RESTRICT x,
        NF_GLOBAL const float* NF_RESTRICT w,
        NF_GLOBAL float* NF_RESTRICT out) {
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (i >= (nf_i64)n_tokens * n_ch) {
        return;
    }
    const int t = (int)(i / n_ch);
    const int c = (int)(i - (nf_i64)t * n_ch);
    const int pad = (d_conv - 1) / 2;
    const nf_i64 w_base = (nf_i64)c * d_conv;

    float sum = x[(nf_i64)x_off + (nf_i64)t * x_stride + c] * w[w_base + pad];
    for (int k = 0; k < d_conv; ++k) {
        const int src = t + k - pad;
        if (k != pad && src >= 0 && src < n_tokens) {
            sum += x[(nf_i64)x_off + (nf_i64)src * x_stride + c] * w[w_base + k];
        }
    }
    out[(nf_i64)t * out_stride + c] = sum;
}
