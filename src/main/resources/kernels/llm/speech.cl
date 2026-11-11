/* speech.cl - causal convolution, activation and attention kernels of the speech generators: the code2wav DAC
 * decoder of qwen3tts (tools/mtmd/models/qwen3tts-gen.cpp:290-700) and the speech-generator graph primitives that
 * decode windows of codec codes into PCM. Requires: common/dialect.h, common/config.h */
/*
 * Layout. Every signal is token/frame major (`[rows][channels]`, element (t, c) at v[t*channels + c]), unlike the
 * channel-major wavtokenizer convolutions of audio.cl; the transpose-free layout matches the token-major rows the
 * matrix kernels write. Convolution weights keep the ggml order of the loader:
 *   LLM_Conv1dCausal            w[OC][IC][K]   (ggml {K, IC, OC})
 *   LLM_Conv1dDepthwiseCausal   w[C][K]        (ggml {K, 1, C})
 *   LLM_ConvTranspose1dCausal   w[IC][OC][K]   (ggml {K, OC, IC})
 * All offsets are float elements. The causal convolutions prepend the persisted left context of the previous
 * call instead of zero padding and write the next context: row j of `nstate` is the full signal's row
 * n + j, where the full signal is `state` (pad rows) followed by `x` (n rows), so a stream of batches produces
 * the samples of the whole sequence.
 *
 * Dispatch: one work-item per output element (and, in the same launch, one per new state element), global =
 * roundUp(max(output elements, state elements), L), local = {L}. Work-items return before touching a buffer they
 * own no element of; a zero-sized state binds a placeholder and is never read or written.
 */

/* Output element (t, oc) of the causal convolution over the full signal. */
NF_INLINE float nf_speech_conv_at(const int t, const int oc, const int ic_n, const int k, const int dil,
                                  const int pad, NF_GLOBAL const float* x, NF_GLOBAL const float* w,
                                  NF_GLOBAL const float* state)
{
    const nf_i64 w_oc = (nf_i64)oc * ic_n * k;
    float acc = 0.0f;
    for (int c = 0; c < ic_n; c++) {
        const nf_i64 w_row = w_oc + (nf_i64)c * k;
        for (int j = 0; j < k; j++) {
            const int r = t + j * dil;
            const float v = r < pad ? state[(nf_i64)r * ic_n + c] : x[(nf_i64)(r - pad) * ic_n + c];
            acc += w[w_row + j] * v;
        }
    }
    return acc;
}

/* Row j of the next left context: the full signal's row n + j. */
NF_INLINE float nf_speech_next_state(const int n, const int j, const int c, const int ic_n, const int pad,
                                     NF_GLOBAL const float* x, NF_GLOBAL const float* state)
{
    const int r = n + j;
    return r < pad ? state[(nf_i64)r * ic_n + c] : x[(nf_i64)(r - pad) * ic_n + c];
}

/*
 * Causal convolution, stride 1, dilation > 0:
 *   y[t][oc] = b[oc] + sum_{c,j} w[oc][c][j] * full[t + j*dil][c],  full = state ++ x
 *   nstate[j][c] = full[n + j][c]                                    for j < pad = (K-1)*dil
 * x[n][IC], w[OC][IC][K], b[OC], state[pad][IC], y[n][OC], nstate[pad][IC].
 */
NF_KERNEL void LLM_Conv1dCausal(
    const int n,
    const int ic_n,
    const int oc_n,
    const int k,
    const int dil,
    const int pad,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL const float* NF_RESTRICT state,
    NF_GLOBAL float* NF_RESTRICT y,
    NF_GLOBAL float* NF_RESTRICT nstate)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    const nf_i64 out_n = (nf_i64)n * oc_n;
    if (idx < out_n) {
        const int t = (int)(idx / oc_n);
        const int oc = (int)(idx - (nf_i64)t * oc_n);
        y[idx] = b[oc] + nf_speech_conv_at(t, oc, ic_n, k, dil, pad, x, w, state);
    }
    const nf_i64 state_n = (nf_i64)pad * ic_n;
    if (idx < state_n) {
        const int j = (int)(idx / ic_n);
        const int c = (int)(idx - (nf_i64)j * ic_n);
        nstate[idx] = nf_speech_next_state(n, j, c, ic_n, pad, x, state);
    }
}

/*
 * Causal depthwise convolution, stride 1, dilation 1:
 *   y[t][c] = b[c] + sum_j w[c][j] * full[t + j][c],  full = state ++ x
 *   nstate[j][c] = full[n + j][c]                     for j < pad = K-1
 * x[n][C], w[C][K] (ggml {K, 1, C}), b[C], state[pad][C], y[n][C], nstate[pad][C].
 */
NF_KERNEL void LLM_Conv1dDepthwiseCausal(
    const int n,
    const int c_n,
    const int k,
    const int pad,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL const float* NF_RESTRICT state,
    NF_GLOBAL float* NF_RESTRICT y,
    NF_GLOBAL float* NF_RESTRICT nstate)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    const nf_i64 out_n = (nf_i64)n * c_n;
    if (idx < out_n) {
        const int t = (int)(idx / c_n);
        const int c = (int)(idx - (nf_i64)t * c_n);
        const nf_i64 w_row = (nf_i64)c * k;
        float acc = b[c];
        for (int j = 0; j < k; j++) {
            const int r = t + j;
            const float v = r < pad ? state[(nf_i64)r * c_n + c] : x[(nf_i64)(r - pad) * c_n + c];
            acc += w[w_row + j] * v;
        }
        y[idx] = acc;
    }
    const nf_i64 state_n = (nf_i64)pad * c_n;
    if (idx < state_n) {
        const int j = (int)(idx / c_n);
        const int c = (int)(idx - (nf_i64)j * c_n);
        nstate[idx] = nf_speech_next_state(n, j, c, c_n, pad, x, state);
    }
}

/* The transposed convolution's full signal at output row `i`: contributions of the input rows it covers. */
NF_INLINE float nf_speech_transpose_at(const int i, const int oc, const int ic_n, const int oc_n, const int n,
                                       const int k, const int stride, NF_GLOBAL const float* x,
                                       NF_GLOBAL const float* w)
{
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        if (i < j || (i - j) % stride != 0) {
            continue;
        }
        const int t = (i - j) / stride;
        if (t >= n) {
            continue;
        }
        for (int c = 0; c < ic_n; c++) {
            acc += w[((nf_i64)c * oc_n + oc) * k + j] * x[(nf_i64)t * ic_n + c];
        }
    }
    return acc;
}

/*
 * Causal ConvTranspose1d: the emitted rows are the full transposed convolution minus its overlap tail, which is
 * added from the previous call's state and replaced by the new tail:
 *   trim = K - stride, emit = n * stride
 *   y[i][oc] = b[oc] + full[i][oc] + (i < trim ? state[i][oc] : 0)
 *   nstate[i][oc] = full[emit + i][oc]                                  for i < trim
 * x[n][IC], w[IC][OC][K] (ggml {K, OC, IC}), b[OC], state[trim][OC], y[emit][OC], nstate[trim][OC].
 */
NF_KERNEL void LLM_ConvTranspose1dCausal(
    const int n,
    const int ic_n,
    const int oc_n,
    const int k,
    const int stride,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL const float* NF_RESTRICT state,
    NF_GLOBAL float* NF_RESTRICT y,
    NF_GLOBAL float* NF_RESTRICT nstate)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    const int emit = n * stride;
    const int trim = k - stride;
    const nf_i64 out_n = (nf_i64)emit * oc_n;
    if (idx < out_n) {
        const int i = (int)(idx / oc_n);
        const int oc = (int)(idx - (nf_i64)i * oc_n);
        float acc = b[oc] + nf_speech_transpose_at(i, oc, ic_n, oc_n, n, k, stride, x, w);
        if (i < trim) {
            acc += state[(nf_i64)i * oc_n + oc];
        }
        y[idx] = acc;
    }
    const nf_i64 state_n = (nf_i64)trim * oc_n;
    if (idx < state_n) {
        const int i = (int)(idx / oc_n);
        const int oc = (int)(idx - (nf_i64)i * oc_n);
        nstate[idx] = nf_speech_transpose_at(emit + i, oc, ic_n, oc_n, n, k, stride, x, w);
    }
}

/* SnakeBeta over token-major rows: y[t][c] = x[t][c] + sin(alpha[c] * x[t][c])^2 * beta[c]. In place. */
NF_KERNEL void LLM_SnakeBeta(
    const int n,
    const int channels,
    NF_GLOBAL const float* x,
    NF_GLOBAL const float* alpha,
    NF_GLOBAL const float* beta,
    NF_GLOBAL float* y)
{
    const int idx = (int)get_global_id(0);
    if (idx >= n) return;
    const int c = idx % channels;
    const float v = x[idx];
    const float s = sin(alpha[c] * v);
    y[idx] = v + s * s * beta[c];
}

/* ELU: y[i] = x[i] > 0 ? x[i] : exp(x[i]) - 1. In place. */
NF_KERNEL void LLM_ELU(
    const int n,
    NF_GLOBAL const float* x,
    NF_GLOBAL float* y)
{
    const int idx = (int)get_global_id(0);
    if (idx >= n) return;
    const float v = x[idx];
    y[idx] = v > 0.0f ? v : exp(v) - 1.0f;
}

/*
 * Per-head RMS norm followed by NEOX RoPE at one position (the code predictor's q/k path):
 *   s = 1 / sqrt(mean(x^2 over the head) + eps); a = x * s * w
 *   y[i] = a[i]*cos - a[i+half]*sin,  y[i+half] = a[i]*sin + a[i+half]*cos,
 *   angle = pos * theta_scale^i, half = head_dim/2.
 * x and y must not overlap: the mean is read by every pair of a head.
 */
NF_KERNEL void LLM_HeadRmsNormRope(
    const int n_heads,
    const int head_dim,
    const int pos,
    const float theta_scale,
    const float eps,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int d_half = head_dim / 2;
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)n_heads * d_half) return;
    const int h = (int)(idx / d_half);
    const int i = (int)(idx - (nf_i64)h * d_half);
    const nf_i64 base = (nf_i64)h * head_dim;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        sum += x[base + d] * x[base + d];
    }
    const float scale = 1.0f / sqrt(sum / (float)head_dim + eps);
    const float x0 = x[base + i] * scale * w[i];
    const float x1 = x[base + i + d_half] * scale * w[i + d_half];
    const float angle = (float)pos * pow(theta_scale, (float)i);
    const float cs = cos(angle);
    const float sn = sin(angle);
    y[base + i] = x0 * cs - x1 * sn;
    y[base + i + d_half] = x0 * sn + x1 * cs;
}

/* Scaled dot product of query head `q_off` and key row `k_off`; every loop is in ascending order. */
NF_INLINE float nf_speech_dot(NF_GLOBAL const float* q, const nf_i64 q_off, NF_GLOBAL const float* keys,
                              const nf_i64 k_off, const int d_head)
{
    float dot = 0.0f;
    for (int d = 0; d < d_head; d++) {
        dot += q[q_off + d] * keys[k_off + d];
    }
    return dot;
}

/* Keys whose softmax weights one work-group stages in local memory at a time. */
#define NF_TTS_CHUNK 256
/* Output dimensions one work-item accumulates per pass over the keys. */
#define NF_TTS_ACC 4

/*
 * The attention of one (query row, head) over the visible key rows j_lo .. j_hi (an empty range writes zeros), run
 * by the whole work-group:
 *   out[o_off + d] = sum_j exp(s_j - max) * values[j][kv][d] / sum_j exp(s_j - max),
 *   s_j = (q[q_off ..] . keys[j][kv]) / sqrt(d_head)
 * Work-items take keys j_lo + lid, + L, ... for the scores (each dot in ascending d), the maximum is an exact tree
 * reduction, and each chunk's weights exp(s_j - max) are staged in local memory: work-item 0 sums them in ascending
 * j and work-item t accumulates output dimensions t, t + L, ... in ascending j. kv_off is the key head's element
 * offset inside a row of kv_ch elements. Every work-item calls it in uniform control flow.
 */
NF_INLINE void nf_tts_attend(NF_GLOBAL const float* q, const nf_i64 q_off, NF_GLOBAL const float* keys,
                             NF_GLOBAL const float* values, const int kv_ch, const int kv_off, const int d_head,
                             const int j_lo, const int j_hi, NF_GLOBAL float* out, const nf_i64 o_off,
                             NF_LOCAL_PTR float* w, NF_LOCAL_PTR float* red)
{
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const float scale = 1.0f / sqrt((float)d_head);

    float mx = -3.402823466e+38f;
    for (int j = j_lo + lid; j <= j_hi; j += lsize) {
        const float score = nf_speech_dot(q, q_off, keys, (nf_i64)j * kv_ch + kv_off, d_head) * scale;
        if (score > mx) mx = score;
    }
    red[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride && red[lid + stride] > red[lid]) {
            red[lid] = red[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d0 = 0; d0 < d_head; d0 += lsize * NF_TTS_ACC) {
        float acc[NF_TTS_ACC];
        for (int a = 0; a < NF_TTS_ACC; a++) {
            acc[a] = 0.0f;
        }
        float sum = 0.0f;
        for (int c0 = j_lo; c0 <= j_hi; c0 += NF_TTS_CHUNK) {
            const int n = min(NF_TTS_CHUNK, j_hi - c0 + 1);
            for (int t = lid; t < n; t += lsize) {
                const float score = nf_speech_dot(q, q_off, keys, (nf_i64)(c0 + t) * kv_ch + kv_off, d_head) * scale;
                w[t] = exp(score - mx);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid == 0) {
                for (int t = 0; t < n; t++) {
                    sum += w[t];
                }
            }
            for (int a = 0; a < NF_TTS_ACC; a++) {
                const int d = d0 + a * lsize + lid;
                if (d < d_head) {
                    for (int t = 0; t < n; t++) {
                        acc[a] += w[t] * values[(nf_i64)(c0 + t) * kv_ch + kv_off + d];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            red[0] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const float total = red[0];
        const float inv = total > 0.0f ? 1.0f / total : 0.0f;
        for (int a = 0; a < NF_TTS_ACC; a++) {
            const int d = d0 + a * lsize + lid;
            if (d < d_head) {
                out[o_off + d] = acc[a] * inv;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/*
 * Causal attention of one step's queries over a key/value cache of rows 0..pos:
 *   out[h][d] = softmax_j(q[h] . keys[j][kv(h)] / sqrt(d_head)) * values[j][kv(h)]
 * q[n_head*d_head], keys[(pos+1)][n_head_kv*d_head], values of the same shape, out[n_head*d_head].
 * One work-group per head (nf_tts_attend): global = {n_head * L}, local = {L}, L a power of two <= LLM_DIMX.
 */
NF_KERNEL void LLM_TtsCausalAttn(
    const int n_head,
    const int n_head_kv,
    const int d_head,
    const int pos,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const float* NF_RESTRICT keys,
    NF_GLOBAL const float* NF_RESTRICT values,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float w[NF_TTS_CHUNK];
    NF_LOCAL float red[LLM_DIMX];
    const int h = (int)get_group_id(0);
    if (h >= n_head) return;
    const int kv_head = h / (n_head / n_head_kv);
    const nf_i64 q_off = (nf_i64)h * d_head;
    nf_tts_attend(q, q_off, keys, values, n_head_kv * d_head, kv_head * d_head, d_head, 0, pos, out, q_off, w, red);
}

/*
 * Banded causal attention of a decoded window over the persisted key/value prefix followed by the window's own
 * rows (tools/mtmd/models/qwen3tts-gen.cpp:420-520): query i attends to key j when
 *   0 <= prefix + i - j < window          (the band)
 *   j - prefix + base >= 0                (the cold-start mask: no key of a negative absolute position)
 * so its visible keys are the contiguous range max(0, prefix - base, prefix + i - window + 1) .. prefix + i.
 * q[n_q][n_head*d_head], keys[prefix+n_q][n_head_kv*d_head], values of the same shape,
 * out[n_q][n_head*d_head]. One work-group per (query row, head) (nf_tts_attend), group index i * n_head + h:
 * global = {n_q * n_head * L}, local = {L}, L a power of two <= LLM_DIMX.
 */
NF_KERNEL void LLM_TtsWindowAttn(
    const int n_q,
    const int n_head,
    const int n_head_kv,
    const int d_head,
    const int prefix,
    const int window,
    const int base,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const float* NF_RESTRICT keys,
    NF_GLOBAL const float* NF_RESTRICT values,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float w[NF_TTS_CHUNK];
    NF_LOCAL float red[LLM_DIMX];
    const nf_i64 g = (nf_i64)get_group_id(0);
    if (g >= (nf_i64)n_q * n_head) return;
    const int i = (int)(g / n_head);
    const int h = (int)(g - (nf_i64)i * n_head);
    const int kv_head = h / (n_head / n_head_kv);
    const nf_i64 q_off = (nf_i64)i * n_head * d_head + (nf_i64)h * d_head;
    const int j_lo = max(max(0, prefix - base), prefix + i - window + 1);
    nf_tts_attend(q, q_off, keys, values, n_head_kv * d_head, kv_head * d_head, d_head, j_lo, prefix + i, out,
                  q_off, w, red);
}

/*
 * RVQ dequantization group sum: y[f][d] = sum_p x[p*n_frames + f][d], the per-frame sum of one codebook page's
 * embedding rows when x holds pages of the same frames (the code2wav quantizer's acoustic group and the code
 * predictor's feedback row). x[n_pages*n_frames][dim], y[n_frames][dim].
 */
NF_KERNEL void LLM_RvqSumRows(
    const int n_pages,
    const int n_frames,
    const int dim,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)n_frames * dim) return;
    const int f = (int)(idx / dim);
    const int d = (int)(idx - (nf_i64)f * dim);
    float acc = 0.0f;
    for (int p = 0; p < n_pages; p++) {
        acc += x[((nf_i64)p * n_frames + f) * dim + d];
    }
    y[idx] = acc;
}

/*
 * One page of a stacked F32 LM head applied to one row: y[r] = sum_c w[row_off + r][c] * x[c].
 * w[rows_total][cols], x[cols], y[rows].
 */
NF_KERNEL void LLM_MatVecRowsF32(
    const int cols,
    const int rows,
    const int row_off,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int r = (int)get_global_id(0);
    if (r >= rows) return;
    const nf_i64 base = (nf_i64)(row_off + r) * cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; c++) {
        acc += w[base + c] * x[c];
    }
    y[r] = acc;
}

/*
 * Per-head RMS norm followed by NEOX RoPE over a block of rows (the depthformer's and the liquid codec's q/k path):
 * row `r` is normalized and rotated at position `pos_base + r`:
 *   s = 1 / sqrt(mean(x^2 over the head) + eps); a = x * s * w
 *   y[i] = a[i]*cos - a[i+half]*sin,  y[i+half] = a[i]*sin + a[i+half]*cos,
 *   angle = (pos_base + r) * theta_scale^i, half = head_dim/2.
 * x and y are the element block `off .. off + rows*n_heads*head_dim`; they must not overlap.
 */
NF_KERNEL void LLM_RmsNormRopeRows(
    const int rows,
    const int n_heads,
    const int head_dim,
    const int pos_base,
    const int off,
    const float theta_scale,
    const float eps,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int d_half = head_dim / 2;
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)rows * n_heads * d_half) return;
    const int r = (int)(idx / ((nf_i64)n_heads * d_half));
    const int rem = (int)(idx - (nf_i64)r * n_heads * d_half);
    const int h = rem / d_half;
    const int i = rem - h * d_half;
    const nf_i64 base = off + (nf_i64)(r * n_heads + h) * head_dim;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        sum += x[base + d] * x[base + d];
    }
    const float scale = 1.0f / sqrt(sum / (float)head_dim + eps);
    const float x0 = x[base + i] * scale * w[i];
    const float x1 = x[base + i + d_half] * scale * w[i + d_half];
    const float angle = (float)(pos_base + r) * pow(theta_scale, (float)i);
    const float cs = cos(angle);
    const float sn = sin(angle);
    y[base + i] = x0 * cs - x1 * sn;
    y[base + i + d_half] = x0 * sn + x1 * cs;
}

/*
 * Per-head NORMAL RoPE over a block of rows (the mimi transformer's q/k path): adjacent pairs (2i, 2i+1) of row
 * `r` are rotated at position `pos_base + r` by `(pos_base + r) * theta_scale^i`. x and y must not overlap.
 */
NF_KERNEL void LLM_NormalRopeRows(
    const int rows,
    const int n_heads,
    const int head_dim,
    const int pos_base,
    const int off,
    const float theta_scale,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int d_half = head_dim / 2;
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)rows * n_heads * d_half) return;
    const int r = (int)(idx / ((nf_i64)n_heads * d_half));
    const int rem = (int)(idx - (nf_i64)r * n_heads * d_half);
    const int h = rem / d_half;
    const int i = rem - h * d_half;
    const nf_i64 base = off + (nf_i64)(r * n_heads + h) * head_dim;
    const float angle = (float)(pos_base + r) * pow(theta_scale, (float)i);
    const float cs = cos(angle);
    const float sn = sin(angle);
    const float x0 = x[base + 2 * i];
    const float x1 = x[base + 2 * i + 1];
    y[base + 2 * i] = x0 * cs - x1 * sn;
    y[base + 2 * i + 1] = x0 * sn + x1 * cs;
}

/*
 * Bidirectional attention over every row (the depthformer's codebook pass):
 *   out[r][h][d] = softmax_j(q[r][h] . keys[j][kv(h)] / sqrt(d_head)) * values[j][kv(h)]
 * q[n_pos][n_head*d_head], keys and values [n_pos][n_head_kv*d_head], out of q's shape.
 * One work-group per (row, head) (nf_tts_attend), group index r * n_head + h: global = {n_pos * n_head * L},
 * local = {L}, L a power of two <= LLM_DIMX.
 */
NF_KERNEL void LLM_TtsFullAttn(
    const int n_pos,
    const int n_head,
    const int n_head_kv,
    const int d_head,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const float* NF_RESTRICT keys,
    NF_GLOBAL const float* NF_RESTRICT values,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float w[NF_TTS_CHUNK];
    NF_LOCAL float red[LLM_DIMX];
    const nf_i64 g = (nf_i64)get_group_id(0);
    if (g >= (nf_i64)n_pos * n_head) return;
    const int r = (int)(g / n_head);
    const int h = (int)(g - (nf_i64)r * n_head);
    const int kv_head = h / (n_head / n_head_kv);
    const nf_i64 q_off = (nf_i64)r * n_head * d_head + (nf_i64)h * d_head;
    nf_tts_attend(q, q_off, keys, values, n_head_kv * d_head, kv_head * d_head, d_head, 0, n_pos - 1, out, q_off,
                  w, red);
}

/*
 * Per-head sigmoid gate over a block of rows: y[r][h][d] = x[r][h][d] * sigmoid(gate[r][h]); x and y must not
 * overlap.
 * x and y [n_rows][n_head*d_head], gate [n_rows][n_head].
 */
NF_KERNEL void LLM_HeadSigmoidGate(
    const int n_rows,
    const int n_head,
    const int d_head,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT gate,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int idx = (int)get_global_id(0);
    if (idx >= n_rows * n_head) return;
    const int r = idx / n_head;
    const int h = idx - r * n_head;
    const float g = 1.0f / (1.0f + exp(-gate[idx]));
    const nf_i64 base = (nf_i64)r * n_head * d_head + (nf_i64)h * d_head;
    for (int d = 0; d < d_head; d++) {
        y[base + d] = x[base + d] * g;
    }
}

/*
 * Per-channel scaled row add: y[r][c] = a[r][c] + b[r][c] * scale[scale_off + c]. y must not overlap a or b.
 */
NF_KERNEL void LLM_RowScaleAdd(
    const int rows,
    const int cols,
    const int scale_off,
    NF_GLOBAL const float* NF_RESTRICT a,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL const float* NF_RESTRICT scale,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)rows * cols) return;
    const int c = (int)(idx % cols);
    y[idx] = a[idx] + b[idx] * scale[scale_off + c];
}

/* Per-channel affine: y[r][c] = x[r][c] * scale[c] + bias[c]. y must not overlap x. */
NF_KERNEL void LLM_RowAffine(
    const int rows,
    const int cols,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT scale,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)rows * cols) return;
    const int c = (int)(idx % cols);
    y[idx] = x[idx] * scale[c] + bias[c];
}

/*
 * Camera-style AdaLN modulation of a normalized row against a projection of the conditioning vector:
 *   y[r][c] = x[r][c] * (1 + mod[r][2*cols + c]) + mod[r][c]
 * x and y [rows][cols], mod [rows][2*cols]. y must not overlap x.
 */
NF_KERNEL void LLM_AdaInModulate(
    const int rows,
    const int cols,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT mod,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= (nf_i64)rows * cols) return;
    const int r = (int)(idx / cols);
    const int c = (int)(idx - (nf_i64)r * cols);
    const nf_i64 mod_row = (nf_i64)r * 2 * cols;
    y[idx] = x[idx] * (1.0f + mod[mod_row + cols + c]) + mod[mod_row + c];
}

/*
 * The flow network's time-embedder normalization ({@code _rms_norm} of llama.cpp's pockettts mlp): the input (not
 * the centered value) over the square root of the unbiased variance, rescaled by the learned alpha:
 *   mean = sum(x)/n;  var = sum((x-mean)^2)/n;  denom = sqrt(var*n/(n-1) + eps);  y[i] = x[i]/denom * w[i].
 * One row of n values; requires n >= 2.
 */
NF_KERNEL void LLM_RmsNormUnbiased(
    const int n,
    const float eps,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int i = (int)get_global_id(0);
    if (i >= n) return;
    float mean = 0.0f;
    for (int d = 0; d < n; d++) {
        mean += x[d];
    }
    mean /= (float)n;
    float var = 0.0f;
    for (int d = 0; d < n; d++) {
        const float diff = x[d] - mean;
        var += diff * diff;
    }
    var /= (float)n;
    const float denom = sqrt(var * (float)n / (float)(n - 1) + eps);
    y[i] = x[i] / denom * w[i];
}

/* Nearest-neighbour row repeat: y[(r*repeat + s)][c] = x[r][c]. y may not overlap x. */
NF_KERNEL void LLM_RepeatRows(
    const int src_rows,
    const int repeat,
    const int cols,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    const nf_i64 total = (nf_i64)src_rows * repeat * cols;
    if (idx >= total) return;
    const int c = (int)(idx % cols);
    const nf_i64 row = idx / cols;
    const int r = (int)(row / repeat);
    y[idx] = x[(nf_i64)r * cols + c];
}

/* The depthwise transposed convolution's full signal at output row `i`, channel `c`. */
NF_INLINE float nf_speech_dw_transpose_at(const int i, const int c, const int n, const int c_n, const int k,
                                          const int stride, NF_GLOBAL const float* x,
                                          NF_GLOBAL const float* w)
{
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        if (i < j || (i - j) % stride != 0) {
            continue;
        }
        const int t = (i - j) / stride;
        if (t >= n) {
            continue;
        }
        acc += w[c * k + j] * x[(nf_i64)t * c_n + c];
    }
    return acc;
}

/*
 * Causal depthwise ConvTranspose1d: the emitted rows are the full transposed convolution minus its overlap tail,
 * which is added from the previous call's state and replaced by the new tail:
 *   trim = K - stride, emit = n * stride
 *   y[i][c] = b[c] + full[i][c] + (i < trim ? state[i][c] : 0)
 *   nstate[i][c] = full[emit + i][c]                                      for i < trim
 * x[n][C], w[C][K] (ggml {K, 1, C}), b[C], state[trim][C], y[emit][C], nstate[trim][C].
 */
NF_KERNEL void LLM_ConvTranspose1dDepthwiseCausal(
    const int n,
    const int c_n,
    const int k,
    const int stride,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL const float* NF_RESTRICT state,
    NF_GLOBAL float* NF_RESTRICT y,
    NF_GLOBAL float* NF_RESTRICT nstate)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    const int emit = n * stride;
    const int trim = k - stride;
    const nf_i64 out_n = (nf_i64)emit * c_n;
    if (idx < out_n) {
        const int i = (int)(idx / c_n);
        const int c = (int)(idx - (nf_i64)i * c_n);
        float acc = b[c] + nf_speech_dw_transpose_at(i, c, n, c_n, k, stride, x, w);
        if (i < trim) {
            acc += state[(nf_i64)i * c_n + c];
        }
        y[idx] = acc;
    }
    const nf_i64 state_n = (nf_i64)trim * c_n;
    if (idx < state_n) {
        const int i = (int)(idx / c_n);
        const int c = (int)(idx - (nf_i64)i * c_n);
        nstate[idx] = nf_speech_dw_transpose_at(emit + i, c, n, c_n, k, stride, x, w);
    }
}
