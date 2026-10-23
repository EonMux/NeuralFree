/* hc.cl - LLM_HC_PrePostComb, LLM_HC_Pre, LLM_HC_Post, LLM_ResStack_Score and LLM_ResStack_Mix hyper-connection and attention-residual kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h */
/*
 * Hyper-connections (deepseek4, glm5-next, dflash: mHC with Sinkhorn mixing; hy_v4: iHC without mixing;
 * qwen4exp: gated low-rank variant) replace the residual add of each sub-block by hc parallel streams.
 * Layouts (element offsets relative to the buffer's <name>_off argument, ggml order with n_embd innermost):
 *   stream X[(t*hc + h)*n_embd + i]       ggml [n_embd, hc, n_tokens]
 *   pre[t*hc + h], post[t*hc + h]         ggml [hc, n_tokens]
 *   comb[t*hc*hc + dst + hc*src]          ggml [dst_hc, src_hc, n_tokens] (dst innermost)
 * References: ggml-cpu/ops.cpp:11429-11733 (dsv4_hc_comb / hc_pre / hc_post), src/models/deepseek4.cpp:279-466,
 * src/models/hy-v4.cpp:160-270, src/models/qwen4exp.cpp:311-395.
 *
 * Attention residuals (kimi-k3, src/models/kimi-k3.cpp:188-235) mix a per-step checkpoint stack
 *   stack[(c*n_tokens + t)*n_embd + i]    checkpoint c of token t (raw layer inputs, append-only)
 * with the current stream through softmax(score(rms(stack_c)), score(rms(cur))), score = sum(norm (.) w).
 */

/* Largest stream count of every hc kernel: comb needs NF_HC_MAX^2 registers per work-item. */
#define NF_HC_MAX 8

/* LLM_HC_PrePostComb n_out codes. */
#define NF_HC_OUT_PRE           1
#define NF_HC_OUT_PRE_POST      2
#define NF_HC_OUT_PRE_POST_COMB 3

/* LLM_HC_Pre mode codes. */
#define NF_HC_PRE_WEIGHTS 0
#define NF_HC_PRE_GATED   1
#define NF_HC_PRE_MEAN    2

/* Sinkhorn column normalization: comb[dst + hc*src] /= (eps + sum_src comb[dst + hc*src]) (ops.cpp:11429-11443). */
NF_INLINE void nf_hc_norm_cols(float* comb, int hc, float eps) {
    for (int dst = 0; dst < hc; dst++) {
        float sum = eps;
        for (int src = 0; src < hc; src++) {
            sum += comb[dst + hc * src];
        }
        const float inv_sum = 1.0f / sum;
        for (int src = 0; src < hc; src++) {
            comb[dst + hc * src] *= inv_sum;
        }
    }
}

/* Sinkhorn row normalization: comb[dst + hc*src] /= (eps + sum_dst comb[dst + hc*src]) (ops.cpp:11445-11459). */
NF_INLINE void nf_hc_norm_rows(float* comb, int hc, float eps) {
    for (int src = 0; src < hc; src++) {
        float sum = eps;
        for (int dst = 0; dst < hc; dst++) {
            sum += comb[dst + hc * src];
        }
        const float inv_sum = 1.0f / sum;
        for (int dst = 0; dst < hc; dst++) {
            comb[dst + hc * src] *= inv_sum;
        }
    }
}

/*
 * LLM_HC_PrePostComb: per-token hyper-connection coefficients from the projected mixes
 * m = mixes[t*n_mix ...], n_mix = (2 + hc)*hc for n_out 3, n_out*hc otherwise:
 *   pre[h]  = sigmoid(m[h]*scale[0] + base[h]) + eps                                   (n_out >= 1)
 *   post[h] = post_mag*sigmoid(m[hc + h]*scale[1] + base[hc + h]) + post_eps           (n_out >= 2)
 *   comb    = Sinkhorn(m[2hc + dst + hc*src]*scale[2] + base[2hc + dst + hc*src])     (n_out == 3):
 *             softmax over dst for each src, + eps, norm_cols, then (iters - 1) x (norm_rows, norm_cols)
 *             (ops.cpp:11461-11546).
 * mHC (deepseek4.cpp:353-409): n_out 3, post_mag 2, post_eps 0. iHC (hy-v4.cpp:175-215): n_out 2,
 * post_mag = hc_magnitude, post_eps = eps. HC head (deepseek4.cpp:448-466, hy-v4.cpp:248-270): n_out 1.
 * Outputs beyond n_out are not written. iters >= 1 when n_out == 3. Every activation buffer is addressed from its
 * <name>_off element offset (mixes, pre, post, comb).
 *
 * Dispatch: 1-D, one work-item per token; global = roundUp(n_tokens, L), any L. Requires 1 <= hc <= NF_HC_MAX;
 * other shapes write nothing (the host rejects them first).
 */
NF_KERNEL void LLM_HC_PrePostComb(
    const int n_tokens,
    const int hc,
    const float eps,
    const int iters,
    const int n_out,
    const float post_mag,
    const float post_eps,
    const int mixes_off,
    const int pre_off,
    const int post_off,
    const int comb_off,
    NF_GLOBAL const float* NF_RESTRICT mixes,
    NF_GLOBAL const float* NF_RESTRICT scale,
    NF_GLOBAL const float* NF_RESTRICT base,
    NF_GLOBAL float* NF_RESTRICT pre,
    NF_GLOBAL float* NF_RESTRICT post,
    NF_GLOBAL float* NF_RESTRICT comb)
{
    const int t = (int)get_global_id(0);
    if (t >= n_tokens || hc < 1 || hc > NF_HC_MAX || n_out < NF_HC_OUT_PRE || n_out > NF_HC_OUT_PRE_POST_COMB) {
        return;
    }
    const int n_mix = n_out == NF_HC_OUT_PRE_POST_COMB ? (2 + hc) * hc : n_out * hc;
    NF_GLOBAL const float* m = mixes + (nf_i64)mixes_off + (nf_i64)t * n_mix;

    const float s_pre = scale[0];
    for (int h = 0; h < hc; h++) {
        pre[(nf_i64)pre_off + (nf_i64)t * hc + h] = nf_sigmoid(m[h] * s_pre + base[h]) + eps;
    }
    if (n_out < NF_HC_OUT_PRE_POST) {
        return;
    }
    const float s_post = scale[1];
    for (int h = 0; h < hc; h++) {
        post[(nf_i64)post_off + (nf_i64)t * hc + h] = post_mag * nf_sigmoid(m[hc + h] * s_post + base[hc + h]) + post_eps;
    }
    if (n_out < NF_HC_OUT_PRE_POST_COMB) {
        return;
    }

    const float s_comb = scale[2];
    const int off = 2 * hc;
    float c[NF_HC_MAX * NF_HC_MAX];
    for (int src = 0; src < hc; src++) {
        float mx = -NF_INF;
        for (int dst = 0; dst < hc; dst++) {
            const int idx = dst + hc * src;
            const float v = m[off + idx] * s_comb + base[off + idx];
            c[idx] = v;
            mx = fmax(mx, v);
        }
        float sum = 0.0f;
        for (int dst = 0; dst < hc; dst++) {
            const int idx = dst + hc * src;
            const float v = exp(c[idx] - mx);
            c[idx] = v;
            sum += v;
        }
        const float inv_sum = 1.0f / sum;
        for (int dst = 0; dst < hc; dst++) {
            const int idx = dst + hc * src;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }
    nf_hc_norm_cols(c, hc, eps);
    for (int i = 1; i < iters; i++) {
        nf_hc_norm_rows(c, hc, eps);
        nf_hc_norm_cols(c, hc, eps);
    }
    NF_GLOBAL float* out = comb + (nf_i64)comb_off + (nf_i64)t * hc * hc;
    for (int idx = 0; idx < hc * hc; idx++) {
        out[idx] = c[idx];
    }
}

/*
 * LLM_HC_Pre: collapse the hc streams of each token into one n_embd row (ops.cpp:11567-11626):
 *   out[t*n_embd + i] = gscale * sum_h X[(t*hc + h)*n_embd + i] * w
 *   mode 0 (weights): w = w_or_gate[t*hc + h]                      (ggml_dsv4_hc_pre, gscale 1)
 *   mode 1 (gated):   w = sigmoid(w_or_gate[(t*hc + h)*n_embd + i]) (ggml_dsv4_hc_pre_gated, qwen4exp gscale 1/hc)
 *   mode 2 (mean):    w = 1, w_or_gate unused                       (build_hc_mean, gscale 1/hc)
 * The sum runs over h in increasing order, as llama.cpp accumulates it. X, w_or_gate and out are addressed from
 * x_off, w_off and out_off; out must not overlap X.
 *
 * Dispatch: 1-D, one work-item per (token, i); global = roundUp(n_tokens*n_embd, L), any L.
 */
NF_KERNEL void LLM_HC_Pre(
    const int n_tokens,
    const int n_embd,
    const int hc,
    const int mode,
    const float gscale,
    const int x_off,
    const int w_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT X,
    NF_GLOBAL const float* NF_RESTRICT w_or_gate,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)n_tokens * n_embd) {
        return;
    }
    const nf_i64 t = gid / n_embd;
    const nf_i64 i = gid - t * n_embd;
    const nf_i64 row0 = t * hc;
    float sum = 0.0f;
    for (int h = 0; h < hc; h++) {
        const nf_i64 xi = (row0 + h) * n_embd + i;
        float wv;
        if (mode == NF_HC_PRE_WEIGHTS) {
            wv = w_or_gate[(nf_i64)w_off + row0 + h];
        } else if (mode == NF_HC_PRE_GATED) {
            wv = nf_sigmoid(w_or_gate[(nf_i64)w_off + xi]);
        } else if (mode == NF_HC_PRE_MEAN) {
            wv = 1.0f;
        } else {
            wv = NF_NAN;
        }
        sum += X[(nf_i64)x_off + xi] * wv;
    }
    out[(nf_i64)out_off + gid] = gscale * sum;
}

/*
 * LLM_HC_Post: expand the sub-block output y back into hc streams (ops.cpp:11647-11720):
 *   Xout[(t*hc + dst)*n_embd + i] = y[t*n_embd + i] * p[dst]
 *                                 + (has_comb ? sum_src X[(t*hc + src)*n_embd + i] * comb[t*hc*hc + dst + hc*src]
 *                                             : X[(t*hc + dst)*n_embd + i])
 *   p[dst] = post_act ? post_mag * sigmoid(post_in_scale * post[t*hc + dst]) : post[t*hc + dst]
 * post_act 1 is the qwen4exp combine weight 2*sigmoid(inject/hc) (qwen4exp.cpp:369-395); mHC/iHC pass the
 * coefficients of LLM_HC_PrePostComb with post_act 0. comb is unused when has_comb is 0.
 * y, X, post, comb and Xout are addressed from y_off, x_off, post_off, comb_off and xout_off. Each work-item reads
 * all hc stream values of its (t, i) before writing any, so Xout may be X with xout_off == x_off.
 *
 * Dispatch: 1-D, one work-item per (token, i); global = roundUp(n_tokens*n_embd, L), any L.
 * Requires 1 <= hc <= NF_HC_MAX; other shapes write nothing (the host rejects them first).
 */
NF_KERNEL void LLM_HC_Post(
    const int n_tokens,
    const int n_embd,
    const int hc,
    const int has_comb,
    const int post_act,
    const float post_in_scale,
    const float post_mag,
    const int y_off,
    const int x_off,
    const int post_off,
    const int comb_off,
    const int xout_off,
    NF_GLOBAL const float* NF_RESTRICT y,
    NF_GLOBAL const float* X,
    NF_GLOBAL const float* NF_RESTRICT post,
    NF_GLOBAL const float* NF_RESTRICT comb,
    NF_GLOBAL float* Xout)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)n_tokens * n_embd || hc < 1 || hc > NF_HC_MAX) {
        return;
    }
    const nf_i64 t = gid / n_embd;
    const nf_i64 i = gid - t * n_embd;
    const nf_i64 row0 = t * hc;
    float xs[NF_HC_MAX];
    for (int h = 0; h < hc; h++) {
        xs[h] = X[(nf_i64)x_off + (row0 + h) * n_embd + i];
    }
    const float yv = y[(nf_i64)y_off + gid];
    const nf_i64 cb = (nf_i64)comb_off + t * hc * hc;
    for (int dst = 0; dst < hc; dst++) {
        const float pr = post[(nf_i64)post_off + row0 + dst];
        const float pv = post_act ? post_mag * nf_sigmoid(post_in_scale * pr) : pr;
        float sum = yv * pv;
        if (has_comb) {
            for (int src = 0; src < hc; src++) {
                sum += xs[src] * comb[cb + dst + hc * src];
            }
        } else {
            sum += xs[dst];
        }
        Xout[(nf_i64)xout_off + (row0 + dst) * n_embd + i] = sum;
    }
}

/*
 * LLM_ResStack_Score: attention-residual mixing probabilities (kimi-k3.cpp:198-223). For checkpoint
 * c < n_ckpt the row is stack[(c*n_tokens + t)*n_embd ...], for c == n_ckpt it is cur[t*n_embd ...]:
 *   score_c = sum_i x_i * w_i / sqrt(sum_i x_i^2 / n_embd + eps)      (rms_norm, mul score_w, sum_rows)
 *   probs[t*(n_ckpt + 1) + c] = softmax_c(score)                      (ggml_soft_max over n_ckpt + 1)
 * score_w is the n_embd {attn,ffn,output}_res_score weight. n_ckpt may be 0 (probs = 1). stack, cur and probs are
 * addressed from stack_off, cur_off and probs_off.
 *
 * Dispatch: 1-D, one workgroup per token; local = {L} with L a power of two <= LLM_DIMX (a multiple of 32 on
 * CUDA and for the subgroup path); global = n_tokens * L.
 */
NF_KERNEL void LLM_ResStack_Score(
    const int n_tokens,
    const int n_embd,
    const int n_ckpt,
    const float eps,
    const int stack_off,
    const int cur_off,
    const int probs_off,
    NF_GLOBAL const float* NF_RESTRICT stack,
    NF_GLOBAL const float* NF_RESTRICT cur,
    NF_GLOBAL const float* NF_RESTRICT score_w,
    NF_GLOBAL float* NF_RESTRICT probs)
{
    NF_LOCAL float red[LLM_DIMX];
    const int t = get_group_id(0);
    if (t >= n_tokens) {
        return; /* whole workgroup: t is uniform */
    }
    const int lid = get_local_id(0);
    const int wg = get_local_size(0);
    NF_GLOBAL float* p = probs + (nf_i64)probs_off + (nf_i64)t * (n_ckpt + 1);

    for (int c = 0; c <= n_ckpt; c++) {
        NF_GLOBAL const float* x = c < n_ckpt ? stack + (nf_i64)stack_off + ((nf_i64)c * n_tokens + t) * n_embd
                                              : cur + (nf_i64)cur_off + (nf_i64)t * n_embd;
        float ss = 0.0f;
        float dw = 0.0f;
        for (int i = lid; i < n_embd; i += wg) {
            const float v = x[i];
            ss += v * v;
            dw += v * score_w[i];
        }
        ss = nf_wg_sum(red, ss);
        dw = nf_wg_sum(red, dw);
        if (lid == 0) {
            p[c] = dw * (1.0f / sqrt(ss / (float)n_embd + eps));
        }
    }

    /* Work-item 0 wrote every score above, so it alone reads them back for the softmax. */
    if (lid == 0) {
        float mx = -NF_INF;
        for (int c = 0; c <= n_ckpt; c++) {
            mx = fmax(mx, p[c]);
        }
        float sum = 0.0f;
        for (int c = 0; c <= n_ckpt; c++) {
            const float e = exp(p[c] - mx);
            p[c] = e;
            sum += e;
        }
        const float inv_sum = 1.0f / sum;
        for (int c = 0; c <= n_ckpt; c++) {
            p[c] *= inv_sum;
        }
    }
}

/*
 * LLM_ResStack_Mix: attention-residual output (kimi-k3.cpp:225-232, raw values, not the normalized ones):
 *   out[t*n_embd + i] = sum_c stack[(c*n_tokens + t)*n_embd + i] * probs[t*(n_ckpt+1) + c]
 *                     + cur[t*n_embd + i] * probs[t*(n_ckpt+1) + n_ckpt]
 * The checkpoint sum runs over c in increasing order before the current-stream term is added, as llama.cpp
 * (ggml_dsv4_hc_pre, then ggml_add). stack, cur, probs and out are addressed from stack_off, cur_off, probs_off and
 * out_off; out may be cur with out_off == cur_off.
 *
 * Dispatch: 1-D, one work-item per (token, i); global = roundUp(n_tokens*n_embd, L), any L.
 */
NF_KERNEL void LLM_ResStack_Mix(
    const int n_tokens,
    const int n_embd,
    const int n_ckpt,
    const int stack_off,
    const int cur_off,
    const int probs_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT stack,
    NF_GLOBAL const float* cur,
    NF_GLOBAL const float* NF_RESTRICT probs,
    NF_GLOBAL float* out)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)n_tokens * n_embd) {
        return;
    }
    const nf_i64 t = gid / n_embd;
    const nf_i64 i = gid - t * n_embd;
    NF_GLOBAL const float* p = probs + (nf_i64)probs_off + t * (n_ckpt + 1);
    float sum = 0.0f;
    for (int c = 0; c < n_ckpt; c++) {
        sum += stack[(nf_i64)stack_off + ((nf_i64)c * n_tokens + t) * n_embd + i] * p[c];
    }
    out[(nf_i64)out_off + gid] = sum + cur[(nf_i64)cur_off + gid] * p[n_ckpt];
}
