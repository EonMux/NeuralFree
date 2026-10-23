/* recurrent/rwkv.cl - RWKV6/RWKV7 WKV, GLA, token-shift and decay kernels. Requires: common/dialect.h, common/config.h, common/reduce.h, common/math.h */
/*
 * Layouts (all F32):
 *   activations  token-major rows of C = H * S channels: x[t * C + h * S + d]
 *   wkv6 / gla   state s[row * H*S*S + h*S*S + i*S + j], i = key, j = value   (ggml-cpu/ops.cpp:10732, 10944)
 *   wkv7         state s[row * H*S*S + h*S*S + i*S + j], i = value, j = key   (ggml-cpu/ops.cpp:11827)
 *   token shift  shift_state[shift_off + c]; att shift at 0, ffn shift at n_embd (llama-graph.cpp:3971)
 * A single sequence is processed per launch: its state row is state_rows[0].
 *
 * WKV / GLA work split. Head size S <= NF_RWKV_MAX_S. Each work-item owns one value index of one head
 * and the state entries of the NF_RWKV_KPT consecutive keys [part * NF_RWKV_KPT, (part + 1) *
 * NF_RWKV_KPT) of that value, held in registers for the whole token loop. P = nf_rwkv_parts(S)
 * work-items share one value (adjacent lanes) and combine their partial dot products with
 * nf_seg_sum. Key-indexed per-token vectors are staged in local memory, zero-padded from S up to
 * P * NF_RWKV_KPT, and register entries of keys >= S stay zero, so the inner loops run unguarded and
 * fully unrolled: every padded term is an exact zero.
 *   local  L = min(256, max(32, P * pow2ceil(S))); values per group VPG = L / P
 *   global H * ceil(S / VPG) * L   (gguf.ops.rwkv.RwkvKernels.wkvLocalSize / wkvGlobalSize)
 * Early returns depend on the group id or on uniform scalars only; inactive lanes (value >= S) carry
 * zeros through every barrier and reduction.
 */

#define NF_RWKV_MAX_S 256   /* largest supported head size; equals 8 * NF_RWKV_KPT */
#define NF_RWKV_KPT   32    /* state entries (keys) held per work-item */

/* Work-items sharing one value index: ceil(S / NF_RWKV_KPT) rounded up to a power of two (1..8). */
NF_INLINE int nf_rwkv_parts(int S) {
    return S <= NF_RWKV_KPT ? 1 : (S <= 2 * NF_RWKV_KPT ? 2 : (S <= 4 * NF_RWKV_KPT ? 4 : 8));
}

/* Sum of v over the P adjacent lanes that share one value; identity when P == 1. P is uniform. */
NF_INLINE float nf_rwkv_part_sum(NF_LOCAL_PTR float* red, float v, int P) {
    return P > 1 ? nf_seg_sum(red, v, P) : v;
}

/* Index setup shared by the WKV / GLA kernels: S, P, lid, L, h, val (this lane's value index), key0
 * (first key of this lane), live (val < S), span (padded key count) and the uniform early exit. */
#define NF_RWKV_WKV_SETUP()                                         \
    const int S = C / H;                                            \
    const int P = nf_rwkv_parts(S);                                 \
    const int lid = get_local_id(0);                                \
    const int L = get_local_size(0);                                \
    const int vpg = L / P;                                          \
    const int gph = (S + vpg - 1) / vpg;                            \
    const int h = get_group_id(0) / gph;                            \
    if (S > NF_RWKV_MAX_S || h >= H) {                              \
        return;                                                     \
    }                                                               \
    const int val = (get_group_id(0) % gph) * vpg + lid / P;        \
    const int key0 = (lid & (P - 1)) * NF_RWKV_KPT;                 \
    const int live = val < S;                                       \
    const int span = P * NF_RWKV_KPT;

/*
 * RWKV6 WKV (ggml_rwkv_wkv6, ggml-cpu/ops.cpp:10732-10930):
 *   kv      = v[t,h,j] * k[t,h,i]
 *   y[t,j]  = sum_i (kv * tf[h,i] + s[i][j]) * r[t,h,i]
 *   s[i][j] = s[i][j] * td[t,h,i] + kv
 * tf is time_mix_first [H][S]; td is the per-token decay [t][C].
 */
NF_KERNEL void LLM_RWKV_WKV6(const int n_tokens, const int C, const int H,
                             NF_GLOBAL const float* NF_RESTRICT k,
                             NF_GLOBAL const float* NF_RESTRICT v,
                             NF_GLOBAL const float* NF_RESTRICT r,
                             NF_GLOBAL const float* NF_RESTRICT tf,
                             NF_GLOBAL const float* NF_RESTRICT td,
                             NF_GLOBAL float* state,
                             NF_GLOBAL const int* NF_RESTRICT state_rows,
                             NF_GLOBAL float* NF_RESTRICT y) {
    NF_LOCAL float sk[NF_RWKV_MAX_S];
    NF_LOCAL float sr[NF_RWKV_MAX_S];
    NF_LOCAL float stf[NF_RWKV_MAX_S];
    NF_LOCAL float std_[NF_RWKV_MAX_S];
    NF_LOCAL float red[LLM_DIMX];

    NF_RWKV_WKV_SETUP()
    const nf_i64 sbase = (nf_i64)state_rows[0] * C * S + (nf_i64)h * S * S;

    float st[NF_RWKV_KPT];
#pragma unroll
    for (int u = 0; u < NF_RWKV_KPT; u++) {
        const int i = key0 + u;
        st[u] = (live && i < S) ? state[sbase + (nf_i64)i * S + val] : 0.0f;
    }
    for (int i = lid; i < span; i += L) {
        stf[i] = i < S ? tf[h * S + i] : 0.0f;
    }

    for (int t = 0; t < n_tokens; t++) {
        const nf_i64 row = (nf_i64)t * C + h * S;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < span; i += L) {
            const int in = i < S;
            sk[i] = in ? k[row + i] : 0.0f;
            sr[i] = in ? r[row + i] : 0.0f;
            std_[i] = in ? td[row + i] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const float vv = live ? v[row + val] : 0.0f;
        float acc = 0.0f;
#pragma unroll
        for (int u = 0; u < NF_RWKV_KPT; u++) {
            const int i = key0 + u;
            const float kv = vv * sk[i];
            acc += (kv * stf[i] + st[u]) * sr[i];
            st[u] = st[u] * std_[i] + kv;
        }
        acc = nf_rwkv_part_sum(red, acc, P);
        if (live && key0 == 0) {
            y[row + val] = acc;
        }
    }

#pragma unroll
    for (int u = 0; u < NF_RWKV_KPT; u++) {
        const int i = key0 + u;
        if (live && i < S) {
            state[sbase + (nf_i64)i * S + val] = st[u];
        }
    }
}

/*
 * Gated linear attention (ggml_gated_linear_attn, ggml-cpu/ops.cpp:10944-11130); the state is updated
 * before the readout:
 *   s[i][j] = s[i][j] * g + v[t,h,j] * k[t,h,i]
 *   y[t,j]  = sum_i (q[t,h,i] * scale) * s[i][j]
 * g_mode 0: g[t * C + h * S + i] (per key channel, qrwkv); g_mode 1: g[t * H + h] (per-head scalar).
 */
NF_KERNEL void LLM_GLA(const int n_tokens, const int C, const int H, const int g_mode, const float scale,
                       NF_GLOBAL const float* NF_RESTRICT k,
                       NF_GLOBAL const float* NF_RESTRICT v,
                       NF_GLOBAL const float* NF_RESTRICT q,
                       NF_GLOBAL const float* NF_RESTRICT g,
                       NF_GLOBAL float* state,
                       NF_GLOBAL const int* NF_RESTRICT state_rows,
                       NF_GLOBAL float* NF_RESTRICT y) {
    NF_LOCAL float sk[NF_RWKV_MAX_S];
    NF_LOCAL float sq[NF_RWKV_MAX_S];
    NF_LOCAL float sg[NF_RWKV_MAX_S];
    NF_LOCAL float red[LLM_DIMX];

    NF_RWKV_WKV_SETUP()
    const nf_i64 sbase = (nf_i64)state_rows[0] * C * S + (nf_i64)h * S * S;

    float st[NF_RWKV_KPT];
#pragma unroll
    for (int u = 0; u < NF_RWKV_KPT; u++) {
        const int i = key0 + u;
        st[u] = (live && i < S) ? state[sbase + (nf_i64)i * S + val] : 0.0f;
    }

    for (int t = 0; t < n_tokens; t++) {
        const nf_i64 row = (nf_i64)t * C + h * S;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < span; i += L) {
            const int in = i < S;
            sk[i] = in ? k[row + i] : 0.0f;
            sq[i] = in ? q[row + i] * scale : 0.0f;
            sg[i] = in ? (g_mode == 0 ? g[row + i] : g[(nf_i64)t * H + h]) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const float vv = live ? v[row + val] : 0.0f;
        float acc = 0.0f;
#pragma unroll
        for (int u = 0; u < NF_RWKV_KPT; u++) {
            const int i = key0 + u;
            st[u] = st[u] * sg[i] + vv * sk[i];
            acc += st[u] * sq[i];
        }
        acc = nf_rwkv_part_sum(red, acc, P);
        if (live && key0 == 0) {
            y[row + val] = acc;
        }
    }

#pragma unroll
    for (int u = 0; u < NF_RWKV_KPT; u++) {
        const int i = key0 + u;
        if (live && i < S) {
            state[sbase + (nf_i64)i * S + val] = st[u];
        }
    }
}

/*
 * RWKV7 WKV (ggml_rwkv_wkv7, ggml-cpu/ops.cpp:11827-12030); the lane's value index is the state row i
 * and its keys are the columns j:
 *   sa      = sum_j a[t,h,j] * s[i][j]
 *   s[i][j] = s[i][j] * w[t,h,j] + v[t,h,i] * k[t,h,j] + sa * b[t,h,j]
 *   y[t,i]  = sum_j s[i][j] * r[t,h,j]
 * The graph passes a = -kk and b = kk * a (models/rwkv7-base.cpp), produced by LLM_RWKV7_KkPrep.
 */
NF_KERNEL void LLM_RWKV_WKV7(const int n_tokens, const int C, const int H,
                             NF_GLOBAL const float* NF_RESTRICT r,
                             NF_GLOBAL const float* NF_RESTRICT w,
                             NF_GLOBAL const float* NF_RESTRICT k,
                             NF_GLOBAL const float* NF_RESTRICT v,
                             NF_GLOBAL const float* NF_RESTRICT a,
                             NF_GLOBAL const float* NF_RESTRICT b,
                             NF_GLOBAL float* state,
                             NF_GLOBAL const int* NF_RESTRICT state_rows,
                             NF_GLOBAL float* NF_RESTRICT y) {
    NF_LOCAL float sr[NF_RWKV_MAX_S];
    NF_LOCAL float sw[NF_RWKV_MAX_S];
    NF_LOCAL float sk[NF_RWKV_MAX_S];
    NF_LOCAL float sa_[NF_RWKV_MAX_S];
    NF_LOCAL float sb[NF_RWKV_MAX_S];
    NF_LOCAL float red[LLM_DIMX];

    NF_RWKV_WKV_SETUP()
    const nf_i64 sbase = (nf_i64)state_rows[0] * C * S + (nf_i64)h * S * S + (nf_i64)val * S;

    float st[NF_RWKV_KPT];
#pragma unroll
    for (int u = 0; u < NF_RWKV_KPT; u++) {
        const int j = key0 + u;
        st[u] = (live && j < S) ? state[sbase + j] : 0.0f;
    }

    for (int t = 0; t < n_tokens; t++) {
        const nf_i64 row = (nf_i64)t * C + h * S;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = lid; j < span; j += L) {
            const int in = j < S;
            sr[j] = in ? r[row + j] : 0.0f;
            sw[j] = in ? w[row + j] : 0.0f;
            sk[j] = in ? k[row + j] : 0.0f;
            sa_[j] = in ? a[row + j] : 0.0f;
            sb[j] = in ? b[row + j] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float sa = 0.0f;
#pragma unroll
        for (int u = 0; u < NF_RWKV_KPT; u++) {
            sa += sa_[key0 + u] * st[u];
        }
        sa = nf_rwkv_part_sum(red, sa, P);

        const float vv = live ? v[row + val] : 0.0f;
        float acc = 0.0f;
#pragma unroll
        for (int u = 0; u < NF_RWKV_KPT; u++) {
            const int j = key0 + u;
            st[u] = st[u] * sw[j] + vv * sk[j] + sa * sb[j];
            acc += st[u] * sr[j];
        }
        acc = nf_rwkv_part_sum(red, acc, P);
        if (live && key0 == 0) {
            y[row + val] = acc;
        }
    }

#pragma unroll
    for (int u = 0; u < NF_RWKV_KPT; u++) {
        const int j = key0 + u;
        if (live && j < S) {
            state[sbase + j] = st[u];
        }
    }
}

/* Token-shift modes of LLM_RWKV_TokenShiftLerp (gguf.ops.rwkv.RwkvKernels.SHIFT_MODE_*). */
#define NF_RWKV_SHIFT_LERP   0
#define NF_RWKV_SHIFT_LERP_X 1
#define NF_RWKV_SHIFT_LORA   2
#define NF_RWKV_SHIFT_PAIR   3

/*
 * Token shift fused with the RWKV lerps (models/rwkv6-base.cpp, models/rwkv7-base.cpp). The input rows
 * start at element cur_off of cur (cur[t][c] = cur[cur_off + t * n_embd + c]). With
 * prev(t) = t == 0 ? shift_state[shift_off + c] : cur[t-1][c] and sx = prev - cur[t][c], slot m of
 * out is the plane out[m * n_tokens * n_embd + t * n_embd + c]:
 *   LERP   (0): out[m] = sx * lerp_fused[m][c] + cur,             m < n_mix        rwkv7 time/channel mix
 *   LERP_X (1): out[0] = sx * lerp_x[c] + cur                                      rwkv6 time-mix LoRA input
 *   LORA   (2): out[m] = (out[m] + lerp_fused[m][c]) * sx + cur,  m < n_mix, in place on the LoRA outputs
 *   PAIR   (3): out[0] = sx * lerp_x[c] + cur, out[1 + m] = sx * lerp_fused[m][c] + cur, m < n_mix
 *                                                                                  rwkv6 channel mix (k, r)
 * Every mode except LERP_X also stores cur[n_tokens-1][c] as the new shift state, so LERP_X followed
 * by LORA reads the same previous token. One work-item per (t, c); the t == 0 item is the only reader
 * of shift_state[shift_off + c] and writes the new state after reading it.
 * Dispatch: 1-D, global = roundUp(n_tokens * n_embd, L)   (RwkvKernels.tokenShiftGlobalSize).
 */
NF_KERNEL void LLM_RWKV_TokenShiftLerp(const int n_tokens, const int n_embd, const int n_mix, const int mode,
                                       const int shift_off, const int cur_off,
                                       NF_GLOBAL const float* NF_RESTRICT cur,
                                       NF_GLOBAL float* shift_state,
                                       NF_GLOBAL const float* NF_RESTRICT lerp_x,
                                       NF_GLOBAL const float* NF_RESTRICT lerp_fused,
                                       NF_GLOBAL float* NF_RESTRICT out) {
    const nf_i64 plane = (nf_i64)n_tokens * n_embd;
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (idx >= plane) {
        return;
    }
    const int t = (int)(idx / n_embd);
    const int c = (int)(idx - (nf_i64)t * n_embd);
    const float x = cur[cur_off + idx];
    const float prev = t == 0 ? shift_state[shift_off + c] : cur[cur_off + idx - n_embd];
    const float sx = prev - x;
    if (mode == NF_RWKV_SHIFT_LERP_X || mode == NF_RWKV_SHIFT_PAIR) {
        out[idx] = sx * lerp_x[c] + x;
    }
    const int m0 = mode == NF_RWKV_SHIFT_PAIR ? 1 : 0;
    const int nm = mode == NF_RWKV_SHIFT_LERP_X ? 0 : n_mix;
    for (int m = 0; m < nm; m++) {
        const float lf = lerp_fused[(nf_i64)m * n_embd + c];
        const nf_i64 o = (nf_i64)(m0 + m) * plane + idx;
        out[o] = mode == NF_RWKV_SHIFT_LORA ? (out[o] + lf) * sx + x : sx * lf + x;
    }
    if (t == 0 && mode != NF_RWKV_SHIFT_LERP_X) {
        shift_state[shift_off + c] = cur[cur_off + plane - n_embd + c];
    }
}

/*
 * RWKV6 decay (models/rwkv6-base.cpp): out[i] = exp(-exp(w_in[i] + base[i % n_cols])).
 * Dispatch: 1-D, global = roundUp(n, L).
 */
NF_KERNEL void LLM_RWKV_Decay6(const int n, const int n_cols,
                               NF_GLOBAL const float* NF_RESTRICT w_in,
                               NF_GLOBAL const float* NF_RESTRICT base,
                               NF_GLOBAL float* NF_RESTRICT out) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const float x = w_in[i] + base[i % n_cols];
    out[i] = exp(-exp(x));
}

/*
 * RWKV7 decay (models/rwkv7-base.cpp): out[i] = exp(sigmoid(w_in[i] + base[i % n_cols]) * -0.606531).
 * Dispatch: 1-D, global = roundUp(n, L).
 */
NF_KERNEL void LLM_RWKV_Decay7(const int n, const int n_cols,
                               NF_GLOBAL const float* NF_RESTRICT w_in,
                               NF_GLOBAL const float* NF_RESTRICT base,
                               NF_GLOBAL float* NF_RESTRICT out) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const float x = w_in[i] + base[i % n_cols];
    out[i] = exp(nf_sigmoid(x) * -0.606531f);
}

/*
 * RWKV7 key preparation (models/rwkv7-base.cpp), per token t and head h over d < S, c = h * S + d:
 *   kk     = l2_norm(k[t,c] * k_k[c]) over the head, eps 1e-12 (ggml-cpu/ops.cpp:4570)
 *   ka     = k[t,c] * k_a[c]
 *   k_out  = k[t,c] + (a[t,c] * ka - ka)
 *   kk_out = kk, aa_out = -kk, bb_out = kk * a[t,c]
 * Dispatch: 1-D, one workgroup per (t, h): global = n_tokens * n_head * L, L a power of two in
 * [32, 256] (RwkvKernels.headRowLocalSize).
 */
NF_KERNEL void LLM_RWKV7_KkPrep(const int n_tokens, const int n_head, const int S,
                                NF_GLOBAL const float* NF_RESTRICT k,
                                NF_GLOBAL const float* NF_RESTRICT k_k,
                                NF_GLOBAL const float* NF_RESTRICT k_a,
                                NF_GLOBAL const float* NF_RESTRICT a,
                                NF_GLOBAL float* NF_RESTRICT kk_out,
                                NF_GLOBAL float* NF_RESTRICT k_out,
                                NF_GLOBAL float* NF_RESTRICT aa_out,
                                NF_GLOBAL float* NF_RESTRICT bb_out) {
    NF_LOCAL float red[LLM_DIMX];
    const int grp = get_group_id(0);
    if (grp >= n_tokens * n_head) {
        return;
    }
    const int h = grp % n_head;
    const int lid = get_local_id(0);
    const int L = get_local_size(0);
    const nf_i64 row = (nf_i64)grp * S;
    const int wrow = h * S;

    float ss = 0.0f;
    for (int d = lid; d < S; d += L) {
        const float x = k[row + d] * k_k[wrow + d];
        ss += x * x;
    }
    ss = nf_wg_sum(red, ss);
    const float inv = 1.0f / fmax(sqrt(ss), 1e-12f);

    for (int d = lid; d < S; d += L) {
        const float kv = k[row + d];
        const float av = a[row + d];
        const float kk = kv * k_k[wrow + d] * inv;
        const float ka = kv * k_a[wrow + d];
        kk_out[row + d] = kk;
        k_out[row + d] = kv + (av * ka - ka);
        aa_out[row + d] = -kk;
        bb_out[row + d] = kk * av;
    }
}

/*
 * RWKV7 r_k bonus (models/rwkv7-base.cpp), per token t and head h over d < S, c = h * S + d:
 *   rk = sum_d k[t,c] * r[t,c] * r_k[c];   x[t,c] += v[t,c] * rk
 * Dispatch as LLM_RWKV7_KkPrep.
 */
NF_KERNEL void LLM_RWKV7_Bonus(const int n_tokens, const int n_head, const int S,
                               NF_GLOBAL const float* NF_RESTRICT k,
                               NF_GLOBAL const float* NF_RESTRICT r,
                               NF_GLOBAL const float* NF_RESTRICT r_k,
                               NF_GLOBAL const float* NF_RESTRICT v,
                               NF_GLOBAL float* x) {
    NF_LOCAL float red[LLM_DIMX];
    const int grp = get_group_id(0);
    if (grp >= n_tokens * n_head) {
        return;
    }
    const int h = grp % n_head;
    const int lid = get_local_id(0);
    const int L = get_local_size(0);
    const nf_i64 row = (nf_i64)grp * S;
    const int wrow = h * S;

    float rk = 0.0f;
    for (int d = lid; d < S; d += L) {
        rk += k[row + d] * r[row + d] * r_k[wrow + d];
    }
    rk = nf_wg_sum(red, rk);

    for (int d = lid; d < S; d += L) {
        x[row + d] += v[row + d] * rk;
    }
}
