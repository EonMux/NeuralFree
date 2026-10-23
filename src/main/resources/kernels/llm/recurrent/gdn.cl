/* recurrent/gdn.cl - LLM_GDN_Scan and LLM_GDN_GatePrep gated DeltaNet / KDA recurrences. Requires: common/dialect.h, common/config.h, common/reduce.h, common/math.h */
/*
 * Gated delta rule (llama.cpp ggml-cpu/ops.cpp:11214-11360, ggml_compute_forward_gated_delta_net):
 * per value head h and token t, with S[i][col] the S_k x S_v state (i = key, col = value),
 *     S[i][col] *= exp(g)              g scalar per head (GDN) or g[i] per key dim (KDA)
 *     kv_col     = sum_i S[i][col] * k[i]
 *     delta_col  = (v[col] - kv_col) * beta
 *     S[i][col] += k[i] * delta_col
 *     o[col]     = scale * sum_i S[i][col] * q[i]
 * The state is stored transposed as in ggml (s[col * S_k + i]), so one value column is contiguous.
 * The decay gate g arrives in log space from LLM_GDN_GatePrep.
 */

/* Largest key head dim of LLM_GDN_Scan: 32 lanes x NF_GDN_ROWS_PER_LANE rows per value column. */
#define NF_GDN_MAX_SK 256
#define NF_GDN_LANES 32
#define NF_GDN_ROWS_PER_LANE (NF_GDN_MAX_SK / NF_GDN_LANES)
/* Value columns per workgroup at the largest local size (LLM_DIMX / NF_GDN_LANES). */
#define NF_GDN_MAX_COLS (LLM_DIMX / NF_GDN_LANES)

/*
 * LLM_GDN_Scan: gated delta rule over n_tokens consecutive tokens of one sequence (decode and prefill).
 *
 * Layouts (element offsets; every *_off and *_stride is in floats):
 *   q[q_off + t*q_stride + kh*S_k + i], k[k_off + t*k_stride + kh*S_k + i]   (key head kh of value head h)
 *   v[v_off + t*v_stride + h*S_v + col], out[o_off + t*o_stride + h*S_v + col]
 *   g[(t*n_v_heads + h)] (gate_mode 0) or g[(t*n_v_heads + h)*S_k + i] (gate_mode 1), beta[t*n_v_heads + h]
 *   state[state_rows[0]*n_v_heads*S_v*S_k + h*S_v*S_k + col*S_k + i]
 * kv_head_mode (GdnHeadMap code): 0 tiled kh = h % n_k_heads (ggml broadcast, qwen35),
 *                                 1 grouped kh = h / (n_v_heads / n_k_heads) (qwen3next repeat-interleave).
 * scale is 1/sqrt(S_k) for every llama.cpp delta-net model (delta-net-base.cpp:45); q is not prescaled.
 *
 * out may alias v when both use the same offset and stride: each (t, h, col) element of v is staged in
 * local memory by the work-item group that later writes the same element of out, behind a barrier.
 *
 * Dispatch: 1-D, local = {L} with L a multiple of 32 and L <= LLM_DIMX; cols = L/32 value columns per
 * workgroup, 32 lanes per column; lane j holds key rows j, j+32, ... of its column in registers.
 * global = n_v_heads * ceil(S_v / cols) * L. Requires S_k <= NF_GDN_MAX_SK; larger heads leave out and
 * state untouched (the host rejects them first).
 */
NF_KERNEL void LLM_GDN_Scan(
    const int S_k,
    const int S_v,
    const int n_v_heads,
    const int n_k_heads,
    const int n_tokens,
    const int kv_head_mode,
    const int gate_mode,
    const int q_stride,
    const int q_off,
    const int k_stride,
    const int k_off,
    const int v_stride,
    const int v_off,
    const int o_stride,
    const int o_off,
    const float scale,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const float* NF_RESTRICT k,
    NF_GLOBAL const float* v,
    NF_GLOBAL const float* NF_RESTRICT g,
    NF_GLOBAL const float* NF_RESTRICT beta,
    NF_GLOBAL float* state,
    NF_GLOBAL const int* NF_RESTRICT state_rows,
    NF_GLOBAL float* out)
{
    NF_LOCAL float q_s[NF_GDN_MAX_SK];
    NF_LOCAL float k_s[NF_GDN_MAX_SK];
    NF_LOCAL float e_s[NF_GDN_MAX_SK];
    NF_LOCAL float v_s[NF_GDN_MAX_COLS];
    NF_LOCAL float red[LLM_DIMX];

    if (S_k > NF_GDN_MAX_SK || S_k <= 0 || S_v <= 0) {
        return; /* uniform over the whole dispatch */
    }
    const int lid = get_local_id(0);
    const int wg = get_local_size(0);
    const int cols = wg / NF_GDN_LANES;
    const int col_blocks = (S_v + cols - 1) / cols;
    const int grp = get_group_id(0);
    const int h = grp / col_blocks;
    if (h >= n_v_heads) {
        return; /* whole workgroup: grp is uniform */
    }
    const int col_local = lid / NF_GDN_LANES;
    const int lane = lid - col_local * NF_GDN_LANES;
    const int col0 = (grp - h * col_blocks) * cols;
    const int col = col0 + col_local;
    const int col_valid = col < S_v;
    const int kh = kv_head_mode == 1 ? h / (n_v_heads / n_k_heads) : h % n_k_heads;

    const nf_i64 head_floats = (nf_i64)S_v * S_k;
    NF_GLOBAL float* s_col = state + (nf_i64)state_rows[0] * n_v_heads * head_floats
                           + (nf_i64)h * head_floats + (nf_i64)(col_valid ? col : 0) * S_k;

    float s[NF_GDN_ROWS_PER_LANE];
    for (int r = 0; r < NF_GDN_ROWS_PER_LANE; r++) {
        const int i = lane + r * NF_GDN_LANES;
        s[r] = (col_valid && i < S_k) ? s_col[i] : 0.0f;
    }

    for (int t = 0; t < n_tokens; t++) {
        /* Stage q, k, the per-key decay and this workgroup's v columns of token t. */
        const nf_i64 qb = (nf_i64)q_off + (nf_i64)t * q_stride + (nf_i64)kh * S_k;
        const nf_i64 kb = (nf_i64)k_off + (nf_i64)t * k_stride + (nf_i64)kh * S_k;
        const nf_i64 gb = (nf_i64)t * n_v_heads + h;
        for (int i = lid; i < S_k; i += wg) {
            q_s[i] = q[qb + i];
            k_s[i] = k[kb + i];
            e_s[i] = gate_mode == 1 ? exp(g[gb * S_k + i]) : 0.0f;
        }
        if (lid < cols) {
            const int c = col0 + lid;
            v_s[lid] = c < S_v ? v[(nf_i64)v_off + (nf_i64)t * v_stride + (nf_i64)h * S_v + c] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const float decay = gate_mode == 1 ? 1.0f : exp(g[gb]);
        const float b = beta[gb];

        float kv = 0.0f;
        for (int r = 0; r < NF_GDN_ROWS_PER_LANE; r++) {
            const int i = lane + r * NF_GDN_LANES;
            if (i < S_k) {
                s[r] *= gate_mode == 1 ? e_s[i] : decay;
                kv += s[r] * k_s[i];
            }
        }
        kv = nf_seg_sum(red, kv, NF_GDN_LANES);

        const float delta = (v_s[col_local] - kv) * b;
        float o = 0.0f;
        for (int r = 0; r < NF_GDN_ROWS_PER_LANE; r++) {
            const int i = lane + r * NF_GDN_LANES;
            if (i < S_k) {
                s[r] += k_s[i] * delta;
                o += s[r] * q_s[i];
            }
        }
        o = nf_seg_sum(red, o, NF_GDN_LANES);

        if (col_valid && lane == 0) {
            out[(nf_i64)o_off + (nf_i64)t * o_stride + (nf_i64)h * S_v + col] = o * scale;
        }
        /* q_s/k_s/e_s/v_s are restaged for the next token only after every work-item is done with them. */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (col_valid) {
        for (int r = 0; r < NF_GDN_ROWS_PER_LANE; r++) {
            const int i = lane + r * NF_GDN_LANES;
            if (i < S_k) {
                s_col[i] = s[r];
            }
        }
    }
}

/*
 * LLM_GDN_GatePrep: decay gate g (log space) and update rate beta of gated DeltaNet and KDA layers.
 *   kda_mode -1 (GDN, qwen3next.cpp:415-446, qwen35.cpp:370-387):
 *       g[t][h] = softplus(a + dt_bias[h]) * ssm_a[h]
 *   kda_mode  0 SOFTPLUS (kimi-linear.cpp:313-324):       g[t][h][d] = softplus(f + dt_bias[h*D+d]) * ssm_a[h]
 *   kda_mode  1 LB_NEG   (kimi-k3.cpp:418-438):           g[t][h][d] = lb * sigmoid(-(ssm_a[h] * (f + dt_bias[h*D+d])))
 *   kda_mode  2 LB_POS   (bailingmoe3.cpp:271-277):       g[t][h][d] = lb * sigmoid(ssm_a[h] * (f + dt_bias[h*D+d]))
 *   beta[t][h] = sigmoid(b) for every mode.
 * Here D = head_dim, a/f is read from a_in and b from b_in:
 *   layout 0 (separate): a_in[a_off + t*a_stride + j] with j = h (GDN) or h*D + d (KDA);
 *                        b_in[b_off + t*b_stride + h]
 *   layout 1 (qwen3next fused ssm_ba, GDN only; R = n_v_heads / n_k_heads, h = kh*R + r):
 *                        b at [kh*2R + r] and a at [kh*2R + R + r] of each token row; the host binds
 *                        the fused projection to both a_in and b_in with equal offsets and strides.
 * Outputs are dense: g_out[t*n_v_heads*G + h*G + d] with G = 1 (GDN) or D (KDA), beta_out[t*n_v_heads + h].
 * lb is read only by kda_mode 1 and 2.
 * Dispatch: 1-D, one work-item per (t, h, d): global = roundUp(n_tokens * n_v_heads * G, L), any local L.
 */
NF_KERNEL void LLM_GDN_GatePrep(
    const int n_tokens,
    const int n_v_heads,
    const int n_k_heads,
    const int layout,
    const int kda_mode,
    const int head_dim,
    const float lb,
    const int a_stride,
    const int a_off,
    const int b_stride,
    const int b_off,
    NF_GLOBAL const float* NF_RESTRICT a_in,
    NF_GLOBAL const float* NF_RESTRICT b_in,
    NF_GLOBAL const float* NF_RESTRICT dt_bias,
    NF_GLOBAL const float* NF_RESTRICT ssm_a,
    NF_GLOBAL float* NF_RESTRICT g_out,
    NF_GLOBAL float* NF_RESTRICT beta_out)
{
    const int per_head = kda_mode < 0 ? 1 : head_dim;
    const nf_i64 gid = get_global_id(0);
    if (gid >= (nf_i64)n_tokens * n_v_heads * per_head) {
        return;
    }
    const int d = (int)(gid % per_head);
    const nf_i64 th = gid / per_head;
    const int h = (int)(th % n_v_heads);
    const int t = (int)(th / n_v_heads);

    int a_idx;
    int b_idx;
    if (layout == 1) {
        const int R = n_v_heads / n_k_heads;
        const int kh = h / R;
        const int r = h - kh * R;
        b_idx = kh * 2 * R + r;
        a_idx = b_idx + R;
    } else {
        a_idx = kda_mode < 0 ? h : h * head_dim + d;
        b_idx = h;
    }
    const float x = a_in[(nf_i64)a_off + (nf_i64)t * a_stride + a_idx] + dt_bias[kda_mode < 0 ? h : h * head_dim + d];
    const float A = ssm_a[h];

    float gv;
    if (kda_mode == 1) {
        gv = lb * nf_sigmoid(-(A * x));
    } else if (kda_mode == 2) {
        gv = lb * nf_sigmoid(A * x);
    } else {
        gv = nf_softplus(x) * A;
    }
    g_out[gid] = gv;

    if (d == 0) {
        beta_out[th] = nf_sigmoid(b_in[(nf_i64)b_off + (nf_i64)t * b_stride + b_idx]);
    }
}
