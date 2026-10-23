/* recurrent/lightning.cl - LLM_LightningAttn lightning attention recurrence. Requires: common/dialect.h, common/config.h, common/reduce.h, common/math.h */
/*
 * MiniMax-01 lightning attention (llama.cpp src/models/minimax-01.cpp:288-398). With s = slopes[h] * layer_scale
 * the block form computed by llama.cpp,
 *     o_t    = e^{-s(t+1)} q_t . kv_old + sum_{j<=t} e^{-s(t-j)} (q_t . k_j) v_j
 *     kv_new = e^{-sT} kv_old + sum_j e^{-s(T-1-j)} k_j (x) v_j,
 * equals the per-token recurrence evaluated here:
 *     kv = e^{-s} kv + k_t (x) v_t,   o_t = q_t . kv
 * The state keeps ggml's [h][v][k] order (kv_old is reshaped to {D(k), D(v), H}): s[h*D*D + v*D + k].
 */

/* Largest head dim of LLM_LightningAttn: 32 lanes x NF_LA_ROWS_PER_LANE key rows per value column. */
#define NF_LA_MAX_D 256
#define NF_LA_LANES 32
#define NF_LA_ROWS_PER_LANE (NF_LA_MAX_D / NF_LA_LANES)
#define NF_LA_MAX_COLS (LLM_DIMX / NF_LA_LANES)

/*
 * LLM_LightningAttn: n_tokens consecutive tokens of one sequence (decode and prefill).
 *   qkv[t*n_head*3D + h*3D + {0, D, 2D} + i]   q | k | v of head h, after silu (minimax-01.cpp:294-305)
 *   out[t*n_head*D + h*D + v]                  head-major row, the layout attn_norm_2 normalizes
 *   state[state_rows[0]*n_head*D*D + h*D*D + v*D + k]
 * No 1/sqrt(D) scaling is applied (llama.cpp applies none).
 *
 * Dispatch: 1-D, local = {L} with L a multiple of 32 and L <= LLM_DIMX; cols = L/32 value columns per
 * workgroup, 32 lanes per column; lane j holds key rows j, j+32, ... of its column in registers.
 * global = n_head * ceil(D / cols) * L. Requires D <= NF_LA_MAX_D; larger heads leave out and state
 * untouched (the host rejects them first).
 */
NF_KERNEL void LLM_LightningAttn(
    const int n_tokens,
    const int n_head,
    const int D,
    const float layer_scale,
    NF_GLOBAL const float* NF_RESTRICT slopes,
    NF_GLOBAL const float* NF_RESTRICT qkv,
    NF_GLOBAL float* state,
    NF_GLOBAL const int* NF_RESTRICT state_rows,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float q_s[NF_LA_MAX_D];
    NF_LOCAL float k_s[NF_LA_MAX_D];
    NF_LOCAL float v_s[NF_LA_MAX_COLS];
    NF_LOCAL float red[LLM_DIMX];

    if (D > NF_LA_MAX_D || D <= 0) {
        return; /* uniform over the whole dispatch */
    }
    const int lid = get_local_id(0);
    const int wg = get_local_size(0);
    const int cols = wg / NF_LA_LANES;
    const int col_blocks = (D + cols - 1) / cols;
    const int grp = get_group_id(0);
    const int h = grp / col_blocks;
    if (h >= n_head) {
        return; /* whole workgroup: grp is uniform */
    }
    const int col_local = lid / NF_LA_LANES;
    const int lane = lid - col_local * NF_LA_LANES;
    const int col0 = (grp - h * col_blocks) * cols;
    const int col = col0 + col_local;
    const int col_valid = col < D;

    const float decay = exp(-(slopes[h] * layer_scale));
    const nf_i64 head_floats = (nf_i64)D * D;
    NF_GLOBAL float* s_col = state + (nf_i64)state_rows[0] * n_head * head_floats
                           + (nf_i64)h * head_floats + (nf_i64)(col_valid ? col : 0) * D;

    float s[NF_LA_ROWS_PER_LANE];
    for (int r = 0; r < NF_LA_ROWS_PER_LANE; r++) {
        const int i = lane + r * NF_LA_LANES;
        s[r] = (col_valid && i < D) ? s_col[i] : 0.0f;
    }

    const nf_i64 row_floats = (nf_i64)n_head * 3 * D;
    for (int t = 0; t < n_tokens; t++) {
        const nf_i64 base = (nf_i64)t * row_floats + (nf_i64)h * 3 * D;
        for (int i = lid; i < D; i += wg) {
            q_s[i] = qkv[base + i];
            k_s[i] = qkv[base + D + i];
        }
        if (lid < cols) {
            const int c = col0 + lid;
            v_s[lid] = c < D ? qkv[base + 2 * D + c] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const float vv = v_s[col_local];
        float o = 0.0f;
        for (int r = 0; r < NF_LA_ROWS_PER_LANE; r++) {
            const int i = lane + r * NF_LA_LANES;
            if (i < D) {
                s[r] = s[r] * decay + k_s[i] * vv;
                o += s[r] * q_s[i];
            }
        }
        o = nf_seg_sum(red, o, NF_LA_LANES);

        if (col_valid && lane == 0) {
            out[(nf_i64)t * n_head * D + (nf_i64)h * D + col] = o;
        }
        /* q_s/k_s/v_s are restaged for the next token only after every work-item is done with them. */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (col_valid) {
        for (int r = 0; r < NF_LA_ROWS_PER_LANE; r++) {
            const int i = lane + r * NF_LA_LANES;
            if (i < D) {
                s_col[i] = s[r];
            }
        }
    }
}
