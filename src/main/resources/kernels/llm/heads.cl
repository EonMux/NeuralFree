/* heads.cl - LLM_SwitchedLoraAdd (graniteswitch per-token LoRA). Requires: common/dialect.h, common/config.h, common/reduce.h */
/*
 * Activations are token-major F32: element (t, c) of a view is v[v_off + t*v_stride + c]. Offsets and
 * strides are in float elements; addresses are formed in nf_i64. Java: gguf.ops.heads.HeadsKernels.
 *
 * LLM_SwitchedLoraAdd(i n_tokens, i in_dim, i out_dim, i rank, i stage, i n_slots, i x_stride, i x_off,
 *                     i y_stride, i y_off, i tmp_off, R<int> slots, R A, R B, R x, RWn tmp, RWn y)
 *     y[t] += B[s_t] (A[s_t] x[t]) per token (models/granite-switch.cpp build_switched_lora_delta, two
 *     ggml_mul_mat_id with the adapter slot as expert id, then ggml_add onto the base projection):
 *       s_t = clamp(slots[t], 0, n_slots-1)   (ggml_clamp(slot, 0, n_adapters); n_slots = n_adapters+1,
 *                                              slot 0 is the base model's zero delta)
 *       A   = lora_a {in_dim, rank, n_slots}:  A[s][r][i] at (s*rank + r)*in_dim + i
 *       B   = lora_b {rank, out_dim, n_slots}: B[s][o][r] at (s*out_dim + o)*rank + r
 *     stage 0: tmp[tmp_off + t*rank + r] = sum_i A[s_t][r][i] * x[x_off + t*x_stride + i]
 *              one workgroup per (t, r): group g = t*rank + r, the work-items stride over i and the
 *              partial sums are reduced with nf_wg_sum. global = n_tokens*rank*L, local = {L}, L a power of
 *              two in [32, LLM_DIMX].
 *     stage 1: y[y_off + t*y_stride + o] += sum_r B[s_t][o][r] * tmp[tmp_off + t*rank + r], r ascending
 *              one work-item per (t, o): global = roundUp(n_tokens*out_dim, L), local = {L}.
 *     The host enqueues stage 0 then stage 1 on one queue. tmp and y may live in shared scratch buffers
 *     (no restrict); x, tmp and y must not overlap one another.
 */

/* Adapter slot of one token, clamped to the stacked slot range [0, n_slots). */
NF_INLINE int nf_heads_lora_slot(const int slot, const int n_slots) {
    return slot < 0 ? 0 : (slot >= n_slots ? n_slots - 1 : slot);
}

NF_KERNEL void LLM_SwitchedLoraAdd(
    const int n_tokens,
    const int in_dim,
    const int out_dim,
    const int rank,
    const int stage,
    const int n_slots,
    const int x_stride,
    const int x_off,
    const int y_stride,
    const int y_off,
    const int tmp_off,
    NF_GLOBAL const int* NF_RESTRICT slots,
    NF_GLOBAL const float* NF_RESTRICT A,
    NF_GLOBAL const float* NF_RESTRICT B,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL float* tmp,
    NF_GLOBAL float* y)
{
    NF_LOCAL float red[LLM_DIMX];
    if (stage == 0) {
        /* The whole workgroup shares g, so this return never splits a workgroup at the barrier below. */
        const nf_i64 g = (nf_i64)get_group_id(0);
        if (g >= (nf_i64)n_tokens * rank) return;
        const int t = (int)(g / rank);
        const int r = (int)(g - (nf_i64)t * rank);
        const int s = nf_heads_lora_slot(slots[t], n_slots);
        const nf_i64 a_row = ((nf_i64)s * rank + r) * in_dim;
        const nf_i64 x_row = (nf_i64)x_off + (nf_i64)t * x_stride;
        float acc = 0.0f;
        for (int i = (int)get_local_id(0); i < in_dim; i += (int)get_local_size(0)) {
            acc += A[a_row + i] * x[x_row + i];
        }
        const float sum = nf_wg_sum(red, acc);
        if (get_local_id(0) == 0) {
            tmp[(nf_i64)tmp_off + (nf_i64)t * rank + r] = sum;
        }
    } else {
        const nf_i64 idx = (nf_i64)get_global_id(0);
        if (idx >= (nf_i64)n_tokens * out_dim) return;
        const int t = (int)(idx / out_dim);
        const int o = (int)(idx - (nf_i64)t * out_dim);
        const int s = nf_heads_lora_slot(slots[t], n_slots);
        const nf_i64 b_row = ((nf_i64)s * out_dim + o) * rank;
        const nf_i64 t_row = (nf_i64)tmp_off + (nf_i64)t * rank;
        float acc = 0.0f;
        for (int r = 0; r < rank; r++) {
            acc += B[b_row + r] * tmp[t_row + r];
        }
        const nf_i64 yi = (nf_i64)y_off + (nf_i64)t * y_stride + o;
        y[yi] = y[yi] + acc;
    }
}
