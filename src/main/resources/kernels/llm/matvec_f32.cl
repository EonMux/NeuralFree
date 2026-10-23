/* matvec_f32.cl - F32 matvec, fused gate/up, logits, batch matmul and embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h */
/*
 * Launch contract of the row kernels (every kernel except the embedding lookups): one workgroup per
 * output row, global = rows * L, local = L with L a power of two <= LLM_DIMX (CUDA: a multiple of 32).
 * The _2D form uses global = {out_dim * L, seq_len}, local = {L, 1}. The `row >= rows` and
 * `s >= seq_len` returns depend only on the workgroup, so every barrier of a live workgroup is
 * reached by all of its work-items. Weight rows, token rows and batch rows are addressed in nf_i64.
 * Embedding lookups are 1-D element grids with a per-element bounds check.
 */

/* This work-item's share of dot(w[0..n), x[0..n)): float4 strides lid*4, lid*4 + L*4, ... over the
 * multiple-of-4 prefix and a scalar tail; the L shares add to the full dot product. */
NF_INLINE float nf_mvf32_partial(NF_GLOBAL const float* NF_RESTRICT w,
                                 NF_GLOBAL const float* NF_RESTRICT x, const int n) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
    float ps = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        ps += dot(vload4(0, w + j), vload4(0, x + j));
    }
    for (int j = n4 + lid; j < n; j += lsize) {
        ps += w[j] * x[j];
    }
    return ps;
}

/* nf_mvf32_partial for two weight rows sharing x; adds into *gs (row g) and *us (row u). */
NF_INLINE void nf_mvf32_partial_dual(NF_GLOBAL const float* NF_RESTRICT g,
                                     NF_GLOBAL const float* NF_RESTRICT u,
                                     NF_GLOBAL const float* NF_RESTRICT x, const int n,
                                     float* gs, float* us) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
    float ga = 0.0f;
    float ua = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        const float4 xv = vload4(0, x + j);
        ga += dot(vload4(0, g + j), xv);
        ua += dot(vload4(0, u + j), xv);
    }
    for (int j = n4 + lid; j < n; j += lsize) {
        const float xv = x[j];
        ga += g[j] * xv;
        ua += u[j] * xv;
    }
    *gs += ga;
    *us += ua;
}

/* output[row] = dot(weight[row, :], input). */
NF_KERNEL void Float_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf32_partial(weight + (nf_i64)row * in_dim, input, in_dim));
    if (get_local_id(0) == 0) output[row] = acc;
}

/* addend_output[row] += dot(weight[row, :], input). */
NF_KERNEL void Float_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf32_partial(weight + (nf_i64)row * in_dim, input, in_dim));
    if (get_local_id(0) == 0) addend_output[row] += acc;
}

/* output[row] = silu(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_KERNEL void Float_MatVecFusedGateUp(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const float* NF_RESTRICT gate_weight,
    NF_GLOBAL const float* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const nf_i64 rbase = (nf_i64)row * in_dim;
    float gs = 0.0f;
    float us = 0.0f;
    nf_mvf32_partial_dual(gate_weight + rbase, up_weight + rbase, input, in_dim, &gs, &us);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (get_local_id(0) == 0) {
        output[row] = nf_silu(gate_val) * up_val;
    }
}

/* output[row] = gelu_tanh(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_KERNEL void Float_MatVecFusedGateUpGelu(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const float* NF_RESTRICT gate_weight,
    NF_GLOBAL const float* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const nf_i64 rbase = (nf_i64)row * in_dim;
    float gs = 0.0f;
    float us = 0.0f;
    nf_mvf32_partial_dual(gate_weight + rbase, up_weight + rbase, input, in_dim, &gs, &us);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (get_local_id(0) == 0) {
        output[row] = nf_gelu_tanh(gate_val) * up_val;
    }
}

/* logits[row] = dot(weight[row, :], hidden); the Float_MatVec structure over vocab rows. */
NF_KERNEL void LLM_ComputeLogits(
    const int vocab_size,
    const int embed_dim,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL float* NF_RESTRICT logits)
{
    const int row = get_group_id(0);
    if (row >= vocab_size) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf32_partial(weight + (nf_i64)row * embed_dim, hidden, embed_dim));
    if (get_local_id(0) == 0) logits[row] = acc;
}

/* output[s * out_dim + row] = dot(weight[row, :], input[s, :]) for s in [0, seq_len); one
 * workgroup per row walks every token. */
NF_KERNEL void LLM_Batch_Float_MatMul(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const float* wrow = weight + (nf_i64)row * in_dim;
    NF_LOCAL float scratch[LLM_DIMX];
    for (int s = 0; s < seq_len; s++) {
        const float acc = nf_wg_sum(scratch, nf_mvf32_partial(wrow, input + (nf_i64)s * in_dim, in_dim));
        if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
    }
}

/* LLM_Batch_Float_MatMul with one (row, token) pair per workgroup. */
NF_KERNEL void LLM_Batch_Float_MatMul_2D(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    const int s = get_global_id(1);
    if (row >= out_dim || s >= seq_len) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf32_partial(weight + (nf_i64)row * in_dim,
                                                          input + (nf_i64)s * in_dim, in_dim));
    if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
}

/* output[i] = weight[token_id, i]; global = embed_dim rounded up to the local size. */
NF_KERNEL void LLM_EmbeddingLookup(
    const int embed_dim,
    const int token_id,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int i = get_global_id(0);
    if (i >= embed_dim) return;
    output[i] = weight[(nf_i64)token_id * embed_dim + i];
}

/* output[s * embed_dim + i] = weight[token_ids[s], i]; global = seq_len * embed_dim rounded up. */
NF_KERNEL void LLM_Batch_EmbeddingLookup(
    const int embed_dim,
    const int seq_len,
    NF_GLOBAL const int* NF_RESTRICT token_ids,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)seq_len * embed_dim) return;

    const int s = (int)(gid / embed_dim);
    const int i = (int)(gid - (nf_i64)s * embed_dim);
    output[gid] = weight[(nf_i64)token_ids[s] * embed_dim + i];
}
