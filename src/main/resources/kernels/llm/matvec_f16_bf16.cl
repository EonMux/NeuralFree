/* matvec_f16_bf16.cl - F16 and BF16 matvec, fused gate/up, logits, batch matmul, multi-column matvec and embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/matvec_n.h, common/quant_blocks.h */
/*
 * Same launch contract as matvec_f32.cl: row kernels run one workgroup per output row with
 * global = rows * L, local = L (L a power of two <= LLM_DIMX; CUDA: a multiple of 32), the _2D forms
 * global = {out_dim * L, seq_len}, local = {L, 1}; their early returns are uniform per workgroup.
 * Embedding lookups are 1-D element grids. Rows and token rows are addressed in nf_i64.
 * F16 elements convert through vload_half / vload_half4, BF16 elements through bf16_to_f32.
 */

/* Four consecutive BF16 elements starting at p (element alignment only). */
NF_INLINE float4 nf_mvbf16_load4(NF_GLOBAL const ushort* p) {
    const ushort4 r = vload4(0, p);
    return NF_F4(bf16_to_f32(r.x), bf16_to_f32(r.y), bf16_to_f32(r.z), bf16_to_f32(r.w));
}

/* This work-item's share of dot(w[0..n), x[0..n)) for F16 w: float4 strides lid*4, lid*4 + L*4, ...
 * over the multiple-of-4 prefix and a scalar tail. */
NF_INLINE float nf_mvf16_partial(NF_GLOBAL const NF_HALF* NF_RESTRICT w,
                                 NF_GLOBAL const float* NF_RESTRICT x, const int n) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
    float ps = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        ps += dot(vload_half4(0, w + j), vload4(0, x + j));
    }
    for (int j = n4 + lid; j < n; j += lsize) {
        ps += vload_half(0, w + j) * x[j];
    }
    return ps;
}

/* nf_mvf16_partial for two F16 rows sharing x; adds into *gs (row g) and *us (row u). */
NF_INLINE void nf_mvf16_partial_dual(NF_GLOBAL const NF_HALF* NF_RESTRICT g,
                                     NF_GLOBAL const NF_HALF* NF_RESTRICT u,
                                     NF_GLOBAL const float* NF_RESTRICT x, const int n,
                                     float* gs, float* us) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
    float ga = 0.0f;
    float ua = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        const float4 xv = vload4(0, x + j);
        ga += dot(vload_half4(0, g + j), xv);
        ua += dot(vload_half4(0, u + j), xv);
    }
    for (int j = n4 + lid; j < n; j += lsize) {
        const float xv = x[j];
        ga += vload_half(0, g + j) * xv;
        ua += vload_half(0, u + j) * xv;
    }
    *gs += ga;
    *us += ua;
}

/* nf_mvf16_partial for BF16 w. */
NF_INLINE float nf_mvbf16_partial(NF_GLOBAL const ushort* NF_RESTRICT w,
                                  NF_GLOBAL const float* NF_RESTRICT x, const int n) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
    float ps = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        ps += dot(nf_mvbf16_load4(w + j), vload4(0, x + j));
    }
    for (int j = n4 + lid; j < n; j += lsize) {
        ps += bf16_to_f32(w[j]) * x[j];
    }
    return ps;
}

/* nf_mvf16_partial_dual for BF16 rows. */
NF_INLINE void nf_mvbf16_partial_dual(NF_GLOBAL const ushort* NF_RESTRICT g,
                                      NF_GLOBAL const ushort* NF_RESTRICT u,
                                      NF_GLOBAL const float* NF_RESTRICT x, const int n,
                                      float* gs, float* us) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
    float ga = 0.0f;
    float ua = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        const float4 xv = vload4(0, x + j);
        ga += dot(nf_mvbf16_load4(g + j), xv);
        ua += dot(nf_mvbf16_load4(u + j), xv);
    }
    for (int j = n4 + lid; j < n; j += lsize) {
        const float xv = x[j];
        ga += bf16_to_f32(g[j]) * xv;
        ua += bf16_to_f32(u[j]) * xv;
    }
    *gs += ga;
    *us += ua;
}

/* ================================ BF16 ================================ */

/* This work-item's register partials of dot(weight row at wrow, float rows t in [0, n_tokens)) for
 * BF16 w, the nf_mvbf16_partial element mapping walked per token with the weight words read once:
 * float4 strides lid*4, lid*4 + L*4, ... over the multiple-of-4 prefix and a scalar tail. */
NF_INLINE void nf_mvbf16_row_partial_n(NF_GLOBAL const ushort* NF_RESTRICT wrow,
                                       NF_GLOBAL const float* NF_RESTRICT x, const int n,
                                       const int n_tokens, float* ps) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
#pragma unroll
    for (int j = 0; j < NF_MVQN_MAX; j++) ps[j] = 0.0f;
    for (int i = lid * 4; i < n4; i += lsize * 4) {
        const float4 wv = nf_mvbf16_load4(wrow + i);
#pragma unroll
        for (int t = 0; t < NF_MVQN_MAX; t++) {
            if (t < n_tokens) ps[t] += dot(wv, vload4(0, x + (nf_i64)t * n + i));
        }
    }
    for (int i = n4 + lid; i < n; i += lsize) {
        const float wv = bf16_to_f32(wrow[i]);
#pragma unroll
        for (int t = 0; t < NF_MVQN_MAX; t++) {
            if (t < n_tokens) ps[t] += wv * x[(nf_i64)t * n + i];
        }
    }
}

/* BF16_MatVec over n_tokens consecutive float rows in one launch (n_tokens in [1, NF_MVQN_MAX]):
 * the weight row is streamed once and dotted against every live token row, where one BF16_MatVec
 * per token re-reads the row per token. Token t's dot lands at output[t * out_dim + row], and
 * NF_MVQTN_TREE reduces the group in one pass, so token t's result is the exact single-row result. */
NF_REQD_SG_ATTR NF_KERNEL void BF16_MatVecN(
    const int out_dim,
    const int in_dim,
    const int n_tokens,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    float ps[NF_MVQN_MAX];
    nf_mvbf16_row_partial_n(weight + (nf_i64)row * in_dim, input, in_dim, n_tokens, ps);
    NF_LOCAL float ts[NF_MVQN_MAX * NF_MVQTN_SLOTS];
    NF_MVQTN_TREE(ts, ps, n_tokens, output, out_dim, row)
}

/* output[row] = dot(weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void BF16_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvbf16_partial(weight + (nf_i64)row * in_dim, input, in_dim));
    if (get_local_id(0) == 0) output[row] = acc;
}

/* addend_output[row] += dot(weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void BF16_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvbf16_partial(weight + (nf_i64)row * in_dim, input, in_dim));
    if (get_local_id(0) == 0) addend_output[row] += acc;
}

/* output[i] = embeddings[token_id, i]; global = embed_dim rounded up to the local size. */
NF_KERNEL void BF16_EmbeddingLookup(
    const int embed_dim,
    const int token_id,
    NF_GLOBAL const ushort* NF_RESTRICT embeddings,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int i = get_global_id(0);
    if (i >= embed_dim) return;
    output[i] = bf16_to_f32(embeddings[(nf_i64)token_id * embed_dim + i]);
}

/* output[row] = silu(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void BF16_MatVecFusedGateUp(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const ushort* NF_RESTRICT gate_weight,
    NF_GLOBAL const ushort* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const nf_i64 rbase = (nf_i64)row * in_dim;
    float gs = 0.0f;
    float us = 0.0f;
    nf_mvbf16_partial_dual(gate_weight + rbase, up_weight + rbase, input, in_dim, &gs, &us);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (get_local_id(0) == 0) {
        output[row] = nf_silu(gate_val) * up_val;
    }
}

/* logits[row] = dot(weight[row, :], hidden). */
NF_REQD_SG_ATTR NF_KERNEL void BF16_ComputeLogits(
    const int vocab_size,
    const int embed_dim,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL float* NF_RESTRICT logits)
{
    const int row = get_group_id(0);
    if (row >= vocab_size) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvbf16_partial(weight + (nf_i64)row * embed_dim, hidden, embed_dim));
    if (get_local_id(0) == 0) logits[row] = acc;
}

/* output[s * embed_dim + i] = weight[token_ids[s], i]; global = seq_len * embed_dim rounded up. */
NF_KERNEL void BF16_BatchEmbeddingLookup(
    const int embed_dim,
    const int seq_len,
    NF_GLOBAL const int* NF_RESTRICT token_ids,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)seq_len * embed_dim) return;

    const int s = (int)(gid / embed_dim);
    const int i = (int)(gid - (nf_i64)s * embed_dim);
    output[gid] = bf16_to_f32(weight[(nf_i64)token_ids[s] * embed_dim + i]);
}

/* output[s * out_dim + row] = dot(weight[row, :], input[s, :]) for s in [0, seq_len). */
NF_REQD_SG_ATTR NF_KERNEL void BF16_BatchMatMul(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const ushort* wrow = weight + (nf_i64)row * in_dim;
    NF_LOCAL float scratch[LLM_DIMX];
    for (int s = 0; s < seq_len; s++) {
        const float acc = nf_wg_sum(scratch, nf_mvbf16_partial(wrow, input + (nf_i64)s * in_dim, in_dim));
        if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
    }
}

/* BF16_BatchMatMul with one (row, token) pair per workgroup. */
NF_REQD_SG_ATTR NF_KERNEL void BF16_BatchMatMul_2D(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    const int s = get_global_id(1);
    if (row >= out_dim || s >= seq_len) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvbf16_partial(weight + (nf_i64)row * in_dim,
                                                           input + (nf_i64)s * in_dim, in_dim));
    if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
}

/* ================================ F16 ================================= */

/* nf_mvbf16_row_partial_n for F16 w. */
NF_INLINE void nf_mvf16_row_partial_n(NF_GLOBAL const NF_HALF* NF_RESTRICT wrow,
                                      NF_GLOBAL const float* NF_RESTRICT x, const int n,
                                      const int n_tokens, float* ps) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
#pragma unroll
    for (int j = 0; j < NF_MVQN_MAX; j++) ps[j] = 0.0f;
    for (int i = lid * 4; i < n4; i += lsize * 4) {
        const float4 wv = vload_half4(0, wrow + i);
#pragma unroll
        for (int t = 0; t < NF_MVQN_MAX; t++) {
            if (t < n_tokens) ps[t] += dot(wv, vload4(0, x + (nf_i64)t * n + i));
        }
    }
    for (int i = n4 + lid; i < n; i += lsize) {
        const float wv = vload_half(0, wrow + i);
#pragma unroll
        for (int t = 0; t < NF_MVQN_MAX; t++) {
            if (t < n_tokens) ps[t] += wv * x[(nf_i64)t * n + i];
        }
    }
}

/* F16_MatVec over n_tokens consecutive float rows in one launch (n_tokens in [1, NF_MVQN_MAX]):
 * the weight row is streamed once and dotted against every live token row, where one F16_MatVec
 * per token re-reads the row per token. Token t's dot lands at output[t * out_dim + row], and
 * NF_MVQTN_TREE reduces the group in one pass, so token t's result is the exact single-row result. */
NF_REQD_SG_ATTR NF_KERNEL void F16_MatVecN(
    const int out_dim,
    const int in_dim,
    const int n_tokens,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    float ps[NF_MVQN_MAX];
    nf_mvf16_row_partial_n(weight + (nf_i64)row * in_dim, input, in_dim, n_tokens, ps);
    NF_LOCAL float ts[NF_MVQN_MAX * NF_MVQTN_SLOTS];
    NF_MVQTN_TREE(ts, ps, n_tokens, output, out_dim, row)
}

/* output[row] = dot(weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void F16_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf16_partial(weight + (nf_i64)row * in_dim, input, in_dim));
    if (get_local_id(0) == 0) output[row] = acc;
}

/* addend_output[row] += dot(weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void F16_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf16_partial(weight + (nf_i64)row * in_dim, input, in_dim));
    if (get_local_id(0) == 0) addend_output[row] += acc;
}

/* output[i] = embeddings[token_id, i]; global = embed_dim rounded up to the local size. */
NF_KERNEL void F16_EmbeddingLookup(
    const int embed_dim,
    const int token_id,
    NF_GLOBAL const NF_HALF* NF_RESTRICT embeddings,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int i = get_global_id(0);
    if (i >= embed_dim) return;
    output[i] = vload_half(0, embeddings + (nf_i64)token_id * embed_dim + i);
}

/* output[row] = silu(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void F16_MatVecFusedGateUp(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const NF_HALF* NF_RESTRICT gate_weight,
    NF_GLOBAL const NF_HALF* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const nf_i64 rbase = (nf_i64)row * in_dim;
    float gs = 0.0f;
    float us = 0.0f;
    nf_mvf16_partial_dual(gate_weight + rbase, up_weight + rbase, input, in_dim, &gs, &us);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (get_local_id(0) == 0) {
        output[row] = nf_silu(gate_val) * up_val;
    }
}

/* logits[row] = dot(weight[row, :], hidden). */
NF_REQD_SG_ATTR NF_KERNEL void F16_ComputeLogits(
    const int vocab_size,
    const int embed_dim,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL float* NF_RESTRICT logits)
{
    const int row = get_group_id(0);
    if (row >= vocab_size) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf16_partial(weight + (nf_i64)row * embed_dim, hidden, embed_dim));
    if (get_local_id(0) == 0) logits[row] = acc;
}

/* output[s * embed_dim + i] = weight[token_ids[s], i]; global = seq_len * embed_dim rounded up. */
NF_KERNEL void F16_BatchEmbeddingLookup(
    const int embed_dim,
    const int seq_len,
    NF_GLOBAL const int* NF_RESTRICT token_ids,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)seq_len * embed_dim) return;

    const int s = (int)(gid / embed_dim);
    const int i = (int)(gid - (nf_i64)s * embed_dim);
    output[gid] = vload_half(0, weight + (nf_i64)token_ids[s] * embed_dim + i);
}

/* output[s * out_dim + row] = dot(weight[row, :], input[s, :]) for s in [0, seq_len). */
NF_REQD_SG_ATTR NF_KERNEL void F16_BatchMatMul(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    NF_GLOBAL const NF_HALF* wrow = weight + (nf_i64)row * in_dim;
    NF_LOCAL float scratch[LLM_DIMX];
    for (int s = 0; s < seq_len; s++) {
        const float acc = nf_wg_sum(scratch, nf_mvf16_partial(wrow, input + (nf_i64)s * in_dim, in_dim));
        if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
    }
}

/* F16_BatchMatMul with one (row, token) pair per workgroup. */
NF_REQD_SG_ATTR NF_KERNEL void F16_BatchMatMul_2D(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    const int s = get_global_id(1);
    if (row >= out_dim || s >= seq_len) return;

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, nf_mvf16_partial(weight + (nf_i64)row * in_dim,
                                                          input + (nf_i64)s * in_dim, in_dim));
    if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
}
