/* matvec_q8_0.cl - Q8_0 matvec, fused gate/up, logits, batch matmul, Q8 activation quantization, the Q8_1-activation family forms, the gelu twin and embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/matvec_n.h, common/quant_blocks.h, common/q8_dot.h, common/matvec_family.h */
/*
 * Q8_0 block (ggml block_q8_0): d (f16) at byte 0, qs[32] (int8) at byte 2; element e = qs[e] * d.
 *
 * Launch contract of the row kernels: one workgroup per output row, global = rows * L, local = L
 * (L a power of two <= LLM_DIMX and a multiple of Q8_0_THREADS_PER_BLOCK; CUDA: a multiple of 32);
 * the _Q8 matvec forms take the same geometry with one Q8_1 activation row (in_dim / 32 * 36 bytes)
 * in place of the float row, and their MatVec2 forms pair output rows (global = ceil(rows / 2) * L,
 * an odd row count repeats the last row's weights and drops the second store); the MatVecN forms
 * keep the same geometry and take n_tokens consecutive rows in one launch (n_tokens in
 * [1, NF_MVQN_MAX]): Q8_0_MatVecN n_tokens float rows; the Q8_1-activation forms
 * (Q8_0_MatVec_Q8, Q8_0_MatVecN_Q8, the counted Q8_0_MatVecN{2..16}_Q8 / Q8_0_MatVec2N{2..16}_Q8,
 * Q8_0_MatVec2_Q8, Q8_0_MatVecAdd_Q8, Q8_0_MatVecFusedGateUp_Q8 and Q8_0_MatVec2FusedGateUp_Q8)
 * come from IQ_ALL_KERNELS_Q8 and the IQ_MVQN_NC_HIGH counts of common/matvec_family.h, whose
 * counted forms fold the token count to the name's NC (a launch passes n_tokens = NC) and whose
 * MatVec2N forms pair output rows on top of the counted singles (global = ceil(out_dim / 2) * L,
 * the tail pair repeats the last row's weights and rewrites its outputs with the identical
 * values), so the Q8_1 rows are read once per two weight rows; the _2D form uses
 * global = {out_dim * L, seq_len}, local = {L, 1}. Except LLM_Batch_Q8_0_MatMul(_2D),
 * which give each work-item whole blocks, the row kernels let Q8_0_THREADS_PER_BLOCK adjacent
 * work-items share one block, Q8_0_ELEMENTS_PER_THREAD consecutive elements each. The row and token
 * early returns are uniform per workgroup. Weight rows, token rows and batch rows are addressed in nf_i64.
 * LLM_QuantizeQ8 is a 1-D grid-stride loop over blocks, one block per work-item per step, so any
 * global size is valid (a single workgroup, global = local = L, suffices); embedding lookups are
 * 1-D element grids.
 */

/* Q8_0_ELEMENTS_PER_THREAD-element slice `sub` (0..Q8_0_THREADS_PER_BLOCK-1) of the dot product of
 * the Q8_0 block at bp with x[0..32), scaled by the block's d. */
NF_INLINE float nf_mvq8_slice_dot(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* x, const int sub) {
    const int e = sub * Q8_0_ELEMENTS_PER_THREAD;
    NF_GLOBAL const nf_i8* q = (NF_GLOBAL const nf_i8*)(bp + 2 + e);
    const float s = dot(vload4(0, x + e), convert_float4(vload4(0, q)))
                  + dot(vload4(0, x + e + 4), convert_float4(vload4(0, q + 4)));
    return s * q8_0_scale(bp);
}

/* This work-item's share of dot(Q8_0 row at wrow, x[0..nblk*32)) under the shared-block mapping. */
NF_INLINE float nf_mvq8_row_partial(NF_GLOBAL const uchar* NF_RESTRICT wrow,
                                    NF_GLOBAL const float* NF_RESTRICT x, const int nblk) {
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / Q8_0_THREADS_PER_BLOCK;
    const int sub = lid % Q8_0_THREADS_PER_BLOCK;
    float ps = 0.0f;
    for (int b = lid / Q8_0_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        ps += nf_mvq8_slice_dot(wrow + b * Q8_0_BLOCK_BYTES, x + b * Q8_0_BLOCK_SIZE, sub);
    }
    return ps;
}

/* nf_mvq8_row_partial for two Q8_0 rows sharing x; adds into *gs (row g) and *us (row u). */
NF_INLINE void nf_mvq8_row_partial_dual(NF_GLOBAL const uchar* NF_RESTRICT grow,
                                        NF_GLOBAL const uchar* NF_RESTRICT urow,
                                        NF_GLOBAL const float* NF_RESTRICT x, const int nblk,
                                        float* gs, float* us) {
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / Q8_0_THREADS_PER_BLOCK;
    const int sub = lid % Q8_0_THREADS_PER_BLOCK;
    float ga = 0.0f;
    float ua = 0.0f;
    for (int b = lid / Q8_0_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        NF_GLOBAL const float* xb = x + b * Q8_0_BLOCK_SIZE;
        ga += nf_mvq8_slice_dot(grow + b * Q8_0_BLOCK_BYTES, xb, sub);
        ua += nf_mvq8_slice_dot(urow + b * Q8_0_BLOCK_BYTES, xb, sub);
    }
    *gs += ga;
    *us += ua;
}

/* This work-item's share of dot(Q8_0 row at wrow, x[0..nblk*32)) with whole blocks per work-item
 * (blocks lid, lid + L, ...). */
NF_INLINE float nf_mvq8_row_partial_whole(NF_GLOBAL const uchar* NF_RESTRICT wrow,
                                          NF_GLOBAL const float* NF_RESTRICT x, const int nblk) {
    const int lsize = get_local_size(0);
    float ps = 0.0f;
    for (int b = get_local_id(0); b < nblk; b += lsize) {
        NF_GLOBAL const uchar* bp = wrow + b * Q8_0_BLOCK_BYTES;
        NF_GLOBAL const float* xb = x + b * Q8_0_BLOCK_SIZE;
        float bs = 0.0f;
#pragma unroll
        for (int sub = 0; sub < Q8_0_THREADS_PER_BLOCK; sub++) {
            const int e = sub * Q8_0_ELEMENTS_PER_THREAD;
            NF_GLOBAL const nf_i8* q = (NF_GLOBAL const nf_i8*)(bp + 2 + e);
            bs += dot(vload4(0, xb + e), convert_float4(vload4(0, q)))
                + dot(vload4(0, xb + e + 4), convert_float4(vload4(0, q + 4)));
        }
        ps += bs * q8_0_scale(bp);
    }
    return ps;
}

/* Integer dot product of slice `sub` (0..Q8_0_THREADS_PER_BLOCK-1) of a Q8_0 weight block at wp
 * with Q8_1 block qblk of the activation row at xq, as ggml vec_dot_q8_0_q8_1: the code products
 * only, the Q8_1 block's sum term unused. The codes read as 32-bit words and the products packed
 * through dp4a: the byte products stay below 2^14 and the total below 2^18, so the packed int sum
 * is the exact scalar byte-product sum. The Q8_0 block layout leaves the weight codes two-byte
 * aligned (nf_read_u32_u16), the Q8_1 codes start at offset 4 of a 36-byte block (nf_read_u32). */
NF_INLINE int nf_mvq8_slice_idot(NF_GLOBAL const uchar* wp, NF_GLOBAL const uchar* xq,
                                 const int qblk, const int sub) {
    const int e = sub * Q8_0_ELEMENTS_PER_THREAD;
    NF_GLOBAL const uchar* w = wp + 2 + e;
    NF_GLOBAL const uchar* q = xq + qblk * Q8_1_BLOCK_BYTES + Q8_1_QS_OFFSET + e;
    int isum = NF_DP4A((int)nf_read_u32_u16(w), (int)nf_read_u32(q), 0);
    isum = NF_DP4A((int)nf_read_u32_u16(w + 4), (int)nf_read_u32(q + 4), isum);
    return isum;
}

/* The Q8_0 decoder the IQ_ALL_KERNELS_Q8 family kernels use: this work-item's share of the integer
 * dot of the Q8_0 block at bp with Q8_1 block blk of the activation row at xq, scaled by both
 * block scales (the Q8_1 sum term unused). */
NF_INLINE float Q8_0_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq,
                               const int blk, const int ltid) {
    return (float)nf_mvq8_slice_idot(bp, xq, blk, ltid) * q8_0_scale(bp) * nf_q8_scale(xq, blk);
}

/* Q8_0_decode_q8 for two Q8_0 blocks sharing the Q8_1 block; adds into *gs and *us. */
NF_INLINE void Q8_0_decode_q8_dual(NF_GLOBAL const uchar* bp0, NF_GLOBAL const uchar* bp1,
                                   NF_GLOBAL const uchar* xq, const int blk, const int ltid,
                                   float* gs, float* us) {
    *gs += Q8_0_decode_q8(bp0, xq, blk, ltid);
    *us += Q8_0_decode_q8(bp1, xq, blk, ltid);
}

/* Q8_0_decode_q8 for two Q8_0 weight rows sharing the Q8_1 row, whole rows per call; adds the
 * row sums into *gs and *us. The fused gate/up kernels accumulate through it. */
NF_INLINE void nf_mvq8q8_row_partial_dual(NF_GLOBAL const uchar* NF_RESTRICT grow,
                                          NF_GLOBAL const uchar* NF_RESTRICT urow,
                                          NF_GLOBAL const uchar* NF_RESTRICT xq, const int nblk,
                                          float* gs, float* us) {
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / Q8_0_THREADS_PER_BLOCK;
    const int sub = lid % Q8_0_THREADS_PER_BLOCK;
    float ga = 0.0f;
    float ua = 0.0f;
    for (int b = lid / Q8_0_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        NF_GLOBAL const uchar* wp_g = grow + b * Q8_0_BLOCK_BYTES;
        NF_GLOBAL const uchar* wp_u = urow + b * Q8_0_BLOCK_BYTES;
        ga += (float)nf_mvq8_slice_idot(wp_g, xq, b, sub) * q8_0_scale(wp_g) * nf_q8_scale(xq, b);
        ua += (float)nf_mvq8_slice_idot(wp_u, xq, b, sub) * q8_0_scale(wp_u) * nf_q8_scale(xq, b);
    }
    *gs += ga;
    *us += ua;
}

/* Element i of the Q8_0 row at rbase; a non-finite block scale (corrupt or padding block) decodes as 0. */
NF_INLINE float nf_mvq8_elem(NF_GLOBAL const uchar* NF_RESTRICT rbase, const int i) {
    NF_GLOBAL const uchar* bp = rbase + (i / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
    float scale = q8_0_scale(bp);
    if (!isfinite(scale)) scale = 0.0f;
    return (float)((nf_i8)bp[2 + i % Q8_0_BLOCK_SIZE]) * scale;
}

/* output[row] = dot(weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void Q8_0_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvq8_row_partial(weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES, input, nblk));
    if (get_local_id(0) == 0) output[row] = acc;
}

/* addend_output[row] += dot(weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void Q8_0_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvq8_row_partial(weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES, input, nblk));
    if (get_local_id(0) == 0) addend_output[row] += acc;
}

/* output[row] = silu(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void Q8_0_MatVecFusedGateUp(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight,
    NF_GLOBAL const uchar* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    const nf_i64 rbase = (nf_i64)row * nblk * Q8_0_BLOCK_BYTES;
    float gs = 0.0f;
    float us = 0.0f;
    nf_mvq8_row_partial_dual(gate_weight + rbase, up_weight + rbase, input, nblk, &gs, &us);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (get_local_id(0) == 0) {
        output[row] = nf_silu(gate_val) * up_val;
    }
}

/* output[row] = gelu_tanh(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_REQD_SG_ATTR NF_KERNEL void Q8_0_MatVecFusedGateUpGelu(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight,
    NF_GLOBAL const uchar* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    const nf_i64 rbase = (nf_i64)row * nblk * Q8_0_BLOCK_BYTES;
    float gs = 0.0f;
    float us = 0.0f;
    nf_mvq8_row_partial_dual(gate_weight + rbase, up_weight + rbase, input, nblk, &gs, &us);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (get_local_id(0) == 0) {
        output[row] = nf_gelu_tanh(gate_val) * up_val;
    }
}

/* logits[row] = dot(weight[row, :], hidden). */
NF_REQD_SG_ATTR NF_KERNEL void LLM_Q8_0_ComputeLogits(
    const int vocab_size,
    const int embed_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL float* NF_RESTRICT logits)
{
    const int row = get_group_id(0);
    if (row >= vocab_size) return;

    const int nblk = embed_dim / Q8_0_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvq8_row_partial(weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES, hidden, nblk));
    if (get_local_id(0) == 0) logits[row] = acc;
}

/* output[i] = weight[token_id, i]; global = embed_dim rounded up to the local size. */
NF_KERNEL void LLM_Q8_0_EmbeddingLookup(
    const int embed_dim,
    const int token_id,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int i = get_global_id(0);
    if (i >= embed_dim) return;

    const nf_i64 row_bytes = (nf_i64)(embed_dim / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
    output[i] = nf_mvq8_elem(weight + (nf_i64)token_id * row_bytes, i);
}

/* output[s * embed_dim + i] = weight[token_ids[s], i]; global = seq_len * embed_dim rounded up. */
NF_KERNEL void LLM_Batch_Q8_0_EmbeddingLookup(
    const int embed_dim,
    const int seq_len,
    NF_GLOBAL const int* NF_RESTRICT token_ids,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)seq_len * embed_dim) return;

    const int s = (int)(gid / embed_dim);
    const int i = (int)(gid - (nf_i64)s * embed_dim);
    const nf_i64 row_bytes = (nf_i64)(embed_dim / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
    output[gid] = nf_mvq8_elem(weight + (nf_i64)token_ids[s] * row_bytes, i);
}

/*
 * Quantizes input[0..size) (size a multiple of 32) into Q8_0 blocks, as ggml quantize_row_q8_0_ref
 * (ggml-quants.c:276-298): d = amax / 127, id = d != 0 ? 1 / d : 0, qs[j] = roundf(x[j] * id) with
 * halfway cases away from zero, d stored as f16 (round to nearest even).
 */
NF_KERNEL void LLM_QuantizeQ8(
    const int size,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL uchar* NF_RESTRICT output)
{
    const int stride = get_global_size(0);
    const int blocks = size / Q8_0_BLOCK_SIZE;

    for (int b = get_global_id(0); b < blocks; b += stride) {
        NF_GLOBAL const float* x = input + (nf_i64)b * Q8_0_BLOCK_SIZE;
        NF_GLOBAL uchar* y = output + (nf_i64)b * Q8_0_BLOCK_BYTES;

        float amax = 0.0f;
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j += 4) {
            const float4 v = fabs(vload4(0, x + j));
            amax = fmax(amax, fmax(fmax(v.x, v.y), fmax(v.z, v.w)));
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        vstore_half(d, 0, (NF_GLOBAL NF_HALF*)y);

        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++) {
            const int q = clamp((int)round(x[j] * id), -127, 127);
            y[2 + j] = (uchar)(q & 0xFF);
        }
    }
}

/* The Q8_1-activation family forms of common/matvec_family.h: Q8_0_MatVec_Q8, Q8_0_MatVecN_Q8,
 * the counted Q8_0_MatVecN{2..16}_Q8 and Q8_0_MatVec2N{2..16}_Q8, Q8_0_MatVec2_Q8,
 * Q8_0_MatVecAdd_Q8, Q8_0_MatVecFusedGateUp_Q8 and Q8_0_MatVec2FusedGateUp_Q8, through
 * Q8_0_decode_q8 / Q8_0_decode_q8_dual. IQ_MVQN_NC_HIGH adds the counts above the dispatched
 * range, keeping Q8_0's measured 2..16 coverage. */
IQ_ALL_KERNELS_Q8(Q8_0, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_THREADS_PER_BLOCK)
IQ_MVQN_NC_HIGH(Q8_0, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_THREADS_PER_BLOCK)

/* Q8_0_MatVec over n_tokens consecutive float rows in one launch (n_tokens in
 * [1, NF_MVQN_MAX]): the block loop streams the weight row once and dots every block against
 * every live token's float block, where one Q8_0_MatVec per token re-reads the row per token.
 * Token t's dot lands at output[t * out_dim + row], and NF_MVQTN_TREE reduces the group in one
 * pass, so token t's result is the exact single-row result. */
NF_REQD_SG_ATTR NF_KERNEL void Q8_0_MatVecN(
    const int out_dim,
    const int in_dim,
    const int n_tokens,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / Q8_0_THREADS_PER_BLOCK;
    const int sub = lid % Q8_0_THREADS_PER_BLOCK;
    float ps[NF_MVQN_MAX];
#pragma unroll
    for (int j = 0; j < NF_MVQN_MAX; j++) ps[j] = 0.0f;
    for (int b = lid / Q8_0_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        NF_GLOBAL const uchar* wp = weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES
                                  + (nf_i64)b * Q8_0_BLOCK_BYTES;
#pragma unroll
        for (int j = 0; j < NF_MVQN_MAX; j++) {
            if (j < n_tokens) {
                ps[j] += nf_mvq8_slice_dot(wp, input + (nf_i64)j * in_dim + b * Q8_0_BLOCK_SIZE, sub);
            }
        }
    }
    NF_LOCAL float ts[NF_MVQN_MAX * NF_MVQTN_SLOTS];
    NF_MVQTN_TREE(ts, ps, n_tokens, output, out_dim, row)
}

/* output[row] = gelu(gate) * up with gate/up = dot(gate/up_weight[row, :], dequant(input)) and a
 * Q8_1-quantized input vector. */
NF_REQD_SG_ATTR NF_KERNEL void Q8_0_MatVecFusedGateUpGelu_Q8(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight,
    NF_GLOBAL const uchar* NF_RESTRICT up_weight,
    NF_GLOBAL const uchar* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    const nf_i64 rbase = (nf_i64)row * nblk * Q8_0_BLOCK_BYTES;
    float gs = 0.0f;
    float us = 0.0f;
    nf_mvq8q8_row_partial_dual(gate_weight + rbase, up_weight + rbase, input, nblk, &gs, &us);

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (get_local_id(0) == 0) {
        output[row] = nf_gelu_tanh(gate_val) * up_val;
    }
}

/* output[s * out_dim + row] = dot(weight[row, :], input[s, :]) for s in [0, seq_len); one
 * workgroup per row walks every token, each work-item taking whole blocks. */
NF_REQD_SG_ATTR NF_KERNEL void LLM_Batch_Q8_0_MatMul(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES;
    NF_LOCAL float scratch[LLM_DIMX];
    for (int s = 0; s < seq_len; s++) {
        const float acc = nf_wg_sum(scratch,
            nf_mvq8_row_partial_whole(wrow, input + (nf_i64)s * in_dim, nblk));
        if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
    }
}

/* LLM_Batch_Q8_0_MatMul with one (row, token) pair per workgroup. */
NF_REQD_SG_ATTR NF_KERNEL void LLM_Batch_Q8_0_MatMul_2D(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    const int s = get_global_id(1);
    if (row >= out_dim || s >= seq_len) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvq8_row_partial_whole(weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES,
                                  input + (nf_i64)s * in_dim, nblk));
    if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
}
