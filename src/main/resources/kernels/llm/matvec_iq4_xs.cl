/* matvec_iq4_xs.cl - IQ4_XS matvec, fused gate/up, logits, batch matmul and embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h */
/*
 * IQ4_XS super-block (ggml block_iq4_xs, 256 elements, 136 bytes): d (f16) at 0, scales_h (u16) at 2,
 * scales_l[4] at 4, qs[128] at 8. Sub-block sb (32 elements) has scale iq4xs_sub_scale(bp, sb) =
 * d * (ls - 32) and quant bytes qs[16*sb .. 16*sb+15]; element j < 16 of the sub-block is the low
 * nibble of byte j, element j + 16 the high nibble; value = IQ4_NL_KVALUES[nibble] * scale
 * (ggml-quants.c dequantize_row_iq4_xs).
 *
 * Launch contract of the row kernels: one workgroup per output row, global = rows * L, local = L
 * (L a power of two <= LLM_DIMX and a multiple of IQ4_XS_THREADS_PER_SUPER = 32); the _2D form uses
 * global = {out_dim * L, seq_len}, local = {L, 1}. 32 adjacent work-items share one super-block:
 * work-item t takes sub-block t / 4 and its quant bytes 4*(t % 4) .. 4*(t % 4)+3. The row and token
 * returns are uniform per workgroup. Weight rows, token rows and batch rows are addressed in nf_i64.
 * Embedding lookups are 1-D element grids.
 */

/* Slice `t` (0..31) of the dot product of the IQ4_XS super-block at bp with x[0..256). */
NF_INLINE float nf_mviq4xs_slice_dot(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* x, const int t) {
    const int sb = t / 4;
    const int k0 = (t % 4) * 4;
    NF_GLOBAL const uchar* qs = bp + IQ4_XS_HEADER_BYTES + sb * IQ4_XS_SUB_BLOCK_QUANT_BYTES + k0;
    NF_GLOBAL const float* xs = x + sb * 32 + k0;
    float s = 0.0f;
    for (int j = 0; j < 4; j++) {
        const uint packed = qs[j];
        s += xs[j] * IQ4_NL_KVALUES[packed & 0x0Fu];
        s += xs[j + 16] * IQ4_NL_KVALUES[packed >> 4];
    }
    return s * iq4xs_sub_scale(bp, sb);
}

/* This work-item's share of dot(IQ4_XS row at wrow, x[0..nsb*256)). */
NF_INLINE float nf_mviq4xs_row_partial(NF_GLOBAL const uchar* NF_RESTRICT wrow,
                                       NF_GLOBAL const float* NF_RESTRICT x, const int nsb) {
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / IQ4_XS_THREADS_PER_SUPER;
    const int t = lid % IQ4_XS_THREADS_PER_SUPER;
    float ps = 0.0f;
    for (int b = lid / IQ4_XS_THREADS_PER_SUPER; b < nsb; b += gpi) {
        ps += nf_mviq4xs_slice_dot(wrow + b * IQ4_XS_BLOCK_BYTES, x + b * IQ4_XS_BLOCK_SIZE, t);
    }
    return ps;
}

/* Element i of the IQ4_XS row at rbase. */
NF_INLINE float nf_mviq4xs_elem(NF_GLOBAL const uchar* NF_RESTRICT rbase, const int i) {
    NF_GLOBAL const uchar* bp = rbase + (i / IQ4_XS_BLOCK_SIZE) * IQ4_XS_BLOCK_BYTES;
    const int within_super = i % IQ4_XS_BLOCK_SIZE;
    const int sb = within_super / 32;
    const int within_sub = within_super % 32;
    NF_GLOBAL const uchar* qs = bp + IQ4_XS_HEADER_BYTES + sb * IQ4_XS_SUB_BLOCK_QUANT_BYTES;
    const uint nibble = within_sub < 16 ? (qs[within_sub] & 0x0Fu) : ((uint)qs[within_sub - 16] >> 4);
    return iq4xs_sub_scale(bp, sb) * IQ4_NL_KVALUES[nibble];
}

/* output[row] = dot(weight[row, :], input). */
NF_KERNEL void IQ4_XS_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nsb = in_dim / IQ4_XS_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mviq4xs_row_partial(weight + (nf_i64)row * nsb * IQ4_XS_BLOCK_BYTES, input, nsb));
    if (get_local_id(0) == 0) output[row] = acc;
}

/* addend_output[row] += dot(weight[row, :], input). */
NF_KERNEL void IQ4_XS_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nsb = in_dim / IQ4_XS_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mviq4xs_row_partial(weight + (nf_i64)row * nsb * IQ4_XS_BLOCK_BYTES, input, nsb));
    if (get_local_id(0) == 0) addend_output[row] += acc;
}

/* output[row] = silu(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_KERNEL void IQ4_XS_MatVecFusedGateUp(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight,
    NF_GLOBAL const uchar* NF_RESTRICT up_weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / IQ4_XS_THREADS_PER_SUPER;
    const int t = lid % IQ4_XS_THREADS_PER_SUPER;
    const int sb = t / 4;
    const int k0 = (t % 4) * 4;
    const int nsb = in_dim / IQ4_XS_BLOCK_SIZE;
    const nf_i64 rbase = (nf_i64)row * nsb * IQ4_XS_BLOCK_BYTES;
    NF_GLOBAL const uchar* grow = gate_weight + rbase;
    NF_GLOBAL const uchar* urow = up_weight + rbase;
    const int qoff = IQ4_XS_HEADER_BYTES + sb * IQ4_XS_SUB_BLOCK_QUANT_BYTES + k0;

    float gs = 0.0f;
    float us = 0.0f;
    for (int b = lid / IQ4_XS_THREADS_PER_SUPER; b < nsb; b += gpi) {
        NF_GLOBAL const uchar* gbp = grow + b * IQ4_XS_BLOCK_BYTES;
        NF_GLOBAL const uchar* ubp = urow + b * IQ4_XS_BLOCK_BYTES;
        NF_GLOBAL const float* xs = input + b * IQ4_XS_BLOCK_SIZE + sb * 32 + k0;
        float gb = 0.0f;
        float ub = 0.0f;
        for (int j = 0; j < 4; j++) {
            const float x_lo = xs[j];
            const float x_hi = xs[j + 16];
            const uint g_packed = gbp[qoff + j];
            const uint u_packed = ubp[qoff + j];
            gb += x_lo * IQ4_NL_KVALUES[g_packed & 0x0Fu] + x_hi * IQ4_NL_KVALUES[g_packed >> 4];
            ub += x_lo * IQ4_NL_KVALUES[u_packed & 0x0Fu] + x_hi * IQ4_NL_KVALUES[u_packed >> 4];
        }
        gs += gb * iq4xs_sub_scale(gbp, sb);
        us += ub * iq4xs_sub_scale(ubp, sb);
    }

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (lid == 0) {
        output[row] = nf_silu(gate_val) * up_val;
    }
}

/* output[i] = weight[token_id, i]; global = embed_dim rounded up to the local size. */
NF_KERNEL void IQ4_XS_EmbeddingLookup(
    const int embed_dim,
    const int token_id,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int i = get_global_id(0);
    if (i >= embed_dim) return;

    const nf_i64 row_bytes = (nf_i64)(embed_dim / IQ4_XS_BLOCK_SIZE) * IQ4_XS_BLOCK_BYTES;
    output[i] = nf_mviq4xs_elem(weight + (nf_i64)token_id * row_bytes, i);
}

/* logits[row] = dot(weight[row, :], hidden). */
NF_KERNEL void IQ4_XS_ComputeLogits(
    const int vocab_size,
    const int embed_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL float* NF_RESTRICT logits)
{
    const int row = get_group_id(0);
    if (row >= vocab_size) return;

    const int nsb = embed_dim / IQ4_XS_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mviq4xs_row_partial(weight + (nf_i64)row * nsb * IQ4_XS_BLOCK_BYTES, hidden, nsb));
    if (get_local_id(0) == 0) logits[row] = acc;
}

/* output[s * embed_dim + i] = weight[token_ids[s], i]; global = seq_len * embed_dim rounded up. */
NF_KERNEL void IQ4_XS_BatchEmbeddingLookup(
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
    const nf_i64 row_bytes = (nf_i64)(embed_dim / IQ4_XS_BLOCK_SIZE) * IQ4_XS_BLOCK_BYTES;
    output[gid] = nf_mviq4xs_elem(weight + (nf_i64)token_ids[s] * row_bytes, i);
}

/* output[s * out_dim + row] = dot(weight[row, :], input[s, :]) for s in [0, seq_len); one
 * workgroup per row walks every token. */
NF_KERNEL void IQ4_XS_BatchMatMul(
    const int out_dim,
    const int in_dim,
    const int seq_len,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nsb = in_dim / IQ4_XS_BLOCK_SIZE;
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * nsb * IQ4_XS_BLOCK_BYTES;
    NF_LOCAL float scratch[LLM_DIMX];
    for (int s = 0; s < seq_len; s++) {
        const float acc = nf_wg_sum(scratch,
            nf_mviq4xs_row_partial(wrow, input + (nf_i64)s * in_dim, nsb));
        if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
    }
}

/* IQ4_XS_BatchMatMul with one (row, token) pair per workgroup. */
NF_KERNEL void IQ4_XS_BatchMatMul_2D(
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

    const int nsb = in_dim / IQ4_XS_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mviq4xs_row_partial(weight + (nf_i64)row * nsb * IQ4_XS_BLOCK_BYTES,
                               input + (nf_i64)s * in_dim, nsb));
    if (get_local_id(0) == 0) output[(nf_i64)s * out_dim + row] = acc;
}
