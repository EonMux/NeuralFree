/* matvec_mxfp4.cl - MXFP4 matvec, fused gate/up and embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h */
/*
 * MXFP4 block (ggml block_mxfp4): e (E8M0 exponent) at byte 0, qs[16] at byte 1. Element j < 16 is
 * the low nibble of qs[j], element j + 16 the high nibble; value = MXFP4_KVALUES[nibble] *
 * mxfp4_scale(e), the doubled e2m1 table paired with the half scale (ggml kvalues_mxfp4 and
 * ggml_e8m0_to_fp32_half, ggml-impl.h:489).
 *
 * Launch contract of the row kernels: one workgroup per output row, global = rows * L, local = L
 * (L a power of two <= LLM_DIMX and a multiple of MXFP4_THREADS_PER_BLOCK; CUDA: a multiple of 32).
 * MXFP4_THREADS_PER_BLOCK adjacent work-items share one block; work-item `sub` takes qs bytes
 * 4*sub .. 4*sub+3, i.e. elements 4*sub .. 4*sub+3 and their +16 partners. The row return is uniform
 * per workgroup. Weight rows are addressed in nf_i64. Embedding lookups are 1-D element grids.
 */

/* Slice `sub` of the dot product of the MXFP4 block at bp with x[0..32), scaled by the block scale. */
NF_INLINE float nf_mvfp4_slice_dot(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* x, const int sub) {
    const int k0 = sub * MXFP4_BYTES_PER_THREAD;
    float s = 0.0f;
    for (int j = 0; j < MXFP4_BYTES_PER_THREAD; j++) {
        const uint packed = bp[1 + k0 + j];
        s += x[k0 + j] * MXFP4_KVALUES[packed & 0x0Fu];
        s += x[k0 + j + 16] * MXFP4_KVALUES[packed >> 4];
    }
    return s * mxfp4_scale(bp[0]);
}

/* This work-item's share of dot(MXFP4 row at wrow, x[0..nblk*32)). */
NF_INLINE float nf_mvfp4_row_partial(NF_GLOBAL const uchar* NF_RESTRICT wrow,
                                     NF_GLOBAL const float* NF_RESTRICT x, const int nblk) {
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / MXFP4_THREADS_PER_BLOCK;
    const int sub = lid % MXFP4_THREADS_PER_BLOCK;
    float ps = 0.0f;
    for (int b = lid / MXFP4_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        ps += nf_mvfp4_slice_dot(wrow + b * MXFP4_BLOCK_BYTES, x + b * MXFP4_BLOCK_SIZE, sub);
    }
    return ps;
}

/* output[row] = dot(weight[row, :], input). */
NF_KERNEL void MXFP4_MatVec(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / MXFP4_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvfp4_row_partial(weight + (nf_i64)row * nblk * MXFP4_BLOCK_BYTES, input, nblk));
    if (get_local_id(0) == 0) output[row] = acc;
}

/* output[row] += dot(weight[row, :], input). */
NF_KERNEL void MXFP4_MatVecAdd(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / MXFP4_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvfp4_row_partial(weight + (nf_i64)row * nblk * MXFP4_BLOCK_BYTES, input, nblk));
    if (get_local_id(0) == 0) output[row] += acc;
}

/* output[row] = silu(dot(gate_weight[row, :], input)) * dot(up_weight[row, :], input). */
NF_KERNEL void MXFP4_MatVecFusedGateUp(
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
    const int gpi = get_local_size(0) / MXFP4_THREADS_PER_BLOCK;
    const int sub = lid % MXFP4_THREADS_PER_BLOCK;
    const int nblk = in_dim / MXFP4_BLOCK_SIZE;
    const nf_i64 rbase = (nf_i64)row * nblk * MXFP4_BLOCK_BYTES;
    NF_GLOBAL const uchar* grow = gate_weight + rbase;
    NF_GLOBAL const uchar* urow = up_weight + rbase;
    const int k0 = sub * MXFP4_BYTES_PER_THREAD;

    float gs = 0.0f;
    float us = 0.0f;
    for (int b = lid / MXFP4_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        NF_GLOBAL const uchar* gp = grow + b * MXFP4_BLOCK_BYTES;
        NF_GLOBAL const uchar* up = urow + b * MXFP4_BLOCK_BYTES;
        NF_GLOBAL const float* xb = input + b * MXFP4_BLOCK_SIZE;
        float gb = 0.0f;
        float ub = 0.0f;
        for (int j = 0; j < MXFP4_BYTES_PER_THREAD; j++) {
            const float x_lo = xb[k0 + j];
            const float x_hi = xb[k0 + j + 16];
            const uint g_packed = gp[1 + k0 + j];
            const uint u_packed = up[1 + k0 + j];
            gb += x_lo * MXFP4_KVALUES[g_packed & 0x0Fu] + x_hi * MXFP4_KVALUES[g_packed >> 4];
            ub += x_lo * MXFP4_KVALUES[u_packed & 0x0Fu] + x_hi * MXFP4_KVALUES[u_packed >> 4];
        }
        gs += gb * mxfp4_scale(gp[0]);
        us += ub * mxfp4_scale(up[0]);
    }

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (lid == 0) {
        output[row] = nf_silu(gate_val) * up_val;
    }
}

/* output[i] = weight[token_id, i]; global = embed_dim rounded up to the local size. */
NF_KERNEL void LLM_MXFP4_EmbeddingLookup(
    const int embed_dim,
    const int token_id,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int i = get_global_id(0);
    if (i >= embed_dim) return;

    const nf_i64 row_bytes = (nf_i64)(embed_dim / MXFP4_BLOCK_SIZE) * MXFP4_BLOCK_BYTES;
    NF_GLOBAL const uchar* bp = weight + (nf_i64)token_id * row_bytes + (i / MXFP4_BLOCK_SIZE) * MXFP4_BLOCK_BYTES;
    const int within = i % MXFP4_BLOCK_SIZE;
    const uint nibble = within < 16 ? (bp[1 + within] & 0x0Fu) : ((uint)bp[1 + within - 16] >> 4);
    output[i] = mxfp4_scale(bp[0]) * MXFP4_KVALUES[nibble];
}
