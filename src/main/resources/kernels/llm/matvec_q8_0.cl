/* matvec_q8_0.cl - Q8_0 matvec, fused gate/up, logits, batch matmul, Q8 activation quantization, Q8 x Q8 matvec and embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h */
/*
 * Q8_0 block (ggml block_q8_0): d (f16) at byte 0, qs[32] (int8) at byte 2; element e = qs[e] * d.
 *
 * Launch contract of the row kernels: one workgroup per output row, global = rows * L, local = L
 * (L a power of two <= LLM_DIMX and a multiple of Q8_0_THREADS_PER_BLOCK; CUDA: a multiple of 32);
 * the _2D form uses global = {out_dim * L, seq_len}, local = {L, 1}. Except LLM_Batch_Q8_0_MatMul(_2D),
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

/* Integer dot product of slice `sub` of two Q8_0 blocks (weight at wp, activation at xp). */
NF_INLINE int nf_mvq8_slice_idot(NF_GLOBAL const uchar* wp, NF_GLOBAL const uchar* xp, const int sub) {
    const int e = 2 + sub * Q8_0_ELEMENTS_PER_THREAD;
    NF_GLOBAL const nf_i8* w = (NF_GLOBAL const nf_i8*)(wp + e);
    NF_GLOBAL const nf_i8* q = (NF_GLOBAL const nf_i8*)(xp + e);
    int isum = 0;
#pragma unroll
    for (int k = 0; k < Q8_0_ELEMENTS_PER_THREAD; k++) {
        isum += (int)w[k] * (int)q[k];
    }
    return isum;
}

/* This work-item's share of dot(Q8_0 weight row at wrow, Q8_0 activation row at xq). */
NF_INLINE float nf_mvq8q8_row_partial(NF_GLOBAL const uchar* NF_RESTRICT wrow,
                                      NF_GLOBAL const uchar* NF_RESTRICT xq, const int nblk) {
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / Q8_0_THREADS_PER_BLOCK;
    const int sub = lid % Q8_0_THREADS_PER_BLOCK;
    float ps = 0.0f;
    for (int b = lid / Q8_0_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        NF_GLOBAL const uchar* wp = wrow + b * Q8_0_BLOCK_BYTES;
        NF_GLOBAL const uchar* xp = xq + b * Q8_0_BLOCK_BYTES;
        ps += (float)nf_mvq8_slice_idot(wp, xp, sub) * q8_0_scale(wp) * q8_0_scale(xp);
    }
    return ps;
}

/* Element i of the Q8_0 row at rbase; a non-finite block scale (corrupt or padding block) decodes as 0. */
NF_INLINE float nf_mvq8_elem(NF_GLOBAL const uchar* NF_RESTRICT rbase, const int i) {
    NF_GLOBAL const uchar* bp = rbase + (i / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
    float scale = q8_0_scale(bp);
    if (!isfinite(scale)) scale = 0.0f;
    return (float)((nf_i8)bp[2 + i % Q8_0_BLOCK_SIZE]) * scale;
}

/* output[row] = dot(weight[row, :], input). */
NF_KERNEL void Q8_0_MatVec(
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
NF_KERNEL void Q8_0_MatVecAdd(
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
NF_KERNEL void Q8_0_MatVecFusedGateUp(
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
NF_KERNEL void Q8_0_MatVecFusedGateUpGelu(
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
NF_KERNEL void LLM_Q8_0_ComputeLogits(
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

/* output[row] = dot(weight[row, :], dequant(input)) with a Q8_0-quantized input vector (integer
 * block products scaled by both block scales). */
NF_KERNEL void Q8_0_MatVec_Q8(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const uchar* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvq8q8_row_partial(weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES, input, nblk));
    if (get_local_id(0) == 0) output[row] = acc;
}

/* addend_output[row] += dot(weight[row, :], dequant(input)) with a Q8_0-quantized input vector. */
NF_KERNEL void Q8_0_MatVecAdd_Q8(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const uchar* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT addend_output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch,
        nf_mvq8q8_row_partial(weight + (nf_i64)row * nblk * Q8_0_BLOCK_BYTES, input, nblk));
    if (get_local_id(0) == 0) addend_output[row] += acc;
}

/* output[row] = silu(gate) * up with gate/up = dot(gate/up_weight[row, :], dequant(input)) and a
 * Q8_0-quantized input vector. */
NF_KERNEL void Q8_0_MatVecFusedGateUp_Q8(
    const int out_dim,
    const int in_dim,
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight,
    NF_GLOBAL const uchar* NF_RESTRICT up_weight,
    NF_GLOBAL const uchar* NF_RESTRICT input,
    NF_GLOBAL float* NF_RESTRICT output)
{
    const int row = get_group_id(0);
    if (row >= out_dim) return;

    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / Q8_0_THREADS_PER_BLOCK;
    const int sub = lid % Q8_0_THREADS_PER_BLOCK;
    const int nblk = in_dim / Q8_0_BLOCK_SIZE;
    const nf_i64 rbase = (nf_i64)row * nblk * Q8_0_BLOCK_BYTES;
    NF_GLOBAL const uchar* grow = gate_weight + rbase;
    NF_GLOBAL const uchar* urow = up_weight + rbase;

    float gs = 0.0f;
    float us = 0.0f;
    for (int b = lid / Q8_0_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        NF_GLOBAL const uchar* xp = input + b * Q8_0_BLOCK_BYTES;
        NF_GLOBAL const uchar* gp = grow + b * Q8_0_BLOCK_BYTES;
        NF_GLOBAL const uchar* up = urow + b * Q8_0_BLOCK_BYTES;
        const float x_scale = q8_0_scale(xp);
        gs += (float)nf_mvq8_slice_idot(gp, xp, sub) * q8_0_scale(gp) * x_scale;
        us += (float)nf_mvq8_slice_idot(up, xp, sub) * q8_0_scale(up) * x_scale;
    }

    NF_LOCAL float scratch[LLM_DIMX];
    const float gate_val = nf_wg_sum(scratch, gs);
    const float up_val = nf_wg_sum(scratch, us);
    if (lid == 0) {
        output[row] = nf_silu(gate_val) * up_val;
    }
}

/* output[s * out_dim + row] = dot(weight[row, :], input[s, :]) for s in [0, seq_len); one
 * workgroup per row walks every token, each work-item taking whole blocks. */
NF_KERNEL void LLM_Batch_Q8_0_MatMul(
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
NF_KERNEL void LLM_Batch_Q8_0_MatMul_2D(
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
