/* cert_argmax.cl - certified exact-argmax LM head: the INT8 screen and the candidate rescore.
 * Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h, sampling.cl, matvec_f16_bf16.cl, matvec_q8_0.cl */
/*
 * The certified head decides greedy tokens in two passes. The screen (LLM_CertScreen) reads each
 * stored row once as per-row absmax-scaled int8 codes and writes the approximate logits; the rescore
 * (LLM_CertRescore_*) recomputes the threshold from the per-row error bounds, gathers the candidates
 * that can still reach it, and dots exactly those rows through the same arithmetic the {T}_MatVec
 * family uses - the distinct-name helpers below repeat the family's partial sums word for word, so a
 * rescored value equals the value the plain matvec of the same row would produce at the same
 * work-group size. A row whose candidate count exceeds the cap (or holds no candidate at all) falls
 * back to the full exact scan inside its own workgroup, so the decision stays on the device.
 *
 * The row dot helpers and the candidate order are exact-copy reasoning only; the certification itself
 * (|exact - screen| <= hidden_norm * bound) holds for every accumulation order of both sides, since
 * the bound is a Cauchy-Schwarz norm product. Ties resolve to the lower token id through the same
 * total order the LLM_Argmax reduction uses (nf_cert_better repeats nf_argmax_better), and the
 * compaction order of the atomic gather therefore cannot change the result.
 *
 * Launch contract: local size L a power of two <= LLM_DIMX and a multiple of Q8_0_THREADS_PER_BLOCK
 * (the host launches both kernels at the matvec work-group size); CUDA: a multiple of 32. The early
 * returns depend on the group id alone. Rows and token rows are addressed in nf_i64. CERT_KMAX caps
 * the gather's local storage; a configured cap above it is clamped by the host (clamping only forces
 * more full-scan fallbacks, which stay exact).
 */
#ifndef CERT_KMAX
#define CERT_KMAX 256
#endif

/* Relative tie slack: one ulp of the f32 logit format (CertifiedArgmaxScreen.TIE_RELATIVE_ULP). */
#define NF_CERT_TIE_REL 0x1p-23f

/* Index sentinel of an argmax candidate that holds no element (nf_argmax_better's order). */
#define NF_CERT_NONE 0x7fffffff

/*
 * Total order of argmax candidates (v, i), as nf_argmax_better (kernels/llm/sampling.cl): a non-NaN
 * value beats NaN; among non-NaN values the larger wins; equal values (and two NaNs) go to the lower
 * index. A copy under a distinct name because the CUDA units compile this file without sampling.cl.
 */
NF_INLINE int nf_cert_better(float v, int i, float bv, int bi) {
    if (isnan(v)) {
        return isnan(bv) && i < bi;
    }
    if (isnan(bv)) {
        return 1;
    }
    return v > bv || (v == bv && i < bi);
}

/* This work-item's share of dot(w[0..n), x[0..n)) for F16 w; nf_mvf16_partial
 * (kernels/llm/matvec_f16_bf16.cl) word for word, so the rescore produces the family's values. */
NF_INLINE float nf_cert_f16_partial(NF_GLOBAL const NF_HALF* NF_RESTRICT w,
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

/* nf_cert_f16_partial for BF16 w; nf_mvbf16_partial word for word. */
NF_INLINE float nf_cert_bf16_partial(NF_GLOBAL const ushort* NF_RESTRICT w,
                                     NF_GLOBAL const float* NF_RESTRICT x, const int n) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = n & ~3;
    float ps = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        const ushort4 r = vload4(0, w + j);
        const float4 wv = NF_F4(bf16_to_f32(r.x), bf16_to_f32(r.y), bf16_to_f32(r.z), bf16_to_f32(r.w));
        ps += dot(wv, vload4(0, x + j));
    }
    for (int j = n4 + lid; j < n; j += lsize) {
        ps += bf16_to_f32(w[j]) * x[j];
    }
    return ps;
}

/* Q8_0_ELEMENTS_PER_THREAD-element slice `sub` of the dot of the Q8_0 block at bp with x[0..32);
 * nf_mvq8_slice_dot (kernels/llm/matvec_q8_0.cl) word for word. */
NF_INLINE float nf_cert_q8_slice_dot(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* x, const int sub) {
    const int e = sub * Q8_0_ELEMENTS_PER_THREAD;
    NF_GLOBAL const nf_i8* q = (NF_GLOBAL const nf_i8*)(bp + 2 + e);
    const float s = dot(vload4(0, x + e), convert_float4(vload4(0, q)))
                  + dot(vload4(0, x + e + 4), convert_float4(vload4(0, q + 4)));
    return s * q8_0_scale(bp);
}

/* This work-item's share of dot(Q8_0 row at wrow, x[0..nblk*32)) under the shared-block mapping;
 * nf_mvq8_row_partial word for word. */
NF_INLINE float nf_cert_q8_row_partial(NF_GLOBAL const uchar* NF_RESTRICT wrow,
                                       NF_GLOBAL const float* NF_RESTRICT x, const int nblk) {
    const int lid = get_local_id(0);
    const int gpi = get_local_size(0) / Q8_0_THREADS_PER_BLOCK;
    const int sub = lid % Q8_0_THREADS_PER_BLOCK;
    float ps = 0.0f;
    for (int b = lid / Q8_0_THREADS_PER_BLOCK; b < nblk; b += gpi) {
        ps += nf_cert_q8_slice_dot(wrow + b * Q8_0_BLOCK_BYTES, x + b * Q8_0_BLOCK_SIZE, sub);
    }
    return ps;
}

/*
 * screen_out[row * vocab + v] = scales[v] * dot(codes[v, :], hidden[row, :]), the int8 screen logits
 * of the certified head: one workgroup per (row, token), global = n_rows * vocab rounded up to L.
 * codes holds vocab rows of cols rounded int8 codes, scales the per-row absmax scales.
 */
NF_KERNEL void LLM_CertScreen(
    const int vocab,
    const int cols,
    const int n_rows,
    NF_GLOBAL const nf_i8* NF_RESTRICT codes,
    NF_GLOBAL const float* NF_RESTRICT scales,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL float* NF_RESTRICT screen_out)
{
    const nf_i64 gid = (nf_i64)get_group_id(0);
    const int row = (int)(gid / vocab);
    const int v = (int)(gid - (nf_i64)row * vocab);
    if (row >= n_rows) return;

    NF_GLOBAL const nf_i8* q = codes + (nf_i64)v * cols;
    NF_GLOBAL const float* x = hidden + (nf_i64)row * cols;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int n4 = cols & ~3;
    float ps = 0.0f;
    for (int j = lid * 4; j < n4; j += lsize * 4) {
        ps += dot(vload4(0, x + j), convert_float4(vload4(0, q + j)));
    }
    for (int j = n4 + lid; j < cols; j += lsize) {
        ps += x[j] * (float)q[j];
    }

    NF_LOCAL float scratch[LLM_DIMX];
    const float acc = nf_wg_sum(scratch, ps);
    if (get_local_id(0) == 0) screen_out[gid] = scales[v] * acc;
}

/*
 * The certified greedy choice of hidden row `row`: per row the threshold
 * thr = max_v(screen - bound * |h|) - tie, the candidates {v : screen + bound * |h| >= thr} compacted
 * into local memory, and the exact argmax over their rescored logits; a row with more candidates
 * than `kmax` (or none) takes the full exact scan of every vocab row instead. ids[row] is the first
 * maximum of the exact logits, NaN never selected, as LLM_Argmax. Dispatch: global = n_rows rounded
 * up to L, local = L; one workgroup per row. `weight` holds the stored rows of the rescore type, in
 * the layout the matching {T}_MatVec family kernel reads (F16/BF16: cols values per row, Q8_0:
 * cols/32 blocks of Q8_0_BLOCK_BYTES).
 */
NF_KERNEL void LLM_CertRescore_F16(
    const int vocab,
    const int cols,
    const int n_rows,
    const int kmax,
    const float ebmax,
    NF_GLOBAL const NF_HALF* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT bounds,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL const float* NF_RESTRICT screened,
    NF_GLOBAL int* NF_RESTRICT ids)
{
    const int row = get_group_id(0);
    if (row >= n_rows) return;

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_GLOBAL const float* x = hidden + (nf_i64)row * cols;
    NF_GLOBAL const float* s = screened + (nf_i64)row * vocab;

    float hs = 0.0f;
    for (int i = lid; i < cols; i += lsize) {
        const float v = x[i];
        hs += v * v;
    }
    NF_LOCAL float scratch[LLM_DIMX];
    const float hn = sqrt(nf_wg_sum(scratch, hs));

    float my = -NF_INF;
    for (int v = lid; v < vocab; v += lsize) {
        my = fmax(my, s[v] - bounds[v] * hn);
    }
    const float lower = nf_wg_max(scratch, my);
    const float thr = lower - ((fabs(lower) + 2.0f * hn * ebmax) * NF_CERT_TIE_REL + 1.0e-30f);

    NF_LOCAL int cand[CERT_KMAX];
    NF_LOCAL int cnt_buf[1];
    if (lid == 0) cnt_buf[0] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int v = lid; v < vocab; v += lsize) {
        if (s[v] + bounds[v] * hn >= thr) {
            const int slot = atomic_add(&cnt_buf[0], 1);
            if (slot < kmax) {
                cand[slot] = v;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int cnt = min(cnt_buf[0], kmax);

    float bv = NF_NAN;
    int bi = NF_CERT_NONE;
    if (cnt == 0 || cnt < cnt_buf[0]) {
        for (int v = 0; v < vocab; v++) {
            const float val = nf_wg_sum(scratch,
                nf_cert_f16_partial(weight + (nf_i64)v * cols, x, cols));
            if (nf_cert_better(val, v, bv, bi)) {
                bv = val;
                bi = v;
            }
        }
    } else {
        for (int j = 0; j < cnt; j++) {
            const int v = cand[j];
            const float val = nf_wg_sum(scratch,
                nf_cert_f16_partial(weight + (nf_i64)v * cols, x, cols));
            if (nf_cert_better(val, v, bv, bi)) {
                bv = val;
                bi = v;
            }
        }
    }

    NF_LOCAL float best_v[LLM_DIMX];
    NF_LOCAL int best_i[LLM_DIMX];
    best_v[lid] = bv;
    best_i[lid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            const float ov = best_v[lid + stride];
            const int oi = best_i[lid + stride];
            if (nf_cert_better(ov, oi, best_v[lid], best_i[lid])) {
                best_v[lid] = ov;
                best_i[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const int r = best_i[0];
        ids[row] = (r == NF_CERT_NONE) ? 0 : r;
    }
}

/* LLM_CertRescore_F16 for a BF16 head weight. */
NF_KERNEL void LLM_CertRescore_BF16(
    const int vocab,
    const int cols,
    const int n_rows,
    const int kmax,
    const float ebmax,
    NF_GLOBAL const ushort* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT bounds,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL const float* NF_RESTRICT screened,
    NF_GLOBAL int* NF_RESTRICT ids)
{
    const int row = get_group_id(0);
    if (row >= n_rows) return;

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_GLOBAL const float* x = hidden + (nf_i64)row * cols;
    NF_GLOBAL const float* s = screened + (nf_i64)row * vocab;

    float hs = 0.0f;
    for (int i = lid; i < cols; i += lsize) {
        const float v = x[i];
        hs += v * v;
    }
    NF_LOCAL float scratch[LLM_DIMX];
    const float hn = sqrt(nf_wg_sum(scratch, hs));

    float my = -NF_INF;
    for (int v = lid; v < vocab; v += lsize) {
        my = fmax(my, s[v] - bounds[v] * hn);
    }
    const float lower = nf_wg_max(scratch, my);
    const float thr = lower - ((fabs(lower) + 2.0f * hn * ebmax) * NF_CERT_TIE_REL + 1.0e-30f);

    NF_LOCAL int cand[CERT_KMAX];
    NF_LOCAL int cnt_buf[1];
    if (lid == 0) cnt_buf[0] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int v = lid; v < vocab; v += lsize) {
        if (s[v] + bounds[v] * hn >= thr) {
            const int slot = atomic_add(&cnt_buf[0], 1);
            if (slot < kmax) {
                cand[slot] = v;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int cnt = min(cnt_buf[0], kmax);

    float bv = NF_NAN;
    int bi = NF_CERT_NONE;
    if (cnt == 0 || cnt < cnt_buf[0]) {
        for (int v = 0; v < vocab; v++) {
            const float val = nf_wg_sum(scratch,
                nf_cert_bf16_partial(weight + (nf_i64)v * cols, x, cols));
            if (nf_cert_better(val, v, bv, bi)) {
                bv = val;
                bi = v;
            }
        }
    } else {
        for (int j = 0; j < cnt; j++) {
            const int v = cand[j];
            const float val = nf_wg_sum(scratch,
                nf_cert_bf16_partial(weight + (nf_i64)v * cols, x, cols));
            if (nf_cert_better(val, v, bv, bi)) {
                bv = val;
                bi = v;
            }
        }
    }

    NF_LOCAL float best_v[LLM_DIMX];
    NF_LOCAL int best_i[LLM_DIMX];
    best_v[lid] = bv;
    best_i[lid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            const float ov = best_v[lid + stride];
            const int oi = best_i[lid + stride];
            if (nf_cert_better(ov, oi, best_v[lid], best_i[lid])) {
                best_v[lid] = ov;
                best_i[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const int r = best_i[0];
        ids[row] = (r == NF_CERT_NONE) ? 0 : r;
    }
}

/* LLM_CertRescore_F16 for a Q8_0 head weight (cols a multiple of Q8_0_BLOCK_SIZE). */
NF_KERNEL void LLM_CertRescore_Q8_0(
    const int vocab,
    const int cols,
    const int n_rows,
    const int kmax,
    const float ebmax,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT bounds,
    NF_GLOBAL const float* NF_RESTRICT hidden,
    NF_GLOBAL const float* NF_RESTRICT screened,
    NF_GLOBAL int* NF_RESTRICT ids)
{
    const int row = get_group_id(0);
    if (row >= n_rows) return;

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int nblk = cols / Q8_0_BLOCK_SIZE;
    NF_GLOBAL const float* x = hidden + (nf_i64)row * cols;
    NF_GLOBAL const float* s = screened + (nf_i64)row * vocab;

    float hs = 0.0f;
    for (int i = lid; i < cols; i += lsize) {
        const float v = x[i];
        hs += v * v;
    }
    NF_LOCAL float scratch[LLM_DIMX];
    const float hn = sqrt(nf_wg_sum(scratch, hs));

    float my = -NF_INF;
    for (int v = lid; v < vocab; v += lsize) {
        my = fmax(my, s[v] - bounds[v] * hn);
    }
    const float lower = nf_wg_max(scratch, my);
    const float thr = lower - ((fabs(lower) + 2.0f * hn * ebmax) * NF_CERT_TIE_REL + 1.0e-30f);

    NF_LOCAL int cand[CERT_KMAX];
    NF_LOCAL int cnt_buf[1];
    if (lid == 0) cnt_buf[0] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int v = lid; v < vocab; v += lsize) {
        if (s[v] + bounds[v] * hn >= thr) {
            const int slot = atomic_add(&cnt_buf[0], 1);
            if (slot < kmax) {
                cand[slot] = v;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int cnt = min(cnt_buf[0], kmax);

    float bv = NF_NAN;
    int bi = NF_CERT_NONE;
    if (cnt == 0 || cnt < cnt_buf[0]) {
        for (int v = 0; v < vocab; v++) {
            const float val = nf_wg_sum(scratch,
                nf_cert_q8_row_partial(weight + (nf_i64)v * nblk * Q8_0_BLOCK_BYTES, x, nblk));
            if (nf_cert_better(val, v, bv, bi)) {
                bv = val;
                bi = v;
            }
        }
    } else {
        for (int j = 0; j < cnt; j++) {
            const int v = cand[j];
            const float val = nf_wg_sum(scratch,
                nf_cert_q8_row_partial(weight + (nf_i64)v * nblk * Q8_0_BLOCK_BYTES, x, nblk));
            if (nf_cert_better(val, v, bv, bi)) {
                bv = val;
                bi = v;
            }
        }
    }

    NF_LOCAL float best_v[LLM_DIMX];
    NF_LOCAL int best_i[LLM_DIMX];
    best_v[lid] = bv;
    best_i[lid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            const float ov = best_v[lid + stride];
            const int oi = best_i[lid + stride];
            if (nf_cert_better(ov, oi, best_v[lid], best_i[lid])) {
                best_v[lid] = ov;
                best_i[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const int r = best_i[0];
        ids[row] = (r == NF_CERT_NONE) ? 0 : r;
    }
}
