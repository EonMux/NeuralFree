/* matvec_sg.cl - subgroup-row integer-dot matvecs of Q8_0, Q4_0, Q4_K, Q5_K and Q6_K weights (OpenCL only). Requires: common/dialect.h, common/config.h, common/math.h, common/quant_blocks.h, common/q8_dot.h */
/*
 * {T}_MatVecSg_Q8, {T}_MatVecAddSg_Q8 and {T}_MatVecFusedGateUpSg_Q8 compute what {T}_MatVec_Q8,
 * {T}_MatVecAdd_Q8 and {T}_MatVecFusedGateUp_Q8 (common/matvec_family.h) compute, with the argument
 * lists of those forms, under a different work mapping built for GPUs whose loads are limited by the
 * number of load messages rather than by DRAM bandwidth (the Intel Xe integrated GPUs):
 *
 * - One 16-lane subgroup (NF_SG_ROW_SIZE, pinned by intel_reqd_sub_group_size) owns NF_SGR_ROWS
 *   consecutive output rows; a work-group holds NF_SGR_SUBGROUPS subgroups. Launch: local =
 *   NF_SGR_SUBGROUPS * NF_SG_ROW_SIZE, global = ceil(rows / (NF_SGR_SUBGROUPS * NF_SGR_ROWS)) * local.
 * - The input row is cut into 64-element units; lane l takes units l, l + 16, ... . Each unit's weight
 *   bytes are read with 16-byte vector loads (a Q8_0 block pair is 68 bytes, so the 4-byte-aligned pair
 *   is four uint4 loads and one uint; a Q4_0 pair is 36 bytes, two uint4 and one uint; a Q4_K quarter is
 *   its 16-byte header and two uint4 of codes, a Q5_K quarter the same plus the block's two uint4 of
 *   fifth bits; a Q6_K quarter is three 16-byte code runs), against one or two 2-byte loads per 4 codes
 *   in the one-work-group-per-row family.
 * - The unit's Q8_1 activation codes, scales and code sums are loaded once per lane and dotted with
 *   every row of the subgroup, and each row is reduced by one sub_group_reduce_add: no local memory and
 *   no barrier. Rows past the end repeat the last row's weights and drop their stores.
 *
 * Every product is the exact integer dot of the weight codes with the Q8_1 codes (NF_DP4A), scaled in
 * float as the family decoders do; only the float summation order differs from the family's. Results
 * are deterministic: the lane assignment and the subgroup reduction are fixed.
 *
 * Preconditions the host checks: in_dim is a multiple of 64 for Q8_0 and Q4_0 (so every row starts
 * 4-byte aligned and holds whole block pairs) and of 256 for the K-quants; weight buffers start 16-byte
 * aligned. The kernels exist only when the host defines NF_SG_ROW_SIZE (a subgroup size the device
 * accepts through cl_intel_required_subgroup_size) for a subgroup program.
 */
#if defined(USE_SUBGROUPS) && defined(cl_intel_required_subgroup_size) && defined(NF_SG_ROW_SIZE)

#define NF_SGR_ATTR __attribute__((intel_reqd_sub_group_size(NF_SG_ROW_SIZE)))
/* Output rows per subgroup and subgroups per work-group (the host mirrors both). */
#define NF_SGR_ROWS 2
#define NF_SGR_SUBGROUPS 4
/* Input elements per lane unit. */
#define NF_SGR_UNIT 64

/* The Q8_1 activation of one lane unit: up to four block scales, the exact code sums of the unit's part
 * of each block, and sixteen code words. */
typedef struct {
    float d[4];
    int s[4];
    uint c[16];
} nf_sgr_act;

/* Float value of the f16 bits in the low half of v. */
NF_INLINE float nf_sgr_half(uint v) {
    const ushort h = (ushort)(v & 0xFFFFu);
    return vload_half(0, (const NF_HALF*)&h);
}

/* Signed-byte sum of a code word. */
NF_INLINE int nf_sgr_sum4(uint w) {
    return NF_DP4A(NF_Q8_ONES, (int)w, 0);
}

/* Sixteen bytes at p (2-byte aligned) as four words: one aligned vector load when p is 4-byte aligned,
 * else the aligned vector below p and its next word funnel-shifted by 16 bits. */
NF_INLINE uint4 nf_sgr_load16(NF_GLOBAL const uchar* p) {
    const uint odd = (uint)((size_t)p & 2u);
    NF_GLOBAL const uint* a = (NF_GLOBAL const uint*)(p - odd);
    const uint4 v = vload4(0, a);
    if (odd == 0u) {
        return v;
    }
    const uint e = a[4];
    return (uint4)((v.x >> 16) | (v.y << 16), (v.y >> 16) | (v.z << 16),
                   (v.z >> 16) | (v.w << 16), (v.w >> 16) | (e << 16));
}

/* ---- Q8_0: unit u is the block pair 2u, 2u + 1 (68 bytes at u * 68 of the row). ---- */

/* Q8_1 blocks 2u and 2u + 1: scales d[0..1], codes c[0..7] and c[8..15]. */
NF_INLINE void Q8_0_sgr_act(NF_GLOBAL const uchar* xq, int u, nf_sgr_act* a) {
    NF_GLOBAL const uchar* b0 = xq + (nf_i64)(2 * u) * Q8_1_BLOCK_BYTES;
    NF_GLOBAL const uint* q0 = (NF_GLOBAL const uint*)(b0 + Q8_1_QS_OFFSET);
    NF_GLOBAL const uint* q1 = (NF_GLOBAL const uint*)(b0 + Q8_1_BLOCK_BYTES + Q8_1_QS_OFFSET);
    a->d[0] = vload_half(0, (NF_GLOBAL const NF_HALF*)b0);
    a->d[1] = vload_half(0, (NF_GLOBAL const NF_HALF*)(b0 + Q8_1_BLOCK_BYTES));
    const uint4 x0 = vload4(0, q0), x1 = vload4(1, q0), x2 = vload4(0, q1), x3 = vload4(1, q1);
    a->c[0] = x0.x; a->c[1] = x0.y; a->c[2] = x0.z; a->c[3] = x0.w;
    a->c[4] = x1.x; a->c[5] = x1.y; a->c[6] = x1.z; a->c[7] = x1.w;
    a->c[8] = x2.x; a->c[9] = x2.y; a->c[10] = x2.z; a->c[11] = x2.w;
    a->c[12] = x3.x; a->c[13] = x3.y; a->c[14] = x3.z; a->c[15] = x3.w;
}

/* Q8_0_sgr_act and the exact code sums s[0..1] of the two blocks, for the types with a bias or minimum term. */
NF_INLINE void nf_sgr_pair_act_sums(NF_GLOBAL const uchar* xq, int u, nf_sgr_act* a) {
    Q8_0_sgr_act(xq, u, a);
    int s0 = 0, s1 = 0;
    for (int i = 0; i < 8; i++) {
        s0 += nf_sgr_sum4(a->c[i]);
        s1 += nf_sgr_sum4(a->c[8 + i]);
    }
    a->s[0] = s0;
    a->s[1] = s1;
}

/* Pair words w0..w16: block 2u's d is the low half of w0 and its code word i is (w_i >> 16) | (w_{i+1} << 16);
 * block 2u + 1's d is the high half of w8 and its code words are w9..w16. */
NF_INLINE float Q8_0_sgr_dot(NF_GLOBAL const uchar* wrow, int u, const nf_sgr_act* a) {
    NF_GLOBAL const uint* p = (NF_GLOBAL const uint*)(wrow + (nf_i64)u * (2 * Q8_0_BLOCK_BYTES));
    const uint4 v0 = vload4(0, p), v1 = vload4(1, p), v2 = vload4(2, p), v3 = vload4(3, p);
    const uint v4 = p[16];
    const uint w[17] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w,
                        v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w, v4};
    int s0 = 0, s1 = 0;
    for (int i = 0; i < 8; i++) {
        s0 = NF_DP4A((int)((w[i] >> 16) | (w[i + 1] << 16)), (int)a->c[i], s0);
        s1 = NF_DP4A((int)w[9 + i], (int)a->c[8 + i], s1);
    }
    return a->d[0] * nf_sgr_half(w[0]) * (float)s0 + a->d[1] * nf_sgr_half(w[8] >> 16) * (float)s1;
}

/* ---- Q4_0: unit u is the block pair 2u, 2u + 1 (36 bytes at u * 36 of the row); y = d * (nibble - 8), the
 * low nibbles of code byte j holding element j and the high nibbles element j + 16. ---- */

/* Q8_1 blocks 2u and 2u + 1 with their code sums for the -8 bias term. */
NF_INLINE void Q4_0_sgr_act(NF_GLOBAL const uchar* xq, int u, nf_sgr_act* a) {
    nf_sgr_pair_act_sums(xq, u, a);
}

/* Pair words w0..w8: block 2u's d is the low half of w0 and its code word i is (w_i >> 16) | (w_{i+1} << 16);
 * block 2u + 1's d is the high half of w4 and its code words are w5..w8. */
NF_INLINE float Q4_0_sgr_dot(NF_GLOBAL const uchar* wrow, int u, const nf_sgr_act* a) {
    NF_GLOBAL const uint* p = (NF_GLOBAL const uint*)(wrow + (nf_i64)u * (2 * Q4_0_BLOCK_BYTES));
    const uint4 v0 = vload4(0, p), v1 = vload4(1, p);
    const uint w[9] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, p[8]};
    int i0 = 0, i1 = 0;
    for (int i = 0; i < 4; i++) {
        const uint q0 = (w[i] >> 16) | (w[i + 1] << 16);
        const uint q1 = w[5 + i];
        i0 = NF_DP4A((int)(q0 & 0x0F0F0F0Fu), (int)a->c[i], i0);
        i0 = NF_DP4A((int)((q0 >> 4) & 0x0F0F0F0Fu), (int)a->c[4 + i], i0);
        i1 = NF_DP4A((int)(q1 & 0x0F0F0F0Fu), (int)a->c[8 + i], i1);
        i1 = NF_DP4A((int)((q1 >> 4) & 0x0F0F0F0Fu), (int)a->c[12 + i], i1);
    }
    return a->d[0] * nf_sgr_half(w[0]) * (float)(i0 - 8 * a->s[0])
         + a->d[1] * nf_sgr_half(w[4] >> 16) * (float)(i1 - 8 * a->s[1]);
}

/* ---- Q4_K: unit u is quarter c = u % 4 of block u / 4: sub-blocks 2c (low nibbles of the 32 code bytes
 * at 16 + 32c) and 2c + 1 (high nibbles), with the block's 16-byte header d, dmin, scales[12]. ---- */

/* Q8_1 blocks 8b + 2c and 8b + 2c + 1 = 2u and 2u + 1, with their code sums for the minimum term. */
NF_INLINE void Q4_K_sgr_act(NF_GLOBAL const uchar* xq, int u, nf_sgr_act* a) {
    nf_sgr_pair_act_sums(xq, u, a);
}

/* Byte k (0..11) of the 6-bit scale table held in words s1..s3 of the header. */
NF_INLINE uint nf_sgr_q4k_byte(uint4 h, int k) {
    const uint w = k < 4 ? h.y : (k < 8 ? h.z : h.w);
    return (w >> ((k & 3) * 8)) & 0xFFu;
}

/* Scale (x) and minimum (y) of sub-block j, as ggml's get_scale_min_k4. */
NF_INLINE float2 nf_sgr_q4k_sm(uint4 h, int j) {
    if (j < 4) {
        return (float2)((float)(nf_sgr_q4k_byte(h, j) & 63u), (float)(nf_sgr_q4k_byte(h, j + 4) & 63u));
    }
    const uint hi = nf_sgr_q4k_byte(h, j + 4);
    return (float2)((float)((hi & 0xFu) | ((nf_sgr_q4k_byte(h, j - 4) >> 6) << 4)),
                    (float)((hi >> 4) | ((nf_sgr_q4k_byte(h, j) >> 6) << 4)));
}

NF_INLINE float Q4_K_sgr_dot(NF_GLOBAL const uchar* wrow, int u, const nf_sgr_act* a) {
    const int c = u & 3;
    NF_GLOBAL const uint* bp = (NF_GLOBAL const uint*)(wrow + (nf_i64)(u >> 2) * Q4_K_BLOCK_BYTES);
    const uint4 h = vload4(0, bp);
    const uint4 qa = vload4(1 + 2 * c, bp), qb = vload4(2 + 2 * c, bp);
    const uint q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    int i0 = 0, i1 = 0;
    for (int i = 0; i < 8; i++) {
        i0 = NF_DP4A((int)(q[i] & 0x0F0F0F0Fu), (int)a->c[i], i0);
        i1 = NF_DP4A((int)((q[i] >> 4) & 0x0F0F0F0Fu), (int)a->c[8 + i], i1);
    }
    const float d = nf_sgr_half(h.x), dmin = nf_sgr_half(h.x >> 16);
    const float2 sm0 = nf_sgr_q4k_sm(h, 2 * c), sm1 = nf_sgr_q4k_sm(h, 2 * c + 1);
    return a->d[0] * (d * sm0.x * (float)i0 - dmin * sm0.y * (float)a->s[0])
         + a->d[1] * (d * sm1.x * (float)i1 - dmin * sm1.y * (float)a->s[1]);
}

/* ---- Q5_K: unit u is quarter c = u % 4 of block u / 4, laid out as Q4_K's quarter with the 32 code bytes at
 * 48 + 32c and the fifth bits in the 32-byte qh at 16: bit 2c of qh byte l for the low nibble of code byte l,
 * bit 2c + 1 for its high nibble. ---- */

/* Q8_1 blocks 2u and 2u + 1 with their code sums for the minimum term, as for Q4_K. */
NF_INLINE void Q5_K_sgr_act(NF_GLOBAL const uchar* xq, int u, nf_sgr_act* a) {
    nf_sgr_pair_act_sums(xq, u, a);
}

NF_INLINE float Q5_K_sgr_dot(NF_GLOBAL const uchar* wrow, int u, const nf_sgr_act* a) {
    const int c = u & 3;
    NF_GLOBAL const uint* bp = (NF_GLOBAL const uint*)(wrow + (nf_i64)(u >> 2) * Q5_K_BLOCK_BYTES);
    const uint4 h = vload4(0, bp);
    const uint4 ha = vload4(1, bp), hb = vload4(2, bp);
    const uint4 qa = vload4(3 + 2 * c, bp), qb = vload4(4 + 2 * c, bp);
    const uint q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
    const uint qh[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
    const uint s0 = (uint)(2 * c), s1 = s0 + 1u;
    int i0 = 0, i1 = 0;
    for (int i = 0; i < 8; i++) {
        const uint lo = (q[i] & 0x0F0F0F0Fu) | (((qh[i] >> s0) & 0x01010101u) << 4);
        const uint hi = ((q[i] >> 4) & 0x0F0F0F0Fu) | (((qh[i] >> s1) & 0x01010101u) << 4);
        i0 = NF_DP4A((int)lo, (int)a->c[i], i0);
        i1 = NF_DP4A((int)hi, (int)a->c[8 + i], i1);
    }
    const float d = nf_sgr_half(h.x), dmin = nf_sgr_half(h.x >> 16);
    const float2 sm0 = nf_sgr_q4k_sm(h, 2 * c), sm1 = nf_sgr_q4k_sm(h, 2 * c + 1);
    return a->d[0] * (d * sm0.x * (float)i0 - dmin * sm0.y * (float)a->s[0])
         + a->d[1] * (d * sm1.x * (float)i1 - dmin * sm1.y * (float)a->s[1]);
}

/* ---- Q6_K: unit u is quarter r = u % 4 of block u / 4: half h = r / 2 and positions l in [16 lh, 16 lh + 16)
 * with lh = r % 2, which cover elements 128h + 32t + l of the four groups t; group t reads ql at
 * 64h + 32(t % 2) + l (low nibbles for t < 2, high for t >= 2), qh at 128 + 32h + l (bits 2t, 2t + 1),
 * scale 192 + 8h + 2t + lh. Blocks are 210 bytes, so odd blocks of a row start 2 bytes off a word. ---- */

/* Q8_1 blocks 8b + 4h + t (t = 0..3), the 16 codes at byte 16 lh of each, and their sums. */
NF_INLINE void Q6_K_sgr_act(NF_GLOBAL const uchar* xq, int u, nf_sgr_act* a) {
    const int r = u & 3;
    const int qb0 = (u >> 2) * 8 + (r >> 1) * 4;
    for (int t = 0; t < 4; t++) {
        NF_GLOBAL const uchar* blk = xq + (nf_i64)(qb0 + t) * Q8_1_BLOCK_BYTES;
        const uint4 x = vload4(0, (NF_GLOBAL const uint*)(blk + Q8_1_QS_OFFSET + 16 * (r & 1)));
        a->d[t] = vload_half(0, (NF_GLOBAL const NF_HALF*)blk);
        a->c[4 * t] = x.x; a->c[4 * t + 1] = x.y; a->c[4 * t + 2] = x.z; a->c[4 * t + 3] = x.w;
        a->s[t] = nf_sgr_sum4(x.x) + nf_sgr_sum4(x.y) + nf_sgr_sum4(x.z) + nf_sgr_sum4(x.w);
    }
}

NF_INLINE float Q6_K_sgr_dot(NF_GLOBAL const uchar* wrow, int u, const nf_sgr_act* a) {
    const int r = u & 3, hf = r >> 1, lh = r & 1;
    NF_GLOBAL const uchar* bp = wrow + (nf_i64)(u >> 2) * Q6_K_BLOCK_BYTES;
    const uint4 lo4 = nf_sgr_load16(bp + 64 * hf + 16 * lh);
    const uint4 hi4 = nf_sgr_load16(bp + 64 * hf + 32 + 16 * lh);
    const uint4 qh4 = nf_sgr_load16(bp + 128 + 32 * hf + 16 * lh);
    const uint lo[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
    const uint hi[4] = {hi4.x, hi4.y, hi4.z, hi4.w};
    const uint qh[4] = {qh4.x, qh4.y, qh4.z, qh4.w};
    const ushort4 sc = vload4(0, (NF_GLOBAL const ushort*)(bp + 192 + 8 * hf));
    const uint shs = (uint)lh * 8u;
    const float scl[4] = {(float)(char)((sc.x >> shs) & 0xFFu), (float)(char)((sc.y >> shs) & 0xFFu),
                          (float)(char)((sc.z >> shs) & 0xFFu), (float)(char)((sc.w >> shs) & 0xFFu)};
    float acc = 0.0f;
    for (int t = 0; t < 4; t++) {
        const uint qlsh = (uint)(t >> 1) * 4u, qhsh = (uint)t * 2u;
        int idot = 0;
        for (int i = 0; i < 4; i++) {
            const uint ql = (t & 1) ? hi[i] : lo[i];
            const int code = (int)(((ql >> qlsh) & 0x0F0F0F0Fu) | (((qh[i] >> qhsh) & 0x03030303u) << 4));
            idot = NF_DP4A(code, (int)a->c[4 * t + i], idot);
        }
        acc += a->d[t] * scl[t] * (float)(idot - 32 * a->s[t]);
    }
    return vload_half(0, (NF_GLOBAL const NF_HALF*)(bp + 208)) * acc;
}

/* ---- Kernel forms. ---- */

/* First output row of this subgroup. */
#define NF_SGR_ROW0() ((int)(get_group_id(0) * get_num_sub_groups() + get_sub_group_id()) * NF_SGR_ROWS)

#define NF_SGR_MATVEC(N, BS, BB) \
NF_SGR_ATTR NF_KERNEL void N##_MatVecSg_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row0 = NF_SGR_ROW0(); \
    if (row0 >= out_dim) return; \
    const int lane = get_sub_group_local_id(); \
    const nf_i64 rb = (nf_i64)(in_dim / BS) * BB; \
    NF_GLOBAL const uchar* wr[NF_SGR_ROWS]; \
    float acc[NF_SGR_ROWS]; \
    for (int r = 0; r < NF_SGR_ROWS; r++) { \
        wr[r] = weight + (nf_i64)min(row0 + r, out_dim - 1) * rb; \
        acc[r] = 0.0f; \
    } \
    for (int u = lane; u < in_dim / NF_SGR_UNIT; u += NF_SG_ROW_SIZE) { \
        nf_sgr_act a; \
        N##_sgr_act(input, u, &a); \
        for (int r = 0; r < NF_SGR_ROWS; r++) acc[r] += N##_sgr_dot(wr[r], u, &a); \
    } \
    for (int r = 0; r < NF_SGR_ROWS; r++) { \
        const float v = sub_group_reduce_add(acc[r]); \
        if (lane == 0 && row0 + r < out_dim) output[row0 + r] = v; \
    } \
}

#define NF_SGR_MATVECADD(N, BS, BB) \
NF_SGR_ATTR NF_KERNEL void N##_MatVecAddSg_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT addend_output) { \
    const int row0 = NF_SGR_ROW0(); \
    if (row0 >= out_dim) return; \
    const int lane = get_sub_group_local_id(); \
    const nf_i64 rb = (nf_i64)(in_dim / BS) * BB; \
    NF_GLOBAL const uchar* wr[NF_SGR_ROWS]; \
    float acc[NF_SGR_ROWS]; \
    for (int r = 0; r < NF_SGR_ROWS; r++) { \
        wr[r] = weight + (nf_i64)min(row0 + r, out_dim - 1) * rb; \
        acc[r] = 0.0f; \
    } \
    for (int u = lane; u < in_dim / NF_SGR_UNIT; u += NF_SG_ROW_SIZE) { \
        nf_sgr_act a; \
        N##_sgr_act(input, u, &a); \
        for (int r = 0; r < NF_SGR_ROWS; r++) acc[r] += N##_sgr_dot(wr[r], u, &a); \
    } \
    for (int r = 0; r < NF_SGR_ROWS; r++) { \
        const float v = sub_group_reduce_add(acc[r]); \
        if (lane == 0 && row0 + r < out_dim) addend_output[row0 + r] += v; \
    } \
}

#define NF_SGR_FUSEDGATEUP(N, BS, BB) \
NF_SGR_ATTR NF_KERNEL void N##_MatVecFusedGateUpSg_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT up_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row0 = NF_SGR_ROW0(); \
    if (row0 >= out_dim) return; \
    const int lane = get_sub_group_local_id(); \
    const nf_i64 rb = (nf_i64)(in_dim / BS) * BB; \
    nf_i64 ro[NF_SGR_ROWS]; \
    float gs[NF_SGR_ROWS], us[NF_SGR_ROWS]; \
    for (int r = 0; r < NF_SGR_ROWS; r++) { \
        ro[r] = (nf_i64)min(row0 + r, out_dim - 1) * rb; \
        gs[r] = 0.0f; \
        us[r] = 0.0f; \
    } \
    for (int u = lane; u < in_dim / NF_SGR_UNIT; u += NF_SG_ROW_SIZE) { \
        nf_sgr_act a; \
        N##_sgr_act(input, u, &a); \
        for (int r = 0; r < NF_SGR_ROWS; r++) { \
            gs[r] += N##_sgr_dot(gate_weight + ro[r], u, &a); \
            us[r] += N##_sgr_dot(up_weight + ro[r], u, &a); \
        } \
    } \
    for (int r = 0; r < NF_SGR_ROWS; r++) { \
        const float g = sub_group_reduce_add(gs[r]); \
        const float v = sub_group_reduce_add(us[r]); \
        if (lane == 0 && row0 + r < out_dim) output[row0 + r] = nf_silu(g) * v; \
    } \
}

#define NF_SGR_ALL(N, BS, BB) \
    NF_SGR_MATVEC(N, BS, BB) \
    NF_SGR_MATVECADD(N, BS, BB) \
    NF_SGR_FUSEDGATEUP(N, BS, BB)

NF_SGR_ALL(Q8_0, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES)
NF_SGR_ALL(Q4_0, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES)
NF_SGR_ALL(Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES)
NF_SGR_ALL(Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES)
NF_SGR_ALL(Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES)

#endif /* USE_SUBGROUPS && cl_intel_required_subgroup_size && NF_SG_ROW_SIZE */
