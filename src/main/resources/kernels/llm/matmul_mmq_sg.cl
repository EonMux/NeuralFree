/* matmul_mmq_sg.cl - subgroup-row int8 MMQ GEMM of Q8_0, Q4_K and Q6_K weights (OpenCL only). Requires: common/dialect.h, common/matmul_mmq_config.h */
/*
 * Q8_0_MatMulSg_MMQ computes what Q8_0_MatMul_MMQ (common/matmul_mmq_family.h) computes, with its argument list
 * and the same staged Q8_1 activation records (LLM_Quant_MMQ_Q8_1, NF_Q8_REC(16) units per 16-deep record), under
 * a work mapping built for the Intel Xe GPUs, whose int8 GEMM is bound by load messages and instruction issue
 * rather than by DRAM:
 *
 * - A work-group is NF_SGM_SUBGROUPS 16-lane subgroups (NF_SG_ROW_SIZE, pinned by intel_reqd_sub_group_size)
 *   and owns NF_SGM_TM = 128 weight rows by NF_SGM_TN = 32 tokens: lane l of the work-group owns row m0 + l and
 *   keeps all 32 token sums of that row in registers. Launch: local = {NF_SGM_WG, 1}, global =
 *   {ceil(M / NF_SGM_TM) * NF_SGM_WG, ceil(N / NF_SGM_TN)}.
 * - A stage is 64 deep: one Q8_0 block pair of each lane's row, read straight from the GGUF bytes with four
 *   16-byte loads and one 4-byte load (the 68-byte pair is 4-byte aligned when K is a multiple of 64), and the
 *   four 16-deep activation records of every token of the tile, which the work-group copies into local memory
 *   (one record per work-item, double-buffered, the next stage's copy issued between the two blocks).
 * - Every token's 16-unit record pair is read by one subgroup block read (a single uniform local load) and its
 *   words are shared across the lanes by constant-index shuffles, so each lane's 8 dp4a per block against that
 *   token cost no per-lane local traffic. One barrier per stage.
 *
 * Q4_K_MatMulSg_MMQ and Q6_K_MatMulSg_MMQ (below) compute what Q4_K_MatMul_MMQ and Q6_K_MatMul_MMQ compute under
 * the same mapping, launch and records, a stage being a quarter of a 256-element super-block.
 *
 * Each output is one lane's fixed-order float sum over K, independent of N and of the tile position, so results
 * are deterministic; only the float summation order differs from the portable {T}_MatMul_MMQ's.
 *
 * Preconditions the host checks: K is a multiple of 64 (of 256 for the K-quants), the weight buffer starts 4-byte
 * aligned, the records are symmetric (zero point unit 0) and 16 deep (NF_MMQ_TK == 16). The kernels exist only
 * when the host defines NF_SG_ROW_SIZE (16) for a subgroup program and the compiler offers subgroup block reads of
 * local memory (cl_intel_subgroup_local_block_io).
 */
#if defined(USE_SUBGROUPS) && defined(cl_intel_required_subgroup_size) && defined(cl_intel_subgroup_local_block_io) \
    && defined(NF_SG_ROW_SIZE) && NF_SG_ROW_SIZE == 16 && NF_MMQ_TK == 16

/* Subgroups per work-group, work-items per work-group (= weight rows per work-group) and tokens per work-group
 * (the host mirrors all three in SubgroupMmqGemm). */
#define NF_SGM_SUBGROUPS 8
#define NF_SGM_WG (NF_SGM_SUBGROUPS * NF_SG_ROW_SIZE)
#define NF_SGM_TM NF_SGM_WG
#define NF_SGM_TN 32
/* Stage depth: one Q8_0 block pair, four 16-deep records. */
#define NF_SGM_TK 64
/* Units of one 16-deep record (NF_Q8_REC(16)): codes 0..3, d 4, code sum 5, zero point 6. */
#define NF_SGM_RC 7
/* Words of a 68-byte Q8_0 block pair. */
#define NF_SGM_PAIR_WORDS 17
/* Local units per token: records 0 and 1 (block A) at 0, records 2 and 3 (block B) at 16, so each block's two
 * records are one 64-byte-aligned 16-unit segment for the subgroup block read. */
#define NF_SGM_XT 32
#define NF_SGM_XBUF (NF_SGM_TN * NF_SGM_XT)

/* Float value of the f16 bits in the low half of v. */
NF_INLINE float nf_sgm_half(uint v) {
    const ushort h = (ushort)(v & 0xFFFFu);
    return vload_half(0, (const NF_HALF*)&h);
}

/* Copies stage st's record of this work-item (token gt, record gc) into dst, or a zero record past N (d = 0). */
#define NF_SGM_GATHER(dst, st) \
    { \
        NF_LOCAL uint* gd = (dst) + gdst; \
        uint4 g0 = (uint4)(0u); \
        uint3 g1 = (uint3)(0u); \
        if (glive) { \
            NF_GLOBAL const uint* gs = qbase + goff + (uint)(st) * (4u * NF_SGM_RC); \
            g0 = vload4(0, gs); \
            g1 = vload3(0, gs + 4); \
        } \
        vstore4(g0, 0, gd); \
        vstore3(g1, 0, gd + 4); \
    }

/* Unit k of the token segment xv held across the subgroup (a constant k is a scalar register region). */
#define NF_SGM_X(k) intel_sub_group_shuffle(xv, (k))

/* One 32-deep block of the stage: codes c[0..7] of this lane's row with block scale dw against the segment of
 * each token at xb (records r at units 0..6 and r + 1 at 7..13). */
#define NF_SGM_BLOCK(xb, c, dw) \
    _Pragma("unroll") \
    for (int t = 0; t < NF_SGM_TN; t++) { \
        const uint xv = intel_sub_group_block_read((const NF_LOCAL uint*)((xb) + t * NF_SGM_XT)); \
        int i0 = NF_DP4A((int)c[0], (int)NF_SGM_X(0), 0); \
        i0 = NF_DP4A((int)c[1], (int)NF_SGM_X(1), i0); \
        i0 = NF_DP4A((int)c[2], (int)NF_SGM_X(2), i0); \
        i0 = NF_DP4A((int)c[3], (int)NF_SGM_X(3), i0); \
        int i1 = NF_DP4A((int)c[4], (int)NF_SGM_X(7), 0); \
        i1 = NF_DP4A((int)c[5], (int)NF_SGM_X(8), i1); \
        i1 = NF_DP4A((int)c[6], (int)NF_SGM_X(9), i1); \
        i1 = NF_DP4A((int)c[7], (int)NF_SGM_X(10), i1); \
        acc[t] = fma(dw, fma((float)i1, as_float(NF_SGM_X(11)), (float)i0 * as_float(NF_SGM_X(4))), acc[t]); \
    }

/*
 * The statement order below is the one the offline register study settled on (Xe-LPG, 128 GRF at SIMD16): the
 * 32 accumulators and the block's codes peak at about 111 GRF, and gathering at the top of the stage instead of
 * between the blocks, or loading the segment per lane instead of by block read, spills.
 */
__attribute__((intel_reqd_sub_group_size(NF_SG_ROW_SIZE))) __attribute__((reqd_work_group_size(NF_SGM_WG, 1, 1)))
NF_KERNEL void Q8_0_MatMulSg_MMQ(
    const int M, const int K, const int N,
    const int q_stride, const int q_off, const int y_stride, const int y_off,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const uint* NF_RESTRICT q8,
    NF_GLOBAL float* NF_RESTRICT y) {
    NF_LOCAL uint xs[2 * NF_SGM_XBUF];
    const int lid = (int)get_local_id(0);
    const int m0 = (int)get_group_id(0) * NF_SGM_TM;
    const int n0 = (int)get_group_id(1) * NF_SGM_TN;
    const uint row_words = (uint)(K / NF_SGM_TK) * NF_SGM_PAIR_WORDS;
    /* Uniform 64-bit bases and 32-bit lane offsets: a 64-bit address per lane costs registers the tile needs.
     * Rows past M repeat row M - 1 and drop their stores. */
    NF_GLOBAL const uint* wbase = (NF_GLOBAL const uint*)weight + (nf_i64)m0 * row_words;
    uint woff = (uint)min(lid, M - 1 - m0) * row_words;
    NF_GLOBAL const uint* qbase = q8 + (nf_i64)q_off + (nf_i64)n0 * q_stride * NF_SGM_RC;
    const int gt = lid >> 2, gc = lid & 3;
    const bool glive = n0 + gt < N;
    const uint goff = ((uint)gt * (uint)q_stride + (uint)gc) * NF_SGM_RC;
    const int gdst = gt * NF_SGM_XT + (gc >> 1) * 16 + (gc & 1) * NF_SGM_RC;
    float acc[NF_SGM_TN];
    _Pragma("unroll")
    for (int t = 0; t < NF_SGM_TN; t++) {
        acc[t] = 0.0f;
    }
    const int stages = K / NF_SGM_TK;
    NF_SGM_GATHER(xs, 0)
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 0; s < stages; s++, woff += NF_SGM_PAIR_WORDS) {
        NF_LOCAL const uint* xb = xs + (s & 1) * NF_SGM_XBUF;
        NF_GLOBAL const uint* p = wbase + woff;
        const uint4 a0 = vload4(0, p), a1 = vload4(1, p);
        const uint w8 = p[8];
        {
            /* Block A: d is the low half of word 0, code word i is (w_i >> 16) | (w_{i+1} << 16). */
            const uint c[8] = {(a0.x >> 16) | (a0.y << 16), (a0.y >> 16) | (a0.z << 16),
                               (a0.z >> 16) | (a0.w << 16), (a0.w >> 16) | (a1.x << 16),
                               (a1.x >> 16) | (a1.y << 16), (a1.y >> 16) | (a1.z << 16),
                               (a1.z >> 16) | (a1.w << 16), (a1.w >> 16) | (w8 << 16)};
            const float dw = nf_sgm_half(a0.x);
            NF_SGM_BLOCK(xb, c, dw)
        }
        if (s + 1 < stages) {
            NF_SGM_GATHER(xs + ((s + 1) & 1) * NF_SGM_XBUF, s + 1)
        }
        {
            /* Block B: d is the high half of word 8, its code words are words 9..16. */
            const uint4 b0 = vload4(0, p + 9), b1 = vload4(0, p + 13);
            const uint c[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            const float dw = nf_sgm_half(w8 >> 16);
            NF_SGM_BLOCK(xb + 16, c, dw)
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int row = m0 + lid;
    if (row < M) {
        NF_GLOBAL float* yr = y + (nf_i64)y_off + (nf_i64)n0 * y_stride + row;
        _Pragma("unroll")
        for (int t = 0; t < NF_SGM_TN; t++) {
            if (n0 + t < N) {
                yr[(nf_i64)t * y_stride] = acc[t];
            }
        }
    }
}


/* ---- K-quant forms: Q4_K_MatMulSg_MMQ and Q6_K_MatMulSg_MMQ. ----
 *
 * The same work mapping, launch geometry and record consumption as Q8_0_MatMulSg_MMQ, with a 64-deep stage that is
 * quarter q = s % 4 of super-block s / 4 (K is a multiple of 256). The record gather of these forms rewrites unit 6
 * (the zero point, 0 for the symmetric records the host admits) in local memory with the activation side of the
 * type's offset term, so the per-token work stays 8 dp4a per 32-deep half and no per-token offset arithmetic:
 * - Q4_K: W = d * sc * c - dmin * m with 4-bit codes c, so a 16-deep record r contributes
 *   dx_r * (d * sc * idot_r - dmin * m * sum_r). Unit 6 of both records of each 32-deep sub-block pair holds
 *   dx_r * sum_r + dx_(r+1) * sum_(r+1) (the pair is summed across adjacent lanes of the gather).
 * - Q6_K: W = d * sc_r * (c - 32) with 6-bit codes c and one int8 scale per 16 elements, so record r contributes
 *   d * sc_r * dx_r * (idot_r - 32 * sum_r). Unit 6 holds -32 * sum_r, the initial value of the record's exact
 *   integer dot.
 */

/* Words of a 144-byte Q4_K block and bytes of a 210-byte Q6_K block. */
#define NF_SGM_Q4K_WORDS 36
#define NF_SGM_Q6K_BYTES 210
/* Low nibbles and low two bits of every byte of a word. */
#define NF_SGM_NIB 0x0F0F0F0Fu
#define NF_SGM_TWO 0x03030303u

/* Copies stage st's record of this work-item into dst + gdst like NF_SGM_GATHER, with unit 6 replaced by
 * FIX(g1), g1 = (d, code sum, zero point) of the record; a record past N stays all zero (FIX(0) is 0 for both
 * forms). FIX runs on every lane, so it may use subgroup shuffles. */
#define NF_SGM_GATHER_KQ(dst, st, FIX) \
    { \
        NF_LOCAL uint* gd = (dst) + gdst; \
        uint4 g0 = (uint4)(0u); \
        uint3 g1 = (uint3)(0u); \
        if (glive) { \
            NF_GLOBAL const uint* gs = qbase + goff + (uint)(st) * (4u * NF_SGM_RC); \
            g0 = vload4(0, gs); \
            g1 = vload3(0, gs + 4); \
        } \
        g1.z = FIX(g1); \
        vstore4(g0, 0, gd); \
        vstore3(g1, 0, gd + 4); \
    }

/* Q4_K unit 6: dx * sum of this record plus that of its sub-block partner, the adjacent lane (gc ^ 1). */
NF_INLINE uint nf_sgm_q4k_fix(uint3 g1) {
    const float ds = as_float(g1.x) * (float)(int)g1.y;
    return as_uint(ds + intel_sub_group_shuffle_xor(ds, 1u));
}
#define NF_SGM_Q4K_FIX(g1) nf_sgm_q4k_fix(g1)
/* Q6_K unit 6: -32 * sum, the code offset of the record's integer dot. */
#define NF_SGM_Q6K_FIX(g1) ((uint)(-32 * (int)(g1).y))

/* Zeroes the NF_SGM_TN accumulators. */
#define NF_SGM_ACC_ZERO \
    _Pragma("unroll") \
    for (int t = 0; t < NF_SGM_TN; t++) { \
        acc[t] = 0.0f; \
    }

/* Stores row m0 + lid of the accumulators, dropping rows past M and tokens past N. */
#define NF_SGM_STORE \
    { \
        const int row = m0 + lid; \
        if (row < M) { \
            NF_GLOBAL float* yr = y + (nf_i64)y_off + (nf_i64)n0 * y_stride + row; \
            _Pragma("unroll") \
            for (int t = 0; t < NF_SGM_TN; t++) { \
                if (n0 + t < N) { \
                    yr[(nf_i64)t * y_stride] = acc[t]; \
                } \
            } \
        } \
    }

/* One Q4_K sub-block of the stage against the tile's tokens: 4-bit codes c[0..7] with the lane's d * sc (dsc) and
 * -dmin * m (nmm) against the segment at xb (records r at units 0..6, r + 1 at 7..13, the pair's dx * sum in 6). */
#define NF_SGM_BLOCK_Q4K(xb, c, dsc, nmm) \
    _Pragma("unroll") \
    for (int t = 0; t < NF_SGM_TN; t++) { \
        const uint xv = intel_sub_group_block_read((const NF_LOCAL uint*)((xb) + t * NF_SGM_XT)); \
        int i0 = NF_DP4A((int)c[0], (int)NF_SGM_X(0), 0); \
        i0 = NF_DP4A((int)c[1], (int)NF_SGM_X(1), i0); \
        i0 = NF_DP4A((int)c[2], (int)NF_SGM_X(2), i0); \
        i0 = NF_DP4A((int)c[3], (int)NF_SGM_X(3), i0); \
        int i1 = NF_DP4A((int)c[4], (int)NF_SGM_X(7), 0); \
        i1 = NF_DP4A((int)c[5], (int)NF_SGM_X(8), i1); \
        i1 = NF_DP4A((int)c[6], (int)NF_SGM_X(9), i1); \
        i1 = NF_DP4A((int)c[7], (int)NF_SGM_X(10), i1); \
        acc[t] = fma(dsc, fma((float)i1, as_float(NF_SGM_X(11)), (float)i0 * as_float(NF_SGM_X(4))), acc[t]); \
        acc[t] = fma(nmm, as_float(NF_SGM_X(6)), acc[t]); \
    }

/* One Q6_K pass of the stage against the tile's tokens: 6-bit codes c[0..3] of record A (units 0..6 of the segment at xb)
 * and c[4..7] of record B (units 7..13), with the lane's d * sc of each (ea, eb); each integer dot starts at its
 * record's -32 * sum. */
#define NF_SGM_BLOCK_Q6K(xb, c, ea, eb) \
    _Pragma("unroll") \
    for (int t = 0; t < NF_SGM_TN; t++) { \
        const uint xv = intel_sub_group_block_read((const NF_LOCAL uint*)((xb) + t * NF_SGM_XT)); \
        int i0 = NF_DP4A((int)c[0], (int)NF_SGM_X(0), (int)NF_SGM_X(6)); \
        i0 = NF_DP4A((int)c[1], (int)NF_SGM_X(1), i0); \
        i0 = NF_DP4A((int)c[2], (int)NF_SGM_X(2), i0); \
        i0 = NF_DP4A((int)c[3], (int)NF_SGM_X(3), i0); \
        int i1 = NF_DP4A((int)c[4], (int)NF_SGM_X(7), (int)NF_SGM_X(13)); \
        i1 = NF_DP4A((int)c[5], (int)NF_SGM_X(8), i1); \
        i1 = NF_DP4A((int)c[6], (int)NF_SGM_X(9), i1); \
        i1 = NF_DP4A((int)c[7], (int)NF_SGM_X(10), i1); \
        acc[t] = fma(eb, (float)i1 * as_float(NF_SGM_X(11)), fma(ea, (float)i0 * as_float(NF_SGM_X(4)), acc[t])); \
    }

/*
 * d * sc (x) and -dmin * m (y) of sub-block j = 2q + i of the Q4_K block whose 16-byte header is h (d, dmin,
 * scales[12]), with the 6-bit scale and minimum decoded as ggml's get_scale_min_k4: sub-blocks 0..3 take the low
 * 6 bits of scale bytes j and j + 4, sub-blocks 4..7 the nibbles of byte j + 4 with the top two bits of bytes j - 4
 * and j. q and i are uniform across the subgroup.
 */
NF_INLINE float2 nf_sgm_q4k_scales(uint4 h, int q, int i) {
    const uint sh = (uint)(8 * (2 * (q & 1) + i));
    const uint lo = (h.y >> sh) & 0xFFu, lm = (h.z >> sh) & 0xFFu, hw = (h.w >> sh) & 0xFFu;
    const uint sc = q < 2 ? lo & 63u : (hw & 0xFu) | ((lo >> 6) << 4);
    const uint mn = q < 2 ? lm & 63u : (hw >> 4) | ((lm >> 6) << 4);
    return (float2)(nf_sgm_half(h.x) * (float)sc, -(nf_sgm_half(h.x >> 16) * (float)mn));
}

__attribute__((intel_reqd_sub_group_size(NF_SG_ROW_SIZE))) __attribute__((reqd_work_group_size(NF_SGM_WG, 1, 1)))
NF_KERNEL void Q4_K_MatMulSg_MMQ(
    const int M, const int K, const int N,
    const int q_stride, const int q_off, const int y_stride, const int y_off,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const uint* NF_RESTRICT q8,
    NF_GLOBAL float* NF_RESTRICT y) {
    NF_LOCAL uint xs[2 * NF_SGM_XBUF];
    const int lid = (int)get_local_id(0);
    const int m0 = (int)get_group_id(0) * NF_SGM_TM;
    const int n0 = (int)get_group_id(1) * NF_SGM_TN;
    const uint row_words = (uint)(K / 256) * NF_SGM_Q4K_WORDS;
    NF_GLOBAL const uint* wbase = (NF_GLOBAL const uint*)weight + (nf_i64)m0 * row_words;
    const uint woff = (uint)min(lid, M - 1 - m0) * row_words;
    NF_GLOBAL const uint* qbase = q8 + (nf_i64)q_off + (nf_i64)n0 * q_stride * NF_SGM_RC;
    const int gt = lid >> 2, gc = lid & 3;
    const bool glive = n0 + gt < N;
    const uint goff = ((uint)gt * (uint)q_stride + (uint)gc) * NF_SGM_RC;
    const int gdst = gt * NF_SGM_XT + (gc >> 1) * 16 + (gc & 1) * NF_SGM_RC;
    float acc[NF_SGM_TN];
    NF_SGM_ACC_ZERO
    const int stages = K / NF_SGM_TK;
    NF_SGM_GATHER_KQ(xs, 0, NF_SGM_Q4K_FIX)
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 0; s < stages; s++) {
        NF_LOCAL const uint* xb = xs + (s & 1) * NF_SGM_XBUF;
        const int q = s & 3;
        /* The 32 code bytes of quarter q (low nibbles sub-block 2q, high nibbles 2q + 1) and the block header.
         * Each half loads its own codes and header: holding sub-block 2q + 1's words or scales across the first
         * half and the gather spills the 32-token tile at SIMD16. */
        NF_GLOBAL const uint* p = wbase + woff + (uint)(s >> 2) * NF_SGM_Q4K_WORDS;
        NF_GLOBAL const uint* pq = p + 4 + 8 * q;
        {
            const uint4 a0 = vload4(0, pq), a1 = vload4(1, pq);
            const uint c[8] = {a0.x & NF_SGM_NIB, a0.y & NF_SGM_NIB, a0.z & NF_SGM_NIB, a0.w & NF_SGM_NIB,
                               a1.x & NF_SGM_NIB, a1.y & NF_SGM_NIB, a1.z & NF_SGM_NIB, a1.w & NF_SGM_NIB};
            const float2 sm = nf_sgm_q4k_scales(vload4(0, p), q, 0);
            NF_SGM_BLOCK_Q4K(xb, c, sm.x, sm.y)
        }
        if (s + 1 < stages) {
            NF_SGM_GATHER_KQ(xs + ((s + 1) & 1) * NF_SGM_XBUF, s + 1, NF_SGM_Q4K_FIX)
        }
        {
            const uint4 b0 = vload4(0, pq), b1 = vload4(1, pq);
            const uint c[8] = {(b0.x >> 4) & NF_SGM_NIB, (b0.y >> 4) & NF_SGM_NIB, (b0.z >> 4) & NF_SGM_NIB,
                               (b0.w >> 4) & NF_SGM_NIB, (b1.x >> 4) & NF_SGM_NIB, (b1.y >> 4) & NF_SGM_NIB,
                               (b1.z >> 4) & NF_SGM_NIB, (b1.w >> 4) & NF_SGM_NIB};
            const float2 sm = nf_sgm_q4k_scales(vload4(0, p), q, 1);
            NF_SGM_BLOCK_Q4K(xb + 16, c, sm.x, sm.y)
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    NF_SGM_STORE
}

/* A word of a byte run that starts sh / 8 bytes (0 or 2) past its covering aligned word a, b the next one. */
#define NF_SGM_FUNNEL(a, b) (((a) >> sh) | (((b) << 16) << (16u - sh)))

/* The 16-byte run whose covering aligned words start at pw, shifted by sh bits. */
NF_INLINE uint4 nf_sgm_run16(NF_GLOBAL const uint* pw, uint sh) {
    const uint4 v = vload4(0, pw);
    const uint e = pw[4];
    return (uint4)(NF_SGM_FUNNEL(v.x, v.y), NF_SGM_FUNNEL(v.y, v.z), NF_SGM_FUNNEL(v.z, v.w), NF_SGM_FUNNEL(v.w, e));
}

/* The 6-bit codes of 4 bytes: the nibble of each ql byte selected by nsh and the two qh bits selected by hsh. */
#define NF_SGM_Q6K_CODE(ql, qh, hsh) ((((ql) >> nsh) & NF_SGM_NIB) | ((((qh) >> (hsh)) & NF_SGM_TWO) << 4))

/*
 * Q6_K quarter q = 2 hf + qo of a block covers groups t = 2 qo and 2 qo + 1 of half hf, l = 0..31: the low
 * (qo = 0) or high (qo = 1) nibbles of ql bytes 64 hf + 32 (t % 2) + l, bits 2t, 2t + 1 of qh bytes
 * 128 + 32 hf + l, and int8 scales 192 + 4q + r for its records r = 0..3 (record r: group 2 qo + r / 2,
 * l = 16 (r % 2) ...). Pass p takes records p and 2 + p (l = 16p .. 16p + 15 of both groups), which the gather
 * places as one 16-unit segment per pass. Blocks are 210 bytes, so a block starts 0 or 2 bytes past a word: every
 * run is read from its covering aligned words and funnel-shifted, and the trailing d is a 2-byte load.
 */
#define NF_SGM_Q6K_PASS(pss, xseg) \
    { \
        const uint4 qh = nf_sgm_run16(pa + 32 + 8 * hf + 4 * (pss), sh); \
        const uint4 la = nf_sgm_run16(pa + 16 * hf + 4 * (pss), sh); \
        const uint4 lb = nf_sgm_run16(pa + 16 * hf + 8 + 4 * (pss), sh); \
        const uint c[8] = { \
            NF_SGM_Q6K_CODE(la.x, qh.x, nsh), NF_SGM_Q6K_CODE(la.y, qh.y, nsh), \
            NF_SGM_Q6K_CODE(la.z, qh.z, nsh), NF_SGM_Q6K_CODE(la.w, qh.w, nsh), \
            NF_SGM_Q6K_CODE(lb.x, qh.x, nsh + 2u), NF_SGM_Q6K_CODE(lb.y, qh.y, nsh + 2u), \
            NF_SGM_Q6K_CODE(lb.z, qh.z, nsh + 2u), NF_SGM_Q6K_CODE(lb.w, qh.w, nsh + 2u)}; \
        const float ea = d * (float)(nf_i8)(scw >> (8 * (pss))); \
        const float eb = d * (float)(nf_i8)(scw >> (8 * (2 + (pss)))); \
        NF_SGM_BLOCK_Q6K(xseg, c, ea, eb) \
    }

__attribute__((intel_reqd_sub_group_size(NF_SG_ROW_SIZE))) __attribute__((reqd_work_group_size(NF_SGM_WG, 1, 1)))
NF_KERNEL void Q6_K_MatMulSg_MMQ(
    const int M, const int K, const int N,
    const int q_stride, const int q_off, const int y_stride, const int y_off,
    NF_GLOBAL const uchar* NF_RESTRICT weight,
    NF_GLOBAL const uint* NF_RESTRICT q8,
    NF_GLOBAL float* NF_RESTRICT y) {
    NF_LOCAL uint xs[2 * NF_SGM_XBUF];
    const int lid = (int)get_local_id(0);
    const int m0 = (int)get_group_id(0) * NF_SGM_TM;
    const int n0 = (int)get_group_id(1) * NF_SGM_TN;
    const uint row_bytes = (uint)(K / 256) * NF_SGM_Q6K_BYTES;
    /* m0 is a multiple of 128, so the tile base is 4-byte aligned; lane offsets are bytes. */
    NF_GLOBAL const uchar* bbase = weight + (nf_i64)m0 * row_bytes;
    NF_GLOBAL const uint* wbase = (NF_GLOBAL const uint*)bbase;
    NF_GLOBAL const ushort* hbase = (NF_GLOBAL const ushort*)bbase;
    /* The lane's block offset is carried through the loop: a row offset held beside it spills the tile. */
    uint bo = (uint)min(lid, M - 1 - m0) * row_bytes;
    NF_GLOBAL const uint* qbase = q8 + (nf_i64)q_off + (nf_i64)n0 * q_stride * NF_SGM_RC;
    const int gt = lid >> 2, gc = lid & 3;
    const bool glive = n0 + gt < N;
    const uint goff = ((uint)gt * (uint)q_stride + (uint)gc) * NF_SGM_RC;
    /* Records 0 and 2 (pass 0) at units 0 and 7, records 1 and 3 (pass 1) at 16 and 23. */
    const int gdst = gt * NF_SGM_XT + (gc & 1) * 16 + (gc >> 1) * NF_SGM_RC;
    float acc[NF_SGM_TN];
    NF_SGM_ACC_ZERO
    const int stages = K / NF_SGM_TK;
    NF_SGM_GATHER_KQ(xs, 0, NF_SGM_Q6K_FIX)
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 0; s < stages; s++) {
        NF_LOCAL const uint* xb = xs + (s & 1) * NF_SGM_XBUF;
        const int q = s & 3, hf = q >> 1;
        const uint nsh = (uint)(4 * (q & 1));
        const uint al = bo & 2u, sh = al << 3;
        NF_GLOBAL const uint* pa = wbase + ((bo - al) >> 2);
        /* Scales 192 + 4q .. 195 + 4q: the two covering words, funnel-shifted when the block is 2 bytes off (an
         * aligned block loads the same word twice, so no read passes the scale run). */
        const uint scw = NF_SGM_FUNNEL(pa[48 + q], pa[48 + q + (al >> 1)]);
        const float d = nf_sgm_half((uint)hbase[(bo + 208u) >> 1]);
        NF_SGM_Q6K_PASS(0, xb)
        if (s + 1 < stages) {
            NF_SGM_GATHER_KQ(xs + ((s + 1) & 1) * NF_SGM_XBUF, s + 1, NF_SGM_Q6K_FIX)
        }
        NF_SGM_Q6K_PASS(1, xb + 16)
        bo += q == 3 ? NF_SGM_Q6K_BYTES : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    NF_SGM_STORE
}

#endif
