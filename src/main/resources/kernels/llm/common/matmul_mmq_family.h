/* common/matmul_mmq_family.h - {T}_MatMul_MMQ macro family: int8 MMQ GEMM for prefill and batched
 * decode over the llama.cpp MMQ quant types. Requires: common/dialect.h, common/config.h,
 * common/matmul_mmq_config.h, common/quant_blocks.h */
/*
 * NF_MMQ_GEMM(T, BS, BB, SG, CODE, SCALE, OFF) defines the kernel T##_MatMul_MMQ for weight type T
 * with BS elements and BB bytes per block and one (scale, offset) pair per SG elements (16 for
 * Q2_K, Q3_K and Q6_K, 32 for the others); NF_MMQ_GEMM_S(T, ...) defines its small-token-tile twin
 * T##_MatMul_MMQ_S over a 16-token tile (see the tiling section):
 *
 *   T_MatMul_MMQ(i M, i K, i N, i q_stride, i q_off, i y_stride, i y_off,
 *                R uchar weight, R uint q8, W float y)
 *   y[y_off + t*y_stride + r] = sum_{k < K} W[r][k] * x[t][k],  r < M, t < N
 *
 * Same contract as T##_MatMul_Tiled (common/matmul_tiled_family.h): ggml_mul_mat(W, X) with X a
 * batch of N activation rows. The weight stays quantized and the activations arrive staged by
 * LLM_Quant_MMQ_Q8_1 (kernels/llm/quant_mmq_q8_1.cl), which quantized every token row once per
 * projection, shared by all row tiles (llama.cpp quantize_mmq_q8_1): q8 holds, per token and
 * NF_MMQ_TK-deep chunk, one record of NF_MMQ_RC 32-bit units, record (t, c) at unit index
 * q_off + (t * q_stride + c) * NF_MMQ_RC with q_stride the record stride between tokens (at least
 * K / NF_MMQ_TK): NF_MMQ_TK / 4 code ints (element depth k of the chunk in byte k % 4 of int k / 4,
 * two's complement), the chunk scale d at NF_MMQ_TK / 4 and the NF_MMQ_TK / 16 code sums of the
 * 16-element halves after it. x[k] = d * q[k] + e[k] with |e[k]| <= d/2, and W[k] = scale * c[k]
 * + offset with c the code, so the exact dot is recovered as
 * d * sum_h (scale_h * sum(c*q) + offset_h * sum_h(q)) plus the quantization error of x.
 *
 * CODE(bp, e) is the integer code of element e (block-relative) of the block at bp, SCALE(bp, k0)
 * and OFF(bp, k0) the scale and offset of the SG-element group holding element k0 % BS; all three
 * use the field offsets of common/quant_decode_legacy.h, common/quant_decode_k.h and
 * common/quant_decode_iq.h. Codes are stored signed; every code range (Q4_0 0..15, the IQ grid
 * bytes up to +-62, up to Q8_0 -128..127) fits nf_i8, so a stage dot of code pairs stays within
 * int (127 * 127 * 32 << 2^31).
 *
 * Tiling (common/matmul_mmq_config.h): a workgroup of NF_MMQ_WG work-items computes a tile of
 * NF_MMQ_TM weight rows x NF_MMQ_TN tokens in NF_MMQ_TK-deep stages: 64 x 64 x 32 on the tensor
 * path, 64 x 16 on the packed-dot path, 64 x 128 on RDNA3 and 128 x 64 on CDNA; the _S twins run
 * the same tiling over 16-token tiles. There are
 * NF_MMQ_TX = NF_MMQ_TM / NF_MMQ_RM row lanes and 16 token lanes, so NF_MMQ_WG = NF_MMQ_TX * 16
 * work-items. Work-item lid, with tx = lid % NF_MMQ_TX and ty = lid / NF_MMQ_TX, loads depths
 * ty (+16h) of the codes of rows tx + NF_MMQ_TX*i (i < NF_MMQ_RM) into the local code tile and the
 * chunk records of its tokens ty + 16*j (j < NF_MMQ_TN / 16) into the local d / code-sum slots; the
 * staged codes themselves are read straight from q8 by the datapath. Datapaths:
 * - CUDA sm_80+ (NF_MMQ_TENSOR): the int8 tensor cores, as llama.cpp ggml-cuda/mmq.cuh on Ampere
 *   and later. Warp w covers rows (w >> 1) * 16 and tokens (w & 1) * NF_MMQ_TN / 2 .. +NF_MMQ_TN / 2
 *   as NF_MMQ_TN / 16 16x8 blocks, loading its B fragments straight from the staged code records in the
 *   fragment layouts of the instructions: one mma.m16n8k32.row.col.s32.s8.s8.s32 per block and
 *   stage when SG is 32, two mma.m16n8k16 (one per half, each with its own scale) when SG is 16.
 * - Elsewhere: work-item lid takes the dot of its NF_MMQ_RM weight-code rows with the staged code
 *   row of each of its NF_MMQ_TN / 16 tokens, 4-way packed with __dp4a on CUDA (sm_61+) and scalar int
 *   arithmetic elsewhere.
 * Both datapaths compute the integer stage dots exactly; they differ only in the activation chunk
 * (32 or 16 elements per d) and the fp32 combine order. Rows past M and tokens past N contribute 0
 * (their codes, d and sums load 0), so every work-item and warp reaches every barrier. K % BS == 0
 * is guaranteed by the host (gguf.ops.gemm.MmqGemmPlan), so every chunk record is whole.
 *
 * Launch contract: 2-D, global = {ceil(M / NF_MMQ_TM) * NF_MMQ_WG, ceil(N / NF_MMQ_TN)},
 * local = {NF_MMQ_WG, 1}; the host (gguf.ops.gemm.MmqGemmPlan with the device's
 * gguf.ops.gemm.MmqConfig) plans it. The _S twin keeps the geometry with the tile's token count
 * fixed at 16 (dimension 1 = ceil(N / 16)).
 *
 * NF_MMQ_GEMM_S(T, BS, BB, SG, CODE, SCALE, OFF) defines T##_MatMul_MMQ_S, the plain kernel over
 * the same TM x 16 x TK tile as the packed-dot path: one weight-row tile streams into one 16-token
 * output tile instead of NF_MMQ_TN, so the per-stage record gather and the mma / token-dot work
 * shrink with the tile (llama.cpp picks the J tile per launch from the batch size,
 * ggml_cuda_mmq_get_J_best). It is generated only when NF_MMQ_TN > 16 — when the device tile is
 * 16 tokens the plain kernel is already the small tile; otherwise the twin is a distinct kernel
 * whose name stays on every architecture that generates it (on CUDA segments below sm_80 the
 * tensor twin compiles out with the datapath, and the host then takes the plain kernel). The host
 * (gguf.ops.gemm.FastGemmLanes) selects it for staged launches of fewer tokens than the device
 * tile; the split-K plans keep the plain kernels.
 *
 * NF_MMQ_GEMM_KSPLIT(...) defines T##_MatMul_MMQ_KSplit, the split-K twin over the same tiles and
 * staged codes:
 *
 *   T_MatMul_MMQ_KSplit(i M, i K, i N, i q_stride, i q_off, i k_slice,
 *                       R uchar weight, R uint q8, W float partial)
 *   partial[(slice * N + t) * M + r] = sum_{k in slice} W[r][k] * x[t][k]
 *
 * with the slice index from the work-group id ({ceil(M / NF_MMQ_TM) * split groups along dimension 0,
 * slice = gid0 / ceil(M / NF_MMQ_TM)}), k in [slice * k_slice, min((slice + 1) * k_slice, K)) and every
 * live element of its [M x N] slice plane written (row stride M), so the host reduce
 * (LLM_SplitK_Reduce, kernels/llm/split_k_reduce.cl) sums the planes into the strided output. The
 * host (gguf.ops.gemm.SplitKGemmPlan) keeps k_slice a multiple of the weight block size BS, so a
 * slice starts at a block and chunk boundary. The launch geometry is the plain kernel's with
 * dimension 0 scaled by the split count.
 */
#ifndef NF_MATMUL_MMQ_FAMILY_H
#define NF_MATMUL_MMQ_FAMILY_H

/* NF_MMQ_TM, NF_MMQ_TN, NF_MMQ_TK, NF_MMQ_RM and NF_MMQ_RC come from common/matmul_mmq_config.h
 * (64 x 64 x 32 tensor, 64 x 16 packed dot, 64 x 128 RDNA3, 128 x 64 CDNA). */
#define NF_MMQ_TX (NF_MMQ_TM / NF_MMQ_RM)      /* row lanes of the tile */
#define NF_MMQ_WG (NF_MMQ_TX * 16)             /* work-items per workgroup: row lanes x token lanes */
#define NF_MMQ_SH (NF_MMQ_TK / 16)             /* 16-element halves per stage */
#define NF_MMQ_WS (NF_MMQ_TM * NF_MMQ_TK)      /* weight codes of one stage */
#define NF_MMQ_QS (NF_MMQ_TK / 4)              /* code ints of one chunk record */
/* Tile-derived counts of a TN-token tile: the generators take the tile's token count as a macro
 * argument (NF_MMQ_TN for the plain kernels, 16 for the _S twins), so every expression below
 * follows it. A work-item gathers TN / 16 tokens (the 16-token groups), a warp covers TN / 2
 * tokens as TN / 16 n8 blocks on the tensor path. */
#if NF_MMQ_TENSOR
#define NF_MMQ_ACC_OF(TN) (4 * ((TN) / 16))    /* accumulators per tensor-core lane */
#else
#define NF_MMQ_ACC_OF(TN) ((NF_MMQ_RM) * ((TN) / 16)) /* accumulators per work-item */
#endif
#define NF_MMQ_ACC NF_MMQ_ACC_OF(NF_MMQ_TN)

/* The chunk scale unit of a record read as a uint, bit-cast to float (the record holds the d value
 * of the quantize kernel in the unit after the code ints); unions, so every dialect parses it. */
NF_INLINE float nf_mmq_rec_d(uint v) {
    union { uint u; float f; } cast;
    cast.u = v;
    return cast.f;
}

NF_INLINE uint nf_mmq_float_rec(float v) {
    union { uint u; float f; } cast;
    cast.f = v;
    return cast.u;
}

/*
 * The software-pipelined stage. The caller owns the two local code tiles (wq, NF_MMQ_WS slots
 * each), the two (scale, offset) / code-sum tiles (wsc / wof / wsu, NF_MMQ_TM * NF_MMQ_SH entries
 * each), the per-token slots (xdv / xsu / xz) and acc; row_bytes and the type macros BS / BB /
 * CODE / SCALE / OFF come from the generator macro. Stages the weight codes and (scale, offset)
 * tiles of rows [m0, m0 + NF_MMQ_TM) and the chunk records of tokens [n0, n0 + TN) from q8 one
 * TK-deep chunk at a time, and calls the combine with the staged tile of the current chunk.
 *
 * The stage loop is software-pipelined so the weight stream never serializes on its load latency:
 * the code gather is tile-major (work-item slots run over the tile in row-major order, slot = s *
 * NF_MMQ_WG + lid, row slot / NF_MMQ_TK, so the lanes of a warp read one row's chunk codes as one
 * contiguous memory span), the codes of chunk c + NF_MMQ_TK load into registers while chunk c is
 * combined, and the commit writes the registers into the tile the combine reads. The two
 * code / (scale, offset) / code-sum tiles alternate per chunk; every lane passes the same three
 * barriers per iteration, with xz[0] uniform over the work-group (chunk c of token n0, or 0 past
 * N), so the code-sum fill under it stays uniform too. COMBINE is a macro invoked as
 * COMBINE(sel, XA, XB) with the pipe's last two arguments forwarded, combining the staged chunk of
 * buffer sel (its pointers are that buffer's offsets into wq / wsc / wof / wsu).
 *
 * NF_MMQ_PIPE(cstart, kend, TN, COMBINE, BS, BB, CODE, SCALE, OFF, XA, XB) runs the loop over
 * chunks [cstart, kend).
 */
#define NF_MMQ_SP (NF_MMQ_WS / NF_MMQ_WG)     /* code slots of one stage per work-item */

/* Loads chunk c's code slots and this work-item's (scale, offset) pair into the prefetch
 * registers; slots of rows past M and pairs of rows past M become zero. */
#define NF_MMQ_PREFETCH(c, m0, kend, M, weight, row_bytes, BS, BB, CODE, SCALE, OFF) \
    for (int s = 0; s < NF_MMQ_SP; s++) { \
        const int slot = s * NF_MMQ_WG + tx + NF_MMQ_TX * ty; \
        const int r = m0 + slot / NF_MMQ_TK; \
        pf[s] = (nf_i8)(r < M && (c) < (kend) \
            ? CODE(weight + (nf_i64)r * row_bytes + (nf_i64)((c) / (BS)) * (BB), \
                   ((c) + slot % NF_MMQ_TK) % (BS)) : 0); \
    } \
    psc = 0.0f; \
    pof = 0.0f; \
    if (lid < NF_MMQ_TM * NF_MMQ_SH && m0 + lid / NF_MMQ_SH < M && (c) < (kend)) { \
        const NF_GLOBAL uchar* sbp = weight + (nf_i64)(m0 + lid / NF_MMQ_SH) * row_bytes \
            + (nf_i64)((c) / (BS)) * (BB); \
        psc = SCALE(sbp, (c) + (lid % NF_MMQ_SH) * 16); \
        pof = OFF(sbp, (c) + (lid % NF_MMQ_SH) * 16); \
    }

/* Writes the prefetch registers into code / (scale, offset) tile sel. */
#define NF_MMQ_STAGE_COMMIT(sel) \
    for (int s = 0; s < NF_MMQ_SP; s++) { \
        const int slot = s * NF_MMQ_WG + tx + NF_MMQ_TX * ty; \
        wq[(sel) * NF_MMQ_WS + slot] = pf[s]; \
    } \
    if (lid < NF_MMQ_TM * NF_MMQ_SH) { \
        wsc[(sel) * NF_MMQ_TM * NF_MMQ_SH + lid] = psc; \
        wof[(sel) * NF_MMQ_TM * NF_MMQ_SH + lid] = pof; \
    }

/* Gathers chunk c's token records (the d, half code sums and code totals of tokens
 * [n0, n0 + TN)) and leaves q8_row0 at the chunk's first record. */
#define NF_MMQ_Q8_GATHER(c, TN, q8, q_stride, q_off, n0, N) \
    for (int j = 0; j < (TN) / 16; j++) { \
        const int tok = n0 + ty + 16 * j; \
        const int ok = tok < N; \
        const nf_i64 rec = (nf_i64)(q_off) + ((nf_i64)tok * (q_stride) + (c) / NF_MMQ_TK) * NF_MMQ_RC; \
        xdv[ty + 16 * j] = ok ? nf_mmq_rec_d(q8[rec + NF_MMQ_QS]) : 0.0f; \
        for (int h = 0; h < NF_MMQ_SH; h++) { \
            xsu[(ty + 16 * j) * NF_MMQ_SH + h] = ok ? (int)q8[rec + NF_MMQ_QS + 1 + h] : 0; \
        } \
        xz[ty + 16 * j] = ok ? (int)q8[rec + NF_MMQ_QS + 1 + NF_MMQ_SH] : 0; \
    } \
    q8_row0 = (q8) + (nf_i64)(q_off) + ((nf_i64)(n0) * (q_stride) + (c) / NF_MMQ_TK) * NF_MMQ_RC;

/* Fills the half code sums of tile sel from its codes, under the zero-total gate of the gather. */
#define NF_MMQ_WSUFILL(sel) \
    if (xz[0] != 0 && ty < NF_MMQ_TM / NF_MMQ_TX) { \
        const int sr = tx + NF_MMQ_TX * ty; \
        for (int h = 0; h < NF_MMQ_SH; h++) { \
            int sum = 0; \
            for (int kk = 0; kk < 16; kk++) { \
                sum += (int)wq[(sel) * NF_MMQ_WS + sr * NF_MMQ_TK + 16 * h + kk]; \
            } \
            wsu[(sel) * NF_MMQ_TM * NF_MMQ_SH + sr * NF_MMQ_SH + h] = sum; \
        } \
    }

#define NF_MMQ_PIPE(cstart, kend, TN, COMBINE, BS, BB, CODE, SCALE, OFF, XA, XB) \
    nf_i8 pf[NF_MMQ_SP]; \
    float psc = 0.0f; \
    float pof = 0.0f; \
    int sel = 0; \
    if ((cstart) < (kend)) { \
        NF_MMQ_PREFETCH((cstart), m0, (kend), M, weight, row_bytes, (BS), (BB), CODE, SCALE, OFF) \
        NF_MMQ_STAGE_COMMIT(0) \
        NF_MMQ_Q8_GATHER((cstart), (TN), q8, q_stride, q_off, n0, N) \
        barrier(CLK_LOCAL_MEM_FENCE); \
        NF_MMQ_WSUFILL(0) \
        barrier(CLK_LOCAL_MEM_FENCE); \
        COMBINE(0, XA, XB); \
        if ((cstart) + NF_MMQ_TK < (kend)) { \
            NF_MMQ_PREFETCH((cstart) + NF_MMQ_TK, m0, (kend), M, weight, row_bytes, (BS), (BB), CODE, SCALE, OFF) \
        } \
        sel = 1; \
        for (int c = (cstart) + NF_MMQ_TK; c < (kend); c += NF_MMQ_TK, sel ^= 1) { \
            barrier(CLK_LOCAL_MEM_FENCE); \
            NF_MMQ_Q8_GATHER(c, (TN), q8, q_stride, q_off, n0, N) \
            NF_MMQ_STAGE_COMMIT(sel) \
            if (c + NF_MMQ_TK < (kend)) { \
                NF_MMQ_PREFETCH((c) + NF_MMQ_TK, m0, (kend), M, weight, row_bytes, (BS), (BB), CODE, SCALE, OFF) \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
            NF_MMQ_WSUFILL(sel) \
            barrier(CLK_LOCAL_MEM_FENCE); \
            COMBINE(sel, XA, XB); \
        } \
    }

/*
 * Packed-dot path: dot of one 16-element half of a weight-code row with the staged code row of the
 * token, 4-way with __dp4a on CUDA sm_61+, scalar int elsewhere (HIP has no __dp4a, so its dialect
 * takes the scalar loop). rec points at the token's first code int; the halves of a 32-deep stage
 * carry their own scales, so the caller sums the half dots under those scales.
 */
NF_INLINE int nf_mmq_half_dot(NF_LOCAL_PTR const nf_i8* wq_row, NF_GLOBAL const uint* rec) {
#if defined(__CUDACC__) && !defined(__HIP__) \
    && (defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610)
    int ib = 0;
    for (int u = 0; u < 4; u++) {
        const int w = *(NF_LOCAL_PTR const int*)(wq_row + 4 * u);   /* rows of int-aligned tiles */
        ib = __dp4a(w, (int)rec[u], ib);
    }
    return ib;
#else
    int ib = 0;
    for (int kk = 0; kk < 16; kk++) {
        const uint packed = rec[kk >> 2] >> (8u * (kk & 3u));
        ib += (int)wq_row[kk] * (int)(uint)((packed & 0x7Fu) - (packed & 0x80u));
    }
    return ib;
#endif
}

#if NF_MMQ_TENSOR
/* One staged code word: 4 consecutive code bytes (element depth k + i in byte i), int-aligned. */
NF_INLINE uint nf_mmq_frag4(NF_GLOBAL const uint* rec, int tok_ok) {
    return tok_ok ? *rec : 0u;
}

/* d = a * b^T + d over one 16x8x32 int8 tile (row-major A fragment, column-major B fragment). */
NF_INLINE void nf_mmq_mma_m16n8k32(int d[4], const uint a[4], const uint b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* d = a * b^T + d over one 16x8x16 int8 tile (row-major A fragment, column-major B fragment). */
NF_INLINE void nf_mmq_mma_m16n8k16(int d[4], const uint a[2], uint b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b));
}

/*
 * The tensor path of one stage. Warp w covers rows (w >> 1) * 16 and the tn / 16 n8 token blocks
 * from (w & 1) * tn / 2 (tn, the tile's token count, is a compile-time constant at every call). Lane
 * l of block nb holds the accumulator positions (r, c), (r, c+1),
 * (r+8, c), (r+8, c+1) at r = (w >> 1) * 16 + (l >> 2), c = block start + (l & 3) * 2 in
 * acc[nb * 4 ..]; its A fragment registers carry rows r and r + 8 at depths (l & 3) * 4 (+16), its
 * B registers token block start + (l >> 2) at the same depths, loaded from the staged code records
 * (PTX mma fragment layouts for .s8, as llama.cpp's tile<16, 8, int> / tile<8, 8, int>).
 * SG == 32 takes one k32 product under the stage's single (scale, offset); SG == 16 takes one k16
 * product per half under that half's pair. xdv / xsu hold the tokens' d and code sums.
 */
NF_INLINE void nf_mmq_tensor_stage(NF_LOCAL_PTR const nf_i8* wq, NF_GLOBAL const uint* q8_row0,
                                   int q_stride, int n0, int tn, int N,
                                   NF_LOCAL_PTR const float* wsc, NF_LOCAL_PTR const float* wof,
                                   NF_LOCAL_PTR const int* wsu,
                                   NF_LOCAL_PTR const float* xdv, NF_LOCAL_PTR const int* xsu,
                                   NF_LOCAL_PTR const int* xz,
                                   int lid, float acc[], const int sg) {
    const int warp = lid >> 5, lane = lid & 31;
    const int r0 = (warp >> 1) * 16 + (lane >> 2);
    const int kf = (lane & 3) * 4;
    const int rr[4] = {r0, r0, r0 + 8, r0 + 8};
    uint a[4];
    a[0] = *(NF_LOCAL_PTR const uint*)(wq + r0 * NF_MMQ_TK + kf);
    a[1] = *(NF_LOCAL_PTR const uint*)(wq + (r0 + 8) * NF_MMQ_TK + kf);
    a[2] = *(NF_LOCAL_PTR const uint*)(wq + r0 * NF_MMQ_TK + kf + 16);
    a[3] = *(NF_LOCAL_PTR const uint*)(wq + (r0 + 8) * NF_MMQ_TK + kf + 16);
    for (int nb = 0; nb < tn / 16; nb++) {
        const int cbase = (warp & 1) * (tn / 2) + nb * 8;
        const int c0 = cbase + (lane & 3) * 2;
        const int cb = cbase + (lane >> 2);
        const int cc[4] = {c0, c0 + 1, c0, c0 + 1};
        /* the staged code words of this lane's token: depths (l & 3) * 4 (+16); tokens past N load 0 */
        const int tokok = n0 + cb < N;
        NF_GLOBAL const uint* rec = q8_row0 + (nf_i64)cb * (q_stride * NF_MMQ_RC);
        const uint b[2] = {nf_mmq_frag4(rec + kf / 4, tokok), tokok ? rec[kf / 4 + NF_MMQ_TK / 8] : 0u};
        float sd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (sg == 32) {
            int d[4] = {0, 0, 0, 0};
            nf_mmq_mma_m16n8k32(d, a, b);
            for (int i = 0; i < 4; i++) {
                const int si = rr[i] * NF_MMQ_SH, xi = cc[i] * NF_MMQ_SH;
                sd[i] = wsc[si] * (float)d[i] + wof[si] * (float)(xsu[xi] + xsu[xi + 1])
                        - (float)xz[cc[i]] * (wsc[si] * (float)(wsu[si] + wsu[si + 1])
                            + wof[si] * (float)(2 * 16));
            }
        } else {
            for (int h = 0; h < NF_MMQ_SH; h++) {
                const uint ah[2] = {a[2 * h], a[2 * h + 1]};
                int d[4] = {0, 0, 0, 0};
                nf_mmq_mma_m16n8k16(d, ah, b[h]);
                for (int i = 0; i < 4; i++) {
                    const int si = rr[i] * NF_MMQ_SH + h;
                    sd[i] += wsc[si] * (float)d[i] + wof[si] * (float)xsu[cc[i] * NF_MMQ_SH + h]
                            - (float)xz[cc[i]] * (wsc[si] * (float)wsu[si] + wof[si] * (float)16);
                }
            }
        }
        for (int i = 0; i < 4; i++) {
            acc[nb * 4 + i] += xdv[cc[i]] * sd[i];
        }
    }
}
#endif /* NF_MMQ_TENSOR */

/*
 * One stage's integer dots and their (scale, offset) / d combine into acc. Tensor path:
 * nf_mmq_tensor_stage. Packed-dot path (one 16-element half per stage): work-item lid accumulates
 * rows tx + NF_MMQ_TX*q against each of its tn / 16 tokens ty + 16*j, reading the token's staged
 * codes directly; acc[j * NF_MMQ_RM + q] holds the result.
 */
NF_INLINE void nf_mmq_combine(NF_LOCAL_PTR const nf_i8* wq, NF_GLOBAL const uint* q8_row0,
                              int q_stride, int n0, int tn, int N,
                              NF_LOCAL_PTR const float* wsc, NF_LOCAL_PTR const float* wof,
                              NF_LOCAL_PTR const int* wsu,
                              NF_LOCAL_PTR const float* xdv, NF_LOCAL_PTR const int* xsu,
                              NF_LOCAL_PTR const int* xz,
                              int lid, float acc[], const int sg) {
#if NF_MMQ_TENSOR
    nf_mmq_tensor_stage(wq, q8_row0, q_stride, n0, tn, N, wsc, wof, wsu, xdv, xsu, xz, lid, acc, sg);
#else
    const int tx = lid % NF_MMQ_TX, ty = lid / NF_MMQ_TX;
    for (int j = 0; j < tn / 16; j++) {
        const int tok = ty + 16 * j;
        if (n0 + tok >= N) {
            continue;
        }
        NF_GLOBAL const uint* rec = q8_row0 + (nf_i64)tok * (q_stride * NF_MMQ_RC);
        for (int q = 0; q < NF_MMQ_RM; q++) {
            const int sr = tx + NF_MMQ_TX * q;
            float sd = 0.0f;
            float corr = 0.0f;
            for (int h = 0; h < NF_MMQ_SH; h++) {
                const int ib = nf_mmq_half_dot(wq + sr * NF_MMQ_TK + 16 * h, rec + 4 * h);
                sd += wsc[sr * NF_MMQ_SH + h] * (float)ib
                        + wof[sr * NF_MMQ_SH + h] * (float)xsu[tok * NF_MMQ_SH + h];
                corr += wsc[sr * NF_MMQ_SH + h] * (float)wsu[sr * NF_MMQ_SH + h]
                        + wof[sr * NF_MMQ_SH + h] * (float)16;
            }
            acc[j * NF_MMQ_RM + q] += xdv[tok] * (sd - (float)xz[tok] * corr);
        }
    }
#endif
}

/* The stage results of one work-item (or lane) to global memory, bounds-checked per element; tn is
 * the tile's token count. */
NF_INLINE void nf_mmq_store(NF_GLOBAL float* NF_RESTRICT y, int M, int N,
                            int y_stride, nf_i64 y_off, int m0, int n0, int tn, int lid, const float acc[]) {
#if NF_MMQ_TENSOR
    const int warp = lid >> 5, lane = lid & 31;
    const int r = m0 + (warp >> 1) * 16 + (lane >> 2);
    for (int nb = 0; nb < tn / 16; nb++) {
        const int c = n0 + (warp & 1) * (tn / 2) + nb * 8 + (lane & 3) * 2;
        const float* v = acc + nb * 4;
        if (r < M && c < N) y[(nf_i64)y_off + (nf_i64)c * y_stride + r] = v[0];
        if (r < M && c + 1 < N) y[(nf_i64)y_off + (nf_i64)(c + 1) * y_stride + r] = v[1];
        if (r + 8 < M && c < N) y[(nf_i64)y_off + (nf_i64)c * y_stride + r + 8] = v[2];
        if (r + 8 < M && c + 1 < N) y[(nf_i64)y_off + (nf_i64)(c + 1) * y_stride + r + 8] = v[3];
    }
#else
    const int tx = lid % NF_MMQ_TX, ty = lid / NF_MMQ_TX;
    for (int j = 0; j < tn / 16; j++) {
        const int tok = n0 + ty + 16 * j;
        if (tok >= N) {
            continue;
        }
        const nf_i64 ybase = (nf_i64)y_off + (nf_i64)tok * y_stride;
        for (int q = 0; q < NF_MMQ_RM; q++) {
            const int r = m0 + tx + NF_MMQ_TX * q;
            if (r < M) y[ybase + r] = acc[j * NF_MMQ_RM + q];
        }
    }
#endif
}

/* ---- kernel generator ---- */

/*
 * The code tiles are declared as int arrays so every 4-byte fragment / dp4a load is aligned; the
 * (scale, offset) tiles hold NF_MMQ_SH pairs per row, and the tokens' d and code sums are gathered
 * from the staged records once per stage. The code / (scale, offset) / code-sum tiles exist in two
 * copies, one per in-flight chunk of the pipelined stage (NF_MMQ_PIPE). row_bytes and the staging
 * arrays belong to the kernel; TN is the kernel's token tile (the per-token arrays size to it).
 */
#define NF_MMQ_KERNEL_BODY(TN, BS, BB, GEMM_ARGS) \
    const int lid = get_local_id(0); \
    const int tx = lid % NF_MMQ_TX, ty = lid / NF_MMQ_TX; \
    const nf_i64 row_bytes = (nf_i64)(K / (BS)) * (BB); \
    NF_LOCAL int wq_w[2 * NF_MMQ_WS / 4]; \
    NF_LOCAL float wsc[2 * NF_MMQ_TM * NF_MMQ_SH]; \
    NF_LOCAL float wof[2 * NF_MMQ_TM * NF_MMQ_SH]; \
    NF_LOCAL float xdv[(TN)]; \
    NF_LOCAL int xsu[(TN) * NF_MMQ_SH]; \
    NF_LOCAL int xz[(TN)]; \
    NF_LOCAL int wsu[2 * NF_MMQ_TM * NF_MMQ_SH]; \
    NF_LOCAL_PTR nf_i8* wq = (NF_LOCAL_PTR nf_i8*)wq_w; \
    NF_GLOBAL const uint* q8_row0; \
    float acc[NF_MMQ_ACC_OF(TN)]; \
    for (int i = 0; i < NF_MMQ_ACC_OF(TN); i++) \
        acc[i] = 0.0f; \
    GEMM_ARGS

/*
 * The GEMM kernel of one NAME over the TN-token tile: the pipelined stage loop combines the staged
 * weight tile against the tile's tokens and stores the tile's results (the _S twin takes TN = 16,
 * the plain kernels the device tile NF_MMQ_TN).
 */
#define NF_MMQ_INT8_COMBINE(sel, TN, SG) \
    nf_mmq_combine(wq + (sel) * NF_MMQ_WS, q8_row0, q_stride, n0, (TN), N, \
                   wsc + (sel) * NF_MMQ_TM * NF_MMQ_SH, wof + (sel) * NF_MMQ_TM * NF_MMQ_SH, \
                   wsu + (sel) * NF_MMQ_TM * NF_MMQ_SH, xdv, xsu, xz, lid, acc, (SG))

#define NF_MMQ_GEMM_TN(NAME, TN, BS, BB, SG, CODE, SCALE, OFF) \
NF_KERNEL void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT y) { \
    NF_MMQ_KERNEL_BODY((TN), (BS), (BB), \
        const int m0 = (int)get_group_id(0) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * (TN); \
        NF_MMQ_PIPE(0, K, (TN), NF_MMQ_INT8_COMBINE, (BS), (BB), CODE, SCALE, OFF, (TN), (SG)) \
        nf_mmq_store(y, M, N, y_stride, (nf_i64)y_off, m0, n0, (TN), lid, acc); \
    ) \
}

#define NF_MMQ_GEMM(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_GEMM_TN(T##_MatMul_MMQ, NF_MMQ_TN, BS, BB, SG, CODE, SCALE, OFF)

#define NF_MMQ_GEMM_S(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_GEMM_TN(T##_MatMul_MMQ_S, 16, BS, BB, SG, CODE, SCALE, OFF)

/*
 * The split-K twin: dimension 0 of the dispatch holds ceil(M / 64) * split workgroups, group
 * gid0 = rt + rtiles * slice covers rows rt * 64 of slice `slice` over k in
 * [slice * k_slice, min((slice + 1) * k_slice, K)) and writes the live [M x N] positions of its
 * slice plane at partial[(slice * N + t) * M + r] (y_stride = M), so the host reduce
 * (LLM_SplitK_Reduce, kernels/llm/split_k_reduce.cl) sums the planes into the strided output. The
 * launch geometry is the plain kernel's with dimension 0 scaled by the split count
 * (gguf.ops.gemm.SplitKGemmPlan).
 */
#define NF_MMQ_GEMM_KSPLIT(T, BS, BB, SG, CODE, SCALE, OFF) \
NF_KERNEL void T##_MatMul_MMQ_KSplit( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int k_slice, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT partial) { \
    NF_MMQ_KERNEL_BODY(NF_MMQ_TN, (BS), (BB), \
        const int rtiles = (M + NF_MMQ_TM - 1) / NF_MMQ_TM; \
        const int slice = (int)get_group_id(0) / rtiles; \
        const int m0 = ((int)get_group_id(0) - slice * rtiles) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * NF_MMQ_TN; \
        const int k0 = slice * k_slice; \
        const int k1 = k0 + k_slice < K ? k0 + k_slice : K; \
        NF_MMQ_PIPE(k0, k1, NF_MMQ_TN, NF_MMQ_INT8_COMBINE, (BS), (BB), CODE, SCALE, OFF, NF_MMQ_TN, (SG)) \
        nf_mmq_store(partial, M, N, M, (nf_i64)slice * M * N, m0, n0, NF_MMQ_TN, lid, acc); \
    ) \
}

/* ---- E4M3 datapath: FP8 weights and 4-bit weights served as E4M3 ----
 *
 * The staged records hold E4M3 codes with one chunk scale instead of int8 codes with code sums
 * (kernels/llm/matmul_mmq.cl): the weight codes are the FP8 type's own bytes for the FP8 weight types,
 * or the E4M3 codes the 4-bit type's code values map to (exact for the E2M1 codebooks). The stage dot
 * is then a product of E4M3 codes, and the chunk scale of the weight block and of the activation
 * record multiply the accumulated dot in fp32; no offset term applies to these types.
 */

/* d = a * b^T + d over one 16x8x32 E4M3 tile (row-major A fragment, column-major B fragment). */
NF_INLINE void nf_fp8_mma_m16n8k32(float d[4], const uint a[4], const uint b[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

/* Scalar E4M3 dot of one NF_MMQ_TK-deep row of staged weight codes with a token's staged codes: the
 * datapath of the packed-dot configs and of the architectures below sm_89. */
NF_INLINE float nf_fp8_stage_dot(NF_LOCAL_PTR const nf_i8* wq_row, NF_GLOBAL const uint* rec) {
    float sum = 0.0f;
    for (int kk = 0; kk < NF_MMQ_TK; kk++) {
        const uint packed = rec[kk >> 2];
        const uint xc = (packed >> (8u * (kk & 3u))) & 0xFFu;
        sum += F8_E4M3_LUT[(uint)(uchar)wq_row[kk]] * F8_E4M3_LUT[xc];
    }
    return sum;
}

/* One stage's E4M3 tile products and their scales. Tensor path: the sm_89+ mma over the fragments (the
 * same layouts as the int8 stage), or the scalar dot below sm_89; packed path: one scalar dot per row.
 * tn is the tile's token count (nf_mmq_tensor_stage's). */
NF_INLINE void nf_fp8_combine(NF_LOCAL_PTR const nf_i8* wq, NF_GLOBAL const uint* q8_row0,
                              int q_stride, int n0, int tn, int N,
                              NF_LOCAL_PTR const float* wsc, NF_LOCAL_PTR const float* xdv,
                              int lid, float acc[]) {
#if NF_MMQ_TENSOR
    const int warp = lid >> 5, lane = lid & 31;
    const int r0 = (warp >> 1) * 16 + (lane >> 2);
    const int kf = (lane & 3) * 4;
    const int rr[4] = {r0, r0, r0 + 8, r0 + 8};
    uint a[4];
    a[0] = *(NF_LOCAL_PTR const uint*)(wq + r0 * NF_MMQ_TK + kf);
    a[1] = *(NF_LOCAL_PTR const uint*)(wq + (r0 + 8) * NF_MMQ_TK + kf);
    a[2] = *(NF_LOCAL_PTR const uint*)(wq + r0 * NF_MMQ_TK + kf + 16);
    a[3] = *(NF_LOCAL_PTR const uint*)(wq + (r0 + 8) * NF_MMQ_TK + kf + 16);
    for (int nb = 0; nb < tn / 16; nb++) {
        const int cbase = (warp & 1) * (tn / 2) + nb * 8;
        const int c0 = cbase + (lane & 3) * 2;
        const int cb = cbase + (lane >> 2);
        const int cc[4] = {c0, c0 + 1, c0, c0 + 1};
        const int tokok = n0 + cb < N;
        NF_GLOBAL const uint* rec = q8_row0 + (nf_i64)cb * (q_stride * NF_MMQ_RC);
        float sd[4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        const uint b[2] = {nf_mmq_frag4(rec + kf / 4, tokok), tokok ? rec[kf / 4 + NF_MMQ_TK / 8] : 0u};
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        nf_fp8_mma_m16n8k32(d, a, b);
        for (int i = 0; i < 4; i++) {
            sd[i] = wsc[rr[i] * NF_MMQ_SH] * d[i];
        }
#else
        for (int i = 0; i < 4; i++) {
            const int tok = n0 + cc[i];
            NF_GLOBAL const uint* trec = q8_row0 + (nf_i64)cc[i] * (q_stride * NF_MMQ_RC);
            sd[i] = tok < N ? wsc[rr[i] * NF_MMQ_SH] * nf_fp8_stage_dot(wq + rr[i] * NF_MMQ_TK, trec) : 0.0f;
        }
#endif
        for (int i = 0; i < 4; i++) {
            acc[nb * 4 + i] += xdv[cc[i]] * sd[i];
        }
    }
#else
    const int tx = lid % NF_MMQ_TX, ty = lid / NF_MMQ_TX;
    for (int j = 0; j < tn / 16; j++) {
        const int tok = ty + 16 * j;
        if (n0 + tok >= N) {
            continue;
        }
        NF_GLOBAL const uint* rec = q8_row0 + (nf_i64)tok * (q_stride * NF_MMQ_RC);
        for (int q = 0; q < NF_MMQ_RM; q++) {
            const int sr = tx + NF_MMQ_TX * q;
            acc[j * NF_MMQ_RM + q] += xdv[tok] * (wsc[sr * NF_MMQ_SH]
                    * nf_fp8_stage_dot(wq + sr * NF_MMQ_TK, rec));
        }
    }
#endif
}

/* The weight offset of the E4M3 lane: every FP8 and 4-bit type it serves is symmetric, and the staged
 * activation records carry a zero zero-point, so no offset term applies. */
#define NF_FP8Q_NO_OFFSET(bp, k0) 0.0f

/* The E4M3 kernel generator: the int8 generator's tiling, staging and store, with nf_fp8_combine. The
 * staged records carry E4M3 codes and one scale per chunk (kernels/llm/quant_mmq_q8_1.cl), so CODE
 * yields the code byte of element e and SCALE the scale of the block holding k0. */
#define NF_FP8Q_KERNEL_BODY(TN, BS, BB, GEMM_ARGS) \
    NF_MMQ_KERNEL_BODY((TN), (BS), (BB), GEMM_ARGS)

/* The E4M3 combine of buffer sel: no offset term, so wof / wsu / xz of the staged chunk are dead;
 * the third argument is the pipe's forwarded tail, unused here. */
#define NF_FP8Q_COMBINE(sel, TN, SG) \
    nf_fp8_combine(wq + (sel) * NF_MMQ_WS, q8_row0, q_stride, n0, (TN), N, \
                   wsc + (sel) * NF_MMQ_TM * NF_MMQ_SH, xdv, lid, acc)

#define NF_FP8Q_GEMM_TN_NAMED(NAME, TN, BS, BB, CODE, SCALE) \
NF_KERNEL void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT y) { \
    NF_FP8Q_KERNEL_BODY((TN), (BS), (BB), \
        const int m0 = (int)get_group_id(0) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * (TN); \
        NF_MMQ_PIPE(0, K, (TN), NF_FP8Q_COMBINE, (BS), (BB), CODE, SCALE, NF_FP8Q_NO_OFFSET, (TN), 0) \
        nf_mmq_store(y, M, N, y_stride, (nf_i64)y_off, m0, n0, (TN), lid, acc); \
    ) \
}

#define NF_FP8Q_GEMM_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_TN_NAMED(NAME, NF_MMQ_TN, BS, BB, CODE, SCALE)

#define NF_FP8Q_GEMM_S_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_TN_NAMED(NAME, 16, BS, BB, CODE, SCALE)

#define NF_FP8Q_GEMM_KSPLIT_NAMED(NAME, BS, BB, CODE, SCALE) \
NF_KERNEL void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int k_slice, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT partial) { \
    NF_FP8Q_KERNEL_BODY(NF_MMQ_TN, (BS), (BB), \
        const int rtiles = (M + NF_MMQ_TM - 1) / NF_MMQ_TM; \
        const int slice = (int)get_group_id(0) / rtiles; \
        const int m0 = ((int)get_group_id(0) - slice * rtiles) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * NF_MMQ_TN; \
        const int k0 = slice * k_slice; \
        const int k1 = k0 + k_slice < K ? k0 + k_slice : K; \
        NF_MMQ_PIPE(k0, k1, NF_MMQ_TN, NF_FP8Q_COMBINE, (BS), (BB), CODE, SCALE, NF_FP8Q_NO_OFFSET, NF_MMQ_TN, 0) \
        nf_mmq_store(partial, M, N, M, (nf_i64)slice * M * N, m0, n0, NF_MMQ_TN, lid, acc); \
    ) \
}

#define NF_FP8Q_GEMM(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_NAMED(T##_MatMul_MMQ, BS, BB, CODE, SCALE)
#define NF_FP8Q_GEMM_S(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_S_NAMED(T##_MatMul_MMQ_S, BS, BB, CODE, SCALE)
#define NF_FP8Q_GEMM_KSPLIT(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_NAMED(T##_MatMul_MMQ_KSplit, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_NAMED(T##_MatMul_W4A8, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM_S(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_S_NAMED(T##_MatMul_W4A8_S, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM_KSPLIT(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_NAMED(T##_MatMul_W4A8_KSplit, BS, BB, CODE, SCALE)

#endif /* NF_MATMUL_MMQ_FAMILY_H */
