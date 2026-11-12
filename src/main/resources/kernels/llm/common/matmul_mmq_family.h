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
 * common/quant_decode_iq.h. Every generator also pastes CODE4(bp, e0), the 4 consecutive codes
 * e0..e0+3 (e0 a multiple of 4) packed little-endian into one uint, as NF_MMQ_PREFETCH loads the
 * weight stream 4 codes per access; the macros below the per-type CODE define it. Codes are stored
 * signed; every code range (Q4_0 0..15, the IQ grid bytes up to +-62, up to Q8_0 -128..127) fits
 * nf_i8, so a stage dot of code pairs stays within int (127 * 127 * 32 << 2^31).
 *
 * Tiling (common/matmul_mmq_config.h): a workgroup of NF_MMQ_WG work-items computes a tile of
 * NF_MMQ_TM weight rows x NF_MMQ_TN tokens in NF_MMQ_TK-deep stages: 128 x 128 x 32 on the tensor
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
 * fixed at 16 (dimension 1 = ceil(N / 16)) and the row tile of the packed-dot family: where the
 * device tile is wider, its generators run with NF_MMQ_TM capped at NF_MMQ_RM * 16 rows so the twin
 * workgroup stays at 256 work-items (the measured RTX 5080 Laptop small steps lose their second
 * block per SM to the 512-item workgroup of the uncapped 128-row twin).
 *
 * NF_MMQ_GEMM_S(T, BS, BB, SG, CODE, SCALE, OFF) defines T##_MatMul_MMQ_S, the plain kernel over
 * the capped TM x 16 x TK tile: one weight-row tile streams into one 16-token output tile instead
 * of NF_MMQ_TN, so the per-stage record gather and the mma / token-dot work shrink with the tile
 * (llama.cpp picks the J tile per launch from the batch size, ggml_cuda_mmq_get_J_best). It is
 * generated only when NF_MMQ_TN > 16 — when the device tile is 16 tokens the plain kernel is
 * already the small tile; otherwise the twin is a distinct kernel whose name stays on every
 * architecture that generates it (on CUDA segments below sm_80 the tensor twin compiles out with
 * the datapath, and the host then takes the plain kernel). The host (gguf.ops.gemm.FastGemmLanes)
 * selects it for staged launches of fewer tokens than the device tile.
 *
 * NF_MMQ_GEMM_S64(T, ...) defines T##_MatMul_MMQ_S64, the same twin form over the capped
 * TM x 64 x TK tile: the J=64 row of llama.cpp's per-launch ladder, the tile the mid batches of
 * 17..64 tokens run (one weight pass over one 64-token output tile; the measured 128-token device
 * tile pads up to 80 dead token columns there and its 512-item workgroup halves the resident
 * blocks). NF_MMQ_GEMM_SKSPLIT64(T, ...) is its split-K twin, like SKSPLIT over 64 tokens.
 *
 * NF_MMQ_GEMM_KSPLIT(...) defines T##_MatMul_MMQ_KSplit, the split-K twin over the plain tiles and
 * staged codes; NF_MMQ_GEMM_SKSPLIT(...) defines T##_MatMul_MMQ_SKSplit, the same split over the
 * capped 16-token twin tile for launches of fewer tokens than the device tile:
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
 * slice starts at a block and chunk boundary. The launch geometry is the tile's plain kernel's with
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
 * The software-pipelined stage. The caller owns NF_MMQ_PIPE_BUF local code tiles (wq, NF_MMQ_WS
 * slots each), NF_MMQ_PIPE_BUF (scale, offset) / code-sum tiles (wsc / wof / wsu, NF_MMQ_TM *
 * NF_MMQ_SH entries each), the shared record tile xr ((TN) * NF_MMQ_RC units, refilled every
 * stage) and acc; row_bytes and the type macros BS / BB / CODE / SCALE / OFF come from the
 * generator macro. Stages the weight codes and (scale, offset) tiles of rows [m0, m0 +
 * NF_MMQ_TM) and the chunk records of tokens [n0, n0 + TN) from q8 one TK-deep chunk at a time,
 * and calls the combine with the staged tile of the current chunk.
 *
 * The stage loop is software-pipelined so the weight stream stays in flight: the weight gather is
 * tile-major (work-item slots run over the tile in row-major 4-code words, slot4 = s * NF_MMQ_WG +
 * lid, row slot4 / (NF_MMQ_TK / 4), so the lanes of a warp read one row's chunk codes as one
 * contiguous span of aligned words instead of one byte load per code), the codes of the two chunks
 * ahead load into registers while chunk c is combined, and the commit writes a register set into
 * the tile the combine reads one iteration later. The gather of the token records runs coalesced
 * (the tile's records are consecutive in q8) into the shared record tile between the barriers, so
 * every combine reads its B fragments, scales and code sums from shared memory instead of q8
 * global. The NF_MMQ_PIPE_BUF code / (scale, offset) / code-sum tiles cycle per chunk; every lane
 * passes the same two barriers per iteration, plus a third after the code-sum fill when the gathered
 * records carry zero points: the record tile is uniform over the work-group after the second
 * barrier, so the fill and its barrier under the token 0 zero-point gate are taken by every lane or
 * by none. COMBINE is a macro invoked as COMBINE(sel, XA, XB) with the pipe's last two arguments
 * forwarded, combining the staged chunk of buffer sel (its pointers are that buffer's offsets into
 * wq / wsc / wof / wsu; the record tile arrives through the pipe's xr alias).
 *
 * NF_MMQ_PIPE(cstart, kend, TN, COMBINE, BS, BB, CODE, SCALE, OFF, XA, XB) runs the loop over
 * chunks [cstart, kend).
 */
#define NF_MMQ_SP (NF_MMQ_WS / NF_MMQ_WG)     /* code slots of one stage per work-item */
#define NF_MMQ_SP4 (NF_MMQ_SP / 4)            /* 4-code words of one stage per work-item */
#define NF_MMQ_PIPE_BUF 3                     /* code / (scale, offset) / code-sum tiles */
#define NF_MMQ_PIPE_SETS 2                    /* prefetch register sets: two chunks in flight */

/*
 * The stage's weight stream reads 4 codes per access: every type defines CODE4(bp, e0), the codes
 * of elements e0..e0+3 (e0 a multiple of 4, the group never crossing a block) packed little-endian,
 * loaded with the aligned words the block layout proves (common/quant_blocks.h nf_read_u32 /
 * nf_read_u32_u16; types whose codes are not byte-contiguous fall back to packing 4 CODE calls).
 * These compose the packed forms the per-type CODE4 macros use.
 */

/* Four code values as their low bytes, little-endian: the word the code tile stores. */
#define NF_MMQ_CODE4_PACK(c0, c1, c2, c3) \
    ((uint)(uchar)(nf_i8)(c0) | ((uint)(uchar)(nf_i8)(c1) << 8) \
        | ((uint)(uchar)(nf_i8)(c2) << 16) | ((uint)(uchar)(nf_i8)(c3) << 24))

/* The 4-bit codes e0..e0+3 of a word holding 4 consecutive qs bytes: code j is the nibble of byte j
 * at `shift` (the low nibbles of the 16-code run when shift is 0, the high ones when it is 4). The
 * masked form puts each byte's nibble in its own result byte in two ALU ops, where the packed form
 * of the four nibbles costs about twelve; the byte layout matches NF_MMQ_CODE4_PACK's. */
#define NF_MMQ_CODE4_NIBBLES(word, shift) \
    (((word) >> (shift)) & 0x0F0F0F0Fu)

/* The bits e0..e0+3 of a 32-bit word as 0x10 per code lane: the 5th bits of 4 codes of one qh word
 * (e0 a multiple of 4 with e0 + 3 < 32, as the stage depth guarantees). */
#define NF_MMQ_HBITS4(word, e0) \
    ((((word) >> (e0)) & 0x1u) << 4 | (((word) >> ((e0) + 1)) & 0x1u) << 12 \
        | (((word) >> ((e0) + 2)) & 0x1u) << 20 | (((word) >> ((e0) + 3)) & 0x1u) << 28)

/* Bit b of each of the word's 4 bytes as 0x10 per code lane: the 5th bits of 4 codes whose qh
 * bytes the word covers (one qh byte per element, the group's bit b constant across the group). */
#define NF_MMQ_HBITS4_B(word, b) \
    ((((word) >> (b)) & 0x1u) << 4 | (((word) >> (8 + (b))) & 0x1u) << 12 \
        | (((word) >> (16 + (b))) & 0x1u) << 20 | (((word) >> (24 + (b))) & 0x1u) << 28)

/*
 * Loads chunk c's code words and this work-item's (scale, offset) pair into prefetch register set
 * `set`; words of rows past M and pairs of rows past M become zero. Work-item slots run over the
 * tile in row-major 4-code groups, slot4 = s * NF_MMQ_WG + lid, row slot4 / (NF_MMQ_TK / 4), so the
 * lanes of a warp read one row's chunk codes as one contiguous span and each lane's word load is
 * aligned (the slot's first code stays 4-aligned: NF_MMQ_TK is a multiple of 4 and c a multiple of
 * NF_MMQ_TK).
 */
#define NF_MMQ_PREFETCH(set, c, m0, kend, M, weight, row_bytes, BS, BB, CODE, SCALE, OFF) \
    for (int s = 0; s < NF_MMQ_SP4; s++) { \
        const int slot4 = s * NF_MMQ_WG + tx + NF_MMQ_TX * ty; \
        const int r = m0 + slot4 / (NF_MMQ_TK / 4); \
        pfw[set][s] = r < M && (c) < (kend) \
            ? CODE##4(weight + (nf_i64)r * row_bytes + (nf_i64)((c) / (BS)) * (BB), \
                      ((c) + 4 * (slot4 % (NF_MMQ_TK / 4))) % (BS)) : 0u; \
    } \
    psc[set] = 0.0f; \
    pof[set] = 0.0f; \
    if (lid < NF_MMQ_TM * NF_MMQ_SH && m0 + lid / NF_MMQ_SH < M && (c) < (kend)) { \
        const NF_GLOBAL uchar* sbp = weight + (nf_i64)(m0 + lid / NF_MMQ_SH) * row_bytes \
            + (nf_i64)((c) / (BS)) * (BB); \
        psc[set] = SCALE(sbp, (c) + (lid % NF_MMQ_SH) * 16); \
        pof[set] = OFF(sbp, (c) + (lid % NF_MMQ_SH) * 16); \
    }

/* Writes prefetch register set `set` into code / (scale, offset) tile sel (the code tile stays the
 * same byte layout the combine reads: each word covers the 4 codes its bytes held before). */
#define NF_MMQ_STAGE_COMMIT(set, sel) \
    for (int s = 0; s < NF_MMQ_SP4; s++) { \
        const int slot4 = s * NF_MMQ_WG + tx + NF_MMQ_TX * ty; \
        wq_w[(sel) * (NF_MMQ_WS / 4) + slot4] = (int)pfw[set][s]; \
    } \
    if (lid < NF_MMQ_TM * NF_MMQ_SH) { \
        wsc[(sel) * NF_MMQ_TM * NF_MMQ_SH + lid] = psc[set]; \
        wof[(sel) * NF_MMQ_TM * NF_MMQ_SH + lid] = pof[set]; \
    }

/* Gathers chunk c's records of tokens [n0, n0 + TN) into a shared record tile, one record of
 * NF_MMQ_RC units (the tk / 4 code ints, the chunk scale d, the half code sums and the zero point)
 * per token: the records of consecutive tokens are consecutive in q8, so the work-group streams
 * them coalesced. Records of tokens past N stage zero, so every consumer reads the tile without a
 * bounds test (a zero record's scale bits are 0.0f and its zero point 0). */
#define NF_MMQ_Q8_GATHER_SYNC(dst, c, TN, q8, q_stride, q_off, n0, N) \
    for (int i = lid; i < (TN) * NF_MMQ_RC; i += NF_MMQ_WG) { \
        const int tok = i / NF_MMQ_RC; \
        dst[i] = n0 + tok < (N) \
            ? (q8)[(nf_i64)(q_off) + ((nf_i64)(n0 + tok) * (q_stride) + (c) / NF_MMQ_TK) * NF_MMQ_RC \
                   + i % NF_MMQ_RC] : 0u; \
    }

/* The staged record tile of a work-group: one synchronous tile on every dialect. A cp.async
 * double-buffered gather (two record tiles + wait_group) was measured on the RTX 5080 Laptop at
 * 3.3x the time of this form: the second tile pushed the kernel past the 3-block shared-memory
 * fit of the SM and the resident warps fell from 1536 to 896, which the weight stream's latency
 * hiding never recovers. The synchronous tile keeps the occupancy; its gather latency hides under
 * the other warps' weight loads at the barrier. */
#define NF_MMQ_XR_BUF 1

/* The record tile buffer j holds: the single synchronous tile everywhere. */
#define NF_MMQ_XR_AT(j, TN) (xrs)

/*
 * The record-tile phases of the pipe: every dialect refills the single tile synchronously between
 * the barriers, and XR_READY is empty (the second barrier retires the gather).
 */
#define NF_MMQ_XR_PROLOGUE(cstart, kend, TN, q8, q_stride, q_off, n0, N) \
    NF_MMQ_Q8_GATHER_SYNC(xrs, (cstart), (TN), q8, q_stride, q_off, n0, N)
#define NF_MMQ_XR_STEP(c, kend, j, TN, q8, q_stride, q_off, n0, N) \
    NF_MMQ_Q8_GATHER_SYNC(xrs, (c), (TN), q8, q_stride, q_off, n0, N)
#define NF_MMQ_XR_READY

/* Fills the half code sums of tile sel from its codes and makes them visible to the combine, under the
 * zero-total gate of the gather (token 0's zero-point unit). Every work-item reads the gate from the
 * record tile after the barrier that retired the gather, so the gate is uniform over the work-group and
 * the barrier inside it is reached by all or none: symmetric records (every staging but the asymmetric
 * int8 one) write no sums and skip the barrier. */
#define NF_MMQ_WSUFILL(sel) \
    if (xr[NF_MMQ_QS + 1 + NF_MMQ_SH] != 0u) { \
        if (ty < NF_MMQ_TM / NF_MMQ_TX) { \
            const int sr = tx + NF_MMQ_TX * ty; \
            for (int h = 0; h < NF_MMQ_SH; h++) { \
                int sum = 0; \
                for (int kk = 0; kk < 16; kk++) { \
                    sum += (int)wq[(sel) * NF_MMQ_WS + sr * NF_MMQ_TK + 16 * h + kk]; \
                } \
                wsu[(sel) * NF_MMQ_TM * NF_MMQ_SH + sr * NF_MMQ_SH + h] = sum; \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

/*
 * The pipe keeps the next two chunks of the weight stream in flight: while chunk c is combined,
 * the register sets hold the codes of chunks c + 1 (issued one iteration ago) and c + 2 (issued
 * this iteration); stage j cycles the tiles buf j % NF_MMQ_PIPE_BUF and register set j % 2. The
 * prologue loads stage 0, commits it to tile 0, then issues stages 1 and 2 before the first
 * combine, so the stream is two-deep from the first iteration on. Tile reuse is three iterations
 * apart and the pre-commit barrier retires the combine that read the tile three iterations before,
 * so the commit cannot race a reader.
 *
 * The staged record tile refills synchronously between the barriers, the same gather on every
 * dialect (one tile per workgroup, see NF_MMQ_XR_BUF). The COMBINE / WSUFILL macros read the
 * tile through the xr alias the pipe declares per phase.
 */
/* One iteration of the pipe: chunk c is stage j, whose prefetch register set SET (j % 2) the caller passes as a
 * constant so the register sets stay registers. */
#define NF_MMQ_PIPE_STEP(c, j, SET, kend, TN, COMBINE, BS, BB, CODE, SCALE, OFF, XA, XB) \
    { \
        barrier(CLK_LOCAL_MEM_FENCE); \
        sel = (j) % NF_MMQ_PIPE_BUF; \
        NF_MMQ_XR_STEP((c), (kend), (j), (TN), q8, q_stride, q_off, n0, N) \
        NF_MMQ_STAGE_COMMIT(SET, sel) \
        if ((c) + 2 * NF_MMQ_TK < (kend)) { \
            NF_MMQ_PREFETCH(SET, (c) + 2 * NF_MMQ_TK, m0, (kend), M, weight, row_bytes, (BS), (BB), CODE, SCALE, OFF) \
        } \
        NF_MMQ_XR_READY \
        barrier(CLK_LOCAL_MEM_FENCE); \
        NF_LOCAL_PTR const uint* xr = NF_MMQ_XR_AT((j), (TN)); \
        NF_MMQ_WSUFILL(sel) \
        COMBINE(sel, XA, XB); \
    }

/* The loop takes two stages per iteration, odd stage j then j + 1, so every stage's register set is a
 * compile-time index. */
#define NF_MMQ_PIPE(cstart, kend, TN, COMBINE, BS, BB, CODE, SCALE, OFF, XA, XB) \
    uint pfw[NF_MMQ_PIPE_SETS][NF_MMQ_SP4]; \
    float psc[NF_MMQ_PIPE_SETS]; \
    float pof[NF_MMQ_PIPE_SETS]; \
    int sel = 0; \
    if ((cstart) < (kend)) { \
        NF_MMQ_PREFETCH(0, (cstart), m0, (kend), M, weight, row_bytes, (BS), (BB), CODE, SCALE, OFF) \
        NF_MMQ_STAGE_COMMIT(0, 0) \
        NF_MMQ_XR_PROLOGUE((cstart), (kend), (TN), q8, q_stride, q_off, n0, N) \
        if ((cstart) + NF_MMQ_TK < (kend)) { \
            NF_MMQ_PREFETCH(1, (cstart) + NF_MMQ_TK, m0, (kend), M, weight, row_bytes, (BS), (BB), CODE, SCALE, OFF) \
        } \
        if ((cstart) + 2 * NF_MMQ_TK < (kend)) { \
            NF_MMQ_PREFETCH(0, (cstart) + 2 * NF_MMQ_TK, m0, (kend), M, weight, row_bytes, (BS), (BB), CODE, SCALE, OFF) \
        } \
        NF_MMQ_XR_READY \
        barrier(CLK_LOCAL_MEM_FENCE); \
        { \
            NF_LOCAL_PTR const uint* xr = NF_MMQ_XR_AT(0, (TN)); \
            NF_MMQ_WSUFILL(0) \
            COMBINE(0, XA, XB); \
        } \
        for (int c = (cstart) + NF_MMQ_TK, j = 1; c < (kend); c += 2 * NF_MMQ_TK, j += 2) { \
            NF_MMQ_PIPE_STEP(c, j, 1, (kend), (TN), COMBINE, (BS), (BB), CODE, SCALE, OFF, XA, XB) \
            if (c + NF_MMQ_TK >= (kend)) { \
                break; \
            } \
            NF_MMQ_PIPE_STEP(c + NF_MMQ_TK, j + 1, 0, (kend), (TN), COMBINE, (BS), (BB), CODE, SCALE, OFF, XA, XB) \
        } \
    }

/*
 * Packed-dot path: dot of one 16-element half of a weight-code row with the staged code row of the
 * token, 4-way with __dp4a on CUDA sm_61+ and with the native packed dot of an OpenCL compiler that
 * reports one (NF_NATIVE_DP4A, common/dialect.h), scalar int elsewhere (HIP has no __dp4a, so its
 * dialect takes the scalar loop). All three are the exact integer dot. rec points at the half's first code int in the staged record tile; the
 * halves of a 32-deep stage carry their own scales, so the caller sums the half dots under those
 * scales.
 */
NF_INLINE int nf_mmq_half_dot(NF_LOCAL_PTR const nf_i8* wq_row, NF_LOCAL_PTR const uint* rec) {
#if defined(__CUDACC__) && !defined(__HIP__) \
    && (defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610)
    int ib = 0;
    for (int u = 0; u < 4; u++) {
        const int w = *(NF_LOCAL_PTR const int*)(wq_row + 4 * u);   /* rows of int-aligned tiles */
        ib = __dp4a(w, (int)rec[u], ib);
    }
    return ib;
#elif defined(NF_NATIVE_DP4A)
    int ib = 0;
    for (int u = 0; u < 4; u++) {
        const int w = *(NF_LOCAL_PTR const int*)(wq_row + 4 * u);   /* rows of int-aligned tiles */
        ib = NF_DP4A(w, (int)rec[u], ib);
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
/* d = a * b^T + c over one 16x8x32 int8 tile (row-major A fragment, column-major B fragment) with every
 * accumulator input c (0, or the folded conversion's NF_MMQ_TC_MAGIC). */
NF_INLINE void nf_mmq_mma_m16n8k32(int d[4], const uint a[4], const uint b[2], int c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c));
}

/* d = a * b^T + c over one 16x8x16 int8 tile with every accumulator input c. */
NF_INLINE void nf_mmq_mma_m16n8k16(int d[4], const uint a[2], uint b, int c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b), "r"(c));
}

/*
 * A chunk's integer dot d under row scale dx, rounded once. exact: the mma accumulated onto 0x4B400000, so d holds
 * the bits of the float 1.5 * 2^23 + dot, and ndx = -(dx * 1.5 * 2^23) is exact for a scale whose product with 3
 * is exact in fp32 (the SCALE##_NARROW trait): the one fma is fmul(dx, (float)dot) with the conversion folded in,
 * bit for bit at every finite dx. A non-finite dx (only a corrupt half scale) gives NaN where fmul gives +-inf.
 */
NF_INLINE float nf_mmq_tc_prod(int d, float dx, float ndx, const int exact) {
    return exact ? __fmaf_rn(dx, __int_as_float(d), ndx) : __fmul_rn(dx, (float)d);
}

/*
 * One output's chunk term added to its accumulator, acc += d_t * (dx * d + dy * s - zp * (dx * W + dy * n)),
 * rounded step by step: p = dx * d (nf_mmq_tc_prod), fma(dy, s, p), then fma(-fma(dy, n, dx * W), zp, .), then
 * fma(d_t, ., acc). The tensor tile and its twins (nf_mmq_tensor_stage) all take their terms here, so they agree
 * bit for bit at finite scales (nf_mmq_tc_prod); a form that drops a term (dy == 0 without offsets, zp == 0 on
 * symmetric records) drops an exact zero (only the sign of a zero sum can differ, and acc starts at +0).
 */
NF_INLINE float nf_mmq_tc_term(float acc, float p, float dx, float dy, float s, float dt, int zp, int wsum,
                               int n, const int has_off, const int asym) {
    float sd = has_off ? __fmaf_rn(dy, s, p) : p;
    if (asym) {
        const float corr = has_off ? __fmaf_rn(dy, (float)n, __fmul_rn(dx, (float)wsum))
                                   : __fmul_rn(dx, (float)wsum);
        sd = __fmaf_rn(-corr, (float)zp, sd);
    }
    return __fmaf_rn(dt, sd, acc);
}

/*
 * The tensor path of one stage. Warp w covers rows (w >> 1) * 16 and the tn / 16 n8 token blocks
 * from (w & 1) * tn / 2 (tn, the tile's token count, is a compile-time constant at every call). Lane
 * l of block nb holds the accumulator positions (r, c), (r, c+1),
 * (r+8, c), (r+8, c+1) at r = (w >> 1) * 16 + (l >> 2), c = block start + (l & 3) * 2 in
 * acc[nb * 4 ..]; its A fragment registers carry rows r and r + 8 at depths (l & 3) * 4 (+16), its
 * B registers token block start + (l >> 2) at the same depths, loaded from the staged record tile
 * xr the gather filled (PTX mma fragment layouts for .s8, as llama.cpp's tile<16, 8, int> /
 * tile<8, 8, int>).
 * The epilogue applies the scales as llama.cpp's generic mma combine does (ggml-cuda/mmq-vec-dot.cuh):
 * the lane's row (scale, offset) pairs are hoisted out of the token-block loop (dmA) and each token's
 * chunk scale, code sums and zero point read once per token (dsB), so one combine costs two FMA per
 * accumulator; the asymmetric zero-point correction is folded in only when the records carry one
 * (the same work-group-uniform gate as the code-sum fill, token 0's zero-point unit).
 * SG == 32 takes one k32 product under the stage's single (scale, offset); SG == 16 takes one k16
 * product per half under that half's pair. The tokens' d, code sums and zero points read the
 * record tile; records of tokens past N are zero, so no bounds test applies.
 */
NF_INLINE void nf_mmq_tensor_stage(NF_LOCAL_PTR const nf_i8* wq, NF_LOCAL_PTR const uint* xr,
                                   int tn,
                                   NF_LOCAL_PTR const float* wsc, NF_LOCAL_PTR const float* wof,
                                   NF_LOCAL_PTR const int* wsu,
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
    float dmx[2][NF_MMQ_SH], dmy[2][NF_MMQ_SH];
    for (int j = 0; j < 2; j++) {
        for (int h = 0; h < NF_MMQ_SH; h++) {
            dmx[j][h] = wsc[rr[2 * j] * NF_MMQ_SH + h];
            dmy[j][h] = wof[rr[2 * j] * NF_MMQ_SH + h];
        }
    }
    /* token 0's zero-point unit, uniform over the work-group after the gather (0 for every
     * symmetric record) */
    const int xz = (int)xr[NF_MMQ_QS + 1 + NF_MMQ_SH];
    for (int nb = 0; nb < tn / 16; nb++) {
        const int cbase = (warp & 1) * (tn / 2) + nb * 8;
        const int c0 = cbase + (lane & 3) * 2;
        const int cb = cbase + (lane >> 2);
        /* the staged code words of this lane's token: depths (l & 3) * 4 (+16); zero for dead tokens */
        const int rb = cb * NF_MMQ_RC;
        const uint b[2] = {xr[rb + kf / 4], xr[rb + kf / 4 + NF_MMQ_TK / 8]};
        /* the lane's two tokens' chunk scale and half code sums, one read per token (dead tokens
         * stage zeros); the zero points read only under the asymmetric gate */
        float tds[2];
        int tsum[2][NF_MMQ_SH];
        int tzp[2];
        for (int t = 0; t < 2; t++) {
            const int tc = (c0 + t) * NF_MMQ_RC;
            tds[t] = nf_mmq_rec_d(xr[tc + NF_MMQ_QS]);
            for (int h = 0; h < NF_MMQ_SH; h++) {
                tsum[t][h] = (int)xr[tc + NF_MMQ_QS + 1 + h];
            }
            tzp[t] = xz != 0 ? (int)xr[tc + NF_MMQ_QS + 1 + NF_MMQ_SH] : 0;
        }
        if (sg == 32) {
            int d[4];
            nf_mmq_mma_m16n8k32(d, a, b, 0);
            for (int i = 0; i < 4; i++) {
                const int j = i >> 1, t = i & 1;
                const int si = rr[2 * j] * NF_MMQ_SH;
                acc[nb * 4 + i] = nf_mmq_tc_term(acc[nb * 4 + i], nf_mmq_tc_prod(d[i], dmx[j][0], 0.0f, 0),
                                                 dmx[j][0], dmy[j][0], (float)(tsum[t][0] + tsum[t][1]), tds[t],
                                                 tzp[t], wsu[si] + wsu[si + 1], NF_MMQ_SH * 16, 1, xz != 0);
            }
        } else {
            for (int h = 0; h < NF_MMQ_SH; h++) {
                const uint ah[2] = {a[2 * h], a[2 * h + 1]};
                int d[4];
                nf_mmq_mma_m16n8k16(d, ah, b[h], 0);
                for (int i = 0; i < 4; i++) {
                    const int j = i >> 1, t = i & 1;
                    const int si = rr[2 * j] * NF_MMQ_SH + h;
                    acc[nb * 4 + i] = nf_mmq_tc_term(acc[nb * 4 + i], nf_mmq_tc_prod(d[i], dmx[j][h], 0.0f, 0),
                                                     dmx[j][h], dmy[j][h], (float)tsum[t][h], tds[t], tzp[t],
                                                     wsu[si], 16, 1, xz != 0);
                }
            }
        }
    }
}
#endif /* NF_MMQ_TENSOR */

/*
 * One stage's integer dots and their (scale, offset) / d combine into acc. Tensor path:
 * nf_mmq_tensor_stage. Packed-dot path (one 16-element half per stage): work-item lid accumulates
 * rows tx + NF_MMQ_TX*q against each of its tn / 16 tokens ty + 16*j, reading the token's staged
 * codes from the record tile; acc[j * NF_MMQ_RM + q] holds the result.
 */
NF_INLINE void nf_mmq_combine(NF_LOCAL_PTR const nf_i8* wq, NF_LOCAL_PTR const uint* xr,
                              int n0, int tn, int N,
                              NF_LOCAL_PTR const float* wsc, NF_LOCAL_PTR const float* wof,
                              NF_LOCAL_PTR const int* wsu,
                              int lid, float acc[], const int sg) {
#if NF_MMQ_TENSOR
    nf_mmq_tensor_stage(wq, xr, tn, wsc, wof, wsu, lid, acc, sg);
#else
    const int tx = lid % NF_MMQ_TX, ty = lid / NF_MMQ_TX;
    /* token 0's zero-point unit, uniform over the work-group after the gather (0 for every
     * symmetric record): its gate skips the weight code-sum term of the asymmetric form */
    const int xz = (int)xr[NF_MMQ_QS + 1 + NF_MMQ_SH];
    for (int j = 0; j < tn / 16; j++) {
        const int tok = ty + 16 * j;
        if (n0 + tok >= N) {
            continue;
        }
        NF_LOCAL_PTR const uint* rec = xr + tok * NF_MMQ_RC;
        for (int q = 0; q < NF_MMQ_RM; q++) {
            const int sr = tx + NF_MMQ_TX * q;
            float sd = 0.0f;
            float corr = 0.0f;
            for (int h = 0; h < NF_MMQ_SH; h++) {
                const int ib = nf_mmq_half_dot(wq + sr * NF_MMQ_TK + 16 * h, rec + 4 * h);
                sd += wsc[sr * NF_MMQ_SH + h] * (float)ib
                        + wof[sr * NF_MMQ_SH + h] * (float)(int)rec[NF_MMQ_QS + 1 + h];
                if (xz != 0) {
                    corr += wsc[sr * NF_MMQ_SH + h] * (float)wsu[sr * NF_MMQ_SH + h]
                            + wof[sr * NF_MMQ_SH + h] * (float)16;
                }
            }
            acc[j * NF_MMQ_RM + q] += nf_mmq_rec_d(rec[NF_MMQ_QS])
                    * (sd - (float)(int)rec[NF_MMQ_QS + 1 + NF_MMQ_SH] * corr);
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

#if NF_MMQ_TENSOR
/* ---- the tensor-core device tile: T_MatMul_MMQ and T_MatMul_MMQ_KSplit on CUDA sm_80+ ----
 *
 * The plain kernels over the NF_MMQ_TM x NF_MMQ_TN device tile stage NF_MMQ_TC_DEPTH NF_MMQ_TK-deep
 * chunks per barrier: two alternating groups of weight buffers and of record stages, one buffer and one
 * stage per chunk of the group. A step starts the asynchronous copies of the next group's records into
 * the free record stages (16-byte cp.async, no register round trip), issues the next group's weight and
 * (scale, offset) loads into registers, combines the current group's chunks from shared memory, then
 * decodes and commits the loaded weights, waits for the copies and passes its one barrier; the weight
 * loads of both chunks go ahead of the combines, or one chunk's ahead of each combine for the types whose
 * decode needs the registers (T_MMQ_CODE_SETS, NF_MMQ_TC_STEP). The load latency and the type's decode
 * overlap the combine instead of sitting between barriers, and each barrier closes NF_MMQ_TC_DEPTH chunks
 * of mma work (llama.cpp's mmq stages a 256-deep slab per pair of barriers). On an RTX 5080 Laptop
 * (sm_120) two chunks per barrier run the pp512 prefill of Qwen3-Embedding-4B 15% faster in Q8_0 and 11%
 * faster in Q4_K_M than one; four would need twice the 43-45 KiB of buffers, whose dynamic shared memory leaves
 * too little of that device's 128 KiB L1 to the weight and record loads. The NF_MMQ_WG / 32 warps form
 * NF_MMQ_TC_RG row groups
 * of NF_MMQ_TC_NTX = NF_MMQ_RM / 4 m16 minitiles by NF_MMQ_TC_TG token groups of NF_MMQ_TC_NB n8
 * blocks: on the 128 x 128 tile at eight rows per work-item, eight warps of 32 rows x 64 tokens
 * (llama.cpp's 256-thread block), so each A fragment read with ldmatrix feeds eight mma and each B
 * fragment and token scale read from the record tile two.
 *
 * The staged weight codes are word-swizzled (word w of row r at r * 8 + (w ^ 4 * ((r >> 2) & 1))) so
 * the ldmatrix reads of the A fragments and the commit stores are free of bank conflicts without
 * padding. A record stage holds the chunk's records as the staging pass wrote them, copied in coalesced
 * 16-byte pieces (each global sector is read once): a token's code words sit NF_MMQ_RC units after the
 * previous token's, which puts the eight token rows of an ldmatrix in distinct banks, so one ldmatrix.x4
 * reads the B fragments of two n8 blocks, and its chunk scale, half code sums and zero point are one
 * 128-bit read (NF_MMQ_TC_XF). The (scale, offset) tiles hold the rows in lane order (NF_MMQ_TC_WPOS): one
 * read holds a lane's four rows.
 *
 * Every output takes the per-chunk fp32 combine of nf_mmq_tensor_stage (nf_mmq_tc_term) in the same order,
 * so the plain kernels equal the small-token twins bit for bit; types whose OFF is identically zero (the
 * HAS_OFF trait of the instantiation) skip the zero offset term, which leaves every sum unchanged, and the
 * SCALE##_NARROW types fold the chunk product's conversion into its scaling (nf_mmq_tc_prod). The asymmetric
 * zero-point form is a property of the staging pass (LLM_Quant_MMQ_Int8Asym writes one static zero point
 * into every record), so the gate the twins read per chunk - token 0's zero-point unit - is read once per
 * tile and selects the pipe's form.
 */
#define NF_MMQ_TC_NTX (NF_MMQ_RM / 4)                  /* m16 row minitiles per warp */
#define NF_MMQ_TC_WROWS (16 * NF_MMQ_TC_NTX)           /* weight rows per warp */
#define NF_MMQ_TC_WARPS (NF_MMQ_WG / 32)
#define NF_MMQ_TC_RG (NF_MMQ_TM / NF_MMQ_TC_WROWS)     /* warp row groups */
#define NF_MMQ_TC_TG (NF_MMQ_TC_WARPS / NF_MMQ_TC_RG)  /* warp token groups */
/* The token-dependent counts of a TN-token tile: the device tile runs NF_MMQ_TN tokens, the small-token
 * tensor tile NF_MMQ_TC_STN; a kernel body names its tile's count as the constant tc_tn. */
#define NF_MMQ_TC_WTOK_OF(TN) ((TN) / NF_MMQ_TC_TG)               /* tokens per warp */
#define NF_MMQ_TC_NB_OF(TN) (NF_MMQ_TC_WTOK_OF(TN) / 8)          /* n8 token blocks per warp */
#define NF_MMQ_TC_ACC_OF(TN) (NF_MMQ_TC_NTX * NF_MMQ_TC_NB_OF(TN) * 4) /* accumulators per lane */
#define NF_MMQ_TC_XU_OF(TN) ((TN) * NF_MMQ_RC)                   /* record units of one chunk */
#define NF_MMQ_TC_XP_OF(TN) (NF_MMQ_TC_XU_OF(TN) / 4)            /* 16-byte record pieces of one chunk */
#define NF_MMQ_TC_XPI_OF(TN) ((NF_MMQ_TC_XP_OF(TN) + NF_MMQ_WG - 1) / NF_MMQ_WG) /* record pieces per work-item */
#define NF_MMQ_TC_WTOK NF_MMQ_TC_WTOK_OF(NF_MMQ_TN)
#define NF_MMQ_TC_NB NF_MMQ_TC_NB_OF(NF_MMQ_TN)
#define NF_MMQ_TC_ACC NF_MMQ_TC_ACC_OF(NF_MMQ_TN)
#define NF_MMQ_TC_XU NF_MMQ_TC_XU_OF(NF_MMQ_TN)
/* Tokens of the small-token tensor tile (T_MatMul_MMQ_StreamK_S): the device tile's rows, warps and pipe over
 * one 16-token column, llama.cpp's mmq_x = 16 row of its J ladder at mmq_y = 128, which its small decode
 * batches run as stream-k. Each warp covers its 32 rows against one n8 token block. */
#define NF_MMQ_TC_STN 16
#define NF_MMQ_TC_CW (NF_MMQ_TM * NF_MMQ_QS)            /* staged code words of one chunk */
#define NF_MMQ_TC_CWI (NF_MMQ_TC_CW / NF_MMQ_WG)        /* code words per work-item and chunk */
#define NF_MMQ_TC_SW (NF_MMQ_TM * NF_MMQ_SH)            /* (scale, offset) slots of one chunk, at most */
#define NF_MMQ_TC_SWI ((NF_MMQ_TC_SW + NF_MMQ_WG - 1) / NF_MMQ_WG)
#define NF_MMQ_TC_XF NF_MMQ_QS                          /* a token's (d, half sums, zero point) in its record */
#define NF_MMQ_TC_DEPTH 2                               /* chunks staged per barrier */
/* The prefetch register sets of a step (NF_MMQ_TC_STEP) for a type asking for S: the two-set form is
 * stack-free on sm_100 and sm_120 (measured on sm_120); the sm_80 to sm_90 compilers spill it for many types,
 * so those targets take the one-set form. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#define NF_MMQ_TC_SETS(S) 1
#else
#define NF_MMQ_TC_SETS(S) (S)
#endif

#if NF_MMQ_TK != 32 || NF_MMQ_TC_DEPTH != 2 || NF_MMQ_TC_CW % NF_MMQ_WG != 0 \
    || NF_MMQ_TM % NF_MMQ_TC_WROWS != 0 \
    || NF_MMQ_TC_WARPS % NF_MMQ_TC_RG != 0 || NF_MMQ_TN % (8 * NF_MMQ_TC_TG) != 0 \
    || NF_MMQ_TC_STN % (8 * NF_MMQ_TC_TG) != 0
#error "the tensor-core MMQ tile needs 32-deep chunks, two per stage, and whole warp tiles"
#endif
/* NF_MMQ_TC_SLOTS maps eight code words to a tile row (slot >> 3), and NF_MMQ_TC_PREFETCH takes the slots below
 * NF_MMQ_TC_CWI / 2 for the tile's upper half (the gate rows of the gate/up tile): that holds only for whole
 * eight-word rows and an even slot count per work-item. */
#if NF_MMQ_QS != 8 || NF_MMQ_TC_CWI % 2 != 0
#error "the tensor-core MMQ tile's code slots must split into the tile's halves at NF_MMQ_TC_CWI / 2"
#endif

/* The lane-order layouts below hold a warp's two 16-row minitiles and its 8-token blocks in pairs and quads
 * (the device tile; the small-token tile's warp holds one block); the record stage needs the 48-byte record
 * (8 code words, d, two half sums, zero point): 16-byte pieces, and token rows 12 banks apart. */
#if NF_MMQ_TC_NTX != 2 || NF_MMQ_TC_NB % 4 != 0 || NF_MMQ_TC_NB_OF(NF_MMQ_TC_STN) != 1 \
    || NF_MMQ_RC != NF_MMQ_QS + 2 + NF_MMQ_SH || NF_MMQ_RC != 12
#error "the tensor-core MMQ tile's shared layouts need two minitiles per warp, block quads and 48-byte records"
#endif

/* The staged position of code word w of tile row r (see the swizzle above). */
#define NF_MMQ_TC_SWZ(r, w) ((r) * NF_MMQ_QS + ((w) ^ ((((r) >> 2) & 1) << 2)))

/* (scale, offset) slots per row and chunk: one per 32-element group, one per half for SG 16. */
#define NF_MMQ_TC_SC(SG) ((SG) == 16 ? NF_MMQ_SH : 1)
/* (scale, offset) slots of one chunk's buffer at scale group SG. */
#define NF_MMQ_TC_SB(SG) (NF_MMQ_TM * NF_MMQ_TC_SC(SG))

/* Lane order of the (scale, offset) tiles: row r of a warp's 32 rows sits at the position that makes the rows
 * g, g + 8, g + 16 and g + 24 of lane group g = r & 7 consecutive, minitile-major (the lane's rows
 * r = wrow0 + 16m + g + 8j read as entry 2m + j). */
#define NF_MMQ_TC_WPOS(r) (((r) & 7) << 2 | (((r) >> 3) & 3))
/* The position of slot h of row r in a (scale, offset) tile of SC slots per row: per warp row group, slot h's
 * 32 rows in lane order, so that a lane's four rows of one slot are one 128-bit read whatever SC. */
#define NF_MMQ_TC_WSLOT(r, h, SC) (((r) & ~31) * (SC) + (h) * 32 + NF_MMQ_TC_WPOS(r))

/* The accumulator of the offset-free forms' mma: 0x4B400000, 1.5 * 2^23 as a float, so that the s32 result
 * dot + 0x4B400000 reads as the float 1.5 * 2^23 + dot (exact for |dot| < 2^22: a 32-deep int8 dot is at
 * most 2^19 in magnitude) and the sum stays below 2^31. */
#define NF_MMQ_TC_MAGIC 0x4B400000
#define NF_MMQ_TC_MAGIC_F 12582912.0f

/* Four 8x8 b16 matrices from shared memory: lane l supplies the row address of matrix l >> 3. */
NF_INLINE void nf_mmq_ldsm_x4(uint a[4], NF_LOCAL_PTR const int* p) {
    const uint addr = (uint)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
}

/* Two 8x8 b16 matrices into a[0..1] (lanes 0-15 supply the row addresses) and one into a[0] (lanes 0-7). */
NF_INLINE void nf_mmq_ldsm_x2(uint a[4], NF_LOCAL_PTR const int* p) {
    const uint addr = (uint)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(addr));
}
NF_INLINE void nf_mmq_ldsm_x1(uint a[4], NF_LOCAL_PTR const int* p) {
    const uint addr = (uint)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                 : "=r"(a[0])
                 : "r"(addr));
}

/* X (1, 2 or 4) 8x8 b16 matrices: the token-block reads of a warp holding one, two or four n8 blocks. */
template <int X>
NF_INLINE void nf_mmq_ldsm(uint a[4], NF_LOCAL_PTR const int* p) {
    if (X == 4) {
        nf_mmq_ldsm_x4(a, p);
    } else if (X == 2) {
        nf_mmq_ldsm_x2(a, p);
    } else {
        nf_mmq_ldsm_x1(a, p);
    }
}

/* cp.async of one 4-byte unit and of one aligned 16-byte unit through L1 (.ca), of one aligned 16-byte unit
 * past it (.cg), the closing of the work-item's copy group, and the wait for all of its groups. */
NF_INLINE void nf_mmq_cp_async4(NF_LOCAL_PTR uint* dst, const NF_GLOBAL uint* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"((uint)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
NF_INLINE void nf_mmq_cp_async16_ca(NF_LOCAL_PTR uint* dst, const NF_GLOBAL uint* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"((uint)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
NF_INLINE void nf_mmq_cp_async16_cg(NF_LOCAL_PTR uint* dst, const NF_GLOBAL uint* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"((uint)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
NF_INLINE void nf_mmq_cp_async_commit(void) {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
NF_INLINE void nf_mmq_cp_async_wait(void) {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

/* The integer code sum of half h (16 codes) of staged row r, for the asymmetric zero-point term. */
NF_INLINE int nf_mmq_tc_half_sum(NF_LOCAL_PTR const int* wq, int r, int h) {
    int s = 0;
    for (int u = 0; u < 4; u++) {
        s = __dp4a(wq[NF_MMQ_TC_SWZ(r, 4 * h + u)], 0x01010101, s);
    }
    return s;
}

/*
 * The lane's four rows' (scale, offset) of slot h in a tile of sc slots per row, entry 2m + j for row
 * wrow0 + 16m + (lane >> 2) + 8j; ndx is the folded conversion's -(scale * NF_MMQ_TC_MAGIC_F) and wsu the
 * weight code sum of the slot (both halves at sc 1) for the asymmetric correction, each only on its form.
 */
NF_INLINE void nf_mmq_tc_rows(NF_LOCAL_PTR const int* wq, NF_LOCAL_PTR const float* wsc,
                              NF_LOCAL_PTR const float* wof, int lane, int wrow0, int h, int sc, float dmx[4],
                              float dmy[4], float ndx[4], int wsu[4], const int has_off, const int asym,
                              const int exact) {
    const int at = NF_MMQ_TC_WSLOT(wrow0 + (lane >> 2), h, sc);
    const float4 vx = *(NF_LOCAL_PTR const float4*)(wsc + at);
    const float4 vy = has_off ? *(NF_LOCAL_PTR const float4*)(wof + at) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float sx[4] = {vx.x, vx.y, vx.z, vx.w};
    const float sy[4] = {vy.x, vy.y, vy.z, vy.w};
    for (int e = 0; e < 4; e++) {
        const int r = wrow0 + 16 * (e >> 1) + (lane >> 2) + 8 * (e & 1);
        dmx[e] = sx[e];
        dmy[e] = sy[e];
        ndx[e] = exact ? -__fmul_rn(sx[e], NF_MMQ_TC_MAGIC_F) : 0.0f;
        wsu[e] = !asym ? 0 : sc == 1 ? nf_mmq_tc_half_sum(wq, r, 0) + nf_mmq_tc_half_sum(wq, r, 1)
                                     : nf_mmq_tc_half_sum(wq, r, h);
    }
}

/*
 * One chunk of the tensor-core tile at 32-element scales: the lane's NF_MMQ_TC_NTX minitiles against its
 * warp's NB n8 token blocks, combined per output exactly as nf_mmq_tensor_stage does. wq / xr / wsc / wof are
 * the chunk's staged buffers in the layouts of the tile section, wrow0 / wtok0 the warp's first tile row and
 * token; has_off, asym (the tile's zero-point form) and exact (the folded conversion of a SCALE##_NARROW type)
 * are compile-time constants. The B fragments of a block pair are one ldmatrix.x4 over the staged code words
 * (matrices: block 2p words 0-3, 4-7, block 2p + 1 words 0-3, 4-7) feeding both minitiles - an ldmatrix.x2 of
 * the one block of the small-token tile's warp - and each token's chunk scale, half code sums and zero point
 * one 128-bit read.
 */
template <int NB>
NF_INLINE void nf_mmq_tc_combine32(NF_LOCAL_PTR const int* wq, NF_LOCAL_PTR const uint* xr,
                                   NF_LOCAL_PTR const float* wsc, NF_LOCAL_PTR const float* wof,
                                   int lane, int wrow0, int wtok0, float acc[],
                                   const int has_off, const int asym, const int exact) {
    constexpr int PB = NB >= 2 ? 2 : 1;    /* n8 blocks per B read: a pair, or the warp's one block */
    const int q = lane & 3;
    /* ldmatrix row of this lane: matrices (rows 0-7 | 8-15) x (words 0-3 | 4-7) of a minitile */
    const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lword = (((lane >> 4) & 1) ^ ((lane >> 2) & 1)) * 4;
    /* and of the B matrices: token (lane & 7) of block lane >> 4 (of the pair), words (lane >> 3) & 1 */
    const int btok = (lane & 7) + (PB == 2 ? 8 * (lane >> 4) : 0);
    const int bword = ((lane >> 3) & 1) * 4;
    uint a[NF_MMQ_TC_NTX][4];
    float dmx[4], dmy[4], ndx[4];
    int wsu[4];
    for (int m = 0; m < NF_MMQ_TC_NTX; m++) {
        nf_mmq_ldsm_x4(a[m], wq + (wrow0 + 16 * m + lrow) * NF_MMQ_QS + lword);
    }
    nf_mmq_tc_rows(wq, wsc, wof, lane, wrow0, 0, 1, dmx, dmy, ndx, wsu, has_off, asym, exact);
    for (int p = 0; p < NB / PB; p++) {
        uint bp[4];
        nf_mmq_ldsm<2 * PB>(bp, (NF_LOCAL_PTR const int*)xr + (wtok0 + 8 * PB * p + btok) * NF_MMQ_RC + bword);
        /* the lane's tokens 2q, 2q + 1 of the read's blocks, entry 2o + i */
        float tdp[2 * PB], tsp[2 * PB];
        int tzq[2 * PB];
        for (int t = 0; t < 2 * PB; t++) {
            const uint4 f = *(NF_LOCAL_PTR const uint4*)(xr + (wtok0 + 8 * PB * p + 8 * (t >> 1) + 2 * q + (t & 1))
                                                              * NF_MMQ_RC + NF_MMQ_TC_XF);
            tdp[t] = nf_mmq_rec_d(f.x);
            tsp[t] = has_off ? (float)((int)f.y + (int)f.z) : 0.0f;
            tzq[t] = asym ? (int)f.w : 0;
        }
        for (int o = 0; o < PB; o++) {
            const uint b[2] = {bp[2 * o], bp[2 * o + 1]};
            for (int m = 0; m < NF_MMQ_TC_NTX; m++) {
                float* ac = acc + (m * NB + PB * p + o) * 4;
                int d[4];
                nf_mmq_mma_m16n8k32(d, a[m], b, exact ? NF_MMQ_TC_MAGIC : 0);
                for (int i = 0; i < 4; i++) {
                    const int e = 2 * m + (i >> 1), t = 2 * o + (i & 1);
                    ac[i] = nf_mmq_tc_term(ac[i], nf_mmq_tc_prod(d[i], dmx[e], ndx[e], exact), dmx[e], dmy[e],
                                           tsp[t], tdp[t], tzq[t], wsu[e], NF_MMQ_SH * 16, has_off, asym);
                }
            }
        }
    }
}

/*
 * The chunk of nf_mmq_tc_combine32 at 16-element scales: two k16 mma per block and minitile, each with its own
 * row scales, run half-major - every output takes its half-0 term before its half-1 term, as in
 * nf_mmq_tensor_stage - so that only one half's A fragments and row scales are live. One ldmatrix.x4 reads a
 * half of both minitiles (matrices: minitile 0 rows 0-7, 8-15, minitile 1 rows 0-7, 8-15) and one a half of a
 * quad of blocks (matrix j: block 4g + j; the warp's NB blocks when it holds fewer than four).
 */
template <int NB>
NF_INLINE void nf_mmq_tc_combine16(NF_LOCAL_PTR const int* wq, NF_LOCAL_PTR const uint* xr,
                                   NF_LOCAL_PTR const float* wsc, NF_LOCAL_PTR const float* wof,
                                   int lane, int wrow0, int wtok0, float acc[],
                                   const int has_off, const int asym, const int exact) {
    constexpr int QB = NB >= 4 ? 4 : NB;    /* n8 blocks per B read */
    const int q = lane & 3;
    const int sw = (lane >> 2) & 1;    /* the swizzle bit of the lane's A ldmatrix row (row bit 2) */
    const int arow = (lane & 7) + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);
    /* a loop, not unrolled: unrolled, the scheduler issues both halves' mma ahead of their epilogues and the tile
     * spills */
#pragma unroll 1
    for (int h = 0; h < NF_MMQ_SH; h++) {
        uint a[4];
        nf_mmq_ldsm_x4(a, wq + (wrow0 + arow) * NF_MMQ_QS + (h ^ sw) * 4);
        float dmx[4], dmy[4], ndx[4];
        int wsu[4];
        nf_mmq_tc_rows(wq, wsc, wof, lane, wrow0, h, NF_MMQ_SH, dmx, dmy, ndx, wsu, has_off, asym, exact);
        for (int g = 0; g < NB / QB; g++) {
            uint bq[4];
            nf_mmq_ldsm<QB>(bq, (NF_LOCAL_PTR const int*)xr + (wtok0 + 8 * QB * g + (lane & (8 * QB - 1)))
                                    * NF_MMQ_RC + h * 4);
            for (int j = 0; j < QB; j++) {
                const int nb = QB * g + j;
                float tdp[2], tsp[2];
                int tzq[2];
                for (int t = 0; t < 2; t++) {    /* the lane's tokens 2q + t of the block */
                    const uint4 f = *(NF_LOCAL_PTR const uint4*)(xr + (wtok0 + 8 * nb + 2 * q + t) * NF_MMQ_RC
                                                                      + NF_MMQ_TC_XF);
                    tdp[t] = nf_mmq_rec_d(f.x);
                    tsp[t] = has_off ? (float)(int)(h == 0 ? f.y : f.z) : 0.0f;
                    tzq[t] = asym ? (int)f.w : 0;
                }
                for (int m = 0; m < NF_MMQ_TC_NTX; m++) {
                    float* ac = acc + (m * NB + nb) * 4;
                    const uint ah[2] = {a[2 * m], a[2 * m + 1]};
                    int d[4];
                    nf_mmq_mma_m16n8k16(d, ah, bq[j], exact ? NF_MMQ_TC_MAGIC : 0);
                    for (int i = 0; i < 4; i++) {
                        const int e = 2 * m + (i >> 1), t = i & 1;
                        ac[i] = nf_mmq_tc_term(ac[i], nf_mmq_tc_prod(d[i], dmx[e], ndx[e], exact), dmx[e],
                                               dmy[e], tsp[t], tdp[t], tzq[t], wsu[e], 16, has_off, asym);
                    }
                }
            }
        }
    }
}

/*
 * Byte offsets of the work-item's weight slots from the tile's first row, and unit offsets of its
 * record pieces from the tile's first token, fixed for the whole tile: code word slot
 * s * NF_MMQ_WG + lid holds word slot & 7 of row slot >> 3 (a warp reads four rows' chunk codes as
 * contiguous spans), (scale, offset) slot i row i / SC. Of the records, work-item lid copies the 16-byte
 * pieces j * NF_MMQ_WG + lid of the record stage (tc_xo, from the tile's first record). Rows past M
 * read row M - 1 and tokens past N token N - 1: an output depends only on its own row's codes and scales
 * and its own token's record, and the outputs of those rows and tokens are never stored, so the loads only
 * need to stay in bounds.
 * The offsets fit int: a tile spans NF_MMQ_TM rows of at most ~1.07 bytes per element.
 * Tile row v reads weight row m0 + (v & tc_rmask) of wtile's weight below tc_rmask + 1 and of
 * wtile_u's above: the plain tile (tc_rmask = NF_MMQ_TM - 1, wtile_u = wtile) reads one weight, the
 * gate/up tile (tc_rmask = NF_MMQ_TM / 2 - 1) the gate rows in its upper half and the up rows of the
 * same indices in its lower. Code slot s holds rows [s, s + 1) * NF_MMQ_WG / 8, so the slots below
 * NF_MMQ_TC_CWI / 2 are the upper half (tc_su marks the (scale, offset) slots of the lower half).
 */
#define NF_MMQ_TC_SLOTS(SC) \
    for (int s = 0; s < NF_MMQ_TC_CWI; s++) { \
        const int slot = s * NF_MMQ_WG + lid; \
        tc_wo[s] = (int)((min(m0 + ((slot >> 3) & tc_rmask), M - 1) - m0) * row_bytes); \
        tc_we[s] = 4 * (slot & 7); \
    } \
    for (int s = 0; s < NF_MMQ_TC_SWI; s++) { \
        const int i = min(s * NF_MMQ_WG + lid, NF_MMQ_TM * (SC) - 1); \
        tc_so[s] = (int)((min(m0 + (i / (SC) & tc_rmask), M - 1) - m0) * row_bytes); \
        tc_su[s] = i / (SC) > tc_rmask; \
        tc_se[s] = (i % (SC)) * 16; \
    } \
    tc_x16 = ((size_t)xtile & 15) == 0; \
    for (int j = 0; j < NF_MMQ_TC_XPI_OF(tc_tn); j++) { \
        const int pc = min(j * NF_MMQ_WG + lid, NF_MMQ_TC_XP_OF(tc_tn) - 1); \
        tc_xo[j] = (min(n0 + pc / (NF_MMQ_RC / 4), N - 1) - n0) * q_stride * NF_MMQ_RC + 4 * (pc % (NF_MMQ_RC / 4)); \
    }

/*
 * Starts the asynchronous copy of chunk c's records of the tile's tokens into the record stage at unit
 * offset xoff (cp.async: the records need no decode, so they go global -> shared without registers)
 * and closes the copy group: consecutive work-items copy consecutive 16-byte pieces, coalesced, as one
 * 16-byte unit when the records are 16-byte aligned (q_off a multiple of 4, uniform over the launch) and
 * as four 4-byte units otherwise. The 16-byte copies of the types of blocks up to 128 elements (BS) go
 * through L1 (.ca): copied past it, the Q8_0, Q4_0, Q4_1 and I8_B128 tiles of an RTX 5080 Laptop run
 * 10-35% slower in some processes. The 256-element K and IQ super-block types copy past L1 (.cg), 3-5%
 * faster than through it.
 */
#define NF_MMQ_TC_XL1(BS) ((BS) <= 128)
#define NF_MMQ_TC_XCOPY(c, xoff, BS) \
    { \
        const NF_GLOBAL uint* xchunk = xtile + ((c) / NF_MMQ_TK) * NF_MMQ_RC; \
        for (int j = 0; j < NF_MMQ_TC_XPI_OF(tc_tn); j++) { \
            NF_LOCAL_PTR uint* xs = tc_xr + (xoff) + 4 * (j * NF_MMQ_WG + lid); \
            if (NF_MMQ_TC_XP_OF(tc_tn) % NF_MMQ_WG == 0 \
                || j * NF_MMQ_WG + lid < NF_MMQ_TC_XP_OF(tc_tn)) { \
                if (tc_x16 && NF_MMQ_TC_XL1(BS)) { \
                    nf_mmq_cp_async16_ca(xs, xchunk + tc_xo[j]); \
                } else if (tc_x16) { \
                    nf_mmq_cp_async16_cg(xs, xchunk + tc_xo[j]); \
                } else { \
                    for (int u = 0; u < 4; u++) { \
                        nf_mmq_cp_async4(xs + u, xchunk + tc_xo[j] + u); \
                    } \
                } \
            } \
        } \
        nf_mmq_cp_async_commit(); \
    }

/*
 * Loads chunk c's weight codes and (scale, offset) pairs into prefetch register set P (0 or 1)
 * through the slot offsets of NF_MMQ_TC_SLOTS: the chunk adds one block offset to every slot, from
 * wtile in the tile's upper half and from wtile_u in its lower (the same pointer in the plain tile, so
 * the choice folds away). Every load is unconditional, so the step stays one branch-free block.
 */
#define NF_MMQ_TC_PREFETCH(c, P, HAS_OFF, BS, BB, CODE, SCALE, OFF) \
    { \
        const NF_GLOBAL uchar* wc = wtile + ((c) / (BS)) * (BB); \
        const NF_GLOBAL uchar* wcu = wtile_u + ((c) / (BS)) * (BB); \
        for (int s = 0; s < NF_MMQ_TC_CWI; s++) { \
            tc_pw##P[s] = CODE##4((s < NF_MMQ_TC_CWI / 2 ? wc : wcu) + tc_wo[s], ((c) % (BS)) + tc_we[s]); \
        } \
        for (int s = 0; s < NF_MMQ_TC_SWI; s++) { \
            const NF_GLOBAL uchar* ws = (tc_su[s] ? wcu : wc) + tc_so[s]; \
            tc_ps##P[s] = SCALE(ws, (c) + tc_se[s]); \
            tc_po##P[s] = HAS_OFF ? OFF(ws, (c) + tc_se[s]) : 0.0f; \
        } \
    }

/* Writes prefetch register set P into weight buffer `buf`, the (scale, offset) slots in lane order. */
#define NF_MMQ_TC_COMMIT(buf, P, SG, HAS_OFF) \
    for (int s = 0; s < NF_MMQ_TC_CWI; s++) { \
        const int slot = s * NF_MMQ_WG + lid; \
        tc_wq[(buf) * NF_MMQ_TC_CW + NF_MMQ_TC_SWZ(slot >> 3, slot & 7)] = (int)tc_pw##P[s]; \
    } \
    for (int s = 0; s < NF_MMQ_TC_SWI; s++) { \
        const int i = s * NF_MMQ_WG + lid; \
        const int sc = NF_MMQ_TC_SC(SG); \
        if (i < NF_MMQ_TC_SB(SG)) { \
            tc_wsc[(buf) * NF_MMQ_TC_SB(SG) + NF_MMQ_TC_WSLOT(i / sc, i % sc, sc)] = tc_ps##P[s]; \
            if (HAS_OFF) { \
                tc_wof[(buf) * NF_MMQ_TC_SB(SG) + NF_MMQ_TC_WSLOT(i / sc, i % sc, sc)] = tc_po##P[s]; \
            } \
        } \
    }

/* The int8 combine of weight buffer buf and the record stage at unit offset xoff in the tile's form
 * (the COMBINE argument of NF_MMQ_TC_LOOP). */
#define NF_MMQ_TC_INT8(buf, xoff, SG, ASYM, HAS_OFF, EXACT) \
    if ((SG) == 32) { \
        nf_mmq_tc_combine32<NF_MMQ_TC_NB_OF(tc_tn)>(tc_wq + (buf) * NF_MMQ_TC_CW, tc_xr + (xoff), \
                                                    tc_wsc + (buf) * NF_MMQ_TC_SB(SG), \
                                                    tc_wof + (buf) * NF_MMQ_TC_SB(SG), lane, wrow0, wtok0, acc, \
                                                    (HAS_OFF), (ASYM), (EXACT)); \
    } else { \
        nf_mmq_tc_combine16<NF_MMQ_TC_NB_OF(tc_tn)>(tc_wq + (buf) * NF_MMQ_TC_CW, tc_xr + (xoff), \
                                                    tc_wsc + (buf) * NF_MMQ_TC_SB(SG), \
                                                    tc_wof + (buf) * NF_MMQ_TC_SB(SG), lane, wrow0, wtok0, acc, \
                                                    (HAS_OFF), (ASYM), (EXACT)); \
    }

/*
 * One step over the chunk group at c (chunks c and c + NF_MMQ_TK, the second only below cend), held in
 * weight buffers 2 PAR, 2 PAR + 1 and record stages 2 PAR, 2 PAR + 1. The next group's record copies
 * start into the other parity's stages 2 - 2 PAR + g (chunk g of the group), its weights load and
 * commit into the other parity's buffers 2 - 2 PAR + g, which every warp finished reading before the
 * previous barrier, and the step ends with the wait for the copies and its single barrier. With SETS 2
 * both chunks' weight loads are issued into register sets 0 and 1 before the group's combines and
 * committed after them, so the loads of both chunks are in flight behind the combines. With SETS 1 the
 * chunks take turns, copying and loading into set 0 before one combine and committing after it: a type
 * whose decode needs the registers keeps those of a one-chunk step. Near the end the next group's chunks repeat the
 * last chunk; their copies and loads land in stages and buffers nothing reads.
 */
#define NF_MMQ_TC_STEP(c, PAR, cend, clast, COMBINE, SG, ASYM, HAS_OFF, EXACT, BS, BB, CODE, SCALE, OFF, SETS) \
    { \
        if ((SETS) == 2) { \
            const int cn0 = min((c) + 2 * NF_MMQ_TK, (clast)); \
            const int cn1 = min((c) + 3 * NF_MMQ_TK, (clast)); \
            NF_MMQ_TC_XCOPY(cn0, (2 - 2 * (PAR)) * NF_MMQ_TC_XU_OF(tc_tn), BS) \
            NF_MMQ_TC_XCOPY(cn1, (3 - 2 * (PAR)) * NF_MMQ_TC_XU_OF(tc_tn), BS) \
            NF_MMQ_TC_PREFETCH(cn0, 0, HAS_OFF, BS, BB, CODE, SCALE, OFF) \
            NF_MMQ_TC_PREFETCH(cn1, 1, HAS_OFF, BS, BB, CODE, SCALE, OFF) \
            COMBINE(2 * (PAR), 2 * (PAR) * NF_MMQ_TC_XU_OF(tc_tn), SG, ASYM, HAS_OFF, EXACT) \
            if ((c) + NF_MMQ_TK < (cend)) { \
                COMBINE(2 * (PAR) + 1, (2 * (PAR) + 1) * NF_MMQ_TC_XU_OF(tc_tn), SG, ASYM, HAS_OFF, EXACT) \
            } \
            NF_MMQ_TC_COMMIT(2 - 2 * (PAR), 0, SG, HAS_OFF) \
            NF_MMQ_TC_COMMIT(3 - 2 * (PAR), 1, SG, HAS_OFF) \
        } else { \
            _Pragma("unroll") \
            for (int tc_g = 0; tc_g < 2 && (c) + tc_g * NF_MMQ_TK < (cend); tc_g++) { \
                const int cn = min((c) + (2 + tc_g) * NF_MMQ_TK, (clast)); \
                NF_MMQ_TC_XCOPY(cn, (2 - 2 * (PAR) + tc_g) * NF_MMQ_TC_XU_OF(tc_tn), BS) \
                NF_MMQ_TC_PREFETCH(cn, 0, HAS_OFF, BS, BB, CODE, SCALE, OFF) \
                COMBINE(2 * (PAR) + tc_g, (2 * (PAR) + tc_g) * NF_MMQ_TC_XU_OF(tc_tn), SG, ASYM, HAS_OFF, \
                        EXACT) \
                NF_MMQ_TC_COMMIT(2 - 2 * (PAR) + tc_g, 0, SG, HAS_OFF) \
            } \
        } \
        nf_mmq_cp_async_wait(); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

/*
 * The tile's chunks [cstart, cend) through COMBINE in one zero-point form, NF_MMQ_TC_DEPTH chunks per
 * step, loaded one group ahead of the combine: the prologue copies the records and commits the weights
 * of the first group; two steps per iteration keep the parity of every step's buffers and stages a
 * compile-time constant. The last group of the range holds fewer chunks when the range's chunk count is
 * not a multiple of NF_MMQ_TC_DEPTH. Every output takes its chunks' terms in chunk order, as with one chunk per step.
 */
#define NF_MMQ_TC_LOOP(cstart, cend, COMBINE, SG, ASYM, HAS_OFF, EXACT, BS, BB, CODE, SCALE, OFF, SETS) \
    if ((cstart) < (cend)) { \
        const int clast = (cend) - NF_MMQ_TK; \
        NF_MMQ_TC_SLOTS(NF_MMQ_TC_SC(SG)) \
        _Pragma("unroll") \
        for (int tc_g = 0; tc_g < NF_MMQ_TC_DEPTH; tc_g++) { \
            const int c0 = min((cstart) + tc_g * NF_MMQ_TK, clast); \
            NF_MMQ_TC_XCOPY(c0, tc_g * NF_MMQ_TC_XU_OF(tc_tn), BS) \
            NF_MMQ_TC_PREFETCH(c0, 0, HAS_OFF, BS, BB, CODE, SCALE, OFF) \
            NF_MMQ_TC_COMMIT(tc_g, 0, SG, HAS_OFF) \
        } \
        nf_mmq_cp_async_wait(); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        for (int c = (cstart); c < (cend); c += 2 * NF_MMQ_TC_DEPTH * NF_MMQ_TK) { \
            NF_MMQ_TC_STEP(c, 0, (cend), clast, COMBINE, SG, ASYM, HAS_OFF, EXACT, BS, BB, CODE, SCALE, OFF, \
                           SETS) \
            if (c + NF_MMQ_TC_DEPTH * NF_MMQ_TK >= (cend)) { \
                break; \
            } \
            NF_MMQ_TC_STEP(c + NF_MMQ_TC_DEPTH * NF_MMQ_TK, 1, (cend), clast, COMBINE, SG, ASYM, HAS_OFF, EXACT, \
                           BS, BB, CODE, SCALE, OFF, SETS) \
        } \
    }

/* The tile's chunks [cstart, cend) into acc through the int8 combine, in the zero-point form of the
 * tile's first record: every staging kernel writes one zero point into all records of a launch
 * (LLM_Quant_MMQ_Int8Asym its static zp, the others 0), so the first record decides the form of the
 * whole tile. The types whose scales are SCALE##_NARROW take the folded conversion of nf_mmq_tc_prod, and
 * CODE##_SETS picks the step's prefetch form (NF_MMQ_TC_STEP). */
#define NF_MMQ_TC_PIPE(cstart, cend, SG, HAS_OFF, BS, BB, CODE, SCALE, OFF) \
    if ((cstart) < (cend) \
        && xtile[((cstart) / NF_MMQ_TK) * NF_MMQ_RC + NF_MMQ_QS + 1 + NF_MMQ_SH] != 0u) { \
        NF_MMQ_TC_LOOP(cstart, cend, NF_MMQ_TC_INT8, SG, 1, HAS_OFF, SCALE##_NARROW, BS, BB, \
                       CODE, SCALE, OFF, NF_MMQ_TC_SETS(CODE##_SETS)) \
    } else { \
        NF_MMQ_TC_LOOP(cstart, cend, NF_MMQ_TC_INT8, SG, 0, HAS_OFF, SCALE##_NARROW, BS, BB, \
                       CODE, SCALE, OFF, NF_MMQ_TC_SETS(CODE##_SETS)) \
    }

/* The lane's accumulators of a tile of NB n8 blocks per warp to global memory, bounds-checked per element. */
template <int NB>
NF_INLINE void nf_mmq_tc_store(NF_GLOBAL float* NF_RESTRICT y, int M, int N, int y_stride, nf_i64 y_off,
                               int m0, int n0, int lane, int wrow0, int wtok0, const float acc[]) {
    const int g = lane >> 2, q = lane & 3;
    for (int m = 0; m < NF_MMQ_TC_NTX; m++) {
        for (int nb = 0; nb < NB; nb++) {
            const float* v = acc + (m * NB + nb) * 4;
            const int r = m0 + wrow0 + 16 * m + g;
            const int c = n0 + wtok0 + nb * 8 + 2 * q;
            if (r < M && c < N) y[y_off + (nf_i64)c * y_stride + r] = v[0];
            if (r < M && c + 1 < N) y[y_off + (nf_i64)(c + 1) * y_stride + r] = v[1];
            if (r + 8 < M && c < N) y[y_off + (nf_i64)c * y_stride + r + 8] = v[2];
            if (r + 8 < M && c + 1 < N) y[y_off + (nf_i64)(c + 1) * y_stride + r + 8] = v[3];
        }
    }
}

/* The work-item's index and warp geometry in a TN-token tile, from the work-item index LID. */
#define NF_MMQ_TC_GEOMETRY(LID, TN) \
    const int lid = (LID); \
    const int lane = lid & 31; \
    const int wrow0 = ((lid >> 5) / NF_MMQ_TC_TG) * NF_MMQ_TC_WROWS; \
    const int wtok0 = ((lid >> 5) % NF_MMQ_TC_TG) * NF_MMQ_TC_WTOK_OF(TN);

/*
 * The work-item index read afresh from the special register (volatile, so never hoisted): a kernel that
 * runs the pipe in a loop over several tiles derives its geometry from it inside the loop, so the
 * compiler keeps the lane-derived addressing per tile instead of hoisting it out of the loop and
 * holding it live across the whole pipe (which costs the stream-k kernels some 80 registers and
 * spills).
 */
NF_INLINE int nf_mmq_tc_fresh_lid(void) {
    int v;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(v));
    return v;
}

/* Shared buffers, prefetch registers and accumulators of a tensor-core tile kernel of TN tokens at scale group
 * SG: the weight buffers and record stages of both parities' chunk groups, and the tile's token count tc_tn
 * that the pipe's record and combine macros read. */
#define NF_MMQ_TC_TILE_DECLS(BS, BB, SG, TN) \
    constexpr int tc_tn = (TN); \
    const nf_i64 row_bytes = (nf_i64)(K / (BS)) * (BB); \
    NF_LOCAL __align__(16) int tc_wq[2 * NF_MMQ_TC_DEPTH * NF_MMQ_TC_CW]; \
    NF_LOCAL __align__(16) float tc_wsc[2 * NF_MMQ_TC_DEPTH * NF_MMQ_TC_SB(SG)]; \
    NF_LOCAL __align__(16) float tc_wof[2 * NF_MMQ_TC_DEPTH * NF_MMQ_TC_SB(SG)]; \
    NF_LOCAL __align__(16) uint tc_xr[2 * NF_MMQ_TC_DEPTH * NF_MMQ_TC_XU_OF(TN)]; \
    int tc_wo[NF_MMQ_TC_CWI], tc_we[NF_MMQ_TC_CWI], tc_so[NF_MMQ_TC_SWI], tc_se[NF_MMQ_TC_SWI]; \
    int tc_su[NF_MMQ_TC_SWI]; \
    int tc_xo[NF_MMQ_TC_XPI_OF(TN)], tc_x16; \
    uint tc_pw0[NF_MMQ_TC_CWI], tc_pw1[NF_MMQ_TC_CWI]; \
    float tc_ps0[NF_MMQ_TC_SWI], tc_ps1[NF_MMQ_TC_SWI]; \
    float tc_po0[NF_MMQ_TC_SWI], tc_po1[NF_MMQ_TC_SWI]; \
    float acc[NF_MMQ_TC_ACC_OF(TN)];

/* Declarations and warp geometry of a one-tile tensor-core kernel over the device tile; GEMM_ARGS is its
 * body. */
#define NF_MMQ_TC_KERNEL_BODY(BS, BB, SG, GEMM_ARGS) \
    NF_MMQ_TC_GEOMETRY(get_local_id(0), NF_MMQ_TN) \
    NF_MMQ_TC_TILE_DECLS(BS, BB, SG, NF_MMQ_TN) \
    for (int i = 0; i < NF_MMQ_TC_ACC; i++) \
        acc[i] = 0.0f; \
    GEMM_ARGS

/* The plain and split-K kernels of the tensor-core tile; PIPE runs the tile's chunks (NF_MMQ_TC_PIPE
 * for the int8 types, NF_FP8_TC_PIPE of kernels/llm/matmul_fp8.cl for the E4M3 lane). */
#define NF_MMQ_TC_GEMM_NAMED(NAME, PIPE, BS, BB, SG, HAS_OFF, CODE, SCALE, OFF) \
NF_KERNEL __launch_bounds__(NF_MMQ_WG) void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT y) { \
    NF_MMQ_TC_KERNEL_BODY((BS), (BB), (SG), \
        const int m0 = (int)get_group_id(0) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * NF_MMQ_TN; \
        const NF_GLOBAL uchar* wtile = weight + (nf_i64)m0 * row_bytes; \
        const NF_GLOBAL uchar* wtile_u = wtile; \
        const int tc_rmask = NF_MMQ_TM - 1; \
        const NF_GLOBAL uint* xtile = q8 + (nf_i64)q_off + (nf_i64)n0 * q_stride * NF_MMQ_RC; \
        PIPE(0, K, (SG), (HAS_OFF), (BS), (BB), CODE, SCALE, OFF) \
        nf_mmq_tc_store<NF_MMQ_TC_NB>(y, M, N, y_stride, (nf_i64)y_off, m0, n0, lane, wrow0, wtok0, \
                                      acc); \
    ) \
}

#define NF_MMQ_TC_KSPLIT_NAMED(NAME, PIPE, BS, BB, SG, HAS_OFF, CODE, SCALE, OFF) \
NF_KERNEL __launch_bounds__(NF_MMQ_WG) void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int k_slice, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT partial) { \
    NF_MMQ_TC_KERNEL_BODY((BS), (BB), (SG), \
        const int rtiles = (M + NF_MMQ_TM - 1) / NF_MMQ_TM; \
        const int slice = (int)get_group_id(0) / rtiles; \
        const int m0 = ((int)get_group_id(0) - slice * rtiles) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * NF_MMQ_TN; \
        const int k0 = slice * k_slice; \
        const int k1 = k0 + k_slice < K ? k0 + k_slice : K; \
        const NF_GLOBAL uchar* wtile = weight + (nf_i64)m0 * row_bytes; \
        const NF_GLOBAL uchar* wtile_u = wtile; \
        const int tc_rmask = NF_MMQ_TM - 1; \
        const NF_GLOBAL uint* xtile = q8 + (nf_i64)q_off + (nf_i64)n0 * q_stride * NF_MMQ_RC; \
        PIPE(k0, k1, (SG), (HAS_OFF), (BS), (BB), CODE, SCALE, OFF) \
        nf_mmq_tc_store<NF_MMQ_TC_NB>(partial, M, N, M, (nf_i64)slice * M * N, m0, n0, lane, wrow0, wtok0, \
                                      acc); \
    ) \
}

#endif /* NF_MMQ_TENSOR */

/* ---- kernel generator ---- */

/*
 * The code tiles are declared as int arrays so every 4-byte fragment / dp4a load is aligned; the
 * (scale, offset) tiles hold NF_MMQ_SH pairs per row, and each stage gathers the tile's whole
 * staged record block (NF_MMQ_RC units per token: the code ints the B fragments and packed dots
 * read, the chunk scale d, the half code sums and the zero point) into the shared record tile xrs
 * (one copy; the pipe exposes it as xr). The code / (scale, offset) / code-sum
 * tiles exist in NF_MMQ_PIPE_BUF copies, one per in-flight chunk of the pipelined stage
 * (NF_MMQ_PIPE). row_bytes and the record tiles belong to the kernel; TN is the kernel's token tile
 * (the record tiles size to it).
 */
#define NF_MMQ_KERNEL_BODY(TN, BS, BB, GEMM_ARGS) \
    const int lid = get_local_id(0); \
    const int tx = lid % NF_MMQ_TX, ty = lid / NF_MMQ_TX; \
    const nf_i64 row_bytes = (nf_i64)(K / (BS)) * (BB); \
    NF_LOCAL int wq_w[NF_MMQ_PIPE_BUF * NF_MMQ_WS / 4]; \
    NF_LOCAL float wsc[NF_MMQ_PIPE_BUF * NF_MMQ_TM * NF_MMQ_SH]; \
    NF_LOCAL float wof[NF_MMQ_PIPE_BUF * NF_MMQ_TM * NF_MMQ_SH]; \
    NF_LOCAL uint xrs[NF_MMQ_XR_BUF * (TN) * NF_MMQ_RC]; \
    NF_LOCAL int wsu[NF_MMQ_PIPE_BUF * NF_MMQ_TM * NF_MMQ_SH]; \
    NF_LOCAL_PTR nf_i8* wq = (NF_LOCAL_PTR nf_i8*)wq_w; \
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
    nf_mmq_combine(wq + (sel) * NF_MMQ_WS, xr, n0, (TN), N, \
                   wsc + (sel) * NF_MMQ_TM * NF_MMQ_SH, wof + (sel) * NF_MMQ_TM * NF_MMQ_SH, \
                   wsu + (sel) * NF_MMQ_TM * NF_MMQ_SH, lid, acc, (SG))

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

#if NF_MMQ_TENSOR
#define NF_MMQ_GEMM(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_TC_GEMM_NAMED(T##_MatMul_MMQ, NF_MMQ_TC_PIPE, BS, BB, SG, T##_MMQ_HAS_OFF, CODE, SCALE, OFF)
#else
#define NF_MMQ_GEMM(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_GEMM_TN(T##_MatMul_MMQ, NF_MMQ_TN, BS, BB, SG, CODE, SCALE, OFF)
#endif

#define NF_MMQ_GEMM_S(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_GEMM_TN(T##_MatMul_MMQ_S, 16, BS, BB, SG, CODE, SCALE, OFF)

#define NF_MMQ_GEMM_S64(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_GEMM_TN(T##_MatMul_MMQ_S64, 64, BS, BB, SG, CODE, SCALE, OFF)

/*
 * The split-K twin of one NAME over the TN-token tile: dimension 0 of the dispatch holds
 * ceil(M / NF_MMQ_TM) * split workgroups, group gid0 = rt + rtiles * slice covers rows
 * rt * NF_MMQ_TM of slice `slice` over k in [slice * k_slice, min((slice + 1) * k_slice, K)) and
 * writes the live [M x N] positions of its slice plane at partial[(slice * N + t) * M + r]
 * (y_stride = M), so the host reduce (LLM_SplitK_Reduce, kernels/llm/split_k_reduce.cl) sums the
 * planes into the strided output. The launch geometry is the tile's plain kernel's with dimension 0
 * scaled by the split count (gguf.ops.gemm.SplitKGemmPlan): the plain kernels' split runs the
 * device tile (NF_MMQ_TN tokens), the small-token tiles' split (_SKSplit) runs 16 tokens.
 */
#define NF_MMQ_KSPLIT_TN(NAME, TN, BS, BB, SG, CODE, SCALE, OFF) \
NF_KERNEL void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int k_slice, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT partial) { \
    NF_MMQ_KERNEL_BODY((TN), (BS), (BB), \
        const int rtiles = (M + NF_MMQ_TM - 1) / NF_MMQ_TM; \
        const int slice = (int)get_group_id(0) / rtiles; \
        const int m0 = ((int)get_group_id(0) - slice * rtiles) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * (TN); \
        const int k0 = slice * k_slice; \
        const int k1 = k0 + k_slice < K ? k0 + k_slice : K; \
        NF_MMQ_PIPE(k0, k1, (TN), NF_MMQ_INT8_COMBINE, (BS), (BB), CODE, SCALE, OFF, (TN), (SG)) \
        nf_mmq_store(partial, M, N, M, (nf_i64)slice * M * N, m0, n0, (TN), lid, acc); \
    ) \
}

#if NF_MMQ_TENSOR
#define NF_MMQ_GEMM_KSPLIT(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_TC_KSPLIT_NAMED(T##_MatMul_MMQ_KSplit, NF_MMQ_TC_PIPE, BS, BB, SG, T##_MMQ_HAS_OFF, CODE, SCALE, \
                           OFF)
#else
#define NF_MMQ_GEMM_KSPLIT(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_KSPLIT_TN(T##_MatMul_MMQ_KSplit, NF_MMQ_TN, BS, BB, SG, CODE, SCALE, OFF)
#endif

#define NF_MMQ_GEMM_SKSPLIT(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_KSPLIT_TN(T##_MatMul_MMQ_SKSplit, 16, BS, BB, SG, CODE, SCALE, OFF)

#define NF_MMQ_GEMM_SKSPLIT64(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_KSPLIT_TN(T##_MatMul_MMQ_SKSplit64, 64, BS, BB, SG, CODE, SCALE, OFF)

#endif /* NF_MATMUL_MMQ_FAMILY_H */
