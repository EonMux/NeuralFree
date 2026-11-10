/* matmul_mmq_stream.cl - T_MatMul_MMQ_StreamK, the stream-k form of the MMQ GEMM's tensor-core tile,
 * T_MatMul_MMQ_StreamK_GateUp, its fused gate/up SwiGLU form, and LLM_MMQ_StreamK_Fixup and
 * LLM_MMQ_StreamK_Fixup_GateUp, which complete their split tiles ([cuda]: CUDA sm_80+ only).
 * Requires: common/dialect.h, common/config.h, common/math.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_legacy.h, common/quant_decode_k.h, common/matmul_mmq_config.h, common/matmul_mmq_family.h, common/matmul_mmq_types.h */
/*
 * The stream-k kernels run the plain T_MatMul_MMQ tile of every MMQ weight type (kernels/llm/matmul_mmq.cl)
 * over the launch's sequence of (tile, chunk) steps split evenly across the device's resident blocks; the
 * host takes them for a staged launch of at least one device tile of tokens whose tile grid would leave the
 * last wave of blocks partly empty, and the gate/up form for the dense FFN's gate and up products of a
 * prefill; it keeps the plain grid and the separate products under batch invariance (a split tile's sum
 * depends on the grid). The file is CUDA-only, and the kernels exist on the sm_80+ segments the tensor tile
 * compiles for.
 *
 * The split tiles are completed by a separate pass, as llama.cpp does (mul_mat_q then
 * mul_mat_q_stream_k_fixup): summing the planes inside the stream-k kernel would add their loads to the tile's
 * register budget and slow its main loop by more than the extra launch costs.
 */

#if NF_MMQ_TENSOR
/*
 * Stream-k decomposition of the tile grid (llama.cpp's mul_mat_q stream-k form, ggml-cuda/mmq.cuh): the
 * launch's output tiles, row tile major with the token tile fastest (the token tiles of one weight row tile
 * run back to back and share its weight stream in L2), times their K / NF_MMQ_TK chunks form one sequence,
 * and block b of nb takes steps [nf_mmq_streamk_start(b), nf_mmq_streamk_start(b + 1)). Every block then
 * carries the same work whatever the tile count, so the last wave of a grid whose tiles are not a multiple
 * of the device's resident blocks no longer runs part-empty. nb never exceeds the step count, so every block
 * runs at least one step.
 */
NF_INLINE int nf_mmq_streamk_start(int b, int nb, int total) {
    return (int)(((nf_i64)b * total) / nb);
}

/* The block whose steps hold step s: the largest b with nf_mmq_streamk_start(b) <= s. */
NF_INLINE int nf_mmq_streamk_owner(int s, int nb, int total) {
    return (int)((((nf_i64)s + 1) * nb - 1) / total);
}

/*
 * Fixup planes (TM x TN floats each, token-major): the plain form gives block b one plane, b, for its last
 * segment when that stops inside a tile; its first segment, when it finishes a tile another block started,
 * goes straight into y, and LLM_MMQ_StreamK_Fixup adds the other contributors' planes onto it. The gate/up
 * form cannot add onto activated outputs, so block b keeps that finishing segment in its head plane 2b and the
 * stopping one in its tail plane 2b + 1, and LLM_MMQ_StreamK_Fixup_GateUp activates the sums. A block's other
 * segments are whole tiles.
 */
#define NF_MMQ_SK_PLANE(p) ((nf_i64)(p) * NF_MMQ_TM * NF_MMQ_TN)

/*
 * The gate/up form's store of a tile its block ran whole. The products of a gate row and of its up row sit
 * in different warps (the warps of tile rows below NF_MMQ_TM / 2 and the warps above), so they meet in
 * shared memory, NF_MMQ_GU_PT tokens of the tile per pass in the record stages (token-major, the row
 * stride padded by 4 floats so the lanes of a store hit distinct banks); every work-item then forms
 * silu(gate) * up of a run of consecutive output rows, one coalesced row segment per token. The barriers
 * open each pass after the pipe's last reads of the stages and close the last pass before the next
 * segment's copies land in them.
 */
#define NF_MMQ_GU_PT 32
#define NF_MMQ_GU_XS (NF_MMQ_TM + 4)
#if NF_MMQ_TN % NF_MMQ_GU_PT != 0 || NF_MMQ_GU_PT * NF_MMQ_GU_XS > 3 * NF_MMQ_TN * NF_Q8_REC(NF_MMQ_TK)
#error "the gate/up exchange passes must tile NF_MMQ_TN and fit the record stages"
#endif
NF_INLINE void nf_mmq_gate_up_store(NF_LOCAL_PTR float* xch, NF_GLOBAL float* NF_RESTRICT y, int M, int N,
                                    int y_stride, nf_i64 y_off, int m0, int n0, int lid, int lane, int wrow0,
                                    int wtok0, const float acc[]) {
    const int g = lane >> 2, q = lane & 3;
    const int half = NF_MMQ_TM / 2;
    for (int p = 0; p < NF_MMQ_TN / NF_MMQ_GU_PT; p++) {
        const int t0 = p * NF_MMQ_GU_PT;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int nb = 0; nb < NF_MMQ_TC_NB; nb++) {
            const int tb = wtok0 + nb * 8 - t0;
            if (tb >= 0 && tb < NF_MMQ_GU_PT) {
                for (int m = 0; m < NF_MMQ_TC_NTX; m++) {
                    for (int i = 0; i < 4; i++) {
                        const int r = wrow0 + 16 * m + g + 8 * (i >> 1);
                        xch[(tb + 2 * q + (i & 1)) * NF_MMQ_GU_XS + r] = acc[(m * NF_MMQ_TC_NB + nb) * 4 + i];
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int idx = lid; idx < half * NF_MMQ_GU_PT; idx += NF_MMQ_WG) {
            const int r = idx % half;
            const int t = idx / half;
            if (m0 + r < M && n0 + t0 + t < N) {
                const float gv = xch[t * NF_MMQ_GU_XS + r];
                const float uv = xch[t * NF_MMQ_GU_XS + half + r];
                y[y_off + (nf_i64)(n0 + t0 + t) * y_stride + m0 + r] = nf_silu(gv) * uv;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/* Stores of a tile run whole: the plain form's straight from the accumulators, the gate/up form's through
 * the pairing exchange. */
#define NF_MMQ_SK_WHOLE(m0, n0) \
    nf_mmq_tc_store(y, M, N, y_stride, (nf_i64)y_off, m0, n0, lane, wrow0, wtok0, acc);
#define NF_MMQ_SK_WHOLE_GATE_UP(m0, n0) \
    nf_mmq_gate_up_store((NF_LOCAL_PTR float*)tc_xr, y, M, N, y_stride, (nf_i64)y_off, m0, n0, lid, lane, \
                         wrow0, wtok0, acc);

/* Stores of a segment of a split tile into the block's plane (NF_MMQ_SK_PLANE), HEAD when it runs the tile's
 * last chunk: the plain form stores that segment through WHOLE (HEAD_IN_Y) and only its tails here. */
#define NF_MMQ_SK_SPLIT(m0, n0, HEAD) \
    nf_mmq_tc_store(fixup, NF_MMQ_TM, NF_MMQ_TN, NF_MMQ_TM, NF_MMQ_SK_PLANE(b), 0, 0, lane, wrow0, wtok0, acc);
#define NF_MMQ_SK_SPLIT_GATE_UP(m0, n0, HEAD) \
    nf_mmq_tc_store(fixup, NF_MMQ_TM, NF_MMQ_TN, NF_MMQ_TM, NF_MMQ_SK_PLANE((HEAD) ? 2 * b : 2 * b + 1), 0, 0, \
                    lane, wrow0, wtok0, acc);

/*
 * The body of a stream-k kernel over output tiles of RT rows: segment s of tile t runs tile rows
 * [m0, m0 + NF_MMQ_TM) of the weight, m0 = t / ntj * RT, through the plain kernel's chunk pipe, the tile
 * rows mapping to weight rows as NF_MMQ_TC_SLOTS describes (RMASK its tc_rmask, WUP the up weight's base).
 * A block stores the tiles it runs whole through WHOLE, and with HEAD_IN_Y also the segment that finishes a
 * split tile, so a segment and its store form one branch as in the plain kernel; the other segments of split
 * tiles go through SPLIT, and the fixup pass after the launch completes the split tiles. The plain form keeps
 * these two store sites: a third raises the register count of the larger IQ types past the point where they
 * spill.
 */
#define NF_MMQ_TC_STREAMK_BODY(PIPE, BS, BB, SG, HAS_OFF, CODE, SCALE, OFF, RT, WUP, RMASK, WHOLE, HEAD_IN_Y, \
                               SPLIT) \
    NF_MMQ_TC_TILE_DECLS((BS), (BB)) \
    { \
        const int kc = K / NF_MMQ_TK; \
        const int ntj = (N + NF_MMQ_TN - 1) / NF_MMQ_TN; \
        const int total = (M + (RT) - 1) / (RT) * ntj * kc; \
        const int b = (int)get_group_id(0); \
        const int nb = (int)get_num_groups(0); \
        const int kstop = nf_mmq_streamk_start(b + 1, nb, total); \
        for (int kbc = nf_mmq_streamk_start(b, nb, total); kbc < kstop;) { \
            NF_MMQ_TC_GEOMETRY(nf_mmq_tc_fresh_lid()) \
            const int tile = kbc / kc; \
            const int kb = kbc - tile * kc; \
            const int ke = min(kc, kb + (kstop - kbc)); \
            const int m0 = tile / ntj * (RT); \
            const int n0 = tile % ntj * NF_MMQ_TN; \
            const NF_GLOBAL uchar* wtile = weight + (nf_i64)m0 * row_bytes; \
            const NF_GLOBAL uchar* wtile_u = (WUP) + (nf_i64)m0 * row_bytes; \
            const int tc_rmask = (RMASK); \
            const NF_GLOBAL uint* xtile = q8 + (nf_i64)q_off + (nf_i64)n0 * q_stride * NF_MMQ_RC; \
            for (int i = 0; i < NF_MMQ_TC_ACC; i++) \
                acc[i] = 0.0f; \
            PIPE(kb * NF_MMQ_TK, ke * NF_MMQ_TK, (SG), (HAS_OFF), (BS), (BB), CODE, SCALE, OFF) \
            if (ke == kc && ((HEAD_IN_Y) || kb == 0)) { \
                WHOLE(m0, n0) \
            } else { \
                SPLIT(m0, n0, ke == kc) \
            } \
            kbc += ke - kb; \
        } \
    }

/*
 * The stream-k kernel of the tensor-core tile: global {nb * NF_MMQ_WG, 1}, local {NF_MMQ_WG, 1} with
 * nb <= the launch's tile count times its chunk count (the host picks the device's resident blocks).
 * Arguments as T_MatMul_MMQ plus fixup, nb planes of NF_MMQ_TM x NF_MMQ_TN floats (NF_MMQ_SK_PLANE), which
 * LLM_MMQ_StreamK_Fixup adds into the split tiles after the launch. Every segment runs the plain kernel's
 * chunk pipe, so a tile run by one block equals T_MatMul_MMQ's output bit for bit; a split tile differs by
 * the fp32 rounding of adding its partial sums.
 */
#define NF_MMQ_TC_STREAMK_NAMED(NAME, PIPE, BS, BB, SG, HAS_OFF, CODE, SCALE, OFF) \
NF_KERNEL __launch_bounds__(NF_MMQ_WG) void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT y, \
    NF_GLOBAL float* NF_RESTRICT fixup) { \
    NF_MMQ_TC_STREAMK_BODY(PIPE, BS, BB, SG, HAS_OFF, CODE, SCALE, OFF, NF_MMQ_TM, weight, NF_MMQ_TM - 1, \
                           NF_MMQ_SK_WHOLE, 1, NF_MMQ_SK_SPLIT) \
}

/*
 * The gate/up SwiGLU form of the stream-k kernel, llama.cpp's fused gate/up MMQ (mul_mat_q with x_gate and
 * its GLU write-back): y[t * y_stride + r] = silu(gate[r] . x_t) * (up[r] . x_t) for the M rows of two
 * weights of one type and shape (weight is the gate, weight_up the up projection, both M x K), the dense
 * FFN's LLM_GLU over the separate products in one launch. Each output tile of NF_MMQ_TM / 2 rows runs the
 * NF_MMQ_TM-row tensor tile over the gate rows and the up rows of the same indices, so the products of a pair
 * land in one block; the arguments and geometry follow T_MatMul_MMQ_StreamK, weight_up last, with 2 * nb fixup
 * planes, which LLM_MMQ_StreamK_Fixup_GateUp activates into the split tiles. The products of a tile run
 * whole are T_MatMul_MMQ's of the two weights bit for bit, so its outputs equal LLM_GLU's over the plain
 * kernel's products; a split tile differs by the fp32 rounding of adding its partial sums. It replaces two
 * stream-k launches and the LLM_GLU pass, whose 2 x n x n_ff products it neither writes nor reads back.
 */
#define NF_MMQ_TC_STREAMK_GATE_UP_NAMED(NAME, PIPE, BS, BB, SG, HAS_OFF, CODE, SCALE, OFF) \
NF_KERNEL __launch_bounds__(NF_MMQ_WG) void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT y, \
    NF_GLOBAL float* NF_RESTRICT fixup, \
    NF_GLOBAL const uchar* NF_RESTRICT weight_up) { \
    NF_MMQ_TC_STREAMK_BODY(PIPE, BS, BB, SG, HAS_OFF, CODE, SCALE, OFF, NF_MMQ_TM / 2, weight_up, \
                           NF_MMQ_TM / 2 - 1, NF_MMQ_SK_WHOLE_GATE_UP, 0, NF_MMQ_SK_SPLIT_GATE_UP) \
}

#define NF_MMQ_GEMM_STREAMK(T, BS, BB, SG, CODE, SCALE, OFF) \
    NF_MMQ_TC_STREAMK_NAMED(T##_MatMul_MMQ_StreamK, NF_MMQ_TC_PIPE, BS, BB, SG, T##_MMQ_HAS_OFF, CODE, SCALE, \
                            OFF) \
    NF_MMQ_TC_STREAMK_GATE_UP_NAMED(T##_MatMul_MMQ_StreamK_GateUp, NF_MMQ_TC_PIPE, BS, BB, SG, T##_MMQ_HAS_OFF, \
                                    CODE, SCALE, OFF)

NF_MMQ_GEMM_STREAMK(Q4_0, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES, 32, Q4_0_MMQ_CODE, Q4_0_MMQ_SCALE, Q4_0_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q4_1, Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES, 32, Q4_1_MMQ_CODE, Q4_1_MMQ_SCALE, Q4_1_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q5_0, Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, 32, Q5_0_MMQ_CODE, Q5_0_MMQ_SCALE, Q5_0_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q5_1, Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES, 32, Q5_1_MMQ_CODE, Q5_1_MMQ_SCALE, Q5_1_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q8_0, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, 32, Q8_0_MMQ_CODE, Q8_0_MMQ_SCALE, Q8_0_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(I8_B128, I8_B128_BLOCK_SIZE, I8_B128_BLOCK_BYTES, 32, I8_B128_MMQ_CODE, I8_B128_MMQ_SCALE, I8_B128_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q2_K, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES, 16, Q2_K_MMQ_CODE, Q2_K_MMQ_SCALE, Q2_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q3_K, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES, 16, Q3_K_MMQ_CODE, Q3_K_MMQ_SCALE, Q3_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, 32, Q4_K_MMQ_CODE, Q4_K_MMQ_SCALE, Q4_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, 32, Q5_K_MMQ_CODE, Q5_K_MMQ_SCALE, Q5_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, 16, Q6_K_MMQ_CODE, Q6_K_MMQ_SCALE, Q6_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ3_XXS, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES, 32, IQ3_XXS_MMQ_CODE, IQ3_XXS_MMQ_SCALE, IQ3_XXS_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ3_S, IQ3_S_BLOCK_SIZE, IQ3_S_BLOCK_BYTES, 32, IQ3_S_MMQ_CODE, IQ3_S_MMQ_SCALE, IQ3_S_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ2_XXS, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES, 32, IQ2_XXS_MMQ_CODE, IQ2_XXS_MMQ_SCALE, IQ2_XXS_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ2_XS, IQ2_XS_BLOCK_SIZE, IQ2_XS_BLOCK_BYTES, 16, IQ2_XS_MMQ_CODE, IQ2_XS_MMQ_SCALE, IQ2_XS_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ2_S, IQ2_S_BLOCK_SIZE, IQ2_S_BLOCK_BYTES, 16, IQ2_S_MMQ_CODE, IQ2_S_MMQ_SCALE, IQ2_S_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ1_S, IQ1_S_BLOCK_SIZE, IQ1_S_BLOCK_BYTES, 32, IQ1_S_MMQ_CODE, IQ1_S_MMQ_SCALE, IQ1_S_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, 32, IQ4_NL_MMQ_CODE, IQ4_NL_MMQ_SCALE, IQ4_NL_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, 32, IQ4_XS_MMQ_CODE, IQ4_XS_MMQ_SCALE, IQ4_XS_MMQ_OFF)
/* Outputs of a tile one work-item of a fixup adds: a tile's outputs split into slices of
 * NF_MMQ_SK_FIX_E * NF_MMQ_WG contiguous plane elements, one work-group each (NF_MMQ_SK_FIX_SLICES of the
 * plain tile, NF_MMQ_SK_FIX_GU_SLICES of the gate/up tile's NF_MMQ_TM / 2 rows), so a split tile's planes are
 * read with that many loads in flight per work-item rather than as one work-group's long dependent chain. */
#define NF_MMQ_SK_FIX_E 4
#define NF_MMQ_SK_FIX_SLICES (NF_MMQ_TM * NF_MMQ_TN / (NF_MMQ_WG * NF_MMQ_SK_FIX_E))
#define NF_MMQ_SK_FIX_GU_SLICES (NF_MMQ_TM / 2 * NF_MMQ_TN / (NF_MMQ_WG * NF_MMQ_SK_FIX_E))
#if NF_MMQ_TM / 2 * NF_MMQ_TN % (NF_MMQ_WG * NF_MMQ_SK_FIX_E) != 0
#error "the stream-k fixup slices must tile the gate/up output tile"
#endif

/*
 * The tile that block b of nb finishes in a launch of total steps, kc chunks per tile, or -1: a tile it did
 * not start whose last chunk its first segment runs (a first segment that also stops inside that tile leaves
 * its plane to the block after it). Every block runs at least one step.
 */
NF_INLINE int nf_mmq_streamk_finished(int b, int nb, int total, int kc) {
    const int k0 = nf_mmq_streamk_start(b, nb, total);
    const int k1 = nf_mmq_streamk_start(b + 1, nb, total);
    const int tile = k0 / kc;
    return k0 == k1 || k0 % kc == 0 || (tile == k1 / kc && k1 % kc != 0) ? -1 : tile;
}

/*
 * The stream-k fixup: global {nb * NF_MMQ_WG, NF_MMQ_SK_FIX_SLICES}, local {NF_MMQ_WG, 1} over the nb blocks
 * of the T_MatMul_MMQ_StreamK launch with the same M, K, N and output view. Work-group (b, s) acts when block
 * b finished a tile (nf_mmq_streamk_finished): it sums slice s of the planes of the blocks before it that ran
 * the rest of the tile, nearest first from +0 (each one's last segment lies in the tile), and adds the sum
 * onto the partial block b stored in y; the work-groups of the other blocks return at once.
 */
NF_KERNEL __launch_bounds__(NF_MMQ_WG) void LLM_MMQ_StreamK_Fixup(
    const int M, const int K, const int N, const int y_stride, const int y_off,
    NF_GLOBAL const float* NF_RESTRICT fixup,
    NF_GLOBAL float* NF_RESTRICT y) {
    const int kc = K / NF_MMQ_TK;
    const int ntj = (N + NF_MMQ_TN - 1) / NF_MMQ_TN;
    const int total = (M + NF_MMQ_TM - 1) / NF_MMQ_TM * ntj * kc;
    const int b = (int)get_group_id(0);
    const int nb = (int)get_num_groups(0);
    const int tile = nf_mmq_streamk_finished(b, nb, total, kc);
    if (tile < 0) {
        return;
    }
    const int f0 = nf_mmq_streamk_owner(tile * kc, nb, total);
    const int e0 = (int)get_group_id(1) * NF_MMQ_SK_FIX_E * NF_MMQ_WG + (int)get_local_id(0);
    float sum[NF_MMQ_SK_FIX_E];
    for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
        sum[i] = 0.0f;
    }
    for (int p = b - 1; p >= f0; p--) {
        const NF_GLOBAL float* plane = fixup + NF_MMQ_SK_PLANE(p) + e0;
        for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
            sum[i] += plane[i * NF_MMQ_WG];
        }
    }
    const int m0 = tile / ntj * NF_MMQ_TM;
    const int n0 = tile % ntj * NF_MMQ_TN;
    for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
        const int e = e0 + i * NF_MMQ_WG;
        const int r = m0 + e % NF_MMQ_TM;
        const int c = n0 + e / NF_MMQ_TM;
        if (r < M && c < N) {
            y[(nf_i64)y_off + (nf_i64)c * y_stride + r] += sum[i];
        }
    }
}

/*
 * The gate/up form's fixup: global {nb * NF_MMQ_WG, NF_MMQ_SK_FIX_GU_SLICES}, local {NF_MMQ_WG, 1} over the
 * T_MatMul_MMQ_StreamK_GateUp launch with the same M, K, N and output view, its output tiles NF_MMQ_TM / 2
 * rows. Work-group (b, s) of a block b that finished a tile sums, for slice s of the tile's outputs, the tail
 * planes of the blocks before it nearest first from +0 and adds them to b's head plane - the gate product of
 * output row r in plane row r, the up product in row NF_MMQ_TM / 2 + r - then writes
 * silu(gate) * up, LLM_GLU's SWIGLU.
 */
NF_KERNEL __launch_bounds__(NF_MMQ_WG) void LLM_MMQ_StreamK_Fixup_GateUp(
    const int M, const int K, const int N, const int y_stride, const int y_off,
    NF_GLOBAL const float* NF_RESTRICT fixup,
    NF_GLOBAL float* NF_RESTRICT y) {
    const int half = NF_MMQ_TM / 2;
    const int kc = K / NF_MMQ_TK;
    const int ntj = (N + NF_MMQ_TN - 1) / NF_MMQ_TN;
    const int total = (M + half - 1) / half * ntj * kc;
    const int b = (int)get_group_id(0);
    const int nb = (int)get_num_groups(0);
    const int tile = nf_mmq_streamk_finished(b, nb, total, kc);
    if (tile < 0) {
        return;
    }
    const int f0 = nf_mmq_streamk_owner(tile * kc, nb, total);
    const int e0 = (int)get_group_id(1) * NF_MMQ_SK_FIX_E * NF_MMQ_WG + (int)get_local_id(0);
    float gate[NF_MMQ_SK_FIX_E], up[NF_MMQ_SK_FIX_E];
    for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
        gate[i] = 0.0f;
        up[i] = 0.0f;
    }
    for (int p = b - 1; p >= f0; p--) {
        const NF_GLOBAL float* plane = fixup + NF_MMQ_SK_PLANE(2 * p + 1);
        for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
            const int e = e0 + i * NF_MMQ_WG;
            const int pe = e / half * NF_MMQ_TM + e % half;
            gate[i] += plane[pe];
            up[i] += plane[pe + half];
        }
    }
    const NF_GLOBAL float* head = fixup + NF_MMQ_SK_PLANE(2 * b);
    const int m0 = tile / ntj * half;
    const int n0 = tile % ntj * NF_MMQ_TN;
    for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
        const int e = e0 + i * NF_MMQ_WG;
        const int pe = e / half * NF_MMQ_TM + e % half;
        const int r = m0 + e % half;
        const int c = n0 + e / half;
        if (r < M && c < N) {
            y[(nf_i64)y_off + (nf_i64)c * y_stride + r] = nf_silu(head[pe] + gate[i]) * (head[pe + half] + up[i]);
        }
    }
}
#endif /* NF_MMQ_TENSOR */
