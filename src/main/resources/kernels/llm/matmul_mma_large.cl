/* matmul_mma_large.cl - {T}_MatMul_MMA_L: the large tensor tile of the dense 16-bit weight GEMM for prefill
 * ([cuda]: CUDA sm_80+ only).
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/matmul_mma_family.h */
/*
 *   T_MatMul_MMA_L(i M, i K, i N, i x_stride, i x_off, i y_stride, i y_off,
 *                  R uchar weight, R float x, W float y)
 *   y[y_off + t*y_stride + r] = sum_{k < K} W[r][k] * x[x_off + t*x_stride + k],  r < M, t < N
 *
 * The arguments, semantics and arithmetic of T##_MatMul_MMA (common/matmul_mma_family.h): fp16 (F16) or
 * bfloat16 (BF16) operands, the activations rounded to nearest even, mma.sync.m16n8k16 products accumulated
 * in fp32 one 16-deep chunk at a time in increasing k. Every output therefore receives the same chunk sums
 * in the same order as from T##_MatMul_MMA, and the two kernels agree bit for bit.
 *
 * Tile: a work-group of NF_MMAL_WG = 256 work-items (8 warps) computes NF_MMAL_TM = 128 weight rows x
 * NF_MMAL_TN = 64 tokens in NF_MMAL_TK = 32-deep stages; warp w covers rows (w & 3) * 32 and tokens
 * (w >> 2) * 32 (two 16-row by four 8-token mma tiles per 16-deep chunk). The weight tile and the
 * activation tile are both stored row-major along k (the activation tile token-major), so the A and B
 * fragments both load with ldmatrix.x4. The row stride of NF_MMAL_S = 40 halfs (80 bytes) puts the eight
 * rows of every ldmatrix matrix on distinct 16-byte bank groups, and both stage arrays are declared 16-byte
 * aligned, which the 16-byte stores and the ldmatrix row addresses need.
 *
 * Pipeline: two shared stages and one barrier per stage. Every work-item loads its share of stage s + 1
 * (two 16-byte weight chunks and two runs of four activations, converted to 16-bit at the load) into
 * registers before it multiplies stage s, and stores them into the other shared stage at the top of the
 * next iteration; the barrier there orders that store after every warp's reads of the stage two steps back.
 * The weight rows load as 16-byte vectors when K is a multiple of 8 and the weight is 16-byte aligned, the
 * activations as float4 when x_stride and x_off are multiples of 4; any other view loads per element.
 * Rows past M clamp to row M - 1 (their outputs are not stored), tokens past N and depths past K load 0.
 *
 * Block order: the dispatch's group ids are read as one linear index with the token tile fastest, so the
 * token tiles of one weight row tile run back to back and share its weight rows in L2 instead of each
 * streaming them from DRAM.
 *
 * Residency: __launch_bounds__(256, 3) holds the kernel to 80 registers without spills (sm_120), so three
 * work-groups share a multiprocessor (90 KiB of shared memory); against two (102 registers) that measured
 * 15-38% faster on 512-token projections of 2560 to 14336 rows and within 3% on grids under one wave.
 *
 * Launch contract: 2-D, global = {ceil(M / 128) * 256, ceil(N / 64)}, local = {256, 1}; shared memory
 * 2 * (128 + 64) * 40 halfs = 30 KiB. The host (gguf.gpu.forward.FastGemmLanes#largeTileKernel) takes this tile
 * on CUDA sm_80+ for launches of at least one token tile that split-K does not split, and keeps T##_MatMul_MMA
 * and its split-K twin for every other launch.
 */

#if defined(__CUDACC__) && !defined(__HIP__)
#if defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

#define NF_MMAL_TM 128                   /* weight rows per work-group tile */
#define NF_MMAL_TN 64                    /* tokens per work-group tile */
#define NF_MMAL_TK 32                    /* reduction depth of one stage */
#define NF_MMAL_WG 256                   /* work-items per work-group */
#define NF_MMAL_S 40                     /* padded half stride of a tile row */

/*
 * The work-item's share of one weight stage: chunks lid and lid + 256 of the tile's TM x (TK / 8) 16-byte
 * chunks, chunk c holding halfs [k + 8 * (c & 3), +8) of tile row c >> 2.
 */
NF_INLINE void nf_mmal_load_w(uint4 wr[2], NF_GLOBAL const uchar* NF_RESTRICT weight, int M, int K, int m0,
                              int k, int lid, int wvec) {
    for (int i = 0; i < 2; i++) {
        const int idx = lid + NF_MMAL_WG * i;
        const int row = min(m0 + (idx >> 2), M - 1);
        const int kk = k + 8 * (idx & 3);
        const nf_i64 at = (nf_i64)row * K + kk;
        if (wvec) {
            wr[i] = kk < K ? *(NF_GLOBAL const uint4*)(weight + 2 * at) : make_uint4(0u, 0u, 0u, 0u);
        } else {
            NF_GLOBAL const ushort* w16 = (NF_GLOBAL const ushort*)weight;
            uint h[8];
            for (int e = 0; e < 8; e++) {
                h[e] = kk + e < K ? (uint)w16[at + e] : 0u;
            }
            wr[i] = make_uint4(h[0] | (h[1] << 16), h[2] | (h[3] << 16), h[4] | (h[5] << 16), h[6] | (h[7] << 16));
        }
    }
}

/*
 * The work-item's share of one activation stage, as 16-bit operands: runs lid and lid + 256 of the tile's
 * TN x (TK / 4) runs of four, run c holding depths [k + 4 * (c & 7), +4) of tile token c >> 3.
 */
NF_INLINE void nf_mmal_load_x(uint2 xr[2], NF_GLOBAL const float* NF_RESTRICT x, int K, int N, int x_stride,
                              int x_off, int n0, int k, int lid, int xvec, const int bf16) {
    for (int i = 0; i < 2; i++) {
        const int idx = lid + NF_MMAL_WG * i;
        const int t = n0 + (idx >> 3);
        const int kk = k + 4 * (idx & 7);
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (t < N) {
            const nf_i64 at = (nf_i64)x_off + (nf_i64)t * x_stride + kk;
            if (xvec && kk + 4 <= K) {
                const float4 f = *(NF_GLOBAL const float4*)(x + at);
                v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
            } else {
                for (int e = 0; e < 4; e++) {
                    v[e] = kk + e < K ? x[at + e] : 0.0f;
                }
            }
        }
        xr[i] = make_uint2((uint)nf_mma_x_f(v[0], bf16) | ((uint)nf_mma_x_f(v[1], bf16) << 16),
                           (uint)nf_mma_x_f(v[2], bf16) | ((uint)nf_mma_x_f(v[3], bf16) << 16));
    }
}

/* The registers of one stage into shared stage tiles a_s (TM rows) and b_s (TN tokens). */
NF_INLINE void nf_mmal_store(NF_LOCAL_PTR ushort* a_s, NF_LOCAL_PTR ushort* b_s, const uint4 wr[2],
                             const uint2 xr[2], int lid) {
    for (int i = 0; i < 2; i++) {
        const int idx = lid + NF_MMAL_WG * i;
        *(NF_LOCAL_PTR uint4*)(a_s + (idx >> 2) * NF_MMAL_S + 8 * (idx & 3)) = wr[i];
        *(NF_LOCAL_PTR uint2*)(b_s + (idx >> 3) * NF_MMAL_S + 4 * (idx & 7)) = xr[i];
    }
}

/*
 * One stage of the warp's 32 x 32 block: per 16-deep chunk, the A fragments of rows wm * 32 + mf * 16
 * (ldmatrix.x4 of rows lane & 15, depths 8 * (lane >> 4)) and the B fragments of tokens wn * 32 + np * 16
 * (ldmatrix.x4 of tokens (lane & 7) + 8 * (lane >> 4), depths 8 * ((lane >> 3) & 1): b0 and b1 of the first
 * eight tokens, then of the next eight), then the eight mma tiles.
 */
NF_INLINE void nf_mmal_stage(NF_LOCAL_PTR const ushort* a_s, NF_LOCAL_PTR const ushort* b_s, int wm, int wn,
                             int lane, float d[2][4][4], const int bf16) {
    for (int kc = 0; kc < NF_MMAL_TK; kc += 16) {
        uint a[2][4];
        for (int mf = 0; mf < 2; mf++) {
            nf_mma_ldmatrix_x4(a[mf], a_s + (wm * 32 + mf * 16 + (lane & 15)) * NF_MMAL_S + kc + 8 * (lane >> 4));
        }
        uint b[4][2];
        for (int np = 0; np < 2; np++) {
            uint q[4];
            nf_mma_ldmatrix_x4(q, b_s + (wn * 32 + np * 16 + (lane & 7) + 8 * (lane >> 4)) * NF_MMAL_S
                                          + kc + 8 * ((lane >> 3) & 1));
            b[2 * np][0] = q[0];
            b[2 * np][1] = q[1];
            b[2 * np + 1][0] = q[2];
            b[2 * np + 1][1] = q[3];
        }
        for (int mf = 0; mf < 2; mf++) {
            for (int nf = 0; nf < 4; nf++) {
                if (bf16) {
                    nf_mma_m16n8k16_bf16(a[mf], b[nf], d[mf][nf]);
                } else {
                    nf_mma_m16n8k16(a[mf], b[nf], d[mf][nf]);
                }
            }
        }
    }
}

/* The warp's accumulators to y, bounds-checked per element (the C fragment layout of m16n8). */
NF_INLINE void nf_mmal_store_y(NF_GLOBAL float* NF_RESTRICT y, int M, int N, int y_stride, int y_off, int m0,
                               int n0, int wm, int wn, int lane, const float d[2][4][4]) {
    for (int mf = 0; mf < 2; mf++) {
        const int r = m0 + wm * 32 + mf * 16 + (lane >> 2);
        for (int nf = 0; nf < 4; nf++) {
            const int c = n0 + wn * 32 + nf * 8 + (lane & 3) * 2;
            for (int e = 0; e < 4; e++) {
                const int rr = r + 8 * (e >> 1), cc = c + (e & 1);
                if (rr < M && cc < N) y[(nf_i64)y_off + rr + (nf_i64)cc * y_stride] = d[mf][nf][e];
            }
        }
    }
}

#define NF_MMA_GEMM_LARGE(T, BF16V) \
NF_KERNEL __launch_bounds__(NF_MMAL_WG, 3) void T##_MatMul_MMA_L( \
    const int M, const int K, const int N, \
    const int x_stride, const int x_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT x, \
    NF_GLOBAL float* NF_RESTRICT y) { \
    NF_LOCAL __align__(16) ushort a_s[2][NF_MMAL_TM * NF_MMAL_S]; \
    NF_LOCAL __align__(16) ushort b_s[2][NF_MMAL_TN * NF_MMAL_S]; \
    const int lid = get_local_id(0), lane = lid & 31, warp = lid >> 5; \
    const int wm = warp & 3, wn = warp >> 2; \
    const int token_tiles = (int)get_num_groups(1); \
    const int tile = (int)get_group_id(0) + (int)get_group_id(1) * (int)get_num_groups(0); \
    const int m0 = (tile / token_tiles) * NF_MMAL_TM; \
    const int n0 = (tile % token_tiles) * NF_MMAL_TN; \
    const int wvec = (K & 7) == 0 && (((size_t)weight) & 15) == 0; \
    const int xvec = ((x_stride | x_off) & 3) == 0 && (((size_t)x) & 15) == 0; \
    float d[2][4][4]; \
    for (int mf = 0; mf < 2; mf++) \
        for (int nf = 0; nf < 4; nf++) \
            for (int e = 0; e < 4; e++) d[mf][nf][e] = 0.0f; \
    uint4 wr[2]; \
    uint2 xr[2]; \
    nf_mmal_load_w(wr, weight, M, K, m0, 0, lid, wvec); \
    nf_mmal_load_x(xr, x, K, N, x_stride, x_off, n0, 0, lid, xvec, BF16V); \
    for (int k = 0, s = 0; k < K; k += NF_MMAL_TK, s ^= 1) { \
        nf_mmal_store(a_s[s], b_s[s], wr, xr, lid); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (k + NF_MMAL_TK < K) { \
            nf_mmal_load_w(wr, weight, M, K, m0, k + NF_MMAL_TK, lid, wvec); \
            nf_mmal_load_x(xr, x, K, N, x_stride, x_off, n0, k + NF_MMAL_TK, lid, xvec, BF16V); \
        } \
        nf_mmal_stage(a_s[s], b_s[s], wm, wn, lane, d, BF16V); \
    } \
    nf_mmal_store_y(y, M, N, y_stride, y_off, m0, n0, wm, wn, lane, d); \
}

NF_MMA_GEMM_LARGE(F16, 0)
NF_MMA_GEMM_LARGE(BF16, 1)

#endif /* NF_CUDA_PREPROCESS, or mma.sync.m16n8k16 on compute capability 8.0+ */
#endif /* __CUDACC__ and not __HIP__: the OpenCL program, the HIP build and the Vulkan shader set define no large tile */
