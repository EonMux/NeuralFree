/* common/matmul_tiled_family.h - {T}_MatMul_Tiled macro family for prefill GEMM. Requires: common/dialect.h, common/config.h, common/quant_blocks.h */
/*
 * NF_GEMM_TILED(T, ELEM, BS, BB) defines the kernel T##_MatMul_Tiled for weight type T with BS
 * elements and BB bytes per block (BS = 1 and BB = element bytes for dense types):
 *
 *   T_MatMul_Tiled(i M, i K, i N, i x_stride, i x_off, i y_stride, i y_off,
 *                  R uchar weight, R float x, W float y)
 *   y[y_off + t*y_stride + r] = sum_{k < K} W[r][k] * x[x_off + t*x_stride + k],  r < M, t < N
 *
 * This is ggml_mul_mat(W, X) (llama.cpp ggml/src/ggml.c:3364-3371, dst ne = {W->ne[1], X->ne[1]}) with W
 * a [M x K] weight (rows of K / BS blocks of BB bytes) and X a batch of N activation rows; output
 * row t holds the M dot products of token t, as ggml dst (ne0 = M contiguous). Activation and
 * output rows may be views (x_off / y_off in elements); x and y are distinct buffers.
 *
 * ELEM is the element loader of the type, with the decode contract of common/matvec_family.h:
 *   NF_INLINE float ELEM(NF_GLOBAL const uchar* weight, int dim, int tok, int i);
 * returning element i of row tok of a [rows x dim] weight. The family calls it with the start of
 * the weight row already formed in nf_i64 and tok = 0, so per-element offsets stay below one row.
 *
 * Tiling (llama.cpp ggml-vulkan/vulkan-shaders/mul_mm.comp:162-195 geometry: BM = 64, TM = 4, TN = 2,
 * padded shared strides): a workgroup of NF_GEMM_WG = 256 work-items computes a tile of
 * NF_GEMM_TM = 64 weight rows x NF_GEMM_TN = 32 tokens. The reduction runs in NF_GEMM_TK = 16 deep
 * stages: each stage dequantizes the 64 x 16 weight slice to local float (4 elements per
 * work-item) and copies the 32 x 16 activation slice (2 per work-item), then every work-item
 * accumulates its 4 rows x 2 tokens register block. Work-item lid owns rows tx + 16*i (i < 4) and
 * tokens ty + 16*j (j < 2) with tx = lid % 16, ty = lid / 16; it loads stage depth kk = lid % 16 of
 * rows (lid / 16) + 16*q (q < 4) and tokens (lid / 16) + 16*q (q < 2). Rows past M, tokens past N
 * and depths past K load 0, so every work-item reaches every barrier.
 *
 * Launch contract: 2-D, global = {ceil(M / 64) * 256, ceil(N / 32)}, local = {256, 1}; the host
 * (gguf.ops.gemm.TiledGemmPlan) guarantees K % BS == 0 for block-quantized types.
 * Local memory: NF_GEMM_TK * (NF_GEMM_TM + 1) + NF_GEMM_TK * (NF_GEMM_TN + 1) floats = 6272 bytes.
 */
#ifndef NF_MATMUL_TILED_FAMILY_H
#define NF_MATMUL_TILED_FAMILY_H

#define NF_GEMM_TM 64                     /* weight rows per workgroup tile */
#define NF_GEMM_TN 32                     /* tokens per workgroup tile */
#define NF_GEMM_TK 16                     /* reduction depth per local-memory stage */
#define NF_GEMM_WG 256                    /* work-items per workgroup */
#define NF_GEMM_RM 4                      /* rows per work-item (register block) */
#define NF_GEMM_RN 2                      /* tokens per work-item (register block) */
#define NF_GEMM_TX (NF_GEMM_TM / NF_GEMM_RM)   /* 16 row lanes */
#define NF_GEMM_AS (NF_GEMM_TM + 1)       /* padded stride of one depth row of the weight tile */
#define NF_GEMM_BS (NF_GEMM_TN + 1)       /* padded stride of one depth row of the activation tile */

/* ---- dense element loaders (ELEM contract) ---- */

NF_INLINE float nf_gemm_f32_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    return ((NF_GLOBAL const float*)weight)[(nf_i64)tok * dim + i];
}

NF_INLINE float nf_gemm_f16_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    return vload_half((size_t)((nf_i64)tok * dim + i), (NF_GLOBAL const NF_HALF*)weight);
}

NF_INLINE float nf_gemm_bf16_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    return bf16_to_f32(((NF_GLOBAL const ushort*)weight)[(nf_i64)tok * dim + i]);
}

/* ---- shared stage compute ---- */

/*
 * acc[i][j] += sum_{kk < NF_GEMM_TK} a_tile[kk][tx + 16*i] * b_tile[kk][ty + 16*j], kk ascending, so
 * every output sums its products in ascending k.
 */
NF_INLINE void nf_gemm_stage_fma(NF_LOCAL_PTR const float* a_tile, NF_LOCAL_PTR const float* b_tile,
                                 int tx, int ty, float acc[NF_GEMM_RM][NF_GEMM_RN]) {
    for (int kk = 0; kk < NF_GEMM_TK; kk++) {
        float a[NF_GEMM_RM];
        float b[NF_GEMM_RN];
        for (int i = 0; i < NF_GEMM_RM; i++) {
            a[i] = a_tile[kk * NF_GEMM_AS + tx + NF_GEMM_TX * i];
        }
        for (int j = 0; j < NF_GEMM_RN; j++) {
            b[j] = b_tile[kk * NF_GEMM_BS + ty + NF_GEMM_TX * j];
        }
        for (int i = 0; i < NF_GEMM_RM; i++) {
            for (int j = 0; j < NF_GEMM_RN; j++) {
                acc[i][j] += a[i] * b[j];
            }
        }
    }
}

/* ---- kernel generator ---- */

#define NF_GEMM_TILED(T, ELEM, BS, BB) \
NF_KERNEL void T##_MatMul_Tiled( \
    const int M, const int K, const int N, \
    const int x_stride, const int x_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT x, \
    NF_GLOBAL float* NF_RESTRICT y) { \
    NF_LOCAL float a_tile[NF_GEMM_TK * NF_GEMM_AS]; \
    NF_LOCAL float b_tile[NF_GEMM_TK * NF_GEMM_BS]; \
    const int lid = get_local_id(0); \
    const int m0 = (int)get_group_id(0) * NF_GEMM_TM; \
    const int n0 = (int)get_group_id(1) * NF_GEMM_TN; \
    const int tx = lid % NF_GEMM_TX, ty = lid / NF_GEMM_TX; \
    const int lk = lid % NF_GEMM_TK, lr = lid / NF_GEMM_TK; \
    const nf_i64 row_bytes = (nf_i64)(K / (BS)) * (BB); \
    NF_GLOBAL const uchar* wrow[NF_GEMM_RM]; \
    int wok[NF_GEMM_RM]; \
    for (int q = 0; q < NF_GEMM_RM; q++) { \
        const int r = m0 + lr + NF_GEMM_TX * q; \
        wok[q] = r < M; \
        wrow[q] = weight + (wok[q] ? (nf_i64)r * row_bytes : (nf_i64)0); \
    } \
    nf_i64 xrow[NF_GEMM_RN]; \
    int xok[NF_GEMM_RN]; \
    for (int q = 0; q < NF_GEMM_RN; q++) { \
        const int t = n0 + lr + NF_GEMM_TX * q; \
        xok[q] = t < N; \
        xrow[q] = (nf_i64)x_off + (xok[q] ? (nf_i64)t * x_stride : (nf_i64)0); \
    } \
    float acc[NF_GEMM_RM][NF_GEMM_RN]; \
    for (int i = 0; i < NF_GEMM_RM; i++) \
        for (int j = 0; j < NF_GEMM_RN; j++) \
            acc[i][j] = 0.0f; \
    for (int k0 = 0; k0 < K; k0 += NF_GEMM_TK) { \
        const int k = k0 + lk; \
        const int kok = k < K; \
        for (int q = 0; q < NF_GEMM_RM; q++) \
            a_tile[lk * NF_GEMM_AS + lr + NF_GEMM_TX * q] = \
                (wok[q] && kok) ? ELEM(wrow[q], K, 0, k) : 0.0f; \
        for (int q = 0; q < NF_GEMM_RN; q++) \
            b_tile[lk * NF_GEMM_BS + lr + NF_GEMM_TX * q] = \
                (xok[q] && kok) ? x[xrow[q] + k] : 0.0f; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        nf_gemm_stage_fma(a_tile, b_tile, tx, ty, acc); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (int j = 0; j < NF_GEMM_RN; j++) { \
        const int t = n0 + ty + NF_GEMM_TX * j; \
        if (t >= N) continue; \
        const nf_i64 ybase = (nf_i64)y_off + (nf_i64)t * y_stride; \
        for (int i = 0; i < NF_GEMM_RM; i++) { \
            const int r = m0 + tx + NF_GEMM_TX * i; \
            if (r < M) y[ybase + r] = acc[i][j]; \
        } \
    } \
}

#endif /* NF_MATMUL_TILED_FAMILY_H */
