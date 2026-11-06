/* common/matmul_mma_family.h - {T}_MatMul_MMA macro family: tensor-core GEMM for prefill and batched
 * decode over the dense 16-bit weight types F16 and BF16 (CUDA mma/ldmatrix, scalar twin elsewhere).
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h (bf16_to_f32) */
/*
 * NF_MMA_GEMM(T, BF16) defines the kernel T##_MatMul_MMA for a dense weight T of 16-bit values:
 *
 *   T_MatMul_MMA(i M, i K, i N, i x_stride, i x_off, i y_stride, i y_off,
 *                R uchar weight, R float x, W float y)
 *   y[y_off + t*y_stride + r] = sum_{k < K} W[r][k] * x[x_off + t*x_stride + k],  r < M, t < N
 *
 * Same contract as T##_MatMul_Tiled (common/matmul_tiled_family.h): ggml_mul_mat(W, X) with X a batch
 * of N activation rows. The row-major [M x K] weight holds K IEEE-half (F16) or bfloat16 (BF16)
 * elements per row, BF16 selecting the bfloat16 mma instruction and the bfloat16 tile conversion.
 *
 * Compute paths inside one kernel, selected at compile time:
 * - CUDA sm_80+ (NF_MMA_TENSOR): shared raw-bit tiles; every warp loads its fragment rows with
 *   ldmatrix (x4 for the two 16x16 A fragments, x2.trans for the k16 x n8 B fragments, matching the
 *   mma.m16n8k16.row.col fragment layouts) and accumulates with mma.sync f32=f16*f16+f32 (F16) or
 *   f32=bf16*bf16+f32 (BF16). Products are computed in 16-bit and accumulated in fp32, so results
 *   differ from the scalar paths by 16-bit rounding, as in llama.cpp ggml-cuda/mma.cuh.
 * - CUDA below sm_80: the same raw-bit tiles, per-element fma over converted floats.
 * - OpenCL (OpenCL C 1.2 has no tensor-core instruction set): float tiles, per-element fma, exact
 *   fp32 arithmetic. The dispatch (gguf.gpu.forward.GpuOps) launches this kernel only on CUDA sm_80+
 *   and on Vulkan devices with cooperative-matrix support, so this path is the fallback twin of the
 *   same kernel name.
 *
 * NF_MMA_GEMM_KSPLIT(T, BF16) defines T##_MatMul_MMA_KSplit, the split-K twin over the same tiles:
 *
 *   T_MatMul_MMA_KSplit(i M, i K, i N, i x_stride, i x_off, i k_slice,
 *                       R uchar weight, R float x, W float partial)
 *   partial[(slice * N + t) * M + r] = sum_{k in slice} W[r][k] * x[x_off + t*x_stride + k]
 *
 * with the slice index taken from the work-group id ({ceil(M / 64) * split groups along dimension 0,
 * slice = gid0 / ceil(M / 64)}), k in [slice * k_slice, min((slice + 1) * k_slice, K)) and every
 * live element of its [M x N] slice plane written (row stride M), so the host reduce
 * (LLM_SplitK_Reduce, kernels/llm/split_k_reduce.cl) sums the planes into the strided output. The
 * launch geometry and the shared-memory footprint are the plain kernel's with dimension 0 scaled by
 * the split count (gguf.ops.gemm.SplitKGemmPlan).
 *
 * Tiling: a workgroup of NF_MMA_WG = 256 work-items (8 warps of 32) computes NF_MMA_TM = 64 weight
 * rows x NF_MMA_TN = 32 tokens in NF_MMA_TK = 16-deep stages. Warp w covers rows wy * 32 and tokens
 * wx * 8, wy = w >> 2, wx = w & 3 (two 16x8 mma tiles per stage). Work-item lid loads weight rows
 * tx + 16*i (i < 4) at depth ty and activation tokens ty + 16*j (j < 2) at depth tx, tx = lid % 16,
 * ty = lid / 16. Tile rows use the padded strides NF_MMA_AS = 24 and NF_MMA_BS = 40 halfs, so every
 * ldmatrix row address is 16-byte aligned and the 8 rows of a matrix sit on distinct 128-byte phases.
 * Rows past M, tokens past N and depths past K load 0, so every work-item and warp reaches every
 * barrier; fragment stores are bounds-checked per lane.
 *
 * Launch contract: 2-D, global = {ceil(M / 64) * 256, ceil(N / 32)}, local = {256, 1}; the host plans
 * this in gguf.ops.gemm.MmaGemmPlan. Local memory: (64 * 24 + 16 * 40) tile elements, raw 16-bit
 * values on CUDA and floats elsewhere.
 */
#ifndef NF_MATMUL_MMA_FAMILY_H
#define NF_MATMUL_MMA_FAMILY_H

#define NF_MMA_TM 64                     /* weight rows per workgroup tile */
#define NF_MMA_TN 32                     /* tokens per workgroup tile */
#define NF_MMA_TK 16                     /* reduction depth of one stage */
#define NF_MMA_WG 256                    /* work-items per workgroup */
#define NF_MMA_RM 4                      /* rows per work-item (scalar paths) */
#define NF_MMA_RN 2                      /* tokens per work-item (scalar paths) */
#define NF_MMA_TX (NF_MMA_TM / NF_MMA_RM)      /* 16 row lanes */
#define NF_MMA_AS 24                     /* padded half stride of a weight-tile row */
#define NF_MMA_BS 40                     /* padded half stride of an activation-tile row */

#if defined(__CUDACC__)
#define NF_MMA_ATYPE NF_HALF             /* tile element: raw fp16 or bf16 bits */
#define NF_MMA_TENSOR (__CUDA_ARCH__ >= 800)
/* Raw bits of weight element idx (fp16 or bf16 bits; the mma instruction consumes the bits). */
#define NF_MMA_W_BITS(w, idx, bf16) (((NF_GLOBAL const NF_HALF*)(w))[idx])
/* A float converted to the tile's 16-bit storage bits (round to nearest even for both formats). */
NF_INLINE NF_MMA_ATYPE nf_mma_x_f(float x, int bf16) {
    return bf16 ? (NF_MMA_ATYPE)__bfloat16_as_ushort(__float2bfloat16_rn(x))
                : __half_as_ushort(__float2half_rn(x));
}
/* A tile element's bits as a float (the scalar paths). */
NF_INLINE float nf_mma_a_f(NF_MMA_ATYPE v, int bf16) {
    return bf16 ? __bfloat162float(__ushort_as_bfloat16((ushort)v))
                : __half2float(__ushort_as_half((ushort)v));
}
#else
#define NF_MMA_ATYPE float
#define NF_MMA_TENSOR 0
#define NF_MMA_W_BITS(w, idx, bf16) ((bf16) ? bf16_to_f32(((NF_GLOBAL const ushort*)(w))[idx]) \
                                            : vload_half((size_t)(idx), (NF_GLOBAL const NF_HALF*)(w)))
NF_INLINE float nf_mma_x_f(float x, int bf16) { return x; }
NF_INLINE float nf_mma_a_f(float v, int bf16) { return v; }
#endif

/*
 * NF_MMA_FRAGMENTS: the fragment helpers below exist. True on CUDA sm_80+ device passes and on the
 * passes that only parse the code (the host pass and the build's NF_CUDA_PREPROCESS inventory pass),
 * so the CUDA-only tensor-core kernels (kernels/llm/moe_gemm_mma.cl), guarded the same way, can call
 * them; NF_MMA_TENSOR selects the tensor path of nf_mma_stage on the device passes only. The PTX
 * fragment instructions have no HIP form, so every HIP pass compiles none of them and the kernels
 * run on the scalar paths (NF_MMA_TENSOR is 0, as __CUDA_ARCH__ is undefined on HIP).
 */
#if defined(__CUDACC__) && !defined(__HIP__) \
    && (defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define NF_MMA_FRAGMENTS 1
#else
#define NF_MMA_FRAGMENTS 0
#endif

#if NF_MMA_FRAGMENTS
/* ldmatrix of four 8x8 fp16 matrices; lanes 0..31 supply the matrix rows. */
NF_INLINE void nf_mma_ldmatrix_x4(uint r[4], NF_LOCAL_PTR const void* p) {
    const uint addr = (uint)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

/* ldmatrix.trans of two 8x8 fp16 matrices: the fragments of a row-major k16 x n8 tile as the
 * column-major B operand of mma. */
NF_INLINE void nf_mma_ldmatrix_x2_trans(uint r[2], NF_LOCAL_PTR const void* p) {
    const uint addr = (uint)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

/* d = a * b^T + d over one 16x8x16 fp16 tile (row-major A fragment, column-major B fragment). */
NF_INLINE void nf_mma_m16n8k16(const uint a[4], const uint b[2], float d[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* d = a * b^T + d over one 16x8x16 bfloat16 tile (same fragment layouts as the fp16 form). */
NF_INLINE void nf_mma_m16n8k16_bf16(const uint a[4], const uint b[2], float d[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/*
 * The tensor path of one stage: the warp (lid >> 5) multiplies its two 16x16 A fragments (rows
 * wy*32 + mf*16 of the 64 x NF_MMA_AS tile a_t) with its k16 x n8 B fragment (tokens wx*8 of the
 * 16 x NF_MMA_BS tile b_t) into d[mf*4 ..]; the tiles hold fp16 bits, or bfloat16 bits when bf16.
 */
NF_INLINE void nf_mma_tensor_stage(NF_LOCAL_PTR const NF_MMA_ATYPE* a_t, NF_LOCAL_PTR const NF_MMA_ATYPE* b_t,
                                   int lid, float d[8], const int bf16) {
    const int warp = lid >> 5, wy = warp >> 2, wx = warp & 3, lane = lid & 31;
    uint bfrag[2];
    nf_mma_ldmatrix_x2_trans(bfrag, b_t + (lane & 15) * NF_MMA_BS + wx * 8);
    for (int mf = 0; mf < 2; mf++) {
        const int mb = wy * 32 + mf * 16;
        uint afrag[4];
        const int r = mb + 8 * ((lane >> 3) & 1) + (lane & 7);
        nf_mma_ldmatrix_x4(afrag, a_t + r * NF_MMA_AS + 8 * (lane >> 4));
        if (bf16) {
            nf_mma_m16n8k16_bf16(afrag, bfrag, d + mf * 4);
        } else {
            nf_mma_m16n8k16(afrag, bfrag, d + mf * 4);
        }
    }
}
#endif /* NF_MMA_FRAGMENTS */

/*
 * One stage of a_t x b_t. Tensor path: the warp (lid >> 5) multiplies its two 16x16 A fragments
 * (rows wy*32 + mf*16 of the tile) with its k16 x n8 B fragment (tokens wx*8) into d[mf*4 ..].
 * Scalar paths: the work-item accumulates rows tx + 16*q, tokens ty + 16*j into acc[q*2 + j].
 */
NF_INLINE void nf_mma_stage(NF_LOCAL_PTR const NF_MMA_ATYPE* a_t, NF_LOCAL_PTR const NF_MMA_ATYPE* b_t,
                            int lid, float acc[NF_MMA_RM * NF_MMA_RN], float d[8], const int bf16) {
#if NF_MMA_TENSOR
    (void)acc;
    nf_mma_tensor_stage(a_t, b_t, lid, d, bf16);
#else
    const int tx = lid % NF_MMA_TX, ty = lid / NF_MMA_TX;
    (void)d;
    for (int kk = 0; kk < NF_MMA_TK; kk++) {
        for (int q = 0; q < NF_MMA_RM; q++) {
            const float a = nf_mma_a_f(a_t[(tx + NF_MMA_TX * q) * NF_MMA_AS + kk], bf16);
            for (int j = 0; j < NF_MMA_RN; j++) {
                acc[q * NF_MMA_RN + j] += a * nf_mma_a_f(b_t[kk * NF_MMA_BS + ty + NF_MMA_TX * j], bf16);
            }
        }
    }
#endif
}

/*
 * The stage results of one work-item (or warp) to global memory, bounds-checked per element: the
 * plain kernel writes only the tile's live rows and tokens.
 */
NF_INLINE void nf_mma_store(NF_GLOBAL float* NF_RESTRICT y, int M, int N,
                            int y_stride, int y_off, int m0, int n0, int lid,
                            const float acc[NF_MMA_RM * NF_MMA_RN], const float d[8]) {
#if NF_MMA_TENSOR
    const int warp = lid >> 5, wy = warp >> 2, wx = warp & 3, lane = lid & 31;
    (void)acc;
    for (int mf = 0; mf < 2; mf++) {
        const int r = m0 + wy * 32 + mf * 16 + (lane >> 2);
        const int c = n0 + wx * 8 + (lane & 3) * 2;
        if (r < M && c < N) y[(nf_i64)y_off + r + (nf_i64)c * y_stride] = d[mf * 4];
        if (r < M && c + 1 < N) y[(nf_i64)y_off + r + (nf_i64)(c + 1) * y_stride] = d[mf * 4 + 1];
        if (r + 8 < M && c < N) y[(nf_i64)y_off + r + 8 + (nf_i64)c * y_stride] = d[mf * 4 + 2];
        if (r + 8 < M && c + 1 < N) y[(nf_i64)y_off + r + 8 + (nf_i64)(c + 1) * y_stride] = d[mf * 4 + 3];
    }
#else
    const int tx = lid % NF_MMA_TX, ty = lid / NF_MMA_TX;
    (void)d;
    for (int j = 0; j < NF_MMA_RN; j++) {
        const int t = n0 + ty + NF_MMA_TX * j;
        if (t >= N) continue;
        const nf_i64 ybase = (nf_i64)y_off + (nf_i64)t * y_stride;
        for (int q = 0; q < NF_MMA_RM; q++) {
            const int r = m0 + tx + NF_MMA_TX * q;
            if (r < M) y[ybase + r] = acc[q * NF_MMA_RN + j];
        }
    }
#endif
}

/* ---- kernel generator ---- */

/*
 * Shared body of the plain and split-K kernels: stages the [k0, k1) k range of the tiles and
 * accumulates it. The staging and the compute path are the plain kernel's; the caller owns the
 * accumulators and the store. Uses the enclosing kernel's a_t, b_t, aok, bok, xrow, acc, d, lid,
 * m0, n0, tx and ty.
 */
#define NF_MMA_STAGE_RANGE(weight, x, K, k0, k1, bf16v) \
    for (int k = k0; k < k1; k += NF_MMA_TK) { \
        const int kq = k + ty; \
        for (int q = 0; q < NF_MMA_RM; q++) { \
            const int r = m0 + tx + NF_MMA_TX * q; \
            a_t[(tx + NF_MMA_TX * q) * NF_MMA_AS + ty] = \
                (aok[q] && kq < k1) ? NF_MMA_W_BITS(weight, (nf_i64)r * K + kq, bf16v) \
                                    : nf_mma_x_f(0.0f, bf16v); \
        } \
        const int kp = k + tx; \
        for (int j = 0; j < NF_MMA_RN; j++) { \
            b_t[tx * NF_MMA_BS + ty + NF_MMA_TX * j] = \
                nf_mma_x_f((bok[j] && kp < k1) ? x[xrow[j] + kp] : 0.0f, bf16v); \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        nf_mma_stage(a_t, b_t, lid, acc, d, bf16v); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

#define NF_MMA_GEMM(T, BF16V) \
NF_KERNEL void T##_MatMul_MMA( \
    const int M, const int K, const int N, \
    const int x_stride, const int x_off, const int y_stride, const int y_off, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT x, \
    NF_GLOBAL float* NF_RESTRICT y) { \
    NF_LOCAL NF_MMA_ATYPE a_t[NF_MMA_TM * NF_MMA_AS]; \
    NF_LOCAL NF_MMA_ATYPE b_t[NF_MMA_TK * NF_MMA_BS]; \
    const int lid = get_local_id(0); \
    const int m0 = (int)get_group_id(0) * NF_MMA_TM; \
    const int n0 = (int)get_group_id(1) * NF_MMA_TN; \
    const int tx = lid % NF_MMA_TX, ty = lid / NF_MMA_TX; \
    int aok[NF_MMA_RM]; \
    for (int q = 0; q < NF_MMA_RM; q++) { \
        aok[q] = m0 + tx + NF_MMA_TX * q < M; \
    } \
    int bok[NF_MMA_RN]; \
    nf_i64 xrow[NF_MMA_RN]; \
    for (int j = 0; j < NF_MMA_RN; j++) { \
        const int t = n0 + ty + NF_MMA_TX * j; \
        bok[j] = t < N; \
        xrow[j] = (nf_i64)x_off + (bok[j] ? (nf_i64)t * x_stride : (nf_i64)0); \
    } \
    float acc[NF_MMA_RM * NF_MMA_RN]; \
    float d[8]; \
    for (int i = 0; i < NF_MMA_RM * NF_MMA_RN; i++) acc[i] = 0.0f; \
    for (int i = 0; i < 8; i++) d[i] = 0.0f; \
    NF_MMA_STAGE_RANGE(weight, x, K, 0, K, BF16V) \
    nf_mma_store(y, M, N, y_stride, y_off, m0, n0, lid, acc, d); \
}

/*
 * The split-K twin: dimension 0 of the dispatch holds ceil(M / 64) * split workgroups, group
 * gid0 = rt + rtiles * slice covers rows rt * 64 of slice `slice` over k in
 * [slice * k_slice, min((slice + 1) * k_slice, K)) and writes the live [M x N] positions of its
 * slice plane at partial[(slice * N + t) * M + r] (y_stride = M), so the host reduce
 * (LLM_SplitK_Reduce, kernels/llm/split_k_reduce.cl) sums the planes into the strided output. The
 * launch geometry and the shared-memory footprint are the plain kernel's with dimension 0 scaled by
 * the split count (gguf.ops.gemm.SplitKGemmPlan).
 */
#define NF_MMA_GEMM_KSPLIT(T, BF16V) \
NF_KERNEL void T##_MatMul_MMA_KSplit( \
    const int M, const int K, const int N, \
    const int x_stride, const int x_off, const int k_slice, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT x, \
    NF_GLOBAL float* NF_RESTRICT partial) { \
    NF_LOCAL NF_MMA_ATYPE a_t[NF_MMA_TM * NF_MMA_AS]; \
    NF_LOCAL NF_MMA_ATYPE b_t[NF_MMA_TK * NF_MMA_BS]; \
    const int lid = get_local_id(0); \
    const int rtiles = (M + NF_MMA_TM - 1) / NF_MMA_TM; \
    const int slice = (int)get_group_id(0) / rtiles; \
    const int m0 = ((int)get_group_id(0) - slice * rtiles) * NF_MMA_TM; \
    const int n0 = (int)get_group_id(1) * NF_MMA_TN; \
    const int k0 = slice * k_slice; \
    const int k1 = k0 + k_slice < K ? k0 + k_slice : K; \
    const int tx = lid % NF_MMA_TX, ty = lid / NF_MMA_TX; \
    int aok[NF_MMA_RM]; \
    for (int q = 0; q < NF_MMA_RM; q++) { \
        aok[q] = m0 + tx + NF_MMA_TX * q < M; \
    } \
    int bok[NF_MMA_RN]; \
    nf_i64 xrow[NF_MMA_RN]; \
    for (int j = 0; j < NF_MMA_RN; j++) { \
        const int t = n0 + ty + NF_MMA_TX * j; \
        bok[j] = t < N; \
        xrow[j] = (nf_i64)x_off + (bok[j] ? (nf_i64)t * x_stride : (nf_i64)0); \
    } \
    float acc[NF_MMA_RM * NF_MMA_RN]; \
    float d[8]; \
    for (int i = 0; i < NF_MMA_RM * NF_MMA_RN; i++) acc[i] = 0.0f; \
    for (int i = 0; i < 8; i++) d[i] = 0.0f; \
    if (k0 < k1) { \
        NF_MMA_STAGE_RANGE(weight, x, K, k0, k1, BF16V) \
    } \
    nf_mma_store(partial, M, N, M, (int)((nf_i64)slice * M * N), m0, n0, lid, acc, d); \
}

#endif /* NF_MATMUL_MMA_FAMILY_H */
