/* gemm.cl - strided batched GEMM and GEMV. Requires: common/dialect.h, common/train.h */
/*
 * Offsets and strides are in float elements and every stride is non-negative (0 broadcasts). The CPU
 * reference is tensor/TensorOps.matmul/bmm (and GpuMatmul's documented gemm reference): each output is a
 * float sum over k in ascending order from 0, adding separately rounded products. Here the sum runs in
 * ascending k through explicit fma() (one rounding per term), so results agree within the GEMM tolerance
 * class that GpuMatmul documents, not bit for bit.
 *
 * TR_Gemm_F32(i M, i N, i K, i batch,
 *             i a_off, i a_rs, i a_cs, i a_bs,
 *             i b_off, i b_rs, i b_cs, i b_bs,
 *             i c_off, i c_rs, i c_bs,
 *             f alpha, f beta,
 *             R A, R B, RW C)
 *     For b < batch, m < M, n < N, with
 *         A(b,m,k) = A[a_off + b*a_bs + m*a_rs + k*a_cs]
 *         B(b,k,n) = B[b_off + b*b_bs + k*b_rs + n*b_cs]
 *         C(b,m,n) = C[c_off + b*c_bs + m*c_rs + n]           (C's column stride is 1)
 *         s = sum_{k < K, ascending} A(b,m,k) * B(b,k,n)      (s = fma(a, b, s), from s = +0)
 *         C(b,m,n) = alpha * s                                 when beta == 0 (C is not read, so NaN or
 *                                                              garbage in C is ignored)
 *         C(b,m,n) = alpha * s + beta * C(b,m,n)               otherwise (two products, one sum, each
 *                                                              rounded; no contraction)
 *     A and B may be the same buffer; C aliases neither, and distinct (b,m,n) map to distinct C elements.
 *     Dispatch (the only multi-dimensional train launch): global = {ceil(N/64)*16, ceil(M/64)*16, batch},
 *     local = {16, 16, 1}. Group (gx, gy, gz) computes the 64 x 64 tile of rows 64*gy.., columns 64*gx..
 *     of batch gz; item (tx, ty) owns rows 64*gy + ty + 16*i and columns 64*gx + tx + 16*j (i, j < 4),
 *     so a warp's stores are contiguous along n. The host keeps every group count within the device
 *     limits (it chunks the batch and adjusts the offsets).
 *     Tiling: K is consumed in 16-deep stages. Per stage the group stages the 64 x 16 A slice and the
 *     16 x 64 B slice in local memory (4 elements per item each, zero outside M, N and K), then each item
 *     applies the stage's products to its 4 x 4 accumulators in ascending k; only the K - k0 valid depths
 *     of the last stage are applied, so every output sums exactly its K products. Loads walk the operand's
 *     unit-stride dimension across neighbouring items when it has one (k for A unless only m is unit
 *     stride; n for B unless only k is unit stride); the choice is uniform per launch and does not affect
 *     the result.
 *
 * TR_Gemv_F32(i M, i K, i a_off, i a_rs, i a_cs, i x_off, i x_s, i y_off, f alpha, f beta,
 *             R A, R x, RW y)
 *     For m < M:
 *         s = sum_{k < K} A[a_off + m*a_rs + k*a_cs] * x[x_off + k*x_s]
 *         y[y_off + m] = alpha * s                              when beta == 0 (y is not read)
 *         y[y_off + m] = alpha * s + beta * y[y_off + m]        otherwise
 *     Fixed order: item l of the row's group accumulates k = l, l + 256, ... ascending with fma() from +0,
 *     then tr_wg_sum combines the 256 partials over its fixed tree. A and x may be the same buffer; y
 *     aliases neither.
 *     Dispatch: 1-D "rows" launch, M groups of TR_LOCAL (global = M * 256); group g handles row g.
 */

#define TR_GM_TILE  64                  /* rows and columns of one workgroup tile */
#define TR_GM_TK    16                  /* reduction depth of one local-memory stage */
#define TR_GM_LX    16                  /* items per tile dimension (local size {16, 16, 1}) */
#define TR_GM_RB    4                   /* outputs per item in each tile dimension (64 / 16) */
#define TR_GM_PAD   (TR_GM_TILE + 1)    /* padded stride of one depth row of a staged slice */

NF_KERNEL void TR_Gemm_F32(
    const int M,
    const int N,
    const int K,
    const int batch,
    const int a_off,
    const int a_rs,
    const int a_cs,
    const int a_bs,
    const int b_off,
    const int b_rs,
    const int b_cs,
    const int b_bs,
    const int c_off,
    const int c_rs,
    const int c_bs,
    const float alpha,
    const float beta,
    NF_GLOBAL const float* NF_RESTRICT A,
    NF_GLOBAL const float* NF_RESTRICT B,
    NF_GLOBAL float* NF_RESTRICT C)
{
    NF_LOCAL float a_tile[TR_GM_TK * TR_GM_PAD];    /* a_tile[kk * PAD + mm] = A(b, m0 + mm, k0 + kk) */
    NF_LOCAL float b_tile[TR_GM_TK * TR_GM_PAD];    /* b_tile[kk * PAD + nn] = B(b, k0 + kk, n0 + nn) */
    const int bi = (int)get_group_id(2);
    if (bi >= batch) return;                        /* uniform per group */
    const int tx = (int)get_local_id(0);
    const int ty = (int)get_local_id(1);
    const int lid = ty * TR_GM_LX + tx;
    const int n0 = (int)get_group_id(0) * TR_GM_TILE;
    const int m0 = (int)get_group_id(1) * TR_GM_TILE;
    const int a_base = a_off + bi * a_bs;
    const int b_base = b_off + bi * b_bs;
    const int a_m_fast = a_cs != 1 && a_rs == 1;
    const int b_k_fast = b_cs != 1 && b_rs == 1;
    const int stages = K > 0 ? (K - 1) / TR_GM_TK + 1 : 0;

    float acc[TR_GM_RB][TR_GM_RB];
    for (int i = 0; i < TR_GM_RB; ++i) {
        for (int j = 0; j < TR_GM_RB; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int st = 0; st < stages; ++st) {
        const int k0 = st * TR_GM_TK;
        for (int q = 0; q < TR_GM_RB; ++q) {
            const int e = lid + q * (TR_GM_LX * TR_GM_LX);
            const int a_mm = a_m_fast ? (e % TR_GM_TILE) : (e / TR_GM_TK);
            const int a_kk = a_m_fast ? (e / TR_GM_TILE) : (e % TR_GM_TK);
            const int am = m0 + a_mm;
            const int ak = k0 + a_kk;
            float av = 0.0f;
            if (am < M && ak < K) av = A[a_base + am * a_rs + ak * a_cs];
            a_tile[a_kk * TR_GM_PAD + a_mm] = av;

            const int b_nn = b_k_fast ? (e / TR_GM_TK) : (e % TR_GM_TILE);
            const int b_kk = b_k_fast ? (e % TR_GM_TK) : (e / TR_GM_TILE);
            const int bn = n0 + b_nn;
            const int bk = k0 + b_kk;
            float bv = 0.0f;
            if (bn < N && bk < K) bv = B[b_base + bk * b_rs + bn * b_cs];
            b_tile[b_kk * TR_GM_PAD + b_nn] = bv;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int depth = (K - k0) < TR_GM_TK ? (K - k0) : TR_GM_TK;
        for (int kk = 0; kk < depth; ++kk) {
            float a[TR_GM_RB];
            float b[TR_GM_RB];
            for (int i = 0; i < TR_GM_RB; ++i) {
                a[i] = a_tile[kk * TR_GM_PAD + ty + TR_GM_LX * i];
            }
            for (int j = 0; j < TR_GM_RB; ++j) {
                b[j] = b_tile[kk * TR_GM_PAD + tx + TR_GM_LX * j];
            }
            for (int i = 0; i < TR_GM_RB; ++i) {
                for (int j = 0; j < TR_GM_RB; ++j) {
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int c_base = c_off + bi * c_bs;
    for (int i = 0; i < TR_GM_RB; ++i) {
        const int m = m0 + ty + TR_GM_LX * i;
        if (m >= M) continue;
        for (int j = 0; j < TR_GM_RB; ++j) {
            const int n = n0 + tx + TR_GM_LX * j;
            if (n >= N) continue;
            const int idx = c_base + m * c_rs + n;
            float r = alpha * acc[i][j];
            if (beta != 0.0f) {
                const float bc = beta * C[idx];
                r = r + bc;
            }
            C[idx] = r;
        }
    }
}

NF_KERNEL void TR_Gemv_F32(
    const int M,
    const int K,
    const int a_off,
    const int a_rs,
    const int a_cs,
    const int x_off,
    const int x_s,
    const int y_off,
    const float alpha,
    const float beta,
    NF_GLOBAL const float* NF_RESTRICT A,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL float* NF_RESTRICT y)
{
    NF_LOCAL float red[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= M) return;                           /* uniform per group */
    const int lid = (int)get_local_id(0);
    const int a_row = a_off + row * a_rs;
    float s = 0.0f;
    /* uint counter: k + TR_LOCAL cannot overflow for any K <= 2^31-1. */
    for (uint k = (uint)lid; k < (uint)K; k += (uint)TR_LOCAL) {
        const int ki = (int)k;
        s = fma(A[a_row + ki * a_cs], x[x_off + ki * x_s], s);
    }
    s = tr_wg_sum(red, s);
    if (lid == 0) {
        const int idx = y_off + row;
        float r = alpha * s;
        if (beta != 0.0f) {
            const float by = beta * y[idx];
            r = r + by;
        }
        y[idx] = r;
    }
}
