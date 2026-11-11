/* matmul_fp8.cl - the E4M3 lanes: the FP8 activation staging of the checkpoint schemes and the E4M3 GEMMs of
 * the FP8 weight types (<T>_MatMul_MMQ) and of the 4-bit weight types (<T>_MatMul_W4A8). CUDA only.
 * Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h, common/fp8_codec.h,
 * common/quant_decode_legacy.h, common/quant_decode_new.h, common/matmul_mmq_config.h, common/matmul_mmq_family.h */
/*
 * The E4M3 lane is the FP8 tensor-core multiply of a checkpoint's own FP8 activation codes with a weight in the
 * E4M3 code form, on CUDA compute capability 8.9 and above (mma.sync.aligned.m16n8k32...e4m3.e4m3.f32); below
 * sm_89 the same kernels evaluate the stage dots scalar. No other backend has FP8 tensor cores or a shader
 * float8 type, and the lane never runs there: the projections keep the exact tiled GEMM with the fake-quantized
 * input. Both the staging passes and the GEMM instantiations therefore live in this CUDA-only source, and their
 * kernels appear in the CUDA fatbins only.
 */

/* ---- E4M3 staging of the FP8, MX and 4-bit activation schemes ----
 *
 * LLM_Quant_MMQ_Fp8Static(i N, i K, i tk, i x_stride, i x_off, f scale, R float x, W uint q8)
 * LLM_Quant_MMQ_Fp8Row(i N, i K, i tk, i x_stride, i x_off, f ub, R float x, W uint q8)
 * LLM_Quant_MMQ_Fp8Group128(i N, i K, i tk, i x_stride, i x_off, R float x, W uint q8)
 * LLM_Quant_MMQ_Fp8Mx(i N, i K, i tk, i x_stride, i x_off, R float x, W uint q8)
 * LLM_Quant_MMQ_Fp4Mx(i N, i K, i tk, i x_stride, i x_off, R float x, W uint q8)
 * LLM_Quant_MMQ_Fp4Nv(i N, i K, i tk, i x_stride, i x_off, f global_scale, R float x, W uint q8)
 *
 * Stage an activation scheme into the record layout of LLM_Quant_MMQ_Q8_1 as E4M3 codes and one chunk
 * scale d, so that the E4M3 GEMMs (T_MatMul_MMQ over the FP8 weight types and T_MatMul_W4A8 over the
 * 4-bit ones) multiply the checkpoint's own activation codes. The code words hold the packed E4M3
 * bytes, the code sums are zero (the E4M3 weight types carry no offset) and the zero point is 0. Per
 * chunk (the MMQ stage depth, 32 or 16):
 *   Fp8Static    FP8_STATIC: d = scale, codes = e4m3(x * (1 / scale)).
 *   Fp8Row       FP8_TOKEN: over the whole token row, a = min(amax, ub),
 *                d = max(a / 448, 1 / (448 * 512)), codes = e4m3(x / d); ub is the kernel's argument,
 *                the scheme's amax bound (the checkpoint's scale_ub, +inf when absent).
 *   Fp8Group128  FP8_GROUP128: per 128-element run, d = max(amax, 1e-10f) / 448, codes = e4m3(x * (1 / d)).
 *   Fp8Mx        MXFP8: d = 2^e for the run's MXFP8 exponent e, codes = e4m3(x * 2^-e).
 *   Fp4Mx        MXFP4: d = 2^e for the run's MXFP4 exponent e, codes = e4m3(e2m1(x * 2^-e)): every E2M1
 *                value is exact in E4M3, so the staged value is the scheme's own.
 *   Fp4Nv        NVFP4: per 16-element run s16 = e4m3(amax / 6 / global_scale) and the scheme's value
 *                v = e2m1(x / (s16 * global_scale)) * s16 * global_scale; d = max|v| over the chunk and
 *                codes = e4m3(v / d). An E4M3 MMA carries one scale per chunk, so the staged value is
 *                the scheme's value rounded once to an E4M3 code of the chunk's scale (at most half an
 *                E4M3 code step of the chunk maximum).
 * A chunk whose scale is 0 stages zero codes and d = 0. The staged value equals gguf.quant.ActFakeQuant's
 * for the FP8, MXFP8 and MXFP4 schemes.
 *
 * Launch: Fp8Static, Fp8Mx and Fp4Mx are 1-D with one work-item per (token, chunk); Fp8Row is 1-D with
 * one workgroup per token row and Fp8Group128 one workgroup per 128-element run; Fp4Nv runs one
 * workgroup per (token, chunk) over nf_wg_max reductions (local size L, a power of two <= LLM_DIMX).
 * tk divides K and 16; K is a multiple of 128 for Fp8Group128, and of tk/16 16-element runs for Fp4Nv.
 */

/* The scaled activation value of one element: x * m for mode 0, x / m for mode 1 (the FP8_TOKEN scheme
 * divides), 2^e * x for mode 2 (the MX schemes, exact). */
NF_INLINE float nf_fp8_scaled(float v, int mode, float m, int e) {
    return mode == 1 ? nf_aq_div(v, m) : (mode == 2 ? ldexp(v, e) : v * m);
}

/* Writes the record of one chunk (tk E4M3 codes of xp scaled per mode, the chunk scale d): tk / 4 packed
 * code ints, d, zero code sums and zero point. */
NF_INLINE void nf_fp8_store(NF_GLOBAL uint* q8, nf_i64 rec, NF_GLOBAL const float* xp, int mode, float m,
                            int e, float d, int tk) {
    for (int u = 0; u < tk / 4; u++) {
        uint packed = 0;
        for (int b = 0; b < 4; b++) {
            packed |= (nf_aq_e4m3_encode(nf_fp8_scaled(xp[4 * u + b], mode, m, e)) & 0xFFu) << (8 * b);
        }
        q8[rec + u] = packed;
    }
    q8[rec + tk / 4] = nf_q8_float_rec(d);
    for (int h = 0; h < tk / 16; h++) {
        q8[rec + tk / 4 + 1 + h] = 0u;
    }
    q8[rec + tk / 4 + 1 + tk / 16] = NF_Q8_NO_ZERO_POINT;
}

NF_KERNEL void LLM_Quant_MMQ_Fp8Static(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off, const float scale,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL uint* NF_RESTRICT q8) {
    const nf_i64 total = (nf_i64)N * (K / tk);
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (i >= total) return;
    const int chunks = K / tk;
    const int t = (int)(i / chunks);
    const int c = (int)(i % chunks);
    NF_GLOBAL const float* row = x + (nf_i64)x_off + (nf_i64)t * x_stride + (nf_i64)c * tk;
    const float inv = scale == 0.0f ? 0.0f : nf_aq_div(1.0f, scale);
    nf_fp8_store(q8, i * (nf_i64)NF_Q8_REC(tk), row, 0, inv, 0, scale, tk);
}

NF_KERNEL void LLM_Quant_MMQ_Fp8Row(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off, const float ub,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL uint* NF_RESTRICT q8) {
    NF_LOCAL float red[LLM_DIMX];
    const int t = get_group_id(0);
    if (t >= N) return;
    const int lid = get_local_id(0), lsize = get_local_size(0);
    NF_GLOBAL const float* row = x + (nf_i64)x_off + (nf_i64)t * x_stride;
    float m = 0.0f;
    for (int k = lid; k < K; k += lsize) {
        m = fmax(m, fabs(row[k]));
    }
    const float a = fmin(nf_wg_max(red, m), ub);
    const float d = fmax(nf_aq_div(a, NF_AQ_FP8_MAX), as_float(NF_AQ_FP8_MIN_SCALE_BITS));
    const int chunks = K / tk;
    for (int c = lid; c < chunks; c += lsize) {
        const nf_i64 rec = ((nf_i64)t * chunks + c) * NF_Q8_REC(tk);
        nf_fp8_store(q8, rec, row + (nf_i64)c * tk, 1, d, 0, d, tk);
    }
}

NF_KERNEL void LLM_Quant_MMQ_Fp8Group128(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL uint* NF_RESTRICT q8) {
    NF_LOCAL float red[LLM_DIMX];
    const int runs_per_row = K / 128;
    const int run = get_group_id(0);
    if (run >= N * runs_per_row) return;
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int t = run / runs_per_row;
    const int r = run - t * runs_per_row;
    NF_GLOBAL const float* run_x = x + (nf_i64)x_off + (nf_i64)t * x_stride + (nf_i64)r * 128;
    float m = 0.0f;
    for (int i = lid; i < 128; i += lsize) {
        m = fmax(m, fabs(run_x[i]));
    }
    const float d = nf_aq_div(fmax(nf_wg_max(red, m), as_float(NF_AQ_GROUP_EPS_BITS)), NF_AQ_FP8_MAX);
    const float inv = nf_aq_div(1.0f, d);
    const int chunks = K / tk;
    const int run_chunks = 128 / tk;
    for (int j = lid; j < run_chunks; j += lsize) {
        const nf_i64 rec = ((nf_i64)t * chunks + r * run_chunks + j) * NF_Q8_REC(tk);
        nf_fp8_store(q8, rec, run_x + (nf_i64)j * tk, 0, inv, 0, d, tk);
    }
}

NF_KERNEL void LLM_Quant_MMQ_Fp8Mx(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL uint* NF_RESTRICT q8) {
    const nf_i64 total = (nf_i64)N * (K / tk);
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (i >= total) return;
    const int chunks = K / tk;
    const int t = (int)(i / chunks);
    const int c = (int)(i % chunks);
    NF_GLOBAL const float* row = x + (nf_i64)x_off + (nf_i64)t * x_stride + (nf_i64)c * tk;
    float m = 0.0f;
    for (int k = 0; k < tk; k++) {
        m = fmax(m, fabs(row[k]));
    }
    const int e = nf_aq_mxfp8_exp(m);
    nf_fp8_store(q8, i * (nf_i64)NF_Q8_REC(tk), row, 2, 0.0f, -e, ldexp(1.0f, e), tk);
}

NF_KERNEL void LLM_Quant_MMQ_Fp4Mx(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL uint* NF_RESTRICT q8) {
    const nf_i64 total = (nf_i64)N * (K / tk);
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (i >= total) return;
    const int chunks = K / tk;
    const int t = (int)(i / chunks);
    const int c = (int)(i % chunks);
    NF_GLOBAL const float* row = x + (nf_i64)x_off + (nf_i64)t * x_stride + (nf_i64)c * tk;
    const nf_i64 rec = i * (nf_i64)NF_Q8_REC(tk);
    float m = 0.0f;
    for (int k = 0; k < tk; k++) {
        m = fmax(m, fabs(row[k]));
    }
    const int e = nf_aq_mxfp4_exp(m);
    for (int u = 0; u < tk / 4; u++) {
        uint packed = 0;
        for (int b = 0; b < 4; b++) {
            const uint code = nf_aq_e2m1_encode(ldexp(row[4 * u + b], -e));
            packed |= (nf_aq_e4m3_encode(nf_aq_e2m1_value(code)) & 0xFFu) << (8 * b);
        }
        q8[rec + u] = packed;
    }
    q8[rec + tk / 4] = nf_q8_float_rec(ldexp(1.0f, e));
    for (int h = 0; h < tk / 16; h++) {
        q8[rec + tk / 4 + 1 + h] = 0u;
    }
    q8[rec + tk / 4 + 1 + tk / 16] = NF_Q8_NO_ZERO_POINT;
}

#if NF_Q8_MAX_TK > 32
#error "LLM_Quant_MMQ_Fp4Nv holds the s16 of at most two 16-element runs per chunk"
#endif
NF_KERNEL void LLM_Quant_MMQ_Fp4Nv(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off, const float global_scale,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL uint* NF_RESTRICT q8) {
    NF_LOCAL float red[LLM_DIMX];
    const int chunk = get_group_id(0);
    if (chunk >= N * (K / tk)) return;
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int chunks = K / tk;
    const int t = chunk / chunks;
    const int c = chunk - t * chunks;
    NF_GLOBAL const float* row = x + (nf_i64)x_off + (nf_i64)t * x_stride + (nf_i64)c * tk;
    const nf_i64 rec = (nf_i64)chunk * NF_Q8_REC(tk);
    /* s16 of the chunk's (at most two) 16-element runs, held as two scalars so they stay registers, and
     * the chunk's largest |v| (every work-item holds the same reduction results). */
    float s16[2] = {0.0f, 0.0f};
    float vmax = 0.0f;
    for (int h = 0; h < tk / 16; h++) {
        float m = 0.0f;
        for (int k = lid; k < 16; k += lsize) {
            m = fmax(m, fabs(row[h * 16 + k]));
        }
        const float s = nf_aq_e4m3(nf_aq_div(nf_aq_div(nf_wg_max(red, m), NF_AQ_FP4_MAX), global_scale));
        s16[0] = h == 0 ? s : s16[0];
        s16[1] = h == 1 ? s : s16[1];
        for (int k = lid; k < 16; k += lsize) {
            const float sd = s * global_scale;
            const float v = sd == 0.0f ? 0.0f
                : (nf_aq_e2m1(nf_aq_div(row[h * 16 + k], sd)) * s) * global_scale;
            vmax = fmax(vmax, fabs(v));
        }
    }
    const float d = nf_wg_max(red, vmax);
    const float inv = d == 0.0f ? 0.0f : nf_aq_div(1.0f, d);
    for (int u = 0; u < tk / 4; u++) {
        uint packed = 0;
        for (int b = 0; b < 4; b++) {
            const int k = 4 * u + b;
            const float s = k < 16 ? s16[0] : s16[1];
            const float sd = s * global_scale;
            const float v = sd == 0.0f ? 0.0f
                : (nf_aq_e2m1(nf_aq_div(row[k], sd)) * s) * global_scale;
            packed |= (nf_aq_e4m3_encode(v * inv) & 0xFFu) << (8 * b);
        }
        q8[rec + u] = packed;
    }
    q8[rec + tk / 4] = nf_q8_float_rec(d);
    for (int h = 0; h < tk / 16; h++) {
        q8[rec + tk / 4 + 1 + h] = 0u;
    }
    q8[rec + tk / 4 + 1 + tk / 16] = NF_Q8_NO_ZERO_POINT;
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
NF_INLINE float nf_fp8_stage_dot(NF_LOCAL_PTR const nf_i8* wq_row, NF_LOCAL_PTR const uint* rec) {
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
 * tn is the tile's token count (nf_mmq_tensor_stage's). The token codes and chunk scales read the
 * staged record tile xr the gather filled (dead tokens stage zero codes and a 0.0f scale). */
NF_INLINE void nf_fp8_combine(NF_LOCAL_PTR const nf_i8* wq, NF_LOCAL_PTR const uint* xr,
                              int n0, int tn, int N,
                              NF_LOCAL_PTR const float* wsc, int lid, float acc[]) {
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
        float sd[4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        const int rb = cb * NF_MMQ_RC;
        const uint b[2] = {xr[rb + kf / 4], xr[rb + kf / 4 + NF_MMQ_TK / 8]};
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        nf_fp8_mma_m16n8k32(d, a, b);
        for (int i = 0; i < 4; i++) {
            sd[i] = wsc[rr[i] * NF_MMQ_SH] * d[i];
        }
#else
        for (int i = 0; i < 4; i++) {
            const int tok = n0 + cc[i];
            NF_LOCAL_PTR const uint* trec = xr + cc[i] * NF_MMQ_RC;
            sd[i] = tok < N ? wsc[rr[i] * NF_MMQ_SH] * nf_fp8_stage_dot(wq + rr[i] * NF_MMQ_TK, trec) : 0.0f;
        }
#endif
        for (int i = 0; i < 4; i++) {
            acc[nb * 4 + i] += nf_mmq_rec_d(xr[cc[i] * NF_MMQ_RC + NF_MMQ_QS]) * sd[i];
        }
    }
#else
    const int tx = lid % NF_MMQ_TX, ty = lid / NF_MMQ_TX;
    for (int j = 0; j < tn / 16; j++) {
        const int tok = ty + 16 * j;
        if (n0 + tok >= N) {
            continue;
        }
        NF_LOCAL_PTR const uint* rec = xr + tok * NF_MMQ_RC;
        for (int q = 0; q < NF_MMQ_RM; q++) {
            const int sr = tx + NF_MMQ_TX * q;
            acc[j * NF_MMQ_RM + q] += nf_mmq_rec_d(rec[NF_MMQ_QS]) * (wsc[sr * NF_MMQ_SH]
                    * nf_fp8_stage_dot(wq + sr * NF_MMQ_TK, rec));
        }
    }
#endif
}

#if NF_MMQ_TENSOR
/*
 * The E4M3 lane on the tensor-core device tile (T_MatMul_MMQ / T_MatMul_W4A8 and their KSplit forms):
 * the int8 tile's staging, pipeline and warp layout with the E4M3 combine. Each output adds
 * d_t * (s_w * dot) - the sm_89+ E4M3 mma dot, or below sm_89 the scalar dot in nf_fp8_stage_dot's
 * element order - rounded step by step as nf_fp8_combine's compiled form rounds it (fmul, then fma),
 * so the plain kernels equal the twins bit for bit.
 */

/* nf_fp8_stage_dot over staged row r of the swizzled code tile. */
NF_INLINE float nf_fp8_tc_dot(NF_LOCAL_PTR const int* wq, int r, NF_LOCAL_PTR const uint* rec) {
    float sum = 0.0f;
    for (int kk = 0; kk < NF_MMQ_TK; kk++) {
        const uint word = (uint)wq[NF_MMQ_TC_SWZ(r, kk >> 2)];
        const uint wc = (word >> (8u * (kk & 3u))) & 0xFFu;
        const uint xc = (rec[kk >> 2] >> (8u * (kk & 3u))) & 0xFFu;
        sum += F8_E4M3_LUT[wc] * F8_E4M3_LUT[xc];
    }
    return sum;
}

/*
 * One chunk of the E4M3 tile: as nf_mmq_tc_combine32 over one scale per row and chunk, no offset. The same
 * shared reads as there - one 128-bit read of the lane's four row scales and, on sm_89+, one ldmatrix.x4 of
 * B per block pair (the E4M3 m16n8k32 fragments have the s8 layout) - so the tile holds few live addresses.
 */
NF_INLINE void nf_fp8_tc_combine(NF_LOCAL_PTR const int* wq, NF_LOCAL_PTR const uint* xr,
                                 NF_LOCAL_PTR const float* wsc, int lane, int wrow0, int wtok0, float acc[]) {
    const int g = lane >> 2, q = lane & 3;
    /* entry 2m + j: row wrow0 + 16m + g + 8j (the lane order of NF_MMQ_TC_WSLOT) */
    const float4 vx = *(NF_LOCAL_PTR const float4*)(wsc + NF_MMQ_TC_WSLOT(wrow0 + g, 0, 1));
    const float dmx[4] = {vx.x, vx.y, vx.z, vx.w};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lword = (((lane >> 4) & 1) ^ ((lane >> 2) & 1)) * 4;
    const int btok = (lane & 7) + 8 * (lane >> 4);
    const int bword = ((lane >> 3) & 1) * 4;
    uint a[NF_MMQ_TC_NTX][4];
    for (int m = 0; m < NF_MMQ_TC_NTX; m++) {
        nf_mmq_ldsm_x4(a[m], wq + (wrow0 + 16 * m + lrow) * NF_MMQ_QS + lword);
    }
#endif
    for (int p = 0; p < NF_MMQ_TC_NB / 2; p++) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        uint bp[4];
        nf_mmq_ldsm_x4(bp, (NF_LOCAL_PTR const int*)xr + (wtok0 + 16 * p + btok) * NF_MMQ_RC + bword);
#endif
        for (int o = 0; o < 2; o++) {
            const int tb = wtok0 + 8 * (2 * p + o);
            float tds[2];
            for (int t = 0; t < 2; t++) {
                tds[t] = nf_mmq_rec_d(xr[(tb + 2 * q + t) * NF_MMQ_RC + NF_MMQ_TC_XF]);
            }
            for (int m = 0; m < NF_MMQ_TC_NTX; m++) {
                float* ac = acc + (m * NF_MMQ_TC_NB + 2 * p + o) * 4;
                float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
                const uint b[2] = {bp[2 * o], bp[2 * o + 1]};
                nf_fp8_mma_m16n8k32(d, a[m], b);
#else
                for (int i = 0; i < 4; i++) {
                    d[i] = nf_fp8_tc_dot(wq, wrow0 + 16 * m + g + 8 * (i >> 1),
                                         xr + (tb + 2 * q + (i & 1)) * NF_MMQ_RC);
                }
#endif
                for (int i = 0; i < 4; i++) {
                    ac[i] = __fmaf_rn(tds[i & 1], __fmul_rn(dmx[2 * m + (i >> 1)], d[i]), ac[i]);
                }
            }
        }
    }
}

/* The E4M3 combine of weight buffer buf and the record stage at unit offset xoff (the COMBINE argument
 * of NF_MMQ_TC_LOOP). */
#define NF_FP8_TC_COMBINE(buf, xoff, SG, ASYM, HAS_OFF, EXACT) \
    nf_fp8_tc_combine(tc_wq + (buf) * NF_MMQ_TC_CW, tc_xr + (xoff), \
                      tc_wsc + (buf) * NF_MMQ_TC_SB(SG), lane, wrow0, wtok0, acc);

/* The E4M3 tile's chunks [cstart, cend): one scale per 32-element chunk, no offset, no zero point; both
 * chunks of a step load ahead of its combines where the target takes two prefetch register sets
 * (NF_MMQ_TC_SETS). */
#define NF_FP8_TC_PIPE(cstart, cend, SG, HAS_OFF, BS, BB, CODE, SCALE, OFF) \
    NF_MMQ_TC_LOOP(cstart, cend, NF_FP8_TC_COMBINE, 32, 0, 0, 0, BS, BB, CODE, SCALE, OFF, NF_MMQ_TC_SETS(2))
#endif /* NF_MMQ_TENSOR */

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
    nf_fp8_combine(wq + (sel) * NF_MMQ_WS, xr, n0, (TN), N, \
                   wsc + (sel) * NF_MMQ_TM * NF_MMQ_SH, lid, acc)

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

/* The device-tile kernels: the tensor-core tile on the tensor path, the pipelined stage elsewhere. */
#if NF_MMQ_TENSOR
#define NF_FP8Q_GEMM_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_MMQ_TC_GEMM_NAMED(NAME, NF_FP8_TC_PIPE, BS, BB, 32, 0, CODE, SCALE, NF_FP8Q_NO_OFFSET)
#else
#define NF_FP8Q_GEMM_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_TN_NAMED(NAME, NF_MMQ_TN, BS, BB, CODE, SCALE)
#endif

#define NF_FP8Q_GEMM_S_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_TN_NAMED(NAME, 16, BS, BB, CODE, SCALE)

#define NF_FP8Q_GEMM_S64_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_TN_NAMED(NAME, 64, BS, BB, CODE, SCALE)

#define NF_FP8Q_GEMM_KSPLIT_TN_NAMED(NAME, TN, BS, BB, CODE, SCALE) \
NF_KERNEL void NAME( \
    const int M, const int K, const int N, \
    const int q_stride, const int q_off, const int k_slice, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uint* NF_RESTRICT q8, \
    NF_GLOBAL float* NF_RESTRICT partial) { \
    NF_FP8Q_KERNEL_BODY((TN), (BS), (BB), \
        const int rtiles = (M + NF_MMQ_TM - 1) / NF_MMQ_TM; \
        const int slice = (int)get_group_id(0) / rtiles; \
        const int m0 = ((int)get_group_id(0) - slice * rtiles) * NF_MMQ_TM; \
        const int n0 = (int)get_group_id(1) * (TN); \
        const int k0 = slice * k_slice; \
        const int k1 = k0 + k_slice < K ? k0 + k_slice : K; \
        NF_MMQ_PIPE(k0, k1, (TN), NF_FP8Q_COMBINE, (BS), (BB), CODE, SCALE, NF_FP8Q_NO_OFFSET, (TN), 0) \
        nf_mmq_store(partial, M, N, M, (nf_i64)slice * M * N, m0, n0, (TN), lid, acc); \
    ) \
}

#if NF_MMQ_TENSOR
#define NF_FP8Q_GEMM_KSPLIT_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_MMQ_TC_KSPLIT_NAMED(NAME, NF_FP8_TC_PIPE, BS, BB, 32, 0, CODE, SCALE, NF_FP8Q_NO_OFFSET)
#else
#define NF_FP8Q_GEMM_KSPLIT_NAMED(NAME, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_TN_NAMED(NAME, NF_MMQ_TN, BS, BB, CODE, SCALE)
#endif

#define NF_FP8Q_GEMM(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_NAMED(T##_MatMul_MMQ, BS, BB, CODE, SCALE)
#define NF_FP8Q_GEMM_S(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_S_NAMED(T##_MatMul_MMQ_S, BS, BB, CODE, SCALE)
#define NF_FP8Q_GEMM_S64(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_S64_NAMED(T##_MatMul_MMQ_S64, BS, BB, CODE, SCALE)
#define NF_FP8Q_GEMM_KSPLIT(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_NAMED(T##_MatMul_MMQ_KSplit, BS, BB, CODE, SCALE)
#define NF_FP8Q_GEMM_SKSPLIT(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_TN_NAMED(T##_MatMul_MMQ_SKSplit, 16, BS, BB, CODE, SCALE)
#define NF_FP8Q_GEMM_SKSPLIT64(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_TN_NAMED(T##_MatMul_MMQ_SKSplit64, 64, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_NAMED(T##_MatMul_W4A8, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM_S(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_S_NAMED(T##_MatMul_W4A8_S, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM_S64(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_S64_NAMED(T##_MatMul_W4A8_S64, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM_KSPLIT(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_NAMED(T##_MatMul_W4A8_KSplit, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM_SKSPLIT(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_TN_NAMED(T##_MatMul_W4A8_SKSplit, 16, BS, BB, CODE, SCALE)
#define NF_W4A8_GEMM_SKSPLIT64(T, BS, BB, CODE, SCALE) \
    NF_FP8Q_GEMM_KSPLIT_TN_NAMED(T##_MatMul_W4A8_SKSplit64, 64, BS, BB, CODE, SCALE)

/* ---- E4M3 GEMMs: FP8 weights and 4-bit weights served as E4M3 ----
 *
 * These kernels exist in the CUDA dialect only: their stage dot is the FP8 tensor-core instruction
 * (mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32, sm_89+) or, below sm_89, a scalar E4M3 dot.
 * The activations arrive staged as E4M3 codes with one scale per chunk (LLM_Quant_MMQ_Fp8*), so both
 * operands are read in the checkpoint's own code form. K is a multiple of the type's block size.
 */
/* F8_E4M3: qs[32] E4M3 codes, no block scale (the tensor's .scale applies after the GEMM). */
#define F8_E4M3_MMQ_CODE(bp, e) (nf_i8)((NF_GLOBAL const uchar*)(bp))[NF_F8_E4M3_QS + (e)]
/* Codes e0..e0+3: the raw E4M3 bytes as one aligned u32 (the 32-byte block is 4-byte aligned). */
#define F8_E4M3_MMQ_CODE4(bp, e0) nf_read_u32((bp) + NF_F8_E4M3_QS + (e0))
#define F8_E4M3_MMQ_SCALE(bp, k0) 1.0f
NF_FP8Q_GEMM(F8_E4M3, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES, F8_E4M3_MMQ_CODE, F8_E4M3_MMQ_SCALE)
NF_FP8Q_GEMM_KSPLIT(F8_E4M3, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES, F8_E4M3_MMQ_CODE,
    F8_E4M3_MMQ_SCALE)

/* F8_E4M3_B128: f32 d, qs[128] E4M3 codes. */
#define F8_E4M3_B128_MMQ_CODE(bp, e) (nf_i8)((NF_GLOBAL const uchar*)(bp))[NF_F8_E4M3_B128_QS + (e)]
#define F8_E4M3_B128_MMQ_CODE4(bp, e0) nf_read_u32((bp) + NF_F8_E4M3_B128_QS + (e0))
#define F8_E4M3_B128_MMQ_SCALE(bp, k0) nf_xq_f32((bp), NF_F8_E4M3_B128_D)
NF_FP8Q_GEMM(F8_E4M3_B128, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F8_E4M3_B128_MMQ_CODE,
    F8_E4M3_B128_MMQ_SCALE)
NF_FP8Q_GEMM_KSPLIT(F8_E4M3_B128, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F8_E4M3_B128_MMQ_CODE,
    F8_E4M3_B128_MMQ_SCALE)

/* F8_E4M3_B32: f32 d, qs[32] E4M3 codes. */
#define F8_E4M3_B32_MMQ_CODE(bp, e) (nf_i8)((NF_GLOBAL const uchar*)(bp))[NF_F8_E4M3_B32_QS + (e)]
#define F8_E4M3_B32_MMQ_CODE4(bp, e0) nf_read_u32((bp) + NF_F8_E4M3_B32_QS + (e0))
#define F8_E4M3_B32_MMQ_SCALE(bp, k0) nf_xq_f32((bp), NF_F8_E4M3_B32_D)
NF_FP8Q_GEMM(F8_E4M3_B32, F8_E4M3_B32_BLOCK_SIZE, F8_E4M3_B32_BLOCK_BYTES, F8_E4M3_B32_MMQ_CODE,
    F8_E4M3_B32_MMQ_SCALE)
NF_FP8Q_GEMM_KSPLIT(F8_E4M3_B32, F8_E4M3_B32_BLOCK_SIZE, F8_E4M3_B32_BLOCK_BYTES, F8_E4M3_B32_MMQ_CODE,
    F8_E4M3_B32_MMQ_SCALE)

/* MXFP8: e(e8m0)@0, qs[32] E4M3 codes; scale 2^(e - 127). */
#define MXFP8_MMQ_CODE(bp, e) (nf_i8)((NF_GLOBAL const uchar*)(bp))[NF_MXFP8_QS + (e)]
/* The 33-byte block keeps no 2-byte alignment across blocks, so the group packs 4 CODE calls. */
#define MXFP8_MMQ_CODE4(bp, e0) \
    NF_MMQ_CODE4_PACK(MXFP8_MMQ_CODE((bp), (e0)), MXFP8_MMQ_CODE((bp), (e0) + 1), \
        MXFP8_MMQ_CODE((bp), (e0) + 2), MXFP8_MMQ_CODE((bp), (e0) + 3))
#define MXFP8_MMQ_SCALE(bp, k0) e8m0_scale((uint)(bp)[NF_MXFP8_E])
NF_FP8Q_GEMM(MXFP8, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES, MXFP8_MMQ_CODE, MXFP8_MMQ_SCALE)
NF_FP8Q_GEMM_KSPLIT(MXFP8, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES, MXFP8_MMQ_CODE, MXFP8_MMQ_SCALE)

/* E4M3 codes of the E2M1 codebook values (0, 1, 2, 3, 4, 6, 8, 12 and their negatives, the doubled
 * MXFP4_KVALUES entries both 4-bit codebooks use); every E2M1 value is exact in E4M3. */
NF_TABLE(uchar, NF_E2M1_E4M3, 16) = {
    0x00u, 0x38u, 0x40u, 0x44u, 0x48u, 0x4Cu, 0x50u, 0x54u,
    0x00u, 0xB8u, 0xC0u, 0xC4u, 0xC8u, 0xCCu, 0xD0u, 0xD4u
};

/* E4M3 codes of the IQ4_NL / IQ4_XS codebook values, rounded to nearest even (the codebook's integers
 * need seven bits of mantissa, so the conversion is one E4M3 rounding, at most 5.3% of the value). */
NF_TABLE(uchar, NF_IQ4_E4M3, 16) = {
    0xF0u, 0xEDu, 0xEAu, 0xE8u, 0xE4u, 0xE1u, 0xDBu, 0xD2u,
    0x38u, 0x55u, 0x5Cu, 0x62u, 0x65u, 0x69u, 0x6Bu, 0x6Eu
};

/* MXFP4: e(e8m0)@0, qs[16] E2M1 nibbles; value = doubled code * half scale. */
#define MXFP4_W4A8_CODE(bp, e) (nf_i8)NF_E2M1_E4M3[(e) < 16 \
    ? ((uint)(bp)[1 + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[1 + ((e) & 15)] >> 4)]
/* The 17-byte block keeps no 2-byte alignment across blocks, so the group packs 4 CODE calls. */
#define MXFP4_W4A8_CODE4(bp, e0) \
    NF_MMQ_CODE4_PACK(MXFP4_W4A8_CODE((bp), (e0)), MXFP4_W4A8_CODE((bp), (e0) + 1), \
        MXFP4_W4A8_CODE((bp), (e0) + 2), MXFP4_W4A8_CODE((bp), (e0) + 3))
#define MXFP4_W4A8_SCALE(bp, k0) mxfp4_scale((uint)(bp)[0])
NF_W4A8_GEMM(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_W4A8_CODE, MXFP4_W4A8_SCALE)
NF_W4A8_GEMM_KSPLIT(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_W4A8_CODE, MXFP4_W4A8_SCALE)

/* IQ4_NL: d(f16)@0, qs[16] codebook nibbles; value = d * kvalues_iq4nl[nibble]. */
#define IQ4_NL_W4A8_CODE(bp, e) (nf_i8)NF_IQ4_E4M3[(e) < 16 \
    ? ((uint)(bp)[2 + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[2 + ((e) & 15)] >> 4)]
/* Codes e0..e0+3: the group's 4 qs bytes (two aligned u16s) through the E4M3 codebook. */
NF_INLINE uint nf_w4a8_iq4nl_code4(NF_GLOBAL const uchar* bp, int e0) {
    const uint word = nf_read_u32_u16(bp + 2 + (e0 & 15));
    const int shift = (e0 & 16) ? 4 : 0;
    return NF_MMQ_CODE4_PACK(NF_IQ4_E4M3[(word >> shift) & 0xFu],
        NF_IQ4_E4M3[(word >> (8 + shift)) & 0xFu],
        NF_IQ4_E4M3[(word >> (16 + shift)) & 0xFu],
        NF_IQ4_E4M3[(word >> (24 + shift)) & 0xFu]);
}
#define IQ4_NL_W4A8_CODE4(bp, e0) nf_w4a8_iq4nl_code4((bp), (e0))
#define IQ4_NL_W4A8_SCALE(bp, k0) nf_ql_half((bp), 0)
NF_W4A8_GEMM(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_W4A8_CODE, IQ4_NL_W4A8_SCALE)
NF_W4A8_GEMM_KSPLIT(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_W4A8_CODE, IQ4_NL_W4A8_SCALE)

/* IQ4_XS: d(f16)@0, sub-block scales@2, qs[128]@8; 32-element sub-blocks carry the scale. */
#define IQ4_XS_W4A8_CODE(bp, e) (nf_i8)NF_IQ4_E4M3[(((e) % 32) < 16 \
    ? ((uint)(bp)[IQ4_XS_HEADER_BYTES + ((e) >> 5) * IQ4_XS_SUB_BLOCK_QUANT_BYTES + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[IQ4_XS_HEADER_BYTES + ((e) >> 5) * IQ4_XS_SUB_BLOCK_QUANT_BYTES + ((e) & 15)] >> 4))]
/* Codes e0..e0+3: the group's qs bytes as one aligned u32 through the E4M3 codebook. */
NF_INLINE uint nf_w4a8_iq4xs_code4(NF_GLOBAL const uchar* bp, int e0) {
    const uint word = nf_read_u32(bp + IQ4_XS_HEADER_BYTES
        + (e0 >> 5) * IQ4_XS_SUB_BLOCK_QUANT_BYTES + (e0 & 15));
    const int shift = ((e0 % 32) < 16) ? 0 : 4;
    return NF_MMQ_CODE4_PACK(NF_IQ4_E4M3[(word >> shift) & 0xFu],
        NF_IQ4_E4M3[(word >> (8 + shift)) & 0xFu],
        NF_IQ4_E4M3[(word >> (16 + shift)) & 0xFu],
        NF_IQ4_E4M3[(word >> (24 + shift)) & 0xFu]);
}
#define IQ4_XS_W4A8_CODE4(bp, e0) nf_w4a8_iq4xs_code4((bp), (e0))
#define IQ4_XS_W4A8_SCALE(bp, k0) iq4xs_sub_scale((bp), ((k0) % 256) >> 5)
NF_W4A8_GEMM(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, IQ4_XS_W4A8_CODE, IQ4_XS_W4A8_SCALE)
NF_W4A8_GEMM_KSPLIT(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, IQ4_XS_W4A8_CODE, IQ4_XS_W4A8_SCALE)

/* NVFP4: d[4](ue4m3)@0, qs[32] E2M1 nibbles, 16-element sub-blocks with their own scale. The stage
 * chunk carries one scale, the larger of the sub-block scales it spans, and a code is the sub-block
 * value relative to it: the E2M1 code value times (sub-block scale / chunk scale), one E4M3 rounding. */
NF_INLINE float nf_nvfp4_w4a8_chunk_scale(NF_GLOBAL const uchar* bp, int e) {
    /* First sub-block of the chunk holding element e of the block: the chunk start (e rounded down to
     * the stage depth, within the block) divided by the sub-block length. */
    const int sub = (((e) % NVFP4_BLOCK_SIZE) / NF_MMQ_TK) * (NF_MMQ_TK / 16);
    const float a = nf_nq_ue4m3_half((uint)bp[NF_NVFP4_D + sub]);
#if NF_MMQ_TK == 32
    return fmax(a, nf_nq_ue4m3_half((uint)bp[NF_NVFP4_D + sub + 1]));
#else
    return a;
#endif
}
NF_INLINE float nf_nvfp4_w4a8_ratio(NF_GLOBAL const uchar* bp, int e) {
    const float d = nf_nvfp4_w4a8_chunk_scale(bp, e);
    return d == 0.0f ? 0.0f : nf_nq_ue4m3_half((uint)bp[NF_NVFP4_D + (e >> 4)]) / d;
}
NF_INLINE nf_i8 nf_nvfp4_w4a8_code(NF_GLOBAL const uchar* bp, int e) {
    const int t = e & 15;
    const uint q = bp[NF_NVFP4_QS + (e >> 4) * 8 + (t & 7)];
    const uint nib = t < 8 ? (q & 0xFu) : (q >> 4);
    return (nf_i8)nf_aq_e4m3_encode(MXFP4_KVALUES[nib] * nf_nvfp4_w4a8_ratio(bp, e));
}
#define NVFP4_W4A8_CODE(bp, e) nf_nvfp4_w4a8_code((bp), (e))
/* The code pipeline is per element (sub-block scale ratio, one E4M3 rounding), so the group packs
 * 4 CODE calls. */
#define NVFP4_W4A8_CODE4(bp, e0) \
    NF_MMQ_CODE4_PACK(NVFP4_W4A8_CODE((bp), (e0)), NVFP4_W4A8_CODE((bp), (e0) + 1), \
        NVFP4_W4A8_CODE((bp), (e0) + 2), NVFP4_W4A8_CODE((bp), (e0) + 3))
#define NVFP4_W4A8_SCALE(bp, k0) nf_nvfp4_w4a8_chunk_scale((bp), (k0))
NF_W4A8_GEMM(NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_W4A8_CODE, NVFP4_W4A8_SCALE)
NF_W4A8_GEMM_KSPLIT(NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_W4A8_CODE, NVFP4_W4A8_SCALE)

/* ---- small-token-tile twins (T_MatMul_MMQ_S / T_MatMul_W4A8_S with their SKSplit over 16-token
 * tiles, and the S64/SKSplit64 forms over 64-token tiles) ---- As in kernels/llm/matmul_mmq.cl:
 * generated when the device tile is wider than 16 tokens, the host serves a staged launch of fewer
 * tokens than the device tile with the twin of its ladder row (16 tokens up to 16, 64 above) and
 * splits it with the matching SKSplit. The twins keep the row tile of the packed-dot family:
 * NF_MMQ_TWIN_RM rows per work-item and at most NF_MMQ_RM * 16 rows, whose 256-item workgroups pack
 * two blocks per SM. The file is CUDA-only ([cuda]), so the twins exist on the sm_80+
 * segments the tensor datapath compiles for and are absent below sm_80, where the packed-dot tile is
 * 16 tokens already. */

#if NF_MMQ_TN > 16
#if defined(NF_MMQ_TWIN_RM) && NF_MMQ_TWIN_RM != NF_MMQ_RM
#undef NF_MMQ_RM
#define NF_MMQ_RM NF_MMQ_TWIN_RM
#endif
#if NF_MMQ_TM / NF_MMQ_RM > 16
#undef NF_MMQ_TM
#define NF_MMQ_TM (NF_MMQ_RM * 16)
#endif
NF_FP8Q_GEMM_S(F8_E4M3, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES, F8_E4M3_MMQ_CODE, F8_E4M3_MMQ_SCALE)
NF_FP8Q_GEMM_S(F8_E4M3_B128, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F8_E4M3_B128_MMQ_CODE,
    F8_E4M3_B128_MMQ_SCALE)
NF_FP8Q_GEMM_S(F8_E4M3_B32, F8_E4M3_B32_BLOCK_SIZE, F8_E4M3_B32_BLOCK_BYTES, F8_E4M3_B32_MMQ_CODE,
    F8_E4M3_B32_MMQ_SCALE)
NF_FP8Q_GEMM_S(MXFP8, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES, MXFP8_MMQ_CODE, MXFP8_MMQ_SCALE)
NF_W4A8_GEMM_S(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_W4A8_CODE, MXFP4_W4A8_SCALE)
NF_W4A8_GEMM_S(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_W4A8_CODE, IQ4_NL_W4A8_SCALE)
NF_W4A8_GEMM_S(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, IQ4_XS_W4A8_CODE, IQ4_XS_W4A8_SCALE)
NF_W4A8_GEMM_S(NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_W4A8_CODE, NVFP4_W4A8_SCALE)
#if NF_MMQ_TN > 64
NF_FP8Q_GEMM_S64(F8_E4M3, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES, F8_E4M3_MMQ_CODE, F8_E4M3_MMQ_SCALE)
NF_FP8Q_GEMM_S64(F8_E4M3_B128, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F8_E4M3_B128_MMQ_CODE,
    F8_E4M3_B128_MMQ_SCALE)
NF_FP8Q_GEMM_S64(F8_E4M3_B32, F8_E4M3_B32_BLOCK_SIZE, F8_E4M3_B32_BLOCK_BYTES, F8_E4M3_B32_MMQ_CODE,
    F8_E4M3_B32_MMQ_SCALE)
NF_FP8Q_GEMM_S64(MXFP8, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES, MXFP8_MMQ_CODE, MXFP8_MMQ_SCALE)
NF_W4A8_GEMM_S64(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_W4A8_CODE, MXFP4_W4A8_SCALE)
NF_W4A8_GEMM_S64(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_W4A8_CODE, IQ4_NL_W4A8_SCALE)
NF_W4A8_GEMM_S64(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, IQ4_XS_W4A8_CODE, IQ4_XS_W4A8_SCALE)
NF_W4A8_GEMM_S64(NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_W4A8_CODE, NVFP4_W4A8_SCALE)
NF_FP8Q_GEMM_SKSPLIT(F8_E4M3, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES, F8_E4M3_MMQ_CODE, F8_E4M3_MMQ_SCALE)
NF_FP8Q_GEMM_SKSPLIT(F8_E4M3_B128, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F8_E4M3_B128_MMQ_CODE,
    F8_E4M3_B128_MMQ_SCALE)
NF_FP8Q_GEMM_SKSPLIT(F8_E4M3_B32, F8_E4M3_B32_BLOCK_SIZE, F8_E4M3_B32_BLOCK_BYTES, F8_E4M3_B32_MMQ_CODE,
    F8_E4M3_B32_MMQ_SCALE)
NF_FP8Q_GEMM_SKSPLIT(MXFP8, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES, MXFP8_MMQ_CODE, MXFP8_MMQ_SCALE)
NF_W4A8_GEMM_SKSPLIT(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_W4A8_CODE, MXFP4_W4A8_SCALE)
NF_W4A8_GEMM_SKSPLIT(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_W4A8_CODE, IQ4_NL_W4A8_SCALE)
NF_W4A8_GEMM_SKSPLIT(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, IQ4_XS_W4A8_CODE, IQ4_XS_W4A8_SCALE)
NF_W4A8_GEMM_SKSPLIT(NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_W4A8_CODE, NVFP4_W4A8_SCALE)
NF_FP8Q_GEMM_SKSPLIT64(F8_E4M3, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES, F8_E4M3_MMQ_CODE, F8_E4M3_MMQ_SCALE)
NF_FP8Q_GEMM_SKSPLIT64(F8_E4M3_B128, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F8_E4M3_B128_MMQ_CODE,
    F8_E4M3_B128_MMQ_SCALE)
NF_FP8Q_GEMM_SKSPLIT64(F8_E4M3_B32, F8_E4M3_B32_BLOCK_SIZE, F8_E4M3_B32_BLOCK_BYTES, F8_E4M3_B32_MMQ_CODE,
    F8_E4M3_B32_MMQ_SCALE)
NF_FP8Q_GEMM_SKSPLIT64(MXFP8, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES, MXFP8_MMQ_CODE, MXFP8_MMQ_SCALE)
NF_W4A8_GEMM_SKSPLIT64(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_W4A8_CODE, MXFP4_W4A8_SCALE)
NF_W4A8_GEMM_SKSPLIT64(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_W4A8_CODE, IQ4_NL_W4A8_SCALE)
NF_W4A8_GEMM_SKSPLIT64(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, IQ4_XS_W4A8_CODE, IQ4_XS_W4A8_SCALE)
NF_W4A8_GEMM_SKSPLIT64(NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_W4A8_CODE, NVFP4_W4A8_SCALE)
#endif /* NF_MMQ_TN > 64 */
#endif /* NF_MMQ_TN > 16 */
