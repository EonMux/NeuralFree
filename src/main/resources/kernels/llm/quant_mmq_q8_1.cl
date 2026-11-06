/* quant_mmq_q8_1.cl - the MMQ activation quantization: stages every token row of a projection once,
 * as 8-bit codes per chunk, shared by all row tiles of the staged GEMMs (llama.cpp
 * quantize_mmq_q8_1), and the forms that quantize with the checkpoint's activation scheme
 * (gguf.quant.ActQuantScheme): the int8 code forms, the E4M3 forms of the FP8 and MX schemes, and
 * the E2M1 forms the 4-bit schemes re-encode as E4M3.
 * Requires: common/dialect.h, common/config.h, common/reduce.h, common/fp8_codec.h,
 * common/matmul_mmq_config.h */
/*
 * LLM_Quant_MMQ_Q8_1(i N, i K, i tk, i x_stride, i x_off, R float x, W uint q8)
 *   for every token t < N and chunk c < K / tk: the record of token t, chunk c at 32-bit unit index
 *   ((nf_i64)t * (K / tk) + c) * (tk / 4 + 2 + tk / 16) of q8 holds tk / 4 code ints (the code of
 *   depth k in byte k % 4, two's complement), the chunk scale d, the tk / 16 code sums of the
 *   16-element halves and the asymmetric zero point.
 *
 * The quantization matches the activation stage of kernels/llm/common/matmul_mmq_family.h exactly
 * (as llama.cpp quantize_row_q8_1): d = maxabs / 127 of the tk-deep chunk (1 for an all-zero chunk,
 * which keeps it out of the accumulator), the codes round to nearest even of x * (127 / maxabs),
 * and the sums add the codes of every 16-element half. tk is the MMQ stage depth (16 packed dot,
 * 32 int8 tensor; the host plans it, gguf.ops.gemm.MmqConfig) and divides K, as does 16.
 *
 * Launch contract: 1-D, global = roundUp(N * (K / tk), L) work-items of local size L; one work-item
 * per (token, chunk) record.
 */
NF_KERNEL void LLM_Quant_MMQ_Q8_1(
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
    float maxabs = 0.0f;
    for (int k = 0; k < tk; k++) {
        const float a = fabs(row[k]);
        maxabs = a > maxabs ? a : maxabs;
    }
    const float id = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
    const float d = maxabs > 0.0f ? maxabs * (1.0f / 127.0f) : 1.0f;
    const nf_i64 rec = i * (nf_i64)NF_Q8_REC(tk);
    int sums[NF_Q8_MAX_TK / 16];
    for (int h = 0; h < tk / 16; h++) {
        sums[h] = 0;
    }
    for (int u = 0; u < tk / 4; u++) {
        uint packed = 0;
        for (int b = 0; b < 4; b++) {
            const int q = nf_rint_i(row[4 * u + b] * id);
            packed |= (uint)(q & 0xFF) << (8 * b);
            sums[(4 * u + b) >> 4] += q;
        }
        q8[rec + u] = packed;
    }
    q8[rec + tk / 4] = nf_q8_float_rec(d);
    for (int h = 0; h < tk / 16; h++) {
        q8[rec + tk / 4 + 1 + h] = (uint)sums[h];
    }
    q8[rec + tk / 4 + 1 + tk / 16] = NF_Q8_NO_ZERO_POINT;
}

/*
 * LLM_Quant_MMQ_Int8Static(i N, i K, i tk, i x_stride, i x_off, f scale, R float x, W uint q8)
 * LLM_Quant_MMQ_Int8Row(i N, i K, i tk, i x_stride, i x_off, R float x, W uint q8)
 * LLM_Quant_MMQ_Int8Group128(i N, i K, i tk, i x_stride, i x_off, R float x, W uint q8)
 *
 * Stage the activations of the int8 activation schemes of gguf.quant.ActQuantScheme into the record
 * layout of LLM_Quant_MMQ_Q8_1, so that the int8 MMQ GEMM multiplies the checkpoint's own int8 codes
 * (with the weight's int8 or low-bit codes) instead of requantizing a fake-quantized float row:
 *   Int8Static    INT8_STATIC: s = scale, codes = sat8(rne(x / s))
 *   Int8Row       INT8_TOKEN: s = amax / 127 over the whole token row, codes = sat8(rne(x * (127 / amax)))
 *   Int8Group128  INT8_GROUP128: per 128-element run, s = max(amax, 1e-10f) / 127,
 *                 codes = trunc(clamp(x / s, -128, 127)) (the Triton cast truncates toward zero)
 * with amax = max|x| of the run. The chunk scale d is the scheme's s, so q * d is exactly the value
 * the host fake-quantization (gguf.quant.ActFakeQuant) produces, and the GEMM recovers
 * sum_k W[r][k] * x'[k] from the integer stage dots. A row whose amax is 0 stages zero codes and
 * d = 0, as the fake-quantization gives +0. The 16-element code sums (the weight-offset term) are the
 * sums of the staged codes.
 *
 * Launch: Int8Static is 1-D with one work-item per (token, chunk), as LLM_Quant_MMQ_Q8_1; Int8Row is
 * 1-D with one workgroup per token row and Int8Group128 with one workgroup per 128-element run, both
 * reducing the run's amax with nf_wg_max (local size L, a power of two <= LLM_DIMX). tk divides K and
 * 16; K is a multiple of 128 for Int8Group128, as the scheme's run length requires.
 */

/* Code of one element: sat8(rne(v)) for mode 0, trunc(clamp(v, -128, 127)) for mode 1, sat8(rne(v))
 * for mode 2, zero codes for mode 3 (a run whose scale is 0) and sat8(rne(v) + azp) for mode 4, the
 * asymmetric form (the rounding precedes the zero-point shift and the saturation, as gguf.quant.ActFakeQuant
 * computes it). */
NF_INLINE int nf_q8_code(float v, int mode, int azp) {
    if (mode == 3) return 0;
    if (mode == 4) {
        return (int)clamp((float)nf_rint_i(v) + (float)azp, -128.0f, 127.0f);
    }
    return mode == 1 ? (int)clamp(v, -128.0f, 127.0f) : nf_rint_i(clamp(v, -128.0f, 127.0f));
}

/*
 * Writes the record of one chunk (tk codes from xp, the chunk scale s) at 32-bit unit rec: tk / 4
 * packed code ints (element depth k in byte k % 4, two's complement), s, the tk / 16 code sums and
 * the zero point. mul is the reciprocal multiplier, mode selects the code form of nf_q8_code and azp
 * its asymmetric zero point (the value the record carries and the GEMM subtracts per weight row).
 */
NF_INLINE void nf_q8_store(NF_GLOBAL uint* q8, nf_i64 rec, NF_GLOBAL const float* xp, float s, float mul,
                           int tk, int mode, int azp) {
    int sums[NF_Q8_MAX_TK / 16];
    for (int h = 0; h < tk / 16; h++) {
        sums[h] = 0;
    }
    for (int u = 0; u < tk / 4; u++) {
        uint packed = 0;
        for (int b = 0; b < 4; b++) {
            const int k = 4 * u + b;
            const int q = nf_q8_code(mode == 0 ? xp[k] / s : xp[k] * mul, mode, azp);
            packed |= (uint)(q & 0xFF) << (8 * b);
            sums[k >> 4] += q;
        }
        q8[rec + u] = packed;
    }
    q8[rec + tk / 4] = nf_q8_float_rec(s);
    for (int h = 0; h < tk / 16; h++) {
        q8[rec + tk / 4 + 1 + h] = (uint)sums[h];
    }
    q8[rec + tk / 4 + 1 + tk / 16] = (uint)azp;
}

NF_KERNEL void LLM_Quant_MMQ_Int8Static(
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
    nf_q8_store(q8, i * (nf_i64)NF_Q8_REC(tk), row, scale, 0.0f, tk, scale == 0.0f ? 3 : 0, 0);
}

NF_KERNEL void LLM_Quant_MMQ_Int8Row(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off,
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
    const float amax = nf_wg_max(red, m);
    const float s = amax / 127.0f;
    const float mul = amax == 0.0f ? 0.0f : 127.0f / amax;
    const int chunks = K / tk;
    for (int c = lid; c < chunks; c += lsize) {
        const nf_i64 rec = ((nf_i64)t * chunks + c) * NF_Q8_REC(tk);
        nf_q8_store(q8, rec, row + (nf_i64)c * tk, s, mul, tk, amax == 0.0f ? 3 : 2, 0);
    }
}

NF_KERNEL void LLM_Quant_MMQ_Int8Group128(
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
    const float amax = nf_wg_max(red, m);
    const float s = fmax(amax, as_float(NF_AQ_GROUP_EPS_BITS)) / 127.0f;
    const float mul = 1.0f / s;
    const int chunks = K / tk;
    const int run_chunks = 128 / tk;
    for (int j = lid; j < run_chunks; j += lsize) {
        const nf_i64 rec = ((nf_i64)t * chunks + r * run_chunks + j) * NF_Q8_REC(tk);
        nf_q8_store(q8, rec, run_x + (nf_i64)j * tk, s, mul, tk, 1, 0);
    }
}

/*
 * LLM_Quant_MMQ_Int8Asym(i N, i K, i tk, i x_stride, i x_off, f scale, i zp, R float x, W uint q8)
 *
 * The asymmetric static int8 scheme (ActQuantScheme#INT8_STATIC_ASYM): per chunk, s = scale and
 * codes = sat8(rne(x * (1 / s)) + zp), the record carries s, the code sums and the zero point zp. The
 * value of a staged code is exactly the host fake-quantization's (q - zp) * s, and the int8 GEMM
 * subtracts the zero-point term zp * sum_k W[r][k] from every stage dot (the weight code sums the
 * kernel reduces per stage), so the product is computed in int8 rather than from fake-quantized
 * floats. A scale of 0 stages zero codes and d = 0.
 *
 * Launch: as LLM_Quant_MMQ_Int8Static (one work-item per (token, chunk)).
 */
NF_KERNEL void LLM_Quant_MMQ_Int8Asym(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off, const float scale, const int zp,
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
    nf_q8_store(q8, i * (nf_i64)NF_Q8_REC(tk), row, scale, inv, tk, scale == 0.0f ? 3 : 4, zp);
}
