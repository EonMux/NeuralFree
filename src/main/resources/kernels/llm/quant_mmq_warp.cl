/* quant_mmq_warp.cl - LLM_Quant_MMQ_Q8_1_Warp, the warp-cooperative form of the MMQ activation staging
 * ([cuda]: it reduces each record across lanes with warp shuffles).
 * Requires: common/dialect.h, common/config.h, common/matmul_mmq_config.h */
/*
 * LLM_Quant_MMQ_Q8_1_Warp(i N, i K, i tk, i x_stride, i x_off, R float x, W uint q8)
 *   writes exactly the records of LLM_Quant_MMQ_Q8_1 (kernels/llm/quant_mmq_q8_1.cl): for every token
 *   t < N and chunk c < K / tk, tk / 4 code ints, the chunk scale d, the tk / 16 code sums of the
 *   16-element halves and the zero point NF_Q8_NO_ZERO_POINT, at unit ((t * (K / tk) + c) * NF_Q8_REC(tk)).
 *
 * The per-record kernel gives each work-item one whole record: 32 serial float loads and 12 scattered
 * 4-byte stores whose lanes sit 48 bytes apart, so a warp's stores touch 32 sectors per instruction. Here
 * tk / 4 consecutive lanes share a record: each loads one float4 of the row (a warp reads 512 contiguous
 * bytes), the chunk's maximum magnitude and the half code sums are reduced with xor shuffles inside the
 * lane group, and lane l stores code int l, so a record's units are written by consecutive lanes.
 * The arithmetic is the per-record kernel's, value for value: the magnitude maximum is order-free
 * (fmax over non-negative values ignores a NaN exactly as the serial `a > maxabs` does), the reciprocal
 * scale and every rounding are the same expressions and the code sums are integers, so both kernels
 * write identical records (CudaQuantMmqWarpParityTest).
 *
 * Launch contract: 1-D, global = roundUp(N * K / 4, L) work-items of local size L, L a multiple of 32;
 * tk is 16 or 32 and divides K; x_off and x_stride are multiples of 4 (the host keeps the per-record
 * kernel for any other view), so every float4 load is 16-byte aligned. Every lane of a warp runs the
 * shuffles; lanes past the last record load zeros and store nothing.
 */
NF_KERNEL void LLM_Quant_MMQ_Q8_1_Warp(
    const int N, const int K, const int tk,
    const int x_stride, const int x_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL uint* NF_RESTRICT q8) {
    const int row4 = K / 4;
    const nf_i64 i = (nf_i64)get_global_id(0);
    const bool live = i < (nf_i64)N * row4;
    const int t = live ? (int)(i / row4) : 0;
    const int u = live ? (int)(i - (nf_i64)t * row4) : 0;
    const int lanes = tk / 4;
    const int l = u & (lanes - 1);
    float4 v = NF_F4S(0.0f);
    if (live) {
        v = *(NF_GLOBAL const float4*)(x + (nf_i64)x_off + (nf_i64)t * x_stride + 4 * (nf_i64)u);
    }
    float maxabs = 0.0f;
    maxabs = fmax(maxabs, fabs(v.x));
    maxabs = fmax(maxabs, fabs(v.y));
    maxabs = fmax(maxabs, fabs(v.z));
    maxabs = fmax(maxabs, fabs(v.w));
    for (int o = lanes >> 1; o > 0; o >>= 1) {
        maxabs = fmax(maxabs, nf_sg_shuffle_xor(maxabs, o));
    }
    const float id = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
    const float d = maxabs > 0.0f ? maxabs * (1.0f / 127.0f) : 1.0f;
    const int q0 = nf_rint_i(v.x * id), q1 = nf_rint_i(v.y * id);
    const int q2 = nf_rint_i(v.z * id), q3 = nf_rint_i(v.w * id);
    /* the four lanes of a 16-element half sum their codes */
    int sum = q0 + q1 + q2 + q3;
    sum += nf_sg_shuffle_xor_i(sum, 1);
    sum += nf_sg_shuffle_xor_i(sum, 2);
    if (!live) {
        return;
    }
    const int chunks = K / tk;
    const int c = u / lanes;
    NF_GLOBAL uint* rec = q8 + ((nf_i64)t * chunks + c) * NF_Q8_REC(tk);
    rec[l] = ((uint)q0 & 0xFFu) | (((uint)q1 & 0xFFu) << 8) | (((uint)q2 & 0xFFu) << 16) | ((uint)q3 << 24);
    if (l == 0) {
        rec[tk / 4] = nf_q8_float_rec(d);
        rec[tk / 4 + 1 + tk / 16] = NF_Q8_NO_ZERO_POINT;
    }
    if ((l & 3) == 0) {
        rec[tk / 4 + 1 + (l >> 2)] = (uint)sum;
    }
}
