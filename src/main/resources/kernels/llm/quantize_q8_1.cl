/* quantize_q8_1.cl - F32 activation row to Q8_1 blocks (LLM_QuantizeQ8_1). Requires: common/dialect.h, common/config.h */
/*
 * Q8_1 block (ggml block_q8_1): d (f16) at byte 0, s (f16) at byte 2, qs[32] (int8) at byte 4,
 * where s = d * sum(qs) (the sum of the block's codes). This is the activation form the integer-dot
 * matvec kernels (common/quant_decode_*_q8.h, NF_DP4A) consume, as llama.cpp quantize_row_q8_1_ref
 * (ggml-quants.c:302-335) does for the CUDA and Vulkan MMVQ path.
 *
 * LLM_QuantizeQ8_1 is a 1-D grid-stride loop over blocks, one block per work-item per step, so any
 * global size is valid (a single workgroup, global = local = L, suffices).
 */
NF_KERNEL void LLM_QuantizeQ8_1(
    const int size,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL uchar* NF_RESTRICT output)
{
    const int stride = get_global_size(0);
    const int blocks = size / Q8_1_BLOCK_SIZE;

    for (int b = get_global_id(0); b < blocks; b += stride) {
        NF_GLOBAL const float* x = input + (nf_i64)b * Q8_1_BLOCK_SIZE;
        NF_GLOBAL uchar* y = output + (nf_i64)b * Q8_1_BLOCK_BYTES;

        float amax = 0.0f;
        for (int j = 0; j < Q8_1_BLOCK_SIZE; j += 4) {
            const float4 v = fabs(vload4(0, x + j));
            amax = fmax(amax, fmax(fmax(v.x, v.y), fmax(v.z, v.w)));
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        vstore_half(d, 0, (NF_GLOBAL NF_HALF*)y);

        int sum = 0;
        for (int j = 0; j < Q8_1_BLOCK_SIZE; j++) {
            const int q = clamp((int)round(x[j] * id), -127, 127);
            y[Q8_1_QS_OFFSET + j] = (uchar)(q & 0xFF);
            sum += q;
        }
        vstore_half((float)sum * d, 1, (NF_GLOBAL NF_HALF*)y);
    }
}
