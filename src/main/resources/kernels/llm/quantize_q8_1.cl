/* quantize_q8_1.cl - F32 activation row to Q8_1 blocks (LLM_QuantizeQ8_1). Requires: common/dialect.h, common/config.h */
/*
 * Q8_1 block (ggml block_q8_1): d (f16) at byte 0, s (f16) at byte 2, qs[32] (int8) at byte 4,
 * where s = d * sum(qs) (the sum of the block's codes). This is the activation form the integer-dot
 * matvec kernels (common/quant_decode_*_q8.h, NF_DP4A) consume, as llama.cpp quantize_row_q8_1_ref
 * (ggml-quants.c:302-335) does for the CUDA and Vulkan MMVQ path.
 *
 * LLM_QuantizeQ8_1 is a 1-D grid-stride loop over blocks, so any global size is valid (a single
 * workgroup, global = local = L, suffices; a multiple of 32 on the shuffle backends). Where the
 * backend has the subgroup shuffle (CUDA, HIP) a 32-lane segment quantizes one block per step, lane
 * j owning element j: one coalesced load, the block max and the code sum by butterfly, one coalesced
 * 32-byte store (the host launches 32 work-items per block there). Elsewhere one work-item quantizes
 * one block per step. Both forms compute the same bits: the max is order-free, and the code sum is
 * an integer.
 */
NF_KERNEL void LLM_QuantizeQ8_1(
    const int size,
    NF_GLOBAL const float* NF_RESTRICT input,
    NF_GLOBAL uchar* NF_RESTRICT output)
{
    const int blocks = size / Q8_1_BLOCK_SIZE;
#ifdef NF_HAS_SG_SHUFFLE
    const int lane = (int)get_global_id(0) & 31;
    const int warps = ((int)get_global_size(0) + 31) >> 5;
    for (int b = (int)get_global_id(0) >> 5; b < blocks; b += warps) {
        const float v = input[(nf_i64)b * Q8_1_BLOCK_SIZE + lane];
        float amax = fmax(0.0f, fabs(v));   /* a NaN element drops out of the max, as in the loop below */
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmax(amax, nf_sg_shuffle_xor(amax, o));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const int q = clamp((int)round(v * id), -127, 127);
        int sum = q;
        for (int o = 16; o > 0; o >>= 1) {
            sum += nf_sg_shuffle_xor_i(sum, o);
        }
        NF_GLOBAL uchar* y = output + (nf_i64)b * Q8_1_BLOCK_BYTES;
        y[Q8_1_QS_OFFSET + lane] = (uchar)(q & 0xFF);
        if (lane == 0) {
            vstore_half(d, 0, (NF_GLOBAL NF_HALF*)y);
            vstore_half((float)sum * d, 1, (NF_GLOBAL NF_HALF*)y);
        }
    }
#else
    const int stride = get_global_size(0);

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
#endif
}
