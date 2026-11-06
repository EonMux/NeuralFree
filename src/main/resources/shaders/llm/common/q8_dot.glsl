/* common/q8_dot.glsl - Q8_1 activation-block readers and packed int8 helpers of the integer-dot matvec
 * family. Requires: common/nf_common.glsl, common/quant_blocks.glsl (through the including shader),
 * GL_EXT_integer_dot_product enabled by that shader before common/nf_common.glsl.
 *
 * The integer-dot matvec shader (matvec_q8.comp) declares the Q8_1 activation row as the aliased
 * byte views xq_u8/xq_u16/xq_u32 of one storage binding before including this file, so the readers
 * below address a block by its 36-byte offset. A weight block contributes
 * d8 * sum_k(w_code_k * q8_k) * scale: the code products are packed int8 dot products
 * (dotPacked4x8EXT, the Vulkan dp4a), never float weight values.
 */
#ifndef NF_Q8_DOT_GLSL
#define NF_Q8_DOT_GLSL

#ifndef NF_QUANT_BLOCKS_GLSL
#error "common/q8_dot.glsl requires common/quant_blocks.glsl"
#endif

const int NF_Q8_ONES = 0x01010101;   /* four bytes valued 1: nf_q8_sum4 sums the signed bytes of a word */

/* Sum of the four signed-byte products of the packed words a and b. */
int nf_q8_dp4(int a, int b) {
    return dotPacked4x8EXT(a, b);
}

/* Sum of the four signed bytes of a packed word. */
int nf_q8_sum4(int u) {
    return dotPacked4x8EXT(NF_Q8_ONES, u);
}

/* Four signed values packed into one word, value k in byte k (two's complement). */
int nf_q8_pack4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

/* Four FP4 codes packed as int8, nibble k in byte k: the doubled E2M1 values both MXFP4 and NVFP4 use. */
int nf_q8_fp4_word(uint nibbles) {
    return nf_q8_pack4(nf_kvalue_fp4(nibbles & 0xFu),
                       nf_kvalue_fp4((nibbles >> 8) & 0xFu),
                       nf_kvalue_fp4((nibbles >> 16) & 0xFu),
                       nf_kvalue_fp4((nibbles >> 24) & 0xFu));
}

/* Four unsigned magnitude bytes (byte k at bits 8k) with sign bit k applied, packed as int8 codes. */
int nf_q8_signed_mags(uint mag, uint signs) {
    int v0 = int(mag & 0xFFu);
    int v1 = int((mag >> 8) & 0xFFu);
    int v2 = int((mag >> 16) & 0xFFu);
    int v3 = int((mag >> 24) & 0xFFu);
    if ((signs & 1u) != 0u) v0 = -v0;
    if ((signs & 2u) != 0u) v1 = -v1;
    if ((signs & 4u) != 0u) v2 = -v2;
    if ((signs & 8u) != 0u) v3 = -v3;
    return nf_q8_pack4(v0, v1, v2, v3);
}

/* Four int8 codes at byte offset off (a multiple of 4) of Q8_1 block qblk. */
int nf_q8_word(uint qblk, uint off) {
    return int(xq_u32[qblk * 9u + ((4u + off) >> 2)]);
}

/* f16 scale d of Q8_1 block qblk. */
float nf_q8_scale_of(uint qblk) {
    return nf_half(uint(xq_u16[qblk * 18u]));
}

#endif /* NF_Q8_DOT_GLSL */
