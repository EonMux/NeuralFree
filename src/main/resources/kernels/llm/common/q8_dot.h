/* common/q8_dot.h - Q8_1 activation-block readers and packed int8 helpers of the integer-dot matvec family. Requires: common/dialect.h, common/config.h, common/quant_blocks.h */
/*
 * The integer-dot matvec kernels replace the F32 activation row of the float matvec forms with one
 * Q8_1 row (ggml block_q8_1: d(f16) @0, s(f16) @2, qs[32] int8 @4). A weight block then contributes
 * the exact dot of its decoded values with the Q8_1-dequantized activations:
 *
 *   sum_k w_k * (d8 * q8_k)  =  d8 * sum_k w_k * q8_k
 *
 * where w_k itself is an integer code times a block scale (and, for the K-quants with a minimum, a
 * per-group minimum). The code products are packed int8 dot products (NF_DP4A), so no weight code is
 * ever converted to float. Q8_1 blocks are 32 elements, so the 32-element group of a K-quant or the
 * 32-element sub-block of an i-quant maps to exactly one Q8_1 block; the four work-items that share
 * an 8-element group read the four 8-byte slices of that block.
 *
 * The min/offset types (Q4_1, Q5_1, and the K-quants' dmin) add m * sum_k (d8 * q8_k) per group: each
 * work-item sums the codes it owns with NF_Q8_ONES, so the work-items of one block add up to the
 * block's whole s term.
 */
#ifndef NF_Q8_DOT_H
#define NF_Q8_DOT_H

#ifndef NF_QUANT_BLOCKS_H
#error "common/q8_dot.h requires common/quant_blocks.h"
#endif

/* Packed words of four bytes each valued 1: NF_DP4A with this word sums the signed bytes of a word. */
#define NF_Q8_ONES 0x01010101

/* f16 scale d of Q8_1 block qblk of the activation row at xq. */
NF_INLINE float nf_q8_scale(NF_GLOBAL const uchar* xq, int qblk) {
    return vload_half(0, (NF_GLOBAL const NF_HALF*)(xq + qblk * Q8_1_BLOCK_BYTES));
}

/* 32-bit word of four int8 codes at byte offset off of Q8_1 block qblk. A Q8_1 block is 36 bytes and
 * its qs start at offset 4, so both the block and off (a multiple of 4) keep the word four-byte aligned. */
NF_INLINE int nf_q8_word(NF_GLOBAL const uchar* xq, int qblk, int off) {
    return (int)nf_read_u32(xq + qblk * Q8_1_BLOCK_BYTES + Q8_1_QS_OFFSET + off);
}

/* Four signed values packed into one word, value k in byte k (two's complement). */
NF_INLINE int nf_q8_pack4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

/* Four FP4 codes packed as int8, nibble k in byte k: the doubled E2M1 values both MXFP4 and NVFP4 use. */
NF_INLINE int nf_q8_fp4_word(uint nibbles) {
    return nf_q8_pack4((int)NF_FP4_CODES[nibbles & 0xFu],
                       (int)NF_FP4_CODES[(nibbles >> 8) & 0xFu],
                       (int)NF_FP4_CODES[(nibbles >> 16) & 0xFu],
                       (int)NF_FP4_CODES[(nibbles >> 24) & 0xFu]);
}

/* Four unsigned magnitude bytes (byte k at bits 8k) with sign bit k applied, packed as int8 codes. */
NF_INLINE int nf_q8_signed_mags(uint mag, uint signs) {
    int v0 = (int)(mag & 0xFFu);
    int v1 = (int)((mag >> 8) & 0xFFu);
    int v2 = (int)((mag >> 16) & 0xFFu);
    int v3 = (int)((mag >> 24) & 0xFFu);
    if ((signs & 1u) != 0u) v0 = -v0;
    if ((signs & 2u) != 0u) v1 = -v1;
    if ((signs & 4u) != 0u) v2 = -v2;
    if ((signs & 8u) != 0u) v3 = -v3;
    return nf_q8_pack4(v0, v1, v2, v3);
}

#endif /* NF_Q8_DOT_H */
