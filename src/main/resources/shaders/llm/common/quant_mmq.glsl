/* common/quant_mmq.glsl - per-type MMQ decoders of the llama.cpp MMQ weight types, shared by the
 * int8 MMQ GEMM (matmul_mmq.comp) and the quantized-weight cooperative-matrix GEMM
 * (matmul_mmq_cm.comp); the GLSL twins of the per-type macros of kernels/llm/matmul_mmq.cl.
 * Requires: common/nf_common.glsl, common/quant_blocks.glsl, common/quant_legacy.glsl,
 * common/quant_k.glsl; the consumer defines QTYPE (QT_<T>, ggml_type_ids) and declares the weight
 * views (NF_BUFFER_VIEWS) before including.
 *
 * Every QTYPE defines MMQ_BS / MMQ_BB (block size and bytes of one weight block) and
 * mmq_code(ws, bp, e) / mmq_scale(ws, bp, k0) / mmq_off(ws, bp, k0): the integer code of element
 * block-relative e and the per-sub-block scale and offset of the stage starting at k0
 * (value = scale * code + offset), following the field offsets and value formulas of the decode
 * triples of kernels/llm/common/quant_decode_legacy.h and quant_decode_k.h.
 */

/* ---- round to nearest even, as OpenCL convert_int_rte / CUDA __float2int_rn ---- */
int nf_mmq_rint(float x) {
    /* The magic-number round of |x| < 2^21: adding 1.5 * 2^23 snaps to the integer scale with the
     * IEEE default (nearest even) rounding. */
    const float s = x < 0.0 ? -1.0 : 1.0;
    float v = abs(x) + 12582912.0;
    return int(s * (v - 12582912.0));
}

/* ---- per-type MMQ_BS / MMQ_BB and code, per-stage scale and offset (value = scale * code + offset) ---- */
#if QTYPE == QT_Q4_0
#define MMQ_BS 32
#define MMQ_BB 18
int mmq_code(uint ws, uint bp, uint e) {
    const uint b = nf_w_u8(ws, bp + 2u + (e & 15u));
    return int((e & 16u) == 0u ? (b & 0xFu) : (b >> 4u));
}
float mmq_scale(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp); }
float mmq_off(uint ws, uint bp, uint k0) { return -8.0 * nf_w_f16(ws, bp); }

#elif QTYPE == QT_Q4_1
#define MMQ_BS 32
#define MMQ_BB 20
int mmq_code(uint ws, uint bp, uint e) {
    const uint b = nf_w_u8(ws, bp + 4u + (e & 15u));
    return int((e & 16u) == 0u ? (b & 0xFu) : (b >> 4u));
}
float mmq_scale(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp); }
float mmq_off(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp + 2u); }

#elif QTYPE == QT_Q5_0
#define MMQ_BS 32
#define MMQ_BB 22
int mmq_code(uint ws, uint bp, uint e) {
    const uint qh = nf_w_u32_any(ws, bp + 2u);
    const uint b = nf_w_u8(ws, bp + 6u + (e & 15u));
    const uint v = ((e & 16u) == 0u ? (b & 0xFu) : (b >> 4u)) | (((qh >> e) & 1u) << 4);
    return int(v);
}
float mmq_scale(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp); }
float mmq_off(uint ws, uint bp, uint k0) { return -16.0 * nf_w_f16(ws, bp); }

#elif QTYPE == QT_Q5_1
#define MMQ_BS 32
#define MMQ_BB 24
int mmq_code(uint ws, uint bp, uint e) {
    const uint qh = nf_w_u32_any(ws, bp + 4u);
    const uint b = nf_w_u8(ws, bp + 8u + (e & 15u));
    const uint v = ((e & 16u) == 0u ? (b & 0xFu) : (b >> 4u)) | (((qh >> e) & 1u) << 4);
    return int(v);
}
float mmq_scale(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp); }
float mmq_off(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp + 2u); }

#elif QTYPE == QT_Q8_0
#define MMQ_BS 32
#define MMQ_BB 34
int mmq_code(uint ws, uint bp, uint e) { return nf_w_i8(ws, bp + 2u + e); }
float mmq_scale(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp); }
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

#elif QTYPE == QT_Q2_K
#define MMQ_BS 256
#define MMQ_BB 84
/* scales[16]@0 (low nibble scale, high nibble min), qs[64]@16 (four 2-bit planes), d@80, dmin@82. */
int mmq_code(uint ws, uint bp, uint e) {
    const uint q = nf_w_u8(ws, bp + 16u + ((e >> 7) << 5) + (e & 16u) + (e & 15u));
    return int((q >> (((e >> 5) & 3u) << 1)) & 3u);
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint sc = nf_w_u8(ws, bp + ((k0 % 256) >> 4));
    return nf_w_f16(ws, bp + 80u) * float(sc & 0xFu);
}
float mmq_off(uint ws, uint bp, uint k0) {
    const uint sc = nf_w_u8(ws, bp + ((k0 % 256) >> 4));
    return -nf_w_f16(ws, bp + 82u) * float(sc >> 4);
}

#elif QTYPE == QT_Q3_K
#define MMQ_BS 256
#define MMQ_BB 110
/* hmask[32]@0 (plane = bit index), qs[64]@32 (four 2-bit planes), scales[12]@96, d@108. */
int mmq_code(uint ws, uint bp, uint e) {
    const uint hf = e >> 7, j = (e >> 5) & 3u, h = (e & 16u) >> 4, l = e & 15u;
    const uint q = nf_w_u8(ws, bp + 32u + (hf << 5) + (h << 4) + l);
    const uint hm = nf_w_u8(ws, bp + (h << 4) + l);
    int v = int((q >> (j << 1)) & 3u);
    return ((hm >> ((hf << 2) + j)) & 1u) != 0u ? v : v - 4;
}
float mmq_scale(uint ws, uint bp, uint k0) {
    return nf_w_f16(ws, bp + 108u) * float(nf_q3k_scale(ws, bp, (k0 % 256) >> 4));
}
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

#elif QTYPE == QT_Q4_K
#define MMQ_BS 256
#define MMQ_BB 144
/* d@0, dmin@2, scales[12]@4 (get_scale_min_k4), qs[128]@16; group si = e >> 5 (32 elements). */
int mmq_code(uint ws, uint bp, uint e) {
    const uint si = e >> 5, l = e & 31u;
    const uint b = nf_w_u8(ws, bp + 16u + ((si >> 1) << 5) + l);
    return int((b >> ((si & 1u) << 2)) & 0xFu);
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint si = (k0 % 256) >> 5;
    return nf_w_f16(ws, bp) * nf_k4_scale_min(ws, bp + 4u, si).x;
}
float mmq_off(uint ws, uint bp, uint k0) {
    const uint si = (k0 % 256) >> 5;
    return -nf_w_f16(ws, bp + 2u) * nf_k4_scale_min(ws, bp + 4u, si).y;
}

#elif QTYPE == QT_Q5_K
#define MMQ_BS 256
#define MMQ_BB 176
/* As Q4_K plus a fifth bit of element e from bit si of qh[e % 32] (qh[32]@16, qs[128]@48). */
int mmq_code(uint ws, uint bp, uint e) {
    const uint si = e >> 5, l = e & 31u;
    const uint b = nf_w_u8(ws, bp + 48u + ((si >> 1) << 5) + l);
    const uint hb = (nf_w_u8(ws, bp + 16u + l) >> si) & 1u;
    return int(((b >> ((si & 1u) << 2)) & 0xFu) | (hb << 4));
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint si = (k0 % 256) >> 5;
    return nf_w_f16(ws, bp) * nf_k4_scale_min(ws, bp + 4u, si).x;
}
float mmq_off(uint ws, uint bp, uint k0) {
    const uint si = (k0 % 256) >> 5;
    return -nf_w_f16(ws, bp + 2u) * nf_k4_scale_min(ws, bp + 4u, si).y;
}

#elif QTYPE == QT_Q6_K
#define MMQ_BS 256
#define MMQ_BB 210
/* ql[128]@0 (low 4 bits), qh[64]@128 (high 2 bits), scales[16] i8@192, d@208. */
int mmq_code(uint ws, uint bp, uint e) {
    const uint hf = e >> 7, t = (e >> 5) & 3u, l = e & 31u;
    const uint lo = nf_w_u8(ws, bp + (hf << 6) + ((t & 1u) << 5) + l) >> ((t >> 1) << 2);
    const uint hi = (nf_w_u8(ws, bp + 128u + (hf << 5) + l) >> (t << 1)) & 3u;
    return int((lo & 0xFu) | (hi << 4)) - 32;
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint k = k0 % 256;
    const uint si = ((k >> 7) << 3) + (((k >> 5) & 3u) << 1) + ((k & 31u) >> 4);
    return nf_w_f16(ws, bp + 208u) * float(nf_w_i8(ws, bp + 192u + si));
}
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* ---- i-quants (the group builders of common/quant_iq.glsl, as per-element codes); the consumer
 * calls nf_iq_init_shared() before any return, from every invocation ---- */

/* IQ3_XXS: d(f16)@0, qs[64] grid indices@2, 8 x u32 scale+sign words@66; value = dl * mag * sign,
 * dl per 32-element sub-block. */
#elif QTYPE == QT_IQ3_XXS
#define MMQ_BS 256
#define MMQ_BB 98
int mmq_code(uint ws, uint bp, uint e) {
    const uint sb = e >> 5u, grp = (e >> 3u) & 3u, j = e & 7u;
    /* the group's two grid words hold elements 0-3 and 4-7: byte j & 3 of word (j >> 2) */
    const uint word = nf_iq3xxs(nf_w_u8(ws, bp + 2u + 8u * sb + 2u * grp + (j >> 2)));
    const uint signs = nf_iq2_ksigns((nf_w_u32_any(ws, bp + 66u + 4u * sb) >> (7u * grp)) & 127u);
    const int mag = int(bitfieldExtract(word, int(8u * (j & 3u)), 8));
    return mag * (1 - 2 * int(bitfieldExtract(signs, int(j), 1)));
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint aux = nf_w_u32_any(ws, bp + 66u + 4u * ((k0 % 256u) >> 5u));
    return nf_w_f16(ws, bp) * (0.5 + float(aux >> 28u)) * 0.5;
}
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* IQ3_S: d(f16)@0, qs[64]@2, qh[8]@66 (index bit 8), signs[32]@74, scales[4]@106; per sub-block a
 * shared scale, per 8-element group its sign byte. */
#elif QTYPE == QT_IQ3_S
#define MMQ_BS 256
#define MMQ_BB 110
int mmq_code(uint ws, uint bp, uint e) {
    const uint sb = e >> 5u, grp = (e >> 3u) & 3u, j = e & 7u;
    const uint qh = nf_w_u8(ws, bp + 66u + sb);
    const uint idx = nf_w_u8(ws, bp + 2u + 8u * sb + 2u * grp + (j >> 2))
        | ((qh << ((j & 4u) != 0u ? (7u - 2u * grp) : (8u - 2u * grp))) & 256u);
    const uint word = nf_iq3s(idx);
    /* the group's two grid words hold elements 0-3 and 4-7: byte j & 3 of word (j >> 2) */
    const int mag = int(bitfieldExtract(word, int(8u * (j & 3u)), 8));
    const uint signs = nf_w_u8(ws, bp + 74u + 4u * sb + grp);
    return mag * (1 - 2 * int(bitfieldExtract(signs, int(j), 1)));
}
float mmq_scale(uint ws, uint bp, uint k0) { return nf_iq3s_scale(ws, bp, (k0 % 256u) >> 5u); }
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* IQ2_XXS: d(f16)@0, 8 sub-blocks of 8 bytes: 4 grid indices and a u32 with the scale (bits 28-31)
 * and the sign indexes; dl per sub-block. */
#elif QTYPE == QT_IQ2_XXS
#define MMQ_BS 256
#define MMQ_BB 66
int mmq_code(uint ws, uint bp, uint e) {
    const uint sb = e >> 5u, grp = (e >> 3u) & 3u, j = e & 7u;
    const uvec2 word = nf_iq2xxs(nf_w_u8(ws, bp + 2u + 8u * sb + grp));
    const uint aux = nf_w_u32_any(ws, bp + 2u + 8u * sb + 4u);
    const uint signs = nf_iq2_ksigns((aux >> (7u * grp)) & 127u);
    return int(nf_iq_grid_byte(word, j)) * (1 - 2 * int(bitfieldExtract(signs, int(j), 1)));
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint aux = nf_w_u32_any(ws, bp + 2u + 8u * ((k0 % 256u) >> 5u) + 4u);
    return nf_w_f16(ws, bp) * (0.5 + float(aux >> 28u)) * 0.25;
}
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* IQ2_XS: d(f16)@0, qs[32] u16 (bits 0-8 grid index, bits 9-15 sign index)@2, scales[8]@66;
 * groups 0-1 of a sub-block use the low scale nibble, groups 2-3 the high one. */
#elif QTYPE == QT_IQ2_XS
#define MMQ_BS 256
#define MMQ_BB 74
int mmq_code(uint ws, uint bp, uint e) {
    const uint sb = e >> 5u, grp = (e >> 3u) & 3u, j = e & 7u;
    const uint qv = nf_w_u16(ws, bp + 2u + 2u * (4u * sb + grp));
    const uvec2 word = nf_iq2xs(qv & 511u);
    const uint signs = nf_iq2_ksigns(qv >> 9);
    return int(nf_iq_grid_byte(word, j)) * (1 - 2 * int(bitfieldExtract(signs, int(j), 1)));
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint sc = nf_w_u8(ws, bp + 66u + ((k0 % 256u) >> 5u));
    const uint nib = ((k0 % 256u) & 16u) != 0u ? (sc >> 4u) : (sc & 15u);
    return nf_w_f16(ws, bp) * (0.5 + float(nib)) * 0.25;
}
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* IQ2_S: d(f16)@0, qs[32] grid indices@2, signs[32]@34, qh[8]@66 (index bits 8-9), scales[8]@74;
 * scale nibbles as IQ2_XS. */
#elif QTYPE == QT_IQ2_S
#define MMQ_BS 256
#define MMQ_BB 82
int mmq_code(uint ws, uint bp, uint e) {
    const uint sb = e >> 5u, grp = (e >> 3u) & 3u, j = e & 7u;
    const uint qh = nf_w_u8(ws, bp + 66u + sb);
    const uint idx = nf_w_u8(ws, bp + 2u + 4u * sb + grp) | ((qh << (8u - 2u * grp)) & 0x300u);
    const uvec2 word = nf_iq2s(idx);
    const uint signs = nf_w_u8(ws, bp + 34u + 4u * sb + grp);
    return int(nf_iq_grid_byte(word, j)) * (1 - 2 * int(bitfieldExtract(signs, int(j), 1)));
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint sc = nf_w_u8(ws, bp + 74u + ((k0 % 256u) >> 5u));
    const uint nib = ((k0 % 256u) & 16u) != 0u ? (sc >> 4u) : (sc & 15u);
    return nf_w_f16(ws, bp) * (0.5 + float(nib)) * 0.25;
}
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* IQ1_S: d(f16)@0, qs[32] grid index low bytes@2, qh[8] u16@34 (bits 3l..3l+2 the index bits 8-10
 * of group l, bits 12-14 the sub-block scale, bit 15 the delta sign); value = dl * (val + delta)
 * over the ternary grid, dl per sub-block. */
#elif QTYPE == QT_IQ1_S
#define MMQ_BS 256
#define MMQ_BB 50
const float NF_MMQ_IQ1_DELTA = 0.125;    /* IQ1S_DELTA (llama.cpp ggml-common.h) */
int mmq_code(uint ws, uint bp, uint e) {
    const uint sb = e >> 5u, grp = (e >> 3u) & 3u, j = e & 7u;
    const uint qh = nf_w_u16(ws, bp + 34u + 2u * sb);
    const uint idx = nf_w_u8(ws, bp + 2u + 4u * sb + grp) | (((qh >> (3u * grp)) & 7u) << 8);
    return nf_iq1s_val(idx, j);
}
float mmq_scale(uint ws, uint bp, uint k0) {
    const uint qh = nf_w_u16(ws, bp + 34u + 2u * ((k0 % 256u) >> 5u));
    return nf_w_f16(ws, bp) * float(2u * ((qh >> 12u) & 7u) + 1u);
}
float mmq_off(uint ws, uint bp, uint k0) {
    const uint qh = nf_w_u16(ws, bp + 34u + 2u * ((k0 % 256u) >> 5u));
    return mmq_scale(ws, bp, k0) * (((qh & 0x8000u) != 0u) ? -NF_MMQ_IQ1_DELTA : NF_MMQ_IQ1_DELTA);
}

/* IQ4_NL: d(f16)@0, qs[16]@2; value = d * kvalues_iq4nl[nibble]. */
#elif QTYPE == QT_IQ4_NL
#define MMQ_BS 32
#define MMQ_BB 18
int mmq_code(uint ws, uint bp, uint e) {
    const uint q = nf_w_u8(ws, bp + 2u + (e & 15u));
    return nf_kvalue_iq4nl((e & 16u) == 0u ? (q & 15u) : (q >> 4u));
}
float mmq_scale(uint ws, uint bp, uint k0) { return nf_w_f16(ws, bp); }
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* IQ4_XS: d(f16)@0, scales_h/l@2 (nf_iq4xs_sub_scale), qs[128]@8; 32-element sub-blocks carry the
 * scale. */
#elif QTYPE == QT_IQ4_XS
#define MMQ_BS 256
#define MMQ_BB 136
int mmq_code(uint ws, uint bp, uint e) {
    const uint sb = e >> 5u;
    const uint q = nf_w_u8(ws, bp + 8u + 16u * sb + (e & 15u));
    return nf_kvalue_iq4nl((e & 16u) == 0u ? (q & 15u) : (q >> 4u));
}
float mmq_scale(uint ws, uint bp, uint k0) { return nf_iq4xs_sub_scale(ws, bp, (k0 % 256u) >> 5u); }
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

/* I8_B128: d(f32)@0, qs[128](int8)@4; value = d * qs. The scale covers the whole 128-element block. */
#elif QTYPE == QT_I8_B128
#define MMQ_BS 128
#define MMQ_BB 132
int mmq_code(uint ws, uint bp, uint e) { return nf_w_i8(ws, bp + 4u + e); }
float mmq_scale(uint ws, uint bp, uint k0) { return nf_w_f32(ws, bp); }
float mmq_off(uint ws, uint bp, uint k0) { return 0.0; }

#else
#error "quant_mmq.glsl: QTYPE is not an MMQ weight type"
#endif
