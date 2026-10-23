/* common/quant_decode_legacy.h - Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, Q8_0, MXFP4 and IQ4_XS decode triples (_decode, _decode_dual, _decode_elem). Requires: common/dialect.h, common/config.h, common/quant_blocks.h */
/*
 * Each triple follows the decoder contract of common/matvec_family.h:
 *   T_decode(bp, input, off, blk, ltid)       partial dot of block blk with input[off + blk*BS ...],
 *                                            over the elements work-item ltid in [0, TPS) owns
 *   T_decode_dual(gbp, ubp, input, off, blk, ltid, gs, us)
 *                                            T_decode of two weight rows sharing the input loads,
 *                                            accumulated into *gs and *us
 *   T_decode_elem(weight, dim, tok, i)       element i of row tok of a [rows x dim] weight
 *
 * Element order is the ggml SPLIT layout of llama.cpp ggml-quants.c:458-588 and 2726-2762: inside a
 * 32-element (sub-)block, byte k of qs holds element k in its low nibble and element k + 16 in its
 * high nibble; for Q5_0/Q5_1 the fifth bit of element e is bit e of the little-endian qh word.
 *
 * Work split of the 32-element types (TPS = 4): work-item ltid owns qs bytes k = 4*ltid .. 4*ltid+3,
 * i.e. elements k and k + 16. Q8_0 (TPS = 4) gives each work-item 8 consecutive elements.
 * IQ4_XS (TPS = 32) gives sub-block ltid/4 to four work-items with the same 4-byte split.
 */
#ifndef NF_QUANT_DECODE_LEGACY_H
#define NF_QUANT_DECODE_LEGACY_H

#define Q4_0_TPS   4
#define Q4_1_TPS   4
#define Q5_0_TPS   4
#define Q5_1_TPS   4
#define IQ4_NL_TPS 4
#define Q8_0_TPS   Q8_0_THREADS_PER_BLOCK
#define MXFP4_TPS  MXFP4_THREADS_PER_BLOCK
#define IQ4_XS_TPS IQ4_XS_THREADS_PER_SUPER

/* f16 at byte offset o of a block. */
NF_INLINE float nf_ql_half(NF_GLOBAL const uchar* bp, int o) {
    return vload_half(0, (NF_GLOBAL const NF_HALF*)(bp + o));
}

/* Fifth bit of element e as the 0x10 bit of a 5-bit quant (ggml xh_0 / xh_1). */
NF_INLINE uint nf_ql_q5_hbit(uint qh, int e) {
    return ((qh >> e) & 1u) << 4;
}

/* Start of block i/BS of row tok in a [rows x dim] weight with BB-byte blocks (64-bit offsets). */
NF_INLINE NF_GLOBAL const uchar* nf_ql_elem_block(NF_GLOBAL const uchar* weight, int dim, int tok,
                                                  int i, int bs, int bb) {
    const nf_i64 row_bytes = (nf_i64)(dim / bs) * bb;
    return weight + (nf_i64)tok * row_bytes + (nf_i64)(i / bs) * bb;
}

/* ================================ Q4_0 ================================
 * 18 B: d(f16)@0, qs[16]@2. y[k] = ((qs[k]&15) - 8)*d, y[k+16] = ((qs[k]>>4) - 8)*d. */
NF_INLINE float Q4_0_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                            int off, int blk, int ltid) {
    const float d = nf_ql_half(bp, 0);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q4_0_BLOCK_SIZE + bo;
    float sum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const uint q = bp[2 + bo + j];
        sum += (float)((int)(q & 0xFu) - 8) * x[j];
        sum += (float)((int)(q >> 4) - 8) * x[j + 16];
    }
    return sum * d;
}

NF_INLINE void Q4_0_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                NF_GLOBAL const float* input, int off, int blk, int ltid,
                                float* gs, float* us) {
    const float gd = nf_ql_half(gbp, 0), ud = nf_ql_half(ubp, 0);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q4_0_BLOCK_SIZE + bo;
    float gsum = 0.0f, usum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const float x0 = x[j], x1 = x[j + 16];
        const uint gq = gbp[2 + bo + j], uq = ubp[2 + bo + j];
        gsum += x0 * (float)((int)(gq & 0xFu) - 8) + x1 * (float)((int)(gq >> 4) - 8);
        usum += x0 * (float)((int)(uq & 0xFu) - 8) + x1 * (float)((int)(uq >> 4) - 8);
    }
    *gs += gsum * gd;
    *us += usum * ud;
}

NF_INLINE float Q4_0_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES);
    const int e = i % Q4_0_BLOCK_SIZE;
    const uint q = bp[2 + (e & 15)];
    const int v = (int)(e < 16 ? (q & 0xFu) : (q >> 4)) - 8;
    return nf_ql_half(bp, 0) * (float)v;
}

/* ================================ Q4_1 ================================
 * 20 B: d(f16)@0, m(f16)@2, qs[16]@4. y[k] = (qs[k]&15)*d + m, y[k+16] = (qs[k]>>4)*d + m
 * (llama.cpp ggml-quants.c:479-498). The dot is d*sum(q*x) + m*sum(x) over the owned elements. */
NF_INLINE float Q4_1_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                            int off, int blk, int ltid) {
    const float d = nf_ql_half(bp, 0), m = nf_ql_half(bp, 2);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q4_1_BLOCK_SIZE + bo;
    float qsum = 0.0f, xsum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const float x0 = x[j], x1 = x[j + 16];
        const uint q = bp[4 + bo + j];
        qsum += (float)(q & 0xFu) * x0 + (float)(q >> 4) * x1;
        xsum += x0 + x1;
    }
    return d * qsum + m * xsum;
}

NF_INLINE void Q4_1_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                NF_GLOBAL const float* input, int off, int blk, int ltid,
                                float* gs, float* us) {
    const float gd = nf_ql_half(gbp, 0), gm = nf_ql_half(gbp, 2);
    const float ud = nf_ql_half(ubp, 0), um = nf_ql_half(ubp, 2);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q4_1_BLOCK_SIZE + bo;
    float gq = 0.0f, uq = 0.0f, xsum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const float x0 = x[j], x1 = x[j + 16];
        const uint gb = gbp[4 + bo + j], ub = ubp[4 + bo + j];
        gq += (float)(gb & 0xFu) * x0 + (float)(gb >> 4) * x1;
        uq += (float)(ub & 0xFu) * x0 + (float)(ub >> 4) * x1;
        xsum += x0 + x1;
    }
    *gs += gd * gq + gm * xsum;
    *us += ud * uq + um * xsum;
}

NF_INLINE float Q4_1_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES);
    const int e = i % Q4_1_BLOCK_SIZE;
    const uint q = bp[4 + (e & 15)];
    const uint v = e < 16 ? (q & 0xFu) : (q >> 4);
    return nf_ql_half(bp, 0) * (float)v + nf_ql_half(bp, 2);
}

/* ================================ Q5_0 ================================
 * 22 B: d(f16)@0, qh(u32 LE)@2, qs[16]@6.
 * y[k] = (((qs[k]&15) | hbit(k)) - 16)*d, y[k+16] = (((qs[k]>>4) | hbit(k+16)) - 16)*d
 * (llama.cpp ggml-quants.c:500-523; hbit(e) = ((qh >> e) & 1) << 4). */
NF_INLINE float Q5_0_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                            int off, int blk, int ltid) {
    const float d = nf_ql_half(bp, 0);
    const uint qh = read_u32_le(bp + 2);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q5_0_BLOCK_SIZE + bo;
    float sum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const int k = bo + j;
        const uint q = bp[6 + k];
        const int lo = (int)((q & 0xFu) | nf_ql_q5_hbit(qh, k)) - 16;
        const int hi = (int)((q >> 4) | nf_ql_q5_hbit(qh, k + 16)) - 16;
        sum += (float)lo * x[j] + (float)hi * x[j + 16];
    }
    return sum * d;
}

NF_INLINE void Q5_0_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                NF_GLOBAL const float* input, int off, int blk, int ltid,
                                float* gs, float* us) {
    const float gd = nf_ql_half(gbp, 0), ud = nf_ql_half(ubp, 0);
    const uint gqh = read_u32_le(gbp + 2), uqh = read_u32_le(ubp + 2);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q5_0_BLOCK_SIZE + bo;
    float gsum = 0.0f, usum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const int k = bo + j;
        const float x0 = x[j], x1 = x[j + 16];
        const uint gq = gbp[6 + k], uq = ubp[6 + k];
        const int glo = (int)((gq & 0xFu) | nf_ql_q5_hbit(gqh, k)) - 16;
        const int ghi = (int)((gq >> 4) | nf_ql_q5_hbit(gqh, k + 16)) - 16;
        const int ulo = (int)((uq & 0xFu) | nf_ql_q5_hbit(uqh, k)) - 16;
        const int uhi = (int)((uq >> 4) | nf_ql_q5_hbit(uqh, k + 16)) - 16;
        gsum += (float)glo * x0 + (float)ghi * x1;
        usum += (float)ulo * x0 + (float)uhi * x1;
    }
    *gs += gsum * gd;
    *us += usum * ud;
}

NF_INLINE float Q5_0_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES);
    const int e = i % Q5_0_BLOCK_SIZE;
    const uint qh = read_u32_le(bp + 2);
    const uint q = bp[6 + (e & 15)];
    const int v = (int)((e < 16 ? (q & 0xFu) : (q >> 4)) | nf_ql_q5_hbit(qh, e)) - 16;
    return nf_ql_half(bp, 0) * (float)v;
}

/* ================================ Q5_1 ================================
 * 24 B: d(f16)@0, m(f16)@2, qh(u32 LE)@4, qs[16]@8.
 * y[k] = ((qs[k]&15) | hbit(k))*d + m, y[k+16] = ((qs[k]>>4) | hbit(k+16))*d + m
 * (llama.cpp ggml-quants.c:525-548). */
NF_INLINE float Q5_1_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                            int off, int blk, int ltid) {
    const float d = nf_ql_half(bp, 0), m = nf_ql_half(bp, 2);
    const uint qh = read_u32_le(bp + 4);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q5_1_BLOCK_SIZE + bo;
    float qsum = 0.0f, xsum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const int k = bo + j;
        const float x0 = x[j], x1 = x[j + 16];
        const uint q = bp[8 + k];
        const uint lo = (q & 0xFu) | nf_ql_q5_hbit(qh, k);
        const uint hi = (q >> 4) | nf_ql_q5_hbit(qh, k + 16);
        qsum += (float)lo * x0 + (float)hi * x1;
        xsum += x0 + x1;
    }
    return d * qsum + m * xsum;
}

NF_INLINE void Q5_1_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                NF_GLOBAL const float* input, int off, int blk, int ltid,
                                float* gs, float* us) {
    const float gd = nf_ql_half(gbp, 0), gm = nf_ql_half(gbp, 2);
    const float ud = nf_ql_half(ubp, 0), um = nf_ql_half(ubp, 2);
    const uint gqh = read_u32_le(gbp + 4), uqh = read_u32_le(ubp + 4);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * Q5_1_BLOCK_SIZE + bo;
    float gsum = 0.0f, usum = 0.0f, xsum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const int k = bo + j;
        const float x0 = x[j], x1 = x[j + 16];
        const uint gq = gbp[8 + k], uq = ubp[8 + k];
        const uint glo = (gq & 0xFu) | nf_ql_q5_hbit(gqh, k);
        const uint ghi = (gq >> 4) | nf_ql_q5_hbit(gqh, k + 16);
        const uint ulo = (uq & 0xFu) | nf_ql_q5_hbit(uqh, k);
        const uint uhi = (uq >> 4) | nf_ql_q5_hbit(uqh, k + 16);
        gsum += (float)glo * x0 + (float)ghi * x1;
        usum += (float)ulo * x0 + (float)uhi * x1;
        xsum += x0 + x1;
    }
    *gs += gd * gsum + gm * xsum;
    *us += ud * usum + um * xsum;
}

NF_INLINE float Q5_1_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES);
    const int e = i % Q5_1_BLOCK_SIZE;
    const uint qh = read_u32_le(bp + 4);
    const uint q = bp[8 + (e & 15)];
    const uint v = (e < 16 ? (q & 0xFu) : (q >> 4)) | nf_ql_q5_hbit(qh, e);
    return nf_ql_half(bp, 0) * (float)v + nf_ql_half(bp, 2);
}

/* =============================== IQ4_NL ===============================
 * 18 B: d(f16)@0, qs[16]@2. y[k] = d*kv[qs[k]&15], y[k+16] = d*kv[qs[k]>>4]
 * (llama.cpp ggml-quants.c:2726-2742, kv = kvalues_iq4nl). */
NF_INLINE float IQ4_NL_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                              int off, int blk, int ltid) {
    const float d = nf_ql_half(bp, 0);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * IQ4_NL_BLOCK_SIZE + bo;
    float sum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const uint q = bp[2 + bo + j];
        sum += x[j] * IQ4_NL_KVALUES[q & 0xFu] + x[j + 16] * IQ4_NL_KVALUES[q >> 4];
    }
    return sum * d;
}

NF_INLINE void IQ4_NL_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                  NF_GLOBAL const float* input, int off, int blk, int ltid,
                                  float* gs, float* us) {
    const float gd = nf_ql_half(gbp, 0), ud = nf_ql_half(ubp, 0);
    const int bo = ltid * 4;
    NF_GLOBAL const float* x = input + off + blk * IQ4_NL_BLOCK_SIZE + bo;
    float gsum = 0.0f, usum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const float x0 = x[j], x1 = x[j + 16];
        const uint gq = gbp[2 + bo + j], uq = ubp[2 + bo + j];
        gsum += x0 * IQ4_NL_KVALUES[gq & 0xFu] + x1 * IQ4_NL_KVALUES[gq >> 4];
        usum += x0 * IQ4_NL_KVALUES[uq & 0xFu] + x1 * IQ4_NL_KVALUES[uq >> 4];
    }
    *gs += gsum * gd;
    *us += usum * ud;
}

NF_INLINE float IQ4_NL_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES);
    const int e = i % IQ4_NL_BLOCK_SIZE;
    const uint q = bp[2 + (e & 15)];
    return nf_ql_half(bp, 0) * IQ4_NL_KVALUES[e < 16 ? (q & 0xFu) : (q >> 4)];
}

/* ================================ Q8_0 ================================
 * 34 B: d(f16)@0, qs[32](int8)@2. y[k] = qs[k]*d (llama.cpp ggml-quants.c:551-566).
 * Work-item ltid owns elements 8*ltid .. 8*ltid+7. */
NF_INLINE float nf_ql_q8_0_part(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* x, int eo) {
    NF_GLOBAL const nf_i8* q = (NF_GLOBAL const nf_i8*)(bp + 2 + eo);
    return dot(vload4(0, x), convert_float4(vload4(0, q)))
         + dot(vload4(0, x + 4), convert_float4(vload4(0, q + 4)));
}

NF_INLINE float Q8_0_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                            int off, int blk, int ltid) {
    const int eo = ltid * Q8_0_ELEMENTS_PER_THREAD;
    NF_GLOBAL const float* x = input + off + blk * Q8_0_BLOCK_SIZE + eo;
    return nf_ql_q8_0_part(bp, x, eo) * q8_0_scale(bp);
}

NF_INLINE void Q8_0_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                NF_GLOBAL const float* input, int off, int blk, int ltid,
                                float* gs, float* us) {
    const int eo = ltid * Q8_0_ELEMENTS_PER_THREAD;
    NF_GLOBAL const float* x = input + off + blk * Q8_0_BLOCK_SIZE + eo;
    *gs += nf_ql_q8_0_part(gbp, x, eo) * q8_0_scale(gbp);
    *us += nf_ql_q8_0_part(ubp, x, eo) * q8_0_scale(ubp);
}

/* A non-finite block scale (corrupt or padding block) decodes as 0, matching both Q8_0 embedding kernels. */
NF_INLINE float Q8_0_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES);
    float d = q8_0_scale(bp);
    if (!isfinite(d)) d = 0.0f;
    return (float)((NF_GLOBAL const nf_i8*)bp)[2 + i % Q8_0_BLOCK_SIZE] * d;
}

/* ================================ MXFP4 ===============================
 * 17 B: e(e8m0)@0, qs[16]@1. y[k] = kv[qs[k]&15]*s, y[k+16] = kv[qs[k]>>4]*s, with the doubled
 * e2m1 values kv and the half scale s = ggml_e8m0_to_fp32_half(e) (llama.cpp ggml-quants.c:569-586). */
NF_INLINE float MXFP4_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                             int off, int blk, int ltid) {
    const int bo = ltid * MXFP4_BYTES_PER_THREAD;
    NF_GLOBAL const float* x = input + off + blk * MXFP4_BLOCK_SIZE + bo;
    float sum = 0.0f;
    for (int j = 0; j < MXFP4_BYTES_PER_THREAD; j++) {
        const uint q = bp[1 + bo + j];
        sum += x[j] * MXFP4_KVALUES[q & 0xFu] + x[j + 16] * MXFP4_KVALUES[q >> 4];
    }
    return sum * mxfp4_scale(bp[0]);
}

NF_INLINE void MXFP4_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                 NF_GLOBAL const float* input, int off, int blk, int ltid,
                                 float* gs, float* us) {
    const int bo = ltid * MXFP4_BYTES_PER_THREAD;
    NF_GLOBAL const float* x = input + off + blk * MXFP4_BLOCK_SIZE + bo;
    float gsum = 0.0f, usum = 0.0f;
    for (int j = 0; j < MXFP4_BYTES_PER_THREAD; j++) {
        const float x0 = x[j], x1 = x[j + 16];
        const uint gq = gbp[1 + bo + j], uq = ubp[1 + bo + j];
        gsum += x0 * MXFP4_KVALUES[gq & 0xFu] + x1 * MXFP4_KVALUES[gq >> 4];
        usum += x0 * MXFP4_KVALUES[uq & 0xFu] + x1 * MXFP4_KVALUES[uq >> 4];
    }
    *gs += gsum * mxfp4_scale(gbp[0]);
    *us += usum * mxfp4_scale(ubp[0]);
}

NF_INLINE float MXFP4_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES);
    const int e = i % MXFP4_BLOCK_SIZE;
    const uint q = bp[1 + (e & 15)];
    return mxfp4_scale(bp[0]) * MXFP4_KVALUES[e < 16 ? (q & 0xFu) : (q >> 4)];
}

/* =============================== IQ4_XS ===============================
 * 136 B: d(f16)@0, scales_h(u16)@2, scales_l[4]@4, qs[128]@8. Sub-block sb (32 elements, qs bytes
 * 8 + 16*sb ..) uses dl = d*(ls - 32) and the IQ4_NL split layout (llama.cpp ggml-quants.c:2744-2762).
 * Work-item ltid owns sub-block ltid/4 and its qs bytes 4*(ltid%4) .. +3. */
NF_INLINE float IQ4_XS_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
                              int off, int blk, int ltid) {
    const int sb = ltid / 4, bo = (ltid % 4) * 4;
    NF_GLOBAL const uchar* qs = bp + IQ4_XS_HEADER_BYTES + sb * IQ4_XS_SUB_BLOCK_QUANT_BYTES + bo;
    NF_GLOBAL const float* x = input + off + blk * IQ4_XS_BLOCK_SIZE + sb * 32 + bo;
    float sum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const uint q = qs[j];
        sum += x[j] * IQ4_NL_KVALUES[q & 0xFu] + x[j + 16] * IQ4_NL_KVALUES[q >> 4];
    }
    return sum * iq4xs_sub_scale(bp, sb);
}

NF_INLINE void IQ4_XS_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                  NF_GLOBAL const float* input, int off, int blk, int ltid,
                                  float* gs, float* us) {
    const int sb = ltid / 4, bo = (ltid % 4) * 4;
    const int qo = IQ4_XS_HEADER_BYTES + sb * IQ4_XS_SUB_BLOCK_QUANT_BYTES + bo;
    NF_GLOBAL const float* x = input + off + blk * IQ4_XS_BLOCK_SIZE + sb * 32 + bo;
    float gsum = 0.0f, usum = 0.0f;
    for (int j = 0; j < 4; j++) {
        const float x0 = x[j], x1 = x[j + 16];
        const uint gq = gbp[qo + j], uq = ubp[qo + j];
        gsum += x0 * IQ4_NL_KVALUES[gq & 0xFu] + x1 * IQ4_NL_KVALUES[gq >> 4];
        usum += x0 * IQ4_NL_KVALUES[uq & 0xFu] + x1 * IQ4_NL_KVALUES[uq >> 4];
    }
    *gs += gsum * iq4xs_sub_scale(gbp, sb);
    *us += usum * iq4xs_sub_scale(ubp, sb);
}

NF_INLINE float IQ4_XS_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_ql_elem_block(weight, dim, tok, i, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES);
    const int w = i % IQ4_XS_BLOCK_SIZE, sb = w / 32, e = w % 32;
    const uint q = bp[IQ4_XS_HEADER_BYTES + sb * IQ4_XS_SUB_BLOCK_QUANT_BYTES + (e & 15)];
    return iq4xs_sub_scale(bp, sb) * IQ4_NL_KVALUES[e < 16 ? (q & 0xFu) : (q >> 4)];
}

#endif /* NF_QUANT_DECODE_LEGACY_H */
