/* common/quant_decode_new.h - TQ1_0, TQ2_0, NVFP4, Q1_0 and Q2_0 decode triples. Requires: common/dialect.h, common/config.h, common/quant_blocks.h */
/*
 * Decode triples (_decode, _decode_dual, _decode_elem) of the ternary, NVFP4 and 1-/2-bit block types
 * (llama.cpp ggml-common.h block_tq1_0, block_tq2_0, block_nvfp4, block_q1_0, block_q2_0) for the
 * IQ_ALL_KERNELS contract of common/matvec_family.h.
 *
 * Work split: every work-item of a block owns NF_NQ_EPT = 8 elements, in two runs of 4 that start at
 * block elements nf_<t>_xo(ltid) and nf_<t>_xo(ltid) + NF_<T>_X1:
 *   TQ1_0, TQ2_0  TPS = 32   elements [8*ltid, 8*ltid + 8)
 *   Q1_0          TPS = 16   elements [8*ltid, 8*ltid + 8)  (qs byte ltid)
 *   Q2_0          TPS = 8    elements [8*ltid, 8*ltid + 8)  (qs bytes 2*ltid, 2*ltid + 1)
 *   NVFP4         TPS = 8    sub-block s = ltid/2, qs bytes 8*s + j0 .. 8*s + j0 + 3 with
 *                            j0 = 4*(ltid%2): elements 16*s + j0 + k (low nibbles) and
 *                            16*s + j0 + k + 8 (high nibbles), k in [0, 4)
 * Each type's 8 elements share one scale, so per type nf_<t>_w8(bp, ltid, &w0, &w1) yields the
 * unscaled quant values of both runs and nf_<t>_scale(bp, ltid) the scale; T_decode is
 * scale * (dot(w0, x run 0) + dot(w1, x run 1)), which equals the dot product of the
 * dequantize_row_<t> output with the activations. T_decode_elem dequantizes one element and forms
 * the row byte offset in nf_i64.
 *
 * f16 fields sit at even block offsets and every block size is even, so vload_half is 2-byte aligned.
 */
#ifndef NF_QUANT_DECODE_NEW_H
#define NF_QUANT_DECODE_NEW_H

#define NF_NQ_EPT 8    /* elements per work-item of every new-quant decoder */

#define TQ1_0_TPS 32
#define TQ2_0_TPS 32
#define NVFP4_TPS 8
#define Q1_0_TPS  16
#define Q2_0_TPS  8

/* Byte offsets of the block fields (llama.cpp ggml-common.h). */
#define NF_TQ1_QS    0     /* u8[48]: five base-3 trits per byte */
#define NF_TQ1_QH    48    /* u8[4]: four base-3 trits per byte */
#define NF_TQ1_D     52    /* f16 */
#define NF_TQ2_QS    0     /* u8[64]: four 2-bit planes */
#define NF_TQ2_D     64    /* f16 */
#define NF_NVFP4_D   0     /* u8[4]: UE4M3 scale of each 16-element sub-block */
#define NF_NVFP4_QS  4     /* u8[32]: split E2M1 nibbles, 8 bytes per sub-block */
#define NF_Q1_D      0     /* f16 */
#define NF_Q1_QS     2     /* u8[16]: one sign bit per element */
#define NF_Q2_D      0     /* f16 */
#define NF_Q2_QS     2     /* u8[16]: four 2-bit codes per byte */

/* Distance from the first to the second 4-element run of a work-item. */
#define NF_TQ1_X1   4
#define NF_TQ2_X1   4
#define NF_NVFP4_X1 8
#define NF_Q1_X1    4
#define NF_Q2_X1    4

/* 3^n for the TQ1_0 trit digits n in [0, 5) (llama.cpp ggml-quants.c:2433 pow3). */
NF_TABLE(uint, NF_TQ1_POW3, 5) = { 1u, 3u, 9u, 27u, 81u };

/* f16 field at byte offset off of block bp. */
NF_INLINE float nf_nq_half(NF_GLOBAL const uchar* bp, int off) {
    return vload_half(0, (NF_GLOBAL const NF_HALF*)(bp + off));
}

/* First byte of the block holding element i of row tok of a [rows x dim] weight. */
NF_INLINE NF_GLOBAL const uchar* nf_nq_block(NF_GLOBAL const uchar* w, int dim, int tok, int i,
                                             int bs, int bb) {
    return w + ((nf_i64)tok * (dim / bs) + i / bs) * bb;
}

/* (b >> sh) & mask - bias of the 4 bytes at p as floats, element k from byte p[k]. */
NF_INLINE float4 nf_nq_bits4(NF_GLOBAL const uchar* p, uint sh, uint mask, float bias) {
    const uchar4 b = vload4(0, p);
    return NF_F4((float)(((uint)b.x >> sh) & mask), (float)(((uint)b.y >> sh) & mask),
                 (float)(((uint)b.z >> sh) & mask), (float)(((uint)b.w >> sh) & mask)) - bias;
}

/* Trit of byte b at digit power pw, minus 1: q = (u8)(b * pw), ((q * 3) >> 8) - 1
 * (llama.cpp ggml-quants.c:2442-2444). */
NF_INLINE float nf_tq1_trit(uint b, uint pw) {
    const uint q = (b * pw) & 0xFFu;
    return (float)((q * 3u) >> 8) - 1.0f;
}

/* nf_tq1_trit of the 4 bytes at p, element k from byte p[k]. */
NF_INLINE float4 nf_tq1_trits4(NF_GLOBAL const uchar* p, uint pw) {
    const uchar4 b = vload4(0, p);
    return NF_F4(nf_tq1_trit((uint)b.x, pw), nf_tq1_trit((uint)b.y, pw),
                 nf_tq1_trit((uint)b.z, pw), nf_tq1_trit((uint)b.w, pw));
}

/* UE4M3 scale byte to float, halved for the doubled E2M1 codebook (llama.cpp ggml-impl.h:514
 * ggml_ue4m3_to_fp32): 0 and 0x7F give 0; otherwise the exact value (e ? 8 + m : m) * 2^(max(e,1) - 11),
 * which equals ldexpf(1 + m/8, e - 7) * 0.5 for e > 0 and ldexpf(m, -9) * 0.5 for e = 0. */
NF_INLINE float nf_nq_ue4m3_half(uint x) {
    if (x == 0u || x == 0x7Fu) {
        return 0.0f;
    }
    const uint e = (x >> 3) & 0xFu, m = x & 7u;
    const float mant = (float)(e == 0u ? m : 8u + m);
    const uint ex = e == 0u ? 1u : e;
    return mant * as_float((ex + 116u) << 23);
}

/* T_decode and T_decode_dual (matvec_family.h contract) from W8, SCALE, XO and the run distance X1. */
#define NF_NQ_DOT_PAIR(T, W8, SCALE, XO, X1) \
NF_INLINE float T##_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input, \
                           int off, int blk, int ltid) { \
    float4 w0, w1; \
    W8(bp, ltid, &w0, &w1); \
    NF_GLOBAL const float* x = input + off + blk * T##_BLOCK_SIZE + XO(ltid); \
    return SCALE(bp, ltid) * (dot(w0, vload4(0, x)) + dot(w1, vload4(0, x + X1))); \
} \
NF_INLINE void T##_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp, \
                               NF_GLOBAL const float* input, int off, int blk, int ltid, \
                               float* gs, float* us) { \
    float4 g0, g1, u0, u1; \
    W8(gbp, ltid, &g0, &g1); \
    W8(ubp, ltid, &u0, &u1); \
    NF_GLOBAL const float* x = input + off + blk * T##_BLOCK_SIZE + XO(ltid); \
    const float4 x0 = vload4(0, x), x1 = vload4(0, x + X1); \
    *gs += SCALE(gbp, ltid) * (dot(g0, x0) + dot(g1, x1)); \
    *us += SCALE(ubp, ltid) * (dot(u0, x0) + dot(u1, x1)); \
}

/* Element offset of the first run of work-item ltid for the consecutive 8-element split. */
NF_INLINE int nf_nq_xo8(int ltid) {
    return ltid * NF_NQ_EPT;
}

/* ---- TQ1_0 (llama.cpp ggml-quants.c:2429-2466) ----
 * Element e is trit n of one byte, y = (trit - 1) * d:
 *   e <  160: byte qs[e % 32],              n = e / 32
 *   e <  240: byte qs[32 + (e - 160) % 16], n = (e - 160) / 16
 *   e >= 240: byte qh[(e - 240) % 4],       n = (e - 240) / 4
 * Work-item ltid < 20 reads qs[8*(ltid%4) ..] with n = ltid/4; ltid in [20, 30) reads
 * qs[32 + 8*((ltid-20)%2) ..] with n = (ltid-20)/2; ltid 30, 31 read qh[0..3] twice, with
 * n = 2*(ltid-30) for the first run and n + 1 for the second. */
NF_INLINE void nf_tq1_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    int b0, b1, n0, n1;
    if (ltid < 20) {
        b0 = NF_TQ1_QS + 8 * (ltid & 3);
        b1 = b0 + 4;
        n0 = ltid >> 2;
        n1 = n0;
    } else if (ltid < 30) {
        b0 = NF_TQ1_QS + 32 + 8 * ((ltid - 20) & 1);
        b1 = b0 + 4;
        n0 = (ltid - 20) >> 1;
        n1 = n0;
    } else {
        b0 = NF_TQ1_QH;
        b1 = NF_TQ1_QH;
        n0 = 2 * (ltid - 30);
        n1 = n0 + 1;
    }
    *w0 = nf_tq1_trits4(bp + b0, NF_TQ1_POW3[n0]);
    *w1 = nf_tq1_trits4(bp + b1, NF_TQ1_POW3[n1]);
}
NF_INLINE float nf_tq1_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_nq_half(bp, NF_TQ1_D);
}
NF_NQ_DOT_PAIR(TQ1_0, nf_tq1_w8, nf_tq1_scale, nf_nq_xo8, NF_TQ1_X1)
NF_INLINE float TQ1_0_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, TQ1_0_BLOCK_SIZE, TQ1_0_BLOCK_BYTES);
    const int e = i % TQ1_0_BLOCK_SIZE;
    int byte_off, n;
    if (e < 160) {
        byte_off = NF_TQ1_QS + (e & 31);
        n = e >> 5;
    } else if (e < 240) {
        byte_off = NF_TQ1_QS + 32 + ((e - 160) & 15);
        n = (e - 160) >> 4;
    } else {
        byte_off = NF_TQ1_QH + ((e - 240) & 3);
        n = (e - 240) >> 2;
    }
    return nf_tq1_trit((uint)bp[byte_off], NF_TQ1_POW3[n]) * nf_nq_half(bp, NF_TQ1_D);
}

/* ---- TQ2_0 (llama.cpp ggml-quants.c:2468-2485) ----
 * Element e = 128*hf + 32*l + m is ((qs[32*hf + m] >> 2l) & 3) - 1, times d. Work-item ltid covers
 * hf = ltid/16, l = (ltid/4)%4, m = 8*(ltid%4) + 0..7. */
NF_INLINE void nf_tq2_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    const uint sh = (uint)((ltid >> 2) & 3) * 2u;
    NF_GLOBAL const uchar* q = bp + NF_TQ2_QS + (ltid >> 4) * 32 + (ltid & 3) * 8;
    *w0 = nf_nq_bits4(q, sh, 3u, 1.0f);
    *w1 = nf_nq_bits4(q + 4, sh, 3u, 1.0f);
}
NF_INLINE float nf_tq2_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_nq_half(bp, NF_TQ2_D);
}
NF_NQ_DOT_PAIR(TQ2_0, nf_tq2_w8, nf_tq2_scale, nf_nq_xo8, NF_TQ2_X1)
NF_INLINE float TQ2_0_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_BYTES);
    const int e = i % TQ2_0_BLOCK_SIZE;
    const int hf = e >> 7, l = (e >> 5) & 3, m = e & 31;
    const uint q = ((uint)bp[NF_TQ2_QS + hf * 32 + m] >> (2 * l)) & 3u;
    return ((float)q - 1.0f) * nf_nq_half(bp, NF_TQ2_D);
}

/* ---- NVFP4 (llama.cpp ggml-quants.c:589-611) ----
 * Sub-block s (16 elements) has scale ue4m3(d[s]); its element j < 8 is kvalues_fp4[qs[8s + j] & 15]
 * and element j + 8 is kvalues_fp4[qs[8s + j] >> 4] (MXFP4_KVALUES = kvalues_fp4). */
NF_INLINE void nf_nvfp4_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    const uchar4 b = vload4(0, bp + NF_NVFP4_QS + (ltid >> 1) * 8 + (ltid & 1) * 4);
    *w0 = NF_F4(MXFP4_KVALUES[b.x & 0xF], MXFP4_KVALUES[b.y & 0xF],
                MXFP4_KVALUES[b.z & 0xF], MXFP4_KVALUES[b.w & 0xF]);
    *w1 = NF_F4(MXFP4_KVALUES[b.x >> 4], MXFP4_KVALUES[b.y >> 4],
                MXFP4_KVALUES[b.z >> 4], MXFP4_KVALUES[b.w >> 4]);
}
NF_INLINE float nf_nvfp4_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_nq_ue4m3_half((uint)bp[NF_NVFP4_D + (ltid >> 1)]);
}
NF_INLINE int nf_nvfp4_xo(int ltid) {
    return (ltid >> 1) * 16 + (ltid & 1) * 4;
}
NF_NQ_DOT_PAIR(NVFP4, nf_nvfp4_w8, nf_nvfp4_scale, nf_nvfp4_xo, NF_NVFP4_X1)
NF_INLINE float NVFP4_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES);
    const int e = i % NVFP4_BLOCK_SIZE;
    const int s = e >> 4, t = e & 15;
    const uint q = bp[NF_NVFP4_QS + s * 8 + (t & 7)];
    const uint nib = t < 8 ? (q & 0xFu) : (q >> 4);
    return MXFP4_KVALUES[nib] * nf_nq_ue4m3_half((uint)bp[NF_NVFP4_D + s]);
}

/* ---- Q1_0 (llama.cpp ggml-quants.c:419-437) ----
 * Element j is +d when bit j%8 of qs[j/8] is set and -d otherwise. Work-item ltid owns qs[ltid]. */
NF_INLINE void nf_q1_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    const uint b = bp[NF_Q1_QS + ltid];
    *w0 = 2.0f * NF_F4((float)(b & 1u), (float)((b >> 1) & 1u), (float)((b >> 2) & 1u), (float)((b >> 3) & 1u)) - 1.0f;
    *w1 = 2.0f * NF_F4((float)((b >> 4) & 1u), (float)((b >> 5) & 1u), (float)((b >> 6) & 1u), (float)(b >> 7)) - 1.0f;
}
NF_INLINE float nf_q1_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_nq_half(bp, NF_Q1_D);
}
NF_NQ_DOT_PAIR(Q1_0, nf_q1_w8, nf_q1_scale, nf_nq_xo8, NF_Q1_X1)
NF_INLINE float Q1_0_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, Q1_0_BLOCK_SIZE, Q1_0_BLOCK_BYTES);
    const int e = i % Q1_0_BLOCK_SIZE;
    const float d = nf_nq_half(bp, NF_Q1_D);
    return (((uint)bp[NF_Q1_QS + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
}

/* ---- Q2_0 (llama.cpp ggml-quants.c:439-457) ----
 * Element j is (((qs[j/4] >> 2*(j%4)) & 3) - 1) * d (codes 00 = -1, 01 = 0, 10 = +1, 11 = +2).
 * Work-item ltid owns qs[2*ltid] (first run) and qs[2*ltid + 1] (second run). */
NF_INLINE float4 nf_q2_codes4(uint b) {
    return NF_F4((float)(b & 3u), (float)((b >> 2) & 3u), (float)((b >> 4) & 3u), (float)(b >> 6)) - 1.0f;
}
NF_INLINE void nf_q2_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    *w0 = nf_q2_codes4((uint)bp[NF_Q2_QS + 2 * ltid]);
    *w1 = nf_q2_codes4((uint)bp[NF_Q2_QS + 2 * ltid + 1]);
}
NF_INLINE float nf_q2_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_nq_half(bp, NF_Q2_D);
}
NF_NQ_DOT_PAIR(Q2_0, nf_q2_w8, nf_q2_scale, nf_nq_xo8, NF_Q2_X1)
NF_INLINE float Q2_0_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, Q2_0_BLOCK_SIZE, Q2_0_BLOCK_BYTES);
    const int e = i % Q2_0_BLOCK_SIZE;
    const uint q = ((uint)bp[NF_Q2_QS + (e >> 2)] >> (2 * (e & 3))) & 3u;
    return ((float)q - 1.0f) * nf_nq_half(bp, NF_Q2_D);
}

#endif /* NF_QUANT_DECODE_NEW_H */
