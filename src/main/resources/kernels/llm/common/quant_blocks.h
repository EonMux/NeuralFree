/* common/quant_blocks.h - quant block geometry, scalar block-field helpers and the MXFP4 / IQ4_NL / E4M3FN / bitsandbytes 4-bit value tables. Requires: common/dialect.h, common/config.h */
/* Portions derived from llama.cpp/ggml (MIT License); see THIRD-PARTY-NOTICES.md. */
/*
 * Block sizes and byte counts follow the ggml block structs (ggml-common.h static_asserts).
 * Q8_0 and Q8_K constants live in common/config.h.
 */
#ifndef NF_QUANT_BLOCKS_H
#define NF_QUANT_BLOCKS_H

#define QK_K 256

/* ---- legacy 32-element blocks ---- */
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BLOCK_BYTES 18       /* d(f16) qs[16] */
#define Q4_1_BLOCK_SIZE 32
#define Q4_1_BLOCK_BYTES 20       /* d(f16) m(f16) qs[16] */
#define Q5_0_BLOCK_SIZE 32
#define Q5_0_BLOCK_BYTES 22       /* d(f16) qh(u32) qs[16] */
#define Q5_1_BLOCK_SIZE 32
#define Q5_1_BLOCK_BYTES 24       /* d(f16) m(f16) qh(u32) qs[16] */
#define IQ4_NL_BLOCK_SIZE 32
#define IQ4_NL_BLOCK_BYTES 18     /* d(f16) qs[16] */

#define MXFP4_BLOCK_SIZE 32
#define MXFP4_BLOCK_BYTES 17      /* e(e8m0) qs[16] */
#define MXFP4_THREADS_PER_BLOCK 4
#define MXFP4_BYTES_PER_THREAD 4  /* 4 nibble-bytes = 8 elements */

/* ---- NeuralFree extension blocks (GGMLType ids 1000-1005, gguf.quant.BlockLayout), little-endian ---- */
#define F8_E4M3_BLOCK_SIZE 32
#define F8_E4M3_BLOCK_BYTES 32        /* qs[32](e4m3fn) */
#define F8_E4M3_B128_BLOCK_SIZE 128
#define F8_E4M3_B128_BLOCK_BYTES 132  /* d(f32) qs[128](e4m3fn) */
#define MXFP8_BLOCK_SIZE 32
#define MXFP8_BLOCK_BYTES 33          /* e(e8m0) qs[32](e4m3fn) */
#define I8_B128_BLOCK_SIZE 128
#define I8_B128_BLOCK_BYTES 132       /* d(f32) qs[128](int8) */
#define NF4_BLOCK_SIZE 64
#define NF4_BLOCK_BYTES 36            /* absmax(f32) qs[32]: byte j = element j | element j+32 << 4 */
#define FP4_BNB_BLOCK_SIZE 64
#define FP4_BNB_BLOCK_BYTES 36        /* absmax(f32) qs[32]: byte j = element j | element j+32 << 4 */

/* Byte offsets of the extension block fields. */
#define NF_F8_E4M3_QS      0
#define NF_F8_E4M3_B128_D  0
#define NF_F8_E4M3_B128_QS 4
#define NF_MXFP8_E         0
#define NF_MXFP8_QS        1
#define NF_I8_B128_D       0
#define NF_I8_B128_QS      4
#define NF_NF4_D           0
#define NF_NF4_QS          4
#define NF_FP4_BNB_D       0
#define NF_FP4_BNB_QS      4
#define NF_BNB4_HI         32         /* NF4 / FP4_BNB: element of the high nibble of qs byte j is j + 32 */

/* ---- 256-element super-blocks ---- */
#define Q2_K_BLOCK_SIZE 256
#define Q2_K_BLOCK_BYTES 84       /* scales[16] qs[64] d(f16) dmin(f16) */
#define Q3_K_BLOCK_SIZE 256
#define Q3_K_BLOCK_BYTES 110      /* hmask[32] qs[64] scales[12] d(f16) */
#define Q4_K_BLOCK_SIZE 256
#define Q4_K_BLOCK_BYTES 144      /* d(f16) dmin(f16) scales[12] qs[128] */
#define Q5_K_BLOCK_SIZE 256
#define Q5_K_BLOCK_BYTES 176      /* d(f16) dmin(f16) scales[12] qh[32] qs[128] */
#define Q6_K_BLOCK_SIZE 256
#define Q6_K_BLOCK_BYTES 210      /* ql[128] qh[64] scales[16] d(f16) */

#define IQ4_XS_BLOCK_SIZE 256
#define IQ4_XS_BLOCK_BYTES 136
#define IQ4_XS_SUB_BLOCKS 8
#define IQ4_XS_SUB_BLOCK_QUANT_BYTES 16
#define IQ4_XS_HEADER_BYTES 8       /* d(2) + scales_h(2) + scales_l(4) */
#define IQ4_XS_THREADS_PER_SUPER 32 /* 4 threads per sub-block * 8 sub-blocks */

#define IQ3_XXS_BLOCK_SIZE 256
#define IQ3_XXS_BLOCK_BYTES 98    /* d(f16) qs[96] */
#define IQ3_S_BLOCK_SIZE 256
#define IQ3_S_BLOCK_BYTES 110     /* d(f16) qs[64] qh[8] signs[32] scales[4] */
#define IQ2_XXS_BLOCK_SIZE 256
#define IQ2_XXS_BLOCK_BYTES 66    /* d(f16) qs[32 x u16] */
#define IQ2_XS_BLOCK_SIZE 256
#define IQ2_XS_BLOCK_BYTES 74     /* d(f16) qs[32 x u16] scales[8] */
#define IQ2_S_BLOCK_SIZE 256
#define IQ2_S_BLOCK_BYTES 82      /* d(f16) qs[64] qh[8] scales[8] */
#define IQ1_S_BLOCK_SIZE 256
#define IQ1_S_BLOCK_BYTES 50      /* d(f16) qs[32] qh[8 x u16] */
#define IQ1_M_BLOCK_SIZE 256
#define IQ1_M_BLOCK_BYTES 56      /* qs[32] qh[16] scales[8] */
#define IQ1_S_DELTA 0.125f
#define IQ1_M_DELTA 0.125f

#define TQ1_0_BLOCK_SIZE 256
#define TQ1_0_BLOCK_BYTES 54      /* qs[48] qh[4] d(f16) */
#define TQ2_0_BLOCK_SIZE 256
#define TQ2_0_BLOCK_BYTES 66      /* qs[64] d(f16) */

/* ---- other block sizes ---- */
#define NVFP4_BLOCK_SIZE 64
#define NVFP4_BLOCK_BYTES 36      /* d[4](ue4m3) qs[32] */
#define Q1_0_BLOCK_SIZE 128
#define Q1_0_BLOCK_BYTES 18       /* d(f16) qs[16] */
#define Q2_0_BLOCK_SIZE 64
#define Q2_0_BLOCK_BYTES 18       /* d(f16) qs[16] */

/* ---- value tables ---- */
/* MXFP4 E2M1 nibble -> value, doubled as in ggml kvalues_mxfp4 (paired with the half-scale below). */
NF_TABLE(float, MXFP4_KVALUES, 16) = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

/* IQ4_NL / IQ4_XS non-linear 4-bit values (ggml kvalues_iq4nl). */
NF_TABLE(float, IQ4_NL_KVALUES, 16) = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

/* E4M3FN code -> value (torch.float8_e4m3fn, utils.numeric.Fp8Codec.E4M3FN): bias 7, subnormal m * 2^-9,
 * largest finite 0x7E = 448. The NaN codes 0x7F and 0xFF decode to 0 here; repackers reject them. */
NF_TABLE(float, F8_E4M3_LUT, 256) = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, 0.0f
};

/* bitsandbytes NF4 codebook (get_4bit_type("nf4"), bitsandbytes functional.py:733-750; max magnitude 1). */
NF_TABLE(float, NF4_CODE, 16) = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

/* bitsandbytes FP4 codebook (get_4bit_type("fp4"), functional.py:764 {0, 0.0625, 8, 12, 4, 6, 2, 3, -0,
 * -0.0625, -8, -12, -4, -6, -2, -3}, each divided by 12 in f32 as functional.py:793-794). */
NF_TABLE(float, FP4_BNB_CODE, 16) = {
    0.0f, 0.00520833349f, 0.666666687f, 1.0f, 0.333333343f, 0.5f, 0.166666672f, 0.25f,
    -0.0f, -0.00520833349f, -0.666666687f, -1.0f, -0.333333343f, -0.5f, -0.166666672f, -0.25f
};

/* ---- scalar helpers ---- */

/* f16 scale at the start of a Q8_0 block. */
NF_INLINE float q8_0_scale(NF_GLOBAL const uchar* block_ptr) {
    return vload_half(0, (NF_GLOBAL const NF_HALF*)block_ptr);
}

/* BF16 bits -> f32 (zero-padded mantissa). */
NF_INLINE float bf16_to_f32(ushort val) {
    return as_float((uint)val << 16);
}

/* f16 bits -> f32 without a memory load (IQ1_M reconstructed scale). */
NF_INLINE float half_bits_to_float(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint expo = (h >> 10) & 0x1Fu;
    uint mant = h & 0x3FFu;
    if (expo == 0u) {
        if (mant == 0u) return as_float(sign);
        while ((mant & 0x400u) == 0u) { mant <<= 1; expo--; }
        expo++; mant &= 0x3FFu;
    } else if (expo == 31u) {
        return as_float(sign | 0x7F800000u | (mant << 13));
    }
    return as_float(sign | ((expo + 112u) << 23) | (mant << 13));
}

/* Little-endian u32 from byte storage (safe for unaligned block offsets). */
NF_INLINE uint read_u32_le(NF_GLOBAL const uchar* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

/* 6-bit scale/min pair j of a Q4_K/Q5_K block (ggml get_scale_min_k4). */
NF_INLINE float2 k4_scale_min(NF_GLOBAL const uchar* sc, int j) {
    if (j < 4)
        return NF_F2((float)(sc[j] & 63u), (float)(sc[j + 4] & 63u));
    return NF_F2((float)((sc[j+4] & 0xFu) | ((uint)(sc[j-4] >> 6) << 4)),
                 (float)((sc[j+4] >> 4)   | ((uint)(sc[j]   >> 6) << 4)));
}

/* E8M0 exponent byte -> float half-scale (ggml_e8m0_to_fp32_half). */
NF_INLINE float mxfp4_scale(uint e) {
    uint bits;
    if (e < 2u) {
        bits = 0x00200000u << e;
    } else {
        bits = (e - 1u) << 23;
    }
    return as_float(bits);
}

/* E8M0 exponent byte -> 2^(e - 127) exactly (FpBlockDecoders.e8m0ToFp32): a normal float for e >= 1, the
 * subnormal 2^-127 for e = 0 (which acts as 0 where denormals are flushed, as in the CUDA build). The NaN
 * code 255 gives +inf; repackers reject it. */
NF_INLINE float e8m0_scale(uint e) {
    return as_float(e == 0u ? 0x00400000u : e << 23);
}

/* 6-bit sub-block scale of IQ4_XS sub-block sb (0-7), biased by -32, times d. */
NF_INLINE float iq4xs_sub_scale(NF_GLOBAL const uchar* block_ptr, int sb) {
    float d = vload_half(0, (NF_GLOBAL const NF_HALF*)block_ptr);
    uint scales_h = (uint)block_ptr[2] | ((uint)block_ptr[3] << 8);
    uint sl_byte  = block_ptr[4 + sb / 2];
    int ls = ((sl_byte >> (4 * (sb & 1))) & 0xF) | (((scales_h >> (2 * sb)) & 3) << 4);
    return d * (float)(ls - 32);
}

/* IQ3_S sub-block scale: d * (1 + 2 * nibble) for 32-element sub-block sb (0-7). */
NF_INLINE float iq3s_scale(NF_GLOBAL const uchar* bp, int sb) {
    float d = vload_half(0, (NF_GLOBAL const NF_HALF*)bp);
    uint sb2 = bp[106 + sb / 2];
    float sc = (sb & 1) ? (float)(1 + 2 * (sb2 >> 4)) : (float)(1 + 2 * (sb2 & 0xFu));
    return d * sc;
}

/* IQ1_M super-block scale reassembled from the top nibbles of its four u16 scale words. */
NF_INLINE float iq1m_reconstruct_d(NF_GLOBAL const uchar* bp) {
    uint sc0 = (uint)bp[48] | ((uint)bp[49] << 8);
    uint sc1 = (uint)bp[50] | ((uint)bp[51] << 8);
    uint sc2 = (uint)bp[52] | ((uint)bp[53] << 8);
    uint sc3 = (uint)bp[54] | ((uint)bp[55] << 8);
    uint f16bits = (sc0 >> 12) | ((sc1 >> 8) & 0xF0u) | ((sc2 >> 4) & 0xF00u) | (sc3 & 0xF000u);
    return half_bits_to_float(f16bits);
}

#endif /* NF_QUANT_BLOCKS_H */
