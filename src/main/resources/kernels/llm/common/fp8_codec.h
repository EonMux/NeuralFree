/* common/fp8_codec.h - the FP8/FP4 codec of the activation quantization schemes: E4M3FN encode and decode,
 * E2M1 encode and decode, the MX scale exponents, and the correctly rounded division the schemes use.
 * Requires: common/dialect.h, common/quant_blocks.h */
/*
 * These are the primitives of gguf.quant.ActFakeQuant and of the activation staging kernels: the value of an
 * E4M3FN code (F8_E4M3_LUT, the torch.float8_e4m3fn codebook shared with utils.numeric.Fp8Codec), the value of
 * an E2M1 code (the doubled MXFP4_KVALUES codebook halved), the round-to-nearest-even encodings of both, the
 * MXFP4/MXFP8 run exponents of vLLM mxfp4_utils.py / mxfp8_utils.py, and a division that equals the host's float
 * division on every dialect.
 */

#define NF_AQ_INT8_MAX 127.0f
#define NF_AQ_FP8_MAX  448.0f
#define NF_AQ_FP4_MAX  6.0f

/* 1e-10f, the amax floor of the per-run schemes (vLLM's eps). */
#define NF_AQ_GROUP_EPS_BITS 0x2EDBE6FFu
/* 1 / (448 * 512) rounded to float, the FP8_TOKEN scale floor (vLLM min_scaling_factor). */
#define NF_AQ_FP8_MIN_SCALE_BITS 0x36924925u

/* FLT_MIN = 2^-126, the MXFP8 floor of amax. */
#define NF_AQ_FLT_MIN_BITS        0x00800000u
/* 448 * 2^-126: from this amax on, amax / 448 is a normal float. */
#define NF_AQ_MXFP8_NORMAL_BITS   0x04E00000u
/* Smallest amax whose correctly rounded quotient amax / 448 exceeds 2^-127 (7 * 2^-121 + 2^-141), i.e. whose
 * biased MXFP8 exponent is 1 rather than 0 below NF_AQ_MXFP8_NORMAL_BITS. */
#define NF_AQ_MXFP8_E1_BITS       0x04600002u

/* Most neighbour steps nf_aq_div takes: OpenCL division is within 3 ulp of the exact quotient (2.5 in the full
 * profile), which is at most 6 floats away when the device quotient lies in the binade below a power of two. */
#define NF_AQ_DIV_MAX_STEPS 6

/*
 * a / b rounded to nearest with ties to even, as the host's float division. CUDA uses the IEEE intrinsic. OpenCL
 * division may be off by a few ulp, so the quotient walks to the neighbour with the smaller remainder |a - q*b|
 * (fma, exact once q is within 1 ulp) until none is smaller; a quotient never lies exactly halfway between two
 * floats, so the closest candidate is unique. Exact for normal operands and quotients.
 */
NF_INLINE float nf_aq_div(float a, float b) {
#ifdef NF_CUDA
    return __fdiv_rn(a, b);
#else
    float q = a / b;
    for (int i = 0; i < NF_AQ_DIV_MAX_STEPS; i++) {
        const float r = fma(-q, b, a);
        const float t = nextafter(q, (r > 0.0f) == (b > 0.0f) ? NF_INF : -NF_INF);
        if (!(fabs(fma(-t, b, a)) < fabs(r))) {
            break;
        }
        q = t;
    }
    return q;
#endif
}

/* Fp8Codec.E4M3FN.encode of clamp(v, -448, 448): round to nearest even, magnitudes >= 448 (the infinities
 * included) to 0x7E, NaN to the NaN code 0x7F, keeping the sign bit (0x80) in every case. */
NF_INLINE uint nf_aq_e4m3_encode(float v) {
    const uint sign = (as_uint(v) >> 24) & 0x80u;
    const float m = fabs(v);
    if (isnan(m)) {
        return sign | 0x7Fu;
    }
    if (m >= NF_AQ_FP8_MAX) {
        return sign | 0x7Eu;
    }
    if (m == 0.0f) {
        return sign;
    }
    int ex = max((int)((as_uint(m) >> 23) & 0xFFu) - 127, -6);
    int units = (int)rint(ldexp(m, 3 - ex));
    if (units < 8) {
        return sign | (uint)units;
    }
    if (units == 16) {
        units = 8;
        ex++;
    }
    return sign | (uint)min(((ex + 7) << 3) | (units - 8), 0x7E);
}

/* Fp8Codec.E4M3FN.decode: the exact value of the code; the NaN codes 0x7F and 0xFF give NaN with their sign. */
NF_INLINE float nf_aq_e4m3_value(uint c) {
    return (c & 0x7Fu) == 0x7Fu ? as_float(((c & 0x80u) << 24) | 0x7FC00000u) : F8_E4M3_LUT[c & 0xFFu];
}

/* FpBlockDecoders.fp32ToE2m1: the E2M1 code (sign bit 3) by the round-to-nearest-even thresholds of vLLM's
 * cast_to_fp4, saturating at 6; NaN takes the largest magnitude. */
NF_INLINE uint nf_aq_e2m1_encode(float v) {
    const uint sign = (as_uint(v) >> 28) & 0x8u;
    const float a = fabs(v);
    uint mag;
    if (isnan(a)) {
        mag = 7u;
    } else if (a <= 0.25f) {
        mag = 0u;
    } else if (a < 0.75f) {
        mag = 1u;
    } else if (a <= 1.25f) {
        mag = 2u;
    } else if (a < 1.75f) {
        mag = 3u;
    } else if (a <= 2.5f) {
        mag = 4u;
    } else if (a < 3.5f) {
        mag = 5u;
    } else if (a <= 5.0f) {
        mag = 6u;
    } else {
        mag = 7u;
    }
    return sign | mag;
}

/* FpBlockDecoders.e2m1ToFp32: half the doubled codebook MXFP4_KVALUES; code 8 (negative zero) gives +0. */
NF_INLINE float nf_aq_e2m1_value(uint c) {
    return MXFP4_KVALUES[c & 15u] * 0.5f;
}

/* e4m3(v): the value of the saturating E4M3FN code of v. */
NF_INLINE float nf_aq_e4m3(float v) {
    return nf_aq_e4m3_value(nf_aq_e4m3_encode(v));
}

/* e2m1(v): the value of the saturating E2M1 code of v. */
NF_INLINE float nf_aq_e2m1(float v) {
    return nf_aq_e2m1_value(nf_aq_e2m1_encode(v));
}

/* MXFP4 scale exponent of a run with maximum magnitude amax (vLLM mxfp4_utils.py:255-280). */
NF_INLINE int nf_aq_mxfp4_exp(float amax) {
    const uint r = (as_uint(amax) + 0x200000u) & 0x7F800000u;
    const int ef = (int)(r >> 23);
    if (ef == 255) {
        return 127;
    }
    return clamp((ef == 0 ? -126 : ef - 127) - 2, -127, 127);
}

/* MXFP8 scale exponent of a run with maximum magnitude amax (vLLM mxfp8_utils.py:55-70). A quotient below 2^-126
 * has ceil(log2) <= -126, so its biased exponent is 1 above 2^-127 and 0 otherwise; that case is decided on amax
 * itself, which keeps it exact on devices that flush subnormals. */
NF_INLINE int nf_aq_mxfp8_exp(float amax) {
    const float m = fmax(amax, as_float(NF_AQ_FLT_MIN_BITS));
    if (m < as_float(NF_AQ_MXFP8_NORMAL_BITS)) {
        return (m >= as_float(NF_AQ_MXFP8_E1_BITS) ? 1 : 0) - 127;
    }
    const uint t = as_uint(nf_aq_div(m, NF_AQ_FP8_MAX));
    const int ceil_log2 = (int)(t >> 23) - 127 + ((t & 0x7FFFFFu) != 0u ? 1 : 0);
    return clamp(ceil_log2 + 127, 0, 254) - 127;
}
