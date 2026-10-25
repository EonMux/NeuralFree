/* act_quant.cl - LLM_ActFakeQuant, LLM_PermuteCols: activation fake-quantization and input column permutation of matmul inputs. Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h */
/*
 * LLM_ActFakeQuant(i n_runs, i run_len, i scheme, f scale, i zero_point, f global_scale, i x_off, i y_off, Rn x, Wn y)
 *
 * Fake-quantizes runs of run_len consecutive elements: run g covers x + x_off + g*run_len and
 * y + y_off + g*run_len, and y = dequant(quant(x)) per the activation quantization scheme of the checkpoint,
 * so that the unchanged weight kernel then computes W * dequant(quant(x)) (vLLM's emulated W8A8/W4A4 GEMMs).
 * scheme is the gguf.quant.ActQuantScheme code; amax is max|x| over the run, div(a, b) is the correctly rounded
 * float quotient (the host's a / b), inv(s) = div(1, s), rne rounds to nearest even with NaN read as 0 (vLLM's
 * cvt.rni.sat), trunc rounds toward zero, sat8 clamps to [-128, 127], and every int8 level is an integer:
 *   0 INT8_TOKEN        s = div(amax, 127); i = amax == 0 ? 0 : div(127, amax); q = sat8(rne(x * i)); y = q * s
 *                       (no floor; vLLM w8a8/int8/scaled_quant.cu:164-176)
 *   1 INT8_STATIC       s = scale; q = sat8(rne(div(x, s))); y = q * s (scaled_quant.cu:99-116)
 *   2 INT8_STATIC_ASYM  s = scale, azp = zero_point; q = sat8(rne(x * inv(s)) + azp); y = (q - azp) * s
 *                       (scaled_quant.cu:119-139)
 *   3 INT8_GROUP128     s = div(max(amax, 1e-10f), 127); q = trunc(clamp(div(x, s), -128, 127)); y = q * s
 *                       (run_len 128; the Triton cast of vLLM int8_utils.py:123-125 truncates)
 *   4 FP8_TOKEN         ub = scale (+inf: none); s = max(div(min(amax, ub), 448), 1/(448*512) as a float);
 *                       y = e4m3(div(x, s)) * s (vLLM utils.cuh:50-54, w8a8/fp8/common.cu:204-243)
 *   5 FP8_STATIC        s = scale; y = e4m3(x * inv(s)) * s (common.cu:184-201)
 *   6 FP8_GROUP128      s = div(max(amax, 1e-10f), 448); y = e4m3(x * inv(s)) * s
 *                       (run_len 128; vLLM w8a8/fp8/per_token_group_quant.cu:46-90)
 *   7 NVFP4             g = global_scale; s16 = e4m3(div(div(amax, 6), g)); y = e2m1(div(x, s16 * g)) * s16 * g
 *                       (run_len 16; the products evaluate left to right)
 *   8 MXFP4             r = amax rounded to a 3-bit mantissa with ties to even ((bits(amax) + 0x200000) & 0x7F800000),
 *                       e = clamp(floor(log2(r)) - 2, -127, 127) with r = 0 read as 2^-126 and r = inf giving 127;
 *                       y = e2m1(x * 2^-e) * 2^e (run_len 32; vLLM mxfp4_utils.py:255-280)
 *   9 MXFP8             e = clamp(ceil(log2(div(max(amax, FLT_MIN), 448))) + 127, 0, 254) - 127, with ceil(log2)
 *                       taken exactly from the quotient's exponent and mantissa bits;
 *                       y = e4m3(x * 2^-e) * 2^e (run_len 32; vLLM mxfp8_utils.py:55-70)
 * A dequant multiplier s (or s16 * g) of 0 gives y = +0. e4m3(v) is the value of the round-to-nearest-even code of
 * clamp(v, -448, 448), bit-exact with utils.numeric.Fp8Codec.E4M3FN encode/decode (the infinities saturate to
 * +-448; NaN gives NaN), and e2m1(v) that of gguf.quant.FpBlockDecoders fp32ToE2m1/e2m1ToFp32 (saturating at 6; NaN
 * and the infinities give 6 by sign; negative zero gives +0). Scaling by 2^-e and 2^e is exact (ldexp). Results
 * equal the host's float arithmetic (gguf.quant.ActFakeQuant) for normal operands; on a device that flushes
 * subnormals the subnormal intermediates differ.
 * Any other scheme code copies x to y.
 *
 * Preconditions (validated on the host): run_len > 0 and run_len divides the row length (the scheme's run length,
 * or the whole row for the per-token and static schemes), so n_runs = rows * row_len / run_len and the runs tile the
 * contiguous rows; scale > 0 and finite for the static schemes, scale > 0 (+inf for none) for FP8_TOKEN, and
 * global_scale > 0 and finite for NVFP4. y may alias x only in place (same buffer, x_off == y_off): every work-item
 * writes exactly the elements it read, and the dynamic schemes read the whole run before the reduction barrier that
 * precedes every write.
 *
 * Launch: 1-D, one workgroup per run; global = n_runs*L, local = L (a power of two <= LLM_DIMX; CUDA: a multiple
 * of 32). Work-items stride over the run, so any such L serves any run length.
 *
 * LLM_PermuteCols(i n_rows, i row_len, i x_off, i y_off, R perm, Rn x, Wn y)
 *
 * Gathers the columns of n_rows contiguous rows of row_len elements:
 *     y[y_off + r*row_len + k] = x[x_off + r*row_len + perm[k]],  r < n_rows, k < row_len
 * (the GPTQ / compressed-tensors act-order input permutation <base>.input_perm, so that the weight stored in
 * permuted K order sees its inputs in the same order).
 *
 * Preconditions (validated on the host): perm is a permutation of [0, row_len); y does not overlap x (out of place).
 *
 * Launch: 1-D over the n_rows*row_len elements, global rounded up to the local size; the tail is bounds-checked.
 */

#define NF_AQ_INT8_TOKEN       0
#define NF_AQ_INT8_STATIC      1
#define NF_AQ_INT8_STATIC_ASYM 2
#define NF_AQ_INT8_GROUP128    3
#define NF_AQ_FP8_TOKEN        4
#define NF_AQ_FP8_STATIC       5
#define NF_AQ_FP8_GROUP128     6
#define NF_AQ_NVFP4            7
#define NF_AQ_MXFP4            8
#define NF_AQ_MXFP8            9

#define NF_AQ_INT8_MAX 127.0f
#define NF_AQ_FP8_MAX  448.0f
#define NF_AQ_FP4_MAX  6.0f

/* 1e-10f, the amax floor of the per-group schemes (vLLM's eps). */
#define NF_AQ_GROUP_EPS_BITS      0x2EDBE6FFu
/* 1 / (448 * 512) rounded to float, the FP8_TOKEN scale floor (vLLM min_scaling_factor). */
#define NF_AQ_FP8_MIN_SCALE_BITS  0x36924925u
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

/* rne(v): v rounded to nearest even, NaN read as 0 (cvt.rni.sat converts NaN to the integer 0). */
NF_INLINE float nf_aq_rne(float v) {
    return isnan(v) ? 0.0f : rint(v);
}

/* (q - azp) * s for the int8 level q = sat8(level + azp); integer arithmetic, so a zero q - azp gives +0. */
NF_INLINE float nf_aq_int8(float level, int azp, float s) {
    const int q = (int)clamp(level + (float)azp, -128.0f, NF_AQ_INT8_MAX);
    return (float)(q - azp) * s;
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

/* True for the schemes whose run parameters derive from max|x| over the run. */
NF_INLINE int nf_aq_is_dynamic(int scheme) {
    return scheme != NF_AQ_INT8_STATIC && scheme != NF_AQ_INT8_STATIC_ASYM && scheme != NF_AQ_FP8_STATIC;
}

/*
 * Fake-quantized value of v under scheme with the run parameters: s (the int8/fp8 dequant multiplier, NVFP4 s16),
 * inv (the reciprocal multiplier of INT8_TOKEN, INT8_STATIC_ASYM, FP8_STATIC and FP8_GROUP128), g (NVFP4 global
 * scale), e (MX exponent) and azp (asymmetric int8 zero point, 0 otherwise).
 */
NF_INLINE float nf_aq_fake(float v, int scheme, float s, float inv, float g, int e, int azp) {
    switch (scheme) {
    case NF_AQ_INT8_TOKEN:
        return s == 0.0f ? 0.0f : nf_aq_int8(nf_aq_rne(v * inv), 0, s);
    case NF_AQ_INT8_STATIC:
        return s == 0.0f ? 0.0f : nf_aq_int8(nf_aq_rne(nf_aq_div(v, s)), 0, s);
    case NF_AQ_INT8_STATIC_ASYM:
        return s == 0.0f ? 0.0f : nf_aq_int8(nf_aq_rne(v * inv), azp, s);
    case NF_AQ_INT8_GROUP128:
        return s == 0.0f ? 0.0f : (float)(int)clamp(nf_aq_div(v, s), -128.0f, NF_AQ_INT8_MAX) * s;
    case NF_AQ_FP8_TOKEN:
        return s == 0.0f ? 0.0f : nf_aq_e4m3(nf_aq_div(v, s)) * s;
    case NF_AQ_FP8_STATIC:
    case NF_AQ_FP8_GROUP128:
        return s == 0.0f ? 0.0f : nf_aq_e4m3(v * inv) * s;
    case NF_AQ_NVFP4: {
        const float d = s * g;
        if (d == 0.0f) {
            return 0.0f;
        }
        return nf_aq_e2m1(nf_aq_div(v, d)) * s * g;
    }
    case NF_AQ_MXFP4:
        return ldexp(nf_aq_e2m1(ldexp(v, -e)), e);
    case NF_AQ_MXFP8:
        return ldexp(nf_aq_e4m3(ldexp(v, -e)), e);
    default:
        return v;
    }
}

NF_KERNEL void LLM_ActFakeQuant(
    const int n_runs, const int run_len, const int scheme, const float scale, const int zero_point,
    const float global_scale, const int x_off, const int y_off,
    NF_GLOBAL const float* x,
    NF_GLOBAL float* y) {
    NF_LOCAL float buf[LLM_DIMX];
    const int run = get_group_id(0);
    /* Uniform per workgroup: every work-item of a dead group leaves before any barrier. */
    if (run >= n_runs) return;
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const nf_i64 xb = (nf_i64)x_off + (nf_i64)run * run_len;
    const nf_i64 yb = (nf_i64)y_off + (nf_i64)run * run_len;

    float amax = 0.0f;
    if (nf_aq_is_dynamic(scheme)) {
        float m = 0.0f;
        for (int i = lid; i < run_len; i += lsize) {
            m = fmax(m, fabs(x[xb + i]));
        }
        amax = nf_wg_max(buf, m);
    }

    float s = scale;
    float inv = 0.0f;
    float g = 0.0f;
    int azp = 0;
    int e = 0;
    switch (scheme) {
    case NF_AQ_INT8_TOKEN:
        s = nf_aq_div(amax, NF_AQ_INT8_MAX);
        inv = amax == 0.0f ? 0.0f : nf_aq_div(NF_AQ_INT8_MAX, amax);
        break;
    case NF_AQ_INT8_STATIC_ASYM:
        inv = nf_aq_div(1.0f, s);
        azp = zero_point;
        break;
    case NF_AQ_INT8_GROUP128:
        s = nf_aq_div(fmax(amax, as_float(NF_AQ_GROUP_EPS_BITS)), NF_AQ_INT8_MAX);
        break;
    case NF_AQ_FP8_TOKEN:
        s = fmax(nf_aq_div(fmin(amax, scale), NF_AQ_FP8_MAX), as_float(NF_AQ_FP8_MIN_SCALE_BITS));
        break;
    case NF_AQ_FP8_STATIC:
        inv = nf_aq_div(1.0f, s);
        break;
    case NF_AQ_FP8_GROUP128:
        s = nf_aq_div(fmax(amax, as_float(NF_AQ_GROUP_EPS_BITS)), NF_AQ_FP8_MAX);
        inv = nf_aq_div(1.0f, s);
        break;
    case NF_AQ_NVFP4:
        g = global_scale;
        s = nf_aq_e4m3(nf_aq_div(nf_aq_div(amax, NF_AQ_FP4_MAX), g));
        break;
    case NF_AQ_MXFP4:
        e = nf_aq_mxfp4_exp(amax);
        break;
    case NF_AQ_MXFP8:
        e = nf_aq_mxfp8_exp(amax);
        break;
    default:
        break;
    }

    for (int i = lid; i < run_len; i += lsize) {
        y[yb + i] = nf_aq_fake(x[xb + i], scheme, s, inv, g, e, azp);
    }
}

NF_KERNEL void LLM_PermuteCols(
    const int n_rows, const int row_len, const int x_off, const int y_off,
    NF_GLOBAL const int* NF_RESTRICT perm,
    NF_GLOBAL const float* x,
    NF_GLOBAL float* y) {
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (i >= (nf_i64)n_rows * row_len) return;
    const nf_i64 r = i / row_len;
    const int k = (int)(i - r * row_len);
    y[(nf_i64)y_off + i] = x[(nf_i64)x_off + r * row_len + perm[k]];
}
