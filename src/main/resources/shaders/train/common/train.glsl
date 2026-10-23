/* common/train.glsl - shared prelude of every train compute shader: Java float semantics, fixed-order
 * workgroup reductions, strided index decode and half-precision bit conversions. GLSL twin of
 * kernels/train/common/train.h (the semantic reference); every helper there exists here under the same name
 * with the same result. It also holds the GLSL-only transcendentals tr_tanh and tr_sincos, which every train
 * shader uses in place of builtins that Vulkan bounds too loosely (the OpenCL and CUDA kernels call the dialect
 * builtins). Requires: nothing; include it first, right after #version and GL_GOOGLE_include_directive. It
 * includes common/nf_base.glsl (resolved through the shaders/ include root).
 *
 * Shader conventions:
 *   - Buffers take bindings 0.. in the .cl parameter order; scalars fill one push-constant block in the
 *     .cl parameter order, one 4-byte int, uint or float member each (member s at offset 4*s). Arrays are
 *     never push members: a strided kernel declares d0..d5 and builds a local int[6].
 *   - Every input-only buffer is readonly, so binding the shared placeholder adds no hazard barrier.
 *   - 1-D shaders read their position only through nf_global_id()/nf_wg_id() (or tr_gid()), because the
 *     backend folds a large 1-D group count into two dimensions; only the multi-dimensional GEMM reads the
 *     built-in global invocation and workgroup ids directly.
 *   - Local size {TR_LOCAL, 1, 1} for every 1-D shader, set through specialization constants 0..2.
 *   - No 8/16-bit storage or arithmetic types and no subgroup operations: BOOL, FLOAT16 and BFLOAT16 data
 *     are read and written as 32-bit words (tr_word_byte, tr_word_half).
 *   - Accumulators of sums are declared precise, so no implementation contracts them into fused
 *     multiply-adds; multiply-add accumulations call fma() explicitly.
 */
#ifndef TR_TRAIN_GLSL
#define TR_TRAIN_GLSL

#include "common/nf_base.glsl"

/* Local size of every 1-D train shader and width of the workgroup reduction tree. */
#define TR_LOCAL    256
/* Dim and stride slots of a strided view; unused slots carry dim 1 and stride 0. */
#define TR_MAX_RANK 6

/* Workgroup reduction scratch (GLSL cannot pass shared arrays to functions). */
shared float tr_red[TR_LOCAL];
shared int tr_ired[TR_LOCAL];

/* Linear index of the calling invocation in a 1-D launch over n items, or -1 for the tail of the launch
 * (group count rounded up, or folded) and for n <= 0. */
int tr_gid(const int n) {
    const uint g = nf_global_id();
    return (n > 0 && g < uint(n)) ? int(g) : -1;
}

/* Java Float.isNaN, decided on the bit pattern. */
bool tr_isnan(const float x) {
    return (floatBitsToUint(x) & 0x7fffffffu) > 0x7f800000u;
}

/* True when x is +0 or -0 by bit pattern (a denormal is not zero, even where arithmetic flushes it). */
bool tr_iszero(const float x) {
    return (floatBitsToUint(x) & 0x7fffffffu) == 0u;
}

/* Total-order key of a non-NaN float: key(a) < key(b) exactly when a < b, except that -0 orders below +0. */
int tr_order_key(const float x) {
    const int i = floatBitsToInt(x);
    return i < 0 ? (i ^ 0x7fffffff) : i;
}

/* Java Math.max(float, float): NaN when either operand is NaN (a's NaN first), +0 above -0. */
float tr_jmax(const float a, const float b) {
    if (tr_isnan(a)) {
        return a;
    }
    if (tr_isnan(b)) {
        return b;
    }
    if (tr_iszero(a) && tr_iszero(b)) {
        return floatBitsToUint(a) == 0x80000000u ? b : a;
    }
    return tr_order_key(a) >= tr_order_key(b) ? a : b;
}

/* Java Math.min(float, float): NaN when either operand is NaN (a's NaN first), -0 below +0. */
float tr_jmin(const float a, const float b) {
    if (tr_isnan(a)) {
        return a;
    }
    if (tr_isnan(b)) {
        return b;
    }
    if (tr_iszero(a) && tr_iszero(b)) {
        return floatBitsToUint(b) == 0x80000000u ? b : a;
    }
    return tr_order_key(a) <= tr_order_key(b) ? a : b;
}

/*
 * Workgroup reductions over tr_red / tr_ired. Preconditions: local size {TR_LOCAL, 1, 1}; every invocation
 * calls the function in uniform control flow (inactive invocations pass the identity: 0 for sum, -inf for
 * max, +inf for min, 0x7fffffff for imin). Invocation l combines slot l with slot l + s for
 * s = 128, 64, ..., 1, exactly as train.h. Every invocation receives the result, and the scratch is free for
 * reuse on return.
 */
float tr_wg_sum(const float v) {
    const int lid = int(gl_LocalInvocationID.x);
    tr_red[lid] = v;
    barrier();
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            precise float t = tr_red[lid] + tr_red[lid + s];
            tr_red[lid] = t;
        }
        barrier();
    }
    const float r = tr_red[0];
    barrier();
    return r;
}

float tr_wg_max(const float v) {
    const int lid = int(gl_LocalInvocationID.x);
    tr_red[lid] = v;
    barrier();
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            tr_red[lid] = tr_jmax(tr_red[lid], tr_red[lid + s]);
        }
        barrier();
    }
    const float r = tr_red[0];
    barrier();
    return r;
}

float tr_wg_min(const float v) {
    const int lid = int(gl_LocalInvocationID.x);
    tr_red[lid] = v;
    barrier();
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            tr_red[lid] = tr_jmin(tr_red[lid], tr_red[lid + s]);
        }
        barrier();
    }
    const float r = tr_red[0];
    barrier();
    return r;
}

int tr_wg_imin(const int v) {
    const int lid = int(gl_LocalInvocationID.x);
    tr_ired[lid] = v;
    barrier();
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            const int o = tr_ired[lid + s];
            if (o < tr_ired[lid]) {
                tr_ired[lid] = o;
            }
        }
        barrier();
    }
    const int r = tr_ired[0];
    barrier();
    return r;
}

/* Element offset of row-major linear index i (last dim fastest) in a view with dims d[0..rank) and strides
 * s[0..rank). Preconditions: 0 <= i < product of the dims, every dim >= 1, rank <= TR_MAX_RANK. */
int tr_coord_offset(int i, const int rank, const int d[TR_MAX_RANK], const int s[TR_MAX_RANK]) {
    int off = 0;
    for (int k = min(rank, TR_MAX_RANK) - 1; k >= 0; --k) {
        const int c = i % d[k];
        i = i / d[k];
        off += c * s[k];
    }
    return off;
}

/* Byte byte_index & 3 of a little-endian 32-bit word (BOOL storage: element e lives in word e >> 2). */
uint tr_word_byte(const uint word, const int byte_index) {
    return (word >> (uint(byte_index & 3) * 8u)) & 0xffu;
}

/* Half half_index & 1 of a little-endian 32-bit word (FLOAT16/BFLOAT16 storage: element e lives in word e >> 1). */
uint tr_word_half(const uint word, const int half_index) {
    return (word >> (uint(half_index & 1) * 16u)) & 0xffffu;
}

/*
 * Half-precision conversions on bit patterns, the integer algorithm of tensor/HalfFloats.java and of
 * train.h (see there for the contract). Arguments and results are raw bits: the float side as a uint,
 * the 16-bit side in the low 16 bits.
 */
uint tr_f32_to_f16_bits(const uint x) {
    const uint sign = (x >> 16) & 0x8000u;
    const uint ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) {
        return sign | 0x7e00u;
    }
    if (ax >= 0x477ff000u) {
        return sign | 0x7c00u;
    }
    if (ax >= 0x38800000u) {
        const uint m = ax - 0x38000000u;
        return sign | ((m + 0x0fffu + ((m >> 13) & 1u)) >> 13);
    }
    if (ax <= 0x33000000u) {
        return sign;
    }
    const uint e = ax >> 23;
    const uint mant = (ax & 0x7fffffu) | 0x800000u;
    const uint shift = 126u - e;
    uint h = mant >> shift;
    const uint rem = mant & ((1u << shift) - 1u);
    const uint halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) {
        h += 1u;
    }
    return sign | h;
}

uint tr_f16_bits_to_f32(const uint h) {
    const uint sign = (h & 0x8000u) << 16;
    const uint e = (h >> 10) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0x1fu) {
        return sign | 0x7f800000u | (m << 13);
    }
    if (e != 0u) {
        return sign | ((e + 112u) << 23) | (m << 13);
    }
    if (m == 0u) {
        return sign;
    }
    uint fe = 113u;
    while ((m & 0x400u) == 0u) {
        m <<= 1;
        fe -= 1u;
    }
    return sign | (fe << 23) | ((m & 0x3ffu) << 13);
}

uint tr_f32_to_bf16_bits(const uint x) {
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        return ((x >> 16) & 0x8000u) | 0x7fc0u;
    }
    return (x + 0x7fffu + ((x >> 16) & 1u)) >> 16;
}

uint tr_bf16_bits_to_f32(const uint b) {
    return (b & 0xffffu) << 16;
}

/* ---- GLSL-only transcendentals ---- */

/* Java Float.isInfinite, decided on the bit pattern. */
bool tr_isinf(const float x) {
    return (floatBitsToUint(x) & 0x7fffffffu) == 0x7f800000u;
}

/* tanh(x): NaN passes; |x| < 1/8 by its odd series to x^7 (truncation below 2e-9 relative), which avoids the
 * cancellation of the exponential form near 0; otherwise sign(x) * (1 - 2 / (e^(2|x|) + 1)), exactly +-1 once
 * e^(2|x|) overflows. The builtin may overflow to NaN for large |x| on some drivers. */
float tr_tanh(const float x) {
    if (tr_isnan(x)) {
        return x;
    }
    const float ax = abs(x);
    if (ax < 0.125) {
        precise float x2 = x * x;
        precise float r = x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0))));
        return r;
    }
    const float twice = 2.0 * ax;
    const float e = tr_isinf(twice) ? NF_INF : exp(twice);
    precise float t = 1.0 - 2.0 / (e + 1.0);
    return floatBitsToInt(x) < 0 ? -t : t;
}

/* 2/pi as a binary fraction, 32 bits per word from the binary point on, behind one zero word: the bits that
 * tr_reduce_large multiplies an argument's significand by. */
const uint tr_two_over_pi[8] = uint[8](0x00000000u, 0xa2f9836eu, 0x4e441529u, 0xfc2757d1u, 0xf534ddc0u,
                                        0xdb629599u, 0x3c439041u, 0xfe5163abu);

/* The 32 bits of tr_two_over_pi that start shift (0..31) bits into word (0..6). */
uint tr_two_over_pi_bits(const int word, const int shift) {
    if (shift == 0) {
        return tr_two_over_pi[word];
    }
    return (tr_two_over_pi[word] << uint(shift)) | (tr_two_over_pi[word + 1] >> uint(32 - shift));
}

/* Cody-Waite reduction of 0 <= ax < 65536: ax = k * pi/2 + r with k = floor(ax * 2/pi + 0.5) < 2^16 and
 * |r| < 0.787. pi/2 is split into three parts of 8 significant bits (k * part is exact) and a rounded fourth
 * part; together they hold pi/2 to 5e-17, so r carries at most 2^-24 |r| plus a few ulp of rounding. */
float tr_reduce_small(const float ax, out int quadrant) {
    precise float k = floor(ax * 0.636619747 + 0.5);
    quadrant = int(k) & 3;
    precise float r = ax - k * 1.5703125;
    r = r - k * 4.825592041015625e-4;
    r = r - k * 1.26659870147705078125e-6;
    r = r - k * 9.92093629470503e-10;
    return r;
}

/* Payne-Hanek reduction of a finite ax >= 65536, exact in integer arithmetic: ax = m * 2^e with the 24-bit
 * significand m and e >= -7. Only the bits of 2^e * 2/pi from weight 2^1 down matter modulo 4; 96 of them
 * (dropping a tail below 2^-70 after the product) times m give the quadrant in bits 31..30 of the third
 * product word and the fraction below, which is rounded to the nearest quadrant, |f| <= 1/2, and scaled by
 * pi/2. */
float tr_reduce_large(const float ax, out int quadrant) {
    const uint b = floatBitsToUint(ax);
    const int e = int(b >> 23) - 150;
    const uint m = (b & 0x7fffffu) | 0x800000u;
    const int p = 30 + e;
    const uint w0 = tr_two_over_pi_bits(p >> 5, p & 31);
    const uint w1 = tr_two_over_pi_bits((p >> 5) + 1, p & 31);
    const uint w2 = tr_two_over_pi_bits((p >> 5) + 2, p & 31);
    uint hi0, lo0, hi1, lo1, hi2, lo2, carry;
    umulExtended(m, w0, hi0, lo0);
    umulExtended(m, w1, hi1, lo1);
    umulExtended(m, w2, hi2, lo2);
    const uint p1 = uaddCarry(hi2, lo1, carry);
    const uint p2 = hi1 + lo0 + carry;
    int q = int(p2 >> 30);
    int u = int(p2 & 0x3fffffffu);
    if (u >= 0x20000000) {
        u -= 0x40000000;
        q += 1;
    }
    quadrant = q & 3;
    precise float f = ldexp(float(u), -30) + ldexp(float(p1), -62);
    precise float r = f * 1.57079637;
    return r;
}

/* sin(x) and cos(x) as Java (float)Math.sin and (float)Math.cos. NaN and +-inf give NaN for both and sin keeps
 * the sign of zero. |x| = k * pi/2 + r with |r| < 0.787 (tr_reduce_small below 65536, tr_reduce_large above),
 * then the Taylor polynomials of sin to r^9 and cos to r^10 (truncation below 3e-9 on |r| < 0.8) evaluated by
 * Horner's rule in r^2, and the quadrant k mod 4 selects +-sin(r) or +-cos(r). Both results are within 2^-22 of
 * the exact values for every finite x, inside the transcendental class; the Vulkan builtins are bounded only to
 * 2^-11 absolute on [-pi, pi]. */
void tr_sincos(const float x, out float s, out float c) {
    if (tr_isnan(x) || tr_isinf(x)) {
        s = NF_NAN;
        c = NF_NAN;
        return;
    }
    const float ax = abs(x);
    int q;
    float r;
    if (ax < 65536.0) {
        r = tr_reduce_small(ax, q);
    } else {
        r = tr_reduce_large(ax, q);
    }
    precise float r2 = r * r;
    precise float sr = r + (r * r2) * (-0.166666672 + r2 * (0.00833333377 + r2 * (-0.000198412701
                     + r2 * 2.75573188e-06)));
    precise float cr = 1.0 + r2 * (-0.5 + r2 * (0.0416666679 + r2 * (-0.00138888892 + r2 * (2.48015876e-05
                     + r2 * (-2.755732e-07)))));
    c = q == 0 ? cr : (q == 1 ? -sr : (q == 2 ? -cr : sr));
    const float v = q == 0 ? sr : (q == 1 ? cr : (q == 2 ? -sr : -cr));
    s = floatBitsToInt(x) < 0 ? -v : v;
}

/*
 * Deferred index errors (train.h tr_report_index_error): slot s is err[4*s .. 4*s+3] = {flag, index, dim, size};
 * the first failing invocation claims the flag by atomicCompSwap and alone writes the other three words. A
 * macro, because GLSL cannot pass a buffer to a function; ERR is a read-write int buffer array.
 */
#define tr_report_index_error(ERR, SLOT, INDEX, DIM, SIZE)                \
    do {                                                                  \
        if (atomicCompSwap(ERR[4 * (SLOT)], 0, 1) == 0) {                 \
            ERR[4 * (SLOT) + 1] = (INDEX);                                \
            ERR[4 * (SLOT) + 2] = (DIM);                                  \
            ERR[4 * (SLOT) + 3] = (SIZE);                                 \
        }                                                                 \
    } while (false)

#endif /* TR_TRAIN_GLSL */
