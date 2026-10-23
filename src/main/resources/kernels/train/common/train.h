/* common/train.h - shared helpers of the train kernel library: Java float semantics, fixed-order workgroup reductions, strided index decode and half-precision bit conversions. Requires: common/dialect.h */
/*
 * Numerical contract (every train kernel):
 *   - no floating-point contraction: OpenCL FP_CONTRACT is off here and the library is built without
 *     -cl-mad-enable; CUDA builds with -fmad=false. Multiply-add accumulations call fma() explicitly;
 *   - Java Math.max/Math.min semantics through tr_jmax/tr_jmin, never fmax/fmin/clamp;
 *   - reductions run over a fixed 256-wide local-memory tree with fixed pairing, so the summation order
 *     is the same on every backend and device (no subgroup operations, no float atomics).
 *
 * Index arithmetic is int: the host guarantees every buffer holds at most 2^31-1 elements and every stride
 * is non-negative. The GLSL twin is shaders/train/common/train.glsl; every helper here exists there under
 * the same name with the same result.
 */
#ifndef TR_TRAIN_H
#define TR_TRAIN_H

#ifndef NF_CUDA
#pragma OPENCL FP_CONTRACT OFF
#endif

/* Local size of every 1-D train kernel and width of the workgroup reduction tree. */
#define TR_LOCAL    256
/* Dim and stride slots of a strided view; unused slots carry dim 1 and stride 0. */
#define TR_MAX_RANK 6

/* Linear index of the calling work-item in a 1-D launch over n items, or -1 for the tail of the launch
 * (global size rounded up to the local size) and for n <= 0. */
NF_INLINE int tr_gid(const int n) {
    const size_t g = get_global_id(0);
    return (n > 0 && g < (size_t)n) ? (int)g : -1;
}

/* Java Float.isNaN, decided on the bit pattern so that no compiler NaN assumption applies. */
NF_INLINE int tr_isnan(const float x) {
    return (as_uint(x) & 0x7fffffffu) > 0x7f800000u;
}

/* True when x is +0 or -0 by bit pattern (a denormal is not zero, even where arithmetic flushes it). */
NF_INLINE int tr_iszero(const float x) {
    return (as_uint(x) & 0x7fffffffu) == 0u;
}

/* Total-order key of a non-NaN float: key(a) < key(b) exactly when a < b, except that -0 orders below +0.
 * Integer compares on these keys stay exact on devices that flush denormals in float comparisons. */
NF_INLINE int tr_order_key(const float x) {
    const int i = as_int(x);
    return i < 0 ? (i ^ 0x7fffffff) : i;
}

/* Java Math.max(float, float): NaN when either operand is NaN (a's NaN first), +0 above -0. */
NF_INLINE float tr_jmax(const float a, const float b) {
    if (tr_isnan(a)) return a;
    if (tr_isnan(b)) return b;
    if (tr_iszero(a) && tr_iszero(b)) return (as_uint(a) == 0x80000000u) ? b : a;
    return tr_order_key(a) >= tr_order_key(b) ? a : b;
}

/* Java Math.min(float, float): NaN when either operand is NaN (a's NaN first), -0 below +0. */
NF_INLINE float tr_jmin(const float a, const float b) {
    if (tr_isnan(a)) return a;
    if (tr_isnan(b)) return b;
    if (tr_iszero(a) && tr_iszero(b)) return (as_uint(b) == 0x80000000u) ? b : a;
    return tr_order_key(a) <= tr_order_key(b) ? a : b;
}

/*
 * Workgroup reductions. Preconditions: local size {TR_LOCAL, 1, 1}; every work-item calls the function in
 * uniform control flow (inactive items pass the identity: 0 for sum, -inf for max, +inf for min, INT_MAX
 * for imin); buf is a kernel-scope NF_LOCAL array of TR_LOCAL elements. Item l combines slot l with slot
 * l + s for s = 128, 64, ..., 1, so the order is fixed. Every item receives the result, and buf is free
 * for reuse on return (a barrier follows the final read).
 */
NF_INLINE float tr_wg_sum(NF_LOCAL_PTR float* buf, const float v) {
    const int lid = (int)get_local_id(0);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) buf[lid] = buf[lid] + buf[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

NF_INLINE float tr_wg_max(NF_LOCAL_PTR float* buf, const float v) {
    const int lid = (int)get_local_id(0);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) buf[lid] = tr_jmax(buf[lid], buf[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

NF_INLINE float tr_wg_min(NF_LOCAL_PTR float* buf, const float v) {
    const int lid = (int)get_local_id(0);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) buf[lid] = tr_jmin(buf[lid], buf[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

NF_INLINE int tr_wg_imin(NF_LOCAL_PTR int* buf, const int v) {
    const int lid = (int)get_local_id(0);
    buf[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = TR_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            const int o = buf[lid + s];
            if (o < buf[lid]) buf[lid] = o;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int r = buf[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

/* Element offset of row-major linear index i (last dim fastest) in a view with dims d[0..rank) and strides
 * s[0..rank). Preconditions: 0 <= i < product of the dims, every dim >= 1, rank <= TR_MAX_RANK. */
NF_INLINE int tr_coord_offset(int i, const int rank, const int d[TR_MAX_RANK], const int s[TR_MAX_RANK]) {
    int off = 0;
    for (int k = (rank < TR_MAX_RANK ? rank : TR_MAX_RANK) - 1; k >= 0; --k) {
        const int c = i % d[k];
        i = i / d[k];
        off += c * s[k];
    }
    return off;
}

/* Byte byte_index & 3 of a little-endian 32-bit word (BOOL storage: element e lives in word e >> 2). */
NF_INLINE uint tr_word_byte(const uint word, const int byte_index) {
    return (word >> ((uint)(byte_index & 3) * 8u)) & 0xffu;
}

/* Half half_index & 1 of a little-endian 32-bit word (FLOAT16/BFLOAT16 storage: element e lives in word e >> 1). */
NF_INLINE uint tr_word_half(const uint word, const int half_index) {
    return (word >> ((uint)(half_index & 1) * 16u)) & 0xffffu;
}

/*
 * Half-precision conversions on bit patterns, the integer algorithm of tensor/HalfFloats.java, so device
 * and host casts are bit-identical and no float arithmetic (or denormal flushing) is involved.
 *   float -> FLOAT16/BFLOAT16: round to nearest even; overflow to +-inf; subnormal results preserved;
 *     NaN -> quiet NaN keeping the sign (0x7E00|s for FLOAT16, 0x7FC0|s for BFLOAT16).
 *   FLOAT16/BFLOAT16 -> float: exact, NaN payload and sign kept.
 * Arguments and results are raw bits: the float side as a uint, the 16-bit side in the low 16 bits.
 */
NF_INLINE uint tr_f32_to_f16_bits(const uint x) {
    const uint sign = (x >> 16) & 0x8000u;
    const uint ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) return sign | 0x7e00u;
    if (ax >= 0x477ff000u) return sign | 0x7c00u;          /* >= 65520 rounds to infinity */
    if (ax >= 0x38800000u) {                                 /* normal half: |x| >= 2^-14 */
        const uint m = ax - 0x38000000u;                     /* rebias exponent 127 -> 15 */
        return sign | ((m + 0x0fffu + ((m >> 13) & 1u)) >> 13);
    }
    if (ax <= 0x33000000u) return sign;                      /* |x| <= 2^-25 rounds to zero */
    const uint e = ax >> 23;                                 /* 102..112 */
    const uint mant = (ax & 0x7fffffu) | 0x800000u;
    const uint shift = 126u - e;                             /* 14..24: value in units of 2^-24 */
    uint h = mant >> shift;
    const uint rem = mant & ((1u << shift) - 1u);
    const uint halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) h += 1u;
    return sign | h;
}

NF_INLINE uint tr_f16_bits_to_f32(const uint h) {
    const uint sign = (h & 0x8000u) << 16;
    const uint e = (h >> 10) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0x1fu) return sign | 0x7f800000u | (m << 13);
    if (e != 0u) return sign | ((e + 112u) << 23) | (m << 13);
    if (m == 0u) return sign;
    uint fe = 113u;                                          /* normalize the subnormal significand */
    while ((m & 0x400u) == 0u) {
        m <<= 1;
        fe -= 1u;
    }
    return sign | (fe << 23) | ((m & 0x3ffu) << 13);
}

NF_INLINE uint tr_f32_to_bf16_bits(const uint x) {
    if ((x & 0x7fffffffu) > 0x7f800000u) return ((x >> 16) & 0x8000u) | 0x7fc0u;
    return (x + 0x7fffu + ((x >> 16) & 1u)) >> 16;
}

NF_INLINE uint tr_bf16_bits_to_f32(const uint b) {
    return (b & 0xffffu) << 16;
}

/*
 * Deferred index errors (tensor/gpu/TrainGpu): slot s of the error buffer is the four ints
 * err[4*s .. 4*s+3] = {flag, index, dim, size}, zeroed by the host after every read. The first failing
 * work-item since the last read claims the slot by an atomic compare-and-swap of the flag and alone writes
 * the out-of-range index, the dimension and the size the host reports, so the first failing launch keeps the
 * slot and every word comes from one work-item.
 */
NF_INLINE void tr_report_index_error(NF_GLOBAL int* err, const int slot, const int index, const int dim,
                                     const int size) {
    if (atomic_cmpxchg(&err[4 * slot], 0, 1) == 0) {
        err[4 * slot + 1] = index;
        err[4 * slot + 2] = dim;
        err[4 * slot + 3] = size;
    }
}

#endif /* TR_TRAIN_H */
