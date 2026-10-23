/* eltwise.cl - elementwise unary and binary ops, axpby, scale, compare, where, gather, scatter, unary backward and dropout apply. Requires: common/dialect.h, common/train.h */
/*
 * Every formula is the CPU reference of the Java code it replaces, evaluated in the same operation order
 * (tensor/CpuElementwise.java, tensor/CpuIndexing.java, nn/layers/Dropout.java). Offsets are element offsets
 * unless named *_bytes. Strided operands take rank (<= TR_MAX_RANK) and six dim and six stride slots; unused
 * slots carry dim 1 and stride 0, and a broadcast dim carries stride 0. Comparisons that decide a branch
 * (x > 0, x < 0, a < b, a == b) are made on bit patterns with Java semantics, so they hold on devices that
 * flush denormals in float comparisons.
 *
 * TR_Unary_F32(i op, i n, i rank, i d0..d5, i s0..s5, i in_off, i out_off, f p0, f p1, R in, W out)
 *     out[out_off + i] = f_op(in[in_off + offset(i, s)]), i < n, offset as tr_coord_offset over d.
 *     Ops (TR_E_*): NEG, ABS, SQRT, EXP, LOG, RELU (Math.max(0, x)), SIGMOID, TANH, GELU (tanh form),
 *     SILU, LEAKY_RELU (slope p0), ELU (alpha p0), SOFTPLUS, MISH, POW (exponent p0, Java Math.pow special
 *     cases), CLAMP (Math.max(p0, Math.min(p1, x))), ADD_SCALAR (x + p0), MUL_SCALAR (x * p0), IDENTITY,
 *     ROUND (Java Math.round(float) as a float: r = floor(x), x - r >= 0.5 ? r + 1 : r; exact for every
 *     float, NaN stays NaN and +-inf stay +-inf, so TR_Cast_F32_I32 then yields Java's int result),
 *     FLOOR, SIN, COS ((float)Math.sin(x), (float)Math.cos(x): the dialect builtins, see tr_e_unary). An unknown
 *     op writes NaN.
 * TR_Binary_F32(i op, i n, i rank, i d0..d5, i as0..as5, i bs0..bs5, i a_off, i b_off, i out_off, R a, R b, W out)
 *     out[out_off + i] = a[a_off + offset(i, as)] op b[b_off + offset(i, bs)]; ops TR_E_ADD, SUB, MUL, DIV.
 *     Broadcasting is stride 0. An unknown op writes NaN.
 * TR_Axpby_F32(i n, f alpha, f beta, i x_off, i y_off, R x, RW y)
 *     y[y_off + i] = alpha * x[x_off + i] + beta * y[y_off + i] (two roundings and a sum, no contraction);
 *     when beta is +-0, y = alpha * x and y is not read.
 * TR_Scale_F32(i n, f alpha, i off, RW y)
 *     y[off + i] = y[off + i] * alpha.
 * TR_Compare_F32(i op, i n, i rank, i d0..d5, i as0..as5, i bs0..bs5, i a_off, i b_off, R a, R b, W words out)
 *     BOOL out byte i = (a[a_off + offset(i, as)] op b[b_off + offset(i, bs)]) ? 1 : 0, Java float ==, >, <
 *     (TR_CMP_EQ, TR_CMP_GT, TR_CMP_LT): NaN compares false, -0 == +0. One work-item per output word.
 * TR_Where_F32(i cond_kind, i n, i rank, i d0..d5, i cs0..cs5, i as0..as5, i bs0..bs5, i c_off, i a_off, i b_off,
 *              R words cond, R words a, R words b, W words out)
 *     out[i] = cond(i) ? a[a_off + offset(i, as)] : b[b_off + offset(i, bs)] (bit copies, so FLOAT32 and
 *     INT32 values pass unchanged). cond_kind TR_WHERE_COND_F32: cond(i) = cond[c_off + offset(i, cs)]
 *     != 0.0f on the bit pattern (NaN and denormals are true, +-0 false; TensorOps.where).
 *     TR_WHERE_COND_BOOL: cond(i) = byte p = c_off + offset(i, cs) of the BOOL storage != 0, read from word
 *     p >> 2; c_off in bytes.
 * TR_Gather_F32(i dim, i n, i rank, i d0..d5, i id0..id5, i in_off, i idx_off, i err_slot, i err_dim,
 *               R words in, R int idx, W words out, RW int err)
 *     Index shape d (n elements), contiguous input shape id, contiguous idx and out. For index element i
 *     with coordinates c: g = idx[idx_off + i]; out[i] = in[in_off + linear(c with c[dim] = g)] over id. An
 *     index outside [0, id[dim]) writes 0 to out[i] and reports {g, err_dim, id[dim]} in slot err_slot
 *     (tr_report_index_error); err_dim is the caller's dimension, which differs from dim when the dispatcher
 *     merged dimensions. Preconditions (dispatcher): 0 <= dim < rank, d[k] <= id[k] for k != dim.
 * TR_Scatter_F32(i dim, i mode, i fibers, i rank, i d0..d5, i od0..od5, i idx_off, i src_off, i err_slot,
 *                i err_dim, R int idx, R src, RW out, RW int err)
 *     Index and src share the contiguous index shape d; out has the contiguous shape od and is pre-filled by
 *     the dispatcher. One work-item per fiber (every coordinate except dim; fibers = product of d[k],
 *     k != dim) walks j = 0 .. d[dim]-1 in ascending order: v = idx[idx_off + lin(c, j)];
 *     TR_SCATTER_OVERWRITE: out[linear(c, v)] = src[src_off + lin(c, j)] (the CPU's last write wins);
 *     TR_SCATTER_ADD: out[linear(c, v)] += src[src_off + lin(c, j)] (CpuIndexing.java scatter order, which
 *     GatherGrad uses). Two index elements that hit one output element lie in one fiber, so the result is
 *     deterministic and needs no float atomics. An index v outside [0, od[dim]) is skipped and reports
 *     {v, err_dim, od[dim]} in slot err_slot as TR_Gather_F32 does. Preconditions (dispatcher):
 *     0 <= dim < rank, d[k] <= od[k] for k != dim, mode is one of the two ids.
 * TR_UnaryBackward_F32(i op, i n, f p0, f p1, i x_off, i y_off, i gy_off, i gx_off, R x, R y, R gy, W gx)
 *     gx[gx_off + i] = gy[gy_off + i] * f'_op (tensor/CpuElementwise.java backward order). The op is the
 *     TR_Unary_F32 id of the forward function: TR_E_RELU, GELU, SILU, SIGMOID (from y), TANH (from y),
 *     LEAKY_RELU (x, slope p0), ELU (x, y, alpha p0), SOFTPLUS, MISH, ABS (gy * sgn(x) as a selection:
 *     x > 0 ? gy : x < 0 ? -gy : +0, so +-0 and NaN x give +0), CLAMP (x < p0 || x > p1 ? 0 : gy); any other id
 *     writes NaN. x is read by every op except SIGMOID and TANH, y only by SIGMOID, TANH and ELU; a buffer an op
 *     does not read may be the placeholder.
 * TR_MaskScale_F32(i n, f scale, i x_off, i m_off, i out_off, R x, R m, W out)
 *     out[out_off + i] = m == +-0 ? +0 : (x * m) * scale (dropout apply: a dropped element is 0 even for a
 *     non-finite x, Dropout.java forward; a kept element is x * keep-scale whether the mask holds the scale and
 *     scale is 1, or the mask holds 1 and scale is the keep-scale).
 *
 * Aliasing: TR_Unary, TR_Binary, TR_Axpby, TR_Scale, TR_UnaryBackward and TR_MaskScale read and write only
 * their own element, so exact in-place use (output == an input at the same offset, contiguous) is legal and
 * their pointers carry no NF_RESTRICT. Compare, where, gather and scatter outputs never alias an input.
 *
 * Dispatch: 1-D, local TR_LOCAL, global = n rounded up, except TR_Compare_F32 (ceil(n/4) items) and
 * TR_Scatter_F32 (fibers items). No barriers.
 */

/* TR_Unary_F32 op ids (tensor/gpu/TrainOps). */
#define TR_E_NEG          0
#define TR_E_ABS          1
#define TR_E_SQRT         2
#define TR_E_EXP          3
#define TR_E_LOG          4
#define TR_E_RELU         5
#define TR_E_SIGMOID      6
#define TR_E_TANH         7
#define TR_E_GELU         8
#define TR_E_SILU         9
#define TR_E_LEAKY_RELU  10
#define TR_E_ELU         11
#define TR_E_SOFTPLUS    12
#define TR_E_MISH        13
#define TR_E_POW         14
#define TR_E_CLAMP       15
#define TR_E_ADD_SCALAR  16
#define TR_E_MUL_SCALAR  17
#define TR_E_IDENTITY    18
#define TR_E_ROUND       19
#define TR_E_FLOOR       20
#define TR_E_SIN         21
#define TR_E_COS         22

/* TR_Binary_F32 op ids. */
#define TR_E_ADD          0
#define TR_E_SUB          1
#define TR_E_MUL          2
#define TR_E_DIV          3

/* TR_Compare_F32 op ids. */
#define TR_CMP_EQ         0
#define TR_CMP_GT         1
#define TR_CMP_LT         2

/* TR_Where_F32 condition kinds. */
#define TR_WHERE_COND_F32    0
#define TR_WHERE_COND_BOOL   1

/* TR_Scatter_F32 modes. */
#define TR_SCATTER_OVERWRITE 0
#define TR_SCATTER_ADD       1

/* ---- Java float predicates on bit patterns ---- */

NF_INLINE int tr_e_isinf(const float x) {
    return (as_uint(x) & 0x7fffffffu) == 0x7f800000u;
}

/* Java x > 0: positive denormals through +inf; false for zeros, negatives and NaN. */
NF_INLINE int tr_e_gtz(const float x) {
    const int i = as_int(x);
    return i > 0 && i <= 0x7f800000;
}

/* Java x < 0: negative denormals through -inf; false for zeros, positives and NaN. */
NF_INLINE int tr_e_ltz(const float x) {
    return tr_e_gtz(as_float(as_uint(x) ^ 0x80000000u));
}

/* Java a < b: false when either is NaN; -0 and +0 are equal. */
NF_INLINE int tr_e_lt(const float a, const float b) {
    if (tr_isnan(a) || tr_isnan(b)) return 0;
    if (tr_iszero(a) && tr_iszero(b)) return 0;
    return tr_order_key(a) < tr_order_key(b);
}

/* Java a == b: false when either is NaN; -0 == +0. */
NF_INLINE int tr_e_eq(const float a, const float b) {
    if (tr_isnan(a) || tr_isnan(b)) return 0;
    return as_uint(a) == as_uint(b) || (tr_iszero(a) && tr_iszero(b));
}

/* ---- scalar functions ---- */

/* Java -x: sign flip of the bit pattern. */
NF_INLINE float tr_e_neg(const float x) {
    return as_float(as_uint(x) ^ 0x80000000u);
}

/* Java Math.abs(float): sign bit cleared. */
NF_INLINE float tr_e_abs(const float x) {
    return as_float(as_uint(x) & 0x7fffffffu);
}

NF_INLINE float tr_e_tanh(const float x) {
    return tanh(x);
}

/* (float)(1.0 / (1.0 + Math.exp(-x))). For x < 0 the equal form e^x / (1 + e^x) keeps the float evaluation
 * on the double reference where e^-x would overflow (x < -88.7). */
NF_INLINE float tr_e_sigmoid(const float x) {
    if (tr_isnan(x)) return x;
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    const float e = exp(x);
    return e / (1.0f + e);
}

/* (float)Math.log(1.0 + Math.exp(x)) as max(x, 0) + log1p(e^-|x|): the same function, evaluated so that
 * float keeps the double reference where e^x overflows float (x > 88.7) or 1 + e^x rounds to 1 in float
 * (x < -17). From 709.782715f, the smallest float whose double Math.exp overflows, the reference is +inf. */
NF_INLINE float tr_e_softplus(const float x) {
    if (tr_isnan(x)) return x;
    if (x >= 709.782715f) return as_float(0x7f800000u);
    if (x > 0.0f) return x + log1p(exp(-x));
    return log1p(exp(x));
}

/* Java (float)Math.pow(x, y) with every Math.pow special case; pow() is applied only to a finite positive
 * base with a finite non-zero exponent, where all dialects agree on its meaning. The sign of y is taken
 * from its bits and only |y| >= 1 can be an integer, so a denormal y keeps Java's result on devices that
 * flush denormals in float compares. */
NF_INLINE float tr_e_pow(const float x, const float y) {
    if (tr_iszero(y)) return 1.0f;
    if (as_uint(y) == 0x3f800000u) return x;
    if (tr_isnan(x) || tr_isnan(y)) return as_float(0x7fc00000u);
    const float ax = tr_e_abs(x);
    const int ypos = tr_e_gtz(y);
    if (tr_e_isinf(y)) {
        if (ax == 1.0f) return as_float(0x7fc00000u);
        return ((ax > 1.0f) == ypos) ? as_float(0x7f800000u) : 0.0f;
    }
    const int yint = tr_e_abs(y) >= 1.0f && floor(y) == y;
    const int yodd = yint && tr_e_abs(y) < 16777216.0f && (((int)y) & 1) != 0;
    float mag;
    if (tr_iszero(ax)) {
        mag = ypos ? 0.0f : as_float(0x7f800000u);
    } else if (tr_e_isinf(ax)) {
        mag = ypos ? as_float(0x7f800000u) : 0.0f;
    } else {
        mag = pow(ax, y);
    }
    if ((as_uint(x) & 0x80000000u) != 0u) {
        if (!yint && !tr_iszero(x) && !tr_e_isinf(x)) return as_float(0x7fc00000u);
        return yodd ? tr_e_neg(mag) : mag;
    }
    return mag;
}

/* Java Math.round(float) kept as a float: x - floor(x) is exact or keeps its order against 0.5. */
NF_INLINE float tr_e_round(const float x) {
    const float r = floor(x);
    return (x - r >= 0.5f) ? r + 1.0f : r;
}

/* TensorOps.gelu: sqrtTwoOverPi * (x + 0.044715f * x * x * x), then 0.5f * x * (1 + tanh). The literals are
 * the exact floats (float)Math.sqrt(2 / PI) and 0.044715f. */
NF_INLINE float tr_e_gelu(const float x) {
    const float s = 0.797884583f;
    const float c = 0.0447149985f;
    const float inner = s * (x + ((c * x) * x) * x);
    return (0.5f * x) * (1.0f + tr_e_tanh(inner));
}

/* SIN and COS call the builtins, whose accuracy holds over the whole float range: OpenCL 1.2 sin/cos are within
 * 4 ulp, and the CUDA float overloads (sinf/cosf, built without fast math) within 2 ulp, with a full range
 * reduction for large arguments; NaN and +-inf give NaN and sin keeps the sign of zero, as Java does. Sharing the
 * GLSL twin's polynomial instead would make OpenCL and CUDA bit-identical to Vulkan but no closer to the Java
 * reference, which the transcendental tolerance class measures; the builtins are closer. */
NF_INLINE float tr_e_unary(const int op, const float x, const float p0, const float p1) {
    switch (op) {
        case TR_E_NEG:        return tr_e_neg(x);
        case TR_E_ABS:        return tr_e_abs(x);
        case TR_E_SQRT:       return sqrt(x);
        case TR_E_EXP:        return exp(x);
        case TR_E_LOG:        return log(x);
        case TR_E_RELU:       return tr_jmax(0.0f, x);
        case TR_E_SIGMOID:    return tr_e_sigmoid(x);
        case TR_E_TANH:       return tr_e_tanh(x);
        case TR_E_GELU:       return tr_e_gelu(x);
        case TR_E_SILU:       return x * tr_e_sigmoid(x);
        case TR_E_LEAKY_RELU: return tr_e_gtz(x) ? x : p0 * x;
        case TR_E_ELU:        return tr_e_gtz(x) ? x : p0 * (exp(x) - 1.0f);
        case TR_E_SOFTPLUS:   return tr_e_softplus(x);
        case TR_E_MISH:       return x * tr_e_tanh(tr_e_softplus(x));
        case TR_E_POW:        return tr_e_pow(x, p0);
        case TR_E_CLAMP:      return tr_jmax(p0, tr_jmin(p1, x));
        case TR_E_ADD_SCALAR: return x + p0;
        case TR_E_MUL_SCALAR: return x * p0;
        case TR_E_IDENTITY:   return x;
        case TR_E_ROUND:      return tr_e_round(x);
        case TR_E_FLOOR:      return floor(x);
        case TR_E_SIN:        return sin(x);
        case TR_E_COS:        return cos(x);
        default:              return as_float(0x7fc00000u);
    }
}

NF_INLINE float tr_e_binary(const int op, const float a, const float b) {
    switch (op) {
        case TR_E_ADD: return a + b;
        case TR_E_SUB: return a - b;
        case TR_E_MUL: return a * b;
        case TR_E_DIV: return a / b;
        default:       return as_float(0x7fc00000u);
    }
}

NF_INLINE int tr_e_compare(const int op, const float a, const float b) {
    switch (op) {
        case TR_CMP_EQ: return tr_e_eq(a, b);
        case TR_CMP_GT: return tr_e_lt(b, a);
        case TR_CMP_LT: return tr_e_lt(a, b);
        default:        return 0;
    }
}

/* GeluGrad: x3 = x * x * x; inner = s * (x + 0.044715f * x3); sech2 = 1 - t * t;
 * dInner = s * (1 + 3.0f * 0.044715f * x * x), where 3.0f * 0.044715f folds to the float 0.134144992f;
 * grad = 0.5f * (1 + t) + 0.5f * x * sech2 * dInner. */
NF_INLINE float tr_e_gelu_grad(const float x) {
    const float s = 0.797884583f;
    const float c = 0.0447149985f;
    const float c3 = 0.134144992f;
    const float x3 = (x * x) * x;
    const float inner = s * (x + c * x3);
    const float t = tr_e_tanh(inner);
    const float sech2 = 1.0f - t * t;
    const float d_inner = s * (1.0f + (c3 * x) * x);
    return 0.5f * (1.0f + t) + ((0.5f * x) * sech2) * d_inner;
}

/* MishGrad: tanhSp + x * sech2 * sigmoid with sp = softplus(x), sech2 = 1 - tanhSp * tanhSp. */
NF_INLINE float tr_e_mish_grad(const float x) {
    const float t = tr_e_tanh(tr_e_softplus(x));
    const float sech2 = 1.0f - t * t;
    return t + (x * sech2) * tr_e_sigmoid(x);
}

/* SiluGrad: sigmoid * (1 + x * (1 - sigmoid)). */
NF_INLINE float tr_e_silu_grad(const float x) {
    const float sg = tr_e_sigmoid(x);
    return sg * (1.0f + x * (1.0f - sg));
}

/* ---- kernels ---- */

NF_KERNEL void TR_Unary_F32(
    const int op,
    const int n,
    const int rank,
    const int d0, const int d1, const int d2, const int d3, const int d4, const int d5,
    const int s0, const int s1, const int s2, const int s3, const int s4, const int s5,
    const int in_off,
    const int out_off,
    const float p0,
    const float p1,
    NF_GLOBAL const float* in,
    NF_GLOBAL float* out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int d[TR_MAX_RANK] = { d0, d1, d2, d3, d4, d5 };
    const int s[TR_MAX_RANK] = { s0, s1, s2, s3, s4, s5 };
    const float x = in[in_off + tr_coord_offset(i, rank, d, s)];
    out[out_off + i] = tr_e_unary(op, x, p0, p1);
}

NF_KERNEL void TR_Binary_F32(
    const int op,
    const int n,
    const int rank,
    const int d0, const int d1, const int d2, const int d3, const int d4, const int d5,
    const int as0, const int as1, const int as2, const int as3, const int as4, const int as5,
    const int bs0, const int bs1, const int bs2, const int bs3, const int bs4, const int bs5,
    const int a_off,
    const int b_off,
    const int out_off,
    NF_GLOBAL const float* a,
    NF_GLOBAL const float* b,
    NF_GLOBAL float* out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int d[TR_MAX_RANK] = { d0, d1, d2, d3, d4, d5 };
    const int as[TR_MAX_RANK] = { as0, as1, as2, as3, as4, as5 };
    const int bs[TR_MAX_RANK] = { bs0, bs1, bs2, bs3, bs4, bs5 };
    const float va = a[a_off + tr_coord_offset(i, rank, d, as)];
    const float vb = b[b_off + tr_coord_offset(i, rank, d, bs)];
    out[out_off + i] = tr_e_binary(op, va, vb);
}

NF_KERNEL void TR_Axpby_F32(
    const int n,
    const float alpha,
    const float beta,
    const int x_off,
    const int y_off,
    NF_GLOBAL const float* x,
    NF_GLOBAL float* y)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float ax = alpha * x[x_off + i];
    if (tr_iszero(beta)) {
        y[y_off + i] = ax;
    } else {
        const float by = beta * y[y_off + i];
        y[y_off + i] = ax + by;
    }
}

NF_KERNEL void TR_Scale_F32(
    const int n,
    const float alpha,
    const int off,
    NF_GLOBAL float* y)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    y[off + i] = y[off + i] * alpha;
}

NF_KERNEL void TR_Compare_F32(
    const int op,
    const int n,
    const int rank,
    const int d0, const int d1, const int d2, const int d3, const int d4, const int d5,
    const int as0, const int as1, const int as2, const int as3, const int as4, const int as5,
    const int bs0, const int bs1, const int bs2, const int bs3, const int bs4, const int bs5,
    const int a_off,
    const int b_off,
    NF_GLOBAL const float* NF_RESTRICT a,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int words = (int)(((uint)n + 3u) >> 2);
    const int w = tr_gid(words);
    if (w < 0) return;
    const int d[TR_MAX_RANK] = { d0, d1, d2, d3, d4, d5 };
    const int as[TR_MAX_RANK] = { as0, as1, as2, as3, as4, as5 };
    const int bs[TR_MAX_RANK] = { bs0, bs1, bs2, bs3, bs4, bs5 };
    uint word = 0u;
    for (int k = 0; k < 4; ++k) {
        const int e = 4 * w + k;
        if (e < n) {
            const float va = a[a_off + tr_coord_offset(e, rank, d, as)];
            const float vb = b[b_off + tr_coord_offset(e, rank, d, bs)];
            if (tr_e_compare(op, va, vb)) word |= 1u << (uint)(8 * k);
        }
    }
    out[w] = word;
}

NF_KERNEL void TR_Where_F32(
    const int cond_kind,
    const int n,
    const int rank,
    const int d0, const int d1, const int d2, const int d3, const int d4, const int d5,
    const int cs0, const int cs1, const int cs2, const int cs3, const int cs4, const int cs5,
    const int as0, const int as1, const int as2, const int as3, const int as4, const int as5,
    const int bs0, const int bs1, const int bs2, const int bs3, const int bs4, const int bs5,
    const int c_off,
    const int a_off,
    const int b_off,
    NF_GLOBAL const uint* NF_RESTRICT cond,
    NF_GLOBAL const uint* NF_RESTRICT a,
    NF_GLOBAL const uint* NF_RESTRICT b,
    NF_GLOBAL uint* NF_RESTRICT out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int d[TR_MAX_RANK] = { d0, d1, d2, d3, d4, d5 };
    const int cs[TR_MAX_RANK] = { cs0, cs1, cs2, cs3, cs4, cs5 };
    const int as[TR_MAX_RANK] = { as0, as1, as2, as3, as4, as5 };
    const int bs[TR_MAX_RANK] = { bs0, bs1, bs2, bs3, bs4, bs5 };
    const int cp = c_off + tr_coord_offset(i, rank, d, cs);
    int take;
    if (cond_kind == TR_WHERE_COND_BOOL) {
        take = tr_word_byte(cond[cp >> 2], cp) != 0u;
    } else {
        take = (cond[cp] & 0x7fffffffu) != 0u;
    }
    out[i] = take ? a[a_off + tr_coord_offset(i, rank, d, as)] : b[b_off + tr_coord_offset(i, rank, d, bs)];
}

NF_KERNEL void TR_Gather_F32(
    const int dim,
    const int n,
    const int rank,
    const int d0, const int d1, const int d2, const int d3, const int d4, const int d5,
    const int id0, const int id1, const int id2, const int id3, const int id4, const int id5,
    const int in_off,
    const int idx_off,
    const int err_slot,
    const int err_dim,
    NF_GLOBAL const uint* NF_RESTRICT in,
    NF_GLOBAL const int* NF_RESTRICT idx,
    NF_GLOBAL uint* NF_RESTRICT out,
    NF_GLOBAL int* NF_RESTRICT err)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int d[TR_MAX_RANK] = { d0, d1, d2, d3, d4, d5 };
    const int id[TR_MAX_RANK] = { id0, id1, id2, id3, id4, id5 };
    const int g = idx[idx_off + i];
    if (g < 0 || g >= id[dim]) {
        tr_report_index_error(err, err_slot, g, err_dim, id[dim]);
        out[i] = 0u;
        return;
    }
    int rem = i;
    int src = 0;
    int stride = 1;
    for (int k = (rank < TR_MAX_RANK ? rank : TR_MAX_RANK) - 1; k >= 0; --k) {
        const int c = rem % d[k];
        rem = rem / d[k];
        src += (k == dim ? g : c) * stride;
        stride *= id[k];
    }
    out[i] = in[in_off + src];
}

NF_KERNEL void TR_Scatter_F32(
    const int dim,
    const int mode,
    const int fibers,
    const int rank,
    const int d0, const int d1, const int d2, const int d3, const int d4, const int d5,
    const int od0, const int od1, const int od2, const int od3, const int od4, const int od5,
    const int idx_off,
    const int src_off,
    const int err_slot,
    const int err_dim,
    NF_GLOBAL const int* NF_RESTRICT idx,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL float* NF_RESTRICT out,
    NF_GLOBAL int* NF_RESTRICT err)
{
    const int f = tr_gid(fibers);
    if (f < 0) return;
    const int d[TR_MAX_RANK] = { d0, d1, d2, d3, d4, d5 };
    const int od[TR_MAX_RANK] = { od0, od1, od2, od3, od4, od5 };
    int rem = f;
    int ibase = 0;
    int obase = 0;
    int istride = 1;
    int ostride = 1;
    int istep = 0;
    int ostep = 0;
    for (int k = (rank < TR_MAX_RANK ? rank : TR_MAX_RANK) - 1; k >= 0; --k) {
        if (k == dim) {
            istep = istride;
            ostep = ostride;
        } else {
            const int c = rem % d[k];
            rem = rem / d[k];
            ibase += c * istride;
            obase += c * ostride;
        }
        istride *= d[k];
        ostride *= od[k];
    }
    const int len = d[dim];
    const int limit = od[dim];
    for (int j = 0; j < len; ++j) {
        const int lin = ibase + j * istep;
        const int v = idx[idx_off + lin];
        if (v < 0 || v >= limit) {
            tr_report_index_error(err, err_slot, v, err_dim, limit);
            continue;
        }
        const int o = obase + v * ostep;
        if (mode == TR_SCATTER_ADD) {
            out[o] = out[o] + src[src_off + lin];
        } else {
            out[o] = src[src_off + lin];
        }
    }
}

NF_KERNEL void TR_UnaryBackward_F32(
    const int op,
    const int n,
    const float p0,
    const float p1,
    const int x_off,
    const int y_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* x,
    NF_GLOBAL const float* y,
    NF_GLOBAL const float* gy,
    NF_GLOBAL float* gx)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float g = gy[gy_off + i];
    float r;
    switch (op) {
        case TR_E_RELU: {
            r = tr_e_gtz(x[x_off + i]) ? g : 0.0f;
            break;
        }
        case TR_E_GELU: {
            r = g * tr_e_gelu_grad(x[x_off + i]);
            break;
        }
        case TR_E_SILU: {
            r = g * tr_e_silu_grad(x[x_off + i]);
            break;
        }
        case TR_E_SIGMOID: {
            const float yv = y[y_off + i];
            r = (g * yv) * (1.0f - yv);
            break;
        }
        case TR_E_TANH: {
            const float yv = y[y_off + i];
            r = g * (1.0f - yv * yv);
            break;
        }
        case TR_E_LEAKY_RELU: {
            r = tr_e_gtz(x[x_off + i]) ? g : p0 * g;
            break;
        }
        case TR_E_ELU: {
            if (tr_e_gtz(x[x_off + i])) {
                r = g;
            } else {
                r = g * (y[y_off + i] + p0);
            }
            break;
        }
        case TR_E_SOFTPLUS: {
            r = g * tr_e_sigmoid(x[x_off + i]);
            break;
        }
        case TR_E_MISH: {
            r = g * tr_e_mish_grad(x[x_off + i]);
            break;
        }
        case TR_E_ABS: {
            const float xv = x[x_off + i];
            r = tr_e_gtz(xv) ? g : (tr_e_ltz(xv) ? tr_e_neg(g) : 0.0f);
            break;
        }
        case TR_E_CLAMP: {
            const float xv = x[x_off + i];
            r = (tr_e_lt(xv, p0) || tr_e_lt(p1, xv)) ? 0.0f : g;
            break;
        }
        default: {
            r = as_float(0x7fc00000u);
            break;
        }
    }
    gx[gx_off + i] = r;
}

NF_KERNEL void TR_MaskScale_F32(
    const int n,
    const float scale,
    const int x_off,
    const int m_off,
    const int out_off,
    NF_GLOBAL const float* x,
    NF_GLOBAL const float* m,
    NF_GLOBAL float* out)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float mv = m[m_off + i];
    if (tr_iszero(mv)) {
        out[out_off + i] = 0.0f;
    } else {
        out[out_off + i] = (x[x_off + i] * mv) * scale;
    }
}
