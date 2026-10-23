/* dense.cl - dense-network bias, activation and activation backward. Requires: common/dialect.h, common/train.h */
/*
 * Per-layer building blocks of the dense network of acceleration/pipeline and ai/v2 (z = x * W^T is a GEMM
 * issued by the host; these kernels add the bias, apply the activation and form the activation gradient).
 *
 * TR_DenseBiasAct_F32(i act, i rows, i cols, i z_off, i b_off, i a_off, RW z, R bias, W a)
 *     For i < rows * cols, c = i % cols (row-major [rows, cols]):
 *         z[z_off + i] += bias[b_off + c];   a[a_off + i] = f_act(z[z_off + i]).
 *     a may be z itself at the same offset (exact in-place: the buffer then holds the activation); otherwise
 *     a, z and bias are disjoint.
 * TR_DenseActBackward_F32(i act, i n, i z_off, i a_off, i gy_off, i gz_off, R z, R a, R gy, W gz)
 *     For i < n: gz[gz_off + i] = gy[gy_off + i] * f'_act(z[z_off + i], a[a_off + i]),
 *     z the pre-activation (bias included) and a = f_act(z). a is read only for SIGMOID and TANH; every other
 *     activation accepts the placeholder buffer with a_off 0. gz may be gy itself at the same offset (exact
 *     in-place); z and a are disjoint from gz.
 *
 * act is the ordinal of ai.v2.NeuralNetworkData.ActivationFunctionType (TR_DENSE_ACT_* below). The host
 * passes 0..17; any other value acts as IDENTITY.
 *
 * Activations (x = z):
 *   0 SIGMOID          f = 1 / (1 + exp(-x))                         f' = a * (1 - a)
 *   1 RELU             f = Math.max(0, x)                            f' = x > 0 ? 1 : 0
 *   2 TANH             f = tanh(x)                                   f' = 1 - a * a
 *   3 LEAKY_RELU       f = Math.max(alpha * x, x), alpha = 0.01      f' = x > 0 ? 1 : alpha
 *   4 ELU              f = x >= 0 ? x : alpha * (exp(x) - 1), alpha = 1          f' = x >= 0 ? 1 : alpha * exp(x)
 *   5 SOFTPLUS         f = Math.max(x, 0) + log(1 + exp(-|x|))  (= log(1 + exp(x)))  f' = 1 / (1 + exp(-x))
 *   6 SOFTSIGN         f = x / (1 + |x|)                             f' = 1 / ((1 + |x|) * (1 + |x|))
 *   7 ISRU             f = x / s, s = sqrt(1 + alpha * x * x), alpha = 1          f' = 1 / (s * s * s)
 *   8 GAUSSIAN         f = exp(-(x * x))                             f' = -2 * x * exp(-(x * x))
 *   9 SINC             f = sin(u) / u, u = pi * x                    f' = pi * (cos(u) / u - sin(u) / (u * u))
 *                      for |u| < 1 both use their Taylor series (f(0) = 1, f'(0) = 0), see tr_dense_sinc
 *  10 SWISH            f = x / (1 + exp(-beta * x)), beta = 1        f' = s * (1 + beta * x * (1 - s)),
 *                                                                         s = 1 / (1 + exp(-beta * x))
 *  11 SELU             f = x > 0 ? scale * x : scale * (alpha * (exp(x) - 1))    f' = x > 0 ? scale : scale * alpha * exp(x)
 *                      alpha = 1.6732632423543772848, scale = 1.0507009873554804934
 *  12 HARD_SIGMOID     y = slope * x + 0.5, slope = 0.2; f = Math.min(Math.max(y, 0), 1)
 *                                                                    f' = 0 <= y <= 1 ? slope : 0
 *  13 BENT_IDENTITY    f = (s - 1) / 2 + x, s = sqrt(x * x + 1)      f' = x / (2 * s) + 1
 *  14 PRELU            f = x >= 0 ? x : alpha * x, alpha = 0.25      f' = x >= 0 ? 1 : alpha
 *  15 SRELU            tl = -0.5, al = 0.5, tr = 0.5, ar = 0.5
 *                      f = x <= tl ? tl + al * (x - tl) : x >= tr ? tr + ar * (x - tr) : x
 *                                                                    f' = x <= tl ? al : x >= tr ? ar : 1
 *  16 SOFTEXPONENTIAL  f = (exp(alpha * x) - 1) / alpha + alpha, alpha = 1   f' = exp(alpha * x)
 *  17 IDENTITY         f = x                                         f' = 1
 * Math.max/Math.min are the Java semantics of tr_jmax/tr_jmin (NaN propagates, +0 above -0). sqrt(1 + x * x)
 * is evaluated without overflow (tr_dense_sqrt1p_sq), so ISRU and BENT_IDENTITY stay finite for every
 * finite x.
 *
 * Numerics (train.h contract): no contraction, so every product is rounded before the following add.
 * Tolerance class: transcendentals (|gpu - ref| <= 1e-5 * max(1, |ref|)) for every activation built on exp,
 * log, tanh, sin, cos or sqrt; RELU, LEAKY_RELU, PRELU, SRELU, HARD_SIGMOID, IDENTITY and the bias add are
 * single-rounding arithmetic.
 *
 * Dispatch: 1-D, local TR_LOCAL; global = rows * cols (TR_DenseBiasAct_F32) or n (TR_DenseActBackward_F32)
 * rounded up. No barriers.
 */

#define TR_DENSE_ACT_SIGMOID          0
#define TR_DENSE_ACT_RELU             1
#define TR_DENSE_ACT_TANH             2
#define TR_DENSE_ACT_LEAKY_RELU       3
#define TR_DENSE_ACT_ELU              4
#define TR_DENSE_ACT_SOFTPLUS         5
#define TR_DENSE_ACT_SOFTSIGN         6
#define TR_DENSE_ACT_ISRU             7
#define TR_DENSE_ACT_GAUSSIAN         8
#define TR_DENSE_ACT_SINC             9
#define TR_DENSE_ACT_SWISH           10
#define TR_DENSE_ACT_SELU            11
#define TR_DENSE_ACT_HARD_SIGMOID    12
#define TR_DENSE_ACT_BENT_IDENTITY   13
#define TR_DENSE_ACT_PRELU           14
#define TR_DENSE_ACT_SRELU           15
#define TR_DENSE_ACT_SOFTEXPONENTIAL 16
#define TR_DENSE_ACT_IDENTITY        17

/* 1 / (1 + exp(-x)). */
NF_INLINE float tr_dense_sigmoid(const float x) {
    return 1.0f / (1.0f + exp(-x));
}

/* sqrt(1 + x * x) without overflow: for |x| > 1 it is |x| * sqrt(1 + t * t) with t = 1 / |x|. */
NF_INLINE float tr_dense_sqrt1p_sq(const float x) {
    const float ax = fabs(x);
    if (!(ax > 1.0f)) return sqrt(1.0f + x * x);
    const float t = 1.0f / ax;
    return ax * sqrt(1.0f + t * t);
}

/*
 * sinc(x) = sin(pi x) / (pi x) and its derivative pi * (cos(u) / u - sin(u) / (u * u)), u = pi x. For
 * |u| < 1 the Taylor series replace the quotients (truncation below 3e-8 for f and 7e-9 for f'), which
 * removes the cancellation of the derivative near 0 (its two quotients grow like 1/u while their difference
 * shrinks like u, so the error of sin, cos and division allowed by OpenCL 1.2 would exceed the transcendental
 * tolerance for smaller |u|) and the division by a zero or flushed u.
 */
NF_INLINE float tr_dense_sinc(const float x) {
    const float pi = 3.14159265358979323846f;
    const float u = pi * x;
    if (fabs(u) < 1.0f) {
        const float u2 = u * u;
        return 1.0f - u2 * (0.16666666666666667f - u2 * (0.0083333333333333333f
            - u2 * (1.9841269841269841e-4f - u2 * 2.7557319223985891e-6f)));
    }
    return sin(u) / u;
}

NF_INLINE float tr_dense_sinc_grad(const float x) {
    const float pi = 3.14159265358979323846f;
    const float u = pi * x;
    if (fabs(u) < 1.0f) {
        const float u2 = u * u;
        return pi * -(u * (0.33333333333333333f - u2 * (0.033333333333333333f - u2 * (1.1904761904761905e-3f
            - u2 * (2.2045855379188713e-5f - u2 * 2.5052108385441719e-7f)))));
    }
    return pi * (cos(u) / u - sin(u) / (u * u));
}

/* f_act(x); see the table above. */
NF_INLINE float tr_dense_act(const int act, const float x) {
    switch (act) {
        case TR_DENSE_ACT_SIGMOID:
            return tr_dense_sigmoid(x);
        case TR_DENSE_ACT_RELU:
            return tr_jmax(0.0f, x);
        case TR_DENSE_ACT_TANH:
            return tanh(x);
        case TR_DENSE_ACT_LEAKY_RELU: {
            const float alpha = 0.01f;
            return tr_jmax(alpha * x, x);
        }
        case TR_DENSE_ACT_ELU: {
            const float alpha = 1.0f;
            return x >= 0.0f ? x : alpha * (exp(x) - 1.0f);
        }
        case TR_DENSE_ACT_SOFTPLUS:
            return tr_jmax(x, 0.0f) + log(1.0f + exp(-fabs(x)));
        case TR_DENSE_ACT_SOFTSIGN:
            return x / (1.0f + fabs(x));
        case TR_DENSE_ACT_ISRU:
            return x / tr_dense_sqrt1p_sq(x);
        case TR_DENSE_ACT_GAUSSIAN:
            return exp(-(x * x));
        case TR_DENSE_ACT_SINC:
            return tr_dense_sinc(x);
        case TR_DENSE_ACT_SWISH: {
            const float beta = 1.0f;
            return x / (1.0f + exp(-(beta * x)));
        }
        case TR_DENSE_ACT_SELU: {
            const float alpha = 1.6732632423543772848f;
            const float scale = 1.0507009873554804934f;
            return x > 0.0f ? scale * x : scale * (alpha * (exp(x) - 1.0f));
        }
        case TR_DENSE_ACT_HARD_SIGMOID: {
            const float slope = 0.2f;
            const float y = slope * x + 0.5f;
            return tr_jmin(tr_jmax(y, 0.0f), 1.0f);
        }
        case TR_DENSE_ACT_BENT_IDENTITY:
            return (tr_dense_sqrt1p_sq(x) - 1.0f) / 2.0f + x;
        case TR_DENSE_ACT_PRELU: {
            const float alpha = 0.25f;
            return x >= 0.0f ? x : alpha * x;
        }
        case TR_DENSE_ACT_SRELU: {
            const float tl = -0.5f, al = 0.5f, tr = 0.5f, ar = 0.5f;
            if (x <= tl) return tl + al * (x - tl);
            if (x >= tr) return tr + ar * (x - tr);
            return x;
        }
        case TR_DENSE_ACT_SOFTEXPONENTIAL: {
            const float alpha = 1.0f;
            return (exp(alpha * x) - 1.0f) / alpha + alpha;
        }
        default:
            return x;
    }
}

/* f'_act at pre-activation x with activation output a (used by SIGMOID and TANH only); see the table above. */
NF_INLINE float tr_dense_act_grad(const int act, const float x, const float a) {
    switch (act) {
        case TR_DENSE_ACT_SIGMOID:
            return a * (1.0f - a);
        case TR_DENSE_ACT_RELU:
            return x > 0.0f ? 1.0f : 0.0f;
        case TR_DENSE_ACT_TANH:
            return 1.0f - a * a;
        case TR_DENSE_ACT_LEAKY_RELU: {
            const float alpha = 0.01f;
            return x > 0.0f ? 1.0f : alpha;
        }
        case TR_DENSE_ACT_ELU: {
            const float alpha = 1.0f;
            return x >= 0.0f ? 1.0f : alpha * exp(x);
        }
        case TR_DENSE_ACT_SOFTPLUS:
            return tr_dense_sigmoid(x);
        case TR_DENSE_ACT_SOFTSIGN: {
            const float d = 1.0f + fabs(x);
            return 1.0f / (d * d);
        }
        case TR_DENSE_ACT_ISRU: {
            const float s = tr_dense_sqrt1p_sq(x);
            return 1.0f / (s * s * s);
        }
        case TR_DENSE_ACT_GAUSSIAN:
            return -2.0f * x * exp(-(x * x));
        case TR_DENSE_ACT_SINC:
            return tr_dense_sinc_grad(x);
        case TR_DENSE_ACT_SWISH: {
            const float beta = 1.0f;
            const float s = 1.0f / (1.0f + exp(-(beta * x)));
            return s * (1.0f + beta * x * (1.0f - s));
        }
        case TR_DENSE_ACT_SELU: {
            const float alpha = 1.6732632423543772848f;
            const float scale = 1.0507009873554804934f;
            return x > 0.0f ? scale : scale * alpha * exp(x);
        }
        case TR_DENSE_ACT_HARD_SIGMOID: {
            const float slope = 0.2f;
            const float y = slope * x + 0.5f;
            return (y >= 0.0f && y <= 1.0f) ? slope : 0.0f;
        }
        case TR_DENSE_ACT_BENT_IDENTITY:
            return x / (2.0f * tr_dense_sqrt1p_sq(x)) + 1.0f;
        case TR_DENSE_ACT_PRELU: {
            const float alpha = 0.25f;
            return x >= 0.0f ? 1.0f : alpha;
        }
        case TR_DENSE_ACT_SRELU: {
            const float tl = -0.5f, al = 0.5f, tr = 0.5f, ar = 0.5f;
            if (x <= tl) return al;
            if (x >= tr) return ar;
            return 1.0f;
        }
        case TR_DENSE_ACT_SOFTEXPONENTIAL: {
            const float alpha = 1.0f;
            return exp(alpha * x);
        }
        default:
            return 1.0f;
    }
}

NF_KERNEL void TR_DenseBiasAct_F32(
    const int act,
    const int rows,
    const int cols,
    const int z_off,
    const int b_off,
    const int a_off,
    NF_GLOBAL float* z,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* a)
{
    const int i = tr_gid(rows * cols);
    if (i < 0) return;
    const float v = z[z_off + i] + bias[b_off + i % cols];
    z[z_off + i] = v;
    a[a_off + i] = tr_dense_act(act, v);
}

NF_KERNEL void TR_DenseActBackward_F32(
    const int act,
    const int n,
    const int z_off,
    const int a_off,
    const int gy_off,
    const int gz_off,
    NF_GLOBAL const float* NF_RESTRICT z,
    NF_GLOBAL const float* NF_RESTRICT a,
    NF_GLOBAL const float* gy,
    NF_GLOBAL float* gz)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float x = z[z_off + i];
    const float av = (act == TR_DENSE_ACT_SIGMOID || act == TR_DENSE_ACT_TANH) ? a[a_off + i] : 0.0f;
    gz[gz_off + i] = gy[gy_off + i] * tr_dense_act_grad(act, x, av);
}
