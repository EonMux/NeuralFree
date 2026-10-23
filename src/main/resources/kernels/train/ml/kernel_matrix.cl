/* ml/kernel_matrix.cl - SVM and Gaussian-process kernel matrices. Requires: common/dialect.h, common/train.h */
/*
 * Cross-kernel form: a is an m x f row-major matrix at element offset a_off, b an n x f row-major matrix at
 * element offset b_off, and K[i*n + j] = kernel(a row i, b row j) is m x n at element offset 0. Training
 * passes a = b = x (the same buffer and offset, read only, so a and b may alias); prediction passes the
 * test rows as a and the training or support vectors as b. One work-item per K element: m*n items.
 *
 * Over the f features, ascending: sqdist = sum of (a - b)^2 as acc = fma(diff, diff, acc) and dot = sum of
 * a*b as acc = fma(a, b, acc), both from +0. Every other operation is one rounding each in Java's
 * left-to-right association; the CPU line each formula mirrors is cited.
 *
 * ML_SVM_Kernel(i m, i n, i f, i type, f gamma, i degree, f coef0, i a_off, i b_off, R a, R b, W K)
 *     type ML_SVM_LINEAR: dot                                                   (ml/SVM.java computeKernel)
 *     type ML_SVM_RBF:    exp(-gamma * sqdist)
 *     type ML_SVM_POLY:   (gamma * dot + coef0)^degree, as (float) Math.pow(x, degree) for the int degree:
 *                         ml_kmat_ipow, binary exponentiation in float (1 for degree 0, also for NaN;
 *                         1 / x^|degree| for degree < 0), so a negative base keeps the sign Math.pow gives
 *                         it on every backend (GLSL pow is undefined for x < 0).
 *     any other type:     dot (the default branch of computeKernel).
 * ML_GP_Kernel(i m, i n, i f, i type, f length_scale, f variance, f noise, i a_off, i b_off,
 *              i add_noise_diag, R a, R b, W K)                         (ml/GaussianProcess.java computeKernel)
 *     dist = sqrt(sqdist), ls = length_scale, s3 = (float) Math.sqrt(3), s5 = (float) Math.sqrt(5):
 *     type ML_GP_RBF:      variance * exp(-sqdist / (2 * ls * ls))
 *     type ML_GP_MATERN32: t = s3 * dist / ls;  variance * (1 + t) * exp(-t)
 *     type ML_GP_MATERN52: t = s5 * dist / ls;  variance * (1 + t + 5 * sqdist / (3 * ls * ls)) * exp(-t)
 *     type ML_GP_LINEAR:   variance * dot
 *     any other type:      the RBF formula (the default branch of computeKernel).
 *     When add_noise_diag != 0 and m == n, the diagonal gets K[i*n + i] = value + noise (the training
 *     covariance K + noise I); every other element and every non-square launch carries no noise.
 *     (-s3 * dist) / ls of the CPU code equals -t exactly, since negation commutes with rounding.
 *
 * Operation ids (mirrored in shaders/train/ml/kernel_matrix.comp and tensor/gpu/TrainOps.java): the type
 * argument, equal to the ordinals of ml.SVM.KernelType and ml.GaussianProcess.KernelType.
 *
 * Preconditions (host): m, n, f >= 1 and m*n, a_off + m*f, b_off + n*f at most 2^31-1. K never aliases a
 * or b.
 *
 * Dispatch: 1-D, local TR_LOCAL, global = m*n rounded up. No barriers and no local memory.
 */

#define ML_SVM_LINEAR   0
#define ML_SVM_RBF      1
#define ML_SVM_POLY     2

#define ML_GP_RBF       0
#define ML_GP_MATERN32  1
#define ML_GP_MATERN52  2
#define ML_GP_LINEAR    3

/* Sum over k < f of (a[ai + k] - b[bi + k])^2, ascending, fused. */
NF_INLINE float ml_kmat_sqdist(NF_GLOBAL const float* a, const int ai, NF_GLOBAL const float* b, const int bi,
                               const int f) {
    float acc = 0.0f;
    for (int k = 0; k < f; ++k) {
        const float diff = a[ai + k] - b[bi + k];
        acc = fma(diff, diff, acc);
    }
    return acc;
}

/* Sum over k < f of a[ai + k] * b[bi + k], ascending, fused. */
NF_INLINE float ml_kmat_dot(NF_GLOBAL const float* a, const int ai, NF_GLOBAL const float* b, const int bi,
                            const int f) {
    float acc = 0.0f;
    for (int k = 0; k < f; ++k) {
        acc = fma(a[ai + k], b[bi + k], acc);
    }
    return acc;
}

/* x^e for an int exponent by binary exponentiation in float: 1 when e == 0 (Math.pow(x, 0) is 1 for every
 * x), the product of the squares x^(2^b) selected by the bits of |e| otherwise, and its reciprocal when
 * e < 0. Exact sign for a negative base. Every intermediate lies between |x| and |x^|e|| in magnitude, so it
 * overflows or underflows only when x^|e| does (a reciprocal of an overflowed x^|e| is 0, where Math.pow
 * may return a value below FLT_MIN). */
NF_INLINE float ml_kmat_ipow(const float x, const int e) {
    uint m = e < 0 ? (uint)(-(e + 1)) + 1u : (uint)e;
    float r = 1.0f;
    float p = x;
    while (m != 0u) {
        if ((m & 1u) != 0u) r = r * p;
        m >>= 1;
        if (m != 0u) p = p * p;
    }
    return e < 0 ? 1.0f / r : r;
}

NF_KERNEL void ML_SVM_Kernel(
    const int m,
    const int n,
    const int f,
    const int type,
    const float gamma,
    const int degree,
    const float coef0,
    const int a_off,
    const int b_off,
    NF_GLOBAL const float* a,
    NF_GLOBAL const float* b,
    NF_GLOBAL float* NF_RESTRICT K)
{
    const int t = tr_gid(m * n);
    if (t < 0) return;
    const int i = t / n;
    const int j = t - i * n;
    const int ai = a_off + i * f;
    const int bj = b_off + j * f;
    float v;
    if (type == ML_SVM_RBF) {
        v = exp(-gamma * ml_kmat_sqdist(a, ai, b, bj, f));
    } else if (type == ML_SVM_POLY) {
        const float base = gamma * ml_kmat_dot(a, ai, b, bj, f) + coef0;
        v = ml_kmat_ipow(base, degree);
    } else {
        v = ml_kmat_dot(a, ai, b, bj, f);
    }
    K[t] = v;
}

NF_KERNEL void ML_GP_Kernel(
    const int m,
    const int n,
    const int f,
    const int type,
    const float length_scale,
    const float variance,
    const float noise,
    const int a_off,
    const int b_off,
    const int add_noise_diag,
    NF_GLOBAL const float* a,
    NF_GLOBAL const float* b,
    NF_GLOBAL float* NF_RESTRICT K)
{
    const int t = tr_gid(m * n);
    if (t < 0) return;
    const int i = t / n;
    const int j = t - i * n;
    const int ai = a_off + i * f;
    const int bj = b_off + j * f;
    float v;
    if (type == ML_GP_LINEAR) {
        v = variance * ml_kmat_dot(a, ai, b, bj, f);
    } else {
        const float sq = ml_kmat_sqdist(a, ai, b, bj, f);
        if (type == ML_GP_MATERN32) {
            const float s = 1.73205078f * sqrt(sq) / length_scale;          /* (float) Math.sqrt(3) */
            v = variance * (1.0f + s) * exp(-s);
        } else if (type == ML_GP_MATERN52) {
            const float s = 2.23606801f * sqrt(sq) / length_scale;          /* (float) Math.sqrt(5) */
            const float q = 5.0f * sq / (3.0f * length_scale * length_scale);
            v = variance * (1.0f + s + q) * exp(-s);
        } else {
            v = variance * exp(-sq / (2.0f * length_scale * length_scale));
        }
    }
    if (add_noise_diag != 0 && m == n && i == j) v = v + noise;
    K[t] = v;
}
