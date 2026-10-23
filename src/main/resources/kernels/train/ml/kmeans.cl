/* ml/kmeans.cl - k-means distances, assignment, centroid update and convergence shift. Requires: common/dialect.h, common/train.h */
/*
 * Lloyd iterations of ml/KMeans on FLOAT32 data, with no float atomics and a summation order that depends
 * only on the shape. points is an n x d row-major matrix at element offset p_off; a centroid matrix is
 * k x d row-major. dist (n x k), assign (n), psum (chunks x k x d), pcount (chunks x k) and the old/new
 * centroid matrices of the update and shift kernels are host scratch at element offset 0. Sums start at +0
 * and run in ascending index order; squared differences accumulate as acc = fma(diff, diff, acc), so a CPU
 * reference written with Math.fma reproduces them bit for bit on normal values.
 *
 * ML_KMeans_Distances(i n, i k, i d, i p_off, i c_off, R points, R centroids, W dist)
 *     dist[i*k + c] = sum over j = 0..d-1 of (points[p_off + i*d + j] - centroids[c_off + c*d + j])^2.
 *     One work-item per (i, c): n*k items.
 * ML_KMeans_Assign(i n, i k, R dist, W int assign)
 *     The CPU loop: best = FLT_MAX, assign = 0; for c ascending, dist[i*k + c] < best takes c. Java float <
 *     is decided on bit patterns (ml_km_less), so NaN and +inf are never taken, ties keep the lowest
 *     cluster and denormals compare exactly on every device. One work-item per point: n items.
 * ML_KMeans_UpdatePartial(i n, i k, i d, i chunks, i p_off, R points, R int assign, W psum, W int pcount)
 *     Chunk ch covers the points [ch*cs, min(n, (ch+1)*cs)), cs = ceil(n / chunks). One work-item per
 *     (ch, c, j), item (ch*k + c)*d + j: psum[(ch*k + c)*d + j] = sum of points[p_off + i*d + j] over the
 *     chunk's points i with assign[i] == c, ascending i; the j == 0 item also writes their count to
 *     pcount[ch*k + c]. chunks*k*d items. The host picks chunks = n > 4096 ? min(256, ceil(n / 1024)) : 1.
 * ML_KMeans_UpdateFinalize(i k, i d, i chunks, R psum, R int pcount, R old_c, W new_c)
 *     One work-item per (c, j), item c*d + j: sum = psum of chunk 0, then + chunk 1, ... in ascending chunk
 *     order, count likewise; new_c[c*d + j] = count > 0 ? sum / (float)count : old_c[c*d + j] (an empty
 *     cluster keeps its centroid). k*d items. With chunks == 1 the sum is the sequential CPU sum.
 * ML_KMeans_Shift(i k, i d, R old_c, R new_c, W shift)
 *     shift[0] = Math.max, starting from 0, over clusters c of the float sqrt of sum over j of
 *     (new_c[c*d + j] - old_c[c*d + j])^2: the convergence measure of ml/KMeans.java (maxShift, compared
 *     with the tolerance). Java (float) Math.sqrt of a float is the correctly rounded float sqrt(). One
 *     work-group: work-item l handles clusters l, l + TR_LOCAL, ...; tr_wg_max combines them (Math.max is
 *     associative and commutative under tr_jmax, so the result equals the sequential one; NaN propagates).
 *
 * Preconditions (host): n, k, d >= 1, 1 <= chunks <= 256, and n*k, n*d, k*d and chunks*k*d at most
 * 2^31-1 (each is the element count of a buffer). Distinct buffers never alias.
 *
 * Dispatch: 1-D, local TR_LOCAL, global = item count rounded up; ML_KMeans_Shift is one work-group
 * (global = local = TR_LOCAL).
 */

/* Java a < b for floats, decided on bit patterns: false when either operand is NaN and between +0 and -0,
 * exact for denormals on devices that flush them in float comparisons. */
NF_INLINE int ml_km_less(const float a, const float b) {
    if (tr_isnan(a) || tr_isnan(b)) return 0;
    if (tr_iszero(a) && tr_iszero(b)) return 0;
    return tr_order_key(a) < tr_order_key(b);
}

NF_KERNEL void ML_KMeans_Distances(
    const int n,
    const int k,
    const int d,
    const int p_off,
    const int c_off,
    NF_GLOBAL const float* NF_RESTRICT points,
    NF_GLOBAL const float* NF_RESTRICT centroids,
    NF_GLOBAL float* NF_RESTRICT dist)
{
    const int t = tr_gid(n * k);
    if (t < 0) return;
    const int i = t / k;
    const int c = t - i * k;
    const int p = p_off + i * d;
    const int q = c_off + c * d;
    float acc = 0.0f;
    for (int j = 0; j < d; ++j) {
        const float diff = points[p + j] - centroids[q + j];
        acc = fma(diff, diff, acc);
    }
    dist[t] = acc;
}

NF_KERNEL void ML_KMeans_Assign(
    const int n,
    const int k,
    NF_GLOBAL const float* NF_RESTRICT dist,
    NF_GLOBAL int* NF_RESTRICT assign)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const int row = i * k;
    float best = as_float(0x7f7fffffu);                     /* Float.MAX_VALUE */
    int best_c = 0;
    for (int c = 0; c < k; ++c) {
        const float v = dist[row + c];
        if (ml_km_less(v, best)) {
            best = v;
            best_c = c;
        }
    }
    assign[i] = best_c;
}

NF_KERNEL void ML_KMeans_UpdatePartial(
    const int n,
    const int k,
    const int d,
    const int chunks,
    const int p_off,
    NF_GLOBAL const float* NF_RESTRICT points,
    NF_GLOBAL const int* NF_RESTRICT assign,
    NF_GLOBAL float* NF_RESTRICT psum,
    NF_GLOBAL int* NF_RESTRICT pcount)
{
    const int t = tr_gid(chunks * k * d);
    if (t < 0) return;
    const int cc = t / d;                                    /* ch*k + c */
    const int j = t - cc * d;
    const int ch = cc / k;
    const int c = cc - ch * k;
    /* cs = ceil(n / chunks); begin and end are computed without int overflow for every n <= 2^31-1. */
    const int cs = n / chunks + (n % chunks != 0 ? 1 : 0);
    const int begin = ch * cs;
    const int end = (n - begin > cs) ? begin + cs : n;
    float s = 0.0f;
    int count = 0;
    for (int i = begin; i < end; ++i) {
        if (assign[i] == c) {
            s = s + points[p_off + i * d + j];
            count += 1;
        }
    }
    psum[t] = s;
    if (j == 0) pcount[cc] = count;
}

NF_KERNEL void ML_KMeans_UpdateFinalize(
    const int k,
    const int d,
    const int chunks,
    NF_GLOBAL const float* NF_RESTRICT psum,
    NF_GLOBAL const int* NF_RESTRICT pcount,
    NF_GLOBAL const float* NF_RESTRICT old_c,
    NF_GLOBAL float* NF_RESTRICT new_c)
{
    const int t = tr_gid(k * d);
    if (t < 0) return;
    const int c = t / d;
    const int stride = k * d;
    float s = psum[t];
    int count = pcount[c];
    for (int ch = 1; ch < chunks; ++ch) {
        s = s + psum[ch * stride + t];
        count += pcount[ch * k + c];
    }
    new_c[t] = count > 0 ? s / (float)count : old_c[t];
}

NF_KERNEL void ML_KMeans_Shift(
    const int k,
    const int d,
    NF_GLOBAL const float* NF_RESTRICT old_c,
    NF_GLOBAL const float* NF_RESTRICT new_c,
    NF_GLOBAL float* NF_RESTRICT shift)
{
    NF_LOCAL float red[TR_LOCAL];
    const int lid = (int)get_local_id(0);
    float m = 0.0f;
    for (int c = lid; c < k; c += TR_LOCAL) {
        const int row = c * d;
        float s = 0.0f;
        for (int j = 0; j < d; ++j) {
            const float diff = new_c[row + j] - old_c[row + j];
            s = fma(diff, diff, s);
        }
        m = tr_jmax(m, sqrt(s));
    }
    m = tr_wg_max(red, m);
    if (lid == 0) shift[0] = m;
}
