/* reduce.cl - row and column reductions, arg-reductions and the max backward. Requires: common/dialect.h, common/train.h */
/*
 * A "row" is a contiguous run of cols elements: row r of a view starts at element in_off + r * cols. The
 * host moves the reduced dims to the end (one strided copy) before launching, so every reduction here runs
 * over the last dim of a contiguous [rows, cols] block. Offsets are element offsets; rows, cols and chunks
 * are >= 0 (chunks >= 1 where it is an argument) and every product of them named below fits in an int.
 *
 * Row-reduction ops (op scalar of TR_ReduceRows*; CPU reference tensor/CpuReductions.java computeReduction,
 * initResultArray and applyReduction; CpuReductions.java var is a mean of squared deviations):
 *   TR_R_SUM   0  sum of x                      (identity 0)
 *   TR_R_MAX   1  tr_jmax over x, Java Math.max (identity -inf; NaN propagates, +0 above -0)
 *   TR_R_MIN   2  tr_jmin over x, Java Math.min (identity +inf; NaN propagates, -0 below +0)
 *   TR_R_SUMSQ 3  sum of x*x, accumulated with fma(x, x, acc)
 *   TR_R_SQDEV 4  sum of (x - mean[mean_off + row])^2, accumulated with fma(d, d, acc)
 * Arg-reduction ops (op scalar of TR_ArgReduceRows_F32; CpuReductions.java argReduce):
 *   TR_R_ARGMAX 0, TR_R_ARGMIN 1
 *
 * TR_ReduceRows_F32(i op, i rows, i cols, i in_off, i mean_off, i out_off, f divisor, R in, R mean, W out)
 *     out[out_off + r] = reduce(op, in[in_off + r*cols + 0 .. cols)) / divisor, r < rows. One TR_LOCAL group
 *     per row: lane l folds columns l, l + TR_LOCAL, ... in ascending order, then the fixed tr_wg_* tree
 *     combines the lanes. divisor == 1 leaves the result untouched (MAX and MIN stay bit-exact on devices
 *     that flush denormals); MEAN passes divisor = the reduction length (sum, then one division, as the CPU).
 *     mean is read only for TR_R_SQDEV (otherwise the placeholder).
 * TR_ReduceRowsSmall_F32(same arguments)
 *     Same result, one work-item per row folding the columns sequentially in ascending order (the CPU
 *     order). The host selects it when cols <= 64.
 * TR_ReduceRowsPartial_F32(i op, i rows, i cols, i chunks, i in_off, i mean_off, R in, R mean, W partial)
 *     Group g < rows*chunks reduces chunk c = g % chunks of row r = g / chunks, the columns
 *     [c*cs, min(cols, (c+1)*cs)) with cs = ceil(cols/chunks), as TR_ReduceRows does, and writes
 *     partial[r*chunks + c] (no division; an empty chunk writes the identity). The host then reduces the
 *     partials as a [rows, chunks] block with TR_ReduceRows/TR_ReduceRowsSmall: op TR_R_MAX/TR_R_MIN keep
 *     their op, TR_R_SUM/TR_R_SUMSQ/TR_R_SQDEV use TR_R_SUM; the second pass carries divisor.
 *     Host shape rule: chunks = (rows < 64 && cols > 65536) ? min(1024, ceil(cols/8192)) : 1.
 * TR_ReduceCols_F32(i rows, i cols, i chunks, i in_off, i out_off, R in, W out)
 *     Column sums of a [rows, cols] block. Work-item i < chunks*cols sums column col = i % cols over the
 *     rows [c*cs, min(rows, (c+1)*cs)) of chunk c = i / cols, cs = ceil(rows/chunks), in ascending row
 *     order, and writes out[out_off + c*cols + col]. With chunks = 1 this is the CPU's sequential order.
 *     Host shape rule: chunks = rows > 4096 ? min(256, ceil(rows/1024)) : 1; when chunks > 1 a second
 *     launch with rows = chunks, chunks = 1 sums the partial rows.
 * TR_ArgReduceRows_F32(i op, i rows, i cols, i in_off, i out_off, R in, W int out)
 *     out[out_off + r] = index of the CPU's argmax/argmin of row r: the CPU starts from (-inf or +inf,
 *     index 0) and replaces only on a strictly better value, so NaN and values equal to the start value
 *     are never selected, ties keep the first index and an empty or all-rejected row gives 0. Each lane
 *     starts from (+-inf, sentinel cols) and keeps the first strictly better element of its columns; the
 *     group then takes the best lane value (tr_wg_max/tr_wg_min) and the lowest index holding a value
 *     equal to it (tr_wg_imin); a sentinel result becomes 0. Comparisons are Java float comparisons
 *     (-0 == +0) evaluated on integer keys, so they stay exact where float compares flush denormals.
 * TR_MaxBackwardRows_F32(i rows, i cols, i x_off, i m_off, i gy_off, i gx_off, R x, R m, R gy, W gx)
 *     gx[gx_off + r*cols + c] = gy[gy_off + r] at the first c with x[x_off + r*cols + c] == m[m_off + r]
 *     (Java float ==; a NaN max matches nothing), else +0 (tensor/CpuReductions.java maxBackward, used by
 *     autograd/GradFunctions.java MaxGrad).
 *
 * Dispatch: 1-D, local TR_LOCAL. TR_ReduceRows, TR_ArgReduceRows, TR_MaxBackwardRows: rows groups;
 * TR_ReduceRowsPartial: rows*chunks groups; TR_ReduceRowsSmall: rows items; TR_ReduceCols: chunks*cols
 * items. Group kernels return per group (uniform control flow) for groups past the last row, so a rounded
 * or folded launch is harmless. Inputs and outputs never alias.
 */

#define TR_R_SUM    0
#define TR_R_MAX    1
#define TR_R_MIN    2
#define TR_R_SUMSQ  3
#define TR_R_SQDEV  4

#define TR_R_ARGMAX 0
#define TR_R_ARGMIN 1

/* Starting value of a row-reduction accumulator. */
NF_INLINE float tr_r_identity(const int op) {
    if (op == TR_R_MAX) return -NF_INF;
    if (op == TR_R_MIN) return NF_INF;
    return 0.0f;
}

/* Folds one element into a row-reduction accumulator; m is the row mean (TR_R_SQDEV only). */
NF_INLINE float tr_r_step(const int op, const float acc, const float x, const float m) {
    switch (op) {
        case TR_R_MAX:   return tr_jmax(acc, x);
        case TR_R_MIN:   return tr_jmin(acc, x);
        case TR_R_SUMSQ: return fma(x, x, acc);
        case TR_R_SQDEV: {
            const float d = x - m;
            return fma(d, d, acc);
        }
        default:         return acc + x;
    }
}

/* Combines the lane accumulators of one group (uniform control flow: op is a kernel argument). */
NF_INLINE float tr_r_group(NF_LOCAL_PTR float* buf, const int op, const float acc) {
    if (op == TR_R_MAX) return tr_wg_max(buf, acc);
    if (op == TR_R_MIN) return tr_wg_min(buf, acc);
    return tr_wg_sum(buf, acc);
}

/* Applies the final division; divisor == 1 returns v unchanged. */
NF_INLINE float tr_r_finish(const float v, const float divisor) {
    return divisor == 1.0f ? v : v / divisor;
}

/* Java float comparison key of a non-NaN value: -0 and +0 share key 0, otherwise tr_order_key. */
NF_INLINE int tr_r_key(const float x) {
    return tr_iszero(x) ? 0 : tr_order_key(x);
}

/* Java a == b for floats (false when either is NaN, -0 == +0), exact under denormal flushing. */
NF_INLINE int tr_r_same(const float a, const float b) {
    return !tr_isnan(a) && !tr_isnan(b) && tr_r_key(a) == tr_r_key(b);
}

/* Java (is_max ? x > best : x < best) for a non-NaN best; false when x is NaN. */
NF_INLINE int tr_r_better(const int is_max, const float x, const float best) {
    if (tr_isnan(x)) return 0;
    return is_max ? tr_r_key(x) > tr_r_key(best) : tr_r_key(x) < tr_r_key(best);
}

NF_KERNEL void TR_ReduceRows_F32(
    const int op,
    const int rows,
    const int cols,
    const int in_off,
    const int mean_off,
    const int out_off,
    const float divisor,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT mean,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float buf[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const uint lid = get_local_id(0);
    const float m = op == TR_R_SQDEV ? mean[mean_off + row] : 0.0f;
    const int base = in_off + row * cols;
    float acc = tr_r_identity(op);
    for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
        acc = tr_r_step(op, acc, in[base + (int)c], m);
    }
    const float r = tr_r_group(buf, op, acc);
    if (lid == 0u) out[out_off + row] = tr_r_finish(r, divisor);
}

NF_KERNEL void TR_ReduceRowsSmall_F32(
    const int op,
    const int rows,
    const int cols,
    const int in_off,
    const int mean_off,
    const int out_off,
    const float divisor,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT mean,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int row = tr_gid(rows);
    if (row < 0) return;
    const float m = op == TR_R_SQDEV ? mean[mean_off + row] : 0.0f;
    const int base = in_off + row * cols;
    float acc = tr_r_identity(op);
    for (int c = 0; c < cols; ++c) {
        acc = tr_r_step(op, acc, in[base + c], m);
    }
    out[out_off + row] = tr_r_finish(acc, divisor);
}

NF_KERNEL void TR_ReduceRowsPartial_F32(
    const int op,
    const int rows,
    const int cols,
    const int chunks,
    const int in_off,
    const int mean_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT mean,
    NF_GLOBAL float* NF_RESTRICT partial)
{
    NF_LOCAL float buf[TR_LOCAL];
    const uint g = get_group_id(0);
    if (chunks <= 0 || g >= (uint)rows * (uint)chunks) return;
    const int row = (int)(g / (uint)chunks);
    const uint chunk = g % (uint)chunks;
    const uint cs = (uint)cols / (uint)chunks + ((uint)cols % (uint)chunks != 0u ? 1u : 0u);
    const uint c0 = chunk * cs;
    const uint c1 = c0 + cs < (uint)cols ? c0 + cs : (uint)cols;
    const uint lid = get_local_id(0);
    const float m = op == TR_R_SQDEV ? mean[mean_off + row] : 0.0f;
    const int base = in_off + row * cols;
    float acc = tr_r_identity(op);
    for (uint c = c0 + lid; c < c1; c += TR_LOCAL) {
        acc = tr_r_step(op, acc, in[base + (int)c], m);
    }
    const float r = tr_r_group(buf, op, acc);
    if (lid == 0u) partial[(int)g] = r;
}

NF_KERNEL void TR_ReduceCols_F32(
    const int rows,
    const int cols,
    const int chunks,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = (cols > 0 && chunks > 0) ? tr_gid(chunks * cols) : -1;
    if (i < 0) return;
    const int chunk = i / cols;
    const int col = i - chunk * cols;
    const uint rs = (uint)rows / (uint)chunks + ((uint)rows % (uint)chunks != 0u ? 1u : 0u);
    const uint r0 = (uint)chunk * rs;
    const uint r1 = r0 + rs < (uint)rows ? r0 + rs : (uint)rows;
    float acc = 0.0f;
    for (uint r = r0; r < r1; ++r) {
        acc = acc + in[in_off + (int)r * cols + col];
    }
    out[out_off + i] = acc;
}

NF_KERNEL void TR_ArgReduceRows_F32(
    const int op,
    const int rows,
    const int cols,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL int* NF_RESTRICT out)
{
    NF_LOCAL float vbuf[TR_LOCAL];
    NF_LOCAL int ibuf[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const uint lid = get_local_id(0);
    const int is_max = op == TR_R_ARGMAX;
    const int base = in_off + row * cols;
    float best = is_max ? -NF_INF : NF_INF;
    int best_idx = cols;
    for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
        const float x = in[base + (int)c];
        if (tr_r_better(is_max, x, best)) {
            best = x;
            best_idx = (int)c;
        }
    }
    const float top = is_max ? tr_wg_max(vbuf, best) : tr_wg_min(vbuf, best);
    const int idx = tr_wg_imin(ibuf, tr_r_same(best, top) ? best_idx : cols);
    if (lid == 0u) out[out_off + row] = idx < cols ? idx : 0;
}

NF_KERNEL void TR_MaxBackwardRows_F32(
    const int rows,
    const int cols,
    const int x_off,
    const int m_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT m,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx)
{
    NF_LOCAL int ibuf[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const uint lid = get_local_id(0);
    const float mv = m[m_off + row];
    const int xb = x_off + row * cols;
    int first = cols;
    for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
        if (tr_r_same(x[xb + (int)c], mv)) {
            first = (int)c;
            break;
        }
    }
    const int idx = tr_wg_imin(ibuf, first);
    const float g = gy[gy_off + row];
    const int ob = gx_off + row * cols;
    for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
        gx[ob + (int)c] = (int)c == idx ? g : 0.0f;
    }
}
