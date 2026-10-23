/* norm.cl - layer, RMS, batch and group normalization, row index-select, and their backward kernels. Requires: common/dialect.h, common/train.h */
/*
 * Offsets are element offsets into the buffer. Every tensor operand is contiguous (the dispatcher makes it
 * so); mean, var, gamma and beta of TR_ChannelNormApply_F32 and mean, var of TR_ChannelStats_F32 are read
 * and written at element offset 0. In the backward kernels the stats, mean, var and rsums operands, the gamma of
 * TR_GroupGradSums_F32 and TR_ChannelNormBackwardApply_F32 and the sums of TR_ChannelNormBackwardApply_F32 are
 * at element offset 0 as well. No operand aliases another one.
 *
 * TR_LayerNormRows_F32(i rows, i cols, f eps, i has_gamma, i has_beta, i in_off, i g_off, i b_off, i out_off,
 *                      R in, R gamma, R beta, W out)
 *     Per row r of cols elements (tensor/CpuNorm.java layerNorm):
 *       mean = sum(x) / cols; var = sum((x - mean)^2) / cols; sdev = sqrt(var + eps);
 *       y = (x - mean) / sdev; y = gamma[j] * y when has_gamma; y = y + beta[j] when has_beta.
 * TR_RMSNormRows_F32(i rows, i cols, f eps, i has_weight, i in_off, i w_off, i out_off, R in, R w, W out)
 *     Per row (tensor/CpuNorm.java rmsNorm): rms = sqrt(sum(x^2) / cols + eps); y = x / rms;
 *     y = y * w[j] when has_weight.
 * TR_ChannelStats_F32(i B, i C, i HW, i in_off, R in, W mean, W var)
 *     Per channel c over the B*HW elements in[in_off + (b*C + c)*HW + s] (tensor/CpuNorm.java batchNorm,
 *     training mode): mean[c] = sum(x) / (B*HW);
 *     var[c] = sum((x - mean[c])^2) / (B*HW), the biased variance of the two-pass CPU loop, which normalizes
 *     the batch; GpuNormOps scales it by B*HW / (B*HW - 1) in the running-variance update. B*HW == 0 gives
 *     0/0 = NaN, as on the CPU.
 * TR_ChannelNormApply_F32(i B, i C, i HW, i G, f eps, i has_affine, i in_off, i out_off,
 *                         R in, R mean, R var, R gamma, R beta, W out)
 *     For element e = (b*C + c)*HW + s of the B*C*HW input: k = c when G == 0 (BatchNorm, per-channel
 *     statistics), else b*G + c/(C/G) (GroupNorm, statistics of B*G rows); sdev = sqrt(var[k] + eps);
 *     y = (x - mean[k]) / sdev; y = gamma[c] * y + beta[c] when has_affine (tensor/CpuNorm.java batchNorm and
 *     groupNorm). Precondition: G == 0, or G > 0 and
 *     C % G == 0.
 * TR_IndexSelectRows_F32(i n_idx, i dim, i rows, i oob_mode, i w_off, i idx_off, i out_off, i err_slot,
 *                        R words w, R int idx, W words out, RW int err)
 *     out[out_off + i*dim + j] = w[w_off + k*dim + j] with k = idx[idx_off + i], for i < n_idx, j < dim.
 *     oob_mode TR_IDX_OOB_ERROR (nn/layers/Embedding.java forward): k outside [0, rows) writes 0 to the element
 *     and reports {k, 0, rows} in slot err_slot (tr_report_index_error). oob_mode TR_IDX_OOB_CLAMP (nn/graph/MessagePassing.java gatherNodes):
 *     k = max(0, min(k, rows - 1)).
 *     With rows == 0 no row exists in either mode, so every element takes the error path. w and out are
 *     32-bit words, so the copy is bit-identical (FLOAT32 or INT32 rows).
 *
 * Backward kernels. gy is the upstream gradient (the shape of the forward output), gx the input gradient. Every
 * normalized value xhat is recomputed from x with the forward's own operation sequence (the same lane helpers,
 * tr_wg_sum calls, divisions and square roots as the forward kernel above, or the same expressions on the
 * statistics the forward read), so xhat is bit-identical to the forward's. d = gy * gamma (or gy * w) when the
 * affine flag is set, else gy. The CPU references are tensor/CpuNorm.java (norm backward) and
 * tensor/CpuIndexing.java (index add). Tolerance: the reduction class (|gpu - ref64| <= 1e-5 * sqrt(K) *
 * sum|terms| + K * FLT_MIN, K the longest sum an output depends on) for the norm kernels; the single-rounding
 * class for TR_IndexAddRows_F32 (same-order float adds from 0, bit-identical for normal values).
 *
 * TR_LayerNormRowsBackward_F32(i rows, i cols, f eps, i has_gamma, i x_off, i g_off, i gy_off, i gx_off,
 *                              R x, R gamma, R gy, W gx, W stats)
 *     Per row r: mean, var, sdev exactly as TR_LayerNormRows_F32; xhat = (x - mean) / sdev;
 *       a = sum(d) / cols; q = sum(d * xhat) / cols (d * xhat accumulated with fma);
 *       gx = ((d - a) - xhat * q) / sdev;
 *       stats[2r] = mean, stats[2r + 1] = sdev (stats holds 2*rows floats; read by TR_NormAffineGradCols_F32).
 * TR_RMSNormRowsBackward_F32(i rows, i cols, f eps, i has_weight, i x_off, i w_off, i gy_off, i gx_off,
 *                            R x, R w, R gy, W gx, W stats)
 *     Per row r: rms exactly as TR_RMSNormRows_F32; xhat = x / rms; q = sum(d * xhat) / cols (fma);
 *       gx = (d - xhat * q) / rms; stats[r] = rms (stats holds rows floats).
 * TR_NormAffineGradCols_F32(i mode, i rows, i cols, i chunks, i has_beta, i x_off, i gy_off, i out_off,
 *                           R x, R gy, R stats, W out)
 *     Column sums over a [rows, cols] block for the LayerNorm gamma/beta and RMSNorm weight gradients.
 *     Work-item i < chunks*cols handles column col = i % cols over the rows [c*rs, min(rows, (c+1)*rs)) of
 *     chunk c = i / cols, rs = ceil(rows/chunks), in ascending row order (the TR_ReduceCols_F32 partition):
 *       xhat = (x - stats[2r]) / stats[2r + 1] for mode TR_N_AFF_LN, x / stats[r] for TR_N_AFF_RMS (the
 *       stats of the matching row-backward kernel); accg = fma(gy, xhat, accg); accb = accb + gy;
 *       out[out_off + c*cols + col] = accg; out[out_off + (chunks + c)*cols + col] = accb when has_beta.
 *     With chunks == 1 these are the gradients; else the host sums each [chunks, cols] plane with
 *     TR_ReduceCols_F32 (rows = chunks). Host shape rule: chunks = rows > 4096 ? min(256, ceil(rows/1024)) : 1.
 * TR_ChannelGradSums_F32(i B, i C, i HW, i G, f eps, i x_off, i gy_off, i sums_off, R x, R mean, R var, R gy,
 *                        W sums)
 *     Per channel c over its B*HW elements e = (b*C + c)*HW + s, in TR_ChannelStats_F32's lane order:
 *       k and xhat exactly as TR_ChannelNormApply_F32 (k = c when G == 0, else b*G + c/(C/G));
 *       sums[sums_off + c] = sum(gy) (beta gradient); sums[sums_off + C + c] = sum(gy * xhat) (gamma
 *       gradient, fma). Planar output: the BatchNorm dispatcher passes one [2C] scratch here and to the BATCH
 *       mode of TR_ChannelNormBackwardApply_F32. Precondition: G == 0, or G > 0 and C % G == 0.
 * TR_GroupGradSums_F32(i B, i C, i HW, i G, f eps, i has_affine, i x_off, i gy_off, R x, R mean, R var,
 *                      R gamma, R gy, W rsums)
 *     Per (batch, group) row r = b*G + g over its M = (C/G)*HW contiguous elements starting at r*M (channel
 *     c = g*(C/G) + j/HW of the row's element j): sdev = sqrt(var[r] + eps); xhat = (x - mean[r]) / sdev
 *     (TR_ChannelNormApply_F32 with k = r); d = gy * gamma[c] when has_affine;
 *       rsums[r] = sum(d); rsums[B*G + r] = sum(d * xhat) (fma). Planar output of 2*B*G floats.
 *     Precondition: G > 0 and C % G == 0.
 * TR_ChannelNormBackwardApply_F32(i mode, i B, i C, i HW, i G, i count, f eps, i has_affine, i x_off, i gy_off,
 *                                 i gx_off, R x, R mean, R var, R gamma, R gy, R sums, W gx)
 *     For element e = (b*C + c)*HW + s: sdev = sqrt(var[k] + eps), xhat = (x - mean[k]) / sdev and n = count:
 *       TR_N_CB_BATCH (k = c; sums from TR_ChannelGradSums_F32, s1 = sums[k], s2 = sums[C + k], n = B*HW):
 *         t = (gy - s1 / n) - xhat * (s2 / n); t = gamma[c] * t when has_affine; gx = t / sdev.
 *       TR_N_CB_GROUP (k = b*G + c/(C/G); sums from TR_GroupGradSums_F32, s1 = sums[k], s2 = sums[B*G + k],
 *         n = (C/G)*HW): gx = ((d - s1 / n) - xhat * (s2 / n)) / sdev.
 *       TR_N_CB_EVAL (k = c; mean and var are the running statistics, sums is the placeholder and x, mean are
 *         not read): gx = d / sdev.
 *     Precondition for TR_N_CB_GROUP: G > 0 and C % G == 0.
 * TR_IndexAddRows_F32(i n_idx, i dim, i rows, i row_blocks, i oob_mode, i skip_row, i gy_off, i idx_off,
 *                     i gw_off, R int idx, R gy, RW gw)
 *     Backward of TR_IndexSelectRows_F32: gw[gw_off + k*dim + j] += gy[gy_off + i*dim + j] for i = 0 .. n_idx-1
 *     in ascending order, k = idx[idx_off + i] clamped as the forward in TR_IDX_OOB_CLAMP mode. An index
 *     outside [0, rows) (TR_IDX_OOB_ERROR, which the forward already reported) and k == skip_row (the
 *     Embedding padding row; a negative skip_row skips no row) contribute nothing. gw is zero-filled by the
 *     dispatcher. Work-item t < dim*row_blocks owns column j = t % dim of the rows
 *     [blk*bs, min(rows, (blk+1)*bs)), blk = t / dim,
 *     bs = ceil(rows/row_blocks), so every gw element has one owner that adds in the CPU loop's order: the
 *     result is bit-identical to the sequential sum for every row_blocks. Host shape rule (speed only):
 *     row_blocks = min(rows, max(1, min(64, ceil(16384/dim)))).
 *
 * Numerics (the contract of common/train.h): per-lane partial sums run in ascending index order, lane l covering
 * indices l, l + TR_LOCAL, ...; squares and products accumulate with fma; the lanes combine through the fixed
 * tr_wg_sum tree. The affine step is a separate multiply and add, as Java evaluates w * n + b; every other
 * product and difference of the backward formulas is a separate operation.
 *
 * Dispatch: local TR_LOCAL. Row kernels (TR_LayerNormRows_F32, TR_RMSNormRows_F32 and their backward kernels)
 * launch one group per row (rows groups); TR_ChannelStats_F32 and TR_ChannelGradSums_F32 launch one group per
 * channel (C groups); TR_GroupGradSums_F32 launches one group per (batch, group) row (B*G groups);
 * TR_ChannelNormApply_F32 and TR_ChannelNormBackwardApply_F32 are 1-D over B*C*HW items,
 * TR_IndexSelectRows_F32 1-D over n_idx*dim items, TR_NormAffineGradCols_F32 1-D over chunks*cols items and
 * TR_IndexAddRows_F32 1-D over dim*row_blocks items. Group-per-row kernels return early only on the group id,
 * so every work-item of a live group reaches every barrier.
 */

/* TR_IndexSelectRows_F32 and TR_IndexAddRows_F32 oob_mode ids (tensor/gpu/TrainOps). */
#define TR_IDX_OOB_ERROR 0
#define TR_IDX_OOB_CLAMP 1

/* TR_NormAffineGradCols_F32 mode ids (tensor/gpu/TrainOps). */
#define TR_N_AFF_LN  0
#define TR_N_AFF_RMS 1

/* TR_ChannelNormBackwardApply_F32 mode ids (tensor/gpu/TrainOps). */
#define TR_N_CB_BATCH 0
#define TR_N_CB_GROUP 1
#define TR_N_CB_EVAL  2

/* Lane partial of sum(x[base + j]) over j in [0, n): j = lid, lid + TR_LOCAL, ... ascending. */
NF_INLINE float tr_n_lane_sum(NF_GLOBAL const float* x, const int base, const int n, const int lid) {
    float acc = 0.0f;
    for (int j = lid; j < n; j += TR_LOCAL) {
        acc = acc + x[base + j];
    }
    return acc;
}

/* Lane partial of sum((x[base + j] - mean)^2) over j in [0, n). */
NF_INLINE float tr_n_lane_sqdev(NF_GLOBAL const float* x, const int base, const int n, const float mean,
                                const int lid) {
    float acc = 0.0f;
    for (int j = lid; j < n; j += TR_LOCAL) {
        const float d = x[base + j] - mean;
        acc = fma(d, d, acc);
    }
    return acc;
}

/* Lane partial of sum(x[base + j]^2) over j in [0, n). */
NF_INLINE float tr_n_lane_sumsq(NF_GLOBAL const float* x, const int base, const int n, const int lid) {
    float acc = 0.0f;
    for (int j = lid; j < n; j += TR_LOCAL) {
        const float v = x[base + j];
        acc = fma(v, v, acc);
    }
    return acc;
}

NF_KERNEL void TR_LayerNormRows_F32(
    const int rows,
    const int cols,
    const float eps,
    const int has_gamma,
    const int has_beta,
    const int in_off,
    const int g_off,
    const int b_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT gamma,
    NF_GLOBAL const float* NF_RESTRICT beta,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float red[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const int lid = (int)get_local_id(0);
    const int src = in_off + row * cols;
    const int dst = out_off + row * cols;

    const float mean = tr_wg_sum(red, tr_n_lane_sum(in, src, cols, lid)) / (float)cols;
    const float var = tr_wg_sum(red, tr_n_lane_sqdev(in, src, cols, mean, lid)) / (float)cols;
    const float sdev = sqrt(var + eps);

    for (int j = lid; j < cols; j += TR_LOCAL) {
        float y = (in[src + j] - mean) / sdev;
        if (has_gamma) y = gamma[g_off + j] * y;
        if (has_beta) y = y + beta[b_off + j];
        out[dst + j] = y;
    }
}

NF_KERNEL void TR_RMSNormRows_F32(
    const int rows,
    const int cols,
    const float eps,
    const int has_weight,
    const int in_off,
    const int w_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float red[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const int lid = (int)get_local_id(0);
    const int src = in_off + row * cols;
    const int dst = out_off + row * cols;

    const float sumsq = tr_wg_sum(red, tr_n_lane_sumsq(in, src, cols, lid));
    const float rms = sqrt(sumsq / (float)cols + eps);

    for (int j = lid; j < cols; j += TR_LOCAL) {
        float y = in[src + j] / rms;
        if (has_weight) y = y * w[w_off + j];
        out[dst + j] = y;
    }
}

NF_KERNEL void TR_ChannelStats_F32(
    const int B,
    const int C,
    const int HW,
    const int in_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT mean,
    NF_GLOBAL float* NF_RESTRICT var)
{
    NF_LOCAL float red[TR_LOCAL];
    const int c = (int)get_group_id(0);
    if (c >= C) return;
    const int lid = (int)get_local_id(0);
    const int count = B * HW;

    float acc = 0.0f;
    for (int j = lid; j < count; j += TR_LOCAL) {
        const int b = j / HW;
        acc = acc + in[in_off + (b * C + c) * HW + (j - b * HW)];
    }
    const float m = tr_wg_sum(red, acc) / (float)count;

    float acc2 = 0.0f;
    for (int j = lid; j < count; j += TR_LOCAL) {
        const int b = j / HW;
        const float d = in[in_off + (b * C + c) * HW + (j - b * HW)] - m;
        acc2 = fma(d, d, acc2);
    }
    const float v = tr_wg_sum(red, acc2) / (float)count;

    if (lid == 0) {
        mean[c] = m;
        var[c] = v;
    }
}

NF_KERNEL void TR_ChannelNormApply_F32(
    const int B,
    const int C,
    const int HW,
    const int G,
    const float eps,
    const int has_affine,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL const float* NF_RESTRICT mean,
    NF_GLOBAL const float* NF_RESTRICT var,
    NF_GLOBAL const float* NF_RESTRICT gamma,
    NF_GLOBAL const float* NF_RESTRICT beta,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int e = tr_gid(B * C * HW);
    if (e < 0) return;
    const int bc = e / HW;
    const int c = bc % C;
    const int k = G == 0 ? c : (bc / C) * G + c / (C / G);
    const float sdev = sqrt(var[k] + eps);
    float y = (in[in_off + e] - mean[k]) / sdev;
    if (has_affine) y = gamma[c] * y + beta[c];
    out[out_off + e] = y;
}

NF_KERNEL void TR_IndexSelectRows_F32(
    const int n_idx,
    const int dim,
    const int rows,
    const int oob_mode,
    const int w_off,
    const int idx_off,
    const int out_off,
    const int err_slot,
    NF_GLOBAL const uint* NF_RESTRICT w,
    NF_GLOBAL const int* NF_RESTRICT idx,
    NF_GLOBAL uint* NF_RESTRICT out,
    NF_GLOBAL int* NF_RESTRICT err)
{
    const int t = tr_gid(n_idx * dim);
    if (t < 0) return;
    const int i = t / dim;
    const int j = t - i * dim;
    int k = idx[idx_off + i];
    if (oob_mode == TR_IDX_OOB_CLAMP && rows > 0) {
        k = k < rows - 1 ? k : rows - 1;
        k = k > 0 ? k : 0;
    }
    if (k < 0 || k >= rows) {
        tr_report_index_error(err, err_slot, k, 0, rows);
        out[out_off + t] = 0u;
        return;
    }
    out[out_off + t] = w[w_off + k * dim + j];
}

NF_KERNEL void TR_LayerNormRowsBackward_F32(
    const int rows,
    const int cols,
    const float eps,
    const int has_gamma,
    const int x_off,
    const int g_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT gamma,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx,
    NF_GLOBAL float* NF_RESTRICT stats)
{
    NF_LOCAL float red[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const int lid = (int)get_local_id(0);
    const int src = x_off + row * cols;
    const int gsrc = gy_off + row * cols;
    const int dst = gx_off + row * cols;

    const float mean = tr_wg_sum(red, tr_n_lane_sum(x, src, cols, lid)) / (float)cols;
    const float var = tr_wg_sum(red, tr_n_lane_sqdev(x, src, cols, mean, lid)) / (float)cols;
    const float sdev = sqrt(var + eps);

    float acc_d = 0.0f;
    float acc_dx = 0.0f;
    for (int j = lid; j < cols; j += TR_LOCAL) {
        const float xhat = (x[src + j] - mean) / sdev;
        float d = gy[gsrc + j];
        if (has_gamma) d = d * gamma[g_off + j];
        acc_d = acc_d + d;
        acc_dx = fma(d, xhat, acc_dx);
    }
    const float a = tr_wg_sum(red, acc_d) / (float)cols;
    const float q = tr_wg_sum(red, acc_dx) / (float)cols;

    for (int j = lid; j < cols; j += TR_LOCAL) {
        const float xhat = (x[src + j] - mean) / sdev;
        float d = gy[gsrc + j];
        if (has_gamma) d = d * gamma[g_off + j];
        gx[dst + j] = ((d - a) - xhat * q) / sdev;
    }
    if (lid == 0) {
        stats[2 * row] = mean;
        stats[2 * row + 1] = sdev;
    }
}

NF_KERNEL void TR_RMSNormRowsBackward_F32(
    const int rows,
    const int cols,
    const float eps,
    const int has_weight,
    const int x_off,
    const int w_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx,
    NF_GLOBAL float* NF_RESTRICT stats)
{
    NF_LOCAL float red[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const int lid = (int)get_local_id(0);
    const int src = x_off + row * cols;
    const int gsrc = gy_off + row * cols;
    const int dst = gx_off + row * cols;

    const float sumsq = tr_wg_sum(red, tr_n_lane_sumsq(x, src, cols, lid));
    const float rms = sqrt(sumsq / (float)cols + eps);

    float acc_dx = 0.0f;
    for (int j = lid; j < cols; j += TR_LOCAL) {
        const float xhat = x[src + j] / rms;
        float d = gy[gsrc + j];
        if (has_weight) d = d * w[w_off + j];
        acc_dx = fma(d, xhat, acc_dx);
    }
    const float q = tr_wg_sum(red, acc_dx) / (float)cols;

    for (int j = lid; j < cols; j += TR_LOCAL) {
        const float xhat = x[src + j] / rms;
        float d = gy[gsrc + j];
        if (has_weight) d = d * w[w_off + j];
        gx[dst + j] = (d - xhat * q) / rms;
    }
    if (lid == 0) {
        stats[row] = rms;
    }
}

NF_KERNEL void TR_NormAffineGradCols_F32(
    const int mode,
    const int rows,
    const int cols,
    const int chunks,
    const int has_beta,
    const int x_off,
    const int gy_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL const float* NF_RESTRICT stats,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = (cols > 0 && chunks > 0) ? tr_gid(chunks * cols) : -1;
    if (i < 0) return;
    const int chunk = i / cols;
    const int col = i - chunk * cols;
    const uint rs = (uint)rows / (uint)chunks + ((uint)rows % (uint)chunks != 0u ? 1u : 0u);
    const uint r0 = (uint)chunk * rs;
    const uint r1 = r0 + rs < (uint)rows ? r0 + rs : (uint)rows;
    float acc_g = 0.0f;
    float acc_b = 0.0f;
    for (uint r = r0; r < r1; ++r) {
        const int ri = (int)r;
        const int e = ri * cols + col;
        const float xv = x[x_off + e];
        const float xhat = mode == TR_N_AFF_RMS ? xv / stats[ri] : (xv - stats[2 * ri]) / stats[2 * ri + 1];
        const float g = gy[gy_off + e];
        acc_g = fma(g, xhat, acc_g);
        acc_b = acc_b + g;
    }
    out[out_off + i] = acc_g;
    if (has_beta) out[out_off + chunks * cols + i] = acc_b;
}

NF_KERNEL void TR_ChannelGradSums_F32(
    const int B,
    const int C,
    const int HW,
    const int G,
    const float eps,
    const int x_off,
    const int gy_off,
    const int sums_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT mean,
    NF_GLOBAL const float* NF_RESTRICT var,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT sums)
{
    NF_LOCAL float red[TR_LOCAL];
    const int c = (int)get_group_id(0);
    if (c >= C) return;
    const int lid = (int)get_local_id(0);
    const int count = B * HW;

    float acc_b = 0.0f;
    float acc_g = 0.0f;
    for (int j = lid; j < count; j += TR_LOCAL) {
        const int b = j / HW;
        const int e = (b * C + c) * HW + (j - b * HW);
        const int k = G == 0 ? c : b * G + c / (C / G);
        const float sdev = sqrt(var[k] + eps);
        const float xhat = (x[x_off + e] - mean[k]) / sdev;
        const float g = gy[gy_off + e];
        acc_b = acc_b + g;
        acc_g = fma(g, xhat, acc_g);
    }
    const float sb = tr_wg_sum(red, acc_b);
    const float sg = tr_wg_sum(red, acc_g);

    if (lid == 0) {
        sums[sums_off + c] = sb;
        sums[sums_off + C + c] = sg;
    }
}

NF_KERNEL void TR_GroupGradSums_F32(
    const int B,
    const int C,
    const int HW,
    const int G,
    const float eps,
    const int has_affine,
    const int x_off,
    const int gy_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT mean,
    NF_GLOBAL const float* NF_RESTRICT var,
    NF_GLOBAL const float* NF_RESTRICT gamma,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT rsums)
{
    NF_LOCAL float red[TR_LOCAL];
    const int r = (int)get_group_id(0);
    if (r >= B * G) return;
    const int lid = (int)get_local_id(0);
    const int cpg = C / G;
    const int m = cpg * HW;
    const int base = r * m;
    const int g = r % G;
    const float sdev = sqrt(var[r] + eps);
    const float mu = mean[r];

    float acc_d = 0.0f;
    float acc_dx = 0.0f;
    for (int j = lid; j < m; j += TR_LOCAL) {
        const float xhat = (x[x_off + base + j] - mu) / sdev;
        float d = gy[gy_off + base + j];
        if (has_affine) d = d * gamma[g * cpg + j / HW];
        acc_d = acc_d + d;
        acc_dx = fma(d, xhat, acc_dx);
    }
    const float s1 = tr_wg_sum(red, acc_d);
    const float s2 = tr_wg_sum(red, acc_dx);

    if (lid == 0) {
        rsums[r] = s1;
        rsums[B * G + r] = s2;
    }
}

NF_KERNEL void TR_ChannelNormBackwardApply_F32(
    const int mode,
    const int B,
    const int C,
    const int HW,
    const int G,
    const int count,
    const float eps,
    const int has_affine,
    const int x_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT mean,
    NF_GLOBAL const float* NF_RESTRICT var,
    NF_GLOBAL const float* NF_RESTRICT gamma,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL const float* NF_RESTRICT sums,
    NF_GLOBAL float* NF_RESTRICT gx)
{
    const int e = tr_gid(B * C * HW);
    if (e < 0) return;
    const int bc = e / HW;
    const int c = bc % C;
    const float g = gy[gy_off + e];

    if (mode == TR_N_CB_EVAL) {
        const float sdev = sqrt(var[c] + eps);
        float d = g;
        if (has_affine) d = d * gamma[c];
        gx[gx_off + e] = d / sdev;
        return;
    }

    const float n = (float)count;
    if (mode == TR_N_CB_GROUP) {
        const int k = (bc / C) * G + c / (C / G);
        const int nk = B * G;
        const float sdev = sqrt(var[k] + eps);
        const float xhat = (x[x_off + e] - mean[k]) / sdev;
        float d = g;
        if (has_affine) d = d * gamma[c];
        gx[gx_off + e] = ((d - sums[k] / n) - xhat * (sums[nk + k] / n)) / sdev;
        return;
    }

    const float sdev = sqrt(var[c] + eps);
    const float xhat = (x[x_off + e] - mean[c]) / sdev;
    float t = (g - sums[c] / n) - xhat * (sums[C + c] / n);
    if (has_affine) t = gamma[c] * t;
    gx[gx_off + e] = t / sdev;
}

NF_KERNEL void TR_IndexAddRows_F32(
    const int n_idx,
    const int dim,
    const int rows,
    const int row_blocks,
    const int oob_mode,
    const int skip_row,
    const int gy_off,
    const int idx_off,
    const int gw_off,
    NF_GLOBAL const int* NF_RESTRICT idx,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gw)
{
    const int t = (dim > 0 && row_blocks > 0) ? tr_gid(dim * row_blocks) : -1;
    if (t < 0) return;
    const int blk = t / dim;
    const int j = t - blk * dim;
    const uint bs = (uint)rows / (uint)row_blocks + ((uint)rows % (uint)row_blocks != 0u ? 1u : 0u);
    const uint r0u = (uint)blk * bs;
    if (r0u >= (uint)rows) return;
    const int r0 = (int)r0u;
    const int r1 = (int)(r0u + bs < (uint)rows ? r0u + bs : (uint)rows);

    for (int i = 0; i < n_idx; ++i) {
        int k = idx[idx_off + i];
        if (oob_mode == TR_IDX_OOB_CLAMP) {
            k = k < rows - 1 ? k : rows - 1;
            k = k > 0 ? k : 0;
        }
        if (k < r0 || k >= r1 || k == skip_row) continue;
        const int o = gw_off + k * dim + j;
        gw[o] = gw[o] + gy[gy_off + i * dim + j];
    }
}
