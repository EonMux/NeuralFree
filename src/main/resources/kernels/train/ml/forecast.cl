/* ml/forecast.cl - forecast-unit forward, gradient and update. Requires: common/dialect.h, common/train.h */
/*
 * The two linear heads of ml/ForecastUnit.java over one shared feature row, with device-resident
 * parameters:
 *   - direction: a logistic unit p = sigmoid(wd . x + bd), trained by the cross-entropy gradient (p - y) on
 *     soft labels y in [0, 1];
 *   - magnitude: a linear unit v = wm . x + bm, trained by the Huber gradient, the error clamped to
 *     [-FC_HUBER_DELTA, FC_HUBER_DELTA] as Math.max(-delta, Math.min(delta, e)) (tr_jmax/tr_jmin: NaN
 *     propagates, +0 above -0).
 * x is n_samples x n_features row-major; every buffer is at element offset 0. Sums start at +0 (the forward
 * pass at the bias) and run in ascending sample or feature order; multiply-add accumulations are fused
 * (acc = fma(a, b, acc)), so a CPU reference written with Math.fma reproduces them bit for bit on normal
 * values. Every other operation is one rounding each in Java's left-to-right association. The logistic is
 * z >= 0 ? 1 / (1 + exp(-z)) : exp(z) / (1 + exp(z)) (NaN takes the second branch and stays NaN).
 *
 * ML_Forecast_Forward(i n_samples, i n_features, R x, R dir_weights, R dir_bias, R move_weights,
 *                     R move_bias, W pred_dir, W pred_move)
 *     pred_dir[i] = sigmoid(dir_bias[0] + sum_j dir_weights[j] * x[i*n_features + j]);
 *     pred_move[i] = move_bias[0] + sum_j move_weights[j] * x[i*n_features + j]. n_samples items.
 * ML_Forecast_Grad(i n_samples, i n_features, R x, R y_dir, R y_move, R sample_w, R pred_dir, R pred_move,
 *                  W grad_dir_w, W grad_dir_b, W grad_move_w, W grad_move_b)
 *     ed_i = pred_dir[i] - y_dir[i]; em_i = huber(pred_move[i] - y_move[i]);
 *     grad_dir_w[j]  = (sum_i (sample_w[i] * ed_i) * x[i*n_features + j]) / n_samples;
 *     grad_move_w[j] = (sum_i (sample_w[i] * em_i) * x[i*n_features + j]) / n_samples;
 *     work-item 0 also writes grad_dir_b[0] = (sum_i sample_w[i] * ed_i) / n_samples and
 *     grad_move_b[0] = (sum_i sample_w[i] * em_i) / n_samples. n_features items.
 * ML_Forecast_Update(i n_features, f learning_rate, f l2, RW dir_weights, R grad_dir_w, RW dir_bias,
 *                    R grad_dir_b, RW move_weights, R grad_move_w, RW move_bias, R grad_move_b)
 *     w[j] = w[j] - learning_rate * (g[j] + l2 * w[j]) for both heads; work-item 0 also steps both biases,
 *     b = b - learning_rate * g_b (no decay). n_features items.
 *
 * Preconditions (host): n_samples, n_features >= 1 and n_samples*n_features at most 2^31-1. Distinct
 * buffers never alias.
 *
 * Dispatch: 1-D, local TR_LOCAL, global = item count rounded up. No barriers and no local memory.
 */

/* Huber bound of the magnitude head's per-sample error (ForecastUnit: errors clamped to +-1). */
#define FC_HUBER_DELTA 1.0f

/* The stable logistic (float exp). */
NF_INLINE float ml_fc_sigmoid(const float z) {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + exp(-z));
    }
    const float e = exp(z);
    return e / (1.0f + e);
}

/* Math.max(-FC_HUBER_DELTA, Math.min(FC_HUBER_DELTA, e)). */
NF_INLINE float ml_fc_huber(const float e) {
    return tr_jmax(-FC_HUBER_DELTA, tr_jmin(FC_HUBER_DELTA, e));
}

NF_KERNEL void ML_Forecast_Forward(
    const int n_samples,
    const int n_features,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT dir_weights,
    NF_GLOBAL const float* NF_RESTRICT dir_bias,
    NF_GLOBAL const float* NF_RESTRICT move_weights,
    NF_GLOBAL const float* NF_RESTRICT move_bias,
    NF_GLOBAL float* NF_RESTRICT pred_dir,
    NF_GLOBAL float* NF_RESTRICT pred_move)
{
    const int i = tr_gid(n_samples);
    if (i < 0) return;
    const int row = i * n_features;
    float zd = dir_bias[0];
    float zm = move_bias[0];
    for (int j = 0; j < n_features; ++j) {
        const float xi = x[row + j];
        zd = fma(dir_weights[j], xi, zd);
        zm = fma(move_weights[j], xi, zm);
    }
    pred_dir[i] = ml_fc_sigmoid(zd);
    pred_move[i] = zm;
}

NF_KERNEL void ML_Forecast_Grad(
    const int n_samples,
    const int n_features,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT y_dir,
    NF_GLOBAL const float* NF_RESTRICT y_move,
    NF_GLOBAL const float* NF_RESTRICT sample_w,
    NF_GLOBAL const float* NF_RESTRICT pred_dir,
    NF_GLOBAL const float* NF_RESTRICT pred_move,
    NF_GLOBAL float* NF_RESTRICT grad_dir_w,
    NF_GLOBAL float* NF_RESTRICT grad_dir_b,
    NF_GLOBAL float* NF_RESTRICT grad_move_w,
    NF_GLOBAL float* NF_RESTRICT grad_move_b)
{
    const int j = tr_gid(n_features);
    if (j < 0) return;
    const float count = (float)n_samples;
    float sum_d = 0.0f;
    float sum_m = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
        const float xi = x[i * n_features + j];
        const float sw = sample_w[i];
        const float ed = pred_dir[i] - y_dir[i];
        const float em = ml_fc_huber(pred_move[i] - y_move[i]);
        sum_d = fma(sw * ed, xi, sum_d);
        sum_m = fma(sw * em, xi, sum_m);
    }
    grad_dir_w[j] = sum_d / count;
    grad_move_w[j] = sum_m / count;
    if (j == 0) {
        float bd = 0.0f;
        float bm = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            const float sw = sample_w[i];
            bd = fma(sw, pred_dir[i] - y_dir[i], bd);
            bm = fma(sw, ml_fc_huber(pred_move[i] - y_move[i]), bm);
        }
        grad_dir_b[0] = bd / count;
        grad_move_b[0] = bm / count;
    }
}

NF_KERNEL void ML_Forecast_Update(
    const int n_features,
    const float learning_rate,
    const float l2,
    NF_GLOBAL float* NF_RESTRICT dir_weights,
    NF_GLOBAL const float* NF_RESTRICT grad_dir_w,
    NF_GLOBAL float* NF_RESTRICT dir_bias,
    NF_GLOBAL const float* NF_RESTRICT grad_dir_b,
    NF_GLOBAL float* NF_RESTRICT move_weights,
    NF_GLOBAL const float* NF_RESTRICT grad_move_w,
    NF_GLOBAL float* NF_RESTRICT move_bias,
    NF_GLOBAL const float* NF_RESTRICT grad_move_b)
{
    const int j = tr_gid(n_features);
    if (j < 0) return;
    const float wd = dir_weights[j];
    const float wm = move_weights[j];
    dir_weights[j] = wd - learning_rate * (grad_dir_w[j] + l2 * wd);
    move_weights[j] = wm - learning_rate * (grad_move_w[j] + l2 * wm);
    if (j == 0) {
        dir_bias[0] = dir_bias[0] - learning_rate * grad_dir_b[0];
        move_bias[0] = move_bias[0] - learning_rate * grad_move_b[0];
    }
}
