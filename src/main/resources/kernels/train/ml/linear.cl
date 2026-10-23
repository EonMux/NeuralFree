/* ml/linear.cl - perceptron and logistic-regression forward, gradient and update. Requires: common/dialect.h, common/train.h */
/*
 * Single logistic units with device-resident parameters: weights (n_features), bias (1) and the batch
 * buffers x (n_samples x n_features row-major), y, sample_w and predictions (n_samples), all at element
 * offset 0. Labels y are floats (0/1 for logistic regression; Java's float - int promotes the int label
 * the same way). Sums start at +0 (the forward pass at the bias) and run in ascending sample or feature
 * order; multiply-add accumulations are fused (acc = fma(a, b, acc)), so a CPU reference written with
 * Math.fma reproduces them bit for bit on normal values. Every other operation is one rounding each in
 * Java's left-to-right association (a * b * c is (a * b) * c).
 *
 * The logistic is the numerically stable form of ml/LogisticRegression.java sigmoid:
 *     z >= 0 ? 1 / (1 + exp(-z)) : exp(z) / (1 + exp(z))      (NaN takes the second branch and stays NaN)
 *
 * Perceptron (ml/Perceptron.java):
 * ML_Perceptron_Forward(i n_samples, i n_features, R x, R weights, R bias, W predictions)
 *     z = bias[0] + sum_j weights[j] * x[i*n_features + j]; predictions[i] = sigmoid(z). n_samples items.
 * ML_Perceptron_Grad(i n_samples, i n_features, R x, R y, R sample_w, R predictions, W grad_w, W grad_b)
 *     e_i = predictions[i] - y[i];
 *     grad_w[j] = (sum_i (sample_w[i] * e_i) * x[i*n_features + j]) / n_samples;
 *     work-item 0 also writes grad_b[0] = (sum_i sample_w[i] * e_i) / n_samples. n_features items.
 * ML_Perceptron_Update(i n_features, f learning_rate, f l2, RW weights, R grad_w, RW bias, R grad_b)
 *     weights[j] = weights[j] - learning_rate * (grad_w[j] + l2 * weights[j]);
 *     work-item 0 also steps bias[0] = bias[0] - learning_rate * grad_b[0] (no decay). n_features items.
 *
 * Logistic regression (ml/LogisticRegression.java fit, with the prediction taking x into account):
 * ML_LogReg_Forward(i n_samples, i n_features, R x, R weights, R bias, W predictions)
 *     As ML_Perceptron_Forward (predictProbability: z = bias + sum_j weights[j] * x[i][j]).
 * ML_LogReg_Grad(i n_samples, i n_features, R x, R y, R predictions, W grad_w, W grad_b)
 *     computeGradientW/computeGradientB: e_i = predictions[i] - y[i];
 *     grad_w[j] = (sum_i e_i * x[i*n_features + j]) / n_samples;
 *     work-item 0 also writes grad_b[0] = (sum_i e_i) / n_samples. n_features items.
 * ML_LogReg_Update(i n_features, f learning_rate, f regularization, RW weights, R grad_w, RW bias,
 *                  R grad_b, W max_grad)
 *     One iteration's tail of fit: max_grad[0] = Math.max, starting from 0, over |grad_w[j]| and then
 *     |grad_b[0]| (the convergence measure compared with the tolerance); weights[j] = weights[j] -
 *     learning_rate * (grad_w[j] + regularization * weights[j]); bias[0] = bias[0] - learning_rate *
 *     grad_b[0]. One work-group: work-item l handles features l, l + TR_LOCAL, ...; tr_wg_max combines the
 *     maxima (Math.max is associative and commutative under tr_jmax; NaN propagates).
 *
 * Preconditions (host): n_samples, n_features >= 1 and n_samples*n_features at most 2^31-1. Distinct
 * buffers never alias.
 *
 * Dispatch: 1-D, local TR_LOCAL, global = item count rounded up; ML_LogReg_Update is one work-group
 * (global = local = TR_LOCAL).
 */

/* The stable logistic of ml/LogisticRegression.java sigmoid (float exp; Java rounds Math.exp to float). */
NF_INLINE float ml_lin_sigmoid(const float z) {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + exp(-z));
    }
    const float e = exp(z);
    return e / (1.0f + e);
}

/* bias[0] + sum over j of weights[j] * x[row + j], ascending, fused. */
NF_INLINE float ml_lin_logit(NF_GLOBAL const float* x, const int row, NF_GLOBAL const float* weights,
                             NF_GLOBAL const float* bias, const int n_features) {
    float z = bias[0];
    for (int j = 0; j < n_features; ++j) {
        z = fma(weights[j], x[row + j], z);
    }
    return z;
}

NF_KERNEL void ML_Perceptron_Forward(
    const int n_samples,
    const int n_features,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT weights,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* NF_RESTRICT predictions)
{
    const int i = tr_gid(n_samples);
    if (i < 0) return;
    predictions[i] = ml_lin_sigmoid(ml_lin_logit(x, i * n_features, weights, bias, n_features));
}

NF_KERNEL void ML_Perceptron_Grad(
    const int n_samples,
    const int n_features,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT y,
    NF_GLOBAL const float* NF_RESTRICT sample_w,
    NF_GLOBAL const float* NF_RESTRICT predictions,
    NF_GLOBAL float* NF_RESTRICT grad_w,
    NF_GLOBAL float* NF_RESTRICT grad_b)
{
    const int j = tr_gid(n_features);
    if (j < 0) return;
    const float count = (float)n_samples;
    float sum = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
        sum = fma(sample_w[i] * (predictions[i] - y[i]), x[i * n_features + j], sum);
    }
    grad_w[j] = sum / count;
    if (j == 0) {
        float b = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            b = fma(sample_w[i], predictions[i] - y[i], b);
        }
        grad_b[0] = b / count;
    }
}

NF_KERNEL void ML_Perceptron_Update(
    const int n_features,
    const float learning_rate,
    const float l2,
    NF_GLOBAL float* NF_RESTRICT weights,
    NF_GLOBAL const float* NF_RESTRICT grad_w,
    NF_GLOBAL float* NF_RESTRICT bias,
    NF_GLOBAL const float* NF_RESTRICT grad_b)
{
    const int j = tr_gid(n_features);
    if (j < 0) return;
    const float w = weights[j];
    weights[j] = w - learning_rate * (grad_w[j] + l2 * w);
    if (j == 0) {
        bias[0] = bias[0] - learning_rate * grad_b[0];
    }
}

NF_KERNEL void ML_LogReg_Forward(
    const int n_samples,
    const int n_features,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT weights,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* NF_RESTRICT predictions)
{
    const int i = tr_gid(n_samples);
    if (i < 0) return;
    predictions[i] = ml_lin_sigmoid(ml_lin_logit(x, i * n_features, weights, bias, n_features));
}

NF_KERNEL void ML_LogReg_Grad(
    const int n_samples,
    const int n_features,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT y,
    NF_GLOBAL const float* NF_RESTRICT predictions,
    NF_GLOBAL float* NF_RESTRICT grad_w,
    NF_GLOBAL float* NF_RESTRICT grad_b)
{
    const int j = tr_gid(n_features);
    if (j < 0) return;
    const float count = (float)n_samples;
    float sum = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
        sum = fma(predictions[i] - y[i], x[i * n_features + j], sum);
    }
    grad_w[j] = sum / count;
    if (j == 0) {
        float b = 0.0f;
        for (int i = 0; i < n_samples; ++i) {
            b = b + (predictions[i] - y[i]);
        }
        grad_b[0] = b / count;
    }
}

NF_KERNEL void ML_LogReg_Update(
    const int n_features,
    const float learning_rate,
    const float regularization,
    NF_GLOBAL float* NF_RESTRICT weights,
    NF_GLOBAL const float* NF_RESTRICT grad_w,
    NF_GLOBAL float* NF_RESTRICT bias,
    NF_GLOBAL const float* NF_RESTRICT grad_b,
    NF_GLOBAL float* NF_RESTRICT max_grad)
{
    NF_LOCAL float red[TR_LOCAL];
    const int lid = (int)get_local_id(0);
    float m = 0.0f;
    for (int j = lid; j < n_features; j += TR_LOCAL) {
        const float g = grad_w[j];
        m = tr_jmax(m, fabs(g));
        const float w = weights[j];
        weights[j] = w - learning_rate * (g + regularization * w);
    }
    m = tr_wg_max(red, m);
    if (lid == 0) {
        const float gb = grad_b[0];
        max_grad[0] = tr_jmax(m, fabs(gb));
        bias[0] = bias[0] - learning_rate * gb;
    }
}
