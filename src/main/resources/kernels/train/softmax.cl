/* softmax.cl - row softmax and log-softmax with optional masking, and their backward. Requires: common/dialect.h, common/train.h */
/*
 * A row is a contiguous run of cols elements starting at <off> + r * cols; the host moves the softmax dim
 * to the end (one strided copy) before launching. One TR_LOCAL group per row: lane l visits columns
 * l, l + TR_LOCAL, ... in ascending order and the fixed tr_wg_* tree combines the lanes. Offsets are element
 * offsets.
 *
 * Flags of TR_SoftmaxRows_F32:
 *   TR_SM_LOG    1  log-softmax instead of softmax
 *   TR_SM_MASKED 2  -inf inputs are masked: they are left out of the sum and their probability is exactly 0,
 *                   so an all -inf row gives all 0 (tensor/CpuReductions.java maskedSoftmax, the attention
 *                   masking of nn/transformer/MultiHeadAttention.java attentionWeights). In log
 *                   mode a masked entry writes -inf, the log of that 0 probability, so exp of the log-softmax
 *                   equals the masked softmax and the log-mode backward routes no gradient through it.
 *
 * TR_SoftmaxRows_F32(i rows, i cols, i flags, i in_off, i out_off, R in, W out)
 *     Per row, with x = in[in_off + r*cols + c]
 *     (tensor/CpuReductions.java softmax, logSoftmax and maskedSoftmax, all through softmaxRows):
 *       mx  = Math.max fold of x from -inf (tr_jmax; NaN propagates)
 *       s   = sum of exp(x - mx) over the row (masked: over the entries that are not -inf)
 *       softmax:     out = exp(x - mx) / s
 *       log-softmax: out = x - (mx + log(s))
 *     Masked entries write 0 (softmax) or -inf (log-softmax). Unmasked, an all -inf row gives NaN, as the CPU.
 * TR_SoftmaxRowsBackward_F32(i rows, i cols, i log_mode, i y_off, i gy_off, i gx_off, R y, R gy, W gx)
 *     y is the forward output (probabilities, or log-probabilities when log_mode != 0), gy its gradient.
 *       softmax (tensor/CpuReductions.java softmaxBackward, used by autograd/GradFunctions.java SoftmaxGrad):
 *         dot = sum of y*gy (fma accumulation);  gx = y * (gy - dot)
 *       log-softmax (CpuReductions.java softmaxBackward in log mode, used by GradFunctions.java LogSoftmaxGrad):
 *         s = sum of gy;                          gx = gy - exp(y) * s
 *
 * Dispatch: 1-D, local TR_LOCAL, rows groups. Groups past the last row return as a whole (uniform control
 * flow), so a rounded or folded launch is harmless. Inputs and outputs never alias.
 */

#define TR_SM_LOG    1
#define TR_SM_MASKED 2

/* True when x is -inf by bit pattern (the CPU's x == Float.NEGATIVE_INFINITY). */
NF_INLINE int tr_sm_is_neginf(const float x) {
    return as_uint(x) == 0xff800000u;
}

NF_KERNEL void TR_SoftmaxRows_F32(
    const int rows,
    const int cols,
    const int flags,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT in,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float buf[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const uint lid = get_local_id(0);
    const int masked = (flags & TR_SM_MASKED) != 0;
    const int ib = in_off + row * cols;
    const int ob = out_off + row * cols;

    float mx = -NF_INF;
    for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
        mx = tr_jmax(mx, in[ib + (int)c]);
    }
    mx = tr_wg_max(buf, mx);

    float s = 0.0f;
    for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
        const float x = in[ib + (int)c];
        if (!(masked && tr_sm_is_neginf(x))) s = s + exp(x - mx);
    }
    s = tr_wg_sum(buf, s);

    if ((flags & TR_SM_LOG) != 0) {
        const float log_sum = mx + log(s);
        for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
            const float x = in[ib + (int)c];
            out[ob + (int)c] = (masked && tr_sm_is_neginf(x)) ? -NF_INF : x - log_sum;
        }
    } else {
        for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
            const float x = in[ib + (int)c];
            out[ob + (int)c] = (masked && tr_sm_is_neginf(x)) ? 0.0f : exp(x - mx) / s;
        }
    }
}

NF_KERNEL void TR_SoftmaxRowsBackward_F32(
    const int rows,
    const int cols,
    const int log_mode,
    const int y_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const float* NF_RESTRICT y,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx)
{
    NF_LOCAL float buf[TR_LOCAL];
    const int row = (int)get_group_id(0);
    if (row >= rows) return;
    const uint lid = get_local_id(0);
    const int yb = y_off + row * cols;
    const int gb = gy_off + row * cols;
    const int ob = gx_off + row * cols;

    float acc = 0.0f;
    if (log_mode != 0) {
        for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
            acc = acc + gy[gb + (int)c];
        }
    } else {
        for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
            acc = fma(y[yb + (int)c], gy[gb + (int)c], acc);
        }
    }
    const float s = tr_wg_sum(buf, acc);

    if (log_mode != 0) {
        for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
            gx[ob + (int)c] = gy[gb + (int)c] - exp(y[yb + (int)c]) * s;
        }
    } else {
        for (uint c = lid; c < (uint)cols; c += TR_LOCAL) {
            const float yv = y[yb + (int)c];
            gx[ob + (int)c] = yv * (gy[gb + (int)c] - s);
        }
    }
}
