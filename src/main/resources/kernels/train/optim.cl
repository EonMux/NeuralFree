/* optim.cl - optimizer steps, gradient-norm clipping and loss-scale unscaling. Requires: common/dialect.h, common/train.h */
/*
 * Every kernel works on FLOAT32 tensors given as (buffer, element offset). Optimizer steps update the
 * parameter p and its state buffers in place, one work-item per element; each work-item reads and writes
 * only its own element, so no parameter is NF_RESTRICT (in-place use is the contract, and norms/coef may
 * share one scratch buffer at disjoint offsets). The formulas are the CPU loops of the Java optimizers,
 * operation by operation and in Java's left-to-right association (a * b * c is (a * b) * c); the CPU method
 * each statement mirrors is cited at the statement. Java's (float) Math.sqrt(x) of a float is the correctly
 * rounded float square root, i.e. sqrt() here. Weight-decay terms are applied only when weight_decay != 0,
 * exactly as the CPU guards them, so a zero decay never turns an infinite parameter into NaN.
 *
 * TR_SgdStep_F32(i n, f lr, f momentum, f weight_decay, i nesterov, i has_momentum,
 *                i p_off, i g_off, i v_off, RW p, R g, RW v)
 *     optim/SGD.java stepWithoutMomentum (has_momentum == 0) and stepWithMomentum (has_momentum != 0;
 *     the host passes has_momentum = momentum > 0, the branch of SGD.step). Without momentum v is never
 *     accessed (bind the placeholder).
 * TR_AdamStep_F32(i n, i mode, f lr, f beta1, f beta2, f eps, f weight_decay, f bc1, f bc2,
 *                 i p_off, i g_off, i m_off, i v_off, RW p, R g, RW m, RW v)
 *     The step size is lr / bc1 (stepSize of optim/Adam.java and optim/AdamW.java step) and mHat is m.
 *     mode TR_ADAM_MODE_ADAM  (optim/Adam.java stepParameter): L2 decay g += weight_decay * p. The host
 *         passes lr = stepSize as Adam computes it and bc1 = 1 (lr / 1 is lr).
 *     mode TR_ADAM_MODE_ADAMW (optim/AdamW.java stepParameter): decoupled decay p -= lr * weight_decay * p
 *         with the raw learning rate, then the Adam update with stepSize = lr / bc1; the host passes the raw
 *         lr and bc1 = 1 - beta1^t.
 *     bc2 = 1 - beta2^t in both modes (vHat = v / bc2).
 * TR_RmspropStep_F32(i n, f lr, f alpha, f eps, f weight_decay, f momentum, i centered, i has_momentum,
 *                    i p_off, i g_off, i sq_off, i buf_off, i gavg_off, RW p, R g, RW sq, RW buf, RW gavg)
 *     optim/RMSprop.java stepPlain (centered == 0) and stepCentered (centered != 0). The host passes
 *     has_momentum = momentum > 0, the momentum branch of both methods. buf is accessed only when
 *     has_momentum != 0 and gavg only when centered != 0; otherwise bind the placeholder.
 * TR_AdagradStep_F32(i n, f clr, f eps, f weight_decay, i p_off, i g_off, i acc_off, RW p, R g, RW acc)
 *     optim/Adagrad.java stepParameter; clr is the learning rate Adagrad applies (its lr).
 * TR_LambUpdate_F32(i n, f beta1, f beta2, f eps, f weight_decay, f bc1, f bc2,
 *                   i p_off, i g_off, i m_off, i v_off, i u_off, R p, R g, RW m, RW v, W u)
 *     optim/LAMB.java stepParameter, first loop: updates m and v and writes the update r into u. The host
 *     passes bc1 = 1 - beta1^t and bc2 = 1 - beta2^t, or bc1 = bc2 = 1 when bias correction is off (x / 1
 *     is x, the uncorrected mHat = m and vHat = v). p is read only.
 *     The host then stores sum(p^2) into norms[norms_off] and sum(u^2) into norms[norms_off + 1]
 *     (LAMB's weightNorm and updateNorm sums) with TR_ReduceRows SUMSQ.
 * TR_LambApply_F32(i n, f lr, f eps, i p_off, i u_off, i norms_off, RW p, R u, R norms)
 *     optim/LAMB.java stepParameter after the first loop: trust ratio from the two norms, then
 *     p -= lr * trust * u.
 * TR_GradNormCoef_F32(i count, f max_norm, i norms_off, i coef_off, R norms, W coef)
 *     training/GradientClipping.java clipByNormOnHost for one parameter group: total = sqrt(sum of
 *     norms[norms_off + k], k = 0..count-1, added in ascending k); coef[coef_off] = total > max_norm
 *     ? max_norm / (total + 1e-6f) : 1; coef[coef_off + 1] = total. A NaN total compares false, as in Java,
 *     so its coefficient is 1.
 * TR_ScaleByDevice_F32(i n, i x_off, i coef_off, RW x, R coef)
 *     x[x_off + i] *= coef[coef_off] (GradientClipping.java clipByNormOnHost). A coefficient of exactly 1
 *     leaves x untouched, which is the IEEE identity and keeps denormal gradients on devices that flush
 *     them, as the CPU path does when it does not clip.
 * TR_UnscaleCheck_F32(i n, f inv_scale, i g_off, i flag_off, RW g, W int flag)
 *     g[g_off + i] *= inv_scale (precision/MixedPrecisionTrainer.java unscaleGradientsAndCheckOverflow,
 *     which multiplies by the same inv_scale = 1 / scale): the single-rounding class for any scale. Any
 *     non-finite result (NaN or +-inf, by bit pattern) writes 1 to flag[flag_off]; the host zeroes that
 *     slot before the launch and reads it afterwards. Concurrent writers store the same value.
 *
 * Operation ids (mirrored in shaders/train/optim.comp and tensor/gpu/TrainOps.java):
 *     TR_ADAM_MODE_ADAM, TR_ADAM_MODE_ADAMW: the mode argument of TR_AdamStep_F32.
 *
 * Dispatch: 1-D, local TR_LOCAL, global = n rounded up, except TR_GradNormCoef_F32: one work-group
 * (launch over 1 item), work-item 0 sums sequentially. No barriers and no local memory.
 */

#define TR_ADAM_MODE_ADAM  0
#define TR_ADAM_MODE_ADAMW 1

/* Java Float.isFinite negated, decided on the bit pattern (exponent all ones: NaN or infinity). */
NF_INLINE int tr_optim_nonfinite(const float x) {
    return (as_uint(x) & 0x7f800000u) == 0x7f800000u;
}

/* Java x > 0f for a value that may be NaN (NaN compares false), robust to no-NaN compiler assumptions. */
NF_INLINE int tr_optim_positive(const float x) {
    return !tr_isnan(x) && x > 0.0f;
}

NF_KERNEL void TR_SgdStep_F32(
    const int n,
    const float lr,
    const float momentum,
    const float weight_decay,
    const int nesterov,
    const int has_momentum,
    const int p_off,
    const int g_off,
    const int v_off,
    NF_GLOBAL float* p,
    NF_GLOBAL const float* g,
    NF_GLOBAL float* v)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float pi = p[p_off + i];
    float gi = g[g_off + i];
    if (weight_decay != 0.0f) {
        gi = gi + weight_decay * pi;                          /* SGD.stepWithoutMomentum / stepWithMomentum */
    }
    if (has_momentum != 0) {
        const float vi = momentum * v[v_off + i] + gi;        /* SGD.stepWithMomentum */
        v[v_off + i] = vi;
        if (nesterov != 0) {
            p[p_off + i] = pi - lr * (momentum * vi + gi);    /* SGD.stepWithMomentum, nesterov */
        } else {
            p[p_off + i] = pi - lr * vi;                      /* SGD.stepWithMomentum */
        }
    } else {
        p[p_off + i] = pi - lr * gi;                          /* SGD.stepWithoutMomentum */
    }
}

NF_KERNEL void TR_AdamStep_F32(
    const int n,
    const int mode,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bc1,
    const float bc2,
    const int p_off,
    const int g_off,
    const int m_off,
    const int v_off,
    NF_GLOBAL float* p,
    NF_GLOBAL const float* g,
    NF_GLOBAL float* m,
    NF_GLOBAL float* v)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    float pi = p[p_off + i];
    float gi = g[g_off + i];
    if (weight_decay != 0.0f) {
        if (mode == TR_ADAM_MODE_ADAMW) {
            pi = pi - lr * weight_decay * pi;                 /* AdamW.stepParameter */
        } else {
            gi = gi + weight_decay * pi;                      /* Adam.stepParameter */
        }
    }
    const float mi = beta1 * m[m_off + i] + (1.0f - beta1) * gi;   /* Adam/AdamW.stepParameter */
    const float vi = beta2 * v[v_off + i] + (1.0f - beta2) * gi * gi;   /* Adam/AdamW.stepParameter */
    m[m_off + i] = mi;
    v[v_off + i] = vi;
    const float step_size = lr / bc1;                         /* Adam/AdamW.step */
    const float v_hat = vi / bc2;                             /* Adam/AdamW.stepParameter */
    p[p_off + i] = pi - step_size * mi / (sqrt(v_hat) + eps); /* Adam/AdamW.stepParameter */
}

NF_KERNEL void TR_RmspropStep_F32(
    const int n,
    const float lr,
    const float alpha,
    const float eps,
    const float weight_decay,
    const float momentum,
    const int centered,
    const int has_momentum,
    const int p_off,
    const int g_off,
    const int sq_off,
    const int buf_off,
    const int gavg_off,
    NF_GLOBAL float* p,
    NF_GLOBAL const float* g,
    NF_GLOBAL float* sq,
    NF_GLOBAL float* buf,
    NF_GLOBAL float* gavg)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float pi = p[p_off + i];
    float gi = g[g_off + i];
    if (weight_decay != 0.0f) {
        gi = gi + weight_decay * pi;                          /* RMSprop.stepPlain / stepCentered */
    }
    float denom_sq;
    if (centered != 0) {
        const float ai = alpha * gavg[gavg_off + i] + (1.0f - alpha) * gi;   /* RMSprop.stepCentered */
        gavg[gavg_off + i] = ai;
        const float si = alpha * sq[sq_off + i] + (1.0f - alpha) * gi * gi;  /* RMSprop.stepCentered */
        sq[sq_off + i] = si;
        denom_sq = si - ai * ai;                              /* RMSprop.stepCentered */
    } else {
        const float si = alpha * sq[sq_off + i] + (1.0f - alpha) * gi * gi;  /* RMSprop.stepPlain */
        sq[sq_off + i] = si;
        denom_sq = si;
    }
    const float update = gi / (sqrt(denom_sq) + eps);         /* RMSprop.stepPlain / stepCentered */
    if (has_momentum != 0) {
        const float bi = momentum * buf[buf_off + i] + update;   /* RMSprop.stepPlain / stepCentered */
        buf[buf_off + i] = bi;
        p[p_off + i] = pi - lr * bi;                          /* RMSprop.stepPlain / stepCentered */
    } else {
        p[p_off + i] = pi - lr * update;                      /* RMSprop.stepPlain / stepCentered */
    }
}

NF_KERNEL void TR_AdagradStep_F32(
    const int n,
    const float clr,
    const float eps,
    const float weight_decay,
    const int p_off,
    const int g_off,
    const int acc_off,
    NF_GLOBAL float* p,
    NF_GLOBAL const float* g,
    NF_GLOBAL float* acc)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float pi = p[p_off + i];
    float gi = g[g_off + i];
    if (weight_decay != 0.0f) {
        gi = gi + weight_decay * pi;                          /* Adagrad.stepParameter */
    }
    const float ai = acc[acc_off + i] + gi * gi;              /* Adagrad.stepParameter */
    acc[acc_off + i] = ai;
    p[p_off + i] = pi - clr * gi / (sqrt(ai) + eps);          /* Adagrad.stepParameter */
}

NF_KERNEL void TR_LambUpdate_F32(
    const int n,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bc1,
    const float bc2,
    const int p_off,
    const int g_off,
    const int m_off,
    const int v_off,
    const int u_off,
    NF_GLOBAL const float* p,
    NF_GLOBAL const float* g,
    NF_GLOBAL float* m,
    NF_GLOBAL float* v,
    NF_GLOBAL float* u)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float gi = g[g_off + i];
    const float mi = beta1 * m[m_off + i] + (1.0f - beta1) * gi;       /* LAMB.stepParameter */
    const float vi = beta2 * v[v_off + i] + (1.0f - beta2) * gi * gi;  /* LAMB.stepParameter */
    m[m_off + i] = mi;
    v[v_off + i] = vi;
    const float m_hat = mi / bc1;                             /* LAMB.stepParameter (mHat = m when bc1 = 1) */
    const float v_hat = vi / bc2;                             /* LAMB.stepParameter (vHat = v when bc2 = 1) */
    float r = m_hat / (sqrt(v_hat) + eps);                    /* LAMB.stepParameter */
    if (weight_decay != 0.0f) {
        r = r + weight_decay * p[p_off + i];                  /* LAMB.stepParameter */
    }
    u[u_off + i] = r;
}

NF_KERNEL void TR_LambApply_F32(
    const int n,
    const float lr,
    const float eps,
    const int p_off,
    const int u_off,
    const int norms_off,
    NF_GLOBAL float* p,
    NF_GLOBAL const float* u,
    NF_GLOBAL const float* norms)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float weight_norm = sqrt(norms[norms_off]);         /* LAMB.stepParameter */
    const float update_norm = sqrt(norms[norms_off + 1]);     /* LAMB.stepParameter */
    float trust = 1.0f;                                       /* LAMB.stepParameter */
    if (tr_optim_positive(weight_norm) && tr_optim_positive(update_norm)) {
        trust = weight_norm / (update_norm + eps);            /* LAMB.stepParameter */
    }
    p[p_off + i] = p[p_off + i] - lr * trust * u[u_off + i];  /* LAMB.stepParameter */
}

NF_KERNEL void TR_GradNormCoef_F32(
    const int count,
    const float max_norm,
    const int norms_off,
    const int coef_off,
    NF_GLOBAL const float* norms,
    NF_GLOBAL float* coef)
{
    if (tr_gid(1) < 0) return;
    float sum = 0.0f;
    for (int k = 0; k < count; ++k) {
        sum = sum + norms[norms_off + k];                     /* clipByNormOnHost, per-tensor partials in order */
    }
    const float total = sqrt(sum);                            /* GradientClipping.clipByNormOnHost */
    float c = 1.0f;
    if (!tr_isnan(total) && total > max_norm) {
        c = max_norm / (total + 1e-6f);                       /* GradientClipping.clipByNormOnHost */
    }
    coef[coef_off] = c;
    coef[coef_off + 1] = total;
}

NF_KERNEL void TR_ScaleByDevice_F32(
    const int n,
    const int x_off,
    const int coef_off,
    NF_GLOBAL float* x,
    NF_GLOBAL const float* coef)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float c = coef[coef_off];
    if (c == 1.0f) return;
    x[x_off + i] = x[x_off + i] * c;                          /* GradientClipping.clipByNormOnHost */
}

NF_KERNEL void TR_UnscaleCheck_F32(
    const int n,
    const float inv_scale,
    const int g_off,
    const int flag_off,
    NF_GLOBAL float* g,
    NF_GLOBAL int* flag)
{
    const int i = tr_gid(n);
    if (i < 0) return;
    const float r = g[g_off + i] * inv_scale;                 /* MixedPrecisionTrainer unscale step */
    g[g_off + i] = r;
    if (tr_optim_nonfinite(r)) {
        flag[flag_off] = 1;
    }
}
