/* common/math.h - scalar activation, GLU, ALiBi and T5-bucket math shared by every kernel family. Requires: common/dialect.h */
/*
 * Every function follows the llama.cpp reference cited next to it, in precise (non-native) math.
 * NF_ACT_* equal gguf.arch.ActOp codes and NF_GLU_* equal gguf.arch.GluOp codes; the host validates
 * op codes, and an unknown code yields NaN so a wiring error never passes silently.
 */
#ifndef NF_MATH_H
#define NF_MATH_H

/* ---- ActOp codes ---- */
#define NF_ACT_RELU           0
#define NF_ACT_RELU_SQR       1
#define NF_ACT_GELU_TANH      2
#define NF_ACT_GELU_ERF       3
#define NF_ACT_GELU_QUICK     4
#define NF_ACT_SILU           5
#define NF_ACT_SIGMOID        6
#define NF_ACT_SOFTPLUS       7
#define NF_ACT_TANH           8
#define NF_ACT_XIELU          9
#define NF_ACT_SCALE_BIAS     10
#define NF_ACT_CLAMP          11
#define NF_ACT_EXP            12
#define NF_ACT_SQRT_SOFTPLUS  13
#define NF_ACT_IDENTITY       14
#define NF_ACT_SOFTCAP        15

/* ---- GluOp codes ---- */
#define NF_GLU_SWIGLU         0
#define NF_GLU_GEGLU_TANH     1
#define NF_GLU_GEGLU_ERF      2
#define NF_GLU_GEGLU_QUICK    3
#define NF_GLU_REGLU          4
#define NF_GLU_SWIGLU_OAI     5
#define NF_GLU_SWIGLU_CLAMP   6
#define NF_GLU_SWIGLU_LIMITED 7
#define NF_GLU_SITU           8
#define NF_GLU_MUL            9
#define NF_GLU_SIGMUL         10

#define NF_GELU_COEF_A        0.044715f
#define NF_GELU_QUICK_COEF    (-1.702f)
#define NF_SQRT_2_OVER_PI     0.79788456080286535587989211986876f
#define NF_SQRT_2_INV         0.70710678118654752440084436210484f

/* x / (1 + e^-x)  (ggml-cuda/unary.cuh:98, ggml-cpu/vec.h:1060) */
NF_INLINE float nf_silu(float x) {
    return x / (1.0f + exp(-x));
}

/* 1 / (1 + e^-x)  (ggml-cuda/unary.cu:48) */
NF_INLINE float nf_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

/* x > 20 ? x : log(1 + e^x)  (ggml-impl.h:107) */
NF_INLINE float nf_softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

/* 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))  (ggml-cuda/unary.cuh:102) */
NF_INLINE float nf_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(NF_SQRT_2_OVER_PI * x * (1.0f + NF_GELU_COEF_A * x * x)));
}

/* 0.5 x (1 + erf(x / sqrt(2)))  (ggml-cuda/unary.cu:24) */
NF_INLINE float nf_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erf(x * NF_SQRT_2_INV));
}

/* x / (1 + e^(-1.702 x))  (ggml-cuda/unary.cu:30) */
NF_INLINE float nf_gelu_quick(float x) {
    return x * (1.0f / (1.0f + exp(NF_GELU_QUICK_COEF * x)));
}

/* xIELU with load-time prepared parameters: alpha_n' = beta + softplus(alpha_n), alpha_p' = softplus(alpha_p)
 * (ggml.c:2871-2872); evaluation as ggml-cpu/unary-ops.cpp:55-62. */
NF_INLINE float nf_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    if (x > 0.0f) {
        return alpha_p * x * x + beta * x;
    }
    const float min_x_eps = fmin(x, eps);
    return (expm1(min_x_eps) - x) * alpha_n + beta * x;
}

/* ggml_clamp: MAX(MIN(x, hi), lo)  (ggml-cpu/ops.cpp clamp_f32) */
NF_INLINE float nf_clamp_ggml(float x, float lo, float hi) {
    return fmax(fmin(x, hi), lo);
}

/*
 * Unary activation dispatcher; op is an NF_ACT_* code. Parameters per op:
 *   XIELU       p0 = alpha_n', p1 = alpha_p', p2 = beta, p3 = eps
 *   SCALE_BIAS  x * p0 + p1                       (ggml_scale_bias)
 *   CLAMP       p0 = lo, p1 = hi                  (ggml_clamp)
 *   SOFTCAP     p0 * tanh(x * p1), p0 = cap, p1 = scale / cap
 * Other ops ignore p0..p3.
 */
NF_INLINE float nf_act(int op, float x, float p0, float p1, float p2, float p3) {
    switch (op) {
        case NF_ACT_RELU:          return fmax(x, 0.0f);
        case NF_ACT_RELU_SQR:      { const float r = fmax(x, 0.0f); return r * r; }
        case NF_ACT_GELU_TANH:     return nf_gelu_tanh(x);
        case NF_ACT_GELU_ERF:      return nf_gelu_erf(x);
        case NF_ACT_GELU_QUICK:    return nf_gelu_quick(x);
        case NF_ACT_SILU:          return nf_silu(x);
        case NF_ACT_SIGMOID:       return nf_sigmoid(x);
        case NF_ACT_SOFTPLUS:      return nf_softplus(x);
        case NF_ACT_TANH:          return tanh(x);
        case NF_ACT_XIELU:         return nf_xielu(x, p0, p1, p2, p3);
        case NF_ACT_SCALE_BIAS:    return x * p0 + p1;
        case NF_ACT_CLAMP:         return nf_clamp_ggml(x, p0, p1);
        case NF_ACT_EXP:           return exp(x);
        case NF_ACT_SQRT_SOFTPLUS: return sqrt(nf_softplus(x));
        case NF_ACT_IDENTITY:      return x;
        case NF_ACT_SOFTCAP:       return p0 * tanh(x * p1);
        default:                   return NF_NAN;
    }
}

/*
 * Gated-linear-unit dispatcher out = f(g, u); op is an NF_GLU_* code. Parameters per op:
 *   SWIGLU_OAI      x = min(g, limit), y = clamp(u, -limit, limit), x * sigmoid(alpha * x) * (y + 1)
 *                   (ggml-cpu/ops.cpp:3546-3549)
 *   SWIGLU_CLAMP    silu(min(g, limit)) * clamp(u, -limit, limit)       (ggml-cpu/ops.cpp:3697-3700)
 *   SWIGLU_LIMITED  min(silu(g), limit) * clamp(u, -limit, limit)       (llama-graph.cpp:2010-2016)
 *   SITU            beta * tanh(g / beta) * sigmoid(g) * (lambda > 0 ? lambda * tanh(u / lambda) : u),
 *                   beta = situ_beta, lambda = situ_linear_beta         (llama-graph.cpp:2455-2464)
 * Other ops ignore limit, alpha, beta and lambda.
 */
NF_INLINE float nf_glu(int op, float g, float u, float limit, float alpha, float beta, float lambda) {
    switch (op) {
        case NF_GLU_SWIGLU:      return nf_silu(g) * u;
        case NF_GLU_GEGLU_TANH:  return nf_gelu_tanh(g) * u;
        case NF_GLU_GEGLU_ERF:   return nf_gelu_erf(g) * u;
        case NF_GLU_GEGLU_QUICK: return nf_gelu_quick(g) * u;
        case NF_GLU_REGLU:       return fmax(g, 0.0f) * u;
        case NF_GLU_SWIGLU_OAI: {
            const float x = fmin(g, limit);
            const float y = nf_clamp_ggml(u, -limit, limit);
            return x / (1.0f + exp(alpha * (-x))) * (y + 1.0f);
        }
        case NF_GLU_SWIGLU_CLAMP: {
            const float x = fmin(g, limit);
            const float y = nf_clamp_ggml(u, -limit, limit);
            return nf_silu(x) * y;
        }
        case NF_GLU_SWIGLU_LIMITED: {
            const float y = nf_clamp_ggml(u, -limit, limit);
            const float a = nf_clamp_ggml(nf_silu(g), -NF_INF, limit);
            return a * y;
        }
        case NF_GLU_SITU: {
            const float a = beta * tanh(g * (1.0f / beta)) * nf_sigmoid(g);
            const float b = (lambda > 0.0f) ? lambda * tanh(u * (1.0f / lambda)) : u;
            return a * b;
        }
        case NF_GLU_MUL:         return g * u;
        case NF_GLU_SIGMUL:      return nf_sigmoid(g) * u;
        default:                 return NF_NAN;
    }
}

/*
 * ALiBi slope of head h (ggml-cpu/ops.cpp:5929-5950, ggml-cuda/common.cuh:962):
 *   n_head_log2 = largest power of two <= n_head, m0 = 2^(-max_bias / n_head_log2),
 *   m1 = 2^(-(max_bias / 2) / n_head_log2), slope = h < n_head_log2 ? m0^(h+1) : m1^(2(h - n_head_log2) + 1).
 * Returns 1 when max_bias <= 0. The power of two is formed with integers, so it is exact.
 */
NF_INLINE float nf_alibi_slope(int h, int n_head, float max_bias) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    int n_head_log2 = 1;
    while ((n_head_log2 << 1) <= n_head) {
        n_head_log2 <<= 1;
    }
    const float m0 = pow(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = pow(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    const float base = h < n_head_log2 ? m0 : m1;
    const int exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;
    return pow(base, (float)exph);
}

/*
 * T5 relative-position bucket, llama_relative_position_bucket(x, y, n_buckets, bidirectional)
 * (llama-graph.cpp:4316-4343), in single precision. max_distance is 128 for every T5 variant
 * llama.cpp supports. The logarithmic branch is evaluated only for positions >= max_exact, so
 * log(0) never occurs.
 */
NF_INLINE int nf_t5_bucket(int x, int y, int n_buckets, int max_distance, int bidirectional) {
    if (bidirectional) {
        n_buckets >>= 1;
    }
    const int max_exact = n_buckets >> 1;
    int relative_position = x - y;
    int relative_bucket = 0;
    if (bidirectional) {
        relative_bucket += (relative_position > 0) ? n_buckets : 0;
        relative_position = relative_position < 0 ? -relative_position : relative_position;
    } else {
        relative_position = relative_position < 0 ? -relative_position : 0;
    }
    if (relative_position < max_exact) {
        return relative_bucket + relative_position;
    }
    const float scaled = log((float)relative_position / (float)max_exact) * (float)(n_buckets - max_exact)
                         / log((float)max_distance / (float)max_exact);
    int if_large = (int)floor((float)max_exact + scaled);
    if (if_large > n_buckets - 1) {
        if_large = n_buckets - 1;
    }
    return relative_bucket + if_large;
}

#endif /* NF_MATH_H */
