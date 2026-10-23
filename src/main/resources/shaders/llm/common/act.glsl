/* common/act.glsl - unary activation and GLU helpers mirroring the NF_ACT_ and NF_GLU_ codes of kernels/llm/common/math.h. Requires: common/nf_common.glsl */
/*
 * GLSL twin of the activation section of kernels/llm/common/math.h: the same op codes (equal to
 * gguf.arch.ActOp / GluOp), the same parameter conventions and the same llama.cpp references.
 * GLSL lacks erf and expm1 and its builtin tanh may overflow to NaN for large |x| on some drivers,
 * so those three follow the llama.cpp Vulkan shaders (ggml-vulkan/vulkan-shaders/unary.comp):
 * erf by Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7), expm1 by a degree-6 Taylor series for
 * |x| <= 1/4, and tanh through exp in an overflow-safe form. An unknown op code yields NaN.
 */
#ifndef NF_ACT_GLSL
#define NF_ACT_GLSL

#ifndef NF_COMMON_GLSL
#error "common/act.glsl requires common/nf_common.glsl"
#endif

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

const float NF_GELU_COEF_A     = 0.044715;
const float NF_GELU_QUICK_COEF = -1.702;
const float NF_SQRT_2_OVER_PI  = 0.79788456080286535587989211986876;
const float NF_SQRT_2_INV      = 0.70710678118654752440084436210484;

/* tanh(x). |x| < 1/8: odd Taylor series to x^7 (truncation < 2e-9 relative), which avoids the
 * cancellation of the exp form near zero; otherwise sign(x) * (1 - 2 / (e^(2|x|) + 1)), which is
 * exactly +-1 once e^(2|x|) overflows. */
float nf_tanh(float x) {
    const float ax = abs(x);
    if (ax < 0.125) {
        const float x2 = x * x;
        return x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0))));
    }
    const float t = 1.0 - 2.0 / (exp(2.0 * ax) + 1.0);
    return x < 0.0 ? -t : t;
}

/* erf(x), Abramowitz-Stegun 7.1.26 (llama.cpp ggml-vulkan/vulkan-shaders/unary.comp op_gelu_erf). */
float nf_erf(float x) {
    const float p  = 0.3275911;
    const float a1 = 0.254829592;
    const float a2 = -0.284496736;
    const float a3 = 1.421413741;
    const float a4 = -1.453152027;
    const float a5 = 1.061405429;
    const float ax = abs(x);
    const float t = 1.0 / (1.0 + p * ax);
    const float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign(x) * y;
}

/* e^x - 1 (llama.cpp ggml-vulkan/vulkan-shaders/unary.comp op_expm1). */
float nf_expm1(float x) {
    if (abs(x) <= 0.25) {
        return x * (1.0 + x * (0.5 + x * ((1.0 / 6.0) + x * ((1.0 / 24.0) + x * ((1.0 / 120.0) + x * (1.0 / 720.0))))));
    }
    return exp(x) - 1.0;
}

/* x / (1 + e^-x)  (ggml-cpu/vec.h ggml_silu_f32) */
float nf_silu(float x) {
    return x / (1.0 + exp(-x));
}

/* 1 / (1 + e^-x) */
float nf_sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

/* x > 20 ? x : log(1 + e^x)  (ggml-impl.h:107) */
float nf_softplus(float x) {
    return (x > 20.0) ? x : log(1.0 + exp(x));
}

/* 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))  (ggml-cpu/vec.h ggml_gelu_f32), with
 * 1 + tanh(v) = 2 - 2 / (e^(2v) + 1) as in llama.cpp ggml-vulkan/vulkan-shaders/unary.comp op_gelu: no
 * cancellation, overflow-safe. */
float nf_gelu_tanh(float x) {
    const float v = NF_SQRT_2_OVER_PI * x * (1.0 + NF_GELU_COEF_A * x * x);
    return 0.5 * x * (2.0 - 2.0 / (exp(2.0 * v) + 1.0));
}

/* 0.5 x (1 + erf(x / sqrt(2))) */
float nf_gelu_erf(float x) {
    return 0.5 * x * (1.0 + nf_erf(x * NF_SQRT_2_INV));
}

/* x / (1 + e^(-1.702 x)) */
float nf_gelu_quick(float x) {
    return x * (1.0 / (1.0 + exp(NF_GELU_QUICK_COEF * x)));
}

/* xIELU with load-time prepared alpha_n' = beta + softplus(alpha_n), alpha_p' = softplus(alpha_p)
 * (ggml.c:2871-2872); evaluation as ggml-cpu/unary-ops.cpp:55-62. */
float nf_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    if (x > 0.0) {
        return alpha_p * x * x + beta * x;
    }
    const float min_x_eps = min(x, eps);
    return (nf_expm1(min_x_eps) - x) * alpha_n + beta * x;
}

/* ggml_clamp: MAX(MIN(x, hi), lo) */
float nf_clamp_ggml(float x, float lo, float hi) {
    return max(min(x, hi), lo);
}

/*
 * Unary activation dispatcher; op is an NF_ACT_* code. Parameters per op:
 *   XIELU       p0 = alpha_n', p1 = alpha_p', p2 = beta, p3 = eps
 *   SCALE_BIAS  x * p0 + p1
 *   CLAMP       p0 = lo, p1 = hi
 *   SOFTCAP     p0 * tanh(x * p1), p0 = cap, p1 = scale / cap
 * Other ops ignore p0..p3.
 */
float nf_act(int op, float x, float p0, float p1, float p2, float p3) {
    switch (op) {
        case NF_ACT_RELU:          return max(x, 0.0);
        case NF_ACT_RELU_SQR:      { const float r = max(x, 0.0); return r * r; }
        case NF_ACT_GELU_TANH:     return nf_gelu_tanh(x);
        case NF_ACT_GELU_ERF:      return nf_gelu_erf(x);
        case NF_ACT_GELU_QUICK:    return nf_gelu_quick(x);
        case NF_ACT_SILU:          return nf_silu(x);
        case NF_ACT_SIGMOID:       return nf_sigmoid(x);
        case NF_ACT_SOFTPLUS:      return nf_softplus(x);
        case NF_ACT_TANH:          return nf_tanh(x);
        case NF_ACT_XIELU:         return nf_xielu(x, p0, p1, p2, p3);
        case NF_ACT_SCALE_BIAS:    return x * p0 + p1;
        case NF_ACT_CLAMP:         return nf_clamp_ggml(x, p0, p1);
        case NF_ACT_EXP:           return exp(x);
        case NF_ACT_SQRT_SOFTPLUS: return sqrt(nf_softplus(x));
        case NF_ACT_IDENTITY:      return x;
        case NF_ACT_SOFTCAP:       return p0 * nf_tanh(x * p1);
        default:                   return NF_NAN;
    }
}

/*
 * Gated-linear-unit dispatcher out = f(g, u); op is an NF_GLU_* code. Parameters per op:
 *   SWIGLU_OAI      x = min(g, limit), y = clamp(u, -limit, limit), x * sigmoid(alpha * x) * (y + 1)
 *   SWIGLU_CLAMP    silu(min(g, limit)) * clamp(u, -limit, limit)
 *   SWIGLU_LIMITED  min(silu(g), limit) * clamp(u, -limit, limit)
 *   SITU            beta * tanh(g / beta) * sigmoid(g) * (lambda > 0 ? lambda * tanh(u / lambda) : u)
 * Other ops ignore limit, alpha, beta and lambda.
 */
float nf_glu(int op, float g, float u, float limit, float alpha, float beta, float lambda) {
    switch (op) {
        case NF_GLU_SWIGLU:      return nf_silu(g) * u;
        case NF_GLU_GEGLU_TANH:  return nf_gelu_tanh(g) * u;
        case NF_GLU_GEGLU_ERF:   return nf_gelu_erf(g) * u;
        case NF_GLU_GEGLU_QUICK: return nf_gelu_quick(g) * u;
        case NF_GLU_REGLU:       return max(g, 0.0) * u;
        case NF_GLU_SWIGLU_OAI: {
            const float x = min(g, limit);
            const float y = nf_clamp_ggml(u, -limit, limit);
            return x / (1.0 + exp(alpha * (-x))) * (y + 1.0);
        }
        case NF_GLU_SWIGLU_CLAMP: {
            const float x = min(g, limit);
            const float y = nf_clamp_ggml(u, -limit, limit);
            return nf_silu(x) * y;
        }
        case NF_GLU_SWIGLU_LIMITED: {
            const float y = nf_clamp_ggml(u, -limit, limit);
            const float a = min(nf_silu(g), limit);
            return a * y;
        }
        case NF_GLU_SITU: {
            const float a = beta * nf_tanh(g * (1.0 / beta)) * nf_sigmoid(g);
            const float b = (lambda > 0.0) ? lambda * nf_tanh(u * (1.0 / lambda)) : u;
            return a * b;
        }
        case NF_GLU_MUL:         return g * u;
        case NF_GLU_SIGMUL:      return nf_sigmoid(g) * u;
        default:                 return NF_NAN;
    }
}

#endif /* NF_ACT_GLSL */
