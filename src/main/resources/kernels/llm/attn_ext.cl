/* attn_ext.cl - LLM_AttnTempScale and LLM_AttnOutGate attention extras. Requires: common/dialect.h, common/config.h, common/math.h */
/*
 * Position-dependent query temperature (llama4 / mistral3 / deepseek2 / beam, llama-graph.cpp:213-238)
 * and the attention output gates applied before wo (afmoe, step35, laguna, beam, k2-horizon, ... ;
 * models/beam.cpp:180-189, models/step35.cpp:257-270, models/laguna.cpp:239-259, models/k2-horizon.cpp:281-294). Both
 * run in place on an offset view of a token-major activation buffer: token t starts at <x>_off + t * <x>_stride, head h
 * at h * head_dim inside the token row.
 *
 * Dispatch: one work-item per element of the [n_tokens][n_heads * head_dim] view over the flattened
 * index idx = get_global_id(0) + get_global_id(1) * get_global_size(0): 1-D global =
 * roundUp(n_tokens * n_heads * head_dim, L), local = {L}. Element counts stay below 2^31 (the host
 * validates).
 */

/* AttnGateKind codes of LLM_AttnOutGate (gguf.arch.AttnGateKind ordinal minus 2). */
#define NF_ATTN_GATE_ELEM_SIGMOID     0
#define NF_ATTN_GATE_ELEM_SOFTPLUS    1
#define NF_ATTN_GATE_HEAD_SIGMOID     2
#define NF_ATTN_GATE_HEAD_SIGMOID_B   3
#define NF_ATTN_GATE_HEAD_SOFTPLUS    4
#define NF_ATTN_GATE_SOFTPLUS_BASE2   5

/* k2-horizon's base-2 softplus scales (models/k2-horizon.cpp:283-288): ln 2 before, 1 / ln 2 after. */
#define NF_ATTN_GATE_LN2      0.6931471805599453f
#define NF_ATTN_GATE_INV_LN2  1.4426950408889634f

/* Flattened work-item index of a 1-D or 2-D range. */
NF_INLINE uint nf_attn_ext_linear_id(void) {
    return (uint)get_global_id(0) + (uint)get_global_id(1) * (uint)get_global_size(0);
}

/*
 * Temperature factor of query position p (llama-graph.cpp:213-238):
 *   pow form  exp(temp_scale * log1p((p + offset) / floor_scale))
 *   log form  log(floor((p + offset) / floor_scale) + 1) * temp_scale + 1
 * Device division may be off by a few ulp (OpenCL 2.5 ulp, CUDA -prec-div=false), so the floored
 * quotient is corrected to the exact one; products of the integral quotient and floor_scale are exact
 * for positions below 2^24, where the host's correctly rounded division floors to the same value.
 */
NF_INLINE float nf_attn_ext_temperature(int p, int pow_form, float floor_scale, float temp_scale, float offset) {
    const float x = (float)p + offset;
    if (pow_form) {
        return exp(temp_scale * log1p(x / floor_scale));
    }
    float q = floor(x / floor_scale);
    if (q * floor_scale > x) {
        q -= 1.0f;
    } else if ((q + 1.0f) * floor_scale <= x) {
        q += 1.0f;
    }
    return log(q + 1.0f) * temp_scale + 1.0f;
}

/*
 * q[q_off + t*row_stride + h*head_dim + d] *= factor(pos[t]) for every token t < n_tokens, head
 * h < n_heads and d < head_dim; pow_form selects the pow form of nf_attn_ext_temperature.
 */
NF_KERNEL void LLM_AttnTempScale(
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int row_stride,
    const int q_off,
    const int pow_form,
    const float floor_scale,
    const float temp_scale,
    const float offset,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL float* q)
{
    const uint width = (uint)n_heads * (uint)head_dim;
    const uint idx = nf_attn_ext_linear_id();
    if (width == 0u || idx >= (uint)n_tokens * width) return;
    const uint t = idx / width;
    const uint e = idx - t * width;
    const float f = nf_attn_ext_temperature(pos[t], pow_form, floor_scale, temp_scale, offset);
    const nf_i64 a = (nf_i64)q_off + (nf_i64)t * row_stride + e;
    q[a] = q[a] * f;
}

/* Gate activation of a kind: softplus (x > 20 ? x : log(1 + e^x), ggml-impl.h:107) for kinds 1 and 4,
 * softplus(g * ln 2) * (1 / ln 2) for kind 5 (models/k2-horizon.cpp:283-288), sigmoid otherwise. */
NF_INLINE float nf_attn_ext_gate_act(int kind, float g) {
    if (kind == NF_ATTN_GATE_ELEM_SOFTPLUS || kind == NF_ATTN_GATE_HEAD_SOFTPLUS) {
        return nf_softplus(g);
    }
    if (kind == NF_ATTN_GATE_SOFTPLUS_BASE2) {
        return nf_softplus(g * NF_ATTN_GATE_LN2) * NF_ATTN_GATE_INV_LN2;
    }
    return nf_sigmoid(g);
}

/*
 * attn[attn_off + t*attn_stride + h*head_dim + d] *= nf_attn_ext_gate_act(kind, g + (has_bias ? bias[h] : 0))
 * with
 *   g = gate[gate_off + t*gate_stride + h*head_dim + d]  for the element-wise kinds (0, 1, 5),
 *   g = gate[gate_off + t*gate_stride + h]               for the per-head kinds (2, 3, 4; models/laguna.cpp:247-254).
 * bias holds one value per head (models/beam.cpp:183); has_bias == 0 binds a placeholder that is never read.
 */
NF_KERNEL void LLM_AttnOutGate(
    const int n_tokens,
    const int n_head,
    const int head_dim,
    const int attn_stride,
    const int attn_off,
    const int gate_stride,
    const int gate_off,
    const int kind,
    const int has_bias,
    NF_GLOBAL const float* NF_RESTRICT gate,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* attn)
{
    const uint width = (uint)n_head * (uint)head_dim;
    const uint idx = nf_attn_ext_linear_id();
    if (width == 0u || idx >= (uint)n_tokens * width) return;
    const uint t = idx / width;
    const uint e = idx - t * width;
    const uint h = e / (uint)head_dim;
    const int per_head = kind == NF_ATTN_GATE_HEAD_SIGMOID || kind == NF_ATTN_GATE_HEAD_SIGMOID_B
        || kind == NF_ATTN_GATE_HEAD_SOFTPLUS;
    float g = gate[(nf_i64)gate_off + (nf_i64)t * gate_stride + (per_head ? h : e)];
    if (has_bias) {
        g += bias[h];
    }
    const float s = nf_attn_ext_gate_act(kind, g);
    const nf_i64 a = (nf_i64)attn_off + (nf_i64)t * attn_stride + e;
    attn[a] = attn[a] * s;
}
