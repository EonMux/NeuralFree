/* attn_ext.cl - LLM_AttnTempScale, LLM_AttnSsmaxScale, LLM_AttnOutGate, LLM_InklingQueryScale,
 * LLM_InklingRelLogits, LLM_LoopAttnMix, LLM_AttnValueEmbd and LLM_AttnXsa attention extras. Requires:
 * common/dialect.h, common/config.h, common/math.h */
/*
 * Position-dependent query temperature (llama4 / mistral3 / deepseek2 / beam, llama-graph.cpp:213-238), the per-head
 * SSMax query scale of mistral-large4 (models/mistral-large4.cpp:181-190, 381, 240), the attention output gates
 * applied before wo (afmoe, step35, laguna, beam, k2-horizon, ... ; models/beam.cpp:180-189, models/step35.cpp:257-270,
 * models/laguna.cpp:239-259, models/k2-horizon.cpp:281-294). The Inkling pair scales the normed queries of a
 * full-attention layer by the token's log factor and builds that step's relative-logit table from the attn_rel
 * output and its shared projection (vLLM compute_log_scaling_tau / qkvr_rel_proj); limite's value embeddings and
 * exclusive self attention gate and add the token's value_embd row and remove each head's projection on the token's
 * own value (models/limite.cpp:87-89, 149-155, 177-191). The in-place kernels run on an offset view of a token-major
 * activation buffer: token t starts at <x>_off + t * <x>_stride, head h at h * head_dim inside the token row.
 *
 * Dispatch: one work-item per element of the [n_tokens][n_heads * head_dim] view over the flattened
 * index idx = get_global_id(0) + get_global_id(1) * get_global_size(0): 1-D global =
 * roundUp(n_tokens * n_heads * head_dim, L), local = {L}. Element counts stay below 2^31 (the host
 * validates). LLM_AttnXsa launches one work-item per (token, KV head) pair instead.
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

/*
 * x[x_off + t*row_stride + h*head_stride + d] *= scale[h] * log(pos[t] + 1) + bias[h] for every token t < n_tokens
 * and head h < n_heads: the SSMax per-head query temperature of mistral-large4 (models/mistral-large4.cpp:181-190,
 * 381, 240). scale and bias hold one value per head; pos one query position per token.
 */
NF_KERNEL void LLM_AttnSsmaxScale(
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int head_stride,
    const int row_stride,
    const int x_off,
    NF_GLOBAL const float* NF_RESTRICT scale,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL float* x)
{
    const uint width = (uint)n_heads * (uint)head_dim;
    const uint idx = nf_attn_ext_linear_id();
    if (width == 0u || idx >= (uint)n_tokens * width) return;
    const uint t = idx / width;
    const uint e = idx - t * width;
    const uint h = e / (uint)head_dim;
    const uint d = e - h * (uint)head_dim;
    const float f = scale[h] * log((float)pos[t] + 1.0f) + bias[h];
    const nf_i64 a = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride + d;
    x[a] = x[a] * f;
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

/*
 * Inkling's per-token log factor of a full-attention layer (vLLM compute_log_scaling_tau, attention.py:47-53):
 *   tau(p) = 1 + alpha * log(max(1, (p + 1) / floor)),   tau == 1 when floor <= 0.
 * q[q_off + t*row_stride + h*head_dim + d] *= tau(pos[t]) for every token t < n_tokens, head h < n_heads and
 * d < head_dim (vLLM fused_qkvr_prep scales the normed queries of a full layer; a local layer keeps them).
 */
NF_INLINE float nf_attn_ext_inkling_tau(int p, int floor, float alpha) {
    if (floor <= 0) {
        return 1.0f;
    }
    const float x = (float)(p + 1) / (float)floor;
    return 1.0f + alpha * log(x < 1.0f ? 1.0f : x);
}

NF_KERNEL void LLM_InklingQueryScale(
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int row_stride,
    const int q_off,
    const int log_floor,
    const float log_alpha,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL float* q)
{
    const uint width = (uint)n_heads * (uint)head_dim;
    const uint idx = nf_attn_ext_linear_id();
    if (width == 0u || idx >= (uint)n_tokens * width) return;
    const uint t = idx / width;
    const uint e = idx - t * width;
    const float tau = nf_attn_ext_inkling_tau(pos[t], log_floor, log_alpha);
    const nf_i64 a = (nf_i64)q_off + (nf_i64)t * row_stride + e;
    q[a] = q[a] * tau;
}

/*
 * Inkling's query-side relative-logit table (vLLM qkvr_rel_proj, ops/qkvr_prep.py): with the shared
 * [d_rel][rel_extent] projection and the per-head [t][h][d_rel] attn_rel output,
 *   out[(t*n_heads + h)*rel_extent + e] = tau(pos[t]) * sum_j r[t][h][j] * proj[j*rel_extent + e].
 * tau is the layer's log factor on a full layer, 1 on a local layer (log_floor <= 0).
 */
NF_KERNEL void LLM_InklingRelLogits(
    const int n_tokens,
    const int n_heads,
    const int d_rel,
    const int rel_extent,
    const int log_floor,
    const float log_alpha,
    NF_GLOBAL const float* NF_RESTRICT r,
    NF_GLOBAL const float* NF_RESTRICT proj,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const uint extent = (uint)rel_extent;
    const uint idx = nf_attn_ext_linear_id();
    if (extent == 0u || idx >= (uint)n_tokens * (uint)n_heads * extent) return;
    const uint t = idx / ((uint)n_heads * extent);
    const uint rest = idx - t * ((uint)n_heads * extent);
    const uint h = rest / extent;
    const uint e = rest - h * extent;
    const nf_i64 rBase = ((nf_i64)t * n_heads + h) * d_rel;
    float acc = 0.0f;
    for (int j = 0; j < d_rel; j++) {
        acc += r[rBase + j] * proj[(nf_i64)j * rel_extent + e];
    }
    const float tau = nf_attn_ext_inkling_tau(pos[t], log_floor, log_alpha);
    out[((nf_i64)t * n_heads + h) * rel_extent + e] = acc * tau;
}

/*
 * IQuestLoopCoder's gated mixed attention (models/iquest_loopcoder.py, LoopCoderAttention.forward for
 * loop_idx > 0): each head of a token gates a read-only global attention against the local sliding-window one:
 *   g = sigmoid(dot(q[t][h][:], gate_w[h][:]) + gate_b[h])
 *   local[attn_off + t*attn_stride + h*head_dim + d] =
 *       local[...] * (1 - g) + global[...] * g
 * gate_w is [n_heads][head_dim], gate_b one value per head; q and both attention rows are token-major views
 * (offsets and strides in floats). The local rows are the in-place operand.
 */
NF_KERNEL void LLM_LoopAttnMix(
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int q_stride,
    const int q_off,
    const int attn_stride,
    const int attn_off,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const float* NF_RESTRICT gate_w,
    NF_GLOBAL const float* NF_RESTRICT gate_b,
    NF_GLOBAL const float* NF_RESTRICT global_rows,
    NF_GLOBAL float* local_rows)
{
    const uint width = (uint)n_heads * (uint)head_dim;
    const uint idx = nf_attn_ext_linear_id();
    if (width == 0u || idx >= (uint)n_tokens * width) return;
    const uint t = idx / width;
    const uint e = idx - t * width;
    const uint h = e / (uint)head_dim;
    const uint d = e - h * (uint)head_dim;
    const nf_i64 qb = (nf_i64)q_off + (nf_i64)t * q_stride + (nf_i64)h * head_dim;
    float sum = gate_b[h];
    for (uint i = 0; i < (uint)head_dim; ++i) {
        sum += q[qb + (nf_i64)i] * gate_w[(nf_i64)h * head_dim + (nf_i64)i];
    }
    const float g = 1.0f / (1.0f + exp(-sum));
    const nf_i64 a = (nf_i64)attn_off + (nf_i64)t * attn_stride + (nf_i64)e;
    local_rows[a] = local_rows[a] * (1.0f - g) + global_rows[a] * g;
}

/*
 * limite value embeddings (models/limite.cpp:87-89, 149-155): the token's value_embd row, gated per KV head by
 * sigmoid(attn_ve_gate @ x) when has_gate, adds to v:
 *   v[t][h][d] += ve[t][h][d] * (has_gate ? sigmoid(gate[t][h]) : 1),  h < n_head_kv, d < head_dim.
 * ve and v are dense [n_tokens][n_head_kv * head_dim] rows; gate is [n_tokens][n_head_kv].
 */
NF_KERNEL void LLM_AttnValueEmbd(
    const int n_tokens,
    const int v_dim,
    const int n_head_kv,
    const int head_dim,
    const int has_gate,
    NF_GLOBAL const float* NF_RESTRICT ve,
    NF_GLOBAL const float* NF_RESTRICT gate,
    NF_GLOBAL float* v)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (v_dim <= 0 || idx >= (nf_i64)n_tokens * v_dim) return;
    const nf_i64 t = idx / v_dim;
    const nf_i64 e = idx - t * v_dim;
    float g = 1.0f;
    if (has_gate) {
        g = nf_sigmoid(gate[t * n_head_kv + e / head_dim]);
    }
    v[idx] += ve[idx] * g;
}

/*
 * limite exclusive self attention (models/limite.cpp:177-191): each query head loses its projection on the token's
 * own L2-normalized value, scaled by the head's attn_xsa_alpha:
 *   u = v[t][h] / max(||v[t][h]||, eps);  attn[t][h][g] -= u * alpha[h * n_group + g] * (attn[t][h][g] . u).
 * One work-item per (token, KV head); the group loop covers the n_group query heads of that KV head.
 */
NF_KERNEL void LLM_AttnXsa(
    const int n_tokens,
    const int n_head_kv,
    const int n_group,
    const int head_dim,
    const int v_dim,
    const float eps,
    NF_GLOBAL const float* NF_RESTRICT alpha,
    NF_GLOBAL const float* NF_RESTRICT v,
    NF_GLOBAL float* attn)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (n_head_kv <= 0 || head_dim <= 0 || idx >= (nf_i64)n_tokens * n_head_kv) return;
    const nf_i64 t = idx / n_head_kv;
    const int kvh = (int)(idx - t * n_head_kv);
    const nf_i64 v_base = t * v_dim + (nf_i64)kvh * head_dim;
    const nf_i64 out_stride = (nf_i64)n_head_kv * n_group * head_dim;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        const float u = v[v_base + d];
        sum += u * u;
    }
    const float norm = max(sqrt(sum), eps);
    for (int g = 0; g < n_group; g++) {
        const int head = kvh * n_group + g;
        const nf_i64 a_base = t * out_stride + (nf_i64)head * head_dim;
        float proj = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            proj += attn[a_base + d] * (v[v_base + d] / norm);
        }
        proj *= alpha[head];
        for (int d = 0; d < head_dim; d++) {
            attn[a_base + d] -= (v[v_base + d] / norm) * proj;
        }
    }
}
