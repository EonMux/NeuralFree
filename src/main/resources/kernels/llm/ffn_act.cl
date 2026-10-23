/* ffn_act.cl - FFN activation kernels. Requires: common/dialect.h, common/config.h, common/math.h */
/*
 * Element-wise activations and gated products in precise math (common/math.h, which cites the
 * llama.cpp reference of each function). One work-item per element (or per four elements for the
 * float4 batch kernels); the only early returns are tail bounds checks.
 *
 * Aliasing: callers run these kernels in place (out == gate, output == input, ...), so no buffer
 * is restrict; every work-item reads and writes only its own elements, which makes in-place use
 * exact.
 *
 * Dispatch: 1-D, global = roundUp(n, L) work-items with n = the element count, or
 * roundUp(ceil(n / 4), L) for the float4 batch kernels; local = {L}.
 */

/* gelu (tanh approximation) of four lanes. */
NF_INLINE float4 nf_ffn_act_gelu4(float4 x) {
    return NF_F4(nf_gelu_tanh(x.x), nf_gelu_tanh(x.y), nf_gelu_tanh(x.z), nf_gelu_tanh(x.w));
}

/* silu of four lanes. */
NF_INLINE float4 nf_ffn_act_silu4(float4 x) {
    return NF_F4(nf_silu(x.x), nf_silu(x.y), nf_silu(x.z), nf_silu(x.w));
}

/* out[i] = silu(gate[i]) * up[i], i in [0, size) (SwiGLU, ggml_swiglu). */
NF_KERNEL void LLM_SiGLUMul(
    const int size,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* up,
    NF_GLOBAL float* out)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    out[i] = nf_silu(gate[i]) * up[i];
}

/* output[i] = silu(input[i]), i in [0, size). */
NF_KERNEL void LLM_SiLU(
    const int size,
    NF_GLOBAL const float* input,
    NF_GLOBAL float* output)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    output[i] = nf_silu(input[i]);
}

/* out[i] = silu(gate[i]) * up[i] over a [seq_len x size] matrix, four elements per work-item. */
NF_KERNEL void LLM_Batch_SiGLUMul(
    const int size,
    const int seq_len,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* up,
    NF_GLOBAL float* out)
{
    const int total = seq_len * size;
    const int i = get_global_id(0) * 4;
    if (i >= total) return;
    if (i + 3 < total) {
        vstore4(nf_ffn_act_silu4(vload4(0, gate + i)) * vload4(0, up + i), 0, out + i);
    } else {
        for (int j = i; j < total; j++) {
            out[j] = nf_silu(gate[j]) * up[j];
        }
    }
}

/*
 * output[i] = swiglu_oai(gate[i], up[i]) with alpha = 1.702 and limit = 7 (gpt-oss):
 * x = min(g, limit), y = clamp(u, -limit, limit), x * sigmoid(alpha * x) * (y + 1)
 * (ggml-cpu/ops.cpp:3546-3549).
 */
NF_KERNEL void LLM_SwiGLUOai(
    const int size,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* up,
    NF_GLOBAL float* output)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    output[i] = nf_glu(NF_GLU_SWIGLU_OAI, gate[i], up[i], 7.0f, 1.702f, 0.0f, 0.0f);
}

/* output[i] = gelu(input[i]) (tanh approximation), i in [0, size). */
NF_KERNEL void LLM_GELU(
    const int size,
    NF_GLOBAL const float* input,
    NF_GLOBAL float* output)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    output[i] = nf_gelu_tanh(input[i]);
}

/* out[i] = gelu(gate[i]) * up[i], i in [0, size) (GeGLU, tanh approximation). */
NF_KERNEL void LLM_GELUMul(
    const int size,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* up,
    NF_GLOBAL float* out)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    out[i] = nf_gelu_tanh(gate[i]) * up[i];
}

/* output[i] = gelu(input[i]), i in [0, total), four elements per work-item. */
NF_KERNEL void LLM_Batch_GELU(
    const int total,
    NF_GLOBAL const float* input,
    NF_GLOBAL float* output)
{
    const int i = get_global_id(0) * 4;
    if (i >= total) return;
    if (i + 3 < total) {
        vstore4(nf_ffn_act_gelu4(vload4(0, input + i)), 0, output + i);
    } else {
        for (int j = i; j < total; j++) {
            output[j] = nf_gelu_tanh(input[j]);
        }
    }
}

/* dst[i] = gelu(gate[i]) * src[offset + i], i in [0, size) (per-layer embedding gate, gemma3n.cpp ggml_gelu). */
NF_KERNEL void LLM_GeluMulOffset(
    const int size,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* src,
    const int offset,
    NF_GLOBAL float* dst)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    dst[i] = nf_gelu_tanh(gate[i]) * src[(nf_i64)offset + i];
}

/*
 * dst[s * n_embd + i] = gelu(gate[s * n_embd + i]) * inp_per_layer[layer_off + s * tok_stride + i]
 * over seq_len tokens: the per-layer input slice of one layer, read from a [seq_len][n_layer][n_embd]
 * table (layer_off = layer * n_embd, tok_stride = n_layer * n_embd).
 */
NF_KERNEL void LLM_Batch_GeluMulLayerToken(
    const int n_embd,
    const int seq_len,
    const int layer_off,
    const int tok_stride,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* inp_per_layer,
    NF_GLOBAL float* dst)
{
    const int j = get_global_id(0);
    if (j >= n_embd * seq_len) return;
    const int s = j / n_embd;
    const int i = j - s * n_embd;
    dst[j] = nf_gelu_tanh(gate[j]) * inp_per_layer[(nf_i64)layer_off + (nf_i64)s * tok_stride + i];
}

/* out[i] = gelu(gate[i]) * up[i] over a [seq_len x size] matrix, four elements per work-item. */
NF_KERNEL void LLM_Batch_GELUMul(
    const int size,
    const int seq_len,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* up,
    NF_GLOBAL float* out)
{
    const int total = seq_len * size;
    const int i = get_global_id(0) * 4;
    if (i >= total) return;
    if (i + 3 < total) {
        vstore4(nf_ffn_act_gelu4(vload4(0, gate + i)) * vload4(0, up + i), 0, out + i);
    } else {
        for (int j = i; j < total; j++) {
            out[j] = nf_gelu_tanh(gate[j]) * up[j];
        }
    }
}

/* output[i] = sigmoid(gate[i]) * input[i], i in [0, n) (attention output gate). */
NF_KERNEL void LLM_SigmoidGateMul(
    const int n,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* input,
    NF_GLOBAL float* output)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    output[i] = nf_sigmoid(gate[i]) * input[i];
}

/* Prefill form of LLM_SigmoidGateMul over n = seq_len * width elements; same arguments and math. */
NF_KERNEL void LLM_Batch_SigmoidGateMul(
    const int n,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* input,
    NF_GLOBAL float* output)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    output[i] = nf_sigmoid(gate[i]) * input[i];
}
