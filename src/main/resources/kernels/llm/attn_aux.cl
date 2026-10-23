/* attn_aux.cl - LLM_SplitQGate and LLM_Batch_SplitQGate. Requires: common/dialect.h */
/*
 * Split of a fused Q+gate projection (attention output gate, e.g. Qwen3.5/Qwen3-Next) into a
 * tight-packed Q buffer and a gate buffer of num_heads * head_dim floats per token.
 * Source layout per token, selected by the runtime flag `interleaved` (uniform branch):
 *   interleaved != 0  [Q_h0(head_dim) | gate_h0(head_dim) | Q_h1 | gate_h1 | ...]
 *                     (llama.cpp qwen35: ggml_view_3d with row stride 2 * head_dim)
 *   interleaved == 0  [Q of all heads (num_heads*head_dim) | gate of all heads]
 * combined never aliases q_out or gate_out.
 * GLSL twin: shaders/llm/attn_aux.comp.
 */

/*
 * LLM_SplitQGate: one token.
 * global: roundUp(num_heads * head_dim, WG), local: WG.
 */
NF_KERNEL void LLM_SplitQGate(
    const int num_heads,
    const int head_dim,
    const int interleaved,
    const NF_GLOBAL float* NF_RESTRICT combined,    /* [2 * num_heads * head_dim] */
    NF_GLOBAL float* NF_RESTRICT q_out,             /* [num_heads * head_dim] */
    NF_GLOBAL float* NF_RESTRICT gate_out)          /* [num_heads * head_dim] */
{
    const int gid = (int)get_global_id(0);
    const int total = num_heads * head_dim;
    if (gid >= total) return;

    if (interleaved) {
        const int head = gid / head_dim;
        const int dim  = gid - head * head_dim;
        const int src  = head * (head_dim * 2) + dim;
        q_out[gid]    = combined[src];
        gate_out[gid] = combined[src + head_dim];
    } else {
        q_out[gid]    = combined[gid];
        gate_out[gid] = combined[gid + total];
    }
}

/*
 * LLM_Batch_SplitQGate: seq_len tokens; token s reads combined[s * 2 * num_heads * head_dim ..] and
 * writes q_out / gate_out[s * num_heads * head_dim ..].
 * global: roundUp(seq_len * num_heads * head_dim, WG), local: WG.
 */
NF_KERNEL void LLM_Batch_SplitQGate(
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int interleaved,
    const NF_GLOBAL float* NF_RESTRICT combined,    /* [seq_len][2 * num_heads * head_dim] */
    NF_GLOBAL float* NF_RESTRICT q_out,             /* [seq_len][num_heads * head_dim] */
    NF_GLOBAL float* NF_RESTRICT gate_out)          /* [seq_len][num_heads * head_dim] */
{
    const int gid = (int)get_global_id(0);
    const int total_per_token = num_heads * head_dim;
    const int total = seq_len * total_per_token;
    if (gid >= total) return;

    const int token  = gid / total_per_token;
    const int within = gid - token * total_per_token;
    const int base   = token * (total_per_token * 2);

    if (interleaved) {
        const int head = within / head_dim;
        const int dim  = within - head * head_dim;
        const int src  = base + head * (head_dim * 2) + dim;
        q_out[gid]    = combined[src];
        gate_out[gid] = combined[src + head_dim];
    } else {
        q_out[gid]    = combined[base + within];
        gate_out[gid] = combined[base + total_per_token + within];
    }
}
