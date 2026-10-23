/* mla.cl - MLA KV assembly and absorbed-KV decompression kernels. Requires: common/dialect.h */
/*
 * Per-head K and V of one token for Multi-head Latent Attention (DeepSeek-V2 family).
 *   head_dim_k = n_nope + n_rot: each K head is [k_nope(n_nope) | k_rope(n_rot)], where k_rope is
 *   the single shared RoPE'd vector kv_pe[kv_lora .. kv_lora + n_rot) broadcast to every head.
 *   k_cur is [num_heads][head_dim_k], v_cur is [num_heads][head_dim_v].
 * Work-item g < num_heads * head_dim_k writes k_cur[g]; the next num_heads * head_dim_v work-items
 * write v_cur.
 * GLSL twin: shaders/llm/mla_kv.comp.
 */

/*
 * LLM_MLA_KV_Assemble: non-absorbed path. kv_decomp = attn_kv_b @ kv_cmpr holds
 * [k_nope(n_nope) | v(head_dim_v)] per head.
 * global: roundUp(num_heads * (head_dim_k + head_dim_v), WG), local: WG.
 */
NF_KERNEL void LLM_MLA_KV_Assemble(
    const int num_heads,
    const int n_nope,
    const int n_rot,
    const int head_dim_k,
    const int head_dim_v,
    const int kv_lora,
    const NF_GLOBAL float* NF_RESTRICT kv_pe,       /* [kv_lora + n_rot] */
    const NF_GLOBAL float* NF_RESTRICT kv_decomp,   /* [num_heads][n_nope + head_dim_v] */
    NF_GLOBAL float* NF_RESTRICT k_cur,
    NF_GLOBAL float* NF_RESTRICT v_cur)
{
    const int gid = (int)get_global_id(0);
    const int total_k = num_heads * head_dim_k;
    const int total_v = num_heads * head_dim_v;
    const int row = n_nope + head_dim_v;

    if (gid < total_k) {
        const int h = gid / head_dim_k;
        const int d = gid - h * head_dim_k;
        k_cur[gid] = (d < n_nope) ? kv_decomp[h * row + d] : kv_pe[kv_lora + (d - n_nope)];
    } else if (gid < total_k + total_v) {
        const int off = gid - total_k;
        const int h   = off / head_dim_v;
        const int d   = off - h * head_dim_v;
        v_cur[off] = kv_decomp[h * row + n_nope + d];
    }
}

/*
 * LLM_MLA_AbsorbedKVDecomp: separate attn_k_b / attn_v_b path.
 *   k_nope[h][d] = sum_l kv_cmpr[l] * k_b[h][l][d]   (k_b ggml ne = {n_nope, kv_lora, num_heads})
 *   v[h][d]      = sum_l kv_cmpr[l] * v_b[h][d][l]   (v_b ggml ne = {kv_lora, head_dim_v, num_heads})
 * kv_cmpr (the normed latent, kv_lora floats) and kv_pe may be the same buffer; both are read-only.
 * global: roundUp(num_heads * (head_dim_k + head_dim_v), WG), local: WG.
 */
NF_KERNEL void LLM_MLA_AbsorbedKVDecomp(
    const int num_heads,
    const int n_nope,
    const int n_rot,
    const int head_dim_k,
    const int head_dim_v,
    const int kv_lora,
    const NF_GLOBAL float* NF_RESTRICT kv_cmpr,     /* [kv_lora] */
    const NF_GLOBAL float* NF_RESTRICT k_b,         /* [num_heads][kv_lora][n_nope] */
    const NF_GLOBAL float* NF_RESTRICT v_b,         /* [num_heads][head_dim_v][kv_lora] */
    const NF_GLOBAL float* NF_RESTRICT kv_pe,       /* [kv_lora + n_rot] */
    NF_GLOBAL float* NF_RESTRICT k_cur,
    NF_GLOBAL float* NF_RESTRICT v_cur)
{
    const int gid = (int)get_global_id(0);
    const int total_k = num_heads * head_dim_k;
    const int total_v = num_heads * head_dim_v;

    if (gid < total_k) {
        const int h = gid / head_dim_k;
        const int d = gid - h * head_dim_k;
        if (d < n_nope) {
            const NF_GLOBAL float* kb = k_b + (nf_i64)h * n_nope * kv_lora + d;
            float s = 0.0f;
            for (int l = 0; l < kv_lora; l++) {
                s += kv_cmpr[l] * kb[(nf_i64)l * n_nope];
            }
            k_cur[gid] = s;
        } else {
            k_cur[gid] = kv_pe[kv_lora + (d - n_nope)];
        }
    } else if (gid < total_k + total_v) {
        const int off = gid - total_k;
        const int h   = off / head_dim_v;
        const int d   = off - h * head_dim_v;
        const NF_GLOBAL float* vb = v_b + ((nf_i64)h * head_dim_v + d) * kv_lora;
        float s = 0.0f;
        for (int l = 0; l < kv_lora; l++) {
            s += kv_cmpr[l] * vb[l];
        }
        v_cur[off] = s;
    }
}
