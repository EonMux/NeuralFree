/* dsv4/engram.cl - LLM_DSV4_EngramAdd: the DeepSeek-V4.1 Engram gate and residual add. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h */
/*
 * DeepSeek-V4.1 Engram n-gram embedding (models/deepseek41.cpp:88-131, 240-292, 456-520; engram.py). Java:
 * gguf.ops.dsv4.GpuEngramExtra (the hash, the gathers and the projections) and gguf.ops.dsv4.CpuEngramExtra (the CPU
 * twin: same formulas and order). The hash and the embedding lookup run elsewhere; this kernel applies the gate and
 * the residual add of the step's rows:
 *
 *   key[r]  = kv_out[t][h], the h-th key of token t                (r = t*hc + h)
 *   s       = scale * sum_i rms(x[t][h])[i] * rms(key[r])[i] * gate_w[h][i]
 *   gate    = sigmoid(copysign(sqrt(max(|s|, 1e-6)), s))
 *   x[t][h] += kv_out[t][value] * gate                             (value = the last row of the token)
 *
 * with scale = 1 / sqrt(n_embd) and gate 0 for an embedding row (active[t] == 0). The caller has already normalized
 * the residual stream into norm_x and the key rows into norm_key (LLM_RowNorm RMS); kv_out holds n_tokens rows of
 * (hc + 1) * n_embd values per token, the hc keys followed by the value (GpuEngramExtra's projection).
 *
 * Launch: 1-D, one workgroup per (token, stream) row, global = n_tokens * hc * L; local = L, a power of two in
 * [32, LLM_DIMX]. n_embd is at most LLM_DIMX * (the workgroup strides over it).
 */
NF_KERNEL void LLM_DSV4_EngramAdd(
    int n_tokens, int n_embd, int hc, float scale, int x_off,
    NF_GLOBAL const int* NF_RESTRICT active,
    NF_GLOBAL const float* NF_RESTRICT norm_x,
    NF_GLOBAL const float* NF_RESTRICT norm_key,
    NF_GLOBAL const float* NF_RESTRICT gate_w,
    NF_GLOBAL const float* NF_RESTRICT kv_out,
    NF_GLOBAL float* NF_RESTRICT x) {
    NF_LOCAL float red[LLM_DIMX];

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const nf_i64 row = (nf_i64)get_group_id(0);
    if (row >= (nf_i64)n_tokens * hc) {
        return;
    }
    const int t = (int)(row / hc);
    const int h = (int)(row - (nf_i64)t * hc);
    const nf_i64 base = row * n_embd;
    const nf_i64 gate_base = (nf_i64)h * n_embd;
    float d = 0.0f;
    for (int i = lid; i < n_embd; i += lsize) {
        d += norm_x[base + i] * norm_key[base + i] * gate_w[gate_base + i];
    }
    const float s = nf_wg_sum(red, d) * scale;
    const float m = sqrt(fmax(fabs(s), 1e-6f));
    const float g = active[t] != 0 ? nf_sigmoid(s < 0.0f ? -m : m) : 0.0f;
    const nf_i64 value = ((nf_i64)t * (hc + 1) + hc) * n_embd;
    for (int i = lid; i < n_embd; i += lsize) {
        x[x_off + base + i] += kv_out[value + i] * g;
    }
}
