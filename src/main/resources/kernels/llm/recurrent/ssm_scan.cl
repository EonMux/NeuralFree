/* recurrent/ssm_scan.cl - LLM_SSM_Scan Mamba1/Mamba2 selective scan and LLM_MambaGateOut. Requires: common/dialect.h, common/config.h, common/reduce.h, common/math.h */
/*
 * LLM_SSM_Scan: ggml_ssm_scan (ggml-cpu/ops.cpp:10092-10308, shapes ggml.c:5711-5764) with the dt bias of the
 * graph folded in (models/mamba-base.cpp:109, 251; plamo2.cpp:362). Per token t, channel
 * ch = h * head_dim + p (h < n_head, p < head_dim), state n < d_state and group g = h / (n_head / n_group):
 *   dt'  = softplus(dt[dt_off + t*dt_stride + h] + (has_dt_bias ? dt_bias[h] : 0))
 *   x_dt = x[x_off + t*x_stride + ch] * dt'
 *   dA   = exp(dt' * A[h*d_state + n])   (a_per_state = 1, Mamba1: A {d_state, n_head})
 *        = exp(dt' * A[h])               (a_per_state = 0, Mamba2 / PLaMo2: A {1, n_head})
 *   s[ch][n] = s[ch][n] * dA + B[b_off + t*b_stride + g*d_state + n] * x_dt
 *   y[t*y_stride + ch] = sum_n s[ch][n] * C[c_off + t*c_stride + g*d_state + n]
 * The state has the llama.cpp layout s[(h*head_dim + p)*d_state + n] in row state_rows[0] of
 * n_head*head_dim*d_state floats; it is read once, kept in registers across the token loop and written
 * back once. The D skip and the z gate are LLM_MambaGateOut. Mamba1 is n_head = d_inner, head_dim = 1,
 * n_group = 1; PLaMo2 passes n_group = 1. Decode and prefill use the same kernel.
 *
 * Work decomposition: a workgroup holds ch_per_wg channels of lanes = L / ch_per_wg
 * consecutive work-items each (lanes a power of two); lane l of a channel owns the states n = l + k*lanes,
 * k < ceil(d_state / lanes) <= NF_SSM_SCAN_MAX_SPL. The host uses lanes = min(pow2ceil(d_state), 64) and
 * ch_per_wg = max(1, 32 / lanes), so L = ch_per_wg * lanes is a power of two >= 32 (a multiple of 32 on
 * CUDA). The per-token sum over the states is a workgroup reduction when ch_per_wg == 1 (subgroup
 * arithmetic when available) and an aligned segment reduction otherwise (register butterfly on CUDA,
 * local-memory tree elsewhere). Work-items past the last channel keep their lane in every reduction.
 * Dispatch: 1-D, local = L, global = ceil(n_head*head_dim / ch_per_wg) * L.
 *
 * Aliasing: x, B and C may be views of one buffer (the conv output xBC); all three are only read.
 */

#define NF_SSM_SCAN_MAX_SPL 8   /* states per lane: d_state <= 8 * lanes */

NF_KERNEL void LLM_SSM_Scan(
        const int d_state,
        const int head_dim,
        const int n_head,
        const int n_group,
        const int n_tokens,
        const int a_per_state,
        const int ch_per_wg,
        const int x_stride,
        const int x_off,
        const int dt_stride,
        const int dt_off,
        const int b_stride,
        const int c_stride,
        const int b_off,
        const int c_off,
        const int y_stride,
        const int has_dt_bias,
        NF_GLOBAL const float* NF_RESTRICT x,
        NF_GLOBAL const float* NF_RESTRICT dt,
        NF_GLOBAL const float* NF_RESTRICT dt_bias,
        NF_GLOBAL const float* NF_RESTRICT A,
        NF_GLOBAL const float* NF_RESTRICT B,
        NF_GLOBAL const float* NF_RESTRICT C,
        NF_GLOBAL float* ssm_state,
        NF_GLOBAL const int* NF_RESTRICT state_rows,
        NF_GLOBAL float* NF_RESTRICT y) {
    NF_LOCAL float red[LLM_DIMX];

    const int lid = (int)get_local_id(0);
    const int lanes = (int)get_local_size(0) / ch_per_wg;
    const int lane = lid & (lanes - 1);
    const int n_ch = n_head * head_dim;
    const int ch = (int)get_group_id(0) * ch_per_wg + lid / lanes;
    const int active = ch < n_ch;
    const int h = active ? ch / head_dim : 0;
    const int g = h / (n_head / n_group);
    const int spl = (d_state + lanes - 1) / lanes;
    const nf_i64 s_base = (nf_i64)state_rows[0] * n_ch * d_state + (nf_i64)ch * d_state;
    const float bias = (active && has_dt_bias) ? dt_bias[h] : 0.0f;
    const float a_head = (active && !a_per_state) ? A[h] : 0.0f;

    float s[NF_SSM_SCAN_MAX_SPL];
    float a[NF_SSM_SCAN_MAX_SPL];
#pragma unroll
    for (int k = 0; k < NF_SSM_SCAN_MAX_SPL; ++k) {
        const int n = lane + k * lanes;
        const int own = active && k < spl && n < d_state;
        s[k] = own ? ssm_state[s_base + n] : 0.0f;
        a[k] = (own && a_per_state) ? A[(nf_i64)h * d_state + n] : 0.0f;
    }

    for (int t = 0; t < n_tokens; ++t) {
        float partial = 0.0f;
        if (active) {
            const float dt_sp = nf_softplus(dt[(nf_i64)dt_off + (nf_i64)t * dt_stride + h] + bias);
            const float x_dt = x[(nf_i64)x_off + (nf_i64)t * x_stride + ch] * dt_sp;
            const float dA_head = a_per_state ? 0.0f : exp(dt_sp * a_head);
            const nf_i64 bt = (nf_i64)b_off + (nf_i64)t * b_stride + (nf_i64)g * d_state;
            const nf_i64 ct = (nf_i64)c_off + (nf_i64)t * c_stride + (nf_i64)g * d_state;
#pragma unroll
            for (int k = 0; k < NF_SSM_SCAN_MAX_SPL; ++k) {
                const int n = lane + k * lanes;
                if (k < spl && n < d_state) {
                    const float dA = a_per_state ? exp(dt_sp * a[k]) : dA_head;
                    s[k] = s[k] * dA + B[bt + n] * x_dt;
                    partial += s[k] * C[ct + n];
                }
            }
        }
        const float sum = ch_per_wg == 1 ? nf_wg_sum(red, partial) : nf_seg_sum(red, partial, lanes);
        if (active && lane == 0) {
            y[(nf_i64)t * y_stride + ch] = sum;
        }
    }

#pragma unroll
    for (int k = 0; k < NF_SSM_SCAN_MAX_SPL; ++k) {
        const int n = lane + k * lanes;
        if (active && k < spl && n < d_state) {
            ssm_state[s_base + n] = s[k];
        }
    }
}

/*
 * LLM_MambaGateOut: the Mamba output gate of models/mamba-base.cpp:140-141 / 286-288 and plamo2.cpp:406-410,
 *   out[t][c] = (y[t][c] + D[c / head_dim] * x[t][c]) * silu(z[t][c])
 * over n = n_tokens * width elements, width = d_inner. y is contiguous (y[y_off + t*width + c]), x is the
 * conv output view x[x_off + t*x_stride + c], z the in_proj view z[z_off + t*z_stride + c], and out is
 * contiguous [t][width]. D is per channel for Mamba1 (head_dim = 1) and per head for Mamba2/PLaMo2.
 * Aliasing: out is distinct from y, x and z (all four are restrict-qualified).
 * Dispatch: 1-D, global = roundUp(n, L).
 */
NF_KERNEL void LLM_MambaGateOut(
        const int n,
        const int width,
        const int head_dim,
        const int x_stride,
        const int x_off,
        const int y_off,
        const int z_stride,
        const int z_off,
        NF_GLOBAL const float* NF_RESTRICT y,
        NF_GLOBAL const float* NF_RESTRICT x,
        NF_GLOBAL const float* NF_RESTRICT D,
        NF_GLOBAL const float* NF_RESTRICT z,
        NF_GLOBAL float* NF_RESTRICT out) {
    const int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    const int t = i / width;
    const int c = i - t * width;
    const float v = y[(nf_i64)y_off + i] + D[c / head_dim] * x[(nf_i64)x_off + (nf_i64)t * x_stride + c];
    out[i] = v * nf_silu(z[(nf_i64)z_off + (nf_i64)t * z_stride + c]);
}
