/* ffn_glu.cl - LLM_Unary and LLM_GLU activation kernels. Requires: common/dialect.h, common/config.h, common/math.h */
/*
 * Generic element-wise activation (nf_act) and gated-linear-unit (nf_glu) kernels over offset views,
 * with the op chosen at run time by its gguf.arch.ActOp / GluOp code. The math, its precision and its
 * llama.cpp references live in common/math.h; an unknown op code yields NaN.
 *
 * Aliasing: no buffer is restrict. Each work-item reads only the input element(s) of its own output
 * element, then writes that output element, so in-place use is exact when every output element either
 * coincides with the input element(s) it is computed from or does not overlap any input element
 * (for example out == in with out_off == in_off, or out == gate with equal offsets and strides).
 *
 * Dispatch: one work-item per output element over the flattened element index
 *     idx = get_global_id(0) + get_global_id(1) * get_global_size(0),
 * so a 1-D range (global = roundUp(n_elements, L), local = {L}) and a 2-D range
 * (global = {roundUp(n_cols, L), n_rows}, local = {L, 1}) are both valid. Element counts stay below
 * 2^31 (the host validates).
 */

/* Flattened work-item index of a 1-D or 2-D range. */
NF_INLINE uint nf_act_linear_id(void) {
    return (uint)get_global_id(0) + (uint)get_global_id(1) * (uint)get_global_size(0);
}

/*
 * out[out_off + i] = nf_act(op, in[in_off + i], p0, p1, p2, p3), i in [0, n).
 * op is an ActOp code; p0..p3 follow nf_act (XIELU prepared parameters, SCALE_BIAS, CLAMP, SOFTCAP).
 */
NF_KERNEL void LLM_Unary(
    const int n,
    const int op,
    const float p0,
    const float p1,
    const float p2,
    const float p3,
    const int in_off,
    const int out_off,
    NF_GLOBAL const float* in,
    NF_GLOBAL float* out)
{
    const uint i = nf_act_linear_id();
    if (i >= (uint)n) return;
    const float x = in[(nf_i64)in_off + i];
    out[(nf_i64)out_off + i] = nf_act(op, x, p0, p1, p2, p3);
}

/*
 * out[out_off + r*out_stride + c] = nf_glu(op, gate[gate_off + r*gate_stride + c],
 *                                          up[up_off + r*up_stride + c]),  r in [0, n_rows), c in [0, n_cols).
 * op is a GluOp code. limit clamps SWIGLU_OAI / SWIGLU_CLAMP / SWIGLU_LIMITED; alpha is the SWIGLU_OAI
 * sigmoid slope or, for SITU, situ_beta; beta is the SITU situ_linear_beta (0 leaves up linear). The
 * fused gate+up layout (one row of width 2*n_ff) passes the same buffer for gate and up with
 * up_off = gate_off + n_ff and both strides 2*n_ff.
 */
NF_KERNEL void LLM_GLU(
    const int n_cols,
    const int n_rows,
    const int op,
    const float limit,
    const float alpha,
    const float beta,
    const int gate_stride,
    const int gate_off,
    const int up_stride,
    const int up_off,
    const int out_stride,
    const int out_off,
    NF_GLOBAL const float* gate,
    NF_GLOBAL const float* up,
    NF_GLOBAL float* out)
{
    const uint idx = nf_act_linear_id();
    if (n_cols <= 0 || idx >= (uint)n_rows * (uint)n_cols) return;
    const uint r = idx / (uint)n_cols;
    const uint c = idx - r * (uint)n_cols;
    const float g = gate[(nf_i64)gate_off + (nf_i64)r * gate_stride + c];
    const float u = up[(nf_i64)up_off + (nf_i64)r * up_stride + c];
    /* SITU reads situ_beta from nf_glu's beta and situ_linear_beta from its lambda; SWIGLU_OAI reads alpha. */
    out[(nf_i64)out_off + (nf_i64)r * out_stride + c] = nf_glu(op, g, u, limit, alpha, alpha, beta);
}
