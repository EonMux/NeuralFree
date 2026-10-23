/* common/dsv4_common.h - DeepSeek-V4 constants and the partial-RoPE row rotation shared by dsv4/compress.cl and dsv4/attn.cl. Requires: common/dialect.h, common/config.h, common/rope_common.h, common/kv_types.h */
/*
 * Java mirror: gguf.ops.dsv4.Dsv4Kernels (MAX_WIDTH, KV_*, ROPE_* constants).
 */
#ifndef NF_DSV4_COMMON_H
#define NF_DSV4_COMMON_H

#define NF_DSV4_CSA_RATIO 4
#define NF_DSV4_HCA_RATIO 128
#define NF_DSV4_MAX_WIDTH 1024     /* largest compressed row / head dim held in local memory (floats) */

/* Cache element types: gguf.memory.KvDType (Dsv4Kernels.kvTypeCode), the NF_KV_* codes of common/kv_types.h. */
#define NF_DSV4_KV_F32    NF_KV_F32
#define NF_DSV4_KV_F16    NF_KV_F16
#define NF_DSV4_KV_Q8_0   NF_KV_Q8_0
#define NF_DSV4_KV_BF16   NF_KV_BF16
#define NF_DSV4_KV_Q4_0   NF_KV_Q4_0
#define NF_DSV4_KV_Q4_1   NF_KV_Q4_1
#define NF_DSV4_KV_IQ4_NL NF_KV_IQ4_NL
#define NF_DSV4_KV_Q5_0   NF_KV_Q5_0
#define NF_DSV4_KV_Q5_1   NF_KV_Q5_1

/* RoPE modes: ggml GGML_ROPE_TYPE_NORMAL / GGML_ROPE_TYPE_NEOX (gguf.arch.RopeType.ggmlMode()). */
#define NF_DSV4_ROPE_NORM NF_ROPE_NORM
#define NF_DSV4_ROPE_NEOX NF_ROPE_NEOX

/*
 * Rotates the last n_rot dims of n_rows local rows of `width` values (row k at rows + k*width) at position p, as
 * ggml rotate_pairs (llama.cpp ggml-cpu/ops.cpp:6375-6391) with n_offs = width - n_rot: NORM pairs (n_offs + 2i,
 * n_offs + 2i + 1), NEOX pairs (n_offs + i, n_offs + i + n_rot/2), i < n_rot/2. Pair i turns by the rope_yarn angle
 * p * theta_scale^i (nf_rope_cs with exponent i and no frequency factor). sin_sign = -1 is the inverse rotation of
 * ggml_rope_ext_back (ops.cpp:6463-6466). No barrier inside: the caller synchronizes before and after.
 */
NF_INLINE void nf_dsv4_rope_rows(NF_LOCAL_PTR float* rows, int n_rows, int width, int n_rot, int mode, int p,
                                 float sin_sign, float theta_scale, float freq_scale, float ext_factor,
                                 float attn_factor, float corr_lo, float corr_hi) {
    const int half_rot = n_rot >> 1;
    const int n_offs = width - n_rot;
    const int span = nf_rope_span(mode, n_rot);
    for (int k = get_local_id(0); k < n_rows * half_rot; k += get_local_size(0)) {
        const int r = k / half_rot;
        const int i = k - r * half_rot;
        const float2 cs = nf_rope_cs(p, i, theta_scale, 1.0f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);
        const float sn = cs.y * sin_sign;
        const int i0 = r * width + n_offs + nf_rope_first(mode, i);
        const int i1 = i0 + span;
        const float x0 = rows[i0];
        const float x1 = rows[i1];
        rows[i0] = x0 * cs.x - x1 * sn;
        rows[i1] = x0 * sn + x1 * cs.x;
    }
}

#endif /* NF_DSV4_COMMON_H */
