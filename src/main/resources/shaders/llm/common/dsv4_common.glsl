/* common/dsv4_common.glsl - DeepSeek-V4 constants and the partial-RoPE row rotation shared by dsv4_compress.comp and dsv4_attn.comp. Requires: common/nf_common.glsl, common/rope_common.glsl, common/kv_types.glsl */
/*
 * GLSL twin of kernels/llm/common/dsv4_common.h (the semantic reference). GLSL cannot pass a shared array to a
 * function, so NF_DSV4_ROPE_ROWS_FN(fn, buf) defines
 *     void fn(int base, int n_rows, int width, int n_rot, int mode, int p, float sin_sign, float theta_scale,
 *             float freq_scale, float ext_factor, float attn_factor, float corr_lo, float corr_hi)
 * over the shared float array buf: nf_dsv4_rope_rows on the n_rows rows of `width` values at buf[base + k*width].
 * It rotates the last n_rot dims of each row at position p (NORM pairs (n_offs + 2i, n_offs + 2i + 1), NEOX pairs
 * (n_offs + i, n_offs + i + n_rot/2), n_offs = width - n_rot) by the angle p * theta_scale^i of nf_rope_cs;
 * sin_sign = -1 is the inverse rotation. No barrier inside: the caller synchronizes before and after.
 */
#ifndef NF_DSV4_COMMON_GLSL
#define NF_DSV4_COMMON_GLSL

#ifndef NF_ROPE_COMMON_GLSL
#error "common/dsv4_common.glsl requires common/rope_common.glsl"
#endif
#ifndef NF_KV_TYPES_GLSL
#error "common/dsv4_common.glsl requires common/kv_types.glsl"
#endif

#define NF_DSV4_CSA_RATIO 4
#define NF_DSV4_MAX_WIDTH 1024

/* Cache element types: gguf.memory.KvDType (Dsv4Kernels.kvTypeCode), the NF_KV_* codes of common/kv_types.glsl. */
#define NF_DSV4_KV_F32    NF_KV_F32
#define NF_DSV4_KV_F16    NF_KV_F16
#define NF_DSV4_KV_Q8_0   NF_KV_Q8_0
#define NF_DSV4_KV_BF16   NF_KV_BF16
#define NF_DSV4_KV_Q4_0   NF_KV_Q4_0
#define NF_DSV4_KV_Q4_1   NF_KV_Q4_1
#define NF_DSV4_KV_IQ4_NL NF_KV_IQ4_NL
#define NF_DSV4_KV_Q5_0   NF_KV_Q5_0
#define NF_DSV4_KV_Q5_1   NF_KV_Q5_1

#define NF_DSV4_ROPE_ROWS_FN(fn, buf) \
    void fn(int base, int n_rows, int width, int n_rot, int mode, int p, float sin_sign, float theta_scale, \
            float freq_scale, float ext_factor, float attn_factor, float corr_lo, float corr_hi) { \
        const int half_rot = n_rot >> 1; \
        const int n_offs = width - n_rot; \
        const int span = nf_rope_span(mode, n_rot); \
        for (int k = int(gl_LocalInvocationID.x); k < n_rows * half_rot; k += int(gl_WorkGroupSize.x)) { \
            const int r = k / half_rot; \
            const int i = k - r * half_rot; \
            const vec2 cs = nf_rope_cs(p, i, theta_scale, 1.0, freq_scale, ext_factor, attn_factor, corr_lo, \
                                       corr_hi, i); \
            const float sn = cs.y * sin_sign; \
            const int i0 = base + r * width + n_offs + nf_rope_first(mode, i); \
            const int i1 = i0 + span; \
            const float x0 = buf[i0]; \
            const float x1 = buf[i1]; \
            buf[i0] = x0 * cs.x - x1 * sn; \
            buf[i1] = x0 * sn + x1 * cs.x; \
        } \
    }

#endif /* NF_DSV4_COMMON_GLSL */
