/* common/rope_common.glsl - RoPE helpers (YaRN correction, mrope/imrope/vision sector selection) shared by the RoPE and KV shaders. Requires: common/nf_common.glsl */
/* Portions derived from llama.cpp/ggml (MIT License), including rope_yarn, which ggml bases on the YaRN reference
 * implementation (MIT License, Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng); see THIRD-PARTY-NOTICES.md. */
/*
 * GLSL twin of kernels/llm/common/rope_common.h (the semantic reference): pair layouts, position sections, theta
 * exponents and rope_yarn (llama.cpp ggml/src/ggml-cpu/ops.cpp:6263-6365). Two helpers exist only here:
 *   nf_rope_cos_sin  sin/cos with explicit range reduction, because Vulkan bounds the GLSL builtins only
 *                    on [-pi, pi] while RoPE angles reach thousands of radians;
 *   nf_rope_f16_bits float -> IEEE half bits rounded to nearest even, because the rounding of
 *                    packHalf2x16 is implementation-defined.
 */
#ifndef NF_ROPE_COMMON_GLSL
#define NF_ROPE_COMMON_GLSL

/* ggml rope modes (GGML_ROPE_TYPE_*), equal to gguf.arch.RopeType.ggmlMode(). */
#define NF_ROPE_NORM   0
#define NF_ROPE_NEOX   2
#define NF_ROPE_MROPE  8
#define NF_ROPE_VISION 24
#define NF_ROPE_IMROPE 40

/* Largest rotated region (floats) of one head held in shared memory by the block-type LLM_KV_RopeShift_<KV>. */
#define NF_ROPE_SHIFT_MAX_DIMS 1024

/* True for the modes that read four position sections per token. */
bool nf_rope_multi(int mode) {
    return (mode & NF_ROPE_MROPE) != 0;
}

/* Rotation pairs per head. */
int nf_rope_pairs(int mode, int n_dims) {
    return mode == NF_ROPE_VISION ? n_dims : (n_dims >> 1);
}

/* Offset of the first element of pair i from the first rotated dim. */
int nf_rope_first(int mode, int i) {
    return mode == NF_ROPE_NORM ? (i << 1) : i;
}

/* Distance between the two elements of a pair. */
int nf_rope_span(int mode, int n_dims) {
    return mode == NF_ROPE_NORM ? 1 : (mode == NF_ROPE_VISION ? n_dims : (n_dims >> 1));
}

/* Position section (0 t, 1 h, 2 w, 3 e) of pair i (C nf_rope_section). */
int nf_rope_section(int mode, int i, int s0, int s1, int s2, int s3) {
    if (!nf_rope_multi(mode)) {
        return 0;
    }
    const int sect_dims = s0 + s1 + s2 + s3;
    if (sect_dims <= 0) {
        return 0;
    }
    const int sector = i % sect_dims;
    if (mode == NF_ROPE_IMROPE) {
        const int m3 = sector % 3;
        if (m3 == 1 && sector < 3 * s1) return 1;
        if (m3 == 2 && sector < 3 * s2) return 2;
        if (m3 == 0 && sector < 3 * s0) return 0;
        return 3;
    }
    if (sector < s0) return 0;
    if (sector < s0 + s1) return 1;
    if (sector < s0 + s1 + s2) return 2;
    return 3;
}

/* Exponent of theta_scale for pair i of section sec (C nf_rope_theta_exp; VISION restarts per section). */
int nf_rope_theta_exp(int mode, int i, int sec, int s0, int s1, int s2, int s3) {
    if (mode != NF_ROPE_VISION) {
        return i;
    }
    const int sect_dims = s0 + s1 + s2 + s3;
    if (sect_dims <= 0) {
        return i;
    }
    const int sec_w = s0 + s1;
    const int sec_e = sec_w + s2;
    int r = -1;
    if (sec == 0) {
        r = 0;
    } else if (sec == 1) {
        r = (s0 != 0) ? s0 : -1;
    } else if (sec == 2) {
        r = (sec_w != 0 && sec_w != s0) ? sec_w : -1;
    } else {
        r = (sec_e != 0 && sec_e != s0 && sec_e != sec_w && sec_e < sect_dims) ? sec_e : -1;
    }
    const int sector = i % sect_dims;
    const int cycle = i - sector;
    int last = 0;
    if (r >= 0 && r <= sector) {
        last = cycle + r;
    } else if (r >= 0 && cycle > 0) {
        last = cycle - sect_dims + r;
    }
    return i - last;
}

/* (cos x, sin x): x = r + k*pi/2 with a fused three-part pi/2 (Cody-Waite), then the Cephes sinf/cosf
 * minimax polynomials on [-pi/4, pi/4] and quadrant selection. */
vec2 nf_rope_cos_sin(float x) {
    const float k = floor(x * 0.636619772367581343f + 0.5f);
    precise float r = fma(-k, 1.5703125f, x);
    r = fma(-k, 4.837512969970703125e-4f, r);
    r = fma(-k, 7.54978995489188216e-8f, r);
    const float z = r * r;
    const float s = ((-1.9515295891e-4f * z + 8.3321608736e-3f) * z - 1.6666654611e-1f) * z * r + r;
    const float c = ((2.443315711809948e-5f * z - 1.388731625493765e-3f) * z + 4.166664568298827e-2f) * z * z
                    - 0.5f * z + 1.0f;
    const int q = int(k) & 3;
    if (q == 0) return vec2(c, s);
    if (q == 1) return vec2(-s, c);
    if (q == 2) return vec2(-c, -s);
    return vec2(s, -c);
}

/* rope_yarn (C nf_rope_yarn): (cos * mscale, sin * mscale) of the YaRN-corrected angle. */
vec2 nf_rope_yarn(float theta_extrap, float freq_scale, float ext_factor, float attn_factor,
                  float corr_lo, float corr_hi, int i) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0) {
        const float y = (float(i) - corr_lo) / max(0.001, corr_hi - corr_lo);
        const float ramp_mix = (1.0 - min(1.0, max(0.0, y))) * ext_factor;
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0 + 0.1 * log(1.0 / freq_scale);
    }
    return nf_rope_cos_sin(theta) * mscale;
}

/* (cos, sin) of pair i at position p of its section: theta_extrap = p * theta_scale^e / ff (C nf_rope_cs). */
vec2 nf_rope_cs(int p, int e, float theta_scale, float ff, float freq_scale, float ext_factor,
                float attn_factor, float corr_lo, float corr_hi, int i) {
    const float theta_base = float(p) * pow(theta_scale, float(e));
    return nf_rope_yarn(theta_base / ff, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);
}

/* IEEE half bits of f, rounded to nearest even (as OpenCL vstore_half and CUDA __float2half_rn);
 * NaN -> quiet NaN, overflow -> infinity, values below 2^-25 -> signed zero. */
uint nf_rope_f16_bits(float f) {
    const uint x = floatBitsToUint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) {
        return sign | 0x7C00u | (ax > 0x7F800000u ? 0x0200u : 0u);
    }
    if (ax >= 0x47800000u) {
        return sign | 0x7C00u;
    }
    if (ax >= 0x38800000u) {
        uint h = (ax >> 13) - 0x1C000u;
        const uint rem = ax & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) {
            h += 1u;
        }
        return sign | h;
    }
    if (ax < 0x33000000u) {
        return sign;
    }
    const uint shift = 126u - (ax >> 23);
    const uint m = (ax & 0x7FFFFFu) | 0x800000u;
    uint h = m >> shift;
    const uint rem = m & ((1u << shift) - 1u);
    const uint halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) {
        h += 1u;
    }
    return sign | h;
}

#endif /* NF_ROPE_COMMON_GLSL */
