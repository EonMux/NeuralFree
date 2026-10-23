/* common/rope_common.h - RoPE helpers: YaRN correction ramp and mrope/imrope/vision sector selection. Requires: common/dialect.h, common/config.h, common/math.h */
/* Portions derived from llama.cpp/ggml (MIT License), including rope_yarn, which ggml bases on the YaRN reference
 * implementation (MIT License, Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng); see THIRD-PARTY-NOTICES.md. */
/*
 * Shared by rope.cl (LLM_RoPE_Ext) and kv_cache.cl (LLM_KV_RopeShift_*). Java mirror: gguf.ops.rope.RopeMath
 * (formulas) and gguf.ops.rope.RopeFreqTable (per-pair tables of the CPU twin).
 *
 * Pair i of a head (relative to the first rotated dim n_offs) rotates the elements
 *   NORM                 (2i, 2i + 1)                 i < n_dims / 2
 *   NEOX, MROPE, IMROPE  (i, i + n_dims / 2)          i < n_dims / 2
 *   VISION               (i, i + n_dims)              i < n_dims        (n_offs == 0, head = 2 * n_dims)
 * by the angle theta = p_sec * theta_scale^e(i) / ff[i] (YaRN-corrected), where p_sec is the position of the
 * section the pair belongs to (always section 0 for NORM/NEOX) and e(i) is i, except for VISION, whose
 * sections restart the exponent (independent sections). Section selection and exponents follow
 * ggml_mrope_cache_init (llama.cpp ggml/src/ggml-cpu/ops.cpp:6303-6365); the angle and magnitude follow
 * rope_yarn / rope_yarn_ramp (ops.cpp:6263-6285, ggml-cuda/rope.cu:15-44); theta_scale^e is evaluated with
 * pow as in ggml-cuda/rope.cu (theta_base = pos * powf(theta_scale, i)).
 */
#ifndef NF_ROPE_COMMON_H
#define NF_ROPE_COMMON_H

/* ggml rope modes (GGML_ROPE_TYPE_*), equal to gguf.arch.RopeType.ggmlMode(). */
#define NF_ROPE_NORM   0
#define NF_ROPE_NEOX   2
#define NF_ROPE_MROPE  8
#define NF_ROPE_VISION 24
#define NF_ROPE_IMROPE 40

/* Largest rotated region (floats) of one head that the block-type LLM_KV_RopeShift_<KV> holds in local memory;
 * Java: RopeKernels.SHIFT_MAX_DIMS. */
#define NF_ROPE_SHIFT_MAX_DIMS 1024

/* True for the modes that read four position sections per token (ggml: mode & GGML_ROPE_TYPE_MROPE). */
NF_INLINE int nf_rope_multi(int mode) {
    return (mode & NF_ROPE_MROPE) != 0;
}

/* Rotation pairs per head. */
NF_INLINE int nf_rope_pairs(int mode, int n_dims) {
    return mode == NF_ROPE_VISION ? n_dims : (n_dims >> 1);
}

/* Offset of the first element of pair i from the first rotated dim. */
NF_INLINE int nf_rope_first(int mode, int i) {
    return mode == NF_ROPE_NORM ? (i << 1) : i;
}

/* Distance between the two elements of a pair. */
NF_INLINE int nf_rope_span(int mode, int n_dims) {
    return mode == NF_ROPE_NORM ? 1 : (mode == NF_ROPE_VISION ? n_dims : (n_dims >> 1));
}

/*
 * Position section (0 t, 1 h, 2 w, 3 e) of pair i for a multi-section mode (ops.cpp:6331-6355):
 *   IMROPE  sector % 3 == 1 && sector < 3*s1 -> h; % 3 == 2 && sector < 3*s2 -> w;
 *           % 3 == 0 && sector < 3*s0 -> t; otherwise e
 *   MROPE, VISION  sector < s0 -> t; < s0+s1 -> h; < s0+s1+s2 -> w; otherwise e
 * with sector = i % (s0+s1+s2+s3). Single-section modes always read section 0.
 */
NF_INLINE int nf_rope_section(int mode, int i, int s0, int s1, int s2, int s3) {
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

/*
 * Exponent e(i) of theta_scale for pair i. Every mode but VISION uses e = i. VISION resets the section's
 * running theta to its base at the section start (ops.cpp:6320-6335 indep_sects, else-if chain):
 * t restarts at sector 0, h at s0 (s0 != 0), w at s0+s1 (s1 != 0), e at s0+s1+s2 (s2 != 0, below
 * sect_dims); e(i) = i - (index of the last restart of section sec at or before i), 0 when it never
 * restarted.
 */
NF_INLINE int nf_rope_theta_exp(int mode, int i, int sec, int s0, int s1, int s2, int s3) {
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

/*
 * rope_yarn (ops.cpp:6268-6285): theta_interp = freq_scale * theta_extrap; with ext_factor != 0 the YaRN
 * ramp mixes interpolation and extrapolation, ramp = 1 - clamp((i - corr_lo) / max(0.001, corr_hi - corr_lo),
 * 0, 1), and the magnitude grows by 1 + 0.1 * log(1 / freq_scale). Returns (cos * mscale, sin * mscale).
 * Precise cos/sin (never native_*): angles reach thousands of radians and need full range reduction.
 */
NF_INLINE float2 nf_rope_yarn(float theta_extrap, float freq_scale, float ext_factor, float attn_factor,
                              float corr_lo, float corr_hi, int i) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float y = ((float)i - corr_lo) / fmax(0.001f, corr_hi - corr_lo);
        const float ramp_mix = (1.0f - fmin(1.0f, fmax(0.0f, y))) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return NF_F2(cos(theta) * mscale, sin(theta) * mscale);
}

/* (cos, sin) of pair i at position p of its section: theta_extrap = p * theta_scale^e / ff. */
NF_INLINE float2 nf_rope_cs(int p, int e, float theta_scale, float ff, float freq_scale, float ext_factor,
                            float attn_factor, float corr_lo, float corr_hi, int i) {
    const float theta_base = (float)p * pow(theta_scale, (float)e);
    return nf_rope_yarn(theta_base / ff, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);
}

#endif /* NF_ROPE_COMMON_H */
