package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;

import java.util.Objects;

/**
 * RoPE scalar math, the single Java source of the formulas the RoPE kernels evaluate
 * ({@code kernels/llm/common/rope_common.h}) and the parameters the resolvers derive from hparams. Every function
 * works in single precision with the operation order of the cited llama.cpp source:
 * <ul>
 *   <li>{@code theta_scale = powf(freq_base, -2.0f / n_dims)} (ggml-cpu/ops.cpp:6455);</li>
 *   <li>YaRN correction dims {@code ggml_rope_yarn_corr_dims} (ggml/src/ggml.c:4492-4504);</li>
 *   <li>{@code rope_yarn_ramp} and {@code rope_yarn} (ggml-cpu/ops.cpp:6263-6285, ggml-cuda/rope.cu:15-44);</li>
 *   <li>pair layout, position sections and theta exponents of {@code ggml_mrope_cache_init}
 *       (ggml-cpu/ops.cpp:6303-6365) and {@code rotate_pairs} (ops.cpp:6368-6386);</li>
 *   <li>the context-level YaRN attention factor (src/llama-context.cpp:265-306) and the DeepSeek KQ mscale
 *       (src/models/deepseek2.cpp:302-309).</li>
 * </ul>
 * Portions derived from llama.cpp/ggml (MIT License), including rope_yarn, which ggml bases on the YaRN reference
 * implementation (MIT License, Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng); see THIRD-PARTY-NOTICES.md.
 */
public final class RopeMath {

    /** Lower bound of the YaRN ramp denominator ({@code MAX(0.001f, high - low)}). */
    private static final float RAMP_MIN_SPAN = 0.001f;

    private RopeMath() {
    }

    // ------------------------------------------------------------------ frequencies

    /**
     * {@code powf(freqBase, -2.0f / nDims)}: ratio of consecutive pair frequencies.
     *
     * @throws IllegalArgumentException for a non-positive base or dim count
     */
    public static float thetaScale(float freqBase, int nDims) {
        if (!(freqBase > 0f) || nDims <= 0) {
            throw new IllegalArgumentException("rope theta scale needs freqBase > 0 and nDims > 0, got " + freqBase
                    + " / " + nDims);
        }
        final float exponent = -2.0f / nDims;
        return (float) Math.pow(freqBase, exponent);
    }

    /** {@code ggml_rope_yarn_corr_dim}: {@code nDims * logf(nCtxOrig / (nRot * 2 pi)) / (2 * logf(base))}. */
    public static float corrDim(int nDims, int nCtxOrig, float nRot, float base) {
        final float ratio = nCtxOrig / (nRot * 2 * (float) Math.PI);
        return nDims * (float) Math.log(ratio) / (2 * (float) Math.log(base));
    }

    /**
     * {@code ggml_rope_yarn_corr_dims}: {@code [max(0, floorf(corrDim(betaFast))), min(nDims - 1,
     * ceilf(corrDim(betaSlow)))]}, the pair-index range over which the YaRN ramp blends.
     *
     * @throws IllegalArgumentException for a non-positive dim count, context or base
     */
    public static float[] corrDims(int nDims, int nCtxOrig, float freqBase, float betaFast, float betaSlow) {
        if (nDims <= 0 || nCtxOrig <= 0 || !(freqBase > 0f)) {
            throw new IllegalArgumentException("YaRN correction dims need nDims, nCtxOrig and freqBase > 0, got "
                    + nDims + ", " + nCtxOrig + ", " + freqBase);
        }
        final float start = (float) Math.floor(corrDim(nDims, nCtxOrig, betaFast, freqBase));
        final float end = (float) Math.ceil(corrDim(nDims, nCtxOrig, betaSlow, freqBase));
        return new float[]{Math.max(0f, start), Math.min(nDims - 1, end)};
    }

    // ------------------------------------------------------------------ YaRN rotation

    /** {@code rope_yarn_ramp}: {@code 1 - min(1, max(0, (i - lo) / max(0.001, hi - lo)))} for pair index {@code i}. */
    public static float yarnRamp(float corrLo, float corrHi, int pair) {
        final float y = (pair - corrLo) / Math.max(RAMP_MIN_SPAN, corrHi - corrLo);
        return 1 - Math.min(1f, Math.max(0f, y));
    }

    /**
     * Magnitude applied to cos/sin inside {@code rope_yarn}: {@code attnFactor * (1 + 0.1 * logf(1 / freqScale))}
     * with YaRN extrapolation ({@code extFactor != 0}), else {@code attnFactor}.
     */
    public static float rotationMscale(float attnFactor, float freqScale, float extFactor) {
        if (extFactor == 0.0f) {
            return attnFactor;
        }
        return attnFactor * (1.0f + 0.1f * (float) Math.log(1.0f / freqScale));
    }

    /**
     * Rotation angle of {@code rope_yarn}: {@code freqScale * thetaExtrap}, blended towards {@code thetaExtrap} by
     * {@code yarnRamp * extFactor} when {@code extFactor != 0}.
     */
    public static float angle(float thetaExtrap, float freqScale, float extFactor, float corrLo, float corrHi,
                              int pair) {
        final float thetaInterp = freqScale * thetaExtrap;
        if (extFactor == 0.0f) {
            return thetaInterp;
        }
        final float rampMix = yarnRamp(corrLo, corrHi, pair) * extFactor;
        return thetaInterp * (1 - rampMix) + thetaExtrap * rampMix;
    }

    /**
     * {@code rope_yarn}: writes {@code cos(theta) * mscale} to {@code dst[off]} and {@code sin(theta) * mscale} to
     * {@code dst[off + 1]} for {@link #angle} and {@link #rotationMscale}.
     */
    public static void cosSin(float thetaExtrap, float freqScale, float extFactor, float attnFactor, float corrLo,
                              float corrHi, int pair, float[] dst, int off) {
        final float theta = angle(thetaExtrap, freqScale, extFactor, corrLo, corrHi, pair);
        final float mscale = rotationMscale(attnFactor, freqScale, extFactor);
        dst[off] = (float) Math.cos(theta) * mscale;
        dst[off + 1] = (float) Math.sin(theta) * mscale;
    }

    // ------------------------------------------------------------------ context-level YaRN parameters

    /** {@code yarn_ext_factor} when not set explicitly: 1 for YaRN scaling, else 0 (llama-context.cpp:265-267). */
    public static float defaultExtFactor(RopeScalingType scaling) {
        Objects.requireNonNull(scaling, "scaling");
        return scaling == RopeScalingType.YARN ? 1.0f : 0.0f;
    }

    /** {@code get_mscale(scale, mscale)}: {@code scale <= 1 ? 1 : 0.1 * mscale * logf(scale) + 1}. */
    public static float yarnGetMscale(float scale, float mscale) {
        return scale <= 1.0f ? 1.0f : (0.1f * mscale * (float) Math.log(scale) + 1.0f);
    }

    /**
     * Effective {@code yarn_attn_factor} of llama-context.cpp:269-305. With {@code extFactor != 0} the factor is
     * {@code get_mscale(1/freqScale, mscale) / get_mscale(1/freqScale, yarnLogMul)} ({@code mscale = yarnLogMul} for
     * DeepSeek-V2 when it differs from 1, else 1) or {@code get_mscale(1/freqScale, 1)} without a log multiplier,
     * then divided by {@code 1 + 0.1 * logf(1/freqScale)} to cancel the factor {@code rope_yarn} applies; without
     * extrapolation it is {@code baseAttnFactor}. Both cases are finally multiplied by {@code ropeAttnFactor}.
     *
     * @param baseAttnFactor  configured {@code yarn_attn_factor} (hparams; 1 by default)
     * @param extFactor       resolved {@code yarn_ext_factor}
     * @param freqScale       context frequency scale ({@code rope_freq_scale}; 1 for scaling type NONE)
     * @param yarnLogMul      {@code rope.scaling.yarn_log_multiplier} (0 when absent)
     * @param deepseek2       true for the DeepSeek-V2 architecture ({@code TAG_DEEPSEEK2_YARN_LOG_MUL_FIX})
     * @param ropeAttnFactor  {@code rope.attn_factor} (1 when absent)
     * @throws IllegalArgumentException for a non-positive frequency scale with extrapolation
     */
    public static float yarnAttnFactor(float baseAttnFactor, float extFactor, float freqScale, float yarnLogMul,
                                       boolean deepseek2, float ropeAttnFactor) {
        float attn = baseAttnFactor;
        if (extFactor != 0.0f) {
            if (!(freqScale > 0f)) {
                throw new IllegalArgumentException("YaRN needs a positive rope frequency scale, got " + freqScale);
            }
            final float factor = 1.0f / freqScale;
            if (yarnLogMul != 0.0f) {
                final float mscale = deepseek2 && yarnLogMul != 1.0f ? yarnLogMul : 1.0f;
                attn = yarnGetMscale(factor, mscale) / yarnGetMscale(factor, yarnLogMul);
            } else {
                attn = yarnGetMscale(factor, 1.0f);
            }
            attn *= 1.0f / (1.0f + 0.1f * (float) Math.log(factor));
        }
        return attn * ropeAttnFactor;
    }

    /**
     * DeepSeek attention-score mscale (deepseek2.cpp:302-306): the original attention factor
     * {@code attnFactor * (1 + 0.1 * logf(1/freqScale))} times {@code 1 + 0.1 * yarnLogMul * logf(1/freqScale)};
     * the KQ scale is {@code mscale^2 / sqrt(headDimK)}.
     */
    public static float deepseekKqMscale(float attnFactor, float freqScale, float yarnLogMul) {
        final float attnFactorOrg = attnFactor * (1.0f + 0.1f * (float) Math.log(1.0f / freqScale));
        return attnFactorOrg * (1.0f + 0.1f * yarnLogMul * (float) Math.log(1.0f / freqScale));
    }

    // ------------------------------------------------------------------ pair geometry and sections

    /** True for the modes that read four position sections per token (ggml {@code mode & GGML_ROPE_TYPE_MROPE}). */
    public static boolean multiSection(RopeType type) {
        return type == RopeType.MROPE || type == RopeType.IMROPE || type == RopeType.VISION;
    }

    /** Positions per token in the kernel's position buffer: 4 for multi-section modes, else 1. */
    public static int positionsPerToken(RopeType type) {
        return multiSection(requireRotary(type)) ? MropeTextPositions.SECTIONS : 1;
    }

    /** Rotation pairs per head: {@code nDims} for VISION, {@code nDims / 2} otherwise. */
    public static int pairCount(RopeType type, int nDims) {
        return requireRotary(type) == RopeType.VISION ? nDims : nDims / 2;
    }

    /** Head elements a rotation covers starting at {@code nOffs}: {@code 2 * nDims} for VISION, else {@code nDims}. */
    public static int rotatedWidth(RopeType type, int nDims) {
        return requireRotary(type) == RopeType.VISION ? 2 * nDims : nDims;
    }

    /** Offset of the first element of pair {@code pair} from the first rotated dim ({@code 2i} for NORM, else {@code i}). */
    public static int pairFirst(RopeType type, int pair) {
        return requireRotary(type) == RopeType.NORM ? 2 * pair : pair;
    }

    /** Distance between the two elements of a pair: 1 (NORM), {@code nDims} (VISION), {@code nDims / 2} otherwise. */
    public static int pairSpan(RopeType type, int nDims) {
        switch (requireRotary(type)) {
            case NORM:
                return 1;
            case VISION:
                return nDims;
            default:
                return nDims / 2;
        }
    }

    /**
     * Position section (0 t, 1 h, 2 w, 3 e) of pair {@code pair}; 0 for single-section modes. IMROPE interleaves
     * {@code sector % 3} (h, w, t limited to {@code 3 * sections[k]} sectors, else e); MROPE and VISION use
     * contiguous sections; {@code sector = pair % sum(sections)} (ggml-cpu/ops.cpp:6331-6355).
     *
     * @throws IllegalArgumentException when {@code sections} does not hold four values
     */
    public static int section(RopeType type, int pair, int[] sections) {
        if (!multiSection(requireRotary(type))) {
            return 0;
        }
        final int[] s = requireSections(sections);
        final int sectDims = s[0] + s[1] + s[2] + s[3];
        if (sectDims <= 0) {
            return 0;
        }
        final int sector = pair % sectDims;
        if (type == RopeType.IMROPE) {
            final int m3 = sector % 3;
            if (m3 == 1 && sector < 3 * s[1]) {
                return 1;
            }
            if (m3 == 2 && sector < 3 * s[2]) {
                return 2;
            }
            if (m3 == 0 && sector < 3 * s[0]) {
                return 0;
            }
            return 3;
        }
        if (sector < s[0]) {
            return 0;
        }
        if (sector < s[0] + s[1]) {
            return 1;
        }
        if (sector < s[0] + s[1] + s[2]) {
            return 2;
        }
        return 3;
    }

    /**
     * Exponent of {@code theta_scale} for pair {@code pair} of section {@code section}: the pair index, except for
     * VISION, whose independent sections restart their running theta at the section start (ops.cpp:6320-6335:
     * t at sector 0, h at {@code s0} when {@code s0 != 0}, w at {@code s0+s1} when {@code s1 != 0}, e at
     * {@code s0+s1+s2} when it is a distinct sector below the cycle length).
     */
    public static int thetaExponent(RopeType type, int pair, int section, int[] sections) {
        if (requireRotary(type) != RopeType.VISION) {
            return pair;
        }
        final int[] s = requireSections(sections);
        final int sectDims = s[0] + s[1] + s[2] + s[3];
        if (sectDims <= 0) {
            return pair;
        }
        final int secW = s[0] + s[1];
        final int secE = secW + s[2];
        final int restart;
        switch (section) {
            case 0:
                restart = 0;
                break;
            case 1:
                restart = s[0] != 0 ? s[0] : -1;
                break;
            case 2:
                restart = secW != 0 && secW != s[0] ? secW : -1;
                break;
            case 3:
                restart = secE != 0 && secE != s[0] && secE != secW && secE < sectDims ? secE : -1;
                break;
            default:
                throw new IllegalArgumentException("rope section " + section + " outside 0..3");
        }
        final int sector = pair % sectDims;
        final int cycle = pair - sector;
        int last = 0;
        if (restart >= 0 && restart <= sector) {
            last = cycle + restart;
        } else if (restart >= 0 && cycle > 0) {
            last = cycle - sectDims + restart;
        }
        return pair - last;
    }

    /**
     * RoPE type used to shift a K cache: MROPE and IMROPE caches are shifted with NEOX and a single position
     * (llama-kv-cache.cpp:1962-1969); NORM and NEOX shift as themselves.
     *
     * @throws IllegalArgumentException for NONE and VISION, which never back a shiftable KV cache
     */
    public static RopeType shiftType(RopeType type) {
        switch (Objects.requireNonNull(type, "type")) {
            case NORM:
                return RopeType.NORM;
            case NEOX:
            case MROPE:
            case IMROPE:
                return RopeType.NEOX;
            default:
                throw new IllegalArgumentException("a KV cache with rope type " + type + " cannot be shifted");
        }
    }

    private static RopeType requireRotary(RopeType type) {
        if (Objects.requireNonNull(type, "type") == RopeType.NONE) {
            throw new IllegalArgumentException("rope type NONE has no rotation");
        }
        return type;
    }

    private static int[] requireSections(int[] sections) {
        Objects.requireNonNull(sections, "sections");
        if (sections.length != MropeTextPositions.SECTIONS) {
            throw new IllegalArgumentException("rope sections need " + MropeTextPositions.SECTIONS + " values, got "
                    + sections.length);
        }
        return sections;
    }
}
