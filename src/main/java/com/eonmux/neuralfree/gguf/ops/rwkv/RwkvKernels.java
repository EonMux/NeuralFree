package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Names, argument indices, mode codes and launch geometry of the RWKV kernels in
 * {@code kernels/llm/recurrent/rwkv.cl} (GLSL twins {@code wkv6.comp}, {@code gla.comp}, {@code wkv7.comp},
 * {@code rwkv_mix.comp}).
 *
 * <p>Argument lists, in order ({@code (dyn)} marks the scalars re-set per step):
 * <ul>
 *   <li>{@link #WKV6}: {@code i n_tokens(dyn), i C, i H, R k, R v, R r, R tf, R td, RWn state, R<int> state_rows, W y}</li>
 *   <li>{@link #GLA}: {@code i n_tokens(dyn), i C, i H, i g_mode, f scale, R k, R v, R q, R g, RWn state,
 *       R<int> state_rows, W y}</li>
 *   <li>{@link #WKV7}: {@code i n_tokens(dyn), i C, i H, R r, R w, R k, R v, R a, R b, RWn state, R<int> state_rows, W y}</li>
 *   <li>{@link #TOKEN_SHIFT_LERP}: {@code i n_tokens(dyn), i n_embd, i n_mix, i mode, i shift_off,
 *       i cur_off, R cur, RWn shift_state, R lerp_x, R lerp_fused, W out}; {@code cur_off} is the element offset of
 *       the input rows in {@code cur}</li>
 *   <li>{@link #DECAY6}, {@link #DECAY7}: {@code i n(dyn), i n_cols, R w_in, R base, W out}</li>
 *   <li>{@link #KK_PREP}: {@code i n_tokens(dyn), i n_head, i S, R k, R k_k, R k_a, R a, W kk_out, W k_out,
 *       W aa_out, W bb_out}</li>
 *   <li>{@link #BONUS}: {@code i n_tokens(dyn), i n_head, i S, R k, R r, R r_k, R v, RWn x}</li>
 * </ul>
 * Activations are token-major rows of {@code C = H * S} floats. The WKV state of one sequence row holds
 * {@code H * S * S} floats; the kernels process one sequence whose row is {@code state_rows[0]}. Semantics
 * are those of the CPU twins {@link CpuWkv} and {@link CpuRwkvMix}.
 */
public final class RwkvKernels {

    /** RWKV6 WKV recurrence; state {@code [h][key][value]}. */
    public static final String WKV6 = "LLM_RWKV_WKV6";
    /** Gated linear attention (qrwkv); state {@code [h][key][value]}. */
    public static final String GLA = "LLM_GLA";
    /** RWKV7 WKV recurrence; state {@code [h][value][key]}. */
    public static final String WKV7 = "LLM_RWKV_WKV7";
    /** Token shift fused with the time-mix / channel-mix lerps; also stores the new shift state. */
    public static final String TOKEN_SHIFT_LERP = "LLM_RWKV_TokenShiftLerp";
    /** RWKV6 decay {@code exp(-exp(w + base))}. */
    public static final String DECAY6 = "LLM_RWKV_Decay6";
    /** RWKV7 decay {@code exp(-0.606531 * sigmoid(w + base))}. */
    public static final String DECAY7 = "LLM_RWKV_Decay7";
    /** RWKV7 key normalization and the {@code a}/{@code b} operands of the WKV7 recurrence. */
    public static final String KK_PREP = "LLM_RWKV7_KkPrep";
    /** RWKV7 r_k bonus {@code x += v * sum(k * r * r_k)} per head. */
    public static final String BONUS = "LLM_RWKV7_Bonus";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.unmodifiableList(Arrays.asList(
            WKV6, GLA, WKV7, TOKEN_SHIFT_LERP, DECAY6, DECAY7, KK_PREP, BONUS));

    /** {@code n_tokens} of {@link #WKV6}, {@link #GLA} and {@link #WKV7}. */
    public static final int ARG_WKV_N_TOKENS = 0;
    /** {@code n_tokens} of {@link #TOKEN_SHIFT_LERP}. */
    public static final int ARG_SHIFT_N_TOKENS = 0;
    /** {@code n} (total element count) of {@link #DECAY6} and {@link #DECAY7}. */
    public static final int ARG_DECAY_N = 0;
    /** {@code n_tokens} of {@link #KK_PREP} and {@link #BONUS}. */
    public static final int ARG_HEAD_ROW_N_TOKENS = 0;

    /** {@link #GLA} {@code g_mode}: one gate per key channel, {@code g[t * C + h * S + i]}. */
    public static final int GLA_G_PER_CHANNEL = 0;
    /** {@link #GLA} {@code g_mode}: one scalar gate per head, {@code g[t * H + h]}. */
    public static final int GLA_G_PER_HEAD = 1;

    /** Token-shift mode: {@code out[m] = sx * lerp_fused[m] + cur} for {@code m < n_mix}; stores the shift state. */
    public static final int SHIFT_MODE_LERP = 0;
    /** Token-shift mode: {@code out[0] = sx * lerp_x + cur}; leaves the shift state unchanged. */
    public static final int SHIFT_MODE_LERP_X = 1;
    /** Token-shift mode: {@code out[m] = (out[m] + lerp_fused[m]) * sx + cur} in place; stores the shift state. */
    public static final int SHIFT_MODE_LORA = 2;
    /** Token-shift mode: {@code out[0] = sx * lerp_x + cur}, {@code out[1 + m] = sx * lerp_fused[m] + cur}; stores the shift state. */
    public static final int SHIFT_MODE_PAIR = 3;

    /** Largest head size the WKV/GLA kernels support ({@code NF_RWKV_MAX_S}). */
    public static final int MAX_HEAD_SIZE = 256;
    /** State entries (keys) one work-item keeps in registers ({@code NF_RWKV_KPT}). */
    public static final int KEYS_PER_ITEM = 32;
    /** Scale inside the RWKV7 decay exponent (models/rwkv7-base.cpp). */
    public static final float RWKV7_DECAY_SCALE = -0.606531f;
    /** l2-norm epsilon of the RWKV7 key normalization (models/rwkv7-base.cpp). */
    public static final float KK_NORM_EPS = 1e-12f;
    /** Work-group size of the one-work-item-per-element kernels ({@link #TOKEN_SHIFT_LERP}, decays). */
    public static final int ELEMENTWISE_LOCAL = 256;

    private static final int MIN_LOCAL = 32;

    private RwkvKernels() {
    }

    /**
     * Validates a WKV/GLA head size.
     *
     * @throws IllegalArgumentException when {@code headSize} is not in {@code [1, MAX_HEAD_SIZE]}
     */
    public static void checkHeadSize(int headSize) {
        if (headSize < 1 || headSize > MAX_HEAD_SIZE) {
            throw new IllegalArgumentException("RWKV head size must be in [1, " + MAX_HEAD_SIZE + "], got " + headSize);
        }
    }

    /** Work-items sharing one value index: {@code ceil(S / KEYS_PER_ITEM)} rounded up to a power of two. */
    public static int wkvParts(int headSize) {
        checkHeadSize(headSize);
        return nextPowerOfTwo((headSize + KEYS_PER_ITEM - 1) / KEYS_PER_ITEM);
    }

    /** Work-group size of {@link #WKV6}, {@link #GLA} and {@link #WKV7}: {@code min(256, max(32, P * pow2ceil(S)))}. */
    public static int wkvLocalSize(int headSize) {
        int parts = wkvParts(headSize);
        return Math.min(LlmKernelConstants.LLM_DIMX, Math.max(MIN_LOCAL, parts * nextPowerOfTwo(headSize)));
    }

    /** Global size of {@link #WKV6}, {@link #GLA} and {@link #WKV7}: {@code H * ceil(S / (L / P)) * L}. */
    public static long wkvGlobalSize(int nHead, int headSize) {
        if (nHead < 1) {
            throw new IllegalArgumentException("RWKV head count must be positive, got " + nHead);
        }
        int local = wkvLocalSize(headSize);
        int valuesPerGroup = local / wkvParts(headSize);
        int groupsPerHead = (headSize + valuesPerGroup - 1) / valuesPerGroup;
        return (long) nHead * groupsPerHead * local;
    }

    /** Work-group size of {@link #KK_PREP} and {@link #BONUS}: {@code min(256, max(32, pow2ceil(S)))}. */
    public static int headRowLocalSize(int headSize) {
        if (headSize < 1) {
            throw new IllegalArgumentException("RWKV head size must be positive, got " + headSize);
        }
        return Math.min(LlmKernelConstants.LLM_DIMX, Math.max(MIN_LOCAL, nextPowerOfTwo(headSize)));
    }

    /** Global size of {@link #KK_PREP} and {@link #BONUS}: one work-group per (token, head). */
    public static long headRowGlobalSize(int nTokens, int nHead, int headSize) {
        if (nTokens < 0 || nHead < 1) {
            throw new IllegalArgumentException("invalid RWKV head-row grid: n_tokens " + nTokens + ", n_head " + nHead);
        }
        return (long) nTokens * nHead * headRowLocalSize(headSize);
    }

    /**
     * Global size of the one-work-item-per-element kernels: {@code n} rounded up to {@link #ELEMENTWISE_LOCAL}.
     * For {@link #DECAY6} and {@link #DECAY7}, {@code n} is their {@code n} argument; {@link #TOKEN_SHIFT_LERP} uses
     * {@link #tokenShiftGlobalSize}.
     */
    public static long elementwiseGlobalSize(long n) {
        if (n < 0) {
            throw new IllegalArgumentException("element count must be non-negative, got " + n);
        }
        return Math.max(1, (n + ELEMENTWISE_LOCAL - 1) / ELEMENTWISE_LOCAL) * ELEMENTWISE_LOCAL;
    }

    /**
     * Global size of {@link #TOKEN_SHIFT_LERP}: one work-item per (token, channel), {@code nTokens * nEmbd} rounded
     * up to {@link #ELEMENTWISE_LOCAL}.
     *
     * @throws IllegalArgumentException for a negative token count, a non-positive width, or more than
     *                                  {@code Integer.MAX_VALUE} elements per output plane
     */
    public static long tokenShiftGlobalSize(int nTokens, int nEmbd) {
        if (nTokens < 0 || nEmbd < 1) {
            throw new IllegalArgumentException("invalid RWKV token-shift grid: n_tokens " + nTokens + ", n_embd " + nEmbd);
        }
        long plane = (long) nTokens * nEmbd;
        if (plane > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("RWKV token-shift plane of " + plane + " elements exceeds the int range");
        }
        return elementwiseGlobalSize(plane);
    }

    /** GLA output scale {@code head_size^-0.5} (models/rwkv6-base.cpp, {@code pow(head_size, -0.5f)}). */
    public static float glaScale(int headSize) {
        checkHeadSize(headSize);
        return (float) Math.pow(headSize, -0.5);
    }

    /** True for the four {@code SHIFT_MODE_*} codes. */
    public static boolean isShiftMode(int mode) {
        return mode >= SHIFT_MODE_LERP && mode <= SHIFT_MODE_PAIR;
    }

    private static int nextPowerOfTwo(int n) {
        return n <= 1 ? 1 : Integer.highestOneBit(n - 1) << 1;
    }
}
