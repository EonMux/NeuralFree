package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Locale;

/**
 * Selection mode of the certified exact-argmax LM head, the greedy decode path that picks the token on
 * the device from an INT8 screen of the head instead of materializing the exact logits row: a per-row
 * INT8 copy of the head weights ({@code s_v} per-row absmax scale, rounded codes) approximates every
 * logit, a Cauchy-Schwarz error bound {@code b_rv = ||h_r|| * eb_v} with
 * {@code eb_v = (||W_v - s_v q_v|| + c_acc ||W_v||) (1 + 2^-20)} certifies which rows can still win,
 * the candidates are rescored with the head's own exact kernel and first-index ties, and a row with
 * more candidates than {@link CertifiedArgmaxTuning#kmax()} falls back to the full exact head. The
 * result is the argmax of the exact logits. Only greedy chains without logprobs, logit bias, grammar,
 * penalties or a verifier logit reader reach it: every other consumer reads the logits rows and takes
 * the regular head by construction.
 *
 * <ul>
 *   <li>{@link #AUTO}: greedy decode and greedy speculative verify take the certified lane on the
 *       (backend, head type) pairs that measured faster than the exact readout, at heads large enough
 *       to amortize the lane's fixed per-token cost ({@link CertifiedArgmaxTuning}), and keep the exact
 *       readout everywhere else &mdash; this is the default.</li>
 *   <li>{@link #ON}: the lane runs wherever its kernels exist ({@link CertifiedArgmaxTuning}),
 *       measured or not, on GPU and CPU heads alike.</li>
 *   <li>{@link #OFF}: the head is always evaluated exactly.</li>
 * </ul>
 *
 * <p>Whichever mode, the head must also carry none of the per-logit transforms the lane cannot certify
 * by name (output bias, logit scale vector, softcap, sigmoid cap, masked range, LoRA deltas, a
 * non-identity weight transform), and on the GPU path a one-row projection another lane serves (a fast
 * GEMM, a tiled GEMM or the int8 matvec lane) keeps that lane, because the certified rescore repeats
 * only the float matvec family's arithmetic.</p>
 *
 * <p>The mode is read from the system property {@value #PROPERTY}; the draft-lane property
 * {@code neuralfree.draft.argmax.head} is accepted as an alias, as is the draft-lane value
 * {@code int8} for {@link #ON}. An unknown value is refused by name. No llama.cpp or vLLM flag pairs
 * with it (vLLM gates its SpecPilot head through the {@code SPECPILOT_D4_HEAD} environment variable,
 * which has no command-line alias either).</p>
 */
public enum CertifiedArgmaxMode {

    /** Greedy decode takes the certified lane exactly where it measured faster (the default). */
    AUTO,
    /** The certified lane runs wherever its kernels exist, measured or not. */
    ON,
    /** The head is always evaluated exactly. */
    OFF;

    /** System property holding the mode: {@code auto} (default), {@code on} or {@code off}. */
    public static final String PROPERTY = "neuralfree.argmax.head";

    /** Draft-lane system property accepted as an alias of {@link #PROPERTY}. */
    public static final String LEGACY_PROPERTY = "neuralfree.draft.argmax.head";

    /** Draft-lane value accepted for {@link #ON}. */
    private static final String ON_ALIAS = "int8";

    /**
     * The configured mode: {@value #PROPERTY}, or its draft-lane alias when the new property is unset
     * or blank, or {@link #AUTO} when neither is set.
     *
     * @throws IllegalArgumentException naming the property, the value and the allowed values when the
     *                                  property holds anything else
     */
    public static CertifiedArgmaxMode configured() {
        String value = System.getProperty(PROPERTY);
        String property = PROPERTY;
        if (value == null || value.trim().isEmpty()) {
            value = System.getProperty(LEGACY_PROPERTY);
            property = LEGACY_PROPERTY;
        }
        if (value == null || value.trim().isEmpty()) {
            return AUTO;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if (ON_ALIAS.equals(normalized)) {
            return ON;
        }
        for (CertifiedArgmaxMode mode : values()) {
            if (mode.name().toLowerCase(Locale.ROOT).equals(normalized)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("-D" + property + " must be one of auto, on, off (the draft alias '"
                + ON_ALIAS + "' accepted for on), got '" + value + "'");
    }

    /**
     * Whether the certified lane serves a head of {@code type} on {@code backend} of {@code elements}
     * weight elements (vocab rows times hidden columns): OFF never, ON wherever the lane's kernels
     * exist, AUTO exactly where the lane measured faster than the exact readout and the head is large
     * enough to amortize its fixed cost ({@link CertifiedArgmaxTuning}).
     */
    public boolean permitsScreen(BackendType backend, GGMLType type, long elements) {
        if (this == OFF) {
            return false;
        }
        return this == ON
                ? CertifiedArgmaxTuning.screenCapable(backend, type)
                : CertifiedArgmaxTuning.screenFaster(backend, type, elements);
    }

    /**
     * Whether the certified lane serves a CPU head of {@code type}: OFF never, ON wherever the CPU lane
     * exists, AUTO exactly where it measured faster ({@link CertifiedArgmaxTuning}).
     */
    public boolean permitsCpuScreen(GGMLType type) {
        if (this == OFF) {
            return false;
        }
        return this == ON
                ? CertifiedArgmaxTuning.cpuCapable(type)
                : CertifiedArgmaxTuning.cpuFaster(type);
    }
}
