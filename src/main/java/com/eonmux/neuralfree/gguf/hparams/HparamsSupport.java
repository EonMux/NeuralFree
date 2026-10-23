package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.SwaType;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Helpers shared by {@link HparamsLoader} and the architecture hooks: SWA and recurrent layer patterns,
 * RoPE sections and YaRN keys, MoE gating and the logit-scale direction. Each helper transcribes one
 * llama.cpp routine; hooks combine them with their architecture's own conditions.
 */
public final class HparamsSupport {

    private HparamsSupport() {
    }

    // ---- sliding window ----

    /**
     * Marks SWA layers by period (llama-hparams.cpp:8-22 {@code set_swa_pattern}): with
     * {@code denseFirst} layer {@code il} is SWA iff {@code n == 0 || il % n != 0}, otherwise iff
     * {@code n == 0 || il % n < n - 1}. NextN blocks are never SWA. {@code n == 0} makes every trunk layer
     * SWA, {@code n == 1} none.
     */
    public static void setSwaPattern(ModelHparamsBuilder b, int nPattern, boolean denseFirst) {
        boolean[] swa = periodicPattern(b, nPattern, denseFirst);
        b.isSwaArray(swa);
    }

    /**
     * Loads SWA layers as llama.cpp {@code load_swa_pattern} (llama-model.cpp:3922-3929): a bool array
     * {@code attention.sliding_window_pattern} marks layers directly; otherwise its integer form, when
     * present, replaces {@code defaultPattern}; then {@link #setSwaPattern} applies. As in llama.cpp
     * {@code get_arr}, missing trailing array entries mean dense layers and entries past the block count are
     * not used.
     *
     * @throws GgufFormatException when the bool array exceeds {@link ModelHparams#MAX_LAYERS} entries
     */
    public static void loadSwaPattern(ModelHparamsBuilder b, GgufKv kv, int defaultPattern, boolean denseFirst) {
        String key = GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN;
        if (kv.isArray(key)) {
            boolean[] flags = kv.boolArray(key);
            if (flags.length > ModelHparams.MAX_LAYERS) {
                throw new GgufFormatException("metadata key '" + kv.key(key) + "' has " + flags.length
                        + " entries, more than the maximum of " + ModelHparams.MAX_LAYERS + " layers");
            }
            boolean[] swa = new boolean[b.nLayerAll()];
            System.arraycopy(flags, 0, swa, 0, Math.min(flags.length, swa.length));
            b.isSwaArray(swa);
            return;
        }
        setSwaPattern(b, kv.u32(key, defaultPattern), denseFirst);
    }

    /**
     * Reads {@code attention.sliding_window} into {@link ModelHparamsBuilder#nSwa()}.
     *
     * @return true when the key is present
     * @throws GgufFormatException when {@code required} and the key is missing
     */
    public static boolean readSlidingWindow(ModelHparamsBuilder b, GgufKv kv, boolean required) {
        if (required) {
            b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));
            return true;
        }
        if (!kv.has(GgufKeys.ATTENTION_SLIDING_WINDOW)) {
            return false;
        }
        b.nSwa(kv.u32(GgufKeys.ATTENTION_SLIDING_WINDOW, 0));
        return true;
    }

    /** Enables sliding-window attention with the given mask type and window. */
    public static void enableSwa(ModelHparamsBuilder b, SwaType type, int nSwa) {
        if (type == SwaType.NONE || nSwa <= 0) {
            throw new IllegalArgumentException("SWA needs a mask type and a positive window, got " + type + "/" + nSwa);
        }
        b.swaType(type);
        b.nSwa(nSwa);
    }

    /** Disables sliding-window attention and clears every SWA layer flag. */
    public static void disableSwa(ModelHparamsBuilder b) {
        b.swaType(SwaType.NONE);
        b.nSwa(0);
        b.isSwaArray(new boolean[b.nLayerAll()]);
    }

    private static boolean[] periodicPattern(ModelHparamsBuilder b, int n, boolean denseFirst) {
        if (n < 0) {
            throw new IllegalArgumentException("layer pattern period must be non-negative, got " + n);
        }
        boolean[] out = new boolean[b.nLayerAll()];
        for (int il = 0; il < b.nLayer(); il++) {
            out[il] = denseFirst ? (n == 0 || il % n != 0) : (n == 0 || il % n < n - 1);
        }
        return out;
    }

    // ---- RoPE ----

    /**
     * Reads {@code rope.dimension_sections} (four entries, scalar broadcast) as llama.cpp
     * {@code get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, rope_sections, 4, required)}.
     *
     * @return true when the key is present
     * @throws GgufFormatException when {@code required} and the key is missing, or the array length is not 4
     */
    public static boolean readRopeSections(ModelHparamsBuilder b, GgufKv kv, boolean required) {
        String key = GgufKeys.ROPE_DIMENSION_SECTIONS;
        if (!kv.has(key)) {
            if (required) {
                throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
            }
            return false;
        }
        b.ropeSections(kv.keyOrArr(key, ModelHparams.MROPE_SECTIONS, 0));
        return true;
    }

    /**
     * Reads {@code rope.freq_base_swa} into the SWA RoPE base, using {@code dflt} when the key is absent:
     * the full-attention base for archs that copy it first (cohere2.cpp:7-10), 10000 for archs that do not
     * (gemma4.cpp:14).
     */
    public static void readRopeFreqBaseSwa(ModelHparamsBuilder b, GgufKv kv, float dflt) {
        b.ropeFreqBaseSwa(kv.f32(GgufKeys.ROPE_FREQ_BASE_SWA, dflt));
    }

    // ---- mixture of experts ----

    /**
     * Router gating from {@code expert_gating_func} (llama-hparams.h:17-23 codes); absent or
     * {@code NONE (0)} yields {@code dflt}.
     *
     * @throws GgufFormatException for an unknown code
     */
    public static GatingFunc readGating(GgufKv kv, GatingFunc dflt) {
        long code = kv.u32Long(GgufKeys.EXPERT_GATING_FUNC, GatingFunc.GGUF_CODE_NONE);
        try {
            GatingFunc g = GatingFunc.fromGgufCode(code);
            return g != null ? g : dflt;
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.EXPERT_GATING_FUNC) + "': "
                    + e.getMessage(), e);
        }
    }

    /**
     * Reads the per-layer expert FFN size {@code expert_feed_forward_length} (scalar broadcast or per-layer
     * array) into {@link ModelHparamsBuilder#nFfExpArray()}.
     *
     * @return true when the key is present
     */
    public static boolean readExpertFfLength(ModelHparamsBuilder b, GgufKv kv, boolean required) {
        String key = GgufKeys.EXPERT_FEED_FORWARD_LENGTH;
        if (!kv.has(key)) {
            if (required) {
                throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
            }
            return false;
        }
        b.nFfExpArray(kv.keyOrArr(key, b.nLayerAll(), 0));
        return true;
    }

    // ---- logit scale ----

    /**
     * Direction in which {@code logit_scale} is applied. Multiplying archs scale logits by the factor
     * (command-r/cohere graphs, grok, talkie, muse-*, dflash, and minicpm3's {@code n_embd_base / n_embd} LM-head
     * scale, minicpm3.cpp:241-242); dividing archs use the granite graph
     * {@code logits * (1 / f_logit_scale)} (granite.cpp:198, minicpm.cpp).
     */
    public static final class ScalingDefaults {

        private static final Set<Arch> MULTIPLIES = Collections.unmodifiableSet(EnumSet.of(
                Arch.COMMAND_R, Arch.COHERE2, Arch.COHERE2MOE, Arch.GROK, Arch.DFLASH, Arch.TALKIE,
                Arch.MUSE_GLIMMER, Arch.MUSE_SPARK, Arch.MINICPM3));

        private static final Set<Arch> DIVIDES = Collections.unmodifiableSet(EnumSet.of(
                Arch.GRANITE, Arch.GRANITE_MOE, Arch.GRANITE_HYBRID, Arch.GRANITE_SWITCH, Arch.GRANITE_SWA,
                Arch.MINICPM));

        private ScalingDefaults() {
        }

        /** True when the architecture multiplies logits by {@code logit_scale}. */
        public static boolean multipliesLogits(Arch arch) {
            return MULTIPLIES.contains(arch);
        }

        /** True when the architecture divides logits by {@code logit_scale}. */
        public static boolean dividesLogits(Arch arch) {
            return DIVIDES.contains(arch);
        }
    }
}
