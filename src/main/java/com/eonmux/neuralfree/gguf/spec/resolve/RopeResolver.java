package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Resolves the rotary embedding of each layer into a {@link RopeSpec} holding every argument of the graph's
 * {@code ggml_rope_ext} / {@code ggml_rope_multi} call.
 *
 * <p>Context-level values follow llama-context.cpp:202-306 with default context parameters:
 * <ul>
 *   <li>{@code freq_base} and {@code freq_scale} are the trained values; scaling type NONE forces
 *       {@code freq_scale = 1} (:261-263);</li>
 *   <li>an unset {@code yarn_ext_factor} becomes 1 for YaRN scaling, else 0 (:265-267);</li>
 *   <li>{@code yarn_attn_factor} is the effective factor of :269-306 ({@link RopeMath#yarnAttnFactor}); the kernel
 *       applies the {@code 1 + 0.1 ln(1/freq_scale)} magnitude itself;</li>
 *   <li>{@code n_ctx_orig} is {@code n_ctx_orig_yarn}, which the loader defaults to {@code n_ctx_train}.</li>
 * </ul>
 * Per layer: the architectures that call {@code get_rope_freq_base/scale} (llama-model.cpp:2755-2761) use the SWA
 * base and scale on SWA layers; {@code get_rope_factors} (:2763-2776) picks {@code rope_freqs}, else
 * {@code rope_factors_long} when the per-sequence context exceeds {@code n_ctx_orig_yarn}, else
 * {@code rope_factors_short}. Graph-specific deviations are listed in {@link #forLayer(int, int, int)}.
 * Immutable.
 */
public final class RopeResolver {

    /** {@code freq_base} of the plain {@code ggml_rope} call (ggml.c:4355). */
    private static final float GGML_ROPE_FREQ_BASE = 10000.0f;

    /** Graphs that call {@code model.get_rope_factors} (src/models/*.cpp, including graph aliases in models.h). */
    private static final Set<Arch> FACTOR_ARCHS = Collections.unmodifiableSet(EnumSet.of(
            Arch.LLAMA, Arch.LLAMA_EMBED, Arch.LLAMA4, Arch.ARCEE, Arch.APERTUS, Arch.DECI, Arch.DEEPSEEK,
            Arch.EXAONE, Arch.EXAONE4, Arch.EXAONE_MOE, Arch.COHERE2, Arch.COHERE2MOE, Arch.GRANITE,
            Arch.GRANITE_MOE, Arch.MINICPM, Arch.GRANITE_HYBRID, Arch.GRANITE_SWA, Arch.GRANITE_SWITCH,
            Arch.MINICPM3, Arch.HY_V3, Arch.HUNYUAN_MOE, Arch.HUNYUAN_VL, Arch.HUNYUAN_DENSE, Arch.MISTRAL3,
            Arch.NANBEIGE, Arch.PHI3, Arch.PHIMOE, Arch.BAILINGMOE, Arch.EAGLE3, Arch.STEP35));

    /** Graphs that read {@code layers[il].rope_freqs} on full-attention layers only (gemma4.cpp:209-213). */
    private static final Set<Arch> FULL_LAYER_FREQS_ARCHS = Collections.unmodifiableSet(EnumSet.of(
            Arch.GEMMA4, Arch.GEMMA4_ASSISTANT));

    /** Graphs that use {@code get_rope_freq_base} and {@code get_rope_freq_scale} per layer. */
    private static final Set<Arch> PER_LAYER_BASE_SCALE = Collections.unmodifiableSet(EnumSet.of(
            Arch.AFMOE, Arch.BEAM, Arch.GEMMA2, Arch.GEMMA3, Arch.GEMMA3N, Arch.GEMMA4, Arch.GEMMA4_ASSISTANT,
            Arch.GEMMA_EMBEDDING, Arch.GEMMA_EMBEDDING2, Arch.KOLIBRI1, Arch.LAGUNA, Arch.LLAMA4, Arch.MAPLE,
            Arch.MAION_CODER, Arch.MELLUM, Arch.MIMO2, Arch.MODERN_BERT, Arch.MUSE_GLIMMER, Arch.MUSE_SPARK,
            Arch.OPENAI_MOE, Arch.PLAMO3, Arch.SMALLTHINKER, Arch.SPARK2_5, Arch.STEP35));

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final int nCtxSeq;
    private final Arch arch;
    private final float freqBase;
    private final float freqScale;
    private final float extFactor;
    private final float attnFactor;
    private final float betaFast;
    private final float betaSlow;
    private final int nCtxOrig;

    /**
     * @param nCtxSeq per-sequence context length of the session ({@code cparams.n_ctx_seq}); selects the long or
     *                short frequency factors
     * @throws IllegalArgumentException for a non-positive context length
     */
    public RopeResolver(ModelHparams hp, TensorCatalog tc, int nCtxSeq) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        if (nCtxSeq <= 0) {
            throw new IllegalArgumentException("per-sequence context length must be positive, got " + nCtxSeq);
        }
        this.nCtxSeq = nCtxSeq;
        this.arch = hp.arch();
        RopeScalingType scaling = hp.ropeScalingType();
        this.freqBase = hp.ropeFreqBase();
        this.freqScale = scaling == RopeScalingType.NONE ? 1.0f : hp.ropeFreqScale();
        this.extFactor = hp.yarnExtFactor() < 0.0f ? RopeMath.defaultExtFactor(scaling) : hp.yarnExtFactor();
        this.attnFactor = RopeMath.yarnAttnFactor(hp.yarnAttnFactor(), extFactor, freqScale, hp.yarnLogMul(),
                arch == Arch.DEEPSEEK2, hp.ropeAttnFactor());
        this.betaFast = hp.yarnBetaFast();
        this.betaSlow = hp.yarnBetaSlow();
        this.nCtxOrig = hp.nCtxOrigYarn() != 0 ? hp.nCtxOrigYarn() : hp.nCtxTrain();
    }

    /** Context {@code rope_freq_base}. */
    public float freqBase() { return freqBase; }
    /** Context {@code rope_freq_scale} (1 for scaling type NONE). */
    public float freqScale() { return freqScale; }
    /** Resolved {@code yarn_ext_factor}. */
    public float extFactor() { return extFactor; }
    /** Effective {@code yarn_attn_factor} passed to the rope calls. */
    public float attnFactor() { return attnFactor; }
    public float betaFast() { return betaFast; }
    public float betaSlow() { return betaSlow; }
    /** {@code cparams.n_ctx_orig_yarn}. */
    public int nCtxOrig() { return nCtxOrig; }

    /**
     * True when the layer's graph rotates Q and K: the model has a RoPE type and the layer is not a NoPE layer.
     * Beyond {@link ModelHparams#hasRope(int)} this applies the graph rules the hparams do not record: the plain
     * BERT graph never rotates (models/bert.cpp:131-138 rotate only nomic-bert, nomic-bert-moe and
     * jina-bert-v3), starcoder uses learned positions (models/starcoder.cpp), and cohere2 rotates its SWA layers
     * only (models/cohere2.cpp:89).
     */
    public boolean appliesRope(int il) {
        if (hp.ropeType() == RopeType.NONE || !hp.hasRope(il)) {
            return false;
        }
        switch (arch) {
            case BERT:
            case STARCODER:
                return false;
            case COHERE2:
                return hp.isSwa(il);
            default:
                return true;
        }
    }

    /**
     * Rotary dims of an attention layer: {@code hparams.n_rot(il)}, except sensenova_u1, whose text RoPE rotates
     * the temporal half of each head (models/sensenova-u1.cpp:110-115), and deepseek2-ocr, whose MHA graph rotates
     * the whole {@code n_embd / n_head} head (models/deepseek2.cpp:467-480); 0 when the layer applies no RoPE.
     */
    public int nRot(int il) {
        if (!appliesRope(il)) {
            return 0;
        }
        switch (arch) {
            case SENSENOVA_U1:
                return hp.headK(il) / 2;
            case DEEPSEEK2OCR:
            case COGVLM:
                return hp.headK(il);
            default:
                return hp.nRot(il);
        }
    }

    /**
     * RoPE of the attention layer {@code il}, rotating {@link #nRot(int)} dims from offset 0.
     *
     * @return the spec, or null when the layer applies no RoPE
     * @throws IllegalArgumentException when the resolved parameters are inconsistent
     */
    public RopeSpec forLayer(int il) {
        if (!appliesRope(il)) {
            return null;
        }
        return forLayer(il, nRot(il), 0);
    }

    /**
     * RoPE of layer {@code il} over {@code nDims} dims starting at {@code nOffs} in each head, for mixers whose
     * rotary part is not the attention head prefix (MLA {@code qk_rope_head_dim}, {@code [nope | rope]} heads).
     * Graph deviations from the context values:
     * <ul>
     *   <li>olmo2 SWA layers: {@code freq_scale 1, ext_factor 0, attn_factor 1} with the full base
     *       (models/olmo2.cpp:117-131);</li>
     *   <li>laguna SWA layers: SWA base/scale, {@code ext 0, attn 1, beta 0/0, n_ctx_orig = n_ctx_train}
     *       (models/laguna.cpp:182-194);</li>
     *   <li>mellum SWA layers: {@code ext 0, attn 1, beta 32/1} (models/mellum.cpp:156-161);</li>
     *   <li>dots3note: per-layer base with the context scale (models/dots3note.cpp:188);</li>
     *   <li>sensenova_u1: NEOX over the temporal half of the head with {@code freq_scale 1, ext 0, attn 1,
     *       beta 32/1}; the two spatial quarters rotate by the height/width positions, which are 0 for text, so
     *       they are the identity (models/sensenova-u1.cpp:108-116);</li>
     *   <li>deepseek2-ocr: NEOX, {@code n_ctx_orig 0, freq_scale 1, ext 0, attn 1, beta 0/0}
     *       (models/deepseek2.cpp OCR branch);</li>
     *   <li>cogvlm: plain {@code ggml_rope} over the whole head, so {@code base 10000, freq_scale 1, ext 0, attn 1,
     *       beta 0/0, n_ctx_orig 0} whatever the file sets (models/cogvlm.cpp:116-117, ggml.c:4348-4357).</li>
     * </ul>
     *
     * @throws IllegalStateException    when the model has no RoPE type
     * @throws IllegalArgumentException when the dims or resolved parameters are inconsistent
     */
    public RopeSpec forLayer(int il, int nDims, int nOffs) {
        if (hp.ropeType() == RopeType.NONE) {
            throw new IllegalStateException(arch.ggufName() + ": the model has no rotary embedding (layer " + il + ")");
        }
        boolean swa = hp.isSwa(il);
        RopeType type = hp.ropeType();
        float base = freqBase;
        float scale = freqScale;
        float ext = extFactor;
        float attn = attnFactor;
        float bFast = betaFast;
        float bSlow = betaSlow;
        int ctxOrig = nCtxOrig;
        if (PER_LAYER_BASE_SCALE.contains(arch) && swa) {
            base = hp.ropeFreqBaseSwa();
            scale = hp.ropeFreqScaleSwa();
        }
        switch (arch) {
            case OLMO2:
                if (swa) {
                    scale = 1.0f;
                    ext = 0.0f;
                    attn = 1.0f;
                }
                break;
            case LAGUNA:
                if (swa) {
                    ext = 0.0f;
                    attn = 1.0f;
                    bFast = 0.0f;
                    bSlow = 0.0f;
                    ctxOrig = hp.nCtxTrain();
                }
                break;
            case MELLUM:
                if (swa) {
                    ext = 0.0f;
                    attn = 1.0f;
                    bFast = 32.0f;
                    bSlow = 1.0f;
                }
                break;
            case DOTS3NOTE:
                if (swa) {
                    base = hp.ropeFreqBaseSwa();
                }
                break;
            case SENSENOVA_U1:
                type = RopeType.NEOX;
                scale = 1.0f;
                ext = 0.0f;
                attn = 1.0f;
                bFast = 32.0f;
                bSlow = 1.0f;
                break;
            case DEEPSEEK2OCR:
                type = RopeType.NEOX;
                scale = 1.0f;
                ext = 0.0f;
                attn = 1.0f;
                bFast = 0.0f;
                bSlow = 0.0f;
                ctxOrig = 0;
                break;
            case COGVLM:
                base = GGML_ROPE_FREQ_BASE;
                scale = 1.0f;
                ext = 0.0f;
                attn = 1.0f;
                bFast = 0.0f;
                bSlow = 0.0f;
                ctxOrig = 0;
                break;
            default:
                break;
        }
        RopeSpec.Builder b = RopeSpec.builder(type)
                .nDims(nDims)
                .nOffs(nOffs)
                .freqBase(base)
                .freqScale(scale)
                .yarn(ext, attn, bFast, bSlow)
                .nCtxOrig(ctxOrig)
                .freqFactors(freqFactors(il));
        if (nDims > 0 && base > 0f) {
            b.thetaScale(RopeMath.thetaScale(base, nDims));
        }
        float[] corr = corrDims(nDims, ctxOrig, base, bFast, bSlow);
        b.corrDims(corr[0], corr[1]);
        if (RopeMath.multiSection(type)) {
            b.sections(hp.ropeSections()).textPositions(hp.mropeTextPositions());
        }
        return b.build();
    }

    /**
     * Frequency factors of layer {@code il} ({@code get_rope_factors}, llama-model.cpp:2763-2776), or null when the
     * graph passes none. step35 skips them on SWA layers (models/step35.cpp:235); gemma4 and gemma4-assistant read
     * {@code rope_freqs} on full-attention layers only; dflash always passes {@code rope_freqs}
     * (models/dflash.cpp, {@code layers[0].rope_freqs}).
     */
    public TensorRef freqFactors(int il) {
        TensorRef freqs = tc.get(TensorNames.weight(TensorNames.ROPE_FREQS));
        if (FULL_LAYER_FREQS_ARCHS.contains(arch)) {
            return hp.isSwa(il) ? null : freqs;
        }
        if (arch == Arch.DFLASH) {
            return freqs;
        }
        if (!FACTOR_ARCHS.contains(arch) || (arch == Arch.STEP35 && hp.isSwa(il))) {
            return null;
        }
        if (freqs != null) {
            return freqs;
        }
        String name = nCtxSeq > hp.nCtxOrigYarn() ? TensorNames.ROPE_FACTORS_LONG : TensorNames.ROPE_FACTORS_SHORT;
        return tc.get(TensorNames.weight(name));
    }

    /**
     * {@code ggml_rope_yarn_corr_dims} (ggml/src/ggml.c) as the rope op computes it from its own arguments. With
     * {@code n_ctx_orig <= 0} the range is unused (such calls pass {@code ext_factor 0}) and is returned as 0/0.
     */
    private static float[] corrDims(int nDims, int nCtxOrig, float base, float betaFast, float betaSlow) {
        if (nDims <= 0 || nCtxOrig <= 0 || !(base > 0f)) {
            return new float[]{0f, 0f};
        }
        return RopeMath.corrDims(nDims, nCtxOrig, base, betaFast, betaSlow);
    }

    @Override
    public String toString() {
        return "RopeResolver{" + arch.ggufName() + ", base=" + freqBase + ", scale=" + freqScale + ", ext=" + extFactor
                + ", attn=" + attnFactor + ", nCtxOrig=" + nCtxOrig + ", nCtxSeq=" + nCtxSeq + '}';
    }
}
