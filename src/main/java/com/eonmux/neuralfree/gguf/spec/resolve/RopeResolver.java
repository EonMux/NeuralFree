package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.spec.FopeSpec;
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
 *   <li>{@code n_ctx_orig} is {@code n_ctx_orig_yarn}, which the loader defaults to {@code n_ctx_train};</li>
 *   <li>scaling type DYNAMIC keeps {@code freq_scale = 1} and replaces the base of each layer with the NTK-aware
 *       base {@link RopeMath#dynamicAlphaBase} or {@link RopeMath#dynamicNtkBase} of the layer's rotary width.</li>
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

    /** Graphs that read the learned model-level {@code rope_freqs} factors (lfm3.cpp:113, mistral-large4.cpp:90). */
    private static final Set<Arch> LEARNED_FREQS_ARCHS = Collections.unmodifiableSet(EnumSet.of(
            Arch.LFM3, Arch.MISTRAL_LARGE4));

    /** Graphs that use {@code get_rope_freq_base} and {@code get_rope_freq_scale} per layer. */
    private static final Set<Arch> PER_LAYER_BASE_SCALE = Collections.unmodifiableSet(EnumSet.of(
            Arch.AFMOE, Arch.BEAM, Arch.GEMMA2, Arch.GEMMA3, Arch.GEMMA3N, Arch.GEMMA4, Arch.GEMMA4_ASSISTANT,
            Arch.GEMMA_EMBEDDING, Arch.GEMMA_EMBEDDING2, Arch.KOLIBRI1, Arch.LAGUNA, Arch.LFM3, Arch.LLAMA4,
            Arch.MAPLE, Arch.MAION_CODER, Arch.MELLUM, Arch.MIMO2, Arch.MODERN_BERT, Arch.MUSE_GLIMMER,
            Arch.MUSE_SPARK, Arch.OPENAI_MOE, Arch.PLAMO3, Arch.RNJ1, Arch.DORY, Arch.SMALLTHINKER,
            Arch.SPARK2_5,
            Arch.STEP35));

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final int nCtxSeq;
    private final Arch arch;
    private final RopeScalingType scaling;
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
        this.scaling = scaling;
        this.freqBase = hp.ropeFreqBase();
        this.freqScale = scaling == RopeScalingType.NONE || scaling == RopeScalingType.DYNAMIC
                ? 1.0f : hp.ropeFreqScale();
        this.extFactor = hp.yarnExtFactor() < 0.0f ? RopeMath.defaultExtFactor(scaling) : hp.yarnExtFactor();
        this.attnFactor = RopeMath.yarnAttnFactor(hp.yarnAttnFactor(), extFactor, freqScale, hp.yarnLogMul(),
                arch == Arch.DEEPSEEK2 || arch == Arch.AXK1, hp.ropeAttnFactor());
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
     * jina-bert-v3), starcoder uses learned positions (models/starcoder.cpp), cohere2 rotates its SWA layers
     * only (models/cohere2.cpp:89), and limite rotates its SWA layers only, the full-attention layers use no
     * positional encoding (models/limite.cpp:163-170).
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
            case LIMITE:
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
     * Rotary embedding of the attention layer {@code il}, rotating {@link #nRot(int)} dims from offset 0.
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
     * Rotary embedding of a lightning-attention layer over {@code nDims} dims of each head, the {@code is_neox_style}
     * rotation vLLM builds for the linear-attention head ({@code get_rope(head_dim, is_neox_style=True, ...)}): NEOX
     * with the context's frequency parameters and no learned frequency factors. The architecture's per-layer graph
     * deviations do not apply.
     *
     * @return the spec, or null when {@code nDims == 0}
     * @throws IllegalArgumentException when the dims or resolved parameters are inconsistent
     */
    public RopeSpec linearAttention(int il, int nDims) {
        if (nDims <= 0) {
            return null;
        }
        return build(il, nDims, 0, RopeType.NEOX, true);
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
        return build(il, nDims, nOffs, hp.ropeType(), false);
    }

    /**
     * The rotary parameters of layer {@code il} at one of the graph's {@code ggml_rope_ext} calls.
     *
     * @param type      the rotation layout
     * @param plain     true for a rotation outside the attention graphs: the per-layer SWA and architecture
     *                  deviations do not apply, the parameters are the context's
     */
    private RopeSpec build(int il, int nDims, int nOffs, RopeType type, boolean plain) {
        boolean swa = !plain && hp.isSwa(il);
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
        if (!plain) {
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
        }
        if (scaling == RopeScalingType.DYNAMIC) {
            base = dynamicBase(base, nDims);
        }
        RopeSpec.Builder b = RopeSpec.builder(type)
                .nDims(nDims)
                .nOffs(nOffs)
                .freqBase(base)
                .freqScale(scale)
                .yarn(ext, attn, bFast, bSlow)
                .nCtxOrig(ctxOrig)
                .freqFactors(plain ? null : freqFactors(il));
        if (scaling == RopeScalingType.NTK) {
            b.computedFreqFactors(RopeMath.ntkFactors(hp.ropeScalingFactor(), hp.ropeScalingMixedB(), nDims));
        }
        if (hp.ropeFope()) {
            b.fope(fope(il, type, nDims, base));
        }
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
     * The NTK-scaled base of a layer when the scaling type is {@link RopeScalingType#DYNAMIC}: the alpha form when
     * {@code rope.scaling.alpha} is positive (vLLM tries {@code alpha} before {@code factor}), else the factor form
     * over the model's context length and its trained length.
     *
     * @throws IllegalArgumentException when the rotary width is 2 or less or the factor form lacks positive lengths
     */
    private float dynamicBase(float base, int nDims) {
        if (hp.ropeScalingAlpha() > 0f) {
            return RopeMath.dynamicAlphaBase(base, hp.ropeScalingAlpha(), nDims);
        }
        final int trained = hp.nCtxOrigYarn() != 0 ? hp.nCtxOrigYarn() : hp.nCtxTrain();
        return RopeMath.dynamicNtkBase(base, hp.ropeScalingFactor(), hp.nCtxTrain(), trained, nDims);
    }

    /**
     * The Fourier position embedding parameters of layer {@code il}: the coefficient tensors and the frequencies they
     * project, the latter selected from the layer's base and rotary width over the model context length
     * ({@link RopeMath#fopeInvFreq}, vLLM {@code FourierRotaryEmbedding._compute_inv_freq}).
     *
     * @param type the layer's rotation layout, which must be NEOX (the whole head rotates in halves)
     * @throws IllegalStateException    when the model's coefficient tensors are absent from the file
     * @throws java.util.NoSuchElementException when the catalog holds no such tensor
     * @throws IllegalArgumentException when the rotation is not NEOX or the frequency selection is invalid
     */
    private FopeSpec fope(int il, RopeType type, int nDims, float base) {
        if (type != RopeType.NEOX) {
            throw new IllegalArgumentException(arch.ggufName() + ": the Fourier position embedding is a NEOX"
                    + " rotation over the whole head, not " + type);
        }
        TensorRef cos = tc.get(TensorNames.weight(TensorNames.ROPE_COS_COEF));
        TensorRef sin = tc.get(TensorNames.weight(TensorNames.ROPE_SIN_COEF));
        if (cos == null || sin == null) {
            throw new IllegalStateException(arch.ggufName() + ": the Fourier position embedding needs the model"
                    + " tensors " + TensorNames.weight(TensorNames.ROPE_COS_COEF) + " and "
                    + TensorNames.weight(TensorNames.ROPE_SIN_COEF));
        }
        final int context = hp.nCtxTrain() > 0 ? hp.nCtxTrain() : nCtxOrig;
        final float[] inv = RopeMath.fopeInvFreq(base, nDims, hp.ropeFopeNumInvFreq(), context);
        return FopeSpec.of(hp.nHeadKv(il), hp.ropeFopeSepHead(), inv, cos, sin);
    }

    /**
     * Frequency factors of layer {@code il} ({@code get_rope_factors}, llama-model.cpp:2763-2776), or null when the
     * graph passes none. lfm3 and mistral-large4 read the learned model-level {@code rope_freqs} (lfm3 skips them
     * on SWA layers, lfm3.cpp:113); step35 skips them on SWA layers (models/step35.cpp:235); gemma4 and
     * gemma4-assistant read {@code rope_freqs} on full-attention layers only; dflash always passes
     * {@code rope_freqs} (models/dflash.cpp, {@code layers[0].rope_freqs}).
     */
    public TensorRef freqFactors(int il) {
        TensorRef freqs = tc.get(TensorNames.weight(TensorNames.ROPE_FREQS));
        if (LEARNED_FREQS_ARCHS.contains(arch)) {
            // lfm3 skips the learned factors on its SWA layers (lfm3.cpp:113); mistral-large4 passes them to
            // the MLA and indexer rotations of every roped layer.
            return arch == Arch.LFM3 && hp.isSwa(il) ? null : freqs;
        }
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
