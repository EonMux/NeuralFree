package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.norm.AdaNormMath;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.NormResolver;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Runs one virtual layer on the CPU in this fixed order, the twin of {@code GpuLayerExecutor}:
 * <ol>
 *   <li>{@code extras.beforeBlock};</li>
 *   <li>{@code attn_norm} unless POST_ONLY (also computed for a mixer-less layer whose FFN reuses it);</li>
 *   <li>the mixer unless {@link MixerKind#NONE}. T5 decoder layers run the self-attention, its residual,
 *       {@code cross_attn_norm} and then the cross-attention (models/t5.cpp:134-176); falcon-h1 adds the attention
 *       branch to the recurrent branch on the same normed input (models/falcon-h1.cpp:135-166);</li>
 *   <li>branch norms: grok's {@code attn_output_norm} (models/grok.cpp:143-146), then {@code post_attention_norm}
 *       under SANDWICH or POST_ONLY;</li>
 *   <li>{@code extras.afterMixer}; any extra returning true has consumed the residual;</li>
 *   <li>the residual by {@link ResidualKind}: STANDARD {@code x += b}; SCALED {@code x += s*b}; NORMED_INPUT_SCALED
 *       {@code x = b + s*attn_norm(x)} (minimax-01.cpp:428-458); PARALLEL defers the add; none when
 *       {@code mixerResidual} is false; HYPER_CONNECTIONS must have been consumed in step 5. BERT-graph layers apply
 *       their post-LN {@code attn_output_norm} (and jina's {@code attn_norm_2} over the re-added layer input) here
 *       (models/bert.cpp:157-165);</li>
 *   <li>the FFN input: its norm tensor as {@link NormResolver#ffnNormTensor} names it ({@code ffn_norm},
 *       {@code post_attention_norm} under PRE_FFN, {@code attn_norm_2} for RWKV, dbrx's {@code attn_output_norm}),
 *       weightless for weightless graphs, or the step-2 {@code attn_norm} output when it is shared. falcon-40b's
 *       parallel block feeds {@code attn_norm_2(x)} to the attention and {@code attn_norm(x)} to the FFN
 *       (models/falcon.cpp:70-86, 125). Voxtral's adaptive {@code ffn_norm} then scales its output by
 *       {@code 1 + ada} ({@link AdaNormMath});</li>
 *   <li>the FFN unless {@link FfnKind#NONE};</li>
 *   <li>the post-FFN norm under SANDWICH or POST_ONLY ({@link NormResolver#postFfnNormTensor});</li>
 *   <li>{@code extras.afterFfn};</li>
 *   <li>the residual as in step 6, PARALLEL adding {@code ffn + attn} to the stream (models/falcon.cpp:134-135,
 *       gptneox.cpp); BERT-graph {@code layer_output_norm} after it (models/bert.cpp:212-215);</li>
 *   <li>{@code discardLayerOutput} restores the layer input (models/deci.cpp:145-148);</li>
 *   <li>{@code extras.afterBlock}; then gemma4's {@code layer_output_scale} (models/gemma4.cpp:400-402) and the
 *       block output scale (models/rwkv6.cpp:164-166).</li>
 * </ol>
 *
 * <p>The residual stream {@code x} holds {@code n} rows of the layer's stream width at the layer boundaries. With
 * hyper-connections it holds the collapsed {@code n_embd} rows between {@code beforeBlock} and the return of the
 * FFN-side residual hook, as the {@code hc} extra's contract states; every other width the executor uses is
 * {@code n_embd}. Mixers and FFN blocks read their {@code in} array without writing it, so the step-2 norm output
 * stays valid for a shared FFN input and the normed residual. Norm biases are applied when their tensors exist.</p>
 *
 * <p>Not thread-safe; one forward pass at a time. A step allocates nothing; row-wise work is sharded with
 * {@link CpuParallel}.</p>
 */
public final class CpuLayerExecutor implements AutoCloseable {

    private static final String TEMP_LAYER_IN = "CpuLayerExecutor.layerIn";
    private static final String TEMP_MIXER_IN = "CpuLayerExecutor.mixerIn";
    private static final String TEMP_AUX = "CpuLayerExecutor.auxBranch";
    private static final String TEMP_FFN = "CpuLayerExecutor.ffnBranch";
    private static final String TEMP_ADA_HIDDEN = "CpuLayerExecutor.adaHidden";
    private static final String TEMP_ADA = "CpuLayerExecutor.ada";
    /** Elements per parallel task of the row-wise residual mixes. */
    private static final int ELEMENT_GRAIN = 1 << 15;

    private final CpuForwardContext ctx;
    private final LayerSpec layer;
    private final NormSpec norm;
    private final ResidualSpec res;
    private final int width;
    private final int streamWidth;
    private final boolean parallel;
    private final boolean ffnReadsAttnNorm;

    private final CpuMixer mixer;
    private final CpuMixer selfAttention;
    private final CpuMixer parallelAttention;
    private final CpuFfnBlock ffn;
    private final List<CpuLayerExtra> extras;

    private final Norm attnNorm;
    private final Norm mixerNorm2;
    private final Norm crossNorm;
    private final Norm attnOutNorm;
    private final Norm postAttnNorm;
    private final Norm postLnAttn;
    private final Norm postLnAttn2;
    private final Norm ffnNorm;
    private final Norm postFfnNorm;
    private final Norm postLnLayer;
    private final float[] layerOutScale;
    /** Adaptive conditioning of {@link #ffnNorm}'s output, or null. */
    private final AdaNormMath.CpuLayer ada;
    private final float[] adaCond;
    private final float[] adaHidden;
    private final float[] adaVector;

    private final float[] normed;
    private final float[] branch;
    private final float[] layerIn;
    private final float[] mixerIn;
    private final float[] aux;
    private final float[] ffnBranch;
    private boolean closed;

    /**
     * Creates and binds the layer's mixer(s), FFN block and extras from {@code registry} and resolves its norm
     * weights.
     *
     * @throws UnsupportedOperationException when the registry serves no block for the layer's kinds or extras
     * @throws MissingTensorException        when a norm tensor the layer's spec places is absent
     * @throws IllegalStateException         when the residual wiring needs an extra the layer does not list
     * @throws IllegalArgumentException      when a norm or scale tensor disagrees with the layer width
     */
    public CpuLayerExecutor(CpuBlockRegistry registry, CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(registry, "registry");
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.norm = layer.norm();
        this.res = layer.residual();
        this.streamWidth = res.streamWidth();
        if (res.hcMult() < 1 || streamWidth % res.hcMult() != 0) {
            throw new IllegalArgumentException(describe() + ": stream width " + streamWidth
                    + " is not a multiple of the hyper-connection count " + res.hcMult());
        }
        this.width = streamWidth / res.hcMult();
        this.parallel = res.kind() == ResidualKind.PARALLEL;
        checkExtras();

        boolean hasMixer = layer.mixerKind() != MixerKind.NONE;
        boolean hasFfn = layer.ffnKind() != FfnKind.NONE;
        PostNormRole role = norm.postRole();
        boolean postOnly = role == PostNormRole.POST_ONLY;
        boolean branchPostNorms = role == PostNormRole.SANDWICH || postOnly;
        boolean falconNorm2 = parallel && res.ffnUsesAttnNorm2();
        boolean sharedFfnInput = hasFfn && (falconNorm2 || norm.sharedAttnNormForFfn());
        boolean postLn = !norm.attnNorm() && !postOnly;

        this.attnNorm = norm.attnNorm() && !postOnly && (hasMixer || sharedFfnInput)
                ? required(TensorNames.ATTN_NORM).grouped(describe(), width, norm.groups()) : null;
        this.ffnReadsAttnNorm = sharedFfnInput && attnNorm != null;
        this.mixerNorm2 = falconNorm2 && hasMixer ? required(TensorNames.ATTN_NORM_2) : null;
        this.crossNorm = layer.mixerKind() == MixerKind.CROSS_ATTENTION_T5
                ? required(TensorNames.DEC_CROSS_ATTN_NORM) : null;
        this.attnOutNorm = hasMixer && norm.attnNorm() && norm.attnOutputNorm()
                ? required(TensorNames.ATTN_OUT_NORM) : null;
        this.postAttnNorm = hasMixer && branchPostNorms && norm.postAttnNorm()
                ? postAttnNorm() : null;
        this.postLnAttn = hasMixer && postLn && norm.attnOutputNorm() ? required(TensorNames.ATTN_OUT_NORM) : null;
        this.postLnAttn2 = postLnAttn != null ? optional(TensorNames.ATTN_NORM_2) : null;
        Norm ffnIn = hasFfn && !sharedFfnInput ? ffnInputNorm() : null;
        this.ffnNorm = ffnIn != null ? ffnIn.grouped(describe(), width, norm.groups()) : null;
        this.postFfnNorm = hasFfn && branchPostNorms && norm.postFfnNorm() ? postFfnNorm() : null;
        this.postLnLayer = postLn && norm.layerOutputNorm() ? required(TensorNames.LAYER_OUT_NORM) : null;
        this.layerOutScale = norm.layerOutputScale() != null ? scaleVector(norm.layerOutputScale()) : null;
        this.ada = norm.ffnAda() ? adaLayer() : null;
        if (res.kind() == ResidualKind.NORMED_INPUT_SCALED
                && ((hasMixer && attnNorm == null) || (hasFfn && ffnNorm == null))) {
            throw new IllegalStateException(describe() + ": a normed-input residual needs attn_norm and the FFN norm");
        }

        CpuScratch scratch = ctx.scratch();
        int ubatch = scratch.ubatch();
        this.normed = scratch.slot(CpuScratch.NORMED);
        this.branch = scratch.slot(CpuScratch.BRANCH);
        this.layerIn = res.discardLayerOutput() || postLnAttn2 != null
                ? scratch.temp(TEMP_LAYER_IN, rows(ubatch, streamWidth)) : null;
        this.mixerIn = mixerNorm2 != null ? scratch.temp(TEMP_MIXER_IN, rows(ubatch, width)) : null;
        this.aux = layer.parallelAttention() ? scratch.temp(TEMP_AUX, rows(ubatch, width)) : null;
        this.ffnBranch = parallel && hasFfn ? scratch.temp(TEMP_FFN, rows(ubatch, width)) : null;
        this.adaCond = ada != null ? AdaNormMath.timeCondition(norm.ffnAdaDelay(), width) : null;
        this.adaHidden = ada != null ? scratch.temp(TEMP_ADA_HIDDEN, ada.hidden()) : null;
        this.adaVector = ada != null ? scratch.temp(TEMP_ADA, width) : null;

        Blocks b = new Blocks();
        try {
            if (layer.mixerKind() == MixerKind.CROSS_ATTENTION_T5) {
                LayerSpec self = attentionView(layer);
                b.selfAttention = registry.mixer(self);
                b.selfAttention.bind(ctx, self);
            }
            if (hasMixer) {
                b.mixer = registry.mixer(layer);
                b.mixer.bind(ctx, layer);
            }
            if (layer.parallelAttention()) {
                LayerSpec attn = attentionView(layer);
                b.parallelAttention = registry.mixer(attn);
                b.parallelAttention.bind(ctx, attn);
            }
            if (hasFfn) {
                b.ffn = registry.ffn(layer);
                b.ffn.bind(ctx, layer);
            }
            b.extras = registry.extras(layer);
            for (CpuLayerExtra e : b.extras) {
                e.bind(ctx, layer);
            }
        } catch (RuntimeException e) {
            b.closeInto(e);
            throw e;
        }
        this.mixer = b.mixer;
        this.selfAttention = b.selfAttention;
        this.parallelAttention = b.parallelAttention;
        this.ffn = b.ffn;
        this.extras = b.extras;
    }

    /** Blocks under construction, closed together when a later bind fails. */
    private static final class Blocks {
        CpuMixer mixer;
        CpuMixer selfAttention;
        CpuMixer parallelAttention;
        CpuFfnBlock ffn;
        List<CpuLayerExtra> extras = Collections.emptyList();

        void closeInto(RuntimeException sink) {
            List<AutoCloseable> all = new ArrayList<>(extras);
            all.add(ffn);
            all.add(parallelAttention);
            all.add(mixer);
            all.add(selfAttention);
            Closeables.closeAll(all, sink);
        }
    }

    /** The extras the residual wiring of this layer relies on. */
    private void checkExtras() {
        if (res.kind() == ResidualKind.HYPER_CONNECTIONS && !layer.hasExtra(LayerSpec.Extras.HC)) {
            throw new IllegalStateException(describe() + ": hyper-connection residual without the '"
                    + LayerSpec.Extras.HC + "' extra");
        }
        if (res.alphaBeta() && !layer.hasExtra(LayerSpec.Extras.RESID_ALPHA_BETA)) {
            throw new IllegalStateException(describe() + ": alpha/beta residual without the '"
                    + LayerSpec.Extras.RESID_ALPHA_BETA + "' extra");
        }
        if (res.embeddingSkip() && !layer.hasExtra(LayerSpec.Extras.EMBD_SKIP)) {
            throw new IllegalStateException(describe() + ": embedding skip without the '"
                    + LayerSpec.Extras.EMBD_SKIP + "' extra");
        }
        ModelSpec spec = ctx.spec();
        LayerSchedule schedule = spec.schedule();
        boolean trunk = layer.il() < spec.nLayer() && spec.layer(layer.il()) == layer;
        if (trunk && schedule.normAfterSlot(layer.il()) && !layer.hasExtra(LayerSpec.Extras.LOOP_NORM)) {
            throw new IllegalStateException(describe() + ": loop boundary without the '"
                    + LayerSpec.Extras.LOOP_NORM + "' extra");
        }
    }

    // ------------------------------------------------------------------ norm resolution

    /** A row norm with its weight and bias (both null for weightless norms). */
    private static final class Norm {
        final NormKind kind;
        final float eps;
        final float[] w;
        final float[] b;
        /** {@code norm(x * w)} instead of {@code norm(x) * w} (muse-spark post-FFN norm). */
        final boolean weightFirst;
        /** Equal row slices normalized on their own before the full-width weight; 1 for a plain row norm. */
        final int groups;

        Norm(NormKind kind, float eps, float[] w, float[] b) {
            this(kind, eps, w, b, false, 1);
        }

        private Norm(NormKind kind, float eps, float[] w, float[] b, boolean weightFirst, int groups) {
            this.kind = kind;
            this.eps = eps;
            this.w = w;
            this.b = b;
            this.weightFirst = weightFirst;
            this.groups = groups;
        }

        /** This norm with epsilon {@code e}. */
        Norm withEps(float e) {
            return new Norm(kind, e, w, b, weightFirst, groups);
        }

        /**
         * This norm over {@code g} equal slices of each {@code width} row, each normalized on its own, then
         * multiplied by the full-width weight (k2-horizon.cpp:124-135).
         *
         * @throws IllegalArgumentException for a weightless, biased or weight-first norm, or a width that is not a
         *                                  multiple of {@code g}
         */
        Norm grouped(String where, int width, int g) {
            if (g == 1) {
                return this;
            }
            if (w == null || b != null || weightFirst || width % g != 0) {
                throw new IllegalArgumentException(where + ": a norm over " + g + " groups needs a weight, no bias and"
                        + " a width divisible by the groups (width " + width + ")");
            }
            return new Norm(kind, eps, w, null, false, g);
        }

        /**
         * This norm with its weight applied before the normalization.
         *
         * @throws IllegalArgumentException for a weightless norm or one with a bias
         */
        Norm weightFirst(String where) {
            if (w == null || b != null) {
                throw new IllegalArgumentException(where + ": a norm applied to the weighted input needs a weight"
                        + " and no bias");
            }
            return new Norm(kind, eps, w, null, true, 1);
        }

        /** Normalizes {@code n} rows of {@code width} from {@code src} into {@code dst} (may be {@code src}). */
        void apply(float[] src, float[] dst, int n, int width) {
            if (groups > 1) {
                int slice = width / groups;
                NormMath.rowNorm(kind, eps, n * groups, slice, slice, groups, src, 0, w, null, dst, 0);
                return;
            }
            if (!weightFirst) {
                CpuOps.norm(kind, eps, src, 0, dst, 0, n, width, w, b);
                return;
            }
            for (int r = 0; r < n; r++) {
                int off = r * width;
                for (int c = 0; c < width; c++) {
                    dst[off + c] = src[off + c] * w[c];
                }
            }
            CpuOps.norm(kind, eps, dst, 0, dst, 0, n, width, null, null);
        }
    }

    /** The layer's norm {@code base}: weightless for weightless graphs, else its required weight and optional bias. */
    private Norm required(String base) {
        if (norm.weightless()) {
            return new Norm(norm.kind(), norm.eps(), null, null);
        }
        Norm n = optional(base);
        if (n == null) {
            CpuWeights weights = ctx.weights();
            throw new MissingTensorException(weights.archName(), layer.il(),
                    layer.tensorName(TensorNames.weight(base)));
        }
        return n;
    }

    /** The layer's norm {@code base} when its weight exists, else null. */
    private Norm optional(String base) {
        CpuWeights weights = ctx.weights();
        float[] w = weights.layerFloatsOrNull(layer, TensorNames.weight(base));
        if (w == null) {
            return null;
        }
        return new Norm(norm.kind(), norm.eps(), checkWidth(w, base),
                checkWidthOrNull(weights.layerFloatsOrNull(layer, TensorNames.bias(base)), base + " bias"));
    }

    /** A norm whose weight tensor was resolved by name, with the bias stored beside it. */
    private Norm fromRef(TensorRef ref) {
        CpuWeights weights = ctx.weights();
        float[] w = checkWidth(weights.floats(ref), ref.name());
        float[] b = null;
        String name = ref.name();
        String suffix = ".weight";
        if (name.endsWith(suffix)) {
            String biasName = name.substring(0, name.length() - suffix.length()) + ".bias";
            b = checkWidthOrNull(weights.floatsOrNull(biasName), biasName);
        }
        return new Norm(norm.kind(), norm.eps(), w, b);
    }

    /** The layer's ada projections and their delta sites; the conditioned {@code ffn_norm} must be applied. */
    private AdaNormMath.CpuLayer adaLayer() {
        if (ffnNorm == null) {
            throw new IllegalStateException(describe() + ": the ada norm conditions an ffn_norm the layer does not"
                    + " apply");
        }
        CpuWeights weights = ctx.weights();
        CpuMatVec mv = ctx.matVec();
        CpuTensorRef down = weights.get(norm.ffnAdaDown());
        CpuTensorRef up = weights.get(norm.ffnAdaUp());
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        return new AdaNormMath.CpuLayer(down, up,
                deltas.serve(layer, TensorNames.weight(TensorNames.FFN_NORM_ADA_DOWN), down, mv),
                deltas.serve(layer, TensorNames.weight(TensorNames.FFN_NORM_ADA_UP), up, mv), width);
    }

    /** True when {@link NormResolver} names this layer's FFN-side norm tensors (main-stack physical layers). */
    private boolean mainStack() {
        return TensorNames.STACK_BLK.equals(layer.stack())
                && layer.weightLayer() < ctx.spec().hparams().nLayerAll();
    }

    /** The FFN input norm (step 7) other than the shared {@code attn_norm} output; null for an un-normed input. */
    private Norm ffnInputNorm() {
        if (!mainStack()) {
            if (norm.ffnInputIsPostAttnNorm()) {
                return required(TensorNames.ATTN_POST_NORM);
            }
            if (norm.attnNorm2() && res.ffnUsesAttnNorm2()) {
                return required(TensorNames.ATTN_NORM_2);
            }
            return norm.ffnNorm() ? required(TensorNames.FFN_NORM) : null;
        }
        TensorRef ref = new NormResolver(ctx.spec().hparams(), ctx.spec().tensors()).ffnNormTensor(layer.weightLayer());
        if (ref != null) {
            return fromRef(ref);
        }
        if (norm.weightless() && norm.ffnNorm()) {
            return new Norm(norm.kind(), norm.eps(), null, null);
        }
        if (norm.ffnNorm() || norm.ffnInputIsPostAttnNorm() || norm.attnNorm2()) {
            throw new MissingTensorException(ctx.weights().archName(), layer.il(),
                    layer.tensorName(TensorNames.weight(TensorNames.FFN_NORM)));
        }
        return null;
    }

    /**
     * The norm on the mixer branch output with the post-norm epsilon: the tensor {@link NormResolver#postAttnNormTensor}
     * names for main-stack layers, {@code post_attention_norm} otherwise; weightless for muse-spark and weightless
     * graphs.
     */
    private Norm postAttnNorm() {
        if (norm.postAttnWeightless() || norm.weightless()) {
            return new Norm(norm.kind(), norm.postEps(), null, null);
        }
        if (!mainStack()) {
            return required(TensorNames.ATTN_POST_NORM).withEps(norm.postEps());
        }
        TensorRef ref = new NormResolver(ctx.spec().hparams(), ctx.spec().tensors())
                .postAttnNormTensor(layer.weightLayer());
        if (ref == null) {
            throw new MissingTensorException(ctx.weights().archName(), layer.il(),
                    layer.tensorName(TensorNames.weight(TensorNames.ATTN_POST_NORM)));
        }
        return fromRef(ref).withEps(norm.postEps());
    }

    /** The norm on the FFN branch output (step 9), with the post-norm epsilon. */
    private Norm postFfnNorm() {
        Norm n = postFfnNormTensor().withEps(norm.postEps());
        return norm.postFfnWeightFirst() ? n.weightFirst(describe()) : n;
    }

    private Norm postFfnNormTensor() {
        if (!mainStack()) {
            return required(TensorNames.FFN_POST_NORM);
        }
        TensorRef ref = new NormResolver(ctx.spec().hparams(), ctx.spec().tensors())
                .postFfnNormTensor(layer.weightLayer());
        if (ref == null) {
            if (norm.weightless()) {
                return new Norm(norm.kind(), norm.eps(), null, null);
            }
            throw new MissingTensorException(ctx.weights().archName(), layer.il(),
                    layer.tensorName(TensorNames.weight(TensorNames.FFN_POST_NORM)));
        }
        return fromRef(ref);
    }

    private float[] scaleVector(TensorRef ref) {
        float[] s = ctx.weights().floats(ref);
        if (s.length != 1 && s.length != streamWidth) {
            throw new IllegalArgumentException(describe() + ": " + ref.name() + " has " + s.length
                    + " values, expected 1 or " + streamWidth);
        }
        return s;
    }

    private float[] checkWidth(float[] v, String what) {
        if (v.length != width) {
            throw new IllegalArgumentException(describe() + ": " + what + " has " + v.length + " values, expected "
                    + width);
        }
        return v;
    }

    private float[] checkWidthOrNull(float[] v, String what) {
        return v == null ? null : checkWidth(v, what);
    }

    /**
     * The layer as a plain attention layer: the self-attention of a T5 decoder layer and the attention branch of a
     * falcon-h1 layer, built by the {@link MixerKind#ATTENTION} factory over the same weights and memory slot.
     */
    static LayerSpec attentionView(LayerSpec l) {
        AttentionSpec a = l.attention();
        if (a == null) {
            throw new IllegalStateException("layer " + l.stack() + "." + l.il() + " has no attention branch");
        }
        return LayerSpec.builder(l.il(), MixerKind.ATTENTION, l.ffnKind())
                .weightLayer(l.weightLayer())
                .stack(l.stack())
                .norm(l.norm())
                .residual(l.residual())
                .attention(a)
                .ffn(l.ffn())
                .moe(l.moe())
                .rope(l.rope())
                .build();
    }

    // ------------------------------------------------------------------ accessors

    public LayerSpec layer() {
        return layer;
    }

    /** Values per token of the residual stream at the layer boundaries. */
    public int streamWidth() {
        return streamWidth;
    }

    /** Values per token inside the block ({@code n_embd}). */
    public int width() {
        return width;
    }

    // ------------------------------------------------------------------ step

    /**
     * Runs the layer over {@code n} tokens at positions {@code pos0 ..}, updating the residual stream {@code x}
     * ({@code n} rows of {@link #streamWidth()} from index 0) in place. A single token with memory runs the
     * mixers' decode path; anything else their prefill path.
     *
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, ubatch]} or {@code x} is too short
     * @throws IllegalStateException    when a hyper-connection residual was not consumed by its extra
     */
    public void forward(int pos0, int n, float[] x) {
        Objects.requireNonNull(x, "x");
        checkOpen();
        int ubatch = ctx.scratch().ubatch();
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException(describe() + ": step of " + n + " tokens outside [1, " + ubatch + "]");
        }
        if ((long) n * streamWidth > x.length) {
            throw new IllegalArgumentException(describe() + ": residual stream holds " + x.length + " floats, need "
                    + (long) n * streamWidth);
        }
        if (layerIn != null) {
            System.arraycopy(x, 0, layerIn, 0, n * streamWidth);
        }
        for (CpuLayerExtra e : extras) {
            e.beforeBlock(pos0, n, x);
        }
        if (attnNorm != null) {
            attnNorm.apply(x, normed, n, width);
        }
        boolean attnPending = mixer != null && mixerBlock(pos0, n, x);
        if (ffn != null) {
            ffnBlock(pos0, n, x, attnPending);
        } else if (attnPending) {
            mix(MixMode.ADD, 0f, n, x, branch, x);
        }
        if (postLnLayer != null) {
            postLnLayer.apply(x, x, n, width);
        }
        if (res.discardLayerOutput()) {
            System.arraycopy(layerIn, 0, x, 0, n * streamWidth);
        }
        for (CpuLayerExtra e : extras) {
            e.afterBlock(pos0, n, x);
        }
        if (layerOutScale != null) {
            scaleRows(n, x);
        }
        if (res.blockOutputScale() != 1.0f) {
            CpuOps.scale(x, 0, n * streamWidth, res.blockOutputScale());
        }
    }

    /**
     * Steps 3-6. Returns true when the mixer branch, left in {@link CpuScratch#BRANCH}, still has to be added with
     * the FFN branch (PARALLEL).
     */
    private boolean mixerBlock(int pos0, int n, float[] x) {
        float[] in = attnNorm != null ? normed : x;
        if (mixerNorm2 != null) {
            mixerNorm2.apply(x, mixerIn, n, width);
            in = mixerIn;
        }
        if (selfAttention != null) {
            run(selfAttention, pos0, n, in, branch);
            residual(n, x, branch, in, false);
            crossNorm.apply(x, normed, n, width);
            in = normed;
        }
        run(mixer, pos0, n, in, branch);
        if (parallelAttention != null) {
            run(parallelAttention, pos0, n, in, aux);
            CpuOps.add(aux, 0, branch, 0, branch, 0, n * width);
        }
        if (attnOutNorm != null) {
            attnOutNorm.apply(branch, branch, n, width);
        }
        if (postAttnNorm != null) {
            postAttnNorm.apply(branch, branch, n, width);
        }
        boolean consumed = false;
        for (CpuLayerExtra e : extras) {
            consumed |= e.afterMixer(pos0, n, x, branch);
        }
        boolean pending = false;
        if (!consumed) {
            requireConsumed("mixer");
            if (parallel) {
                pending = res.mixerResidual();
            } else if (res.mixerResidual()) {
                residual(n, x, branch, in, false);
            }
        }
        if (postLnAttn != null) {
            postLnAttn.apply(x, x, n, width);
            if (postLnAttn2 != null) {
                mix(MixMode.ADD, 0f, n, x, layerIn, x);
                postLnAttn2.apply(x, x, n, width);
            }
        }
        return pending;
    }

    /** Steps 7-11. */
    private void ffnBlock(int pos0, int n, float[] x, boolean attnPending) {
        float[] in;
        if (ffnNorm != null) {
            ffnNorm.apply(x, normed, n, width);
            in = normed;
            if (ada != null) {
                ada.apply(ctx.matVec(), adaCond, adaHidden, adaVector, normed, 0, n);
            }
        } else if (ffnReadsAttnNorm) {
            in = normed;
        } else {
            in = x;
        }
        float[] out = ffnBranch != null ? ffnBranch : branch;
        ffn.forward(n, in, out);
        if (postFfnNorm != null) {
            postFfnNorm.apply(out, out, n, width);
        }
        boolean consumed = false;
        for (CpuLayerExtra e : extras) {
            consumed |= e.afterFfn(pos0, n, x, out);
        }
        if (consumed) {
            return;
        }
        requireConsumed("FFN");
        if (parallel) {
            if (attnPending) {
                CpuOps.add(out, 0, branch, 0, out, 0, n * width);
            }
            mix(MixMode.ADD, 0f, n, x, out, x);
        } else {
            residual(n, x, out, in, true);
        }
    }

    private void run(CpuMixer m, int pos0, int n, float[] in, float[] out) {
        if (n == 1 && ctx.hasMemory()) {
            m.decode(pos0, in, out);
        } else {
            m.prefill(pos0, n, in, out);
        }
    }

    /**
     * Sequential residual of one sub-block: {@code normedIn} is that sub-block's normed input, {@code ffn} tells the
     * FFN sub-block from the mixer.
     */
    private void residual(int n, float[] x, float[] br, float[] normedIn, boolean ffn) {
        switch (res.kind()) {
            case STANDARD:
                mix(MixMode.ADD, 0f, n, x, br, x);
                break;
            case SCALED:
                mix(MixMode.SCALED, res.residualScale(), n, x, br, x);
                break;
            case NORMED_INPUT_SCALED:
                mix(MixMode.NORMED_INPUT_SCALED, res.residualScale(), n, normedIn, br, x);
                break;
            case DEPTH_SCALED:
                CpuOps.scale(x, 0, n * width, res.streamScale(ffn));
                mix(MixMode.SCALED, res.branchScale(ffn), n, x, br, x);
                break;
            default:
                throw new IllegalStateException(describe() + ": no sequential residual for " + res.kind());
        }
    }

    private void requireConsumed(String block) {
        if (res.kind() == ResidualKind.HYPER_CONNECTIONS) {
            throw new IllegalStateException(describe() + ": the hyper-connection extra did not consume the " + block
                    + " residual");
        }
    }

    private void mix(MixMode mode, float s, int n, float[] inp, float[] cur, float[] out) {
        mixRows(mode, s, n, width, inp, cur, null, null, 0, out);
    }

    /**
     * {@link NormMath#residualMix} over {@code n} rows of {@code width}, sharded by rows; alpha / beta vectors of
     * {@code abLen} values are indexed from each row start.
     */
    static void mixRows(MixMode mode, float s, int n, int width, float[] inp, float[] cur, float[] alpha,
                        float[] beta, int abLen, float[] out) {
        int grain = Math.max(1, ELEMENT_GRAIN / Math.max(1, width));
        CpuParallel.parallelFor(n, grain, (from, to) -> NormMath.residualMix(mode, (to - from) * width, s,
                inp, from * width, cur, from * width, alpha, beta, abLen, out, from * width));
    }

    private void scaleRows(int n, float[] x) {
        int grain = Math.max(1, ELEMENT_GRAIN / streamWidth);
        CpuParallel.parallelFor(n, grain, (from, to) -> NormMath.mulRowScale(to - from, streamWidth, layerOutScale,
                0, layerOutScale.length, x, from * streamWidth));
    }

    // ------------------------------------------------------------------ memory operations

    /**
     * Re-rotates this layer's cached keys after {@code HeapModelMemory.shiftPositions} moved the rows: only a layer
     * that owns its KV cache shifts it, through the mixer that writes it. Layers without a KV cache do nothing.
     *
     * @throws UnsupportedOperationException when that mixer cannot shift its memory
     */
    public void shiftContext(int nKeep, int nDiscard) {
        checkOpen();
        CpuMixer owner = cacheMixer();
        if (owner != null) {
            owner.shiftContext(nKeep, nDiscard);
        }
    }

    /**
     * Why {@link #shiftContext} would refuse, or null: the refusal of the mixer owning the layer's KV rows
     * ({@link CpuMixer#shiftRefusal}), asked before the memory moves any row.
     */
    public String shiftRefusal() {
        checkOpen();
        CpuMixer owner = cacheMixer();
        String refusal = owner != null ? owner.shiftRefusal() : null;
        return refusal != null ? describe() + ": " + refusal : null;
    }

    /**
     * The bound cross-attention mixer of a {@link MixerKind#CROSS_ATTENTION_T5} layer, so the encoder-decoder
     * backend can hand it the encoder output before the first decoder step; null for other layers.
     */
    public CpuMixer crossAttentionMixer() {
        checkOpen();
        return layer.mixerKind() == MixerKind.CROSS_ATTENTION_T5 ? mixer : null;
    }

    /** The mixer owning this layer's KV rows; null without KV or for a shared-KV reader. */
    private CpuMixer cacheMixer() {
        if (!layer.usesKv()) {
            return null;
        }
        AttentionSpec a = layer.attention();
        if (a != null && a.sharesKv(layer.il())) {
            return null;
        }
        return selfAttention != null ? selfAttention : parallelAttention != null ? parallelAttention : mixer;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException(describe() + ": executor is closed");
        }
    }

    /** Closes the layer's extras, FFN block and mixers. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        List<AutoCloseable> all = new ArrayList<>(extras);
        all.add(ffn);
        all.add(parallelAttention);
        all.add(mixer);
        all.add(selfAttention);
        RuntimeException failure = new IllegalStateException(describe() + ": closing the layer blocks failed");
        Closeables.closeAll(all, failure);
        if (failure.getSuppressed().length > 0) {
            throw failure;
        }
    }

    private static int rows(int ubatch, int w) {
        return Math.multiplyExact(ubatch, w);
    }

    private String describe() {
        return ctx.spec().arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
    }

    @Override
    public String toString() {
        return "CpuLayerExecutor{" + describe() + ", mixer=" + layer.mixerKind()
                + (layer.parallelAttention() ? "+ATTENTION" : "") + ", ffn=" + layer.ffnKind() + ", residual="
                + res.kind() + ", extras=" + layer.extras() + '}';
    }
}
