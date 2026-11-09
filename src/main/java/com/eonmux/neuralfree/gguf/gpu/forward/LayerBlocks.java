package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSource;
import com.eonmux.neuralfree.gguf.ops.norm.GpuAdaNorm;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.List;

/** One layer's bound blocks and resolved norm operands. */
final class LayerBlocks {
    final LayerSpec spec;
    final int streamWidth;
    final int width;
    final ResidualKind kind;
    final float residualScale;
    final boolean mixerResidual;
    final boolean discard;
    final boolean keepInput;
    final float blockScale;
    final NormKind normKind;
    final float normEps;
    final List<GpuLayerExtra> extras;
    GpuMixer mixer;
    GpuMixer parallelAttention;
    GpuMixer crossMixer;
    GpuFfnBlock ffn;
    Norm attnNorm;
    Norm postAttn;
    Norm attnOutBranch;
    Norm postLnAttn;
    Norm postLnAttn2;
    Norm postLnLayer;
    Norm crossNorm;
    Norm ffnNorm;
    Norm postFfn;
    boolean sharedFfnNorm;
    /** Whether the FFN residual may fuse with the next layer's attention norm; set by the executor. */
    boolean carriesFfnResidual;
    GpuWeight outScale;
    /** Adaptive conditioning of {@link #ffnNorm}'s output, or null. */
    GpuAdaNorm.Layer ada;
    /** SLERP residual: {@code |alpha * scale|} and its complement per sub-block; null without SLERP. */
    GpuWeight attnSlerpLr;
    GpuWeight attnSlerpOneMinus;
    GpuWeight ffnSlerpLr;
    GpuWeight ffnSlerpOneMinus;

    private LayerBlocks(ModelSpec model, LayerSpec spec) {
        this.spec = spec;
        ResidualSpec r = spec.residual();
        this.streamWidth = r.streamWidth();
        this.width = r.streamWidth() / r.hcMult();
        this.kind = r.kind();
        this.residualScale = r.residualScale();
        this.mixerResidual = r.mixerResidual();
        this.discard = r.discardLayerOutput();
        this.keepInput = keepsLayerInput(model, spec);
        this.blockScale = r.blockOutputScale();
        this.normKind = spec.norm().kind();
        this.normEps = spec.norm().eps();
        this.extras = new ArrayList<>();
    }

    /**
     * BERT-graph post-LN: no {@code attn_norm}, {@code attn_output_norm} on the stream after the mixer residual
     * and {@code layer_output_norm} after the FFN residual (bert.cpp:156-215).
     */
    static boolean postLayerNorm(NormSpec ns) {
        return !ns.attnNorm() && ns.postRole() == PostNormRole.NONE
                && (ns.attnOutputNorm() || ns.layerOutputNorm());
    }

    /**
     * falcon-40b: {@code attn_norm_2} of the layer input feeds the attention and {@code attn_norm} of the same
     * input feeds the FFN of the parallel residual (falcon.cpp:71-88, 125).
     */
    static boolean falconNorm2(LayerSpec l) {
        NormSpec ns = l.norm();
        return l.residual().kind() == ResidualKind.PARALLEL && l.residual().ffnUsesAttnNorm2() && ns.attnNorm2()
                && ns.attnNorm() && ns.postRole() != PostNormRole.POST_ONLY;
    }

    /**
     * jina-bert-v2: after {@code attn_output_norm} the layer input is added again and {@code attn_norm_2} is
     * applied (bert.cpp:162-165).
     */
    static boolean postLnAttnNorm2(ModelSpec model, LayerSpec l) {
        NormSpec ns = l.norm();
        return postLayerNorm(ns) && ns.attnOutputNorm() && !ns.weightless() && l.mixerKind() != MixerKind.NONE
                && model.tensors().layer(l.stack(), l.weightLayer(),
                TensorNames.weight(TensorNames.ATTN_NORM_2)) != null;
    }

    /** True when the layer reads its input again after the block started ({@code discardLayerOutput}, BERT). */
    static boolean keepsLayerInput(ModelSpec model, LayerSpec l) {
        return l.residual().discardLayerOutput() || postLnAttnNorm2(model, l);
    }

    /** The mixer that owns the layer's KV rows (the falcon-h1 parallel attention), or null without KV. */
    GpuMixer cacheMixer() {
        if (!spec.usesKv()) {
            return null;
        }
        return parallelAttention != null ? parallelAttention : mixer;
    }

    static LayerBlocks bind(GpuForwardContext ctx, GpuBlockRegistry registry, LayerSpec spec, NormNames names) {
        LayerBlocks b = new LayerBlocks(ctx.spec(), spec);
        try {
            b.resolveNorms(ctx, names);
            b.bindSlerp(ctx, names);
            b.bindBlocks(ctx, registry);
        } catch (RuntimeException e) {
            b.close(e);
            throw e;
        }
        return b;
    }

    private void resolveNorms(GpuForwardContext ctx, NormNames names) {
        NormSpec ns = spec.norm();
        GpuWeights w = ctx.weightSource().weightsFor(spec.weightLayer());
        String arch = ctx.spec().arch().ggufName();
        boolean weightless = ns.weightless();
        PostNormRole role = ns.postRole();
        boolean postLn = postLayerNorm(ns);
        boolean postBranches = role == PostNormRole.SANDWICH || role == PostNormRole.POST_ONLY;
        boolean falconNorm2 = falconNorm2(spec);
        if (ns.attnNorm() && role != PostNormRole.POST_ONLY) {
            attnNorm = named(w, arch, falconNorm2 ? TensorNames.ATTN_NORM_2 : TensorNames.ATTN_NORM, weightless);
        }
        if (postBranches && ns.postAttnNorm()) {
            postAttn = weightless || ns.postAttnWeightless() ? Norm.WEIGHTLESS
                    : fromRef(w, arch, names.postAttnNorm(spec), "post-attention norm");
            postAttn = postAttn.withEps(ns.postEps());
        }
        if (ns.attnOutputNorm()) {
            Norm n = named(w, arch, TensorNames.ATTN_OUT_NORM, weightless);
            if (postLn) {
                postLnAttn = n;
                if (postLnAttnNorm2(ctx.spec(), spec)) {
                    postLnAttn2 = named(w, arch, TensorNames.ATTN_NORM_2, false);
                }
            } else {
                attnOutBranch = n;
            }
        }
        if (postLn && ns.layerOutputNorm()) {
            postLnLayer = named(w, arch, TensorNames.LAYER_OUT_NORM, weightless);
        }
        if (spec.crossAttention() != null) {
            crossNorm = named(w, arch, TensorNames.DEC_CROSS_ATTN_NORM, weightless);
        }
        sharedFfnNorm = !falconNorm2 && ns.sharedAttnNormForFfn() && attnNorm != null
                && spec.ffnKind() != FfnKind.NONE;
        if (spec.ffnKind() != FfnKind.NONE && !sharedFfnNorm) {
            ffnNorm = falconNorm2 ? named(w, arch, TensorNames.ATTN_NORM, weightless)
                    : ffnInputNorm(w, arch, names, ns, weightless);
        }
        if (spec.ffnKind() != FfnKind.NONE && postBranches && ns.postFfnNorm()) {
            postFfn = weightless ? Norm.WEIGHTLESS : fromRef(w, arch, names.postFfnNorm(spec), "post-FFN norm");
            postFfn = postFfn.withEps(ns.postEps());
            if (ns.postFfnWeightFirst()) {
                postFfn = postFfn.weightFirst(arch, spec.il(), width);
            }
        }
        if (ns.groups() > 1) {
            if (attnNorm != null) {
                attnNorm = attnNorm.grouped(arch, spec.il(), width, ns.groups());
            }
            if (ffnNorm != null) {
                ffnNorm = ffnNorm.grouped(arch, spec.il(), width, ns.groups());
            }
        }
        if (ns.ffnAda()) {
            if (ffnNorm == null) {
                throw new IllegalStateException(arch + " layer " + spec.il() + ": the ada norm conditions an"
                        + " ffn_norm the layer does not apply");
            }
            GpuWeight down = vector(w, arch, ns.ffnAdaDown().name());
            GpuWeight up = vector(w, arch, ns.ffnAdaUp().name());
            GpuProjectionDeltas deltas = ctx.projectionDeltas();
            ada = new GpuAdaNorm.Layer(down, up,
                    deltas.serve(spec, TensorNames.weight(TensorNames.FFN_NORM_ADA_DOWN), down),
                    deltas.serve(spec, TensorNames.weight(TensorNames.FFN_NORM_ADA_UP), up), width);
        }
        TensorRef scale = ns.layerOutputScale();
        if (scale != null) {
            outScale = vector(w, arch, scale.name());
            boolean fits = outScale.cols() == 1 || outScale.cols() == streamWidth;
            if (outScale.type() != GGMLType.F32 || !outScale.isSingleShard() || outScale.rows() != 1 || !fits) {
                throw new IllegalArgumentException(arch + " layer " + spec.il() + ": layer_output_scale must be"
                        + " a single-buffer F32 vector of 1 or " + streamWidth + " values, got " + outScale);
            }
        }
    }

    /**
     * SLERP residuals: the per-element {@code |alpha * slerpScale|} of each sub-block and its complement,
     * uploaded as two F32 vectors this layer owns.
     */
    private void bindSlerp(GpuForwardContext ctx, NormNames names) {
        if (kind != ResidualKind.SLERP) {
            return;
        }
        ResidualSpec r = spec.residual();
        attnSlerpLr = slerpVector(ctx, r.attnSlerp(), true);
        attnSlerpOneMinus = slerpVector(ctx, r.attnSlerp(), false);
        ffnSlerpLr = slerpVector(ctx, r.ffnSlerp(), true);
        ffnSlerpOneMinus = slerpVector(ctx, r.ffnSlerp(), false);
    }

    private GpuWeight slerpVector(GpuForwardContext ctx, TensorRef ref, boolean lr) {
        if (ref == null) {
            return null;
        }
        GpuWeights weights = ctx.weightSource().weightsFor(spec.weightLayer());
        String arch = ctx.spec().arch().ggufName();
        GpuWeight alpha = vector(weights, arch, ref.name());
        if (alpha.cols() != width) {
            throw new IllegalArgumentException(arch + " layer " + spec.il() + ": " + ref.name() + " must hold "
                    + width + " values, got " + alpha.cols());
        }
        LayerWeightSource source = ctx.weightSource();
        source.prefetch(spec.weightLayer());
        source.await(spec.weightLayer());
        float[] host = new float[width];
        GpuEvent read = ctx.queue().enqueueReadBuffer(alpha.shard(0), 0, (long) width * Float.BYTES, host);
        if (read != null) {
            read.close();
        }
        ctx.queue().finish();
        float[] values = new float[width];
        float scale = spec.residual().slerpScale();
        for (int i = 0; i < width; i++) {
            float v = Math.abs(host[i] * scale);
            values[i] = lr ? v : 1.0f - v;
        }
        GpuBuffer buffer = ctx.backend().allocateBuffer((long) width * Float.BYTES, true);
        GpuEvent write = ctx.queue().enqueueWriteBuffer(buffer, 0, (long) width * Float.BYTES, values);
        if (write != null) {
            write.close();
        }
        return GpuWeight.of(ref.name() + (lr ? ".lr" : ".inv"), GGMLType.F32, new long[]{width}, buffer);
    }

    private Norm ffnInputNorm(GpuWeights w, String arch, NormNames names, NormSpec ns, boolean weightless) {            boolean expected = ns.ffnNorm() || ns.ffnInputIsPostAttnNorm()
                || (ns.attnNorm2() && spec.residual().ffnUsesAttnNorm2());
        if (!expected) {
            return null;
        }
        if (weightless) {
            return Norm.WEIGHTLESS;
        }
        TensorRef ref = names.ffnNorm(spec);
        if (ref == null) {
            throw new MissingTensorException(arch, spec.il(), spec.tensorName(TensorNames.weight(
                    TensorNames.FFN_NORM)));
        }
        return fromRef(w, arch, ref, "FFN norm");
    }

    private Norm named(GpuWeights w, String arch, String base, boolean weightless) {
        if (weightless) {
            return Norm.WEIGHTLESS;
        }
        GpuWeight weight = w.layerOrNull(spec, TensorNames.weight(base));
        if (weight == null) {
            throw new MissingTensorException(arch, spec.il(), spec.tensorName(TensorNames.weight(base)));
        }
        return new Norm(weight, w.layerOrNull(spec, TensorNames.bias(base)));
    }

    private Norm fromRef(GpuWeights w, String arch, TensorRef ref, String what) {
        if (ref == null) {
            throw new IllegalStateException(arch + " layer " + spec.il() + ": the spec has a " + what
                    + " but no tensor names it");
        }
        GpuWeight weight = vector(w, arch, ref.name());
        String bias = biasName(ref.name());
        return new Norm(weight, bias != null ? w.find(bias) : null);
    }

    private GpuWeight vector(GpuWeights w, String arch, String name) {
        GpuWeight v = w.find(name);
        if (v == null) {
            v = w.getOrNull(name);
        }
        if (v == null) {
            throw new MissingTensorException(arch, spec.il(), name);
        }
        return v;
    }

    private static String biasName(String weightName) {
        String suffix = "." + TensorNames.WEIGHT;
        return weightName.endsWith(suffix)
                ? weightName.substring(0, weightName.length() - suffix.length()) + "." + TensorNames.BIAS : null;
    }

    private void bindBlocks(GpuForwardContext ctx, GpuBlockRegistry registry) {
        if (spec.mixerKind() == MixerKind.CROSS_ATTENTION_T5) {
            LayerSpec self = attentionView(spec);
            mixer = registry.mixer(self);
            mixer.bind(ctx, self);
            crossMixer = registry.mixer(spec);
            crossMixer.bind(ctx, spec);
        } else if (spec.mixerKind() != MixerKind.NONE) {
            mixer = registry.mixer(spec);
            mixer.bind(ctx, spec);
            if (spec.parallelAttention()) {
                LayerSpec attn = attentionView(spec);
                parallelAttention = registry.mixer(attn);
                parallelAttention.bind(ctx, attn);
            }
        }
        if (spec.ffnKind() != FfnKind.NONE) {
            ffn = registry.ffn(spec);
            ffn.bind(ctx, spec);
        }
        extras.addAll(registry.extras(spec));
        for (GpuLayerExtra e : extras) {
            e.bind(ctx, spec);
        }
    }

    /**
     * The layer's softmax-attention branch as a plain ATTENTION layer (T5 decoder self-attention, falcon-h1
     * parallel attention), bound through the registry's ATTENTION factory.
     */
    static LayerSpec attentionView(LayerSpec l) {
        return LayerSpec.builder(l.il(), MixerKind.ATTENTION, FfnKind.NONE)
                .weightLayer(l.weightLayer())
                .stack(l.stack())
                .norm(l.norm())
                .residual(l.residual())
                .attention(l.attention())
                .rope(l.rope())
                .build();
    }

    /** Closes every bound block; returns {@code first} or the first close failure. */
    RuntimeException close(RuntimeException first) {
        RuntimeException err = first;
        List<AutoCloseable> all = new ArrayList<>();
        all.add(mixer);
        all.add(parallelAttention);
        all.add(crossMixer);
        all.add(ffn);
        all.addAll(extras);
        for (AutoCloseable c : all) {
            if (c == null) {
                continue;
            }
            try {
                c.close();
            } catch (Exception e) {
                RuntimeException re = e instanceof RuntimeException ? (RuntimeException) e
                        : new IllegalStateException("closing " + c + " failed", e);
                if (err == null) {
                    err = re;
                } else if (err != re) {
                    err.addSuppressed(re);
                }
            }
        }
        mixer = null;
        parallelAttention = null;
        crossMixer = null;
        ffn = null;
        extras.clear();
        closeOwned(attnSlerpLr);
        closeOwned(attnSlerpOneMinus);
        closeOwned(ffnSlerpLr);
        closeOwned(ffnSlerpOneMinus);
        attnSlerpLr = null;
        attnSlerpOneMinus = null;
        ffnSlerpLr = null;
        ffnSlerpOneMinus = null;
        return err;
    }

    private static void closeOwned(GpuWeight w) {
        if (w != null) {
            w.shard(0).close();
        }
    }
}
