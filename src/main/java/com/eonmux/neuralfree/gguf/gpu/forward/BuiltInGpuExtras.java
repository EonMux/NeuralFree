package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Layer extras the executor ships with, registered like any family's extras ({@link #factories()}):
 * <ul>
 *   <li>{@value LayerSpec.Extras#EMBD_SKIP} (talkie): after the block,
 *       {@code x += embedding * layer_output_scale * embeddingSkipScale} (models/talkie.cpp:123-126), the embedding
 *       being the embedding-stage output kept in {@link ScratchSlots#INPUT_EMBD};</li>
 *   <li>{@value LayerSpec.Extras#RESID_ALPHA_BETA} (muse-spark): the norm-preserving residuals
 *       {@code x = x * attn_resid_alpha + branch * attn_resid_beta} after the mixer and the same with the
 *       {@code ffn_resid_*} pair after the FFN (models/muse-spark.cpp:166-168, 208-210), each replacing the
 *       executor's residual;</li>
 *   <li>{@value LayerSpec.Extras#LOOP_NORM} (nanbeige): after the slots that end a loop
 *       ({@link LayerSchedule#normAfterSlot(int)}), {@code x = rms(x) * output_norm} with {@code f_norm_rms_eps}
 *       (models/nanbeige.cpp:164-171).</li>
 * </ul>
 * Each runs {@value NormKernels#RESIDUAL_MIX} or {@value NormKernels#ROW_NORM}; per step they allocate nothing and
 * never wait on the host. Not thread-safe.
 */
public final class BuiltInGpuExtras {

    /** Kernels the built-in extras launch, for prewarming. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            NormKernels.RESIDUAL_MIX, NormKernels.ROW_NORM, GpuOps.AXPBY, GpuOps.COPY2D));

    /** Shared temporary holding ones (alpha of the embedding skip); every binder writes the same content. */
    static final String TEMP_ONES = "builtin.ones";
    /** Prefix of the per-layer embedding-skip beta temporaries. */
    static final String TEMP_SKIP_BETA = "builtin.embd_skip.beta.";
    private static final int ONE_BITS = Float.floatToRawIntBits(1.0f);

    private static final List<GpuLayerExtraFactory> FACTORIES = Collections.unmodifiableList(Arrays.asList(
            factory(LayerSpec.Extras.EMBD_SKIP, Kind.EMBEDDING_SKIP),
            factory(LayerSpec.Extras.RESID_ALPHA_BETA, Kind.ALPHA_BETA),
            factory(LayerSpec.Extras.LOOP_NORM, Kind.LOOP_NORM)));

    private enum Kind { EMBEDDING_SKIP, ALPHA_BETA, LOOP_NORM }

    private BuiltInGpuExtras() {
    }

    /** Factories of the embedding skip, alpha/beta residual and loop norm extras. */
    public static List<GpuLayerExtraFactory> factories() {
        return FACTORIES;
    }

    private static GpuLayerExtraFactory factory(String id, Kind kind) {
        return new GpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return id;
            }

            @Override
            public GpuLayerExtra create(LayerSpec layer) {
                Objects.requireNonNull(layer, "layer");
                switch (kind) {
                    case EMBEDDING_SKIP:
                        return new EmbeddingSkip();
                    case ALPHA_BETA:
                        return new AlphaBetaResidual();
                    default:
                        return new LoopNorm();
                }
            }

            @Override
            public String toString() {
                return "BuiltInGpuExtras." + id;
            }
        };
    }

    /** {@code out = inp * alpha + cur * beta} over {@code n} floats (alpha/beta of {@code abLen}, broadcast). */
    static void alphaBeta(GpuOps ops, GpuBuffer placeholder, int n, ScratchSlot inp, ScratchSlot cur,
                          GpuBuffer alpha, GpuBuffer beta, int abLen, ScratchSlot out) {
        int local = ops.local();
        ops.enqueue(NormKernels.RESIDUAL_MIX, new long[]{(n + (long) local - 1) / local * local}, new long[]{local},
                NormKernels.residualMixArgs(MixMode.ALPHA_BETA, n, 1.0f, inp.buffer(), inp.offset(), cur.buffer(),
                        cur.offset(), alpha, beta, abLen, out.buffer(), out.offset(), placeholder));
    }

    /** A single-buffer F32 vector of {@code 1} or {@code width} values by stored name or alias. */
    static GpuWeight vector(GpuWeights weights, String arch, LayerSpec layer, String name, int width) {
        GpuWeight w = weights.find(name);
        if (w == null) {
            w = weights.getOrNull(name);
        }
        if (w == null) {
            throw new MissingTensorException(arch, layer.il(), name);
        }
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() != 1
                || (w.cols() != width && w.cols() != 1)) {
            throw new IllegalArgumentException(arch + " layer " + layer.il() + ": " + name + " must be a single-buffer"
                    + " F32 vector of 1 or " + width + " values, got " + w);
        }
        return w;
    }

    private static void requireUnbound(Object state, LayerSpec layer, String what) {
        if (state != null) {
            throw new IllegalStateException(what + " of layer " + layer.il() + " is already bound");
        }
    }

    private static void requireBound(Object state, String what) {
        if (state == null) {
            throw new IllegalStateException(what + " used before bind()");
        }
    }

    /** talkie embedding skip. */
    static final class EmbeddingSkip implements GpuLayerExtra {
        private GpuOps ops;
        private GpuBuffer placeholder;
        private ScratchSlot embedding;
        private ScratchSlot ones;
        private ScratchSlot beta;
        private int width;
        private int abLen;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            requireUnbound(ops, layer, "embedding skip");
            String arch = ctx.spec().arch().ggufName();
            EmbeddingSpec e = ctx.spec().embedding();
            ResidualSpec r = layer.residual();
            int w = r.streamWidth();
            if (r.hcMult() != 1 || e.nEmbd() != w || Math.max(e.nEmbdInp(), e.nEmbd()) != w) {
                throw new IllegalStateException(arch + " layer " + layer.il() + ": the embedding skip needs the"
                        + " embedding rows (" + e.nEmbd() + "/" + e.nEmbdInp() + ") as wide as the stream (" + w + ")");
            }
            GpuOps o = ctx.ops();
            int ubatch = ctx.scratch().ubatch();
            GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
            GpuWeight scale = weights.layerOrNull(layer, TensorNames.weight(TensorNames.LAYER_OUT_SCALE));
            int len = scale != null ? (int) vector(weights, arch, layer, scale.name(), w).cols() : 1;
            ScratchSlot b = ctx.scratch().temp(TEMP_SKIP_BETA + layer.stack() + "." + layer.il(), len);
            ScratchSlot a = ctx.scratch().temp(TEMP_ONES, w);
            if (a.offset() != 0 || b.offset() != 0) {
                throw new IllegalStateException("embedding-skip temporaries must start at element 0");
            }
            o.fill(a.buffer(), 0, (long) w * Float.BYTES, ONE_BITS);
            float s = r.embeddingSkipScale();
            if (scale != null) {
                o.copy(scale.shard(0), 0, b.buffer(), 0, len);
                if (s != 1.0f) {
                    o.scale(b.buffer(), 0, len, s);
                }
            } else {
                o.fill(b.buffer(), 0, (long) len * Float.BYTES, Float.floatToRawIntBits(s));
            }
            this.embedding = ctx.scratch().slot(ScratchSlots.INPUT_EMBD).require((long) ubatch * w);
            this.placeholder = ctx.placeholders().readOnlyFloats();
            this.ones = a;
            this.beta = b;
            this.width = w;
            this.abLen = len;
            o.prewarm(NormKernels.RESIDUAL_MIX, new long[]{o.local()});
            this.ops = o;
        }

        @Override
        public void afterBlock(int pos0, int nTokens, ScratchSlot x) {
            requireBound(ops, "embedding skip");
            alphaBeta(ops, placeholder, nTokens * width, x, embedding, ones.buffer(), beta.buffer(), abLen, x);
        }

        @Override
        public void close() {
            ops = null;
        }

        @Override
        public String toString() {
            return "EmbeddingSkip{width=" + width + '}';
        }
    }

    /** muse-spark norm-preserving residuals. */
    static final class AlphaBetaResidual implements GpuLayerExtra {
        private GpuOps ops;
        private GpuBuffer placeholder;
        private GpuWeight attnAlpha;
        private GpuWeight attnBeta;
        private GpuWeight ffnAlpha;
        private GpuWeight ffnBeta;
        private int width;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            requireUnbound(ops, layer, "alpha/beta residual");
            String arch = ctx.spec().arch().ggufName();
            ResidualSpec r = layer.residual();
            if (!r.alphaBeta()) {
                throw new IllegalStateException(arch + " layer " + layer.il() + ": the alpha/beta residual extra needs"
                        + " attn_resid_* or ffn_resid_* tensors in the ResidualSpec");
            }
            int w = r.streamWidth() / r.hcMult();
            GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
            this.attnAlpha = pairPart(weights, arch, layer, r.attnAlpha(), w);
            this.attnBeta = pairPart(weights, arch, layer, r.attnBeta(), w);
            this.ffnAlpha = pairPart(weights, arch, layer, r.ffnAlpha(), w);
            this.ffnBeta = pairPart(weights, arch, layer, r.ffnBeta(), w);
            this.width = w;
            this.placeholder = ctx.placeholders().readOnlyFloats();
            GpuOps o = ctx.ops();
            o.prewarm(NormKernels.RESIDUAL_MIX, new long[]{o.local()});
            this.ops = o;
        }

        private static GpuWeight pairPart(GpuWeights weights, String arch, LayerSpec layer, TensorRef ref, int w) {
            if (ref == null) {
                return null;
            }
            GpuWeight v = vector(weights, arch, layer, ref.name(), w);
            if (v.cols() != w) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + ": " + v.name() + " must hold "
                        + w + " values, got " + v.cols());
            }
            return v;
        }

        @Override
        public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
            return apply(attnAlpha, attnBeta, nTokens, x, branch);
        }

        @Override
        public boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
            return apply(ffnAlpha, ffnBeta, nTokens, x, branch);
        }

        private boolean apply(GpuWeight alpha, GpuWeight beta, int nTokens, ScratchSlot x, ScratchSlot branch) {
            requireBound(ops, "alpha/beta residual");
            if (alpha == null) {
                return false;
            }
            alphaBeta(ops, placeholder, nTokens * width, x, branch, alpha.shard(0), beta.shard(0), width, x);
            return true;
        }

        @Override
        public void close() {
            ops = null;
        }

        @Override
        public String toString() {
            return "AlphaBetaResidual{attn=" + (attnAlpha != null) + ", ffn=" + (ffnAlpha != null) + '}';
        }
    }

    /** nanbeige loop boundary norm. */
    static final class LoopNorm implements GpuLayerExtra {
        private GpuOps ops;
        private GpuWeight weight;
        private float eps;
        private boolean active;
        private int width;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            requireUnbound(ops, layer, "loop norm");
            String arch = ctx.spec().arch().ggufName();
            LayerSchedule schedule = ctx.spec().schedule();
            if (!TensorNames.STACK_BLK.equals(layer.stack()) || layer.il() >= schedule.slots()) {
                throw new IllegalStateException(arch + " layer " + layer.il() + " (" + layer.stack() + ") is not a"
                        + " trunk slot of " + schedule);
            }
            this.active = schedule.normAfterSlot(layer.il());
            this.width = layer.residual().streamWidth();
            this.eps = ctx.spec().hparams().normRmsEps();
            if (!(eps > 0f)) {
                throw new IllegalStateException(arch + ": the loop norm needs a positive RMS epsilon, got " + eps);
            }
            String name = TensorNames.weight(TensorNames.OUTPUT_NORM);
            GpuWeight w = ctx.weights().getOrNull(name);
            if (w == null) {
                throw new MissingTensorException(arch, -1, name);
            }
            this.weight = w;
            this.ops = ctx.ops();
        }

        @Override
        public void afterBlock(int pos0, int nTokens, ScratchSlot x) {
            requireBound(ops, "loop norm");
            if (active) {
                ops.norm(NormKind.RMS, weight, null, eps, x.buffer(), x.offset(), x.buffer(), x.offset(), nTokens,
                        width);
            }
        }

        @Override
        public void close() {
            ops = null;
        }

        @Override
        public String toString() {
            return "LoopNorm{active=" + active + '}';
        }
    }
}
