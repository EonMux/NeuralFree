package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
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
 *   <li>{@value LayerSpec.Extras#RESID_SCALE} (limite without MUDD): the learned {@code attn_resid_scale} and
 *       {@code ffn_resid_scale} fold each sub-block's residual, {@code x = x * scale + branch};</li>
 *   <li>{@value LayerSpec.Extras#MUDD} (limite): the dense cross-layer mix (models/limite.cpp:100-140) whose
 *       attention input and residual are weighted sums of earlier layers' inputs, plus the FFN residual scale. It
 *       keeps the per-layer input bank in a shared temporary and mixes with {@value #MUDD_MIX}.</li>
 * </ul>
 * Each runs {@value NormKernels#RESIDUAL_MIX}, {@value NormKernels#ROW_NORM} or {@value #MUDD_MIX}; per step they
 * allocate nothing and never wait on the host. Not thread-safe.
 */
public final class BuiltInGpuExtras {

    /** limite MUDD dense cross-layer mix ({@code kernels/llm/eltwise_ext.cl}). */
    public static final String MUDD_MIX = "LLM_MuddMix";

    /** Kernels the built-in extras launch, for prewarming. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            NormKernels.RESIDUAL_MIX, NormKernels.ROW_NORM, GpuOps.AXPBY, GpuOps.COPY2D, ActKernels.UNARY,
            MUDD_MIX));

    /** Shared temporary holding ones (alpha of the embedding skip); every binder writes the same content. */
    static final String TEMP_ONES = "builtin.ones";
    /** Prefix of the per-layer embedding-skip beta temporaries. */
    static final String TEMP_SKIP_BETA = "builtin.embd_skip.beta.";
    /** Shared bank of the limite trunk layer inputs; slot il holds layer il's input of the current step. */
    static final String TEMP_BANK = "builtin.limite.bank";
    private static final int ONE_BITS = Float.floatToRawIntBits(1.0f);

    private static final List<GpuLayerExtraFactory> FACTORIES = Collections.unmodifiableList(Arrays.asList(
            factory(LayerSpec.Extras.EMBD_SKIP, Kind.EMBEDDING_SKIP),
            factory(LayerSpec.Extras.RESID_ALPHA_BETA, Kind.ALPHA_BETA),
            factory(LayerSpec.Extras.RESID_SCALE, Kind.RESID_SCALE),
            factory(LayerSpec.Extras.MUDD, Kind.MUDD),
            factory(LayerSpec.Extras.LOOP_NORM, Kind.LOOP_NORM)));

    private enum Kind { EMBEDDING_SKIP, ALPHA_BETA, LOOP_NORM, RESID_SCALE, MUDD }

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
                    case RESID_SCALE:
                        return new ResidScale();
                    case MUDD:
                        return new Mudd();
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

    // ================================================================== limite

    /**
     * Common state of the limite extras: the learned residual scales, the shared bank of every trunk layer's input
     * and the ones factor of the alpha/beta mix. Every hook folds one step row at a time over the bank, so a paged or
     * multi-segment step is exact.
     */
    abstract static class LimiteResid implements GpuLayerExtra {
        static final String TEMP_CUR = "builtin.limite.cur.";
        static final String TEMP_RESID = "builtin.limite.resid.";
        static final String TEMP_W = "builtin.limite.w.";
        static final String TEMP_NORMED = "builtin.limite.normed.";
        static final String TEMP_DOWN = "builtin.limite.down.";
        static final String TEMP_TAPS = "builtin.limite.taps.";

        GpuOps ops;
        GpuBuffer placeholder;
        ScratchSlot ones;
        GpuWeight attnScale;
        GpuWeight ffnScale;
        GpuScratch scratch;
        ScratchSlot bank;
        int nEmbd;
        int il;
        int ubatch;
        boolean bound;

        @Override
        public String batchRefusal() {
            return null;
        }

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            requireUnbound(bound ? this : null, layer, featureId());
            String arch = ctx.spec().arch().ggufName();
            GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
            this.nEmbd = ctx.spec().hparams().nEmbd();
            if (layer.residual().streamWidth() != nEmbd) {
                throw new IllegalStateException(arch + " layer " + layer.il() + ": the '" + featureId()
                        + "' extra needs an " + nEmbd + "-wide stream, got " + layer.residual().streamWidth());
            }
            this.il = layer.il();
            this.scratch = ctx.scratch();
            this.ubatch = scratch.ubatch();
            this.attnScale = scalar(weights, arch, layer, TensorNames.weight(TensorNames.ATTN_RESID_SCALE));
            this.ffnScale = scalar(weights, arch, layer, TensorNames.weight(TensorNames.FFN_RESID_SCALE));
            ScratchSlot a = scratch.temp(TEMP_ONES, 1);
            if (a.offset() != 0) {
                throw new IllegalStateException("a limite temporary must start at element 0");
            }
            ops = ctx.ops();
            ops.fill(a.buffer(), 0, Float.BYTES, ONE_BITS);
            this.ones = a;
            this.placeholder = ctx.placeholders().readOnlyFloats();
            long bankFloats = (long) ctx.spec().nLayer() * ubatch * nEmbd;
            if (bankFloats > Integer.MAX_VALUE - 8) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + ": the layer bank of " + bankFloats
                        + " floats exceeds the kernels' int addressing; lower neuralfree.ubatch");
            }
            this.bank = scratch.temp(TEMP_BANK, bankFloats);
            this.bound = true;
            ops.prewarm(NormKernels.RESIDUAL_MIX, new long[]{ops.local()});
        }

        /** Feature id of the concrete extra, for the double-bind check. */
        abstract String featureId();

        /** An F32 single-buffer scalar of the layer. */
        private static GpuWeight scalar(GpuWeights weights, String arch, LayerSpec layer, String name) {
            GpuWeight w = weights.layerOrNull(layer, name);
            if (w == null) {
                throw new MissingTensorException(arch, layer.il(), name);
            }
            if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() * w.cols() != 1L) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + ": " + name
                        + " must be a single-buffer F32 scalar, got " + w);
            }
            return w;
        }

        /** An F32 single-buffer vector of the layer holding exactly {@code n} values. */
        static GpuWeight vector(GpuWeights weights, String arch, LayerSpec layer, String name, long n) {
            GpuWeight w = weights.layerOrNull(layer, name);
            if (w == null) {
                throw new MissingTensorException(arch, layer.il(), name);
            }
            if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() * w.cols() != n) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + ": " + name + " must be a"
                        + " single-buffer F32 tensor of " + n + " values, got " + w);
            }
            return w;
        }

        /** Copies {@code n} rows of the residual stream into this layer's bank slot. */
        void bankRows(int n, ScratchSlot x) {
            checkStep(n);
            ops.copy(x.buffer(), x.offset(), bank.buffer(), bank.offset() + il * ubatch * nEmbd, n * nEmbd);
        }

        /** {@code x = x * scale + branch} over the step rows. */
        boolean fold(GpuWeight scale, int n, ScratchSlot x, ScratchSlot branch) {
            checkStep(n);
            alphaBeta(ops, placeholder, n * nEmbd, x, branch, scale.shard(0), ones.buffer(), 1, x);
            return true;
        }

        void checkStep(int n) {
            if (!bound) {
                throw new IllegalStateException(featureId() + " extra used before bind()");
            }
            if (n < 1 || n > ubatch) {
                throw new IllegalArgumentException("layer " + il + ": step of " + n + " tokens outside [1, " + ubatch
                        + "]");
            }
        }

        long roundUp(long items) {
            int local = ops.local();
            return Math.max(1, (items + local - 1) / local) * local;
        }
    }

    /** limite without MUDD: the learned scales fold the plain residual and the FFN input. */
    static final class ResidScale extends LimiteResid {
        @Override
        String featureId() {
            return LayerSpec.Extras.RESID_SCALE;
        }

        @Override
        public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
            return fold(attnScale, nTokens, x, branch);
        }

        @Override
        public boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
            return fold(ffnScale, nTokens, x, branch);
        }

        @Override
        public void close() {
            ops = null;
        }
    }

    /**
     * limite MUDD (models/limite.cpp:100-140): the attention input and the residual of the layer become weighted
     * sums of the banked inputs of its taps, {@code w = mudd_up @ gelu_erf(mudd_down @ rms(x)) + mudd_up_b} holding
     * one weight per tap for each mix ({@value #MUDD_MIX} adds the bias as it reads the weights). {@link #beforeBlock}
     * banks the input, rewrites the stream with the attention mix (which the mixer reads) and keeps the residual mix
     * for {@link #afterMixer}. Every scratch view enters the mix at its own offset.
     */
    static final class Mudd extends LimiteResid {
        private GpuWeight down;
        private GpuWeight up;
        private GpuWeight upBias;
        private ScratchSlot cur;
        private ScratchSlot resid;
        private ScratchSlot w;
        private ScratchSlot normed;
        private ScratchSlot downOut;
        private ScratchSlot taps;
        private int nTaps;
        private int nFf;
        private float eps;

        @Override
        String featureId() {
            return LayerSpec.Extras.MUDD;
        }

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            super.bind(ctx, layer);
            String arch = ctx.spec().arch().ggufName();
            GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
            ModelHparams hp = ctx.spec().hparams();
            this.nTaps = hp.muddTapCount();
            if (nTaps <= 0) {
                throw new IllegalStateException("limite layer " + il + " lists '" + LayerSpec.Extras.MUDD
                        + "' but the file sets no mudd.tap_count");
            }
            int[] tapIdx = new int[nTaps];
            for (int j = 0; j < nTaps; j++) {
                tapIdx[j] = hp.muddTapIdx(il, j);
                if (tapIdx[j] < 0 || tapIdx[j] > il) {
                    throw new IllegalStateException("limite layer " + il + ": every MUDD tap of a mixing layer must be"
                            + " in [0, " + il + "], got " + Arrays.toString(tapIdx) + " (limite.cpp:120)");
                }
            }
            this.down = weights.layerOrNull(layer, TensorNames.weight(TensorNames.MUDD_DOWN));
            this.up = weights.layerOrNull(layer, TensorNames.weight(TensorNames.MUDD_UP));
            if (down == null || up == null) {
                throw new MissingTensorException(arch, il, down == null ? TensorNames.weight(TensorNames.MUDD_DOWN)
                        : TensorNames.weight(TensorNames.MUDD_UP));
            }
            this.nFf = (int) down.rows();
            if (down.cols() != nEmbd || up.rows() != 2L * nTaps || up.cols() != nFf) {
                throw new IllegalArgumentException("limite layer " + il + ": mudd_down must be [" + nEmbd + ", " + nFf
                        + "] and mudd_up [" + (2 * nTaps) + ", " + nFf + "], got [" + down.cols() + ", " + down.rows()
                        + "] and [" + up.cols() + ", " + up.rows() + "]");
            }
            this.upBias = vector(weights, arch, layer, TensorNames.bias(TensorNames.MUDD_UP), 2L * nTaps);
            this.eps = hp.normRmsEps();
            this.cur = scratch.temp(TEMP_CUR + il, (long) ubatch * nEmbd);
            this.resid = scratch.temp(TEMP_RESID + il, (long) ubatch * nEmbd);
            this.w = scratch.temp(TEMP_W + il, (long) ubatch * 2 * nTaps);
            this.normed = scratch.temp(TEMP_NORMED + il, (long) ubatch * nEmbd);
            this.downOut = scratch.temp(TEMP_DOWN + il, (long) ubatch * nFf);
            this.taps = writeTaps(ctx, scratch.temp(TEMP_TAPS + il, nTaps), tapIdx);
            ops.prewarm(ActKernels.UNARY, new long[]{ops.local()});
            ops.prewarm(MUDD_MIX, new long[]{ops.local()});
        }

        /** Uploads the layer's tap indices into the int elements of {@code dst} for {@value #MUDD_MIX}. */
        private static ScratchSlot writeTaps(GpuForwardContext ctx, ScratchSlot dst, int[] tapIdx) {
            byte[] bytes = new byte[tapIdx.length * Integer.BYTES];
            for (int j = 0; j < tapIdx.length; j++) {
                int v = tapIdx[j];
                int b = j * Integer.BYTES;
                bytes[b] = (byte) v;
                bytes[b + 1] = (byte) (v >>> 8);
                bytes[b + 2] = (byte) (v >>> 16);
                bytes[b + 3] = (byte) (v >>> 24);
            }
            GpuEvent e = ctx.queue().enqueueWriteBuffer(dst.buffer(), (long) dst.offset() * Integer.BYTES,
                    bytes.length, bytes);
            if (e != null) {
                e.close();
            }
            return dst;
        }

        @Override
        public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
            checkStep(nTokens);
            bankRows(nTokens, x);
            ops.norm(NormKind.RMS, null, null, eps, x.buffer(), x.offset(), normed.buffer(), normed.offset(),
                    nTokens, nEmbd);
            ops.matmul(down, normed.buffer(), normed.offset(), downOut.buffer(), downOut.offset(), nTokens);
            int downLen = nTokens * nFf;
            ops.unary(ActOp.GELU_ERF, null, downOut.buffer(), downOut.offset(), downOut.buffer(), downOut.offset(),
                    downLen);
            ops.matmul(up, downOut.buffer(), downOut.offset(), w.buffer(), w.offset(), nTokens);
            int bankStride = ubatch * nEmbd;
            ops.enqueue(MUDD_MIX, new long[]{roundUp((long) nTokens * nEmbd)}, new long[]{ops.local()}, nTokens,
                    nEmbd, nTaps, bank.offset(), bankStride, w.offset(), taps.offset(), cur.offset(), resid.offset(),
                    bank.buffer(), w.buffer(), upBias.shard(0), taps.buffer(), cur.buffer(), resid.buffer());
            ops.copy(cur.buffer(), cur.offset(), x.buffer(), x.offset(), nTokens * nEmbd);
        }

        @Override
        public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
            checkStep(nTokens);
            alphaBeta(ops, placeholder, nTokens * nEmbd, resid, branch, attnScale.shard(0), ones.buffer(), 1, x);
            return true;
        }

        @Override
        public boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
            return fold(ffnScale, nTokens, x, branch);
        }

        @Override
        public void close() {
            ops = null;
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
