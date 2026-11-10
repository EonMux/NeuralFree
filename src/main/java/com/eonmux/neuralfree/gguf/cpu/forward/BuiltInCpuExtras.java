package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.ops.ssm.CpuSsmConv;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The layer extras that belong to the forward path itself rather than to an op family, the twin of
 * {@code BuiltInGpuExtras}; the CPU block catalog registers {@link #cpuExtras()}.
 * <ul>
 *   <li>{@link LayerSpec.Extras#EMBD_SKIP} (talkie): after the layer, {@code x += inp * out_scale * s} where
 *       {@code inp} is the model input ({@code CpuScratch.INPUT_EMBD}, written by {@code CpuEmbeddingStage}),
 *       {@code out_scale} the layer's {@code layer_output_scale} (1 when absent) and {@code s}
 *       {@link ResidualSpec#embeddingSkipScale()} (models/talkie.cpp:53, 123-125);</li>
 *   <li>{@link LayerSpec.Extras#RESID_ALPHA_BETA} (muse-spark): each sub-block's residual becomes
 *       {@code x = x * alpha + branch * beta} with {@code attn_resid_alpha/beta} and {@code ffn_resid_alpha/beta}
 *       (models/muse-spark.cpp:166-168, 208-210); the hook consumes the residual;</li>
 *   <li>{@link LayerSpec.Extras#LOOP_NORM} (nanbeige): after a slot that closes a loop pass
 *       ({@link LayerSchedule#normAfterSlot(int)}), {@code x = rms(x) * output_norm} with {@code f_norm_rms_eps}
 *       (models/nanbeige.cpp:163-170).</li>
 * </ul>
 */
public final class BuiltInCpuExtras {

    /** Elements per parallel task of the element-wise loops. */
    private static final int ELEMENT_GRAIN = 1 << 15;

    private static final List<CpuLayerExtraFactory> FACTORIES = Collections.unmodifiableList(Arrays.asList(
            factory(LayerSpec.Extras.EMBD_SKIP, EmbeddingSkip::new),
            factory(LayerSpec.Extras.RESID_ALPHA_BETA, AlphaBetaResidual::new),
            factory(LayerSpec.Extras.RESID_SCALE, ResidScale::new),
            factory(LayerSpec.Extras.MUDD, Mudd::new),
            factory(LayerSpec.Extras.LOOP_NORM, LoopNorm::new),
            factory(LayerSpec.Extras.INKLING_SCONV, InklingSconv::new)));

    private BuiltInCpuExtras() {
    }

    /** Factories of the built-in extras, one per feature id (unmodifiable). */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return FACTORIES;
    }

    /** Supplier of unbound extras. */
    private interface ExtraSupplier {
        CpuLayerExtra get();
    }

    private static CpuLayerExtraFactory factory(String id, ExtraSupplier supplier) {
        return new CpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return id;
            }

            @Override
            public CpuLayerExtra create(LayerSpec layer) {
                Objects.requireNonNull(layer, "layer");
                return supplier.get();
            }

            @Override
            public String toString() {
                return "BuiltInCpuExtras[" + id + "]";
            }
        };
    }

    private static String describe(CpuForwardContext ctx, LayerSpec layer) {
        return ctx.spec().arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
    }

    private static void requireUnbound(Object state, CpuForwardContext ctx, LayerSpec layer, String id) {
        if (state != null) {
            throw new IllegalStateException("'" + id + "' extra of " + describe(ctx, layer) + " is already bound");
        }
    }

    // ================================================================== embedding skip

    /** talkie: {@code x += inp * out_scale * embeddingSkipScale} after the layer. */
    private static final class EmbeddingSkip implements CpuLayerExtra {
        private CpuScratch scratch;
        private float[] outScale;
        private float skipScale;
        private int width;
        private int inputWidth;

        /** The add is a per-row product with the model input, exact over any step layout. */
        @Override
        public String batchRefusal() {
            return null;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            Objects.requireNonNull(ctx, "ctx");
            Objects.requireNonNull(layer, "layer");
            requireUnbound(scratch, ctx, layer, LayerSpec.Extras.EMBD_SKIP);
            ResidualSpec res = layer.residual();
            if (!res.embeddingSkip()) {
                throw new IllegalStateException(describe(ctx, layer) + " lists '" + LayerSpec.Extras.EMBD_SKIP
                        + "' but its residual has no embedding skip");
            }
            EmbeddingSpec embd = ctx.spec().embedding();
            int nEmbd = ctx.spec().hparams().nEmbd();
            if (res.streamWidth() != nEmbd || embd == null) {
                throw new IllegalStateException(describe(ctx, layer) + ": the embedding skip needs an "
                        + nEmbd + "-wide stream and an embedding stage");
            }
            float[] s = ctx.weights().layerFloatsOrNull(layer, TensorNames.weight(TensorNames.LAYER_OUT_SCALE));
            if (s != null && s.length != 1 && s.length != nEmbd) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": layer_output_scale has " + s.length
                        + " values, expected 1 or " + nEmbd);
            }
            this.outScale = s;
            this.skipScale = res.embeddingSkipScale();
            this.width = nEmbd;
            this.inputWidth = Math.max(embd.nEmbdInp(), embd.nEmbd());
            this.scratch = ctx.scratch();
        }

        @Override
        public void afterBlock(int pos0, int n, float[] x) {
            if (scratch == null) {
                throw new IllegalStateException("embedding-skip extra used before bind()");
            }
            float[] inp = scratch.slot(CpuScratch.INPUT_EMBD);
            int w = width;
            int iw = inputWidth;
            float[] os = outScale;
            float ss = skipScale;
            CpuParallel.parallelFor(n, Math.max(1, ELEMENT_GRAIN / w), (from, to) -> {
                for (int t = from; t < to; t++) {
                    int xb = t * w;
                    int ib = t * iw;
                    for (int i = 0; i < w; i++) {
                        float skip = inp[ib + i];
                        if (os != null) {
                            skip *= os.length == 1 ? os[0] : os[i];
                        }
                        if (ss != 1.0f) {
                            skip *= ss;
                        }
                        x[xb + i] += skip;
                    }
                }
            });
        }
    }

    // ================================================================== alpha / beta residual

    /** muse-spark: {@code x = x * alpha + branch * beta} per sub-block; consumes the residual. */
    private static final class AlphaBetaResidual implements CpuLayerExtra {
        private float[] attnAlpha;
        private float[] attnBeta;
        private float[] ffnAlpha;
        private float[] ffnBeta;
        private int width;
        private boolean bound;

        /** The mix is a per-row product of x and the branch, exact over any step layout. */
        @Override
        public String batchRefusal() {
            return null;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            Objects.requireNonNull(ctx, "ctx");
            Objects.requireNonNull(layer, "layer");
            requireUnbound(bound ? this : null, ctx, layer, LayerSpec.Extras.RESID_ALPHA_BETA);
            ResidualSpec res = layer.residual();
            if (!res.alphaBeta()) {
                throw new IllegalStateException(describe(ctx, layer) + " lists '"
                        + LayerSpec.Extras.RESID_ALPHA_BETA + "' but its residual has no alpha/beta tensors");
            }
            this.width = res.streamWidth() / res.hcMult();
            CpuWeights w = ctx.weights();
            this.attnAlpha = vector(w, res.attnAlpha(), ctx, layer);
            this.attnBeta = vector(w, res.attnBeta(), ctx, layer);
            this.ffnAlpha = vector(w, res.ffnAlpha(), ctx, layer);
            this.ffnBeta = vector(w, res.ffnBeta(), ctx, layer);
            if ((attnAlpha == null) != (attnBeta == null) || (ffnAlpha == null) != (ffnBeta == null)) {
                throw new IllegalStateException(describe(ctx, layer) + ": alpha/beta residual tensors come in pairs");
            }
            if ((attnAlpha != null && attnAlpha.length != attnBeta.length)
                    || (ffnAlpha != null && ffnAlpha.length != ffnBeta.length)) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": alpha and beta lengths differ");
            }
            this.bound = true;
        }

        private float[] vector(CpuWeights weights, TensorRef ref, CpuForwardContext ctx, LayerSpec layer) {
            if (ref == null) {
                return null;
            }
            float[] v = weights.floats(ref);
            if (v.length != 1 && v.length != width) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": " + ref.name() + " has " + v.length
                        + " values, expected 1 or " + width);
            }
            return v;
        }

        @Override
        public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
            return apply(n, x, branch, attnAlpha, attnBeta);
        }

        @Override
        public boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
            return apply(n, x, branch, ffnAlpha, ffnBeta);
        }

        private boolean apply(int n, float[] x, float[] branch, float[] alpha, float[] beta) {
            if (!bound) {
                throw new IllegalStateException("alpha/beta residual extra used before bind()");
            }
            if (alpha == null) {
                return false;
            }
            CpuLayerExecutor.mixRows(MixMode.ALPHA_BETA, 0f, n, width, x, branch, alpha, beta, alpha.length, x);
            return true;
        }
    }

    // ================================================================== limite

    /** Common state of the limite extras: the bank of every trunk layer's input and the residual scales. */
    abstract static class LimiteResid implements CpuLayerExtra {
        /** Shared bank of the trunk layer inputs; slot il holds layer il's input of the current step. */
        static final String TEMP_BANK = "builtin.limite.bank";

        CpuScratch scratch;
        int nEmbd;
        int il;
        float attnScale;
        float ffnScale;
        float[] bank;
        boolean bound;

        /** Every hook folds one step row at a time over the bank, exact over any step layout. */
        @Override
        public String batchRefusal() {
            return null;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            Objects.requireNonNull(ctx, "ctx");
            Objects.requireNonNull(layer, "layer");
            requireUnbound(bound ? this : null, ctx, layer, featureId());
            CpuWeights w = ctx.weights();
            this.attnScale = scalar(w, ctx, layer, TensorNames.ATTN_RESID_SCALE);
            this.ffnScale = scalar(w, ctx, layer, TensorNames.FFN_RESID_SCALE);
            this.nEmbd = ctx.spec().hparams().nEmbd();
            this.il = layer.il();
            this.scratch = ctx.scratch();
            long bankFloats = (long) ctx.spec().nLayer() * scratch.ubatch() * nEmbd;
            if (bankFloats > Integer.MAX_VALUE - 8) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": the layer bank of " + bankFloats
                        + " floats exceeds a Java array; lower neuralfree.ubatch");
            }
            this.bank = scratch.temp(TEMP_BANK, (int) bankFloats);
            this.bound = true;
        }

        /** Feature id of the concrete extra, for the double-bind check. */
        abstract String featureId();

        private float scalar(CpuWeights w, CpuForwardContext ctx, LayerSpec layer, String suffix) {
            float[] v = w.layerFloatsOrNull(layer, TensorNames.weight(suffix));
            if (v == null) {
                throw new MissingTensorException(ctx.spec().arch().ggufName(), layer.il(), TensorNames.weight(suffix));
            }
            if (v.length != 1) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": " + TensorNames.weight(suffix) + " has "
                        + v.length + " values, expected 1");
            }
            return v[0];
        }

        /** Banks the layer input; every limite layer runs this before its block. */
        void bank(int n, float[] x) {
            if (!bound) {
                throw new IllegalStateException(featureId() + " extra used before bind()");
            }
            if (n < 1 || n > scratch.ubatch()) {
                throw new IllegalArgumentException("layer " + il + ": step of " + n + " tokens outside [1, "
                        + scratch.ubatch() + "]");
            }
            if (x.length < n * nEmbd) {
                throw new IllegalArgumentException("layer " + il + ": the stream holds " + x.length
                        + " floats, the step needs " + (n * nEmbd));
            }
            System.arraycopy(x, 0, bank, il * scratch.ubatch() * nEmbd, n * nEmbd);
        }

        /** {@code x[i] = x[i] * scale + branch[i]}. */
        void fold(int n, float[] x, float[] branch, float scale) {
            int len = n * nEmbd;
            for (int i = 0; i < len; i++) {
                x[i] = x[i] * scale + branch[i];
            }
        }
    }

    /** limite without MUDD: the learned scales fold the plain residual and the FFN input. */
    static final class ResidScale extends LimiteResid {
        @Override
        String featureId() {
            return LayerSpec.Extras.RESID_SCALE;
        }

        @Override
        public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
            fold(n, x, branch, attnScale);
            return true;
        }

        @Override
        public boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
            fold(n, x, branch, ffnScale);
            return true;
        }
    }

    /**
     * limite MUDD (limite.cpp:100-140): the attention input and the residual of the layer become weighted sums of
     * the banked inputs of its taps, {@code w = mudd_up @ gelu_erf(mudd_down @ rms(x)) + mudd_up_b} holding one
     * weight per tap for each mix. The mixer reads the attention mix through the stream, which {@code beforeBlock}
     * rewrites; the residual mix rides a per-layer scratch temporary into {@code afterMixer}.
     */
    static final class Mudd extends LimiteResid {
        private static final String TEMP_CUR = "builtin.limite.cur.";
        private static final String TEMP_RESID = "builtin.limite.resid.";
        private static final String TEMP_W = "builtin.limite.w.";
        private static final String TEMP_NORMED = "builtin.limite.normed.";
        private static final String TEMP_DOWN = "builtin.limite.down.";

        CpuMatVec matVec;
        CpuTensorRef down;
        float[] up;
        float[] upBias;
        int[] taps;
        int nTaps;
        int nFf;
        float eps;
        float[] cur;
        float[] resid;
        float[] w;
        float[] normed;
        float[] downOut;

        @Override
        String featureId() {
            return LayerSpec.Extras.MUDD;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            super.bind(ctx, layer);
            ModelHparams hp = ctx.spec().hparams();
            CpuWeights weights = ctx.weights();
            String arch = ctx.spec().arch().ggufName();
            this.matVec = ctx.matVec();
            this.nTaps = hp.muddTapCount();
            if (nTaps <= 0) {
                throw new IllegalStateException(describe(ctx, layer) + " lists '" + LayerSpec.Extras.MUDD
                        + "' but the file sets no mudd.tap_count");
            }
            this.taps = new int[nTaps];
            for (int j = 0; j < nTaps; j++) {
                taps[j] = hp.muddTapIdx(il, j);
                if (taps[j] < 0 || taps[j] > il) {
                    throw new IllegalStateException(describe(ctx, layer) + ": every MUDD tap of a mixing layer must"
                            + " be in [0, " + il + "], got " + java.util.Arrays.toString(taps) + " (limite.cpp:120)");
                }
            }
            this.down = weights.layerOrNull(layer, TensorNames.weight(TensorNames.MUDD_DOWN));
            CpuTensorRef upRef = weights.layerOrNull(layer, TensorNames.weight(TensorNames.MUDD_UP));
            if (down == null || upRef == null) {
                throw new MissingTensorException(arch, il, down == null ? TensorNames.weight(TensorNames.MUDD_DOWN)
                        : TensorNames.weight(TensorNames.MUDD_UP));
            }
            this.nFf = (int) down.rows();
            if (down.cols() != nEmbd || upRef.rows() != 2L * nTaps || upRef.cols() != nFf) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": mudd_down must be [" + nEmbd + ", "
                        + nFf + "] and mudd_up [" + (2 * nTaps) + ", " + nFf + "], got [" + down.cols() + ", "
                        + down.rows() + "] and [" + upRef.cols() + ", " + upRef.rows() + "]");
            }
            this.upBias = weights.layerFloatsOrNull(layer, TensorNames.bias(TensorNames.MUDD_UP));
            if (upBias == null || upBias.length != 2L * nTaps) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": " + TensorNames.bias(TensorNames.MUDD_UP)
                        + " must hold " + (2 * nTaps) + " values, got " + (upBias == null ? 0 : upBias.length));
            }
            // the mixer weights are small: decode them once, every step reuses the values
            this.up = new float[(int) (2L * nTaps * nFf)];
            for (int r = 0; r < 2 * nTaps; r++) {
                upRef.decodeRow(r, up, r * nFf);
            }
            this.eps = hp.normRmsEps();
            this.cur = scratch.temp(TEMP_CUR + il, (int) (scratch.ubatch()) * nEmbd);
            this.resid = scratch.temp(TEMP_RESID + il, (int) (scratch.ubatch()) * nEmbd);
            this.w = scratch.temp(TEMP_W + il, (int) (scratch.ubatch()) * 2 * nTaps);
            this.normed = scratch.temp(TEMP_NORMED + il, (int) (scratch.ubatch()) * nEmbd);
            this.downOut = scratch.temp(TEMP_DOWN + il, (int) (scratch.ubatch()) * nFf);
        }

        @Override
        public void beforeBlock(int pos0, int n, float[] x) {
            bank(n, x);
            NormMath.rowNorm(NormKind.RMS, eps, n, nEmbd, nEmbd, 1, x, 0, null, null, normed, 0);
            matVec.matmul(down, normed, 0, downOut, 0, n);
            mixTaps(n);
            // the mixer reads the attention mix through the stream
            System.arraycopy(cur, 0, x, 0, n * nEmbd);
        }

        /** {@code w[t] = mudd_up @ gelu_erf(downOut[t]) + mudd_up_b}, then the two tap mixes. */
        private void mixTaps(int n) {
            CpuParallel.parallelFor(n, 1, (from, to) -> {
                for (int t = from; t < to; t++) {
                    int fBase = t * nFf;
                    for (int c = 0; c < nFf; c++) {
                        downOut[fBase + c] = ActMath.geluErf(downOut[fBase + c]);
                    }
                    int wBase = t * 2 * nTaps;
                    for (int j = 0; j < 2 * nTaps; j++) {
                        float acc = upBias[j];
                        int uBase = j * nFf;
                        for (int c = 0; c < nFf; c++) {
                            acc += up[uBase + c] * downOut[fBase + c];
                        }
                        w[wBase + j] = acc;
                    }
                    MuddMath.mixRow(bank, 0, scratch.ubatch() * nEmbd, taps, t, nEmbd, w, wBase, cur, resid,
                            t * nEmbd);
                }
            });
        }

        @Override
        public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
            if (!bound) {
                throw new IllegalStateException(featureId() + " extra used before bind()");
            }
            int len = n * nEmbd;
            for (int i = 0; i < len; i++) {
                x[i] = resid[i] * attnScale + branch[i];
            }
            return true;
        }

        @Override
        public boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
            fold(n, x, branch, ffnScale);
            return true;
        }
    }

    // ================================================================== loop norm

    /** nanbeige: {@code x = rms(x) * output_norm} after a loop-closing slot. */
    private static final class LoopNorm implements CpuLayerExtra {
        private float[] weight;
        private float eps;
        private int width;
        private boolean active;

        /** The norm is a per-row product with the weight, exact over any step layout. */
        @Override
        public String batchRefusal() {
            return null;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            Objects.requireNonNull(ctx, "ctx");
            Objects.requireNonNull(layer, "layer");
            requireUnbound(weight, ctx, layer, LayerSpec.Extras.LOOP_NORM);
            HeadSpec head = ctx.spec().head();
            TensorRef ref = head != null ? head.outputNorm() : null;
            if (ref == null) {
                throw new IllegalStateException(describe(ctx, layer) + ": the loop norm needs output_norm");
            }
            float[] w = ctx.weights().floats(ref);
            int streamWidth = layer.residual().streamWidth();
            if (w.length != streamWidth) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": output_norm has " + w.length
                        + " values, expected " + streamWidth);
            }
            LayerSchedule schedule = ctx.spec().schedule();
            this.active = layer.il() < schedule.slots() && schedule.normAfterSlot(layer.il());
            this.eps = ctx.spec().hparams().normRmsEps();
            this.width = streamWidth;
            this.weight = w;
        }

        @Override
        public void afterBlock(int pos0, int n, float[] x) {
            if (weight == null) {
                throw new IllegalStateException("loop-norm extra used before bind()");
            }
            if (active) {
                CpuOps.norm(NormKind.RMS, eps, x, 0, x, 0, n, width, weight, null);
            }
        }
    }

    // ================================================================== Inkling short convolutions

    /**
     * Inkling's two residual-stream short convolutions (vLLM {@code _sconv_add_norm}): after the mixer and after the
     * FFN the branch passes through its layer's conv state and adds back into the residual stream, once with the
     * built-in residual of the convolution ({@code h = hidden + conv(branch) + branch}). The state row holds the K/V
     * streams of the attention mixer and both of these streams (vLLM's {@code InklingSconvMetadata} per layer).
     */
    private static final class InklingSconv implements CpuLayerExtra {
        private float[] attnW;
        private float[] ffnW;
        private float[] convState;
        private float[] tmp;
        private int convRow;
        private int stateSlots;
        private int nEmbd;
        private int width;
        private int attnOff;
        private int mlpOff;
        /** {@code ffn_global_scale}: the dense MLP's output multiplier, 1 without one. */
        private float ffnScale = 1.0f;

        /** The convolutions are chronological state updates, exact over any segment layout. */
        @Override
        public String batchRefusal() {
            return null;
        }

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            Objects.requireNonNull(ctx, "ctx");
            Objects.requireNonNull(layer, "layer");
            requireUnbound(convState, ctx, layer, LayerSpec.Extras.INKLING_SCONV);
            RecurrentSpec r = layer.recurrent();
            if (r == null || r.kind() != MixerKind.INKLING) {
                throw new IllegalStateException(describe(ctx, layer) + " lists '"
                        + LayerSpec.Extras.INKLING_SCONV + "' but its recurrent mixer is not Inkling");
            }
            int nEmbd = ctx.spec().hparams().nEmbd();
            if (layer.residual().streamWidth() != nEmbd) {
                throw new IllegalStateException(describe(ctx, layer) + ": the short convolutions expect a stream of "
                        + nEmbd + " floats, the layer streams " + layer.residual().streamWidth());
            }
            if (!ctx.hasMemory()) {
                throw new IllegalStateException(describe(ctx, layer) + " needs memory for its conv state");
            }
            RecurrentStateAccessor state = ctx.memory().recurrent(layer.il());
            if (state == null) {
                throw new IllegalStateException(describe(ctx, layer) + " has no recurrent state");
            }
            this.width = r.dConv();
            this.nEmbd = nEmbd;
            this.attnW = convWeights(ctx, layer, TensorNames.ATTN_SCONV, nEmbd, width);
            this.ffnW = convWeights(ctx, layer, TensorNames.FFN_SCONV, nEmbd, width);
            this.convState = state.tensor(LayerMemoryPlan.STATE_CONV);
            this.convRow = state.slotFloats(LayerMemoryPlan.STATE_CONV);
            this.stateSlots = state.slots();
            this.ffnScale = ffnScale(ctx, layer);
            this.attnOff = r.inklingStreamOffset(RecurrentSpec.INKLING_ATTN);
            this.mlpOff = r.inklingStreamOffset(RecurrentSpec.INKLING_MLP);
            this.tmp = ctx.scratch().temp("inkling.sconv." + layer.il(), ctx.scratch().ubatch() * nEmbd);
        }

        /** The dense MLP's output multiplier ({@code ffn_global_scale}), or 1 when the layer has none. */
        private static float ffnScale(CpuForwardContext ctx, LayerSpec layer) {
            float[] v = ctx.weights().layerFloatsOrNull(layer, TensorNames.weight(TensorNames.FFN_GLOBAL_SCALE));
            if (v == null) {
                return 1.0f;
            }
            if (v.length != 1) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": "
                        + TensorNames.weight(TensorNames.FFN_GLOBAL_SCALE) + " holds " + v.length
                        + " values, expected 1");
            }
            return v[0];
        }

        private static float[] convWeights(CpuForwardContext ctx, LayerSpec layer, String base, int nEmbd, int width) {
            float[] w = ctx.weights().layerFloatsOrNull(layer, TensorNames.weight(base));
            if (w == null) {
                throw new MissingTensorException(ctx.spec().arch().ggufName(), layer.il(), TensorNames.weight(base));
            }
            if (w.length != (long) nEmbd * width) {
                throw new IllegalArgumentException(describe(ctx, layer) + ": " + TensorNames.weight(base)
                        + " holds " + w.length + " values, expected " + nEmbd + " x " + width);
            }
            return w;
        }

        @Override
        public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
            apply(0, n, x, branch, attnW, attnOff, 0, 1.0f);
            return true;
        }

        @Override
        public boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
            apply(0, n, x, branch, ffnW, mlpOff, 0, ffnScale);
            return true;
        }

        @Override
        public boolean afterMixer(StepLayout s, float[] x, float[] branch) {
            forEachSegment(s, (seq, start, len) -> apply(seq, len, x, branch, attnW, attnOff, start, 1.0f));
            return true;
        }

        @Override
        public boolean afterFfn(StepLayout s, float[] x, float[] branch) {
            forEachSegment(s, (seq, start, len) -> apply(seq, len, x, branch, ffnW, mlpOff, start, ffnScale));
            return true;
        }

        private void forEachSegment(StepLayout s, SegmentHook hook) {
            if (convState == null) {
                throw new IllegalStateException("Inkling short-convolution extra used before bind()");
            }
            if (s.contiguousSingle()) {
                hook.run(0, 0, s.nTokens());
                return;
            }
            for (int seg = 0; seg < s.nSeg(); seg++) {
                hook.run(s.seq(seg), s.tokStart(seg), s.len(seg));
            }
        }

        /** Runs one segment through its sequence's state row. */
        @FunctionalInterface
        private interface SegmentHook {
            void run(int seq, int start, int len);
        }

        private void apply(int seq, int len, float[] x, float[] branch, float[] weight, int stateOff, int start,
                           float scale) {
            if (len < 1) {
                return;
            }
            if (seq < 0 || seq >= stateSlots) {
                throw new UnsupportedOperationException("Inkling conv sequence " + seq + ", the memory keeps "
                        + stateSlots + " state slots");
            }
            if (scale != 1.0f) {
                /* The MLP's global scale applies to its output before the convolution reads it (vLLM's DenseMLP). */
                int from = start * nEmbd;
                int to = from + len * nEmbd;
                for (int i = from; i < to; i++) {
                    branch[i] *= scale;
                }
            }
            CpuSsmConv.forward(nEmbd, width, len, 0, branch, start * nEmbd, nEmbd, weight, null, convState,
                    seq * convRow, stateOff, tmp, 0, nEmbd);
            int end = (start + len) * nEmbd;
            for (int i = start * nEmbd, j = 0; i < end; i++, j++) {
                x[i] += branch[i] + tmp[j];
            }
        }
    }
}
