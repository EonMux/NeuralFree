package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
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
            factory(LayerSpec.Extras.LOOP_NORM, LoopNorm::new)));

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

    // ================================================================== loop norm

    /** nanbeige: {@code x = rms(x) * output_norm} after a loop-closing slot. */
    private static final class LoopNorm implements CpuLayerExtra {
        private float[] weight;
        private float eps;
        private int width;
        private boolean active;

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
}
