package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The mixture-of-experts family: FFN factories for {@link FfnKind#MOE}, {@link FfnKind#MOE_LATENT},
 * {@link FfnKind#MOE_CHUNKED} and {@link FfnKind#MOE_PARALLEL_DENSE} ({@link GpuMoeFfn}, {@link CpuMoeFfn}), the
 * {@value #LAYER_INPUT} layer extra, and the per-architecture wiring of the routed FFN ({@link Wiring}).
 */
public final class MoeBlocks {

    /**
     * Feature id of the layer extra that keeps the layer input (the residual stream before the attention norm) for
     * the MoE block of the same layer: arctic normalizes it with {@code ffn_norm_exps} as the expert input
     * (models/arctic.cpp:136-139), smallthinker routes on it (models/smallthinker.cpp:108).
     */
    public static final String LAYER_INPUT = LayerSpec.Extras.MOE_LAYER_INPUT;

    /**
     * Scratch temporary holding the layer input between {@code beforeBlock} and the FFN of one layer. Only the
     * {@value #LAYER_INPUT} extra writes it and only MoE blocks read it, so it survives the mixer of the layer. Every
     * holder requests the same model-wide size ({@link #layerInputFloats}), so all of them share one buffer.
     */
    static final String LAYER_INPUT_TEMP = "moe.layer_input";

    private static final FfnKind[] KINDS = {
            FfnKind.MOE, FfnKind.MOE_LATENT, FfnKind.MOE_CHUNKED, FfnKind.MOE_PARALLEL_DENSE};

    private MoeBlocks() {
    }

    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.emptyList();
    }

    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.emptyList();
    }

    public static List<GpuFfnFactory> gpuFfns() {
        List<GpuFfnFactory> out = new ArrayList<>(KINDS.length);
        for (FfnKind k : KINDS) {
            out.add(new GpuFfnFactory() {
                @Override
                public FfnKind kind() {
                    return k;
                }

                @Override
                public GpuFfnBlock create(LayerSpec layer) {
                    return new GpuMoeFfn(k, layer);
                }
            });
        }
        return Collections.unmodifiableList(out);
    }

    public static List<CpuFfnFactory> cpuFfns() {
        List<CpuFfnFactory> out = new ArrayList<>(KINDS.length);
        for (FfnKind k : KINDS) {
            out.add(new CpuFfnFactory() {
                @Override
                public FfnKind kind() {
                    return k;
                }

                @Override
                public CpuFfnBlock create(LayerSpec layer) {
                    return new CpuMoeFfn(k, layer);
                }
            });
        }
        return Collections.unmodifiableList(out);
    }

    public static List<GpuLayerExtraFactory> gpuExtras() {
        return Collections.singletonList(new GpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return LAYER_INPUT;
            }

            @Override
            public GpuLayerExtra create(LayerSpec layer) {
                return new GpuLayerInput();
            }
        });
    }

    public static List<CpuLayerExtraFactory> cpuExtras() {
        return Collections.singletonList(new CpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return LAYER_INPUT;
            }

            @Override
            public CpuLayerExtra create(LayerSpec layer) {
                return new CpuLayerInput();
            }
        });
    }

    // ------------------------------------------------------------------ layer checks

    /**
     * Checks that {@code layer} is a layer of {@code kind} with the MoE facets that kind needs.
     *
     * @throws IllegalArgumentException when the layer's kind differs or a facet is missing
     */
    static MoeSpec checkLayer(FfnKind kind, LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        if (layer.ffnKind() != kind) {
            throw new IllegalArgumentException("layer " + layer.il() + " has FFN kind " + layer.ffnKind()
                    + ", not " + kind);
        }
        MoeSpec moe = layer.moe();
        if (moe == null) {
            throw new IllegalArgumentException("layer " + layer.il() + " of kind " + kind + " has no MoeSpec");
        }
        boolean ok;
        switch (kind) {
            case MOE:
                ok = !moe.latent() && !moe.chunked();
                break;
            case MOE_LATENT:
                ok = moe.latent() && !moe.chunked();
                break;
            case MOE_CHUNKED:
                ok = moe.chunked() && !moe.latent();
                break;
            case MOE_PARALLEL_DENSE:
                ok = !moe.latent() && !moe.chunked() && layer.ffn() != null
                        && layer.ffn().role() == FfnSpec.Role.PARALLEL_DENSE;
                break;
            default:
                throw new IllegalArgumentException(kind + " is not a MoE kind");
        }
        if (!ok) {
            throw new IllegalArgumentException("layer " + layer.il() + " spec does not match " + kind + ": " + moe
                    + ", ffn " + layer.ffn());
        }
        return moe;
    }

    /**
     * Size of the layer-input temporary: {@code ubatch} rows of the widest residual stream of the model. The size
     * is the same for every layer, so the extra and the block of a layer are served the same scratch buffer whatever
     * order the layers bind in.
     */
    static long layerInputFloats(int ubatch, ModelSpec spec) {
        long width = 1;
        for (List<LayerSpec> stack : Arrays.asList(spec.layers(), spec.nextnLayers(), spec.encoderLayers())) {
            for (LayerSpec l : stack) {
                width = Math.max(width, l.residual().streamWidth());
            }
        }
        return Math.multiplyExact((long) ubatch, width);
    }

    // ------------------------------------------------------------------ wiring

    /** What the router multiplies by {@code ffn_gate_inp}. */
    enum RouterInput {
        /** The normalized FFN input (build_moe_ffn, llama-graph.cpp:2220). */
        FFN_INPUT,
        /** The layer input before the attention norm (smallthinker.cpp:108). */
        LAYER_INPUT,
        /** The routed experts' own normalized input (arctic: {@code ffn_norm_exps(inpSA)}, arctic.cpp:136-145). */
        EXPERT_INPUT,
        /**
         * {@code rms_norm(attn_out) / sqrt(n_embd) * ffn_gate_inp.scale}, {@code attn_out} being the residual stream
         * at the FFN (gemma4.cpp:324-328).
         */
        SCALED_RESIDUAL
    }

    /** How the router picks the experts of a token. */
    enum Selection {
        /** Per the {@link MoeSpec} (selection mode, {@code exp_probs_b}, groups, hash ids). */
        SPEC,
        /**
         * Top-k of {@code logits + exp_probs_b}, weights the gating probabilities of the unbiased logits at the picked
         * ids: kolibri1 selects before {@code build_moe_ffn} and passes the ids in without {@code exp_probs_b}
         * (models/kolibri1.cpp:134-153).
         */
        BIASED_LOGITS
    }

    /** What the routed experts read (before a latent projection). */
    enum ExpertInput {
        /** The normalized FFN input. */
        FFN_INPUT,
        /** {@code rms_norm(layer input) * ffn_norm_exps} (arctic.cpp:136-139). */
        NORM_EXPS_OF_LAYER_INPUT,
        /** {@code rms_norm(attn_out) * pre_ffw_norm_2} of the residual stream at the FFN (gemma4.cpp:318-321). */
        PRE_NORM_2_OF_RESIDUAL
    }

    /** How a parallel dense FFN joins the experts. */
    enum Combine {
        /** No parallel dense FFN. */
        NONE,
        /** {@code moe + ffn} (arctic.cpp:133-153; the residual is added by the executor). */
        SUM,
        /** {@code (ffn + moe) * sqrt(2) / 2} (grok.cpp:171-180). */
        SCALED_SUM,
        /** {@code post_ffw_norm_1(ffn) + post_ffw_norm_2(moe)} (gemma4.cpp:313-350). */
        NORMED_SUM
    }

    /** The per-architecture inputs and combine rule of a MoE layer. Immutable. */
    static final class Wiring {
        final RouterInput router;
        final Selection selection;
        final ExpertInput experts;
        final Combine combine;

        private Wiring(RouterInput router, Selection selection, ExpertInput experts, Combine combine) {
            this.router = router;
            this.selection = selection;
            this.experts = experts;
            this.combine = combine;
        }

        /** True when the block reads the {@value #LAYER_INPUT} temporary. */
        boolean needsLayerInput() {
            return router == RouterInput.LAYER_INPUT || experts == ExpertInput.NORM_EXPS_OF_LAYER_INPUT;
        }

        /** True when the block reads the residual stream {@code X} at the FFN. */
        boolean needsResidual() {
            return router == RouterInput.SCALED_RESIDUAL || experts == ExpertInput.PRE_NORM_2_OF_RESIDUAL;
        }

        /**
         * The wiring of a layer of {@code kind} in an {@code arch} model.
         *
         * @throws UnsupportedOperationException for a parallel dense FFN of an architecture whose graph has none, or
         *                                       an arctic / gemma4 MoE layer without its parallel dense FFN, or a
         *                                       layer that needs the layer input without the {@value #LAYER_INPUT}
         *                                       extra in its spec, or a kolibri1 layer whose MoE spec lacks the
         *                                       {@code exp_probs_b} selection bias or routes otherwise
         */
        static Wiring of(Arch arch, FfnKind kind, LayerSpec layer) {
            Objects.requireNonNull(arch, "arch");
            RouterInput router = RouterInput.FFN_INPUT;
            Selection selection = Selection.SPEC;
            ExpertInput experts = ExpertInput.FFN_INPUT;
            Combine combine = Combine.NONE;
            switch (arch) {
                case ARCTIC:
                    router = RouterInput.EXPERT_INPUT;
                    experts = ExpertInput.NORM_EXPS_OF_LAYER_INPUT;
                    combine = Combine.SUM;
                    break;
                case GEMMA4:
                    router = RouterInput.SCALED_RESIDUAL;
                    experts = ExpertInput.PRE_NORM_2_OF_RESIDUAL;
                    combine = Combine.NORMED_SUM;
                    break;
                case GROK:
                    combine = kind == FfnKind.MOE_PARALLEL_DENSE ? Combine.SCALED_SUM : Combine.NONE;
                    break;
                case SMALLTHINKER:
                    router = RouterInput.LAYER_INPUT;
                    break;
                case KOLIBRI1:
                    selection = Selection.BIASED_LOGITS;
                    checkBiasedLogitSelection(arch, layer);
                    break;
                default:
                    break;
            }
            boolean parallel = kind == FfnKind.MOE_PARALLEL_DENSE;
            if (parallel != (combine != Combine.NONE)) {
                throw new UnsupportedOperationException(arch.ggufName() + ": layer " + layer.il() + " of kind " + kind
                        + (parallel ? " has no parallel dense FFN in the llama.cpp graph"
                        : " lacks the parallel dense FFN its graph adds to the experts"));
            }
            Wiring w = new Wiring(router, selection, experts, combine);
            if (w.needsLayerInput() && !layer.hasExtra(LAYER_INPUT)) {
                throw new UnsupportedOperationException(arch.ggufName() + ": layer " + layer.il() + " MoE reads the"
                        + " layer input before attention, which needs the '" + LAYER_INPUT + "' layer extra in its"
                        + " LayerSpec");
            }
            return w;
        }

        /** The kolibri1 router needs {@code exp_probs_b} and plain top-k routing in the spec it reinterprets. */
        private static void checkBiasedLogitSelection(Arch arch, LayerSpec layer) {
            MoeSpec m = layer.moe();
            if (m == null || !m.probBias() || m.hashRouting() || m.chunked() || m.nGroups() > 1
                    || m.selectionMode() != MoeSpec.SelectionMode.PROBS) {
                throw new UnsupportedOperationException(arch.ggufName() + ": layer " + layer.il() + " selects experts"
                        + " by logits + exp_probs_b, which needs a MoE spec with exp_probs_b and plain top-k routing,"
                        + " got " + m);
            }
        }

        @Override
        public String toString() {
            return "Wiring{router=" + router + ", selection=" + selection + ", experts=" + experts + ", combine="
                    + combine + '}';
        }
    }

    // ------------------------------------------------------------------ layer-input extra

    /** Copies the layer input into {@value #LAYER_INPUT_TEMP} before the layer runs. */
    static final class GpuLayerInput implements GpuLayerExtra {
        private GpuForwardContext ctx;
        private ScratchSlot keep;
        private int width;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            this.ctx = Objects.requireNonNull(ctx, "ctx");
            GpuScratch scratch = ctx.scratch();
            this.width = layer.residual().streamWidth();
            this.keep = scratch.temp(LAYER_INPUT_TEMP, layerInputFloats(scratch.ubatch(), ctx.spec()));
        }

        @Override
        public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
            if (ctx == null) {
                throw new IllegalStateException("layer-input extra used before bind");
            }
            int n = Math.multiplyExact(nTokens, width);
            x.require(n);
            ctx.ops().copy(x.buffer(), x.offset(), keep.buffer(), keep.offset(), n);
        }
    }

    /** CPU twin of {@link GpuLayerInput}. */
    static final class CpuLayerInput implements CpuLayerExtra {
        private float[] keep;
        private int width;

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            CpuScratch scratch = Objects.requireNonNull(ctx, "ctx").scratch();
            this.width = layer.residual().streamWidth();
            this.keep = scratch.temp(LAYER_INPUT_TEMP,
                    Math.toIntExact(layerInputFloats(scratch.ubatch(), ctx.spec())));
        }

        @Override
        public void beforeBlock(int pos0, int n, float[] x) {
            if (keep == null) {
                throw new IllegalStateException("layer-input extra used before bind");
            }
            System.arraycopy(x, 0, keep, 0, Math.multiplyExact(n, width));
        }
    }
}
