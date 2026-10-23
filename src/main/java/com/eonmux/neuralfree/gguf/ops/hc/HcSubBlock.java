package com.eonmux.neuralfree.gguf.ops.hc;

import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.CoefficientParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PostParams;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels.PreParams;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.List;
import java.util.Objects;

/**
 * One hyper-connection collapse site resolved from the model's tensors: the mixer or FFN sub-block of a layer
 * (HC wraps each sub-block with its own weights) or the output head. The variant follows the tensors the
 * llama.cpp graphs read:
 * <ul>
 *   <li>{@link Kind#MIXING} (mHC; deepseek4.cpp:353-446, glm5-next.cpp:450-540, dflash.cpp:1200-1249):
 *       {@code hc_*_fn} has {@code (2 + hc)*hc} rows; {@code mixes = fn . rms(flatten X)}, pre / post / Sinkhorn
 *       comb from {@code hc_*_scale} (3) and {@code hc_*_base}; post mixes the streams through comb;</li>
 *   <li>{@link Kind#IDENTITY} (iHC; hy-v4.cpp:175-245): {@code hc_*_fn} has {@code 2*hc} rows; post is
 *       {@code hc_magnitude*sigmoid(.) + eps} and each stream keeps its own residual;</li>
 *   <li>{@link Kind#GATED} (qwen4exp.cpp:311-395): {@code hc_*_inject} present; the streams are grouped-RMS-normed
 *       with {@code hc_*_norm} ({@code [n_embd, hc]}), collapsed by
 *       {@code mean_h(xn * sigmoid(up . silu(down . xn / hc)))} and re-expanded with
 *       {@code 2*sigmoid((inject . xn) / hc)};</li>
 *   <li>{@link Kind#HEAD_WEIGHTS} (deepseek4.cpp:448-466, hy-v4.cpp:248-270, dflash): {@code output_hc_fn} with
 *       {@code hc} rows, pre only;</li>
 *   <li>{@link Kind#HEAD_GATED} (qwen4exp.cpp:503-511): the gated collapse with {@code output_hc_norm/down/up} and
 *       no inject; it is also the final norm (qwen4exp has no {@code output_norm});</li>
 *   <li>{@link Kind#HEAD_MEAN} (glm5-next.cpp:791): the plain stream mean.</li>
 * </ul>
 * The flattened-stream RMS norm and the grouped norm use {@code norm_rms_eps}; the coefficient eps is
 * {@code dsv4_hc_eps}. Immutable.
 */
public final class HcSubBlock {

    /** Scratch temporary: saved residual stream of the current layer ({@code ubatch * hc * n_embd}). */
    static final String TEMP_STREAM = "hc.stream";
    /** Scratch temporary: normalized stream ({@code ubatch * hc * n_embd}). */
    static final String TEMP_FLAT = "hc.flat";
    /** Scratch temporary: projected mixes ({@code ubatch * mixDim}). */
    static final String TEMP_MIXES = "hc.mixes";
    /** Scratch temporary: pre coefficients ({@code ubatch * hc}). */
    static final String TEMP_PRE = "hc.pre";
    /** Scratch temporary: post coefficients or qwen4exp inject logits ({@code ubatch * hc}). */
    static final String TEMP_POST = "hc.post";
    /** Scratch temporary: Sinkhorn comb ({@code ubatch * hc * hc}). */
    static final String TEMP_COMB = "hc.comb";
    /** Scratch temporary: low-rank gate activations ({@code ubatch * rank}). */
    static final String TEMP_LOW_RANK = "hc.lowrank";
    /** Scratch temporary: per-element gate logits ({@code ubatch * hc * n_embd}). */
    static final String TEMP_GATE = "hc.gate";
    /** Scratch temporary: collapsed head rows ({@code ubatch * n_embd}). */
    static final String TEMP_COLLAPSED = "hc.collapsed";
    /** Scratch temporary: kimi-k3 checkpoint stack, shared by every layer of one step. */
    static final String TEMP_RES_STACK = "hc.res_stack";
    /** Scratch temporary: kimi-k3 running prefix sum of the current layer ({@code ubatch * n_embd}). */
    static final String TEMP_RES_PREFIX = "hc.res_prefix";
    /** Scratch temporary: kimi-k3 mixing probabilities ({@code ubatch * (n_ckpt + 1)}). */
    static final String TEMP_RES_PROBS = "hc.res_probs";

    /** Variant of a collapse site. */
    public enum Kind {
        /** mHC sub-block: pre, {@code 2*sigmoid} post and Sinkhorn comb. */
        MIXING(false, true, true),
        /** iHC sub-block: pre and magnitude post, streams keep their own residual. */
        IDENTITY(false, true, true),
        /** qwen4exp sub-block: grouped norm, low-rank gate, inject post. */
        GATED(false, false, true),
        /** HC head from {@code output_hc_fn/scale/base}: pre only. */
        HEAD_WEIGHTS(true, true, false),
        /** qwen4exp head: grouped norm and low-rank gate, no inject. */
        HEAD_GATED(true, false, false),
        /** Stream mean. */
        HEAD_MEAN(true, false, false);

        private final boolean head;
        private final boolean projected;
        private final boolean expands;

        Kind(boolean head, boolean projected, boolean expands) {
            this.head = head;
            this.projected = projected;
            this.expands = expands;
        }

        /** True for the output-head variants. */
        public boolean head() { return head; }

        /** True when the coefficients come from {@code fn . rms(flatten X)} and {@code LLM_HC_PrePostComb}. */
        public boolean projected() { return projected; }

        /** True when the site has a post map (a sub-block, not the head). */
        public boolean expands() { return expands; }

        /** True for the grouped-norm low-rank gate variants. */
        public boolean gated() { return this == GATED || this == HEAD_GATED; }
    }

    private final Kind kind;
    private final String site;
    private final int il;
    private final int hc;
    private final int nEmbd;
    private final float normEps;
    private final TensorRef fn;
    private final TensorRef scale;
    private final TensorRef base;
    private final TensorRef norm;
    private final TensorRef down;
    private final TensorRef up;
    private final TensorRef inject;
    private final CoefficientParams coefficients;
    private final PreParams pre;
    private final PostParams post;

    private HcSubBlock(Kind kind, String site, int il, int hc, int nEmbd, float normEps, TensorRef fn,
                       TensorRef scale, TensorRef base, TensorRef norm, TensorRef down, TensorRef up, TensorRef inject,
                       CoefficientParams coefficients, PreParams pre, PostParams post) {
        this.kind = kind;
        this.site = site;
        this.il = il;
        this.hc = hc;
        this.nEmbd = nEmbd;
        this.normEps = normEps;
        this.fn = fn;
        this.scale = scale;
        this.base = base;
        this.norm = norm;
        this.down = down;
        this.up = up;
        this.inject = inject;
        this.coefficients = coefficients;
        this.pre = pre;
        this.post = post;
    }

    /**
     * The mixer ({@code ffn == false}) or FFN sub-block site of a hyper-connection layer, from the layer's
     * {@code hc_attn_*} or {@code hc_ffn_*} tensors at its weight layer.
     *
     * @throws UnsupportedOperationException when the layer's residual is not per-sub-block hyper-connections
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor the variant needs is absent
     * @throws IllegalArgumentException      when a tensor shape or a hyper-connection hparam does not fit
     */
    public static HcSubBlock forLayer(ModelSpec spec, LayerSpec layer, boolean ffn) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(layer, "layer");
        ModelHparams hp = spec.hparams();
        String arch = spec.arch().ggufName();
        ResidualSpec r = layer.residual();
        if (r.kind() != ResidualKind.HYPER_CONNECTIONS || !r.hcPerSubBlock()) {
            throw new UnsupportedOperationException(arch + " layer " + layer.il() + ": the hyper-connection extra needs"
                    + " per-sub-block hyper-connections, got " + r);
        }
        int hc = r.hcMult();
        HcKernels.validateStreams(hc);
        int nEmbd = r.streamWidth() / hc;
        String site = ffn ? "ffn" : "attn";
        TensorCatalog tc = spec.tensors();
        String stack = layer.stack();
        int wl = layer.weightLayer();
        TensorRef inject = tc.layer(stack, wl, TensorNames.weight(ffn ? TensorNames.HC_FFN_INJECT
                : TensorNames.HC_ATTN_INJECT));
        if (inject != null) {
            TensorRef norm = tc.require(stack, wl, TensorNames.weight(ffn ? TensorNames.HC_FFN_NORM
                    : TensorNames.HC_ATTN_NORM));
            TensorRef down = tc.require(stack, wl, TensorNames.weight(ffn ? TensorNames.HC_FFN_DOWN
                    : TensorNames.HC_ATTN_DOWN));
            TensorRef up = tc.require(stack, wl, TensorNames.weight(ffn ? TensorNames.HC_FFN_UP
                    : TensorNames.HC_ATTN_UP));
            checkGated(arch, layer.il(), hc, nEmbd, norm, down, up);
            requireShape(arch, layer.il(), inject, (long) hc * nEmbd, hc);
            return new HcSubBlock(Kind.GATED, site, layer.il(), hc, nEmbd, hp.normRmsEps(), null, null, null, norm,
                    down, up, inject, null, PreParams.gated(hc), PostParams.gatedInject(hc));
        }
        TensorRef fn = tc.require(stack, wl, TensorNames.weight(ffn ? TensorNames.HC_FFN_FN : TensorNames.HC_ATTN_FN));
        TensorRef scale = tc.require(stack, wl, TensorNames.weight(ffn ? TensorNames.HC_FFN_SCALE
                : TensorNames.HC_ATTN_SCALE));
        TensorRef base = tc.require(stack, wl, TensorNames.weight(ffn ? TensorNames.HC_FFN_BASE
                : TensorNames.HC_ATTN_BASE));
        long mixRows = fn.rows();
        Kind kind;
        CoefficientParams coef;
        PostParams post;
        if (mixRows == HcKernels.mixDim(hc, HcKernels.OUT_PRE_POST_COMB)) {
            kind = Kind.MIXING;
            coef = CoefficientParams.mhc(hc, hp.dsv4HcEps(), hp.dsv4HcSinkhornIters());
            post = PostParams.mixing();
        } else if (mixRows == HcKernels.mixDim(hc, HcKernels.OUT_PRE_POST)) {
            kind = Kind.IDENTITY;
            coef = CoefficientParams.ihc(hc, hp.dsv4HcEps(), hp.hcMagnitude());
            post = PostParams.identity();
        } else {
            throw new IllegalArgumentException(arch + " layer " + layer.il() + ": " + fn.name() + " has " + mixRows
                    + " rows; hyper-connections with " + hc + " streams need " + HcKernels.mixDim(hc,
                    HcKernels.OUT_PRE_POST_COMB) + " (mHC) or " + HcKernels.mixDim(hc, HcKernels.OUT_PRE_POST)
                    + " (iHC)");
        }
        checkProjected(arch, layer.il(), hc, nEmbd, fn, scale, base, coef);
        return new HcSubBlock(kind, site, layer.il(), hc, nEmbd, hp.normRmsEps(), fn, scale, base, null, null, null,
                null, coef, PreParams.weights(), post);
    }

    /**
     * True when the model's final hidden rows are hyper-connection streams that the head must collapse: the last
     * trunk layer has a hyper-connection residual.
     */
    public static boolean headApplies(ModelSpec spec) {
        List<LayerSpec> layers = spec.layers();
        return !layers.isEmpty() && layers.get(layers.size() - 1).residual().kind() == ResidualKind.HYPER_CONNECTIONS;
    }

    /**
     * The output-head collapse: {@link HeadSpec#hcHead()} selects {@link Kind#HEAD_WEIGHTS}; otherwise
     * {@code output_hc_down} selects {@link Kind#HEAD_GATED}; otherwise {@link Kind#HEAD_MEAN}.
     *
     * @throws IllegalArgumentException when {@link #headApplies} is false or a head tensor does not fit
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor the variant needs is absent
     */
    public static HcSubBlock forHead(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        String arch = spec.arch().ggufName();
        if (!headApplies(spec)) {
            throw new IllegalArgumentException(arch + ": the trunk ends without hyper-connection streams");
        }
        ModelHparams hp = spec.hparams();
        ResidualSpec r = spec.layers().get(spec.layers().size() - 1).residual();
        int hc = r.hcMult();
        HcKernels.validateStreams(hc);
        int nEmbd = r.streamWidth() / hc;
        TensorCatalog tc = spec.tensors();
        HeadSpec head = spec.head();
        if (head != null && head.hcHead()) {
            TensorRef fn = tc.require(TensorNames.weight(TensorNames.HC_HEAD_FN));
            TensorRef scale = tc.require(TensorNames.weight(TensorNames.HC_HEAD_SCALE));
            TensorRef base = tc.require(TensorNames.weight(TensorNames.HC_HEAD_BASE));
            CoefficientParams coef = CoefficientParams.head(hc, hp.dsv4HcEps());
            if (fn.rows() != coef.mixDim()) {
                throw new IllegalArgumentException(arch + ": " + fn.name() + " has " + fn.rows() + " rows, the HC head"
                        + " needs " + coef.mixDim());
            }
            checkProjected(arch, -1, hc, nEmbd, fn, scale, base, coef);
            return new HcSubBlock(Kind.HEAD_WEIGHTS, "head", -1, hc, nEmbd, hp.normRmsEps(), fn, scale, base, null,
                    null, null, null, coef, PreParams.weights(), null);
        }
        TensorRef down = tc.get(TensorNames.weight(TensorNames.HC_HEAD_DOWN));
        if (down != null) {
            TensorRef norm = tc.require(TensorNames.weight(TensorNames.HC_HEAD_NORM));
            TensorRef up = tc.require(TensorNames.weight(TensorNames.HC_HEAD_UP));
            checkGated(arch, -1, hc, nEmbd, norm, down, up);
            return new HcSubBlock(Kind.HEAD_GATED, "head", -1, hc, nEmbd, hp.normRmsEps(), null, null, null, norm,
                    down, up, null, null, PreParams.gated(hc), null);
        }
        return new HcSubBlock(Kind.HEAD_MEAN, "head", -1, hc, nEmbd, hp.normRmsEps(), null, null, null, null, null,
                null, null, null, PreParams.mean(hc), null);
    }

    /**
     * The gated collapse that ends the qwen4exp MTP head: {@link Kind#HEAD_GATED} from the NextN block's own
     * {@code nextn.hc_head_norm/down/up} (loaded at qwen4exp.cpp:298-300, applied to the block's output streams at
     * qwen4exp.cpp:606-609). Like the model head it is also the final norm. {@link #il()} is the block's index.
     *
     * @throws UnsupportedOperationException when the block's residual is not per-sub-block hyper-connections
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a {@code nextn.hc_head_*} tensor is absent
     * @throws IllegalArgumentException      when a tensor shape does not fit the block's streams
     */
    public static HcSubBlock forNextnGatedHead(ModelSpec spec, LayerSpec block) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(block, "block");
        String arch = spec.arch().ggufName();
        ResidualSpec r = block.residual();
        if (r.kind() != ResidualKind.HYPER_CONNECTIONS || !r.hcPerSubBlock()) {
            throw new UnsupportedOperationException(arch + " NextN block " + block.il() + ": a hyper-connection head"
                    + " needs per-sub-block hyper-connections, got " + r);
        }
        int hc = r.hcMult();
        HcKernels.validateStreams(hc);
        int nEmbd = r.streamWidth() / hc;
        TensorCatalog tc = spec.tensors();
        String stack = block.stack();
        int wl = block.weightLayer();
        TensorRef norm = tc.require(stack, wl, TensorNames.weight(TensorNames.NEXTN_HC_HEAD_NORM));
        TensorRef down = tc.require(stack, wl, TensorNames.weight(TensorNames.NEXTN_HC_HEAD_DOWN));
        TensorRef up = tc.require(stack, wl, TensorNames.weight(TensorNames.NEXTN_HC_HEAD_UP));
        checkGated(arch, block.il(), hc, nEmbd, norm, down, up);
        return new HcSubBlock(Kind.HEAD_GATED, "head", block.il(), hc, nEmbd, spec.hparams().normRmsEps(), null, null,
                null, norm, down, up, null, null, PreParams.gated(hc), null);
    }

    private static void checkProjected(String arch, int il, int hc, int nEmbd, TensorRef fn, TensorRef scale,
                                       TensorRef base, CoefficientParams coef) {
        if (fn.cols() != (long) hc * nEmbd) {
            throw new IllegalArgumentException(where(arch, il) + fn.name() + " has " + fn.cols()
                    + " columns, the flattened stream has " + (long) hc * nEmbd);
        }
        if (scale.numElements() != coef.scaleCount()) {
            throw new IllegalArgumentException(where(arch, il) + scale.name() + " has " + scale.numElements()
                    + " elements, needs " + coef.scaleCount());
        }
        if (base.numElements() != coef.baseCount()) {
            throw new IllegalArgumentException(where(arch, il) + base.name() + " has " + base.numElements()
                    + " elements, needs " + coef.baseCount());
        }
    }

    /**
     * The grouped-norm gammas are checked by element count only: llama.cpp loads {@code hc_*_norm} and
     * {@code output_hc_norm} with {@code TENSOR_ALLOW_RESHAPE} (qwen4exp.cpp:186, 232, 236), so a GGUF may store the
     * {@code [n_embd, hc]} gammas flattened; both layouts put gamma {@code (h, i)} at {@code h*n_embd + i}.
     */
    private static void checkGated(String arch, int il, int hc, int nEmbd, TensorRef norm, TensorRef down,
                                   TensorRef up) {
        long streamWidth = (long) hc * nEmbd;
        if (norm.numElements() != streamWidth) {
            throw new IllegalArgumentException(where(arch, il) + norm.name() + " must hold " + streamWidth
                    + " gammas ([" + nEmbd + ", " + hc + "] or flattened), got " + norm);
        }
        if (down.cols() != streamWidth || down.rows() < 1) {
            throw new IllegalArgumentException(where(arch, il) + down.name() + " must be [" + streamWidth
                    + ", rank], got " + down);
        }
        requireShape(arch, il, up, down.rows(), streamWidth);
    }

    private static void requireShape(String arch, int il, TensorRef t, long cols, long rows) {
        if (t.cols() != cols || t.rows() != rows) {
            throw new IllegalArgumentException(where(arch, il) + t.name() + " must be [" + cols + ", " + rows
                    + "], got " + t);
        }
    }

    private static String where(String arch, int il) {
        return il < 0 ? arch + ": " : arch + " layer " + il + ": ";
    }

    public Kind kind() { return kind; }

    /** {@code "attn"}, {@code "ffn"} or {@code "head"}. */
    public String site() { return site; }

    /** Layer index; -1 for the model head, the NextN block's index for its MTP head. */
    public int il() { return il; }

    /** Stream count. */
    public int hc() { return hc; }

    /** Width of one stream and of the collapsed row. */
    public int nEmbd() { return nEmbd; }

    /** Values per token of the stream: {@code hc * nEmbd}. */
    public int streamWidth() { return hc * nEmbd; }

    /** Eps of the flattened-stream and grouped RMS norms ({@code norm_rms_eps}). */
    public float normEps() { return normEps; }

    /** {@code hc_*_fn} / {@code output_hc_fn}; null unless {@link Kind#projected()}. */
    public TensorRef fn() { return fn; }

    /** {@code hc_*_scale}; null unless {@link Kind#projected()}. */
    public TensorRef scale() { return scale; }

    /** {@code hc_*_base}; null unless {@link Kind#projected()}. */
    public TensorRef base() { return base; }

    /**
     * Grouped-norm gammas, {@code hc*n_embd} values with gamma {@code (h, i)} at {@code h*n_embd + i}
     * ({@code [n_embd, hc]} or flattened); null unless {@link Kind#gated()}.
     */
    public TensorRef norm() { return norm; }

    /** Low-rank down projection {@code [hc*n_embd, rank]}; null unless {@link Kind#gated()}. */
    public TensorRef down() { return down; }

    /** Low-rank up projection {@code [rank, hc*n_embd]}; null unless {@link Kind#gated()}. */
    public TensorRef up() { return up; }

    /** Inject projection {@code [hc*n_embd, hc]}; null unless {@link Kind#GATED}. */
    public TensorRef inject() { return inject; }

    /** Low-rank width; 0 unless {@link Kind#gated()}. */
    public int lowRank() { return down == null ? 0 : (int) down.rows(); }

    /** Scale applied to the low-rank activations before the SiLU: {@code 1/hc} (qwen4exp.cpp:332). */
    public float lowRankScale() { return 1.0f / hc; }

    /** {@code LLM_HC_PrePostComb} scalars; null unless {@link Kind#projected()}. */
    public CoefficientParams coefficients() { return coefficients; }

    /** Width of one token's mixes row; 0 unless {@link Kind#projected()}. */
    public int mixDim() { return coefficients == null ? 0 : coefficients.mixDim(); }

    /** {@code LLM_HC_Pre} scalars. */
    public PreParams pre() { return pre; }

    /** {@code LLM_HC_Post} scalars; null for the head. */
    public PostParams post() { return post; }

    @Override
    public String toString() {
        return "HcSubBlock{" + kind + ", " + site + (il >= 0 ? " " + il : "") + ", hc=" + hc + ", nEmbd=" + nEmbd
                + (down != null ? ", rank=" + lowRank() : "") + '}';
    }
}
