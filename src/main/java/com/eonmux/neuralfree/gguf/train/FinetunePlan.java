package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The trainable geometry of one open model: which tensors a finetuning run reads, how each block computes, and every
 * reason a model of another shape is refused. Built from the model's {@link ModelSpec}, so the tensor names, head
 * counts and norms are the ones the inference engine resolved for the same file.
 *
 * <p>The supported family is the decoder block llama.cpp's {@code llama-finetune} trains: causal GQA/MHA attention
 * with RoPE (optional per-head Q/K RMS or LayerNorm), a dense gated or ungated FFN or a softmax-routed mixture of
 * experts, and RMS or LayerNorm before each sub-block (or the sandwich's pre- and post-norms). The plan also follows
 * the whole-file scalings the reference trainer keeps: a standard sliding window, attention and final logit soft-caps,
 * a residual scale and an embedding scale. Everything else — recurrent, hybrid, multi-head-latent, parallel-attention
 * blocks, non-sandwich post-norms, group-limited or shared-expert routing, extra layer hooks, sigmoid caps and logit
 * scalings, encoder, embedding, classifier and draft models — is refused with the architecture, layer and field that
 * place it outside the family. Weights may be stored as F32, F16 or BF16: the run keeps float32 master values and a
 * save writes them back in the tensor's stored type.</p>
 *
 * <p>Immutable. Tensor names are the stored names of the file, so a run can write the trained values back.</p>
 */
public final class FinetunePlan {

    /** How a layer's dense FFN combines its projections. */
    public enum FfnForm {
        /** {@code down(act(gate(x)) * up(x))} with separate gate and up tensors. */
        PARALLEL_GLU,
        /** {@code down(act(gate(up(x))))}. */
        SEQUENTIAL_GLU,
        /** {@code down(act(up(x)))}. */
        UNGATED
    }

    /** One norm: its kind, epsilon, and the stored weight/bias tensors (either may be null for a weightless norm). */
    public static final class Norm {
        private final NormKind kind;
        private final float eps;
        private final String weight;
        private final String bias;
        private final int rows;
        private final int rowLen;

        Norm(NormKind kind, float eps, String weight, String bias, int rows, int rowLen) {
            this.kind = kind;
            this.eps = eps;
            this.weight = weight;
            this.bias = bias;
            this.rows = rows;
            this.rowLen = rowLen;
        }

        /** {@link NormKind#RMS} or {@link NormKind#LAYER}. */
        public NormKind kind() {
            return kind;
        }

        /** The norm epsilon. */
        public float eps() {
            return eps;
        }

        /** Stored weight tensor name, or null. */
        public String weight() {
            return weight;
        }

        /** Stored bias tensor name, or null. */
        public String bias() {
            return bias;
        }

        /** Normalized rows per token (one, or the query/key head count for a per-head Q/K norm). */
        public int rows() {
            return rows;
        }

        /** Values per normalized row. */
        public int rowLen() {
            return rowLen;
        }
    }

    /** Per-head Q/K norm of one layer: the geometry the mixer normalizes Q and K with. */
    public static final class QkNorm {
        private final Norm query;
        private final Norm key;
        private final boolean afterRope;

        QkNorm(Norm query, Norm key, boolean afterRope) {
            this.query = query;
            this.key = key;
            this.afterRope = afterRope;
        }

        /** The query norm, or null. */
        public Norm query() {
            return query;
        }

        /** The key norm, or null. */
        public Norm key() {
            return key;
        }

        /** Whether the norm runs after RoPE ({@code blk.0.attn_q_norm} graphs differ on the order}). */
        public boolean afterRope() {
            return afterRope;
        }
    }

    /** The routed mixture-of-experts FFN of one layer: its router and the expert tensors. */
    public static final class Moe {
        private final int nExpert;
        private final int nUsed;
        private final int nFfExp;
        private final float weightsScale;
        private final boolean normWeights;
        private final String router;
        private final String routerBias;
        private final String gateExps;
        private final String upExps;
        private final String downExps;
        private final FfnForm form;
        private final GluOp glu;
        private final ActOp act;

        Moe(int nExpert, int nUsed, int nFfExp, float weightsScale, boolean normWeights, String router,
            String routerBias, String gateExps, String upExps, String downExps, FfnForm form, GluOp glu, ActOp act) {
            this.nExpert = nExpert;
            this.nUsed = nUsed;
            this.nFfExp = nFfExp;
            this.weightsScale = weightsScale;
            this.normWeights = normWeights;
            this.router = router;
            this.routerBias = routerBias;
            this.gateExps = gateExps;
            this.upExps = upExps;
            this.downExps = downExps;
            this.form = form;
            this.glu = glu;
            this.act = act;
        }

        /** Experts the router chooses among. */
        public int nExpert() {
            return nExpert;
        }

        /** Experts one token is routed to. */
        public int nUsed() {
            return nUsed;
        }

        /** Expert FFN width. */
        public int nFfExp() {
            return nFfExp;
        }

        /** Multiplier of the routed output; 0 or 1 leaves it unchanged. */
        public float weightsScale() {
            return weightsScale;
        }

        /** Whether the selected weights are renormalized to sum to one. */
        public boolean normWeights() {
            return normWeights;
        }

        /** Stored name of the router matrix ({@code nExpert} rows of {@code nEmbd}). */
        public String router() {
            return router;
        }

        /** Stored name of the router bias, or null. */
        public String routerBias() {
            return routerBias;
        }

        /** Stored name of the stacked expert gate weights (expert-major rows), or null when ungated. */
        public String gateExps() {
            return gateExps;
        }

        /** Stored name of the stacked expert up weights. */
        public String upExps() {
            return upExps;
        }

        /** Stored name of the stacked expert down weights. */
        public String downExps() {
            return downExps;
        }

        /** How the expert FFN combines its projections. */
        public FfnForm form() {
            return form;
        }

        /** The GLU op of a gated expert, else null. */
        public GluOp glu() {
            return glu;
        }

        /** The activation of an ungated expert, else null. */
        public ActOp act() {
            return act;
        }
    }

    /** One trainable block. */
    public static final class Layer {
        private final int il;
        private final Norm attnNorm;
        private final Norm ffnNorm;
        private final Norm postAttnNorm;
        private final Norm postFfnNorm;
        private final QkNorm qkNorm;
        private final int nHead;
        private final int nHeadKv;
        private final int headDimK;
        private final int headDimV;
        private final int qDim;
        private final int kDim;
        private final int vDim;
        private final int outDim;
        private final float scoreScale;
        private final int swaWindow;
        private final float attnSoftcap;
        private final String attnQ;
        private final String attnK;
        private final String attnV;
        private final String attnOut;
        private final String attnQBias;
        private final String attnKBias;
        private final String attnVBias;
        private final String attnOutBias;
        private final FfnForm form;
        private final GluOp glu;
        private final ActOp act;
        private final String ffnGate;
        private final String ffnUp;
        private final String ffnDown;
        private final String ffnGateBias;
        private final String ffnUpBias;
        private final String ffnDownBias;
        private final int nFf;
        private final Moe moe;
        private final RopeSpec rope;
        private final float residualScale;

        Layer(Builder b) {
            this.il = b.il;
            this.attnNorm = b.attnNorm;
            this.ffnNorm = b.ffnNorm;
            this.postAttnNorm = b.postAttnNorm;
            this.postFfnNorm = b.postFfnNorm;
            this.qkNorm = b.qkNorm;
            this.nHead = b.nHead;
            this.nHeadKv = b.nHeadKv;
            this.headDimK = b.headDimK;
            this.headDimV = b.headDimV;
            this.qDim = b.qDim;
            this.kDim = b.kDim;
            this.vDim = b.vDim;
            this.outDim = b.outDim;
            this.scoreScale = b.scoreScale;
            this.swaWindow = b.swaWindow;
            this.attnSoftcap = b.attnSoftcap;
            this.attnQ = b.attnQ;
            this.attnK = b.attnK;
            this.attnV = b.attnV;
            this.attnOut = b.attnOut;
            this.attnQBias = b.attnQBias;
            this.attnKBias = b.attnKBias;
            this.attnVBias = b.attnVBias;
            this.attnOutBias = b.attnOutBias;
            this.form = b.form;
            this.glu = b.glu;
            this.act = b.act;
            this.ffnGate = b.ffnGate;
            this.ffnUp = b.ffnUp;
            this.ffnDown = b.ffnDown;
            this.ffnGateBias = b.ffnGateBias;
            this.ffnUpBias = b.ffnUpBias;
            this.ffnDownBias = b.ffnDownBias;
            this.nFf = b.nFf;
            this.moe = b.moe;
            this.rope = b.rope;
            this.residualScale = b.residualScale;
        }

        /** Layer index in the trunk. */
        public int il() {
            return il;
        }

        /** The attention input norm. */
        public Norm attnNorm() {
            return attnNorm;
        }

        /** The FFN input norm; null when the FFN shares the attention norm output. */
        public Norm ffnNorm() {
            return ffnNorm;
        }

        /** The post-attention norm of a sandwich block (applied to the mixer output), or null. */
        public Norm postAttnNorm() {
            return postAttnNorm;
        }

        /** The post-FFN norm of a sandwich block (applied to the FFN output), or null. */
        public Norm postFfnNorm() {
            return postFfnNorm;
        }

        /** The per-head Q/K norm, or null when the layer has none. */
        public QkNorm qkNorm() {
            return qkNorm;
        }

        public int nHead() {
            return nHead;
        }

        public int nHeadKv() {
            return nHeadKv;
        }

        public int headDimK() {
            return headDimK;
        }

        public int headDimV() {
            return headDimV;
        }

        public int qDim() {
            return qDim;
        }

        public int kDim() {
            return kDim;
        }

        public int vDim() {
            return vDim;
        }

        public int outDim() {
            return outDim;
        }

        /** The attention score scale ({@code kq_scale * outScale}). */
        public float scoreScale() {
            return scoreScale;
        }

        /** The sliding window of a local-attention layer, or 0 for a full causal window. */
        public int swaWindow() {
            return swaWindow;
        }

        /** The attention logit soft-cap, or 0 for none. */
        public float attnSoftcap() {
            return attnSoftcap;
        }

        public String attnQ() {
            return attnQ;
        }

        public String attnK() {
            return attnK;
        }

        public String attnV() {
            return attnV;
        }

        public String attnOut() {
            return attnOut;
        }

        public String attnQBias() {
            return attnQBias;
        }

        public String attnKBias() {
            return attnKBias;
        }

        public String attnVBias() {
            return attnVBias;
        }

        public String attnOutBias() {
            return attnOutBias;
        }

        public FfnForm form() {
            return form;
        }

        /** The GLU op of a gated FFN, else null. */
        public GluOp glu() {
            return glu;
        }

        /** The activation of an ungated or sequential FFN, else null. */
        public ActOp act() {
            return act;
        }

        public String ffnGate() {
            return ffnGate;
        }

        public String ffnUp() {
            return ffnUp;
        }

        public String ffnDown() {
            return ffnDown;
        }

        public String ffnGateBias() {
            return ffnGateBias;
        }

        public String ffnUpBias() {
            return ffnUpBias;
        }

        public String ffnDownBias() {
            return ffnDownBias;
        }

        public int nFf() {
            return nFf;
        }

        /** The routed mixture of experts, or null for a dense FFN. */
        public Moe moe() {
            return moe;
        }

        /** The layer's RoPE, or null for a NoPE layer. */
        public RopeSpec rope() {
            return rope;
        }

        /** Multiplier of the residual adds; 1 for the standard block. */
        public float residualScale() {
            return residualScale;
        }
    }

    private final String arch;
    private final ModelHparams hparams;
    private final int nLayer;
    private final int nEmbd;
    private final int nVocab;
    private final int nCtxTrain;
    private final List<Layer> layers;
    private final Norm outputNorm;
    private final boolean tiedOutput;
    private final String tokenEmbd;
    private final String outputWeight;
    private final String outputBias;
    private final float embdScale;
    private final float finalSoftcap;

    private FinetunePlan(ModelSpec spec) {
        this.arch = spec.arch().ggufName();
        this.hparams = spec.hparams();
        this.nLayer = spec.nLayer();
        this.nEmbd = hparams.nEmbd();
        HeadSpec head = spec.head();
        this.nVocab = Math.toIntExact(head.vocabRows());
        this.nCtxTrain = hparams.nCtxTrain();
        this.layers = Collections.unmodifiableList(planLayers(spec));
        this.outputNorm = outputNorm(spec);
        this.tiedOutput = head.tied();
        this.tokenEmbd = tensorName(spec, spec.embedding().tokenEmbd());
        this.outputWeight = tensorName(spec, head.output());
        this.outputBias = head.outputBias() != null ? head.outputBias().name() : null;
        this.embdScale = spec.embedding().embdScale();
        this.finalSoftcap = head.finalSoftcap();
    }

    /**
     * The trainable plan of {@code spec}.
     *
     * @throws UnsupportedOperationException naming the architecture and field outside the supported family
     * @throws IllegalArgumentException      when the model's tensors contradict its resolved spec
     */
    public static FinetunePlan of(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        checkModel(spec);
        return new FinetunePlan(spec);
    }

    /** The GGUF architecture name. */
    public String arch() {
        return arch;
    }

    /** The model hparams. */
    public ModelHparams hparams() {
        return hparams;
    }

    public int nLayer() {
        return nLayer;
    }

    public int nEmbd() {
        return nEmbd;
    }

    public int nVocab() {
        return nVocab;
    }

    /** The context the model was trained with, the largest training window. */
    public int nCtxTrain() {
        return nCtxTrain;
    }

    /** The trunk blocks in execution order (unmodifiable). */
    public List<Layer> layers() {
        return layers;
    }

    public Layer layer(int il) {
        return layers.get(il);
    }

    /** The final norm before the LM head. */
    public Norm outputNorm() {
        return outputNorm;
    }

    /** Whether the LM head shares the token embedding table. */
    public boolean tiedOutput() {
        return tiedOutput;
    }

    /** Stored name of the token embedding table. */
    public String tokenEmbd() {
        return tokenEmbd;
    }

    /** Stored name of the LM head weight (the token table when tied). */
    public String outputWeight() {
        return outputWeight;
    }

    /** Stored name of the LM head bias, or null. */
    public String outputBias() {
        return outputBias;
    }

    /** Multiplier of the token-embedding rows (gemma's {@code sqrt(n_embd)}), 1 when none. */
    public float embdScale() {
        return embdScale;
    }

    /** The final logit soft-cap ({@code cap * tanh(logits / cap)}), or 0 for none. */
    public float finalSoftcap() {
        return finalSoftcap;
    }

    /** Every stored tensor name the plan trains, once each, tied tensors included once. */
    public List<String> trainableTensorNames() {
        List<String> names = new ArrayList<>();
        add(names, tokenEmbd);
        add(names, outputWeight);
        add(names, outputBias);
        addNorm(names, outputNorm);
        for (Layer l : layers) {
            addNorm(names, l.attnNorm);
            addNorm(names, l.ffnNorm);
            addNorm(names, l.postAttnNorm);
            addNorm(names, l.postFfnNorm);
            if (l.qkNorm != null) {
                addNorm(names, l.qkNorm.query);
                addNorm(names, l.qkNorm.key);
            }
            add(names, l.attnQ);
            add(names, l.attnK);
            add(names, l.attnV);
            add(names, l.attnOut);
            add(names, l.attnQBias);
            add(names, l.attnKBias);
            add(names, l.attnVBias);
            add(names, l.attnOutBias);
            if (l.moe != null) {
                add(names, l.moe.router);
                add(names, l.moe.routerBias);
                add(names, l.moe.gateExps);
                add(names, l.moe.upExps);
                add(names, l.moe.downExps);
            } else {
                add(names, l.ffnGate);
                add(names, l.ffnUp);
                add(names, l.ffnDown);
                add(names, l.ffnGateBias);
                add(names, l.ffnUpBias);
                add(names, l.ffnDownBias);
            }
        }
        List<String> unique = new ArrayList<>(names.size());
        for (String name : names) {
            if (!unique.contains(name)) {
                unique.add(name);
            }
        }
        return Collections.unmodifiableList(unique);
    }

    private static void add(List<String> names, String name) {
        if (name != null) {
            names.add(name);
        }
    }

    private static void addNorm(List<String> names, Norm norm) {
        if (norm != null) {
            add(names, norm.weight);
            add(names, norm.bias);
        }
    }

    // ------------------------------------------------------------------ model-level validation

    private static void checkModel(ModelSpec spec) {
        String arch = spec.arch().ggufName();
        SupportClass support = spec.supportClass();
        if (support != SupportClass.GEN) {
            throw new UnsupportedOperationException("finetune trains autoregressive decoder models; " + arch
                    + " is a " + support + " model");
        }
        if (spec.layers().isEmpty()) {
            throw new UnsupportedOperationException("finetune needs a trunk of decoder layers; " + arch + " has none");
        }
        if (spec.nLayer() != spec.hparams().nLayerAll() || spec.schedule().kind()
                != com.eonmux.neuralfree.gguf.spec.LayerSchedule.Kind.LINEAR) {
            throw new UnsupportedOperationException("finetune trains the plain linear layer schedule; " + arch
                    + " runs " + spec.schedule().kind() + " with " + spec.nLayer() + " of "
                    + spec.hparams().nLayerAll() + " layer slots");
        }
        if (!spec.nextnLayers().isEmpty()) {
            throw new UnsupportedOperationException("finetune trains the trunk only; " + arch + " carries "
                    + spec.nextnLayers().size() + " NextN/MTP layer(s)");
        }
        if (!spec.encoderLayers().isEmpty()) {
            throw new UnsupportedOperationException("finetune trains decoder-only models; " + arch + " is an"
                    + " encoder-decoder model");
        }
        ModelHparams hp = spec.hparams();
        if (hp.nEmbdInp() != hp.nEmbd() || hp.nEmbdOut() != hp.nEmbd()) {
            throw new UnsupportedOperationException("finetune needs matching input, trunk and output widths; " + arch
                    + " has " + hp.nEmbdInp() + " / " + hp.nEmbd() + " / " + hp.nEmbdOut());
        }
        checkEmbedding(spec);
        checkHead(spec);
        for (LayerSpec layer : spec.layers()) {
            checkLayer(arch, hp, layer);
        }
    }

    private static void checkEmbedding(ModelSpec spec) {
        com.eonmux.neuralfree.gguf.spec.EmbeddingSpec emb = spec.embedding();
        String arch = spec.arch().ggufName();
        if (emb == null || emb.tokenEmbd() == null) {
            throw new UnsupportedOperationException("finetune needs a token embedding table; " + arch
                    + " has none");
        }
        if (emb.tokenEmbd().ne(0) != spec.hparams().nEmbd()) {
            throw new IllegalArgumentException(arch + ": token_embd is " + emb.tokenEmbd().ne(0) + " wide, the trunk "
                    + spec.hparams().nEmbd());
        }
        refuse(emb.inputRmsNorm(), arch, "an input RMS norm");
        refuse(emb.inputProjection() != null, arch, "an input projection (project_in)");
        refuse(emb.posEmbd() != null, arch, "position embeddings (pos_embd)");
        refuse(emb.tokenTypes() != null, arch, "token type embeddings");
        refuse(emb.tokenEmbdNorm() != null, arch, "a token-embedding norm (token_embd_norm)");
        refuse(emb.perLayerEmbeddings(), arch, "per-layer token embeddings");
        refuse(emb.ngram(), arch, "n-gram embeddings");
        refuse(emb.posOffset() != 0, arch, "a position offset of " + emb.posOffset());
    }

    private static void checkHead(ModelSpec spec) {
        HeadSpec head = spec.head();
        String arch = spec.arch().ggufName();
        if (head == null || head.output() == null) {
            throw new UnsupportedOperationException("finetune trains a model with an LM head; " + arch + " has none");
        }
        refuse(head.outputNormGroups() != 1, arch, "a grouped output norm (" + head.outputNormGroups() + " groups)");
        refuse(head.outputNorm() == null, arch, "no output norm (output_norm)");
        refuse(head.outputProjection() != null, arch, "a project_out projection");
        refuse(head.embeddingProjection() != null, arch, "an embedding projection");
        refuse(head.sigmoidCapped(), arch, "a sigmoid logit cap");
        refuse(head.logitScale() != 1.0f || head.logitScaleDivides(),
                arch, "a logit scale of " + head.logitScale());
        refuse(head.logitScaleVector() != null, arch, "a per-vocabulary logit scale");
        refuse(head.masksLogits(), arch, "a masked logit range [" + head.maskedLogitsStart() + ", "
                + head.maskedLogitsEnd() + ")");
        refuse(head.hcHead(), arch, "a hyper-connection head");
        refuse(head.d2t() != null, arch, "a draft token map (d2t)");
        refuse(head.dense2() != null || head.dense3() != null, arch, "an extra dense head projection");
        refuse(head.cls() != null || head.clsOutput() != null, arch, "a classifier head");
        refuse(head.colbertOutput() != null || head.sparseOutput() != null, arch, "a multi-output head");
        int nEmbd = spec.hparams().nEmbd();
        if (head.output().ne(0) != nEmbd) {
            throw new IllegalArgumentException(arch + ": the LM head reads " + head.output().ne(0) + " values, the"
                    + " trunk produces " + nEmbd);
        }
        NormSpec trunk = spec.layer(0).norm();
        if (trunk == null || !isRowNorm(trunk.kind())) {
            throw new UnsupportedOperationException("finetune needs the output norm to be an RMS or layer norm; "
                    + arch + " declares " + (trunk == null ? "no trunk norm" : trunk.kind()));
        }
    }

    // ------------------------------------------------------------------ per-layer validation

    private static void checkLayer(String arch, ModelHparams hp, LayerSpec layer) {
        String where = arch + " layer " + layer.il();
        if (layer.weightLayer() != layer.il() || !TensorNames.STACK_BLK.equals(layer.stack())) {
            throw new UnsupportedOperationException("finetune trains main-stack layers bound to their own weights; "
                    + where + " runs stack " + layer.stack() + " weight layer " + layer.weightLayer());
        }
        if (layer.mixerKind() != MixerKind.ATTENTION) {
            throw new UnsupportedOperationException("finetune trains softmax-attention blocks; " + where
                    + " runs a " + layer.mixerKind() + " mixer");
        }
        if (layer.parallelAttention()) {
            throw new UnsupportedOperationException("finetune trains sequential attention and FFN; " + where
                    + " runs a parallel attention branch");
        }
        refuse(!layer.extras().isEmpty(), where, "layer extras " + layer.extras());
        refuse(layer.mla() != null, where, "multi-head latent attention");
        refuse(layer.dsa() != null, where, "sparse attention (DSA)");
        refuse(layer.msa() != null, where, "multi-sequence attention (MSA)");
        refuse(layer.step5() != null, where, "Step5 sparse region attention");
        refuse(layer.dsv4() != null, where, "DeepSeek-V4 attention");
        refuse(layer.recurrent() != null, where, "a recurrent mixer");
        refuse(layer.zamba2() != null, where, "a Zamba2 shared block");
        checkFfnKind(where, layer);
        checkResidual(where, layer.residual());
        checkNorm(where, layer.norm());
        checkAttention(where, layer.il(), layer.attention());
        checkRope(where, layer.rope(), layer.attention());
        checkQkNorm(where, layer.attention());
    }

    /** The dense or routed FFN the layer runs; every other FFN kind is refused by name. */
    private static void checkFfnKind(String where, LayerSpec layer) {
        FfnKind kind = layer.ffnKind();
        if (kind == FfnKind.DENSE) {
            if (layer.moe() != null) {
                throw new IllegalStateException(where + " runs a dense FFN with a routed spec");
            }
            checkFfn(where, layer.ffn());
            return;
        }
        if (kind == FfnKind.MOE) {
            if (layer.moe() == null) {
                throw new IllegalStateException(where + " runs routed experts without a MoE spec");
            }
            refuse(layer.ffn() != null, where, "a dense FFN beside the routed experts");
            checkMoe(where, layer.moe());
            return;
        }
        throw new UnsupportedOperationException("finetune trains dense or routed-expert feed-forward blocks; " + where
                + " runs " + kind);
    }

    /**
     * The routed expert subset the trainer represents: a softmax router over the whole expert set, top-k selection by
     * probability, separate (or fused-free) gate/up/down expert stacks, and an optional router bias. Group-limited,
     * hash, latent, chunked, biased, scaled, shared-expert and input-weighted routing are refused by name.
     */
    private static void checkMoe(String where, MoeSpec moe) {
        if (moe.gating() != GatingFunc.SOFTMAX) {
            throw new UnsupportedOperationException("finetune trains a softmax-routed mixture of experts; " + where
                    + " gates with " + moe.gating());
        }
        refuse(moe.selectionMode() != MoeSpec.SelectionMode.PROBS, where,
                "the " + moe.selectionMode() + " expert selection");
        refuse(moe.nGroups() != 1, where, "group-limited routing over " + moe.nGroups() + " groups");
        refuse(moe.zeroExperts() != 0, where, moe.zeroExperts() + " identity expert slot(s)");
        refuse(moe.fusedGateUp(), where, "a fused expert gate_up tensor");
        refuse(moe.probBias(), where, "an expert selection bias (exp_probs_b)");
        refuse(moe.expertScales(), where, "per-expert output scales");
        refuse(moe.expertBiases(), where, "per-expert biases");
        refuse(moe.latent(), where, "latent experts");
        refuse(moe.chunked(), where, "chunked experts");
        refuse(moe.hashRouting(), where, "hash routing");
        refuse(moe.mediaRouting(), where, "media-row routing");
        refuse(moe.weightBeforeFfn(), where, "router weights before the expert input");
        refuse(moe.sharedExpert() != null, where, "a shared expert");
        refuse(moe.sharedExpertGated(), where, "a shared-expert gate");
        refuse(moe.postExpertNorm(), where, "a norm after the experts");
        refuse(moe.sharedCombineScale() != 1.0f, where, "a shared combine scale of " + moe.sharedCombineScale());
        refuse(moe.params() != ActParams.NONE, where, "expert activation parameters " + moe.params());
        if (moe.gated() && moe.glu() == null) {
            throw new UnsupportedOperationException("finetune needs the GLU op of gated experts; " + where
                    + " declares none");
        }
        if (!moe.gated() && moe.act() == null) {
            throw new UnsupportedOperationException("finetune needs the activation of ungated experts; " + where
                    + " declares none");
        }
        refuse(moe.glu() != null && !supportedGlu(moe.glu()), where, "the " + moe.glu() + " expert gate");
        refuse(moe.act() != null && !supportedAct(moe.act()), where, "the " + moe.act() + " expert activation");
        if (moe.nExpert() <= 0 || moe.nUsed() <= 0 || moe.nUsed() > moe.nExpert() || moe.nFfExp() <= 0) {
            throw new UnsupportedOperationException("finetune needs 0 < used <= experts and an expert width; " + where
                    + " has " + moe.nUsed() + "/" + moe.nExpert() + " experts of width " + moe.nFfExp());
        }
    }

    private static void checkResidual(String where, ResidualSpec res) {
        if (res == null || res.kind() != ResidualKind.STANDARD) {
            throw new UnsupportedOperationException("finetune trains the standard {@code x += branch} residual; "
                    + where + " runs " + (res == null ? "no residual spec" : res.kind()));
        }
        refuse(!res.mixerResidual(), where, "no attention residual");
        refuse(res.residualScale() != 1.0f, where, "a residual scale of " + res.residualScale());
        refuse(res.discardLayerOutput(), where, "a discarded layer output");
        refuse(res.embeddingSkip(), where, "an embedding skip connection");
        refuse(res.attnAlpha() != null || res.attnBeta() != null || res.ffnAlpha() != null
                || res.ffnBeta() != null, where, "per-channel residual weights");
        refuse(res.blockOutputScale() != 1.0f, where, "a block output scale of " + res.blockOutputScale());
    }

    private static void checkNorm(String where, NormSpec norm) {
        if (norm == null || !isRowNorm(norm.kind())) {
            throw new UnsupportedOperationException("finetune needs RMS or layer norms; " + where + " declares "
                    + (norm == null ? "no norm" : norm.kind()));
        }
        PostNormRole post = norm.postRole();
        if (post != PostNormRole.NONE && post != PostNormRole.SANDWICH) {
            throw new UnsupportedOperationException("finetune trains pre-norm or sandwich-norm blocks; " + where
                    + " declares the " + post + " post-norm role");
        }
        if (post == PostNormRole.SANDWICH) {
            refuse(!norm.postAttnNorm(), where, "a sandwich block whose post-attention norm is missing");
            refuse(!norm.postFfnNorm(), where, "a sandwich block whose post-FFN norm is missing");
        }
        refuse(post != PostNormRole.SANDWICH && norm.postAttnNorm(), where, "a post-attention norm");
        refuse(post != PostNormRole.SANDWICH && norm.postFfnNorm(), where, "a post-FFN norm");
        refuse(norm.postFfnWeightFirst(), where, "a post-FFN norm that normalizes the weighted branch");
        refuse(norm.attnSubNorm(), where, "an attention sub-norm");
        refuse(norm.ffnSubNorm(), where, "an FFN sub-norm");
        refuse(norm.attnOutputNorm(), where, "an attention output norm");
        refuse(norm.layerOutputNorm(), where, "a layer output norm");
        refuse(norm.ffnAda(), where, "an adaptive FFN norm");
        refuse(norm.groups() != 1, where, "a norm over " + norm.groups() + " groups");
        refuse(!norm.attnNorm(), where, "no attention norm (attn_norm)");
        refuse(norm.attnNorm2(), where, "an attn_norm_2");
    }

    private static void checkAttention(String where, int il, AttentionSpec attn) {
        if (attn == null) {
            throw new UnsupportedOperationException("finetune needs an attention spec; " + where + " has none");
        }
        if (attn.isSwaLayer() && attn.swaType() != SwaType.STANDARD) {
            throw new UnsupportedOperationException("finetune trains a standard sliding window; " + where + " runs "
                    + attn.swaType());
        }
        if (attn.isSwaLayer() && attn.nSwa() <= 0) {
            throw new IllegalArgumentException(where + ": a sliding-window layer needs a positive window, got "
                    + attn.nSwa());
        }
        refuse(attn.fusedQkv(), where, "a fused attn_qkv projection");
        refuse(attn.linearReplace(), where, "a linear replacement (n_head_kv = 0)");
        refuse(attn.gate() != com.eonmux.neuralfree.gguf.arch.AttnGateKind.NONE, where,
                "an attention output gate (" + attn.gate() + ")");
        refuse(attn.temperature().enabled(), where, "a position-dependent attention temperature");
        refuse(attn.clampKqv() != 0f, where, "a Q/K/V clamp of " + attn.clampKqv());
        refuse(attn.sinks(), where, "attention sinks");
        refuse(attn.softcap() < 0f, where, "a negative attention softcap of " + attn.softcap());
        refuse(attn.valueScale() != 0f, where, "a value scale of " + attn.valueScale());
        refuse(attn.biasMode() != AttentionSpec.BiasMode.NONE, where, "a " + attn.biasMode() + " position bias");
        refuse(attn.valueEmbeddings(), where, "value embeddings");
        refuse(attn.valueExperts() != null, where, "routed value experts");
        refuse(attn.alibiMaxBias() != 0f, where, "an ALiBi bias");
        refuse(attn.t5Buckets() != 0, where, "a T5 relative-position bias");
        refuse(attn.kvOwnerLayer() != il, where, "a shared KV cache of layer " + attn.kvOwnerLayer());
        refuse(attn.outScale() != 0f && attn.outScale() != 1.0f, where,
                "an attention output scale of " + attn.outScale());
        if (attn.nHead() <= 0 || attn.nHeadKv() <= 0 || attn.headDimK() <= 0 || attn.headDimV() <= 0) {
            throw new UnsupportedOperationException("finetune needs positive head counts and dims; " + where
                    + " has " + attn.nHead() + " heads / " + attn.nHeadKv() + " KV heads of "
                    + attn.headDimK() + " x " + attn.headDimV());
        }
        if (attn.nHead() % attn.nHeadKv() != 0) {
            throw new IllegalArgumentException(where + ": " + attn.nHead() + " query heads do not group into "
                    + attn.nHeadKv() + " KV heads");
        }
    }

    private static void checkFfn(String where, FfnSpec ffn) {
        if (ffn == null) {
            throw new UnsupportedOperationException("finetune needs a dense FFN spec; " + where + " has none");
        }
        refuse(ffn.role() != FfnSpec.Role.DENSE, where, "the " + ffn.role() + " FFN role");
        refuse(ffn.gating() == FfnGating.FUSED_UP, where, "a fused ffn_up gate and up tensor");
        refuse(ffn.justNorm(), where, "an L2 norm before the down projection");
        refuse(ffn.activationSparsityStdMul() != 0f, where, "activation sparsity");
        if (ffn.gated() && ffn.glu() == null) {
            throw new UnsupportedOperationException("finetune needs the GLU op of a gated FFN; " + where
                    + " declares none");
        }
        if (!ffn.gated() && ffn.act() == null) {
            throw new UnsupportedOperationException("finetune needs the activation of an ungated FFN; " + where
                    + " declares none");
        }
        refuse(ffn.glu() != null && !supportedGlu(ffn.glu()), where, "the " + ffn.glu() + " gate");
        refuse(ffn.act() != null && !supportedAct(ffn.act()), where, "the " + ffn.act() + " activation");
        refuse(ffn.gating() == FfnGating.PAR && !ffn.gated(), where, "a gate projection it does not have");
    }

    private static void checkRope(String where, RopeSpec rope, AttentionSpec attn) {
        if (attn.noRope() || rope == null || rope.type() == RopeType.NONE) {
            return;
        }
        refuse(rope.type() != RopeType.NORM && rope.type() != RopeType.NEOX, where,
                "the " + rope.type() + " rotary embedding");
        refuse(rope.nOffs() != 0, where, "a rotary offset of " + rope.nOffs() + " dims");
        refuse(rope.fope() != null, where, "a learned Fourier position embedding (FoPE)");
        refuse(rope.positionsPerToken() > 1, where, "several positions per token");
        if (rope.nDims() > attn.headDimK()) {
            throw new IllegalArgumentException(where + ": the rotary embedding rotates " + rope.nDims()
                    + " dims of a " + attn.headDimK() + "-dim head");
        }
    }

    private static void checkQkNorm(String where, AttentionSpec attn) {
        QkNormKind kind = attn.qkNorm();
        if (kind == QkNormKind.NONE) {
            return;
        }
        refuse(kind != QkNormKind.RMS_HEAD && kind != QkNormKind.LN_HEAD, where,
                "the " + kind + " Q/K norm geometry");
    }

    // ------------------------------------------------------------------ the plan itself

    private static List<Layer> planLayers(ModelSpec spec) {
        List<Layer> layers = new ArrayList<>(spec.nLayer());
        TensorCatalog catalog = spec.tensors();
        for (LayerSpec ls : spec.layers()) {
            layers.add(planLayer(spec, catalog, ls));
        }
        return layers;
    }

    private static Layer planLayer(ModelSpec spec, TensorCatalog catalog, LayerSpec ls) {
        int il = ls.il();
        AttentionSpec attn = ls.attention();
        FfnSpec ffn = ls.ffn();
        NormSpec norm = ls.norm();
        ModelHparams hp = spec.hparams();
        Builder b = new Builder(il);
        b.nHead = attn.nHead();
        b.nHeadKv = attn.nHeadKv();
        b.headDimK = attn.headDimK();
        b.headDimV = attn.headDimV();
        b.qDim = attn.qDim();
        b.kDim = attn.kDim();
        b.vDim = attn.vDim();
        b.outDim = attn.outDim();
        b.scoreScale = attn.scoreScale();
        b.attnNorm = norm(catalog, ls, TensorNames.ATTN_NORM, norm.kind(), norm.eps(), 1, hp.nEmbd());
        b.ffnNorm = norm.sharedAttnNormForFfn() ? null
                : norm(catalog, ls, TensorNames.FFN_NORM, norm.kind(), norm.eps(), 1, hp.nEmbd());
        b.postAttnNorm = norm.postRole() == PostNormRole.SANDWICH
                ? postNorm(catalog, ls, TensorNames.ATTN_POST_NORM, norm, norm.postAttnWeightless(), hp.nEmbd())
                : null;
        b.postFfnNorm = norm.postRole() == PostNormRole.SANDWICH
                ? postNorm(catalog, ls, TensorNames.FFN_POST_NORM, norm, false, hp.nEmbd())
                : null;
        b.qkNorm = qkNorm(catalog, ls, attn, hp, spec.arch().ggufName() + " layer " + il);
        b.swaWindow = attn.isSwaLayer() ? attn.nSwa() : 0;
        b.attnSoftcap = attn.softcap();
        b.attnQ = ls.tensorName(TensorNames.weight(TensorNames.ATTN_Q));
        b.attnK = ls.tensorName(TensorNames.weight(TensorNames.ATTN_K));
        b.attnV = ls.tensorName(TensorNames.weight(TensorNames.ATTN_V));
        b.attnOut = ls.tensorName(TensorNames.weight(TensorNames.ATTN_OUT));
        boolean inBias = attn.inBias();
        b.attnQBias = inBias ? optional(catalog, ls, TensorNames.bias(TensorNames.ATTN_Q)) : null;
        b.attnKBias = inBias ? optional(catalog, ls, TensorNames.bias(TensorNames.ATTN_K)) : null;
        b.attnVBias = inBias ? optional(catalog, ls, TensorNames.bias(TensorNames.ATTN_V)) : null;
        b.attnOutBias = attn.outBias() ? optional(catalog, ls, TensorNames.bias(TensorNames.ATTN_OUT)) : null;
        if (ffn != null) {
            boolean biases = ffn.biases();
            b.ffnGate = ffn.gated() ? ls.tensorName(ffn.gateWeight()) : null;
            b.ffnUp = ls.tensorName(ffn.upWeight());
            b.ffnDown = ls.tensorName(ffn.downWeight());
            b.ffnGateBias = biases && ffn.gated() ? optional(catalog, ls, TensorNames.bias(ffn.role().gate())) : null;
            b.ffnUpBias = biases ? optional(catalog, ls, TensorNames.bias(ffn.role().up())) : null;
            b.ffnDownBias = biases ? optional(catalog, ls, TensorNames.bias(ffn.role().down())) : null;
            b.nFf = ffn.nFf();
            b.form = ffn.gated()
                    ? (ffn.gating() == FfnGating.PAR ? FfnForm.PARALLEL_GLU : FfnForm.SEQUENTIAL_GLU)
                    : FfnForm.UNGATED;
            b.glu = ffn.gated() ? ffn.glu() : null;
            b.act = ffn.gated() ? null : ffn.act();
        } else {
            b.moe = planMoe(catalog, ls, ls.moe());
            b.nFf = ls.moe().nFfExp();
        }
        b.rope = attn.noRope() ? null : ls.rope();
        b.residualScale = ls.residual().residualScale();
        return new Layer(b);
    }

    /** A post-attention or post-FFN norm of a sandwich block. */
    private static Norm postNorm(TensorCatalog catalog, LayerSpec ls, String base, NormSpec spec, boolean weightless,
                                 int rowLen) {
        String weight = weightless ? null
                : hasTensor(catalog, ls, TensorNames.weight(base)) ? ls.tensorName(TensorNames.weight(base)) : null;
        String bias = weight != null && hasTensor(catalog, ls, TensorNames.bias(base))
                ? ls.tensorName(TensorNames.bias(base)) : null;
        return new Norm(spec.kind(), spec.postEps(), weight, bias, 1, rowLen);
    }

    /** The routed-expert FFN of a MoE layer. */
    private static Moe planMoe(TensorCatalog catalog, LayerSpec ls, MoeSpec moe) {
        String router = ls.tensorName(TensorNames.weight(TensorNames.FFN_GATE_INP));
        String routerBias = moe.routerBias() ? optional(catalog, ls, TensorNames.bias(TensorNames.FFN_GATE_INP)) : null;
        String gate = moe.gated() ? ls.tensorName(TensorNames.weight(TensorNames.FFN_GATE_EXPS)) : null;
        String up = ls.tensorName(TensorNames.weight(TensorNames.FFN_UP_EXPS));
        String down = ls.tensorName(TensorNames.weight(TensorNames.FFN_DOWN_EXPS));
        FfnForm form = moe.gated() ? FfnForm.PARALLEL_GLU : FfnForm.UNGATED;
        return new Moe(moe.nExpert(), moe.nUsed(), moe.nFfExp(), moe.weightsScale(), moe.normWeights(), router,
                routerBias, gate, up, down, form, moe.gated() ? moe.glu() : null, moe.gated() ? null : moe.act());
    }

    private static QkNorm qkNorm(TensorCatalog catalog, LayerSpec ls, AttentionSpec attn, ModelHparams hp,
                                 String where) {
        if (attn.qkNorm() == QkNormKind.NONE) {
            return null;
        }
        refuse(!attn.qkNormBias() && (hasTensor(catalog, ls, TensorNames.bias(TensorNames.ATTN_Q_NORM))
                || hasTensor(catalog, ls, TensorNames.bias(TensorNames.ATTN_K_NORM))), where,
                "Q/K norm biases the architecture loads without applying");
        NormKind kind = attn.qkNorm() == QkNormKind.LN_HEAD ? NormKind.LAYER : NormKind.RMS;
        float eps = kind == NormKind.LAYER ? hp.normEps() : hp.normRmsEps();
        boolean hasQ = hasTensor(catalog, ls, TensorNames.weight(TensorNames.ATTN_Q_NORM));
        boolean hasK = hasTensor(catalog, ls, TensorNames.weight(TensorNames.ATTN_K_NORM));
        Norm q = hasQ || !hasK ? norm(catalog, ls, TensorNames.ATTN_Q_NORM, kind, eps, attn.nHead(),
                attn.headDimK()) : null;
        Norm k = hasK || !hasQ ? norm(catalog, ls, TensorNames.ATTN_K_NORM, kind, eps, attn.nHeadKv(),
                attn.headDimK()) : null;
        return new QkNorm(q, k, attn.qkNormAfterRope());
    }

    /**
     * A layer norm over {@code rows} rows of {@code rowLen}, its weight and bias read from the file when present.
     * A weightless graph has neither.
     */
    private static Norm norm(TensorCatalog catalog, LayerSpec ls, String base, NormKind kind, float eps,
                             int rows, int rowLen) {
        String weight = hasTensor(catalog, ls, TensorNames.weight(base))
                ? ls.tensorName(TensorNames.weight(base)) : null;
        String bias = weight != null && hasTensor(catalog, ls, TensorNames.bias(base))
                ? ls.tensorName(TensorNames.bias(base)) : null;
        return new Norm(kind, eps, weight, bias, rows, rowLen);
    }

    private static boolean hasTensor(TensorCatalog catalog, LayerSpec ls, String suffix) {
        return catalog != null && catalog.hasLayer(ls.weightLayer(), suffix);
    }

    private static String optional(TensorCatalog catalog, LayerSpec ls, String suffix) {
        return hasTensor(catalog, ls, suffix) ? ls.tensorName(suffix) : null;
    }

    private static Norm outputNorm(ModelSpec spec) {
        HeadSpec head = spec.head();
        NormSpec trunk = spec.layer(0).norm();
        TensorRef ref = head.outputNorm();
        String weight = ref != null ? ref.name() : null;
        String bias = head.outputNormBias() != null ? head.outputNormBias().name() : null;
        return new Norm(trunk.kind(), trunk.eps(), weight, bias, 1, spec.hparams().nEmbd());
    }

    private static String tensorName(ModelSpec spec, TensorRef ref) {
        if (ref == null) {
            return null;
        }
        return ref.name();
    }

    private static boolean isRowNorm(NormKind kind) {
        return kind == NormKind.RMS || kind == NormKind.LAYER;
    }

    private static boolean supportedGlu(GluOp op) {
        return op == GluOp.SWIGLU || op == GluOp.GEGLU_TANH || op == GluOp.GEGLU_ERF
                || op == GluOp.GEGLU_QUICK || op == GluOp.REGLU;
    }

    private static boolean supportedAct(ActOp op) {
        return op == ActOp.SILU || op == ActOp.GELU_TANH || op == ActOp.GELU_ERF || op == ActOp.GELU_QUICK
                || op == ActOp.RELU;
    }

    private static void refuse(boolean condition, String where, String what) {
        if (condition) {
            throw new UnsupportedOperationException("finetune does not train " + what + ": " + where);
        }
    }

    /** Builder of {@link Layer}; package-private fields keep the plan construction in one place. */
    private static final class Builder {
        final int il;
        Norm attnNorm;
        Norm ffnNorm;
        Norm postAttnNorm;
        Norm postFfnNorm;
        QkNorm qkNorm;
        int nHead;
        int nHeadKv;
        int headDimK;
        int headDimV;
        int qDim;
        int kDim;
        int vDim;
        int outDim;
        float scoreScale;
        int swaWindow;
        float attnSoftcap;
        String attnQ;
        String attnK;
        String attnV;
        String attnOut;
        String attnQBias;
        String attnKBias;
        String attnVBias;
        String attnOutBias;
        FfnForm form;
        GluOp glu;
        ActOp act;
        String ffnGate;
        String ffnUp;
        String ffnDown;
        String ffnGateBias;
        String ffnUpBias;
        String ffnDownBias;
        int nFf;
        Moe moe;
        RopeSpec rope;
        float residualScale;

        Builder(int il) {
            this.il = il;
        }
    }
}
