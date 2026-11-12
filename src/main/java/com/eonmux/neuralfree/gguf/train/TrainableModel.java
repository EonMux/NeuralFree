package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A float32, fully trainable view of one open GGUF model: forward pass, cross-entropy loss and backward pass for one
 * window of tokens. The weights are the model's own tensors, decoded once into float32 master arrays (F16 and BF16
 * tensors included); each parameter keeps a gradient array the backward pass accumulates into, so an optimizer steps
 * the same arrays a later save writes back to GGUF in each tensor's stored type.
 *
 * <p>One {@link #step} runs the whole training computation: embedding rows, the trunk blocks of {@link FinetunePlan}
 * (norms, GQA attention with RoPE, dense or routed FFN), the output norm and the LM head, then the mean next-token
 * cross entropy of the window and its gradient. The layer buffers are recomputed during the backward pass instead of
 * being retained for every layer, so the memory a step needs is one layer's activations per token plus one saved
 * residual row per layer.</p>
 *
 * <p>Not thread-safe; one step at a time. The parameter arrays are the model's; {@link #close()} closes nothing the
 * model owns.</p>
 */
public final class TrainableModel implements Trainer {

    /** The outcome of one forward or training step over a window. */
    public static final class StepResult {
        private final double loss;
        private final int correct;
        private final int predictions;

        StepResult(double loss, int correct, int predictions) {
            this.loss = loss;
            this.correct = correct;
            this.predictions = predictions;
        }

        /** Mean cross entropy over the window's predicted positions. */
        public double loss() {
            return loss;
        }

        /** Predicted positions whose argmax equals the label. */
        public int correct() {
            return correct;
        }

        /** Predicted positions in the window. */
        public int predictions() {
            return predictions;
        }

        /** Accuracy in {@code [0, 1]}; NaN for a window with no position. */
        public double accuracy() {
            return predictions == 0 ? Double.NaN : (double) correct / predictions;
        }

        @Override
        public String toString() {
            return String.format(java.util.Locale.ROOT, "loss=%.6f acc=%.2f%%", loss, 100.0 * accuracy());
        }
    }

    /**
     * One trainable tensor: its stored GGUF name, type and shape, its float32 master values and its accumulated
     * gradient. A tensor stored as F16 or BF16 keeps float32 master values; the stored type is what a save writes the
     * trained values back as.
     *
     * <p>A trainer that keeps the authoritative copy elsewhere (the device trainer keeps its masters, gradients and
     * optimizer state in device memory) hands out parameters bound to a {@link HostMirror}: every accessor that
     * returns an array first brings it up to date from that copy, so the arrays read the same as the host trainer's
     * at any point.</p>
     */
    public static final class Parameter {
        private final String name;
        private final GGMLType storedType;
        private final long[] ne;
        private final int rows;
        private final int cols;
        private final float[] value;
        private final float[] grad;
        private final HostMirror mirror;

        Parameter(String name, GGMLType storedType, long[] ne, int rows, int cols, float[] value) {
            this(name, storedType, ne, rows, cols, value, new float[value.length], null);
        }

        private Parameter(String name, GGMLType storedType, long[] ne, int rows, int cols, float[] value,
                          float[] grad, HostMirror mirror) {
            this.name = name;
            this.storedType = storedType;
            this.ne = ne;
            this.rows = rows;
            this.cols = cols;
            this.value = value;
            this.grad = grad;
            this.mirror = mirror;
        }

        /** This parameter's arrays, kept current from the copy {@code mirror} manages. */
        Parameter mirroredBy(HostMirror mirror) {
            return new Parameter(name, storedType, ne, rows, cols, value, grad,
                    Objects.requireNonNull(mirror, "mirror"));
        }

        /** Stored GGUF tensor name. */
        public String name() {
            return name;
        }

        /** The type the tensor is stored as: {@code F32}, {@code F16} or {@code BF16}. */
        public GGMLType storedType() {
            return storedType;
        }

        /** ggml shape, {@code ne[0]} first. */
        public long[] ne() {
            return ne.clone();
        }

        /** Output rows ({@code ne[1]}, 1 for a vector). */
        public int rows() {
            return rows;
        }

        /** Input width ({@code ne[0]}). */
        public int cols() {
            return cols;
        }

        /** Element count. */
        public int size() {
            return value.length;
        }

        /**
         * The values (not a copy; a host optimizer updates this array). On a mirrored parameter the array is first
         * brought up to date, and the caller may write it: the next window of the trainer that owns the mirror runs
         * with what the array then holds.
         */
        public float[] value() {
            if (mirror != null) {
                mirror.beforeValues(name, true);
            }
            return value;
        }

        /** The accumulated gradient of the last backward pass. */
        public float[] grad() {
            if (mirror != null) {
                mirror.beforeGrad(name);
            }
            return grad;
        }

        /** A copy of the current values. */
        public float[] values() {
            if (mirror != null) {
                mirror.beforeValues(name, false);
            }
            return value.clone();
        }

        void zeroGrad() {
            Arrays.fill(grad, 0f);
        }

        @Override
        public String toString() {
            return name + " [" + rows + " x " + cols + "]";
        }
    }

    /**
     * The other copy a mirrored {@link Parameter}'s arrays follow: called before an accessor hands an array out, by
     * the parameter's stored name.
     */
    interface HostMirror {
        /**
         * Brings the values of {@code name} up to date; {@code mayWrite} when the caller receives the live array and
         * may change it.
         */
        void beforeValues(String name, boolean mayWrite);

        /** Brings the gradient of {@code name} up to date. */
        void beforeGrad(String name);
    }

    /** The tensors and RoPE table of one block, bound to its parameters. */
    private static final class BoundLayer {
        final FinetunePlan.Layer plan;
        final Parameter attnNormW;
        final Parameter attnNormB;
        final Parameter ffnNormW;
        final Parameter ffnNormB;
        final Parameter postAttnNormW;
        final Parameter postAttnNormB;
        final Parameter postFfnNormW;
        final Parameter postFfnNormB;
        final Parameter qkQW;
        final Parameter qkQB;
        final Parameter qkKW;
        final Parameter qkKB;
        final Parameter wq;
        final Parameter bq;
        final Parameter wk;
        final Parameter bk;
        final Parameter wv;
        final Parameter bv;
        final Parameter wo;
        final Parameter bo;
        final Parameter wGate;
        final Parameter bGate;
        final Parameter wUp;
        final Parameter bUp;
        final Parameter wDown;
        final Parameter bDown;
        final Parameter moeRouter;
        final Parameter moeRouterBias;
        final Parameter moeGateExp;
        final Parameter moeUpExp;
        final Parameter moeDownExp;
        final RopeFreqTable rope;
        final int qkWRows;
        final int qkKRows;

        BoundLayer(Builder b) {
            this.plan = b.plan;
            this.attnNormW = b.attnNormW;
            this.attnNormB = b.attnNormB;
            this.ffnNormW = b.ffnNormW;
            this.ffnNormB = b.ffnNormB;
            this.postAttnNormW = b.postAttnNormW;
            this.postAttnNormB = b.postAttnNormB;
            this.postFfnNormW = b.postFfnNormW;
            this.postFfnNormB = b.postFfnNormB;
            this.qkQW = b.qkQW;
            this.qkQB = b.qkQB;
            this.qkKW = b.qkKW;
            this.qkKB = b.qkKB;
            this.wq = b.wq;
            this.bq = b.bq;
            this.wk = b.wk;
            this.bk = b.bk;
            this.wv = b.wv;
            this.bv = b.bv;
            this.wo = b.wo;
            this.bo = b.bo;
            this.wGate = b.wGate;
            this.bGate = b.bGate;
            this.wUp = b.wUp;
            this.bUp = b.bUp;
            this.wDown = b.wDown;
            this.bDown = b.bDown;
            this.moeRouter = b.moeRouter;
            this.moeRouterBias = b.moeRouterBias;
            this.moeGateExp = b.moeGateExp;
            this.moeUpExp = b.moeUpExp;
            this.moeDownExp = b.moeDownExp;
            this.rope = b.rope;
            this.qkWRows = b.qkWRows;
            this.qkKRows = b.qkKRows;
        }

        int nHead() {
            return plan.nHead();
        }

        int nHeadKv() {
            return plan.nHeadKv();
        }

        int headDimK() {
            return plan.headDimK();
        }

        int headDimV() {
            return plan.headDimV();
        }

        int qDim() {
            return plan.qDim();
        }

        int kDim() {
            return plan.kDim();
        }

        int vDim() {
            return plan.vDim();
        }

        int outDim() {
            return plan.outDim();
        }

        int nFf() {
            return plan.nFf();
        }

        boolean gated() {
            return plan.form() != FinetunePlan.FfnForm.UNGATED;
        }

        static final class Builder {
            FinetunePlan.Layer plan;
            Parameter attnNormW;
            Parameter attnNormB;
            Parameter ffnNormW;
            Parameter ffnNormB;
            Parameter postAttnNormW;
            Parameter postAttnNormB;
            Parameter postFfnNormW;
            Parameter postFfnNormB;
            Parameter qkQW;
            Parameter qkQB;
            Parameter qkKW;
            Parameter qkKB;
            Parameter wq;
            Parameter bq;
            Parameter wk;
            Parameter bk;
            Parameter wv;
            Parameter bv;
            Parameter wo;
            Parameter bo;
            Parameter wGate;
            Parameter bGate;
            Parameter wUp;
            Parameter bUp;
            Parameter wDown;
            Parameter bDown;
            Parameter moeRouter;
            Parameter moeRouterBias;
            Parameter moeGateExp;
            Parameter moeUpExp;
            Parameter moeDownExp;
            RopeFreqTable rope;
            int qkWRows = 1;
            int qkKRows = 1;
        }
    }

    private final FinetunePlan plan;
    private final GGUFModel source;
    private final int nCtx;
    private final int nEmbd;
    private final int nVocab;
    private final Map<String, Parameter> parameters;
    private final List<Parameter> parameterList;
    private final List<BoundLayer> layers;
    private final Parameter tokenEmbd;
    private final Parameter outputWeight;
    private final Parameter outputBias;
    private final Parameter outputNormW;
    private final Parameter outputNormB;
    private final int maxQDim;
    private final int maxKDim;
    private final int maxVDim;
    private final int maxOutDim;
    private final int maxFf;
    private final int maxExperts;
    private final int maxUsed;
    private final int maxHeads;
    private final int maxNormRows;

    private final float[] stream;
    private final float[] normed;
    private final float[] dNormed;
    private final float[] dx;
    private final float[] h;
    private final float[] h2;
    private final float[] x2;
    private final float[] q;
    private final float[] k;
    private final float[] v;
    private final float[] attn;
    private final float[] proj;
    private final float[] postAttnOut;
    private final float[] ffnOut;
    private final float[] postFfnOut;
    private final float[] gate;
    private final float[] up;
    private final float[] act;
    private final float[] dH;
    private final float[] dH2;
    private final float[] dX2;
    private final float[] dProj;
    private final float[] dPostAttn;
    private final float[] dPostFfn;
    private final float[] dQ;
    private final float[] dK;
    private final float[] dV;
    private final float[] dAttn;
    private final float[] dGate;
    private final float[] dUp;
    private final float[] dAct;
    private final float[] attnNormStats;
    private final float[] ffnNormStats;
    private final float[] postAttnNormStats;
    private final float[] postFfnNormStats;
    private final float[] qkQNormStats;
    private final float[] qkKNormStats;
    private final float[] qPreNorm;
    private final float[] kPreNorm;
    private final float[] qkNormGradTmp;
    private final float[] outputNormStats;
    private final float[] routerLogits;
    private final float[] routerProbs;
    private final float[] dRouter;
    private final int[] expertIndex;
    private final float[] expertWeight;
    private final float[] expertOutTmp;
    private final float[] expertInGradTmp;
    private final float[] scores;
    private final float[] probs;
    private final float[] dp;
    private final float[] logits;
    private final float[] logitProbs;
    private final float[] dLogits;
    private final float[][] saved;

    private TrainableModel(GGUFModel model, FinetunePlan plan, int nCtx) {
        this.source = model;
        this.plan = plan;
        this.nCtx = nCtx;
        this.nEmbd = plan.nEmbd();
        this.nVocab = plan.nVocab();
        CpuWeights weights = model.cpuWeights();
        Map<String, Parameter> params = new LinkedHashMap<>();
        this.tokenEmbd = parameter(params, weights, plan.tokenEmbd(), nVocab, nEmbd);
        this.outputWeight = plan.tiedOutput() && plan.outputWeight().equals(plan.tokenEmbd())
                ? tokenEmbd : parameter(params, weights, plan.outputWeight(), nVocab, nEmbd);
        this.outputBias = plan.outputBias() != null
                ? parameter(params, weights, plan.outputBias(), 1, nVocab) : null;
        this.outputNormW = plan.outputNorm().weight() != null
                ? parameter(params, weights, plan.outputNorm().weight(), 1, nEmbd) : null;
        this.outputNormB = plan.outputNorm().bias() != null
                ? parameter(params, weights, plan.outputNorm().bias(), 1, nEmbd) : null;
        this.layers = Collections.unmodifiableList(bindLayers(params, weights));
        int qd = 1;
        int kd = 1;
        int vd = 1;
        int od = 1;
        int ff = 1;
        int heads = 1;
        int normRows = 1;
        int experts = 1;
        int used = 1;
        for (FinetunePlan.Layer l : plan.layers()) {
            qd = Math.max(qd, l.qDim());
            kd = Math.max(kd, l.kDim());
            vd = Math.max(vd, l.vDim());
            od = Math.max(od, l.outDim());
            ff = Math.max(ff, l.nFf());
            heads = Math.max(heads, Math.max(l.nHead(), l.nHeadKv()));
            normRows = Math.max(normRows, Math.max(l.nHead(), l.nHeadKv()));
            if (l.moe() != null) {
                experts = Math.max(experts, l.moe().nExpert());
                used = Math.max(used, l.moe().nUsed());
            }
        }
        this.maxQDim = qd;
        this.maxKDim = kd;
        this.maxVDim = vd;
        this.maxOutDim = od;
        this.maxFf = ff;
        this.maxHeads = heads;
        this.maxNormRows = normRows;
        this.maxExperts = experts;
        this.maxUsed = used;
        this.parameters = Collections.unmodifiableMap(params);
        this.parameterList = Collections.unmodifiableList(new ArrayList<>(params.values()));
        int tokens = nCtx;
        long elements = (long) tokens * nEmbd;
        this.stream = new float[(int) elements];
        this.normed = new float[(int) elements];
        this.dNormed = new float[(int) elements];
        this.dx = new float[(int) elements];
        this.h = new float[(int) elements];
        this.h2 = new float[(int) elements];
        this.x2 = new float[(int) elements];
        this.proj = new float[(int) elements];
        this.postAttnOut = new float[(int) elements];
        this.ffnOut = new float[(int) elements];
        this.postFfnOut = new float[(int) elements];
        this.dH = new float[(int) elements];
        this.dH2 = new float[(int) elements];
        this.dX2 = new float[(int) elements];
        this.dProj = new float[(int) elements];
        this.dPostAttn = new float[(int) elements];
        this.dPostFfn = new float[(int) elements];
        this.q = new float[tokens * qd];
        this.k = new float[tokens * kd];
        this.v = new float[tokens * vd];
        this.attn = new float[tokens * od];
        this.dQ = new float[tokens * qd];
        this.dK = new float[tokens * kd];
        this.dV = new float[tokens * vd];
        this.dAttn = new float[tokens * od];
        this.gate = new float[tokens * ff];
        this.up = new float[tokens * ff];
        this.act = new float[tokens * ff];
        this.dGate = new float[tokens * ff];
        this.dUp = new float[tokens * ff];
        this.dAct = new float[tokens * ff];
        this.attnNormStats = new float[2 * tokens];
        this.ffnNormStats = new float[2 * tokens];
        this.postAttnNormStats = new float[2 * tokens];
        this.postFfnNormStats = new float[2 * tokens];
        this.qkQNormStats = new float[2 * tokens * normRows];
        this.qkKNormStats = new float[2 * tokens * normRows];
        this.qPreNorm = new float[tokens * qd];
        this.kPreNorm = new float[tokens * kd];
        this.qkNormGradTmp = new float[tokens * Math.max(qd, kd)];
        this.outputNormStats = new float[2 * tokens];
        this.routerLogits = new float[tokens * experts];
        this.routerProbs = new float[tokens * experts];
        this.dRouter = new float[tokens * experts];
        this.expertIndex = new int[tokens * used];
        this.expertWeight = new float[tokens * used];
        this.expertOutTmp = new float[nEmbd];
        this.expertInGradTmp = new float[nEmbd];
        this.scores = new float[maxHeads * tokens];
        this.probs = new float[maxHeads * tokens];
        this.dp = new float[tokens];
        this.logits = new float[nVocab];
        this.logitProbs = new float[nVocab];
        this.dLogits = new float[nVocab];
        this.saved = new float[plan.nLayer()][];
        for (int i = 0; i < saved.length; i++) {
            saved[i] = new float[(int) elements];
        }
    }

    /**
     * A trainable view of {@code model} that runs windows of {@code nCtx} tokens.
     *
     * @throws UnsupportedOperationException when the model is outside {@link FinetunePlan}'s family, or a tensor is
     *                                       not stored as F32, F16 or BF16
     * @throws IllegalArgumentException      when the window cannot fit, or a tensor shape contradicts the plan
     */
    public static TrainableModel open(GGUFModel model, int nCtx) {
        Objects.requireNonNull(model, "model");
        if (nCtx < 2) {
            throw new IllegalArgumentException("a training window needs at least 2 tokens, got " + nCtx);
        }
        FinetunePlan plan = FinetunePlan.of(model.spec());
        if (nCtx > plan.nCtxTrain()) {
            throw new IllegalArgumentException("the training window of " + nCtx + " tokens exceeds the "
                    + plan.nCtxTrain() + " tokens " + plan.arch() + " was trained with");
        }
        try {
            return new TrainableModel(model, plan, nCtx);
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("cannot train " + plan.arch() + ": " + e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------------ binding

    private List<BoundLayer> bindLayers(Map<String, Parameter> params, CpuWeights weights) {
        List<BoundLayer> bound = new ArrayList<>(plan.nLayer());
        for (FinetunePlan.Layer l : plan.layers()) {
            BoundLayer.Builder b = new BoundLayer.Builder();
            b.plan = l;
            b.attnNormW = normWeight(params, weights, l.attnNorm(), nEmbd);
            b.attnNormB = normBias(params, weights, l.attnNorm(), nEmbd);
            b.ffnNormW = normWeight(params, weights, l.ffnNorm(), nEmbd);
            b.ffnNormB = normBias(params, weights, l.ffnNorm(), nEmbd);
            b.postAttnNormW = normWeight(params, weights, l.postAttnNorm(), nEmbd);
            b.postAttnNormB = normBias(params, weights, l.postAttnNorm(), nEmbd);
            b.postFfnNormW = normWeight(params, weights, l.postFfnNorm(), nEmbd);
            b.postFfnNormB = normBias(params, weights, l.postFfnNorm(), nEmbd);
            if (l.qkNorm() != null) {
                b.qkQW = normWeight(params, weights, l.qkNorm().query(), l.headDimK());
                b.qkQB = normBias(params, weights, l.qkNorm().query(), l.headDimK());
                b.qkKW = normWeight(params, weights, l.qkNorm().key(), l.headDimK());
                b.qkKB = normBias(params, weights, l.qkNorm().key(), l.headDimK());
                b.qkWRows = weightRows(b.qkQW, l.headDimK(), l.nHead(), l.il(), "attn_q_norm");
                b.qkKRows = weightRows(b.qkKW, l.headDimK(), l.nHeadKv(), l.il(), "attn_k_norm");
            }
            b.wq = matrix(params, weights, l.attnQ(), l.qDim(), nEmbd);
            b.bq = vector(params, weights, l.attnQBias(), l.qDim());
            b.wk = matrix(params, weights, l.attnK(), l.kDim(), nEmbd);
            b.bk = vector(params, weights, l.attnKBias(), l.kDim());
            b.wv = matrix(params, weights, l.attnV(), l.vDim(), nEmbd);
            b.bv = vector(params, weights, l.attnVBias(), l.vDim());
            b.wo = matrix(params, weights, l.attnOut(), nEmbd, l.outDim());
            b.bo = vector(params, weights, l.attnOutBias(), nEmbd);
            if (l.moe() != null) {
                FinetunePlan.Moe m = l.moe();
                b.moeRouter = matrix(params, weights, m.router(), m.nExpert(), nEmbd);
                b.moeRouterBias = vector(params, weights, m.routerBias(), m.nExpert());
                b.moeGateExp = m.gateExps() != null
                        ? matrix(params, weights, m.gateExps(), m.nFfExp() * m.nExpert(), nEmbd) : null;
                b.moeUpExp = matrix(params, weights, m.upExps(), m.nFfExp() * m.nExpert(), nEmbd);
                b.moeDownExp = matrix(params, weights, m.downExps(), nEmbd * m.nExpert(), m.nFfExp());
            } else {
                b.wGate = l.ffnGate() != null ? matrix(params, weights, l.ffnGate(), l.nFf(), nEmbd) : null;
                b.bGate = vector(params, weights, l.ffnGateBias(), l.nFf());
                b.wUp = matrix(params, weights, l.ffnUp(), l.nFf(), nEmbd);
                b.bUp = vector(params, weights, l.ffnUpBias(), l.nFf());
                b.wDown = matrix(params, weights, l.ffnDown(), nEmbd, l.nFf());
                b.bDown = vector(params, weights, l.ffnDownBias(), nEmbd);
            }
            b.rope = rope(l, weights);
            bound.add(new BoundLayer(b));
        }
        return bound;
    }

    private RopeFreqTable rope(FinetunePlan.Layer layer, CpuWeights weights) {
        RopeSpec spec = layer.rope();
        if (spec == null || spec.type() == com.eonmux.neuralfree.gguf.arch.RopeType.NONE) {
            return null;
        }
        float[] factors = spec.freqFactors() != null ? weights.floats(spec.freqFactors()) : null;
        return RopeFreqTable.of(spec, factors);
    }

    private static int weightRows(Parameter weight, int rowLen, int rows, int il, String what) {
        if (weight == null) {
            return 1;
        }
        if (weight.size() % rowLen != 0) {
            throw new IllegalArgumentException("layer " + il + ": " + what + " holds " + weight.size()
                    + " values, not whole rows of " + rowLen);
        }
        int wRows = weight.size() / rowLen;
        if (wRows < 1 || rows % wRows != 0) {
            throw new IllegalArgumentException("layer " + il + ": " + what + " has " + wRows + " weight row(s) for "
                    + rows + " normalized rows");
        }
        return wRows;
    }

    private Parameter normWeight(Map<String, Parameter> params, CpuWeights weights, FinetunePlan.Norm norm,
                                 int rowLen) {
        return norm != null && norm.weight() != null ? parameter(params, weights, norm.weight(), 1, rowLen) : null;
    }

    private Parameter normBias(Map<String, Parameter> params, CpuWeights weights, FinetunePlan.Norm norm,
                               int rowLen) {
        return norm != null && norm.bias() != null ? parameter(params, weights, norm.bias(), 1, rowLen) : null;
    }

    private static Parameter matrix(Map<String, Parameter> params, CpuWeights weights, String name, int rows,
                                    int cols) {
        if (name == null) {
            return null;
        }
        return parameter(params, weights, name, rows, cols);
    }

    private static Parameter vector(Map<String, Parameter> params, CpuWeights weights, String name, int cols) {
        if (name == null) {
            return null;
        }
        return parameter(params, weights, name, 1, cols);
    }

    /** The parameter of {@code name}, decoded once; the same stored tensor always yields the same parameter. */
    private static Parameter parameter(Map<String, Parameter> params, CpuWeights weights, String name, int rows,
                                       int cols) {
        Parameter existing = params.get(name);
        if (existing != null) {
            return existing;
        }
        CpuTensorRef ref = weights.get(name);
        if (ref.type() != GGMLType.F32 && ref.type() != GGMLType.F16 && ref.type() != GGMLType.BF16) {
            throw new UnsupportedOperationException("finetune trains F32, F16 or BF16 weights only; " + name
                    + " is stored as " + ref.type() + " (convert the model to F16, BF16 or F32 first)");
        }
        if (ref.rows() != rows || ref.cols() != cols) {
            throw new IllegalArgumentException(name + " is " + ref.rows() + " x " + ref.cols() + ", the model plans "
                    + rows + " x " + cols);
        }
        long elements = ref.rows() * ref.cols();
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " holds " + elements + " elements, more than one array");
        }
        float[] value = new float[(int) elements];
        ref.decodeRows(0, (int) ref.rows(), value, 0);
        Parameter parameter = new Parameter(name, ref.type(), weights.describe(ref).ne(), rows, cols, value);
        params.put(name, parameter);
        return parameter;
    }

    // ------------------------------------------------------------------ accessors

    /** The plan this model trains. */
    public FinetunePlan plan() {
        return plan;
    }

    /** The window length of one step. */
    public int nCtx() {
        return nCtx;
    }

    public int nEmbd() {
        return nEmbd;
    }

    public int nVocab() {
        return nVocab;
    }

    /** Every trainable parameter, each tied tensor once, in plan order (unmodifiable). */
    public List<Parameter> parameters() {
        return parameterList;
    }

    /** The parameter of a stored tensor name, or null. */
    public Parameter parameter(String name) {
        return parameters.get(name);
    }

    /** The trained values by stored tensor name (unmodifiable; the arrays are the live values). */
    public Map<String, Parameter> trainedParameters() {
        return parameters;
    }

    /** The model the weights were read from. */
    public GGUFModel source() {
        return source;
    }

    // ------------------------------------------------------------------ steps

    /**
     * A forward pass over {@code tokens} scored against {@code labels}: the mean cross-entropy loss and the number
     * of positions whose argmax matches. Runs without recording gradients and leaves the gradient arrays alone.
     *
     * @throws IllegalArgumentException when either array is not the window length
     */
    public StepResult forward(int[] tokens, int[] labels) {
        checkWindow(tokens, labels);
        currentWindow = tokens.length;
        embedTokens = tokens;
        embed(tokens);
        for (int i = 0; i < layers.size(); i++) {
            System.arraycopy(stream, 0, saved[i], 0, currentWindow * nEmbd);
            layerForward(i, saved[i], stream);
        }
        outputNorm();
        return score(currentWindow, labels);
    }

    /**
     * A training step: {@link #forward} followed by the backward pass, which accumulates every parameter's gradient.
     * The caller zeroes the gradients before the step and applies the optimizer after it.
     *
     * @throws IllegalArgumentException when either array is not the window length
     */
    public StepResult step(int[] tokens, int[] labels) {
        StepResult result = forward(tokens, labels);
        backward(tokens.length, labels);
        return result;
    }

    /** Zeroes every parameter's gradient. */
    public void zeroGrad() {
        for (Parameter p : parameterList) {
            p.zeroGrad();
        }
    }

    /** The sum of squared gradients of every parameter, for a gradient check. */
    public double gradNormSquared() {
        double sum = 0.0;
        for (Parameter p : parameterList) {
            for (float g : p.grad) {
                sum += (double) g * g;
            }
        }
        return sum;
    }

    /**
     * The loss of one already-embedded window without touching the gradients: the LM head and cross entropy over the
     * stream the last forward pass left. Used by the gradient check.
     */
    double lossOnly(int[] labels) {
        outputNorm();
        double loss = 0.0;
        int n = labels.length;
        for (int t = 0; t < n; t++) {
            headRow(t);
            loss += TrainMath.crossEntropy(logits, nVocab, labels[t], logitProbs);
        }
        return loss / n;
    }

    @Override
    public void close() {
        // The parameter arrays and the source model are owned elsewhere; nothing device-side is held.
    }

    // ------------------------------------------------------------------ forward

    private void checkWindow(int[] tokens, int[] labels) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(labels, "labels");
        if (tokens.length != labels.length) {
            throw new IllegalArgumentException("the window holds " + tokens.length + " tokens and " + labels.length
                    + " labels");
        }
        if (tokens.length < 1 || tokens.length > nCtx) {
            throw new IllegalArgumentException("the window holds " + tokens.length + " tokens, outside [1, " + nCtx
                    + "]");
        }
        for (int t = 0; t < tokens.length; t++) {
            if (tokens[t] < 0 || tokens[t] >= nVocab) {
                throw new IllegalArgumentException("token " + tokens[t] + " at position " + t + " is outside the "
                        + nVocab + "-token vocabulary");
            }
            if (labels[t] < 0 || labels[t] >= nVocab) {
                throw new IllegalArgumentException("label " + labels[t] + " at position " + t + " is outside the "
                        + nVocab + "-token vocabulary");
            }
        }
    }

    private void embed(int[] tokens) {
        int n = tokens.length;
        float scale = plan.embdScale();
        for (int t = 0; t < n; t++) {
            int src = tokens[t] * nEmbd;
            int dst = t * nEmbd;
            if (scale == 1.0f) {
                System.arraycopy(tokenEmbd.value, src, stream, dst, nEmbd);
            } else {
                for (int c = 0; c < nEmbd; c++) {
                    stream[dst + c] = scale * tokenEmbd.value[src + c];
                }
            }
        }
    }

    /**
     * One block: the intermediates into the shared buffers and, when {@code out} is not null, the block output into
     * it. {@code out} may equal {@code in}.
     */
    private void layerForward(int il, float[] in, float[] out) {
        BoundLayer l = layers.get(il);
        int n = currentWindow;
        normForward(l.plan.attnNorm(), l.attnNormW, l.attnNormB, in, n, nEmbd, 1, h, attnNormStats);
        boolean qk = l.plan.qkNorm() != null;
        boolean qkAfter = qk && l.plan.qkNorm().afterRope();
        TrainMath.linear(l.wq.value, l.qDim(), nEmbd, bias(l.bq), h, n, q);
        if (qk && !qkAfter) {
            qkNormForward(l.plan.qkNorm().query(), l.qkQW, l.qkQB, l.qkWRows, q, qPreNorm, n, qkQNormStats);
        }
        if (l.rope != null) {
            TrainMath.ropeForward(l.rope, q, n, l.nHead(), l.headDimK(), l.qDim());
        }
        if (qkAfter) {
            qkNormForward(l.plan.qkNorm().query(), l.qkQW, l.qkQB, l.qkWRows, q, qPreNorm, n, qkQNormStats);
        }
        TrainMath.linear(l.wk.value, l.kDim(), nEmbd, bias(l.bk), h, n, k);
        if (qk && !qkAfter) {
            qkNormForward(l.plan.qkNorm().key(), l.qkKW, l.qkKB, l.qkKRows, k, kPreNorm, n, qkKNormStats);
        }
        if (l.rope != null) {
            TrainMath.ropeForward(l.rope, k, n, l.nHeadKv(), l.headDimK(), l.kDim());
        }
        if (qkAfter) {
            qkNormForward(l.plan.qkNorm().key(), l.qkKW, l.qkKB, l.qkKRows, k, kPreNorm, n, qkKNormStats);
        }
        TrainMath.linear(l.wv.value, l.vDim(), nEmbd, bias(l.bv), h, n, v);
        TrainMath.attentionForward(n, l.nHead(), l.nHeadKv(), l.headDimK(), l.headDimV(), l.plan.scoreScale(),
                l.plan.swaWindow(), l.plan.attnSoftcap(), q, l.qDim(), k, l.kDim(), v, l.vDim(), attn, l.outDim(),
                scores, probs);
        TrainMath.linear(l.wo.value, nEmbd, l.outDim(), bias(l.bo), attn, n, proj);
        float scale = l.plan.residualScale();
        float[] attnBranch = proj;
        if (l.plan.postAttnNorm() != null) {
            normForward(l.plan.postAttnNorm(), l.postAttnNormW, l.postAttnNormB, proj, n, nEmbd, 1, postAttnOut,
                    postAttnNormStats);
            attnBranch = postAttnOut;
        }
        for (int i = 0; i < n * nEmbd; i++) {
            x2[i] = in[i] + scale * attnBranch[i];
        }
        float[] ffnIn = h;
        if (l.plan.ffnNorm() != null) {
            normForward(l.plan.ffnNorm(), l.ffnNormW, l.ffnNormB, x2, n, nEmbd, 1, h2, ffnNormStats);
            ffnIn = h2;
        }
        if (l.plan.moe() != null) {
            moeForward(l, ffnIn, n, ffnOut);
        } else if (l.gated()) {
            TrainMath.linear(l.wGate.value, l.nFf(), nEmbd, bias(l.bGate), ffnIn, n, gate);
            TrainMath.linear(l.wUp.value, l.nFf(), nEmbd, bias(l.bUp), ffnIn, n, up);
            TrainMath.gluForward(l.plan.glu(), gate, up, act, n * l.nFf());
            TrainMath.linear(l.wDown.value, nEmbd, l.nFf(), bias(l.bDown), act, n, ffnOut);
        } else {
            TrainMath.linear(l.wUp.value, l.nFf(), nEmbd, bias(l.bUp), ffnIn, n, up);
            for (int i = 0; i < n * l.nFf(); i++) {
                act[i] = TrainMath.activation(l.plan.act(), up[i]);
            }
            TrainMath.linear(l.wDown.value, nEmbd, l.nFf(), bias(l.bDown), act, n, ffnOut);
        }
        float[] ffnBranch = ffnOut;
        if (l.plan.postFfnNorm() != null) {
            normForward(l.plan.postFfnNorm(), l.postFfnNormW, l.postFfnNormB, ffnOut, n, nEmbd, 1, postFfnOut,
                    postFfnNormStats);
            ffnBranch = postFfnOut;
        }
        if (out == null) {
            return;
        }
        for (int i = 0; i < n * nEmbd; i++) {
            out[i] = x2[i] + scale * ffnBranch[i];
        }
    }

    /**
     * The routed-expert FFN of one layer: the softmax router, its top-{@code nUsed} selection and the selected
     * experts' gated or ungated FFNs, weighted and summed per token. The routing decision and weights are kept for
     * the backward pass; the expert activations are recomputed there.
     */
    private void moeForward(BoundLayer l, float[] x, int n, float[] out) {
        FinetunePlan.Moe m = l.plan.moe();
        int experts = m.nExpert();
        int used = m.nUsed();
        int ff = m.nFfExp();
        float scale = m.weightsScale() == 0f || m.weightsScale() == 1f ? 1f : m.weightsScale();
        TrainMath.linear(l.moeRouter.value, experts, nEmbd, bias(l.moeRouterBias), x, n, routerLogits);
        for (int t = 0; t < n; t++) {
            TrainMath.softmaxRow(routerLogits, t * experts, experts, routerProbs, t * experts);
            TrainMath.selectExperts(routerProbs, t * experts, experts, used, m.normWeights(), scale, expertIndex,
                    t * used, expertWeight, t * used);
            int outBase = t * nEmbd;
            Arrays.fill(out, outBase, outBase + nEmbd, 0f);
            for (int s = 0; s < used; s++) {
                int e = expertIndex[t * used + s];
                float w = expertWeight[t * used + s];
                expertForward(l, m, e, x, t, ff, expertOutTmp);
                for (int c = 0; c < nEmbd; c++) {
                    out[outBase + c] += w * expertOutTmp[c];
                }
            }
        }
    }

    /** One expert's gated or ungated FFN on token {@code t}, into {@code dst} (length {@code nEmbd}). */
    private void expertForward(BoundLayer l, FinetunePlan.Moe m, int expert, float[] x, int t, int ff, float[] dst) {
        int xBase = t * nEmbd;
        int gateBase = expert * ff;
        int downBase = expert * nEmbd;
        if (m.gateExps() != null) {
            for (int j = 0; j < ff; j++) {
                gate[j] = dotRow(l.moeGateExp.value, gateBase + j, nEmbd, x, xBase);
            }
            for (int j = 0; j < ff; j++) {
                up[j] = dotRow(l.moeUpExp.value, gateBase + j, nEmbd, x, xBase);
                act[j] = TrainMath.gluValue(m.glu(), gate[j], up[j]);
            }
        } else {
            for (int j = 0; j < ff; j++) {
                up[j] = dotRow(l.moeUpExp.value, gateBase + j, nEmbd, x, xBase);
                act[j] = TrainMath.activation(m.act(), up[j]);
            }
        }
        for (int c = 0; c < nEmbd; c++) {
            dst[c] = dotRow(l.moeDownExp.value, downBase + c, ff, act, 0);
        }
    }

    private static float dotRow(float[] w, int row, int cols, float[] x, int xBase) {
        int base = row * cols;
        float sum = 0f;
        for (int i = 0; i < cols; i++) {
            sum += w[base + i] * x[xBase + i];
        }
        return sum;
    }

    private void outputNorm() {
        FinetunePlan.Norm norm = plan.outputNorm();
        normForward(norm, outputNormW, outputNormB, stream, currentWindow, nEmbd, 1, normed, outputNormStats);
    }

    /** The logits of token {@code t} from the final normed row into {@link #logits}. */
    private void headRow(int t) {
        int base = t * nEmbd;
        int rows = nVocab;
        int cols = nEmbd;
        float[] w = outputWeight.value;
        float cap = plan.finalSoftcap();
        for (int v = 0; v < rows; v++) {
            int wBase = v * cols;
            float sum = 0f;
            for (int c = 0; c < cols; c++) {
                sum += w[wBase + c] * normed[base + c];
            }
            sum += outputBias != null ? outputBias.value[v] : 0f;
            logits[v] = cap > 0f ? cap * (float) Math.tanh(sum / cap) : sum;
        }
    }

    private StepResult score(int n, int[] labels) {
        double loss = 0.0;
        int correct = 0;
        for (int t = 0; t < n; t++) {
            headRow(t);
            loss += TrainMath.crossEntropy(logits, nVocab, labels[t], logitProbs);
            if (TrainMath.argmax(logits, nVocab) == labels[t]) {
                correct++;
            }
        }
        return new StepResult(loss / n, correct, n);
    }

    private void normForward(FinetunePlan.Norm norm, Parameter weight, Parameter bias, float[] x, int nRows,
                             int rowLen, int wRows, float[] dst, float[] stats) {
        TrainMath.normForward(norm.kind(), norm.eps(), x, nRows, rowLen, wRows,
                weight != null ? weight.value : null, bias != null ? bias.value : null, dst, stats);
    }

    /**
     * Normalizes {@code x} in place. The pre-norm values are copied into {@code preNorm}, which the backward pass
     * needs: the norm's gradient formula is written in terms of its input, not its output.
     */
    private void qkNormForward(FinetunePlan.Norm norm, Parameter weight, Parameter bias, int wRows, float[] x,
                               float[] preNorm, int n, float[] stats) {
        if (norm == null) {
            return;
        }
        int rows = n * norm.rows();
        System.arraycopy(x, 0, preNorm, 0, rows * norm.rowLen());
        TrainMath.normForward(norm.kind(), norm.eps(), x, rows, norm.rowLen(), wRows,
                weight != null ? weight.value : null, bias != null ? bias.value : null, x, stats);
    }

    // ------------------------------------------------------------------ backward

    private void backward(int n, int[] labels) {
        headBackward(n, labels);
        FinetunePlan.Norm norm = plan.outputNorm();
        Arrays.fill(dx, 0, n * nEmbd, 0f);
        TrainMath.normBackward(norm.kind(), stream, dNormed, n, nEmbd, 1,
                outputNormW != null ? outputNormW.value : null, outputNormStats, dx,
                outputNormW != null ? outputNormW.grad : null, outputNormB != null ? outputNormB.grad : null);
        for (int i = layers.size() - 1; i >= 0; i--) {
            layerBackward(i, n);
        }
        embeddingBackward(n);
    }

    /** The LM head and cross entropy: the head's weight/bias gradients and {@link #dNormed}. */
    private void headBackward(int n, int[] labels) {
        Arrays.fill(dNormed, 0f);
        float[] w = outputWeight.value;
        float[] dw = outputWeight.grad;
        float cap = plan.finalSoftcap();
        for (int t = 0; t < n; t++) {
            headRow(t);
            TrainMath.crossEntropy(logits, nVocab, labels[t], logitProbs);
            float inv = 1f / n;
            for (int v = 0; v < nVocab; v++) {
                float g = (logitProbs[v] - (v == labels[t] ? 1f : 0f)) * inv;
                if (cap > 0f) {
                    float tanh = logits[v] / cap;
                    g *= 1f - tanh * tanh;
                }
                dLogits[v] = g;
            }
            int normBase = t * nEmbd;
            for (int v = 0; v < nVocab; v++) {
                float g = dLogits[v];
                if (g == 0f) {
                    continue;
                }
                int wBase = v * nEmbd;
                for (int c = 0; c < nEmbd; c++) {
                    dNormed[normBase + c] += w[wBase + c] * g;
                    dw[wBase + c] += g * normed[normBase + c];
                }
                if (outputBias != null) {
                    outputBias.grad[v] += g;
                }
            }
        }
    }

    private void embeddingBackward(int n) {
        float[] grad = tokenEmbd.grad;
        float scale = plan.embdScale();
        for (int t = 0; t < n; t++) {
            int token = embedTokens[t];
            int base = token * nEmbd;
            int src = t * nEmbd;
            for (int c = 0; c < nEmbd; c++) {
                grad[base + c] += scale * dx[src + c];
            }
        }
    }

    /** One block's backward, turning {@link #dx} from {@code dL/dx_{i+1}} into {@code dL/dx_i}. */
    private void layerBackward(int il, int n) {
        BoundLayer l = layers.get(il);
        float[] xIn = saved[il];
        layerForward(il, xIn, null);
        // The FFN branch's output gradient, through the sandwich post-FFN norm when the block has one.
        float[] dFfnBranch = dx;
        if (l.plan.postFfnNorm() != null) {
            zero(dPostFfn, n * nEmbd);
            TrainMath.normBackward(l.plan.postFfnNorm().kind(), ffnOut, dx, n, nEmbd, 1,
                    l.postFfnNormW != null ? l.postFfnNormW.value : null, postFfnNormStats, dPostFfn,
                    l.postFfnNormW != null ? l.postFfnNormW.grad : null,
                    l.postFfnNormB != null ? l.postFfnNormB.grad : null);
            dFfnBranch = dPostFfn;
        }
        // FFN (dense or routed) into dH2, the gradient of the FFN input.
        zero(dH2, n * nEmbd);
        if (l.plan.moe() != null) {
            moeBackward(l, dFfnBranch, n);
        } else {
            denseFfnBackward(l, dFfnBranch, n);
        }
        // dX2 = dL/dx2: the block output's gradient plus the FFN's term through its input norm.
        System.arraycopy(dx, 0, dX2, 0, n * nEmbd);
        boolean sharedNorm = l.plan.ffnNorm() == null;
        if (!sharedNorm) {
            TrainMath.normBackward(l.plan.ffnNorm().kind(), x2, dH2, n, nEmbd, 1,
                    l.ffnNormW != null ? l.ffnNormW.value : null, ffnNormStats, dX2,
                    l.ffnNormW != null ? l.ffnNormW.grad : null, l.ffnNormB != null ? l.ffnNormB.grad : null);
        }
        // The gradient of the raw attention projection, through the sandwich post-attention norm when present.
        float[] dAttnProj = dX2;
        if (l.plan.postAttnNorm() != null) {
            zero(dPostAttn, n * nEmbd);
            TrainMath.normBackward(l.plan.postAttnNorm().kind(), proj, dX2, n, nEmbd, 1,
                    l.postAttnNormW != null ? l.postAttnNormW.value : null, postAttnNormStats, dPostAttn,
                    l.postAttnNormW != null ? l.postAttnNormW.grad : null,
                    l.postAttnNormB != null ? l.postAttnNormB.grad : null);
            dAttnProj = dPostAttn;
        }
        // attention output projection: dAttn = W_o^T dAttnProj
        zero(dAttn, n * l.outDim());
        TrainMath.linearInputGrad(l.wo.value, nEmbd, l.outDim(), dAttnProj, n, dAttn);
        TrainMath.linearWeightGrad(nEmbd, l.outDim(), attn, dAttnProj, n, l.wo.grad);
        if (l.bo != null) {
            TrainMath.linearBiasGrad(nEmbd, dAttnProj, n, l.bo.grad);
        }
        // attention
        zero(dQ, n * l.qDim());
        zero(dK, n * l.kDim());
        zero(dV, n * l.vDim());
        TrainMath.attentionBackward(n, l.nHead(), l.nHeadKv(), l.headDimK(), l.headDimV(), l.plan.scoreScale(),
                l.plan.swaWindow(), l.plan.attnSoftcap(), q, l.qDim(), k, l.kDim(), v, l.vDim(), dAttn, l.outDim(),
                dQ, dK, dV, scores, probs, dp);
        // rope and the per-head Q/K norms, in the reverse of the forward order
        if (l.plan.qkNorm() != null && l.plan.qkNorm().afterRope()) {
            qkNormBackward(l.plan.qkNorm().query(), l.qkQW, l.qkQB, l.qkWRows, qPreNorm, dQ, n, qkQNormStats);
            qkNormBackward(l.plan.qkNorm().key(), l.qkKW, l.qkKB, l.qkKRows, kPreNorm, dK, n, qkKNormStats);
        }
        if (l.rope != null) {
            TrainMath.ropeBackward(l.rope, dQ, n, l.nHead(), l.headDimK(), l.qDim());
            TrainMath.ropeBackward(l.rope, dK, n, l.nHeadKv(), l.headDimK(), l.kDim());
        }
        if (l.plan.qkNorm() != null && !l.plan.qkNorm().afterRope()) {
            qkNormBackward(l.plan.qkNorm().query(), l.qkQW, l.qkQB, l.qkWRows, qPreNorm, dQ, n, qkQNormStats);
            qkNormBackward(l.plan.qkNorm().key(), l.qkKW, l.qkKB, l.qkKRows, kPreNorm, dK, n, qkKNormStats);
        }
        // Q/K/V projections into dH
        zero(dH, n * nEmbd);
        TrainMath.linearInputGrad(l.wq.value, l.qDim(), nEmbd, dQ, n, dH);
        TrainMath.linearWeightGrad(l.qDim(), nEmbd, h, dQ, n, l.wq.grad);
        if (l.bq != null) {
            TrainMath.linearBiasGrad(l.qDim(), dQ, n, l.bq.grad);
        }
        TrainMath.linearInputGrad(l.wk.value, l.kDim(), nEmbd, dK, n, dH);
        TrainMath.linearWeightGrad(l.kDim(), nEmbd, h, dK, n, l.wk.grad);
        if (l.bk != null) {
            TrainMath.linearBiasGrad(l.kDim(), dK, n, l.bk.grad);
        }
        TrainMath.linearInputGrad(l.wv.value, l.vDim(), nEmbd, dV, n, dH);
        TrainMath.linearWeightGrad(l.vDim(), nEmbd, h, dV, n, l.wv.grad);
        if (l.bv != null) {
            TrainMath.linearBiasGrad(l.vDim(), dV, n, l.bv.grad);
        }
        if (sharedNorm) {
            for (int i = 0; i < n * nEmbd; i++) {
                dH[i] += dH2[i];
            }
        }
        // The input gradient is the block output's gradient carried through both residual branches: the FFN norm's
        // term is already in dX2, and the attention norm's term is accumulated here.
        TrainMath.normBackward(l.plan.attnNorm().kind(), xIn, dH, n, nEmbd, 1,
                l.attnNormW != null ? l.attnNormW.value : null, attnNormStats, dX2,
                l.attnNormW != null ? l.attnNormW.grad : null, l.attnNormB != null ? l.attnNormB.grad : null);
        System.arraycopy(dX2, 0, dx, 0, n * nEmbd);
    }

    /** The dense FFN's backward from the gradient of its output into {@link #dH2}. */
    private void denseFfnBackward(BoundLayer l, float[] dOut, int n) {
        int ff = l.nFf();
        zero(dAct, n * ff);
        TrainMath.linearInputGrad(l.wDown.value, nEmbd, ff, dOut, n, dAct);
        TrainMath.linearWeightGrad(nEmbd, ff, act, dOut, n, l.wDown.grad);
        if (l.bDown != null) {
            TrainMath.linearBiasGrad(nEmbd, dOut, n, l.bDown.grad);
        }
        zero(dGate, n * ff);
        zero(dUp, n * ff);
        if (l.gated()) {
            TrainMath.gluBackward(l.plan.glu(), gate, up, dAct, dGate, dUp, n * ff);
        } else {
            for (int i = 0; i < n * ff; i++) {
                dUp[i] = dAct[i] * TrainMath.activationDerivative(l.plan.act(), up[i]);
            }
        }
        if (l.gated()) {
            TrainMath.linearInputGrad(l.wGate.value, ff, nEmbd, dGate, n, dH2);
            TrainMath.linearWeightGrad(ff, nEmbd, ffnInput(l), dGate, n, l.wGate.grad);
            if (l.bGate != null) {
                TrainMath.linearBiasGrad(ff, dGate, n, l.bGate.grad);
            }
        }
        TrainMath.linearInputGrad(l.wUp.value, ff, nEmbd, dUp, n, dH2);
        TrainMath.linearWeightGrad(ff, nEmbd, ffnInput(l), dUp, n, l.wUp.grad);
        if (l.bUp != null) {
            TrainMath.linearBiasGrad(ff, dUp, n, l.bUp.grad);
        }
    }

    /**
     * The routed FFN's backward into {@link #dH2}: the selected experts' gradients and the router's, with the top-k
     * selection treated as fixed (llama.cpp's sparse-expert backward). The expert activations are recomputed from the
     * saved routing decision, so the forward pass keeps only the router probabilities and the selection.
     */
    private void moeBackward(BoundLayer l, float[] dOut, int n) {
        FinetunePlan.Moe m = l.plan.moe();
        int experts = m.nExpert();
        int used = m.nUsed();
        int ff = m.nFfExp();
        float scale = m.weightsScale() == 0f || m.weightsScale() == 1f ? 1f : m.weightsScale();
        float[] x = ffnInput(l);
        Arrays.fill(dRouter, 0, n * experts, 0f);
        float[] expertDot = new float[used];
        for (int t = 0; t < n; t++) {
            int rb = t * experts;
            int ib = t * used;
            int xBase = t * nEmbd;
            // Each selected expert's output and the layer output's dot with dOut.
            float dotOut = 0f;
            for (int s = 0; s < used; s++) {
                int e = expertIndex[ib + s];
                expertForward(l, m, e, x, t, ff, expertOutTmp);
                float dot = 0f;
                for (int c = 0; c < nEmbd; c++) {
                    dot += expertOutTmp[c] * dOut[xBase + c];
                }
                expertDot[s] = dot;
                dotOut += expertWeight[ib + s] * dot;
            }
            float sum = 1f;
            if (m.normWeights()) {
                sum = 0f;
                for (int s = 0; s < used; s++) {
                    sum += routerProbs[rb + expertIndex[ib + s]];
                }
                sum = Math.max(sum, MoeSpec.NORM_SUM_MIN);
            }
            // Gradient through the router's softmax: the selected experts' weighted-share derivatives first. The
            // renormalization only contributes when the selected probabilities are divided by their sum.
            for (int s = 0; s < used; s++) {
                dRouter[rb + expertIndex[ib + s]] = m.normWeights()
                        ? (scale * expertDot[s] - dotOut) / sum
                        : scale * expertDot[s];
            }
            float dotProbs = 0f;
            for (int e = 0; e < experts; e++) {
                dotProbs += routerProbs[rb + e] * dRouter[rb + e];
            }
            for (int e = 0; e < experts; e++) {
                dRouter[rb + e] = routerProbs[rb + e] * (dRouter[rb + e] - dotProbs);
            }
            // The selected experts' weight and input gradients. The routing weight is already carried by the
            // expert output gradient, so the gate/up and input gradients below must not apply it again.
            for (int s = 0; s < used; s++) {
                int e = expertIndex[ib + s];
                float w = expertWeight[ib + s];
                expertForward(l, m, e, x, t, ff, expertOutTmp);
                expertDownGrads(l.moeDownExp, e, dOut, xBase, ff, w, dAct);
                zero(dGate, ff);
                zero(dUp, ff);
                if (m.gateExps() != null) {
                    TrainMath.gluBackward(m.glu(), gate, up, dAct, dGate, dUp, ff);
                    expertUpGrads(l.moeGateExp, e, dGate, x, xBase, ff);
                } else {
                    for (int j = 0; j < ff; j++) {
                        dUp[j] = dAct[j] * TrainMath.activationDerivative(m.act(), up[j]);
                    }
                }
                expertUpGrads(l.moeUpExp, e, dUp, x, xBase, ff);
            }
            // The router itself reads the block input.
            int rbIn = xBase;
            for (int e = 0; e < experts; e++) {
                float de = dRouter[rb + e];
                if (de == 0f) {
                    continue;
                }
                int rowBase = e * nEmbd;
                for (int c = 0; c < nEmbd; c++) {
                    dH2[rbIn + c] += de * l.moeRouter.value[rowBase + c];
                }
            }
        }
        TrainMath.linearWeightGrad(experts, nEmbd, x, dRouter, n, l.moeRouter.grad);
        if (l.moeRouterBias != null) {
            TrainMath.linearBiasGrad(experts, dRouter, n, l.moeRouterBias.grad);
        }
    }

    /**
     * The down-projection gradients of expert {@code e}: accumulates its weight gradient and writes the gradient of
     * its activated input into {@code dAct[0 .. ff)}.
     */
    private void expertDownGrads(Parameter down, int e, float[] dOut, int dOutBase, int ff, float weight,
                                 float[] dAct) {
        int rowBase = e * nEmbd;
        Arrays.fill(dAct, 0, ff, 0f);
        for (int c = 0; c < nEmbd; c++) {
            float g = weight * dOut[dOutBase + c];
            int wb = (rowBase + c) * ff;
            for (int j = 0; j < ff; j++) {
                down.grad[wb + j] += g * act[j];
                dAct[j] += g * down.value[wb + j];
            }
        }
    }

    /**
     * The gate or up projection gradients of expert {@code e}: accumulates its weight gradient and adds its share of
     * the expert input gradient into {@link #dH2}. The routing weight is already folded into {@code d}.
     */
    private void expertUpGrads(Parameter w, int e, float[] d, float[] x, int xBase, int ff) {
        int rowBase = e * ff;
        for (int j = 0; j < ff; j++) {
            float g = d[j];
            if (g == 0f) {
                continue;
            }
            int wb = (rowBase + j) * nEmbd;
            for (int c = 0; c < nEmbd; c++) {
                w.grad[wb + c] += g * x[xBase + c];
                dH2[xBase + c] += g * w.value[wb + c];
            }
        }
    }

    /** The FFN input the forward pass used: the shared attention norm output or the FFN norm output. */
    private float[] ffnInput(BoundLayer l) {
        return l.plan.ffnNorm() == null ? h : h2;
    }

    /**
     * The per-head norm backward. The norm's input gradient replaces the incoming gradient rather than accumulating
     * onto it, so it is computed into a scratch row buffer and copied back.
     */
    private void qkNormBackward(FinetunePlan.Norm norm, Parameter weight, Parameter bias, int wRows, float[] x,
                                float[] d, int n, float[] stats) {
        if (norm == null) {
            return;
        }
        int len = n * norm.rows() * norm.rowLen();
        zero(qkNormGradTmp, len);
        TrainMath.normBackward(norm.kind(), x, d, n * norm.rows(), norm.rowLen(), wRows,
                weight != null ? weight.value : null, stats, qkNormGradTmp,
                weight != null ? weight.grad : null, bias != null ? bias.grad : null);
        System.arraycopy(qkNormGradTmp, 0, d, 0, len);
    }

    /** Zeroes {@code len} leading elements of {@code a}. */
    private static void zero(float[] a, int len) {
        for (int i = 0; i < len; i++) {
            a[i] = 0f;
        }
    }

    private static float[] bias(Parameter p) {
        return p != null ? p.value : null;
    }

    // ------------------------------------------------------------------ window bookkeeping

    /** The number of tokens of the window being processed. */
    private int currentWindow;
    private int[] embedTokens;

    /** The logits of the last forward pass (a copy), for tests and diagnostics. */
    public float[] logitsOfLastStep() {
        return logits.clone();
    }

    /** Largest element difference between each layer's pass and its recompute after the last forward pass. */
    double recomputeDifference() {
        float[] tmp = new float[currentWindow * nEmbd];
        double worst = 0.0;
        for (int i = 0; i < layers.size(); i++) {
            layerForward(i, saved[i], tmp);
            float[] expected = i + 1 < layers.size() ? saved[i + 1] : stream;
            for (int k = 0; k < currentWindow * nEmbd; k++) {
                worst = Math.max(worst, Math.abs(tmp[k] - expected[k]));
            }
        }
        return worst;
    }

    /**
     * The whole logits table of {@code tokens}: one row of {@code nVocab} values per position, the row of a position
     * predicting its label. Runs a forward pass and leaves the model in that state.
     *
     * @throws IllegalArgumentException when the window cannot run
     */
    public float[][] logits(int[] tokens) {
        checkWindow(tokens, tokens);
        currentWindow = tokens.length;
        embedTokens = tokens;
        embed(tokens);
        for (int i = 0; i < layers.size(); i++) {
            System.arraycopy(stream, 0, saved[i], 0, currentWindow * nEmbd);
            layerForward(i, saved[i], stream);
        }
        outputNorm();
        float[][] rows = new float[currentWindow][];
        for (int t = 0; t < currentWindow; t++) {
            headRow(t);
            rows[t] = logits.clone();
        }
        return rows;
    }
}
